// memory.rs

use crate::{cpu, error, optimisations};
use crate::cpu::Exception;

use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::ffi::CString;
use std::sync::Arc;
use std::io::Write;
use std::fs::File;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::Sender;
use crate::optimisations::{in_rom_memory, MemoryAreaPointer};

pub const MEMORY_RAM_SIZE: usize = 0x04000000; // 64 MiB
pub const MEMORY_ROM_SIZE: usize = 0x00080000; // 512 KiB

pub const MEMORY_RAM_START: usize = 0x00000000;
pub const MEMORY_ROM_START: usize = 0xF0000000;

pub type MemoryRam = [u8; MEMORY_RAM_SIZE];
pub type MemoryRom = [u8; MEMORY_ROM_SIZE];

#[derive(Debug)]
pub struct MemoryPage {
    physical_address: u64,
    present: bool,
    rw: bool,
}

struct MemoryInner {
    ram: Arc<MemoryAreaPointer>,
    rom: Arc<MemoryAreaPointer>,
    mmu_enabled: Arc<AtomicBool>,
    tlb: Box<HashMap<u64, MemoryPage>>,
    paging_directory_address: Box<usize>,
    exception_sender: Arc<Sender<Exception>>,
}

impl MemoryInner {
    pub fn new(rom: &[u8], exception_sender: Sender<Exception>) -> Self {
        let this = Self {
            // HACK: allocate directly on the heap to avoid a stack overflow
            //       at runtime while trying to move around a 64MB array
            ram: Arc::new(MemoryAreaPointer::allocate(MEMORY_RAM_SIZE)),
            rom: Arc::new(MemoryAreaPointer::allocate(MEMORY_ROM_SIZE)),
            mmu_enabled: Arc::new(AtomicBool::new(false)),
            tlb: Box::from(HashMap::with_capacity(1024)),
            paging_directory_address: Box::from(0x00000000),
            exception_sender: Arc::new(exception_sender),
        };
        this.rom.write_all(rom).expect("failed to copy ROM to memory");
        this
    }
}

#[derive(Clone)]
pub struct Memory {
    inner: Arc<UnsafeCell<MemoryInner>>,
    ram: Arc<MemoryAreaPointer>,
    rom: Arc<MemoryAreaPointer>,
    mmu_enabled: Arc<AtomicBool>,
}

// SAFETY: once MemoryInner is initialzed, there is no way to modify the Box
//         pointers it contains and it does not matter if contents of the byte
//         arrays are corrupted
unsafe impl Send for Memory {}
unsafe impl Sync for Memory {}

impl Memory {
    pub fn new(rom: &[u8], exception_sender: Sender<Exception>) -> Self {
        let mut inner = Arc::new(UnsafeCell::new(MemoryInner::new(rom, exception_sender)));
        Self {
            inner: inner.clone(),
            mmu_enabled: unsafe { (*inner.get()).mmu_enabled.clone() },
            ram: unsafe { (*inner.get()).ram.clone() },
            rom: unsafe { (*inner.get()).rom.clone() },
        }
    }

    fn inner(&self) -> &mut MemoryInner {
        unsafe { &mut *self.inner.get() }
    }

    pub fn ram(&self) -> Arc<MemoryAreaPointer> { self.ram.clone() }
    pub fn rom(&self) -> Arc<MemoryAreaPointer> { self.rom.clone() }
    pub fn read_mmu_enabled(&self) -> bool { self.mmu_enabled.load(Ordering::Relaxed) }
    pub fn write_mmu_enabled(&self, value: bool) { self.mmu_enabled.store(value, Ordering::Relaxed); }
    pub fn tlb(&self) -> &mut HashMap<u64, MemoryPage> { &mut self.inner().tlb }
    pub fn paging_directory_address(&self) -> &mut usize { &mut self.inner().paging_directory_address }
    pub fn exception_sender(&self) -> Arc<Sender<Exception>> { unsafe { optimisations::EXCEPTION_TOGGLE.store(true, Ordering::SeqCst) }; self.inner().exception_sender.clone() }

    pub fn dump(&self) {
        let mut file = File::create("memory.dump").expect("failed to open memory dump file");
        file.write_all(self.ram().export().unwrap().as_slice()).expect("failed to write memory dump file");
    }

    // each table contains 1024 entries
    // the paging directory contains pointers to paging tables with the following format:
    // bit 0: present
    // remaining bits are ignored, should be zero
    // bits 12-31: physical address of paging table

    // the paging table contains pointers to physical memory pages with the following format:
    // bit 0: present
    // bit 1: r/w
    // remaining bits are ignored, should be zero
    // bits 12-31: physical address

    pub fn flush_tlb(&self, paging_directory_address: Option<u64>) {
        if let Some(address) = paging_directory_address {
            *self.paging_directory_address() = address as usize;
        };

        self.tlb().clear();
    }

    pub fn flush_page(&self, virtual_address: u64) {
        let virtual_page = virtual_address & 0xFFFFF000;
        self.tlb().remove(&virtual_page);
    }

    pub fn insert_tlb_entry_from_tables(&mut self, page_directory_index: u64, page_table_index: u64) -> bool {
        let old_state = self.read_mmu_enabled();
        self.write_mmu_enabled(false);
        let directory_address = *self.paging_directory_address() as u64;
        let directory = self.read_opt_64((directory_address + (page_directory_index * 4)) as usize);
        match directory {
            Some(directory) => {
                let dir_present = directory & 0b1 != 0;
                let dir_address = directory & 0xFFFFF000;
                if dir_present {
                    let table = self.read_opt_64((dir_address + (page_table_index * 4)) as usize);
                    match table {
                        Some(table) => {
                            let table_present = table & 0b01 != 0;
                            let table_rw = table & 0b10 != 0;
                            let table_address = table & 0xFFFFF000;

                            if table_present {
                                let tlb_entry = MemoryPage {
                                    physical_address: table_address,
                                    present: table_present,
                                    rw: table_rw,
                                };
                                self.tlb().entry((page_directory_index << 22) | (page_table_index << 12)).or_insert(tlb_entry);
                            }
                        },
                        None => {}
                    }
                }
                self.write_mmu_enabled(old_state);
                dir_present
            },
            None => {
                self.write_mmu_enabled(old_state);
                false
            }
        }
    }

    pub fn virtual_to_physical(&mut self, virtual_address: u64) -> Option<(u64, bool)> {
        let virtual_page = virtual_address & 0xFFFFF000;
        let offset = virtual_address & 0x00000FFF;
        let physical_page = self.tlb().get(&virtual_page);
        let physical_address = match physical_page {
            Some(page) => {
                if page.present {
                    Some((page.physical_address | offset, page.rw))
                } else {
                    None
                }
            },
            None => {
                let page_directory_index = virtual_address >> 22;
                let page_table_index = (virtual_address >> 12) & 0x03FF;
                let dir_present = self.insert_tlb_entry_from_tables(page_directory_index, page_table_index);
                if !dir_present {
                    return None;
                }
                // try again after inserting the TLB entry
                let physical_page = self.tlb().get(&virtual_page);
                let physical_address = match physical_page {
                    Some(page) => {
                        if page.present {
                            Some((page.physical_address | offset, page.rw))
                        } else {
                            None
                        }
                    },
                    None => None,
                };
                physical_address
            },
        };
        physical_address
    }

    pub fn read_opt_8(&mut self, mut address: usize) -> Option<u8> {
        if self.read_mmu_enabled() {
            let address_maybe = self.virtual_to_physical(address as u64);
            match address_maybe {
                Some(addr) => address = addr.0 as usize,
                None => return None,
            }
        }

        let address = address as usize;

        if in_rom_memory(address) {
            let read = self.rom().read_8(address - MEMORY_ROM_START);
            read
        } else {
            self.ram().read_8(address - MEMORY_RAM_START)
        }
    }
    pub fn read_opt_16(&mut self, mut address: usize) -> Option<u16> {
        if self.read_mmu_enabled() {
            let address_maybe = self.virtual_to_physical(address as u64);
            match address_maybe {
                Some(addr) => address = addr.0 as usize,
                None => return None,
            }
        }

        let address = address as usize;

        if in_rom_memory(address) {
            let read = self.rom().read_16(address - MEMORY_ROM_START);
            read
        } else {
            self.ram().read_16(address - MEMORY_RAM_START)
        }
    }
    pub fn read_opt_32(&mut self, address: usize) -> Option<u32> {
        if in_rom_memory(address) {
            self.rom().read_32(address - MEMORY_ROM_START)
        } else {
            self.ram().read_32(address - MEMORY_RAM_START)
        }
    }

    pub fn read_opt_64(&mut self, address: usize) -> Option<u64> {
        if in_rom_memory(address) {
            self.rom().read_64(address - MEMORY_ROM_START)
        } else {
            self.ram().read_64(address - MEMORY_RAM_START)
        }
    }

    pub fn read_opt_usize(&mut self, address: usize) -> Option<usize> {
        if in_rom_memory(address) {
            self.rom().read_usize(address - MEMORY_ROM_START)
        } else {
            self.ram().read_usize(address - MEMORY_RAM_START)
        }
    }

    pub fn read_real_8(&mut self, address: usize) -> Option<u8> {
        let ptr = unsafe { Box::from_raw(address as *mut u8) };
        let value = *ptr;
        Box::into_raw(ptr);
        Some(value)
    }

    pub fn read_real_16(&mut self, address: usize) -> Option<u16> {
        let ptr = unsafe { Box::from_raw(address as *mut u16) };
        let value = *ptr;
        Box::into_raw(ptr);
        Some(value)
    }

    pub fn read_real_32(&mut self, address: usize) -> Option<u32> {
        let ptr = unsafe { Box::from_raw(address as *mut u32) };
        let value = *ptr;
        Box::into_raw(ptr);
        Some(value)
    }

    pub fn read_real_64(&mut self, address: usize) -> Option<u64> {
        let ptr = unsafe { Box::from_raw(address as *mut u64) };
        let value = *ptr;
        Box::into_raw(ptr);
        Some(value)
    }

    pub fn read_real_usize(&mut self, address: usize) -> Option<usize> {
        let ptr = unsafe { Box::from_raw(address as *mut usize) };
        let value = *ptr;
        Box::into_raw(ptr);
        Some(value)
    }

    pub fn read_real_cstring(&mut self, address: usize) -> Option<String> {
        let mut string = unsafe { CString::from_raw(address as *mut i8) };
        let value = string.to_str().unwrap().to_string();
        string.into_raw();
        Some(value)
    }

    pub fn write_real_8(&mut self, address: usize, value: u8) -> Option<()> {
        let ptr = unsafe { Box::from_raw(address as *mut u8) };
        *ptr = value;
        Box::into_raw(ptr);
        Some(())
    }

    pub fn write_real_16(&mut self, address: usize, value: u16) -> Option<()> {
        let ptr = unsafe { Box::from_raw(address as *mut u16) };
        *ptr = value;
        Box::into_raw(ptr);
        Some(())
    }

    pub fn write_real_32(&mut self, address: usize, value: u32) -> Option<()> {
        let ptr = unsafe { Box::from_raw(address as *mut u32) };
        *ptr = value;
        Box::into_raw(ptr);
        Some(())
    }

    pub fn write_real_64(&mut self, address: usize, value: u64) -> Option<()> {
        let ptr = unsafe { Box::from_raw(address as *mut u64) };
        *ptr = value;
        Box::into_raw(ptr);
        Some(())
    }

    pub fn write_real_usize(&mut self, address: usize, value: usize) -> Option<()> {
        let ptr = unsafe { Box::from_raw(address as *mut usize) };
        *ptr = value;
        Box::into_raw(ptr);
        Some(())
    }

    pub fn read_8(&mut self, address: u64) -> Option<u8> {
        if cpu::REAL_MODE_FLAG.load(Ordering::Relaxed) {
            self.read_real_8(address as usize)
        } else {
            let mut read_ok = true;
            let value = self.read_opt_8(address as usize).unwrap_or_else(|| {
                read_ok = false;
                0
            });
            if read_ok {
                Some(value)
            } else {
                self.exception_sender().send(Exception::PageFaultRead(address)).unwrap();
                None
            }
        }
    }
    pub fn read_16(&mut self, address: u64) -> Option<u16> {
        if cpu::REAL_MODE_FLAG.load(Ordering::Relaxed) {
            self.read_real_16(address as usize)
        } else {
            let mut read_ok = true;
            let value = self.read_opt_16(address as usize).unwrap_or_else(|| {
                read_ok = false;
                0
            });
            if read_ok {
                Some(value)
            } else {
                self.exception_sender().send(Exception::PageFaultRead(address)).unwrap();
                None
            }
        }
    }
    pub fn read_32(&mut self, address: u64) -> Option<u32> {
        if cpu::REAL_MODE_FLAG.load(Ordering::Relaxed) {
            self.read_real_32(address as usize)
        } else {
            let mut read_ok = true;
            let value = self.read_opt_32(address as usize).unwrap_or_else(|| {
                read_ok = false;
                0
            });
            if read_ok {
                Some(value)
            } else {
                self.exception_sender().send(Exception::PageFaultRead(address)).unwrap();
                None
            }
        }
    }


    pub fn read_64(&mut self, address: u64) -> Option<u64> {
        if cpu::REAL_MODE_FLAG.load(Ordering::Relaxed) {
            self.read_real_64(address as usize)
        } else {
            let mut read_ok = true;
            let value = self.read_opt_64(address as usize).unwrap_or_else(|| {
                read_ok = false;
                0
            });
            if read_ok {
                Some(value)
            } else {
                self.exception_sender().send(Exception::PageFaultRead(address)).unwrap();
                None
            }
        }
    }

    pub fn read_usize(&mut self, address: u64) -> Option<usize> {
        if cpu::REAL_MODE_FLAG.load(Ordering::Relaxed) {
            self.read_real_usize(address as usize)
        } else {
            let mut read_ok = true;
            let value = self.read_opt_usize(address as usize).unwrap_or_else(|| {
                read_ok = false;
                0
            });
            if read_ok {
                Some(value)
            } else {
                self.exception_sender().send(Exception::PageFaultRead(address)).unwrap();
                None
            }
        }
    }

    pub fn read_cstring(&mut self, address: u64) -> Option<String> {
        if cpu::REAL_MODE_FLAG.load(Ordering::Relaxed) {
            self.read_real_cstring(address as usize)
        } else {
            let mut read_ok = true;
            let value = if in_rom_memory(address as usize) {
                self.rom().read_cstring(address as usize - MEMORY_ROM_START)
            } else {
                self.ram().read_cstring(address as usize - MEMORY_RAM_START)
            }.unwrap_or_else(|| {
                read_ok = false;
                String::new()
            });
            if read_ok {
                Some(value)
            } else {
                self.exception_sender().send(Exception::PageFaultRead(address)).unwrap();
                None
            }
        }
    }

    pub fn write_8(&mut self, mut address: u64, byte: u8) -> Option<()> {
        if cpu::REAL_MODE_FLAG.load(Ordering::Relaxed) {
            self.write_real_8(address as usize, byte)
        } else {
            let original_address = address;
            let mut writable = true;
            if self.read_mmu_enabled() {
                (address, writable) = self.virtual_to_physical(address).unwrap_or_else(|| {
                    (0, false)
                });
            }

            if writable {
                let address = address as usize;

                if in_rom_memory(address) {
                    error(&format!("attempting to write to ROM address: {:#010X}", address));
                }

                match self.ram().in_bounds(address - MEMORY_RAM_START) {
                    true => {
                        self.ram().write_8(address - MEMORY_RAM_START, byte).unwrap();
                    }
                    false => {
                        println!("attempting to write to invalid address: {:#010X}", address);
                        self.exception_sender().send(Exception::PageFaultWrite(original_address)).unwrap();
                    }
                }
                Some(())
            } else {
                self.exception_sender().send(Exception::PageFaultWrite(original_address)).unwrap();
                None
            }
        }
    }
    pub fn write_16(&mut self, address: u64, half: u16) -> Option<()> {
        if cpu::REAL_MODE_FLAG.load(Ordering::Relaxed) {
            self.write_real_16(address as usize, half)
        } else {
            // first check if we can write to all addresses without faulting
            if self.read_mmu_enabled() {
                let (_, writable_0) = self.virtual_to_physical(address).unwrap_or_else(|| (0, false));
                let (_, writable_1) = self.virtual_to_physical(address + 1).unwrap_or_else(|| (0, false));
                if !writable_0 {
                    self.exception_sender().send(Exception::PageFaultWrite(address)).unwrap();
                    return None
                }
                if !writable_1 {
                    self.exception_sender().send(Exception::PageFaultWrite(address + 1)).unwrap();
                    return None
                }
            }

            // then do the actual writes
            self.write_8(address, (half & 0x00FF) as u8)?;
            self.write_8(address + 1, (half >> 8) as u8)?;
            Some(())
        }
    }
    pub fn write_32(&mut self, address: u64, word: u32) -> Option<()> {
        if cpu::REAL_MODE_FLAG.load(Ordering::Relaxed) {
            self.write_real_32(address as usize, word)
        } else {
            // first check if we can write to all addresses without faulting
            if self.read_mmu_enabled() {
                let (_, writable_0) = self.virtual_to_physical(address).unwrap_or_else(|| (0, false));
                let (_, writable_1) = self.virtual_to_physical(address + 1).unwrap_or_else(|| (0, false));
                let (_, writable_2) = self.virtual_to_physical(address + 2).unwrap_or_else(|| (0, false));
                let (_, writable_3) = self.virtual_to_physical(address + 3).unwrap_or_else(|| (0, false));
                if !writable_0 {
                    self.exception_sender().send(Exception::PageFaultWrite(address)).unwrap();
                    return None
                }
                if !writable_1 {
                    self.exception_sender().send(Exception::PageFaultWrite(address + 1)).unwrap();
                    return None
                }
                if !writable_2 {
                    self.exception_sender().send(Exception::PageFaultWrite(address + 2)).unwrap();
                    return None
                }
                if !writable_3 {
                    self.exception_sender().send(Exception::PageFaultWrite(address + 3)).unwrap();
                    return None
                }
            }

            // then do the actual writes
            self.write_8(address, (word & 0x000000FF) as u8)?;
            self.write_8(address + 1, ((word & 0x0000FF00) >> 8) as u8)?;
            self.write_8(address + 2, ((word & 0x00FF0000) >> 16) as u8)?;
            self.write_8(address + 3, ((word & 0xFF000000) >> 24) as u8)?;
            Some(())
        }
    }

    pub fn write_64(&mut self, address: u64, long: u64) -> Option<()> {
        if cpu::REAL_MODE_FLAG.load(Ordering::Relaxed) {
            self.write_real_64(address as usize, long)
        } else {
            // first check if we can write to all addresses without faulting
            if self.read_mmu_enabled() {
                let (_, writable_0) = self.virtual_to_physical(address).unwrap_or_else(|| (0, false));
                let (_, writable_1) = self.virtual_to_physical(address + 1).unwrap_or_else(|| (0, false));
                let (_, writable_2) = self.virtual_to_physical(address + 2).unwrap_or_else(|| (0, false));
                let (_, writable_3) = self.virtual_to_physical(address + 3).unwrap_or_else(|| (0, false));
                let (_, writable_4) = self.virtual_to_physical(address).unwrap_or_else(|| (0, false));
                let (_, writable_5) = self.virtual_to_physical(address + 1).unwrap_or_else(|| (0, false));
                let (_, writable_6) = self.virtual_to_physical(address + 2).unwrap_or_else(|| (0, false));
                let (_, writable_7) = self.virtual_to_physical(address + 3).unwrap_or_else(|| (0, false));
                if !writable_0 {
                    self.exception_sender().send(Exception::PageFaultWrite(address)).unwrap();
                    return None
                }
                if !writable_1 {
                    self.exception_sender().send(Exception::PageFaultWrite(address + 1)).unwrap();
                    return None
                }
                if !writable_2 {
                    self.exception_sender().send(Exception::PageFaultWrite(address + 2)).unwrap();
                    return None
                }
                if !writable_3 {
                    self.exception_sender().send(Exception::PageFaultWrite(address + 3)).unwrap();
                    return None
                }

                if !writable_4 {
                    self.exception_sender().send(Exception::PageFaultWrite(address + 4)).unwrap();
                    return None
                }
                if !writable_5 {
                    self.exception_sender().send(Exception::PageFaultWrite(address + 5)).unwrap();
                    return None
                }
                if !writable_6 {
                    self.exception_sender().send(Exception::PageFaultWrite(address + 6)).unwrap();
                    return None
                }
                if !writable_7 {
                    self.exception_sender().send(Exception::PageFaultWrite(address + 7)).unwrap();
                    return None
                }
            }

            // then do the actual writes
            self.write_8(address, (long & 0x000000FF) as u8)?;
            self.write_8(address + 1, ((long & 0x0000FF00) >> 8) as u8)?;
            self.write_8(address + 2, ((long & 0x00FF0000) >> 16) as u8)?;
            self.write_8(address + 3, ((long & 0xFF000000) >> 24) as u8)?;
            self.write_8(address + 4, ((long & 0x000000FF) >> 32) as u8)?;
            self.write_8(address + 5, ((long & 0x0000FF00) >> 40) as u8)?;
            self.write_8(address + 6, ((long & 0x00FF0000) >> 48) as u8)?;
            self.write_8(address + 7, ((long & 0xFF000000) >> 56) as u8)?;
            Some(())
        }
    }

    pub fn write_usize(&mut self, address: u64, value: usize) -> Option<()> {
        if cpu::REAL_MODE_FLAG.load(Ordering::Relaxed) {
            self.write_real_usize(address as usize, value)
        } else {
            if cfg!(target_pointer_width = "32") {
                self.write_32(address, value as u32)?;
            } else {
                self.write_64(address, value as u64)?;
            }
            Some(())
        }
    }

    pub fn get_actual_pointer(&mut self, address: u64) -> Option<*mut u8> {
        if cpu::REAL_MODE_FLAG.load(Ordering::Relaxed) {
            Some(address as *mut u8)
        } else {
            if address == 0 {
                return Some(std::ptr::null_mut::<u8>());
            }
            let mut read_ok = true;
            let value = if in_rom_memory(address as usize) {
                self.rom().get_irl_pointer(address as usize - MEMORY_ROM_START)
            } else {
                self.ram().get_irl_pointer(address as usize - MEMORY_RAM_START)
            }.unwrap_or_else(|| {
                read_ok = false;
                std::ptr::null_mut::<u8>()
            });
            if read_ok {
                Some(value)
            } else {
                self.exception_sender().send(Exception::PageFaultRead(address)).unwrap();
                None
            }
        }
    }
}
