// memory.rs

use crate::{error, optimisations};
use crate::cpu::Exception;

use std::cell::UnsafeCell;
use std::collections::HashMap;
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
    physical_address: u32,
    present: bool,
    rw: bool,
}

struct MemoryInner {
    ram: Arc<MemoryAreaPointer>,
    rom: Arc<MemoryAreaPointer>,
    mmu_enabled: Arc<AtomicBool>,
    tlb: Box<HashMap<u32, MemoryPage>>,
    paging_directory_address: Box<u32>,
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
    pub fn tlb(&self) -> &mut HashMap<u32, MemoryPage> { &mut self.inner().tlb }
    pub fn paging_directory_address(&self) -> &mut u32 { &mut self.inner().paging_directory_address }
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

    pub fn flush_tlb(&self, paging_directory_address: Option<u32>) {
        if let Some(address) = paging_directory_address {
            *self.paging_directory_address() = address;
        };

        self.tlb().clear();
    }

    pub fn flush_page(&self, virtual_address: u32) {
        let virtual_page = virtual_address & 0xFFFFF000;
        self.tlb().remove(&virtual_page);
    }

    pub fn insert_tlb_entry_from_tables(&mut self, page_directory_index: u32, page_table_index: u32) -> bool {
        let old_state = self.read_mmu_enabled();
        self.write_mmu_enabled(false);
        let directory_address = *self.paging_directory_address();
        let directory = self.read_opt_32(directory_address + (page_directory_index * 4));
        match directory {
            Some(directory) => {
                let dir_present = directory & 0b1 != 0;
                let dir_address = directory & 0xFFFFF000;
                if dir_present {
                    let table = self.read_opt_32(dir_address + (page_table_index * 4));
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

    pub fn virtual_to_physical(&mut self, virtual_address: u32) -> Option<(u32, bool)> {
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

    pub fn read_opt_8(&mut self, mut address: u32) -> Option<u8> {
        if self.read_mmu_enabled() {
            let address_maybe = self.virtual_to_physical(address as u32);
            match address_maybe {
                Some(addr) => address = addr.0,
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
    pub fn read_opt_16(&mut self, mut address: u32) -> Option<u16> {
        if self.read_mmu_enabled() {
            let address_maybe = self.virtual_to_physical(address as u32);
            match address_maybe {
                Some(addr) => address = addr.0,
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
    pub fn read_opt_32(&mut self, address: u32) -> Option<u32> {
        let address = address as usize;

        if in_rom_memory(address) {
            self.rom().read_32(address - MEMORY_ROM_START)
        } else {
            self.ram().read_32(address - MEMORY_RAM_START)
        }
    }

    pub fn read_8(&mut self, address: u32) -> Option<u8> {
        let mut read_ok = true;
        let value = self.read_opt_8(address).unwrap_or_else(|| { read_ok = false; 0 });
        if read_ok {
            Some(value)
        } else {
            self.exception_sender().send(Exception::PageFaultRead(address)).unwrap();
            None
        }
    }
    pub fn read_16(&mut self, address: u32) -> Option<u16> {
        let mut read_ok = true;
        let value = self.read_opt_16(address).unwrap_or_else(|| { read_ok = false; 0 });
        if read_ok {
            Some(value)
        } else {
            self.exception_sender().send(Exception::PageFaultRead(address)).unwrap();
            None
        }
    }
    pub fn read_32(&mut self, address: u32) -> Option<u32> {
        let mut read_ok = true;
        let value = self.read_opt_32(address).unwrap_or_else(|| { read_ok = false; 0 });
        if read_ok {
            Some(value)
        } else {
            self.exception_sender().send(Exception::PageFaultRead(address)).unwrap();
            None
        }
    }

    pub fn write_8(&mut self, mut address: u32, byte: u8) -> Option<()> {
        let original_address = address;
        let mut writable = true;
        if self.read_mmu_enabled() {
            (address, writable) = self.virtual_to_physical(address as u32).unwrap_or_else(|| {
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
    pub fn write_16(&mut self, address: u32, half: u16) -> Option<()> {
        // first check if we can write to all addresses without faulting
        if self.read_mmu_enabled() {
            let (_, writable_0) = self.virtual_to_physical(address).unwrap_or_else(|| (0, false));
            let (_, writable_1) = self.virtual_to_physical(address + 1).unwrap_or_else(|| (0, false));
            if !writable_0 { self.exception_sender().send(Exception::PageFaultWrite(address)).unwrap(); return None }
            if !writable_1 { self.exception_sender().send(Exception::PageFaultWrite(address + 1)).unwrap(); return None }
        }

        // then do the actual writes
        self.write_8(address, (half & 0x00FF) as u8)?;
        self.write_8(address + 1, (half >> 8) as u8)?;
        Some(())
    }
    pub fn write_32(&mut self, address: u32, word: u32) -> Option<()> {
        // first check if we can write to all addresses without faulting
        if self.read_mmu_enabled() {
            let (_, writable_0) = self.virtual_to_physical(address).unwrap_or_else(|| (0, false));
            let (_, writable_1) = self.virtual_to_physical(address + 1).unwrap_or_else(|| (0, false));
            let (_, writable_2) = self.virtual_to_physical(address + 2).unwrap_or_else(|| (0, false));
            let (_, writable_3) = self.virtual_to_physical(address + 3).unwrap_or_else(|| (0, false));
            if !writable_0 { self.exception_sender().send(Exception::PageFaultWrite(address)).unwrap(); return None }
            if !writable_1 { self.exception_sender().send(Exception::PageFaultWrite(address + 1)).unwrap(); return None }
            if !writable_2 { self.exception_sender().send(Exception::PageFaultWrite(address + 2)).unwrap(); return None }
            if !writable_3 { self.exception_sender().send(Exception::PageFaultWrite(address + 3)).unwrap(); return None }
        }

        // then do the actual writes
        self.write_8(address, (word & 0x000000FF) as u8)?;
        self.write_8(address + 1, ((word & 0x0000FF00) >>  8) as u8)?;
        self.write_8(address + 2, ((word & 0x00FF0000) >> 16) as u8)?;
        self.write_8(address + 3, ((word & 0xFF000000) >> 24) as u8)?;
        Some(())
    }
}
