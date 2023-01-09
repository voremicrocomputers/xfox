use std::{alloc, mem, slice};
use std::alloc::{alloc, dealloc, Layout};
use std::cell::UnsafeCell;
use std::ptr::write;
use std::sync::atomic::AtomicBool;

pub static mut DEBUG_TOGGLE: AtomicBool = AtomicBool::new(false);
pub static mut INTERRUPT_TOGGLE: AtomicBool = AtomicBool::new(false);
pub static mut EXCEPTION_TOGGLE: AtomicBool = AtomicBool::new(false);

pub struct MemoryAreaPointer {
    pub pointer: *mut u8,
    pub size: usize,
}

#[derive(Copy, Clone, Debug)]
pub enum MemoryAreaPointerError {
    OutOfBounds,
    ReadFail,
}

impl Drop for MemoryAreaPointer {
    fn drop(&mut self) {
        unsafe {
            let inner = Vec::from_raw_parts(self.pointer, self.size, self.size);
            drop(inner);
        }
    }
}

impl MemoryAreaPointer {
    pub fn allocate(size: usize) -> MemoryAreaPointer {
        let allocated_vec: Vec<u8> = Vec::with_capacity(size);
        let pointer = allocated_vec.as_ptr() as *mut u8;
        mem::forget(allocated_vec);
        MemoryAreaPointer {
            pointer,
            size,
        }
    }

    pub fn write_all(&self, buf: &[u8]) -> Result<(), MemoryAreaPointerError> {
        if buf.len() > self.size {
            return Err(MemoryAreaPointerError::OutOfBounds);
        }

        unsafe {
            let memory_ram = self.pointer;
            for (i, byte) in buf.iter().enumerate() {
                write(memory_ram.add(i), *byte);
            }
        }

        Ok(())
    }

    pub fn export(&self) -> Result<Vec<u8>, MemoryAreaPointerError> {
        let mut vec = Vec::with_capacity(self.size);
        unsafe {
            let mut memory_ram = self.pointer;
            for _ in 0..self.size {
                vec.push(*memory_ram);
                memory_ram = memory_ram.add(1);
            }
        }
        Ok(vec)
    }

    pub fn export_area(&self, offset: usize, size: usize) -> Result<Vec<u8>, MemoryAreaPointerError> {
        if offset + size > self.size {
            return Err(MemoryAreaPointerError::OutOfBounds);
        }

        let mut vec = Vec::with_capacity(size);
        unsafe {
            let mut memory_ram = self.pointer.add(offset) as *mut u8;
            for _ in 0..size {
                vec.push(*memory_ram);
                memory_ram = memory_ram.offset(1);
            }
        }
        Ok(vec)
    }

    pub fn read_8(&self, offset: usize) -> Option<u8> {

        //if offset > self.size {
        //    println!("optim warn: read out of bounds");
        //    return None;
        //}

        unsafe {
            let memory_ram = self.pointer.add(offset) as *mut u8;
            Some(*memory_ram)
        }
    }

    pub fn read_16(&self, offset: usize) -> Option<u16> {

        //if offset + 1 > self.size {
        //    println!("optim warn: read out of bounds");
        //    return None;
        //}

        unsafe {
            let memory_ram = self.pointer.add(offset) as *mut u16;
            Some(*memory_ram)
        }
    }

    pub fn read_32(&self, offset: usize) -> Option<u32> {
        //if offset + 4 > self.size {
        //    println!("optim warn: read out of bounds");
        //    return None;
        //}

        unsafe {
            let memory_ram = self.pointer.add(offset) as *mut u32;
            Some(*memory_ram)
        }
    }

    pub fn read_64(&self, offset: usize) -> Option<u64> {
        //if offset + 8 > self.size {
        //    println!("optim warn: read out of bounds");
        //    return None;
        //}

        unsafe {
            let memory_ram = self.pointer.add(offset) as *mut u64;
            Some(*memory_ram)
        }
    }

    pub fn read_usize(&self, offset: usize) -> Option<usize> {
        //if offset + 4 > self.size {
        //    println!("optim warn: read out of bounds");
        //    return None;
        //}

        unsafe {
            let memory_ram = self.pointer.add(offset) as *mut usize;
            Some(*memory_ram)
        }
    }

    pub fn read_cstring(&self, offset: usize) -> Option<String> {
        let mut vec = Vec::new();
        unsafe {
            let mut memory_ram = self.pointer.add(offset) as *mut u8;
            for _ in 0..self.size {
                let byte = *memory_ram;
                if byte == 0 {
                    break;
                }
                vec.push(byte);
                memory_ram = memory_ram.offset(1);
            }
        }
        Some(String::from_utf8(vec).unwrap())
    }

    pub fn write_8(&self, offset: usize, value: u8) -> Result<(), MemoryAreaPointerError> {
        if offset > self.size {
            return Err(MemoryAreaPointerError::OutOfBounds);
        }

        unsafe {
            let memory_ram = self.pointer;
            *memory_ram.add(offset) = value;
        }

        Ok(())
    }

    pub fn get_irl_pointer(&self, offset: usize) -> Option<*mut u8> {
        if offset > self.size {
            return None;
        }

        unsafe {
            let memory_ram = self.pointer;
            Some(memory_ram.add(offset))
        }
    }

    pub fn in_bounds(&self, offset: usize) -> bool {

        offset < self.size
    }
}

pub fn in_rom_memory(address: usize) -> bool {
    address >= crate::memory::MEMORY_ROM_START
}