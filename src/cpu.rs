// cpu.rs

// TODO: in the instruction match statement, all of the register ones have `let result` inside the if statement
//       move this up to match all of the other ones (or move all of the other ones down, which would probably be better anyways)

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};
use std::sync::mpsc::Receiver;
use lazy_static::lazy_static;

use crate::{Bus, cleanup, interop, optimisations};
use crate::interop::Argument;

lazy_static!{
    pub static ref REAL_MODE_FLAG: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
}

#[derive(Clone)]
pub struct Flag {
    pub swap_sp: bool,
    pub interrupt: bool,
    pub carry: bool,
    pub zero: bool,
    pub real_mode: Arc<AtomicBool>,
}

impl std::convert::From<Flag> for u8 {
    fn from(flag: Flag) -> u8 {
        (if flag.real_mode.load(Ordering::Relaxed) { 1 } else { 0 }) << 4 |
        (if flag.swap_sp { 1 } else { 0 }) << 3 |
            (if flag.interrupt { 1 } else { 0 }) << 2 |
            (if flag.carry { 1 } else { 0 }) << 1 |
            (if flag.zero { 1 } else { 0 }) << 0
    }
}

impl std::convert::From<u8> for Flag {
    fn from(byte: u8) -> Self {
        let real_mode = (byte >> 4) & 1 != 0;
        let swap_sp = ((byte >> 3) & 1) != 0;
        let interrupt = ((byte >> 2) & 1) != 0;
        let carry = ((byte >> 1) & 1) != 0;
        let zero = ((byte >> 0) & 1) != 0;
        REAL_MODE_FLAG.store(real_mode, Ordering::Relaxed);
        Flag { swap_sp, interrupt, carry, zero, real_mode: REAL_MODE_FLAG.clone() }
    }
}

#[derive(Debug)]
pub enum Exception {
    DivideByZero,
    InvalidOpcode(u64),
    PageFaultRead(u64),
    PageFaultWrite(u64),
}

#[derive(Debug)]
pub enum Interrupt {
    Exception(Exception),
    Request(u8), // u8 contains the interrupt vector value
}

pub struct Cpu {
    pub instruction_pointer: u64,
    pub stack_pointer: u64,
    pub exception_stack_pointer: u64,
    pub frame_pointer: u64,

    pub register: [u64; 32],
    pub flag: Flag,
    pub halted: bool,

    pub bus: Bus,

    pub next_interrupt: Option<u8>,
    pub next_soft_interrupt: Option<u8>,
    pub next_exception: Option<u8>,
    pub next_exception_operand: Option<u64>,

    pub debug: bool,
}

impl Cpu {
    pub fn new(bus: Bus) -> Self {
        Cpu {
            instruction_pointer: 0xF0000000,
            stack_pointer: 0x00000000,
            exception_stack_pointer: 0x00000000,
            frame_pointer: 0x00000000,
            register: [0; 32],
            flag: Flag { swap_sp: false, interrupt: false, carry: false, zero: false, real_mode: REAL_MODE_FLAG.clone() },
            halted: false,
            bus,
            next_interrupt: None,
            next_soft_interrupt: None,
            next_exception: None,
            next_exception_operand: None,
            debug: false,
        }
    }
    fn check_condition(&self, condition: Condition) -> bool {
        match condition {
            Condition::Always => true,
            Condition::Zero => self.flag.zero,
            Condition::NotZero => !self.flag.zero,
            Condition::Carry => self.flag.carry,
            Condition::NotCarry => !self.flag.carry,
            Condition::GreaterThan => !self.flag.carry && !self.flag.zero,
            Condition::LessThanEqualTo => self.flag.carry || self.flag.zero,
        }
    }
    fn relative_to_absolute(&self, relative_address: u64) -> u64 {
        self.instruction_pointer.wrapping_add(relative_address)
    }
    fn read_source(&mut self, source: Operand) -> Option<(u64, u64)> {
        self.read_source_and_i_care_if_its_a_pointer(source).map(|(a, b, _)| (a, b))

    }
    fn read_source_and_i_care_if_its_a_pointer(&mut self, source: Operand) -> Option<(u64, u64, bool)> {
        let mut instruction_pointer_offset = 2; // increment past opcode half
        let mut pointer = false;
        let source_value = match source {
            Operand::Register => {
                let register = self.bus.memory.read_8_bypass((self.instruction_pointer + instruction_pointer_offset) as u64)?;
                let value = self.read_register(register);
                instruction_pointer_offset += 1; // increment past 8 bit register number
                value
            }
            Operand::RegisterPtr(size) => {
                pointer = true;
                let register = self.bus.memory.read_8_bypass((self.instruction_pointer + instruction_pointer_offset) as u64)?;
                let pointer = self.read_register(register);
                let value = match size {
                    Size::Byte => self.bus.memory.read_8_bypass(pointer as u64)? as usize,
                    Size::Half => self.bus.memory.read_16_bypass(pointer as u64)? as usize,
                    Size::Word => self.bus.memory.read_32_bypass(pointer as u64)? as usize,
                    Size::Long => self.bus.memory.read_64_bypass(pointer as u64)? as usize,
                    Size::System => self.bus.memory.read_usize_bypass(pointer as u64)? as usize,
                };
                instruction_pointer_offset += 1; // increment past 8 bit register number
                value as u64
            }
            Operand::Immediate8 => {
                let value = self.bus.memory.read_8_bypass((self.instruction_pointer + instruction_pointer_offset) as u64)?;
                instruction_pointer_offset += 1; // increment past 8 bit immediate
                value as u64
            }
            Operand::Immediate16 => {
                let value = self.bus.memory.read_16_bypass((self.instruction_pointer + instruction_pointer_offset) as u64)?;
                instruction_pointer_offset += 2; // increment past 16 bit immediate
                value as u64
            }
            Operand::Immediate32 => {
                let value = self.bus.memory.read_32_bypass((self.instruction_pointer + instruction_pointer_offset) as u64)?;
                instruction_pointer_offset += 4; // increment past 32 bit immediate
                value as u64
            }
            Operand::Immediate64 => {
                let value = self.bus.memory.read_64_bypass((self.instruction_pointer + instruction_pointer_offset) as u64)?;
                instruction_pointer_offset += 8; // increment past 32 bit immediate
                value as u64
            }
            Operand::Immediate64Fit => {
                let value = self.bus.memory.read_usize_bypass((self.instruction_pointer + instruction_pointer_offset) as u64)? as u64;
                instruction_pointer_offset += 8; // increment past 32 bit immediate
                value as u64
            }
            Operand::ImmediatePtr(size) => {
                pointer = true;
                let pointer = self.bus.memory.read_64_bypass((self.instruction_pointer + instruction_pointer_offset) as u64)?;
                let value = match size {
                    Size::Byte => self.bus.memory.read_8_bypass(pointer as u64)? as usize,
                    Size::Half => self.bus.memory.read_16_bypass(pointer as u64)? as usize,
                    Size::Word => self.bus.memory.read_32_bypass(pointer as u64)? as usize,
                    Size::Long => self.bus.memory.read_64_bypass(pointer as u64)? as usize,
                    Size::System => self.bus.memory.read_usize_bypass(pointer as u64)? as usize,
                };
                instruction_pointer_offset += 8; // increment past 32 bit pointer
                value as u64
            }
        };
        Some((source_value, instruction_pointer_offset, pointer))
    }
    pub fn read_register(&self, register: u8) -> u64 {
        match register {
            0..=31 => self.register[register as usize],
            32 => self.stack_pointer as u64,
            33 => self.exception_stack_pointer as u64,
            34 => self.frame_pointer as u64,
            _ => panic!("Invalid register: {}", register),
        }
    }
    pub fn write_register(&mut self, register: u8, word: u64) {
        match register {
            0..=31 => {
                self.register[register as usize] = word;
            },
            32 => self.stack_pointer = word,
            33 => self.exception_stack_pointer = word,
            34 => self.frame_pointer = word,
            _ => panic!("Invalid register: {}", register),
        };
    }
    pub fn write_register_and_specify_if_its_a_pointer(&mut self, register: u8, word: u64, pointer: bool) {
        match register {
            0..=31 => {
                self.register[register as usize] = word;
            },
            32 => self.stack_pointer = word,
            33 => self.exception_stack_pointer = word,
            34 => self.frame_pointer = word,
            _ => panic!("Invalid register: {}", register),
        };
    }
    pub fn print_registers(&mut self) {
        for index in 0..2 {
            println!("r{}: {:#010X} | r{}:  {:#010X} | r{}: {:#010X} | r{}: {:#010X}",
                     index, self.register[index],
                     index + 8, self.register[index + 8],
                     index + 16, self.register[index + 16],
                     index + 24, self.register[index + 24]
            );
        }
        for index in 2..8 {
            println!("r{}: {:#010X} | r{}: {:#010X} | r{}: {:#010X} | r{}: {:#010X}",
                     index, self.register[index],
                     index + 8, self.register[index + 8],
                     index + 16, self.register[index + 16],
                     index + 24, self.register[index + 24]
            );
        }
        println!("rsp: {:#010X} | resp: {:#010X}", self.stack_pointer, self.exception_stack_pointer);
        println!("rfp: {:#010X}", self.frame_pointer);
    }
    pub fn push_stack_8(&mut self, byte: u8) {
        let decremented_stack_pointer = self.stack_pointer.overflowing_sub(1);
        if let Some(_) = self.bus.memory.write_8_bypass(decremented_stack_pointer.0 as u64, byte) {
            self.stack_pointer = decremented_stack_pointer.0;
        }
    }
    pub fn pop_stack_8(&mut self) -> Option<u8> {
        let byte = self.bus.memory.read_8_bypass(self.stack_pointer as u64);
        match byte {
            Some(byte) => {
                let incremented_stack_pointer = self.stack_pointer.overflowing_add(1);
                self.stack_pointer = incremented_stack_pointer.0;
                Some(byte)
            }
            None => None,
        }
    }
    pub fn push_stack_16(&mut self, half: u16) {
        let decremented_stack_pointer = self.stack_pointer.overflowing_sub(2);
        if let Some(_) = self.bus.memory.write_16_bypass(decremented_stack_pointer.0 as u64, half) {
            self.stack_pointer = decremented_stack_pointer.0;
        }
    }
    pub fn pop_stack_16(&mut self) -> Option<u16> {
        let half = self.bus.memory.read_16_bypass(self.stack_pointer as u64);
        match half {
            Some(half) => {
                let incremented_stack_pointer = self.stack_pointer.overflowing_add(2);
                self.stack_pointer = incremented_stack_pointer.0;
                Some(half)
            }
            None => None,
        }
    }
    pub fn push_stack_32(&mut self, word: u32) {
        let decremented_stack_pointer = self.stack_pointer.overflowing_sub(4);
        if let Some(_) = self.bus.memory.write_32_bypass(decremented_stack_pointer.0 as u64, word) {
            self.stack_pointer = decremented_stack_pointer.0;
        }
    }
    pub fn push_stack_64(&mut self, word: u64) {
        let decremented_stack_pointer = self.stack_pointer.overflowing_sub(8);
        if let Some(_) = self.bus.memory.write_64_bypass(decremented_stack_pointer.0 as u64, word) {
            self.stack_pointer = decremented_stack_pointer.0;
        }
    }
    pub fn push_stack_usize(&mut self, word: usize) {
        let decremented_stack_pointer = self.stack_pointer.overflowing_sub(if cfg!(target_pointer_width = "32") { 4 } else { 8 });
        if let Some(_) = self.bus.memory.write_usize_bypass(decremented_stack_pointer.0 as u64, word) {
            self.stack_pointer = decremented_stack_pointer.0;
        }
    }
    pub fn pop_stack_32(&mut self) -> Option<u32> {
        let word = self.bus.memory.read_32_bypass(self.stack_pointer as u64);
        match word {
            Some(word) => {
                let incremented_stack_pointer = self.stack_pointer.overflowing_add(4);
                self.stack_pointer = incremented_stack_pointer.0;
                Some(word as u32)
            }
            None => None,
        }
    }
    pub fn pop_stack_64(&mut self) -> Option<u64> {
        let word = self.bus.memory.read_64_bypass(self.stack_pointer);
        match word {
            Some(word) => {
                let incremented_stack_pointer = self.stack_pointer.overflowing_add(8);
                self.stack_pointer = incremented_stack_pointer.0;
                Some(word as u64)
            }
            None => None,
        }
    }
    pub fn pop_stack_usize(&mut self) -> Option<usize> {
        let word = self.bus.memory.read_usize_bypass(self.stack_pointer);
        match word {
            Some(word) => {
                let incremented_stack_pointer = self.stack_pointer.overflowing_add(if cfg!(target_pointer_width = "32") { 4 } else { 8 });
                self.stack_pointer = incremented_stack_pointer.0;
                Some(word)
            }
            None => None,
        }
    }
    pub fn exception_to_vector(&mut self, exception: Exception) -> (Option<u8>, Option<u64>) {
        match exception {
            Exception::DivideByZero => {
                (Some(0), None)
            }
            Exception::InvalidOpcode(opcode) => {
                (Some(1), Some(opcode))
            }
            Exception::PageFaultRead(virtual_address) => {
                (Some(2), Some(virtual_address))
            }
            Exception::PageFaultWrite(virtual_address) => {
                (Some(3), Some(virtual_address))
            }
        }
    }
    fn handle_interrupt(&mut self, vector: u8) {
        if self.debug { println!("interrupt!!! vector: {:#04X}", vector); }
        let address_of_pointer = vector as u64 * 4;

        let old_mmu_state = self.bus.memory.read_mmu_enabled();
        self.bus.memory.write_mmu_enabled(false);
        let address_maybe = self.bus.memory.read_64_bypass(address_of_pointer);
        if address_maybe == None {
            self.bus.memory.write_mmu_enabled(old_mmu_state);
            return;
        }
        let address = address_maybe.unwrap();
        self.bus.memory.write_mmu_enabled(old_mmu_state);

        if self.flag.swap_sp {
            let old_stack_pointer = self.stack_pointer;
            self.stack_pointer = self.exception_stack_pointer;
            self.push_stack_64(old_stack_pointer);
            self.push_stack_64(self.instruction_pointer);
            self.push_stack_8(u8::from(self.flag.clone()));
            self.flag.swap_sp = false;
        } else {
            self.push_stack_64(self.instruction_pointer);
            self.push_stack_8(u8::from(self.flag.clone()));
        }

        self.flag.interrupt = false; // prevent interrupts while already servicing an interrupt
        self.instruction_pointer = address;
    }
    fn handle_exception(&mut self, vector: u8, operand: Option<u64>) {
        if self.debug { println!("exception!!! vector: {:#04X}, operand: {:?}", vector, operand); }
        let address_of_pointer = (256 + vector as u64) * 4;

        let old_mmu_state = self.bus.memory.read_mmu_enabled();
        self.bus.memory.write_mmu_enabled(false);
        let address_maybe = self.bus.memory.read_usize_bypass(address_of_pointer);
        if address_maybe == None {
            self.bus.memory.write_mmu_enabled(old_mmu_state);
            return;
        }
        let address = address_maybe.unwrap();
        self.bus.memory.write_mmu_enabled(old_mmu_state);

        if self.flag.swap_sp {
            let old_stack_pointer = self.stack_pointer;
            self.stack_pointer = self.exception_stack_pointer;
            self.push_stack_64(old_stack_pointer);
            self.push_stack_64(self.instruction_pointer);
            self.push_stack_8(u8::from(self.flag.clone()));
            self.flag.swap_sp = false;
        } else {
            self.push_stack_64(self.instruction_pointer);
            self.push_stack_8(u8::from(self.flag.clone()));
        }

        if let Some(operand) = operand {
            self.push_stack_64(operand as u64);
        }

        self.flag.interrupt = false; // prevent interrupts while already servicing an interrupt
        self.instruction_pointer = address as u64;
    }
    // execute instruction from memory at the current instruction pointer
    pub fn execute_memory_instruction(&mut self) -> bool {
        if let Some(vector) = self.next_exception {
            self.handle_exception(vector, self.next_exception_operand);
            self.next_exception = None;
            self.next_exception_operand = None;
        } else if let Some(vector) = self.next_soft_interrupt {
            if self.flag.interrupt {
                self.handle_interrupt(vector);
                self.next_soft_interrupt = None;
            }
        } else if let Some(vector) = self.next_interrupt {
            if self.flag.interrupt {
                self.handle_interrupt(vector);
                self.next_interrupt = None;
            }
        }

        // if instruction pointer is 0x00000000, exit
        if self.instruction_pointer == 0 {
            if self.debug { println!("instruction pointer is 0x00000000, exiting"); }
            return false;
        }

        let opcode_maybe = self.bus.memory.read_16_bypass(self.instruction_pointer);
        if opcode_maybe == None {
            return true;
        }
        let opcode = opcode_maybe.unwrap();

        if let Some(instruction) = Instruction::from_half(opcode) {
            if self.debug { println!("{:#010X}: {:?}", self.instruction_pointer, instruction); }
            let next_instruction_pointer = self.execute_instruction(instruction);
            if let Some(next) = next_instruction_pointer {
                self.instruction_pointer = next;
            }
            if unsafe { optimisations::DEBUG_TOGGLE.load(Ordering::Relaxed) } {
                self.debug = !self.debug;
                unsafe { optimisations::DEBUG_TOGGLE.store(false, Ordering::Relaxed); }
            }
        } else {
            let size = ((opcode & 0b1100000000000000) >> 14) as u8;
            let instruction = ((opcode & 0b0011111100000000) >> 8) as u8;
            let empty = ((opcode & 0b0000000010000000) >> 7) as u8;
            let condition = ((opcode & 0b0000000001110000) >> 4) as u8;
            let destination = ((opcode & 0b0000000000001100) >> 2) as u8;
            let source = (opcode & 0b0000000000000011) as u8;

            println!("{:#010X}: bad opcode {:#06X}", self.instruction_pointer, opcode);
            println!("size instr  . cond dest src");
            println!("{:02b}   {:06b} {:01b} {:03b}  {:02b}   {:02b}", size, instruction, empty, condition, destination, source);
            println!("dumping RAM");
            self.bus.memory.dump();
            panic!("bad opcode");
        }
        return true;
    }
    // execute one instruction and return the next instruction pointer value
    fn execute_instruction(&mut self, instruction: Instruction) -> Option<u64> {
        match instruction {
            Instruction::Nop() => {
                Some(self.instruction_pointer + 2) // increment past opcode half
            }
            Instruction::Halt(condition) => {
                let instruction_pointer_offset = 2; // increment past opcode half
                let should_run = self.check_condition(condition);
                if should_run {
                    self.halted = true;
                    //if DEBUG { self.print_registers(); }
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Brk(condition) => {
                let instruction_pointer_offset = 2; // increment past opcode half
                let should_run = self.check_condition(condition);
                if should_run {
                    //self.breakpoint = true;
                    println!("Breakpoint reached");
                    self.print_registers();
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }

            Instruction::Add(size, condition, destination, source) => {
                let (source_value, mut instruction_pointer_offset) = self.read_source(source)?;
                if self.check_condition(condition) {
                    cleanup::mdma_with_overflow(self, &size, destination, &mut instruction_pointer_offset, source_value as u64, |a, b, bits| {
                        match bits {
                            -1 => { let res = (a as usize).overflowing_add(b as usize); (res.0 as u64, res.1) },
                            8  => { let res = (a as u8).overflowing_add(b as u8); (res.0 as u64, res.1) },
                            16 => { let res = (a as u16).overflowing_add(b as u16); (res.0 as u64, res.1) },
                            32 => { let res = (a as u32).overflowing_add(b as u32); (res.0 as u64, res.1) },
                            64 => { let res = (a as u64).overflowing_add(b as u64); (res.0 as u64, res.1) },
                            _ => panic!("invalid bit size"),
                        }
                    }, false);
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Inc(size, condition, source) => {
                let mut instruction_pointer_offset = 2; // increment past opcode half
                if self.check_condition(condition) {
                    cleanup::mdma_with_overflow(self, &size, source, &mut instruction_pointer_offset, 1, |a, b, bits| {
                        match bits {
                            -1 => { let res = (a as usize).overflowing_add(b as usize); (res.0 as u64, res.1) },
                            8  => { let res = (a as u8).overflowing_add(b as u8); (res.0 as u64, res.1) },
                            16 => { let res = (a as u16).overflowing_add(b as u16); (res.0 as u64, res.1) },
                            32 => { let res = (a as u32).overflowing_add(b as u32); (res.0 as u64, res.1) },
                            64 => { let res = (a as u64).overflowing_add(b as u64); (res.0 as u64, res.1) },
                            _ => panic!("invalid bit size"),
                        }
                    }, false);
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Sub(size, condition, destination, source) => {
                let (source_value, mut instruction_pointer_offset) = self.read_source(source)?;
                if self.check_condition(condition) {
                    cleanup::mdma_with_overflow(self, &size, destination, &mut instruction_pointer_offset, source_value as u64, |a, b, bits| {
                        match bits {
                            -1 => { let res = (a as usize).overflowing_sub(b as usize); (res.0 as u64, res.1) },
                            8  => { let res = (a as u8).overflowing_sub(b as u8); (res.0 as u64, res.1) },
                            16 => { let res = (a as u16).overflowing_sub(b as u16); (res.0 as u64, res.1) },
                            32 => { let res = (a as u32).overflowing_sub(b as u32); (res.0 as u64, res.1) },
                            64 => { let res = (a as u64).overflowing_sub(b as u64); (res.0 as u64, res.1) },
                            _ => panic!("invalid bit size"),
                        }
                    }, false);
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Dec(size, condition, source) => {
                let mut instruction_pointer_offset = 2; // increment past opcode half
                if self.check_condition(condition) {
                    cleanup::mdma_with_overflow(self, &size, source, &mut instruction_pointer_offset, 1, |a, b, bits| {
                        match bits {
                            -1 => { let res = (a as usize).overflowing_sub(b as usize); (res.0 as u64, res.1) },
                            8  => { let res = (a as u8).overflowing_sub(b as u8); (res.0 as u64, res.1) },
                            16 => { let res = (a as u16).overflowing_sub(b as u16); (res.0 as u64, res.1) },
                            32 => { let res = (a as u32).overflowing_sub(b as u32); (res.0 as u64, res.1) },
                            64 => { let res = (a as u64).overflowing_sub(b as u64); (res.0 as u64, res.1) },
                            _ => panic!("invalid bit size"),
                        }
                    }, false);
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Mul(size, condition, destination, source) => {
                let (source_value, mut instruction_pointer_offset) = self.read_source(source)?;
                if self.check_condition(condition) {
                    cleanup::mdma_with_overflow(self, &size, destination, &mut instruction_pointer_offset, source_value as u64, |a, b, bits| {
                        match bits {
                            -1 => { let res = (a as usize).overflowing_mul(b as usize); (res.0 as u64, res.1) },
                            8  => { let res = (a as u8).overflowing_mul(b as u8); (res.0 as u64, res.1) },
                            16 => { let res = (a as u16).overflowing_mul(b as u16); (res.0 as u64, res.1) },
                            32 => { let res = (a as u32).overflowing_mul(b as u32); (res.0 as u64, res.1) },
                            64 => { let res = (a as u64).overflowing_mul(b as u64); (res.0 as u64, res.1) },
                            _ => panic!("invalid bit size"),
                        }
                    }, false);
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Div(size, condition, destination, source) => {
                let (source_value, mut instruction_pointer_offset) = self.read_source(source)?;
                if self.check_condition(condition) {
                    cleanup::mdma_with_overflow(self, &size, destination, &mut instruction_pointer_offset, source_value as u64, |a, b, bits| {
                        match bits {
                            -1 => { let res = (a as usize).overflowing_mul(b as usize); (res.0 as u64, res.1) },
                            8  => { let res = (a as u8).overflowing_mul(b as u8); (res.0 as u64, res.1) },
                            16 => { let res = (a as u16).overflowing_mul(b as u16); (res.0 as u64, res.1) },
                            32 => { let res = (a as u32).overflowing_mul(b as u32); (res.0 as u64, res.1) },
                            64 => { let res = (a as u64).overflowing_mul(b as u64); (res.0 as u64, res.1) },
                            _ => panic!("invalid bit size"),
                        }
                    }, false);
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Rem(size, condition, destination, source) => {
                let (source_value, mut instruction_pointer_offset) = self.read_source(source)?;
                if self.check_condition(condition) {
                    cleanup::mdma_with_overflow(self, &size, destination, &mut instruction_pointer_offset, source_value as u64, |a, b, bits| {
                        match bits {
                            -1 => { let res = (a as usize) % (b as usize); (res as u64, false) },
                            8  => { let res = (a as u8) % (b as u8); (res as u64, false) },
                            16 => { let res = (a as u16) % (b as u16); (res as u64, false) },
                            32 => { let res = (a as u32) % (b as u32); (res as u64, false) },
                            64 => { let res = (a as u64) % (b as u64); (res as u64, false) },
                            _ => panic!("invalid bit size"),
                        }
                    }, true);
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }

            Instruction::And(size, condition, destination, source) => {
                let (source_value, mut instruction_pointer_offset) = self.read_source(source)?;
                if self.check_condition(condition) {
                    cleanup::mdma_with_overflow(self, &size, destination, &mut instruction_pointer_offset, source_value as u64, |a, b, bits| {
                        match bits {
                            -1 => { let res = (a as usize) & (b as usize); (res as u64, false) },
                            8  => { let res = (a as u8) & (b as u8); (res as u64, false) },
                            16 => { let res = (a as u16) & (b as u16); (res as u64, false) },
                            32 => { let res = (a as u32) & (b as u32); (res as u64, false) },
                            64 => { let res = (a as u64) & (b as u64); (res as u64, false) },
                            _ => panic!("invalid bit size"),
                        }
                    }, true);
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Or(size, condition, destination, source) => {
                let (source_value, mut instruction_pointer_offset) = self.read_source(source)?;
                if self.check_condition(condition) {
                    cleanup::mdma_with_overflow(self, &size, destination, &mut instruction_pointer_offset, source_value as u64, |a, b, bits| {
                        match bits {
                            -1 => { let res = (a as usize) | (b as usize); (res as u64, false) },
                            8  => { let res = (a as u8) | (b as u8); (res as u64, false) },
                            16 => { let res = (a as u16) | (b as u16); (res as u64, false) },
                            32 => { let res = (a as u32) | (b as u32); (res as u64, false) },
                            64 => { let res = (a as u64) | (b as u64); (res as u64, false) },
                            _ => panic!("invalid bit size"),
                        }
                    }, true);
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Xor(size, condition, destination, source) => {
                let (source_value, mut instruction_pointer_offset) = self.read_source(source)?;
                if self.check_condition(condition) {
                    cleanup::mdma_with_overflow(self, &size, destination, &mut instruction_pointer_offset, source_value as u64, |a, b, bits| {
                        match bits {
                            -1 => { let res = (a as usize) ^ (b as usize); (res as u64, false) },
                            8  => { let res = (a as u8) ^ (b as u8); (res as u64, false) },
                            16 => { let res = (a as u16) ^ (b as u16); (res as u64, false) },
                            32 => { let res = (a as u32) ^ (b as u32); (res as u64, false) },
                            64 => { let res = (a as u64) ^ (b as u64); (res as u64, false) },
                            _ => panic!("invalid bit size"),
                        }
                    }, true);
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Not(size, condition, source) => {
                let mut instruction_pointer_offset = 2; // increment past opcode half
                if self.check_condition(condition) {
                    cleanup::mdma_with_overflow(self, &size, source, &mut instruction_pointer_offset, 0, |a, b, bits| {
                        match bits {
                            -1 => { let res = !(a as usize); (res as u64, false) },
                            8  => { let res = !(a as u8); (res as u64, false) },
                            16 => { let res = !(a as u16); (res as u64, false) },
                            32 => { let res = !(a as u32); (res as u64, false) },
                            64 => { let res = !(a as u64); (res as u64, false) },
                            _ => panic!("invalid bit size"),
                        }
                    }, true);
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Sla(size, condition, destination, source) => {
                let (source_value, mut instruction_pointer_offset) = self.read_source(source)?;
                if self.check_condition(condition) {
                    cleanup::mdma_with_overflow(self, &size, destination, &mut instruction_pointer_offset, source_value as u64, |a, b, bits| {
                        match bits {
                            -1 => { let res = (a as usize) << (b as usize); (res as u64, a & (1 << (usize::BITS - 1)) != 0) },
                            8  => { let res = (a as u8) << (b as u8); (res as u64, a & (1 << 7) != 0) },
                            16 => { let res = (a as u16) << (b as u16); (res as u64, a & (1 << 15) != 0) },
                            32 => { let res = (a as u32) << (b as u32); (res as u64, a & (1 << 31) != 0) },
                            64 => { let res = (a as u64) << (b as u64); (res as u64, a & (1 << 63) != 0) },
                            _ => panic!("invalid bit size"),
                        }
                    }, false);
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Sra(size, condition, destination, source) => {
                let (source_value, mut instruction_pointer_offset) = self.read_source(source)?;
                if self.check_condition(condition) {
                    cleanup::mdma_with_overflow(self, &size, destination, &mut instruction_pointer_offset, source_value as u64, |a, b, bits| {
                        match bits {
                            -1 => { let res = (a as usize) >> (b as usize); (res as u64, a & (1 << 0) != 0) },
                            8  => { let res = (a as u8) >> (b as u8); (res as u64, a & (1 << 0) != 0) },
                            16 => { let res = (a as u16) >> (b as u16); (res as u64, a & (1 << 0) != 0) },
                            32 => { let res = (a as u32) >> (b as u32); (res as u64, a & (1 << 0) != 0) },
                            64 => { let res = (a as u64) >> (b as u64); (res as u64, a & (1 << 0) != 0) },
                            _ => panic!("invalid bit size"),
                        }
                    }, false);
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Srl(size, condition, destination, source) => { // i love dj s3rl!
                let (source_value, mut instruction_pointer_offset) = self.read_source(source)?;
                if self.check_condition(condition) {
                    cleanup::mdma_with_overflow(self, &size, destination, &mut instruction_pointer_offset, source_value as u64, |a, b, bits| {
                        match bits {
                            -1 => { let res = (a as usize) >> (b as usize); (res as u64 & { if usize::BITS == 64 { 0x7FFF_FFFF_FFFF_FFFF } else { 0x7FFF_FFFF } }, a & (1 << 0) != 0) },
                            8  => { let res = (a as u8) >> (b as u8); (res as u64 & 0x7F, a & (1 << 0) != 0) },
                            16 => { let res = (a as u16) >> (b as u16); (res as u64 & 0x7FFF, a & (1 << 0) != 0) },
                            32 => { let res = (a as u32) >> (b as u32); (res as u64 & 0x7FFF_FFFF, a & (1 << 0) != 0) },
                            64 => { let res = (a as u64) >> (b as u64); (res as u64 & 0x7FFF_FFFF_FFFF_FFFF, a & (1 << 0) != 0) },
                            _ => panic!("invalid bit size"),
                        }
                    }, false);
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Rol(size, condition, destination, source) => {
                let (source_value, mut instruction_pointer_offset) = self.read_source(source)?;
                if self.check_condition(condition) {
                    cleanup::mdma_with_overflow(self, &size, destination, &mut instruction_pointer_offset, source_value as u64, |a, b, bits| {
                        match bits {
                            -1 => { let res = (a as usize).rotate_left(b as u32); (res as u64, a & (1 << (usize::BITS - 1)) != 0) },
                            8  => { let res = (a as u8).rotate_left(b as u32); (res as u64, a & (1 << 7) != 0) },
                            16 => { let res = (a as u16).rotate_left(b as u32); (res as u64, a & (1 << 15) != 0) },
                            32 => { let res = (a as u32).rotate_left(b as u32); (res as u64, a & (1 << 31) != 0) },
                            64 => { let res = (a as u64).rotate_left(b as u32); (res as u64, a & (1 << 63) != 0) },
                            _ => panic!("invalid bit size"),
                        }
                    }, false);
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Ror(size, condition, destination, source) => {
                let (source_value, mut instruction_pointer_offset) = self.read_source(source)?;
                if self.check_condition(condition) {
                    cleanup::mdma_with_overflow(self, &size, destination, &mut instruction_pointer_offset, source_value as u64, |a, b, bits| {
                        match bits {
                            -1 => { let res = (a as usize).rotate_right(b as u32); (res as u64, a & (1 << 0) != 0) },
                            8  => { let res = (a as u8).rotate_right(b as u32); (res as u64, a & (1 << 0) != 0) },
                            16 => { let res = (a as u16).rotate_right(b as u32); (res as u64, a & (1 << 0) != 0) },
                            32 => { let res = (a as u32).rotate_right(b as u32); (res as u64, a & (1 << 0) != 0) },
                            64 => { let res = (a as u64).rotate_right(b as u32); (res as u64, a & (1 << 0) != 0) },
                            _ => panic!("invalid bit size"),
                        }
                    }, false);
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }

            Instruction::Bse(size, condition, destination, source) => {
                let (source_value, mut instruction_pointer_offset) = self.read_source(source)?;
                let should_run = self.check_condition(condition);
                match destination {
                    Operand::Register => {
                        let register = self.bus.memory.read_8(self.instruction_pointer + instruction_pointer_offset)?;
                        match size {
                            Size::Byte => {
                                let result = self.read_register(register) as u8 | (1 << source_value);
                                if should_run {
                                    self.write_register(register, (self.read_register(register) & 0xFFFFFF00) | (result as u64));
                                }
                            }
                            Size::Half => {
                                let result = self.read_register(register) as u16 | (1 << source_value);
                                if should_run {
                                    self.write_register(register, (self.read_register(register) & 0xFFFF0000) | (result as u64));
                                }
                            }
                            Size::Word => {
                                let result = self.read_register(register) as u32 | (1 << source_value);
                                if should_run {
                                    self.write_register(register, (self.read_register(register)) | (result as u64));
                                }
                            }
                            Size::Long => {
                                let result = self.read_register(register) as u64 | (1 << source_value);
                                if should_run {
                                    self.write_register(register, result);
                                }
                            }
                            Size::System => {
                                let result = self.read_register(register) as usize | (1 << source_value);
                                if should_run {
                                    self.write_register(register, result as u64);
                                }
                            }
                        }
                        instruction_pointer_offset += 1; // increment past 8 bit register number
                    }
                    Operand::RegisterPtr(_) => {
                        let register = self.bus.memory.read_8(self.instruction_pointer + instruction_pointer_offset)?;
                        let pointer = self.read_register(register);
                        match size {
                            Size::Byte => {
                                let result = self.bus.memory.read_8(pointer)? | (1 << source_value);
                                if should_run {
                                    self.bus.memory.write_8(pointer, result)?;
                                }
                            }
                            Size::Half => {
                                let result = self.bus.memory.read_16(pointer)? | (1 << source_value);
                                if should_run {
                                    self.bus.memory.write_16(pointer, result)?;
                                }
                            }
                            Size::Word => {
                                let result = self.bus.memory.read_32(pointer)? | (1 << source_value);
                                if should_run {
                                    self.bus.memory.write_32(pointer, result)?;
                                }
                            }
                            Size::Long => {
                                let result = self.bus.memory.read_64(pointer)? | (1 << source_value);
                                if should_run {
                                    self.bus.memory.write_64(pointer, result)?;
                                }
                            }
                            Size::System => {
                                let result = self.bus.memory.read_usize(pointer)? | (1 << source_value);
                                if should_run {
                                    self.bus.memory.write_usize(pointer, result)?;
                                }
                            }
                        }
                        instruction_pointer_offset += 1; // increment past 8 bit register number
                    }
                    Operand::ImmediatePtr(_) => {
                        let pointer = self.bus.memory.read_64(self.instruction_pointer + instruction_pointer_offset)?;
                        match size {
                            Size::Byte => {
                                let result = self.bus.memory.read_8(pointer)? | (1 << source_value);
                                if should_run {
                                    self.bus.memory.write_8(pointer, result)?;
                                }
                            }
                            Size::Half => {
                                let result = self.bus.memory.read_16(pointer)? | (1 << source_value);
                                if should_run {
                                    self.bus.memory.write_16(pointer, result)?;
                                }
                            }
                            Size::Word => {
                                let result = self.bus.memory.read_32(pointer)? | (1 << source_value);
                                if should_run {
                                    self.bus.memory.write_32(pointer, result)?;
                                }
                            }
                            Size::Long => {
                                let result = self.bus.memory.read_64(pointer)? | (1 << source_value);
                                if should_run {
                                    self.bus.memory.write_64(pointer, result)?;
                                }
                            }
                            Size::System => {
                                let result = self.bus.memory.read_usize(pointer)? | (1 << source_value);
                                if should_run {
                                    self.bus.memory.write_usize(pointer, result)?;
                                }
                            }
                        }
                        instruction_pointer_offset += 8; // increment past 32 bit pointer
                    }
                    _ => panic!("Attempting to use an immediate value as a destination"),
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Bcl(size, condition, destination, source) => {
                let (source_value, mut instruction_pointer_offset) = self.read_source(source)?;
                let should_run = self.check_condition(condition);
                match destination {
                    Operand::Register => {
                        let register = self.bus.memory.read_8(self.instruction_pointer + instruction_pointer_offset)?;
                        match size {
                            Size::Byte => {
                                let result = self.read_register(register) as u8 & !(1 << source_value);
                                if should_run {
                                    self.write_register(register, (self.read_register(register) & 0xFFFFFF00) | (result as u64));
                                }
                            }
                            Size::Half => {
                                let result = self.read_register(register) as u16 & !(1 << source_value);
                                if should_run {
                                    self.write_register(register, (self.read_register(register) & 0xFFFF0000) | (result as u64));
                                }
                            }
                            Size::Word => {
                                let result = self.read_register(register) & !(1 << source_value);
                                if should_run {
                                    self.write_register(register, result);
                                }
                            }
                            Size::Long => {
                                let result = self.read_register(register) & !(1 << source_value);
                                if should_run {
                                    self.write_register(register, result);
                                }
                            }
                            Size::System => {
                                let result = self.read_register(register) as usize & !(1 << source_value);
                                if should_run {
                                    self.write_register(register, result as u64);
                                }
                            }
                        }
                        instruction_pointer_offset += 1; // increment past 8 bit register number
                    }
                    Operand::RegisterPtr(_) => {
                        let register = self.bus.memory.read_8(self.instruction_pointer + instruction_pointer_offset)?;
                        let pointer = self.read_register(register);
                        match size {
                            Size::Byte => {
                                let result = self.bus.memory.read_8(pointer)? & !(1 << source_value);
                                if should_run {
                                    self.bus.memory.write_8(pointer, result)?;
                                }
                            }
                            Size::Half => {
                                let result = self.bus.memory.read_16(pointer)? & !(1 << source_value);
                                if should_run {
                                    self.bus.memory.write_16(pointer, result)?;
                                }
                            }
                            Size::Word => {
                                let result = self.bus.memory.read_32(pointer)? & !(1 << source_value);
                                if should_run {
                                    self.bus.memory.write_32(pointer, result)?;
                                }
                            }
                            Size::Long => {
                                let result = self.bus.memory.read_64(pointer)? & !(1 << source_value);
                                if should_run {
                                    self.bus.memory.write_64(pointer, result)?;
                                }
                            }
                            Size::System => {
                                let result = self.bus.memory.read_usize(pointer)? & !(1 << source_value);
                                if should_run {
                                    self.bus.memory.write_usize(pointer, result)?;
                                }
                            }
                        }
                        instruction_pointer_offset += 1; // increment past 8 bit register number
                    }
                    Operand::ImmediatePtr(_) => {
                        let pointer = self.bus.memory.read_64(self.instruction_pointer + instruction_pointer_offset)?;
                        match size {
                            Size::Byte => {
                                let result = self.bus.memory.read_8(pointer)? & !(1 << source_value);
                                if should_run {
                                    self.bus.memory.write_8(pointer, result)?;
                                }
                            }
                            Size::Half => {
                                let result = self.bus.memory.read_16(pointer)? & !(1 << source_value);
                                if should_run {
                                    self.bus.memory.write_16(pointer, result)?;
                                }
                            }
                            Size::Word => {
                                let result = self.bus.memory.read_32(pointer)? & !(1 << source_value);
                                if should_run {
                                    self.bus.memory.write_32(pointer, result)?;
                                }
                            }
                            Size::Long => {
                                let result = self.bus.memory.read_64(pointer)? & !(1 << source_value);
                                if should_run {
                                    self.bus.memory.write_64(pointer, result)?;
                                }
                            }
                            Size::System => {
                                let result = self.bus.memory.read_usize(pointer)? & !(1 << source_value);
                                if should_run {
                                    self.bus.memory.write_usize(pointer, result)?;
                                }
                            }
                        }
                        instruction_pointer_offset += 8; // increment past 32 bit pointer
                    }
                    _ => panic!("Attempting to use an immediate value as a destination"),
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Bts(size, condition, destination, source) => {
                let (source_value, mut instruction_pointer_offset) = self.read_source(source)?;
                let should_run = self.check_condition(condition);
                match destination {
                    Operand::Register => {
                        let register = self.bus.memory.read_8(self.instruction_pointer + instruction_pointer_offset)?;
                        match size {
                            Size::Byte => {
                                let result = self.read_register(register) as u8 & (1 << source_value) == 0;
                                if should_run {
                                    self.flag.zero = result;
                                }
                            }
                            Size::Half => {
                                let result = self.read_register(register) as u16 & (1 << source_value) == 0;
                                if should_run {
                                    self.flag.zero = result;
                                }
                            }
                            Size::Word => {
                                let result = self.read_register(register) as u32 & (1 << source_value) == 0;
                                if should_run {
                                    self.flag.zero = result;
                                }
                            }
                            Size::Long => {
                                let result = self.read_register(register) & (1 << source_value) == 0;
                                if should_run {
                                    self.flag.zero = result;
                                }
                            }
                            Size::System => {
                                let result = self.read_register(register) as usize & (1 << source_value) == 0;
                                if should_run {
                                    self.flag.zero = result;
                                }
                            }
                        }
                        instruction_pointer_offset += 1; // increment past 8 bit register number
                    }
                    Operand::RegisterPtr(_) => {
                        let register = self.bus.memory.read_8(self.instruction_pointer + instruction_pointer_offset)?;
                        let pointer = self.read_register(register);
                        match size {
                            Size::Byte => {
                                let result = self.bus.memory.read_8(pointer)? & (1 << source_value) == 0;
                                if should_run {
                                    self.flag.zero = result;
                                }
                            }
                            Size::Half => {
                                let result = self.bus.memory.read_16(pointer)? & (1 << source_value) == 0;
                                if should_run {
                                    self.flag.zero = result;
                                }
                            }
                            Size::Word => {
                                let result = self.bus.memory.read_32(pointer)? & (1 << source_value) == 0;
                                if should_run {
                                    self.flag.zero = result;
                                }
                            }
                            Size::Long => {
                                let result = self.bus.memory.read_64(pointer)? & (1 << source_value) == 0;
                                if should_run {
                                    self.flag.zero = result;
                                }
                            }
                            Size::System => {
                                let result = self.bus.memory.read_usize(pointer)? & (1 << source_value) == 0;
                                if should_run {
                                    self.flag.zero = result;
                                }
                            }
                        }
                        instruction_pointer_offset += 1; // increment past 8 bit register number
                    }
                    Operand::ImmediatePtr(_) => {
                        let pointer = self.bus.memory.read_64(self.instruction_pointer + instruction_pointer_offset)?;
                        match size {
                            Size::Byte => {
                                let result = self.bus.memory.read_8(pointer)? & (1 << source_value) == 0;
                                if should_run {
                                    self.flag.zero = result;
                                }
                            }
                            Size::Half => {
                                let result = self.bus.memory.read_16(pointer)? & (1 << source_value) == 0;
                                if should_run {
                                    self.flag.zero = result;
                                }
                            }
                            Size::Word => {
                                let result = self.bus.memory.read_32(pointer)? & (1 << source_value) == 0;
                                if should_run {
                                    self.flag.zero = result;
                                }
                            }
                            Size::Long => {
                                let result = self.bus.memory.read_64(pointer)? & (1 << source_value) == 0;
                                if should_run {
                                    self.flag.zero = result;
                                }
                            }
                            Size::System => {
                                let result = self.bus.memory.read_usize(pointer)? & (1 << source_value) == 0;
                                if should_run {
                                    self.flag.zero = result;
                                }
                            }
                        }
                        instruction_pointer_offset += 8; // increment past 32 bit pointer
                    }
                    _ => panic!("Attempting to use an immediate value as a destination"),
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }

            Instruction::Cmp(size, condition, destination, source) => {
                let (source_value, mut instruction_pointer_offset) = self.read_source(source)?;
                let should_run = self.check_condition(condition);
                match destination {
                    Operand::Register => {
                        let register = self.bus.memory.read_8(self.instruction_pointer + instruction_pointer_offset)?;
                        match size {
                            Size::Byte => {
                                if should_run {
                                    let result = (self.read_register(register) as u8).overflowing_sub(source_value as u8);
                                    self.flag.zero = result.0 == 0;
                                    self.flag.carry = result.1;
                                }
                            }
                            Size::Half => {
                                if should_run {
                                    let result = (self.read_register(register) as u16).overflowing_sub(source_value as u16);
                                    self.flag.zero = result.0 == 0;
                                    self.flag.carry = result.1;
                                }
                            }
                            Size::Word => {
                                if should_run {
                                    let result = (self.read_register(register) as u32).overflowing_sub(source_value as u32);
                                    self.flag.zero = result.0 == 0;
                                    self.flag.carry = result.1;
                                }
                            }
                            Size::Long => {
                                if should_run {
                                    let result = self.read_register(register).overflowing_sub(source_value as u64);
                                    self.flag.zero = result.0 == 0;
                                    self.flag.carry = result.1;
                                }
                            }
                            Size::System => {
                                if should_run {
                                    let result = (self.read_register(register) as usize).overflowing_sub(source_value as usize);
                                    self.flag.zero = result.0 == 0;
                                    self.flag.carry = result.1;
                                }
                            }
                        }
                        instruction_pointer_offset += 1; // increment past 8 bit register number
                    }
                    Operand::RegisterPtr(_) => {
                        let register = self.bus.memory.read_8(self.instruction_pointer + instruction_pointer_offset)?;
                        let pointer = self.read_register(register);
                        match size {
                            Size::Byte => {
                                let result = self.bus.memory.read_8(pointer)?.overflowing_sub(source_value as u8);
                                if should_run {
                                    self.flag.zero = result.0 == 0;
                                    self.flag.carry = result.1;
                                }
                            }
                            Size::Half => {
                                let result = self.bus.memory.read_16(pointer)?.overflowing_sub(source_value as u16);
                                if should_run {
                                    self.flag.zero = result.0 == 0;
                                    self.flag.carry = result.1;
                                }
                            }
                            Size::Word => {
                                let result = self.bus.memory.read_32(pointer)?.overflowing_sub(source_value as u32);
                                if should_run {
                                    self.flag.zero = result.0 == 0;
                                    self.flag.carry = result.1;
                                }
                            }
                            Size::Long => {
                                let result = self.bus.memory.read_64(pointer)?.overflowing_sub(source_value as u64);
                                if should_run {
                                    self.flag.zero = result.0 == 0;
                                    self.flag.carry = result.1;
                                }
                            }
                            Size::System => {
                                let result = self.bus.memory.read_usize(pointer)?.overflowing_sub(source_value as usize);
                                if should_run {
                                    self.flag.zero = result.0 == 0;
                                    self.flag.carry = result.1;
                                }
                            }
                        }
                        instruction_pointer_offset += 1; // increment past 8 bit register number
                    }
                    Operand::ImmediatePtr(_) => {
                        let pointer = self.bus.memory.read_64(self.instruction_pointer + instruction_pointer_offset)?;
                        match size {
                            Size::Byte => {
                                let result = self.bus.memory.read_8(pointer)?.overflowing_sub(source_value as u8);
                                if should_run {
                                    self.flag.zero = result.0 == 0;
                                    self.flag.carry = result.1;
                                }
                            }
                            Size::Half => {
                                let result = self.bus.memory.read_16(pointer)?.overflowing_sub(source_value as u16);
                                if should_run {
                                    self.flag.zero = result.0 == 0;
                                    self.flag.carry = result.1;
                                }
                            }
                            Size::Word => {
                                let result = self.bus.memory.read_32(pointer)?.overflowing_sub(source_value as u32);
                                if should_run {
                                    self.flag.zero = result.0 == 0;
                                    self.flag.carry = result.1;
                                }
                            }
                            Size::Long => {
                                let result = self.bus.memory.read_64(pointer)?.overflowing_sub(source_value as u64);
                                if should_run {
                                    self.flag.zero = result.0 == 0;
                                    self.flag.carry = result.1;
                                }
                            }
                            Size::System => {
                                let result = self.bus.memory.read_usize(pointer)?.overflowing_sub(source_value as usize);
                                if should_run {
                                    self.flag.zero = result.0 == 0;
                                    self.flag.carry = result.1;
                                }
                            }
                        }
                        instruction_pointer_offset += 8; // increment past 32 bit pointer
                    }
                    _ => panic!("Attempting to use an immediate value as a destination"),
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Mov(size, condition, destination, source) => {
                let (source_value, mut instruction_pointer_offset, isptr) = self.read_source_and_i_care_if_its_a_pointer(source)?;
                let should_run = self.check_condition(condition);
                match destination {
                    Operand::Register => {
                        let register = self.bus.memory.read_8(self.instruction_pointer + instruction_pointer_offset)?;
                        match size {
                            Size::Byte => {
                                if should_run {
                                    self.write_register_and_specify_if_its_a_pointer(register, (self.read_register(register) & 0xFFFFFF00) | (source_value as u64 & 0x000000FF), isptr);
                                }
                            }
                            Size::Half => {
                                if should_run {
                                    self.write_register_and_specify_if_its_a_pointer(register, (self.read_register(register) & 0xFFFF0000) | (source_value as u64 & 0x0000FFFF), isptr);
                                }
                            }
                            Size::Word => {
                                if should_run {
                                    self.write_register_and_specify_if_its_a_pointer(register, source_value as u64, isptr);
                                }
                            }
                            Size::Long => {
                                if should_run {
                                    self.write_register_and_specify_if_its_a_pointer(register, source_value as u64, isptr);
                                }
                            }
                            Size::System => {
                                if should_run {
                                    self.write_register_and_specify_if_its_a_pointer(register, source_value as u64, isptr);
                                }
                            }
                        }
                        instruction_pointer_offset += 1; // increment past 8 bit register number
                    }
                    Operand::RegisterPtr(_) => {
                        let register = self.bus.memory.read_8(self.instruction_pointer + instruction_pointer_offset)?;
                        let pointer = self.read_register(register);
                        match size {
                            Size::Byte => {
                                if should_run {
                                    self.bus.memory.write_8(pointer, source_value as u8)?;
                                }
                            }
                            Size::Half => {
                                if should_run {
                                    self.bus.memory.write_16(pointer, source_value as u16)?;
                                }
                            }
                            Size::Word => {
                                if should_run {
                                    self.bus.memory.write_32(pointer, source_value as u32)?;
                                }
                            }
                            Size::Long => {
                                if should_run {
                                    self.bus.memory.write_64(pointer, source_value as u64)?;
                                }
                            }
                            Size::System => {
                                if should_run {
                                    self.bus.memory.write_usize(pointer, source_value as usize)?;
                                }
                            }
                        }
                        instruction_pointer_offset += 1; // increment past 8 bit register number
                    }
                    Operand::ImmediatePtr(_) => {
                        let pointer = self.bus.memory.read_64(self.instruction_pointer + instruction_pointer_offset)?;
                        match size {
                            Size::Byte => {
                                if should_run {
                                    self.bus.memory.write_8(pointer, source_value as u8)?;
                                }
                            }
                            Size::Half => {
                                if should_run {
                                    self.bus.memory.write_16(pointer, source_value as u16)?;
                                }
                            }
                            Size::Word => {
                                if should_run {
                                    self.bus.memory.write_32(pointer, source_value as u32)?;
                                }
                            }
                            Size::Long => {
                                if should_run {
                                    self.bus.memory.write_64(pointer, source_value as u64)?;
                                }
                            }
                            Size::System => {
                                if should_run {
                                    self.bus.memory.write_usize(pointer, source_value as usize)?;
                                }
                            }
                        }
                        instruction_pointer_offset += 8; // increment past 32 bit pointer
                    }
                    _ => panic!("Attempting to use an immediate value as a destination"),
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Movz(size, condition, destination, source) => {
                let (source_value, mut instruction_pointer_offset, isptr) = self.read_source_and_i_care_if_its_a_pointer(source)?;
                let should_run = self.check_condition(condition);
                match destination {
                    Operand::Register => {
                        let register = self.bus.memory.read_8(self.instruction_pointer + instruction_pointer_offset)?;
                        match size {
                            Size::Byte => {
                                if should_run {
                                    self.write_register_and_specify_if_its_a_pointer(register, (source_value & 0x000000FF) as u64, isptr);
                                }
                            }
                            Size::Half => {
                                if should_run {
                                    self.write_register_and_specify_if_its_a_pointer(register, (source_value & 0x0000FFFF) as u64, isptr);
                                }
                            }
                            Size::Word => {
                                if should_run {
                                    self.write_register_and_specify_if_its_a_pointer(register, source_value as u64, isptr);
                                }
                            }
                            Size::Long => {
                                if should_run {
                                    self.write_register_and_specify_if_its_a_pointer(register, source_value as u64, isptr);
                                }
                            }
                            Size::System => {
                                if should_run {
                                    self.write_register_and_specify_if_its_a_pointer(register, source_value as u64, isptr);
                                }
                            }
                        }
                        instruction_pointer_offset += 1; // increment past 8 bit register number
                    }
                    _ => panic!("MOVZ only operates on registers"),
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }

            Instruction::Jmp(condition, source) => {
                let (source_value, instruction_pointer_offset) = self.read_source(source)?;
                let should_run = self.check_condition(condition);
                if should_run {
                    Some(source_value)
                } else {
                    Some(self.instruction_pointer + instruction_pointer_offset)
                }
            }
            Instruction::Call(condition, source) => {
                let (source_value, instruction_pointer_offset) = self.read_source(source)?;
                let mut lib_man = interop::LIBRARYMANAGER.lock().unwrap();
                let func_addrs = lib_man.function_addresses.lock().unwrap();
                if let Some(l) = func_addrs.get(&(source_value as usize)) {
                    if self.debug {
                        println!("Calling C interop function at {:X}", source_value);
                    }
                    let layout = &lib_man.libraries.get(*l).unwrap().functions.get(&(source_value as usize)).unwrap().layout;
                    let mut args = Vec::new();
                    for (i, arg) in layout.iter().enumerate() {
                        // first 8 arguments are from registers r0-r7
                        // any other arguments are from stack
                        if i <= 7 {
                            match arg {
                                Argument::u8(_) => {
                                    args.push(Argument::u8(self.register[i] as u8));
                                }
                                Argument::u16(_) => {
                                    args.push(Argument::u16(self.register[i] as u16));
                                }
                                Argument::u32(_) => {
                                    args.push(Argument::u32(self.register[i] as u32));
                                }
                                Argument::u64(_) => {
                                    args.push(Argument::u64(self.register[i] as u64));
                                }
                                Argument::usize(_) => {
                                    args.push(Argument::usize(self.register[i] as usize));
                                }
                                Argument::ptr(_) => {
                                    args.push(Argument::ptr(self.bus.memory.get_actual_pointer(self.register[i]).unwrap()));
                                }
                            }
                        } else {
                            // pop from stack
                            match arg {
                                Argument::u8(_) => {
                                    args.push(Argument::u8(self.pop_stack_8()?));
                                }
                                Argument::u16(_) => {
                                    args.push(Argument::u16(self.pop_stack_16()?));
                                }
                                Argument::u32(_) => {
                                    args.push(Argument::u32(self.pop_stack_32()?));
                                }
                                Argument::u64(_) => {
                                    args.push(Argument::u64(self.pop_stack_64()?));
                                }
                                Argument::usize(_) => {
                                    args.push(Argument::usize(self.pop_stack_usize()?));
                                }
                                Argument::ptr(_) => {
                                    let word = self.pop_stack_64()?;
                                    args.push(Argument::ptr(self.bus.memory.get_actual_pointer(word).unwrap()));
                                }
                            }
                        }
                    }
                    let should_run = self.check_condition(condition);
                    if should_run {
                        let result = lib_man.libraries[*l].attempt_call(source_value as usize, args);
                        if let Err(e) = result {
                            println!("CINTEROP: error calling C interop function: {:?}", e);
                        } else {
                            self.write_register(0, result.unwrap() as u64);
                        }
                    }
                    Some(self.instruction_pointer + instruction_pointer_offset)
                } else {
                    drop(func_addrs);
                    drop(lib_man);
                    if self.debug {
                        println!("Calling {:08X}", source_value);
                    }
                    let should_run = self.check_condition(condition);
                    if should_run {
                        self.push_stack_64((self.instruction_pointer + instruction_pointer_offset) as u64);
                        Some(source_value)
                    } else {
                        Some(self.instruction_pointer + instruction_pointer_offset)
                    }
                }
            }
            Instruction::Loop(condition, source) => {
                let (source_value, instruction_pointer_offset) = self.read_source(source)?;
                let should_run = self.check_condition(condition);
                let result = self.read_register(31).overflowing_sub(1);
                self.write_register(31, result.0);
                if should_run {
                    if result.0 != 0 {
                        Some(source_value)
                    } else {
                        Some(self.instruction_pointer + instruction_pointer_offset)
                    }
                } else {
                    Some(self.instruction_pointer + instruction_pointer_offset)
                }
            }
            Instruction::Rjmp(condition, source) => {
                let (source_value, instruction_pointer_offset) = self.read_source(source)?;
                let should_run = self.check_condition(condition);
                if should_run {
                    Some(self.relative_to_absolute(source_value))
                } else {
                    Some(self.instruction_pointer + instruction_pointer_offset)
                }
            }
            Instruction::Rcall(condition, source) => {
                let (source_value, instruction_pointer_offset) = self.read_source(source)?;
                let should_run = self.check_condition(condition);
                if should_run {
                    self.push_stack_64((self.instruction_pointer + instruction_pointer_offset) as u64);
                    Some(self.relative_to_absolute(source_value))
                } else {
                    Some(self.instruction_pointer + instruction_pointer_offset)
                }
            }
            Instruction::Rloop(condition, source) => {
                let (source_value, instruction_pointer_offset) = self.read_source(source)?;
                let should_run = self.check_condition(condition);
                let result = self.read_register(31).overflowing_sub(1);
                self.write_register(31, result.0);
                if should_run {
                    if result.0 != 0 {
                        Some(self.relative_to_absolute(source_value))
                    } else {
                        Some(self.instruction_pointer + instruction_pointer_offset)
                    }
                } else {
                    Some(self.instruction_pointer + instruction_pointer_offset)
                }
            }

            Instruction::Rta(condition, destination, source) => {
                let (source_value, mut instruction_pointer_offset) = self.read_source(source)?;
                let should_run = self.check_condition(condition);
                match destination {
                    Operand::Register => {
                        let register = self.bus.memory.read_8(self.instruction_pointer + instruction_pointer_offset)?;
                        if should_run {
                            self.write_register(register, self.relative_to_absolute(source_value) as u64);
                        }
                        instruction_pointer_offset += 1; // increment past 8 bit register number
                    }
                    Operand::RegisterPtr(_) => {
                        let register = self.bus.memory.read_8(self.instruction_pointer + instruction_pointer_offset)?;
                        let pointer = self.relative_to_absolute(self.read_register(register));
                        if should_run {
                            // INFO: register contains a relative address instead of an absolute address
                            self.bus.memory.write_32(pointer, self.relative_to_absolute(source_value) as u32)?;
                        }
                        instruction_pointer_offset += 1; // increment past 8 bit register number
                    }
                    Operand::ImmediatePtr(_) => {
                        let word = self.bus.memory.read_64(self.instruction_pointer + instruction_pointer_offset)?;
                        let pointer = self.relative_to_absolute(word);
                        if should_run {
                            self.bus.memory.write_32(pointer, self.relative_to_absolute(source_value) as u32)?;
                        }
                        instruction_pointer_offset += 8; // increment past 32 bit pointer
                    }
                    _ => panic!("Attempting to use an immediate value as a destination"),
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }

            Instruction::Push(size, condition, source) => {
                let (source_value, instruction_pointer_offset) = self.read_source(source)?;
                let should_run = self.check_condition(condition);
                match size {
                    Size::Byte => {
                        if should_run {
                            self.push_stack_8(source_value as u8);
                        }
                    }
                    Size::Half => {
                        if should_run {
                            self.push_stack_16(source_value as u16);
                        }
                    }
                    Size::Word => {
                        if should_run {
                            self.push_stack_32(source_value as u32);
                        }
                    }
                    Size::Long => {
                        if should_run {
                            self.push_stack_64(source_value as u64);
                        }
                    }
                    Size::System => {
                        if should_run {
                            self.push_stack_usize(source_value as usize);
                        }
                    }
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Pop(size, condition, source) => {
                let mut instruction_pointer_offset = 2; // increment past opcode half
                let should_run = self.check_condition(condition);
                match source {
                    Operand::Register => {
                        let register = self.bus.memory.read_8(self.instruction_pointer + instruction_pointer_offset)?;
                        match size {
                            Size::Byte => {
                                if should_run {
                                    let value = self.pop_stack_8()? as u64;
                                    self.write_register(register, (self.read_register(register) & 0xFFFFFF00) | value);
                                }
                            }
                            Size::Half => {
                                if should_run {
                                    let value = self.pop_stack_16()? as u64;
                                    self.write_register(register, (self.read_register(register) & 0xFFFF0000) | value);
                                }
                            }
                            Size::Word => {
                                if should_run {
                                    let value = self.pop_stack_32()? as u64;
                                    self.write_register(register, value);
                                }
                            }
                            Size::Long => {
                                if should_run {
                                    let value = self.pop_stack_64()?;
                                    self.write_register(register, value);
                                }
                            }
                            Size::System => {
                                if should_run {
                                    let value = self.pop_stack_usize()? as u64;
                                    self.write_register(register, value);
                                }
                            }
                        }
                        instruction_pointer_offset += 1; // increment past 8 bit register number
                    }
                    Operand::RegisterPtr(_) => {
                        let register = self.bus.memory.read_8(self.instruction_pointer + instruction_pointer_offset)?;
                        let pointer = self.read_register(register);
                        match size {
                            Size::Byte => {
                                if should_run {
                                    let value = self.pop_stack_8()?;
                                    let success = self.bus.memory.write_8(pointer, value);
                                    if let None = success {
                                        self.stack_pointer = self.stack_pointer.overflowing_sub(1).0;
                                    }
                                }
                            }
                            Size::Half => {
                                if should_run {
                                    let value = self.pop_stack_16()?;
                                    let success = self.bus.memory.write_16(pointer, value);
                                    if let None = success {
                                        self.stack_pointer = self.stack_pointer.overflowing_sub(2).0;
                                    }
                                }
                            }
                            Size::Word => {
                                if should_run {
                                    let value = self.pop_stack_32()?;
                                    let success = self.bus.memory.write_32(pointer, value);
                                    if let None = success {
                                        self.stack_pointer = self.stack_pointer.overflowing_sub(4).0;
                                    }
                                }
                            }
                            Size::Long => {
                                if should_run {
                                    let value = self.pop_stack_64()?;
                                    let success = self.bus.memory.write_64(pointer, value);
                                    if let None = success {
                                        self.stack_pointer = self.stack_pointer.overflowing_sub(8).0;
                                    }
                                }
                            }
                            Size::System => {
                                if should_run {
                                    let value = self.pop_stack_usize()?;
                                    let success = self.bus.memory.write_usize(pointer, value);
                                    if let None = success {
                                        self.stack_pointer = self.stack_pointer.overflowing_sub(std::mem::size_of::<usize>() as u64).0;
                                    }
                                }
                            }
                        }
                        instruction_pointer_offset += 1; // increment past 8 bit register number
                    }
                    Operand::ImmediatePtr(_) => {
                        let pointer = self.bus.memory.read_64(self.instruction_pointer + instruction_pointer_offset)?;
                        match size {
                            Size::Byte => {
                                if should_run {
                                    let value = self.pop_stack_8()?;
                                    let success = self.bus.memory.write_8(pointer, value);
                                    if let None = success {
                                        self.stack_pointer = self.stack_pointer.overflowing_sub(1).0;
                                    }
                                }
                            }
                            Size::Half => {
                                if should_run {
                                    let value = self.pop_stack_16()?;
                                    let success = self.bus.memory.write_16(pointer, value);
                                    if let None = success {
                                        self.stack_pointer = self.stack_pointer.overflowing_sub(2).0;
                                    }
                                }
                            }
                            Size::Word => {
                                if should_run {
                                    let value = self.pop_stack_32()?;
                                    let success = self.bus.memory.write_32(pointer, value);
                                    if let None = success {
                                        self.stack_pointer = self.stack_pointer.overflowing_sub(4).0;
                                    }
                                }
                            }
                            Size::Long => {
                                if should_run {
                                    let value = self.pop_stack_64()?;
                                    let success = self.bus.memory.write_64(pointer, value);
                                    if let None = success {
                                        self.stack_pointer = self.stack_pointer.overflowing_sub(8).0;
                                    }
                                }
                            }
                            Size::System => {
                                if should_run {
                                    let value = self.pop_stack_usize()? as u64;
                                    let success = self.bus.memory.write_64(pointer, value);
                                    if let None = success {
                                        self.stack_pointer = self.stack_pointer.overflowing_sub(8).0;
                                    }
                                }
                            }
                        }
                        instruction_pointer_offset += 8; // increment past 32 bit pointer
                    }
                    _ => panic!("Attempting to use an immediate value as a destination"),
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Ret(condition) => {
                let instruction_pointer_offset = 2; // increment past opcode half
                let should_run = self.check_condition(condition);
                if should_run {
                    self.pop_stack_64()
                } else {
                    Some(self.instruction_pointer + instruction_pointer_offset)
                }
            }
            Instruction::Reti(condition) => {
                let instruction_pointer_offset = 2; // increment past opcode half
                let should_run = self.check_condition(condition);
                if should_run {
                    let flag = Flag::from(self.pop_stack_8()?);
                    let instruction_pointer = self.pop_stack_64()?;
                    self.flag = flag;
                    if self.flag.swap_sp {
                        self.stack_pointer = self.pop_stack_64()?;
                    }
                    Some(instruction_pointer)
                } else {
                    Some(self.instruction_pointer + instruction_pointer_offset)
                }
            }

            Instruction::In(condition, destination, source) => {
                unimplemented!("in instruction not implemented for xfox");
                /*
                let (source_value, mut instruction_pointer_offset) = self.read_source(source)?;
                let should_run = self.check_condition(condition);
                match destination {
                    Operand::Register => {
                        let register = self.bus.memory.read_8(self.instruction_pointer + instruction_pointer_offset)?;
                        let value = self.bus.read_io(source_value);
                        instruction_pointer_offset += 1; // increment past 8 bit register number
                        if should_run {
                            self.write_register(register, value);
                        }
                    }
                    Operand::RegisterPtr(_) => {
                        let register = self.bus.memory.read_8(self.instruction_pointer + instruction_pointer_offset)?;
                        let pointer = self.read_register(register);
                        let value = self.bus.read_io(source_value);
                        instruction_pointer_offset += 1; // increment past 8 bit register number
                        if should_run {
                            self.bus.memory.write_32(pointer, value)?;
                        }
                    }
                    Operand::ImmediatePtr(_) => {
                        let pointer = self.bus.memory.read_32(self.instruction_pointer + instruction_pointer_offset)?;
                        let value = self.bus.read_io(source_value);
                        instruction_pointer_offset += 4; // increment past 32 bit pointer
                        if should_run {
                            self.bus.memory.write_32(pointer, value)?;
                        }
                    }
                    _ => panic!("Attempting to use an immediate value as a destination"),
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
                 */
            }
            Instruction::Out(condition, destination, source) => {
                unimplemented!("out instruction not implemented for xfox");
                /*
                let (source_value, mut instruction_pointer_offset) = self.read_source(source)?;
                let should_run = self.check_condition(condition);
                match destination {
                    Operand::Register => {
                        let register = self.bus.memory.read_8(self.instruction_pointer + instruction_pointer_offset)?;
                        instruction_pointer_offset += 1; // increment past 8 bit register number
                        if should_run {
                            self.bus.write_io(self.read_register(register), source_value);
                        }
                    }
                    Operand::RegisterPtr(_) => {
                        let register = self.bus.memory.read_8(self.instruction_pointer + instruction_pointer_offset)?;
                        let pointer = self.read_register(register);
                        instruction_pointer_offset += 1; // increment past 8 bit register number
                        if should_run {
                            let word = self.bus.memory.read_32(pointer)?;
                            self.bus.write_io(word, source_value);
                        }
                    }
                    Operand::ImmediatePtr(_) => {
                        let pointer = self.bus.memory.read_32(self.instruction_pointer + instruction_pointer_offset)?;
                        instruction_pointer_offset += 4; // increment past 32 bit pointer
                        if should_run {
                            let word = self.bus.memory.read_32(pointer)?;
                            self.bus.write_io(word, source_value);
                        }
                    }
                    _ => panic!("Attempting to use an immediate value as a destination"),
                }
                Some(self.instruction_pointer + instruction_pointer_offset)

                 */
            }

            Instruction::Ise(condition) => {
                let instruction_pointer_offset = 2; // increment past opcode half
                let should_run = self.check_condition(condition);
                if should_run {
                    self.flag.interrupt = true;
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Icl(condition) => {
                let instruction_pointer_offset = 2; // increment past opcode half
                let should_run = self.check_condition(condition);
                if should_run {
                    self.flag.interrupt = false;
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Int(condition, source) => {
                let (source_value, instruction_pointer_offset) = self.read_source(source)?;
                let should_run = self.check_condition(condition);
                if should_run {
                    self.next_soft_interrupt = Some(source_value as u8);
                    Some(self.instruction_pointer + instruction_pointer_offset)
                } else {
                    Some(self.instruction_pointer + instruction_pointer_offset)
                }
            }

            Instruction::Mse(condition) => {
                let instruction_pointer_offset = 2; // increment past opcode half
                let should_run = self.check_condition(condition);
                if should_run {
                    self.bus.memory.write_mmu_enabled(true);
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Mcl(condition) => {
                let instruction_pointer_offset = 2; // increment past opcode half
                let should_run = self.check_condition(condition);
                if should_run {
                    self.bus.memory.write_mmu_enabled(false);
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Tlb(condition, source) => {
                let (source_value, instruction_pointer_offset) = self.read_source(source)?;
                let should_run = self.check_condition(condition);
                if should_run {
                    self.bus.memory.flush_tlb(Some(source_value));
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Flp(condition, source) => {
                let (source_value, instruction_pointer_offset) = self.read_source(source)?;
                let should_run = self.check_condition(condition);
                if should_run {
                    self.bus.memory.flush_page(source_value);
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Ldl(condition, destination, source) => {
                let (source_value, mut instruction_pointer_offset) = self.read_source(source)?;
                let should_run = self.check_condition(condition);
                match destination {
                    Operand::Register => {
                        let register = self.bus.memory.read_8(self.instruction_pointer + instruction_pointer_offset)?;
                        instruction_pointer_offset += 1; // increment past 8 bit register number
                        if should_run {
                            let res = crate::interop::LIBRARYMANAGER.lock().unwrap().load_dynamic_library(self.bus.memory.read_cstring(source_value)?);
                            if let Err(e) = res {
                                panic!("CINTEROP: failed to load dynamic library: {:?}", e);
                            } else {
                                self.write_register(register, res.unwrap() as u64);
                            }
                        }
                    }
                    Operand::RegisterPtr(_) => {
                        let register = self.bus.memory.read_8(self.instruction_pointer + instruction_pointer_offset)?;
                        let pointer = self.read_register(register);
                        instruction_pointer_offset += 1; // increment past 8 bit register number
                        if should_run {
                            let word = self.bus.memory.read_64(pointer)?;
                            let res = crate::interop::LIBRARYMANAGER.lock().unwrap().load_dynamic_library(self.bus.memory.read_cstring(source_value)?);
                            if let Err(e) = res {
                                panic!("CINTEROP: failed to load dynamic library: {:?}", e);
                            } else {
                                self.bus.memory.write_32(word, res.unwrap())?;
                            }
                        }
                    }
                    Operand::ImmediatePtr(_) => {
                        let pointer = self.bus.memory.read_64(self.instruction_pointer + instruction_pointer_offset)?;
                        instruction_pointer_offset += 8; // increment past 32 bit pointer
                        if should_run {
                            let word = self.bus.memory.read_64(pointer)?;
                            let res = crate::interop::LIBRARYMANAGER.lock().unwrap().load_dynamic_library(self.bus.memory.read_cstring(source_value)?);
                            if let Err(e) = res {
                                panic!("CINTEROP: failed to load dynamic library: {:?}", e);
                            } else {
                                self.bus.memory.write_32(word, res.unwrap())?;
                            }
                        }
                    }
                    _ => panic!("Attempting to use an immediate value as a destination"),
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Bind(condition, destination, source) => {
                let (source_value, mut instruction_pointer_offset) = self.read_source(source)?;
                let should_run = self.check_condition(condition);
                match destination {
                    Operand::Register => {
                        let register = self.bus.memory.read_8(self.instruction_pointer + instruction_pointer_offset)?;
                        instruction_pointer_offset += 1; // increment past 8 bit register number
                        if should_run {
                            let final_destination = self.read_register(register);
                            let res = crate::interop::LIBRARYMANAGER.lock().unwrap().libraries.get_mut(source_value as usize).unwrap()
                                .load_func(self.bus.memory.read_cstring(source_value)?, final_destination as usize);
                            if let Err(e) = res {
                                panic!("CINTEROP: failed to load dynamic library function: {:?}", e);
                            }
                        }
                    }
                    Operand::RegisterPtr(_) => {
                        let register = self.bus.memory.read_8(self.instruction_pointer + instruction_pointer_offset)?;
                        let pointer = self.read_register(register);
                        instruction_pointer_offset += 1; // increment past 8 bit register number
                        if should_run {
                            let word = self.bus.memory.read_64(pointer)?;
                            let res = crate::interop::LIBRARYMANAGER.lock().unwrap().libraries.get_mut(source_value as usize).unwrap()
                                .load_func(self.bus.memory.read_cstring(source_value)?, word as usize);
                            if let Err(e) = res {
                                panic!("CINTEROP: failed to load dynamic library function: {:?}", e);
                            }
                        }
                    }
                    Operand::ImmediatePtr(_) => {
                        let pointer = self.bus.memory.read_64(self.instruction_pointer + instruction_pointer_offset)?;
                        instruction_pointer_offset += 8; // increment past 32 bit pointer
                        if should_run {
                            let res = crate::interop::LIBRARYMANAGER.lock().unwrap().libraries.get_mut(source_value as usize).unwrap()
                                .load_func(self.bus.memory.read_cstring(pointer)?, pointer as usize);
                            if let Err(e) = res {
                                panic!("CINTEROP: failed to load dynamic library function: {:?}", e);
                            }
                        }
                    }
                    _ => panic!("Attempting to use an immediate value as a destination"),
                }
                Some(self.instruction_pointer + instruction_pointer_offset)
            }
            Instruction::Rse(condition) => {
                let should_run = self.check_condition(condition);
                if should_run {
                    self.flag.real_mode.store(true, Ordering::SeqCst);
                }
                Some(self.instruction_pointer + 2)
            }
            Instruction::Rcl(condition) => {
                let should_run = self.check_condition(condition);
                if should_run {
                    self.flag.real_mode.store(false, Ordering::SeqCst);
                }
                Some(self.instruction_pointer + 2)
            }
        }
    }
}

#[derive(Debug)]
pub enum Operand {
    Register,
    RegisterPtr(Size),
    Immediate8,
    Immediate16,
    Immediate32,
    Immediate64,
    Immediate64Fit,
    ImmediatePtr(Size),
}

#[derive(Copy, Clone, Debug)]
pub enum Size {
    Byte,
    Half,
    Word,
    Long,
    System,
}

#[derive(Debug)]
enum Condition {
    Always,
    Zero,
    NotZero,
    Carry,
    NotCarry,
    GreaterThan,
    // GreaterThanEqualTo is equivalent to NotCarry
    // LessThan is equivalent to Carry
    LessThanEqualTo,
}

#[derive(Debug)]
enum Instruction {
    Nop(),
    Halt(Condition),
    Brk(Condition),

    Add(Size, Condition, Operand, Operand),
    Inc(Size, Condition, Operand),
    Sub(Size, Condition, Operand, Operand),
    Dec(Size, Condition, Operand),

    Mul(Size, Condition, Operand, Operand),
    Div(Size, Condition, Operand, Operand),
    Rem(Size, Condition, Operand, Operand),

    And(Size, Condition, Operand, Operand),
    Or(Size, Condition, Operand, Operand),
    Xor(Size, Condition, Operand, Operand),
    Not(Size, Condition, Operand),

    Sla(Size, Condition, Operand, Operand),
    Rol(Size, Condition, Operand, Operand),

    Sra(Size, Condition, Operand, Operand),
    Srl(Size, Condition, Operand, Operand),
    Ror(Size, Condition, Operand, Operand),

    Bse(Size, Condition, Operand, Operand),
    Bcl(Size, Condition, Operand, Operand),
    Bts(Size, Condition, Operand, Operand),

    Cmp(Size, Condition, Operand, Operand),
    Mov(Size, Condition, Operand, Operand),
    Movz(Size, Condition, Operand, Operand),

    Jmp(Condition, Operand),
    Call(Condition, Operand),
    Loop(Condition, Operand),

    Rjmp(Condition, Operand),
    Rcall(Condition, Operand),
    Rloop(Condition, Operand),

    Rta(Condition, Operand, Operand),

    Push(Size, Condition, Operand),
    Pop(Size, Condition, Operand),
    Ret(Condition),
    Reti(Condition),

    In(Condition, Operand, Operand),
    Out(Condition, Operand, Operand),

    Ise(Condition),
    Icl(Condition),
    Int(Condition, Operand),

    Mse(Condition),
    Mcl(Condition),
    Tlb(Condition, Operand),
    Flp(Condition, Operand),

    // C interop stuff
    Ldl(Condition, Operand, Operand),
    Bind(Condition, Operand, Operand),
    Rse(Condition),
    Rcl(Condition),
}

impl Instruction {
    fn from_half(half: u16) -> Option<Instruction> {
        // see encoding.md for more info

        // first two bits, 0b1100_0000_0000_0000
        let size = match half & 0b1100_0000_0000_0000 {
            0b0000_0000_0000_0000 => Size::Byte,
            0b0100_0000_0000_0000 => Size::Half,
            0b1000_0000_0000_0000 => Size::Word,
            0b1100_0000_0000_0000 => Size::Long,
            _ => return None,
        };
        let opcode = ((half >> 8) as u8) & 0b00111111;
        let source = match half & 0b0000_0000_0000_0011 {
            0b0000_0000_0000_0000 => Operand::Register,
            0b0000_0000_0000_0001 => Operand::RegisterPtr(size),
            0b0000_0000_0000_0010 => match size {
                Size::Byte => Operand::Immediate8,
                Size::Half => Operand::Immediate16,
                Size::Word => Operand::Immediate32,
                Size::Long => Operand::Immediate64,
                Size::System => Operand::Immediate64Fit,
            },
            0b0000_0000_0000_0011 => Operand::ImmediatePtr(size),
            _ => return None,
        };
        let destination = match half & 0b0000_0000_0000_1100 {
            0b0000_0000_0000_0000 => Operand::Register,
            0b0000_0000_0000_0100 => Operand::RegisterPtr(size),
            // 0x02 is invalid, can't use an immediate value as a destination
            0b0000_0000_0000_1100 => Operand::ImmediatePtr(size),
            _ => return None,
        };
        let condition = match (half & 0x00F0) as u8 {
            0x00 => Condition::Always,
            0x10 => Condition::Zero,
            0x20 => Condition::NotZero,
            0x30 => Condition::Carry,
            0x40 => Condition::NotCarry,
            0x50 => Condition::GreaterThan,
            0x60 => Condition::LessThanEqualTo,
            _ => return None,
        };
        match opcode {
            0x00 => Some(Instruction::Nop()),
            0x10 => Some(Instruction::Halt(condition)),
            0x20 => Some(Instruction::Brk(condition)),

            0x01 => Some(Instruction::Add(size, condition, destination, source)),
            0x11 => Some(Instruction::Inc(size, condition, source)),
            0x21 => Some(Instruction::Sub(size, condition, destination, source)),
            0x31 => Some(Instruction::Dec(size, condition, source)),

            0x02 => Some(Instruction::Mul(size, condition, destination, source)),
            0x22 => Some(Instruction::Div(size, condition, destination, source)),
            0x32 => Some(Instruction::Rem(size, condition, destination, source)),

            0x03 => Some(Instruction::And(size, condition, destination, source)),
            0x13 => Some(Instruction::Or(size, condition, destination, source)),
            0x23 => Some(Instruction::Xor(size, condition, destination, source)),
            0x33 => Some(Instruction::Not(size, condition, source)),

            0x04 => Some(Instruction::Sla(size, condition, destination, source)),
            0x24 => Some(Instruction::Rol(size, condition, destination, source)),

            0x05 => Some(Instruction::Sra(size, condition, destination, source)),
            0x15 => Some(Instruction::Srl(size, condition, destination, source)),
            0x25 => Some(Instruction::Ror(size, condition, destination, source)),

            0x06 => Some(Instruction::Bse(size, condition, destination, source)),
            0x16 => Some(Instruction::Bcl(size, condition, destination, source)),
            0x26 => Some(Instruction::Bts(size, condition, destination, source)),

            0x07 => Some(Instruction::Cmp(size, condition, destination, source)),
            0x17 => Some(Instruction::Mov(size, condition, destination, source)),
            0x27 => Some(Instruction::Movz(size, condition, destination, source)),

            0x08 => Some(Instruction::Jmp(condition, source)),
            0x18 => Some(Instruction::Call(condition, source)),
            0x28 => Some(Instruction::Loop(condition, source)),

            0x09 => Some(Instruction::Rjmp(condition, source)),
            0x19 => Some(Instruction::Rcall(condition, source)),
            0x29 => Some(Instruction::Rloop(condition, source)),

            0x39 => Some(Instruction::Rta(condition, destination, source)),

            0x0A => Some(Instruction::Push(size, condition, source)),
            0x1A => Some(Instruction::Pop(size, condition, source)),
            0x2A => Some(Instruction::Ret(condition)),
            0x3A => Some(Instruction::Reti(condition)),

            0x0B => Some(Instruction::In(condition, destination, source)),
            0x1B => Some(Instruction::Out(condition, destination, source)),

            0x0C => Some(Instruction::Ise(condition)),
            0x1C => Some(Instruction::Icl(condition)),
            0x2C => Some(Instruction::Int(condition, source)),

            0x0D => Some(Instruction::Mse(condition)),
            0x1D => Some(Instruction::Mcl(condition)),
            0x2D => Some(Instruction::Tlb(condition, source)),
            0x3D => Some(Instruction::Flp(condition, source)),

            0x0E => Some(Instruction::Ldl(condition, destination, source)),
            0x1E => Some(Instruction::Bind(condition, destination, source)),
            0x2E => Some(Instruction::Rse(condition)),
            0x3E => Some(Instruction::Rcl(condition)),

            _ => None,
        }
    }
}
