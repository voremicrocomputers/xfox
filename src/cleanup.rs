use crate::cpu;
use crate::cpu::{Operand, Size};

pub fn mdma_with_overflow(cpu: &mut cpu::Cpu, size: &Size, destination: cpu::Operand, instruction_pointer_offset: &mut u64, source_value: u64, fun: fn(a: u64, b: u64, bits: isize) -> (u64, bool), leave_carry_unset: bool) -> Option<(u64, bool)> {
    let mut fin_result = (0u64, false);
    match destination {
        Operand::Register => {
            let register = cpu.bus.memory.read_8(cpu.instruction_pointer + *instruction_pointer_offset).expect("failed to read instruction");
            match size {
                Size::Byte => {
                    let result = fun(cpu.read_register(register), source_value, 8);
                    cpu.write_register(register, (cpu.read_register(register) & 0xFFFFFF00) | (result.0 as u64));
                    cpu.flag.zero = result.0 == 0;
                    if !leave_carry_unset { cpu.flag.carry = result.1 };
                    fin_result.0 = result.0 as u64;
                    fin_result.1 = result.1;
                }
                Size::Half => {
                    let result = fun(cpu.read_register(register), source_value, 16);
                    cpu.write_register(register, (cpu.read_register(register) & 0xFFFF0000) | (result.0 as u64));
                    cpu.flag.zero = result.0 == 0;
                    if !leave_carry_unset { cpu.flag.carry = result.1 };
                    fin_result.0 = result.0 as u64;
                    fin_result.1 = result.1;
                }
                Size::Word => {
                    let result = fun(cpu.read_register(register), source_value, 32);
                    cpu.write_register(register, result.0 as u64);
                    cpu.flag.zero = result.0 == 0;
                    if !leave_carry_unset { cpu.flag.carry = result.1 };
                    fin_result.0 = result.0 as u64;
                    fin_result.1 = result.1;
                }
                Size::Long => {
                    let result = fun(cpu.read_register(register), source_value, 64);
                    cpu.write_register(register, result.0 as u64);
                    cpu.flag.zero = result.0 == 0;
                    if !leave_carry_unset { cpu.flag.carry = result.1 };
                    fin_result.0 = result.0 as u64;
                    fin_result.1 = result.1;
                }
                Size::System => {
                    let result = fun(cpu.read_register(register), source_value, -1);
                    cpu.write_register(register, result.0 as u64);
                    cpu.flag.zero = result.0 == 0;
                    if !leave_carry_unset { cpu.flag.carry = result.1 };
                    fin_result.0 = result.0 as u64;
                    fin_result.1 = result.1;
                }
            }
            *instruction_pointer_offset += 1; // increment past 8 bit register number
        }
        Operand::RegisterPtr(_) => {
            let register = cpu.bus.memory.read_8(cpu.instruction_pointer + *instruction_pointer_offset).expect("failed to read instruction");
            let pointer = cpu.read_register(register);
            match size {
                Size::Byte => {
                    let result = fun(cpu.bus.memory.read_8(pointer as u64)? as u64, source_value, 8);
                    cpu.bus.memory.write_8(pointer as u64, result.0 as u8)?;
                    cpu.flag.zero = result.0 == 0;
                    if !leave_carry_unset { cpu.flag.carry = result.1 };
                    fin_result.0 = result.0 as u64;
                    fin_result.1 = result.1;
                }
                Size::Half => {
                    let result = fun(cpu.bus.memory.read_16(pointer as u64)? as u64, source_value, 16);
                    cpu.bus.memory.write_16(pointer as u64, result.0 as u16)?;
                    cpu.flag.zero = result.0 == 0;
                    if !leave_carry_unset { cpu.flag.carry = result.1 };
                    fin_result.0 = result.0 as u64;
                    fin_result.1 = result.1;
                }
                Size::Word => {
                    let result = fun(cpu.bus.memory.read_32(pointer as u64)? as u64, source_value, 32);
                    cpu.bus.memory.write_32(pointer as u64, result.0 as u32)?;
                    cpu.flag.zero = result.0 == 0;
                    if !leave_carry_unset { cpu.flag.carry = result.1 };
                    fin_result.0 = result.0 as u64;
                    fin_result.1 = result.1;
                }
                Size::Long => {
                    let result = fun(cpu.bus.memory.read_64(pointer as u64)? as u64, source_value, 64);
                    cpu.bus.memory.write_64(pointer as u64, result.0 as u64)?;
                    cpu.flag.zero = result.0 == 0;
                    if !leave_carry_unset { cpu.flag.carry = result.1 };
                    fin_result.0 = result.0 as u64;
                    fin_result.1 = result.1;
                }
                Size::System => {
                    let result = fun(cpu.bus.memory.read_usize(pointer as u64)? as u64, source_value, -1);
                    cpu.bus.memory.write_usize(pointer as u64, result.0 as usize)?;
                    cpu.flag.zero = result.0 == 0;
                    if !leave_carry_unset { cpu.flag.carry = result.1 };
                    fin_result.0 = result.0 as u64;
                    fin_result.1 = result.1;
                }
            }
            *instruction_pointer_offset += 1; // increment past 8 bit register number
        }
        Operand::ImmediatePtr(_) => {
            let pointer = cpu.bus.memory.read_64(cpu.instruction_pointer + *instruction_pointer_offset)?;
            match size {
                Size::Byte => {
                    let result = fun(cpu.bus.memory.read_8(pointer)? as u64, source_value, 8);
                    cpu.bus.memory.write_8(pointer, result.0 as u8)?;
                    cpu.flag.zero = result.0 == 0;
                    if !leave_carry_unset { cpu.flag.carry = result.1 };
                    fin_result.0 = result.0 as u64;
                    fin_result.1 = result.1;
                }
                Size::Half => {
                    let result = fun(cpu.bus.memory.read_16(pointer)? as u64, source_value, 16);
                    cpu.bus.memory.write_16(pointer, result.0 as u16)?;
                    cpu.flag.zero = result.0 == 0;
                    if !leave_carry_unset { cpu.flag.carry = result.1 };
                    fin_result.0 = result.0 as u64;
                    fin_result.1 = result.1;
                }
                Size::Word => {
                    let result = fun(cpu.bus.memory.read_32(pointer)? as u64, source_value, 32);
                    cpu.bus.memory.write_32(pointer, result.0 as u32)?;
                    cpu.flag.zero = result.0 == 0;
                    if !leave_carry_unset { cpu.flag.carry = result.1 };
                    fin_result.0 = result.0 as u64;
                    fin_result.1 = result.1;
                }
                Size::Long => {
                    let result = fun(cpu.bus.memory.read_64(pointer)? as u64, source_value, 64);
                    cpu.bus.memory.write_64(pointer, result.0 as u64)?;
                    cpu.flag.zero = result.0 == 0;
                    if !leave_carry_unset { cpu.flag.carry = result.1 };
                    fin_result.0 = result.0 as u64;
                    fin_result.1 = result.1;
                }
                Size::System => {
                    let result = fun(cpu.bus.memory.read_usize(pointer)? as u64, source_value, -1);
                    cpu.bus.memory.write_usize(pointer, result.0 as usize)?;
                    cpu.flag.zero = result.0 == 0;
                    if !leave_carry_unset { cpu.flag.carry = result.1 };
                    fin_result.0 = result.0 as u64;
                    fin_result.1 = result.1;
                }
            }
            *instruction_pointer_offset += 8; // increment past 32 bit pointer
        }
        _ => panic!("Attempting to use an immediate value as a destination"),
    }
    Some(fin_result)
}