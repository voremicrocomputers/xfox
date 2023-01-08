// main.rs

pub mod memory;
pub mod audio;
pub mod bus;
pub mod cpu;
pub mod keyboard;
pub mod mouse;
pub mod disk;
pub mod optimisations;

use audio::AudioChannel;
use bus::Bus;
use cpu::{Cpu, Exception, Interrupt};
use keyboard::Keyboard;
use mouse::Mouse;
use disk::DiskController;
use memory::{MEMORY_RAM_START, MEMORY_ROM_START, MemoryRam, Memory};

use std::sync::{Arc, mpsc, Mutex};
use std::thread;
use std::process::exit;
use std::env;
use std::fs::{File, read};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::SystemTime;

use chrono::prelude::*;
use image;
use log::error;
use pixels::{Pixels, SurfaceTexture};
use winit::dpi::LogicalSize;
use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{WindowBuilder, Icon};
use winit_input_helper::WinitInputHelper;

const WIDTH: usize = 640;
const HEIGHT: usize = 480;

const FRAMEBUFFER_ADDRESS: usize = 0x02000000;

pub struct Display {
    background: Vec<u8>,
    overlays: Arc<Mutex<Vec<Overlay>>>,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct Overlay {
    enabled: bool,
    width: usize,
    height: usize,
    x: usize,
    y: usize,
    framebuffer_pointer: u32,
}

fn read_rom() -> Vec<u8> {
    /*read("fox32.rom").unwrap_or_else(|_| {
        read("../fox32rom/fox32.rom").unwrap_or_else(|_| {
            println!("fox32.rom file not found, using embedded ROM");
            include_bytes!("fox32.rom").to_vec()
        })
    })*/
    // collect env args
    let args = std::env::args().collect::<Vec<String>>();
    // check if there's an input file
    if args.len() < 2 {
        panic!("No input file specified");
    } else {
        // read the file
        println!("Reading file: {}", args[1]);
        read(&args[1]).unwrap_or_else(|_| {
            panic!("Input file not found");
        })
    }
}

pub fn error(message: &str) -> ! {
    println!("Error: {}", message);
    exit(1);
}

pub fn warn(message: &str) {
    println!("Warning: {}", message);
}

fn main() {
    let version_string = format!("fox32 {} ({})", env!("VERGEN_BUILD_SEMVER"), option_env!("VERGEN_GIT_SHA_SHORT").unwrap_or("unknown"));
    println!("{}", version_string);

    let args: Vec<String> = env::args().collect();

    let (exception_sender, exception_receiver) = mpsc::channel::<Exception>();

    let memory = Memory::new(read_rom().as_slice(), exception_sender);
    let mut bus = Bus {
        memory: memory.clone(),
        startup_time: Local::now().timestamp_millis(),
    };

    let memory_cpu = memory.clone();
    let memory_eventloop = memory.clone();

    let ram_size = memory_cpu.ram().size;
    let ram_bottom_address = MEMORY_RAM_START;
    let ram_top_address = ram_bottom_address + ram_size - 1;
    println!("RAM: {:.2} MiB mapped at physical {:#010X}-{:#010X}", ram_size / 1048576, ram_bottom_address, ram_top_address);

    let rom_size = memory_cpu.rom().size;
    let rom_bottom_address = MEMORY_ROM_START;
    let rom_top_address = rom_bottom_address + rom_size - 1;
    println!("ROM: {:.2} KiB mapped at physical {:#010X}-{:#010X}", rom_size / 1024, rom_bottom_address, rom_top_address);

    let mut cpu = Cpu::new(bus);

    let (interrupt_sender, interrupt_receiver) = mpsc::channel::<Interrupt>();

    //   let builder = thread::Builder::new().name("cpu".to_string());
//    builder.spawn({
//        move || {
    let start_time = SystemTime::now();
    let mut i = 0;
    let mut exit = false;
    loop {
        // print cpu instruction pointer
        while !cpu.halted {
            if unsafe { optimisations::EXCEPTION_TOGGLE.load(Ordering::Relaxed) } {
                unsafe { optimisations::EXCEPTION_TOGGLE.store(false, Ordering::Relaxed) };
                if let Ok(exception) = exception_receiver.try_recv() {
                    (cpu.next_exception, cpu.next_exception_operand) = cpu.exception_to_vector(exception);
                }
            } else if unsafe { optimisations::INTERRUPT_TOGGLE.load(Ordering::Relaxed) } {
                unsafe { optimisations::INTERRUPT_TOGGLE.store(false, Ordering::Relaxed) };
                if let Ok(interrupt) = interrupt_receiver.try_recv() {
                    if let Interrupt::Request(vector) = interrupt {
                        cpu.next_interrupt = Some(vector);
                    }
                }
            }
            if !cpu.execute_memory_instruction() {
                exit = true;
            }
            i += 1;
            if exit {
                // the rest of the VM has exited, stop the CPU thread
                break;
            }

        }
        if exit {
            // the rest of the VM has exited, stop the CPU thread
            break;
        }
        if !cpu.flag.interrupt {
            // the cpu was halted and interrupts are disabled
            // at this point, the cpu is dead and cannot resume, break out of the loop
            break;
        }
        if let Ok(interrupt) = interrupt_receiver.recv() {
            if let Interrupt::Request(vector) = interrupt {
                cpu.next_interrupt = Some(vector);
                cpu.halted = false;
            }
        } else {
            // sender is closed, break
            break;
        }
    }
    println!("CPU halted");
    let elapsed = start_time.elapsed().unwrap();
    println!("CPU execution time: {}.{:03} seconds", elapsed.as_secs(), elapsed.subsec_millis());
    println!("CPU executed {} instructions", i);
    //    }
//    }).unwrap();
}

impl Display {
    fn new() -> Self {
        Self {
            background: vec![0; (HEIGHT * WIDTH * 4) as usize],
            overlays: Arc::new(Mutex::new(vec![Overlay { enabled: false, width: 16, height: 16, x: 0, y: 0, framebuffer_pointer: 0 }; 32])),
        }
    }

    fn update(&mut self, ram: &MemoryRam) {
        let overlay_lock = self.overlays.lock().unwrap();

        for i in 0..(HEIGHT * WIDTH * 4) as usize {
            self.background[i] = ram[FRAMEBUFFER_ADDRESS + i];
        }

        for index in 0..=31 {
            if overlay_lock[index].enabled {
                blit_overlay(&mut self.background, &overlay_lock[index], ram);
            }
        }
    }

    fn draw(&self, frame: &mut [u8]) {
        for (i, pixel) in frame.chunks_exact_mut(4).enumerate() {
            //let x = (i % WIDTH as usize) as i16;
            //let y = (i / WIDTH as usize) as i16;

            let i = i * 4;

            let slice = &self.background[i..i + 4];
            pixel.copy_from_slice(slice);
        }
    }
}

// modified from https://github.com/parasyte/pixels/blob/main/examples/invaders/simple-invaders/src/sprites.rs
fn blit_overlay(framebuffer: &mut [u8], overlay: &Overlay, ram: &[u8]) {
    //assert!(overlay.x + overlay.width <= WIDTH);
    //assert!(overlay.y + overlay.height <= HEIGHT);

    let mut width = overlay.width * 4;
    let mut height = overlay.height;

    // FIXME: this is a hack, and it only allows overlays to go off-screen on the bottom and right sides
    //        it also completely fucks up the image on the right side :p
    if overlay.x + overlay.width > WIDTH {
        let difference = (overlay.x + overlay.width) - WIDTH;
        width = width - (difference * 4);
        //println!("width: {}, difference: {}", width, difference);
    }
    if overlay.y + overlay.height > HEIGHT {
        let difference = (overlay.y + overlay.height) - HEIGHT;
        height = height - difference;
        //println!("height: {}, difference: {}", height, difference);
    }

    let overlay_framebuffer = &ram[(overlay.framebuffer_pointer as usize)..((overlay.framebuffer_pointer + ((width as u32) * (height as u32))) as usize)];

    let mut overlay_framebuffer_index = 0;
    for y in 0..height {
        let index = overlay.x * 4 + overlay.y * WIDTH * 4 + y * WIDTH * 4;
        // merge overlay onto screen
        // this is such a dumb hack
        let mut zipped = framebuffer[index..index + width].iter_mut().zip(&overlay_framebuffer[overlay_framebuffer_index..overlay_framebuffer_index + width]);
        while let Some((left, &right)) = zipped.next() {
            let (left_0, &right_0) = (left, &right);
            let (left_1, &right_1) = zipped.next().unwrap();
            let (left_2, &right_2) = zipped.next().unwrap();
            let (left_3, &right_3) = zipped.next().unwrap();
            // ensure that the alpha byte is greater than zero, meaning that this pixel shouldn't be transparent
            if right_3 > 0 {
                *left_0 = right_0;
                *left_1 = right_1;
                *left_2 = right_2;
                *left_3 = right_3;
            }
        }
        overlay_framebuffer_index += width;
    }
}
