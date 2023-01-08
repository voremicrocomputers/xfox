// bus.rs

use crate::{Memory, AudioChannel, DiskController, Keyboard, Mouse, Overlay};

use chrono::prelude::*;
use std::sync::{Arc, Mutex};
use std::io::{Write, stdout};

pub struct Bus {
    pub memory: Memory,

    pub startup_time: i64,
}

impl Bus {
    pub fn read_io(&mut self, port: u32) -> u32 {
        match port {
            0x80000000..=0x8000031F => { // overlay port
                unimplemented!("overlay port");
            }
            0x80000400..=0x80000401 => { // mouse port
                unimplemented!("mouse port");
            }
            0x80000500 => { // keyboard port
                unimplemented!("keyboard port");
            }
            0x80000600..=0x80000603 => { // audio port
                unimplemented!("audio port");
            }
            0x80000700..=0x80000706 => { // RTC port
                let setting = (port & 0x000000FF) as u8;
                match setting {
                    0 => { // year
                        Local::now().year() as u32
                    },
                    1 => { // month
                        Local::now().month()
                    },
                    2 => { // day
                        Local::now().day()
                    },
                    3 => { // hour
                        Local::now().hour()
                    },
                    4 => { // minute
                        Local::now().minute()
                    },
                    5 => { // second
                        Local::now().second()
                    },
                    6 => { // milliseconds elapsed since startup
                        (Local::now().timestamp_millis() - self.startup_time) as u32
                    },
                    _ => panic!("invalid RTC port"),
                }
            }
            0x80001000..=0x80002003 => { // disk controller port
                unimplemented!("disk controller port");
            }
            _ => 0,
        }
    }
    pub fn write_io(&mut self, port: u32, word: u32) {
        match port {
            0x00000000 => { // terminal output port
                print!("{}", word as u8 as char);
                stdout().flush().expect("could not flush stdout");
            }
            0x80000000..=0x8000031F => { // overlay port
                unimplemented!("overlay port");
            }
            0x80000400..=0x80000401 => { // mouse port
                unimplemented!("mouse port");
            }
            0x80000600..=0x80000603 => { // audio port
                unimplemented!("audio port");
            }
            0x80001000..=0x80005003 => { // disk controller port
                unimplemented!("disk controller port");
            }
            _ => (),
        }
    }
}
