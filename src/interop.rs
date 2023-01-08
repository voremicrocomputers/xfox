use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex};
use lazy_static::lazy_static;

lazy_static!{
    pub static ref LIBRARYMANAGER: Arc<Mutex<LibraryManager>> = Arc::new(Mutex::new(LibraryManager::new()));
}

// func_with_arg_types!(u8, u8, u16, *mut u8, none, none, none, none); -> extern "C" fn(u8, u8, u16, *mut u8) -> u8
macro_rules! func_with_arg_types {
    ($($arg:ty),*) => {
        extern "C" fn($($arg),*)
    }
}

#[derive(Debug, Copy, Clone)]
pub enum Argument {
    u8(u8),
    u16(u16),
    u32(u32),
    ptr(*mut u8),
}

impl Argument {
    pub fn v(&self) -> usize {
        match self {
            Argument::u8(v) => *v as usize,
            Argument::u16(v) => *v as usize,
            Argument::u32(v) => *v as usize,
            Argument::ptr(v) => *v as usize,
        }
    }
}

pub struct Function {
    pub internal_name: String,
    pub layout: Vec<Argument>,
    inner: *mut u8,
}

unsafe impl Send for Function {}
unsafe impl Sync for Function {}

pub struct Library {
    pub name: String,
    pub functions: BTreeMap<usize, Function>,
    pub function_addresses: Arc<Mutex<BTreeMap<usize, usize>>>, // address, library index
    index: usize,
    inner: libloading::Library,
}

pub struct LibraryManager {
    pub libraries: Vec<Library>,
    pub function_addresses: Arc<Mutex<BTreeMap<usize, usize>>>, // address, library index
}

#[derive(Debug)]
pub enum LibraryError {
    LibraryNotFound,
    LibraryAlreadyExists,
    FunctionNotFound,
    CannotRunOnOperatingSystem,
    CannotRunOnArchitecture,
}

impl Library {
    pub fn load_func(&mut self, name: String, dest: usize) -> Result<(), LibraryError> {
        let name_layout = name.split("(").collect::<Vec<&str>>();
        // remove last character, which is a closing bracket
        let name = name_layout[0].to_string();
        let layout = name_layout[1].trim_end_matches(")").split(",").collect::<Vec<&str>>();
        let mut layout_vec = Vec::new();

        for (i, arg) in layout.iter().enumerate() {
            if arg == &"u8" {
                layout_vec.push(Argument::u8(0));
            } else if arg == &"u16" {
                layout_vec.push(Argument::u16(0));
            } else if arg == &"u32" {
                layout_vec.push(Argument::u32(0));
            } else if arg == &"ptr" {
                layout_vec.push(Argument::ptr(0 as *mut u8));
            } else {
                return Err(LibraryError::FunctionNotFound);
            }
        }

        let func = unsafe {
            self.inner.get::<*mut u8>(name.as_bytes())
        };
        match func {
            Ok(func) => {
                let func = Function {
                    internal_name: name.to_string(),
                    inner: *func,
                    layout: layout_vec,
                };
                self.functions.insert(dest, func);
                self.function_addresses.lock().unwrap().insert(dest, self.index);
                Ok(())
            },
            Err(e) => {
                println!("CINTEROP: failed to load function: {}", e);
                Err(LibraryError::FunctionNotFound)
            },
        }
    }

    pub fn attempt_call(&self, address: usize, arguments: Vec<Argument>) -> Result<usize, LibraryError> {
        let func = self.functions.get(&address);
        match func {
            Some(func) => {
                let mut layout_expanded = Vec::new();
                layout_expanded.extend_from_slice(arguments.as_slice());
                if layout_expanded.len() < 8 {
                    for _ in 0..(8 - layout_expanded.len()) {
                        layout_expanded.push(Argument::u8(0));
                    }
                }
                println!("{:?} {:?}", layout_expanded, arguments);
                let func: extern "C" fn(_, ...) -> _ = unsafe { std::mem::transmute(func.inner) };
                let result = unsafe { func(
                    layout_expanded[0].v(),
                    layout_expanded[1].v(),
                    layout_expanded[2].v(),
                    layout_expanded[3].v(),
                    layout_expanded[4].v(),
                    layout_expanded[5].v(),
                    layout_expanded[6].v(),
                    layout_expanded[7].v())
                };
                println!("DEBUG C FUNC RESULT: {:?}", result);
                Ok(result)
            },
            None => Err(LibraryError::FunctionNotFound),
        }
    }
}

impl LibraryManager {
    pub fn new() -> LibraryManager {
        LibraryManager {
            libraries: Vec::new(),
            function_addresses: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    pub fn load_dynamic_library(&mut self, name: String) -> Result<u32, LibraryError> {
        if self.libraries.iter().any(|lib| lib.name == name) {
            return Err(LibraryError::LibraryAlreadyExists);
        }
        let full_name = if name.contains(".so") || name.contains(".dll") { // todo: this may need to be changed for further compatibility
            name.clone()
        } else {
            format!("{}.{}", name.clone(), if std::env::consts::OS == "windows" { "dll" } else { "so" })
        };
        let lib = unsafe { libloading::Library::new(full_name) };
        if let Err(e) = lib {
            panic!("Failed to load library: {}", e);
        }
        let lib = lib.unwrap();
        let functions = BTreeMap::new();
        self.libraries.push(Library {
            name,
            functions,
            function_addresses: self.function_addresses.clone(),
            inner: lib,
            index: self.libraries.len(),
        });
        Ok(self.libraries.len() as u32 - 1)
    }
}