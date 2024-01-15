use motif_finder::data_struct::{AllData, AllDataUse};

use std::io::{Read, Write};
use std::{env, fs};

use serde::{Serialize, Deserialize};

fn main() {

    let args: Vec<String> = env::args().collect();

    let bincode_file = &args[1];

    println!("db {bincode_file}");
    
    let mut bincode_file_handle = fs::File::open(bincode_file).expect("Binarray file MUST be valid!");

    let mut buffer: Vec<u8> = Vec::new();

    bincode_file_handle.read_to_end(&mut buffer);

    
    let prior_state: AllData = bincode::deserialize(&buffer).expect("Binarray file MUST be a valid motif set!");

    let using: AllDataUse = AllDataUse::new(&prior_state).unwrap();
    println!("{} {}", using.size(), using.number_bp());
}

