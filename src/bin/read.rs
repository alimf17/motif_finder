use motif_finder::data_struct::{AllData, AllDataUse};

use std::io::{Read};
use std::{env, fs};



fn main() {

    let args: Vec<String> = env::args().collect();

    let bincode_file = &args[1];

    println!("db {bincode_file}");
    
    let mut bincode_file_handle = fs::File::open(bincode_file).expect("Binarray file MUST be valid!");

    let mut buffer: Vec<u8> = Vec::new();

    bincode_file_handle.read_to_end(&mut buffer).expect("something went wrong reading the file");

    
    let (prior_state, _bytes): (AllData, usize) = bincode::serde::decode_from_slice(&buffer, bincode::config::standard()).expect("Binarray file MUST be a valid motif set!");

    let using: AllDataUse = AllDataUse::new(&prior_state, 0.0).unwrap();
    println!("{} {}", using.size(), using.number_bp());
    println!("Back: {:?}", using.background_ref());
}

