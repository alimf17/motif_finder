use std::fs;
use std::io::Read;

use motif_finder::base::*;
use motif_finder::waveform::*;




use motif_finder::data_struct::{AllData, AllDataUse};

use statrs::distribution::Continuous;







//use rand::prelude::*;




use plotters::prelude::*;
use plotters::prelude::full_palette::ORANGE;

fn main() {

    let args: Vec<String> = std::env::args().collect();

    let file_out = args[1].clone();

    let mut try_bincode = fs::File::open(file_out).unwrap();

    let mut buffer: Vec<u8> = Vec::new();
    let _ = try_bincode.read_to_end(&mut buffer);//We don't need to handle this specially, because this will create a different warning later
    let (pre_data, _bytes): (AllData, usize) = bincode::serde::decode_from_slice(&buffer, bincode::config::standard()).unwrap();

    let data = AllDataUse::new(&pre_data, 0.0).unwrap();

    buffer.clear();

    let ranges = data.legal_coordinate_ranges();

    print!("[");
    for r in ranges {
        print!("{:?}, ", r);
    }
    print!("]");
}
