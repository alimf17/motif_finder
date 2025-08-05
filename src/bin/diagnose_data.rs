use gzp::{deflate::Mgzip, par::compress::{ParCompress, ParCompressBuilder}, syncz::{SyncZ, SyncZBuilder}, par::decompress::{ParDecompress, ParDecompressBuilder},ZWriter, Compression};

use clap::{Parser, ValueEnum};

use motif_finder::sequence::*;
use motif_finder::data_struct::*;

use std::path::*;
use std::time::{Instant};
use std::env;
use std::fs::File;
use std::io::Read;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {

    /// Sets the input file from preprocessing to check
    #[arg(short, long)]
    input: String,
}

fn main() {

    let  Cli {input: data_file} = Cli::parse();

    let mut data_file_handle: ParDecompress<Mgzip> = ParDecompressBuilder::new().from_reader( File::open(data_file.as_str()).expect("You initialization file must be valid for inference to work!"));

    let mut buffer: Vec<u8> = Vec::new();

    _ = data_file_handle.read_to_end(&mut buffer).expect("Something went wrong when reading the data input file!");

    let (mut total_data, _bytes): (AllData, usize) = bincode::serde::decode_from_slice(&buffer, bincode::config::standard()).expect("Something was incorrect with your saved data input bincode mgz file!");



    println!("have all data");

    let data_ref = AllDataUse::new(&total_data, 0.0).unwrap();

    println!("{:?}", data_ref.data().seq().diagnose_hamming_optimum());
}
