use motif_finder::{NULL_CHAR, NUM_CHECKPOINT_FILES, NUM_RJ_STEPS, MAX_E_VAL, MAX_TF_NUM, ECOLI_FREQ};
use motif_finder::{PROPOSE_EXTEND, DIRICHLET_PWM, THRESH, NECESSARY_MOTIF_IMPROVEMENT};
use motif_finder::base::*;

use motif_finder::data_struct::*;

use log::warn;

use gzp::{deflate::Mgzip, par::compress::{ParCompress, ParCompressBuilder}, syncz::{SyncZ, SyncZBuilder}, par::decompress::{ParDecompress, ParDecompressBuilder},ZWriter, Compression};

use clap::{Parser, ValueEnum};


use std::path::*;
use std::time::{Instant};
use std::env;
use std::fs::{File, create_dir_all};
use std::io::{Read, Write};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {

    /// Finds your output file
    #[arg(short, long)]
    name: String,

    /// Sets your new output file name
    #[arg(short, long)]
    output_name: String,

    /// Sets what your new min height will be
    #[arg(short, long)]
    min_height: Option<f64>,

    /// Sets what your new credibility will be
    #[arg(short, long)]
    credibility: Option<f64>,
}

fn main() {

    let Cli {name, output_name, min_height, credibility} = Cli::parse();

    match (min_height, credibility) {
        (None, None) => {
            println!("No changes made! Not making new AllData.");
            return;
        }
        (min_height, credibility) => {

            let mut data_file_handle: ParDecompress<Mgzip> = ParDecompressBuilder::new().from_reader( File::open(&name).expect("You initialization file must be valid for inference to work!"));

            let mut buffer: Vec<u8> = Vec::new();

            _ = data_file_handle.read_to_end(&mut buffer).expect("Something went wrong when reading the data input file!");

            let (mut total_data, _bytes): (AllData, usize) = bincode::serde::decode_from_slice(&buffer, bincode::config::standard()).expect("Something was incorrect with your saved data input bincode mgz file!");

            if let Some(height) = min_height { total_data.set_min_height(height) ;};
            if let Some(cred) = credibility { total_data.set_credibility(cred) ;};

            buffer.clear();

            buffer = bincode::serde::encode_to_vec(&total_data, bincode::config::standard()).expect("serializable");


            let mut outfile_handle = match File::create(output_name.clone()) {
                Err(_) => {
                    create_dir_all(output_name.clone()).unwrap();
                    File::create(output_name.clone()).unwrap()
                },
                Ok(o) => o
            };

            let mut parz = SyncZBuilder::<Mgzip, _>::new().compression_level(Compression::new(9)).from_writer(outfile_handle);
            parz.write_all(&buffer).expect("We just created this file");

            println!("Saved new preprocessing file to {output_name}.");

        }
    }
}
