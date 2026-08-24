
use motif_finder::{NULL_CHAR, NUM_CHECKPOINT_FILES, NUM_RJ_STEPS, MAX_E_VAL, MAX_TF_NUM, ECOLI_FREQ};
use motif_finder::{PROPOSE_EXTEND, DIRICHLET_PWM, THRESH, NECESSARY_MOTIF_IMPROVEMENT};
use motif_finder::base::*;

use motif_finder::data_struct::*;

use gzp::{deflate::Mgzip, par::compress::{ParCompress, ParCompressBuilder}, syncz::{SyncZ, SyncZBuilder}, par::decompress::{ParDecompress, ParDecompressBuilder},ZWriter, Compression};


use log::warn;

use clap::{Parser, ValueEnum};

use rand::prelude::*;

use std::path::*;
use std::time::{Instant};
use std::env;
use std::fs::{File, create_dir_all};
use std::io::{Read, Write};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {

    /// Sets the input file from preprocessing for us to slim budding yeast data
    #[arg(short, long)]
    input: String,

    /// Sets the output file for your slimmed budding yeast data with only chrXVI
    #[arg(short, long)]
    small_output: String,

    /// Sets the output file for your slimmed budding yeast data with only chrI-IX
    #[arg(short, long)]
    large_output: String,


}



fn main() {


    let Cli { input, small_output, large_output} = Cli::parse();

    let mut data_file_handle : ParDecompress<Mgzip> = ParDecompressBuilder::new().from_reader( File::open(input.as_str()).expect("You initialization file must be valid for inference to work!"));

    let mut buffer: Vec<u8> = Vec::new();

    _ = data_file_handle.read_to_end(&mut buffer).expect("Something went wrong when reading the data input file!");

    let (mut total_data, _bytes): (AllData, usize) = bincode::serde::decode_from_slice(&buffer, bincode::config::standard()).expect("Something was incorrect with your saved data input bincode file!");

    let data_ref = AllDataUse::new(&total_data, 0.0).unwrap();


    let mut buffer: Vec<u8> = Vec::new();



    let loo_set = data_ref.retain_only_named_chrs(&["chrXVI"]).unwrap();

    let osb_set = data_ref.retain_only_named_chrs(&["chrI", "chrII","chrIII","chrIV","chrV","chrVI","chrVII","chrVIII","chrIX"]).unwrap();

    let loo_file_str_short = small_output;
    let loo_file_str = loo_file_str_short.clone();
    let osb_file_str_short = large_output;
    let osb_file_str = osb_file_str_short.clone(); 

    let mut loo_file_handle = SyncZBuilder::<Mgzip, _>::new().compression_level(Compression::new(9)).from_writer(File::create(loo_file_str).unwrap_or_else(|_| File::create(loo_file_str_short).unwrap()));
    buffer = bincode::serde::encode_to_vec(&loo_set, bincode::config::standard()).expect("Serializable");
    loo_file_handle.write(&buffer).expect("Just created this file");
    buffer.clear();

    let mut osb_file_handle = SyncZBuilder::<Mgzip, _>::new().compression_level(Compression::new(9)).from_writer(File::create(osb_file_str).unwrap_or_else(|_| File::create(osb_file_str_short).unwrap()));
    buffer = bincode::serde::encode_to_vec(&osb_set, bincode::config::standard()).expect("Serializable");
    osb_file_handle.write(&buffer).expect("Just created this file");
    buffer.clear();










}





