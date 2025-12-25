use std::fs;
use std::io::{Read, Write};

use motif_finder::base::*;
use motif_finder::base::{SQRT_2, SQRT_3, LN_2, BPS};
use motif_finder::{NECESSARY_MOTIF_IMPROVEMENT, ECOLI_FREQ};
use motif_finder::data_struct::{AllData, AllDataUse};
use motif_finder::gene_loci::*;
use motif_finder::waveform::WaveOutputFile;
    
use clap::{Parser, ValueEnum};

use gzp::{deflate::Mgzip, par::compress::{ParCompress, ParCompressBuilder}, syncz::{SyncZ, SyncZBuilder}, par::decompress::{ParDecompress, ParDecompressBuilder},ZWriter, Compression};

use itertools::Itertools;

use log::warn;

use rustfft::{FftPlanner, num_complex::Complex};

use ndarray::prelude::*;

use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelIterator;

use rand::prelude::*;

use regex::Regex;

use poloto;
use poloto::Croppable;
use plotters::prelude::*;
use plotters::prelude::full_palette::*;
use plotters::coord::ranged1d::{NoDefaultFormatting, KeyPointHint,ValueFormatter};

use spectrum_analyzer::{samples_fft_to_spectrum, FrequencyLimit};

use core::ops::Range;


const INTERVAL_CELL_LENGTH: f64 = 0.01;

const UPPER_LETTERS: [char; 26] = [
    'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'I', 'J', 
    'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'];


const PALETTE: [&RGBColor; 26] = [
    &full_palette::RED, &full_palette::BLUE, &full_palette::GREEN, &YELLOW_700,
    &AMBER, &BLUEGREY, &full_palette::CYAN,  &ORANGE, 
    &PINK, &TEAL, &LIME, &full_palette::YELLOW, 
    &DEEPORANGE, &INDIGO, &CYAN_A700, &YELLOW_A700,
    &BROWN, &BLUEGREY_A700, &LIME_A700, &YELLOW_800, 
    &RED_A700, &BLUE_A700, &GREEN_A700, &YELLOW_A700, &GREY, &full_palette::BLACK];

const BASE_COLORS: [&RGBColor; BASE_L] = [&full_palette::RED, &YELLOW_700, &full_palette::GREEN, &full_palette::BLUE];

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {

    /// This is the directory your run output to 
    #[arg(short, long)]
    output: String,
    
    /// This is prefix your runs share. When doing this processing, we assume
    /// that all of your outputted bin files are of the form 
    /// "{output}/{base_file}_{<Letter starting from A and going up>}_
    /// {<0-indexed count of runs>}_trace_from_step_{<7 digit number, with 
    /// leading zeros>}.bin"
    #[arg(short, long)]
    base_file: String,

    /// This is the number of independent chains run on this inference.
    /// If this is larger than 26, we will only take the first 26 chains.
    #[arg(short, long)]
    num_chains: usize, 
    // Yeah, the 26 number is because of english letters in the alphabet.
    // But seriously, this should probably be like 2-6. We usually used 4.


    /// This is the number of sequential runs per chain. For example, 
    /// if the last output file of the initial run of chain "A" is 
    /// "{output}/{base_file}_A_0_trace_from_step_0100000.bin", it would
    /// be immediately followed by "{output}/{base_file}_A_1_trace_from_step_0000000.bin"
    #[arg(short, long)]
    max_runs: usize,

}

pub fn main() {

    let Cli { output: out_dir, base_file, mut num_chains, max_runs: max_chain} = Cli::parse(); 

    let min_chain = 0_usize; 

    if min_chain > max_chain {
        panic!("Discarding everything you should infer on!");
    }


    if num_chains > 26 {
        warn!("Can only sustain up to 26 independent chains. Only considering the first 26.");
        num_chains = 26;
    }

    let mut buffer: Vec<u8> = Vec::new();



    /*let tf_analyzer: Option<TFAnalyzer> = tf_file.map(|a| match TFAnalyzer::from_regulon_tsv(&a, 3, 6,7, None) {
      Ok(b) => Some(b),
      Err(e) => {println!("{e}"); None}
      }).or(None).flatten();
     */


    //println!("tf analyzer: \n {} \n {:?}",tf_analyzer.as_ref().map(|a| a.output_state()).unwrap_or(String::new()), tf_analyzer ); 
    //let max_max_length = 100000;
    //This is the code that actually sets up our independent chain reading
    let mut set_trace_collections: Vec<SetTraceDef> = Vec::with_capacity(max_chain-min_chain);
    for chain in 0..num_chains {
        let base_str = format!("{}/{}_{}", out_dir.clone(), base_file, UPPER_LETTERS[chain]);
        let chain_file = base_str.clone()+format!("_{}_trace", min_chain).as_str()+".bin.gz";

        let len_file = base_str.clone()+format!("_{min_chain}_bits.txt").as_str();

        let skip_some: bool = if min_chain == 0 { true } else {false};

        //fs::File::open(file_y.expect("Must have matches to proceed")).expect("We got this from a list of directory files").read_to_end(&mut buffer).expect("something went wrong reading the file");

        let mut string_buff: String = String::new();
        fs::File::open(&len_file).unwrap().read_to_string(&mut string_buff);
        let mut buffer_lens: Vec<usize> = string_buff.split_terminator('\n').map(|a| {
            a.parse::<usize>().expect("This whole file should be unsigned integers of buffer lengths")
        }).collect();

        let mut read_file: ParDecompress<Mgzip> = ParDecompressBuilder::new().from_reader(fs::File::open(chain_file).unwrap());

        if skip_some {
            for i in 0..10 {
                let mut handle = read_file.by_ref().take(buffer_lens[i] as u64);
                handle.read_to_end(&mut buffer).unwrap();
            }

            let _: Vec<_> = buffer_lens.drain(0..10).collect();
        }

        let mut handle = read_file.by_ref().take(buffer_lens[0] as u64);

        handle.read_to_end(&mut buffer).unwrap();

        set_trace_collections.push(bincode::serde::decode_from_slice(&buffer, bincode::config::standard()).expect("All read in files must be correct bincode!").0);

        let mut index: usize = 0;
        for byte_len in &buffer_lens[1..] {
            buffer.clear();
            index = index+1;
            let mut handle = read_file.by_ref().take(*byte_len as u64);
            handle.read_to_end(&mut buffer).unwrap(); 
            let (interim, _bytes): (SetTraceDef, usize) = bincode::serde::decode_from_slice(&buffer, bincode::config::standard()).expect("All read in files must be correct bincode!");
            //interim.reduce(max_max_length);
            set_trace_collections[chain].append(interim);
        }


        if (min_chain+1) < max_chain {for bead in (min_chain+1)..max_chain {


            let chain_file = base_str.clone()+format!("_{}_trace", bead).as_str()+".bin.gz";


            let len_file = base_str.clone()+format!("_{bead}_bits.txt").as_str();

            let mut string_buff: String = String::new();
            fs::File::open(&len_file).unwrap().read_to_string(&mut string_buff);

            let mut buffer_lens: Vec<usize> = string_buff.split_terminator('\n').map(|a| {
                a.parse::<usize>().expect("This whole file should be unsigned integers of buffer lengths")
            }).collect();


            let mut read_file: ParDecompress<Mgzip> = ParDecompressBuilder::new().from_reader(fs::File::open(chain_file).unwrap());

            for byte_len in buffer_lens.iter() {

                buffer.clear();
                let mut handle = read_file.by_ref().take(*byte_len as u64);
                handle.read_to_end(&mut buffer).unwrap();
                let (interim, _bytes): (SetTraceDef, usize) = bincode::serde::decode_from_slice(&buffer, bincode::config::standard()).expect("All read in files must be correct bincode!"); 

                set_trace_collections[chain].append(interim);
            }




        }}

        buffer.clear();
        }
    //num executed steps each chain (note: this is multiplied by 30, the default sparsing): 
    println!("{:.1e} \\\\", set_trace_collections.iter().map(|a| a.len()*30).format(" & "));
    println!("\\hline");


}
