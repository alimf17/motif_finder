use std::fs;
use std::io::{Read, Write};
use std::error::Error;

use motif_finder::base::*;
use motif_finder::base::{SQRT_2, SQRT_3, LN_2, BPS};
use motif_finder::{NECESSARY_MOTIF_IMPROVEMENT, ECOLI_FREQ};
use motif_finder::data_struct::{AllData, AllDataUse};
use motif_finder::gene_loci::*;

use clap::{Parser, ValueEnum};

use gzp::{deflate::Mgzip, par::compress::{ParCompress, ParCompressBuilder}, syncz::{SyncZ, SyncZBuilder}, par::decompress::{ParDecompress, ParDecompressBuilder},ZWriter, Compression};


use log::warn;

use rustfft::{FftPlanner, num_complex::Complex};

use ndarray::prelude::*;

use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelIterator;

use rand::prelude::*;

use regex::Regex;
use serde::*;
use serde::de::DeserializeOwned;
use poloto;
use poloto::Croppable;
use plotters::prelude::*;
use plotters::prelude::full_palette::*;
use plotters::coord::ranged1d::{NoDefaultFormatting, KeyPointHint,ValueFormatter};

use spectrum_analyzer::{samples_fft_to_spectrum, FrequencyLimit};

use core::ops::Range;

use sptr::*;

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

    
    /// This is the `MotifSet` you're comparing the traces to
    #[arg(short, long)]
    comparison_set: String,

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


    /// Number of sequential runs per chain to discard as burn in. 
    /// If this is not provided, it's set as 0. If this is larger
    /// than max_runs, this script will panic. 
    #[arg(short, long)]
    discard_num: Option<usize>,

}


pub fn main() {

    let Cli { comparison_set, output: out_dir, base_file, mut num_chains, max_runs: max_chain, discard_num} = Cli::parse();
   


    let min_chain = discard_num.unwrap_or(0);

    if min_chain > max_chain {
        panic!("Discarding everything you should infer on!");
    }


    if num_chains > 26 {
        warn!("Can only sustain up to 26 independent chains. Only considering the first 26.");
        num_chains = 26;
    }

    let mut buffer: Vec<u8> = Vec::new();
    
    let parsed_set: StrippedMotifSet = attempt_bincode_serde_read_with_decomp(&comparison_set, Some(&mut buffer), None).unwrap();

    buffer.clear();

    //let max_max_length = 100000;
    //This is the code that actually sets up our independent chain reading
    let mut set_trace_collections: Vec<SetTraceDef> = Vec::with_capacity(max_chain-min_chain);
    for chain in 0..num_chains {
        let base_str = format!("{}/{}_{}", out_dir.clone(), base_file, UPPER_LETTERS[chain]);
        println!("Base str {} min chain {}", base_str, min_chain);
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

        println!("buffer 0 {}", buffer_lens[0]);
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


            let chain_file = base_str.clone()+format!("_{}_trace", min_chain).as_str()+".bin.gz";

            let len_file = base_str.clone()+format!("_{min_chain}_bits.txt").as_str();

            let mut string_buff: String = String::new();
            fs::File::open(&len_file).unwrap().read_to_string(&mut string_buff);
            
            let mut buffer_lens: Vec<usize> = string_buff.split_terminator('\n').map(|a| {
                a.parse::<usize>().expect("This whole file should be unsigned integers of buffer lengths")
            }).collect();


            let mut read_file: ParDecompress<Mgzip> = ParDecompressBuilder::new().from_reader(fs::File::open(chain_file).unwrap());

            for byte_len in &buffer_lens[1..] {
                buffer.clear();
                let mut handle = read_file.by_ref().take(*byte_len as u64);
                handle.read_to_end(&mut buffer).unwrap();
                let (interim, _bytes): (SetTraceDef, usize) = bincode::serde::decode_from_slice(&buffer, bincode::config::standard()).expect("All read in files must be correct bincode!"); 
                
                set_trace_collections[chain].append(interim);
            }




        }}
    
        buffer.clear();
    }

    let max_ln_post_per_trace = set_trace_collections.iter().filter_map(|a| a.ln_posterior_trace().into_iter().max_by(|a,b| a.partial_cmp(&b).unwrap())).collect::<Vec<_>>();

    let min_allowable_ln_post = max_ln_post_per_trace.iter().max_by(|a,b| a.partial_cmp(b).unwrap()).expect("traces exist")-1000.0_f64;

    let alpha: f64 = 0.05;
        
    //We assume that two motifs are related enough to be a "hit"
    //if they are in a distance that has a 0.05 chance to occur 
    //under the null hypothesis. Our version of the aitchison metric
    //for the E. coli genome, U.00092.2 (with slight modifications of about single bps),
    //empirically fit a roughly normal distribution with mean ~8.25 and sd ~ 1.0
    //when we simulated pairs of valid motifs under the prior distribution
    //and checked their distances
    let distance_to_hit =5.435515_f64; // 5.923652_f64;

    let expected_hits: fn(&StrippedMotifSet) -> f64 = |x| { 1.0-(1.0-alpha).powi(x.num_motifs() as i32)};

    for mot in parsed_set.set_iter() {

        println!("Checking motif {}\n {:?}", mot.best_motif_string(), mot);


        let comparison = |x: &StrippedMotifSet| if x.set_iter().map(|s| (&mot).distance_function(s)).min_by(|a,b| a.0.partial_cmp(&b.0).unwrap()).expect("have at least one motif in every trace").0 <= *&(distance_to_hit) { 1.0 } else {0.0};

        set_trace_collections.iter().enumerate().map(|(i, trace)| {
            
            let proportion_hits = trace.evaluate_posterior_probability(min_allowable_ln_post, &comparison);
            let expected_hits = trace.evaluate_posterior_probability(min_allowable_ln_post, &expected_hits);

            if expected_hits.is_nan() { println!("Set {} not good enough to have a posterior calculation", UPPER_LETTERS[i]);} else {

                let bayes_factor = proportion_hits*(1.0-expected_hits)/((1.0-proportion_hits)*expected_hits);

                println!("Bayes factor for this motif to appear in the posterior of set {} (with max like off by {}) is: {}",UPPER_LETTERS[i], max_ln_post_per_trace[i]-(min_allowable_ln_post+1000.), bayes_factor);
            }


        }).collect::<Vec<_>>();

    }
 


}


pub fn attempt_bincode_serde_read_with_decomp<T: DeserializeOwned>(file_name: &str, preferred_scratch: Option<&mut Vec<u8>>, config: Option<bincode::config::Configuration>) -> Result<T, Box<dyn Error+Send+Sync>> {

    let config = config.unwrap_or(bincode::config::standard());

    let mut alter_scratch: Vec<u8> = Vec::new();

    let mut buffer_handle = if let Some(handle) = preferred_scratch {std::mem::drop(alter_scratch); handle} else {&mut alter_scratch};

    buffer_handle.clear();

    let mut file_handle = fs::File::open(file_name)?;

    file_handle.read_to_end(buffer_handle);

    let no_comp_trial: Result<(T, usize), _>  = bincode::serde::decode_from_slice(buffer_handle, config);

    if let Ok((no_comp, _)) = no_comp_trial {
        return Ok(no_comp);
    };

    buffer_handle.clear();

    let mut file_handle = fs::File::open(file_name)?;

    let mut read_file: ParDecompress<Mgzip> = ParDecompressBuilder::new().from_reader(file_handle);

    read_file.read_to_end(buffer_handle);

    let (comped, _): (T, usize) = bincode::serde::decode_from_slice(buffer_handle, config)?;

    Ok(comped)

}

fn parse_set_names_and_files(split: Vec<String>) -> Result<Vec<(String, String)>, Box<dyn Error+Send+Sync>> {

    //let split: Vec<&str> = s.split(' ').collect();

    let (pair_off, []) = split.as_chunks::<2>() else {return Err(Box::new(ParityError {}));};

    if pair_off.len() < 2 { return Err(Box::new(NoCompError {}));}

    Ok(pair_off.into_iter().map(|a| (a[0].to_owned(), a[1].to_owned())).collect::<Vec<(String, String)>>())

}

#[derive(Debug)]
pub struct ParityError {}

impl std::fmt::Display for ParityError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Must have an even number of sets and file names!")
    }
}


impl Error for ParityError {}

#[derive(Debug)]
pub struct NoCompError {}

impl std::fmt::Display for NoCompError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Must have at least two sets to compare!")
    }
}


impl Error for NoCompError {}
