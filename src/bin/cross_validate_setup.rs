
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

    /// Sets the input file from preprocessing for us to get cross-validation from
    #[arg(short, long)]
    input: String,

    /// Sets the output directory for your cross validation data
    #[arg(short, long)]
    output: String,

    /// Sets the k for k-fold validation. If larger than the number of blocks, 
    /// we automatically assume you're trying to LOO for every block and truncate.
    /// If this is less than 2, this code panics.
    #[arg(short)]
    k: usize,

    /// Sets whether we should sort sets for count off by sequence order
    /// or by order of "strength", which we define as the sum of all 
    /// non negative data points on the actual data. Default is sequence order
    #[arg(short, long)]
    by_strength: bool,

    /// Sets the 0 indexed blocks that you want separate for the k-fold validation.
    /// If you provide more blocks than k, we will minimize the number of blocks
    /// that are overlapping. EG, if you provide [0,1,2,3,4], and k = 3, with 9
    /// sequence blocks total, we might give a division of [0,1,7], [3,4,6], [2,5,8],
    /// or [0,4,6], [1,2,8], [3,5,7], but never [0,1,2], [3,4,5], [6,7,8]. 
    /// If k exceeds the number of blocks you give, some of the blocks created 
    /// won't have your special blocks. If  any block you give exceeds the max 
    /// block index of the data, it will be omitted.
    special_blocks: Option<Vec<usize>>

}



fn main() {


    let Cli { input, output, mut k, by_strength, special_blocks} = Cli::parse();

    let mut data_file_handle : ParDecompress<Mgzip> = ParDecompressBuilder::new().from_reader( File::open(input.as_str()).expect("You initialization file must be valid for inference to work!"));

    let mut buffer: Vec<u8> = Vec::new();

    _ = data_file_handle.read_to_end(&mut buffer).expect("Something went wrong when reading the data input file!");

    let (mut total_data, _bytes): (AllData, usize) = bincode::serde::decode_from_slice(&buffer, bincode::config::standard()).expect("Something was incorrect with your saved data input bincode file!");

    let data_ref = AllDataUse::new(&total_data, 0.0).unwrap();

    let num_blocks = data_ref.num_blocks();

    if k > num_blocks { k = num_blocks;}

    if k < 2 { panic!("No {k}-fold validation possible: need at least 2-fold validation!");} 

    // After this, we retain an ordered list of unique indices that map to a block
    let special_blocks: Option<Vec<usize>> = special_blocks.map(|mut a| {
        a.sort_unstable(); 
        a.dedup(); 
        a.retain(|&x| x < num_blocks); 
        if a.len() == 0 { None } else {Some(a)}
    }).flatten();

    let mut block_divisions: Vec<Vec<usize>> = vec![Vec::with_capacity(num_blocks/k + 1); k]; 

    let mut rng = rand::thread_rng();

    let mut index_use: Vec<usize> = (0..num_blocks).collect();

    let mut place_ind: usize = 0;
    
    // If we're doing strength sorting, we do that before any constraints to
    // minimize the amount of complexities I have to worry about.
    // Note that I very much specifically do a descending stable sort here. 
    if by_strength {
        let strengths = data_ref.return_estimated_block_peakiness();

        let mut strength_checker: Vec<_> = index_use.iter().zip(strengths.iter()).map(|(&a, &b)| (a,b)).collect();

        strength_checker.sort_by(|a,b| b.1.partial_cmp(&a.1).unwrap());

        index_use = strength_checker.into_iter().map(|a| a.0).collect();
    }

    if let Some(mut constraint) = special_blocks {

        // We're treating the blocks in constraint specially, so remove them from 
        // any normal consideration.
        index_use.retain(|a| !constraint.contains(a));

        // if constraint is less than k, the following code still just works,
        // as min_stack_size will be 0, and thus drained will contain no values,
        // and all parts of the logic up to the last pushes simplify to no-ops
        // We generally want to keep the vecs in block_divisions at about the
        // same length, hence start ind changing. 
        let min_stack_size = constraint.len()/k;
        place_ind = constraint.len() % k;

        for i in 0..k {
            let mut drained: Vec<usize> = constraint.drain(0..min_stack_size).collect();
            block_divisions[i].append(&mut drained);
        }

        for i in 0..constraint.len() {
            block_divisions[i].push(constraint[i])
        }

    };


    //I used to shuffle the blocks. Now, I just put them in one after the next
    //in each different bin. If I don't have special blocks, this simplifies to
    //just sorting block n into group n % k. 
    //index_use.shuffle(&mut rng);

    for block_ind in index_use.into_iter() {
        block_divisions[place_ind].push(block_ind);
        place_ind = (place_ind + 1) % k;
    }

    // Now block_divisions contains every set of active blocks I want to generate leave out
    // and retain AllDatas for.
    let loo_directory = format!("{output}/exclude_blocks");

    create_dir_all(loo_directory.clone());

    let osb_directory = format!("{output}/only_with_blocks");
    
    create_dir_all(osb_directory.clone());

    let mut buffer: Vec<u8> = Vec::new();

    for (i, block_set) in block_divisions.into_iter().enumerate() {


        let loo_set = data_ref.with_removed_blocks(&block_set).unwrap();

        let mut osb_set_inds: Vec<usize> = (0..num_blocks).collect();
        osb_set_inds.retain(|a| !block_set.contains(a));

        let osb_set = data_ref.with_removed_blocks(&osb_set_inds).unwrap();

        let loo_file_str_short = format!("{loo_directory}/omit_set_{i}");
        let loo_file_str = loo_file_str_short.clone();
        let osb_file_str_short = format!("{osb_directory}/keep_set_{i}");
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










}





