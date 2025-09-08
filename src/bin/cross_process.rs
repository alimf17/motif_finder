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

    /// This is a saved copy of the genome you're comparing everything on. 
    /// It should probably be derived from using `AllData::create_inference_data()` 
    /// where the `retain_null` parameter is `Some(true)`.
    #[arg(short, long)]
    comparison_file: String,
    
    /// These are the save files of the `MotifSet`s you're comparing, each 
    /// followed by the name you want them to have in the processing
    #[arg(short, long, num_args = 2.., value_delimiter = ' ')]
    motif_set_files: Vec<String>,

    /// This is the file where the Aitchison distance matrix should be output
    /// Note that the values are tab delimited
    #[arg(short, long)]
    distance_matrix: String,

    /// This is the file where the binding distance matrix should be output
    /// Note that this is table delimited
    #[arg(short, long)]
    binding_matrix: String,

    /// This is an optional argument if you want to annotate your occupancy
    /// traces with gene loci and analyze go terms. This should point to a gff file
    #[arg(short, long)]
    gff_file: Option<String>,

    /// This is an extremely optional argument if you want more annotations
    /// for your genes with more go terms. This should point to a tsv file
    /// like you'd get from uniprot
    #[arg(short, long)]
    additional_annotations_file: Option<String>,

}

// #[arg(num_args = 2.., value_delimiter = ' ', value_parser=parse_set_names_and_files)]

pub fn main() {

    let Cli { comparison_file, motif_set_files, distance_matrix, binding_matrix, gff_file, additional_annotations_file} = Cli::parse();

    let motif_set_files: Vec<(String, String)> = parse_set_names_and_files(motif_set_files).expect("file parsing went wrong");

    let mut scratch_space_for_buffer: Vec<u8> = Vec::new();

    let data_struct: AllData = attempt_bincode_serde_read_with_decomp(&comparison_file, Some(&mut scratch_space_for_buffer), None).expect("Your data file must be valid!");

    let data_ref = AllDataUse::new(&data_struct, 0.0).unwrap();

    let motif_sets: Vec<(String, StrippedMotifSet)> = motif_set_files.into_iter().filter_map(|(name, file)| {

        let parsed_set: Result<StrippedMotifSet, _> = attempt_bincode_serde_read_with_decomp(&file, Some(&mut scratch_space_for_buffer), None);

        println!("{:?}", parsed_set);

        match parsed_set {
            Ok(set) => Some((name, set)),
            Err(_) => None,
        }

    }).collect();

    println!("{:?}", motif_sets);

    //let reactivated: Vec<_> = motif_sets.into_iter().map(|(a,b)| (a, b.reactivate_set(&data_ref))).collect();
  
    //println!("reactive {:?}", reactivated);

    //let ref_0 = motif_sets[0].1.get_nth_motif(0) as *const Motif;
    //let ref_1 = motif_sets[0].1.get_nth_motif(1) as *const Motif;

    //println!("{} {} {} {}", ref_0.addr(), ref_1.addr(), ref_1.addr()-ref_0.addr(), std::mem::size_of::<Motif>()) 

    let size_of_motif = std::mem::size_of::<Motif>();


    let assignments = motif_sets.iter().map(|a| motif_sets.iter().map(|b| a.1.pair_to_other_set_potential_binding(&b.1, data_ref.min_height()).into_iter().enumerate().map(|(i,d)|  d.map(|c| (c.0, a.1.get_nth_motif(i).best_motif_string(), if c.1 {c.2.rev_complement_best_motif() } else {c.2.best_motif_string()}, ((c.2 as *const Motif).addr()-(b.1.get_nth_motif(0) as *const Motif).addr())/size_of_motif))).collect::<Vec<_>>()).collect::<Vec<_>>()).collect::<Vec<_>>();

    println!("{:?}", assignments);

    let mut sorted_matches: Vec<(String, usize, String,  String, usize, String, f64)> = Vec::with_capacity(assignments.len());

    //Now, what I have in assignments is a three nested vector. The index of the
    //first layer is the index of the motif set A, the index of the second layer 
    //is the index of the motif set B it's being compared to, and the index j of 
    //the third layer is the assignment of the jth motif of motif set A to its 
    //match in motif set B. The elements I have are a float indicating the 
    //distance, followed by the index k of the motif from MotifSet B.
    //If the jth element is None, it means that the jth motif of motif set A did 
    //not get assigned to anything in motif set B. 

    let rows: Vec<String> = motif_sets.iter().map(|set| (0..set.1.num_motifs()).map(|set_id| format!("{}:Motif_{}", set.0, set_id))).flatten().collect();
    let mot_rows: Vec<String> = motif_sets.iter().map(|set| set.1.set_iter().map(|a| a.best_motif_string())).flatten().collect();

    let header: Vec<String> = motif_sets.iter().map(|s| format!("{0: <30}", s.0)).collect();

    let mut assignment_file = std::fs::File::create(&distance_matrix).unwrap();
    let _ = assignment_file.write(format!("Motif getting matched\t(sequence)            \t{}\n", header.join("\t\t")).as_bytes());

    for (which_set, set_assignment) in assignments.iter().enumerate() {

        let num_motifs = motif_sets[which_set].1.num_motifs();
        
        let set_name = motif_sets[which_set].0.clone();
        
        for i in 0..num_motifs {
            let best_motif = motif_sets[which_set].1.get_nth_motif(i).best_motif_string();
            let match_list: Vec<String> = set_assignment.iter().enumerate().map(|(j, compared_set)| {
                if let Some(matcher) = &compared_set[i] {
                    if which_set < j && matcher.0 != 0.0{
                    let mut alter_name = matcher.2.clone();
                    alter_name = alter_name.split_whitespace().collect::<Vec<_>>().join(" ");
                    sorted_matches.push((set_name.clone(), i, best_motif.clone(), header[j].clone(), matcher.3, alter_name, matcher.0));
                    }
                    format!("{}: {} ({:.02})", matcher.3, matcher.2, matcher.0)
                } else { "None".to_owned()}
            }).collect();
            let _ = assignment_file.write(format!("{set_name} Motif {i}\t({best_motif})\t{}\n", match_list.join("\t\t")).as_bytes()); //Insert the matching motifs here
           
        }

    }

    sorted_matches.sort_unstable_by(|a,b| a.6.partial_cmp(&b.6).unwrap());

    
    let filtered_matches = sorted_matches.iter().filter_map(|a| if a.6 < 6.25 {Some(format!("{:?}", a))} else {None}).collect::<Vec<_>>().join("\n");

    println!("{}", filtered_matches);

    println!("min height {}", data_ref.min_height());

    //let assignments = motif_sets.iter().map(|a| motif_sets.iter().map(|b| a.1.pair_to_other_set_binding(&b.1, &data_ref).into_iter().enumerate().map(|(i,d)| d.map(|c| (c.0, a.1.get_nth_motif(i).best_motif_string(), c.1.best_motif_string(), ((c.1 as *const Motif).addr()-(b.1.get_nth_motif(0) as *const Motif).addr())/size_of_motif))).collect::<Vec<_>>()).collect::<Vec<_>>()).collect::<Vec<_>>();
 
    //let assignments= (0..motif_sets.len()).map(|i| (i+1..motif_sets.len()).map(|j| motif_sets[i].1.pair_to_other_set_binding(&motif_sets[j].1, &data_ref).into_iter().enumerate().map(|(k,d)| d.map(|c| (c.0, motif_sets[i].1.get_nth_motif(k).best_motif_string(), c.1.best_motif_string(), ((c.1 as *const Motif).addr()-(motif_sets[j].1.get_nth_motif(0) as *const Motif).addr())/size_of_motif))).collect::<Vec<_>>()).collect::<Vec<_>>()).collect::<Vec<_>>();
    //println!("{:?}", assignments);
    
    println!("{:?}", motif_sets[0].1.pair_to_other_set_rmse_data(&motif_sets[2].1, &data_ref));

    let mut assignment_file = std::fs::File::create(&binding_matrix).unwrap();
    let _ = assignment_file.write(format!("Motif getting matched\t(sequence)            \t{}\n", header.join("\t\t")).as_bytes());

    for (which_set, set_assignment) in assignments.iter().enumerate() {

        let num_motifs = motif_sets[which_set].1.num_motifs();
        
        let set_name = motif_sets[which_set].0.clone();
        
        for i in 0..num_motifs {
            let best_motif = motif_sets[which_set].1.get_nth_motif(i).best_motif_string();
            let match_list: Vec<String> = set_assignment.iter().map(|compared_set| {
                if let Some(matcher) = &compared_set[i] { format!("{}: {} ({:.02})", matcher.3, matcher.2, matcher.0)} else { "None".to_owned()}
            }).collect();
            let _ = assignment_file.write(format!("{set_name} Motif {i}\t({best_motif})\t{}\n", match_list.join("\t\t")).as_bytes()); //Insert the matching motifs here
            println!("finish write");
        }

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
