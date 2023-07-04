#[allow(unused_parens)]
mod base;
mod sequence;
mod waveform;
mod data_struct;
mod modified_t;

use crate::base::*;
use crate::sequence::Sequence;
use crate::waveform::*;
use crate::data_struct::*;
use statrs::distribution::{Continuous, ContinuousCDF, LogNormal, Normal, StudentsT};
use statrs::statistics::{Min, Max};
use statrs::function::gamma;
use rand::Rng;
use std::collections::VecDeque;
use rand::distributions::{Distribution, Uniform};
use aberth::aberth;
use num_complex::Complex;
const EPSILON: f64 = 1e-8;
use once_cell::sync::Lazy;
use num_traits::cast;
use num_traits::float::Float;
use num_traits::float::FloatConst;
use num_traits::identities::{One, Zero};
use num_traits::MulAdd;
use core::iter::zip;
use std::time::{Duration, Instant};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;
use std::env;

use serde_json::value::Serializer as Json_Serializer;

use serde::{ser, Serialize,Serializer, Deserialize};
use serde::de::{
    self, DeserializeSeed, EnumAccess, IntoDeserializer, MapAccess, SeqAccess,
    VariantAccess, Visitor,
};


const NULL_CHAR: Option<char> = None;
 
const MOMENTUM_SD: f64 = 1.0;
const NUM_CHECKPOINT_FILES: usize = 25;
const NUM_RJ_STEPS: usize = 5;
const NUM_HMC_STEPS: usize = 50;

const HMC_TRACE_STEPS: usize = 2; 
const HMC_EPSILON: f64 = 1.0/16.0; 

//This only matters when taking in a meme file
const MAX_E_VAL: f64 = 0.01;

fn main() {

    //Must have the following arguments:

    //1) Name of run
    //2) Output directory
    //3) FASTA file name
    //4) "true" if the genome is circular, "false" if it's linear
    //5) Data file name
    //6) The average length of the DNA fragments after your DNAse digest
    //7) A positive integer corresponding to the spacing between data points in bp
    //8) Number of advances to run the set trace
    //May also have the following arguments:

    //9) Either the word "meme" or the word "json". If one of these is not the argument,
    //   no other arguments will be considered.
    //10) The name of the initial condition file. Can either be a meme or json file. If 
    //   it is a JSON file containing the SetTraceDef, the name of the processed data
    //   as implied by the Fasta and data files will be checked against the name in the SetTraceDef. 
    //   If they're not identical, the program will panic on the spot
    let args: Vec<String> = env::args().collect();

    if args.len() < 9 {
        panic!("Not enough arguments!");
    }

    let run_name = args[1].as_str();
    let output_dir = args[2].as_str();
    let fasta_file = args[3].as_str();
    let is_circular: bool = args[4].parse().expect("Circularity must be either 'true' or 'false'!");
    let data_file = args[5].as_str();
    let fragment_length: usize = args[6].parse().expect("Fragment length must be a positive integer!");
    let spacing: usize = args[7].parse().expect("Spacing must be a positive integer!");
    
    assert!(spacing > 0, "The spacing cannot be zero!");
    assert!(fragment_length > spacing, "The fragment length must be strictly greater than the spacing!");

    let num_advances: usize = args[8].parse().expect("The number of advances to run must be a positive integer!");
 
    let (total_data,data_string): (All_Data, String) = All_Data::create_inference_data(fasta_file, data_file, output_dir, is_circular, fragment_length, spacing, &NULL_CHAR);

    let data: Waveform = total_data.validated_data();

    let background = total_data.background();

    let save_step = (num_advances/NUM_CHECKPOINT_FILES);
    let capacity: usize = save_step*(NUM_RJ_STEPS+NUM_HMC_STEPS+2);

    //Initialize trace
    let mut current_trace: SetTrace = SetTrace::new_empty(capacity, &data, &background);

    let mut rng = rand::thread_rng();
    
    let check: Option<&str> = match args.get(9) { Some(x) => Some(x.as_str()), None => None};
    match check {

        Some("meme") => current_trace.trace_from_meme(args.get(10).expect("Must include a string indicating MEME output file").as_str(),data.seq(), MAX_E_VAL, fragment_length),
        Some("json") => current_trace.push_last_state_from_json(false, args.get(10).expect("Must inlcude a string indicating a Json output file").as_str()),
        _ => current_trace.push_set(MotifSet::rand_with_one(&data, &background, fragment_length, &mut rng)),
    };

    //run MCMC and make sure that I'm saving and clearing periodically
    

    let mut acceptances: [usize;5]= [0;5];
    let mut trials: [usize;5] = [0;5];

    for step in 0..num_advances {
 
        let (this_acceptance, this_trials) = current_trace.advance(&mut rng, MOMENTUM_SD, HMC_TRACE_STEPS, HMC_EPSILON, NUM_RJ_STEPS, NUM_HMC_STEPS);

        for (total_acc, curr_acc) in acceptances.iter_mut().zip(this_acceptance.iter()) {*total_acc += *curr_acc;}
        for (total_try, curr_try) in trials.iter_mut().zip(this_trials.iter()) {*total_try += *curr_try;}
 
        let rates: Vec<f64> = acceptances.iter().zip(trials.iter()).map(|(&a, &b)| (a as f64)/(b as f64)).collect();

        println!("Step {}. Acceptance rates for {:#?} and HMC respectively are: {:#?}", step, RJ_MOVE_NAMES, rates);

        if step % save_step == 0 {
            current_trace.save_and_drop_history(output_dir, run_name, step*save_step);
        }

    }


    println!("Finished run");

    



}
