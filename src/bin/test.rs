#[allow(unused_parens)]


use motif_finder::{NULL_CHAR, NUM_CHECKPOINT_FILES, NUM_RJ_STEPS, NUM_HMC_STEPS, MAX_E_VAL};
use motif_finder::base::*;
use motif_finder::sequence::Sequence;
use motif_finder::waveform::*;
use motif_finder::data_struct::*;
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

fn main() {

    //Must have the following arguments:

    //1) Name of run
    //2) Output directory
    //3) FASTA file name
    //4) "true" if the genome is circular, "false" if it's linear
    //5) Data file name
    //6) The average length of the DNA fragments after your DNAse digest
    //7) A positive integer corresponding to the spacing between data points in bp
    //May also have the following arguments:

    //9) Either the word "meme" or the word "json". If one of these is not the argument,
    //   no other arguments will be considered.
    //10) The name of the initial condition file. Can either be a meme or json file. If 
    //   it is a JSON file containing the SetTraceDef, the name of the processed data
    //   as implied by the Fasta and data files will be checked against the name in the SetTraceDef. 
    //   If they're not identical, the program will panic on the spot
    let args: Vec<String> = env::args().collect();

    if args.len() < 8 {
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

    let (total_data,data_string): (AllData, String) = AllData::create_inference_data(fasta_file, data_file, output_dir, is_circular, fragment_length, spacing, false, &NULL_CHAR);

    let data: Waveform = total_data.validated_data();

    let background = total_data.background();


    //Initialize trace
    let mut current_trace: SetTrace = SetTrace::new_empty(10,data_string.clone(), &data, &background);

    let mut rng = rand::thread_rng();
    
    let check: Option<&str> = match args.get(9) { Some(x) => Some(x.as_str()), None => None};
    match check {

        Some("meme") => current_trace.trace_from_meme(args.get(10).expect("Must include a string indicating MEME output file").as_str(),data.seq(), MAX_E_VAL, fragment_length, &mut rng),
        Some("json") => {
            //If you're picking up from a JSON, you have the right to tell the motif finder to trust your motif set and minimize recalculation.
            //Note that it doesn't COMPLETELY trust you: it will validate on making sure your motif set signal is compatible with your processed ChIP data regardless. 
            //The default behavior is that it doesn't trust you.
            let (validate, mut maybe_rng) = match (args.get(10).map(|x| x.parse::<bool>().ok()).flatten()) { 
                None | Some(false) => (true, Some(&mut rng)),
                Some(true) => (false, None),
            };
            current_trace.push_last_state_from_json(validate, validate, &mut maybe_rng, args.get(10).expect("Must inlcude a string indicating a Json output file").as_str());
        },
        _ => current_trace.push_set(MotifSet::rand_with_one(&data, &background, fragment_length, &mut rng)),
    };

    //run MCMC and make sure that I'm saving and clearing periodically
    

    current_trace.save_trace(output_dir, run_name, 0);



    println!("Finished run");

  


}
