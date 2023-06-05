#[allow(unused_parens)]
mod base;
mod sequence;
mod waveform;
mod data_struct;

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



fn main() {

    //Must have the following arguments:

    //1) Name of run
    //2) Output directory
    //3) FASTA file name
    //4) Data file name
    
    //May also have the following arguments:

    //5) Either the word "meme" or the word "json". If one of these is not the argument,
    //   no other arguments will be considered.
    //6) The name of the initial condition file. Can either be a meme or json file. If 
    //   it is a JSON file containing the Set_Trace_Def, the name of the processed data
    //   as implied by the Fasta and data files will be checked against the name in the Set_Trace_Def. 
    //   If they're not identical, the program will panic on the spot
    let args: Vec<String> = env::args().collect();

    if args.len() < 5 {
        panic!("Not enough arguments!");
    }

    let run_name = args[1].as_str();
    let output_dir = args[2].as_str();
    let fasta_file = args[3].as_str();
    let data_file = args[4].as_str();

    




    



}
