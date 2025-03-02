use motif_finder::data_struct::{AllData, AllDataUse};
use motif_finder::base::{MotifSet};

use std::io::{Read};
use std::{env, fs};

use rand::*;

fn main() {

    let args: Vec<String> = env::args().collect();

    let output_dir = args[1].as_str();
    let fasta_file = args[2].as_str();
    let is_circular: bool = args[3].parse().expect("Circularity must be either 'true' or 'false'!");
    let data_file = args[4].as_str();
    let fragment_length: usize = args[5].parse().expect("Fragment length must be a positive integer!");
    let spacing: usize = args[6].parse().expect("Spacing must be a positive integer!");

    let (data, _) = AllData::create_inference_data(fasta_file, data_file, output_dir, true, fragment_length, spacing, 2.0, &None, None).unwrap();

    let use_data = AllDataUse::new(&data, 0.0).unwrap();

    let mut rng = rand::thread_rng();

    let try_set = MotifSet::rand_with_one(&use_data, &mut rng);

    let sig = try_set.recalced_signal();
    
    println!("{:?}", sig);
   
}
