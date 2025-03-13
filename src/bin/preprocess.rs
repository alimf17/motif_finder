

use motif_finder::{NULL_CHAR, NUM_CHECKPOINT_FILES, NUM_RJ_STEPS, MAX_E_VAL, MAX_TF_NUM, ECOLI_FREQ};
use motif_finder::{PROPOSE_EXTEND, DIRICHLET_PWM, THRESH, NECESSARY_MOTIF_IMPROVEMENT};
use motif_finder::base::*;

use motif_finder::data_struct::*;

use log::warn;


use clap::{Parser, ValueEnum};


use std::path::*;
use std::time::{Instant};
use std::env;
use std::fs::{File, create_dir_all};
use std::io::{Read, Write};

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {

    /// Sets the name of your output file
    name: String,

    /// Sets the directory location of your output file
    output_dir: String,

    /// Sets the name of the fasta file for your inference 
    #[arg(short, long)]
    fasta: String,

    /// Sets the name of your data file for your inference 
    #[arg(short, long)]
    data: String,

    /// Indicates whether your genome is circular (true) or linear (false)
    #[arg(short, long)]
    circular: bool,

    /// Gives the average length of the DNA fragments in your prep, in base pairs
    /// This should come out to about half the width of your singleton peaks. 
    /// Note the word "singleton": a "peak" might actually be multiple binding 
    /// kernels glommed next to each other.
    #[arg(short, long)]
    length_of_fragment: usize,

    /// A positive integer corresponding to the spacing between data points, in
    /// base pairs. Note that this shoud probably be close to whatever spacing
    /// you have for your probes in your data file: we linearly interpolate if 
    /// points are missing, but we only linearly interpolate. If this is 0 or  
    /// not less than length_of_fragment, the preprocessing will panic. 
    #[arg(short, long)]
    spacing: usize,

    /// The minimum height of your inference. If this is less than 1.0, it's 
    /// set to 1.0.
    #[arg(short, long)]
    min_height: f64, 

    /// The penalty on increasing your number of inferred motifs. It corresponds
    /// to -ln(1-p) in a geometric distribution. If this is not positive, the 
    /// preprocessing will panic. 
    #[arg(short, long)]
    prior: f64,

    /// A scaling of the peak height cutoff for determining whether a region is
    /// considered peaky or not. More than 1.0 means being stricter about peaks,
    /// less than 1.0 means being laxer about calling a region peak-y. If this
    /// is not supplied or infinite, we set it to 1.0. If this is negative, we
    /// take the absolute value. 
    #[arg(short, long)]
    height_scale: Option<f64>,

}


fn main() {

    //Must have the following arguments:

    //0) Name of output file
    //3) FASTA file name -prep
    //4) "true" if the genome is circular, "false" if it's linear -prep
    //5) Data file name -prep 
    //6) The average length of the DNA fragments after your DNAse digest -prep
    //7) A positive integer corresponding to the spacing between data points in bp -prep

    //11) The min height, -prep 

    //12) A non negative double for how much to scale the cutoff for a data block
    //   to be considered peaky: less than 1.0 is more permissive, greater than 1.0 
    //   is more strict. If this exists but can't parse to a double, it's turned into a 1.0 -prep
    //16) A strictly positive double indicating how much ln likelihood a PWM needs to bring before
    //    being considered. Will only work if (9), (10), (11), and (12) included -prep 
    //

    let Cli { name: name, output_dir: output_dir, fasta: fasta_file, data: data_file, circular: is_circular, length_of_fragment: fragment_length, spacing: spacing, min_height: min_height, prior: credibility, height_scale: peak_cutoff}= Cli::parse();

    assert!(spacing > 0, "The spacing cannot be zero!");
    assert!(fragment_length > spacing, "The fragment length must be strictly greater than the spacing!");


    println!("c {credibility}");
    if !(credibility > 0.0) {panic!("Motif prior threshold must be a valid strictly positive float");}
    //SAFETY: This modification is made before any inference is done, preventing data races



    //By the end of this block, init_check_index holds the index where we check what type of
    //initial condition we have, if any. If this is larger than base_check_index, then we have arguments to parse
    //that change statics relevant for inference.
    //

    println!("Args parsed");


    let total_data: AllData  = AllData::create_inference_data(&fasta_file, &data_file, &output_dir, is_circular, fragment_length, spacing, min_height, credibility, &NULL_CHAR, peak_cutoff).unwrap();


    let data_ref = AllDataUse::new(&total_data, 0.0).unwrap();
    
    println!("have all data");

    let buffer: Vec<u8> = bincode::serialize(&total_data).expect("serializable");

    let output_name = format!("{output_dir}/{name}");

    let mut outfile_handle = match File::create(output_name.clone()) {
        Err(_) => {
            create_dir_all(output_name.clone()).unwrap();
            File::create(output_name.clone()).unwrap()
        },
        Ok(o) => o
    };

    outfile_handle.write(&buffer).expect("We just created this file");

    println!("Saved preprocessing file to {output_name}.");

}
