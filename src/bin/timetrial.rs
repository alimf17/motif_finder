use std::fs::File;
use std::io::{Read};


use motif_finder::{NUM_CHECKPOINT_FILES, NUM_RJ_STEPS};
use motif_finder::{PROPOSE_EXTEND, DIRICHLET_PWM, THRESH};
use motif_finder::base::*;

use motif_finder::data_struct::*;
use motif_finder::modified_t::SymmetricBaseDirichlet;









use std::time::{Instant};

use std::env;

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
    //9) The minimum thermodynamic beta for the chains. The absolute value of this will be taken, and if you give a number > 1.0, the reciprocal will be taken instead
    //10) The number of intermediate traces between the base inference chain and the minimum thermodynamic beta chain. Must be a non-negative integer
    //May also have the following arguments:

    //10) A non negative double for how much to scale the cutoff for a data block
    //   to be considered peaky: less than 1.0 is more permissive, greater than 1.0 
    //   is more strict
    //11) A strictly positive double for alpha of a symmetric dirichlet distribution from which we draw base
    //   extensions and contractions on a single motif Will only work if (9) included
    //12) A strictly positive double for alpha of a symmetric dirichlet distribution from which we draw
    //    base identities for entirely NEW PWMs. Will only work if (9) and (10) included
    //13) A strictly positive double less than 1.0 indicating how weak a motif needs to be in a
    //    certain position before giving up drawing it. Will only work if (9), (10), and (11) included 
    //14) A strictly positive double indicating how much ln likelihood a PWM needs to bring before
    //    being considered. Will only work if (9), (10), (11), and (12) included
    //
    //Penultimate argument) Either the word "meme" or the word "json". If one of these is not the argument,
    //   no other arguments will be considered.
    //Final argument) The name of the initial condition file. Can either be a meme or json file. If 
    //   it is a JSON file containing the SetTraceDef, the name of the processed data
    //   as implied by the Fasta and data files will be checked against the name in the SetTraceDef. 
    //   If they're not identical, the program will panic on the spot
    let args: Vec<String> = env::args().collect();

    if args.len() < 5 {
        panic!("Not enough arguments!");
    }

    let run_name = args[1].clone();

    let output_dir = args[2].as_str();

    let mut check_initialization = File::open(&args[3]).expect("This won't work without a valid input file");
    
    let mut buffer: Vec<u8> = Vec::new();

    let _ = check_initialization.read_to_end(&mut buffer);

    let total_data: AllData = bincode::deserialize(&buffer).expect("Your data file does not parse");

    let num_advances: usize = args[4].parse().expect("The number of advances to run must be a positive integer!");
 
    let mut min_thermo_beta: f64 = args[5].parse().expect("The minimum thermodynamic beta must be a float!");

    let num_intermediate_traces: usize = args[6].parse().expect("The number of intermediate traces must be a non-negative integer!");

    unsafe {
        THRESH = 1e-2;
    }


    min_thermo_beta = min_thermo_beta.abs();

    PROPOSE_EXTEND.set(SymmetricBaseDirichlet::new(20.0_f64).expect("obviously valid")).expect("Nothing should have written to this before now");
    
    DIRICHLET_PWM.set(SymmetricBaseDirichlet::new(1.0_f64).expect("obviously valid")).expect("Nothing should have written to this before now");

    println!("Args set!");

    let data_ref = AllDataUse::new(&total_data).unwrap();

    let _data = data_ref.data();

    let _background = data_ref.background_ref();

    let save_step = (1+(num_advances/NUM_CHECKPOINT_FILES)).min(1000);
    let capacity: usize = save_step*(NUM_RJ_STEPS+2);

    let steps_per_exchange_attempt: usize = 50;


    let mut rng = rand::thread_rng();

    let maybe_init: Option<MotifSet> = None;

    //    pub fn new_parallel_traces<R: Rng+?Sized>(min_thermo_beta: f64, num_intermediate_traces: usize, capacity_per_trace: usize, step_num_estimate: usize, how_to_track: &TrackingOptions, data_ref: &'a AllDataUse<'a>, initial_condition: Option<MotifSet<'a>>, sparse: Option<usize>, rng: &mut R) -> Result<Self, InitializationError> {

    let mut initialization_chains = TemperSetTraces::new_parallel_traces(min_thermo_beta, num_intermediate_traces, capacity, num_advances, TrackingOptions::TrackAllTraces, args[3].clone(), &data_ref, maybe_init, None, &mut rng).unwrap();



    //run MCMC and make sure that I'm saving and clearing periodically
    

    let _track_hmc: usize = 0;

    let start_inference_time = Instant::now();

    //for step in 0..10000 {

    let pushes = num_advances/steps_per_exchange_attempt + ((num_advances % steps_per_exchange_attempt) > 0) as usize;
 
    for step in 0..pushes {

        println!("push {step}");
        initialization_chains.iter_and_swap(10, steps_per_exchange_attempt, rand::thread_rng);

      //  if step % 5 == 0 {

            initialization_chains.print_acceptances(TrackingOptions::TrackAllTraces);

      //  }
      //  if ((step+1) % 5 == 0) || (step+1 == pushes) {
        
            let root_signal: String = format!("{}/{}_dist_of",output_dir,run_name);

            let num_bins: usize = 100;

            initialization_chains.handle_histograms(TrackingOptions::TrackAllTraces, &root_signal, num_bins);
            
            initialization_chains.save_trace_and_clear(output_dir, &run_name, step);

    //    }

    }

    println!("Finished run in {:?}", start_inference_time.elapsed());

  


}
