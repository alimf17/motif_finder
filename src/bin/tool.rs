

use motif_finder::{NULL_CHAR, NUM_CHECKPOINT_FILES, NUM_RJ_STEPS, MAX_E_VAL};
use motif_finder::{PROPOSE_EXTEND, DIRICHLET_PWM, THRESH, NECESSARY_MOTIF_IMPROVEMENT};
use motif_finder::base::*;
use motif_finder::waveform::*;
use motif_finder::data_struct::*;
use motif_finder::modified_t::SymmetricBaseDirichlet;

use log::warn;

use plotters::prelude::*;
use plotters::coord::types::RangedSlice;
use plotters::coord::Shift;

use statrs::distribution::Normal;
use once_cell::sync::Lazy;
use std::time::{Duration, Instant};

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

    if args.len() < 10 {
        panic!("Not enough arguments!");
    }

    let output_dir = args[2].as_str();
    let fasta_file = args[3].as_str();
    let is_circular: bool = args[4].parse().expect("Circularity must be either 'true' or 'false'!");
    let data_file = args[5].as_str();
    let fragment_length: usize = args[6].parse().expect("Fragment length must be a positive integer!");
    let spacing: usize = args[7].parse().expect("Spacing must be a positive integer!");
    
    assert!(spacing > 0, "The spacing cannot be zero!");
    assert!(fragment_length > spacing, "The fragment length must be strictly greater than the spacing!");

    let num_advances: usize = args[8].parse().expect("The number of advances to run must be a positive integer!");
 
    let mut min_thermo_beta: f64 = args[9].parse().expect("The minimum thermodynamic beta must be a float!");

    let mut num_intermediate_traces: usize = args[10].parse().expect("The number of intermediate traces must be a non-negative integer!");

    min_thermo_beta = min_thermo_beta.abs();

    if min_thermo_beta > 1.0 {
        warn!("Your min thermodynamic beta was set > 1.0! We're taking the reciprocal!");
        min_thermo_beta = 1.0/min_thermo_beta;
    }
    
    let mut check_for_init: bool = true;
    let base_check_index: usize = 11;
    let mut init_check_index: usize = base_check_index;

    while check_for_init {
        match args.get(init_check_index) {
            None => check_for_init = false,
            Some(x) => if x.parse::<f64>().is_ok() { init_check_index += 1;} else {check_for_init = false;},
        };
    }

    //By the end of this block, init_check_index holds the index where we check what type of
    //initial condition we have, if any. If this is larger than base_check_index, then we have arguments to parse
    //that change statics relevant for inference.
    //

    let mut peak_cutoff: Option<f64> = None;

    let initialize_func = |a: &f64| { SymmetricBaseDirichlet::new(*a).unwrap() };
    let run_name;
    if init_check_index > base_check_index {
        peak_cutoff = args[base_check_index].parse().ok();
        run_name = match peak_cutoff{
            Some(a) => format!("{}_custom_scale_{}", args[1].as_str(), args[9]),
            None => args[1].clone(),
        };
    } else {
        run_name = args[1].clone();
    }
    if init_check_index > base_check_index+1 { 
        let extend_alpha: f64 = args[base_check_index+1].parse().expect("We already checked that this parsed to f64");
        //I wrote the condition as follows in case an argument is passed in that makes this a NaN
        if !(extend_alpha > 0.0) {panic!("Dirichlet alpha must be a valid strictly positive float");} 
        //let mut w = PROPOSE_EXTEND.write().expect("This is the only thread accessing this to write, and the mutable reference goes out of scope immediately");
        //*w = Lazy::new(|&extend_alpha| SymmetricBaseDirichlet::new(extend_alpha.clone()).unwrap());
        PROPOSE_EXTEND.set(SymmetricBaseDirichlet::new(extend_alpha).expect("checked parameter validity already")).expect("Nothing should have written to this before now");
    } else {

        //Changed this from 10.0 because extensions not working well
        PROPOSE_EXTEND.set(SymmetricBaseDirichlet::new(20.0_f64).expect("obviously valid")).expect("Nothing should have written to this before now");
    }
    if init_check_index > base_check_index+2 { 
        let pwm_alpha: f64 = args[base_check_index+2].parse().expect("We already checked that this parsed to f64");
        //I wrote the condition as follows in case an argument is passed in that makes this a NaN
        if !(pwm_alpha > 0.0) {panic!("Dirichlet alpha must be a valid strictly positive float");} 
        //SAFETY: This modification is made before any inference is done, preventing data races
        //let mut w = DIRICHLET_PWM.write().expect("This is the only thread accessing this to write, and the mutable reference goes out of scope immediately");
        //*w = Lazy::new(|| SymmetricBaseDirichlet::new(pwm_alpha).unwrap());

        DIRICHLET_PWM.set(SymmetricBaseDirichlet::new(pwm_alpha).expect("checked parameter validity already")).expect("Nothing should have written to this before now");
    } else {

        //Changed this from 10.0 because I'm worried that new PWMs are too promiscuous
        DIRICHLET_PWM.set(SymmetricBaseDirichlet::new(1.0_f64).expect("obviously valid")).expect("Nothing should have written to this before now");
    }


    if init_check_index > base_check_index+3 { 
        let threshold: f64 = args[base_check_index+3].parse().expect("We already checked that this parsed to f64");
        //I wrote the condition as follows in case an argument is passed in that makes this a NaN
        if !(threshold > 0.0) || (threshold >= 1.0) {panic!("Peak drawing threshold must be a valid strictly positive float strictly less than 1.0");} 
        //SAFETY: This modification is made before any inference is done, preventing data races
        let mut w = THRESH.write().expect("This is the only thread accessing this to write, and the mutable reference goes out of scope immediately");
        *w = threshold; 
    }
    if init_check_index > base_check_index+4 { 
        let credibility: f64 = args[base_check_index+4].parse().expect("We already checked that this parsed to f64");
        //I wrote the condition as follows in case an argument is passed in that makes this a NaN
        if !(credibility > 0.0) {panic!("Motif prior threshold must be a valid strictly positive float");} 
        //SAFETY: This modification is made before any inference is done, preventing data races
        let mut w = NECESSARY_MOTIF_IMPROVEMENT.write().expect("This is the only thread accessing this to write, and the mutable reference goes out of scope immediately");
        *w = credibility; 
    }




    let (total_data,data_string): (AllData, String) = AllData::create_inference_data(fasta_file, data_file, output_dir, is_circular, fragment_length, spacing, false, &NULL_CHAR, peak_cutoff).unwrap();


    let data_ref = AllDataUse::new(&total_data).unwrap();

    let data = data_ref.data();

    let background = data_ref.background_ref();

    let save_step = (1+(num_advances/NUM_CHECKPOINT_FILES)).min(1000);
    let capacity: usize = save_step*(NUM_RJ_STEPS+2);

    let steps_per_exchange_attempt: usize = 500;


    let mut rng = rand::thread_rng();
    
    let check: Option<(&str, &str)> = match args.get(init_check_index) { 
        Some(x) => { 
            match args.get(init_check_index+1) {
                Some(y) => Some((x.as_str(), y.as_str())), 
                None => None,
            }
        },
        None => None,
    };
    let maybe_init = match check {

        Some(("meme", meme_file)) => {
            match MotifSet::set_from_meme(&meme_file , &data_ref, MAX_E_VAL, &mut rng) { 
                Err(e) => { 
                    eprintln!("Meme file did not parse. Using random initial condition instead. Reason:\n {}", e);
                    None
                },
                Ok(m) => Some(m),
            }
        },
        Some(("json", json_file)) => {
            match MotifSet::set_from_json(&data_ref, true, true, json_file, &mut rng) { 
                Err(e) => { 
                    eprintln!("Json file did not parse. Using random initial condition instead. Reason:\n {}", e);
                    None
                },
                Ok(m) => Some(m),
            }
        },
        Some(("bincode", bincode_file)) => {
            match MotifSet::set_from_bincode(&data_ref, true, true, bincode_file, &mut rng) { 
                Err(e) => { 
                    eprintln!("Bincode file did not parse. Using random initial condition instead. Reason:\n {}", e);
                    None
                },
                Ok(m) => Some(m),
            }
        },
        _ => None,
    };

    //    pub fn new_parallel_traces<R: Rng+?Sized>(min_thermo_beta: f64, num_intermediate_traces: usize, capacity_per_trace: usize, step_num_estimate: usize, how_to_track: &TrackingOptions, data_ref: &'a AllDataUse<'a>, initial_condition: Option<MotifSet<'a>>, sparse: Option<usize>, rng: &mut R) -> Result<Self, InitializationError> {

    let mut initialization_chains = TemperSetTraces::new_parallel_traces(min_thermo_beta, num_intermediate_traces, capacity, num_advances, TrackingOptions::TrackAllTraces, &data_ref, maybe_init, None, &mut rng).unwrap();



    //run MCMC and make sure that I'm saving and clearing periodically
    

    let mut track_hmc: usize = 0;

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
