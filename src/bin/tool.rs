

use motif_finder::{NULL_CHAR, NUM_CHECKPOINT_FILES, NUM_RJ_STEPS, MAX_E_VAL, MAX_TF_NUM, ECOLI_FREQ};
use motif_finder::{PROPOSE_EXTEND, DIRICHLET_PWM, THRESH, NECESSARY_MOTIF_IMPROVEMENT};
use motif_finder::base::*;

use motif_finder::data_struct::*;
use motif_finder::modified_t::SymmetricBaseDirichlet;

use log::warn;







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

    //11) The number of expected binding sites, approximately

    //12) A non negative double for how much to scale the cutoff for a data block
    //   to be considered peaky: less than 1.0 is more permissive, greater than 1.0 
    //   is more strict. If this exists but can't parse to a double, it's turned into a 1.0
    //May also have the following arguments:
    //13) A strictly positive double for alpha of a symmetric dirichlet distribution from which we draw base
    //   extensions and contractions on a single motif Will only work if (9) included
    //14) A strictly positive double for alpha of a symmetric dirichlet distribution from which we draw
    //    base identities for entirely NEW PWMs. Will only work if (9) and (10) included
    //15) A strictly positive double less than 1.0 indicating how weak a motif needs to be in a
    //    certain position before giving up drawing it. Will only work if (9), (10), and (11) included 
    //16) A strictly positive double indicating how much ln likelihood a PWM needs to bring before
    //    being considered. Will only work if (9), (10), (11), and (12) included
    //
    //Penultimate argument) Either the word "meme" or the word "json". If one of these is not the argument,
    //   no other arguments will be considered.
    //Final argument) The name of the initial condition file. Can either be a meme or json file. If 
    //   it is a JSON file containing the SetTraceDef, the name of the processed data
    //   as implied by the Fasta and data files will be checked against the name in the SetTraceDef. 
    //   If they're not identical, the program will panic on the spot
    let args: Vec<String> = env::args().collect();

    if args.len() < 12 {
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

    let num_intermediate_traces: usize = args[10].parse().expect("The number of intermediate traces must be a non-negative integer!");

    let min_height: f64 = args[11].parse().expect("The minimum peak height must be floating point number!");

    min_thermo_beta = min_thermo_beta.abs();

    println!("pre min {min_thermo_beta}");
    if min_thermo_beta > 1.0 {
        warn!("Your min thermodynamic beta was set > 1.0! We're taking the reciprocal!");
        min_thermo_beta = 1.0/min_thermo_beta;
    }
    println!("post min {min_thermo_beta}");

    let credibility: f64 = args[12].parse().expect("We need to know how much to punish additional motifs!");

    println!("c {credibility}");
    if !(credibility > 0.0) {panic!("Motif prior threshold must be a valid strictly positive float");}
    //SAFETY: This modification is made before any inference is done, preventing data races
    unsafe{ NECESSARY_MOTIF_IMPROVEMENT = credibility; }

    let filter: f64 = match args.get(13) {
        None => 0.0,
        Some(arg) => arg.parse::<f64>().unwrap_or(0.0),
    };

    let max_tf: usize = args.get(14).map(|a| a.parse().ok()).flatten().unwrap_or(5);

    //MAX_TF_NUM.set(max_tf).unwrap();

    MAX_TF_NUM.set(200).unwrap();

    println!("Max TF num {:?}", MAX_TF_NUM);

    let default_burn: usize = 10;

    let (burn_in_after_swap,value): (usize, bool) = match args.get(15) {
        None => (default_burn, false),
        Some(burn) => match burn.parse::<usize>(){
            Err(_) => (default_burn, false), 
            Ok(burning) => (burning, true),
        },
    };



    let base_check_index: usize = if value {16} else {15};
    let mut init_check_index: usize = base_check_index+1;


    println!("init {:?}", args.get(init_check_index));

    //By the end of this block, init_check_index holds the index where we check what type of
    //initial condition we have, if any. If this is larger than base_check_index, then we have arguments to parse
    //that change statics relevant for inference.
    //

    let mut peak_cutoff: Option<f64> = None;

    let run_name;
    peak_cutoff = args.get(base_check_index).map(|a| a.parse().ok()).flatten();
    run_name = match peak_cutoff{
        None => {
            peak_cutoff = None;
            args[1].clone()
        },
        Some(a) => 
            if a == 1.0 { 
                peak_cutoff = None;
                args[1].clone()
            } else{
                format!("{}_custom_scale_{}", args[1].as_str(), args[9])
            },
    };

    println!("filt {} cut {:?} {run_name}", filter, peak_cutoff);
    PROPOSE_EXTEND.set(SymmetricBaseDirichlet::new(1.0_f64).expect("obviously valid")).expect("Nothing should have written to this before now");
    DIRICHLET_PWM.set(SymmetricBaseDirichlet::new(1.0_f64).expect("obviously valid")).expect("Nothing should have written to this before now");

    println!("Args parsed");

    println!("rayon {:?}", rayon::current_num_threads());

    println!("output directory {output_dir}");

    let (mut total_data,data_string): (AllData, String) = AllData::create_inference_data(fasta_file, data_file, output_dir, is_circular, fragment_length, spacing, min_height, &NULL_CHAR, peak_cutoff).unwrap();

    total_data.clear_props();


    println!("have all data");

    let data_ref = AllDataUse::new(&total_data, filter).unwrap();

    let _data = data_ref.data();

    let background = data_ref.background_ref();

    println!("min_height {}", data_ref.min_height());
    println!("background: {:?}", background);

    let save_step = (1+(num_advances/NUM_CHECKPOINT_FILES)).min(1000);
    let capacity: usize = save_step*(NUM_RJ_STEPS+2);

    let steps_per_exchange_attempt: usize = 10;
    //let steps_per_exchange_attempt: usize = 1;


    let mut rng = rand::thread_rng();

    let check: Option<(&str, &str)> = match args.get(init_check_index-1) { 
        Some(x) => { 
            match args.get(init_check_index) {
                Some(y) => Some((x.as_str(), y.as_str())), 
                None => None,
            }
        },
        None => None,
    };
    
    println!("checker {:?}", check);

    let maybe_init = match check {

        Some(("meme", meme_file)) => {
            println!("meme match");
            match MotifSet::set_from_meme(&meme_file , &data_ref, Some(ECOLI_FREQ),MAX_E_VAL, true, &mut rng) { 
                Err(e) => {
                    println!("Meme file did not parse. Using random initial condition instead. Reason:\n {}", e);
                    eprintln!("Meme file did not parse. Using random initial condition instead. Reason:\n {}", e);
                    None
                },
                Ok(m) => Some(m),
            }
        },
        Some(("json", json_file)) => {
            println!("json match");
            match MotifSet::set_from_json(&data_ref, json_file, &mut rng) { 
                Err(e) => { 
                    eprintln!("Json file did not parse. Using random initial condition instead. Reason:\n {}", e);
                    None
                },
                Ok(m) => Some(m),
            }
        },
        Some(("bincode", bincode_file)) => {
            println!("bin match");
            match MotifSet::set_from_bincode(&data_ref, bincode_file, &mut rng) { 
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

    println!("pre initial");



    //pub fn new_trace<R: Rng + ?Sized>(capacity: usize, initial_condition: Option<MotifSet<'a>>, all_data_file: String,data_ref: &'a AllDataUse<'a>, mut thermo_beta: f64, null_attention: Vec<usize>, sparse: Option<usize>, rng: &mut R) -> SetTrace<'a> {


    let pushes = num_advances/steps_per_exchange_attempt + ((num_advances % steps_per_exchange_attempt) > 0) as usize;


    let mut initialization_chains = TemperSetTraces::new_parallel_traces(min_thermo_beta, num_intermediate_traces, capacity, num_advances, TrackingOptions::TrackAllTraces, data_string, &data_ref, maybe_init, None, &mut rng).unwrap();



    println!("initialized");
    //run MCMC and make sure that I'm saving and clearing periodically


    let _track_hmc: usize = 0;

    let start_inference_time = Instant::now();

    //for step in 0..10000 {

    let pushes = num_advances/steps_per_exchange_attempt + ((num_advances % steps_per_exchange_attempt) > 0) as usize;

    let mut step_penalty = 0usize;

    let step_checker = 5usize;

    let abort_saving = step_checker*10;

    for step in 0..pushes {

        println!("push {step}");
        //initialization_chains.iter_and_swap(1, 1, 1, rand::thread_rng);
        initialization_chains.iter_and_swap(10, steps_per_exchange_attempt, burn_in_after_swap, rand::thread_rng);

        if step % step_checker == 0 {

            initialization_chains.print_acceptances(TrackingOptions::TrackAllTraces);

            //  if ((step+1) % 5 == 0) || (step+1 == pushes) {
            //  }

            /*if step % 50 == 0 {
                let root_signal: String = format!("{}/{}_dist_of",output_dir,run_name);

                let num_bins: usize = 100;

                initialization_chains.handle_histograms(TrackingOptions::TrackAllTraces, &root_signal, num_bins);

            }*/
       
            let mut try_save_trace: usize = 0;
            let save_trial: usize = 5;
            while try_save_trace < save_trial {
            
                let Err(e) = initialization_chains.save_trace_and_clear(output_dir, &run_name, step-step_penalty) else {step_penalty = 0; break;};
                try_save_trace += 1;
                eprintln!("Error trying to save trace in step {step}: {:?}. Times occured: {try_save_trace}", e);
                if try_save_trace >= save_trial {
                    eprintln!("Aborting attempt to save step {step}. Will keep inference and try again next step.");
                    step_penalty += step_checker;
                    if step_penalty >= abort_saving { panic!("Haven't saved for too many steps! Something is wrong with the files to save to! Aborting inference."); }
                }
            }
        
        }
    
    }

    println!("Finished run in {:?}", start_inference_time.elapsed());




}
