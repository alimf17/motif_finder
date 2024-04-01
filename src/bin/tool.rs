
use motif_finder::{NULL_CHAR, NUM_CHECKPOINT_FILES, NUM_RJ_STEPS, NUM_HMC_STEPS, MAX_E_VAL};
use motif_finder::{PROPOSE_EXTEND, DIRICHLET_PWM, THRESH, NECESSARY_MOTIF_IMPROVEMENT, MOMENTUM_SD, HMC_EPSILON, MOMENTUM_DIST};
use motif_finder::base::*;
use motif_finder::waveform::*;
use motif_finder::data_struct::*;
use motif_finder::modified_t::SymmetricBaseDirichlet;

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
    //May also have the following arguments:

    //9) A non negative double for how much to scale the cutoff for a data block
    //   to be considered peaky: less than 1.0 is more permissive, greater than 1.0 
    //   is more strict
    //10) A strictly positive double for alpha of a symmetric dirichlet distribution from which we draw base
    //   extensions and contractions on a single motif Will only work if (9) included
    //11) A strictly positive double for alpha of a symmetric dirichlet distribution from which we draw
    //    base identities for entirely NEW PWMs. Will only work if (9) and (10) included
    //12) A strictly positive double less than 1.0 indicating how weak a motif needs to be in a
    //    certain position before giving up drawing it. Will only work if (9), (10), and (11) included 
    //13) A strictly positive double indicating how much ln likelihood a PWM needs to bring before
    //    being considered. Will only work if (9), (10), (11), and (12) included
    //
    //Penultimate argument) Either the word "meme" or the word "json". If one of these is not the argument,
    //   no other arguments will be considered.
    //Final argument) The name of the initial condition file. Can either be a meme or json file. If 
    //   it is a JSON file containing the SetTraceDef, the name of the processed data
    //   as implied by the Fasta and data files will be checked against the name in the SetTraceDef. 
    //   If they're not identical, the program will panic on the spot
    let args: Vec<String> = env::args().collect();

    if args.len() < 9 {
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
 

    
    let mut check_for_init: bool = true;
    let mut init_check_index: usize = 9;

    while check_for_init {
        match args.get(init_check_index) {
            None => check_for_init = false,
            Some(x) => if x.parse::<f64>().is_ok() { init_check_index += 1;} else {check_for_init = false;},
        };
    }

    //By the end of this block, init_check_index holds the index where we check what type of
    //initial condition we have, if any. If this is larger than 9, then we have arguments to parse
    //that change statics relevant for inference.
    //

    let mut peak_cutoff: Option<f64> = None;

    let initialize_func = |a: &f64| { SymmetricBaseDirichlet::new(*a).unwrap() };
    let run_name;
    if init_check_index > 9 {
        peak_cutoff = Some(args[9].parse().expect("We already checked that this parsed to f64"));
        run_name = format!("{}_custom_scale_{}", args[1].as_str(), args[9]);
    } else {
        run_name = args[1].clone();
    }
    if init_check_index > 10 { 
        let extend_alpha: f64 = args[10].parse().expect("We already checked that this parsed to f64");
        //I wrote the condition as follows in case an argument is passed in that makes this a NaN
        if !(extend_alpha > 0.0) {panic!("Dirichlet alpha must be a valid strictly positive float");} 
        //let mut w = PROPOSE_EXTEND.write().expect("This is the only thread accessing this to write, and the mutable reference goes out of scope immediately");
        //*w = Lazy::new(|&extend_alpha| SymmetricBaseDirichlet::new(extend_alpha.clone()).unwrap());
        PROPOSE_EXTEND.set(SymmetricBaseDirichlet::new(extend_alpha).expect("checked parameter validity already")).expect("Nothing should have written to this before now");
    } else {

        PROPOSE_EXTEND.set(SymmetricBaseDirichlet::new(1.0_f64).expect("obviously valid")).expect("Nothing should have written to this before now");
    }
    if init_check_index > 11 { 
        let pwm_alpha: f64 = args[11].parse().expect("We already checked that this parsed to f64");
        //I wrote the condition as follows in case an argument is passed in that makes this a NaN
        if !(pwm_alpha > 0.0) {panic!("Dirichlet alpha must be a valid strictly positive float");} 
        //SAFETY: This modification is made before any inference is done, preventing data races
        //let mut w = DIRICHLET_PWM.write().expect("This is the only thread accessing this to write, and the mutable reference goes out of scope immediately");
        //*w = Lazy::new(|| SymmetricBaseDirichlet::new(pwm_alpha).unwrap());

        DIRICHLET_PWM.set(SymmetricBaseDirichlet::new(pwm_alpha).expect("checked parameter validity already")).expect("Nothing should have written to this before now");
    } else {

        DIRICHLET_PWM.set(SymmetricBaseDirichlet::new(10.0_f64).expect("obviously valid")).expect("Nothing should have written to this before now");
    }


    if init_check_index > 12 { 
        let threshold: f64 = args[12].parse().expect("We already checked that this parsed to f64");
        //I wrote the condition as follows in case an argument is passed in that makes this a NaN
        if !(threshold > 0.0) || (threshold >= 1.0) {panic!("Peak drawing threshold must be a valid strictly positive float strictly less than 1.0");} 
        //SAFETY: This modification is made before any inference is done, preventing data races
        let mut w = THRESH.write().expect("This is the only thread accessing this to write, and the mutable reference goes out of scope immediately");
        *w = threshold; 
    }
    if init_check_index > 13 { 
        let credibility: f64 = args[13].parse().expect("We already checked that this parsed to f64");
        //I wrote the condition as follows in case an argument is passed in that makes this a NaN
        if !(credibility > 0.0) {panic!("Motif prior threshold must be a valid strictly positive float");} 
        //SAFETY: This modification is made before any inference is done, preventing data races
        let mut w = NECESSARY_MOTIF_IMPROVEMENT.write().expect("This is the only thread accessing this to write, and the mutable reference goes out of scope immediately");
        *w = credibility; 
    }

    if init_check_index > 14 { 
        let momentum: f64 = args[14].parse().expect("We already checked that this parsed to f64");
        //I wrote the condition as follows in case an argument is passed in that makes this a NaN
        //SAFETY: This modification is made before any inference is done, preventing data races
        //let mut w = DIRICHLET_PWM.write().expect("This is the only thread accessing this to write, and the mutable reference goes out of scope immediately");
        //*w = Lazy::new(|| SymmetricBaseDirichlet::new(pwm_alpha).unwrap());

        if !(momentum > 0.0) {panic!("Momentum distribution standard deviation must be a valid strictly positive float");}
        
        let mut w = MOMENTUM_SD.write().expect("This is the only thread accessing this to write, and the mutable reference goes out of scope immediately");
        *w = momentum;
    } 


    if init_check_index > 15 { 
        let eps: f64 = args[15].parse().expect("We already checked that this parsed to f64");
        //I wrote the condition as follows in case an argument is passed in that makes this a NaN
        if !(eps > 0.0) {panic!("HMC epsilon must be a valid strictly positive float");} 
        //SAFETY: This modification is made before any inference is done, preventing data races
        //let mut w = DIRICHLET_PWM.write().expect("This is the only thread accessing this to write, and the mutable reference goes out of scope immediately");
        //*w = Lazy::new(|| SymmetricBaseDirichlet::new(pwm_alpha).unwrap());

        {
        let f = HMC_EPSILON.write();
        let mut m = f.expect("This is the only thread accessing this to write, and the mutable reference goes out of scope immediately");
        *m = eps;
        }
    }


    MOMENTUM_DIST.set(Normal::new(0.0, *MOMENTUM_SD.read().expect("Nothing should be writing to this now")).expect("checked parameter validity already")).expect("Nothing should have written to this before now");



    let (total_data,data_string): (AllData, String) = AllData::create_inference_data(fasta_file, data_file, output_dir, is_circular, fragment_length, spacing, false, &NULL_CHAR, peak_cutoff).unwrap();


    let data_ref = AllDataUse::new(&total_data).unwrap();

    let data = data_ref.data();

    let background = data_ref.background_ref();

    let save_step = (1+(num_advances/NUM_CHECKPOINT_FILES)).min(1000);
    let capacity: usize = save_step*(NUM_RJ_STEPS+NUM_HMC_STEPS+2);



    let mut rng = rand::thread_rng();
    
    //Initialize trace
    let mut current_trace: SetTrace = SetTrace::new_trace(capacity,data_string.clone(), InitializeSet::Rng(&mut rng), &data_ref, None);
    
    let check: Option<&str> = match args.get(init_check_index) { Some(x) => Some(x.as_str()), None => None};
    match check {

        Some("meme") => current_trace.trace_from_meme(args.get(init_check_index+1).expect("Must include a string indicating MEME output file").as_str(),data.seq(), MAX_E_VAL, &mut rng),
        Some("json") => {
            //If you're picking up from a JSON, you have the right to tell the motif finder to trust your motif set and minimize recalculation.
            //Note that it doesn't COMPLETELY trust you: it will validate on making sure your motif set signal is compatible with your processed ChIP data regardless. 
            //The default behavior is that it doesn't trust you.
            let (validate, mut maybe_rng) = match args.get(init_check_index+1).map(|x| x.parse::<bool>().ok()).flatten() { 
                None | Some(false) => (true, Some(&mut rng)),
                Some(true) => (false, None),
            };
            current_trace.push_last_state_from_json(validate, validate, &mut maybe_rng, args.get(init_check_index+1).expect("Must inlcude a string indicating a Json output file").as_str());
        },
        Some("bincode") => {
            let (validate, mut maybe_rng) = match args.get(init_check_index+1).map(|x| x.parse::<bool>().ok()).flatten() {
                None | Some(false) => (true, Some(&mut rng)),
                Some(true) => (false, None),
            };
            println!("bincode");
            current_trace.push_last_state_from_bincode(validate, validate, &mut maybe_rng, args.get(init_check_index+1).expect("Must inlcude a string indicating a bincode output file").as_str());
        }
        _ => (),
    };

    println!("init {:?}", current_trace.current_set_to_print());
    //run MCMC and make sure that I'm saving and clearing periodically
    

    let mut track_hmc: usize = 0;

    let start_inference_time = Instant::now();

    let eps_sizes = [0.04_f64, 0.004, 0.0004];
    let momentum_sds = [0.1, 1_f64, 10.];
    let base_ratio_sds = [0.1_f64, 0.5, 1.];
    let base_linear_sds = [0.1_f64, 0.5, 1.];
    let height_sds = [0.1, 1_f64, 2.0];
    let mut attempts_per_move = vec![0_usize; eps_sizes.len()*momentum_sds.len()+2*base_ratio_sds.len()*base_linear_sds.len()+height_sds.len()+6];
    let mut successes_per_move = vec![0_usize; eps_sizes.len()*momentum_sds.len()+2*base_ratio_sds.len()*base_linear_sds.len()+height_sds.len()+6];
    let mut immediate_failures_per_move = vec![0_usize; eps_sizes.len()*momentum_sds.len()+2*base_ratio_sds.len()*base_linear_sds.len()+height_sds.len()+6];
    let mut distances_per_attempted_move = vec![Vec::<[f64; 4]>::with_capacity(num_advances/20); eps_sizes.len()*momentum_sds.len()+2*base_ratio_sds.len()*base_linear_sds.len()+height_sds.len()+6];

    //for step in 0..10000 {
 
    for step in 0..num_advances {
        current_trace.advance(&eps_sizes, &momentum_sds, &base_ratio_sds, &base_linear_sds, &height_sds, 
                              &mut attempts_per_move, &mut successes_per_move, &mut immediate_failures_per_move, 
                              &mut distances_per_attempted_move, &mut rng);

        /*if step % 10 == 0 {
            println!("Step {}. Trials/acceptences/acceptance rates for {:?}, base leaping, and HMC, respectively are: {:?}/{:?}/{:?}", step, RJ_MOVE_NAMES, trials, acceptances, rates);
                println!("Not really changing motif???");
                println!("{:?}", current_trace.current_set_to_print());
        }*/
        if step % save_step == 0 {
            
            current_trace.save_trace(output_dir, &run_name, step);
            
            if step != 0 {
                current_trace.save_and_drop_history(output_dir, &run_name, step);
            }
        }

    }

    //let init_sd: f64 = MOMENTUM_SD.read().expect("Nothing should write to this right now");
    /*let (number_burn_in, new_sd) = current_trace.burn_in_momentum(*(MOMENTUM_SD.read().expect("Nothing should write to this right now")), &mut rng);

    println!("Momentum burn in took {} trials to stabilize to sd of {}", number_burn_in, new_sd);

    let new_momentum_dist = Normal::new(0.0, new_sd).unwrap();

    for step in 0..num_advances {
 
        let (selected_move, accepted) = current_trace.advance(&new_momentum_dist, &mut rng);

        trials[selected_move] += 1;
        if accepted {acceptances[selected_move] += 1;}
        rates[selected_move] = (acceptances[selected_move] as f64)/(trials[selected_move] as f64);

        //println!("Step {} ", step);
        if step % 10 == 0 {
            println!("Step {}. Trials/acceptences/acceptance rates for {:?}, base leaping, and HMC, respectively are: {:?}/{:?}/{:?}", step, RJ_MOVE_NAMES, trials, acceptances, rates);
            if (acceptances[5]-track_hmc) == 0 {
                println!("Not really changing motif???");
                println!("{:?}", current_trace.current_set_to_print());
            }
            track_hmc=acceptances[5];
        }
        if step % save_step == 0 {
            
            current_trace.save_trace(output_dir, &run_name, step+10000);
            current_trace.save_and_drop_history(output_dir, &run_name, step+10000);

        }

    }
*/
    println!("Finished run in {:?}", start_inference_time.elapsed());

  


}
