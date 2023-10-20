
use motif_finder::{NULL_CHAR, NUM_CHECKPOINT_FILES, NUM_RJ_STEPS, NUM_HMC_STEPS, MAX_E_VAL};
use motif_finder::base::*;
use motif_finder::waveform::*;
use motif_finder::data_struct::*;

//use std::time::{Duration, Instant};

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
 
    let (total_data,data_string): (AllData, String) = AllData::create_inference_data(fasta_file, data_file, output_dir, is_circular, fragment_length, spacing, false, &NULL_CHAR);

    let data: Waveform = total_data.validated_data();

    let background = total_data.background();

    let save_step = (1+(num_advances/NUM_CHECKPOINT_FILES)).min(1000);
    let capacity: usize = save_step*(NUM_RJ_STEPS+NUM_HMC_STEPS+2);

    //Initialize trace
    let mut current_trace: SetTrace = SetTrace::new_empty(capacity,data_string.clone(), &data, &background);

    let mut rng = rand::thread_rng();
    
    let check: Option<&str> = match args.get(9) { Some(x) => Some(x.as_str()), None => None};
    match check {

        Some("meme") => current_trace.trace_from_meme(args.get(10).expect("Must include a string indicating MEME output file").as_str(),data.seq(), MAX_E_VAL, &mut rng),
        Some("json") => {
            //If you're picking up from a JSON, you have the right to tell the motif finder to trust your motif set and minimize recalculation.
            //Note that it doesn't COMPLETELY trust you: it will validate on making sure your motif set signal is compatible with your processed ChIP data regardless. 
            //The default behavior is that it doesn't trust you.
            let (validate, mut maybe_rng) = match args.get(10).map(|x| x.parse::<bool>().ok()).flatten() { 
                None | Some(false) => (true, Some(&mut rng)),
                Some(true) => (false, None),
            };
            current_trace.push_last_state_from_json(validate, validate, &mut maybe_rng, args.get(10).expect("Must inlcude a string indicating a Json output file").as_str());
        },
        _ => current_trace.push_set(MotifSet::rand_with_one(&data, &background, &mut rng)),
    };

    //run MCMC and make sure that I'm saving and clearing periodically
    

    let mut acceptances: [usize;6]= [0;6];
    let mut trials: [usize;6] = [0;6];
    let mut rates: [f64; 6] = [0.;6];

    let mut track_hmc: usize = 0;

    for step in 0..num_advances {
 
        let (selected_move, accepted) = current_trace.advance(&mut rng);

        trials[selected_move] += 1;
        if accepted {acceptances[selected_move] += 1;}
        rates[selected_move] = (acceptances[selected_move] as f64)/(trials[selected_move] as f64);

        if step % 10 == 0 {
            println!("Step {}. Trials/acceptences/acceptance rates for {:?}, base leaping, and HMC, respectively are: {:?}/{:?}/{:?}", step, RJ_MOVE_NAMES, trials, acceptances, rates);
            if (acceptances[5]-track_hmc) == 0 {
                println!("Not really changing motif???");
                println!("{:?}", current_trace.current_set_to_print());
            }
            track_hmc=acceptances[5];
        }
        if step % save_step == 0 {
            
            current_trace.save_trace(output_dir, run_name, step);
            current_trace.save_and_drop_history(output_dir, run_name, step);

        }

    }


    println!("Finished run");

  


}
