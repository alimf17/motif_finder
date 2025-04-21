

use motif_finder::{NULL_CHAR, NUM_CHECKPOINT_FILES, NUM_RJ_STEPS, MAX_E_VAL, MAX_TF_NUM, ECOLI_FREQ};
use motif_finder::{PROPOSE_EXTEND, DIRICHLET_PWM, THRESH, NECESSARY_MOTIF_IMPROVEMENT};
use motif_finder::base::*;

use motif_finder::data_struct::*;

use log::warn;

use clap::{Parser, ValueEnum};

use std::path::*;
use std::time::{Instant};
use std::env;
use std::fs::File;
use std::io::Read;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    
    /// Sets the name of your run. Note that if you want multiple runs to 
    /// be considered in parallel, use the syntax `<name>_<letter starting from A>`
    /// for your names. If you want runs to be considered as sequential continuations, 
    /// use the syntax `<name with possible letter>_<number starting from 0>`
    #[arg(short, long)]
    name: String,

    /// Sets the input file from preprocessing for your run to infer on
    #[arg(short, long)]
    input: String,

    /// Sets the output directory for your run 
    #[arg(short, long)]
    output: String,

    /// Number of advances you want the algorithm to run for. Note, this is 
    /// in units of number of parallel tempering swaps, which require a 
    /// number of standard Monte Carlo steps beforehand
    #[arg(short, long)]
    advances: usize,

    /// The minimum thermodynamic beta for the chains. The absolute value of 
    /// this will be taken, and if you give a number > 1.0, the reciprocal 
    /// will be taken instead. I have personally seen success with beta = 1/64
    /// when I have 126 intermediate traces. If your chain is accepting too
    /// many swaps and not having high acceptance in the smallest beta threads,
    /// make this closer to 0. If your chain is not swapping and has many 
    /// acceptances in the smallest beta threads, make this closer to 1.
    #[arg(short, long)]
    beta: f64, 

    /// The number of intermediate traces between the beta = 1.0 thread and 
    /// the minimum beta you supplied. This number + 2 is also the maximum number
    /// of parallel threads we can use productively.
    #[arg(short, long)]
    trace_num: usize,

    #[arg(short, long, group="initial")]
    condition_type: Option<InitialType>,

    #[arg(requires="initial")]
     #[arg(short, long)]
    file_initial: Option<String>,

    /// This sets an initial guess on the number of transcription factors.
    /// It will be ignored if you supply a valid initial condition
    #[arg(short, long)]
    starting_tf: Option<usize>,
}

#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, ValueEnum)]
enum InitialType {

    /// Initial condition is the name of a meme file
    Meme,
    /// Initial condition is the name of a serialized bincode of a StrippedMotifSet 
    Bincode,
    /// Initial condition is the name of a serialized JSON of a StrippedMotifSet 
    Json,

}

fn main() {

    //Must have the following arguments:

    //1) Name of run -run , obligate
    //2) Output directory -run , obligate 
    //3)   Name of prepped filed -run, obligate
    //8) Number of advances to run the set trace -run , obligate 
    //9) The minimum thermodynamic beta for the chains. The absolute value of this will be taken, and if you give a number > 1.0, the reciprocal will be taken instead -run , obligate
    //10) The number of intermediate traces between the base inference chain and the minimum thermodynamic beta chain. Must be a non-negative integer -run , obligate


    //Penultimate argument) Either the word "meme" or the word "json". If one of these is not the argument,
    //   no other arguments will be considered. -run , optional  
    //Final argument) The name of the initial condition file. Can either be a meme or json file. If 
    //   it is a JSON file containing the SetTraceDef, the name of the processed data
    //   as implied by the Fasta and data files will be checked against the name in the SetTraceDef.  
    //   If they're not identical, the program will panic on the spot -run , optional 

    let Cli { name: run_name, input: data_file, output: output_dir, advances: num_advances, beta: mut min_thermo_beta, trace_num: num_intermediate_traces, condition_type: initial_type, file_initial: initial_file, starting_tf}= Cli::parse();

    min_thermo_beta = min_thermo_beta.abs();

    println!("pre min {min_thermo_beta}");
    if min_thermo_beta > 1.0 {
        warn!("Your min thermodynamic beta was set > 1.0! We're taking the reciprocal!");
        min_thermo_beta = 1.0/min_thermo_beta;
    }
    println!("post min {min_thermo_beta}");

    let mut data_file_handle = File::open(data_file.as_str()).expect("You initialization file must be valid for inference to work!");

    let mut buffer: Vec<u8> = Vec::new();

    _ = data_file_handle.read_to_end(&mut buffer).expect("Something went wrong when reading the data input file!");

    let mut total_data: AllData = bincode::deserialize(&buffer).expect("Something was incorrect with your saved data input bincode file!");



    println!("have all data");

    let data_ref = AllDataUse::new(&total_data, 0.0).unwrap();

    let steps_per_exchange_attempt: usize = 100;
    
    let save_step = 10usize;
    
    let pushes = num_advances/steps_per_exchange_attempt + ((num_advances % steps_per_exchange_attempt) > 0) as usize;
    
    let capacity: usize = save_step*(steps_per_exchange_attempt);

    let mut rng = rand::thread_rng();

    let mut maybe_init = match (initial_type, initial_file) {

        (None, _) => None,
        (_, None) => {
            eprintln!("Did not supply a file name with your initial condition type! Ignoring option and initializing from a random state.");
            None
        },
        (Some(InitialType::Meme), Some(meme_file)) => {
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
        (Some(InitialType::Json), Some(json_file)) => {
            println!("json match");
            match MotifSet::set_from_json(&data_ref, &json_file, &mut rng) { 
                Err(e) => { 
                    eprintln!("Json file did not parse. Using random initial condition instead. Reason:\n {}", e);
                    None
                },
                Ok(m) => Some(m),
            }
        },
        (Some(InitialType::Bincode), Some(bincode_file)) => {
            println!("bin match");
            match MotifSet::set_from_bincode(&data_ref, &bincode_file, &mut rng) { 
                Err(e) => { 
                    eprintln!("Bincode file did not parse. Using random initial condition instead. Reason:\n {}", e);
                    None
                },
                Ok(m) => Some(m),
            }
        },
        _ => None,
    };

    if maybe_init.is_none() {
        if let Some(tf_num) = starting_tf {
            maybe_init = Some(MotifSet::rand_with_n_motifs(tf_num, &data_ref, &mut rng));
        }
    }

    println!("pre initial");





    let mut initialization_chains = TemperSetTraces::new_parallel_traces(min_thermo_beta, num_intermediate_traces, capacity, num_advances, TrackingOptions::TrackAllTraces, data_file, &data_ref, maybe_init, None, &mut rng).unwrap();



    println!("initialized");
    //run MCMC and make sure that I'm saving and clearing periodically




    let mut step_penalty = 0usize;

    let abort_saving = save_step*10;
    
    let start_inference_time = Instant::now();

    for step in 0..pushes {

        println!("push {step}");
        
        initialization_chains.iter_and_swap(steps_per_exchange_attempt, rand::thread_rng);

        if step % save_step == 0 {

            let mut try_save_trace: usize = 0;
            let save_trial: usize = 5;
            while try_save_trace < save_trial {
            
                let Err(e) = initialization_chains.save_trace_and_clear(&output_dir, &run_name, step-step_penalty) else {step_penalty = 0; break;};
                try_save_trace += 1;
                eprintln!("Error trying to save trace in step {step}: {:?}. Times occured: {try_save_trace}", e);
                if try_save_trace >= save_trial {
                    eprintln!("Aborting attempt to save step {step}. Will keep inference and try again next step.");
                    step_penalty += save_step;
                    if step_penalty >= abort_saving { panic!("Haven't saved for too many steps! Something is wrong with the files to save to! Aborting inference."); }
                }
            }
        
        }
    
    }

    println!("Finished run in {:?}", start_inference_time.elapsed());




}
