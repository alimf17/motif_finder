
use motif_finder::{NULL_CHAR, NUM_CHECKPOINT_FILES, NUM_RJ_STEPS, MAX_E_VAL};
use motif_finder::{PROPOSE_EXTEND, DIRICHLET_PWM, THRESH, NECESSARY_MOTIF_IMPROVEMENT};
use motif_finder::base::*;
use motif_finder::waveform::*;
use motif_finder::data_struct::*;
use motif_finder::modified_t::SymmetricBaseDirichlet;

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
        warn!("Your min thermodynamic beta was set > 1.0! We're taking the reciprocal!")
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
        peak_cutoff = Some(args[base_check_index].parse().expect("We already checked that this parsed to f64"));
        run_name = format!("{}_custom_scale_{}", args[1].as_str(), args[9]);
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

        PROPOSE_EXTEND.set(SymmetricBaseDirichlet::new(10.0_f64).expect("obviously valid")).expect("Nothing should have written to this before now");
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

        DIRICHLET_PWM.set(SymmetricBaseDirichlet::new(10.0_f64).expect("obviously valid")).expect("Nothing should have written to this before now");
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



    let mut rng = rand::thread_rng();
    
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

    //    pub fn new_parallel_traces<R: Rng+?Sized>(min_thermo_beta: f64, num_intermediate_traces: usize, capacity_per_trace: usize, step_num_estimate: usize, how_to_track: &TrackingOptions, data_ref: &'a AllDataUse<'a>, initial_condition: Option<MotifSet<'a>>, sparse: Option<usize>, rng: &mut R) -> Result<Self, InitializationError> {

    let mut initialization_chains = TemperSetTraces::new_parallel_traces(min_thermo_beta, num_intermediate_traces, capacity, num_advances, &TrackingOptions::TrackAllTraces, &data_ref, maybe_init, None, &mut rng).unwrap();


    println!("init {:?}", current_trace.current_set_to_print());
    //run MCMC and make sure that I'm saving and clearing periodically
    

    let mut track_hmc: usize = 0;

    let start_inference_time = Instant::now();

    //for step in 0..10000 {
 
    for step in 0..num_advances {
        current_trace.advance(&base_ratio_sds, &base_linear_sds, &height_sds, 
                              &mut attempts_per_move, &mut successes_per_move, &mut immediate_failures_per_move, 
                              &mut distances_per_attempted_move, &mut rng);

        if step % 100 == 0 {

            //Do all available MoveTracker prints here

            if ((step >= 10000) && ((step % 10000) == 0)) || (step+1 == num_advances){
            
                let root_signal: String = format!("{}/{}_step_{:0>7}_dist_of",output_dir,run_name, step);

                let num_bins: usize = 100;
                
                let mut ind = 0_usize;
                //TODO: 
                //Move histograms:
                //Each histogram needs 8 panels apiece, split into two batches of four:
                //      one for acccepted moves, the other for attempted failures.
                //      1) RMSE, 2) Finite likelihood differences (with a blur of text indicating proportion of infinities)
                //      3) Euclidean distance of heights 4) Euclidean distances of PWMs

                //All necessary data is found in the distances_per_attempted_move: Vec<Vec<([f64; 4], bool)>>

                //Base move and motif move each need #base ratio sds * #base linear sds
                for i in 0..base_ratio_sds.len(){
                    for j in 0..base_linear_sds.len() {
                        let file_string = format!("{}_base_scale_ratio_sd_{}_linear_sd_{}.png", root_signal, base_ratio_sds[i], base_linear_sds[j]);
                        let plotting = BitMapBackend::new(&file_string, (5000, 3000)).into_drawing_area();
                        let check = sort_move_hists(&distances_per_attempted_move[ind], &plotting, num_bins); 
                        if let Err(err) = check {println!("{}", err);};
                        ind += 1;
                        //TODO: Generate histograms for accepted and failed
                    }
                }
                for i in 0..base_ratio_sds.len(){
                    for j in 0..base_linear_sds.len() {
                        let file_string = format!("{}_motif_scale_ratio_sd_{}_linear_sd_{}.png", root_signal, base_ratio_sds[i], base_linear_sds[j]);
                        let plotting = BitMapBackend::new(&file_string, (5000, 3000)).into_drawing_area();
                        let check = sort_move_hists(&distances_per_attempted_move[ind], &plotting, num_bins); 
                        if let Err(err) = check {println!("{}", err);};
                        ind += 1;
                        //TODO: Generate histograms for accepted and failed
                    }
                }
                //Height moves needs # of height sds
                for i in 0..height_sds.len() {
                    let file_string = format!("{}_height_sd_{}.png", root_signal, height_sds[i]);
                    let plotting = BitMapBackend::new(&file_string, (5000, 3000)).into_drawing_area();
                    let check = sort_move_hists(&distances_per_attempted_move[ind], &plotting, num_bins); 
                    if let Err(err) = check {println!("{}", err);};
                    ind += 1;
                    //TODO: Generate histograms for accepted and failed
                }
                let file_string = format!("{}_motif_birth.png", root_signal);
                let plotting = BitMapBackend::new(&file_string, (5000, 3000)).into_drawing_area();
                let check = sort_move_hists(&distances_per_attempted_move[ind], &plotting, num_bins); 
                if let Err(err) = check {println!("{}", err);};
                ind += 1;
                //TODO: Generate histograms for accepted and failed
                let file_string = format!("{}_motif_death.png", root_signal);
                let plotting = BitMapBackend::new(&file_string, (5000, 3000)).into_drawing_area();
                let check = sort_move_hists(&distances_per_attempted_move[ind], &plotting, num_bins); 
                ind += 1;
                //TODO: Generate histograms for accepted and failed

                //PWM expand and contract are fairly normal, all things considered
                let file_string = format!("{}_motif_expand.png", root_signal);
                let plotting = BitMapBackend::new(&file_string, (5000, 3000)).into_drawing_area();
                let check = sort_move_hists(&distances_per_attempted_move[ind], &plotting, num_bins); 
                if let Err(err) = check {println!("{}", err);};
                ind += 1;
                //TODO: Generate histograms for accepted and failed
                let file_string = format!("{}_motif_contract.png", root_signal);
                let plotting = BitMapBackend::new(&file_string, (5000, 3000)).into_drawing_area();
                let check = sort_move_hists(&distances_per_attempted_move[ind], &plotting, num_bins); 
                if let Err(err) = check {println!("{}", err);};
                ind += 1;
                //TODO: Generate histograms for accepted and failed

                //The two gibbs moves do not need failure panels
                let file_string = format!("{}_base_leap.png", root_signal);
                let plotting = BitMapBackend::new(&file_string, (5000, 3000)).into_drawing_area();
                let check = sort_move_hists(&distances_per_attempted_move[ind], &plotting, num_bins); 
                if let Err(err) = check {println!("{}", err);};
                ind += 1;
                //TODO: Generate histograms for accepted
                let file_string = format!("{}_secondary_shuffle.png", root_signal);
                let plotting = BitMapBackend::new(&file_string, (5000, 3000)).into_drawing_area();
                let check = sort_move_hists(&distances_per_attempted_move[ind], &plotting, num_bins); 
                if let Err(err) = check {println!("{}", err);};
                ind += 1;
                //TODO: Generate histograms for accepted
            }
        }
        if (step+1) % save_step == 0 {
            
            current_trace.save_trace(output_dir, &run_name, step);
            current_trace.save_and_drop_history(output_dir, &run_name, step);




        }

    }

    println!("Finished run in {:?}", start_inference_time.elapsed());

  


}

fn sort_move_hists<DB: DrawingBackend>(data: &Vec<([f64; 4], bool)>, plotting: &DrawingArea<DB, Shift>, num_bins: usize) -> Result<(), String>{

    if data.len() == 0 {return Err("No data for plotting".to_string());}

    plotting.fill(&WHITE).expect("This should just work");

    let (left, right) = plotting.split_horizontally((50).percent_width());

    let lchart = ChartBuilder::on(&left).margin(30).set_label_area_size(LabelAreaPosition::Top, 60).caption("Accepted Moves", ("sans-serif", 50));
    let rchart = ChartBuilder::on(&right).margin(30).set_label_area_size(LabelAreaPosition::Top, 60).caption("All Moves", ("sans-serif", 50));

    let left_subs = left.split_evenly((2,2));
    let right_subs = right.split_evenly((2,2));


    let labs = ["Occupancy Signal RMSE between moves", "Finite Likelihood differences", "Euclidean distance of heights", "Total distance of all PWMs"];

    for (j, area) in right_subs.iter().enumerate() {
        let trial_data = data.iter().map(|(a, _)| a[j]).collect::<Vec<_>>();
        let hist = quick_hist(&trial_data, area, labs[j].clone().to_string(), num_bins);
    }

    for (j, area) in left_subs.iter().enumerate() {
        let trial_data = data.iter().filter(|(_, b)| *b).map(|(a, _)| a[j]).collect::<Vec<_>>();
        let hist = quick_hist(&trial_data, area, labs[j].clone().to_string(), num_bins);
    }

    Ok(())

}

fn quick_hist<'a, 'b, DB: DrawingBackend, N: Copy+Into<f64>>(raw_data: &[N], area: &'a DrawingArea<DB, Shift>, label: String, num_bins: usize) -> ChartBuilder<'a, 'b, DB> {

            let mut hist = ChartBuilder::on(area);

            hist.margin(10).set_left_and_bottom_label_area_size(20);

            hist.caption(label, ("Times New Roman", 20));

            let mut data: Vec<f64> = raw_data.iter().map(|&a| a.into()).filter(|a| a.is_finite()).collect();

            if data.len() > 0 {
            
                let (xs, hist_form) = build_hist_bins(data, num_bins);

                let range = RangedSlice::from(xs.as_slice());

                let max_prob = hist_form.iter().map(|&x| x.1).fold(0_f64, |x,y| x.max(y));

                let mut hist_context = hist.build_cartesian_2d(range, 0_f64..max_prob).unwrap();

                hist_context.configure_mesh().disable_x_mesh().disable_y_mesh().x_label_formatter(&|x| format!("{:.04}", *x)).draw().unwrap();

                //hist_context.draw_series(Histogram::vertical(&hist_context).style(CYAN.filled()).data(trial_data.iter().map(|x| (x, inverse_size)))).unwrap();
                hist_context.draw_series(Histogram::vertical(&hist_context).style(CYAN.filled()).margin(0).data(hist_form.iter().map(|x| (&x.0, x.1)))).unwrap();
            }

            hist
}


fn build_hist_bins(mut data: Vec<f64>, num_bins: usize) -> (Vec<f64>, Vec<(f64, f64)>) {

    let length = data.len() as f64;

    data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let min = data[0];
    let max = *data.last().unwrap();

    let step = (max-min)/(num_bins as f64);

    let add_into = 1./length;

    let xs = (0..num_bins).map(|i| (min+(i as f64)*step)).collect::<Vec<_>>();

    let mut bins: Vec<(f64, f64)> = xs.iter().clone().map(|&x| (x, 0.0)).collect();

    let mut j: usize = 1;

    for &dat in data.iter() {

        //Because data and bins are sorted, we only need to find the first bin
        //where the data is less than the top end. We use short circuit && to prevent overflow 
        while (j < (bins.len()-1)) && (dat >= bins[j+1].0) { j+= 1;}
        bins[j].1 += add_into;
    }

    (xs, bins)

}

