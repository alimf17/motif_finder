use std::fs;
use std::io::Read;

use motif_finder::base::*;
use motif_finder::base::{SQRT_2, SQRT_3, LN_2, BPS};

use motif_finder::{NECESSARY_MOTIF_IMPROVEMENT};

use motif_finder::data_struct::{AllData, AllDataUse};

use log::warn;

use std::sync::Arc;
use rustfft::{FftPlanner, num_complex::Complex};

use ndarray::prelude::*;

use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelIterator;

use regex::Regex;

use poloto;
use poloto::Croppable;
use plotters::prelude::*;
use plotters::prelude::full_palette::*;
use plotters::coord::ranged1d::{NoDefaultFormatting, KeyPointHint,ValueFormatter};

use spectrum_analyzer::{samples_fft_to_spectrum, FrequencyLimit};

use core::ops::Range;


const INTERVAL_CELL_LENGTH: f64 = 0.01;

const UPPER_LETTERS: [char; 26] = [
    'A', 'B', 'C', 'D', 'E',
    'F', 'G', 'H', 'I', 'J', 
    'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'];


const PALETTE: [&RGBColor; 26] = [
    &full_palette::RED, &full_palette::BLUE, &full_palette::GREEN, &YELLOW_700,
    &AMBER, &BLUEGREY, &full_palette::CYAN,  &ORANGE, 
    &PINK, &TEAL, &LIME, &full_palette::YELLOW, 
    &DEEPORANGE, &INDIGO, &CYAN_A700, &YELLOW_A700,
    &BROWN, &BLUEGREY_A700, &LIME_A700, &YELLOW_800, 
    &RED_A700, &BLUE_A700, &GREEN_A700, &YELLOW_A700, &GREY, &full_palette::BLACK];

const BASE_COLORS: [&RGBColor; BASE_L] = [&full_palette::RED, &YELLOW_700, &full_palette::GREEN, &full_palette::BLUE];

pub fn main() {

    let args: Vec<String> = std::env::args().collect();

    if args.len() < 5 {
        panic!("Not enough arguments!");
    }
    
    let out_dir: String = args[1].to_string();
    let base_file: &str = &args[2];
    let mut num_chains: usize = args[3].parse().expect("The number of chains you have should be numeric!");
    let mut max_chain: usize = args[4].parse().expect("The number of strung together inferences you have per chain should be numeric!");
    let burn_in: Option<usize> = args.get(5).map(|a| a.parse::<usize>().ok()).flatten();
    let scale_thresh: Option<f64> = args.get(6).map(|a| a.parse::<f64>().ok()).flatten();

    let mut min_chain: usize = burn_in.unwrap_or(0);

    println!("num ch {num_chains} max chain {max_chain}");
    let potential_prior_change: Option<f64> = args.get(7).map(|a| a.parse::<f64>().ok()).flatten();

    match potential_prior_change {
        Some(credibility) => {
            if !(credibility > 0.0) {panic!("Motif prior threshold must be a valid strictly positive float");}
            //SAFETY: This is never a multithreaded spot
            unsafe{ NECESSARY_MOTIF_IMPROVEMENT = credibility; }
        },
        None => (),
    };

    if max_chain < min_chain {
        warn!("Your upper bound id for beads in a single chain to consider for inference exceeds the minimum. We swap their roles, but make sure you didn't mess up.");
        let temp = min_chain;
        min_chain = max_chain;
        max_chain = temp;
    }

    if num_chains > 26 {
        warn!("Can only sustain up to 26 independent chains. Only considering the first 26.");
        num_chains = 26;
    }

    let mut buffer: Vec<u8> = Vec::new();

    //let max_max_length = 100000;
    //This is the code that actually sets up our independent chain reading
    let mut set_trace_collections: Vec<SetTraceDef> = Vec::with_capacity(max_chain-min_chain);
    for chain in 0..num_chains {
        let base_str = if scale_thresh.is_none() {
            format!("{}/{}_{}", out_dir.clone(), base_file, UPPER_LETTERS[chain])
        } else {
            format!("{}/{}_{}_custom_scale_{:.1}", out_dir.clone(), base_file,UPPER_LETTERS[chain], scale_thresh.unwrap())
        };
        println!("Base str {} min chain {}", base_str, min_chain);
        let regex = Regex::new(&(base_str.clone()+format!("_{}_trace_from_step_", min_chain).as_str()+"\\d{7}.bin")).unwrap();
        //let regex = Regex::new(&(base_str.clone()+"_trace_from_step_"+"0\\d{6}.bin")).unwrap();
        let directory_iter = fs::read_dir(&out_dir).expect("This directory either doesn't exist, you're not allowed to touch it, or isn't a directory at all!");
        
        let mut chain_files = directory_iter.filter(|a| regex.is_match(a.as_ref().unwrap().path().to_str().unwrap())).map(|a| a.unwrap().path().to_str().unwrap().to_string()).collect::<Vec<_>>();

        chain_files.sort();


        let skip_some: bool = if min_chain == 0 && chain_files.len() > 1 { true } else {false};

        let mut iter_files = chain_files.iter();


        if skip_some { 
            for _ in 0..(chain_files.len().min(10)){
            _ = iter_files.next(); 
            }
        }

        let file_y = iter_files.next();

        println!("file {:?} ", file_y);


        fs::File::open(file_y.expect("Must have matches to proceed")).expect("We got this from a list of directory files").read_to_end(&mut buffer).expect("something went wrong reading the file");

        set_trace_collections.push(bincode::deserialize(&buffer).expect("All read in files must be correct bincode!"));


        for file_name in iter_files {
            buffer.clear();
            fs::File::open(file_name).expect("We got this from a list of directory files").read_to_end(&mut buffer).expect("something went wrong reading the file");
            let interim: SetTraceDef = bincode::deserialize(&buffer).expect("All read in files must be correct bincode!");
            //interim.reduce(max_max_length);
            set_trace_collections[chain].append(interim);
        }


        if (min_chain+1) < max_chain {for bead in (min_chain+1)..max_chain {

            let regex = Regex::new(&(base_str.clone()+format!("_{}_trace_from_step_\\d{{7}}.bin", bead).as_str())).unwrap();
            let directory_iter = fs::read_dir(&out_dir).expect("This directory either doesn't exist, you're not allowed to touch it, or isn't a directory at all!");

            let mut chain_files = directory_iter.filter(|a| regex.is_match(a.as_ref().unwrap().path().to_str().unwrap())).map(|a| a.unwrap().path().to_str().unwrap().to_string()).collect::<Vec<_>>();

            chain_files.sort();

            let iter_files = chain_files.iter();

            for file_name in iter_files {
                println!("{file_name}");
                buffer.clear();
                fs::File::open(file_name).expect("We got this from a list of directory files").read_to_end(&mut buffer).expect("something went wrong reading the file");
                let interim: SetTraceDef = bincode::deserialize(&buffer).expect("All read in files must be correct bincode!"); 
                
                /*if interim.len() >= 5 {
                    println!("{} {:?} inter", file_name, interim.index_into(0..5));
                } else{
                    println!("{} {:?} inter", file_name, interim.index_into(0..1));
                }*/
                set_trace_collections[chain].append(interim);
            }




        }}
    
        buffer.clear();
    }

    let all_data_file = set_trace_collections[0].data_name().to_owned();

    //TODO: change out data_name for a way to supply the correct data file to the processing directly
    /*let mut all_dat_file = fs::File::open(set_trace_collections[0].data_name()).expect("Big trouble if we're saving invalid data files");
    all_dat_file.read_to_end(&mut buffer);
    let all_dat: AllData = bincode::deserialize(&buffer).expect("Big trouble if we're saving invalid data files");
    let all_dat_use: AllDataUse = AllDataUse::new(&all_dat).expect("Big trouble if we're saving invalid data files");*/

    //set_trace_collections is now the chains we want
    
    let mut min_len: usize = usize::MAX;
    for trace in set_trace_collections.iter() {
        min_len = min_len.min(trace.len());
        println!("Min len {}", min_len);
    }
    
    let best_motif_sets: Vec<&StrippedMotifSet> = set_trace_collections.iter().map(|a| a.extract_highest_posterior_set(min_len)).collect::<Vec<_>>();

    let best_single_motif_set: &StrippedMotifSet = *best_motif_sets.iter().max_by(|a,b| a.ln_posterior().partial_cmp(&b.ln_posterior()).unwrap()).expect("The set traces should all have at least SOME elements");

    best_single_motif_set.set_iter().enumerate().for_each(|(i,m)| {
        let prep: Vec<[(usize, f64); BASE_L]> = m.pwm_ref().iter().map(|b| core::array::from_fn(|a| (a, b.scores()[a]))).collect();
        draw_pwm(&prep, format!("{}/{}_best_trace_pwm_{}.png", out_dir.clone(), base_file, i).as_str());
    });

    buffer.clear();

    let mut try_bincode = fs::File::open(all_data_file.as_str()).expect("a trace should always refer to a valid data file");
    let _ = try_bincode.read_to_end(&mut buffer);

    let data_reconstructed: AllData = bincode::deserialize(&buffer).expect("Monte Carlo chain should always point to data in proper format for inference!");

    let data_ref = AllDataUse::new(&data_reconstructed, 0.0).expect("AllData file must be valid!");

    println!("Best motif set: {:?}", best_single_motif_set);
    let activated = best_single_motif_set.reactivate_set(&data_ref);
    for i in 0..best_single_motif_set.num_motifs(){
        let mot = activated.get_nth_motif(i);
        println!("null binds best {:?}", mot.return_any_null_binds_in_group(data_ref.null_seq(), data_ref.background_ref().noise_spread_par() * 5.0));
    }
    let save_file = format!("{}_best_trace", base_file);

    best_single_motif_set.save_this_trace(&data_ref, &out_dir, &save_file);

    let _ = best_single_motif_set.generate_pr_curve(&data_ref,"/home/alimf/motif_finder_project/Data/Fasta/NC_000913.2.fasta", &out_dir, &save_file);

    let reference_motifs: Vec<Motif> = best_single_motif_set.set_iter().map(|a| a.clone()).collect();

    let distances_file = format!("{}/{}_distances_to_refs.png",out_dir.clone(), base_file);
    let autocorrs_file = format!("{}/{}_distance_autocorrelations.png",out_dir.clone(), base_file);
    

    let distances_plot = BitMapBackend::new(distances_file.as_str(), (2600, 2600)).into_drawing_area();
    let autocorrelation_plot = BitMapBackend::new(autocorrs_file.as_str(), (1350, 1350)).into_drawing_area();

    distances_plot.fill(&full_palette::WHITE).unwrap();
    autocorrelation_plot.fill(&full_palette::WHITE).unwrap();

    let mut autocorrelations_ctx = ChartBuilder::on(&autocorrelation_plot)
                               .caption("Autocorrelations of each trace's distances", ("Times New Roman", 30))
                               .set_label_area_size(LabelAreaPosition::Left, 40)
                               .set_label_area_size(LabelAreaPosition::Bottom, 40)
                               .build_cartesian_2d(0..set_trace_collections[0].len(), 0_f64..1.5_f64)
                               .unwrap();

    autocorrelations_ctx.configure_mesh()
                               .x_desc("Lag (in units of step save period)")
                               .y_desc("Autocorrelation coefficient")
                               .draw().unwrap();




    let distances_panels = distances_plot.split_evenly((2, (reference_motifs.len()+1)/2));
    let mut distances_ctxes = distances_panels.iter().enumerate().map(|(i,a)| ChartBuilder::on(a)
                                                                      .caption(format!("Distance from Reference {}", i).as_str(), ("Times New Roman", 30))
                                                                      .set_label_area_size(LabelAreaPosition::Left, 40)
                                                                      .set_label_area_size(LabelAreaPosition::Bottom, 40)
                                                                      .build_cartesian_2d(0..set_trace_collections[0].len(), 0_f64..100_f64)
                                                                      .unwrap()
                                                                      ).collect::<Vec<_>>(); 

    for ctx in distances_ctxes.iter_mut() { ctx.configure_mesh()
                                               .x_desc("Saved steps")
                                               .y_desc("Mean of motif set distance to reference")
                                               .draw().unwrap(); }

    let correlation_len = 1000;

    for (chain, collection) in set_trace_collections.iter().enumerate() {

        let _chain_name = format!("{}_{}",base_file, UPPER_LETTERS[chain]);
        
    
        //TODO: This makes a vector V of length reference_motifs, of vectors of length equal 
        //      to each collection. V[i][j] is the mean distance of the motifs in collection's
        //      jth set to the ith reference motif. I eventually want to: 
        //          1-Create a plot for each ith REFERENCE motif, with lines from each of the collections of its distances
        //          2-Calculate autocorrelations for each REFERENCE motif and plot them as well
        let distance_traces = collection.create_distance_traces(&reference_motifs);

        let collection_autocorrelation = AllData::compute_autocorrelation_coeffs(&distance_traces, correlation_len);

        println!("Collection Autocorrelation: {:?}", collection_autocorrelation);

        let style = ShapeStyle{ color: (*PALETTE[chain]).into(), filled: true, stroke_width: 0};

        let color = unsafe{*PALETTE.as_ptr().add(chain)};

        let letter = UPPER_LETTERS[chain];

        for (i, trace) in distance_traces.iter().enumerate() {
            distances_ctxes[i].draw_series(trace.iter().enumerate().map(|(i, p)| Circle::new((i,*p), 5, style.clone())))
                              .unwrap().label(format!("{}", letter).as_str())
                              .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.clone()));
                              
        }

        autocorrelations_ctx.draw_series(collection_autocorrelation.iter().enumerate().map(|(i, p)| Circle::new((i,*p), 5, ShapeStyle{ color: (*PALETTE[chain]).into(), filled : true, stroke_width : 0})))
                                                                   .unwrap().label(format!("{}", UPPER_LETTERS[chain]).as_str())
                                                                   .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.clone()));


    }

    autocorrelations_ctx.configure_series_labels().draw().unwrap();
    for ctx in distances_ctxes.iter_mut() { ctx.configure_series_labels().draw().unwrap(); }

    let autocorrs_file = format!("{}/{}_ln_post_autocorrelations.png",out_dir.clone(), base_file);
    

    let autocorrelation_plot = BitMapBackend::new(autocorrs_file.as_str(), (1350, 1350)).into_drawing_area();

    autocorrelation_plot.fill(&full_palette::WHITE).unwrap();

    let mut autocorrelations_post = ChartBuilder::on(&autocorrelation_plot)
                               .caption("Autocorrelations of each trace's ln posterior density", ("Times New Roman", 30))
                               .set_label_area_size(LabelAreaPosition::Left, 40)
                               .set_label_area_size(LabelAreaPosition::Bottom, 40)
                               .build_cartesian_2d(0..correlation_len, 0_f64..1.5_f64)
                               .unwrap();

    autocorrelations_post.configure_mesh()
                               .x_desc("Lag (in units of step save period)")
                               .y_desc("Autocorrelation coefficient")
                               .draw().unwrap();


    std::mem::drop(buffer);
    let mut plot_post = poloto::plot(format!("{} Ln Posterior", base_file), "Step", "Ln Posterior");
    println!("{}/{}_ln_post.svg", out_dir.clone(), base_file);
    let plot_post_file = fs::File::create(format!("{}/{}_ln_post.svg", out_dir.clone(), base_file).as_str()).unwrap();
    for (i, trace) in set_trace_collections.iter().enumerate() {
        let letter = UPPER_LETTERS[i];
        let tracey = trace.ln_posterior_trace();
        let mut ordered_tracey = tracey.clone();
        ordered_tracey.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let minind = ordered_tracey.len()/10;
        let maxind = (ordered_tracey.len()*9)/10;
        let mid = (ordered_tracey[maxind]+ordered_tracey[minind])/2.0;
        let min_y = mid-(mid-ordered_tracey[minind])*1.5;
        let max_y = mid+(ordered_tracey[maxind]-mid)*1.5;
        let trace_mean = tracey.iter().sum::<f64>()/(tracey.len() as f64);


        //println!("conj {} {:?}", conj.len(), conj);

        /*        let res = samples_fft_to_spectrum(
        &samples[0..len],
        samples.len() as u32,
        FrequencyLimit::All,
        None,).unwrap();
       

        
        */
       

        plot_post.line(format!("Chain {}", letter), tracey.clone().into_iter().enumerate().map(|(a, b)| (a as f64, b)).crop_below(min_y).crop_above(max_y));//.xmarker(0).ymarker(0);
        let collection_autocorrelation = AllData::compute_autocorrelation_coeffs(&vec![tracey.clone()], 1000);

        let color = unsafe{*PALETTE.as_ptr().add(i)};
        autocorrelations_post.draw_series(collection_autocorrelation.iter().enumerate().map(|(j, p)| Circle::new((j,*p), 5, ShapeStyle{ color: (*PALETTE[i]).into(), filled : true, stroke_width : 0})))
                                                                   .unwrap().label(format!("{}", UPPER_LETTERS[i]).as_str())
                                                                   .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color.clone()));

    }
    autocorrelations_post.configure_series_labels().draw().unwrap();

    plot_post.simple_theme(poloto::upgrade_write(plot_post_file));

    let mut plot_tf_num = poloto::plot(format!("{} Motif number", base_file), "Step", "Motifs");
    let plot_tf_num_file = fs::File::create(format!("{}/{}_tf_num.svg", out_dir.clone(), base_file).as_str()).unwrap();
    for (i, trace) in set_trace_collections.iter().enumerate() {
        let letter = UPPER_LETTERS[i];
        plot_tf_num.line(format!("Chain {}", letter), trace.motif_num_trace().into_iter().enumerate().map(|(a, b)| (a as f64, b))).xmarker(0).ymarker(0);
    };
    plot_tf_num.simple_theme(poloto::upgrade_write(plot_tf_num_file));
   
    /*
    let mut plot_like = poloto::plot(format!("{} Ln Likelihood", base_file), "Step", "Ln Likelihood");
    let plot_like_file = fs::File::create(format!("{}/{}_ln_like.svg", out_dir.clone(), base_file).as_str()).unwrap();
    for (i, trace) in set_trace_collections.iter().enumerate() {
        let letter = UPPER_LETTERS[i];
        plot_like.line(format!("Chain {}", letter), trace.ln_likelihood_trace(&all_dat).into_iter().enumerate().map(|(a, b)| (a as f64, b)));//.xmarker(0).ymarker(0);
    }; */

    /*
    let mut plot_like = poloto::plot(format!("{} Median distance to data", base_file), "Step", "Median Dist");
    let plot_like_file = fs::File::create(format!("{}/{}_med_dist.svg", out_dir.clone(), base_file).as_str()).unwrap();
    for (i, trace) in set_trace_collections.iter().enumerate() {
        let letter = UPPER_LETTERS[i];
        plot_like.line(format!("Chain {}", letter), trace.wave_dist_trace(&all_dat_use).into_iter().enumerate().map(|(a, b)| (a as f64, b)));//.xmarker(0).ymarker(0);
    };*/

    //plot_like.simple_theme(poloto::upgrade_write(plot_like_file));
    
    let _rng = rand::thread_rng();
    
    //let ref_per_chain: usize = 5;

    //let num_ref_pwm = ref_per_chain*set_trace_collections.len();


    let cluster_per_chain: usize = min_len.min(2000);

    let motif_num_traces = set_trace_collections.iter().map(|a| a.motif_num_trace()).collect::<Vec<_>>();

    //TODO: generate motif_num_traces plot here
    println!("Motif num rhat: {}", rhat(&motif_num_traces, min_len));


    println!("Finish best motif sets");

    let total_sets = min_len*set_trace_collections.len();
    let _tf_num = (motif_num_traces.into_iter().map(|a| tail_slice(&a, min_len).iter().sum::<f64>()).sum::<f64>()/(total_sets as f64)).floor() as usize;

    let _num_sort_mots: usize = cluster_per_chain*set_trace_collections.len();
    

    //If I don't document this as I'm writing it, it's going to blow a hole 
    //through my head and the head of anybody reading it
    //"pairings" is nested in the following way:
    //      1) Over the independent chains
    //      2) For each independent chain, over the StrippedMotifSets in that chain
    //      3) For each StrippedMotifSet in the chain, there's a best_single_motif_set.len() number of pairings: 
    //                                                 which, if they exist, are the motif and distance of that pairing
    let pairings: Vec<Vec<Vec<Option<(f64,bool,&Motif)>>>> = set_trace_collections.iter().map(|set_trace| {
        set_trace.ref_to_trace().into_par_iter().map(|set| set.pair_to_other_set(&best_single_motif_set)).collect::<Vec<_>>()
    }).collect();

    let num_clusts = best_single_motif_set.num_motifs();
    let likely_cap = pairings[0].len()*pairings.len();

    //This is nested so that the outer layer corresponds to the Motifs in the 
    //best_single_motif_set, and the inner layer to all the matches, regardless of chain
    let mut results: Vec<Vec<((f64, bool, &Motif), usize)>> = (0..num_clusts).map(|i| {
        let mut cluster: Vec<((f64, bool, &Motif), usize)> = Vec::with_capacity(likely_cap);
        for (c_id, chain) in pairings.iter().enumerate() {
            for pairing in chain.iter() {
                if let Some(dist_mot_tup) = pairing[i] {
                    cluster.push((dist_mot_tup, c_id));
                };
            }
        }
        cluster
    }).collect();

    for result in results.iter_mut() {
        result.sort_by(|((a, _, _),_), ((b,_, _),_)| b.partial_cmp(a).unwrap())
    }

    //This is nested over 
    //      1) the Motifs in the best_single_motif_set
    //      2) the indepdendent chains
    //      3) the distances present in those chains (NOTE: we skip over nones)
    let distance_assessments: Vec<Vec<Vec<f64>>> = (0..num_clusts).map(|i| {
        pairings.iter().map(|chain| {
            let mut dist_vec: Vec<f64> = Vec::with_capacity(chain.len());
            for pairing in chain.iter() {
                if let Some((dist,_, _)) = pairing[i] {dist_vec.push(dist);};
            }
            dist_vec
        }).collect()
    }).collect();




    println!("min {}", min_len);
    for (i, mut trace) in results.into_iter().enumerate() {
        println!("Motif {}: \n", i);
        println!("Rhat motif {}: {}", i, rhat(&distance_assessments[i], min_len));
        println!("Motif: \n {:?}", best_single_motif_set.get_nth_motif(i));
        
        let credible_percent: usize = 95;
        let cluster_term = (trace.len()*credible_percent)/100;

        trace.truncate(cluster_term);

        //trace now holds the credible_percent region

        let (pwm_traces, chain_ids) = create_offset_traces(trace);
        graph_tetrahedral_traces(&pwm_traces, &chain_ids, format!("{}/{}_new_tetrahedra_{}.png", out_dir.clone(), base_file, i).as_str());
        println!("tetrad");
        /*let cis = create_credible_intervals(&pwm_traces, 0.95, &good_motifs_count);


        println!("Number motifs near medoid: {}", num_good_motifs);
        println!("Lower {} CI bound: \n {:?}", 0.95, cis[0]);
        println!("Posterior mean: \n {:?}", cis[1]);
        println!("Upper {} CI bound: \n {:?}", 0.95, cis[2]);*/
    }

    //TODO: generate lnlikelihood and lnposterior traces here

    
}




pub fn rhat(chain_pars: &Vec<Vec<f64>>, chain_length: usize) -> f64 {

    let max_poss_chain = chain_pars.iter().map(|a| a.len()).min().expect("Empty chain vectors not allowed");
    
    let real_length = if max_poss_chain < chain_length {
        warn!("Your chain length is larger than the minimum chain length of your parameters!\n Defaulting to the minimum chain length of your parameters.");
        max_poss_chain
        } else {chain_length};

    let chain_sums: Vec<f64> = chain_pars.iter().map(|a| tail_slice(a, real_length).iter().sum::<f64>()).collect();

    let chain_means: Vec<f64> = chain_sums.iter().map(|a| (*a)/(real_length as f64)).collect();
  
    let big_mean = chain_means.iter().sum::<f64>()/((chain_pars.len()) as f64);

    let chain_vars: Vec<f64> = chain_pars.iter().zip(chain_means.iter()).map(|(chain, mean)| {

        tail_slice(chain, real_length).iter().map(|val| (val-mean).powi(2)).sum::<f64>()/((real_length-1) as f64)

    }).collect();

    println!("chain_pars len {} chain_means {:?}, big_mean {}", chain_pars.len(), chain_means, big_mean);
    let b: f64 = ((real_length as f64)/((chain_pars.len()-1) as f64))*chain_means.iter().map(|a| (a-big_mean).powi(2)).sum::<f64>();

    let w: f64 = chain_vars.iter().sum::<f64>()/(chain_pars.len() as f64);

    println!("b {} w {} rl {}", b, w, real_length);
    (1.0+((b/w)-1.0)/(real_length as f64)).sqrt()

}

//Panics: if slicer is empty
pub fn tail_slice<T>(slicer: &[T], last_n: usize) -> &[T] {
    let first = if last_n > slicer.len() {0} else {slicer.len()-last_n};
    &slicer[first..slicer.len()]
}

pub fn establish_dist_array(motif_list: &Vec<Motif>) -> Array2<f64> {

    let mut dist_array = Array2::<f64>::zeros([motif_list.len(), motif_list.len()]);

    for i in 0..(motif_list.len()-1) {
        for j in (i+1)..(motif_list.len()) {
            dist_array[[i,j]] = motif_list[i].distance_function(&motif_list[j]).0;
            dist_array[[j,i]] = dist_array[[i,j]];
        }
    }
    dist_array

}


pub fn create_offset_traces(best_motifs: Vec<((f64, bool, &Motif), usize)>) -> (Array3<f64>, Vec<usize>){


    let num_samples = best_motifs.len();
    let chain_ids = best_motifs.iter().map(|(_,i)| *i).collect::<Vec<usize>>();
    let pwms_offsets = best_motifs.into_iter()
                                  .map(|((_, c, a), _)| {
                                      if c {a.rev_complement()} else {a.pwm()}
                                  }).collect::<Vec<Vec<Base>>>();

    let _neutral: f64 = 0.0;


    let mut samples = Array3::<f64>::zeros([num_samples, MAX_BASE, BASE_L-1]);

    for (k, pwm) in pwms_offsets.into_iter().enumerate() {

        for j in 0..pwm.len() {

            //let mut slice = samples.slice_mut(s![k,j,..]);  //I went with this because it's the fastest possible assignment.
                                                            //It will slow down calculations of credible intervals and means,
                                                            //But I do that once, not 10s of thousands of times. I did not
                                                            //benchmark this, though
            let _begin = if MAX_BASE & 1 == 0 {(MAX_BASE-pwm.len())/2} else {(MAX_BASE+1-pwm.len())/2 }; //This should be optimized away

            let probs = pwm[j].as_simplex();
            for i in 0..(BASE_L-1) {samples[[k,j+(MAX_BASE-pwm.len())/2,i]] = probs[i];}
        }

    }

    (samples, chain_ids)

}

pub fn graph_tetrahedral_traces(samples: &Array3::<f64>, chain_ids: &Vec<usize>, file_name: &str){

    let (_ , num_bases, _) = samples.dim();

    let num_rows = if (num_bases & 3) == 0 {num_bases/4} else{ num_bases/4 +1 } as u32; 

    let plot = BitMapBackend::new(file_name, (2600, num_rows*600)).into_drawing_area();

    plot.fill(&full_palette::WHITE).unwrap();

    let panels = plot.split_evenly((num_rows as usize, 4));

    let cis = create_credible_intervals(samples);

    println!("Finished cred");
    for j in 0..num_bases {
        
        println!("{j}");

        let mut chart = ChartBuilder::on(&(panels[j])).margin(10).caption(format!("Base {}", j).as_str(), ("serif", 20))
            .build_cartesian_2d(-(4.0*SQRT_2/3.)..(2.0*SQRT_2/3.), -(2.*SQRT_2*SQRT_3/3.)..(2.*SQRT_2*SQRT_3/3.)).unwrap();
        chart.configure_mesh().draw().unwrap();


        //Draws the tetrahedral faces
        let base_tetras = create_tetrahedral_traces(&SIMPLEX_ITERATOR.iter().map(|&[a,b,c]| (*a,*b,*c)).collect::<Vec<_>>());


        for face in base_tetras.iter() {
            chart.draw_series(LineSeries::new(face.iter().map(|&a| a), &full_palette::BLACK)).unwrap();
        }

        //Draws the three vertices that show up once
        chart.draw_series(PointSeries::of_element([SIMPLEX_VERTICES_POINTS[0]].iter().map(|a| turn_to_no_t(&(a[0], a[1], a[2]))), 5, ShapeStyle::from(&full_palette::RED).filled(), &|coord, size, style| {
            EmptyElement::at(coord)
                + Circle::new((0, 0), size, style)
                + Text::new(format!("A"), (0, 15), ("sans-serif", 15))
        })).unwrap();
        chart.draw_series(PointSeries::of_element([SIMPLEX_VERTICES_POINTS[1]].iter().map(|a| turn_to_no_t(&(a[0], a[1], a[2]))), 5, ShapeStyle::from(&YELLOW_700).filled(), &|coord, size, style| {
            EmptyElement::at(coord)
                + Circle::new((0, 0), size, style)
                + Text::new(format!("C"), (0, 15), ("sans-serif", 15))
        })).unwrap();
        chart.draw_series(PointSeries::of_element([SIMPLEX_VERTICES_POINTS[2]].iter().map(|a| turn_to_no_t(&(a[0], a[1], a[2]))), 5, ShapeStyle::from(&full_palette::GREEN).filled(), &|coord, size, style| {
            EmptyElement::at(coord)
                + Circle::new((0, 0), size, style)
                + Text::new(format!("G"), (0, 15), ("sans-serif", 15))

        })).unwrap();

        //Draws the vertex that shows up three times
        chart.draw_series(PointSeries::of_element(create_tetrahedral_traces(&vec![(SIMPLEX_VERTICES_POINTS[3][0], SIMPLEX_VERTICES_POINTS[3][1], SIMPLEX_VERTICES_POINTS[3][2])]).into_iter().take(3).map(|a| a).flatten(), 5, ShapeStyle::from(&full_palette::BLUE).filled(), &|coord, size, style| {
            EmptyElement::at(coord)
                + Circle::new((0, 0), size, style)
                + Text::new("T", (0, 15), ("sans-serif", 15))
        })).unwrap();
        
        

        let slice_bases: Vec<(f64, f64, f64)> = (0..chain_ids.len()).map(|k| (samples[[k,j,0]], samples[[k,j,1]], samples[[k,j,2]]) ).collect::<Vec<_>>();

        let tetra_traces = create_tetrahedral_traces(&slice_bases);
            
        for (m, little_trace) in tetra_traces.iter().enumerate() {
            chart.draw_series(little_trace.iter().step_by(10).map(|&a| Circle::new(a,2_u32, PALETTE[(chain_ids[m]+BASE_L) % 26]))).unwrap();
        }

        
 
        /*let prep_pwm: Vec<[(usize, f64); BASE_L]> = cis.iter().map(|(_, tetrahedral_mean)| {
            let b = Base::vect_to_base(tetrahedral_mean);
            println!("{:?}", b);
            b.seqlogo_heights()
        }).collect();

        draw_pwm(&prep_pwm, &format!("{}_pwm.png", file_name));*/
        //This draws the credible region
       
        //let ci_color = &ORANGE.mix(0.1);
        //let ci_low_rect = create_tetrahedral_traces(&cis[j]);
 
        //let ci_up = cis[j].iter().map(|a|  (a.0+INTERVAL_CELL_LENGTH, a.1+INTERVAL_CELL_LENGTH, a.2+INTERVAL_CELL_LENGTH)).collect::<Vec<_>>();

        //let ci_hi_react = create_tetrahedral_traces(&ci_up);

        //for b in 0..BASE_L {
            //chart.draw_series(ci_low_rect[b].iter().zip(ci_hi_react[b].iter()).map(|(&a, &b)| Rectangle::new([a,b], &ci_color))).unwrap();
          //  chart.draw_series(ci_low_rect[b].iter().map(|a| Rectangle::new([a,(a.0+INTERVAL_CELL_LENGTH, a.1+INTERVAL_CELL_LENGTH)], &ci_color))).unwrap();
        //}


    }
}

fn draw_pwm(map_heights: &[[(usize, f64); BASE_L]], file_name: &str) {

    //energy in units of kcal/mol
    const RT: f64 = 0.592_186_869_074_967_85;
    let max_energy = SCORE_THRESH*RT*LN_2;
    let len = map_heights.len();
    let plot = BitMapBackend::new(file_name, (120*(len as u32), 600)).into_drawing_area();

    plot.fill(&full_palette::WHITE).unwrap();

    let mut chart = ChartBuilder::on(&plot)
            .set_label_area_size(LabelAreaPosition::Left, 70)
            .set_label_area_size(LabelAreaPosition::Bottom, 70)
            .margin(5)
            .caption("PWM", ("sans-serif", 40))
            .build_cartesian_2d(0_f64..((len+1) as f64), max_energy..0_f64).unwrap();



    chart.configure_mesh()
        .disable_x_mesh()
        .disable_y_mesh()
        .x_desc("Motif position (bp)")
        .y_desc("-ΔΔG (kcal/mol)")
        .axis_desc_style(("serif", 40))
        .draw().unwrap();

    chart.configure_series_labels()
        .margin(40)
        .label_font(("serif", 50))
        .draw().unwrap();


    chart.draw_series(map_heights.iter().enumerate().map(|(i, hs)| {

        println!("hs {:?}", hs);

        let pos = i;
        hs.iter().enumerate().map(move |(e, h)| {
            let height = h.1*RT*LN_2;
            Text::new(format!("{}", BPS[h.0]), ((pos as f64)+(0.0625*(h.0 as f64)), height), ("serif", 40).into_font().color(BASE_COLORS[h.0]))
        }) 
    }).flatten()).unwrap();

}

struct CustomizedRange(Range<f64>);

impl Ranged for CustomizedRange {

    type ValueType = f64;
    type FormatOption = NoDefaultFormatting;

    fn map(&self, value: &Self::ValueType, limit: (i32, i32)) -> i32 {
        let size = limit.1 - limit.0;
        (((*value as f64)-self.0.start) / (self.0.end -self.0.start) * size as f64) as i32 + limit.0
    }

    fn range(&self) -> std::ops::Range<Self::ValueType> {
        self.0.start..self.0.end
    }

    fn key_points<Hint: KeyPointHint>(&self, hint: Hint) -> Vec<Self::ValueType> {
        if hint.max_num_points() < (self.0.end as usize) {
            return vec![];
        }

        ((self.0.start as usize)..(self.0.end as usize)).map(|a| a as f64).collect()
    }

}

impl ValueFormatter<f64> for CustomizedRange {
    fn format_ext(&self, value: &f64) -> String {
        format!("{} ", *value as usize)
    }
}


//We want to eat best_motifs so that we can save on memory
//You'll notice that this entire function is designed to drop memory I don't
//need anymore as much as humanly possible.
pub fn create_credible_intervals(samples: &Array3<f64>) -> Vec<(Vec<(f64, f64, f64)>, [f64; BASE_L-1])>{


    //I haven't removed num_bases entirely because I might need it if I uncomment
    //some currently commented code
    let (num_samples, _num_bases, _) = samples.dim();
    let credible = 1.0_f64;
    let num_interval = ((num_samples as f64)*credible).floor() as u32;
    println!("samples dim  {:?}, num_interval {} credible {credible}", samples.dim(), num_interval);

    let z_steps: usize = ((4./3.)/INTERVAL_CELL_LENGTH).ceil() as usize;
    let x_steps: usize = (SQRT_2/INTERVAL_CELL_LENGTH).ceil()  as usize;
    let y_steps: usize = ((2.*SQRT_2/SQRT_3)/INTERVAL_CELL_LENGTH).ceil() as usize;

    let zs = (0..=z_steps).map(|a| -1./3.+(a as f64)*INTERVAL_CELL_LENGTH).collect::<Vec<_>>();
    let xs = (0..=x_steps).map(|a| -SQRT_2/3.+(a as f64)*INTERVAL_CELL_LENGTH).collect::<Vec<_>>();
    let ys = (0..=y_steps).map(|a| -SQRT_2/SQRT_3+(a as f64)*INTERVAL_CELL_LENGTH).collect::<Vec<_>>();



    let mut i = 0;
    
    //This will yield a vector where the ith position corresponds to the cells of the
    //credible region for the ith base
    let credible_cells_vec = samples.axis_iter(ndarray::Axis(1)).map(|base_vecs| {

        println!("initialized {i}");
        //I picked the smallest unsized int that I could: I can have more than 65000 points in a cell
        //theoretically (hard but not impossible). But I can't have more than 4 billion: a million
        //steps still takes a couple of weeks. Billion would just not be practical
        //The unusual order here is for memory accessibility.
        let mut cell_counts: Array3<u32> = Array3::<u32>::zeros([x_steps, y_steps, z_steps]);

        for base_vector in base_vecs.axis_iter(ndarray::Axis(0)) {

            //println!("b len {}", base_vector.dim());
            let x_ind = ((base_vector[0]-(-SQRT_2/3.))/INTERVAL_CELL_LENGTH).floor() as usize;
            let y_ind = ((base_vector[1]-(-SQRT_2/SQRT_3))/INTERVAL_CELL_LENGTH).floor() as usize;
            let z_ind = ((base_vector[2]-(-1.0/3.))/INTERVAL_CELL_LENGTH).floor() as usize;
            if x_ind >= x_steps { println!("about to break x {} {}", x_ind, x_steps);}
            if y_ind >= y_steps { println!("about to break y {} {}", y_ind, y_steps);}
            if z_ind >= z_steps { println!("about to break z {} {}", z_ind, z_steps);}
            //println!("dim {:?} x {} y {} z {}", cell_counts.dim(), x_ind, y_ind, z_ind);
            cell_counts[[x_ind, y_ind, z_ind]] += 1;

        }

        //println!("{cell_counts} counts of cell"); 
 
        let mut cells_and_counts = cell_counts.indexed_iter().map(|(a, &b)| (a, b)).collect::<Vec<_>>();

        //b.cmp(a) sorts from greatest to least
        cells_and_counts.sort_unstable_by(|(_, a), (_, b)| b.cmp(a));

        let mut to_interval: u32 = 0;
        let mut index: usize = 0;

        while to_interval < num_interval {
            to_interval += cells_and_counts[index].1;
            index += 1;
        }

        let region = cells_and_counts.drain(0..index).map(|(a, _)| (xs[a.0], ys[a.1], zs[a.2])).filter(|a| a.0.is_finite() && a.1.is_finite() && a.2.is_finite()).collect::<Vec<(f64, f64, f64)>>();

        println!("region sampler {:?}", &region[0..(10.min(region.len()-1))]);
        let space_region = region.iter().map(|&(a,b,c)| Base::simplex_to_vect(&[a,b,c])).filter(|a| a[0].is_finite() && a[1].is_finite() && a[2].is_finite()).collect::<Vec<[f64;3]>>();
        println!("space sampler {:?}", &space_region[0..(10.min(space_region.len()-1))]);
        
        let index = space_region.len();

        let posterior_sum = space_region.iter().fold((0.0, 0.0, 0.0), |acc, x| (acc.0+x[0], acc.1+x[1], acc.2+x[2]));
        let posterior_mean = [posterior_sum.0/(index as f64), posterior_sum.1/(index as f64), posterior_sum.2/(index as f64)];

        println!("{i} {:?}", posterior_mean);
        i+=1; 
        (region, posterior_mean)

    }).collect::<Vec<_>>();



    credible_cells_vec




}


//This is ONLY valid if BASE_L = 4 and if the base vertices are as I've set them.
//I'd set a cfg about it if it would work, but it doesn't so just don't 
//think you're smarter than me about my own project.
fn create_tetrahedral_traces(tetra_bases: &Vec<(f64, f64, f64)>) -> [Vec<(f64, f64)>; BASE_L] {
 
    [tetra_bases.iter().map(|b| turn_to_no_a(b)).collect(), 
     tetra_bases.iter().map(|b| turn_to_no_c(b)).collect(),
     tetra_bases.iter().map(|b| turn_to_no_g(b)).collect(),
     tetra_bases.iter().map(|b| turn_to_no_t(b)).collect()]
}

fn turn_to_no_a(tetra_base: &(f64, f64, f64)) -> (f64, f64) {
    (-tetra_base.0/3.-2.*SQRT_2*tetra_base.2/3.-2.*SQRT_2/3., tetra_base.1)
}
fn turn_to_no_c(tetra_base: &(f64, f64, f64)) -> (f64, f64) {
    (2.*tetra_base.0/3.+tetra_base.1/SQRT_3+SQRT_2*tetra_base.2/3.+SQRT_2/3.,  tetra_base.0/SQRT_3-SQRT_2*tetra_base.2/SQRT_3-SQRT_2/SQRT_3)
}
fn turn_to_no_g(tetra_base: &(f64, f64, f64)) -> (f64, f64) {
    (2.*tetra_base.0/3.-tetra_base.1/SQRT_3+SQRT_2*tetra_base.2/3.+SQRT_2/3., -tetra_base.0/SQRT_3+SQRT_2*tetra_base.2/SQRT_3+SQRT_2/SQRT_3)
}
fn turn_to_no_t(tetra_base: &(f64, f64, f64)) -> (f64, f64) {
    (tetra_base.0, tetra_base.1)
}

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn rhat_test() {
        
        let chain_pars = vec![vec![1.0_f64, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let r_hat = rhat(&chain_pars, 3);
        println!("{:?} {}", tail_slice(&chain_pars[0], 3), r_hat);
        assert!((r_hat-(31.0_f64/6.0).sqrt()).abs() < 1e-6);

        let r2_hat = rhat(&chain_pars, 2);
        assert!((r2_hat-(19.0/2.0_f64).sqrt()).abs() < 1e-6);

    }

    #[test]
    fn draw_test() {
        /*let fake_pwm: Vec<[f64;BASE_L]> = vec![[0.0000000000_f64, -0.9373375710, -0.6575943318, -0.0449069248],
 [0.0000000000, -0.2426674731, -0.8136657737, -0.0097526719],
 [-0.7145200271, -0.0085354689, -0.6001008046, 0.0000000000],
 [-0.3619683015, -0.2308579279, 0.0000000000, -0.2290570633],
 [-0.9846391755, -0.9844559311, -0.9437771540, 0.0000000000],
 [0.0000000000, -0.3639913736, -0.9134701136, -0.6012717625],
 [-0.9561565201, 0.0000000000, -0.4270882632, -0.2608035802],
 [-0.5780579340, -0.8384023401, -0.5976190870, 0.0000000000],
 [0.0000000000, -0.8672219435, -0.5609217472, -0.6172965626],
 [-0.9151729961, -0.6965618861, 0.0000000000, -0.9247262662],
 [0.0000000000, -0.4985151214, -0.4206063934, -0.0591641271],
 [-0.0225296773, -0.9302017094, 0.0000000000, -0.2334024800],
 [0.0000000000, 0.0000000000, 0.0000000000, 0.0000000000],
 [0.0000000000, -0.3070905257, -0.1906124036, -0.3894854667],
 [-0.7979453377, 0.0000000000, -0.8297522125, -0.2747172531],
 [-0.7464523576, -0.5613249143, -0.3167131540, 0.0000000000],
 [0.0000000000, -0.6643002659, -0.0788330898, -0.4313914962]];*/

        let fake_pwm = vec![[0.0000000000, -0.9071355878, -0.9071355878, -0.9071355878],
 [-0.2455777698, 0.0000000000, -0.6942205884, -0.7608479938],
 [-0.5997385325, 0.0000000000, -0.8712945903, -0.1327220584],
 [-0.9721568978, 0.0000000000, -0.9721568978, -0.9721568978],
 [0.0000000000, -0.9274847350, -0.7453737927, -0.9274847350],
 [-0.9999718880, 0.0000000000, -0.9999718880, -0.9999718880],
 [-0.9910737175, 0.0000000000, -0.9910737175, -0.4136016247],
 [-0.9030173858, -0.8587768116, 0.0000000000, -0.9030173858],
 [-0.0000000000, -0.0000000000, 0.0000000000, -0.0000000002],
 [-0.9908276634, -0.9909689999, 0.0000000000, -0.9909689999],
 [-0.5544366450, -0.2101287318, 0.0000000000, -0.9015001136],
 [0.0000000000, -0.6571994400, -0.2726659948, -0.7984469325],
 [0.0000000000, -0.0202160245, -0.5525238105, -0.8975118436],
 [0.0000000000, -0.6111939575, -0.9513296432, -0.9518097588],
 [0.0000000000, -0.8944513527, -0.8944513527, -0.6553022338],
 [0.0000000000, -0.8175869737, -0.8175869737, -0.8175869737],
 [-0.8599825910, 0.0000000000, -0.8599825910, -0.8599825910],
 [-0.8861682151, -0.8861682151, 0.0000000000, -0.8861682151],
 [-0.8026550167, -0.8026550167, -0.8026550167, 0.0000000000]];
        let prep_pwm: Vec<[(usize, f64); BASE_L]> = fake_pwm.iter().map(|scores| {
            core::array::from_fn(|a| (a, scores[a]))
        }).collect();
        draw_pwm(&prep_pwm,"fake_pwm_arg.png");
    }

}
