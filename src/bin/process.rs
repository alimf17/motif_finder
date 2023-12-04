use std::fs; 

use motif_finder::base::*;
use motif_finder::base::{SQRT_2, SQRT_3};

use motif_finder::{NECESSARY_MOTIF_IMPROVEMENT};

use log::warn;

use kmedoids;

use ndarray::prelude::*;

use regex::Regex;

use poloto;
use plotters::prelude::*;
use plotters::prelude::full_palette::*;

//use spectrum_analyzer::{samples_fft_to_spectrum, FrequencyLimit};


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

    let mut min_chain: usize = burn_in.unwrap_or(0);

    let potential_prior_change: Option<f64> = args.get(6).map(|a| a.parse::<f64>().ok()).flatten();

    match potential_prior_change {
        Some(credibility) => {
            if !(credibility > 0.0) {panic!("Motif prior threshold must be a valid strictly positive float");}
            let mut w = NECESSARY_MOTIF_IMPROVEMENT.write().expect("This should be the only thread writing or reading to this");
            *w = credibility;
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


    let max_max_length = 100000;
    //This is the code that actually sets up our independent chain reading
    let mut set_trace_collections: Vec<SetTraceDef> = Vec::with_capacity(max_chain-min_chain);
    for chain in 0..num_chains {
        let base_str = format!("{}/{}_{}", out_dir.clone(), base_file, UPPER_LETTERS[chain]);
        //let regex = Regex::new(&(base_str.clone()+format!("_{}_trace_from_step_", min_chain).as_str()+"\\d{7}.json")).unwrap();
        let regex = Regex::new(&(base_str.clone()+"_trace_from_step_"+"\\d{7}.json")).unwrap();
        let directory_iter = fs::read_dir(&out_dir).expect("This directory either doesn't exist, you're not allowed to touch it, or isn't a directory at all!");
        
        let mut chain_files = directory_iter.filter(|a| regex.is_match(a.as_ref().unwrap().path().to_str().unwrap())).map(|a| a.unwrap().path().to_str().unwrap().to_string()).collect::<Vec<_>>();

        chain_files.sort();

        let mut iter_files = chain_files.iter();

        let file_y = iter_files.next();

        println!("file {:?} ", file_y);

        set_trace_collections.push(serde_json::from_str(&fs::read_to_string(file_y.expect("Must have matches to proceed"))
                                                                          .expect("File must exist or wouldn't appear"))
                                                                          .expect("All read in files must be correct json!"));

        for file_name in iter_files {
            let mut interim: SetTraceDef = serde_json::from_str(&fs::read_to_string(file_name).expect("File must exist or wouldn't appear")).expect("All read in files must be correct json!");
            interim.reduce(max_max_length);
            set_trace_collections[chain].append(interim);
        }


        if (min_chain+1) < max_chain {for bead in (min_chain+1)..max_chain {

            let regex = Regex::new(&(base_str.clone()+format!("_{}_trace_from_step_{}d{{7}}.json", bead, '\\').as_str())).unwrap();
            let directory_iter = fs::read_dir(&out_dir).expect("This directory either doesn't exist, you're not allowed to touch it, or isn't a directory at all!");

            let mut chain_files = directory_iter.filter(|a| regex.is_match(a.as_ref().unwrap().path().to_str().unwrap())).map(|a| a.unwrap().path().to_str().unwrap().to_string()).collect::<Vec<_>>();

            chain_files.sort();

            let iter_files = chain_files.iter();

            for file_name in iter_files {
                let interim: SetTraceDef = serde_json::from_str(&fs::read_to_string(file_name).expect("File must exist or wouldn't appear")).expect("All read in files must be correct json!");
                
                /*if interim.len() >= 5 {
                    println!("{} {:?} inter", file_name, interim.index_into(0..5));
                } else{
                    println!("{} {:?} inter", file_name, interim.index_into(0..1));
                }*/
                set_trace_collections[chain].append(interim);
            }




        }}
    }

    //set_trace_collections is now the chains we want

    for (chain, collection) in set_trace_collections.iter().enumerate() {
        let chain_name = format!("{}_{}",base_file, UPPER_LETTERS[chain]);
        collection.save_best_trace(&out_dir, &chain_name);//, data: &Waveform, background: &Background)
    }
    let mut plot_post = poloto::plot(format!("{} Ln Posterior", base_file), "Step", "Ln Posterior");
    println!("{}/{}_ln_post.svg", out_dir.clone(), base_file);
    let plot_post_file = fs::File::create(format!("{}/{}_ln_post.svg", out_dir.clone(), base_file).as_str()).unwrap();
    for (i, trace) in set_trace_collections.iter().enumerate() {
        let letter = UPPER_LETTERS[i];
        let tracey = trace.ln_posterior_trace();
        //let trace_mean = tracey.iter().sum::<f64>()/(tracey.len() as f64);
        //let samples: Vec<f32> = tracey.iter().map(|a| (a-trace_mean) as f32).collect();
        /*let res = samples_fft_to_spectrum(
        &samples[0..16384],
        samples.len() as u32,
        FrequencyLimit::All,
        None,).unwrap();
        let daaa = res.data().iter().map(|(a, b)| (a.val(), b.val().powi(2)/(4.0*16384.))).collect::<Vec<_>>();
        //println!("FFT {:?}", daaa.iter().map(|(b,a)| (b, a/(daaa[0].1))).collect::<Vec<_>>());*/
        plot_post.line(format!("Chain {}", letter), tracey.into_iter().enumerate().map(|(a, b)| (a as f64, b)));//.xmarker(0).ymarker(0);
    };
    plot_post.simple_theme(poloto::upgrade_write(plot_post_file));

    let mut plot_tf_num = poloto::plot(format!("{} Motif number", base_file), "Step", "Motifs");
    let plot_tf_num_file = fs::File::create(format!("{}/{}_tf_num.svg", out_dir.clone(), base_file).as_str()).unwrap();
    for (i, trace) in set_trace_collections.iter().enumerate() {
        let letter = UPPER_LETTERS[i];
        plot_tf_num.line(format!("Chain {}", letter), trace.motif_num_trace().into_iter().enumerate().map(|(a, b)| (a as f64, b))).xmarker(0).ymarker(0);
    };
    plot_tf_num.simple_theme(poloto::upgrade_write(plot_tf_num_file));
    
    let mut plot_like = poloto::plot(format!("{} Ln Likelihood", base_file), "Step", "Ln Likelihood");
    let plot_like_file = fs::File::create(format!("{}/{}_ln_like.svg", out_dir.clone(), base_file).as_str()).unwrap();
    for (i, trace) in set_trace_collections.iter().enumerate() {
        let letter = UPPER_LETTERS[i];
        plot_like.line(format!("Chain {}", letter), trace.ln_likelihood_trace().unwrap().into_iter().enumerate().map(|(a, b)| (a as f64, b)));//.xmarker(0).ymarker(0);
    };


    plot_like.simple_theme(poloto::upgrade_write(plot_like_file));
    
    let mut rng = rand::thread_rng();
    
    //let ref_per_chain: usize = 5;

    //let num_ref_pwm = ref_per_chain*set_trace_collections.len();

    let mut min_len: usize = usize::MAX;
    for trace in set_trace_collections.iter() {
        min_len = min_len.min(trace.len());
        println!("Min len {}", min_len);
    }

    let cluster_per_chain: usize = min_len.min(10000);

    let motif_num_traces = set_trace_collections.iter().map(|a| a.motif_num_trace()).collect::<Vec<_>>();

    //TODO: generate motif_num_traces plot here
    println!("Motif num rhat: {}", rhat(&motif_num_traces, min_len));

    let total_sets = min_len*set_trace_collections.len();
    let tf_num = (motif_num_traces.into_iter().map(|a| tail_slice(&a, min_len).iter().sum::<f64>()).sum::<f64>()/(total_sets as f64)).floor() as usize;

    let num_sort_mots: usize = cluster_per_chain*set_trace_collections.len();
    
    let mut clustering_motifs: Vec<Motif> = Vec::with_capacity(num_sort_mots);



    for trace in set_trace_collections.iter() {
        clustering_motifs.append(&mut (trace.ret_rand_motifs(cluster_per_chain, &mut rng)));
    }

    let mut meds = kmedoids::random_initialization(num_sort_mots, tf_num, &mut rng);
    
   
    println!("clust {:?}", &clustering_motifs[0..5]);
    let dist_array = establish_dist_array(&clustering_motifs);
    let (_loss, _assigns, n_iter, n_swap): (f64, Vec<usize>,_, _) = kmedoids::fasterpam(&dist_array, &mut meds, 100);

    println!("iter {} swap {}", n_iter, n_swap);
    println!("min {}", min_len);
    for (i, medoid) in meds.iter().enumerate() {
        println!("Medoid {}: \n {:?}", i, medoid);
        println!("Rhat medoid {}: {}", i, rhat(&set_trace_collections.iter().map(|a| a.trace_min_dist(&clustering_motifs[*medoid])).collect::<Vec<_>>(), min_len));
        let mut num_good_motifs: usize = 0; 
        let mut good_motifs_count: Vec<usize> = vec![0];
        let mot_size = set_trace_collections[0].len() as f64;
        let distance_cutoff = mot_size*0.5753378-0.0*0.239545*mot_size.sqrt();//I picked this to be two standard deviations below the sum of independent mot_size independent weibulls with k = 2.5776088 and lambda = 0.6479129
        let trace_scoop = set_trace_collections.iter().map(|a| {
            let set_extracts = a.extract_best_motif_per_set(&clustering_motifs[*medoid], min_len, distance_cutoff);
            num_good_motifs+= set_extracts.len();
            good_motifs_count.push(num_good_motifs);
            set_extracts
        }).flatten().collect::<Vec<_>>();
        let pwm_traces = create_offset_traces(trace_scoop);
        graph_tetrahedral_traces(&pwm_traces, &good_motifs_count,0.95, format!("{}/{}_new_tetrahedra_{}.png", out_dir.clone(), base_file, i).as_str());
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


pub fn create_offset_traces(best_motifs: Vec<(Motif, (f64, isize, bool))>) -> Array3<f64> {

    let mut min_offset: isize = isize::MAX;
    let mut max_offset_plus_len: isize = isize::MIN;

    let num_samples = best_motifs.len();
    let pwms_offsets = best_motifs.into_iter()
                                  .map(|(a, (_, b, c))| {
                                      min_offset = min_offset.min(b);
                                      max_offset_plus_len = max_offset_plus_len.max(b+(a.len() as isize));
                                      (if c {a.rev_complement()} else {a.pwm()}, b)
                                  }).collect::<Vec<(Vec<Base>, isize)>>();

    let neutral: f64 = 0.0;

    let num_bases = (max_offset_plus_len-min_offset) as usize;


    let mut samples = Array3::<f64>::zeros([num_samples, num_bases, BASE_L-1]);

    for (k, (pwm, offset)) in pwms_offsets.into_iter().enumerate() {

        for j in 0..num_bases {

            //let mut slice = samples.slice_mut(s![k,j,..]);  //I went with this because it's the fastest possible assignment.
                                                            //It will slow down calculations of credible intervals and means,
                                                            //But I do that once, not 10s of thousands of times. I did not
                                                            //benchmark this, though

            let ind = (j as isize)-(offset-min_offset);


            if ind < 0 || ind >= (pwm.len() as isize) {
                for i in 0..(BASE_L-1) {samples[[k,j,i]] = neutral;}
            } else {
                let ind = ind as usize;
                let probs = pwm[ind].as_simplex();
                for i in 0..(BASE_L-1) {samples[[k,j,i]] = probs[i];}
            }
        }

    }

    samples

}

pub fn graph_tetrahedral_traces(samples: &Array3::<f64>, good_motifs_count: &Vec<usize>, credible: f64, file_name: &str){

    let (_ , num_bases, _) = samples.dim();

    let num_rows = if (num_bases & 3) == 0 {num_bases/4} else{ num_bases/4 +1 } as u32; 

    let plot = BitMapBackend::new(file_name, (2600, num_rows*600)).into_drawing_area();

    plot.fill(&full_palette::WHITE).unwrap();

    let panels = plot.split_evenly((num_rows as usize, 4));

    let cis = create_credible_intervals(samples, credible);

    for j in 0..num_bases {
        

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
        
        
        for m in 0..(good_motifs_count.len()-1){
            if good_motifs_count[m] < good_motifs_count[m+1] {

                let slice_bases: Vec<(f64, f64, f64)> = ((good_motifs_count[m])..(good_motifs_count[m+1])).map(|k| (samples[[k,j,0]], samples[[k,j,1]], samples[[k,j,2]]) ).collect::<Vec<_>>();

                let tetra_traces = create_tetrahedral_traces(&slice_bases);

            
                for little_trace in tetra_traces.iter() {
                    chart.draw_series(little_trace.iter().step_by(10).map(|&a| Circle::new(a,2_u32,  PALETTE[(m+BASE_L) % 26]))).unwrap();
                }
            } else { warn!("The motifs in trace {} are nowhere close to the medoid", m);}
        }

        println!("Creating credible region at {} samples", credible);
        
 
        let prep_pwm: Vec<[(usize, f64); BASE_L]> = cis.iter().map(|(_, tetrahedral_mean)| Base::simplex_to_base(tetrahedral_mean).seqlogo_heights()).collect();

        draw_pwm(&prep_pwm, &format!("{}_pwm.png", file_name));
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

    let len = map_heights.len();
    let plot = BitMapBackend::new(file_name, (600, (len as u32)*600)).into_drawing_area();

    plot.fill(&full_palette::WHITE).unwrap();

    let mut chart = ChartBuilder::on(&plot)
            .set_label_area_size(LabelAreaPosition::Left, 40)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .caption("PWM", ("sans-serif", 40))
            .build_cartesian_2d(0..(len+1), (0_f64)..2_f64).unwrap();


    chart.configure_mesh().draw().unwrap();

    chart.draw_series(map_heights.iter().enumerate().map(|(i, hs)| {

        println!("hs {:?}", hs);
        let mut end_heights: [f64; BASE_L] = [0.0; BASE_L];

        for j in 0..(BASE_L-1) { end_heights[j] = hs[j+1].1; }

        let pos = i;
        end_heights.into_iter().zip(hs.iter()).enumerate().map(move |(j, (e, h))| {
            println!("{} {} {:?}", j, e, h);
            Rectangle::new([(pos, h.1), (pos+1, e)],BASE_COLORS[h.0].filled())
        }) 
    }).flatten()).unwrap();

}

//We want to eat best_motifs so that we can save on memory
//You'll notice that this entire function is designed to drop memory I don't
//need anymore as much as humanly possible.
pub fn create_credible_intervals(samples: &Array3<f64>, credible: f64) -> Vec<(Vec<(f64, f64, f64)>, [f64; BASE_L-1])>{

    if credible <= 0.0 {
        panic!("Can't make credible intervals of literally at most zero length!");
    } else if credible >= 1.0 {
        warn!("Will make credible intervals out of literally all of the data!");
    }

    //I haven't removed num_bases entirely because I might need it if I uncomment
    //some currently commented code
    let (num_samples, _num_bases, _) = samples.dim();
    let credible = credible.min(1.0);
    let num_interval = ((num_samples as f64)*credible).floor() as u32;

    let z_steps: usize = ((4./3.)/INTERVAL_CELL_LENGTH).floor() as usize;
    let x_steps: usize = (SQRT_2/INTERVAL_CELL_LENGTH).floor()  as usize;
    let y_steps: usize = ((2.*SQRT_2/SQRT_3)/INTERVAL_CELL_LENGTH).floor() as usize;

    let zs = (0..z_steps).map(|a| -1./3.+(a as f64)*INTERVAL_CELL_LENGTH).collect::<Vec<_>>();
    let xs = (0..x_steps).map(|a| -SQRT_2/3.+(a as f64)*INTERVAL_CELL_LENGTH).collect::<Vec<_>>();
    let ys = (0..y_steps).map(|a| -SQRT_2/SQRT_3+(a as f64)*INTERVAL_CELL_LENGTH).collect::<Vec<_>>();




    /*
    //We know that the number of cells scales with 1/INTERVAL_CELL_LENGTH cubed. 8.*SQRT_3/27. is
    //the volume of the tetrahedron.
    let mut actual_cells: Vec<(usize, usize, usize)> = Vec::with_capacity((8.*SQRT_3/27.*INTERVAL_CELL_LENGTH.powi(-3)).ceil() as usize); 


    //Establishes which indices actually are in the tetrahedron
    for z_ind in 0..z_steps{
        let z = zs[z_ind];
        let x_lo_thresh = SQRT_2*z/4-SQRT_2/4;
        let x_hi_thresh =  1./SQRT_2-z/SQRT_2;
        for x_ind in 0..x_steps {
            let x = xs[x_ind];
            if (x >= x_lo_thresh) && (x <= x_hi_thresh) {
                let y_lo_thresh =  (z-1.)/(SQRT_2*SQRT_3)+x/SQRT_3;
                let y_hi_thresh = -(z-1.)/(SQRT_2*SQRT_3)-x/SQRT_3;
                for y_ind in 0..ysteps {
                    let y = ys[y_ind];
                    if (y >= y_lo_thresh) && (y <= y_hi_thresh) {
                        actual_cells.push((x_ind, y_ind, z_ind));
                    }
                }
            }

        }
    }

    //Actual cells can no longer mutate
    let actual_cells = actual_cells;
    */              
    
    //This will yield a vector where the ith position corresponds to the cells of the
    //credible region for the ith base
    let credible_cells_vec = samples.axis_iter(ndarray::Axis(1)).map(|base_vecs| {

        //I picked the smallest unsized int that I could: I can have more than 65000 points in a cell
        //theoretically (hard but not impossible). But I can't have more than 4 billion: a million
        //steps still takes a couple of weeks. Billion would just not be practical
        //The unusual order here is for memory accessibility.
        let mut cell_counts: Array3<u32> = Array3::<u32>::zeros([x_steps, y_steps, z_steps]);

        for base_vector in base_vecs.axis_iter(ndarray::Axis(0)) {

            let x_ind = ((base_vector[0]-(-SQRT_2/3.))/INTERVAL_CELL_LENGTH).floor() as usize;
            let y_ind = ((base_vector[1]-(-SQRT_2/SQRT_3))/INTERVAL_CELL_LENGTH).floor() as usize;
            let z_ind = ((base_vector[2]-(-1.0/3.))/INTERVAL_CELL_LENGTH).floor() as usize;
            if x_ind > x_steps { println!("about to break x {} {}", x_ind, x_steps);}
            if y_ind > y_steps { println!("about to break y {} {}", y_ind, y_steps);}
            if z_ind > z_steps { println!("about to break z {} {}", z_ind, z_steps);}
            cell_counts[[x_ind, y_ind, z_ind]] += 1;

        }

 
        let mut cells_and_counts = cell_counts.indexed_iter().map(|(a, &b)| (a, b)).collect::<Vec<_>>();

        //b.cmp(a) sorts from greatest to least
        cells_and_counts.sort_unstable_by(|(_, a), (_, b)| b.cmp(a));

        let mut to_interval: u32 = 0;
        let mut index: usize = 0;

        while to_interval < num_interval {
            to_interval += cells_and_counts[index].1;
            index += 1;
        }

        let region = cells_and_counts.drain(0..index).map(|(a, _)| (xs[a.0], ys[a.1], zs[a.2])).collect::<Vec<(f64, f64, f64)>>();

        let posterior_sum = region.iter().fold((0.0, 0.0, 0.0), |acc, x| (acc.0+x.0, acc.1+x.1, acc.2+x.2));
        let posterior_mean = [posterior_sum.0/(index as f64), posterior_sum.1/(index as f64), posterior_sum.2/(index as f64)];

        (region, posterior_mean)

    }).collect::<Vec<_>>();
    /*
    let mut means = Array2::<f64>::zeros([num_bases, BASE_L-1]);
    let mut lower_ci = Array2::<f64>::zeros([num_bases, BASE_L-1]);
    let mut upper_ci = Array2::<f64>::zeros([num_bases, BASE_L-1]);
        
        
    for j in 0..num_bases {
        for i in 0..(BASE_L-1) {

            let mut data = samples.slice(s![..,j, i]).clone().to_vec();
            data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

            let mut min_range_start: usize = 0;
            let mut min_range: f64 = f64::INFINITY;

            for d in 0..(data.len()-num_interval) {

                let range = data[d+num_interval]-data[d];
                if range < min_range {
                    min_range_start = d;
                    min_range = range;
                }

            }

            lower_ci[[j,i]] = data[min_range_start];
            upper_ci[[j,i]] = data[min_range_start+num_interval];
            means[[j, i]] = data.mean(); //This consumes the data clone
        }
    }


    [lower_ci, means, upper_ci]

    */

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
        let fake_pwm = vec![[0.1_f64, -0.1, -0.1], [-0.1, 0.1, -0.1], SIMPLEX_VERTICES_POINTS[2], [SQRT_2/3., 0., 1./3.], [0., 0., 0.]]; 
        let prep_pwm: Vec<[(usize, f64); BASE_L]> = fake_pwm.iter().map(|tetrahedral_mean| Base::simplex_to_base(tetrahedral_mean).seqlogo_heights()).collect();
        draw_pwm(&prep_pwm,"fake_pwm.png");
    }

}
