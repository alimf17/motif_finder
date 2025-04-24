
use std::fs::File;
use std::io::Read;

use motif_finder::base::*;
use motif_finder::waveform::*;


use gzp::{deflate::Mgzip, par::compress::{ParCompress, ParCompressBuilder}, syncz::{SyncZ, SyncZBuilder}, par::decompress::{ParDecompress, ParDecompressBuilder},ZWriter, Compression};


use motif_finder::data_struct::{AllData, AllDataUse};

use statrs::distribution::Continuous;






 
use rand::prelude::*;




use plotters::prelude::*;
use plotters::prelude::full_palette::ORANGE;


fn main() {


    let args: Vec<String> = std::env::args().collect();

    let data_file = args[1].clone();

    let motif_set_meme = args[2].clone();

    let mut try_bincode: ParDecompress<Mgzip> = ParDecompressBuilder::new().from_reader( File::open(data_file.as_str()).expect("You initialization file must be valid for inference to work!"));


    let mut buffer: Vec<u8> = Vec::new();
    let _ = try_bincode.read_to_end(&mut buffer);//We don't need to handle this specially, because this will create a different warning later
    let (pre_data, _bytes): (AllData, usize) = bincode::serde::decode_from_slice(&buffer, bincode::config::standard()).unwrap();

    let data = AllDataUse::new(&pre_data, 0.0).unwrap();

    buffer.clear();

    let mut rng = rand::thread_rng();

    let mut mot_set: MotifSet = MotifSet::set_from_meme_max_selective(&motif_set_meme, &data, f64::INFINITY, false, &mut rng).unwrap();

    let mut rmse = f64::INFINITY;

    let len = mot_set.len();

    for id in 0..len {
        (mot_set, rmse) = mot_set.best_height_set_from_rmse(id).unwrap();
    }

    println!("Best set RMSE {}\n set {:?}", rmse,mot_set);

    let wave_file = format!("{}_waves", motif_set_meme);

    mot_set.recalced_signal().save_waveform_to_directory(&data, &wave_file, "total", &BLUE, false);

    //SAFETY: I used 3000 for the fragment length when generating the data I'm using now
    //let datas: Vec<AllDataUse> = background_lens.iter().map(|&a| unsafe{ data.with_new_fragment_length(a) }).collect();

    /*for _ in 0..cap {
    //for id in 0..trace.loan_active().len() {
        /*for base_id in 0..trace.loan_active().nth_motif(id).len() {
            for base_pair in trace.loan_active().nth_motif(id).pwm()[base_id].all_non_best() {
                for i in 0..2 {
*/
 //                   println!("PWM {id} Base position {base_id} Bp {:?} sign {}", base_pair, (-1_i32).pow(i));
    let mut active_set = trace.loan_active().clone();

    let pre_post = active_set.ln_posterior();

    //let E = -0.5_f64 + 0.49 * ((-1_i32).pow(i) as f64);

    //let (new_set, ln_post) = active_set.manual_change_energy(E , id, base_id, base_pair);

    let (mut new_set, scaled_ln_post) = active_set.propose_kill_motif(&mut rng).unwrap();


    let num_new_null = new_set.null_peak_scores().len() as isize;

    let ln_post = new_set.ln_posterior();

    let heights = vec![0.0001, 0.01, 0.1, 1.0, 2.0, 3_f64, 4_f64];

    let props: Vec<f64> = heights.iter().map(|&a| HEIGHT_PROPOSAL_DIST.ln_pdf(a)).collect();

    println!("{:?}", props);
    
    let weights: [f64; 16] = core::array::from_fn(|a| (a as f64)/16_f64);

    let mut beta: f64 = -f64::INFINITY;

    /*
    for weight in weights {

        let (_, like) = active_set.manual_change_energy(-weight, id, base_id, base_pair);

        let try_beta = weight*pre_post+(1.0-weight)*ln_post-like;

        beta = beta.max(try_beta);
    }*/

    /*println!("old");
    println!("{:?}", trace.loan_active().nth_motif(id).pwm()[base_id]);
    println!("new");
    println!("{:?}", new_set.nth_motif(id).pwm()[base_id]);
*/
    /*
    posterior_ratios.push(ln_post-pre_post);
    let diff_post = scaled_ln_post-pre_post;

    evaluated_ratios.push(diff_post);

    println!("{pre_post} {ln_post} {} nullnum diff{} like diff {} post diff {} height {} alpha {} beta {} ratio {}", scaled_ln_post-ln_post, numnull-num_new_null,diff_post+(trace.loan_active().ln_prior()-new_set.ln_prior()), -trace.loan_active().ln_prior()+new_set.ln_prior(), trace.loan_active().nth_motif(0).peak_height(), diff_post, beta, (diff_post)/beta);
        /*        }
            }
        }
    }*/*/
    }

    println!("posterior ratios: \n {:?}", posterior_ratios);
    println!("evaluated ratios: \n {:?}", evaluated_ratios);

    let copy_wave = data.data().clone();

    let copy_noise = copy_wave.produce_noise(&data);

    let mut noise_vec = copy_noise.resids();

    for i in 0..noise_vec.len() { noise_vec[i] = data.background_ref().sample(&mut rng) }

    let new_noise = Noise::new(noise_vec, Vec::new(), data.background_ref());

    println!("noise {:?}", new_noise.resids());

    let parameter = copy_noise.ad_calc(copy_wave.spacer());

    println!("par {parameter}");
    
    let parameter = new_noise.ad_calc(copy_wave.spacer());

    println!("par {parameter}");

    println!("like {}", Noise::ad_like(parameter));
*/
}
