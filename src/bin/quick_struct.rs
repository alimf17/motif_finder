
use std::fs;
use std::io::Read;

use motif_finder::base::*;
use motif_finder::waveform::*;




use motif_finder::data_struct::{AllData, AllDataUse};

use statrs::distribution::Continuous;







//use rand::prelude::*;




use plotters::prelude::*;
use plotters::prelude::full_palette::ORANGE;


fn main() {

    //let file_out = "/Users/afarhat/Downloads/NC_000913.2_GSM639836_TrpR_Trp_ln_ratio_25_data.bin";
    //let trace_file =  "/Users/afarhat/Downloads/removedHMC_20240407_GSM639836_TrpR_Trp_ln_ratio_D_1_trace_from_step_0999999.bin";

    //let file_out = "/Users/afarhat/Downloads/NC_000913.2_TrpR_Trp_lb_ratio_unstranded_minus_mean_25_data.bin";
    let file_out = "/Users/afarhat/Downloads/NC_000913.2_ArgR_lb_ratio_unstranded_minus_mean_25_data.bin";
    let trace_file =  "/Users/afarhat/Downloads/newLikelihood_20241008_filt_0.00_ArgR_Arg_D_0_trace_from_step_0004140.bin";

    //let file_out = "/Users/afarhat/Downloads/NC_000913.2_ArgR_Arg_TrpR_Trp_ln_ratio_25_data.bin";
    //let trace_file =  "/Users/afarhat/Downloads/ReportingReplicaExchange_20240421_ArgR_Arg_TrpR_Trp_ln_ratio_D_1_trace_from_step_0000004.bin";


/*    let regulon_argR_raw: Vec<[f64; BASE_L]> = vec![
    
         [-0.6641540527, -0.5166015625, -0.6641540527, 0.0000000000],
         [-0.7500305176, -0.7720642090, 0.0000000000, -0.8171386719],
         //this is the non permuted version
         //[0.0000000000, -0.1327819824, -0.3785400391, -0.5770874023],
         [-0.1327819824, 0.0000000000, -0.3785400391, -0.5770874023],
         [0.0000000000, -0.6918945312, -0.6481628418, -0.6268005371],
         [-0.7412719727, -0.8073730469, -0.8531799316, 0.0000000000],
         //this is the non permuted version 
         //[0.0000000000, -0.4613342285, -0.2660827637, -0.6156616211],
         [-0.6156616211, -0.4613342285, -0.2660827637, 0.0000000000],
         [-0.0320739746, -0.5305175781, -0.5536193848, 0.0000000000],
         [-0.4103088379, -0.6138305664, -0.7040405273, 0.0000000000],
         [-0.1520080566, -0.3020324707, -0.5611267090, 0.0000000000],
         [0.0000000000, -0.9272460938, -0.9038085938, -0.8135986328],
         [-0.7757873535, -0.7978210449, -0.8894653320, 0.0000000000],
         [-0.4417114258, -0.5536193848, -0.0984191895, 0.0000000000],
         [-0.8912963867, 0.0000000000, -0.8912963867, -0.8011169434]

    ];
*/

    let regulon_ArgR_nums : Vec<[f64; BASE_L]> = vec![ 
       [0.065574, 0.180328, 0.065574, 0.688525],
       [0.081967, 0.065574, 0.819672, 0.032787],
       [0.360656, 0.491803, 0.147541, 0.000000],
       [0.721311, 0.065574, 0.098361, 0.114754],
       [0.098361, 0.049180, 0.016393, 0.836066],
       [0.016393, 0.131148, 0.295082, 0.557377],
       [0.459016, 0.032787, 0.016393, 0.491803],
       [0.245902, 0.081967, 0.016393, 0.655738],
       [0.327869, 0.196721, 0.000000, 0.475410],
       [0.901639, 0.000000, 0.016393, 0.081967],
       [0.081967, 0.065574, 0.000000, 0.852459],
       [0.098361, 0.016393, 0.393443, 0.491803],
       [0.016393, 0.885246, 0.016393, 0.081967],
       [0.754098, 0.081967, 0.049180, 0.114754],
];
        /*let mut refine_argr: Vec<[f64; BASE_L]> = regulon_argR_raw.iter().map(|a| {]
          let max = *a.iter().max_by(|b,c| b.partial_cmp(c).unwrap()).unwrap();]
          core::array::from_fn(|i|, 0.5_f64.max(a[i]/max))]
          }).collect();*/
/*]
    let regulon_TrpR_raw: Vec<[f64; BASE_L]> = vec![]
                        [0.700000,, 0.300000,, 0.000000,, 0.000000],]
						[0.000000,, 0.000000,, 0.600000,, 0.400000],]
						//[0.200000,, 0.100000,, 0.000000,, 0.700000], //This is the official base in regulon for what's below]
						[0.700000,, 0.100000,, 0.000000,, 0.200000],]
						[0.000000,, 0.000000,, 0.700000,, 0.300000],]
						[0.600000,, 0.000000,, 0.000000,, 0.400000],]
						[0.900000,, 0.000000,, 0.100000,, 0.000000],]
						[0.000000, 1.000000,, 0.000000,, 0.000000],]
						[0.000000,, 0.000000,, 0.100000,, 0.900000],]
						[0.800000, 0.000000, 0.200000, 0.000000],
						[0.000000, 0.000000, 1.000000, 0.000000],
						[0.000000, 0.000000, 0.000000, 1.000000],
						//[0.500000, 0.000000, 0.200000, 0.300000], //This is the official base in regulon for what's below
						[0.200000, 0.000000, 0.500000, 0.300000],
						[0.100000, 0.700000, 0.000000, 0.200000],
						[0.600000, 0.000000, 0.400000, 0.000000],
    ];*/
   let mut refine_argr = Motif::raw_pwm(regulon_ArgR_nums.iter().map(|&a| Base::new(a)).collect(), 2.5, KernelWidth::Wide, KernelVariety::Gaussian);
   //let mut refine_trpr = Motif::raw_pwm(regulon_TrpR_raw.iter().map(|a| Base::new(core::array::from_fn(|i| a[i].log2()))).collect(), 4.0, KernelWidth::Wide, KernelVariety::Gaussian);

//   refine_argr = Motif::raw_pwm(refine_argr.rev_complement(), refine_argr.peak_height(), KernelWidth::Wide, KernelVariety::Gaussian);


   println!("ref {:?}", refine_argr);
   //println!("ref {:?}", refine_trpr);

    let mut try_bincode = fs::File::open(file_out).unwrap();
    //let mut try_bin_trace = fs::File::open(trace_file).unwrap();

    let mut buffer: Vec<u8> = Vec::new();
    let _ = try_bincode.read_to_end(&mut buffer);//We don't need to handle this specially, because this will create a different warning later
    let pre_data: AllData = bincode::deserialize(&buffer).unwrap();

    let data = AllDataUse::new(&pre_data, 0.0).unwrap();

    buffer.clear();

    //let _ = try_bin_trace.read_to_end(&mut buffer);
    //let pre_trace: SetTraceDef = bincode::deserialize(&buffer).unwrap();

    let mut rng = rand::thread_rng();

    //let trace = pre_trace.get_set_trace(&data, &mut rng, None);
    //println!("{:?}", trace.loan_active());
    //mod_save_trace(&trace, &data);

    //let numnull = trace.loan_active().null_peak_scores().len() as isize;

    let cap = 100_usize;

    let mut posterior_ratios: Vec<f64> = Vec::with_capacity(cap);
    let mut evaluated_ratios: Vec<f64> = Vec::with_capacity(cap);

    let base_background = data.background_ref().get_sd_df();

    let background_lens = vec![100_f64, 200., 300., 500., 800., 1000., 1500., 2000., 2500., 3000.];

    //SAFETY: I used 3000 for the fragment length when generating the data I'm using now
    //let datas: Vec<AllDataUse> = background_lens.iter().map(|&a| unsafe{ data.with_new_fragment_length(a) }).collect();


    for fragment in background_lens {

        //SAFETY: I used 3000 for the fragment length when generating the data I'm using now
        let data_use = unsafe{ data.with_new_fragment_length(fragment/(2.0*3.0)) };

        let mut set = MotifSet::manual_set(&data_use, refine_argr.clone());
        //let mut set = MotifSet::manual_set(&data_use, refine_trpr.clone());

        let num_moves = 1000_usize;
        let mut accep = 0_usize;

        /*
        for _ in 0..num_moves {

    let mut rng = rand::thread_rng();
            let (try_set, modded) = set.propose_height_move_custom(&mut rng, 0.1).unwrap();

            let accept = MotifSet::accept_test(set.ln_posterior(), modded, 1.0, &mut rng);

            if accept {
                set = try_set;
                accep += 1;
            }

        }

*/
        println!("Fragment length {fragment} ln LIKELIHOOD {} ln prior {} height {} height prop {}", set.ln_likelihood(), set.ln_prior(), set.nth_motif(0).peak_height(), (accep as f64)/(num_moves as f64));

        //set.save_set_trace_and_sub_traces("/Users/afarhat/Downloads", format!("TrpR_Fragment_higher_{:04}", fragment).as_str());
        set.save_set_trace_and_sub_traces("/Users/afarhat/Downloads", format!("ArgR_Fragment_init_attempt_{:04}", fragment).as_str());
        //set.save_set_trace_and_sub_traces("/Users/afarhat/Downloads", format!("ArgR_Fragment_higher_{:04}", fragment).as_str());

        let (new, _) = set.propose_new_motif(&mut rng).unwrap();

        new.save_set_trace_and_sub_traces("/Users/afarhat/Downloads", format!("tryout_attempt_{:04}", fragment).as_str());

   }



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

