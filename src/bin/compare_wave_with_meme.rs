
use std::fs::File;
use std::io::Read;


use motif_finder::ECOLI_FREQ;
use motif_finder::base::*;
use motif_finder::waveform::*;




use motif_finder::data_struct::{AllData, AllDataUse};

use statrs::distribution::Continuous;

use gzp::{deflate::Mgzip, par::compress::{ParCompress, ParCompressBuilder}, syncz::{SyncZ, SyncZBuilder}, par::decompress::{ParDecompress, ParDecompressBuilder},ZWriter, Compression};





 
use rand::prelude::*;




use plotters::prelude::*;
use plotters::prelude::full_palette::ORANGE;


fn main() {


    let args: Vec<String> = std::env::args().collect();

    let data_file = args[1].clone();

    let motif_set_meme = args[2].clone();

    let meme_name = motif_set_meme.clone();

    let mut try_bincode: ParDecompress<Mgzip> = ParDecompressBuilder::new().from_reader( File::open(data_file.as_str()).expect("You initialization file must be valid for inference to work!"));

    let mut buffer: Vec<u8> = Vec::new();
    let _ = try_bincode.read_to_end(&mut buffer);//We don't need to handle this specially, because this will create a different warning later
    let (pre_data, _bytes): (AllData, usize) = bincode::serde::decode_from_slice(&buffer, bincode::config::standard()).unwrap();

    let data = AllDataUse::new(&pre_data, 0.0).unwrap();

    buffer.clear();

    let mut rng = rand::thread_rng();

    println!("Freq change");
    let mut mot_set_meme: MotifSet = MotifSet::set_from_meme(&motif_set_meme, &data,None, f64::INFINITY, HandleImpossibleMotif::LeaveUnchanged,false, &mut rng).unwrap();

    let mut rmse = f64::INFINITY;

    let len = mot_set_meme.len();

    for id in 0..len {
        (mot_set_meme, rmse) = mot_set_meme.best_height_set_from_rmse_noise(id).unwrap();
    }

    println!("Best set RMSE {}\n set {:?} {} {}", rmse,mot_set_meme, mot_set_meme.ln_prior(), mot_set_meme.nth_motif(0).pwm_prior(data.data().seq()));


    let motif_set_bin = args[3].clone();

    let bin_name = motif_set_bin.clone();

    println!("mot set bin {motif_set_bin}");

    let mut try_bincode: ParDecompress<Mgzip> = ParDecompressBuilder::new().from_reader(File::open(motif_set_bin).unwrap());

    let mut buffer: Vec<u8> = Vec::new();
    let _ = try_bincode.read_to_end(&mut buffer);//We don't need to handle this specially, because this will create a different warning later

    let (prep_mot_set_bin, _bytes): (StrippedMotifSet, usize) = bincode::serde::decode_from_slice(&buffer, bincode::config::standard()).unwrap();

    let mut mot_set_bin = prep_mot_set_bin.reactivate_set(&data);

    mot_set_bin.sort_by_height();

    let wave_file = format!("{}_waves", args[4].clone());



    mot_set_meme.save_set_trace_comparisons(&mot_set_bin, &wave_file, "compare_wave", &args[5], &args[6]);
    
    if args.len() > 7 {

        for i in 7..args.len() {

            let num_motifs: usize = args[i].parse().unwrap();
            let sub_bin = mot_set_bin.only_n_strongest(num_motifs);

            mot_set_meme.save_set_trace_comparisons(&sub_bin, &wave_file, "compare_sub_wave", &args[5], &format!("TARJIM {num_motifs} Motifs"));


        }

    }



}
