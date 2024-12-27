use std::fs;
use std::io::Read;


use motif_finder::ECOLI_FREQ;
use motif_finder::base::*;
use motif_finder::waveform::*;




use motif_finder::data_struct::{AllData, AllDataUse};

use statrs::distribution::Continuous;







use rand::prelude::*;




use plotters::prelude::*;
use plotters::prelude::full_palette::ORANGE;

fn main() {

    let args: Vec<String> = std::env::args().collect();

    let file_out = args[1].clone();

    let motif_set_sup_set = args[2].clone();

    let motif_set_sub_set = args[3].clone();

    
    let mut try_bincode = fs::File::open(file_out).unwrap();

    let mut buffer: Vec<u8> = Vec::new();
    let _ = try_bincode.read_to_end(&mut buffer);//We don't need to handle this specially, because this will create a different warning later
    let pre_data: AllData = bincode::deserialize(&buffer).unwrap();

    let data = AllDataUse::new(&pre_data, 0.0).unwrap();

    buffer.clear();

    let mut rng = rand::thread_rng();


    let mut try_bin_sup = fs::File::open(motif_set_sup_set).unwrap();

    let _ = try_bin_sup.read_to_end(&mut buffer);

    let mut stripped_sup: StrippedMotifSet = bincode::deserialize(&buffer).unwrap();

    stripped_sup.sort_by_height();

    //let mut super_set = stripped_sup.reactivate_set(&data);

    buffer.clear();

    let mut try_bin_sub = fs::File::open(motif_set_sub_set).unwrap();

    let _ = try_bin_sub.read_to_end(&mut buffer);

    let mut stripped_sub: StrippedMotifSet = bincode::deserialize(&buffer).unwrap();

    stripped_sub.sort_by_height();

    //let mut sub_set = stripped_sub.reactivate_set(&data);

    buffer.clear();

    //I only care about naive assignments, hence not doing a fully optimized assignment problem


    let assigned = stripped_sub.pair_to_other_set_rmse_data(&stripped_sup, &data);

    for (i, check) in assigned.into_iter().enumerate() {

        if let Some((rmse, sup_mot)) = check {
            println!("{} {} {} {} {}", stripped_sub.get_nth_motif(i).peak_height(), sup_mot.peak_height(), stripped_sub.get_nth_motif(i).best_motif_string(), sup_mot.best_motif_string(), rmse);
        };

    }

}
