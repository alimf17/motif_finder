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
    let file_out_trp = "/Users/afarhat/Downloads/NC_000913.2_TrpR_lb_ratio_unstranded_minus_mean_25_data.bin";
    let file_out_arg = "/Users/afarhat/Downloads/NC_000913.2_ArgR_lb_ratio_unstranded_minus_mean_25_data.bin";


    let mut try_bincode = fs::File::open(file_out_trp).unwrap();

    let mut buffer: Vec<u8> = Vec::new();
    let _ = try_bincode.read_to_end(&mut buffer);//We don't need to handle this specially, because this will create a different warning later
    let pre_data: AllData = bincode::deserialize(&buffer).unwrap();

    let data = AllDataUse::new(&pre_data, 0.0).unwrap();

    buffer.clear();

    data.save_data_to_directory("/Users/afarhat/Downloads", "TrpR_plain");

    let mut try_bincode = fs::File::open(file_out_arg).unwrap();

    let _ = try_bincode.read_to_end(&mut buffer);//We don't need to handle this specially, because this will create a different warning later
    let pre_data: AllData = bincode::deserialize(&buffer).unwrap();
    let data = AllDataUse::new(&pre_data, 0.0).unwrap();
    data.save_data_to_directory("/Users/afarhat/Downloads", "ArgR_plain");
}
