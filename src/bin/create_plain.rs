use std::fs::File;
use std::io::Read;

use motif_finder::base::*;
use motif_finder::waveform::*;


use gzp::{deflate::Mgzip, par::compress::{ParCompress, ParCompressBuilder}, syncz::{SyncZ, SyncZBuilder}, par::decompress::{ParDecompress, ParDecompressBuilder},ZWriter, Compression};


use motif_finder::data_struct::{AllData, AllDataUse};

use statrs::distribution::Continuous;







//use rand::prelude::*;




use plotters::prelude::*;
use plotters::prelude::full_palette::ORANGE;

fn main() {
    //let file_out_trp = "/Users/afarhat/Downloads/NC_000913.2_TrpR_lb_ratio_unstranded_minus_mean_25_data.bin";

//    let file_out_trp = "/expanse/lustre/scratch/alimf/temp_project/ProcessedData/IPOD_HR_min_height_1.0_try_again.bin";

    let file_out_trp = "/work/hdd/bbgs/alimf/AlterProcessedData/IPOD_HR_unstat_omit_neg_wind.bin";

    let mut try_bincode : ParDecompress<Mgzip> = ParDecompressBuilder::new().from_reader( File::open(file_out_trp).expect("You initialization file must be valid for inference to work!"));

    let mut buffer: Vec<u8> = Vec::new();
    let _ = try_bincode.read_to_end(&mut buffer);//We don't need to handle this specially, because this will create a different warning later
    let (pre_data, _bytes): (AllData, usize) = bincode::serde::decode_from_slice(&buffer, bincode::config::standard()).unwrap();

    let data = AllDataUse::new(&pre_data, 0.0).unwrap();

    buffer.clear();

    data.save_data_to_directory("/work/hdd/bbgs/alimf", "IPOD_HR_unstat_no_neg_plain");

}
