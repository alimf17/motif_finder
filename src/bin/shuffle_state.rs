use std::fs;
use std::io::{Read, Write};
use std::error::Error;

use serde::*;
use serde::de::DeserializeOwned;
use motif_finder::base::*;

use gzp::{deflate::Mgzip, par::compress::{ParCompress, ParCompressBuilder}, syncz::{SyncZ, SyncZBuilder}, par::decompress::{ParDecompress, ParDecompressBuilder},ZWriter, Compression};

use rand::*;

fn main() {

    let args: Vec<String> = std::env::args().collect();

    let file_to_read: String = args[1].to_string();

    let output_dir: String = args[2].to_string();
    
    let output_name: String = args[3].to_string();

    
    let mut set : StrippedMotifSet = attempt_bincode_serde_read_with_decomp(&file_to_read, None, None).expect("This did not give a bincode of a trace NOR a single set");
    
    let mut rng = rand::thread_rng();

    set.shuffle_motifs(&mut rng);

    set.output_to_meme(None, &output_dir, &output_name);

}

pub fn attempt_bincode_serde_read_with_decomp<T: DeserializeOwned>(file_name: &str, preferred_scratch: Option<&mut Vec<u8>>, config: Option<bincode::config::Configuration>) -> Result<T, Box<dyn Error+Send+Sync>> {

    let config = config.unwrap_or(bincode::config::standard());

    let mut alter_scratch: Vec<u8> = Vec::new();

    let mut buffer_handle = if let Some(handle) = preferred_scratch {std::mem::drop(alter_scratch); handle} else {&mut alter_scratch};

    buffer_handle.clear();

    let mut file_handle = fs::File::open(file_name)?;

    file_handle.read_to_end(buffer_handle);

    let no_comp_trial: Result<(T, usize), _>  = bincode::serde::decode_from_slice(buffer_handle, config);

    if let Ok((no_comp, _)) = no_comp_trial {
        return Ok(no_comp);
    };

    buffer_handle.clear();

    let mut file_handle = fs::File::open(file_name)?;

    let mut read_file: ParDecompress<Mgzip> = ParDecompressBuilder::new().from_reader(file_handle);

    read_file.read_to_end(buffer_handle);

    let (comped, _): (T, usize) = bincode::serde::decode_from_slice(buffer_handle, config)?;

    Ok(comped)

}

