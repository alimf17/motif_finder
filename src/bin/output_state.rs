use std::fs::File;
use std::io::Read;

use motif_finder::base::*;

use gzp::{deflate::Mgzip, par::compress::{ParCompress, ParCompressBuilder}, syncz::{SyncZ, SyncZBuilder}, par::decompress::{ParDecompress, ParDecompressBuilder},ZWriter, Compression};


fn main() {

    let args: Vec<String> = std::env::args().collect();

    let file_to_read: String = args[1].to_string();

    
    let mut buffer: Vec<u8> = Vec::new();
    let mut decomp: ParDecompress<Mgzip> = ParDecompressBuilder::new().from_reader( File::open(file_to_read.as_str()).expect("You initialization file must be valid for inference to work!"));

    decomp.read_to_end(&mut buffer).expect("something went wrong reading the file");
    
    let interim: SingleSetOrTrace = {
        
        let try_trace: Result<(SetTraceDef, usize), _> = bincode::serde::decode_from_slice(&buffer, bincode::config::standard());
        if let Ok((trace, _bytes)) = try_trace {
            SingleSetOrTrace::Trace(trace)
        } else{
            let (set, _bytes): (StrippedMotifSet, usize) = bincode::serde::decode_from_slice(&buffer, bincode::config::standard()).expect("This did not give a bincode of a trace NOR a single set");
            SingleSetOrTrace::Singleton(set)
        }
    };


    println!("{:?}", interim);
}

#[derive(Debug)]
enum SingleSetOrTrace {
    Singleton(StrippedMotifSet),
    Trace(SetTraceDef)
}

