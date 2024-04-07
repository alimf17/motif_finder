use std::fs;
use std::io::Read;

use motif_finder::base::*;

use motif_finder::data_struct::{AllData, AllDataUse};

fn main() {

    let args: Vec<String> = std::env::args().collect();

    let file_to_read: String = args[1].to_string();

    
    let mut buffer: Vec<u8> = Vec::new();
    fs::File::open(file_to_read).expect("Must be valid file").read_to_end(&mut buffer);
    
    let mut interim: SingleSetOrTrace = {
        
        let tryTrace: Result<SetTraceDef, _> = bincode::deserialize(&buffer);
        if let Ok(trace) = tryTrace {
            SingleSetOrTrace::Trace(trace)
        } else{
            let set: StrippedMotifSet = bincode::deserialize(&buffer).expect("This did not give a bincode of a trace NOR a single set");
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

