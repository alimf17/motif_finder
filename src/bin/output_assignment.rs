use std::fs;
use std::io::{Read, Write};
use std::error::Error;

use serde::*;
use serde::de::DeserializeOwned;
use motif_finder::base::*;
use motif_finder::gene_loci::*;
use motif_finder::data_struct::*;

use gzp::{deflate::Mgzip, par::compress::{ParCompress, ParCompressBuilder}, syncz::{SyncZ, SyncZBuilder}, par::decompress::{ParDecompress, ParDecompressBuilder},ZWriter, Compression};

use rand::*;

fn main() {

    let args: Vec<String> = std::env::args().collect();

    let file_to_read: String = args[1].to_string();

//    let output_dir: String = args[2].to_string();
    let null_retained_sequence_file: String = args[2].to_string();  
    
    let tf_analyzer_file: String = args[3].to_string();

    let annotations_file: String = args[4].to_string();

    let tsv_output_file: String = args[5].to_string();

    let additional_annotations_file = args.get(6).map(|a| a.to_string());

    let mut buffer: Vec<u8> = Vec::new();
    
    let mut set : StrippedMotifSet = attempt_bincode_serde_read_with_decomp(&file_to_read, Some(&mut buffer), None).expect("This did not give a bincode of a trace NOR a single set");
    
    let data : AllData = attempt_bincode_serde_read_with_decomp(&null_retained_sequence_file, Some(&mut buffer), None).expect("This did not give a bincode of an AllData");


    let mut data_ref = AllDataUse::new(&data, 0.0).expect("AllData file not valid!");
    
    data_ref.set_min_height(data_ref.min_height().max(1.0));

    let tf_analyzer = TFAnalyzer::from_regulon_tsv(&tf_analyzer_file, data_ref.number_bp(), 0, 1,2, None).expect("TF analyzer is invalid!");

    let mut annotations = GenomeAnnotations::from_gff_file(&annotations_file).expect("annotations file is invalid!");

    if let Some(added_annotations) = additional_annotations_file {
        let ontology = if annotations.ontologies().contains("CDS") { Some("CDS") } else if annotations.ontologies().contains("gene") { Some("gene")} else {None};
        if let Err(e) = annotations.add_go_terms_from_tsv_proteome(ontology, &added_annotations) { println!("Adding terms went wrong {:?}. Left the genome annotations unmodified.", e);};
    };


    let activated = set.reactivate_set(&data_ref);

    let mut std = std::io::stdout();
    
    let (poses_and_scores,_, _, a) = activated.output_tf_assignment(&mut std, &data_ref, &annotations, 200, &tf_analyzer, 0.05);
    println!("{:?}", poses_and_scores);
    let _ = a.unwrap();
    let activated_ref = &activated;
    //motif id, peak height, start position, reverse complement or no, score
    let mut fimo_like_prep: Vec<(usize, f64, usize, bool, f64)> = poses_and_scores.iter().enumerate().map(|(id, vec_of_scores)| vec_of_scores.iter().map(move |(pos, rev, score)| (id, activated_ref.nth_motif(id).peak_height(), *pos, *rev, *score))).flatten().collect();
    //I want to sort fimo_like_prep by the binding score (greatest to least) and break ties by putting the smallest id motif first and then by position of the motif
    fimo_like_prep.sort_unstable_by(|a, b| {
        let by_score = a.4-b.4;
        if by_score > 0.001 { return std::cmp::Ordering::Less;} //Remember, we want descending order of scores, and rust's default sort is ascending
        if by_score < -0.001 { return std::cmp::Ordering::Greater;}
        let by_id = a.0.cmp(&b.0);
        if by_id != std::cmp::Ordering::Equal { return by_id;}
        a.2.cmp(&b.2) //If this is ALSO equal, then screw it, I don't care about order beyond this: a and b are the "same" hit
    });

    let mut fimo_like_file = fs::File::create(&tsv_output_file).unwrap();

    fimo_like_file.write(b"motif_id\tmotif_alt_id\tsequence_name\tstart\tstop\tstrand\tscore\tp-value\tq-value\tmatched_sequence\n");
    for (id, height, position, rev, score) in fimo_like_prep {

        fimo_like_file.write(&format!("motif_{id}\t{}\t{}\t{position}\t{}\t{}\t{score}\t0.001\t0.1\t{}\n",activated.nth_motif(id).peak_height(), "sequence", position+activated.nth_motif(id).len(), if rev {"-"} else {"+"}, match data_ref.return_kmer_by_loc(position, activated.nth_motif(id).len()) { Some(bases) => bases.iter().map(|a| format!("{a}")).collect::<Vec<_>>().concat(), None => "NA".to_string()}).into_bytes()) ;


    }



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

