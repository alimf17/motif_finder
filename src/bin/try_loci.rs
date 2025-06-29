use std::fs::{File, create_dir_all};
use std::io::{Read, Write};

use motif_finder::gene_loci::*;

use clap::{Parser, ValueEnum};

#[derive(Parser)]
struct Cli {
    #[arg(short, long)]
    gff_file: String,

    #[arg(short, long)]
    out_file: String,
}

fn main() {

    let Cli { gff_file, out_file } = Cli::parse();

    let annotated = GenomeAnnotations::from_gff_file(&gff_file).unwrap();

    let mut outfile_handle = match File::create(out_file.clone()) {
        Err(_) => {
            create_dir_all(out_file.clone()).unwrap();
            File::create(out_file.clone()).unwrap()
        },
        Ok(o) => o
    };



    outfile_handle.write_all(format!("{}", annotated).as_bytes());

}
