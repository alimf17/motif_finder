use std::collections::{HashSet, HashMap};
use std::error::Error;

use thiserror::Error;

use regex::Regex;

/// This is the struct that tells you where and what the locus is
pub struct Locus {
    name: String, 
    start: u64, 
    end: u64, 
    positively_oriented: bool, 
    chromosome: String,
    sequence_ontology: String, 
    go_terms: HashSet<u64>,
}

/// This is the struct that collects all the loci
pub struct GenomeAnnotations {
    loci: Vec<Locus>,
    sequence_ontologies: HashSet<String>,
    go_meanings: HashMap<u64, String>,
}

impl Locus {

    pub fn from_gff_line(line: &str) -> Result<Self, Box<dyn Error+Send+Sync>> {

        let regions: Vec<&str> = line.split('\t').collect();

        if regions.len() < 9 { return Err(Box::new(GffFormatError::NotEnoughRegions));}

        let annotations: Vec<&str> = regions[8].split(';').collect();

        let start: u64 = regions[3].parse::<u64>()?-1;
        let end: u64 = regions[4].parse()?;
        let positively_oriented = regions[6] != "-";
        let chromosome: String = regions[0].to_owned();
        let sequence_ontology: String = regions[2].to_owned();

        let gene_regex = Regex::new("gene=").unwrap();
        let id_regex = Regex::new("ID=").unwrap();
        let ont_regex = Regex::new("Ontology_term=").unwrap();
        
        let name = annotations.iter().find(|x| gene_regex.is_match(*x)).unwrap_or_else(|| annotations.iter().find(|x| id_regex.is_match(*x)).unwrap_or(&"=")).split('=').collect::<Vec<_>>()[1].to_owned();

        let mut go_terms: HashSet<u64> = HashSet::new();

        if let Some(term_matches) = annotations.iter().find(|x| ont_regex.is_match(*x)) {
            go_terms = term_matches.split(",").map(|x| x.replace("GO:", "").parse::<u64>()).filter_map(|x| x.ok()).collect();
        }

        Ok(Self {
            name,
            start,
            end,
            positively_oriented,
            chromosome,
            sequence_ontology,
            go_terms,
        })

    }
}

impl GenomeAnnotations {

    pub fn from_gff_file(gff_file_name: &str) -> Result<Self, Box<dyn Error+Send+Sync>> {

        let file = std::fs::read_to_string(gff_file_name)?;

        let go_parse_regex = Regex::new(r"go_\w\+=[a-zA-Z -]\+|\d\+||").unwrap();

        let go_meanings: HashMap<u64, String> = go_parse_regex.find_iter(&file).map(|a| a.as_str().split('|').collect::<Vec<_>>()).filter_map(|b| b[1].parse::<u64>().map(|c| (c, b[0].to_owned())).ok()).collect();

        let loci: Vec<Locus> = file.split('\n').map(|a| Locus::from_gff_line(a)).filter_map(|a| a.ok()).collect();

        let sequence_ontologies: HashSet<String> = loci.iter().map(|a| a.sequence_ontology.clone()).collect();

        Ok(Self {
            loci, 
            sequence_ontologies, 
            go_meanings,
        })

    }

}

#[derive(Error, Debug, Copy, Clone)]
pub enum GffFormatError {
    NotEnoughRegions,
}

impl std::fmt::Display for GffFormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { write!(f, "Not enough Regions for GFF formatted line") }
}
