use std::collections::{HashSet, HashMap, VecDeque};
use std::error::Error;
use std::io::{BufReader, BufRead};

use itertools::Itertools;

use thiserror::Error;

use once_cell::sync::Lazy;

use regex::Regex;


static GENE_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new("gene=").unwrap());
static ID_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new("ID=").unwrap());
static ONT_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new("Ontology_term=").unwrap());
static GO_PARSE_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"go_\w+=([a-zA-Z -]+\|\d+\|\|,?)+;").unwrap());
static GO_TERM_ANALYZE_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(r"([\w ]+)\[GO:(\d{7})\]").unwrap());

/// This is the struct that tells you where and what the locus is
#[derive(Debug,Clone)]
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
#[derive(Debug, Clone)]
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

        //gff files are 1-indexed (booooo), while our genomes are 0-indexed (better indexing scheme)
        let start: u64 = regions[3].parse::<u64>()?-1;
        let end: u64 = regions[4].parse()?;
        let positively_oriented = regions[6] != "-";
        let chromosome: String = regions[0].to_owned();
        let sequence_ontology: String = regions[2].to_owned();

        let gene_regex = Regex::new("gene=").unwrap();
        let id_regex = Regex::new("ID=").unwrap();
        let ont_regex = Regex::new("Ontology_term=").unwrap();
        
        let name = annotations.iter().find(|x| GENE_REGEX.is_match(*x)).unwrap_or_else(|| annotations.iter().find(|x| ID_REGEX.is_match(*x)).unwrap_or(&"=")).split('=').collect::<Vec<_>>()[1].to_owned();

        let mut go_terms: HashSet<u64> = HashSet::new();

        if let Some(term_matches) = annotations.iter().find(|x| ONT_REGEX.is_match(*x)) {
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

    pub fn print_with_context(&self, annotated: &GenomeAnnotations) -> String {
        let (start, end, orient) = if self.positively_oriented { (self.start, self.end, "(+)") } else { (self.end, self.start, "(-)") };
        let mut out = format!("\t{}, a {} on {} from {start} to {end} {orient}\n", self.name, self.sequence_ontology, self.chromosome);

        if self.go_terms.len() > 0 { 
            out.push_str("\t\tFunctions:\n");
            for &term in self.go_terms.iter() {
                out.push_str(&format!("\t\t\tGO:{:07}: {}\n", term, annotated.go_meanings.get(&term).unwrap_or(&String::from("No meanings found"),)))
            }
        }

        out

    }

    pub fn can_regulate_locus(&self, location: u64,regulatory_distance: u64,  chr_name: Option<&str>) -> bool {

        if let Some(name) = chr_name {
            if &self.chromosome != name {return false;}
        };

        //I don't bother defining a behavior on overflow for self.end+regulatory_distance
        //There aren't even _genomes_ that overflow a u64. 
        let (cut_back, cut_forward) = if self.positively_oriented {(self.start.saturating_sub(regulatory_distance), self.end)} else { (self.start, self.end+regulatory_distance) };

        (location >= cut_back) && (location < cut_forward)
    }

    pub fn which_regulate_locus(&self, locations: &[u64], regulatory_distance: u64, chr_name: Option<&str>) -> Vec<bool> {

        if let Some(name) = chr_name {
            if &self.chromosome != name {return vec![false; locations.len()];}
        };

        //I don't bother defining a behavior on overflow for self.end+regulatory_distance
        //There aren't even _genomes_ that overflow a u64.
        let (cut_back, cut_forward) = if self.positively_oriented {(self.start.saturating_sub(regulatory_distance), self.end)} else { (self.start, self.end+regulatory_distance) };

        locations.iter().map(|&location| (location >= cut_back) && (location < cut_forward)).collect()
    }

    pub fn any_regulate_locus(&self, locations: &[u64],regulatory_distance: u64, chr_name: Option<&str>) -> bool {

        if let Some(name) = chr_name {
            if &self.chromosome != name {return false;}
        };

        //I don't bother defining a behavior on overflow for self.end+regulatory_distance
        //There aren't even _genomes_ that overflow a u64.
        let (cut_back, cut_forward) = if self.positively_oriented {(self.start.saturating_sub(regulatory_distance), self.end)} else { (self.start, self.end+regulatory_distance) };

        locations.iter().map(|&location| (location >= cut_back) && (location < cut_forward)).any(|x| x)
    }


    ///This returns the limits of where the gene locus should be drawn given a plotting window from `start` to `end`
    ///Returns `None` if `chr_name` is `Some(string) != Some(self.chromosome)`, 
    ///if the plotting range doesn't intersect the gene locus, or if `start >= end`
    pub fn yield_locus_boundaries_in_range(&self, start: u64, end: u64, chr_name: &Option<&str>) -> Option<(u64, u64)> {

        if let Some(name) = *chr_name {
            if &self.chromosome != name {return None;}
        };

        if (start >= end) || (self.start >= end) || (self.end <= start) { return None;}

        Some((self.start.max(start), self.end.min(end)))

    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn name_clone(&self) -> String {
        self.name.clone()
    }

    pub fn locus_bounds(&self) -> (u64, u64) {
        (self.start, self.end)
    }
    
    pub fn positively_oriented(&self) -> bool {
        self.positively_oriented
    }

    pub fn chromosome_name(&self) -> &str { 
        &self.chromosome
    }
    
    pub fn locus_type(&self) -> &str {
        &self.sequence_ontology
    }

    pub fn go_terms(&self) -> &HashSet<u64> {
        &self.go_terms
    }

}


impl GenomeAnnotations {

    pub fn from_gff_file(gff_file_name: &str) -> Result<Self, Box<dyn Error+Send+Sync>> {

        let file = std::fs::read_to_string(gff_file_name)?;


        let go_meanings: HashMap<u64, String> = GO_PARSE_REGEX.find_iter(&file).map(|a| a.as_str().split(&['=','|',',']).collect::<VecDeque<_>>()).map(|mut b| {
            let go_type = b.pop_front().expect("anything the go parse regex matches is definitionally not empty").to_owned();
            (0..(b.len()/4)).filter_map(move |i| b[4*i+1].parse::<u64>().map(|c| {
                let mut function = go_type.clone();
                function.push('=');
                function.push_str(b[4*i]);
                (c, function)
            }).ok())
           // b[1].parse::<u64>().map(|c| (c, b[0].to_owned())).ok()
        }).flatten().collect();

        let loci: Vec<Locus> = file.split('\n').map(|a| Locus::from_gff_line(a)).filter_map(|a| a.ok()).collect();

        let sequence_ontologies: HashSet<String> = loci.iter().map(|a| a.sequence_ontology.clone()).collect();

        Ok(Self {
            loci, 
            sequence_ontologies, 
            go_meanings,
        })

    }

    pub fn loci_with_ontology<'b>(&self, ontology: &'b str) -> Vec<&Locus> {

        self.loci.iter().filter_map(|a| if a.sequence_ontology == ontology { Some(a) } else {None}).collect()
    }

    //Implementation note: I don't use the implementation from loci_with_ontology so that I don't have to iterate over
    //all the possible sequence ontologies a bajillion times
    pub fn bin_by_ontology(&self) -> HashMap<String, Vec<&Locus>> {
        let mut ontology_sections: HashMap<String, Vec<&Locus>> = HashMap::with_capacity(self.sequence_ontologies.len());
        let _: Vec<_> = self.loci.iter().map(|locus| {
            match ontology_sections.get_mut(&locus.sequence_ontology) {
                None => {_ = ontology_sections.insert(locus.sequence_ontology.clone(), vec![locus]);},
                Some(loci) => loci.push(locus),
            }
        }).collect();
        ontology_sections
    }

    //    pub fn yield_locus_boundaries_in_range(&self: Locus, start: u64, end: u64, chr_name: Option<&str>) -> Option<(u64, u64)> {

    //The lifetime of the returned &str is connected to the lifetime of the locus names. This is only guarenteed to be correct if binned_loci was made by self.bin_by_ontology() if it's Some
    pub fn collect_ranges<'a>(&'a self, start: u64, end: u64, chr_name: &Option<&str>, ontologies: Option<&[&str]>, binned_loci: Option<&HashMap<String, Vec<&'a Locus>>>) -> Vec<Vec<(&'a str, u64, u64, bool)>> {

        match ontologies {

            None => vec![self.loci.iter().map(|a| a.yield_locus_boundaries_in_range(start, end, chr_name).map(|b| (a.name(), b.0, b.1, a.positively_oriented()))).filter_map(|a| a).collect()],
            Some(onts) => {

                match binned_loci {
                    None => {
                        let bins = self.bin_by_ontology();
                        onts.iter().filter_map(|&c| bins.get(c)).map(|a| a.iter().map(|l| l.yield_locus_boundaries_in_range(start, end, chr_name).map(|b| (l.name(), b.0, b.1, l.positively_oriented()))).filter_map(|l| l).collect()).collect()
                    }, 

                    Some(bins) => {
                        onts.iter().filter_map(|&c| bins.get(c)).map(|a| a.iter().map(|l| l.yield_locus_boundaries_in_range(start, end, chr_name).map(|b| (l.name(), b.0, b.1, l.positively_oriented()))).filter_map(|l| l).collect()).collect()
                    },

                }

            },

        }

    }
    
    //STRONGLY recommend using this with the ontology, probably the CDS ontology. There can be redundancies, which we will ignore.
    //Also, note that if a gene has multiple names in different `Locus`es in `self`, even if that gene is represented on the same line of the tsv, 
    //we will find and modify all of the different `Locus`es. This is to account for the possibility that the different loci diverged from the same evolutionary source, 
    //but had two different names. 
    pub fn add_go_terms_from_tsv_proteome<'b>(&mut self, ontology: Option<&'b str>, tsv_file_name: &'b str) -> Result<(), Box<dyn Error+Send+Sync>> {
        
        let f = std::fs::File::open(tsv_file_name)?;
        let f = BufReader::new(f);

        let gene_name_locs: HashMap<String, usize> = self.loci.iter().filter_map(|a| match ontology {
            Some(ont) => if a.locus_type() == ont { Some(a) } else {None},
            None => Some(a)
        }).enumerate().map(|(i,a)| (a.name_clone(), i)).collect();

        let lines = f.lines().skip(1); //these tsvs have column symbols we're ignoring

        for line in lines {
            //_entry, _reviewed, _entry_name,  _protein_names,  gene_names, _organism,  _length, _gene_names_ordered_locus, _gene_names_orf, _gene_names_primary, _gene_names_synonyms, go_ids,go
            let Some(split_line): Option<Vec<_>> = line.ok().map(|a| a.split('\t').map(|b| b.to_owned()).collect()) else {continue;}; 

            let Some(gene_names) = split_line.get(4) else {continue;};
            let Some(go_ids) = split_line.get(11) else {continue;};
            let Some(go) = split_line.get(12) else {continue;};

            let go_terms: Vec<u64> = go_ids.split("; ").map(|x| x.replace("GO:", "").parse::<u64>()).filter_map(|x| x.ok()).collect();
            let go_meanings: Vec<(u64, String)> = go.split("; ").filter_map(|x| GO_TERM_ANALYZE_REGEX.captures(x).map(|cap| cap.extract()))
                                                                .filter_map(|(_, [description, go_term])| {
                                                                    let go_term: Option<u64> = go_term.parse().ok();
                                                                    go_term.map(|a| (a, description.to_owned()))
                                                                }).collect();
                //let Some((_, [description, go_term])) = GO_TERM_ANALYZE_REGEX.captures(x).map(|cap| cap.extract()) else {continue;};
                //let Some(go_term): Option<u64> = go_term.parse() else {continue;};

                //(go_term, description)
            //}).collect();

            for &locus_ind in gene_names.split(' ').filter_map(|a| gene_name_locs.get(a)) {

                let locus_changing: &mut Locus = self.loci.get_mut(locus_ind).expect("We extracted this from the indices of loci");

                //If the go term already exists, we don't care
                for go_term in go_terms.iter() { _ = locus_changing.go_terms.insert(*go_term);}
            
                for (term, meaning) in go_meanings.iter() {
                    if self.go_meanings.get(term).is_none() { self.go_meanings.insert(*term, meaning.clone()) ;}
                }

            }

        } 

        Ok(())

    }

    pub fn loci(&self) -> &Vec<Locus> {
        &self.loci
    }

    pub fn ontologies(&self) -> &HashSet<String> {
        &self.sequence_ontologies
    }

    pub fn go_explanations(&self) -> &HashMap<u64, String> {
        &self.go_meanings
    }
}

/* pub struct GenomeAnnotations {
    loci: Vec<Locus>,
    sequence_ontologies: HashSet<String>,
    go_meanings: HashMap<u64, String>,
}
*/

impl std::fmt::Display for GenomeAnnotations {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {

        //guessing on capacity
        let mut output: String = String::with_capacity(self.loci.len()*100);
        let ontology_sections = self.bin_by_ontology();

        let _: Vec<_> = ontology_sections.iter().map(|(name, loci)| {
            output.push_str(&format!("{name}\n"));
            let _: Vec<_> = loci.iter().map(|locus| output.push_str(&locus.print_with_context(&self))).collect();
            output.push_str("\n");
        }).collect();

        write!(f, "{}", output)
    }
}

#[derive(Error, Debug, Copy, Clone)]
pub enum GffFormatError {
    NotEnoughRegions,
}

impl std::fmt::Display for GffFormatError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result { write!(f, "Not enough Regions for GFF formatted line") }
}
