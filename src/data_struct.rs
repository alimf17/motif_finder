use rand::Rng;
use rand::seq::SliceRandom;
use rand::distributions::{Distribution, Uniform, WeightedIndex};
use statrs::distribution::{Continuous, ContinuousCDF, LogNormal, Normal, Dirichlet, Exp};
use statrs::statistics::{Min, Max, Distribution as OtherDistribution};
use statrs::Result as otherResult;
use crate::waveform::{Kernel, Waveform, Waveform_Def, Noise, Background};
use crate::sequence::{Sequence, BP_PER_U8, U64_BITMASK, BITS_PER_BP};
use crate::base::{BPS, BASE_L, MIN_BASE, MAX_BASE};
use statrs::function::gamma;
use statrs::{consts, Result, StatsError};
use std::f64;
use std::fmt;
use std::collections::{VecDeque, HashMap};
use std::time::{Duration, Instant};
use once_cell::sync::Lazy;
use assume::assume;
use rayon::prelude::*;

use serde::{ser::*, Serialize,Serializer, Deserialize};
use serde::de::{
    self, DeserializeSeed, EnumAccess, IntoDeserializer, MapAccess, SeqAccess,
    VariantAccess, Visitor,
};

use std::env;
use std::fs;

use regex::Regex;

#[derive(Serialize, Deserialize)]
pub struct All_Data {

    seq: Sequence, 
    data: Waveform_Def, 
    background: Background

}

const ILLEGAL_FASTA_FILE_PANIC: &str = "FASTA file must be alternating pairs of lines: first in pair should start with '>', second should be the bases";

static GET_BASE_USIZE: Lazy<HashMap<char, usize>> = Lazy::new(|| {
    let mut map: HashMap<char, usize> = HashMap::new();
    for i in 0..BASE_L {
        _ = map.insert(BPS[i], i);
    }
    map
});

impl All_Data {
   
    
    //Important getters for the other structs to access the data from. 

    pub fn validated_data(&self) -> Waveform {

        let len = self.data.len();
        let spacer = self.data.spacer();

        let (point_lens, start_dats) = Waveform::make_dimension_arrays(&self.seq, spacer);

        //This is our safety check for our data
        if len != (point_lens.last().unwrap()+start_dats.last().unwrap()) {
            panic!("This sequence and waveform combination is invalid!");
        }

        unsafe{
            self.data.get_waveform(point_lens, start_dats, &self.seq)
        }

    }
    
    pub fn background(&self) -> &Background {
        &self.background
    }
 
    //TODO: Read from FASTA file and data table, then synchronize
    //SAFETY: I'm spending a lot of effort to validate your FASTA file
    //If your FASTA file is invalid and it somehow gets through, 
    //there will be parts of the code that are unsafe.

    //At the end of this function, we will have a Vector of usizes in terms of bases
    fn process_fasta(fasta_file_name: &str, null_char: Option<char>, seq_lines_are_contiguous: bool) -> (Vec<Vec<usize>>, String) {

        let file_string = fs::read_to_string(fasta_file_name).expect("Invalid file name!");
        let mut fasta_as_vec = file_string.split("\n").collect::<Vec<_>>();
        
        //We want to trim the whitespace from the bottom of the Fasta file
        //From the top can be dealt with by the user
        while fasta_as_vec.last()
              .expect("FASTA file should not be empty or all whitespace!")
              .chars().all(|c| c.is_whitespace()) {_ = fasta_as_vec.pop();}


        let mut base_vec: Vec<Vec<usize>> = Vec::new();
       
        let mut outer_index = 0usize;

        let mut fasta_iter = fasta_as_vec.iter().enumerate();

        let first_line = fasta_iter.next().expect("FASTA file should not be empty!");

        if !first_line.1.starts_with('>') {
            panic!("{}", ILLEGAL_FASTA_FILE_PANIC);
        }

        let mut last_null = true;   //We need to make sure that skip all BLOCKS of null characters, not at each null 
        
        base_vec[0] = Vec::new();
        for (line_pos, line) in fasta_iter {

            if line.starts_with('>'){
                if seq_lines_are_contiguous { 
                    continue;
                }
                if !last_null {
                    outer_index+=1;
                    base_vec[outer_index] = Vec::new();
                    last_null = true;
                }
                continue;
            }

            for (char_pos, chara) in line.chars().enumerate(){

                if (Some(chara) == null_char) {
                    if !last_null {
                        outer_index += 1; 
                        base_vec[outer_index] = Vec::new();
                    }
                    last_null = true;
                    continue;
                }
                last_null = false;

                base_vec[outer_index].push(*((*GET_BASE_USIZE).get(&chara)
                                            .expect(&(format!("Invalid base on line {}, position {}", line_pos+1, char_pos+1))))); 

            }

        }


        //This regular expression cleans the fasta_file_name to remove the last extension
        //Examples: ref.fasta -> ref, ref.txt.fasta -> ref.txt, ref_Fd -> ref_Fd
        let re = Regex::new(r"\.\pL+$").unwrap();
        let seq_name = re.replace(fasta_file_name, "");
        (base_vec, String::from(seq_name))
    }




    







}
