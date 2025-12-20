use std::{f64, fs, fmt};
use std::io::{Read, Write};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
//use std::time::{Duration, Instant};

use core::f64::consts::PI;
use std::error::Error;

use crate::waveform::{Waveform, WaveformDef, Background, WIDE, Kernel, KernelWidth, KernelVariety, WaveCreationError};
use crate::sequence::{Sequence, NullSequence, BP_PER_U8};
use crate::base::{BPS, BASE_L, MIN_BASE, MAX_BASE, LOG_HEIGHT_MEAN, LOG_HEIGHT_SD, MAX_HEIGHT,MotifSet, Motif, TruncatedLogNormal, Bp};


use thiserror::Error;

use rayon::prelude::*;
use rand::Rng;
use rand::prelude::SliceRandom;

use statrs::distribution::{Continuous, StudentsT, Normal, ContinuousCDF, LogNormal, Gamma, Gumbel, Triangular};

use statrs::statistics::*;

use once_cell::sync::Lazy;

use plotters::prelude::*;
            

use plotters::prelude::full_palette::ORANGE;


//use rand::distributions::Distribution;

use argmin::argmin_error;
use argmin::core::*;
use argmin::core::Error as ArgminError;
use argmin::solver::neldermead::NelderMead;

use bincode::config;
use serde::{Serialize, Deserialize, Serializer, ser::SerializeStruct, Deserializer, de::Visitor, de::SeqAccess, de::MapAccess};

use log::warn;

use regex::Regex;

const DATA_SUFFIX: &str = "data.bin";
const MAD_ADJUSTER: f64 = 1.4826;

const MINMER_NUM: usize = BASE_L.pow(MIN_BASE as u32);

static GET_BASE_USIZE: Lazy<HashMap<char, usize>> = Lazy::new(|| {
    let mut map: HashMap<char, usize> = HashMap::new();
    for i in 0..BASE_L {
        _ = map.insert(BPS[i], i);
    }
    map
});

#[derive(Debug, Clone)]
pub struct AllData {

    seq: Sequence,
    null_seq: NullSequence,
    data: WaveformDef,
    start_genome_coordinates: Vec<usize>,
    start_nullbp_coordinates: Vec<usize>,
    genome_block_chrs: Vec<usize>,
    nullbp_block_chrs: Vec<usize>,
    chr_names: Vec<String>,
    background: Background,
    min_height: f64,
    credibility: f64, 
}

#[derive(Clone)]
pub struct AllDataUse<'a> {
    data: Waveform<'a>, 
    null_seq: &'a NullSequence, 
    start_genome_coordinates: &'a Vec<usize>,
    start_nullbp_coordinates: &'a Vec<usize>,
    genome_block_chrs:  &'a Vec<usize>,
    nullbp_block_chrs: &'a Vec<usize>,
    chr_names:  &'a Vec<String>,
    background: Background,
    offset: f64,
    min_height: f64,
    credibility: f64, 
    height_dist: TruncatedLogNormal,
}

struct FitTDist<'a> {
    
    decorrelated_data: &'a Vec<f64>,

}

impl<'a> FitTDist<'a> {


    fn lnlike(&self, sd: f64, df: f64) -> f64 {
     
        let dist = StudentsT::new(0., sd, df).unwrap();
        let mut like = 0.0;
        for &i in self.decorrelated_data {
            like += dist.ln_pdf(i);
        }
        like
    }

}

impl<'a> CostFunction for FitTDist<'a> {
    /// Type of the parameter vector
    type Param = Vec<f64>;
    /// Type of the return value computed by the cost function
    type Output = f64;

    /// Apply the cost function to a parameter `p`
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, ArgminError> {
        
        if p.len() < 2 {
            return Err(argmin_error!(InvalidParameter, "t distribution must have two parameters!"));
        }

        if (p[0] < 0.0) || (p[1] < 2.0) || (p[1] > 50.0) { //We need to include a limit on the degrees of freedom, or else we just blow up too big sometimes
            return Ok(f64::INFINITY);
        }

        Ok(-self.lnlike(p[0], p[1]))
    }
}

struct FitTruncNormDist<'a> {
    
    decorrelated_data: &'a Vec<f64>,
    q1: f64,
    q3: f64,
}

impl<'a> FitTruncNormDist<'a> {


    fn lnlike(&self, sd: f64) -> f64 {
     
        let dist = Normal::new(0., sd).unwrap();
        let mut like = 0.0;
        for &i in self.decorrelated_data {
            if i < self.q1 || i > self.q3 {
                return -f64::INFINITY;
            }
            like += dist.ln_pdf(i)-((-(dist.cdf(self.q1)+dist.sf(self.q3))).ln_1p());
        }
        like
    }

}

impl<'a> CostFunction for FitTruncNormDist<'a> {
    /// Type of the parameter vector
    type Param = Vec<f64>;
    /// Type of the return value computed by the cost function
    type Output = f64;

    /// Apply the cost function to a parameter `p`
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, ArgminError> {

        if p.len() < 1 {
            return Err(argmin_error!(InvalidParameter, "normal distribution must have a parameter!"));
        }

        if p[0] <= 0.0 { 
            return Ok(f64::INFINITY);
        }

        Ok(-self.lnlike(p[0]))
    }
}

struct FitMedianHalfNormalDist<'a> {

    half_data: &'a Vec<f64>,
    median: f64,
    quantile: f64,
}

impl<'a> FitMedianHalfNormalDist<'a> {

    //This is the Cramer von Mises distance evaluated only on the first half of the half normal.
    //Technically, this is the distance multiplied by the length of the data, but that bijects.
    fn cvm_distance_from_half_normal(&self, sd: f64) -> f64 {

        //Remember, this is only half the CDF
        let empirical_cdf_coeff: Vec<f64> = (0..self.half_data.len()).map(|a| self.quantile*((2*a+1) as f64)/((2*self.half_data.len()) as f64)).collect();


        let theoretical_dist = Normal::new(0., sd).unwrap();

        let cost = if self.half_data.is_sorted() {
            if self.half_data.get(0).map(|&a| a < 0.0).unwrap_or(true) { f64::INFINITY}
            else{
            self.quantile/(12.0*((self.half_data.len()) as f64))+self.half_data.iter().zip(empirical_cdf_coeff.into_iter()).map(|(&a, b)| if a == 0.0 {0.0} else {( (2.0*(theoretical_dist.cdf(a)-0.5))-b).powi(2)}).sum::<f64>()}
        } else {

            let mut half_data = self.half_data.clone();
            half_data.sort_unstable_by(|a,b| a.partial_cmp(&b).unwrap());
            if half_data.get(0).map(|&a| a < 0.0).unwrap_or(true) { f64::INFINITY}
            else{self.quantile/(12.0*((half_data.len()) as f64))+half_data.iter().zip(empirical_cdf_coeff.into_iter()).map(|(&a, b)|if a == 0.0 {0.0} else {( (2.0*(theoretical_dist.cdf(a)-0.5))-b).powi(2)}).sum::<f64>()}

        };


        cost



    }

}

impl<'a> CostFunction for FitMedianHalfNormalDist<'a> {
    /// Type of the parameter vector
    type Param = Vec<f64>;
    /// Type of the return value computed by the cost function
    type Output = f64;

    /// Apply the cost function to a parameter `p`
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, ArgminError> {

        if p.len() < 1 {
            return Err(argmin_error!(InvalidParameter, "normal distribution must have a parameter!"));
        }

        if p[0] <= 0.0 {
            return Ok(f64::INFINITY);
        }

        Ok(self.cvm_distance_from_half_normal(p[0]))
    }
}

#[derive(Clone)]
struct FitNormalDist {
    three_quarters_pdf: Vec<[f64;2]>,
    
}

impl FitNormalDist {

    fn new(data: &Vec<f64>) -> Option<Self> {
        if data.len() < 4 { return None; }
        
        //This song and dance is brought to you by "I don't want to sort things if I don't have to"
        let mut to_copy = if data.is_sorted() { None} else {Some(data.clone())};
        if let Some(elem) = to_copy.iter_mut().next() { elem.sort_unstable_by(|a,b| a.partial_cmp(&b).unwrap()) ;}
        let data = match to_copy.as_ref() {
            None => data,
            Some(elem) => elem,
        };
        //End song and dance


        let median_cut_index = if data.len() & 1 == 1 { data.len()/2 +1 } else { data.len()/2};

        let q1 = if median_cut_index % 1 == 1 { data[median_cut_index/2] } else { 0.5*(data[median_cut_index/2] + data[median_cut_index/2 -1])};

        let length_after_median = data.len()-median_cut_index;

        let q3_ind = median_cut_index + length_after_median/2;

        let q3 = if length_after_median % 1 == 1 { data[median_cut_index + length_after_median/2]} else { 0.5*(data[median_cut_index + length_after_median/2]+data[median_cut_index + length_after_median/2-1])};

        let iqr = q3-q1;

        let bin_width = 2.0*iqr/( (data.len() as f64).cbrt());

        let num_bins = ((q3-data[0])/bin_width) as usize;

        let mut track_ind = 0_usize;

        let mut lower_threshold = data[0];

        let three_quarters_pdf: Vec<[f64;2]> = (0..num_bins).map(|i| {

            let upper_threshold = data[0]+((i+1) as f64)*bin_width;

            let mut count = 0_usize;

            while data[track_ind] >= lower_threshold && data[track_ind] < upper_threshold && track_ind < q3_ind {
                count+=1;
                track_ind += 1;
            }
        
            let midpoint = 0.5*(lower_threshold+upper_threshold);

            lower_threshold = upper_threshold;

            [midpoint, (count as f64)/(bin_width*(data.len() as f64))]
                
        }).collect();

        println!("three quart {:?}", three_quarters_pdf);
        Some(Self{ three_quarters_pdf })
    }
    fn distance_from_normal(&self, mean: f64, sd: f64) -> f64 {
        let norm = Normal::new(mean, sd).unwrap();
        self.three_quarters_pdf.iter().map(|&a| (a[1]-norm.pdf(a[0])).powi(2)).sum::<f64>()
    }
}

impl CostFunction for FitNormalDist {
    /// Type of the parameter vector
    type Param = Vec<f64>;
    /// Type of the return value computed by the cost function
    type Output = f64;

    /// Apply the cost function to a parameter `p`
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, ArgminError> {

        if p.len() < 2 {
            return Err(argmin_error!(InvalidParameter, "normal distribution must have a parameter!"));
        }

        if p[1] <= 0.0 {
            return Ok(f64::INFINITY);
        }

        Ok(self.distance_from_normal(p[0], p[1]))
    }

}

#[derive(Clone)]
struct FitGaussianMixture<'a> {

    data: &'a Vec<f64>,
    max_val: f64,
    //weight_middle: f64,
    //sd_middle: f64, 
    //mean_peak: f64, 
    //sd_peak: f64,
}

impl<'a> FitGaussianMixture<'a>{

    fn cvm_distance_from_half_normal(&self, weight_middle: f64,sd_middle: f64,mean_peak: f64,sd_peak: f64) -> f64 {

        //Remember, this is only half the CDF
        let empirical_cdf_coeff: Vec<f64> = (0..self.data.len()).map(|a| ((2*a+1) as f64)/((2*self.data.len()) as f64)).collect();


        let theoretical_middle_dist = Normal::new(0., sd_middle).unwrap();
        let theoretical_peak_dist = Triangular::new(0.,mean_peak, sd_peak).unwrap();

        let mut theoretical_cdf: Vec<(f64, f64)> = self.data.iter().map(|&d| (d,weight_middle*theoretical_middle_dist.cdf(d)+(1.0-weight_middle)*theoretical_peak_dist.cdf(d))).collect();

        theoretical_cdf.sort_unstable_by(|a,b| a.0.partial_cmp(&b.0).unwrap());

        1.0/(12.0*(self.data.len() as f64)) + theoretical_cdf.into_iter().zip(empirical_cdf_coeff.into_iter()).map(|(a,b)| (a.1-b).powi(2)).sum::<f64>()


    }
    
}

impl<'a> CostFunction for FitGaussianMixture<'a> {

    /// Type of the parameter vector
    type Param = Vec<f64>;
    /// Type of the return value computed by the cost function
    type Output = f64;

    /// Apply the cost function to a parameter `p`
    fn cost(&self, p: &Self::Param) -> Result<Self::Output, ArgminError> {

        if p.len() < 4 {
            return Err(argmin_error!(InvalidParameter, "mixture models must have four parameters!"));
        }

        // Normal >= half density, Normal cannot be more than all density, background sd is
        // positive, the log normal fitting the bad peaks must exceed the normal substantially, the
        // sd normal cannot have negative density, and the meanlog cannot effectively exceed the
        // maximum value of the data
        if p[0] < 0.5 || p[0] > 1.0 || p[1] <= 0.0 || p[3] <= 0.0 || p[2] <= 0.0 || p[3] >= p[2]{ 
            return Ok(f64::INFINITY);
        }

        Ok(self.cvm_distance_from_half_normal(p[0], p[1], p[2], p[3]))
    }


}


impl AllData {
  

    /// This actually initializes the inference data that you need. 
    /// There are several requirements, and failure to uphold these requirements 
    /// will cause errors. This is the most finicky function in the public API. 
    /// For your own sanity, pay attention here. 
    /// # Inputs
    /// - `fasta_file`: A string that points to a fasta file
    /// - `data_file`: A string which points to your protein-DNA binding file.
    ///                This file starts with a header `loc data`, followed by 
    ///                lines which start with a non negative integer followed by 
    ///                a float. No line should be empty. If the smallest location
    ///                is 1, we will assume it's 1-indexed. Otherwise, we will 
    ///                assume it's zero indexed. 
    /// - `output_dir`: A string indicating the directory into which this AllData
    ///    should be serialized. 
    /// - `is_circular`: Are your chromosomes circular? Note that this must be a 
    ///                  global yes or not: we don't handle cases such as 
    ///                  Agrobacterium tumefaciens where different chromosomes have 
    ///                  different answers. If this is your use case, just say "no"
    ///                  but keep in mind that there might be edge cases we miss from 
    ///                  the circular chromosome. You're honestly lucky that I handle
    ///                  multiple chromsomes at all, even if they're circles
    /// - `fragment_length`: Your estimated fragment length. It should be about 
    ///                      one half the width of your peaks. 
    /// - `spacing`: The spacing of your location probes. Pick an average value:
    ///              we will interpolate if you're off. This should be positive.
    /// - `min_height`: What is the minimum height at which you want full binding 
    ///                 of peaks? 
    /// - `null_char`: If your fasta_file has `'N'` or `'X'` to indicate an 
    ///                unknown base, this should be `Some('N')` or `Some('X')`.
    ///                Otherwise, this should be `None`.
    /// - `peak_scale`: This is a number you can increase to favor assigning
    ///                 data blocks as having no binding, or decrease to favor
    ///                 assigning data blocks as having binding. If `None`, we 
    ///                 set this to `1.0`.
    /// # Errors
    /// There are a great deal of errors that can result from this function. 
    /// We have to uphold a lot of invariants, and while we will do our best, 
    /// pay close atttention to all of the errors you get here. We try to 
    /// find and collate them into one bundle, so you can fix many errors at once.
    /// Here are the conditions that will cause an error:
    /// - If your fasta file does not exist. 
    /// - If your fasta file is blank
    /// - If your fasta file does not start every other line with `>`
    /// - If your fasta file has bases other than A, C, G, or T, or your null character, if provided
    /// - If your data file does not exist. 
    /// - If your data file is blank
    /// - If `spacer = 0`
    /// - If your data file is formatted incorrectly.
    /// - If your data has locations that go beyond your sequence length
    /// - If there is not enough data without binding to fit the background distribution
    /// - If there is not enough data with binding to actually run inference on. 
    /// - If your data or fasta files are big enough that the resulting structs
    ///   overrun `[usize::MAX]` for the platform you're on. If you're working
    ///   on a 32 bit platform, this is conceivable, albeit unlikely. If you're
    ///   workng on a 64 platform, this should be impossible unless you have an
    ///   an organism with octillions the size of the largest genomes I'm aware of.
    ///   This can also technically cause a panic if there are errors on the fasta
    ///   file that are super late into a fasta file that large. 
    pub fn create_inference_data(fasta_file: &str, data_file: &str, output_dir: &str, is_circular: bool, 
                                 fragment_length: usize, spacing: usize, min_height: f64, credibility: f64, null_char: &Option<char>, peak_scale: Option<f64>, retain_null: Option<bool>) -> Result<Self, AllProcessingError> {


        let pre_sequence = Self::process_fasta(fasta_file, *null_char);

        println!("processed fasta");
        let sequence_len_or_fasta_error = match (pre_sequence).as_ref() { 
            Ok(seq_ref) => Ok(seq_ref.iter().map(|(name, seq)| (name.clone(), seq.len())).collect::<HashMap<String, usize>>()),
            Err(e) => Err(e.clone()),
        };

        let (pre_data, background,min_height, pre_noise, thresh) = Self::process_data(data_file, sequence_len_or_fasta_error, fragment_length, spacing, min_height, is_circular, peak_scale, retain_null.unwrap_or(false))?;
 
        println!("processed data");

        let pre_seq = pre_sequence.expect("already returned if this was an error");

        let pre_seq_len = pre_seq.len();

        
        let test_sync = Self::synchronize_sequence_and_data(pre_seq, pre_data, pre_noise, pre_seq_len, spacing, thresh);
       
        if test_sync.is_err() { return Err(AllProcessingError::Synchronization(BadDataSequenceSynchronization::PoorSynchronize(test_sync.unwrap_err()))); }

        let (seq, null_seq, data, starts, start_nulls, genome_chr_blocks, null_chr_blocks, chr_names) = test_sync.expect("We already returned if this was an error.");

        println!("synchronized fasta and data");
        let mut full_data: AllData = AllData {seq: seq, null_seq: null_seq, data: data, start_genome_coordinates: starts, start_nullbp_coordinates: start_nulls, genome_block_chrs: genome_chr_blocks, nullbp_block_chrs: null_chr_blocks, chr_names: chr_names, background: background, min_height: min_height, credibility: credibility.abs()};

        //I've inserted this as a check to make sure that this isn't so massive
        //that TARJIM can't work with it or save it. Because if it is, you want
        //to know that now, not later. Though I honestly cannot imagine how this
        //would happen for any good faith run of TARJIM, at least on a 64 bit 
        //machine or greater. 
        let Ok(buffer) = bincode::serde::encode_to_vec(&full_data, config::standard()) else { return Err(AllProcessingError::WayTooBig);};
        
        let use_data = AllDataUse::new(&full_data, 0.0)?; //This won't be returned: it's just a validation check.

        println!("validated data");
       
        let mut max = -f64::INFINITY;


        Ok(full_data)


    }

    //Important getters for the other structs to access the data from. 
    fn validated_data(&self) -> Result<Waveform, AllProcessingError> {

        let len = self.data.len();
        let spacer = self.data.spacer();

        let (point_lens, start_dats) = Waveform::make_dimension_arrays(&self.seq, spacer);

        //This is our safety check for our data
        if len != (point_lens.last().unwrap()+start_dats.last().unwrap()) {
            return Err(AllProcessingError::Synchronization(BadDataSequenceSynchronization::SequenceMismatch));
        }

        //we already errorred out if such correct blocking if the 
        //waveform is impossible
        //TODO: verify that this is impossible to return an error or modify the result type on validated_data
        Ok(self.data.get_waveform(&self.seq).unwrap())

    }
   

    pub fn background(&self) -> &Background {
        &self.background
    }

    pub fn seq(&self) -> &Sequence {
        &self.seq
    }

    /// This gives the starting locations of each gene block in genome coordinates.
    /// Note that if you have a circular genome, this can loop over and produce
    /// genome coordinates that need to be modded by the sequence length
    pub fn zero_locs(&self) -> &Vec<usize> {
        &self.start_genome_coordinates
    }

    //SAFETY: genome_block_chrs has to have the same length as start_genome_coordinates, start_nullbp_coordinates has to ahve the same length as nullbp_block_chrs, all values of both must be less than chr_names.len(), and the sequence, nullsequence, waveform, and background must all be compatible
    pub(crate) unsafe fn create_manual(seq: Sequence, null_seq: NullSequence, data: WaveformDef, start_genome_coordinates: Vec<usize>, start_nullbp_coordinates: Vec<usize>, genome_block_chrs: Vec<usize>, nullbp_block_chrs: Vec<usize>, chr_names: Vec<String>, background: Background, min_height: f64, credibility: f64) -> Self {

        Self { 
            seq,
            null_seq,
            data, 
            start_genome_coordinates,
            start_nullbp_coordinates,
            genome_block_chrs, 
            nullbp_block_chrs, 
            chr_names,
            background,
            min_height,
            credibility,
        }


    }

    //SAFETY: I'm spending a lot of effort to validate your FASTA file
    //If your FASTA file is invalid and it somehow gets through, 
    //there will be parts of the code that are unsafe. This WILL panic if:
    //
    //  1) Your file doesn't exist. 
    //  2) Your file doesn't start with '>'
    //  3) You have ANY characters that aren't bases or a null character (usually, 'N', or 'X') on a line that doesn't start with '>'
    //
    //At the end of this function, we will have a Vector of usizes in terms of bases, 
    //with null bases being marked with an index of BASE_L.
    //WARNING: ALL sequences in your FASTA file will be considered a single continguous block. 
    //         If they are not, why not? The FASTA file input shouldn't be fragments: it should be
    //         a reference genome. If you have multiple chromsomes that you're trying to throw into the inference
    //         you can pretend that they're contiguous by throwing in your null characters a number of times 
    //         equal to your fragment length. Note: you will have to massage your position data if you do this.
    //         For example, if you have two linear chromosomes, where chr1 is 29384 bp long, and your fragment
    //         length is 350bp, then, for zero indexed data, locations from 0 to 29383 will be chr1
    //         29384 to 29733 will point to the fake spacer, 29384 would be position 0 on chr2, 
    //         29386 would be position 2 on chr2, etc. Note: this motif finder cannot support a system with
    //         multiple chromosomes where any of the chromosomes are circular. You CAN try to throw them in
    //         but the only possible fix here would have to not know that one of the chromosomes is circular.
    //         If it's really important that you need to do inference on that chromosome with circularity, 
    //         do multiple inferences.
    fn process_fasta(fasta_file_name: &str, null_char: Option<char>) -> Result<HashMap<String, Vec<Option<usize>>>, FastaError> {

        println!("If this fasta file has multiple chromosomes, we VERY strongly recommend chromosome names match the names from any gff file you're using to assign loci, later");
        let file_string = match fs::read_to_string(fasta_file_name) {

            Ok(file) => file, 
            Err(e) => return Err(FastaError::InvalidFile(Arc::new(e))),
        };//TODO: make this a FastaError variant
        
        let mut fasta_as_vec = file_string.split("\n").collect::<Vec<_>>();
        
        //We want to trim the whitespace from the bottom of the Fasta file
        //From the top can be dealt with by the user

        let mut fasta_is_empty_or_blank = true;
        while let Some(line) = fasta_as_vec.last() {
            if line.chars().all(|c| c.is_whitespace()) {_ = fasta_as_vec.pop();}
            else { fasta_is_empty_or_blank = false; break; }
        }
        if fasta_is_empty_or_blank { return Err(FastaError::BlankFile(EmptyFastaError));}


        let mut chromosome_base_vec: HashMap<String, Vec<Option<usize>>> = HashMap::new();

        let mut base_vec: Vec<Option<usize>> = Vec::new();
       
        let mut fasta_iter = fasta_as_vec.iter().enumerate();

        let first_line = fasta_iter.next().expect("We already returned an error if the fasta file is empty or is all whitespace");

        let mut current_chromosome: String = if first_line.1.starts_with('>') {
            let mut chr_name = first_line.1.trim_start_matches('>').trim().to_owned();
            if let Some(found_chromosome_name) = chr_name.find("chromosome:") {
                chr_name = (&chr_name[(found_chromosome_name+11)..].trim()).to_string();
            } 
            chr_name
        } else { return Err(FastaError::BadFastaStart(MaybeNotFastaError));};

        let mut has_a_valid_bp = false;

        let mut potential_invalid_bp_err: Option<InvalidBasesError> = None;

        for (line_pos, line) in fasta_iter {

            if line.starts_with('>'){

                //TODO: add way to also include any located bad bases before this point
                let None = chromosome_base_vec.insert(current_chromosome.clone(), base_vec) else { return Err(FastaError::RedundantChromosome(ChromosomeError::new(current_chromosome, potential_invalid_bp_err)));};
                base_vec = Vec::new();
                let mut chr_name = first_line.1.trim_start_matches('>').to_owned();
                if let Some(found_chromosome_name) = chr_name.find("chromosome:") {
                    chr_name = (&chr_name[(found_chromosome_name+11)..].trim()).to_string();
                }
                current_chromosome = chr_name;
                continue;
            }
            for (char_pos, chara) in line.chars().enumerate(){

                let (valid_base_or_known_null, to_push) = {
                    let mut valid_answer = true;
                    //We make no semantic distinction between A v a, T vs t, etc
                    let can_push = GET_BASE_USIZE.get(&chara.to_ascii_uppercase()).copied();
                    if can_push.is_none() {valid_answer = Some(chara) == null_char;
                    } else {has_a_valid_bp |= true;}

                    (valid_answer, can_push)
                };

                match (valid_base_or_known_null, potential_invalid_bp_err.is_none()) {
                    (true, true) => { base_vec.push(to_push);}, 
                    (true, false) => {},
                    (false, true) => { potential_invalid_bp_err = Some(InvalidBasesError::new(line_pos as u64, char_pos as u64));}, //If we panic at line_pos or char_pos's conversion, your file is WAY too big
                    (false, false) => {
                        match potential_invalid_bp_err.iter_mut().next() {
                            Some(a) => a.add_invalid_if_possible(line_pos as u64, char_pos as u64),
                            None => unreachable!(),
                        };
                    },
                };
            }
        }

        let None = chromosome_base_vec.insert(current_chromosome.clone(), base_vec) else { return Err(FastaError::RedundantChromosome(ChromosomeError::new(current_chromosome, potential_invalid_bp_err)));};
        if !has_a_valid_bp { return Err(FastaError::BlankFile(EmptyFastaError));}
        if let Some(invalid_bp) = potential_invalid_bp_err {
            return Err(FastaError::BadFastaInput(invalid_bp));
        }



        //This regular expression cleans the fasta_file_name to remove the last extension
        //Examples: ref.fasta -> ref, ref.txt.fasta -> ref.txt, ref_Fd -> ref_Fd
        Ok(chromosome_base_vec)
    }


    //Note: we assign the FIRST location in a bedgraph line as the location of the data
    fn try_read_bedgraph_line<E: Error>(line: (usize, &str), potential_error: &mut Option<DataFileBadFormat>, chromosome_check: & Result<HashMap<String, usize>, E>, chromsomes_acquired: &mut HashSet<String>) -> Option<(String, usize, f64)> {

        let check_chromosome = chromosome_check.as_ref().is_ok_and(|a| a.len() > 1);

        let ensure_only_one_chromosome = chromosome_check.as_ref().is_ok_and(|a| a.len() == 1);

        let line_number = line.0 as u64; 

        let length_checking = |x: &str| if check_chromosome { chromosome_check.as_ref().ok().map(|a| a.get(x).copied()).flatten() } else if ensure_only_one_chromosome { chromosome_check.as_ref().ok().map(|a| a.values().next()).flatten().copied() } else {None};
        let Some(split_line): Option<[&str; 4]> = line.1.split('\t').collect::<Vec<_>>().try_into().ok() else { potential_error.add_problem_line(line_number, true, false, false, false, false); return None;};

        if check_chromosome { 
            if !(chromosome_check.as_ref().is_ok_and(|a| a.get(split_line[0]).is_some())) { 
                potential_error.add_problem_line(line_number, false, true, false, false, false);
            } else {
                let _ = chromsomes_acquired.insert(split_line[0].to_string());
            }
        } 
        if ensure_only_one_chromosome {
            let inserted = chromsomes_acquired.insert(split_line[0].to_string());
            if inserted && chromsomes_acquired.len() > 1 {
                potential_error.add_problem_line(line_number, false, true, false, false, false);//We shouldn't have multiple chromosomes in the data if there's only one fasta chromosome
            }
        }

        //I write this in this weird redundant way because I don't want to return before catching all the errors I can. 
        if split_line[1].parse::<usize>().is_err() { potential_error.add_problem_line(line_number, false, false, true, false, false); }
        if split_line[3].parse::<f64>().is_err() { potential_error.add_problem_line(line_number, false, false, false, true, false);}

        let Some(loc): Option<usize> = split_line[1].parse().ok() else {return None;};
        let Some(data): Option<f64> = split_line[3].parse().ok() else {return None;};

        if data.is_nan() { potential_error.add_problem_line(line_number, false, false, false, true, false); return None;}

        if let Some(length) = length_checking(split_line[0]) { if loc >= length { potential_error.add_problem_line(line_number, false, false, false, false, true); return None;}};

        Some((split_line[0].to_string(), loc, data))
    }
    


    fn try_read_wiggle_line(line: (usize, &str), potential_error: &mut Option<DataFileBadFormat>, chromosome_name: &str) -> Option<(String, usize, f64)> {

        let line_number = line.0 as u64; 
        
        let Some(split_line): Option<[&str; 2]> = line.1.split('\t').collect::<Vec<_>>().try_into().ok() else { potential_error.add_problem_line(line_number, true, false, false, false, false); return None;};

        if split_line[0].parse::<usize>().is_err() { potential_error.add_problem_line(line_number, false, false, true, false, false); }
        if split_line[1].parse::<f64>().is_err() { potential_error.add_problem_line(line_number, false, false, false, true, false);}

        //wiggle files are 1-indexed. 
        let Some(loc): Option<usize> = split_line[0].parse::<usize>().map(|a| a-1).ok() else {return None;};
        let Some(data): Option<f64> = split_line[1].parse().ok() else {return None;};

        Some((chromosome_name.clone().to_string(), loc, data))
    }


    //TODO: I need a data processing function that reads in a CSV along with a
    //      boolean that tells us if the genome is circular or not
    //      The output needs to be a Vec<Vec<f64>> giving blocks of data we're going to infer on,
    //      a vec of base ranges for each block

    //NOTE: fragment_length actually determines two things: the peak width (defined as the 
    //      the standard deviation of the Gaussian kernel and set to fragment_length/6)
    //      and the length of non-interaction (defined as the length across which a 
    //      a read in one location does not influence the presence of a read in another 
    //      location and set to the fragment length)
    fn process_data(data_file_name: &str, possible_sequence_len: Result<HashMap<String, usize>, FastaError>, fragment_length: usize, spacing: usize, min_height: f64, is_circular: bool, scale_peak_thresh: Option<f64>, retain_null: bool) -> Result<(Vec<(String, Vec<Vec<(usize, f64)>>)>, Background, f64, Vec<(String, Vec<Vec<(usize, f64)>>)>, f64), AllProcessingError> {
       
        if spacing == 0 {
            let err = DataProcessError::BadSpacer(BadSpacerError);
            return Err(err.collect_wrongdoing(possible_sequence_len));
        }

        let att_file = fs::read_to_string(data_file_name);

        if att_file.is_err() {
            let err = DataProcessError::InvalidFile(Arc::new(att_file.unwrap_err()));
            return Err(err.collect_wrongdoing(possible_sequence_len));
        }

        let file_string = att_file.expect("We already returned if this was an error");

        let mut data_as_vec = file_string.split("\n").collect::<Vec<_>>();
       
        while data_as_vec.last() == Some(&"") {
            _ = data_as_vec.pop();
        }

        //let mut locs: Vec<usize> = Vec::with_capacity(data_as_vec.len());
        //let mut data: Vec<f64> = Vec::with_capacity(data_as_vec.len());


        let mut data_iter = data_as_vec.into_iter().peekable();

        let mut format_error: Option<DataFileBadFormat> = None;

        if data_iter.peek().is_none() { return Err(DataProcessError::BlankFile(EmptyDataError).collect_wrongdoing(possible_sequence_len)); }

        let mut collected_chromosomes: HashSet<String> = HashSet::new();

        let mut raw_locs_data: Vec<(String, usize, f64)> = data_iter.enumerate().filter_map(|a| AllData::try_read_bedgraph_line(a, &mut format_error, &possible_sequence_len, &mut collected_chromosomes)).collect();

        if let Some(bad_file) = format_error {
            let err = DataProcessError::BadDataFormat(bad_file);
            return Err(err.collect_wrongdoing(possible_sequence_len));
        }

        //We already returned if there were any data errors. Now we just check if we have sequence errors
        if possible_sequence_len.is_err() { return Err(AllProcessingError::SequenceOnly(possible_sequence_len.unwrap_err())); }

        //Between peeking the data_file iterator and the code to process it after, we know that we have at least ONE data point now
        raw_locs_data.sort_by(|(chr_a, loc_a, _), (chr_b, loc_b, _)| {
            if chr_a.cmp(chr_b) == std::cmp::Ordering::Equal {
                loc_a.cmp(loc_b)
            } else {
                chr_a.cmp(chr_b)
        
            }
        });



        println!("Data sorted");

        let mut sorted_raw_data: Vec<f64> = raw_locs_data.iter().map(|(_, _,a)| *a).collect();

        sorted_raw_data.sort_unstable_by(|a,b| a.partial_cmp(&b).unwrap());

        let median = if sorted_raw_data.len() & 1 == 0 {
            (sorted_raw_data[sorted_raw_data.len()/2]+sorted_raw_data[sorted_raw_data.len()/2-1])/2.0
        } else {
            sorted_raw_data[sorted_raw_data.len()/2]
        };



        let mut peak_scale: f64 = scale_peak_thresh.unwrap_or(1.0);

        if !(peak_scale.is_finite()) {
            warn!("peak scaling must be a valid finite value! Program will use default scaling.");
            peak_scale = 1.0;
        }
        if peak_scale < 0.0 {
            warn!("peak scaling must be a nonnegative value! Taking absolute value of your supplied number!");
            peak_scale = peak_scale.abs();
        }


        let pars = match AllData::estimate_middle_normal(&sorted_raw_data) {
            Some(p) => p,
            None => {
                warn!("Fit on background data failed to converge! Approximating from mad!");
                let dat_off = Data::new(sorted_raw_data.iter().map(|&a| (a-median).abs()).collect::<Vec<f64>>());
                vec![median, dat_off.median()*MAD_ADJUSTER]
            }
        };

        for dat in raw_locs_data.iter_mut() {
            dat.2 -= pars[0];
        }

        for dat in sorted_raw_data.iter_mut() {
            *dat -= pars[0];
        }

        let background_dist = { 
            Background::new(pars[1], f64::INFINITY, (fragment_length as f64)/(WIDE)).expect("We have hard limits on our t distribution fit to prevent an error")//Based on analysis, sd of the kernel is about 1/3 the high fragment length
        } ;
        println!("inferred background dist");

        //sorted_raw_data[index_min_height] is the normal min height. The max is just a guard against pathological data
        let min_height = min_height.max(1.0);

        //Note: this is the last possible place where possible_sequence_len can 
        //      possibly be live. If it died before this, we already returned.
        //TODO: sequence_len needs to be massively updated to account for multiple chromosomes. This get() is only to type check for the compiler for now
        //Ali, this is your next assignment 
        //let sequence_len = *(possible_sequence_len.expect("We only get here if we didn't have an error").get("").unwrap());

        let sequence_len_hashmap = possible_sequence_len.expect("We only get here if we didn't have an error");

        if sequence_len_hashmap.len() == 0 { return Err(AllProcessingError::Synchronization(BadDataSequenceSynchronization::NoChromosomes));}
        
        let unused_chromosomes: Vec<String> = sequence_len_hashmap.keys().filter_map(|a| if collected_chromosomes.contains(a) { Some(a.clone())} else {None}).collect();

        if unused_chromosomes.len() > 0 {
            warn!("You have some chromosomes in your FASTA file which are unaccounted for in your data! \nThese are the unseen chromosomes: {:?}", unused_chromosomes);
        }

        println!("Normalized data location");

        //Compress all data so that locations are unique by taking the mean of the data

        let mut refined_locs_data: Vec<(String, usize, f64)> = Vec::with_capacity(raw_locs_data.len());

        let mut i: usize = 0;

        //Remember, raw_locs_data is sorted, so we will run into all same locations in one stretch
        //I'm using a while loop instead of a for loop because I will likely have runs of same location
        //data that come out to only producing a single data row
        while i < raw_locs_data.len() {

            let curr_chromosome = &(raw_locs_data[i].0);
            let curr_loc = raw_locs_data[i].1;

            let mut to_next_unique = 0;

            //Find the run of data which is all the same location
            //Whether it's stopped by a new location or the end of the data
            while ((i+to_next_unique) < raw_locs_data.len()) 
                && (&(raw_locs_data[i+to_next_unique].0) == curr_chromosome)
                && (raw_locs_data[i+to_next_unique].1 == curr_loc) {
                to_next_unique+=1;
            }

            let mut sum_of_data: f64 = 0.0;

            for j in i..(i+to_next_unique) {
                sum_of_data += raw_locs_data[j].2;
            }

            //For each location, we want to push it onto the data one time, with the mean of its data points
            refined_locs_data.push((curr_chromosome.to_string(), curr_loc, sum_of_data/(to_next_unique as f64))); 

            //We want to skip to the next unique location
            i += to_next_unique;
        }

       


        println!("refined data values");
        //Here, refined_locs_data ultimately has one data value for every represented location in the data,
        //with the data sorted by location in ascending order
 
        let mut chromosome_boundaries: HashMap<String, (usize, usize)> = HashMap::new();
        let mut internal_cuts: HashMap<String, Vec<usize>> = sequence_len_hashmap.keys().map(|a| (a.clone(), vec![])).collect();
        let mut try_circle : HashMap<String, bool> = sequence_len_hashmap.keys().map(|a| (a.clone(), is_circular)).collect();

        let mut current_chrom: String = refined_locs_data[0].0.clone();
        let mut current_chrom_start_ind: usize = 0;

        for i in 1..refined_locs_data.len() { 
            if refined_locs_data[i].0 != current_chrom {
                chromosome_boundaries.insert(current_chrom.clone(), (current_chrom_start_ind, i));
                current_chrom = refined_locs_data[i].0.clone();
                current_chrom_start_ind = i;
            }
        }

        chromosome_boundaries.insert(current_chrom.clone(), (current_chrom_start_ind, refined_locs_data.len()));


        if !retain_null {
            for (name, (start, end)) in chromosome_boundaries.iter() {
                
                for i in (*start)..((*end)-1) {

                    let jump: usize = refined_locs_data[i+1].1-refined_locs_data[i].1; //We can guarentee this is fine because we sorted the data already, and we wouldn't reach here if these were on different chromosomes
                    if jump >= 2*(fragment_length)+5 { //We need the times 2 because the ar correlations can be at least that big. We then add 5 because we want to leave some buffer: there's a division and then multiplication by 6 for the kernels and we don't want to ever have a sequence block with too small a number of spots
                        println!("jump {} {} {} {} {}", i, refined_locs_data[i+1].1, refined_locs_data[i].1, jump, fragment_length);
                        internal_cuts.get_mut(name).expect("Already errorred if this would be None").push(i+1);
                
                    }
                }

                if is_circular { 

                    //We should only circularize if the non present data between the start and the end does not exceed 2*(fragment_length)+5
                    *try_circle.get_mut(name).expect("Already errorred if this would be None") = (refined_locs_data[*start].1+(*sequence_len_hashmap.get(name).expect("Already errorred if this would be None")-refined_locs_data[*end-1].1)) < (2*(fragment_length)+5); 
                }

            }

        } 

        let mut max_valid_run: usize = chromosome_boundaries.iter().map(|(chrom, (start, end))| {
            let mut bounds: Vec<usize> = vec![*start];
            bounds.extend(internal_cuts.get(chrom).expect("has the same keys as chromosome_boundaries by construction"));
            bounds.push(*end);
            let mut lengths = bounds.windows(2).map(|a| a[1]-a[0]).collect::<Vec<_>>();
            if lengths.len() > 1 && *(try_circle.get(chrom).expect("has the same keys as chromosome_boundaries by construction")) {
                let to_add = lengths.pop().expect("You saw that we didn't enter this unless we had a positive length, right?");
                lengths[0]+= to_add;
            }
            lengths.iter().max().copied().expect("definitely non empty because there's at least one end-start element")
        }).max().expect("definitely non empty because there's at least one chromosome");

        let mut certify_cut: HashMap<String, bool> = internal_cuts.iter().map(|(a, b)| (a.clone(), !is_circular || b.len() > 0 || !try_circle.get(a).expect("has the same keys as chromosome_boundaries by construction"))).collect();


        let mut first_blocks: Vec<(String, Vec<Vec<(usize, f64)>>)> = chromosome_boundaries.iter().map(|(chrom,(start, end))| {

            let number_cuts = internal_cuts.get(chrom).expect("has the same keys as chromosome_boundaries by construction").len();
            let mut bounds = vec![*start];
            bounds.extend(internal_cuts.get(chrom).expect("has the same keys as chromosome_boundaries by construction"));
            bounds.push(*end);

            let mut pre_block: Vec<Vec<_>> = bounds.windows(2).map(|b| refined_locs_data[b[0]..b[1]].iter().map(|a| (a.1, a.2)).collect::<Vec<(usize, f64)>>()).collect(); 

            if *(try_circle.get(chrom).expect("has the same keys as chromosome_boundaries by construction")) && (pre_block.len() > 1) {

                let mut to_append = pre_block.remove(0); //Yeah yeah, muh performance. I'm pretty sure I have to take this hit somewhere.
                to_append = to_append.into_iter().map(|mut a| {a.0 += *sequence_len_hashmap.get(chrom).expect("has the same keys as chromosome_boundaries by construction"); a}).collect();
                pre_block.last_mut().expect("didn't get here if we would be empty at this point").append(&mut to_append);
                *(try_circle.get_mut(chrom).expect("has the same keys as chromosome_boundaries by construction")) = false;
            }

            (chrom.clone(), pre_block)

        }).collect();



        first_blocks.sort_unstable_by(|a, b| a.0.cmp(&b.0));


        //Now, we have finished first_blocks, which is a vec of Vec<(usize, f64)>s such that I can lerp each block according to the spacer

        let mut lerped_blocks: Vec<(String, Vec<Vec<(usize, f64)>>)> = first_blocks.into_iter().map(|(a, block_vec)| (a, block_vec.into_iter().map(|block| Self::lerp(&block, spacing)).collect::<Vec<Vec<(usize, f64)>>>())).collect();

        let num_lerped_blocks = lerped_blocks.iter().map(|a| a.1.len()).sum::<usize>();


        //TODO: This is where I got to for implementing the multiple chromosomes

        //Now, we have lerped_blocks, which is a vec of Vec<(usize, f64)>s such that all independent blocks are lerped according to the spacer, amd are non-empty where they exist
        //Sort data into two parts: kept data that has peaks, and not kept data that I can derive the AR model from
        //Keep data that has peaks in preparation for being synced with the sequence
        //Cut up data so that I can derive AR model from not kept data
        println!("length cut up {}", num_lerped_blocks);

        //let peak_thresh = std::f64::consts::SQRT_2*mad_adjust*peak_scale*(raw_locs_data.len() as f64).ln().sqrt();

        let norm_back = Normal::new(0.0, pars[1]).unwrap();
        let peak_thresh = AllData::find_qval_cutoff(&sorted_raw_data, &norm_back, None);
        println!("peak thresh is {peak_thresh} or in pre terms {}", peak_thresh+pars[0]);
        let mut ar_blocks: Vec<(String, Vec<Vec<(usize, f64)>>)> = Vec::with_capacity(lerped_blocks.len());
        let mut data_blocks: Vec<(String, Vec<Vec<(usize, f64)>>)> = Vec::with_capacity(lerped_blocks.len());

        //The maximum spacing where things can start interfering with one another is 3 kernel sds
        //away, because we decided that fragment lengths should be about 6 kernel sds long
        let data_zone: usize = (100/spacing).max(2*fragment_length/spacing); //(2500/spacing).max(2*fragment_length/(spacing));
        

        if retain_null { data_blocks = lerped_blocks;} else {
            
            for (chrom, this_chrom_lerped_blocks) in lerped_blocks.into_iter() { 
            
                let mut this_chrom_ar_blocks: Vec<Vec<(usize, f64)>> = Vec::with_capacity(this_chrom_lerped_blocks.len());
                let mut this_chrom_data_blocks: Vec<Vec<(usize, f64)>> = Vec::with_capacity(this_chrom_lerped_blocks.len());

                for block in this_chrom_lerped_blocks {

                    //Even though I only match positive peaks, I match on b.abs() because it lets me
                    //make stronger assumptions about any data NOT included in the actual inference
                    let poss_peak_vec: Vec<bool> = block.iter().map(|(_, b)| b.abs() > peak_thresh).collect();

                    //Trying what happens on NOT abs filtering
                    //let poss_peak_vec: Vec<bool> = block.iter().map(|(_, b)| *b > peak_thresh).collect();


                    let mut next_ar_ind = 0_usize;
                    let mut curr_data_start: usize;

                    let mut check_ind = 0_usize;

                    while check_ind < block.len() {

                        if poss_peak_vec[check_ind] {
                            curr_data_start = if (next_ar_ind + data_zone) > check_ind {next_ar_ind} else {check_ind-data_zone}; //the if should only ever activate on the 0th block
                            if curr_data_start > next_ar_ind { this_chrom_ar_blocks.push(block[next_ar_ind..curr_data_start].to_vec()); }

                            next_ar_ind = block.len().min(check_ind+data_zone+1);



                            while check_ind < next_ar_ind {
                                if poss_peak_vec[check_ind] {
                                    next_ar_ind = block.len().min(check_ind+data_zone+1);
                                }
                                check_ind += 1;
                            }

                            if spacing < BASE_L {

                                next_ar_ind -= ((next_ar_ind+1-curr_data_start) % BASE_L);

                            }


                            this_chrom_data_blocks.push(block[curr_data_start..next_ar_ind].to_vec());
                        } else {
                            check_ind += 1;
                        }
                    }
                } //This finished the block splitting logic for this chromosome. 


                if *(try_circle.get(&chrom).expect("has the same keys as chromosome_boundaries by construction")) { //This code is only run if chromosomes are circular and weren't cut already by the existence of missing data

                    let starts_data = this_chrom_data_blocks[0][0].0 < this_chrom_ar_blocks[0][0].0;
                    let ends_data = (*(*this_chrom_data_blocks.last().unwrap()).last().unwrap()).0 > 
                        (*(*this_chrom_ar_blocks.last().unwrap()).last().unwrap()).0;
                    let sequence_len = sequence_len_hashmap.get(&chrom).copied().expect("has the same keys as chromosome_boundaries by construction");
                    match (starts_data, ends_data) {
                        (true, true) => { //If both the beginning and end are data, glom the beginning onto the end in data
                            if this_chrom_data_blocks.len() > 1 {
                                let mut rem_block = this_chrom_data_blocks.remove(0); 
                                rem_block = rem_block.into_iter().map(|(a,b)| (a+sequence_len, b)).collect();
                                if let Some(end_vec) = this_chrom_data_blocks.last_mut() { //This is infallible, as the length for data blocks is more than 1 in this chromosome
                                    end_vec.append(&mut rem_block);
                                    *end_vec = Self::lerp(&*end_vec, spacing);
                                    *try_circle.get_mut(&chrom).expect("has the same keys as chromosome_boundaries by construction") = false;
                                    //I need to record that the circularization is taken care of only here, as this could be a single glob of data across the chromosome.
                                    //If this is the case, then I need to issue a warning, and thus I will remember it if we take this conditional. 
                                };
                            }
                        },
                        (false, false) => { //If both the beginning and end are for AR inference, glom the beginning onto the end for AR inference
                            *try_circle.get_mut(&chrom).expect("has the same keys as chromosome_boundaries by construction") = false; //This is now always false, as there's no way for us to just be a single glob of data
                            if this_chrom_data_blocks.len() > 1 {
                                let mut rem_block = this_chrom_ar_blocks.remove(0);
                                rem_block = rem_block.into_iter().map(|(a,b)| (a+sequence_len, b)).collect();
                                if let Some(end_vec) = this_chrom_ar_blocks.last_mut() { //This is infallible, as the length for ar blocks is more than 1 in this chromosome here
                                    end_vec.append(&mut rem_block);
                                    *end_vec = Self::lerp(&*end_vec, spacing);
                                };
                            }
                        },
                        (true, false) => {

                            *try_circle.get_mut(&chrom).expect("has the same keys as chromosome_boundaries by construction") = false; //This is now always false, as there's no way for us to just be a single glob of data
                            let first_force_data = this_chrom_data_blocks[0].iter().position(|(_, b)| b.abs() > peak_thresh).unwrap();
                            let last_data_place = this_chrom_data_blocks[0][first_force_data].0+sequence_len-data_zone;
                            let bleed_into_final_ar = last_data_place <= (*(*this_chrom_ar_blocks.last().unwrap()).last().unwrap()).0;
                            let bleed_into_last_data = last_data_place <= (*this_chrom_ar_blocks.last().unwrap())[0].0;

                            if bleed_into_last_data {
                                let mut rem_block = this_chrom_data_blocks.remove(0);
                                rem_block = rem_block.into_iter().map(|(a,b)| (a+sequence_len, b)).collect();
                                let mut fuse_ar = this_chrom_ar_blocks.pop().unwrap();
                                if let Some(end_vec) = this_chrom_data_blocks.last_mut()  {
                                    end_vec.append(&mut fuse_ar);
                                    end_vec.append(&mut rem_block);
                                    *end_vec = Self::lerp(&*end_vec, spacing);
                                } else {
                                    fuse_ar.append(&mut rem_block);
                                    fuse_ar = Self::lerp(&*fuse_ar, spacing);
                                    this_chrom_data_blocks.push(fuse_ar);
                                };

                            } else if bleed_into_final_ar {
                                let mut rem_block = this_chrom_data_blocks.remove(0);
                                rem_block = rem_block.into_iter().map(|(a,b)| (a+sequence_len, b)).collect();
                                let mut split = 1_usize;
                                if let Some(end_vec) = this_chrom_ar_blocks.last_mut() { //This match is infallible
                                    while (*end_vec)[split].0 < last_data_place {split+=1;}
                                    let mut begin_fin_dat = end_vec.split_off(split);
                                    begin_fin_dat.append(&mut rem_block);
                                    begin_fin_dat = Self::lerp(&begin_fin_dat, spacing);
                                    this_chrom_data_blocks.push(begin_fin_dat);
                                };
                            } //Don't need a final else. It basically boils down to "else: I don't care"


                        },

                        (false, true) => {
                            *try_circle.get_mut(&chrom).expect("has the same keys as chromosome_boundaries by construction") = false; //This is now always false, as there's no way for us to just be a single glob of data
                            let last_force_data = this_chrom_data_blocks.last().unwrap().len()-1-(this_chrom_data_blocks.last().unwrap().iter().rev().position(|(_, b)| b.abs() > peak_thresh).unwrap());
                            if this_chrom_data_blocks.last().unwrap()[last_force_data].0+data_zone > sequence_len {

                                let last_data_place = this_chrom_data_blocks.last().expect("this_chrom_data_blocks must have data if we're here")[last_force_data].0+data_zone-sequence_len;
                                let bleed_into_final_ar = last_data_place <= this_chrom_ar_blocks[0][0].0; 
                                let bleed_into_last_data = last_data_place <= (*((this_chrom_ar_blocks[0]).last().expect("this_chrom_ar_blocks must have data if we're here"))).0;

                                if bleed_into_last_data {
                                    let mut rem_block = this_chrom_data_blocks.remove(0);
                                    rem_block = rem_block.into_iter().map(|(a,b)| (a+sequence_len, b)).collect();
                                    let mut fuse_ar = this_chrom_ar_blocks.remove(0);
                                    fuse_ar = fuse_ar.into_iter().map(|(a,b)| (a+sequence_len, b)).collect();
                                    if let Some(end_vec) = this_chrom_data_blocks.last_mut()  {
                                        end_vec.append(&mut fuse_ar);
                                        end_vec.append(&mut rem_block);
                                        *end_vec = Self::lerp(&*end_vec, spacing);
                                    } else {
                                        fuse_ar.append(&mut rem_block);
                                        fuse_ar = Self::lerp(&*fuse_ar, spacing);
                                        this_chrom_data_blocks.push(fuse_ar);
                                    };

                                } else if bleed_into_final_ar {

                                    let start_vec = &mut this_chrom_ar_blocks[0];
                                    let mut split = start_vec.len()-2;
                                    while (*start_vec)[split].0 < last_data_place {split-=1;}
                                    let mut begin_fin_dat = start_vec.drain(0..split).map(|(a, b)| (a+sequence_len, b)).collect::<Vec<_>>();
                                    this_chrom_data_blocks.last_mut().expect("this_chrom_data_blocks must have data if we're here").append(&mut begin_fin_dat);
                                    *(this_chrom_data_blocks.last_mut().expect("this_chrom_data_blocks must have data if we're here")) = Self::lerp(this_chrom_data_blocks.last().expect("this_chrom_data_blocks must have data if we're here"), spacing);
                                } //Don't need a final else. It basically boils down to "else: I don't care"

                            }
                        },
                    };

                }


                ar_blocks.push((chrom.clone(), this_chrom_ar_blocks));
                data_blocks.push((chrom, this_chrom_data_blocks));
            }
        }


        println!("sorted data data away from background inference data");


        let ar_lens = ar_blocks.iter().map(|a| a.1.iter().map(|b| b.len()).sum::<usize>()).sum::<usize>();
        let data_lens = data_blocks.iter().map(|a|  a.1.iter().map(|b| b.len()).sum::<usize>()).sum::<usize>();

        if !retain_null && (ar_lens == 0) && (data_lens == 0) {
            return Err(AllProcessingError::Synchronization(BadDataSequenceSynchronization::NeedDifferentExperiment));
        }

        println!("past first");

        if !retain_null && ar_lens == 0 {
            return Err(AllProcessingError::Synchronization(BadDataSequenceSynchronization::NotEnoughNullData));
        }

        if data_lens == 0 {
            return Err(AllProcessingError::Synchronization(BadDataSequenceSynchronization::NotEnoughPeakData));
        }

        //TODO:This is where I stopped. 

        println!("Ensured that cutting happened");


        //Now, we have data_blocks and ar_blocks, the former of which will be returned and the latter of which will be processed by AR prediction


        ar_blocks = ar_blocks.into_iter().map(|(a, mut b)| { b.retain(|c| c.len() > data_zone); (a, b)}).collect();

        //SAFETY: This data block filtering is what allows us to guarentee Kernel
        //is always compatible with the sequence to be synchronized, the data Waveform, and all other Waveforms 
        //derived from the data. The invariant is upheld because (fragment_length as f64)/(2.0*WIDE)
        //never rounds down
        data_blocks = data_blocks.into_iter().map(|(c, mut b)| { b.retain(|a| (a.len()+1)*spacing > fragment_length); (c, b)}).collect();


       
        let ar_lens = ar_blocks.iter().map(|a| a.1.iter().map(|b| b.len()).sum::<usize>()).sum::<usize>();
        let data_lens = data_blocks.iter().map(|a|  a.1.iter().map(|b| b.len()).sum::<usize>()).sum::<usize>();

        if !retain_null && (ar_lens == 0) && (data_lens == 0) {
            return Err(AllProcessingError::Synchronization(BadDataSequenceSynchronization::NeedDifferentExperiment));
        }


        if !retain_null && ar_lens == 0 {
            return Err(AllProcessingError::Synchronization(BadDataSequenceSynchronization::NotEnoughNullData));
        }

        if data_lens == 0 {
            return Err(AllProcessingError::Synchronization(BadDataSequenceSynchronization::NotEnoughPeakData));
        }

        println!("past second");

        //This is a hard limit, because technically, we only ever computed the
        //limit of the eth statistic as the number of points increases without bound
        //Based on our experience with the Anderson Darling statistic, which the
        //eth statistic is based on, it should converge like 1/sqrt(n). So, we are
        //blatantly assuming that 100 is enough. The Marsaglias claim that there's
        //a max error of about 0.005 even if n=8 for the AD statistic when comparing
        //to ADinf (Marsaglia and Marsaglia, "Evaluating the Anderson-Darling 
        //Distribution", 2004, Journal of Statistical Software), but I'm not super
        //confident in the application of this to eth.
        //Again, I'm blatantly pretending that 100 is sufficient. I would guess 
        //it's _probably_ fine???? 100 vs 8 is a lot of wiggle room, and eth is
        //quite related to AD
        if data_lens < 100 { return Err(AllProcessingError::Synchronization(BadDataSequenceSynchronization::NotEnoughPeakData)); } 

        let ar_info = if !retain_null { ar_blocks.iter().map(|(b, c)| (b, c.iter().map(|a| (a.len(), a[0].0, a[a.len()-1].0)).collect::<Vec<_>>())).collect::<Vec<_>>()} else {Vec::new()} ;
        let data_info = data_blocks.iter().map(|(b, c)| (b, c.iter().map(|a| (a.len(), a[0].0, a[a.len()-1].0)).collect::<Vec<_>>())).collect::<Vec<_>>();

        println!("AR lens, start loc, and end datas {:?}", ar_info);
        println!("total AR amount {}", ar_lens);
        println!("data lens, start loc, and end datas {:?}", data_info); 
        println!("total data amount {}", data_lens);

        //let ar_inference: Vec<Vec<f64>> = ar_blocks.iter().map(|a| a.iter().map(|(_, b)| *b).collect::<Vec<f64>>() ).collect();


        //Despite yule_walker_ar_coefficients_with_bic and estimate_t_dist both
        //having the CAPACITY to panic if ar_inference has only empty vectors, 
        //the invariants we guarenteed if we're at this point of the function
        //mean that this will not be the case. There ARE pathological cases in
        //the inference of the results, but only if we have like fewer than 2 data
        //points, which doesn't happen with the way we defined data_zone


        if try_circle.values().any(|&a| a) { 

            let chromosomes_uncut: Vec<String> = try_circle.iter().filter_map(|(name, uncut)| if *uncut { Some(name.clone())} else {None}).collect();

            warn!("You indicated that your chromosomes are circular, but you have at least one chromosome which is so peaky that we couldn't cut it! We will continue to do inference as though each of them are linear chromosomes, but be aware that this can be wrong. \nThe troublesome chromosomes are {:?}", chromosomes_uncut);

        }

        //Send off the kept data with locations in a vec of vecs and the background distribution from the AR model
        Ok((data_blocks, background_dist, min_height, ar_blocks, peak_thresh))



    }



    //TODO: I need a function which marries my processed FASTA file and my processed data
    //      By the end, I should have a Sequence and WaveformDef, which I can 
    //      use in my public function to create an AllData instance
    //
    //      If you're eagle eyed, you'll notice that there's some bits of the 
    //      sequence that get assigned neither to null sequence nor to positive
    //      Yes, you're very clever. This issue exists because I don't actually 
    //      care: I'm losing about spacer bps for each block in both data sets
    //      But these are on the edge between being positive and negative data,
    //      so they're in a little bit of a grey area to classify, and saving 
    //      them isn't my highest priority anyway.
    fn synchronize_sequence_and_data(pre_sequence_map: HashMap<String, Vec<Option<usize>>>, pre_chrom_data: Vec<(String, Vec<Vec<(usize, f64)>>)>, pre_chrom_null_data: Vec<(String, Vec<Vec<(usize, f64)>>)>, sequence_len: usize, spacing: usize, peak_thresh: f64) -> Result<(Sequence, NullSequence, WaveformDef, Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>, Vec<String>), WaveCreationError> {
        let data_block_num = pre_chrom_data.iter().map(|a| a.1.len()).sum::<usize>();
        let null_block_num = pre_chrom_null_data.iter().map(|a| a.1.len()).sum::<usize>();
        let mut sequence_blocks: Vec<Vec<usize>> = Vec::with_capacity(data_block_num);
        let mut null_sequence_blocks: Vec<Vec<usize>> = Vec::with_capacity(null_block_num);
        let mut start_data: Vec<f64> = Vec::with_capacity(pre_chrom_data.iter().map(|a| a.1.iter().map(|b| b.len()).sum::<usize>()).sum::<usize>());
        let mut starting_coords: Vec<usize> = Vec::with_capacity(data_block_num);
        let mut null_starts_bps: Vec<usize> = Vec::with_capacity(null_block_num);
        let mut which_chromosomes_seq: Vec<usize> = Vec::with_capacity(data_block_num);
        let mut which_chromosomes_null: Vec<usize> = Vec::with_capacity(data_block_num);
        let peak_thresh = peak_thresh.max(0.0);
        //let mut windows_are_positive: Vec<bool> = Vec::with_capacity(pre_data.len());

        let mut positive_window_parts: Vec<Vec<usize>> = Vec::with_capacity(data_block_num);

        let mut chromosome_id: HashMap<String, usize> = pre_chrom_data.iter().enumerate().map(|(i, a)| (a.0.clone(), i)).collect();

        let mut match_chromosomes: Vec<String> = pre_chrom_data.iter().map(|a| a.0.clone()).collect();

        for (chrom, _) in pre_chrom_null_data.iter() {
            if chromosome_id.get(chrom).is_none() { 
                chromosome_id.insert(chrom.clone(), match_chromosomes.len());
                match_chromosomes.push(chrom.clone());
            }
        }


        for (chromosome_name, pre_data) in pre_chrom_data.into_iter() {

            let this_id = *(chromosome_id.get(&chromosome_name).expect("We constructed chromosome_id to have this chromosome"));

            let mut i = 0_usize;

            let pre_sequence = pre_sequence_map.get(&chromosome_name).expect("This is a private function where I expect pre_chrom_data to have proper chromosome names or to have panicked if not");
            let seq_len = pre_sequence.len();
            while i < pre_data.len() {

                let mut no_null_base = true;


                let mut bp_ind = pre_data[i][0].0;
                let bp_prior = bp_ind;
                let mut float_batch: Vec<f64> = pre_data[i].iter().map(|&(_,b)| b).collect();
                //let window_is_positive = float_batch.iter().any(|&b| b >= peak_thresh);
                //let min_target_bp = (*(pre_data[i].last().unwrap())).0+1;//We include the +1 here because we want to include the bp corresponding to the last location

                let positive_window_part = float_batch.iter().enumerate().filter_map(|(i, &f)| if f >= peak_thresh { Some(i*spacing)} else {None}).collect::<Vec<usize>>();
                //This needs to be float_batch.len()-1 because we need the bp that matches
                //the last INDEX of float_batch, not its length. We handle this slightly 
                //differently if spacing is small, because that edge case needs special care
                let min_target_bp = if spacing >= BP_PER_U8 {bp_prior + (float_batch.len()-1)*spacing + 1} else {bp_prior + float_batch.len()*spacing +1};

                //SAFETY: This line, in conjunction with the previous checks on pre_data
                //        necessary to call this function, upholds our safety invariants 
                //        when we actually generate the occupancy signals with Waveform::place_peak
                let target_bp = if ((min_target_bp-bp_prior) % BP_PER_U8) == 0 {min_target_bp} else {min_target_bp+BP_PER_U8-((min_target_bp-bp_prior) % BP_PER_U8)};

                let mut bases_batch: Vec<usize> = Vec::with_capacity(pre_data[i].len()*spacing);



                while no_null_base && (bp_ind < target_bp){ 
                    match pre_sequence[bp_ind % sequence_len] { //I don't need to explicitly check for circularity: process_data handled this for me already
                        Some(bp) => bases_batch.push(bp),
                        None => no_null_base = false,
                    };
                    bp_ind+=1;
                }

                if no_null_base {
                    //10 Seq block 0, 1104 276 110 4210650 110 1100 1101 1104
                    //10 Seq block 1, 2032 508 203 4212520 203 2030 2031 2032
                    println!("{spacing} Seq block {i}, {} {} {} {} {} {} {} {}", bases_batch.len(), bases_batch.len()/BASE_L, bases_batch.len()/(spacing), bp_prior, float_batch.len(), float_batch.len()*spacing, min_target_bp-bp_prior, target_bp-bp_prior);
                    sequence_blocks.push(bases_batch);
                    which_chromosomes_seq.push(this_id);
                    starting_coords.push(bp_prior % seq_len);
                    positive_window_parts.push(positive_window_part);
                    start_data.append(&mut float_batch);
                } else {
                    //This gives the 1-indexed position of the null base in vim
                    warn!("You have a null base in position {} of your sequence in a position with data relevant to a possible peak. We are discarding this data for inference purposes.",bp_ind);
                }




                i += 1;
            }
        }

        for (chromosome_name, pre_null_data) in pre_chrom_null_data.into_iter() {

            let this_id = *(chromosome_id.get(&chromosome_name).expect("We constructed chromosome_id to have this chromosome"));


            let mut i = 0_usize;
            let pre_sequence = pre_sequence_map.get(&chromosome_name).expect("This is a private function where I expect pre_chrom_data to have proper chromosome names or to have panicked if not");
            let seq_len = pre_sequence.len();

            while i < pre_null_data.len() {

                let mut bp_ind = pre_null_data[i][0].0;
                let mut bp_prior = bp_ind;
                let min_target_bp = (*(pre_null_data[i].last().unwrap())).0+1;
                let target_bp = if ((min_target_bp-bp_prior) % BP_PER_U8) == 0 {min_target_bp} else {min_target_bp+BP_PER_U8-((min_target_bp-bp_prior) % BP_PER_U8)};

                println!("bp_ind {bp_ind}, target_bp {target_bp}, i {i}");

                let mut bases_batch: Vec<usize> = Vec::with_capacity(pre_null_data[i].len()*spacing);

                while bp_ind < target_bp {
                    match pre_sequence[bp_ind % sequence_len] {
                        Some(bp) => {
                            //println!("In some case");
                            bases_batch.push(bp)
                        },
                        None => {

                            //println!("In None Case");
                            let final_prev_bp = bp_ind-((bp_ind-bp_prior) % BP_PER_U8);//This keeps our number of bases in each block divisible by BP_PER_U8

                            let previous_batch: Vec<usize> = bases_batch.drain(0..(final_prev_bp-bp_prior)).collect();

                            if previous_batch.len() >= MAX_BASE { //we don't need little blocks that can't having binding in them anyway, but we don't need to uphold any place_peak invariants like we do for the positive sequence and data
                                null_starts_bps.push(bp_prior);
                                null_sequence_blocks.push(previous_batch);
                                which_chromosomes_null.push(this_id);
                            }
                            bases_batch.clear(); //In case there are a couple of straggling bases before the current bp
                            bp_prior = bp_ind+1; //We want to skip the null base
                        },
                    };
                    bp_ind += 1;
                }

                if bases_batch.len() >= MAX_BASE { //we don't need little blocks that can't having binding in them anyway, but we don't need to uphold any place_peak invariants like we do for the positive sequence and data
                    null_sequence_blocks.push(bases_batch);
                    null_starts_bps.push(bp_prior % seq_len);
                    which_chromosomes_null.push(this_id);
                }

                i += 1;
            }
        }

        let seq = Sequence::new(sequence_blocks, &positive_window_parts);
        let null_seq = NullSequence::new(null_sequence_blocks);
        // TODO: either verify this can't fail at this stage or make this function return a Result
        let wave = match Waveform::new(start_data, &seq, spacing) {
            Ok(w) => w,
            Err(e) => return Err(e),
        };
        let wave_ret = WaveformDef::from(&wave);

        //TODO: this needs to actually use the right names and chromosome coordinates
        Ok((seq, null_seq, wave_ret, starting_coords, null_starts_bps, which_chromosomes_seq, which_chromosomes_null, match_chromosomes))

    }

    fn chop_file_name(name: &str) -> String {
        
        let named_split = name.split("/").collect::<Vec<_>>();

        let named = if named_split[named_split.len()-1] == "" {named_split[named_split.len()-2]} else {named_split[named_split.len()-1]};

        let re = Regex::new(r"\.\pL+$").unwrap();
        let piece_name = re.replace(named, "");
        String::from(piece_name)
    }


    /// This is a helper function that computes autocorrelation coefficients across 
    /// independent blocks
    pub fn compute_autocorrelation_coeffs(data: &Vec<Vec<f64>>, mut num_coeffs: usize) -> Vec<f64>{

        let min_data_len = data.iter().map(|a| a.len()).min().expect("Why are you trying to get autocorrelations from no data?");

        if num_coeffs >= min_data_len {
            num_coeffs = min_data_len-1;
        }

        let mut coeffs = vec![0.0; num_coeffs+1];

        coeffs[0] = 1.0; //all data is perfectly autocorrelated with itself

        for lag in 1..coeffs.len() {



            let domain_vecs  = data.iter().map(|a| a[0..(a.len()-lag)].to_vec());
            let range_vecs   = data.iter().map(|a| a[lag..(a.len())].to_vec());

            let len_domain_and_range = domain_vecs.clone().map(|a| a.len()).sum::<usize>() as f64;

            let domain_mean = domain_vecs.clone().map(|a| a.iter().sum::<f64>()).sum::<f64>()/len_domain_and_range;
            let range_mean = range_vecs.clone().map(|a| a.iter().sum::<f64>()).sum::<f64>()/len_domain_and_range;

            let domain_var = domain_vecs.clone().map(|a| a.iter().map(|d| (d-domain_mean).powi(2)).sum::<f64>()).sum::<f64>()/len_domain_and_range;
            let range_var = range_vecs.clone().map(|a| a.iter().map(|r| (r-range_mean).powi(2)).sum::<f64>()).sum::<f64>()/len_domain_and_range;

            let covar = domain_vecs.zip(range_vecs).map(|(dv, rv)| dv.into_iter().zip(rv).map(|(d, r)| (d-domain_mean)*(r-range_mean)).sum::<f64>()).sum::<f64>()/len_domain_and_range;

            coeffs[lag] = covar/(domain_var*range_var).sqrt();
        }
        coeffs
    }


    /// This uses the Nelder-Mead algorithm to fit the spread and degrees of
    /// of freedom parameter of a zero centered T distribution. Note that
    /// the degrees of freedom have a minimum of `2.0`. This returns the
    /// fit in order of (spread, degrees of freedom)
    pub fn estimate_t_dist(decorrelated_data: &Vec<f64>) -> (f64, f64) {

        let total_sample_variance = decorrelated_data.iter().map(|a| a.powi(2)).sum::<f64>()/((decorrelated_data.len()-1) as f64);

        let init_dfs = vec![2.01, 4.0, 23.0]; //df really close to 2, df sanely close to 2, and df above our "give up to normal" point

        let init_simplex: Vec<Vec<f64>> = init_dfs.iter().map(|&a| vec![(total_sample_variance*(a-2.0)/a).sqrt(), a]).collect();

        let cost = FitTDist { decorrelated_data };

        println!("began nelder mead");
        //This is fine to unwrap because we are setting a valid tolerance
        let solver = NelderMead::new(init_simplex).with_sd_tolerance(1e-6).unwrap();

        let executor = Executor::new(cost, solver);

        //This is fine to unwrap because a rerun would probably fix it: I 
        //do not know what advice to give if this breaks, because it's probably fine
        let res = executor.run().unwrap();


        println!("end nelder mead");

        let best_vec = res.state().get_best_param().unwrap();

        println!("Results (sd then df) {:?}", best_vec);

        let sd_guess = best_vec[0];
        let df_guess = best_vec[1];

        (sd_guess, df_guess)

    }

    /// This uses the Nelder-Mead algorithm to fit the spread of a 
    /// normal distribution ONLY by the data between the 25%ile and 75%ile. 
    /// This returns the fit through the estimated standard deviation. 
    /// Returns `None` if you try to get the variance of a single value
    /// or an empty slice (come on, dude), if the slice more generally has 
    /// fewer than four elements, or if the fit fails to converge. 
    pub fn estimate_middle_normal(data: &[f64]) -> Option<Vec<f64>> {

        
        let length_data = data.len();
       
        if length_data < 4 {return None;}

        let mut sorted_raw_data: Vec<f64> = data.iter().map(|&a| a).collect();

        sorted_raw_data.sort_unstable_by(|a,b| a.partial_cmp(b).unwrap());

        let max_val = *sorted_raw_data.last().unwrap();

        let sd_estimate = 0.25_f64;

        let cost = FitNormalDist::new(&sorted_raw_data)?;

        let mut rng = rand::thread_rng();

        let a: [Vec<_>;2] = core::array::from_fn(|_| {
        let guess_mean_0: f64 = -0.1+0.025*rng.gen::<f64>();
        let guess_sd_0: f64 = 0.5+0.25* rng.gen::<f64>();
        let guess_mean_1: f64 = 0.1+0.025*rng.gen::<f64>();
        let guess_sd_1: f64 = 1.0+0.25*rng.gen::<f64>();

        let init_simplex = vec![vec![guess_mean_0, guess_sd_0], vec![guess_mean_1, guess_sd_0], vec![guess_mean_0, guess_sd_1]];

        let solver = NelderMead::new(init_simplex).with_sd_tolerance(1e-8).unwrap();

        let executor = Executor::new(cost.clone(), solver);

        //This is fine to unwrap because a rerun would probably fix it: I 
        //do not know what advice to give if this breaks, because it's probably fine
        let res = executor.run().unwrap();


        println!("end nelder mead");
        //println!("q1 {q1} q3 {q3}");


        let best_vec = res.state().get_best_param().unwrap();

        println!("Results {:?}", best_vec);
        
         best_vec.to_vec()

        });

        Some(a[0].clone())
    }

    //If weight_normal is less than 0.9999, calculates cutoff where density at log_normal_dist*(1-weight_normal) 
    //exceeds the density at weight_normal*normal_dist. Otherwise, calculates it based on the spread of normal_dist
    //and the size of the data set
    fn find_cutoff(weight_normal: f64, normal_dist: &Normal, log_normal_dist: &LogNormal, data_size: usize, peak_scale: Option<f64>) -> f64 {


        let mut top_cutoff = std::f64::consts::SQRT_2*MAD_ADJUSTER*normal_dist.std_dev().unwrap()*peak_scale.unwrap_or(1.0)*(data_size as f64).ln().sqrt();
        
        if weight_normal > 0.9999 { return top_cutoff;} 
        
        let cutoff_function = |val: f64| -> f64 {weight_normal*normal_dist.pdf(val)-(1.0-weight_normal)*log_normal_dist.pdf(val)};

        let mut top_diff = cutoff_function(top_cutoff);

        //I would be genuinely shocked if this snippet of code executes, ever. But it's here, jic
        if top_diff > 0.0 { return top_cutoff;}

        let mut bottom_cutoff = normal_dist.mean().unwrap() + f64::EPSILON;

        let mut bottom_diff = cutoff_function(bottom_cutoff);

        if bottom_diff < 0.0 { return normal_dist.mean().unwrap();}

        let mut diff = top_cutoff-bottom_cutoff;

        //top_cutoff should always be greater than bottom_cutoff
        while diff > 0.001 {

            let mut test_cutoff = 0.5*(top_cutoff+bottom_cutoff);//(bottom_cutoff*top_diff-top_cutoff*bottom_diff)/(top_diff-bottom_diff);
            let mut test_diff = cutoff_function(test_cutoff);

            println!("{top_cutoff} {top_diff} {test_cutoff} {test_diff} {bottom_cutoff} {bottom_diff}");
            //We cheat with the Illinois algorithm if we are failing to make progress
            if test_cutoff == bottom_cutoff {
                test_cutoff = (bottom_cutoff*top_diff-0.5*top_cutoff*bottom_diff)/(top_diff-0.5*bottom_diff);
                test_diff = cutoff_function(test_cutoff);
            } else if test_cutoff == top_cutoff {
                test_cutoff = (0.5*bottom_cutoff*top_diff-top_cutoff*bottom_diff)/(0.5*top_diff-bottom_diff);
                test_diff = cutoff_function(test_cutoff);
            }

            if test_diff == 0.0 { //If this branch ever runs, I will die of shock
                return test_cutoff;
            } else if test_diff < 0.0 {
                top_cutoff = test_cutoff;
                top_diff = test_diff;
            } else if test_diff > 0.0 {
                bottom_cutoff = test_cutoff;
                bottom_diff=test_diff;
            }
            diff = top_cutoff-bottom_cutoff;
        }

        (bottom_cutoff*top_diff-top_cutoff*bottom_diff)/(top_diff-bottom_diff)

    }
    
    fn find_qval_cutoff(sorted_data: &[f64], normal_dist: &Normal, peak_scale: Option<f64>) -> f64 {

        let len = sorted_data.len() as f64;
        //peak_scale.unwrap_or(1.0)*sorted_data.iter().enumerate().map(|(i, &a)| (a, len*normal_dist.sf(a)/((sorted_data.len()-i) as f64)))
          //                                        .filter(|a| a.1 <= 0.0001).next().unwrap().0
        let m: Vec<_> = sorted_data.iter().enumerate().map(|(i, &a)| (a, len*normal_dist.sf(a)/((sorted_data.len()-i) as f64))).collect();
        println!("m {:?}", m);
        peak_scale.unwrap_or(1.0)*m.iter().filter(|a| a.1 <= 0.005).next().unwrap().0
    }

    fn lerp(data: &[(usize, f64)], spacer: usize ) -> Vec<(usize, f64)> {

        let begins = data[0].0;
        let mut ends = (*(data.last().unwrap())).0;
        

        let capacity = 1+((ends-begins)/spacer);

        let mut locs_to_fill: Vec<usize> = Vec::with_capacity(capacity);

        let mut loc = begins;

        while loc <= ends {
            locs_to_fill.push(loc);
            loc += spacer;
        }

        let mut lerped: Vec<(usize, f64)> = Vec::with_capacity(locs_to_fill.len());

        lerped.push(data[0].clone());

        let mut prev_ind = 0;

        for loc in locs_to_fill {
            let mut curr_ind = prev_ind+1; //This should never cause a problem because we constructed locs_to_fill to always fit inside the block
            
            while data[curr_ind].0 < loc {
                prev_ind = curr_ind;
                curr_ind += 1;
            }

            if data[curr_ind].0 == loc {
                lerped.push(data[curr_ind].clone());
            } else {
                let progress: f64 = ((loc - data[prev_ind].0) as f64)/((data[curr_ind].0-data[prev_ind].0) as f64);
                lerped.push((loc, (1.0-progress)*data[prev_ind].1+progress*data[curr_ind].1));
            }
        }

        lerped

    }

    /// This will try to set the minimum height based on the minimum height you give it.
    /// However, if `min_height <  1.0`, this will set min_height to 1.0 instead. 
    pub fn set_min_height(&mut self, min_height: f64) {
        let mut min_height = min_height;
        if min_height < 1.0 {
            warn!("Motif finder relies on the minimum height being at least 1.0 for mixing reasons! Setting min_height to 1.0");
            min_height = 1.0;
        }
        self.min_height = min_height;
    }

    /// This will try to set the penalty for each additional motif to `credibility.abs()`.
    /// 
    /// # Panics
    /// If `credibility.is_nan()`.
    pub fn set_credibility(&mut self, credibility: f64) {
        assert!(!credibility.is_nan(), "Do you think NaNs are gonna play well with motif inferences? Come on, dude.");
        self.credibility = credibility.abs();
    }

}

/*pub struct AllData {

    seq: Sequence,
    null_seq: NullSequence,
    data: WaveformDef,
    start_genome_coordinates: Vec<usize>,
    start_nullbp_coordinates: Vec<usize>,
    genome_block_chrs: Vec<usize>,
    nullbp_block_chrs: Vec<usize>,
    chr_names: Vec<String>,
    background: Background,
    min_height: f64,
    credibility: f64,
}
*/

impl Serialize for AllData {

    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {

        let neglect_chr_names = self.chr_names.len() == 0 || (self.chr_names.len() == 1 && self.chr_names[0].is_empty()); 

        let mut state = if neglect_chr_names { serializer.serialize_struct("AllData", 8)?} else {serializer.serialize_struct("AllData", 11)?};


        
        state.serialize_field("seq", &self.seq)?;
        state.serialize_field("null_seq", &self.null_seq)?;
        state.serialize_field("data", &self.data)?;
        state.serialize_field("start_genome_coordinates", &self.start_genome_coordinates)?;
        state.serialize_field("start_nullbp_coordinates", &self.start_nullbp_coordinates)?;
        if !neglect_chr_names{
            state.serialize_field("genome_block_chrs", &self.genome_block_chrs)?;
            state.serialize_field("nullbp_block_chrs", &self.nullbp_block_chrs)?;
            state.serialize_field("chr_names", &self.chr_names)?;
        }
        state.serialize_field("background", &self.background)?;
        state.serialize_field("min_height", &self.min_height)?;
        state.serialize_field("credibility", &self.credibility)?;


        state.end()
    }

}

impl<'de> Deserialize<'de> for AllData {

    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {

        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field {Seq, Null_Seq, Data, Start_Genome_Coordinates, Start_Nullbp_Coordinates, Genome_Block_Chrs, Nullbp_Block_Chrs, Chr_Names, Background, Min_Height, Credibility}

        struct AllDataVisitor;
        impl<'de> Visitor<'de> for AllDataVisitor {
            type Value = AllData;
            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct AllData")
            }

            fn visit_seq<V>(self, mut seq_acc: V) -> Result<AllData, V::Error>
            where
                V: SeqAccess<'de>,
            {   
                let seq: Sequence = seq_acc.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(0, &self))?;
                let null_seq: NullSequence = seq_acc.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(1, &self))?;
                let data: WaveformDef = seq_acc.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(2, &self))?;
                let start_genome_coordinates: Vec<usize> = seq_acc.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(3, &self))?;
                let start_nullbp_coordinates: Vec<usize> = seq_acc.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(4, &self))?;
                let genome_block_chrs: Vec<usize> = seq_acc.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(5, &self))?;
                let nullbp_block_chrs: Vec<usize> = seq_acc.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(6, &self))?;
                let chr_names: Vec<String> = seq_acc.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(7, &self))?;
                let background: Background = seq_acc.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(8, &self))?;
                let min_height: f64 = seq_acc.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(9, &self))?;
                let credibility: f64 = seq_acc.next_element()?.ok_or_else(|| serde::de::Error::invalid_length(10, &self))?;

                Ok(AllData {
                    seq,
                    null_seq,
                    data,
                    start_genome_coordinates,
                    start_nullbp_coordinates,
                    genome_block_chrs,
                    nullbp_block_chrs,
                    chr_names,
                    background,
                    min_height,
                    credibility,
                })

            }

            fn visit_map<V>(self, mut map: V) -> Result<AllData, V::Error>
            where
                V: MapAccess<'de>,
            {

                let mut seq: Option<Sequence> = None;
                let mut null_seq: Option<NullSequence> = None;
                let mut data: Option<WaveformDef> = None;
                let mut start_genome_coordinates: Option<Vec<usize>> = None;
                let mut start_nullbp_coordinates: Option<Vec<usize>> = None;
                let mut genome_block_chrs: Option<Vec<usize>> = None;
                let mut nullbp_block_chrs: Option<Vec<usize>> = None;
                let mut chr_names: Option<Vec<String>> = None;
                let mut background: Option<Background> = None;
                let mut min_height: Option<f64> = None;
                let mut credibility: Option<f64> = None;

                while let Some(key) = map.next_key()? {

                    match key {
                        Field::Seq => {
                            if seq.is_some(){

                                return Err(serde::de::Error::duplicate_field("seq"));
                            }
                            seq = Some(map.next_value()?);
                        },
                        Field::Null_Seq => {
                            if null_seq.is_some(){

                                return Err(serde::de::Error::duplicate_field("null_seq"));
                            }
                            null_seq = Some(map.next_value()?);
                        },
                        Field::Data => {
                            if data.is_some(){

                                return Err(serde::de::Error::duplicate_field("data"));
                            }
                            data = Some(map.next_value()?);
                        },
                        Field::Start_Genome_Coordinates => {
                            if start_genome_coordinates.is_some(){

                                return Err(serde::de::Error::duplicate_field("start_genome_coordinates"));
                            }
                            start_genome_coordinates = Some(map.next_value()?);
                        },
                        Field::Start_Nullbp_Coordinates => {
                            if start_nullbp_coordinates.is_some(){

                                return Err(serde::de::Error::duplicate_field("start_nullbp_coordinates"));
                            }
                            start_nullbp_coordinates = Some(map.next_value()?);
                        },
                        Field::Genome_Block_Chrs => {
                            if genome_block_chrs.is_some(){

                                return Err(serde::de::Error::duplicate_field("genome_block_chrs"));
                            }
                            genome_block_chrs = Some(map.next_value()?);
                        },
                        Field::Nullbp_Block_Chrs => {
                            if nullbp_block_chrs.is_some(){

                                return Err(serde::de::Error::duplicate_field("nullbp_block_chrs"));
                            }
                            nullbp_block_chrs = Some(map.next_value()?);
                        },
                        Field::Chr_Names => {
                            if chr_names.is_some(){

                                return Err(serde::de::Error::duplicate_field("chr_names"));
                            }
                            chr_names = Some(map.next_value()?);
                        },
                        Field::Background => {
                            if background.is_some(){

                                return Err(serde::de::Error::duplicate_field("background"));
                            }
                            background = Some(map.next_value()?);
                        },
                        Field::Min_Height => {
                            if min_height.is_some(){

                                return Err(serde::de::Error::duplicate_field("min_height"));
                            }
                            min_height = Some(map.next_value()?);
                        },
                        Field::Credibility => {
                            if credibility.is_some(){

                                return Err(serde::de::Error::duplicate_field("credibility"));
                            }
                            credibility = Some(map.next_value()?);
                        }
                    }

                } //This ends the while loop

                let seq = seq.ok_or_else(|| serde::de::Error::missing_field("seq"))?;
                let null_seq = null_seq.ok_or_else(|| serde::de::Error::missing_field("null_seq"))?;
                let data = data.ok_or_else(|| serde::de::Error::missing_field("data"))?;
                let start_genome_coordinates = start_genome_coordinates.ok_or_else(|| serde::de::Error::missing_field("start_genome_coordinates"))?;
                let start_nullbp_coordinates = start_nullbp_coordinates.ok_or_else(|| serde::de::Error::missing_field("start_nullbp_coordinates"))?;
                let background = background.ok_or_else(|| serde::de::Error::missing_field("background"))?;
                let min_height = min_height.ok_or_else(|| serde::de::Error::missing_field("min_height"))?;
                let credibility = credibility.ok_or_else(|| serde::de::Error::missing_field("credibility"))?;


                if genome_block_chrs.is_none() && nullbp_block_chrs.is_none() && chr_names.is_none() {
                    let genome_block_chrs = vec![0, start_genome_coordinates.len()];
                    let nullbp_block_chrs = vec![0, start_nullbp_coordinates.len()];
                    let chr_names = vec!["".to_owned()];

                    return Ok(AllData {
                        seq,
                        null_seq,
                        data,
                        start_genome_coordinates,
                        start_nullbp_coordinates,
                        genome_block_chrs,
                        nullbp_block_chrs,
                        chr_names,
                        background,
                        min_height,
                        credibility,
                    });

                } else {
                    let mut error_string: Option<String> = None;
                    if genome_block_chrs.is_none() { error_string.insert("genome_block_chrs is missing when it's needed".to_string());}
                    if nullbp_block_chrs.is_none() { 
                        match error_string.as_mut() {
                            Some(message) => message.push_str("\nnullbp_block_chrs is missing when it's needed"),
                            None => {error_string.insert("nullbp_block_chrs is missing when it's needed".to_string());},
                        }
                    }

                    if chr_names.is_none() {
                        match error_string.as_mut() { 
                            Some(message) => message.push_str("\nchr_names is missing when it's needed"),
                            None => {error_string.insert("chr_names is missing when it's needed".to_string());},
                        }
                    }

                    if let Some(error) = error_string {return Err(serde::de::Error::custom(error))};

                    //These are technically infallible at this point.
                    let genome_block_chrs = genome_block_chrs.ok_or_else(|| serde::de::Error::missing_field("genome_block_chrs"))?;
                    let nullbp_block_chrs = nullbp_block_chrs.ok_or_else(|| serde::de::Error::missing_field("nullbp_block_chrs"))?;
                    let chr_names = chr_names.ok_or_else(|| serde::de::Error::missing_field("chr_names"))?;

                    if genome_block_chrs.len() != start_genome_coordinates.len() { return Err(serde::de::Error::custom("block sequence chromosome indications must be as many blocks as the sequence"));}
                    if nullbp_block_chrs.len() != start_nullbp_coordinates.len() { return Err(serde::de::Error::custom("block null sequence chromosome indications must be as many blocks as the null sequence"));}

                    if genome_block_chrs.iter().any(|a| *a >= chr_names.len()) || nullbp_block_chrs.iter().any(|a| *a >= chr_names.len()) { return Err(serde::de::Error::custom("all chromosome indications must be less than the number of chromosomes"));}

                    return Ok(AllData {
                        seq,
                        null_seq,
                        data,
                        start_genome_coordinates,
                        start_nullbp_coordinates,
                        genome_block_chrs,
                        nullbp_block_chrs,
                        chr_names,
                        background,
                        min_height,
                        credibility,
                    });
                } 

            }//This ends visit_map

        } //This ends the visitor

        deserializer.deserialize_struct("AllData",&["seq", "null_seq", "data", "start_genome_coordinates", "start_nullbp_coordinates", "genome_block_chrs", "nullbp_block_chrs", "chr_names", "background", "min_height", "credibility"], AllDataVisitor)
    }
}


impl<'a> AllDataUse<'a> {

    /// This takes an `[AllData]` and creates the `[AllDataUse]` that the inference
    /// will actually use. 
    /// # Errors
    ///   If the kernel recorded in `reference_struct` is too large for its data blocks
    pub fn new(reference_struct: &'a AllData, offset: f64) -> Result<Self, AllProcessingError> {

        let background = reference_struct.background();

        let mut data = reference_struct.validated_data()?;

        data.subtact_self(offset);

        let kernel_width = background.kernel.len();

        let sequence_lengths = reference_struct.seq.block_lens();

        let kernel_variant_upheld = sequence_lengths.into_iter().all(|a| a > kernel_width);

        if !kernel_variant_upheld {
            return Err(AllProcessingError::Synchronization(BadDataSequenceSynchronization::KernelMismatch));
        }

        Ok( Self{ 
            data: data,
            null_seq: &reference_struct.null_seq,
            start_genome_coordinates: &reference_struct.start_genome_coordinates,
            start_nullbp_coordinates: &reference_struct.start_nullbp_coordinates,
            genome_block_chrs: &reference_struct.genome_block_chrs,
            nullbp_block_chrs: &reference_struct.nullbp_block_chrs,
            chr_names: &reference_struct.chr_names,
            background: reference_struct.background().clone(),
            offset: offset,
            min_height: reference_struct.min_height,
            credibility: reference_struct.credibility,
            height_dist: TruncatedLogNormal::new(LOG_HEIGHT_MEAN, LOG_HEIGHT_SD, reference_struct.min_height, MAX_HEIGHT).unwrap()
        })

    }

    //SAFETY: This must uphold the safety invariant that the kernel length
    //        is SHORTER than the number of bps in any sequence block in 
    //        the sequence data points to
    //        AND start_genome_coordinates must be the same length as the number of blocks in waveform
    //        Also, chromosome indications must be compatible
    pub(crate) unsafe fn new_unchecked_data(data: Waveform<'a>, null_seq: &'a NullSequence, start_genome_coordinates: &'a Vec<usize>, start_nullbp_coordinates: &'a Vec<usize>, genome_block_chrs: &'a Vec<usize>, nullbp_block_chrs: &'a Vec<usize>, chr_names: &'a Vec<String>, background: &'a Background, min_height: f64, credibility: f64) -> Self {
        Self{
            data: data,
            null_seq: null_seq,
            start_genome_coordinates: start_genome_coordinates,
            start_nullbp_coordinates: start_nullbp_coordinates,
            genome_block_chrs: genome_block_chrs, 
            nullbp_block_chrs: nullbp_block_chrs, 
            chr_names: chr_names,
            background: background.clone(),
            offset: 0.0,
            min_height: min_height.max(1.0),
            credibility: credibility.max(0.01), 
            height_dist: TruncatedLogNormal::new(LOG_HEIGHT_MEAN, LOG_HEIGHT_SD, min_height.max(1.0), MAX_HEIGHT).unwrap()
        }
    }

    /// This makes a copy of `self`, but with a new fragment length. 
    /// #Safety
    ///  This must uphold the safety invariant that the kernel length
    ///  is SHORTER than the number of bps in any sequence block in 
    ///  the sequence data points to. So in general, REDUCING fragment 
    ///  length is fine, but increasing it can result in UB
    pub unsafe fn with_new_fragment_length(&self, fragment_length: f64) -> Self {

        let mut new_data = self.clone();
        let backround_pars = self.background.dist.get_sd_df();
        let new_background = Background::new(backround_pars.0, backround_pars.1, fragment_length).unwrap_unchecked();
        new_data.background = new_background;

        new_data
    }


    pub fn data(&self) -> &Waveform {
        &self.data
    }

    pub fn null_seq(&self) -> &NullSequence{
        &self.null_seq
    }

    pub fn size(&self) -> usize {
        self.data.amount_data()
    }

    pub fn background_ref(&self) -> &Background {
        &self.background
    }

    pub fn unit_kernel_ref(&self, kernel_width: KernelWidth, kernel_variety: KernelVariety) -> &Kernel {
        self.background.kernel_ref(kernel_width, kernel_variety)
    }

    pub fn number_bp(&self) -> usize {
        self.data.number_bp()
    }

    pub fn zero_locs(&self) -> &Vec<usize> {
        &self.start_genome_coordinates
    }
    pub fn null_zero_locs(&self) -> &Vec<usize> {
        &self.start_nullbp_coordinates
    }
    pub fn offset(&self) -> f64 {
        self.offset
    }

    pub fn min_height(&self) -> f64 {
        self.min_height
    }

    pub fn set_min_height(&mut self, height: f64) {
        self.min_height = height.max(1.0);
    }

    pub fn credibility(&self) -> f64 {
        self.credibility
    }

    pub fn height_dist(&self) -> &TruncatedLogNormal {
        &self.height_dist
    }

    pub fn num_blocks(&self) -> usize {
        self.data.point_lens().len()
    }

    pub fn propensity_minmer(&self, minmer: u64) -> f64 {
        if self.data.seq().id_of_u64_kmer(MIN_BASE, minmer).is_some() {
            (self.data.seq().number_unique_kmers(MIN_BASE) as f64).recip()
        } else {
            0.0
        }
    }

    pub fn legal_coordinate_ranges(&self) -> Vec<(usize, usize)> {

        let mut ranges: Vec<(usize, usize)> = self.start_genome_coordinates.iter().zip(self.data.seq().block_lens()).map(|(&i, j)| (i, i+j)).collect();
        let mut null_r: Vec<(usize, usize)> = self.start_nullbp_coordinates.iter().zip(self.null_seq.block_lens()).map(|(&i, j)| (i, i+j)).collect();

        ranges.append(&mut null_r);
        ranges.sort_unstable_by(|a,b| a.cmp(b));
        ranges
    }

    pub fn rand_minmer_by_propensity<R: Rng + ?Sized>(&self, rng: &mut R) -> u64 {

        //This returns None only if there are NO MIN_BASE-mers that are 
        //legal in the sequence associated with this data struct. This is
        //so deeply pathological that we would want to abort anyway. 
        *self.data.seq().unique_kmers_ref(MIN_BASE).choose(rng).unwrap()
    }

    /// This takes each block of data, sums the positive values, and returns
    /// an array of those values. In general, the larger the value, the more
    /// "peaky" a data block likely is. These absolute numbers are pointless:
    /// they only matter in context with each other, usually by ordering them.
    pub fn return_estimated_block_peakiness(&self) -> Vec<f64> {
        self.data.return_estimated_block_peakiness()
    }

    pub(crate) fn basic_peak(&self, min_height_sens: f64) -> Vec<((usize, usize), f64)> {

        let num_forward = self.background.bp_span()/self.data.spacer();

        let spacing_float = self.data.spacer() as f64;

        let kern_sd = self.background.kernel_sd();

        let target_integral = min_height_sens*kern_sd*(2.0*PI).sqrt();

        let locs_and_data = self.data.generate_all_indexed_locs_and_data(&self.start_genome_coordinates).expect("It is an invariant that the start_genome_coordinates vector length matches the data number of blocks");

        let mut location_vs_integral: Vec<((usize, usize), f64)> = Vec::with_capacity(locs_and_data.len());

        for block in locs_and_data {

            let integrals = block.1.windows(num_forward).map(|a| a.iter().sum::<f64>()*spacing_float).collect::<Vec<_>>();

            /*for (i, integral_slice) in integrals.windows(3).enumerate() {
              if (integral_slice[1] >= target_integral) && (integral_slice[1] >= integral_slice[0]) && (integral_slice[1] >= integral_slice[2]) { 
              location_vs_integral.push(((block.0[i+1], block.0[i+1+num_forward]), integral_slice[1]));
              }
              }*/
            for (i, integral) in integrals.iter().enumerate() {

                if *integral >= target_integral {
                    if let Some(last_entry) = location_vs_integral.last_mut() {
                        if last_entry.0.1 >= block.0[i] { 
                            last_entry.0.1 = block.0[(i+num_forward).min(block.0.len()-1)];
                            if last_entry.1 < *integral { last_entry.1 = *integral;}
                        } else {
                            location_vs_integral.push(((block.0[i], block.0[(i+num_forward).min(block.0.len()-1)]), *integral));
                        }
                    } else {
                        location_vs_integral.push(((block.0[i], block.0[(i+num_forward).min(block.0.len()-1)]), *integral));
                    }
                }
            }
        }

        location_vs_integral
    }

    pub(crate) fn basic_peak_output(&self, min_height_sens: f64, bed_name: &str, chromosome_name: Option<&str>) -> Result<(), Box<dyn Error>> {

        let mut location_vs_integral = self.basic_peak(min_height_sens);

        let chr = chromosome_name.unwrap_or("chr");

        location_vs_integral.sort_by(|(_,f), (_,g)| g.partial_cmp(f).unwrap());

        let max_integral = location_vs_integral[0].1;

        for i in 0..location_vs_integral.len() { location_vs_integral[i].1 = (location_vs_integral[i].1/max_integral)*1000.;} 

        let mut file = fs::File::create(bed_name)?;

        let mut buffer: Vec<u8> = Vec::with_capacity(location_vs_integral.len()*(chr.len()+32));

        for (id, ((base, end_base), score)) in location_vs_integral.iter().enumerate() {

            let thousand_score = *score as usize;
            if thousand_score == 0 { break;}
            let line = format!("{chr}\t{base}\t{end_base}\tpeak_{id}\t{thousand_score}\n");
            let mut buffer_line = line.into_bytes();
            buffer.append(&mut buffer_line);
        }

        file.write(&buffer)?;

        Ok(())

    }

    /// This saves the occupancy trace of just the data to a set of traces 
    /// `{signal_directory}/{signal_name}/from_{begin}_to_{end}.png`. 
    pub fn save_data_to_directory(&self, signal_directory: &str, signal_name: &str) -> Result<(), Box<dyn Error+Send+Sync>> {

        let zero_locs = self.zero_locs();

        let block_lens = self.data().seq().block_lens();

        let blocked_locs_and_data = self.data().generate_all_indexed_locs_and_data(self.zero_locs()).expect("Our data BETTER correspond to data_ref");

        let total_dir = format!("{}/{}", signal_directory,signal_name);

        /*if let Err(creation) = std::fs::create_dir_all(&total_dir) {
          warn!("Could not make or find directory \"{}\"! \n{}", total_dir, creation);
          println!("Could not make or find directory \"{}\"! \n{}", total_dir, creation);
          return;
          };*/

        std::fs::create_dir_all(&total_dir)?;

        let derived_color = DerivedColorMap::new(&[WHITE, ORANGE, RED]);


        for i in 0..blocked_locs_and_data.len() {

            let loc_block = &blocked_locs_and_data[i].0;
            let dat_block = &blocked_locs_and_data[i].1;

            let min_data_o = dat_block.iter().min_by(|a,b| a.partial_cmp(b).unwrap()).expect("Waves have elements");

            let min = *min_data_o-1.0;

            let max_data_o = dat_block.iter().max_by(|a,b| a.partial_cmp(b).unwrap()).expect("Waves have elements");

            let max = *max_data_o+1.0;

            let signal_file = format!("{}/from_{:011}_to_{:011}.png", total_dir, zero_locs[i], zero_locs[i]+block_lens[i]);

            let plot = BitMapBackend::new(&signal_file, (3300, 1500)).into_drawing_area();

            plot.fill(&WHITE).unwrap();

            let (left, right) = plot.split_horizontally((95).percent_width());

            let (right_space, _) = right.split_vertically((95).percent_height());

            let mut bar = ChartBuilder::on(&right_space).margin(10).set_label_area_size(LabelAreaPosition::Right, 100).caption("Deviance", ("sans-serif", 50)).build_cartesian_2d(0_f64..1_f64, 0_f64..1_f64).unwrap();

            bar.configure_mesh()
                .y_label_style(("sans-serif", 40))
                .disable_mesh().draw().unwrap();

            let deviances = (0..10000_usize).map(|x| (x as f64)/10000.0).collect::<Vec<_>>();

            bar.draw_series(deviances.windows(2).map(|x| Rectangle::new([( 0.0, x[0]), (1.0, x[1])], derived_color.get_color(x[0]).filled()))).unwrap();

            let (upper, lower) = left.split_vertically((86).percent_height());

            let mut chart = ChartBuilder::on(&upper)
                .set_label_area_size(LabelAreaPosition::Left, 100)
                .set_label_area_size(LabelAreaPosition::Bottom, 100)
                .caption("Signal Comparison", ("Times New Roman", 80))
                .build_cartesian_2d((loc_block[0] as f64)..(*loc_block.last().unwrap() as f64), min..max).unwrap();

            chart.configure_mesh()
                .x_label_style(("sans-serif", 40))
                .y_label_style(("sans-serif", 40))
                .x_label_formatter(&|v| format!("{:.0}", v))
                .x_desc("Genome Location (Bp)")
                .y_desc("Signal Intensity")
                .disable_mesh().draw().unwrap();

            const HORIZ_OFFSET: i32 = -5;

            chart.draw_series(dat_block.iter().zip(loc_block.iter()).map(|(&k, &i)| Circle::new((i as f64, k),2_u32, Into::<ShapeStyle>::into(&BLACK).filled()))).unwrap();//.label("True Occupancy Data").legend(|(x,y)| Circle::new((x+2*HORIZ_OFFSET,y),5_u32, Into::<ShapeStyle>::into(&BLACK).filled()));


            //chart.configure_series_labels().position(SeriesLabelPosition::LowerRight).margin(40).legend_area_size(10).border_style(&BLACK).label_font(("Calibri", 40)).draw().unwrap();

            let abs_resid: Vec<(f64, f64)> = dat_block.iter().map(|&a| {

                let tup = self.background_ref().cd_and_sf(a);
                if tup.0 >= tup.1 { (tup.0-0.5)*2.0 } else {(tup.1-0.5)*2.0} } ).zip(loc_block.iter()).map(|(a, &b)| (a, b as f64)).collect();

            let mut map = ChartBuilder::on(&lower)
                .set_label_area_size(LabelAreaPosition::Left, 100)
                .set_label_area_size(LabelAreaPosition::Bottom, 50)
                .build_cartesian_2d((loc_block[0] as f64)..(*loc_block.last().unwrap() as f64), 0_f64..1_f64).unwrap();

            map.configure_mesh().x_label_style(("sans-serif", 0)).y_label_style(("sans-serif", 0)).x_desc("Deviance").axis_desc_style(("sans-serif", 40)).set_all_tick_mark_size(0_u32).disable_mesh().draw().unwrap();


            map.draw_series(abs_resid.windows(2).map(|x| Rectangle::new([(x[0].1, 0.0), (x[1].1, 1.0)], derived_color.get_color(x[0].0).filled()))).unwrap();

        }

        Ok(())

    }

    /// This creates an `[AllData]` from `self` which omits the blocks indexed by `remove`. 
    /// We ignore any duplicated elements of `remove` as well as any elements 
    /// that are at least the number of blocks in `self`. Returns `Ok(None)` if `remove` 
    /// contains all blocks. The AllData that comes out retains all data about the null sequence. 
    pub fn with_removed_blocks(&self, remove: &[usize]) -> Option<AllData> {

        let new_seq = self.data.seq().with_removed_blocks(remove)?;
        let new_wave = self.data.with_removed_blocks(remove)?;

        let mut new_coords = self.start_genome_coordinates.clone();

        let mut new_chrs = self.genome_block_chrs.clone();

        let mut remove_descend: Vec<usize> = remove.to_vec();

        // We always want to remove blocks in descending order
        // Otherwise, previously remove blocks screw with the remaining blocks
        remove_descend.sort_unstable();
        remove_descend.reverse();
        remove_descend.dedup();

        remove_descend = remove_descend.into_iter().filter(|a| *a < self.start_genome_coordinates.len()).collect();

        //This is only possible if we have all blocks listed in remove_descend,
        //thanks to sorting and dedup().
        if remove_descend.len() == self.start_genome_coordinates.len() { return None;}

        for ind in remove_descend {
            new_coords.remove(ind);
            new_chrs.remove(ind);
        }

        let to_return = AllData{

            seq: new_seq,
            null_seq: self.null_seq.clone(), 
            data: new_wave,
            start_genome_coordinates: new_coords,
            start_nullbp_coordinates: self.start_nullbp_coordinates.clone(),
            genome_block_chrs: new_chrs,
            nullbp_block_chrs: self.nullbp_block_chrs.clone(),
            chr_names: self.chr_names.clone(),
            background: self.background.clone(), 
            min_height: self.min_height,
            credibility: self.credibility,
        };

        _ = AllDataUse::new(&to_return, 0.0).expect("AllData should always produce a legal AllDataUse");

        Some(to_return)

    }

    pub fn return_kmer_by_loc(&self, coordinate_start: usize, size_kmer: usize) -> Option<Vec<Bp>> {

        let block_in_pos = self.start_genome_coordinates.iter().enumerate().find(|&(i,start)| coordinate_start >= *start && (coordinate_start-*start)+size_kmer<= self.data().seq().ith_block_len(i));

        match block_in_pos {
            Some(block_id) => Some(self.data().seq().return_bases(block_id.0, coordinate_start-block_id.1, size_kmer)),
            None => self.start_nullbp_coordinates.iter().enumerate()
                .find(|&(i,start)| coordinate_start >= *start && (coordinate_start-*start)+size_kmer<= self.null_seq().ith_block_len(i))
                .map(|a| self.null_seq().return_bases(a.0, coordinate_start-a.1, size_kmer)),
        }
    }

}

/// This documents all the things that can go wrong from a processing a Fasta File
#[derive(Error, Debug, Clone)]
pub enum FastaError {    
    #[error(transparent)]
    InvalidFile(#[from] Arc<std::io::Error>),
    #[error(transparent)]
    BlankFile(#[from] EmptyFastaError),
    #[error(transparent)]
    BadFastaStart(#[from] MaybeNotFastaError),
    #[error(transparent)]
    BadFastaInput(#[from] InvalidBasesError),
    #[error(transparent)]
    RedundantChromosome(#[from] ChromosomeError),
}


#[derive(Error, Debug, Clone)]
struct EmptyFastaError;

impl std::fmt::Display for EmptyFastaError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {write!(f, "FASTA file cannot be empty or only whitespace, and must have at least one line with valid base pairs in it!")}
}

#[derive(Error, Debug, Clone)]
struct MaybeNotFastaError;

impl std::fmt::Display for MaybeNotFastaError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {write!(f, "FASTA file first line must start with a '>'! Are you certain this is a FASTA file?") }
}

#[derive(Error, Debug, Clone)]
struct InvalidBasesError {
    bad_base_locations: Vec<(u64, u64)>, 
    too_many_bad_bases: bool,
}

#[derive(Error, Debug)]
struct SequenceWaveformIncompatible;

impl std::fmt::Display for SequenceWaveformIncompatible {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {write!(f, "The sequence and the data you are using are incompatible!") }
}


impl InvalidBasesError {

    fn new(line: u64, pos: u64) -> Self {
        Self {
            bad_base_locations: vec![(line, pos)],
            too_many_bad_bases: false,
        }
    }

    fn add_invalid_if_possible(&mut self, line: u64, pos: u64) {

        if self.too_many_bad_bases == true {return;}

        if let Err(tuple) = improvised_push_within_capacity(&mut self.bad_base_locations, (line, pos)) {
            let check = self.bad_base_locations.try_reserve(1);
            match check {
                Ok(()) => {improvised_push_within_capacity(&mut self.bad_base_locations, tuple).expect("Only runs if reserve successful!");}
                Err(_) => {self.too_many_bad_bases = true;}
            }
        }

    }


}

fn improvised_push_within_capacity<T>(vector: &mut Vec<T>, element: T) -> Result<(), T> {

    if vector.len() == vector.capacity() { return Err(element); }

    vector.push(element);
    Ok(())

}

impl std::fmt::Display for InvalidBasesError{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.too_many_bad_bases { write!(f, "Too many bad bases for this architecture to track\n")?;}
        if self.bad_base_locations.len() == 0 { write!(f, "This error itself was somehow generated in error")?; }
        for &(line, position) in self.bad_base_locations.iter() {
            write!(f, "Invalid base on line {}, position {}\n", line+1, position+1)?;
        }

        Ok(())
    }
}

#[derive(Error, Debug, Clone)]
struct ChromosomeError {
    redundant_chromosome_name: String,
    also_bad_base: Option<InvalidBasesError>,
}

impl ChromosomeError {

    fn new(bad_chromosome: String, also_bad_base: Option<InvalidBasesError>) -> Self {

        Self  {
            redundant_chromosome_name: bad_chromosome, 
            also_bad_base: also_bad_base,
        }

    }
}

impl std::fmt::Display for ChromosomeError {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Redundant chromosome name: \"{}\"! Chromosome names cannot repeat because of how we coordinate the bedgraph with the FASTA!\nNOTE: You might also have bad base pairs in parts of the FASTA file beyond this chromosome: this is a lethal point of failure and we don't do further checks after this point.", self.redundant_chromosome_name)?;
        if let Some(bad) = self.also_bad_base.as_ref() { bad.fmt(f)?; };
        Ok(())
    }

}

/// This documents all the things that can go wrong from processing a data file
#[derive(Error, Debug)]
pub enum DataProcessError {
    #[error(transparent)]
    InvalidFile(#[from] Arc<std::io::Error>),
    #[error(transparent)]
    BlankFile(#[from] EmptyDataError),
    #[error(transparent)]
    BadSpacer(#[from] BadSpacerError),
    #[error(transparent)]
    BadDataFormat(#[from] DataFileBadFormat),
    #[error(transparent)]
    NumBindingSitesTooLarge(#[from] TooManyBindingSites), 
}

impl DataProcessError {

    fn collect_wrongdoing<T>(self, possible_sequence_err: Result<T, FastaError>) -> AllProcessingError {
        match possible_sequence_err {
            Ok(_) => AllProcessingError::DataOnly(self),
            Err(e) => AllProcessingError::DataAndSequence((self, e))
        }
    }
}

#[derive(Error, Debug)]
struct BadSpacerError;
impl std::fmt::Display for BadSpacerError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Spacer must be positive!")}
}

#[derive(Error, Debug)]
struct EmptyDataError;
impl std::fmt::Display for EmptyDataError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Data file should not be empty!")}
}

#[derive(Error, Debug)]
struct DataFileBadFormat {
    bad_header: bool, 
    empty_lines: Vec<u64>,
    bad_chromosome_lines: Vec<u64>,
    no_location_lines: Vec<u64>, 
    no_data_lines: Vec<u64>,
    location_bad_lines: Vec<u64>,
}

impl DataFileBadFormat {

    fn new(bad_header: bool) -> Self {
        Self {
            bad_header: bad_header, 
            empty_lines: Vec::new(),
            bad_chromosome_lines: Vec::new(),
            no_location_lines: Vec::new(),
            no_data_lines: Vec::new(),
            location_bad_lines: Vec::new(),
        }
    }
}

#[derive(Error, Debug)]
struct TooManyBindingSites {
    num_binding_sites: usize, 
    data_length: usize,
}
impl TooManyBindingSites {
    fn new(num_binding_sites: usize, data_length: usize) -> Self{ 

        Self {
            num_binding_sites,
            data_length
        }

    }
}

impl std::fmt::Display for TooManyBindingSites {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "You approximated {} binding sites when you have to have less than {} binding sites.", self.num_binding_sites, self.data_length)}
}

trait UpgradeToErr {
    fn add_problem_line(&mut self, line: u64, empty: bool, bad_chromosome: bool, locationless: bool, dataless: bool, too_long: bool);
}

impl UpgradeToErr for Option<DataFileBadFormat> {

    fn add_problem_line(&mut self, line: u64, empty: bool, bad_chromosome: bool, locationless: bool, dataless: bool, too_long: bool) { 

        if self.is_none() { *self = Some(DataFileBadFormat::new(false));}
        let inside = self.as_mut().expect("already returned if this was null");
        if empty { inside.empty_lines.push(line);} 
        if bad_chromosome { inside.bad_chromosome_lines.push(line);} 
        if locationless { inside.no_location_lines.push(line); }
        if dataless { inside.no_data_lines.push(line); }
        if too_long { inside.location_bad_lines.push(line);}
    }
}

impl std::fmt::Display for DataFileBadFormat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let mut header_tip = false;
        write!(f, "Your data is formatted poorly:\n")?;
        if self.bad_header { write!(f, "Your data header is incorrect: it must start 'loc data'\n")?;}
        if self.empty_lines.len() > 0 {
            header_tip |= self.empty_lines[0] ==0;
            write!(f, "You have empty or unreadable lines. Recall that a data line in bedgraph file is tab delimited, with four entries: a chromosome name, a start location (that we use), an ending location (which we do not), and a data point (which we use as the occupancy trace).\nThe problems can be found at the following line numbers, using vi numbering: {:?}\n", self.empty_lines.iter().map(|a| a+1).collect::<Vec<_>>())?;
        }
        if self.bad_chromosome_lines.len() > 0 {
            header_tip |= self.bad_chromosome_lines[0] ==0;
            write!(f, "You have bad chromosome names. This results if: \n- Your FASTA file has multiple chromosomes and the chromosome names in your bedgraph don't match the names in your FASTA file, OR \n- Your FASTA file has a single chromosome, and the bedgraph has multiple chromosome names. Chromosome names in the FASTA file are sought in the lines that immediately follow a '>'. If the string 'chromosome:' can be found in that line, the chromosome name is taken as everything that follows that to the end of the line, trimming whitespace on either end. Otherwise, the whole line after '>' is taken (also trimming whitespace). \nThe problems can be found at the following line numbers, using vi numbering: {:?}\n", self.bad_chromosome_lines.iter().map(|a| a+1).collect::<Vec<_>>())?;
        }
        if self.no_location_lines.len() > 0 {
            header_tip |= self.no_location_lines[0] ==0; 
            write!(f, "You have lines with invalid location values (a nonnegative integer). \nThe problems can be found at the following line numbers, using vi numbering: {:?}\n", self.no_location_lines.iter().map(|a| a+1).collect::<Vec<_>>())?;
        }

        if self.no_data_lines.len() > 0 {
            header_tip |= self.no_data_lines[0] ==0;
            write!(f, "You have lines that do not have valid data after the location: either that data doesn't match to a 64 bit float, or matches to a NaN.  \nThe problems can be found at the following line numbers, using vi numbering: {:?}\n", self.no_data_lines.iter().map(|a| a+1).collect::<Vec<_>>())?;
        }

        if self.location_bad_lines.len() > 0 {
            write!(f, "You have lines whose locations are too large for the chromosomes they purport to come from. We assume 0 indexing on the chromosome position. Also note that we only validate the starting location. \nThe problems can be found at the following line numbers, using vi numbering: {:?}\n", self.location_bad_lines.iter().map(|a| a+1).collect::<Vec<_>>())?;
        }

        if header_tip {
            write!(f, "Your first error is at the beginning of the file. A reminder that we do not process headers for this algorithm. For us, a suitable bedgraph file is a tab delimited, four column file where each row is the chromosome name, the locations, and then finally, the data. Furthermore, if the fasta file being used has multiple chromosomes, then the names from that fasta file must match the names used in the bedgraph file.")?;
        }

        Ok(())
    }
}

/// This is the master error type. It documents all the things that can go wrong 
/// when you try to start up an inference. 
#[derive(Error, Debug)]
pub enum AllProcessingError {

    DataOnly(DataProcessError),
    SequenceOnly(FastaError),
    DataAndSequence((DataProcessError, FastaError)),
    Synchronization(BadDataSequenceSynchronization),
    WayTooBig,
}

impl std::fmt::Display for AllProcessingError {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AllProcessingError::DataOnly(d) => write!(f, "Only found data errors:\n {}", d),
            AllProcessingError::SequenceOnly(s) => write!(f, "Only found sequence errors: \n {}", s),
            AllProcessingError::DataAndSequence((d, s)) => write!(f, "Found both data and sequencing errors: \n Sequence errors:\n \t{}\n Data errors:\n \t{}", s, d),
            AllProcessingError::Synchronization(y) => write!(f, "Data and sequence both fine, but they are not compatible:\n {}", y),
            AllProcessingError::WayTooBig => {

                if cfg!(target_pointer_width = "32") {
                    write!(f, "The combination of sequence and data ends up too large to store,
                           which means things will almost certainly break if we try to use them. 
                           You should probably increase the threshold at which you consider something peak-y,
                           or run this on a 64 bit platform.")
                } else if cfg!(target_pointer_width = "64") {
                    write!(f, "The combination of sequence and data ends up too large to store,
                               which means things will almost certainly break if we try to use them.
                               You're getting this error on a 64 bit platform, which means either you
                               combined way too many genomes and IP data sets together into one--which
                               literally could never work, this motif finder is designed to work on single 
                               genomes with single IP sets corresponding to it--or you're using a genome 
                               that's literally almost 10 million times bigger than Paris japonica")
                } else if cfg!(target_pointer_width = "128") {

                    write!(f, "The combination of sequence and data ends up too large to store,
                               which means things will almost certainly break if we try to use them.
                               You're getting this error on a 128 bit platform. HOW are you working 
                               with data files that big? And WHO got the Nobel Prize for discovering 
                               a genome literally octillions of times bigger than Paris japonica?")
                } else {
                    write!(f, "The combination of sequence and data ends up too large to store, 
                               which means things will almost certainly break if we try to use them.
                               You should probably increase the threshold at which you consider something peak-y, 
                               but I can't help you more than that, because this platform is apparently
                               exotically sized, which I don't think std Rust supports. Or maybe your
                               platform is 256 bits or more, in which case: Hi! You DEFINITELY should not
                               have genome sizes large enough to get this error! I am not even going to 
                               compare it to the largest genome I'm aware of (Paris japonica), because
                               genomes big enough to break the motif finder like this MUST violate some
                               kind of physical constraint or something!")
                }
            },
        }
    }

}

#[derive(Error, Debug)]
pub enum BadDataSequenceSynchronization {
    SequenceMismatch,
    NoChromosomes,
    KernelMismatch,
    NotEnoughNullData, 
    NotEnoughPeakData, 
    PoorSynchronize(WaveCreationError),
    NeedDifferentExperiment, //This is NotEnoughNullData AND NotEnoughPeakData. You can't screw with peak_thresh to make this better: you're basically screwed and need a new set
}

impl std::fmt::Display for BadDataSequenceSynchronization {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {

        match self {

            BadDataSequenceSynchronization::SequenceMismatch => write!(f, "You have data in locations too large for your sequence!"),
            BadDataSequenceSynchronization::NoChromosomes => write!(f, "You somehow managed to skirt a bunch of checks around making sure you have chromosomes, and reached the point where you simply do not despite all this. I do not have a quick fix for this. As far as I know, it should be impossible to trip this error without tripping others which would supercede it. Yet, here we are. I sincerely hope nobody in the world ever sees this error the hard way. Yet, I worry that such a thing is inevitable."),
            BadDataSequenceSynchronization::KernelMismatch => write!(f, "You have data blocks that are too small for your fragment length!"),
            BadDataSequenceSynchronization::NotEnoughNullData => write!(f, "Not enough null data to infer the background distribution! Make peak_thresh bigger to allow more null data!"),
            BadDataSequenceSynchronization::NotEnoughPeakData => write!(f, "You do not have enough peak-like data that is within ungapped valid base pairs! Make peak_thresh smaller to allow more data to count as data!"),
            Self::PoorSynchronize(w) => write!(f, "{}", format!("Your occupancy trace creation had an error: {}", w)),
            BadDataSequenceSynchronization::NeedDifferentExperiment => write!(f, "You neither have enough null data nor enough peak data! This data can't have inference done on it! Supply a different data set!"),
        }?;

        Ok(())
    }


}


#[cfg(test)]
mod tests{

    use super::*;

    #[test]
    fn test_norm_fit() {

        let quantiles = [0.1_f64, 0.2, 0.25, 0.5, 0.6, 0.75, 0.9];
        let norm_data: Vec<f64> = vec![0.53,-0.07,-0.06,0.45,-0.15,-0.13,-0.4,1.69,-0.72,0.0,-0.06,-0.23,-0.59,-0.5,-0.13,-0.25,0.58,-0.17,0.47,0.3,-0.03,0.25,-0.47,0.0,-0.28,0.11,0.02,0.55,0.08,-0.26,0.05,-0.1,0.16,-0.06,-0.09,0.12,-0.16,0.32,-0.01,0.04,0.2,1.85,0.21,0.07,0.16,-0.35,-0.19,0.32,0.09,0.38,-0.24,0.06,-0.02,0.04,0.2,-0.28,-0.08,-0.27,0.36,-0.18,0.05,-0.04,-0.41,-0.25,0.08,-0.12,0.6,1.26,0.31,-0.09,-0.26,0.42,0.46,-0.14,-0.11,-0.07,0.1,0.08,-0.17,0.49,0.79,-0.03,0.47,-0.34,1.67,0.42,0.2,0.33,0.08,-0.26,-0.13,0.12,-0.32,0.05,0.54,-0.11,-0.09,0.15,-0.03,0.33,0.1,-0.17,-0.42,-0.13,0.02,0.0,-0.39,-0.21,-0.49,0.07,-0.01,-0.52,0.23,-0.36,-0.17,0.06,-0.04,0.24,0.34,-0.25,-0.11,0.07,-0.15,0.97,-0.23,-0.01,-0.08,0.32,-0.27,0.19,-0.16,0.03,-0.49,0.38,1.83,0.5,-0.23,-0.03,0.25,0.52,-0.11,0.2,0.1,0.62,0.95,-0.29,0.18,-0.51,-0.1,-0.37,-0.14,2.02,-0.51,0.48,1.35,-0.24,-0.36,-0.02,0.05,1.19,0.15,-0.36,0.08,0.12,-0.26,-0.06,0.23,0.03,0.28,-0.25,0.12,-0.01,0.0,0.22,0.02,0.0,0.04,-0.01,-0.13,0.18,-0.6,-0.26,-0.5,0.27,-0.06,-0.08,0.04,-0.09,-0.52,0.28,0.09,-0.05,-0.25,-0.13,-0.25,0.56,0.3,0.66,-0.01,0.17,0.73,-0.19,0.36,-0.31,-0.13,-0.05,-0.23,-0.03,-0.25,-0.39,-0.2,-0.1,-0.14,0.0,-0.21,0.16,-0.69,-0.37,0.39,0.18,-0.07,0.4,-0.09,0.31,-0.14,-0.71,2.23,0.17,-0.18,-0.18,1.07,0.23,-0.43,0.26,0.01,-0.19,-0.03,0.26,0.15,-0.39,-0.16,-0.5,0.02,-0.18,-0.33,-0.12,0.36,0.21,-0.29,0.08,-0.32,0.19,-0.39,0.0,-0.3,0.14,-0.55,-0.43,0.38,-0.17,-0.23,-0.24,0.2,0.87,-0.97,-0.14,-0.37,0.0,0.69,-0.22,-0.02,-0.13,-0.3,-0.23,0.88,-0.2,-0.08,-0.26,-0.19,-0.04,-0.21,0.08,0.87,0.05,-0.39,0.0,-0.24,0.34,-0.4,0.43,0.08,0.3,-0.16,-0.06,0.13,-0.23,0.02,0.67,0.25,-0.13,-0.3,-0.38,-0.4,0.31,0.12,-0.61,0.24,0.32,0.03,0.2,-0.25,-0.25,-0.18,0.01,-0.26,0.49,-0.07,-0.47,-0.41,0.06,-0.24,0.28,-0.07,-0.1,0.57,0.42,1.13,-0.26,1.01,-0.63,-0.1,-0.06,-0.54,-0.26,0.4,-0.52,0.25,-0.31,-0.01,-0.39,0.22,0.16,0.04,-0.11,0.37,-0.27,-0.32,-0.29,0.43,0.48,-0.29,0.65,-0.39,-0.26,0.37,-0.03,-0.33,0.47,0.07,-0.1,0.69,-0.18,0.04,-0.1,-0.41,-0.1,0.13,0.41,0.37,0.3,-0.24,1.7,-0.28,-0.42,-0.15,-0.74,0.35,0.58,-0.11,-0.07,0.15,-0.13,0.89,-0.27,-0.17,0.33,0.07,0.34,-0.24,0.31,0.43,-0.11,0.07,-0.22,0.12,0.42,0.52,0.18,0.46,-0.41,0.16,-0.25,-0.22,0.17,-0.05,0.09,0.21,0.01,0.12,0.23,0.02,0.1,-0.1,0.09,0.96,-0.13,-0.22,0.06,0.12,-0.09,-0.1,0.48,0.22,0.57,0.0,0.06,-0.34,-0.03,0.22,0.21,0.58,0.01,-0.12,-0.49,-0.06,-0.25,0.11,-0.05,0.14,0.04,0.54,-0.44,-0.16,-0.19,-0.56,0.54,0.3,-0.21,0.17,-0.3,-0.27,-0.44,-0.08,-0.46,0.45,0.51,-0.23,-0.21,-0.11,-0.38,-0.23,-0.09,0.01,-0.48,-0.08,0.98,0.11,0.24,0.34,0.16,-0.56,0.3,-0.26,0.05,-0.3,-0.21,0.2,0.41,0.01,0.37,0.05,-0.58,0.14,-0.33,-0.12,-0.23,-0.13,0.16,-0.04,-0.33,-0.18,-0.27,0.15,-0.16,-0.48,-0.11,0.37,0.04,0.29,-0.24,-0.16,0.7,0.81,0.06,0.02,0.25,0.01,-0.25,0.55,0.02,0.39,1.02,-0.07,-0.32,-0.32,-0.49,0.14,0.26,0.07,1.56,0.3,0.02,0.09,-0.1,0.06,-0.65,-0.4,1.72,-0.13,-0.06,-0.13,0.1,0.53,-0.13,0.85,0.51,1.4,1.4,1.15,0.05,0.24,0.06,0.78,-0.17,-0.38,-0.16,0.09,-0.31,-0.26,-0.15,0.1,-0.47,-0.11,-0.02,1.38,0.17,0.52,0.17,0.44,-0.01,0.1,-0.14,0.31,-0.11,0.04,-0.16,0.4,0.2,0.02,0.22,0.07,-0.17,0.2,-0.42,-0.17,0.47,0.31,-0.4,-0.32,0.16,-0.27,0.77,-0.02,0.09,0.22,0.06,-0.24,0.12,0.01,-0.38,-0.07,0.07,-0.04,-0.38,0.08,-0.28,1.31,0.19,-0.28,-0.25,-0.06,0.05,-0.33,1.5,-0.11,-0.03,0.27,0.21,-0.39,0.09,-0.13,0.14,-0.21,-0.34,0.0,-0.11,-0.12,-0.38,-0.27,-0.18,0.0,-0.25,-0.37,-0.09,-0.24,-0.03,-0.33,0.61,0.06,-0.35,-0.25,-0.2,-0.21,0.1,-0.04,-0.03,0.2,-0.07,-0.19,-0.23,-0.06,0.07,0.01,-0.68,0.46,0.02,-0.36,-0.15,-0.31,-0.36,0.0,0.4,0.36,0.41,0.71,0.05,0.21,-0.26,-0.02,0.77,-0.03,-0.85,-0.48,-0.33,0.11,-0.17,-0.12,-0.39,-0.07,-0.11,0.14,-0.06,-0.14,0.16,-0.05,0.59,0.09,-0.12,0.21,0.59,0.18,0.07,-0.62,-0.41,0.59,0.78,-0.05,0.21,-0.02,-0.11,-0.14,0.99,0.11,-0.41,-0.03,-0.03,-0.44,-0.73,-0.15,0.12,-0.42,1.26,-0.25,0.04,0.76,0.23,0.22,-0.01,-0.45,0.51,0.23,0.0,0.39,-0.23,0.94,-0.27,-0.21,-0.15,-0.29,0.11,-0.14,-0.12,-0.37,-0.2,0.05,-0.32,0.17,0.05,0.23,0.47,0.04,0.66,0.14,0.31,0.03,-0.02,0.85,0.44,-0.14,0.15,1.13,0.35,-0.02,-0.45,0.87,0.13,0.1,-0.27,0.28,-0.05,-0.06,-0.33,0.3,-0.12,-0.36,0.15,-0.3,0.0,0.25,-0.06,-0.16,-0.59,0.08,-0.09,0.2,0.12,-0.04,-0.76,0.28,-0.18,0.63,-0.11,-0.08,-0.16,0.01,-0.26,-0.13,0.78,0.01,-0.08,-0.36,0.27,-0.61,0.19,0.78,1.2,-0.04,0.08,-0.02,1.3,0.16,0.32,-0.04,0.2,0.38,0.11,0.06,-0.45,-0.16,-0.1,-0.34,-0.07,-0.2,-0.35,0.82,-0.43,0.37,0.03,-0.05,0.22,0.72,0.94,0.51,0.37,-0.06,-0.12,-0.29,0.17,0.18,-0.12,-0.26,0.04,-0.23,-0.29,0.32,-0.3,-0.04,1.02,-0.11,0.05,0.1,-0.08,0.74,-0.08,0.53,0.47,-0.27,-0.11,0.18,-0.36,0.25,0.11,-0.17,1.66,-0.01,0.12,0.86,0.01,-0.19,-0.23,-0.25,0.18,0.24,-0.32,0.19,0.28,-0.12,0.17,0.26,-0.39,-0.23,0.32,-0.11,-0.26,0.82,0.19,0.14,0.38,-0.01,0.29,-0.28,-0.21,-0.03,0.07,-0.26,0.29,0.77,-0.1,0.52,0.26,0.09,0.76,-0.38,-0.01,-0.04,-0.14,-0.35,-0.07,0.17,-0.32,-0.1,-0.63,-0.06,-0.16,0.09,0.06,-0.04,0.36,-0.3,0.03,0.52,0.57,0.91,0.08,0.11,-0.24,-0.01,-0.02,0.53,-0.44,0.58,0.49,-0.02,0.57,0.73,-0.28,-0.3,-0.12,-0.14,-0.47,-0.1,0.08,0.32,0.92,0.09,0.1,0.48,0.21,-0.07,-0.1,0.02,0.09,-0.31,0.0,0.03,-0.07,-0.05,0.71,0.08,-0.03,-0.32,-0.29,-0.11,0.26,-0.06,-0.19,-0.41,-0.24,0.07,-0.28,-0.5,-0.12,-0.1,0.01,0.17,-0.35,0.37,-0.25,0.21,0.56,-0.34,-0.17,0.39,0.56,-0.26,0.58,-0.27,-0.39,0.91,-0.51,-0.15,-0.08,0.59,-0.02,-0.32,-0.02,0.2,-0.24,-0.17,0.62,-0.36,0.15,-0.55,0.22,-0.29,-0.26,0.03,0.7,0.14,-0.17,-0.12,-0.31,0.39,0.44,-0.08,-0.02,-0.11,-0.45,0.05,-0.36,0.0,0.2,0.27,-0.13,0.1,-0.23,0.08,0.21,0.25,-0.03,0.12,0.13,0.29,-0.13,-0.06,1.31,-0.47,-0.12,-0.26,0.76,0.92,-0.23,0.25,0.09,-0.25,-0.12,0.02,0.14,0.24,-0.16,0.48,0.39,0.04,0.09,-0.24,0.35,-0.17,-0.1,-0.14,-0.11,0.12,0.22,0.86,1.54,-0.22,-0.41,0.15,0.07,-0.32,-0.34,-0.16,0.38,-0.41,-0.46,-0.18,0.09,-0.09,0.31,-0.29,-0.31,-0.05,0.13,0.17,1.55,-0.03,1.38,-0.08,1.07,0.44,-0.25,-0.07,0.12,0.72,0.11,0.04,0.13,0.97,-0.02,0.17,-0.25,0.07,0.11,0.2,0.06,0.14,0.14,-0.37,-0.31,0.36,0.21,0.02,0.38,0.16,-0.05,0.08,-0.18,0.18,-0.25,0.7,0.27,0.57,0.19,0.03,0.02,0.1,-0.65,-0.24,-0.42,0.92,0.33,-0.02,0.29,0.81,0.17,0.82,-0.15,0.38,0.06,0.56,-0.77,-0.18,-0.04,0.21,-0.46,-0.27,0.02,0.19,-0.48,0.07,0.85,-0.18,-0.27,-0.08,1.05,0.32,-0.02,0.15,0.14,-0.19,-0.31,0.09,-0.17,0.13,-0.15,-0.18,-0.2,-0.01,0.22,0.04,-0.2,-0.01,0.11,0.01,0.24,0.33,-0.05,0.1,0.1,-0.28,0.22,-0.03,0.29,0.16,-0.08,0.28,-0.05,-0.21,1.27,0.18,0.21,-0.32,-0.09,-0.08,-0.02,0.73,0.11,-0.45,0.21,-0.11,0.34,0.5,0.0,-0.25,0.21,-0.16,-0.09,-0.17,-0.21,-0.02,-0.01,0.15,-0.38,-0.11,0.55,0.29,0.21,0.12,0.05,0.27,0.37,0.16,0.38,0.8,1.39,0.25,-0.06,0.15,0.44,0.3,0.36,-0.23,0.18,0.0,0.16,0.11,-0.09,-0.17,0.02,1.1,0.84,-0.03,0.0,0.12,-0.25,-0.2,0.2,-0.22,-0.19,-0.18,0.56,1.28,-0.27,-0.17,-0.04,0.61,0.01,0.0,-0.56,0.1,0.34,0.14,-0.13,-0.57,0.84,0.07,-0.02,-0.38,1.31,-0.01,0.04,-0.06,-0.06,0.35,-0.01,-0.13,0.31,0.05,0.02,-0.58,0.03,-0.05,-0.2,0.07,0.31,-0.28,0.83,-0.18,0.01,0.05,-0.22,0.2,0.15,0.04,-0.12,-0.47,0.33,-0.42,0.98,-0.43,0.18,0.0,-0.32,-0.27,-0.14,0.43,0.49,-0.08,0.11,0.07,-0.07,0.04,-0.14,-0.13,-0.03,-0.11,0.08,1.3,-0.1,0.22,0.16,1.26,0.28,-0.26,0.4,0.09,0.25,-0.26,-0.08,-0.24,-0.36,-0.26,-0.18,0.35,0.02,0.06,-0.15,-0.21,0.5,0.01,-0.23,0.02,-0.84,0.59,-0.23,-0.09,-0.35,0.34,0.18,0.08,0.29,0.0,-0.14,-0.04,-0.07,0.03,0.08,1.2,-0.02,-0.1,0.22,0.13,-0.13,-0.08,0.08,-0.13,-0.51,-0.14,0.16,-0.13,-0.75,0.07,-0.17,0.31,0.88,0.37,0.41,0.15,-0.36,0.05,0.19,0.22,0.15,1.07,0.02,-0.05,-0.29,0.01,-0.26,-0.07,-0.08,-0.18,-0.26,-0.32,0.08,0.09,0.2,0.08,0.38,-0.14,0.0,-0.34,-0.13,-0.06,0.19,-0.36,-0.23,-0.15,0.56,0.18,-0.63,0.19,0.14,-0.56,-0.27,0.27,0.02,-0.23,-0.09,-0.15,0.14,0.04,0.05,0.06,0.56,0.05,-0.35,0.2,0.1,1.15,-0.81,-0.29,-0.25,0.19,-0.04,-0.16,0.18,-0.17,0.04,0.04,-0.55,1.3,-0.21,0.04,-0.33,-0.16,-0.03,-1.2,0.27,0.03,-0.17,1.28,-0.09,0.15,-0.05,0.62,0.41,0.03,0.4,0.71,0.45,-0.34,0.12,0.05,0.08,-0.11,-0.51,-0.25,-0.09,0.12,-0.09,0.77,0.24,-0.28,-0.34,-0.1,-0.01,0.39,0.1,-0.24,0.75,-0.18,0.14,-0.16,0.29,-0.4,0.33,0.35,1.13,0.46,-0.42,0.18,0.66,0.05,-0.06,0.14,0.01,0.18,0.01,0.3,-0.21,0.01,-0.26,-0.06,-0.09,0.21,0.0,-0.06,-0.43,0.27,-0.29,-0.16,0.04,0.2,0.34,0.23,-0.86,-0.06,0.09,-0.48,-0.16,0.93,-0.11,-0.21,-0.17,0.75,0.35,0.63,0.42,-0.47,-0.09,-0.52,0.4,-0.21,-0.11,-0.04,-0.18,0.26,-0.19,-0.27,0.05,-0.27,0.22,-0.51,-0.3,0.07,0.02,0.0,-0.09,0.61,0.02,-0.07,0.15,0.32,-0.37,-0.43,-0.31,0.43,-0.13,0.05,1.23,-0.1,-0.35,0.19,0.37,2.19,-0.05,0.28,-0.09,0.45,-0.18,0.36,1.78,0.0,0.12,0.3,-0.47,0.09,0.13,-0.42,-0.2,-0.07,0.38,-0.25,0.38,0.19,-0.38,-0.11,-0.19,0.18,-0.12,0.05,-0.08,0.14,-0.01,-0.24,-0.25,0.33,-0.19,-0.25,1.36,-0.3,-0.35,0.1,-0.09,0.08,-0.04,0.05,-0.44,0.03,-0.32,-0.33,-0.17,-0.08,-0.18,-0.91,-0.14,0.25,-0.01,-0.03,-0.18,0.44,-0.16,0.1,0.09,0.01,0.51,0.05,-0.11,-0.11,0.01,-0.02,0.32,-0.25,0.35,-0.02,0.12,-0.12,0.12,0.32,-0.3,-0.14,0.39,0.17,-0.09,-0.32,0.1,-0.38,-0.48,-0.38,-0.16,-0.01,-0.29,1.29,0.23,0.04,-0.27,0.01,0.13,0.02,0.5,-0.44,-0.25,-0.56,0.36,0.46,0.41,0.41,0.11,-0.05,1.71,0.27,-0.18,-0.23,-0.29,0.13,0.09,-0.18,-0.1,-0.05,-0.2,-0.36,0.12,-0.08,0.34,-0.18,0.0,0.36,-0.23,-0.14,-0.08,-0.12,0.77,-0.29,-0.03,-0.12,0.1,-0.21,0.05,0.11,0.15,-0.21,0.34,0.01,-0.04,0.8,0.4,-0.22,0.28,0.35,0.3,-0.12,-0.17,-0.07,-0.08,0.05,0.39,0.13,-0.13,1.54,0.0,-0.08,-0.03,0.09,1.35,0.1,-0.28,-0.11,0.03,0.11,-0.46,-0.05,0.35,-0.25,0.36,-0.08,-0.24,0.13,0.55,-0.06,1.24,0.4,-0.16,0.81,-0.12,0.39,-0.24,-0.22,0.03,0.06,0.0,-0.42,-0.07,-0.03,0.06,0.17,-0.33,0.4,-0.73,-0.16,-0.12,0.13,-0.15,0.1,0.39,0.03,-0.23,0.23,-0.19,-0.09,-0.38,0.21,-0.13,-0.24,-0.35,-0.57,-0.14,-0.34,-0.01,0.12,0.27,0.03,-0.29,-0.06,0.31,-0.07,-0.21,0.65,-0.04,-0.24,-0.13,-0.08,0.55,0.29,0.16,-0.2,0.16,-0.12,-0.15,0.01,-0.15,0.02,-0.06,-0.21,0.02,-0.02,0.31,-0.23,-0.07,-0.29,0.23,-0.19,-0.27,0.38,-0.32,0.2,0.55,0.2,0.34,0.51,-0.48,-0.11,-0.19,0.04,0.25,0.28,0.72,0.05,-0.1,0.28,0.11,-0.23,0.01,0.6,-0.3,-0.18,0.55,0.22,0.86,-0.14,-0.31,-0.41,-0.03,-0.02,0.52,0.02,-0.3,1.21,-0.13,0.01,0.8,-0.04,-0.29,0.34,0.04,-0.22,0.03,-0.3,-0.09,-0.07,-0.03,-0.4,-0.47,0.15,-0.27,0.11,-0.05,1.2,0.01,-0.3,0.3,0.24,-0.13,1.02,-0.54,-0.14,-0.12,-0.53,-0.15,0.03,0.05,0.95,-0.04,0.03,0.0,0.24,-0.07,0.12,-0.84,0.16,0.08,-0.2,-0.04,0.33,-0.37,-0.18,0.09,0.2,-0.26,-0.04,-0.08,-0.26,0.74,-0.08,0.33,0.02,-0.21,0.32,0.83,-0.02,-0.01,-0.09,-0.41,-0.27,-0.38,0.56,-0.44,-0.06,0.37,0.92,0.56,-0.06,0.05,1.01,-0.26,-0.27,-0.28,-0.04,0.41,-0.14,0.34,-0.15,-0.09,0.4,-0.03,0.05,0.12,0.17,-0.4,0.18,-0.22,-0.31,0.03,-0.12,-0.1,0.14,-0.19,-0.6,-0.17,-0.24,-0.22,-0.13,-0.2,0.39,-0.84,-0.13,0.26,-0.04,0.13,0.34,-0.26,-0.04,0.6,0.2,-0.06,-0.21,-0.32,0.07,0.45,-0.12,-0.36,-0.11,0.26,0.27,-0.29,0.45,-0.18,0.72,-0.15,0.13,-0.16,2.3,0.16,0.2,-0.12,0.7,-0.07,0.11,-0.36,0.18,-0.18,-0.46,-0.46,-0.28,-0.38,-0.43,0.28,0.88,0.37,-0.4,-0.35,0.01,0.26,-0.14,0.22,0.04,-0.17,0.09,-0.28,0.64,0.12,0.01,0.07,0.24,-0.15,-0.64,-0.12,0.45,0.03,-0.52,-0.12,0.48,0.21,0.41,0.4,0.3,-0.18,0.15,0.5,0.1,-0.12,-0.32,-0.14,0.48,-0.22,-0.19,0.16,-0.05,0.07,-0.09,0.72,-0.14,-0.09,0.27,-0.37,-0.14,-0.49,-0.05,0.16,0.03,-0.45,-0.14,0.31,-0.16,-0.61,-0.41,-0.09,0.37,0.01,0.05,-0.06,-0.11,0.11,-0.44,0.04,-0.09,-0.31,-0.73,0.12,-0.15,0.0,-0.09,0.05,0.15,-0.34,0.12,0.19,0.04,0.2,-0.16,-0.25,-0.05,0.13,0.66,-0.41,0.26,1.2,-0.2,-0.05,-0.16,0.08,0.24,-0.38,0.05,0.31,0.3,-0.01,0.11,0.31,0.03,-0.06,-0.17,0.26,-0.15,1.31,-0.21,0.17,-0.17,-0.21,0.25,-0.35,0.21,-0.18,-0.23,-0.1,-0.02,0.05,-0.09,0.45,-0.24,-0.06,0.37,0.77,-0.62,0.17,-0.14,-0.04,-0.02,-0.01,0.48,0.04,0.19,1.02,0.35,-0.32,-0.06,0.21,-0.26,0.15,0.02,0.27,-0.11,0.0,0.12,0.09,-0.23,1.69,-0.07,2.2,-0.62,0.29,0.14,-0.16,-0.22,0.08,0.38,0.1,-0.13,0.82,-0.07,0.79,0.17,-0.14,0.58,-0.18,-0.47,-0.35,0.78,-0.11,0.03,-0.08,0.06,0.24,0.03,0.04,0.46,-0.04,-0.03,-0.24,0.36,-0.16,0.23,-0.11,-0.53,0.04,-0.12,-0.12,-0.93,0.14,0.59,-0.18,-0.03,0.26,-0.11,-0.08,-0.11,-0.01,-0.63,-0.17,0.41,-0.21,-0.22,0.09,3.02,-0.36,-0.17,0.33,1.75,-0.04,0.84,0.12,-0.33,0.19,-0.39,-0.24,0.18,0.16,-0.69,-0.08,1.22,-0.04,0.04,0.16,-0.41,0.05,0.53,0.24,-0.09,-0.31,0.25,-0.08,0.25,0.79,-0.21,0.22,-0.08,0.12,-0.13,0.2,0.64,0.28,-0.16,-0.05,0.25,0.83,0.17,-0.03,-0.27,0.08,0.22,0.28,-0.27,1.14,0.06,0.06,-0.46,-0.02,-0.04,-0.36,-0.18,0.15,-0.52,0.3,0.07,-0.3,0.14,0.34,0.06,0.16,0.29,-0.21,0.16,0.05,-0.44,-0.57,-0.34,0.11,1.48,0.15,-0.26,-0.22,0.57,1.1,0.0,-0.32,0.09,0.06,-0.3,-0.03,-0.14,0.15,0.15,0.19,0.02,-0.47,-0.45,0.78,-0.17,0.05,-0.14,-0.07,0.06,-0.2,0.07,-0.01,-0.3,0.14,0.17,0.18,-0.03,0.32,-0.1,-1.07,-0.13,0.56,0.0,0.15,-0.2,1.39,-0.41,0.27,-0.23,-0.39,0.22,-0.39,-0.12,-0.04,-0.22,-0.32,-0.18,-0.15,-0.49,0.08,0.76,1.3,0.33,0.59,0.08,0.14,-0.29,0.17,0.34,-0.14,0.96,-0.42,-0.08,0.08,0.37,0.12,-0.2,0.23,0.21,0.01,-0.12,0.26,0.26,-0.5,1.15,0.18,-0.41,-0.04,-0.18,-0.16,0.05,0.32,-0.02,-0.23,0.3,0.16,-0.12,-0.15,-0.27,-0.12,-0.05,-0.01,-0.84,1.22,-0.22,-0.12,-0.13,0.21,0.55,-0.15,-0.27,0.0,-0.04,-0.35,0.31,1.81,0.18,-0.13,0.04,-0.1,-0.21,-0.13,0.05,-0.25,-0.57,-0.23,0.21,-0.35,0.14,0.07,0.78,0.04,0.37,0.35,-0.12,0.23,-0.07,-0.26,-0.21,0.24,0.74,0.15,0.0,0.15,-0.24,0.55,-0.21,0.0,-0.04,0.29,0.2,0.05,0.71,-0.12,0.27,0.13,0.47,-0.29,0.02,0.34,0.39,-0.21,0.01,-0.05,-0.03,-0.36,0.22,-0.09,0.0,0.04,0.33,0.39,-0.71,-0.18,0.41,0.16,-0.36,0.02,0.0,0.7,0.03,0.83,-0.38,0.09,-0.22,-0.23,0.13,0.12,0.58,-0.12,-0.23,0.08,-0.81,-0.46,0.66,0.57,-0.37,-0.3,0.09,-0.19,-0.32,0.18,-0.19,-0.39,0.02,-0.08,-0.16,0.38,0.38,-0.14,-0.35,0.19,-0.07,-0.25,-0.06,0.08,0.27,0.41,-0.31,-0.26,-0.03,0.04,-0.25,0.05,0.22,0.3,0.46,0.01,-0.4,0.02,0.77,-0.13,0.0,0.04,-0.27,1.14,-0.16,1.57,-0.24,0.07,-0.61,-0.02,0.46,-0.17,0.02,0.17,-0.63,0.28,-0.14,-0.11,-0.75,0.25,0.13,-0.29,-0.1,-0.05,-0.06,0.06,-0.19,-0.14,-0.65,0.34,-0.35,0.79,-0.19,0.12,0.12,0.22,-0.12,-0.04,-0.1,0.22,0.07,-0.26,1.49,0.56,-0.01,0.24,0.35,0.08,0.19,-0.03,0.1,-0.19,0.87,0.03,-0.13,0.0,0.06,0.11,1.45,0.21,0.04,0.49,1.19,-0.29,1.05,0.54,0.08,-0.01,0.66,0.07,-0.14,0.09,-0.18,1.17,0.66,0.24,-0.27,-0.16,-0.2,0.9,-0.35,0.35,-0.02,0.18,0.35,0.22,0.08,-0.06,-0.06,0.01,0.33,0.36,0.28,0.85,0.59,-0.05,-0.06,-0.1,0.15,-0.01,-0.27,-0.07,-0.22,0.24,-0.08,-0.1,0.09,0.2,-0.09,0.52,0.05,0.02,-0.52,0.08,-0.19,-0.24,0.18,-0.11,0.21,1.21,0.1,0.43,-0.25,-0.09,-0.19,-0.13,-0.46,0.03,0.07,-0.14,-0.19,0.44,-0.46,0.51,-0.2,0.1,0.12,-0.24,-0.05,-0.36,0.27,-0.35,-0.1,-0.04,0.1,-0.06,0.13,0.05,-0.22,-0.21,0.07,0.04,-0.06,-0.6,-0.28,-0.15,1.24,0.34,-0.22,-0.3,-0.18,0.08,-0.08,0.22,0.35,0.07,0.04,0.15,-0.15,-0.04,0.16,-0.52,0.35,-0.45,-0.1,-0.19,-0.09,-0.19,-0.21,0.0,0.82,-0.54,0.52,-0.26,0.47,0.08,-0.07,-0.32,-0.09,-0.26,-0.04,-0.55,0.56,-0.08,-0.13,0.31,0.04,-0.32,-0.06,-0.02,0.38,0.52,-0.12,-0.21,-0.21,0.33,-0.6,-0.04,0.14,0.04,-0.52,-0.35,0.12,0.38,-0.34,-0.06,0.05,-0.33,-0.19,0.04,0.07,-0.01,0.07,-0.29,-0.17,0.11,-0.03,-0.24,0.2,-0.1,0.95,-0.02,0.19,-0.28,-0.09,-0.36,0.24,0.09,1.22,1.86,-0.28,-0.15,-0.26,-0.12,0.02,-0.13,0.02,0.1,0.54,-0.09,-0.07,-0.2,0.13,-0.03,0.47,-0.12,0.0,0.15,0.19,0.03,0.33,0.43,1.02,0.05,1.26,0.22,0.87,-0.16,0.18,0.02,0.22,1.28,-0.65,0.11,0.37,1.05,0.0,0.24,0.62,-0.21,0.16,0.41,0.25,-0.05,0.1,0.01,0.54,0.15,-0.09,-0.01,-0.27,0.21,0.03,1.11,0.4,-0.26,-0.1,-0.01,-0.08,1.38,0.34,0.3,-0.19,0.94,1.62,-0.24,-0.15,0.34,0.02,0.3,-0.26,0.19,-0.07,-0.3,0.04,-0.58,-0.09,0.02,-0.21,-0.14,0.16,0.27,-0.26,-0.29,-0.24,-0.23,-0.16,-0.12,0.88,-0.8,0.15,0.07,0.43,0.09,-0.2,-0.07,-0.36,-0.42,0.29,-0.02,0.51,-0.23,-0.01,0.08,-0.63,0.02,0.58,-0.16,0.02,0.55,-0.14,-0.27,0.05,-0.25,-0.19,-0.16,0.18,0.17,0.25,0.09,-0.11,-0.55,-0.13,0.23,0.4,0.18,-0.11,-0.15,-0.23,-0.09,-0.03,0.1,-0.33,-0.13,0.24,-0.16,0.28,-0.21,-0.78,-0.47,0.28,-0.01,0.54,0.31,-0.03,-0.4,-0.09,0.01,-0.23,-0.07,0.0,0.38,0.11,-0.13,-0.32,-0.24,-0.13,0.56,0.04,-0.96,-0.37,-0.02,-0.11,-0.16,-0.11,0.14,-0.46,-0.17,0.43,0.19,-0.25,-0.23,0.11,0.11,-0.17,0.08,0.1,-0.15,0.13,-0.13,0.22,-0.26,-0.23,-0.16,0.07,1.21,0.0,-0.17,0.08,-0.18,-0.16,-0.24,0.45,0.43,1.39,-0.65,0.06,-0.13,-0.24,-0.05,-0.47,-0.46,0.5,0.04,-0.15,0.34,0.14,0.11,-0.32,-0.03,-0.09,0.19,-0.61,0.21,-0.37,-0.22,-0.17,-0.57,-0.15,-0.78,-0.46,0.11,-0.25,-0.13,0.02,0.02,0.23,0.41,0.71,0.64,0.05,-0.4,-0.18,-0.83,0.52,1.06,0.33,-0.1,-0.28,-0.04,-0.13,-0.01,0.28,-0.11,0.68,-0.18,-0.17,-0.32,0.12,0.48,-0.05,-0.1,0.08,-0.22,-0.3,0.38,-0.2,0.34,0.44,-0.19,-0.33,-0.16,-0.02,0.24,-0.16,-0.55,1.29,0.06,0.0,0.15,0.41,0.0,-0.08,-0.19,0.27,0.38,-0.5,0.16,-0.05,0.18,0.0,-0.06,0.35,-0.42,-0.25,-0.01,-0.21,-0.23,-0.32,0.29,-0.29,0.17,-0.02,0.05,0.11,-0.01,-0.07,-0.14,0.0,-0.01,0.41,-0.2,-0.04,0.05,1.49,-0.26,0.13,-0.04,-0.16,-0.44,-0.07,-0.04,-0.25,-0.39,-0.49,-0.08,0.02,1.43,0.09,0.13,-0.07,0.19,-0.1,0.02,1.29,0.12,1.1,0.63,-0.72,-0.32,-0.25,0.63,0.08,-0.41,0.52,0.43,0.19,-0.04,0.03,-0.94,-0.24,0.07,-0.09,0.63,0.38,0.04,0.9,-0.01,0.03,-0.29,-0.46,0.09,0.02,0.4,0.14,0.68,0.11,0.13,-0.24,-0.22,1.95,-0.35,-0.49,1.31,-0.08,0.81,-0.27,0.2,-0.3,-0.21,-0.02,0.11,-0.44,0.23,0.38,-0.14,-0.01,-0.17,0.38,0.21,-0.23,-0.43,-0.23,-0.57,0.57,-0.41,0.4,0.17,-0.03,-0.15,-0.02,-0.3,0.02,0.27,0.09,0.25,-0.32,0.22,0.36,0.1,-0.15,-0.58,0.13,-0.1,-0.07,0.29,0.07,1.31,-0.16,-0.28,0.03,0.26,-0.07,0.11,0.05,-0.19,-0.42,0.07,0.21,0.36,-0.3,1.64,0.23,-0.3,-0.26,0.48,0.37,0.31,0.66,0.25,-0.24,0.22,-0.29,0.16,0.31,0.27,-0.29,-0.26,-0.04,0.51,-0.17,1.49,-0.08,0.49,-0.02,0.08,0.43,-0.92,-0.66,-0.38,-0.25,0.02,0.37,-0.32,0.03,-0.26,-0.06,1.01,-0.02,-0.03,0.29,0.33,0.12,-0.13,-0.07,-0.07,-0.03,-0.02,-0.36,-0.37,0.06,-0.21,-0.12,-0.11,-0.42,0.66,0.22,0.57,0.05,-0.2,0.48,-0.25,-0.21,0.81,0.18,-0.01,-0.13,-0.17,0.21,-0.3,0.27,0.82,-0.25,0.08,-0.18,-0.14,0.6,-0.1,-0.07,-0.03,0.23,0.21,-0.04,0.12,-0.42,0.55,-0.37,-0.08,0.19,-0.39,0.47,0.81,-0.27,0.56,0.34,-0.06,-0.26,-0.46,0.51,-0.13,0.24,-0.24,-0.33,-0.18,0.24,2.52,-0.3,-0.47,0.06,0.27,-0.25,0.07,0.09,-0.03,0.49,1.13,1.3,-0.22,0.11,-0.28,0.01,0.13,-0.2,0.03,0.41,0.36,0.26,1.2,-0.44,0.08,0.13,-0.04,-0.3,-0.27,-0.27,-0.65,0.13,-0.27,-0.09,1.21,-0.22,-0.13,1.22,0.39,0.15,1.32,2.13,0.11,0.48,0.62,0.26,0.15,-0.01,-0.16,-0.29,0.54,0.13,0.44,-0.43,0.19,-0.26,1.26,-0.07,-0.09,-0.22,-0.05,0.4,0.15,-0.21,0.17,1.45,0.35,-0.12,0.19,-0.55,0.2,-0.06,0.21,-0.03,0.27,0.08,0.31,-0.07,-0.1,-0.3,0.28,-0.13,-0.41,-0.42,0.31,0.1,-0.68,-0.4,-0.01,-0.61,-0.03,0.01,-0.32,0.17,0.24,0.01,0.28,-0.2,-0.1,0.3,0.31,0.28,0.56,-0.65,0.18,-0.34,-0.12,0.23,0.4,0.08,0.4,0.85,0.64,0.08,-0.43,0.6,-0.63,-0.15,0.41,-0.68,-0.13,0.17,0.06,-0.14,0.01,0.33,-0.27,1.56,-0.45,-0.27,0.22,1.33,-0.34,1.93,0.48,0.39,0.05,0.01,-0.19,-0.1,-0.13,0.77,0.57,0.19,0.79,-0.23,-0.12,-0.14,-0.45,0.53,-0.33,-0.07,-0.2,0.75,-0.03,-0.14,-0.17,0.08,-0.28,-0.09,0.12,-0.36,0.09,0.07,1.0,0.03,1.26,0.5,-0.35,1.36,-0.11,-0.13,-0.33,-0.13,-0.11,-0.17,0.13,-0.12,0.08,0.27,0.06,0.09,0.17,0.15,-0.09,-0.34,-0.42,-0.11,-0.02,0.09,0.05,-0.13,0.11,-0.18,-0.1,0.14,0.23,-0.52,-0.05,0.06,-0.19,-0.18,-0.18,-0.47,-0.3,-0.22,-0.22,-0.45,-0.28,0.38,0.65,0.33,0.66,-0.32,-0.04,0.62,-0.25,0.25,1.13,-0.48,-0.16,0.15,-0.23,2.5,0.07,-0.22,0.38,-0.46,-0.39,-0.08,-0.15,0.12,0.08,0.51,0.81,0.12,1.24,0.38,-0.05,-0.41,-0.05,-0.01,0.09,0.02,0.03,0.34,0.51,0.59,-0.26,-0.54,-0.04,-0.36,-0.48,0.4,-0.02,0.39,0.62,0.26,0.84,0.07,0.06,0.18,0.41,-0.44,0.2,0.03,0.28,1.19,-0.33,-0.14,-0.17,0.05,-0.52,0.31,0.29,-0.05,0.06,-0.27,0.09,0.03,0.8,0.27,0.2,-0.04,0.13,0.35,-0.43,-0.08,-0.56,-0.16,-0.12,-0.62,0.08,0.28,0.04,1.26,0.19,-0.27,-0.22,0.21,0.11,1.41,0.31,0.11,0.53,0.24,0.08,0.21,-0.16,0.22,0.21,-0.06,0.37,0.29,0.01,0.03,-0.09,-0.05,0.02,-0.35,-0.07,-0.3,-0.15,0.37,-0.06,-0.21,2.1,-0.02,0.15,0.09,0.1,0.52,0.08,0.27,0.0,-0.21,0.13,-0.32,0.64,-0.26,0.43,-0.44,0.18,0.05,-0.23,0.16,-0.24,-0.24,-0.09,0.17,-0.39,-0.32,0.05,-0.04,-0.16,-0.26,0.11,-0.03,-0.03,-0.12,0.06,0.07,0.17,-0.15,-0.51,0.28,0.33,1.62,0.39,-0.56,-0.03,-1.1,0.11,0.34,0.13,0.6,-0.19,0.25,-0.32,-0.14,-0.09,-0.06,-0.09,-0.24,-0.14,-0.27,-0.32,-0.26,0.7,-0.11,0.04,-0.08,-0.23,0.06,-0.07,-0.15,0.29,0.17,-0.04,0.06,-0.17,0.41,0.21,0.0,1.21,-0.01,-0.1,-0.2,-0.35,0.23,-0.01,-0.42,0.02,-0.18,-0.12,-0.17,-0.24,0.01,0.0,-0.16,-0.29,0.05,-0.66,0.4,0.16,0.42,0.51,0.13,-0.51,-0.32,0.08,0.41,0.14,-0.4,-0.11,0.49,-0.29,-0.3,-0.08,-0.18,0.26,-0.27,-0.12,0.87,0.1,0.19,0.17,-0.06,-0.23,-0.21,1.01,0.18,0.09,1.55,0.07,-0.13,-0.42,-0.18,-0.34,0.01,-0.02,0.23,0.11,0.46,0.15,0.4,-0.36,-0.36,-0.11,0.05,0.05,-0.25,-0.07,-0.09,-0.06,0.08,-0.25,-0.2,0.04,-0.25,0.06,-0.54,0.43,0.94,-0.15,-0.2,0.27,0.03,-0.25,-0.12,-0.18,0.72,-0.28,0.16,-0.13,-0.57,-0.19,-0.48,-0.35,0.23,0.84,0.06,-0.33,-0.36,0.28,-0.09,-0.22,-0.33,-0.05,0.36,0.12,0.04,0.03,-0.38,-0.27,0.28,0.19,0.0,0.03,-0.06,1.44,0.15,0.11,0.22,0.63,0.54,-0.26,0.18,0.03,0.58,1.35,-0.02,0.34,-0.01,0.14,0.1,-0.15,1.17,0.02,0.13,0.17,-0.28,0.39,0.4,-0.23,0.4,-0.07,0.2,0.78,0.34,-0.31,1.7,-0.15,0.15,0.28,0.06,-0.27,1.42,0.15,-0.05,0.36,-0.23,-0.05,0.71,-0.23,0.46,-0.38,-0.39,-0.13,1.43,-0.09,0.16,-0.31,0.49,-0.16,-0.08,0.02,-0.28,0.17,0.43,-0.32,0.1,-0.3,0.13,0.1,-0.25,-0.11,-0.19,-0.39,0.23,0.0,-0.04,0.05,0.07,-0.31,0.1,-0.11,-0.12,-0.26,0.62,0.32,-0.24,0.15,1.21,-0.15,-0.23,-0.15,-0.32,0.64,-0.26,1.22,0.13,-0.38,0.87,0.06,0.17,-0.14,-0.4,0.19,-0.38,1.13,-0.12,-0.12,-0.11,0.04,-0.01,-0.56,-0.22,0.26,-0.5,-0.28,-0.27,0.3,0.08,0.04,0.25,0.09,1.03,0.09,-0.06,0.11,0.16,-0.1,-0.3,-0.08,-0.36,0.22,0.38,0.13,-0.29,-0.31,-0.19,-0.15,-0.12,-0.19,0.02,0.12,0.17,0.02,0.33,-0.12,-0.05,0.04,0.52,-0.05,0.51,0.18,0.36,0.18,0.04,-0.07,0.84,0.06,0.45,1.11,-0.26,-0.02,-0.6,0.06,0.1,0.06,-0.22,0.01,1.02,1.52,-0.31,0.33,0.61,0.06,0.36,-0.21,-0.22,-0.05,0.29,-0.06,0.07,-0.15,-0.12,0.19,-0.17,0.65,-0.06,-0.24,0.24,-0.13,0.07,-0.25,0.33,0.02,-0.06,-0.24,0.45,-0.21,-0.21,0.27,0.31,-0.03,0.07,-0.13,-0.19,-0.25,0.04,-0.03,-0.03,0.15,-0.27,-0.23,-0.05,0.0,-0.1,-0.1,0.64,0.06,-0.38,-0.39,-0.3,0.03,0.81,-0.22,0.28,0.24,-0.64,0.19,0.92,-0.09,-0.14,-0.13,0.58,-0.23,1.28,-0.53,0.08,0.31,-0.51,-0.12,-0.22,1.1,-0.2,0.07,-0.68,-0.24,-0.03,0.21,0.37,0.59,-0.18,0.1,-0.06,0.03,-0.59,-0.31,1.52,0.4,-0.42,-0.39,-0.03,0.18,-0.39,0.11,-0.1,0.08,0.4,0.52,-0.46,0.14,0.07,-1.0,-0.07,-0.1,0.21,1.11,0.38,-0.12,-0.16,0.2,-0.15,-0.21,0.54,-0.56,0.34,-0.61,0.01,0.07,-0.31,0.25,0.19,-0.1,0.24,-0.26,0.13,0.11,0.34,-0.03,-0.37,0.24,-0.48,0.1,0.25,-0.01,0.13,0.23,-0.11,-0.2,0.24,0.0,0.25,-0.11,0.56,0.86,-0.05,-0.06,-0.09,0.04,-0.47,0.19,0.46,-0.67,0.37,-0.31,0.24,0.11,-0.19,-0.33,-0.08,0.11,0.08,-0.57,-0.35,0.0,-0.18,-0.16,0.09,0.43,-0.03,0.1,-0.08,0.09,0.17,-0.33,-0.12,-0.22,-0.03,0.04,0.18,0.06,-0.37,1.84,-0.66,-0.19,0.12,-0.2,-0.33,0.09,0.61,0.18,1.39,0.02,0.84,-0.39,-0.14,-0.18,-0.31,0.05,0.27,-0.05,-0.06,0.04,-0.04,0.06,-0.13,0.1,0.32,0.2,1.0,0.21,-0.15,-0.01,-0.4,0.14,-0.19,0.39,-0.23,-0.14,0.37,-0.16,0.3,0.22,0.22,0.03,0.1,0.22,0.04,-0.41,0.01,-0.14,-0.11,0.01,-0.19,-0.08,-0.29,0.11,0.34,0.2,0.55,0.04,0.89,0.16,0.74,-0.07,0.08,0.01,-0.79,0.02,0.05,0.27,0.27,-0.38,0.03,-0.25,-0.43,1.16,0.04,0.19,0.0,-0.24,0.02,0.11,0.06,0.4,-0.21,-0.01,0.1,1.08,0.25,-0.04,1.23,-0.25,0.0,-0.3,0.02,1.31,-0.04,-0.02,0.14,0.16,-0.35,0.09,0.13,-0.03,0.16,-0.11,-0.14,-0.13,0.1,-0.16,0.18,-0.51,-0.27,1.32,0.29,-0.39,0.13,-0.01,-0.55,-0.13,-0.11,0.0,-0.11,1.32,1.07,0.11,0.1,0.19,0.01,0.12,0.39,0.56,-0.13,-0.25,-0.58,0.57,0.35,0.91,0.02,-0.27,0.2,-0.02,0.34,-0.08,0.38,-0.43,-0.3,-0.51,0.28,-0.1,-0.09,0.17,0.13,-0.45,0.12,0.03,-0.04,0.05,-0.12,0.38,0.31,1.47,-0.21,0.04,-0.3,-0.21,-0.36,-0.05,-0.19,-0.24,0.68,-0.25,-0.33,0.57,0.14,-0.26,-0.17,0.04,0.47,-0.19,0.19,0.19,-0.08,0.01,2.17,0.23,0.39,0.0,-0.31,-0.12,0.07,-0.3,-0.27,-0.02,-0.05,1.05,-0.15,1.74,0.2,-0.18,-0.17,-0.45,-0.48,-0.03,0.47,-0.09,0.02,0.52,-0.29,0.06,-0.44,0.68,0.32,-0.08,-0.14,0.19,-0.15,-0.18,-0.16,-0.04,-0.27,0.84,1.35,0.3,-0.14,-0.29,-0.03,0.25,0.76,-0.11,-0.29,0.16,-0.31,0.6,0.4,-0.1,-0.73,0.02,-0.32,0.05,0.1,0.26,-0.59,-0.46,0.07,-0.23,-0.63,-0.29,0.37,-0.22,0.05,0.15,-0.01,0.13,0.12,-0.3,-0.18,0.16,-0.02,-0.15,0.04,0.52,-0.1,-0.28,0.54,0.12,-0.29,-0.13,1.02,-0.21,0.26,0.15,-0.72,-0.32,-0.1,-0.16,0.01,-0.3,0.11,0.1,0.2,-0.11,0.03,-0.08,-0.13,-0.21,-0.14,0.17,-0.08,0.01,-0.17,0.31,0.13,-0.29,0.01,0.13,0.16,-0.13,0.21,0.87,0.3,-0.11,-0.06,-0.17,-0.19,-0.25,0.78,-0.29,-0.26,1.0,0.08,-0.3,0.62,0.47,-0.51,0.01,-0.05,0.08,0.02,0.43,-0.11,0.23,0.36,0.71,0.1,1.48,-0.2,-0.06,-0.14,-0.12,0.07,-0.04,-0.27,0.45,-0.28,-0.31,-0.03,-0.17,-0.11,0.02,0.07,-0.11,-0.12,1.55,-0.07,-0.2,-0.34,-0.3,0.09,-0.65,0.32,-0.24,0.09,0.54,0.3,-0.1,0.22,1.4,0.39,0.0,0.53,0.13,-0.08,0.64,-0.6,0.0,-0.03,-0.03,0.12,0.07,0.56,-0.06,-0.72,0.2,0.12,0.31,-0.21,-0.32,0.11,-0.1,0.05,-0.17,-0.63,-0.7,-0.03,0.58,-0.02,-0.08,-0.53,1.2,0.26,0.13,-0.48,-0.03,-0.38,0.25,-0.05,0.15,-0.11,0.06,-0.03,0.06,0.03,-0.06,-0.21,0.42,0.39,-0.26,-0.27,-0.07,0.55,0.1,0.01,0.52,-0.32,0.24,-0.2,0.36,0.46,-0.17,0.31,2.13,-0.32,0.08,-0.32,0.26,-0.22,0.04,-0.71,0.09,-0.02,-0.38,0.34,0.06,0.69,0.26,0.5,-0.13,-0.1,-0.15,-0.29,-0.06,-0.24,0.27,-0.1,-0.02,0.02,0.37,0.04,0.07,0.12,0.04,0.56,-0.25,-0.26,0.27,-0.13,1.58,-1.06,1.72,0.0,0.52,-0.4,0.26,0.62,0.24,0.24,0.94,0.07,-0.12,-0.16,0.24,0.31,-0.47,-0.1,-0.42,0.14,0.25,0.54,0.43,0.58,-0.02,0.09,0.01,0.1,0.79,0.57,0.09,0.13,0.07,-0.03,-0.15,-0.3,0.08,0.93,0.06,-0.14,-0.17,-0.06,-0.11,0.06,0.02,0.07,-0.59,-0.14,0.23,-0.32,0.39,-0.24,0.26,-0.11,0.56,-0.17,-0.16,0.34,0.05,-0.1,-0.21,-0.2,2.19,0.98,0.54,0.43,-0.04,-0.67,-0.74,-0.09,-0.21,-0.01,0.17,-0.11,-0.42,0.22,0.41,0.97,0.12,-0.4,-0.02,-0.35,0.0,-0.08,0.3,0.07,-0.02,0.14,-0.18,0.31,-0.14,0.15,0.37,-0.14,-0.01,-0.32,0.34,-0.14,0.1,-0.22,-0.13,-0.14,-0.16,-0.39,0.13,0.11,0.22,0.04,0.76,-0.33,-0.34,-0.29,0.18,1.0,-0.08,1.49,0.42,-0.31,-0.09,0.14,-0.19,-0.15,0.45,0.34,0.6,-0.32,0.04,-0.12,0.05,-0.33,0.12,1.38,-0.38,0.13,-0.23,0.62,-0.23,0.0,0.04,-0.41,-0.1,0.11,-0.08,-0.24,0.57,-0.08,-0.14,-0.05,-0.35,-0.18,0.24,0.5,-0.13,0.42,0.05,-0.15,0.05,0.17,-0.39,-0.25,0.31,0.13,-0.06,-0.1,-0.12,-0.65,-0.2,0.28,0.07,0.33,0.02,-0.56,-0.06,-0.48,0.41,-0.27,0.54,-0.07,-0.48,-0.37,-0.52,0.05,-0.53,0.01,-0.3,-0.07,0.32,-0.17,-0.35,0.19,-0.15,0.28,0.27,1.25,-0.17,-0.11,0.22,0.27,-0.1,0.28,-0.29,0.13,-0.12,0.11,0.4,-0.1,0.0,-0.04,0.05,-0.41,-0.12,-0.06,0.08,0.2,-0.16,0.24,0.81,1.1,0.02,-0.26,-0.48,-0.38,-0.13,0.05,-0.08,-0.51,-0.26,-0.17,0.07,0.19,0.21,0.11,-0.06,-0.29,0.21,0.29,-0.59,0.73,0.15,0.39,0.77,0.14,-0.23,0.11,-0.01,0.42,0.36,0.57,1.54,0.05,-0.01,-0.05,-0.07,-0.55,-0.09,0.3,-0.03,0.22,-0.06,0.46,-0.17,0.12,-0.39,0.28,-0.26,0.31,-0.15,0.19,-0.31,-0.09,-0.47,-0.4,0.04,-0.41,-0.25,-0.29,-0.45,-0.07,0.05,0.19,-0.06,0.44,0.08,-0.18,-0.15,1.86,0.03,0.25,0.96,-0.05,0.45,0.05,0.07,-0.01,0.19,-0.02,0.43,-0.36,0.31,0.45,-0.14,0.14,-0.1,0.29,-0.18,-0.44,-0.27,-0.02,0.01,1.31,0.05,0.81,-0.11,-0.44,0.79,-0.19,-0.17,-0.35,0.11,0.12,0.19,0.88,-0.26,-0.31,-0.05,-0.17,0.16,0.33,-0.04,0.26,0.2,0.44,-0.1,0.01,0.58,0.06,-0.23,0.83,0.47,0.09,-0.33,-0.05,-0.27,-0.58,0.59,-0.09,-0.05,0.98,-0.1,0.49,-0.16,-0.39,-0.05,0.3,0.28,-0.25,-0.33,-0.51,-0.24,-0.15,-0.28,0.86,-0.35,0.12,0.02,-0.27,0.13,0.04,0.21,-0.13,-0.34,0.12,-0.12,0.52,0.14,0.38,0.49,1.18,-0.26,0.0,0.04,-0.33,0.19,0.03,-0.66,0.37,-0.16,0.67,0.33,0.12,0.52,0.44,0.1,-0.03,0.29,0.07,-0.13,-0.02,0.04,0.11,0.37,0.09,0.25,-0.2,0.04,1.17,-0.4,0.16,0.28,0.44,0.01,-0.29,0.19,0.1,-0.11,-0.02,0.19,0.92,0.33,-0.41,-0.22,0.31,-0.03,0.29,0.32,-0.02,-0.14,-0.4,1.1,-0.61,0.1,-0.27,0.05,-0.03,-0.05,0.14,0.0,-0.5,-0.47,0.24,-0.23,-0.16,0.04,0.06,-0.17,-0.28,0.04,0.16,1.35,0.21,-0.17,0.03,-0.11,0.04,-0.05,0.43,-0.2,0.01,-0.22,-0.04,0.28,-0.31,-0.01,-0.21,0.04,0.03,0.2,-0.22,1.02,0.82,0.26,0.43,0.28,-0.29,0.13,-0.34,0.06,0.16,0.11,1.41,0.04,0.05,0.04,0.47,-0.12,0.35,0.47,0.36,-0.21,0.06,0.38,0.16,0.66,0.08,0.06,0.27,-0.07,-0.15,0.01,0.06,-0.02,-0.37,0.49,0.16,-0.21,0.75,-0.3,0.31,0.4,0.33,-0.27,0.58,0.15,0.02,-0.26,-0.05,1.99,-0.56,-0.13,0.09,0.6,0.1,-0.09,-0.56,0.39,-0.24,-0.54,-0.41,-0.05,-0.07,0.21,0.04,0.13,0.1,-0.08,-0.18,-0.24,0.72,0.06,0.26,1.61,-0.26,0.31,1.04,-0.05,0.12,-0.36,1.63,0.19,-0.26,0.06,0.13,-0.51,0.13,-0.18,-0.12,0.22,-0.04,-0.08,1.07,-0.2,0.92,-0.01,-0.13,0.1,0.22,-0.51,0.67,-0.07,-0.26,-0.14,0.34,0.17,0.13,-0.01,0.29,0.05,-0.11,0.26,0.17,-0.13,0.02,-0.41,0.23,0.33,-0.11,0.14,0.11,0.04,0.39,-0.43,0.05,0.16,-0.02,-0.23,0.28,0.13,0.57,-0.17,-0.19,-0.05,-0.03,0.72,-0.3,0.18,-0.13,-0.17,0.78,0.2,-0.35,-0.26,0.04,-0.02,-0.13,-0.01,-0.33,0.1,-0.11,0.04,0.11,-0.22,-0.06,-0.27,-0.2,-0.15,0.07,0.0,0.4,-0.3,0.18,-0.76,-0.12,-0.2,-0.08,-0.07,-0.21,-0.04,0.71,0.13,-0.06,-0.18,-0.27,0.06,-0.39,-0.07,-0.35,0.1,-0.13,-0.41,0.16,-0.16,-0.26,0.1,0.3,0.22,0.0,0.02,0.15,0.02,-0.27,0.15,-0.33,-0.08,-0.29,-0.39,0.07,0.07,0.1,0.02,-0.1,-0.18,0.77,-0.23,0.09,-0.47,1.98,-0.27,-0.28,-0.2,-0.5,-0.12,-0.01,-0.2,-0.24,-0.01,0.26,0.02,-0.13,-0.44,0.27,0.13,0.19,1.3,0.03,-0.08,-0.08,0.05,-0.29,0.31,0.02,0.15,0.18,0.22,0.15,0.0,0.29,-0.15,0.28,-0.11,-0.78,0.17,-0.34,-0.11,-0.22,-0.66,-0.37,-0.33,0.16,-0.68,0.66,-0.37,0.03,0.79,-0.03,-0.36,-0.46,0.35,0.1,0.14,-0.06,-0.24,-0.05,0.1,0.13,0.06,-0.08,0.09,-0.14,-0.37,0.43,-0.65,-0.05,0.16,0.41,0.02,0.65,0.11,0.56,-0.32,-0.73,-0.2,-0.02,-0.2,0.37,-0.23,1.89,-0.02,-0.14,1.12,0.15,0.2,0.25,0.27,0.01,-0.13,-0.1,-0.1,-0.14,0.35,0.0,0.16,-0.32,0.61,-0.34,0.14,-0.23,-0.18,0.05,0.21,-0.17,-0.58,0.06,-0.03,-0.35,0.87,-0.05,0.35,0.08,0.28,-0.4,0.08,-0.12,-0.48,-0.01,0.41,-0.05,-0.05,-0.05,1.16,0.59,0.34,0.01,0.29,-0.03,0.05,-0.43,-0.35,0.36,0.12,-0.46,-0.1,-0.21,1.19,0.61,0.44,-0.23,0.35,-0.25,-0.11,2.25,0.45,0.04,0.09,0.44,-0.17,0.0,0.61,0.19,-0.14,0.04,0.33,-0.23,-0.12,-0.43,-0.18,-0.3,-0.27,0.54,-0.34,-0.31,0.29,0.07,-0.04,-0.27,-0.4,0.66,0.04,-0.12,0.07,-0.09,0.8,0.0,0.6,0.17,0.21,0.04,-0.03,0.23,-0.33,-0.31,-0.04,-0.4,-0.16,-0.25,0.45,0.06,0.16,-0.2,0.23,0.36,-0.43,0.67,0.37,-0.41,0.66,-0.13,-0.45,0.24,0.67,0.5,0.41,0.11,-0.18,0.03,0.4,-0.03,-0.17,0.48,-0.46,0.15,0.08,0.18,0.28,-0.11,0.42,-0.22,0.34,0.34,0.16,-0.49,-0.19,-0.21,-0.13,-0.14,-0.29,-0.31,-0.11,-0.16,-0.43,0.05,0.12,0.06,0.02,0.21,-0.29,-0.18,-0.04,-0.15,0.01,-0.16,0.04,0.53,-0.05,-0.08,-0.01,-0.05,-0.01,0.5,-0.03,-0.12,-0.24,1.13,-0.04,-0.09,0.57,0.14,0.01,-0.22,-0.72,-0.28,-0.57,0.23,-0.49,0.03,0.16,0.29,-0.05,-0.52,-0.19,0.41,-0.28,0.5,-0.08,0.27,0.08,0.01,0.01,-0.1,-0.46,-0.01,0.37,-0.19,0.27,0.0,0.28,2.25,1.32,0.07,-0.05,0.59,0.66,0.12,-0.3,0.08,0.46,-0.1,-0.17,-0.24,-0.29,0.03,-0.06,-0.49,-0.43,0.47,0.63,-0.07,0.33,-1.06,0.41,0.3,-0.46,-0.36,-0.37,-0.4,0.37,0.1,0.08,0.24,-0.1,0.02,0.1,-0.12,-0.22,0.24,0.12,0.17,-0.25,0.07,0.13,-0.24,0.1,0.16,-0.1,-0.13,-0.12,-0.07,-0.06,-0.18,-0.39,0.4,-0.23,-0.06,-0.08,-0.19,0.2,0.35,0.1,-0.3,-0.03,0.29,-0.09,-0.11,-0.08,0.79,0.21,0.23,0.05,-0.03,-0.61,1.87,0.97,0.63,0.38,-0.56,-0.22,-0.25,-0.29,-0.23,-0.28,-0.37,0.15,0.01,0.11,0.45,0.92,-0.13,-0.02,0.0,0.2,0.07,-0.14,0.31,-0.1,-0.26,-0.23,-0.07,0.3,0.16,0.72,-0.31,-0.77,0.01,0.17,-0.17,-0.35,0.2,1.36,0.22,0.25,-0.16,0.04,-0.19,-0.4,1.25,-0.26,1.09,0.09,1.17,0.12,-0.16,-0.21,0.79,0.43,0.08,-0.09,-0.49,-0.16,0.36,0.17,-0.23,-0.58,0.41,0.36,-0.25,0.67,0.15,0.17,0.11,-0.35,-0.44,0.23,-0.2,-0.19,-0.28,-0.02,0.71,0.74,-0.09,-0.08,1.25,0.08,-0.05,-0.24,-0.1,0.02,-0.43,0.62,-0.06,-0.16,0.53,0.48,0.13,0.06,0.34,-0.16,0.62,-0.36,-0.03,-0.22,-0.14,-0.11,0.57,-0.25,0.03,-0.05,0.29,-0.19,0.39,-0.25,-0.24,-0.26,0.21,-0.02,-0.4,-0.41,-0.16,0.07,-0.39,0.08,0.25,0.13,-0.22,0.35,-0.19,-0.52,0.66,0.09,0.68,0.24,0.07,-0.08,-0.09,-0.36,-0.02,0.21,-0.12,0.04,0.17,0.22,0.37,0.37,-0.28,-0.35,-0.53,-0.17,-0.05,0.01,0.09,0.58,-0.41,0.05,0.03,1.14,-0.02,1.24,-0.08,-0.05,0.05,0.25,0.35,-0.13,-0.15,-0.3,-0.3,-0.08,-0.01,-0.37,-0.17,-0.28,0.05,0.19,0.1,0.44,-0.55,-0.03,-0.36,-0.15,1.5,-0.13,0.2,-0.08,-0.15,0.24,0.14,0.22,-0.23,0.13,-0.13,0.73,0.36,0.21,0.22,-0.41,-0.44,-0.63,0.04,0.08,0.36,0.21,-0.11,-0.4,0.05,-0.11,-0.18,-0.15,-0.19,-0.1,0.24,0.05,-0.34,0.05,0.31,0.38,-0.11,-0.22,-0.05,-0.32,-0.11,0.36,-0.39,0.26,-0.07,0.49,-0.27,0.36,0.04,-0.24,-0.5,0.23,0.58,-0.74,0.26,-0.15,0.17,0.01,-0.13,-0.12,-0.14,-0.34,0.07,0.07,0.32,-0.16,-0.06,-0.53,-0.28,0.09,1.0,-0.09,-0.2,-0.09,-0.03,0.26,-0.3,0.16,0.13,-0.02,-0.09,-0.38,-0.21,0.47,-0.09,-0.17,0.16,0.92,0.28,0.15,-0.12,-0.4,-0.12,0.17,-0.59,-0.15,0.27,-0.16,0.36,0.25,-0.09,0.59,0.27,-0.11,0.11,0.13,-0.24,0.28,0.18,1.14,0.32,0.8,0.55,0.11,-0.02,0.04,0.17,0.03,0.32,-0.04,-0.08,-0.02,0.38,0.0,-0.18,-0.49,-0.16,-0.36,-0.36,0.02,-0.36,-0.28,0.11,-0.31,-0.04,-0.3,1.15,-0.12,-0.05,0.05,0.15,0.22,-0.33,0.03,-0.01,0.19,0.58,-0.35,0.23,-0.42,0.12,0.05,-0.43,0.74,0.0,-0.06,-0.19,0.0,0.21,-0.15,0.16,0.1,0.25,0.21,-0.16,0.01,-0.05,-0.12,0.36,-0.23,-0.12,0.86,0.32,0.03,0.06,-0.37,-0.57,-0.08,-0.37,0.41,-0.19,0.2,0.13,-0.27,0.17,0.01,-0.39,0.04,0.23,0.83,0.17,0.21,0.1,-0.02,-0.14,-0.24,-0.29,0.08,-0.13,0.1,-0.24,0.36,-0.26,-0.22,0.42,0.1,0.0,0.1,0.0,1.48,-0.2,-0.1,-0.08,-0.59,-0.16,0.15,-0.04,-0.18,0.26,0.17,-0.19,-0.24,0.22,0.12,0.23,-0.57,-0.04,0.04,-0.17,0.44,-0.83,0.07,-0.38,0.05,0.06,0.52,1.06,-0.77,0.3,-0.32,-0.47,-0.07,-0.14,0.88,0.19,0.47,0.0,0.37,-0.21,0.74,-0.31,0.0,-0.07,0.6,0.51,0.17,0.05,-0.05,0.47,0.21,-0.83,0.05,1.07,0.25,0.55,0.08,-0.25,0.15,-0.54,1.11,-0.33,-0.18,0.13,-0.23,0.05,-0.04,-0.57,0.06,-0.34,0.08,-0.49,1.16,-0.11,0.16,1.71,0.52,0.03,-0.16,-0.31,0.43,-0.19,-0.45,0.09,0.23,1.25,0.04,0.47,0.21,-0.04,-0.06,0.07,-0.02,0.22,-0.04,0.11,0.42,-0.2,-0.12,0.14,-0.54,0.33,-0.1,-0.5,0.21,0.36,-0.79,-0.27,0.31,-0.04,0.03,0.14,-0.04,0.18,-0.2,0.35,0.08,0.06,-0.17,-0.38,0.27,0.42,0.28,-0.13,0.35,-0.17,-0.01,-0.05,-0.09,0.9,-0.23,-0.01,0.5,0.27,0.87,-0.09,-0.16,-0.02,-0.31,1.43,-0.45,-0.29,-0.11,0.89,0.2,0.17,-0.2,-0.42,0.01,-0.05,0.13,-0.2,-0.25,0.11,0.79,-0.33,-0.38,0.09,-0.11,0.08,0.29,0.09,0.06,-0.17,-0.14,-0.11,-0.07,-0.19,-0.27,0.32,0.0,-0.3,-0.18,-0.29,0.08,-0.05,-0.02,-0.3,-0.01,0.46,-0.42,1.94,-0.59,0.03,1.11,-0.28,0.02,-0.04,-0.3,0.38,0.49,0.59,0.27,0.4,-0.02,-0.36,-0.49,0.21,-0.11,0.16,-0.34,0.11,-0.33,0.14,-0.16,0.08,-0.08,-0.19,-0.09,0.04,-0.2,-0.2,-0.05,-0.23,0.04,0.17,0.05,-0.18,-0.14,0.74,0.61,-0.24,-0.06,0.02,0.06,0.09,0.24,-0.31,-0.18,-0.09,0.43,1.21,0.27,-0.2,-0.35,0.02,-0.21,-0.06,-0.35,-0.41,0.12,0.19,-0.06,0.14,0.63,-0.29,0.07,-0.1,0.11,0.4,0.35,0.52,-0.03,0.11,0.26,0.12,0.21,0.39,-0.3,-0.01,-0.07,-0.09,0.58,-0.2,0.52,-0.25,0.38,0.22,0.07,0.01,-0.51,0.45,0.42,-0.07,0.43,0.05,-0.05,0.19,-0.29,-0.39,0.2,0.43,0.29,0.1,0.26,0.06,-0.03,0.0,-0.05,-0.35,0.38,0.48,0.4,-0.5,-0.39,0.16,-0.38,1.05,0.69,-0.2,-0.12,0.25,-0.14,0.41,-0.07,0.07,0.19,0.0,-0.14,0.23,0.13,0.03,0.16,-0.27,-0.17,0.53,-0.21,-0.19,0.18,0.17,-0.38,0.72,0.26,0.12,0.52,0.18,-0.2,-0.16,0.09,0.07,0.33,-0.11,0.18,0.26,-0.19,0.41,0.09,0.29,0.24,0.07,-0.41,-0.18,-0.35,-0.16,0.15,0.25,-0.21,-0.15,0.18,0.11,0.09,0.05,-0.48,0.81,0.34,-0.2,-0.29,0.18,-0.16,-0.16,0.45,-0.4,-0.02,0.5,-0.13,0.4,-0.29,1.36,0.15,-0.69,0.0,-0.12,0.04,-0.19,0.38,0.07,0.08,-0.05,-0.29,-0.03,-0.12,0.47,-0.33,0.87,-0.08,0.33,-0.4,0.22,-0.23,0.05,-0.13,0.12,-0.08,0.34,0.09,0.21,1.27,-0.35,-0.27,1.27,0.02,0.27,0.38,0.09,0.68,-0.1,0.32,-0.2,-0.04,-0.22,0.21,-0.11,-0.03,0.01,0.63,0.15,0.42,0.0,0.31,-0.06,0.25,0.0,-0.07,0.12,0.19,-0.09,0.01,0.01,-0.26,0.19,-0.1,-0.31,2.07,-0.02,0.16,0.17,0.48,1.16,-0.03,-0.04,0.07,0.25,0.29,-0.36,0.1,-0.03,-0.17,0.03,0.01,1.24,0.2,0.24,-0.57,-0.25,-0.07,1.21,-0.37,0.07,0.51,0.45,-0.38,0.5,0.06,1.15,-0.04,-0.06,-0.1,0.46,0.69,-0.25,-0.34,-0.25,0.08,-0.14,0.34,0.71,0.11,0.01,-0.07,0.15,0.4,-0.54,-0.53,-0.63,-0.25,0.45,-0.29,-0.11,-0.28,0.57,-0.12,0.52,0.42,0.57,0.06,-0.22,-0.22,0.43,0.29,-0.19,-0.39,-0.17,-0.03,0.29,0.01,-0.3,1.2,1.64,0.07,0.14,-0.25,0.02,-0.02,-0.2,0.14,-0.66,0.45,-0.17,0.13,-0.21,0.54,0.28,0.21,-0.11,0.14,-0.1,0.22,-0.01,0.51,0.33,0.06,-0.38,0.13,0.33,-0.11,-0.13,-0.33,-0.14,0.2,0.36,-0.17,0.07,-0.14,-0.49,0.36,-0.24,0.32,-0.21,0.64,0.58,-0.02,0.06,-0.04,0.54,-0.1,0.07,0.56,-0.21,-0.18,0.42,0.03,-0.04,0.27,0.24,-0.33,0.13,0.09,-0.01,-0.2,0.43,-0.47,-0.2,0.2,-0.34,0.06,0.32,-0.39,-0.05,0.28,0.16,1.06,-0.19,-0.39,-0.33,0.42,-0.01,-0.31,-0.14,0.35,0.24,-0.29,0.1,0.45,0.65,0.14,0.21,-0.04,-0.08,1.04,-0.2,-0.32,-0.33,-0.3,-0.26,-0.85,0.42,0.16,-0.05,-0.33,0.09,-0.09,0.01,0.07,0.16,-0.19,-0.01,-0.12,0.46,-0.24,0.44,0.07,0.66,0.05,0.18,0.19,0.22,1.16,0.37,0.3,0.05,-0.21,0.12,-0.19,0.11,0.25,0.32,0.18,0.37,0.0,0.08,-0.08,-0.2,-0.1,0.25,-0.03,-0.2,0.01,0.82,1.19,1.15,-0.11,-0.62,-0.12,0.07,0.24,-0.44,0.13,0.46,-0.17,-0.02,0.21,-0.4,-0.22,0.67,-0.13,0.01,-0.15,0.31,0.21,-0.44,0.38,0.58,-0.34,0.14,-0.15,0.25,-0.06,-0.42,-0.2,-0.04,-0.03,-0.26,0.34,-0.28,0.11,-0.33,-0.24,-0.32,-0.91,1.37,0.64,0.54,-0.05,-0.17,-0.27,0.01,1.11,0.48,-0.35,0.02,0.18,-0.33,0.59,0.63,-0.01,-0.03,0.9,0.12,0.18,0.27,-0.07,-0.1,-0.04,0.15,0.46,0.33,1.0,0.66,0.2,-0.09,0.02,0.18,0.25,-0.11,0.02,-0.15,-0.22,0.99,0.11,-0.67,-0.3,-0.28,0.32,0.04,-0.24,0.07,0.32,0.85,0.04,0.07,-0.13,-0.18,-0.49,-0.3,0.41,-0.02,0.11,-0.14,-0.05,0.39,-0.03,0.12,-0.16,0.11,-0.33,-0.34,0.14,0.1,0.35,-0.21,-0.31,0.33,-0.01,-0.16,0.28,0.05,-0.06,0.31,0.35,-0.06,0.17,-0.45,-0.5,-0.15,-0.19,0.04,-0.49,0.97,0.74,-0.15,-0.56,0.13,-0.06,-0.13,-0.29,0.18,-0.16,0.33,0.37,-0.34,-0.33,0.03,0.17,-0.28,0.07,0.38,-0.21,0.13,0.0,-0.15,-0.06,0.15,0.27,1.45,-0.27,-0.89,0.83,-0.13,0.06,0.3,-0.18,-0.31,-0.43,-0.23,-0.19,0.18,-0.01,1.2,0.23,-0.46,0.24,0.15,0.0,-0.19,0.12,0.33,0.08,2.2,0.12,0.31,0.01,-0.01,0.36,0.02,-0.39,-0.18,0.19,0.19,0.43,-0.23,0.83,0.08,1.62,0.1,-0.17,-0.36,-0.36,-0.54,-0.3,-0.14,0.2,0.04,0.11,-0.37,0.55,0.12,0.21,0.32,0.16,0.55,0.19,0.24,0.4,0.12,-0.19,0.02,0.53,0.5,0.7,-0.07,-0.35,-0.22,0.35,-0.2,-0.56,-0.43,0.12,-0.16,-0.15,0.36,0.25,0.38,-0.21,0.31,-0.24,-0.32,-0.21,1.12,0.22,0.36,-0.01,1.12,0.7,-0.1,-0.07,-0.25,-0.22,-0.21,-0.15,-0.34,0.93,-0.38,0.02,-0.19,0.22,0.0,-0.1,0.3,-0.47,-0.02,-0.3,-0.03,-0.24,0.27,-0.25,-0.3,-0.25,-0.55,0.07,0.38,0.47,-0.06,-0.2,-0.08,-0.94,1.2,0.65,-0.07,1.11,0.04,-0.04,-0.28,-0.29,-0.12,0.01,-0.29,0.61,-0.11,0.52,-0.12,0.0,0.63,0.09,0.14,0.02,-0.14,1.63,0.12,-0.2,0.1,0.15,0.19,-0.15,0.48,0.08,0.31,-0.4,-0.29,-0.46,-0.31,-0.07,0.29,0.81,0.05,-0.2,0.47,-0.38,0.06,-0.21,0.74,0.13,1.04,-0.6,-0.06,0.04,0.05,-0.48,-0.14,-0.34,-0.34,0.27,0.01,-0.13,0.01,-0.19,-0.22,0.18,0.03,0.35,-0.02,0.08,0.13,0.16,0.16,0.08,-0.2,0.0,-0.11,-0.3,0.09,1.09,0.22,-0.06,-0.2,-0.15,0.05,0.07,-0.09,-0.28,0.24,0.0,-0.42,-0.22,-0.05,-0.02,0.05,0.01,0.14,-0.16,-0.73,-0.18,-0.22,-0.06,-0.21,-0.48,-0.23,0.32,0.21,0.0,0.13,-0.24,-0.18,-0.02,0.02,0.01,0.13,0.23,1.69,0.08,-0.23,0.18,-0.18,0.13,0.0,-0.54,-0.25,0.2,0.16,0.33,-0.19,-0.27,0.24,-0.18,0.22,-0.25,-0.19,-0.35,-0.16,1.17,-0.34,0.41,0.47,0.39,1.37,0.18,0.01,0.07,0.39,-0.3,1.42,0.13,-0.14,0.23,1.22,0.09,-0.24,0.15,0.12,-0.04,0.22,0.61,-0.22,-0.12,-0.11,0.2,0.29,0.21,-0.26,0.07,-0.19,-0.02,-0.35,0.07,-0.05,0.03,1.64,-0.31,2.12,0.07,-0.02,0.26,-0.42,1.27,-0.06,0.15,0.14,1.2,0.03,0.07,-0.1,0.33,0.1,0.23,-0.1,-0.01,0.24,0.24,-0.14,0.01,0.32,-0.13,-0.36,0.0,0.02,0.74,0.08,0.09,0.73,-0.12,-0.13,0.41,-0.28,0.52,-0.26,0.12,0.33,0.55,-0.05,0.08,-0.47,0.37,-0.15,-0.2,-0.23,-0.24,-0.22,-0.42,-0.26,0.36,0.67,0.39,-0.16,-0.28,0.22,0.26,-0.28,-0.26,0.06,0.18,-0.19,1.15,-0.22,-0.04,0.0,1.19,-0.45,-0.08,0.23,-0.4,0.03,0.33,-0.27,0.49,-0.25,0.13,-0.08,0.24,-0.2,0.31,0.35,-0.18,-0.27,0.42,0.92,-0.34,-0.35,0.06,-0.08,-0.29,-0.12,-0.23,0.15,-0.06,-0.01,0.07,-0.11,0.24,0.28,1.47,0.4,0.28,0.24,-0.03,-0.2,0.02,0.45,0.08,1.72,0.13,0.64,0.22,0.14,0.23,0.05,0.04,0.2,0.01,1.15,0.04,-0.53,0.05,-0.22,-0.02,0.11,-0.06,0.36,0.02,0.22,0.71,0.29,-0.17,-0.33,-0.07,-0.03,-0.26,0.18,-0.35,-0.1,1.31,0.15,0.42,1.22,-0.23,0.36,-0.15,-0.04,-0.59,0.06,-0.36,0.05,-0.36,-0.11,-0.58,-0.19,-0.17,0.07,-0.91,0.32,0.48,-0.07,-0.57,-0.33,0.2,-0.32,0.33,0.11,-0.21,0.51,0.09,-0.21,0.08,0.57,-0.35,-0.03,-0.04,-0.02,0.09,-0.12,-0.14,-0.09,0.23,0.0,-0.35,0.22,0.08,0.11,-0.13,0.08,-0.03,-0.14,-0.84,0.29,0.13,-0.18,-0.25,-0.37,-0.24,-0.07,-0.6,-0.39,0.29,0.11,0.31,-0.2,-0.32,-0.27,-0.21,-0.24,0.37,-0.17,-0.38,0.95,-0.5,-0.15,-0.45,0.26,1.14,-0.33,0.31,0.09,-0.15,0.15,0.22,1.52,0.04,1.2,0.41,0.55,0.04,1.21,0.0,0.01,0.09,-0.17,0.95,-0.41,-0.32,-0.06,0.12,-0.21,-0.28,-0.68,-0.33,0.82,0.09,-0.14,0.02,0.17,-0.21,-0.07,0.19,0.11,-0.2,0.09,0.21,-0.17,0.22,-0.05,-0.25,-0.09,0.61,-0.37,0.25,0.86,-0.26,-0.46,-0.17,0.37,0.19,-0.64,-0.32,0.12,0.96,0.08,-0.04,-0.6,-0.22,0.01,-0.36,-0.41,0.11,0.25,-0.08,1.44,0.23,-0.32,-0.08,-0.12,0.0,0.16,0.19,0.64,0.01,0.48,0.21,-0.25,0.18,1.13,-0.14,-0.62,0.61,-0.14,-0.08,-0.32,0.04,0.47,0.15,0.14,-0.01,0.24,-0.01,-0.05,0.05,0.37,-0.32,-0.38,-0.28,0.03,0.2,-0.31,0.35,0.06,0.2,0.25,-0.39,-0.29,0.44,0.05,-0.25,0.11,-0.35,-0.31,-0.03,-0.02,-0.32,0.14,0.07,0.04,-0.12,0.22,-0.41,0.0,0.2,-0.46,-0.01,1.29,0.1,0.23,-0.33,0.08,0.04,0.76,-0.16,-0.19,-0.01,-0.34,-0.26,0.6,0.04,-0.35,-0.09,-0.25,-0.92,1.05,0.05,0.18,-0.03,1.28,-0.12,0.95,-0.18,0.05,0.2,-0.19,0.01,-0.23,0.32,0.9,0.2,-0.29,-0.22,0.07,-0.18,0.01,0.64,0.49,-0.1,-0.76,0.42,-0.2,-0.09,0.18,0.1,0.14,2.13,-0.03,-0.16,0.33,0.02,-0.07,0.19,-0.19,-0.42,0.03,-0.07,0.35,0.14,-0.17,0.98,-0.07,-0.18,0.35,-0.14,-0.58,0.03,-0.2,-0.12,-0.38,0.28,1.61,0.26,-0.5,0.72,0.37,0.45,-0.27,-0.29,0.05,-0.3,0.35,-0.2,0.21,0.15,0.25,0.09,-0.18,-0.21,0.25,0.0,-0.02,0.49,-0.3,0.05,-0.08,-0.19,0.03,-0.34,-0.12,0.34,-0.35,0.43,-0.16,0.04,-0.1,0.06,-0.3,0.48,-0.09,0.03,0.2,-0.11,0.06,0.07,-0.08,0.02,-0.05,-0.38,-0.32,0.19,-0.23,-0.2,-0.23,-0.42,0.62,0.16,0.0,-0.33,-0.16,-0.35,-0.07,-0.29,-0.14,-0.32,-0.21,-0.14,-0.67,0.17,0.2,0.05,-0.19,0.24,-0.12,-0.03,-0.45,0.48,0.0,0.04,-0.04,0.09,0.69,-0.36,0.16,0.32,0.23,-0.24,-0.14,-0.23,0.33,-0.31,0.14,0.22,0.32,0.37,0.18,0.18,0.07,-0.41,0.03,0.44,0.02,0.05,1.15,0.09,-0.4,-0.4,1.24,0.39,-0.1,0.1,0.64,0.08,0.52,-0.02,-0.24,-0.37,0.17,0.26,-0.43,0.0,-0.01,-0.15,-0.46,-0.08,0.2,1.35,-0.08,-0.13,0.54,-0.11,-0.39,0.16,-0.16,-0.09,0.12,-0.15,0.26,-0.37,-0.51,-0.13,-0.38,-0.05,0.19,0.09,0.15,-0.21,0.05,0.04,-0.24,-0.2,0.42,0.46,0.0,-0.4,1.1,0.09,-0.45,-0.25,-0.03,-0.21,0.15,-0.14,-0.23,-0.07,0.21,-0.18,0.08,-0.01,0.01,-0.32,0.66,-0.77,1.24,-0.35,0.68,-0.14,0.5,-0.23,-0.07,-0.13,-0.16,-0.08,-0.36,-0.15,0.05,-0.44,0.23,0.44,0.41,0.12,0.08,-0.18,-0.13,-0.52,-0.01,-0.04,0.23,0.03,0.11,0.58,-0.13,0.55,0.1,-0.19,0.9,0.02,-0.12,0.95,-0.17,0.07,0.3,-0.07,0.03,0.29,1.54,0.07,-0.14,-0.14,-0.23,0.08,0.04,-0.04,1.51,0.37,0.12,0.39,-0.09,-0.2,0.54,0.03,-0.63,0.23,0.29,-0.19,-0.18,-0.15,0.14,0.58,0.3,-0.2,-0.03,-0.26,0.1,0.17,0.02,-0.39,0.35,-0.26,0.07,-0.18,0.05,-0.05,-0.23,-0.1,0.05,0.14,1.47,-0.41,0.12,0.05,-0.23,0.98,-0.14,0.14,-0.43,-0.16,-0.42,0.01,0.22,0.15,-0.3,-0.35,0.15,0.14,-0.44,-0.03,0.33,-0.19,0.01,0.41,-0.13,0.6,-0.29,0.3,0.47,0.38,0.39,-0.31,2.24,0.28,-0.34,-0.33,-0.1,0.41,-0.49,0.11,-0.23,0.03,-0.21,-0.5,-0.14,-0.08,0.02,0.07,0.4,0.39,0.2,0.66,-0.38,-0.05,0.04,-0.05,-0.23,0.22,0.16,-0.64,0.42,0.18,0.11,-0.18,-0.11,0.38,-0.42,-0.16,0.48,0.23,1.77,0.02,-0.4,0.08,-0.11,0.41,-0.08,0.27,-0.17,-0.02,0.08,-0.24,0.26,0.11,0.06,0.15,-0.17,0.45,-0.21,-0.11,-0.09,0.38,0.84,-0.39,0.31,-0.33,-0.18,-0.06,-0.15,-0.01,-0.31,0.49,-0.28,-0.09,0.58,-0.48,-0.16,-0.3,0.17,0.31,-0.17,0.14,0.18,0.24,-0.21,-0.36,-0.1,-0.11,-0.82,2.76,-0.15,-0.16,0.01,0.52,-0.1,0.18,0.1,-0.15,0.11,-0.05,-0.13,0.14,-0.39,-0.13,-0.05,0.22,-0.38,-0.37,0.34,0.46,0.13,0.07,0.08,-0.02,0.0,-0.46,-0.45,0.08,0.06,0.21,-0.06,-0.17,0.23,-0.13,1.08,0.14,0.33,0.66,0.26,0.48,0.32,-0.16,-0.12,-0.02,0.04,0.77,-0.22,0.08,-0.02,0.0,0.39,0.1,-0.51,-0.03,-0.2,-0.12,0.29,0.01,1.38,0.23,0.0,0.17,0.09,-0.24,-0.12,0.2,0.08,-0.3,-0.08,0.8,-0.44,-0.37,0.82,0.35,-0.04,-0.05,0.09,-0.41,0.14,-0.19,0.16,0.14,0.69,0.07,0.12,-0.05,-0.16,0.22,0.33,0.01,-0.39,-0.15,0.22,0.31,0.25,1.58,-0.41,-0.72,-0.13,-0.09,0.07,0.35,-0.34,0.16,0.34,-0.19,0.32,-0.17,0.07,-0.03,0.68,-0.04,0.19,0.15,0.48,1.56,0.28,0.56,0.12,0.26,-0.4,-0.07,0.02,0.13,-0.07,0.15,0.11,-0.32,1.07,-0.42,0.04,-0.62,-0.06,0.04,1.04,0.65,-0.41,0.26,0.23,-0.23,0.71,1.32,0.04,0.27,0.18,-0.15,0.3,-0.21,-0.24,0.01,0.05,-0.22,-0.55,-0.37,-0.24,0.88,0.07,0.17,-0.15,-0.56,-0.34,0.74,-0.27,-0.16,0.08,0.51,-0.53,-0.13,-0.18,1.08,-0.2,-0.14,1.2,0.84,-0.26,0.01,-0.14,0.13,0.18,-0.19,0.1,-0.24,-0.13,-0.56,0.27,0.22,-0.2,0.31,0.26,-0.04,0.26,0.39,1.43,-0.18,0.33,-0.03,-0.34,0.55,0.06,0.31,0.01,-0.27,0.22,-0.01,0.15,-0.37,-0.53,0.03,0.08,0.06,0.15,0.63,0.06,0.75,0.08,-0.3,-0.32,-0.26,0.04,-0.06,-0.11,-0.08,-0.26,0.13,-0.01,-0.52,0.25,0.01,-0.17,0.27,-0.1,0.28,0.36,-0.42,0.87,-0.3,-0.13,0.29,0.11,0.4,-0.02,-0.29,0.78,-0.02,0.46,0.25,-0.15,0.13,-0.08,0.44,0.39,0.46,0.4,0.31,-0.2,0.16,0.18,0.05,-0.16,-0.29,0.35,-0.1,0.25,-0.36,-0.28,-0.05,0.03,0.06,-0.25,-0.07,-0.1,-0.05,-0.25,-0.26,-0.17,-0.08,-0.27,0.27,0.34,0.13,-0.03,-0.34,0.35,0.57,-0.55,-0.05,0.66,-0.47,-0.05,-0.02,-0.01,0.76,0.31,-0.28,0.25,0.07,-0.22,-0.2,0.45,-0.02,0.11,-0.31,-0.35,-0.1,-0.19,0.35,-0.13,0.34,-0.11,0.38,0.29,-0.03,0.36,0.86,-0.06,0.26,0.36,-0.38,0.02,-0.01,-0.29,0.43,0.41,-0.02,0.05,-0.18,-0.35,-0.28,-0.09,0.15,0.13,1.04,-0.36,0.09,0.53,-0.41,-0.47,0.42,0.1,-0.1,0.08,0.61,0.05,0.13,-0.22,0.8,0.01,1.38,-0.28,-0.59,0.01,-0.16,-0.39,-0.15,-0.13,0.11,0.39,-0.4,-0.09,-0.27,0.14,-0.2,0.24,0.31,0.55,-0.27,-0.37,-0.29,0.6,-0.23,-0.29,0.6,-0.15,-0.23,-0.19,-0.1,-0.08,0.35,0.6,0.44,-0.44,0.31,-0.05,0.14,-0.1,-0.74,-0.29,-0.3,-0.32,0.27,-0.38,-0.02,0.02,-0.37,-0.08,0.66,-0.37,0.45,0.12,-0.29,1.36,-0.23,0.0,-0.65,0.04,1.47,-0.17,0.14,0.03,-0.04,-0.35,-0.22,0.02,-0.02,-0.21,-0.19,0.1,-0.52,-0.58,0.47,-0.03,0.61,0.24,0.1,-0.23,-0.07,0.39,-0.04,-0.41,-0.2,1.44,0.12,-0.12,-0.18,0.38,-0.17,0.25,-0.19,0.06,-0.18,-0.21,0.0,0.17,0.29,-0.19,0.2,-0.11,-0.16,-0.19,-0.23,0.38,-0.01,0.45,0.27,0.22,0.14,0.32,-0.3,-0.45,0.22,0.18,0.28,0.16,-0.35,0.03,-0.12,1.01,-0.12,0.36,0.09,2.03,0.19,-0.36,-0.36,-0.03,-0.01,0.24,1.49,0.09,-0.13,-0.18,0.09,-0.57,0.25,-0.12,-0.13,0.04,-0.14,0.06,0.53,-0.38,-0.06,0.95,0.23,0.2,-0.04,-0.14,0.26,-0.25,-0.04,-0.34,0.96,0.47,0.25,0.09,-0.23,0.07,-0.2,0.9,-0.36,-0.36,0.44,-0.28,-0.07,-0.15,-0.19,0.23,-0.42,-0.27,0.17,-0.03,0.08,1.48,0.49,-0.08,0.15,0.01,0.41,-0.18,0.44,0.47,0.3,0.21,0.36,0.24,-0.22,0.15,-0.09,0.09,-0.15,0.25,-0.06,-0.7,-0.18,0.32,0.17,-0.06,0.15,0.2,0.04,-0.66,0.05,0.04,1.24,-0.3,0.0,0.36,-0.35,-0.01,-0.12,-0.31,0.2,-0.08,0.11,-0.24,-0.4,-0.35,-0.11,-0.33,-0.35,-0.26,-0.19,0.08,0.17,0.05,-0.53,-0.04,-0.02,0.0,-0.44,-0.44,0.16,-0.3,0.2,-0.09,0.14,0.06,-0.45,0.0,0.1,0.2,0.06,0.02,-0.21,0.37,-0.47,0.19,0.32,0.98,0.15,0.06,0.18,-0.26,0.03,0.28,0.16,-0.03,0.05,0.74,-0.03,0.47,-0.17,0.4,-0.43,-0.15,-0.14,-0.35,0.73,-0.02,-0.08,-0.39,-0.1,0.03,-0.2,-0.21,-0.16,-0.63,-0.1,-0.09,0.0,0.02,0.82,0.12,-0.48,0.02,0.75,0.15,0.27,-0.03,-0.34,0.51,-0.3,-0.19,0.67,-0.2,0.0,-0.07,0.04,-0.11,0.04,0.12,0.16,0.25,0.55,0.17,-0.23,-0.11,-0.4,-0.42,-0.2,0.07,0.06,0.08,-0.14,0.07,-0.34,0.93,0.41,-0.37,-0.11,1.16,-0.23,0.11,0.18,0.08,0.19,-0.26,0.22,-0.44,-0.1,-0.17,-0.19,0.07,0.49,-0.28,0.05,-0.54,-0.41,-0.09,-0.1,-0.17,0.07,-0.01,-0.07,0.38,0.11,0.53,0.02,0.19,-0.17,0.11,-0.24,0.24,1.18,-0.28,-0.09,0.37,0.3,0.25,-0.11,-0.11,-0.75,-0.47,0.26,-0.4,-0.53,0.19,0.27,-0.05,-0.25,0.08,-0.08,-0.05,-0.37,0.12,-0.07,-0.19,-0.06,-0.17,0.16,0.25,-0.17,-0.17,0.37,0.09,-0.02,0.26,0.63,0.02,-0.42,-0.16,-0.24,0.16,-0.13,-0.16,0.23,0.21,0.59,-0.49,-0.25,-0.04,0.09,-0.19,0.46,-0.58,0.33,1.36,-0.05,-0.05,0.0,0.49,0.39,-0.2,0.08,-0.17,-0.3,0.08,0.32,-0.14,-0.23,-0.25,-0.06,0.34,-0.17,0.22,-0.13,-0.11,-0.14,-0.14,0.74,-0.28,-0.48,0.23,0.26,-0.48,0.74,-0.1,0.1,-0.32,-0.22,1.8,0.57,0.59,-0.2,-0.14,0.09,-0.28,-0.03,-0.4,-0.17,-0.12,0.09,0.35,-0.1,0.0,-0.23,-0.1,-0.07,2.13,0.64,-0.05,-0.13,-0.13,1.27,0.7,0.05,-0.44,0.15,-0.1,0.07,0.55,-0.28,-0.09,-0.27,0.08,-0.12,0.03,-0.24,-0.18,0.05,-0.14,1.46,0.04,0.17,-0.25,-0.4,0.47,0.18,0.37,0.1,0.75,-0.13,0.18,0.84,0.01,-0.05,-0.46,0.47,0.32,-0.21,-0.23,0.08,0.23,0.35,0.08,-0.07,0.27,-0.3,1.17,0.36,1.16,0.31,0.02,0.51,0.27,0.05,0.31,-0.04,-0.01,-0.23,0.31,-0.37,-0.32,0.13,-0.02,-0.35,0.01,0.86,-0.01,0.05,0.22,0.74,0.48,-0.32,-0.02,-0.5,-0.5,-0.07,0.07,0.11,0.21,-0.03,0.09,-0.38,-0.16,0.14,0.09,-0.46,0.18,0.19,-0.25,-0.56,0.1,-0.11,-0.24,-0.25,0.14,-0.14,-0.26,-0.19,-0.01,1.04,-0.14,-0.03,-0.22,-0.11,-0.07,-0.17,0.29,-0.02,0.63,0.04,1.95,0.02,0.22,-0.38,-0.13,-0.18,-0.2,-0.32,-0.09,0.04,-0.09,-0.05,0.08,-0.08,0.63,0.08,-0.22,-0.29,-0.35,-0.24,-0.36,0.9,0.3,-0.21,-0.16,0.29,0.03,-0.22,0.09,-0.39,-0.02,-0.52,0.32,0.04,0.69,-0.05,0.02,-0.21,0.09,0.26,0.03,0.04,0.01,-0.36,-0.31,0.23,-0.21,0.01,-0.18,-0.02,-0.17,-0.03,-0.33,0.49,0.14,1.29,-0.4,-0.02,-0.06,-0.19,0.15,-0.01,0.44,0.37,-0.18,-0.09,-0.03,0.13,-0.08,-0.05,0.25,0.83,-0.49,0.12,0.53,-0.07,0.2,-0.08,-0.04,-0.07,0.03,-0.18,-0.03,0.09,0.18,0.62,0.58,0.19,-0.04,-0.15,0.2,0.67,-0.13,0.18,-0.18,-0.21,0.11,-0.11,-0.18,-0.34,-0.06,0.06,0.04,-0.4,-0.35,0.26,0.06,1.59,-0.36,-0.23,-1.05,-0.28,0.08,0.37,-0.13,0.3,1.24,-0.31,-0.63,1.25,-0.33,0.2,-0.31,-0.36,0.19,0.19,-0.2,-0.13,0.24,0.81,-0.1,0.05,-0.12,-0.16,-0.4,0.42,0.09,0.57,-0.57,0.24,-0.19,-0.14,0.1,-0.6,0.49,0.17,-0.14,0.11,0.05,-0.19,-0.15,-0.07,0.05,-0.25,-0.08,0.07,0.0,0.29,-0.38,0.08,-0.18,-0.18,-0.1,0.33,-0.35,0.76,0.11,-0.03,0.21,-0.45,0.37,0.16,0.02,0.34,0.07,-0.07,-0.06,-0.07,0.16,1.6,0.39,-0.25,0.5,3.28,-0.07,0.2,-0.48,0.1,-0.3,0.35,-0.15,-0.05,0.71,-0.16,-0.01,0.14,0.34,0.16,0.03,0.28,0.26,0.04,0.16,0.05,-0.05,-0.12,0.01,-0.2,-0.15,-0.2,-0.07,0.01,0.69,0.16,0.02,-0.19,-0.18,-0.11,-0.06,0.8,-0.28,0.77,-0.13,-0.04,0.39,0.01,-0.17,0.41,0.15,-0.15,-0.12,0.05,-0.36,0.53,-0.07,0.1,0.05,0.35,0.33,-0.27,-0.13,0.03,-0.07,0.01,0.12,0.0,0.16,-0.16,0.19,-0.22,0.1,-0.42,0.73,0.09,-0.33,0.14,0.31,0.12,-0.25,0.97,-0.18,0.45,0.1,-0.03,1.59,0.15,-0.12,0.36,-0.03,-0.03,-0.05,0.06,-0.26,0.43,-0.55,0.01,0.34,-0.13,0.18,-0.38,-0.62,0.11,0.4,-0.19,0.05,-0.12,0.51,-0.11,-0.24,-0.39,-0.32,0.37,-0.38,0.23,0.73,0.24,-0.16,-0.42,-0.28,0.08,-0.04,0.16,0.9,-0.05,1.06,0.23,0.36,0.17,-0.37,-0.27,0.21,0.23,0.38,-0.52,0.44,0.0,-0.02,0.25,-0.39,0.61,-0.31,0.37,-0.13,-0.01,-0.31,-0.4,-0.3,-0.22,0.24,-0.02,0.2,0.57,-0.31,-0.02,-0.49,-0.03,-0.17,0.06,-0.21,0.19,-0.33,0.06,0.17,0.17,-0.27,0.45,0.29,0.27,-0.33,1.36,0.25,0.14,0.06,0.16,-0.15,-0.14,0.56,0.88,-0.02,-0.16,-0.27,0.1,0.05,0.0,0.5,-0.11,-0.29,0.04,-0.26,-0.16,-0.01,0.24,-0.24,0.24,-0.26,0.32,0.12,0.21,-0.08,0.2,-0.16,-0.13,1.06,0.01,-0.01,0.21,0.17,0.07,-0.2,0.01,-0.26,-0.29,0.04,0.33,-0.04,0.25,-0.63,0.8,1.21,0.25,0.02,0.17,-0.31,-0.14,-0.05,0.2,0.09,0.36,0.42,-0.02,1.37,-0.22,0.33,-0.36,-0.32,-0.19,0.21,0.28,-0.35,-0.39,0.47,0.11,0.09,0.64,0.38,0.14,0.23,0.84,0.1,-0.19,0.41,-0.18,-0.19,0.39,-0.29,-0.32,-0.04,0.17,-0.18,0.42,0.4,0.04,-0.19,0.35,0.2,0.3,0.47,-0.08,-0.09,0.24,0.17,0.05,-0.33,0.22,-0.72,-0.05,-0.27,-0.38,-0.17,0.34,0.26,-0.33,-0.18,-0.1,0.36,0.32,0.07,-0.14,0.26,0.25,0.15,-0.05,0.13,0.39,-0.29,-0.07,0.16,0.75,-0.02,-0.02,1.59,-0.09,0.1,-0.22,-0.22,-0.01,-0.61,-0.23,-0.32,-0.09,0.01,-0.81,-0.72,-0.28,-0.16,0.75,-0.07,-0.05,0.36,-0.28,-0.27,-0.22,0.35,-0.14,-0.13,0.17,0.49,0.24,0.22,-0.23,-0.22,1.66,0.03,0.09,0.16,-0.08,-0.23,-0.01,-0.01,-0.25,0.03,0.05,0.21,-0.06,-0.39,0.27,-0.08,-0.04,-0.27,-0.24,0.24,0.02,-0.18,-0.07,0.02,-0.03,-0.7,0.11,-0.22,-0.18,0.16,0.0,-0.37,-0.01,-0.14,-0.24,0.03,0.15,-0.07,0.04,-0.01,-0.05,-0.11,0.14,0.24,-0.21,0.38,0.0,-0.21,0.31,0.45,0.54,-0.32,0.21,0.17,-0.16,-0.02,0.28,-0.06,-0.06,-0.19,0.43,0.05,0.39,0.08,0.42,0.08,0.11,0.89,-0.25,-0.02,-0.17,0.11,-0.42,0.44,-0.41,-0.31,-0.18,0.14,0.06,0.22,0.46,-0.18,0.11,0.18,0.9,1.98,-0.27,-0.4,0.08,0.38,-0.17,0.38,-0.61,0.04,0.1,-0.16,-0.22,0.03,0.17,1.18,0.42,0.01,-0.24,0.17,-0.25,-0.54,-0.06,0.05,-0.03,0.04,0.16,0.04,0.63,0.12,-0.34,-0.13,-0.16,0.06,0.75,-0.07,1.45,-0.13,-0.74,0.54,-0.37,-0.24,-0.03,-0.03,0.16,-0.24,0.75,-0.07,-0.15,0.38,-0.07,0.35,-0.16,0.28,-0.04,0.03,0.36,-0.28,-0.03,0.51,-0.49,0.58,0.22,-0.39,-0.2,-0.01,-0.02,-0.01,-0.31,-0.59,0.31,0.16,0.41,1.08,0.22,-0.21,-0.19,-0.76,0.33,-0.09,1.16,0.37,-0.16,-0.15,-0.28,0.2,-0.21,0.09,0.01,0.25,0.44,1.61,-0.21,-0.13,0.56,-0.05,-0.13,-0.24,-0.07,-0.11,-0.08,0.04,0.3,0.03,0.13,0.12,0.0,0.04,0.78,0.02,-0.1,0.29,-0.11,-0.05,-0.22,0.28,-0.35,-0.2,0.33,0.34,-0.28,0.0,-0.11,-0.12,0.22,-0.27,-0.18,0.09,1.4,-0.11,-0.01,0.31,0.29,0.07,-0.13,-0.31,-0.04,-0.03,-0.22,0.1,0.19,0.1,-0.09,-0.17,-0.59,-0.22,0.21,0.14,0.04,-0.04,-0.32,-0.34,0.27,-0.11,0.82,-0.2,-0.1,0.1,0.71,0.15,-0.16,0.06,0.53,-0.04,-0.62,0.26,1.56,-0.47,-0.3,-0.09,-0.36,0.47,-0.41,0.04,0.04,1.38,-0.06,0.09,0.11,0.08,-0.14,0.54,0.21,0.2,-0.36,-0.12,0.32,0.07,-0.02,0.02,0.03,0.21,1.48,0.4,-0.32,-0.25,-0.05,0.78,-0.03,-0.24,-0.22,0.14,0.5,-0.25,0.12,-0.03,-0.16,0.04,0.23,-0.29,1.54,-0.24,0.4,-0.53,0.54,0.15,-0.33,-0.02,-0.14,0.0,0.04,-0.11,-0.25,0.03,0.03,0.11,-0.23,0.29,-0.41,0.01,0.3,-0.16,0.36,0.22,1.15,0.76,-0.41,-0.3,0.2,-0.13,-0.18,-0.06,1.8,0.05,0.2,0.03,0.07,0.18,-0.1,1.13,0.44,1.66,-0.01,-0.46,-0.31,0.0,-0.21,-0.14,0.05,-0.01,0.66,0.98,0.08,0.36,0.06,-0.33,-0.08,-0.28,0.09,0.09,-0.35,-0.39,-0.16,-0.28,-0.08,0.12,0.01,0.13,0.64,0.67,-0.31,0.13,-0.17,-0.14,0.2,0.1,-0.44,-0.11,0.25,0.2,1.29,0.14,-0.18,-0.13,-0.41,1.81,0.0,-0.16,0.22,-0.19,0.54,0.04,-0.15,0.2,-0.08,-0.12,-0.03,-0.68,-0.27,-0.05,-0.99,-0.09,-0.2,0.22,-0.16,0.14,0.19,0.83,0.06,0.27,-0.22,-0.17,-0.09,-0.17,-0.82,-0.05,1.56,0.22,0.38,-0.63,-0.9,-0.15,0.02,-0.12,-0.51,-0.04,0.09,0.93,0.1,0.26,-0.59,0.15,0.02,0.53,0.02,0.48,-0.21,0.16,-0.24,-0.08,0.05,-0.01,-0.24,0.19,-0.06,-0.22,0.45,0.32,0.13,-0.35,-0.01,0.25,-0.09,-0.2,1.28,-0.17,-0.09,0.26,0.0,0.11,-0.06,0.54,0.46,-0.12,0.12,1.37,0.06,-0.17,-0.15,-0.12,-0.67,0.07,-0.33,0.13,-0.53,0.26,-0.06,-0.75,0.2,-0.15,0.09,1.58,0.55,-0.2,0.39,-0.31,-0.36,0.15,0.13,-0.27,0.28,0.63,0.13,0.47,-0.05,-0.27,0.29,0.14,-0.17,-0.82,0.14,0.05,-0.17,0.06,0.16,0.12,0.51,-0.02,-0.21,0.28,0.04,-0.14,-0.23,0.29,0.24,-0.25,-0.18,-0.06,-0.15,-0.15,0.0,0.04,0.54,0.0,-0.14,0.18,-0.32,-0.1,0.27,0.27,-1.3,-0.32,0.3,-0.2,1.75,0.04,0.08,0.71,0.1,-0.34,-0.48,0.33,0.64,-0.51,0.17,-0.03,-0.36,-0.1,-0.28,0.02,-0.14,0.12,1.24,0.26,0.24,2.0,-0.36,0.09,0.2,0.4,-0.22,-0.31,0.3,-0.14,-0.07,0.53,0.44,-0.77,0.19,-0.19,-0.41,0.21,-0.05,0.4,-0.27,0.66,0.09,2.13,-0.61,0.47,0.71,0.13,0.02,0.15,-0.32,0.85,0.23,-0.28,0.95,0.15,-0.09,-0.43,-0.08,0.51,0.28,-0.26,-0.1,1.31,-0.18,0.05,0.36,1.16,-0.09,0.79,-0.13,-0.29,0.08,-0.31,0.01,-0.06,-0.51,-0.09,-0.17,0.41,-0.23,0.2,0.34,-0.4,0.02,0.12,-0.02,0.2,0.01,-0.3,0.04,0.03,-0.13,0.25,-0.15,-0.38,1.2,-0.04,-0.04,0.2,0.1,-0.58,-0.21,-0.12,0.42,-0.03,-0.87,-0.3,-0.05,-0.09,-0.27,0.15,0.28,-0.31,1.33,-0.35,-0.22,-0.5,0.59,0.43,-0.36,0.5,-0.21,0.38,-0.48,-0.18,0.09,0.56,-0.07,-0.29,-0.42,-0.1,-0.2,-0.65,-0.31,0.23,0.16,-0.13,0.1,-0.01,-0.02,-0.02,-0.02,0.15,0.09,0.17,-0.11,0.07,0.12,-0.14,-0.09,-0.12,-0.08,-0.31,0.29,0.55,0.24,0.22,-0.17,0.02,0.08,-0.09,-0.06,0.26,-0.18,-0.32,-0.07,1.67,0.43,-0.03,-0.3,0.18,-0.49,0.32,-0.27,0.39,-0.07,-0.49,0.04,-0.32,0.07,-0.35,0.06,0.64,0.11,0.12,-0.01,0.13,0.77,-0.2,-0.07,0.06,-0.08,-0.11,0.13,-0.19,0.62,1.78,-0.2,-0.47,0.61,-0.53,0.03,0.04,0.54,0.56,-0.02,0.2,0.29,0.31,-0.23,-0.02,-0.09,-0.36,-0.23,-0.05,-0.14,0.37,0.03,-0.21,0.92,0.17,-0.32,-0.24,0.02,-0.09,-0.13,-0.31,-0.02,0.24,-0.27,-0.26,0.33,-0.24,0.14,-0.39,0.07,0.1,0.14,-0.18,-0.04,0.24,-0.49,-0.24,0.39,-0.33,0.1,0.14,0.31,1.26,-0.1,0.22,-0.01,0.4,0.07,-0.96,0.13,0.36,0.73,0.01,-0.54,-0.03,-0.22,0.31,-0.04,0.91,0.37,-0.33,0.51,-0.28,-0.02,-0.37,-0.3,-0.12,0.37,0.24,0.04,0.4,0.8,-0.11,0.12,0.34,0.02,-0.2,0.08,-0.07,0.56,0.1,1.53,0.79,0.45,-0.08,-0.34,0.55,-0.03,-0.2,0.2,-0.35,-0.32,-0.07,0.81,0.35,1.61,0.16,-0.03,1.26,0.15,-0.31,-0.52,-0.05,0.47,0.3,0.01,1.35,-0.15,0.17,-0.26,0.02,-0.24,0.19,0.33,0.42,0.11,0.36,-0.23,-0.38,-0.2,-0.05,-0.43,-0.15,0.71,-0.07,-0.13,-0.13,-0.3,0.05,-0.36,0.87,-0.44,-0.19,-0.8,-0.1,0.38,0.42,-0.26,-0.05,0.29,0.17,2.08,0.69,-0.27,0.14,-0.16,0.27,0.16,0.08,-0.08,0.18,0.01,-0.16,0.47,-0.44,0.44,-0.11,0.25,0.65,-0.16,1.68,0.48,0.67,-0.15,0.45,-0.12,-0.15,-0.16,-0.13,0.71,0.18,-0.07,-0.03,0.01,-0.22,-0.05,-0.28,0.13,-0.54,0.33,1.04,0.3,-0.29,0.03,-0.51,0.29,-0.1,-0.14,0.33,-0.17,0.33,-0.03,0.05,-0.14,-0.05,0.07,-0.2,0.26,-0.14,-0.05,0.08,0.46,0.55,-0.19,0.01,0.26,0.02,0.54,-0.17,-0.32,-0.29,-0.2,0.24,0.08,-0.37,0.29,0.07,0.59,-0.28,0.66,-0.15,1.43,0.03,-0.04,0.04,1.43,0.55,0.58,-0.16,0.03,0.37,-0.23,-0.29,-0.28,0.09,0.41,-0.26,2.25,0.58,-0.53,0.39,-0.52,0.05,0.41,-0.37,0.31,-0.14,0.09,-0.43,-0.11,-0.05,-0.06,1.07,0.17,1.09,-0.17,0.03,0.25,-0.05,0.93,0.16,-0.14,0.48,0.09,-1.26,-0.23,0.23,-0.15,0.38,-0.24,0.14,-0.01,-0.04,0.03,0.14,-0.08,-0.39,-0.02,0.47,0.39,-0.13,-0.46,-0.41,-0.65,0.34,0.64,0.23,-0.38,-0.37,0.11,-0.29,-0.32,0.43,-0.18,0.36,-0.12,0.6,-0.13,-0.06,-0.19,0.13,0.06,0.15,0.05,0.16,0.0,0.51,-0.07,0.25,-0.06,-0.16,-0.13,0.01,0.57,0.3,-0.16,0.0,-0.27,0.52,0.84,-0.27,0.07,0.07,-0.09,0.16,0.25,-0.58,-0.49,-0.22,0.15,0.58,0.3,-0.05,-0.29,0.61,0.2,-0.04,-0.27,0.12,0.12,-0.09,-0.22,-0.17,-0.07,2.27,-0.09,0.11,-0.57,-0.52,0.63,0.07,0.02,-0.39,1.24,-0.06,-0.04,1.22,0.14,-0.08,0.16,0.05,0.02,0.05,0.11,-0.48,0.15,-0.25,1.06,-0.22,-0.26,-0.14,0.25,1.39,-0.3,-0.59,0.4,-0.16,-0.04,1.53,0.65,0.07,0.37,0.39,0.01,0.14,0.82,-0.37,0.2,-0.15,-0.27,0.34,0.13,-0.26,-0.06,0.07,-0.43,-0.03,-0.04,0.0,-0.03,0.05,-0.07,-0.08,-0.81,-0.43,0.41,-0.04,-0.13,0.53,-0.16,-0.16,0.11,-0.13,-0.24,1.12,-0.53,-0.21,-0.23,0.15,0.33,0.02,0.35,0.45,0.34,-0.08,-0.02,-0.19,-0.55,0.38,-0.05,-0.05,0.03,0.69,-0.27,0.45,-0.34,0.06,-0.04,0.32,0.58,0.03,-0.11,-0.34,-0.16,-0.58,-0.48,-0.07,0.06,0.12,0.34,-0.02,-0.04,-0.27,-0.13,-0.01,-0.56,-0.21,0.13,0.29,-0.23,-0.56,0.16,0.26,0.1,-0.35,-0.39,-0.02,0.07,-0.35,0.21,-0.17,-0.21,0.01,0.84,-0.25,0.28,0.35,-0.17,-0.31,0.03,-0.22,-0.31,0.11,0.28,0.23,0.02,-0.3,0.05,0.42,-0.08,-0.09,-0.25,0.04,0.38,-0.32,0.37,0.09,-0.32,-0.05,0.14,-0.38,0.32,0.05,0.22,-0.09,0.1,-0.02,-0.01,0.03,-0.12,0.11,-0.06,1.98,0.59,-0.51,-0.38,-0.16,-0.21,0.81,-0.45,0.09,-0.06,0.36,0.22,-0.04,-0.4,1.16,-0.49,-0.15,0.56,0.21,0.53,-0.19,-0.36,0.9,0.8,0.4,-0.06,-0.03,0.07,1.02,0.6,0.39,-0.25,-0.13,0.12,-0.09,-0.32,0.13,-0.38,-0.4,-0.06,1.77,-0.25,-0.28,-0.25,-0.11,0.22,-0.37,0.08,0.16,-0.08,0.01,0.65,0.14,-0.05,0.17,0.03,-0.05,1.47,0.0,-0.16,0.48,0.21,-0.42,-0.02,-0.29,0.35,-0.28,-0.31,0.09,0.14,-0.46,-0.29,0.05,-0.18,-0.17,-0.24,-0.01,-0.07,0.32,0.27,-0.29,-0.03,-0.34,0.04,0.37,0.38,-0.21,0.95,-0.1,-0.18,0.41,0.18,-0.17,0.57,-0.3,-0.08,0.08,0.14,0.47,0.0,-0.06,0.49,0.12,-0.21,0.07,-0.1,-0.18,0.0,0.54,0.78,-0.22,-0.02,-0.16,0.55,0.24,-0.04,-0.6,0.05,1.17,0.02,-0.3,-0.33,0.18,0.06,-0.22,-0.46,0.04,0.12,-0.16,-0.38,0.46,0.52,-0.43,1.48,-0.13,-0.3,0.44,-0.01,-0.23,-0.24,-0.16,0.89,-0.1,-0.07,-0.26,0.03,0.27,-0.08,0.13,-0.05,1.4,0.49,-0.19,-0.12,-0.08,0.12,0.41,-0.32,-0.14,0.21,0.47,-0.27,-0.24,0.39,-0.19,-0.35,-0.29,-0.14,-0.01,-0.07,-0.11,0.0,0.05,-0.17,0.9,0.15,-0.01,0.47,-0.01,-0.51,-0.02,-0.14,-0.17,-0.18,0.17,-0.08,-0.19,-0.04,0.29,0.24,-0.59,-0.39,0.09,-0.02,-0.69,-0.22,-0.26,-0.3,0.4,-0.23,0.14,0.97,-0.17,1.44,0.04,-0.34,-0.14,-0.38,-0.11,0.17,0.19,0.02,1.09,0.11,-0.13,-0.1,-0.35,-0.21,-0.19,-0.22,-0.1,0.01,-0.57,0.25,0.03,0.1,-0.52,-0.67,0.2,-0.61,-0.48,0.23,0.43,-0.24,-0.1,0.02,-0.21,-0.1,0.58,-0.32,-0.41,0.0,-0.16,0.01,-0.39,-0.13,-0.06,0.05,-0.28,0.22,0.23,0.16,-0.15,0.16,0.0,-0.06,0.23,-0.1,0.01,-0.12,-0.09,-0.08,0.18,-0.12,0.1,-0.07,0.83,-0.34,-0.01,0.21,-0.11,-0.25,0.01,-0.32,0.41,-0.37,0.11,-0.14,0.34,-0.24,0.02,0.14,0.31,-0.34,-0.29,-0.21,-0.43,-0.11,0.05,1.69,-0.09,0.03,-0.02,0.55,-0.11,-0.36,-0.28,1.35,-0.2,0.02,0.08,0.2,0.02,-0.02,-0.31,-0.27,0.21,0.42,0.12,-0.45,-0.19,0.08,0.16,0.02,-0.09,1.17,-0.15,-0.21,-0.18,-0.28,0.02,-0.45,0.09,-0.03,0.28,0.72,-0.15,-0.24,-0.26,0.08,-0.2,-0.19,0.01,0.13,-0.16,-0.2,0.22,0.08,0.41,0.38,-0.13,-0.08,0.27,-0.43,0.0,-0.39,-0.14,-0.07,-0.54,-0.24,-0.36,-0.01,-0.33,-0.04,0.35,-0.2,-0.35,0.35,-0.37,-0.16,0.84,0.14,0.04,0.16,0.34,0.07,0.16,0.81,0.17,1.64,-0.19,0.21,-0.44,0.63,-0.23,1.14,-0.23,0.57,-0.03,-0.22,0.17,-1.0,0.46,0.34,-0.07,0.03,0.19,-0.09,0.42,-0.07,0.4,-0.12,0.26,0.38,1.23,0.3,0.06,0.15,0.1,0.46,-0.01,0.14,-0.05,0.58,1.51,-0.39,-0.12,0.05,0.63,-0.21,1.11,-0.46,-0.11,-0.3,0.14,1.37,-0.32,0.37,-0.1,-0.06,-0.45,0.53,0.27,-0.06,0.02,-0.3,-0.06,-0.3,0.04,-0.49,-0.18,-0.16,0.21,0.41,-0.25,-0.38,-0.01,-0.09,0.05,0.51,-0.21,0.26,-0.21,-0.17,0.7,-0.16,-0.01,-0.06,0.68,0.09,-0.23,0.81,-0.02,-0.02,0.63,-0.03,-0.01,-0.02,-0.13,0.13,-0.27,0.14,0.47,0.17,0.31,0.22,0.37,0.19,0.26,-0.21,-0.29,0.18,0.1,0.15,0.28,0.1,-0.32,0.1,0.03,-0.39,-0.14,0.05,-0.21,1.16,-0.08,-0.07,0.72,-0.05,-0.35,0.29,-0.08,0.16,0.74,-0.11,-0.3,0.02,-0.13,-0.05,-0.08,-0.46,0.55,-0.18,0.04,0.13,0.35,0.04,-0.08,-0.31,-0.07,-0.02,-0.05,-0.14,0.48,-0.46,0.07,0.04,-0.2,0.17,-0.16,0.07,-0.33,1.02,0.09,-0.28,0.87,0.57,-0.34,0.04,0.1,0.48,0.41,0.05,0.13,0.16,-0.09,-0.24,0.43,0.13,-0.08,-0.12,0.51,-0.56,-0.01,0.06,0.11,0.03,0.07,0.21,0.12,-0.44,0.06,-0.05,0.02,-0.7,-0.19,0.05,-0.05,-0.7,0.15,-0.28,0.59,-0.1,-0.15,0.04,-0.12,0.11,0.24,0.51,0.21,0.21,0.12,0.11,0.11,-0.1,0.07,-0.06,-0.31,0.79,0.27,-0.22,-0.22,0.81,-0.21,-0.05,-0.1,-0.32,-0.04,-0.1,-0.08,0.04,-0.39,-0.25,-0.41,0.75,-0.19,-0.09,-0.5,-0.03,-0.08,-0.28,-0.13,-0.46,0.11,-0.13,-0.24,0.36,-0.39,0.24,0.05,-0.02,0.02,0.08,-0.1,0.2,-0.08,0.04,-0.1,-0.29,0.08,-0.36,0.24,-0.38,0.28,0.09,0.67,-0.08,-0.36,-0.25,-0.36,-0.32,-0.27,0.04,-0.44,-0.04,-0.23,0.03,0.9,0.29,0.06,0.05,0.24,1.85,0.16,0.15,-0.36,0.46,0.27,-0.12,-0.23,0.01,-0.09,-0.72,0.24,0.05,0.19,-0.07,-0.22,-0.23,0.38,0.23,0.05,-0.4,0.14,-0.17,0.38,-0.06,0.25,0.47,-0.28,-0.29,0.03,0.08,-0.03,-0.36,-0.07,1.09,-0.29,0.27,-0.54,0.01,0.02,-0.19,-0.01,0.05,-0.01,0.01,0.17,-0.09,0.25,0.07,-0.28,-0.06,0.26,-0.15,-0.09,-0.19,0.0,0.44,0.11,0.25,-0.33,0.19,-0.15,-0.16,-0.33,0.6,0.4,-0.14,0.55,0.43,0.03,-0.05,-0.35,0.36,-0.16,-0.05,0.32,-0.18,-0.01,-0.21,0.03,0.19,0.21,0.56,0.0,0.06,0.23,-0.23,0.14,0.9,-0.16,-0.07,0.0,0.14,0.16,0.53,0.12,-0.22,1.32,0.39,0.17,-0.95,-0.12,-0.36,-0.51,-0.42,0.08,0.02,0.5,0.08,0.13,-0.09,0.14,0.12,0.96,-0.13,0.63,0.16,-0.28,0.01,-0.11,-0.06,-0.25,0.01,0.31,0.44,-0.31,-0.15,-0.11,-0.15,-0.19,-0.4,-0.16,-0.49,0.18,0.07,0.7,0.69,0.09,0.0,-0.72,0.17,-0.68,0.35,-0.1,0.07,-0.52,-0.34,0.13,-0.04,0.56,-0.17,0.01,0.03,0.18,-0.31,-0.23,-0.04,0.14,-0.14,-0.04,0.5,0.0,-0.08,-0.28,1.14,0.2,0.03,-0.06,-0.05,-0.4,-0.56,0.19,0.38,0.29,-0.24,-0.32,0.09,-0.07,-0.14,0.0,-0.4,0.47,-0.18,0.16,-0.21,-0.16,-0.08,-0.13,0.0,-0.31,0.01,0.11,-0.18,-0.35,0.16,-0.01,-0.46,-0.02,0.37,0.02,0.27,-0.21,1.39,0.08,-0.02,-0.47,0.07,-0.49,-0.52,0.76,0.68,0.45,-0.25,-0.18,-0.25,0.18,-0.26,0.09,-0.52,0.52,0.26,0.08,0.11,0.61,0.28,-0.23,0.42,-0.01,-0.38,1.27,-0.14,-0.08,0.6,0.5,0.13,1.58,-0.33,0.37,0.26,0.13,-0.14,0.56,0.91,0.94,-0.11,-0.08,-0.3,-0.04,-0.11,0.42,0.43,0.32,-0.3,-0.41,0.15,0.15,-0.22,0.03,-0.01,-0.03,0.26,-0.01,0.17,-0.17,0.3,-0.35,-0.37,-0.08,0.0,0.0,-0.43,0.36,-0.3,-0.11,-0.26,2.0,0.11,-0.39,0.15,0.46,0.41,-0.09,-0.52,-0.54,0.57,0.09,0.14,-0.31,0.21,1.34,0.02,0.04,-0.23,0.45,0.02,0.15,0.01,0.0,-0.56,0.01,-0.24,-0.2,0.23,0.08,0.0,-0.07,0.17,-0.36,-0.17,0.84,0.16,0.54,0.96,-0.28,-0.55,-0.59,0.27,0.49,-0.12,-0.26,0.61,-0.03,0.12,0.35,-0.15,-0.2,-0.2,0.45,0.52,0.8,-0.23,-0.48,0.1,0.19,-0.08,0.29,0.1,0.68,0.23,-0.23,0.09,-0.34,-0.08,0.05,0.47,0.15,0.21,0.01,0.04,0.12,0.28,0.31,-0.38,-0.22,-0.1,-0.25,-0.26,0.08,-0.09,0.76,-0.44,0.02,-0.36,-0.36,-0.08,-0.19,-0.17,-0.47,0.89,1.06,-0.15,0.04,0.2,0.21,-0.22,0.65,0.07,-0.09,0.08,0.06,-0.01,-0.18,-0.19,0.07,0.09,-0.15,-0.4,-0.09,-0.08,0.32,-0.04,0.11,-0.2,-0.24,0.47,0.08,-0.06,0.02,0.29,0.48,0.11,-0.45,0.72,-0.33,-0.15,-0.26,0.17,0.43,0.63,0.1,-0.02,0.35,0.13,0.07,0.0,-0.07,0.16,-0.15,-0.05,-0.52,-0.35,-0.11,0.41,0.17,-0.13,2.22,-0.14,0.16,-0.28,-0.29,-0.1,0.04,-0.38,-0.08,0.53,0.7,0.0,0.08,0.17,0.2,0.32,-0.25,-0.15,-0.42,1.09,0.01,0.04,0.13,-0.3,0.43,-0.42,-0.01,0.02,-0.1,0.79,0.31,-0.28,-0.3,-0.28,0.71,-0.15,0.67,-0.03,0.07,-0.24,-0.47,1.32,-0.35,-0.58,-0.18,0.55,0.01,-0.21,-0.28,-0.19,0.47,0.18,-0.37,0.02,0.15,-0.14,-0.01,-0.19,-0.13,0.08,-0.23,0.39,-0.34,-0.16,0.29,-0.16,-0.04,0.48,0.15,-0.02,0.15,0.19,0.05,-0.3,1.07,-0.36,-0.19,0.24,0.05,0.03,-0.16,0.45,-0.13,0.3,0.49,-0.41,-0.24,-0.18,0.22,-0.76,0.25,0.11,0.21,-0.14,0.19,-0.14,1.62,0.31,0.0,-1.08,-0.05,-0.23,0.47,-0.62,0.26,-0.35,-0.16,0.15,0.07,0.03,0.38,-0.14,0.0,-0.1,0.1,-0.18,-0.55,0.67,0.33,-0.25,0.38,0.03,-0.28,-0.46,-0.18,0.03,-0.31,0.55,0.03,2.4,-0.58,0.53,-0.3,-0.4,0.19,0.04,-0.14,-0.13,-0.05,0.25,0.23,0.03,0.01,0.05,-0.22,-0.09,0.0,0.14,0.11,-0.28,-0.24,0.04,0.99,0.24,-0.65,-0.07,-0.12,-0.32,-0.36,-0.04,-0.43,-0.19,0.28,0.1,0.23,1.16,1.29,0.12,-0.02,-0.27,-0.03,0.18,-0.08,0.98,-0.08,0.18,0.36,0.08,-0.16,-0.15,-0.21,-0.23,-0.14,0.27,-0.29,0.04,0.15,-0.1,-0.16,0.1,0.0,-0.15,0.09,-0.03,-0.01,-0.24,0.34,0.73,-0.19,0.1,-0.16,0.05,-0.08,0.02,0.02,0.3,0.17,-0.39,-0.43,-0.28,-0.21,0.2,-0.61,0.48,0.0,-0.13,-0.43,-0.22,0.76,-0.44,0.78,-0.19,-0.12,0.04,0.42,0.21,-0.15,0.03,-0.44,-0.1,0.06,-0.16,-0.43,-0.12,-0.06,-0.18,0.3,-0.11,0.39,-0.07,-0.15,0.18,-0.25,0.42,0.0,0.75,-0.25,-0.07,-0.44,-0.03,0.26,-0.31,-0.23,-0.08,-0.49,-0.21,0.61,1.13,0.51,0.49,-0.01,0.02,1.03,0.83,-0.15,0.0,-0.04,0.06,0.3,0.07,0.49,0.14,-0.21,-0.05,0.13,0.34,0.13,-0.03,-0.45,1.49,0.2,0.48,-0.14,0.58,-0.15,-0.07,-0.47,-0.24,-0.1,1.58,0.62,-0.1,-0.43,0.11,-0.6,0.85,1.25,-0.47,0.37,0.03,0.67,-0.14,0.96,-0.22,0.04,-0.29,1.07,-0.1,0.29,0.13,0.34,-0.07,0.3,-0.38,0.4,0.02,-0.02,-0.43,-0.07,-0.41,0.11,0.28,-0.05,0.46,0.35,0.21,0.0,-0.07,0.13,-0.07,0.09,0.32,0.98,-0.58,0.09,0.06,0.28,0.3,-0.38,0.31,0.13,0.38,0.01,-0.07,-0.12,-0.38,0.2,0.09,0.0,0.04,0.28,-0.11,0.3,-0.52,-0.34,0.4,1.08,-0.54,-0.42,-0.34,0.73,0.28,0.35,0.14,0.39,0.08,-0.2,-0.01,-0.17,0.04,0.44,0.86,-0.01,0.46,-0.01,0.22,-0.32,-0.37,-0.06,0.05,0.08,1.35,-0.19,-0.07,-0.16,-0.07,-0.22,-0.14,0.23,0.35,0.72,-0.11,0.13,-0.06,0.28,-0.36,-0.14,0.23,0.27,0.34,-0.23,-0.17,-0.6,0.08,-0.36,0.2,-0.34,-0.13,-0.21,-0.02,0.14,-0.32,0.42,-0.26,0.07,0.43,-0.14,0.44,-0.08,1.34,-0.2,0.27,-0.07,0.42,0.0,-0.19,-0.16,-0.16,0.39,0.11,0.77,-0.12,0.28,-0.1,0.53,-0.79,-0.25,0.19,-0.17,0.05,-0.07,0.32,-0.06,-0.31,-0.02,0.01,-0.03,0.05,0.09,-0.1,0.65,0.67,-0.17,0.43,-0.34,0.0,-0.4,-0.14,0.86,0.05,0.1,-0.01,-0.46,-0.18,0.06,-0.09,0.07,0.07,0.17,-0.26,-0.03,-0.1,-0.09,0.1,-0.24,0.36,-0.12,0.65,-0.14,-0.17,-0.09,0.59,-0.02,0.07,-0.36,-0.1,-0.2,0.09,-0.25,0.18,0.07,0.21,0.0,-0.41,-0.33,-0.16,0.13,0.55,-0.18,-0.28,0.77,0.4,0.2,0.01,-0.04,0.29,-0.31,-0.05,0.35,0.4,0.04,-0.53,-0.16,-0.18,0.03,0.79,-0.17,0.1,-0.8,-0.17,0.2,-0.3,0.04,-0.03,-0.17,-0.33,-0.06,1.03,0.41,0.14,-0.18,-0.06,0.87,0.05,-0.15,-0.19,0.04,0.88,0.01,0.22,0.33,-0.47,0.01,0.67,0.48,-0.2,-0.08,0.16,0.26,-0.04,-0.24,0.26,1.55,0.06,-0.42,-0.36,-0.35,0.09,-0.27,-0.26,0.17,-0.24,0.4,0.04,0.1,0.19,0.46,-0.03,0.2,0.05,-0.36,-0.22,0.04,-0.21,0.41,1.21,0.34,-0.22,-0.04,0.13,-0.12,0.0,0.11,-0.56,-0.16,-0.29,0.4,1.55,0.5,-0.14,0.05,0.03,0.64,-0.54,0.93,0.23,0.04,0.14,-0.05,1.73,-0.02,0.41,-0.18,0.27,-0.26,-0.2,-0.76,-0.2,-0.49,-0.15,-0.14,-0.29,0.22,0.0,0.1,0.55,-0.11,-0.1,-0.11,-0.15,-0.01,0.03,0.15,-0.31,0.11,-0.37,-0.15,-0.35,-0.16,-0.2,0.13,-0.24,-0.22,-0.02,0.24,0.27,0.02,0.03,0.08,-0.09,-0.22,0.06,0.31,0.11,0.3,-0.48,0.12,0.03,1.0,-0.01,0.01,-0.25,-0.2,0.1,0.8,-0.1,1.38,1.18,1.08,-0.16,-0.1,-0.07,-0.09,-0.18,-0.03,-0.24,-0.45,-0.01,0.09,-0.17,-0.43,0.71,-0.33,-0.08,0.21,1.06,-0.48,0.59,-0.04,0.05,-0.29,-0.09,-0.38,0.26,0.11,-0.06,-0.13,0.08,0.29,0.14,0.16,0.01,-0.84,0.22,0.74,0.54,-0.23,0.04,1.18,0.59,0.33,-0.02,-0.05,-0.03,-0.48,0.16,0.1,0.21,-0.44,0.07,-0.18,-0.04,-0.08,0.18,0.2,-0.32,0.0,-0.07,0.31,-0.19,-0.06,0.13,-0.4,-0.14,-0.08,0.5,0.34,0.54,0.06,-0.31,-0.18,-0.02,0.18,-0.06,0.02,0.85,1.36,0.02,1.39,0.15,-0.32,-0.31,-0.05,-0.76,-0.21,-0.41,-0.17,-0.25,0.23,0.28,0.61,0.4,-0.34,0.16,-0.18,0.28,-0.27,0.0,-0.22,0.17,-0.15,-0.01,0.07,-0.19,0.15,0.14,0.28,-0.37,-0.65,0.24,0.67,-0.05,-0.33,-0.03,0.12,-0.12,-0.11,-0.06,0.55,-0.38,-0.04,0.06,-0.65,-0.16,0.21,-0.07,0.18,-0.17,0.22,0.28,-0.11,0.03,0.45,0.52,-0.34,-0.12,0.01,-0.47,0.35,-0.41,-0.37,-0.37,-0.08,0.1,0.03,0.32,0.08,-0.28,0.07,0.15,-0.29,0.34,-0.2,0.32,-0.04,-0.07,0.05,1.57,-0.13,0.18,0.16,0.61,-0.04,-0.4,-0.15,0.4,-0.13,-0.04,-0.03,-0.12,0.25,-0.07,0.37,0.59,-0.18,0.24,1.05,0.4,-0.54,-0.16,0.1,0.39,-0.23,0.08,-0.09,-0.21,0.32,0.75,0.18,-0.16,-0.02,0.3,0.65,-0.13,0.14,0.91,-0.05,-0.03,0.15,0.08,0.2,0.33,-0.31,0.04,0.01,0.05,0.14,0.3,-0.22,-0.05,0.15,0.23,0.2,0.14,0.05,-0.07,-0.35,0.18,0.27,0.17,-0.11,0.08,0.31,-0.14,-0.35,-0.14,-0.08,-0.2,-0.29,0.09,-0.09,-0.36,-0.2,-0.33,-0.11,0.04,-0.04,0.47,0.18,0.04,0.02,-0.38,0.25,0.0,0.32,1.5,0.03,-0.14,0.03,0.11,-0.67,-0.17,-0.05,0.36,-0.44,0.06,0.16,-0.3,-0.55,-0.14,0.1,0.1,-0.01,-0.23,0.25,0.07,0.02,-0.04,-0.59,0.1,0.38,0.06,-0.18,-0.56,-0.4,-0.22,0.01,0.0,-0.06,-0.03,0.13,0.01,-0.29,-0.06,0.45,0.16,0.18,-0.32,0.19,0.15,-0.13,0.03,0.15,-0.28,-0.2,1.66,0.19,-0.06,-0.21,-0.16,-0.14,0.16,0.25,-0.55,-0.46,-0.32,0.17,0.12,1.31,0.08,1.5,1.62,0.46,0.29,-0.11,0.23,-0.11,1.28,-0.14,1.02,-0.17,0.23,-0.55,-0.62,1.13,0.22,0.04,1.15,0.16,-0.05,0.16,0.92,-0.04,-0.15,0.33,0.12,-0.15,1.16,-0.21,0.31,-0.05,-0.12,-0.46,0.1,0.05,0.1,-0.36,-0.04,0.66,0.16,0.14,0.66,0.55,-0.08,0.01,-0.1,0.33,0.0,0.36,0.02,-0.17,0.0,-0.01,-0.32,-0.15,-0.24,0.07,-0.26,3.09,0.32,-0.12,-0.43,0.11,0.42,0.03,-0.25,0.09,-0.34,0.01,-0.66,0.05,0.0,0.94,-0.67,0.09,0.38,-0.15,-0.25,0.4,0.17,-0.18,0.06,-0.17,-0.18,0.27,-0.06,-0.1,-0.04,0.33,-0.11,-0.48,0.06,0.07,0.02,0.24,-0.33,0.1,-0.01,-0.12,-0.37,0.06,0.11,-0.22,0.03,-0.05,0.84,-0.15,0.37,-0.25,-0.26,0.07,0.24,-0.05,0.11,-0.21,0.27,-0.23,-0.42,-0.12,0.34,-0.11,-0.31,-0.38,-0.17,0.03,-0.16,0.38,-0.27,-0.14,0.16,-0.1,0.33,0.1,-0.26,0.4,-0.18,-0.07,0.16,0.07,-0.08,0.39,0.09,-0.02,-0.26,-0.29,0.0,1.18,0.28,-0.08,0.19,0.08,-0.17,0.29,0.74,-0.41,-0.25,0.11,-0.22,-0.09,-0.27,-0.26,-0.33,0.17,-0.15,-0.07,-0.05,-0.06,0.17,-0.05,0.01,-0.1,-0.2,-0.35,0.2,0.26,-0.03,0.3,0.12,0.01,0.12,-0.04,-0.16,0.15,0.94,0.14,-0.11,0.12,-0.15,0.09,-0.24,0.07,-0.22,0.93,-0.38,0.69,-0.46,0.09,-0.54,-0.13,1.21,0.17,1.32,-0.26,-0.09,0.02,-0.07,-0.03,0.49,-0.37,0.14,-0.09,0.36,0.01,-0.15,-0.04,-0.37,-0.24,-0.03,0.7,-0.02,-0.18,0.44,0.31,-0.4,0.12,0.18,-0.2,0.2,0.72,0.07,0.36,-0.14,-0.69,0.4,0.64,-0.21,0.09,-0.42,0.18,-0.23,-0.23,-0.03,-0.18,0.12,0.57,-0.04,0.07,-0.25,0.79,-0.19,-0.32,-0.3,0.11,-0.36,-0.17,-0.43,-0.31,1.2,-0.12,0.03,-0.11,0.24,0.03,-0.25,-0.12,-0.29,0.49,0.94,1.0,0.21,-0.22,1.14,0.44,-0.23,-0.35,0.33,-0.42,-0.19,0.68,0.28,-0.4,-0.08,0.03,0.44,-0.06,0.23,0.18,0.0,0.11,-0.49,-0.21,-0.14,-0.4,0.2,0.01,-0.24,0.01,-0.36,0.77,-0.24,-0.34,-0.13,0.33,0.04,-0.04,-0.05,-0.31,-0.17,0.35,0.62,-0.21,0.21,2.51,0.15,-0.07,0.06,-0.12,0.17,-0.15,-0.17,0.81,-0.22,-0.39,-0.27,-0.38,-0.07,-0.02,0.16,0.42,0.48,-0.14,0.41,0.69,-0.22,0.36,-0.1,-0.06,-0.09,-0.3,-0.76,0.06,-0.44,-0.21,0.0,-0.35,-0.19,-0.23,0.29,1.65,0.11,0.03,-0.16,0.07,-0.27,0.33,0.08,-0.5,-0.08,-0.2,-0.35,-0.09,0.17,0.09,0.36,0.14,0.06,-0.12,-0.02,-0.07,-0.53,-0.16,-0.39,0.68,-0.32,0.06,0.11,0.27,-0.19,0.28,-0.35,-0.14,0.59,-0.06,-0.14,0.02,0.26,0.35,-0.11,-0.07,0.19,0.0,1.38,-0.46,-0.1,-0.19,0.09,0.39,0.57,-0.17,0.41,-0.18,0.49,-0.14,-0.2,0.03,-0.41,-0.34,-0.16,0.28,-0.07,0.83,0.13,0.1,-0.24,0.0,0.08,-0.11,0.14,0.1,0.2,0.28,-0.04,0.2,-0.3,-0.28,-0.72,0.34,0.18,0.2,-0.14,0.11,-0.41,-0.33,-0.07,-0.13,0.53,-0.04,0.48,1.34,0.05,0.37,-0.09,0.04,-0.24,2.37,-0.41,-0.22,-0.05,0.15,0.07,0.88,1.93,0.29,0.74,0.37,-0.33,-0.18,-0.31,-0.27,0.06,-0.12,0.23,-0.01,0.4,-0.35,-0.22,0.35,-0.32,0.39,-0.35,0.16,0.75,0.06,0.1,-0.46,-0.19,0.37,-0.01,-0.21,-0.5,-0.4,0.49,-0.5,-0.16,0.43,-0.09,-0.17,-0.22,0.01,1.42,0.27,-0.05,-0.33,0.14,-0.1,-0.27,0.36,-0.22,-0.44,0.5,-0.08,0.22,0.05,0.28,0.15,-0.16,0.09,0.07,0.18,-0.23,0.21,-0.13,0.2,-0.39,0.35,-0.01,-0.36,-0.16,-0.19,-0.44,-0.05,-0.26,-0.22,-0.58,-0.11,-0.98,0.35,0.02,-0.28,0.1,-0.19,0.02,0.21,-0.02,-0.04,-0.15,0.02,-0.31,0.25,0.13,-0.39,-0.06,-0.01,0.27,0.1,-0.05,0.25,-0.83,1.65,-0.5,0.94,-0.26,0.0,-0.17,-0.07,-0.03,0.12,0.02,-0.21,-0.46,0.03,0.32,-0.08,0.11,0.36,-0.12,0.27,-0.38,0.03,0.31,-0.43,-0.06,-0.38,1.19,-0.41,1.81,0.07,0.01,0.02,0.2,-0.52,-0.34,-0.28,-0.03,0.21,0.54,0.24,1.83,-0.22,-0.12,-0.06,0.03,0.32,-0.04,0.21,-0.05,-0.51,1.04,0.16,0.7,-0.06,-0.26,0.09,-0.25,0.33,0.26,-0.1,1.37,0.36,0.11,-0.04,-0.18,0.26,0.19,0.32,-0.07,0.29,0.18,-0.28,-0.23,0.2,-0.05,0.26,-0.15,0.41,1.4,-0.22,-0.13,-0.08,-0.04,1.52,0.1,0.15,-0.19,-0.15,0.11,-0.27,-0.4,0.6,0.15,0.01,0.51,0.03,-0.26,0.09,0.17,0.16,1.0,-0.12,0.27,-0.38,0.28,0.0,0.17,1.19,-0.22,-0.22,-0.25,-0.13,0.05,-0.04,-0.02,0.22,0.12,-0.01,0.28,0.5,0.19,0.18,0.08,1.1,-0.32,0.02,0.21,-0.72,0.02,-0.06,0.07,-0.18,-0.11,1.05,0.19,0.48,-0.12,0.29,0.01,0.32,-0.14,0.01,0.18,-1.0,-0.37,0.24,-0.36,-0.48,0.07,-0.47,0.51,-0.07,-0.01,-0.21,-0.1,0.26,-0.13,0.0,0.7,-0.18,-0.37,-0.48,-0.17,0.45,0.13,0.16,0.09,-0.13,-0.21,0.26,-0.26,-0.31,-0.28,-0.5,1.51,0.6,0.2,0.88,-0.22,-0.36,-0.04,-0.07,-0.23,-0.47,-0.22,-0.11,-0.5,0.29,-0.29,1.49,-0.96,-0.08,-0.18,0.08,0.36,-0.23,-0.15,-0.11,0.31,0.04,0.28,-0.74,0.26,-0.16,-0.34,-0.44,0.15,0.5,-0.15,0.0,0.19,1.05,0.17,0.12,-0.05,1.24,-0.09,0.69,-0.39,0.12,0.03,0.48,-0.33,-0.47,0.76,0.09,0.03,0.3,-0.33,-0.13,1.13,-0.25,1.19,0.61,-0.04,0.12,-0.04,0.3,-0.29,0.05,1.34,-0.33,-0.34,-0.63,-0.08,0.44,0.16,0.08,1.39,0.13,-0.26,-0.18,-0.16,-0.25,-0.15,0.15,0.11,0.08,-0.26,-0.08,-0.08,-0.38,-0.05,0.35,-0.02,-0.21,0.03,0.19,0.06,0.7,0.18,-0.32,0.09,-0.4,-0.11,-0.32,0.25,0.58,-0.22,-0.11,0.04,0.0,-0.38,0.25,0.14,-0.3,-0.08,0.37,0.01,0.06,-0.01,-0.19,-0.25,1.04,0.6,1.42,-0.31,0.28,-0.21,0.28,-0.06,0.19,0.52,0.1,0.12,-0.38,-0.19,-0.08,0.17,0.06,0.36,-0.28,-0.26,0.79,-0.37,0.42,-0.12,-0.06,0.33,-0.11,0.44,-0.03,1.52,-0.02,0.28,-0.3,-0.03,-0.24,-0.13,0.47,1.31,0.0,-0.05,-0.4,0.19,-0.12,-0.28,-0.27,-0.25,-0.01,-0.28,-0.25,0.27,0.55,-0.17,0.6,-0.16,-0.22,0.62,-0.19,-0.17,-0.03,0.25,0.06,-0.17,-0.07,0.0,-0.35,0.09,0.24,0.26,0.18,-0.22,-0.13,0.47,-0.14,0.85,-0.09,0.38,0.39,-0.2,0.09,-0.31,0.12,-0.26,-0.22,0.15,-0.17,-0.02,-0.15,-0.51,1.59,-0.16,-0.09,0.56,0.33,0.02,0.28,-0.04,-0.35,-0.05,0.27,-0.18,0.21,-0.02,-0.19,-0.13,-0.1,0.44,-0.06,0.92,-0.03,-0.09,0.61,0.31,-0.34,0.17,-0.33,0.15,-0.27,-0.18,-0.23,0.48,0.38,0.38,0.09,-0.01,0.05,0.44,-0.23,-0.12,-0.35,-0.03,0.38,0.46,0.3,-0.33,0.0,0.24,-0.61,-0.29,0.11,-0.14,0.15,-0.17,0.13,0.3,0.52,0.5,-0.13,-0.08,-0.09,-0.05,-0.29,0.89,0.59,0.41,1.63,0.27,-0.22,-0.01,-0.01,0.17,-0.27,0.65,0.21,-0.36,0.07,0.04,1.56,0.46,0.66,-0.43,-0.07,0.08,0.0,-0.23,0.37,-0.13,0.28,-0.14,-0.02,0.11,0.04,-0.23,0.03,-0.26,0.22,0.31,0.24,-0.04,-0.28,-0.16,-0.04,0.27,-0.12,0.09,0.84,-0.56,0.06,0.1,0.05,0.44,-0.55,0.13,0.63,-0.06,-0.02,0.29,0.32,0.54,-0.27,-0.42,0.64,0.13,0.02,2.16,0.33,-0.31,0.2,-0.16,0.18,-0.23,-0.31,0.46,-0.26,-0.2,-0.21,0.09,-0.1,0.12,0.57,-0.08,0.09,0.36,0.11,-0.21,0.08,-0.87,-0.06,-0.3,0.79,0.15,-0.24,-0.29,0.23,0.31,0.1,-0.15,0.2,0.55,-0.37,0.46,0.83,0.42,0.44,-0.08,-0.19,0.15,0.24,1.05,-0.66,-0.01,-0.12,-0.29,0.32,-0.29,0.46,1.07,-0.13,0.78,0.23,-0.27,0.55,-0.14,0.18,-0.07,0.99,0.22,0.12,-0.08,-0.2,-0.2,0.22,0.04,-0.12,-0.1,0.11,0.89,0.05,0.17,0.28,-0.25,-0.29,2.14,-0.37,-0.03,-0.28,0.13,0.09,1.71,0.12,-0.42,-0.05,0.57,0.17,-0.35,0.5,-0.12,-0.18,0.32,-0.28,-0.01,1.56,-0.53,0.01,-0.15,-0.07,-0.2,-0.08,-0.25,-0.3,0.02,1.43,-0.05,-0.41,-0.21,-0.08,0.08,0.17,0.0,0.04,0.03,-0.46,-0.06,0.08,-0.29,-0.11,0.2,-0.09,0.35,-0.14,0.06,0.29,-0.1,-0.27,-0.36,0.06,0.12,0.17,0.03,-0.12,-0.24,0.32,0.16,0.3,0.28,0.9,0.3,-0.04,0.25,-0.17,-0.66,-0.28,-0.07,-0.14,-0.11,0.87,-0.12,0.06,-0.19,-0.18,0.29,0.01,0.85,0.37,0.01,0.09,0.62,0.84,0.29,0.16,-0.34,-0.16,-0.25,0.2,-0.02,-0.07,-0.11,-0.32,-0.44,-0.01,0.57,-0.39,0.77,0.1,0.48,0.28,-0.31,1.0,-0.02,-0.01,-0.24,0.36,0.1,0.27,-0.29,0.17,-0.07,0.23,-0.26,-0.32,0.07,0.6,0.2,0.08,-0.28,-0.55,0.06,1.35,0.13,-0.15,0.25,-0.58,-0.28,0.28,-0.07,-0.34,0.48,0.1,-0.01,-0.01,-0.31,-0.19,-0.17,0.31,-0.46,0.34,-0.38,0.33,-0.29,-0.02,-0.18,-0.14,-0.33,-0.19,0.14,1.23,-0.12,0.76,0.19,-0.26,0.13,0.09,-0.18,-0.25,-0.1,-0.36,0.02,1.03,0.16,-0.04,-0.03,-0.08,-0.25,-0.18,0.02,0.08,-0.26,-0.06,-0.25,0.26,0.1,-0.18,-0.18,0.22,-0.02,0.22,0.31,1.0,-0.38,0.35,-0.16,-0.15,0.15,0.4,-0.37,-0.18,-0.02,0.26,2.52,0.09,0.34,0.0,-0.46,0.33,-0.31,-0.25,-0.35,-0.09,0.0,0.7,-0.45,0.2,-0.04,0.04,0.03,-0.39,0.62,-0.3,-0.23,0.26,-0.24,-0.17,0.4,0.22,0.54,-0.11,-0.07,0.88,1.22,1.64,-0.19,0.5,-0.35,-0.14,0.41,-0.61,0.03,-0.41,1.23,-0.33,-0.04,0.6,2.51,-0.24,0.31,-0.3,0.17,0.35,0.05,0.78,-0.3,0.13,-0.1,0.17,0.15,-0.03,0.26,-0.34,0.25,-0.14,-0.15,0.25,-0.24,0.04,-0.2,-0.28,-0.28,-0.35,-0.16,-0.12,0.32,-0.15,-0.06,-0.01,-0.26,0.14,0.18,0.24,-0.18,-0.19,-0.01,-0.21,-0.46,0.22,0.34,-0.33,0.11,-0.14,0.05,0.46,-0.04,0.5,-0.31,0.45,0.01,0.56,-0.15,0.0,-0.22,0.27,0.26,-0.35,-0.05,0.06,0.31,0.36,-0.02,0.03,0.12,-0.23,-0.41,-0.22,0.3,0.17,-0.09,-0.43,-0.11,-0.06,0.04,-0.12,-0.12,0.54,-0.21,0.11,-0.1,-0.17,0.1,-0.02,-0.03,0.01,-0.09,-0.02,-0.36,0.13,1.16,0.14,-0.06,-0.47,-0.23,0.24,-0.19,0.01,-0.01,0.34,-0.07,0.54,0.76,0.2,0.12,0.22,0.03,0.13,0.51,-0.24,-0.43,-0.35,-0.6,0.91,0.11,-0.43,-0.28,-0.3,0.3,-0.05,-0.06,-0.2,0.13,-0.21,-0.23,0.3,0.17,0.06,0.01,0.32,-0.84,-0.25,0.45,0.21,0.1,0.21,1.22,-0.08,-0.71,0.34,0.12,0.26,0.03,-0.06,-0.06,0.5,-0.02,-0.03,0.26,0.05,0.0,-0.02,1.36,0.01,-0.04,0.48,0.04,-0.37,-0.2,0.27,-0.09,-0.03,0.12,-0.05,-0.06,-0.2,0.41,-0.17,-0.37,-0.2,-0.16,-0.46,-0.13,-0.11,-0.7,0.3,0.16,0.3,0.06,0.05,0.2,0.07,-0.18,-0.12,-0.17,0.42,0.43,-0.4,-0.26,0.15,0.59,0.49,0.42,0.01,0.22,0.12,-0.18,-0.23,0.07,0.18,0.15,0.56,-0.07,-0.28,-0.51,-0.28,-0.09,0.2,0.0,-0.07,0.35,0.1,0.1,0.21,-0.28,0.51,-0.19,0.47,0.09,0.33,0.14,1.2,-0.28,0.11,0.59,0.28,-0.49,-0.77,0.06,-0.26,-0.1,-0.13,0.27,1.38,0.92,0.35,-0.29,-0.25,0.43,-0.23,-0.29,-0.24,-0.09,0.16,-0.1,0.3,0.08,-0.47,-0.24,-0.09,0.87,-0.1,-0.36,-0.07,-0.04,0.38,-0.12,-0.01,-0.12,-0.08,0.44,-0.04,1.01,-0.39,-0.1,-0.04,0.37,-0.09,-0.46,0.41,-0.01,0.08,-0.13,-0.43,0.17,-0.35,-0.12,0.12,-0.18,0.9,-0.25,-0.37,-0.2,0.1,0.23,-0.24,0.43,-0.64,0.12,0.15,0.41,-0.07,0.18,0.33,-0.15,0.42,0.26,0.07,-0.18,-0.03,0.19,1.18,-0.16,-0.02,0.23,-0.04,1.27,-0.29,0.4,0.23,-0.2,0.06,0.01,-0.03,0.58,-0.34,0.23,-0.04,0.03,0.27,0.04,-0.27,-0.2,0.99,0.27,-0.22,-0.62,0.24,-0.45,0.12,0.17,0.01,0.21,-0.04,0.75,1.3,-0.02,0.41,0.25,-0.38,-0.24,0.0,-0.06,0.32,-0.71,0.22,-0.11,0.07,-0.01,0.09,0.47,0.34,-0.12,0.03,0.27,-0.16,-0.5,0.5,-0.03,0.31,2.16,0.06,0.33,-1.04,0.04,-0.51,-0.07,0.35,0.4,0.45,0.75,0.47,0.22,0.04,-0.19,-0.33,0.6,1.24,-0.23,0.01,0.05,-0.37,-0.52,1.77,-0.07,0.41,-0.03,0.45,0.44,0.06,-0.2,-0.43,-0.42,-0.3,-0.11,0.34,0.29,0.52,-0.41,-0.39,0.52,0.55,0.55,0.24,0.13,-0.04,0.41,-0.12,-0.45,-0.2,0.51,0.28,-0.87,0.47,0.01,-0.07,0.19,0.49,0.6,0.24,0.17,-0.02,-0.03,0.18,-0.07,0.21,-0.04,0.33,0.16,0.09,-0.19,0.28,0.26,0.14,0.13,-0.05,-0.04,-0.16,-0.25,0.28,-0.2,0.5,-0.31,-0.1,0.58,0.89,-0.22,0.17,0.17,0.1,-0.26,0.16,-0.48,-0.39,-0.49,1.27,-0.74,-0.14,-0.28,0.42,-0.24,-0.43,0.18,-0.39,-0.11,1.05,-0.54,-0.33,0.06,0.11,-0.03,0.83,0.03,-0.11,0.16,-0.48,0.57,-0.09,-0.08,-0.39,-0.25,0.03,1.57,-0.24,-0.17,-0.11,-0.2,-0.52,0.09,-0.3,-0.11,-0.18,-0.02,0.27,-0.19,-0.06,-0.22,-0.46,0.13,-0.01,-0.14,-0.55,0.3,-0.22,-0.2,-0.29,0.08,-0.19,-0.8,0.12,-0.17,-0.39,-0.15,0.01,0.17,-0.07,-0.02,0.4,-0.04,0.17,0.07,-0.02,0.36,0.77,-0.08,-0.38,0.36,-0.23,-0.61,0.0,-0.16,-0.06,0.04,0.13,-0.09,-0.16,-0.12,0.09,-0.06,-0.05,-0.36,-0.11,-0.42,0.09,0.62,0.21,-0.15,-0.12,0.38,-0.03,0.03,0.03,-0.07,-0.36,-0.06,-0.14,1.43,-0.09,-0.32,0.35,-0.45,-0.11,0.39,0.1,-0.08,-0.45,0.05,-0.26,-0.26,-0.17,-0.38,0.51,-0.37,-0.42,0.1,0.02,-0.05,-0.06,-0.02,0.06,0.92,-0.03,-0.13,0.03,-0.05,0.0,0.74,0.0,-0.51,0.41,-0.37,0.27,0.48,0.19,-0.41,0.03,-0.25,-0.19,1.25,0.38,-0.07,0.09,-0.37,-0.12,1.45,-0.42,0.67,-0.56,1.46,-0.17,0.14,-0.18,0.14,-0.38,-0.08,-0.09,-0.06,0.8,-0.55,-0.12,0.08,0.1,0.08,0.02,-0.6,0.05,-0.22,-0.45,-0.41,-0.39,-0.4,0.97,-0.42,-0.24,0.55,0.18,0.23,-0.09,-0.04,-0.6,-0.11,0.15,0.86,-0.11,-0.08,-0.24,-0.06,-0.17,-0.07,-0.18,0.17,-0.23,-0.15,-0.21,-0.32,-0.1,0.18,-0.13,0.74,-0.23,-0.35,0.11,-0.08,-0.04,0.22,-0.21,-0.47,0.01,-0.15,0.21,-0.6,-0.28,-0.24,-0.12,-0.29,-0.33,-0.22,-0.32,-0.33,0.7,-0.43,-0.09,-0.17,-0.08,0.59,0.39,-0.07,0.11,-0.02,-0.15,-0.22,0.82,-0.11,-0.05,-0.14,-0.63,-0.27,-0.06,-0.07,-0.04,-0.07,-0.24,0.4,-0.09,0.58,0.89,0.03,-0.21,0.21,-0.69,0.14,-0.26,-0.19,-0.46,0.15,-0.17,0.44,-0.17,0.36,0.01,-0.12,0.18,-0.33,-0.4,0.11,0.09,0.42,-0.28,0.01,0.44,0.01,0.14,-0.12,-0.02,-0.04,-0.13,-0.03,-0.1,0.02,0.39,0.4,-0.22,-0.12,0.37,1.07,0.5,-0.01,0.06,0.63,0.12,-0.11,-0.23,-0.25,-0.15,-0.01,-0.25,-0.07,-0.34,0.54,0.39,-0.18,-0.41,0.78,0.54,-0.04,-0.39,0.03,-0.67,-0.05,0.32,-0.05,0.29,0.4,-0.11,-0.09,-0.05,1.05,0.04,0.54,-0.12,0.59,0.02,-0.08,-0.79,1.17,-0.07,-0.06,0.04,0.17,0.05,-0.26,-0.18,-0.03,0.63,0.26,-0.06,-0.48,0.27,0.59,-0.42,0.69,-0.58,-0.1,0.25,-0.31,0.23,0.1,0.12,0.42,1.52,0.07,0.03,0.16,-0.32,0.08,0.14,1.4,-0.12,-0.71,-0.44,-0.13,-0.1,0.78,0.12,0.03,-0.42,-0.19,0.3,0.09,-0.18,0.27,-0.32,-0.01,-0.31,-0.47,-0.02,0.5,-0.35,1.03,0.17,0.08,-0.2,-0.16,0.27,-0.14,-0.02,0.17,0.01,2.3,0.02,-0.06,0.4,-0.2,-0.04,0.62,-0.04,0.17,0.47,0.21,0.01,0.12,-0.27,-0.08,-0.09,0.36,-0.16,0.02,-0.05,-0.13,-0.31,-0.39,0.2,0.35,0.59,-0.08,-0.44,0.02,-0.09,-0.62,-0.05,0.05,-0.16,0.06,0.09,0.4,0.1,0.93,-0.03,-0.12,-0.05,-0.36,0.15,0.16,0.19,-0.09,0.23,-0.33,-0.07,0.12,0.89,-0.24,0.13,0.19,0.1,-0.25,-0.3,0.31,0.45,-0.26,-0.05,-0.29,-0.37,0.74,0.22,0.03,0.03,0.04,0.06,-0.07,-0.04,-0.11,0.41,-0.07,0.96,0.25,-0.13,-0.03,0.35,-0.02,-0.18,-0.31,-0.45,0.22,-0.28,1.0,0.09,-0.32,-0.29,-0.26,0.08,0.11,-0.47,0.08,0.11,-0.23,-0.57,-0.05,-0.09,0.27,0.51,-0.55,0.48,0.29,-0.06,0.06,-0.37,-0.37,0.15,-0.27,0.74,0.02,0.21,-0.46,0.19,-0.19,0.03,0.1,0.02,-0.02,0.1,-0.01,-0.31,0.18,0.33,-0.14,-0.01,0.48,0.56,0.0,-0.26,-0.2,-0.05,-0.12,-0.12,-0.34,-0.35,0.0,0.33,-0.16,-0.31,0.04,0.0,-0.17,0.35,0.34,-0.47,-0.2,-0.27,-0.1,0.19,-0.66,-0.26,-0.03,-0.03,-0.07,1.65,-0.25,-0.15,-0.3,0.15,0.09,0.12,0.33,-0.42,0.16,-0.36,0.0,-0.06,-0.45,-0.43,1.35,0.85,0.06,0.1,0.13,-0.15,-0.16,-0.05,0.21,0.45,-0.03,-0.23,-0.26,-0.05,0.84,-0.04,0.0,-0.19,-0.12,-0.11,-0.25,0.13,0.44,0.07,-0.22,0.08,0.18,-0.31,0.06,0.02,-0.66,-0.36,-0.27,-0.27,-0.16,-0.22,0.25,1.0,1.05,0.12,0.25,1.5,1.08,-0.91,-0.26,0.13,-0.24,0.04,-0.69,0.24,0.06,-0.23,0.2,-0.28,-0.3,0.43,-0.21,-0.53,0.0,0.01,-0.38,-0.04,-0.03,0.06,-0.27,0.33,-0.15,0.62,0.19,-0.08,-0.33,0.33,0.36,0.34,-0.04,-0.57,-0.33,0.15,0.08,-0.06,-0.11,0.58,0.01,0.29,-0.29,0.07,-0.02,0.14,0.46,0.11,0.35,0.28,-0.07,-0.43,0.04,0.24,0.71,-0.09,0.0,0.32,-0.14,0.34,0.08,0.05,-0.55,-0.01,-0.11,0.2,-0.26,-0.36,0.51,-0.42,-0.25,0.26,-0.03,-0.21,0.04,-0.26,-0.26,-0.49,0.22,-0.17,-0.17,-0.11,-0.11,0.27,-0.02,0.01,0.22,-0.17,-0.4,0.49,-0.19,1.77,-0.23,0.17,0.02,0.08,0.02,-0.08,-0.18,-0.28,-0.04,0.25,-0.13,-0.09,0.01,0.77,0.14,-0.11,-0.07,0.54,0.04,0.16,-0.09,-0.12,-0.16,0.06,-0.27,-0.32,0.56,0.04,0.11,-0.31,-0.09,0.29,-0.14,0.71,-0.22,-0.36,-0.05,-0.16,0.02,0.02,0.04,0.29,-0.06,0.31,-0.25,-0.47,-0.03,0.27,-0.31,-0.19,-0.41,-0.23,-0.2,-0.07,1.22,-0.02,0.58,0.09,0.02,-0.09,-0.17,-0.07,0.14,0.39,-1.13,0.05,-0.3,0.18,0.06,0.36,0.61,1.78,0.02,-0.14,0.11,-0.5,-0.3,-0.12,-0.08,0.48,0.32,0.24,-0.03,-0.18,-0.67,-0.1,1.44,0.11,-0.29,0.52,0.3,0.03,0.33,0.1,0.36,-0.18,-0.06,0.07,-0.09,0.15,1.26,-0.22,0.11,-0.25,0.04,-0.09,0.28,-0.19,-0.02,2.92,-0.24,-0.27,-0.04,-0.62,0.22,1.29,0.09,0.41,-0.02,0.54,-0.27,-0.19,0.06,-0.35,0.18,0.3,-0.18,-0.05,0.61,0.74,-0.39,-0.23,0.01,-0.04,-0.34,0.01,0.03,-0.17,0.24,0.86,0.74,-0.15,0.06,-0.38,0.1,-0.39,0.7,-0.02,0.64,0.17,0.15,-0.18,-0.49,-0.21,-0.44,1.43,0.04,-0.34,0.28,0.18,-0.44,-0.45,0.05,-0.39,0.07,0.62,-0.15,-0.13,0.13,-0.08,-0.03,-0.02,-0.26,0.31,0.02,0.15,0.01,-0.25,-0.35,0.09,-0.13,-0.35,-0.33,-0.11,-0.24,0.17,-0.37,-0.18,0.52,0.18,0.31,-0.16,0.24,0.18,-0.95,0.08,-0.31,0.37,0.13,0.48,0.26,-0.21,-0.09,0.35,0.03,0.09,-0.25,1.94,0.36,0.18,0.24,-0.15,0.1,-0.13,-0.16,-0.16,0.28,0.0,-0.62,0.26,0.11,-0.34,-0.08,-0.22,-0.2,0.32,0.0,-0.23,-0.75,-0.1,0.39,-0.14,0.14,0.07,-0.01,0.03,0.61,0.15,-0.06,-0.43,0.61,0.09,0.48,0.26,-0.04,-0.23,0.07,-0.14,-0.11,-0.44,0.34,-0.46,0.29,0.21,0.18,0.18,-0.55,-0.22,-0.15,0.2,0.09,0.32,0.39,0.39,0.76,0.13,0.01,-0.03,0.41,-0.13,-0.14,-0.21,0.16,0.27,0.91,0.58,0.36,0.27,-0.38,-0.14,0.26,-0.06,0.07,-0.07,-0.17,0.19,0.22,-0.16,0.29,0.43,-0.14,-0.37,-0.19,0.17,-0.11,-0.11,0.36,0.16,0.12,-0.14,0.43,0.36,0.33,0.4,-0.39,0.07,-0.11,0.16,0.11,1.32,0.02,-0.37,0.02,0.76,0.26,-0.13,-0.06,0.15,0.26,0.54,-0.53,-0.22,-0.07,0.29,-0.25,0.65,-0.27,0.19,-0.19,-0.13,-0.17,0.15,0.04,0.1,-0.01,0.48,0.0,-0.08,-0.05,0.19,0.0,0.08,0.33,0.01,0.09,0.09,-0.06,-0.53,0.46,-0.18,0.06,-0.12,0.3,-0.13,0.23,-0.14,0.03,0.15,-0.15,-0.03,-0.33,0.07,1.35,0.16,0.15,-0.14,-0.15,-0.11,-0.18,0.18,-0.25,0.23,-0.39,-0.46,0.0,0.17,-0.27,-0.22,0.47,-0.15,0.47,-0.32,0.26,-0.2,-0.11,-0.24,-0.31,0.08,0.09,0.62,0.38,0.16,-0.1,0.21,0.66,-0.29,0.47,-0.2,1.51,-0.07,0.0,-0.07,-0.05,-0.03,-0.08,0.11,0.09,-0.25,0.52,0.16,0.33,1.14,-0.02,-0.03,-0.23,0.95,-0.15,-0.21,-0.08,-0.04,0.52,0.36,-0.09,0.18,0.18,0.53,0.05,0.46,0.12,0.46,-0.03,-0.17,0.36,0.38,-0.08,0.08,-0.5,-0.02,-0.34,0.18,0.3,-0.09,0.04,0.38,-0.48,-0.04,0.81,-0.05,-0.06,0.1,-0.07,0.08,0.78,-0.38,0.47,-0.02,0.25,0.04,-0.38,-0.11,-0.84,-0.27,-0.2,0.25,-0.03,-0.08,-0.22,0.21,0.54,1.4,-0.09,-0.2,0.26,0.4,0.2,0.08,0.12,-0.26,-0.07,-0.15,0.02,-0.01,-0.35,-0.02,-0.16,0.19,-0.46,-0.14,-0.13,0.12,0.07,0.26,0.13,-0.31,-0.05,-0.05,-0.07,2.16,-0.08,-0.14,0.15,-0.32,-0.43,0.49,0.09,0.39,1.68,1.48,-0.06,0.4,-0.38,-0.37,0.19,-0.11,-0.17,0.23,1.25,-0.05,0.27,0.02,0.27,-0.08,-0.53,-0.24,-0.26,0.06,-0.01,-0.28,0.22,0.82,-0.27,0.36,0.04,0.48,-0.44,-0.31,0.31,-0.31,0.06,0.67,-0.49,0.37,-0.08,0.05,-0.07,0.44,0.07,0.75,0.05,-0.69,-0.12,0.04,0.02,0.26,0.97,-0.15,-0.03,-0.5,-0.39,-0.39,-0.35,0.68,0.03,-0.09,0.11,-0.24,-0.22,-0.05,-0.36,0.3,0.05,-0.03,0.21,0.37,0.14,0.19,0.39,0.09,-0.18,0.31,-0.07,-0.01,-0.08,-0.24,-0.28,0.02,-0.84,-0.28,0.08,-0.25,-0.15,-0.44,-0.22,0.02,0.08,-0.1,0.91,-0.62,0.19,-0.53,-0.31,-0.09,0.16,0.19,-0.09,-0.07,-0.29,0.59,0.27,-0.47,-0.01,1.17,-0.39,0.2,-0.4,-0.16,1.35,0.21,0.32,-0.39,-0.22,0.13,-0.4,0.06,0.34,-0.09,0.34,0.96,1.01,0.11,-0.12,0.41,0.4,-0.23,-0.09,0.09,0.09,-0.29,0.15,-0.04,-0.16,-0.08,-0.41,0.69,-0.25,0.27,-0.06,-0.22,-0.09,-0.01,-0.22,0.02,0.61,-0.03,0.05,-0.6,0.14,0.82,-0.22,-0.11,-0.3,-0.19,-0.74,-0.02,-0.08,0.41,0.28,0.09,-0.16,0.53,-0.29,0.25,0.01,0.2,0.08,0.57,-0.22,0.02,0.13,0.67,-0.32,0.76,-0.55,0.15,0.0,0.95,0.13,0.16,0.22,-0.13,-0.21,0.09,0.09,-0.35,-0.19,0.37,0.1,-0.06,0.25,0.41,-0.04,0.77,0.02,-0.34,-0.12,0.37,-0.31,-0.21,0.1,0.22,-0.48,1.23,-0.35,-0.09,-0.05,-0.13,0.52,0.33,-0.45,-0.23,0.17,0.38,-0.22,-0.42,-0.38,0.16,0.29,-0.07,0.01,-0.11,-0.34,-0.33,0.06,0.34,0.99,0.12,-0.26,-0.04,0.25,-0.05,0.38,0.2,0.43,-0.32,-0.14,0.03,-0.07,-0.08,0.37,-0.48,-0.3,-0.28,-0.62,-0.2,0.07,0.95,-0.49,0.4,-0.36,0.1,0.46,-0.27,0.26,-0.04,-0.08,0.09,0.26,-0.16,-0.34,0.04,1.5,0.69,1.36,-0.18,0.15,0.15,-0.01,-0.19,-0.42,-0.1,-0.27,0.03,0.01,0.19,-0.69,-0.04,0.09,-0.02,0.47,-0.33,0.84,0.32,-0.47,0.64,-0.22,1.36,-0.46,-0.15,-0.53,0.21,-0.17,0.08,0.13,-0.15,-0.33,0.1,-0.1,0.31,0.44,-0.25,1.28,-0.21,-0.41,-0.17,-0.47,-0.17,0.71,-0.24,-0.08,-0.33,-0.09,0.11,-0.14,-0.12,0.58,0.11,-0.37,0.38,0.36,-0.15,0.44,0.46,-0.28,-0.37,0.53,1.36,-0.14,1.34,-0.21,0.7,0.45,0.06,0.21,0.32,0.44,0.12,-0.19,0.03,0.05,-0.21,-0.17,-0.27,-0.54,0.84,1.54,0.04,-0.27,0.76,0.19,0.19,-0.45,0.05,-0.27,0.78,-0.02,0.21,0.33,0.18,-0.29,0.36,0.2,0.5,-0.1,-0.14,-0.04,0.73,-0.14,0.14,-0.11,-0.05,0.0,1.01,0.02,0.08,0.03,0.27,-0.15,0.89,-0.14,-0.36,0.36,0.22,1.22,-0.17,-0.18,0.15,-0.13,0.47,0.5,-0.31,-0.07,0.2,0.17,0.41,0.37,-0.15,-0.17,0.32,-0.19,-0.15,0.27,-0.29,-0.03,-0.26,-0.31,-0.1,0.34,-0.16,-0.2,0.1,0.21,-0.34,-0.07,-0.14,-0.14,-0.31,-0.36,0.12,0.27,-0.02,-0.15,0.17,-0.16,-0.09,-0.19,0.27,-0.06,0.27,-0.33,-0.13,0.3,-0.33,0.46,-0.19,0.16,-0.21,0.88,0.07,0.41,0.27,-0.26,-0.07,-0.02,0.18,0.99,0.15,-0.11,0.25,0.03,-0.16,-0.15,0.09,-0.23,1.49,-0.11,0.38,-0.08,-0.55,-0.23,1.09,-0.35,-0.02,-0.34,0.35,0.0,-0.2,0.03,0.18,0.23,0.55,0.18,0.42,0.27,-0.11,0.38,-0.17,0.24,-0.29,0.39,-0.12,0.08,-0.86,-0.2,-0.02,0.01,0.06,0.64,-0.01,-0.1,-0.17,0.32,0.32,0.52,0.49,-0.18,-0.3,0.61,0.01,0.03,-0.48,-0.22,0.27,0.98,-0.21,-0.18,0.14,0.13,-0.27,0.43,-0.32,0.3,0.46,-0.39,-0.06,-0.33,-0.28,0.24,-0.41,-0.36,0.19,0.0,0.18,0.7,-0.29,0.57,0.11,0.17,-0.14,-0.1,-0.47,0.16,-0.18,0.65,1.33,0.21,-0.45,1.47,-0.32,0.43,-0.27,0.28,-0.17,-0.16,0.02,-0.33,0.02,-0.02,0.61,0.87,-0.07,-0.15,-0.18,-0.43,-0.23,-0.32,0.33,-0.21,0.18,-0.05,0.23,0.12,-0.17,0.01,-0.13,0.29,0.1,0.11,0.61,-0.17,-0.14,0.35,-0.02,-0.09,-0.15,-0.42,0.33,0.17,-0.06,-0.23,0.13,-0.4,0.28,-0.3,-0.45,-0.1,-0.34,-0.47,-0.38,0.01,0.15,0.46,-0.39,0.29,0.76,-0.08,-0.02,0.77,-0.24,-0.25,-0.11,0.03,-0.29,-0.07,0.63,0.8,0.69,-0.18,-0.35,-0.25,-0.11,0.14,-0.23,-0.09,0.03,-0.05,0.37,-0.04,0.01,0.35,-0.1,-0.2,-0.31,0.14,-0.05,0.41,-0.13,0.23,-0.12,0.1,0.26,0.23,0.26,-0.18,-0.05,0.34,0.05,0.25,0.02,0.09,-0.06,0.25,-0.19,0.78,-0.24,0.25,0.32,-0.06,-0.05,-0.73,-0.89,0.24,0.15,0.38,0.02,0.09,-0.27,0.66,0.14,0.08,0.2,-0.14,0.07,-0.4,0.19,-0.11,-0.14,0.22,0.1,0.13,-0.52,0.36,-0.06,0.11,0.14,0.02,-0.32,-0.42,-0.11,2.49,-0.16,-0.17,0.09,-0.06,0.05,0.33,0.98,0.19,-0.17,0.03,-0.11,0.01,0.37,0.16,-0.2,-0.44,-0.24,-0.5,0.84,0.01,-0.11,0.14,-0.34,0.11,0.72,-0.37,-0.4,-0.41,-0.15,-0.07,-0.31,-0.04,-0.15,-0.15,0.12,0.3,0.54,0.23,1.04,0.21,0.19,-0.11,-0.21,0.25,0.1,0.53,-0.24,-0.14,-0.06,0.22,-0.32,-0.3,-0.15,-0.3,-0.33,0.37,0.59,0.18,-0.1,-0.11,-0.1,-0.23,0.4,-0.07,0.12,-0.51,-0.31,0.1,-0.04,-0.09,0.24,2.06,-0.11,0.06,0.26,0.3,-0.05,0.32,-0.3,-0.27,-0.33,0.33,-0.09,-0.09,0.28,0.02,0.3,-0.31,-0.08,-0.22,-0.02,-0.23,-0.07,-0.16,-0.22,0.08,-0.84,-0.33,-0.63,-0.1,-0.29,-0.01,0.46,0.33,-0.63,0.57,-0.14,-0.16,0.16,-0.02,0.37,0.31,-0.61,-0.35,-0.05,-0.41,0.78,-0.33,0.51,0.23,-0.16,-0.05,-0.13,-0.35,-0.08,-0.06,1.34,-0.04,-0.33,0.1,-0.44,0.04,0.59,0.12,-0.31,0.54,0.27,0.11,-0.35,-0.46,-0.39,-0.08,-0.16,0.06,-0.07,-0.25,-0.5,-0.32,-0.74,-0.19,-0.09,-0.07,0.26,0.19,-0.16,-0.37,-0.29,-0.01,0.09,0.21,0.01,-0.17,-0.4,-0.19,0.05,-0.18,0.79,0.07,0.5,0.2,-0.07,0.0,0.06,-0.12,0.15,0.18,0.24,0.21,0.06,0.45,0.11,-0.01,-0.22,0.45,0.14,1.12,-0.17,-0.75,0.08,0.02,0.56,-0.25,0.28,0.21,-0.18,0.21,-0.34,-0.21,0.38,-0.08,0.35,-0.09,-0.03,0.29,-0.2,0.35,-0.44,0.09,0.61,0.04,0.0,0.26,0.79,0.27,0.07,0.24,-0.24,-0.01,0.11,-0.6,0.2,-0.12,-0.1,0.01,0.07,0.6,-0.24,0.04,-0.28,0.82,0.25,0.18,-0.25,1.66,0.21,1.19,-0.03,-0.08,-0.46,0.03,-0.07,0.39,-0.17,0.14,0.3,-0.25,-0.05,0.04,0.11,-0.24,-0.42,0.03,-0.19,0.38,0.18,-0.09,-0.07,-0.33,0.08,0.23,0.31,0.35,-0.44,0.6,0.09,0.19,0.19,0.66,-0.09,-0.42,0.14,-0.06,-0.52,1.55,0.46,0.32,-0.04,0.17,-0.22,0.01,0.07,-0.39,0.16,0.14,-0.27,-0.12,0.17,0.09,-0.72,0.01,-0.33,1.29,-0.26,-0.59,0.3,-0.32,0.7,-0.1,-0.27,0.0,0.16,-0.3,-0.06,0.4,0.37,-0.15,0.18,0.06,-0.36,0.01,0.19,-0.31,0.51,0.25,-0.03,-0.14,-0.21,-0.17,0.18,0.15,0.35,0.1,0.09,0.83,-0.08,-0.38,0.05,0.24,-0.1,0.13,-0.11,-0.41,0.22,0.24,0.02,0.09,0.17,0.02,0.18,-0.45,-0.03,-0.13,0.25,1.2,0.43,0.14,1.3,0.01,0.23,1.27,0.18,-0.33,-0.18,0.61,0.39,-0.3,0.31,-0.31,0.04,-0.09,0.23,0.26,0.42,0.08,0.01,-0.02,-0.4,0.16,-0.17,0.25,0.12,-0.05,-0.14,-0.07,0.39,-0.17,-0.12,-0.5,0.01,-0.35,0.41,-0.22,0.09,0.24,-0.26,0.67,-0.13,0.78,-0.14,0.09,0.55,-0.07,-0.14,-0.05,-0.15,-0.33,0.32,-0.01,-0.27,-0.16,-0.32,0.4,0.07,-0.17,0.04,0.94,0.24,-0.09,-0.21,0.09,-0.13,0.31,-0.03,0.05,0.28,0.31,-0.24,-0.13,0.27,-0.12,-0.04,0.11,-0.06,0.03,-0.37,0.27,-0.4,-0.15,-0.06,-0.15,-0.17,1.36,0.04,0.28,-0.12,0.39,0.22,1.1,-0.32,-0.03,-0.37,0.94,-0.14,-0.03,0.23,0.45,0.03,0.21,0.32,-0.12,0.14,0.18,0.21,-0.08,-0.34,0.04,-0.09,-0.08,-0.12,0.5,0.16,0.31,0.62,0.21,-0.25,-0.35,0.34,-0.09,0.38,-0.01,0.11,-0.66,0.58,-0.07,0.44,1.51,0.23,-0.45,0.57,-0.37,-0.07,0.34,0.17,-0.31,0.08,0.01,-0.18,-0.34,0.83,-0.14,-0.23,0.15,0.08,0.0,0.25,-0.13,0.12,0.02,0.49,-0.06,0.04,0.08,-0.29,-0.39,0.14,0.28,-0.02,0.58,1.29,0.03,-0.31,1.09,-0.47,0.1,-0.21,0.15,-0.11,0.35,0.84,0.02,-0.09,-0.23,0.07,-0.44,0.22,0.28,-0.04,0.02,-0.19,-0.43,0.29,0.26,0.26,1.23,-0.07,-0.24,0.77,0.31,-0.21,0.17,0.16,0.06,-0.74,0.09,-0.22,-0.51,0.21,0.51,0.22,0.41,0.17,-0.13,-0.36,-0.19,0.01,-0.46,0.18,-0.27,0.31,1.05,0.76,-0.16,-0.05,0.97,-0.22,0.33,-0.19,-0.06,-0.05,0.6,-0.33,0.15,1.41,-0.02,-0.11,-0.02,0.06,0.9,0.2,-0.01,-0.01,-0.14,-0.45,-0.37,-0.41,-0.24,0.36,-0.04,0.53,0.11,0.92,0.15,-0.33,0.02,0.09,-0.04,-0.28,0.17,-0.06,-0.16,0.03,0.27,0.28,0.2,-0.42,0.18,0.02,-0.62,-0.27,-0.25,0.29,-0.3,0.66,-0.08,-0.37,-0.21,0.42,-0.31,-0.21,0.13,-0.39,0.21,0.79,0.83,0.12,-0.03,0.13,-0.22,1.68,-0.33,0.13,-0.15,0.13,-0.04,0.0,0.05,-0.04,0.3,0.28,0.08,0.05,-0.22,-0.08,0.7,-0.05,0.36,0.3,-0.07,-0.1,0.22,0.16,-0.19,-0.07,0.74,0.02,-0.09,-0.26,0.28,0.18,0.03,0.06,-0.04,-0.13,-0.3,-0.18,-0.36,0.04,-0.12,1.32,-0.45,0.0,0.12,0.16,-0.1,-0.08,0.09,-0.18,-0.27,-0.27,1.04,0.1,-0.27,-0.14,-0.15,-0.11,-0.17,0.2,0.45,-0.58,-0.25,1.15,-0.35,-0.2,0.12,0.12,1.96,-0.18,-0.25,-0.42,0.02,0.19,0.27,-0.23,-0.01,-0.16,-0.25,0.0,-0.25,-0.39,-0.09,-0.27,-0.29,0.07,0.08,0.27,-0.11,0.24,0.0,0.29,0.61,-0.2,0.24,0.01,-0.2,-0.09,-0.42,-0.24,-0.38,-0.34,0.06,-0.2,1.31,1.06,0.47,0.11,-0.04,-0.07,0.79,-0.12,0.01,0.12,-0.1,-0.44,-0.14,1.67,0.06,0.06,0.86,-0.16,0.09,-0.04,-0.31,0.35,-0.01,-0.18,0.1,-0.16,0.5,0.04,-0.49,1.45,-0.02,-0.31,-0.03,0.77,-0.23,0.49,0.01,-0.48,0.05,0.11,-0.18,-0.32,0.31,-0.27,1.45,0.14,0.39,-0.14,-0.28,-0.21,1.17,0.41,0.52,0.12,0.3,0.18,-0.2,-0.15,-0.05,0.23,0.08,0.36,0.12,-0.33,-0.13,-0.12,0.14,-0.35,0.39,-0.3,-0.21,-0.09,-0.19,0.15,-0.39,-0.4,0.58,0.36,-0.13,-0.14,0.12,0.03,-0.21,0.25,0.29,1.44,-0.43,-0.12,0.47,-0.35,-0.23,0.08,0.03,-0.28,-0.16,0.56,0.04,0.44,0.22,-0.16,-0.05,-0.22,-0.18,0.09,0.06,0.26,-0.07,-0.01,1.52,0.11,-0.07,0.37,0.22,0.23,1.72,0.01,-0.14,0.15,0.23,-0.17,0.2,-0.13,0.47,-0.53,-0.44,-0.41,-0.35,0.13,0.54,-0.12,0.1,1.36,-0.52,0.43,0.33,-0.22,-0.22,0.49,0.13,-0.56,-0.17,-0.11,-0.06,0.47,0.11,-0.17,-0.7,0.26,0.37,-0.48,0.26,-0.29,0.02,-0.15,-0.23,-0.14,1.38,-0.34,0.08,0.84,0.11,0.21,0.06,-0.11,-0.27,0.03,-0.04,-0.08,-0.23,0.43,0.26,0.5,-0.11,0.16,0.05,-0.25,-0.05,2.43,0.48,0.05,1.37,-0.38,0.18,-0.32,0.32,-0.37,0.25,0.23,1.75,-0.17,-0.16,0.51,-0.04,-0.38,0.03,0.04,-0.29,0.05,-0.19,-0.57,-0.42,-0.37,0.32,0.18,0.4,0.22,0.1,0.16,2.5,-0.06,0.15,-0.48,0.62,0.06,0.08,-0.41,-0.21,-0.09,0.26,0.13,0.15,-0.25,-0.12,0.29,-0.32,0.93,0.21,0.46,0.19,0.01,0.66,-0.07,-0.11,0.49,0.07,0.6,0.34,1.34,0.14,0.17,0.03,-0.25,0.49,0.27,-0.21,-0.07,0.18,0.87,-0.77,-0.08,-0.42,0.08,0.7,0.1,-0.15,0.0,0.09,0.15,1.31,0.6,0.0,-0.07,0.03,-0.13,-0.26,-0.23,-0.73,0.48,0.26,0.01,0.28,0.2,0.12,0.21,0.22,-0.19,-0.07,-0.3,0.22,-0.24,-0.47,0.27,0.32,0.49,0.41,0.06,-0.18,2.49,0.0,1.93,0.54,0.11,0.28,-0.47,0.03,0.21,-0.29,1.66,-0.14,-0.25,-0.28,0.09,-0.12,-0.06,-0.35,0.53,0.28,-0.11,-0.04,-0.26,-0.09,-0.11,0.23,-0.38,0.11,0.16,-0.14,-0.36,-0.06,-0.34,0.4,0.13,0.2,0.11,0.09,-0.25,-0.07,-0.18,0.16,-0.17,-0.01,-0.14,0.12,1.01,-0.26,0.82,0.05,0.32,-0.16,-0.27,-0.12,-0.22,-0.27,-0.25,-0.12,0.86,0.22,-0.85,-0.2,0.14,-0.25,-0.05,-0.4,0.98,0.22,0.35,-0.17,0.02,0.09,0.27,0.06,-0.01,0.06,-0.33,-0.12,-0.06,-0.21,0.09,-0.09,0.91,0.83,0.04,0.05,0.46,-0.02,-0.1,-0.1,-0.07,0.03,-0.13,-0.25,0.34,-0.17,-0.14,-0.35,0.88,0.11,0.18,-0.03,0.0,-0.21,0.37,0.75,-0.36,0.15,-0.08,-0.24,-0.14,0.08,-0.22,-0.31,-0.17,0.21,-0.31,-0.24,0.47,-0.29,0.29,-0.31,-0.3,0.57,0.44,0.45,-0.22,0.38,0.15,0.4,0.28,0.38,0.13,-0.29,0.15,-0.25,-0.41,0.45,0.03,0.24,0.47,0.11,-0.11,-0.12,-0.2,-0.19,0.32,-0.17,0.44,0.27,0.32,0.38,0.36,-0.37,-0.13,-0.34,0.08,-0.08,-0.06,0.24,-0.42,-0.2,0.29,-0.25,0.25,0.27,0.14,-0.38,-0.4,-0.27,-0.09,-0.18,0.03,-0.25,0.88,0.1,-0.16,0.63,0.06,0.54,0.03,-0.01,1.5,-0.08,1.03,0.09,-0.24,1.83,-0.48,0.01,-0.11,-0.2,0.15,-0.22,0.07,0.24,-0.07,-0.32,0.0,-0.05,1.08,-0.09,0.09,0.09,1.55,-0.54,-0.12,-0.5,0.17,0.2,-0.15,-0.25,-0.32,-0.12,-0.22,-0.25,0.02,-0.06,-0.31,0.71,-0.2,0.26,-0.2,0.03,0.12,-0.23,-0.17,0.72,-0.05,0.26,-0.23,-0.13,0.19,-0.11,-0.42,-0.18,0.58,0.18,0.05,-0.06,-0.18,-0.44,0.13,0.72,-0.39,-0.17,0.16,-0.11,0.12,-0.27,-0.19,0.55,-0.23,0.11,1.08,-0.16,0.13,-0.39,-0.04,1.54,-0.15,-0.11,0.76,0.1,0.25,-0.01,0.26,-0.09,-0.01,-0.26,-0.08,0.06,0.11,-0.24,-0.24,0.28,-0.25,0.21,-0.09,-0.28,0.43,0.08,0.34,-0.2,0.3,-0.09,-0.33,-0.09,-0.68,0.18,-0.18,-0.28,0.06,0.48,0.28,-0.26,0.01,-0.09,0.08,-0.13,0.69,0.38,0.47,-0.06,0.09,0.18,-0.32,0.01,-0.5,0.23,1.89,0.05,-0.23,-0.53,1.23,-0.24,-0.36,0.03,0.03,-0.2,-0.28,-0.01,-0.18,-0.1,0.39,0.21,-0.15,0.21,0.4,0.54,-0.14,0.31,-0.33,-0.24,0.82,0.62,0.0,-0.18,-0.01,-0.29,-0.2,-0.09,0.71,0.72,0.36,0.08,-0.45,0.16,-0.31,-0.08,0.02,0.28,-1.06,-0.19,0.3,1.29,-0.18,0.23,0.09,0.17,0.25,-0.17,0.91,1.0,-0.22,0.33,-0.73,0.0,0.01,-0.12,0.02,-0.55,1.58,-0.16,-0.27,0.02,0.14,-0.65,0.19,0.46,0.21,-0.08,-0.2,-0.23,-0.08,0.02,-0.51,0.12,0.29,0.1,0.32,-0.28,0.2,-0.16,0.01,-0.34,-0.02,-0.02,-0.3,0.07,0.09,-0.26,0.2,0.15,0.12,0.0,0.41,0.22,-0.45,-0.03,0.23,0.12,-0.37,-0.05,-0.14,0.24,-0.37,0.02,-0.12,-0.29,-0.15,-0.02,0.08,0.34,-0.11,-0.28,-0.29,-0.04,-0.37,0.27,0.44,0.2,0.5,-0.22,-0.48,0.9,1.57,-0.33,0.06,1.37,0.19,0.28,-0.1,0.12,0.29,-0.2,0.84,-0.43,0.21,0.87,0.01,0.36,0.24,0.02,-0.27,-0.2,-0.12,-0.04,0.26,-0.45,0.33,0.11,-0.37,-0.54,-0.4,0.41,-0.14,-0.03,0.67,-0.1,-0.23,0.09,0.2,-0.25,0.03,0.09,-0.04,-0.2,-1.04,0.74,-0.1,-0.55,-0.12,0.3,0.08,0.39,-0.05,0.12,0.82,0.1,0.11,-0.11,-0.41,0.06,-0.39,-0.01,0.28,0.82,-0.43,0.37,0.72,-0.1,-0.05,0.27,0.37,0.22,0.09,-0.46,0.08,-0.25,0.15,0.29,-0.21,-0.01,0.01,0.81,0.17,0.17,0.05,0.49,-0.01,0.36,0.44,0.13,-0.11,-0.34,-0.42,-0.04,-0.52,0.12,0.25,-0.06,-0.52,0.03,-0.28,-0.15,-0.53,0.49,0.4,0.04,0.07,-0.13,-0.34,0.02,-0.13,-0.48,-0.46,-0.17,-0.27,0.61,0.09,0.17,-0.01,0.35,-0.15,-0.61,-0.37,0.27,-0.12,-0.14,0.58,-0.53,-0.01,0.55,0.06,-0.21,0.2,-0.08,-0.27,0.11,0.46,-0.28,1.28,-0.14,-0.49,-0.05,-0.42,0.55,-0.31,0.64,-0.25,0.04,0.16,-0.33,0.59,0.15,0.16,-0.11,-0.34,-0.18,-0.16,-0.07,0.21,0.17,0.15,-0.12,0.03,-0.24,-0.57,-0.1,0.11,0.05,0.52,-0.42,0.0,0.15,0.08,0.03,0.0,0.19,0.01,0.47,-0.19,-0.2,0.23,-0.62,-0.28,0.12,-0.13,-0.17,0.13,-0.41,0.13,0.38,1.69,0.2,0.08,0.33,0.35,0.08,0.68,1.41,0.0,0.0,0.43,-0.46,-0.3,-0.16,-0.41,0.45,-0.54,-0.17,-0.01,-0.41,-0.14,0.11,-0.2,-0.49,0.24,-0.18,-0.22,-0.1,0.53,-0.08,-0.04,1.24,0.88,0.09,0.67,-0.14,0.92,-0.09,0.07,-0.02,0.23,0.01,0.06,-0.12,0.22,-0.1,-0.3,-0.5,0.27,-0.12,1.34,-0.25,0.98,0.24,-0.08,0.3,0.35,-0.18,-0.07,0.01,-0.09,0.14,-0.18,-0.2,0.1,-0.15,0.25,0.3,0.17,-0.12,0.4,-0.22,0.78,-0.14,0.08,0.21,0.04,0.18,-0.12,0.23,-0.01,0.33,0.27,-0.15,0.31,0.13,-0.58,0.28,0.02,-0.34,-0.22,0.13,-0.77,-0.35,0.11,-0.33,-0.47,0.07,-0.32,0.47,-0.16,0.12,1.6,0.45,0.08,0.08,-0.12,-0.14,0.01,0.62,-0.2,-0.17,-0.04,-0.88,1.19,0.02,0.17,-0.11,-0.2,-0.12,0.07,0.24,0.43,0.03,0.43,0.78,-0.46,-0.22,0.14,-0.08,-0.08,0.95,-0.39,0.05,-0.03,-0.36,-0.23,0.22,0.32,-0.1,-0.75,0.44,0.1,0.15,0.41,-0.12,0.08,0.43,0.1,0.29,-0.09,0.01,2.07,-0.04,1.64,-0.28,1.08,0.01,0.44,0.29,0.01,0.29,0.35,0.45,-0.03,0.43,0.2,0.07,-0.05,0.7,1.13,-0.26,-0.27,-0.56,-0.36,-0.11,-0.23,0.16,0.15,0.3,-0.02,0.27,0.19,-0.03,-0.13,0.19,-0.29,-0.01,1.17,0.19,-0.11,-0.19,0.16,0.06,0.05,0.9,-0.07,-0.14,0.56,-0.18,-0.2,0.4,-0.19,0.15,1.71,-0.13,0.13,0.31,0.22,-0.04,0.38,0.24,-0.23,0.07,-0.1,-0.05,-0.21,0.1,-0.4,0.02,-0.29,-0.39,0.25,0.48,-0.06,-0.12,-0.05,-0.55,-0.2,-0.32,-0.24,0.05,-0.1,0.01,-0.13,-0.38,-0.06,-0.08,-0.12,0.49,-0.29,0.03,0.47,0.22,1.84,1.02,1.2,0.04,-0.05,0.14,0.61,-0.44,0.14,-0.5,0.0,-0.32,0.39,-0.11,-0.19,-0.1,0.76,0.27,0.5,0.05,0.71,0.23,0.03,-0.4,0.01,-0.45,-0.12,-0.08,-0.2,1.27,0.53,-0.04,-0.35,0.09,-0.18,-0.07,0.35,0.11,0.3,-0.23,0.1,0.02,0.17,0.05,-0.04,-0.55,-0.61,-0.11,-0.23,0.33,-0.04,0.17,0.19,0.26,0.37,-0.43,-0.24,-0.31,-0.26,0.05,-0.27,-0.27,0.42,-0.42,0.08,0.17,-0.04,0.43,-0.34,0.82,0.38,-0.24,-0.19,0.34,-0.5,-0.07,0.39,0.4,0.17,0.43,0.11,0.01,0.95,0.07,1.46,-0.1,1.09,-0.11,-0.06,-0.02,-0.48,-0.21,0.01,-0.15,-0.3,-0.23,-0.39,0.32,-0.29,1.37,-0.18,0.3,-0.18,0.12,-0.07,-0.26,-0.01,0.71,-0.38,-0.08,0.22,-0.18,0.08,-0.2,0.08,0.04,0.01,0.35,-0.23,-0.39,-0.18,-0.11,0.45,-0.2,0.22,-0.23,-0.45,-0.44,0.22,-0.23,-0.18,0.1,-0.3,-0.23,0.09,-0.13,0.05,0.84,-1.31,0.5,-0.19,0.18,0.01,0.06,-0.15,-0.38,-0.56,0.12,-0.43,0.59,0.08,-0.14,-0.13,0.62,0.19,0.0,-0.31,0.08,0.4,0.07,-0.06,-0.06,0.12,0.06,0.21,0.24,-0.12,-0.02,-0.02,-0.11,0.06,-0.11,-0.25,0.03,-0.08,-0.09,0.95,0.28,0.03,-0.28,-0.07,0.02,2.26,-0.05,-0.16,0.3,-0.65,0.7,1.13,-0.03,-0.58,-0.18,-0.42,-0.13,-0.21,0.04,-0.24,-0.07,-0.03,-0.09,-0.29,-0.51,0.19,-0.29,0.16,0.45,1.2,0.78,0.14,0.89,0.63,-0.08,-0.13,0.12,0.29,-0.45,1.02,-0.77,-0.12,0.4,0.03,0.5,0.71,0.23,0.11,-0.03,0.19,-0.1,-0.18,0.39,1.06,-0.16,0.28,0.02,0.15,-0.32,0.38,0.04,0.5,-0.05,-0.05,-0.16,-0.06,-0.09,0.53,-0.53,-0.09,0.67,-0.13,0.01,0.99,0.16,-0.17,-0.38,-0.22,0.01,0.06,-0.01,-0.12,-0.41,0.03,0.07,0.1,0.29,0.23,0.04,0.48,0.11,-0.31,-0.02,-0.17,-0.34,0.44,-0.33,0.07,-0.13,-0.27,0.26,0.77,-0.28,-0.11,0.02,0.11,-0.04,-0.17,-0.14,-0.28,-0.22,-0.47,0.37,-0.17,0.28,-0.06,-0.14,-0.1,0.12,0.93,0.18,0.07,-0.2,1.59,-0.19,0.24,-0.18,0.39,-0.68,0.32,0.22,0.42,1.2,-0.06,-0.53,-0.04,0.55,-0.34,1.08,0.06,0.08,-0.17,-0.37,-0.5,-0.03,0.39,-0.48,-0.26,-0.08,0.56,0.66,0.6,-0.24,0.22,-0.02,1.09,-0.21,0.04,0.85,-0.16,-0.1,0.27,0.0,1.2,0.24,-0.45,-0.03,-0.57,-0.4,-0.4,-0.33,0.2,-0.37,0.03,-0.3,0.2,0.06,0.15,-0.04,0.39,-0.39,-0.03,-0.71,-0.33,0.04,-0.14,-0.14,-0.05,0.04,-0.25,0.46,-0.09,-0.15,-0.22,0.03,0.19,0.53,1.27,-0.41,0.04,0.03,-0.22,-0.15,0.86,-0.43,-0.09,-0.09,0.39,0.37,0.04,-0.05,-0.16,0.15,0.26,1.38,0.68,-0.21,0.11,-0.5,-0.1,-0.12,-0.13,-0.12,1.28,0.24,-0.45,1.13,0.26,1.31,0.32,0.03,-0.29,0.2,-0.1,-0.03,1.12,0.06,-0.28,-0.21,-0.11,1.9,0.04,0.23,0.15,-0.11,-0.31,-0.02,0.09,0.18,0.09,0.41,0.06,0.05,-0.08,-0.28,0.54,0.09,-0.16,-0.14,-0.04,0.21,0.34,0.14,-0.16,0.04,0.99,-0.15,0.55,-0.02,0.18,-0.01,-0.08,-0.1,0.05,-0.01,0.93,0.09,-0.01,0.18,-0.27,0.28,-0.18,0.08,0.3,0.51,-0.26,-0.08,0.08,-0.27,0.72,-0.27,0.11,-0.4,0.09,0.03,-0.1,0.17,-0.07,0.34,-0.25,0.19,-0.23,-0.18,-0.18,-0.25,-0.35,0.38,-0.21,0.05,0.09,-0.39,-0.04,-0.14,0.25,0.19,0.16,-0.08,0.05,-0.42,0.26,-0.15,-0.07,-0.05,0.12,0.3,0.02,-0.23,-0.07,0.14,0.15,0.87,0.01,0.07,0.32,-0.17,0.08,-0.21,-0.07,-0.08,0.62,0.5,-0.23,-0.15,0.02,0.03,0.05,-0.21,-0.05,0.82,0.27,-0.4,-0.46,-0.02,-0.44,0.03,0.23,-0.2,0.2,0.82,0.05,0.27,0.3,-0.07,0.12,-0.04,0.42,0.28,0.12,0.0,-0.34,0.87,0.18,-0.19,0.28,-0.11,-0.2,-0.35,0.34,-0.01,-0.71,0.02,-0.09,1.64,-0.07,0.04,0.38,-0.1,-0.1,0.8,-0.04,-0.11,-0.35,0.04,0.07,-0.03,-0.16,-0.12,0.08,-0.12,-0.08,-0.07,1.11,-0.16,0.47,0.3,1.17,0.02,0.4,-0.08,0.35,0.42,-0.18,-0.52,-0.31,0.27,0.1,0.24,-0.29,0.01,-0.32,0.41,0.0,-0.29,-0.38,0.44,-0.72,0.23,-0.13,-0.12,-0.16,-0.13,-0.29,0.5,0.31,0.12,-0.25,0.33,0.47,-0.11,0.3,0.2,0.33,0.02,-0.2,0.1,0.21,-0.2,-0.83,-0.55,0.48,-0.21,-0.03,-0.1,-0.47,-0.31,0.38,-0.03,-0.3,-0.1,-0.52,0.09,0.81,-0.13,1.18,-0.01,-0.06,1.1,0.2,-0.22,0.74,0.13,0.2,-0.27,0.5,-0.14,-0.17,0.11,0.12,-0.16,0.04,0.23,-0.2,0.11,0.15,-0.21,0.15,0.04,0.01,-0.16,-0.3,0.66,-0.04,-0.2,-0.2,0.39,1.37,-0.12,0.13,0.87,-0.33,0.03,1.06,-0.05,-0.17,-0.16,0.25,0.92,-0.16,0.16,-0.03,-0.28,0.01,0.55,0.23,-0.23,0.13,0.24,0.19,-0.39,-0.41,-0.3,0.08,-0.04,0.38,-0.09,-0.28,-0.15,-0.07,0.39,0.13,-0.04,0.11,-0.31,-0.36,-0.04,1.08,0.05,-0.09,1.18,0.37,0.18,0.24,0.35,-0.33,0.45,0.02,0.2,0.99,-0.24,0.47,1.67,0.06,-0.03,0.32,0.25,0.16,-0.21,-0.56,0.53,0.18,-0.08,0.79,-0.15,0.62,1.27,-0.14,0.67,-0.26,-0.5,0.32,-0.1,0.46,-0.16,0.34,-0.55,-0.62,0.06,0.58,0.42,-0.55,-0.06,0.2,-0.11,-0.21,-0.24,-0.23,-0.08,-0.32,-0.12,-0.14,-0.06,0.37,-0.07,-0.06,-0.29,0.08,0.0,0.05,-0.12,0.18,0.29,0.02,-0.19,0.63,-0.18,-0.27,0.29,0.19,0.09,-0.32,0.43,-0.18,-0.02,0.28,-0.06,-0.32,0.02,-0.44,0.32,0.2,-0.07,-0.64,1.1,1.44,-0.31,0.0,0.46,-0.06,-0.15,-0.68,-0.09,0.07,1.21,-0.09,0.12,0.45,0.86,-0.16,-0.15,0.02,-0.13,0.98,-0.1,1.3,-0.17,-0.09,0.07,0.1,0.48,-0.41,0.19,0.6,0.16,-0.27,0.41,-0.05,-0.09,-0.6,0.36,0.34,0.04,0.09,-0.25,0.1,-0.18,0.15,0.11,-0.14,0.14,-0.21,1.03,0.14,-0.31,-0.17,0.8,0.26,-0.14,0.01,-0.06,0.13,-0.12,0.84,0.41,-0.35,0.42,-0.05,0.11,-0.08,-0.17,-0.02,-0.53,0.2,0.01,0.36,0.42,-0.03,-0.11,0.35,0.46,0.23,1.04,-0.67,-0.43,0.72,-0.08,0.41,-0.05,0.0,0.53,0.29,0.22,-0.06,0.21,-0.27,0.23,0.4,0.21,0.32,0.4,-0.17,0.64,0.02,-0.11,-0.02,0.18,0.47,-0.05,0.57,-0.04,-0.07,0.03,0.11,-0.39,-0.14,-0.24,0.25,0.96,-0.05,0.37,-0.19,0.83,-0.03,-0.51,-0.2,-0.45,-0.16,0.51,0.02,-0.32,-0.01,-0.2,0.33,1.03,0.4,-0.02,-0.38,0.15,-0.08,-0.58,0.02,0.11,0.21,-0.06,0.36,-0.09,0.48,-0.1,-0.1,-0.42,-0.02,0.1,0.44,-0.32,-0.14,-0.37,-0.03,0.26,-0.46,-0.34,0.12,-0.1,0.11,0.19,0.03,-0.24,0.27,0.48,1.28,-0.07,1.73,-0.18,-0.24,-0.16,-0.12,-0.18,0.05,0.18,1.02,0.35,0.04,-0.27,-0.17,0.12,-0.13,0.4,-0.08,-0.16,-0.06,-0.44,-0.25,0.97,0.61,-0.66,0.09,0.31,0.07,-0.33,0.44,-0.12,0.33,0.12,-0.17,-0.25,-0.28,1.54,0.38,-0.31,-0.41,-0.39,0.37,-0.16,0.14,-0.02,0.47,1.08,-0.02,-0.13,-0.29,0.27,0.02,0.34,0.56,0.25,0.33,0.18,-0.5,-0.31,-0.13,1.99,0.99,-0.06,-0.69,0.39,-0.08,0.06,-0.17,0.18,-0.23,0.6,-0.04,0.22,-0.34,-0.22,1.07,-0.07,-0.04,0.15,0.55,-0.74,-0.39,0.03,0.41,0.13,-0.06,-0.2,-0.44,0.16,-0.19,-0.06,0.36,-0.53,-0.28,-0.47,0.3,-0.47,-0.49,0.13,-0.36,0.37,0.33,-0.15,0.16,-0.28,0.35,-0.26,-0.26,0.11,-0.05,0.16,-0.18,-0.01,0.0,0.5,0.07,0.04,0.06,-0.47,0.04,-0.08,0.0,0.22,-0.05,0.32,0.43,-0.12,0.09,0.0,0.23,0.12,-0.12,-0.03,-0.05,0.01,-0.19,0.21,-0.19,-0.02,-0.04,0.19,-0.1,0.48,-0.04,-0.16,-0.44,0.21,-0.29,-0.22,0.08,-0.45,0.25,-0.13,-0.29,-0.14,0.25,0.47,-0.1,-1.22,-0.11,0.18,-0.02,0.12,-0.03,-0.28,1.18,0.07,0.99,0.03,-0.05,0.07,0.39,-0.06,-0.16,0.25,-0.39,-0.2,-0.23,0.25,0.21,-0.11,-0.02,-0.38,-0.03,0.13,0.06,-0.24,-0.41,0.21,0.52,0.39,0.19,-0.45,-0.62,0.93,0.36,-0.42,-0.12,0.21,0.18,0.25,-0.05,-0.1,-0.18,0.03,0.25,0.0,-0.09,-0.38,0.02,-0.55,-0.05,-0.17,-0.16,0.0,0.04,1.31,0.18,0.81,0.0,0.38,0.06,-0.53,-0.15,0.16,-0.04,0.22,-0.16,-0.13,0.05,0.02,-0.22,0.04,-0.42,0.16,-0.12,-0.08,0.4,0.15,0.06,-0.41,0.34,0.18,0.11,0.43,-0.43,-0.51,0.01,0.39,0.02,-0.42,0.33,-0.12,0.06,0.01,0.17,-0.05,0.28,-0.13,0.61,1.27,-0.18,0.09,0.28,0.21,0.0,-0.34,-0.31,-0.11,0.13,-0.02,-0.46,-0.37,-0.11,-0.12,-0.13,-0.34,0.76,0.16,1.28,0.69,-0.18,0.91,0.12,0.62,-0.22,0.19,-0.06,-0.56,0.28,-0.23,0.1,0.03,0.01,-0.19,0.08,0.64,0.28,-0.48,-0.32,-0.11,0.47,-0.16,0.09,0.34,-0.17,-0.31,-0.59,-0.58,0.03,0.0,0.25,0.01,-0.06,0.32,-0.32,0.32,-0.11,-0.4,0.13,-0.27,0.53,-0.37,-0.34,-0.21,-0.07,-0.13,-0.01,-0.01,-0.07,-0.04,0.98,0.28,-0.39,0.06,-0.14,0.27,0.09,-0.37,0.09,0.18,-0.12,0.45,-0.19,0.14,-0.1,0.13,0.08,-0.06,0.04,1.1,0.68,0.39,-0.07,0.43,0.25,0.19,-0.23,0.3,0.08,-0.05,-0.04,0.07,0.08,0.36,0.44,-0.23,-0.52,0.01,-0.66,0.07,0.26,0.74,0.17,0.25,0.35,-0.69,-0.12,0.21,-0.13,0.62,-0.53,-0.44,0.97,0.46,0.24,-0.22,1.18,0.5,-0.2,1.97,0.18,0.24,-0.38,-0.16,-0.09,0.33,-0.34,0.45,0.22,1.38,-0.01,0.07,-0.05,0.11,0.77,-0.04,0.33,0.15,0.0,0.0,0.14,-0.05,-0.25,0.07,0.24,0.07,0.4,-0.26,0.0,-0.13,0.23,0.62,0.38,-0.01,-0.21,1.1,-0.29,0.0,1.86,0.18,0.15,0.43,-0.1,-0.17,-0.2,-0.62,0.02,0.05,0.0,-0.01,-0.38,-0.66,-0.28,0.23,0.19,-0.32,-0.32,-0.34,-0.16,-0.4,-0.12,-0.17,0.43,-0.07,0.17,-0.02,0.27,-0.07,-0.27,-0.34,-0.02,-0.08,0.19,0.07,0.23,0.29,0.07,0.07,-0.34,0.39,0.04,0.26,0.03,0.08,0.04,1.44,-0.2,-0.06,0.15,0.57,0.78,0.17,0.33,0.38,-0.23,-0.18,0.11,-0.05,-0.13,0.02,0.33,-0.3,-0.15,0.63,0.57,-0.1,-0.43,0.08,0.41,-0.41,0.33,-0.16,0.02,-0.29,0.34,1.15,0.32,0.15,-0.1,-0.39,0.05,1.09,0.06,0.05,0.02,-0.19,0.65,0.32,-0.01,-0.26,-0.13,0.64,-0.07,-0.52,-0.59,0.01,-0.02,0.08,0.22,0.4,0.01,0.45,0.77,-0.21,-0.15,-0.11,0.05,-0.09,0.15,-0.09,-0.37,0.62,-0.23,0.22,2.2,-0.06,-0.39,0.14,0.41,0.36,0.24,-0.23,0.25,0.14,0.18,-0.11,0.01,-0.48,0.11,-0.31,-0.29,0.41,-0.32,-0.08,0.41,-0.24,-0.33,0.2,-0.09,-0.13,0.04,0.54,-0.14,0.44,-0.02,-0.16,0.13,0.15,-0.26,0.81,-0.24,1.3,-0.2,-0.25,-0.07,-0.06,-0.39,0.29,-0.59,0.13,0.07,0.39,0.11,0.3,0.09,2.4,0.27,1.26,-0.24,0.1,-0.14,-0.03,-0.05,-0.36,-0.03,-0.21,-0.07,-0.29,-0.12,-0.07,-0.3,0.0,-0.26,0.17,-0.21,-0.19,0.3,0.14,0.28,-0.13,0.22,0.0,0.4,-0.55,0.15,0.15,0.13,0.23,-0.8,0.86,0.17,-0.16,-0.39,0.15,0.47,0.1,-0.22,0.2,0.19,0.62,-0.09,-0.05,-0.02,-0.1,0.72,-0.19,0.54,0.24,1.06,-0.31,0.93,-0.53,-0.16,0.08,0.01,0.07,0.31,0.06,0.44,0.23,0.71,-0.08,0.38,-0.09,0.07,0.18,1.31,-0.01,-0.09,0.39,-0.11,-0.5,0.35,0.21,0.61,0.53,0.11,-0.13,0.49,-0.09,0.09,-0.64,0.2,0.52,-0.24,0.12,-0.15,-0.18,-0.11,-0.24,-0.26,0.08,0.0,-0.07,-0.13,-0.12,-0.03,0.49,0.31,0.26,-0.26,0.16,-0.17,-0.37,0.34,0.22,0.14,0.07,-0.53,0.08,-0.11,-0.36,-0.25,-0.06,1.2,0.01,-0.25,0.24,0.02,-0.16,0.02,-0.1,-0.82,0.45,0.19,-0.19,1.21,-0.07,0.03,0.05,0.19,0.09,-0.17,-0.23,-0.08,-0.17,-0.01,0.46,-0.3,-0.25,1.1,0.59,-0.13,-0.06,0.01,0.65,-0.08,0.21,-0.33,-0.36,0.19,0.26,-0.45,0.5,0.79,0.27,1.09,-0.01,-0.12,-0.42,0.23,0.12,0.41,-0.15,0.47,-0.08,0.24,-0.02,-0.83,0.03,0.31,0.04,-0.16,-0.29,1.05,0.1,-0.12,0.88,1.39,0.51,0.01,0.0,-0.39,-0.23,0.14,-0.48,0.06,0.41,0.02,-0.03,0.04,-0.02,0.58,-0.13,0.28,0.68,0.49,-0.17,-0.19,-0.44,0.41,0.14,0.07,0.25,-0.06,0.33,-0.23,-0.14,-0.47,-0.34,-0.06,-0.13,0.38,0.26,0.37,0.1,0.54,0.26,0.15,-0.39,-0.17,-0.29,-0.38,-0.43];

        for &quantile in quantiles.iter() {
            let potential_sd = AllData::estimate_middle_normal(&norm_data);

            println!("sd {:?}", potential_sd);

            let pars = potential_sd.unwrap();

            let norm = Normal::new(pars[0], pars[1]).unwrap();
            //let lnorm = LogNormal::new(pars[2], pars[3]).unwrap();

            //find_cutoff(weight_normal: f64, normal_dist: &Normal, log_normal_dist: &LogNormal, data_size: usize, peak_scale: Option<f64>)

            //let cutoff = AllData::find_cutoff(pars[0], &norm, &lnorm, 928331, None);

            //find_qval_cutoff(sorted_data: &[f64], normal_dist: &Normal, peak_scale: Option<f64>

            let mut sorted_data = norm_data.clone();
            sorted_data.sort_unstable_by(|a,b| a.partial_cmp(&b).unwrap());

            let cutoff = AllData::find_qval_cutoff(&sorted_data, &norm, None);
            println!("cutoff {cutoff}");


        }

        let other_norm_data: Vec<f64> = vec![0.67,0.81,0.77,0.53,0.62,0.7,0.8,0.46,0.45,0.35,0.3,0.0,0.29,0.22,0.32,0.12,0.12,0.11,0.18,-0.03,0.13,0.02,0.21,0.07,0.07,-0.07,0.06,0.28,0.15,0.21,0.02,0.02,0.08,0.39,0.22,0.16,-0.04,0.23,0.24,0.09,0.21,0.0,0.31,0.39,0.23,0.01,0.45,-0.06,0.41,0.16,0.13,-0.19,0.04,0.42,0.08,0.42,0.18,0.34,0.45,0.04,0.33,0.54,0.27,0.42,0.36,-0.15,0.31,-0.08,0.28,-0.07,0.25,0.3,0.09,0.05,0.08,0.34,0.41,0.14,0.13,0.28,0.34,0.39,0.15,0.4,0.08,0.41,0.17,0.24,0.44,0.0,0.31,0.01,-0.09,0.28,0.13,0.23,0.17,0.09,0.23,0.04,0.19,0.04,0.26,0.53,0.35,0.42,0.35,0.37,0.05,0.37,0.19,0.18,0.04,0.25,0.13,0.47,0.39,-0.19,-0.06,-0.12,-0.14,-0.21,-0.07,-0.02,0.0,0.48,-0.09,0.31,-0.11,-0.04,0.55,0.14,0.06,0.16,0.06,0.13,-0.03,-0.22,0.03,-0.08,-0.16,0.21,-0.04,-0.11,-0.01,-0.19,0.06,0.0,0.03,0.04,-0.27,-0.09,-0.27,-0.13,-0.13,-0.04,0.08,0.23,0.38,-0.16,-0.1,-0.15,0.01,0.12,0.1,0.21,0.06,0.19,0.26,0.15,-0.03,0.19,0.11,0.01,0.22,-0.07,0.15,0.05,0.41,-0.22,-0.07,0.44,-0.07,0.1,-0.11,-0.19,0.17,-0.19,-0.03,0.35,-0.11,0.5,0.49,-0.06,0.03,0.35,0.26,0.16,0.07,0.34,0.11,0.23,0.08,0.2,0.18,0.18,0.2,-0.1,0.33,0.03,0.06,0.02,-0.15,0.03,0.06,-0.01,0.09,0.37,0.09,-0.2,0.01,0.06,0.04,0.2,0.1,-0.06,-0.1,0.06,0.01,0.08,-0.01,-0.21,-0.04,0.14,0.04,0.22,0.29,0.0,0.13,-0.15,-0.06,-0.23,-0.01,-0.14,0.13,-0.34,0.03,-0.08,-0.33,0.2,-0.36,-0.06,0.22,0.18,0.0,-0.21,0.0,0.04,0.09,-0.17,0.14,0.17,0.01,0.15,0.02,0.1,0.04,0.26,0.39,-0.03,0.33,-0.12,0.23,0.22,0.03,0.17,-0.16,0.05,0.23,-0.11,0.11,0.01,-0.3,0.09,0.23,0.11,0.13,0.24,0.28,0.05,0.3,-0.12,-0.17,0.06,-0.25,0.05,0.39,0.25,0.04,0.3,-0.05,0.06,0.09,0.06,-0.18,0.04,-0.02,0.12,-0.04,-0.02,-0.11,0.04,0.17,0.36,-0.03,0.46,0.3,0.21,0.22,0.13,0.04,0.64,-0.08,0.21,0.09,0.37,0.19,0.43,0.54,0.21,0.12,0.05,0.54,0.11,0.16,0.12,0.31,0.12,0.17,0.31,0.04,0.01,0.04,-0.02,0.12,0.02,0.06,-0.01,-0.08,0.06,-0.03,-0.16,-0.08,-0.23,0.16,-0.66,0.37,-0.15,0.04,-0.07,-0.23,0.14,0.2,0.11,-0.03,0.05,0.14,0.11,-0.03,0.19,0.04,0.18,0.1,0.04,0.18,-0.09,-0.03,-0.05,-0.04,0.13,-0.25,-0.31,-0.09,-0.02,0.24,0.06,0.14,-0.06,-0.19,-0.19,0.27,0.0,-0.34,-0.32,-0.24,-0.28,-0.36,-0.18,-0.56,-0.29,-0.26,-0.37,-0.25,-0.3,-0.22,-0.08,-0.14,-0.36,-0.06,-0.11,-0.12,0.27,0.05,-0.03,-0.08,0.29,0.31,0.26,-0.03,-0.01,0.03,0.38,0.04,-0.21,-0.02,0.06,-0.16,-0.19,-0.15,-0.16,0.02,0.12,-0.14,0.26,0.29,-0.12,-0.2,-0.24,-0.28,-0.01,-0.24,-0.08,-0.06,-0.3,-0.01,-0.16,0.03,0.32,-0.18,0.04,0.04,0.26,0.23,0.04,0.02,0.05,0.13,0.41,0.22,0.04,0.04,0.17,0.47,0.35,0.18,0.21,0.37,0.24,0.13,0.31,0.16,0.02,0.05,0.19,0.44,-0.07,0.1,0.3,0.04,0.82,0.41,-0.05,0.37,0.03,0.02,0.2,0.37,0.33,-0.01,0.27,0.13,-0.05,0.32,-0.08,0.18,-0.22,-0.05,-0.06,-0.24,0.09,0.23,0.08,0.49,-0.27,-0.17,0.24,-0.15,-0.19,0.03,-0.25,-0.19,-0.16,-0.19,-0.34,-0.25,0.1,0.12,-0.06,0.09,0.14,-0.22,0.21,-0.06,0.27,-0.04,0.06,0.26,-0.03,0.12,0.13,0.13,0.32,0.06,-0.08,0.03,0.16,0.09,0.0,-0.03,0.03,0.03,-0.13,-0.14,-0.28,0.01,0.07,-0.18,-0.14,-0.1,0.01,-0.2,0.01,-0.06,-0.14,-0.25,0.15,-0.2,-0.17,0.21,-0.03,0.04,-0.05,0.13,0.22,-0.15,0.08,0.04,0.2,0.44,0.01,-0.09,-0.02,-0.05,-0.14,-0.16,0.19,-0.04,-0.12,-0.07,0.18,-0.4,-0.4,-0.33,-0.35,-0.23,-0.44,-0.38,-0.43,-0.05,-0.42,-0.38,-0.39,-0.31,-0.57,-0.4,-0.53,-0.28,-0.45,-0.45,-0.52,-0.17,-0.44,-0.27,-0.18,-0.27,-0.14,-0.48,-0.46,-0.4,-0.21,-0.32,-0.3,-0.23,-0.21,0.06,0.17,0.02,0.19,0.35,0.03,0.15,-0.03,-0.07,-0.16,-0.08,-0.06,0.01,0.16,-0.16,-0.05,-0.12,-0.02,-0.3,-0.07,-0.14,-0.14,-0.21,-0.1,-0.03,0.03,-0.2,-0.11,0.06,0.1,-0.06,0.23,-0.23,0.14,0.15,0.09,-0.01,0.28,0.02,0.04,-0.27,0.14,0.12,-0.16,0.34,0.03,0.23,-0.03,0.16,0.1,0.28,0.02,0.26,0.23,-0.19,0.0,-0.04,0.04,0.63,0.26,0.3,0.19,0.06,0.27,0.1,0.04,0.06,0.43,0.22,0.1,0.29,0.2,0.3,0.15,0.25,0.3,0.08,0.23,0.4,0.24,0.22,0.09,0.19,0.14,0.33,0.34,-0.07,0.06,0.1,0.0,-0.11,0.23,0.16,-0.15,0.05,-0.07,0.04,0.05,0.02,0.23,-0.13,-0.01,-0.07,-0.08,0.15,0.22,0.01,-0.12,-0.02,0.26,0.57,0.29,0.16,-0.01,0.18,0.16,0.29,0.41,0.45,0.08,0.21,0.19,0.39,0.19,0.26,0.12,0.5,0.02,0.38,0.26,0.51,0.41,0.12,0.31,0.24,-0.09,0.24,0.06,0.17,0.08,0.29,0.25,0.04,0.06,-0.17,0.39,0.32,-0.03,0.21,0.04,0.08,0.14,-0.09,-0.04,0.13,0.16,0.35,0.16,0.11,0.32,0.4,0.22,0.34,0.15,0.33,0.43,0.35,0.4,0.1,0.2,0.42,0.1,0.04,0.23,0.18,-0.05,0.24,0.33,0.42,0.25,-0.05,0.06,0.18,0.18,0.32,0.04,-0.11,-0.07,0.15,-0.14,-0.19,-0.06,0.06,-0.12,-0.09,-0.01,0.19,0.13,-0.06,0.04,0.23,0.04,0.03,0.0,-0.06,0.33,0.16,0.03,0.27,0.16,0.35,0.33,0.03,0.07,0.19,0.09,0.32,0.21,0.21,0.21,0.19,-0.01,0.12,0.1,0.04,0.19,0.23,0.37,0.33,0.4,0.29,0.24,0.13,0.17,0.21,0.08,0.15,0.29,0.11,0.52,0.14,0.11,0.12,-0.06,0.12,0.17,0.04,0.0,0.28,-0.23,0.06,0.02,0.44,-0.06,-0.14,-0.22,-0.11,-0.09,-0.15,-0.18,-0.04,-0.09,-0.35,-0.03,-0.07,-0.12,-0.16,-0.03,-0.21,0.08,0.1,-0.09,0.07,-0.06,-0.17,-0.02,-0.19,-0.08,-0.05,0.06,-0.1,-0.04,0.2,-0.16,0.12,0.01,-0.17,-0.29,-0.2,-0.3,-0.34,-0.33,-0.3,-0.15,-0.16,-0.27,-0.01,-0.37,0.03,-0.45,-0.1,-0.2,-0.14,-0.22,-0.14,-0.16,-0.01,-0.24,-0.18,-0.3,-0.06,0.06,0.08,-0.3,-0.45,0.08,-0.44,-0.22,-0.11,-0.13,-0.25,0.08,-0.38,-0.23,0.1,-0.27,0.0,-0.23,-0.32,-0.46,-0.26,-0.04,-0.06,0.1,-0.02,0.04,0.6,0.13,0.06,0.21,0.01,-0.05,0.34,-0.09,0.06,0.21,0.04,-0.06,-0.04,-0.11,0.25,0.04,-0.09,-0.27,-0.19,-0.03,-0.05,-0.03,-0.14,-0.45,-0.25,-0.04,-0.34,0.06,-0.11,-0.03,-0.07,-0.22,0.0,-0.11,-0.14,-0.2,-0.25,0.01,0.06,0.13,-0.11,0.09,0.01,0.19,0.18,0.06,0.24,-0.22,0.05,0.14,-0.19,0.01,-0.32,0.21,0.23,-0.14,-0.24,0.1,-0.14,-0.07,0.06,-0.16,-0.04,-0.03,-0.1,-0.09,-0.02,0.05,0.19,0.3,0.03,-0.07,0.24,-0.16,0.19,-0.21,-0.25,0.36,-0.24,-0.66,-0.17,-0.07,0.08,0.13,0.06,0.11,-0.16,0.23,0.14,0.3,0.21,0.12,0.37,-0.08,-0.03,0.01,0.2,0.1,0.19,0.18,0.06,0.06,0.1,0.09,0.04,0.05,0.08,0.14,0.33,0.18,-0.21,0.17,0.02,0.12,0.14,0.06,-0.14,0.12,-0.22,0.07,-0.06,0.04,-0.28,-0.28,-0.05,-0.07,-0.15,0.0,-0.14,-0.35,-0.2,0.29,0.05,-0.09,-0.04,0.07,-0.19,0.23,-0.15,-0.26,0.06,-0.04,0.03,-0.06,0.26,-0.25,0.04,0.09,-0.09,-0.13,-0.14,-0.07,-0.3,-0.21,-0.02,-0.43,-0.17,-0.07,-0.24,-0.11,-0.25,0.06,-0.06,-0.08,-0.35,-0.18,-0.29,-0.08,-0.06,-0.07,0.15,-0.25,-0.01,0.16,0.04,-0.04,0.07,-0.16,-0.22,0.15,-0.07,-0.19,0.03,0.07,-0.15,-0.05,-0.38,0.04,-0.16,0.17,-0.6,-0.15,-0.1,-0.39,-0.05,-0.44,-0.19,-0.21,-0.04,-0.38,-0.35,-0.03,0.06,-0.06,-0.22,-0.29,-0.42,-0.27,-0.17,-0.34,0.16,-0.03,-0.1,-0.24,0.06,0.4,0.45,0.28,0.31,0.41,0.25,0.17,0.28,0.24,0.44,0.38,0.4,0.41,0.27,0.17,0.27,0.36,0.17,0.14,0.62,0.28,0.21,0.41,0.31,0.3,0.24,0.1,0.27,0.19,0.24,-0.05,0.14,0.07,0.35,-0.15,0.11,-0.05,0.01,-0.19,0.01,-0.29,-0.38,-0.53,0.2,0.14,0.04,-0.25,0.03,-0.22,-0.19,0.0,-0.25,0.08,-0.21,-0.13,0.02,-0.25,-0.16,-0.05,0.0,0.1,-0.26,-0.07,-0.21,-0.3,0.0,0.22,-0.09,-0.09,0.21,0.1,0.06,0.18,0.06,0.38,0.25,-0.38,0.06,0.29,-0.28,0.04,0.22,-0.1,-0.22,-0.09,0.51,0.2,0.1,0.15,0.04,0.28,0.31,0.06,-0.07,0.01,0.25,0.21,0.1,-0.06,0.03,0.56,0.09,0.25,0.06,0.13,-0.17,0.02,-0.18,0.06,-0.19,-0.05,0.06,-0.22,-0.01,0.3,0.14,-0.3,-0.05,-0.03,0.01,0.22,0.14,0.1,0.16,0.07,-0.08,0.0,-0.03,0.06,-0.01,0.22,-0.06,0.27,0.0,-0.06,-0.25,-0.02,-0.31,0.04,-0.53,-0.17,-0.19,-0.25,-0.38,0.04,-0.27,0.22,-0.13,0.02,-0.05,-0.46,-0.37,-0.3,-0.26,-0.19,-0.27,-0.38,-0.17,-0.44,-0.21,-0.08,0.05,-0.23,-0.09,0.02,-0.21,0.06,-0.06,-0.05,-0.15,-0.01,-0.21,-0.27,-0.57,-0.32,-0.02,-0.24,0.0,0.1,-0.03,-0.24,-0.39,-0.29,0.13,-0.13,-0.35,-0.3,0.28,-0.1,-0.2,-0.2,-0.1,0.14,0.34,0.01,0.02,0.06,0.06,0.45,0.09,0.35,0.12,0.41,0.33,0.38,0.23,0.37,0.69,0.14,0.31,0.24,0.15,0.14,0.27,0.4,-0.04,0.46,-0.11,0.3,0.16,0.26,0.33,0.09,-0.13,0.25,0.06,0.01,-0.04,-0.02,0.01,-0.06,-0.24,-0.11,-0.26,0.21,0.05,-0.25,-0.23,-0.01,-0.32,-0.23,-0.16,-0.13,-0.33,-0.05,-0.15,0.04,-0.15,-0.43,-0.32,-0.23,-0.37,-0.14,-0.23,-0.18,0.01,0.0,-0.16,-0.11,-0.03,0.22,0.04,0.31,-0.19,-0.06,-0.27,-0.26,0.13,-0.17,-0.03,0.16,-0.01,0.4,0.01,0.24,0.15,0.23,0.34,0.3,0.1,0.26,0.32,0.33,0.19,0.35,0.22,0.19,0.32,0.32,0.38,0.41,0.29,0.04,0.03,0.32,0.08,-0.01,0.26,0.46,0.1,-0.06,0.13,-0.24,-0.02,0.11,-0.02,-0.04,0.17,0.1,0.03,-0.04,-0.25,0.09,0.07,-0.23,-0.32,-0.05,0.2,-0.23,-0.23,-0.1,-0.3,-0.14,-0.08,-0.12,-0.05,-0.27,-0.3,0.04,-0.21,-0.24,-0.3,-0.11,-0.12,-0.03,-0.43,-0.21,0.04,-0.31,-0.17,-0.4,-0.13,-0.26,0.1,0.06,-0.01,0.33,-0.14,-0.21,-0.26,-0.21,-0.19,0.07,-0.08,0.11,0.16,0.23,0.19,0.1,-0.11,-0.01,0.12,-0.17,-0.06,-0.3,-0.17,-0.19,0.1,-0.19,-0.12,-0.16,-0.11,0.23,-0.08,0.0,-0.19,-0.11,-0.25,-0.07,0.15,-0.13,-0.16,0.0,0.27,-0.08,0.06,-0.19,0.23,-0.03,-0.24,-0.2,0.08,-0.1,-0.04,-0.17,-0.03,-0.19,0.11,-0.07,-0.2,-0.15,-0.14,-0.09,-0.03,-0.31,-0.37,-0.25,0.17,-0.18,-0.18,-0.29,0.02,-0.03,-0.09,-0.1,-0.15,-0.2,0.04,0.4,-0.18,-0.08,0.01,0.06,-0.07,-0.05,0.11,-0.31,-0.06,-0.06,0.06,-0.22,-0.22,-0.4,-0.29,-0.23,-0.26,0.01,-0.1,-0.29,-0.11,0.0,-0.25,-0.33,-0.14,-0.22,-0.44,-0.05,-0.22,-0.03,0.04,-0.12,0.29,0.31,0.04,0.06,0.22,0.09,-0.25,0.02,-0.15,0.13,0.12,0.02,0.01,0.04,0.21,0.33,0.09,0.16,0.25,0.06,0.04,-0.03,0.05,-0.16,0.12,-0.08,0.14,-0.03,-0.28,0.01,-0.06,0.03,0.01,0.13,-0.06,-0.08,-0.09,0.46,-0.01,-0.15,-0.07,-0.17,0.04,-0.03,0.14,0.0,0.08,-0.15,0.12,0.15,0.39,0.08,0.45,0.41,0.08,0.41,0.64,0.4,0.3,0.12,0.44,0.33,0.38,0.32,0.52,0.86,0.73,0.39,0.37,0.21,0.49,0.56,0.46,0.17,0.39,0.57,0.49,0.55,0.24,0.2,0.16,-0.06,0.43,0.04,0.29,-0.06,0.01,-0.32,-0.18,-0.04,-0.13,-0.08,-0.33,-0.27,0.01,-0.06,0.08,0.16,-0.2,0.0,0.01,-0.55,-0.35,-0.16,-0.07,0.15,-0.02,-0.07,0.21,-0.04,0.18,0.03,-0.22,-0.16,-0.25,-0.06,-0.44,0.01,-0.23,-0.14,-0.14,-0.3,-0.16,-0.49,-0.42,-0.19,-0.28,-0.18,-0.42,-0.28,-0.29,-0.18,-0.38,-0.53,-0.54,-0.33,-0.45,-0.5,-0.41,-0.25,-0.08,-0.46,-0.62,-0.14,-0.13,-0.25,-0.5,-0.56,-0.18,-0.25,-0.5,-0.07,-0.17,-0.23,-0.51,-0.12,-0.01,-0.22,-0.08,-0.51,-0.21,0.23,0.08,-0.19,-0.04,0.29,-0.08,-0.02,0.52,0.06,-0.04,0.05,0.09,0.11,-0.07,0.17,0.32,0.34,0.52,0.04,0.04,0.09,0.12,-0.13,0.14,0.13,0.0,0.09,-0.07,-0.01,-0.08,-0.06,0.16,0.06,-0.03,-0.28,0.03,0.04,0.37,0.06,0.39,0.32,0.41,0.11,0.07,0.01,0.15,0.14,-0.06,0.33,0.0,0.12,-0.14,0.19,-0.12,0.08,-0.03,-0.12,0.02,0.23,0.01,0.15,0.23,-0.21,-0.03,-0.26,0.43,-0.05,-0.28,-0.03,-0.03,-0.21,0.09,-0.38,-0.15,0.02,-0.07,0.4,-0.03,-0.14,0.08,0.05,0.1,-0.16,0.26,0.06,-0.03,0.0,0.1,-0.08,0.15,-0.07,0.06,0.14,0.01,0.01,0.16,-0.06,0.06,-0.1,0.06,-0.01,-0.13,-0.14,0.02,0.36,0.32,0.13,0.18,-0.07,0.12,-0.18,0.23,-0.01,0.04,0.1,-0.22,-0.2,-0.11,-0.22,-0.08,-0.17,-0.44,-0.26,-0.58,-0.51,-0.33,-0.08,-0.32,-0.37,-0.34,-0.54,-0.45,-0.66,-0.55,-0.46,-0.39,-0.32,-0.6,-0.36,-0.32,-0.59,-0.03,-0.17,-0.22,-0.1,0.12,-0.05,0.06,-0.09,-0.25,-0.1,-0.02,-0.19,0.04,0.15,-0.03,0.32,0.1,0.16,0.19,-0.05,0.31,0.65,0.76,0.17,0.23,0.03,0.4,0.06,0.37,0.14,0.08,0.25,0.22,0.21,0.27,-0.06,-0.27,-0.14,-0.16,-0.02,0.04,0.09,-0.18,0.26,-0.23,-0.15,-0.21,-0.19,0.09,0.19,0.32,-0.24,-0.12,-0.23,0.16,0.09,0.14,0.33,0.3,0.19,0.12,0.22,0.52,0.53,0.47,0.5,0.3,0.14,0.06,-0.03,0.25,0.36,-0.04,0.02,-0.1,0.0,0.1,0.48,0.43,-0.04,0.37,0.31,0.21,0.13,-0.04,0.08,-0.19,0.01,0.29,0.12,0.14,0.13,-0.21,-0.23,-0.24,0.04,-0.03,0.28,-0.06,-0.12,0.02,-0.03,-0.19,0.42,-0.21,-0.16,-0.1,-0.26,-0.43,-0.21,-0.31,-0.03,-0.26,-0.16,-0.08,-0.14,-0.01,-0.15,0.32,-0.37,-0.28,0.02,-0.32,-0.27,-0.08,-0.16,-0.11,0.02,-0.17,-0.12,0.0,-0.03,0.35,-0.02,-0.2,0.47,0.06,-0.05,-0.1,-0.23,-0.13,-0.01,-0.29,-0.29,-0.22,-0.24,-0.16,-0.24,-0.12,-0.39,-0.36,-0.38,-0.31,-0.25,-0.38,-0.14,-0.42,-0.15,-0.39,-0.19,-0.36,-0.09,0.05,-0.41,-0.12,-0.27,-0.37,0.01,-0.32,-0.07,-0.21,0.06,-0.07,-0.24,0.06,-0.15,-0.15,-0.33,0.03,-0.25,-0.27,-0.3,-0.31,0.06,-0.26,-0.53,-0.32,-0.3,-0.31,-0.28,-0.4,-0.31,-0.09,-0.17,-0.27,-0.22,-0.38,-0.3,-0.17,0.04,-0.35,-0.19,0.08,-0.22,-0.22,-0.48,0.06,-0.32,-0.57,-0.43,-0.51,-0.37,-0.12,-0.13,-0.36,-0.49,-0.31,-0.53,-0.39,-0.2,0.03,-0.1,-0.39,0.1,0.14,-0.18,-0.22,-0.15,-0.03,-0.18,0.2,0.41,0.11,0.3,-0.13,-0.15,0.0,-0.11,-0.23,-0.28,0.22,0.23,0.36,-0.27,-0.19,0.1,-0.18,-0.25,-0.08,-0.18,-0.22,0.21,-0.05,-0.27,-0.13,-0.03,0.06,-0.23,0.04,0.08,-0.04,0.03,-0.05,-0.14,-0.23,0.17,-0.04,0.45,0.19,-0.16,0.09,-0.08,0.03,0.32,0.31,0.04,0.06,0.16,0.19,0.04,-0.23,-0.11,-0.03,0.12,-0.1,-0.27,-0.16,-0.21,-0.09,0.17,-0.04,0.04,0.13,0.12,-0.29,-0.07,-0.15,0.41,-0.05,-0.2,0.01,-0.04,0.12,-0.06,-0.18,0.06,0.17,-0.22,-0.08,-0.14,-0.14,0.24,0.09,-0.24,-0.13,0.37,0.16,0.14,-0.02,-0.04,0.12,0.09,-0.25,-0.06,-0.03,0.12,-0.13,-0.08,-0.29,-0.05,0.12,-0.2,-0.2,-0.26,-0.14,0.08,-0.12,0.09,0.05,-0.09,-0.16,-0.01,-0.05,-0.33,-0.21,0.26,-0.07,-0.12,-0.11,-0.1,-0.25,-0.11,0.01,0.26,-0.24,-0.14,-0.14,-0.01,-0.12,-0.16,0.02,-0.03,-0.07,-0.17,0.25,0.02,0.45,0.19,-0.14,-0.03,0.33,0.43,0.3,0.09,0.15,0.59,0.21,0.12,0.09,0.09,0.12,0.27,0.1,0.38,0.11,0.41,0.0,-0.11,0.21,0.24,0.08,-0.15,-0.01,0.18,0.1,-0.12,-0.27,0.18,-0.09,-0.13,0.06,-0.11,-0.13,0.06,-0.15,-0.18,0.08,0.06,0.15,0.22,0.03,-0.12,0.2,0.3,0.07,0.39,0.22,0.38,0.15,0.18,0.71,0.27,-0.2,0.08,0.16,0.35,0.36,0.19,0.15,0.21,0.04,-0.23,0.04,-0.14,0.04,0.0,0.31,-0.18,0.12,0.01,0.12,0.34,0.09,-0.12,0.0,0.06,0.06,0.29,0.02,-0.09,-0.07,0.06,0.22,0.17,0.24,0.26,0.4,0.06,0.0,0.3,0.18,0.19,0.18,0.06,-0.1,0.28,-0.15,-0.13,0.07,0.14,0.06,-0.17,-0.11,-0.34,-0.07,0.03,-0.12,-0.19,-0.32,-0.24,-0.37,0.21,-0.2,-0.24,-0.24,-0.14,-0.12,0.23,-0.08,-0.3,-0.31,0.06,-0.29,-0.15,-0.41,-0.4,0.06,-0.11,-0.34,-0.28,-0.48,-0.31,-0.24,0.08,-0.23,-0.11,-0.1,0.2,-0.2,-0.15,-0.35,-0.19,0.04,-0.12,-0.07,-0.27,-0.34,-0.14,-0.07,-0.26,0.25,-0.42,-0.14,-0.15,-0.31,-0.22,-0.53,-0.3,-0.19,-0.05,-0.26,-0.22,-0.39,-0.39,0.18,-0.06,0.13,-0.35,-0.28,0.13,-0.2,-0.01,-0.22,-0.17,-0.13,-0.22,-0.29,-0.14,-0.21,-0.09,-0.25,-0.19,-0.42,-0.18,0.04,-0.16,0.19,-0.45,-0.38,-0.09,-0.33,-0.09,-0.01,0.05,-0.13,-0.39,-0.06,0.18,-0.09,-0.39,-0.32,-0.4,-0.2,-0.23,-0.37,-0.16,-0.41,-0.37,-0.1,-0.52,0.12,-0.19,-0.3,-0.1,-0.12,-0.42,-0.34,-0.27,-0.25,-0.12,-0.19,-0.3,-0.16,-0.49,-0.16,-0.3,-0.37,0.19,0.04,-0.2,0.14,-0.19,-0.36,-0.01,0.27,0.12,-0.13,-0.08,0.19,0.1,0.29,0.06,0.35,-0.03,0.39,0.1,0.08,0.12,-0.13,-0.04,-0.04,-0.07,0.09,-0.22,-0.13,-0.24,-0.02,-0.09,-0.31,-0.28,-0.03,-0.08,-0.2,-0.03,-0.34,-0.13,-0.36,-0.04,-0.5,-0.22,-0.1,-0.18,-0.25,-0.29,-0.16,-0.22,-0.5,-0.33,-0.35,-0.02,-0.2,-0.36,-0.25,-0.19,-0.07,-0.15,-0.14,-0.16,-0.04,-0.03,0.16,-0.27,0.02,-0.27,-0.38,0.45,-0.19,0.03,0.03,0.01,0.1,0.09,0.24,0.34,0.64,0.01,0.02,0.28,-0.12,-0.26,-0.19,0.2,-0.28,0.21,-0.15,-0.26,-0.65,-0.39,-0.21,-0.48,-0.45,-0.25,-0.5,-0.44,-0.51,-0.51,-0.08,-0.4,-0.28,-0.42,-0.31,-0.37,-0.5,-0.42,0.02,-0.71,-0.48,-0.42,-0.33,-0.39,-0.35,-0.38,0.1,-0.18,0.09,-0.22,-0.31,-0.56,-0.38,-0.2,-0.42,-0.41,0.14,-0.3,-0.31,-0.3,-0.25,-0.23,-0.07,-0.06,-0.29,-0.19,-0.22,0.01,0.11,-0.05,0.51,0.05,-0.06,0.2,0.41,-0.11,-0.34,-0.32,-0.17,-0.22,0.06,-0.05,-0.38,-0.16,0.12,-0.03,-0.2,0.02,-0.04,-0.14,-0.42,-0.17,-0.4,-0.38,-0.13,-0.03,-0.37,-0.56,-0.38,-0.21,-0.3,-0.17,-0.25,-0.57,-0.17,-0.54,-0.47,-0.61,-0.52,-0.5,-0.47,-0.52,-0.53,-0.36,-0.35,-0.44,-0.47,-0.14,-0.43,-0.32,-0.41,-0.42,-0.17,-0.25,-0.14,-0.23,-0.27,-0.4,-0.19,-0.06,-0.17,-0.28,-0.31,-0.22,-0.59,-0.48,-0.08,-0.31,-0.15,-0.27,-0.46,0.15,-0.18,-0.2,-0.29,-0.12,-0.09,0.12,-0.13,0.23,-0.21,-0.34,-0.23,-0.17,-0.19,-0.13,-0.07,-0.13,-0.4,-0.51,-0.3,-0.38,-0.32,-0.18,-0.24,-0.2,-0.44,-0.08,-0.44,-0.02,-0.17,-0.21,-0.17,-0.66,-0.34,-0.55,-0.44,-0.7,-0.41,-0.51,-0.43,-0.19,-0.3,-0.29,-0.24,-0.19,-0.35,-0.34,-0.12,0.1,-0.3,-0.32,-0.13,-0.38,-0.18,-0.11,-0.24,-0.33,-0.1,-0.09,-0.21,-0.31,-0.36,-0.29,0.0,-0.17,0.0,0.25,-0.42,-0.21,-0.08,-0.05,-0.23,0.01,-0.06,-0.18,-0.29,-0.02,0.04,-0.03,-0.13,-0.1,0.01,0.04,0.19,0.1,0.02,-0.02,0.36,0.32,0.09,0.06,0.19,-0.04,0.13,0.03,0.11,0.04,0.13,-0.2,-0.07,0.06,0.13,-0.09,-0.14,-0.18,-0.03,-0.05,-0.11,-0.08,-0.2,-0.15,-0.03,-0.29,0.04,-0.31,0.0,0.23,0.12,0.09,-0.16,0.0,0.33,0.31,0.47,0.57,0.24,0.16,0.19,0.01,0.16,0.25,0.29,0.44,0.31,0.35,0.35,0.44,-0.01,0.03,0.56,0.15,0.3,-0.13,-0.18,0.11,0.25,-0.11,-0.21,0.07,0.0,0.04,-0.01,0.1,0.31,-0.01,0.18,0.1,-0.06,0.2,-0.14,-0.07,-0.11,-0.09,-0.21,0.03,-0.14,-0.23,-0.01,-0.18,0.0,-0.29,0.08,-0.33,-0.16,-0.53,-0.13,-0.14,-0.1,0.17,-0.43,-0.39,-0.14,-0.18,0.08,-0.33,-0.55,-0.07,-0.07,0.04,0.09,-0.02,-0.13,0.1,0.03,0.33,0.27,0.25,0.04,-0.16,-0.04,0.11,0.08,-0.05,0.05,0.01,0.25,-0.16,-0.03,0.1,-0.07,-0.11,0.02,0.11,0.03,0.27,-0.29,0.14,-0.03,-0.14,0.24,-0.1,-0.08,-0.3,-0.36,-0.44,0.0,-0.38,-0.25,-0.25,-0.35,-0.41,0.0,0.26,-0.25,0.14,-0.29,-0.03,0.04,-0.03,-0.12,0.15,-0.36,0.19,0.12,-0.29,-0.17,-0.28,-0.31,-0.13,0.17,-0.33,-0.08,-0.09,0.31,-0.32,-0.25,-0.46,-0.17,-0.29,-0.25,-0.04,-0.03,-0.12,-0.17,-0.22,-0.22,-0.14,0.04,-0.1,-0.19,-0.39,-0.13,0.02,-0.07,0.06,0.04,0.14,-0.1,-0.11,-0.1,0.08,-0.02,-0.09,0.0,-0.11,-0.03,0.07,-0.26,-0.29,-0.04,-0.03,-0.1,-0.29,0.22,0.1,0.02,-0.27,-0.23,0.04,0.25,-0.14,0.28,-0.15,-0.08,-0.3,-0.35,0.32,-0.29,-0.36,-0.15,-0.03,0.19,-0.12,-0.31,0.12,-0.11,-0.04,0.07,-0.36,-0.52,-0.31,-0.52,-0.28,-0.29,-0.41,0.07,-0.41,-0.07,-0.08,-0.21,-0.18,0.06,-0.26,-0.06,0.06,-0.27,-0.39,-0.05,-0.19,-0.15,-0.21,0.12,-0.45,0.14,-0.38,-0.33,-0.14,-0.1,-0.29,-0.13,-0.55,-0.33,-0.26,-0.36,-0.32,-0.21,-0.06,-0.21,-0.19,-0.39,-0.43,-0.33,-0.24,0.16,-0.2,0.04,0.15,0.19,-0.16,0.35,0.31,0.27,0.31,0.36,0.06,0.14,-0.06,0.12,-0.01,0.47,0.08,0.1,-0.06,0.18,-0.03,0.13,0.2,-0.09,-0.05,-0.14,-0.03,0.41,0.13,-0.02,-0.07,0.08,0.04,0.08,0.27,0.24,-0.02,0.25,0.17,-0.12,0.05,0.25,0.13,0.25,0.17,-0.09,-0.07,0.37,0.24,0.36,0.04,0.62,0.07,0.13,0.38,0.52,0.1,-0.14,0.34,0.23,-0.1,0.26,0.0,0.01,-0.24,0.11,0.13,-0.03,-0.07,0.06,-0.03,0.4,-0.15,0.24,0.03,-0.11,-0.07,-0.08,-0.04,-0.26,-0.34,-0.01,-0.23,-0.06,-0.14,-0.41,-0.05,-0.34,-0.4,-0.06,-0.42,-0.35,-0.14,-0.52,-0.29,-0.26,0.01,-0.17,-0.32,-0.19,-0.09,-0.31,-0.42,-0.24,0.26,0.24,0.13,0.0,-0.11,0.03,0.3,0.13,0.12,0.31,0.29,0.31,-0.15,0.22,0.3,0.23,0.26,0.06,0.1,0.24,-0.01,0.15,-0.29,-0.18,-0.16,-0.02,-0.06,0.12,-0.22,-0.16,0.07,0.18,0.21,0.0,-0.01,-0.25,0.01,-0.13,-0.26,0.1,0.1,-0.09,0.09,-0.1,-0.21,-0.26,0.14,-0.09,-0.2,0.1,-0.27,-0.25,0.14,0.09,-0.17,0.16,0.09,-0.17,-0.07,0.3,-0.11,-0.14,-0.24,-0.03,0.0,-0.2,-0.16,0.32,-0.33,-0.06,0.06,-0.3,-0.09,-0.39,-0.1,0.03,0.04,-0.07,-0.01,-0.15,0.17,0.32,-0.23,0.14,0.01,0.41,0.1,0.05,0.23,-0.05,0.11,0.04,0.51,-0.01,0.16,0.22,0.05,-0.05,-0.03,0.24,0.13,0.46,0.67,0.14,0.26,0.06,0.13,0.01,-0.08,0.09,0.12,0.04,0.09,0.17,0.03,0.23,0.0,0.39,0.01,0.13,0.14,0.14,0.29,0.12,-0.15,0.32,-0.27,0.08,-0.03,0.0,0.4,0.02,-0.06,0.3,-0.17,-0.18,0.17,0.3,0.06,-0.15,0.06,0.06,-0.11,-0.04,0.18,-0.05,0.06,0.02,0.2,-0.29,0.11,0.2,0.33,0.44,0.41,0.32,0.34,0.41,0.48,0.37,0.75,0.82,0.5,0.78,0.65,0.8,0.52,0.42,0.27,0.45,0.54,0.37,0.36,0.46,0.34,0.71,0.34,0.37,0.64,0.46,0.36,0.29,0.37,0.64,-0.14,0.06,0.19,0.37,0.27,0.35,0.44,0.33,0.12,0.2,0.03,0.0,0.46,0.01,0.31,0.14,0.02,0.55,0.17,0.2,0.21,0.27,0.5,0.16,-0.03,-0.03,0.02,0.36,-0.09,-0.07,0.18,0.06,-0.1,0.02,0.09,0.18,-0.01,-0.01,0.21,0.22,0.2,0.2,0.31,0.25,0.04,0.31,0.09,0.19,0.02,0.07,0.1,0.21,0.1,-0.03,0.15,0.31,0.21,0.25,-0.07,0.11,0.16,0.19,-0.08,0.32,0.18,0.19,0.68,0.25,0.14,0.19,-0.07,0.02,0.37,0.24,0.22,0.21,0.03,0.13,0.23,0.19,-0.16,0.48,-0.01,-0.07,0.45,-0.01,0.2,-0.15,0.2,0.07,-0.1,-0.14,-0.14,-0.16,-0.37,-0.33,-0.17,-0.2,0.21,-0.23,-0.24,-0.12,-0.18,-0.06,-0.15,-0.18,0.08,-0.17,-0.1,-0.19,0.04,0.07,-0.03,-0.18,-0.26,0.06,-0.22,0.17,0.0,0.21,-0.07,0.04,0.28,0.09,0.16,0.15,0.25,-0.05,-0.14,0.06,0.26,0.24,0.14,0.27,0.66,0.08,0.23,0.4,0.29,0.06,-0.05,0.15,0.26,0.2,-0.04,0.2,-0.07,0.08,-0.01,-0.03,0.44,-0.1,-0.16,-0.29,-0.21,-0.23,-0.29,0.05,-0.19,-0.09,-0.36,0.0,-0.07,-0.03,-0.13,0.04,-0.16,0.34,-0.04,-0.32,-0.24,0.06,0.14,-0.24,0.14,0.36,0.2,0.08,0.0,-0.13,0.02,0.0,0.32,0.31,-0.19,-0.1,0.15,0.18,0.16,0.05,-0.02,0.16,0.33,0.17,0.15,0.4,0.26,0.11,0.34,0.34,0.57,0.28,0.23,0.41,0.2,0.05,0.11,0.06,0.0,-0.01,0.17,0.14,-0.15,0.08,0.06,0.15,0.24,0.14,-0.07,0.13,0.09,0.05,0.09,0.14,0.05,0.33,0.1,0.6,0.04,0.41,0.0,-0.12,0.1,0.0,0.13,0.0,-0.05,-0.14,-0.28,-0.28,-0.18,-0.51,-0.2,-0.2,-0.31,-0.3,-0.46,-0.16,-0.28,-0.14,-0.16,0.09,-0.21,-0.18,0.1,-0.14,-0.08,-0.21,0.1,0.03,-0.04,0.13,-0.3,0.19,-0.05,-0.15,0.23,0.08,-0.05,-0.07,-0.02,0.09,-0.07,0.11,-0.13,-0.04,-0.12,-0.1,-0.01,0.02,-0.19,0.11,-0.2,0.27,-0.21,0.08,-0.12,-0.3,-0.3,-0.09,0.01,-0.38,-0.27,-0.17,-0.29,-0.24,-0.01,-0.25,0.03,-0.29,-0.09,-0.04,0.2,-0.28,-0.05,-0.17,0.0,-0.15,-0.04,0.18,0.36,0.01,-0.17,0.04,0.35,0.08,0.15,-0.08,0.02,0.07,0.23,-0.19,0.03,-0.06,0.2,0.04,-0.09,-0.13,-0.2,0.16,0.04,-0.03,-0.09,-0.06,-0.32,0.21,-0.22,-0.04,0.21,-0.11,-0.3,-0.17,-0.24,-0.14,0.02,0.11,0.08,0.15,0.32,0.1,-0.34,0.01,0.19,-0.17,0.02,-0.3,-0.2,-0.02,-0.12,-0.18,0.02,-0.31,-0.42,-0.16,-0.28,-0.34,-0.04,-0.03,0.06,-0.22,0.14,0.04,-0.07,-0.01,-0.17,-0.19,0.01,0.09,0.22,0.11,-0.14,-0.09,-0.1,-0.04,0.06,0.0,-0.09,-0.26,0.0,0.01,0.04,-0.06,-0.01,-0.03,-0.03,0.26,0.08,-0.21,-0.03,-0.12,-0.21,0.13,-0.07,0.16,0.18,-0.02,0.03,-0.04,0.11,-0.05,0.02,-0.06,-0.02,-0.08,-0.1,0.21,0.02,-0.13,-0.04,-0.21,-0.07,-0.07,-0.04,-0.22,-0.13,-0.21,-0.29,-0.31,-0.35,-0.32,-0.44,-0.16,-0.1,0.16,-0.23,0.0,-0.04,-0.01,-0.2,-0.26,-0.09,-0.11,-0.08,-0.18,-0.09,0.17,-0.04,-0.34,-0.1,-0.25,-0.14,0.08,0.46,0.25,0.1,0.33,0.18,0.14,-0.2,0.23,0.12,0.35,0.09,0.0,0.17,0.27,-0.1,-0.18,-0.03,0.24,-0.21,-0.12,0.1,0.14,-0.33,-0.1,-0.04,-0.21,0.1,-0.2,0.03,-0.11,-0.03,0.05,0.0,-0.2,-0.26,-0.21,-0.2,-0.18,-0.09,-0.24,-0.2,0.03,-0.19,0.09,0.15,-0.11,0.36,0.17,0.13,0.11,-0.03,0.12,0.23,0.16,0.14,0.14,0.02,0.19,0.04,-0.09,0.24,0.09,-0.15,0.24,0.17,-0.3,0.28,-0.46,-0.2,0.09,0.06,0.11,-0.15,0.2,0.08,0.15,0.08,0.04,-0.12,0.28,0.13,0.01,-0.2,-0.2,-0.11,0.2,-0.22,-0.07,0.2,-0.35,-0.32,-0.14,-0.03,0.04,-0.06,-0.1,-0.23,-0.31,0.12,0.09,-0.22,-0.24,-0.48,-0.09,-0.48,-0.26,-0.3,-0.17,-0.23,-0.12,-0.01,0.06,0.08,0.01,0.34,0.25,0.17,-0.02,0.06,0.04,-0.02,-0.29,0.02,-0.08,0.0,-0.19,0.21,0.15,0.38,0.37,0.08,0.11,0.05,0.38,0.29,0.18,0.02,0.01,0.04,0.15,0.09,0.39,0.11,0.01,-0.09,-0.12,0.25,-0.24,-0.05,-0.24,-0.07,0.1,0.03,-0.17,-0.02,0.37,0.25,0.37,-0.29,0.17,-0.21,0.02,0.29,0.13,0.21,-0.44,-0.3,0.04,-0.38,-0.15,-0.09,-0.28,0.12,-0.09,-0.27,-0.17,-0.09,-0.3,-0.16,-0.24,-0.1,-0.04,-0.28,-0.19,-0.2,-0.08,-0.21,-0.42,-0.39,0.32,-0.1,0.02,-0.36,-0.11,-0.31,0.19,-0.19,-0.07,0.01,0.06,-0.03,-0.22,0.18,0.25,0.06,-0.09,0.04,0.1,-0.24,0.08,-0.18,-0.26,-0.13,-0.2,-0.21,-0.15,-0.2,0.01,-0.41,-0.05,-0.24,-0.13,-0.36,-0.12,-0.19,-0.1,0.07,-0.41,-0.23,-0.11,-0.2,-0.25,-0.15,0.23,-0.32,-0.39,-0.19,-0.11,-0.25,-0.08,-0.23,-0.26,-0.21,-0.29,-0.33,-0.09,-0.2,0.0,-0.14,-0.25,0.12,-0.24,-0.35,-0.21,-0.01,-0.04,0.44,-0.1,0.11,-0.05,0.2,-0.04,0.42,-0.09,-0.17,-0.06,0.42,-0.11,-0.14,0.0,0.3,-0.03,0.15,0.0,0.1,0.12,0.17,0.08,0.19,-0.2,0.44,0.33,0.06,0.1,0.02,0.03,-0.03,-0.07,0.0,0.21,-0.05,0.06,-0.08,0.04,-0.01,0.27,0.1,0.17,-0.17,-0.22,0.04,0.12,-0.17,-0.23,-0.29,-0.17,-0.06,0.24,-0.17,-0.35,0.08,-0.11,0.18,0.27,-0.25,0.25,-0.03,-0.15,-0.05,-0.03,0.02,-0.08,-0.08,0.14,0.04,0.08,0.05,-0.13,-0.3,0.04,0.38,0.26,0.08,0.01,0.41,0.01,-0.06,0.09,0.15,0.44,0.17,0.52,0.37,-0.08,0.43,0.28,-0.09,-0.11,-0.2,0.23,0.27,0.13,0.3,-0.16,0.33,0.4,-0.14,-0.17,-0.31,0.08,-0.18,0.22,0.04,0.02,0.23,-0.06,0.03,0.0,-0.01,0.1,0.09,-0.04,-0.16,0.24,0.15,0.2,0.01,0.0,0.0,-0.04,-0.07,0.17,0.03,-0.19,0.1,0.14,0.15,-0.06,-0.19,-0.14,-0.05,-0.05,-0.08,-0.07,-0.19,0.18,0.07,0.06,-0.03,-0.17,0.04,-0.07,-0.1,0.1,-0.04,0.29,-0.12,-0.04,0.1,-0.01,0.38,0.12,0.26,0.27,0.12,0.42,0.24,-0.07,0.06,-0.05,-0.01,-0.04,0.4,-0.07,0.14,0.1,-0.1,0.26,-0.17,0.13,0.29,0.35,0.23,0.08,-0.02,0.11,-0.01,0.11,0.17,-0.1,0.24,0.04,-0.02,-0.15,-0.44,0.19,-0.15,0.03,-0.09,0.04,0.08,0.23,-0.11,0.0,0.23,0.09,-0.08,-0.06,0.03,-0.27,0.08,0.08,0.06,-0.12,0.2,-0.13,-0.2,0.07,0.34,0.32,-0.12,0.42,0.11,-0.29,0.02,0.06,0.04,0.06,0.11,0.22,-0.04,0.14,0.14,0.32,0.1,0.21,0.17,0.01,0.0,-0.14,0.04,0.35,-0.09,-0.21,-0.08,-0.1,0.18,0.0,-0.1,0.13,-0.2,0.04,-0.1,0.26,0.27,0.08,-0.06,0.18,0.0,0.02,0.29,-0.13,0.05,0.2,0.28,-0.32,0.15,-0.03,0.02,0.1,-0.19,0.07,-0.05,-0.31,-0.1,0.01,-0.06,-0.24,-0.05,-0.14,0.05,-0.02,0.11,0.04,0.19,0.01,0.01,-0.17,-0.25,-0.04,-0.16,-0.14,0.05,0.02,0.03,0.01,-0.04,-0.06,0.03,-0.11,-0.07,-0.24,-0.11,0.07,-0.03,-0.17,0.15,-0.21,0.24,-0.21,-0.04,-0.09,-0.26,-0.24,-0.18,-0.39,-0.37,0.01,-0.24,-0.1,-0.5,0.08,0.09,-0.29,-0.18,-0.22,0.03,-0.26,-0.06,0.01,-0.25,-0.19,-0.15,0.01,-0.16,-0.02,-0.03,0.02,-0.04,0.12,0.16,-0.01,0.17,-0.04,-0.19,-0.06,0.0,0.18,-0.03,-0.02,-0.11,0.14,0.04,-0.04,0.0,-0.05,0.07,0.04,0.11,0.0,-0.03,-0.04,0.24,0.16,0.29,0.35,0.28,0.79,0.21,-0.04,0.06,0.4,0.1,0.07,-0.03,0.17,0.22,-0.24,0.21,-0.25,0.06,-0.16,-0.17,-0.12,-0.06,0.23,0.04,0.24,0.46,-0.09,0.16,-0.36,0.1,0.07,-0.11,-0.3,-0.28,-0.07,-0.34,-0.3,-0.18,-0.27,-0.14,-0.24,-0.2,-0.12,-0.23,-0.04,-0.13,-0.03,-0.22,0.09,0.02,0.04,0.06,-0.19,0.12,0.04,-0.01,0.04,-0.03,0.09,0.06,0.14,0.27,-0.03,0.34,-0.09,-0.09,0.1,0.09,0.04,0.03,0.18,-0.09,0.09,-0.16,-0.04,0.01,0.06,-0.03,0.04,0.28,0.19,0.04,-0.04,0.45,0.37,0.12,0.21,-0.09,0.37,0.45,0.04,0.62,0.44,0.46,0.36,0.3,0.35,0.48,0.33,0.18,0.2,0.62,0.57,0.21,0.04,0.23,0.33,-0.09,0.3,0.0,0.02,0.11,-0.05,0.0,0.33,0.31,0.12,-0.23,0.13,-0.1,-0.29,-0.07,-0.13,-0.37,-0.22,0.1,-0.18,-0.02,0.1,-0.05,0.01,0.12,-0.01,0.08,-0.03,0.03,-0.12,-0.06,0.24,0.48,-0.03,-0.05,0.31,0.04,0.17,-0.12,0.02,-0.2,-0.01,-0.12,0.0,0.12,-0.21,0.2,0.15,-0.17,-0.22,-0.32,-0.16,-0.15,-0.14,-0.09,-0.18,-0.08,-0.2,-0.14,-0.16,-0.15,0.56,-0.23,-0.3,-0.24,-0.27,-0.11,0.27,-0.29,-0.17,-0.05,-0.1,0.04,-0.27,-0.42,-0.2,-0.2,-0.33,-0.39,-0.09,-0.36,-0.38,-0.34,-0.2,-0.08,-0.33,0.12,-0.39,-0.39,-0.21,-0.43,-0.31,-0.16,-0.07,-0.3,-0.17,-0.35,-0.4,-0.57,-0.34,-0.44,-0.26,0.06,-0.18,-0.25,-0.43,0.06,0.12,-0.1,-0.25,-0.12,-0.21,-0.36,-0.27,-0.19,-0.19,0.14,-0.02,-0.11,-0.37,-0.09,0.24,-0.3,-0.3,-0.31,-0.2,-0.37,-0.4,-0.32,-0.48,-0.25,-0.22,-0.27,-0.18,-0.3,-0.32,-0.32,-0.39,-0.2,-0.45,-0.25,-0.27,-0.14,-0.18,-0.23,-0.36,-0.14,0.03,0.1,0.11,-0.06,0.24,0.09,-0.03,-0.13,-0.04,-0.26,-0.18,-0.29,0.31,-0.16,-0.23,0.2,-0.1,0.29,-0.16,0.13,0.17,-0.01,-0.3,-0.15,0.1,-0.09,-0.15,-0.33,-0.25,-0.35,-0.35,-0.1,-0.35,-0.46,-0.26,-0.07,-0.27,0.02,-0.23,0.02,-0.03,0.04,0.14,0.02,-0.38,0.04,-0.47,-0.27,-0.02,-0.15,-0.1,-0.25,-0.11,0.24,-0.14,-0.35,-0.13,-0.05,0.05,-0.05,0.19,-0.31,-0.25,-0.19,-0.38,0.21,-0.13,-0.25,-0.01,0.02,-0.14,-0.17,-0.32,-0.29,0.06,-0.03,0.04,0.23,0.28,0.3,0.24,0.04,0.13,0.09,0.08,0.06,0.05,0.17,0.15,-0.28,-0.12,-0.03,-0.1,-0.14,-0.14,0.15,-0.29,-0.2,-0.18,0.12,-0.03,0.08,-0.05,-0.11,-0.14,-0.17,0.02,0.04,-0.07,0.14,0.03,0.07,0.16,0.28,0.01,-0.09,-0.08,-0.07,0.09,0.02,-0.09,-0.39,-0.24,-0.11,-0.13,0.19,0.09,-0.25,-0.21,-0.42,-0.26,-0.28,-0.27,-0.07,-0.13,0.02,0.06,-0.1,-0.11,-0.2,-0.12,-0.03,-0.33,0.26,-0.03,-0.14,-0.09,-0.05,0.01,-0.23,-0.02,-0.15,-0.23,-0.15,-0.21,-0.05,-0.03,0.07,-0.24,0.01,-0.01,-0.15,-0.07,0.01,-0.04,0.06,0.2,0.23,0.12,0.11,0.23,0.12,0.3,0.25,0.38,0.31,0.21,0.47,0.2,0.1,0.25,0.09,0.34,0.12,0.29,0.32,0.24,0.12,-0.07,0.02,0.28,0.23,-0.16,0.04,-0.07,-0.19,-0.21,-0.17,-0.1,-0.18,-0.12,-0.17,-0.18,-0.16,-0.46,-0.27,-0.34,-0.15,-0.04,-0.28,-0.13,-0.33,-0.2,0.27,-0.06,0.02,0.04,0.06,0.0,0.23,-0.05,0.0,-0.08,-0.02,0.0,0.28,-0.27,-0.23,-0.09,-0.01,-0.25,-0.16,0.01,-0.16,-0.05,-0.28,0.03,-0.09,-0.32,-0.09,-0.13,-0.28,-0.19,0.03,-0.21,-0.1,-0.14,0.1,-0.09,0.03,-0.14,-0.39,-0.28,-0.03,0.01,-0.3,-0.15,-0.41,0.0,-0.06,0.08,0.02,-0.29,-0.13,-0.27,-0.24,-0.1,-0.11,-0.16,-0.09,-0.44,-0.15,-0.43,-0.03,-0.1,-0.18,-0.24,-0.54,-0.21,-0.29,-0.2,-0.38,-0.38,-0.38,-0.05,-0.2,-0.17,-0.17,-0.11,-0.3,-0.12,-0.01,0.06,-0.04,-0.3,-0.12,-0.13,0.01,-0.06,-0.24,0.01,0.28,0.12,-0.09,0.21,0.06,-0.06,0.03,-0.09,0.04,0.06,-0.16,0.33,-0.2,0.37,0.15,-0.04,0.12,-0.13,-0.11,0.06,-0.03,-0.02,0.14,-0.08,0.04,-0.2,-0.11,0.36,0.28,0.4,-0.05,0.16,-0.3,0.22,0.01,-0.09,0.09,-0.11,-0.18,-0.07,-0.03,-0.2,0.02,0.36,-0.39,-0.15,0.01,0.01,-0.01,0.07,0.23,0.12,-0.25,0.0,0.12,0.08,0.09,-0.09,-0.21,-0.17,-0.22,0.39,-0.35,0.0,0.12,0.0,-0.14,-0.23,-0.23,-0.25,-0.13,-0.06,-0.08,0.26,-0.33,-0.26,-0.03,-0.12,-0.03,-0.18,-0.22,0.17,-0.33,-0.32,-0.12,-0.24,-0.19,-0.11,-0.27,0.06,-0.44,-0.01,-0.11,-0.21,-0.33,-0.06,-0.08,-0.17,-0.45,-0.08,-0.38,-0.33,-0.29,-0.25,-0.22,0.02,-0.19,-0.07,0.43,0.23,-0.23,0.17,0.17,-0.18,-0.05,0.04,0.06,0.08,0.34,0.06,0.15,-0.17,0.34,0.31,0.12,0.5,0.08,0.11,-0.04,-0.02,-0.01,-0.16,-0.13,0.0,-0.06,-0.15,0.05,-0.37,0.03,-0.06,-0.25,-0.21,-0.21,-0.29,-0.34,-0.36,-0.21,-0.19,-0.18,0.0,0.04,0.03,-0.4,-0.01,-0.11,-0.09,0.0,-0.11,-0.38,-0.24,-0.04,-0.18,-0.14,-0.01,0.22,-0.31,0.03,0.06,0.2,0.04,-0.08,-0.28,-0.32,-0.19,-0.25,0.06,-0.35,-0.33,-0.2,0.01,-0.18,-0.37,-0.1,-0.04,0.1,-0.21,0.03,-0.32,0.16,-0.24,-0.17,-0.13,-0.29,-0.19,0.26,0.04,-0.14,-0.19,0.01,-0.14,0.04,0.18,0.05,0.29,-0.06,-0.16,0.08,0.0,-0.11,-0.16,-0.23,0.09,-0.12,-0.28,-0.13,0.04,-0.07,0.08,-0.14,-0.09,-0.13,-0.24,0.21,-0.06,-0.01,0.17,-0.07,0.19,0.19,0.08,0.39,-0.24,0.0,0.02,0.17,-0.06,-0.05,0.06,0.15,-0.35,0.04,0.25,-0.05,-0.24,-0.18,-0.11,-0.26,-0.08,0.1,0.26,0.01,0.04,0.31,0.06,0.01,-0.03,0.25,0.09,0.15,0.2,0.37,0.15,0.24,0.16,0.26,0.07,0.17,-0.14,0.04,0.28,0.42,0.18,0.0,0.06,0.26,0.17,-0.03,0.32,0.15,-0.09,-0.08,-0.17,0.12,-0.14,0.18,-0.11,0.02,-0.09,-0.18,-0.16,-0.22,-0.13,-0.08,-0.36,-0.45,-0.15,-0.27,-0.31,0.06,-0.16,-0.25,-0.12,-0.26,0.03,-0.19,-0.09,-0.05,-0.22,-0.14,0.0,-0.17,-0.35,-0.3,-0.26,-0.37,-0.3,-0.35,-0.27,-0.33,-0.39,-0.27,0.0,-0.27,-0.18,-0.37,0.07,-0.46,-0.19,-0.09,-0.27,0.02,-0.17,-0.06,-0.24,-0.22,-0.27,-0.28,-0.14,-0.26,0.04,-0.26,-0.09,-0.16,-0.25,-0.07,-0.23,-0.14,-0.12,-0.14,0.05,-0.2,0.24,-0.44,-0.13,-0.11,0.18,-0.45,-0.07,0.09,-0.36,-0.09,-0.27,-0.25,-0.2,-0.36,-0.16,-0.16,-0.17,-0.32,-0.39,-0.27,-0.22,-0.19,-0.36,-0.29,-0.28,0.08,0.01,-0.03,-0.09,-0.07,0.23,0.16,0.42,0.54,0.04,0.14,0.05,0.46,0.08,0.34,0.0,-0.25,-0.13,-0.03,-0.06,-0.11,-0.15,-0.12,-0.19,0.07,0.26,0.06,0.19,-0.32,0.1,0.04,0.25,0.0,-0.12,-0.11,0.05,0.17,0.13,-0.07,0.27,-0.28,0.32,-0.09,0.2,-0.03,0.46,0.2,-0.03,0.34,-0.12,0.14,0.07,0.15,0.06,-0.01,0.21,-0.2,0.03,-0.12,-0.2,0.1,-0.17,0.04,0.01,0.17,0.13,0.26,-0.25,0.08,0.17,0.2,-0.01,0.28,0.1,0.19,-0.29,-0.05,-0.37,0.1,-0.19,0.15,0.0,0.15,0.31,-0.3,-0.21,0.08,-0.19,-0.01,-0.24,0.03,0.08,-0.12,-0.17,0.11,0.13,0.19,0.16,0.33,-0.09,0.06,0.13,-0.2,-0.07,0.14,0.2,0.42,-0.03,-0.04,-0.16,-0.15,-0.07,-0.23,-0.08,-0.09,-0.08,-0.22,-0.39,-0.27,-0.27,-0.28,-0.22,-0.35,0.02,-0.4,-0.15,-0.25,-0.14,-0.22,-0.16,-0.15,-0.01,0.06,-0.01,-0.1,0.0,-0.06,-0.02,0.12,-0.08,-0.03,-0.14,-0.05,-0.14,-0.27,0.01,-0.06,-0.19,-0.16,-0.31,0.16,0.04,-0.35,-0.19,0.01,-0.39,-0.3,-0.23,-0.14,-0.37,-0.36,0.09,-0.2,0.04,-0.17,0.16,-0.18,-0.07,0.16,0.2,-0.03,0.08,0.12,0.11,-0.14,-0.32,0.37,-0.28,-0.04,0.14,0.04,0.15,0.17,-0.04,0.05,-0.12,0.38,-0.24,-0.26,-0.03,-0.38,-0.17,0.2,-0.33,-0.63,-0.26,-0.5,-0.36,-0.39,-0.53,-0.55,-0.37,-0.35,-0.54,-0.26,-0.2,-0.21,-0.47,-0.19,-0.32,-0.27,-0.35,-0.42,0.06,-0.19,-0.15,-0.46,-0.36,-0.19,-0.38,-0.07,-0.09,-0.45,-0.35,-0.23,-0.05,-0.32,-0.04,-0.12,-0.08,-0.11,-0.31,0.0,0.24,-0.42,-0.04,-0.21,-0.04,-0.24,-0.43,-0.23,-0.29,0.04,-0.23,-0.07,-0.38,0.09,-0.03,0.13,-0.17,-0.05,-0.28,-0.28,-0.28,-0.55,-0.36,-0.31,0.16,-0.25,-0.56,-0.64,-0.35,-0.2,-0.35,-0.3,-0.13,-0.36,-0.28,-0.39,-0.36,-0.24,0.1,-0.3,0.14,-0.22,-0.01,0.11,0.14,0.49,0.25,0.27,-0.01,-0.1,0.24,-0.01,0.05,-0.16,0.11,0.06,0.06,0.08,-0.03,-0.13,-0.14,0.04,-0.16,-0.06,0.21,0.14,0.02,0.21,-0.2,-0.05,0.15,0.12,0.21,0.12,0.14,-0.1,0.04,0.07,0.03,0.08,0.04,0.08,0.08,-0.05,0.22,0.2,0.19,0.01,0.21,0.16,0.1,0.39,0.34,0.47,-0.01,0.03,0.31,0.01,0.06,0.01,0.47,0.06,0.45,-0.01,0.09,0.01,-0.07,0.24,0.28,-0.08,0.21,0.4,0.26,-0.15,0.38,0.29,0.2,0.55,0.52,0.41,0.54,0.21,0.19,0.09,-0.27,0.22,0.04,-0.15,-0.22,-0.14,-0.14,0.06,-0.04,-0.05,0.01,0.13,0.13,-0.01,0.33,0.0,-0.03,0.1,0.04,0.01,-0.1,-0.03,-0.03,0.01,-0.17,0.44,0.19,0.04,0.23,0.16,0.19,-0.03,0.4,0.16,0.41,-0.15,0.52,0.13,0.1,0.27,-0.06,0.2,0.41,0.11,0.09,0.17,-0.06,0.28,0.09,0.04,0.04,0.05,0.3,0.02,-0.04,-0.07,-0.09,-0.05,0.29,-0.17,-0.07,-0.33,0.22,-0.13,0.38,0.08,0.14,0.18,0.29,-0.06,0.01,0.23,0.3,0.0,0.04,0.0,-0.02,0.22,0.04,0.08,0.1,0.07,0.27,0.13,0.05,-0.2,0.14,0.17,-0.05,0.17,0.22,0.25,0.21,0.02,-0.12,-0.03,0.15,-0.01,0.13,0.02,-0.1,-0.02,0.08,0.5,0.24,-0.18,0.07,-0.08,0.19,-0.07,0.22,-0.03,0.0,0.23,0.02,-0.04,0.09,0.08,0.34,-0.14,0.01,-0.16,0.04,-0.28,-0.09,0.02,-0.08,-0.36,-0.18,0.12,-0.36,0.09,0.11,-0.17,0.21,-0.08,-0.34,0.15,0.49,-0.09,0.14,0.29,-0.02,0.36,-0.15,-0.19,0.07,0.19,0.14,0.04,0.0,-0.04,-0.26,-0.03,-0.23,0.27,-0.3,-0.27,-0.11,0.13,-0.1,-0.32,-0.09,-0.45,-0.03,-0.29,-0.36,-0.29,-0.41,-0.29,0.01,-0.36,-0.03,0.1,-0.04,0.38,-0.17,-0.23,0.14,0.32,0.12,0.06,0.27,0.03,-0.19,-0.26,0.0,-0.21,-0.02,-0.03,-0.03,-0.18,0.0,-0.31,0.15,-0.07,-0.13,-0.13,-0.09,0.15,-0.49,-0.57,-0.47,-0.16,-0.42,-0.49,-0.18,-0.49,-0.32,-0.5,-0.4,-0.35,-0.26,0.09,-0.48,-0.37,-0.36,-0.03,-0.38,-0.21,-0.19,-0.23,-0.08,-0.23,-0.2,0.02,-0.07,-0.16,0.37,0.03,-0.1,0.13,0.13,-0.03,0.02,-0.12,-0.2,0.0,0.48,0.0,-0.3,0.04,0.28,-0.06,0.27,0.19,-0.02,-0.12,0.06,0.04,0.5,-0.06,0.22,-0.01,0.29,0.19,0.26,0.49,-0.09,0.27,0.1,0.23,0.45,0.4,0.44,0.61,0.33,0.33,0.23,0.37,0.64,0.43,0.25,0.26,0.45,0.27,0.59,0.42,0.19,0.31,0.6,0.49,0.41,0.65,0.72,0.92,0.74,0.69,0.76,0.63,0.51,0.68,0.7,0.45,0.52,0.63,0.5,0.33,0.27,0.57,0.27,0.47,0.65,0.44,0.3,0.7,0.72,0.58,0.75,0.75,0.39,0.32,0.61,0.47,0.41,0.44,0.46,0.32,0.25,0.44,0.23,0.06,0.52,-0.03,0.23,-0.05,0.08,0.07,-0.06,-0.07,-0.08,0.05,0.12,0.27,0.17,0.28,0.18,0.42,0.24,0.19,0.1,0.28,0.31,0.08,0.05,0.29,0.39,0.68,0.5,0.22,0.46,0.53,0.21,0.38,0.61,0.23,0.51,0.24,0.38,0.58,0.25,0.18,0.46,0.26,0.51,0.16,0.2,0.78,0.28,0.69,0.6,0.53,0.77,0.49,0.5,0.26,0.38,0.62,0.31,0.37,0.36,0.48,0.4,0.52,0.37,0.23,0.45,0.66,0.53,0.27,0.31,0.14,0.6,0.47,0.31,0.33,0.0,0.13,0.45,0.22,0.5,0.12,0.0,0.1,0.59,0.3,0.14,0.4,0.03,0.23,0.17,-0.01,0.2,-0.17,0.1,0.42,0.29,0.33,0.12,0.1,0.49,0.24,0.23,0.36,0.17,0.48,-0.02,0.14,0.11,0.5,0.04,0.41,0.13,0.33,0.13,0.14,0.03,0.27,0.24,0.21,0.38,0.32,-0.09,-0.03,0.33,0.36,0.22,0.0,-0.04,-0.05,0.39,0.26,-0.12,-0.21,-0.03,0.19,0.29,0.22,-0.2,0.15,-0.03,0.0,0.06,0.35,-0.03,0.13,0.28,-0.09,0.1,-0.04,0.14,0.22,0.5,0.16,0.36,0.06,0.14,0.06,0.31,0.28,0.33,0.02,0.51,0.41,0.46,0.24,0.32,0.23,0.32,0.2,0.25,0.26,0.52,0.09,0.15,0.17,-0.04,-0.02,-0.14,0.2,0.05,0.07,0.0,-0.08,0.08,0.09,0.21,-0.09,0.54,-0.08,0.03,0.17,0.09,0.32,0.23,0.32,0.35,-0.13,-1.34,0.04,0.14,0.39,0.4,0.1,0.03,0.43,0.18,-0.05,0.48,0.14,0.45,0.41,0.24,0.16,0.27,0.03,-0.02,0.59,0.23,0.31,0.29,0.61,0.3,0.33,0.26,0.14,0.42,0.44,0.38,0.32,0.34,-0.17,0.19,0.46,0.34,0.32,0.36,0.63,0.24,0.29,0.53,0.31,0.17,0.52,0.26,0.09,0.42,0.18,0.14,0.32,-0.3,0.13,-0.03,0.1,0.1,0.2,-0.34,0.03,-0.15,-0.17,-0.13,-0.09,-0.39,-0.12,-0.13,-0.31,0.01,-0.13,-0.36,-0.13,-0.26,-0.18,-0.26,-0.35,-0.09,0.09,-0.23,-0.4,-0.42,-0.32,-0.08,-0.22,0.32,-0.03,0.15,-0.08,0.02,-0.03,0.27,-0.09,-0.03,0.02,-0.14,-0.07,-0.21,-0.28,0.12,-0.09,-0.02,-0.32,-0.28,-0.3,-0.07,-0.03,-0.32,-0.04,0.0,-0.18,0.0,-0.02,-0.05,-0.11,0.08,0.02,0.07,0.03,-0.22,-0.32,0.1,-0.26,-0.09,-0.36,0.13,-0.17,-0.24,-0.32,0.15,-0.03,-0.15,-0.2,-0.23,-0.12,0.2,-0.3,-0.08,-0.1,-0.24,-0.13,0.08,-0.29,-0.1,0.01,-0.22,-0.29,-0.11,-0.2,-0.14,-0.23,-0.17,-0.21,0.09,-0.08,0.21,-0.12,-0.15,0.45,0.19,-0.22,-0.02,0.11,-0.25,-0.13,0.26,0.1,-0.1,-0.02,0.03,0.1,0.1,-0.08,0.1,0.01,0.12,-0.03,-0.14,0.03,0.23,-0.11,-0.21,0.01,0.15,-0.01,0.21,0.2,0.08,0.23,0.16,0.08,0.04,0.41,0.04,0.14,0.34,0.06,0.01,0.13,0.28,-0.11,0.08,0.04,-0.1,-0.1,-0.28,0.03,-0.11,0.02,-0.05,0.03,0.09,-0.14,-0.26,-0.27,-0.21,-0.47,0.08,-0.11,-0.24,-0.14,-0.28,0.06,-0.29,-0.33,-0.11,-0.33,-0.17,-0.21,-0.01,-0.04,0.01,0.02,0.03,-0.39,-0.16,-0.27,-0.33,-0.04,-0.08,-0.17,-0.4,-0.24,-0.17,-0.27,-0.35,-0.28,-0.25,-0.18,-0.25,0.06,-0.14,0.06,0.13,-0.43,-0.28,-0.09,-0.5,-0.29,-0.18,-0.12,-0.3,-0.38,-0.06,-0.21,0.01,-0.27,-0.4,-0.24,0.13,0.09,0.2,0.01,-0.03,0.34,0.18,0.0,-0.17,-0.29,0.08,-0.22,0.0,-0.28,-0.09,0.06,-0.07,-0.03,-0.26,-0.26,-0.15,-0.2,-0.06,-0.05,-0.21,0.03,0.12,0.21,-0.07,-0.09,0.09,-0.24,-0.26,0.05,-0.07,0.02,0.22,-0.08,0.54,0.23,-0.06,-0.01,0.11,-0.05,0.22,0.23,0.04,0.09,0.19,0.17,-0.09,0.08,0.18,-0.33,-0.15,0.11,-0.07,-0.04,0.06,-0.13,-0.02,0.21,-0.04,-0.12,-0.03,0.17,-0.04,-0.15,-0.05,0.01,0.08,0.17,0.18,0.17,0.35,-0.05,-0.01,-0.08,-0.07,0.35,0.3,-0.09,0.14,-0.04,0.22,-0.01,0.03,0.04,0.09,0.43,0.28,-0.1,-0.05,0.13,-0.13,-0.05,-0.01,-0.01,0.18,-0.21,-0.25,-0.3,-0.03,-0.14,0.28,0.04,0.17,0.12,-0.04,-0.13,0.12,-0.15,-0.19,0.02,-0.29,-0.21,-0.28,-0.26,-0.19,-0.22,-0.21,-0.37,-0.3,0.01,-0.49,-0.47,-0.02,-0.19,-0.56,-0.49,-0.36,-0.25,-0.11,-0.46,-0.11,-0.18,-0.43,-0.55,-0.56,-0.51,-0.39,-0.14,-0.41,-0.38,-0.49,-0.28,-0.3,-0.24,-0.17,-0.55,-0.31,-0.24,-0.63,-0.15,-0.41,-0.38,-0.31,-0.44,-0.38,-0.41,-0.14,-0.28,0.1,-0.14,0.09,-0.09,-0.3,0.15,0.06,0.01,0.28,0.25,0.44,0.08,0.36,-0.06,0.35,0.14,0.09,0.0,0.06,0.06,-0.03,-0.1,0.16,-0.35,-0.02,-0.27,-0.29,0.14,-0.35,-0.37,-0.45,-0.04,-0.1,-0.28,-0.39,-0.33,-0.42,-0.51,-0.36,-0.55,-0.25,-0.23,-0.19,-0.3,-0.1,-0.05,-0.03,-0.06,-0.04,0.12,-0.07,-0.13,-0.09,0.1,-0.18,0.03,0.09,-0.03,-0.07,0.14,-0.06,0.14,-0.04,-0.18,-0.09,0.06,-0.25,-0.14,-0.1,0.36,-0.07,-0.24,0.37,-0.03,-0.26,0.09,-0.03,0.32,0.24,0.04,-0.02,0.36,0.28,0.44,0.59,0.21,0.64,0.48,0.61,0.35,0.7,0.6,0.48,0.25,0.48,0.29,0.81,0.65,0.65,0.83,0.52,0.63,0.84,0.89,1.07,1.25,1.13,0.91,0.94,0.96,0.67,1.27,0.82,0.84,0.78,0.86,1.03,0.79,0.89,0.68,0.71,0.6,0.74,0.58,0.77,0.92,0.53,0.66,0.65,0.48,0.62,0.58,0.9,0.81,0.42,0.5,0.44,0.53,0.49,0.54,0.41,0.43,0.7,0.76,0.39,0.11,0.12,0.42,0.37,0.22,0.28,0.46,0.24,0.16,0.21,-0.09,0.21,0.06,0.01,0.14,0.24,0.02,0.14,-0.13,-0.13,-0.12,0.05,0.11,0.04,0.05,-0.04,0.0,0.02,0.06,-0.14,-0.4,-0.03,-0.42,-0.23,-0.15,-0.14,-0.23,-0.5,-0.35,-0.12,-0.22,-0.06,-0.46,-0.03,-0.41,0.14,-0.21,-0.16,-0.21,-0.32,-0.15,0.04,0.09,0.1,0.18,-0.01,-0.26,0.14,0.34,-0.06,0.03,-0.09,-0.03,0.21,0.12,0.12,0.45,-0.02,0.04,-0.3,-0.16,-0.21,0.01,0.07,-0.06,0.06,0.13,0.33,0.25,-0.08,0.14,-0.13,-0.09,0.03,-0.05,0.22,0.24,-0.16,-0.06,0.1,-0.02,-0.15,0.1,-0.09,-0.39,0.21,-0.38,0.06,-0.17,-0.38,0.06,0.04,-0.31,-0.16,-0.28,-0.16,-0.08,0.28,0.21,-0.02,0.36,0.14,0.03,0.18,-0.16,-0.28,-0.26,-0.19,0.31,-0.34,-0.28,-0.12,-0.33,-0.11,-0.32,-0.29,-0.55,-0.25,-0.53,-0.54,-0.35,-0.23,-0.24,-0.14,-0.41,-0.42,-0.37,-0.48,-0.47,-0.57,-0.28,-0.29,-0.16,-0.5,-0.4,-0.25,-0.31,-0.16,-0.12,-0.01,-0.03,-0.01,0.0,-0.18,-0.05,-0.04,0.07,0.06,0.1,0.01,0.12,-0.06,-0.08,0.08,0.04,0.02,-0.09,0.35,-0.13,0.22,0.26,0.07,-0.08,0.01,0.5,-0.09,-0.14,0.0,0.26,0.14,0.16,0.0,0.02,0.03,-0.1,-0.41,-0.27,-0.12,-0.15,-0.03,0.06,0.13,-0.04,0.1,0.08,0.18,0.26,0.1,0.3,0.17,0.05,0.04,0.25,0.06,0.17,0.2,0.07,-0.03,0.26,0.35,0.27,0.14,0.05,0.6,0.36,0.23,0.67,0.12,0.29,0.37,0.16,-0.04,-0.06,0.03,0.24,-0.01,0.27,0.05,-0.15,0.04,0.04,0.16,0.3,0.39,-0.04,-0.11,0.03,0.05,0.29,0.15,0.25,0.34,0.39,0.11,0.13,0.21,0.04,0.61,0.21,0.22,0.12,0.19,0.27,0.2,0.03,-0.05,-0.03,0.19,0.27,-0.01,-0.21,0.04,0.08,0.15,-0.06,0.09,-0.11,0.15,-0.12,-0.18,-0.25,-0.06,-0.12,-0.02,-0.11,0.28,-0.03,-0.04,-0.17,-0.03,0.03,-0.04,0.17,0.06,-0.07,0.39,-0.07,0.1,0.06,0.44,-0.18,0.24,-0.05,-0.18,0.02,-0.14,-0.07,-0.07,0.0,-0.02,-0.28,-0.27,-0.25,-0.32,-0.4,-0.59,-0.21,-0.28,-0.42,-0.41,-0.37,-0.17,-0.28,-0.26,-0.34,-0.25,-0.17,-0.19,0.08,-0.12,-0.17,-0.2,-0.08,-0.33,-0.41,-0.38,-0.18,-0.4,-0.01,-0.05,-0.07,-0.09,0.0,0.0,-0.06,-0.06,-0.12,-0.03,0.24,0.21,0.24,0.12,-0.11,-0.04,-0.33,-0.15,-0.4,-0.26,-0.43,-0.24,-0.27,0.04,0.0,-0.09,-0.46,-0.38,-0.17,-0.5,-0.18,-0.34,-0.08,-0.27,-0.35,-0.16,-0.13,-0.18,-0.16,0.08,-0.06,0.02,-0.09,0.26,0.26,0.22,-0.03,-0.06,0.0,-0.16,0.26,0.3,0.33,0.1,0.41,-0.09,0.12,0.23,0.45,-0.05,0.24,0.54,0.22,0.33,0.12,0.36,0.31,0.13,0.14,0.26,0.28,0.31,0.29,0.46,0.5,-0.06,0.25,0.23,0.14,0.1,0.19,0.09,0.1,-0.01,0.14,0.06,-0.15,-0.1,-0.13,-0.09,-0.12,0.12,-0.14,-0.28,-0.18,0.03,-0.15,-0.24,0.05,-0.25,-0.23,0.23,0.32,0.01,-0.09,-0.02,-0.33,-0.19,-0.18,0.2,-0.12,-0.02,-0.04,-0.23,-0.17,-0.06,-0.27,-0.19,-0.19,-0.03,-0.15,0.2,-0.31,-0.07,-0.01,-0.13,0.11,-0.03,0.17,0.21,-0.29,-0.04,0.06,-0.07,-0.07,0.02,-0.25,-0.17,-0.1,-0.14,0.03,0.1,-0.14,-0.01,-0.08,-0.29,-0.19,-0.19,-0.23,-0.33,-0.16,-0.19,-0.06,-0.15,0.05,-0.29,-0.19,-0.25,-0.04,0.31,0.04,-0.11,0.29,-0.25,-0.3,-0.25,-0.03,-0.16,-0.25,-0.1,0.18,-0.25,-0.32,-0.15,-0.39,-0.34,-0.24,-0.41,0.13,-0.23,-0.14,-0.14,-0.11,-0.22,-0.13,-0.17,0.11,-0.23,0.14,-0.28,0.21,-0.09,-0.03,0.12,0.09,0.3,0.15,0.36,0.25,0.39,0.45,0.22,0.35,0.19,0.31,0.11,0.11,0.04,0.1,0.28,0.03,0.13,0.24,0.3,0.31,-0.04,0.18,0.45,0.1,0.11,0.04,-0.08,-0.03,-0.13,-0.23,-0.21,-0.25,-0.27,-0.11,-0.13,-0.3,0.03,-0.07,-0.11,-0.39,-0.33,-0.22,0.0,-0.17,-0.08,0.12,0.05,-0.03,0.17,-0.08,0.02,0.1,0.23,0.16,-0.05,-0.06,0.14,-0.03,0.13,0.01,-0.22,0.2,0.01,-0.07,0.22,0.0,0.02,0.0,0.08,-0.18,-0.1,0.0,-0.16,0.12,0.31,-0.31,-0.05,-0.09,0.16,-0.22,0.57,0.14,-0.11,0.17,-0.08,0.16,0.09,0.04,0.23,0.2,0.19,0.03,-0.08,0.25,0.16,0.52,0.38,-0.13,0.08,-0.25,-0.03,0.04,0.13,-0.12,0.11,-0.03,-0.16,-0.17,0.03,-0.14,-0.34,-0.05,-0.22,-0.03,0.05,0.11,-0.03,-0.08,-0.13,-0.44,-0.15,0.31,-0.15,-0.19,-0.33,-0.18,-0.06,-0.41,-0.28,-0.29,-0.53,-0.33,-0.44,-0.24,-0.61,-0.41,-0.25,-0.34,-0.28,-0.07,-0.04,-0.39,-0.3,-0.32,-0.37,-0.63,-0.23,-0.17,-0.09,-0.22,-0.4,-0.31,-0.27,-0.3,-0.25,-0.2,-0.12,-0.15,-0.2,-0.3,-0.35,-0.32,-0.17,-0.56,-0.42,0.02,-0.22,-0.35,-0.23,-0.31,-0.28,-0.1,-0.2,0.09,-0.18,-0.22,-0.01,0.12,0.04,0.04,-0.07,-0.12,0.06,-0.16,0.01,-0.13,-0.12,0.09,-0.13,-0.26,-0.07,-0.46,-0.39,-0.14,-0.4,-0.33,-0.53,-0.41,-0.18,-0.28,-0.05,-0.16,-0.04,-0.26,-0.28,-0.16,0.02,0.01,-0.1,0.06,0.19,-0.23,-0.24,0.25,-0.23,-0.09,-0.01,0.02,0.35,0.27,0.2,0.14,0.19,0.18,-0.09,0.17,-0.06,0.36,0.02,0.02,0.38,0.1,0.08,0.5,0.1,-0.15,-0.07,0.05,-0.06,-0.13,-0.05,0.07,-0.19,-0.04,0.01,-0.26,0.1,-0.03,-0.2,-0.07,-0.24,-0.1,-0.3,-0.35,-0.26,-0.15,-0.15,-0.01,-0.29,-0.47,0.09,0.16,-0.03,0.06,-0.09,-0.08,0.25,-0.11,-0.03,-0.18,-0.14,-0.07,0.02,0.12,-0.05,-0.26,0.03,0.1,0.05,-0.26,0.02,0.26,0.19,-0.15,0.18,-0.09,0.29,0.16,0.12,0.09,0.46,-0.07,-0.03,0.26,-0.02,-0.01,-0.17,-0.08,-0.13,-0.2,-0.04,0.21,-0.16,0.3,-0.03,-0.11,0.09,-0.14,-0.09,0.06,-0.04,-0.04,0.19,0.29,0.06,-0.08,-0.01,0.29,0.38,0.15,0.27,-0.08,0.08,0.16,-0.16,0.02,0.16,0.3,0.06,0.04,0.23,0.02,-0.07,-0.24,-0.21,-0.32,-0.36,0.12,-0.09,-0.26,-0.34,-0.11,0.03,-0.04,0.02,-0.19,0.09,-0.13,-0.25,-0.06,0.36,0.0,0.04,-0.15,0.11,0.48,0.11,0.26,0.13,0.17,-0.05,0.1,0.13,0.09,0.04,0.02,-0.08,0.06,-0.1,0.19,-0.12,0.35,0.17,0.1,0.43,0.12,0.04,0.06,0.17,-0.02,0.2,-0.17,0.1,0.0,0.08,0.07,-0.23,-0.03,0.35,-0.01,0.08,-0.13,-0.01,-0.05,-0.03,-0.09,-0.1,-0.06,0.04,-0.25,0.18,-0.07,-0.49,-0.12,-0.2,0.26,-0.27,0.44,-0.13,-0.03,-0.05,-0.03,-0.03,-0.01,-0.2,-0.19,0.24,-0.03,0.39,-0.27,0.0,0.03,-0.04,0.28,0.15,-0.17,0.13,0.15,0.19,0.14,-0.17,-0.06,0.04,-0.1,-0.28,0.21,-0.09,-0.21,0.05,-0.05,0.13,-0.03,0.48,0.23,0.18,0.38,0.31,-0.13,0.16,-0.05,-0.15,-0.11,0.35,-0.08,-0.09,0.03,0.32,-0.01,0.37,0.08,0.03,-0.01,-0.01,-0.03,0.13,0.16,-0.09,-0.04,-0.06,-0.1,0.31,-0.24,-0.01,-0.1,-0.07,0.54,0.06,-0.12,0.23,0.14,0.14,0.16,0.34,0.42,0.31,0.15,0.25,0.41,0.32,0.55,0.17,-0.03,0.09,0.06,0.14,-0.18,-0.13,-0.33,0.1,-0.07,0.13,-0.25,-0.04,-0.03,0.07,-0.03,-0.12,0.1,0.0,0.16,0.08,0.09,0.17,0.25,0.08,0.27,0.52,0.37,0.17,0.21,0.04,-0.13,-0.07,0.08,-0.03,-0.07,-0.23,0.19,0.4,0.16,0.02,0.16,0.29,0.01,0.05,0.04,0.33,0.17,-0.03,-0.01,-0.1,0.15,0.39,0.11,-0.06,-0.08,0.31,0.2,0.09,0.06,0.27,0.16,0.09,0.15,0.12,-0.06,0.18,0.15,-0.14,-0.15,0.03,-0.1,-0.34,0.01,-0.11,0.17,0.24,-0.2,0.16,0.25,-0.15,-0.13,0.07,-0.01,-0.06,-0.23,-0.23,-0.32,-0.1,-0.3,-0.03,0.06,0.13,-0.49,-0.01,-0.19,-0.16,-0.25,-0.22,-0.45,-0.12,-0.24,-0.46,-0.51,-0.29,-0.3,-0.37,-0.15,-0.28,-0.12,-0.56,-0.3,-0.31,-0.23,-0.47,-0.32,-0.01,0.02,-0.38,-0.11,-0.17,0.29,-0.24,-0.03,0.02,-0.26,-0.21,-0.34,-0.11,0.0,-0.18,-0.31,-0.19,-0.21,-0.18,-0.39,-0.24,-0.37,-0.09,0.04,-0.05,-0.2,-0.39,-0.16,-0.19,-0.13,-0.31,0.07,0.26,-0.11,-0.17,0.17,0.29,0.04,-0.13,0.09,-0.23,0.3,0.14,0.02,-0.03,0.11,0.14,0.18,-0.03,0.02,-0.29,0.01,-0.03,-0.05,-0.03,0.23,0.15,-0.21,0.2,0.03,-0.1,-0.25,-0.1,-0.09,0.08,0.4,0.15,0.11,0.15,0.2,0.26,-0.03,0.57,0.12,-0.07,-0.1,0.35,0.04,-0.15,0.23,-0.09,0.2,-0.05,0.31,0.18,0.24,0.44,0.06,0.04,0.13,0.11,0.05,0.18,0.19,0.08,-0.11,-0.29,0.06,-0.01,0.34,0.12,0.02,0.11,0.02,0.02,-0.14,0.29,0.16,0.17,0.16,0.04,-0.13,0.03,-0.03,0.27,0.37,0.23,0.09,0.23,0.41,0.44,0.21,0.42,0.28,0.22,0.04,-0.1,0.65,0.2,0.27,0.18,0.21,0.04,0.2,0.21,0.24,0.39,0.35,0.59,0.16,0.44,0.14,0.4,0.55,0.06,0.1,0.38,0.11,0.04,-0.04,0.09,0.24,0.18,-0.12,0.19,0.0,0.04,0.15,-0.09,-0.21,-0.13,0.38,0.08,-0.03,-0.06,-0.28,-0.11,0.16,0.04,-0.28,0.19,0.01,-0.19,-0.13,0.04,0.09,-0.16,-0.07,0.12,0.02,-0.24,-0.09,-0.13,0.18,-0.16,-0.01,0.03,0.1,0.41,-0.17,0.09,-0.2,-0.21,-0.22,-0.14,-0.2,-0.22,0.1,-0.07,0.38,0.01,-0.17,-0.41,-0.16,-0.09,-0.16,-0.43,-0.41,-0.02,-0.16,-0.12,-0.24,-0.02,-0.12,-0.16,0.21,-0.13,0.01,-0.35,-0.12,-0.07,0.13,0.01,0.2,0.29,0.03,-0.01,0.39,0.03,0.0,0.14,0.09,-0.04,0.03,0.02,0.03,-0.03,-0.17,0.09,0.3,0.13,0.01,-0.05,-0.29,-0.11,-0.38,-0.11,0.1,-0.02,0.23,-0.19,-0.14,-0.12,0.04,-0.03,-0.1,-0.26,-0.14,-0.26,-0.36,-0.33,-0.28,-0.54,-0.23,-0.39,-0.44,-0.16,-0.24,-0.46,-0.38,-0.17,-0.12,-0.27,-0.35,-0.16,-0.17,-0.41,0.1,0.16,-0.31,0.11,0.17,-0.2,0.03,-0.15,-0.23,-0.18,-0.25,0.19,-0.3,-0.12,-0.3,0.14,-0.32,-0.06,-0.14,-0.2,-0.28,-0.27,-0.38,-0.48,-0.39,-0.36,-0.18,-0.38,-0.28,-0.26,-0.27,-0.3,-0.4,0.03,-0.13,-0.05,-0.01,-0.22,0.09,0.02,0.06,-0.52,-0.16,0.04,-0.17,-0.26,-0.02,0.11,-0.23,-0.04,-0.13,-0.14,-0.1,0.03,-0.13,-0.08,-0.05,-0.01,-0.23,0.15,-0.03,-0.13,-0.14,-0.05,0.04,0.12,0.38,0.07,0.04,0.12,-0.26,-0.27,-0.17,-0.46,-0.22,-0.18,-0.53,-0.39,-0.42,-0.56,-0.38,-0.19,-0.28,-0.39,-0.16,-0.23,-0.06,-0.13,-0.37,-0.3,-0.22,0.16,0.01,-0.4,-0.13,-0.44,0.03,-0.18,0.26,-0.17,0.02,-0.09,0.09,-0.13,0.1,0.05,0.2,-0.16,-0.29,0.16,-0.19,-0.07,0.12,0.12,0.1,-0.05,0.09,-0.18,-0.03,-0.03,0.17,0.02,0.03,-0.17,0.2,-0.14,-0.22,-0.32,-0.29,-0.46,-0.27,-0.31,-0.39,-0.03,-0.51,-0.43,-0.46,-0.37,-0.25,-0.46,-0.19,-0.27,-0.37,-0.37,-0.18,-0.32,0.05,-0.45,-0.2,-0.06,-0.22,-0.12,-0.28,0.21,-0.05,0.17,0.43,-0.06,0.1,0.25,0.11,0.04,0.02,0.34,0.13,0.1,-0.11,0.35,0.31,-0.09,0.02,0.23,0.19,0.11,0.12,0.15,-0.01,-0.02,0.15,-0.17,-0.03,-0.11,-0.13,0.04,-0.21,-0.24,0.14,-0.24,-0.19,-0.24,0.01,0.27,0.15,-0.22,-0.04,-0.13,-0.13,-0.29,-0.24,0.15,-0.02,-0.05,-0.34,-0.15,0.15,-0.17,-0.07,0.06,-0.16,0.15,0.21,-0.01,-0.1,-0.14,-0.1,0.17,0.21,0.01,0.04,0.08,0.03,0.06,0.2,-0.16,-0.14,-0.19,-0.09,0.04,-0.19,0.06,0.04,-0.08,-0.33,0.04,-0.09,-0.16,-0.02,-0.32,0.0,-0.19,0.02,-0.07,0.1,0.13,0.21,0.09,0.06,0.08,0.0,0.06,-0.1,-0.16,-0.07,-0.06,-0.04,0.43,-0.03,0.18,-0.1,0.19,-0.03,0.02,0.12,-0.24,-0.12,0.18,-0.13,-0.03,-0.12,-0.2,-0.26,-0.28,0.04,-0.14,-0.17,-0.22,-0.1,-0.18,-0.33,-0.29,0.02,-0.36,-0.25,-0.27,-0.16,-0.24,-0.57,0.18,-0.37,-0.08,-0.21,-0.33,-0.27,-0.3,-0.31,-0.29,0.18,-0.16,-0.34,-0.26,-0.04,-0.27,0.08,-0.25,-0.24,-0.3,-0.06,-0.27,0.09,0.17,-0.12,-0.08,0.08,0.06,0.08,0.12,0.31,0.31,0.16,0.19,0.25,0.18,0.23,0.3,0.17,-0.04,0.03,0.24,-0.1,0.04,0.19,0.49,0.11,0.32,-0.1,0.24,0.17,0.18,0.04,-0.21,0.1,-0.07,-0.03,-0.09,-0.1,-0.37,-0.07,-0.09,0.16,-0.16,-0.13,0.16,-0.05,-0.05,0.39,0.19,-0.01,0.09,-0.07,0.04,0.13,0.24,0.17,0.08,-0.11,0.27,-0.03,-0.31,0.0,-0.16,-0.11,0.09,-0.03,-0.18,0.01,0.03,0.15,0.25,0.12,-0.29,-0.03,0.23,0.23,-0.02,-0.03,-0.32,0.08,-0.27,0.04,-0.16,-0.09,-0.03,-0.06,-0.28,-0.23,-0.07,-0.16,0.12,-0.11,-0.24,-0.57,-0.42,-0.27,-0.26,0.09,-0.06,-0.47,-0.21,-0.21,-0.2,0.0,-0.27,-0.19,0.09,-0.33,-0.18,0.08,0.0,0.21,0.05,-0.13,0.26,-0.2,0.26,-0.28,-0.14,0.03,0.04,0.1,0.06,0.32,-0.05,0.2,0.02,0.34,-0.07,-0.03,0.24,0.0,0.01,0.19,0.1,0.17,0.17,0.57,0.19,0.36,0.12,0.15,0.17,0.08,0.14,-0.06,-0.06,0.24,0.11,0.24,0.14,0.25,0.04,0.2,-0.06,0.11,0.19,-0.07,0.24,-0.02,0.04,-0.27,-0.27,-0.01,-0.08,0.14,-0.22,-0.14,-0.27,0.06,-0.12,0.09,-0.01,-0.12,0.04,-0.14,-0.2,0.04,-0.04,-0.08,0.06,-0.08,-0.3,-0.21,0.25,-0.45,-0.03,-0.08,0.12,-0.21,0.0,0.23,-0.13,-0.16,-0.22,-0.18,-0.02,-0.13,0.02,-0.16,-0.12,-0.07,-0.36,0.21,0.43,0.07,-0.22,0.09,-0.09,-0.11,0.09,-0.01,-0.08,-0.01,0.17,0.04,0.0,0.12,-0.01,0.26,0.19,0.08,-0.02,-0.02,-0.06,0.09,-0.08,-0.25,-0.17,-0.22,-0.24,0.06,-0.24,-0.52,-0.16,-0.44,-0.1,-0.16,-0.27,0.09,-0.1,0.0,-0.22,-0.19,-0.03,-0.05,0.08,0.03,-0.03,-0.26,0.19,-0.12,-0.06,-0.12,-0.35,-0.11,-0.04,-0.22,0.18,-0.06,0.15,-0.03,0.48,0.06,0.09,0.02,0.33,0.42,0.21,0.4,0.14,0.09,0.17,0.21,0.28,-0.02,0.21,0.26,0.1,0.09,0.38,-0.08,0.34,0.12,-0.14,0.01,0.06,-0.04,0.11,0.19,-0.07,-0.14,0.16,-0.11,0.02,-0.03,-0.13,0.26,0.18,0.36,0.01,0.36,0.26,0.43,0.12,0.1,0.58,0.35,-0.15,0.01,-0.06,-0.01,0.03,0.01,-0.21,-0.27,0.12,0.13,0.16,0.52,0.08,0.28,0.28,0.22,0.12,0.16,0.06,0.17,0.09,-0.24,-0.35,-0.06,-0.19,-0.4,-0.33,-0.22,-0.13,-0.46,-0.18,0.15,0.1,-0.27,-0.12,0.1,-0.45,-0.27,-0.22,-0.37,-0.24,-0.22,-0.38,-0.12,-0.42,-0.37,-0.21,-0.57,-0.07,0.05,-0.24,0.07,-0.17,-0.15,0.2,0.02,-0.24,-0.11,0.07,-0.09,0.12,0.38,-0.12,-0.17,0.18,0.15,0.06,0.08,0.15,-0.19,-0.06,-0.06,-0.05,0.33,-0.14,0.28,-0.1,-0.07,-0.13,0.06,0.16,0.27,0.16,0.41,0.23,0.56,0.21,0.1,0.14,0.19,0.2,0.18,0.48,0.12,0.44,0.31,0.45,0.63,0.14,0.23,0.15,0.4,0.11,0.18,-0.04,0.09,0.62,0.13,0.29,0.0,0.03,0.28,0.09,0.26,0.1,0.14,0.08,0.23,0.32,-0.01,0.2,0.01,-0.05,-0.26,0.23,0.23,-0.03,0.2,0.02,-0.03,0.07,0.16,0.42,0.02,0.23,0.18,0.47,0.03,0.18,0.36,0.14,0.39,0.35,0.3,0.1,0.38,0.22,0.16,0.13,0.45,0.43,0.37,0.23,0.38,-0.06,0.25,0.27,0.15,0.31,0.18,0.17,0.15,0.5,0.25,0.32,0.48,0.21,0.35,0.28,0.31,0.34,0.13,0.27,0.33,0.37,0.46,0.41,0.24,0.46,0.02,0.47,0.04,0.22,0.23,0.15,0.1,0.32,0.33,0.09,0.58,0.3,0.19,0.25,0.43,0.22,0.45,0.25,0.21,0.18,0.33,0.24,0.17,0.23,0.42,0.26,0.06,0.49,0.39,0.29,0.25,0.03,0.17,0.4,0.08,0.14,0.21,0.22,0.06,-0.05,0.23,-0.02,-0.13,0.18,0.04,-0.03,0.25,-0.08,0.03,0.13,0.09,-0.39,-0.13,-0.11,-0.18,-0.24,0.03,-0.1,-0.2,-0.39,-0.15,-0.25,-0.17,-0.09,0.06,-0.3,-0.2,-0.07,-0.46,-0.38,-0.56,-0.67,-0.55,-0.3,-0.71,-0.65,-0.64,-0.55,-0.99,-1.01,-0.77,-0.88,-0.71,-0.99,-1.0,-0.5,-0.96,-0.77,-0.98,-1.14,-0.67,-0.92,-1.18,-0.61,-0.7,-1.06,-0.91,-0.93,-0.81,-1.11,-1.17,-0.86,-1.2,-0.88,-0.64,-1.11,-0.81,-0.56,-1.0,-0.74,-1.17,-1.1,-0.98,-1.03,-0.98,-1.04,-0.72,-0.87,-0.82,-0.6,-0.76,-0.69,-0.86,-0.68,-0.36,-0.79,-0.71,-0.46,-0.78,-0.63,-0.41,-0.53,-0.64,-0.45,-0.38,-0.25,-0.29,-0.34,-0.17,-0.03,-0.14,-0.14,-0.06,-0.13,0.37,-0.06,0.13,0.12,0.03,-0.22,0.21,0.04,-0.04,0.03,-0.27,-0.4,0.25,0.14,0.52,0.06,-0.12,0.02,0.14,0.1,0.4,0.01,0.25,0.04,-0.06,0.23,0.08,0.28,0.27,-0.03,0.19,0.09,0.56,0.4,0.3,-0.03,0.16,-0.14,-0.1,-0.14,-0.38,0.09,-0.02,-0.05,-0.29,-0.18,0.08,0.03,0.17,-0.08,0.08,-0.07,-0.09,-0.02,0.0,0.23,-0.05,0.38,-0.07,-0.03,-0.18,-0.12,0.07,-0.03,0.06,-0.12,0.09,0.02,-0.05,0.31,0.28,0.11,0.07,-0.04,-0.07,-0.02,-0.09,-0.06,-0.3,0.0,-0.43,-0.21,-0.02,0.06,-0.24,-0.17,-0.13,0.0,-0.2,-0.1,0.04,-0.39,0.29,-0.28,-0.12,-0.14,0.15,0.18,0.16,-0.24,-0.12,0.1,0.13,0.37,0.06,0.07,0.05,-0.14,-0.28,-0.24,-0.12,-0.34,0.09,-0.15,-0.33,-0.28,-0.09,-0.14,0.02,-0.13,-0.19,-0.13,0.06,-0.07,-0.11,-0.21,-0.02,0.13,-0.18,0.06,0.12,-0.01,-0.22,-0.48,0.03,-0.34,-0.17,-0.31,-0.09,-0.14,0.01,-0.33,-0.07,-0.34,-0.18,-0.25,-0.39,-0.38,-0.38,-0.16,-0.37,0.2,-0.56,-0.29,-0.37,-0.09,-0.28,0.11,-0.1,-0.2,-0.27,0.02,-0.16,-0.15,0.03,0.35,-0.03,0.18,0.0,0.03,0.15,0.22,0.64,0.03,-0.05,-0.18,-0.09,-0.19,-0.01,0.19,-0.01,0.49,0.17,0.04,0.15,0.14,0.41,-0.05,-0.03,0.5,0.24,0.45,0.19,0.56,0.16,0.01,0.34,0.27,-0.06,0.08,-0.08,0.2,-0.06,0.19,0.08,0.11,0.19,0.16,0.11,0.08,0.26,0.12,0.2,0.09,0.14,-0.08,0.11,0.44,0.28,0.02,0.08,0.06,0.02,-0.01,-0.06,0.06,0.15,0.19,-0.22,0.41,0.04,0.1,-0.14,0.38,0.54,0.09,0.11,0.0,0.27,0.25,0.2,0.11,-0.05,0.17,0.1,0.12,-0.01,0.15,0.13,0.11,0.28,0.1,0.19,0.02,0.14,0.15,0.19,0.27,0.31,0.24,0.16,0.08,0.1,0.07,-0.83,0.52,0.19,0.1,0.12,0.15,0.04,-0.3,0.17,0.08,0.21,0.0,-0.13,-0.16,-0.13,-0.4,0.2,0.02,-0.38,0.02,-0.05,-0.13,-0.28,-0.31,-0.39,-0.42,-0.19,-0.05,-0.17,-0.38,-0.27,-0.15,-0.18,-0.16,-0.21,0.02,-0.08,0.1,0.3,0.02,-0.1,-0.15,-0.16,-0.21,0.01,-0.22,-0.29,0.31,0.11,0.02,0.29,0.38,0.64,0.51,0.37,0.17,0.24,0.51,0.7,0.45,0.78,1.1,0.45,0.54,0.41,0.39,0.42,0.35,0.72,0.09,0.59,0.64,0.52,0.64,0.35,0.22,0.21,0.49,0.59,0.66,0.5,0.6,0.32,0.18,0.66,0.03,0.51,0.36,0.29,0.54,0.65,0.08,0.12,0.41,0.31,0.46,0.66,0.5,0.68,0.47,0.41,0.61,0.59,0.46,0.72,0.65,0.58,0.5,0.44,0.82,0.53,0.43,0.55,0.44,0.95,0.73,0.66,0.63,0.62,0.55,0.49,0.59,0.44,0.43,0.54,0.53,0.57,0.19,0.45,0.33,0.4,0.49,0.45,0.21,0.09,0.05,0.43,0.38,0.17,0.24,0.61,0.14,-0.01,0.11,0.05,0.29,0.17,0.22,0.23,-0.22,-0.07,0.3,-0.22,-0.09,-0.05,0.08,0.21,-0.03,0.06,0.24,0.06,0.18,0.03,0.21,0.25,0.12,0.16,-0.4,0.03,-0.27,0.04,0.06,-0.12,-0.33,-0.12,-0.11,0.05,-0.03,0.01,0.04,-0.19,-0.13,-0.12,-0.08,-0.11,0.18,0.08,-0.15,-0.07,-0.18,-0.05,0.08,0.14,0.03,-0.27,-0.2,-0.07,0.23,0.1,0.1,-0.16,-0.22,-0.11,-0.08,-0.32,-0.28,-0.45,-0.23,-0.28,-0.29,-0.27,-0.34,-0.16,-0.46,-0.34,-0.13,-0.48,-0.5,0.1,0.06,-0.23,-0.13,-0.07,-0.07,-0.05,-0.19,-0.11,-0.27,-0.15,-0.29,-0.24,0.03,-0.12,-0.06,-0.09,-0.07,-0.09,-0.09,0.1,-0.12,-0.2,0.22,-0.3,-0.24,0.05,0.13,-0.15,-0.31,-0.28,0.0,0.06,-0.32,-0.13,-0.16,-0.19,-0.24,-0.3,-0.07,-0.1,-0.16,-0.3,-0.05,-0.25,-0.22,-0.42,-0.4,-0.39,-0.08,-0.19,-0.04,-0.46,0.14,-0.46,-0.4,-0.37,-0.45,-0.48,-0.28,-0.31,-0.21,-0.43,-0.39,-0.39,-0.23,-0.46,-0.37,-0.36,-0.3,0.01,-0.2,0.03,-0.09,0.06,0.13,0.03,-0.12,-0.1,-0.17,-0.13,0.1,0.06,0.1,-0.12,-0.05,0.28,0.01,-0.04,-0.03,0.12,0.18,0.22,-0.11,-0.1,-0.12,0.0,0.09,0.19,0.04,-0.08,0.17,0.16,0.02,-0.05,0.02,0.0,0.44,0.22,-0.07,-0.05,0.31,-0.07,-0.07,0.0,0.21,0.24,0.15,0.02,-0.03,0.1,0.19,-0.11,0.04,-0.21,-0.02,-0.14,-0.08,-0.07,-0.01,-0.11,0.01,0.13,0.06,-0.01,-0.06,0.11,-0.01,0.13,0.06,-0.11,-0.06,-0.23,0.39,0.3,-0.23,-0.13,-0.2,-0.28,-0.27,0.06,-0.11,-0.29,0.02,-0.14,0.06,-0.06,-0.13,-0.16,-0.02,-0.1,-0.1,-0.01,-0.14,0.26,-0.14,-0.19,0.04,0.0,-0.16,-0.25,0.11,0.1,-0.05,-0.13,0.03,0.23,-0.14,0.45,0.04,0.34,0.13,0.04,0.17,0.49,0.47,0.04,0.22,0.14,0.04,0.15,0.42,0.2,0.27,0.36,0.28,0.14,0.22,0.16,0.46,-0.01,0.48,0.31,-0.15,0.22,0.1,0.38,0.4,0.4,0.05,0.24,0.28,0.69,0.3,-0.37,0.36,0.78,0.56,0.54,0.52,0.54,0.44,0.23,0.1,0.15,0.08,0.03,0.02,-0.28,-0.11,0.02,-0.4,-0.15,-0.34,-0.18,-0.37,-0.21,0.06,-0.27,-0.29,-0.27,-0.09,-0.05,-0.22,-0.03,-0.3,-0.21,0.01,-0.06,-0.05,-0.04,-0.08,0.18,0.09,-0.22,-0.28,0.26,0.02,0.17,0.04,0.43,0.0,-0.03,-0.13,0.06,0.35,0.06,0.24,0.37,0.24,0.18,0.23,0.07,-0.09,0.22,0.2,0.13,-0.16,-0.06,-0.05,-0.24,0.06,0.02,0.21,0.0,0.0,-0.26,0.06,-0.18,-0.05,-0.13,0.25,0.01,-0.15,0.23,-0.34,-0.52,-0.5,-0.21,-0.51,-0.25,-0.41,-0.26,-0.41,-0.21,-0.3,-0.1,-0.08,-0.3,-0.12,-0.38,-0.15,-0.32,-0.09,-0.04,-0.24,-0.52,-0.24,-0.2,0.06,-0.57,-0.07,-0.34,-0.31,-0.46,-0.28,-0.07,-0.49,-0.14,-0.33,-0.34,-0.11,-0.24,-0.19,-0.38,-0.4,-0.1,-0.51,-0.34,-0.51,-0.34,-0.23,-0.28,-0.3,-0.31,-0.23,-0.33,-0.13,0.23,-0.09,0.16,-0.2,0.16,0.03,0.32,0.0,0.13,0.19,-0.15,0.32,0.1,0.49,0.01,0.31,0.12,0.42,0.16,-0.03,0.32,-0.1,0.11,-0.14,-0.13,-0.21,0.04,-0.25,-0.16,-0.03,-0.31,-0.31,-0.35,0.03,0.14,0.26,-0.41,0.09,-0.22,-0.15,-0.24,-0.28,-0.31,0.19,-0.25,-0.43,0.08,-0.34,-0.34,-0.04,0.14,-0.4,-0.06,0.0,-0.35,-0.3,-0.21,-0.31,-0.38,-0.04,0.06,-0.07,0.09,-0.07,-0.18,0.15,-0.03,-0.05,0.01,0.08,-0.23,-0.05,-0.07,0.09,0.19,0.0,-0.15,0.34,-0.1,0.35,0.18,0.16,0.54,0.42,0.17,0.37,0.31,0.25,0.31,0.47,0.37,0.64,0.33,0.34,0.46,0.33,0.06,0.39,0.38,0.26,0.28,0.2,0.34,0.18,0.22,0.38,0.45,0.12,0.52,0.17,0.14,0.08,0.42,0.24,-0.08,0.49,0.58,0.28,0.37,0.18,-0.05,0.19,0.22,0.18,0.16,0.09,0.16,0.13,0.01,0.16,0.13,0.09,0.29,0.14,0.13,0.26,0.14,0.33,-0.01,-0.12,0.13,0.04,-0.13,-0.03,0.09,-0.15,0.02,-0.5,-0.04,-0.12,0.01,-0.28,-0.2,-0.04,-0.32,-0.24,-0.01,-0.13,0.22,0.03,-0.01,-0.47,0.17,-0.3,-0.04,-0.14,-0.22,0.17,-0.22,0.09,-0.31,-0.21,0.12,0.24,-0.12,-0.24,-0.12,-0.31,-0.3,0.42,0.21,0.12,-0.15,-0.15,0.46,0.42,0.16,0.21,0.28,-0.04,0.08,-0.21,0.04,0.03,0.09,0.19,-0.13,-0.07,0.03,-0.03,0.04,0.03,-0.05,0.0,-0.26,0.13,-0.31,-0.26,-0.24,-0.24,0.03,-0.13,0.34,0.31,0.04,-0.22,-0.05,-0.17,-0.15,0.01,0.09,-0.26,-0.12,-0.34,0.02,0.27,-0.01,-0.14,-0.07,-0.06,0.11,0.36,0.53,0.2,0.12,0.27,0.13,0.09,0.2,0.23,-0.1,0.11,0.21,0.04,-0.1,-0.18,0.21,-0.3,0.06,0.03,0.0,0.03,-0.16,-0.32,-0.22,0.03,-0.2,-0.27,-0.24,0.09,-0.01,-0.05,0.15,-0.35,-0.19,-0.06,0.0,0.02,0.35,-0.17,-0.06,-0.27,-0.25,-0.59,-0.42,-0.36,-0.47,0.27,-0.39,-0.42,-0.3,-0.04,-0.01,0.08,-0.03,-0.13,0.2,-0.26,-0.27,-0.16,-0.26,-0.42,-0.12,-0.17,-0.2,-0.24,0.13,-0.16,0.06,-0.01,0.31,0.3,0.04,0.18,0.36,0.17,0.39,0.21,0.1,0.19,-0.01,0.18,0.45,0.02,0.39,0.51,0.19,0.18,-0.03,0.16,0.3,0.08,0.4,0.02,0.04,0.36,0.41,0.34,0.26,0.04,0.41,0.04,0.3,0.18,0.44,0.44,0.79,0.3,0.05,0.04,-0.01,0.27,0.27,0.13,-0.09,0.33,0.03,0.0,-0.17,0.01,0.22,-0.03,0.38,-0.06,0.37,-0.04,0.3,0.24,0.01,0.32,-0.03,0.15,0.12,0.28,0.08,0.08,0.36,0.33,0.29,0.58,0.27,0.4,0.34,0.18,0.23,0.08,0.09,-0.12,-0.38,0.08,0.61,0.17,0.11,-0.17,0.21,0.0,-0.05,-0.17,-0.03,-0.37,-0.21,-0.52,-0.32,-0.38,-0.51,-0.03,-0.32,-0.32,-0.23,-0.3,-0.35,-0.34,0.08,-0.22,-0.23,-0.42,-0.55,-0.54,-0.18,-0.47,-0.04,-0.41,-0.3,-0.28,-0.26,-0.12,-0.32,-0.07,-0.09,-0.06,-0.26,-0.27,-0.27,-0.26,-0.03,-0.3,0.03,0.08,-0.14,-0.04,-0.08,-0.09,0.01,-0.26,-0.04,0.12,0.0,-0.43,-0.19,-0.18,-0.3,0.1,0.14,-0.18,-0.34,0.0,-0.12,0.24,-0.19,0.01,0.09,0.44,-0.1,0.31,-0.05,0.04,0.23,0.1,0.28,0.02,0.16,0.22,0.36,0.16,0.01,0.07,0.02,0.11,0.19,-0.05,0.23,0.14,0.0,0.54,0.06,0.02,-0.03,0.17,0.53,0.39,0.31,0.08,0.43,0.26,0.39,0.15,0.74,0.68,0.47,0.28,0.14,0.37,0.28,0.33,0.34,0.2,0.27,0.66,0.25,0.1,0.08,-0.11,0.1,0.18,0.24,-0.06,0.32,-0.14,-0.06,0.06,-0.11,-0.21,-0.04,-0.13,0.16,0.05,0.25,0.19,-0.14,0.26,-0.05,0.03,0.2,-0.23,-0.15,-0.12,-0.04,0.17,0.21,-0.12,-0.29,-0.14,0.02,0.0,-0.13,-0.23,-0.27,-0.03,-0.12,0.25,-0.24,-0.05,-0.21,-0.11,0.0,-0.48,-0.46,0.13,-0.49,-0.4,0.01,-0.52,-0.25,-0.45,-0.16,-0.34,-0.43,-0.27,-0.37,-0.44,-0.35,-0.26,-0.19,-0.4,-0.22,-0.17,-0.39,-0.4,-0.18,-0.12,-0.43,-0.16,-0.5,0.1,-0.16,-0.24,-0.22,-0.25,-0.05,-0.03,-0.06,-0.38,-0.17,-0.28,-0.29,0.02,-0.03,-0.32,-0.4,-0.08,-0.15,-0.26,-0.29,-0.08,-0.13,-0.09,-0.07,-0.18,-0.27,-0.24,-0.32,0.33,-0.05,-0.24,-0.19,0.1,-0.16,-0.1,0.16,0.0,-0.19,0.17,0.13,0.13,0.04,-0.06,-0.09,-0.16,-0.21,-0.29,-0.23,-0.29,0.1,-0.07,-0.15,-0.09,-0.28,-0.31,0.24,-0.13,0.11,0.02,0.25,-0.01,-0.25,-0.14,-0.08,-0.41,-0.23,0.0,-0.42,-0.15,-0.08,0.0,-0.17,0.18,-0.11,0.0,-0.13,-0.08,-0.09,-0.23,-0.1,-0.32,0.17,-0.46,-0.2,-0.21,-0.11,-0.26,-0.18,-0.45,-0.1,-0.22,-0.47,-0.05,-0.04,-0.11,-0.35,-0.21,-0.5,-0.39,-0.14,-0.2,-0.54,-0.12,-0.16,-0.08,0.31,0.28,0.03,-0.09,0.13,-0.09,-0.03,0.08,0.09,0.27,0.24,0.16,-0.16,-0.09,0.03,-0.07,-0.05,0.17,0.18,-0.29,0.08,-0.13,-0.13,-0.2,0.06,-0.22,-0.11,0.09,-0.17,0.06,0.0,-0.16,-0.32,0.19,0.0,-0.22,-0.12,-0.42,-0.03,-0.1,0.04,-0.25,0.17,0.14,0.19,0.19,0.21,0.09,0.06,-0.09,0.04,0.22,-0.01,0.16,0.2,0.14,0.5,0.54,0.2,0.32,0.2,0.29,0.39,0.33,0.37,0.28,0.28,0.32,0.36,0.25,0.54,0.44,0.19,-0.01,0.23,0.2,0.22,0.32,0.34,0.2,-0.03,0.13,0.36,-0.12,-0.05,0.08,0.28,0.18,0.14,0.22,0.13,-0.06,0.04,-0.15,0.24,-0.02,-0.18,-0.12,0.01,-0.1,-0.21,0.04,-0.17,-0.04,0.28,-0.03,-0.18,0.01,0.15,0.46,-0.2,-0.07,0.23,-0.02,0.1,0.1,0.04,0.18,0.03,0.1,0.26,0.21,0.34,0.17,0.16,0.04,-0.03,0.19,0.1,0.13,0.15,0.18,0.33,0.34,0.31,0.43,0.08,0.1,0.06,0.22,0.17,0.5,0.19,0.0,0.06,0.46,0.42,0.08,0.58,0.19,0.28,0.42,0.15,0.32,0.23,0.62,0.23,0.15,0.49,0.32,0.14,0.41,0.13,0.11,-0.06,-0.25,0.13,-0.45,-0.08,-0.26,0.06,-0.4,0.13,-0.14,-0.05,0.14,0.11,-0.06,-0.05,0.16,0.26,-0.19,0.09,-0.17,0.1,-0.12,-0.04,0.03,-0.22,-0.03,-0.16,-0.19,-0.1,0.2,-0.18,0.34,0.03,-0.17,0.1,0.12,0.23,0.45,0.19,0.55,-0.02,0.06,0.35,0.08,0.03,0.16,-0.06,-0.32,0.12,-0.04,-0.04,-0.08,-0.23,-0.33,-0.17,-0.54,-0.54,-0.32,-0.51,-0.34,-0.65,-0.48,-0.48,-0.21,-0.33,-0.12,-0.27,-0.52,0.08,-0.15,-0.19,-0.32,0.15,0.08,0.16,0.49,0.06,0.0,0.21,-0.01,0.05,-0.22,-0.06,-0.11,-0.15,-0.06,-0.03,0.17,0.27,0.39,0.16,-0.04,0.07,0.23,-0.11,-0.16,-0.26,0.08,-0.26,-0.04,-0.07,0.18,-0.12,-0.29,-0.15,-0.2,-0.39,0.09,-0.08,0.02,0.08,-0.33,-0.27,0.02,-0.28,-0.17,-0.05,-0.05,-0.24,-0.16,-0.25,-0.2,-0.09,-0.22,0.27,-0.14,-0.04,-0.03,-0.14,0.42,0.37,0.1,0.17,0.04,0.0,-0.03,-0.18,0.11,-0.03,-0.08,-0.31,-0.2,-0.01,0.21,0.01,-0.41,0.31,-0.07,-0.14,-0.13,-0.18,0.13,-0.26,-0.32,-0.28,0.06,0.44,0.47,-0.06,-0.1,-0.15,-0.01,0.17,-0.23,-0.03,0.22,-0.13,0.02,-0.29,-0.18,-0.26,-0.16,0.21,-0.23,-0.22,-0.06,-0.2,-0.43,-0.33,-0.11,-0.19,-0.21,0.0,-0.1,-0.28,0.1,0.16,-0.04,-0.22,-0.05,0.29,0.65,0.21,0.15,-0.23,-0.16,0.0,-0.09,-0.18,0.06,0.12,-0.29,-0.11,0.0,0.07,-0.18,-0.23,0.07,-0.3,0.21,-0.18,-0.07,0.17,-0.06,0.03,0.15,-0.25,-0.07,-0.05,0.06,0.22,-0.17,0.02,0.06,-0.12,-0.23,-0.01,-0.12,-0.07,-0.19,-0.19,-0.06,-0.17,-0.3,-0.27,0.0,-0.23,-0.06,0.15,-0.06,-0.36,0.09,-0.26,-0.31,-0.18,-0.11,-0.23,-0.44,-0.31,-0.48,-0.42,-0.59,-0.34,-0.23,-0.59,-0.3,-0.49,-0.38,-0.32,-0.44,-0.36,-0.45,-0.31,-0.25,-0.45,-0.59,-0.48,-0.49,-0.49,-0.5,-0.06,-0.21,0.17,-0.26,0.1,-0.05,-0.07,-0.36,-0.18,0.28,0.15,0.23,0.1,-0.22,-0.28,0.03,-0.14,-0.13,-0.03,0.02,-0.03,0.16,-0.08,0.01,-0.23,0.26,0.24,-0.15,0.36,0.39,0.17,0.23,0.04,0.06,0.19,-0.2,-0.02,-0.08,0.06,-0.24,0.08,-0.09,-0.15,-0.07,-0.03,-0.21,-0.04,0.21,0.0,0.2,0.15,0.07,0.11,0.45,0.13,0.08,0.14,0.08,0.28,0.29,0.46,0.4,0.45,0.2,0.1,0.01,0.1,0.03,0.1,0.22,0.42,0.04,0.04,0.15,0.33,0.3,-0.13,-0.03,-0.03,-0.06,-0.04,-0.19,0.13,0.21,-0.27,0.28,0.0,-0.05,0.1,0.09,0.18,-0.13,-0.18,-0.01,0.1,-0.12,0.01,0.0,-0.15,-0.05,-0.12,-0.01,-0.33,-0.03,0.16,-0.3,-0.09,-0.25,-0.45,-0.21,-0.06,0.01,-0.14,-0.33,-0.22,-0.06,0.48,-0.28,0.05,-0.12,-0.19,0.0,0.14,-0.03,-0.3,-0.17,-0.38,-0.14,-0.27,-0.11,-0.07,-0.03,-0.37,-0.12,-0.11,0.0,-0.08,-0.35,-0.33,-0.21,-0.15,-0.43,-0.31,0.09,-0.53,-0.28,-0.28,-0.36,-0.08,-0.18,-0.29,-0.21,-0.13,-0.55,0.01,-0.43,-0.36,-0.22,-0.61,-0.34,-0.32,-0.41,-0.31,-0.12,-0.24,-0.27,-0.07,-0.34,-0.23,-0.39,-0.31,-0.08,-0.29,-0.03,-0.19,-0.21,-0.1,-0.15,-0.19,-0.03,-0.27,-0.18,0.07,-0.15,-0.07,-0.1,-0.31,-0.22,-0.29,-0.48,-0.54,-0.53,-0.58,-0.5,-0.59,-0.4,-0.39,-0.44,-0.39,-0.4,-0.32,-0.4,-1.15,-0.55,-0.22,-0.66,-0.99,-0.87,-0.79,-1.06,-0.93,-0.5,-0.49,-0.55,-0.78,-2.32,-0.41,0.02,-0.52,-0.06,-0.52,-0.79,-0.65,-0.58,-0.59,-0.73,-0.79,0.25,-0.83,-0.5,-0.66,-0.64,-0.61,-0.38,-0.5,-0.36,0.03,0.1,-0.17,-0.31,0.02,0.03,-0.17,-0.26,-0.15,-0.2,-0.38,-0.67,-0.58,-0.59,-0.68,-0.25,-0.64,-0.94,-1.05,-1.31,-0.79,-0.31,-0.61,-0.34,-0.4,-0.04,-0.14,0.01,0.09,-0.01,-0.04,0.15,0.27,0.02,-0.01,0.02,0.1,-0.26,-0.09,0.27,0.05,-0.1,0.01,-0.23,0.02,0.02,0.03,-0.01,-0.22,0.06,-0.03,-0.34,-0.01,-0.16,-0.05,0.01,-0.2,-0.19,-0.44,-0.21,-0.37,-0.05,-0.44,-0.18,-0.15,-0.49,-0.26,-0.13,-0.37,-0.18,-0.41,-0.36,-0.13,-0.47,-0.25,0.2,-0.12,-0.23,-0.37,-0.51,-0.29,-0.12,-0.25,-0.16,-0.25,-0.26,-0.34,-0.26,-0.43,-0.38,-0.3,-0.26,0.01,-0.21,-0.26,-0.27,-0.07,-0.02,-0.31,-0.14,-0.19,0.04,-0.37,-0.13,-0.48,-0.39,-0.08,-0.26,-0.51,-0.34,-0.2,-0.32,-0.28,-0.54,0.13,-0.09,-0.3,-0.15,-0.13,-0.05,-0.1,-0.24,-0.44,-0.23,-0.45,-0.56,-0.19,-0.74,-0.51,-0.61,-0.53,-0.27,-0.16,-0.38,-0.01,-0.22,-0.2,0.0,0.03,-0.09,0.06,0.1,-0.11,0.17,-0.1,-0.05,-0.42,0.04,-0.34,0.09,-0.05,0.25,-0.13,0.02,0.06,0.03,-0.29,-1.12,-0.4,-1.01,-0.94,-1.09,-1.03,-0.77,-0.56,-0.41,-0.69,-0.54,-0.48,-0.57,-0.39,-0.39,-0.43,-0.27,-0.06,-0.03,0.01,-0.22,-0.26,-0.03,-0.19,-0.21,-0.04,-0.17,-0.12,-0.03,-0.06,-0.08,-0.21,-0.17,-0.19,-0.24,-0.49,-0.25,-0.01,-0.15,0.14,-0.39,0.14,-0.24,0.12,0.04,0.26,-0.15,-0.01,0.08,-0.06,-0.07,0.09,-0.04,-0.2,-0.27,0.31,-0.22,-0.17,-0.34,0.02,-0.1,-0.27,-0.13,-0.09,0.03,-0.13,-0.12,0.09,-0.19,-0.34,0.13,0.09,0.1,-0.03,0.3,0.13,0.09,0.15,0.27,0.26,0.33,0.23,0.23,0.28,0.38,0.59,0.31,0.45,0.64,0.13,0.19,0.33,0.33,0.07,0.06,0.16,0.11,-0.05,-0.02,0.09,0.1,0.11,0.46,-0.04,-0.07,-0.1,-0.2,0.01,-0.27,-0.05,-0.21,-0.22,0.04,0.31,0.34,-0.03,0.27,0.31,0.46,0.33,0.22,0.09,0.24,0.04,0.18,0.29,0.06,0.03,-0.02,-0.26,0.09,-0.07,0.2,0.13,0.35,0.02,0.1,0.17,0.1,0.14,-0.02,0.22,-0.17,0.02,0.11,0.07,0.14,-0.11,0.04,-0.01,0.18,0.52,-0.01,0.07,-0.07,0.29,-0.03,0.1,0.21,0.11,0.03,-0.06,0.06,0.33,0.21,-0.12,-0.01,0.08,-0.31,0.16,-0.03,-0.15,-0.09,-0.07,0.37,-0.06,0.17,0.27,0.48,0.0,-0.02,0.31,-0.01,0.28,0.34,0.1,-0.02,-0.02,0.03,0.45,0.21,0.06,0.26,0.2,0.33,0.83,0.2,0.35,0.48,0.65,0.35,0.29,0.32,0.23,0.2,0.56,0.41,0.34,0.36,0.34,0.41,0.85,0.19,0.65,0.42,0.76,0.49,0.79,0.79,0.58,0.79,0.78,0.55,0.55,0.84,0.89,0.54,0.76,0.68,0.52,0.51,0.57,0.55,0.4,0.33,0.65,0.6,0.52,0.44,0.29,0.39,0.75,0.53,0.51,0.45,0.54,0.63,0.43,0.39,0.6,0.41,0.72,0.63,0.43,0.29,0.34,0.58,0.7,0.16,0.21,0.28,0.43,0.58,0.1,0.52,0.47,0.36,0.34,0.56,0.64,0.54,0.28,0.48,0.39,0.21,0.25,0.2,0.2,0.2,0.06,0.21,0.06,0.25,0.13,0.38,0.06,0.76,0.38,0.56,0.15,0.02,0.37,0.26,0.18,0.31,0.21,0.2,0.06,-0.03,0.21,0.29,0.3,0.36,0.22,0.21,0.24,0.53,0.38,0.33,0.35,0.21,0.58,0.56,0.13,0.25,0.17,0.42,0.25,0.19,0.4,0.18,0.18,0.7,0.52,0.22,0.22,0.35,0.2,0.17,0.46,0.67,0.05,0.3,0.62,0.34,0.09,0.61,0.24,0.46,0.41,0.17,0.47,0.65,0.39,0.33,0.3,0.49,0.45,0.64,0.5,0.55,0.7,0.63,0.6,0.73,0.9,0.71,0.53,0.22,0.73,0.4,0.49,0.58,0.37,0.4,0.69,0.33,0.43,0.11,0.49,0.3,0.24,-0.13,-0.24,-0.18,-0.17,-0.16,-0.14,-0.2,-0.33,-0.12,-0.14,-0.38,-0.01,-0.1,-0.41,0.0,-0.22,-0.23,0.05,0.12,-0.38,-0.25,-0.38,-0.51,-0.19,-0.31,-0.3,-0.48,-0.3,-0.05,-0.37,-0.53,-0.33,-0.13,-0.01,0.04,-0.15,-0.08,-0.08,-0.2,-0.33,0.21,-0.04,-0.07,-0.08,-0.42,-0.16,-0.49,-0.27,-0.41,-0.45,-0.69,-0.19,-0.36,-0.27,-0.37,-0.4,-0.42,-0.04,-0.47,-0.23,-0.45,-0.29,-0.16,-0.28,-0.09,-0.18,-0.04,-0.55,-0.18,-0.24,-0.36,-0.55,-0.56,-0.22,-0.17,-0.33,-0.2,0.18,-0.3,-0.33,-0.51,-0.61,-0.36,-0.17,-0.48,-0.4,-0.33,-0.58,-0.14,-0.26,0.02,-0.22,-0.49,-0.47,-0.3,-0.28,-0.39,-0.34,-0.38,-0.31,-0.24,-0.37,-0.33,0.0,-0.15,-0.25,-0.1,-0.49,-0.29,-0.51,-0.53,-0.48,-0.62,-0.25,-0.41,-0.2,-0.32,-0.47,-0.12,-0.57,0.01,-0.4,-0.36,-0.04,-0.24,-0.34,0.24,-0.14,-0.08,-0.12,-0.2,-0.29,-0.24,-0.24,-0.18,0.0,-0.02,-0.15,-0.17,-0.28,0.01,-0.15,-0.1,-0.27,-0.17,-0.28,-0.48,-0.25,0.0,-0.17,-0.01,-0.22,-0.15,-0.44,-0.28,0.03,-0.35,-0.21,-0.53,-0.25,-0.3,-0.12,-0.25,-0.29,-0.21,-0.55,-0.26,-0.31,-0.23,-0.33,-0.46,-0.38,0.03,-0.07,-0.29,-0.29,-0.33,-0.27,-0.15,-0.27,-0.35,-0.13,-0.27,0.08,-0.35,-0.27,0.16,0.04,-0.26,0.0,-0.13,0.0,-0.16,-0.09,0.33,0.17,-0.18,0.08,0.31,-0.1,0.02,0.09,0.34,0.36,0.33,0.16,0.24,0.23,0.37,0.44,0.37,0.19,0.37,0.02,0.39,0.34,0.45,0.34,0.55,0.33,0.2,0.12,0.22,0.77,0.24,0.63,0.28,0.29,0.33,0.4,0.56,0.24,0.59,0.22,0.25,0.21,0.1,0.3,0.15,0.46,0.22,0.34,0.34,0.65,0.32,0.46,-0.06,-0.07,-0.11,0.19,0.5,0.15,0.14,0.22,0.22,0.0,0.36,0.02,0.12,0.1,0.23,0.08,-0.2,-0.15,0.08,0.06,-0.13,-0.01,-0.14,-0.3,-0.33,-0.23,-0.39,-0.31,-0.32,-0.06,-0.03,-0.23,-0.26,-0.16,-0.22,-0.26,-0.25,-0.28,-0.44,-0.17,-0.36,-0.03,0.23,0.08,0.12,0.06,-0.04,0.03,0.12,-0.24,-0.15,0.11,0.06,0.32,0.15,0.01,-0.14,-0.15,0.07,-0.08,0.16,-0.21,0.16,-0.18,-0.15,0.08,-0.06,-0.12,0.02,0.22,0.18,0.02,0.01,-0.09,-0.07,-0.04,0.08,0.12,0.28,0.19,0.17,0.35,0.26,0.32,0.22,0.28,0.27,0.04,0.07,0.25,0.16,-0.19,-0.12,0.53,-0.16,-0.39,-0.26,-0.1,-0.04,-0.21,-0.01,-0.07,-0.33,-0.14,-0.18,0.2,-0.17,-0.07,-0.14,0.04,-0.2,-0.16,-0.01,0.09,0.01,0.16,-0.21,0.46,-0.06,0.33,0.13,-0.17,-0.1,-0.19,0.06,0.19,-0.32,-0.32,-0.16,-0.2,-0.29,-0.39,-0.26,-0.17,-0.32,-0.44,-0.43,-0.23,-0.28,-0.41,-0.16,0.04,-0.16,-0.37,-0.3,-0.5,-0.35,-0.33,-0.13,-0.39,-0.19,-0.39,-0.52,-0.28,-0.19,-0.23,-0.04,-0.3,-0.09,0.12,0.13,-0.12,-0.08,-0.25,-0.14,-0.32,-0.13,0.08,-0.12,-0.17,-0.1,-0.36,0.01,-0.17,-0.25,-0.24,-0.22,-0.19,0.06,-0.09,-0.34,-0.14,0.06,-0.34,-0.16,0.04,-0.36,-0.46,-0.66,-0.42,-0.3,-0.33,-0.4,-0.47,-0.53,-0.44,-0.33,-0.52,-0.46,-0.56,-0.62,-0.49,-0.47,-0.22,-0.47,-0.47,-0.44,-0.35,-0.27,0.35,-0.21,-0.27,-0.05,-0.23,0.19,-0.32,-0.19,-0.29,-0.46,-0.42,-0.3,0.08,-0.33,-0.49,-0.45,-0.07,-0.35,-0.14,-0.09,-0.44,-0.33,-0.21,-0.28,0.0,-0.43,-0.3,-0.36,-0.02,-0.42,-0.29,-0.32,0.08,-0.18,-0.07,-0.21,-0.06,-0.38,0.0,-0.21,0.1,-0.25,0.1,-0.16,-0.02,0.06,-0.08,-0.22,-0.31,0.06,-0.07,-0.23,-0.03,0.17,-0.03,0.08,0.04,0.51,0.12,0.1,0.31,0.32,0.27,0.2,0.56,0.13,0.34,0.41,0.48,0.47,0.34,0.46,0.83,0.57,0.54,0.36,0.97,0.68,0.37,0.72,0.3,0.14,0.4,0.53,0.29,0.43,0.51,0.43,0.31,0.57,0.25,0.15,0.3,0.28,0.34,0.25,0.23,0.24,0.34,0.07,0.62,0.53,0.38,0.17,0.41,0.31,0.25,0.3,0.44,0.35,0.37,0.32,0.13,0.4,0.17,0.38,0.16,-0.06,0.15,0.64,0.17,0.03,0.12,0.28,0.34,0.15,0.14,0.25,0.37,-0.2,-0.18,0.13,0.41,0.22,0.19,0.22,0.1,0.0,0.27,0.37,0.0,0.17,0.77,0.17,0.17,0.02,0.25,0.36,0.0,0.18,0.46,0.49,0.15,0.47,0.21,0.49,0.54,0.49,0.65,0.43,0.42,0.34,0.46,0.22,0.36,0.63,0.49,0.27,0.09,0.59,0.14,0.52,0.3,0.39,0.08,0.42,0.39,0.68,0.37,0.43,0.52,0.42,0.52,0.37,0.74,0.58,0.48,0.47,0.97,0.28,0.47,0.58,0.46,0.68,0.78,0.43,0.4,0.44,0.48,0.54,0.18,0.37,0.47,0.56,0.45,0.61,0.26,0.34,0.35,0.18,0.41,0.19,0.1,0.17,0.54,0.51,0.19,0.17,0.08,0.07,-0.14,0.08,-0.16,-0.03,-0.15,0.39,0.31,-0.15,0.23,-0.03,0.04,0.04,0.02,-0.05,0.24,0.13,0.27,0.27,-0.04,-0.21,0.15,-0.14,0.16,-0.27,-0.03,0.16,0.13,-0.06,0.08,0.2,0.29,0.02,0.2,-0.03,0.09,-0.14,-0.12,0.02,-0.03,-0.2,-0.01,-0.12,-0.11,-0.16,-0.1,-0.49,-0.37,-0.34,-0.65,-0.37,-0.4,-0.41,-0.66,-0.25,-0.18,-0.58,-0.14,-0.41,-0.2,-0.35,-0.08,-0.09,-0.3,-0.32,-0.22,-0.26,-0.36,-0.15,0.07,-0.25,-0.18,-0.14,0.12,0.63,0.08,0.21,0.0,0.47,0.31,0.49,0.14,0.28,0.23,0.36,0.37,0.38,0.39,0.36,0.45,0.6,-0.05,0.37,0.27,0.39,0.7,0.45,0.37,0.35,0.35,0.39,0.06,0.4,0.13,0.39,-0.08,0.14,0.08,0.23,0.37,0.47,0.02,0.16,0.24,0.26,0.53,0.66,0.18,0.4,0.66,0.23,0.1,0.25,0.19,0.43,0.27,0.27,0.21,0.55,0.21,0.22,0.09,0.16,0.05,0.15,0.3,0.27,0.39,0.19,0.31,0.21,0.26,0.46,0.3,0.28,0.6,0.14,0.34,0.4,0.49,0.63,0.4,0.41,0.52,0.36,0.69,0.62,0.21,0.7,0.48,0.45,0.42,0.66,0.5,0.11,0.02,0.51,0.22,0.68,0.17,0.35,0.44,0.28,0.45,0.05,0.31,0.37,0.36,0.11,0.46,0.43,0.03,0.55,0.02,0.32,0.21,-0.06,0.1,0.45,0.15,0.25,0.55,0.27,0.21,0.27,0.08,0.49,0.07,0.44,0.22,0.2,0.1,0.48,0.24,0.17,0.19,0.21,0.26,0.05,0.19,-0.03,0.49,0.28,0.38,0.42,0.08,0.19,0.31,0.43,0.18,0.22,0.49,0.22,0.48,0.44,0.67,0.3,0.14,0.29,0.43,0.85,0.46,0.16,0.64,0.66,0.32,0.53,0.54,0.21,0.17,0.18,0.21,0.1,0.17,0.28,-0.13,0.22,0.04,0.06,0.1,0.5,-0.09,0.07,0.21,-0.1,-0.31,-0.03,0.23,0.16,-0.01,-0.11,0.12,0.53,0.23,0.43,0.2,0.27,0.51,0.18,0.32,0.47,0.41,0.31,0.62,0.55,0.38,0.39,0.59,0.1,0.15,-0.04,0.01,0.11,0.07,0.15,0.27,0.36,0.4,-0.01,-0.03,0.03,0.12,-0.33,-0.08,-0.25,-0.18,0.01,-0.03,-0.03,-0.21,-0.09,-0.16,0.01,-0.05,0.27,-0.16,0.42,0.13,-0.03,0.08,0.33,0.06,0.15,-0.01,0.33,0.25,0.5,0.06,0.08,0.09,0.19,0.26,0.27,0.1,0.12,-0.23,0.0,-0.16,-0.12,-0.22,-0.14,0.08,0.19,-0.32,-0.25,-0.1,-0.18,0.1,-0.03,0.05,-0.15,-0.16,-0.21,-0.04,-0.16,0.1,0.3,-0.28,-0.21,-0.03,-0.19,-0.19,-0.02,0.09,-0.09,-0.34,-0.32,-0.13,-0.15,-0.39,0.04,-0.04,-0.04,0.1,0.18,-0.15,0.06,0.08,0.23,-0.09,-0.07,0.06,0.39,0.04,0.4,0.22,0.19,0.14,0.1,0.49,0.37,0.71,0.25,0.17,0.18,0.31,0.31,0.44,0.29,0.23,0.02,0.08,0.44,0.23,0.09,0.25,0.58,0.32,0.4,0.19,0.4,0.19,0.25,0.24,0.33,0.23,0.51,0.13,0.35,0.26,0.45,0.28,0.3,0.07,0.29,0.43,0.72,0.38,0.69,0.26,0.43,0.19,0.21,0.44,0.31,0.22,0.29,0.19,0.27,0.35,0.48,0.2,0.15,0.45,0.19,0.33,0.13,0.24,0.29,0.08,0.12,0.01,0.0,0.01,0.15,0.13,0.03,0.29,-0.08,-0.28,-0.25,-0.15,-0.27,-0.33,-0.26,-0.16,-0.31,0.01,-0.46,-0.33,-0.19,-0.03,-0.04,-0.36,-0.01,-0.02,-0.13,0.01,0.09,-0.24,0.11,-0.09,-0.04,-0.04,-0.1,-0.07,-0.04,-0.03,-0.01,0.12,-0.24,-0.12,0.13,-0.08,0.23,0.4,0.08,0.21,0.13,-0.03,0.12,0.03,-0.24,-0.16,0.1,-0.01,-0.15,0.06,0.02,-0.06,-0.09,-0.37,-0.13,-0.4,-0.17,-0.2,-0.19,-0.32,-0.36,-0.34,-0.33,-0.42,-0.35,0.18,-0.28,-0.46,-0.56,-0.21,-0.24,-0.51,-0.57,-0.17,-0.2,0.13,-0.37,-0.14,-0.26,-0.34,-0.38,-0.16,-0.22,-0.4,-0.23,0.01,-0.46,-0.18,-0.34,-0.31,-0.45,-0.26,-0.47,-0.21,-0.14,-0.41,-0.51,-0.37,-0.54,-0.26,-0.31,-0.17,-0.56,-0.32,-0.39,-0.14,-0.32,-0.17,-0.46,0.16,-0.47,-0.2,-0.31,-0.43,-0.16,-0.19,-0.01,-0.47,-0.07,-0.27,-0.62,-0.18,-0.03,-0.07,-0.03,-0.18,-0.2,-0.29,-0.42,-0.3,-0.19,-0.35,-0.35,-0.46,-0.28,-0.44,0.03,-0.07,-0.15,-0.44,-0.28,-0.31,0.01,-0.42,-0.32,0.02,-0.06,-0.18,0.27,0.09,-0.14,-0.38,-0.25,-0.15,-0.07,-0.08,-0.18,-0.37,-0.45,-0.2,-0.24,-0.41,-0.23,-0.27,-0.18,0.17,-0.38,-0.49,-0.28,-0.08,-0.32,-0.21,-0.27,0.04,-0.15,-0.29,0.04,0.06,-0.06,-0.12,-0.22,0.38,0.21,0.08,0.15,0.06,0.06,-0.11,0.06,-0.13,-0.26,0.1,-0.13,-0.17,-0.22,-0.38,-0.22,-0.5,-0.16,-0.23,-0.56,-0.21,-0.22,-0.3,0.11,-0.16,-0.22,-0.45,-0.19,-0.34,0.11,-0.16,-0.18,-0.12,-0.19,0.06,-0.13,-0.31,0.08,-0.18,-0.07,-0.08,0.2,0.08,-0.3,0.06,0.1,0.12,0.04,0.0,-0.06,-0.1,0.2,0.31,0.1,0.2,0.19,-0.04,0.09,0.26,0.39,0.41,0.41,0.24,0.15,0.28,0.28,0.93,0.42,0.65,0.66,0.61,0.98,0.52,0.47,0.37,0.56,0.26,0.19,0.21,0.34,0.19,0.41,0.15,0.06,0.37,0.24,0.11,0.16,0.19,0.28,-0.07,0.09,0.09,0.12,-0.14,0.12,-0.07,-0.27,-0.12,-0.2,0.02,0.35,0.03,0.0,-0.15,-0.07,0.1,-0.07,-0.29,-0.24,0.1,-0.03,-0.03,-0.14,-0.08,0.15,-0.03,0.44,-0.01,0.03,0.12,0.08,0.26,0.21,0.26,0.5,0.28,0.33,0.16,0.28,0.42,0.33,0.21,-0.07,-0.03,0.02,0.06,-0.17,0.27,0.12,0.16,-0.13,0.01,-0.15,-0.13,-0.2,0.31,-0.23,0.03,-0.2,0.15,-0.25,-0.12,0.05,0.41,-0.01,0.21,0.33,0.08,0.1,0.2,0.5,0.17,0.16,0.03,0.22,0.13,0.13,0.54,0.3,0.2,0.62,0.62,0.1,0.47,0.45,0.41,0.37,0.16,0.68,0.19,0.41,0.0,0.55,0.09,0.14,0.68,0.25,0.3,0.59,0.27,0.33,0.23,0.17,0.47,0.55,0.47,0.41,0.31,0.45,0.27,0.51,0.41,0.34,0.35,0.24,0.54,0.29,0.21,0.59,0.52,0.4,0.45,0.63,0.32,0.16,0.21,0.06,-0.01,0.14,0.28,0.61,0.36,0.31,0.13,0.12,0.07,0.14,-0.24,-0.16,-0.31,-0.2,-0.03,-0.12,-0.14,-0.05,-0.2,0.15,-0.12,0.04,-0.07,0.02,-0.04,-0.18,0.04,0.17,-0.01,0.04,-0.05,-0.1,-0.25,-0.03,-0.08,0.32,-0.03,0.02,-0.19,0.04,0.06,0.09,0.33,0.13,-0.05,0.12,0.15,0.02,0.16,-0.08,-0.02,-0.15,0.04,-0.16,-0.14,0.02,0.02,-0.03,-0.16,-0.29,-0.13,-0.26,-0.15,-0.33,-0.29,0.0,-0.21,-0.69,-0.57,-0.47,-0.39,-0.3,-0.51,-0.46,-0.33,-0.26,-0.17,-0.4,-0.04,-0.16,-0.11,0.04,-0.12,0.17,-0.09,-0.07,-0.15,-0.13,0.11,0.06,0.08,0.16,-0.23,-0.31,0.16,-0.01,-0.32,-0.35,-0.3,-0.27,0.25,-0.04,-0.35,-0.52,0.02,-0.08,-0.29,-0.32,-0.21,-0.22,-0.29,-0.16,-0.14,-0.38,0.03,-0.38,-0.3,-0.29,-0.24,-0.37,-0.18,-0.43,-0.13,-0.13,-0.25,-0.24,-0.5,-0.26,-0.18,0.11,-0.29,-0.5,-0.28,-0.5,-0.36,-0.29,-0.41,-0.11,-0.38,-0.11,-0.29,-0.3,-0.15,-0.15,-0.01,-0.05,-0.24,-0.23,-0.19,-0.28,0.15,-0.34,-0.45,-0.38,-0.63,-0.44,-0.73,-0.61,-0.52,-0.42,-0.29,-0.18,0.09,-0.39,-0.06,-0.3,-0.08,-0.42,0.01,-0.08,-0.16,-0.14,0.07,0.02,-0.32,0.01,-0.11,-0.01,-0.22,-0.99,0.18,0.01,-0.06,0.46,0.0,-0.3,-0.11,0.14,0.15,-0.18,-0.15,-0.08,-0.03,-0.3,-0.01,-0.27,-0.16,-0.17,-0.23,-0.14,0.23,-0.22,-0.35,-0.3,0.11,-0.35,-0.23,-0.46,-0.53,-0.35,0.01,0.3,-0.29,-0.24,0.17,-0.4,-0.06,-0.36,-0.19,0.02,0.06,-0.39,-0.06,-0.05,-0.18,-0.23,0.13,0.02,-0.36,-0.13,0.13,-0.22,-0.31,-0.26,-0.16,-0.42,-0.05,-0.33,-0.06,-0.13,-0.37,-0.3,0.05,-0.26,-0.21,-0.17,-0.22,-0.36,0.09,-0.07,-0.2,-0.44,0.06,-0.33,0.04,-0.27,-0.17,-0.21,-0.16,-0.15,-0.03,-0.21,0.26,-0.04,-0.26,-0.29,-0.26,-0.33,0.08,0.06,0.06,0.02,0.01,0.16,0.2,-0.1,0.26,0.43,0.03,0.1,0.12,0.06,0.33,0.19,0.12,0.33,0.05,0.12,0.26,-0.05,0.13,0.38,-0.14,0.33,-0.19,-0.08,0.03,0.06,-0.14,-0.07,0.31,-0.09,-0.27,-0.14,0.06,-0.14,-0.09,-0.06,-0.03,-0.14,0.22,-0.15,0.11,0.09,-0.04,0.23,0.39,0.04,-0.06,-0.07,-0.03,-0.23,0.28,-0.08,0.06,0.52,0.05,0.38,0.21,0.37,0.22,-0.52,0.28,0.49,0.24,0.43,0.28,0.23,0.43,0.7,0.59,0.36,0.46,0.4,0.42,0.5,0.35,0.17,0.69,0.45,0.58,0.58,0.27,0.45,0.61,0.6,0.78,0.35,0.31,0.18,0.38,0.28,0.12,0.64,0.36,0.13,0.44,0.14,0.5,0.29,0.08,0.57,0.29,0.49,0.37,0.53,0.09,0.38,0.24,0.49,0.34,0.35,0.12,0.06,0.12,0.21,0.04,0.2,0.46,0.06,0.29,0.18,-0.04,0.16,0.32,0.33,0.19,0.04,0.26,-0.05,0.31,0.06,0.04,-0.04,0.27,-0.08,0.34,-0.08,-0.06,0.01,-0.01,0.25,0.16,0.3,0.1,0.38,0.24,0.23,0.22,0.37,0.4,0.06,0.23,0.42,0.18,0.6,0.24,0.55,0.26,0.39,0.44,0.12,0.23,0.21,0.25,0.03,0.2,0.34,0.12,0.27,0.24,-0.1,0.34,0.41,0.22,0.47,0.25,0.41,0.14,0.24,-0.1,0.21,0.34,-0.13,0.29,-0.11,-0.25,-0.29,-0.08,-0.19,-0.03,-0.11,-0.19,0.06,-0.28,-0.3,-0.25,-0.34,0.02,-0.12,-0.09,-0.08,-0.23,-0.37,-0.19,-0.15,-0.33,0.02,-0.23,-0.04,-0.14,0.04,-0.16,-0.11,0.15,-0.17,0.62,-0.04,-0.11,-0.11,0.23,-0.13,-0.07,-0.13,0.12,0.09,0.15,0.2,-0.23,-0.05,-0.23,-0.07,-0.03,0.08,-0.19,0.09,0.13,0.28,0.17,0.16,0.24,0.02,0.33,-0.16,0.14,-0.08,-0.07,0.24,-0.01,0.24,0.06,0.12,0.24,0.16,-0.2,0.23,0.29,-0.03,-0.36,0.07,-0.28,0.2,0.18,-0.1,0.09,-0.12,-0.13,-0.13,-0.03,-0.03,-0.05,0.04,-0.03,-0.29,0.05,-0.31,-0.2,0.04,-0.22,-0.36,-0.35,-0.26,-0.19,-0.25,-0.24,-0.26,-0.06,-0.16,-0.09,0.15,-0.32,-0.41,-0.16,0.06,-0.02,0.07,-0.24,-0.14,0.09,0.0,0.1,0.32,0.29,0.34,0.17,0.29,0.38,0.42,0.26,0.23,0.39,0.09,0.02,0.24,0.39,0.14,-0.02,0.25,-0.12,-0.18,0.0,0.38,0.04,-0.11,-0.13,-0.28,-0.16,-0.12,0.23,-0.2,-0.35,-0.05,-0.16,0.46,0.22,0.16,-0.33,-0.12,-0.03,0.16,0.14,0.03,0.06,0.46,-0.02,0.22,-0.04,0.2,0.25,0.28,0.31,0.44,-0.29,0.04,-0.12,-0.06,-0.13,-0.23,-0.01,-0.28,-0.44,-0.22,-0.01,-0.35,-0.1,-0.14,-0.23,-0.06,-0.35,-0.29,-0.27,-0.36,-0.24,-0.33,-0.35,-0.22,0.01,-0.36,-0.32,-0.17,-0.44,-0.27,-0.24,0.09,-0.58,-0.37,-0.33,-0.62,-0.06,-0.39,-0.52,-0.53,-0.12,-0.19,-0.11,-0.13,-0.39,0.0,-0.31,-0.1,-0.34,-0.03,-0.36,-0.11,-0.32,-0.24,-0.12,-0.1,-0.05,-0.12,-0.26,-0.29,-0.04,-0.28,-0.38,-0.36,-0.26,-0.16,-0.38,-0.33,-0.08,-0.12,-0.07,-0.01,-0.08,-0.34,-0.37,-0.14,-0.2,0.03,0.1,-0.22,-0.27,-0.11,-0.05,-0.27,-0.24,-0.33,-0.45,-0.17,-0.35,-0.57,-0.14,0.0,-0.31,-0.43,-0.37,-0.52,-0.35,-0.09,-0.41,-0.4,-0.17,-0.27,-0.24,-0.32,-0.36,0.08,-0.25,-0.32,-0.33,-0.24,-0.42,-0.35,-0.08,-0.57,-0.04,-0.24,-0.34,-0.14,-0.12,-0.34,-0.21,-0.16,0.1,-0.25,0.06,-0.05,0.1,0.01,-0.2,-0.21,0.1,-0.15,0.0,-0.27,-0.19,-0.33,-0.33,-0.36,-0.2,0.13,-0.08,-0.12,-0.18,0.1,0.07,0.02,-0.06,0.15,-0.01,0.32,-0.03,0.4,0.16,0.1,-0.08,0.08,0.01,0.2,-0.22,0.03,-0.01,-0.04,0.25,0.13,-0.18,-0.08,-0.18,-0.19,-0.1,0.16,0.1,-0.25,-0.14,-0.04,0.07,0.12,-0.14,0.12,-0.21,-0.17,-0.07,-0.19,-0.23,-0.04,0.02,0.06,-0.01,-0.11,0.05,-0.08,-0.21,0.18,0.01,-0.13,-0.02,-0.06,-0.1,-0.08,-0.18,0.13,-0.15,-0.13,-0.2,-0.22,0.09,-0.13,0.04,-0.04,-0.11,-0.31,-0.19,-0.42,-0.05,-0.17,-0.5,-0.1,-0.38,-0.27,0.16,0.07,0.12,-0.22,-0.31,-0.03,0.1,-0.06,-0.42,0.32,-0.25,-0.29,-0.26,-0.27,-0.2,-0.25,-0.15,-0.15,0.17,-0.29,-0.45,-0.58,-0.1,-0.27,-0.25,-0.11,-0.13,0.0,-0.38,-0.28,-0.16,-0.11,-0.33,-0.08,-0.39,-0.17,-0.42,-0.24,-0.22,0.0,-0.52,0.15,-0.28,-0.15,-0.12,-0.04,-0.15,-0.07,-0.06,-0.25,-0.06,-0.03,-0.09,-0.06,-0.36,0.06,-0.05,-0.29,-0.12,-0.35,-0.47,-0.33,-0.06,-0.18,-0.2,0.01,0.07,0.01,-0.07,-0.35,-0.48,-0.25,-0.04,0.17,-0.09,-0.24,0.04,0.01,0.15,0.1,0.23,-0.16,-0.24,0.0,-0.22,0.14,0.17,-0.03,-0.25,0.21,-0.14,-0.01,-0.18,0.24,0.01,0.13,-0.19,0.29,0.34,-0.4,-0.38,0.04,0.05,-0.12,-0.22,-0.09,-0.19,0.09,0.11,-0.07,-0.05,0.21,-0.15,-0.27,-0.11,-0.06,-0.13,-0.14,-0.28,-0.23,-0.12,-0.06,-0.21,-0.05,-0.04,-0.34,0.07,0.04,0.06,-0.34,-0.35,0.12,-0.24,-0.16,-0.17,-0.01,-0.15,-0.34,-0.48,-0.32,-0.41,-0.41,-0.25,0.08,0.23,-0.06,-0.2,0.12,-0.3,-0.3,-0.21,-0.2,-0.2,-0.22,-0.18,0.0,0.11,-0.25,0.07,-0.2,-0.29,-0.22,-0.29,-0.2,0.09,0.01,-0.24,0.15,-0.01,0.23,0.24,0.13,0.1,0.01,0.01,0.24,0.03,0.12,0.21,-0.16,0.15,0.32,0.49,0.29,0.34,0.46,0.17,0.13,0.03,0.18,0.25,0.25,0.14,0.28,0.72,0.35,0.12,0.49,0.16,0.31,0.73,0.17,0.18,0.24,0.2,0.2,0.04,0.44,0.16,-0.15,0.09,-0.03,0.39,0.01,0.27,0.0,0.1,0.04,0.08,0.26,0.06,-0.08,0.23,-0.01,0.3,0.1,0.37,0.01,-0.11,0.06,0.22,0.25,0.19,-0.01,0.41,0.12,0.21,0.36,0.43,0.43,0.42,0.15,0.14,0.17,0.3,0.51,0.47,0.11,0.26,0.15,0.14,0.14,0.29,0.39,0.27,0.13,0.34,0.03,-0.03,-0.19,0.04,0.28,0.12,-0.05,-0.08,0.24,-0.22,0.03,0.01,0.03,-0.1,0.08,0.04,-0.03,0.21,0.31,0.03,0.0,-0.01,-0.18,0.24,-0.25,-0.04,-0.46,-0.47,-0.33,-0.2,-0.27,-0.33,-0.32,-0.31,-0.19,-0.25,-0.66,-0.47,-0.37,-0.55,-0.48,-0.49,-0.25,-0.37,-0.69,-0.37,-0.6,-0.5,-0.41,-0.31,-0.31,-0.53,-0.41,-0.12,-0.07,-0.16,-0.06,-0.3,-0.42,-0.11,-0.3,-0.89,-0.18,-0.1,-0.38,-0.36,-0.29,-0.13,-0.37,0.17,0.0,-0.05,-0.07,-0.26,-0.29,-0.25,0.04,-0.15,-0.02,-0.13,0.03,0.06,-0.21,-0.14,-0.12,-0.08,-0.12,-0.23,-0.3,-0.24,-0.14,-0.07,-0.14,-0.2,-0.08,-0.07,-0.32,-0.41,-0.08,-0.13,0.08,-0.19,0.19,-0.07,0.24,-0.25,-0.21,-0.12,-0.21,-0.04,-0.36,-0.36,-0.28,-0.03,-0.21,-0.3,-0.13,-0.15,-0.24,-0.05,-0.12,-0.26,0.06,-0.01,0.03,0.31,-0.09,0.08,0.19,-0.25,-0.27,-0.29,-0.24,-0.15,-0.17,-0.31,-0.61,-0.59,-0.6,-0.1,-0.27,-0.23,-0.29,-0.32,-0.4,-0.04,-0.27,-0.01,0.09,0.03,0.23,0.15,0.0,0.25,0.22,0.12,0.06,0.56,0.03,0.01,-0.08,-0.12,0.09,0.19,-0.01,-0.07,0.06,0.01,-0.18,0.07,-0.13,-0.2,-0.11,-0.22,0.02,-0.03,0.0,-0.15,-0.03,-0.13,-0.24,-0.32,-0.24,-0.23,-0.17,-0.36,-0.09,-0.23,-0.38,-0.25,-0.38,-0.08,-0.05,-0.25,-0.03,-0.32,-0.21,-0.17,-0.18,-0.12,-0.3,-0.09,-0.37,0.24,-0.27,-0.31,-0.3,0.0,-0.21,0.06,-0.21,-0.09,-0.18,0.05,-0.01,-0.07,-0.33,-0.1,0.15,0.06,-0.2,-0.34,-0.3,0.03,0.11,-0.25,-0.44,-0.45,0.07,-0.24,-0.49,0.06,-0.55,-0.17,-0.2,-0.24,-0.3,-0.44,0.25,0.04,-0.17,-0.28,-0.27,-0.23,-0.15,-0.19,-0.24,-0.13,-0.15,0.13,-0.23,-0.15,-0.22,-0.28,-0.27,-0.11,-0.08,-0.17,0.05,-0.32,-0.16,-0.31,-0.23,0.09,-0.11,-0.21,-0.35,-0.36,-0.12,0.02,-0.29,0.12,-0.03,-0.23,-0.18,-0.36,-0.11,-0.22,0.03,-0.19,-0.03,-0.11,-0.05,-0.13,-0.16,-0.01,-0.17,-0.33,-0.03,-0.05,-0.02,-0.07,-0.33,-0.24,-0.36,-0.13,-0.34,-0.52,0.02,-0.39,-0.38,-0.31,-0.02,-0.04,0.19,-0.52,-0.16,-0.17,-0.41,0.06,-0.15,-0.11,-0.16,-0.17,-0.06,-0.25,-0.4,-0.06,-0.17,-0.1,-0.14,-0.18,0.33,0.01,-0.08,0.12,-0.2,-0.03,-0.22,0.1,0.14,-0.09,0.0,-0.31,0.28,-0.19,-0.29,-0.14,0.08,-0.25,-0.23,0.21,-0.27,-0.15,-0.2,-0.23,-0.18,-0.15,-0.2,0.06,-0.18,-0.03,-0.45,-0.35,0.0,-0.2,0.09,0.06,-0.16,-0.21,-0.19,-0.35,-0.08,-0.29,-0.24,-0.15,0.27,-0.06,-0.07,0.15,-0.19,-0.27,0.14,-0.14,-0.32,-0.32,0.08,-0.07,0.01,0.03,-0.09,-0.01,-0.24,-0.3,0.09,0.14,-0.16,-0.18,0.1,-0.16,-0.09,-0.13,0.04,-0.14,-0.02,-0.16,-0.12,-0.26,-0.18,-0.04,0.06,0.08,-0.07,-0.01,-0.32,-0.17,-0.54,0.0,-0.19,-0.64,-0.23,-0.37,-0.58,-0.07,0.23,-0.02,-0.01,-0.18,-0.18,-0.16,-0.18,-0.13,0.04,-0.06,-0.12,0.0,-0.24,-0.01,-0.08,-0.24,-0.18,-0.07,-0.28,0.12,-0.08,-0.2,-0.05,0.19,0.12,0.0,-0.21,-0.14,-0.39,-0.13,-0.25,-0.29,-0.18,-0.02,-0.47,0.15,-0.22,-0.28,-0.42,-0.49,-0.28,0.06,-0.22,0.11,0.02,0.05,-0.23,-0.48,-0.21,-0.2,-0.25,0.07,0.12,0.14,-0.2,0.02,-0.27,0.06,-0.18,-0.02,0.09,0.34,-0.07,0.2,0.06,0.01,-0.35,-0.22,-0.04,0.0,0.09,-0.02,0.26,-0.12,0.03,-0.26,0.37,0.01,0.17,-0.27,0.36,0.33,-0.1,0.32,-0.06,0.63,0.06,0.23,0.36,-0.14,0.07,-0.06,-0.16,-0.14,-0.09,-0.12,-0.1,-0.03,0.1,-0.18,-0.11,-0.04,0.0,-0.2,0.33,-0.1,0.1,0.13,0.03,0.1,-0.08,-0.11,-0.03,0.01,0.26,-0.15,-0.04,0.02,0.19,-0.07,0.19,-0.11,-0.01,-0.08,-0.05,0.34,0.0,0.16,0.36,-0.18,0.43,0.03,0.32,0.47,0.1,0.12,0.1,0.45,0.06,0.12,0.1,-0.07,-0.13,0.26,0.07,0.19,0.13,0.15,-0.01,0.18,0.21,-0.09,-0.03,0.27,-0.03,-0.11,0.0,0.1,0.14,-0.15,-0.09,0.26,-0.18,-0.03,0.06,0.26,-0.03,0.04,0.16,0.09,-0.03,0.05,0.09,-0.05,-0.26,-0.01,0.04,-0.01,-0.25,-0.01,-0.24,0.02,0.1,-0.08,-0.38,-0.2,-0.1,-0.17,-0.31,-0.26,-0.26,0.12,0.3,0.09,0.07,-0.23,0.05,-0.03,-0.05,-0.04,0.14,-0.06,-0.3,0.25,0.18,-0.25,-0.37,-0.15,-0.27,-0.36,-0.14,-0.29,-0.32,-0.13,0.11,-0.6,-0.29,-0.19,-0.06,-0.46,-0.33,-0.32,-0.28,-0.31,-0.1,0.04,-0.09,-0.25,-0.23,-0.19,-0.23,-0.53,-0.25,-0.07,-0.11,0.04,-0.35,-0.19,-0.04,-0.06,-0.27,-0.23,0.33,-0.15,-0.11,0.16,0.27,-0.19,0.03,-0.21,-0.04,-0.4,-0.24,0.14,-0.35,-0.25,-0.31,-0.46,-0.29,-0.11,-0.17,-0.31,-0.06,-0.47,-0.17,0.08,-0.35,-0.19,-0.35,-0.19,0.1,-0.24,-0.17,-0.09,-0.2,-0.11,0.04,0.24,0.01,-0.04,-0.05,0.04,0.1,0.37,0.22,0.02,-0.01,0.28,0.1,0.1,0.09,0.36,0.37,0.13,0.16,0.2,0.03,0.06,0.36,0.38,0.23,0.21,0.08,0.14,0.22,0.42,0.04,0.06,0.02,0.1,-0.09,0.19,-0.05,-0.04,-0.06,0.21,-0.06,0.22,0.0,-0.11,0.23,-0.14,-0.03,0.17,0.09,0.33,-0.04,-0.08,-0.22,-0.2,-0.01,-0.28,-0.04,-0.03,-0.24,0.34,-0.18,-0.19,-0.24,0.16,0.06,0.15,0.06,-0.26,0.14,-0.08,-0.24,-0.18,0.26,0.16,0.03,0.21,0.26,0.23,0.07,0.13,0.53,0.28,0.38,0.07,0.01,0.28,0.27,0.25,0.42,0.06,0.33,0.24,0.16,0.23,0.18,0.39,-0.05,0.35,0.36,0.55,0.55,0.53,0.83,0.49,0.18,0.16,0.66,0.35,0.35,0.44,0.35,0.22,0.23,0.09,0.02,0.01,0.57,0.31,-0.1,0.41,0.02,-0.03,0.04,-0.03,0.2,-0.12,-0.08,0.07,-0.07,-0.09,-0.08,0.19,0.01,0.12,0.04,0.33,-0.04,0.48,0.18,0.23,0.39,0.25,0.18,0.27,0.28,0.25,0.19,0.37,0.1,0.06,0.41,0.03,0.28,0.31,0.23,0.17,0.23,0.41,0.44,0.27,0.3,0.43,0.18,0.38,0.38,0.22,0.45,0.4,0.79,0.41,0.31,0.38,0.32,0.16,0.43,0.28,0.32,0.29,0.42,0.31,0.31,0.04,0.62,0.41,0.32,0.71,0.51,0.55,0.5,0.46,0.63,0.34,0.34,0.22,0.36,0.28,0.24,-0.05,0.08,0.05,0.0,0.08,-0.12,-0.01,0.01,-0.37,-0.35,-0.36,-0.35,-0.41,-0.21,-0.26,-0.43,-0.32,-0.49,-0.33,-0.07,-0.4,-0.49,-0.25,-0.28,-0.39,-0.36,-0.29,-0.16,-0.02,-0.25,0.09,-0.03,-0.21,0.06,-0.06,-0.2,-0.23,-0.29,-0.2,-0.33,-0.2,-0.26,-0.16,-0.23,-0.08,-0.15,-0.4,-0.26,-0.24,-0.37,0.03,-0.37,0.14,-0.15,0.04,0.16,-0.14,0.03,-0.41,-0.57,-0.31,-0.3,-0.22,-0.28,-0.7,-0.13,-0.47,-0.41,-0.29,-0.34,-0.67,-0.44,-0.55,-0.63,-0.28,-0.21,-0.32,-0.48,-0.44,-0.11,-0.39,-0.03,-0.28,-0.28,-0.16,-0.01,-0.38,0.03,0.03,0.0,-0.23,0.2,-0.04,0.01,-0.23,-0.28,-0.15,-0.19,-0.37,-0.47,-0.24,-0.09,-0.38,-0.23,-0.25,-0.35,-0.42,-0.45,-0.23,-0.36,-0.16,-0.2,-0.1,-0.29,-0.26,-0.1,-0.34,0.06,-0.06,0.0,0.1,-0.09,-0.16,0.22,0.17,0.36,-0.31,-0.16,0.21,0.07,0.16,-0.03,0.02,0.07,-0.08,0.02,0.03,0.22,0.35,0.03,0.16,0.15,0.3,0.15,-0.05,0.14,0.14,0.26,0.06,-0.05,0.05,-0.23,-0.16,-0.2,-0.22,0.02,-0.11,0.13,-0.3,-0.24,-0.1,-0.28,-0.21,0.04,-0.32,-0.14,0.0,0.29,-0.28,0.11,0.23,-0.16,-0.05,0.13,0.09,0.22,0.16,0.35,0.39,0.74,0.46,0.48,0.6,0.63,0.6,0.6,0.77,0.79,0.59,0.55,0.41,0.52,0.78,0.66,0.48,0.37,0.42,0.41,0.43,0.24,0.27,0.25,0.19,0.35,0.34,0.27,0.19,0.1,0.29,0.39,0.2,0.65,0.21,0.27,0.37,0.56,0.18,0.5,0.57,0.55,0.14,0.04,0.1,-0.04,-0.04,0.46,0.14,-0.01,0.17,0.22,0.28,0.21,0.53,0.56,0.17,0.37,-0.02,0.11,0.14,0.12,0.78,-0.03,0.42,-0.16,0.13,0.24,0.49,0.29,0.65,0.49,0.16,0.18,0.09,0.27,-0.06,0.09,0.2,0.74,-0.07,0.06,0.41,-0.03,-0.14,-0.04,-0.12,-0.22,-0.01,-0.21,-0.25,-0.3,-0.01,-0.19,-0.14,-0.3,-0.26,-0.35,-0.17,-0.21,-0.21,-0.48,-0.45,-0.46,-0.52,-0.18,-0.45,0.06,-0.22,-0.45,-0.47,-0.21,0.0,-0.35,0.1,-0.1,-0.21,-0.18,0.01,0.02,0.29,0.17,0.17,-0.13,0.31,0.56,0.04,0.24,0.07,0.23,0.17,0.06,-0.2,0.02,-0.19,0.08,-0.28,-0.21,-0.01,-0.11,-0.32,-0.01,-0.07,-0.05,-0.12,-0.13,-0.11,0.0,-0.1,-0.23,0.17,-0.19,0.06,-0.22,-0.21,-0.1,-0.14,-0.43,-0.1,-0.14,-0.13,0.16,0.05,0.26,0.02,-0.24,-0.22,-0.09,-0.17,-0.15,-0.12,-0.07,-0.11,0.24,-0.16,-0.06,-0.09,-0.01,-0.21,0.1,-0.17,0.22,0.0,-0.01,0.02,0.09,0.17,-0.13,0.32,0.22,0.2,0.17,0.62,0.19,0.67,0.37,0.48,0.19,0.08,0.13,0.04,0.2,0.04,0.14,0.11,0.03,0.28,0.2,0.39,-0.03,-0.09,0.3,0.06,-0.07,0.0,0.51,0.24,-0.02,-0.04,0.09,0.37,0.02,-0.04,-0.12,-0.05,-0.21,-0.14,-0.14,-0.3,-0.1,0.08,-0.27,-0.15,0.14,-0.04,-0.49,-0.09,-0.41,-0.65,-0.22,-0.59,-0.26,-0.58,-0.01,-0.36,-0.3,0.16,-0.45,-0.48,-0.28,-0.35,-0.21,-0.39,-0.23,-0.24,-0.46,-0.32,-0.09,-0.24,-0.25,-0.22,0.0,-0.07,0.01,0.15,0.27,0.39,0.12,0.2,0.08,-0.03,0.26,0.31,0.33,0.59,0.33,-0.02,0.03,0.13,0.27,0.42,0.37,0.04,0.13,0.09,0.31,-0.34,0.18,0.57,0.09,-0.09,-0.16,0.12,-0.06,-0.23,-0.17,0.06,-0.13,0.0,-0.24,-0.27,-0.05,0.05,0.41,0.31,-0.09,-0.2,-0.06,0.42,0.47,0.03,-0.04,-0.04,-0.22,0.18,-0.09,0.05,-0.23,-0.26,-0.34,0.23,0.12,-0.26,-0.32,-0.31,-0.33,-0.07,-0.3,0.05,-0.13,-0.26,0.13,-0.05,0.18,-0.12,-0.12,0.01,-0.36,-0.04,-0.3,-0.13,-0.19,-0.04,-0.04,-0.21,-0.37,-0.25,-0.01,0.08,-0.26,-0.13,-0.18,-0.23,0.14,-0.32,-0.14,-0.12,-0.23,0.12,0.01,0.12,-0.1,0.01,0.04,-0.15,-0.23,0.04,-0.19,0.14,0.03,0.01,0.02,0.15,-0.07,0.1,0.11,0.28,0.38,-0.03,0.12,0.22,0.54,0.35,0.06,0.37,0.45,0.38,0.42,0.13,0.15,0.57,0.24,0.41,0.51,0.25,0.39,0.31,0.2,0.16,0.17,0.44,0.23,0.04,-0.15,0.17,-0.03,0.01,0.01,-0.19,-0.03,-0.14,0.12,0.01,-0.2,-0.15,0.27,-0.02,0.01,0.01,-0.19,0.23,-0.16,-0.03,-0.19,0.16,0.06,-0.08,-0.11,0.05,-0.18,-0.36,-0.23,0.04,-0.12,-0.25,0.03,-0.25,-0.16,0.04,-0.04,0.08,0.39,0.22,0.17,0.16,0.23,0.57,0.1,0.13,0.21,0.23,0.37,0.47,0.47,0.28,0.42,0.32,0.47,0.4,0.36,0.2,0.36,0.13,0.33,0.29,0.16,0.03,0.48,0.21,0.36,0.25,0.34,0.17,0.17,-0.01,0.43,-0.08,-0.08,0.57,-0.03,-0.06,0.19,-0.18,0.12,0.2,0.0,0.28,-0.07,-0.01,0.13,-0.07,0.11,-0.06,0.19,-0.07,0.04,-0.14,-0.29,-0.04,-0.03,-0.16,-0.02,-0.34,-0.27,-0.16,0.06,-0.33,-0.12,-0.24,-0.47,-0.43,-0.32,-0.47,-0.25,-0.23,-0.35,-0.06,-0.43,0.01,-0.34,-0.1,-0.3,-0.1,-0.03,0.06,-0.19,-0.22,0.16,-0.03,-0.01,0.12,0.03,-0.09,0.03,0.48,-0.07,0.17,0.21,0.44,0.52,0.1,0.2,0.08,0.46,-0.03,-0.07,0.02,0.15,0.05,0.23,0.0,0.3,-0.03,-0.07,0.09,-0.26,0.09,0.29,-0.19,-0.38,-0.39,-0.17,0.17,-0.1,-0.12,-0.15,-0.15,0.14,0.09,-0.29,-0.08,-0.15,-0.23,-0.1,-0.06,-0.18,0.38,0.21,-0.12,0.02,-0.14,-0.19,-0.2,0.08,-0.24,0.22,-0.1,-0.34,-0.07,-0.28,-0.29,-0.18,-0.13,-0.15,-0.18,-0.1,-0.21,-0.2,0.33,-0.13,0.09,0.38,0.1,0.03,0.1,-0.07,0.21,0.11,0.06,-0.19,0.02,0.12,-0.13,0.2,0.2,0.2,0.1,0.11,0.08,0.11,0.2,0.17,0.17,0.11,-0.13,0.31,0.04,0.0,0.01,0.13,0.7,0.06,0.07,0.0,0.19,0.1,0.22,0.12,0.26,0.37,0.09,0.23,0.53,0.22,0.33,0.6,0.29,0.75,0.65,0.34,0.58,0.46,0.91,0.62,0.69,0.62,0.57,1.02,0.81,0.88,0.73,0.79,0.79,0.85,0.78,0.91,0.73,0.33,0.69,0.51,0.58,0.9,0.58,0.63,0.33,0.45,0.45,0.68,0.62,0.73,0.55,0.33,0.36,0.25,0.62,0.52,1.02,0.43,0.37,0.25,0.78,0.33,0.24,0.45,0.72,0.54,0.56,0.29,0.42,0.48,0.38,0.3,0.29,0.25,0.04,0.03,0.41,0.24,0.62,0.27,0.49,0.47,0.66,0.2,0.25,0.26,0.37,0.15,0.09,0.28,0.19,0.47,0.63,0.55,0.25,0.11,0.32,0.37,0.22,0.25,0.21,-0.05,0.19,-0.23,0.02,0.1,-0.1,0.01,0.2,0.14,-0.07,0.2,0.42,0.2,0.22,0.33,-0.15,0.37,0.2,0.24,0.12,0.04,0.01,0.15,0.08,0.02,0.0,-0.09,0.02,0.08,0.01,0.1,0.23,0.18,0.3,0.04,-0.13,0.38,0.02,0.4,-0.13,0.18,0.16,0.05,-0.1,0.13,0.13,0.09,-0.09,-0.14,0.2,-0.09,0.15,0.04,0.15,0.22,0.01,-0.08,0.06,-0.23,0.07,0.12,0.18,-0.21,0.43,0.17,-0.1,0.3,-0.12,0.0,0.05,-0.03,-0.04,-0.13,-0.17,0.02,0.06,0.16,0.09,-0.05,-0.16,-0.25,-0.35,-0.08,-0.38,-0.26,-0.17,-0.19,-0.52,0.0,0.04,-0.18,0.0,-0.31,-0.41,-0.51,-0.27,-0.06,-0.35,-0.01,-0.08,0.4,-0.27,-0.15,-0.12,-0.07,0.05,0.02,-0.28,0.24,-0.06,-0.13,0.0,0.04,-0.29,0.04,0.22,0.23,-0.03,-0.03,0.01,-0.3,-0.16,-0.19,-0.18,-0.16,-0.3,-0.15,0.0,0.09,-0.25,-0.18,0.0,0.12,-0.07,0.19,0.06,-0.16,0.04,-0.11,0.22,-0.07,-0.18,0.15,0.2,-0.1,0.04,0.04,-0.07,0.14,-0.05,-0.2,0.02,-0.16,-0.17,-0.3,0.1,0.28,0.1,-0.19,-0.34,-0.29,0.06,-0.11,0.21,-0.29,0.2,0.12,0.06,-0.12,0.23,-0.1,0.52,0.4,0.12,0.37,-0.07,0.0,-0.03,0.04,-0.01,0.47,-0.11,0.13,-0.14,-0.17,-0.1,-0.32,0.03,-0.19,0.12,-0.13,-0.09,-0.23,-0.34,-0.07,-0.18,-0.23,-0.12,-0.21,-0.22,-0.53,-0.18,-0.34,-0.49,-0.08,0.14,-0.33,-0.31,-0.17,-0.48,0.18,-0.16,-0.3,-0.21,0.17,-0.18,-0.32,-0.26,-0.3,-0.36,-0.16,-0.41,-0.35,-0.29,-0.22,-0.19,-0.3,-0.16,0.43,0.0,-0.25,0.34,0.01,-0.13,0.12,-0.14,0.09,0.16,-0.15,0.1,0.28,0.16,0.13,-0.1,-0.07,-0.36,-0.21,-0.11,-0.04,0.02,-0.18,-0.29,-0.31,0.17,-0.38,-0.43,-0.14,-0.27,-0.45,-0.42,-0.49,-0.54,-0.3,-0.21,-0.29,-0.27,-0.17,-0.39,-0.46,-0.31,-0.43,-0.48,-0.26,-0.28,-0.15,0.06,0.03,-0.23,-0.26,-0.19,-0.05,-0.13,-0.21,-0.18,-0.13,-0.05,-0.05,-0.03,-0.2,-0.14,-0.05,-0.07,-0.03,0.17,0.1,0.0,-0.11,0.01,-0.31,-0.31,-0.34,-0.05,0.06,-0.04,-0.13,0.01,-0.06,0.2,0.1,0.13,0.1,0.38,-0.14,-0.09,0.01,-0.25,-0.12,-0.19,-0.2,-0.28,-0.14,-0.3,0.12,-0.42,0.09,-0.33,-0.13,0.31,-0.37,-0.18,-0.33,-0.43,-0.39,-0.36,-0.18,-0.09,-0.03,-0.1,-0.32,-0.16,-0.29,-0.26,-0.35,-0.2,-0.37,-0.28,0.09,-0.38,-0.35,-0.3,-0.26,-0.27,-0.17,-0.35,-0.01,0.02,0.17,0.3,-0.04,-0.16,0.17,0.05,-0.14,-0.13,0.01,0.23,0.38,0.47,-0.14,-0.17,0.27,-0.28,0.08,0.01,-0.01,0.1,0.04,0.0,-0.04,0.45,-0.03,0.02,0.2,-0.17,0.21,-0.1,0.08,0.16,0.04,-0.04,0.16,0.17,-0.08,0.19,0.15,0.17,-0.09,0.19,0.1,-0.14,-0.25,0.03,-0.22,-0.07,-0.06,0.15,0.16,-0.08,0.03,0.03,0.03,0.05,-0.09,0.14,-0.21,-0.22,-0.41,-0.17,-0.14,-0.07,-0.11,0.35,-0.21,-0.04,0.28,-0.16,-0.03,0.19,-0.09,-0.34,-0.1,-0.15,-0.07,0.25,0.19,0.3,0.02,0.27,0.17,0.36,0.19,0.32,0.3,0.15,0.26,0.1,0.28,0.14,0.27,0.13,0.34,0.38,0.34,0.27,0.15,0.43,0.29,0.23,0.3,0.13,-0.13,0.02,0.04,0.19,-0.03,0.52,-0.13,0.14,-0.13,-0.05,-0.08,-0.18,-0.21,-0.1,-0.25,0.42,0.13,0.11,-0.21,0.04,-0.2,-0.04,-0.48,-0.14,-0.24,-0.19,-0.37,-0.27,-0.31,-0.14,-0.09,-0.37,-0.07,0.02,-0.34,-0.16,0.14,-0.11,-0.04,-0.13,0.08,-0.04,-0.03,-0.15,0.38,0.08,-0.21,0.05,-0.24,-0.16,0.3,-0.05,0.1,0.01,-0.27,-0.09,0.11,0.06,0.1,-0.04,0.25,0.19,0.18,0.1,-0.03,0.26,0.15,0.1,0.42,0.31,0.31,0.27,0.4,0.38,0.22,0.27,0.57,-0.05,0.06,-0.03,0.21,0.25,-0.11,-0.1,0.48,-0.05,0.13,0.03,0.54,0.04,-0.12,-0.35,0.08,-0.13,-0.04,0.06,-0.06,-0.14,0.26,-0.29,-0.12,-0.2,-0.26,-0.23,-0.37,-0.15,0.0,0.05,0.26,0.27,0.0,-0.21,-0.18,-0.06,-0.43,-0.14,-0.32,-0.21,-0.14,-0.07,0.34,-0.05,-0.36,-0.43,-0.41,-0.41,-0.52,-0.41,-0.43,-0.19,-0.43,-0.18,-0.18,-0.31,-0.24,-0.07,-0.23,-0.21,-0.24,0.06,-0.25,-0.15,0.17,-0.15,-0.12,-0.24,-0.31,-0.26,-0.11,-0.09,-0.06,-0.09,-0.15,-0.2,-0.38,0.05,-0.14,0.1,0.09,0.09,0.11,0.31,0.04,0.14,0.1,-0.13,0.02,0.01,0.06,0.09,0.16,0.32,-0.3,-0.07,-0.17,-0.38,-0.33,-0.29,-0.22,-0.17,-0.35,-0.29,-0.23,-0.13,-0.2,-0.08,-0.14,-0.13,-0.42,-0.36,-0.39,-0.5,-0.29,-0.31,-0.47,-0.22,-0.47,-0.26,-0.37,-0.46,0.14,-0.18,-0.43,-0.45,-0.02,-0.09,-0.09,-0.19,0.1,0.01,0.19,-0.2,-0.21,-0.05,0.28,-0.06,-0.09,-0.13,-0.16,-0.1,-0.04,-0.15,-0.37,0.04,-0.26,-0.28,-0.18,-0.06,0.12,-0.43,-0.24,-0.14,-0.24,-0.05,0.16,-0.26,-0.19,-0.13,-0.21,-0.1,0.13,-0.26,-0.06,-0.05,-0.04,-0.17,-0.21,0.04,0.4,-0.06,-0.08,-0.12,-0.38,-0.49,-0.28,-0.09,0.0,-0.08,-0.49,-0.36,-0.28,-0.41,-0.31,-0.27,-0.5,-0.48,-0.26,-0.59,-0.3,-0.45,-0.31,-0.51,-0.21,-0.49,-0.18,-0.5,-0.61,0.04,-0.55,0.0,-0.36,-0.11,-0.4,-0.49,-0.4,-0.03,0.13,-0.08,0.15,0.09,-0.03,-0.16,-0.3,-0.11,-0.11,-0.07,-0.13,0.01,0.18,0.28,-0.19,-0.03,0.11,-0.05,-0.01,0.34,0.16,0.31,0.2,0.04,0.04,0.23,0.23,0.22,0.17,0.04,-0.14,0.02,0.06,-0.12,0.07,0.17,-0.16,0.02,-0.06,0.16,0.23,0.39,0.21,-0.03,-0.06,0.02,0.31,-0.04,-0.05,0.1,-0.16,-0.2,-0.06,-0.31,0.16,-0.14,-0.03,-0.21,0.19,-0.3,-0.27,-0.05,-0.28,-0.06,-0.2,-0.25,-0.04,-0.22,-0.05,-0.24,-0.14,-0.03,-0.26,-0.08,-0.39,-0.11,-0.25,-0.17,-0.04,-0.07,-0.28,-0.47,-0.46,-0.02,-0.1,-0.37,-0.31,-0.07,0.19,-0.39,-0.19,-0.04,-0.17,-0.04,0.11,-0.01,-0.09,0.11,0.06,-0.31,-0.13,-0.04,-0.28,0.17,-0.01,-0.43,0.1,0.12,-0.29,0.28,0.06,0.19,-0.06,0.21,0.28,0.41,0.08,0.01,-0.04,0.14,0.13,-0.02,-0.01,0.06,-0.05,0.17,-0.08,-0.03,-0.13,0.18,0.62,0.06,0.08,0.16,0.14,0.1,-0.09,-0.06,-0.24,0.12,0.07,0.04,-0.01,0.27,-0.08,-0.16,-0.26,-0.06,-0.05,-0.11,-0.07,0.0,-0.04,-0.05,0.23,-0.14,0.15,0.24,0.28,-0.06,0.18,0.05,0.12,0.09,0.07,-0.16,0.22,0.31,0.57,0.52,0.16,0.03,0.1,0.16,0.12,-0.03,-0.04,0.01,0.4,0.04,0.17,-0.03,0.38,-0.18,-0.04,0.03,0.1,-0.26,-0.08,-0.03,0.18,-0.43,-0.08,0.29,-0.33,-0.16,-0.19,-0.23,-0.35,-0.01,-0.26,-0.05,-0.19,-0.12,0.11,-0.2,0.04,0.12,0.17,0.44,0.48,0.42,0.39,0.27,0.25,0.37,0.35,0.1,0.07,0.53,0.21,-0.03,0.35,0.42,0.23,0.17,-0.23,0.03,0.14,0.21,0.14,0.04,-0.17,0.05,-0.15,-0.19,-0.16,0.03,-0.2,-0.06,0.06,-0.09,-0.12,0.13,0.4,0.14,0.18,0.24,0.14,0.3,0.27,0.26,0.04,0.19,0.04,-0.1,0.26,0.0,0.08,0.1,-0.05,0.38,-0.08,0.12,0.02,-0.08,0.34,0.24,0.35,-0.01,0.04,0.02,0.05,0.18,0.36,0.39,0.06,-0.15,-0.26,-0.37,0.06,0.13,-0.27,-0.31,0.16,-0.03,-0.06,-0.17,0.31,-0.23,-0.15,0.0,0.14,-0.06,0.04,0.17,-0.08,-0.25,0.0,-0.13,-0.21,-0.1,0.19,-0.08,-0.01,0.1,0.03,0.2,0.14,0.31,-0.01,0.02,0.26,0.01,-0.1,-0.29,-0.12,0.02,-0.04,-0.06,-0.2,-0.08,-0.31,-0.35,-0.51,-0.22,-0.43,0.08,-0.24,-0.17,0.1,-0.44,-0.1,-0.01,-0.32,-0.26,0.27,0.14,0.02,-0.1,0.17,0.06,-0.11,-0.19,0.01,-0.16,-0.04,-0.13,0.37,-0.03,0.28,0.04,-0.08,-0.19,0.29,-0.05,0.04,-0.09,0.17,-0.11,-0.2,-0.2,0.03,0.19,-0.14,-0.17,0.1,-0.19,-0.2,-0.19,-0.3,-0.19,-0.11,-0.33,-0.38,-0.33,-0.37,-0.39,-0.2,-0.13,-0.04,-0.2,-0.29,-0.04,-0.2,-0.36,0.1,-0.12,-0.05,0.27,0.01,-0.29,-0.1,-0.09,0.13,0.03,0.08,-0.03,-0.29,-0.08,0.18,-0.14,0.11,0.0,-0.03,-0.15,-0.36,0.28,0.01,-0.08,-0.2,-0.25,0.06,-0.09,-0.06,0.34,-0.03,-0.04,-0.27,0.04,0.39,0.23,0.09,0.12,0.04,-0.03,0.09,-0.1,0.22,0.13,-0.13,-0.27,0.14,0.23,0.36,0.02,-0.12,0.28,0.02,-0.08,-0.27,-0.26,-0.29,0.07,-0.12,-0.08,-0.2,-0.26,-0.13,-0.09,0.08,-0.32,-0.2,0.08,0.14,-0.06,-0.09,0.3,0.1,0.13,0.15,-0.24,-0.17,0.1,0.11,0.27,-0.39,0.1,0.03,0.11,0.1,-0.05,-0.06,-0.18,-0.2,-0.12,-0.09,-0.33,-0.27,-0.36,-0.38,-0.23,-0.41,0.05,-0.36,-0.19,0.01,-0.38,0.06,-0.2,0.06,-0.09,-0.08,-0.05,-0.27,-0.08,-0.21,-0.04,-0.05,-0.38,-0.44,-0.37,-0.2,-0.44,0.21,-0.52,-0.33,-0.48,-0.28,-0.13,-0.04,0.09,-0.29,-0.11,0.13,-0.19,-0.09,0.0,0.05,-0.28,-0.04,-0.24,-0.19,-0.12,-0.38,-0.12,-0.02,0.09,-0.3,-0.08,-0.2,-0.24,0.04,-0.07,-0.12,0.01,-0.14,0.18,-0.1,0.1,0.11,0.41,-0.1,0.01,0.2,-0.06,-0.13,-0.32,-0.08,-0.05,0.04,-0.15,0.03,-0.03,-0.09,-0.01,-0.16,-0.16,0.1,0.07,-0.06,-0.08,-0.15,0.16,-0.17,0.1,0.22,0.27,-0.04,-0.05,-0.06,0.19,0.3,0.28,-0.03,0.26,0.51,0.18,0.27,0.1,0.22,0.2,0.29,-0.01,0.13,-0.07,0.2,0.05,0.33,0.14,-0.03,0.03,0.05,0.37,0.1,0.25,-0.11,0.08,0.12,-0.11,-0.12,-0.08,-0.07,-0.1,0.17,0.26,0.09,0.03,0.03,0.04,0.41,-0.02,0.21,0.36,-0.05,0.19,-0.04,0.36,0.04,-0.04,0.13,0.03,-0.18,0.11,-0.19,-0.01,-0.04,-0.1,-0.19,-0.08,-0.09,-0.01,-0.03,0.09,-0.03,0.32,0.13,0.05,-0.16,0.14,0.3,0.08,0.17,-0.05,0.36,0.1,0.12,0.34,0.38,0.14,0.09,0.05,0.11,0.18,0.21,-0.08,0.42,0.18,0.02,-0.13,0.0,0.23,0.07,-0.11,-0.15,-0.16,-0.26,0.02,-0.15,-0.14,-0.14,-0.18,0.04,-0.1,-0.13,0.17,0.1,-0.1,0.03,-0.23,-0.26,-0.28,-0.34,-0.32,0.0,-0.25,-0.12,-0.06,-0.29,-0.27,-0.34,-0.38,-0.18,-0.26,-0.36,-0.34,-0.22,-0.2,-0.17,-0.2,-0.27,-0.16,0.17,-0.11,-0.16,0.42,0.09,0.17,-0.17,0.37,0.1,0.06,-0.01,-0.06,-0.23,-0.12,0.03,-0.06,-0.28,-0.23,-0.11,0.09,-0.04,-0.04,-0.19,-0.22,-0.26,-0.2,-0.07,-0.16,0.28,0.02,-0.04,0.15,0.19,0.06,0.01,0.16,0.42,0.23,0.04,0.02,0.15,0.63,0.1,0.24,0.28,-0.03,0.2,0.16,0.12,0.2,-0.15,-0.05,0.2,0.34,0.07,0.06,0.19,0.25,-0.18,0.08,0.22,-0.03,-0.12,0.06,-0.08,-0.28,-0.17,-0.07,-0.1,0.0,-0.18,-0.12,-0.26,0.01,-0.14,-0.14,-0.2,-0.28,0.0,-0.06,0.02,0.26,-0.06,0.26,0.26,-0.1,0.04,-0.12,0.29,-0.05,0.09,0.06,0.09,0.15,0.02,0.28,0.13,-0.17,0.21,-0.01,-0.09,0.1,0.46,0.06,0.08,0.27,0.16,0.3,-0.34,-0.12,-0.41,-0.06,-0.16,-0.06,-0.28,-0.01,-0.05,-0.07,-0.14,-0.02,0.27,0.1,0.41,0.23,0.0,0.18,0.37,0.0,0.17,0.14,-0.05,0.37,0.17,-0.01,0.14,-0.03,0.03,0.08,0.1,0.15,-0.11,0.29,0.02,0.21,-0.13,0.06,-0.08,-0.06,-0.17,-0.09,0.2,-0.39,-0.3,-0.26,-0.09,-0.4,-0.42,-0.29,-0.52,-0.1,-0.05,-0.45,-0.33,-0.3,-0.28,-0.29,-0.15,-0.03,-0.08,0.36,-0.23,0.03,0.13,-0.06,0.64,0.11,0.28,0.12,0.0,-0.2,-0.06,-0.17,-0.06,-0.06,-0.04,0.0,-0.09,0.2,0.02,0.03,0.05,-0.27,-0.15,-0.04,-0.23,-0.13,0.1,-0.3,-0.11,-0.14,-0.06,-0.15,-0.23,0.12,-0.15,-0.05,-0.11,-0.36,-0.18,0.03,-0.24,-0.24,-0.42,0.0,0.04,0.54,-0.02,-0.17,0.06,0.05,0.13,-0.21,-0.16,-0.12,0.25,0.06,0.08,-0.19,-0.15,-0.1,0.28,-0.14,-0.09,0.14,-0.33,-0.14,-0.09,-0.31,0.04,-0.29,-0.19,-0.01,-0.19,-0.01,-0.15,0.43,0.11,0.01,0.0,-0.39,-0.19,-0.13,0.15,0.01,-0.18,0.09,0.23,0.04,0.02,0.23,-0.18,-0.24,-0.09,-0.06,-0.1,-0.14,0.18,-0.33,-0.06,-0.14,-0.16,-0.03,0.15,-0.07,-0.26,-0.1,0.13,0.14,-0.26,0.16,-0.08,0.02,-0.37,0.05,0.1,0.11,-0.08,0.11,-0.04,0.04,0.35,0.13,0.22,0.41,0.16,0.14,0.29,0.28,0.6,0.32,0.63,0.48,0.28,0.36,0.19,0.17,0.25,0.14,0.2,0.14,0.31,0.19,0.31,0.4,0.0,0.24,-0.1,0.01,-0.06,-0.05,0.03,0.11,0.03,-0.16,0.12,0.03,-0.1,-0.18,-0.08,0.2,-0.03,0.08,-0.08,0.2,0.04,0.11,0.13,-0.17,0.05,0.18,-0.06,-0.18,0.28,-0.15,-0.16,-0.11,0.41,0.02,-0.26,0.42,-0.23,0.08,-0.2,0.04,-0.04,-0.07,-0.08,0.11,-0.11,-0.08,-0.21,-0.39,-0.18,-0.35,0.04,-0.39,-0.19,-0.19,-0.25,0.1,-0.12,-0.2,-0.2,0.13,0.18,-0.24,0.0,0.12,0.19,0.16,0.36,-0.05,0.12,0.07,0.12,-0.04,0.16,-0.12,0.19,0.2,0.42,0.27,0.24,0.1,-0.08,-0.07,-0.13,-0.03,-0.09,0.14,0.32,0.35,0.3,0.6,0.39,0.36,0.28,0.27,0.33,-0.02,0.23,0.2,0.16,0.2,0.03,0.06,0.28,0.38,0.29,-0.08,0.35,0.19,-0.07,0.03,0.08,-0.11,0.02,0.2,0.15,0.15,0.18,-0.03,0.14,0.12,-0.01,0.11,-0.18,-0.17,0.04,0.45,0.1,0.1,0.2,0.13,0.46,0.08,0.05,-0.08,0.04,0.46,0.14,-0.01,0.01,0.12,0.04,-0.17,-0.07,-0.2,-0.06,0.09,0.07,-0.02,-0.13,-0.31,-0.13,-0.37,-0.09,-0.23,-0.17,-0.32,-0.05,-0.25,-0.33,-0.34,0.0,-0.05,-0.26,0.01,0.0,-0.36,-0.1,0.06,0.01,0.07,0.08,-0.18,0.26,-0.03,0.29,0.1,-0.21,0.16,0.26,-0.03,-0.21,-0.09,0.12,0.18,0.01,-0.03,-0.03,-0.12,0.19,0.26,-0.16,-0.28,0.06,0.01,0.16,0.08,-0.25,-0.29,-0.38,-0.07,-0.12,-0.67,0.02,-0.13,-0.33,-0.13,-0.27,-0.25,-0.34,0.08,0.1,0.0,-0.08,-0.08,-0.24,-0.29,-0.21,-0.36,-0.33,-0.21,-0.07,-0.38,0.09,-0.15,-0.17,-0.14,-0.03,-0.45,-0.45,-0.22,-0.49,-0.28,-0.34,-0.24,-0.26,-0.5,-0.05,-0.25,-0.33,-0.13,-0.33,-0.33,0.03,-0.38,-0.23,-0.18,0.05,-0.2,-0.07,0.07,0.13,0.17,-0.01,-0.03,-0.21,0.21,-0.09,-0.35,-0.08,-0.35,-0.18,0.03,-0.03,-0.1,0.03,0.15,0.32,-0.04,0.15,-0.04,0.06,-0.05,0.03,0.12,0.03,-0.08,-0.17,0.4,0.05,0.18,0.3,-0.01,-0.42,-0.1,0.01,0.13,-0.09,0.02,-0.27,-0.16,0.04,-0.13,0.12,-0.12,0.22,-0.03,0.06,0.07,0.0,0.19,-0.12,0.15,-0.01,0.11,0.04,0.33,-0.08,-0.03,-0.09,0.09,0.02,0.02,0.06,-0.17,-0.03,-0.17,-0.21,-0.16,-0.26,-0.08,-0.15,0.29,-0.29,-0.28,-0.21,-0.23,-0.13,-0.18,-0.01,-0.13,-0.03,-0.13,0.05,-0.16,-0.03,0.01,-0.07,-0.29,-0.07,-0.42,-0.33,-0.28,-0.11,-0.03,-0.33,-0.06,-0.3,0.13,-0.06,-0.08,-0.11,-0.2,-0.02,-0.01,-0.12,0.12,0.34,0.11,0.02,-0.02,-0.15,0.3,0.45,0.25,-0.12,-0.16,-0.03,-0.1,0.14,0.35,0.18,0.15,0.46,0.14,0.49,0.31,0.12,0.39,0.1,0.04,-0.03,0.16,0.17,0.21,0.06,0.11,0.04,0.06,0.26,0.05,-0.08,0.04,-0.08,-0.03,-0.12,-0.16,-0.06,0.27,-0.35,-0.16,-0.29,-0.21,-0.3,0.03,-0.05,-0.03,-0.19,-0.4,-0.24,0.09,-0.26,-0.19,-0.11,0.0,-0.06,-0.22,0.08,-0.18,-0.15,0.07,-0.07,0.09,-0.15,0.08,0.0,0.21,-0.14,-0.28,-0.14,-0.21,-0.21,-0.23,-0.17,-0.33,-0.33,-0.43,-0.28,-0.18,-0.52,-0.09,-0.31,-0.15,-0.47,-0.33,-0.15,-0.28,-0.39,-0.29,-0.31,-0.14,-0.26,-0.43,-0.23,-0.1,-0.11,-0.19,-0.2,-0.29,-0.14,-0.15,-0.2,-0.11,0.36,-0.3,-0.07,-0.01,0.26,0.17,0.05,0.44,0.15,-0.01,-0.01,0.05,-0.15,0.18,0.03,-0.21,0.46,-0.03,-0.17,-0.13,-0.03,-0.07,-0.26,0.24,0.08,0.19,0.03,-0.3,0.24,-0.21,-0.11,-0.26,-0.13,0.04,0.31,-0.06,-0.02,-0.27,0.12,-0.23,0.24,-0.17,-0.12,-0.07,-0.38,-0.24,-0.17,-0.25,-0.2,0.21,-0.1,-0.2,-0.23,-0.2,-0.1,-0.08,0.0,0.14,-0.41,-0.27,-0.33,-0.39,-0.17,-0.26,-0.31,-0.32,-0.16,-0.35,-0.14,-0.22,-0.38,-0.41,0.12,-0.3,0.01,-0.25,-0.05,-0.3,0.02,-0.27,-0.18,-0.19,-0.03,0.34,-0.09,-0.17,0.29,-0.08,-0.07,0.1,-0.09,0.5,0.52,0.29,0.33,0.03,0.19,-0.19,0.23,0.13,0.02,0.39,0.26,0.15,-0.14,-0.13,0.06,-0.15,-0.03,-0.28,0.01,0.25,-0.05,0.14,0.06,-0.19,0.17,0.22,-0.1,0.31,0.16,-0.03,0.19,0.01,0.23,0.1,0.06,-0.17,-0.16,0.08,-0.1,0.06,-0.09,0.01,-0.13,0.1,-0.14,-0.15,0.1,0.12,-0.19,-0.08,-0.07,0.13,0.06,-0.05,0.17,0.13,0.06,0.19,0.15,0.09,0.1,-0.12,-0.08,-0.25,-0.01,-0.27,-0.11,-0.11,-0.08,-0.06,0.18,0.21,0.16,0.03,-0.2,0.22,-0.07,-0.21,0.18,-0.01,-0.23,-0.17,0.15,-0.1,0.32,-0.18,0.01,0.12,-0.14,-0.1,-0.04,-0.24,-0.12,-0.13,-0.06,0.11,0.02,0.27,0.06,-0.11,0.18,0.29,0.08,0.33,0.04,0.03,-0.09,0.08,0.26,0.22,0.01,0.15,0.3,0.24,0.18,0.17,-0.03,-0.07,0.28,0.17,0.33,0.02,-0.07,0.09,0.04,0.01,-0.1,-0.14,-0.19,0.04,0.06,0.22,0.14,0.05,-0.25,0.03,-0.18,0.12,-0.08,-0.17,-0.34,-0.27,-0.17,-0.07,-0.15,-0.43,-0.32,-0.23,-0.16,0.04,-0.08,0.06,-0.36,-0.29,-0.16,-0.21,-0.23,-0.17,0.06,-0.15,0.25,0.01,-0.08,0.15,-0.08,-0.01,-0.14,-0.17,-0.56,-0.3,-0.05,-0.13,-0.04,-0.2,-0.18,-0.24,0.12,0.04,-0.35,-0.5,-0.32,-0.42,-0.21,-0.38,-0.23,-0.39,-0.39,-0.14,-0.16,-0.35,-0.17,-0.34,0.03,-0.18,-0.3,-0.19,-0.14,-0.18,-0.24,-0.1,0.01,-0.17,-0.3,0.01,0.06,0.02,-0.02,-0.14,-0.2,-0.11,-0.4,-0.23,0.07,0.05,-0.16,-0.22,0.11,-0.1,-0.25,0.05,-0.06,0.0,-0.11,-0.25,-0.23,-0.46,0.25,-0.37,-0.25,-0.48,-0.01,-0.1,-0.4,-0.2,-0.16,-0.48,-0.43,-0.21,-0.32,-0.35,0.13,-0.18,-0.39,-0.09,-0.22,-0.04,0.03,-0.11,-0.06,-0.05,-0.23,-0.29,-0.04,-0.3,-0.2,0.23,-0.23,-0.24,0.15,0.13,-0.14,0.1,-0.11,-0.2,-0.12,-0.38,-0.05,-0.14,-0.23,0.09,-0.09,-0.2,-0.2,-0.2,-0.12,0.07,-0.2,-0.18,-0.2,-0.16,-0.12,0.06,-0.31,-0.07,-0.05,0.09,0.13,0.16,-0.04,0.54,-0.22,0.19,0.19,0.23,-0.12,0.02,-0.01,-0.2,-0.08,-0.18,-0.26,0.03,-0.29,-0.33,0.3,-0.16,-0.21,-0.16,-0.2,-0.06,-0.09,-0.07,0.27,-0.07,-0.26,-0.11,-0.53,-0.38,-0.39,-0.39,-0.34,-0.18,-0.63,-0.29,-0.41,-0.19,-0.34,-0.39,-0.5,-0.12,-0.48,-0.57,-0.15,-0.64,-0.54,-0.54,-0.45,-0.42,-0.23,-0.36,-0.3,-0.63,-0.46,-0.16,-0.4,-0.06,-0.4,-0.39,-0.54,-0.48,-0.59,-0.14,-0.16,-0.34,-0.39,-0.12,-0.18,-0.12,-0.09,0.16,-0.2,-0.07,-0.07,0.22,0.08,-0.13,0.22,-0.11,0.14,-0.06,0.25,0.37,0.31,-0.01,0.17,-0.02,-0.25,0.1,-0.03,-0.12,-0.22,-0.3,-0.13,-0.21,-0.18,-0.32,-0.42,-0.3,-0.17,0.1,-0.16,-0.24,-0.02,-0.39,-0.44,-0.07,-0.15,-0.19,-0.42,-0.48,-0.42,-0.48,-0.17,0.17,0.0,0.07,-0.5,-0.46,-0.47,-0.2,-0.34,-0.07,-0.09,-0.23,0.04,-0.06,-0.18,-0.05,-0.18,-0.02,0.14,0.06,-0.04,-0.21,-0.01,0.01,-0.03,0.19,0.46,0.23,0.09,0.21,0.23,-0.1,0.35,0.29,0.04,0.01,0.11,0.39,0.15,-0.12,0.26,0.18,0.34,0.09,-0.1,0.09,-0.18,0.01,0.15,0.21,-0.01,0.19,-0.15,0.2,0.12,0.3,0.2,0.1,0.03,0.03,-0.03,-0.09,0.2,0.19,-0.28,0.0,-0.33,-0.05,-0.14,0.12,-0.12,-0.06,0.28,0.05,0.28,-0.1,-0.07,0.22,0.11,0.02,0.29,0.17,0.0,0.27,0.08,0.09,0.17,0.1,-0.03,-0.03,-0.13,-0.17,0.04,-0.26,0.18,0.03,-0.15,0.1,0.31,0.01,0.08,-0.23,0.17,-0.2,0.1,0.2,-0.1,0.16,0.25,0.4,0.3,0.27,0.37,0.03,0.36,0.52,0.04,0.35,0.39,0.19,0.29,0.86,0.33,0.2,0.41,0.23,0.2,0.47,0.19,0.15,0.18,0.08,0.02,0.46,0.3,0.19,0.18,0.05,-0.03,-0.14,0.06,0.17,0.0,-0.1,0.04,0.48,0.17,0.13,0.31,0.33,0.21,0.01,0.43,0.04,0.27,0.38,0.23,0.59,0.92,0.37,0.49,0.57,1.04,0.81,0.74,0.87,0.45,0.38,0.61,0.5,0.72,0.42,0.42,0.55,0.31,0.39,0.31,0.14,0.28,0.18,0.44,0.47,0.22,0.24,0.52,0.38,0.41,0.54,0.37,0.63,0.71,0.3,0.71,0.6,0.47,0.63,0.44,0.44,0.52,0.6,0.71,0.49,0.52,0.41,0.43,0.54,0.22,0.18,0.39,0.49,0.21,0.12,0.24,0.03,-0.08,0.14,0.04,0.0,0.27,0.02,0.09,0.1,0.17,-0.09,0.29,-0.26,-0.05,0.02,-0.21,0.15,-0.06,0.03,-0.05,-0.12,-0.05,-0.11,0.1,-0.31,0.02,-0.05,-0.28,-0.17,-0.21,-0.07,-0.02,0.01,-0.41,-0.48,-0.2,-0.39,-0.17,-0.02,0.05,-0.3,0.04,-0.25,0.05,-0.4,-0.08,-0.15,-0.32,-0.15,0.01,-0.13,-0.21,-0.27,-0.16,-0.13,0.04,0.01,-0.25,0.0,0.13,-0.16,0.1,0.03,-0.23,0.15,0.03,0.27,-0.1,-0.32,0.05,-0.18,0.16,-0.26,-0.05,-0.21,-0.03,-0.2,0.04,0.35,0.11,-0.03,-0.09,-0.23,0.0,-0.27,0.0,-0.09,0.06,0.19,0.05,-0.29,-0.39,-0.29,-0.39,-0.02,-0.29,-0.35,-0.3,-0.16,-0.2,-0.4,-0.38,-0.36,-0.53,-0.15,-0.21,-0.12,-0.24,-0.37,-0.33,0.14,-0.28,-0.41,0.03,-0.03,-0.05,0.04,-0.08,-0.03,-0.16,0.03,-0.32,-0.04,-0.1,0.13,0.13,-0.01,0.02,-0.15,-0.25,-0.15,0.04,-0.07,0.01,-0.17,-0.26,-0.19,-0.14,-0.23,-0.12,-0.24,-0.21,-0.23,-0.39,-0.11,-0.28,-0.2,-0.33,-0.4,0.16,0.1,-0.3,-0.19,-0.19,-0.17,-0.01,-0.31,0.04,0.2,-0.01,-0.17,-0.16,-0.22,-0.13,-0.15,-0.04,0.23,0.1,0.1,-0.12,0.2,0.13,0.22,0.12,0.15,0.26,0.36,0.47,0.55,0.44,0.18,0.43,0.69,0.63,0.6,0.23,0.72,0.4,0.65,0.4,0.74,0.3,0.41,0.4,0.54,0.04,0.39,0.73,0.54,0.43,0.62,0.32,0.45,0.33,0.31,0.34,0.49,0.28,0.2,0.25,0.52,0.32,0.48,0.25,-0.01,0.12,-0.04,-0.04,0.24,-0.09,-0.16,0.23,-0.22,0.11,-0.15,-0.21,-0.28,-0.22,-0.41,-0.27,-0.13,-0.32,-0.3,-0.36,-0.07,-0.13,-0.41,-0.34,-0.05,-0.23,-0.05,-0.2,0.01,-0.11,-0.04,-0.04,-0.04,-0.09,-0.12,-0.3,-0.35,-0.11,0.25,0.22,0.31,0.06,-0.08,-0.07,-0.23,-0.04,-0.08,-0.17,0.04,0.06,-0.33,-0.09,-0.27,-0.28,-0.32,0.18,-0.59,-0.17,-0.51,-0.33,-0.68,-0.32,-0.43,-0.13,-0.17,-0.28,-0.41,-0.38,-0.25,-0.41,-0.31,-0.17,-0.39,-0.45,-0.26,0.26,-0.38,-0.42,-0.35,-0.23,-0.13,-0.37,0.03,0.27,-0.29,-0.3,-0.3,-0.02,-0.33,0.17,-0.16,-0.18,-0.23,-0.46,-0.03,-0.46,-0.21,-0.03,-0.19,-0.16,-0.13,-0.03,-0.31,-0.35,-0.3,-0.21,-0.36,0.04,-0.2,-0.28,-0.05,-0.26,-0.21,0.15,-0.17,0.11,0.06,0.06,0.18,-0.32,0.17,-0.29,0.06,0.14,-0.19,0.12,-0.12,-0.05,-0.04,-0.19,0.13,-0.11,0.02,0.15,0.05,0.12,0.27,0.17,0.24,0.19,-0.12,-0.04,-0.03,0.05,0.55,0.24,-0.05,0.04,-0.04,-0.1,-0.04,-0.33,-0.15,0.06,-0.21,-0.22,-0.21,-0.08,-0.31,-0.25,0.02,0.0,-0.14,0.03,0.17,0.12,0.07,0.14,0.11,0.12,0.5,0.19,0.21,0.28,-0.07,0.16,-0.19,0.08,-0.24,-0.02,0.08,0.04,-0.15,0.25,-0.3,0.02,0.04,0.03,-0.01,0.15,0.0,-0.16,-0.03,-0.03,-0.21,-0.23,-0.1,-0.33,-0.3,-0.2,-0.25,-0.29,0.01,-0.15,-0.37,0.16,-0.31,-0.11,-0.08,-0.11,0.04,-0.24,-0.14,-0.13,-0.28,0.03,-0.24,-0.1,0.28,0.12,0.01,-0.17,-0.26,0.08,-0.24,-0.23,0.04,-0.16,-0.23,-0.25,-0.17,-0.09,-0.07,-0.28,-0.16,-0.26,-0.23,-0.5,-0.4,-0.43,-0.53,-0.71,-0.49,-0.6,-0.17,-0.55,0.01,-0.49,-0.61,-0.34,-0.28,-0.32,-0.35,-0.64,-0.5,-0.33,-0.17,-0.52,-0.55,-0.51,-0.54,-0.37,-0.53,-0.4,-0.38,-0.34,0.15,-0.3,-0.21,-0.01,-0.18,-0.21,-0.34,0.02,-0.01,-0.29,0.03,-0.08,-0.11,-0.27,-0.11,-0.28,-0.13,0.26,0.26,-0.11,0.15,0.16,-0.07,0.1,0.08,-0.03,-0.26,-0.14,-0.06,0.36,0.12,0.01,0.06,0.14,0.06,0.04,0.02,0.13,0.14,-0.17,-0.07,-0.09,-0.1,-0.19,-0.27,-0.18,-0.2,-0.4,-0.31,-0.42,-0.14,0.22,0.07,0.0,-0.16,-0.02,-0.16,-0.26,-0.01,-0.03,-0.13,-0.08,-0.18,0.21,-0.31,-0.05,-0.11,-0.15,0.09,0.03,-0.01,0.21,-0.09,-0.12,-0.03,0.09,-0.44,-0.17,0.04,-0.29,-0.38,-0.33,-0.24,-0.26,-0.19,-0.03,-0.41,-0.27,-0.19,-0.29,-0.2,-0.24,-0.22,-0.14,-0.12,-0.31,-0.53,-0.44,-0.4,-0.33,-0.18,-0.32,-0.08,-0.25,-0.16,-0.23,0.01,-0.2,0.03,-0.17,-0.21,-0.14,-0.37,-0.26,0.06,-0.07,-0.23,-0.07,-0.15,-0.2,-0.24,-0.24,-0.2,-0.16,-0.31,-0.32,-0.2,-0.36,-0.38,-0.32,-0.28,0.03,-0.34,-0.1,-0.21,-0.32,-0.07,0.04,-0.44,-0.19,-0.15,-0.2,-0.16,-0.01,-0.09,-0.26,-0.13,0.01,-0.13,0.15,0.29,-0.18,0.44,0.12,0.54,0.35,0.22,0.02,0.32,-0.23,0.04,0.03,-0.05,0.06,0.19,-0.03,-0.08,-0.05,-0.11,-0.22,-0.08,-0.04,0.27,0.18,0.21,0.02,0.29,0.43,0.08,0.02,-0.07,0.17,0.36,-0.07,0.15,0.08,0.23,0.25,0.13,0.05,-0.18,-0.24,0.06,0.02,-0.38,0.07,-0.31,-0.05,-0.16,-0.43,-0.27,-0.08,-0.23,-0.26,0.29,0.09,-0.14,0.23,-0.11,0.17,-0.39,-0.11,-0.12,-0.14,-0.38,-0.1,0.04,-0.25,0.18,0.17,-0.04,0.33,0.06,-0.01,0.15,0.33,-0.04,-0.05,0.3,0.02,0.05,-0.18,0.23,0.19,-0.19,0.06,0.37,-0.29,0.26,-0.09,-0.07,0.11,-0.26,0.15,-0.11,-0.25,-0.03,-0.14,-0.33,-0.14,-0.18,-0.25,-0.16,-0.01,-0.12,-0.07,0.19,-0.06,0.03,0.21,0.07,0.22,0.04,0.02,-0.11,0.23,0.48,0.28,0.36,0.15,0.23,-0.09,-0.12,-0.04,0.04,-0.16,0.22,-0.05,0.34,0.18,0.0,-0.19,0.11,-0.22,-0.28,-0.1,-0.09,-0.17,-0.12,-0.2,-0.17,-0.13,-0.38,-0.03,-0.09,-0.21,-0.17,0.26,-0.03,0.27,-0.04,-0.27,0.21,-0.05,-0.23,-0.04,0.16,-0.16,-0.16,-0.13,0.12,-0.19,0.06,0.34,0.06,-0.12,0.03,0.04,0.3,-0.04,0.23,-0.09,-0.07,0.13,-0.14,-0.11,0.18,-0.28,-0.03,0.43,0.16,-0.08,0.17,0.1,0.06,0.38,0.02,0.06,-0.1,-0.06,0.15,-0.13,0.23,-0.03,0.12,-0.1,-0.14,-0.01,-0.18,-0.02,0.16,0.19,-0.01,-0.14,-0.22,0.0,0.24,0.08,0.18,0.14,0.16,-0.07,0.19,0.11,0.18,-0.13,0.09,0.03,-0.08,-0.14,0.0,-0.2,0.1,-0.26,0.0,-0.36,-0.1,-0.2,-0.35,-0.08,-0.04,0.0,-0.03,0.16,-0.29,-0.15,-0.11,-0.31,0.14,0.06,-0.11,0.1,-0.06,-0.17,0.0,0.4,0.01,0.15,0.03,0.07,-0.11,0.55,0.31,-0.08,0.17,0.44,0.2,0.05,0.25,0.09,0.43,0.15,0.53,0.37,0.31,0.08,0.08,0.03,0.19,0.17,-0.08,0.22,0.06,-0.06,0.03,0.16,0.1,0.07,0.02,0.21,-0.27,-0.04,-0.01,0.05,-0.14,-0.22,-0.27,-0.15,-0.16,-0.14,-0.24,-0.31,-0.34,-0.04,-0.31,-0.2,-0.29,-0.21,-0.07,0.06,-0.42,-0.14,0.09,-0.13,-0.12,-0.11,0.08,-0.22,-0.33,-0.07,-0.08,-0.1,-0.16,0.04,-0.28,-0.29,-0.27,-0.35,-0.42,-0.29,-0.5,-0.23,-0.41,-0.42,-0.18,-0.36,-0.31,-0.42,-0.12,0.06,-0.17,0.03,-0.25,0.06,-0.22,-0.18,0.06,0.26,-0.09,-0.19,0.19,-0.12,0.2,-0.36,0.29,0.21,0.07,0.01,0.04,-0.03,0.0,0.18,0.51,0.27,0.16,0.3,0.09,0.03,-0.14,0.17,-0.08,-0.12,0.14,0.01,-0.35,-0.13,0.05,-0.14,0.38,-0.24,0.05,-0.14,-0.14,-0.01,0.01,-0.16,0.13,0.06,0.11,-0.02,-0.04,0.12,0.24,0.23,0.12,0.13,-0.08,0.06,-0.11,0.0,0.02,0.11,0.27,0.14,0.08,-0.12,0.13,0.22,0.03,0.12,0.06,0.38,0.1,-0.07,-0.03,-0.01,-0.02,0.0,-0.18,0.11,0.05,-0.17,0.0,-0.05,-0.03,-0.39,-0.06,-0.14,-0.01,-0.23,0.09,-0.2,-0.39,-0.11,0.01,-0.2,-0.07,-0.24,-0.11,0.12,-0.04,-0.17,-0.1,0.09,0.16,0.21,0.32,-0.04,0.33,-0.04,0.11,0.35,-0.07,0.27,0.0,0.3,0.45,0.02,0.1,0.24,0.27,0.24,0.45,0.49,0.1,-0.24,0.04,0.1,-0.11,0.11,0.08,-0.11,0.12,0.13,0.01,-0.26,-0.08,0.39,0.01,-0.13,-0.02,-0.07,-0.05,-0.34,-0.37,-0.09,-0.09,0.17,-0.11,-0.2,0.16,0.01,-0.29,-0.43,-0.23,-0.09,-0.01,0.05,-0.31,-0.08,-0.23,-0.18,0.02,0.06,0.04,0.19,-0.12,0.21,-0.11,-0.01,-0.07,-0.28,0.04,0.15,-0.06,0.11,0.19,-0.02,-0.33,-0.32,0.25,-0.13,-0.34,-0.17,0.15,-0.28,-0.03,-0.09,0.06,-0.17,-0.35,-0.35,-0.1,-0.03,0.05,-0.39,-0.21,0.03,-0.3,-0.31,-0.21,0.08,-0.08,-0.23,-0.11,0.06,0.25,-0.08,0.13,0.1,0.16,0.31,-0.03,-0.07,-0.05,-0.15,-0.39,-0.03,-0.52,-0.24,-0.01,-0.27,-0.38,-0.1,0.1,0.01,-0.38,-0.04,-0.49,-0.13,0.02,0.14,0.01,0.2,-0.21,-0.15,-0.07,0.12,-0.28,-0.18,-0.07,0.03,-0.11,0.03,0.16,0.04,0.13,0.08,0.19,0.02,-0.01,0.32,0.09,-0.09,0.17,0.16,-0.06,-0.07,-0.14,0.14,-0.01,-0.03,0.08,0.06,0.16,-0.35,-0.19,0.07,0.23,0.21,-0.03,-0.04,0.0,0.03,-0.09,-0.09,-0.05,0.0,0.16,-0.32,0.39,0.17,0.06,0.06,-0.09,0.08,0.36,0.32,0.4,-0.08,0.18,-0.01,0.15,0.12,0.09,0.01,-0.08,0.03,-0.35,-0.26,-0.33,-0.1,0.03,-0.11,-0.05,-0.05,-0.14,-0.03,-0.06,0.03,-0.14,-0.14,-0.16,-0.01,0.42,-0.09,-0.03,-0.15,-0.13,-0.06,-0.2,0.0,0.26,0.0,0.29,0.22,-0.04,0.12,-0.06,0.01,-0.1,0.07,0.47,0.35,0.48,0.33,0.08,0.4,-0.2,0.09,0.01,-0.25,-0.19,-0.29,0.13,-0.11,0.02,0.02,-0.04,-0.03,-0.3,-0.19,-0.18,-0.3,-0.12,-0.4,-0.16,-0.16,-0.02,-0.28,0.0,0.06,-0.18,-0.14,-0.18,0.08,-0.31,-0.16,-0.16,-0.14,-0.57,-0.25,0.03,-0.44,-0.22,-0.16,-0.18,0.02,-0.31,-0.1,-0.18,0.22,-0.32,0.1,-0.04,-0.03,0.15,0.0,0.04,0.24,-0.11,-0.25,-0.14,-0.25,0.1,-0.15,-0.18,-0.11,0.11,-0.43,0.39,-0.15,0.22,0.16,-0.08,-0.19,0.21,0.06,0.12,0.4,0.26,0.45,0.36,0.04,0.42,0.02,0.13,0.32,0.06,0.04,0.38,0.12,0.45,0.34,0.15,0.32,0.12,0.07,0.48,0.0,0.15,0.18,-0.15,0.02,0.35,0.23,-0.04,-0.07,0.04,0.14,0.06,0.17,0.21,0.34,0.47,0.62,0.62,0.41,0.14,0.4,0.26,0.13,0.43,0.31,0.31,0.43,0.31,0.23,0.24,0.26,0.43,-0.36,0.31,0.14,0.06,0.27,0.21,0.02,0.05,0.22,-0.01,0.1,0.12,-0.03,-0.01,0.16,0.08,-0.39,0.11,-0.17,-0.06,-0.05,0.01,-0.09,-0.21,-0.46,-0.04,-0.33,-0.16,-0.5,-0.14,-0.33,-0.36,-0.41,-0.47,0.02,-0.41,-0.17,-0.53,-0.31,-0.42,-0.3,-0.47,-0.12,0.06,-0.51,-0.44,-0.14,-0.28,-0.09,-0.34,-0.29,0.08,-0.41,-0.11,0.03,-0.05,-0.21,-0.11,-0.29,-0.11,-0.57,-0.4,-0.25,-0.45,-0.1,-0.33,-0.33,-0.63,-0.29,-0.42,-0.44,-0.23,-0.32,-0.42,-0.16,-0.2,-0.17,-0.14,-0.11,-0.3,-0.23,-0.1,-0.1,-0.34,0.09,0.19,0.13,-0.09,-0.01,0.09,0.04,-0.29,-0.12,-0.06,-0.38,-0.02,-0.12,0.14,0.12,-0.06,0.11,-0.19,0.03,0.06,-0.17,-0.07,-0.13,-0.01,-0.03,-0.14,0.19,0.13,0.09,0.09,-0.15,0.48,0.13,-0.18,-0.13,-0.21,-0.05,-0.22,0.16,0.23,0.01,-0.26,0.06,0.31,-0.19,-0.01,-0.06,-0.09,-0.03,-0.04,-0.17,0.0,-0.13,0.0,0.1,-0.16,-0.48,-0.25,-0.38,-0.48,-0.47,-0.42,-0.32,-0.22,-0.42,-0.55,-0.21,-0.4,-0.19,-0.04,-0.27,-0.11,-0.22,-0.47,-0.1,-0.25,-0.43,-0.05,0.05,-0.25,0.24,-0.23,-0.09,-0.03,-0.12,-0.24,0.09,-0.28,-0.09,-0.41,-0.09,-0.3,0.05,-0.2,-0.17,0.15,-0.19,-0.22,-0.22,0.22,-0.01,0.15,-0.06,-0.11,0.17,0.1,0.13,-0.03,0.22,-0.07,-0.1,-0.07,-0.03,0.04,-0.14,-0.15,-0.21,-0.08,-0.25,-0.13,-0.1,-0.21,0.0,0.02,-0.05,-0.1,0.08,0.02,-0.07,-0.11,-0.2,-0.03,-0.12,-0.33,-0.17,-0.26,0.02,-0.24,-0.07,-0.11,0.06,-0.14,-0.14,0.52,0.15,0.27,-0.03,0.45,0.04,0.29,0.22,0.14,0.2,0.4,0.1,0.33,0.33,0.4,0.37,0.35,0.69,0.2,0.19,0.26,0.42,0.26,0.04,0.23,0.36,0.12,0.2,-0.08,-0.21,-0.14,-0.24,-0.18,-0.45,-0.26,-0.27,-0.32,-0.47,-0.41,-0.7,-0.69,-0.87,-0.91,-0.69,-0.77,-1.12,-0.66,-1.06,-1.05,-0.93,-1.0,-0.95,-0.9,0.22,-1.04,-0.75,-0.65,-0.42,-0.64,-0.58,-0.49,-0.47,-0.64,-0.52,-0.04,-0.76,-0.55,-0.6,-0.69,-0.48,-0.8,-0.66,-0.38,-0.02,-0.15,-0.25,0.12,-0.1,-0.12,-0.12,-0.13,-0.18,-0.34,-0.46,-0.35,-0.2,-0.14,-0.49,-0.22,-0.37,-0.16,-0.19,-0.19,-0.08,-0.06,-0.35,-0.22,-0.23,-0.35,-0.52,0.06,-0.46,-0.04,-0.27,-0.03,-0.18,-0.2,-0.17,-0.12,-0.32,0.14,-0.1,-0.3,-0.19,0.15,-0.32,0.28,-0.22,-0.17,-0.18,-0.11,-0.26,-0.25,-0.26,-0.29,-0.23,-0.16,-0.11,-0.25,-0.29,-0.18,-0.04,-0.04,-0.03,-0.31,-0.22,-0.09,-0.29,-0.3,-0.12,-0.07,-0.06,-0.14,0.11,-0.17,-0.27,-0.48,-0.22,-0.49,-0.34,-0.51,-0.14,0.12,-0.01,-0.15,-0.23,0.0,-0.04,0.17,0.1,-0.23,-0.01,-0.15,-0.01,0.1,-0.17,-0.04,-0.05,-0.07,0.31,-0.01,0.09,-0.12,0.15,0.01,0.27,0.34,0.16,0.38,0.33,0.56,0.56,0.47,0.5,0.54,0.55,0.47,0.54,0.58,0.71,0.82,0.6,0.3,0.83,0.45,0.52,0.67,0.69,1.1,0.12,0.79,0.76,0.66,0.6,0.45,0.58,0.64,0.55,0.49,0.74,0.58,0.54,0.43,0.59,0.38,0.48,0.43,0.4,0.13,0.45,0.27,0.45,0.35,0.51,0.45,0.58,0.47,0.78,0.83,1.1,0.45,0.86,0.62,0.49,0.55,0.74,0.93,0.72,0.68,0.9,0.81,0.59,0.67,0.58,0.6,0.48,0.4,0.63,0.79,0.54,0.77,0.74,0.84,0.48,0.5,0.54,0.84,0.6,0.48,0.46,0.4,0.8,0.54,0.45,0.38,0.34,0.62,0.59,0.69,0.18,0.23,0.43,0.47,0.44,0.37,0.2,0.1,0.44,-0.03,0.07,0.1,0.14,0.07,-0.02,0.21,-0.06,-0.01,0.0,0.18,-0.07,0.38,0.12,-0.12,-0.03,0.02,0.01,-0.06,0.21,0.0,-0.03,-0.15,0.21,-0.17,0.0,-0.13,-0.11,-0.09,0.3,0.14,0.16,0.24,0.01,0.17,0.15,0.06,0.45,-0.19,0.22,0.34,0.42,0.24,-0.08,-0.08,-0.19,0.34,-0.23,-0.12,0.12,0.34,0.05,0.02,-0.2,-0.33,-0.12,0.01,0.14,-0.03,-0.26,-0.11,-0.15,-0.31,0.06,-0.15,0.07,0.01,-0.08,-0.2,-0.14,0.04,0.07,-0.13,0.31,-0.25,-0.26,-0.33,-0.11,-0.51,0.04,-0.29,-0.28,-0.14,-0.47,-0.25,-0.34,-0.25,-0.4,-0.37,-0.46,-0.17,-0.22,-0.31,-0.14,0.13,-0.35,-0.03,-0.21,0.03,-0.09,0.11,0.24,0.17,0.0,0.02,0.03,0.03,0.13,0.0,-0.07,0.06,0.33,0.06,0.35,0.01,0.34,0.25,0.12,0.21,0.05,0.45,0.16,0.19,-0.03,0.03,0.04,-0.3,-0.24,-0.09,-0.17,-0.26,-0.08,-0.01,-0.15,0.1,0.1,0.19,0.01,0.2,0.16,-0.11,0.23,0.16,0.11,-0.14,-0.02,-0.11,-0.11,-0.04,0.19,-0.04,0.4,0.03,-0.38,0.06,0.24,0.4,-0.12,-0.05,-0.28,-0.34,-0.17,0.07,-0.08,0.08,0.0,-0.31,-0.15,-0.36,-0.33,-0.33,-0.12,-0.31,0.0,-0.08,0.03,-0.03,0.13,0.0,-0.27,-0.05,-0.35,-0.33,0.07,-0.38,-0.11,-0.19,-0.29,0.03,-0.6,-0.31,-0.19,-0.03,-0.06,-0.18,-0.15,-0.07,0.26,-0.07,-0.16,-0.16,0.18,0.1,-0.44,-0.27,-0.41,-0.38,-0.11,-0.11,-0.06,-0.17,-0.12,0.06,-0.01,-0.24,-0.39,-0.3,-0.22,-0.27,-0.5,-0.46,-0.25,-0.18,-0.1,-0.16,-0.32,0.0,0.06,-0.09,-0.26,-0.38,-0.15,-0.04,-0.33,-0.19,-0.07,-0.35,0.06,-0.07,0.04,0.12,-0.01,0.11,0.13,0.37,0.4,0.16,0.32,0.43,0.34,0.66,0.83,0.1,0.37,0.47,0.38,0.28,0.42,0.36,0.58,0.2,0.45,0.34,0.31,0.26,0.15,0.33,0.02,0.2,0.16,-0.06,-0.07,0.06,-0.05,0.08,-0.01,0.1,-0.3,-0.06,-0.03,0.46,0.33,0.48,0.13,0.42,0.18,0.1,0.28,0.03,0.0,0.3,-0.07,-0.01,0.28,0.37,0.08,0.51,0.19,0.1,0.16,0.43,-0.04,0.28,0.17,0.06,-0.03,0.06,0.19,0.4,-0.1,-0.03,-0.24,0.14,0.18,0.14,0.11,0.36,0.2,0.0,0.12,0.15,0.13,-0.02,0.06,0.08,0.06,0.21,0.32,0.25,0.39,0.17,0.35,0.1,0.22,0.08,0.06,0.22,0.33,-0.08,0.42,0.21,0.57,-0.06,0.65,0.1,-0.01,0.08,-0.11,0.1,-0.03,0.23,-0.06,0.28,0.12,0.09,-0.07,-0.1,0.04,0.14,0.24,0.3,-0.01,0.11,0.2,-0.12,-0.07,0.4,-0.06,0.04,0.05,0.06,0.29,0.3,0.08,0.12,0.23,0.06,0.53,0.23,0.25,0.42,0.1,0.33,0.0,0.17,0.17,0.17,0.18,0.0,-0.06,0.0,0.04,-0.32,-0.18,-0.37,-0.28,-0.28,-0.38,-0.54,-0.23,-0.35,0.02,-0.36,-0.29,-0.22,-0.42,-0.2,-0.31,-0.14,0.09,-0.16,-0.21,-0.13,-0.19,-0.26,-0.12,0.01,-0.1,0.13,0.25,-0.13,0.16,-0.04,0.2,0.12,-0.3,0.06,-0.2,-0.36,-0.37,0.04,-0.07,-0.08,-0.17,-0.15,-0.14,-0.14,0.21,0.19,-0.08,-0.11,0.14,0.26,0.06,0.16,0.01,-0.01,0.17,0.07,0.1,-0.08,-0.07,-0.1,0.0,-0.17,0.05,-0.13,0.1,-0.22,0.47,0.39,0.29,0.32,0.22,0.19,0.23,0.08,0.28,0.12,0.37,0.36,0.37,0.4,0.08,0.39,0.18,0.37,0.1,0.13,0.17,0.1,0.53,0.37,0.41,0.57,0.59,0.43,0.3,0.43,0.45,0.27,0.22,0.25,0.26,0.18,0.2,-0.05,0.39,-0.03,0.13,-0.15,0.24,-0.16,-0.03,-0.3,0.01,0.07,-0.41,-0.28,-0.2,-0.32,-0.13,-0.24,-0.49,0.01,0.44,0.06,-0.02,-0.05,-0.25,0.19,-0.38,0.35,0.01,-0.12,-0.03,-0.17,0.01,-0.19,0.04,-0.13,-0.18,-0.19,0.04,0.29,-0.2,0.35,-0.24,0.59,-0.08,0.1,-0.11,-0.18,0.06,0.06,0.25,0.12,0.09,0.18,0.35,0.16,0.4,-0.07,0.11,0.2,0.0,-0.17,-0.09,0.15,0.11,-0.09,0.04,-0.01,-0.05,-0.11,-0.23,-0.37,0.15,0.04,-0.12,-0.03,-0.34,-0.27,0.37,0.12,0.06,0.3,0.04,0.18,0.2,-0.1,0.08,0.13,0.13,0.33,0.27,0.0,0.2,0.13,0.1,-0.04,-0.11,0.06,0.02,0.24,-0.01,-0.13,-0.06,-0.03,-0.21,0.0,0.08,-0.19,-0.11,-0.1,-0.03,-0.22,-0.17,0.06,-0.14,0.26,0.0,0.13,0.1,0.13,0.37,-0.04,0.0,0.15,-0.21,0.17,0.02,0.12,0.1,0.04,0.25,0.06,0.2,-0.1,-0.23,-0.18,0.25,0.0,-0.02,-0.03,0.09,0.06,0.18,0.3,0.35,0.05,0.48,0.46,-0.11,0.12,0.01,0.04,0.08,-0.35,-0.14,-0.23,-0.09,-0.25,-0.2,-0.33,-0.31,-0.39,-0.36,-0.35,-0.22,-0.04,-0.37,-0.02,-0.22,-0.35,-0.29,-0.21,-0.09,-0.29,-0.34,-0.21,-0.23,0.06,0.27,0.37,0.41,0.27,0.45,0.11,0.16,-0.11,0.36,0.4,0.03,0.25,0.57,-0.1,-0.04,0.02,-0.03,0.27,-0.01,0.1,-0.03,-0.12,-0.01,0.31,-0.16,-0.3,0.02,-0.05,-0.02,0.15,-0.06,-0.11,-0.09,-0.06,-0.22,-0.19,-0.22,0.17,-0.08,0.15,0.15,-0.04,-0.19,0.12,-0.08,-0.04,-0.14,-0.29,-0.19,0.06,-0.26,-0.04,-0.27,-0.34,-0.11,-0.27,-0.01,-0.39,-0.14,-0.43,-0.4,-0.17,-0.38,-0.23,-0.34,-0.06,-0.48,-0.09,-0.28,-0.49,-0.34,-0.27,-0.24,-0.32,-0.02,-0.26,-0.34,-0.05,-0.14,-0.14,0.13,0.17,-0.05,-0.3,-0.33,-0.1,-0.34,0.19,-0.09,-0.04,-0.24,-0.29,-0.12,-0.21,-0.33,0.08,-0.05,-0.41,-0.13,-0.15,-0.31,-0.06,0.09,-0.36,-0.29,-0.44,-0.1,-0.35,-0.26,-0.29,-0.21,-0.37,-0.18,-0.22,-0.26,-0.46,-0.22,-0.43,-0.16,-0.28,-0.11,-0.39,-0.13,-0.2,0.0,-0.14,-0.25,-0.28,-0.41,-0.44,-0.19,0.01,-0.22,-0.57,-0.28,-0.22,-0.22,-0.29,-0.05,-0.26,-0.21,-0.01,-0.35,-0.13,-0.26,-0.31,-0.2,0.0,-0.16,-0.35,-0.29,-0.31,-0.34,-0.2,-0.14,0.07,-0.24,-0.26,-0.46,-0.62,-0.54,-0.6,-0.5,-0.2,-0.4,-0.37,-0.52,-0.34,-0.33,-0.18,-0.63,-0.25,-0.43,-0.45,-0.15,0.06,-0.18,-0.24,0.3,-0.22,-0.07,-0.05,0.21,-0.12,-0.12,0.02,-0.2,0.14,0.31,0.14,0.13,0.08,0.12,0.15,-0.05,0.45,-0.01,-0.12,-0.1,0.12,-0.03,-0.03,-0.25,-0.13,-0.26,-0.13,-0.06,0.52,0.38,-0.09,0.04,0.16,0.09,0.19,-0.11,0.0,-0.05,-0.19,-0.07,-0.3,-0.23,-0.24,0.15,-0.29,0.04,-0.26,-0.02,0.01,0.32,0.16,0.14,0.03,0.06,-0.15,-0.16,-0.22,-0.03,-0.33,0.25,-0.44,-0.01,-0.21,-0.13,-0.46,-0.18,-0.26,-0.09,-0.23,-0.43,-0.18,-0.31,0.04,-0.03,-0.23,0.12,-0.03,-0.29,-0.23,-0.02,0.03,-0.2,-0.41,-0.1,-0.14,-0.32,0.02,-0.04,-0.16,-0.36,-0.38,-0.3,0.11,-0.21,0.01,0.16,-0.07,-0.33,-0.17,-0.16,0.06,-0.31,-0.04,-0.18,-0.21,-0.2,-0.22,-0.31,-0.32,-0.04,-0.41,-0.17,-0.31,-0.14,-0.03,-0.13,-0.17,0.05,0.24,0.0,-0.2,0.17,0.01,0.15,0.35,0.39,-0.09,0.15,0.27,0.23,-0.06,-0.04,0.04,0.02,0.11,0.2,-0.24,-0.09,-0.14,0.38,0.16,0.12,0.35,0.05,0.31,0.26,0.37,0.24,0.19,0.38,0.03,0.18,0.13,0.23,0.04,0.17,0.13,0.14,0.01,0.08,-0.01,0.03,0.01,0.22,-0.01,-0.04,-0.32,-0.21,-0.13,-0.27,-0.23,-0.34,-0.16,0.06,-0.19,-0.32,-0.1,0.18,-0.01,0.02,0.01,-0.23,0.03,0.3,-0.38,0.04,-0.09,-0.07,-0.32,-0.46,-0.35,-0.1,-0.35,-0.21,-0.27,-0.18,-0.12,-0.3,0.09,0.05,-0.21,-0.24,-0.44,-0.33,-0.49,0.05,-0.4,-0.61,-0.25,-0.41,-0.55,-0.15,-0.43,-0.13,-0.03,-0.04,-0.27,-0.22,-0.27,-0.35,-0.04,0.11,0.09,-0.14,0.09,-0.19,-0.37,-0.16,-0.13,0.3,0.18,0.14,-0.26,-0.19,-0.1,-0.13,0.06,-0.23,0.44,0.03,-0.18,0.06,0.35,-0.1,-0.14,-0.24,0.13,0.21,-0.25,-0.1,-0.35,-0.15,-0.01,0.28,-0.25,-0.08,-0.06,-0.11,-0.14,-0.41,0.08,-0.28,-0.23,-0.19,-0.03,0.17,-0.39,-0.28,0.06,-0.03,0.17,0.23,-0.04,-0.12,-0.09,0.4,-0.01,0.0,0.07,0.05,-0.03,-0.06,0.02,0.33,0.13,-0.08,0.2,0.78,0.18,-0.12,0.29,0.11,0.26,0.17,0.2,0.41,0.36,0.1,0.46,0.04,0.44,0.42,0.31,-0.05,0.24,0.12,-0.18,0.33,-0.2,0.22,0.14,0.48,0.47,0.27,-0.03,-0.04,-0.04,0.09,-0.05,0.0,0.33,0.23,0.03,-0.04,0.04,0.15,0.13,0.1,0.31,0.24,0.05,0.06,0.01,0.2,0.03,0.07,0.34,-0.03,0.32,0.39,0.29,0.08,0.12,0.01,0.2,0.25,0.08,-0.08,0.06,0.21,0.09,0.05,0.46,0.22,0.12,0.4,0.29,0.22,0.1,0.36,0.3,0.55,0.25,0.41,0.21,0.78,0.31,0.3,0.28,0.37,0.34,0.35,0.14,0.25,0.32,0.45,0.15,0.16,0.1,0.19,0.28,0.52,0.25,0.42,0.51,0.24,0.54,0.1,0.11,0.1,0.3,0.57,0.2,0.13,-0.05,0.06,0.19,0.55,0.06,0.08,-0.08,-0.1,0.17,0.1,0.17,0.12,0.48,-0.14,0.04,0.05,-0.05,0.14,-0.17,0.12,0.35,0.45,0.04,0.03,0.04,0.18,0.19,0.03,0.09,0.17,-0.15,-0.03,-0.19,-0.03,-0.26,-0.49,-0.29,-0.22,-0.6,-0.1,-0.31,-0.59,-0.14,0.06,-0.26,-0.34,-0.16,-0.16,-0.05,-0.3,-0.22,0.09,-0.01,-0.14,-0.29,-0.06,-0.06,0.18,0.27,-0.11,0.25,-0.25,-0.03,-0.14,-0.06,-0.01,-0.14,0.14,-0.04,0.02,0.04,0.15,0.56,0.09,0.34,-0.07,0.31,0.27,0.48,0.35,0.4,0.29,0.48,0.32,0.14,0.37,0.2,0.46,-0.05,0.4,0.28,0.46,0.36,0.23,0.19,0.11,0.18,0.36,0.08,0.23,0.1,0.29,0.14,-0.04,0.03,0.04,0.33,0.34,-0.03,0.13,0.07,-0.15,0.22,0.08,0.16,-0.07,0.03,0.23,0.1,0.16,0.15,0.04,0.25,0.39,0.64,0.35,0.3,0.25,0.16,0.22,0.33,0.37,0.33,0.5,0.7,0.64,0.22,0.27,0.61,0.39,0.39,0.46,0.45,0.36,0.63,0.48,0.42,0.39,0.46,0.5,0.72,0.18,0.36,0.21,0.3,0.3,0.12,0.27,0.39,0.29,0.29,0.06,0.17,-0.07,0.04,0.06,0.15,0.14,-0.24,0.17,0.17,0.1,0.14,-0.18,-0.41,-0.28,-0.03,-0.1,-0.14,0.05,-0.2,-0.15,-0.09,-0.06,0.18,-0.06,0.31,0.0,-0.03,0.22,0.2,0.41,0.16,0.3,0.04,-0.05,0.28,0.18,0.16,0.26,0.27,0.24,0.42,0.52,0.15,0.28,0.34,0.59,0.18,0.32,0.25,0.03,0.27,0.1,0.31,0.34,0.61,0.37,0.08,0.09,0.03,-0.03,0.1,0.24,-0.41,0.19,-0.22,-0.38,0.14,0.04,-0.21,-0.28,-0.1,0.12,0.1,0.02,0.18,-0.18,0.16,0.2,0.13,0.13,0.09,0.31,0.25,0.25,0.2,-0.12,-0.02,0.07,0.35,-0.13,-0.11,0.01,-0.4,0.03,-0.16,-0.34,-0.23,-0.2,-0.07,-0.31,-0.47,-0.33,-0.01,-0.66,-0.2,-0.56,-0.65,-0.48,-0.43,-0.53,-0.25,-0.55,-0.22,-0.19,-0.09,-0.06,-0.21,-0.17,-0.11,-0.26,-0.21,-0.21,0.08,0.17,0.18,-0.22,-0.09,0.18,0.5,0.14,0.07,-0.02,0.04,0.11,0.1,-0.01,-0.11,0.17,0.17,0.35,0.69,0.24,0.04,0.15,0.29,0.19,0.0,0.37,0.47,0.24,0.74,0.2,0.25,0.51,0.23,0.61,0.32,0.49,0.43,0.59,0.25,0.55,0.42,-0.17,0.68,0.34,0.34,0.39,0.31,0.13,0.36,0.18,0.24,0.08,0.38,0.49,0.35,0.28,0.45,0.83,0.26,0.51,0.21,0.33,0.48,0.3,0.28,0.63,0.41,0.41,0.7,0.29,0.91,0.23,0.31,0.76,0.46,0.69,0.7,0.57,0.34,0.42,0.52,0.15,0.42,0.76,0.49,0.48,0.81,0.48,0.57,0.74,0.96,0.94,1.04,0.5,0.42,0.51,0.54,0.8,0.83,0.58,0.74,0.99,0.7,0.7,0.58,0.63,0.62,0.53,0.43,0.66,0.37,0.74,0.57,0.49,0.44,0.48,0.54,0.32,0.46,0.58,0.57,0.34,0.31,0.41,0.41,0.46,0.48,0.46,0.5,0.06,0.34,0.13,0.16,0.19,0.03,0.5,0.43,0.66,0.36,0.1,0.02,0.13,-0.01,0.27,0.15,0.21,0.23,0.34,0.2,0.62,0.61,0.44,0.66,0.57,0.65,0.49,0.35,0.46,0.39,0.6,0.25,0.33,0.32,0.09,0.21,0.36,0.49,0.58,0.38,0.62,0.38,0.29,0.24,0.82,0.46,0.65,0.54,0.3,0.16,0.46,0.26,0.4,0.19,0.11,0.12,0.29,0.14,0.21,0.3,0.24,0.02,0.21,0.0,0.1,0.05,0.09,-0.11,0.16,-0.04,-0.09,0.04,0.19,0.27,0.35,-0.25,0.0,0.13,-0.17,0.02,0.16,-0.11,0.1,0.11,0.27,-0.16,-0.02,-0.08,-0.18,-0.08,-0.09,-0.07,-0.27,-0.3,-0.34,-0.16,-0.16,-0.27,-0.28,-0.25,-0.18,-0.33,0.0,-0.04,0.05,0.1,0.22,-0.1,-0.17,-0.08,-0.06,0.11,-0.07,0.04,0.26,-0.07,0.09,0.14,0.29,0.0,0.24,0.24,0.2,0.32,0.18,0.48,0.15,0.39,0.37,0.31,0.47,0.37,0.09,0.09,0.06,0.27,0.01,0.1,0.06,-0.16,-0.03,0.4,0.12,0.31,0.15,0.31,-0.01,-0.11,0.12,0.2,-0.08,0.17,0.02,-0.17,-0.25,0.06,-0.14,0.1,-0.13,-0.29,-0.16,-0.12,-0.16,-0.26,-0.38,-0.21,-0.26,-0.08,-0.21,-0.22,-0.06,-0.07,-0.41,-0.25,-0.21,-0.12,0.02,-0.15,0.12,0.12,-0.22,-0.12,-0.17,-0.08,-0.16,-0.12,-0.01,-0.11,0.14,-0.23,-0.11,0.12,-0.01,0.21,0.15,-0.18,0.37,-0.06,0.09,-0.15,0.56,0.29,0.47,0.32,-0.05,0.35,0.21,0.12,0.13,0.1,0.12,0.22,0.18,0.18,0.08,0.31,0.14,0.11,-0.05,0.12,0.09,0.24,0.35,0.16,-0.09,0.28,0.26,0.19,0.55,0.15,0.19,0.18,0.09,0.57,0.19,0.31,0.45,0.42,0.29,0.35,0.59,0.23,0.45,0.33,0.16,0.25,0.54,0.16,-0.03,0.07,0.56,0.18,0.23,0.15,0.39,0.22,0.08,0.42,0.12,0.29,0.38,0.21,0.08,0.32,0.04,-0.17,0.27,0.04,0.32,0.0,0.06,0.25,-0.11,0.19,0.04,0.52,0.04,-0.11,0.07,0.07,-0.12,0.08,-0.06,0.1,0.15,0.18,-0.18,-0.1,-0.23,0.02,0.0,0.13,-0.03,0.1,0.04,-0.07,0.15,0.08,0.14,0.33,0.09,0.06,0.38,0.06,-0.09,0.04,0.22,0.32,0.19,0.12,0.12,0.03,0.0,0.18,0.32,0.04,0.09,0.22,0.17,0.14,0.36,0.02,-0.04,0.06,0.08,0.41,0.19,0.26,0.27,0.31,0.35,0.49,0.22,0.2,0.35,0.87,0.29,0.65,0.28,0.5,0.54,0.36,0.41,0.46,0.65,0.78,0.26,0.43,0.48,0.24,0.35,0.4,0.41,0.04,0.26,0.19,0.27,0.51,0.46,0.45,0.18,0.29,0.58,0.23,0.31,-0.06,0.61,0.25,0.26,0.52,0.28,0.46,0.73,0.08,0.39,0.22,0.49,0.49,0.77,0.35,0.28,0.24,0.3,0.69,0.2,0.27,0.42,0.17,0.04,0.22,0.16,0.21,-0.1,-0.1,-0.12,-0.06,-0.05,-0.06,0.15,-0.39,-0.08,-0.07,-0.24,-0.11,-0.32,-0.24,-0.07,-0.41,-0.39,-0.11,-0.05,-0.21,-0.26,-0.15,-0.01,-0.03,-0.07,-0.06,-0.03,-0.22,-0.18,0.08,0.06,0.19,0.04,0.14,-0.09,-0.37,0.04,-0.01,0.16,-0.1,-0.19,0.09,0.2,0.13,0.53,0.26,0.04,0.23,-0.04,0.24,0.36,0.17,-0.2,0.29,-0.01,-0.05,0.23,0.16,0.13,0.28,0.26,0.13,0.43,0.02,0.17,0.27,0.15,0.17,0.02,-0.03,0.33,-0.09,-0.03,0.4,0.2,0.01,0.0,0.19,0.25,0.1,0.33,0.17,0.26,0.49,0.2,0.22,0.54,0.44,0.33,0.29,0.17,0.26,0.61,0.31,0.45,0.46,0.3,0.35,0.46,0.56,0.36,0.56,0.52,0.48,0.4,0.78,0.51,0.38,0.13,0.4,0.63,0.47,0.5,0.53,0.55,0.83,0.73,0.93,0.56,0.47,0.7,0.78,0.89,1.06,0.67,0.66,0.82,0.81,0.76,0.56,0.56,0.72,0.61,0.54,0.65,0.71,0.47,0.57,0.69,0.76,0.68,0.39,0.68,0.53,0.51,0.88,0.54,0.53,0.6,0.44,0.56,0.34,0.64,0.63,0.39,0.46,0.39,0.44,0.73,0.8,0.66,0.67,0.59,0.57,0.43,0.38,0.59,0.51,0.66,0.7,0.86,0.33,0.34,0.48,0.16,0.31,0.28,0.42,0.28,0.74,0.25,0.5,0.38,0.27,0.52,0.58,0.4,0.33,0.27,0.5,0.45,0.3,0.66,0.32,0.52,0.54,0.6,0.25,0.42,0.25,0.25,0.18,0.45,0.25,-0.1,0.08,-0.03,0.31,-0.03,-0.06,0.23,0.22,0.18,-0.19,-0.26,-0.37,0.1,-0.27,-0.15,-0.11,-0.35,-0.25,-0.19,0.07,-0.17,-0.25,0.0,-0.17,0.13,0.46,0.23,-0.17,0.03,0.23,-0.19,0.21,0.14,-0.03,0.23,0.08,0.29,0.21,0.45,-0.01,0.21,-0.02,0.14,0.41,0.35,0.18,-0.07,0.19,0.36,0.14,0.18,0.08,-0.05,-0.13,0.01,-0.04,0.08,0.01,-0.05,-0.25,0.28,-0.26,-0.02,-0.28,-0.22,-0.21,-0.29,-0.07,-0.44,-0.46,-0.41,0.14,-0.07,-0.09,0.27,-0.12,-0.17,-0.01,-0.29,-0.09,-0.17,-0.17,-0.26,-0.25,0.07,-0.16,0.02,0.04,0.06,-0.36,-0.36,0.1,-0.37,-0.17,-0.29,-0.31,-0.39,-0.07,-0.33,-0.43,-0.45,-0.06,-0.13,-0.45,-0.19,-0.32,-0.03,0.0,-0.27,-0.19,-0.41,-0.27,-0.28,-0.07,-0.19,-0.24,0.15,0.04,-0.45,-0.03,-0.21,-0.01,-0.01,-0.13,-0.18,-0.34,-0.01,0.06,0.02,-0.24,-0.05,-0.18,-0.19,-0.02,-0.31,-0.26,-0.11,-0.1,-0.13,-0.19,-0.07,0.29,-0.07,0.27,-0.17,0.07,-0.05,-0.2,0.14,-0.07,-0.14,0.0,-0.06,-0.03,0.01,-0.22,-0.14,-0.1,0.06,-0.1,-0.16,-0.08,0.2,-0.24,0.11,-0.07,0.12,-0.06,0.19,-0.01,-0.29,-0.38,-0.07,0.06,0.05,-0.26,-0.06,-0.38,-0.25,-0.07,-0.07,-0.32,-0.34,-0.13,-0.09,-0.17,-0.37,-0.57,-0.19,0.01,-0.1,-0.19,0.22,0.34,-0.04,-0.22,0.15,-0.1,-0.06,-0.3,0.22,0.05,-0.17,0.03,0.19,0.09,-0.09,-0.04,0.1,-0.16,-0.26,-0.18,0.09,-0.22,-0.36,0.01,-0.46,-0.23,-0.17,-0.26,-0.2,-0.49,-0.44,-0.08,-0.39,-0.12,-0.32,-0.27,-0.45,-0.31,-0.16,-0.26,-0.33,-0.12,-0.35,0.15,-0.33,-0.02,0.04,-0.04,-0.07,0.07,0.22,0.27,-0.2,-0.1,-0.04,0.0,0.01,0.06,0.17,-0.09,0.09,0.2,0.07,0.43,-0.03,0.41,-0.14,0.17,0.04,-0.27,-0.16,0.03,-0.38,-0.14,-0.03,0.02,0.04,-0.32,-0.21,0.09,-0.18,-0.27,0.36,-0.02,-0.05,0.19,0.07,-0.04,-0.11,0.15,-0.14,0.14,0.01,0.23,0.0,-0.2,-0.24,-0.11,0.21,0.07,-0.15,0.05,0.02,-0.23,0.13,-0.09,0.28,0.09,0.3,0.06,-0.22,-0.17,-0.09,0.02,-0.13,-0.23,0.04,-0.29,-0.25,-0.33,-0.1,-0.51,-0.4,-0.45,-0.17,-0.31,-0.05,-0.43,-0.05,0.08,-0.16,0.34,-0.04,-0.15,-0.12,-0.08,0.08,-0.26,0.01,0.14,0.13,-0.15,-0.22,-0.09,-0.15,-0.08,0.13,0.12,0.24,-0.23,0.15,0.06,0.06,0.12,-0.08,0.12,0.25,0.35,0.02,0.06,0.08,0.17,-0.19,0.12,0.18,-0.1,-0.12,-0.24,-0.01,-0.21,-0.15,-0.09,0.0,0.13,0.1,-0.01,-0.07,-0.08,0.03,0.12,0.29,0.01,-0.05,0.13,0.38,0.01,0.05,-0.2,-0.13,-0.21,-0.15,0.14,-0.03,0.1,0.02,0.06,-0.09,-0.29,-0.06,-0.22,0.08,0.16,-0.1,0.21,0.06,-0.2,-0.28,-0.08,-0.06,0.05,-0.16,-0.25,-0.05,-0.06,-0.22,-0.33,-0.25,-0.34,-0.16,-0.3,-0.16,0.03,-0.18,0.04,-0.12,-0.27,0.0,0.2,-0.21,-0.3,-0.09,0.02,-0.05,0.15,0.11,0.17,0.01,0.13,0.41,0.08,-0.05,0.13,0.31,0.18,0.05,0.18,0.32,0.06,0.22,0.12,0.24,0.36,0.28,0.33,0.22,0.35,0.87,0.27,0.24,0.29,0.36,0.36,0.46,0.3,-0.03,0.03,-0.19,0.14,0.01,0.1,0.09,-0.13,-0.11,-0.08,-0.31,0.2,0.03,-0.22,0.04,0.05,-0.25,-0.14,-0.34,-0.26,-0.35,-0.46,-0.35,-0.09,-0.33,-0.1,0.13,-0.18,-0.09,-0.11,-0.45,-0.23,-0.57,-0.44,-0.5,-0.57,-0.29,-0.12,-0.47,-0.22,-0.13,-0.3,0.08,0.0,-0.17,-0.2,0.1,-0.44,-0.35,-0.4,0.01,-0.33,-0.25,-0.2,-0.52,-0.45,-0.3,-0.18,-0.57,-0.44,-0.5,-0.55,-0.22,-0.22,-0.14,-0.33,-0.34,-0.06,-0.29,-0.21,-0.26,-0.24,-0.09,0.18,-0.22,-0.32,-0.29,-0.3,0.04,-0.27,-0.18,-0.2,-0.1,-0.28,-0.3,-0.03,-0.06,-0.28,-0.16,-0.07,-0.03,-0.06,-0.15,-0.16,-0.36,0.26,-0.19,0.17,-0.06,0.06,-0.11,-0.12,0.06,-0.2,0.06,-0.1,0.22,-0.33,-0.25,-0.25,0.02,-0.18,-0.07,0.04,-0.12,0.05,-0.26,0.15,-0.22,0.05,0.2,-0.03,-0.2,-0.02,-0.28,-0.23,-0.03,0.09,-0.05,-0.14,-0.09,-0.42,-0.17,-0.23,-0.08,0.03,-0.38,-0.08,-0.36,-0.33,0.11,-0.3,-0.41,-0.22,-0.31,-0.32,-0.08,0.11,-0.16,-0.41,-0.01,0.02,-0.07,-0.3,-0.45,0.18,0.2,-0.15,-0.36,-0.3,-0.21,-0.32,0.08,0.06,-0.09,-0.08,-0.1,0.37,0.02,0.02,-0.26,-0.21,0.03,-0.07,-0.09,-0.13,0.01,-0.14,0.15,-0.06,-0.18,0.02,0.03,-0.21,-0.26,0.26,0.19,-0.1,0.16,0.29,-0.1,-0.1,0.13,-0.01,0.03,-0.05,-0.06,-0.35,-0.28,-0.21,-0.25,0.16,-0.03,0.02,0.0,0.24,0.03,-0.08,-0.15,0.06,0.29,0.07,0.21,0.02,0.06,-0.01,-0.05,0.02,-0.28,0.08,-0.03,0.16,-0.03,0.08,-0.07,-0.05,0.17,0.14,-0.11,-0.01,0.17,0.18,-0.11,-0.07,0.1,0.01,0.08,0.36,0.19,0.24,0.14,-0.01,-0.09,0.03,0.17,0.1,0.08,0.45,0.08,0.03,0.18,-0.08,0.26,0.19,0.18,0.12,0.24,0.28,0.34,0.21,0.12,0.25,-0.01,0.43,0.06,0.26,0.29,0.04,-0.03,-0.07,0.09,0.08,0.04,0.23,-0.16,-0.31,-0.09,0.08,-0.08,0.12,0.12,0.29,-0.06,0.42,-0.1,0.43,0.31,0.36,0.11,-0.04,0.02,-0.05,0.21,0.4,-0.05,0.19,-0.07,-0.03,0.23,0.44,0.15,0.11,-0.01,0.21,0.0,-0.35,-0.14,0.15,-0.09,0.38,-0.28,-0.16,-0.16,-0.35,-0.29,0.17,-0.26,-0.31,0.19,0.31,0.03,0.42,0.29,0.32,0.3,0.16,0.13,0.58,0.17,0.06,-0.17,0.11,0.1,0.26,0.17,0.39,0.31,0.22,0.2,0.05,0.03,-0.04,-0.05,0.04,0.34,-0.11,0.38,0.22,0.16,-0.08,0.51,0.19,0.17,0.31,0.06,-0.06,0.52,0.29,0.08,0.1,-0.22,0.26,-0.15,-0.01,0.26,-0.26,0.3,-0.31,-0.09,-0.03,-0.1,-0.08,-0.45,-0.14,-0.17,0.1,0.08,-0.18,-0.08,-0.29,-0.15,-0.36,-0.16,-0.05,0.01,0.24,0.16,-0.16,0.29,0.0,-0.12,-0.09,-0.15,0.1,0.03,0.13,0.08,-0.11,0.22,0.03,0.13,0.46,0.39,0.33,0.47,0.21,0.32,0.23,0.13,0.2,0.49,0.11,0.09,0.15,0.27,0.53,-0.09,-0.11,-0.01,0.13,-0.22,0.09,0.15,-0.08,0.22,0.17,0.0,0.12,-0.06,0.26,0.43,0.08,0.22,0.1,-0.07,0.13,-0.09,-0.1,-0.05,-0.23,-0.07,-0.1,-0.1,-0.16,0.11,-0.23,0.23,0.2,-0.3,-0.11,0.18,-0.08,-0.16,-0.1,0.0,-0.25,0.01,-0.03,-0.02,-0.07,0.31,-0.1,0.04,-0.03,0.36,0.1,-0.11,-0.16,-0.05,-0.1,0.02,0.14,0.22,-0.07,0.25,0.02,0.31,0.22,0.32,0.07,0.14,0.1,-0.14,0.27,0.45,0.38,0.49,0.27,0.01,0.22,0.09,-0.13,0.27,0.44,0.09,0.04,-0.1,-0.14,-0.08,-0.1,-0.4,0.15,-0.23,-0.22,0.0,-0.17,-0.07,0.01,-0.12,-0.34,-0.37,-0.02,-0.31,-0.16,-0.17,0.06,-0.01,0.01,-0.38,-0.21,-0.14,0.31,-0.17,-0.24,-0.13,0.0,0.27,-0.01,-0.07,-0.11,-0.06,-0.12,0.13,0.17,-0.18,-0.22,0.0,0.19,0.27,0.0,0.68,-0.19,-0.15,-0.26,-0.06,-0.04,0.13,0.07,-0.04,-0.36,0.1,-0.12,0.04,0.19,0.53,0.16,0.43,0.29,0.09,0.01,0.04,0.1,0.1,0.06,0.21,0.17,-0.08,0.13,0.2,-0.02,-0.01,0.12,0.18,-0.04,-0.06,0.04,-0.05,-0.16,0.01,0.02,0.04,-0.1,-0.23,-0.23,-0.2,-0.26,0.35,-0.06,-0.2,-0.04,-0.12,-0.22,0.1,0.06,-0.17,-0.28,0.43,-0.14,0.01,-0.01,-0.03,0.14,-0.01,0.01,-0.05,0.14,0.11,0.24,0.42,0.13,0.21,0.16,0.42,0.07,0.04,-0.02,0.48,0.38,0.23,0.17,0.19,0.04,0.32,0.06,0.46,0.11,0.68,0.02,-0.06,-0.24,-0.07,0.09,0.08,-0.01,0.05,0.01,-0.06,0.13,-0.12,-0.17,0.09,0.27,0.2,-0.06,-0.26,-0.09,-0.1,-0.01,0.27,0.0,-0.18,0.16,-0.16,-0.19,0.06,0.14,0.16,0.13,0.03,0.32,0.44,0.18,-0.08,0.39,0.27,0.25,0.0,-0.1,-0.28,0.15,-0.05,0.11,-0.21,0.2,0.03,0.01,0.14,-0.04,0.06,-0.24,-0.09,-0.04,0.17,0.13,0.04,0.04,-0.04,-0.14,-0.12,-0.03,-0.19,-0.09,0.1,-0.06,-0.01,0.09,-0.09,-0.25,-0.25,-0.34,-0.05,-0.2,-0.13,-0.32,-0.37,-0.48,-0.2,-0.38,-0.36,-0.08,-0.16,0.1,-0.18,-0.31,-0.13,0.11,0.08,-0.03,0.16,0.1,0.1,-0.26,0.07,-0.18,-0.1,-0.11,-0.21,-0.25,0.2,-0.12,-0.22,-0.4,-0.1,-0.14,-0.12,-0.03,-0.18,-0.34,-0.14,-0.2,-0.07,-0.16,-0.03,0.0,-0.28,-0.22,-0.25,-0.15,-0.24,-0.08,-0.18,-0.3,-0.22,-0.28,-0.41,-0.21,-0.32,-0.17,-0.28,-0.14,-0.16,-0.27,-0.25,0.09,0.14,-0.28,-0.23,-0.01,-0.26,-0.01,0.01,-0.22,-0.45,-0.11,-0.39,-0.1,-0.36,-0.3,-0.28,-0.19,-0.33,0.15,-0.37,-0.28,-0.03,-0.15,-0.36,0.03,-0.12,-0.08,0.03,0.22,-0.24,0.04,0.08,0.08,-0.21,-0.05,-0.02,-0.09,-0.14,0.0,0.04,-0.04,0.13,0.06,0.29,0.06,-0.05,-0.07,-0.33,0.21,-0.1,0.0,-0.05,-0.01,0.04,0.17,0.13,0.08,0.11,0.3,0.07,-0.08,0.39,0.39,0.22,0.38,0.19,-0.43,0.13,0.48,0.55,0.53,0.76,0.26,0.12,0.18,0.38,-0.05,0.28,0.22,0.17,0.25,-0.03,0.34,-0.08,-0.01,0.39,0.31,0.27,0.46,0.16,0.21,0.22,0.09,0.54,0.31,0.48,0.12,0.2,0.14,0.74,0.67,0.42,-0.02,0.07,0.01,-0.29,-0.16,-0.02,-0.12,0.01,-0.01,0.21,-0.26,-0.16,-0.18,-0.12,-0.06,-0.15,-0.22,-0.02,0.24,0.16,0.09,0.16,-0.06,0.14,0.29,-0.01,0.09,0.23,0.03,0.02,-0.3,0.21,-0.11,0.04,-0.38,-0.29,0.06,-0.25,0.08,-0.19,0.02,-0.03,0.0,0.35,-0.06,0.13,0.43,0.0,0.31,0.32,0.27,0.36,0.03,-0.19,0.01,-0.12,0.01,0.03,-0.02,-0.13,-0.16,0.27,-0.04,-0.35,-0.18,-0.25,-0.32,-0.36,-0.55,-0.1,-0.27,-0.37,-0.27,-0.24,-0.45,-0.35,-0.31,-0.29,0.0,-0.07,-0.18,-0.13,0.06,-0.17,-0.13,-0.1,0.22,-0.07,0.2,0.1,-0.04,0.01,-0.36,-0.19,-0.13,-0.05,0.07,-0.12,-0.24,-0.05,-0.26,-0.44,-0.23,-0.38,-0.14,0.04,-0.26,-0.02,-0.22,-0.08,-0.24,-0.34,-0.03,0.02,-0.11,-0.09,-0.24,-0.13,-0.57,-0.13,-0.1,-0.29,-0.45,0.1,-0.28,0.28,0.14,0.19,0.31,-0.04,0.2,0.12,0.35,0.13,0.3,0.4,0.06,-0.15,0.29,-0.05,-0.43,-0.15,-0.03,-0.13,0.14,-0.18,-0.18,0.16,-0.05,-0.17,0.17,-0.19,-0.19,0.18,0.08,-0.23,0.06,0.0,-0.3,-0.09,0.02,0.1,-0.02,-0.08,-0.04,-0.26,-0.21,-0.19,-0.2,-0.43,-0.2,-0.28,-0.28,-0.23,-0.25,-0.02,0.04,0.1,0.01,0.53,0.03,-0.1,0.07,0.04,-0.27,-0.23,0.0,-0.1,0.09,-0.09,0.11,0.08,-0.1,-0.04,-0.34,-0.31,0.04,-0.2,-0.23,-0.12,-0.05,-0.21,-0.25,0.0,-0.2,-0.12,-0.06,-0.18,-0.26,-0.05,0.06,-0.22,-0.19,-0.19,0.04,-0.4,0.0,0.03,0.02,0.17,-0.14,0.09,0.21,-0.01,-0.1,-0.07,0.04,-0.05,0.14,-0.28,-0.09,-0.32,0.1,0.05,0.21,-0.04,0.15,0.03,0.13,0.31,0.28,-0.01,0.09,0.18,0.25,0.19,0.13,0.09,-0.12,-0.02,0.45,-0.18,-0.16,0.27,-0.32,-0.16,0.02,-0.03,0.03,-0.11,-0.09,0.28,0.28,0.26,-0.17,-0.12,0.01,-0.14,-0.1,0.06,-0.1,0.09,-0.04,0.18,-0.06,-0.18,-0.04,-0.09,0.04,-0.06,0.62,0.29,0.05,0.1,0.37,0.1,-0.01,-0.01,-0.21,0.1,0.05,0.19,0.53,-0.1,0.26,-0.11,-0.01,0.57,0.3,0.06,0.19,0.32,0.01,0.49,-0.04,0.22,0.25,-0.2,-0.14,-0.15,0.03,-0.01,-0.16,0.13,-0.05,-0.05,-0.21,-0.16,-0.1,-0.03,-0.26,-0.13,-0.1,-0.2,0.13,-0.05,-0.15,0.26,-0.21,0.26,-0.03,0.3,-0.08,0.11,-0.25,-0.23,-0.03,-0.02,-0.1,-0.3,0.08,0.1,0.63,0.21,0.03,0.38,-0.15,0.37,0.4,0.34,0.17,0.15,0.48,0.45,0.04,0.04,0.02,0.0,0.15,-0.07,0.04,0.22,-0.03,-0.12,-0.05,-0.2,-0.17,-0.21,-0.09,0.01,0.04,0.04,0.11,-0.01,-0.15,0.23,0.27,-0.3,-0.28,0.08,-0.1,-0.2,-0.05,0.06,-0.34,0.16,0.08,-0.31,-0.29,-0.31,-0.18,-0.5,-0.28,-0.04,-0.27,-0.35,-0.16,-0.45,-0.17,-0.35,-0.25,-0.34,-0.42,0.1,-0.27,-0.48,-0.28,-0.53,-0.52,-0.54,-0.41,-0.45,-0.67,-0.26,-0.53,-0.22,-0.14,-0.09,-0.32,-0.34,-0.28,-0.24,0.12,0.28,0.04,0.12,0.22,0.18,0.18,0.3,0.47,0.39,0.35,0.18,0.35,0.06,0.04,0.43,0.0,-0.02,0.51,0.15,0.1,0.39,0.16,0.2,0.19,0.23,-0.23,0.27,-0.04,-0.1,0.34,0.2,0.44,-0.26,-0.22,-0.05,0.02,-0.28,-0.29,0.08,-0.03,-0.22,-0.13,-0.48,-0.4,-0.28,-0.38,-0.32,-0.14,-0.08,-0.23,-0.35,-0.05,-0.23,-0.14,-0.23,-0.14,-0.17,0.19,-0.01,-0.09,0.07,0.09,0.14,-0.11,-0.03,-0.19,0.04,-0.23,-0.04,-0.12,0.0,-0.22,-0.22,-0.27,0.18,0.02,-0.29,-0.29,-0.38,-0.17,-0.36,-0.27,-0.15,-0.41,-0.26,-0.09,-0.37,-0.39,-0.44,-0.35,-0.25,-0.61,-0.37,-0.19,0.05,0.07,0.03,-0.06,0.08,-0.05,-0.04,-0.26,0.22,0.27,0.04,-0.28,0.37,0.45,-0.05,-0.16,-0.08,0.21,0.04,0.14,0.02,0.18,0.1,0.0,0.67,0.08,-0.04,0.1,0.17,-0.31,-0.13,-0.09,0.08,-0.1,0.05,-0.37,-0.37,-0.13,-0.08,-0.19,-0.36,-0.26,-0.15,-0.49,-0.42,-0.27,-0.55,-0.81,-0.47,-0.3,-0.51,-0.27,-0.6,-0.41,-0.39,-0.57,-0.28,-0.39,-0.12,-0.23,-0.19,0.3,0.04,0.06,0.42,0.03,0.06,0.08,-0.03,0.41,0.12,-0.12,-0.27,0.0,-0.31,-0.26,-0.33,-0.38,-0.12,-0.38,-0.38,-0.23,-0.27,-0.17,-0.28,-0.24,-0.03,-0.19,-0.27,-0.17,-0.18,-0.23,0.07,-0.06,0.04,0.46,-0.04,-0.03,0.0,0.1,0.01,-0.07,0.02,-0.09,0.36,0.21,0.14,0.48,0.29,-0.01,0.28,-0.2,0.02,0.04,-0.09,0.07,0.28,-0.04,-0.14,0.26,0.31,-0.18,0.05,-0.09,0.01,-0.03,0.18,0.27,-0.09,0.31,0.08,0.06,0.27,0.04,-0.16,-0.03,0.04,0.16,0.18,0.1,0.24,0.13,-0.2,0.02,0.17,-0.11,0.02,0.2,0.39,0.25,0.13,0.15,0.06,0.23,0.18,-0.16,0.27,-0.01,0.37,0.15,0.44,-0.05,0.06,0.35,0.06,-0.1,-0.1,0.27,-0.19,0.08,-0.05,-0.1,-0.12,0.12,-0.18,-0.05,-0.23,-0.13,0.16,0.0,0.13,-0.05,-0.02,0.57,0.27,0.59,0.34,0.39,0.31,0.64,0.21,0.65,0.43,0.83,0.46,0.6,0.47,0.52,0.59,0.33,0.37,0.55,0.4,0.54,0.37,0.86,0.8,0.35,0.36,0.49,0.64,0.61,0.57,0.4,0.45,0.73,0.37,0.5,0.42,0.39,0.62,0.3,0.51,0.45,0.58,0.63,0.7,0.46,0.57,0.59,0.54,0.4,0.55,0.29,0.44,0.43,0.19,0.25,0.23,0.32,0.16,0.41,0.62,0.33,0.2,0.47,0.39,0.29,0.24,0.17,0.28,-0.07,0.0,-0.04,0.16,0.17,0.09,-0.01,0.29,-0.01,0.04,0.06,0.06,0.5,-0.15,0.06,0.21,-0.08,-0.01,-0.07,-0.09,-0.15,-0.06,-0.03,0.1,0.1,0.09,-0.18,-0.1,0.12,0.05,0.15,0.44,0.29,0.45,0.55,0.32,0.39,-0.1,-0.08,-0.18,-0.2,-0.28,0.02,0.06,-0.45,-0.24,-0.38,-0.38,-0.18,-0.39,-0.2,-0.3,-0.41,-0.11,-0.22,-0.31,-0.38,-0.12,-0.06,-0.2,-0.35,-0.06,-0.1,-0.06,-0.15,0.17,-0.23,-0.03,-0.25,0.16,0.34,0.06,-0.26,-0.08,0.02,-0.12,-0.19,0.06,-0.12,-0.09,-0.13,-0.16,0.06,0.18,0.0,0.06,-0.09,-0.06,-0.08,0.24,0.23,0.1,-0.11,-0.06,-0.13,-0.29,-0.18,0.27,0.41,-0.05,0.29,-0.28,-0.35,-0.26,0.01,0.03,-0.16,-0.22,-0.01,0.0,0.06,-0.03,0.08,0.09,-0.04,0.0,-0.18,0.39,-0.33,-0.04,0.08,0.03,0.07,0.08,-0.09,0.29,0.1,0.1,-0.04,0.01,0.19,0.24,-0.03,0.04,0.17,0.35,0.28,0.03,0.77,0.44,0.04,0.17,0.03,0.04,0.54,0.3,0.28,0.35,0.12,0.62,0.12,0.04,0.09,0.07,0.12,0.17,-0.13,0.25,-0.17,-0.03,0.51,0.25,-0.03,-0.18,0.15,-0.08,0.11,-0.22,0.02,0.22,0.08,0.25,-0.23,0.25,-0.03,0.0,-0.19,-0.04,0.23,0.1,-0.03,0.06,-0.03,-0.03,0.14,0.02,0.0,0.11,-0.18,-0.12,-0.12,0.04,0.04,0.39,-0.01,0.27,-0.03,0.05,0.38,0.18,0.3,0.17,-0.03,0.2,0.15,0.01,0.52,0.32,0.31,0.27,0.13,0.3,0.02,0.09,0.09,0.16,0.22,-0.23,-0.17,0.37,0.37,0.04,0.0,-0.16,0.13,0.09,0.28,0.09,0.21,0.2,0.04,0.34,0.1,0.1,0.01,-0.14,-0.08,0.16,0.0,0.31,0.3,0.06,0.1,0.55,0.21,0.34,0.42,0.39,0.03,0.19,0.43,0.16,0.38,0.29,0.2,0.55,0.15,0.07,0.02,0.03,0.17,0.1,-0.09,-0.16,-0.1,-0.29,-0.17,-0.01,-0.33,-0.25,0.03,-0.23,-0.07,-0.2,-0.22,-0.27,-0.19,0.09,-0.06,0.38,0.11,0.09,-0.18,0.45,0.16,-0.05,0.32,0.13,0.26,0.11,0.52,0.13,0.11,0.75,0.28,0.08,0.35,0.26,0.13,0.25,0.22,0.08,0.26,-0.17,-0.09,0.31,-0.21,-0.12,0.07,-0.14,-0.05,-0.06,-0.29,-0.18,-0.19,-0.26,-0.38,-0.11,-0.07,-0.05,-0.05,-0.21,-0.04,-0.04,-0.05,-0.12,-0.21,-0.02,0.25,0.37,0.0,-0.06,0.14,0.1,-0.13,0.1,0.1,0.06,-0.03,-0.11,0.0,0.1,-0.14,-0.05,-0.39,-0.04,-0.34,-0.35,-0.19,-0.49,-0.4,-0.56,0.04,0.1,-0.16,-0.1,-0.01,0.01,-0.22,-0.17,-0.04,-0.25,0.09,-0.45,-0.23,-0.45,-0.3,-0.13,-0.5,-0.23,-0.05,0.04,-0.21,0.06,-0.17,-0.36,-0.06,-0.28,-0.44,-0.45,-0.17,-0.48,-0.33,-0.35,-0.25,-0.39,-0.31,-0.51,-0.51,-0.54,-0.42,-0.76,-0.52,-0.72,-0.12,-0.57,-0.64,-0.21,-0.4,-0.07,-0.68,-0.58,-0.57,-0.71,-0.54,-0.5,-0.66,-0.6,-0.75,-0.56,-0.68,-0.64,-0.66,-0.77,-0.43,-0.53,-0.46,-0.24,-0.49,-0.32,-0.47,-0.48,-0.38,-0.36,-0.27,-0.07,-0.19,0.08,-0.05,-0.21,-0.04,-0.28,-0.12,-0.08,0.13,-0.43,-0.21,-0.13,-0.12,-0.32,-0.26,-0.23,-0.17,-0.2,-0.17,-0.1,-0.21,-0.3,-0.11,-0.32,-0.31,0.01,-0.21,-0.29,-0.3,-0.42,-0.13,-0.17,0.06,-0.48,-0.62,-0.49,-0.14,-0.27,-0.32,-0.19,-0.48,-0.01,-0.58,-0.19,-0.07,-0.14,-0.45,-0.31,0.18,-0.19,-0.31,-0.28,0.06,0.16,-0.15,-0.08,-0.26,-0.21,-0.22,0.06,-0.06,-0.27,-0.12,0.01,0.01,-0.14,0.04,-0.56,0.0,-0.16,-0.15,-0.14,-0.18,0.42,-0.25,-0.16,-0.05,-0.21,-0.22,0.15,0.09,0.22,-0.08,0.01,-0.1,0.0,-0.21,0.18,-0.1,-0.24,0.09,0.15,-0.12,0.03,-0.03,0.09,0.09,-0.04,0.01,0.12,-0.03,-0.03,0.4,0.23,-0.27,-0.05,-0.03,-0.08,-0.05,0.02,0.05,-0.14,0.16,0.17,0.04,0.33,0.21,0.19,-0.04,0.38,0.64,0.51,0.14,0.03,0.35,0.07,0.43,0.3,0.2,0.15,0.25,0.5,0.47,0.42,0.53,0.19,0.57,0.43,0.37,0.53,0.4,0.17,0.58,0.14,0.25,0.07,0.0,0.15,0.06,0.38,-0.31,-0.23,0.29,-0.05,0.01,0.0,0.31,0.13,-0.13,0.46,0.38,-0.12,-0.01,-0.01,-0.13,0.09,-0.05,0.01,0.18,0.02,0.26,0.26,0.15,-0.1,-0.26,0.0,0.2,-0.09,-0.21,0.17,-0.07,-0.37,-0.26,-0.27,-0.23,-0.13,-0.08,0.06,-0.26,-0.12,0.39,-0.07,-0.03,-0.22,-0.26,0.35,-0.01,0.1,0.27,0.04,-0.11,0.22,-0.09,0.11,-0.26,0.06,-0.2,-0.09,-0.47,0.16,-0.05,-0.01,-0.05,-0.02,-0.23,-0.06,0.3,0.25,0.11,0.2,-0.16,0.18,-0.12,0.03,-0.2,-0.28,-0.24,-0.17,-0.24,-0.31,-0.16,-0.13,0.07,-0.22,-0.37,-0.1,-0.01,-0.2,-0.27,-0.14,-0.16,-0.34,-0.04,-0.24,-0.25,-0.11,-0.07,0.01,-0.2,-0.16,-0.24,0.15,0.06,0.13,-0.41,-0.12,0.01,0.16,0.05,-0.02,0.02,-0.37,0.08,-0.21,-0.15,-0.09,0.12,-0.12,-0.14,0.06,-0.07,-0.05,-0.11,0.06,0.04,0.34,0.13,-0.07,-0.12,0.42,0.31,0.28,0.23,0.04,0.26,0.32,0.24,-0.24,0.32,0.38,0.51,0.33,0.44,0.56,0.32,0.59,0.25,0.24,0.51,0.35,0.79,0.38,0.43,0.52,0.29,0.16,0.45,0.33,0.27,0.63,0.43,0.49,0.52,0.64,0.5,0.49,0.49,0.33,0.49,0.53,0.15,0.46,0.56,0.43,0.42,0.39,0.16,0.66,0.29,0.63,0.39,0.35,0.51,0.47,0.45,0.19,0.14,0.32,0.32,0.48,0.32,0.87,0.44,0.05,0.4,0.12,0.43,0.23,0.39,0.54,0.65,0.33,0.43,0.27,0.62,0.45,0.32,0.29,0.55,0.71,0.33,0.4,0.5,0.45,0.15,0.39,0.09,0.3,0.22,0.19,0.37,0.01,-0.11,-0.08,-0.05,0.03,-0.08,0.14,0.09,0.12,-0.37,0.04,0.04,-0.2,0.04,-0.17,-0.39,-0.2,-0.42,-0.07,0.02,-0.09,-0.23,-0.24,-0.03,0.09,0.04,-0.07,-0.17,0.08,0.01,0.09,0.24,0.12,0.14,0.1,-0.04,0.24,-0.16,0.07,-0.14,-0.14,0.19,-0.1,-0.32,0.25,-0.04,0.03,-0.27,0.1,-0.05,-0.03,-0.08,0.1,-0.12,0.18,-0.3,-0.19,0.02,0.13,-0.22,-0.41,-0.51,-0.52,-0.09,-0.16,-0.25,-0.01,-0.34,-0.18,-0.26,-0.15,-0.09,-0.28,-0.15,-0.36,-0.31,-0.23,-0.22,-0.17,0.02,-0.16,-0.12,0.25,-0.23,-0.05,-0.05,-0.22,-0.1,-0.02,0.04,-0.01,0.17,-0.03,0.45,0.06,0.01,-0.16,0.22,0.34,0.14,0.3,0.2,-0.06,-0.1,-0.16,-0.27,0.02,0.13,0.36,-0.11,0.07,0.14,-0.01,-0.07,0.18,0.1,0.08,-0.05,0.16,0.25,-0.06,-0.02,0.1,0.22,0.16,-0.02,0.18,0.12,0.45,0.27,0.28,0.28,0.35,0.22,0.67,0.21,0.53,0.48,0.2,0.41,0.41,0.22,0.34,0.48,0.17,0.06,0.2,0.08,0.14,0.28,0.28,0.33,0.08,0.61,0.46,0.29,0.56,-0.08,0.26,0.06,0.02,0.1,-0.17,0.17,-0.02,0.03,-0.06,0.22,0.39,0.12,-0.18,0.07,0.31,-0.13,0.01,-0.04,0.01,0.1,-0.13,0.0,0.26,-0.08,0.46,0.16,0.06,0.26,0.24,0.43,0.32,0.19,0.21,-0.14,0.01,0.21,0.4,0.16,0.44,0.23,0.16,0.13,0.24,-0.18,-0.1,0.03,-0.16,-0.06,0.06,0.23,0.16,0.31,0.22,-0.25,0.06,0.13,0.27,0.09,0.08,0.27,-0.24,0.03,0.28,0.09,-0.25,-0.04,-0.26,0.01,0.02,-0.04,0.13,-0.1,0.12,0.12,-0.14,0.18,0.22,-0.09,-0.14,0.09,0.06,0.28,-0.18,-0.28,0.05,0.06,-0.23,-0.09,-0.05,-0.24,-0.28,-0.21,-0.1,0.25,-0.36,-0.45,-0.1,-0.12,-0.18,0.07,-0.13,-0.14,-0.2,-0.22,-0.13,-0.5,0.12,-0.22,-0.11,-0.25,-0.23,0.06,-0.32,-0.35,0.16,-0.07,0.25,-0.13,-0.22,0.01,-0.19,0.03,0.19,-0.17,-0.17,0.03,0.22,0.16,-0.13,-0.02,0.06,-0.3,-0.15,-0.24,0.16,-0.03,0.06,-0.16,-0.33,-0.52,-0.26,-0.13,-0.06,-0.19,-0.18,-0.18,0.17,-0.11,-0.1,-0.03,0.21,-0.08,-0.1,-0.05,0.13,0.15,0.06,-0.11,0.02,-0.02,-0.01,0.2,0.1,0.04,0.31,0.08,0.08,0.25,-0.15,0.12,0.04,0.26,-0.14,0.38,-0.19,-0.16,-0.03,0.17,-0.03,0.24,-0.01,0.17,-0.19,-0.11,-0.21,0.04,-0.32,-0.16,-0.09,0.0,-0.07,0.04,0.19,-0.07,0.28,-0.09,-0.07,-0.09,-0.32,-0.01,0.41,-0.32,-0.06,-0.3,-0.09,-0.26,-0.28,-0.23,-0.11,-0.35,-0.24,-0.34,-0.56,0.08,-0.26,-0.13,-0.36,-0.28,-0.34,-0.04,-0.03,-0.04,-0.21,-0.16,-0.26,-0.39,-0.54,-0.68,-0.39,-0.69,-0.29,-0.41,-0.2,-0.72,-0.37,-0.17,-0.53,-0.45,-0.32,-0.55,-0.14,-0.34,-0.41,-0.33,-0.27,-0.27,-0.62,-0.67,-0.54,-0.72,-0.48,-0.42,-0.24,-0.18,0.04,-0.63,-0.53,-0.45,-0.49,0.04,-0.09,-0.44,-0.49,-0.2,-0.1,-0.27,-0.55,-0.06,-0.2,-0.18,-0.22,-0.32,-0.16,-0.23,-0.03,0.34,0.23,0.1,0.02,0.11,0.49,0.02,0.01,0.08,0.03,0.13,-0.09,0.59,0.13,0.1,-0.05,0.49,0.17,0.15,-0.03,-0.11,0.15,0.31,-0.1,0.15,-0.27,0.17,-0.15,-0.1,-0.05,-0.24,-0.04,-0.03,0.23,0.23,0.18,0.15,0.14,0.11,-0.19,0.1,-0.11,0.29,-0.18,-0.13,0.18,-0.01,0.27,0.24,0.01,-0.04,0.19,0.31,0.44,0.01,0.34,0.35,0.21,0.06,-0.06,0.42,0.2,0.06,0.02,0.43,0.2,-0.23,0.2,0.05,0.23,0.33,0.0,0.19,0.03,-0.1,-0.21,-0.16,-0.38,-0.03,0.14,-0.37,-0.19,-0.48,-0.54,-0.61,-0.34,-0.42,-0.21,0.04,-0.23,0.13,-0.37,-0.22,-0.34,0.0,-0.29,-0.07,-0.13,0.14,0.27,0.04,0.34,0.09,0.1,-0.09,-0.06,0.01,0.33,-0.16,-0.15,-0.11,-0.11,-0.05,0.0,-0.09,0.31,0.15,0.13,0.04,0.08,-0.12,0.16,0.06,-0.2,0.09,-0.27,-0.27,0.0,-0.04,-0.19,0.02,0.2,-0.04,-0.24,-0.21,-0.27,0.11,-0.01,-0.23,-0.04,-0.29,-0.03,-0.24,-0.03,-0.2,-0.08,0.17,0.14,-0.21,-0.01,0.01,-0.09,0.03,0.19,0.18,-0.08,0.15,0.04,0.02,-0.03,0.19,-0.04,-0.16,0.09,0.06,0.25,-0.25,-0.04,-0.36,-0.18,-0.09,-0.33,0.27,0.08,-0.18,0.21,-0.27,-0.07,-0.05,0.2,-0.25,-0.42,-0.33,-0.45,-0.05,-0.22,-0.04,-0.53,-0.07,-0.28,-0.4,0.09,0.01,-0.13,-0.1,-0.32,0.17,0.26,-0.24,-0.12,-0.22,-0.18,-0.24,0.04,-0.17,0.08,0.01,-0.11,0.07,-0.28,-0.33,-0.21,-0.11,0.12,0.21,0.06,-0.04,-0.15,0.01,0.0,-0.2,-0.18,-0.19,-0.19,0.24,-0.11,-0.41,-0.25,-0.08,-0.21,0.01,-0.11,-0.17,-0.08,0.3,-0.17,0.02,-0.02,0.3,0.04,0.1,0.27,0.63,0.01,-0.21,-0.05,-0.03,-0.09,-0.29,-0.21,0.14,0.03,-0.14,0.13,0.14,-0.25,-0.16,-0.06,-0.06,-0.2,-0.3,0.04,-0.28,-0.1,-0.11,0.03,-0.06,-0.28,-0.47,-0.35,-0.01,-0.09,-0.28,-0.12,-0.33,-0.35,-0.26,-0.22,-0.43,-0.15,0.26,0.13,-0.12,-0.16,-0.1,-0.1,0.01,-0.01,-0.03,0.11,0.08,0.06,0.18,0.49,0.32,0.27,0.55,0.01,0.0,0.28,0.1,0.18,0.46,0.61,0.06,0.25,0.11,0.31,0.23,0.27,0.22,-0.05,0.1,0.26,0.12,0.14,0.25,0.36,-0.06,0.04,0.06,0.16,0.17,0.14,0.0,0.33,0.38,0.05,-0.18,-0.2,-0.22,-0.2,0.01,0.17,0.0,-0.06,0.31,0.27,0.14,0.25,0.18,0.01,0.1,0.22,-0.03,0.49,0.1,0.04,0.34,0.36,0.5,0.12,0.15,0.16,0.01,0.25,0.35,0.05,0.33,0.25,0.05,0.31,0.03,0.24,0.35,0.11,-0.03,0.23,0.06,-0.06,-0.03,-0.03,0.02,0.3,0.04,-0.15,0.39,0.04,-0.15,-0.04,-0.05,-0.07,0.63,0.23,-0.04,0.04,-0.08,-0.26,0.06,0.17,0.12,0.12,0.15,0.53,-0.07,0.09,0.24,-0.03,0.22,0.09,0.03,-0.04,0.26,0.05,0.29,0.2,0.1,0.23,0.15,0.36,-0.07,0.1,0.03,0.34,0.49,0.09,0.26,0.31,0.35,0.3,0.62,0.26,0.16,0.26,0.55,0.3,0.28,0.37,-0.03,0.6,0.38,0.28,0.26,0.13,0.21,0.14,0.32,0.31,0.34,0.59,0.23,0.28,0.12,0.23,0.4,0.01,0.04,0.35,0.29,0.04,0.04,0.26,0.02,0.17,0.01,0.1,-0.08,-0.1,-0.03,0.06,0.1,0.22,0.24,0.13,0.37,-0.03,0.25,0.19,0.21,0.32,-0.08,0.13,0.26,0.39,0.27,0.41,0.09,-0.06,0.05,-0.02,-0.16,0.02,-0.08,-0.09,0.21,0.04,0.19,-0.17,0.08,-0.05,-0.08,0.05,0.02,0.1,0.32,0.01,0.13,-0.06,0.06,0.18,0.23,-0.11,-0.06,0.26,0.21,0.16,0.16,0.21,-0.07,-0.03,-0.07,0.35,0.22,0.23,0.1,0.13,0.03,0.08,-0.01,0.01,-0.03,0.22,0.14,-0.03,-0.2,-0.04,-0.21,-0.04,-0.16,-0.32,-0.29,-0.36,-0.3,-0.34,-0.29,-0.15,-0.13,0.0,-0.42,-0.14,-0.35,-0.01,-0.15,0.16,0.15,-0.17,0.47,-0.31,-0.34,-0.09,-0.32,-0.23,-0.29,-0.12,0.02,-0.18,-0.16,-0.04,0.02,0.39,0.2,0.01,-0.07,-0.02,0.38,0.24,-0.01,0.04,0.18,0.14,0.04,0.36,0.28,0.29,0.11,-0.18,0.06,0.19,0.1,0.51,0.09,0.08,0.23,0.29,0.21,-0.01,0.45,0.0,0.17,-0.01,0.04,-0.11,0.1,0.04,0.33,0.18,-0.14,0.05,0.0,0.16,-0.18,-0.22,-0.26,-0.06,-0.25,0.04,0.01,-0.37,-0.07,-0.24,-0.23,0.12,-0.05,-0.07,-0.01,0.03,-0.05,0.1,-0.03,-0.1,-0.11,0.11,0.34,0.2,0.39,0.46,0.03,0.29,0.43,0.57,0.15,0.25,0.04,0.12,0.04,0.13,0.24,-0.04,0.1,0.2,0.09,0.43,0.22,0.03,0.17,0.27,0.11,0.09,-0.07,0.11,-0.09,0.16,-0.26,-0.16,-0.08,0.13,0.0,0.04,-0.15,-0.16,0.02,0.04,0.26,-0.22,0.28,0.6,0.14,0.2,0.14,-0.21,-0.11,0.04,-0.17,0.0,-0.08,0.08,-0.03,-0.19,0.0,-0.2,0.25,0.04,-0.11,0.13,-0.22,-0.32,-0.09,-0.35,-0.23,0.08,-0.1,-0.01,0.09,0.06,-0.04,-0.01,0.02,0.02,0.11,-0.12,-0.18,-0.03,0.55,0.14,0.25,0.03,-0.06,0.19,0.11,-0.16,-0.2,0.09,0.0,0.34,-0.13,-0.14,-0.04,0.09,0.06,0.1,0.04,0.06,0.12,0.23,0.03,-0.02,0.06,-0.03,0.01,-0.19,-0.06,-0.12,-0.05,-0.12,-0.21,-0.11,-0.32,-0.12,0.13,0.05,-0.17,0.25,0.36,0.1,-0.04,0.03,0.06,0.11,0.04,0.56,-0.18,-0.23,-0.02,-0.07,0.25,-0.06,0.21,0.0,0.29,0.14,0.13,0.1,0.41,-0.07,-0.28,-0.23,-0.07,-0.21,-0.08,0.15,-0.06,-0.08,-0.21,0.01,-0.17,0.01,-0.47,-0.25,-0.18,-0.13,-0.17,-0.23,-0.04,-0.22,-0.27,-0.17,-0.2,0.16,-0.28,-0.16,-0.11,0.39,-0.22,-0.09,0.18,-0.08,-0.08,-0.08,-0.24,-0.17,-0.3,0.52,0.18,0.08,0.04,-0.05,-0.17,0.02,-0.07,-0.05,0.26,0.12,0.05,0.14,-0.16,-0.31,0.26,0.06,-0.07,-0.46,-0.19,-0.29,-0.22,-0.2,-0.39,-0.13,-0.03,-0.04,-0.32,-0.36,-0.45,-0.65,-0.01,-0.61,-0.13,-0.18,-0.33,-0.43,-0.38,-0.11,-0.07,-0.33,-0.13,-0.24,0.13,-0.35,-0.09,-0.02,-0.22,-0.34,-0.18,-0.12,-0.39,-0.15,-0.44,-0.45,-0.22,-0.3,-0.11,-0.39,-0.14,-0.37,-0.52,-0.69,-0.49,-0.32,-0.33,-0.54,-0.3,-0.31,-0.39,-0.3,-0.07,-0.17,0.04,-0.25,-0.12,-0.02,0.01,-0.17,0.01,-0.12,0.02,0.06,0.03,-0.06,-0.13,-0.18,-0.14,0.04,-0.07,-0.3,0.31,-0.17,0.11,0.0,-0.12,-0.28,0.33,-0.37,-0.22,-0.1,-0.05,-0.15,0.12,-0.13,-0.07,-0.05,0.18,0.13,-0.08,0.26,0.01,0.42,0.1,0.04,0.43,-0.12,0.3,0.33,-0.2,0.31,0.55,0.14,0.06,0.38,-0.04,-0.05,0.25,0.13,-0.14,-0.01,0.01,-0.01,0.0,0.29,-0.07,0.27,0.09,0.16,0.0,-0.05,-0.22,-0.07,-0.25,0.0,-0.04,0.0,-0.03,-0.13,-0.08,-0.06,0.07,-0.23,-0.32,-0.17,-0.2,-0.2,-0.46,-0.21,0.01,-0.2,-0.06,0.56,0.14,0.47,-0.02,-0.13,-0.13,-0.18,-0.19,-0.27,0.22,-0.12,-0.1,0.17,-0.12,0.15,-0.01,0.28,0.02,0.12,0.04,0.19,0.17,0.13,0.29,0.15,-0.14,0.2,0.22,-0.06,-0.22,-0.23,0.02,0.36,-0.16,-0.02,-0.26,-0.3,-0.41,-0.34,-0.29,-0.32,-0.28,-0.35,0.06,-0.13,-0.29,-0.43,-0.25,-0.51,-0.52,-0.22,-0.19,0.08,-0.44,-0.41,-0.45,-0.47,-0.41,-0.45,-0.37,-0.59,-0.22,-0.59,-0.57,-0.44,-0.48,-0.64,-0.19,-0.42,-0.43,-0.48,0.2,0.1,-0.24,0.0,-0.24,-0.11,-0.17,0.01,0.4,0.23,0.11,0.06,0.03,0.08,-0.05,0.04,-0.2,0.01,0.23,-0.19,-0.17,0.36,-0.11,-0.17,0.17,-0.06,-0.17,-0.17,0.02,-0.14,-0.08,0.08,-0.31,-0.28,-0.21,-0.2,-0.1,-0.35,-0.08,-0.2,-0.28,-0.45,-0.23,-0.31,-0.24,-0.03,-0.27,-0.32,-0.13,-0.25,-0.36,-0.24,-0.12,-0.16,0.09,-0.23,0.04,0.1,-0.02,-0.11,0.25,-0.16,-0.08,0.38,0.1,-0.3,-0.17,0.13,-0.5,0.02,-0.3,-0.06,-0.16,-0.14,-0.03,0.08,0.33,-0.11,-0.23,-0.12,-0.3,-0.41,-0.16,-0.08,-0.25,-0.14,0.06,-0.06,0.06,0.03,0.01,-0.03,-0.07,-0.01,0.09,-0.06,0.29,0.19,0.5,0.3,0.15,0.1,0.17,0.08,0.49,0.28,0.17,-0.15,0.17,0.55,-0.03,-0.34,-0.09,-0.32,0.14,0.02,-0.06,-0.53,-0.12,-0.37,-0.27,-0.43,-0.41,-0.26,-0.23,-0.29,-0.58,-0.1,-0.25,-0.28,-0.1,-0.23,-0.25,0.04,-0.1,-0.08,-0.48,-0.27,0.15,-0.08,-0.03,-0.29,-0.08,-0.05,-0.3,-0.29,-0.03,-0.19,-0.42,-0.18,0.22,-0.28,-0.21,0.01,-0.1,-0.3,-0.18,-0.49,0.1,-0.21,-0.38,-0.05,-0.41,-0.33,0.09,-0.36,-0.18,-0.3,-0.07,0.12,0.14,-0.12,-0.03,0.1,0.07,0.05,0.33,-0.03,-0.01,0.09,0.25,0.21,0.15,0.02,-0.14,-0.14,-0.06,-0.05,0.2,-0.26,0.0,0.1,-0.19,0.01,0.01,0.17,0.24,0.01,-0.17,0.19,0.04,0.09,-0.15,-0.04,0.23,-0.09,-0.06,0.05,-0.18,0.16,0.08,-0.29,-0.47,0.25,-0.08,-0.18,-0.26,-0.36,-0.19,-0.15,-0.11,-0.13,-0.09,-0.09,0.05,0.06,-0.05,-0.07,0.09,0.06,0.27,-0.03,-0.05,-0.15,0.09,-0.28,0.04,0.13,-0.08,-0.08,-0.11,-0.17,-0.27,-0.33,-0.28,-0.04,-0.26,0.09,-0.17,-0.4,-0.25,0.17,-0.15,-0.42,-0.37,-0.05,-0.3,-0.01,-0.02,-0.12,0.03,0.17,-0.23,-0.18,-0.21,-0.06,-0.2,-0.14,-0.03,-0.13,-0.01,-0.05,0.37,-0.01,0.12,0.07,0.03,-0.01,-0.05,0.02,-0.14,-0.22,-0.25,-0.17,0.1,-0.03,-0.13,-0.11,0.23,-0.05,-0.03,-0.14,-0.22,-0.07,-0.14,0.1,-0.06,-0.28,0.04,0.17,0.2,0.55,0.09,0.03,0.04,0.02,-0.09,0.04,0.19,-0.21,-0.21,-0.44,0.09,0.18,-0.33,-0.33,-0.1,0.17,-0.01,0.08,-0.34,0.07,-0.05,-0.38,-0.17,0.21,0.0,0.2,0.16,0.04,0.08,-0.19,0.25,0.03,0.17,0.19,-0.05,0.16,0.53,0.17,0.38,0.26,0.57,0.52,0.21,0.01,0.35,-0.09,0.04,0.13,0.25,0.5,0.13,0.33,0.16,0.29,0.11,0.27,0.08,0.2,0.24,0.31,0.44,0.31,0.26,-0.1,0.3,0.18,0.28,-0.05,0.3,0.1,0.01,-0.15,-0.15,-0.07,-0.14,0.01,-0.32,-0.12,-0.28,-0.5,0.06,0.09,-0.34,-0.08,-0.46,-0.14,-0.13,-0.49,-0.16,-0.57,-0.37,-0.59,-0.52,-0.52,-0.38,-0.17,-0.41,0.01,-0.18,-0.23,-0.11,0.04,-0.21,-0.27,0.0,-0.03,-0.12,-0.33,-0.58,-0.32,0.2,-0.21,-0.42,-0.34,-0.38,-0.32,-0.01,-0.34,-0.46,-0.05,0.03,-0.28,-0.23,-0.36,-0.07,-0.25,0.01,-0.45,-0.23,-0.2,0.16,0.27,0.13,-0.1,0.1,0.3,0.16,-0.25,0.11,0.1,0.1,0.04,0.16,0.44,0.22,0.41,0.45,0.47,0.43,0.06,0.42,0.54,0.42,0.18,0.28,0.45,0.02,0.06,-0.26,-0.1,0.11,-0.1,-0.07,-0.18,-0.29,-0.04,-0.29,-0.28,-0.02,0.01,-0.1,-0.05,-0.07,-0.18,-0.22,0.0,-0.43,-0.32,-0.45,-0.11,-0.25,-0.32,-0.59,-0.12,-0.17,-0.24,-0.03,0.08,-0.03,-0.29,-0.26,0.09,-0.21,0.09,0.08,-0.06,0.29,-0.13,0.12,-0.2,0.18,-0.08,-0.19,-0.25,0.1,0.16,0.04,-0.06,0.06,0.12,-0.11,-0.27,0.06,0.17,0.14,0.16,0.48,0.34,0.21,0.1,0.36,-0.17,0.13,0.2,-0.82,-0.06,0.09,-0.08,0.3,-0.06,0.09,0.12,0.15,0.28,0.17,-0.01,0.14,0.05,0.17,0.04,-0.05,0.35,0.08,0.09,0.04,0.09,0.15,0.1,0.1,-0.02,-0.14,0.03,-0.39,-0.31,-0.34,-0.03,-0.16,0.18,-0.11,-0.09,-0.39,-0.31,-0.21,-0.31,-0.24,-0.45,0.0,0.1,-0.1,0.02,-0.09,-0.28,-0.33,-0.02,-0.02,-0.19,-0.1,-0.37,-0.1,0.08,-0.23,0.0,-0.22,-0.11,0.01,0.12,-0.2,-0.09,-0.08,0.27,0.1,-0.25,0.03,-0.4,-0.07,-0.26,0.1,-0.25,0.04,0.07,0.0,0.06,-0.11,-0.04,0.18,0.25,0.1,0.41,0.14,0.1,0.12,0.17,-0.49,0.15,0.31,0.39,-0.05,0.08,0.27,0.42,0.29,0.27,-0.06,0.19,0.19,0.25,-0.13,-0.01,0.0,0.12,0.25,0.13,0.09,-0.08,0.1,0.35,0.06,0.07,0.11,0.06,0.08,0.09,-0.1,0.16,0.33,-0.03,-0.16,-0.03,-0.12,0.29,0.12,-0.01,0.02,-0.12,0.22,0.31,-0.19,0.02,-0.03,-0.05,-0.01,0.28,0.08,0.25,0.13,0.26,-0.06,0.51,0.19,0.35,0.02,0.01,0.76,0.39,0.16,-0.12,-0.03,0.06,0.12,0.46,0.19,0.02,-0.09,-0.01,0.22,0.09,-0.22,-0.11,0.04,-0.04,-0.11,-0.11,0.22,0.09,-0.04,-0.16,-0.39,-0.23,0.06,-0.23,-0.28,-0.05,0.1,-0.15,-0.23,-0.31,-0.19,-0.25,-0.4,-0.2,0.14,-0.03,-0.18,-0.12,0.13,0.23,0.09,0.1,0.23,0.08,0.28,0.16,0.13,0.11,0.26,0.3,0.38,0.07,0.27,0.08,-0.2,0.12,-0.1,0.14,0.16,0.15,0.16,0.01,0.0,0.29,0.03,0.2,0.27,0.04,-0.16,-0.05,-0.09,-0.02,0.11,0.33,0.01,-0.02,-0.03,-0.15,-0.05,-0.39,-0.03,0.1,-0.11,-0.15,0.17,-0.09,-0.34,-0.27,0.17,-0.29,-0.34,-0.11,0.01,-0.5,-0.49,-0.45,-0.27,-0.38,-0.27,-0.44,-0.34,-0.22,-0.05,-0.04,-0.24,-0.18,-0.44,-0.17,-0.31,-0.32,-0.54,-0.02,-0.2,-0.23,-0.32,-0.39,-0.16,-0.11,-0.03,-0.34,-0.26,-0.26,-0.17,-0.3,-0.04,-0.02,-0.15,0.07,-0.14,0.1,0.36,-0.4,-0.18,0.12,-0.12,-0.12,-0.17,0.46,-0.02,-0.07,0.3,0.29,0.15,0.11,0.17,0.37,0.17,0.39,0.32,0.69,-0.01,0.05,0.05,0.21,0.16,0.03,0.15,0.2,0.43,0.03,0.21,0.17,0.49,0.67,0.39,0.32,0.5,0.66,0.54,0.38,0.33,0.37,0.39,0.57,0.26,0.71,0.43,0.27,0.33,0.53,0.0,0.28,0.14,0.09,0.02,0.28,0.15,0.09,0.37,0.14,0.1,0.0,0.34,0.23,0.27,0.22,-0.04,0.06,0.1,-0.05,-0.23,0.19,0.22,0.08,-0.2,-0.14,-0.14,-0.26,-0.22,-0.32,-0.19,0.08,-0.15,-0.3,-0.19,-0.17,-0.15,0.02,-0.12,-0.2,-0.09,-0.2,-0.07,-0.17,-0.06,-0.33,-0.23,-0.11,-0.03,-0.33,-0.03,-0.25,-0.23,-0.38,-0.02,-0.21,-0.22,0.12,-0.01,0.29,-0.2,0.11,-0.35,0.11,-0.2,-0.37,-0.23,-0.23,-0.22,0.12,0.08,-0.04,-0.17,0.09,-0.06,0.35,0.01,-0.13,0.12,0.29,0.12,0.46,0.05,0.27,0.08,0.29,0.16,-0.06,-0.05,0.37,0.16,-0.26,0.03,-0.1,-0.31,-0.17,-0.17,-0.14,-0.11,-0.18,0.32,-0.13,0.12,-0.19,0.22,0.12,-0.27,-0.4,-0.22,-0.21,-0.09,-0.24,-0.17,0.09,-0.21,-0.42,-0.06,-0.36,-0.19,-0.15,-0.19,-0.15,0.14,0.13,0.15,0.07,0.14,0.09,0.15,-0.04,-0.01,-0.28,0.0,-0.19,0.27,-0.03,0.04,-0.12,0.01,0.11,0.05,-0.19,-0.2,-0.36,-0.16,0.18,-0.05,-0.05,-0.2,-0.13,-0.07,-0.06,-0.08,0.3,0.18,0.53,0.43,0.52,0.17,0.19,0.19,0.1,0.49,0.08,0.0,0.32,0.1,0.24,0.35,-0.01,0.32,-0.08,-0.08,-0.03,0.17,0.11,0.19,0.04,0.0,0.2,0.29,0.12,-0.04,0.15,0.34,0.31,0.18,0.15,0.36,0.18,0.05,0.16,0.3,0.02,0.04,0.46,0.55,0.13,0.16,0.01,0.25,-0.1,0.2,-0.01,0.02,0.15,0.14,-0.09,-0.03,-0.08,0.1,0.01,-0.07,0.44,-0.35,-0.19,-0.2,-0.35,0.01,0.32,-0.07,-0.19,-0.05,-0.04,0.02,-0.07,0.04,0.31,-0.13,-0.2,0.06,0.09,-0.05,-0.07,-0.28,0.06,0.18,0.27,-0.08,0.1,-0.17,0.17,0.03,-0.16,-0.09,-0.44,-0.15,-0.17,0.0,-0.11,-0.04,-0.14,-0.32,-0.37,-0.24,-0.15,0.09,-0.23,-0.07,-0.06,-0.59,-0.45,-0.08,-0.2,-0.22,-0.11,-0.09,0.09,0.18,0.15,-0.14,0.2,0.2,0.39,-0.02,0.08,-0.04,0.0,-0.08,-0.12,0.06,-0.25,0.09,-0.16,-0.25,-0.34,-0.16,-0.12,-0.51,-0.21,-0.34,-0.26,-0.26,-0.33,-0.05,-0.27,-0.08,-0.31,-0.2,0.06,-0.04,-0.14,-0.29,-0.09,0.13,-0.32,-0.34,-0.19,-0.18,-0.52,-0.41,-0.36,-0.36,-0.6,-0.18,-0.33,-0.54,-0.05,-0.38,0.04,-0.25,-0.48,-0.43,-0.57,-0.18,-0.26,-0.16,0.09,-0.2,-0.35,-0.4,-0.51,-0.38,-0.4,-0.32,-0.07,-0.07,-0.31,0.01,-0.2,-0.27,0.29,-0.21,-0.07,0.03,0.13,-0.04,-0.2,-0.31,-0.04,-0.51,-0.03,-0.16,-0.55,-0.17,-0.36,-0.22,-0.29,-0.59,0.2,-0.1,0.15,-0.57,-0.27,-0.29,-0.43,-0.41,-0.13,-0.29,-0.01,-0.31,-0.29,-0.29,-0.13,-0.15,-0.27,0.0,-0.04,0.0,0.04,0.04,-0.05,-0.25,-0.4,-0.47,-0.38,0.1,-0.44,-0.42,-0.32,-0.33,-0.42,-0.41,-0.47,-0.35,-0.14,0.01,-0.21,0.08,0.06,-0.05,0.12,-0.07,-0.08,-0.03,0.17,0.26,-0.2,0.14,-0.19,0.07,0.11,-0.1,-0.19,0.09,-0.11,-0.04,-0.01,-0.22,-0.18,-0.02,-0.18,-0.03,-0.16,-0.09,-0.17,0.1,-0.2,-0.1,-0.3,-0.03,0.16,-0.21,-0.33,-0.14,-0.05,-0.07,-0.31,-0.08,-0.29,-0.36,0.2,-0.07,-0.12,-0.1,-0.03,0.1,0.2,-0.25,0.04,0.26,-0.09,0.13,-0.02,0.03,-0.16,0.0,-0.05,0.16,-0.07,-0.16,-0.14,-0.13,-0.05,-0.15,-0.18,-0.54,-0.2,-0.27,-0.29,-0.05,-0.03,-0.47,-0.03,0.14,-0.01,-0.31,-0.14,-0.26,-0.31,-0.05,-0.33,-0.51,-0.37,-0.25,-0.3,-0.2,-0.27,-0.34,-0.41,-0.21,0.15,-0.14,-0.14,-0.35,0.03,-0.07,-0.24,0.26,0.22,-0.09,0.43,0.18,0.0,0.49,0.37,0.3,0.17,0.04,0.16,0.51,0.46,0.39,0.68,0.21,0.21,0.33,0.41,0.36,0.14,0.39,0.4,0.1,0.54,0.39,-0.43,-0.07,0.01,-0.06,0.11,0.06,-0.2,0.17,-0.32,-0.14,-0.29,-0.17,0.02,-0.43,-0.51,-0.26,-0.21,-0.32,-0.11,-0.26,-0.09,0.11,-0.42,-0.35,-0.37,-0.39,-0.28,-0.26,-0.67,0.08,0.06,-0.09,-0.51,-0.22,-0.44,-0.27,-0.45,-0.6,-0.35,-0.66,-0.38,-0.61,-0.43,-0.35,-0.37,-0.46,-0.19,-0.66,0.06,-0.04,-0.25,0.0,-0.2,0.09,-0.39,-0.04,-0.11,-0.23,-0.31,-0.08,-0.4,0.06,0.13,-0.12,-0.1,-0.03,-0.29,-0.14,-0.23,-0.48,-0.18,-0.29,-0.2,-0.32,-0.18,-0.3,-0.45,-0.03,-0.03,0.23,-0.25,-0.42,-0.32,-0.24,-0.06,-0.15,-0.11,0.14,-0.26,0.3,-0.3,-0.4,-0.23,-0.48,-0.34,-0.35,-0.04,-0.26,-0.32,-0.35,-0.31,-0.28,0.19,-0.3,-0.03,-0.04,-0.26,-0.25,-0.34,-0.27,-0.22,-0.41,0.1,-0.32,-0.3,-0.52,-0.33,-0.22,-0.09,-0.24,-0.33,-0.13,-0.43,-0.07,-0.32,-0.15,0.18,-0.32,-0.41,-0.2,-0.34,0.12,-0.34,-0.04,-0.29,-0.16,0.01,-0.28,0.11,-0.24,0.15,-0.28,0.15,-0.04,-0.15,0.34,0.06,-0.25,0.09,0.03,-0.05,0.1,-0.11,-0.01,0.01,0.05,0.08,0.01,0.28,0.25,0.17,-0.15,0.15,0.29,0.29,0.25,0.32,0.2,0.29,0.35,0.55,0.43,0.91,0.67,0.48,0.55,0.56,0.28,0.91,0.36,0.1,0.23,0.35,0.54,0.35,0.42,0.24,0.09,0.14,0.14,0.08,0.01,-0.31,-0.08,-0.11,-0.18,-0.22,-0.48,0.01,-0.27,-0.42,-0.44,-0.3,-0.01,-0.23,-0.04,0.02,-0.07,-0.4,-0.14,-0.28,-0.04,-0.4,-0.29,-0.03,-0.2,-0.08,0.15,-0.12,0.12,-0.19,0.0,-0.07,-0.03,0.27,0.3,0.07,0.23,-0.17,-0.2,0.02,0.24,0.22,0.02,0.4,0.17,-0.16,0.08,0.39,0.23,0.21,0.12,0.02,0.35,0.12,0.41,0.35,0.2,0.61,0.57,0.41,0.51,0.3,0.58,0.46,0.78,0.56,0.75,0.64,0.63,0.78,0.68,0.97,0.62,0.88,0.85,0.94,0.87,0.65,0.82,0.81,0.64,0.63,0.52,0.72,0.71,0.66,0.62,0.71,0.5,0.58,0.66,0.4,0.66,0.61,0.42,0.33,0.55,0.31,0.78,0.31,0.86,0.64,0.72,0.42,0.53,0.41,0.52,0.41,0.63,0.46,0.29,0.52,0.42,0.62,0.55,0.72,0.22,0.39,0.07,0.2,0.39,0.04,0.4,0.44,0.64,0.54,0.58,0.19,0.44,0.46,0.38,0.36,0.37,0.42,0.49,0.64,0.37,0.38,0.32,0.51,0.41,0.36,0.24,0.57,0.32,0.42,0.14,0.28,0.23,0.19,0.54,0.18,0.13,0.74,0.83,0.44,0.34,0.21,0.72,0.36,0.5,0.3,0.32,0.43,0.45,0.69,0.15,0.26,0.41,0.26,0.44,0.32,0.81,0.38,0.36,0.38,0.72,0.37,0.25,0.57,0.65,0.28,0.63,0.47,0.44,0.63,0.49,0.51,0.57,0.53,0.36,0.37,0.43,0.51,0.82,0.17,0.27,0.28,0.41,0.1,0.42,0.21,0.16,0.1,0.25,0.33,-0.08,0.04,0.33,-0.01,-0.02,0.21,0.17,0.24,0.26,0.21,-0.13,-0.06,-0.1,0.09,-0.04,-0.28,-0.03,-0.03,0.03,0.04,-0.16,-0.15,0.7,0.27,0.09,0.18,-0.01,0.08,-0.11,0.15,0.2,-0.17,-0.23,0.15,-0.31,-0.07,-0.07,-0.2,-0.39,-0.39,-0.18,-0.1,0.08,-0.13,-0.02,-0.27,0.04,-0.26,-0.28,-0.16,0.06,-0.14,-0.25,0.01,-0.43,0.01,-0.03,-0.27,-0.17,0.22,0.17,0.28,-0.01,0.25,-0.06,-0.1,0.13,-0.36,-0.16,-0.24,-0.13,0.13,-0.35,0.0,-0.29,-0.06,-0.07,-0.27,0.23,-0.23,0.01,-0.04,-0.15,-0.15,-0.01,-0.34,-0.07,-0.05,0.01,-0.33,0.02,-0.26,-0.14,0.21,-0.36,-0.26,0.0,-0.15,0.21,-0.11,-0.01,-0.01,-0.11,0.29,0.04,-0.23,-0.14,-0.08,0.09,0.12,-0.1,0.05,-0.04,0.07,-0.08,0.06,0.12,0.24,0.3,0.1,0.1,0.27,-0.04,0.3,0.05,0.04,0.45,-0.22,-0.24,0.12,0.14,0.17,0.1,-0.07,0.0,-0.03,-0.01,0.27,0.03,0.26,0.25,0.09,0.2,0.19,-0.26,-0.04,-0.14,0.11,-0.27,0.13,-0.2,-0.21,-0.1,-0.33,0.22,-0.03,-0.25,0.06,-0.15,-0.07,0.04,0.25,-0.09,0.01,-0.06,-0.14,0.01,0.15,0.01,0.06,0.1,-0.05,-0.23,0.38,-0.11,-0.06,-0.06,0.03,-0.38,-0.26,-0.15,-0.22,0.04,-0.01,0.11,-0.01,-0.27,-0.46,-0.22,-0.28,-0.32,-0.22,0.14,-0.25,-0.23,-0.38,0.08,-0.08,0.34,-0.04,0.13,-0.44,-0.09,-0.08,-0.12,-0.25,-0.37,-0.23,-0.38,-0.14,-0.47,-0.29,-0.23,-0.22,-0.2,-0.2,-0.3,-0.11,-0.31,0.08,-0.03,-0.23,-0.04,0.06,-0.19,-0.21,-0.11,-0.06,0.06,-0.13,-0.04,-0.01,0.01,-0.09,0.23,-0.06,0.09,-0.05,-0.14,-0.02,-0.26,-0.59,-0.45,-0.24,-0.36,-0.19,-0.22,-0.08,-0.21,-0.42,-0.19,-0.37,0.0,0.0,-0.2,-0.19,0.06,0.09,-0.09,-0.23,0.2,-0.05,-0.19,-0.03,0.0,-0.13,-0.12,-0.09,0.06,-0.02,0.13,-0.06,-0.06,-0.16,-0.36,-0.21,-0.41,0.43,-0.25,-0.03,-0.29,-0.45,-0.34,-0.32,-0.29,0.09,-0.1,-0.01,-0.09,-0.09,-0.24,-0.31,-0.15,-0.24,-0.28,-0.21,-0.17,-0.16,-0.15,-0.29,-0.25,0.07,-0.29,-0.14,0.1,-0.3,-0.14,0.14,0.18,0.09,-0.29,0.29,0.0,-0.2,-0.31,-0.42,-0.35,-0.08,-0.23,-0.17,-0.14,-0.22,-0.36,-0.28,0.01,-0.18,-0.04,-0.29,-0.18,-0.1,0.1,-0.09,0.0,0.07,-0.01,0.11,0.52,0.16,0.03,0.35,0.78,0.06,0.37,0.27,0.21,0.11,0.14,0.36,0.4,0.46,0.2,0.32,0.35,0.25,0.15,0.64,0.35,0.35,-0.08,0.24,0.31,0.17,0.14,-0.17,0.15,0.24,0.19,0.03,0.17,-0.06,0.36,-0.03,-0.07,0.0,-0.07,0.1,-0.22,0.11,-0.19,0.01,-0.08,-0.31,0.26,-0.03,-0.28,-0.27,0.25,-0.16,0.2,-0.43,-0.07,-0.03,-0.09,-0.07,-0.1,-0.21,-0.24,-0.01,0.21,0.1,0.09,-0.14,0.0,0.03,0.03,0.37,-0.25,-0.13,-0.03,0.1,0.1,-0.01,0.15,-0.04,-0.16,-0.03,-0.09,0.1,0.01,0.0,-0.28,0.4,0.4,-0.24,-0.13,-0.08,-0.38,-0.03,0.01,-0.54,-0.22,-0.31,-0.4,-0.13,-0.12,0.04,0.19,-0.32,0.01,0.03,0.09,0.04,-0.11,-0.15,-0.51,-0.36,-0.36,-0.27,-0.16,-0.38,-0.08,-0.18,-0.11,0.05,-0.03,-0.37,-0.49,0.07,-0.17,-0.25,-0.41,-0.21,-0.25,-0.22,-0.55,-0.22,-0.22,-0.23,-0.25,-0.24,-0.24,-0.19,0.05,-0.13,-0.01,0.06,-0.23,-0.24,0.09,-0.31,-0.28,-0.2,-0.12,0.01,-0.41,-0.18,-0.23,-0.13,0.28,-0.19,0.0,0.04,0.06,-0.04,-0.15,0.1,-0.06,0.04,-0.05,0.26,0.23,0.04,-0.2,0.42,0.08,0.49,0.22,0.18,-0.12,0.1,0.37,0.04,0.29,0.28,-0.06,0.0,0.2,0.09,0.21,-0.08,0.1,-0.07,0.03,0.23,-0.04,0.22,0.07,0.02,-0.04,0.08,0.06,0.54,0.04,0.49,0.18,0.15,0.57,0.27,0.64,0.43,0.29,0.47,0.41,0.08,0.27,0.41,0.04,0.42,0.65,0.41,0.38,0.13,0.01,0.3,0.32,0.01,0.12,-0.09,-0.12,-0.11,0.03,-0.1,0.23,-0.11,0.28,0.45,-0.04,-0.1,-0.06,0.27,-0.04,-0.09,-0.36,0.2,-0.16,-0.15,-0.22,-0.4,-0.23,-0.22,-0.11,-0.19,-0.1,-0.18,-0.03,-0.24,-0.12,-0.37,0.11,-0.1,0.12,-0.01,-0.05,0.0,0.16,0.14,0.16,0.29,0.17,0.38,0.53,0.19,0.33,0.29,0.02,0.09,0.0,0.44,0.3,0.16,0.03,0.24,0.41,0.17,0.35,0.17,0.41,0.4,0.1,0.39,0.25,0.2,0.27,0.37,0.26,0.36,0.57,0.15,0.29,0.39,0.23,0.1,0.41,0.35,0.06,0.38,0.53,0.06,0.19,0.16,0.36,0.22,0.26,0.15,-0.19,0.21,-0.15,0.09,0.11,0.1,-0.21,0.08,-0.22,0.11,0.04,0.29,0.27,0.34,0.15,0.03,0.06,0.1,-0.37,-0.1,-0.2,-0.08,0.04,-0.07,-0.22,-0.2,-0.27,-0.36,-0.23,-0.41,-0.26,-0.12,-0.24,-0.05,-0.26,-0.15,-0.34,-0.27,-0.13,-0.42,-0.29,-0.3,-0.39,-0.19,0.04,-0.18,-0.3,-0.35,-0.19,-0.55,-0.26,-0.41,-0.08,-0.09,-0.11,-0.25,0.08,0.06,-0.35,-0.43,-0.16,-0.3,-0.09,0.0,-0.16,0.16,-0.33,0.29,-0.11,0.35,-0.03,0.04,0.13,-0.16,-0.07,0.02,-0.11,-0.03,-0.08,0.11,0.13,-0.18,-0.01,-0.1,-0.14,-0.13,-0.04,0.16,-0.21,-0.03,-0.02,0.27,0.33,0.22,-0.25,-0.22,-0.18,-0.33,-0.06,-0.03,-0.14,-0.14,0.12,-0.15,-0.14,-0.01,-0.07,-0.09,-0.27,0.12,0.02,-0.11,-0.07,0.1,0.17,0.0,-0.03,-0.19,0.04,0.34,-0.14,-0.17,0.02,-0.23,0.17,0.04,0.32,0.27,0.07,0.21,0.6,0.38,0.19,0.55,0.08,0.12,0.14,0.57,0.44,0.12,0.54,0.22,0.18,-0.01,0.33,0.06,-0.17,0.11,-0.19,-0.19,0.01,-0.04,-0.27,0.21,-0.48,-0.35,-0.16,0.08,0.15,-0.2,0.03,-0.17,-0.03,-0.18,-0.22,0.14,-0.06,-0.03,-0.31,-0.38,-0.28,-0.23,0.01,-0.21,-0.38,-0.4,0.01,0.1,-0.39,-0.24,-0.34,0.04,-0.28,-0.19,-0.21,-0.12,0.0,-0.52,-0.51,-0.27,-0.3,-0.39,-0.22,-0.07,-0.15,-0.11,-0.16,0.01,-0.15,-0.07,-0.2,-0.42,-0.03,-0.31,-0.38,-0.23,0.03,-0.45,-0.66,-0.33,-0.3,-0.35,-0.28,-0.5,-0.19,-0.39,-0.13,-0.16,0.01,-0.35,-0.32,-0.14,-0.25,-0.49,-0.12,-0.17,-0.24,-0.17,0.03,-0.2,-0.21,-0.28,-0.07,-0.13,-0.53,-0.32,-0.22,-0.46,-0.08,-0.11,-0.04,-0.19,-0.26,0.15,0.22,-0.01,0.43,0.03,0.1,0.16,-0.04,-0.14,0.17,-0.04,-0.04,-0.13,-0.26,-0.33,-0.01,-0.25,-0.1,-0.15,-0.21,0.06,-0.19,0.04,-0.23,-0.3,-0.28,-0.38,-0.25,-0.15,-0.35,-0.36,-0.11,-0.24,0.12,-0.12,-0.13,0.0,-0.03,0.25,-0.07,-0.05,-0.05,0.15,0.15,0.29,0.04,-0.02,0.03,0.23,0.04,0.04,0.33,-0.1,-0.17,-0.25,0.1,-0.02,0.16,-0.4,-0.15,0.14,0.0,-0.03,-0.32,-0.11,-0.15,-0.13,0.05,-0.21,0.02,-0.34,-0.03,-0.6,0.1,-0.11,-0.21,0.06,-0.48,-0.03,-0.17,0.0,-0.04,0.03,0.32,-0.37,-0.05,-0.12,-0.01,0.25,-0.18,-0.14,0.06,0.06,-0.37,-0.08,0.04,0.02,-0.15,-0.19,-0.32,-0.27,0.12,-0.18,-0.07,-0.15,-0.32,-0.17,-0.23,-0.11,-0.26,-0.3,-0.22,-0.09,-0.03,-0.32,-0.12,-0.19,-0.25,0.01,0.0,-0.36,0.09,-0.24,-0.41,-0.34,-0.34,-0.36,-0.14,-0.16,-0.02,-0.17,-0.29,0.32,-0.29,0.19,0.12,0.3,0.28,0.06,0.39,0.29,0.08,0.3,0.47,0.36,0.15,0.45,0.36,0.39,0.52,0.46,0.23,0.12,0.32,0.18,0.29,0.28,0.4,0.21,0.33,0.2,0.25,0.23,-0.01,0.02,0.37,0.45,0.39,0.05,-0.03,-0.12,0.0,0.07,-0.02,-0.18,0.05,-0.05,0.02,0.15,0.01,0.1,0.02,-0.02,0.05,-0.09,-0.45,-0.14,-0.43,-0.32,-0.24,-0.52,-0.46,-0.32,-0.33,-0.1,-0.44,-0.56,-0.05,-0.44,0.05,0.0,-0.4,-0.4,-0.28,-0.11,-0.4,-0.44,-0.27,-0.26,-0.36,-0.17,-0.35,-0.26,-0.11,-0.03,-0.31,-0.12,-0.05,-0.17,0.13,0.02,0.09,-0.07,-0.14,0.03,-0.05,-0.15,0.18,-0.15,-0.19,-0.19,0.02,-0.01,-0.22,0.0,-0.04,0.22,-0.03,0.08,-0.02,-0.18,0.33,-0.22,-0.05,-0.34,-0.08,-0.04,-0.06,-0.06,-0.1,0.47,0.21,-0.22,-0.04,0.4,0.06,0.13,0.32,0.13,0.02,-0.01,0.06,0.08,0.1,0.32,0.46,-0.03,0.02,0.3,-0.04,0.01,0.52,0.01,0.16,0.07,-0.01,-0.17,0.06,0.13,0.01,0.15,-0.09,0.1,-0.1,-0.03,0.17,0.06,0.03,0.04,0.0,0.1,0.01,0.15,0.22,0.01,-0.01,0.32,-0.22,0.13,-0.09,0.4,0.29,0.4,-0.03,0.34,0.49,-0.07,-0.3,0.09,-0.18,0.01,0.11,0.2,0.06,-0.01,-0.06,0.15,-0.05,0.13,0.08,-0.09,0.34,0.14,0.2,0.3,-0.12,-0.11,-0.11,0.1,-0.05,0.21,0.31,0.0,-0.07,-0.29,-0.24,-0.17,-0.24,-0.15,-0.16,-0.05,0.08,0.03,-0.18,-0.08,-0.25,0.29,0.06,0.19,0.06,-0.03,0.01,-0.23,0.18,-0.23,-0.01,-0.11,0.26,0.36,-0.06,0.45,0.32,0.44,0.48,0.36,0.26,0.24,-0.14,0.1,0.14,0.29,0.49,0.35,0.47,0.18,0.32,0.22,0.84,0.52,0.75,0.54,0.62,0.53,0.71,0.42,0.36,0.36,0.71,0.47,0.32,0.43,0.97,0.65,0.52,0.64,0.64,0.44,0.45,0.45,0.18,0.74,0.17,0.5,0.51,0.09,0.27,0.46,0.15,0.26,0.16,0.35,0.3,0.48,0.16,0.37,0.05,0.0,0.31,0.32,0.19,0.11,0.09,0.34,0.21,-0.03,0.2,0.07,0.15,0.06,-0.03,0.2,-0.01,-0.08,0.29,-0.21,-0.2,-0.1,-0.32,-0.11,-0.36,-0.37,-0.54,-0.29,-0.12,-0.2,-0.31,-0.37,0.06,-0.11,-0.32,-0.29,-0.19,-0.02,-0.23,-0.38,-0.24,-0.26,-0.22,0.34,-0.17,0.21,-0.35,0.06,-0.08,0.17,-0.08,-0.08,-0.08,0.13,-0.26,-0.06,-0.01,-0.15,-0.42,-0.16,-0.41,0.13,-0.04,0.16,-0.33,-0.06,-0.35,-0.55,-0.47,-0.08,-0.37,-0.3,-0.3,0.1,-0.45,-0.08,-0.19,0.09,-0.28,-0.25,-0.13,0.08,0.03,0.1,-0.1,0.1,-0.29,-0.03,0.11,0.24,-0.24,0.0,-0.06,0.09,-0.15,0.4,0.18,0.28,0.26,0.01,-0.05,0.12,-0.06,-0.29,0.05,0.25,0.03,0.26,0.39,0.11,0.24,0.03,0.33,0.04,0.51,0.39,0.21,-0.22,-0.08,-0.07,-0.14,0.14,0.17,0.21,0.02,-0.05,0.12,0.17,0.45,0.05,0.24,-0.13,0.29,0.54,0.19,0.43,0.29,-0.03,0.05,0.14,0.28,-0.06,0.04,0.06,-0.04,-0.05,0.04,-0.11,-0.22,0.26,0.02,-0.02,0.2,0.04,-0.21,-0.06,-0.16,-0.04,-0.04,0.01,-0.05,0.12,-0.2,-0.13,-0.18,-0.05,-0.19,-0.1,-0.13,-0.1,0.06,0.09,-0.13,0.07,0.01,0.36,-0.01,0.39,0.04,0.0,0.16,0.13,0.26,0.0,-0.01,0.15,-0.21,0.25,0.22,0.42,0.3,0.04,0.43,0.55,0.1,-0.01,-0.05,0.0,-0.19,0.05,0.09,0.17,-0.23,0.15,-0.02,0.18,0.12,0.32,-0.12,-0.06,0.5,0.18,0.06,0.26,0.25,0.22,0.11,0.35,0.65,0.28,-0.22,0.1,-0.09,-0.01,0.18,0.0,-0.13,0.01,-0.13,0.04,0.07,0.0,0.18,0.31,0.26,0.17,0.2,0.25,0.33,0.01,0.19,0.22,0.23,0.07,0.27,0.24,-0.1,-0.18,-0.24,-0.18,-0.27,-0.25,-0.08,-0.25,0.26,-0.18,-0.29,-0.14,-0.12,-0.26,-0.25,-0.03,0.2,0.19,-0.24,0.31,-0.24,-0.22,-0.15,-0.16,0.2,-0.21,-0.16,-0.16,-0.13,-0.52,-0.24,0.21,0.04,-0.18,-0.17,0.09,0.24,0.17,-0.12,-0.22,0.12,0.08,0.11,0.09,-0.22,0.09,0.12,-0.03,0.16,-0.06,-0.27,-0.16,-0.03,-0.04,-0.1,-0.1,0.07,0.14,0.06,0.02,-0.06,0.01,0.12,-0.13,-0.15,-0.16,0.0,0.21,0.07,0.05,0.06,0.27,-0.06,-0.13,0.18,0.1,0.07,0.0,0.1,0.04,-0.09,-0.3,-0.06,-0.29,-0.16,-0.05,0.04,-0.26,0.02,-0.02,0.14,-0.34,0.26,0.0,0.0,-0.03,0.06,-0.33,-0.03,0.12,-0.07,0.04,0.21,0.08,-0.01,0.04,-0.04,-0.02,-0.1,-0.12,-0.15,0.2,0.12,0.01,-0.14,0.45,-0.07,-0.27,-0.17,-0.01,-0.13,-0.1,0.19,-0.27,0.12,-0.1,0.17,0.3,0.11,0.29,-0.05,-0.21,0.19,-0.03,0.04,-0.08,0.31,-0.01,-0.01,0.29,-0.1,0.03,-0.05,0.03,0.04,-0.08,-0.14,-0.24,-0.19,0.11,0.3,0.15,0.27,0.14,-0.04,0.38,0.0,0.31,0.26,0.14,0.41,0.1,0.25,0.16,0.3,0.27,0.33,0.04,0.02,0.44,0.33,0.33,0.28,0.15,0.14,0.02,0.21,0.24,0.05,-0.03,-0.14,-0.06,0.02,-0.35,-0.23,-0.17,0.0,-0.1,-0.11,0.24,-0.26,-0.11,-0.02,0.1,-0.32,-0.29,-0.17,-0.16,-0.12,-0.17,-0.37,0.12,-0.22,0.51,0.07,0.04,0.03,0.48,0.6,0.19,-0.03,0.12,-0.13,-0.1,-0.04,0.08,0.47,0.74,0.25,0.26,0.19,0.02,0.41,0.18,0.31,0.37,0.15,0.17,0.25,-0.01,0.21,0.17,0.06,-0.17,0.29,-0.08,-0.1,0.14,-0.2,-0.22,0.06,0.01,0.18,0.07,0.28,-0.17,0.06,-0.12,0.05,-0.16,-0.29,-0.28,-0.12,-0.33,0.06,-0.21,-0.13,-0.21,0.21,-0.35,0.02,-0.18,-0.19,0.15,-0.18,-0.13,0.1,0.03,0.27,0.19,-0.08,0.17,0.18,0.31,0.35,0.28,0.31,0.29,0.33,0.27,0.58,0.26,0.32,0.1,0.22,-0.03,0.0,-0.08,-0.09,0.18,0.13,-0.04,0.15,0.33,0.15,0.07,0.05,0.12,0.32,0.29,0.09,0.06,-0.19,0.2,0.61,0.05,0.11,0.21,0.05,-0.05,0.13,0.04,0.33,0.15,-0.06,0.25,-0.05,0.17,0.25,-0.03,0.18,0.15,0.15,0.34,0.16,-0.04,0.29,0.04,-0.08,0.13,0.03,0.06,0.06,-0.08,-0.03,-0.07,0.15,0.04,-0.24,0.11,0.17,0.06,0.04,-0.12,0.06,0.06,0.12,-0.21,-0.25,-0.19,0.07,-0.28,0.16,0.32,0.06,0.03,-0.1,0.22,0.06,0.31,0.03,0.06,0.37,0.48,0.16,0.55,0.18,0.41,0.19,0.53,0.31,0.23,0.56,0.33,0.26,0.21,0.26,0.08,0.09,0.36,0.43,0.21,0.25,0.33,0.3,0.28,0.32,0.21,0.38,0.19,0.38,0.2,0.51,0.25,0.02,-0.01,0.18,-0.19,0.16,0.08,0.08,-0.17,-0.07,-0.07,-0.29,-0.06,-0.36,-0.22,-0.25,-0.34,-0.29,-0.58,-0.4,-0.21,-0.58,-0.49,-0.39,-0.61,-0.44,-0.42,-0.14,-0.37,-0.33,-0.4,-0.15,-0.31,-0.33,-0.18,-0.16,-0.29,0.02,0.21,0.15,-0.04,-0.19,-0.3,0.04,-0.17,0.02,0.21,-0.26,-0.15,-0.08,0.08,0.02,-0.06,-0.11,-0.29,-0.26,-0.35,-0.33,-0.01,-0.21,-0.25,-0.26,-0.11,-0.48,-0.21,-0.03,-0.22,-0.02,-0.36,-0.28,-0.18,0.01,-0.13,0.09,-0.05,-0.3,-0.5,-0.19,-0.26,-0.28,0.22,-0.05,-0.24,0.12,0.07,0.32,-0.29,-0.11,-0.04,-0.1,-0.01,-0.34,0.01,-0.34,-0.55,-0.38,-0.3,-0.34,-0.49,-0.13,-0.11,-0.8,-0.34,-0.59,-0.5,-0.32,-0.49,-0.48,-0.22,-0.35,-0.28,-0.34,0.04,-0.35,-0.33,-0.36,-0.44,-0.48,-0.36,-0.05,-0.39,-0.55,-0.41,-0.37,-0.36,-0.47,-0.4,-0.32,-0.42,-0.4,0.03,-0.18,-0.48,-0.24,-0.25,-0.52,-0.47,-0.27,-0.49,0.08,-0.44,-0.49,-0.3,-0.42,-0.16,-0.13,-0.35,-0.56,-0.32,-0.38,0.21,-0.05,0.01,-0.04,-0.1,0.32,0.21,0.3,-0.08,-0.3,0.19,0.04,-0.1,0.16,-0.17,0.03,0.49,0.12,-0.01,0.2,0.28,-0.03,0.04,0.08,0.29,0.14,0.32,0.02,0.15,0.18,0.0,-0.14,0.1,0.03,0.18,-0.2,-0.2,0.04,0.12,-0.14,0.12,0.29,0.17,-0.03,0.12,-0.06,0.18,0.0,0.24,-0.23,-0.16,0.22,-0.12,0.04,-0.15,-0.18,0.14,-0.42,-0.1,-0.12,-0.13,-0.44,0.0,0.07,0.06,-0.05,-0.24,0.04,-0.11,-0.02,-0.03,0.07,0.08,0.06,0.36,0.14,0.18,0.41,0.08,-0.26,0.15,0.13,-0.11,0.18,0.03,0.3,0.17,-0.04,0.01,0.08,0.09,0.01,-0.2,0.03,-0.01,0.03,-0.13,-0.19,-0.18,0.29,0.06,-0.31,-0.27,-0.4,-0.23,-0.32,-0.35,-0.36,-0.3,-0.13,0.15,-0.44,0.2,-0.34,0.25,-0.2,-0.2,0.0,-0.31,0.01,0.1,0.03,0.42,0.49,0.07,0.01,0.1,0.1,0.23,0.19,0.07,0.38,0.23,-0.18,-0.01,0.19,-0.01,-0.14,0.23,0.24,0.38,0.26,0.07,0.34,-0.13,0.04,0.08,0.04,-0.04,0.31,-0.01,-0.22,-0.03,-0.14,0.15,0.11,-0.29,-0.36,-0.36,0.03,-0.4,-0.28,-0.31,-0.04,-0.39,-0.23,-0.14,0.02,-0.02,0.0,-0.19,-0.19,0.2,-0.07,-0.15,-0.08,-0.06,0.05,0.01,-0.3,0.17,-0.06,-0.21,0.01,-0.03,-0.09,-0.04,-0.01,0.17,0.04,-0.16,-0.08,0.24,0.01,-0.32,-0.03,-0.07,-0.14,0.11,-0.3,-0.14,0.08,-0.02,0.03,0.28,0.01,0.29,-0.18,-0.24,-0.13,-0.08,-0.26,0.36,-0.2,-0.09,0.34,-0.11,0.09,0.1,0.12,-0.05,0.04,-0.03,-0.27,0.14,-0.01,-0.32,-0.14,0.0,-0.06,-0.03,0.09,-0.22,-0.12,0.12,-0.01,0.02,-0.14,0.1,-0.13,0.01,-0.02,-0.04,0.04,0.2,0.29,0.1,0.07,0.26,0.04,0.29,0.03,-0.07,-0.19,-0.22,-0.21,-0.24,-0.05,-0.47,-0.23,-0.3,-0.06,0.06,-0.41,-0.23,-0.05,-0.24,0.04,0.04,0.46,0.22,-0.1,0.1,0.19,-0.04,0.23,0.47,0.42,0.2,0.31,0.29,0.08,-0.1,0.01,0.06,-0.16,-0.35,-0.45,-0.15,-0.17,-0.35,-0.19,-0.42,-0.41,-0.4,-0.28,-0.02,-0.17,-0.07,-0.11,-0.25,-0.35,0.02,-0.21,-0.15,0.05,-0.03,-0.02,-0.23,0.0,-0.26,-0.03,0.1,-0.15,-0.04,-0.11,-0.06,-0.14,-0.12,0.09,0.15,0.08,0.0,0.22,-0.07,0.44,0.13,-0.18,-0.1,-0.07,0.3,0.22,-0.06,0.12,0.3,0.16,-0.04,0.12,-0.19,-0.08,0.17,-0.18,-0.17,0.14,0.34,0.22,0.19,0.33,0.22,0.22,0.1,0.4,0.51,0.19,0.53,0.12,0.1,0.13,0.03,0.43,0.2,0.08,0.35,0.1,0.28,0.12,0.1,0.2,0.36,-0.01,-0.22,0.06,0.05,-0.05,0.02,0.02,0.33,0.03,0.01,-0.15,0.18,-0.03,-0.24,0.14,-0.21,-0.04,0.11,0.04,0.03,0.18,-0.14,-0.43,-0.25,-0.13,0.14,-0.08,-0.12,-0.38,-0.11,0.09,-0.11,0.27,0.13,0.09,-0.04,0.21,0.02,-0.03,-0.2,0.13,0.08,-0.14,0.01,-0.09,-0.14,0.09,-0.08,0.34,0.09,0.35,0.12,-0.01,0.18,-0.16,0.04,-0.14,-0.07,0.04,-0.08,0.01,-0.04,-0.35,-0.34,-0.26,-0.3,-0.22,-0.22,-0.09,-0.1,0.04,-0.03,-0.13,-0.03,-0.11,-0.09,-0.03,-0.05,0.07,0.16,-0.03,0.2,0.11,0.26,0.17,-0.05,0.32,0.25,0.02,0.38,0.29,-0.14,0.3,-0.08,0.1,-0.07,-0.04,0.08,0.54,0.14,0.36,0.47,0.19,0.21,0.12,0.46,0.0,0.06,0.45,0.57,0.06,0.36,0.26,0.28,0.36,0.08,0.08,0.03,0.06,0.08,0.06,0.16,-0.25,0.03,0.21,-0.28,-0.12,-0.27,-0.11,-0.05,0.06,0.04,-0.17,0.2,-0.08,-0.14,-0.29,-0.18,0.02,-0.07,-0.15,0.07,-0.06,0.02,0.16,0.14,-0.14,0.24,-0.25,0.1,-0.09,-0.21,-0.13,0.09,0.1,-0.37,0.14,-0.2,-0.23,0.44,0.09,-0.35,-0.37,-0.06,-0.29,-0.27,-0.31,-0.13,-0.19,-0.03,0.16,0.11,0.1,0.1,0.28,0.19,-0.12,-0.03,-0.24,0.18,-0.02,0.16,-0.08,-0.13,-0.03,0.26,-0.03,-0.23,-0.29,-0.09,-0.12,-0.22,-0.19,0.08,0.39,-0.2,0.08,0.0,0.04,0.3,-0.2,-0.19,-0.14,-0.11,0.14,0.11,-0.12,-0.15,-0.1,0.08,-0.27,0.07,0.16,0.16,0.08,0.1,0.13,0.01,-0.12,0.0,-0.08,0.25,-0.13,-0.01,0.04,-0.13,0.29,0.12,0.11,-0.01,-0.11,-0.09,-0.06,-0.11,-0.06,0.15,0.12,-0.07,0.22,0.13,0.21,0.04,-0.07,0.12,0.18,0.14,-0.01,-0.2,-0.17,0.04,0.01,0.03,0.19,0.55,0.16,0.04,-0.05,0.32,0.08,0.04,0.16,-0.03,-0.18,0.23,0.41,-0.03,0.12,0.17,0.23,-0.1,0.04,0.11,-0.01,-0.23,-0.01,0.1,-0.03,-0.11,-0.07,-0.06,-0.04,-0.17,-0.04,0.08,-0.09,-0.47,-0.36,-0.02,-0.36,-0.38,-0.26,0.0,-0.48,-0.09,-0.49,-0.46,-0.32,-0.31,-0.05,-0.21,-0.15,-0.15,-0.25,-0.08,-0.27,0.14,-0.07,0.13,0.24,-0.03,0.02,-0.11,-0.21,0.25,-0.2,-0.44,0.14,-0.11,-0.06,-0.2,-0.18,-0.25,-0.37,-0.17,0.0,0.15,-0.35,-0.1,-0.39,-0.03,-0.01,0.04,0.0,-0.14,-0.25,-0.08,0.02,-0.26,-0.5,-0.19,0.08,0.24,0.09,0.11,0.02,-0.04,-0.13,0.16,0.09,0.02,-0.14,0.06,0.27,0.17,0.0,0.16,0.23,0.65,0.14,0.12,0.06,-0.03,0.04,-0.21,-0.08,0.06,0.14,-0.08,0.31,-0.06,0.47,-0.1,-0.12,-0.15,-0.12,-0.13,-0.06,-0.29,-0.24,-0.43,-0.1,-0.18,-0.31,-0.08,-0.31,-0.13,0.04,-0.09,-0.14,-0.3,-0.24,-0.17,-0.17,-0.23,-0.24,-0.22,-0.07,0.04,-0.28,0.12,0.26,-0.02,-0.19,-0.32,-0.22,-0.37,-0.17,-0.29,-0.17,0.01,-0.16,-0.3,-0.28,0.03,-0.03,-0.01,-0.11,-0.04,-0.3,-0.03,-0.51,-0.07,-0.47,-0.32,-0.31,-0.14,-0.34,-0.03,-0.3,-0.28,-0.03,-0.14,-0.31,-0.38,-0.58,-0.52,-0.45,-0.29,-0.11,-0.52,-0.36,-0.29,-0.33,-0.35,-0.12,-0.16,-0.04,0.03,-0.13,-0.25,-0.27,-0.13,0.25,-0.33,-0.08,-0.05,0.24,-0.11,0.16,-0.12,0.06,0.15,-0.26,-0.04,0.07,-0.32,0.12,-0.17,-0.13,0.06,-0.05,-0.06,-0.04,-0.21,0.48,0.15,-0.02,0.02,-0.16,0.1,0.45,0.1,0.03,-0.15,0.06,-0.02,0.27,0.09,-0.17,-0.06,0.12,0.29,0.17,0.43,0.19,-0.03,0.27,0.13,0.28,0.03,0.42,0.18,0.35,0.26,0.22,0.33,0.51,-0.07,0.16,0.11,-0.1,-0.05,-0.07,-0.02,0.02,0.24,-0.06,0.2,-0.09,-0.15,-0.08,0.02,-0.11,-0.05,-0.08,0.09,-0.19,-0.18,-0.13,-0.07,0.22,-0.03,-0.24,-0.24,-0.1,-0.17,-0.1,-0.2,-0.24,-0.29,-0.07,-0.37,-0.44,-0.48,-0.2,-0.05,-0.26,-0.27,-0.4,-0.15,-0.25,-0.41,-0.06,-0.07,-0.21,-0.28,-0.15,-0.28,-0.09,-0.29,-0.06,0.0,-0.11,0.01,-0.33,-0.23,0.13,-0.09,-0.04,0.03,-0.28,0.03,-0.33,-0.04,-0.1,-0.18,0.16,-0.22,-0.07,0.41,0.23,-0.2,0.08,0.45,-0.25,0.01,0.16,0.2,0.04,0.37,0.34,0.0,0.22,0.06,0.2,0.16,0.37,0.36,0.21,0.29,0.08,0.08,0.13,0.09,0.25,0.3,0.0,0.29,-0.05,0.1,-0.21,0.14,-0.1,0.18,0.38,-0.01,0.1,0.17,0.05,0.29,0.04,0.17,0.18,-0.17,0.4,0.1,0.25,0.14,-0.01,0.1,0.08,0.01,0.31,0.23,0.77,0.7,0.36,0.34,0.11,0.29,0.37,0.22,0.43,0.29,-0.1,0.13,-0.03,0.42,0.46,0.02,0.22,0.18,-0.08,-0.03,-0.03,-0.03,-0.17,-0.11,0.04,-0.01,-0.1,-0.2,-0.04,-0.18,0.0,0.0,-0.25,0.1,-0.25,-0.12,-0.01,0.08,-0.15,-0.05,-0.23,-0.15,-0.23,-0.31,-0.1,-0.04,-0.14,-0.03,0.01,-0.08,-0.01,0.01,-0.22,-0.06,-0.03,-0.14,0.06,0.27,-0.38,-0.22,-0.08,-0.39,-0.2,0.0,-0.05,0.02,-0.2,-0.11,-0.16,0.09,-0.09,-0.02,-0.1,-0.14,-0.08,0.06,0.27,-0.09,-0.08,0.24,-0.22,0.25,-0.22,-0.01,-0.15,-0.35,-0.35,-0.08,-0.22,-0.12,0.15,-0.19,0.21,0.35,0.14,-0.03,0.21,0.02,0.04,-0.25,0.21,0.02,0.14,-0.07,0.16,0.31,0.14,-0.21,0.04,0.04,0.29,-0.36,-0.09,0.21,-0.04,-0.22,0.06,-0.03,-0.02,-0.06,-0.39,-0.13,0.08,-0.09,-0.05,0.06,-0.15,0.25,-0.08,0.0,0.27,-0.12,-0.18,0.1,0.03,-0.08,0.11,-0.18,0.4,0.06,-0.01,-0.03,-0.08,0.0,-0.05,0.33,0.39,-0.05,0.23,-0.03,0.58,0.16,0.38,-0.03,0.38,0.17,0.39,0.09,0.09,0.29,0.1,0.78,0.36,0.56,0.38,0.44,0.3,0.47,0.2,0.36,0.42,0.5,-0.11,0.0,0.37,-0.01,0.35,0.57,0.02,0.21,0.25,0.19,0.42,0.45,0.31,0.01,0.49,0.25,0.09,0.36,0.56,0.33,0.19,0.32,0.08,0.12,-0.09,0.24,0.41,0.35,0.09,0.12,0.23,0.46,0.33,0.09,0.3,0.35,-0.04,0.19,-0.17,0.08,-0.19,0.33,-0.15,-0.16,0.03,-0.17,-0.18,-0.07,-0.09,0.3,0.08,0.04,-0.01,0.1,0.15,0.1,0.08,-0.12,-0.02,0.25,-0.18,-0.01,0.07,-0.23,-0.18,0.21,0.01,-0.19,-0.33,-0.32,0.01,-0.09,-0.05,0.1,0.16,0.13,-0.06,-0.3,-0.27,0.09,-0.07,-0.17,-0.18,-0.15,-0.18,-0.18,-0.1,0.02,0.01,-0.45,-0.01,-0.22,-0.3,-0.3,-0.29,-0.26,-0.22,-0.22,-0.14,-0.29,-0.16,-0.13,-0.15,-0.2,-0.36,0.28,0.02,-0.07,-0.42,0.0,-0.28,0.06,-0.18,-0.3,-0.39,-0.21,-0.31,-0.16,0.13,-0.03,-0.19,-0.22,-0.15,-0.41,-0.43,-0.19,-0.02,-0.47,-0.31,-0.17,-0.07,-0.19,-0.42,0.19,0.05,-0.27,0.13,0.01,0.0,0.45,0.24,0.19,0.28,0.06,0.1,0.26,0.1,0.41,0.22,0.39,-0.01,0.38,0.47,0.12,0.28,0.58,0.42,0.44,0.03,0.5,0.2,0.09,0.21,0.09,-0.08,-0.14,0.02,0.06,0.05,0.0,0.33,0.06,0.74,0.16,0.06,-0.14,0.03,-0.24,0.07,-0.03,0.35,-0.07,0.02,0.34,-0.08,-0.24,-0.1,0.28,-0.09,-0.05,0.05,0.03,-0.07,-0.03,0.33,-0.08,0.01,-0.03,0.12,0.0,-0.18,0.06,0.19,0.24,-0.08,0.09,-0.16,0.33,-0.09,0.2,0.2,0.37,0.05,0.1,0.24,0.59,0.23,0.2,0.15,0.36,0.29,0.24,0.12,0.12,0.04,0.38,0.18,0.19,0.21,0.0,-0.1,0.13,0.18,-0.1,-0.22,0.01,0.15,-0.1,-0.19,-0.13,-0.38,0.22,-0.31,-0.09,-0.16,-0.25,-0.04,0.1,-0.29,-0.18,-0.38,-0.25,0.11,-0.26,-0.05,-0.24,-0.13,-0.39,-0.33,-0.24,0.07,-0.12,-0.44,-0.03,-0.31,-0.16,-0.02,0.09,-0.07,-0.02,-0.16,-0.12,-0.03,-0.25,-0.08,-0.21,0.15,0.02,-0.23,-0.26,0.14,-0.24,-0.23,0.31,-0.1,0.04,-0.28,-0.13,-0.07,-0.4,-0.16,-0.33,-0.19,-0.27,-0.04,-0.23,-0.42,-0.56,-0.37,-0.12,0.04,-0.36,-0.47,-0.31,-0.18,-0.15,-0.14,-0.19,-0.03,0.37,0.07,-0.19,-0.11,-0.01,-0.18,-0.23,0.05,-0.27,-0.1,-0.07,-0.41,-0.2,-0.27,-0.53,-0.48,-0.75,-0.19,-0.39,-0.55,-0.48,-0.46,-0.14,-0.23,-0.5,-0.02,-0.24,-0.27,-0.32,-0.31,-0.16,-0.12,-0.22,-0.15,-0.4,-0.29,-0.12,-0.26,-0.41,-0.31,-0.14,-0.27,-0.05,-0.16,-0.35,-0.28,-0.22,-0.04,-0.27,0.19,-0.36,0.25,-0.26,-0.06,-0.15,-0.03,-0.42,-0.38,-0.2,-0.1,-0.14,-0.15,-0.27,-0.36,-0.38,-0.35,0.01,-0.22,-0.48,-0.36,-0.28,0.03,-0.06,-0.03,-0.33,-0.05,0.04,-0.05,0.04,-0.09,-0.19,0.22,0.23,-0.24,-0.01,-0.05,0.32,-0.07,-0.09,-0.08,0.0,0.02,-0.24,-0.26,0.05,-0.05,-0.13,0.16,0.41,0.01,0.36,0.09,0.22,0.1,0.08,0.22,0.08,-0.3,-0.08,0.01,-0.28,-0.26,-0.41,-0.3,-0.26,-0.61,-0.24,-0.06,-0.53,-0.27,-0.59,-0.5,0.02,-0.25,-0.28,-0.18,-0.23,-0.27,-0.05,-0.18,-0.18,-0.12,-0.33,-0.2,0.21,-0.13,-0.38,0.01,0.01,-0.41,0.05,-0.03,-0.11,-0.12,-0.11,-0.32,0.2,-0.27,-0.03,0.19,0.04,-0.13,0.0,0.02,-0.05,-0.16,0.18,0.26,-0.34,-0.23,-0.17,-0.2,-0.3,0.22,0.43,-0.11,-0.04,0.57,-0.09,0.49,-0.04,-0.29,-0.03,0.13,-0.09,-0.24,0.1,-0.11,0.16,0.05,0.02,-0.29,-0.24,-0.15,-0.32,-0.36,-0.55,-0.42,-0.08,-0.19,-0.45,-0.3,-0.48,0.06,0.15,-0.04,0.06,0.01,-0.23,-0.31,-0.34,-0.27,0.26,-0.08,0.29,-0.13,0.21,-0.13,-0.15,-0.19,0.12,-0.32,-0.13,-0.12,-0.21,0.04,0.07,-0.19,-0.24,-0.13,-0.05,-0.15,-0.05,-0.04,-0.28,0.15,-0.07,-0.04,0.06,0.06,-0.07,0.19,-0.07,-0.09,0.11,0.27,-0.32,-0.07,0.05,0.26,-0.11,0.02,-0.12,0.01,-0.27,-0.03,0.0,0.03,-0.02,-0.15,-0.04,-0.23,0.19,0.19,0.06,0.06,0.09,0.3,0.13,0.18,0.48,0.37,0.76,0.37,0.37,0.24,0.15,0.23,0.36,0.11,0.73,0.52,0.14,0.25,0.15,-0.18,0.18,-0.41,-0.29,-0.24,0.04,-0.35,-0.06,0.0,-0.19,-0.22,-0.22,-0.11,-0.23,-0.13,-0.25,0.0,-0.13,-0.18,-0.08,0.01,-0.27,0.1,-0.16,-0.17,0.02,-0.08,0.04,-0.14,0.2,0.0,-0.15,-0.1,0.04,-0.01,0.23,0.11,0.04,-0.02,0.04,-0.25,0.02,0.13,0.17,-0.13,-0.14,0.05,0.25,-0.1,-0.05,-0.4,0.04,-0.19,-0.24,-0.19,-0.09,-0.14,-0.07,-0.15,-0.29,-0.06,0.2,-0.1,-0.1,-0.28,-0.34,0.04,-0.34,-0.1,-0.17,0.0,0.18,-0.04,0.04,-0.18,-0.03,0.03,0.33,0.05,0.34,-0.16,-0.16,0.06,0.03,0.08,-0.12,-0.01,-0.2,0.16,0.2,0.3,0.01,0.02,0.65,0.27,0.26,0.34,0.4,0.39,0.49,0.2,0.07,0.14,0.17,0.28,0.59,0.45,0.18,0.8,0.35,0.08,0.3,0.34,0.21,0.67,0.59,0.47,0.22,-0.06,0.17,0.18,0.13,0.13,-0.3,-0.27,0.08,0.0,0.09,-0.35,-0.03,-0.23,-0.42,-0.4,-0.09,-0.18,-0.14,-0.12,-0.05,-0.16,0.01,-0.05,-0.09,0.01,0.0,-0.41,-0.23,-0.32,-0.08,-0.14,-0.24,0.01,-0.21,-0.42,-0.21,-0.26,-0.27,-0.08,0.0,-0.2,-0.08,-0.04,0.08,0.03,0.0,-0.04,-0.17,-0.03,-0.21,-0.09,-0.31,-0.18,-0.2,-0.2,-0.1,-0.11,-0.03,-0.12,-0.22,-0.25,-0.22,-0.14,-0.47,-0.26,-0.48,-0.49,-0.33,-0.47,-0.32,-0.63,-0.44,0.13,-0.72,-0.66,-0.15,-0.23,-0.34,-0.29,-0.32,-0.29,-0.17,-0.2,-0.2,-0.17,-0.26,-0.25,-0.23,-0.12,0.02,-0.15,0.09,-0.27,-0.36,-0.33,-0.53,-0.24,0.01,-0.06,-0.15,-0.42,-0.15,-0.61,-0.15,-0.21,0.1,-0.2,-0.03,-0.06,0.1,-0.07,-0.15,0.26,-0.24,-0.03,-0.11,-0.09,-0.07,0.44,0.3,0.17,0.13,0.07,0.29,0.26,0.01,0.15,0.11,0.45,0.12,0.15,0.41,0.25,0.09,0.07,-0.17,-0.03,0.08,0.15,0.21,0.15,-0.08,0.0,0.38,0.13,0.25,0.11,-0.12,0.01,0.17,0.25,0.2,0.29,0.19,-0.03,-0.24,-0.19,-0.03,-0.06,-0.19,-0.19,-0.26,-0.32,-0.39,-0.21,-0.4,-0.27,-0.24,-0.06,-0.22,-0.19,-0.3,-0.3,-0.26,-0.61,-0.32,-0.13,-0.31,-0.24,-0.37,-0.34,-0.08,-0.11,-0.06,-0.34,0.22,0.15,0.19,-0.11,-0.19,-0.23,-0.09,-0.37,-0.31,-0.08,-0.12,-0.05,-0.1,-0.28,-0.03,-0.29,-0.15,-0.36,-0.11,-0.16,-0.1,-0.22,-0.11,-0.46,-0.15,0.0,-0.33,0.1,-0.05,-0.25,-0.22,-0.3,-0.1,-0.11,-0.24,-0.18,-0.06,0.03,-0.09,0.58,-0.12,0.04,-0.06,-0.13,0.23,-0.22,-0.12,-0.13,0.02,-0.07,-0.03,0.2,0.08,0.17,0.09,0.06,-0.19,-0.36,0.06,0.08,0.15,0.1,-0.08,-0.11,-0.01,0.08,-0.02,-0.25,0.17,-0.13,0.0,-0.3,-0.27,-0.09,-0.28,-0.24,-0.01,-0.41,-0.18,-0.12,-0.16,-0.19,-0.11,-0.09,-0.04,-0.18,0.21,-0.29,-0.19,-0.32,-0.19,-0.28,0.18,-0.01,0.04,-0.21,-0.51,-0.05,-0.28,-0.03,-0.14,0.0,0.51,-0.11,-0.15,-0.41,-0.23,0.08,-0.25,-0.19,0.28,-0.37,-0.2,0.04,-0.27,-0.23,-0.18,0.12,-0.08,0.11,0.1,-0.18,0.03,0.01,0.25,-0.23,0.03,0.09,0.17,0.06,0.04,-0.09,0.03,0.22,0.34,0.17,0.38,-0.12,-0.08,-0.1,-0.1,-0.08,-0.16,-0.12,-0.1,-0.03,-0.09,-0.03,0.02,0.05,-0.23,-0.32,0.2,0.1,-0.12,-0.16,-0.02,0.14,-0.13,-0.03,0.14,-0.04,-0.13,0.12,0.32,0.16,0.28,0.19,0.0,0.15,0.41,0.2,0.48,0.29,0.23,0.7,0.44,0.36,0.03,0.41,-0.12,0.28,0.3,0.26,0.31,0.12,0.09,0.29,0.23,0.27,0.21,0.23,0.3,0.35,0.09,0.15,0.38,0.26,0.2,0.27,0.43,0.05,0.08,0.17,0.23,0.37,0.34,0.52,0.01,0.21,-0.08,-0.1,0.33,0.1,0.04,0.34,0.28,-0.07,0.2,-0.08,0.29,0.57,0.03,0.21,0.24,0.28,0.33,0.35,0.29,0.6,0.56,0.34,0.17,0.4,0.39,-0.08,0.3,0.32,0.54,0.18,0.09,-0.26,0.22,0.1,0.03,0.24,0.0,0.06,0.17,0.33,0.11,0.16,0.11,0.15,0.32,0.13,0.31,0.1,-0.05,0.44,0.22,-0.11,0.04,-0.07,0.29,-0.03,-0.07,0.0,-0.03,0.28,0.08,-0.11,0.03,0.1,0.19,0.33,-0.04,0.04,0.29,0.39,0.25,-0.06,0.4,0.58,0.52,0.17,0.52,0.24,0.36,0.46,0.36,0.23,0.48,0.41,0.1,0.29,0.01,0.59,0.12,0.25,0.34,0.02,0.18,0.09,-0.06,0.56,-0.11,-0.13,-0.23,0.08,-0.27,0.09,-0.16,-0.38,-0.23,0.02,-0.03,0.14,0.04,-0.2,0.25,0.05,0.38,0.04,0.25,-0.12,-0.01,-0.05,0.09,0.15,0.22,-0.08,0.06,0.02,-0.05,0.01,0.03,-0.21,0.19,0.01,0.17,-0.18,-0.01,-0.23,0.19,-0.04,-0.14,-0.06,-0.08,-0.37,-0.05,-0.13,-0.19,-0.17,-0.35,-0.16,-0.17,-0.17,-0.08,0.08,-0.22,0.06,0.18,0.5,-0.02,-0.07,0.02,0.86,0.1,0.2,0.4,0.62,0.41,0.57,0.54,0.68,0.41,0.74,0.68,0.6,0.21,0.42,0.36,0.43,0.41,0.65,0.46,0.39,0.4,0.54,0.31,0.88,0.46,0.55,0.46,0.45,0.2,0.28,0.28,0.48,0.36,0.37,0.35,0.4,0.17,0.2,0.14,0.46,0.26,0.35,0.0,0.09,0.2,0.1,0.29,0.17,0.06,0.36,0.16,0.24,0.35,0.26,0.14,0.18,0.03,0.09,0.06,0.1,0.04,-0.08,0.14,-0.11,0.06,-0.26,-0.04,0.06,-0.12,-0.18,-0.08,0.04,-0.18,0.12,-0.39,-0.18,-0.18,0.18,-0.1,-0.03,-0.01,0.29,0.25,0.23,0.41,0.16,-0.14,0.06,0.01,0.17,0.33,0.23,0.1,0.09,0.12,0.16,-0.07,-0.08,0.1,0.0,-0.01,-0.18,0.02,-0.11,0.14,-0.08,-0.43,-0.37,-0.43,-0.04,-0.2,-0.48,-0.45,-0.33,-0.31,-0.27,-0.14,-0.59,-0.24,-0.34,-0.38,0.09,-0.38,-0.37,-0.29,-0.42,-0.33,-0.25,-0.24,0.03,0.18,-0.12,0.13,0.39,-0.05,0.11,0.32,0.14,0.19,0.36,0.31,0.32,0.46,0.2,-0.24,0.14,0.06,0.28,0.13,0.03,0.13,-0.07,-0.08,-0.14,0.0,0.19,0.06,0.12,0.03,0.01,-0.07,0.07,-0.09,0.3,0.24,0.09,0.06,-0.03,-0.09,-0.03,0.13,-0.29,-0.04,-0.23,-0.04,0.1,-0.26,-0.21,-0.13,-0.18,0.04,-0.34,0.03,0.35,-0.02,0.1,0.12,-0.1,0.32,-0.1,-0.09,-0.08,-0.14,0.05,0.13,0.15,-0.09,-0.01,0.0,0.1,0.25,0.11,0.0,0.22,0.01,-0.08,-0.14,0.28,-0.13,0.17,0.13,-0.04,-0.14,-0.14,0.11,0.22,-0.13,0.2,0.0,0.04,-0.3,-0.16,0.14,-0.25,-0.23,-0.57,-0.44,-0.29,-0.27,-0.11,-0.35,0.02,-0.4,-0.28,-0.46,-0.35,-0.39,-0.45,-0.43,-0.55,-0.27,-0.42,-0.29,-0.16,-0.58,-0.53,-0.73,-0.2,-0.03,-0.29,-0.15,-0.23,-0.6,-0.19,-0.48,-0.26,-0.53,-0.3,-0.03,-0.03,-0.1,-0.22,-0.11,0.0,0.04,-0.22,0.06,-0.04,0.19,0.11,0.26,0.17,0.12,0.35,0.42,0.1,0.1,0.15,0.04,0.24,0.01,-0.21,-0.26,-0.13,0.03,-0.19,-0.28,-0.03,-0.31,-0.2,-0.26,-0.01,-0.32,-0.21,0.03,-0.28,-0.1,0.18,-0.07,0.01,0.05,0.27,-0.25,-0.12,0.13,0.14,-0.16,0.09,-0.16,-0.03,-0.21,0.08,-0.19,-0.19,0.25,0.15,0.06,0.02,-0.1,-0.09,-0.32,-0.14,-0.3,-0.31,-0.29,-0.1,-0.33,-0.25,-0.29,-0.28,-0.33,-0.24,-0.06,0.09,-0.24,-0.33,-0.2,-0.33,0.04,0.0,-0.14,0.31,0.04,0.19,-0.07,-0.08,-0.1,0.14,-0.05,-0.11,-0.2,-0.03,-0.21,0.27,0.33,-0.15,0.01,-0.16,-0.07,-0.24,0.28,0.01,0.29,0.51,-0.06,0.21,-0.03,0.37,0.04,0.33,0.37,-0.11,0.0,0.1,0.12,-0.11,-0.19,-0.2,0.11,-0.37,-0.04,0.02,0.19,0.39,-0.08,-0.16,-0.07,-0.15,-0.18,0.2,-0.23,-0.03,0.2,-0.03,-0.19,0.1,-0.12,0.06,-0.3,-0.07,-0.28,0.34,0.14,0.22,-0.27,-0.05,-0.06,0.24,0.06,-0.03,0.06,0.06,0.28,0.3,0.3,0.43,0.56,0.17,0.16,0.55,0.69,0.54,0.2,0.52,0.25,0.01,0.35,-0.17,0.18,-0.14,0.08,-0.26,0.06,0.04,-0.35,-0.12,-0.23,0.08,0.01,-0.15,-0.13,0.02,-0.34,-0.29,-0.12,0.1,0.17,-0.21,0.03,0.01,0.04,0.56,0.35,0.01,-0.29,0.14,0.13,-0.02,-0.03,-0.12,0.06,-0.01,0.26,-0.19,-0.06,0.03,-0.1,-0.15,0.11,-0.02,-0.06,-0.12,0.21,0.43,0.06,0.15,-0.12,0.54,-0.04,0.09,0.04,-0.13,0.24,-0.03,-0.06,0.0,-0.08,0.02,-0.31,-0.28,0.12,-0.18,-0.34,-0.25,-0.07,-0.28,-0.32,-0.28,-0.18,-0.34,0.04,0.02,-0.18,-0.26,-0.23,-0.15,-0.28,-0.18,0.13,0.0,0.09,-0.04,-0.18,-0.12,-0.37,0.14,-0.32,-0.26,-0.14,-0.16,0.32,0.12,-0.09,-0.22,-0.38,-0.11,-0.06,-0.12,-0.14,-0.25,-0.42,-0.07,-0.06,-0.27,-0.32,-0.12,-0.25,-0.09,-0.26,-0.05,-0.26,-0.46,-0.16,-0.27,0.0,0.02,-0.07,0.03,-0.03,0.02,-0.22,-0.09,0.09,-0.03,0.01,0.01,0.02,0.04,0.15,0.18,0.06,0.23,0.36,0.24,0.38,0.08,0.45,0.24,-0.04,0.15,0.17,0.17,0.1,0.35,0.03,0.3,0.24,0.45,0.21,0.15,0.17,0.17,0.21,0.26,0.22,-0.16,0.23,0.16,0.06,-0.27,-0.24,0.42,0.05,0.04,0.13,-0.04,0.03,-0.37,-0.18,0.14,-0.07,0.1,0.06,-0.2,-0.12,-0.32,0.18,0.0,0.19,-0.2,0.26,0.0,0.0,0.17,0.05,-0.12,-0.15,-0.07,0.01,-0.17,0.05,0.13,0.08,0.16,-0.04,0.22,0.23,-0.11,-0.13,0.02,0.47,0.12,0.1,-0.09,-0.14,0.07,0.01,-0.11,-0.29,-0.43,-0.3,-0.13,-0.09,-0.39,-0.26,-0.19,-0.24,0.03,-0.22,-0.28,-0.23,0.09,0.07,-0.13,-0.13,-0.36,-0.18,-0.45,-0.44,-0.26,0.09,-0.1,0.04,-0.2,0.04,-0.09,-0.23,0.01,-0.06,-0.25,0.16,0.41,0.11,0.02,0.08,0.01,0.29,0.1,0.14,-0.06,0.38,-0.05,0.14,0.12,0.03,0.23,-0.18,-0.04,0.04,0.12,-0.21,0.02,-0.03,0.06,-0.32,-0.07,0.04,0.14,0.16,0.12,0.09,-0.11,0.25,-0.12,-0.21,0.01,0.23,0.17,0.2,-0.03,0.42,-0.09,0.07,-0.1,0.13,0.2,0.01,-0.25,0.08,-0.05,-0.2,-0.16,0.09,-0.26,-0.19,0.08,0.1,-0.14,-0.14,0.22,0.06,0.19,-0.03,0.11,-0.24,-0.11,0.12,0.18,-0.06,0.16,0.08,-0.25,-0.07,0.04,0.22,0.29,0.15,0.1,-0.06,0.0,0.12,0.5,0.12,-0.2,0.1,-0.22,0.08,0.27,-0.16,-0.21,0.08,-0.07,-0.05,0.33,0.1,0.17,-0.03,-0.01,0.27,0.13,0.36,0.13,0.08,0.29,0.27,-0.05,-0.12,0.1,-0.03,-0.14,0.07,-0.09,0.02,0.31,-0.03,0.19,0.1,-0.09,0.1,0.06,-0.23,-0.16,0.03,-0.28,-0.02,0.3,-0.11,0.01,-0.08,-0.15,0.04,0.31,-0.18,0.18,0.16,0.02,-0.1,0.1,-0.2,-0.21,-0.05,0.17,-0.14,0.22,-0.08,0.17,0.13,0.31,0.37,0.32,0.16,0.24,0.15,0.32,0.46,0.2,0.31,0.4,0.38,0.23,0.6,0.32,0.11,0.16,0.1,0.0,0.31,0.19,0.43,-0.11,0.35,0.2,0.32,0.53,0.3,0.35,0.65,0.48,0.22,0.68,0.44,0.45,0.28,0.53,0.34,0.5,0.47,0.54,0.32,0.39,0.31,0.26,0.19,0.29,0.4,0.57,0.53,0.3,0.17,0.29,0.68,0.8,0.27,0.65,0.29,0.24,0.18,0.55,0.42,0.57,0.7,0.42,0.24,0.07,0.25,0.15,0.06,0.28,0.21,0.37,0.11,0.26,0.06,0.24,0.16,0.21,0.28,0.32,0.13,0.43,0.34,0.03,-0.02,0.56,0.17,0.27,0.15,0.42,0.38,0.34,0.44,0.31,0.32,0.2,0.29,0.52,0.6,0.01,0.5,0.62,0.5,0.61,0.28,0.34,0.52,0.18,0.35,0.33,0.06,0.25,0.42,0.53,0.42,0.26,0.73,0.38,0.23,0.78,0.53,0.27,0.32,0.68,0.6,0.5,0.31,0.22,0.6,0.67,0.14,0.04,0.24,0.77,0.59,0.66,0.27,0.48,0.38,0.23,0.31,0.4,0.43,0.41,0.3,0.17,0.41,0.46,0.28,0.1,0.21,0.21,0.11,0.21,0.02,0.3,-0.08,0.19,0.01,0.18,0.02,0.28,-0.01,-0.1,-0.17,-0.06,-0.26,-0.37,-0.12,0.06,-0.27,0.02,-0.23,0.02,0.08,-0.23,-0.39,0.15,-0.12,-0.28,-0.1,-0.1,0.01,-0.03,-0.14,-0.2,0.64,0.0,-0.15,-0.12,-0.26,0.24,-0.37,-0.42,-0.04,-0.19,-0.29,-0.03,-0.18,0.01,0.4,-0.14,0.06,-0.16,-0.18,0.27,-0.14,0.12,-0.04,-0.15,-0.07,0.06,-0.11,-0.12,0.01,0.14,0.06,-0.05,-0.04,-0.05,0.28,-0.06,0.14,-0.03,0.14,-0.03,0.28,0.19,0.2,0.19,0.02,0.21,0.22,-0.05,-0.07,0.04,0.09,-0.08,-0.06,0.27,0.2,-0.23,-0.17,-0.12,0.06,-0.18,-0.22,-0.28,-0.47,-0.05,-0.07,-0.36,0.27,-0.03,-0.28,0.24,-0.3,-0.15,-0.42,-0.03,0.1,0.32,0.08,0.09,0.29,0.17,0.37,0.04,0.19,0.14,0.32,0.06,0.68,0.12,0.26,0.2,0.06,0.49,0.29,0.03,0.02,0.17,0.04,0.23,0.04,0.29,0.08,-0.1,0.24,-0.13,0.06,0.01,-0.08,0.21,-0.2,0.01,-0.06,-0.16,0.38,-0.01,0.08,0.23,0.38,0.15,0.11,0.28,0.36,0.38,0.13,0.41,0.25,0.39,0.18,0.47,0.48,0.03,0.21,0.13,0.23,-0.27,-0.03,-0.32,-0.29,-0.13,0.06,-0.18,0.21,-0.26,-0.44,-0.15,-0.32,-0.06,-0.15,-0.17,-0.3,0.01,0.19,-0.11,0.16,0.13,0.06,0.17,0.02,0.3,0.2,0.08,0.06,0.37,0.17,0.07,0.23,0.37,0.07,0.49,0.29,0.19,0.71,-0.03,0.44,0.68,0.5,0.56,0.63,0.37,0.53,0.4,0.43,0.15,0.2,0.18,0.04,0.32,0.17,0.2,0.03,0.0,-0.04,0.0,0.35,0.0,-0.17,-0.04,0.07,0.52,-0.04,-0.09,-0.14,0.05,0.15,-0.06,0.0,-0.06,-0.13,0.0,-0.21,-0.24,-0.43,-0.19,-0.18,-0.32,-0.16,-0.04,-0.07,-0.09,-0.53,-0.06,-0.47,-0.19,-0.05,-0.27,-0.1,-0.07,-0.08,0.22,0.22,0.5,0.06,0.13,0.23,0.08,0.2,-0.11,0.81,0.02,0.36,0.12,0.2,0.28,0.21,0.11,0.39,0.26,0.08,0.1,0.22,0.24,0.2,0.12,-0.11,-0.1,0.02,-0.12,0.45,-0.06,0.13,0.16,0.12,0.01,-0.22,-0.01,0.15,-0.15,0.02,-0.29,-0.05,0.0,-0.04,-0.43,-0.04,-0.16,-0.03,-0.13,-0.3,-0.26,-0.2,-0.36,-0.15,-0.19,-0.29,-0.18,0.02,-0.24,0.03,-0.15,-0.09,-0.15,-0.01,0.12,0.04,0.04,0.02,-0.03,-0.09,0.09,0.06,-0.16,0.1,-0.01,0.26,0.09,0.06,-0.17,-0.16,0.0,-0.25,-0.04,0.03,0.08,0.15,-0.03,-0.02,0.1,-0.04,0.09,0.47,-0.23,-0.2,0.03,-0.23,0.3,0.04,0.09,-0.1,0.14,0.34,0.3,0.35,0.23,0.21,0.53,0.22,0.38,0.3,0.37,0.36,0.01,0.29,0.16,0.34,0.17,0.2,0.32,0.04,0.04,0.16,0.15,0.48,-0.06,0.47,0.2,0.31,0.48,0.09,-0.03,0.14,0.44,0.25,-0.01,-0.25,0.04,0.13,-0.07,-0.03,0.08,-0.08,-0.09,-0.05,0.06,-0.03,0.13,-0.1,-0.19,-0.2,0.08,-0.16,-0.21,0.12,-0.08,0.08,0.03,-0.05,0.03,0.12,0.11,-0.14,0.28,0.5,0.24,0.21,0.11,0.17,0.27,0.07,0.13,-0.04,0.05,-0.02,0.16,-0.14,-0.12,0.14,-0.18,0.36,0.04,-0.07,0.02,0.01,-0.12,0.24,-0.13,-0.03,-0.01,-0.11,-0.2,-0.41,-0.16,-0.15,-0.06,-0.03,0.42,-0.26,-0.09,0.09,-0.15,0.14,-0.34,0.16,-0.07,0.06,-0.2,0.03,-0.15,0.07,0.06,-0.01,-0.22,-0.19,-0.17,-0.06,-0.19,-0.31,-0.58,-0.36,-0.13,-0.12,-0.24,-0.1,-0.26,-0.16,-0.57,-0.1,-0.41,-0.28,0.0,0.02,-0.28,-0.11,-0.18,-0.11,-0.12,-0.29,0.17,-0.05,-0.18,-0.01,0.23,0.06,-0.05,-0.04,-0.13,0.01,-0.16,0.14,0.16,0.11,0.17,-0.04,0.29,-0.21,-0.13,-0.06,0.18,0.09,0.02,0.09,0.0,-0.04,0.14,0.09,0.17,0.11,0.05,0.1,-0.12,-0.11,0.49,0.12,0.25,-0.01,0.39,0.02,0.26,-0.01,0.15,0.02,0.04,-0.03,0.06,-0.05,-0.17,0.06,0.23,-0.22,0.02,-0.09,0.11,-0.31,-0.3,0.01,-0.06,-0.02,-0.1,-0.03,0.1,-0.14,0.55,-0.06,-0.04,0.12,-0.08,-0.22,-0.23,-0.11,0.22,0.0,0.01,0.44,-0.23,0.3,-0.01,-0.02,0.19,0.31,-0.01,0.2,0.05,-0.05,-0.22,-0.06,-0.14,-0.49,-0.23,-0.14,-0.49,-0.5,-0.7,-0.22,-0.46,-0.04,-0.05,0.04,-0.21,0.0,-0.24,0.28,0.07,-0.15,-0.11,0.07,-0.18,-0.19,0.23,-0.08,0.06,0.03,0.08,-0.49,-0.25,-0.14,-0.05,-0.24,-0.52,-0.4,-0.17,-0.2,-0.29,-0.42,-0.26,-0.67,-0.5,-0.57,-0.5,-0.32,-0.21,-0.01,-0.41,-0.39,0.1,0.19,-0.11,0.29,-0.17,-0.06,-0.06,0.23,0.19,0.19,0.18,0.19,0.28,0.28,0.23,0.01,0.0,0.3,0.18,-0.03,0.13,0.33,0.04,0.28,0.13,0.02,0.08,-0.01,0.65,0.19,0.4,0.2,0.03,0.15,0.37,0.43,0.04,0.78,0.44,0.37,0.35,0.45,0.69,0.36,0.39,0.28,0.58,0.37,0.25,0.45,0.45,0.65,0.38,0.47,0.17,0.31,0.23,0.02,0.06,0.22,0.37,-0.07,0.03,-0.1,0.15,0.23,-0.12,0.26,0.17,0.08,-0.01,0.07,0.09,0.23,0.07,0.04,0.08,0.06,-0.29,0.18,0.1,0.23,-0.02,-0.18,0.25,-0.13,-0.1,-0.04,0.02,0.08,0.06,0.41,0.02,-0.06,0.19,0.03,0.1,-0.22,-0.27,-0.17,-0.16,0.17,0.06,0.05,0.2,-0.1,-0.23,-0.18,-0.13,-0.08,0.15,-0.37,-0.17,-0.29,-0.36,-0.27,-0.12,-0.16,0.06,-0.37,-0.09,-0.31,-0.17,-0.25,-0.17,-0.26,-0.17,-0.24,-0.51,-0.11,-0.06,-0.33,0.02,-0.39,-0.09,-0.16,-0.15,-0.25,0.25,-0.3,0.06,-0.15,0.06,0.09,0.28,0.15,0.17,0.38,0.43,0.46,0.2,-0.08,-0.21,0.11,0.02,-0.05,0.16,0.04,0.06,-0.1,0.25,0.24,0.14,0.05,0.0,0.01,0.36,0.23,0.27,-0.08,0.22,-0.04,-0.06,0.02,0.2,0.32,0.38,0.27,0.08,-0.06,-0.17,-0.16,0.07,0.32,0.0,-0.15,-0.07,-0.03,-0.04,0.08,0.03,-0.06,-0.11,-0.02,-0.03,0.02,0.51,-0.12,-0.18,0.1,-0.28,0.06,0.06,-0.18,0.22,0.41,0.09,-0.13,0.07,-0.11,0.26,-0.29,0.17,0.04,0.51,0.32,0.0,-0.06,-0.23,-0.09,-0.24,0.04,0.01,0.03,0.15,0.33,0.3,0.09,0.31,-0.12,0.17,0.12,-0.26,-0.33,-0.19,0.0,-0.04,-0.19,0.13,-0.16,-0.03,-0.16,-0.15,-0.25,-0.21,0.15,0.06,-0.01,-0.09,0.22,-0.15,-0.08,0.03,0.23,-0.2,-0.24,-0.01,-0.24,-0.2,-0.28,-0.09,-0.29,-0.3,-0.23,-0.15,-0.4,-0.06,-0.32,-0.44,-0.18,-0.09,-0.17,-0.21,-0.45,-0.24,-0.44,0.0,-0.08,-0.21,-0.35,-0.47,0.12,-0.41,-0.08,-0.3,-0.04,-0.1,0.08,-0.14,-0.32,0.18,-0.02,-0.11,0.23,0.36,0.53,0.1,0.44,0.15,0.37,0.45,0.1,0.57,0.25,0.44,0.24,0.09,0.18,0.17,0.51,0.25,0.29,0.36,0.35,0.52,0.23,0.51,0.17,0.29,0.21,0.44,0.66,0.55,0.04,0.35,-0.03,-0.1,-0.18,0.23,0.27,-0.2,0.0,-0.14,-0.06,-0.09,-0.15,-0.1,0.09,0.04,-0.13,-0.06,0.29,-0.09,0.06,0.04,0.13,0.01,-0.22,0.27,0.22,0.16,0.13,-0.04,0.01,-0.09,0.33,0.24,0.06,-0.19,0.09,-0.45,-0.17,0.14,-0.2,0.14,-0.16,0.1,-0.01,-0.1,-0.32,-0.13,0.26,0.21,-0.38,-0.37,-0.15,-0.36,0.1,-0.2,0.32,0.24,0.05,0.25,-0.03,-0.07,-0.13,-0.2,-0.07,-0.28,-0.06,-0.14,-0.24,0.04,0.12,0.23,0.36,0.0,0.19,0.29,-0.02,0.12,0.38,0.41,0.38,0.19,0.08,0.09,-0.03,0.23,-0.07,-0.03,0.03,0.14,0.3,0.1,0.33,0.21,0.33,0.16,0.19,0.37,0.14,0.25,0.1,0.28,0.36,0.11,-0.3,0.04,-0.07,-0.04,-0.12,-0.03,0.06,-0.36,-0.39,-0.16,-0.28,-0.21,-0.19,-0.45,0.04,0.19,0.31,-0.01,-0.06,0.25,0.66,0.46,0.06,0.35,-0.04,-0.12,0.15,-0.16,0.08,-0.17,-0.1,-0.24,-0.18,0.15,-0.06,0.13,-0.26,-0.29,-0.27,-0.26,-0.27,-0.2,0.0,-0.27,0.35,-0.11,0.22,0.23,-0.28,-0.06,-0.03,0.09,-0.21,-0.06,-0.14,0.16,-0.11,0.22,-0.07,0.08,-0.15,-0.12,-0.21,-0.17,-0.12,0.24,0.0,0.06,-0.04,0.07,0.03,-0.2,-0.16,-0.16,-0.1,-0.15,0.04,-0.05,-0.19,-0.03,-0.36,-0.25,0.02,-0.31,0.14,-0.22,-0.04,-0.28,-0.2,-0.27,0.0,-0.09,-0.01,-0.29,-0.27,-0.15,-0.05,-0.33,-0.19,-0.01,-0.14,-0.37,0.08,0.09,-0.06,-0.1,-0.4,-0.36,-0.22,-0.37,-0.23,-0.12,-0.42,-0.3,-0.12,0.25,-0.42,-0.03,-0.11,-0.29,-0.17,-0.04,0.06,-0.13,0.15,-0.15,0.13,-0.1,-0.21,0.14,-0.23,0.04,-0.03,-0.08,0.02,0.32,0.34,-0.1,-0.15,0.03,-0.28,-0.23,0.01,-0.23,-0.15,-0.23,-0.19,0.03,-0.25,0.13,-0.18,-0.34,-0.34,-0.3,0.01,-0.16,-0.21,0.12,-0.2,-0.3,0.04,-0.23,-0.12,-0.4,-0.33,0.1,-0.35,-0.02,-0.34,0.03,0.07,0.14,-0.19,0.08,0.23,0.04,-0.04,-0.03,-0.37,-0.04,-0.32,-0.25,-0.28,-0.03,-0.12,-0.55,-0.52,-0.09,-0.11,-0.37,-0.28,-0.19,0.23,-0.03,-0.26,-0.42,-0.42,-0.51,-0.24,-0.49,-0.16,0.0,-0.41,-0.25,-0.47,-0.19,-0.02,-0.45,-0.3,-0.23,0.06,-0.24,-0.11,-0.29,0.03,-0.32,-0.35,-0.25,0.14,-0.52,-0.3,-0.36,-0.44,-0.42,-0.29,-0.32,-0.21,-0.09,-0.22,-0.12,-0.31,-0.14,-0.28,-0.41,0.25,-0.33,-0.05,-0.32,-0.24,-0.29,-0.13,0.04,-0.1,-0.02,0.19,0.19,-0.37,-0.26,0.2,-0.32,0.04,-0.19,-0.2,-0.04,-0.17,0.04,-0.02,0.0,-0.34,-0.24,0.04,0.22,-0.13,-0.11,-0.1,0.0,-0.1,-0.03,-0.09,-0.27,0.09,-0.2,0.22,-0.12,-0.22,-0.24,-0.41,-0.03,0.27,-0.39,-0.32,-0.13,-0.16,-0.06,-0.21,0.29,-0.12,-0.39,-0.1,-0.12,-0.4,-0.49,-0.4,-0.41,-0.36,-0.54,-0.22,-0.18,-0.39,-0.38,-0.16,-0.34,-0.25,-0.4,0.02,-0.3,-0.19,-0.01,-0.15,0.09,-0.28,-0.02,-0.03,-0.39,-0.03,-0.32,-0.05,0.12,-0.23,-0.24,0.18,-0.03,0.29,0.08,-0.17,-0.05,0.04,0.05,-0.29,-0.26,-0.16,-0.36,-0.3,-0.41,-0.36,-0.48,-0.32,-0.39,-0.39,0.01,-0.17,-0.14,-0.34,-0.27,-0.03,-0.49,-0.47,-0.39,-0.35,-0.28,-0.28,-0.43,-0.28,-0.15,-0.09,-0.49,-0.07,-0.23,0.06,-0.42,-0.27,-0.16,-0.24,-0.19,-0.33,-0.33,-0.53,-0.23,-0.1,-0.44,-0.25,-0.57,-0.23,-0.42,-0.31,-0.36,-0.18,-0.35,-0.51,-0.19,-0.36,-0.48,-0.4,-0.24,-0.43,-0.41,-0.17,-0.1,-0.26,-0.15,-0.04,-0.23,-0.14,-0.35,-0.31,0.02,-0.25,-0.19,0.11,0.04,0.17,-0.45,-0.19,-0.17,-0.49,-0.3,-0.34,-0.31,-0.28,-0.19,0.06,-0.35,-0.12,-0.18,-0.1,0.19,-0.01,-0.12,-0.06,0.52,0.13,0.33,0.37,0.41,0.08,0.17,0.08,0.27,0.04,0.61,0.14,-0.1,0.43,0.33,0.27,0.01,0.25,0.12,0.04,0.08,0.02,0.45,-0.07,0.06,0.23,0.25,0.31,0.25,0.0,0.04,0.11,-0.04,0.1,0.23,0.09,0.29,-0.04,0.13,0.23,0.2,0.03,0.25,0.35,-0.11,0.27,0.36,0.29,0.38,0.3,0.37,0.18,0.23,0.28,0.12,-0.13,-0.31,0.62,-0.55,0.19,-0.26,0.03,-0.14,0.04,-0.28,0.24,-0.08,-0.23,-0.26,0.18,0.06,-0.07,0.04,0.22,-0.09,0.3,0.04,0.69,0.58,0.22,0.04,0.27,0.2,0.22,0.31,0.47,0.46,0.37,0.61,0.41,0.34,0.49,0.94,0.32,0.33,0.52,0.51,0.39,0.07,0.22,0.24,0.43,0.5,0.3,0.3,0.46,0.37,0.28,0.04,0.16,0.06,0.14,0.1,0.25,0.14,0.18,0.21,0.39,0.25,0.29,0.48,0.13,0.08,-0.11,0.0,-0.05,0.17,-0.21,0.01,-0.25,0.04,-0.03,-0.08,-0.28,0.14,-0.05,-0.04,0.13,0.03,-0.21,-0.22,-0.08,-0.34,-0.07,-0.01,-0.17,0.04,-0.12,-0.17,0.22,-0.13,-0.21,0.03,-0.05,-0.08,0.09,0.18,-0.07,0.37,0.3,0.07,-0.07,-0.01,0.14,-0.2,-0.04,-0.09,-0.23,-0.09,0.16,0.12,-0.13,0.22,-0.1,0.02,0.1,-0.1,-0.02,-0.05,-0.01,0.16,0.22,0.06,0.32,0.49,-0.09,-0.08,-0.11,0.01,-0.1,-0.09,0.06,0.23,0.2,-0.15,0.08,-0.09,-0.18,0.03,0.47,-0.02,0.12,0.04,0.05,-0.02,0.02,0.08,0.16,-0.01,0.04,0.18,0.46,0.09,0.22,0.4,0.06,0.44,0.26,0.33,0.26,0.28,0.26,0.12,-0.03,0.1,0.38,0.28,0.01,0.08,0.16,-0.05,0.1,-0.05,0.16,-0.11,-0.11,-0.05,-0.39,-0.01,0.26,-0.02,-0.05,0.14,0.16,0.01,-0.07,-0.18,-0.24,0.04,-0.31,-0.1,-0.45,0.01,-0.04,0.24,0.09,0.3,-0.19,-0.26,0.11,0.13,-0.01,-0.06,-0.15,-0.18,0.05,-0.26,-0.64,-0.29,-0.2,0.02,-0.08,-0.24,-0.07,-0.2,-0.28,-0.24,0.08,-0.39,-0.21,-0.36,-0.32,-0.39,-0.21,-0.22,-0.16,-0.22,0.19,-0.15,-0.42,-0.33,-0.32,-0.08,0.0,-0.3,0.15,-0.09,0.0,-0.09,0.16,-0.13,-0.19,-0.2,0.12,0.19,-0.06,0.2,0.32,0.01,0.26,0.46,-0.1,0.11,0.2,0.06,-0.1,-0.11,0.09,0.01,0.14,0.15,0.37,0.07,-0.14,-0.17,-0.01,-0.25,-0.13,0.17,-0.11,-0.26,0.3,0.36,0.24,0.24,0.29,0.18,0.45,0.17,0.22,0.24,0.6,0.12,-0.06,0.02,0.1,-0.1,-0.15,0.03,-0.12,0.14,0.65,0.12,0.2,0.26,0.26,-0.73,0.17,0.08,0.19,0.14,0.24,0.21,0.64,0.21,0.15,0.34,0.13,0.1,0.36,0.43,0.4,-0.04,0.36,-0.01,-0.01,-0.09,-0.15,0.09,-0.03,-0.04,0.02,-0.12,-0.2,-0.23,0.02,-0.34,-0.25,-0.28,-0.13,-0.23,-0.44,-0.45,-0.37,-0.38,-0.45,-0.22,-0.27,-0.22,-0.33,-0.12,-0.39,0.04,-0.3,-0.01,-0.14,-0.08,0.1,0.06,-0.01,0.02,-0.13,0.1,-0.01,0.09,0.5,-0.01,0.29,0.1,-0.09,0.42,0.07,-0.05,-0.13,0.1,0.09,-0.11,-0.11,-0.25,-0.05,0.01,0.14,-0.15,-0.1,-0.13,0.14,-0.33,-0.14,-0.12,-0.17,-0.3,0.08,0.01,-0.01,0.23,-0.21,0.06,-0.09,0.22,-0.07,-0.05,-0.11,0.05,-0.11,-0.3,-0.18,-0.04,-0.09,-0.13,-0.05,-0.33,-0.41,-0.15,-0.08,-0.3,-0.07,0.09,-0.27,0.31,-0.14,0.17,0.03,0.04,0.4,-0.06,0.15,0.02,0.04,0.44,0.57,0.1,0.28,0.14,0.46,0.28,0.35,0.03,-0.03,0.27,0.14,0.02,-0.01,0.21,-0.03,0.12,0.26,0.33,0.42,0.29,0.24,0.21,0.24,0.02,0.09,0.52,0.12,0.38,-0.02,0.47,0.02,0.32,0.16,0.13,0.15,0.34,0.2,0.39,0.32,0.31,0.1,-0.12,0.57,0.25,0.08,-0.05,0.01,-0.09,0.05,-0.28,0.2,0.13,0.06,-0.08,-0.11,0.11,-0.06,-0.08,-0.24,-0.12,-0.26,-0.15,-0.16,-0.28,-0.22,-0.32,-0.19,-0.37,-0.21,-0.04,0.06,0.19,0.1,-0.22,-0.33,-0.32,-0.04,-0.2,-0.19,-0.12,-0.08,-0.16,-0.32,-0.16,-0.3,-0.33,-0.33,-0.31,-0.22,-0.21,-0.23,0.1,-0.2,-0.29,-0.31,-0.03,-0.1,-0.13,-0.23,-0.65,-0.19,-0.17,-0.32,-0.31,-0.19,-0.29,0.15,-0.03,-0.15,-0.25,-0.16,-0.27,-0.08,-0.15,-0.09,-0.36,0.04,-0.33,-0.05,-0.08,-0.06,-0.07,-0.05,-0.02,0.08,0.02,-0.37,-0.42,0.01,0.01,0.05,0.12,-0.22,-0.23,-0.27,0.0,-0.12,0.07,-0.29,-0.16,-0.08,-0.14,0.0,-0.31,-0.06,0.23,-0.03,-0.25,-0.6,-0.08,-0.31,-0.36,-0.27,-0.63,-0.49,-0.27,-0.35,-0.52,-0.47,-0.58,-0.03,-0.12,-0.41,-0.2,-0.03,-0.19,0.15,-0.06,0.15,-0.15,0.09,-0.04,-0.06,-0.24,-0.01,-0.11,-0.03,-0.04,0.22,0.21,0.24,0.09,0.04,0.36,0.15,0.16,0.02,0.29,-0.06,-0.24,-0.17,-0.12,-0.26,-0.01,-0.2,-0.22,0.08,-0.26,-0.1,-0.48,-0.2,-0.1,-0.27,0.19,-0.2,0.06,0.06,-0.02,-0.25,-0.04,-0.17,-0.3,-0.2,-0.09,0.04,-0.1,0.13,-0.13,-0.2,-0.12,-0.13,-0.21,0.11,0.06,0.23,-0.03,-0.03,-0.03,-0.13,-0.09,0.03,-0.07,-0.52,-0.72,-0.45,-0.67,-0.67,-0.45,-0.54,-0.6,-0.53,0.24,-0.45,-0.27,-0.33,-0.37,-0.28,-0.19,-0.37,0.03,-0.14,-0.15,-0.19,-0.13,0.01,0.02,-0.15,0.28,-0.01,-0.11,-0.05,0.22,-0.03,-0.06,0.06,-0.08,0.32,-0.13,0.04,0.29,0.01,-0.22,-0.03,-0.15,-0.07,-0.03,0.23,0.36,0.36,-0.03,0.1,-0.07,0.23,0.4,-0.14,-0.1,-0.03,0.37,-0.1,0.04,0.17,-0.19,-0.03,0.16,0.41,0.26,0.19,-0.03,0.06,-0.09,-0.1,-0.1,-0.04,0.02,0.17,-0.04,0.15,-0.09,-0.29,-0.05,0.04,-0.07,-0.11,0.07,0.15,0.32,0.32,0.23,0.12,0.16,-0.2,0.08,0.09,-0.07,-0.2,-0.1,-0.27,0.33,-0.23,0.0,-0.07,-0.05,-0.04,-0.16,-0.12,0.06,-0.29,-0.32,-0.12,-0.08,-0.07,-0.01,0.04,-0.32,-0.03,-0.21,-0.22,-0.24,-0.33,-0.4,-0.3,-0.54,-0.22,-0.08,-0.28,-0.32,0.08,-0.33,-0.2,-0.11,-0.22,-0.32,-0.21,-0.07,0.16,-0.07,-0.27,-0.09,-0.04,-0.19,-0.42,-0.14,0.25,-0.02,-0.25,-0.24,-0.01,0.25,-0.13,-0.17,-0.11,-0.25,-0.26,0.25,0.1,-0.11,-0.13,-0.35,-0.04,0.02,-0.24,-0.19,-0.08,0.06,-0.19,-0.07,-0.47,-0.22,-0.19,-0.14,-0.18,0.05,-0.16,-0.07,-0.26,-0.12,0.04,-0.21,-0.12,-0.16,-0.15,-0.45,-0.27,-0.29,-0.01,-0.22,-0.27,-0.17,0.62,0.09,-0.06,0.14,0.05,-0.11,-0.03,-0.01,-0.03,0.11,0.39,0.1,0.19,0.24,0.16,0.11,0.17,-0.05,-0.01,0.46,0.16,0.06,0.09,0.13,-0.19,-0.09,-0.04,-0.06,0.0,-0.4,-0.12,-0.16,-0.03,0.04,-0.14,0.05,-0.05,0.14,-0.01,-0.12,-0.13,-0.08,-0.25,-0.09,-0.28,0.02,-0.22,-0.26,-0.19,-0.03,-0.12,0.0,0.33,0.38,-0.16,-0.07,-0.2,0.14,-0.15,-0.28,-0.29,0.0,-0.29,0.02,-0.02,0.0,0.21,0.08,-0.15,-0.23,-0.21,0.03,-0.17,-0.12,0.16,0.09,-0.22,-0.28,0.12,-0.03,-0.16,-0.07,-0.27,0.07,0.04,-0.23,-0.19,-0.05,-0.11,-0.2,0.03,0.06,-0.21,-0.38,-0.13,-0.23,-0.17,-0.09,-0.38,-0.25,-0.32,-0.42,0.14,-0.16,-0.25,-0.39,-0.2,-0.18,-0.35,-0.12,-0.15,-0.05,-0.27,-0.28,-0.54,-0.22,-0.04,-0.16,0.04,-0.04,-0.31,0.11,-0.2,-0.1,0.03,-0.03,0.12,-0.08,-0.2,0.02,-0.38,-0.11,-0.15,-0.03,-0.23,-0.05,0.23,-0.13,-0.14,0.06,-0.21,0.47,0.07,0.48,-0.15,0.28,0.29,0.07,-0.16,-0.14,0.16,-0.13,0.17,-0.1,-0.11,0.16,-0.27,-0.09,-0.17,-0.13,0.19,0.23,0.24,-0.02,0.06,0.15,0.32,0.13,0.08,-0.11,0.07,0.07,0.14,-0.08,0.05,-0.26,-0.2,0.06,0.03,0.21,0.29,0.3,0.0,0.25,0.15,-0.1,0.22,-0.08,0.24,0.03,0.16,0.25,0.15,0.11,0.05,0.33,-0.17,0.13,-0.05,0.16,0.09,0.47,-0.03,-0.27,0.07,-0.06,-0.07,-0.24,-0.32,0.43,0.19,0.05,0.01,0.21,-0.25,0.2,0.19,-0.2,-0.34,-0.16,-0.14,-0.18,-0.06,-0.27,0.11,-0.11,0.01,-0.29,-0.35,0.04,0.03,-0.12,-0.05,-0.05,-0.15,-0.09,-0.1,0.28,-0.05,-0.19,-0.17,-0.39,-0.17,-0.33,-0.23,0.01,-0.16,-0.29,0.04,-0.1,0.11,-0.07,0.0,0.15,0.15,0.3,-0.13,-0.29,-0.3,0.04,-0.2,0.26,-0.03,-0.38,-0.31,0.2,0.08,-0.31,-0.22,-0.04,-0.14,-0.33,-0.11,-0.1,-0.16,0.01,-0.11,0.07,-0.28,0.05,0.15,0.22,0.0,-0.04,-0.11,0.17,0.09,-0.01,0.25,-0.14,0.22,-0.06,-0.04,-0.18,0.0,0.06,-0.28,-0.21,0.13,-0.05,0.05,-0.12,0.04,0.04,0.09,0.12,-0.06,0.41,0.29,0.37,-0.09,0.11,0.25,0.05,-0.01,0.25,0.04,-0.05,0.01,-0.04,0.3,-0.19,0.02,0.02,0.09,0.22,-0.23,-0.15,-0.26,-0.08,-0.21,-0.13,-0.23,-0.27,-0.41,-0.11,-0.13,-0.13,-0.12,0.04,-0.29,0.02,0.24,-0.03,-0.07,-0.09,-0.09,0.12,-0.3,-0.15,-0.11,0.01,-0.17,0.03,0.09,-0.03,0.2,-0.06,0.0,0.32,0.04,-0.21,0.01,0.01,0.06,-0.06,0.02,-0.02,0.09,0.2,0.1,0.08,0.36,0.1,0.13,0.04,0.42,0.22,0.06,0.19,0.26,0.13,-0.02,0.27,0.04,-0.05,0.13,0.0,0.16,0.04,0.35,-0.07,-0.13,0.11,-0.07,-0.01,-0.25,0.36,-0.12,-0.03,0.33,0.22,0.1,0.17,0.19,0.09,0.33,0.06,0.12,0.22,0.24,0.06,-0.01,0.09,0.15,0.07,0.01,0.22,-0.07,0.08,-0.04,-0.03,0.3,-0.08,0.04,0.1,-0.12,0.51,0.17,0.18,0.01,0.43,0.04,0.26,0.32,0.1,0.02,0.22,0.3,0.27,0.87,0.21,0.16,0.23,0.06,0.43,0.39,0.43,0.48,0.22,0.24,-0.14,0.04,0.0,-0.07,0.04,0.11,-0.08,0.23,0.01,0.06,0.06,0.4,-0.08,0.28,-0.11,0.1,0.1,0.28,0.08,-0.22,0.09,-0.06,-0.26,0.3,0.35,0.19,0.0,0.27,0.09,0.27,0.02,-0.01,0.08,0.21,0.23,0.28,0.29,0.36,0.49,0.34,0.28,0.34,0.3,0.42,0.28,0.14,0.03,0.21,0.2,-0.02,0.07,0.13,0.13,-0.03,0.41,0.28,-0.09,0.02,0.02,-0.01,0.08,0.13,0.26,0.18,0.04,0.18,-0.12,0.15,-0.03,0.06,-0.03,0.05,0.17,0.1,-0.02,0.04,-0.08,0.28,0.14,0.48,0.08,0.02,0.21,0.29,0.06,0.45,0.24,0.09,0.32,0.28,0.46,0.28,0.12,0.27,0.08,0.13,0.02,-0.03,0.33,0.14,0.4,0.26,0.42,0.6,0.59,0.44,0.63,0.61,0.51,0.77,0.43,0.26,0.34,0.5,0.52,0.51,0.4,0.36,0.68,0.39,0.47,0.35,0.36,0.34,0.23,0.35,0.19,-0.03,0.23,0.16,0.08,0.23,0.3,-0.03,-0.06,-0.22,-0.05,0.15,-0.07,-0.06,-0.09,-0.19,-0.21,-0.11,-0.1,-0.14,0.05,-0.21,-0.33,-0.03,-0.36,-0.36,-0.05,-0.2,-0.44,-0.63,0.0,0.03,-0.29,-0.12,-0.3,-0.03,-0.32,-0.13,0.13,-0.28,-0.14,-0.36,-0.21,-0.1,0.44,-0.28,-0.09,-0.08,-0.01,-0.1,-0.27,-0.07,0.02,-0.02,-0.03,0.04,-0.12,-0.23,-0.2,-0.19,0.27,-0.04,0.12,-0.24,0.04,-0.27,-0.17,-0.25,0.09,-0.08,-0.1,-0.09,-0.01,-0.25,-0.04,0.02,0.47,-0.21,-0.36,-0.29,-0.13,-0.05,-0.09,-0.01,0.02,-0.06,0.13,0.04,-0.06,-0.02,0.04,0.07,0.14,0.1,0.04,0.09,0.23,-0.13,-0.04,0.01,-0.1,-0.18,-0.09,-0.08,-0.14,-0.15,0.27,0.21,0.22,0.39,0.2,0.3,0.48,0.19,0.18,0.42,0.31,0.34,0.12,0.12,0.1,0.2,0.03,0.21,0.54,-0.02,0.06,0.15,0.17,0.03,0.39,0.22,0.33,0.28,0.15,0.17,0.03,0.14,0.13,0.39,-0.05,0.66,0.3,0.19,0.34,-0.03,0.45,0.18,0.3,0.44,0.37,0.34,0.54,0.24,0.59,0.29,0.41,0.79,0.45,0.4,0.33,0.35,0.2,0.56,0.1,0.26,0.32,0.09,0.01,-0.01,-0.06,-0.03,0.04,-0.09,0.01,-0.21,-0.15,0.12,-0.22,-0.11,0.02,-0.25,-0.37,-0.14,-0.16,-0.22,-0.2,-0.01,-0.07,-0.1,0.19,0.02,0.05,-0.01,0.01,0.03,-0.06,-0.07,-0.16,-0.22,-0.16,-0.01,0.03,0.24,0.16,-0.28,-0.06,0.04,-0.04,-0.04,-0.1,0.01,-0.04,-0.33,0.05,-0.33,0.03,-0.24,0.04,-0.32,-0.06,-0.39,-0.1,0.05,-0.27,-0.17,-0.1,-0.35,-0.33,-0.35,-0.1,-0.34,-0.22,-0.09,-0.01,0.54,-0.15,-0.21,0.34,-0.06,-0.29,0.29,-0.07,-0.1,0.01,0.29,0.11,0.1,0.15,0.07,-0.07,0.52,0.16,0.22,-0.01,0.12,0.24,0.06,0.03,0.02,0.22,0.1,0.16,0.39,0.2,0.24,0.36,0.06,0.1,0.06,0.18,0.04,0.2,0.1,0.01,0.51,0.11,0.31,0.14,0.23,0.36,0.08,0.0,-0.16,0.18,0.23,0.09,0.06,-0.09,0.09,0.26,-0.15,0.13,-0.07,0.09,0.01,-0.21,-0.11,-0.09,-0.16,-0.12,0.17,-0.21,-0.08,-0.05,-0.11,0.06,-0.23,-0.29,0.09,-0.13,-0.46,0.16,-0.12,-0.26,0.02,-0.04,-0.25,-0.38,0.07,0.02,0.01,0.03,0.03,-0.05,-0.16,0.01,0.27,-0.09,0.14,-0.11,0.16,0.18,0.15,-0.15,0.12,0.05,-0.12,-0.13,-0.13,-0.09,-0.19,-0.35,0.21,-0.1,0.21,0.13,-0.01,0.06,-0.04,0.02,0.1,0.22,0.22,0.1,-0.01,-0.07,0.15,-0.12,0.35,-0.21,0.03,0.23,-0.11,-0.06,0.06,-0.13,-0.1,-0.12,-0.05,0.06,0.24,0.32,-0.1,0.0,0.13,0.03,0.41,0.25,-0.01,0.06,0.04,0.07,-0.12,0.01,0.15,0.22,0.01,0.21,-0.23,0.19,0.01,0.17,-0.25,-0.11,0.05,0.08,0.04,0.06,0.37,0.05,0.04,0.13,0.05,0.04,-0.03,0.12,-0.22,-0.36,0.16,-0.26,-0.2,0.22,-0.03,-0.18,-0.12,-0.16,-0.18,-0.1,-0.28,-0.01,-0.38,-0.08,-0.03,0.35,-0.07,-0.22,0.02,-0.31,-0.07,-0.15,-0.04,0.21,-0.17,-0.12,-0.23,-0.2,0.16,0.06,0.05,0.09,-0.22,0.12,-0.07,-0.09,0.15,-0.15,0.05,-0.13,0.11,0.0,0.03,0.17,-0.01,-0.06,0.18,-0.03,-0.04,-0.1,0.1,0.04,0.41,0.37,0.17,0.08,0.23,0.08,0.2,0.01,0.28,0.09,-0.13,0.06,-0.2,-0.18,-0.22,-0.21,0.09,-0.05,0.02,0.08,0.27,-0.12,0.11,-0.16,-0.06,-0.08,-0.01,0.34,0.06,0.4,-0.18,0.06,-0.1,-0.03,-0.1,-0.1,-0.14,-0.2,0.2,-0.12,0.2,0.15,0.18,0.06,-0.02,0.01,-0.05,0.13,-0.01,-0.07,-0.09,0.1,0.1,0.44,0.12,-0.11,0.11,0.04,0.15,0.06,-0.01,0.47,0.16,0.24,0.12,0.05,0.12,0.29,0.02,0.09,0.34,0.17,0.34,0.2,0.39,0.27,-0.03,0.41,0.19,0.4,0.24,0.38,0.19,0.08,0.56,0.3,0.06,0.34,0.7,0.45,0.55,0.5,0.41,0.27,0.49,0.51,0.37,0.34,0.16,0.14,0.01,0.07,-0.36,-0.01,-0.44,-0.41,-0.02,-0.17,-0.01,-0.38,-0.26,-0.22,0.02,-0.05,-0.11,0.11,-0.16,0.06,0.44,-0.21,0.2,0.06,-0.07,-0.26,-0.12,0.15,0.01,0.5,0.16,0.06,0.18,0.48,-0.08,0.08,-0.26,0.11,0.35,-0.05,0.41,0.08,0.38,0.02,0.1,0.21,0.25,-0.13,0.09,0.23,-0.18,-0.17,0.09,0.18,0.01,0.17,0.22,-0.2,-0.09,-0.07,-0.28,0.0,-0.2,0.19,0.12,-0.18,-0.03,0.14,0.21,0.02,-0.2,0.05,0.03,0.26,-0.19,-0.08,0.12,0.22,0.05,-0.38,-0.16,-0.24,-0.03,-0.07,-0.19,-0.59,-0.53,-0.63,-0.07,-0.41,-0.23,-0.2,-0.2,-0.22,-0.29,-0.31,-0.27,-0.3,-0.17,-0.34,-0.25,-0.17,-0.27,-0.27,-0.01,0.32,-0.1,0.22,0.11,-0.05,-0.16,-0.24,-0.13,-0.19,-0.32,-0.12,-0.11,0.46,-0.36,-0.26,0.01,-0.21,-0.49,-0.41,-0.16,-0.17,-0.11,-0.08,0.04,-0.24,-0.08,-0.32,-0.26,-0.4,-0.06,-0.17,-0.26,-0.14,-0.04,-0.38,-0.01,0.05,-0.17,-0.3,-0.06,-0.09,0.36,-0.04,-0.13,0.16,-0.13,-0.13,0.17,-0.17,0.14,0.11,0.28,0.06,0.29,0.04,-0.07,-0.32,0.0,-0.09,0.05,-0.08,-0.1,-0.1,-0.24,-0.02,-0.15,-0.3,0.0,-0.23,-0.46,-0.5,-0.07,-0.39,-0.21,-0.51,-0.42,-0.26,-0.37,-0.49,-0.3,-0.3,-0.23,-0.69,-0.23,-0.2,-0.33,-0.42,-0.37,0.1,-0.2,-0.25,-0.29,0.0,-0.24,-0.27,-0.28,-0.43,-0.39,-0.28,-0.22,-0.4,-0.53,-0.36,-0.38,-0.12,-0.03,-0.06,-0.18,-0.14,-0.2,0.06,0.12,0.15,0.0,-0.11,-0.04,-0.15,-0.09,-0.14,0.12,-0.12,0.26,-0.09,-0.17,0.34,-0.01,-0.05,-0.03,-0.02,0.11,-0.05,-0.01,0.06,-0.17,-0.12,-0.08,0.06,0.09,-0.3,-0.11,-0.15,-0.11,-0.24,-0.1,0.02,0.02,-0.31,-0.29,-0.1,-0.39,-0.38,-0.4,-0.21,-0.17,-0.22,-0.28,-0.29,0.1,-0.21,-0.39,0.1,-0.3,-0.09,-0.23,-0.41,0.19,0.21,-0.02,-0.08,-0.05,-0.04,-0.36,-0.24,-0.16,0.09,-0.08,-0.15,0.07,-0.05,0.19,-0.02,-0.11,-0.11,0.33,0.25,-0.09,0.32,0.11,-0.16,-0.17,-0.04,-0.11,0.09,-0.38,-0.32,-0.16,-0.42,0.03,-0.19,-0.24,-0.19,-0.31,-0.37,-0.25,-0.18,0.02,-0.03,-0.07,-0.08,-0.13,-0.36,0.03,-0.06,0.18,0.25,-0.02,-0.39,0.07,0.1,-0.08,0.01,-0.34,0.22,-0.26,-0.13,-0.25,0.07,0.04,-0.15,-0.12,-0.21,0.01,-0.18,0.06,0.06,0.0,-0.06,0.33,-0.25,0.05,-0.1,0.22,0.17,0.11,0.26,0.15,-0.14,0.13,0.41,0.14,-0.09,0.04,-0.23,-0.19,-0.25,0.44,-0.07,-0.01,-0.1,0.13,0.38,-0.09,0.14,0.57,-0.17,-0.07,-0.06,0.15,-0.08,0.08,0.04,0.04,0.17,-0.18,0.06,0.04,-0.01,0.12,0.22,0.17,0.35,0.35,0.06,0.32,0.32,0.09,0.05,0.22,0.15,-0.1,0.0,-0.18,0.08,-0.24,-0.21,-0.25,0.18,-0.55,-0.3,-0.05,-0.08,-0.26,-0.06,-0.14,0.09,0.0,-0.01,0.19,-0.14,-0.16,-0.32,-0.16,-0.23,-0.21,-0.27,-0.15,-0.23,-0.11,-0.12,0.05,-0.2,-0.04,0.09,0.12,-0.02,-0.09,-0.29,-0.11,0.14,0.1,0.34,0.22,0.04,-0.01,0.13,0.06,0.2,0.2,0.19,-0.11,0.21,0.14,-0.05,-0.19,0.06,0.03,0.34,0.78,-0.12,0.04,0.09,-0.15,0.16,-0.18,0.18,0.04,0.0,0.16,-0.34,0.25,-0.03,-0.09,-0.11,0.26,0.03,-0.03,0.09,0.29,0.19,0.21,0.1,-0.05,0.0,-0.06,0.0,-0.07,-0.25,-0.35,-0.29,-0.08,0.0,-0.36,-0.17,-0.16,-0.24,-0.28,-0.08,-0.51,-0.46,-0.11,-0.25,-0.32,-0.33,-0.33,-0.46,-0.3,-0.3,-0.24,0.04,-0.42,-0.45,-0.25,-0.35,-0.7,-0.59,-0.47,-0.39,-0.32,-0.35,-0.29,-0.25,-0.19,-0.22,0.04,-0.05,0.28,0.11,0.03,-0.11,0.07,-0.24,-0.2,-0.31,-0.28,-0.18,0.26,-0.34,-0.45,-0.14,-0.23,-0.08,-0.04,-0.39,-0.39,0.08,-0.09,-0.17,0.12,-0.35,-0.15,-0.05,-0.13,-0.12,0.25,0.01,0.21,0.0,0.0,0.08,-0.09,0.09,0.07,-0.1,0.1,-0.29,0.06,-0.01,0.03,0.08,0.29,0.01,-0.05,0.49,-0.1,0.32,0.04,0.01,0.04,-0.08,-0.14,0.16,-0.08,0.1,-0.17,0.21,-0.03,0.28,0.24,-0.03,-0.1,-0.1,-0.13,0.25,-0.08,-0.13,-0.1,-0.38,-0.21,0.0,-0.1,-0.16,0.02,-0.26,0.13,-0.01,-0.04,-0.01,-0.1,-0.26,-0.36,0.19,-0.32,-0.21,-0.25,0.01,0.15,-0.22,0.06,-0.32,-0.19,-0.26,-0.21,0.18,-0.08,-0.11,0.03,0.09,-0.11,0.01,-0.17,-0.03,-0.31,0.1,-0.21,0.28,-0.17,0.25,-0.19,0.12,0.22,0.04,0.06,-0.06,0.08,0.12,0.17,0.16,0.26,0.27,0.1,0.06,-0.06,0.08,0.0,-0.09,-0.17,0.16,0.28,-0.03,0.32,-0.11,0.28,0.28,0.0,0.26,0.21,-0.02,0.52,0.11,0.07,0.19,0.15,-0.17,0.1,0.04,0.42,-0.15,0.02,0.17,0.48,-0.08,0.29,-0.02,0.21,-0.06,-0.08,0.02,-0.09,0.06,0.24,-0.27,0.09,-0.03,-0.22,0.08,-0.08,-0.2,-0.26,-0.27,-0.27,0.1,-0.25,-0.29,-0.01,-0.29,0.26,-0.11,0.1,0.15,-0.2,0.16,-0.16,-0.17,0.11,-0.32,-0.3,-0.07,-0.2,-0.11,-0.31,0.18,0.17,-0.24,-0.11,0.36,-0.03,0.02,-0.22,0.13,-0.08,-0.24,-0.03,0.14,0.01,0.06,0.02,-0.09,-0.06,0.13,0.15,0.01,-0.11,0.34,0.58,0.2,0.22,0.64,0.2,0.61,0.28,0.18,0.17,0.42,0.44,0.35,0.27,0.02,0.01,-0.03,0.03,-0.01,0.13,0.21,0.26,0.35,-0.06,0.34,0.31,-0.01,0.0,0.4,-0.16,-0.3,0.24,-0.21,-0.01,-0.14,-0.14,-0.31,-0.37,0.34,-0.08,-0.08,0.35,0.26,0.24,0.16,0.29,0.02,0.05,-0.12,0.06,0.38,0.23,0.23,0.02,0.1,0.43,-0.09,-0.08,-0.18,-0.13,-0.16,-0.1,-0.09,0.13,-0.15,-0.1,-0.33,-0.21,0.06,0.06,0.17,0.01,-0.13,0.12,0.03,0.02,-0.05,0.24,-0.05,0.11,0.2,-0.03,0.19,0.27,-0.08,-0.04,-0.1,-0.08,0.19,-0.12,0.11,0.14,0.34,-0.03,-0.15,-0.1,0.2,-0.14,0.02,-0.03,0.54,0.21,0.13,0.3,0.15,0.0,0.09,0.62,0.44,0.12,0.08,0.04,0.12,0.04,0.38,-0.01,0.1,0.06,0.13,0.04,0.01,0.08,0.1,0.26,-0.07,-0.26,0.03,-0.12,-0.12,-0.07,-0.13,-0.21,0.04,-0.19,0.12,-0.03,-0.15,-0.18,-0.01,-0.17,0.06,0.07,0.17,0.06,0.15,0.1,0.22,0.31,0.1,-0.02,0.02,-0.02,0.12,0.18,0.2,0.07,0.53,0.39,0.43,0.16,-0.12,0.13,0.32,0.33,0.15,-0.07,0.08,0.06,0.36,0.3,0.24,0.2,0.18,0.16,0.27,0.7,0.02,0.17,-0.43,0.15,0.26,0.43,0.04,-0.09,-0.1,0.19,0.25,0.05,0.49,0.08,0.4,0.08,0.15,0.04,0.56,0.0,0.24,0.14,-0.05,0.13,-0.04,0.17,0.59,0.02,-0.01,0.09,-0.15,-0.13,0.13,0.02,-0.14,-0.11,-0.24,-0.11,-0.25,-0.33,0.09,-0.14,-0.18,0.1,-0.08,0.17,0.0,-0.07,-0.05,0.06,-0.24,-0.07,-0.06,0.16,0.01,0.12,0.08,0.38,-0.17,0.24,0.07,0.07,-0.13,-0.05,-0.15,0.15,-0.32,0.3,-0.08,-0.02,0.16,-0.03,0.07,0.0,-0.03,-0.11,0.17,0.17,-0.03,-0.09,-0.12,0.05,-0.12,-0.05,-0.21,-0.15,-0.11,0.21,-0.03,-0.01,-0.15,-0.12,0.32,0.17,0.05,0.29,0.14,0.08,-0.04,0.04,0.1,0.06,0.3,0.11,0.11,0.04,0.36,0.03,0.08,0.04,0.14,0.12,0.24,0.1,0.09,0.23,0.34,0.38,0.06,0.37,0.38,0.4,0.36,0.38,0.5,0.77,0.37,0.46,0.3,0.15,0.55,0.64,0.73,0.11,0.24,0.18,0.2,0.1,0.31,0.19,-0.14,-0.05,0.16,0.18,0.43,0.52,0.0,0.12,-0.07,-0.12,-0.33,0.02,0.23,-0.2,0.16,-0.03,-0.2,0.09,0.02,0.19,-0.06,0.2,0.18,0.01,-0.05,0.04,-0.31,-0.17,-0.34,-0.42,-0.07,-0.14,-0.38,-0.19,-0.53,-0.49,-0.42,-0.78,-0.39,-0.52,-0.37,-0.26,-0.19,-0.23,-0.21,-0.36,-0.21,-0.1,-0.47,-0.27,-0.29,-0.29,-0.12,-0.19,-0.43,-0.13,-0.43,0.25,0.06,-0.39,-0.19,-0.13,-0.33,-0.2,-0.13,-0.15,-0.03,-0.27,-0.27,-0.53,-0.22,-0.39,0.18,-0.45,-0.25,-0.42,-0.25,-0.13,-0.22,-0.17,-0.38,-0.12,-0.23,-0.33,-0.33,-0.19,-0.33,-0.5,-0.43,-0.04,0.04,-0.18,-0.29,-0.13,0.17,-0.26,-0.26,0.1,0.04,0.1,0.3,0.58,0.32,0.1,0.23,0.04,-0.07,0.21,0.12,0.21,-0.03,0.1,-0.14,-0.12,0.07,-0.16,-0.11,0.17,-0.33,0.1,-0.37,-0.06,-0.04,-0.13,-0.25,-0.03,-0.05,-0.19,-0.14,-0.41,0.12,0.18,-0.28,-0.11,-0.05,0.01,0.33,0.16,-0.01,-0.04,-0.43,-0.41,0.11,-0.25,-0.01,-0.11,-0.28,-0.1,-0.11,-0.27,-0.11,-0.19,0.25,0.07,0.13,-0.17,-0.05,-0.04,0.08,0.28,-0.09,-0.43,-0.25,-0.13,-0.41,-0.54,-0.51,-0.16,-0.55,-0.69,-0.5,-0.5,-0.39,-0.47,-0.47,-0.28,-0.23,0.1,-0.25,-0.16,-0.5,-0.48,-0.6,-0.58,-0.37,-0.07,-0.33,-0.61,-0.31,-0.63,-0.64,-0.72,-0.64,-0.57,-0.73,-0.37,-0.68,-0.97,-0.44,-0.55,-0.47,0.06,0.18,0.02,-0.06,-0.29,0.2,0.1,0.23,-0.31,-0.09,0.05,-0.38,-0.2,0.02,-0.24,-0.43,-0.17,-0.4,-0.45,-0.31,-0.09,-0.25,-0.13,-0.3,0.1,-0.24,-0.04,-0.2,-0.03,-0.02,0.19,0.06,-0.03,0.1,-0.23,-0.08,-0.17,0.02,-0.27,-0.26,-0.13,0.06,0.16,0.08,-0.12,-0.01,0.29,0.36,-0.18,-0.17,-0.23,-0.05,-0.29,-0.07,0.02,-0.25,0.1,-0.11,0.08,0.4,0.37,0.09,0.23,0.32,-0.01,0.13,0.28,0.1,0.15,0.12,0.07,0.24,0.19,0.06,0.12,0.4,0.12,0.04,0.45,0.21,0.18,0.15,0.22,0.0,0.21,0.14,0.4,0.58,0.54,0.2,0.54,0.27,0.47,0.21,0.56,0.43,0.31,0.42,0.22,0.08,0.19,0.14,0.02,0.22,0.07,0.31,0.1,0.24,0.38,0.04,0.4,0.24,0.18,0.48,0.31,0.3,0.09,0.06,0.03,0.37,0.06,0.33,0.37,0.05,0.32,0.0,-0.01,-0.11,-0.05,-0.15,0.08,-0.37,0.13,-0.15,-0.15,-0.11,-0.15,0.0,-0.36,-0.26,-0.09,-0.29,0.28,0.21,-0.02,0.13,-0.13,-0.21,-0.11,-0.31,0.08,-0.1,0.2,-0.15,-0.07,0.09,0.06,-0.27,-0.16,-0.04,-0.11,0.24,0.08,0.18,0.31,0.01,-0.01,-0.11,0.22,0.24,0.04,-0.11,0.17,0.18,-0.03,-0.12,-0.1,0.16,-0.01,0.47,0.04,0.13,0.35,0.41,0.16,0.66,0.32,0.36,0.18,0.53,0.48,0.41,0.54,0.15,0.23,0.15,0.44,0.23,0.22,-0.07,0.21,0.42,0.24,0.16,0.06,0.15,0.2,0.13,0.01,0.24,0.29,0.2,-0.21,-0.17,-0.05,-0.24,-0.07,-0.19,-0.02,-0.24,0.1,-0.29,-0.15,-0.09,0.14,0.12,0.32,0.2,0.17,-0.01,0.08,0.09,-0.04,-0.05,0.12,0.18,-0.15,0.04,0.08,-0.19,0.03,0.14,-0.04,0.18,0.3,0.44,-0.22,0.2,-0.03,-0.08,0.39,0.14,-0.06,0.2,0.03,-0.03,-0.07,0.13,0.37,-0.1,0.21,0.21,-0.09,-0.1,0.18,-0.26,-0.25,0.1,-0.26,0.03,-0.18,-0.03,-0.18,-0.24,-0.36,-0.37,-0.32,-0.09,-0.26,-0.28,-0.34,-0.2,-0.17,-0.38,0.13,-0.07,-0.12,-0.03,-0.18,-0.34,0.18,-0.04,0.06,0.03,-0.03,-0.12,0.21,0.14,0.46,-0.19,0.56,0.23,0.23,0.56,0.01,0.06,0.06,0.24,-0.02,-0.07,-0.1,0.05,-0.13,0.04,0.29,0.23,-0.03,-0.09,0.09,0.03,0.04,-0.01,0.13,-0.07,-0.02,-0.12,-0.17,-0.31,-0.22,-0.33,-0.06,-0.34,0.1,-0.18,-0.23,-0.08,-0.03,0.0,-0.25,-0.26,-0.19,-0.23,-0.26,-0.23,0.01,-0.18,-0.14,-0.14,-0.32,0.12,0.03,-0.02,-0.29,-0.23,0.0,0.01,-0.2,0.22,0.1,0.01,-0.09,0.0,-0.33,-0.14,-0.03,0.04,0.03,0.17,0.14,-0.08,0.16,0.25,0.23,0.08,0.23,0.24,-0.03,0.37,0.29,0.18,0.2,0.46,0.13,0.34,0.19,0.2,0.12,0.17,0.36,0.34,0.09,0.28,0.66,0.62,0.32,0.53,0.28,0.17,0.24,0.49,0.07,0.34,0.35,-0.04,-0.06,0.2,-0.01,0.0,-0.21,-0.11,-0.14,-0.1,-0.17,-0.26,-0.03,-0.03,-0.1,-0.14,-0.44,-0.03,-0.45,-0.4,-0.06,-0.18,0.04,-0.34,-0.24,-0.27,0.26,-0.34,-0.25,-0.08,-0.05,-0.03,-0.07,-0.16,0.0,-0.48,-0.35,-0.35,-0.09,-0.28,-0.31,0.04,0.01,-0.06,-0.13,-0.07,0.08,-0.31,-0.34,-0.22,-0.23,-0.36,0.06,-0.2,-0.24,-0.03,-0.19,-0.07,-0.22,0.22,0.29,-0.23,-0.15,-0.14,0.07,-0.28,-0.13,-0.25,0.09,-0.17,-0.03,-0.36,0.03,-0.14,-0.13,0.1,0.01,0.02,0.13,0.05,-0.15,-0.09,-0.29,-0.24,0.13,-0.12,-0.15,0.02,-0.37,-0.07,-0.43,-0.45,-0.42,-0.35,-0.13,-0.18,0.17,-0.03,-0.34,-0.15,-0.17,-0.03,0.18,-0.41,-0.17,-0.29,-0.41,-0.28,-0.03,-0.31,-0.21,-0.44,-0.14,-0.23,-0.26,0.1,-0.15,-0.19,-0.02,-0.13,-0.08,0.32,0.11,0.12,0.13,0.13,0.24,0.17,-0.03,0.26,-0.02,0.06,-0.42,-0.1,-0.2,0.13,-0.06,0.1,0.03,-0.01,0.03,-0.06,-0.27,-0.06,0.06,-0.1,-0.05,0.13,-0.35,0.41,0.03,0.01,0.09,0.17,-0.5,0.06,-0.12,-0.3,-0.03,-0.22,-0.24,-0.2,-0.16,-0.22,-0.06,-0.13,-0.11,0.05,0.02,0.2,0.19,0.28,0.05,-0.14,-0.04,-0.1,-0.27,-0.02,-0.01,0.02,-0.44,0.08,0.19,-0.24,0.12,-0.01,-0.12,-0.06,-0.05,0.09,-0.14,-0.24,-0.22,-0.13,-0.07,-0.07,-0.2,-0.19,-0.2,0.06,-0.18,0.04,-0.3,0.12,-0.14,-0.19,-0.3,-0.18,-0.18,-0.04,-0.04,0.39,-0.29,-0.09,0.01,-0.25,-0.18,0.0,0.01,-0.2,-0.11,-0.23,-0.08,-0.26,-0.01,-0.02,0.05,-0.13,-0.18,-0.19,0.26,-0.12,-0.2,-0.03,-0.03,0.04,-0.15,0.0,-0.13,-0.2,-0.09,-0.06,-0.29,-0.12,-0.22,-0.1,0.19,-0.17,-0.1,-0.29,-0.35,-0.1,-0.02,-0.1,-0.29,-0.31,-0.35,0.23,-0.22,-0.03,0.11,-0.17,0.01,0.15,-0.14,-0.14,-0.19,0.07,-0.3,0.01,0.28,-0.03,-0.15,0.19,-0.03,-0.2,0.27,-0.23,-0.29,-0.15,0.02,-0.03,0.04,0.18,0.27,0.26,0.36,0.31,0.32,0.17,0.39,0.12,-0.21,-0.12,0.3,0.09,-0.03,0.12,0.25,0.14,0.29,0.13,0.35,0.15,0.08,0.16,-0.07,-0.02,-0.16,0.3,-0.28,0.06,-0.26,0.17,-0.09,-0.19,-0.1,-0.31,0.13,-0.04,-0.25,-0.31,0.03,0.26,-0.04,-0.03,0.02,0.21,-0.04,0.1,0.08,0.57,0.35,0.21,0.25,-0.09,0.12,0.28,0.17,0.03,0.06,-0.19,-0.06,0.1,0.01,0.1,-0.08,-0.22,0.04,-0.18,-0.04,-0.14,0.22,0.02,0.2,-0.1,0.22,0.19,0.01,0.34,0.19,-0.03,-0.11,0.3,0.3,0.13,0.01,0.01,0.06,0.18,0.06,0.02,-0.23,0.0,-0.03,-0.08,0.1,0.2,0.14,0.22,0.27,0.09,0.29,0.2,0.21,-0.2,0.3,-0.03,0.03,0.31,0.05,-0.09,-0.02,0.25,0.08,-0.03,0.06,0.23,-0.22,-0.18,0.03,0.18,0.28,-0.09,-0.09,-0.16,-0.17,0.26,-0.04,-0.24,-0.19,0.23,-0.12,0.05,-0.02,0.17,0.28,-0.04,0.07,0.21,0.1,0.13,0.0,-0.23,-0.11,0.03,-0.06,0.36,0.15,0.37,0.19,0.54,0.23,0.26,0.23,0.27,0.46,0.6,0.38,0.35,0.24,0.14,0.36,0.27,0.05,0.19,0.12,0.0,-0.23,0.17,-0.15,0.21,0.17,-0.12,0.01,-0.04,-0.09,0.41,-0.12,-0.04,-0.17,0.06,-0.07,-0.03,-0.08,-0.03,0.1,-0.09,-0.01,-0.11,-0.1,-0.14,0.1,0.2,0.09,-0.14,-0.08,-0.27,0.17,0.01,-0.02,0.08,-0.31,-0.19,-0.22,-0.11,-0.09,-0.05,0.0,0.1,0.09,-0.13,0.29,0.11,0.29,-0.01,0.12,0.34,0.34,0.12,0.19,0.06,0.44,0.13,0.15,0.26,0.27,0.25,0.24,0.6,0.1,0.09,-0.06,0.18,0.23,0.09,-0.09,-0.05,0.22,-0.05,0.06,-0.32,-0.17,0.12,-0.04,-0.19,0.06,-0.02,0.02,-0.27,-0.31,-0.39,0.14,-0.03,-0.2,-0.04,-0.08,-0.22,0.17,0.25,0.25,-0.06,0.18,0.2,-0.13,0.31,0.45,0.08,0.21,0.16,0.17,0.08,0.25,0.15,0.57,0.14,0.21,0.27,0.11,-0.05,0.55,0.04,0.75,0.5,0.23,0.43,0.23,0.43,0.24,0.28,0.06,0.31,0.27,0.23,0.56,0.29,0.2,0.18,0.42,0.16,0.25,0.29,0.48,0.36,0.43,0.19,0.4,-0.01,0.26,0.26,0.04,0.12,0.14,-0.11,0.08,0.01,0.1,0.24,0.07,-0.01,0.3,0.13,0.41,0.29,0.37,-0.18,-0.07,0.03,-0.02,0.12,0.11,-0.01,0.07,0.31,0.34,0.52,0.02,0.21,0.03,0.02,0.14,0.04,0.11,0.24,-0.03,0.0,0.1,0.19,0.1,0.28,0.17,0.12,-0.33,-0.07,-0.32,-0.27,-0.26,-0.56,-0.18,-0.14,-0.11,-0.09,-0.27,0.04,-0.31,-0.43,-0.3,-0.28,-0.5,-0.62,-0.61,-0.29,-0.29,-0.37,-0.62,-0.32,-0.25,-0.23,-0.11,0.24,0.13,0.16,-0.41,-0.21,-0.4,-0.04,-0.11,-0.06,-0.36,-0.33,-0.03,-0.1,-0.35,0.02,-0.22,-0.11,-0.2,-0.38,-0.34,-0.31,-0.36,-0.29,0.0,-0.23,-0.24,0.1,-0.15,-0.21,-0.25,0.12,0.06,-0.04,-0.03,-0.15,-0.4,-0.13,-0.05,0.0,-0.27,-0.1,-0.12,-0.05,-0.39,-0.19,-0.15,-0.24,-0.04,-0.04,0.2,0.21,0.18,0.34,-0.03,0.01,0.01,-0.09,-0.01,0.08,0.04,0.07,0.06,-0.02,0.29,-0.05,0.18,0.14,0.49,0.23,0.12,0.01,0.16,0.38,0.14,0.36,0.09,0.04,-0.1,-0.01,-0.36,0.08,-0.1,-0.1,-0.06,0.08,-0.14,-0.14,0.18,0.1,0.03,0.08,-0.05,-0.25,-0.07,-0.05,-0.29,-0.07,-0.09,-0.2,-0.06,-0.18,-0.07,0.0,-0.12,-0.29,-0.06,0.09,-0.32,-0.29,-0.19,-0.05,-0.32,0.29,-0.32,0.02,0.03,-0.21,-0.32,-0.23,-0.17,0.04,0.02,0.02,0.01,-0.17,-0.17,-0.14,-0.32,-0.41,-0.13,-0.21,-0.37,-0.18,-0.09,0.04,-0.29,-0.18,-0.13,-0.07,-0.18,-0.43,0.03,-0.02,-0.14,-0.21,-0.36,-0.05,-0.03,-0.07,-0.2,-0.42,-0.29,-0.35,-0.22,-0.25,-0.33,-0.26,-0.35,-0.41,-0.14,-0.35,-0.12,-0.35,-0.49,-0.34,-0.27,-0.06,-0.08,-0.22,-0.42,-0.16,0.09,-0.25,-0.03,0.01,-0.17,-0.04,-0.36,0.04,-0.05,0.1,-0.15,-0.11,-0.12,0.15,-0.13,-0.24,0.32,-0.08,0.23,0.1,0.16,0.1,-0.08,0.13,0.04,0.04,0.08,0.08,-0.1,0.34,0.12,-0.11,0.03,0.27,0.13,-0.03,0.23,0.08,-0.02,-0.08,0.25,0.41,0.09,0.32,0.24,0.29,0.42,0.76,0.03,0.45,0.65,0.64,0.47,0.24,0.4,0.19,0.44,-0.04,0.19,0.14,0.03,0.06,0.04,0.03,-0.1,0.15,0.08,-0.26,-0.07,-0.07,0.16,0.42,-0.23,0.17,-0.21,-0.02,0.0,0.06,-0.13,-0.06,-0.14,-0.15,-0.07,0.17,-0.1,-0.28,0.39,0.03,0.19,0.09,0.33,0.48,0.21,0.04,-0.09,0.0,-0.29,-0.11,0.09,-0.24,-0.24,0.06,-0.23,0.3,-0.34,-0.27,-0.07,0.1,0.27,0.16,-0.22,-0.33,-0.03,-0.03,-0.21,-0.14,-0.18,0.38,0.07,-0.05,-0.15,-0.07,-0.03,-0.13,0.08,-0.2,0.06,0.13,-0.03,0.03,-0.03,-0.04,-0.07,0.1,0.21,0.0,0.09,0.04,0.09,0.04,0.25,0.1,0.32,-0.06,0.36,0.2,0.41,0.39,0.24,0.24,0.37,0.27,0.27,0.1,0.3,0.17,0.26,0.39,0.21,0.08,0.42,0.11,0.2,0.03,0.13,0.51,0.3,0.44,0.13,0.08,0.15,0.12,0.43,-0.19,0.22,0.09,-0.06,0.22,0.04,0.45,0.15,0.4,0.13,0.34,0.29,0.22,0.33,0.48,0.2,0.32,0.53,0.37,0.47,0.26,0.16,0.06,0.05,0.32,-0.15,0.2,-0.02,0.3,0.41,0.21,0.18,0.35,0.15,-0.1,0.41,0.01,0.2,0.17,0.25,-0.03,0.12,-0.08,-0.07,-0.27,0.41,-0.26,-0.17,-0.06,0.3,-0.08,-0.08,0.06,-0.28,-0.11,-0.59,-0.19,-0.36,-0.18,-0.32,-0.62,-0.38,-0.31,-0.43,-0.38,-0.47,-0.48,-0.31,-0.22,-0.52,-0.19,-0.62,-0.63,-0.59,-0.07,-0.23,-0.41,-0.23,-0.13,-0.5,-0.08,-0.21,-0.32,-0.28,-0.33,-0.28,0.13,0.2,-0.25,-0.03,-0.36,0.13,-0.3,-0.24,-0.27,-0.27,0.16,-0.33,-0.35,-0.33,0.09,-0.21,0.14,-0.22,-0.42,-0.03,-0.03,0.08,-0.1,0.0,-0.3,0.09,-0.15,-0.22,-0.23,0.03,-0.35,-0.04,-0.22,0.2,-0.28,-0.04,-0.38,-0.11,-0.15,-0.29,0.09,-0.26,-0.21,-0.24,-0.25,-0.04,-0.33,-0.29,-0.17,-0.34,-0.27,-0.24,-0.09,0.06,-0.26,-0.06,-0.19,-0.19,0.06,-0.23,-0.06,-0.31,-0.2,0.0,0.07,0.03,-0.03,0.05,-0.39,-0.01,0.14,0.13,-0.28,0.0,-0.23,0.06,-0.08,-0.12,-0.23,-0.1,-0.04,0.04,-0.27,-0.11,0.19,-0.23,0.24,-0.01,0.16,-0.17,-0.23,-0.13,-0.04,0.44,0.0,-0.09,-0.07,-0.08,0.22,-0.11,0.01,-0.21,0.05,-0.19,-0.05,-0.21,0.22,0.01,-0.03,-0.38,-0.34,-0.35,-0.38,-0.5,-0.44,-0.44,-0.43,-0.63,-0.44,-0.16,-0.05,-0.46,-0.36,-0.51,-0.36,-0.27,-0.4,-0.44,-0.33,-0.33,-0.78,-0.37,-0.64,-0.23,-0.43,-0.55,-0.59,-0.47,-0.26,-0.46,-0.3,-0.35,-0.49,-0.19,-0.43,-0.46,-0.27,-0.13,-0.31,-0.29,-0.19,-0.03,-0.03,-0.23,-0.19,-0.32,0.0,-0.25,-0.11,-0.07,-0.11,-0.17,-0.35,-0.2,0.04,-0.12,-0.21,-0.41,-0.3,0.01,0.06,-0.07,-0.06,0.01,-0.09,0.03,0.17,0.1,0.03,0.47,0.15,0.23,0.04,0.61,-0.02,0.12,-0.13,0.06,0.03,-0.15,-0.11,0.24,-0.14,-0.03,0.02,-0.32,0.01,0.1,-0.19,-0.17,-0.13,-0.42,-0.32,-0.4,0.2,-0.49,-0.06,-0.38,-0.27,-0.32,-0.2,-0.08,-0.43,0.06,-0.23,-0.1,-0.26,-0.18,-0.51,-0.28,-0.52,-0.35,-0.34,-0.17,-0.31,-0.39,-0.18,-0.43,-0.09,-0.12,-0.02,-0.05,-0.07,-0.09,-0.07,0.15,-0.05,-0.25,-0.25,-0.2,-0.36,0.06,-0.34,-0.04,0.0,0.26,0.09,0.04,0.1,0.26,-0.17,-0.08,0.16,0.02,-0.16,0.02,0.48,-0.16,-0.24,0.23,-0.22,-0.26,-0.35,0.07,0.29,-0.09,0.18,0.09,0.05,0.34,0.03,0.17,0.22,0.1,0.34,0.44,-0.04,0.06,0.42,0.1,0.0,-0.29,0.11,0.04,-0.02,0.5,-0.2,0.05,0.23,0.04,0.13,0.22,0.37,0.17,-0.15,-0.23,0.18,0.27,0.1,-0.26,-0.31,0.01,-0.05,-0.26,-0.16,-0.1,-0.23,-0.3,-0.37,-0.2,-0.02,-0.37,0.03,-0.46,-0.03,-0.03,-0.17,-0.12,-0.22,-0.07,-0.27,-0.28,-0.03,-0.18,-0.2,-0.55,-0.28,-0.42,-0.16,0.01,-0.01,-0.17,-0.35,-0.17,-0.46,-0.28,-0.06,-0.08,-0.2,-0.16,-0.17,-0.44,-0.22,-0.67,-0.31,-0.7,-0.13,-0.26,0.01,-0.26,-0.19,0.04,-0.4,-0.37,-0.34,-0.33,-0.2,-0.04,-0.03,-0.05,-0.27,0.16,-0.2,0.2,-0.26,0.06,-0.1,-0.04,-0.2,0.23,-0.2,-0.34,0.41,0.06,-0.25,-0.06,-0.2,0.06,-0.12,0.01,-0.17,-0.23,-0.07,0.01,-0.18,-0.31,0.07,-0.03,-0.09,0.22,-0.03,0.14,0.2,-0.02,-0.17,-0.32,-0.18,-0.1,-0.43,-0.24,-0.36,-0.25,-0.25,-0.47,-0.42,-0.39,0.19,-0.31,0.01,-0.29,0.15,-0.25,-0.44,-0.04,-0.2,-0.12,-0.36,-0.23,0.02,0.03,0.0,-0.09,-0.35,0.04,0.11,0.18,-0.22,-0.01,0.13,-0.19,0.28,-0.08,0.49,0.17,0.2,0.07,0.0,-0.03,0.23,0.12,0.03,0.01,0.08,-0.14,0.01,0.17,0.63,0.19,0.02,-0.08,-0.11,0.15,0.08,-0.14,-0.01,0.0,-0.28,0.1,0.16,-0.24,0.08,-0.16,0.18,-0.04,0.11,0.16,0.15,-0.05,0.26,0.13,-0.01,0.35,0.18,0.05,0.15,-0.38,-0.2,-0.03,-0.03,-0.28,-0.38,-0.42,0.0,-0.14,-0.08,-0.3,-0.36,-0.53,-0.26,-0.23,-0.25,0.06,-0.06,-0.29,-0.35,-0.31,-0.05,-0.12,-0.49,0.0,-0.06,-0.18,-0.07,-0.3,0.06,-0.03,0.01,0.39,-0.2,0.06,-0.03,0.23,-0.19,0.01,0.03,-0.27,-0.21,-0.15,-0.14,-0.21,-0.21,-0.32,-0.23,-0.43,-0.2,-0.09,-0.2,-0.33,-0.2,-0.2,-0.12,-0.04,-0.21,0.05,-0.16,-0.1,-0.16,0.01,-0.25,0.09,-0.45,0.27,-0.13,-0.44,-0.1,-0.17,0.06,-0.11,-0.21,-0.11,-0.27,0.06,-0.17,0.06,-0.17,-0.31,-0.2,-0.31,-0.33,-0.17,-0.4,-0.33,-0.33,-0.41,-0.31,0.1,-0.25,0.1,-0.42,-0.23,-0.46,-0.5,-0.41,-0.33,-0.39,-0.31,-0.52,-0.3,-0.44,-0.23,-0.5,-0.34,-0.18,0.01,-0.37,-0.12,-0.11,-0.31,-0.08,-0.07,0.12,-0.06,-0.13,0.04,-0.19,-0.05,-0.03,-0.08,-0.2,0.02,-0.2,-0.03,-0.15,-0.13,0.01,-0.48,-0.32,-0.48,-0.35,0.01,-0.65,-0.12,-0.45,-0.6,-0.4,-0.36,-0.46,-0.61,-0.12,-0.61,0.21,-0.21,0.25,-0.18,0.24,0.24,-0.24,0.1,0.06,-0.07,0.06,0.07,0.12,0.16,-0.17,0.07,0.23,-0.03,-0.03,-0.14,-0.04,0.16,-0.08,0.32,-0.14,0.06,0.24,0.06,-0.08,-0.24,-0.13,0.18,0.07,0.18,-0.03,0.22,0.04,0.12,0.09,0.1,0.22,0.18,0.19,0.06,0.11,0.29,0.46,0.38,0.06,0.33,0.21,0.0,0.03,0.04,0.06,0.06,-0.26,-0.32,-0.16,0.13,0.32,-0.04,0.04,0.17,-0.41,0.04,0.12,0.12,0.24,0.19,0.26,0.26,0.3,0.33,-0.03,0.55,0.23,0.33,0.29,0.26,0.06,0.38,0.52,0.42,0.52,0.49,0.77,0.3,0.55,0.41,0.71,0.47,0.54,0.62,0.41,0.63,0.31,0.54,0.25,0.46,0.74,0.2,0.37,0.28,0.15,0.56,0.45,0.06,0.38,0.34,0.06,0.13,0.38,0.04,0.08,0.41,0.36,0.26,0.26,0.34,0.07,-0.04,-0.15,-0.12,0.11,0.03,-0.02,0.11,0.07,0.14,0.27,0.09,0.1,0.04,0.56,0.42,0.32,0.27,0.04,0.12,-0.05,0.1,0.29,0.27,0.4,0.2,0.22,0.1,0.33,-0.08,0.02,0.1,-0.26,0.26,-0.15,-0.14,0.33,0.42,-0.05,0.08,-0.33,-0.05,0.17,-0.18,-0.16,-0.05,0.01,-0.11,-0.03,0.04,0.23,-0.15,0.0,0.1,0.11,-0.06,0.12,-0.03,0.1,-0.12,-0.15,-0.12,-0.07,-0.07,-0.22,-0.06,0.12,-0.05,-0.24,0.02,-0.03,0.12,-0.12,-0.26,-0.03,-0.08,-0.09,0.03,0.09,-0.09,0.17,0.06,0.23,0.29,-0.03,0.08,-0.21,0.06,-0.03,-0.19,0.17,0.17,-0.08,-0.08,-0.03,0.02,0.17,0.06,0.04,0.02,-0.22,-0.37,-0.35,-0.21,-0.38,-0.22,-0.4,-0.12,-0.37,-0.07,-0.49,-0.48,-0.43,-0.44,-0.2,-0.14,-0.23,-0.32,-0.26,-0.37,0.0,-0.4,-0.37,-0.42,-0.32,-0.28,-0.35,-0.33,-0.14,-0.43,-0.35,-0.51,-0.31,0.0,-0.45,-0.36,-0.34,-0.4,-0.29,-0.32,-0.18,-0.16,-0.26,-0.09,-0.41,-0.1,-0.14,-0.02,-0.03,0.04,-0.19,0.12,-0.15,-0.13,-0.41,-0.35,-0.29,-0.29,-0.18,-0.34,-0.24,-0.14,0.0,0.06,-0.09,-0.08,-0.16,-0.45,-0.25,-0.09,-0.28,-0.41,-0.21,-0.09,0.16,-0.02,0.23,0.13,-0.17,-0.12,0.04,0.08,0.08,-0.1,-0.15,0.1,-0.04,-0.13,-0.25,-0.12,-0.16,-0.18,-0.11,-0.31,0.3,-0.38,-0.08,-0.4,-0.3,-0.36,-0.13,-0.15,-0.32,-0.5,-0.53,-0.25,-0.38,0.0,0.15,-0.15,-0.22,-0.23,-0.19,-0.12,-0.14,0.04,0.02,0.15,0.18,0.17,-0.12,-0.03,-0.04,0.15,0.19,-0.04,0.37,-0.08,0.04,-0.12,-0.04,-0.09,0.12,-0.13,-0.08,0.02,-0.05,-0.27,0.02,0.08,-0.16,0.14,0.0,0.23,0.06,0.26,-0.02,0.31,0.14,0.36,0.09,0.49,0.45,0.71,0.17,0.34,0.01,0.32,0.41,0.1,0.26,-0.1,0.22,0.1,0.32,-0.07,-0.05,0.07,-0.01,-0.04,0.12,-0.1,0.1,0.0,0.09,-0.04,0.06,0.15,-0.1,0.17,0.22,-0.08,0.01,0.19,0.04,0.1,0.07,0.24,0.03,-0.17,0.15,-0.14,0.31,-0.1,-0.11,-0.02,0.37,0.08,0.31,0.03,0.37,0.39,0.3,0.54,0.06,0.34,0.06,-0.2,-0.1,-0.19,-0.06,-0.43,-0.2,-0.17,-0.35,-0.36,-0.44,-0.48,-0.41,-0.69,-0.55,-0.43,-0.3,-0.77,-0.58,-0.43,-0.51,-0.51,-0.5,-0.31,-0.43,-0.24,-0.41,-0.3,0.18,-0.02,-0.18,-0.23,0.08,0.08,-0.25,-0.26,-0.15,-0.17,-0.26,-0.29,-0.13,-0.03,-0.13,-0.18,-0.15,0.13,-0.13,0.09,0.03,0.08,0.19,0.06,-0.32,0.0,-0.03,-0.15,-0.14,-0.28,0.06,-0.12,0.11,-0.05,-0.08,-0.37,-0.25,-0.3,-0.13,-0.42,-0.45,-0.4,-0.19,-0.49,0.27,-0.5,-0.47,-0.36,-0.13,-0.37,-0.27,-0.14,-0.16,-0.28,-0.22,0.15,0.13,-0.2,0.06,-0.18,-0.31,-0.45,-0.12,0.06,0.12,0.1,-0.3,-0.04,-0.3,-0.33,-0.05,-0.29,-0.1,-0.13,-0.16,-0.37,-0.16,-0.49,-0.22,-0.48,-0.39,-0.28,-0.29,-0.35,-0.18,0.1,-0.4,-0.38,-0.35,0.0,-0.03,-0.19,-0.23,-0.09,-0.28,-0.37,-0.33,-0.19,-0.36,-0.13,-0.28,-0.29,-0.28,-0.25,0.02,0.04,-0.06,-0.19,0.06,-0.03,-0.07,0.08,-0.08,0.05,-0.06,0.06,0.41,0.18,0.07,-0.04,-0.11,-0.07,0.04,0.12,-0.22,-0.03,0.14,-0.03,0.26,0.02,-0.04,-0.11,-0.13,-0.08,-0.12,0.11,0.19,0.38,0.21,-0.03,0.41,0.04,0.0,0.53,0.45,0.32,0.12,0.58,0.25,0.26,0.14,0.37,0.19,0.46,0.27,0.6,0.46,0.3,0.66,0.24,0.45,0.42,0.4,0.61,0.46,0.28,0.37,0.61,0.46,0.43,0.5,0.21,0.21,0.46,0.77,0.44,0.27,0.69,0.54,0.6,0.52,0.48,0.78,0.3,0.55,0.73,0.57,0.48,0.83,1.02,0.69,0.82,0.5,0.91,0.51,0.48,0.77,0.45,0.26,0.54,0.67,0.73,0.43,0.64,0.49,0.39,0.21,0.22,0.21,0.59,0.24,0.26,0.19,0.36,0.44,0.33,0.19,0.07,-0.01,0.31,-0.08,0.35,0.47,0.27,-0.05,-0.28,-0.32,-0.43,-0.39,-0.45,-0.01,-0.19,-0.28,-0.19,-0.03,-0.09,-0.02,0.07,0.12,-0.05,-0.17,0.06,-0.12,0.29,0.04,-0.01,-0.33,-0.09,0.13,-0.13,0.23,0.06,-0.23,-0.07,-0.07,-0.2,0.14,0.02,0.0,0.34,-0.07,-0.32,-0.08,-0.22,-0.01,-0.19,-0.11,-0.03,-0.06,0.05,-0.05,-0.03,0.09,-0.21,-0.15,-0.13,-0.14,-0.09,-0.26,-0.2,-0.18,-0.14,-0.17,-0.11,-0.3,-0.34,-0.37,-0.12,0.14,-0.22,-0.22,-0.25,0.03,-0.3,0.13,-0.34,0.1,0.12,-0.19,-0.16,0.1,0.14,0.21,-0.1,0.13,0.1,0.13,0.33,0.21,-0.1,0.23,0.22,0.47,0.0,0.08,0.12,0.27,0.13,-0.01,0.04,-0.07,-0.15,0.14,0.24,-0.17,0.28,-0.04,-0.16,0.03,-0.09,0.11,0.0,0.01,0.41,-0.03,-0.21,0.03,-0.08,0.19,0.05,0.28,0.2,0.42,0.22,0.4,0.14,0.29,0.04,-0.01,-0.08,0.06,-0.13,-0.17,-0.15,0.29,-0.21,-0.12,-0.16,-0.01,0.11,0.09,-0.05,0.07,-0.01,-0.04,-0.18,0.04,0.36,0.29,0.0,0.19,0.23,0.31,0.17,-0.24,-0.36,-0.29,-0.45,-0.29,-0.26,-0.45,-0.37,-0.23,-0.31,-0.55,-0.46,-0.35,-0.28,-0.09,-0.08,-0.57,-0.43,-0.28,-0.12,-0.2,-0.22,-0.44,-0.44,-0.25,-0.47,-0.49,-0.17,-0.5,-0.53,-0.39,-0.24,-0.58,-0.03,-0.36,-0.3,-0.12,-0.2,-0.53,-0.1,-0.47,-0.36,-0.56,-0.28,-0.33,-0.33,-0.37,-0.27,-0.41,-0.44,-0.17,-0.11,-0.07,-0.33,0.18,-0.07,-0.28,-0.24,-0.15,-0.14,-0.23,0.1,-0.21,-0.36,-0.14,-0.06,-0.28,-0.1,0.06,-0.2,-0.08,-0.19,-0.07,-0.42,-0.1,-0.35,-0.37,-0.28,-0.23,-0.38,-0.15,-0.33,-0.26,-0.01,-0.44,0.02,-0.17,-0.39,-0.43,0.14,-0.11,-0.11,-0.16,-0.28,-0.44,-0.2,-0.1,-0.4,0.01,-0.13,-0.2,-0.15,-0.18,-0.29,0.07,-0.19,-0.3,-0.17,-0.31,-0.26,-0.44,-0.49,-0.35,-0.4,-0.45,0.06,-0.27,-0.59,-0.33,-0.01,-0.18,-0.29,-0.13,-0.41,-0.5,-0.32,-0.37,-0.2,0.01,-0.21,-0.07,-0.38,-0.03,-0.41,0.12,-0.36,-0.33,-0.42,-0.24,-0.03,-0.14,-0.13,-0.5,0.16,-0.06,-0.13,-0.05,0.06,-0.12,-0.13,-0.11,0.04,-0.1,0.14,0.03,0.04,-0.28,-0.11,-0.17,-0.2,0.07,-0.03,0.03,-0.4,0.05,-0.08,0.1,-0.17,-0.05,-0.11,-0.44,-0.14,-0.35,-0.31,-0.26,-0.32,-0.3,-0.04,-0.14,-0.16,-0.18,0.28,-0.04,-0.13,-0.02,-0.01,0.18,0.04,-0.06,-0.08,0.09,0.07,0.13,0.19,0.15,0.35,0.04,0.1,0.04,0.29,0.06,0.13,-0.1,0.21,-0.18,0.2,0.06,-0.03,-0.23,0.12,0.12,0.15,-0.09,-0.13,-0.3,-0.27,-0.11,-0.21,0.25,-0.3,-0.24,-0.44,-0.03,-0.35,-0.58,-0.37,-0.23,-0.4,-0.24,-0.48,-0.3,-0.29,-0.2,-0.22,-0.34,-0.06,-0.1,0.02,-0.5,-0.27,-0.31,-0.43,-0.35,-0.35,-0.47,-0.39,-0.47,-0.5,-0.26,-0.55,-0.52,-0.31,-0.35,-0.47,-0.47,-0.03,-0.48,-0.35,-0.3,-0.46,-0.32,-0.44,-0.26,-0.13,0.02,-0.55,-0.41,-0.07,-0.47,-0.38,-0.3,-0.31,-0.52,-0.56,-0.34,-0.49,-0.16,-0.53,-0.37,-0.31,0.14,-0.31,-0.06,-0.05,0.0,-0.14,0.31,0.12,-0.21,-0.15,-0.05,0.14,0.32,-0.02,0.04,-0.22,0.0,0.45,0.0,-0.15,0.27,-0.25,-0.23,-0.43,-0.22,-0.34,-0.16,-0.07,-0.28,0.04,-0.51,-0.59,-0.2,-0.19,-0.23,-0.36,-0.19,-0.37,-0.37,-0.19,-0.27,-0.43,-0.23,0.07,-0.04,0.11,-0.07,0.34,-0.47,-0.44,-0.22,-0.42,-0.48,-0.21,-0.47,-0.2,-0.68,-0.39,-0.48,-0.31,-0.23,-0.51,-0.68,-0.41,-0.3,-0.56,-0.6,-0.51,0.03,-0.55,-0.45,-0.61,-0.22,-0.4,-0.38,-0.3,-0.18,-0.23,-0.15,0.11,-0.28,-0.38,-0.25,-0.41,-0.5,-0.01,-0.44,-0.12,-0.44,0.08,-0.1,-0.14,-0.04,-0.06,0.02,-0.08,0.13,0.04,-0.07,0.1,0.18,0.14,0.47,0.45,0.6,0.16,0.34,0.2,0.6,0.18,0.27,0.55,0.43,0.26,0.43,0.38,0.58,0.5,0.17,0.13,0.22,0.29,0.42,0.31,0.64,0.12,0.12,0.39,0.3,0.3,0.23,0.3,0.01,0.51,0.1,0.23,0.38,0.25,0.15,-0.08,0.14,0.22,0.04,0.14,0.08,0.24,-0.04,0.0,0.01,0.02,-0.09,0.24,-0.03,-0.22,0.08,-0.19,-0.38,-0.26,-0.37,-0.33,0.04,-0.03,-0.06,0.15,0.01,-0.27,-0.14,0.1,0.01,-0.05,0.02,0.03,-0.09,-0.09,0.04,-0.27,0.16,-0.05,0.1,-0.05,-0.17,-0.23,0.0,0.13,0.15,0.13,0.18,0.29,0.08,0.09,0.44,0.25,0.15,0.17,0.18,0.21,0.32,0.34,0.17,-0.02,0.01,0.36,0.1,0.12,0.18,0.12,0.01,-0.07,-0.17,0.0,-0.21,0.2,0.0,-0.01,0.05,0.57,0.57,0.33,0.68,0.23,0.1,0.17,0.14,0.22,0.06,0.22,-0.02,0.24,0.4,-0.21,-0.03,-0.21,-0.08,-0.04,0.2,0.12,-0.08,0.0,0.31,0.02,-0.09,-0.08,-0.09,0.08,0.12,0.0,0.27,0.5,0.07,0.1,0.01,0.19,0.22,0.03,-0.01,0.1,-0.09,0.03,0.44,-0.02,0.2,0.1,-0.22,0.09,-0.08,-0.05,0.03,0.18,0.18,-0.08,0.06,0.28,-0.09,0.1,0.04,0.12,0.06,-0.3,0.34,-0.11,0.02,-0.08,0.14,0.12,0.15,-0.13,0.15,0.06,-0.04,0.26,-0.08,-0.02,0.06,0.04,0.12,-0.19,0.02,0.02,-0.25,-0.23,-0.17,-0.35,-0.44,-0.24,-0.35,-0.19,-0.06,0.05,0.06,-0.24,-0.14,-0.08,-0.16,-0.06,0.01,0.22,0.0,-0.15,-0.02,0.02,0.13,-0.05,0.12,0.12,-0.01,0.14,0.22,0.16,0.11,0.64,0.31,0.16,0.21,0.18,0.11,0.27,0.1,0.2,0.04,0.24,0.24,0.07,0.36,0.2,-0.01,0.52,0.23,0.18,-0.09,-0.03,-0.29,0.36,0.13,-0.02,-0.14,-0.1,0.03,-0.08,-0.22,-0.03,-0.4,-0.16,0.01,0.18,-0.24,-0.19,-0.04,-0.49,-0.27,0.43,-0.1,-0.18,-0.03,-0.08,-0.14,0.07,0.11,-0.11,-0.09,0.24,-0.06,-0.07,-0.08,-0.03,0.03,0.2,0.26,-0.06,0.35,0.47,-0.01,0.1,0.43,0.16,0.08,0.19,0.18,0.1,-0.02,0.08,0.09,0.0,0.04,0.31,0.0,0.43,0.02,-0.03,0.22,0.35,-0.02,0.51,0.53,0.3,0.27,0.22,0.43,0.46,0.31,0.3,0.32,0.41,0.03,0.43,0.53,0.13,0.2,0.54,0.17,0.28,0.03,0.31,0.1,0.28,0.2,0.04,0.18,0.09,0.22,0.3,0.02,-0.37,0.0,0.15,-0.01,-0.25,0.1,0.51,0.41,0.25,-0.08,0.41,0.21,-0.01,0.02,0.11,0.18,0.31,0.35,0.18,0.52,0.32,0.2,0.32,0.34,0.35,0.44,0.43,0.47,0.45,0.57,0.29,0.61,0.47,0.43,0.57,0.25,0.3,0.28,0.11,0.27,0.31,0.28,0.56,-0.03,0.04,0.37,0.09,0.26,0.12,0.14,0.06,0.19,-0.01,0.15,-0.05,0.1,0.1,0.23,0.26,0.17,0.14,0.21,0.47,-0.03,0.26,0.31,0.06,0.1,0.13,0.08,0.1,-0.18,0.04,0.07,0.37,-0.05,0.27,-0.14,0.09,0.06,-0.16,-0.04,-0.19,-0.22,-0.13,0.31,0.4,0.35,0.06,-0.04,0.29,0.02,0.05,0.01,0.08,-0.11,0.09,0.18,0.22,-0.03,-0.04,0.17,0.08,0.14,0.37,0.16,0.1,0.56,0.08,0.28,0.0,0.49,-0.16,0.02,0.42,-0.09,0.0,0.44,0.08,-0.07,0.24,0.0,-0.04,0.11,-0.2,0.0,0.11,-0.38,-0.16,-0.09,-0.01,0.1,0.03,-0.25,-0.04,-0.15,0.12,0.22,-0.03,-0.07,0.13,-0.09,-0.01,-0.06,-0.12,0.24,-0.24,-0.05,-0.27,-0.06,0.02,-0.1,0.16,-0.3,-0.23,-0.1,-0.09,-0.28,-0.26,-0.34,0.02,-0.25,-0.05,-0.11,-0.18,-0.17,-0.24,-0.01,0.02,-0.28,-0.31,0.07,-0.31,0.03,-0.03,0.18,0.13,-0.19,0.12,0.11,0.19,0.13,0.11,0.24,-0.01,0.26,0.16,0.35,0.15,0.35,0.3,0.51,0.6,0.36,0.31,0.15,0.08,0.2,0.39,0.25,0.41,0.22,0.26,0.33,0.33,0.22,0.3,0.56,0.42,0.59,0.28,0.18,0.46,0.62,0.78,0.8,0.4,0.45,0.64,0.57,0.43,0.92,0.69,0.55,0.58,0.49,0.89,0.42,0.36,0.76,0.38,0.66,0.42,0.71,0.15,0.55,0.47,0.48,0.14,0.37,0.48,0.15,0.4,0.56,0.27,0.57,0.19,0.3,0.61,0.37,0.42,0.39,0.4,0.4,0.35,0.35,0.47,0.59,0.6,0.1,0.27,-0.02,0.16,-0.16,0.21,0.07,0.2,-0.03,0.22,0.27,0.09,-0.03,-0.06,0.06,0.07,0.09,-0.12,0.38,-0.14,-0.08,-0.29,-0.15,-0.08,-0.01,-0.11,0.4,-0.2,-0.13,0.04,-0.2,-0.61,-0.34,-0.03,-0.54,-0.48,-0.56,-0.17,0.01,-0.29,-0.16,-0.18,-0.46,-0.31,-0.43,0.21,-0.05,-0.16,-0.24,-0.15,-0.11,-0.26,-0.1,-0.08,-0.29,-0.05,-0.3,-0.02,0.13,0.28,-0.1,0.26,0.06,0.28,0.3,0.28,0.11,0.36,0.28,0.31,0.38,0.04,0.17,0.26,0.26,0.47,0.45,0.28,0.36,0.12,0.42,0.77,0.36,0.53,0.41,0.47,0.36,0.29,0.36,0.39,0.68,0.33,0.38,0.43,1.01,0.38,0.35,0.33,0.57,0.33,0.31,0.66,0.44,0.38,0.46,0.26,0.37,0.68,0.43,0.83,0.5,0.58,0.41,1.06,0.55,0.66,0.42,0.58,0.14,0.27,0.18,0.12,0.19,0.11,0.09,0.11,-0.14,0.25,-0.1,0.1,0.11,-0.04,0.01,0.01,0.02,0.11,-0.07,-0.05,0.15,0.21,-0.21,0.06,-0.03,0.19,0.1,0.06,0.31,-0.11,0.15,0.12,0.32,-0.13,-0.05,0.0,0.03,0.35,0.28,0.19,0.13,-0.14,0.23,-0.22,0.25,0.15,0.2,0.55,0.02,0.09,-0.03,0.08,-0.02,0.14,-0.02,-0.08,-0.16,-0.17,-0.29,-0.25,-0.1,0.08,0.02,-0.15,-0.14,-0.07,-0.25,-0.03,-0.16,-0.36,0.04,-0.15,-0.23,-0.54,-0.26,0.02,0.18,-0.05,-0.24,-0.04,0.0,-0.13,-0.03,-0.08,0.28,-0.18,-0.2,0.02,-0.14,-0.1,-0.02,0.03,0.09,-0.17,0.15,-0.09,0.09,0.05,0.24,-0.06,-0.04,0.23,-0.32,-0.08,-0.02,-0.1,-0.15,-0.14,0.15,0.12,0.18,0.0,0.13,-0.07,0.42,-0.05,0.16,-0.24,-0.24,0.2,0.02,0.03,-0.11,-0.26,0.16,0.24,-0.24,-0.14,-0.03,-0.04,0.37,0.27,-0.26,-0.27,0.01,-0.08,-0.17,-0.05,-0.28,0.24,-0.28,0.42,-0.23,-0.21,-0.04,-0.01,-0.33,-0.08,-0.14,0.02,0.29,0.2,-0.1,0.16,0.2,0.52,-0.05,0.35,-0.03,-0.04,0.04,-0.06,0.15,-0.08,0.1,0.02,-0.16,-0.13,-0.01,0.03,0.04,-0.15,-0.02,-0.08,0.14,-0.04,-0.09,0.13,-0.12,0.09,0.2,0.06,0.03,-0.09,-0.22,-0.06,-0.23,-0.25,-0.17,0.06,0.31,0.15,-0.22,-0.02,-0.08,-0.33,0.08,-0.14,0.27,-0.07,0.02,0.25,0.29,0.14,0.41,0.24,0.18,-0.03,0.31,-0.06,0.12,0.04,0.19,0.33,0.55,0.44,0.08,0.13,0.36,0.35,0.19,0.06,-0.12,0.02,0.19,-0.05,0.24,0.45,0.05,0.1,0.2,0.08,0.51,0.37,0.35,0.02,0.25,0.24,0.35,0.25,0.41,0.24,0.21,0.37,0.25,0.62,0.26,0.53,0.47,0.4,0.91,0.95,0.41,0.35,0.5,0.54,0.42,0.33,0.41,0.46,0.57,0.53,0.1,0.34,0.18,0.45,0.47,0.28,0.13,0.61,0.25,0.06,0.57,0.19,0.57,0.36,0.39,0.66,0.7,0.38,0.73,0.3,0.29,0.26,0.18,0.44,0.31,0.14,0.06,0.5,0.42,0.43,0.15,0.02,0.13,0.33,0.36,0.1,0.1,0.15,-0.08,0.04,-0.14,0.22,-0.15,0.47,0.09,0.01,0.03,0.26,0.15,0.15,0.08,-0.04,0.01,0.21,0.14,-0.03,0.18,0.33,0.17,0.26,0.16,0.05,-0.36,-0.02,0.25,0.35,0.34,-0.1,0.17,0.23,0.14,0.13,0.09,0.33,-0.03,0.09,0.19,0.1,0.06,0.15,0.33,-0.05,0.08,0.17,0.3,0.11,0.3,0.05,0.22,0.19,0.22,0.17,0.34,0.25,0.14,0.06,-0.04,0.27,0.17,0.16,-0.14,-0.03,-0.05,-0.25,-0.27,-0.2,-0.05,0.13,-0.08,-0.17,0.02,-0.35,-0.07,-0.1,0.14,-0.24,-0.32,-0.18,-0.26,-0.43,-0.3,-0.2,0.03,-0.18,-0.24,-0.27,-0.08,-0.1,-0.29,-0.22,-0.02,0.06,-0.08,-0.28,-0.38,-0.11,0.02,0.18,-0.07,-0.03,-0.15,-0.34,0.22,0.21,0.03,-0.16,-0.07,-0.13,-0.05,-0.08,-0.17,-0.27,-0.05,-0.04,-0.14,-0.27,-0.21,-0.39,-0.17,-0.12,-0.09,0.0,-0.26,-0.16,-0.05,-0.1,-0.25,0.19,-0.32,-0.29,0.04,0.02,0.15,0.05,-0.11,0.14,0.33,0.15,0.04,-0.01,0.21,0.33,0.04,0.35,0.31,0.12,0.49,0.26,0.26,0.33,0.04,0.22,0.4,0.42,0.25,0.45,0.24,0.49,0.28,0.4,0.75,0.44,0.59,0.23,0.41,0.35,0.36,0.44,0.44,-0.08,0.24,0.2,0.08,0.14,0.13,0.31,0.39,0.36,0.42,0.21,0.03,0.45,0.36,0.41,0.38,0.1,0.16,0.29,0.26,0.42,0.15,0.16,0.32,0.26,0.45,0.34,0.22,0.05,0.2,0.1,0.24,0.21,0.24,0.2,0.3,0.03,0.15,0.06,0.2,0.25,0.16,0.41,0.01,0.26,0.18,0.05,0.44,0.24,0.01,-0.01,0.29,0.1,-0.08,0.29,0.31,-0.2,-0.05,0.12,0.42,-0.26,-0.06,0.01,-0.12,-0.15,-0.27,0.02,0.07,-0.04,0.07,0.15,0.0,0.07,-0.11,0.26,0.64,0.0,0.21,0.16,0.16,0.04,0.01,-0.11,-0.02,0.24,0.42,0.26,0.4,0.15,0.18,0.26,0.19,0.14,0.12,0.02,0.06,0.33,0.02,0.13,0.17,-0.11,-0.09,0.06,-0.25,-0.18,0.1,0.24,0.09,0.04,-0.21,-0.19,-0.26,-0.12,-0.15,-0.28,-0.25,-0.29,-0.08,-0.28,-0.05,-0.3,0.22,-0.02,-0.37,-0.28,-0.39,-0.07,-0.21,-0.05,-0.1,-0.27,-0.16,-0.22,-0.21,-0.05,0.23,-0.13,-0.52,-0.17,-0.4,0.0,-0.03,-0.42,-0.45,-0.35,0.1,-0.01,-0.2,-0.07,0.06,-0.09,-0.13,-0.13,0.01,-0.08,-0.08,0.02,-0.05,-0.33,-0.15,0.2,-0.28,0.13,-0.38,-0.15,-0.21,-0.24,0.48,-0.14,0.0,0.26,0.11,0.17,-0.12,-0.19,-0.24,-0.03,-0.01,-0.19,-0.28,-0.24,-0.07,0.23,0.22,-0.2,0.5,-0.2,-0.2,0.28,-0.11,0.01,0.22,-0.08,-0.06,-0.14,-0.03,0.06,-0.27,0.06,0.08,0.16,0.32,-0.04,0.3,0.54,0.17,0.13,0.23,0.24,-0.03,-0.08,0.04,0.3,0.14,0.32,0.13,0.01,0.17,0.27,0.09,-0.21,-0.06,-0.03,-0.2,0.21,0.23,0.33,0.23,0.15,0.12,0.52,-0.03,0.08,0.01,0.13,0.34,0.24,0.19,0.27,-0.03,0.07,0.01,-0.13,-0.09,0.13,0.26,-0.19,0.17,0.19,-0.07,-0.16,-0.27,-0.1,0.01,0.27,-0.2,-0.05,0.22,0.21,-0.25,0.09,0.32,-0.14,-0.16,-0.13,-0.36,-0.01,0.12,-0.04,-0.05,0.07,0.0,0.06,0.0,0.12,0.27,0.23,0.37,0.09,0.12,0.0,0.0,0.37,0.28,0.53,0.2,0.02,0.2,0.2,0.23,-0.12,0.0,0.07,-0.06,0.33,-0.02,0.16,0.06,0.1,-0.21,0.39,0.03,-0.2,0.05,0.3,0.15,0.05,0.1,0.09,-0.01,-0.13,0.06,0.07,0.08,-0.16,0.16,0.12,0.16,0.16,0.34,0.31,0.58,0.09,-0.04,0.2,0.22,-0.06,0.15,0.08,0.12,0.12,-0.21,-0.12,-0.32,-0.1,0.41,-0.11,-0.34,-0.34,-0.4,0.1,-0.44,-0.01,-0.24,-0.13,-0.03,0.05,-0.09,-0.03,-0.17,-0.26,-0.07,-0.45,-0.04,-0.22,-0.06,-0.03,-0.03,-0.02,0.22,-0.11,0.18,0.21,0.18,0.12,0.15,0.21,0.23,-0.07,-0.03,-0.01,-0.12,0.1,0.0,0.04,-0.03,-0.28,0.01,0.12,0.12,0.35,-0.06,0.33,-0.05,0.03,0.4,-0.08,0.09,0.07,0.4,-0.04,0.21,-0.02,0.24,0.05,0.1,0.05,0.17,0.15,0.03,0.32,0.28,0.5,0.12,-0.17,0.01,0.27,0.32,0.29,0.56,0.19,0.21,0.13,0.21,0.08,0.47,-0.1,0.13,0.23,0.46,0.36,0.13,0.35,0.3,0.59,0.49,0.46,0.49,0.37,0.38,0.9,0.33,0.3,0.24,0.39,0.72,0.7,0.32,-0.03,0.23,0.19,0.07,0.09,0.04,0.16,-0.07,-0.26,0.05,-0.08,0.08,-0.02,-0.12,0.03,-0.03,0.14,0.06,0.04,-0.28,0.06,-0.15,-0.19,0.04,0.03,0.03,-0.23,-0.34,-0.16,0.19,-0.06,-0.1,0.24,0.23,-0.17,0.38,0.03,0.01,0.06,-0.19,0.18,-0.18,0.01,0.19,-0.07,-0.25,0.06,-0.17,0.03,0.39,-0.03,-0.17,0.32,0.27,-0.02,0.04,-0.24,0.0,0.25,0.14,0.04,0.17,0.25,0.12,0.08,-0.45,0.12,0.34,0.33,0.13,0.25,0.23,0.28,-0.38,-0.07,-0.21,0.25,0.29,-0.21,-0.06,0.22,-0.18,-0.16,-0.43,0.06,-0.29,-0.27,-0.18,-0.18,-0.29,-0.12,-0.38,-0.19,0.0,-0.28,0.1,0.08,-0.11,0.3,0.15,0.16,0.2,0.1,0.09,-0.01,0.19,-0.06,0.0,0.0,-0.04,0.04,-0.05,-0.08,0.07,-0.12,0.1,-0.03,-0.11,-0.05,-0.27,0.1,0.31,-0.13,-0.15,0.08,0.02,0.0,0.15,0.1,0.1,-0.04,0.06,0.22,0.26,0.25,0.32,-0.04,-0.01,-0.12,-0.3,-0.3,-0.09,0.1,-0.2,0.27,0.04,0.14,0.24,-0.07,-0.24,0.19,0.06,0.2,-0.04,0.03,0.19,0.16,-0.07,-0.09,0.17,0.21,-0.05,0.02,0.01,-0.05,-0.09,-0.15,0.2,0.06,0.14,0.25,0.1,0.39,0.04,0.08,0.04,-0.03,0.22,0.25,0.11,-0.18,-0.14,0.07,0.24,0.21,0.03,-0.16,0.36,-0.05,-0.13,-0.13,-0.1,-0.08,-0.27,-0.07,-0.31,0.12,-0.04,-0.22,-0.07,-0.23,-0.2,0.04,-0.3,-0.29,0.06,-0.13,-0.05,0.32,0.02,-0.18,0.25,0.13,-0.16,-0.17,0.24,-0.04,-0.14,0.03,0.04,-0.13,0.04,0.25,0.17,-0.05,-0.01,0.56,0.44,0.18,0.13,0.0,0.09,0.27,0.14,0.01,0.0,0.06,-0.07,0.18,0.21,0.42,0.0,0.12,-0.1,0.21,0.51,0.26,-0.07,0.13,0.04,0.01,0.11,-0.28,0.2,0.03,-0.19,0.16,-0.11,-0.26,0.01,0.4,0.1,-0.19,0.08,-0.24,-0.02,0.09,-0.27,-0.34,0.1,-0.32,0.06,0.09,0.03,-0.35,-0.11,-0.16,-0.54,0.08,-0.29,-0.05,-0.19,-0.11,0.05,-0.33,-0.26,0.17,-0.04,-0.07,-0.23,-0.32,0.09,-0.29,-0.27,-0.57,-0.61,-0.13,-0.39,-0.25,-0.58,-0.48,-0.46,-0.7,-0.28,-0.32,-0.35,-0.15,-0.05,-0.44,-0.47,-0.25,-0.28,0.13,-0.53,-0.35,-0.43,-0.34,-0.25,-0.24,-0.51,0.15,-0.03,-0.2,-0.31,0.5,0.14,0.01,0.01,-0.13,-0.01,-0.17,0.03,-0.15,-0.06,0.09,-0.12,0.23,-0.07,-0.17,-0.16,0.06,-0.3,-0.24,-0.13,0.15,-0.25,-0.19,-0.47,-0.24,0.06,-0.24,-0.25,-0.43,-0.26,-0.26,-0.28,-0.18,-0.13,-0.16,-0.01,-0.12,-0.16,0.0,0.16,0.19,0.06,0.02,0.4,0.41,0.24,0.22,0.32,0.04,-0.14,0.31,0.08,0.37,0.28,0.32,0.01,-0.03,0.21,0.06,-0.18,0.06,0.18,0.16,0.25,0.15,0.03,-0.21,0.47,-0.05,0.22,-0.32,0.06,-0.05,0.13,0.17,0.38,0.19,0.15,0.04,-0.11,0.07,0.29,-0.07,-0.19,-0.01,-0.14,-0.26,0.1,-0.11,-0.1,-0.11,0.27,-0.13,0.01,0.0,-0.01,-0.04,-0.33,0.01,-0.37,-0.1,-0.23,-0.41,-0.41,-0.14,-0.26,-0.02,-0.33,-0.25,-0.3,-0.25,-0.12,-0.37,-0.21,-0.65,-0.32,-0.13,-0.17,-0.44,-0.41,-0.33,-0.24,-0.26,-0.36,-0.09,-0.08,0.44,-0.25,0.04,0.04,0.05,-0.28,-0.45,-0.15,-0.18,-0.38,-0.36,-0.25,-0.17,-0.33,-0.05,0.05,-0.27,-0.13,-0.01,-0.11,0.13,0.25,0.06,0.06,0.24,-0.28,0.12,-0.23,-0.23,0.05,-0.11,-0.25,-0.09,-0.17,-0.28,-0.06,0.25,-0.1,-0.18,-0.1,-0.17,-0.1,0.02,-0.11,-0.22,-0.2,-0.3,-0.22,-0.36,0.0,-0.35,-0.16,-0.27,-0.03,-0.41,-0.26,-0.29,-0.11,-0.1,-0.1,-0.21,0.13,-0.45,-0.25,-0.1,-0.3,-0.27,0.0,-0.03,-0.2,-0.05,0.17,-0.32,-0.31,-0.51,-0.19,-0.38,-0.45,-0.39,-0.25,-0.27,-0.37,-0.37,-0.22,-0.22,-0.35,-0.39,-0.23,-0.01,-0.08,-0.18,-0.17,-0.01,-0.1,0.01,-0.28,-0.21,0.15,-0.05,0.1,-0.36,-0.07,0.18,0.04,-0.12,-0.17,-0.16,-0.04,-0.05,-0.19,-0.11,-0.33,-0.14,-0.14,-0.14,-0.3,-0.12,-0.34,-0.26,-0.25,0.08,-0.37,-0.53,-0.21,-0.06,-0.52,-0.44,-0.42,-0.3,-0.32,-0.46,-0.07,-0.35,-0.58,-0.25,0.21,-0.05,-0.44,-0.39,-0.27,-0.32,-0.11,-0.1,0.14,-0.25,0.04,-0.04,0.34,-0.25,-0.22,-0.12,-0.13,-0.03,-0.34,-0.05,0.2,-0.19,0.0,0.11,-0.01,-0.25,-0.09,-0.04,-0.15,-0.27,-0.29,-0.09,-0.47,-0.14,-0.39,-0.28,-0.26,-0.43,-0.03,-0.62,-0.32,-0.23,-0.47,-0.59,-0.37,-0.51,-0.21,-0.47,-0.41,-0.32,-0.17,-0.09,-0.03,-0.23,0.03,-0.27,-0.51,-0.2,-0.31,-0.26,-0.32,-0.3,-0.17,-0.53,-0.3,-0.12,-0.03,-0.08,-0.27,-0.4,-0.25,-0.43,0.03,-0.09,-0.23,-0.04,-0.03,0.19,-0.19,-0.16,-0.04,0.36,0.04,0.04,0.04,-0.14,0.1,-0.03,0.19,0.01,-0.01,0.06,-0.04,-0.29,-0.05,0.04,-0.13,-0.11,-0.2,0.32,-0.07,-0.24,-0.04,0.03,0.14,0.06,-0.1,-0.19,0.01,0.13,0.17,-0.24,0.13,-0.19,-0.02,-0.12,-0.01,-0.06,0.13,0.14,0.22,-0.29,-0.26,0.13,-0.14,-0.24,-0.26,-0.3,-0.14,-0.12,0.0,0.04,0.11,-0.11,0.0,0.02,0.19,-0.13,0.1,0.23,0.02,-0.2,-0.18,-0.2,-0.31,-0.26,-0.14,-0.43,-0.27,-0.32,-0.31,-0.04,-0.22,-0.39,-0.3,-0.16,-0.28,-0.38,-0.23,-0.16,-0.37,-0.15,-0.31,-0.07,-0.19,-0.2,-0.11,-0.22,-0.21,-0.22,-0.2,-0.22,-0.25,-0.09,-0.11,-0.36,-0.23,0.5,-0.27,0.04,-0.27,-0.36,-0.17,-0.13,0.06,0.29,0.13,0.4,0.01,-0.1,-0.09,0.16,-0.01,-0.13,0.03,0.04,0.06,-0.32,-0.07,-0.08,0.23,0.1,-0.03,0.09,-0.04,-0.24,0.16,0.09,0.0,-0.18,-0.12,-0.2,-0.26,-0.22,-0.07,-0.03,0.04,-0.01,-0.34,-0.1,0.11,0.13,-0.19,-0.32,0.1,0.05,-0.06,-0.04,-0.37,0.0,0.0,-0.17,-0.37,0.03,0.04,-0.2,-0.26,0.18,-0.09,-0.32,-0.28,0.05,-0.21,-0.41,-0.36,-0.33,-0.29,-0.37,0.01,-0.05,-0.31,-0.19,-0.06,0.06,-0.1,0.16,0.17,0.4,0.19,0.2,0.08,0.16,-0.02,0.04,0.1,0.14,0.18,0.04,0.06,0.08,0.38,0.24,0.19,-0.06,0.16,0.07,-0.04,-0.13,0.03,-0.17,-0.09,-0.24,0.1,-0.05,-0.01,0.01,-0.08,-0.12,-0.24,-0.01,-0.14,-0.27,-0.12,-0.17,-0.28,-0.01,-0.16,0.14,-0.08,-0.03,-0.04,-0.23,0.24,-0.18,-0.09,-0.11,-0.12,0.39,-0.32,-0.13,-0.02,0.01,0.04,0.1,-0.06,-0.05,-0.19,-0.27,-0.14,-0.15,-0.21,-0.21,-0.09,-0.12,-0.04,-0.23,-0.22,-0.39,-0.1,-0.27,-0.06,-0.2,-0.21,0.09,-0.07,-0.19,0.03,-0.27,-0.36,-0.08,-0.25,-0.25,-0.22,-0.11,0.07,-1.9,0.08,0.03,-0.14,-0.29,-0.12,-0.16,-0.19,0.03,0.04,-0.3,-0.35,0.14,-0.53,-0.11,-0.24,-0.19,0.06,0.36,-0.29,-0.27,-0.04,-0.19,-0.29,-0.27,-0.09,-0.19,-0.09,-0.34,-0.22,-0.35,0.06,-0.18,-0.09,0.08,-0.07,-0.15,-0.22,-0.02,-0.07,-0.33,0.34,-0.06,-0.11,0.09,0.04,0.13,0.57,0.27,0.67,0.45,0.19,0.41,0.36,0.63,0.35,0.28,0.52,0.47,0.31,0.35,0.52,0.12,0.02,0.38,0.25,0.24,0.22,0.73,0.38,0.52,0.75,0.05,0.18,0.0,0.17,0.21,0.48,0.21,0.33,0.28,0.13,0.16,0.18,0.18,0.04,0.14,0.42,0.12,0.09,0.2,0.39,0.01,-0.14,0.16,0.51,0.02,-0.11,-0.06,-0.22,-0.04,0.15,-0.04,-0.15,0.01,0.18,0.03,-0.03,0.01,-0.24,-0.12,-0.08,-0.3,-0.14,-0.2,0.14,-0.36,-0.07,-0.17,-0.22,-0.19,-0.19,-0.23,0.19,-0.2,-0.31,-0.24,-0.01,-0.27,-0.18,-0.35,0.22,-0.18,-0.14,-0.26,-0.08,-0.11,-0.05,-0.07,-0.03,0.09,0.1,0.07,0.44,0.15,0.17,0.04,0.06,0.04,0.21,0.23,-0.02,0.54,0.13,0.13,0.1,0.43,0.04,0.24,0.25,-0.08,-0.02,-0.09,0.2,0.32,-0.18,0.09,0.29,0.36,0.01,0.03,0.31,0.05,0.21,0.48,0.09,0.24,0.13,0.34,-0.09,0.09,0.1,0.33,-0.04,0.38,0.33,0.24,0.0,-0.07,0.08,0.0,0.13,0.05,-0.19,-0.1,0.13,0.12,0.39,-0.05,0.21,-0.03,0.04,0.08,-0.2,-0.09,0.0,0.02,0.0,-0.01,-0.16,-0.06,0.38,0.03,0.36,-0.02,0.22,0.26,0.09,0.09,0.06,0.57,0.37,-0.05,0.17,0.01,0.05,0.24,-0.12,-0.16,0.31,-0.03,0.02,0.23,0.04,0.12,0.19,0.23,0.09,0.1,-0.09,-0.03,0.19,-0.02,-0.09,0.15,0.07,0.0,0.12,0.24,-0.24,-0.24,-0.21,-0.12,-0.28,-0.01,-0.12,0.12,-0.02,-0.29,0.34,0.08,0.07,0.1,0.07,-0.21,-0.26,-0.1,0.08,-0.16,-0.22,-0.29,-0.34,-0.23,-0.16,-0.13,-0.14,0.07,-0.1,-0.18,-0.17,-0.22,-0.27,-0.07,-0.1,-0.02,-0.42,-0.2,-0.14,-0.17,0.06,-0.22,0.01,0.27,-0.04,-0.1,0.1,-0.14,-0.03,0.35,0.01,0.06,-0.01,0.27,0.18,0.11,0.39,0.09,-0.18,0.0,0.45,0.21,0.26,-0.09,-0.03,0.06,0.21,0.04,0.23,-0.08,0.29,0.39,0.37,0.21,0.34,0.44,0.1,0.29,-0.21,0.41,0.41,0.57,0.0,0.12,-0.03,0.18,0.18,0.21,0.11,0.04,0.01,-0.15,0.2,0.0,0.0,0.34,0.42,0.02,-0.09,0.35,0.03,0.47,0.06,0.34,0.52,0.4,0.24,0.18,0.25,0.19,0.2,0.24,0.29,0.29,0.0,0.27,0.37,0.12,-0.03,0.53,0.09,-0.12,0.04,0.08,0.17,-0.1,0.09,0.04,0.22,-0.3,-0.35,-0.1,-0.25,-0.12,-0.1,-0.03,-0.04,-0.19,-0.09,-0.02,-0.18,0.06,-0.19,0.13,0.06,0.16,-0.25,-0.03,-0.12,-0.22,-0.2,0.03,-0.17,0.17,-0.28,0.14,0.09,0.03,-0.19,0.32,-0.26,-0.26,-0.25,-0.03,-0.09,-0.18,-0.17,0.06,-0.05,-0.17,0.04,-0.08,-0.01,-0.09,0.08,-0.19,0.06,0.24,-0.09,0.35,-0.09,0.31,0.2,0.24,-0.02,0.09,-0.2,0.12,-0.01,0.03,0.04,0.04,-0.19,-0.06,-0.14,-0.1,-0.24,-0.09,0.1,0.18,0.36,-0.05,0.05,0.25,0.15,0.37,0.19,0.02,0.16,0.22,0.06,-0.25,-0.31,-0.2,-0.31,-0.12,-0.4,-0.42,-0.15,-0.25,-0.63,-0.19,-0.23,-0.18,-0.12,0.2,0.11,0.2,-0.21,-0.14,-0.23,-0.28,0.1,0.1,-0.07,-0.01,-0.06,0.21,0.12,0.15,0.25,-0.04,-0.03,-0.01,-0.09,0.22,0.3,0.22,0.15,-0.01,0.15,-0.05,-0.21,-0.29,0.15,-0.31,0.15,0.26,0.13,-0.09,-0.18,-0.06,0.08,0.29,-0.59,-0.22,0.08,-0.35,-0.03,-0.15,-0.22,-0.32,0.01,-0.3,-0.27,-0.08,-0.58,-0.22,-0.04,0.25,-0.04,-0.33,-0.12,-0.07,-0.26,0.08,-0.19,-0.18,-0.59,-0.57,0.14,-0.3,-0.53,-0.31,-0.12,-0.22,-0.5,-0.1,-0.44,-0.06,-0.13,0.0,-0.13,-0.21,0.01,-0.21,-0.01,0.02,-0.15,-0.2,0.09,-0.06,-0.06,-0.07,0.07,-0.17,-0.04,-0.05,-0.1,0.15,0.1,0.08,0.2,-0.26,0.13,-0.12,-0.08,-0.02,0.12,-0.1,-0.11,0.17,0.31,0.28,-0.09,0.23,0.25,0.1,0.28,0.35,0.2,0.19,0.45,0.34,0.16,0.21,0.31,0.57,0.38,0.4,0.75,0.32,0.41,0.41,0.22,0.52,0.25,0.48,0.15,0.13,0.08,0.2,0.16,0.06,0.26,0.35,0.03,0.11,0.04,0.27,0.41,0.39,0.25,0.17,0.62,0.2,0.2,0.08,-0.08,0.06,-0.13,-0.09,-0.08,0.23,-0.14,-0.02,0.19,0.41,0.33,0.23,0.06,-0.25,0.22,0.19,0.22,0.2,0.23,0.17,-0.09,0.14,0.25,-0.05,0.28,0.39,0.41,-0.04,0.13,-0.35,-0.33,-0.26,-0.17,-0.07,-0.23,-0.12,-0.13,-0.04,-0.27,-0.02,-0.29,-0.36,-0.22,0.2,-0.27,-0.47,-0.46,0.04,-0.28,-0.12,-0.56,-0.28,-0.29,-0.26,-0.03,-0.42,-0.36,-0.36,0.0,-0.09,0.04,-0.01,-0.18,-0.13,-0.01,0.04,-0.11,-0.19,-0.05,-0.13,0.05,0.18,0.51,0.21,0.2,0.11,-0.01,0.12,-0.04,0.29,-0.17,0.22,-0.04,-0.06,-0.14,0.2,-0.14,0.04,0.03,0.18,0.08,0.44,0.03,0.24,0.22,0.04,0.21,0.18,-0.14,-0.06,0.27,-0.05,0.22,0.09,0.13,0.1,0.15,0.04,0.14,0.29,-0.15,0.18,-0.1,0.06,0.2,0.06,0.4,-0.15,0.06,0.29,0.15,0.45,0.08,0.34,0.23,0.24,0.33,0.19,0.27,0.66,0.4,0.33,0.57,0.1,0.07,0.14,0.29,0.02,0.08,0.2,0.31,0.05,-0.13,-0.18,-0.11,-0.22,-0.11,-0.26,0.06,0.4,-0.07,0.35,0.14,0.23,0.37,0.04,0.22,0.31,0.03,-0.08,-0.1,-0.03,-0.23,0.02,0.13,-0.07,0.22,0.03,-0.18,-0.36,-0.2,-0.38,-0.37,-0.16,-0.4,-0.34,-0.37,-0.55,-0.33,-0.06,-0.09,-0.43,-0.16,-0.17,-0.25,-0.31,-0.5,-0.33,-0.25,-0.31,-0.01,0.16,0.05,-0.18,-0.39,-0.24,0.03,-0.24,0.08,0.05,-0.02,-0.46,-0.27,-0.06,-0.42,0.04,-0.32,-0.26,0.36,-0.15,-0.24,-0.32,-0.19,0.09,0.03,-0.23,-0.05,-0.04,-0.14,0.02,0.14,-0.11,-0.27,-0.36,0.01,0.33,0.03,-0.1,-0.04,-0.15,0.35,0.13,0.24,0.46,0.09,0.03,-0.02,-0.09,0.33,-0.03,-0.11,-0.09,0.03,-0.05,-0.21,0.0,-0.2,0.05,0.14,-0.14,-0.03,-0.21,-0.06,-0.15,-0.15,0.04,0.0,0.03,-0.01,0.21,-0.18,-0.04,0.28,0.21,0.02,-0.06,-0.04,0.02,0.03,-0.11,0.43,0.0,-0.04,-0.17,0.03,0.01,0.01,-0.2,-0.33,0.1,0.17,0.16,0.08,0.05,0.2,0.15,0.41,0.17,0.14,0.42,0.37,0.14,0.43,0.15,0.3,0.45,0.32,0.47,0.19,0.25,0.04,-0.16,-0.02,-0.33,0.21,0.41,-0.2,0.13,-0.29,-0.03,-0.02,-0.02,-0.43,-0.14,-0.37,-0.34,-0.23,-0.66,-0.11,-0.24,0.12,-0.24,-0.1,-0.39,-0.14,0.22,0.02,-0.33,-0.13,-0.21,0.04,-0.07,0.0,0.04,0.27,0.03,-0.11,0.06,0.04,0.59,-0.17,-0.07,0.06,0.34,0.13,-0.01,0.14,-0.01,0.22,-0.19,0.04,-0.01,0.14,0.13,-0.04,-0.24,-0.1,-0.18,-0.28,0.0,-0.45,-0.49,0.03,-0.28,-0.44,-0.29,0.05,-0.26,-0.37,-0.51,-0.53,-0.35,-0.35,-0.2,-0.11,-0.14,0.27,0.08,-0.14,0.0,0.1,0.16,0.2,0.06,0.1,-0.01,-0.3,0.11,0.07,0.46,0.0,-0.19,-0.19,-0.2,-0.11,-0.44,-0.32,-0.09,0.07,-0.17,-0.18,-0.31,0.15,0.03,-0.14,-0.33,-0.25,-0.11,-0.4,-0.36,-0.27,-0.39,-0.18,-0.06,-0.13,-0.16,0.1,-0.05,-0.53,-0.26,0.04,0.0,0.04,0.17,0.04,0.08,0.17,0.14,-0.06,0.18,0.18,0.08,-0.08,0.11,-0.13,-0.16,0.01,-0.01,-0.03,0.0,0.17,0.21,-0.13,0.18,0.05,-0.17,0.03,0.11,-0.19,0.02,-0.05,-0.19,-0.06,0.09,-0.27,-0.23,0.0,-0.15,0.08,-0.3,0.04,-0.38,-0.2,-0.2,-0.36,-0.18,-0.2,0.28,0.06,-0.34,-0.45,-0.23,0.14,-0.13,-0.37,-0.25,-0.34,0.1,0.0,-0.1,-0.16,-0.08,-0.06,-0.09,-0.05,-0.13,0.44,0.06,-0.05,-0.17,-0.07,0.19,-0.11,-0.03,-0.21,-0.23,0.18,-0.2,-0.29,-0.03,-0.04,-0.33,-0.16,-0.17,-0.08,-0.35,-0.23,-0.07,-0.11,-0.03,-0.29,-0.01,-0.28,-0.16,-0.03,0.02,-0.05,-0.22,-0.19,-0.03,-0.05,-0.12,-0.37,-0.44,-0.37,-0.18,-0.25,-0.31,-0.2,-0.26,-0.12,-0.32,-0.28,-0.53,-0.54,-0.33,-0.22,-0.27,-0.25,-0.36,0.17,-0.13,-0.05,-0.04,-0.15,0.24,-0.27,-0.13,0.04,-0.24,-0.14,-0.2,-0.49,-0.29,-0.05,-0.3,-0.06,-0.18,0.03,-0.33,-0.18,-0.35,-0.37,0.09,0.01,0.18,-0.03,0.07,0.05,-0.08,0.04,0.1,-0.31,-0.16,-0.26,-0.41,-0.29,0.16,-0.21,-0.22,-0.06,-0.33,-0.23,-0.33,-0.23,-0.34,0.04,0.1,-0.18,-0.25,-0.15,-0.21,0.16,-0.12,-0.19,0.05,-0.39,-0.23,-0.26,-0.23,-0.24,-0.31,-0.07,0.34,-0.11,-0.28,-0.19,-0.4,-0.43,-0.02,-0.07,-0.23,-0.19,0.09,0.0,-0.18,-0.22,-0.05,-0.17,-0.22,0.01,0.06,-0.05,0.0,-0.03,0.0,-0.32,-0.34,0.06,0.12,-0.25,0.3,-0.33,0.25,0.12,0.24,0.13,0.15,0.12,0.07,0.03,0.0,0.42,0.47,0.06,0.2,-0.03,0.3,0.22,0.08,-0.08,-0.01,0.09,0.01,0.11,0.12,0.07,0.03,0.21,0.18,-0.24,0.2,-0.07,-0.02,0.1,0.18,0.2,-0.03,-0.03,-0.04,-0.03,-0.01,0.35,0.51,0.15,0.17,0.11,0.41,0.17,0.01,0.06,-0.02,0.16,0.29,0.1,0.05,-0.03,-0.25,-0.25,0.02,-0.06,0.26,-0.11,-0.1,-0.03,-0.35,-0.16,-0.21,-0.31,-0.43,-0.17,-0.32,-0.11,-0.34,-0.25,-0.27,0.06,-0.27,0.01,-0.01,-0.01,-0.25,0.1,-0.07,-0.24,-0.11,-0.17,-0.32,0.01,-0.2,-0.3,-0.05,-0.08,-0.01,-0.11,-0.04,-0.37,-0.32,-0.38,-0.41,0.1,-0.03,-0.25,-0.12,0.02,-0.25,-0.59,0.17,-0.3,-0.23,-0.37,0.0,-0.18,-0.11,-0.49,-0.02,-0.33,-0.23,-0.13,-0.24,-0.29,0.01,-0.27,-0.06,-0.05,-0.17,0.01,0.3,-0.21,-0.26,-0.54,-0.12,-0.27,-0.12,-0.03,-0.2,-0.02,-0.1,-0.24,-0.07,-0.17,-0.17,-0.3,-0.13,-0.16,-0.09,-0.11,-0.16,-0.06,-0.33,-0.23,0.22,-0.02,-0.23,0.09,0.04,0.04,0.23,-0.24,-0.03,0.09,0.45,0.1,0.15,0.08,0.04,0.16,-0.02,-0.03,0.4,0.25,0.03,0.3,0.36,-0.08,0.06,-0.01,0.14,-0.3,-0.17,-0.36,-0.04,-0.1,-0.19,-0.09,-0.15,0.06,-0.01,-0.2,-0.12,-0.21,0.16,-0.09,-0.02,-0.06,-0.03,0.06,0.04,-0.04,-0.05,0.08,-0.03,-0.3,-0.08,-0.19,0.08,-0.46,-0.1,-0.35,-0.01,-0.2,-0.43,-0.28,-0.15,-0.28,-0.19,0.03,-0.03,-0.06,0.06,-0.2,-0.22,-0.38,-0.32,-0.47,-0.46,0.05,-0.31,0.09,-0.01,-0.03,0.17,-0.25,-0.05,0.02,0.06,0.21,-0.04,0.39,0.21,-0.17,0.16,0.19,0.23,0.07,0.31,0.1,0.19,0.28,0.17,-0.03,0.06,0.25,0.18,0.17,0.69,0.16,0.37,0.45,0.52,0.24,0.53,0.4,0.37,0.81,0.55,0.57,0.22,0.29,0.7,0.77,0.79,0.57,0.17,0.1,0.27,0.58,0.63,0.35,0.66,0.55,0.62,0.28,0.62,0.32,0.45,0.41,0.54,0.15,0.26,0.26,0.29,0.19,0.23,0.1,0.08,0.45,0.3,0.09,0.36,0.01,-0.12,0.22,0.27,0.3,0.16,-0.14,0.14,0.08,-0.19,0.16,0.1,-0.03,-0.2,-0.22,0.0,0.03,-0.07,0.04,-0.21,-0.07,-0.26,0.15,0.29,-0.25,-0.03,-0.18,-0.21,-0.16,0.25,0.1,-0.12,0.07,0.06,-0.01,0.08,0.11,0.09,0.19,0.6,0.14,0.1,0.05,0.04,0.37,0.21,-0.06,0.08,0.14,0.24,0.17,0.06,0.12,0.32,0.09,0.02,-0.11,0.02,-0.15,-0.01,0.15,0.14,0.03,-0.09,0.27,-0.08,0.07,-0.06,0.04,0.2,-0.11,0.18,-0.19,0.18,-0.22,0.15,-0.13,0.05,-0.25,-0.13,-0.29,0.03,-0.5,0.01,-0.31,-0.37,-0.12,-0.06,0.0,0.08,-0.43,-0.24,0.14,-0.31,-0.17,0.09,0.06,-0.1,-0.45,-0.39,-0.05,-0.21,-0.2,-0.28,-0.09,0.23,0.04,0.04,0.01,-0.34,-0.09,-0.02,-0.09,0.06,0.15,0.04,-0.17,-0.01,-0.21,-0.16,0.06,-0.29,-0.32,-0.22,-0.16,-0.37,0.33,-0.11,-0.07,-0.1,0.13,0.0,0.25,-0.07,0.12,-0.05,0.3,0.2,0.01,0.0,-0.03,-0.11,-0.19,-0.04,0.35,0.29,0.03,0.13,0.1,-0.05,0.06,0.04,0.3,-0.16,0.01,0.14,0.24,-0.06,0.21,0.17,0.21,0.19,-0.07,0.24,0.12,0.13,0.04,-0.21,0.25,-0.12,0.07,-0.03,0.19,-0.04,-0.21,0.22,0.15,0.14,0.12,0.01,0.0,0.07,0.25,0.1,0.17,0.02,0.07,-0.13,0.1,-0.17,-0.28,0.08,0.16,-0.07,-0.27,-0.37,-0.03,0.04,-0.31,-0.11,0.1,-0.17,-0.26,0.16,-0.03,-0.19,-0.01,0.1,0.1,-0.18,-0.22,-0.24,-0.05,-0.21,-0.38,-0.09,0.15,-0.21,-0.25,-0.09,-0.01,0.03,-0.24,-0.14,0.06,-0.07,-0.05,0.13,-0.21,-0.16,0.09,0.03,-0.08,-0.14,-0.46,-0.28,-0.32,-0.36,-0.31,-0.3,-0.45,-0.39,-0.35,-0.16,-0.27,-0.21,-0.07,0.18,-0.19,-0.19,0.03,-0.19,0.28,0.35,-0.02,-0.06,0.13,0.1,0.34,0.38,0.22,0.08,0.37,0.27,0.3,0.13,0.38,0.5,0.1,0.09,0.2,0.46,-0.07,-0.06,0.03,0.18,0.06,-0.14,-0.28,0.25,0.03,-0.08,-0.06,0.1,-0.18,-0.23,-0.06,0.18,-0.03,-0.11,-0.02,-0.04,0.26,0.01,0.26,0.06,0.17,-0.2,0.12,0.04,0.09,-0.19,0.12,0.23,-0.19,-0.21,0.07,0.35,0.28,0.14,-0.2,0.17,-0.09,-0.05,-0.2,0.1,0.36,0.22,0.23,-0.09,0.12,0.04,0.19,-0.11,0.04,0.27,0.06,0.13,0.17,0.36,0.16,0.16,0.4,0.03,0.15,0.2,0.21,0.2,0.24,0.22,-0.06,0.12,0.32,0.19,0.21,0.86,0.53,0.52,0.39,0.55,0.35,0.4,0.31,0.14,0.14,0.22,0.52,0.45,-0.18,0.07,-0.05,-0.09,-0.04,0.1,0.0,0.14,-0.25,-0.06,0.04,-0.05,-0.1,-0.06,-0.1,-0.06,-0.09,-0.18,-0.09,0.22,-0.08,0.02,0.1,0.15,-0.13,0.17,-0.15,0.01,0.23,-0.22,-0.19,-0.17,-0.27,0.05,-0.22,-0.01,-0.07,0.08,0.13,0.08,-0.07,0.17,0.06,-0.05,0.12,0.26,0.02,0.04,0.03,0.01,0.04,-0.19,-0.03,-0.14,-0.03,-0.58,-0.38,0.0,-0.61,-0.16,-0.53,-0.25,-0.58,-0.34,0.08,-0.39,-0.08,-0.13,-0.17,-0.38,-0.21,-0.23,-0.16,-0.13,-0.09,0.28,0.19,-0.33,-0.21,-0.09,-0.18,-0.14,0.0,-0.08,-0.03,-0.34,-0.16,-0.07,0.07,-0.1,0.26,-0.12,-0.06,-0.19,0.17,-0.09,-0.16,-0.11,-0.01,-0.02,0.01,0.09,-0.05,0.03,-0.16,0.23,-0.26,-0.22,-0.03,-0.04,-0.15,-0.25,-0.08,0.16,0.11,-0.17,-0.23,0.1,0.12,0.28,0.08,0.16,0.15,-0.25,-0.01,0.25,0.1,0.16,0.09,0.22,0.06,0.13,-0.14,0.09,0.1,0.02,0.2,0.39,0.38,0.08,0.15,0.06,0.08,0.18,-0.18,-0.03,-0.21,-0.01,-0.09,-0.14,-0.05,0.1,0.41,-0.14,0.16,0.33,-0.24,-0.12,0.13,-0.03,0.17,-0.03,-0.08,0.06,-0.16,0.47,0.04,0.11,0.22,0.12,0.17,0.18,0.3,0.29,0.24,0.19,0.36,0.4,0.5,0.58,0.16,0.15,0.44,0.12,0.37,0.11,-0.04,0.52,0.15,0.17,0.0,0.15,-0.18,0.28,-0.09,0.32,0.04,-0.2,0.15,-0.06,0.21,-0.12,0.42,-0.05,0.18,0.36,-0.03,0.08,0.31,-0.07,0.12,-0.17,0.1,0.13,0.18,-0.03,0.23,0.09,-0.02,0.25,0.01,0.4,0.14,0.1,0.16,0.16,0.09,0.01,0.26,0.33,0.37,0.3,0.52,0.16,0.68,0.45,0.53,0.78,0.5,0.41,0.17,0.72,0.6,0.3,0.38,0.5,0.36,0.31,0.61,0.57,0.55,0.48,0.44,0.49,0.52,0.66,0.7,0.7,0.65,0.75,0.73,0.89,0.63,0.34,0.22,0.61,0.53,0.54,0.41,0.33,0.64,0.34,0.38,0.51,0.8,0.35,0.72,0.55,0.47,0.38,0.38,0.54,0.47,0.27,0.31,0.47,0.57,0.31,0.35,0.53,0.4,0.35,0.25,0.43,0.17,0.24,0.24,0.27,0.24,0.04,0.4,0.63,0.57,0.21,0.58,0.71,0.47,0.55,0.41,0.66,0.49,0.63,0.59,0.58,0.61,0.59,0.74,0.75,0.77,0.63,0.7,0.52,0.54,0.71,0.41,0.5,0.78,0.41,0.75,0.54,0.66,1.02,0.78,0.7,0.61,0.49,0.7,0.77,0.29,0.47,0.74,0.87,0.57,0.4,0.23,0.34,0.61,0.51,0.35,0.52,0.5,0.23,0.22,0.6,0.73,0.28,0.32,0.28,0.66,0.1,0.49,0.39,0.35,0.11,0.35,0.33,0.52,0.48,0.45,0.17,0.21,0.22,0.55,0.13,0.27,0.17,0.15,0.43,0.28,0.25,0.08,-0.1,0.31,-0.11,0.19,0.14,0.13,0.06,0.09,0.18,0.22,0.03,0.27,-0.13,0.26,0.29,0.24,0.1,0.24,0.37,0.23,0.36,0.31,0.24,0.38,0.41,0.25,0.08,0.23,0.34,0.21,-0.01,0.13,-0.14,0.22,0.34,-0.12,0.03,-0.16,-0.18,-0.1,-0.04,-0.15,-0.33,0.07,-0.01,-0.2,-0.14,0.06,0.1,0.14,0.3,0.19,0.31,0.15,0.26,-0.07,0.09,0.33,0.23,0.25,0.47,0.22,0.45,0.18,0.42,0.27,0.3,0.34,0.2,0.12,0.24,0.25,0.64,0.19,0.05,0.39,0.15,0.36,0.15,0.39,0.22,0.3,0.26,0.48,0.3,0.64,0.44,0.67,0.53,0.45,0.09,0.33,0.4,0.24,0.07,0.15,0.22,0.16,0.41,0.13,0.09,0.33,0.01,0.17,0.03,0.19,0.44,0.3,-0.03,0.15,0.25,-0.09,0.44,0.18,0.1,0.48,0.13,0.38,0.13,0.06,0.13,0.16,0.02,0.3,0.06,0.03,0.01,0.35,0.45,-0.03,0.06,0.1,-0.1,0.0,-0.22,0.1,0.1,0.25,0.14,-0.23,-0.17,0.15,-0.15,-0.26,-0.25,-0.1,-0.07,-0.2,-0.24,-0.18,0.0,0.14,-0.16,-0.33,-0.09,0.24,-0.09,-0.11,-0.33,-0.04,-0.22,-0.26,-0.53,-0.15,-0.22,-0.24,-0.2,-0.54,-0.36,-0.37,-0.14,-0.72,0.02,-0.49,-0.43,-0.42,-0.58,-0.16,0.06,-0.04,-0.03,-0.12,-0.3,-0.37,-0.05,-0.23,-0.36,-0.07,-0.52,-0.34,-0.29,-0.32,-0.27,-0.53,-0.25,-0.61,-0.4,-0.35,-0.55,-0.07,-0.25,-0.28,-0.03,-0.29,-0.19,-0.31,-0.29,-0.13,0.17,-0.09,0.19,-0.23,0.22,-0.02,0.04,-0.19,0.28,0.01,-0.02,0.08,-0.31,-0.24,-0.28,-0.34,-0.19,-0.26,-0.27,-0.6,-0.13,-0.34,-0.42,-0.7,-0.24,-0.52,-0.46,-0.25,-0.48,-0.25,-0.23,-0.29,-0.48,-0.26,-0.36,-0.1,-0.26,-0.1,-0.37,-0.11,0.06,0.02,-0.31,-0.17,-0.68,-0.08,-0.08,-0.16,-0.4,-0.35,-0.52,-0.66,-0.36,-0.39,-0.48,-0.42,-0.16,-0.39,-0.27,-0.21,-0.28,-0.21,-0.32,-0.36,-0.23,-0.33,0.04,-0.34,-0.09,-0.21,-0.3,0.06,-0.17,-0.17,-0.06,-0.16,-0.21,-0.4,-0.28,-0.09,-0.37,-0.17,-0.35,0.0,-0.26,-0.14,-0.03,0.0,-0.04,0.0,-0.08,-0.16,-0.25,-0.01,0.22,0.14,0.01,-0.04,-0.01,0.17,0.06,0.06,0.45,0.19,0.06,-0.15,0.19,0.18,0.01,-0.03,0.18,0.05,0.32,-0.06,-0.13,0.01,0.18,-0.16,0.01,-0.08,0.16,0.04,-0.12,-0.01,0.12,0.14,-0.01,-0.13,-0.1,0.12,-0.06,-0.07,0.13,-0.2,-0.04,0.27,-0.05,0.16,-0.31,-0.19,-0.25,0.2,0.22,0.38,0.0,0.06,-0.06,0.0,0.24,0.14,0.1,0.28,0.12,0.12,0.33,0.48,0.08,0.27,0.22,0.04,0.32,0.08,0.14,0.1,0.26,0.01,-0.07,0.06,0.37,0.31,0.31,0.23,0.01,0.37,0.35,0.4,0.14,0.54,0.29,0.28,-0.02,0.16,0.06,-0.37,0.03,0.3,0.09,0.32,0.14,0.56,0.21,0.21,0.31,0.24,0.48,0.43,0.43,0.4,0.67,0.28,0.59,0.51,0.51,0.55,0.71,0.25,0.33,0.3,0.18,0.36,0.71,0.39,0.56,0.41,0.55,0.52,0.47,0.61,0.48,0.52,0.49,0.47,0.66,0.5,0.59,0.37,0.72,0.49,0.31,0.51,0.55,0.62,0.71,0.45,0.37,0.41,0.41,0.66,0.18,0.63,0.34,0.55,0.27,0.45,0.38,0.21,0.12,0.48,0.56,0.58,0.43,0.81,0.37,0.58,0.72,0.9,0.33,0.79,0.66,0.55,0.8,0.62,0.38,0.54,0.49,0.48,0.63,0.45,0.73,1.02,0.82,0.58,0.58,0.42,0.58,0.52,0.52,0.64,0.61,0.58,0.29,0.63,0.63,0.61,0.64,0.67,0.89,0.56,0.64,0.66,0.63,0.52,0.57,0.39,0.51,0.74,0.64,0.34,0.63,0.36,0.97,0.46,0.52,0.76,0.36,0.51,0.61,0.09,0.34,0.13,0.23,0.29,0.21,0.04,0.54,0.1,0.04,0.06,0.27,0.19,0.34,0.31,0.04,0.36,0.04,0.03,0.09,-0.06,0.18,0.14,0.45,0.16,0.1,0.35,0.0,0.36,-0.04,0.03,0.18,0.15,0.43,-0.25,-0.16,0.18,0.06,0.14,0.09,-0.05,0.24,0.22,0.01,0.23,0.3,0.25,0.14,0.43,0.06,0.09,0.38,0.39,0.21,0.02,0.24,0.26,0.39,0.36,0.23,0.02,0.46,0.25,-0.01,0.22,0.37,0.08,0.14,0.18,0.19,0.28,0.56,0.04,0.44,0.15,0.1,0.19,0.09,0.36,0.15,0.27,0.05,0.06,0.21,0.1,0.23,0.48,0.29,0.32,0.11,0.61,0.6,0.28,0.71,0.64,0.83,0.42,0.32,0.24,0.5,0.87,0.49,0.5,0.49,0.46,0.42,0.63,0.4,0.37,0.44,0.39,0.71,0.87,0.38,0.65,0.35,0.53,0.69,0.87,0.52,0.73,0.46,0.56,0.39,0.61,0.36,0.55,0.39,0.63,0.34,0.53,0.25,0.53,0.51,0.82,0.42,0.38,0.39,0.67,0.91,0.51,0.63,0.51,0.33,0.41,0.62,0.12,0.63,0.69,0.58,0.72,0.31,0.47,0.26,0.18,0.23,0.39,0.73,0.48,-0.08,0.51,0.38,0.29,0.53,0.22,0.55,0.4,0.43,0.84,0.57,0.34,0.53,0.5,0.42,0.08,0.05,0.39,0.38,0.14,0.3,-0.02,0.39,0.31,0.1,0.33,0.31,0.2,0.11,0.11,0.38,0.61,0.25,-0.01,0.08,0.17,0.33,0.12,0.41,0.1,0.06,-0.09,0.03,0.43,-0.03,0.04,0.33,0.22,0.15,0.26,-0.17,0.34,0.22,0.2,0.36,0.12,0.4,0.19,0.02,0.4,0.57,0.16,0.5,0.21,0.33,0.22,0.47,0.33,0.61,0.42,0.43,0.52,0.41,0.48,0.39,0.53,0.38,0.25,0.2,0.37,0.36,0.1,0.28,0.45,0.18,0.36,0.27,0.61,0.62,0.37,0.38,0.23,0.46,0.33,0.62,0.86,0.43,0.4,0.48,0.33,0.45,0.16,0.49,0.43,0.52,0.35,0.39,0.95,0.72,0.15,0.31,0.45,0.49,0.48,0.69,0.51,0.29,0.42,0.57,0.38,0.3,0.52,0.56,0.22,0.94,0.84,0.67,0.15,0.39,0.46,0.36,0.39,0.37,0.62,0.13,0.26,0.51,0.31,0.22,0.15,0.21,0.27,0.32,0.39,0.19,0.04,0.13,0.53,0.06,0.21,0.06,0.3,0.21,0.27,0.41,0.34,0.19,0.26,0.13,0.37,0.18,0.34,0.1,0.37,-0.11,0.36,0.13,-0.05,0.24,-0.05,0.04,0.23,-0.11,0.33,0.16,-0.2,0.06,-0.16,0.19,-0.04,0.28,-0.03,0.15,-0.03,-0.04,0.01,0.2,-0.06,-0.28,-0.05,0.1,0.19,0.01,-0.08,-0.12,-0.02,0.32,-0.11,-0.02,-0.08,-0.41,-0.26,0.25,0.07,-0.28,0.17,-0.36,-0.21,-0.17,0.1,-0.32,-0.31,-0.03,-0.1,-0.11,-0.01,0.19,0.26,-0.14,0.24,0.01,0.1,0.54,0.12,0.0,0.31,0.36,0.22,0.15,0.13,0.36,0.07,0.1,0.21,0.04,0.45,0.1,0.09,0.34,0.36,0.58,0.27,0.17,0.15,0.48,0.37,0.07,0.46,0.1,0.32,0.42,0.63,-0.1,0.04,0.06,0.4,0.23,0.28,0.01,-0.03,-0.08,-0.06,0.11,-0.09,0.09,0.41,0.02,0.16,0.22,0.17,0.19,0.14,0.19,0.0,0.02,0.08,0.28,0.12,0.41,0.56,0.44,0.08,0.27,0.29,0.23,0.06,0.29,0.17,0.29,0.24,0.14,0.1,-0.07,0.53,0.1,0.31,0.06,-0.04,0.3,0.34,-0.07,0.07,-0.17,-0.04,0.31,0.2,0.2,-0.03,0.16,-0.07,0.15,0.07,0.0,0.03,0.1,-0.1,0.19,0.05,0.22,-0.03,0.28,0.04,0.47,0.09,-0.03,0.09,-0.03,0.03,0.1,0.11,-0.01,-0.04,0.17,0.06,0.3,0.06,0.19,0.25,0.15,0.3,-0.04,-0.1,0.04,-0.12,-0.01,0.34,-0.03,0.41,-0.11,0.19,0.01,0.05,-0.03,0.13,0.18,0.03,0.56,0.21,0.0,0.04,0.11,0.45,0.31,0.52,0.32,0.42,0.32,0.52,0.44,0.42,0.37,0.28,0.22,0.31,0.24,0.24,0.35,0.21,0.25,0.33,0.46,0.31,0.49,0.2,-0.03,0.56,0.12,0.1,0.12,-0.1,0.39,0.12,0.18,0.44,0.15,0.32,0.1,0.51,0.25,0.66,0.42,0.49,-0.01,0.08,0.3,0.71,0.53,0.18,0.43,0.22,0.16,0.04,0.45,0.51,0.11,-0.01,0.16,0.38,0.35,0.1,0.18,0.14,0.43,0.03,0.05,0.04,-0.09,0.11,0.13,0.23,0.06,-0.19,0.03,0.24,0.38,-0.02,-0.17,0.14,0.09,-0.17,-0.15,-0.23,-0.01,0.18,0.15,0.14,-0.12,0.04,0.08,-0.06,-0.32,-0.31,0.21,-0.08,-0.1,-0.26,-0.11,0.01,-0.04,-0.06,-0.14,-0.01,-0.13,-0.07,0.24,-0.26,0.23,-0.11,0.31,-0.09,0.37,0.33,0.26,0.01,-0.01,0.44,0.21,0.19,0.14,0.03,0.2,0.07,0.4,0.29,0.1,0.38,0.36,0.1,0.39,0.15,0.28,0.16,0.28,0.42,0.25,0.34,0.46,0.61,0.44,0.52,0.68,0.4,0.27,0.68,1.0,0.66,0.23,0.42,0.7,0.49,0.81,0.47,0.53,0.68,0.93,0.52,0.36,0.5,0.45,0.55,0.61,0.42,0.53,0.58,0.65,0.54,0.35,0.37,0.39,0.5,0.78,0.32,0.59,0.33,0.32,0.51,0.08,0.19,-0.04,-0.02,0.21,-0.02,0.04,0.13,0.18,0.03,0.14,-0.01,0.35,0.04,-0.19,0.25,-0.03,0.15,0.1,0.24,0.02,-0.24,-0.22,-0.14,-0.01,0.01,-0.1,0.15,-0.11,0.03,0.2,0.05,-0.05,0.15,0.02,-0.14,-0.02,-0.06,-0.08,0.18,-0.22,-0.11,-0.09,-0.35,0.06,0.02,-0.32,-0.09,-0.26,-0.47,0.02,-0.25,-0.32,-0.24,-0.36,-0.21,-0.05,-0.17,-0.21,-0.11,-0.11,0.0,-0.12,-0.2,-0.09,-0.14,0.08,-0.14,-0.24,-0.11,-0.11,0.26,-0.04,0.16,0.38,0.25,0.39,0.14,0.19,0.35,0.18,0.15,0.28,0.1,-0.03,0.25,0.09,-0.03,0.42,0.33,0.22,0.42,-0.12,-0.1,-0.09,0.07,0.22,-0.11,-0.06,0.14,0.11,0.02,-0.24,0.0,0.04,-0.05,0.16,0.19,0.31,0.13,-0.03,0.19,0.02,0.26,0.33,-0.22,-0.07,-0.25,0.09,0.04,0.0,0.03,0.02,-0.23,0.24,-0.35,-0.29,-0.03,-0.31,0.03,-0.34,-0.01,-0.21,-0.24,-0.35,0.04,-0.3,-0.33,-0.07,-0.22,0.08,0.1,-0.18,-0.34,-0.24,-0.03,-0.06,-0.46,-0.01,0.26,0.09,-0.4,0.36,-0.14,-0.2,0.14,-0.01,-0.22,0.2,-0.12,-0.51,-0.24,-0.12,-0.09,-0.04,-0.03,0.26,0.14,0.38,0.18,0.22,0.26,0.69,0.2,0.22,0.04,0.21,0.39,0.3,0.2,0.12,0.1,0.08,-0.1,0.58,-0.06,0.45,-0.15,0.07,0.04,0.07,-0.01,-0.08,-0.17,-0.15,-0.09,-0.28,-0.34,-0.2,-0.31,0.24,0.43,-0.03,0.47,0.22,-0.14,-0.25,0.04,0.01,-0.03,-0.2,-0.03,0.29,0.15,0.0,0.1,-0.02,0.1,0.18,0.0,0.04,0.13,0.15,0.2,0.19,0.03,-0.15,0.03,0.18,-0.19,0.39,0.31,0.05,0.1,0.27,0.09,0.2,0.19,0.06,0.13,0.26,0.42,0.15,0.21,0.01,0.06,0.09,0.04,0.1,-0.06,0.15,0.15,0.3,0.27,0.03,0.1,0.1,0.46,0.32,0.17,0.14,0.39,0.54,0.3,0.15,0.38,0.43,0.34,0.06,0.17,0.18,0.02,0.5,0.16,0.14,0.26,0.15,0.26,0.37,0.32,0.2,0.06,0.2,0.3,0.09,0.17,-0.01,-0.04,-0.08,0.02,0.08,-0.01,-0.05,-0.09,-0.06,0.05,0.13,-0.01,-0.01,-0.19,-0.17,-0.17,-0.06,0.08,0.14,0.06,-0.38,0.24,0.04,-0.05,0.12,0.17,0.16,0.23,0.01,0.21,0.25,0.19,0.44,0.04,0.0,-0.04,0.06,0.22,0.2,-0.26,0.09,-0.24,-0.06,0.32,-0.04,-0.11,-0.15,-0.08,0.22,0.17,-0.06,0.18,0.19,-0.13,-0.04,0.23,0.04,0.02,0.0,0.06,-0.05,-0.1,-0.03,-0.19,-0.28,-0.26,-0.11,-0.19,0.1,-0.13,0.01,0.11,-0.27,-0.23,-0.32,-0.33,-0.14,-0.31,0.07,-0.03,0.15,0.18,-0.01,-0.11,-0.26,0.02,-0.22,0.13,0.08,0.03,0.08,-0.02,0.23,0.4,0.0,0.19,0.17,0.46,-0.04,0.23,0.1,0.12,0.15,0.23,0.44,0.31,0.04,0.03,0.08,0.27,-0.11,0.09,-0.02,0.21,-0.15,0.41,0.14,-0.11,-0.05,-0.14,-0.06,-0.32,-0.1,-0.32,-0.26,-0.09,-0.15,-0.06,0.02,-0.18,-0.04,-0.01,-0.3,-0.04,0.03,0.3,0.14,-0.06,-0.21,-0.07,0.13,-0.01,0.06,0.0,0.17,0.15,0.03,0.03,0.14,-0.01,-0.13,-0.08,-0.12,0.04,-0.17,-0.16,0.12,0.32,-0.23,0.06,0.14,0.01,-0.16,-0.17,-0.21,0.13,-0.26,-0.03,-0.25,0.1,-0.07,0.0,-0.17,-0.09,0.1,0.03,0.06,0.07,-0.16,-0.01,-0.12,-0.22,0.13,0.01,0.06,-0.07,0.29,0.28,0.29,0.22,0.23,0.23,0.31,0.39,0.16,0.01,0.07,-0.14,0.1,-0.16,0.24,-0.17,-0.16,-0.03,-0.11,-0.3,-0.09,-0.17,-0.12,0.13,0.2,-0.06,-0.33,-0.25,-0.21,0.12,0.23,0.47,-0.08,0.18,0.12,0.31,0.18,-0.15,0.15,-0.15,0.28,0.0,-0.1,-0.33,0.19,0.04,0.04,-0.16,-0.09,0.06,-0.1,0.42,0.08,-0.14,0.15,-0.17,-0.03,0.06,0.52,0.13,-0.15,0.09,-0.22,-0.03,0.15,-0.12,0.08,0.26,-0.05,-0.2,0.19,0.21,-0.06,0.19,0.22,0.27,0.3,0.08,-0.01,0.29,0.06,0.15,0.08,0.23,0.34,0.47,0.22,0.04,0.16,0.16,0.46,0.41,0.26,0.17,0.17,0.13,0.15,0.0,0.12,0.27,0.28,0.56,0.23,0.1,0.2,0.1,0.27,0.08,-0.09,0.4,0.06,0.36,0.04,0.26,0.22,0.18,0.17,0.21,0.37,0.16,0.04,0.06,0.21,0.23,-0.04,-0.08,0.31,-0.01,0.02,-0.24,-0.18,-0.28,0.01,-0.13,0.1,0.06,-0.1,0.2,0.28,0.03,0.04,0.17,0.26,-0.03,-0.1,0.18,0.17,0.18,-0.14,0.05,0.16,0.26,0.14,0.18,-0.1,-0.28,-0.03,-0.05,-0.04,0.04,-0.35,-0.36,-0.08,-0.27,-0.37,0.01,0.08,-0.34,-0.27,-0.58,-0.11,0.09,-0.41,-0.11,-0.14,-0.41,-0.46,-0.44,-0.48,-0.31,-0.16,0.0,-0.21,-0.48,-0.12,-0.2,-0.13,-0.29,0.06,-0.12,0.32,0.29,0.28,-0.01,0.1,0.2,-0.17,-0.01,-0.11,-0.11,0.12,0.41,0.19,0.24,0.33,0.0,0.51,0.44,0.29,0.33,0.51,0.12,0.19,0.02,0.2,0.39,0.1,0.3,-0.04,-0.15,0.27,0.28,0.22,0.19,-0.15,0.06,0.23,0.08,-0.05,-0.16,-0.16,0.04,-0.08,0.0,-0.1,-0.05,-0.08,-0.03,0.01,0.09,-0.07,-0.23,-0.26,-0.18,-0.12,-0.45,-0.32,0.01,-0.28,-0.03,-0.25,-0.1,-0.35,-0.34,-0.24,-0.31,-0.2,-0.32,-0.2,0.1,-0.05,-0.02,0.3,0.09,-0.03,-0.19,0.06,0.19,-0.02,0.11,0.11,0.16,0.13,-0.1,-0.01,0.07,0.15,0.02,-0.03,-0.15,-0.11,0.01,0.23,-0.11,-0.19,-0.05,-0.32,0.04,-0.27,-0.24,0.04,-0.06,-0.11,-0.08,-0.08,-0.19,0.23,0.02,0.2,0.33,0.1,-0.01,0.3,0.3,0.38,0.21,0.12,0.13,0.22,0.09,0.37,-0.08,0.3,-0.24,0.05,-0.03,0.0,0.09,0.04,0.16,0.16,0.25,0.03,0.04,-0.01,0.39,0.23,0.13,0.42,0.48,0.32,0.59,0.39,0.25,0.23,0.2,0.48,0.63,0.36,0.44,0.51,0.78,0.77,0.6,0.65,0.57,0.53,0.82,1.31,0.66,0.98,0.82,0.76,0.77];

        for &quantile in quantiles.iter() {
            let potential_sd = AllData::estimate_middle_normal(&other_norm_data);

            println!("sd {:?}", potential_sd);

            let pars = potential_sd.unwrap();

            let norm = Normal::new(pars[0], pars[1]).unwrap();

            //find_cutoff(weight_normal: f64, normal_dist: &Normal, log_normal_dist: &LogNormal, data_size: usize, peak_scale: Option<f64>)

            let mut sorted_data = other_norm_data.clone();
            sorted_data.sort_unstable_by(|a,b| a.partial_cmp(&b).unwrap());

            let cutoff = AllData::find_qval_cutoff(&sorted_data, &norm, None);
            println!("cutoff {cutoff}");

        }

    }

    #[test]
    fn test_t_fit() {

        let t_data: Vec<f64> = vec![-1.74673575356551, -0.716212161115709, 1.1419346952728, 0.961701458348633, -0.835426104008679, -15.4178695019772, -0.342190084777164, -4.11182532534328, 1.37115600780661, 0.301607939800756, -0.205377589776568, -0.90209301150046, 0.584243057984495, 0.697970662639408, -0.723693672817299, 1.18182927997903, 4.25363386814657, 0.187051361007695, 1.34583720830727, -0.460493488852858, 0.0300804217394068, -0.994828930297786, 0.780050625446081, -1.73664005166326, -0.893075810581782, 1.27353063256346, -1.22813937341447, 0.792676733473614, 0.43820159807829, 0.457086692870032, -0.783977561380295, -0.273952858933103, 0.206600429356068, 0.34377789912316, -0.444962614830732, 0.0698021332068082, -1.41249423804385, -0.0453454688707125, -1.5626895521187, 2.04258042988626, 0.654428487058633, 0.547611739714776, -1.38462762378615, -0.551067772051512, 3.09977603643769, -0.151164906877533, -1.34552308476782, -0.506646993773148, 1.6452955064922, -1.26897824163202, -4.46699807541815, -0.433755781591723, 1.34538697216528, -2.00943238236091, 0.587179468171735, 0.172470687149357, -0.87384912247087, 2.93987614622441, -0.504826827297448, -1.96941000463708, -1.5851004115494, 3.28083530374256, 0.10194852088865, 0.515452551154593, -0.372851236878127, 0.94656723665731, 0.840093215140395, 0.333572971152078, 3.17652973042094, -1.18925294265308, -2.5950350935876, 0.662659646427808, -1.15503221143739, 4.84564939097141, 1.37461548683109, -1.92334860070084, 0.478081587498505, -0.917798312000448, -2.04790327347322, 0.127808406780702, -0.820024530281208, -0.56174887096964, -1.36229884465125, -1.12341246610811, -0.844798404382191, 0.43337186133017, -0.217249728310736, -0.0250667905299137, 0.782726091927995, 0.911112354313892, -1.59029614336338, 2.0314713987557, -0.786260582568002, 1.07332826799846, -0.586612712262912, -0.641609719438793, 0.34656649111713, 4.09963407263855, 0.203497812856241, 0.772252131191766, 0.202517672604001, 2.1972579904831, 0.329668691358791, 0.674870002993557, -1.4849519633224, 3.22438881148741, -0.895474094617742, 0.842347137655452, -0.155784952089526, -1.59765416271067, -2.12902113919246, 0.954190340852104, 0.799460274953457, -2.56544365616153, 0.062540724771491, 0.128062098510196, -4.20073441692631, -0.608944454114285, 1.42724366293682, 3.56153326332633, 0.146255024285299, -7.62388551008954, 0.181090441589188, 1.07134499921983, 0.159308103522203, -0.961716570383767, -0.48445102822011, -1.83700772974534, 0.277333529277746, 1.62462306526431, 2.21245291351148, 1.13659941360372, -1.04565281642519, -0.0149955614332196, 1.64280653750695, -2.78130813619043, -0.275911212294598, -0.970396221516611, 0.840809704936196, -0.126276667808504, 0.727477706295465, -0.496782198322523, 0.920945232682381, -0.560369527354825, 0.543249330089808, -0.0171776036506103, 0.431788738998995, 0.421171199579898, 0.0557133908944791, 0.307716580201441, -0.497288888798293, -0.412797777480554, -0.581465986498314, -2.48781256043586, 0.535127009825743, 0.704500172339479, -1.04619946971893, -0.975608668950808, -2.08162301987992, -0.437092803640378, 0.959111096507471, -0.0262197789507664, -1.02356899368407, 0.183402072104488, -0.387637111531151, 1.1746398409871, 0.631380499388218, -0.446814386019508, 0.264215044913558, 2.83165518511906, 0.354771690206051, -0.811830905711359, -0.046497815244381, 0.00572877001089766, 0.524031652366584, 0.0333513156482017, 2.15076232128801, 0.219537372607845, 0.470967309061242, -0.801064463394317, -1.64519985854295, -0.46025735420641, 1.86201956599003, -1.0394624230191, -1.83811655838106, -0.282814012048462, -1.51978378736349, -1.10481341054229, -0.820730496742721, 1.8743553859607, 1.5851148531285, 2.86416670792911, 0.722320656599473, 0.995976415852503, -1.55792953787465, 0.443346477799279, -0.0164441261152659, 0.534815817801359, -1.38283064617859, -0.188480007555302, 0.275426595308692, 0.935632902065095, -0.641915881961345, 0.927932005478441, 0.139749586144445, -0.0912125581181163, -1.17218700022986, 0.494495153243435, 0.462192061139227, 0.138627905298802, -0.614383827540185, -0.230606719891999, -0.288603434350296, -0.511543363561587, -4.28627120153408, -0.742992660909253, -4.03722528601833, -0.582932356918311, -1.77804857570474, 1.53646755024283, 1.28507723800106, 0.337129607204999, 0.174350026230671, -1.8813529247703, -1.19099948062733, 1.12731982279498, 0.194991852183234, -0.437521269797305, -2.15260591025704, -1.03348452807266, -0.0921897558663505, 0.512498857260962, 1.58383383885318, -2.18018985639833, 0.515355823753861, 1.63482870824571, -0.672520804228583, 0.162932606662963, 1.61274405933403, -2.1215109409226, 1.24019736869035, 0.967692455194374, -0.0495410621281779, 0.798997677535636, 1.80733026483821, -0.21301931984333, -5.38434826340137, -0.818882448970119, -3.48964200171198, -14.9772868701973, 3.36713691949314, 2.25474490532692, -1.48566277162739, 2.25581725141627, -4.22213866181079, -0.228127621488178, 3.14507296753417, -0.396553339751026, 1.56287891226499, -0.40738895941158, 0.434639360551197, 1.44900129363455, -0.421199439478994, -1.07854452826977, 0.0578746657657618, 1.30258428907769, -1.8385414260468, -0.271747140197581, -0.174919721664282, 0.348926213966345, 1.69821914963413, 0.149928211717208, -0.510939251089885, -1.74083083681475, -1.05126925299071, -0.380237898334832, 0.0737540042024673, 0.295298631333461, 0.290695107838783, 0.274636954983808, -4.02706879685937, -1.50813530087038, -1.42499456792205, -1.38000804284457, 0.612693256068562, -0.774235462481855, -0.659952417042899, 0.523371291406965, -0.222037316566273, 0.0379764954255171, -0.603144909424826, -0.187862279123839, -1.86195656675779, -1.10069061498158, -0.505748315707019, 0.187797438328183, 3.01684664846919, -0.100535486434181, -0.745260267515785, 0.367842322961049, 1.31119186759396, 1.0152772427299, 0.590604835533575, -0.223752417299306, 0.243479341360219, -1.94863651542325, 1.53549737156324, 1.23225261375545, -0.349100069680047, 0.0706968532757556, -0.0645085019571481, -1.20563631750184, -0.489821854910099, 0.726190946282985, 1.02601032937587, 0.128668155371176, -0.552248596240967, 1.18330417652389, -0.325010964075305, -0.2472580459586, -1.92098820325846, -1.1169125620275, -1.18244297884563, -10.7113226361415, -1.26652095266941, -0.147943242960308, 1.2417008054995, 4.59771178316017, -3.34447450522792, -0.0851661085637802, -0.357891376949495, 0.696473153879502, -0.128216983204886, -0.739271803276655, 1.00420219713201, -3.00500812881333, -0.0316411051916937, 2.69321302445021, -1.65556214465242, 0.955357921033628, 0.0606954876009854, 0.0390007905458031, 4.82021017655767, -0.682579414023819, -0.960366131175079, -0.556479338498371, -1.45242895933646, 0.596168593539281, -7.91677908513323, 1.35939126835711, -0.460697836816571, 0.912254951518001, 2.326027861631, 1.62918574129847, -1.57161894968773, 0.626754484862087, 1.36847664705086, -3.13034573280537, 0.372463248324763, -0.795570220933577, -0.368188155462083, -0.95602302493361, -4.23322723163552, -3.56657815993771, 0.136019476706842, -1.10452678811085, -1.93846945688305, -1.5101578361396, -0.469446118976909, 1.15205145102958, 0.922046509245872, 0.359943393511745, -0.6265824067111, -1.58782796003766, 0.411916648361149, -0.977869399259474, 0.966486694122306, 0.235399066804392, -0.395464872619317, -0.492797194938103, 8.74962653453128, 5.0230140465103, 0.168313156057922, -2.39968341812687, 1.22653800177954, -0.416475258358083, -1.62924941421045, -0.559573852342494, -0.899003447091992, 0.314519301285714, -0.297425137299043, 2.66605310441356, 0.881761656260239, -0.479224654653184, 0.452797828932011, 0.195494116897998, -0.34755956745453, 1.34779858369339, 1.03163756301158, 1.21872632352012, -16.065424469021, 1.17465040935234, -0.232868101424964, 1.48061043147822, -8.09959084142855, 0.254460621671512, 0.34503704702956, -0.342847291251827, -4.02677906818036, 0.289439539937635, -0.728419317069381, -0.0873862296686149, -0.858446369722462, 0.607452642156903, -0.358063516765653, -0.938874477014561, 0.843638227654484, 0.233803293978078, -0.579060476995525, -0.116377255095641, -1.05561222038889, 3.77410071333684, -4.82726273613664, 2.04993940116503, 1.75065607173716, -1.19013952002075, 0.127457674135824, 0.252588956541844, 0.963057360883677, -0.618997449944483, 2.48659279484705, -0.416189491621265, -2.00216411912282, -0.677017193080277, 0.574404938035795, 3.3042134591326, 0.232860864998359, -2.08889167428848, 2.3821979929406, -0.937914426866838, -1.35166897711362, 0.79909109101246, 4.15108782043384, -0.441367350831788, -0.172454855810497, 0.140924337860944, -0.0780061519062443, -1.51657988679857, 0.877256792120846, 0.280281837163135, 2.16777048111879, 0.706506418751284, 0.57087491128625, -0.347334652470865, -0.505559200009588, 0.442920680856992, 1.75180770814366, 0.0600551600613354, -0.299402089711286, -0.0574117945313828, -0.585234657431769, -0.44163833436539, -0.105686294306999, 2.17928490258259, 0.271094522077331, 4.42192535078716, -0.271258052440317, -2.72322954950695, 0.69743405283341, 2.37804315601357, -0.217390212169295, -0.231925114150219, -2.13481549880803, 0.552375082358016, -0.401572088105497, 1.4857614267862, 0.230970686779611, -0.412353709403135, 1.25917344637959, 0.507378084912823, 0.0775143043651686, 2.18276903555154, -1.59922674583737, -1.80511958235206, 1.18075900190091, 1.4822440904096, -0.125796131785605, -4.61777569655889, 1.83180400709663, -0.574473359206843, -1.05469564572063, -0.658350365996412, -0.45347356858538, -0.695425413148897, 0.923448013949057, 0.0854343328369129, 1.54271363031215, -0.32560276386411, 3.95869470654606, 1.07285442614339];

        let (sd, df) = AllData::estimate_t_dist(&t_data);
        println!("{:?}", (sd, df));

        assert!(((sd-0.9859).abs() < 0.01) && ((df-2.28496661).abs() < 0.01), "parameters fitting off");
    }

    #[test]
    fn see_data() {



        let _a = AllData::process_fasta("/Users/afarhat/Downloads/sequence.fasta", None);

        //There's a vector that we define below, but it takes up most of the page, since it was copy pasted
        //It was generated in R using a unity scale parameter and 2.5 degrees of freedom, with AR coefficients of 
        //0.9, -0.4, 0.03. The results I get give me a decent whack at the T distribution, and the three AR coefficient 
        //model even gives me 0.8887481074141186, -0.44659493235176806, 0.1416619773753163 for the AR coefficients.
        //But the BIC actually favored a single AR coefficient of 0.6346814683664462, though with similar T distribution success.
        //I suspect that the data just isn't long enough: it's "only" 1000 data points, and 
        //using ar from R gives similar results with the AR coefficients. 
        let _dat = vec![vec![0.92030358445241, 1.13882193502168, -0.117210084078362, -0.807361601078728, -0.438976319932791, 0.268127549092788, -0.052278140706468, -0.097668502664112, -1.47144080768618, -1.33214313894382, -2.17597210917359, 0.572919562976975, 2.00048064323903, 2.05359733016391, -0.319394697044939, -1.59994751216035, 1.84918907421627, 2.1435024238699, -0.204044958336222, -1.49221275359722, 0.44822708430529, -0.274810113068454, -4.93838439350979, -4.754930877994, -0.96694136401746, -1.12585879058432, -0.181964824086987, 0.430037620784272, -0.44780509784776, 2.35937756512511, 1.81063614907778, -1.29702264945256, -3.40589392873406, 0.788658912135261, 2.13518843382043, 2.01948175887685, 0.614266739946927, 0.755670252073416, 1.335074198794, 1.25129965143572, 3.79133984475769, 1.76248528301837, -2.48779528723107, -2.16761002994477, -2.0578885650047, 3.78595983582816, 5.54010646417997, 1.48662662577973, -0.286418239896885, -1.60402218429414, -3.3323571446058, -2.23829669684377, -1.5496693651271, -1.1611033351847, -1.85457300117196, -2.37457691404281, -2.27492152660748, -0.719863937034595, -0.0263959684201648, 0.170874766907551, 0.92547585140182, 1.6748988347599, -0.447951289632964, 0.986119979724126, 0.330674880079694, 0.963049129491544, 0.177445151639323, -0.857208488357629, -0.467007735175719, 4.02753386087264, 3.98936512706117, 2.73765696914249, 1.19148890983396, 2.29421617548849, 1.99999739443913, 1.59292585508839, -0.782491166253844, 1.94297634165677, 1.21398885502754, 1.13427183552991, 0.437753448126084, -1.62096512795834, -3.7289629785395, -1.74035768520633, 0.676094595844811, -1.37268433517485, -1.49552374577999, -0.64855270074651, -4.22740287934142, -4.19905167759636, -0.680398276185732, 4.50197339941747, 4.34421884390742, -5.53528985862542, -6.40329876675413, -2.3471813820915, 0.442105670582741, 0.182952122974677, -0.567051827238743, -2.4072720533326, -1.65690202413685, 1.07930851005701, 3.77437322061755, 4.0521048474126, 1.12387151330085, -0.511121941808973, 0.854811329980915, -1.77381301708499, -2.22960111791772, -2.24186768090914, -0.33824515122749, 0.399161734912879, 1.15476529778078, 0.372694521178202, 0.806439034677838, 0.0509907543173711, 0.277746230739652, 0.236590873328464, 0.535151755328273, 0.816503816966147, 0.583603850232557, 0.522413071284131, -0.236063550226613, -0.816712085691182, -1.20640905139121, -3.25397777891827, -1.93539073321495, 0.228047352471594, -0.164419892576064, -1.27286723525435, -3.15459415400426, -2.77201324491975, -0.312049179376233, 0.707103432958397, -0.345516629618616, -0.419765743116912, -0.606006525500174, 0.786774766395147, 1.56928742705041, 0.6326541960028, 0.229492093487767, 2.83221501366845, 2.83094799099263, 0.610021043519263, -0.544891622064043, -0.643753667524667, 0.180910631725579, 0.437325602549168, 2.45267850086629, 2.2574451013196, 1.53471626797884, -0.249217507715209, -2.41565876963859, -2.48862175575569, 0.581046968433886, 0.406458787784514, -1.77938108942122, -2.02940909858835, -2.62230577669098, -2.70650640281147, -2.2685462215543, 0.836597174385689, 3.16427560661299, 5.3093194974799, 4.26009587691776, 2.80126317428491, -0.58155144686093, -1.07275221778199, -0.665262648146138, 0.347733778176803, -0.835947753094469, -1.09988437605543, -0.369658228558103, 1.01781581419214, 0.38898511095316, 0.859802532802686, 0.788512295711363, 0.286197048229631, -1.20422049912365, -0.680126746388365, 0.340352100486049, 0.680868879317883, -0.158146478740161, -0.635075539470715, -0.776486761998339, -0.96109562793401, -4.85971482805948, -4.75559235784913, -6.40220534569677, -4.58847166974754, -3.48945871093429, 0.0392772179300767, 2.57855606841942, 2.53743542028242, 1.42779779365498, -1.5339523965412, -3.06655269236792, -0.976162707910025, 0.497047920115144, 0.308290360699297, -2.10324863491103, -3.03481300616886, -1.97297329663293, -0.112349366288463, 2.18086433766167, -0.23166140488643, -0.568855656697248, 1.2809491093026, 0.700925814676067, 0.264320526449464, 1.6086906805472, -0.758389764569628, -0.0779000758477091, 1.2491990131757, 1.08314638713195, 1.27181281840868, 2.55617921694851, 1.61131123966082, -4.91848544993378, -5.81335847326639, -6.70593111048839, -18.8348360438283, -11.076243829955, -0.381118057415875, 2.03678342736545, 3.90908224411287, -1.53011155477786, -3.10775741561244, 1.07740838271751, 1.77031382429637, 2.63196527857635, 1.28557651307011, 0.591981525612649, 1.54651301981518, 0.772436963501713, -0.984197023275923, -1.0904820499888, 0.738002362303196, -0.767672390676678, -1.29056769822753, -1.00722162292929, -0.0643763390993075, 2.00445206266964, 1.94946895507173, 0.439870693233837, -2.0646012330529, -3.02687457137971, -2.26538839855839, -0.816283762939789, 0.375992366969616, 0.887640091330601, 0.898627577505309, -3.56208024262775, -5.04682934749756, -4.5153500562936, -3.53195376178861, -0.911329987448676, -0.317111448159027, -0.686779338260212, 0.00477456661292525, 0.0474581852446739, 0.0581756553527472, -0.569626856706835, -0.72237296674375, -2.28249622448385, -2.88307683602075, -2.20919016733446, -0.715717864599052, 3.16988433218321, 2.9723718533503, 0.640449131688225, -0.14960566989417, 1.00953826761443, 2.00291742549119, 1.98492704133305, 0.791801097732398, 0.222217035550891, -2.00581381128042, -0.334867839877523, 1.73986359344438, 1.29054988603249, 0.526200278130919, -0.0549522982489849, -1.42685700059732, -1.73622622780417, -0.267318427449307, 1.43710852577524, 1.47690641271449, 0.194104212068498, 0.810348658072999, 0.370968335744429, -0.231702880655758, -2.25359767040422, -3.04134026305666, -3.02516123385461, -12.2850395715001, -11.2042322813694, -5.40849130460833, 0.487200356754749, 6.86346165764169, 2.47550610480946, -0.587979266589305, -1.6713713090744, -0.498304134507453, 0.0742184413704876, -0.523294691512467, 0.488600474187374, -2.3537232721986, -2.36113108059075, 1.52434237502359, 0.590186726939155, 0.80595509285171, 0.59571065164257, 0.270463941691607, 4.84952211620864, 3.59167623343659, 0.340447550685146, -1.54126137277012, -2.86799292810053, -1.35830706612259, -8.03829611458646, -5.41779219616468, -2.16214157951388, 0.892295524983789, 3.83141670003702, 4.65567831395285, 1.11269371860454, -0.119149992973854, 0.955834515351163, -2.18905386024164, -1.9835935318224, -1.67650782001654, -1.14927939655526, -1.3792791597814, -5.06516195141717, -7.60799063419726, -4.72648568829726, -2.467122512442, -2.49652516178786, -2.9119760474208, -2.16562817231374, 0.292880760061899, 1.96453118080445, 1.94590030704158, 0.347701820106399, -1.9943205093344, -1.46367552887112, -1.48701811690653, 0.153810985174844, 0.924725934358231, 0.330653530725958, -0.560485061472788, 8.14067034494613, 12.5837309874727, 8.22058835496073, 0.209573816697079, -1.4955689755532, -1.59969921238597, -2.46446390063563, -2.18257874722677, -1.92552973571341, -0.619359878984718, -0.150114496516727, 2.72092811707099, 3.37206196386128, 1.46275643109807, 0.502081674887892, 0.163426910773712, -0.357425324780405, 0.975807477328182, 2.05773722984232, 2.66964407970353, -14.4565654649049, -12.8423840240482, -5.92829821471527, 0.848398683906602, -5.34998426074795, -5.07773363300574, -2.05947755785923, -0.325783167945275, -3.64852490517759, -2.92570393427986, -1.91191639088858, -0.747285154911719, -0.854007570815973, 0.0804023986605571, 0.0334831157078864, -0.966520859466166, -0.0372117201884032, 0.587925583066218, -0.064038389944552, -0.410298390877877, -1.38162764870917, 2.6928340341514, -1.86336999764305, -0.745676039835554, 1.90568065596692, 0.767342386354405, 0.0334232782729526, 0.0329033721247472, 1.0023213560774, 0.270933120023467, 2.33049116160095, 1.60294894649253, -1.48357853831922, -2.58350272131656, -1.10922762742664, 3.29480232682567, 3.56436892847262, -0.232157398216184, 0.846352832961757, 0.0235971499393957, -1.67593739729935, -0.693300841543859, 4.19819993646229, 3.56405480668283, 1.33511547037282, 0.0428523366172192, -0.466563592899391, -1.91357459094372, -0.65704933247023, 0.440370366530435, 2.76951630625596, 3.00321146779537, 2.17916982079561, 0.495719088314716, -0.840983604810727, -0.44687710417468, 1.70088332896018, 1.74437148965104, 0.576772606265343, -0.185038544884187, -0.930147245644142, -1.18745225930348, -0.807885585769002, 1.89926436174256, 2.26796311417413, 5.67914984127379, 3.98976948988871, -1.3360580516914, -1.93055149440612, 1.29466311642128, 1.67994544882157, 0.704243998388495, -2.13413418629437, -1.59964492119767, -0.966471522713996, 1.19177099923384, 1.64216384753974, 0.559891208007671, 1.14196312454762, 1.3604533292288, 0.861933787092273, 2.4485870059795, 0.300541644584132, -2.48820889100537, -1.10538804765819, 1.4916946532569, 1.58423800847872, -3.82180099166054, -2.19676124919164, -0.975310946560743, -1.16842502769846, -1.38571134977648, -1.26250310070165, -1.31244641470074, 0.205676140505754, 0.757646332151339, 2.10295148060503, 1.27016532003506, 4.28339229230014, 4.48292990561765, 2.10562909543797, -2.71852949855216, 0.723910326172097, 2.44179494933801, 6.48289894798974, 4.69471299611186, 1.67744299489166, 1.3966035029669, 0.35936870269668, -1.22486652810243, 0.269995457628434, 0.759061909981033, 0.946641927025073, -1.72846034103023, -1.73818606278888, -0.567904098733486, 0.639322223001504, 0.688665499408402, 1.71224988361683, 1.17573556087929, -0.41575971995599, -2.41333500847634, -1.9350045617501, -1.88619779027742, -4.42719992370515, -2.59560242752011, -0.881805083016513, -1.28065489822725, -1.04963419365515, -0.506449452641461, -2.50843368933851, -0.668397124292493, 1.37657526425846, -1.55807299632588, -1.55946789943153, -1.12842232992949, -2.33363536729417, -2.01925011809299, -0.565919112099035, -0.0609487335389706, 0.317933373700252, -0.304417191953899, 0.983700975247276, 1.91452101421138, 1.18934685577676, 0.0143581321445323, 5.11618800145032, 5.27895214624172, 5.52692149213463, 2.11695955618689, -2.11740766158195, -5.25839644433757, -6.01815665282092, -1.82109088522659, -0.155593030300168, -1.43950271574513, -0.9188527465227, 0.815157605995218, 0.373297424278821, 0.199535493630113, 0.439568578819839, 0.156387149739652, -0.195906002002079, 0.242447953004853, -0.376141898640107, -2.22056282192827, -2.77698192302861, -1.9528087368393, -2.7175854999197, -1.37325274211328, 1.73178489393634, 39.3756188656999, 33.8909649888857, 14.4694535421062, 0.752712478816419, 15.4268099521774, 14.2158561610901, 6.78642948035382, -0.594434074904627, -3.21412649658063, -2.61233815128654, -1.35769241640445, 1.2852729899965, 1.90217731860234, 1.02137924636608, -2.0723925958159, -0.960204356007614, -0.232432636192525, -1.50204014499291, -1.66926512530075, 0.244202364048352, -5.04145650251787, -5.14174187549786, -3.72568346298302, -2.32346552146221, 0.304534764715309, -4.44045607257843, -4.57535110007076, -3.13198444723472, -2.77124551099526, -3.23772091612678, -0.554117999156476, 1.08014628105036, 0.0684690899019627, -0.565335019423998, 2.42761136922389, 1.11275508593899, -0.580558015119916, 0.15500967086521, -0.537243481976309, -0.728207123729817, 0.10547763314165, 1.66789563918996, 3.9608848018793, 2.81805754820624, -0.336439131244874, -2.29609377902806, -1.80844475512746, -0.0792687057484585, 0.177036360993458, 0.265469329917371, -0.142596638223382, -0.611260775935601, -1.80995528248858, -0.693267766393418, -0.0122818812108099, 1.936504109985, 2.21341078569281, 1.73986581729794, -1.69726138161603, -1.83683100568767, -1.3516335312074, -0.288339644404166, 2.43793017610994, 4.021428674649, -5.62574595404721, -7.16564773674705, -3.6736013112829, -0.298555021497173, 1.46606642601136, 1.73817628961036, -0.631252574249609, -1.70109847571968, -1.0588101460626, 1.31347166509332, 1.64084894143721, 1.99886186392205, 1.12831155370992, 0.690924501872247, 0.957891378127895, -3.37623128298566, -2.93643120666729, -1.15984483519665, 2.00528280804466, 2.04759965000856, 0.0583231911811567, -0.453288259561501, -0.109519427650551, 2.57436669556402, 1.49430321737785, 0.134278896045808, -0.144516907042029, -0.777281265785068, 0.0991679994599942, 1.20490956766865, -1.39587600004585, -1.62744850333245, -2.05432035624814, -0.58996530704156, -0.358105747461861, -0.969668985312145, -1.02105920863618, 0.239728865376966, 0.47317512622918, -0.530543630477763, 0.0293416979434837, 0.142727136639976, 1.42021631976613, 0.380389447413165, 1.12413364743777, 2.43727019448504, 1.93515168714463, 5.26755041331374, 5.58070524402661, 2.78931132295857, 0.902856378124465, 0.122908263977391, -0.595275969687596, -0.735158242565535, 0.295448457389827, 1.49651326401657, 0.391698494275945, 0.920326367938329, 2.46530234749983, 1.01213008657988, 0.433856062354837, -0.437494735853989, 0.40535423982244, 1.15126516377248, 0.23339538930523, 0.339574870019606, 0.719039091000082, 0.479682592330763, 0.135817819973831, -0.91965154072523, -2.91630334627842, -3.77195809993614, -3.60170693982958, -1.24802781745584, -1.86833465278352, -0.743798884986824, -1.08410657133127, -0.472732160855543, 4.39132589296646, 3.08930499464002, 2.36512322295162, 3.08063745146265, 2.45705961087452, 1.57483894070708, 1.01831101736379, 0.378458335470199, -1.70675948398771, -2.1822923904093, -0.0739376552535938, 1.86124731132437, 2.08536472008376, 0.786708041879707, -1.2435609369744, -1.34280907050502, -1.33119214771747, 3.23998953547924, 4.31159864671867, 2.35261316384532, -0.721232295730943, -0.000767556419266957, -0.279212186920793, -5.10912053107504, -3.15606637872521, 0.810156598975859, 2.47891375120775, 1.63767018222001, 0.0631812488987988, -1.17732679139782, 1.92794816920531, 1.74904845806404, -0.156383771283357, -2.19262557443902, -2.51112473710029, -1.90062704605517, 0.402995421482425, 0.376778023464264, -6.8365740860473, -6.34240343535013, -4.29305045524306, -0.878874430238532, 0.374237390895764, -0.300067592209067, 0.3388784295558, 1.11268864866726, 1.19515848411267, 1.83213014975393, 2.49724894714788, 0.84771674552611, -2.46736329705546, -2.5191181746953, -1.01569034667967, 1.62932821753159, 2.50766843790521, 1.20717574713547, 0.352815370455899, -0.999757054645123, -0.580679577639651, -0.816810009237473, -3.29534040799893, -2.24485379985796, 0.691784280946011, 0.216108751988089, -1.62615761990548, 0.408471829519518, 1.70754234973563, 0.875278523833068, -1.79755391245616, -0.765345611478853, 0.961522905168801, -0.0677954204535503, -2.80295381399077, -2.4576751843832, -0.688791062450808, -0.576354591112729, -5.80502724424317, -4.03925077048987, -0.912748784409071, -0.374515942597487, -1.7538124093463, -1.15408745900706, 1.2580056536649, 0.721364645131387, 1.96752835877277, -0.150059868557396, 3.51202955275845, 1.13520234255457, -1.61726747838628, -2.56756908397205, -2.63204713631664, 0.30226372880621, 1.28794514858261, 0.671352038071078, 0.261478416834908, 1.08695050750061, 0.423545944792975, -1.19784121781733, -1.25031210199527, -0.91963774603955, 0.891940983526066, 0.901987405160857, 2.30650675939515, 4.6435005147262, 2.45932855078223, 0.297604982802341, -0.997613930977183, -1.91751949014799, -2.35529748124354, -1.00887004861156, 0.1095743811857, 1.12890262680236, 0.846089323966845, 0.261837521224171, 0.115115610489419, 0.216847568992397, 0.875783561934371, -1.17854992494425, -0.968731905389821, -0.399063747524496, -0.297912634101954, -2.96613279091619, -1.97240254891385, -1.4599198731415, -0.378141799678276, 0.479491771953498, 0.834233759616706, 1.37053675230751, 2.0941572442331, -1.16460959444015, -0.764486018234382, -0.352710584937402, 0.572253210602834, 0.668029528531709, 2.20159709483534, 1.41890332324469, 0.875550582431663, 1.20473379996235, 0.211898542324319, -1.12347409650265, -0.692781057630004, 1.46757550969897, 1.26208281132218, 1.85169115910733, 2.39689108393837, 1.06281480117464, -0.144443954694692, -1.00027614124819, -1.14885621925509, -1.88654568223697, -2.74513112047395, -2.39747105619643, -0.83925646988332, 5.40097572023663, 5.99695305508785, 3.097246571825, 1.17225029496594, 0.983828421192127, -0.707895763171177, -1.37884892823024, -1.07299573491272, -0.0122615442489361, -0.0961852583005038, 1.09626349174379, 0.721317139639417, -0.746425254133715, 2.49353605610903, 2.0467897331417, 0.433641666116876, 0.856641385611092, 1.15680501747799, 2.77394548396039, 3.69593532895137, 2.6535133350672, 3.38675711442973, 2.99091319034043, 1.80789866198575, 1.12677881443069, 0.371618523335552, -0.671924380110139, -1.83224885481871, -0.892266039477728, 0.684609001079026, 1.90098662280984, 2.2262546101034, 0.709380974435907, -0.145153169339163, -0.928384207891866, 0.0965052514508109, 0.0763368034118342, 1.32492946814549, 0.570716813320824, -0.911002702860683, -3.28224164226898, -2.39303599575728, -0.411549693165511, -0.0291862375250271, 0.893623116037588, -0.295265546820206, -0.614764926640702, 0.698562080422445, 0.735524713229628, -0.117251474877304, -3.46268336651757, -3.25253880571123, 0.147233575684406, 1.12966327041718, 3.65147122399176, 1.48012569343478, 6.26864823635147, 4.34178548204398, -0.838469751480694, -2.15903520515314, -4.34713860185419, -2.73079810536133, -0.829798401512069, -0.631684474979878, -1.7531360696792, -1.78284547084014, -0.0697809897071269, -0.611863907510918, -0.778320088928396, 1.04919107826397, 1.46916383599131, -0.137959299596261, 0.563145799441769, 1.36750378759618, 2.48441675538289, 3.29818845077975, 1.32467920894071, -1.36733118501851, -1.72566923694308, -1.79596223580596, 0.114151912857804, 0.862165475999593, 1.59554655288258, -0.0276642018549236, -0.338791392457265, -0.225791127427969, -1.80500087400132, 5.81066879292208, 6.82681146222648, 4.20923536804268, -0.534462767896464, -5.03485588235798, -3.84848478395311, -2.23592167242184, -0.544468735595892, -0.117816678225373, 0.199580905498544, 0.166682957910035, 0.538351479212281, -0.165184340363526, 1.70846490291775, -0.428924165142648, 0.144241067191096, 1.06316121304816, 1.86974858624133, 2.35126689470158, 1.22896901886272, 1.12070903797652, 0.246119902768321, -0.265001413346171, -5.51699517536417, -4.25858344932621, -1.58139965152254, -0.0433448659995377, 2.92385315119381, -2.08187248048284, -3.6032172038181, -4.85253734027665, -1.33172778444363, -0.17840284510077, -0.740238718902239, -0.825467030574543, -1.32830256094649, -0.777800718943696, -0.178624767186471, -3.28185149322273, -3.26786225848825, -3.25883383892214, -1.08076857124956, 0.0342743588621588, 0.431637922000286, 0.0507999088172149, 1.03008395151914, 0.892826084191693, 0.0895019158796122, 1.41473827823274, 2.91493612377213, -1.97902276208714, 1.06367253923166, 5.27385712911363, 3.67042320487516, 0.782561388675378, -0.360141954181277, -0.624567865802663, 0.281060321724171, -0.671012108419172, 2.50260431027482, 2.07970610518249, -1.20989782009227, -3.16824951099057, -1.71394421323118, -0.417022106203542, -0.756027370113074, -1.58167131244656, -0.238544715546333, 0.357049482958872, 0.407612838454049, 0.456734414382648, -1.16074618579003, 0.333172331591581, 0.199152592671834, -2.11020086331783]];

        //let (blocks, back) = AllData::process_data("/Users/afarhat/Downloads/GSE26054_RAW/GSM639826_ArgR_Arg_ln_ratio_1.pair", Ok(4641652), 498, 25, true, false, None).unwrap();
        let (blocks, back, min_height, null_blocks, _) = AllData::process_data("/Users/afarhat/Downloads/GSE26054_RAW/GSM639836_TrpR_Trp_ln_ratio.pair", Ok(4641652), 498, 25, 3.0, true, None, false).unwrap();

        //        // /Users/afarhat/Downloads/GSE26054_RAW/GSM639836_TrpR_Trp_ln_ratio.pair

        println!("BAADS");

        for (i, bl) in blocks.iter().enumerate() {
            if bl.len() == 0 { 
                println!("no data in block {i}");
            } else {
                println!("start {:?}", &bl[0]);
            }
        }

        println!("finish standard");

        for null in null_blocks.iter() {
            if null.len() == 0 {println!("no null here");} else{
                println!("start {:?} end {:?}", &null[0], null.last().unwrap());}
        }

        println!("{:?}", back);

        //let a = AllData::create_inference_data("/Users/afarhat/Downloads/sequence(1).fasta", "/Users/afarhat/Downloads/GSE26054_RAW/GSM639826_ArgR_Arg_ln_ratio_1.pair", "/Users/afarhat/Downloads/", true,
        //                       498, 25, false, &None, None).unwrap();

        // /Users/afarhat/Downloads/GSE26054_RAW/GSM639836_TrpR_Trp_ln_ratio.pair
        let a = AllData::create_inference_data("/Users/afarhat/Downloads/sequence(1).fasta", "/Users/afarhat/Downloads/GSE26054_RAW/GSM639836_TrpR_Trp_ln_ratio.pair", "/Users/afarhat/Desktop/", true,
        498, 25, 4.126, 3.0, &None, None, None).unwrap();
        for (i, b) in a.start_genome_coordinates.iter().enumerate(){
            println!("{:?} start b (zero indexed) {} Look for match in file on vim line {} vim column {}", a.seq.return_bases(i, 0, MAX_BASE), b, (b/70)+2, b-(b/70)*70+1);
        }

        println!("Null bp starts {:?}", a.start_nullbp_coordinates);
        for (i, b) in a.start_nullbp_coordinates.iter().enumerate(){
            println!("{:?} start b null (zero indexed) {} Look for match in file on vim line {} vim column {}", a.null_seq.return_bases(i, 0, MAX_BASE), b, (b/70)+2, b-70*(b/70)+1);
        }
        //println!("{:?} {:?}", serde_json::to_string(&a.0), a.1);
    }
} 
