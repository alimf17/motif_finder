use rand::Rng;
use rand::seq::SliceRandom;
use rand::distributions::{Distribution, Uniform, WeightedIndex};
use statrs::distribution::{Continuous, ContinuousCDF, LogNormal, Normal, Dirichlet, Exp};
use statrs::statistics::{Min, Max, Distribution as OtherDistribution};
use statrs::Result as otherResult;
use crate::waveform::{Kernel, Waveform, Waveform_Def, Noise, Background};
use crate::sequence::{Sequence, BP_PER_U8, U64_BITMASK, BITS_PER_BP};
use crate::base::{BPS, BASE_L, MIN_BASE, MAX_BASE};
use statrs::function::gamma::*;
use statrs::{consts, Result, StatsError};
use std::{f64, ops::Mul};
use std::fmt;
use std::collections::{VecDeque, HashMap};
use std::time::{Duration, Instant};
use once_cell::sync::Lazy;
use assume::assume;
use rayon::prelude::*;
use nalgebra::{DMatrix, DVector, dmatrix,dvector};
use core::f64::consts::PI;

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
  
    //TODO: I need to create a master function which takes in a fasta file and a IPOD/ChIP-seq data set 
    //      with a spacer and boolean indicating circularity, and outputs an All_Data instance
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
    //         a reference genome.
    //and the String that will be used to make the file name for the full data struct
    fn process_fasta(fasta_file_name: &str, null_char: Option<char>) -> Vec<usize> {

        let file_string = fs::read_to_string(fasta_file_name).expect("Invalid file name!");
        let mut fasta_as_vec = file_string.split("\n").collect::<Vec<_>>();
        
        //We want to trim the whitespace from the bottom of the Fasta file
        //From the top can be dealt with by the user
        while fasta_as_vec.last()
              .expect("FASTA file should not be empty or all whitespace!")
              .chars().all(|c| c.is_whitespace()) {_ = fasta_as_vec.pop();}


        let mut base_vec: Vec<usize> = Vec::new();
       
        let mut fasta_iter = fasta_as_vec.iter().enumerate();

        let first_line = fasta_iter.next().expect("FASTA file should not be empty!");

        if !first_line.1.starts_with('>') {
            panic!("{}", ILLEGAL_FASTA_FILE_PANIC);
        }

        for (line_pos, line) in fasta_iter {

            if line.starts_with('>'){
                continue;
            }
            for (char_pos, chara) in line.chars().enumerate(){

                if (Some(chara) == null_char) {
                    base_vec.push(BASE_L);
                } else {
                    base_vec.push(*((*GET_BASE_USIZE)
                            .get(&chara)
                            .expect(&(format!("Invalid base on line {}, position {}", line_pos+1, char_pos+1))))); 
                }
            }

        }


        //This regular expression cleans the fasta_file_name to remove the last extension
        //Examples: ref.fasta -> ref, ref.txt.fasta -> ref.txt, ref_Fd -> ref_Fd
        base_vec
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
    fn process_data(data_file_name: &str, sequence_len: usize, fragment_length: usize, ) {

    }



    //TODO: I need a function which marries my processed FASTA file and my processed data
    //      By the end, I should have a full All_Data instance 




    

    fn chop_file_name(name: &str) -> String {
        let re = Regex::new(r"\.\pL+$").unwrap();
        let piece_name = re.replace(name, "");
        String::from(piece_name)
    }



    fn compute_autocorrelation_coeffs(data: &Vec<Vec<f64>>, mut num_coeffs: usize) -> Vec<f64>{

        let min_data_len = data.iter().map(|a| a.len()).min().expect("Why are you trying to get autocorrelations from no data?");

        if num_coeffs >= min_data_len {
            eprintln!("You're asking for more autocorrelation coefficients than this data can yield. Forcing smaller number of coefficients.");
            num_coeffs = min_data_len-1;
        }

        let mut coeffs = vec![0.0; (num_coeffs+1)];

        coeffs[0] = 1.0; //all data is perfectly autocorrelated with itself

        for lag in 1..coeffs.len() {

            let domain_iter = data.iter().map(|a| a[0..(a.len()-lag)].to_vec()).flatten();
            let range_iter  = data.iter().map(|a| a[lag..(a.len())].to_vec()).flatten();

            let denominator = domain_iter.clone().map(|d| d.powi(2)).sum::<f64>();
            let numerator = domain_iter.zip(range_iter).map(|(d, r)| d*r).sum::<f64>();

            coeffs[lag] = numerator/denominator;
        }
        coeffs
    }

    fn compute_ar_coeffs(correlations: &Vec<f64>) -> Vec<f64> {

        let mut corr_matrix = vec![vec![0.0;correlations.len()-1]; correlations.len()-1];


        for i in 0..(correlations.len()-1) {
            for j in 0..(correlations.len()-1) {
                if i <= j {
                    corr_matrix[i][j] = correlations[j-i];
                } else {
                    corr_matrix[i][j] = correlations[i-j];
                }
            }
            
        }


        //We adapted the pseudocode on wikipedia for the Cholesky–Banachiewicz algorithm, since we know our correlation
        //matrix must be positive definite
        let mut L_matrix = vec![vec![0.0;correlations.len()-1]; correlations.len()-1];
        let mut L_transpose = vec![vec![0.0;correlations.len()-1]; correlations.len()-1];
        for i in 0..(correlations.len()-1) {
            for j in 0..=i {
                let mut sum = 0.0;
                
                for k in 0..j {
                    sum += L_matrix[i][k] * L_matrix[j][k];
                }

                if (i == j) {
                    L_matrix[i][j] = (corr_matrix[i][i] - sum).sqrt();
                    L_transpose[j][i] = (corr_matrix[i][i] - sum).sqrt();
                }
                else {
                    L_matrix[i][j] = (1.0 / L_matrix[j][j] * (corr_matrix[i][j] - sum));
                    L_transpose[j][i] = (1.0 / L_matrix[j][j] * (corr_matrix[i][j] - sum));
                }
            }
        }


        
        let corr_vec = correlations.clone()[1..].to_vec();

        let mut y_vec = vec![0.0_f64; corr_vec.len()];

        let mut ar_vec = vec![0.0_f64; corr_vec.len()];

        //Forward substitution to get the intermediate y_vec
        for i in 0..y_vec.len(){
            let divisor = L_matrix[i][i];
            y_vec[i] = corr_vec[i];
            
            for j in 0..i {
                y_vec[i] -= L_matrix[i][j]*y_vec[j];
            }
            y_vec[i]/=divisor;
        }

        //Backward substitution to solve for the actual ar_vec
        for i in ar_vec.len()..0{
            let divisor = L_transpose[i][i];
            ar_vec[i] = y_vec[i];
            
            for j in (i+1)..ar_vec.len() {
                ar_vec[i] -= L_transpose[i][j]*ar_vec[j];
            }
            ar_vec[i]/=divisor;
        }

        ar_vec


    }

    fn yule_walker_ar_coefficients_with_bic(raw_data_blocks: &Vec<Vec<f64>>) -> Background {

        //let total_sample_variance = raw_data_blocks.iter().flatten(|a| a.powi(2)).sum::<f64>()/((raw_data_blocks.len()-1) as f64);

        let mut bic: f64 = -f64::INFINITY;
        let mut bic_worse = false;

        

        //let correlations = Self::compute_autocorrelation_coeffs(raw_data_blocks);
        
        todo!();



    }
    

    //This adapts the produce_noise function
    fn undo_ar_correlation(connected: &Vec<f64>, ar_coeffs: Vec<f64>) -> Option<Vec<f64>> {

        if connected.len() <= ar_coeffs.len() {
            return None;
        }


        let mut filtered = vec![0.0_f64; connected.len()-ar_coeffs.len()];

        let l_c = ar_coeffs.len();

        let mut block = connected[l_c..].to_vec();

        for i in 0..l_c {
            for j in 0..block.len() {
                block[j] -= (ar_coeffs[i]*connected[l_c+j-(i+1)]);
            }
        }
        Some(block)
    }

    pub fn estimate_t_dist(decorrelated_data: &Vec<f64>) -> (f64, f64) {

        let total_sample_variance = decorrelated_data.iter().map(|a| a.powi(2)).sum::<f64>()/((decorrelated_data.len()-1) as f64);

        let mut error = true;
        
        let mut eps = 1e-6;

        let mut df_guess = 3.0f64;
        let mut sd_guess = (total_sample_variance*(df_guess-2.0)/df_guess).sqrt();

        while error {

            let mut dist = f64::INFINITY;

            df_guess = 3.0f64;
            sd_guess = (total_sample_variance*(df_guess-2.0)/df_guess).sqrt();

            error = false;
            while !error && dist.abs() > 1e-3 {

                let old_df_guess = df_guess;
                let old_sd_guess = sd_guess;

                let (d_sd, d_df) = Self::gradient_t_like(decorrelated_data, sd_guess, df_guess);

                sd_guess += eps*d_sd;
                df_guess += eps*d_df;

                dist = (d_sd).powi(2)+(d_df).powi(2).sqrt();

                //println!("Current guesses: {} {} {} {}", df_guess, sd_guess, dist, eps);

                if df_guess < 2.0 || sd_guess < 0. {
                    error = true;
                    eps /= 2.;
                }
            }
        }

        (sd_guess, df_guess)

    }

    fn gradient_t_like(decorrelated_data: &Vec<f64>, sd: f64, df:f64) -> (f64, f64) {

        let help_series = Self::helper_series(decorrelated_data, sd, df);

        let n = decorrelated_data.len() as f64;
        let d_sd = (-n+(df+1.0)*help_series)/sd;
        let mut d_df = 0.5*n*(digamma((df+1.0)/2.0)-digamma(df/2.)-1./df)
                  +0.5*(df+1.0)*help_series/df;

        for &data in decorrelated_data {
            d_df -= (1.0+data.powi(2)/(sd.powi(2)*df)).ln()*0.5;
        }


        (d_sd, d_df)

    }

    fn helper_series(decorrelated_data: &Vec<f64>, sd: f64, df:f64) -> f64 {

        let mut sum = 0.0f64;

        for &data in decorrelated_data {
            sum += 1.0/(sd.powi(2)*df/data.powi(2) + 1.);
        }

        sum
    }
    /*
    //Returns standard deviation and degrees of freedom, in that order
    pub fn estimate_t_dist(decorrelated_data: &Vec<f64>) -> (f64, f64) {

        let precision = 1e-3;
        let total_sample_variance = decorrelated_data.iter().map(|a| a.powi(2)).sum::<f64>()/((decorrelated_data.len()-1) as f64); 

        let mut old_guess_df = 2.5_f64;
        let mut guess_df = 8_f64;

        let mut diff = f64::INFINITY;
        
        let h = 1e-4;

        while (diff.abs() > precision) {
            old_guess_df = guess_df;

            let fx = Self::lnlike(decorrelated_data, total_sample_variance, guess_df);
            let dfx = Self::d_lnlike_d_df(decorrelated_data, total_sample_variance, guess_df, h);
            let ddfx = (Self::d_lnlike_d_df(decorrelated_data, total_sample_variance, guess_df+h, h)
                        -Self::d_lnlike_d_df(decorrelated_data, total_sample_variance, guess_df, h))/h;

            guess_df -= (dfx/(ddfx));
            if guess_df <= 2.0 {
                guess_df = 100_f64;
            }
            diff = guess_df-old_guess_df;
            println!("guess {}",guess_df);
        }

        let sd = (total_sample_variance*guess_df/(guess_df-2.0)).sqrt();

        (sd, guess_df)

    }

    
    fn lnlike(decorrelated_data: &Vec<f64>, total_variance: f64, df: f64) -> f64 {

        let df_only_terms = ln_gamma((df+1.)/2.)-ln_gamma(df/2.)-(total_variance*(df-2.)).ln()-PI.ln()/2.0;

        let mut data_terms = 0.0_f64;

        for &dat in decorrelated_data {
            data_terms -= ((df+1.)/2.)*(1.0+dat.powi(2)/(total_variance*(df-2.))).ln();
        }

        df_only_terms+data_terms
    }

    fn d_lnlike_d_df(decorrelated_data: &Vec<f64>, total_variance: f64, df: f64, h: f64) -> f64 {
        (Self::lnlike(decorrelated_data, total_variance, df+h)-Self::lnlike(decorrelated_data, total_variance, df))/h
    }

    */








}
