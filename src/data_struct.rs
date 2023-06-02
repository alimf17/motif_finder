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
use log::warn;

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
    fn process_fasta(fasta_file_name: &str, null_char: Option<char>) -> Vec<Option<usize>> {

        let file_string = fs::read_to_string(fasta_file_name).expect("Invalid file name!");
        let mut fasta_as_vec = file_string.split("\n").collect::<Vec<_>>();
        
        //We want to trim the whitespace from the bottom of the Fasta file
        //From the top can be dealt with by the user
        while fasta_as_vec.last()
              .expect("FASTA file should not be empty or all whitespace!")
              .chars().all(|c| c.is_whitespace()) {_ = fasta_as_vec.pop();}


        let mut base_vec: Vec<Option<usize>> = Vec::new();
       
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
                    base_vec.push(None);
                } else {
                    base_vec.push(Some(*((*GET_BASE_USIZE)
                            .get(&chara)
                            .expect(&(format!("Invalid base on line {}, position {}", line_pos+1, char_pos+1)))))); 
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
    fn process_data(data_file_name: &str, sequence_len: usize, fragment_length: usize, spacing: usize, is_circular: bool) -> (Vec<Vec<(usize, f64)>>, Background) {
        let file_string = fs::read_to_string(data_file_name).expect("Invalid file name!");
        let mut data_as_vec = file_string.split("\n").collect::<Vec<_>>();
       
        //let mut locs: Vec<usize> = Vec::with_capacity(data_as_vec.len());
        //let mut data: Vec<f64> = Vec::with_capacity(data_as_vec.len());

        let mut raw_locs_data: Vec<(usize, f64)> = Vec::with_capacity(data_as_vec.len());
        let mut data_iter = data_as_vec.iter();

        let first_line = data_iter.next().expect("FASTA file should not be empty!");

        let header = first_line.split(" ").collect::<Vec<_>>();

        if (header[0] != "loc") || (header[1] != "data") {
            panic!("Data file is not correctly formatted!");
        }


        //This gets us the raw data and location data, paired together and sorted
        for (line_num, line) in data_iter.enumerate() {

            let mut line_iter = line.split(" ");

            let loc: usize = line_iter.next().expect(format!("Line {} is empty!", line_num).as_str())
                                      .parse().expect(format!("Line {} does not start with a location in base pairs!", line_num).as_str());

            let data: f64 = line_iter.next().expect(format!("Line {} does not have data after location!", line_num).as_str())
                                     .parse().expect(format!("Line {} does not have data parsable as a float after its location!", line_num).as_str());

            raw_locs_data.push((loc,data));

        }

        raw_locs_data.sort_by(|(a, _), (b, _)| a.cmp(b));

        let start_loc: usize = raw_locs_data[0].0;
        let last_loc: usize = raw_locs_data.last().unwrap().0;


        if last_loc > sequence_len {
            panic!("This data has sequence locations too large to correspond to your sequence!");
        }

        let mut hits_end: bool = last_loc == sequence_len;
        let mut zero_indexed: bool = start_loc == 0;

        if hits_end && zero_indexed {
            //Same panic as overrunning the sequence, because it overruns the sequence by one
            panic!("This data has sequence locations too large to correspond to your sequence!");
        }

        if (!hits_end) && (!zero_indexed) {
            warn!("Program is assuming that your locations are zero indexed, but doesn't have evidence of this!");
            zero_indexed = true;
            hits_end = last_loc == (sequence_len-1);
        }

        if !zero_indexed {

            for point in raw_locs_data.iter_mut() {
                (*point).0 -= 1;
            }
        }

        //Compress all data so that locations are unique by taking the mean of the data

        let mut refined_locs_data: Vec<(usize, f64)> = Vec::with_capacity(raw_locs_data.len());

        let mut i: usize = 0;

        //Remember, raw_locs_data is sorted, so we will run into all same locations in one stretch
        //I'm using a while loop instead of a for loop because I will likely have runs of same location
        //data that come out to only producing a single data row
        while i < raw_locs_data.len() {

            let curr_loc = raw_locs_data[i].0;

            let mut to_next_unique = 0;

            //Find the run of data which is all the same location
            //Whether it's stopped by a new location or the end of the data
            while ((i+to_next_unique) < raw_locs_data.len()) 
                && (raw_locs_data[i+to_next_unique].0 == curr_loc) {
                to_next_unique+=1;
            }

            let mut sum_of_data: f64 = 0.0;

            for j in (i..(i+to_next_unique)) {
                sum_of_data += raw_locs_data[j].1;
            }

            //For each location, we want to push it onto the data one time, with the mean of its data points
            refined_locs_data.push((curr_loc, sum_of_data/(to_next_unique as f64))); 

            //We want to skip to the next unique location
            i += to_next_unique;
        }


        //Here, refined_locs_data ultimately has one data value for every represented location in the data,
        //with the data sorted by location in ascending order
 
        let mut start_gaps: Vec<usize> = Vec::new();

        for i in 0..(refined_locs_data.len()-1){

            let jump: usize = refined_locs_data[i+1].0-refined_locs_data[i].0; //We can guarentee this is fine because we sorted the data already
            if jump >= fragment_length {
                start_gaps.push(i);
            } 
        }

        let mut max_valid_run: usize = sequence_len;

        if start_gaps.len() >= 2 {
            max_valid_run = start_gaps.windows(2).map(|a| refined_locs_data[a[1]].0-refined_locs_data[a[0]].0).max().unwrap(); 
        }

        let num_blocks = if is_circular { 1_usize.max(start_gaps.len()) } else {start_gaps.len() + 1};

        let mut first_blocks: Vec<Vec<(usize, f64)>> = vec![Vec::with_capacity(max_valid_run); num_blocks];

        let mut remaining = true;
        let mut cut = !is_circular || (start_gaps.len() > 0); //If we have a completely uncut circular genome, we technically can't do inference
                                                                             //But there IS another set of cuts coming that will likely fix this: 
                                                                             //we will remove parts of the genome that are just noise
                                                                             //BUT, we still need to track cut, because if it's still true after that, we need to panic

        //This match statement handles our first block, which we need to treat slightly carefully
        //Once we isolate the first block, we can iterate on all but the last gap straightforwardly
        //And the last gap only needs to be handled specially if our genome is NOT circular
        first_blocks[0] = match (start_gaps.len(), is_circular) {

            (0, _) => {remaining = false; refined_locs_data.clone()}, //If there are no gaps, do not cut
            (1, true) => { //If there is only gap in a circle, cut at that gap
                let (end, start) = refined_locs_data.split_at(start_gaps[0]);
                let (mut rewrap, mut get_end) = (start.to_vec(), end.to_vec());
                get_end = get_end.into_iter().map(|(a,b)| (a+sequence_len, b)).collect(); //We want to treat this as overflow for interpolation purposes
                rewrap.clone().append(&mut get_end);
                remaining = false;
                rewrap},
            (_, false) => refined_locs_data[0..start_gaps[0]].to_vec(), //If there is a gap on a line, the first fragment should be the beginning to the gap
            (_, true) => { //If there are many gaps on a circle, glom together the beginning and the end fragments

                let mut rewrap = refined_locs_data[(*start_gaps.last().unwrap())..].to_vec();
                let mut get_end = refined_locs_data[..start_gaps[0]].to_vec();
                get_end = get_end.into_iter().map(|(a,b)| (a+sequence_len, b)).collect();//We want to treat this as overflow for interpolation purposes
                rewrap.clone().append(&mut get_end);
                rewrap},

        };
        

        if start_gaps.len() > 1 {
            for i in 0..(start_gaps.len()-1) {
                first_blocks[i+1] = refined_locs_data[start_gaps[i]..start_gaps[i+1]].to_vec();
            }
        }

        if remaining && !is_circular {
            first_blocks[num_blocks] = refined_locs_data[start_gaps[start_gaps.len()]..].to_vec(); 
        }


        //Now, we have finished first_blocks, which is a vec of Vec<(usize, f64)>s such that I can lerp each block according to the spacer

        let mut lerped_blocks: Vec<Vec<(usize, f64)>> = Vec::with_capacity(first_blocks.len());

        for block in first_blocks {
            lerped_blocks.push(Self::lerp(&block, spacing));
        }

        //Now, we have lerped_blocks, which is a vec of Vec<(usize, f64)>s such that all independent blocks are lerped according to the spacer

        //Sort data into two parts: kept data that has peaks, and not kept data that I can derive the AR model from
        //Keep data that has peaks in preparation for being synced with the sequence
        //Cut up data so that I can derive AR model from not kept data

        //This is a rough approximation of the maximum value of the transformed data
        //that can solely be accounted for by fluctuation when the sd is 1.
        //This is based on the result found here: http://www.gautamkamath.com/writings/gaussian_max.pdf
        //Though I decided to make the coefficient 2.0.sqrt() because I want to err on the side of cutting more
        //Numerical experiments lead me to believe that the true coefficient should be ~1.25 
        let peak_thresh = 2.0_f64.sqrt()*(raw_locs_data.len() as f64).ln().sqrt();

        let mut ar_blocks: Vec<Vec<(usize, f64)>> = Vec::with_capacity(lerped_blocks.len());
        let mut data_blocks: Vec<Vec<(usize, f64)>> = Vec::with_capacity(lerped_blocks.len());

        let data_zone: usize = fragment_length/spacing;

        for block in lerped_blocks {

            let poss_peak_vec: Vec<bool> = block.iter().map(|(_, b)| b.abs() > peak_thresh).collect();

            let mut next_ar_ind = 0_usize;
            let mut curr_data_start = 0_usize;

            let mut check_ind = 0_usize;

            while check_ind < block.len() {

                if poss_peak_vec[check_ind] {
                    curr_data_start = if (next_ar_ind + data_zone) > check_ind {next_ar_ind} else {check_ind-data_zone}; //the if should only ever activate on the 0th block
                    if curr_data_start > next_ar_ind { ar_blocks.push(block[next_ar_ind..curr_data_start].to_vec()); }

                    next_ar_ind = block.len().min(check_ind+data_zone+1);

                    while check_ind < next_ar_ind {
                        if poss_peak_vec[check_ind] {
                            next_ar_ind = block.len().min(check_ind+data_zone+1);
                        }
                        check_ind += 1;
                    }
                    data_blocks.push(block[curr_data_start..next_ar_ind].to_vec());
                } else {
                    check_ind += 1;
                }
            }
        }

        if ar_blocks.len() == 0 {
            panic!("Not enough null data to infer the background distribution! Make peak_thresh bigger to allow more null data!");
        }

        if data_blocks.len() == 0 {
            panic!("You have nothing that counts as a peak in this data! Make peak_thresh smaller to allow more data to count as data!");
        }

        if !cut { //This code is only run if the genome is circular and has no missing data

            let starts_data = (data_blocks[0][0].0 < ar_blocks[0][0].0);
            let ends_data = ((*(*data_blocks.last().unwrap()).last().unwrap()).0 > 
                             (*(*ar_blocks.last().unwrap()).last().unwrap()).0);
            match (starts_data, ends_data) {
                (true, true) => { //If both the beginning and end are data, glom the beginning onto the end in data
                    let mut rem_block = data_blocks.remove(0); 
                    rem_block = rem_block.into_iter().map(|(a,b)| (a+sequence_len, b)).collect();
                    let Some(end_vec) = data_blocks.last_mut() else {
                        panic!("Not enough null data to infer the background distribution! Make peak_thresh bigger to allow more null data!");
                    };
                    end_vec.append(&mut rem_block);

                    *end_vec = Self::lerp(&*end_vec, spacing);
                },
                (false, false) => { //If both the beginning and end are for AR inference, glom the beginning onto the end for AR inference

                    let mut rem_block = ar_blocks.remove(0);
                    rem_block = rem_block.into_iter().map(|(a,b)| (a+sequence_len, b)).collect();
                    let Some(end_vec) = ar_blocks.last_mut() else {
                        panic!("You have nothing that counts as a peak in this data! Make peak_thresh smaller to allow more data to count as data!");
                    };
                    end_vec.append(&mut rem_block);
                    *end_vec = Self::lerp(&*end_vec, spacing);
                },
                (true, false) => {
                    let first_force_data = data_blocks[0].iter().position(|(_, b)| b.abs() > peak_thresh).unwrap();
                    let last_data_place = data_blocks[0][first_force_data].0+sequence_len-data_zone;
                    let bleed_into_final_ar = last_data_place <= (*(*ar_blocks.last().unwrap()).last().unwrap()).0;
                    let bleed_into_last_data = last_data_place <= (*ar_blocks.last().unwrap())[0].0;

                    if bleed_into_last_data {
                        let mut rem_block = data_blocks.remove(0);
                        rem_block = rem_block.into_iter().map(|(a,b)| (a+sequence_len, b)).collect();
                        let mut fuse_ar = ar_blocks.pop().unwrap();
                        let Some(end_vec) = data_blocks.last_mut() else {
                            panic!("Not enough null data to infer the background distribution! Make peak_thresh bigger to allow more null data!");
                        };

                        end_vec.append(&mut fuse_ar);
                        end_vec.append(&mut rem_block);
                        *end_vec = Self::lerp(&*end_vec, spacing);
                    } else if bleed_into_final_ar {
                        let mut rem_block = data_blocks.remove(0);
                        rem_block = rem_block.into_iter().map(|(a,b)| (a+sequence_len, b)).collect();
                        let mut split = 1_usize;
                        let Some(end_vec) = ar_blocks.last_mut() else {
                            panic!("You have nothing that counts as a peak in this data! Make peak_thresh smaller to allow more data to count as data!");
                        };
                        while((*end_vec)[split].0 < last_data_place) {split+=1;}
                        let mut begin_fin_dat = end_vec.split_off(split);
                        begin_fin_dat.append(&mut rem_block);
                        begin_fin_dat = Self::lerp(&begin_fin_dat, spacing);
                        data_blocks.push(begin_fin_dat);
                    } //Don't need a final else. It basically boils down to "else: I don't care"
                    

                },

                (false, true) => {
                    let last_force_data = data_blocks.last().unwrap().len()-1-(data_blocks.last().unwrap().iter().rev().position(|(_, b)| b.abs() > peak_thresh).unwrap());
                    if data_blocks.last().unwrap()[last_force_data].0+data_zone > sequence_len {
                        
                        let last_data_place = data_blocks.last().unwrap()[last_force_data].0+data_zone-sequence_len;
                        let bleed_into_final_ar = last_data_place <= ar_blocks[0][0].0; 
                        let bleed_into_last_data = last_data_place <= (*((ar_blocks[0]).last().unwrap())).0;

                        if bleed_into_last_data {
                            let mut rem_block = data_blocks.remove(0);
                            rem_block = rem_block.into_iter().map(|(a,b)| (a+sequence_len, b)).collect();
                            let mut fuse_ar = ar_blocks.remove(0);
                            fuse_ar = fuse_ar.into_iter().map(|(a,b)| (a+sequence_len, b)).collect();
                            let Some(end_vec) = data_blocks.last_mut() else {
                                panic!("Not enough null data to infer the background distribution! Make peak_thresh bigger to allow more null data!");
                            };

                            end_vec.append(&mut fuse_ar);
                            end_vec.append(&mut rem_block);
                            *end_vec = Self::lerp(&*end_vec, spacing);
                        } else if bleed_into_final_ar {

                            let start_vec = &mut ar_blocks[0];
                            let mut split = start_vec.len()-2;
                            while((*start_vec)[split].0 < last_data_place) {split-=1;}
                            let mut begin_fin_dat = start_vec.drain(0..split).map(|(a, b)| (a+sequence_len, b)).collect::<Vec<_>>();
                            data_blocks.last_mut().unwrap().append(&mut begin_fin_dat);
                            *(data_blocks.last_mut().unwrap()) = Self::lerp(data_blocks.last().unwrap(), spacing);
                        } //Don't need a final else. It basically boils down to "else: I don't care"
                    
                    }
                },
            };

        }


        //Now, we have data_blocks and ar_blocks, the former of which will be returned and the latter of which will be processed by AR prediction

        ar_blocks.retain(|a| a.len() > data_zone);

        if ar_blocks.len() == 0 {
            panic!("Not enough null data to infer the background distribution! Make peak_thresh bigger to allow more null data!");
        }

        let ar_inference: Vec<Vec<f64>> = ar_blocks.iter().map(|a| a.iter().map(|(_, b)| *b).collect::<Vec<f64>>() ).collect();

        let background_dist = Self::yule_walker_ar_coefficients_with_bic(&ar_inference, data_zone);




        //Send off the kept data with locations in a vec of vecs and the background distribution from the AR model
        (data_blocks, background_dist)




    }



    //TODO: I need a function which marries my processed FASTA file and my processed data
    //      By the end, I should have a full All_Data instance 




    

    fn chop_file_name(name: &str) -> String {
        let re = Regex::new(r"\.\pL+$").unwrap();
        let piece_name = re.replace(name, "");
        String::from(piece_name)
    }

    fn yule_walker_ar_coefficients_with_bic(raw_data_blocks: &Vec<Vec<f64>>, max_order: usize) -> Background {

        let mut bic: f64 = f64::INFINITY;
        let mut bic_worse = false;

        let mut num_coeffs = 1usize;
        let mut coeffs: Vec<f64> = Vec::new();

        let data_len = raw_data_blocks.iter().map(|a| a.len()).sum::<usize>();
        let mut noises: Vec<f64> = Vec::with_capacity(data_len);

        let (mut sd, mut df) = (1.0_f64, 3.0_f64);

        while !bic_worse {

            let correlations = Self::compute_autocorrelation_coeffs(raw_data_blocks, num_coeffs);

            let new_coeffs = Self::compute_ar_coeffs(&correlations);

            for data_block in raw_data_blocks {
                let mut little_noise = Self::undo_ar_correlation(data_block, &coeffs);

                match little_noise {
                    Some(mut vec) => noises.append(&mut vec),
                    None => (),
                };
            }

            let (pre_sd, pre_df) = Self::estimate_t_dist(&noises);
            let lnlike = Self::lnlike(&noises, pre_sd, pre_df);

            let new_bic = Self::bic(lnlike, data_len, num_coeffs);

            bic_worse = new_bic >= bic; //This lets us choose the smallest model possible AND avoids infinite loops: we'll likely only be equal if we did the EXACT same calculation

            if !bic_worse {
                bic = new_bic;
                sd = pre_sd;
                df = pre_df;
                coeffs = new_coeffs;
                num_coeffs += 1;
            }


        }

        

        //let correlations = Self::compute_autocorrelation_coeffs(raw_data_blocks);
        
        Background::new(sd, df, &coeffs)


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

    

    //This adapts the produce_noise function
    fn undo_ar_correlation(connected: &Vec<f64>, ar_coeffs: &Vec<f64>) -> Option<Vec<f64>> {

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


    fn lnlike(decorrelated_data: &Vec<f64>, sd: f64, df: f64) -> f64 {
        let df_only_terms = ln_gamma((df+1.)/2.)-ln_gamma(df/2.)-sd.ln()-PI.ln()/2.0-df.ln()/2.0;
        let mut data_terms = 0.0_f64;
        for &dat in decorrelated_data {
            data_terms -= ((df+1.)/2.)*(1.0+dat.powi(2)/(sd.powi(2)*df)).ln();
        }
        df_only_terms+data_terms
    }

    fn bic(lnlike: f64, data_len: usize, num_coeffs: usize) -> f64 {

        (2.0+(num_coeffs as f64))*(data_len as f64).ln()-2.0*lnlike

    }

    fn lerp(data: &Vec<(usize, f64)>, spacer: usize ) -> Vec<(usize, f64)> {

        let begins = data[0].0;
        let ends = (*(data.last().unwrap())).0;

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

}
