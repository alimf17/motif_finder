use std::{f64, fs};
use std::collections::HashMap;
//use std::time::{Duration, Instant};

use core::f64::consts::PI;

use crate::waveform::{Waveform, WaveformDef, Background};
use crate::sequence::{Sequence, BP_PER_U8};
use crate::base::{BPS, BASE_L};

use statrs::distribution::{Continuous, StudentsT};
use statrs::function::gamma::*;
use statrs::statistics::*;

use once_cell::sync::Lazy;


use argmin::argmin_error;
use argmin::core::*;
use argmin::core::Error as ArgminError;
use argmin::solver::neldermead::NelderMead;

use serde::{Serialize, Deserialize};

use log::warn;

use regex::Regex;

const DATA_SUFFIX: &str = "data.json";
const MAD_ADJUSTER: f64 = 1.4826;

const ILLEGAL_FASTA_FILE_PANIC: &str = "FASTA file must be alternating pairs of lines: first in pair should start with '>', second should be the bases";

static GET_BASE_USIZE: Lazy<HashMap<char, usize>> = Lazy::new(|| {
    let mut map: HashMap<char, usize> = HashMap::new();
    for i in 0..BASE_L {
        _ = map.insert(BPS[i], i);
    }
    map
});

#[derive(Serialize, Deserialize)]
pub struct AllData {

    seq: Sequence,
    data: WaveformDef, 
    background: Background

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

        if (p[0] < 0.0) || (p[1] < 2.0) {
            return Ok(f64::INFINITY);
        }

        Ok(-self.lnlike(p[0], p[1]))
    }
}


impl AllData {
  
    //TODO: I need to create a master function which takes in a fasta file and a IPOD/ChIP-seq data set 
    //      with a spacer and boolean indicating circularity, and outputs an AllData instance

    pub fn create_inference_data(fasta_file: &str, data_file: &str, output_dir: &str, is_circular: bool, 
                                 fragment_length: usize, spacing: usize, use_ar_model: bool, null_char: &Option<char>) -> (Self, String) {


        let file_out = format!("{}/{}_{}_{}_{}", output_dir,&Self::chop_file_name(fasta_file),&Self::chop_file_name(data_file),spacing,DATA_SUFFIX);

        println!("formatted");
        let check_initialization = fs::read_to_string(file_out.as_str());

        match check_initialization {
            Ok(try_json) => return (serde_json::from_str(try_json.as_str()).expect("Provided data file not in proper format for inference!"), file_out),
            Err(_) => (), //We just want the full preprocessing to try to continue if we get an error from read_to_string
        };

        println!("checked if initialized and isn't");
        let pre_sequence = Self::process_fasta(fasta_file, *null_char);

        println!("processed fasta");
        let sequence_len = pre_sequence.len();

        let (pre_data, background) = Self::process_data(data_file, sequence_len, fragment_length, spacing, is_circular, use_ar_model);
 
        println!("processed data");

        let (seq, data) = Self::synchronize_sequence_and_data(pre_sequence, pre_data, sequence_len, spacing); 

        println!("synchronized fasta and data");
        let full_data: AllData = AllData {seq: seq, data: data, background: background};

        let wave = full_data.validated_data(); //This won't be returned: it's just a validation check.

        println!("validated data");
        //This probably isn't necessary, but helps ensure that the validation check doesn't get optimized away by the compiler
        assert!(spacing == wave.spacer(), "Somehow, the data waveform got mangled");
        
        let json = serde_json::to_string(&full_data);

        match json {

            Ok(j_str) => fs::write(file_out.as_str(), j_str).expect("Your path doesn't exist"),
            Err(error) => unreachable!("Something deeply wrong in the data has occurred. It shouldn't be possible to get here. {:?}", error),
        };

        (full_data, file_out)


    }

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

    pub fn seq(&self) -> &Sequence {
        &self.seq
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
    fn process_fasta(fasta_file_name: &str, null_char: Option<char>) -> Vec<Option<usize>> {

        let file_string = fs::read_to_string(fasta_file_name).expect("Invalid FASTA file name!");
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

                if Some(chara) == null_char {
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
    fn process_data(data_file_name: &str, sequence_len: usize, fragment_length: usize, spacing: usize, is_circular: bool, is_ar: bool) -> (Vec<Vec<(usize, f64)>>, Background) {
        let file_string = fs::read_to_string(data_file_name).expect("Invalid data file name!");
        let mut data_as_vec = file_string.split("\n").collect::<Vec<_>>();
       
        if data_as_vec.last().unwrap() == &"" {
            _ = data_as_vec.pop();
        }

        //let mut locs: Vec<usize> = Vec::with_capacity(data_as_vec.len());
        //let mut data: Vec<f64> = Vec::with_capacity(data_as_vec.len());

        let mut raw_locs_data: Vec<(usize, f64)> = Vec::with_capacity(data_as_vec.len());
        let mut data_iter = data_as_vec.iter();

        let first_line = data_iter.next().expect("Data file should not be empty!");

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

        println!("Data sorted");
        let start_loc: usize = raw_locs_data[0].0;
        let last_loc: usize = raw_locs_data.last().unwrap().0;


        if last_loc > sequence_len {
            panic!("This data has sequence locations too large to correspond to your sequence!");
        }

        let hits_end: bool = last_loc == sequence_len;
        let mut zero_indexed: bool = start_loc == 0;

        if hits_end && zero_indexed {
            //Same panic as overrunning the sequence, because it overruns the sequence by one
            panic!("This data has sequence locations too large to correspond to your sequence!");
        }

        if (!hits_end) && (!zero_indexed) {
            warn!("Program is assuming that your locations are zero indexed, but doesn't have evidence of this!");
            zero_indexed = true;
        }

        if !zero_indexed {

            for point in raw_locs_data.iter_mut() {
                (*point).0 -= 1;
            }
        }

        println!("Normalized data location");

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

            for j in i..(i+to_next_unique) {
                sum_of_data += raw_locs_data[j].1;
            }

            //For each location, we want to push it onto the data one time, with the mean of its data points
            refined_locs_data.push((curr_loc, sum_of_data/(to_next_unique as f64))); 

            //We want to skip to the next unique location
            i += to_next_unique;
        }

       
        let mut dat = Data::new(refined_locs_data.iter().map(|&(_, a)| a).collect::<Vec<f64>>());

        let median = dat.median();

        dat = Data::new(refined_locs_data.iter().map(|&(_, a)| (a-median).abs()).collect::<Vec<f64>>());

        let mad_as_sd = dat.median()*MAD_ADJUSTER;

        for (_, dat) in refined_locs_data.iter_mut() {
            *dat = (*dat-median)/mad_as_sd;
        }

        println!("refined data values");
        //Here, refined_locs_data ultimately has one data value for every represented location in the data,
        //with the data sorted by location in ascending order
 
        let mut start_gaps: Vec<usize> = Vec::new();

        for i in 0..(refined_locs_data.len()-1){

            let jump: usize = refined_locs_data[i+1].0-refined_locs_data[i].0; //We can guarentee this is fine because we sorted the data already
            if jump >= 2*(fragment_length)+5 { //We need the times 2 because the ar correlations can be at least that big. We then add 5 because we want to leave some buffer: there's a division and then multiplication by 6 for the kernels and we don't want to ever have a sequence block with too small a number of spots
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
        let cut = !is_circular || (start_gaps.len() > 0); //If we have a completely uncut circular genome, we technically can't do inference
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
                let (rewrap, mut get_end) = (start.to_vec(), end.to_vec());
                get_end = get_end.into_iter().map(|(a,b)| (a+sequence_len, b)).collect(); //We want to treat this as overflow for interpolation purposes
                rewrap.clone().append(&mut get_end);
                remaining = false;
                rewrap},
            (_, false) => refined_locs_data[0..start_gaps[0]].to_vec(), //If there is a gap on a line, the first fragment should be the beginning to the gap
            (_, true) => { //If there are many gaps on a circle, glom together the beginning and the end fragments

                let rewrap = refined_locs_data[(*start_gaps.last().unwrap())..].to_vec();
                let mut get_end = refined_locs_data[..start_gaps[0]].to_vec();
                get_end = get_end.into_iter().map(|(a,b)| (a+sequence_len, b)).collect();//We want to treat this as overflow for interpolation purposes
                rewrap.clone().append(&mut get_end);
                rewrap},

        };
        
        println!("Cut out invalid data");

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

        //let peak_thresh = 1.0_f64*(raw_locs_data.len() as f64).ln().sqrt();

        let mut ar_blocks: Vec<Vec<(usize, f64)>> = Vec::with_capacity(lerped_blocks.len());
        let mut data_blocks: Vec<Vec<(usize, f64)>> = Vec::with_capacity(lerped_blocks.len());

        let data_zone: usize = fragment_length/spacing;
        
        for block in lerped_blocks {

            let poss_peak_vec: Vec<bool> = block.iter().map(|(_, b)| b.abs() > peak_thresh).collect();
   

            let mut next_ar_ind = 0_usize;
            let mut curr_data_start: usize;

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

        println!("sorted data data away from background inference data");

        if ar_blocks.len() == 0 {
            panic!("Not enough null data to infer the background distribution! Make peak_thresh bigger to allow more null data!");
        }

        if data_blocks.len() == 0 {
            panic!("You have nothing that counts as a peak in this data! Make peak_thresh smaller to allow more data to count as data!");
        }

        if !cut { //This code is only run if the genome is circular and has no missing data

            let starts_data = data_blocks[0][0].0 < ar_blocks[0][0].0;
            let ends_data = (*(*data_blocks.last().unwrap()).last().unwrap()).0 > 
                             (*(*ar_blocks.last().unwrap()).last().unwrap()).0;
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
                        while (*end_vec)[split].0 < last_data_place {split+=1;}
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
                            while (*start_vec)[split].0 < last_data_place {split-=1;}
                            let mut begin_fin_dat = start_vec.drain(0..split).map(|(a, b)| (a+sequence_len, b)).collect::<Vec<_>>();
                            data_blocks.last_mut().unwrap().append(&mut begin_fin_dat);
                            *(data_blocks.last_mut().unwrap()) = Self::lerp(data_blocks.last().unwrap(), spacing);
                        } //Don't need a final else. It basically boils down to "else: I don't care"
                    
                    }
                },
            };

        }

        println!("Ensured that cutting happened");


        //Now, we have data_blocks and ar_blocks, the former of which will be returned and the latter of which will be processed by AR prediction

        ar_blocks.retain(|a| a.len() > data_zone);

        if ar_blocks.len() == 0 {
            panic!("Not enough null data to infer the background distribution! Make peak_thresh bigger to allow more null data!");
        }

        let ar_inference: Vec<Vec<f64>> = ar_blocks.iter().map(|a| a.iter().map(|(_, b)| *b).collect::<Vec<f64>>() ).collect();


        let background_dist = if is_ar {Self::yule_walker_ar_coefficients_with_bic(&ar_inference, data_zone.min(30))} else { 
            let (sd, df) = Self::estimate_t_dist(&(ar_inference.concat()));
            Background::new(sd, df, None)
        } ;
        println!("inferred background dist");

        let trimmed_data_blocks: Vec<Vec<(usize, f64)>> = data_blocks.iter()
                                                                     .map(|a| Self::trim_data_to_fulfill_data_seq_compatibility(a, spacing)).collect();

        //Send off the kept data with locations in a vec of vecs and the background distribution from the AR model
        (trimmed_data_blocks, background_dist)




    }



    //TODO: I need a function which marries my processed FASTA file and my processed data
    //      By the end, I should have a Sequence and WaveformDef, which I can use in my public function to create an AllData instance
    fn synchronize_sequence_and_data(pre_sequence: Vec<Option<usize>>, pre_data: Vec<Vec<(usize, f64)>>, sequence_len: usize, spacing: usize) -> (Sequence, WaveformDef) {
        let mut sequence_blocks: Vec<Vec<usize>> = Vec::with_capacity(pre_data.len());
        let mut start_data: Vec<f64> = Vec::with_capacity(pre_data.iter().map(|a| a.len()).sum::<usize>());

        let mut i = 0_usize;

        while i < pre_data.len() {

            let mut no_null_base = true;


            let mut bp_ind = pre_data[i][0].0;
            let bp_prior = bp_ind;
            let min_target_bp = (*(pre_data[i].last().unwrap())).0+1;//We include the +1 here because we want to include the bp corresponding to the last location

            let target_bp = if ((min_target_bp-bp_prior) % BP_PER_U8) == 0 {min_target_bp} else {min_target_bp+BP_PER_U8-((min_target_bp-bp_prior) % BP_PER_U8)};

            let mut bases_batch: Vec<usize> = Vec::with_capacity(pre_data[i].len()*spacing);
            let mut float_batch: Vec<f64> = pre_data[i].iter().map(|&(_,b)| b).collect();


            
            while no_null_base && (bp_ind < target_bp){ 
                match pre_sequence[(bp_ind % sequence_len)] { //I don't need to explicitly check for circularity: process_data handled this for me already
                    Some(bp) => bases_batch.push(bp),
                    None => no_null_base = false,
                };
                bp_ind+=1;
            }

            if no_null_base {
                sequence_blocks.push(bases_batch);
                start_data.append(&mut float_batch);
            } else {
                //This gives the 1-indexed position of the null base in vim
                warn!("You have a null base in position {} of your sequence in a position with data relevant to a possible peak. We are discarding this data for inference purposes.",bp_ind);
            }
         



            i += 1;
        }

        let seq = Sequence::new(sequence_blocks);
        let wave = Waveform::new(start_data, &seq, spacing);

        let wave_ret = WaveformDef::from(&wave);

        (seq, wave_ret)


    }

    fn chop_file_name(name: &str) -> String {
        
        let named_split = name.split("/").collect::<Vec<_>>();

        let named = if named_split[named_split.len()-1] == "" {named_split[named_split.len()-2]} else {named_split[named_split.len()-1]};

        let re = Regex::new(r"\.\pL+$").unwrap();
        let piece_name = re.replace(named, "");
        String::from(piece_name)
    }

    //This will give AR coefficients that are slightly different from R's equivalents. 
    //This is for two reasons: I select models off of BIC instead of AIC, giving a smaller number of AR 
    //coefficients in general, and I solve the Yule-Walker equations with Cholesky decomposition, not Levinson recursion
    //I use Cholesky decomposition because I usually have a clear cap on the order size--capping computational cost--
    //and it was what I saw first saw
    fn yule_walker_ar_coefficients_with_bic(raw_data_blocks: &Vec<Vec<f64>>, max_order: usize) -> Background {



        let data_len = raw_data_blocks.iter().map(|a| a.len()).sum::<usize>();

 
        let models: Vec<([f64; 3], Vec<f64>)> = (0..max_order).into_iter().map(|a| {
            let num_coeffs = a+1;
            let correlations = Self::compute_autocorrelation_coeffs(raw_data_blocks, num_coeffs);

            println!("correlations {}", a);
            let new_coeffs = Self::compute_ar_coeffs(&correlations);
            println!("coefficients {}", a);

            let mut noises: Vec<f64> = Vec::with_capacity(data_len);
            
            for data_block in raw_data_blocks {
                let little_noise = Self::undo_ar_correlation(data_block, &new_coeffs);

                match little_noise {
                    Some(mut vec) => noises.append(&mut vec),
                    None => (),
                };
            }
            println!("filters {}", a);

            let (pre_sd, pre_df) = Self::estimate_t_dist(&noises);
            println!("estimatess {}", a);
            let lnlike = Self::lnlike(&noises, pre_sd, pre_df);

            let new_bic = Self::bic(lnlike, data_len, num_coeffs);
            println!("bics {}", a);
            ([pre_sd, pre_df, new_bic], new_coeffs)
        }).collect();

        let ([sd, df, _], coeffs) = models.iter().min_by(|([_, _, a], _),([_, _, b], _)| a.total_cmp(b)).unwrap(); 
        
        println!("sorted");


        //let correlations = Self::compute_autocorrelation_coeffs(raw_data_blocks);
        
        Background::new(*sd, *df, Some(coeffs))


    }


    fn compute_autocorrelation_coeffs(data: &Vec<Vec<f64>>, mut num_coeffs: usize) -> Vec<f64>{

        let min_data_len = data.iter().map(|a| a.len()).min().expect("Why are you trying to get autocorrelations from no data?");

        if num_coeffs >= min_data_len {
            num_coeffs = min_data_len-1;
        }

        let mut coeffs = vec![0.0; num_coeffs+1];

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
        let mut l_matrix = vec![vec![0.0;correlations.len()-1]; correlations.len()-1];
        let mut l_transpose = vec![vec![0.0;correlations.len()-1]; correlations.len()-1];
        for i in 0..(correlations.len()-1) {
            for j in 0..=i {
                let mut sum = 0.0;
                
                for k in 0..j {
                    sum += l_matrix[i][k] * l_matrix[j][k];
                }

                if i == j {
                    l_matrix[i][j] = (corr_matrix[i][i] - sum).sqrt();
                    l_transpose[j][i] = (corr_matrix[i][i] - sum).sqrt();
                }
                else {
                    l_matrix[i][j] = 1.0 / l_matrix[j][j] * (corr_matrix[i][j] - sum);
                    l_transpose[j][i] = 1.0 / l_matrix[j][j] * (corr_matrix[i][j] - sum);
                }
            }
        }


        
        let corr_vec = correlations.clone()[1..].to_vec();

        let mut y_vec = vec![0.0_f64; corr_vec.len()];

        let mut ar_vec = vec![0.0_f64; corr_vec.len()];

        //Forward substitution to get the intermediate y_vec
        for i in 0..y_vec.len(){
            let divisor = l_matrix[i][i];
            y_vec[i] = corr_vec[i];
            
            for j in 0..i {
                y_vec[i] -= l_matrix[i][j]*y_vec[j];
            }
            y_vec[i]/=divisor;
        }

        //Backward substitution to solve for the actual ar_vec
        for looping in 0..ar_vec.len(){
            let i = ar_vec.len()-1-looping;
            let divisor = l_transpose[i][i];
            ar_vec[i] = y_vec[i];
            
            for j in (i+1)..ar_vec.len() {
                ar_vec[i] -= l_transpose[i][j]*ar_vec[j];
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


        let l_c = ar_coeffs.len();

        let mut block = connected[l_c..].to_vec();

        for i in 0..l_c {
            for j in 0..block.len() {
                block[j] -= ar_coeffs[i]*connected[l_c+j-(i+1)];
            }
        }
        Some(block)
    }

    pub fn estimate_t_dist(decorrelated_data: &Vec<f64>) -> (f64, f64) {

        let total_sample_variance = decorrelated_data.iter().map(|a| a.powi(2)).sum::<f64>()/((decorrelated_data.len()-1) as f64);

        let init_dfs = vec![2.1, 3.0, 25.0]; //df really close to 2, df sanely close to 2, and df above our "give up to normal" point

        let init_simplex: Vec<Vec<f64>> = init_dfs.iter().map(|&a| vec![(total_sample_variance*(a-2.0)/a).sqrt(), a]).collect();

        let cost = FitTDist { decorrelated_data };

        let solver = NelderMead::new(init_simplex).with_sd_tolerance(1e-6).unwrap();

        let executor = Executor::new(cost, solver);

        let res = executor.run().unwrap();

        let best_vec = res.state().get_best_param().unwrap();

        let sd_guess = best_vec[0];
        let df_guess = best_vec[1];

        (sd_guess, df_guess)

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

    //This function exists to help fulfill the invariant that each sequence block 
    //must have 1+floor((sequence block bp-1)/spacer) data points. This function only ever sees non-trivial
    //use if the spacer is less than BP_PER_U8, which means that there is a lot of data already.
    //And by the design of my sanitization, the ends of my sanitized data are going to be non-peaky anyway,
    //either because there's no data there or because I decided the data there was far enough away
    //from a peak that it can be cut
    fn trim_data_to_fulfill_data_seq_compatibility(data:  &Vec<(usize, f64)>, spacer: usize) -> Vec<(usize, f64)> {

        
        let mut trimmed_data = data.clone();
        if spacer >= BP_PER_U8 {
            return trimmed_data;
        }

        let mut cover_bp = (((trimmed_data.len()*spacer) as f64)/(BP_PER_U8 as f64)).ceil()*(BP_PER_U8 as f64);

        //If spacer is 2 or 3 for BP_PER_U8 = 4, this should only drop one element. 
        //If spacer is 1, it could drop up to three elements
        while cover_bp >= ((trimmed_data.len()+1)*spacer) as f64 {
            let _ = trimmed_data.pop();
            cover_bp = (((trimmed_data.len()*spacer) as f64)/(BP_PER_U8 as f64)).ceil();
        }

        trimmed_data

    }

}

#[cfg(test)]
mod tests{

    use super::*;

    #[test]
    fn test_t_fit() {

        let t_data: Vec<f64> = vec![-1.74673575356551, -0.716212161115709, 1.1419346952728, 0.961701458348633, -0.835426104008679, -15.4178695019772, -0.342190084777164, -4.11182532534328, 1.37115600780661, 0.301607939800756, -0.205377589776568, -0.90209301150046, 0.584243057984495, 0.697970662639408, -0.723693672817299, 1.18182927997903, 4.25363386814657, 0.187051361007695, 1.34583720830727, -0.460493488852858, 0.0300804217394068, -0.994828930297786, 0.780050625446081, -1.73664005166326, -0.893075810581782, 1.27353063256346, -1.22813937341447, 0.792676733473614, 0.43820159807829, 0.457086692870032, -0.783977561380295, -0.273952858933103, 0.206600429356068, 0.34377789912316, -0.444962614830732, 0.0698021332068082, -1.41249423804385, -0.0453454688707125, -1.5626895521187, 2.04258042988626, 0.654428487058633, 0.547611739714776, -1.38462762378615, -0.551067772051512, 3.09977603643769, -0.151164906877533, -1.34552308476782, -0.506646993773148, 1.6452955064922, -1.26897824163202, -4.46699807541815, -0.433755781591723, 1.34538697216528, -2.00943238236091, 0.587179468171735, 0.172470687149357, -0.87384912247087, 2.93987614622441, -0.504826827297448, -1.96941000463708, -1.5851004115494, 3.28083530374256, 0.10194852088865, 0.515452551154593, -0.372851236878127, 0.94656723665731, 0.840093215140395, 0.333572971152078, 3.17652973042094, -1.18925294265308, -2.5950350935876, 0.662659646427808, -1.15503221143739, 4.84564939097141, 1.37461548683109, -1.92334860070084, 0.478081587498505, -0.917798312000448, -2.04790327347322, 0.127808406780702, -0.820024530281208, -0.56174887096964, -1.36229884465125, -1.12341246610811, -0.844798404382191, 0.43337186133017, -0.217249728310736, -0.0250667905299137, 0.782726091927995, 0.911112354313892, -1.59029614336338, 2.0314713987557, -0.786260582568002, 1.07332826799846, -0.586612712262912, -0.641609719438793, 0.34656649111713, 4.09963407263855, 0.203497812856241, 0.772252131191766, 0.202517672604001, 2.1972579904831, 0.329668691358791, 0.674870002993557, -1.4849519633224, 3.22438881148741, -0.895474094617742, 0.842347137655452, -0.155784952089526, -1.59765416271067, -2.12902113919246, 0.954190340852104, 0.799460274953457, -2.56544365616153, 0.062540724771491, 0.128062098510196, -4.20073441692631, -0.608944454114285, 1.42724366293682, 3.56153326332633, 0.146255024285299, -7.62388551008954, 0.181090441589188, 1.07134499921983, 0.159308103522203, -0.961716570383767, -0.48445102822011, -1.83700772974534, 0.277333529277746, 1.62462306526431, 2.21245291351148, 1.13659941360372, -1.04565281642519, -0.0149955614332196, 1.64280653750695, -2.78130813619043, -0.275911212294598, -0.970396221516611, 0.840809704936196, -0.126276667808504, 0.727477706295465, -0.496782198322523, 0.920945232682381, -0.560369527354825, 0.543249330089808, -0.0171776036506103, 0.431788738998995, 0.421171199579898, 0.0557133908944791, 0.307716580201441, -0.497288888798293, -0.412797777480554, -0.581465986498314, -2.48781256043586, 0.535127009825743, 0.704500172339479, -1.04619946971893, -0.975608668950808, -2.08162301987992, -0.437092803640378, 0.959111096507471, -0.0262197789507664, -1.02356899368407, 0.183402072104488, -0.387637111531151, 1.1746398409871, 0.631380499388218, -0.446814386019508, 0.264215044913558, 2.83165518511906, 0.354771690206051, -0.811830905711359, -0.046497815244381, 0.00572877001089766, 0.524031652366584, 0.0333513156482017, 2.15076232128801, 0.219537372607845, 0.470967309061242, -0.801064463394317, -1.64519985854295, -0.46025735420641, 1.86201956599003, -1.0394624230191, -1.83811655838106, -0.282814012048462, -1.51978378736349, -1.10481341054229, -0.820730496742721, 1.8743553859607, 1.5851148531285, 2.86416670792911, 0.722320656599473, 0.995976415852503, -1.55792953787465, 0.443346477799279, -0.0164441261152659, 0.534815817801359, -1.38283064617859, -0.188480007555302, 0.275426595308692, 0.935632902065095, -0.641915881961345, 0.927932005478441, 0.139749586144445, -0.0912125581181163, -1.17218700022986, 0.494495153243435, 0.462192061139227, 0.138627905298802, -0.614383827540185, -0.230606719891999, -0.288603434350296, -0.511543363561587, -4.28627120153408, -0.742992660909253, -4.03722528601833, -0.582932356918311, -1.77804857570474, 1.53646755024283, 1.28507723800106, 0.337129607204999, 0.174350026230671, -1.8813529247703, -1.19099948062733, 1.12731982279498, 0.194991852183234, -0.437521269797305, -2.15260591025704, -1.03348452807266, -0.0921897558663505, 0.512498857260962, 1.58383383885318, -2.18018985639833, 0.515355823753861, 1.63482870824571, -0.672520804228583, 0.162932606662963, 1.61274405933403, -2.1215109409226, 1.24019736869035, 0.967692455194374, -0.0495410621281779, 0.798997677535636, 1.80733026483821, -0.21301931984333, -5.38434826340137, -0.818882448970119, -3.48964200171198, -14.9772868701973, 3.36713691949314, 2.25474490532692, -1.48566277162739, 2.25581725141627, -4.22213866181079, -0.228127621488178, 3.14507296753417, -0.396553339751026, 1.56287891226499, -0.40738895941158, 0.434639360551197, 1.44900129363455, -0.421199439478994, -1.07854452826977, 0.0578746657657618, 1.30258428907769, -1.8385414260468, -0.271747140197581, -0.174919721664282, 0.348926213966345, 1.69821914963413, 0.149928211717208, -0.510939251089885, -1.74083083681475, -1.05126925299071, -0.380237898334832, 0.0737540042024673, 0.295298631333461, 0.290695107838783, 0.274636954983808, -4.02706879685937, -1.50813530087038, -1.42499456792205, -1.38000804284457, 0.612693256068562, -0.774235462481855, -0.659952417042899, 0.523371291406965, -0.222037316566273, 0.0379764954255171, -0.603144909424826, -0.187862279123839, -1.86195656675779, -1.10069061498158, -0.505748315707019, 0.187797438328183, 3.01684664846919, -0.100535486434181, -0.745260267515785, 0.367842322961049, 1.31119186759396, 1.0152772427299, 0.590604835533575, -0.223752417299306, 0.243479341360219, -1.94863651542325, 1.53549737156324, 1.23225261375545, -0.349100069680047, 0.0706968532757556, -0.0645085019571481, -1.20563631750184, -0.489821854910099, 0.726190946282985, 1.02601032937587, 0.128668155371176, -0.552248596240967, 1.18330417652389, -0.325010964075305, -0.2472580459586, -1.92098820325846, -1.1169125620275, -1.18244297884563, -10.7113226361415, -1.26652095266941, -0.147943242960308, 1.2417008054995, 4.59771178316017, -3.34447450522792, -0.0851661085637802, -0.357891376949495, 0.696473153879502, -0.128216983204886, -0.739271803276655, 1.00420219713201, -3.00500812881333, -0.0316411051916937, 2.69321302445021, -1.65556214465242, 0.955357921033628, 0.0606954876009854, 0.0390007905458031, 4.82021017655767, -0.682579414023819, -0.960366131175079, -0.556479338498371, -1.45242895933646, 0.596168593539281, -7.91677908513323, 1.35939126835711, -0.460697836816571, 0.912254951518001, 2.326027861631, 1.62918574129847, -1.57161894968773, 0.626754484862087, 1.36847664705086, -3.13034573280537, 0.372463248324763, -0.795570220933577, -0.368188155462083, -0.95602302493361, -4.23322723163552, -3.56657815993771, 0.136019476706842, -1.10452678811085, -1.93846945688305, -1.5101578361396, -0.469446118976909, 1.15205145102958, 0.922046509245872, 0.359943393511745, -0.6265824067111, -1.58782796003766, 0.411916648361149, -0.977869399259474, 0.966486694122306, 0.235399066804392, -0.395464872619317, -0.492797194938103, 8.74962653453128, 5.0230140465103, 0.168313156057922, -2.39968341812687, 1.22653800177954, -0.416475258358083, -1.62924941421045, -0.559573852342494, -0.899003447091992, 0.314519301285714, -0.297425137299043, 2.66605310441356, 0.881761656260239, -0.479224654653184, 0.452797828932011, 0.195494116897998, -0.34755956745453, 1.34779858369339, 1.03163756301158, 1.21872632352012, -16.065424469021, 1.17465040935234, -0.232868101424964, 1.48061043147822, -8.09959084142855, 0.254460621671512, 0.34503704702956, -0.342847291251827, -4.02677906818036, 0.289439539937635, -0.728419317069381, -0.0873862296686149, -0.858446369722462, 0.607452642156903, -0.358063516765653, -0.938874477014561, 0.843638227654484, 0.233803293978078, -0.579060476995525, -0.116377255095641, -1.05561222038889, 3.77410071333684, -4.82726273613664, 2.04993940116503, 1.75065607173716, -1.19013952002075, 0.127457674135824, 0.252588956541844, 0.963057360883677, -0.618997449944483, 2.48659279484705, -0.416189491621265, -2.00216411912282, -0.677017193080277, 0.574404938035795, 3.3042134591326, 0.232860864998359, -2.08889167428848, 2.3821979929406, -0.937914426866838, -1.35166897711362, 0.79909109101246, 4.15108782043384, -0.441367350831788, -0.172454855810497, 0.140924337860944, -0.0780061519062443, -1.51657988679857, 0.877256792120846, 0.280281837163135, 2.16777048111879, 0.706506418751284, 0.57087491128625, -0.347334652470865, -0.505559200009588, 0.442920680856992, 1.75180770814366, 0.0600551600613354, -0.299402089711286, -0.0574117945313828, -0.585234657431769, -0.44163833436539, -0.105686294306999, 2.17928490258259, 0.271094522077331, 4.42192535078716, -0.271258052440317, -2.72322954950695, 0.69743405283341, 2.37804315601357, -0.217390212169295, -0.231925114150219, -2.13481549880803, 0.552375082358016, -0.401572088105497, 1.4857614267862, 0.230970686779611, -0.412353709403135, 1.25917344637959, 0.507378084912823, 0.0775143043651686, 2.18276903555154, -1.59922674583737, -1.80511958235206, 1.18075900190091, 1.4822440904096, -0.125796131785605, -4.61777569655889, 1.83180400709663, -0.574473359206843, -1.05469564572063, -0.658350365996412, -0.45347356858538, -0.695425413148897, 0.923448013949057, 0.0854343328369129, 1.54271363031215, -0.32560276386411, 3.95869470654606, 1.07285442614339];

        let (sd, df) = AllData::estimate_t_dist(&t_data);
        println!("{:?}", (sd, df));

        assert!(((sd-0.9859).abs() < 0.01) && ((df-2.28496661).abs() < 0.01), "parameters fitting off");
    }

    /* #[test]
    fn see_data() {


        /*
        let a = AllData::process_fasta("/Users/afarhat/Downloads/sequence.fasta", None);

        //There's a vector that we define below, but it takes up most of the page, since it was copy pasted
        //It was generated in R using a unity scale parameter and 2.5 degrees of freedom, with AR coefficients of 
        //0.9, -0.4, 0.03. The results I get give me a decent whack at the T distribution, and the three AR coefficient 
        //model even gives me 0.8887481074141186, -0.44659493235176806, 0.1416619773753163 for the AR coefficients.
        //But the BIC actually favored a single AR coefficient of 0.6346814683664462, though with similar T distribution success.
        //I suspect that the data just isn't long enough: it's "only" 1000 data points, and 
        //using ar from R gives similar results with the AR coefficients. 
        let dat = vec![vec![0.92030358445241, 1.13882193502168, -0.117210084078362, -0.807361601078728, -0.438976319932791, 0.268127549092788, -0.052278140706468, -0.097668502664112, -1.47144080768618, -1.33214313894382, -2.17597210917359, 0.572919562976975, 2.00048064323903, 2.05359733016391, -0.319394697044939, -1.59994751216035, 1.84918907421627, 2.1435024238699, -0.204044958336222, -1.49221275359722, 0.44822708430529, -0.274810113068454, -4.93838439350979, -4.754930877994, -0.96694136401746, -1.12585879058432, -0.181964824086987, 0.430037620784272, -0.44780509784776, 2.35937756512511, 1.81063614907778, -1.29702264945256, -3.40589392873406, 0.788658912135261, 2.13518843382043, 2.01948175887685, 0.614266739946927, 0.755670252073416, 1.335074198794, 1.25129965143572, 3.79133984475769, 1.76248528301837, -2.48779528723107, -2.16761002994477, -2.0578885650047, 3.78595983582816, 5.54010646417997, 1.48662662577973, -0.286418239896885, -1.60402218429414, -3.3323571446058, -2.23829669684377, -1.5496693651271, -1.1611033351847, -1.85457300117196, -2.37457691404281, -2.27492152660748, -0.719863937034595, -0.0263959684201648, 0.170874766907551, 0.92547585140182, 1.6748988347599, -0.447951289632964, 0.986119979724126, 0.330674880079694, 0.963049129491544, 0.177445151639323, -0.857208488357629, -0.467007735175719, 4.02753386087264, 3.98936512706117, 2.73765696914249, 1.19148890983396, 2.29421617548849, 1.99999739443913, 1.59292585508839, -0.782491166253844, 1.94297634165677, 1.21398885502754, 1.13427183552991, 0.437753448126084, -1.62096512795834, -3.7289629785395, -1.74035768520633, 0.676094595844811, -1.37268433517485, -1.49552374577999, -0.64855270074651, -4.22740287934142, -4.19905167759636, -0.680398276185732, 4.50197339941747, 4.34421884390742, -5.53528985862542, -6.40329876675413, -2.3471813820915, 0.442105670582741, 0.182952122974677, -0.567051827238743, -2.4072720533326, -1.65690202413685, 1.07930851005701, 3.77437322061755, 4.0521048474126, 1.12387151330085, -0.511121941808973, 0.854811329980915, -1.77381301708499, -2.22960111791772, -2.24186768090914, -0.33824515122749, 0.399161734912879, 1.15476529778078, 0.372694521178202, 0.806439034677838, 0.0509907543173711, 0.277746230739652, 0.236590873328464, 0.535151755328273, 0.816503816966147, 0.583603850232557, 0.522413071284131, -0.236063550226613, -0.816712085691182, -1.20640905139121, -3.25397777891827, -1.93539073321495, 0.228047352471594, -0.164419892576064, -1.27286723525435, -3.15459415400426, -2.77201324491975, -0.312049179376233, 0.707103432958397, -0.345516629618616, -0.419765743116912, -0.606006525500174, 0.786774766395147, 1.56928742705041, 0.6326541960028, 0.229492093487767, 2.83221501366845, 2.83094799099263, 0.610021043519263, -0.544891622064043, -0.643753667524667, 0.180910631725579, 0.437325602549168, 2.45267850086629, 2.2574451013196, 1.53471626797884, -0.249217507715209, -2.41565876963859, -2.48862175575569, 0.581046968433886, 0.406458787784514, -1.77938108942122, -2.02940909858835, -2.62230577669098, -2.70650640281147, -2.2685462215543, 0.836597174385689, 3.16427560661299, 5.3093194974799, 4.26009587691776, 2.80126317428491, -0.58155144686093, -1.07275221778199, -0.665262648146138, 0.347733778176803, -0.835947753094469, -1.09988437605543, -0.369658228558103, 1.01781581419214, 0.38898511095316, 0.859802532802686, 0.788512295711363, 0.286197048229631, -1.20422049912365, -0.680126746388365, 0.340352100486049, 0.680868879317883, -0.158146478740161, -0.635075539470715, -0.776486761998339, -0.96109562793401, -4.85971482805948, -4.75559235784913, -6.40220534569677, -4.58847166974754, -3.48945871093429, 0.0392772179300767, 2.57855606841942, 2.53743542028242, 1.42779779365498, -1.5339523965412, -3.06655269236792, -0.976162707910025, 0.497047920115144, 0.308290360699297, -2.10324863491103, -3.03481300616886, -1.97297329663293, -0.112349366288463, 2.18086433766167, -0.23166140488643, -0.568855656697248, 1.2809491093026, 0.700925814676067, 0.264320526449464, 1.6086906805472, -0.758389764569628, -0.0779000758477091, 1.2491990131757, 1.08314638713195, 1.27181281840868, 2.55617921694851, 1.61131123966082, -4.91848544993378, -5.81335847326639, -6.70593111048839, -18.8348360438283, -11.076243829955, -0.381118057415875, 2.03678342736545, 3.90908224411287, -1.53011155477786, -3.10775741561244, 1.07740838271751, 1.77031382429637, 2.63196527857635, 1.28557651307011, 0.591981525612649, 1.54651301981518, 0.772436963501713, -0.984197023275923, -1.0904820499888, 0.738002362303196, -0.767672390676678, -1.29056769822753, -1.00722162292929, -0.0643763390993075, 2.00445206266964, 1.94946895507173, 0.439870693233837, -2.0646012330529, -3.02687457137971, -2.26538839855839, -0.816283762939789, 0.375992366969616, 0.887640091330601, 0.898627577505309, -3.56208024262775, -5.04682934749756, -4.5153500562936, -3.53195376178861, -0.911329987448676, -0.317111448159027, -0.686779338260212, 0.00477456661292525, 0.0474581852446739, 0.0581756553527472, -0.569626856706835, -0.72237296674375, -2.28249622448385, -2.88307683602075, -2.20919016733446, -0.715717864599052, 3.16988433218321, 2.9723718533503, 0.640449131688225, -0.14960566989417, 1.00953826761443, 2.00291742549119, 1.98492704133305, 0.791801097732398, 0.222217035550891, -2.00581381128042, -0.334867839877523, 1.73986359344438, 1.29054988603249, 0.526200278130919, -0.0549522982489849, -1.42685700059732, -1.73622622780417, -0.267318427449307, 1.43710852577524, 1.47690641271449, 0.194104212068498, 0.810348658072999, 0.370968335744429, -0.231702880655758, -2.25359767040422, -3.04134026305666, -3.02516123385461, -12.2850395715001, -11.2042322813694, -5.40849130460833, 0.487200356754749, 6.86346165764169, 2.47550610480946, -0.587979266589305, -1.6713713090744, -0.498304134507453, 0.0742184413704876, -0.523294691512467, 0.488600474187374, -2.3537232721986, -2.36113108059075, 1.52434237502359, 0.590186726939155, 0.80595509285171, 0.59571065164257, 0.270463941691607, 4.84952211620864, 3.59167623343659, 0.340447550685146, -1.54126137277012, -2.86799292810053, -1.35830706612259, -8.03829611458646, -5.41779219616468, -2.16214157951388, 0.892295524983789, 3.83141670003702, 4.65567831395285, 1.11269371860454, -0.119149992973854, 0.955834515351163, -2.18905386024164, -1.9835935318224, -1.67650782001654, -1.14927939655526, -1.3792791597814, -5.06516195141717, -7.60799063419726, -4.72648568829726, -2.467122512442, -2.49652516178786, -2.9119760474208, -2.16562817231374, 0.292880760061899, 1.96453118080445, 1.94590030704158, 0.347701820106399, -1.9943205093344, -1.46367552887112, -1.48701811690653, 0.153810985174844, 0.924725934358231, 0.330653530725958, -0.560485061472788, 8.14067034494613, 12.5837309874727, 8.22058835496073, 0.209573816697079, -1.4955689755532, -1.59969921238597, -2.46446390063563, -2.18257874722677, -1.92552973571341, -0.619359878984718, -0.150114496516727, 2.72092811707099, 3.37206196386128, 1.46275643109807, 0.502081674887892, 0.163426910773712, -0.357425324780405, 0.975807477328182, 2.05773722984232, 2.66964407970353, -14.4565654649049, -12.8423840240482, -5.92829821471527, 0.848398683906602, -5.34998426074795, -5.07773363300574, -2.05947755785923, -0.325783167945275, -3.64852490517759, -2.92570393427986, -1.91191639088858, -0.747285154911719, -0.854007570815973, 0.0804023986605571, 0.0334831157078864, -0.966520859466166, -0.0372117201884032, 0.587925583066218, -0.064038389944552, -0.410298390877877, -1.38162764870917, 2.6928340341514, -1.86336999764305, -0.745676039835554, 1.90568065596692, 0.767342386354405, 0.0334232782729526, 0.0329033721247472, 1.0023213560774, 0.270933120023467, 2.33049116160095, 1.60294894649253, -1.48357853831922, -2.58350272131656, -1.10922762742664, 3.29480232682567, 3.56436892847262, -0.232157398216184, 0.846352832961757, 0.0235971499393957, -1.67593739729935, -0.693300841543859, 4.19819993646229, 3.56405480668283, 1.33511547037282, 0.0428523366172192, -0.466563592899391, -1.91357459094372, -0.65704933247023, 0.440370366530435, 2.76951630625596, 3.00321146779537, 2.17916982079561, 0.495719088314716, -0.840983604810727, -0.44687710417468, 1.70088332896018, 1.74437148965104, 0.576772606265343, -0.185038544884187, -0.930147245644142, -1.18745225930348, -0.807885585769002, 1.89926436174256, 2.26796311417413, 5.67914984127379, 3.98976948988871, -1.3360580516914, -1.93055149440612, 1.29466311642128, 1.67994544882157, 0.704243998388495, -2.13413418629437, -1.59964492119767, -0.966471522713996, 1.19177099923384, 1.64216384753974, 0.559891208007671, 1.14196312454762, 1.3604533292288, 0.861933787092273, 2.4485870059795, 0.300541644584132, -2.48820889100537, -1.10538804765819, 1.4916946532569, 1.58423800847872, -3.82180099166054, -2.19676124919164, -0.975310946560743, -1.16842502769846, -1.38571134977648, -1.26250310070165, -1.31244641470074, 0.205676140505754, 0.757646332151339, 2.10295148060503, 1.27016532003506, 4.28339229230014, 4.48292990561765, 2.10562909543797, -2.71852949855216, 0.723910326172097, 2.44179494933801, 6.48289894798974, 4.69471299611186, 1.67744299489166, 1.3966035029669, 0.35936870269668, -1.22486652810243, 0.269995457628434, 0.759061909981033, 0.946641927025073, -1.72846034103023, -1.73818606278888, -0.567904098733486, 0.639322223001504, 0.688665499408402, 1.71224988361683, 1.17573556087929, -0.41575971995599, -2.41333500847634, -1.9350045617501, -1.88619779027742, -4.42719992370515, -2.59560242752011, -0.881805083016513, -1.28065489822725, -1.04963419365515, -0.506449452641461, -2.50843368933851, -0.668397124292493, 1.37657526425846, -1.55807299632588, -1.55946789943153, -1.12842232992949, -2.33363536729417, -2.01925011809299, -0.565919112099035, -0.0609487335389706, 0.317933373700252, -0.304417191953899, 0.983700975247276, 1.91452101421138, 1.18934685577676, 0.0143581321445323, 5.11618800145032, 5.27895214624172, 5.52692149213463, 2.11695955618689, -2.11740766158195, -5.25839644433757, -6.01815665282092, -1.82109088522659, -0.155593030300168, -1.43950271574513, -0.9188527465227, 0.815157605995218, 0.373297424278821, 0.199535493630113, 0.439568578819839, 0.156387149739652, -0.195906002002079, 0.242447953004853, -0.376141898640107, -2.22056282192827, -2.77698192302861, -1.9528087368393, -2.7175854999197, -1.37325274211328, 1.73178489393634, 39.3756188656999, 33.8909649888857, 14.4694535421062, 0.752712478816419, 15.4268099521774, 14.2158561610901, 6.78642948035382, -0.594434074904627, -3.21412649658063, -2.61233815128654, -1.35769241640445, 1.2852729899965, 1.90217731860234, 1.02137924636608, -2.0723925958159, -0.960204356007614, -0.232432636192525, -1.50204014499291, -1.66926512530075, 0.244202364048352, -5.04145650251787, -5.14174187549786, -3.72568346298302, -2.32346552146221, 0.304534764715309, -4.44045607257843, -4.57535110007076, -3.13198444723472, -2.77124551099526, -3.23772091612678, -0.554117999156476, 1.08014628105036, 0.0684690899019627, -0.565335019423998, 2.42761136922389, 1.11275508593899, -0.580558015119916, 0.15500967086521, -0.537243481976309, -0.728207123729817, 0.10547763314165, 1.66789563918996, 3.9608848018793, 2.81805754820624, -0.336439131244874, -2.29609377902806, -1.80844475512746, -0.0792687057484585, 0.177036360993458, 0.265469329917371, -0.142596638223382, -0.611260775935601, -1.80995528248858, -0.693267766393418, -0.0122818812108099, 1.936504109985, 2.21341078569281, 1.73986581729794, -1.69726138161603, -1.83683100568767, -1.3516335312074, -0.288339644404166, 2.43793017610994, 4.021428674649, -5.62574595404721, -7.16564773674705, -3.6736013112829, -0.298555021497173, 1.46606642601136, 1.73817628961036, -0.631252574249609, -1.70109847571968, -1.0588101460626, 1.31347166509332, 1.64084894143721, 1.99886186392205, 1.12831155370992, 0.690924501872247, 0.957891378127895, -3.37623128298566, -2.93643120666729, -1.15984483519665, 2.00528280804466, 2.04759965000856, 0.0583231911811567, -0.453288259561501, -0.109519427650551, 2.57436669556402, 1.49430321737785, 0.134278896045808, -0.144516907042029, -0.777281265785068, 0.0991679994599942, 1.20490956766865, -1.39587600004585, -1.62744850333245, -2.05432035624814, -0.58996530704156, -0.358105747461861, -0.969668985312145, -1.02105920863618, 0.239728865376966, 0.47317512622918, -0.530543630477763, 0.0293416979434837, 0.142727136639976, 1.42021631976613, 0.380389447413165, 1.12413364743777, 2.43727019448504, 1.93515168714463, 5.26755041331374, 5.58070524402661, 2.78931132295857, 0.902856378124465, 0.122908263977391, -0.595275969687596, -0.735158242565535, 0.295448457389827, 1.49651326401657, 0.391698494275945, 0.920326367938329, 2.46530234749983, 1.01213008657988, 0.433856062354837, -0.437494735853989, 0.40535423982244, 1.15126516377248, 0.23339538930523, 0.339574870019606, 0.719039091000082, 0.479682592330763, 0.135817819973831, -0.91965154072523, -2.91630334627842, -3.77195809993614, -3.60170693982958, -1.24802781745584, -1.86833465278352, -0.743798884986824, -1.08410657133127, -0.472732160855543, 4.39132589296646, 3.08930499464002, 2.36512322295162, 3.08063745146265, 2.45705961087452, 1.57483894070708, 1.01831101736379, 0.378458335470199, -1.70675948398771, -2.1822923904093, -0.0739376552535938, 1.86124731132437, 2.08536472008376, 0.786708041879707, -1.2435609369744, -1.34280907050502, -1.33119214771747, 3.23998953547924, 4.31159864671867, 2.35261316384532, -0.721232295730943, -0.000767556419266957, -0.279212186920793, -5.10912053107504, -3.15606637872521, 0.810156598975859, 2.47891375120775, 1.63767018222001, 0.0631812488987988, -1.17732679139782, 1.92794816920531, 1.74904845806404, -0.156383771283357, -2.19262557443902, -2.51112473710029, -1.90062704605517, 0.402995421482425, 0.376778023464264, -6.8365740860473, -6.34240343535013, -4.29305045524306, -0.878874430238532, 0.374237390895764, -0.300067592209067, 0.3388784295558, 1.11268864866726, 1.19515848411267, 1.83213014975393, 2.49724894714788, 0.84771674552611, -2.46736329705546, -2.5191181746953, -1.01569034667967, 1.62932821753159, 2.50766843790521, 1.20717574713547, 0.352815370455899, -0.999757054645123, -0.580679577639651, -0.816810009237473, -3.29534040799893, -2.24485379985796, 0.691784280946011, 0.216108751988089, -1.62615761990548, 0.408471829519518, 1.70754234973563, 0.875278523833068, -1.79755391245616, -0.765345611478853, 0.961522905168801, -0.0677954204535503, -2.80295381399077, -2.4576751843832, -0.688791062450808, -0.576354591112729, -5.80502724424317, -4.03925077048987, -0.912748784409071, -0.374515942597487, -1.7538124093463, -1.15408745900706, 1.2580056536649, 0.721364645131387, 1.96752835877277, -0.150059868557396, 3.51202955275845, 1.13520234255457, -1.61726747838628, -2.56756908397205, -2.63204713631664, 0.30226372880621, 1.28794514858261, 0.671352038071078, 0.261478416834908, 1.08695050750061, 0.423545944792975, -1.19784121781733, -1.25031210199527, -0.91963774603955, 0.891940983526066, 0.901987405160857, 2.30650675939515, 4.6435005147262, 2.45932855078223, 0.297604982802341, -0.997613930977183, -1.91751949014799, -2.35529748124354, -1.00887004861156, 0.1095743811857, 1.12890262680236, 0.846089323966845, 0.261837521224171, 0.115115610489419, 0.216847568992397, 0.875783561934371, -1.17854992494425, -0.968731905389821, -0.399063747524496, -0.297912634101954, -2.96613279091619, -1.97240254891385, -1.4599198731415, -0.378141799678276, 0.479491771953498, 0.834233759616706, 1.37053675230751, 2.0941572442331, -1.16460959444015, -0.764486018234382, -0.352710584937402, 0.572253210602834, 0.668029528531709, 2.20159709483534, 1.41890332324469, 0.875550582431663, 1.20473379996235, 0.211898542324319, -1.12347409650265, -0.692781057630004, 1.46757550969897, 1.26208281132218, 1.85169115910733, 2.39689108393837, 1.06281480117464, -0.144443954694692, -1.00027614124819, -1.14885621925509, -1.88654568223697, -2.74513112047395, -2.39747105619643, -0.83925646988332, 5.40097572023663, 5.99695305508785, 3.097246571825, 1.17225029496594, 0.983828421192127, -0.707895763171177, -1.37884892823024, -1.07299573491272, -0.0122615442489361, -0.0961852583005038, 1.09626349174379, 0.721317139639417, -0.746425254133715, 2.49353605610903, 2.0467897331417, 0.433641666116876, 0.856641385611092, 1.15680501747799, 2.77394548396039, 3.69593532895137, 2.6535133350672, 3.38675711442973, 2.99091319034043, 1.80789866198575, 1.12677881443069, 0.371618523335552, -0.671924380110139, -1.83224885481871, -0.892266039477728, 0.684609001079026, 1.90098662280984, 2.2262546101034, 0.709380974435907, -0.145153169339163, -0.928384207891866, 0.0965052514508109, 0.0763368034118342, 1.32492946814549, 0.570716813320824, -0.911002702860683, -3.28224164226898, -2.39303599575728, -0.411549693165511, -0.0291862375250271, 0.893623116037588, -0.295265546820206, -0.614764926640702, 0.698562080422445, 0.735524713229628, -0.117251474877304, -3.46268336651757, -3.25253880571123, 0.147233575684406, 1.12966327041718, 3.65147122399176, 1.48012569343478, 6.26864823635147, 4.34178548204398, -0.838469751480694, -2.15903520515314, -4.34713860185419, -2.73079810536133, -0.829798401512069, -0.631684474979878, -1.7531360696792, -1.78284547084014, -0.0697809897071269, -0.611863907510918, -0.778320088928396, 1.04919107826397, 1.46916383599131, -0.137959299596261, 0.563145799441769, 1.36750378759618, 2.48441675538289, 3.29818845077975, 1.32467920894071, -1.36733118501851, -1.72566923694308, -1.79596223580596, 0.114151912857804, 0.862165475999593, 1.59554655288258, -0.0276642018549236, -0.338791392457265, -0.225791127427969, -1.80500087400132, 5.81066879292208, 6.82681146222648, 4.20923536804268, -0.534462767896464, -5.03485588235798, -3.84848478395311, -2.23592167242184, -0.544468735595892, -0.117816678225373, 0.199580905498544, 0.166682957910035, 0.538351479212281, -0.165184340363526, 1.70846490291775, -0.428924165142648, 0.144241067191096, 1.06316121304816, 1.86974858624133, 2.35126689470158, 1.22896901886272, 1.12070903797652, 0.246119902768321, -0.265001413346171, -5.51699517536417, -4.25858344932621, -1.58139965152254, -0.0433448659995377, 2.92385315119381, -2.08187248048284, -3.6032172038181, -4.85253734027665, -1.33172778444363, -0.17840284510077, -0.740238718902239, -0.825467030574543, -1.32830256094649, -0.777800718943696, -0.178624767186471, -3.28185149322273, -3.26786225848825, -3.25883383892214, -1.08076857124956, 0.0342743588621588, 0.431637922000286, 0.0507999088172149, 1.03008395151914, 0.892826084191693, 0.0895019158796122, 1.41473827823274, 2.91493612377213, -1.97902276208714, 1.06367253923166, 5.27385712911363, 3.67042320487516, 0.782561388675378, -0.360141954181277, -0.624567865802663, 0.281060321724171, -0.671012108419172, 2.50260431027482, 2.07970610518249, -1.20989782009227, -3.16824951099057, -1.71394421323118, -0.417022106203542, -0.756027370113074, -1.58167131244656, -0.238544715546333, 0.357049482958872, 0.407612838454049, 0.456734414382648, -1.16074618579003, 0.333172331591581, 0.199152592671834, -2.11020086331783]];

        let back = AllData::yule_walker_ar_coefficients_with_bic(&dat, 3);
        println!("{:?}", back);
        let (blocks, back) = AllData::process_data("/Users/afarhat/Downloads/GSE26054_RAW/GSM639826_ArgR_Arg_ln_ratio_1.pair", 4641652, 498, 25, true, true);


        println!("{:#?}", &blocks[0][0..20]);
        println!("{:?}", back);
*/
        let a = AllData::create_inference_data("/Users/afarhat/Downloads/sequence.fasta", "/Users/afarhat/Downloads/GSE26054_RAW/GSM639826_ArgR_Arg_ln_ratio_1.pair", "/Users/afarhat/Downloads/", true,
                                 498, 25, &None, false);

        println!("{:?} {:?}", serde_json::to_string(&a.0), a.1);
    }*/ 
}
