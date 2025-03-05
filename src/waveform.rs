use std::ops::{Add, AddAssign, Sub, SubAssign, Mul, Range};
use std::cmp::{max, min};
use std::error::Error;
use std::fmt;

//use std::time::{Duration, Instant};

use core::f64::consts::PI;
use core::iter::zip;

use crate::base::{Bp, LN_2, MAX_BASE};
use crate::sequence::{Sequence, BP_PER_U8};
use crate::modified_t::*;
use crate::data_struct::AllDataUse;

use log::warn;

use statrs::distribution::{StudentsT, Continuous, ContinuousCDF};


use aberth;
use num_complex::Complex;

use num_traits::float::{Float, FloatConst};
use num_traits::MulAdd;

use plotters::prelude::*;


use plotters::prelude::full_palette::ORANGE;

use rayon::prelude::*;

use assume::assume;


use serde::{Serialize, Deserialize};

use strum::{EnumCount, VariantArray as VariantArrayArr, IntoEnumIterator};
use strum_macros::{EnumCount as EnumCountMacro, VariantArray, EnumIter};

pub(crate) const WIDE: f64 = 3.0;

use rand::Rng;
use rand::distributions::{Distribution, Standard};
use rand::seq::SliceRandom;

//This is about the first point of equality for the fit functions
//Technically, they cross again later, but based on visual inspection,
//this is a nothingburger: the low tail just happens to oscillate and
//the high tail is more accurate throughout that range
const A0: f64 = 1.97;


//This is just a constant which determines how sharp the transition between the low and high tail are while still being analytic
const K: f64 = 16.0;

//This is a constant that tells us when to stop computing the high tail because it literally doesn't matter, since it would be 2^(-53) times its value at most added to the function
//It is equal to a round number greater than 2^(53/K). Since I chose K = 16, the cutoff is 20.0
const HIGH_CUT: f64 = 20.0;

/// This is an enum denoting how wide the Kernel should be
/// It's a unit type in my code: I go with fragment length
/// determining a single width. But if you are editing this code, 
/// it's an option
#[derive(Serialize, Deserialize, Debug, Clone, Copy, VariantArray, EnumCountMacro, EnumIter, PartialEq, Eq)]
pub enum KernelWidth {
     Wide = 0,
}

impl Distribution<KernelWidth> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> KernelWidth {
        //SAFETY: So long as KernelWidth can produce elements of its type, this is safe
        unsafe {
            *KernelWidth::VARIANTS.choose(rng).unwrap_unchecked()
        }
    }
}

/// This is an enum denoting the shape of the Kernel should be
#[derive(Serialize, Deserialize, Debug, Clone, Copy, VariantArray, EnumCountMacro, EnumIter, PartialEq, Eq)]
#[non_exhaustive]
pub enum KernelVariety {
    Gaussian = 0,
  //  HalfLeft = 1,
  //  HalfRight = 2,
  //  AbsExponential = 3,
  //  Footprint = 4,
}


impl Distribution<KernelVariety> for Standard {
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> KernelVariety {
        //SAFETY: So long as KernelVariety can produce elements of its type, this is safe
        unsafe {
            *KernelVariety::VARIANTS.choose(rng).unwrap_unchecked()
        }
    }
}

/// This specifies the shape of a binding kernel
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Kernel{

    peak_height: f64,
    peak_width: f64,
    kernel_type: KernelVariety,
    kernel_width: KernelWidth,
    kernel: Vec<f64>,
}

impl Mul<f64> for &Kernel {

    type Output = Kernel;

    fn mul(self, rhs: f64) -> Kernel {

        
        Kernel {
            peak_height: self.peak_height*rhs,
            peak_width: self.peak_width,
            kernel_type: self.kernel_type,
            kernel_width: self.kernel_width,
            kernel: self.kernel.iter().map(|a| a*rhs).collect(),
        }


    }

}

impl Kernel {
    
    pub fn new(peak_width: f64, peak_height: f64, kern_width: KernelWidth, variety: KernelVariety) -> Kernel {

        let span = (peak_width*WIDE) as isize;

        let domain: Vec<isize> = (-span..(span+1)).collect();

        let true_width = match kern_width {
            //KernelWidth::Narrow => peak_width/3.0,
            //KernelWidth::Medium => peak_width*2.0/3.0,
            KernelWidth::Wide => peak_width,
        };

        let range = match variety {

            KernelVariety::Gaussian => domain.iter().map(|a| (-((*a as f64).powf(2.0))/(2.0*true_width.powf(2.0))).exp()*peak_height).collect(),
            /*KernelVariety::HalfRight => domain.iter().map(|a| { 
                if *a >= 0 {
                    (-((*a as f64).powf(2.0))/(2.0*true_width.powf(2.0))).exp()*peak_height
                } else {
                     (-((*a as f64).powf(2.0))/(2.0*(span as f64).powi(4))).exp()*peak_height
                }
            }).collect(),
            KernelVariety::HalfLeft => domain.iter().map(|a| { 
                if *a <= 0 {
                    (-((*a as f64).powf(2.0))/(2.0*true_width.powf(2.0))).exp()*peak_height
                } else {
                     (-((*a as f64).powf(2.0))/(2.0*(span as f64).powi(4))).exp()*peak_height
                }
            }).collect(),
            KernelVariety::AbsExponential => domain.iter().map(|a| ( (-((*a as f64).abs())/(2.0*true_width.abs())).exp()*peak_height)).collect(),
            KernelVariety::Footprint => domain.iter().map(|a| ((-((*a as f64).powf(2.0))/(2.0*true_width.powf(2.0))).exp()-(-((*a as f64).powf(2.0))/(2.0*(MAX_BASE as f64))).exp())*peak_height).collect(), */
        };
        Kernel{
            peak_height: peak_height,
            peak_width: peak_width,
            kernel_width: kern_width,
            kernel_type: variety,
            kernel: range,
        }

    }

    pub fn get_sd(&self) -> f64 {
        self.peak_width
    }
    pub fn get_true_width(&self) -> f64 {
        match self.kernel_width {
            //KernelWidth::Narrow => self.peak_width/3.0,
            //KernelWidth::Medium => self.peak_width*2.0/3.0,
            KernelWidth::Wide => self.peak_width,
        }
    }

    pub fn get_curve(&self) -> &Vec<f64> {

        &self.kernel
    
    }

    pub fn len(&self) -> usize {
        self.kernel.len()
    }
   
    pub fn get_height(&self) -> f64 {
        self.peak_height
    }
    pub fn scaled_kernel(&self, scale_by: f64) -> Kernel {
        self*scale_by
    }

}

/// This is an error that results from failing to create a Waveform object
#[derive(Clone, Debug)]
pub enum WaveCreationError {
    DumbSpacer,
    MisalignedSequenceSpacer,
    OverruningDataBlock,
}

impl std::fmt::Display for WaveCreationError {

    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {

        match self {
            Self::DumbSpacer => write!(f, "Your data has to increment in location."),
            Self::MisalignedSequenceSpacer => write!(f, "Your data array and sequence are incompatible"),
            Self::OverruningDataBlock => write!(f, "You have a data array which overruns one of your sequence blocks"),
        }
    }

}

impl Error for WaveCreationError {}

/// This is the struct which holds occupancy traces. 
/// This needs to uphold a lot of invariants, and thus initialization and usage
/// are both quite finicky. 
#[derive(Clone, Debug)]
pub struct Waveform<'a> {
    wave: Vec<f64>,
    spacer: usize,
    point_lens: Vec<usize>,
    start_dats: Vec<usize>,
    seq: &'a Sequence,
}

impl<'a> Waveform<'a> {

    /*
       This function has an additional constraint for correctness:

       The start_data must be organized in accordance with spacer and sequence.
       In particular, each block must have 1+floor((block length-1spacer) data points
       HOWEVER, this initializer only knows TO panic if the total number of data points
       cannot be reconciled with seq and spacer. If you have one too MANY points in one
       block and one too FEW points in another, this will NOT know to break. If you get 
       this wrong, your run will be memory safe, but it will give garbage and incorrect
       answers.
       */

    /// This initializes the Waveform object. This gets very finicky: pay close
    /// attention to the errors you get if you even mildly screw this up.
    /// # Invariants
    /// - Every data block will have a length that is less than the length of 
    /// its corresponding sequence block in `[Bp]`s/`spacer`. If `spacer >=
    /// [BP_PER_U8]`, this length is number of base pairs minus 1/spacer. If
    /// `spacer < [BP_PER_U8]`, this length is 1 + the length in the former case
    /// # Errors
    /// - If `spacer == 0`.
    /// - If the length of data makes it impossible to reconcile `start_data` with
    ///   `seq` and `spacer`. NOTE OF WARNING: if `start_data` is not partitioned 
    ///   correctly but it is still possible to to reconcile the final length, this
    ///   will simply mis-partition your waveform. 
    /// - If the waveform blocks that result from this would overrun the sequence blocks.
    pub fn new(start_data: Vec<f64>, seq: &'a Sequence, spacer: usize) -> Result<Waveform<'a>, WaveCreationError>  {


        if spacer == 0 { return Err(WaveCreationError::DumbSpacer);}

        let (point_lens, start_dats) = Self::make_dimension_arrays(seq, spacer);

        //println!("bl {:?}", seq.block_lens());
        //println!("{} {} pl {:?}\n sd {:?}", start_data.len(), point_lens.len(), point_lens, start_dats);
        
        if (point_lens.last().unwrap() + start_dats.last().unwrap()) != start_data.len() {
           // println!("{:?} {:?}", &start_data[28910..28919], seq.block_lens());
            panic!("IMPOSSIBLE DATA FOR THIS SEQUENCE AND SPACER")
        }
 
        //This check needs to be point_lens[i]-1 because we need the last INDEX of point_lens
        //to fall within the sequence, not length
        if spacer >= BP_PER_U8 {
            _ = seq.block_lens().iter().enumerate().map(|(i,a)| if(a/spacer < point_lens[i]-1) { panic!("failed");} else{ true}).collect::<Vec<_>>();
        } else {
            _ = seq.block_lens().iter().enumerate().map(|(i,a)| if(a/spacer < point_lens[i]) { panic!("failed");} else{ true}).collect::<Vec<_>>();
        }

        Ok(Waveform {
            wave: start_data,
            spacer: spacer,
            point_lens: point_lens,
            start_dats: start_dats,
            seq: seq,
        })
    }

    pub(crate) fn make_dimension_arrays(seq: &'a Sequence, spacer: usize) -> (Vec<usize>, Vec<usize>) {

        /*We require that all data begin with the 0th base for mainly this reason
          We are using the usize round down behavior here to denote how many
          data points per block there are. 

          Say spacer = 5, and a block has length 15. We want three data points:
          base 0, base 5, and base 10. If the block has length 16, we want 4 points:
          base 0, base 5, base 10, and base 15. If data wasn't normalized so that 
          the first base in each block was considered base 0, this behavior
          couldn't be captured. 

          The biggest advantage to doing things like this is that bps can be
          converted to index easily by just dividing by spacer. A bp with a 
          non-integer part in its representation is dropped in the waveform.
         */
        let point_lens: Vec<usize> = seq.block_lens().iter().map(|a| if spacer >= BP_PER_U8 {1+((a-1)/spacer)} else {(a-1)/spacer}).collect();

        let mut size: usize = 1;

        let mut start_dats: Vec<usize> = Vec::new();

        for i in 0..point_lens.len(){
            start_dats.push(size-1);
            size += point_lens[i];
        }

        (point_lens, start_dats)
    }


    //Returns None if data_ind is >= length of the backing vector
    pub(crate) fn block_and_base(&self, data_ind: usize) -> Option<(usize, usize)> {

        if data_ind >= self.wave.len() { return None;}

        let block: usize = self.start_dats.partition_point(|&a| a <= data_ind)-1;

        let data_point = data_ind-self.start_dats[block];

        let base = data_point*self.spacer;

        Some((block, base))

    }

    pub(crate) fn data_ind(&self, block: usize, base: usize) -> Option<usize> {

        if block >= self.start_dats.len() { return None; }
        let data_point = base/self.spacer;

        Some(self.start_dats[block]+data_point)

    }

    pub(crate) fn intersect_kmer_start_num(&self, data_ind: usize, k: usize) -> usize {

        self.intersect_kmer_start_range(data_ind, k).map(|a| a.1.len()).unwrap_or(0)

    }

    pub(crate) fn intersect_kmer_start_range(&self, data_ind: usize, k: usize) -> Option<(usize, Range<usize>)>{

        if k == 0 { return None; }

        let Some((block, base)) = self.block_and_base(data_ind) else { return None; };

        self.intersect_kmer_start_range_block_and_base(block, base, k)
    
    }

    pub(crate) fn intersect_kmer_start_range_block_and_base(&self, block: usize, base: usize, k: usize) -> Option<(usize, Range<usize>)> {

        if k == 0 { return None; }

        if base < k { return None;}

        let block_len = self.seq.ith_block_len(block);
        
        if base+k >= block_len {return None;}

        let begin = base+1-k;

        let end = base; 

        Some((block, begin..(end+1)))


    }

   
    pub(crate) fn intersect_data_start_range_block_and_base(&self, block: usize, base: usize, k: usize) ->  Option<Vec<usize>> {

        let (block, base_range) = self.intersect_kmer_start_range_block_and_base(block, base, k)?;

        let mut prior_dat: Option<usize> = None;

        let mut intersect_data: Vec<usize> = Vec::with_capacity(k/self.spacer + 1);

        for base_id in base_range {

            let Some(data_ind) = self.data_ind(block, base_id) else {continue;};

            match prior_dat {
                None => {
                    prior_dat = Some(data_ind);
                    intersect_data.push(data_ind);
                },
                Some(dat) => {
                    if dat != data_ind {
                        prior_dat = Some(data_ind);
                        intersect_data.push(data_ind);
                    }
                },
            };
        }

        Some(intersect_data)

    }

    pub(crate) fn intersect_kmer_data_block_and_base(&self, block: usize, base: usize, k: usize) -> Vec<f64> {

        if k == 0 { return vec![]; }

        let begin = if base < k { 0_usize } else { base+1-k };

        let block_len = self.seq.ith_block_len(block);

        let end = if (base+k) >= block_len { block_len-k } else { base } ;

        let begin_dat_ind = (begin + (begin % self.spacer))/self.spacer;

        let end_dat_ind = (((end - (end % self.spacer))/self.spacer)+1).min(self.point_lens[block]);

        self.wave[(self.start_dats[block]+begin_dat_ind)..(self.start_dats[block]+end_dat_ind)].iter().map(|&a| a).collect()

    }

    /// Creates a `Waveform` which is compatible with `seq` and `spacer`, but all `0.0`.
    pub fn create_zero(seq: &'a Sequence, spacer: usize) -> Waveform<'a> {
       

        let (point_lens, start_dats) = Self::make_dimension_arrays(seq,spacer);

        let tot_l: usize = point_lens.iter().sum();

        Waveform {
            wave: vec![0.0; tot_l],
            spacer: spacer,
            point_lens: point_lens,
            start_dats: start_dats,
            seq: seq,
        }
    }

    /// Creates a `Waveform` identical to self, except all data are `0.0`
    pub fn derive_zero(&self) -> Waveform {

        let tot_l: usize = self.point_lens.iter().sum();

        
        Waveform {
            wave: vec![0.0; tot_l],
            spacer: self.spacer,
            point_lens: self.point_lens.clone(),
            start_dats: self.start_dats.clone(),
            seq: self.seq,
        }

    }

    pub(crate) fn subtact_self(&mut self, offset: f64) {
        self.wave = self.wave.iter().map(|&a| a-offset).collect();
    }

    // NOTE: This is THE workhorse function of the code. It is the reason for 
    //       all of the invariants I have, the place where I optimized most 
    //       heavily, and where your code will explode with UB if you modify 
    //       without checking all of the invariants I have on other parts of
    //       the code
    //SAFETY: -block must be less than the number of blocks
    //        -center must be less than the number of bps in the blockth block
    //        -the length of peak MUST be strictly less than the number of base pairs represented in the smallest data block
    pub(crate) unsafe fn place_peak(&mut self, peak: &Kernel, block: usize, center: usize) {



        //Given how we construct kernels, this will never need to be rounded: they always have an odd number of data points
        let place_bp = (((peak.len()-1)/2) as isize)-(center as isize); //This moves the center of the peak to where it should be, taking the rest of it with it
        let cc = (place_bp).rem_euclid(self.spacer as isize); // This defines the congruence class of the kernel indices that will be necessary for the signal
       
        // If the function invariant is obeyed, this is always in bounds 
        let zerdat: usize = *self.start_dats.get_unchecked(block); //This will ensure the peak is in the correct block


        let completion: usize = ((cc-((peak.len() % self.spacer) as isize)).rem_euclid(self.spacer as isize)) as usize; //This tells us how much is necessary to add to the length 
                                                                            //of the kernel to hit the next base in the cc
       
        // This max is my way of guarding against peak drawing that is outside of the sequence block on the 5' end
        // `cc` is effectively the "0" point of the block with the given offset from `place_bp`
        let min_kern_cc = max(cc, place_bp);
        // This min is my way of guarding against peak drawing that is outside of the sequence block on the 3' end
        let nex_kern_cc = min(((*self.point_lens.get_unchecked(block)*self.spacer) as isize)+place_bp, (peak.len()+completion) as isize);
        let min_data: usize = ((min_kern_cc-place_bp)/((self.spacer) as isize)) as usize;  //Always nonnegative: if place_bp > 0, min_kern_bp = place_bp
        let nex_data: usize = ((nex_kern_cc-place_bp)/((self.spacer) as isize)) as usize; //Assume nonnegative for the same reasons as nex_kern_bp


        let kern_change = self.wave.get_unchecked_mut((min_data+zerdat)..(nex_data+zerdat));
        
        let kern_start = min_kern_cc as usize;
        for i in 0..kern_change.len() {
            *kern_change.get_unchecked_mut(i) += peak.kernel.get_unchecked(kern_start+i*self.spacer);
        }


    }


    pub(crate) fn kmer_propensities(&self, k: usize) -> Vec<f64> {

        let coded_sequence = self.seq.seq_blocks();
        let block_lens = self.seq.block_lens(); //bp space
        let block_starts = self.seq.block_u8_starts(); //stored index space

        let mut uncoded_seq: Vec<Bp> = vec![Bp::A; self.seq.max_len()];

        let mut store: [Bp ; BP_PER_U8];

        let mut propensities: Vec<f64> = vec![0.0; self.seq.number_unique_kmers(k)];

        {
            let uncoded_seq = uncoded_seq.as_mut_slice();
            for i in 0..(block_starts.len()) {


                for jd in 0..(block_lens[i]/BP_PER_U8) {

                    store = Sequence::code_to_bases(coded_sequence[block_starts[i]+jd]);
                    for k in 0..BP_PER_U8 {
                        uncoded_seq[BP_PER_U8*jd+k] = store[k];
                    }

                }


                for j in 0..((block_lens[i])-k) {

                    let center = j + ((k-1)/2);

                    let lower_data_ind = center/self.spacer; 

                    //Center cannot possibly beyond the bases because of how I take j: I stop before j+k runs 
                    //into the sequence end, which will always make lower_data_ind hit, at most the largest
                    //location. BUT, I CAN be over any ability to check the next location

                    //SAFETY: notice how we defined j, and how it guarentees that get_unchecked is fine
                    let u64_mot = Sequence::kmer_to_u64(&(unsafe { uncoded_seq.get_unchecked(j..(j+k)) }).to_vec());
                    let between = center % self.spacer;
                    //The second statement in the OR seems bizarre, until you remember that we stop
                    //including DATA before we stop including BASES. This is our check to make sure
                    //that if we're in that intermediate, then we include such overrun bases
                    let to_add = if (between == 0) || (lower_data_ind + 1 >= self.point_lens[i]) {
                        /*#[cfg(test)]{
                            println!("block {} j {} kmer {} amount {}", i, j,u64_mot, self.wave[self.start_dats[i]+lower_data_ind]); 
                        }*/
                        self.wave[self.start_dats[i]+lower_data_ind]
                    } else {
                        let weight = (between as f64)/(self.spacer as f64);
                        /*#[cfg(test)]{
                            println!("block {} j {} kmer {} amount {}", i, j, u64_mot, self.wave[self.start_dats[i]+lower_data_ind]*(1.-weight)+self.wave[self.start_dats[i]+lower_data_ind+1]*weight);
                        }*/
                        self.wave[self.start_dats[i]+lower_data_ind]*(1.-weight)+self.wave[self.start_dats[i]+lower_data_ind+1]*weight
                    };

                    //A motif should never be proposed unless there's a place in the signal that
                    //could reasonably considered motif-ish
                    //Note: this does NOT interfere with detailed balance: motifs are always
                    //proposed with a minimum height and so when death is proposed, it should
                    //always be the case that there's SOMEWHERE with the height minimum
                    if to_add.abs() > (0.0) {
                        propensities[self.seq.id_of_u64_kmer_or_die(k, u64_mot)] += to_add.powi(2);
                    }
                }

            

            }
            //println!("unique u64 {}-mers: {:?}", k, self.seq.unique_kmers(k));

        }


        propensities


    }

    /// Yields the Euclidean distance between `self` and `other_wave` 
    /// # Panics
    ///   If `other_wave` doesn't point to the same AllDataUse as `self`
    pub fn rmse_with_wave(&self, other_wave: &Waveform) -> f64 {

        let residual = self-other_wave;

        let length: f64 = residual.wave.len() as f64;

        (residual.wave.into_iter().map(|a| a.powi(2)).sum::<f64>()).sqrt()

    }



    pub(crate) fn generate_extraneous_binding(background: &Background, spacer: usize, scaled_heights_array: &[f64]) -> Vec<f64> {

        //Is our potential binding strong enough to even attempt to try extraneous binding?
        let caring_threshold = f64::EPSILON;//4.0*background.noise_spread_par();

        let extraneous_bindings: Vec<_> = scaled_heights_array.iter().filter(|&a| *a > caring_threshold).collect();

        //Do we have any extraneous binding that we need to account for?
        if extraneous_bindings.len() == 0 { return Vec::new();}

        let scaled_heights = extraneous_bindings;

        let sd_over_spacer = background.kernel[0].get_sd()/(spacer as f64);

        let ln_caring_threshold = caring_threshold.ln();


        //TODO: if Kernel goes from being exclusively Gaussians to incorporating other shapes, this needs to change to check those shapes
        //Note: this relies on the fact that casting an f64 to a usize uses the floor function
        let generate_size_kernel = |x: f64| (sd_over_spacer*(2.0*(x.ln()-ln_caring_threshold)).sqrt()) as usize;
        let quadriatic_ratio = (-0.5*(sd_over_spacer).powi(-2)).exp();

        let size_hint: usize = generate_size_kernel(*scaled_heights[0]);

        let mut output_vec: Vec<f64> = Vec::with_capacity(background.kernel_ref(KernelWidth::Wide, KernelVariety::Gaussian).len());

        for height in scaled_heights {

            let kernel_size = generate_size_kernel(*height);;

            //If the size of your kernel is genuinely overflowing integers after filtering out by spacer
            //You have such massive problems that this should really be the least of your concerns
            //SAFETY: we have a panic set on BackgroundDist in case this conversion could fail. 
            //        We choose to do that because my word, if your log2 data's 
            //        background distribution has an SD of > a MILLION, your experiment is horribly degenerate
            //        It should be completely impossible to unless you're getting
            //        heights more than exp((i32::MAX/sd_over_spacer)^2) times larger than caring_threshold
            //        And it's only possible for that to happen within f64's specs if the background sd
            //        is more than a million. MORE than that if spacer is bigger than 1

            let kernel_size_i32: i32 = unsafe{ kernel_size.try_into().unwrap_unchecked() };
            let mut kernel_bit: Vec<f64> = Vec::with_capacity(2*kernel_size+1);

            kernel_bit.push(*height);

            if kernel_size > 0 {
                for i in 1..=kernel_size_i32 {
                    let ele = height*quadriatic_ratio.powi(i*i);
                    kernel_bit.push(ele);
                    kernel_bit.push(ele);
                }
            }

            output_vec.append(&mut kernel_bit);
        }

        output_vec


    }

    pub(crate) fn produce_noise<'b>(&self, data_ref: &'b AllDataUse) -> Noise<'b> {
        
        let residual = self-data_ref.data();

        return Noise::new(residual.wave, Vec::new(), data_ref.background_ref());
    }

    pub(crate) fn produce_noise_with_extraneous<'b>(&self, data_ref: &'b AllDataUse, extraneous_bind_array: &[f64]) -> Noise<'b> {

        let mut noise = self.produce_noise(data_ref);


        noise.replace_extraneous(extraneous_bind_array.to_vec());

        noise

    }

    /// Yields the median distance between `self` and `other_wave` 
    /// # Panics
    ///   If `data` doesn't point to the same AllDataUse as `self`
    pub fn median_distance_between_waves(&self, data: &Self) -> f64 {
        let residual = self-data;
        let mut abs_wave = residual.wave.iter().map(|&a| a.abs()).collect::<Vec<f64>>();
        abs_wave.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        if (abs_wave.len() & 1) == 1 {
            abs_wave[abs_wave.len()/2]
        } else {
            (abs_wave[abs_wave.len()/2]+abs_wave[abs_wave.len()/2 -1])/2.0
        }
    }
   

    pub(crate) fn generate_all_locs(&self) -> Vec<usize> {

        let mut length: usize = 0;
        for &a in self.point_lens.iter() { length += a; }

        let mut locations: Vec<usize> = Vec::with_capacity(length);

        //println!("starts {:?}", self.start_dats);
        //println!("lens {:?}", self.point_lens);
        for i in 0..(self.start_dats.len()){

            for j in 0..(self.point_lens[i]){
                locations.push((self.start_dats[i]+j)*self.spacer);
            }
        }

        locations

    }

    pub(crate) fn generate_all_indexed_locs_and_data(&self, start_lens: &[usize]) -> Option<Vec<(Vec<usize>, Vec<f64>)>> {


        if self.start_dats.len() != start_lens.len() {
            return None;
        }

        let mut locations_and_data: Vec<(Vec<usize>, Vec<f64>)> = Vec::with_capacity(start_lens.len());

        for i in 0..start_lens.len() {

            let small_loc: Vec<usize> = (0..self.point_lens[i]).map(|j| start_lens[i]+(j*self.spacer)).collect();
            let end_len = if i < (start_lens.len()-1) { self.start_dats[i+1] } else { self.wave.len() };
            let small_dat: Vec<f64> = self.wave[self.start_dats[i]..end_len].to_owned();
            locations_and_data.push((small_loc, small_dat));
        }

        Some(locations_and_data)

    }

    /// Saves an image set of occupancy traces compared to data twice: once to 
    /// `{signal_directory}/{signal_name}/from_{start}_to_{end}.png, and once 
    /// to `{signal_directory}/from_{start}_to_{end}/{signal_name}.png
    /// If this fails, will simply warn you but won't return a hard error
    pub fn save_waveform_to_directory(&self, data_ref: &AllDataUse, signal_directory: &str, signal_name: &str, trace_color: &RGBColor, only_sig: bool) {

        let current_resid = data_ref.data()-&self;

        let zero_locs = data_ref.zero_locs();
        
        let block_lens = data_ref.data().seq().block_lens();

        let blocked_locs_and_signal = self.generate_all_indexed_locs_and_data(data_ref.zero_locs()).expect("We designed signal to correspond to data_ref");

        let blocked_locs_and_data = data_ref.data().generate_all_indexed_locs_and_data(data_ref.zero_locs()).expect("Our data BETTER correspond to data_ref");

        let blocked_locs_and_resid = current_resid.generate_all_indexed_locs_and_data(data_ref.zero_locs()).expect("We designed signal to correspond to data_ref");

        let total_dir = format!("{}/{}", signal_directory,signal_name);

        if let Err(creation) = std::fs::create_dir_all(&total_dir) {
            warn!("Could not make or find directory \"{}\"! \n{}", total_dir, creation);
            println!("Could not make or find directory \"{}\"! \n{}", total_dir, creation);
            return;
        };

        for i in 0..blocked_locs_and_signal.len() {

            if only_sig {
                if blocked_locs_and_signal[i].1.iter().all(|&x| x == 0.0) { continue; }
            }

            let loc_block = &blocked_locs_and_signal[i].0;
            let sig_block = &blocked_locs_and_signal[i].1;
            let dat_block = &blocked_locs_and_data[i].1;
            let res_block = &blocked_locs_and_resid[i].1;

            let min_signal = sig_block.iter().min_by(|a,b| a.partial_cmp(b).unwrap()).expect("Waves have elements");
            let min_data_o = dat_block.iter().min_by(|a,b| a.partial_cmp(b).unwrap()).expect("Waves have elements");

            let min = min_signal.min(*min_data_o)-1.0;

            let max_signal = sig_block.iter().max_by(|a,b| a.partial_cmp(b).unwrap()).expect("Waves have elements");
            let max_data_o = dat_block.iter().max_by(|a,b| a.partial_cmp(b).unwrap()).expect("Waves have elements");

            let max = max_signal.max(*max_data_o)+1.0;


            let signal_file = format!("{}/from_{:011}_to_{:011}.png", total_dir, zero_locs[i], zero_locs[i]+block_lens[i]);

            let plot = BitMapBackend::new(&signal_file, (3300, 1500)).into_drawing_area();

            let derived_color = DerivedColorMap::new(&[WHITE, ORANGE, RED]);

            plot.fill(&WHITE).unwrap();

            let (left, right) = plot.split_horizontally((95).percent_width());

            let (right_space, _) = right.split_vertically((95).percent_height());

            let mut bar = ChartBuilder::on(&right_space).margin(10).set_label_area_size(LabelAreaPosition::Right, 100).caption("Deviance", ("serif", 50)).build_cartesian_2d(0_f64..1_f64, 0_f64..1_f64).unwrap();

            bar.configure_mesh()
                .y_label_style(("serif", 40))
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
                .x_label_style(("serif", 40))
                .y_label_style(("serif", 40))
                .x_label_formatter(&|v| format!("{:.0}", v))
                .x_desc("Genome Location (Bp)")
                .y_desc("Signal Intensity")
                .disable_mesh().draw().unwrap();

            const HORIZ_OFFSET: i32 = -5;

            chart.draw_series(dat_block.iter().zip(loc_block.iter()).map(|(&k, &i)| Circle::new((i as f64, k),2_u32, Into::<ShapeStyle>::into(&BLACK).filled()))).unwrap().label("True Occupancy Data").legend(|(x,y)| Circle::new((x+2*HORIZ_OFFSET,y),5_u32, Into::<ShapeStyle>::into(&BLACK).filled()));


            chart.draw_series(LineSeries::new(sig_block.iter().zip(loc_block.iter()).map(|(&k, &i)| (i as f64, k)), trace_color.filled().stroke_width(10))).unwrap().label("Proposed Occupancy Trace").legend(|(x, y)| Rectangle::new([(x+4*HORIZ_OFFSET, y-4), (x+4*HORIZ_OFFSET + 20, y+3)], Into::<ShapeStyle>::into(trace_color).filled()));

            chart.configure_series_labels().position(SeriesLabelPosition::LowerRight).margin(40).legend_area_size(10).border_style(&BLACK).label_font(("Calibri", 40)).draw().unwrap();

            let abs_resid: Vec<(f64, f64)> = res_block.iter().map(|&a| {

                let tup = data_ref.background_ref().cd_and_sf(a);
                if tup.0 >= tup.1 { (tup.0-0.5)*2.0 } else {(tup.1-0.5)*2.0} } ).zip(loc_block.iter()).map(|(a, &b)| (a, b as f64)).collect();

            let mut map = ChartBuilder::on(&lower)
                .set_label_area_size(LabelAreaPosition::Left, 100)
                .set_label_area_size(LabelAreaPosition::Bottom, 50)
                .build_cartesian_2d((loc_block[0] as f64)..(*loc_block.last().unwrap() as f64), 0_f64..1_f64).unwrap();

            map.configure_mesh().x_label_style(("serif", 0)).y_label_style(("serif", 0)).x_desc("Deviance").axis_desc_style(("serif", 40)).set_all_tick_mark_size(0_u32).disable_mesh().draw().unwrap();


            map.draw_series(abs_resid.windows(2).map(|x| Rectangle::new([(x[0].1, 0.0), (x[1].1, 1.0)], derived_color.get_color(x[0].0).filled()))).unwrap();


            let by_loc_dir = format!("{}/from_{:011}_to_{:011}", signal_directory,zero_locs[i], zero_locs[i]+block_lens[i]);
            
            if let Err(creation) = std::fs::create_dir_all(&by_loc_dir) {
                warn!("Could not make or find directory \"{}\"! \n{}", by_loc_dir, creation);
                continue;
            }


            let by_loc_file = format!("{}/{}.png", by_loc_dir, signal_name);

            let plot = BitMapBackend::new(&by_loc_file, (3300, 1500)).into_drawing_area();

            let derived_color = DerivedColorMap::new(&[WHITE, ORANGE, RED]);

            plot.fill(&WHITE).unwrap();

            let (left, right) = plot.split_horizontally((95).percent_width());

            let (right_space, _) = right.split_vertically((95).percent_height());

            let mut bar = ChartBuilder::on(&right_space).margin(10).set_label_area_size(LabelAreaPosition::Right, 100).caption("Deviance", ("serif", 50)).build_cartesian_2d(0_f64..1_f64, 0_f64..1_f64).unwrap();

            bar.configure_mesh()
                .y_label_style(("serif", 40))
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
                .x_label_style(("serif", 70))
                .y_label_style(("serif", 70))
                .x_label_formatter(&|v| format!("{:.0}", v))
                .x_desc("Genome Location (Bp)")
                .y_desc("Signal Intensity")
                .disable_mesh().draw().unwrap();


            chart.draw_series(dat_block.iter().zip(loc_block.iter()).map(|(&k, &i)| Circle::new((i as f64, k),2_u32, Into::<ShapeStyle>::into(&BLACK).filled()))).unwrap().label("True Occupancy Data").legend(|(x,y)| Circle::new((x+2*HORIZ_OFFSET,y),5_u32, Into::<ShapeStyle>::into(&BLACK).filled()));


            chart.draw_series(LineSeries::new(sig_block.iter().zip(loc_block.iter()).map(|(&k, &i)| (i as f64, k)), trace_color.filled().stroke_width(10))).unwrap().label("Proposed Occupancy Trace").legend(|(x, y)| Rectangle::new([(x+4*HORIZ_OFFSET, y-4), (x+4*HORIZ_OFFSET + 20, y+3)], Into::<ShapeStyle>::into(trace_color).filled()));

            chart.configure_series_labels().position(SeriesLabelPosition::LowerRight).margin(40).legend_area_size(10).border_style(&BLACK).label_font(("Calibri", 40)).draw().unwrap();

            let abs_resid: Vec<(f64, f64)> = res_block.iter().map(|&a| {

                let tup = data_ref.background_ref().cd_and_sf(a);
                if tup.0 >= tup.1 { (tup.0-0.5)*2.0 } else {(tup.1-0.5)*2.0} } ).zip(loc_block.iter()).map(|(a, &b)| (a, b as f64)).collect();

            let mut map = ChartBuilder::on(&lower)
                .set_label_area_size(LabelAreaPosition::Left, 100)
                .set_label_area_size(LabelAreaPosition::Bottom, 50)
                .build_cartesian_2d((loc_block[0] as f64)..(*loc_block.last().unwrap() as f64), 0_f64..1_f64).unwrap();

            map.configure_mesh().x_label_style(("serif", 0)).y_label_style(("serif", 0)).x_desc("Deviance").axis_desc_style(("serif", 40)).set_all_tick_mark_size(0_u32).disable_mesh().draw().unwrap();


            map.draw_series(abs_resid.windows(2).map(|x| Rectangle::new([(x[0].1, 0.0), (x[1].1, 1.0)], derived_color.get_color(x[0].0).filled()))).unwrap();






        }
    

    }
    
    /// Saves an image set of `self` and `alternative` compared to data
    /// to `{signal_directory}/{signal_name}/from_{start}_to_{end}.png 
    /// # Errors
    /// - If a necessary file or directory could not be created 
    pub fn save_waveform_comparison_to_directory(&self, alternative: &Waveform, data_ref: &AllDataUse, signal_directory: &str, signal_name: &str, trace_color: &RGBColor, alter_color: &RGBColor, self_name: &str, alter_name: &str) -> Result<(), Box<dyn Error+Send+Sync>> {

        let current_resid = data_ref.data()-&self;

        let zero_locs = data_ref.zero_locs();
        
        let block_lens = data_ref.data().seq().block_lens();

        let blocked_locs_and_signal = self.generate_all_indexed_locs_and_data(data_ref.zero_locs()).expect("We designed signal to correspond to data_ref");
        let alterna_locs_and_signal = alternative.generate_all_indexed_locs_and_data(data_ref.zero_locs()).expect("We designed signal to correspond to data_ref");

        let blocked_locs_and_data = data_ref.data().generate_all_indexed_locs_and_data(data_ref.zero_locs()).expect("Our data BETTER correspond to data_ref");

        let blocked_locs_and_resid = current_resid.generate_all_indexed_locs_and_data(data_ref.zero_locs()).expect("We designed signal to correspond to data_ref");

        let total_dir = format!("{}/{}", signal_directory,signal_name);

        /*
        if let Err(creation) = std::fs::create_dir_all(&total_dir) {
            warn!("Could not make or find directory \"{}\"! \n{}", total_dir, creation);
            println!("Could not make or find directory \"{}\"! \n{}", total_dir, creation);
            return;
        };*/

        std::fs::create_dir_all(&total_dir)?;

        for i in 0..blocked_locs_and_signal.len() {


            let loc_block = &blocked_locs_and_signal[i].0;
            let sig_block = &blocked_locs_and_signal[i].1;
            let alt_block = &alterna_locs_and_signal[i].1;
            let dat_block = &blocked_locs_and_data[i].1;
            let res_block = &blocked_locs_and_resid[i].1;

            let min_signal = sig_block.iter().min_by(|a,b| a.partial_cmp(b).unwrap()).expect("Waves have elements");
            let min_alter  = alt_block.iter().min_by(|a,b| a.partial_cmp(b).unwrap()).expect("Waves have elements");
            let min_data_o = dat_block.iter().min_by(|a,b| a.partial_cmp(b).unwrap()).expect("Waves have elements");


            let max_signal = sig_block.iter().max_by(|a,b| a.partial_cmp(b).unwrap()).expect("Waves have elements");
            let max_alter  = alt_block.iter().max_by(|a,b| a.partial_cmp(b).unwrap()).expect("Waves have elements");
            let max_data_o = dat_block.iter().max_by(|a,b| a.partial_cmp(b).unwrap()).expect("Waves have elements");

            let max = max_signal.max(*max_data_o).max(*max_alter)+1.0;

            let min = min_signal.min(*min_data_o).min(*min_alter).min(-max/3.0)-1.0;

            let signal_file = format!("{}/from_{:011}_to_{:011}.png", total_dir, zero_locs[i], zero_locs[i]+block_lens[i]);

            let plot = BitMapBackend::new(&signal_file, (3300, 1500)).into_drawing_area();

            plot.fill(&WHITE).unwrap();

            let mut chart = ChartBuilder::on(&plot)
                .set_label_area_size(LabelAreaPosition::Left, 200)
                .set_label_area_size(LabelAreaPosition::Bottom, 200)
                .caption("Signal Comparison", ("Times New Roman", 80))
                .build_cartesian_2d((loc_block[0] as f64)..(*loc_block.last().unwrap() as f64), min..max).unwrap();

            chart.configure_mesh()
                .x_label_style(("serif", 70))
                .y_label_style(("serif", 70))
                .x_label_formatter(&|v| format!("{:.0}", v))
                .x_desc("Genome Location (Bp)")
                .y_desc("Signal Intensity")
                .disable_mesh().draw().unwrap();

            const HORIZ_OFFSET: i32 = -5;

            chart.draw_series(dat_block.iter().zip(loc_block.iter()).map(|(&k, &i)| Circle::new((i as f64, k),2_u32, Into::<ShapeStyle>::into(&BLACK).filled()))).unwrap().label("True Occupancy Data").legend(|(x,y)| Circle::new((x+2*HORIZ_OFFSET,y),5_u32, Into::<ShapeStyle>::into(&BLACK).filled()));


            chart.draw_series(LineSeries::new(sig_block.iter().zip(loc_block.iter()).map(|(&k, &i)| (i as f64, k)), trace_color.filled().stroke_width(10))).unwrap().label(format!("{} Occupancy Trace", self_name).as_str()).legend(|(x, y)| Rectangle::new([(x+4*HORIZ_OFFSET, y-4), (x+4*HORIZ_OFFSET + 20, y+3)], Into::<ShapeStyle>::into(trace_color).filled()));
            
            chart.draw_series(LineSeries::new(alt_block.iter().zip(loc_block.iter()).map(|(&k, &i)| (i as f64, k)), alter_color.filled().stroke_width(10))).unwrap().label(format!("{} Occupancy Trace", alter_name).as_str()).legend(|(x, y)| Rectangle::new([(x+4*HORIZ_OFFSET, y-4), (x+4*HORIZ_OFFSET + 20, y+3)], Into::<ShapeStyle>::into(alter_color).filled()));

            chart.configure_series_labels().position(SeriesLabelPosition::LowerRight).margin(40).legend_area_size(10).border_style(&BLACK).label_font(("Times New Roman", 60)).draw().unwrap();

        }

        Ok(())

    }

    pub(crate) fn with_removed_blocks(&self, remove: &[usize]) -> Option<WaveformDef> {

        let mut new_wave = self.wave.clone();

        let mut remove_descend: Vec<usize> = remove.to_vec();

        // We always want to remove blocks in descending order
        // Otherwise, previously remove blocks screw with the remaining blocks
        remove_descend.sort_unstable();
        remove_descend.reverse();
        remove_descend.dedup();

        remove_descend = remove_descend.into_iter().filter(|a| *a >= self.start_dats.len()).collect();

        //This is only possible if we have all blocks listed in remove_descend,
        //thanks to sorting and dedup().
        if remove_descend.len() == self.start_dats.len() { return None;}

        let mut i = 0;

        if (remove_descend[i] == self.start_dats.len()-1) {

            let ind = remove_descend[i];
            _ = new_wave.drain(self.start_dats[ind]..).collect::<Vec<_>>();
            i += 1;
        }

        while (i < remove_descend.len()) {
            let ind = remove_descend[i];
            _ = new_wave.drain(self.start_dats[ind]..self.start_dats[ind+1]).collect::<Vec<_>>();
            i += 1;
        }

        Some(WaveformDef {
            wave: new_wave,
            spacer: self.spacer
        })


    }

    pub fn spacer(&self) -> usize {
        self.spacer
    }

    pub fn read_wave(&self) -> &Vec<f64> {
        &self.wave
    }

    pub fn raw_wave(&self) -> Vec<f64> {
        self.wave.clone()
    }

    pub fn start_dats(&self)  -> Vec<usize> {
         self.start_dats.clone()
    }

    pub fn point_lens(&self)  -> Vec<usize> {
        self.point_lens.clone()
    }

    pub fn seq(&self) -> &Sequence {
        self.seq
    }

    pub fn number_bp(&self) -> usize {
        self.seq.number_bp()
    }

    pub fn amount_data(&self) -> usize {
        self.wave.len()
    }


}

impl<'a, 'b> Add<&'b Waveform<'b>> for &'a Waveform<'a> {

    type Output = Waveform<'a>;

    fn add(self, wave2: &'b Waveform) -> Waveform<'a> {

        if !std::ptr::eq(self.seq, wave2.seq) || (self.spacer != wave2.spacer()) {
            panic!("These signals do not add! Spacers must be equal and waves must point to the same sequence!");
        }

        let other_wave = wave2.raw_wave();
        
        Waveform {
            wave: self.wave.iter().zip(other_wave).map(|(a, b)| a+b).collect(),
            spacer: self.spacer,
            point_lens: self.point_lens.clone(),
            start_dats: self.start_dats.clone(),
            seq: self.seq,
        }

    }

}

impl<'a, 'b> AddAssign<&'b Waveform<'b>> for Waveform<'a> {

    fn add_assign(&mut self, wave2: &'b Waveform) {

        if !std::ptr::eq(self.seq, wave2.seq) || (self.spacer != wave2.spacer()) {
            panic!("These signals do not add! Spacers must be equal and waves must point to the same sequence!");
        }

        let other_wave = wave2.raw_wave();
        
        let n = self.wave.len();


        //SAFETY: If we have the same sequence pointer and the same spacer, our lengths are provably always identical
        assume!(unsafe: other_wave.len() == n);

        for i in 0..n {
            self.wave[i] += other_wave[i];
        }


    }

}

impl<'a, 'b> Sub<&'b Waveform<'b>> for &'a Waveform<'a> {

    type Output = Waveform<'a>;

    fn sub(self, wave2: &'b Waveform<'b>) -> Waveform<'a> {

    if !std::ptr::eq(self.seq, wave2.seq) || (self.spacer != wave2.spacer()) {
            panic!("These signals do not add! Spacers must be equal and waves must point to the same sequence!");
    }
        let other_wave = wave2.raw_wave();
        
        Waveform {
            wave: self.wave.iter().zip(other_wave).map(|(a, b)| a-b).collect(),
            spacer: self.spacer,
            point_lens: self.point_lens.clone(),
            start_dats: self.start_dats.clone(),
            seq: self.seq,
        }

    }

}

impl<'a, 'b> SubAssign<&'b Waveform<'b>> for Waveform<'a> {

    fn sub_assign(&mut self, wave2: &'b Waveform) {

        if !std::ptr::eq(self.seq, wave2.seq) || (self.spacer != wave2.spacer()) {
            panic!("These signals do not add! Spacers must be equal and waves must point to the same sequence!");
        }

        let other_wave = wave2.raw_wave();
        
        let n = self.wave.len();

        //SAFETY: If we have the same sequence pointer and the same spacer, our lengths are always identical
        assume!(unsafe: other_wave.len() == n);
        
        for i in 0..n {
            self.wave[i] -= other_wave[i];
        }


    }

}

impl<'a> Mul<f64> for &'a Waveform<'a> {

    type Output = Waveform<'a>;

    fn mul(self, rhs: f64) -> Waveform<'a> {
        
        Waveform{
            wave: self.wave.iter().map(|a| a*rhs).collect(),
            spacer: self.spacer,
            point_lens: self.point_lens.clone(),
            start_dats: self.start_dats.clone(),
            seq: self.seq,
        }
    }

}

/// This is simply a `Waveform` in a form that can be serialized.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WaveformDef {
    wave: Vec<f64>,
    spacer: usize,

}
impl<'a> From<&'a Waveform<'a>> for WaveformDef {
    fn from(other: &'a Waveform) -> Self {
        Self {
            wave: other.raw_wave(),
            spacer: other.spacer(),
        }
    }
}

impl WaveformDef {

    //SAFETY: If the relevant lengths of the waveform do not correspond to the sequence, this will be unsafe. 
    //        By correspond, I mean that every element of point_lens needs to be 1+seq.block_lens/spacer
    //        And start_dats needs to match the cumulative sum of point_lens. self.wave.len() must also equal
    //        point_lens.last().unwrap()+start_dats.last().unwrap(). 
    //
    //ACCURACY:  Make sure that wave is in fact organized into the blocks implied by point_lens and start_dats.
    pub(crate) fn get_waveform<'a>(&self, seq: &'a Sequence) -> Result<Waveform<'a>, WaveCreationError> {
        Waveform::new(self.wave.clone(), seq, self.spacer)
    }

    pub fn len(&self) -> usize {
        self.wave.len()
    }

    pub fn spacer(&self) -> usize {
        self.spacer
    }
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "StudentsT")]
struct StudentsTDef {
    #[serde(getter = "StudentsT::location")]
    location: f64,
    #[serde(getter = "StudentsT::scale")]
    scale: f64,
    #[serde(getter = "StudentsT::freedom")]
    freedom: f64,
}

impl From<StudentsTDef> for StudentsT {

    fn from(def: StudentsTDef) -> StudentsT {
        StudentsT::new(def.location, def.scale, def.freedom).unwrap()
    }

}

/// This holds both the background distribution of binding and the prototype Kernel shapes
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Background {
    pub dist: BackgroundDist,
    pub kernel: [Kernel; KernelWidth::COUNT * KernelVariety::COUNT],
}

impl Background {

    pub fn new(sigma_background : f64, df : f64, fragment_length: f64) -> Result<Background, Box<dyn Error>> {

        let dist = BackgroundDist::new(sigma_background, df)?;
        
        let kernel: [Kernel; KernelWidth::COUNT * KernelVariety::COUNT] = core::array::from_fn(|a| {
            let var = KernelVariety::VARIANTS[a/KernelWidth::COUNT];
            let wid = KernelWidth::VARIANTS[a % KernelWidth::COUNT];
            Kernel::new(fragment_length, 1.0, wid, var)
        });
        Ok(Background{dist: dist, kernel: kernel})
    }


    //The functions get_near_u64, get_near_f64,  
    //and get_lookup_index are all legacy code from when 
    //I had to implement my distribution with lookup tables
    //and raw f64 bit manipulation to derive array indices.
    //They're mainly still here because they're so terrifying that
    //I love them and I can't bear to part with them.
    #[allow(dead_code)]
    fn get_near_u64(calc: f64) -> u64 {

        //These are based on using floats with a maximum of 8 binary sigfigs
        //after the abscissa and exponents ranging from -12 to 3, inclusive
        const MIN_DT_FLOAT: f64 = 0.000244140625;
        const MAX_DT_FLOAT: f64 = 15.96875;
        const EXP_OFFSET: u64 = (1023_u64-12) << 52;
        let mut bit_check: u64 = calc.to_bits();
       
        if calc.abs() < MIN_DT_FLOAT {
            bit_check = (calc.signum()*MIN_DT_FLOAT).to_bits();
        } else if calc.abs() > MAX_DT_FLOAT {
            bit_check = (calc.signum()*MAX_DT_FLOAT).to_bits();
        }

        (bit_check-EXP_OFFSET) & 0xfffff00000000000


    }
    
    #[allow(dead_code)]
    fn get_near_f64(calc: f64) -> f64 {
        f64::from_bits(Self::get_near_u64(calc))
    }

    #[allow(dead_code)]
    fn get_lookup_index(calc: f64) -> usize {

        let bitprep = Self::get_near_u64(calc);
        (((bitprep & 0x8_000_000_000_000_000) >> 51) 
      + ((bitprep & 0x7_fff_f00_000_000_000) >> 44)) as usize
        
    }

    pub fn ln_cd_and_sf(&self, calc: f64) -> (f64, f64) {
        self.dist.ln_cd_and_sf(calc)
    }
    pub fn cd_and_sf(&self, calc: f64) -> (f64, f64) {
        self.dist.cd_and_sf(calc)
    }
    pub fn cdf(&self, calc: f64) -> f64{
        self.dist.cdf(calc)
    }
    pub fn pdf(&self, calc: f64) -> f64{
        self.dist.pdf(calc)
    }

    pub fn ln_cdf(&self, calc: f64) -> f64{
        self.dist.ln_cdf(calc)
    }
    pub fn ln_sf(&self, calc: f64) -> f64{
        self.dist.ln_sf(calc)
    }
    pub fn ln_pdf(&self, calc: f64) -> f64{
        self.dist.ln_pdf(calc)
    }

    pub fn noise_spread_par(&self) -> f64 {
        self.dist.get_spread_par()
    }

    pub fn kernel_ref(&self, kernel_width: KernelWidth, kernel_variety: KernelVariety) -> &Kernel {
        &self.kernel[(kernel_width as usize)+KernelWidth::COUNT*(kernel_variety as usize)]
    }

    pub fn bp_span(&self) -> usize{
        self.kernel[0].len()
    }

    pub fn kernel_sd(&self) -> f64 {
        self.kernel[0].peak_width
    }


    pub fn sample<R: Rng+?Sized>(&self, rng: &mut R) -> f64 {
        self.dist.sample(rng)
    }

    pub fn get_sd_df(&self) -> (f64, f64) {
        self.dist.get_sd_df()
    }

}


#[derive(Clone)]
pub(crate) struct Noise<'a> {
    resids: Vec<f64>,
    extraneous_resids: Vec<f64>,
    pub background: &'a Background,
}

impl<'a> Noise<'a> {


    pub(crate) fn new(resids: Vec<f64>, extraneous_resids: Vec<f64>, background: &'a Background) -> Noise<'a> {
        Noise{ resids: resids, extraneous_resids: extraneous_resids, background: background}

    }

    pub(crate) fn resids(&self) -> Vec<f64> {
        self.resids.clone()
    }
    pub(crate) fn extraneous_resids(&self) -> Vec<f64> {
        self.extraneous_resids.clone()
    }
    
    
    pub(crate) fn dist(&self) -> BackgroundDist {
        self.background.dist.clone()
    }

    pub(crate) fn replace_extraneous(&mut self, extraneous_resids: Vec<f64>) {
        self.extraneous_resids = extraneous_resids;
    }

    pub(crate) fn noise_with_new_extraneous(&self, extraneous_resids: Vec<f64>) -> Noise<'a> {

        let mut new_noise = self.clone();
        new_noise.extraneous_resids = extraneous_resids;
        new_noise

    }

    
    pub(crate) fn rmse_noise(&self, spacer: usize) -> f64 {

        
        let extra = Waveform::generate_extraneous_binding(&self.background, spacer, &self.extraneous_resids);

        let a = self.resids.iter().map(|&a| a.powi(2)).sum::<f64>();
        let b = extra.into_iter().map(|a| a.powi(2)).sum::<f64>();

        let dist = (a+b).sqrt();

        dist

    }


    //This does not work on calculating the AD based on the cdf of the background directly.
    //It works on the CDF of the |residuals|
    pub(crate) fn ad_calc(&self, spacer: usize) -> f64 {

        //let time = Instant::now();
        let mut forward: Vec<f64> = self.resids().into_iter().map(|a| a.abs()).collect();

        //I don't need to take the absolute value here because I always generate the positive
        //values
        let mut extras = Waveform::generate_extraneous_binding(&self.background, spacer, &self.extraneous_resids);

        //let mut extras: Vec<f64> = self.extraneous_resids();
        forward.append(&mut extras);
        drop(extras);
        forward.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap()); //Tried having par sort, something is playing nastily with rayon

        let n = forward.len();
        let mut a_d = (n as f64)/2.0;
        for (i, &f) in forward.iter().enumerate() {

            //We rely on the fact that x >= 0 => cdf(x) >= sf(x)
            //We actually rely on the symmetry around 0 implying the fact that SF < 0.5 for all the absolute values of residuals
            let cdf = 2.0*self.background.dist.cdf(f)-1.0; //ln(CDF-(1-CDF)) = ln(1-SF-SF) = ln(1-2SF) = ln_1p(-2SF)
            let ln_sf = self.background.dist.ln_sf(f)+LN_2; //ln(1-(1-2SF)) = ln(2SF) = ln(2)+ln(SF)
            //The forumula uses (2n-2i+1), but also goes from 1 to n, not 0 to n-1
            let coeff = 2.0-((2*i+1) as f64)/(n as f64);

            a_d -= coeff*ln_sf+2.0*cdf;
        }
        a_d
    }

    
    //Note: this is kinda bad for some of the main body, but it's Fine(tm)
    fn low_val(la: f64) -> f64 {

        //This was a prior fit on some of the low data using a standard approximation for a sum of
        //chi squares
        //1.5*ln(3/2)-lngamma(1.5) is this magic constant
        //1.7687006706374098+la.ln()/2.0-3.0*la
        //
        let ln_la = la.ln();

        -1.4286015+ln_la*(-1.5951525-1.0 + ln_la*(-0.9975172 + ln_la *(-0.3141198-ln_la*0.1146426)))
    }

    fn high_val(ha: f64) -> f64 {

        -(ha).ln()/2.0+MULT_CONST_FOR_H*ha+ADD_CONST_FOR_H

    }


    fn weight_fn(a: f64) -> f64 {
        1.0/(1.0+(a/A0).powf(K))
    }


    pub fn ad_like(a: f64) -> f64 {


        //This has to exist because of numerical issues causing NANs otherwise
        //This is basically caused by cdf implementations that return 0.0.
        //My FastT implementation can't, but the statrs normal DOES. 
        if a == f64::INFINITY { return -f64::INFINITY;} 
        
        let hi = Self::high_val(a);
        if a >= HIGH_CUT {return hi;}

        let lo = Self::low_val(a);
        let w = Self::weight_fn(a);

        w*lo+(1.0-w)*hi
    }


}

pub(crate) const MULT_CONST_FOR_H: f64 = -1.835246330265487;
pub(crate) const ADD_CONST_FOR_H: f64 = 0.18593237745127472;


impl Mul<&Vec<f64>> for &Noise<'_> {

    type Output = f64;

    //fn mul(self, rhs: &Noise) -> f64 {
    fn mul(self, rhs: &Vec<f64>) -> f64 {

        //let rhs_r = rhs.resids();

        //if(self.resids.len() != rhs_r.len()){
        if self.resids.len() != rhs.len() {
            panic!("Residuals aren't the same length?!")
        }

        //self.resids.iter().zip(rhs_r).map(|(a,b)| a*b).sum()
        //self.resids.iter().zip(rhs).map(|(a,b)| a*b).sum()

        let mut sum = 0.0;

        for i in 0..self.resids.len() {
            sum += self.resids[i]*rhs[i];
        }

        sum
    }
}




#[cfg(test)]
mod tests{
   
    use super::*;
    use crate::sequence::{Sequence, NullSequence};
    use rand::Rng;
    use std::time::Instant;

    /*fn empirical_noise_grad(n: &Noise) -> Vec<f64>{

        let h = 0.00001;
        let back = n.background;
        let ad = n.ad_calc(1);
        let mut grad = vec![0_f64; n.resids.len()]; 
        for i in 0..grad.len() {
             let mut n_res = n.resids();
             n_res[i] += h;
             let nnoise = Noise::new(n_res, vec![], back);
             grad[i] = (nnoise.ad_calc()-ad)/h;
        }
        grad
    }*/

    #[test]
    fn wave_check(){

        let sd = 5;
        let height = 2.0;
        let _spacer = 5;
        let k = Kernel::new(sd as f64, height, KernelWidth::Wide, KernelVariety::Gaussian);

        let kern = k.get_curve();
        let kernb = &k*4.0;


        println!("kern len {}", (kern.len()));
        assert!(kern.len() == 6*(sd as  usize)+1);

        assert!(kern.iter().zip(kernb.get_curve()).map(|(&a,b)| ((b/a)-4.0).abs() < 1e-6).fold(true, |acc, mk| acc && mk));

        assert!((k.get_sd()-(sd as f64)).abs() < 1e-6);
    }

    #[test]
    fn real_wave_check(){
        let k = Kernel::new(5.0, 2.0, KernelWidth::Wide, KernelVariety::Gaussian);
        //let seq = Sequence::new_manual(vec![85;56], vec![84, 68, 72]);
        let seq = Sequence::new_manual(vec![192, 49, 250, 10, 164, 119, 66, 254, 19, 229, 212, 6, 240, 221, 195, 112, 207, 180, 135, 45, 157, 89, 196, 117, 168, 154, 246, 210, 245, 16, 97, 125, 46, 239, 150, 205, 74, 241, 122, 64, 43, 109, 17, 153, 250, 224, 17, 178, 179, 123, 197, 168, 85, 181, 237, 32], vec![84, 68, 72]);
        let mut signal = Waveform::create_zero(&seq, 5);

        let zeros: Vec<usize> = vec![0, 465, 892]; //Blocks terminate at bases 136, 737, and 1180

        let null_zeros: Vec<usize> = vec![144, 813];

        //This is in units of number of u8s
        let null_sizes: Vec<usize> = vec![78,17];

        println!("pre do {:?}", signal.raw_wave());
        unsafe{

        signal.place_peak(&k, 1, 20);

        let t = Instant::now();
        //kmer_propensities is tested by inspection, not asserts, because coming up with a good assert case was hard and I didn't want to
        //It passed the inspections I gave it, though
        println!("duration {:?}", t.elapsed());

        println!("{:?}", signal.raw_wave());
        println!("start dats {:?}", signal.start_dats());
        println!("point lens {:?}", signal.point_lens());
        //Waves are in the correct spot

        let ind = signal.start_dats()[1]+4;
        println!("{ind} {:?}", &signal.raw_wave()[15..25]);
        assert!((signal.raw_wave()[ind]-2.0).abs() < 1e-6);

        signal.place_peak(&k, 1, 2);

        //Waves are not contagious
        assert!(signal.raw_wave()[0..signal.start_dats()[1]].iter().fold(true, |acc, ch| acc && ((ch-0.0).abs() < 1e-6)));

        //point_lens: Vec<usize>,
        //start_dats: Vec<usize>,
        //

        signal.place_peak(&k, 1, 67);

        //Waves are not contagious
        assert!(signal.raw_wave()[(signal.start_dats[2])..(signal.start_dats[2]+signal.point_lens[2])].iter().fold(true, |acc, ch| acc && ((ch-0.0).abs() < 1e-6)));

        signal.place_peak(&k, 2, 20);

        let ind2 = signal.start_dats()[2]+4;

        //Waves are in the correct spot
        assert!((signal.raw_wave()[ind2]-2.0).abs() < 1e-6);

        //This is a check just for miri
        signal.place_peak(&k, 2, 70);

        //This is a check to make sure that miri can catch it when I obviously screw up. This test is designed to have UB. Do NOT uncomment it unless you're checking that Miri can catch stuff
        //signal.place_peak(&k, 2, 456);

        }

        let base_w = &signal*0.4;


        let background: Background = Background::new(0.25, 2.64, 25.0).unwrap();

        let mut rng = rand::thread_rng();

        let null_makeup: Vec<Vec<usize>> = null_sizes.iter().map(|&a| (0..(4*a)).map(|_| rng.gen_range(0_usize..4)).collect::<Vec<usize>>()).collect::<Vec<Vec<usize>>>();

        println!("{:?} null sizes", null_makeup.iter().map(|a| a.len()).collect::<Vec<_>>());

        let invented_null: NullSequence =  NullSequence::new(null_makeup);

        let invented_propensities = vec![1.0/65536_f64; 65536];

        let data_seq = unsafe{ AllDataUse::new_unchecked_data(base_w, &invented_null, &zeros, &null_zeros, &background, &invented_propensities, 3.0)};

        let noise: Noise = signal.produce_noise(&data_seq);

        let noi: Vec<f64> = noise.resids();


        let raw_resid = &signal-data_seq.data();

        let _w = raw_resid.raw_wave();

        println!("Noi {:?}", noi);



        //This is based on a peak with height 1.5 after accounting for binding and an sd equal to 5*spacer. Also, moved threshold to 4*spread
        let fake_extraneous: Vec<f64> = vec![1.5, 1.47029800996013, 1.47029800996013, 1.38467451957995, 1.38467451957995, 1.25290531711691, 1.25290531711691, 1.08922355561054, 1.08922355561054, 0.90979598956895, 0.90979598956895, 0.730128383939957, 0.730128383939957, 0.562966648277099, 0.562966648277099, 0.417055950679791, 0.417055950679791, 0.296848048625422, 0.296848048625422, 0.203002924854919, 0.203002924854919, 0.13338242618908, 0.13338242618908, 0.0842021442512006, 0.0842021442512006, 0.051071182101899, 0.051071182101899, 0.0297616421165554, 0.0297616421165554, 0.0166634948073635, 0.0166634948073635, 0.00896403434250891, 0.00896403434250891, 0.00463307311235515, 0.00463307311235515, 0.00230071601898669, 0.00230071601898669, 0.00109770362832071, 0.00109770362832071, 0.000503193941853768, 0.000503193941853768, 0.00022162254034805, 0.00022162254034805, 9.37822556622304e-05, 9.37822556622304e-05, 3.81290197742989e-05, 3.81290197742989e-05, 1.48942564587766e-05, 1.48942564587766e-05, 5.58997975811801e-06, 5.58997975811801e-06, 2.01571841644728e-06, 2.01571841644728e-06, 6.98357357367463e-07, 6.98357357367463e-07, 2.32462970355435e-07, 2.32462970355435e-07, 7.43460797875875e-08, 7.43460797875875e-08, 2.28449696170689e-08, 2.28449696170689e-08, 6.7445241934213e-09, 6.7445241934213e-09, 1.91311144428907e-09, 1.91311144428907e-09, 5.21383692185988e-10, 5.21383692185988e-10, 1.36522061467319e-10, 1.36522061467319e-10, 3.43460226846833e-11, 3.43460226846833e-11, 8.30191510755152e-12, 8.30191510755152e-12, 1.92800583770048e-12, 1.92800583770048e-12, 4.30196251333215e-13, 4.30196251333215e-13, 9.22259461905713e-14, 9.22259461905713e-14, 1.89962483236413e-14, 1.89962483236413e-14, 3.75933283071791e-15, 3.75933283071791e-15, 7.14795710294863e-16, 7.14795710294863e-16];

        let generated_extraneous = Waveform::generate_extraneous_binding(data_seq.background_ref(),5, &[1.5]);

        println!("theoretical extra {:?}, gen extra {:?}", fake_extraneous, generated_extraneous);

        assert!(fake_extraneous.len() == generated_extraneous.len(), "Not even generating the right length of kernel from binding heights!");

        for i in 0..fake_extraneous.len() {
            assert!((fake_extraneous[i]-generated_extraneous[i]).abs() < 1e-7, "generated noise not generating correctly");
        }


    }



    #[test]
    fn noise_check(){

        
        let background: Background = Background::new(0.25, 2.64, 5.0).unwrap();

        let n1 = Noise::new(vec![0.4, 0.4, 0.3, 0.2, -1.4],vec![], &background);
        let n2 = Noise::new(vec![0.4, 0.4, 0.3, -0.2, 1.4],vec![], &background);

        assert!(((&n1*&n2.resids())+1.59).abs() < 1e-6);

        println!("{:?}", n1.resids());
        println!("ad_calc {}", n1.ad_calc(1));

        //The + f64::MIN_POSITIVE is meant to do nothing for most input, but allow us to avoid
        //breaking if we have an EXACT zero coincidentally. 
        let mut noise_arr = n1.resids().into_iter().map(|a| a.abs()+f64::MIN_POSITIVE).collect::<Vec<f64>>();
        noise_arr.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let noise_length = noise_arr.len();
        let mut ad_try = (noise_length as f64)/2.0;

        for i in 0..noise_length {

            //let ln_cdf = n1.dist().ln_cdf(noise_arr[i]);
            //let ln_sf = n1.dist().ln_sf(noise_arr[noise_length-1-i]);
            let mult = 2.0+ (-2.0*((i+1) as f64)+1.0)/(noise_length as f64);

            let cdf = n1.dist().cdf(noise_arr[i]);
            let sf = 1.0-cdf;

            let real_cdf = (cdf-sf);
            let real_ln_sf = (sf-cdf).ln_1p();
            

            ad_try -= mult*real_ln_sf+2.0*real_cdf;
        }

        println!("ad_try {}", ad_try);

        assert!((n1.ad_calc(1)-ad_try).abs() < 1e-6, "AD calculation not matching theory without extraneous binding");
 
        //This is based on a peak with height 1.5 after accounting for binding and an sd equal to 5*spacer
        //let fake_extraneous: Vec<f64> = vec![1.50000000, 1.47029801, 1.47029801, 1.38467452, 1.38467452, 1.25290532, 1.25290532, 1.08922356, 1.08922356, 0.90979599, 0.90979599];

        let fake_extraneous: Vec<f64> = vec![1.5];
        
        let mut extra_resids = Waveform::generate_extraneous_binding(&background, 1, &fake_extraneous);//n1_with_extraneous.extraneous_resids();

        let n1_with_extraneous = n1.noise_with_new_extraneous(fake_extraneous);

        let mut extraneous_noises = n1_with_extraneous.resids().into_iter().map(|a| a.abs()).collect::<Vec<f64>>();

        extraneous_noises.append(&mut extra_resids);

        extraneous_noises.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let noise_length = extraneous_noises.len();
        let mut ad_try = (noise_length as f64)/2.0;
        
        for i in 0..noise_length {

            //let ln_cdf = n1.dist().ln_cdf(noise_arr[i]);
            //let ln_sf = n1.dist().ln_sf(noise_arr[noise_length-1-i]);
            let mult = 2.0+ (-2.0*((i+1) as f64)+1.0)/(noise_length as f64);

            let cdf = n1.dist().cdf(extraneous_noises[i]);
            let sf = 1.0-cdf;

            let real_cdf = (cdf-sf);
            let real_ln_sf = (sf-cdf).ln_1p();
            

            ad_try -= mult*real_ln_sf+2.0*real_cdf;
        }

        println!("extraneous ad {} ad_try {}", n1_with_extraneous.ad_calc(1), ad_try);

        assert!((n1_with_extraneous.ad_calc(1)-ad_try).abs() < 1e-6, "AD calculation not matching theory with extraneous binding");

        //This was for when I was using the Anderson Darling statistic. Since using the eth statistic, I don't have a gold standard to compare against. 
        //Calculated these with the ln of the numerical derivative of the fast implementation
        //of the pAD function in the goftest package in R
        //This uses Marsaglia's implementation, and is only guarenteed up to 8
        /*let calced_ads: [(f64, f64); 6] = [(1.0, -0.644472305368), 
                                           (0.46, 0.026743661078),
                                           (0.82, -0.357453548256),
                                           (2.82, -3.221007453503),
                                           (3.84, -4.439627768456),
                                           (4.24, -4.865014182520)];

        for pairs in calced_ads {

            //I'm considering 5% error mission accomplished for these
           //We're fighting to make sure this approximation is roughly
           //compaitble with another approximation with propogated errors
           //This will not be an exact science
            println!("Noi {} {} {} {}",Noise::ad_like(pairs.0), pairs.1, Noise::ad_like(pairs.0)-pairs.1, (Noise::ad_like(pairs.0)-pairs.1)/pairs.1);
            assert!(((Noise::ad_like(pairs.0)-pairs.1)/pairs.1).abs() < 5e-2); 

        } */



    }

    #[test]
    #[should_panic(expected = "Residuals aren't the same length?!")]
    fn panic_noise() {
        
        let background: Background = Background::new(0.25, 2.64, 5.0).unwrap();
        let n1 = Noise::new(vec![0.4, 0.4, 0.3, 0.2, -1.4], vec![], &background);
        let n2 = Noise::new(vec![0.4, 0.4, 0.3, -0.2], vec![], &background);

        let _ = &n1*&n2.resids();
    }









}


