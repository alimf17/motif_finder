
use std::{f64, fmt, fs};

use std::collections::{HashSet, VecDeque, HashMap};

use std::ops::{Index, IndexMut};

use std::io::{Read, Write};
//use std::time::{Duration, Instant};
use std::error::Error;
use std::ptr;
use std::mem::size_of;

use std::process::*;
use std::sync::Arc;

use core::fmt::{Debug, Formatter};
use core::slice::{Iter, IterMut};

use crate::MAX_TF_NUM;
use crate::waveform::{Kernel, Waveform, Noise, KernelWidth, KernelVariety, MULT_CONST_FOR_H, ADD_CONST_FOR_H};
use crate::sequence::{Sequence, NullSequence, BP_PER_U8, U64_BITMASK, BITS_PER_BP};
use crate::modified_t::{ContinuousLnCDF, SymmetricBaseDirichlet};
use crate::gene_loci::*;

use crate::{THRESH, NECESSARY_MOTIF_IMPROVEMENT};
use crate::data_struct::{AllData, AllDataUse};

use itertools::Itertools;

use gzp::{deflate::Mgzip, par::compress::{ParCompress, ParCompressBuilder}, syncz::{SyncZ, SyncZBuilder}, par::decompress::{ParDecompress, ParDecompressBuilder},ZWriter, Compression};


use rand::Rng;
use rand::prelude::IteratorRandom;
use rand::seq::SliceRandom;
use rand::distributions::{Distribution, Uniform, WeightedIndex};

use  rand_distr::weighted_alias::*;

use statrs::{consts};
use statrs::distribution::{Continuous, ContinuousCDF, LogNormal, Normal, Exp, Beta, Categorical, Discrete, DiscreteCDF, DiscreteUniform, LogNormalError};
use statrs::statistics::{Min, Max};

use thiserror::Error;

use ordered_float::*;

use once_cell::sync::Lazy;

use rayon::prelude::*;

use pathfinding::*;

use plotters::prelude::*;
use plotters::coord::types::RangedSlice;
use plotters::coord::Shift;

use ::num_traits::{Float, Num, NumAssignOps};
use num_traits::Bounded;

use log::warn;

use regex::Regex;

use bincode::config;
use serde::{Serialize, Deserialize};

use wyhash2::WyHash;

pub const SQRT_2: f64 = 1.41421356237;
pub const SQRT_3: f64 = 1.73205080757;

pub const LN_2: f64 = 0.6931471805599453;

/// This is 1/LN_2, the binary logarithm of Euler's number
pub const LB_E: f64 = 1.4426950408889634;

/// This lists the identity of DNA bases, associating them with integers
/// For those who want to modify this code, know that there are safety invariants
/// that rely on this being four elements long. 
pub const BPS: [char; 4] = ['A', 'C', 'G', 'T'];

/// This is all the variants of Bp collected into an array for easy iteration
pub const BP_ARRAY: [Bp; BASE_L] = [Bp::A, Bp::C, Bp::G, Bp::T];

/// This is the number of possible bases in the sequence. For those who want to 
/// modify this code, there are safety invariants which rely on being 4. 
pub const BASE_L: usize = BPS.len();

/// This is an array of arrays which stores the vertex coordinates of a base 
/// vector in simplex space. Each element corresponds to having all the 
/// "probability" of that base being in a particular position concentrated
/// at the corresponding element of BPS. EG, [0.,0.,1.0] is the coordinate
/// of a base position where all of the probability is at T. 
pub const SIMPLEX_VERTICES_POINTS :  [[f64; BASE_L-1]; BASE_L] = [[2.*SQRT_2/3. , 0.               , -1.0/3.          ],
                                                                    [ -SQRT_2/3.  , SQRT_2*SQRT_3/3. , -1.0/3.          ],
                                                                    [ -SQRT_2/3.  ,-SQRT_2*SQRT_3/3. , -1.0/3.          ],
                                                                    [0.0          , 0.0              , 1.0              ]];

/// This is an array of arrays which stores the vertex coordinates of a base 
/// vector in simplex space. The sequence of same index elements correspond to having all the 
/// "probability" of that base being in a particular position concentrated
/// at the corresponding element of BPS. EG, [2.*SQRT_2/3. , -SQRT_2/3., -SQRT_2/3., 0.0]
/// is the x coordinate of each vertex where the probability is all A, C, G, T, respectively
/// This is also the transpose of SIMPLEX_VERTICES_POINTS. 
pub const SIMPLEX_VERTICES: [[f64; BASE_L]; BASE_L-1] = [[2.*SQRT_2/3. , -SQRT_2/3., -SQRT_2/3., 0.0], 
                                                           [0.              , SQRT_2*SQRT_3/3.   , -SQRT_2*SQRT_3/3., 0.0],
                                                           [-1.0/3.          , -1.0/3.           , -1.0/3.           , 1.0]];


/// This inverts the simplex form of a base back to a set of probabilities of binding
/// To perform the inversion, multiply this matrix by the simplex vector appended
/// with 1.0 at the end. 
pub const INVERT_SIMPLEX: [[f64; BASE_L]; BASE_L] = [[ 1.0/SQRT_2,  0.0             , -0.25, 0.25], 
                                                     [-SQRT_2/4.0,  SQRT_2*SQRT_3/4., -0.25, 0.25],
                                                     [-SQRT_2/4.0, -SQRT_2*SQRT_3/4., -0.25, 0.25],
                                                     [0.0        ,  0.0             ,  0.75, 0.25]];


/// This is the dot product of the vertices of a simplex form of base vector
/// In particular, this assumes the simplex is regular with unit vector vertices 
pub const VERTEX_DOT: f64 = -1.0/((BASE_L-1) as f64);
       
/// This is the maximum length of the null scores array. The larger this is, the 
/// more null binding we capture from motifs. Note that, in practical inference, 
/// the motifs inferred are not going to have much null binding
pub const CAPACITY_FOR_NULL: usize = 1000;


const CLOSE: f64 = 1e-5;

/// MIN_BASE is the minimum length, in base vectors, that a PWM can be. 
/// If you are modifying this code, please notice the safety warning
/// # Safety 
/// Never, ever, ever, ever, EVER touch MIN_BASE. 
/// There are several safety guarentees that rely upon it being 8
/// YOU WILL GET UNDEFINED BEHAVIOR IF YOU IGNORE THIS
pub const MIN_BASE: usize = 8;

/// MAX_BASE is the maximum length, in base pairs, that a PWM can be. 
/// For a 4 base system like ours, the hardware limit on MAX_BASE is 32.
/// # Safety 
/// If modifying this code, do NOT set this above 32, ever. Furthermore, always
/// regenerate the AllData and AllDataUse you use if you have changed this.
/// A major part of how we can quickly and safely generate waveforms is that
/// there are guanrantees on sequence block sizes which rely on this
pub const MAX_BASE: usize = 20; 


/// MAX_HEIGHT is the maximum possible max peak height for a motif
/// In reality, this just needs to be set to a "big" value
/// I've never seen an log2(ratio of signal to input) over 15
pub const MAX_HEIGHT: f64 = 15.;


pub const LOG_HEIGHT_MEAN: f64 = 1.38629436112; //This is ~ln(4). Can't use ln in a constant, and this depends on no other variables
pub const LOG_HEIGHT_SD: f64 = 0.25;


static NORMAL_DIST: Lazy<Normal> = Lazy::new(|| Normal::new(0.0, 1.0).unwrap());

/// This is a probability distribution which samples the amount a height should
/// exceed the minimum possible height of the inference
pub static HEIGHT_PROPOSAL_DIST: Lazy<Exp> = Lazy::new(|| Exp::new(4.0).unwrap());

const PROB_POS_PEAK: f64 = 1.0;

/// This is the number of possible values that energy penalties can take
/// It is equal to -SCORE_THRESH/BASE_RESOLUTION, but Rust does not have 
/// const time division
pub const NUM_BASE_VALUES: usize = 4;

const BASE_RATIO_SDS: [f64; 3] = [0.01_f64, 0.05, 0.1];
const BASE_LINEAR_SDS: [f64; 3] = [0.01_f64, 0.05, 0.1];
 
const RATIO_LINEAR_SD_COMBO: Lazy<[[f64;2]; BASE_RATIO_SDS.len()*BASE_LINEAR_SDS.len()]> = 
                            Lazy::new(|| core::array::from_fn(|a| [BASE_RATIO_SDS[a/BASE_LINEAR_SDS.len()], BASE_LINEAR_SDS[a % BASE_LINEAR_SDS.len()]]));


const GOOD_V_BAD_CUT: [f64; 5] = [1.0,1.5, 2.0, 2.5, 3.0];

const HEIGHT_SDS: [f64; 3] = [0.05_f64, 0.1, 0.5];

//const NUM_MOVES: usize = 2*BASE_RATIO_SDS.len()*BASE_LINEAR_SDS.len()+HEIGHT_SDS.len()+2*SCALE_SDS.len()+8;
//const NUM_MOVES: usize = 1+HEIGHT_SDS.len()+GOOD_V_BAD_CUT+10;
//const VARIANT_NUMS: [usize; 9] = [BASE_RATIO_SDS.len()*BASE_LINEAR_SDS.len(), BASE_RATIO_SDS.len()*BASE_LINEAR_SDS.len(), HEIGHT_SDS.len(), 4, 1, 1, SCALE_SDS.len(), SCALE_SDS.len(), 1]; 
const VARIANT_NUMS: [usize; 8] = [1, GOOD_V_BAD_CUT.len(), HEIGHT_SDS.len(), 6, 1, 1, 1, 1]; 

const NUM_MOVES: usize = { 
    let mut i: usize  = 0;
    let mut size: usize = 0;
    while i < VARIANT_NUMS.len() {
        size += VARIANT_NUMS[i];
        i+=1;
    }
    size
};

static PICK_MOVE: Lazy<WeightedAliasIndex<u32>> = Lazy::new(|| WeightedAliasIndex::<u32>::new(vec![80_u32, 10, 20, 20, 80, 40, 0, 0]).expect("The weights should always be valid, with length equal to the length of VARIANT_NUMS"));

static SAMPLE_VARIANTS: Lazy<[Uniform<usize>; 8]> = Lazy::new(|| core::array::from_fn(|a| Uniform::new(0, VARIANT_NUMS[a])));

static VARIANT_CUMULATIVE: Lazy<[usize; 8]> = Lazy::new(|| core::array::from_fn(|a| if a == 0 {0} else {VARIANT_NUMS[0..a].iter().sum()}));

static HIST_END_NAMES: Lazy<[String; NUM_MOVES]> = Lazy::new(|| {
                                                   let m = (0..1).map(|_| format!("_rook_move.png"));
                                                   let m = m.chain(GOOD_V_BAD_CUT.iter().map(|a| format!("_resid_lean_cut_{a}.png"))); 
                                                   let m = m.chain(HEIGHT_SDS.iter().map(|a| format!("_height_sd_{a}.png"))); 
                                                   let m = m.chain(["_motif_birth.png".to_owned(), "_motif_death.png".to_owned(),"_motif_expand.png".to_owned(),
                                                   "_motif_contract.png".to_owned(), "_motif_split.png".to_owned(), "_motif_merge.png".to_owned(), "_base_leap.png".to_owned(), "_secondary_shuffle.png".to_owned()]);
                                                   let m = m.chain(["_kernel_move.png".to_owned()]);
                                                   let m = m.chain(["_random_motif.png".to_owned()]);
                                                   m.collect::<Vec<_>>().try_into().unwrap()
});

//This controls how much shuffling we do for the secondary base shuffling move
//Faster if SHUFFLE_BASES <= MIN_BASE. Also, each base shuffle calculates 
//4^SHUFFLE_BASES PWMs
const SHUFFLE_BASES: usize = 3;

//const MAX_VECT_COORD: f64 = 943.-9.5367431640625e-7;

const REDUCE_MOTIF_SCALE_MOVE: [f64; MAX_BASE+1-MIN_BASE] = [0.024800, 0.021573, 0.019085, 0.017109, 0.015503, 0.014171, 0.013049, 0.012091, 0.011264, 0.010543, 0.0099079, 0.0093451, 0.0088427];

/*


pub const THRESH: f64 = 1e-2; //SAFETY: This must ALWAYS be strictly greater than 0, or else we violate safety guarentees later.  

//This is roughly how much an additional motif should improve the ln posterior before it's taken seriously
//The more you increase this, the fewer motifs you will get, on average
const NECESSARY_MOTIF_IMPROVEMENT: f64 = 20.0_f64;
*/

/// These simply name the reversible jump moves for the purposes of tracking acceptance rates
pub const RJ_MOVE_NAMES: [&str; 6] = ["New motif", "Delete motif", "Extend motif", "Contract Motif", "New Motif Alt", "Kill Motif Alt"];//, "Split Motif", "Merge Motif"];

/// This is the worst possible energy penalty for a base mismatch, in binary 
/// energy units. It can be tinkered with for your applications, but must 
/// always be negative. I set it to -1.0 BEU, which correspond to a free energy
/// penalty of +0.4 kcal/mol
pub const SCORE_THRESH: f64 = -1.0;

//This needs to equal 1/(2^(-SCORE_THRESH)-1). Since our SCORE_THRESH is -1.0, 1/(2-1) = 1
const PWM_CONVERTER: f64 = 1.0;
const PWM_UNCONVERT: f64 = 1.0+PWM_CONVERTER;

const BARRIER: f64 = -SCORE_THRESH*2.0;

const SPLIT_SD: f64 = 0.05;
static SPLIT_DIST: Lazy<Normal> = Lazy::new(|| Normal::new(0.0, SPLIT_SD).unwrap());
const HEIGHT_SPLIT_SD: f64 = 0.001;
static HEIGHT_SPLIT_DIST: Lazy<Normal> = Lazy::new(|| Normal::new(0.0, HEIGHT_SPLIT_SD).unwrap());

static SPLIT_BASE_DIST: Lazy<Beta> = Lazy::new(|| Beta::new(0.5, 1.0).unwrap());

static BASE_CHOOSE_DIST: Lazy<Categorical> = Lazy::new(|| {
    let slant = 5_i32;
    let weights: Vec<f64> = (0..NUM_BASE_VALUES).map(|a| (((a+1) as f64)/(NUM_BASE_VALUES as f64)).powi(slant)-(((a as f64)/(NUM_BASE_VALUES as f64)).powi(slant))).collect();
    Categorical::new(&weights).unwrap()
});

static BASE_PENALTY_PRIOR: Lazy<Categorical> = Lazy::new(|| {
    let front_load = 0.9_f64; //This will break things if it's not between 0 and 1, exclusive
    let mut weights: Vec<f64> = vec![1.0; NUM_BASE_VALUES];
    weights[0] = front_load*(NUM_BASE_VALUES as f64)/(1.0-front_load);
    Categorical::new(&weights).unwrap()
});

//static BASE_CHOOSE_ALT: Lazy<DiscreteUniform> = Lazy::new(|| DiscreteUniform::new(0, (NUM_BASE_VALUES-1) as i64 ).unwrap());

static BASE_CHOOSE_ALT: Lazy<Categorical> = Lazy::new(|| Categorical::new(&vec![1.0; NUM_BASE_VALUES]).unwrap());

//These were determined through numerical experiments and rough fits
const ADDITIVE_WEIGHT_CONST: f64 = -344000.0;
const MULTIPLY_WEIGHT_CONST: f64 = 43000.0;
const EXPONENT_WEIGHT_CONST: f64 = 1.4;
 
/// This is the granularity of the energy penalties of bases
/// All energies are in binary energy units
pub const BASE_RESOLUTION: f64 = 0.25;


//BEGIN BASE

//NOTE: These bases being assigned to these values specifically is necessary.
//If you add bases, in addition to safety considerations for conversions, 
//you need to also ensure that reverse complements are assigned to values 
//which are the ^ 3 duals of each other
/// The type which indexes base pairs: a dependent usize with `BASE_L` variants. 
#[repr(usize)]
#[derive(Debug, Copy, Hash, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Bp {
    A = 0,
    C = 1, 
    G = 2, 
    T = 3,
}

trait SampleToBase<K: TryFrom<i64>+Bounded + Clone + Num + NumAssignOps + Ord+ std::fmt::Debug>: Distribution<f64> + Discrete<K ,f64> + DiscreteCDF<K ,f64> {
    fn sample_energy<R: Rng+?Sized>(&self, rng: &mut R) -> f64 {
        (self.sample(rng)+1.0)*BASE_RESOLUTION*SCORE_THRESH
    }
    fn energy_ln_pmf(&self, energy: f64) -> f64  where <K as TryFrom<i64>>::Error: Error {
        let sample: K = (((energy/(BASE_RESOLUTION*SCORE_THRESH))-1.0) as i64).try_into().expect(format!("How did you get here {energy} {} {:?}??", energy/(BASE_RESOLUTION*SCORE_THRESH)-1.0, TryInto::<K>::try_into((energy/(BASE_RESOLUTION*SCORE_THRESH)-1.0) as i64)).as_str());
        self.ln_pmf(sample)
    }
}

impl SampleToBase<u64> for Categorical {}
impl SampleToBase<i64> for DiscreteUniform {}

/// Bp is always aligned like usize
impl From<Bp> for usize {
    fn from(bp: Bp) -> Self {
        //SAFETY: Bp is repr to a valid usize
        unsafe{std::mem::transmute::<Bp, usize>(bp)}
    }
}

/// Returns the Bp with value `bp` when bp < BASE_L. Otherwise, returns an error
impl TryFrom<usize> for Bp {
    type Error = &'static str;
    fn try_from(bp: usize) -> Result<Self, Self::Error> {
        if bp < BASE_L {
            //SAFETY: We checked that this is a valid variant and Bp is repr(usize)
            Ok(unsafe{std::mem::transmute::<usize, Bp>(bp)})
        } else {
            Err("usize not a valid base pair!")
        }
    }

}


impl Bp {

    /// The complement operation. In unmodified DNA bps, A complements T, C complements G
    pub fn complement(&self) -> Bp {
        /*match self {
            Bp::A => Bp::T,
            Bp::C => Bp::G,
            Bp::G => Bp::C,
            Bp::T => Bp::A, 
        }*/

        //Compiler recognized this optimization at high enough optimization
        //SAFETY: enum never returns incorrect variant, which is always in bounds
        unsafe{std::mem::transmute::<usize, Bp>((*self as usize) ^ 3)}
    }
 
    /// An unsafe conversion from usize to Bp
    /// # SAFETY
    /// Must be ABSOLUTELY sure that bp is < BASE_L (4). 
    pub unsafe fn usize_to_bp(bp: usize) -> Bp {
        std::mem::transmute::<usize, Bp>(bp)
    }
}

impl fmt::Display for Bp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {

        //Writing the peak height (unitless) and the peak width (as defined by # base pairs from one end to another)
        
        let chara = match self{
            Bp::A => "A",
            Bp::C => "C", 
            Bp::G => "G", 
            Bp::T => "T",
        };
        write!(f, "{}", chara)?;
        Ok(())
    }
}

//This felt like a lot of operations, but because it relies exclusively on constants
//(a binary power and -1 at that), it's about as fast as a single addition 
/// This rounds an f64 to the nearest float that is an integer multiple 
/// of SCORE_THRESH*BASE_RESOLUTION rounding ties. However, 
/// if num != 0.0, it will not return 0.0, and instead give SCORE_THRESH*BASE_RESOLUTION
pub fn base_ceil(num: f64) -> f64 { 
    let a = SCORE_THRESH*BASE_RESOLUTION*(num/(SCORE_THRESH*BASE_RESOLUTION)).round();

    if a == 0.0 && num != 0.0 { SCORE_THRESH*BASE_RESOLUTION } else {a} 
}

#[derive(Serialize, Deserialize, Debug, Clone)]
/// This is the struct which hold the negative free energies of binding in a position
/// There are several invariants which must be maintained:
/// * There must exist a unique Bp in Base which which has a negative free energy of 0.0
/// * All other positions in the Base must have a value which is less than 0.0 but greater than or equal to SCORE_THRESH
/// In general, functions which produce a Base will either statically guarentee this or return a Result
pub struct Base {
   scores: [ f64; BASE_L],
}

/// This is the error type for attempting to initialize an invalid Base
#[derive(Debug)]
pub enum InvalidBase {
    NegativeBase,
    NoSingleBestBase(Vec<Bp>),
}

impl fmt::Display for InvalidBase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            InvalidBase::NegativeBase => "No negative values allowed in a Base!".to_owned(),
            InvalidBase::NoSingleBestBase(bps) => format!("Every Base needs one best value == 0.0! {:?}", bps),
        };
        write!(f, "{}",message)?;
        Ok(())
    }
}

impl Error for InvalidBase {}

impl PartialEq for Base {
    fn eq(&self, other: &Self) -> bool {
        self.dist_sq(Some(other)) < CLOSE    
    }
} 

impl Index<Bp> for Base {
type Output = f64;

    fn index(&self, bp: Bp) -> &f64 {
        &self.scores[bp as usize]
    }

}

impl IndexMut<Bp> for Base {

    fn index_mut(&mut self, bp: Bp) -> &mut f64 {
        &mut self.scores[bp as usize]
    }

}


//Note: in our paper, we talk about the parameterization of these by 
//      the dimensionless free energies. This is actually slightly different 
//      than what we actually do in the code: we use exp(-dimensionless free energies).
//      The parameterizations are equivalent
impl Base {

    /// This creates a new Base by taking your scores, subtracting the maximum 
    /// value from all values, and then capping any penalties smaller than SCORE_THRESH
    pub fn new(scores: [ f64; BASE_L]) -> Base {

        let mut scores = scores;

        let max = Self::max(&scores);

        for i in 0..scores.len() {
                scores[i] = scores[i]-max;
                if scores[i] < SCORE_THRESH { scores[i] = SCORE_THRESH;}
        }

        Base { scores }
    }


    /// This outputs the Base whose value is 0.0 in the best_bp position and SCORE_THRESH otherwise
    pub fn max_selective(best_bp: Bp) -> Base {

        let scores:  [ f64; BASE_L] = core::array::from_fn(|_| SCORE_THRESH);

        let mut base = Base { scores };

        base[best_bp] = 0.0;

        base

    }

    /// This takes a set of proposed scores, and ensures they follow the invariants
    /// of base by reflection once the maximum is subtracted off, then ensures that
    /// they are rounded to the nearest integer multiple of BASE_RESOLUTION*SCORE_THRESH
    pub fn new_by_reflections(scores: [f64; BASE_L]) -> Base {

        let mut scores = scores;

        let max = Self::max(&scores);

        for i in 0..scores.len() {
                scores[i] = scores[i]-max;
        }

        scores.iter_mut().for_each(|pos| {
            if *pos < SCORE_THRESH { 
                *pos = *pos - 2.0*SCORE_THRESH*((*pos+SCORE_THRESH)/(2.0*SCORE_THRESH)).floor();
                if *pos > 0.0 { *pos = -*pos;}
                if *pos == 0.0 { *pos += BASE_RESOLUTION*SCORE_THRESH; }
            }
            *pos = base_ceil(*pos)
        });

        Base { scores }

    }

    /// This takes a set of probabilities of binding, and converts them to a Base 
    /// by assuming the total sample number of props is counts, then adds
    /// psuedo as a psuedocount before converting to scores
    /// # Errors
    ///   When props has a negative value or if there is no unique maximum prop
    pub fn new_with_pseudo(props: [f64; BASE_L], counts: f64, pseudo: f64) -> Result<Base, InvalidBase> {

        let mut props = props;
        
        let mut any_neg: bool = false;

        let max_prop = Self::max(&props);

        let mut count_max: usize = 0;

        let mut maxes: Vec<Bp> = Vec::with_capacity(BASE_L);

        for i in 0..props.len() {
            any_neg |= props[i] < 0.0 ;
            if props[i] == max_prop { count_max += 1; maxes.push(BP_ARRAY[i]);}
        }

        if any_neg{
            return Err(InvalidBase::NegativeBase);
        }

        if count_max > 1 {
            return Err(InvalidBase::NoSingleBestBase(maxes));
        }

        let sum = props.iter().sum::<f64>();

        if sum == 0.0 { 
            for p in props.iter_mut() { *p += pseudo; } 
        } else {
            for p in props.iter_mut() {*p = (*p/sum)*counts+pseudo;}
        }

        let scores: [f64; BASE_L] = core::array::from_fn(|a| props[a].log2());

        Ok(Base::new(scores))
    }

    /// This takes a set of probabilities of binding from a MEME PWM converts them to a Base
    /// by multiplying by a background distribution. If discretize is true, 
    /// penalties are rounded to the nearest integer multiple of BASE_RESOLUTION*SCORE_THRESH
    /// # Errors
    ///   When props or background_dist have a negative value or if there is no unique maximum prop
    pub fn from_pwm(props: [f64; BASE_L], background_dist: Option<[f64; BASE_L]>, discretize: bool) -> Result<Base, InvalidBase> {

        let mut props = props;

        let mut background = background_dist.unwrap_or([0.25; BASE_L]);

        let mut sum: f64 = 0.0;
        

        let mut too_many_max: bool = false;

        for prop in background {
            if prop <= 0.0 {
                eprintln!("Your background distribution needs to have positive density");
                return Err(InvalidBase::NegativeBase); }
            sum += prop;
        }

        background = core::array::from_fn(|a| background[a]/(sum*0.25));

        let mut max = 0_f64;
        let mut sum: f64 = 0.0;
        for (i, prop) in props.iter_mut().enumerate() {
            if *prop < 0.0 { return Err(InvalidBase::NegativeBase); }
            *prop = *prop * background[i];
            if *prop >= max {
                if *prop == max { too_many_max = true;} else { too_many_max = false; max = *prop;}
            }
            sum += *prop
        }

        if too_many_max { return Err(InvalidBase::NoSingleBestBase(props.iter().enumerate().filter_map(|(i,&s)| if s == max {Some(BP_ARRAY[i])} else {None}).collect()));}
        let scores: [f64; BASE_L] = core::array::from_fn(|a| {
            let mut val = (props[a]/sum + (max/sum)*PWM_CONVERTER).log2() - (2.0*(max/sum)*PWM_CONVERTER).log2();

            if discretize { base_ceil(val) } else {val} 
        });

        Ok(Base::new(scores))

    }
    pub fn from_pwm_alt(props: [f64; BASE_L], background_dist: Option<[f64; BASE_L]>, discretize: bool) -> Result<Base, InvalidBase> {

        let mut props = props;

        let mut background = background_dist.unwrap_or([0.25; BASE_L]);

        let mut sum: f64 = 0.0;
        

        let mut too_many_max: bool = false;

        for prop in background {
            if prop <= 0.0 {
                eprintln!("Your background distribution needs to have positive density");
                return Err(InvalidBase::NegativeBase); }
            sum += prop;
        }

        background = core::array::from_fn(|a| background[a]/(sum*0.25));

        let mut max = 0_f64;
        let mut sum: f64 = 0.0;
        for (i, prop) in props.iter_mut().enumerate() {
            if *prop < 0.0 { return Err(InvalidBase::NegativeBase); }
            *prop = *prop * background[i];
            if *prop >= max {
                if *prop == max { too_many_max = true;} else { too_many_max = false; max = *prop;}
            }
            sum += *prop
        }

        if too_many_max { return Err(InvalidBase::NoSingleBestBase(props.iter().enumerate().filter_map(|(i,&s)| if s == max {Some(BP_ARRAY[i])} else {None}).collect()));}
        let scores: [f64; BASE_L] = core::array::from_fn(|a| {
            let mut val = if props[a] == 0.0 {-PWM_CONVERTER} else {(props[a]/sum).log2()}; // (props[a]/sum + (max/sum)*PWM_CONVERTER).log2() - (2.0*(max/sum)*PWM_CONVERTER).log2();

            if discretize { base_ceil(val) } else {val} 
        });

        Ok(Base::new(scores))

    }

    /// This takes a Base and converts it to set of probabilities of binding from a MEME PWM
    /// by dividing out a background distribution. If discretize is true, 
    /// penalties are rounded to the nearest integer multiple of BASE_RESOLUTION*SCORE_THRESH
    /// # Errors
    ///   When background_dist has a negative value 
    pub fn to_pwm(&self,background_dist: Option<[f64; BASE_L]> ) -> Result<[f64; BASE_L], InvalidBase> {

        let mut background = background_dist.unwrap_or([0.25; BASE_L]);

        let mut sum: f64 = 0.0;

        for prop in background {
            if prop <= 0.0 {
                eprintln!("Your background distribution needs to have positive density");
                return Err(InvalidBase::NegativeBase); }
            sum += prop;
        }

        background = core::array::from_fn(|a| background[a]/sum);
        let mut sum = 0_f64;
        let mut scores: [f64; BASE_L] = core::array::from_fn(|a| {
            let mut s = (self.scores[a]).exp2()*PWM_UNCONVERT-PWM_CONVERTER;
            s /= background[a];
            sum += s;
            s
        });
 
        //println!("Base {:?}, scores {:?} uncon {PWM_UNCONVERT} sum {sum}", self, scores);
        Ok(core::array::from_fn(|a| scores[a]/sum))
              
    }

    /// This generates a random Base by sampling energy penalties 
    /// from integer multiples of BASE_RESOLUTION*SCORE_THRESH <= SCORE_THRESH,
    /// favoring worse penalities, and then picking a best Bp uniformly
    pub fn rand_new<R: Rng + ?Sized>(rng: &mut R) -> Base {


        let best = rng.gen_range(0..BASE_L);
        let samps: [f64;BASE_L-1] = core::array::from_fn(|_| BASE_CHOOSE_DIST.sample_energy(rng)); 

        let nonbest: [usize; BASE_L-1] = (0..BASE_L).filter(|&a| a != best).collect::<Vec<_>>().try_into().unwrap();

        let mut att =  [0.0_f64; BASE_L];

        nonbest.into_iter().enumerate().for_each(|(i,a)| {att[a] = samps[i];});


        Base { scores: att}
    }
   
    /// This generates a random Base by sampling energy penalties
    /// from integer multiples of BASE_RESOLUTION*SCORE_THRESH <= SCORE_THRESH,
    /// favoring less strict penalities, and then picking a best Bp uniformly
    pub fn rand_new_prior<R: Rng + ?Sized>(rng: &mut R) -> Base {


        let best = rng.gen_range(0..BASE_L);
        let samps: [f64;BASE_L-1] = core::array::from_fn(|_| BASE_PENALTY_PRIOR.sample_energy(rng));

        let nonbest: [usize; BASE_L-1] = (0..BASE_L).filter(|&a| a != best).collect::<Vec<_>>().try_into().unwrap();

        let mut att =  [0.0_f64; BASE_L];

        nonbest.into_iter().enumerate().for_each(|(i,a)| {att[a] = samps[i];});


        Base { scores: att}
    }


    /// This generates a random Base by uniformly sampling energy penalties 
    /// from integer multiples of BASE_RESOLUTION*SCORE_THRESH <= SCORE_THRESH,
    /// then picking a best Bp uniformly
    pub fn rand_new_alt<R: Rng + ?Sized>(rng: &mut R) -> Base {

        let best = rng.gen_range(0..BASE_L);
        let samps: [f64;BASE_L-1] = core::array::from_fn(|_| base_ceil(BASE_CHOOSE_ALT.sample_energy(rng))); 

        let nonbest: [usize; BASE_L-1] = (0..BASE_L).filter(|&a| a != best).collect::<Vec<_>>().try_into().unwrap();

        let mut att =  [0.0_f64; BASE_L];

        nonbest.into_iter().enumerate().for_each(|(i,a)| {att[a] = samps[i];});


        Base { scores: att}
    }


    /// This generates a random Base with best as its best Bp by sampling energy penalties 
    /// from integer multiples of BASE_RESOLUTION*SCORE_THRESH <= SCORE_THRESH, favoring 
    /// worse penalities
    pub fn from_bp<R: Rng + ?Sized>(best: Bp, rng: &mut R) -> Base {

        Base::rand_new(rng).make_best(best)

    }

    /// This generates a random Base with best as its best Bp by sampling energy penalties
    /// from integer multiples of BASE_RESOLUTION*SCORE_THRESH <= SCORE_THRESH, favoring
    /// less strict penalities
    pub fn from_bp_prior<R: Rng + ?Sized>(best: Bp, rng: &mut R) -> Base {

        Base::rand_new_prior(rng).make_best(best)

    }


    /// This generates a random Base with best as its best Bp by uniformly sampling 
    /// energy penalties from integer multiples of BASE_RESOLUTION*SCORE_THRESH <= SCORE_THRESH
    pub fn from_bp_alt<R: Rng + ?Sized>(best: Bp, rng: &mut R) -> Base {

        Base::rand_new_alt(rng).make_best(best)

    }

    /// This takes Self and returns a Base with its current best Bp swapped with best
    /// If best is its best Bp, returns a clone of Self
    pub fn make_best(&self, best: Bp) -> Base {

        let which_b = self.best_base();

        if best != which_b {
            self.swap_pair(best, which_b)
        } else {
            self.clone()
        }


    }

    /// This takes Self and returns a copy with the values in bp1 and bp2 swapped
    pub fn swap_pair(&self, bp1: Bp, bp2: Bp) -> Base {
        
        let mut base2 = self.clone();
        base2[bp1] = self[bp2];
        base2[bp2] = self[bp1];

        base2

    }

    /// This takes Self and returns which Bp has the best binding
    pub fn best_base(&self) -> Bp {
        //Safety: self.scores has a size of BASE_L, so this always produces a valid result
        unsafe{ std::mem::transmute::<usize, Bp>(Self::argmax(&self.scores))}
 
        //Genuine surprise: this is worse when I profile it on godbolt
        /*let mut best = if (self[Bp::A] < self[Bp::C]) { Bp::C } else {Bp::A};
        if (self[best] < self[Bp::G]) { best = Bp::G; }
        if (self[best] < self[Bp::T]) { best = Bp::T; }
        best*/
    }

    /// This takes self and returns an ordered array of every Bp which isn't best
    pub fn all_non_best(&self) -> [Bp; BASE_L-1] {
        let best = self.best_base();


        match best {

            Bp::A => [Bp::C, Bp::G, Bp::T],
            Bp::C => [Bp::A, Bp::G, Bp::T],
            Bp::G => [Bp::A, Bp::C, Bp::T],
            Bp::T => [Bp::A, Bp::C, Bp::G],
        }

    }

    /// This function has two behaviors. 
    /// If base is None, this returns the squared Aitchison magnitude of Self
    /// If base is Some(&Base), it returns the squared Aitchison distance 
    /// between it and self. 
    pub fn dist_sq(&self, base: Option<&Base>) -> f64 {
        
        match base {

            None => 4.0*self.scores.iter().map(|a| a.powi(2)).sum::<f64>()-self.scores.iter().map(|a| a).sum::<f64>().powi(2),
            Some(other) => {
                //Remember, this is the DISTANCE. Which is the inner product OF THE DIFFERENCE. Not the inner product of the two 
                4.0*self.scores.iter().zip(other.scores.iter()).map(|(a,b)| ((a-b).powi(2))).sum::<f64>()-(self.scores.iter().zip(other.scores.iter()).map(|(a,b)| (a-b)).sum::<f64>()).powi(2)
            },
        }


    }


    /// This function converts the scores to probabilities of binding. 
    /// However, this does so directly, without any pseudocount. 
    /// Thus, it should not be used for PWM generation, which assigns 
    /// 0.0 weight to any location with penalty == SCORE_THRESH.
    pub fn as_probabilities(&self) -> [f64; BASE_L] {

        let mut props: [f64; BASE_L] = core::array::from_fn(|a| self.scores[a].exp2());
        let summed = props.iter().sum::<f64>();
        let ret: [f64; BASE_L] = core::array::from_fn(|a| props[a]/summed); // self.scores.iter().map(|&a| a/summed).collect::<Vec<_>>().try_into().unwrap();
        ret
    }

    /// This gives the ln density of the Base vector as a function of its penalties.
    /// This uses `[BASE_PENALTY_PRIOR]` as the prior for the secondary penalties
    pub fn prior_per_base(&self) -> f64 {
        self.all_non_best().into_iter().map(|a| BASE_CHOOSE_ALT.energy_ln_pmf(self[a])).sum()
    }

    pub fn scores(&self) -> [f64; BASE_L] {
        self.scores
    }

    pub fn taxicab(&self) -> f64 {
        self.scores.iter().sum::<f64>()
    }

    /// This converts a Base to the graphical representation of it as a simplex
    pub fn as_simplex(&self) -> [f64; BASE_L-1] {
        let probs = self.as_probabilities();
        Self::prob_slice_to_simplex(&probs)
    }

    fn prob_slice_to_simplex(probs: &[f64; BASE_L]) -> [f64; BASE_L-1] {
        let simplex: [f64; BASE_L-1] = SIMPLEX_VERTICES.iter().map(|a| a.iter().zip(probs.iter()).map(|(&s, &b)| s*b).sum::<f64>())
                                                       .collect::<Vec<f64>>().try_into().unwrap();

        simplex
    }


    //This converts our base representation directly to a vector space representation
    //In particular, our bases form a metric space under 
    //"Vector addition" is elementwise multiplication and then renormalize
    //"Scalar multiplication" is elementwise powering to the scalar and then renormalize
    //And the metric is the Aitchison metric
    //This vector space is isometric to R3 under the transformation specified in the functions
    //Where if p = 1/(2*(1+e)) and q = e*p
    //e1 = [p,q,q,p], e2 = [p,q,p,q], e3 = [p,p,q,q]
    fn base_to_vect(&self) -> [f64; BASE_L-1] {

        //let ln_vec: [f64; BASE_L] = core::array::from_fn(|a| (self.scores[a]+VEC_PAD_EPS).ln());
        /*let ln_vec: [f64; BASE_L] = core::array::from_fn(|a| {
            let b = (self.scores[a]).ln();
            if b.is_infinite() { b.signum()*(MAX_VECT_COORD)} else {b}
        });*/


        Base::reflect_triple_to_finite(
            &[self[Bp::C]+self[Bp::G]-(self[Bp::A]+self[Bp::T]),
              self[Bp::C]+self[Bp::T]-(self[Bp::A]+self[Bp::G]),
              self[Bp::T]+self[Bp::G]-(self[Bp::A]+self[Bp::C])])
    }

    fn vect_to_base(base_as_vec: &[f64; BASE_L-1]) -> Self {



        //This clamp exists to make sure that we don't over or underflow exp and break things
        let sanitized_base: [f64; BASE_L-1] = Self::reflect_triple_to_finite(base_as_vec); 


        let unnormalized_base: [f64; BASE_L] = 
            [-(sanitized_base[0]+sanitized_base[1]+sanitized_base[2])*0.25, 
              (sanitized_base[0]+sanitized_base[1]-sanitized_base[2])*0.25,
              (sanitized_base[0]-sanitized_base[1]+sanitized_base[2])*0.25,
             (-sanitized_base[0]+sanitized_base[1]+sanitized_base[2])*0.25];


        Base::new(unnormalized_base)
    }


    fn rook_move<R: Rng + ?Sized>(&self, rng: &mut R) -> Base {

        let mut new_base = self.clone();
        //SAFETY: all_non_best() never returns an empty value
        let change_bp = unsafe { *(self.all_non_best().choose(rng).unwrap_unchecked())};

        let energy: f64 = BASE_CHOOSE_ALT.sample_energy(rng);

        new_base[change_bp] = energy;

        new_base

    }


    fn max( arr: &[f64]) -> f64 {
        arr.iter().fold(f64::NAN, |x, y| x.max(*y))
    }

    fn argmax( arr: &[f64] ) -> usize {

        if arr.len() == 1 {
            return 0;
        }

        let mut arg = 0;

        for ind in 1..(arr.len()) {
            arg = if arr[ind] > arr[arg] { ind } else {arg};
        }

        arg

    }

    fn check_vec_kept_best_bp(vec_res: &[f64; BASE_L-1], best_base: Bp) -> [f64; BASE_L-1] {

        match best_base {
               Bp::A => [  vec_res[0]+vec_res[1] ,   vec_res[1]+vec_res[2] ,   vec_res[2]+vec_res[0] ],
               Bp::C => [-(vec_res[0]+vec_res[1]),  -vec_res[1]+vec_res[2] ,   vec_res[2]-vec_res[0] ],
               Bp::G => [ -vec_res[0]+vec_res[1] ,   vec_res[1]-vec_res[2] , -(vec_res[2]+vec_res[0])],
               Bp::T => [  vec_res[0]-vec_res[1] , -(vec_res[1]+vec_res[2]),  -vec_res[2]+vec_res[0] ],
        }
    }


    fn reflect_triple_to_finite(triple: &[f64; BASE_L-1]) -> [f64; BASE_L-1] {
    

        let mut return_triple = triple.clone();
        let mut maybe_id = Self::terminate_refect(triple); 

        let mut is_some = maybe_id.is_some();

        while is_some {

            return_triple = Self::reflect_outer_barrier(return_triple, maybe_id.expect("only here if we're some"));
            maybe_id = Self::terminate_refect(&return_triple);
            is_some = maybe_id.is_some();
        }

       

        return_triple


    }

    fn terminate_refect(triple: &[f64; BASE_L-1]) -> Option<usize> {
        

        let first_checks = [triple[0].abs()+triple[1].abs()-BARRIER,
                            triple[0].abs()+triple[2].abs()-BARRIER,
                            triple[1].abs()+triple[2].abs()-BARRIER];

        let mut first_barrier: Option<(usize, f64)> = None;

        for (id, &check) in first_checks.iter().enumerate() {
            first_barrier = match first_barrier {
                Some(bar) => if check > bar.1 { Some((id, check))} else {Some(bar)},
                None => if check > 0.0 { Some((id, check)) } else {None},
            };
        }

        first_barrier.map(|a| a.0)

    }

    //NOTE: At some point, I should turn this algorithm into some kind of division based on
    //      the "penalty" scores to do this reflection in one go, but I'm currently just a 
    //      little too out of it to figure it out and it's not urgent
    fn reflect_outer_barrier(triple: [f64; BASE_L-1], id: usize) -> [f64; BASE_L-1] {

        //let i = ((((2*BASE_L-1) as f64)-(((2*BASE_L-1).pow(2)-8*id) as f64).sqrt())/2.) as usize;
        //let i = ((7_f64-(49_f64-8.*(id as f64)).sqrt())/2.) as usize;
        //let j = (id+((BASE_L-1-i)*(BASE_L-2-i))/2)-((BASE_L-1)*BASE_L-2)/2;
        //let j = (id+((3-i)*(2-i))/2)-3;

        //Implementation based on this math stack exchange answer: https://math.stackexchange.com/questions/4908539/reflective-dodecadron
        //
        let triple_in_basis = [(-triple[0]+triple[1]+triple[2])/(4.0* BARRIER),
                               ( triple[0]-triple[1]+triple[2])/(4.0* BARRIER),
                               ( triple[0]+triple[1]-triple[2])/(4.0* BARRIER)];

        let reduce_reflection: [f64;BASE_L-1] = core::array::from_fn(|a| triple_in_basis[a].fract());

        let triple = [(reduce_reflection[1]+reduce_reflection[2])*2.0* BARRIER,
                      (reduce_reflection[0]+reduce_reflection[2])*2.0* BARRIER,
                      (reduce_reflection[0]+reduce_reflection[1])*2.0* BARRIER];

        let (i, j): (usize, usize) = [(0, 1), (0, 2), (1, 2)][id];

        let sum = triple[i]+triple[j];
        let diff = triple[i]-triple[j];

        let mut new_triple = triple;

        let swap = new_triple[i];
        new_triple[i] = new_triple[j];
        new_triple[j] = swap;

        //It's possible that both sum and diff are problems
        //We need to pick the biggest problem, because that's the first reflection
        if sum.abs() > diff.abs() {
            let sign = sum.signum();
            new_triple[i] = -new_triple[i]+sign* BARRIER;
            new_triple[j] = -new_triple[j]+sign* BARRIER;
        } else {
            let sign  = diff.signum();
            new_triple[i] += sign* BARRIER;
            new_triple[j] -= sign* BARRIER;
        }
       
        new_triple

    }
/*
    pub fn add_in_vector_space(&self, addend: [f64; BASE_L-1], confine_base: bool) -> Self {

       let tetra = self.base_to_vect();
       let mut vec_res: [f64; BASE_L-1] = core::array::from_fn(|a| tetra[a]+addend[a]);
       vec_res = Self::reflect_triple_to_finite(&vec_res);
       if confine_base {
           let best_base = self.best_base();

           let mut conds_check = Self::check_vec_kept_best_bp(&vec_res, best_base); 
           let mut conds: [bool; BASE_L-1] = core::array::from_fn(|a| conds_check[a] > 0.);
           let mut failed = conds.iter().any(|&a| a);

           #[cfg(test)]
           let mut failures: usize = 0;

           while failed {
               let next_reflect = conds_check.iter().enumerate().max_by(|(_, &a), (_, b)| a.partial_cmp(b).unwrap()).expect("conds_check is not empty").0;
               vec_res = match next_reflect {
                   0 => { //If A or C
                       if (best_base as usize) < 2 {[-vec_res[1], -vec_res[0], vec_res[2]]} else {[vec_res[1], vec_res[0], vec_res[2]]}
                   }, 
                   1 => {
                       if (best_base == Bp::A) || (best_base == Bp::T) {[vec_res[0], -vec_res[2], -vec_res[1]]} else {[vec_res[0], vec_res[2], vec_res[1]]}
                   }, 
                   2 => { //If A or G
                       if (best_base as usize) & 1 == 0 {[-vec_res[2], vec_res[1],-vec_res[0]]} else {[vec_res[2], vec_res[1],vec_res[0]]}
                   }, 
                   _ => unreachable!(),
               };
               
               conds_check = Self::check_vec_kept_best_bp(&vec_res, best_base);
               conds = core::array::from_fn(|a| conds_check[a] > 0.);
               failed = conds.iter().any(|&a| a);

               #[cfg(test)]
               {
                   failures += 1;

               
                   if (failures > 0) && (failures % 10 == 0) {
                   
                       println!("{} failures: {:?}", failures, vec_res);
               
                   }
               }
           }
       }
       Base::vect_to_base(&vec_res)

    }*/



    /// This clones the Base and reverses it. This also gives the complement of the base
    pub fn rev(&self) -> Base {

        let mut revved = self.scores.clone();

        revved.reverse();

        Self::new(revved)

    }


    
    /// This returns the binding score of base bp
    pub fn bind(&self, bp: usize) -> f64 {
        self.scores[bp]
    }


}

//This should never be part of the public API
fn nicer_ln_ratio(numerator: [f64;2], denominator: [f64;2]) -> f64 {

    let n = [exponent_mantissa_f64(numerator[0]), exponent_mantissa_f64(numerator[1])];
    let d = [exponent_mantissa_f64(denominator[0]), exponent_mantissa_f64(denominator[1])];

    //I am essentially always expecting the same sign on exponents, here
    //So I want the subtractions to happen before the additions to lower
    //chances of overflow
    let mut exponent = (n[0].0-d[0].0)+(n[1].0-d[1].0);

    let mant_ratio = n[0].1 * n[1].1 /(d[0].1 * d[1].1);

    if !( (mant_ratio > 0.0) && (mant_ratio.is_finite()) )  { return mant_ratio.ln(); }

    let (miniexp, final_mant) = exponent_mantissa_f64(mant_ratio);

    exponent += miniexp;

    final_mant.ln()+(exponent as f64)*LN_2


}

//Ignores sign. Will not produce a stable result for infs or nans
//Also shifts zeros and subnormals to be the smallest normal number, effectively
//This should never be part of the public API
fn exponent_mantissa_f64(f: f64) -> (i64, f64) {

    if f.abs() < f64::MIN_POSITIVE { return ((f64::MIN_EXP-1) as i64, 1.0); }
    
    /*if f.is_infinite() { return (0, f64::INFINITY); }
    if f.is_nan() { return (0, f64::NAN);} */
    
    /*if f.is_subnormal() { 
        let lead = f.to_bits().leading_zeros();
        let e = -1011-(lead as i64);
        let m = f64::from_bits(0x3ff_0_0000_0000_0000_u64+(f.to_bits() << (lead+1)));
        return (e, m);
    }*/ //I don't want to deal with subnormal ratios on bases for now

    (((f.to_bits() & 0x7FF_0_0000_0000_0000) >> 52) as i64-1023, 
     f64::from_bits((f.to_bits() & 0x000_F_FFFF_FFFF_FFFF)+0x3FF_0_0000_0000_0000))
}



fn reflect_abs_height(a: f64, min_height: f64) -> f64 {
    if (a > min_height) && (a < MAX_HEIGHT) {
        return a;
    }

    let reflect_check = (a.abs()-min_height)/(MAX_HEIGHT-min_height);
    let try_end = reflect_check-reflect_check.floor();
    let flip_space = (reflect_check.floor() as i32 & 0x01_i32) == 1;

    if flip_space {
        MAX_HEIGHT-try_end*(MAX_HEIGHT-min_height)
    } else {
        min_height+try_end*(MAX_HEIGHT-min_height)
    }
    //-reflect_cond*(MAX_HEIGHT-min_height)*a_sign*a.signum()+a_sign*a
}



//BEGIN MOTIF


/// This contains a single motif, along with all of the information needed
/// for that motif to generate its occupancy trace. This does not do checks
/// on whether it is legal for this `Motif` to exist in a particular experiment. 
/// Note that `Motif`s pre reserve the memory to contain `MAX_BASE` `Base`s.
/// Also, note that if a `Motif` has a length less than `MIN_BASE`, it is legal 
/// to construct, but will cause panics or errors when it is used in inference.
/// # Safety
/// It is a safety invariant that Motif NEVER has more than MAX_BASE `Base`s, 
/// and never less than MIN_BASE `Base`s. 
/// All methods which allow you to input custom motifs will automatically truncate
/// to a maximum of MAX_BASE elements if you try to give it a longer motif, and 
/// will pad your motif with `Bp::A` or `Base::max_selective(Bp::A)` if you 
/// give it a smaller motif
#[derive(Serialize, Deserialize, Clone)]
pub struct Motif {

    peak_height: f64,
    kernel_width: KernelWidth,
    kernel_variety: KernelVariety,
    pwm: Vec<Base>,

}

fn exact_capacity<T>(squish: &mut Vec<T>, capacity: usize) {
    if capacity <= squish.len() {squish.shrink_to_fit(); return;}
    if squish.capacity() > capacity {
        squish.shrink_to(capacity);
    } else {
        squish.reserve_exact(capacity-squish.capacity())
    }
}

impl Motif {


    pub fn raw_pwm(pwm: Vec<Base>, peak_height: f64, kernel_width: KernelWidth, kernel_variety: KernelVariety) -> Motif {

        let mut pwm = pwm;
        
        pwm.truncate(MAX_BASE);
       
        while pwm.len() < MIN_BASE {
            pwm.push(Base::max_selective(Bp::A));
        }

        exact_capacity(&mut pwm, MAX_BASE);

        let m = Motif {
            peak_height: peak_height,
            kernel_width: kernel_width,
            kernel_variety: kernel_variety,
            pwm: pwm,
        };

        m
    }

    /// This takes a sequence of best bases and turns them into a `Motif` with that Bp
    /// sequence as its best, with all energy penalties at SCORE_THRESH
    pub fn from_motif_max_selective(best_bases: Vec<Bp>, peak_height: f64, kernel_width: KernelWidth, kernel_variety: KernelVariety) -> Motif {

        let mut best_bases = best_bases;
        best_bases.truncate(MAX_BASE);

        while best_bases.len() < MIN_BASE {
            best_bases.push(Bp::A);
        }

        let mut pwm: Vec<Base> = best_bases.iter().map(|a| Base::max_selective(*a)).collect();

        exact_capacity(&mut pwm, MAX_BASE);

        let m = Motif {
            peak_height: peak_height,
            kernel_width: kernel_width,
            kernel_variety: kernel_variety,
            pwm: pwm,
        };

        m
    }

    pub fn from_meme_pwm(meme_pwm: Vec<[f64; BASE_L]>, background_dist: Option<[f64; BASE_L]>, kernel_width: KernelWidth, kernel_variety: KernelVariety, min_height: f64) -> Result<Motif, InvalidBase> {

        let mut background = background_dist.unwrap_or([0.25; BASE_L]);

        let mut sum: f64 = 0.0;

        for prop in background {
            if prop <= 0.0 { 
                eprintln!("Your background distribution needs to have positive density");
                return Err(InvalidBase::NegativeBase); }
            sum += prop;
        }

        background = core::array::from_fn(|a| background[a]/sum);
 
        let mut meme_pwm = meme_pwm;

        meme_pwm.truncate(MAX_BASE);

        let try_pwm = meme_pwm.into_iter().map(|a| {
            Base::from_pwm_alt(a, background_dist, true)
        }).collect::<Result<Vec<Base>, _>>();

        let mut pwm = try_pwm?;

        while pwm.len() < MIN_BASE { pwm.push(Base::max_selective(Bp::A));}

        exact_capacity(&mut pwm, MAX_BASE);

        Ok(Motif {
            peak_height: min_height, 
            kernel_width: kernel_width,
            kernel_variety: kernel_variety,
            pwm: pwm, 
        })

    }

    pub fn rand_height_pwm<R: Rng + ?Sized>(pwm: Vec<Base>, height_dist: &impl ::rand::distributions::Distribution<f64>, rng: &mut R) -> Motif {

        let mut pwm = pwm;
        while pwm.len() < MIN_BASE { pwm.push(Base::max_selective(Bp::A));}
        pwm.truncate(MAX_BASE);

        exact_capacity(&mut pwm, MAX_BASE);

        let sign: f64 = rng.gen();
        let sign: f64 = if sign < PROB_POS_PEAK {1.0} else {-1.0};

        let peak_height: f64 = sign*height_dist.sample(rng);
        let kernel_width: KernelWidth = rng.gen();
        let kernel_variety: KernelVariety = rng.gen();
        Self::raw_pwm(pwm, peak_height, kernel_width, kernel_variety)
    }


    /// This generates a motif where best_bases is the sequence of `Bp`s with 
    /// the best matches. Then, it generates energy penalties randomly, favoring
    /// stricter penalties and using discretely spaced penalties
    pub fn from_motif<R: Rng + ?Sized>(best_bases: Vec<Bp>, height_dist: &impl ::rand::distributions::Distribution<f64>, rng: &mut R) -> Motif {

        let mut best_bases = best_bases;
        while best_bases.len() < MIN_BASE {
            best_bases.push(Bp::A);
        }

        best_bases.truncate(MAX_BASE);
        
        let mut pwm: Vec<Base> = best_bases.iter().map(|a| Base::from_bp(*a, rng)).collect();
        exact_capacity(&mut pwm, MAX_BASE);

        //let height_dist: truncatedlognormal = truncatedlognormal::new(log_height_mean, log_height_sd, min_height, max_height).unwrap();


        let sign: f64 = rng.gen();
        let sign: f64 = if sign < PROB_POS_PEAK {1.0} else {-1.0};

        let peak_height: f64 = sign*height_dist.sample(rng);

        let kernel_width: KernelWidth = rng.gen();
        let kernel_variety: KernelVariety = rng.gen();

        Motif {
            peak_height: peak_height,
            kernel_width: kernel_width,
            kernel_variety: kernel_variety,
            pwm: pwm,
        }



    }

    /// This generates a motif where best_bases is the sequence of `Bp`s with
    /// the best matches. Then, it generates energy penalties randomly, favoring
    /// stricter penalties and using discretely spaced penalties
    pub fn from_motif_prior<R: Rng + ?Sized>(best_bases: Vec<Bp>, height_dist: &impl ::rand::distributions::Distribution<f64>, rng: &mut R) -> Motif {

        let mut best_bases = best_bases;
        while best_bases.len() < MIN_BASE {
            best_bases.push(Bp::A);
        }

        best_bases.truncate(MAX_BASE);

        let mut pwm: Vec<Base> = best_bases.iter().map(|a| Base::from_bp_prior(*a, rng)).collect();
        exact_capacity(&mut pwm, MAX_BASE);

        //let height_dist: truncatedlognormal = truncatedlognormal::new(log_height_mean, log_height_sd, min_height, max_height).unwrap();


        let sign: f64 = rng.gen();
        let sign: f64 = if sign < PROB_POS_PEAK {1.0} else {-1.0};

        let peak_height: f64 = sign*height_dist.sample(rng);

        let kernel_width: KernelWidth = rng.gen();
        let kernel_variety: KernelVariety = rng.gen();

        Motif {
            peak_height: peak_height,
            kernel_width: kernel_width,
            kernel_variety: kernel_variety,
            pwm: pwm,
        }



    }

    
    /// This generates a motif where best_bases is the sequence of `Bp`s with
    /// the best matches. Then, it generates energy penalties randomly, uniformly
    /// sampling the discretely spaced penalties
    pub fn from_motif_alt<R: Rng + ?Sized>(best_bases: Vec<Bp>,height_dist: &impl ::rand::distributions::Distribution<f64>, rng: &mut R) -> Motif {

        let mut best_bases = best_bases;
        while best_bases.len() < MIN_BASE {
            best_bases.push(Bp::A);
        }

        best_bases.truncate(MAX_BASE);

        let mut pwm: Vec<Base> = best_bases.iter().map(|a| Base::from_bp_alt(*a, rng)).collect();
        
        exact_capacity(&mut pwm, MAX_BASE);

        //let height_dist: truncatedlognormal = truncatedlognormal::new(log_height_mean, log_height_sd, min_height, max_height).unwrap();


        let sign: f64 = rng.gen();
        let sign: f64 = if sign < PROB_POS_PEAK {1.0} else {-1.0};

        let peak_height: f64 = sign*height_dist.sample(rng);

        let kernel_width: KernelWidth = rng.gen();
        let kernel_variety: KernelVariety = rng.gen();

        Motif {
            peak_height: peak_height,
            kernel_width: kernel_width,
            kernel_variety: kernel_variety,
            pwm: pwm,
        }


    }

    /// This generates a motif where best_bases is the sequence of `Bp`s with 
    /// the best matches. Then, it generates energy penalties randomly, favoring
    /// stricter penalties that are integer multiples of BASE_RESOLUTION*SCORE_THRESH
    /// and less than or equal to SCORE_THRESH
    pub fn from_motif_with_height<R: Rng + ?Sized>(best_bases: Vec<Bp>, height: f64, rng: &mut R) -> Motif {

        let mut best_bases = best_bases;
        while best_bases.len() < MIN_BASE {
            best_bases.push(Bp::A);
        }

        best_bases.truncate(MAX_BASE);

        let mut pwm: Vec<Base> = best_bases.iter().map(|a| Base::from_bp(*a, rng)).collect();
        exact_capacity(&mut pwm, MAX_BASE);

        let kernel_width: KernelWidth = rng.gen();
        let kernel_variety: KernelVariety = rng.gen();
        //let height_dist: truncatedlognormal = truncatedlognormal::new(log_height_mean, log_height_sd, min_height, max_height).unwrap();

        Motif {
            peak_height: height,
            kernel_width: kernel_width,
            kernel_variety: kernel_variety,
            pwm: pwm,
        }


    }
    
    /// This generates a motif where best_bases is the sequence of `Bp`s with
    /// the best matches. Then, it generates energy penalties randomly, uniformly
    /// sampling penalties that are integer multiples of BASE_RESOLUTION*SCORE_THRESH
    /// and less than or equal to SCORE_THRESH
    pub fn from_motif_with_height_alt<R: Rng + ?Sized>(best_bases: Vec<Bp>, height: f64, rng: &mut R) -> Motif {

        let mut best_bases = best_bases;
        while best_bases.len() < MIN_BASE {
            best_bases.push(Bp::A);
        }

        best_bases.truncate(MAX_BASE);

        let mut pwm: Vec<Base> = best_bases.iter().map(|a| Base::from_bp_alt(*a, rng)).collect();
        exact_capacity(&mut pwm, MAX_BASE);

        let kernel_width: KernelWidth = rng.gen();
        let kernel_variety: KernelVariety = rng.gen();
        //let height_dist: truncatedlognormal = truncatedlognormal::new(log_height_mean, log_height_sd, min_height, max_height).unwrap();

        Motif {
            peak_height: height,
            kernel_width: kernel_width,
            kernel_variety: kernel_variety,
            pwm: pwm,
        }


    }


    /// This generates a random Motif that is valid for `seq`. To generate
    /// the Motif, it first uniformly samples a length from MIN_BASE..=MAX_BASE,
    /// then uniformly samples a random valid motif of that length from seq,
    /// and finally generates energy penalties randomly, favoring stricter 
    /// penalties that are integer multiples of BASE_RESOLUTION*SCORE_THRESH
    /// and less than or equal to SCORE_THRESH. 
    pub fn rand_mot<R: Rng + ?Sized>(seq: &Sequence, height_dist: &impl ::rand::distributions::Distribution<f64>,rng: &mut R) -> Motif {


        let num_bases = rng.gen_range(MIN_BASE..(MAX_BASE+1));

        let mot = seq.random_valid_motif(num_bases, rng);


        Self::from_motif(mot, height_dist, rng)


    }
    
    /// This generates a random Motif that is valid for `seq`. To generate
    /// the Motif, it first uniformly samples a length from MIN_BASE..=MAX_BASE,
    /// then uniformly samples a random valid motif of that length from seq,
    /// and finally generates energy penalties randomly, uniformly sampling 
    /// penalties that are integer multiples of BASE_RESOLUTION*SCORE_THRESH
    /// and less than or equal to SCORE_THRESH. 
    pub fn rand_mot_alt<R: Rng + ?Sized>(seq: &Sequence,height_dist: &impl ::rand::distributions::Distribution<f64>, rng: &mut R) -> Motif {


        let num_bases = rng.gen_range(MIN_BASE..(MAX_BASE+1));

        let mot = seq.random_valid_motif(num_bases, rng);


        Self::from_motif_alt(mot, height_dist, rng)


    }


    /// This generates a random Motif that is valid for `seq`. To generate
    /// the Motif, it first uniformly samples a length from MIN_BASE..=MAX_BASE,
    /// then uniformly samples a random valid motif of that length from seq,
    /// and finally generates energy penalties randomly, favoring stricter 
    /// penalties that are integer multiples of BASE_RESOLUTION*SCORE_THRESH
    /// and less than or equal to SCORE_THRESH. 
    pub fn rand_mot_with_height<R: Rng + ?Sized>(peak_height: f64, seq: &Sequence,rng: &mut R) -> Motif {


        let num_bases = rng.gen_range(MIN_BASE..(MAX_BASE+1));

        let mot = seq.random_valid_motif(num_bases, rng);

        Self::from_motif_with_height(mot, peak_height, rng)

    }
    
    /// This generates a random Motif that is valid for `seq`. To generate
    /// the Motif, it first uniformly samples a length from MIN_BASE..=MAX_BASE,
    /// then uniformly samples a random valid motif of that length from seq,
    /// and finally generates energy penalties randomly, uniformly sampling 
    /// penalties that are integer multiples of BASE_RESOLUTION*SCORE_THRESH
    /// and less than or equal to SCORE_THRESH. 
    pub fn rand_mot_with_height_alt<R: Rng + ?Sized>(peak_height: f64, seq: &Sequence,rng: &mut R) -> Motif {


        let num_bases = rng.gen_range(MIN_BASE..(MAX_BASE+1));

        let mot = seq.random_valid_motif(num_bases, rng);

        Self::from_motif_with_height_alt(mot, peak_height, rng)

    }

    
    /// This generates a random Motif that is valid for `seq`. To generate,
    /// the Motif, it uniformly samples a random valid motif of length 
    /// num_bases.min(MAX_BASE).max(MIN_BASE) from seq, then generates energy 
    /// penalties randomly, favoring stricter penalties that are integer multiples
    /// of BASE_RESOLUTION*SCORE_THRESH and less than or equal to SCORE_THRESH. 
    pub fn rand_mot_with_height_and_motif_len<R: Rng + ?Sized>(peak_height: f64, num_bases: usize, seq: &Sequence, rng: &mut R) -> Motif {

        let num_bases = num_bases.min(MAX_BASE).max(MIN_BASE);

        let mot = seq.random_valid_motif(num_bases, rng);

        let mut pwm: Vec<Base> = mot.iter().map(|a| Base::from_bp(*a, rng)).collect();
        exact_capacity(&mut pwm, MAX_BASE);

        let kernel_width: KernelWidth = rng.gen();
        let kernel_variety: KernelVariety = rng.gen();

        Motif {
            peak_height: peak_height,
            kernel_width: kernel_width,
            kernel_variety: kernel_variety,
            pwm: pwm,
        }



    }

    /// This generates a random Motif that is valid for `seq`. To generate,
    /// the Motif, it uniformly samples a random valid motif of length
    /// num_bases.min(MAX_BASE).max(MIN_BASE) from seq, then generates
    /// energy penalties randomly,uniformly sampling penalties that are integer
    /// multiples of BASE_RESOLUTION*SCORE_THRESH and less than or equal to SCORE_THRESH.
    pub fn rand_mot_with_height_and_motif_len_alt<R: Rng + ?Sized>(peak_height: f64, num_bases: usize, seq: &Sequence, rng: &mut R) -> Motif {

        let num_bases = num_bases.min(MAX_BASE).max(MIN_BASE);

        let mot = seq.random_valid_motif(num_bases, rng);

        let mut pwm: Vec<Base> = mot.iter().map(|a| Base::from_bp_alt(*a, rng)).collect();
        exact_capacity(&mut pwm, MAX_BASE);

        let kernel_width: KernelWidth = rng.gen();
        let kernel_variety: KernelVariety = rng.gen();

        Motif {
            peak_height: peak_height,
            kernel_width: kernel_width,
            kernel_variety: kernel_variety,
            pwm: pwm,
        }



    }


    fn make_opposite(&self) -> Motif {

        let mut opposite = self.clone();

        opposite.peak_height = -self.peak_height;
        opposite

    }

    /// This takes `self` and compares it to the valid motifs in `seq`.
    /// If `self`'s best kmer is valid in `seq`, it is unchanged and this 
    /// returns `None`. If `self`'s best bases are not valid in `seq`, this
    /// finds the smallest Hamming distance to the best motif of `self` in which
    /// seq has valid motifs, then it randomly samples that pool to find a new
    /// sequence of best bps. Then, each mismatched `Base` in `self` has their
    /// penalties swapped to match the new best kmer, and 
    /// `Some(`the minimum Hamming distance`)` is returned. 
    pub fn scramble_to_close_random_valid<R: Rng + ?Sized>(&mut self, seq: &Sequence, rng: &mut R) -> Option<usize> {

        let best_motif = self.best_motif();
        if seq.kmer_in_seq(&best_motif) {return None;}

        let mut hamming = 0;
        let mut kmer_ids: Vec<usize> = Vec::new();

        while kmer_ids.len() == 0 {
            hamming += 1;
            kmer_ids = seq.all_kmers_within_hamming(&best_motif, hamming);
        }

        *self = self.scramble_by_id_to_valid(*kmer_ids.choose(rng).expect("We should never try this unless we have kmers to choose from"), false, seq);

        Some(hamming)

    }

    fn scramble_by_id_to_valid(&self, id: usize, opposite: bool, seq: &Sequence) -> Motif {

        let mut new_mot: Motif = if opposite {self.make_opposite()} else {self.clone()};
        let new_best: u64 = seq.idth_unique_kmer(self.len(), id);
        let old_best: u64 = Sequence::kmer_to_u64(&self.best_motif());

        for i in 0..self.len() {

            let old_base: u64 = (old_best & (U64_BITMASK << (BITS_PER_BP*i))) >> (BITS_PER_BP*i);
            let new_base: u64 = (new_best & (U64_BITMASK << (BITS_PER_BP*i))) >> (BITS_PER_BP*i);

            if new_base != old_base {
                //SAFETY: new_base is always constrained to be less than BASE_L (4)
                new_mot.pwm[i] = new_mot.pwm[i].make_best(unsafe{Bp::usize_to_bp(new_base as usize)});
            }

        }

        new_mot

    }
    
    /// This returns `None` if `self` and `kmer` do not have the same length.
    /// Otherwise, this takes each position `i` where `self`'s best kmer mismatches
    /// `kmer`, and swaps the penalties of the 'i'th `Base` so that `kmer[i]` 
    /// is its best Bp.
    pub fn scramble_by_kmer(&self, kmer: &[Bp]) -> Option<Motif> {

        if self.len() != kmer.len() { return None; }
        
        let mut new_mot: Motif = self.clone();

        let old_best = self.best_motif();

        for i in 0..self.len() {
            new_mot.pwm[i] = new_mot.pwm[i].make_best(kmer[i]);
        }

        Some(new_mot)
    }


    /// This returns a vector of Motifs with all the values of `self`, but
    /// where every position in `to_scramble` has each possible combination of 
    /// pair wise swaps that do not alter the best kmer. If an element of 
    /// `to_scramble` is `>= self.len()`, it is simply omitted
    pub fn scramble_secondaries(&self, to_scramble: &[usize]) -> Vec<Motif> {

        let to_scramble: Vec<usize> = to_scramble.iter().filter_map(|&a| if a < self.len() {Some(a)} else {None}).collect();
        
        let final_length = (BASE_L-1).pow(to_scramble.len() as u32);
        
        let swapped_bases: Vec<[Base; ((BASE_L-1)*(BASE_L-2))/2]> = to_scramble.iter().map(|&a| {
            let non_best = self.pwm[a].all_non_best();
            non_best.into_iter().combinations(2).map(|n| self.pwm[a].swap_pair(n[0], n[1])).collect::<Vec<_>>().try_into().expect("Should always be the correct length")
        }).collect();


        let mut final_swaps: Vec<Motif> = Vec::with_capacity(final_length);

        let mut power_scrambles = to_scramble.iter().powerset();

        final_swaps.push(self.clone());

        let _ = power_scrambles.next();

        for scramble in power_scrambles {
          
            for gray_code in 0..((BASE_L-1).pow(scramble.len() as u32)) {
                
                let inner_swap = scramble.iter().enumerate().map(|(i, &_s)| (i, ((gray_code/((BASE_L-1).pow(i as u32))) % (BASE_L-1))));

                let mut mot_to_be = self.clone();

                for (b, s) in inner_swap { mot_to_be.pwm[to_scramble[b]] = swapped_bases[b][s].clone(); }

                final_swaps.push(mot_to_be);
            }
        }

        final_swaps

    }



    //HELPERS

    pub fn pwm(&self) -> Vec<Base> {
        self.pwm.clone()
    }

    pub fn pwm_ref(&self) -> &Vec<Base> {
        &self.pwm
    }

    pub fn best_motif(&self) -> Vec<Bp> {
        self.pwm.iter().map(|a| a.best_base()).collect()
    }

    pub fn best_motif_string(&self) -> String {
        self.pwm.iter().map(|a| format!("{}", a.best_base())).collect::<Vec<_>>().concat()
    }

    pub fn rev_complement(&self) -> Vec<Base> {
        self.pwm.iter().rev().map(|a| a.rev()).collect()
    }

    pub fn rev_complement_best_motif(&self) -> String {
        self.pwm.iter().rev().map(|a| format!("{}", a.best_base().complement())).collect::<Vec<_>>().concat()
    }

    pub fn peak_height(&self) -> f64 {
        self.peak_height
    }

    pub fn kernel_width(&self) -> KernelWidth {
        self.kernel_width
    }
    pub fn kernel_variety(&self) -> KernelVariety {
        self.kernel_variety
    }

    pub fn len(&self) -> usize {
        self.pwm.len()
    }

    pub fn taxicab(&self) -> f64 {
        self.peak_height + self.pwm.iter().map(|a| a.taxicab()).sum::<f64>()
    }


    //PRIORS

    pub fn pwm_prior(&self, seq: &Sequence) -> f64 {

        if seq.kmer_in_seq(&self.best_motif()) {
            //We have to normalize by the probability that our kmer is possible
            //Which ultimately is to divide by the fraction (number unique kmers)/(number possible kmers)
            //number possible kmers = BASE_L^k, but this actually cancels with our integral
            //over the regions of possible bases, leaving only number unique kmers. 
            //The commented out part is technically "necessary." It just get cancelled with the commented out part that would get added for each base
            //prior probability of motif = P(pwm is of length k) * P(we select the best motif we have |pwm is of length k)*P(We select this PWM| we select the best motif we have and motif is length k)
            let mut prior = -((MAX_BASE+1-MIN_BASE) as f64).ln()+((seq.number_unique_kmers(self.len()) as f64).ln());//-(self.len() as f64)*(BASE_L as f64).ln(); 

            //The probability is P(this best motif)*f(this motif| this best motif). So we have to add a BASE_L.ln() for each base
            //prior += self.pwm.iter().map(|a| BASE_PRIOR.ln_pdf(a)).sum::<f64>();//+(BASE_L as f64).ln();

            //prior -= ((self.len() as f64)*(BASE_L as f64).ln());

            //For each base position, the probability that each of the three off bases are off as they is 1/number possible off values
            //There are three off bases per position, and self.len() positions. Hence, this form
            //prior += BASE_RESOLUTION.ln() * (((BASE_L-1)*self.len()) as f64);

            prior += self.pwm.iter().map(|a| a.prior_per_base()).sum::<f64>();

            prior

        } else {-f64::INFINITY}
    }


    pub fn height_prior(&self, height_dist: &(impl ::rand::distributions::Distribution<f64> + Continuous<f64,f64>)) -> f64 {

        let mut prior = if self.peak_height > 0.0 { PROB_POS_PEAK.ln() } else { (1.0-PROB_POS_PEAK).ln() };
        prior += height_dist.ln_pdf(self.peak_height.abs());
        prior
    }




    //BINDING FUNCTIONS

    //SAFETY: If kmer is not the same length as the pwm, this will produce undefined behavior
    unsafe fn prop_binding(&self, kmer: &[Bp]) -> (f64, bool) { 


        //let kmer: Vec<usize> = kmer_slice.to_vec();

        let mut bind_forward: f64 = 0.0;
        let mut bind_reverse: f64 = 0.0;

        let l = self.len();
        //

        for i in 0..l { 
            let bf = *kmer.get_unchecked(i);
            bind_forward += self.pwm[i][bf];
            let br = *kmer.get_unchecked(l-1-i);
            let brc = br.complement();
            //bind_reverse *= self.pwm[i].rel_bind(BASE_L-1-*kmer.get_unchecked(self.len()-1-i));
            bind_reverse += self.pwm[i][brc];
        }

        let reverse: bool = bind_reverse > bind_forward;

        let bind: f64 = if reverse {bind_reverse} else {bind_forward};

        return (bind, reverse);
    }



    unsafe fn cut_prop_binding(&self, kmer: &[Bp], max_compensation: f64) -> Option<f64> { 

        let cutoff = max_compensation;
        let l = self.len();


        let mut bind_forward: f64 = 0.0;
        let mut bind_reverse: f64 = 0.0;

        let mut try_forward = true;
        let mut try_reverse = true;


        

        let mut for_bind: f64 = 0.0;
        let mut rev_bind: f64 = 0.0;

        for (i, base) in self.pwm.iter().enumerate() { 
            let base = self.pwm.get_unchecked(i);
            let bf = *kmer.get_unchecked(i);
            let brc = kmer.get_unchecked(l-1-i).complement();
            for_bind += base[bf];
            rev_bind += base[brc];
        }
        
        let bind = for_bind.max(rev_bind);
        if bind <= cutoff {None} else {Some(bind)}

    }

    fn calc_6mer_prefix_binding(&self, kmer: u64) -> f64 {

        let mut kmer_track = kmer;

        let mut forward_score: f64 = 0.0;

        for i in 0..6 {

            //SAFETY: the result of this bit manipulation must always be at most 3
            let for_bp = unsafe{ Bp::usize_to_bp((kmer_track & 3) as usize)};
            kmer_track = kmer_track >> 2;
            forward_score += self.pwm[i][for_bp];
        }

        forward_score

    }

    fn return_bind_score(&self, seq: &Sequence) -> Vec<f64> {

        let coded_sequence = seq.seq_blocks();
        let block_lens = seq.block_lens(); //bp space
        let block_starts = seq.block_u8_starts(); //stored index space




        let mut ind: usize;

        let mut store: [Bp; BP_PER_U8];


            let mut bind_scores: Vec<f64> = vec![-f64::INFINITY; BP_PER_U8*coded_sequence.len()];
            let mut rev_comp: Vec<bool> = vec![false; BP_PER_U8*coded_sequence.len()];

            let mut uncoded_seq: Vec<Bp> = vec![Bp::A; seq.max_len()];

            {
                let uncoded_seq = uncoded_seq.as_mut_slice();
                for i in 0..(block_starts.len()) {


                    for jd in 0..(block_lens[i]/BP_PER_U8) {

                        store = Sequence::code_to_bases(coded_sequence[block_starts[i]+jd]);
                        for k in 0..BP_PER_U8 {
                            uncoded_seq[BP_PER_U8*jd+k] = store[k];
                        }

                    }


                    for j in 0..=((block_lens[i])-self.len()) {

                        ind = BP_PER_U8*block_starts[i]+(j as usize);


                        //SAFETY: notice how we defined j, and how it guarentees that get_unchecked is fine
                        let binding_borrow = unsafe { uncoded_seq.get_unchecked(j..(j+self.len())) };
                        (bind_scores[ind], _) = unsafe {self.prop_binding(binding_borrow) };

                    }

                }
            }


            bind_scores

    }


    fn return_best_bind_inds(&self, seq: &Sequence) -> Vec<(usize, usize)> {

        let coded_sequence = seq.seq_blocks();
        let block_lens = seq.block_lens(); //bp space
        let block_starts = seq.block_u8_starts(); //stored index space

        let best_motif = self.best_motif();

        let mut ind: usize;

        let mut store: [Bp; BP_PER_U8];

        let mut uncoded_seq: Vec<Bp> = vec![Bp::A; seq.max_len()];

        let mut blocks_and_bps: Vec<(usize, usize)> = vec![];

        {
            let uncoded_seq = uncoded_seq.as_mut_slice();
            for i in 0..(block_starts.len()) {


                for jd in 0..(block_lens[i]/BP_PER_U8) {

                    store = Sequence::code_to_bases(coded_sequence[block_starts[i]+jd]);
                    for k in 0..BP_PER_U8 {
                        uncoded_seq[BP_PER_U8*jd+k] = store[k];
                    }

                }


                for j in 0..=((block_lens[i])-self.len()) {

                    ind = BP_PER_U8*block_starts[i]+(j as usize);


                    //SAFETY: notice how we defined j, and how it guarentees that get_unchecked is fine
                    let binding_borrow = unsafe { uncoded_seq.get_unchecked(j..(j+self.len())) };

                    if binding_borrow.into_iter().zip(best_motif.iter()).all(|(&a, &b)| a == b) { blocks_and_bps.push((i, j)); }

                }

            }
        }


        blocks_and_bps


    }
    

    fn has_multiple_binding_sites(&self, seq: &Sequence, thresh: f64) -> bool {

        let coded_sequence = seq.seq_blocks();
        let block_lens = seq.block_lens(); //bp space
        let block_starts = seq.block_u8_starts(); //stored index space

        let best_motif = self.best_motif();

        let mut ind: usize;

        let mut store: [Bp; BP_PER_U8];

        let mut uncoded_seq: Vec<Bp> = vec![Bp::A; seq.max_len()];

        let mut blocks_and_bps: Vec<(usize, usize)> = vec![];

        let mut found_one = false;
        let mut found_two = false;

        {
            let uncoded_seq = uncoded_seq.as_mut_slice();
            for i in 0..(block_starts.len()) {


                for jd in 0..(block_lens[i]/BP_PER_U8) {

                    store = Sequence::code_to_bases(coded_sequence[block_starts[i]+jd]);
                    for k in 0..BP_PER_U8 {
                        uncoded_seq[BP_PER_U8*jd+k] = store[k];
                    }

                }


                for j in 0..=((block_lens[i])-self.len()) {

                    ind = BP_PER_U8*block_starts[i]+(j as usize);


                    //SAFETY: notice how we defined j, and how it guarentees that get_unchecked is fine
                    let binding_borrow = unsafe { uncoded_seq.get_unchecked(j..(j+self.len())) };

                    let (bind_score, _) = unsafe {self.prop_binding(binding_borrow) };

                    let bound = bind_score > thresh;
                    //If we previously found a binding site AND now have another one, we found multiple binding sites
                    if found_one && bound { return true; } 
                    found_one = found_one || bound;

                }

            }
        }

        false

    }


    fn return_bind_score_alt(&self, seq: &Sequence) -> Vec<(f64,bool)> {

        let coded_sequence = seq.seq_blocks();
        let block_lens = seq.block_lens(); //bp space
        let block_starts = seq.block_u8_starts(); //stored index space


        let mut pre_bind_scores: Vec<(f64, f64)>= vec![(0.0, 0.0); BP_PER_U8*coded_sequence.len()];
        //let mut bind_scores: Vec<f64> = vec![-f64::INFINITY; BP_PER_U8*coded_sequence.len()];
        //let mut rev_comp: Vec<bool> = vec![false; BP_PER_U8*coded_sequence.len()];

        let mut uncoded_seq: Vec<Bp> = vec![Bp::A; seq.max_len()];


        let mut ind: usize;

        let mut store: [Bp; BP_PER_U8];


        for (b, base) in self.pwm.iter().enumerate() {
            
            let uncoded_seq = uncoded_seq.as_mut_slice();
            for i in 0..(block_starts.len()) {


                for jd in 0..(block_lens[i]/BP_PER_U8) {

                    store = Sequence::code_to_bases(coded_sequence[block_starts[i]+jd]);
                    for k in 0..BP_PER_U8 {
                        uncoded_seq[BP_PER_U8*jd+k] = store[k];
                    }

                }


                for j in 0..=((block_lens[i])-self.len()) {

                    ind = BP_PER_U8*block_starts[i]+(j as usize);

                    unsafe{
                    pre_bind_scores[ind].0 += base[*uncoded_seq.get_unchecked(j+b)];
                    pre_bind_scores[ind].1 += base[uncoded_seq.get_unchecked(j+self.len()-1-b).complement()];
                    //SAFETY: notice how we defined j, and how it guarentees that get_unchecked is fine
                    //let binding_borrow = unsafe { uncoded_seq.get_unchecked(j..(j+self.len())) };
                    //(bind_scores[ind], rev_comp[ind]) = unsafe {self.prop_binding(binding_borrow) };

                    }
                }

            }
        }

        pre_bind_scores.into_iter().map(|(a,b)| if a >= b { (a, false) } else {(b,true)}).collect()

    }

    /*fn return_any_null_binds_by_hamming(&self, seq: &NullSequence, min_height:f64, distribution_cutoff: f64) -> Vec<f64> {

        if seq.seq_blocks().len() == 0 { return Vec::new();}

        //This 0.01 is just a way to omit more if there's a "technically 0 but float precision is
        //mean" score
        let cutoff = -min_height-distribution_cutoff+0.01;

        let standin_height = self.peak_height+distribution_cutoff;
        
        let mut check_cutoff = cutoff;

        const CHECK_CUTOFF_INDEX: usize = CAPACITY_FOR_NULL-1;

        let len = self.len();

        
        let mut forward_checks: Vec<(u64, f64)> = Vec::with_capacity(CAPACITY_FOR_NULL);

        for sixmer in 0_u64..=0b11_11_11_11_11_11 {

            let forward_score = self.calc_6mer_prefix_binding(sixmer);
            if forward_score >= cutoff { forward_checks.push((sixmer, forward_score)); }
        }


        forward_checks.sort_unstable_by(|g,h| h.1.partial_cmp(&g.1).unwrap());


        if forward_checks.len() > CAPACITY_FOR_NULL {
        check_cutoff = forward_checks[CHECK_CUTOFF_INDEX].1 + ((len-6) as f64)*SCORE_THRESH;
        let forward_partition = forward_checks.partition_point(|x| x.1 >= check_cutoff);
        if forward_partition < forward_checks.len() {
            _ = forward_checks.drain(forward_partition..).collect::<Vec<_>>();
        }
        }

        forward_checks = forward_checks.into_iter().map(|(sixmer, score)| {
            (0_u64..=0b11_11).filter_map(move |twomer| {
                let eightmer = (twomer << 12) + sixmer;
                if seq.kmer_count(eightmer, 8).is_none() {return None;};
                //SAFETY: two bits can only add up to 3 at most
                let b1 = unsafe{ Bp::usize_to_bp((twomer & 0b_00_11) as usize)};
                let b2 = unsafe{ Bp::usize_to_bp(((twomer & 0b_11_00) >> 2) as usize)};
                //SAFETY: MIN_BASE is set to 8, so elements 6 and 7 must exist
                let new_score = unsafe {score + self.pwm.get_unchecked(6)[b1]+self.pwm.get_unchecked(7)[b2]};
                if new_score >= cutoff { Some((eightmer, new_score))} else {None}
            })
        }).flatten().collect();
        
        let mut accounted_length: usize = 8;

        while accounted_length < len {


            if forward_checks.len() == 0 {
                return Vec::new();
            }

            check_cutoff = cutoff;
            
            forward_checks.sort_unstable_by(|g,h| h.1.partial_cmp(&g.1).unwrap());

            if forward_checks.len() > CAPACITY_FOR_NULL {
                check_cutoff = forward_checks[CHECK_CUTOFF_INDEX].1 + ((len-accounted_length) as f64)*SCORE_THRESH;

            let forward_partition = forward_checks.partition_point(|x| x.1 >= check_cutoff);
            if forward_partition < forward_checks.len() {
                _ = forward_checks.drain(forward_partition..).collect::<Vec<_>>();
            }
            }
            
            forward_checks = forward_checks.into_iter().map(|(kmer, score)| {
                BP_ARRAY.iter().filter_map(move |&base| {
                    
                    let kplus1mer = ((base as usize as u64) << (2*accounted_length)) + kmer;
                    if seq.kmer_count(kplus1mer, accounted_length+1).is_none() {return None;};
                    //SAFETY: two bits can only add up to 3 at most
                    //SAFETY: MIN_BASE is set to 8, so elements 6 and 7 must exist

                    let new_score = unsafe {score + self.pwm.get_unchecked(accounted_length)[base]};
                    

                    if new_score >= cutoff { Some((kplus1mer, new_score))} else {None}
                })
            }).flatten().collect();

            accounted_length+=1;
        }

        if forward_checks.len() == 0 { return Vec::new();}

        check_cutoff = cutoff;
        
        forward_checks.sort_unstable_by(|g,h| h.1.partial_cmp(&g.1).unwrap());

        if forward_checks.len() > CAPACITY_FOR_NULL {
            let potential_cutoff = forward_checks[CHECK_CUTOFF_INDEX].1;
            check_cutoff = cutoff.max(potential_cutoff);
        }


        let forward_partition = forward_checks.partition_point(|x| x.1 >= check_cutoff);
        if forward_partition < forward_checks.len() {
            _ = forward_checks.drain(forward_partition..).collect::<Vec<_>>();
        }

        
        let mut final_binds: Vec<f64> = Vec::with_capacity(CAPACITY_FOR_NULL);
        let mut accounted_kmers: HashSet<u64> = HashSet::with_capacity(CAPACITY_FOR_NULL);


        while final_binds.len() < CAPACITY_FOR_NULL {

            match forward_checks.get(0) {

                None => break,
                Some(t) => {
                    //Notice: I don't seem to care if accounted_kmers contains the reverse complement
                    //That's because it _can't_: we only store any particular kmer in one direction
                    //And we check for the reverse complement through the reverse complement MATRIX
                    //If we get the SAME kmer, it happens because both directions happened to have acceptable scores
                    //Which corresponds to both forward and reverse binding happening to seem fine
                    //But we already have the score which is actually the better one, and thus we ignore the other
                    if accounted_kmers.contains(&t.0) {
                        _ = forward_checks.drain(0..1).collect::<Vec<_>>();
                        continue;
                    } else {
                    
                        accounted_kmers.insert(crate::sequence::reverse_complement_u64_kmer(t.0, len));
                    }
                    let times_to_add = seq.kmer_count(t.0, len).expect("We only get here because this kmer exists");
                    for _ in 0..times_to_add { final_binds.push(t.1+standin_height); }
                    _ = forward_checks.drain(0..1).collect::<Vec<_>>();
                },
            };

        }
            

        //This can technically happen if we add more than one element at the end
        if final_binds.len() > CAPACITY_FOR_NULL { 
            _ = final_binds.drain(CAPACITY_FOR_NULL..).collect::<Vec<_>>();
        }



        final_binds


    } */

    fn return_any_null_binds_by_hamming(&self, seq: &NullSequence, min_height:f64, distribution_cutoff: f64) -> Vec<f64> {

        if seq.seq_blocks().len() == 0 { return Vec::new();}

        let cutoff = -min_height-distribution_cutoff+0.01;

        let mut checks: Vec<f64> = Vec::with_capacity(CAPACITY_FOR_NULL);

        for &kmer_u64 in seq.kmer_list_borrow(self.len()).iter() {

            let mut kmer_track = kmer_u64;
            let mut reverse_kmer_track = crate::sequence::reverse_complement_u64_kmer(kmer_u64, self.len());

            let mut forward_score: f64 = 0.0;
            let mut reverse_score: f64 = 0.0;

            for i in 0..self.len() {

            //SAFETY: the result of this bit manipulation must always be at most 3
            
                if forward_score >= cutoff {
                    let for_bp = unsafe{ Bp::usize_to_bp((kmer_track & 3) as usize)};
                    forward_score += unsafe{ self.pwm.get_unchecked(i)[for_bp]};
                    kmer_track = kmer_track >> 2;
                }
                if reverse_score >= cutoff {
                    let rev_bp = unsafe{ Bp::usize_to_bp((reverse_kmer_track & 3) as usize)};
                    reverse_score += unsafe{self.pwm.get_unchecked(i)[rev_bp]};
                    reverse_kmer_track >> 2;
                }
            }
            
            if forward_score >= cutoff || reverse_score >= cutoff {
                let pre_count = seq.kmer_count(kmer_u64, self.len()).expect("we selected kmer_u64 because it exists");
                let rev_comp = crate::sequence::reverse_complement_u64_kmer(kmer_u64, self.len());
                //If the reverse complement is the same as the motif, then we double counted the
                //motif, and pre_count is strictly even AND we only want to count half of it. If
                //the reverse complement is not the same, then the count could be even or odd. If
                //it's odd, we want to add a count for the odd number, but only to one. Thankfully,
                //the integers have a strict order
                let iter = if kmer_u64 == rev_comp {pre_count} else{pre_count/2 + (pre_count & 1) & ((kmer_u64 >= rev_comp) as usize)};
                #[cfg(test)]
                println!("test null {} {} {} {} {:?}", kmer_u64, rev_comp, iter, pre_count, seq.kmer_count(rev_comp, self.len()));
                for _ in 0..pre_count {checks.push(forward_score.max(reverse_score)+self.peak_height()+distribution_cutoff)};
                //for _ in 0..iter {checks.push(forward_score.max(reverse_score)+self.peak_height()+distribution_cutoff)};
            }
       
            if checks.len() >= CAPACITY_FOR_NULL {
                //_ = checks.drain(CAPACITY_FOR_NULL..).collect::<Vec<_>>();
                //println!("failing");
                return vec![f64::INFINITY, f64::INFINITY];
            }
        }

        checks.sort_unstable_by(|g,h| h.partial_cmp(&g).unwrap());

        checks
    }
    fn return_any_null_binds_by_hamming_no_limit(&self, seq: &NullSequence, min_height: f64, distribution_cutoff: f64) -> Vec<f64> {

        if seq.seq_blocks().len() == 0 { return Vec::new();}
        
        let cutoff = -min_height-distribution_cutoff;

        let standin_height = self.peak_height+distribution_cutoff;
        
        let mut check_cutoff = cutoff;


        let len = self.len();

        
        let mut forward_checks: Vec<(u64, f64)> = Vec::with_capacity(CAPACITY_FOR_NULL);

        for sixmer in 0_u64..=0b11_11_11_11_11_11 {

            let forward_score = self.calc_6mer_prefix_binding(sixmer);
            if forward_score >= cutoff { forward_checks.push((sixmer, forward_score)); }
        }


        forward_checks.sort_unstable_by(|g,h| h.1.partial_cmp(&g.1).unwrap());


        forward_checks = forward_checks.into_iter().map(|(sixmer, score)| {
            (0_u64..=0b11_11).filter_map(move |twomer| {
                let eightmer = (twomer << 12) + sixmer;
                if seq.kmer_count(eightmer, 8).is_none() {return None;};
                //SAFETY: two bits can only add up to 3 at most
                let b1 = unsafe{ Bp::usize_to_bp((twomer & 0b_00_11) as usize)};
                let b2 = unsafe{ Bp::usize_to_bp(((twomer & 0b_11_00) >> 2) as usize)};
                //SAFETY: MIN_BASE is set to 8, so elements 6 and 7 must exist
                let new_score = unsafe {score + self.pwm.get_unchecked(6)[b1]+self.pwm.get_unchecked(7)[b2]};
                if new_score >= cutoff { Some((eightmer, new_score))} else {None}
            })
        }).flatten().collect();
        
        let mut accounted_length: usize = 8;

        while accounted_length < len {


            if forward_checks.len() == 0 {
                return Vec::new();
            }

            check_cutoff = cutoff;
            
            forward_checks.sort_unstable_by(|g,h| h.1.partial_cmp(&g.1).unwrap());

            
            forward_checks = forward_checks.into_iter().map(|(kmer, score)| {
                BP_ARRAY.iter().filter_map(move |&base| {
                    
                    let kplus1mer = ((base as usize as u64) << (2*accounted_length)) + kmer;
                    if seq.kmer_count(kplus1mer, accounted_length+1).is_none() {return None;};
                    //SAFETY: two bits can only add up to 3 at most
                    //SAFETY: MIN_BASE is set to 8, so elements 6 and 7 must exist

                    let new_score = unsafe {score + self.pwm.get_unchecked(accounted_length)[base]};
                    

                    if new_score >= cutoff { Some((kplus1mer, new_score))} else {None}
                })
            }).flatten().collect();

            accounted_length+=1;
        }

        if forward_checks.len() == 0 { return Vec::new();}

        
        forward_checks.sort_unstable_by(|g,h| h.1.partial_cmp(&g.1).unwrap());


        
        let mut final_binds: Vec<f64> = Vec::with_capacity(CAPACITY_FOR_NULL);
        let mut accounted_kmers: HashSet<u64> = HashSet::with_capacity(CAPACITY_FOR_NULL);


        /*while forward_checks.len() > 0 {

            match forward_checks.get(0) {

                None => break,
                Some(t) => {
                    //Notice: I don't seem to care if accounted_kmers contains the reverse complement
                    //That's because it _can't_: we only store any particular kmer in one direction
                    //And we check for the reverse complement through the reverse complement MATRIX
                    //If we get the SAME kmer, it happens because both directions happened to have acceptable scores
                    //Which corresponds to both forward and reverse binding happening to seem fine
                    //But we already have the score which is actually the better one, and thus we ignore the other
                    if accounted_kmers.contains(&t.0) {
                        _ = forward_checks.drain(0..1).collect::<Vec<_>>();
                        continue;
                    } else {
                    
                        accounted_kmers.insert(crate::sequence::reverse_complement_u64_kmer(t.0, len));
                    }
                    let times_to_add = seq.kmer_count(t.0, len).expect("We only get here because this kmer exists");
                    for _ in 0..times_to_add { final_binds.push(t.1+standin_height); }
                    _ = forward_checks.drain(0..1).collect::<Vec<_>>();
                },
            };

        }*/
          

        _ = forward_checks.into_iter().map(|t| {
            if !accounted_kmers.contains(&t.0) {
                accounted_kmers.insert(crate::sequence::reverse_complement_u64_kmer(t.0, len));
                let times_to_add = seq.kmer_count(t.0, len).expect("We only get here because this kmer exists");
                for _ in 0..times_to_add { final_binds.push(t.1+standin_height); }
            }
        }).collect::<Vec<_>>();

        final_binds


    }
    

    /// This generates the occupancy trace of `self` onto the data from `data_ref`.
    /// It does this by calculating adding the energy penalties of each position, 
    /// and if the energy penalty `> -data_ref.min_height()`, it draws a binding 
    /// kernel of height `self.peak_height()+` the energy penalty.
    /// There is a particular quirk when we draw the occupancy trace. 
    /// If the penalty to the draw is less than 2.0 BASE_RESOLUITON*SCORE_THRESH, 
    /// we revert to drawing the Gaussian kernel, on the assumption that the binding
    /// is not strong enough to generate a more distinct shape
    pub fn generate_waveform<'a>(&self, data_ref: &'a AllDataUse) -> Waveform<'a> {

        let data = data_ref.data();
        
        let unit_kernel = data_ref.unit_kernel_ref(self.kernel_width, self.kernel_variety);

        let unit_gaussian = data_ref.unit_kernel_ref(self.kernel_width, KernelVariety::Gaussian);

        let mut occupancy_trace: Waveform = data.derive_zero();

        let mut actual_kernel: Kernel;

        let starts = data.seq().block_u8_starts();

        let lens = data.seq().block_lens();

        let bind_score_floats = self.return_bind_score(data.seq());
        //let (bind_score_floats, _) = self.return_bind_score(data.seq());

        let thresh = -data_ref.min_height();

        for i in 0..starts.len() { //Iterating over each block
            for j in 0..(lens[i]-self.len()) {
                //if bind_score_floats[starts[i]*BP_PER_U8+j] > *THRESH.read().expect("no writes expected now") {i //}
                //SAFETY: 
                //  -block = i, which is always less than the number of sequence blocks
                //  -center = j, which is always less than the number of bps in the Sequence by at least the motif length if Sequence is correctly constructed
                //SAFETY: THRESH is never modified at this point

                let bind = unsafe{*bind_score_floats.get_unchecked(starts.get_unchecked(i)*BP_PER_U8+j)};
                if bind > thresh {
                    let actual_height = bind+self.peak_height;
                    actual_kernel = if (actual_height-self.peak_height) < (2.0*BASE_RESOLUTION*SCORE_THRESH) {unit_kernel*(actual_height) } else {unit_gaussian*(actual_height)};

                    unsafe {occupancy_trace.place_peak(&actual_kernel, i, j+(self.len()-1)/2);} 

                }
            }
        }

        //println!("num peaks {}", count);


        occupancy_trace

    }

    /// This generates the occupancy trace of `self` onto the data from `data_ref`.
    /// It does this by calculating adding the energy penalties of each position, 
    /// and if the energy penalty `> -data_ref.min_height()`, it draws a binding 
    /// kernel of height `self.peak_height()+` the energy penalty.
    pub fn generate_waveform_alt<'a>(&self, data_ref: &'a AllDataUse) -> Waveform<'a> {

        let data = data_ref.data();
        
        let unit_kernel = data_ref.unit_kernel_ref(self.kernel_width, self.kernel_variety);

        let mut occupancy_trace: Waveform = data.derive_zero();

        let mut actual_kernel: Kernel;

        let starts = data.seq().block_u8_starts();

        let lens = data.seq().block_lens();

        let bind_score_vec = self.return_bind_score_alt(data.seq());

        let thresh = -self.peak_height;


        for i in 0..starts.len() { //Iterating over each block
            for j in 0..(lens[i]-self.len()) {
                //if bind_score_floats[starts[i]*BP_PER_U8+j] > *THRESH.read().expect("no writes expected now") {i //}
                //SAFETY:
                //  -block = i, which is always less than the number of sequence blocks
                //  -center = j, which is always less than the number of bps in the Sequence by at least the motif length if Sequence is correctly constructed
                //SAFETY: THRESH is never modified at this point

                let bind = unsafe{bind_score_vec.get_unchecked(starts.get_unchecked(i)*BP_PER_U8+j).0};
                if bind > thresh {
                    actual_kernel = unit_kernel*(bind+self.peak_height) ;

                    unsafe {occupancy_trace.place_peak(&actual_kernel, i, j+(self.len()-1)/2);}

                }
            }
        }


        //println!("num peaks {}", count);


        occupancy_trace

    }



    /// This generates distance between two motifs, based on their `Base`s. 
    /// In particular, it tries every possible overlapping offset of the motifs,
    /// along with reverse complementing, and calculates the sum of the squared
    /// Aitchison distances between them. If a Base does not have a corresponding 
    /// partner, it add its squared Aitchison metric to the total. Then, the minimum
    /// such distance is found. The function returns both the distance and whether
    /// the best such distance needed reverse complementing.
    /// 
    /// # Example
    /// ```
    /// use motif_finder::base::{Motif, Bp, Base, TruncatedLogNormal, LOG_HEIGHT_MEAN, LOG_HEIGHT_SD, MAX_HEIGHT};
    /// let mut rng = rand::thread_rng();
    /// let height_dist = TruncatedLogNormal::new(LOG_HEIGHT_MEAN, LOG_HEIGHT_SD, 2.0, MAX_HEIGHT).unwrap();
    /// let m = Motif::from_motif(vec![Bp::A, Bp::C, Bp::G, Bp::G, Bp::G, Bp::G, Bp::T, Bp::T, Bp::T], &height_dist,&mut rng);
    /// let dself = m.distance_function(&m);
    /// assert!(dself.0.abs() < 1e-16);
    /// assert!(!dself.1);
    /// let drev = m.distance_function(&(Motif::raw_pwm(m.rev_complement(), m.peak_height(), m.kernel_width(), m.kernel_variety())));
    /// assert!(drev.0.abs() < 1e-16);
    /// assert!(drev.1);
    /// let mut new_b_vec: Vec<Base> = Vec::with_capacity(m.len()+2);
    /// new_b_vec.append(&mut m.rev_complement());
    /// new_b_vec.push(Base::rand_new(&mut rng));
    /// new_b_vec.push(Base::rand_new(&mut rng));
    /// let newmot = Motif::raw_pwm(new_b_vec, m.peak_height(),m.kernel_width(), m.kernel_variety());
    /// println!("new {:?} old {:?}", newmot, m);
    /// ```
    pub fn distance_function(&self, other_mot: &Motif) -> (f64, bool) {
        let rev = other_mot.rev_complement();

        //println!("other_mot {:?}\n PWM {:?}", other_mot, &other_mot.pwm);
        let best_forward = (self.little_distance(&other_mot.pwm), false);
        let best_reverse = (self.little_distance(&rev), true);

        let best = if best_reverse.0 < best_forward.0 { best_reverse } else {best_forward};

        best

    }

    fn little_distance(&self, other_mot: &Vec<Base>) -> f64 {

        let _total_len = (self.len()).max(other_mot.len());
 

        let (pwm_short, pwm_long) = if self.len() <= other_mot.len() {(&self.pwm, other_mot)} else {(other_mot, &self.pwm)};

        let short_len = pwm_short.len();
        let long_len = pwm_long.len();


        let off_center = ((long_len-1)/2) - ((short_len-1)/2);


        //println!("S {:?} O {:?}", self, other_mot);

        let mut distance: f64 = 0.0;

        if off_center > 0 { 
            for i in 0..off_center {
                //TODO: change this to get_unchecked if there's a performance benefit
                distance += pwm_long[i].dist_sq(None);
            }
        }
        
        if (short_len+off_center) < long_len {
            for i in (short_len+off_center)..long_len {
                distance += pwm_long[i].dist_sq(None);
            }
        }

        for ind in 0..short_len {

            let b1 = &pwm_short[ind]; 
            let b2 = &pwm_long[off_center+ind];
          

            distance += b1.dist_sq(Some(b2));
            
        }


        //This isn't a CODE problem, it's an f64 precision one
        if distance > 0.0 { distance.sqrt() } else {0.0}
    }

    /// This calculates motif distances by the RMSD of their binding scores, 
    /// without considering differences from heights
    pub fn dist_by_binding(&self, other_mot: &Motif, seq: &Sequence) -> f64 {

        let binds_0 = self.return_bind_score(seq);
        let binds_1 = other_mot.return_bind_score(seq);

        binds_0.into_iter().zip(binds_1.into_iter()).map(|(a,b)| (a.exp2()-b.exp2()).powi(2)).sum::<f64>().sqrt()

    }

    fn for_meme(&self, background_dist: Option<[f64; BASE_L]>) -> Result<String, InvalidBase> {

        const MEME_DIGITS: usize = 6;

        let mut meme_string = format!("letter-probability matrix: alength= {} w= {}\n", BPS.len(), self.len());

        for base in self.pwm.iter() {
            let probs = base.to_pwm(background_dist)?;
            for prob in probs.iter() {
                meme_string = format!("{} {:.MEME_DIGITS$} ", meme_string, prob);
            }
            meme_string = format!("{}\n", meme_string);
        }
        meme_string = format!("{}\n", meme_string);

        Ok(meme_string)
    }


}

impl fmt::Display for Motif { 
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const DIGITS: usize = 10;

        //Writing the peak height (unitless) and the peak width (as defined by # base pairs from one end to another)
        write!(f, "Peak height: {:.DIGITS$}. \n", self.peak_height)?;

        //I want people to be free to define their own bases, if they like
        //But they will have to change it in the code for formatting: it's not my problem
        //The two extra spaces on either side let the output align in a pretty way
        for b in BPS { write!(f, "  {:<DIGITS$}  ", b)?; }
        write!(f, "\n")?;

        for i in 0..self.pwm.len() {

            let base = self.pwm[i].scores();
            let begin: bool = i == 0;
            let end: bool = i == (self.pwm.len()-1);
            let j = (begin, end);

            match j {
                (true, true) => panic!("Something pathological is going on!"),
                (true, false) => write!(f, "[{base:.DIGITS$?},\n", base = base, DIGITS = DIGITS),
                (false, false) => write!(f, " {base:.DIGITS$?},\n", base = base, DIGITS = DIGITS),
                (false, true) => write!(f, " {base:.DIGITS$?}]\n", base = base, DIGITS = DIGITS),
            }?;
        }
        Ok(())
    }
}

impl fmt::Debug for Motif { 
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const DIGITS: usize = 10;

        //Writing the peak height (unitless) and the peak width (as defined by # base pairs from one end to another)
        write!(f, "Peak height: {:.DIGITS$}. Kernel type {:?}. Kernel width type {:?}.\n", self.peak_height, self.kernel_variety, self.kernel_width)?;
        //I want people to be free to define their own bases, if they like
        //But they will have to change it in the code for formatting: it's not my problem
        //The two extra spaces on either side let the output align in a pretty way
        for b in BPS { write!(f, "  {:<DIGITS$}  ", b)?; }
        write!(f, "\n")?;

        for i in 0..self.pwm.len() {

            let base = self.pwm[i].scores();
            let begin: bool = i == 0;
            let end: bool = i == (self.pwm.len()-1);
            let j = (begin, end);

            match j {
                (true, true) => panic!("Something pathological is going on!"),
                (true, false) => write!(f, "[{base:.DIGITS$?},\n", base = base, DIGITS = DIGITS),
                (false, false) => write!(f, " {base:.DIGITS$?},\n", base = base, DIGITS = DIGITS),
                (false, true) => write!(f, " {base:.DIGITS$?}]\n", base = base, DIGITS = DIGITS),
            }?;
        }
        write!(f, "\n")?;
        Ok(())
    }
}

/// This stores a single state of the inference
/// Each Motif stored also comes with a vector of its matches to NullSequence
#[derive(Clone)]
pub struct MotifSet<'a> {

    set: Vec<(Motif, Vec<f64>)>, 
    signal: Waveform<'a>,
    ln_post: Option<f64>,
    data_ref: &'a AllDataUse<'a>, 
}

impl<'a> MotifSet<'a> {

    /// This generates a single Motif MotifSet, along with its ancillaries
    pub fn manual_set(data_ref: &'a AllDataUse<'a>, mot: Motif) -> Self {

        let signal = mot.generate_waveform(data_ref);

        let nulls = mot.return_any_null_binds_by_hamming(data_ref.null_seq(), data_ref.min_height(),data_ref.offset()*2.0);

        let mut mot_set = MotifSet {
            set: vec![(mot, nulls)],
            signal: signal, 
            ln_post: None,
            data_ref: data_ref
        };

        _ = mot_set.ln_posterior();

        mot_set
    }
    
    pub(crate) fn manual_set_null_free(data_ref: &'a AllDataUse<'a>, mot: Motif) -> Self {

        let signal = mot.generate_waveform(data_ref);

        let mut mot_set = MotifSet {
            set: vec![(mot, Vec::new())],
            signal: signal, 
            ln_post: None,
            data_ref: data_ref
        };

        _ = mot_set.ln_posterior();

        mot_set
    }

    /// This randomly generates a single Motif MotifSet with a valid motif for data_ref
    pub fn rand_with_one<R: Rng+?Sized>(data_ref: &'a AllDataUse<'a>, rng: &mut R) -> Self {

        let mut valid: bool;

        let mot_set = loop { //We occasionally seem to randomly generate sets with such high occupancy everywhere as to be ridiculous

            let set = vec![Motif::rand_mot(data_ref.data().seq(), data_ref.height_dist(), rng)];

            let signal = set[0].generate_waveform(data_ref);

            let set_with_nulls: Vec<(Motif, Vec<f64>)> = set.into_iter().map(|a| {
                let null = a.return_any_null_binds_by_hamming(data_ref.null_seq(),data_ref.min_height(), data_ref.offset()*2.0);
                (a,null)
            }).collect();

            let mut mot_set_try = MotifSet{ set: set_with_nulls, signal: signal, ln_post: None, data_ref: data_ref};

            let like = mot_set_try.ln_posterior();

            valid = like.is_finite();

            if valid {
                break mot_set_try;
            }
        };

        mot_set
    }
    
    /// This randomly generates a single `n` Motif MotifSet with valid motifs for data_ref
    pub fn rand_with_n_motifs<R: Rng+?Sized>(n: usize, data_ref: &'a AllDataUse<'a>, rng: &mut R) -> Self {

        let mut n = n;

        let mut valid: bool;

        let mot_set = loop { //We occasionally seem to randomly generate sets with such high occupancy everywhere as to be ridiculous

            let set: Vec<Motif> = (0..n).map(|i| {
                let mut mot = Motif::rand_mot(data_ref.data().seq(), data_ref.height_dist(), rng);
                println!("rand {i} out of {n}");
                while !mot.has_multiple_binding_sites(data_ref.data().seq(), -data_ref.min_height()) {
                    mot = Motif::rand_mot(data_ref.data().seq(), data_ref.height_dist(), rng);
                }
                mot
            }).collect();

            let mut signal = data_ref.data().derive_zero();
            for mot in set.iter() {
                signal += &(mot.generate_waveform(data_ref));
            }

            let set_with_nulls: Vec<(Motif, Vec<f64>)> = set.into_iter().map(|a| {
                let null = a.return_any_null_binds_by_hamming(data_ref.null_seq(),data_ref.min_height(), data_ref.offset()*2.0);
                (a,null)
            }).collect();

            let mut mot_set_try = MotifSet{ set: set_with_nulls, signal: signal, ln_post: None, data_ref: data_ref};

            let like = mot_set_try.ln_posterior();

            valid = like.is_finite();

            if valid {
                break mot_set_try;
            }
            if n > 1 {n = n-1;}
        };

        mot_set
    }

    /// This randomly generates a single Motif MotifSet with
    /// with height peak_height and a random valid motif for data_ref
    pub fn rand_with_one_height<R: Rng+?Sized >(peak_height: f64, data_ref: &'a AllDataUse<'a>, rng: &mut R) -> Self {

        let set = vec![Motif::rand_mot_with_height(peak_height, data_ref.data().seq(), rng)];

        let signal = set[0].generate_waveform(data_ref);
            
        let set_with_nulls: Vec<(Motif, Vec<f64>)> = set.into_iter().map(|a| {
            let null = a.return_any_null_binds_by_hamming(data_ref.null_seq(), data_ref.min_height(), data_ref.offset()*2.0);
            (a,null)
        }).collect();
        
        let mut mot_set = MotifSet{ set: set_with_nulls, signal: signal, ln_post: None, data_ref: data_ref};


        let _ = mot_set.ln_posterior();

        mot_set
    }

    /// This randomly generates a single Motif MotifSet with
    /// with height peak_height and a random valid motif for data_ref
    /// of length peak_height.min(MAX_BASE)
    pub fn rand_with_one_height_and_motif_len<R: Rng + ?Sized>(peak_height: f64, motif_len: usize, data_ref: &'a AllDataUse<'a>, rng: &mut R) -> Self {


        let set = vec![Motif::rand_mot_with_height_and_motif_len(peak_height, motif_len, data_ref.data().seq(), rng)];

        let signal = set[0].generate_waveform(data_ref);
        
        let set_with_nulls: Vec<(Motif, Vec<f64>)> = set.into_iter().map(|a| {
            let null = a.return_any_null_binds_by_hamming(data_ref.null_seq(), data_ref.min_height(), data_ref.offset()*2.0);
            (a,null)
        }).collect();

        let mut mot_set = MotifSet{ set: set_with_nulls, signal: signal, ln_post: None, data_ref: data_ref};

        let _ = mot_set.ln_posterior();

        mot_set
    }
    
    

    /// This attempts to generate a MotifSet based on input from a meme file
    /// If a motif it reads is not valid for data_ref, it transforms it
    /// into a valid motif with the nearest possible Hamming distance. 
    /// # Errors
    /// If the meme file is not correctly formatted. This can be for a variety
    /// of reasons, from an empty matrix, an alphabet of the wrong length, 
    /// a column of the wrong length, an inability to parse floats from the meme
    /// file, or a negative value in the PWM.
    pub fn set_from_meme<R: Rng+?Sized>(meme_file_name: &str, data_ref: &'a AllDataUse<'a>, background_dist: Option<[f64; BASE_L]>, e_value_cutoff: f64, make_poss: HandleImpossibleMotif, discretize: bool, rng: &mut R) -> Result<Self, Box<dyn Error+Send+Sync>> {

        let meme_file_string = fs::read_to_string(meme_file_name)?;
        let meme_as_vec = meme_file_string.split("\n").collect::<Vec<_>>();

        //This should be infallible, because it never changes and I already validated it once. 
        //If THIS is where you're breaking, this whole function is invalid and needs to be fixed. 
        let re = Regex::new(r"letter-probability matrix: alength= (\d+) w= (\d+)")?;

        let start_matrix_lines: Vec<usize> = meme_as_vec.iter().enumerate().filter(|&(_,a)| re.is_match(a)).map(|(n, _)| n).collect();

        let num_motifs_possibly_possible = start_matrix_lines.len();
        if start_matrix_lines.len() == 0 { return Err(Box::new(MemeParseError::EmptyMatrix)); }

        let mut set: Vec<Motif> = Vec::new();

        for (mot_num, &line) in start_matrix_lines.iter().enumerate() {
            let captures = re.captures(meme_as_vec[line]).expect("We can only get here because we matched on regex");
            let alphabet_len: usize = captures[1].parse().expect("We can only get here because we matched on regex");

            if alphabet_len != BASE_L { return Err(Box::new(MemeParseError::InvalidAlphabet{ motif_num: mot_num }));}

            let motif_len: usize = captures[2].parse().unwrap();
            if (motif_len < MIN_BASE) || (motif_len > MAX_BASE) { return Err(Box::new(MemeParseError::InvalidMotifLength{ motif_num: mot_num, captured_length: motif_len })); }


            //let e_value: f64 = captures.get(3).map(|a| a.parse()).flatten().unwrap_or(0.0);

            //if (set.len() > 0) && (e_value > e_value_cutoff) {
            //    break;
            //}

            let mut base_vec: Vec<Result<Base, (InvalidBase, [f64;BASE_L])>> = Vec::with_capacity(MAX_BASE);

            for i in 1..(motif_len+1) {
                let mut scores: [ f64; BASE_L] = [0.0; BASE_L];
                let mut line_split = meme_as_vec[line+i].split_whitespace();
                for j in 0..BASE_L {
                    let Some(prop_str) = line_split.next() else { return Err(Box::new(MemeParseError::ColumnLengthFailure{line_num: line+i+1})); };
                    let Ok(prop)= prop_str.parse::<f64>() else { return Err(Box::new(MemeParseError::FloatParseFailure{line_num: line+i+1}));}; 
                    scores[j] = prop;
                }


                //NOTE: in my motif runs, this was true. For assessing the raw PWMs, this is false
                base_vec.push(Base::from_pwm(scores, background_dist, discretize).map_err(|a| (a, scores)));
            }

            let num_poss_motifs: usize = base_vec.iter().map(|a| match a {
                Ok(_) => 1,
                Err((InvalidBase::NoSingleBestBase(maxes), _)) => maxes.len(),
                _ => 0,
            }).product::<usize>();

            println!("num_poss_motifs {} {}", set.len(), num_poss_motifs);
            if num_poss_motifs >= 0b1_0000_0000_0000_0000{ //auto omit any motif with more than 2^16 possible "best" motifs, because OOMs suck
                continue;
            }

            if num_poss_motifs == 0 { return Err(Box::new(InvalidBase::NegativeBase)); }

            let mut motif_check: Vec<Vec<Base>> = vec![Vec::with_capacity(base_vec.len()); num_poss_motifs];

            let mut hold_chunk: usize = 1;

            for element in base_vec {

                match element {
                    Ok(base) => {for motif in motif_check.iter_mut() { motif.push(base.clone())}},
                    Err((InvalidBase::NoSingleBestBase(maxes), props)) => {

                        let prop_vec: Vec<Base> = maxes.iter().map(|a| {
                            let mut props_alt = props;
                            props_alt[*a as usize] += 0.00001;
                            Base::from_pwm(props_alt, background_dist, discretize).expect("We resolved the error")
                        }).collect();

                        let num_alts = prop_vec.len();

                        for i in 0..num_poss_motifs {
                            let which_base = prop_vec[(i/hold_chunk) % num_alts].clone();
                            motif_check[i].push(which_base);
                        }
                        
                        hold_chunk *= num_alts;
                    },
                    Err(elsing) => return Err(Box::new(elsing.0)),
           
                }
            }

            let grab_from: Vec<(Motif, usize)> = motif_check.iter().filter_map(|a| {

                let mot_check = a.iter().map(|a| a.best_base()).collect::<Vec<Bp>>();
                if data_ref.data().seq().kmer_in_seq(&mot_check) { 
                    let mot = Motif::raw_pwm(a.clone(), data_ref.min_height(), KernelWidth::Wide, KernelVariety::Gaussian);
                    let extra_binding = mot.return_any_null_binds_by_hamming(data_ref.null_seq(),data_ref.min_height(), data_ref.offset()*2.0).len();
                    if extra_binding >= (CAPACITY_FOR_NULL) { None } else {Some((mot, extra_binding))}
                } else {None}
            }).collect();

            if grab_from.len() > 0 {
                set.push(grab_from.iter().min_by(|a,b| a.1.cmp(&b.1)).expect("we already know this is non empty").0.clone());
                continue;
            }

            match make_poss {
               HandleImpossibleMotif::OmitFromSet => continue, 
               HandleImpossibleMotif::MakePossible => {
                   let mot_vec = motif_check.choose(rng).expect("know this is non empty").clone();
                   let mut motif = Motif::raw_pwm(mot_vec, data_ref.min_height(), KernelWidth::Wide, KernelVariety::Gaussian);
                   let Some(hamming) = motif.scramble_to_close_random_valid(data_ref.data().seq(), rng) else { unreachable!();};
                   set.push(motif);
                   warn!("{}", format!("Motif number {} from the MEME file does not exist in the parts of the sequence with peaks! Moving it to a valid motif within a Hamming distance of {}!", mot_num, hamming));
               },
               HandleImpossibleMotif::LeaveUnchanged => {
                   let mot_vec = motif_check.choose(rng).expect("know this is non empty").clone();
                   let mut motif = Motif::raw_pwm(mot_vec, data_ref.min_height(), KernelWidth::Wide, KernelVariety::Gaussian);
                   set.push(motif);
               },
            }

        }

        if set.len() == 0 { return Err(Box::new(MemeParseError::GivesEmptySet));}

        let mut signal = data_ref.data().derive_zero();

        for mot in &set {
            signal += &(mot.generate_waveform(data_ref));
        }

        let set_with_nulls: Vec<(Motif, Vec<f64>)> = set.into_iter().map(|a| (a.clone(), a.return_any_null_binds_by_hamming(data_ref.null_seq(),data_ref.min_height(), data_ref.offset()*2.0))).collect();

        let mut full_set = MotifSet {
            set: set_with_nulls,
            signal: signal, 
            ln_post: None, 
            data_ref: data_ref, 
        };

        let _ = full_set.ln_posterior();

        Ok(full_set)

    }
    
    /// This attempts to generate a MotifSet based on input from a meme file
    /// But instead of respecting proportionality, it makes all `Base`s as 
    /// selective as possible. f a motif it reads is not valid for data_ref, it
    /// transforms it into a valid motif with the nearest possible Hamming distance. 
    /// # Errors
    /// If the meme file is not correctly formatted. This can be for a variety
    /// of reasons, from an empty matrix, an alphabet of the wrong length, 
    /// a column of the wrong length, an inability to parse floats from the meme
    /// file, or a negative value in the PWM.
    pub fn set_from_meme_max_selective<R: Rng+?Sized>(meme_file_name: &str, data_ref: &'a AllDataUse<'a>, e_value_cutoff: f64, make_poss: bool, rng: &mut R) -> Result<Self, Box<dyn Error+Send+Sync>> {

        let meme_file_string = fs::read_to_string(meme_file_name)?;
        let meme_as_vec = meme_file_string.split("\n").collect::<Vec<_>>();

        //This should be infallible, because it never changes and I already validated it once. 
        //If THIS is where you're breaking, this whole function is invalid and needs to be fixed. 
        let re = Regex::new(r"letter-probability matrix: alength= (\d+) w= (\d+)")?;

        let start_matrix_lines: Vec<usize> = meme_as_vec.iter().enumerate().filter(|&(_,a)| re.is_match(a)).map(|(n, _)| n).collect();

        if start_matrix_lines.len() == 0 { return Err(Box::new(MemeParseError::EmptyMatrix)); }

        let mut set: Vec<Motif> = Vec::new();

        for (mot_num, &line) in start_matrix_lines.iter().enumerate() {
            let captures = re.captures(meme_as_vec[line]).expect("We can only get here because we matched on regex");
            let alphabet_len: usize = captures[1].parse().expect("We can only get here because we matched on regex");

            if alphabet_len != BASE_L { return Err(Box::new(MemeParseError::InvalidAlphabet{ motif_num: mot_num }));}

            let motif_len: usize = captures[2].parse().unwrap();
            if (motif_len < MIN_BASE) || (motif_len > MAX_BASE) { return Err(Box::new(MemeParseError::InvalidMotifLength{ motif_num: mot_num, captured_length: motif_len })); }


            //let e_value: f64 = captures.get(3).map(|a| a.parse()).flatten().unwrap_or(0.0);

            //if (set.len() > 0) && (e_value > e_value_cutoff) {
            //    break;
            //}

            let mut bp_vec: Vec<Bp> = Vec::with_capacity(MAX_BASE);

            for i in 1..(motif_len+1) {
                let mut scores: [ f64; BASE_L] = [0.0; BASE_L];
                let mut line_split = meme_as_vec[line+i].split_whitespace();
                for j in 0..BASE_L {
                    let Some(prop_str) = line_split.next() else { return Err(Box::new(MemeParseError::ColumnLengthFailure{line_num: line+i+1})); };
                    let Ok(prop)= prop_str.parse::<f64>() else { return Err(Box::new(MemeParseError::FloatParseFailure{line_num: line+i+1}));}; 
                    
                    scores[j] = prop;
                }
                //Safety: self.scores has a size of BASE_L, so this always produces a valid result
                let bp = unsafe{ std::mem::transmute::<usize, Bp>(Base::argmax(&scores))};

                bp_vec.push(bp);
            }

            let mut motif = Motif::from_motif_max_selective(bp_vec, data_ref.min_height(), KernelWidth::Wide, KernelVariety::Gaussian);

            if make_poss {
                let poss_hamming = motif.scramble_to_close_random_valid(data_ref.data().seq(), rng);

                match poss_hamming {
                    Some(hamming) => warn!("{}", format!("Motif number {} from the MEME file does not exist in the parts of the sequence with peaks! Moving it to a valid motif within a Hamming distance of {}!", mot_num, hamming)),
                    None => (),
                };

            }

            println!("init motif {mot_num} {:?}", motif);

            set.push(motif);    

        }


        let mut signal = data_ref.data().derive_zero();

        for mot in &set {
            signal += &(mot.generate_waveform(data_ref));
        }

        let set_with_nulls: Vec<(Motif, Vec<f64>)> = set.into_iter().map(|a| (a.clone(), a.return_any_null_binds_by_hamming(data_ref.null_seq(), data_ref.min_height(),data_ref.offset()*2.0))).collect();

        let mut full_set = MotifSet {
            set: set_with_nulls,
            signal: signal, 
            ln_post: None, 
            data_ref: data_ref, 
        };

        let _ = full_set.ln_posterior();

        Ok(full_set)

    }


    pub fn set_from_json<R: Rng+?Sized>(data_ref: &'a AllDataUse<'a>, json_file: &str, _rng: &mut R) -> Result<Self, Box<dyn Error+Send+Sync>> {

        let json_string: String = fs::read_to_string(json_file)?;

        let prior_state: StrippedMotifSet = serde_json::from_str(&json_string)?;

        Ok(prior_state.reactivate_set(data_ref))

        //self.push_set_def(always_recalculate, validate_motif, prior_state, rng);

    }
 
    pub fn set_from_bincode<R: Rng+?Sized>(data_ref: &'a AllDataUse<'a>, bincode_file: &str, _rng: &mut R) -> Result<Self, Box<dyn Error+Send+Sync>> {

        let mut bincode_file_handle = fs::File::open(bincode_file)?;

        let mut buffer: Vec<u8> = Vec::new();

        bincode_file_handle.read_to_end(&mut buffer)?;

        let (prior_state, _bytes_read): (StrippedMotifSet, usize) = match bincode::serde::decode_from_slice(&buffer, config::standard()) {
            Ok(a) => a,
            Err(_) => {
          
                let mut bincode_file_handle: ParDecompress<Mgzip> = ParDecompressBuilder::new().from_reader( fs::File::open(bincode_file)?);
                buffer.clear();
                bincode_file_handle.read_to_end(&mut buffer)?;
                bincode::serde::decode_from_slice(&buffer, config::standard())?
            },
        };
/*
        let (prior_state, _bytes_read): (StrippedMotifSet, usize) = bincode::serde::decode_from_slice(&buffer, config::standard()).unwrap_or_else(|_| {
            //let mut bincode_file_handle: ParDecompress<Mgzip> = ParDecompressBuilder::new().from_reader( fs::File::open(bincode_file).expect("Would have already broken if we didn't get here"));
            
            let mut bincode_file_handle: ParDecompress<Mgzip> = ParDecompressBuilder::new().from_reader( fs::File::open(bincode_file)?);
            buffer.clear();
            bincode_file_handle.read_to_end(&mut buffer)?;
            bincode::serde::decode_from_slice(&buffer, config::standard())?
            /*match bincode_file_handle.read_to_end(&mut buffer) {
                Ok(_) => {}
                Err(e) => return Err(Box::new(e)),
            };
            match bincode::serde::decode_from_slice(&buffer, config::standard()) {
                Ok(p) => p,
                Err(e) => return Err(Box::new(e)),
            }*/
        });
*/
        Ok(prior_state.reactivate_set(data_ref))

    }

    /// This calculates, from scratch, the total occupancy trace for a MotifSet
    /// Note: we don't actually use this in the inference: because occupancy 
    /// traces are assumed linear and they are also the major time suck of the
    /// inference, we generally calculate occupancy traces by taking the difference
    /// between the new MotifSet and some prior MotifSet
    pub fn recalced_signal(&self) -> Waveform {
        let mut signal = self.data_ref.data().derive_zero();
        for (mot, _) in self.set.iter() {
            signal += &(mot.generate_waveform(self.data_ref));
        }

        signal
    }

    fn recalc_signal(&mut self) {
        self.signal = self.data_ref.data().derive_zero();
        for (mot, _) in self.set.iter() {
            self.signal += &(mot.generate_waveform(self.data_ref));
        }
        self.ln_post = None;
    }

    pub fn resids_hist(&self, nclass: usize) -> Vec<(f64, f64)> {
        self.signal.produce_noise(self.data_ref).noise_densities(nclass)
    }

    fn calc_motif_null_binds(&self, mot: &Motif) -> Vec<f64> {
    
        mot.return_any_null_binds_by_hamming(self.data_ref.null_seq(), self.data_ref.min_height(), self.data_ref.offset()*2.0)
    }
    fn calc_motif_null_binds_entire(&self, mot: &Motif) -> Vec<f64> {
    
        mot.return_any_null_binds_by_hamming_no_limit(self.data_ref.null_seq(), self.data_ref.min_height(), self.data_ref.offset()*2.0)
    }

    ///This shows the vector of all null binding scores that TARJIM takes into consideration
    pub fn calculate_nulls(&self) -> Vec<f64> {
        let mut nulls = self.nth_motif(0).return_any_null_binds_by_hamming(self.data_ref.null_seq(), self.data_ref.min_height(), self.data_ref.offset()*2.0);
        if self.set.len() > 1 {
            for i in 1..self.set.len() { nulls.append(&mut self.nth_motif(i).return_any_null_binds_by_hamming(self.data_ref.null_seq(), self.data_ref.min_height(), self.data_ref.offset()*2.0));}
        }
        nulls
    }
    
    ///This shows the vector of all null binding scores that TARJIM could theoretically account for
    pub fn calculate_unlimited_nulls(&self) -> Vec<f64> {
        let mut nulls = self.nth_motif(0).return_any_null_binds_by_hamming_no_limit(self.data_ref.null_seq(), self.data_ref.min_height(), self.data_ref.offset()*2.0);
        if self.set.len() > 1 {
            for i in 1..self.set.len() { nulls.append(&mut self.nth_motif(i).return_any_null_binds_by_hamming_no_limit(self.data_ref.null_seq(), self.data_ref.min_height(), self.data_ref.offset()*2.0));}
        }
        nulls
    }

    /// This takes the current MotifSet and sorts its motifs in descending `peak_height()` order
    pub fn sort_by_height(&mut self) {
        self.set.sort_unstable_by(|(a,_),(b,_)| b.peak_height().partial_cmp(&a.peak_height()).unwrap() );
    }

    /// This takes the current MotifSet and gives the set with only the `n` 
    /// largest `peak_height()` `Motif`s
    pub fn only_n_strongest(&'a self, n: usize) -> MotifSet<'a> {

        let mut sort_set = self.clone();
        sort_set.sort_by_height();
        
        if n >= self.set.len() { return sort_set;} 

        let mut cumulative_signal = self.signal.derive_zero();

        for i in 0..n {
            cumulative_signal += &sort_set.nth_motif(i).generate_waveform(self.data_ref);
        }

        let mut sub_set = MotifSet {

            set: (0..n).map(|a| sort_set.set[a].clone()).collect(),
            signal: cumulative_signal.clone(),
            ln_post: None, 
            data_ref: self.data_ref
        };

        _ = sub_set.ln_posterior();

        sub_set

    }

    /// This takes the MotifSet and outputs the ln posterior densities
    /// `self.only_n_strongest(n)` for `n=1..=self.len()`, in order.
    pub fn ln_posts_by_strength(&self) -> Vec<f64> {

        let mut motif_set = self.clone();

        motif_set.set.sort_unstable_by(|(a,_),(b,_)| b.peak_height().partial_cmp(&a.peak_height()).unwrap() );

        let mut cumulative_signal = self.signal.derive_zero();

        let mut ln_posts: Vec<f64> = Vec::with_capacity(self.set.len());

        for i in 0..self.set.len() {

            cumulative_signal += &motif_set.nth_motif(i).generate_waveform(self.data_ref);

            let mut cumulative_set = MotifSet {

                set: (0..=i).map(|a| motif_set.set[a].clone()).collect(),
                signal: cumulative_signal.clone(),
                ln_post: None,
                data_ref: self.data_ref,

            };

            ln_posts.push(cumulative_set.ln_posterior());


        }

        ln_posts

    }
    
    /// This takes the MotifSet and outputs the Euclidean distances of 
    /// the occupancy traces of `self.only_n_strongest(n)` for `n=1..=self.len()`,
    /// in order.
    pub fn distances_by_strength(&self) -> Vec<f64> {

        let mut motif_set = self.clone();

        motif_set.set.sort_unstable_by(|(a,_),(b,_)| b.peak_height().partial_cmp(&a.peak_height()).unwrap() );

        let mut cumulative_signal = self.signal.derive_zero();

        let mut dists: Vec<f64> = Vec::with_capacity(self.set.len());

        for i in 0..self.set.len() {

            cumulative_signal += &motif_set.nth_motif(i).generate_waveform(self.data_ref);

            let mut cumulative_set = MotifSet {

                set: (0..i).map(|a| motif_set.set[a].clone()).collect(),
                signal: cumulative_signal.clone(),
                ln_post: None,
                data_ref: self.data_ref,

            };

            dists.push(cumulative_set.magnitude_signal_with_noise());


        }

        dists

    }

    /// This takes the current MotifSet and attempt to save a series of files to
    /// the directory `dir={output_dir}/{file_name}_new_occupancy`, creating it if
    /// necessary. The files are:
    /// - `{dir}/motif_set.bin`, which is a serialized binary of a skinny version of self
    /// - `{dir}/total/from_{start}_to_{end}.png`, the portions of the occupancy trace base pair `start` to `end`
    /// - `{dir}/Motif_{i}/from_{start}_to_{end}.png`, the portions of the occupancy trace where the `i`th strongest motif generated positive occupancy from base pair `start` to `end`
    /// - `{dir}/Strongest_{i}_Motifs/from_{start}_to_{end}.png`, the portions of the occupancy trace where the sum of the `i` strongest motifs generated positive occupancy from base pair `start` to `end`
    /// - `{dir}/from_{start}_to_{end}/total.png`, the total occuapncy from base pair `start` to `end`
    /// - `{dir}/from_{start}_to_{end}/Motif_{i}.png`, the occupancy from base pair `start` to `end` for the `i`th strongest motif, generated only if the `i`th strongest motif had positive occupancy there. 
    /// - `{dir}/from_{start}_to_{end}/Strongest_{i}_motifs.png`, the sum of occupancy from base pair `start` to `end` for the set of the `i`th strongest motifs, generated only if the sum has positive occupancy
    /// # Errors
    /// If `dir` does not exist and cannot be created
    pub fn save_set_trace_and_sub_traces(&self, output_dir: &str, file_name: &str, annotations: Option<&GenomeAnnotations>, ontologies: Option<&[&str]>) -> Result<(), Box<dyn Error+Send+Sync>> {

        let mut motif_set = self.clone();

        motif_set.set.sort_unstable_by(|(a,_),(b,_)| b.peak_height().partial_cmp(&a.peak_height()).unwrap() );

        let signal = self.recalced_signal();

        let signal_directory: String = format!("{}/{}_new_occupancy",output_dir,file_name);

        println!("sig {signal_directory}");

        let mut outfile_handle = match fs::File::create(format!("{}/motif_set.bin", signal_directory)) {
            Err(_) => {
                fs::create_dir_all(signal_directory.clone())?;
                fs::File::create(format!("{}/motif_set.bin", signal_directory))?
            },
            Ok(o) => o
        };

        let mot_to_save: StrippedMotifSet = self.into();

        let buffer: Vec<u8> = bincode::serde::encode_to_vec( &mot_to_save, config::standard()).expect("serializable");

        outfile_handle.write(&buffer).expect("We just created this file");

        
        signal.save_waveform_to_directory(self.data_ref, &signal_directory, "total", &BLUE, false, annotations, ontologies);

        let mut cumulative_signal = self.signal.derive_zero();

        if self.set.len() > 1 {
            for i in 0..self.set.len() {

                let signal_name = format!("Motif_{}",i);
                let sub_signal = motif_set.nth_motif(i).generate_waveform(self.data_ref);
                sub_signal.save_waveform_to_directory(self.data_ref,&signal_directory, &signal_name, &GREEN, true,annotations, ontologies);

                cumulative_signal += &sub_signal;

                if i >= 1 { 
                    let accumulator_name = format!("Strongest_{}_motifs", i+1);
                    cumulative_signal.save_waveform_to_directory(self.data_ref,&signal_directory, &accumulator_name, &plotters::prelude::full_palette::PURPLE, true, annotations, ontologies);
                
                }

            }
        }

        Ok(())
    }

    /// This takes `self` and `other_set` and plots both of their occupancy 
    /// traces in each location in the directory 
    pub fn save_set_trace_comparisons(&self, other_set: &MotifSet, output_dir: &str, file_name: &str, self_name: &str, other_name: &str) -> Result<(), Box<dyn Error+Send+Sync>> {

        let self_sig = self.recalced_signal();
        let alter_sig = other_set.recalced_signal();

        let signal_directory: String = format!("{}/{}_{}_occupancy_compare",output_dir,file_name, other_name);

       self_sig.save_waveform_comparison_to_directory(&alter_sig, self.data_ref, &signal_directory, self_name, &plotters::prelude::full_palette::PINK_600, &plotters::prelude::full_palette::BLUE_A200,self_name, other_name)?;

       Ok(())

    }
    
    //Note: It is technically allowed to have a negative thermodynamic beta
    //      This will invert your mechanics to find your LOWEST likelihood region
    //      Which is bad for most use cases! So be warned. 
    fn accept_test<R: Rng + ?Sized>(old: f64, new: f64, thermo_beta: f64, rng: &mut R) -> bool {

        let diff_ln_like = (new-old)*thermo_beta;

        //println!("accept ln prob {}", diff_ln_like);
        //Always accept if new likelihood is better
        if diff_ln_like > 0.0 {
            true
        } else {
            //Accept according to the metropolis criterion
            let thresh: f64 = rng.gen();
            let a = thresh < diff_ln_like.exp();
            a
        }
    }

    fn derive_set(&self) -> Self {

        MotifSet {
            set: self.set.clone(),
            signal: self.signal.clone(),
            ln_post: None,
            data_ref: self.data_ref, //pointer
        }

    }

    /// Panics: if `n` is not less than the number of motifs
    pub fn nth_motif(&self, n: usize) -> &Motif {
        &self.set[n].0
    }

    /// Panics: if `n` is not less than the number of motifs
    pub fn nth_motif_mut(&mut self, n: usize) -> &mut Motif {
        &mut self.set[n].0
    }

    /// Panics: if `n` is not less than the number of motifs
    pub fn get_nth_motif(&self, n: usize) -> Motif {
        self.set[n].0.clone()
    }
    
    /// Panics: if `n` is not less than the number of motifs
    pub fn nth_motif_and_scores(&self, n: usize) -> &(Motif, Vec<f64>) {
        &self.set[n]
    }

    /// Panics: if `n` is not less than the number of motifs
    pub fn nth_motif_and_scores_mut(&mut self, n: usize) -> &mut (Motif, Vec<f64>) {
        &mut self.set[n]
    }

    /// Panics: if `n` is not less than the number of motifs
    pub fn nth_motif_as_set(&self, n: usize) -> Self {

        let new_set = vec!(self.nth_motif_and_scores(n).clone());
        let new_sig = new_set[0].0.generate_waveform(self.data_ref);

        let mut new_mot_set = MotifSet {
            set: new_set,
            signal: new_sig,
            ln_post: None,
            data_ref: self.data_ref
        };

        _ = new_mot_set.ln_posterior();

        new_mot_set
        

    }

    pub fn get_nth_motif_and_scores(&self, n: usize) -> (Motif, Vec<f64>) {
        self.set[n].clone()
    }

    pub fn null_peak_scores(&self) -> Vec<f64> {
        self.set.iter().map(|a| a.1.iter().map(|&b| b)).flatten().collect()
    }

    pub fn num_motifs(&self) -> usize { self.set.len() }
    
    pub fn len(&self) -> usize {
        self.set.len()
    }

    pub fn set_copy(&self) -> Vec<Motif> {
        self.set.iter().map(|a| a.0.clone()).collect()
    }

    pub fn taxicab(&self) -> f64 {
        self.set.iter().map(|a| a.0.taxicab()).sum::<f64>()
    }

    /// This is our prior on the number of motifs
    /// We do not justify this with a maximum entropy prior
    /// Instead, we only want to consider an additional motif if 
    /// it brings an improvement of at least NECESSARY_MOTIF_IMPROVEMENT to the ln posterior
    /// This amounts to a geometric prior with positive integer support 
    /// and p = 1-(-NECESSARY_MOTIF_IMPROVEMENT).exp().
    /// Note: the ommission of ln(p) term is deliberate. It amounts to a normalization constant
    /// for the motif set prior, and would obfuscate the true point of this prior
    pub fn motif_num_prior(&self, credibility: f64) -> f64 {
         if self.set.len() > MAX_TF_NUM { return -f64::INFINITY; } ;
        -((self.set.len()-1) as f64)* credibility
    }

    pub fn ln_prior(&self) -> f64 {
        //println!("motif num {} heights {:?} pwms {:?}",self.motif_num_prior(), self.set.iter().map(|a| a.0.height_prior()).collect::<Vec<_>>(), self.set.iter().map(|a| a.0.pwm_prior(self.data_ref.data().seq())).collect::<Vec<_>>());
        if !self.set.iter().all(|a| a.0.has_multiple_binding_sites(self.data_ref.data().seq(), -self.data_ref.min_height())) { return -f64::INFINITY; }
        self.motif_num_prior(self.data_ref.credibility()) + self.set.iter().map(|a| a.0.height_prior(self.data_ref.height_dist())+a.0.pwm_prior(self.data_ref.data().seq())).sum::<f64>()
    }

    pub fn ln_likelihood(&self) -> f64 {
        //println!("made it to like");
        //println!("nulls {:?}", self.null_peak_scores());
        //println!("noise {:?}", self.signal.produce_noise_with_extraneous(self.data_ref, &self.null_peak_scores()).extraneous_resids());
        //println!("nulls {:?} ad_calc {}", self.null_peak_scores(), Noise::ad_like((self.signal).produce_noise_with_extraneous(self.data_ref, &self.null_peak_scores()).ad_calc(self.data_ref.data().spacer())));
        Noise::ad_like((self.signal).produce_noise_with_extraneous(self.data_ref, &self.null_peak_scores()).ad_calc(self.data_ref.data().spacer()))
    }

    pub fn eth(&self) -> f64 {
        (self.signal).produce_noise_with_extraneous(self.data_ref, &self.null_peak_scores()).ad_calc(self.data_ref.data().spacer())
    }

    pub fn magnitude_signal_with_noise(&self) -> f64 {
        let a = (self.signal).produce_noise_with_extraneous(self.data_ref, &self.null_peak_scores()).rmse_noise(self.data_ref.data().spacer());
        a
            
    }
                    

    /// This checks if we have calculated the ln posterior for `self`. 
    /// If we have not, it sets and returns it. If we have, it just returns it.
    /// If you only have a `&MotifSet` and wish to get its ln posterior
    /// without mutation, use `self.calc_ln_post()`
    pub fn ln_posterior(&mut self) -> f64 { //By using this particular structure, I always have the ln_posterior when I need it and never regenerate it when unnecessary
        match self.ln_post {
            None => {
                let ln_prior = self.ln_prior();
                if ln_prior > -f64::INFINITY { //This short circuits the noise calculation if our motif set is somehow impossible
                    self.ln_post = Some(ln_prior+self.ln_likelihood());
                } else{
                    self.ln_post = Some(-f64::INFINITY);
                }
                self.ln_post.unwrap()}, 
            Some(f) => f,
        }
    }

    /// This checks if we have calculated the ln posterior for `self`. 
    /// If we have not, it calculates and returns it. If we have, it just returns it.
    /// If you have a `&mut MotifSet` and wish to save the results of the 
    /// calculation, use `self.ln_posterior()`
    pub fn calc_ln_post(&self) -> f64 { 
        match self.ln_post {
            None => {
                let mut ln_post: f64 = self.ln_prior();
                    if ln_post > -f64::INFINITY { //This short circuits the noise calculation if our motif set is somehow impossible
                        ln_post += self.ln_likelihood();
                    } 
                    ln_post
                }, 
                Some(f) => f,
            }
       
    }


    fn add_motif(&mut self, new_mot: Motif) -> f64 {

        self.signal += &new_mot.generate_waveform(self.data_ref) ;
        let new_nulls = self.calc_motif_null_binds(&new_mot);
        self.set.push((new_mot, new_nulls));
        self.ln_post = None;
        self.ln_posterior()

    }

    fn insert_motif(&mut self, new_mot: Motif, position: usize) -> f64 {

        self.signal += &new_mot.generate_waveform(self.data_ref) ;
        let new_nulls = self.calc_motif_null_binds(&new_mot);
        self.set.insert(position, (new_mot, new_nulls));
        self.ln_post = None;
        self.ln_posterior()

    }

    fn remove_motif(&mut self, rem_id: usize) -> f64{
        assert!(rem_id < self.set.len());

        let rem_mot = self.set.swap_remove(rem_id);
        self.signal -= &rem_mot.0.generate_waveform(self.data_ref);
        self.ln_post = None;
        self.ln_posterior()

    }

    fn remove_motif_void(&mut self, rem_id: usize) {

        assert!(rem_id < self.set.len());
        let rem_mot = self.set.swap_remove(rem_id);
        self.signal -= &rem_mot.0.generate_waveform(self.data_ref);
        self.ln_post = None;

    }


    fn replace_motif(&mut self, new_mot: Motif, rem_id: usize) -> f64 {
        let rem_mot = self.nth_motif(rem_id).clone();
        //println!("rep 1");
        self.signal -= &rem_mot.generate_waveform(self.data_ref);
        //println!("rep 2");
        self.signal += &new_mot.generate_waveform(self.data_ref) ;
        //println!("rep 3");
        //println!("{:?}", new_mot);
        let new_nulls = self.calc_motif_null_binds(&new_mot); 
        //println!("rep 4");
        self.set[rem_id] = (new_mot, new_nulls);
        //println!("rep 5");
        self.ln_post = None;
        //println!("rep 6");
        self.ln_posterior()

    }


    fn replace_motif_entire(&mut self, new_mot: Motif, rem_id: usize) -> f64 {
        let rem_mot = self.nth_motif(rem_id).clone();
        //println!("rep 1");
        self.signal -= &rem_mot.generate_waveform(self.data_ref);
        //println!("rep 2");
        self.signal += &new_mot.generate_waveform(self.data_ref) ;
        //println!("rep 3");
        //println!("{:?}", new_mot);
        let new_nulls = self.calc_motif_null_binds_entire(&new_mot); 
        //println!("rep 4");
        self.set[rem_id] = (new_mot, new_nulls);
        //println!("rep 5");
        self.ln_post = None;
        //println!("rep 6");
        self.ln_posterior()

    }

    //Note: this can get really slow if other_set points to a different
    //      data ref than self does
    fn combine_motif_sets(&self, other_set: &Self) -> Self {

        let mut new_set = self.clone();

        if !ptr::eq(self.data_ref, other_set.data_ref) {
            for i in 0..other_set.len() {
                _ = new_set.add_motif(other_set.get_nth_motif(i));
            }
            return new_set;
        }

        //If we point at the same data waveform, we don't need to reinvent the 
        //wheel: we already calculated the occupancy traces.
        new_set.set.append(&mut other_set.set.clone());
        new_set.signal += &other_set.signal;
        new_set.ln_post = None;

        _ = new_set.ln_posterior();

        new_set

    }

    fn recalc_negatives(&mut self) {
        for i in 0..self.set.len() {
            let nulls = self.calc_motif_null_binds(self.nth_motif(i));
            self.set[i].1 = nulls;
        };
    }

    /// This proposes a new motif for the motif set by the following process:
    /// 1. We pick `prefix`, a `MIN_BASE-mer` based on the propensities in `self.data_ref()`
    /// 2. We sample a motif length `k` uniformly from `MIN_BASE` to `MAX_BASE`
    /// 3. We uniformly sample from all the possible `k`-mers that start with `prefix`
    /// 4. We randomly generate a `Motif` with the best match of the `k`-mer from step 3, 
    ///    using `Base::from_bp()`, which favors more selective `Base`s
    /// Returns `[None]` if there are no `k`-mers which start with `prefix`
    /// (which can happen if `prefix` is generated from the edge of a sequence 
    /// block, albeit rarely), or if `self` has the maximum number of `Motif`s.
    /// Otherwise, returns a tuple of a `MotifSet` with the new `Motif` added, 
    /// followed by the ln posterior density - ln proposal probability
    pub fn propose_new_motif<R: Rng + ?Sized>(&self, rng: &mut R ) -> Option<(Self, f64)> {
        

        if self.set.len() > MAX_TF_NUM { return None; } ;

        let mut new_set = self.derive_set();
        
        let minmer_choice = self.data_ref.rand_minmer_by_propensity(rng);

        let k = rng.gen_range(MIN_BASE..=MAX_BASE);

        let (pot_kmer_choice, num) = self.data_ref.data().seq().rand_kmer_start_minmer_and_number(minmer_choice, k, rng);

        let Some(kmer_choice) = pot_kmer_choice else {return None;};

        let motif_choice = self.data_ref.data().seq().idth_unique_kmer_vec(k, kmer_choice);

        let proposal_method = rng.gen_range(0_usize..3);

        let (new_mot, base_dist) = match proposal_method {
                0 => (Motif::from_motif(motif_choice, self.data_ref.height_dist(), rng), &BASE_CHOOSE_DIST),
                1 => (Motif::from_motif_prior(motif_choice, self.data_ref.height_dist(), rng), &BASE_PENALTY_PRIOR),
                2 => (Motif::from_motif_alt(motif_choice, self.data_ref.height_dist(), rng), &BASE_CHOOSE_ALT),
                _ => unreachable!(),
        };
        //let new_mot = Motif::from_motif(motif_choice, self.data_ref.height_dist(), rng); //There may not always be a place with decent propensities


        //let pick_prob = (new_mot.len() as f64)*(-(BASE_L as f64).ln()-((BASE_L-1) as f64)*(-SCORE_THRESH).ln());

        //Probability of picking the best motif we did: P(pick minmer)*P(pick length k)*P(pick the kmer| starting minmer AND k)
        let pick_motif_prob = self.data_ref.propensity_minmer(minmer_choice).ln()-((MAX_BASE+1-MIN_BASE) as f64).ln() + (num as f64).ln();


        //Probability of picking the particular base vector we did GIVEN the best motif we already picked
        let pick_prob = new_mot.pwm.iter().map(|a| a.scores.iter().map(|&b| if b < 0.0 {base_dist.energy_ln_pmf(b)} else {0.0}).sum::<f64>()).sum::<f64>();

        let ln_gen_prob = self.data_ref.height_dist().ln_pdf(new_mot.peak_height-self.data_ref.min_height())+pick_prob+pick_motif_prob;
        //let h_prior = new_mot.height_prior();

        let ln_post = new_set.add_motif(new_mot);
        //println!("propose birth: like: {} height: {}, pick_prob: {}, len sel: {}",ln_post, h_prior, pick_prob.ln(), ((MAX_BASE+1-MIN_BASE) as f64).ln());
        Some((new_set, ln_post-ln_gen_prob)) //Birth moves subtract the probability of their generation
    }

    /// This reverses `propose_new_motif()`. 
    /// Returns `[None]` if `self` only has one `Motif`. 
    /// Otherwise, returns a tuple of a `MotifSet` with 
    /// the new `Motif` added, followed by the ln posterior density 
    /// + ln proposal probability for the reverse `propose_new_motif()`.
    pub fn propose_kill_motif<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<(Self, f64)> {

        if self.set.len() <= 1 { //We never want to propose a set with no motifs, ever
            None
        } else {
            let mut new_set = self.derive_set();
            let rem_id = rng.gen_range(0..self.set.len());

            let new_mot = self.nth_motif(rem_id);

            //let pick_prob = (self.nth_motif(rem_id).len() as f64)*(-(BASE_L as f64).ln()-((BASE_L-1) as f64)*(-SCORE_THRESH).ln());

            let contraction_method = rng.gen_range(0_usize..3);

            let base_dist = match contraction_method {
                0 => &BASE_CHOOSE_DIST,
                1 => &BASE_PENALTY_PRIOR,
                2 => &BASE_CHOOSE_ALT,
                _ => unreachable!(),
            };


            let pick_prob = new_mot.pwm.iter().map(|a| a.scores.iter().map(|&a| if a < 0.0 {
                base_dist.energy_ln_pmf(a)
            } 
            else {0.0}).sum::<f64>()).sum::<f64>();


            const MINMER_MASK: u64 = (1_u64 << ((MIN_BASE * 2) as u64)) - 1;

            let minmer_choice = Sequence::kmer_to_u64(&new_mot.best_motif()) & MINMER_MASK;

            let num = self.data_ref.data().seq().all_kmers_start_minmer(minmer_choice, new_mot.len()).len();

            //Probability of picking the best motif we did: P(pick minmer)*P(pick length k)*P(pick the kmer| starting minmer AND k)
            let pick_motif_prob = self.data_ref.propensity_minmer(minmer_choice).ln()-((MAX_BASE+1-MIN_BASE) as f64).ln() + (num as f64).ln();


            let ln_gen_prob = HEIGHT_PROPOSAL_DIST.ln_pdf(self.nth_motif(rem_id).peak_height-self.data_ref.min_height())+pick_prob+pick_motif_prob;

            let ln_post = new_set.remove_motif(rem_id);
            //println!("propose death: like: {} height: {}, pick_prob: {}, len sel: {}",ln_post, self.nth_motif(rem_id).height_prior(), pick_prob.ln(), ((MAX_BASE+1-MIN_BASE) as f64).ln());
            Some((new_set, ln_post+ln_gen_prob)) //Death moves add the probability of the generation of their deleted variable(s)
        }
    }
    
    /// This proposes a new motif for the motif set by the following process:
    /// 1. We pick `prefix`, a `MIN_BASE-mer` drawn uniformly 
    /// 2. We sample a motif length `k` uniformly from `MIN_BASE` to `MAX_BASE`
    /// 3. We uniformly sample from all the possible `k`-mers that start with `prefix`
    /// 4. We randomly generate a `Motif` with the best match of the `k`-mer from step 3, 
    ///    using `Base::from_bp_alt()`, which uniformly samples `Base` penalties
    /// Returns `[None]` if there are no `k`-mers which start with `prefix`
    /// (which can happen if `prefix` is generated from the edge of a sequence 
    /// block, albeit rarely), or if `self` has the maximum number of `Motif`s.
    /// Otherwise, returns a tuple of a `MotifSet` with the new `Motif` added, 
    /// followed by the ln posterior density - ln proposal probability
    pub fn propose_new_motif_alt<R: Rng + ?Sized>(&self, rng: &mut R ) -> Option<(Self, f64)> {
        
        if self.set.len() > MAX_TF_NUM { return None; } ;

        let mut new_set = self.derive_set();

        let k = rng.gen_range(MIN_BASE..=MAX_BASE);

        let motif_choice = self.data_ref.data().seq().random_valid_motif(k, rng);
        
        let proposal_method = rng.gen_range(0_usize..3);

        let (new_mot, base_dist) = match proposal_method {
                0 => (Motif::from_motif(motif_choice, self.data_ref.height_dist(), rng), &BASE_CHOOSE_DIST),
                1 => (Motif::from_motif_prior(motif_choice, self.data_ref.height_dist(), rng), &BASE_PENALTY_PRIOR),
                2 => (Motif::from_motif_alt(motif_choice, self.data_ref.height_dist(), rng), &BASE_CHOOSE_ALT),
                _ => unreachable!(),
        };



        //let pick_prob = (new_mot.len() as f64)*(-(BASE_L as f64).ln()-((BASE_L-1) as f64)*(-SCORE_THRESH).ln());

        //Probability of picking the best motif we did: P(pick minmer)*P(pick length k)*P(pick the kmer| starting minmer AND k)
        let pick_motif_prob = (-(self.data_ref.data().seq().number_unique_kmers(k) as f64)).ln()-((MAX_BASE+1-MIN_BASE) as f64).ln();

        //Probability of picking the particular base vector we did GIVEN the best motif we already picked
        let pick_prob = new_mot.pwm.iter().map(|a| a.scores.iter().map(|&b| if b < 0.0 {base_dist.energy_ln_pmf(b)} else {0.0}).sum::<f64>()).sum::<f64>();

        let ln_gen_prob = self.data_ref.height_dist().ln_pdf(new_mot.peak_height-self.data_ref.min_height())+pick_prob+pick_motif_prob;
        //let h_prior = new_mot.height_prior();

        let ln_post = new_set.add_motif(new_mot);
        //println!("propose birth: like: {} height: {}, pick_prob: {}, len sel: {}",ln_post, h_prior, pick_prob.ln(), ((MAX_BASE+1-MIN_BASE) as f64).ln());
        Some((new_set, ln_post-ln_gen_prob)) //Birth moves subtract the probability of their generation
    }

    /// This reverses `propose_new_motif_alt()`. 
    /// Returns `[None]` if `self` only has one `Motif`. 
    /// Otherwise, returns a tuple of a `MotifSet` with 
    /// the new `Motif` added, followed by the ln posterior density 
    /// + ln proposal probability for the reverse `propose_new_motif_alt()`.
    pub fn propose_kill_motif_alt<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<(Self, f64)> {

        if self.set.len() <= 1 { //We never want to propose a set with no motifs, ever
            None
        } else {
            let mut new_set = self.derive_set();
            let rem_id = rng.gen_range(0..self.set.len());

            let new_mot = self.nth_motif(rem_id);

            //let pick_prob = (self.nth_motif(rem_id).len() as f64)*(-(BASE_L as f64).ln()-((BASE_L-1) as f64)*(-SCORE_THRESH).ln());

            let contraction_method = rng.gen_range(0_usize..3);

            let base_dist = match contraction_method {
                0 => &BASE_CHOOSE_DIST,
                1 => &BASE_PENALTY_PRIOR,
                2 => &BASE_CHOOSE_ALT,
                _ => unreachable!(),
            };


            let pick_prob = new_mot.pwm.iter().map(|a| a.scores.iter().map(|&a| if a < 0.0 {
                base_dist.energy_ln_pmf(a)
            } 
            else {0.0}).sum::<f64>()).sum::<f64>();
       

            //Probability of picking the best motif we did: P(pick minmer)*P(pick length k)*P(pick the kmer| starting minmer AND k)
            let pick_motif_prob = (-(self.data_ref.data().seq().number_unique_kmers(self.nth_motif(rem_id).len()) as f64)).ln()-((MAX_BASE+1-MIN_BASE) as f64).ln();


            let ln_gen_prob = self.data_ref.height_dist().ln_pdf(self.nth_motif(rem_id).peak_height-self.data_ref.min_height())+pick_prob+pick_motif_prob;

            let ln_post = new_set.remove_motif(rem_id);
            //println!("propose death: like: {} height: {}, pick_prob: {}, len sel: {}",ln_post, self.nth_motif(rem_id).height_prior(), pick_prob.ln(), ((MAX_BASE+1-MIN_BASE) as f64).ln());
            Some((new_set, ln_post+ln_gen_prob)) //Death moves add the probability of the generation of their deleted variable(s)
        }
    }
    
/*     pub fn rand_new<R: Rng + ?Sized>(rng: &mut R) -> Base {


        let best = rng.gen_range(0..BASE_L);
        let samps: [f64;BASE_L-1] = core::array::from_fn(|_| BASE_CHOOSE_DIST.sample_energy(rng));

        let nonbest: [usize; BASE_L-1] = (0..BASE_L).filter(|&a| a != best).collect::<Vec<_>>().try_into().unwrap();

        let mut att =  [0.0_f64; BASE_L];

        nonbest.into_iter().enumerate().for_each(|(i,a)| {att[a] = samps[i];});


        Base { scores: att}
    }

    /// This generates a random Base by sampling energy penalties
    /// from integer multiples of BASE_RESOLUTION*SCORE_THRESH <= SCORE_THRESH,
    /// favoring less strict penalities, and then picking a best Bp uniformly
    pub fn rand_new_prior<R: Rng + ?Sized>(rng: &mut R) -> Base {


        let best = rng.gen_range(0..BASE_L);
        let samps: [f64;BASE_L-1] = core::array::from_fn(|_| BASE_PENALTY_PRIOR.sample_energy(rng));

        let nonbest: [usize; BASE_L-1] = (0..BASE_L).filter(|&a| a != best).collect::<Vec<_>>().try_into().unwrap();

        let mut att =  [0.0_f64; BASE_L];

        nonbest.into_iter().enumerate().for_each(|(i,a)| {att[a] = samps[i];});


        Base { scores: att}
    }


    /// This generates a random Base by uniformly sampling energy penalties 
    /// from integer multiples of BASE_RESOLUTION*SCORE_THRESH <= SCORE_THRESH,
    /// then picking a best Bp uniformly
    pub fn rand_new_alt<R: Rng + ?Sized>(rng: &mut R) -> Base {

        let best = rng.gen_range(0..BASE_L);
        let samps: [f64;BASE_L-1] = core::array::from_fn(|_| base_ceil(BASE_CHOOSE_ALT.sample_energy(rng)));

        let nonbest: [usize; BASE_L-1] = (0..BASE_L).filter(|&a| a != best).collect::<Vec<_>>().try_into().unwrap();

        let mut att =  [0.0_f64; BASE_L];

        nonbest.into_iter().enumerate().for_each(|(i,a)| {att[a] = samps[i];});


        Base { scores: att}
    }

*/

    /// This proposes to extend a `Motif` in `self` by the following process: 
    /// 1. Uniformly sample a `Motif` in `self`.
    /// 2. Take the best matching kmer of this motif, and uniformly sample 
    ///    from all of its valid single `Bp` 3' extensions
    /// 3. Generate a new `Base` matching the `Bp` using `Base::from_bp()`
    /// Returns `[None]` if the sampled motif has length `MAX_BASE`.
    /// Otherwise, returns a tuple of the `MotifSet` with the extended `Motif`, 
    /// followed by the ln posterior density - ln proposal probability
    pub fn propose_extend_motif<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<(Self, f64)> {
        let mut new_set = self.derive_set();

        let extend_id = rng.gen_range(0..self.set.len());

        if self.nth_motif(extend_id).len() >= MAX_BASE { //We want to immediately reject any move which extends a motif beyond maximum length
            None
        } else {

            let mut add_base_mot = self.nth_motif(extend_id).best_motif();
            add_base_mot.push(Bp::A);
            let last_ind = add_base_mot.len()-1;
            let mut valid_extends: Vec<Bp> = Vec::with_capacity(BASE_L);
            for &b in BP_ARRAY.iter() {
                add_base_mot[last_ind] = b;
                if self.data_ref.data().seq().kmer_in_seq(&add_base_mot) { valid_extends.push(b); }
            }
            if valid_extends.len() == 0 { return None; }

            //println!("lock");
            let new_bp = valid_extends.choose(rng).expect("We already returned if there are no valid extensions");

            let mut new_mot = self.nth_motif(extend_id).clone();
            
            let extension_method = rng.gen_range(0_usize..3);

            let (new_base, base_dist) = match extension_method {
                0 => (Base::from_bp(*new_bp, rng), &BASE_CHOOSE_DIST),
                1 => (Base::from_bp_prior(*new_bp, rng), &BASE_PENALTY_PRIOR),
                2 => (Base::from_bp_alt(*new_bp, rng), &BASE_CHOOSE_ALT),
                _ => unreachable!(),
            };


            //println!("lock 2");
            let base_ln_density = new_base.scores.iter().map(|&a| if a < 0.0 {base_dist.energy_ln_pmf(a)} else {0.0}).sum::<f64>() - ((valid_extends.len() as f64).ln());//+LN_2+1.09861228867);
            //ln(2) for two choices: front vs back. Ln(3) for three choices of base distribution;
            //println!("lock 3");
            if rng.gen::<bool>() { new_mot.pwm.push(new_base)} else {new_mot.pwm.insert(0, new_base)};
            //println!("lock 4");
            //let ln_gen_prob = new_mot.height_prior()+new_mot.pwm_prior(self.data.seq());
            let ln_post = new_set.replace_motif(new_mot, extend_id);
            //println!("lock 5");
            Some((new_set, ln_post-base_ln_density)) //Birth moves subtract the ln probability of their generation
        }
    }

    /// This reverses `self.propose_extend_motif()`.
    /// Returns `[None]` if the `Motif` selected for contraction has length 
    /// `MIN_BASE`. Otherwise, returns a tuple of the `MotifSet` with the contracted
    /// `Motif`, followed by the ln posterior density + ln proposal probability
    /// of the corresponding forward extension. 
    pub fn propose_contract_motif<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<(Self, f64)> {
        let mut new_set = self.derive_set();

        let contract_id = rng.gen_range(0..self.set.len());

        if self.nth_motif(contract_id).len() <= MIN_BASE { //We want to immediately reject any move which contracts a motif below minimum length
            None
        } else {
            let mut new_mot = self.nth_motif(contract_id).clone();
            //println!("contract 0");
            let contraction_method = rng.gen_range(0_usize..3);

            let base_dist = match contraction_method {
                0 => &BASE_CHOOSE_DIST,
                1 => &BASE_PENALTY_PRIOR,
                2 => &BASE_CHOOSE_ALT,
                _ => unreachable!(),
            };

            let new_mot_bps = new_mot.best_motif();
            //println!("contract 1");
            let pop_back = rng.gen::<bool>();
            let old_base = if pop_back {new_mot.pwm.pop()?} else {new_mot.pwm.remove(0)};
            //println!("contract 2");
            let ln_post = new_set.replace_motif(new_mot, contract_id);
            let ln_matches = if pop_back {self.data_ref.data().seq().number_kmers_neighboring_by_last_bp(&new_mot_bps).ln()} else {self.data_ref.data().seq().number_kmers_neighboring_by_first_bp(&new_mot_bps).ln()};
            //println!("contract 3");
            let base_ln_density = old_base.scores.iter().map(|&a| if a < 0.0 {base_dist.energy_ln_pmf(a)} else {0.0}).sum::<f64>()-(ln_matches);//+LN_2+1.09861228867);
            //ln(2) for two choices: front vs back. Ln(3) for three choices of base distribution;
            //println!("contract 4");
            Some((new_set, ln_post+base_ln_density)) //Birth moves subtract the probability of their generation
        }
    }



    //MOVE TO CALL
    //For borrow checker reasons, this will only work if the motif calling already has a generated likelihood
    //And I can't have it do it here
    //If you use this on a motif without such a likelihood, it will panic
    // (rmse, likelihood_diff, pwm_dists_and_height_diffs)
    /// Runs a randomly selected reversible jump move. 
    /// Returns a tuple of (the next `MotifSet` in the Monte Carlo chain, 
    /// a `[usize]` indicating which reversible jump move was attempted, 
    /// `true` (`false`) if the reversible jump move was accepted (rejected))
    /// # Panics
    /// If `self` has not had its ln posterior generated yet
    pub fn run_rj_move<R: Rng + ?Sized>(&self, thermo_beta: f64, rng: &mut R) -> (Self, usize, bool) {

        let which_rj = rng.gen_range(0..RJ_MOVE_NAMES.len());

        //println!("which {which_rj}");

        let proposal: Option<(Self, f64)> = match which_rj {

            0 => self.propose_new_motif(rng),
            1 => self.propose_kill_motif(rng),
            2 => self.propose_extend_motif(rng),
            3 => self.propose_contract_motif(rng),
            4 => self.propose_new_motif_alt(rng),
            5 => self.propose_kill_motif_alt(rng),
            _ => unreachable!("How you managed to get here, I do not know. You're somehow trying to make a move run when it doesn't exist.
                              \n There's a REASON that I coded my number of moves as a magic number that I use only once. 
                              \n Namely, there's just no good way to change that number and expect everything to work correctly.
                              \n Unless you coded up new moves entirely for your system, of course.
                              \n In which case, CHANGE THIS MATCH STATEMENT. If you did not code new moves, you probably somehow 
                              \n changed the generation of which_rj. In which case, the correct form is `let which_rj = rng.gen_range(0..4);`"),
        };

        match proposal {

            None => {
                //println!("Failed move {}", which_rj);
                (self.clone(), which_rj, false)
            },
            Some((new_mot, modded_ln_like)) => {
                let accepted = Self::accept_test(self.ln_post.unwrap(), modded_ln_like, thermo_beta, rng);
                //println!("old ln P {}, modded ln P {}, move {:?}, accepted: {}", self.ln_post.unwrap(), modded_ln_like, RJ_MOVE_NAMES[which_rj], accepted);
                /*if accepted {
                    (new_mot, which_rj, true)
                } else { 
                    (self.clone(), which_rj, false)
                }*/

                (new_mot, which_rj, accepted)
            }
        }

    }

    //Panics: if which_rj >= number of rj moves encoded here
    fn run_rj_move_known<R: Rng + ?Sized>(&self, which_rj: usize, thermo_beta: f64, rng: &mut R) -> Option<(Self, bool)> {

        let proposal: Option<(Self, f64)> = match which_rj {

            0 => self.propose_new_motif(rng),
            1 => self.propose_kill_motif(rng),
            2 => self.propose_extend_motif(rng),
            3 => self.propose_contract_motif(rng),
            4 => self.propose_new_motif_alt(rng),
            5 => self.propose_kill_motif_alt(rng),
            _ => panic!("You're trying to make a move run when it doesn't exist.")
        };


        match proposal {

            None => {
                //println!("Failed move {}", which_rj);
                None
            },
            Some((new_mot, modded_ln_like)) => {
                let accepted = Self::accept_test(self.ln_post.unwrap(), modded_ln_like, thermo_beta, rng);
                Some((new_mot, accepted))
            }
        }
    }


    /// This proposes to change the best `Bp`s of a `Motif` in `self` by swapping.
    /// The proposed best bases are generated by the following random generation:
    /// 1. A `Motif` is randomly selected out of `self`.
    /// 2. The occupancy of the selected `Motif` is subtracted from `self`'s total trace
    /// 3. The remaining trace is subtracted from the data
    /// 4. Locations are divided up into two classes: residual `>= cutoff` and `< cutoff`.
    /// 5. 99% of the time, we uniformly choose a kmer that is under the first class. 
    ///    1% of the time, we uniformly choose a kmer that is under the second class.
    /// 6. We shuffle the `Motif` to have a best `kmer` equal to the kmer we chose. 
    /// 
    /// Returns `[None]` when we attempt an impossible move. Otherwise, returns the
    /// tuple (Proposed `MotifSet`, ln posterior of propoosal + ln probability we 
    /// propose the reverse move - ln probability we propose the forward move)
    pub fn propose_resid_leap<R: Rng + ?Sized>(&self, cutoff: f64, rng: &mut R) -> Option<(Self, f64)> {


        //This can't be fully one, or else we could never get rid of a motif that bound somewhere unpeaky
        //Because then that move couldn't be reversed
        const PICK_PEAK_THING: f64 = 0.99;

        //I wish Rust had const ln
        let peak_pick_ln = PICK_PEAK_THING.ln();

        let unpeak_pick_ln = (-PICK_PEAK_THING).ln_1p();

        let mut new_set = self.derive_set();

        let id = rng.gen_range(0..self.set.len());

        let (old_motif, old_off_targets) = self.set[id].clone();

        let num_bases = old_motif.len();

        _ = new_set.remove_motif(id);

        let unaccounted_trace = self.data_ref.data()-&new_set.signal;

        //Still have to specify types of peaky and unpeaky
        //let (peaky, unpeaky): (Vec<_>, Vec<_>)  = unaccounted_trace.read_wave().into_iter().enumerate().partition(|(_, &a)| a >= self.data_ref.min_height()); 

        let unaccounted_wave = unaccounted_trace.read_wave();


        let mut peaky_locs: Vec<usize> = Vec::with_capacity(unaccounted_wave.len());

        let mut unpeaky_locs: Vec<usize> = Vec::with_capacity(unaccounted_wave.len());

        let mut is_peaky: Vec<bool> = vec![false; unaccounted_wave.len()];

        for (i, &remain) in unaccounted_wave.into_iter().enumerate() {
            if remain >= cutoff {
            
                peaky_locs.push(i);
                is_peaky[i] = true;

            } else {
                unpeaky_locs.push(i);
                //Default is already false
                //is_peaky[i] = false;
            }
        }

        let r: f64 = rng.gen();

        //println!("peaky locs len {} unpeaky locs len {}", peaky_locs.len(), unpeaky_locs.len());

        let &choice = if r <= PICK_PEAK_THING {
            //Autoreject if we chose to do the peak move and couldn't
            //if peaky_locs.len() == 0 { return None;}
            //peaky_locs.choose(rng)?
            let Some(cho) = peaky_locs.choose(rng) else { return None;};
            cho
        } else {
            //unpeaky_locs.choose(rng)?
            let Some(cho) = peaky_locs.choose(rng) else { return None;};
            cho
        };
        
        let (block, start_range) = self.data_ref.data().intersect_kmer_start_range(choice, num_bases)?;

        //This should basically never return None, but in case, there's this short circuit
        //
        let choose_start = start_range.choose(rng);


        let chosen_start = match choose_start {
            Some(choice) => choice,
            None => {
                println!("chosen start nulls out {:?}", self.data_ref.data().intersect_kmer_start_range(choice, num_bases));
                return None;
            },
        };
       

        //let chosen_start = start_range.choose(rng)?;

        let new_kmer = self.data_ref.data().seq().return_bases(block, chosen_start, num_bases);

        //scramble to kmer
        let new_mot = old_motif.scramble_by_kmer(&new_kmer)?;


        //Determine likelihood of choices
        let old_best_binds = old_motif.return_best_bind_inds(self.data_ref.data().seq());

        let select_reverse_ln_prob = old_best_binds.iter().filter_map(|&(block, base)| self.data_ref.data().intersect_data_start_range_block_and_base(block, base, num_bases)).flatten()
                                                  .map(|data_ind| { 
                                                      let (bin_select_ln_prob, bin_size_ln_prob) = if is_peaky[data_ind] { (peak_pick_ln, -(peaky_locs.len() as f64).ln()) } else { (unpeak_pick_ln, -(unpeaky_locs.len() as f64).ln()) };
                                                      let kmer_select_ln_prob = -(self.data_ref.data().intersect_kmer_start_num(data_ind, num_bases) as f64).ln();
                                                      bin_select_ln_prob+bin_size_ln_prob+kmer_select_ln_prob
                                                  }).sum::<f64>();

        let new_best_binds = new_mot.return_best_bind_inds(self.data_ref.data().seq());

        let select_forward_ln_prob = new_best_binds.iter().filter_map(|&(block, base)| self.data_ref.data().intersect_data_start_range_block_and_base(block, base, num_bases)).flatten()
                                                  .map(|data_ind| {
                                                      let (bin_select_ln_prob, bin_size_ln_prob) = if is_peaky[data_ind] { (peak_pick_ln, -(peaky_locs.len() as f64).ln()) } else { (unpeak_pick_ln, -(unpeaky_locs.len() as f64).ln()) };
                                                      let kmer_select_ln_prob = -(self.data_ref.data().intersect_kmer_start_num(data_ind, num_bases) as f64).ln();
                                                      bin_select_ln_prob+bin_size_ln_prob+kmer_select_ln_prob
                                                  }).sum::<f64>();
        let mut modded_like = new_set.insert_motif(new_mot, id);

        modded_like = modded_like +select_reverse_ln_prob-select_forward_ln_prob;
        //println!("select ratio old like {:?} new_like {:?} modded like {} ratio {}", self.ln_post, new_set.ln_post, modded_like, modded_like-self.ln_post.unwrap());

        Some((new_set, modded_like))
    }

    /// This proposes to change the best `Bp`s  of a `Motif` in `self` by swapping.
    /// The proposed best bases are generated by the following random generation:
    /// 1. A `Motif` is randomly selected out of `self`.
    /// 2. All kmers with a maximum Hamming distance of `7` from the best kmer is chosen.
    /// 3. A random Hamming distance `H`, up to 7 is chosen uniformly from all 
    ///    distances . 
    /// 4. We 
    /// 3. We shuffle the `Motif` to have a best `kmer` equal to the kmer we chose. 
    /// 
    /// Returns `[None]` when we attempt an impossible move. Otherwise, returns the
    /// tuple (Proposed `MotifSet`, ln posterior of propoosal + ln probability we 
    /// propose the reverse move - ln probability we propose the forward move)
    pub fn propose_base_leap<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<(Self, f64)> { 

        const MINMER_MASK: u64 = (1_u64 << ((MIN_BASE * 2) as u64)) - 1;

        let mut new_set = self.derive_set();

        let id = rng.gen_range(0..self.set.len());

        //let threshold = 7_usize;

        //We are very roughly inverting the binomial distribution CDF for p = 3/4 
        //To get either roughly 200 k-mers for each k or all k-mers if we approximate 
        //that there are likely fewer than 200 k-mers. We use the roughest approximation
        //from Gil, Segura, and Temme's 2020 "Asymptotic inversion of the binomial and 
        //negative binomial cumulative distribution functions" here, which is just
        //"pretend the binomial is a normal distribution and move on". 
        
        let length_motif = self.nth_motif(id).len();

        let ratio_check = 200_f64/(self.data_ref.data().seq().number_unique_kmers(length_motif) as f64);

        let threshold = if ratio_check >= 1.0 {length_motif} else{
            (0.75*(length_motif as f64)+(0.75*0.25*(length_motif as f64)).sqrt()*NORMAL_DIST.inverse_cdf(ratio_check)).round() as usize
        };

        let (hamming_counts, kmer_ids) = self.data_ref.data().seq().all_different_kmers_within_hamming(&self.nth_motif(id).best_motif(), threshold);

        //If we don't have any new kmers within the hamming distance, we immediately fail
        if kmer_ids.len() < 1 {return None;}

        let mut num_ks = 0_f64;

        let pick_hamming_vec = hamming_counts.iter().enumerate().map(|(b, &a)| if a > 0 {num_ks+=1.0; (b,1_usize)} else {(b,0)}).collect::<Vec<_>>(); 

        //select_hamming will actually be the hamming_distance - 1
        let select_hamming = pick_hamming_vec.choose_weighted(rng, |i| i.1).expect("returned already if pick_hamming_vec would have been empty").0;

        let pool_k_counts = hamming_counts[select_hamming] as f64;

        let mut begin_slice: usize = 0;
        let mut end_slice: usize = hamming_counts[0];

        if select_hamming > 0 {for i in 1..=select_hamming {
            begin_slice += hamming_counts[i-1];
            end_slice += hamming_counts[i];
        }}



        let &(new_kmer_id, new_ham) = kmer_ids[begin_slice..end_slice].choose(rng).expect("should not be empty");

        //println!("proposed ham {} actual ham {new_ham} {} {:?} {:?}", select_hamming+1, kmer_ids[begin_slice..end_slice].iter().all(|a| a.1 == new_ham), begin_slice.checked_sub(1).map(|i| kmer_ids.get(i)).flatten(), kmer_ids.get(end_slice));

        let add_mot = self.nth_motif(id).scramble_by_id_to_valid(new_kmer_id, false, self.data_ref.data().seq());
 
        //let (alter_hammings, alter_check) = self.data_ref.data().seq().all_different_kmers_within_hamming(&add_mot.best_motif(), threshold);
        let (alter_hammings, _) = self.data_ref.data().seq().all_different_kmers_within_hamming(&add_mot.best_motif(), threshold);

        let alt_ks = alter_hammings.iter().map(|&a| if a > 0 {1_f64} else {0.0}).sum::<f64>();        

        let alt_k_counts = alter_hammings[select_hamming] as f64;

        //println!("alt proposed ham {} actual ham {new_ham} {} {}", select_hamming+1, alter_hammings[select_hamming],alter_check.iter().filter(|a| a.1 == new_ham).count());
        //No need to normalize the slants: the normalization constants cancel each other out
        let select_ratio = -alt_ks.ln()+(-alt_k_counts.ln())-(-num_ks.ln()+(-pool_k_counts.ln()));

        let ln_post = new_set.replace_motif(add_mot, id);

        //Some((new_set, ln_post+(num_alter_kmers.ln() - ((kmer_ids.len() as f64).ln()))))
        Some((new_set, ln_post+select_ratio))


    }

    /// This proposes to change the secondary `Bp`s  of a `Motif` in `self` by swapping.
    /// It first picks a `Motif` uniformly from `self`, then picks `SHUFFLE_BASES` 
    /// positions from that motiif without replacement. From the set of all
    /// pairwise secondary swaps of those positions, we randomly pick one as out proposal.
    /// Despite returning an `Option`, this never returns `[None]`: it always returns
    /// a tuple of the proposed motif followed by its ln posterior density
    pub fn propose_secondary_shuffle<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<(Self, f64)> {

        let mut new_set = self.derive_set();

        let id = rng.gen_range(0..self.set.len());

        let pick_elements = (0..self.nth_motif(id).len()).choose_multiple(rng, SHUFFLE_BASES);

        let mut mots = self.nth_motif(id).scramble_secondaries(&pick_elements);

        let add_mot = mots.swap_remove(rng.gen_range(0..mots.len()));

        let ln_post = new_set.replace_motif(add_mot, id);


        Some((new_set, ln_post))

    }
    /// This proposes to change the kernel shape and size of a `Motif` in `self`
    /// by uniformly smapling the possible kernel shapes and widths. This returns 
    /// `[None]` if we propose to swap to the same exact Kernel. Otherwise, it 
    /// returns a tuple of the proposed motif followed by its ln posterior density
    pub fn propose_kernel_swap<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<(Self, f64)> {

        let mut new_set = self.derive_set();

        let id = rng.gen_range(0..self.set.len());


        let kernel_width: KernelWidth = rng.gen();
        let kernel_variety: KernelVariety = rng.gen();

        if kernel_width == self.set[id].0.kernel_width && kernel_variety == self.set[id].0.kernel_variety {
            return None;
        }

        let mut new_mot = self.get_nth_motif(id);

        new_mot.kernel_width = kernel_width;
        new_mot.kernel_variety = kernel_variety;

        let ln_post = new_set.replace_motif(new_mot, id);

        Some((new_set, ln_post))

    }
    //MOVE TO CALL
    fn base_leap<R: Rng + ?Sized>(&self, thermo_beta: f64, rng: &mut R) -> Self {


        //We want to shuffle this randomly, in case there is some kind of codependency between particular TFs
        let mut ids: Vec<usize> = (0..self.set.len()).collect();
        let id = *ids.choose(rng).unwrap();

        let mut current_set = self.clone();

        //for id in ids {

            let current_mot = current_set.nth_motif(id).clone();

            let mut base_set = current_set.clone();
            base_set.remove_motif_void(id);
            //This was numerically derived, and not a hard rule. I wanted less than 150 kmers per leap
            //Numerically, the answers I got when I tried to solve the equation 
            //(n choose k)*4^(k-n)*sequence.len()= 300 looked a lot like
            //k = n/2+2-(0.5*sequence.len().ln()).ceil(), where k is hamming distance of a current n-mer
            //I don't know why. This is strictly a numerical analysis.
            //If YOU have a way to invert (n choose k)*4^(k-n), be my guest. 
            let threshold = if current_mot.len() < 10 {2} else { current_mot.len()/2-2}; 

            let kmer_ids = self.data_ref.data().seq().all_kmers_within_hamming(&current_mot.best_motif(), threshold+1);

            let ids_cartesian_bools = if PROB_POS_PEAK == 1.0 {
                kmer_ids.into_iter().map(|k| (k, false)).collect::<Vec<_>>()
            } else {
                kmer_ids.into_iter().flat_map(|k| [(k, true), (k, false)]).collect::<Vec<_>>()
            };

            let likes_and_mots: Vec<(f64, Self)> = ids_cartesian_bools.clone().into_par_iter().map(|a| {
                //let likes_and_mots: Vec<(f64, Self)> = ids_cartesian_bools.clone().into_iter().map(|a| {//}
                let mut to_add = base_set.clone();
                let add_mot = current_mot.scramble_by_id_to_valid(a.0, a.1, self.data_ref.data().seq());
                //The REAL ln_posterior is different from the ln_posterior we use to select a motif, because that needs a thermodynamic beta to alter it
                let lnlike = to_add.insert_motif(add_mot, id)*thermo_beta;
                (lnlike, to_add)
            }).collect();

            //We want to pick these based on their relative ln posteriors
            //But these are going to be small. We normalize based on the max
            //ln likelihood because it prevents errors from infinities

            let mut selection_probs: Vec<f64> = vec![0.0; likes_and_mots.len()];

            let normalize_ln_like: f64 = likes_and_mots.iter().map(|(a, _)| a).fold(-f64::INFINITY, |a, &b| a.max(b)) ;

            for i in 0..selection_probs.len() {
                //This subtraction might seem technically unnecessary, but
                //but computers are not infinitely precise. We want to 
                //ensure that we minimize numerical issues

                selection_probs[i] = (likes_and_mots[i].0-normalize_ln_like).exp().abs();
            }

            let dist = match WeightedIndex::new(&selection_probs) {

                Ok(weights) => weights, 
                Err(_) => {warn!("Issue with base leap weights in this step. Randomly selecting. Discard inference before this point {:?}", selection_probs);
                    WeightedIndex::new(&(vec![1.0; likes_and_mots.len()])).expect("This is statically valid")},
            };

            current_set = likes_and_mots[dist.sample(rng)].1.clone();
        //}

        current_set

    }
     
    fn secondary_shuffle<R: Rng + ?Sized>(&self, thermo_beta: f64, rng: &mut R) -> Self {


        //We want to shuffle this randomly, in case there is some kind of codependency between particular TFs
        let mut ids: Vec<usize> = (0..self.set.len()).collect();
        //ids.shuffle(rng);

        let id = *ids.choose(rng).unwrap();


        let mut current_set = self.clone();

        //for id in ids {

            let current_mot = current_set.get_nth_motif(id).clone();

            let mut base_set = current_set.clone();
            base_set.remove_motif_void(id);
          
            let pick_elements = (0..current_mot.len()).choose_multiple(rng, SHUFFLE_BASES);

            let mots = current_mot.scramble_secondaries(&pick_elements);

            let likes_and_mots: Vec<(f64, Self)> = mots.into_par_iter().map(|a| {
                let mut to_add = base_set.clone();
                let lnlike = to_add.insert_motif(a, id)*thermo_beta;
                (lnlike, to_add)
            }).collect();

            //We want to pick these based on their relative ln posteriors
            //But these are going to be small. We normalize based on the max
            //ln likelihood because it prevents errors from infinities

            let mut selection_probs: Vec<f64> = vec![0.0; likes_and_mots.len()];

            let normalize_ln_like: f64 = likes_and_mots.iter().map(|(a, _)| a).fold(-f64::INFINITY, |a, &b| a.max(b)) ;

            for i in 0..selection_probs.len() {
                //This subtraction might seem technically unnecessary, but
                //but computers are not infinitely precise. We want to 
                //ensure that we minimize numerical issues

                selection_probs[i] = (likes_and_mots[i].0-normalize_ln_like).exp().abs();
            }

            let dist = match WeightedIndex::new(&selection_probs) {

                Ok(weights) => weights, 
                Err(_) => {warn!("Issue with base leap weights in this step. Randomly selecting. Discard inference before this point {:?}", selection_probs);
                    WeightedIndex::new(&(vec![1.0; likes_and_mots.len()])).expect("This is statically valid")},
            };

            current_set = likes_and_mots[dist.sample(rng)].1.clone();
        //}

        current_set

    }
    
    
    fn randomize_motifs<R: Rng + ?Sized>(&self, thermo_beta: f64, rng: &mut R) -> Self {


        //We want to shuffle this randomly, in case there is some kind of codependency between particular TFs
        let mut ids: Vec<usize> = (0..self.set.len()).collect();
        //ids.shuffle(rng);

        let id = *ids.choose(rng).unwrap();

        let mut current_set = self.clone();

        //for id in ids {

            let current_mot = current_set.get_nth_motif(id).clone();

            let mut base_set = current_set.clone();
            base_set.remove_motif_void(id);

            let mut kmer_ids = self.data_ref.data().seq().n_random_valid_motifs(current_mot.len(), 50, rng);

            let mut likes_and_mots: Vec<(f64, Self)> = kmer_ids.into_par_iter().map(|a| {
                let mut to_add = base_set.clone();
                let mut rng = rand::thread_rng();
                let add_mot = Motif::from_motif_with_height(self.data_ref.data().seq().idth_unique_kmer_vec(current_mot.len(), a), current_mot.peak_height(), &mut rng); 
                //The REAL ln_posterior is different from the ln_posterior we use to select a motif, because that needs a thermodynamic beta to alter it
                let lnlike = to_add.insert_motif(add_mot, id)*thermo_beta;
                (lnlike, to_add)
            }).collect();

            let mut current_clone = current_set.clone();

            let current_like = current_clone.ln_posterior();

            likes_and_mots.push((current_like*thermo_beta, current_clone));

            //We want to pick these based on their relative ln posteriors
            //But these are going to be small. We normalize based on the max
            //ln likelihood because it prevents errors from infinities

            let mut selection_probs: Vec<f64> = vec![0.0; likes_and_mots.len()];

            let normalize_ln_like: f64 = likes_and_mots.iter().map(|(a, _)| a).fold(-f64::INFINITY, |a, &b| a.max(b)) ;

            for i in 0..selection_probs.len() {
                //This subtraction might seem technically unnecessary, but
                //but computers are not infinitely precise. We want to 
                //ensure that we minimize numerical issues

                selection_probs[i] = (likes_and_mots[i].0-normalize_ln_like).exp().abs();
            }

            let dist = match WeightedIndex::new(&selection_probs) {

                Ok(weights) => weights, 
                Err(_) => {warn!("Issue with base leap weights in this step. Randomly selecting. Discard inference before this point {:?}", selection_probs);
                    WeightedIndex::new(&(vec![1.0; likes_and_mots.len()])).expect("This is statically valid")},
            };

            current_set = likes_and_mots[dist.sample(rng)].1.clone();
        //}

        current_set

    }

    /*fn propose_ordered_base_move<R: Rng + ?Sized>(&self, rng: &mut R ) -> Option<(Self, f64)> {

        let mut new_set = self.derive_set();

        let c_id = rng.gen_range(0..self.set.len());

        let mut replacement = new_set.set[c_id].clone();

        let base_change = rng.gen_range(0..replacement.pwm.len());

        let Some(attempt_new) = replacement.pwm[base_change].moved_base(L_SD_VECTOR_SPACE_SINGLE, SD_LEAST_MOVE_SINGLE, rng) else { return None;};

        replacement.pwm[base_change] = attempt_new;

        let ln_post = new_set.replace_motif(replacement, c_id);

        Some((new_set, ln_post))
    }

    fn propose_ordered_motif_move<R: Rng + ?Sized>(&self, rng: &mut R ) -> Option<(Self, f64)> {

        let mut new_set = self.derive_set();

        let c_id = rng.gen_range(0..self.set.len());

        let mut replacement = new_set.set[c_id].clone();

        let Some(attempt_new) = replacement.pwm.iter().map(|a| a.moved_base(L_SD_VECTOR_SPACE, SD_LEAST_MOVE, rng)).collect::<Option<Vec<Base>>>() else {return None;};

        replacement.pwm = attempt_new;

        let ln_post = new_set.replace_motif(replacement, c_id);

        Some((new_set, ln_post))
    }


    fn propose_height_move<R: Rng + ?Sized>(&self, rng: &mut R ) -> Option<(Self, f64)> {

        let mut new_set = self.derive_set();

        let c_id = rng.gen_range(0..self.set.len());

        let mut replacement = new_set.set[c_id].clone();

        replacement.peak_height += NORMAL_DIST.sample(rng)*HEIGHT_MOVE_SD;

        replacement.peak_height = replacement.peak_height.signum()*reflect_abs_height(replacement.peak_height.abs());

        let ln_post = new_set.replace_motif(replacement, c_id);

        Some((new_set, ln_post))

    }*/

    /// This proposes to change one of the penalties of one position 
    /// of one `Motif` in `self` by simply scrambling it to another 
    /// possible penalty. This never returns `[None]`: it always returns
    /// a tuple of the proposed motif followed by its ln posterior density
    pub fn propose_rook_move<R: Rng + ?Sized>(&self, rng: &mut R)  -> Option<(Self, f64)> {

        let mut new_set = self.derive_set();

        let c_id = rng.gen_range(0..self.set.len());

        let mut replacement = new_set.nth_motif(c_id).clone();

        let base_change = rng.gen_range(0..replacement.pwm.len());

        let attempt_new = replacement.pwm[base_change].rook_move(rng);

        replacement.pwm[base_change] = attempt_new;

        let ln_post = new_set.replace_motif(replacement, c_id);

        Some((new_set, ln_post))


    }
/*
    pub fn propose_ordered_base_move_custom<R: Rng + ?Sized>(&self, rng: &mut R , ratio_sd: f64, linear_sd: f64) -> Option<(Self, f64)> {

        let mut new_set = self.derive_set();

        let c_id = rng.gen_range(0..self.set.len());

        let mut replacement = new_set.nth_motif(c_id).clone();

        let base_change = rng.gen_range(0..replacement.pwm.len());

        let Some(attempt_new) = replacement.pwm[base_change].moved_base(ratio_sd, linear_sd, rng) else { return None;};

        replacement.pwm[base_change] = attempt_new;

        let ln_post = new_set.replace_motif(replacement, c_id);

        Some((new_set, ln_post))
    }

    pub fn propose_ordered_motif_move_custom<R: Rng + ?Sized>(&self, rng: &mut R, ratio_sd: f64, linear_sd: f64) -> Option<(Self, f64)> {


        let mut new_set = self.derive_set();

        let c_id = rng.gen_range(0..self.set.len());

        let mut replacement = new_set.nth_motif(c_id).clone();
        
        //let scaler = REDUCE_MOTIF_SCALE_MOVE[self.nth_motif(c_id).len()-MIN_BASE]*0.25;

        let scaler = (1.0/(self.nth_motif(c_id).len().pow(2) as f64))/4096.;

        let Some(attempt_new) = replacement.pwm.iter().map(|a| a.moved_base(ratio_sd*scaler, linear_sd*scaler, rng)).collect::<Option<Vec<Base>>>() else {return None;};

        replacement.pwm = attempt_new;

        let ln_post = new_set.replace_motif(replacement, c_id);

        Some((new_set, ln_post))
    }*/


    /// This proposes to change the peak height of one `Motif` in `self` by adding 
    /// a normally distributed variable, with reflection. This never returns 
    /// `[None]`: it always returns a tuple of the proposed motif followed by 
    /// its ln posterior density
    pub fn propose_height_move_custom<R: Rng + ?Sized>(&self, rng: &mut R, height_sd: f64 ) -> Option<(Self, f64)> {

        let mut new_set = self.derive_set();

        let c_id = rng.gen_range(0..self.set.len());

        let mut replacement = new_set.nth_motif(c_id).clone();

        replacement.peak_height += NORMAL_DIST.sample(rng)*height_sd;

        replacement.peak_height = replacement.peak_height.signum()*reflect_abs_height(replacement.peak_height.abs(), self.data_ref.min_height());

        let ln_post = new_set.replace_motif(replacement, c_id);

        Some((new_set, ln_post))

    }



    /// This tries to crudely optimize the height of the `c_id`th `Motif`
    /// by minimizing the Euclidean distance between it and the data trace.
    /// It is not appropriate for inferential purposes: it is very crude. 
    /// Returns `[None]` if `c_id` is not less than the number of `Motif`s 
    /// in `self`. Otherwise, returns a tuple of `MotifSet` with the `c_id`th
    /// height optimized, followed by the Euclidean distance. 
    pub fn best_height_set_from_rmse(&self, c_id: usize) -> Option<(Self, f64)> {

        if c_id >= self.set.len() { return None; }

        let height_len: usize = ((MAX_HEIGHT-self.data_ref.min_height())/0.5) as usize + 1;

        let heights: Vec<f64> = (0..height_len).map(|i| self.data_ref.min_height()+0.5*(i as f64)).collect();

        let mut try_set = self.derive_set();

        let mut replacement = self.nth_motif(c_id).clone();

        let mut best: Option<(usize, f64)> = None;

        let mut id: usize = 0;

        for height in &heights {

            replacement.peak_height = *height;

            _ = try_set.replace_motif(replacement.clone(), c_id);

            let rmse = try_set.signal.rmse_with_wave(self.data_ref.data());

            if best == None {
                best = Some((id, rmse));
            } else {

                let unwrappable = best.clone();
                let check = unwrappable.unwrap().1;

                if rmse <= check {
                    best = Some((id, rmse));
                }
            }

            id += 1;

        }

        let (id, rmse) = best.unwrap();

        replacement.peak_height = heights[id];
        _ = try_set.replace_motif(replacement.clone(), c_id);

        Some((try_set, rmse))

    }

    /// This tries to crudely optimize the height of the `c_id`th `Motif`
    /// by minimizing the Euclidean distance between it and the data trace,
    /// accounting for matches in the `NullSequence` of `self.data_ref()`.
    /// It is not appropriate for inferential purposes: it is very crude. 
    /// Returns `[None]` if `c_id` is not less than the number of `Motif`s 
    /// in `self`. Otherwise, returns a tuple of `MotifSet` with the `c_id`th
    /// height optimized, followed by the Euclidean distance. 
    pub fn best_height_set_from_rmse_noise(&self, c_id: usize) -> Option<(Self, f64)> {

        if c_id >= self.set.len() { return None; }

        let height_len: usize = ((MAX_HEIGHT-self.data_ref.min_height())/0.5) as usize + 1;

        let heights: Vec<f64> = (0..height_len).map(|i| self.data_ref.min_height()+0.5*(i as f64)).collect();

        let mut try_set = self.derive_set();

        let mut replacement = self.nth_motif(c_id).clone();

        let mut best: Option<(usize, f64)> = None;

        let mut id: usize = 0;

        for height in &heights {

            replacement.peak_height = *height;

            let like = try_set.replace_motif_entire(replacement.clone(), c_id);

            let rmse = try_set.magnitude_signal_with_noise();

            let rmse2 = try_set.signal.rmse_with_wave(self.data_ref.data());

            println!("{height} {like} {rmse} {rmse2}");

            if best == None {
                best = Some((id, rmse));
            } else {

                let unwrappable = best.clone();
                let check = unwrappable.unwrap().1;

                if rmse <= check {
                    best = Some((id, rmse));
                }
            }

            id += 1;

        }

        let (id, rmse) = best.unwrap();

        replacement.peak_height = heights[id];
        _ = try_set.replace_motif(replacement.clone(), c_id);

        Some((try_set, rmse))

    }

    fn measure_movement(&mut self, proposal: &mut Self) -> [f64; 4] {

        let rmse = self.signal.rmse_with_wave(&proposal.signal);
        let likelihood_diff = proposal.ln_posterior()-self.ln_posterior();
        if self.set.len() != proposal.set.len() {
            return [rmse, likelihood_diff, 0.0, 0.0];
        }


        let mut height_dists_sq = 0_f64;
        let mut pwm_dists_sq = 0_f64;

        for i in 0..self.set.len() {
            pwm_dists_sq += self.nth_motif(i).distance_function(proposal.nth_motif(i)).0.powi(2); 
            height_dists_sq += (proposal.nth_motif(i).peak_height-self.nth_motif(i).peak_height).powi(2);
        }
        
        [rmse, likelihood_diff, height_dists_sq.sqrt(), pwm_dists_sq.sqrt()]
    }

    pub fn ref_signal(&self) -> &Waveform {
        &self.signal
    }

    pub fn signal_rmse(&self) -> f64 {
        self.signal.rmse_with_wave(self.data_ref.data())
    }

    /// This performs a lasso based on the ln likelihood
    /// and the lambda you supply on this set. The taxicab metric
    /// is the sum of the maximum peak heights for all motifs in the set
    /// Motif order in this set is guaranteed to be returned in order of 
    /// greatest impact on the likelihood at each step, and we always return
    /// a motif set with at least one motif. 
    /// 
    /// The vector of floats we return is the vector of lasso'd likelihoods 
    /// after each step, and it will always have a length equal to the number 
    /// of motifs in `[MotifSet]` we return. The `[Option]<[f64]>` we return is either 
    /// `Some<the_first_failing_lasso_likelihood>` if there are more motifs
    /// to potentially add, or `None` if we added all motifs in the set.
    ///
    /// For example, suppose you had a `[MotifSet]` with four motifs, numbered
    /// 0 to 3. In the first step of the lasso, motif 2 has the greatest likelihood
    /// after subtracting manhattan distance. After accounting for motif 2, then motif 0
    /// has the best improvement. After that, neither motif 1 nor motif 3 improve the
    /// likelihood enough to overcome the taxicab metric penalty, so we return the
    /// `[MotifSet]` with motif 2, then motif 0. 
    ///
    /// This function will exploit parallelization if available
    pub fn lasso_self(&self, lambda: f64) -> (Self, Vec<f64>, Option<f64>) {

        //It is likely an invariant that we have at least two motifs in a motif set.
        //But I'm defending against an empty motif set in case
        if self.set.len() < 2 {
            let likelihood = self.calc_ln_post()-self.ln_prior();
            let lassoed_like = likelihood-lambda*self.set.get(0).map(|a| a.0.peak_height.abs()).unwrap_or(0.0);
            return (self.clone(), vec![lassoed_like], None);
        }

        let mut lasso_likes: Vec<f64> = Vec::new();

        let mut single_sets_and_lasso_likes: Vec<(MotifSet, f64)> = (0..self.set.len()).into_par_iter().map(|i| {

            let single_set = self.nth_motif_as_set(i);

            let likelihood = single_set.calc_ln_post()-single_set.ln_prior();
            let lassoed_like = likelihood-lambda*single_set.set[0].0.peak_height.abs();
            (single_set, lassoed_like)
        }).collect();

        //TODO: I have the vector of single sets with lassod likes. Find the best one, 
        //      remove it, and use it start my returns
        let mut target_index: usize = 0;
        let mut target_lasso: f64 = single_sets_and_lasso_likes[0].1;
      
        // We already know we have at least two motifs, since we returned
        // already if we had less than that in self
        for index in 1..single_sets_and_lasso_likes.len() {
            if single_sets_and_lasso_likes[index].1 >= target_lasso {
                target_index = index;
                target_lasso = single_sets_and_lasso_likes[index].1;
            }
        }

        println!("single sets {:?}", single_sets_and_lasso_likes);
        let (mut motif_set, _ ) = single_sets_and_lasso_likes.swap_remove(target_index);
        println!("single sets after remove {target_index} {:?}", single_sets_and_lasso_likes);
        
        //target_lasso is a float, so it is still alive after this
        lasso_likes.push(target_lasso);

        //TODO: ITERATE over the remaining motifs by:
        //      -Finding new best lassoed likelihood
        //      -Checking it against the previous lassoed likelihood. 
        //      -If it's better, add it to the set of motifs and reiterate
        //      -If it's worse, set my last Option to the lassoed likelihood and return 
        //       the sete I have. 

        while single_sets_and_lasso_likes.len() > 0 {

            let combined_sets_and_likes : Vec<_> = single_sets_and_lasso_likes.par_iter().map(|set_and_lass| {
                let check_set = motif_set.combine_motif_sets(&set_and_lass.0);
                let likelihood = check_set.calc_ln_post()-check_set.ln_prior();
                let lassoed_like = likelihood-lambda*set_and_lass.0.set[0].0.peak_height.abs();
                (check_set, lassoed_like)
            }).collect();

            let mut target_index: Option<usize> = None;
            //target_lasso is already in scope
            let mut potential_lasso = combined_sets_and_likes[0].1;
            for index in 0..combined_sets_and_likes.len() {
                potential_lasso = potential_lasso.max(combined_sets_and_likes[index].1);
                if combined_sets_and_likes[index].1 >= target_lasso {
                    target_index = Some(index);
                    target_lasso = combined_sets_and_likes[index].1;
                }
            }

            match target_index {
                None => return (motif_set, lasso_likes, Some(potential_lasso)),
                Some(index) => {
                    println!("single sets {:?}", single_sets_and_lasso_likes);
                    (motif_set, _) = combined_sets_and_likes[index].clone();
                    _ = single_sets_and_lasso_likes.swap_remove(index);
                    println!("single sets after remove {index} {:?}", single_sets_and_lasso_likes);
                    lasso_likes.push(target_lasso);
                },
            }

            
        }
        
        //If I get to the end without hitting a worse lasso like, return the whole thing with the last option None.
        (motif_set, lasso_likes, None)

    }
    
    /// This performs a lasso based on the ln likelihood
    /// and the lambda you supply on this set. The taxicab metric
    /// is the sum of the maximum peak heights for all motifs in the set
    /// Motif order in this set is guaranteed to be returned in order of 
    /// greatest impact on the likelihood at each step, and `self` in
    /// the end will always have at least one Motif
    /// 
    /// The vector of floats we return is the vector of lasso'd likelihoods 
    /// after each step, and it will always have a length equal to the number 
    /// of motifs in `[MotifSet]` we return. The `[Option]<[f64]>` we return is either 
    /// `Some<the_first_failing_lasso_likelihood>` if there are more motifs
    /// to potentially add, or `None` if we added all motifs in the set. 
    ///
    /// The Option<Vec<Vec<String>>> is `None` if annotations is `None`. Otherwise,
    /// it is a vector of `self.motif_num()` of vectors of `String`s, where the `i`th
    /// vector is a vector of gene names that the `i`th motif could be regulating.
    ///
    /// The Option<Vec<Vec<(u64, usize)>>> is `None` if annotations is `None`.
    /// Otherwise, it is a vector of `self.motif_num()` of vectors of `(u64, usize)`
    /// tuples, where the `i`th vector is a list of `u64`s for go terms and `usize`s 
    /// is the number of genes relevant to that go term that the ith motif might regulate
    ///
    /// For example, suppose you had a `[MotifSet]` with four motifs, numbered
    /// 0 to 3. In the first step of the lasso, motif 2 has the greatest likelihood
    /// after subtracting manhattan distance. After accounting for motif 2, then motif 0
    /// has the best improvement. After that, neither motif 1 nor motif 3 improve the
    /// likelihood enough to overcome the taxicab metric penalty, so self will be a 
    /// `[MotifSet]` with motif 2, then motif 0. 
    ///
    /// This function will exploit parallelization if available
    pub fn sort_self_by_lasso_and_yield_genes_and_go_terms(&mut self, lambda: f64, regulatory_distance: u64, annotations: Option<&GenomeAnnotations>) -> (Vec<f64>, Option<f64>, Option<Vec<Vec<String>>>, Option<Vec<Vec<(u64, usize)>>>) {

        let mut gene_names: Option<Vec<Vec<String>>> = None; //annotations.map(|_| vec![]) ;
        let mut go_term_counts: Option<Vec<Vec<(u64, usize)>>> = None; //annotations.map(|_| vec![]);

        //It is likely an invariant that we have at least two motifs in a motif set.
        //But I'm defending against an empty motif set in case
        /*if self.set.len() < 2 {
            let likelihood = self.calc_ln_post()-self.ln_prior();
            let lassoed_like = likelihood-lambda*self.set.get(0).map(|a| a.0.peak_height.abs()).unwrap_or(0.0);
            let wave = self.nth_motif(0).generate_waveform(self.data_ref);
            if let Some(annotations) = annotations {
                let loci = wave.return_regulated_loci(None, regulatory_distance, self.data_ref.zero_locs(),annotations);
                gene_names = if let Ok(ref locus_vec) = loci { Some(vec![locus_vec.iter().map(|a| a.name_clone()).collect()])} else {None};
                go_term_counts = match  wave.return_go_terms(None, regulatory_distance, self.data_ref.zero_locs(),annotations) {
                    Ok(go) => Some(vec![go]),
                    Err(_) => None,
                };
            }
            return (vec![lassoed_like], None, gene_names, go_term_counts);
        }

        let mut lasso_likes: Vec<f64> = Vec::new();
*/
//lasso_self(&self, lambda: f64) -> (Self, Vec<f64>, Option<f64>)
        let (lassoed_self, lasso_likes, potential_leftover) = self.lasso_self(lambda);

        if let Some(annotations) = annotations {

            for i in 0..self.set.len() {
                let wave = self.nth_motif(i).generate_waveform(self.data_ref);
                let loci = wave.return_regulated_loci(None, regulatory_distance, self.data_ref.zero_locs(),annotations);
                let collect_names: Vec<String> = if let Ok(ref locus_vec) = loci { locus_vec.iter().map(|a| a.name_clone()).collect() } else {vec![]};
                if let Some(ref mut names) = gene_names.as_mut() { names.push(collect_names);} else { gene_names = Some(vec![collect_names]);};
                let go_terms = wave.return_go_terms(None, regulatory_distance, self.data_ref.zero_locs(),annotations).unwrap_or_else(|_| vec![]);
                if let Some(ref mut gos) = go_term_counts { gos.push(go_terms)} else {go_term_counts = Some(vec![go_terms])}; 
            }

        };

        *self=lassoed_self;
        (lasso_likes, potential_leftover, gene_names, go_term_counts)

        /*let mut single_sets_and_lasso_likes: Vec<(MotifSet, f64, Option<Vec<String>>, Option<Vec<(u64, usize)>>)> = (0..self.set.len()).into_par_iter().map(|i| {

            let single_set = self.nth_motif_as_set(i);

            let likelihood = single_set.calc_ln_post()-single_set.ln_prior();
            let lassoed_like = likelihood-lambda*single_set.set[0].0.peak_height.abs();
            let mut motif_gene_names: Option<Vec<String>> = None;
            let mut motif_go_terms: Option<Vec<(u64, usize)>> = None;
            if let Some(annotations) = annotations {
                let loci = single_set.ref_signal().return_regulated_loci(None, regulatory_distance, self.data_ref.zero_locs(),annotations);
                motif_gene_names = if let Ok(ref locus_vec) = loci { Some(locus_vec.iter().map(|a| a.name_clone()).collect())} else {None};
                motif_go_terms = single_set.ref_signal().return_go_terms(None, regulatory_distance, self.data_ref.zero_locs(),annotations).ok();
            }
            (single_set, lassoed_like, motif_gene_names, motif_go_terms)
        }).collect();

        //TODO: I have the vector of single sets with lassod likes. Find the best one, 
        //      remove it, and use it start my returns
        let mut target_index: usize = 0;
        let mut target_lasso: f64 = single_sets_and_lasso_likes[0].1;
      
        // We already know we have at least two motifs, since we returned
        // already if we had less than that in self
        for index in 1..single_sets_and_lasso_likes.len() {
            if single_sets_and_lasso_likes[index].1 >= target_lasso {
                target_index = index;
                target_lasso = single_sets_and_lasso_likes[index].1;
            }
        }

        let (mut motif_set, _, mut motif_gene_names, mut motif_go_terms ) = single_sets_and_lasso_likes.swap_remove(target_index);
       
        gene_names.as_mut().map(|a| a.push( motif_gene_names.unwrap_or_else(|| vec![])));
        go_term_counts.as_mut().map(|a| a.push( motif_go_terms.unwrap_or_else(|| vec![])));

        //target_lasso is a float, so it is still alive after this
        lasso_likes.push(target_lasso);

        //TODO: ITERATE over the remaining motifs by:
        //      -Finding new best lassoed likelihood
        //      -Checking it against the previous lassoed likelihood. 
        //      -If it's better, add it to the set of motifs and reiterate
        //      -If it's worse, set my last Option to the lassoed likelihood and return 
        //       the sete I have. 

        while single_sets_and_lasso_likes.len() > 0 {

            single_sets_and_lasso_likes = single_sets_and_lasso_likes.into_par_iter().map(|set_and_lass| {
                let check_set = motif_set.combine_motif_sets(&set_and_lass.0);
                let likelihood = check_set.calc_ln_post()-check_set.ln_prior();
                let lassoed_like = likelihood-lambda*check_set.set[0].0.peak_height.abs();
                (check_set, lassoed_like, set_and_lass.2, set_and_lass.3)
            }).collect();

            let mut target_index: Option<usize> = None;
            //target_lasso is already in scope
            let mut potential_lasso = single_sets_and_lasso_likes[0].1;
            for index in 0..single_sets_and_lasso_likes.len() {
                potential_lasso = potential_lasso.max(single_sets_and_lasso_likes[index].1);
                if single_sets_and_lasso_likes[index].1 >= target_lasso {
                    target_index = Some(index);
                    target_lasso = single_sets_and_lasso_likes[index].1;
                }
            }

            match target_index {
                None => {
                    *self =  motif_set;
                    return (lasso_likes, Some(potential_lasso), gene_names, go_term_counts);
                },
                Some(index) => {
                    (motif_set, _, motif_gene_names, motif_go_terms) = single_sets_and_lasso_likes.swap_remove(index);
                    lasso_likes.push(target_lasso);
                    gene_names.as_mut().map(|a| a.push( motif_gene_names.unwrap_or_else(|| vec![])));
                    go_term_counts.as_mut().map(|a| a.push( motif_go_terms.unwrap_or_else(|| vec![])));
                },
            };

            
        } */ 

        //If I get to the end without hitting a worse lasso like, return the whole thing with the last option None.
        //(lasso_likes, None, gene_names, go_term_counts)
    }


}

/// An I/O error that is returned when a .meme file contains an invalid motif set
pub enum MemeParseError {
    EmptyMatrix,
    InvalidAlphabet{ motif_num: usize},
    InvalidMotifLength{ motif_num: usize, captured_length: usize },
    ColumnLengthFailure{ line_num: usize },
    FloatParseFailure{ line_num: usize },
    GivesEmptySet,
}

impl fmt::Display for MemeParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemeParseError::EmptyMatrix =>  write!(f, "This file has no MEME matrix input!"),
            MemeParseError::InvalidAlphabet{ motif_num } => write!(f, "The MEME matrix in position {motif_num} has a number of bps which does not match the proper {BASE_L}"),
            MemeParseError::InvalidMotifLength{ motif_num, captured_length } => write!(f, "The MEME matrix in position {motif_num} is of length {captured_length}, which is not in the range [{MIN_BASE}, {MAX_BASE}]"),
            MemeParseError::ColumnLengthFailure{ line_num } => write!(f, "The items on line {line_num} are fewer than the claimed number of columns from the base pair alphabet!"),
            MemeParseError::FloatParseFailure{ line_num } => write!(f, "The items on line {line_num} do not parse to floats, even though they should be part of a matrix!"),
            MemeParseError::GivesEmptySet => write!(f, "With this setting, you get any empty set of motifs!")
        }
    }
}
impl Debug for MemeParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self) }
}
impl Error for MemeParseError {}

impl Debug for MotifSet<'_> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, " ln_post: {:?}, \n set: {:#?}", self.ln_post, self.set)
    }
}

#[derive(Debug, Clone, Copy)]
pub enum HandleImpossibleMotif {
    MakePossible,
    OmitFromSet,
    LeaveUnchanged,
}

/// This is a MotifSet, freed from data that it was inferred from and its 
/// occupancy trace. It is mainly just a `Vec<[Motif]>`, but it also saves its
/// ln unnormalized posterior density. Note that there are no checks that the 
/// saved density is correct. When this is created from a `[MotifSet]`, the 
/// density is correct then, but this can be created from an entirely different
/// `[AllDataUse]` than what you're trying to run on, and can be mutated. 
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrippedMotifSet {
    set: Vec<Motif>,
    ln_post: f64,
}

impl StrippedMotifSet {

   
    /// This revives a `[StrippedMotifSet]` into a form suitable for inference. 
    /// Note that while `[StrippedMotifSet]` saved an unnormalized posterior density, 
    /// we don't actually believe it when we revive it. It's only for recording
    pub fn reactivate_set<'a>(&self, data_ref: &'a AllDataUse) -> MotifSet<'a> {

        let cutoff = data_ref.offset()*2.0;

        let mut revived = MotifSet {

            set: self.set.clone().into_iter().map(|a| {
                let nulls = a.return_any_null_binds_by_hamming(data_ref.null_seq(), data_ref.min_height(), cutoff);
                (a, nulls)
            }).collect(),
            signal: data_ref.data().derive_zero(),
            ln_post: None,
            data_ref: data_ref,
        };


        revived.recalc_signal();

        _ = revived.ln_posterior();

        revived
    }

    pub fn num_motifs(&self) -> usize { self.set.len() }
    
    /// This sorts `[Motif]`s in the set in descending order of peak height
    pub fn sort_by_height(&mut self) {
        self.set.sort_unstable_by(|a, b| b.peak_height().partial_cmp(&a.peak_height()).unwrap() );
    }

    pub fn motif_num_prior(&self, credibility: f64) -> f64 {
        if self.set.len() > MAX_TF_NUM { return -f64::INFINITY; } ;
        -((self.set.len()-1) as f64)* credibility
    }

    pub fn ln_prior(&self, data_ref: &AllDataUse) -> f64 {
        self.motif_num_prior(data_ref.credibility()) + self.set.iter().map(|a| a.height_prior(data_ref.height_dist())+a.pwm_prior(data_ref.data().seq())).sum::<f64>()
    }

    pub fn bic(&self, data_ref: &AllDataUse) -> f64 {
        //each motif has a height, then each position on the motif has BASE_L 
        //parameters: 1 to determine which is the best base, then 3 for penalties
        let k = (self.set.iter().map(|a| 1+a.len()*BASE_L).sum::<usize>()) as f64;
        let len_dat = data_ref.data().read_wave().len() as f64;
        let ln_like = self.ln_posterior()-self.ln_prior(data_ref);

        k*len_dat.ln()-2.0*ln_like

    }

    pub fn ln_posterior(&self) -> f64 { self.ln_post }

    pub fn ln_likelihood(&self, data_ref: &AllDataUse) -> f64 {
        self.ln_posterior()-self.ln_prior(data_ref)
    }

    pub fn set_iter(&self) -> Iter<'_, Motif> {
        self.set.iter()
    }
    /// If you use this, the ln density is almost definitely no longer valid
    pub fn mutable_set_iter(&mut self) -> IterMut<'_, Motif> {
        self.set.iter_mut()
    }

    pub fn taxicab(&self) -> f64 {
        self.set.iter().map(|a| a.taxicab()).sum::<f64>()
    }


    /// This prints the StrippedMotifSet in descending height order into a meme 
    /// file `{output_dir}/{run_name}_savestate.meme`. The header of each motif
    /// is `Motif motif_{index} {motif_height}`. 
    pub fn output_to_meme(&self, background_dist: Option<[f64; BASE_L]>, output_dir: &str, run_name: &str) -> std::io::Result<String> {

        let savestate_file: String = output_dir.to_owned()+"/"+run_name+"_savestate.meme";

        println!("{savestate_file}");
        let mut outfile_handle = fs::File::create(&savestate_file)?;

        outfile_handle.write(b"MEME Version 4\n\n")?;

        let alphabet = "ALPHABET= ".to_owned()+&BPS.iter().collect::<String>()+"\n";

        outfile_handle.write(&alphabet.into_bytes())?;

        let mut a = self.clone();

        a.sort_by_height();

        for (i, mot) in a.set_iter().enumerate() {
            
            let header_str = format!("MOTIF motif_{i} {}\n", mot.peak_height());
            outfile_handle.write(&header_str.into_bytes())?;

            let body_str = match mot.for_meme(background_dist) {
                Ok(m) => m,
                Err(e) => return Err(std::io::Error::new(std::io::ErrorKind::Other, Box::new(e))),
            };
            outfile_handle.write(&body_str.into_bytes())?;
        }

        Ok(savestate_file)
    }

    /// Generates many fimo runs, one for each `[StrippedMotifSet]` in 
    /// `self.over_additions()`, saving the result of the `i`th element of this
    /// vector to `{output_dir}/{run_name}_{i}_savestate.meme`.
    pub fn generate_ascending_fimo(&self, background_dist: Option<[f64; BASE_L]>, fasta_file: &str, output_dir: &str, run_name: &str) -> Result<(), Box<dyn Error+Send+Sync>> {

        let over_add = self.over_additions();


        let a: Result<Vec<()>, Box<dyn Error+Send+Sync>> = over_add.into_par_iter().enumerate().map(|(i, set)| {

            let run_mod = format!("{}_{}", run_name, i);
            let mut command=set.generate_fimo_command(background_dist, fasta_file, output_dir, &run_mod)?;
            command.status()?;
            Ok(())
        }).collect();


        a?;

        Ok(())

    }

    //Note: do NOT trust the ln posteriors that come out of this. They are explicitly erroneous
    /// This takes `self` and outputs a `Vec<[StrippedMotifSet]>` such that its
    /// `i`th element contains a `[StrippedMotifSet]` with the `i+1` highest 
    /// peaked `[Motif]`s in `self`
    pub fn over_additions(&self) -> Vec<StrippedMotifSet> {

        let mut clone = self.clone();

        clone.sort_by_height();


        (0..clone.num_motifs()).map(|a| StrippedMotifSet { set: clone.set[0..=a].to_vec(), ln_post: self.ln_post}).collect()


    }

    /// This outputs `self` to the meme file `{output_dir}/{run_name}_savestate.meme`,
    /// then runs fimo with an output directory of `{output_dir}/{run_name}_fimo`.
    /// # Errors
    /// If the needed directories or files cannot be generated or written to,
    /// or if the MEME suite is not installed on your machine
    pub fn generate_fimo(&self, background_dist: Option<[f64; BASE_L]>, fasta_file: &str, output_dir: &str, run_name: &str) -> Result<(), Box<dyn Error+Send+Sync>> {

       
        let meme_file = match self.output_to_meme(background_dist, output_dir, run_name) {
            Ok(m) => m,
            Err(e) => {
                println!("couldn't make meme file");
                return Err(Box::new(e))
            },
        };

        let fimo_output = format!("{}/{}_fimo", output_dir, run_name);

        let a = fs::create_dir(&fimo_output);

        println!("fimo output {fimo_output}");
        println!("creating dir {:?}", a);

        let mut bind_command = Command::new("fimo");
        let mut commander = bind_command.arg("--oc").arg(&fimo_output).arg("--bfile").arg("--uniform--").arg(&meme_file).arg(fasta_file);
        
        println!("command: {:?}", commander);
        commander.status()?;


        Ok(())
    }

    /// This outputs `self` to the meme file `{output_dir}/{run_name}_savestate.meme`,
    /// and returns the fimo command for this meme file that has an output directory of `{output_dir}/{run_name}_fimo`.
    pub fn generate_fimo_command(&self, background_dist: Option<[f64; BASE_L]>, fasta_file: &str, output_dir: &str, run_name: &str) -> Result<Command, Box<dyn Error+Send+Sync>> {


        let meme_file = match self.output_to_meme(background_dist, output_dir, run_name) {
            Ok(m) => m,
            Err(e) => return Err(Box::new(e)),
        };

        let fimo_output = format!("{}/{}_fimo", output_dir, run_name);

        let _ = fs::create_dir(&fimo_output);

        let mut bind_command = Command::new("fimo");
        let _ = bind_command.arg("--oc").arg(&fimo_output).arg("--bfile").arg("--uniform--").arg(&meme_file).arg(fasta_file);

        Ok(bind_command)

    }


    /// This uses the Hungarian algorithm to pair `[Motif]`s from `self`
    /// and `reference_set`, using the minimum squared Aitchison metric * 4
    /// as the distance metric. This outputs a vector with length `self.num_motifs()`.
    /// An element of the vector is `None` only if `self.num_motifs() > reference_set.num_motifs()`
    /// and thus that element does not have a match. Otherwise, the elements are
    /// are tuples of the distance followed by a bool which is true if minimizing 
    /// the Aitchison metric required reverse complementing, followed by a reference
    /// to the Motif in `reference_set`
    pub fn pair_to_other_set<'a>(&self, reference_set: &'a StrippedMotifSet) -> Vec<Option<(f64, bool, &'a Motif)>>/*Some type here */ {
    
        let distance_by_index = |(i, j): (usize, usize)| OrderedFloat(self.get_nth_motif(j).distance_function(&reference_set.get_nth_motif(i)).0);
        let rev_comp_by_index = |(i, j): (usize, usize)| self.get_nth_motif(j).distance_function(reference_set.get_nth_motif(i)).1;

        //We want the self to act like the columns, because the positions of the reference set should be pulled
        let check_matrix = matrix::Matrix::from_fn(reference_set.num_motifs(), self.num_motifs(), distance_by_index);
        let rev_c_matrix = matrix::Matrix::from_fn(reference_set.num_motifs(), self.num_motifs(), rev_comp_by_index);

        if self.num_motifs() >= reference_set.num_motifs() {
            let mut matches: Vec<Option<(f64, bool, &Motif)>> = vec![None; self.num_motifs()];
            let (_, pre_matches) = kuhn_munkres::kuhn_munkres_min(&check_matrix);
            for (i, j) in pre_matches.into_iter().enumerate() {
                matches[j] = Some((check_matrix.get((i, j)).unwrap().into_inner(), *rev_c_matrix.get((i,j)).unwrap(), reference_set.get_nth_motif(i)));
            }
            matches
        } else {
            let transpose = check_matrix.transposed();
            let (_, pre_matches) = kuhn_munkres::kuhn_munkres_min(&transpose);
            let mut matches: Vec<Option<(f64, bool,&Motif)>> = vec![None; self.num_motifs()];
            for (i, j) in pre_matches.into_iter().enumerate() {
                matches[i] = Some((check_matrix.get((j, i)).unwrap().into_inner(),*rev_c_matrix.get((j,i)).unwrap(), reference_set.get_nth_motif(j)));
            }
            matches
        }
    }

    /// This uses the Hungarian algorithm to pair `[Motif]`s from `self`
    /// and `reference_set`, using the Euclidean distance of the occpuancy
    /// traces on the parts of the sequence with captured binding for
    /// the distance metric. This outputs a vector with length `self.num_motifs()`.
    /// An element of the vector is `None` only if `self.num_motifs() > reference_set.num_motifs()`
    /// and thus that element does not have a match. Otherwise, the elements are
    /// are tuples of the RMSE followed by a reference to the Motif in `reference_set`
    pub fn pair_to_other_set_rmse_data<'a>(&'a self, reference_set: &'a StrippedMotifSet, data_seq: &AllDataUse) -> Vec<Option<(f64, &'a Motif)>>/*Some type here */ {
   
        let self_waves: Vec<Waveform> = self.set_iter().map(|a| {let mut b = a.clone(); b.peak_height = 5.0; b.generate_waveform(&data_seq)}).collect();
        let alts_waves: Vec<Waveform> = reference_set.set_iter().map(|a| {let mut b = a.clone(); b.peak_height = 5.0; b.generate_waveform(&data_seq)}).collect();

        //println!("lens {} {}", self_waves.len(), alts_waves.len());

        let distance_by_index = |(i, j): (usize, usize)| OrderedFloat(self_waves[j].rmse_with_wave(&alts_waves[i]));

        //We want the self to act like the columns, because the positions of the reference set should be pulled
        let check_matrix = matrix::Matrix::from_fn(reference_set.num_motifs(), self.num_motifs(), distance_by_index);

        //println!("Made matrix");

        if self.num_motifs() >= reference_set.num_motifs() {
            //println!("first");
            let mut matches: Vec<Option<(f64, &Motif)>> = vec![None; self.num_motifs()];
            let (_, pre_matches) = kuhn_munkres::kuhn_munkres_min(&check_matrix);
            for (i, j) in pre_matches.into_iter().enumerate() {
              //  println!("{i} {j}");
               // println!("{} {}", self.set.len(), reference_set.set.len());
                matches[j] = Some((check_matrix.get((i, j)).unwrap().into_inner(), reference_set.get_nth_motif(i)));
            }
            matches
            //matches.into_iter().enumerate().map(|(i,x)| Some((check_matrix.get((i,x)).unwrap().into_inner(),reference_set.get_nth_motif(x)))).collect()
        } else {
           // println!("second");
            let transpose = check_matrix.transposed();
            let (_, pre_matches) = kuhn_munkres::kuhn_munkres_min(&transpose);
            let mut matches: Vec<Option<(f64, &Motif)>> = vec![None; self.num_motifs()];
            //println!("{:?}", pre_matches);
            //println!("{:?}", check_matrix);
            for (i, j) in pre_matches.into_iter().enumerate() {
                //println!("{i} {j}");
                //println!("{} {}", self.set.len(), reference_set.set.len());
                
                matches[i] = Some((check_matrix.get((j, i)).unwrap().into_inner(), reference_set.get_nth_motif(j)));
            }
            matches
        }
    }

    /// This uses the Hungarian algorithm to pair `[Motif]`s from `self`
    /// and `reference_set`, using the Euclidean distance of the base 2 exponentiated
    /// binding scores on the parts of the sequence with captured binding for
    /// the distance metric. This outputs a vector with length `self.num_motifs()`.
    /// An element of the vector is `None` only if `self.num_motifs() > reference_set.num_motifs()`
    /// and thus that element does not have a match. Otherwise, the elements are
    /// are tuples of the RMSE followed by a reference to the Motif in `reference_set`
    pub fn pair_to_other_set_binding<'a>(&'a self, reference_set: &'a StrippedMotifSet, data_seq: &AllDataUse) -> Vec<Option<(f64, &'a Motif)>>/*Some type here */ {
   
        let self_waves: Vec<Vec<f64>> = self.set_iter().map(|a| a.return_bind_score(data_seq.data().seq())).collect();
        let alts_waves: Vec<Vec<f64>> = reference_set.set_iter().map(|a| a.return_bind_score(data_seq.data().seq())).collect();



        let distance_by_index = |(i, j): (usize, usize)| OrderedFloat(self_waves[j].iter().zip(alts_waves[i].iter()).map(|(&a, &b)| (a.exp2()-b.exp2()).powi(2)).sum::<f64>().log2());

        //We want the self to act like the columns, because the positions of the reference set should be pulled
        let check_matrix = matrix::Matrix::from_fn(reference_set.num_motifs(), self.num_motifs(), distance_by_index);

        //println!("check matrix {:?}", check_matrix);
        //println!("Made matrix");

        if self.num_motifs() >= reference_set.num_motifs() {
            //println!("first");
            let mut matches: Vec<Option<(f64, &Motif)>> = vec![None; self.num_motifs()];
            let (_, pre_matches) = kuhn_munkres::kuhn_munkres_min(&check_matrix);
            for (i, j) in pre_matches.into_iter().enumerate() {
               // println!("{i} {j}");
               // println!("{} {}", self.set.len(), reference_set.set.len());
                matches[j] = Some((check_matrix.get((i, j)).unwrap().into_inner(), reference_set.get_nth_motif(i)));
            }
            matches
            //matches.into_iter().enumerate().map(|(i,x)| Some((check_matrix.get((i,x)).unwrap().into_inner(),reference_set.get_nth_motif(x)))).collect()
        } else {
          //  println!("second");
            let transpose = check_matrix.transposed();
            let (_, pre_matches) = kuhn_munkres::kuhn_munkres_min(&transpose);
            let mut matches: Vec<Option<(f64, &Motif)>> = vec![None; self.num_motifs()];
            //println!("{:?}", pre_matches);
            //println!("{:?}", check_matrix);
            for (i, j) in pre_matches.into_iter().enumerate() {
              //  println!("{i} {j}");
              //  println!("{} {}", self.set.len(), reference_set.set.len());
                
                matches[i] = Some((check_matrix.get((j, i)).unwrap().into_inner(), reference_set.get_nth_motif(j)));
            }
            matches
        }
    }


    pub fn get_nth_motif(&self, index: usize) -> &Motif {
        &self.set[index]
    }
   
    /// This saves `self`'s occupancy traces in a file called `{output_dir}/{file_name}_new_occupancy`.
    /// See `[MotifSet::save_set_trace_and_sub_traces]()` for more detail. 
    pub fn save_this_trace(&self, data_ref: &AllDataUse, output_dir: &str, file_name: &str, annotations: Option<&GenomeAnnotations>, ontologies: Option<&[&str]>) -> Result<(), Box<dyn Error+Send+Sync>>  {

        let current_active = &self.reactivate_set(data_ref);

        current_active.save_set_trace_and_sub_traces(output_dir, file_name, annotations, ontologies)

    }

}


impl<'a> From<&'a MotifSet<'a>> for StrippedMotifSet {
    fn from(other: &'a MotifSet) -> StrippedMotifSet {
        let ln_post: f64 = other.calc_ln_post();
        StrippedMotifSet {
            set: other.set.iter().map(|a| a.0.clone()).collect(),
            ln_post: ln_post,
        }
    }
}



/// This is a single Monte Carlo trace, for a given thermodynamic beta
pub struct SetTrace<'a> {
    trace: Vec<StrippedMotifSet>,
    active_set: MotifSet<'a>,
    all_data_file: String,
    data_ref: &'a AllDataUse<'a>, 
    sparse: usize, 
    sparse_count: usize,
    thermo_beta: f64,
}


impl<'a> SetTrace<'a> {

    //All three of these references should be effectively static. They won't be ACTUALLY, because they're going to depend on user input, but still
    /// This initializes a trace
    /// - `capacity` sets the number of `[StrippedMotifSet]`s that the trace 
    ///   should remember in its monte carlo trace before outputting and deleting
    ///   them so that we can save memory
    /// - If `initial_condition` is `[None]`, we initialize with a random single Motif MotifSet.
    /// - `all_data_file` allows the saved trace to remember where `data_ref` 
    ///    should be: it is not necessary for inference, but it is for the saved trace
    ///    to reconstitute its occupancy traces. 
    /// - `thermo_beta` is the thermodynamic beta of the inference, but we 
    ///   automatically take the absolute value of it. 
    /// - The trace saves every `sparse`th `[MotifSet]` into its trace. If 
    ///   `sparse` is `[None]`, the default value is `30`. 
    pub fn new_trace<R: Rng + ?Sized>(capacity: usize, initial_condition: Option<MotifSet<'a>>, all_data_file: String,data_ref: &'a AllDataUse<'a>, mut thermo_beta: f64, sparse: Option<usize>, rng: &mut R) -> SetTrace<'a> {

        thermo_beta = thermo_beta.abs();

        let mut active_set = match initial_condition {
            Some(set) => set,
            None => MotifSet::rand_with_one(data_ref, rng),
        };

        println!("initialized as {:?}", active_set);

        let _ = active_set.ln_posterior();

        println!("initialized with {:?}", active_set);
        
        SetTrace{
            trace: Vec::<StrippedMotifSet>::with_capacity(capacity),
            active_set: active_set,
            all_data_file: all_data_file,
            data_ref: data_ref, 
            sparse: sparse.unwrap_or(30),
            sparse_count: 0_usize,
            thermo_beta: thermo_beta,
        }

    }

    pub fn thermo_beta(&self) -> f64 { self.thermo_beta }

    ///This runs the monte carlo move for an individual trace
    pub fn advance<R: Rng + ?Sized>(&mut self, track: Option<&mut MoveTracker>, burning_in: bool, no_motif_change: bool, rng: &mut R) { 

        let which_move = PICK_MOVE.sample(rng);


        let mut which_variant = SAMPLE_VARIANTS[which_move].sample(rng); 

        //println!("get to advance {} {}", self.thermo_beta, which_variant);
        //Can't use non const values in match statements. Using a bajillion if-elses 
        
        let potential_set_and_accept = match which_move {
        
            0 =>  {
           
                self.active_set.propose_rook_move(rng).map(|(new_mot, modded_ln_like)| { 
                    let accepted = MotifSet::accept_test(self.active_set.ln_posterior(), modded_ln_like, self.thermo_beta, rng); 
                    (new_mot, accepted)
                })  
            },
            1 =>  {
                self.active_set.propose_resid_leap(GOOD_V_BAD_CUT[which_variant], rng).map(|(new_mot, modded_ln_like)| { 
                    let accepted = MotifSet::accept_test(self.active_set.ln_posterior(), modded_ln_like, self.thermo_beta, rng); 
                    (new_mot, accepted)
                })
            }, 
            2 => {
                let height_sd = HEIGHT_SDS[which_variant];
                self.active_set.propose_height_move_custom(rng, height_sd).map(|(new_mot, modded_ln_like)| {
                        let accepted = MotifSet::accept_test(self.active_set.ln_posterior(), modded_ln_like, self.thermo_beta, rng);
                        (new_mot, accepted)
                })
            },
            
            3 => {
                if MAX_TF_NUM == 1 || no_motif_change {
                    let extend: bool = rng.gen();
                    which_variant = if extend {2} else {3};
                }
                self.active_set.run_rj_move_known(which_variant, self.thermo_beta, rng)
            },
            //4 => Some((self.active_set.base_leap(self.thermo_beta, rng), true)),
            //5 => Some((self.active_set.secondary_shuffle(self.thermo_beta, rng), true)),
            4 => {
                self.active_set.propose_base_leap(rng).map(|(new_mot, modded_ln_like)| {
                    let accepted = MotifSet::accept_test(self.active_set.ln_posterior(), modded_ln_like, self.thermo_beta, rng);
                    (new_mot, accepted)
                })
            },
            5 => {
                self.active_set.propose_secondary_shuffle(rng).map(|(new_mot, modded_ln_like)| {
                    let accepted = MotifSet::accept_test(self.active_set.ln_posterior(), modded_ln_like, self.thermo_beta, rng);
                    (new_mot, accepted)
                })
            },
            6 => {
                self.active_set.propose_kernel_swap(rng).map(|(new_mot, modded_ln_like)| {
                    let accepted = MotifSet::accept_test(self.active_set.ln_posterior(), modded_ln_like, self.thermo_beta, rng);
                    (new_mot, accepted)
                })
            },
            7 => {
                Some((self.active_set.randomize_motifs(self.thermo_beta, rng), true))
            },
            _ => unreachable!(),

        };

        //println!("execute move");

        //attempts_per_move: &mut [usize], successes_per_move: &mut [usize], immediate_failures_per_move: &mut [usize],
        //                            distances_per_attempted_move


        let (mut set_to_add, movement_tracker) = match potential_set_and_accept {
            None => (self.active_set.clone(), None),
            Some((mut actual_set, accept)) => {
                let tracked = self.active_set.measure_movement(&mut actual_set);
                let setting = if accept { actual_set } else { self.active_set.clone() };
                (setting, Some((tracked, accept)))
            }
        };
       
        //println!("measure move");

        if let Some(tracker) = track {
            let total_id = VARIANT_CUMULATIVE[which_move]+which_variant;
            //println!("total id {total_id}");
            tracker.document_motion(total_id, movement_tracker).expect("We only track the proper moves");
        };


        //println!("doc motion");
        if !burning_in {
            self.sparse_count = (self.sparse_count+1) % self.sparse;

            if self.sparse_count == 0 {

                //This order INCLUDES the last possible motif set
                //and specifically EXCLUDES the initial state

                //We include this pre processing step because we want the ln posteriors to avoid any selection pitfalls
                //Remember: we post process based on the maximum posterior estimate
                //let (full_set, _) = set_to_add.set_with_new_attention((0..self.data_ref.null_seq().num_sequence_blocks()).collect::<Vec<usize>>());
                //self.trace.push(StrippedMotifSet::from(&full_set));
                self.trace.push(StrippedMotifSet::from(&set_to_add));
            }
        }

        //println!("sparse");

        //NOTE: we do NOT include this in the actual acceptance steps on purpose.
        //      The motif finder must randomly choose which null sequence it's going to pay attention to for sampling
        //set_to_add.change_set_attention(self.data_ref.null_seq().yield_random_block_set(rng).clone());

        self.active_set = set_to_add;


    }


    pub fn loan_active(&self) -> &MotifSet<'a> {
        &self.active_set
    }

    pub fn stripped_current_set(&self) -> StrippedMotifSet {
        StrippedMotifSet::from(&self.active_set)
    }

    pub fn save_first_archived_state(&self, output_dir: &str, run_name: &str)-> Result<(), Box<dyn Error+Send+Sync>> {

        let savestate_file: String = output_dir.to_owned()+"/"+run_name+"_savestate.bin";

        let mut outfile_handle = fs::File::create(savestate_file)?;

        let mot_to_save: StrippedMotifSet = match self.trace.get(0) {
            Some(set) => set.clone(), 
            None => self.stripped_current_set(),
        };

        let buffer: Vec<u8> = bincode::serde::encode_to_vec( &mot_to_save, config::standard()).expect("serializable");

        outfile_handle.write(&buffer)?;
        Ok(())
    }

    pub fn save_last_archived_state(&self, output_dir: &str, run_name: &str)-> Result<(), Box<dyn Error+Send+Sync>> {

        let savestate_file: String = output_dir.to_owned()+"/"+run_name+"_savestate.bin";

        if self.trace.last().is_none() {warn!("trace is empty!"); return Ok(());}


        let last = self.trace.last().unwrap();

        let mut outfile_handle = fs::File::create(savestate_file)?;

        let buffer: Vec<u8> = bincode::serde::encode_to_vec( &last, config::standard()).expect("serializable");

        outfile_handle.write(&buffer)?;
        Ok(())
    }

/*
    pub fn save_current_bedgraph(&self, output_dir: &str, run_name: &str, zeroth_step: usize) -> Result<(), Box<dyn Error+Send+Sync>> { 

        //generate_bedgraph(&self, data_seq: &AllDataUse, chromosome_name: Option<&str>, file_name: &str)

        self.loan_active().set.iter().enumerate().map(|(i,(mot, _))| {
            let bedgraph_file:  String = format!("{}/{}_bedgraph_{:0>7}_motif_{i}.bed",output_dir,run_name,zeroth_step);
            mot.generate_bedgraph(self.data_ref, None, &bedgraph_file)
        }).collect::<Result<_,_>>()
    }
*/

    //We don't need to worry about saving any state in trace, since inference works on the ACTIVE state
    fn drop_history(&mut self) {
        self.trace.clear();
    }

    /// This saves the trace as a binarray file 
    /// `{output_dir}/{run_name}_trace_from_step_{zeroth_step}.bin', with 7 digits
    /// This binarray file stores a `[SetTraceDef]` for reserialization
    pub fn save_and_drop_history(&mut self, output_dir: &str, run_name: &str, zeroth_step: usize) -> Result<(), Box<dyn Error+Send+Sync>> {

        let trace_file: String = format!("{}/{}_trace_from_step_{:0>7}.bin",output_dir,run_name,zeroth_step);
        let mut outfile_handle = fs::File::create(trace_file)?;

        let len_trace = self.trace.len();

        //We don't need to worry about saving any state in trace, since inference works on the ACTIVE state
        let trace: Vec<StrippedMotifSet> = self.trace.drain(0..len_trace).collect();


        let buffer: Vec<u8> = bincode::serde::encode_to_vec( &(SetTraceDef {
            trace: trace,
            all_data_file: self.all_data_file.clone(),
            thermo_beta: self.thermo_beta,
        }), config::standard()).expect("serializable");


        //fs::write(trace_file.as_str(), serde_json::to_string(&history).unwrap()).expect("Need to give a valid file to write to or inference is pointless");

        outfile_handle.write(&buffer)?;

        self.save_first_archived_state(output_dir, run_name)
    }

    /// This saves the trace to an mgzip file, as indicated by `gz_writer`.
    /// `length_writer` saves the bit lengths of the compressed traces in bits.
    /// This allows us to save the whole motif trace in chunks that we can then
    /// parse.
    pub fn append_to_mgzip_and_drop_history<W1: Write, W2: Write, W3: Write>(&mut self, gz_writer: W1, length_writer: &mut W2, mut savestate_writer: W3) -> Result<(), Box<dyn Error+Send+Sync>> {

        let mut syncz = SyncZBuilder::<Mgzip,_>::new().compression_level(Compression::new(9)).from_writer(gz_writer);

        let len_trace = self.trace.len();

        let trace: Vec<StrippedMotifSet> = self.trace.drain(0..len_trace).collect();

        let buffer: Vec<u8> = bincode::serde::encode_to_vec( &(SetTraceDef {
            trace: trace,
            all_data_file: self.all_data_file.clone(),
            thermo_beta: self.thermo_beta,
        }), config::standard()).expect("serializable");

        let num_bytes = buffer.len();

        syncz.write_all(&buffer)?;
        
        length_writer.write_all(format!("{}\n", buffer.len()).as_bytes());


        let mot_to_save: StrippedMotifSet = self.stripped_current_set();

        let buffer: Vec<u8> = bincode::serde::encode_to_vec( &mot_to_save, config::standard()).expect("serializable");

        savestate_writer.write(&buffer)?;
        
        Ok(())

    }

    /// This saves the occupancy trace of the current active set in the directory 
    /// `{output_dir}/{run_name}/{zeroth_step}_new_occupancy`. For more details, 
    /// see `[MotifSet::save_set_trace_and_sub_traces]()`
    pub fn save_trace(&self, output_dir: &str, run_name: &str, zeroth_step: usize, annotations: Option<&GenomeAnnotations>, ontologies: Option<&[&str]>) -> Result<(), Box<dyn Error+Send+Sync>> {

        let current_active = &self.active_set;

        let file_name = format!("{}/{:0>7}", run_name,zeroth_step);

        current_active.save_set_trace_and_sub_traces(output_dir, &file_name, annotations, ontologies)

    }

    pub fn active_set_motif_lens(&self) -> Vec<usize> {
        self.active_set.set.iter().map(|a| a.0.len()).collect::<Vec<_>>()
    }

    pub fn active_set_peak_heights(&self) -> Vec<f64> {
        self.active_set.set.iter().map(|a| a.0.peak_height()).collect::<Vec<_>>()
    }



}

/// This is a `[SetTrace]` that is basically in an "archive" form and can be 
/// (de)serialized 
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SetTraceDef {

    trace: Vec<StrippedMotifSet>,
    all_data_file: String,
    thermo_beta: f64,

}




impl SetTraceDef {

    /// This takes `self` and reactivates it with respect to data_ref 
    /// # Correctness
    /// If you supply a data file where the posterior density of a set does not 
    /// match what its posterior density should be, nothing help you here. Now, 
    /// there are no _safety_ invariants that will break things here, but 
    /// consider this a correctness invariant on the posterior trace. It would 
    /// take too much time to validate the whole trace so we only rely on the 
    /// ACTIVE motif having a correct likelihood for this reason
    pub fn get_set_trace<'a, R: Rng+ ?Sized>(mut self, data_ref: &'a AllDataUse, rng: &mut R, sparse: Option<usize>) -> SetTrace<'a> {

        //We only validate the ln likelihood for the active motif
        let last_state = match self.trace.pop() {
            Some(state) => state.reactivate_set(data_ref),
            None => {
                warn!("Your trace has no sets! Initializing!");
                MotifSet::rand_with_one(data_ref, rng)
            },
        };


        SetTrace {
            trace: self.trace,
            data_ref: data_ref,
            all_data_file: self.all_data_file,
            active_set: last_state,
            sparse: sparse.unwrap_or(30),
            sparse_count: 0,
            thermo_beta: self.thermo_beta,
        }


    }

    /// This pares down the trace to just the first `max_elements` states
    pub fn reduce(&mut self, max_elements: usize) {
        if self.trace.len() >= max_elements {
            let _ = self.trace.drain(max_elements..).collect::<Vec<_>>();
        }
    }

    pub fn len(&self) -> usize {
        self.trace.len()
    }

    pub fn ln_posterior_trace(&self) -> Vec<f64> {
        self.trace.iter().map(|a| a.ln_post).collect::<Vec<f64>>()
    }

    pub fn bic_trace(&self, full_data: &AllDataUse) -> Vec<f64> {
        self.trace.iter().map(|a| a.bic(full_data)).collect::<Vec<f64>>()
    }

    /// This approximates eth from the ln likelihood
    /// This will give good answers for big eths
    /// But poorer answers for small eths
    /// We don't explicitly calculate eth for this because that will take
    /// 10,000 times longer (literally).
    pub fn approx_eth_trace(&self, full_data: &AllDataUse) -> Vec<f64> {


        // This is based on the asymptotic expansion of W0, modulo subbing exp(x)
        fn lambert_w_exp_x(x: f64) -> f64 {
            let lnx = x.ln();
            x-lnx+lnx/x+0.5*lnx*(lnx-2.0)/(x*x)
        }

        let ln_neg_twice_mul = (-2.0*MULT_CONST_FOR_H).ln();

        // This solves for eth assuming we are exclusively in high tail regime
        self.trace.iter().map(|a| a.ln_post-a.ln_prior(full_data))
                         .map(|b| 2.0*(ADD_CONST_FOR_H-b)+ln_neg_twice_mul)
                         .map(|b| lambert_w_exp_x(b)/2.0*ln_neg_twice_mul)
                         .collect()

    }
    
    /// This approximates ln eth from the ln likelihood
    /// This will give good answers for big eths
    /// But poorer answers for small eths
    /// We don't explicitly calculate eth for this because that will take
    /// 10,000 times longer (literally).
    pub fn approx_ln_eth_trace(&self, full_data: &AllDataUse) -> Vec<f64> {


        // This is based on the asymptotic expansion of W0, modulo subbing exp(x)
        fn lambert_w_exp_x(x: f64) -> f64 {
            let lnx = x.ln();
            x-lnx+lnx/x+0.5*lnx*(lnx-2.0)/(x*x)
        }

        let ln_neg_twice_mul = (-2.0*MULT_CONST_FOR_H).ln();

        // This solves for eth assuming we are exclusively in high tail regime
        self.trace.iter().map(|a| a.ln_post-a.ln_prior(full_data))
                         .map(|b| 2.0*(ADD_CONST_FOR_H-b)+ln_neg_twice_mul)
                         .map(|b| lambert_w_exp_x(b)/2.0*ln_neg_twice_mul)
                         .map(|b| b.ln())
                         .collect()

    }


    pub fn ln_likelihood_trace(&self, full_data: &AllDataUse) -> Vec<f64> {

        self.trace.iter().map(|a| a.ln_post-a.ln_prior(full_data)).collect::<Vec<f64>>()

    }

    pub fn motif_num_trace(&self) -> Vec<f64> {
        self.trace.par_iter().map(|a| a.set.len() as f64).collect::<Vec<_>>()
    }


    /// This returns a sparsed trace of occupancy signal Euclidean distances
    /// versus the data from `waypost`, using `number_samps` Monte Carlo samples
    /// # Panics
    /// If `number_samps > self.len()` 
    pub fn wave_rmse_trace(&self, waypost: &AllDataUse, number_samps: usize) -> Vec<(usize, f64)> {

        let step_size: usize = self.trace.len()/number_samps;
        self.trace.par_iter().enumerate().step_by(step_size).map(|(i, a)| (i, a.reactivate_set(waypost).signal_rmse())).collect()
    }
    
    /// This returns a sparsed trace of occupancy signal Euclidean distances
    /// versus the data from `waypost`, using `number_samps` Monte Carlo samples
    /// and accounting for matching binding in the `[NullSequence]`
    /// # Panics
    /// If `number_samps > self.len()` 
    pub fn wave_rmse_noise_trace(&self, waypost: &AllDataUse, number_samps: usize) -> Vec<(usize, f64)> {

        let step_size: usize = self.trace.len()/number_samps;
        self.trace.par_iter().enumerate().step_by(step_size).map(|(i, a)| (i, a.reactivate_set(waypost).magnitude_signal_with_noise())).collect()
    }

    /// This generates fimo files for `number_samps` steps in `self`.
    /// # Errors
    /// If MEME is not installed on your machine
    pub fn many_fimo_gen(&self, number_samps: usize, background_dist: Option<[f64; BASE_L]>, fasta_file: &str, output_dir: &str, run_name: &str) -> Result<(), Box<dyn Error+Send+Sync>> {
        let step_size: usize = self.trace.len()/number_samps;
        
        let m: Vec<Result<(), Box<dyn Error+Send+Sync>>> = self.trace.par_iter().step_by(step_size).enumerate().map(|(i, a)| {
         
            let save_file = format!("{}_{i}", run_name);
            let b = a.generate_fimo(background_dist,fasta_file,output_dir, &save_file);
        
            b
        }).collect();

        let n = m.into_iter().find(|a| a.is_err());
        match n {
            None => Ok(()),
            Some(e) => e,
        }
    }

    pub fn data_name(&self) -> &str {
        &self.all_data_file
    }

    pub fn initial_set_pwm(&self) -> Motif {
        self.trace[0].set[0].clone()
    }

    /// Sparses `self` to pick `num_motifs` sets, then picks a random motif from each set
    pub fn ret_rand_motifs<R: Rng + ?Sized>(&self, num_motifs: usize, rng: &mut R) -> Vec<Motif> {

        let spacing = self.trace.len()/num_motifs;

        let picks = (0..(self.trace.len()/spacing)).map(|a| a*spacing); 

        let set_picks: Vec<&StrippedMotifSet> = picks.map(|a| &(self.trace[a])).collect();

        let pwms: Vec<Motif> = set_picks.iter().map(|a| a.set.choose(rng).expect("No motif set should be empty").clone()).collect();

        pwms
    }

    /// Creates a trace of the minimum Aitchison distance of a `[Motif]` from
    /// each step of the `self` trace
    pub fn trace_min_dist(&self, reference_motif: &Motif) -> Vec<f64> {

        self.trace.iter().map(|mot_set| {

            mot_set.set.iter().map(|mot| mot.distance_function(&reference_motif).0)
                .min_by(|a,b| a.partial_cmp(b).expect("No NANs should be present in distances"))
                .expect("Motif sets all need at least one motif")

        }).collect::<Vec<f64>>()

    }

    /// Creates a trace of the minimum Aitchison distance of a `[Motif]` from
    /// each step of the `self` trace, returning of a tuple of that `[Motif]`
    /// followed by a tuple of the Aitchison distance and a `bool` that is `true`
    /// if we had to get our best match by reverse complementing and `false` otherwise.
    pub fn extract_best_motif_per_set(&self, reference_motif: &Motif, tail_start: usize, cutoff: f64) -> Vec<(Motif, (f64, bool))> {

        self.trace[(self.len()-tail_start)..self.len()].iter().map(|mot_set| {

            mot_set.set.iter().map(|mot| (mot.clone(),mot.distance_function(&reference_motif)))
                .min_by(|a,b| a.1.0.partial_cmp(&b.1.0).expect("No NANs should be present in distances"))
                .expect("Motif sets all need at least one motif")


        }).filter(|(_, (b, _))| *b < cutoff).collect::<Vec<_>>()

    }

    pub fn extract_highest_posterior_set(&self, tail_start: usize) -> &StrippedMotifSet {
        self.trace[(self.len()-tail_start)..self.len()].iter().max_by(|a,b| a.ln_post.partial_cmp(&b.ln_post).expect("No NaNs allowed in posterior")).expect("trace should have at least one motif set")
    }
    
    pub fn extract_highest_likelihood_set(&self, data_ref: &AllDataUse, tail_start: usize) -> &StrippedMotifSet {
        self.trace[(self.len()-tail_start)..self.len()].iter().max_by(|a,b| a.ln_likelihood(data_ref).partial_cmp(&b.ln_likelihood(data_ref)).expect("No NaNs allowed in prior")).expect("trace should have at least one motif set")
    }
    
    pub fn extract_lowest_bic_set(&self, data_ref: &AllDataUse, tail_start: usize) -> &StrippedMotifSet {
        self.trace[(self.len()-tail_start)..self.len()].iter().min_by(|a,b| a.bic(data_ref).partial_cmp(&b.bic(data_ref)).expect("No NaNs allowed in prior")).expect("trace should have at least one motif set")
    }

    fn posterior_probability_setup(&self, minimum_density_to_include: f64) -> Vec<&StrippedMotifSet> {

        //const BIGGEST_TRACKABLE_POST_DIFF: f64 = -708.4; //ln(minimum postive f64), rounded DOWN. Some of these could get rounded to zero, but I care more to capture everything I can than to cut everything I can

        if self.trace.len() == 0 { return vec![];}

        let max_posterior_density = self.trace.iter().max_by(|x,y| x.ln_post.partial_cmp(&y.ln_post).unwrap()).expect("returned if we were empty").ln_post;

        self.trace.iter().filter_map(|a| {

            if a.ln_post < minimum_density_to_include {None} else {
                Some(a)
            }
        }).collect()
    }

    pub fn evaluate_posterior_probability(&self, minimum_density_to_include: f64, evaluator: &dyn Fn(&StrippedMotifSet) -> f64) -> f64 {

        let sets_that_matter_and_posts = self.posterior_probability_setup(minimum_density_to_include);

        sets_that_matter_and_posts.iter().map(|a| evaluator(a)).sum::<f64>()/(sets_that_matter_and_posts.len() as f64)
    }

    /// Returns a vector of `[self.trace_min_dist()]` for each `[Motif]` in `reference_mots` 
    pub fn create_distance_traces(&self, reference_mots: &Vec<Motif>) -> Vec<Vec<f64>> {

        let mut distances: Vec<Vec<f64>> = vec![vec![0_f64; self.trace.len()] ; reference_mots.len()];


        for (i, trace_set) in self.trace.iter().enumerate() {

            for j in 0..reference_mots.len() {

                let inv_num_mot = 1.0/(trace_set.set.len() as f64);

                let mean_dist = trace_set.set.iter().map(|a| a.distance_function(&reference_mots[j]).0).sum::<f64>()*inv_num_mot;


                //SAFETY: j is always in bound of distances and i in bounds of distances[j] by construction
                //        I'm just not convinced the compiler will notice 
                unsafe {
                    *distances.get_unchecked_mut(j).get_unchecked_mut(i) = mean_dist;
                }

            }

        }

        distances


    }

    pub fn append(&mut self, mut attachment: SetTraceDef) {
        self.trace.append(&mut attachment.trace);
    }

    pub fn index_into(&self, range: std::ops::Range<usize>) -> &[StrippedMotifSet] {
        &self.trace[range]
    }

    pub fn ref_to_trace(&self) -> &Vec<StrippedMotifSet> {
        &self.trace
    }

    
}

impl Index<usize> for SetTraceDef {
type Output = StrippedMotifSet;

    fn index(&self, index: usize) -> &StrippedMotifSet {
        &self.trace[index]
    }

}


///This tracks move sizes and ratios, for diagnostic purposes
#[derive(Clone)]
pub struct MoveTracker {

    attempts_per_move: [usize; NUM_MOVES],
    successes_per_move: [usize; NUM_MOVES], 
    immediate_failures_per_move: [usize; NUM_MOVES], 
    distances_per_attempted_move: [Vec<([f64; 4], bool)>; NUM_MOVES], 

}

impl MoveTracker {

    /// This initializes the move tracker. `likely_steps` should be an estimate
    /// of how many steps should be saved in the move tracker
    pub fn new(likely_steps: usize) -> Self {

        let distances_per_attempted_move: [Vec::<([f64; 4], bool)>; NUM_MOVES] = core::array::from_fn(|_a| Vec::<([f64; 4], bool)>::with_capacity(likely_steps));

        MoveTracker {
            attempts_per_move: [0; NUM_MOVES],
            successes_per_move: [0; NUM_MOVES],
            immediate_failures_per_move: [0; NUM_MOVES],
            distances_per_attempted_move: distances_per_attempted_move,
        }
    }

    fn document_motion(&mut self, move_id: usize, motion_and_accceptance: Option<([f64;4], bool)>) -> Result<(), String> {

        if move_id >= NUM_MOVES { return Err("Invalid move id".to_string()); }

        self.attempts_per_move[move_id] += 1;

        match motion_and_accceptance {
            None => {self.immediate_failures_per_move[move_id] += 1;},
            Some((move_arr, accept)) => {
                if accept { self.successes_per_move[move_id] += 1;}
                self.distances_per_attempted_move[move_id].push((move_arr, accept));
            },
        };

        Ok(())

    }


    /// This yields the results of the move acceptance rates
    pub fn give_status(&self) {
        let mut ind: usize = 0;

            println!("Single base sample. Attempts: {}. Successes {}. Immediate failures {}. Rate of success {}. Rate of immediate failures {}.", 
                     self.attempts_per_move[ind], self.successes_per_move[ind], self.immediate_failures_per_move[ind], 
                     (self.successes_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64), (self.immediate_failures_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64));
            ind += 1;

            for cut in GOOD_V_BAD_CUT {
            println!("Guided scramble with cut {cut}. Attempts: {}. Successes {}. Immediate failures {}. Rate of success {}. Rate of immediate failures {}.",
                     self.attempts_per_move[ind], self.successes_per_move[ind], self.immediate_failures_per_move[ind],
                     (self.successes_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64), (self.immediate_failures_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64));
            ind += 1;
            }
            for i in 0..HEIGHT_SDS.len() {
                println!("Height move with sd {}. Attempts: {}. Successes {}. Immediate failures {}. Rate of success {}. Rate of immediate failures {}.",
                         HEIGHT_SDS[i], self.attempts_per_move[ind], self.successes_per_move[ind], self.immediate_failures_per_move[ind],
                         (self.successes_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64), (self.immediate_failures_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64));
                ind += 1;
            }
            println!("New motif move. Attempts: {}. Successes {}. Immediate failures {}. Rate of success {}. Rate of immediate failures {}.",
                     self.attempts_per_move[ind], self.successes_per_move[ind], self.immediate_failures_per_move[ind],
                     (self.successes_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64), (self.immediate_failures_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64));
            ind += 1;

            println!("Kill motif move. Attempts: {}. Successes {}. Immediate failures {}. Rate of success {}. Rate of immediate failures {}.",
                     self.attempts_per_move[ind], self.successes_per_move[ind], self.immediate_failures_per_move[ind],
                     (self.successes_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64), (self.immediate_failures_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64));

            ind += 1;
            println!("Extend motif move. Attempts: {}. Successes {}. Immediate failures {}. Rate of success {}. Rate of immediate failures {}.",
                     self.attempts_per_move[ind], self.successes_per_move[ind], self.immediate_failures_per_move[ind],
                     (self.successes_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64), (self.immediate_failures_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64));

            ind += 1;
            println!("Contract motif move. Attempts: {}. Successes {}. Immediate failures {}. Rate of success {}. Rate of immediate failures {}.",
                     self.attempts_per_move[ind], self.successes_per_move[ind], self.immediate_failures_per_move[ind],
                     (self.successes_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64), (self.immediate_failures_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64));
            ind += 1;
            println!("New motif move alt. Attempts: {}. Successes {}. Immediate failures {}. Rate of success {}. Rate of immediate failures {}.",
                     self.attempts_per_move[ind], self.successes_per_move[ind], self.immediate_failures_per_move[ind],
                     (self.successes_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64), (self.immediate_failures_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64));
            ind += 1;

            println!("Kill motif move alt. Attempts: {}. Successes {}. Immediate failures {}. Rate of success {}. Rate of immediate failures {}.",
                     self.attempts_per_move[ind], self.successes_per_move[ind], self.immediate_failures_per_move[ind],
                     (self.successes_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64), (self.immediate_failures_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64));


            /*ind += 1;
            println!("Split motif move. Attempts: {}. Successes {}. Immediate failures {}. Rate of success {}. Rate of immediate failures {}.",
                     self.attempts_per_move[ind], self.successes_per_move[ind], self.immediate_failures_per_move[ind],
                     (self.successes_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64), (self.immediate_failures_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64));
            ind += 1;
            println!("Merge motif move. Attempts: {}. Successes {}. Immediate failures {}. Rate of success {}. Rate of immediate failures {}.",
                     self.attempts_per_move[ind], self.successes_per_move[ind], self.immediate_failures_per_move[ind],
                     (self.successes_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64), (self.immediate_failures_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64));
            */
            ind += 1;
                        println!("Base leap move. Attempts: {}. Successes {}. Immediate failures {}. Rate of success {}. Rate of immediate failures {}.",
                     self.attempts_per_move[ind], self.successes_per_move[ind], self.immediate_failures_per_move[ind],
                     (self.successes_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64), (self.immediate_failures_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64));

            //println!("Base leap move (always accepts). Times {}. Last distance is {}", self.attempts_per_move[ind], match self.distances_per_attempted_move[ind].last() { Some(dist) => format!("{:?}", dist.0), None => "None tried".to_owned() });
            ind += 1;
                        println!("Secondary shuffle move. Attempts: {}. Successes {}. Immediate failures {}. Rate of success {}. Rate of immediate failures {}.",
                     self.attempts_per_move[ind], self.successes_per_move[ind], self.immediate_failures_per_move[ind],
                     (self.successes_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64), (self.immediate_failures_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64));

            //println!("Secondary shuffle move (always accepts). Times {}. Last distance is {}", self.attempts_per_move[ind], match self.distances_per_attempted_move[ind].last() { Some(dist) => format!("{:?}", dist.0), None => "None tried".to_owned() });
            ind += 1;
            println!("Randomize kernel. Attempts: {}. Successes {}. Immediate failures {}. Rate of success {}. Rate of immediate failures {}.", self.attempts_per_move[ind], self.successes_per_move[ind], self.immediate_failures_per_move[ind],(self.successes_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64), (self.immediate_failures_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64));
            ind += 1;
            println!("Randomize move (always accepts). Times {}. Last distance is {}", self.attempts_per_move[ind], match self.distances_per_attempted_move[ind].last() { Some(dist) => format!("{:?}", dist.0), None => "None tried".to_owned() });
            ind += 1;
    }
   

    //Note: the fact that we need to pass plotting into this means that this function cannot in context where we need all arguments to be Send or Sync
    fn sort_move_hists<DB: DrawingBackend>(&self, move_id: usize, plotting: &DrawingArea<DB, Shift>, num_bins: usize) -> Result<(), String>{

        if move_id >= NUM_MOVES { return Err("Invalid move id".to_string()); }
        
        let data = &self.distances_per_attempted_move[move_id];
        
        if data.len() == 0 {return Err("No data for plotting".to_string());}

        plotting.fill(&WHITE).expect("This should just work");


        let left_subs = plotting.split_evenly((2,2));

        let labs = ["Occupancy Signal RMSE between moves", "Finite Likelihood differences", "Euclidean distance of heights", "Total distance of all PWMs"];

        for (j, area) in left_subs.iter().enumerate() {
            let trial_data = data.iter().map(|(a, _)| a[j]).collect::<Vec<_>>();
            let trial_data_2 = data.iter().filter(|(_, b)| *b).map(|(a, _)| a[j]).collect::<Vec<_>>();
            let _hist = quick_hist(&trial_data, &trial_data_2, area, labs[j].to_string(), num_bins);
        }
        Ok(())

    }

    fn all_move_hists(&self, base_file_name: &str, num_bins: usize) -> Result<(), Vec<String>> {
        
        let v: Vec<String> = (0..NUM_MOVES).map(|i|
                                                {
                                                    let file = format!("{}{}", base_file_name,HIST_END_NAMES[i]);
                                                    let plotting = BitMapBackend::new(&file, (8000, 4000)).into_drawing_area();
                                                    (i, self.sort_move_hists(i, &plotting, num_bins))
                                                }).filter(|(_,x)| x.is_err()).map(|(i, a)| {
                                                    let mut m = a.unwrap_err();
                                                    m.push_str(&format!(" {}",i));
                                                    m
        }).collect();
        if v.len() != 0 { Err(v)} else { Ok(()) }
    }

}

fn build_hist_bins(mut data: Vec<f64>, mut data_2: Vec<f64>, num_bins: usize) -> (Vec<f64>, Vec<(f64, f64)>, Vec<(f64, f64)>) {

    let length = data.len() as f64;

    let mut big_data = data.clone();

    big_data.append(&mut data_2.clone());

    data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    data_2.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    big_data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let min = big_data[0];
    let max = *big_data.last().unwrap();

    let step = (max-min)/(num_bins as f64);

    let add_into = 1./length;

    let xs = (0..num_bins).map(|i| (min+(i as f64)*step)).collect::<Vec<_>>();

    let mut bins: Vec<(f64, f64)> = xs.iter().clone().map(|&x| (x, 0.0)).collect();

    let mut bins_2 = bins.clone();

    let mut j: usize = 1;
    let mut k: usize = 1;
    for &dat in data.iter() {

        //Because data and bins are sorted, we only need to find the first bin
        //where the data is less than the top end. We use short circuit && to prevent overflow
        while (j < (bins.len()-1)) && (dat >= bins[j+1].0) { j+= 1;}

        bins[j].1 += add_into;

    }
    for &dat in data_2.iter() {

        //Because data and bins are sorted, we only need to find the first bin
        //where the data is less than the top end. We use short circuit && to prevent overflow
        while (k < (bins.len()-1)) && (dat >= bins[k+1].0) { k+= 1;}

        bins_2[k].1 += add_into;

    }

    (xs, bins, bins_2)

}

fn quick_hist<'a, 'b, DB: DrawingBackend, N: Copy+Into<f64>>(raw_data: &[N], raw_data_2: &[N], area: &'a DrawingArea<DB, Shift>, label: String, num_bins: usize) -> ChartBuilder<'a, 'b, DB> {

    let mut hist = ChartBuilder::on(area);

    hist.margin(10).y_label_area_size(200).x_label_area_size(200);

    hist.caption(label, ("Times New Roman", 80));

    let data: Vec<f64> = raw_data.iter().map(|&a| a.into()).collect();
    let data_2: Vec<f64> = raw_data_2.iter().map(|&a| a.into()).collect();


    let (xs, hist_form, hist_form_2) = build_hist_bins(data, data_2, num_bins);

    let range = RangedSlice::from(xs.as_slice());

    let max_prob = hist_form.iter().map(|&x| x.1).fold(0_f64, |x,y| x.max(y));

    let mut hist_context = hist.build_cartesian_2d(range, 0_f64..max_prob).unwrap();

    hist_context.configure_mesh().x_label_style(("serif", 70))
        .y_label_style(("serif", 70))
        .x_label_formatter(&|v| format!("{:.0}", v))
        .axis_desc_style(("serif",90))
        .x_desc("Ln posterior density proposed set/current set")
        .y_desc("Probability").disable_x_mesh().disable_y_mesh().x_label_formatter(&|x| format!("{:.04}", *x)).draw().unwrap();

    //hist_context.draw_series(Histogram::vertical(&hist_context).style(CYAN.filled()).data(trial_data.iter().map(|x| (x, inverse_size)))).unwrap();
    hist_context.draw_series(Histogram::vertical(&hist_context).style(CYAN.mix(0.8).filled()).margin(0).data(hist_form.iter().map(|x| (&x.0, x.1)))).unwrap().label("Proposed Moves").legend(|(x, y)| Rectangle::new([(x-20, y-10), (x, y+10)], Into::<ShapeStyle>::into(&CYAN).filled()));

    hist_context.draw_series(Histogram::vertical(&hist_context).style(RED.mix(0.8).filled()).margin(0).data(hist_form_2.iter().map(|x| (&x.0, x.1)))).unwrap().label("Accepted Moves").legend(|(x, y)| Rectangle::new([(x-20, y-10), (x, y+10)], Into::<ShapeStyle>::into(&RED).filled()));

    hist_context.configure_series_labels().position(SeriesLabelPosition::UpperLeft).margin(40).legend_area_size(20).border_style(&BLACK).label_font(("Calibri", 80)).draw().unwrap();

    hist


}


/// This enum denotes how much tracking of move success rates we do
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub enum TrackingOptions {
    /// Indicates no move success rates need to be tracked
    NoTracking,
    /// Indicates that only the beta = 1.0 trace move success rates need tracking.
    /// Note that only one of these traces is tracked if there are multiple
    TrackTrueTrace,
    /// Indicates that all traces have their move success rates tracked
    TrackAllTraces,
}


/// This is the struct which actually holds all of the information for the parallel tempering
pub struct TemperSetTraces<'a> {
    //Each trace knows its thermodynamic beta
    parallel_traces: Vec<(SetTrace<'a>, Option<MoveTracker>, f64)>,
    currently_tracking: usize,
    track: TrackingOptions,
}

impl<'a> TemperSetTraces<'a> {

    /// This initializes the necessary scaffolding for the inferences. 
    /// `min_thermo_beta` is the minimum thermodynamic beta for an inference. 
    /// `num_intermediate_traces` is the number of parallel traces with a 
    /// thermodynamic beta between `min_thermo_beta` and `1.0`
    /// `capacity_per_trace` is a usize that sets the number of inference steps
    /// which should occur before a trace tries to save its current state 
    /// and output its history
    /// `step_num_estimate` is a capacity estimate for how many moves will run 
    /// per trace, and mainly exists to mitigate memory reallocation
    /// `sparse` indicates how many inference steps should be skipped over
    /// before saving the current trace. If it's `[None]`, it defaults to `30`.
    ///
    /// Errors 
    /// If `min_thermo_beta =< 0.0 || > 1.0`, or if an initial condition is 
    /// provided that is not compatible with `data_ref`
    pub fn new_parallel_traces<R: Rng+?Sized>(min_thermo_beta: f64, num_intermediate_traces: usize, capacity_per_trace: usize, step_num_estimate: usize, how_to_track: TrackingOptions, all_data_file: String, data_ref: &'a AllDataUse<'a>, initial_condition: Option<MotifSet<'a>>, sparse: Option<usize>, rng: &mut R) -> Result<Self, InitializationError> {
        if let Some(a) = initial_condition.as_ref(){
            if !ptr::eq(a.data_ref, data_ref) { return Err(InitializationError::UnsynchedData); }
        }

        if min_thermo_beta <= 0.0 { return Err(InitializationError::NegativeTemperature); }
        if min_thermo_beta > 1.0 { return Err(InitializationError::UselessBeta); }

        println!("begin");


        let num_null_blocks = data_ref.null_seq().num_sequence_blocks();

        let num_traces = num_intermediate_traces+2;


        //I always want my intermediate states to be even, because I want to be able to try to swap everybody at once
        //So I always want to randomly 
        let mut thermos = vec![1_f64; num_traces];


        /// This distributes the thermodyanmic beta log uniformly
        let thermo_step = -(min_thermo_beta.log2())/((thermos.len()-1) as f64);

        *thermos.last_mut().expect("thermos always has elements") = min_thermo_beta;

        //TODO: check that this DOES produce a linear decrease in thermodynamic beta
        for i in 1..(num_traces-1) {
            thermos[i] = (-(i as f64)*thermo_step).exp2();
        }


        let mut parallel_traces: Vec<(SetTrace, Option<MoveTracker>, f64)> = Vec::with_capacity(thermos.len());

        //pub fn new_trace(capacity: usize, initial_condition: Option<MotifSet<'a>>, data_ref: &'a AllDataUse<'a>, mut thermo_beta: f64, sparse: Option<usize>, mut rng: R) -> SetTrace<'a>

        let past_initial: bool = false;

        let mut current_weight: f64 = 0.0;

        for (i, thermo_beta) in thermos.into_iter().enumerate() {
            let potential_tracker = match how_to_track {
                TrackingOptions::NoTracking => None, 
                TrackingOptions::TrackAllTraces => Some(MoveTracker::new(step_num_estimate)),
                TrackingOptions::TrackTrueTrace => if past_initial { None } else { Some(MoveTracker::new(step_num_estimate)) },
            };
            let set_trace = SetTrace::new_trace(capacity_per_trace, initial_condition.clone(), all_data_file.clone(),data_ref, thermo_beta, sparse, rng); 
            println!("thermo {}", set_trace.thermo_beta);
        
            parallel_traces.push((set_trace, potential_tracker,current_weight));

            current_weight += thermo_step*(ADDITIVE_WEIGHT_CONST+0.5*MULTIPLY_WEIGHT_CONST*(EXPONENT_WEIGHT_CONST*thermo_beta)*(1.0+thermo_step.exp()));
        }

        Ok(TemperSetTraces { parallel_traces: parallel_traces, currently_tracking: 0, track: how_to_track })
    }


   

    /// This actually runs one set of parallel monte carlo steps, followed by 
    /// attempting parallel swaps. In particular, this runs `iters_before_swaps` 
    /// steps between attempting parallel tempering steps. 
    /// First, on all adjacent pairs starting with the index `1` trace, then
    /// all adjacent pairs starting with the index `0` trace
    pub fn iter_and_swap<R: Rng>(&mut self, iters_before_swaps: usize, no_motif_change: bool, rng_maker: fn() -> R) {

       
        //for _ in 0..iters_before_swaps {
            self.parallel_traces.par_iter_mut().for_each(|(set, track, _)| {
                let mut rng = rng_maker();
                for _ in 0..iters_before_swaps {
                    set.advance(track.as_mut(), no_motif_change, no_motif_change, &mut rng); //The second no_motif_change is the one that actually ensures no motif change. The first one is there because I don't want steps saved while the reversible jump on the number of motifs is turned off. 
                }
            });
        //}
        let data_seq = self.parallel_traces[0].0.data_ref;


        
                
        println!("Ln posteriors of each trace before swaps: {:?}", self.parallel_traces.iter_mut().map(|x| x.0.active_set.ln_posterior()).collect::<Vec<f64>>());

        

        let odd_swaps: Vec<([f64;2], bool)> = self.parallel_traces[1..].par_chunks_exact_mut(2).map(|x| {
            let (c, d) = x.split_at_mut(1);
            let (a, b) = (&mut c[0], &mut d[0]);
            let mut rng = rng_maker();
            //NOTE: I don't need to do anything special for the reversible jump case. Having to
            //account for the forward and reverse swaps already does dimension matching for me.
            let accept_swap = MotifSet::accept_test(a.0.active_set.ln_posterior(), b.0.active_set.ln_posterior(), a.0.thermo_beta-b.0.thermo_beta, &mut rng);
            if accept_swap {
                //let tmp = a[0].0.active_set;
                //a[0].0.active_set = a[1].0.active_set;
                //a[1].0.active_set = tmp;
                std::mem::swap(&mut a.0.active_set, &mut b.0.active_set);
            }
            ([a.0.thermo_beta, b.0.thermo_beta], accept_swap)
        }).collect();

        println!("Odd swaps: {:?}", odd_swaps);


        let even_swaps: Vec<([f64;2], bool)> = self.parallel_traces.par_chunks_exact_mut(2).map(|x| {
            let (c, d) = x.split_at_mut(1);
            let (a, b) = (&mut c[0], &mut d[0]);
            let mut rng = rng_maker();
            //NOTE: I don't need to do anything special for the reversible jump case. Having to
            //account for the forward and reverse swaps already does dimension matching for me.
            let accept_swap = MotifSet::accept_test(a.0.active_set.ln_posterior(), b.0.active_set.ln_posterior(), a.0.thermo_beta-b.0.thermo_beta, &mut rng);
            if accept_swap {
                //let tmp = a[0].0.active_set;
                //a[0].0.active_set = a[1].0.active_set;
                //a[1].0.active_set = tmp;
                std::mem::swap(&mut a.0.active_set, &mut b.0.active_set);
            }
            ([a.0.thermo_beta, b.0.thermo_beta], accept_swap)
        }).collect();

        println!("Even swaps: {:?}", even_swaps);

        println!("Ln posteriors of each trace after swaps: {:?}", self.parallel_traces.iter_mut().map(|x| x.0.active_set.ln_posterior()).collect::<Vec<f64>>());
        //println!("distances of each trace after swaps: {:?}", self.parallel_traces.par_iter_mut().map(|x| x.0.active_set.signal_rmse()).collect::<Vec<f64>>());
        //println!("distances of each trace after swaps accounting noise: {:?}", self.parallel_traces.par_iter_mut().map(|x| x.0.active_set.magnitude_signal_with_noise()).collect::<Vec<f64>>());

        println!("tf numbs of each treach after swaps: {:?}", self.parallel_traces.iter().map(|x| x.0.active_set.set.len()).collect::<Vec<usize>>());


    }


    pub fn print_acceptances(&self) {

        let tracker = self.track;

        match tracker {
            TrackingOptions::NoTracking => (),
            TrackingOptions::TrackTrueTrace => {
                println!("Motif lengths: {:?}", self.parallel_traces[0].0.active_set.set.iter().map(|a| a.0.len()).collect::<Vec<_>>());
                println!("Motif heights: {:?}", self.parallel_traces[0].0.active_set.set.iter().map(|a| a.0.peak_height()).collect::<Vec<_>>());
                println!("Motif strings: {:?}", self.parallel_traces[0].0.active_set.set.iter().map(|a| a.0.best_motif_string()).collect::<Vec<_>>());
                self.parallel_traces[0].1.as_ref().map(|a| a.give_status());},
            TrackingOptions::TrackAllTraces => { _ = self.parallel_traces.iter().map(|b| { 
                println!("Thermodynamic beta: {}", b.0.thermo_beta);
                println!("Motif lengths: {:?}", b.0.active_set.set.iter().map(|a| a.0.len()).collect::<Vec<_>>());
                println!("Motif heights: {:?}", b.0.active_set.set.iter().map(|a| a.0.peak_height()).collect::<Vec<_>>());
                println!("Motif strings: {:?}", b.0.active_set.set.iter().map(|a| a.0.best_motif_string()).collect::<Vec<_>>());
                b.1.as_ref().map(|a| a.give_status())
            }).collect::<Vec<_>>();},
        }

    }

    pub fn handle_histograms(&self, track: TrackingOptions, pre_file_name: &str, num_bins: usize) {
        let tracker = self.track.min(track);
        
        match tracker {
            TrackingOptions::NoTracking => (),
            TrackingOptions::TrackTrueTrace => {self.parallel_traces[0].1.as_ref().map(|a| a.all_move_hists(pre_file_name, num_bins));},
            TrackingOptions::TrackAllTraces => {self.parallel_traces.par_iter().map(|b| b.1.as_ref() //Tried having this be a par_iter, but something plays very nastily with Rayon
                                                                                .map(|a| {
                                                                                    let alter_name = format!("{}_thermo_beta_{}", pre_file_name, b.0.thermo_beta);
                                                                                    a.all_move_hists(&alter_name, num_bins)
                                                                                })).collect::<Vec<_>>();},
        }


    }




    /*pub fn save_trace_and_clear(&mut self, output_dir: &str, run_name: &str, zeroth_step: usize) -> Result<(), Box<dyn Error+Send+Sync>> {

        self.parallel_traces[0].0.save_trace(output_dir, run_name, zeroth_step)?;
        self.parallel_traces[0].0.save_and_drop_history(output_dir, run_name, zeroth_step)?;

        for i in 1..self.parallel_traces.len() { self.parallel_traces[i].0.drop_history(); }

        Ok(())

    }*/

    pub fn save_trace_by_gzip_and_clear<W1: Write, W2: Write, W3: Write>(&mut self, gz_writer: W1, length_writer: &mut W2, savestate_writer: W3) -> Result<(), Box<dyn Error+Send+Sync>> {

        self.parallel_traces[0].0.append_to_mgzip_and_drop_history(gz_writer, length_writer, savestate_writer)?;

        for i in 1..self.parallel_traces.len() { self.parallel_traces[i].0.drop_history(); }

        Ok(())

    }

}

/// This is the error returned if parallel tempering is badly initialized. 
pub enum InitializationError {
    /// Parallel traces have an initial motif which points to the wrong AllDataUse
    UnsynchedData,
    /// Parallel traces have a negative thermodynamic beta, which will mess up the inference
    NegativeTemperature, 
    /// Parallel traces have a thermodyanmic beta greater than 1.0,
    /// which we could technically infer with but which would be pointless to use.
    UselessBeta, 
}

impl fmt::Display for InitializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsynchedData => write!(f, "Tried to initialize with a set that references a different data set!"),
            Self::NegativeTemperature => write!(f, "Negative temperature will invert your inference!"),
            Self::UselessBeta => write!(f, "Thermodynamic beta cooler than 1.0 will make your inference compute more while also being useless!"),
        }
    }
}
impl fmt::Debug for InitializationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsynchedData => write!(f, "Tried to initialize with a set that references a different data set!"),
            Self::NegativeTemperature => write!(f, "Negative temperature will invert your inference!"),
            Self::UselessBeta => write!(f, "Thermodynamic beta cooler than 1.0 will make your inference compute more while also being useless!"),
        }
    }
}
impl Error for InitializationError {}

//BEGIN TRUNCATED LOGNORMAL

/// This is a Log Normal distribution, truncated on both ends and renormalized
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct TruncatedLogNormal {
    location: f64,
    scale: f64,
    min: f64, 
    max: f64,
    ref_dist: LogNormal,
}

impl TruncatedLogNormal {

    pub fn new(location: f64, scale: f64, min: f64, max: f64) -> Result<TruncatedLogNormal, TruncatedLogNormalError> {
        if min.is_nan() || max.is_nan() {
            Err(TruncatedLogNormalError::NaNPars)
        } else if min < 0.0 || (max < 0.0) {
            Err(TruncatedLogNormalError::NegPars)
        } else if min >= max{
            Err(TruncatedLogNormalError::BadMinMaxRange)
        } else {
            let ref_dist = match LogNormal::new(location, scale) {
                Ok(dist) => dist,
                Err(err) => return Err(TruncatedLogNormalError::BadLogNormal(err)),
            };
            Ok(TruncatedLogNormal { location: location, scale: scale, min: min, max: max, ref_dist: ref_dist })
        }
    }

    pub fn d_ln_pdf(&self, x: f64) -> f64 {
        if x < self.min || x > self.max {
            0.0
        } else {
            -(x.ln()+self.scale.powi(2)-self.location)/(x*self.scale.powi(2))
        }
    }

}
#[derive(Error, Debug, Clone)]
pub enum TruncatedLogNormalError {
    NaNPars,
    NegPars,
    BadMinMaxRange,
    BadLogNormal(LogNormalError),
}

impl std::fmt::Display for TruncatedLogNormalError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self{ 
            Self::NaNPars => write!(f, "Don't submit NaNs for your range"),
            Self::NegPars => write!(f, "Don't submit negative values for your range"),
            Self::BadMinMaxRange => write!(f, "Your minimum must be less than your maximum"),
            Self::BadLogNormal(e) => write!(f, "{}", e),
        }
    }
}


impl ::rand::distributions::Distribution<f64> for TruncatedLogNormal {

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {

        let norm: Normal = Normal::new(self.location, self.scale).unwrap();
        let mut sample: f64 = norm.sample(rng).exp();
        let mut invalid: bool = (sample > self.max) | (sample < self.min);
        while invalid {
            sample = norm.sample(rng).exp();
            invalid = (sample > self.max) | (sample < self.min);
        }
        sample
    }
}

impl Min<f64> for TruncatedLogNormal {
    fn min(&self) -> f64 {
        self.min
    }
}

impl Max<f64> for TruncatedLogNormal {
    fn max(&self) -> f64 {
        self.max
    }
}

impl ContinuousCDF<f64, f64> for TruncatedLogNormal {

    fn cdf(&self, x: f64) -> f64 {
        if x <= self.min {
            0.0
        } else if x >= self.max {
            1.0
        } else {

            self.ref_dist.cdf(x)/(self.ref_dist.cdf(self.max)-self.ref_dist.cdf(self.min))
        }
    }

    fn sf(&self, x: f64) -> f64 {
        if x <= self.min {
            1.0
        } else if x >= self.max() {
            0.0
        } else {
            self.ref_dist.sf(x)/(self.ref_dist.cdf(self.max)-self.ref_dist.cdf(self.min))
        }
    }
}


impl Continuous<f64, f64> for TruncatedLogNormal {

    fn pdf(&self, x: f64) -> f64 {
        if x < self.min || x > self.max {
            0.0
        } else {
            let d = (x.ln() - self.location) / self.scale;
            let usual_density = (-0.5 * d * d).exp() / (x * consts::SQRT_2PI * self.scale);
            let scale_density = self.ref_dist.cdf(self.max)-self.ref_dist.cdf(self.min);

            usual_density/scale_density

        }
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        if x < self.min || x > self.max {
            f64::NEG_INFINITY
        } else {
            let d = (x.ln() - self.location) / self.scale;
            let usual_density = (-0.5 * d * d) - consts::LN_SQRT_2PI - (x * self.scale).ln();
            let scale_density = (self.ref_dist.cdf(self.max)-self.ref_dist.cdf(self.min)).ln();

            usual_density-scale_density
        }
    }
}


impl ContinuousLnCDF<f64, f64> for TruncatedLogNormal {

    fn ln_cdf(&self, x: f64) -> f64{
        self.cdf(x).ln()
    }

    fn ln_sf(&self, x: f64) -> f64 {
        self.sf(x).ln()
    }
}



//}
#[cfg(test)]
mod tester{

    use std::time::{Instant};
    /*use crate::base::bases::Base;
      use crate::base::bases::GBase;
      use crate::base::bases::TruncatedLogNormal;
      use crate::base::bases::{Motif, THRESH};*/
    use super::*;
    use crate::sequence::{Sequence};
    use statrs::distribution::{Continuous, ContinuousCDF, LogNormal, Normal};
    use statrs::statistics::{Min, Max};
    use statrs::function::gamma::{gamma, ln_gamma};
    use rand::Rng;
    use crate::waveform::*;
    use rand::distributions::{Distribution};
    use crate::SymmetricBaseDirichlet;
    const MIN_HEIGHT: f64 = 3.0;



    #[test]
    fn distance_fallibility() {
        let mut rng = rand::thread_rng();

        for _ in 0..10000 {

            let base_a = Base::rand_new(&mut rng);
            let base_b = Base::rand_new(&mut rng);
            let dist_1 = base_a.dist_sq(Some(&base_b));
            let dist_2 = base_b.dist_sq(Some(&base_a));

            println!("{base_a:?} {base_b:?} {dist_1} {dist_2} {}", base_a.prior_per_base());

            let scores: [f64; BASE_L] = [0.0, rng.gen::<f64>()*SCORE_THRESH, rng.gen::<f64>()*SCORE_THRESH, rng.gen::<f64>()*SCORE_THRESH];
            let mut base_true_rand = Base { scores: scores};


            let base_e = Base { scores: [0.0, SCORE_THRESH, SCORE_THRESH, SCORE_THRESH]};

        }

        println!("alts {} {} {} {}", BASE_CHOOSE_ALT.energy_ln_pmf(-0.25), BASE_CHOOSE_ALT.energy_ln_pmf(-0.5), BASE_CHOOSE_ALT.energy_ln_pmf(-0.75), BASE_CHOOSE_ALT.energy_ln_pmf(-1.0));

    }

    #[test]
    fn distances_test() {
        let mut rng = rand::thread_rng();
        static HEIGHT_DIST: Lazy<TruncatedLogNormal> = Lazy::new(|| TruncatedLogNormal::new(LOG_HEIGHT_MEAN, LOG_HEIGHT_SD, 3.0, MAX_HEIGHT).unwrap() );
        let m = Motif::from_motif(vec![Bp::A, Bp::C, Bp::G, Bp::G, Bp::G, Bp::G, Bp::T, Bp::T, Bp::T], &*HEIGHT_DIST, &mut rng);
        let dself = m.distance_function(&m);
        println!("{:?} dself", dself);
        assert!(dself.0.abs() < 1e-8);
        assert!(!dself.1);
        let revved = Motif::raw_pwm(m.rev_complement(), m.peak_height(), KernelWidth::Wide, KernelVariety::Gaussian);
        println!("forward {:?} revved {:?}", m, revved);
        let drev = m.distance_function(&(Motif::raw_pwm(m.rev_complement(), m.peak_height(), KernelWidth::Wide, KernelVariety::Gaussian)));
        println!("{:?} drev", drev);
        assert!(drev.0.abs() < 1e-8);
        assert!(drev.1);
        let zero_base = Base::new([0.0;BASE_L]);
        println!("zero dist {}", zero_base.dist_sq(None));
        let mut m_star_pwm = m.pwm.clone();
        m_star_pwm.insert(0,zero_base.clone());
        m_star_pwm.push(zero_base);
        let unaltered_check = Motif::raw_pwm(m_star_pwm.clone(), m.peak_height(), KernelWidth::Wide, KernelVariety::Gaussian);
        let d_unalt = m.distance_function(&unaltered_check);
        println!("m {:?} \n m_star {:?} {:?}", m, unaltered_check, m_star_pwm);
        println!("unalt {:?}", d_unalt);
        assert!(d_unalt.0.abs() < 1e-16);
        assert!(!d_unalt.1);
        let mut new_b_vec: Vec<Base> = Vec::with_capacity(m.len()+2);
        new_b_vec.push(Base::rand_new(&mut rng));
        new_b_vec.append(&mut m.rev_complement());
        new_b_vec.push(Base::rand_new(&mut rng));
        let newmot = Motif::raw_pwm(new_b_vec, m.peak_height(), KernelWidth::Wide, KernelVariety::Gaussian);
        let d_unalt_rev = m.distance_function(&newmot);
        println!("new_b_vec {:?} {:?} {}", newmot, d_unalt_rev, (newmot.pwm[0].dist_sq(None)+newmot.pwm[newmot.len()-1].dist_sq(None)).sqrt());
        assert!(d_unalt_rev.0.abs()-(newmot.pwm[0].dist_sq(None)+newmot.pwm[newmot.len()-1].dist_sq(None)).sqrt() < 1e-9);
        assert!(d_unalt_rev.1);

        //let r = Motif::from_motif(vec![Bp::A, Bp::G, Bp::G, Bp::T, Bp::A, Bp::G, Bp::G, Bp::T], &mut rng);



    }

    #[test]
    fn extract_bp_test() {

        let mut rng = rand::thread_rng();

        let base: Base = Base::rand_new(&mut rng);


        let u64_vec: Vec<u64> = (0..10000).map(|_| rng.gen()).collect();
        let usize_vec: Vec<usize> = u64_vec.iter().map(|&a| {
            let mut b = a;
            let mut v: Vec<usize> = vec![0;32];

            for i in 0..32{
                v[i] = (b&3) as usize;
                b = b >> 2;
            }
            v

        }).flatten().collect();
        let bp_vec: Vec<Bp> = u64_vec.iter().map(|&a| Sequence::u64_to_kmer(a, 32)).flatten().collect();


        println!("{:?} {:?}", &bp_vec[0..10], &usize_vec[0..10]);

        let bad_safe_access = Instant::now();
        let accessed_c: Vec<f64> = usize_vec.iter().map(|&b| base.scores[b]).collect();
        let bad_safe_time = bad_safe_access.elapsed();

        let safe_access = Instant::now();
        let accessed: Vec<f64> = bp_vec.iter().map(|&b| base[b]).collect();
        let safe_time = safe_access.elapsed();

        let unsafe_access = Instant::now();
        let accessed_b: Vec<f64> = usize_vec.iter().map(|&b| base.bind(b)).collect();
        let unsafe_time = unsafe_access.elapsed();


        println!("{:?}", accessed.len());
        println!("{:?}", accessed_b.len());
        println!("{:?}", accessed_c.len());
        println!("safe {:?} unsafe {:?} bad_safe: {:?}", safe_time, unsafe_time, bad_safe_time);
    }


/*
    #[test]
    fn simplex_test() {
        println!("start");
        let mut rng = rand::thread_rng();
        let b = Base::rand_new(&mut rng);

        let simplex = b.base_to_vect();

        let mod_b = Base::vect_to_base(&simplex);

        println!("st {:?} {:?} {:?} {} {} {}", b, simplex, mod_b, b.dist_sq(None), b.dist_sq(Some(&b)), simplex.iter().map(|&a| a.powi(2)).sum::<f64>().sqrt());



        let b0 = b.add_in_vector_space([1e-6, 0.0, 0.0], false);
        let b1 =  b.add_in_vector_space([0.0, 1e-6, 0.0], false);
        let b2 =  b.add_in_vector_space([0.0, 0.0, 1e-6], false);

        println!("{:?} {:?} {} {} {}", b.base_to_vect(), b0.base_to_vect(), b0.base_to_vect()[0]-b.base_to_vect()[0],  b0.base_to_vect()[1]-b.base_to_vect()[1],  b0.base_to_vect()[2]-b.base_to_vect()[2]);
        println!("{:?} {:?} {} {} {}", b.base_to_vect(), b1.base_to_vect(), b1.base_to_vect()[0]-b.base_to_vect()[0],  b1.base_to_vect()[1]-b.base_to_vect()[1],  b1.base_to_vect()[2]-b.base_to_vect()[2]);
        println!("{:?} {:?} {} {} {}", b.base_to_vect(), b2.base_to_vect(), b2.base_to_vect()[0]-b.base_to_vect()[0],  b2.base_to_vect()[1]-b.base_to_vect()[1],  b2.base_to_vect()[2]-b.base_to_vect()[2]);

        
        println!("{:?} {:?} {:?}", b0.scores.iter().zip(b.scores.iter()).map(|(&a, &b)| (a-b)/1e-6).collect::<Vec<_>>(), b1.scores.iter().zip(b.scores.iter()).map(|(&a, &b)| (a-b)/1e-6).collect::<Vec<_>>(),
                 b2.scores.iter().zip(b.scores.iter()).map(|(&a, &b)| (a-b)/1e-6).collect::<Vec<_>>());

        let simp = [0.1_f64, -0.15, -0.11];
        let simp_b = Base::vect_to_base(&simp);

        let b3a = simp_b.add_in_vector_space([0.0, 0.0, 0.02], false);
        let b3 = simp_b.add_in_vector_space([0.0, 0.0, 0.02], true);

        let sim_b3a = b3a.base_to_vect();
        let sim_b3 = b3.base_to_vect();

        println!("sim {:?} simb {:?} noref {:?} norefb {:?} ref {:?} refb{:?}", simp, simp_b, b3a.base_to_vect(), b3a, b3.base_to_vect(), b3);

        assert!((simp[0]-sim_b3a[0]).abs() < 1e-3, "0th element changes with no reflection");
        assert!((0.09-sim_b3[0] ).abs() < 1e-3, "0th element changes with a reflection");

        assert!((simp[1]-sim_b3a[1]).abs() < 1e-3, "1st element changes with no reflection");
        assert!((simp[1]-sim_b3[1] ).abs() < 1e-3, "1st element changes with a reflection");

        println!("{:?} {:?}", sim_b3a, sim_b3);
        assert!((-0.09-sim_b3a[2]).abs() < 1e-3, "2nd element incorrect with no reflection");
        assert!((-simp[0]-sim_b3[2]).abs() < 1e-3, "2nd element incorrect with a reflection");

        let b_a = b.add_in_vector_space([10.0, -10.0, 1./3.], false);

        println!("b_a {:?}", b_a);

        let b_b = b.add_in_vector_space([10.0, -10.0, 1./3.], true);

        println!("best o {:?} best nr {:?} best r {:?}",b.best_base(),b_a.best_base(),b_b.best_base());

        println!("b {:?}, nr {:?}, r {:?}", b, b_a, b_b);

        let confine_start = Base::simplex_to_base(&[1_f64/3., 0., 0.]);


        let mom_confine_ref = [0_f64, 0., SQRT_2/3.];

        //let confine_flip = [0_f64, 0., - SQRT_2/3.];
        println!("pre ref");
        let confine_end = confine_start.add_in_vector_space(mom_confine_ref, true);

        let confine_end_simp = confine_end.base_to_vect();

        //let should_start = confine_end.add_in_vector_space(confine_flip, true);
        println!("end simp {:?} prop_end_simp {:?}", confine_end_simp, [5_f64/9., 0., SQRT_2/3.-(2.*SQRT_2)/9.]);

        //println!("start {:?} back_to_start {:?}", confine_start, should_start);

        //let random_base_dist = SymmetricBaseDirichlet::new(1.0_f64).expect("obviously valid");

        let _length_x_to_test: usize = 1000;
        //let xs_on_t_face = (0..length_x_to_test).map(|i| SQRT_2*(i as f64)/(length_x_to_test as f64)-(SQRT_2/3.0)).collect::<Vec<f64>>();
        //let most_ys_on_t_face = (0..(2*length_x_to_test)).map(|i| (SQRT_2/SQRT_3)*(1.0-(i as f64)/(length_x_to_test as f64))).collect::<Vec<f64>>();


        let random_base_dist = SymmetricBaseDirichlet::new(0.0518590998043053).expect("obviously valid");

        let t = Instant::now();
        for i in 0..10000 {
            let b = Base::vect_to_base(&(random_base_dist.sample(&mut rng)).base_to_vect());
            println!("Base {i} {:?}", b);
            let f = b.base_to_vect();
            let b2 = Base::vect_to_base(&f); 
            println!("back to simplex {:?}", f);
       
            println!("base once more {:?}", Base::vect_to_base(&f));
      

            let bitarr = core::array::from_fn::<_,4,_>(|m| { let a = BP_ARRAY[m]; (b2[a]).to_bits() & 0x7ff_0_0000_0000_0000}) ;
            let difarr = core::array::from_fn::<_,4,_>(|m| { let a = BP_ARRAY[m]; (b2[a]-b[a]).to_bits() & 0x7ff_0_0000_0000_0000} ) ;

            let mag_diff = core::array::from_fn::<_,4,_>(|a| if difarr[a] == 0 { None } else { Some((bitarr[a]-difarr[a]) >> 52) });

            println!("mag_diff {:?}", mag_diff);

            assert!(mag_diff.iter().all(|&a| a.is_none() || (a.unwrap() > 40)), "magnitude of relative difference between back and forth exceeds 2^(-45)")


        }
        
        let random_base_dist = SymmetricBaseDirichlet::new(1000.).expect("obviously valid");

        for i in 0..10000 {
            let b = Base::vect_to_base(&(random_base_dist.sample(&mut rng)).base_to_vect());
            println!("Base {i} {:?}", b);
            let f = b.base_to_vect();
            let b2 = Base::vect_to_base(&f); 
            println!("back to simplex {:?}", f);
       
            println!("base once more {:?}", Base::vect_to_base(&f));
      

            let bitarr = core::array::from_fn::<_,4,_>(|m| { let a = BP_ARRAY[m]; (b2[a]).to_bits() & 0x7ff_0_0000_0000_0000}) ;
            let difarr = core::array::from_fn::<_,4,_>(|m| { let a = BP_ARRAY[m]; (b2[a]-b[a]).to_bits() & 0x7ff_0_0000_0000_0000} ) ;

            let mag_diff = core::array::from_fn::<_,4,_>(|a| if difarr[a] == 0 { None } else { Some((bitarr[a]-difarr[a]) >> 52) });

            println!("mag_diff {:?}", mag_diff);

            assert!(mag_diff.iter().all(|&a| a.is_none() || (a.unwrap() > 40)), "magnitude of relative difference between back and forth exceeds 2^(-46)")


        }

        let elapse = t.elapsed();
        println!("time converting between stuff {:?}", elapse);

        let s = [-SQRT_2/3., -0.01, -0.3333333333333];
        println!("simplex {:?}", s);

        let b = Base::vect_to_base(&s);

        println!("Base tr {:?}", b);

        let s2 = b.base_to_vect();
        println!("simplex again {:?}", s2);
        println!("base once more {:?}",Base::vect_to_base(&s2));

        let b3 = b.add_in_vector_space([0.00001*(-1.0), 0.0, 0.0], false);
        println!("b3 {:?}", b3);
        let b3 = b.add_in_vector_space([0.0, 0.0, 0.00001*(-1.0)], false);
        println!("b4 {:?}", b3);
        let b3 = b.add_in_vector_space([1./0.00001*(-1.0), 1.0, 1./0.00001*(-1.0)], true);
        println!("b5 {:?}", b3);

        for p in SIMPLEX_VERTICES_POINTS {

            println!("point {:?}", p);
            let mut b = Base::vect_to_base(&p);
            for _ in 0..10 {
                println!("Base {:?}", Base::vect_to_base(&p));
                let f = b.base_to_vect();
                println!("back to simplex {:?}", f);
                b = Base::vect_to_base(&f);
                println!("base once more {:?}", b);

            }
            let mut to_add = b.base_to_vect();
            let _ = to_add.iter_mut().map(|a| {*a = *a/100.;}).collect::<Vec<_>>();

            println!("trying to add {:?}", b.add_in_vector_space(to_add, true));
        }

        /*for i in 0..length_x_to_test {

          for j in i..(2*length_x_to_test-i) {

        //let a_base_vec: [f64; 3] = [xs_on_t_face[i], most_ys_on_t_face[j], -1.0/3.0];
        let a_base_vec: [f64; 3] = [SQRT_2*(i as f64)/(length_x_to_test as f64)-(SQRT_2/3.0), (SQRT_2/SQRT_3)*(1.0-(j as f64)/(length_x_to_test as f64)), -1.0/3.0];
        println!("point {:?}", a_base_vec);
        println!("Base {:?}", Base::simplex_to_base(&a_base_vec));

        }
        }*/






    }*/

    #[test]
    fn leap_test() {


        println!("begin");
        let mut rng = rand::thread_rng(); //fastrand::Rng::new();

        let block_n: usize = 5;
        let u8_per_block: usize = 3000;
        let bp_per_block: usize = u8_per_block*4;
        let bp: usize = block_n*bp_per_block;
        let u8_count: usize = u8_per_block*block_n;

        println!("begin grad set gen");
        let start_gen = Instant::now();
        //let blocks: Vec<u8> = (0..u8_count).map(|_| rng.u8(..)).collect();
        //let preblocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.u8(..)).collect();
        let preblocks: Vec<u8> = (0..(u8_count/block_n)).map(|_| rng.gen::<u8>()).collect();
        let blocks: Vec<u8> = preblocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>(); 
        let _block_u8_starts: Vec<usize> = (0..block_n).map(|a| a*u8_per_block).collect();
        let block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
        let _start_bases: Vec<usize> = (0..block_n).map(|a| a*bp_per_block).collect();
        let sequence: Sequence = Sequence::new_manual(blocks, block_lens.clone(), &((0..block_n).map(|i| (i, 0, bp_per_block)).collect::<Vec<_>>()));
        

        let pre_null_blocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.gen::<u8>()).collect();
        let null_blocks: Vec<u8> = pre_null_blocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>(); 
        let null_block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
        let start_null_bases: Vec<usize> = (0..block_n).map(|a| 2*a*bp_per_block).collect();
        let null_seq: NullSequence = NullSequence::new_manual(null_blocks, null_block_lens.clone());

        let wave: Waveform = Waveform::create_zero(&sequence, 5);
        let duration = start_gen.elapsed();
        println!("grad gen {} bp {:?}", bp, duration);

        let background = Background::new(0.25, 2.64, 350./6.).unwrap();

        let start_bases_ids: Vec<usize> = (0..block_n).map(|a| (2*a+1)*bp_per_block+10).collect();

        let data_seq = unsafe{ AllDataUse::new_unchecked_data(wave.clone(),&null_seq, &start_bases_ids,&start_null_bases, &background, MIN_HEIGHT, 4.126)};

        let mut motif_set = MotifSet::rand_with_one_height(9.6, &data_seq, &mut rng);

        _ = motif_set.add_motif(Motif::rand_mot_with_height(13.2, wave.seq(), &mut rng));

        let mot = motif_set.get_nth_motif(0);
        let mot1 = motif_set.get_nth_motif(1);
        let mot_kmer = mot.best_motif();
        let mot1_kmer = mot1.best_motif();
        let id: usize = 4;
        let mot_scram = mot.scramble_by_id_to_valid(id, true, &sequence);
        let scram_kmer = Sequence::u64_to_kmer(sequence.idth_unique_kmer(mot.len(),id), mot.len());
        
        println!("old {} \n new {}", mot, mot_scram);

        let mut all_base_correct = true;
        let mut all_scramble_correct = true;
        for base in 0..mot.len() {
            all_base_correct &= mot_scram.pwm[base][scram_kmer[base]] == 0.0;
            let best_old = mot_kmer[base];
            let best_new = scram_kmer[base];
            for bp in BP_ARRAY {

                if bp == best_old {
                    all_scramble_correct &= mot_scram.pwm[base][bp] == mot.pwm[base][best_new];
                } else if bp == best_new {
                    all_scramble_correct &= mot_scram.pwm[base][bp] == mot.pwm[base][best_old];
                } else {
                    all_scramble_correct &= mot_scram.pwm[base][bp] == mot.pwm[base][bp];
                }

            }

        }

        assert!(mot_scram.peak_height() == -mot.peak_height(), "peak height doesn't flip when it should");
        assert!(all_base_correct, "Not all the new swapped bases are what they should be");
        assert!(all_scramble_correct, "Not all of the swaps preserve values unless they're being swapped and also swap correctly.");

        let mot_scram = mot.scramble_by_id_to_valid(id, false, &sequence);
        assert!(mot_scram.peak_height() == mot.peak_height(), "peak height isn't preserved when it should be");

        //let hamming_fn = |x: usize| if x < 10 {2} else {(x/2)-2} ;
        
        let hamming_fn = |x: usize| if x < 10 {1} else { x/2-3};

        let ln_num_hammings = |x: &Motif| ((sequence.all_kmers_within_hamming(&x.best_motif(), hamming_fn(x.len())+1).len()) as f64).ln() ;
        println!(" motif 0 neighbors {} motif 1 neighbors {}", ln_num_hammings(motif_set.nth_motif(0)).exp(), ln_num_hammings(motif_set.nth_motif(1)).exp());
        //This should usually be Some
        //TODO: make this output Some significantly more often: it tends to make false positives on giving errors because it doesn't always Some
        let (leaped, leap_score) = motif_set.propose_base_leap(&mut rng).unwrap();

        let leap = leaped.get_nth_motif(0);
        let leap1 = leaped.get_nth_motif(1);

        assert!((mot.peak_height() == leap.peak_height()) || (mot.peak_height() == -leap.peak_height()), "zeroth motif height is wrong");
        assert!((mot1.peak_height() == leap1.peak_height()) || (mot1.peak_height() == -leap1.peak_height()), "first motif height is wrong");

        let leap_kmer = leap.best_motif();
        let leap1_kmer = leap1.best_motif();

        
        let kmer_num_ln_rat = ln_num_hammings(&leap)+ln_num_hammings(&leap1)-(ln_num_hammings(&motif_set.get_nth_motif(0))+ln_num_hammings(&motif_set.get_nth_motif(1)));

        println!("ln rats {} {} {} {}", ln_num_hammings(&leap), ln_num_hammings(&leap1), ln_num_hammings(&motif_set.get_nth_motif(0)), ln_num_hammings(&motif_set.get_nth_motif(1)));
        println!("shuff check {} {:?} {}", leap_score, leaped.ln_post, kmer_num_ln_rat);
        assert!((leap_score-(leaped.ln_post.unwrap()+kmer_num_ln_rat)).abs() < 1e-6);

        let mut all_scramble_correct = true;
        for base in 0..mot.len() {
            let best_old = mot_kmer[base];
            let best_new = leap_kmer[base];
            for bp in BP_ARRAY {
                if bp == best_old {
                    all_scramble_correct &= leap.pwm[base][bp] == mot.pwm[base][best_new];
                } else if bp == best_new {
                    all_scramble_correct &= leap.pwm[base][bp] == mot.pwm[base][best_old];
                } else {
                    all_scramble_correct &= leap.pwm[base][bp] == mot.pwm[base][bp];
                }

            }

        }

        assert!(all_scramble_correct, "zeroth motif not correctly leaped");

        let mut all_scramble_correct = true;
        for base in 0..mot1.len() {
            let best_old = mot1_kmer[base];
            let best_new = leap1_kmer[base];
            for bp in BP_ARRAY {
                if bp == best_old {
                    all_scramble_correct &= leap1.pwm[base][bp] == mot1.pwm[base][best_new];
                } else if bp == best_new {
                    all_scramble_correct &= leap1.pwm[base][bp] == mot1.pwm[base][best_old];
                } else {
                    all_scramble_correct &= leap1.pwm[base][bp] == mot1.pwm[base][bp];
                }

            }

        }

        assert!(all_scramble_correct, "first motif not correctly leaped");

        let mut alt_leaped = leaped.clone();
        alt_leaped.ln_post = None;
        let ln_post = alt_leaped.ln_posterior();

        println!("{ln_post}, {}", leaped.ln_post.unwrap());
        println!("diff ln_post {}", ln_post-leaped.ln_post.unwrap());

        assert!((ln_post-leaped.ln_post.unwrap()).abs() <= 1e-8, "ln posteriors not lining up"); 

        let recalced_signal = leaped.recalced_signal();

        let sig_diff = &leaped.signal-&recalced_signal;

        let any_diffs: Vec<f64> = sig_diff.raw_wave().iter().filter(|a| a.abs() > 1e-8).map(|&a| a).collect();

        println!("any diff {:?}", any_diffs);

        assert!(any_diffs.len() == 0, "waves not lining up"); 
        

        //let shuffed = motif_set.secondary_shuffle(1.0, &mut rng);

        //This should never be None
        let (shuffed, shuffed_score) = motif_set.propose_secondary_shuffle( &mut rng).unwrap();

        let shuff = shuffed.get_nth_motif(0);
        let shuff1 = shuffed.get_nth_motif(1);

        assert!(mot.peak_height() == shuff.peak_height(), "zeroth motif height is wrong");
        assert!(mot1.peak_height() == shuff1.peak_height(), "first motif height is wrong");

        let shuff_kmer = shuff.best_motif();
        let shuff1_kmer = shuff1.best_motif();


        let mut all_scramble_correct = true;
        for base in 0..mot.len() {
            let best_old = mot_kmer[base];
            for bp in BP_ARRAY {
                if bp == best_old {
                    all_scramble_correct &= shuff.pwm[base][bp] == mot.pwm[base][bp];
                } 
            }

        }

        assert!(all_scramble_correct, "zeroth motif not correctly shuffed");

        let mut all_scramble_correct = true;
        for base in 0..mot1.len() {
            let best_old = mot1_kmer[base];
            for bp in BP_ARRAY {
                if bp == best_old {
                    all_scramble_correct &= shuff1.pwm[base][bp] == mot1.pwm[base][bp];
                } 
            }

        }

        assert!(all_scramble_correct, "first motif not correctly shuffed");

        let mut alt_shuffed = shuffed.clone();
        alt_shuffed.ln_post = None;
        let ln_post = alt_shuffed.ln_posterior();

        println!("{:?} {:?}", motif_set, shuffed);

        println!("{ln_post}, {}", shuffed.ln_post.unwrap());
        println!("diff ln_post {}", ln_post-shuffed.ln_post.unwrap());

        assert!((ln_post-shuffed.ln_post.unwrap()).abs() <= 64.0*std::f64::EPSILON, "ln posteriors not lining up"); 

        let recalced_signal = shuffed.recalced_signal();

        let sig_diff = &shuffed.signal-&recalced_signal;

        let pre_diffs: Vec<(usize, f64)> = sig_diff.raw_wave().iter().enumerate().filter(|(_, a)| a.abs() > 64.0*std::f64::EPSILON).map(|(i,a)| (i, *a)).collect();

        let any_diffs: Vec<f64> = pre_diffs.iter().map(|(_,a)| *a).collect();

        let any_diff_inds: Vec<usize> = pre_diffs.iter().map(|(i,_)| *i).collect();

        println!("any diff {:?}", any_diffs);

        println!("Any diff inds {:?}", any_diff_inds);

        let signal_remaining = any_diff_inds.iter().map(|&a| shuffed.signal.raw_wave()[a]).collect::<Vec<_>>();

        println!("sig {:?}", signal_remaining);

        let no_forgive_diffs = any_diffs.into_iter().zip(signal_remaining.into_iter()).filter(|(a, b)| (a/b).abs() > 4.0*std::f64::EPSILON).collect::<Vec<_>>();

        println!("mot set a {:?}\n mot set b {:?}", motif_set, shuffed);

        println!("unforgiven {:?}", no_forgive_diffs);

        assert!(no_forgive_diffs.len() == 0, "waves not lining up"); 
       
        /*
        let mut swap = motif_set.propose_kernel_swap( &mut rng );

        while swap.is_none() {
            swap = motif_set.propose_kernel_swap( &mut rng );
        }

        let (swapped, swapped_score) = swap.unwrap();

        let swapp = swapped.get_nth_motif(0);
        let swapp1 = swapped.get_nth_motif(1);

        let zero_swap = !(swapp.kernel_width == mot.kernel_width && swapp.kernel_variety == mot.kernel_variety);
        let one_swap = !(swapp1.kernel_width == mot1.kernel_width && swapp1.kernel_variety == mot1.kernel_variety);

        assert!(zero_swap != one_swap, "Not swapping correctly {} {}", zero_swap, one_swap);

        assert!(mot.peak_height() == swapp.peak_height(), "zeroth motif height is wrong");
        assert!(mot1.peak_height() == swapp1.peak_height(), "first motif height is wrong");

        let swapp_kmer = swapp.best_motif();
        let swapp1_kmer = swapp1.best_motif();


        let mut all_swap_correct = true;
        for base in 0..mot.len() {
            for bp in BP_ARRAY {
                all_swap_correct &= swapp.pwm[base][bp] == mot.pwm[base][bp]
            }

        }

        assert!(all_swap_correct, "zeroth motif not correctly swapped");

        let mut all_swap_correct = true;
        for base in 0..mot1.len() {
            for bp in BP_ARRAY {
                    all_swap_correct &= swapp1.pwm[base][bp] == mot1.pwm[base][bp];
            }

        }

        assert!(all_swap_correct, "first motif not correctly swapped");
        let mut alt_swapped = swapped.clone();
        alt_swapped.ln_post = None;
        let ln_post = alt_swapped.ln_posterior();
    */
      
    }

    #[test]
    fn rj_manipulator_tests() {

        let mut rng = rand::thread_rng(); //fastrand::Rng::new();

        let block_n: usize = 5;
        let u8_per_block: usize = 90;
        let bp_per_block: usize = u8_per_block*4;
        let bp: usize = block_n*bp_per_block;
        let u8_count: usize = u8_per_block*block_n;

        println!("begin grad set gen");
        let start_gen = Instant::now();
        //let blocks: Vec<u8> = (0..u8_count).map(|_| rng.u8(..)).collect();
        //let preblocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.u8(..)).collect();
        let preblocks: Vec<u8> = (0..(u8_count/block_n)).map(|_| rng.gen::<u8>()).collect();
        let blocks: Vec<u8> = preblocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>(); 
        let _block_u8_starts: Vec<usize> = (0..block_n).map(|a| a*u8_per_block).collect();
        let block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
        let _start_bases: Vec<usize> = (0..block_n).map(|a| a*bp_per_block).collect();
        let sequence: Sequence = Sequence::new_manual(blocks, block_lens.clone(), &((0..block_n).map(|i| (i, 0, bp_per_block)).collect::<Vec<_>>()));
        

        let _pre_null_blocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.gen::<u8>()).collect();
        let null_blocks: Vec<u8> = preblocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>(); 
        let _block_u8_null_starts: Vec<usize> = (0..block_n).map(|a| a*u8_per_block).collect();
        let null_block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
        let start_null_bases: Vec<usize> = (0..block_n).map(|a| 2*a*bp_per_block).collect();
        let null_seq: NullSequence = NullSequence::new_manual(null_blocks, null_block_lens.clone());

        let wave: Waveform = Waveform::create_zero(&sequence, 5);
        let duration = start_gen.elapsed();
        println!("grad gen {} bp {:?}", bp, duration);

        let background = Background::new(0.25, 2.64, 350./6.).unwrap();


        let minmers = sequence.unique_kmers(MIN_BASE);

        let mut sum: f64 = 0.0;

        let start_bases_ids: Vec<usize> = (0..block_n).map(|a| (2*a+1)*bp_per_block+10).collect();

        let data_seq = unsafe{ AllDataUse::new_unchecked_data(wave.clone(),&null_seq, &start_bases_ids,&start_null_bases, &background, MIN_HEIGHT, 4.126)};
        let _block_n: usize = 5;

        let duration = start_gen.elapsed();
        println!("grad gen {} bp {:?}", bp, duration);

        let mut motif_set = MotifSet::rand_with_one_height(9.6, &data_seq, &mut rng);

        let check_set = motif_set.clone();

        let mid_mot = (MIN_BASE+MAX_BASE)/2;
        //TESTING THE MANIPULATOR FUNCTIONS: these should simply mutate the motif set to conform and usually output the new ln posterior



        let add_mot = Motif::rand_mot_with_height_and_motif_len(10.0,mid_mot, data_seq.data().seq(), &mut rng);


        //Testing: fn add_motif(&mut self, new_mot: Motif) -> f64 
        let new_like = motif_set.add_motif(add_mot.clone());

        //I'm comfortable enforcing exactly equality because new_like should literally be a copy of ln_post
        assert!(motif_set.ln_post == Some(new_like));

        let wave_diff = &motif_set.signal-&check_set.signal;

        let check_diff = &wave_diff-&add_mot.generate_waveform(&data_seq);

        let all_good = check_diff.read_wave().iter().map(|&a| a.abs() < 1e-9).fold(true, |f, x| f && x);

        println!("{:?}", check_diff);

        assert!(all_good);

        assert!(motif_set.set.len() == 2);

        println!("add motif correct wave and like");

        //Testing fn remove_motif(&mut self, rem_id: usize) -> f64
        let _new_like = motif_set.remove_motif(1);

        assert!((motif_set.calc_ln_post()- check_set.calc_ln_post()).abs() < 1e-8, "{}", format!("{:?} {:?}", motif_set.ln_post, check_set.ln_post));

        let wave_diff = &motif_set.signal-&check_set.signal;

        //let check_diff = &wave_diff-&add_mot.generate_waveform(&data_seq);

        let all_good = wave_diff.read_wave().iter().map(|&a| a.abs() < 1024.0*std::f64::EPSILON).fold(true, |f, x| f && x);

        assert!(all_good);

        assert!(motif_set.set.len() == 1);
        println!("rem motif correct wave and like");
        _ = motif_set.add_motif(add_mot.clone());

        //Testing fn remove_motif_void(&mut self, rem_id: usize). This explicitly does not return a likelihood and Nones the current ln posterior
        //It is otherwise identical to remove_motif()
        motif_set.remove_motif_void(1);

        assert!(motif_set.ln_post.is_none());

        let wave_diff = &motif_set.signal-&check_set.signal;

        let all_good = wave_diff.read_wave().iter().map(|&a| a.abs() < 1024.0*std::f64::EPSILON).fold(true, |f, x| f && x);

        assert!(all_good);
        assert!(motif_set.set.len() == 1);
        println!("rem motif void correct wave and like");

        //Testing fn insert_motif(&mut self, new_mot: Motif, position: usize) -> f64. This is designed to put a new motif in a particular position of the set. 
        //It is otherwise identical to add_motif: it exists for base_leap, mainly
        let new_like = motif_set.insert_motif(add_mot.clone(), 0);

        assert!(motif_set.ln_post == Some(new_like));

        let wave_diff = &motif_set.signal-&check_set.signal;

        let check_diff = &wave_diff-&add_mot.generate_waveform(&data_seq);

        let all_good = check_diff.read_wave().iter().map(|&a| a.abs() < 1024.0*std::f64::EPSILON).fold(true, |f, x| f && x);

        assert!(all_good);

        //I'm comfortable enforcing exactly equality because this should be an exact copy
        assert!(motif_set.nth_motif(0).peak_height() == add_mot.peak_height());


        assert!(motif_set.set.len() == 2);

        //Testing fn replace_motif(&mut self, new_mot: Motif, rem_id: usize) -> f64

        let add_mot2 = Motif::rand_mot_with_height(6.2, data_seq.data().seq(), &mut rng);

        let new_like = motif_set.replace_motif(add_mot2.clone(), 0);

        assert!(motif_set.ln_post == Some(new_like));

        let bring_wave_back = &motif_set.signal -&add_mot2.generate_waveform(&data_seq);

        let check_diff = &bring_wave_back-&check_set.signal;

        let all_good = check_diff.read_wave().iter().map(|&a| a.abs() < 1024.0*std::f64::EPSILON).fold(true, |f, x| f && x);

        let max_err = check_diff.read_wave().iter().map(|&a| a.abs()).fold(-f64::INFINITY, |f, x| f.max(x));
        println!("ad {}", max_err);
        assert!(all_good);

        assert!(motif_set.set.len() == 2);

        assert!(motif_set.nth_motif(0).peak_height() == add_mot2.peak_height());


        //move tests. These should all produce Options of tuples
        //None should be produced iff executing the move would produce
        //a theoretically impossible result (eg, a motif set with no motifs).
        //Otherwise Some((MotifSet, f64)) should be produced, with the f64
        //being a sum of the ln posterior of the new motif set and the 
        //ln ancillary probability density: the lapd of growth moves is
        //-ln(density of proposing the particular growth), while the 
        //lapd of reverse moves is +ln(density of proposing the reversing growth)


        let mut failures = 0_usize;

        //let (single_mot, _ln_prop) = loop { if let Some(r) = motif_set.propose_ordered_base_move_custom(&mut rng, 0.1, 0.1) { break r; } else {println!("fail base {failures}"); failures+=1;}};
        let (single_mot, _ln_prop) = loop { if let Some(r) = motif_set.propose_rook_move(&mut rng) { break r; } else {println!("fail base {failures}"); failures+=1;}};

        println!("failed in single base move {} times before succeeding", failures);
        assert!(single_mot.set.len() == motif_set.set.len());
        let mut found_change: bool = false;
        for i in 0..motif_set.set.len() {
            assert!(motif_set.nth_motif(i).peak_height() == single_mot.nth_motif(i).peak_height());
            let unchanged = motif_set.nth_motif(i).pwm.iter().zip(single_mot.nth_motif(i).pwm.iter()).all(|(a, b)| a.dist_sq(Some(b)).sqrt() < 1e-6);
            if !unchanged {
                if found_change {
                    panic!("multiple motifs changed in single base scale move!");
                } else { 
                    found_change = true; 

                    let changed_bases = motif_set.nth_motif(i).pwm.iter().zip(single_mot.nth_motif(i).pwm.iter()).enumerate().filter(|(_, (a, b))| a.dist_sq(Some(*b)).sqrt() >= 1e-6).map(|(i,_)| i).collect::<Vec<_>>();

                    println!("{}", changed_bases.len());
                    assert!(changed_bases.len() == 1);

                    let old_base = motif_set.nth_motif(i).pwm[changed_bases[0]].clone();
                    let new_base = single_mot.nth_motif(i).pwm[changed_bases[0]].clone();

                    println!("{:?} {:?} {:?} {:?}", old_base, old_base.base_to_vect(), new_base, new_base.base_to_vect());
                    let mut old_order = old_base.scores.iter().enumerate().collect::<Vec<_>>();
                    let mut new_order = new_base.scores.iter().enumerate().collect::<Vec<_>>();

                    old_order.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
                    new_order.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());

                    //let order_maintained = old_order.into_iter().zip(new_order).all(|((a,_), (b,_))| a==b);

                    //assert!(order_maintained);
                }
            }
        }


        /*let mut failures = 0_usize;
        let (single_mot, _ln_prop) = loop { if let Some(r) = motif_set.propose_ordered_motif_move_custom(&mut rng, 0.1, 0.1) { break r; } else {println!("fail base {failures}"); failures+=1;}};

        println!("failed in single motif move {} times before succeeding", failures);
        assert!(single_mot.set.len() == motif_set.set.len());
        let mut found_change: bool = false;
        for i in 0..motif_set.set.len() {
            assert!(motif_set.nth_motif(i).peak_height() == single_mot.nth_motif(i).peak_height());
            let unchanged = motif_set.nth_motif(i).pwm.iter().zip(single_mot.nth_motif(i).pwm.iter()).all(|(a, b)| a.dist_sq(Some(b)).sqrt() < 1e-6);
            if !unchanged {
                if found_change {
                    panic!("multiple motifs changed in single base scale move!");
                } else { 
                    found_change = true; 

                    let changed_bases = motif_set.nth_motif(i).pwm.iter().zip(single_mot.nth_motif(i).pwm.iter()).enumerate().filter(|(_, (a, b))| a.dist_sq(Some(*b)).sqrt() >= 1e-6).map(|(i,_)| i).collect::<Vec<_>>();

                    println!("{}", changed_bases.len());

                    for j in 0..changed_bases.len(){
                        let old_base = motif_set.nth_motif(i).pwm[changed_bases[j]].clone();
                        let new_base = single_mot.nth_motif(i).pwm[changed_bases[j]].clone();

                        println!("{:?} {:?} {:?} {:?}", old_base, old_base.base_to_vect(), new_base, new_base.base_to_vect());
                        let mut old_order = old_base.scores.iter().enumerate().collect::<Vec<_>>();
                        let mut new_order = new_base.scores.iter().enumerate().collect::<Vec<_>>();

                        old_order.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
                        new_order.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());

                        //let order_maintained = old_order.into_iter().zip(new_order).all(|((a,_), (b,_))| a==b);

                        //assert!(order_maintained);
                    }
                }
            }
        }

        */
        
        let mut failures = 0_usize;
        
        let single_height = motif_set.propose_height_move_custom(&mut rng, 1.0);

        assert!(single_height.is_some()); //scaling a single base should always make possible motifs

        let (single_mot, _ln_prop) = single_height.unwrap();


        assert!(single_mot.set.len() == motif_set.set.len());
        let mut found_change: bool = false;
        for i in 0..motif_set.set.len() {
            let height_unchanged = motif_set.nth_motif(i).peak_height() == single_mot.nth_motif(i).peak_height();

            if found_change { 
                assert!(height_unchanged, "Multiple heights changing");
            } else  {
                if !height_unchanged { found_change = true; }
            }

            let unchanged = motif_set.nth_motif(i).pwm.iter().zip(single_mot.nth_motif(i).pwm.iter()).all(|(a, b)| a.dist_sq(Some(b)).sqrt() < 1e-6);
            assert!(unchanged, "Changing motif during height move!");
        }

        assert!(found_change, "no height changed during height move!");
        //propose_new_motif<R: Rng + ?Sized>(&self, rng: &mut R ) -> Option<(Self, f64)>

        let birthed = motif_set.propose_new_motif(&mut rng);

        assert!(birthed.is_some());//proposing a new motif should NEVER create an impossible motif

        let (birth_mot, ln_prop) = birthed.expect("Birth should never produce an impossible set!");
        assert!(birth_mot.set.len() == (motif_set.set.len()+1));

        let l = birth_mot.set.len()-1;
        for i in 0..motif_set.set.len() {
            assert!(motif_set.nth_motif(i).peak_height() == birth_mot.nth_motif(i).peak_height());
        }

        println!("birth mot {:?}", birth_mot);
        println!("ln_prop {ln_prop} {}", birth_mot.calc_ln_post());
        let should_prior = ln_prop-(birth_mot.calc_ln_post());

        let remaining = motif_set.data_ref.data()-&motif_set.signal;
        //let pick_prob = (birth_mot.nth_motif(l).len() as f64)*(-(BASE_L as f64).ln() - ((BASE_L-1) as f64)*(-SCORE_THRESH).ln());

        let new_mot = birth_mot.nth_motif(l);

        const MINMER_MASK: u64 = (1_u64 << ((MIN_BASE * 2) as u64)) - 1;

        let minmer_choice = Sequence::kmer_to_u64(&new_mot.best_motif()) & MINMER_MASK;

        let num = motif_set.data_ref.data().seq().all_kmers_start_minmer(minmer_choice, new_mot.len()).len();

         
        let pick_motif_prob = motif_set.data_ref.propensity_minmer(minmer_choice).ln()-((MAX_BASE+1-MIN_BASE) as f64).ln() + (num as f64).ln();

        let pick_prob = new_mot.pwm.iter().map(|a| a.scores.iter().map(|&a| if a < 0.0 {BASE_CHOOSE_DIST.energy_ln_pmf(a)} else {0.0}).sum::<f64>()).sum::<f64>();

        let actual_prior = HEIGHT_PROPOSAL_DIST.ln_pdf(birth_mot.nth_motif(l).peak_height()-MIN_HEIGHT)+pick_prob+pick_motif_prob;

        println!("actuals {pick_motif_prob} {pick_prob} {actual_prior}");
        println!("{should_prior} {actual_prior} asd");
        assert!((should_prior+actual_prior).abs() < 1e-6, "{}", format!("{}", should_prior+actual_prior).as_str());

        println!("done birth");
        //fn propose_kill_motif<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<(Self, f64)>

        let killed = motif_set.propose_kill_motif(&mut rng);

        assert!(killed.is_some());//proposing to kill a motif when we have multiple to pick from should always work 

        let (death_mot, ln_prop) = killed.expect("Death should not produce an impossible set when it has multiple motifs to kill!");
        assert!(death_mot.set.len() == (motif_set.set.len()-1));

        assert!(death_mot.propose_kill_motif(&mut rng).is_none(),"Death should be impossible if there's only one motif!");

        let mut l = motif_set.set.len()-1;
        let mut found_off = false;
        for i in 0..death_mot.set.len() {
            let j = if found_off {i+1} else {i};
            let mut matching = motif_set.nth_motif(j).peak_height() == death_mot.nth_motif(i).peak_height();
            if !matching {
                assert!(!found_off, "Found two mismatches when looking for deleted motif!");
                found_off = true;
                l = j;
                matching = motif_set.nth_motif(j+1).peak_height() == death_mot.nth_motif(i).peak_height();
            }
            assert!(matching, "there is a mismatch!");
        }

        let should_prior = ln_prop-(death_mot.calc_ln_post());

        let remaining = motif_set.data_ref.data()-&death_mot.signal;
        let propensities = remaining.kmer_propensities(motif_set.nth_motif(l).len());
        //let pick_prob = (motif_set.nth_motif(l).len() as f64)*(-(BASE_L as f64).ln() - ((BASE_L-1) as f64)*(-SCORE_THRESH).ln());

        let new_mot = motif_set.nth_motif(l);



        let minmer_choice = Sequence::kmer_to_u64(&new_mot.best_motif()) & MINMER_MASK;

        let num = motif_set.data_ref.data().seq().all_kmers_start_minmer(minmer_choice, new_mot.len()).len();


        let pick_motif_prob = motif_set.data_ref.propensity_minmer(minmer_choice).ln()-((MAX_BASE+1-MIN_BASE) as f64).ln() + (num as f64).ln();

        let pick_prob = new_mot.pwm.iter().map(|a| a.scores.iter().map(|&a| if a < 0.0 {BASE_CHOOSE_DIST.energy_ln_pmf(a)} else {0.0}).sum::<f64>()).sum::<f64>();

        let actual_prior = HEIGHT_PROPOSAL_DIST.ln_pdf(birth_mot.nth_motif(l).peak_height()-MIN_HEIGHT)+pick_prob+pick_motif_prob;



        println!("priors {} {} {} {} {}", motif_set.nth_motif(l).height_prior(data_seq.height_dist()), should_prior, pick_prob, ((MAX_BASE+1-MIN_BASE) as f64).ln(), propensities[motif_set.data_ref.data().seq().id_of_u64_kmer_or_die(motif_set.nth_motif(l).len(),Sequence::kmer_to_u64(&motif_set.nth_motif(l).best_motif()))]);

        //Remember, we can sometimes have a motif that's impossible to kill because it's impossible to be created
        assert!((should_prior == -f64::INFINITY && actual_prior == -f64::INFINITY) || ((should_prior-actual_prior).abs() < 1e-6), "{}", format!("{}", should_prior-actual_prior).as_str());

        //fn propose_extend_motif<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<(Self, f64)>

        let mut track = None;
        let mut tracker = 0_usize;
        while let None = track {
            println!("extend");
            track = motif_set.propose_extend_motif(&mut rng);
            tracker += 1;
            if tracker >= 1000 {break;}
        };

        if let Some(track_d) = track {
        let (extend_mot, ln_prop) = track_d; 

        println!("{:?}", extend_mot);

        assert!(extend_mot.set.len() == motif_set.set.len());

        let mut l: Option<usize> = None;
        let mut found_off = false;
        for i in 0..extend_mot.set.len() {
            assert!(motif_set.nth_motif(i).peak_height() == extend_mot.nth_motif(i).peak_height());
            let mut matching = motif_set.nth_motif(i).len() == extend_mot.nth_motif(i).len();
            if !matching {
                assert!(!found_off, "Found two mismatches when looking for extended motif!");
                assert!(sequence.kmer_in_seq(&(extend_mot.nth_motif(i).best_motif())), "Extension can produce illegal kmer!");
                found_off = true;
                l = Some(i);
                matching = (motif_set.nth_motif(i).len()+1) == extend_mot.nth_motif(i).len();
                matching = motif_set.nth_motif(i).pwm.iter()
                    .zip(extend_mot.nth_motif(i).pwm[0..(extend_mot.nth_motif(i).len()-1)].iter())
                    .map(|(a, b)| *a == *b) //We implemented a fuzzy equality for partialeq of bases
                    .fold(matching, |acc, b| acc && b);

            }
            assert!(matching, "there is a mismatch {} {}!", i, found_off);
        }

        println!("something");
        assert!(l.is_some(), "Not finding a changed motif!");

        let should_prior = ln_prop-(extend_mot.calc_ln_post());

        let actual_prior = extend_mot.nth_motif(l.unwrap()).pwm.last().expect("We know this is bigger than 0").scores.iter().map(|&a| if a < 0.0 {BASE_CHOOSE_DIST.energy_ln_pmf(a)} else {0.0}).sum::<f64>(); 

        println!("hanging");
        assert!((should_prior+actual_prior).abs() < 1e-6, "{}", format!("{}", should_prior+actual_prior).as_str());

        }
        let cant_extend_set = MotifSet::rand_with_one_height_and_motif_len(9.6, MAX_BASE, &data_seq, &mut rng);

        assert!(cant_extend_set.propose_extend_motif(&mut rng).is_none(), "Can extend PWM beyond allowed limits!");


        //fn propose_contract_motif<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<(Self, f64)>

        let mut track = None;
        let mut tracker = 0_usize;

        while let None = track {
            println!("contract");
            track = motif_set.propose_contract_motif(&mut rng);
            tracker += 1;
            if tracker >= 1000 {break;}

        };
            
        if let Some(track_d) = track {

        let (contract_mot, ln_prop) = track_d;

        println!("{:?}", contract_mot);

        assert!(contract_mot.set.len() == motif_set.set.len());

        let mut l: Option<usize> = None;
        let mut found_off = false;
        for i in 0..contract_mot.set.len() {
            assert!(motif_set.nth_motif(i).peak_height() == contract_mot.nth_motif(i).peak_height());
            let mut matching = motif_set.nth_motif(i).len() == contract_mot.nth_motif(i).len();
            if !matching {
                assert!(!found_off, "Found two mismatches when looking for contracted motif!");
                found_off = true;
                l = Some(i);
                matching = (motif_set.nth_motif(i).len()-1) == contract_mot.nth_motif(i).len();
                matching = contract_mot.nth_motif(i).pwm.iter()
                    .zip(motif_set.nth_motif(i).pwm[0..(contract_mot.nth_motif(i).len()-1)].iter())
                    .map(|(a, b)| *a == *b) //We implemented a fuzzy equality for partialeq of bases
                    .fold(matching, |acc, b| acc && b);

            }
            assert!(matching, "there is a mismatch {} {}!", i, found_off);
        }

        assert!(l.is_some(), "Not finding a changed motif!");

        let should_prior = ln_prop-(contract_mot.calc_ln_post());

        let actual_prior = (motif_set.nth_motif(l.unwrap()).pwm.last().expect("We know this is bigger than 0")).scores.iter().map(|&a| if a < 0.0 {BASE_CHOOSE_DIST.energy_ln_pmf(a)} else {0.0}).sum::<f64>();

        assert!((should_prior-actual_prior).abs() < 1e-6, "{}", format!("{}", should_prior-actual_prior).as_str());
        }
        let cant_contract_set = MotifSet::rand_with_one_height_and_motif_len(2.6, MIN_BASE, &data_seq, &mut rng);

        assert!(cant_contract_set.propose_contract_motif(&mut rng).is_none(), "Can contract PWM beyond allowed limits!");



        //Testing full RJ move
        for _ in 0..3{
            let add_mot = Motif::rand_mot_with_height_and_motif_len(3.2,mid_mot, data_seq.data().seq(), &mut rng);
            _ = motif_set.add_motif(add_mot);
        }
        for i in 0..100 {
            let (step_set, selected_move, accepted) = motif_set.run_rj_move(1.0, &mut rng);
            println!("{i} {selected_move} check");
            //For when run_rj_move gives a clone of the old set when rejecting, rather than telling us what could have been
            /*match accepted {

                //Yes, I'm asserting float equality here: this should be a clone, not a recalculation
                false => assert!(motif_set.calc_ln_post() == step_set.calc_ln_post(), "Not cloning old set when rejecting move?!"),
                true => { println!("{selected_move} accepted"); match selected_move {
                    0 => assert!(step_set.set.len() == motif_set.set.len()+1, "birth not birthing"),
                    1 => assert!(step_set.set.len() == motif_set.set.len()-1, "death not deathing"),
                    2 => {
                        assert!(step_set.set.len() == motif_set.set.len(), "extend not maintaining");
                        assert!(step_set.set.iter().zip(motif_set.set.iter())
                                .map(|(a,b)| (a.pwm.len() == b.pwm.len()) || (a.pwm.len() == b.pwm.len()+1))
                                .fold(true, |acc, y| acc && y), "extend messed up extending" );
                    },
                    3 => {
                        assert!(step_set.set.len() == motif_set.set.len(), "contract not maintaining");
                        assert!(step_set.set.iter().zip(motif_set.set.iter())
                                .map(|(a,b)| (a.pwm.len() == b.pwm.len()) || (a.pwm.len() == b.pwm.len()-1))
                                .fold(true, |acc, y| acc && y), "contract messed up contracting" );
                    },
                    _ => assert!(false, "Picking an impossible move!"),
                };},
            };*/

            if accepted {
                println!("{selected_move} accepted"); 
                match selected_move {
                    0 => assert!(step_set.set.len() == motif_set.set.len()+1, "birth not birthing"),
                    1 => assert!(step_set.set.len() == motif_set.set.len()-1, "death not deathing"),
                    2 => {
                        assert!(step_set.set.len() == motif_set.set.len(), "extend not maintaining");
                        assert!(step_set.set.iter().zip(motif_set.set.iter())
                                .map(|((a,_),(b,_))| (a.pwm.len() == b.pwm.len()) || (a.pwm.len() == b.pwm.len()+1))
                                .fold(true, |acc, y| acc && y), "extend messed up extending" );
                    },
                    3 => {
                        assert!(step_set.set.len() == motif_set.set.len(), "contract not maintaining");
                        assert!(step_set.set.iter().zip(motif_set.set.iter())
                                .map(|((a, _),(b, _))| (a.pwm.len() == b.pwm.len()) || (a.pwm.len() == b.pwm.len()-1))
                                .fold(true, |acc, y| acc && y), "contract messed up contracting" );
                    },
                    4 => assert!(step_set.set.len() == motif_set.set.len()+1, "split not birthing"),
                    5 => assert!(step_set.set.len() == motif_set.set.len()-1, "merge not deathing"),
                    _ => assert!(false, "Picking an impossible move!"),
                };
            }
        }
    }

    #[test]
    fn it_works() {
        let mut rng = rand::thread_rng();
        let base = Bp::T;
        let try_base: Base = Base::rand_new(&mut rng);
        let b = try_base.make_best(base);
        println!("{:?}", try_base);
        assert_eq!(base, b.best_base());

        let bc: Base = b.rev();

        assert_eq!(bc.best_base(), base.complement());

        let tc: Base = try_base.rev();

        assert_eq!(tc.best_base(), try_base.best_base().complement());

        assert!(!(tc == try_base));
        assert!(b == b.clone());

        let _b_mag: f64 = b.scores().iter().sum();
        let supposed_default_dist = (b.base_to_vect()).iter().map(|a| a.powi(2)).sum::<f64>();

        println!("{supposed_default_dist}, {}", b.dist_sq(None));
        assert!((supposed_default_dist - b.dist_sq(None)).abs() < 1e-7);

        for i in 0..1000 {

            let try_base: Base = Base::rand_new(&mut rng);

            let to_pwm = try_base.to_pwm(None).unwrap();

            let from_pwm = Base::from_pwm(to_pwm, None, false).unwrap();

            let from_pwm_discrete = Base::from_pwm(to_pwm, None, true).unwrap();

            let dist = try_base.dist_sq(Some(&from_pwm));

            println!("{i} original {:?} to pwm {:?} \n from_pwm {:?}\n disc_pwm {:?} \n {dist}", try_base, to_pwm, from_pwm, from_pwm_discrete);

            assert!(dist.sqrt() < 1e-3, "dist {i} failed");

            let mut stick_break: [f64; BASE_L] = rng.gen();

            stick_break.sort_unstable_by(|a,b| a.partial_cmp(&b).unwrap());

            let like_pwm: [f64; BASE_L] = [stick_break[0], stick_break[1]-stick_break[0], stick_break[2]-stick_break[1], 1.0-stick_break[2]];

            let to_base = Base::from_pwm(like_pwm, None, false).unwrap();

            let to_base_disc = Base::from_pwm(like_pwm, None, true).unwrap();

            let back_pwm = to_base.to_pwm(None).unwrap();

            let dist_pwm = (0..4).map(|i| (back_pwm[i]-like_pwm[i]).powi(2)).sum::<f64>().sqrt();

            println!("{i} original pwm {:?} \n from pwm {:?}\n disc pwm {:?}\n to pwm {:?} {dist_pwm}", like_pwm, to_base, to_base_disc, back_pwm);


            assert!(dist_pwm < 1e-3, "dist {i} failed");

        }
        //println!("Conversion dists: {:?}, {:?}, {}", b.scores(),  b.to_gbase().to_base().scores(), b.dist_sq(Some(&b.to_gbase().to_base())));
        //assert!(b == b.to_gbase().to_base());


        //let td: Base = Base::new([0.1, 0.2, 0.4, 0.3]);

        //assert!((td[Bp::C]-0.5_f64).abs() < 1e-6);
        //assert!((td[Bp::G]-1_f64).abs() < 1e-6);

        //let tg: GBase = GBase::new([0.82094531732, 0.41047265866, 0.17036154577], 2);


        //assert!(tg.to_base() == td);

    }

    #[test]
    fn trun_ln_normal_tests() {

        let dist: TruncatedLogNormal = TruncatedLogNormal::new(LOG_HEIGHT_MEAN, LOG_HEIGHT_SD, MIN_HEIGHT, MAX_HEIGHT).unwrap();

        let free_dist: LogNormal = LogNormal::new(LOG_HEIGHT_MEAN, LOG_HEIGHT_SD).unwrap();

        assert!((dist.pdf(6.0)-dist.ln_pdf(6.0).exp()).abs() < 1e-6);

        let mut rng = rand::thread_rng();

        let mut h: f64;
        for _i in 0..30000 { h = dist.sample(&mut rng); assert!((h >= dist.min()) && (h <= dist.max()))}

        let val1: f64 = rng.gen_range(MIN_HEIGHT..MAX_HEIGHT);
        let val2: f64 = rng.gen_range(MIN_HEIGHT..MAX_HEIGHT);

        println!("{} {} {} {}", free_dist.pdf(val1)/free_dist.pdf(val2),dist.pdf(val1)/dist.pdf(val2), free_dist.cdf(val1)/free_dist.cdf(val2), dist.cdf(val1)/dist.cdf(val2));
        assert!(((free_dist.pdf(val1)/free_dist.pdf(val2))-(dist.pdf(val1)/dist.pdf(val2))).abs() < 1e-6);
        assert!(((free_dist.cdf(val1)/free_dist.cdf(val2))-(dist.cdf(val1)/dist.cdf(val2))).abs() < 1e-6);

        assert!(dist.ln_pdf(MAX_HEIGHT+1.0).is_infinite() && dist.ln_pdf(MAX_HEIGHT+1.0) < 0.0);

        let dist: TruncatedLogNormal = TruncatedLogNormal::new(LOG_HEIGHT_MEAN, LOG_HEIGHT_SD, -1.0, MAX_HEIGHT).unwrap();

        assert!(dist.min().abs() < 1e-6);

    } 

    #[test]
    fn motif_establish_tests() {

        std::env::set_var("RUST_BACKTRACE", "1");

        println!("rayon {:?}", rayon::current_num_threads());

        let mut rng = rand::thread_rng(); //fastrand::Rng::new();
        let spacing_dist = rand::distributions::Uniform::from(500..5000);
        let block_n: usize = 20;
        let u8_per_block: usize = 5000;
        let bp_per_block: usize = u8_per_block*4;
        let bp: usize = block_n*bp_per_block;
        let u8_count: usize = u8_per_block*block_n;

        println!("DF {}", block_n*bp_per_block);
        let start_gen = Instant::now();
        //let blocks: Vec<u8> = (0..u8_count).map(|_| rng.u8(..)).collect();
        //let preblocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.u8(..)).collect();
        
        let preblocks: Vec<u8> = (0..(u8_count/block_n)).map(|_| rng.gen::<u8>()).collect();
        let blocks: Vec<u8> = preblocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>(); 
        let block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
        let mut start_bases: Vec<usize> = (0..block_n).map(|a| a*bp_per_block).collect();
        let sequence: Sequence = Sequence::new_manual(blocks, block_lens.clone(), &((0..block_n).map(|i| (i, 0, (bp_per_block-MAX_BASE))).collect::<Vec<_>>()) );
        

        //let pre_null_blocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.gen::<u8>()).collect();
        //let null_blocks: Vec<u8> = pre_null_blocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>(); 
        let null_blocks: Vec<u8> = (0..(u8_count)).map(|_| rng.gen::<u8>()).collect();
        let _block_u8_null_starts: Vec<usize> = (0..block_n).map(|a| a*u8_per_block).collect();
        let null_block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
        let start_null_bases: Vec<usize> = (0..block_n).map(|a| 2*a*bp_per_block).collect();
        let null_seq: NullSequence = NullSequence::new_manual(null_blocks, null_block_lens.clone());

        let start_bases_copy: Vec<_> = (0..block_n).map(|a| (2*a+1)*bp_per_block+10).collect();


        let wave: Waveform = Waveform::create_zero(&sequence, 5);
        let duration = start_gen.elapsed();
        println!("grad gen {} bp {:?}", bp, duration);

        let background = Background::new(0.25, 2.64, 20.).unwrap();

        let base_seq = unsafe{ AllData::create_manual(sequence.clone(), null_seq.clone(), WaveformDef::from(&wave), start_bases_copy.clone(), start_null_bases.clone(), background.clone(),3.0, 4.126)};

        let data_seq = unsafe{ AllDataUse::new_unchecked_data(wave.clone(),&null_seq, &start_bases_copy,&start_null_bases, &background, 3.0, 4.126)};
        println!("Done gen {} bp {:?}", bp, duration);

        println!("{} gamma", gamma(4.));
        println!("{} gamma", ln_gamma(4.));

        let mut buffer: Vec<u8> = bincode::serde::encode_to_vec(&base_seq, config::standard()).expect("serializable");

        let (other_data, _bytes): (AllData, usize) = bincode::serde::decode_from_slice(&buffer, config::standard()).unwrap();

        let mut check_kmers: Vec<u64> = vec![0; 10000];
        rng.fill(check_kmers.as_mut_slice());

        for i in MIN_BASE..=MAX_BASE {
        
            let dict = base_seq.seq().unique_kmers_ref(i);
            let dict2 = other_data.seq().unique_kmers_ref(i);
            assert!(dict.iter().zip(dict2.iter()).all(|(i, j)| i == j));
            for &kmer in check_kmers.iter() {
                let adjust_kmer = kmer & ((1_u64 << ((i*BITS_PER_BP) as u64))-1);
                assert!(base_seq.seq().id_of_u64_kmer(i, adjust_kmer) == other_data.seq().id_of_u64_kmer(i, adjust_kmer));
            }
            assert!(base_seq.seq().number_unique_kmers(i) == other_data.seq().number_unique_kmers(i));
            println!("kmer dict check {i}");
        }


        //println!("{:?}", wave.raw_wave());

        let mut motif: Motif = Motif::from_motif_alt(sequence.return_bases(0,0,MIN_BASE+6), data_seq.height_dist(), &mut rng); //sequence

        let motif2: Motif = Motif::from_motif_alt(sequence.return_bases(0,2,MAX_BASE), data_seq.height_dist(), &mut rng); //sequence

        let start = Instant::now();

        let waveform = motif.generate_waveform(&data_seq);
        let duration = start.elapsed();

        let start = Instant::now();
        let waveform_alt = motif.generate_waveform_alt(&data_seq);
        let duration_alt = start.elapsed();

        println!("previous gen wave {:?} alt gen wave {:?}", duration, duration_alt);

        let waveform2 = &waveform + &(motif2.generate_waveform(&data_seq));

        let coordinate_bases: Vec<usize> = start_bases.iter().map(|&a| a+spacing_dist.sample(&mut rng)).collect();

        let data_seq_2 = unsafe{ AllDataUse::new_unchecked_data(waveform2, &null_seq, &coordinate_bases,&start_null_bases ,&background, 3.0, 4.126) }; 

        let _noise: Noise = waveform.produce_noise(&data_seq_2);


        for _ in 0..1000 {
            let len = rng.gen_range(MIN_BASE+1..=MAX_BASE);
            let motif: u64 = rng.gen_range(0_u64..4_u64.pow(len as u32));
            let rev_mot = crate::sequence::reverse_complement_u64_kmer(motif, len);


            let red_mot = motif & (4_u64.pow((len-1) as u32)-1);
            
            
            let check = null_seq.kmer_count(motif, len);
            if check.is_some() {
                println!("checking less");
                assert!(null_seq.kmer_count(red_mot, len-1).unwrap() >= check.unwrap());
            }

            assert!(check == null_seq.kmer_count(rev_mot, len));
        }
            
                

        let block_lens = null_seq.block_lens().clone();

        for (i, blen) in block_lens.into_iter().enumerate() {

            for mot_len in MIN_BASE..=MAX_BASE {

                for j in 0..(blen-mot_len){
                let mot_check = Sequence::kmer_to_u64(&null_seq.return_bases(i, j, mot_len));
                assert!(null_seq.kmer_count(mot_check, mot_len).is_some());

                }
            }


        }


        let waveform_raw = waveform.raw_wave();

        let binds = motif.return_bind_score(&sequence);

        let null_blocks: Vec<usize> = (0..null_block_lens.len()).collect();

        let old_height = motif.peak_height();

        for h in 2..12 {
            motif.peak_height = h as f64;

        let start_null = Instant::now();
        let null_binds = motif.return_any_null_binds_by_hamming(&null_seq, data_seq.min_height(),0.0);
        let duration_null = start_null.elapsed();

        println!("height {} null binds {:?}", motif.peak_height(), null_binds);
        println!("null dur {:?}", duration_null);

        }

        motif.peak_height = old_height;
        //let start_b = Instant::now();
        //let unsafe_waveform = unsafe{ motif.generate_waveform_from_binds(&binds, &data_seq) };
        //let duration_b = start_b.elapsed();



       // let unsafe_raw = unsafe_waveform.raw_wave();

       // assert!(unsafe_raw.len() == waveform_raw.len());
        //assert!((unsafe_raw.iter().zip(&waveform_raw).map(|(a, &b)| (a-b).powf(2.0)).sum::<f64>()).powf(0.5) < 1e-6);

        println!("Time elapsed in generate_waveform() is: {:?}", duration);
      //  println!("Time elapsed in the unsafe generate_waveform() is: {:?}", duration_b);

        println!("{}", motif);

        let random_motif = Motif::rand_mot(&sequence, data_seq.height_dist(), &mut rng);

        println!("Random motif\n{}", random_motif);

        println!("Kernel length {}", background.kernel_ref(KernelWidth::Wide, KernelVariety::Gaussian).len());

        assert!(background.kernel_ref(KernelWidth::Wide, KernelVariety::Gaussian).len() == 121);

        assert!((random_motif.peak_height.abs() >= MIN_HEIGHT) && (random_motif.peak_height.abs() <= MAX_HEIGHT));

        let matrix = motif.pwm();
        let _nbases = matrix.len();

        for base in matrix {
            println!("{:?}", base.scores());
        }

        println!("{:?}", motif.best_motif());

        println!("{:?}", sequence.return_bases(0,0,MAX_BASE));
        //println!("{:?}", sequence.unique_kmers(motif.len()));
        println!("{}", Sequence::kmer_to_u64(&motif.best_motif()));
        let matrix = motif.rev_complement();

        for base in &matrix {
            println!("{:?}", base.scores());
        }

        //assert!(((motif.pwm_prior()/gamma::ln_gamma(BASE_L as f64))+(motif.len() as f64)).abs() < 1e-6);

        //println!("{} {} {} PWM PRIOR",sequence.kmer_in_seq(&motif.best_motif()), motif.pwm_prior(&sequence), ((MAX_BASE+1-MIN_BASE) as f64).ln()-(sequence.number_unique_kmers(motif.len()) as f64).ln() - motif.pwm_ref().iter().map(|a| BASE_PRIOR.ln_pdf(a)).sum::<f64>());
        assert!((motif.pwm_prior(&sequence)-(sequence.number_unique_kmers(motif.len()) as f64).ln()+((MAX_BASE+1-MIN_BASE) as f64).ln()
                 -((motif.len() as f64)*(BASE_RESOLUTION.ln() * ((BASE_L-1) as f64) ))).abs() < 1e-6);

        let un_mot: Motif = Motif::from_motif(vec![Bp::C;MAX_BASE],data_seq.height_dist(),&mut rng);//Sequence

        assert!(un_mot.pwm_prior(&sequence) < 0.0 && un_mot.pwm_prior(&sequence).is_infinite());

        let dist: TruncatedLogNormal = TruncatedLogNormal::new(LOG_HEIGHT_MEAN, LOG_HEIGHT_SD, MIN_HEIGHT, MAX_HEIGHT).unwrap();

        let mut base_prior = motif.height_prior(&dist);
        if motif.peak_height() > 0.0 {
            base_prior -= PROB_POS_PEAK.ln();
        } else {
            base_prior -= (1.0-PROB_POS_PEAK).ln();
        }

        println!("{} {} pro", base_prior.exp(), dist.pdf(motif.peak_height().abs()));
        assert!((base_prior.exp()-dist.pdf(motif.peak_height().abs())).abs() < 1e-6);


        let best_mot = motif.best_motif();

        let bindy = unsafe{ motif.prop_binding(&best_mot) };

        println!("bindy {:?}", bindy);
        assert!(((bindy.0-0.0).abs() < 1e-6) && !bindy.1);

        let rev_best = best_mot.iter().rev().map(|a| a.complement()).collect::<Vec<Bp>>();

        let bindy = unsafe {motif.prop_binding(&rev_best) };

        assert!(((bindy.0-0.0).abs() < 1e-6) && bindy.1);

        let pwm = motif.pwm();

        
        for i in 0..motif.len() {
            for j in BP_ARRAY {

                let mut for_mot = best_mot.clone();
                for_mot[i] = j;
                let mut rev_mot = rev_best.clone();
                rev_mot[motif.len()-1-i] = j.complement();

                let defect: f64 = pwm[i][j] ;

                let bindy = unsafe{ motif.prop_binding(&for_mot)};
                let rbind = unsafe{ motif.prop_binding(&rev_mot) };


                assert!(((bindy.0-defect).abs() < 1e-6) && !bindy.1);
                assert!(((rbind.0-defect).abs() < 1e-6) && rbind.1);

                //println!("for pre {:?}, rev pre {:?}", for_mot, rev_mot);
                for k in 0..motif.len() {
                    if k != i {
                        for m in BP_ARRAY {

                            let defect_2 = defect+pwm[k][m];
                            let mut for_mot_2 = for_mot.clone();
                            for_mot_2[k] = m;
                            let mut rev_mot_2 = rev_mot.clone();
                            rev_mot_2[motif.len()-1-k] = m.complement();

                            //println!("for {k} {m} {:?}, rev {k} {m} {:?}", for_mot_2, rev_mot_2);
                            let bindy = unsafe{ motif.prop_binding(&for_mot_2)};
                            let rbind = unsafe{ motif.prop_binding(&rev_mot_2) };

                            println!("defect {defect} pos {}, defect_2 {defect_2} bindy {:?} rbind {:?}",pwm[k][m], bindy, rbind);
                            if bindy.1 {
                                assert!(defect_2 < bindy.0);
                                assert!(defect_2 < rbind.0);
                                assert!((bindy.0-rbind.0).abs() < 1e-6);
                                assert!(!rbind.1);
                                println!("SOME REVERSAL HERE");
                            }
                            else{
                                assert!(((bindy.0-defect_2).abs() < 1e-6) && !bindy.1);
                                assert!(((rbind.0-defect_2).abs() < 1e-6) && rbind.1);
                            }
                        }
                    }
                }
            
            }
        }

        let rand_pwm = random_motif.pwm();
        for _ in 0..100 {
        
            let altermot = data_seq.data().seq().random_valid_motif(random_motif.pwm().len(), &mut rng);

            let forward_defect = altermot.iter().enumerate().map(|(i, &b)| rand_pwm[i][b]).sum::<f64>();
            let reverse_defect = altermot.iter().rev().enumerate().map(|(i, &b)| rand_pwm[i][b.complement()]).sum::<f64>();

            let is_rev = reverse_defect > forward_defect;

            let proper_defect = if is_rev {reverse_defect} else {forward_defect};

            //println!("alter for {forward_defect} rev {reverse_defect} is_rev {is_rev}");

            let bindy = unsafe{ random_motif.prop_binding(&altermot)};

            //println!("bindy  {:?}", bindy);
            assert!(((bindy.0-proper_defect).abs() < 1e-6) && (bindy.1 == is_rev));

        }
       
        println!("Motif {:?}", motif);

        let mut binds = motif.return_any_null_binds_by_hamming(&null_seq, data_seq.min_height(), 0.0);
        binds.sort_unstable_by(|b, a| a.partial_cmp(b).unwrap());
        let mut significant_binds: Vec<f64> = Vec::with_capacity(null_block_lens[0]*50);

        let mut binds_checker: Vec<(usize, usize, usize, String, bool, f64)> = Vec::with_capacity(null_block_lens[0]*50);

        for i in 0..null_block_lens.len() {
        

            let group = [i];


            //println!("initial bases {:?}", null_seq.return_bases(i, 0, 45));

            println!("null {}", null_block_lens[i]);
            for j in 0..(null_block_lens[i]-motif.len()) {

                let mot = null_seq.return_bases(i, j, motif.len());
                let (prop, rev) = unsafe{ motif.prop_binding(&mot)};

                if prop > (-data_seq.min_height()+0.01) { 
                
                    println!("sig");
                    significant_binds.push(prop+motif.peak_height); 
                    let mot_string: String = mot.iter().map(|&a| match a {
                        Bp::A => 'A',
                        Bp::C => 'C',
                        Bp::G => 'G',
                        Bp::T => 'T',
                    }).collect();

                    let mot_u64 = Sequence::kmer_to_u64(&mot);
                    println!("{i} {j} {rev} {:?} {} {:?}", mot, mot_u64, (MIN_BASE..=motif.len()).map(|l| null_seq.kmer_count(mot_u64 & (4_u64.pow(l as u32)-1), l)).collect::<Vec<_>>());
                    assert!(null_seq.kmer_count(mot_u64, motif.len()).is_some());
                    println!("{:?}", null_seq.kmer_count(mot_u64, motif.len()));

                    binds_checker.push((i, j, null_block_lens[i],mot_string, rev, prop+motif.peak_height));
                    //println!("sig calc {i} {j} prop {prop} seq {:?}", mot);
                }

            }


        }
        significant_binds.sort_unstable_by(|b, a| a.partial_cmp(b).unwrap());

        binds_checker.sort_unstable_by(|b,a| a.5.partial_cmp(&b.5).unwrap());

        if significant_binds.len() > CAPACITY_FOR_NULL {
            let _ = significant_binds.drain(CAPACITY_FOR_NULL..).collect::<Vec<_>>();
            let _ = binds_checker.drain(CAPACITY_FOR_NULL..).collect::<Vec<_>>();
        }


        println!("binds {:?}\n sig_binds {:?} \nheight {} width {}", binds, significant_binds, motif.peak_height, motif.len());
        println!("binds_checker {:#?}", binds_checker);

        assert!(binds.len() == significant_binds.len(), "bind scores giving different lengths in null seq");
        for k in 0..binds.len() {

            assert!((binds[k]-significant_binds[k]).abs() < 1e-6, "not capturing proper binding {} {} {}", k, binds[k], significant_binds[k]);

        }


        let wave_block: Vec<u8> = vec![2,0,0,0, 170, 170, 170, 170, 170, 170, 170, 170,170, 170, 170, 170, 170, 170, 170]; 
        let _wave_inds: Vec<usize> = vec![0, 9]; 
        let _wave_starts: Vec<usize> = vec![0, 36];
        let wave_lens: Vec<usize> = vec![36, 40];
        let wave_start_bases: Vec<usize> = vec![0, 9324];
        let wave_seq: Sequence = Sequence::new_manual(wave_block, wave_lens.clone(), &((0..wave_lens.len()).map(|i| (i, 0, wave_lens[i])).collect::<Vec<_>>()));
        let wave_wave: Waveform = Waveform::create_zero(&wave_seq,1);
        let wave_background = Background::new(0.25, 2.64, 1.0).unwrap();


        let null_zeros: Vec<usize> = vec![144, 813];

        //This is in units of number of u8s
        let null_sizes: Vec<usize> = vec![78,17];

        let null_makeup: Vec<Vec<usize>> = null_sizes.iter().map(|&a| (0..(4*a)).map(|_| rng.gen_range(0_usize..4)).collect::<Vec<usize>>()).collect::<Vec<Vec<usize>>>();

        let invented_null: NullSequence =  NullSequence::new(null_makeup);



        let wave_data_seq = unsafe{ AllDataUse::new_unchecked_data(wave_wave,&invented_null, &wave_start_bases, &null_zeros, &wave_background,3.0, 4.126) }; 
        let theory_base = [1.0.log2(), 1e-5.log2(), 1e-5.log2(), 0.2_f64.log2()];

        let mat: Vec<Base> = (0..15).map(|_| Base::new(theory_base.clone())).collect::<Vec<_>>();


        println!("DF");
        let little_motif: Motif = Motif::raw_pwm(mat, 10.0, KernelWidth::Wide, KernelVariety::Gaussian); //wave_seq

        print!("{}", little_motif);
        println!("{:?}",little_motif.generate_waveform(&wave_data_seq).raw_wave());

        let small_block: Vec<u8> = vec![44, 24, 148, 240, 84, 64, 200, 80, 68, 92, 196, 144]; 
        let _small_inds: Vec<usize> = vec![0, 6]; 
        let small_lens: Vec<usize> = vec![24, 24];
        let small: Sequence = Sequence::new_manual(small_block, small_lens.clone(),  &((0..small_lens.len()).map(|i| (i, 0, small_lens[i])).collect::<Vec<_>>()));
        let _small_wave: Waveform = Waveform::new(vec![0.1, 0.6, 0.9, 0.6, 0.1, -0.2, -0.4, -0.6, -0.6,-0.1], &small, 5).unwrap();

        let mat: Vec<Base> = (0..15).map(|_| Base::new(theory_base.clone())).collect::<Vec<_>>();
        let wave_motif: Motif = Motif::raw_pwm(mat, 10.0, KernelWidth::Wide, KernelVariety::Gaussian); //small

        let rev_comp: Vec<bool> = (0..48).map(|_| rng.gen::<bool>()).collect();


        let forward: Vec<bool> = vec![true, false, false, true, true, false, false, false, true, true, false, false, true, false, false, false, true, true, true, false, true, false, true, false, true, true, false, false, true, false, true, false, true, false, false, false, true, false, true, false, true, true, false, false, false, false, false, false];

        let reverse: Vec<bool> = vec![true, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

        let _correct: Vec<bool> = rev_comp.iter().enumerate().map(|(a, &b)| if b {reverse[a]} else {forward[a]}).collect();

       // println!("correct: {:?}", correct);

        //println!("small bl: {:?} {:?} {:?} {:?}", small.seq_blocks(), small.block_lens(), small.block_u8_starts(), small.return_bases(0, 0, 24));
        //println!("blocks in seq: {:?}", small.seq_blocks());



        for i in 0..2 {
            for j in 0..24 {

                let ind = if rev_comp[24*i+j] { j+wave_motif.len()-1-4 } else {j+4}; 
                if ind < 24 {
                    let bp = small.return_bases(i, ind, 1)[0];
                    let _bp2 = small.return_bases(i, ind, 1)[0];
                    let _matcher = if rev_comp[24*i+j] { bp == Bp::T } else { bp == Bp::A};
                    //println!("start loc: {}, bp: {:?}, bp2: {:?}, ind: {}, rev: {}, matcher: {},  correct: {}", 24*i+j, bp, bp2, ind, rev_comp[24*i+j], matcher, correct[24*i+j]);
                }
            }
        }


        let start = Instant::now();

        //        let (bind_score_floats, bind_score_revs) = self.return_bind_score(data.seq());

        let binds = motif.return_bind_score(&sequence);

        let duration = start.elapsed();
        println!("Time elapsed in bind_score() is: {:?}", duration);

        let start = Instant::now();
        let duration = start.elapsed();
        println!("Time elapsed in check is: {:?}", duration);


        //TESTING return_bind_score()

        //println!("binds {:?}", binds);
        for i in 0..block_n {
            for j in 0..(bp_per_block-(motif.len()-1)) {

                //let test_against = motif.prop_binding(&VecDeque::from(sequence.return_bases(i, j, motif.len())));
                //SAFETY: sequence.return_bases is called with motif.len() as its length
                //        and always called so that we have that many bps left available
                let test_against = unsafe{ motif.prop_binding(&(sequence.return_bases(i, j, motif.len())))};
                assert!((binds[i*bp_per_block+j]-test_against.0).abs() < 1e-6);
                //assert!((binds.0[i*bp_per_block+j]-test_against.0).abs() < 1e-6);
                //assert!(binds.1[i*bp_per_block+j] == test_against.1);

            }
        }

        //TESTING generate_waveform() 


        let pwm_bp: Bp = Bp::T;
        let pwm_pos: usize = 6;
        let _prop_bp = motif.pwm[pwm_pos][pwm_bp];
        let wave_main = motif.generate_waveform(&data_seq);
        let _start0 = Instant::now();
        let _duration0 = start.elapsed();
        let wave_gen: Vec<f64> = wave_main.raw_wave();
        let mut scores: Vec<f64> = vec![0.0;wave_gen.len()];

        let point_lens = wave_main.point_lens();
        start_bases.push(bp);
        let start_dats = wave_main.start_dats();
        let space: usize = wave_main.spacer();

        let half_len: usize = (motif.len()-1)/2;

        let kernel_check = (background.kernel_ref(motif.kernel_width, motif.kernel_variety)*1.0).get_curve().clone();

        let kernel_mid = (kernel_check.len()-1)/2;

        //println!("STARTS: {:?}", sequence.block_u8_starts().iter().map(|a| a*BP_PER_U8).collect::<Vec<_>>());






        println!("unsafe filter same as filter when properly made");

        //println!("filts {:?}", checked.iter().enumerate().filter(|(_, &b)| b).map(|(a, _)| a).collect::<Vec<_>>());
        for i in 0..block_n {
            for j in 0..point_lens[i] {

                let cut_low = if space*j >= kernel_mid+half_len {start_bases[i]+space*j-(kernel_mid+half_len)} else {start_bases[i]} ;
                let cut_high = if j*space+kernel_mid <= ((start_bases[i+1]+half_len)-start_bases[i]) {space*j+start_bases[i]+kernel_mid+1-half_len} else {start_bases[i+1]};
                let relevant_binds = (cut_low..cut_high).filter(|&a| {
                    //println!("{a} bindslen {}", binds.0.len());
                    //binds.0[a] > (-motif.peak_height()) 
                    binds[a] > (-MIN_HEIGHT) 
                }).collect::<Vec<_>>();


                if relevant_binds.len() > 0 {

                    let mut score: f64 = 0.0;
                    for k in 0..relevant_binds.len() {
                        let to_add = (motif.peak_height()+binds[relevant_binds[k]])*kernel_check[(kernel_mid+(start_bases[i]+space*j))-(relevant_binds[k]+half_len)];
                        //let to_add = (motif.peak_height()+binds.0[relevant_binds[k]])*kernel_check[(kernel_mid+(start_bases[i]+space*j))-(relevant_binds[k]+half_len)];
                        //println!("to_add {to_add}");
                        score+=to_add;
                    }

                    scores[start_dats[i]+j] += score;
                    if (wave_gen[start_dats[i]+j]-score).abs() >= 1e-6{
                        println!("error would have occurred");
                        println!("wave {} score {} ratio {}", wave_gen[start_dats[i]+j], score, wave_gen[start_dats[i]+j]/score); 


                    }
                   // assert!((wave_gen[start_dats[i]+j]-score).abs() < 1e-6);


                } else {
                    assert!(wave_gen[start_dats[i]+j].abs() < 1e-6);
                }


            }
        }

        assert!( scores.iter().zip(wave_gen.iter()).map(|(&a, &b)| (a-b).abs()).all(|a| a < 1e-6));

       

    }



    #[test]
    fn write_async() {

        use gzp::{deflate::Mgzip, par::compress::{ParCompress, ParCompressBuilder}, par::decompress::{ParDecompress, ParDecompressBuilder},ZWriter, Compression, syncz::{SyncZ, SyncZBuilder}};
        use std::fs::File;

        std::env::set_var("RUST_BACKTRACE", "1");

        println!("rayon {:?}", rayon::current_num_threads());

        let mut rng = rand::thread_rng(); //fastrand::Rng::new();
        let spacing_dist = rand::distributions::Uniform::from(500..5000);
        let block_n: usize = 20;
        let u8_per_block: usize = 5000;
        let bp_per_block: usize = u8_per_block*4;
        let bp: usize = block_n*bp_per_block;
        let u8_count: usize = u8_per_block*block_n;

        println!("DF {}", block_n*bp_per_block);
        let start_gen = Instant::now();
        //let blocks: Vec<u8> = (0..u8_count).map(|_| rng.u8(..)).collect();
        //let preblocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.u8(..)).collect();

        let preblocks: Vec<u8> = (0..(u8_count/block_n)).map(|_| rng.gen::<u8>()).collect();
        let blocks: Vec<u8> = preblocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>();
        let block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
        let mut start_bases: Vec<usize> = (0..block_n).map(|a| a*bp_per_block).collect();
        let sequence: Sequence = Sequence::new_manual(blocks, block_lens.clone(),  &((0..block_lens.len()).map(|i| (i, 0, block_lens[i])).collect::<Vec<_>>()));


        //let pre_null_blocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.gen::<u8>()).collect();
        //let null_blocks: Vec<u8> = pre_null_blocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>();
        let null_blocks: Vec<u8> = (0..(u8_count)).map(|_| rng.gen::<u8>()).collect();
        let _block_u8_null_starts: Vec<usize> = (0..block_n).map(|a| a*u8_per_block).collect();
        let null_block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
        let start_null_bases: Vec<usize> = (0..block_n).map(|a| 2*a*bp_per_block).collect();
        let null_seq: NullSequence = NullSequence::new_manual(null_blocks, null_block_lens.clone());

        let start_bases_copy: Vec<_> = (0..block_n).map(|a| (2*a+1)*bp_per_block+10).collect();


        let wave: Waveform = Waveform::create_zero(&sequence, 5);
        let duration = start_gen.elapsed();
        println!("grad gen {} bp {:?}", bp, duration);

        let background = Background::new(0.25, 2.64, 20.).unwrap();

        let base_seq = unsafe{ AllData::create_manual(sequence.clone(), null_seq.clone(), WaveformDef::from(&wave), start_bases_copy.clone(), start_null_bases.clone(), background.clone(),3.0, 4.126)};

        let data_seq = unsafe{ AllDataUse::new_unchecked_data(wave.clone(),&null_seq, &start_bases_copy,&start_null_bases, &background, 3.0, 4.126)};

        for i in 0..10 {
            let mut buffer: Vec<u8> = bincode::serde::encode_to_vec(&base_seq, config::standard()).expect("serializable");

            println!("adsfdasffaafd {}", buffer.len());
            
            let start = Instant::now();

            let mut writer = File::create("../try_write.gz").unwrap();
            let mut parz: ParCompress<Mgzip> = ParCompressBuilder::new().compression_level(Compression::new(i)).from_writer(writer);
            parz.write_all(&buffer).unwrap();

            parz.finish().unwrap();

            let encode_time = start.elapsed();
            let start = Instant::now();

            let mut read_buffer: Vec<u8> = Vec::new();
            let mut read_file: ParDecompress<Mgzip> = ParDecompressBuilder::new().from_reader(File::open("../try_write.gz").unwrap());
            let bytes = read_file.read_to_end(&mut read_buffer);

            let decode_time = start.elapsed();
            println!("new size {} {:?} {i} encode {:?} decode {:?}", read_buffer.len(), bytes, encode_time, decode_time);
        }
        
        let mut rng = rand::thread_rng();

        let set_num_distr = rand::distributions::Uniform::try_from(80_usize..150).unwrap();

        let motif_sets: Vec<StrippedMotifSet> = (0..10).map(|j| {
            let n = set_num_distr.sample(&mut rng);
            println!("{j}");
            (&MotifSet::rand_with_n_motifs(n, &data_seq, &mut rng)).into()
        }).collect();


        //let mut buffer_lens: Vec<usize> = Vec::with_capacity(motif_sets.len());


        let mut writers: Vec<_> = (0..10).map(|i| File::create(&format!("../write_buffer_{i}.gz")).unwrap()).collect();

        let mut len_write = File::create("../write_lens.txt").unwrap();

        let mut j = 0_usize;
        for set in motif_sets.iter(){
            let mut buffer: Vec<u8> = bincode::serde::encode_to_vec(set, config::standard()).expect("serializable");
            //buffer_lens.push(buffer.len());
            len_write.write_all(format!("{}\n", buffer.len()).as_bytes());
            println!("Motif Set {}. Number Motifs: {}. Buffer length: {} bytes", j ,set.num_motifs(), buffer.len());
            for i in 0..10 {


                let start = Instant::now();

                let mut parz = SyncZBuilder::<Mgzip, _>::new().compression_level(Compression::new(i)).from_writer((std::io::Write::by_ref( &mut writers[i as usize])));
                parz.write_all(&buffer).unwrap();

                parz.finish().unwrap();

                let encode_time = start.elapsed();
                print!("{i}: {:?}\t", encode_time);
            }
            j = j+1;
            println!("");
        }

        let mut string_buff: String = String::new();
        File::open("../write_lens.txt").unwrap().read_to_string(&mut string_buff);
        let buffer_lens: Vec<usize> = string_buff.split_terminator('\n').map(|a| {
            println!("{}",a);
            a.parse::<usize>().expect("This whole file should be unsigned integers of buffer lengths")
        }).collect();

        let mut read_buffer: Vec<u8> = Vec::with_capacity(*(buffer_lens.iter().max().unwrap()));
        let mut read_file: ParDecompress<Mgzip> = ParDecompressBuilder::new().from_reader(File::open("../write_buffer_9.gz").unwrap());
        //let bytes = read_file.read_to_end(&mut read_buffer);

        for j in 0..motif_sets.len() {
            
            let mut handle = read_file.by_ref().take(buffer_lens[j] as u64);
            handle.read_to_end(&mut read_buffer).unwrap();

            let (new_set, bytes): (StrippedMotifSet, usize) = bincode::serde::decode_from_slice(&read_buffer, config::standard()).expect("deserializable");

            println!("Motif set {j} with {} motifs, buffer length {} bytes and recaptiulated buffer length {} bytes", motif_sets[j].num_motifs(), buffer_lens[j], bytes);
            println!("Old set 80th motif {:?}", motif_sets[j].get_nth_motif(80));
            println!("New set 80th motif {:?}", new_set.get_nth_motif(80));

            read_buffer.clear();
        }

}





} 
