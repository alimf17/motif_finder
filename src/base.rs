//pub mod bases {

use std::{f64, fmt, fs};

use std::collections::HashSet;

use std::ops::{Index, IndexMut};

use std::io::{Read, Write};
//use std::time::{Duration, Instant};
use std::error::Error;
use std::ptr;


use std::process::*;


use core::fmt::{Debug, Formatter};
use core::slice::{Iter, IterMut};

use crate::waveform::{Kernel, Waveform, Noise};
use crate::sequence::{Sequence, NullSequence, BP_PER_U8, U64_BITMASK, BITS_PER_BP};
use crate::modified_t::{ContinuousLnCDF};

use crate::{PROPOSE_EXTEND, DIRICHLET_PWM, THRESH, NECESSARY_MOTIF_IMPROVEMENT};
use crate::data_struct::{AllData, AllDataUse};

use itertools::Itertools;


use rand::Rng;
use rand::prelude::IteratorRandom;
use rand::seq::SliceRandom;
use rand::distributions::{Distribution, Uniform, WeightedIndex};

use  rand_distr::weighted_alias::*;

use statrs::{consts, StatsError};
use statrs::distribution::{Continuous, ContinuousCDF, LogNormal, Normal, Exp};
use statrs::statistics::{Min, Max};

use ordered_float::*;

use once_cell::sync::Lazy;

use rayon::prelude::*;

use pathfinding::*;

use plotters::prelude::*;
use plotters::coord::types::RangedSlice;
use plotters::coord::Shift;


use log::warn;

use regex::Regex;

use serde::{Serialize, Deserialize};


pub const SQRT_2: f64 = 1.41421356237;
pub const SQRT_3: f64 = 1.73205080757;

pub const LN_2: f64 = 0.69314718055994531;

pub const BPS: [char; 4] = ['A', 'C', 'G', 'T'];
pub const BASE_L: usize = BPS.len();

pub const BASE_PRIOR_DENS: f64 = 6.; //Set to 1/Gamma(BASE_L)

pub const VERTICES: [[f64; BASE_L-1]; BASE_L] = [[2.*SQRT_2/3., 0., -1.0/3.],[-SQRT_2/3., SQRT_2*SQRT_3/3., -1.0/3.], [-SQRT_2/3., -SQRT_2*SQRT_3/3., -1.0/3.],[0., 0., 1.0]];

//This MUST be the transpose of VERTICES
pub const SIMPLEX_VERTICES: [[f64; BASE_L]; BASE_L-1] = [[2.*SQRT_2/3. , -SQRT_2/3., -SQRT_2/3., 0.0], 
                                                           [0.              , SQRT_2*SQRT_3/3.   , -SQRT_2*SQRT_3/3., 0.0],
                                                           [-1.0/3.          , -1.0/3.           , -1.0/3.           , 1.0]];

pub const SIMPLEX_VERTICES_POINTS :  [[f64; BASE_L-1]; BASE_L] = [[2.*SQRT_2/3. , 0.               , -1.0/3.          ],
                                                                    [ -SQRT_2/3.  , SQRT_2*SQRT_3/3. , -1.0/3.          ],
                                                                    [ -SQRT_2/3.  ,-SQRT_2*SQRT_3/3. , -1.0/3.          ],
                                                                    [0.0          , 0.0              , 1.0              ]];

pub const SIMPLEX_ITERATOR: [&[f64; BASE_L-1]; (BASE_L-1)*(BASE_L-1)] = [&SIMPLEX_VERTICES_POINTS[0], &SIMPLEX_VERTICES_POINTS[1], &SIMPLEX_VERTICES_POINTS[2],
                                                                            &SIMPLEX_VERTICES_POINTS[0], &SIMPLEX_VERTICES_POINTS[1], &SIMPLEX_VERTICES_POINTS[3],
                                                                            &SIMPLEX_VERTICES_POINTS[0], &SIMPLEX_VERTICES_POINTS[3], &SIMPLEX_VERTICES_POINTS[2]];
pub const INVERT_SIMPLEX: [[f64; BASE_L]; BASE_L] = [[ 1.0/SQRT_2,  0.0             , -0.25, 0.25], 
                                                     [-SQRT_2/4.0,  SQRT_2*SQRT_3/4., -0.25, 0.25],
                                                     [-SQRT_2/4.0, -SQRT_2*SQRT_3/4., -0.25, 0.25],
                                                     [0.0        ,  0.0             ,  0.75, 0.25]];


//Obviously, this should always be the transpose of INVERT_SIMPLEX. I'm just not good enough at compile time code to make it happen automatically
pub const COL_PRIMARY_INVERT_SIMPLEX: [[f64; BASE_L]; BASE_L] = [[ 1.0/SQRT_2, -SQRT_2/4.0, -SQRT_2/4.0, 0.0], 
                                                                 [ 0.0           , SQRT_2*SQRT_3/4., -SQRT_2*SQRT_3/4.,   0.0          ],
                                                                 [ -0.25          ,  -0.25           ,  -0.25            ,   0.75          ],
                                                                 [  0.25          ,   0.25           ,   0.25            ,   0.25          ]];

pub const VERTEX_DOT: f64 = -1.0/((BASE_L-1) as f64);
        
pub const CAPACITY_FOR_NULL: usize = 200;

/*
const MULT_TRAJ_COMP: fn([f64; BASE_L-1], f64) -> [f64; BASE_L-1] = mult_traj;
const ADD_MULT_TRAJ_COMP: fn([f64; BASE_L-1], [f64; BASE_L-1], f64) -> [f64; BASE_L-1] = add_mult_traj;


static SIMPLEX_CONFINING_NORMALS: Lazy<[[[f64; BASE_L-1]; BASE_L-1]; BASE_L]> = Lazy::new(|| {


    let mut scn = [[[0_f64; BASE_L-1]; BASE_L-1]; (BASE_L)];
    let div_mag = 1./(2.*SQRT_2/SQRT_3);
    for i in 0..BASE_L {
        for j in 0..BASE_L {
            if j < i {
                scn[i][j] = MULT_TRAJ_COMP(ADD_MULT_TRAJ_COMP(VERTICES[i], VERTICES[j], -1_f64),div_mag);
            }
            if j > i {
                scn[i][j-1] = MULT_TRAJ_COMP(ADD_MULT_TRAJ_COMP(VERTICES[i], VERTICES[j], -1_f64),div_mag);
            }
        }
    }

    scn
});
*/

const CLOSE: f64 = 1e-5;

pub const MIN_BASE: usize = 8;
pub const MAX_BASE: usize = 20; //For a four base system, the hardware limit here is 32. 
                                //To make sure this runs before the heat death of the universe
                                //while also not taking more memory than every human brain combined,
                                //We store many kmers as u64s. 


pub const MIN_HEIGHT: f64 = 2.0;
pub const MAX_HEIGHT: f64 = 15.;
const LOG_HEIGHT_MEAN: f64 = 2.302585092994046; //This is ~ln(10). Can't use ln in a constant, and this depends on no other variables
const LOG_HEIGHT_SD: f64 = 0.25;

static HEIGHT_DIST: Lazy<TruncatedLogNormal> = Lazy::new(|| TruncatedLogNormal::new(LOG_HEIGHT_MEAN, LOG_HEIGHT_SD, MIN_HEIGHT, MAX_HEIGHT).unwrap() );

static NORMAL_DIST: Lazy<Normal> = Lazy::new(|| Normal::new(0.0, 1.0).unwrap());

static HEIGHT_PROPOSAL_DIST: Lazy<Exp> = Lazy::new(|| Exp::new(4.0).unwrap());

//static HEIGHT_PROPOSAL_DIST: Lazy<Exp> = Lazy::new(|| Exp::new(20.0).unwrap());

/*const L_SD_VECTOR_SPACE: f64 = 1.0;

const SD_LEAST_MOVE: f64 = 1.0;

const L_SD_VECTOR_SPACE_SINGLE: f64 = 1.0;

const SD_LEAST_MOVE_SINGLE: f64 = 1.0;

const HEIGHT_MOVE_SD: f64 = 1.0; */

const PROB_POS_PEAK: f64 = 1.0;


const BASE_RATIO_SDS: [f64; 3] = [0.1_f64, 0.5, 1.0];
const BASE_LINEAR_SDS: [f64; 3] = [0.1_f64, 0.5, 1.0];
 
const RATIO_LINEAR_SD_COMBO: Lazy<[[f64;2]; BASE_RATIO_SDS.len()*BASE_LINEAR_SDS.len()]> = 
                            Lazy::new(|| core::array::from_fn(|a| [BASE_RATIO_SDS[a/BASE_LINEAR_SDS.len()], BASE_LINEAR_SDS[a % BASE_LINEAR_SDS.len()]]));


const SCALE_SDS: [f64; 3] = [1.0, 10.0, 50.0];

const HEIGHT_SDS: [f64; 3] = [1_f64, 2.0, 10.0];

const NUM_MOVES: usize = 2*BASE_RATIO_SDS.len()*BASE_LINEAR_SDS.len()+HEIGHT_SDS.len()+2*HEIGHT_SDS.len()+7;
const VARIANT_NUMS: [usize; 9] = [BASE_RATIO_SDS.len()*BASE_LINEAR_SDS.len(), BASE_RATIO_SDS.len()*BASE_LINEAR_SDS.len(), HEIGHT_SDS.len(), 4, 1, 1, SCALE_SDS.len(), SCALE_SDS.len(), 1]; 

static PICK_MOVE: Lazy<WeightedAliasIndex<u32>> = Lazy::new(|| WeightedAliasIndex::<u32>::new(vec![20_u32, 20, 20, 20, 20, 20, 20, 20, 20]).expect("The weights should always be valid, with length equal to the length of VARIANT_NUMS"));

static SAMPLE_VARIANTS: Lazy<[Uniform<usize>; 9]> = Lazy::new(|| core::array::from_fn(|a| Uniform::new(0, VARIANT_NUMS[a])));

static VARIANT_CUMULATIVE: Lazy<[usize; 9]> = Lazy::new(|| core::array::from_fn(|a| if a == 0 {0} else {VARIANT_NUMS[0..a].iter().sum()}));

static HIST_END_NAMES: Lazy<[String; NUM_MOVES]> = Lazy::new(|| {
                                                   let b = RATIO_LINEAR_SD_COMBO;
                                                   let m = b.iter().map(|&[a,b]| format!("_base_scale_ratio_sd_{a}_linear_sd_{b}.png"));
                                                   let m = m.chain(b.iter().map(|&[a,b]| format!("_motif_scale_ratio_sd_{a}_linear_sd_{b}.png"))); 
                                                   let m = m.chain(HEIGHT_SDS.iter().map(|a| format!("_height_sd_{a}.png"))); 
                                                   let m = m.chain(["_motif_birth.png".to_owned(), "_motif_death.png".to_owned(),"_motif_expand.png".to_owned(),
                                                   "_motif_contract.png".to_owned(), "_base_leap.png".to_owned(), "_secondary_shuffle.png".to_owned()]);
                                                   let m = m.chain(SCALE_SDS.iter().map(|a| format!("_scale_base_sd_{a}.png")));
                                                   let m = m.chain(SCALE_SDS.iter().map(|a| format!("_scale_mot_sd_{a}.png")));
                                                   let m = m.chain(["_random_motif.png".to_owned()]);
                                                   m.collect::<Vec<_>>().try_into().unwrap()
});

//This controls how much shuffling we do for the secondary base shuffling move
//Faster if SHUFFLE_BASES <= MIN_BASE. Also, each base shuffle calculates 
//4^SHUFFLE_BASES PWMs
const SHUFFLE_BASES: usize = 4;

//const MAX_VECT_COORD: f64 = 943.-9.5367431640625e-7;

const REDUCE_MOTIF_SCALE_MOVE: [f64; MAX_BASE+1-MIN_BASE] = [0.024800, 0.021573, 0.019085, 0.017109, 0.015503, 0.014171, 0.013049, 0.012091, 0.011264, 0.010543, 0.0099079, 0.0093451, 0.0088427];

/*
static PROPOSE_EXTEND: Lazy<SymmetricBaseDirichlet> = Lazy::new(|| SymmetricBaseDirichlet::new(0.1_f64).unwrap());

static DIRICHLET_PWM: Lazy<SymmetricBaseDirichlet> = Lazy::new(|| SymmetricBaseDirichlet::new(0.1_f64).unwrap()); 

pub const THRESH: f64 = 1e-2; //SAFETY: This must ALWAYS be strictly greater than 0, or else we violate safety guarentees later.  

//This is roughly how much an additional motif should improve the ln posterior before it's taken seriously
//The more you increase this, the fewer motifs you will get, on average
const NECESSARY_MOTIF_IMPROVEMENT: f64 = 20.0_f64;
*/

pub const RJ_MOVE_NAMES: [&str; 4] = ["New motif", "Delete motif", "Extend motif", "Contract Motif"];

pub const BP_ARRAY: [Bp; BASE_L] = [Bp::A, Bp::C, Bp::G, Bp::T];

//This is 2^(-61), or 0x3c20000000000000 converted to f64.
//We have to use unsafe transmute instead of from_bits because from_bits
//is not stably const yet. Yes, this is obnoxious. 
//I picked this so that it has NO impact on values greater than about 0.01
//So our Base to vector and back sequence isn't technically COMPLETELY inversible
//But this is needed to prevent issues with "perfect" bases like [1.0, 0.0, 0.0, 0.0]
//Which have vector values that are all infinite.
//SAFETY: u64 bit patterns are laid out identically to f64 on all platforms
//        and technically, all valid u64s are valid f64s. Edge cases exist for NaN and Inf
//        But 2^(-61) is neither of those, and it's not subnormal, either. 
pub const VEC_PAD_EPS: f64 = unsafe{ std::mem::transmute::<u64, f64>(0x3c20000000000000) }; 
        
//This was NOT chosen randomly. I picked it so that proportions are
//all normal numbers
//const BARRIER: f64 = 2044_f64*LN_2;

//SAFETY: THRESH should be established at this point
static BARRIER: Lazy<f64> = Lazy::new(|| unsafe {
                                      println!("THRESH is {THRESH}");
                                      -2.0* THRESH.ln() } );

//BEGIN BASE

#[repr(usize)]
#[derive(Debug, Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Bp {
    A = 0,
    C = 1, 
    G = 2, 
    T = 3,
}



impl From<Bp> for usize {
    fn from(bp: Bp) -> Self {
        //SAFETY: Bp is repr to a valid usize
        unsafe{std::mem::transmute::<Bp, usize>(bp)}
    }
}

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

    //Safety: must be ABSOLUTELY sure that usize is < BASE_L (4)
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


#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Base {
   props: [ f64; BASE_L],
}

#[derive(Debug)]
pub enum InvalidBase {
    NegativeBase,
    NoNonZeroBase,
}

impl fmt::Display for InvalidBase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let message = match self {
            InvalidBase::NegativeBase => "No negative bases allowed!",
            InvalidBase::NoNonZeroBase => "You don't have any non zero bases!",
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
        &self.props[bp as usize]
    }

}

impl IndexMut<Bp> for Base {

    fn index_mut(&mut self, bp: Bp) -> &mut f64 {
        &mut self.props[bp as usize]
    }

}

impl Base {

    //Note: This will break if all the bindings are zeros, or any are negative
    pub fn new(props: [ f64; BASE_L]) -> Result<Base, InvalidBase> {

        let mut props = props;

        //#[cfg(test)]
        //println!("Base props {:?}", props);

        let mut any_neg: bool = false;

        for i in 0..props.len() {
            any_neg |= props[i] < 0.0 ;
        }
        
        if any_neg{
            return Err(InvalidBase::NegativeBase);
        }

        let max = Self::max(&props);

        if max == 0.0 {
            return Err(InvalidBase::NoNonZeroBase);
        }
        
        for i in 0..props.len() {
                props[i] = props[i]/max;
        }




        Ok(Base { props })
    }

    pub fn new_with_pseudo(props: [f64; BASE_L], counts: f64, pseudo: f64) -> Result<Base, InvalidBase> {

        let mut props = props;
        
        let mut any_neg: bool = false;

        for i in 0..props.len() {
            any_neg |= props[i] < 0.0 ;
        }

        if any_neg{
            return Err(InvalidBase::NegativeBase);
        }

        let sum = props.iter().sum::<f64>();

        if sum == 0.0 { 
            for p in props.iter_mut() { *p += pseudo; } 
        } else {
            for p in props.iter_mut() {*p = (*p/sum)*counts+pseudo;}
        }

        Base::new(props)
    }


    pub fn rand_new<R: Rng + ?Sized>(rng: &mut R) -> Base {


        let mut samps = vec![0.0_f64; BASE_L+1];
        samps[BASE_L] = 1.0;

        let mut att: [f64; BASE_L] = [0.0; BASE_L];

        for i in 1..(samps.len()-1) {
            samps[i] = rng.gen();
        }

        samps.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let mut max_p = 0.0_f64;
        for i in 0..att.len() {
            att[i] = samps[i+1]-samps[i];
            max_p = max_p.max(att[i]);
        }
        for i in 0..att.len() {
            att[i] /=max_p;
        }

        Base { props: att}
    }

    pub fn rand_dirichlet_new<R: Rng + ?Sized>(rng: &mut R) -> Base {
        DIRICHLET_PWM.get().expect("no writes expected now").sample(rng)
    }

    pub fn propose_safe_new<R: Rng + ?Sized>(rng: &mut R) -> Base {

        PROPOSE_EXTEND.get().expect("no writes expected now").sample(rng)

    }

    pub fn from_bp<R: Rng + ?Sized>(best: Bp, rng: &mut R) -> Base {

        Base::rand_new(rng).make_best(best)

    }

    pub fn from_bp_with_uniforms<R: Rng + ?Sized>(best: Bp, rng: &mut R) -> Base {

        let mut props: [f64; BASE_L] = [0.; BASE_L];
        
        for i in 0..BASE_L{
            if i == (best as usize) {
                props[i] = 1.0;
            } else {
                props[i] = rng.gen();
            }
        }
        
        Base{ props: props }

    }

    pub fn from_bp_with_dirichlet<R: Rng + ?Sized>(best: Bp, rng: &mut R) -> Base {
        Base::rand_dirichlet_new(rng).make_best(best)
    }


    pub fn make_best(&self, best: Bp) -> Base {

        let which_b = self.best_base();

        if best != which_b {
            self.swap_pair(best, which_b)
        } else {
            self.clone()
        }


    }

    pub fn swap_pair(&self, bp1: Bp, bp2: Bp) -> Base {
        
        let mut base2 = self.clone();
        base2[bp1] = self[bp2];
        base2[bp2] = self[bp1];

        base2

    }

    pub fn best_base(&self) -> Bp {
        //Safety: self.props has a size of BASE_L, so this always produces a valid result
        unsafe{ std::mem::transmute::<usize, Bp>(Self::argmax(&self.props))}
 
        //Genuine surprise: this is worse when I profile it on godbolt
        /*let mut best = if (self[Bp::A] < self[Bp::C]) { Bp::C } else {Bp::A};
        if (self[best] < self[Bp::G]) { best = Bp::G; }
        if (self[best] < self[Bp::T]) { best = Bp::T; }
        best*/
    }

    pub fn all_non_best(&self) -> [Bp; BASE_L-1] {
        let best = self.best_base();


        match best {

            Bp::A => [Bp::C, Bp::G, Bp::T],
            Bp::C => [Bp::A, Bp::G, Bp::T],
            Bp::G => [Bp::A, Bp::C, Bp::T],
            Bp::T => [Bp::A, Bp::C, Bp::G],
        }

    }

    pub fn dist_sq(&self, base: Option<&Base>) -> f64 {
        
        match base {

            None => 4.0*self.props.iter().map(|a| a.ln().powi(2)).sum::<f64>()-self.props.iter().map(|a| a.ln()).sum::<f64>().powi(2),
            Some(other) => {
                4.0*self.props.iter().zip(other.props.iter()).map(|(a,b)| (a.ln()-b.ln()).powi(2)).sum::<f64>()-(self.props.iter().map(|a| a.ln()).sum::<f64>()-other.props.iter().map(|a| a.ln()).sum::<f64>()).powi(2)
            },
        }

    }

    pub fn as_probabilities(&self) -> [f64; BASE_L] {
        let summed = self.props.iter().sum::<f64>();
        let ret: [f64; BASE_L] = core::array::from_fn(|a| self.props[a]/summed); // self.props.iter().map(|&a| a/summed).collect::<Vec<_>>().try_into().unwrap();
        ret
    }

    pub fn as_simplex(&self) -> [f64; BASE_L-1] {
        let probs = self.as_probabilities();
        Self::prob_slice_to_simplex(&probs)
    }

    pub fn prob_slice_to_simplex(probs: &[f64; BASE_L]) -> [f64; BASE_L-1] {
        let simplex: [f64; BASE_L-1] = SIMPLEX_VERTICES.iter().map(|a| a.iter().zip(probs.iter()).map(|(&s, &b)| s*b).sum::<f64>())
                                                       .collect::<Vec<f64>>().try_into().unwrap();

        simplex
    }

    pub fn sum_bases_and_ln_max(&self, other_base: &Base) -> Result<(Self, f64), InvalidBase> {

        let mut max_prop: f64 = 0.0;

        let pre_base: [f64; BASE_L] = core::array::from_fn(|a| {
            let p = self.props[a]*other_base.props[a];
            max_prop = max_prop.max(p);
            p
        });

        match Base::new(pre_base) { 
            Ok(base) => Ok((base, max_prop.ln())),
            Err(e) => Err(e),
        }

    }

    pub fn subtract_bases_and_ln_max(&self, subtrahend: &Base) -> Result<(Self, f64), InvalidBase> {

        let best = self.best_base();

        let pre_difference: [f64; BASE_L] = core::array::from_fn(|a| self.props[a]/subtrahend.props[a]);

        match Base::new(pre_difference) { 
            Ok(base) => 
            {
                let ln_max = (subtrahend[best]*base[best]).ln();
                Ok((base, ln_max))
            },
            Err(e) => Err(e),
        }
    }

    //This converts our base representation directly to a vector space representation
    //In particular, our bases form a metric space under 
    //"Vector addition" is elementwise multiplication and then renormalize
    //"Scalar multiplication" is elementwise powering to the scalar and then renormalize
    //And the metric is the Aitchison metric
    //This vector space is isometric to R3 under the transformation specified in the functions
    //Where if p = 1/(2*(1+e)) and q = e*p
    //e1 = [p,q,q,p], e2 = [p,q,p,q], e3 = [p,p,q,q]
    pub fn base_to_vect(&self) -> [f64; BASE_L-1] {

        //let ln_vec: [f64; BASE_L] = core::array::from_fn(|a| (self.props[a]+VEC_PAD_EPS).ln());
        /*let ln_vec: [f64; BASE_L] = core::array::from_fn(|a| {
            let b = (self.props[a]).ln();
            if b.is_infinite() { b.signum()*(MAX_VECT_COORD)} else {b}
        });*/


        Base::reflect_triple_to_finite(
            &[nicer_ln_ratio([self[Bp::C], self[Bp::G]],[self[Bp::A], self[Bp::T]]),
              nicer_ln_ratio([self[Bp::C], self[Bp::T]],[self[Bp::A], self[Bp::G]]), 
              nicer_ln_ratio([self[Bp::T], self[Bp::G]],[self[Bp::A], self[Bp::C]])] )
    }

    pub fn vect_to_base(base_as_vec: &[f64; BASE_L-1]) -> Self {



        //This clamp exists to make sure that we don't over or underflow exp and break things
        let sanitized_base: [f64; BASE_L-1] = Self::reflect_triple_to_finite(base_as_vec); 


        //TODO: fix this so that it starts with best base at 1.0
        let pre_exp_base: [f64; BASE_L] = 
            [-(sanitized_base[0]+sanitized_base[1]+sanitized_base[2])*0.25, 
              (sanitized_base[0]+sanitized_base[1]-sanitized_base[2])*0.25,
              (sanitized_base[0]-sanitized_base[1]+sanitized_base[2])*0.25,
             (-sanitized_base[0]+sanitized_base[1]+sanitized_base[2])*0.25];


        let max_val = pre_exp_base.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).expect("We obviously know this has items in it");

        let unnormalized_base: [f64; BASE_L] = core::array::from_fn(|a| (pre_exp_base[a]-max_val).exp());

        //SAFETY: reflect_triple_to_finite is written so that no operation on exp returns all zeros here
        unsafe { Base::new(unnormalized_base).unwrap_unchecked()}
    }

    //If ratio_sd and small_sd are negative, the move generated is just the opposite
    //which gives a valid result. Hence, no erroring out. You SHOULDN'T use negatives
    //because that's out of convention for no reason, but it won't break.
    pub fn moved_base<R: Rng + ?Sized>(&self, ratio_sd: f64, small_sd: f64, rng: &mut R) -> Option<Base> {

        let mut new_base_vect = self.base_to_vect();

        let signs_base_vect: [f64; BASE_L-1] = core::array::from_fn(|a| new_base_vect[a].signum());

        let mut largest_abs_to_smallest: [(f64, usize); BASE_L-1] = core::array::from_fn(|a| (new_base_vect[a].abs(), a));

        largest_abs_to_smallest.sort_unstable_by(|(a,_), (b, _)| b.partial_cmp(a).unwrap());


        //let mut large_abs_ratio = if largest_abs_to_smallest[1].0 == 0.0 { 1.0} else {largest_abs_to_smallest[0].0/largest_abs_to_smallest[1].0};

        //let mut small_abs_ratio = largest_abs_to_smallest[1].0/largest_abs_to_smallest[2].0};

        let (mut large_abs_ratio, mut small_abs_ratio) = 
            if largest_abs_to_smallest[1].0 == 0.0 {
                (1.0, 1.0) 
            } else {
                let la = largest_abs_to_smallest[0].0/largest_abs_to_smallest[1].0;
                let sm = if largest_abs_to_smallest[2].0 == 0.0 { 1.0 } else {largest_abs_to_smallest[1].0/largest_abs_to_smallest[2].0};
                (la, sm)
            };


        large_abs_ratio *= (NORMAL_DIST.sample(rng)*ratio_sd).exp();

        if large_abs_ratio < 1.0 { large_abs_ratio = 1.0/large_abs_ratio; }

        small_abs_ratio *= (NORMAL_DIST.sample(rng)*ratio_sd).exp();
        
        if small_abs_ratio < 1.0 { small_abs_ratio = 1.0/small_abs_ratio; }

        new_base_vect[largest_abs_to_smallest[2].1] += NORMAL_DIST.sample(rng)*small_sd;

        new_base_vect[largest_abs_to_smallest[1].1] = small_abs_ratio*new_base_vect[largest_abs_to_smallest[2].1].abs()*signs_base_vect[largest_abs_to_smallest[1].1];

        new_base_vect[largest_abs_to_smallest[0].1] = large_abs_ratio*new_base_vect[largest_abs_to_smallest[1].1].abs()*signs_base_vect[largest_abs_to_smallest[0].1];


        let refl = Base::reflect_triple_to_finite(&new_base_vect);

       
        Some(Base::vect_to_base(&refl))


    }

    pub fn scaled_base<R: Rng + ?Sized>(&self, scale_sd: f64, rng: &mut R) -> Option<Base> {

        let mut new_base_vect = self.base_to_vect();

        let scale_exponent = (NORMAL_DIST.sample(rng)*scale_sd).exp();

        new_base_vect = core::array::from_fn(|a| new_base_vect[a]*scale_exponent);

        Some(Base::vect_to_base(&Base::reflect_triple_to_finite(&new_base_vect)))

    }

    //Panic: if given a point outside of the base terahedron, this function will panic 
    pub fn simplex_to_base(simplex_coords: &[f64; BASE_L-1]) -> Base {

        let mut mod_simplex = simplex_coords.to_vec();
        mod_simplex.push(1.0);

        let mod_simplex = mod_simplex;

        //let probs: [f64; BASE_L] = INVERT_SIMPLEX.iter().map(|a| a.iter().zip(mod_simplex.iter()).map(|(&b, &c)| b*c).sum::<f64>()).collect::<Vec<_>>().try_into().unwrap();

        let mut probs = [0.0_f64; BASE_L];

        for i in 0..BASE_L{
            for j in 0..BASE_L {
                probs[i] += INVERT_SIMPLEX[i][j]*mod_simplex[j];
            }
        }

        let max = probs[Base::argmax(&probs)];

        let b_form: [f64; BASE_L]  = core::array::from_fn(|a| probs[a]/max); //probs.into_iter().map(|a| a/max).collect::<Vec<_>>().try_into().unwrap();

        Base{props : b_form}

    }

    pub fn simplex_to_vect(simplex_coords: &[f64; BASE_L-1]) -> [f64; BASE_L-1] {

        let pre_ln_1p: [f64; BASE_L] = [2.*SQRT_2*simplex_coords[0]-simplex_coords[2], 
                                         -SQRT_2*simplex_coords[0]+SQRT_2*SQRT_3*simplex_coords[1]-simplex_coords[2],
                                         -SQRT_2*simplex_coords[0]-SQRT_2*SQRT_3*simplex_coords[1]-simplex_coords[2],
                                         3.*simplex_coords[2] ];

        let ln_vec: [f64; BASE_L] = core::array::from_fn(|a| pre_ln_1p[a].ln_1p());

        [ -ln_vec[0]+ln_vec[1]+ln_vec[2]-ln_vec[3],
          -ln_vec[0]+ln_vec[1]-ln_vec[2]+ln_vec[3],
          -ln_vec[0]-ln_vec[1]+ln_vec[2]+ln_vec[3]]

    }

    pub fn vect_to_simplex(vect_coords: &[f64; BASE_L-1]) -> [f64; BASE_L-1] {
        Self::vect_to_base(vect_coords).as_simplex()
    }


    /*fn d_base_d_vect(&self) ->  [[f64; BASE_L-1]; BASE_L-1] {
        
        let other_bases = self.all_non_best();
        let proto_column: [f64; BASE_L-1] = core::array::from_fn(|a| 0.5 * self[other_bases[a]]);

        let mut grad = [proto_column.clone(), proto_column.clone(), proto_column.clone()];

        let (zeros, neg_cols) = match self.best_base() {
            Bp::A => ([2_usize, 1, 0], None),
            Bp::C => ([1, 2, 0], Some([0_usize, 1])),
            Bp::G => ([1, 0, 2], Some([0_usize, 2])),
            Bp::T => ([0, 1, 2], Some([1_usize, 2])),
        };

        for i in 0..(BASE_L-1) {
            grad[i][zeros[i]] = 0.0;
        }

        if let Some(negs) = neg_cols {
            for i in 0..(BASE_L-1) {
                grad[negs[0]][i] *= -1.0;
                grad[negs[1]][i] *= -1.0;
            }
        }
        
        grad


    }*/
 

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
        

        let first_checks = [triple[0].abs()+triple[1].abs()-*BARRIER,
                            triple[0].abs()+triple[2].abs()-*BARRIER,
                            triple[1].abs()+triple[2].abs()-*BARRIER];

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
        let triple_in_basis = [(-triple[0]+triple[1]+triple[2])/(4.0* *BARRIER),
                               ( triple[0]-triple[1]+triple[2])/(4.0* *BARRIER),
                               ( triple[0]+triple[1]-triple[2])/(4.0* *BARRIER)];

        let reduce_reflection: [f64;BASE_L-1] = core::array::from_fn(|a| triple_in_basis[a].fract());

        let triple = [(reduce_reflection[1]+reduce_reflection[2])*2.0* *BARRIER,
                      (reduce_reflection[0]+reduce_reflection[2])*2.0* *BARRIER,
                      (reduce_reflection[0]+reduce_reflection[1])*2.0* *BARRIER];

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
            new_triple[i] = -new_triple[i]+sign* *BARRIER;
            new_triple[j] = -new_triple[j]+sign* *BARRIER;
        } else {
            let sign  = diff.signum();
            new_triple[i] += sign* *BARRIER;
            new_triple[j] -= sign* *BARRIER;
        }
       
        new_triple

    }

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

    }


    pub fn rev(&self) -> Base {

        let mut revved = self.props.clone();

        revved.reverse();

        //SAFETY: the reverse of a valid Base is still valid
        unsafe{ Self::new(revved).unwrap_unchecked() }


    }

    pub fn show(&self) -> [f64; BASE_L] {
        let prop_rep = self.props.clone();
        prop_rep
    }


    
    //SAFETY: bp MUST be less than the BASE_L, or else this will produce undefined behavior
    //        Why do I do this in an unsafe way? My code is literally half again as fast when this
    //        is unsafe. When we use the bases from the Sequence object, the initialization of Sequence
    //        ensures that this is within bounds, but the compiler isn't smart enough to know that. 
    pub unsafe fn rel_bind(&self, bp: usize) -> f64 {
        *self.props.get_unchecked(bp)
    }
    pub fn safe_bind(&self, bp: usize) -> f64 {
        self.props[bp]
    }

    pub fn seqlogo_heights(&self) -> [(usize, f64); BASE_L] {
        let probs = self.as_probabilities();
        let information_content = 2.0_f64 + probs.iter().map(|&p| if p == 0.0 { 0.0_f64 } else { p*p.log2() } ).sum::<f64>();
        let mut heights: [(usize, f64); BASE_L] = probs.iter().enumerate().map(|(i, &p)| (i, p*information_content)).collect::<Vec<_>>().try_into().expect("we know how big this is");
        heights.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let mut cumulative_height: f64 = 0.0;
        for i in 0..BASE_L {
            heights[i].1 += cumulative_height;
            cumulative_height = heights[i].1;
        }

        heights = heights.into_iter().rev().collect::<Vec<_>>().try_into().expect("it's always the same length");
        heights
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



fn reflect_abs_height(a: f64) -> f64 {
    if (a > MIN_HEIGHT) && (a < MAX_HEIGHT) {
        return a;
    }

    let reflect_check = (a.abs()-MIN_HEIGHT)/(MAX_HEIGHT-MIN_HEIGHT);
    let try_end = reflect_check-reflect_check.floor();
    let flip_space = (reflect_check.floor() as i32 & 0x01_i32) == 1;

    if flip_space {
        MAX_HEIGHT-try_end*(MAX_HEIGHT-MIN_HEIGHT)
    } else {
        MIN_HEIGHT+try_end*(MAX_HEIGHT-MIN_HEIGHT)
    }
    //-reflect_cond*(MAX_HEIGHT-MIN_HEIGHT)*a_sign*a.signum()+a_sign*a
}
                                 
//Note that arrays of Copy are themselves Copy
/*fn reflect_tetra(start: [f64; BASE_L-1], push: [f64; BASE_L-1], confine_base: bool) -> [f64; BASE_L-1] {
    let mut end_start = start;
    let mut end_push = Some(push);
    //let mut count: usize = 0;
    while let Some(push_vec) = end_push {
        (end_start, end_push) = wall_collide(end_start, push_vec, confine_base);
      //  count += 1;
    //if end_push.is_some() {println!("cc {} {:?} {:?} {:?} {:?} {:?}", count, push, push_vec, start, end_start, end_push);}
    }
    end_start
}

fn wall_collide(start: [f64; BASE_L-1], push: [f64; BASE_L-1], confine_base: bool) -> ([f64; BASE_L-1], Option<[f64; BASE_L-1]>) {

    //This checks our reflections across the regular tetrahedral walls
    let mut min_stop: Option<f64> = None;
    let mut vert_min: Option<[f64; BASE_L-1]> = None;
    for i in 0..BASE_L {
        let poss_stop = detect_reflect(start, push, VERTICES[i], VERTEX_DOT); //(VERTEX_DOT-VERTICES[i].iter().zip(start.iter()).map(|(&a, &b)| a*b).sum::<f64>())/(VERTICES[i].iter().zip(push.iter()).map(|(&a, &b)| a*b).sum::<f64>());
        if let Some(stop) = poss_stop { 
            (min_stop, vert_min) = match min_stop {
                Some(ms) => if ms > stop { (Some(stop), Some(VERTICES[i])) } else {(min_stop, vert_min)},
                None => (Some(stop), Some(VERTICES[i])),
            }
        }
    }

    //This reflects us towards maintaining the same best bast when necessary
    if confine_base {
        let mut close_base = 0_usize;
        let mut close_dot = -f64::INFINITY;
        for i in 0..BASE_L {
            let alignment = dot_prod(start, VERTICES[i]);
            if alignment > close_dot {
                close_dot = alignment;
                close_base = i;
            }
        }

        for i in 0..(BASE_L-1) {

            let poss_stop = detect_reflect(start, push, SIMPLEX_CONFINING_NORMALS[close_base][i], 0.0);
            if let Some(stop) = poss_stop {
                (min_stop, vert_min) = match min_stop {
                    Some(ms) => if ms > stop { (Some(stop), Some(SIMPLEX_CONFINING_NORMALS[close_base][i])) } else {(min_stop, vert_min)},
                    None => (Some(stop), Some(SIMPLEX_CONFINING_NORMALS[close_base][i])),
                }
            }
        }
        
    }

    //This DOES our reflection
    match min_stop {
        None =>  (add_traj(start, push), None), //(start.iter().zip(push.iter()).map(|(&a, &b)| a+b).collect::<Vec<f64>>().try_into().expect("Size is pinned down"), None),
        Some(dot) => {
            let vert = vert_min.expect("this should not be None if min_stop has a value");

            //The 1e-6s that we subtract from dot are a numerical guard. If we let things go
            //right up to the boundary, numerical errors can end with us outside of the
            //tetrahedron.
            let new_start: [f64; BASE_L-1] = add_mult_traj(start, push, dot-1e-6);//start.iter().zip(push.iter()).map(|(&a, &b)| a+(dot-1e-6)*b).collect::<Vec<f64>>().try_into().expect("Size is pinned down");
            let mut remaining_trajectory: [f64; BASE_L-1] = mult_traj(push, 1.0-(dot-1e-6));//push.iter().map(|&b| (1.0-(dot-1e-6))*b).collect::<Vec<f64>>().try_into().expect("Size is pinned down");
            let traj_dot = dot_prod(remaining_trajectory, vert); //remaining_trajectory.iter().zip(VERTICES[i].iter()).map(|(&a, &b)| a*b).sum::<f64>();
            remaining_trajectory = add_mult_traj(remaining_trajectory, vert,-2.0*traj_dot); //remaining_trajectory.into_iter().zip(VERTICES[i].iter()).map(|(a, &b)| a-2.0*traj_dot*b).collect::<Vec<_>>().try_into().expect("Size is pinned down");
            (new_start, Some(remaining_trajectory))
        },
    }

}
//Usage: Always pick normal so that the region you want to remain in has a dot product GREATER than boundary_dot_value
fn detect_reflect(start: [f64; BASE_L-1], push: [f64; BASE_L-1], normal: [f64; BASE_L-1], boundary_dot_value: f64) -> Option<f64> {

    //For our tetrahedral walls, using vertices as the normal vector of the plane means that outside
    //the tetrahedron dots to a negative value with the normal, so we can only theoretically approach 
    //if our push vector dotted with the normal is negative. This is an important check: the start
    //vector is allowed to start ON a boundary, so this check keeps us from reflecting if we start 
    //on the boundary but move away. For reflecting with bases, we choose the normal vector
    //base_vertex_start-base_vertex_end to keep the same barrier. And in general, it is always
    //possible to choose the normal such that the negative dot direction is the direction hitting
    //the wall. If you picked the normal such that this direction is positive, take its negation.
    let approach_wall = dot_prod(push, normal);

    //If approach_wall is exactly 0, I'm moving parallel to the boundary, not crossing it.
    if approach_wall >= 0.0 { return None; }

    let stop = (boundary_dot_value-dot_prod(start, normal))/approach_wall;

    //stop <= 0.0 means that we would hit the wall going backwards, which means that we're moving
    //away from the wall. But we already checked this with approach_wall >= 0.0 returning None
    if stop > 1.0 { //|| (stop <= 0.0)   {
        return None;
    }

    Some(stop)
}

fn dot_prod(a: [f64; BASE_L-1], b: [f64; BASE_L-1]) -> f64 {

    let mut dot: f64 = 0.0;

    for i in 0..(BASE_L-1) {
        dot += a[i]*b[i];
    }

    dot
}

fn add_traj(a: [f64; BASE_L-1], b: [f64; BASE_L-1]) -> [f64; BASE_L-1] {

    let mut out = [0.0_f64; BASE_L-1];
    for i in 0..(BASE_L-1) {
        out[i] = a[i]+b[i];
    }
    out

}

fn mult_traj(a: [f64; BASE_L-1], lambda: f64) -> [f64; BASE_L-1] {
    let mut out = [0.0_f64; BASE_L-1];
    for i in 0..(BASE_L-1) {
        out[i] = a[i] * lambda;
    }
    out
}

fn add_mult_traj(a: [f64; BASE_L-1], b: [f64; BASE_L-1], lambda: f64) -> [f64; BASE_L-1] {

    let mut out = [0.0_f64; BASE_L-1];
    for i in 0..(BASE_L-1) {
        out[i] = a[i]+b[i] * lambda;
    }
    out

}*/



//BEGIN MOTIF

#[derive(Serialize, Deserialize, Clone)]
pub struct Motif {

    peak_height: f64,
    pwm: Vec<Base>,

}

impl Motif {

    //GENERATORS
    //PANIC: if any proposed PWM is made with a capacity greater than MAX_BASE, the program will panic
    //NOTE: all pwm vectors are reserved with a capacity exactly equal to MAX_BASE. This is because motifs can only change size up to that point.        
    //TODO: make sure that this is used iff there's a guarentee that a motif is allowed
    pub fn raw_pwm(mut pwm: Vec<Base>, peak_height: f64) -> Motif {

        pwm.reserve_exact(MAX_BASE-pwm.len());

        let m = Motif {
            peak_height: peak_height,
            pwm: pwm,
        };

        m
    }

    pub fn rand_height_pwm<R: Rng + ?Sized>(pwm: Vec<Base>, rng: &mut R) -> Motif {

        //let height_dist: TruncatedLogNormal = TruncatedLogNormal::new(LOG_HEIGHT_MEAN, LOG_HEIGHT_SD, MIN_HEIGHT, MAX_HEIGHT).unwrap();

        let sign: f64 = rng.gen();
        let sign: f64 = if sign < PROB_POS_PEAK {1.0} else {-1.0};

        let peak_height: f64 = sign*HEIGHT_DIST.sample(rng);

        Self::raw_pwm(pwm, peak_height)
    }


    //TODO: make sure that this is used iff there's a guarentee that a motif is allowed
    pub fn from_motif<R: Rng + ?Sized>(best_bases: Vec<Bp>, rng: &mut R) -> Motif {

        let mut pwm: Vec<Base> = best_bases.iter().map(|a| Base::from_bp(*a, rng)).collect();
        pwm.reserve_exact(MAX_BASE-best_bases.len());

        //let height_dist: truncatedlognormal = truncatedlognormal::new(log_height_mean, log_height_sd, min_height, max_height).unwrap();


        let sign: f64 = rng.gen();
        let sign: f64 = if sign < PROB_POS_PEAK {1.0} else {-1.0};

        let peak_height: f64 = sign*HEIGHT_DIST.sample(rng);


        Motif {
            peak_height: peak_height,
            pwm: pwm,
        }


    }

    pub fn from_motif_with_height<R: Rng + ?Sized>(best_bases: Vec<Bp>, height: f64, rng: &mut R) -> Motif {

        let mut pwm: Vec<Base> = best_bases.iter().map(|a| Base::from_bp(*a, rng)).collect();
        pwm.reserve_exact(MAX_BASE-best_bases.len());

        //let height_dist: truncatedlognormal = truncatedlognormal::new(log_height_mean, log_height_sd, min_height, max_height).unwrap();

        Motif {
            peak_height: height,
            pwm: pwm,
        }


    }


    pub fn from_motif_uniform<R: Rng + ?Sized>(best_bases: Vec<Bp>, rng: &mut R) -> Motif {

        let mut pwm: Vec<Base> = best_bases.iter().map(|a| Base::from_bp_with_uniforms(*a, rng)).collect();
        pwm.reserve_exact(MAX_BASE-best_bases.len());

        //let height_dist: truncatedlognormal = truncatedlognormal::new(log_height_mean, log_height_sd, min_height, max_height).unwrap();


        let sign: f64 = rng.gen();
        let sign: f64 = if sign < PROB_POS_PEAK {1.0} else {-1.0};

        let peak_height: f64 = sign*HEIGHT_DIST.sample(rng);


        Motif {
            peak_height: peak_height,
            pwm: pwm,
        }


    }

    pub fn from_motif_dirichlet<R: Rng + ?Sized>(best_bases: Vec<Bp>, rng: &mut R) -> Motif {

        let mut pwm: Vec<Base> = best_bases.iter().map(|a| Base::from_bp_with_dirichlet(*a, rng)).collect();
        pwm.reserve_exact(MAX_BASE-best_bases.len());

        //let height_dist: truncatedlognormal = truncatedlognormal::new(log_height_mean, log_height_sd, min_height, max_height).unwrap();


        let sign: f64 = rng.gen();
        let sign: f64 = if sign < PROB_POS_PEAK {1.0} else {-1.0};

        let peak_height: f64 = sign*(HEIGHT_PROPOSAL_DIST.sample(rng)+MIN_HEIGHT);


        Motif {
            peak_height: peak_height,
            pwm: pwm,
        }


    }



    pub fn rand_mot<R: Rng + ?Sized>(seq: &Sequence, rng: &mut R) -> Motif {


        let num_bases = rng.gen_range(MIN_BASE..(MAX_BASE+1));

        let mot = seq.random_valid_motif(num_bases, rng);


        Self::from_motif(mot,  rng)


    }

    pub fn rand_mot_dirichlet<R: Rng + ?Sized>(seq: &Sequence, rng: &mut R) -> Motif {


        let num_bases = rng.gen_range(MIN_BASE..(MAX_BASE+1));

        let mot = seq.random_valid_motif(num_bases, rng);

        Self::from_motif_dirichlet(mot, rng)


    }



    pub fn rand_propensity_mot<R: Rng + ?Sized>(wave: &Waveform, rng: &mut R) -> Option<(Motif, f64)> {

        let num_bases = rng.gen_range(MIN_BASE..(MAX_BASE+1));

        let propensities = wave.kmer_propensities(num_bases);

        let mut sum_prop: f64 = 0.0;

        for p in propensities.iter() { sum_prop += *p; }

        if sum_prop <= 0.0 {
            return None;
        }

        let dist = WeightedIndex::new(&propensities).unwrap();

        let selection = dist.sample(rng);

        let mot = Sequence::u64_to_kmer(wave.seq().idth_unique_kmer(num_bases, selection), num_bases);

        let motif = Self::from_motif_uniform(mot, rng);

        let dirichlet_like = motif.pwm.iter().map(|a| DIRICHLET_PWM.get().expect("no writes expected now").ln_pdf(a)).sum::<f64>();

        Some((motif, dirichlet_like+propensities[selection].ln()-sum_prop.ln()))
    }

    pub fn rand_mot_with_height<R: Rng + ?Sized>(peak_height: f64, seq: &Sequence, rng: &mut R) -> Motif {


        let num_bases = rng.gen_range(MIN_BASE..(MAX_BASE+1));

        let mot = seq.random_valid_motif(num_bases, rng);

        let mut pwm: Vec<Base> =  mot.iter().map(|a| Base::from_bp(*a, rng)).collect();
        pwm.reserve_exact(MAX_BASE-pwm.len());

        Motif {
            peak_height: peak_height,
            pwm: pwm,
        }



    }

    //Panics: if num_bases < MIN_BASE or num_bases > MAX_BASE
    pub fn rand_mot_with_height_and_motif_len<R: Rng + ?Sized>(peak_height: f64, num_bases: usize, seq: &Sequence, rng: &mut R) -> Motif {

        assert!((num_bases >= MIN_BASE) && (num_bases <= MAX_BASE), 
                "Only motif lengths with a length between {MIN_BASE} and {MAX_BASE}, inclusive, are allowed. You tried to make a {num_bases} Base long motif.");

        let mot = seq.random_valid_motif(num_bases, rng);

        let mut pwm: Vec<Base> = mot.iter().map(|a| Base::from_bp(*a, rng)).collect();
        pwm.reserve_exact(MAX_BASE-pwm.len());


        Motif {
            peak_height: peak_height,
            pwm: pwm,
        }



    }


    pub fn make_opposite(&self) -> Motif {

        let mut opposite = self.clone();

        opposite.peak_height = -self.peak_height;
        opposite

    }

    pub fn scramble_to_close_random_valid<R: Rng + ?Sized>(&mut self, seq: &Sequence, randomizer: &mut Option<&mut R>) -> Option<usize> {

        let best_motif = self.best_motif();
        if seq.kmer_in_seq(&best_motif) {return None;}

        let mut hamming = 0;
        let mut kmer_ids: Vec<usize> = Vec::new();

        while kmer_ids.len() == 0 {
            hamming += 1;
            kmer_ids = seq.all_kmers_within_hamming(&best_motif, hamming);
        }

        match randomizer {
            Some(rng) => *self = self.scramble_by_id_to_valid(*kmer_ids.choose(rng).expect("We should never try this unless we have kmers to choose from"), false, seq),
            None => *self = self.scramble_by_id_to_valid(kmer_ids[0], false, seq),
        };

        Some(hamming)

    }

    pub fn scramble_by_id_to_valid(&self, id: usize, opposite: bool, seq: &Sequence) -> Motif {

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

    pub fn scramble_secondaries(&self, to_scramble: &[usize]) -> Vec<Motif> {

        
        let final_length = SHUFFLE_BASES.pow(to_scramble.len() as u32);
        
        let swapped_bases: Vec<[Base; BASE_L-1]> = to_scramble.iter().map(|&a| {
            let non_best = self.pwm[a].all_non_best();
            [self.pwm[a].swap_pair(non_best[0], non_best[1]), 
            self.pwm[a].swap_pair(non_best[1], non_best[2]), self.pwm[a].swap_pair(non_best[0], non_best[2])]
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

    pub fn rev_complement(&self) -> Vec<Base> {
        self.pwm.iter().rev().map(|a| a.rev()).collect()
    }

    pub fn peak_height(&self) -> f64 {
        self.peak_height
    }


    pub fn len(&self) -> usize {
        self.pwm.len()
    }



    //PRIORS

    pub fn pwm_prior(&self, seq: &Sequence) -> f64 {

        if seq.kmer_in_seq(&self.best_motif()) {
            //We have to normalize by the probability that our kmer is possible
            //Which ultimately is to divide by the fraction (number unique kmers)/(number possible kmers)
            //number possible kmers = BASE_L^k, but this actually cancels with our integral
            //over the regions of possible bases, leaving only number unique kmers. 
            let mut prior = -((seq.number_unique_kmers(self.len()) as f64).ln()); 

            prior += (self.len() as f64) * BASE_PRIOR_DENS.ln();

            prior

        } else {-f64::INFINITY}
    }

    pub fn pwm_gen_prob(&self, seq: &Sequence) -> f64 {

        if seq.kmer_in_seq(&self.best_motif()) {
            //We have to normalize by the probability that our kmer is possible
            //Which ultimately is to divide by the fraction (number unique kmers)/(number possible kmers)
            //number possible kmers = BASE_L^k, but this actually cancels with our integral
            //over the regions of possible bases, leaving only number unique kmers. 
            let mut prior = -((seq.number_unique_kmers(self.len()) as f64).ln()); 

            prior += (self.len() as f64) * BASE_PRIOR_DENS.ln();

            prior

        } else {-f64::INFINITY}
    }

    pub fn height_prior(&self) -> f64 {

        let mut prior = if self.peak_height > 0.0 { PROB_POS_PEAK.ln() } else { (1.0-PROB_POS_PEAK).ln() };
        prior += HEIGHT_DIST.ln_pdf(self.peak_height.abs());
        prior
    }


    pub fn d_height_prior(&self) -> f64 {
        HEIGHT_DIST.d_ln_pdf(self.peak_height.abs())*self.peak_height.signum()
    }


    //BINDING FUNCTIONS

    //SAFETY: If kmer is not the same length as the pwm, this will produce undefined behavior
    unsafe fn prop_binding(&self, kmer: &[Bp]) -> (f64, bool) { 


        //let kmer: Vec<usize> = kmer_slice.to_vec();

        let mut bind_forward: f64 = 1.0;
        let mut bind_reverse: f64 = 1.0;

        for i in 0..self.len() {
            bind_forward *= self.pwm[i][*kmer.get_unchecked(i)];
            //bind_reverse *= self.pwm[i].rel_bind(BASE_L-1-*kmer.get_unchecked(self.len()-1-i));
            bind_reverse *= self.pwm[i][kmer.get_unchecked(self.len()-1-i).complement()];
        }

        let reverse: bool = bind_reverse > bind_forward;

        let bind: f64 = if reverse {bind_reverse} else {bind_forward};

        return (bind, reverse)

    }

    pub fn return_bind_score(&self, seq: &Sequence) -> (Vec<f64>, Vec<bool>) {

        let coded_sequence = seq.seq_blocks();
        let block_lens = seq.block_lens(); //bp space
        let block_starts = seq.block_u8_starts(); //stored index space


        let mut bind_scores: Vec<f64> = vec![0.0; BP_PER_U8*coded_sequence.len()];
        let mut rev_comp: Vec<bool> = vec![false; BP_PER_U8*coded_sequence.len()];

        let mut uncoded_seq: Vec<Bp> = vec![Bp::A; seq.max_len()];


        let mut ind: usize;

        let mut store: [Bp; BP_PER_U8];


        {
            let uncoded_seq = uncoded_seq.as_mut_slice();
            for i in 0..(block_starts.len()) {


                for jd in 0..(block_lens[i]/BP_PER_U8) {

                    store = Sequence::code_to_bases(coded_sequence[block_starts[i]+jd]);
                    for k in 0..BP_PER_U8 {
                        uncoded_seq[BP_PER_U8*jd+k] = store[k];
                    }

                }


                for j in 0..((block_lens[i])-self.len()) {

                    ind = BP_PER_U8*block_starts[i]+(j as usize);


                    //SAFETY: notice how we defined j, and how it guarentees that get_unchecked is fine
                    let binding_borrow = unsafe { uncoded_seq.get_unchecked(j..(j+self.len())) };
                    (bind_scores[ind], rev_comp[ind]) = unsafe {self.prop_binding(binding_borrow) };

                }

            }
        }



        (bind_scores, rev_comp)



    }
    
    //Warning: if group exceeds the number of blocks in seq, this will just return an empty vector
    pub fn return_any_null_binds_in_group(&self, seq: &NullSequence, groups: &[usize]) -> Vec<f64> {


        if groups.len() == 0 { return Vec::new();}

        let coded_sequence = seq.seq_blocks();
        let block_lens = seq.block_lens(); //bp space
        let block_starts = seq.block_u8_starts(); //stored index space

        let mut bind_scores: Vec<f64> = Vec::new();
        
        let mut uncoded_seq: Vec<Bp> = vec![Bp::A; seq.max_len()];

        let num_blocks = block_lens.len();

        let mut store: [Bp; BP_PER_U8];

        let max_ignore_bind = (-self.peak_height).exp2();

        

        {
            let uncoded_seq = uncoded_seq.as_mut_slice();
            
            for &i in groups.iter() {

                if i >= num_blocks {
                    continue;
                }

                for jd in 0..(block_lens[i]/BP_PER_U8) {

                    store = Sequence::code_to_bases(coded_sequence[block_starts[i]+jd]);
                    for k in 0..BP_PER_U8 {
                        uncoded_seq[BP_PER_U8*jd+k] = store[k];
                    }

                }


                for j in 0..((block_lens[i])-self.len()) {

                    //SAFETY: notice how we defined j, and how it guarentees that get_unchecked is fine
                    let binding_borrow = unsafe { uncoded_seq.get_unchecked(j..(j+self.len())) };
                    let (bind, _) = unsafe {self.prop_binding(binding_borrow) };

                    if bind > max_ignore_bind {
                        bind_scores.push(bind);
                    }
                }

            }
        }


        bind_scores.sort_by(|f, g| g.partial_cmp(f).unwrap());

        if bind_scores.len() > CAPACITY_FOR_NULL {
            _ = bind_scores.drain(CAPACITY_FOR_NULL..).collect::<Vec<_>>();
            

        }


        bind_scores

    }

    pub fn gen_single_mot_binding_report(&self, data_seq: &AllDataUse) -> Vec<(usize, bool, f64)> {

        let data = data_seq.data();

        let (binds, revs) = self.return_bind_score(data.seq());

        let starts = data.seq().block_u8_starts();
        let lens = data.seq().block_lens();

        let zero_locs = data_seq.zero_locs();

        let block_and_id = (0_usize..starts.len()).map(|i| (0_usize..lens[i]).map(move |j| (i,j))).flatten().collect::<Vec<(usize, usize)>>();

        assert!(binds.len() == block_and_id.len(), "{} {}", binds.len(), block_and_id.len());

        let binds_locs = binds.iter().zip(revs.iter()).zip(block_and_id.iter()).map(|((&n,&r),&(l,j))| (zero_locs[l]+j, r, n)).collect::<Vec<_>>();

        binds_locs
    }

    pub fn generate_bedgraph(&self, data_seq: &AllDataUse, chromosome_name: Option<&str>, file_name: &str) -> Result<(), Box<dyn Error>> {

        let chr = chromosome_name.unwrap_or("chr");

        let mut binds_locs = self.gen_single_mot_binding_report(data_seq);
        
        binds_locs.sort_by(|(f,_,_), (g,_,_)| g.partial_cmp(f).unwrap());

        let mut file = fs::File::create(file_name)?;

        let mut buffer: Vec<u8> = Vec::with_capacity(binds_locs.len()*(chr.len()+22));

        for (id, (base, is_rev, bind)) in binds_locs.into_iter().enumerate() {

            let strand = if is_rev { "-" } else { "+" };
            let end_base = base+self.len();
            let thousand_score = (bind*1000.) as usize;

            if thousand_score == 0 { break; }

            let line = format!("{chr}\t{base}\t{end_base}\tbind_{id}\t{thousand_score}\t{strand}\n");
            let mut buffer_line = line.into_bytes();
            buffer.append(&mut buffer_line);
        }

        file.write(&buffer)?;

        Ok(())

    }




    pub fn generate_waveform<'a>(&self, data_ref: &'a AllDataUse) -> Waveform<'a> {

        let data = data_ref.data();
        
        let unit_kernel = data_ref.unit_kernel_ref();

        let mut occupancy_trace: Waveform = data.derive_zero();

        let mut actual_kernel: Kernel;

        let starts = data.seq().block_u8_starts();

        let lens = data.seq().block_lens();

        let (bind_score_floats, _) = self.return_bind_score(data.seq());

        let thresh = (-self.peak_height).exp2();

        for i in 0..starts.len() { //Iterating over each block
            for j in 0..(lens[i]-self.len()) {
                //if bind_score_floats[starts[i]*BP_PER_U8+j] > *THRESH.read().expect("no writes expected now") {i //}
                //SAFETY: 
                //  -block = i, which is always less than the number of sequence blocks
                //  -center = j, which is always less than the number of bps in the Sequence by at least the motif length if Sequence is correctly constructed
                //SAFETY: THRESH is never modified at this point

                let bind = unsafe{*bind_score_floats.get_unchecked(starts.get_unchecked(i)*BP_PER_U8+j)};
                if bind > thresh {
                    actual_kernel = unit_kernel*(bind.log2()+self.peak_height) ;

                    unsafe {occupancy_trace.place_peak(&actual_kernel, i, j+(self.len()-1)/2);} 

                }
            }
        }

        //println!("num peaks {}", count);


        occupancy_trace

    }


    //Safety: You MUST ensure that the binding score and reverse complement is valid for this particular motif, because you can technically use 
    //        ANY binding score here, and this code won't catch it, especially if the dimensions check out. We code this primarily for speed of calculation in the gradient calculation
    #[allow(dead_code)]
    unsafe fn generate_waveform_from_binds<'a>(&self, binds: &(Vec<f64>, Vec<bool>), data_ref: &'a AllDataUse) -> Waveform<'a> {

        let data = data_ref.data();
        let unit_kernel = data_ref.unit_kernel_ref();

        let mut occupancy_trace: Waveform = data.derive_zero();

        let mut actual_kernel: Kernel;

        let starts = data.seq().block_u8_starts();

        let lens = data.seq().block_lens();

        let h = self.peak_height;

        let cutoff = (-h).exp2();


        for i in 0..starts.len() { //Iterating over each block
            for j in 0..(lens[i]-self.len()) { //-self.len() is critical for maintaining safety of place_peak. 
                                               //It's why we don't allow sequence blocks unless they're bigger than the max motif size
                //if binds.0[starts[i]*BP_PER_U8+j] > *THRESH.read().expect("no writes expected now") { //}
                //SAFETY: THRESH is never modified at this point
                if binds.0[starts[i]*BP_PER_U8+j] > cutoff {
                    actual_kernel = unit_kernel*(binds.0[starts[i]*BP_PER_U8+j].log2()+h) ;
                    //println!("{}, {}, {}, {}", i, j, lens[i], actual_kernel.len());
                    occupancy_trace.place_peak(&actual_kernel, i, j+(self.len()-1)/2);//SAFETY Note: this technically means that we round down if the motif length is even
                                                                                      //This looks like we can violate the safety guarentee for place peak, but return_bind_score()
                                                                                      //has zeros where we can be going over the motif length. Because THRESH forbids trying
                                                                                      //to place peaks under a certain strength, this preserves our safety guarantees

                                                                                      //println!("peak {} {} {}", starts[i]*BP_PER_U8+j, bind_score_floats[starts[i]*BP_PER_U8+j], occupancy_trace.read_wave()[(starts[i]*BP_PER_U8+j)/data.spacer()]);
                }
            }
        }

        occupancy_trace

    }


    /// #Example
    /// ```
    /// use motif_finder::base::{Motif, Bp, Base};
    /// let mut rng = rand::thread_rng();
    /// let m = Motif::from_motif(vec![Bp::A, Bp::C, Bp::G, Bp::G, Bp::G, Bp::G, Bp::T, Bp::T, Bp::T], &mut rng);
    /// let dself = m.distance_function(&m);
    /// assert!(dself.0.abs() < 1e-16);
    /// assert!(!dself.1);
    /// let drev = m.distance_function(&(Motif::raw_pwm(m.rev_complement(), m.peak_height())));
    /// assert!(drev.0.abs() < 1e-16);
    /// assert!(drev.1);
    /// let mut new_b_vec: Vec<Base> = Vec::with_capacity(m.len()+2);
    /// new_b_vec.append(&mut m.rev_complement());
    /// new_b_vec.push(Base::rand_new(&mut rng));
    /// new_b_vec.push(Base::rand_new(&mut rng));
    /// let newmot = Motif::raw_pwm(new_b_vec, m.peak_height());
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


        distance.sqrt()

    }

    fn sum_motifs(&self, other_mot: &Motif) -> Result<(Self, f64), InvalidBase> {

        let (pwm_short, pwm_long) = if self.len() <= other_mot.len() {(&self, &other_mot)} else {(&other_mot, &self)};

        let total_len = pwm_long.len();

        let short_len = pwm_short.len();

        let (_, rc) = pwm_long.distance_function(pwm_short);
        
        let off_center = ((pwm_long.len()-1)/2) - ((pwm_short.len()-1)/2);

        //sum_bases_and_ln_max

        let mut correction: f64 = 0.0;

        //SAFETY: all 1.0 fulfills all the invariants of Base::new necessary not to cause an error
        //        The specific values here are also unimportant
        let mut new_pwm: Vec<Base> = (0..total_len).map(|_| unsafe{ Base::new([1.0; BASE_L]).unwrap_unchecked()}).collect();

        if off_center > 0 { 
            for i in 0..off_center {
                //TODO: change this to get_unchecked if there's a performance benefit
                new_pwm[i] =  pwm_long.pwm[i].clone();
            }
        }
        
        if (short_len+off_center) < total_len {
            for i in (short_len+off_center)..total_len {
                new_pwm[i] = pwm_long.pwm[i].clone();
            }
        }

        for ind in 0..short_len {

            let b1 = &pwm_short.pwm[ind];
            let b2 = &pwm_long.pwm[off_center+ind];

            match b2.sum_bases_and_ln_max(b1) {

                Ok((b, s)) => {
                new_pwm[off_center+ind] = b;

                correction += s;
                },
                Err(e) => { return Err(e);}
            };
        }

        //"correction" is a sum of ln(proporition products), which makes it a negative dimensionless free energy
        //We want to ADD the sum of dimensionless free energies, so we want to SUBTRACT correction
        let new_height = self.peak_height()+other_mot.peak_height-correction;

        Ok((Motif::raw_pwm(new_pwm, new_height), correction))
        

    }

    fn for_meme(&self) -> String {

        const MEME_DIGITS: usize = 6;

        let mut meme_string = format!("letter-probability matrix: alength= {} w= {}\n", BPS.len(), self.len());

        for base in self.pwm.iter() {
            let probs = base.as_probabilities();
            for prob in probs.iter() {
                meme_string = format!("{} {:.MEME_DIGITS$} ", meme_string, prob);
            }
            meme_string = format!("{}\n", meme_string);
        }
        meme_string = format!("{}\n", meme_string);

        meme_string
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

            let base = self.pwm[i].show();
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
        write!(f, "Peak height: {:.DIGITS$}.\n", self.peak_height)?;

        //I want people to be free to define their own bases, if they like
        //But they will have to change it in the code for formatting: it's not my problem
        //The two extra spaces on either side let the output align in a pretty way
        for b in BPS { write!(f, "  {:<DIGITS$}  ", b)?; }
        write!(f, "\n")?;

        for i in 0..self.pwm.len() {

            let base = self.pwm[i].show();
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

#[derive(Clone)]
//DEFINITELY CANNOT BE DIRECTLY SERIALIZED
//
pub struct MotifSet<'a> {

    set: Vec<Motif>, 
    signal: Waveform<'a>,
    ln_post: Option<f64>,
    null_seq_attentions: Vec<usize>,
    null_peak_scores: Vec<f64>,
    data_ref: &'a AllDataUse<'a>, 
}

impl<'a> MotifSet<'a> {

    pub fn rand_with_one<R: Rng+?Sized>(data_ref: &'a AllDataUse<'a>, rng: &mut R) -> Self {

        let mut valid: bool;

        let mot_set = loop { //We occasionally seem to randomly generate sets with such high occupancy everywhere as to be ridiculous

            let set = vec![Motif::rand_mot(data_ref.data().seq(), rng)];

            let signal = set[0].generate_waveform(data_ref);

            let mut mot_set_try = MotifSet{ set: set, signal: signal, ln_post: None, null_seq_attentions: Vec::new(), null_peak_scores: Vec::new(), data_ref: data_ref};

            let like = mot_set_try.ln_posterior();

            valid = like.is_finite();

            if valid {
                break mot_set_try;
            }
        };

        mot_set
    }

    pub fn rand_with_one_height<R: Rng+?Sized >(peak_height: f64, data_ref: &'a AllDataUse<'a>, rng: &mut R) -> Self {

        let set = vec![Motif::rand_mot_with_height(peak_height, data_ref.data().seq(), rng)];

        let signal = set[0].generate_waveform(data_ref);

        let mut mot_set = MotifSet{ set: set,  signal: signal, ln_post: None, null_seq_attentions: Vec::new(), null_peak_scores: Vec::new(), data_ref: data_ref};

        let _ = mot_set.ln_posterior();

        mot_set
    }

    pub fn rand_with_one_height_and_motif_len<R: Rng + ?Sized>(peak_height: f64, motif_len: usize, data_ref: &'a AllDataUse<'a>, rng: &mut R) -> Self {


        let set = vec![Motif::rand_mot_with_height_and_motif_len(peak_height, motif_len, data_ref.data().seq(), rng)];

        let signal = set[0].generate_waveform(data_ref);

        let mut mot_set = MotifSet{ set: set, signal: signal, ln_post: None, null_seq_attentions: Vec::new(), null_peak_scores: Vec::new(), data_ref: data_ref};

        let _ = mot_set.ln_posterior();

        mot_set
    }
    
    
 
    pub fn set_from_meme<R: Rng+?Sized>(meme_file_name: &str, data_ref: &'a AllDataUse<'a>, e_value_cutoff: f64, rng: &mut R) -> Result<Self, Box<dyn Error>> {

        let meme_file_string = fs::read_to_string(meme_file_name)?;
        let meme_as_vec = meme_file_string.split("\n").collect::<Vec<_>>();

        //This should be infallible, because it never changes and I already validated it once. 
        //If THIS is where you're breaking, this whole function is invalid and needs to be fixed. 
        let re = Regex::new(r"letter-probability matrix: alength= (\d+)  w= (\d+) nsites= \d+ E= (\d+\.\de[-+]\d\d\d)")?;

        let start_matrix_lines: Vec<usize> = meme_as_vec.iter().enumerate().filter(|&(_,a)| re.is_match(a)).map(|(n, _)| n).collect();

        if start_matrix_lines.len() == 0 { return Err(Box::new(MemeParseError::EmptyMatrix)); }

        let mut set: Vec<Motif> = Vec::new();

        for (mot_num, &line) in start_matrix_lines.iter().enumerate() {
            let captures = re.captures(meme_as_vec[line]).expect("We can only get here because we matched on regex");
            let alphabet_len: usize = captures[1].parse().expect("We can only get here because we matched on regex");

            if alphabet_len != BASE_L { return Err(Box::new(MemeParseError::InvalidAlphabet{ motif_num: mot_num }));}

            let motif_len: usize = captures[2].parse().unwrap();
            if (motif_len < MIN_BASE) || (motif_len > MAX_BASE) { return Err(Box::new(MemeParseError::InvalidMotifLength{ motif_num: mot_num, captured_length: motif_len })); }


            let e_value: f64 = captures[3].parse().unwrap();

            if (set.len() > 0) && (e_value > e_value_cutoff) {
                break;
            }

            let mut base_vec: Vec<Base> = Vec::with_capacity(MAX_BASE);

            for i in 1..(motif_len+1) {
                let mut props: [ f64; BASE_L] = [0.0; BASE_L];
                let mut line_split = meme_as_vec[line+i].split_whitespace();
                for j in 0..BASE_L {
                    let Some(prop_str) = line_split.next() else { return Err(Box::new(MemeParseError::ColumnLengthFailure{line_num: line+i+1})); };
                    let Ok(prop)= prop_str.parse::<f64>() else { return Err(Box::new(MemeParseError::FloatParseFailure{line_num: line+i+1}));}; 
                    props[j] = prop;
                }

                base_vec.push(Base::new(props)?);
            }

            let mut motif = Motif::rand_height_pwm(base_vec, rng);

            let poss_hamming = motif.scramble_to_close_random_valid(data_ref.data().seq(), &mut Some(rng));

            match poss_hamming {
                Some(hamming) => warn!("{}", format!("Motif number {} from the MEME file does not exist in the parts of the sequence with peaks! Moving it to a valid motif within a Hamming distance of {}!", mot_num, hamming)),
                None => (),
            };


            set.push(motif);    

        }


        let mut signal = data_ref.data().derive_zero();

        for mot in &set {
            signal += &(mot.generate_waveform(data_ref));
        }

        let mut full_set = MotifSet {
            set: set,
            signal: signal, 
            ln_post: None, 
            null_seq_attentions: Vec::new(),
            null_peak_scores: Vec::new(),
            data_ref: data_ref, 
        };

        let _ = full_set.ln_posterior();

        Ok(full_set)

    }


    pub fn set_from_json<R: Rng+?Sized>(data_ref: &'a AllDataUse<'a>, json_file: &str, _rng: &mut R) -> Result<Self, Box<dyn Error>> {

        let json_string: String = fs::read_to_string(json_file)?;

        let prior_state: StrippedMotifSet = serde_json::from_str(&json_string)?;

        Ok(prior_state.reactivate_set(data_ref))

        //self.push_set_def(always_recalculate, validate_motif, prior_state, rng);

    }
 
    pub fn set_from_bincode<R: Rng+?Sized>(data_ref: &'a AllDataUse<'a>, bincode_file: &str, _rng: &mut R) -> Result<Self, Box<dyn Error>> {


        let mut bincode_file_handle = fs::File::open(bincode_file)?;

        let mut buffer: Vec<u8> = Vec::new();

        bincode_file_handle.read_to_end(&mut buffer)?;

        let prior_state: StrippedMotifSet = bincode::deserialize(&buffer)?;

        println!("bincode state {:?}", prior_state.set);
        Ok(prior_state.reactivate_set(data_ref))

    }

    pub fn recalced_signal(&self) -> Waveform {
        let mut signal = self.data_ref.data().derive_zero();
        for mot in self.set.iter() {
            signal += &(mot.generate_waveform(self.data_ref));
        }

        signal
    }

    fn recalc_signal(&mut self) {
        self.signal = self.data_ref.data().derive_zero();
        for mot in self.set.iter() {
            self.signal += &(mot.generate_waveform(self.data_ref));
        }
        self.ln_post = None;
    }

    pub fn save_set_trace_and_sub_traces(&self, data_ref: &AllDataUse, output_dir: &str, file_name: &str) {

        let signal = self.recalced_signal();

        let signal_directory: String = format!("{}/{}_occupancy",output_dir,file_name);

        signal.save_waveform_to_directory(data_ref, &signal_directory, &BLUE);

        if self.set.len() > 1 {
            for (i, _mot) in self.set.iter().enumerate() {

                let sub_directory = format!("{}/Motif_{}", signal_directory, i);
                let sub_signal = self.set[i].generate_waveform(data_ref);
                sub_signal.save_waveform_to_directory(data_ref, &sub_directory, &GREEN);

            }
        }

    }

    //I deliberately chose not to scale anything by peak heights here
    pub fn generate_set_binding_report(&self, data_ref:&AllDataUse) -> Vec<(usize, usize, usize, f64)> {

        let mut binding = self.set[0].gen_single_mot_binding_report(data_ref).into_iter().map(|(a,_,c)| (a, a + self.set[0].len(), 0, c)).collect::<Vec<(usize, usize, usize, f64)>>();

        if self.set.len() > 1 { for i in 1..self.set.len() {

            let new_binding = self.set[i].gen_single_mot_binding_report(data_ref).into_iter().map(|(a,_,c)| (a, a + self.set[i].len(), i, c)).collect::<Vec<(usize, usize, usize, f64)>>();

            for j in 0..binding.len() {
                if new_binding[j].2 > binding[j].2 {
                    binding[j] = new_binding[j];
                }
            }

        }}

        binding

    }

    pub fn generate_set_binding_report_with_omission(&self, data_ref:&AllDataUse, omit: usize) -> Vec<(usize, usize, usize, f64)>  {

        if self.set.len() == 1 { return Vec::new();} 

        if omit >= self.set.len() { return self.generate_set_binding_report(data_ref); }

        let mut binding = if omit == 0 {
            self.set[1].gen_single_mot_binding_report(data_ref).into_iter().map(|(a,_,c)| (a, a + self.set[0].len(), 0, c)).collect::<Vec<(usize, usize, usize, f64)>>()
        } else {
            self.set[0].gen_single_mot_binding_report(data_ref).into_iter().map(|(a,_,c)| (a, a + self.set[0].len(), 0, c)).collect::<Vec<(usize, usize, usize, f64)>>()
        };

        if self.set.len() == 2 { return binding;}

        let start_id = if omit == 1 {2_usize} else {1};

        for i in start_id..self.set.len() {

            if i != omit {
                let new_binding = self.set[i].gen_single_mot_binding_report(data_ref).into_iter().map(|(a,_,c)| (a, a + self.set[i].len(), i, c)).collect::<Vec<(usize, usize, usize, f64)>>();

                for j in 0..binding.len() {
                    if new_binding[j].2 > binding[j].2 {
                        binding[j] = new_binding[j];
                    }
                }
            }
        }

        binding


    }

    //This gives a vector of precisions for each peak recalled
    //If somehow, nothing pings on a peak that should be recalled, there will be a zero entry
    //So for example, if this gives vec![1.0, 0.5, 0.6, 0.0, 0.0], there were five peaks that 
    //should have been recalled: one was found at the top, one was found in position 4, 
    //one was found in position 5, and the other two were never found somehow (this is probably an error)
    pub fn precision_for_each_recall(&self, data_ref:&AllDataUse, peak_locations: &[((usize, usize), f64)]) -> Vec<f64> {

        let mut binding_scores = self.generate_set_binding_report(data_ref);
        binding_scores.sort_by(|(_,_,_,f), (_,_,_,g)| g.partial_cmp(f).unwrap());

        let mut precisions: Vec<f64> = vec![0.0; peak_locations.len()];

        let mut try_hits: f64 = 0.0;
        let mut hits: f64 = 0.0; 
        let mut hit_id: usize = 0;

        for (start_loc, end_loc, _, _) in binding_scores {
            
            try_hits += 1.0;

            let mut found = false;

            for ((kern_start, kern_end), _) in peak_locations {

                if start_loc >= *kern_end { break; }

                if (start_loc >= *kern_start) && (end_loc <= *kern_end) {
                    found = true;
                }

            }

            if found {
                hits += 1.0;
                precisions[hit_id] = hits/try_hits;
                hit_id += 1;
            }

        }

        precisions
    }

    pub fn precision_for_each_recall_omit_motif(&self, data_ref:&AllDataUse, peak_locations: &[((usize, usize), f64)], omit: usize) -> Vec<f64> {

        let mut binding_scores = self.generate_set_binding_report_with_omission(data_ref, omit);
        binding_scores.sort_by(|(_,_,_,f), (_,_,_,g)| g.partial_cmp(f).unwrap());

        let mut precisions: Vec<f64> = vec![0.0; peak_locations.len()];

        let mut try_hits: f64 = 0.0;
        let mut hits: f64 = 0.0;
        let mut hit_id: usize = 0;

        for (start_loc, end_loc, _, _) in binding_scores {

            try_hits += 1.0;

            let mut found = false;

            for ((kern_start, kern_end), _) in peak_locations {

                if start_loc >= *kern_end { break; }

                if (start_loc >= *kern_start) && (end_loc <= *kern_end) {
                    found = true;
                }

            }

            if found {
                hits += 1.0;
                precisions[hit_id] = hits/try_hits;
                hit_id += 1;
            }

        }

        precisions

    }

    /*pub fn generate_pr_curve(&self, data_ref: &AllDataUse, omit_weakest: bool, base_file_name: &str) {

        let file = format!("{}{}", base_file_name,"_pr_curve.png");
        
        let area = BitMapBackend::new(&file, (8000, 4000)).into_drawing_area();

        let mut pr = ChartBuilder::on(&area);

        pr.margin(10).y_label_area_size(200).x_label_area_size(200);

        pr.caption("Precision vs recall", ("Times New Roman", 80));

        let peak_locations = data_ref.basic_peak(MIN_HEIGHT);

        let recall: Vec<f64> = (0..peak_locations.len()).map(|a| ((a+1) as f64)/(peak_locations.len() as f64)).collect();

        let precision: Vec<f64> = self.precision_for_each_recall(data_ref, &peak_locations);

        let mut pr_context = pr.build_cartesian_2d(0_f64..1_f64, 0_f64..1_f64).unwrap();

        pr_context.configure_mesh().x_label_style(("sans-serif", 70))
            .y_label_style(("sans-serif", 70))
            .x_label_formatter(&|v| format!("{:.0}", v))
            .axis_desc_style(("sans-serif",90))
            .x_desc("Recall")
            .y_desc("Precision").disable_x_mesh().disable_y_mesh().x_label_formatter(&|x| format!("{:.04}", *x)).draw().unwrap();

        pr_context.draw_series(LineSeries::new(precision.iter().zip(recall.iter()).map(|(&k, &i)| (i, k)), &BLUE)).unwrap().label("Total Motif Set");

        if omit_weakest && self.set.len() > 1 {

            let weakest = self.set.iter().map(|a| a.peak_height()).enumerate().min_by(|(_, a), (_,b)| a.partial_cmp(b).unwrap()).unwrap().0;

            let precision_weakest: Vec<f64> = self.precision_for_each_recall_omit_motif(data_ref,&peak_locations, weakest);

            pr_context.draw_series(LineSeries::new(precision_weakest.iter().zip(recall.iter()).map(|(&k, &i)| (i, k)), &BLUE)).unwrap().label("Omit Weakest Motif");

        }


    }*/

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
            thresh < diff_ln_like.exp()
        }
    }

    fn derive_set(&self) -> Self {

        MotifSet {
            set: self.set.clone(),
            signal: self.signal.clone(),
            null_seq_attentions: self.null_seq_attentions.clone(),
            null_peak_scores: self.null_peak_scores.clone(),
            ln_post: None,
            data_ref: self.data_ref, //pointer
        }

    }

    pub fn get_nth_motif(&self, n: usize) -> Motif {
        self.set[n].clone()
    }

    //This is our prior on the number of motifs
    //We do not justify this with a maximum entropy prior
    //Instead, we only want to consider an additional motif if 
    //it brings an improvement of at least NECESSARY_MOTIF_IMPROVEMENT to the ln posterior
    //This amounts to a geometric prior with positive integer support 
    //and p = 1-(-NECESSARY_MOTIF_IMPROVEMENT).exp().
    //NOTE: the ommission of ln(p) term is deliberate. It amounts to a normalization constant
    //for the motif set prior, and would obfuscate the true point of this prior
    pub fn motif_num_prior(&self) -> f64 {
        -((self.set.len()-1) as f64)* unsafe{ NECESSARY_MOTIF_IMPROVEMENT } 
    }

    pub fn ln_prior(&self) -> f64 {
        self.motif_num_prior() + self.set.iter().map(|a| a.height_prior()+a.pwm_prior(self.data_ref.data().seq())).sum::<f64>()
    }

    pub fn ln_likelihood(&self) -> f64 {
        Noise::ad_like((self.signal).produce_noise_with_extraneous(self.data_ref, &self.null_peak_scores).ad_calc(self.data_ref.data().spacer()))
    }

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

    pub fn calc_ln_post(&self) -> f64 { //ln_posterior() should be preferred if you can mutate self, since it ensures the calculation isn't redone too much
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

    pub fn median_data_dist(&self) -> f64 {
        self.signal.median_distance_between_waves(self.data_ref.data())
    }
    fn add_motif(&mut self, new_mot: Motif) -> f64 {

        self.signal += &new_mot.generate_waveform(self.data_ref) ;
        self.set.push(new_mot);
        self.ln_post = None;
        self.ln_posterior()

    }

    fn insert_motif(&mut self, new_mot: Motif, position: usize) -> f64 {

        self.signal += &new_mot.generate_waveform(self.data_ref) ;
        self.set.insert(position, new_mot);
        self.ln_post = None;
        self.ln_posterior()

    }

    fn remove_motif(&mut self, rem_id: usize) -> f64{
        assert!(rem_id < self.set.len());

        let rem_mot = self.set.swap_remove(rem_id);
        self.signal -= &rem_mot.generate_waveform(self.data_ref);
        self.ln_post = None;
        self.ln_posterior()

    }

    fn remove_motif_void(&mut self, rem_id: usize) {

        assert!(rem_id < self.set.len());
        let rem_mot = self.set.swap_remove(rem_id);
        self.signal -= &rem_mot.generate_waveform(self.data_ref);
        self.ln_post = None;

    }

    pub fn replace_motif(&mut self, new_mot: Motif, rem_id: usize) -> f64 {
        let rem_mot = self.set[rem_id].clone();
        self.signal -= &rem_mot.generate_waveform(self.data_ref);
        self.signal += &new_mot.generate_waveform(self.data_ref) ;
        self.set[rem_id] = new_mot;
        self.ln_post = None;
        self.ln_posterior()
    }

    pub fn current_attention(&self) -> &Vec<usize> {
        &self.null_seq_attentions
    }

    pub fn change_set_attention(&mut self, negative_blocks: Vec<usize>) -> f64 {

        self.null_seq_attentions = negative_blocks;

        self.null_peak_scores = if self.null_seq_attentions.len() == 0 {Vec::new()} else {
            self.set.iter().map(|a| a.return_any_null_binds_in_group(self.data_ref.null_seq(), &self.null_seq_attentions)
                                     .iter().map(|&b| a.peak_height()+b.log2()).filter(|&b| b > (self.data_ref.background_ref().noise_spread_par() * 4.0)).collect::<Vec<f64>>()).flatten().collect::<Vec<f64>>()
        };

        self.ln_post = None;
        self.ln_posterior()

    }

    pub fn set_with_new_attention(&self, negative_blocks: Vec<usize>) -> (Self, f64) {

        let mut clone = self.clone();
        let clone_like = clone.change_set_attention(negative_blocks);

        (clone, clone_like)
    }


    //This proposes a new motif for the next motif set, but does not do any testing vis a vis whether such a move will be _accepted_
    fn propose_new_motif<R: Rng + ?Sized>(&self, rng: &mut R ) -> Option<(Self, f64)> {
        let mut new_set = self.derive_set();

        let new_mot = Motif::rand_mot_dirichlet(self.data_ref.data().seq(), rng); //There may not always be a place with decent propensities

        let pick_prob = new_mot.pwm.iter().map(|a| DIRICHLET_PWM.get().expect("no writes expected now").ln_pdf(a)).sum::<f64>();

        let ln_gen_prob = HEIGHT_PROPOSAL_DIST.ln_pdf(new_mot.peak_height-MIN_HEIGHT)+pick_prob-(((MAX_BASE+1-MIN_BASE) as f64).ln() + (self.data_ref.data().seq().number_unique_kmers(new_mot.len()) as f64).ln());
        //let h_prior = new_mot.height_prior();

        let ln_post = new_set.add_motif(new_mot);
        //println!("propose birth: like: {} height: {}, pick_prob: {}, len sel: {}",ln_post, h_prior, pick_prob.ln(), ((MAX_BASE+1-MIN_BASE) as f64).ln());
        Some((new_set, ln_post-ln_gen_prob)) //Birth moves subtract the probability of their generation
    }

    //This proposes removing an old motif for the next motif set
    //It does no testing save for checking if removing a motif would produce a possible set
    fn propose_kill_motif<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<(Self, f64)> {

        if self.set.len() <= 1 { //We never want to propose a set with no motifs, ever
            None
        } else {
            let mut new_set = self.derive_set();
            let rem_id = rng.gen_range(0..self.set.len());

            let pick_prob = self.set[rem_id].pwm.iter().map(|a| DIRICHLET_PWM.get().expect("no writes expected now").ln_pdf(a)).sum::<f64>();

            let ln_gen_prob = HEIGHT_PROPOSAL_DIST.ln_pdf(self.set[rem_id].peak_height-MIN_HEIGHT)+pick_prob-(((MAX_BASE+1-MIN_BASE) as f64).ln() + (self.data_ref.data().seq().number_unique_kmers(self.set[rem_id].len()) as f64).ln());

            let ln_post = new_set.remove_motif(rem_id);
            //println!("propose death: like: {} height: {}, pick_prob: {}, len sel: {}",ln_post, self.set[rem_id].height_prior(), pick_prob.ln(), ((MAX_BASE+1-MIN_BASE) as f64).ln());
            Some((new_set, ln_post+ln_gen_prob)) //Death moves add the probability of the generation of their deleted variable(s)
        }
    }
    /*
       fn propose_split_motif<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<(Self, f64)> {

       let heights: Vec<Option<f64>> = self.set.iter().map(|a| a.

       }

       fn propose_merge_motif<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<(Self, f64)> {

       }
     */

    //I'm only extending motifs on one end
    //This saves a bit of time from not having to reshuffle motif vectors
    fn propose_extend_motif<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<(Self, f64)> {
        let mut new_set = self.derive_set();

        let extend_id = rng.gen_range(0..self.set.len());

        if self.set[extend_id].len() >= MAX_BASE { //We want to immediately reject any move which extends a motif beyond maximum length
            None
        } else {

            let mut add_base_mot = self.set[extend_id].best_motif();
            add_base_mot.push(Bp::A);
            let last_ind = add_base_mot.len()-1;
            let mut valid_extends: Vec<Bp> = Vec::with_capacity(BASE_L);
            for &b in BP_ARRAY.iter() {
                add_base_mot[last_ind] = b;
                if self.data_ref.data().seq().kmer_in_seq(&add_base_mot) { valid_extends.push(b); }
            }
            if valid_extends.len() == 0 { return None; }

            let new_bp = valid_extends.choose(rng).expect("We already returned if there are no valid extensions");
            
            let mut new_mot = self.set[extend_id].clone();
            let new_base = Base::propose_safe_new(rng).make_best(*new_bp);
            let base_ln_density = PROPOSE_EXTEND.get().expect("no writes expected now").ln_pdf(&new_base) - (valid_extends.len() as f64).ln();
            new_mot.pwm.push(new_base);
            //let ln_gen_prob = new_mot.height_prior()+new_mot.pwm_prior(self.data.seq());
            let ln_post = new_set.replace_motif(new_mot, extend_id);
            Some((new_set, ln_post-base_ln_density)) //Birth moves subtract the probability of their generation
        }
    }

    //I'm only extending motifs on one end
    //This saves a bit of time from not having to reshuffle motif vectors
    fn propose_contract_motif<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<(Self, f64)> {
        let mut new_set = self.derive_set();

        let contract_id = rng.gen_range(0..self.set.len());

        if self.set[contract_id].len() <= MIN_BASE { //We want to immediately reject any move which contracts a motif below minimum length
            None
        } else {
            let mut new_mot = self.set[contract_id].clone();
            let new_mot_bps = new_mot.best_motif();
            let old_base = new_mot.pwm.pop();
            let ln_post = new_set.replace_motif(new_mot, contract_id);
            let base_ln_density = PROPOSE_EXTEND.get().expect("no writes expected now").ln_pdf(&(old_base.expect("We know this is bigger than 0")))-self.data_ref.data().seq().number_kmer_reextends(&new_mot_bps).ln();
            Some((new_set, ln_post+base_ln_density)) //Birth moves subtract the probability of their generation
        }
    }



    //MOVE TO CALL
    //For borrow checker reasons, this will only work if the motif calling already has a generated likelihood
    //And I can't have it do it here
    //If you use this on a motif without such a likelihood, it will panic
    // (rmse, likelihood_diff, pwm_dists_and_height_diffs)
    pub fn run_rj_move<R: Rng + ?Sized>(&self, thermo_beta: f64, rng: &mut R) -> (Self, usize, bool) {

        let which_rj = rng.gen_range(0..4);

        let proposal: Option<(Self, f64)> = match which_rj {

            0 => self.propose_new_motif(rng),
            1 => self.propose_kill_motif(rng),
            2 => self.propose_extend_motif(rng),
            3 => self.propose_contract_motif(rng),
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



    //MOVE TO CALL
    pub fn base_leap<R: Rng + ?Sized>(&self, thermo_beta: f64, rng: &mut R) -> Self {


        //We want to shuffle this randomly, in case there is some kind of codependency between particular TFs
        let mut ids: Vec<usize> = (0..self.set.len()).collect();
        ids.shuffle(rng);

        let mut current_set = self.clone();

        for id in ids {

            let current_mot = current_set.set[id].clone();

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
        }

        current_set

    }
    
    pub fn secondary_shuffle<R: Rng + ?Sized>(&self, thermo_beta: f64, rng: &mut R) -> Self {


        //We want to shuffle this randomly, in case there is some kind of codependency between particular TFs
        let mut ids: Vec<usize> = (0..self.set.len()).collect();
        ids.shuffle(rng);

        let mut current_set = self.clone();

        for id in ids {

            let current_mot = current_set.set[id].clone();

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
        }

        current_set

    }
    
    
    pub fn randomize_motifs<R: Rng + ?Sized>(&self, thermo_beta: f64, rng: &mut R) -> Self {


        //We want to shuffle this randomly, in case there is some kind of codependency between particular TFs
        let mut ids: Vec<usize> = (0..self.set.len()).collect();
        ids.shuffle(rng);

        let mut current_set = self.clone();

        for id in ids {

            let current_mot = current_set.set[id].clone();

            let mut base_set = current_set.clone();
            base_set.remove_motif_void(id);

            let mut kmer_ids = self.data_ref.data().seq().n_random_valid_motifs(current_mot.len(), 200, rng);

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
        }

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

    fn propose_ordered_base_move_custom<R: Rng + ?Sized>(&self, rng: &mut R , ratio_sd: f64, linear_sd: f64) -> Option<(Self, f64)> {

        let mut new_set = self.derive_set();

        let c_id = rng.gen_range(0..self.set.len());

        let mut replacement = new_set.set[c_id].clone();

        let base_change = rng.gen_range(0..replacement.pwm.len());

        let Some(attempt_new) = replacement.pwm[base_change].moved_base(ratio_sd, linear_sd, rng) else { return None;};

        replacement.pwm[base_change] = attempt_new;

        let ln_post = new_set.replace_motif(replacement, c_id);

        Some((new_set, ln_post))
    }

    fn propose_ordered_motif_move_custom<R: Rng + ?Sized>(&self, rng: &mut R, ratio_sd: f64, linear_sd: f64) -> Option<(Self, f64)> {


        let mut new_set = self.derive_set();

        let c_id = rng.gen_range(0..self.set.len());

        let mut replacement = new_set.set[c_id].clone();
        
        let scaler = REDUCE_MOTIF_SCALE_MOVE[self.set[c_id].len()-MIN_BASE]*0.25;

        let Some(attempt_new) = replacement.pwm.iter().map(|a| a.moved_base(ratio_sd*scaler, linear_sd*scaler, rng)).collect::<Option<Vec<Base>>>() else {return None;};

        replacement.pwm = attempt_new;

        let ln_post = new_set.replace_motif(replacement, c_id);

        Some((new_set, ln_post))
    }


    fn propose_height_move_custom<R: Rng + ?Sized>(&self, rng: &mut R, height_sd: f64 ) -> Option<(Self, f64)> {

        let mut new_set = self.derive_set();

        let c_id = rng.gen_range(0..self.set.len());

        let mut replacement = new_set.set[c_id].clone();

        replacement.peak_height += NORMAL_DIST.sample(rng)*height_sd;

        replacement.peak_height = replacement.peak_height.signum()*reflect_abs_height(replacement.peak_height.abs());

        let ln_post = new_set.replace_motif(replacement, c_id);

        Some((new_set, ln_post))

    }

    fn propose_scale_base_custom<R: Rng+?Sized>(&self, rng: &mut R, scale_exponent_sd: f64 ) -> Option<(Self, f64)> {

        let mut new_set = self.derive_set();

        let c_id = rng.gen_range(0..self.set.len());

        let mut replacement = new_set.set[c_id].clone();

        let base_change = rng.gen_range(0..replacement.pwm.len());

        let Some(attempt_new) = replacement.pwm[base_change].scaled_base(scale_exponent_sd, rng) else { return None;};

        replacement.pwm[base_change] = attempt_new;

        let ln_post = new_set.replace_motif(replacement, c_id);

        Some((new_set, ln_post))


    }

    fn propose_ordered_scale_move_custom<R: Rng + ?Sized>(&self, rng: &mut R, scale_exponent_sd: f64) -> Option<(Self, f64)> {


        let mut new_set = self.derive_set();

        let c_id = rng.gen_range(0..self.set.len());

        let mut replacement = new_set.set[c_id].clone();

        let scaler = REDUCE_MOTIF_SCALE_MOVE[self.set[c_id].len()-MIN_BASE]*0.25;

        let Some(attempt_new) = replacement.pwm.iter().map(|a| a.scaled_base(scale_exponent_sd*scaler, rng)).collect::<Option<Vec<Base>>>() else {return None;};

        replacement.pwm = attempt_new;

        let ln_post = new_set.replace_motif(replacement, c_id);

        Some((new_set, ln_post))
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
            pwm_dists_sq += self.set[i].distance_function(&proposal.set[i]).0.powi(2); 
            height_dists_sq += (proposal.set[i].peak_height-self.set[i].peak_height).powi(2);
        }
        
        [rmse, likelihood_diff, height_dists_sq.sqrt(), pwm_dists_sq.sqrt()]
    }



}


pub enum MemeParseError {
    EmptyMatrix,
    InvalidAlphabet{ motif_num: usize},
    InvalidMotifLength{ motif_num: usize, captured_length: usize },
    ColumnLengthFailure{ line_num: usize },
    FloatParseFailure{ line_num: usize },
}

impl fmt::Display for MemeParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MemeParseError::EmptyMatrix =>  write!(f, "This file has no MEME matrix input!"),
            MemeParseError::InvalidAlphabet{ motif_num } => write!(f, "The MEME matrix in position {motif_num} has a number of bps which does not match the proper {BASE_L}"),
            MemeParseError::InvalidMotifLength{ motif_num, captured_length } => write!(f, "The MEME matrix in position {motif_num} is of length {captured_length}, which is not in the range [{MIN_BASE}, {MAX_BASE}]"),
            MemeParseError::ColumnLengthFailure{ line_num } => write!(f, "The items on line {line_num} are fewer than the claimed number of columns from the base pair alphabet!"),
            MemeParseError::FloatParseFailure{ line_num } => write!(f, "The items on line {line_num} do not parse to floats, even though they should be part of a matrix!"),
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrippedMotifSet {
    set: Vec<Motif>,
    null_seq_attentions: Vec<usize>,
    ln_post: f64,
}

impl StrippedMotifSet {

    //Yes, self, not &self. I'm destroying the stripped motif set whenever I make the active motif set
    pub fn reactivate_set<'a>(&self, data_ref: &'a AllDataUse) -> MotifSet<'a> {

        let mut revived = MotifSet {

            set: self.set.clone(),
            signal: data_ref.data().derive_zero(),
            ln_post: None,
            null_seq_attentions: Vec::new(),
            null_peak_scores: Vec::new(),
            data_ref: data_ref,
        };


        revived.recalc_signal();

        let _ = revived.change_set_attention(self.null_seq_attentions.clone()); //I don't believe these ln posteriors if I'm going to be using them for inference. 
                                                                                //I only have this struct save a ln posterior because I want to record it for post processing

        revived
    }

    pub fn num_motifs(&self) -> usize { self.set.len() }

    pub fn motif_num_prior(&self) -> f64 {
        -((self.set.len()-1) as f64)* unsafe { NECESSARY_MOTIF_IMPROVEMENT }
    }

    pub fn ln_prior(&self, seq: &Sequence) -> f64 {
        self.motif_num_prior() + self.set.iter().map(|a| a.height_prior()+a.pwm_prior(seq)).sum::<f64>()
    }

    pub fn ln_posterior(&self) -> f64 { self.ln_post }

    pub fn set_iter(&self) -> Iter<'_, Motif> {
        self.set.iter()
    }
    //If you use this, the ln posterior that we save is no longer valid
    pub fn mutable_set_iter(&mut self) -> IterMut<'_, Motif> {
        self.set.iter_mut()
    }

    pub fn current_attention(&self) -> &Vec<usize> { &self.null_seq_attentions }

    pub fn output_to_meme(&self, output_dir: &str, run_name: &str) -> std::io::Result<String> {

        let savestate_file: String = output_dir.to_owned()+"/"+run_name+"_savestate.meme";

        let mut outfile_handle = fs::File::create(&savestate_file)?;

        outfile_handle.write(b"MEME Version 4\n\n")?;

        let alphabet = "ALPHABET= ".to_owned()+&BPS.iter().collect::<String>()+"\n";

        outfile_handle.write(&alphabet.into_bytes())?;

        for (i, mot) in self.set_iter().enumerate() {
            
            let header_str = format!("MOTIF motif_{i} {i}\n");
            outfile_handle.write(&header_str.into_bytes())?;

            let body_str = mot.for_meme();
            outfile_handle.write(&body_str.into_bytes())?;
        }

        Ok(savestate_file)
    }

    pub fn generate_pr_curve(&self, data_ref: &AllDataUse, fasta_file: &str, output_dir: &str, run_name: &str) -> Result<(), Box<dyn Error>> {

        println!("start pr");
        let mut peak_locations = data_ref.basic_peak(5.0);
        
        peak_locations.sort_by(|((a, _), _), ((b, _), _)| a.cmp(b));

        let meme_file = self.output_to_meme(output_dir, run_name)?;
       
        let fimo_output = format!("{}/{}_fimo", output_dir, run_name);

        let _ = fs::create_dir(&fimo_output);

        println!("created dir");
        Command::new("fimo").arg("--oc").arg(&fimo_output).arg("--bfile").arg("--uniform--").arg(&meme_file).arg(fasta_file).output()?;

        let fimo_tsv = format!("{}/fimo.tsv", fimo_output);

        let tsv_file_handle = std::fs::File::open(&fimo_tsv)?;

        println!("tsv read");

        let mut reader = csv::ReaderBuilder::new().has_headers(true)
                                           .delimiter(b'\t')
                                           .comment(Some(b'#'))
                                           .from_reader(tsv_file_handle);

        type Record = (String, String, String, usize, usize, char, f64, f64, f64, String);

        let mut number_hits: f64 = 0.0;
        let mut number_correct_hits: f64 = 0.0;

        let mut precision: Vec<f64> = Vec::with_capacity(peak_locations.len());
        let mut recall: Vec<f64> = Vec::with_capacity(peak_locations.len());
        let mut hit_peaks: HashSet<usize> = HashSet::new();

        let mut number_hit_peaks: f64 = 0.0;
        let number_peaks = peak_locations.len() as f64;

        for line in reader.deserialize() {
            let record: Record = line?;
            number_hits+=1.0;
            let start = record.3;
            let end = record.4;
            for (i, ((start_check, end_check), _)) in peak_locations.iter().enumerate() {
                println!("Record start and end {} {} locs start and end {} {}", start, end, start_check, end_check);
                if *start_check > end {break;}
                if (start >= *start_check) && (end <= *end_check) {
                    number_correct_hits += 1.0;
                    precision.push(number_correct_hits/number_hits);
                    if hit_peaks.insert(i) { //This returns true if this is a new insertion
                        number_hit_peaks += 1.0;
                    }
                    recall.push(number_hit_peaks/number_peaks);
                    println!("Success {} {}", start, end);
                    break;
                }
                println!("Start {} End {}", start_check, end_check);
            }
            println!("Fail {} {}", start, end);
        }

        for i in 0..peak_locations.len() {
            if hit_peaks.contains(&i) {
                println!("Hit peak {:?}", peak_locations[i].0);
            } else {
                println!("Miss peak {:?}", peak_locations[i].0);
            }
        }
    
        println!("Precision {:?} Recall {:?}", precision, recall);
        
        let file = format!("{}/{}{}", output_dir, run_name,"_pr_curve.png");

        let area = BitMapBackend::new(&file, (4000, 4200)).into_drawing_area();

        area.fill(&full_palette::WHITE).unwrap();

        let mut pr = ChartBuilder::on(&area);

        pr.margin(10).y_label_area_size(200).x_label_area_size(200);

        pr.caption("Precision vs recall", ("Times New Roman", 80));



        
        let mut pr_context = pr.build_cartesian_2d(0_f64..1_f64, 0_f64..1_f64).unwrap();

        pr_context.configure_mesh().x_label_style(("sans-serif", 70))
            .y_label_style(("sans-serif", 70))
            .x_label_formatter(&|v| format!("{:.0}", v))
            .axis_desc_style(("sans-serif",90))
            .x_desc("Recall")
            .y_desc("Precision").disable_x_mesh().disable_y_mesh().x_label_formatter(&|x| format!("{:.04}", *x)).draw().unwrap();

        pr_context.draw_series(LineSeries::new(precision.iter().zip(recall.iter()).map(|(&k, &i)| (i, k)), BLUE.filled().stroke_width(10))).unwrap().label("Total Motif Set");


        Ok(())
    }

    //The ith position of the vector is:
    //  Some(j), if the ith set in the reference set was matched to the jth set in self
    //  None, if the ith set in the reference set failed to match
    //The vector has length equal to reference_set.num_motifs()
    //Notice that the lifetime of the potential &Motif also matches the lifetime of &self. This is also intentional
    pub fn pair_to_other_set(&self, reference_set: &StrippedMotifSet) -> Vec<Option<(f64, bool, &Motif)>>/*Some type here */ {
    
        let distance_by_index = |(i, j): (usize, usize)| OrderedFloat(self.set[j].distance_function(&reference_set.set[i]).0);
        let rev_comp_by_index = |(i, j): (usize, usize)| self.set[j].distance_function(&reference_set.set[i]).1;

        //We want the self to act like the columns, because the positions of the reference set should be pulled
        let check_matrix = matrix::Matrix::from_fn(reference_set.num_motifs(), self.num_motifs(), distance_by_index);
        let rev_c_matrix = matrix::Matrix::from_fn(reference_set.num_motifs(), self.num_motifs(), rev_comp_by_index);

        if self.num_motifs() >= reference_set.num_motifs() {
            let (_, matches) = kuhn_munkres::kuhn_munkres_min(&check_matrix);
            matches.into_iter().enumerate().map(|(i,x)| Some((check_matrix.get((i,x)).unwrap().into_inner(), *rev_c_matrix.get((i,x)).unwrap(),&self.set[x]))).collect()
        } else {
            let transpose = check_matrix.transposed();
            let (_, pre_matches) = kuhn_munkres::kuhn_munkres_min(&transpose);
            let mut matches: Vec<Option<(f64, bool,&Motif)>> = vec![None; reference_set.num_motifs()];
            for (i, j) in pre_matches.into_iter().enumerate() {
                matches[j] = Some((check_matrix.get((j, i)).unwrap().into_inner(),*rev_c_matrix.get((j,i)).unwrap(), &self.set[i]));
            }
            matches
        }
    }

    pub fn get_nth_motif(&self, index: usize) -> &Motif {
        &self.set[index]
    }
    
    pub fn save_this_trace(&self, data_ref: &AllDataUse, output_dir: &str, file_name: &str) {

        let current_active = &self.reactivate_set(data_ref);

        current_active.save_set_trace_and_sub_traces(data_ref, output_dir, file_name);


    }

}

/*
enum AnyMotifSet<'a> {
    Active(MotifSet<'a>),
    Passive(StrippedMotifSet),
}

impl<'a> AnyMotifSet<'a> {

    fn give_activated(&self, data_ref: &'a AllDataUse) -> MotifSet<'a> {
        match self {
            AnyMotifSet::Passive(set) => set.reactivate_set(data_ref),
            AnyMotifSet::Active(ready) => ready.clone(),
        }
    }

    fn give_stored(&self) -> StrippedMotifSet {

        match self {
            AnyMotifSet::Passive(set) => set.clone(),
            AnyMotifSet::Active(ready) => StrippedMotifSet::from(ready),
        }

    }

    fn activate(&mut self, data_ref: &'a AllDataUse){
        *self = AnyMotifSet::Active(self.give_activated(data_ref));                                
    }

    fn store(&mut self) {

        if let AnyMotifSet::Active(set) = self {
            *self = AnyMotifSet::Passive(StrippedMotifSet::from(&*set));
        }


    }

}*/

impl<'a> From<&'a MotifSet<'a>> for StrippedMotifSet {
    fn from(other: &'a MotifSet) -> StrippedMotifSet {
        let ln_post: f64 = other.calc_ln_post();
        StrippedMotifSet {
            set: other.set.clone(),
            null_seq_attentions: other.null_seq_attentions.clone(),
            ln_post: ln_post,
        }
    }
}
/*
#[derive(Serialize, Deserialize, Clone)]
pub struct MotifSetDef {

    set: Vec<Motif>,
    signal: WaveformDef,
    ln_post: f64,

}

impl<'a> From<&'a MotifSet<'a>> for MotifSetDef {
    fn from(other: &'a MotifSet) -> Self {

        let signal = WaveformDef::from(&other.signal);

        Self {
            set: other.set.clone(), 
            signal: signal,
            ln_post: other.calc_ln_post(),
        }
    }
}

impl MotifSetDef {

    //SOUNDNESS: if data has the same length and spacer as self.signal, the kernels all match what should be defined by width,
    //            and the kernel heights are all permitted by the bounds of the height prior, then we skip a lot of recalculation
    //            to generate the correct signal: we assume that the signal given IS correct for the waveform. 
    //            If recalculate is set to true, then we do not skip. You MUST set recalculate to true if the motif set was not 
    //            generated by these data. 
    pub fn get_motif_set<'a, R: Rng+?Sized >(self, recalculate: bool, validate_motifs: bool, randomizer: &mut Option<&mut R>, data_ref: &'a AllDataUse ) -> MotifSet<'a> {

        let mut set = self.set.clone();

        //We fix up any kernels that don't match the width defined in our MotifSet
        let mut scrambled_motifs = false;
        for mot in set.iter_mut() {

            if validate_motifs && mot.pwm_prior(data_ref.data().seq()) == f64::NEG_INFINITY {
                scrambled_motifs = true;
                _ = mot.scramble_to_close_random_valid(data_ref.data().seq(), randomizer);
            }

        }

        //We check to see if we need to recalculate the wave signal.
        let signal_will_make_ub =  (self.signal.len() != data_ref.data().read_wave().len()) || (self.signal.spacer() != data_ref.data().spacer());
        let redo_signal = recalculate || scrambled_motifs || signal_will_make_ub;

        let signal = if !redo_signal { //Notice how we carefully checked to make sure that signal follows our safety guarentees
            self.signal.get_waveform(data_ref.data().seq())
        } else {
            data_ref.data().derive_zero() //This isn't our final signal. This just ensures our safety guarentees
        };

        let mut pre_set = MotifSet {
            set: set,
            signal: signal,
            ln_post: None,
            null_seq_attentions: Vec::new(),
            null_peak_scores: Vec::new(), 
            data_ref: data_ref, 
        };

        if !redo_signal {
            pre_set.ln_post = Some(self.ln_post);
        } else {
            if !recalculate {
                warn!("Had to adjust your motif set and recalculate its occupancy trace because:");
                if scrambled_motifs { warn!("-Some of your motifs' best binding kmers were unrepresented in your sequence, and you wanted them validated. They've been scrambled to have a best kmer that is compatible with your sequence while remaining as close to the unrepresented kmers as possible.");}
                if signal_will_make_ub { warn!("The occupancy trace you reported for your motif set is incompatible with your ChIP data. We calculated the signal to prevent segfaults.");}
            }

            pre_set.recalc_signal();
            _ = pre_set.ln_posterior();
        }

        pre_set


    }

}
*/


//Why this weird lifetime annotation?
//If this is Set, the associated lifetime of Set matters.
//But, if this is Rng, I can generate the random variable and Rng's life is irrelevant
pub enum InitializeSet<'a,'b, R: Rng + ?Sized>{
    Set(MotifSet<'a>),
    Rng(&'b mut R),
}


//THIS SHOULD BE (DE)SERIALIZABLE WITH A CUSTOM IMPLEMENTATION
//SINCE ALL MOTIFS IN THE SET AND THE WAVEFORM SHOULD POINT TO seq
//AND THE MOTIF_SET ITSELF SHOULD ALSO POINT TO data AND background
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
    pub fn new_trace<R: Rng + ?Sized>(capacity: usize, initial_condition: Option<MotifSet<'a>>, all_data_file: String,data_ref: &'a AllDataUse<'a>, mut thermo_beta: f64, null_attention: Vec<usize>, sparse: Option<usize>, rng: &mut R) -> SetTrace<'a> {

        thermo_beta = thermo_beta.abs();

        let mut active_set = match initial_condition {
            Some(set) => set,
            None => MotifSet::rand_with_one(data_ref, rng),
        };

        active_set.change_set_attention(null_attention);

        SetTrace{
            trace: Vec::<StrippedMotifSet>::with_capacity(capacity),
            active_set: active_set,
            all_data_file: all_data_file,
            data_ref: data_ref, 
            sparse: sparse.unwrap_or(10),
            sparse_count: 0_usize,
            thermo_beta: thermo_beta,
        }

    }

    pub fn thermo_beta(&self) -> f64 { self.thermo_beta }

    pub fn advance<R: Rng + ?Sized>(&mut self, track: Option<&mut MoveTracker>, burning_in: bool, rng: &mut R) { 

        let which_move = PICK_MOVE.sample(rng);

        let which_variant = SAMPLE_VARIANTS[which_move].sample(rng); 

        //Can't use non const values in match statements. Using a bajillion if-elses 
        
        let potential_set_and_accept = match which_move {
        
            0 =>  {
                let [ratio_s, liney_s] = RATIO_LINEAR_SD_COMBO[which_variant];
                self.active_set.propose_ordered_base_move_custom(rng, ratio_s, liney_s).map(|(new_mot, modded_ln_like)| { 
                    let accepted = MotifSet::accept_test(self.active_set.ln_posterior(), modded_ln_like, self.thermo_beta, rng); 
                    (new_mot, accepted)
                })
            }, 
            1 =>  {
                let [ratio_s, liney_s] = RATIO_LINEAR_SD_COMBO[which_variant];
                self.active_set.propose_ordered_motif_move_custom(rng, ratio_s, liney_s).map(|(new_mot, modded_ln_like)| { 
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
            3 => self.active_set.run_rj_move_known(which_variant, self.thermo_beta, rng),
            4 => Some((self.active_set.base_leap(self.thermo_beta, rng), true)),
            5 => Some((self.active_set.secondary_shuffle(self.thermo_beta, rng), true)),
            6 => {
                let scale_s = SCALE_SDS[which_variant];
                self.active_set.propose_scale_base_custom(rng, scale_s).map(|(new_mot, modded_ln_like)| { 
                    let accepted = MotifSet::accept_test(self.active_set.ln_posterior(), modded_ln_like, self.thermo_beta, rng); 
                    (new_mot, accepted)
                })
            }, 

            7 => {
                let scale_s = SCALE_SDS[which_variant];
                self.active_set.propose_ordered_scale_move_custom(rng, scale_s).map(|(new_mot, modded_ln_like)| {
                    let accepted = MotifSet::accept_test(self.active_set.ln_posterior(), modded_ln_like, self.thermo_beta, rng);
                    (new_mot, accepted)
                })
            },
            _ => unreachable!(),

        };
 

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
        
        if let Some(tracker) = track {
            let total_id = VARIANT_CUMULATIVE[which_move]+which_variant;
            tracker.document_motion(total_id, movement_tracker).expect("We only track the proper moves");
        };


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


        //NOTE: we do NOT include this in the actual acceptance steps on purpose.
        //      The motif finder must randomly choose which null sequence it's going to pay attention to for sampling
        //set_to_add.change_set_attention(self.data_ref.null_seq().yield_random_block_set(rng).clone());

        self.active_set = set_to_add;


    }

    pub fn serially_temper<R: Rng+?Sized>(&mut self, alter_beta: f64, track: &mut [MoveTracker; 3], steps_to_track: usize, steps_to_equilibrate: usize, rng: &mut R) {

        self.thermo_beta = 1.0;

        //This guarentees that our alter_beta is nonnegative and less than 1.0
        //Note that, at least on x86, this compiles to basically two pieces of bitmath, as it should
        let alter_beta_b = (-alter_beta.abs().ln().abs()).exp();

        for _ in 0..steps_to_track {
            self.advance(Some(&mut track[0]), false, rng);
        }

        let cooldown_target = self.active_set.ln_posterior();

        println!("Beta = 1.0 when we're sampling: final Ln posterior {}", cooldown_target);


        self.thermo_beta = alter_beta_b;

        for _ in 0..(2*steps_to_equilibrate) {
            self.advance(Some(&mut track[1]), true, rng);
        }

        println!("Beta = {alter_beta} when we're tempering. Final Ln posterior {}", self.active_set.ln_posterior());

        self.thermo_beta = 10000.0;

        for _ in 0..steps_to_equilibrate{
            self.advance(Some(&mut track[1]), true, rng);
        }

        println!("Beta = 10000.0. Final Ln posterior {}", self.active_set.ln_posterior());

        self.thermo_beta = 1.0;

        let mut check_like = -f64::INFINITY;

        let mut number_iter: usize = 0;

        for _ in 0..steps_to_equilibrate {
        //while check_like < cooldown_target {
            self.advance(Some(&mut track[2]), true, rng);
            number_iter += 1;
            check_like = self.active_set.ln_posterior();
        //}
        }

        println!("Beta = 1.0 when we're equilibrating. Iters to cool {number_iter}.  Final Ln posterior {}", self.active_set.ln_posterior());



    }

    pub fn advance_only_shuffles<R: Rng + ?Sized>(&mut self, track: Option<&mut MoveTracker>, do_base_leap: bool, burning_in: bool, rng: &mut R) {

        let mut set = match do_base_leap {
            true => self.active_set.base_leap(self.thermo_beta, rng),
            false => self.active_set.secondary_shuffle(self.thermo_beta, rng),
        };

        if let Some(tracker) = track {
            let tracked = self.active_set.measure_movement(&mut set);
            let total_id = if do_base_leap { VARIANT_CUMULATIVE[4] } else { VARIANT_CUMULATIVE[5] };
            tracker.document_motion(total_id, Some((tracked, true))).expect("We only track the proper moves");
        }

        if !burning_in{
            self.sparse_count = (self.sparse_count+1) % self.sparse;

            if self.sparse_count == 0 {

                //This order INCLUDES the last possible motif set
                //and specifically EXCLUDES the initial state
                let (full_set, _) = set.set_with_new_attention((0..self.data_ref.null_seq().num_sequence_blocks()).collect::<Vec<usize>>());
                self.trace.push(StrippedMotifSet::from(&full_set));
            }
        }

        self.active_set = set;

    }

    //WARNING: Be aware that this motif set WILL be coerced to follow the seq and data of the motif set
    //         The ln posterior will also be recalculated if the motif set isn't correctly pointed
    //         To ensure the safety of this function, there is a recalculation step that occurs if the 
    //         sequence or data changes. I assure you, you do not want this recalculation to occur:
    //         It's going to be very slow
    pub fn push_set_to_active(&mut self, set: MotifSet<'a>) {


        let mut repoint_set = set.clone();

        let recalc_ln_post = !std::ptr::eq(self.data_ref, repoint_set.data_ref);

        if recalc_ln_post {
            repoint_set.data_ref = &self.data_ref;
        }

        if recalc_ln_post {
            repoint_set.signal = self.data_ref.data().derive_zero();
            for i in 0..repoint_set.set.len() {
                repoint_set.signal += &repoint_set.set[i].generate_waveform(&self.data_ref);
            }
            repoint_set.ln_post = None;
            repoint_set.ln_posterior();
        }



        self.active_set = repoint_set;


    }



    //Note: if the likelihoods are calculated off of a different sequence/data, this WILL 
    //      just give you a wrong answer that seems to work
    /*pub fn push_set_def<R: Rng+?Sized>(&mut self, always_recalculate: bool,validate_motif: bool, set: MotifSetDef, rng: &mut R) {
        
        let mut validate_randomizer = if validate_motif { Some(rng) } else {None};
        let setter = set.get_motif_set(always_recalculate, validate_motif, &mut validate_randomizer,&self.data_ref);
        self.push_set_to_active(setter);
    }*/


    /* pub fn try_replace_or_create_last<R: Rng + ?Sized>(&mut self, set: StrippedMotifSet) -> Result<(), String>{

       let active_set = set.reactivate_set(self.data_ref);

    //SAFETY: reactivate_set ALWAYS initializes the posterior
    let Some(posterior) = active_set.ln_post else { unsafe {unreachable_unchecked()} };

    if !(posterior > -f64::INFINITY) {
    return Err("This motif set is impossible to have, and thus we reject it. The motif trace that called this is unchanged.".to_string());
    }

    if self.trace.len() == 0 {
    self.trace.push(AnyMotifSet::Active(active_set));
    } else {

    //SAFETY: we only hit this branch if the trace has a last position with an element
    let Some(last_pos) = self.trace.last_mut() else { unsafe {unreachable_unchecked()} };
     *last_pos = AnyMotifSet::Active(active_set);
     }

     Ok(())
     }*/


    /*pub fn current_set(&self) -> MotifSet<'a> {
      self.trace[self.trace.len()-1].give_activated(self.data_ref)
      }

      pub fn current_set_mut_ptr(&mut self) -> &mut MotifSet<'a> {

      self.trace[self.trace.len()-1].activate(self.data_ref);
      &mut self.trace[self.trace.len()-1]
      }


      pub fn quiet_last_set(&mut self) {

      if let Some(set_mut_ref) = self.trace.last_mut() {
      set_mut_ref.store();
      }

      }*/

    pub fn loan_active(&self) -> &MotifSet<'a> {
        &self.active_set
    }

    pub fn current_set_to_print(&self) -> StrippedMotifSet {
        StrippedMotifSet::from(&self.active_set)
    }
    pub fn save_initial_state(&self, output_dir: &str, run_name: &str) {

        let savestate_file: String = output_dir.to_owned()+"/"+run_name+"_savestate.bin";

        let mut outfile_handle = fs::File::create(savestate_file).expect("Output directory must be valid!");

        let mot_to_save: StrippedMotifSet = match self.trace.get(0) {
            Some(set) => set.clone(), 
            None => self.current_set_to_print(),
        };

        let buffer: Vec<u8> = bincode::serialize( &mot_to_save).expect("serializable");

        outfile_handle.write(&buffer).expect("buffer should write");
    }

    pub fn save_final_state(&self, output_dir: &str, run_name: &str) {

        let savestate_file: String = output_dir.to_owned()+"/"+run_name+"_savestate.bin";

        if self.trace.last().is_none() {warn!("trace is empty!"); return;}


        let last = self.trace.last().unwrap();

        let mut outfile_handle = fs::File::create(savestate_file).expect("Output directory must be valid!");

        let buffer: Vec<u8> = bincode::serialize( &last).expect("serializable");

        outfile_handle.write(&buffer).expect("buffer should write");
    }


    pub fn save_current_bedgraph(&self, output_dir: &str, run_name: &str, zeroth_step: usize) -> Result<(), Box<dyn Error>> { 

        //generate_bedgraph(&self, data_seq: &AllDataUse, chromosome_name: Option<&str>, file_name: &str)

        self.loan_active().set.iter().enumerate().map(|(i,mot)| {
            let bedgraph_file:  String = format!("{}/{}_bedgraph_{:0>7}_motif_{i}.bed",output_dir,run_name,zeroth_step);
            mot.generate_bedgraph(self.data_ref, None, &bedgraph_file)
        }).collect::<Result<_,_>>()
    }

    //We don't need to worry about saving any state in trace, since inference works on the ACTIVE state
    fn drop_history(&mut self) {
        self.trace.clear();
    }

    pub fn save_and_drop_history(&mut self, output_dir: &str, run_name: &str, zeroth_step: usize) {

        let trace_file: String = format!("{}/{}_trace_from_step_{:0>7}.bin",output_dir,run_name,zeroth_step);
        let mut outfile_handle = fs::File::create(trace_file).expect("Need to give a valid file to write to or inference is pointless");

        let len_trace = self.trace.len();

        //We don't need to worry about saving any state in trace, since inference works on the ACTIVE state
        let trace: Vec<StrippedMotifSet> = self.trace.drain(0..len_trace).collect();


        let buffer: Vec<u8> = bincode::serialize( &(SetTraceDef {
            trace: trace,
            all_data_file: self.all_data_file.clone(),
            thermo_beta: self.thermo_beta,
        })).expect("serializable");


        //fs::write(trace_file.as_str(), serde_json::to_string(&history).unwrap()).expect("Need to give a valid file to write to or inference is pointless");

        outfile_handle.write(&buffer).expect("buffer should write");

        self.save_initial_state(output_dir, run_name);
    }

    pub fn save_trace(&self, output_dir: &str, run_name: &str, zeroth_step: usize) {

        let current_active = &self.active_set;

        let file_name = format!("{}/{:0>7}", run_name,zeroth_step);

        current_active.save_set_trace_and_sub_traces(self.data_ref, output_dir, &file_name);

    }

    pub fn active_set_motif_lens(&self) -> Vec<usize> {
        self.active_set.set.iter().map(|a| a.len()).collect::<Vec<_>>()
    }

    pub fn active_set_peak_heights(&self) -> Vec<f64> {
        self.active_set.set.iter().map(|a| a.peak_height()).collect::<Vec<_>>()
    }



}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SetTraceDef {

    trace: Vec<StrippedMotifSet>,
    all_data_file: String,
    thermo_beta: f64,

}




impl SetTraceDef {

    //Panics: all MotifSetDefs in the trace must be of the same dimension and spacing as the data.
    //        We also check to make sure that the first set's posterior density is accurate
    //Correctness: If you supply a data file where the posterior density of a set does not match what its posterior density should be,  
    //             NOTHING can help you here. Now, there are no SAFETY invariants that will break things here, but consider this a 
    //             a correctness invariant on the posterior trace. It would take WAY too much work to validate, here, so we only rely
    //             on the ACTIVE motif having a correct likelihood for this reason
    //NOTE: When we save SetTraceDef, we do not keep information about rng. So "reinitializing" might use a different RNG altogether
    //      I think trying to do this another way is pointless record keeping and doesn't actually matter, so long as your RNG is good enough
    //      Basically, don't rely on RNG objects being consistent from run to run for any invariants
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
            sparse: sparse.unwrap_or(10),
            sparse_count: 0,
            thermo_beta: self.thermo_beta,
        }


    }

    pub fn reduce(&mut self, max_elements: usize) {
        if self.trace.len() >= max_elements {
            let _ = self.trace.drain(max_elements..);
        }
    }
    pub fn len(&self) -> usize {
        self.trace.len()
    }

    pub fn ln_posterior_trace(&self) -> Vec<f64> {
        self.trace.iter().map(|a| a.ln_post).collect::<Vec<f64>>()
    }

    pub fn ln_likelihood_trace(&self, full_data: &AllData) -> Vec<f64> {

        self.trace.iter().map(|a| a.ln_post-a.ln_prior(full_data.seq())).collect::<Vec<f64>>()

    }

    pub fn motif_num_trace(&self) -> Vec<f64> {
        self.trace.par_iter().map(|a| a.set.len() as f64).collect::<Vec<_>>()
    }


    pub fn wave_dist_trace(&self, waypost: &AllDataUse) -> Vec<f64> {
        self.trace.par_iter().map(|a| a.reactivate_set(waypost).median_data_dist()).collect()
    }

    pub fn data_name(&self) -> &str {
        &self.all_data_file
    }

    pub fn initial_set_pwm(&self) -> Motif {
        self.trace[0].set[0].clone()
    }

    //PWMs are chosen by making a random choice of SET, and a random choice of ONE motif per set
    pub fn ret_rand_motifs<R: Rng + ?Sized>(&self, num_motifs: usize, rng: &mut R) -> Vec<Motif> {

        let spacing = self.trace.len()/num_motifs;

        let picks = (0..(self.trace.len()/spacing)).map(|a| a*spacing); 

        let set_picks: Vec<&StrippedMotifSet> = picks.map(|a| &(self.trace[a])).collect();

        let pwms: Vec<Motif> = set_picks.iter().map(|a| a.set.choose(rng).expect("No motif set should be empty").clone()).collect();

        pwms
    }

    pub fn trace_min_dist(&self, reference_motif: &Motif) -> Vec<f64> {

        self.trace.iter().map(|mot_set| {

            mot_set.set.iter().map(|mot| mot.distance_function(&reference_motif).0)
                .min_by(|a,b| a.partial_cmp(b).expect("No NANs should be present in distances"))
                .expect("Motif sets all need at least one motif")

        }).collect::<Vec<f64>>()

    }

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

    pub fn save_best_trace(&self, buffer: &mut Vec<u8>, output_dir: &str, run_name: &str) {

        let mut try_bincode = fs::File::open(self.all_data_file.as_str()).expect("a trace should always refer to a valid data file");
        let _ = try_bincode.read_to_end(buffer);

        let data_reconstructed: AllData = bincode::deserialize(&buffer).expect("Monte Carlo chain should always point to data in proper format for inference!");

        let data_ref = AllDataUse::new(&data_reconstructed).expect("AllData file must be valid!");
        let index_best = self.trace.iter().map(|mot_set| mot_set.ln_post).enumerate().fold((0, -f64::INFINITY), |a, b| std::cmp::max_by(a, b, |x,y| x.1.partial_cmp(&y.1).expect("No NaNs in valid traces"))).0;

        let current_active = self.trace[index_best].clone().reactivate_set(&data_ref);

        let locs = data_ref.data().generate_all_locs();

        let signal_file: String = format!("{}/{}_occupancy_signal_highest_posterior_density.png",output_dir,run_name);

        let plot = BitMapBackend::new(&signal_file, (3000, 1000)).into_drawing_area();

        plot.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&plot)
            .set_label_area_size(LabelAreaPosition::Left, 40)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .caption("Signal Comparison", ("sans-serif", 40))
            .build_cartesian_2d(0_f64..(*locs.last().unwrap() as f64), (-16_f64)..16_f64).unwrap();


        chart.configure_mesh().draw().unwrap();

        chart.draw_series(current_active.data_ref.data().read_wave().iter().zip(locs.iter()).map(|(&k, &i)| Circle::new((i as f64, k),1_u32, &BLACK))).unwrap().label("Data Wave");

        let current_resid = current_active.data_ref.data()-&current_active.signal;
        chart.draw_series(LineSeries::new(current_resid.read_wave().iter().zip(locs.iter()).map(|(&k, &i)| (i as f64, k)), &BLUE)).unwrap().label("Occpuancy Trace");

        let max_draw = current_active.set.len().min(3);

        let colors = [&RED, &GREEN, &CYAN];

        for i in 0..max_draw {

            let occupancy = current_active.set[i].generate_waveform(current_active.data_ref);
            chart.draw_series(LineSeries::new(occupancy.read_wave().iter().zip(locs.iter()).map(|(&k, &i)| (i as f64, k)), colors[i])).unwrap().label(format!("Motif {}", i).as_str());

        }

        chart.configure_series_labels().border_style(&BLACK).draw().unwrap();

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


/*

   pub struct SetTrace<'a, R: Rng > {
    trace: Vec<StrippedMotifSet>,
    active_set: MotifSet<'a>,
    data_ref: &'a AllDataUse<'a>,
    sparse: usize,
    sparse_count: usize,
    thermo_beta: f64,
    rng: R,
}



   */

//base_ratio_sds: &[f64], base_linear_sds: &[f64], height_move_sds: &[f64],
//                                    attempts_per_move: &mut [usize], successes_per_move: &mut [usize], immediate_failures_per_move: &mut [usize],
//                                    distances_per_attempted_move: &mut [Vec<([f64; 4], bool)>]

#[derive(Clone)]
pub struct MoveTracker {

    attempts_per_move: [usize; NUM_MOVES],
    successes_per_move: [usize; NUM_MOVES], 
    immediate_failures_per_move: [usize; NUM_MOVES], 
    distances_per_attempted_move: [Vec<([f64; 4], bool)>; NUM_MOVES], 

}

impl MoveTracker {

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


    pub fn give_status(&self) {
        let mut ind: usize = 0;
            for i in 0..BASE_RATIO_SDS.len(){
                for j in 0..BASE_LINEAR_SDS.len() {
                    println!("Single base move with ratio sd {} and linear sd {}. Attempts: {}. Successes {}. Immediate failures {}. Rate of success {}. Rate of immediate failures {}.",
                             BASE_RATIO_SDS[i], BASE_LINEAR_SDS[j], self.attempts_per_move[ind], self.successes_per_move[ind], self.immediate_failures_per_move[ind],
                             (self.successes_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64), (self.immediate_failures_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64));
                    ind += 1;
                }
            }
            for i in 0..BASE_RATIO_SDS.len(){
                for j in 0..BASE_LINEAR_SDS.len() {
                    println!("Motif bases move with ratio sd {} and linear sd {}. Attempts: {}. Successes {}. Immediate failures {}. Rate of success {}. Rate of immediate failures {}.",
                             BASE_RATIO_SDS[i], BASE_LINEAR_SDS[j], self.attempts_per_move[ind], self.successes_per_move[ind], self.immediate_failures_per_move[ind],
                             (self.successes_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64), (self.immediate_failures_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64));
                    ind += 1;
                }
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
            println!("Base leap move (always accepts). Times {}. Last distance is {}", self.attempts_per_move[ind], match self.distances_per_attempted_move[ind].last() { Some(dist) => format!("{:?}", dist.0), None => "None tried".to_owned() });
            ind += 1;
            println!("Secondary shuffle move (always accepts). Times {}. Last distance is {}", self.attempts_per_move[ind], match self.distances_per_attempted_move[ind].last() { Some(dist) => format!("{:?}", dist.0), None => "None tried".to_owned() });
            for i in 0..SCALE_SDS.len() {
                println!("Base scale move with sd {}. Attempts: {}. Successes {}. Immediate failures {}. Rate of success {}. Rate of immediate failures {}.",
                         SCALE_SDS[i], self.attempts_per_move[ind], self.successes_per_move[ind], self.immediate_failures_per_move[ind],
                         (self.successes_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64), (self.immediate_failures_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64));
                ind += 1;
            }
            for i in 0..SCALE_SDS.len() {
                println!("Motif scale move with sd {}. Attempts: {}. Successes {}. Immediate failures {}. Rate of success {}. Rate of immediate failures {}.",
                         SCALE_SDS[i], self.attempts_per_move[ind], self.successes_per_move[ind], self.immediate_failures_per_move[ind],
                         (self.successes_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64), (self.immediate_failures_per_move[ind] as f64)/(self.attempts_per_move[ind] as f64));
                ind += 1;
            }
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

    pub fn all_move_hists(&self, base_file_name: &str, num_bins: usize) -> Result<(), Vec<String>> {
        
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

    hist_context.configure_mesh().x_label_style(("sans-serif", 70))
        .y_label_style(("sans-serif", 70))
        .x_label_formatter(&|v| format!("{:.0}", v))
        .axis_desc_style(("sans-serif",90))
        .x_desc("Ln posterior density proposed set/current set")
        .y_desc("Probability").disable_x_mesh().disable_y_mesh().x_label_formatter(&|x| format!("{:.04}", *x)).draw().unwrap();

    //hist_context.draw_series(Histogram::vertical(&hist_context).style(CYAN.filled()).data(trial_data.iter().map(|x| (x, inverse_size)))).unwrap();
    hist_context.draw_series(Histogram::vertical(&hist_context).style(CYAN.mix(0.8).filled()).margin(0).data(hist_form.iter().map(|x| (&x.0, x.1)))).unwrap().label("Proposed Moves").legend(|(x, y)| Rectangle::new([(x-20, y-10), (x, y+10)], Into::<ShapeStyle>::into(&CYAN).filled()));

    hist_context.draw_series(Histogram::vertical(&hist_context).style(RED.mix(0.8).filled()).margin(0).data(hist_form_2.iter().map(|x| (&x.0, x.1)))).unwrap().label("Accepted Moves").legend(|(x, y)| Rectangle::new([(x-20, y-10), (x, y+10)], Into::<ShapeStyle>::into(&RED).filled()));

    hist_context.configure_series_labels().position(SeriesLabelPosition::UpperLeft).margin(40).legend_area_size(20).border_style(&BLACK).label_font(("Calibri", 80)).draw().unwrap();

    hist


}


#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub enum TrackingOptions {
    NoTracking,
    TrackTrueTrace,
    TrackAllTraces,
}

pub struct TemperSetTraces<'a> {
    //Each trace knows its thermodynamic beta
    parallel_traces: Vec<(SetTrace<'a>, Option<MoveTracker>)>,
    track: TrackingOptions,
}

impl<'a> TemperSetTraces<'a> {

    //Remember, smaller thermodynamic beta means that chains are LAXER about acceptances
    //
    //Error: if min_thermo_beta < 0.0 OR > 1.0. Note that any thermo_beta == 0.0 means that a trace ALWAYS accepts ANY move, and 1.0 means that a trace is strictly following Metropolis Hastings
    //
    //Invariants:
    //      parallel_traces.len() >= 2
    //      parallel_traces[0] will always have thermo_beta = 1_f64. Because of this, it is the "canonical" trace, and the others "just" seed it periodically
    //      parallel_traces.last().unwrap() will never panic, and will always have thermo_beta = min_thermo_beta
    pub fn new_parallel_traces<R: Rng+?Sized>(min_thermo_beta: f64, num_intermediate_traces: usize, capacity_per_trace: usize, step_num_estimate: usize, how_to_track: TrackingOptions, all_data_file: String, data_ref: &'a AllDataUse<'a>, initial_condition: Option<MotifSet<'a>>, sparse: Option<usize>, rng: &mut R) -> Result<Self, InitializationError> {
        if let Some(a) = initial_condition.as_ref(){
            if !ptr::eq(a.data_ref, data_ref) { return Err(InitializationError::UnsynchedData); }
        }

        if min_thermo_beta < 0.0 { return Err(InitializationError::NegativeTemperature); }
        if min_thermo_beta > 1.0 { return Err(InitializationError::UselessBeta); }

        println!("begin");


        let num_null_blocks = data_ref.null_seq().num_sequence_blocks();

        let num_traces = num_intermediate_traces+2;

        /*let null_block_groups: Vec<Vec<usize>> = if (num_null_blocks >= (num_intermediate_traces+2)){
            let mut null_b_g = vec![Vec::with_capacity(1+num_null_blocks/(num_intermediate_traces+2));num_intermediate_traces+2];
            for i in 0..num_null_blocks { null_b_g[i % (num_intermediate_traces+2)].push(i); } 
            null_b_g
        } else {
            let number_splits = num_null_blocks*((num_intermediate_traces+2)/num_null_blocks);
            (0..number_splits).map(|i| vec![i % num_null_blocks]).collect()
        };*/

        //I always want my intermediate states to be even, because I want to be able to try to swap everybody at once
        //So I always want to randomly 
        let mut thermos = vec![1_f64; num_traces];

        //let null_block_groups: Vec<Vec<usize>> = (0..(num_intermediate_traces+2)).map(|_| data_ref.null_seq().yield_random_block_set(rng).clone()).collect();

        //let mut null_block_groups: Vec<Vec<usize>> = vec![Vec::with_capacity(1+num_null_blocks/(num_intermediate_traces+2));num_intermediate_traces+2];

        let null_block_groups: Vec<Vec<usize>> = vec![(0..num_null_blocks).collect();num_intermediate_traces+2];

        //null_block_groups[0] = (0..num_null_blocks).collect();

        //for i in 0..num_null_blocks { null_block_groups[1+(i % (num_intermediate_traces+1))].push(i); }

       // for i in 0..num_null_blocks { null_block_groups[i % (num_intermediate_traces+2)].push(i); }


        let thermo_step = (1_f64-min_thermo_beta)/((thermos.len()-1) as f64);

        *thermos.last_mut().expect("thermos always has elements") = min_thermo_beta;

        //TODO: check that this DOES produce a linear decrease in thermodynamic beta
        for i in 1..(num_traces-1) {
            thermos[i] = 1_f64-(i as f64)*thermo_step
        }


        let mut parallel_traces: Vec<(SetTrace, Option<MoveTracker>)> = Vec::with_capacity(thermos.len());

        //pub fn new_trace(capacity: usize, initial_condition: Option<MotifSet<'a>>, data_ref: &'a AllDataUse<'a>, mut thermo_beta: f64, sparse: Option<usize>, mut rng: R) -> SetTrace<'a>

        let past_initial: bool = false;

        for (i, thermo_beta) in thermos.into_iter().enumerate() {
            println!("thermo check {i} {:?}", null_block_groups[i]);
            let potential_tracker = match how_to_track {
                TrackingOptions::NoTracking => None, 
                TrackingOptions::TrackAllTraces => Some(MoveTracker::new(step_num_estimate)),
                TrackingOptions::TrackTrueTrace => if past_initial { None } else { Some(MoveTracker::new(step_num_estimate)) },
            };
            let set_trace = SetTrace::new_trace(capacity_per_trace, initial_condition.clone(), all_data_file.clone(),data_ref, thermo_beta, null_block_groups[i].clone(), sparse, rng); 
            println!("thermo {} attention {:?}", set_trace.thermo_beta, set_trace.active_set.current_attention());
            parallel_traces.push((set_trace, potential_tracker));
        }

        Ok(TemperSetTraces { parallel_traces: parallel_traces, track: how_to_track })
    }

    pub fn do_shuffles<R: Rng+?Sized>(&mut self, number_secondaries: usize, burning_in: bool, rng: &mut R) {

        self.parallel_traces.iter_mut().for_each(|(set, track)| {
            set.advance_only_shuffles(track.as_mut(), true, burning_in, rng);
            for _ in 0..number_secondaries { set.advance_only_shuffles(track.as_mut(), false, burning_in, rng); } 
        });

    }

    pub fn iter_and_swap<R: Rng>(&mut self, iters_between_shuffles: usize, iters_before_swaps: usize, cooldown_iters: usize, rng_maker: fn() -> R) {



        for _ in 0..iters_before_swaps {
       
            self.parallel_traces.par_iter_mut().for_each(|(set, track)| {
                let mut rng = rng_maker();
                for _ in 0..iters_between_shuffles { set.advance(track.as_mut(), false, &mut rng); }
            });
        }

        let data_seq = self.parallel_traces[0].0.data_ref;
        let list_of_nulls = (0..data_seq.null_seq().num_sequence_blocks()).collect::<Vec<_>>();


        
        for j in 0..self.parallel_traces.len(){
            for (i, motif) in self.parallel_traces[j].0.active_set.set.iter().enumerate(){
            let binds = motif.return_any_null_binds_in_group(data_seq.null_seq(), &list_of_nulls).into_iter().filter(|&b| b > ((self.parallel_traces[j].0.data_ref.background_ref().noise_spread_par() * 4.0)-motif.peak_height()).exp2()).collect::<Vec<f64>>();
            if binds.len() >0{
                //let binds_diffs = binds.windows(2).map(|a| a[0]-a[1]).collect::<Vec<_>>();
            //let binds_ratio = binds.windows(2).map(|a| a[0]/a[1]).collect::<Vec<_>>();
             println!("{j} {i} binds {} {} len {}", binds[0], binds.last().unwrap(), binds.len());
            //println!("{:?} \n {:?}", binds_diffs, binds_ratio);
            } else {println!("{j} {i} no extraneous");}
        }
        }

        //We only guarentee that there are two parallel traces, hence this check and the other one for odd_attention_swaps
        //Yes, we deliberately skip the first possible even swap: I want the beta = 1 chain to always pay attention to 
        //ALL of the negative sequence
        //if self.parallel_traces.len() >= 4  {
            //let even_attention_swaps: Vec<([f64;2], bool)> = self.parallel_traces[2..].par_chunks_exact_mut(2).map(|x| {//} 
            /*let even_attention_swaps: Vec<(_, bool)> = self.parallel_traces.par_chunks_exact_mut(2).map(|x| { 
                let (c, d) = x.split_at_mut(1);
                let (a, b) = (&mut c[0], &mut d[0]);
                let mut rng = rng_maker();
                let attention_0 = a.0.active_set.current_attention().clone();
                let attention_1 = b.0.active_set.current_attention().clone();
                let (mut swap_attention_a, swap_ln_post_a) = a.0.active_set.set_with_new_attention(attention_1.clone());
                let (mut swap_attention_b, swap_ln_post_b) = b.0.active_set.set_with_new_attention(attention_0.clone());

                let accept_swap = MotifSet::accept_test(a.0.thermo_beta*swap_ln_post_a+b.0.thermo_beta*swap_ln_post_b,
                                                        a.0.thermo_beta*a.0.active_set.ln_posterior()+b.0.thermo_beta*b.0.active_set.ln_posterior(), 1.0, &mut rng);

                if accept_swap {
                    std::mem::swap(&mut a.0.active_set, &mut swap_attention_a);
                    std::mem::swap(&mut b.0.active_set, &mut swap_attention_b);
                }

                //([a.0.thermo_beta, b.0.thermo_beta], accept_swap)
                ([attention_0, attention_1], accept_swap)
            }).collect();

            println!("Even attention swaps: {:?}", even_attention_swaps.iter().map(|a| format!("{:?} att {}", a.0, a.1)).collect::<Vec<_>>() );
        //}*/
                
        println!("Ln posteriors of each trace before swaps: {:?}", self.parallel_traces.iter_mut().map(|x| x.0.active_set.ln_posterior()).collect::<Vec<f64>>());

        /*

                let odd_swaps: Vec<([f64;2], bool)> = self.parallel_traces[1..].par_chunks_exact_mut(2).map(|x| {
            let (c, d) = x.split_at_mut(1);
            let (a, b) = (&mut c[0], &mut d[0]);
            let mut rng = rng_maker();
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

        println!("tf numbs of each treach after swaps: {:?}", self.parallel_traces.iter().map(|x| x.0.active_set.set.len()).collect::<Vec<usize>>());

        println!("Current attentions {:?}", self.parallel_traces.iter().map(|x| x.0.active_set.current_attention()).collect::<Vec<_>>());

        */
        
        let mut rng = rng_maker();

        let swap_odd: bool = rng.gen();


        //This swaps the pairs of adjacent traces starting from index 1
        
        if swap_odd {
        
            println!("chose odd");
            self.parallel_traces[1..].par_chunks_exact_mut(2).for_each(|x| {
            let (c, d) = x.split_at_mut(1);
            let (a, b) = (&mut c[0], &mut d[0]);
                std::mem::swap(&mut a.0.active_set, &mut b.0.active_set);
            });

            let num_traces = self.parallel_traces.len();
            let swap_slice = if num_traces & 1 == 0 {&mut self.parallel_traces[1..(num_traces-1)]} else { &mut self.parallel_traces[1..]};
            
            for _ in 0..cooldown_iters {
                swap_slice.par_iter_mut().for_each(|(set, track)| {
                    let mut rng = rng_maker();
                    for _ in 0..iters_between_shuffles { set.advance(track.as_mut(), false, &mut rng); }
                });
            }

        } else {
            println!("chose even");
            self.parallel_traces.par_chunks_exact_mut(2).for_each(|x| {
                let (c, d) = x.split_at_mut(1);
            let (a, b) = (&mut c[0], &mut d[0]);
                std::mem::swap(&mut a.0.active_set, &mut b.0.active_set);
            });
            
            let num_traces = self.parallel_traces.len();
            let swap_slice = if num_traces & 1 == 0 {self.parallel_traces.as_mut_slice()} else { &mut self.parallel_traces[1..(num_traces-1)]};
            
            for _ in 0..cooldown_iters {
                swap_slice.par_iter_mut().for_each(|(set, track)| {
                    let mut rng = rng_maker();
                    for _ in 0..iters_between_shuffles { set.advance(track.as_mut(), false, &mut rng); }
                });
            }
        }


        println!("Ln posteriors of each trace after swaps and cool down: {:?}", self.parallel_traces.iter_mut().map(|x| x.0.active_set.ln_posterior()).collect::<Vec<f64>>());
        
        /*if self.parallel_traces.len() >= 3  {
            let odd_attention_swaps: Vec<(_, bool)> = self.parallel_traces[1..].par_chunks_exact_mut(2).map(|x| { 
                let (c, d) = x.split_at_mut(1);
                let (a, b) = (&mut c[0], &mut d[0]);
                let mut rng = rng_maker();
                let attention_0 = a.0.active_set.current_attention().clone();
                let attention_1 = b.0.active_set.current_attention().clone();
                let (mut swap_attention_a, swap_ln_post_a) = a.0.active_set.set_with_new_attention(attention_1);
                let (mut swap_attention_b, swap_ln_post_b) = b.0.active_set.set_with_new_attention(attention_0);

                let accept_swap = MotifSet::accept_test(a.0.thermo_beta*swap_ln_post_a+b.0.thermo_beta*swap_ln_post_b,
                                                        a.0.thermo_beta*a.0.active_set.ln_posterior()+b.0.thermo_beta*b.0.active_set.ln_posterior(), 1.0, &mut rng);

                if accept_swap {
                    std::mem::swap(&mut a.0.active_set, &mut swap_attention_a);
                    std::mem::swap(&mut b.0.active_set, &mut swap_attention_b);
                }

                //([a.0.thermo_beta, b.0.thermo_beta], accept_swap)
                ([a.0.active_set.current_attention().clone(), b.0.active_set.current_attention().clone()], accept_swap)
            }).collect();

            println!("Odd attention swaps: {:?}", odd_attention_swaps.iter().map(|a| format!("{:?} att {}", a.0, a.1)).collect::<Vec<_>>() );
        }*/
    }

    pub fn print_acceptances(&self, track: TrackingOptions) {

        let tracker = self.track.min(track);

        match tracker {
            TrackingOptions::NoTracking => (),
            TrackingOptions::TrackTrueTrace => {
                println!("Motif lengths: {:?}", self.parallel_traces[0].0.active_set.set.iter().map(|a| a.len()).collect::<Vec<_>>());
                println!("Motif heights: {:?}", self.parallel_traces[0].0.active_set.set.iter().map(|a| a.peak_height()).collect::<Vec<_>>());
                self.parallel_traces[0].1.as_ref().map(|a| a.give_status());},
            TrackingOptions::TrackAllTraces => { _ = self.parallel_traces.iter().map(|b| { 
                println!("Thermodynamic beta: {}", b.0.thermo_beta);
                println!("Motif lengths: {:?}", b.0.active_set.set.iter().map(|a| a.len()).collect::<Vec<_>>());
                println!("Motif heights: {:?}", b.0.active_set.set.iter().map(|a| a.peak_height()).collect::<Vec<_>>());
                b.1.as_ref().map(|a| a.give_status())
            }).collect::<Vec<_>>();},
        }

    }

    pub fn handle_histograms(&self, track: TrackingOptions, pre_file_name: &str, num_bins: usize) {
        let tracker = self.track.min(track);
        
        match tracker {
            TrackingOptions::NoTracking => (),
            TrackingOptions::TrackTrueTrace => {self.parallel_traces[0].1.as_ref().map(|a| a.all_move_hists(pre_file_name, num_bins));},
            TrackingOptions::TrackAllTraces => {self.parallel_traces.par_iter().map(|b| b.1.as_ref()
                                                                                .map(|a| {
                                                                                    let alter_name = format!("{}_thermo_beta_{}", pre_file_name, b.0.thermo_beta);
                                                                                    a.all_move_hists(&alter_name, num_bins)
                                                                                })).collect::<Vec<_>>();},
        }


    }


    pub fn save_trace_and_clear(&mut self, output_dir: &str, run_name: &str, zeroth_step: usize) {

        self.parallel_traces[0].0.save_trace(output_dir, run_name, zeroth_step);
        self.parallel_traces[0].0.save_and_drop_history(output_dir, run_name, zeroth_step);

        for i in 1..self.parallel_traces.len() { self.parallel_traces[i].0.drop_history(); }

    }
}

pub enum InitializationError {
    UnsynchedData,
    NegativeTemperature, 
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

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct TruncatedLogNormal {
    location: f64,
    scale: f64,
    min: f64, 
    max: f64,
    ref_dist: LogNormal,
}

impl TruncatedLogNormal {

    pub fn new(location: f64, scale: f64, min: f64, max: f64) -> Result<TruncatedLogNormal, StatsError> {
        if location.is_nan() || scale.is_nan() || scale <= 0.0 || (min > max) || (max < 0.0) {
            Err(StatsError::BadParams)
        } else {
            let min = if min >= 0.0 {min} else {0.0} ;
            let ref_dist = LogNormal::new(location, scale).unwrap();
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
    use crate::{DIRICHLET_PWM, PROPOSE_EXTEND};
    use crate::SymmetricBaseDirichlet;




    #[test]
    fn distance_fallibility() {
        let mut rng = rand::thread_rng();

        for _ in 0..10000 {

            let base_a = Base::rand_new(&mut rng);
            let base_b = Base::rand_new(&mut rng);
            let dist_1 = base_a.dist_sq(Some(&base_b));
            let dist_2 = base_b.dist_sq(Some(&base_a));
            println!("{base_a:?} {base_b:?} {dist_1} {dist_2}");
        }

    }

    #[test]
    fn distances_test() {
        let mut rng = rand::thread_rng();
        let m = Motif::from_motif(vec![Bp::A, Bp::C, Bp::G, Bp::G, Bp::G, Bp::G, Bp::T, Bp::T, Bp::T], &mut rng);
        let dself = m.distance_function(&m);
        assert!(dself.0.abs() < 1e-16);
        assert!(!dself.1);
        let drev = m.distance_function(&(Motif::raw_pwm(m.rev_complement(), m.peak_height())));
        assert!(drev.0.abs() < 1e-16);
        assert!(drev.1);
        let zero_base = Base::new([1.0;BASE_L]).unwrap();
        println!("zero dist {}", zero_base.dist_sq(None));
        let mut m_star_pwm = m.pwm.clone();
        m_star_pwm.insert(0,zero_base.clone());
        m_star_pwm.push(zero_base);
        let unaltered_check = Motif::raw_pwm(m_star_pwm.clone(), m.peak_height());
        let d_unalt = m.distance_function(&unaltered_check);
        println!("m {:?} \n m_star {:?} {:?}", m, unaltered_check, m_star_pwm);
        println!("unalt {:?}", d_unalt);
        assert!(d_unalt.0.abs() < 1e-16);
        assert!(!d_unalt.1);
        let mut new_b_vec: Vec<Base> = Vec::with_capacity(m.len()+2);
        new_b_vec.push(Base::rand_new(&mut rng));
        new_b_vec.append(&mut m.rev_complement());
        new_b_vec.push(Base::rand_new(&mut rng));
        let newmot = Motif::raw_pwm(new_b_vec, m.peak_height());
        let d_unalt_rev = m.distance_function(&newmot);
        println!("new_b_vec {:?} {:?} {}", newmot, d_unalt_rev, (newmot.pwm[0].dist_sq(None)+newmot.pwm[newmot.len()-1].dist_sq(None)).sqrt());
        assert!(d_unalt_rev.0.abs()-(newmot.pwm[0].dist_sq(None)+newmot.pwm[newmot.len()-1].dist_sq(None)).sqrt() < 1e-16);
        assert!(d_unalt_rev.1);
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
        let accessed_c: Vec<f64> = usize_vec.iter().map(|&b| base.props[b]).collect();
        let bad_safe_time = bad_safe_access.elapsed();

        let safe_access = Instant::now();
        let accessed: Vec<f64> = bp_vec.iter().map(|&b| base[b]).collect();
        let safe_time = safe_access.elapsed();

        let unsafe_access = Instant::now();
        let accessed_b: Vec<f64> = usize_vec.iter().map(|&b| unsafe{base.rel_bind(b)}).collect();
        let unsafe_time = unsafe_access.elapsed();


        println!("{:?}", accessed.len());
        println!("{:?}", accessed_b.len());
        println!("{:?}", accessed_c.len());
        println!("safe {:?} unsafe {:?} bad_safe: {:?}", safe_time, unsafe_time, bad_safe_time);
    }



    #[test]
    fn simplex_test() {
        println!("start");
        let mut rng = rand::thread_rng();
        let b = Base::rand_new(&mut rng);

        let simplex = b.base_to_vect();

        let mod_b = Base::vect_to_base(&simplex);

        println!("st {:?} {:?} {:?} {} {} {}", b, simplex, mod_b, b.dist_sq(None), b.dist_sq(Some(&b)), simplex.iter().map(|&a| a.powi(2)).sum::<f64>().sqrt());

        let _mot = Motif::from_motif(vec![Bp::A,Bp::C, Bp::T, Bp::G, Bp::C,Bp::T, Bp::T, Bp::A, Bp::C] , &mut rng);


        let b0 = b.add_in_vector_space([1e-6, 0.0, 0.0], false);
        let b1 =  b.add_in_vector_space([0.0, 1e-6, 0.0], false);
        let b2 =  b.add_in_vector_space([0.0, 0.0, 1e-6], false);

        println!("{:?} {:?} {} {} {}", b.base_to_vect(), b0.base_to_vect(), b0.base_to_vect()[0]-b.base_to_vect()[0],  b0.base_to_vect()[1]-b.base_to_vect()[1],  b0.base_to_vect()[2]-b.base_to_vect()[2]);
        println!("{:?} {:?} {} {} {}", b.base_to_vect(), b1.base_to_vect(), b1.base_to_vect()[0]-b.base_to_vect()[0],  b1.base_to_vect()[1]-b.base_to_vect()[1],  b1.base_to_vect()[2]-b.base_to_vect()[2]);
        println!("{:?} {:?} {} {} {}", b.base_to_vect(), b2.base_to_vect(), b2.base_to_vect()[0]-b.base_to_vect()[0],  b2.base_to_vect()[1]-b.base_to_vect()[1],  b2.base_to_vect()[2]-b.base_to_vect()[2]);

        
        println!("{:?} {:?} {:?}", b0.props.iter().zip(b.props.iter()).map(|(&a, &b)| (a-b)/1e-6).collect::<Vec<_>>(), b1.props.iter().zip(b.props.iter()).map(|(&a, &b)| (a-b)/1e-6).collect::<Vec<_>>(),
                 b2.props.iter().zip(b.props.iter()).map(|(&a, &b)| (a-b)/1e-6).collect::<Vec<_>>());

        let simp = [0.1_f64, -0.15, -0.11];
        let simp_b = Base::vect_to_base(&simp);

        let b3a = simp_b.add_in_vector_space([0.0, 0.0, 0.02], false);
        let b3 = simp_b.add_in_vector_space([0.0, 0.0, 0.02], true);

        let sim_b3a = b3a.base_to_vect();
        let sim_b3 = b3.base_to_vect();

        println!("sim {:?} simb {:?} noref {:?} norefb {:?} ref {:?} refb{:?}", simp, simp_b, b3a.base_to_vect(), b3a, b3.base_to_vect(), b3);

        assert!((simp[0]-sim_b3a[0]).abs() < 1e-6, "0th element changes with no reflection");
        assert!((0.09-sim_b3[0] ).abs() < 1e-6, "0th element changes with a reflection");

        assert!((simp[1]-sim_b3a[1]).abs() < 1e-6, "1st element changes with no reflection");
        assert!((simp[1]-sim_b3[1] ).abs() < 1e-6, "1st element changes with a reflection");

        println!("{:?} {:?}", sim_b3a, sim_b3);
        assert!((-0.09-sim_b3a[2]).abs() < 1e-6, "2nd element incorrect with no reflection");
        assert!((-simp[0]-sim_b3[2]).abs() < 1e-6, "2nd element incorrect with a reflection");

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






    }

    #[test]
    fn leap_test() {


        println!("begin");
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
        let preblocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.gen::<u8>()).collect();
        let blocks: Vec<u8> = preblocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>(); 
        let _block_u8_starts: Vec<usize> = (0..block_n).map(|a| a*u8_per_block).collect();
        let block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
        let _start_bases: Vec<usize> = (0..block_n).map(|a| a*bp_per_block).collect();
        let sequence: Sequence = Sequence::new_manual(blocks, block_lens.clone());
        

        let pre_null_blocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.gen::<u8>()).collect();
        let null_blocks: Vec<u8> = pre_null_blocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>(); 
        let null_block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
        let start_null_bases: Vec<usize> = (0..block_n).map(|a| 2*a*bp_per_block).collect();
        let null_seq: NullSequence = NullSequence::new_manual(null_blocks, null_block_lens.clone());

        let wave: Waveform = Waveform::create_zero(&sequence, 5);
        let duration = start_gen.elapsed();
        println!("grad gen {} bp {:?}", bp, duration);

        let background = Background::new(0.25, 2.64, 350./6.);

        let start_bases_ids: Vec<usize> = (0..block_n).map(|a| (2*a+1)*bp_per_block+10).collect();

        let data_seq = unsafe{ AllDataUse::new_unchecked_data(wave.clone(),&null_seq, &start_bases_ids,&start_null_bases, &background)};

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
            all_base_correct &= mot_scram.pwm[base][scram_kmer[base]] == 1.0;
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

        let leaped = motif_set.base_leap(1.0, &mut rng);

        let leap = leaped.get_nth_motif(0);
        let leap1 = leaped.get_nth_motif(1);

        assert!((mot.peak_height() == leap.peak_height()) || (mot.peak_height() == -leap.peak_height()), "zeroth motif height is wrong");
        assert!((mot1.peak_height() == leap1.peak_height()) || (mot1.peak_height() == -leap1.peak_height()), "first motif height is wrong");

        let leap_kmer = leap.best_motif();
        let leap1_kmer = leap1.best_motif();


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

        assert!((ln_post-leaped.ln_post.unwrap()).abs() <= 64.0*std::f64::EPSILON, "ln posteriors not lining up"); 

        let recalced_signal = leaped.recalced_signal();

        let sig_diff = &leaped.signal-&recalced_signal;

        let any_diffs: Vec<f64> = sig_diff.raw_wave().iter().filter(|a| a.abs() > 64.0*std::f64::EPSILON).map(|&a| a).collect();

        println!("any diff {:?}", any_diffs);

        assert!(any_diffs.len() == 0, "waves not lining up"); 
        

        let shuffed = motif_set.secondary_shuffle(1.0, &mut rng);

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
    }

    #[test]
    fn rj_manipulator_tests() {

        {
            DIRICHLET_PWM.try_insert(SymmetricBaseDirichlet::new(10.0_f64).expect("obviously valid")).ok();
            PROPOSE_EXTEND.try_insert(SymmetricBaseDirichlet::new(1.0_f64).expect("obviously valid")).ok();
        }
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
        let preblocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.gen::<u8>()).collect();
        let blocks: Vec<u8> = preblocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>(); 
        let _block_u8_starts: Vec<usize> = (0..block_n).map(|a| a*u8_per_block).collect();
        let block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
        let _start_bases: Vec<usize> = (0..block_n).map(|a| a*bp_per_block).collect();
        let sequence: Sequence = Sequence::new_manual(blocks, block_lens.clone());
        

        let _pre_null_blocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.gen::<u8>()).collect();
        let null_blocks: Vec<u8> = preblocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>(); 
        let _block_u8_null_starts: Vec<usize> = (0..block_n).map(|a| a*u8_per_block).collect();
        let null_block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
        let start_null_bases: Vec<usize> = (0..block_n).map(|a| 2*a*bp_per_block).collect();
        let null_seq: NullSequence = NullSequence::new_manual(null_blocks, null_block_lens.clone());

        let wave: Waveform = Waveform::create_zero(&sequence, 5);
        let duration = start_gen.elapsed();
        println!("grad gen {} bp {:?}", bp, duration);

        let background = Background::new(0.25, 2.64, 350./6.);

        let start_bases_ids: Vec<usize> = (0..block_n).map(|a| (2*a+1)*bp_per_block+10).collect();

        let data_seq = unsafe{ AllDataUse::new_unchecked_data(wave.clone(),&null_seq, &start_bases_ids,&start_null_bases, &background)};
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

        let all_good = check_diff.read_wave().iter().map(|&a| a.abs() < 1024.0*std::f64::EPSILON).fold(true, |f, x| f && x);

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
        assert!(motif_set.set[0].peak_height() == add_mot.peak_height());


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

        assert!(motif_set.set[0].peak_height() == add_mot2.peak_height());


        //move tests. These should all produce Options of tuples
        //None should be produced iff executing the move would produce
        //a theoretically impossible result (eg, a motif set with no motifs).
        //Otherwise Some((MotifSet, f64)) should be produced, with the f64
        //being a sum of the ln posterior of the new motif set and the 
        //ln ancillary probability density: the lapd of growth moves is
        //-ln(density of proposing the particular growth), while the 
        //lapd of reverse moves is +ln(density of proposing the reversing growth)


        let mut failures = 0_usize;

        let (single_mot, _ln_prop) = loop { if let Some(r) = motif_set.propose_ordered_base_move_custom(&mut rng, 1.0, 1.0) { break r; } else {failures+=1;}};

        println!("failed in single base move {} times before succeeding", failures);
        assert!(single_mot.set.len() == motif_set.set.len());
        let mut found_change: bool = false;
        for i in 0..motif_set.set.len() {
            assert!(motif_set.set[i].peak_height() == single_mot.set[i].peak_height());
            let unchanged = motif_set.set[i].pwm.iter().zip(single_mot.set[i].pwm.iter()).all(|(a, b)| a.dist_sq(Some(b)).sqrt() < 1e-6);
            if !unchanged {
                if found_change {
                    panic!("multiple motifs changed in single base scale move!");
                } else { 
                    found_change = true; 

                    let changed_bases = motif_set.set[i].pwm.iter().zip(single_mot.set[i].pwm.iter()).enumerate().filter(|(_, (a, b))| a.dist_sq(Some(*b)).sqrt() >= 1e-6).map(|(i,_)| i).collect::<Vec<_>>();

                    println!("{}", changed_bases.len());
                    assert!(changed_bases.len() == 1);

                    let old_base = motif_set.set[i].pwm[changed_bases[0]].clone();
                    let new_base = single_mot.set[i].pwm[changed_bases[0]].clone();

                    println!("{:?} {:?} {:?} {:?}", old_base, old_base.base_to_vect(), new_base, new_base.base_to_vect());
                    let mut old_order = old_base.props.iter().enumerate().collect::<Vec<_>>();
                    let mut new_order = new_base.props.iter().enumerate().collect::<Vec<_>>();

                    old_order.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
                    new_order.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());

                    //let order_maintained = old_order.into_iter().zip(new_order).all(|((a,_), (b,_))| a==b);

                    //assert!(order_maintained);
                }
            }
        }


        let mut failures = 0_usize;
        let (single_mot, _ln_prop) = loop { if let Some(r) = motif_set.propose_ordered_motif_move_custom(&mut rng, 1.0, 1.0) { break r; } else {failures+=1;}};

        println!("failed in single motif move {} times before succeeding", failures);
        assert!(single_mot.set.len() == motif_set.set.len());
        let mut found_change: bool = false;
        for i in 0..motif_set.set.len() {
            assert!(motif_set.set[i].peak_height() == single_mot.set[i].peak_height());
            let unchanged = motif_set.set[i].pwm.iter().zip(single_mot.set[i].pwm.iter()).all(|(a, b)| a.dist_sq(Some(b)).sqrt() < 1e-6);
            if !unchanged {
                if found_change {
                    panic!("multiple motifs changed in single base scale move!");
                } else { 
                    found_change = true; 

                    let changed_bases = motif_set.set[i].pwm.iter().zip(single_mot.set[i].pwm.iter()).enumerate().filter(|(_, (a, b))| a.dist_sq(Some(*b)).sqrt() >= 1e-6).map(|(i,_)| i).collect::<Vec<_>>();

                    println!("{}", changed_bases.len());

                    for j in 0..changed_bases.len(){
                        let old_base = motif_set.set[i].pwm[changed_bases[j]].clone();
                        let new_base = single_mot.set[i].pwm[changed_bases[j]].clone();

                        println!("{:?} {:?} {:?} {:?}", old_base, old_base.base_to_vect(), new_base, new_base.base_to_vect());
                        let mut old_order = old_base.props.iter().enumerate().collect::<Vec<_>>();
                        let mut new_order = new_base.props.iter().enumerate().collect::<Vec<_>>();

                        old_order.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());
                        new_order.sort_by(|a, b| a.1.partial_cmp(b.1).unwrap());

                        //let order_maintained = old_order.into_iter().zip(new_order).all(|((a,_), (b,_))| a==b);

                        //assert!(order_maintained);
                    }
                }
            }
        }
        

        let single_height = motif_set.propose_height_move_custom(&mut rng, 1.0);

        assert!(single_height.is_some()); //scaling a single base should always make possible motifs

        let (single_mot, _ln_prop) = single_height.unwrap();


        assert!(single_mot.set.len() == motif_set.set.len());
        let mut found_change: bool = false;
        for i in 0..motif_set.set.len() {
            let height_unchanged = motif_set.set[i].peak_height() == single_mot.set[i].peak_height();

            if found_change { 
                assert!(height_unchanged, "Multiple heights changing");
            } else  {
                if !height_unchanged { found_change = true; }
            }

            let unchanged = motif_set.set[i].pwm.iter().zip(single_mot.set[i].pwm.iter()).all(|(a, b)| a.dist_sq(Some(b)).sqrt() < 1e-6);
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
            assert!(motif_set.set[i].peak_height() == birth_mot.set[i].peak_height());
        }

        println!("birth mot {:?}", birth_mot);
        println!("ln_prop {ln_prop} {}", birth_mot.calc_ln_post());
        let should_prior = ln_prop-(birth_mot.calc_ln_post());

        let remaining = motif_set.data_ref.data()-&motif_set.signal;
        let _propensities = remaining.kmer_propensities(birth_mot.set[l].len());
        let pick_prob = birth_mot.set[l].pwm.iter().map(|a| DIRICHLET_PWM.get().expect("no writes expected now").pdf(a)).product::<f64>();


        let actual_prior = HEIGHT_PROPOSAL_DIST.ln_pdf(birth_mot.set[l].peak_height()-MIN_HEIGHT)+pick_prob.ln()-((MAX_BASE+1-MIN_BASE) as f64).ln()-((motif_set.data_ref.data().seq().number_unique_kmers(birth_mot.set[l].len()) as f64).ln());

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
            let mut matching = motif_set.set[j].peak_height() == death_mot.set[i].peak_height();
            if !matching {
                assert!(!found_off, "Found two mismatches when looking for deleted motif!");
                found_off = true;
                l = j;
                matching = motif_set.set[j+1].peak_height() == death_mot.set[i].peak_height();
            }
            assert!(matching, "there is a mismatch!");
        }

        let should_prior = ln_prop-(death_mot.calc_ln_post());

        let remaining = motif_set.data_ref.data()-&death_mot.signal;
        let propensities = remaining.kmer_propensities(motif_set.set[l].len());
        let pick_prob = motif_set.set[l].pwm.iter().map(|a| DIRICHLET_PWM.get().expect("no writes expected now").pdf(a)).product::<f64>();


        let actual_prior = HEIGHT_PROPOSAL_DIST.ln_pdf(birth_mot.set[l].peak_height()-MIN_HEIGHT)+pick_prob.ln()-((MAX_BASE+1-MIN_BASE) as f64).ln()-((motif_set.data_ref.data().seq().number_unique_kmers(motif_set.set[l].len()) as f64).ln());

        println!("priors {} {} {} {} {}", motif_set.set[l].height_prior(), should_prior, pick_prob.ln(), ((MAX_BASE+1-MIN_BASE) as f64).ln(), propensities[motif_set.data_ref.data().seq().id_of_u64_kmer_or_die(motif_set.set[l].len(),Sequence::kmer_to_u64(&motif_set.set[l].best_motif()))]);

        //Remember, we can sometimes have a motif that's impossible to kill because it's impossible to be created
        assert!((should_prior == -f64::INFINITY && actual_prior == -f64::INFINITY) || ((should_prior-actual_prior).abs() < 1e-6), "{}", format!("{}", should_prior-actual_prior).as_str());

        //fn propose_extend_motif<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<(Self, f64)>

        let mut track = None;
        while let None = track {
            track = motif_set.propose_extend_motif(&mut rng);
        };

        let (extend_mot, ln_prop) = track.expect("we didn't let this proceed until it worked");

        println!("{:?}", extend_mot);

        assert!(extend_mot.set.len() == motif_set.set.len());

        let mut l: Option<usize> = None;
        let mut found_off = false;
        for i in 0..extend_mot.set.len() {
            assert!(motif_set.set[i].peak_height() == extend_mot.set[i].peak_height());
            let mut matching = motif_set.set[i].len() == extend_mot.set[i].len();
            if !matching {
                assert!(!found_off, "Found two mismatches when looking for extended motif!");
                assert!(sequence.kmer_in_seq(&(extend_mot.set[i].best_motif())), "Extension can produce illegal kmer!");
                found_off = true;
                l = Some(i);
                matching = (motif_set.set[i].len()+1) == extend_mot.set[i].len();
                matching = motif_set.set[i].pwm.iter()
                    .zip(extend_mot.set[i].pwm[0..(extend_mot.set[i].len()-1)].iter())
                    .map(|(a, b)| *a == *b) //We implemented a fuzzy equality for partialeq of bases
                    .fold(matching, |acc, b| acc && b);

            }
            assert!(matching, "there is a mismatch {} {}!", i, found_off);
        }

        assert!(l.is_some(), "Not finding a changed motif!");

        let should_prior = ln_prop-(extend_mot.calc_ln_post());

        let actual_prior = PROPOSE_EXTEND.get().expect("no writes expected now").ln_pdf(&(extend_mot.set[l.unwrap()].pwm.last().expect("We know this is bigger than 0")));

        assert!((should_prior+actual_prior).abs() < 1e-6, "{}", format!("{}", should_prior+actual_prior).as_str());

        let cant_extend_set = MotifSet::rand_with_one_height_and_motif_len(9.6, MAX_BASE, &data_seq, &mut rng);

        assert!(cant_extend_set.propose_extend_motif(&mut rng).is_none(), "Can extend PWM beyond allowed limits!");


        //fn propose_contract_motif<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<(Self, f64)>

        let mut track = None;
        while let None = track {
            track = motif_set.propose_contract_motif(&mut rng);
        };

        let (contract_mot, ln_prop) = track.expect("we didn't let this proceed until it worked");

        println!("{:?}", contract_mot);

        assert!(contract_mot.set.len() == motif_set.set.len());

        let mut l: Option<usize> = None;
        let mut found_off = false;
        for i in 0..contract_mot.set.len() {
            assert!(motif_set.set[i].peak_height() == contract_mot.set[i].peak_height());
            let mut matching = motif_set.set[i].len() == contract_mot.set[i].len();
            if !matching {
                assert!(!found_off, "Found two mismatches when looking for contracted motif!");
                found_off = true;
                l = Some(i);
                matching = (motif_set.set[i].len()-1) == contract_mot.set[i].len();
                matching = contract_mot.set[i].pwm.iter()
                    .zip(motif_set.set[i].pwm[0..(contract_mot.set[i].len()-1)].iter())
                    .map(|(a, b)| *a == *b) //We implemented a fuzzy equality for partialeq of bases
                    .fold(matching, |acc, b| acc && b);

            }
            assert!(matching, "there is a mismatch {} {}!", i, found_off);
        }

        assert!(l.is_some(), "Not finding a changed motif!");

        let should_prior = ln_prop-(contract_mot.calc_ln_post());

        let actual_prior = PROPOSE_EXTEND.get().expect("no writes expected now").ln_pdf(&(motif_set.set[l.unwrap()].pwm.last().expect("We know this is bigger than 0")));

        assert!((should_prior-actual_prior).abs() < 1e-6, "{}", format!("{}", should_prior-actual_prior).as_str());

        let cant_contract_set = MotifSet::rand_with_one_height_and_motif_len(9.6, MIN_BASE, &data_seq, &mut rng);

        assert!(cant_contract_set.propose_contract_motif(&mut rng).is_none(), "Can contract PWM beyond allowed limits!");



        //Testing full RJ move
        for _ in 0..3{
            let add_mot = Motif::rand_mot_with_height_and_motif_len(13.2,mid_mot, data_seq.data().seq(), &mut rng);
            _ = motif_set.add_motif(add_mot);
        }
        for _i in 0..100 {
            let (step_set, selected_move, accepted) = motif_set.run_rj_move(1.0, &mut rng);
            
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
        assert_eq!(base, b.best_base());

        let bc: Base = b.rev();

        assert_eq!(bc.best_base(), base.complement());

        let tc: Base = try_base.rev();

        assert_eq!(tc.best_base(), try_base.best_base().complement());

        assert!(!(tc == try_base));
        assert!(b == b.clone());

        let _b_mag: f64 = b.show().iter().sum();
        let supposed_default_dist = (b.base_to_vect()).iter().map(|a| a.powi(2)).sum::<f64>();

        println!("{supposed_default_dist}, {}", b.dist_sq(None));
        assert!((supposed_default_dist - b.dist_sq(None)).abs() < 1e-7);

        //println!("Conversion dists: {:?}, {:?}, {}", b.show(),  b.to_gbase().to_base().show(), b.dist_sq(Some(&b.to_gbase().to_base())));
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

        let val1: f64 = 3.0;
        let val2: f64 = 15.0;

        assert!(((free_dist.pdf(val1)/free_dist.pdf(val2))-(dist.pdf(val1)/dist.pdf(val2))).abs() < 1e-6);
        assert!(((free_dist.cdf(val1)/free_dist.cdf(val2))-(dist.cdf(val1)/dist.cdf(val2))).abs() < 1e-6);

        assert!(dist.ln_pdf(MAX_HEIGHT+1.0).is_infinite() && dist.ln_pdf(MAX_HEIGHT+1.0) < 0.0);

        let dist: TruncatedLogNormal = TruncatedLogNormal::new(LOG_HEIGHT_MEAN, LOG_HEIGHT_SD, -1.0, MAX_HEIGHT).unwrap();

        assert!(dist.min().abs() < 1e-6);

    } 

    #[test]
    fn motif_establish_tests() {

        std::env::set_var("RUST_BACKTRACE", "1");

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
        
        let preblocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.gen::<u8>()).collect();
        let blocks: Vec<u8> = preblocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>(); 
        let block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
        let mut start_bases: Vec<usize> = (0..block_n).map(|a| a*bp_per_block).collect();
        let sequence: Sequence = Sequence::new_manual(blocks, block_lens.clone());
        

        let pre_null_blocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.gen::<u8>()).collect();
        let null_blocks: Vec<u8> = pre_null_blocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>(); 
        let _block_u8_null_starts: Vec<usize> = (0..block_n).map(|a| a*u8_per_block).collect();
        let null_block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
        let start_null_bases: Vec<usize> = (0..block_n).map(|a| 2*a*bp_per_block).collect();
        let null_seq: NullSequence = NullSequence::new_manual(null_blocks, null_block_lens.clone());

        let start_bases_copy = (0..block_n).map(|a| (2*a+1)*bp_per_block+10).collect();


        let wave: Waveform = Waveform::create_zero(&sequence, 5);
        let duration = start_gen.elapsed();
        println!("grad gen {} bp {:?}", bp, duration);

        let background = Background::new(0.25, 2.64, 20.);


        let data_seq = unsafe{ AllDataUse::new_unchecked_data(wave.clone(),&null_seq, &start_bases_copy,&start_null_bases, &background)};
        println!("Done gen {} bp {:?}", bp, duration);

        println!("{} gamma", gamma(4.));
        println!("{} gamma", ln_gamma(4.));

        //println!("{:?}", wave.raw_wave());

        let motif: Motif = Motif::from_motif(sequence.return_bases(0,0,20), &mut rng); //sequence

        let motif2: Motif = Motif::from_motif(sequence.return_bases(0,2,20), &mut rng); //sequence

        let start = Instant::now();

        let waveform = motif.generate_waveform(&data_seq);
        let duration = start.elapsed();

        let waveform2 = &waveform + &(motif2.generate_waveform(&data_seq));

        let coordinate_bases: Vec<usize> = start_bases.iter().map(|&a| a+spacing_dist.sample(&mut rng)).collect();

        let data_seq_2 = unsafe{ AllDataUse::new_unchecked_data(waveform2, &null_seq, &coordinate_bases,&start_null_bases ,&background) }; 

        let _noise: Noise = waveform.produce_noise(&data_seq_2);



        let waveform_raw = waveform.raw_wave();

        let binds = motif.return_bind_score(&sequence);

        let null_blocks: Vec<usize> = (0..null_block_lens.len()).collect();

        let start_null = Instant::now();
        let _null_binds = motif.return_any_null_binds_in_group(&null_seq,&null_blocks);
        let duration_null = start_null.elapsed();

        println!("null dur {:?}", duration_null);


        let start_b = Instant::now();
        let unsafe_waveform = unsafe{ motif.generate_waveform_from_binds(&binds, &data_seq) };
        let duration_b = start_b.elapsed();



        let unsafe_raw = unsafe_waveform.raw_wave();

        assert!(unsafe_raw.len() == waveform_raw.len());
        assert!((unsafe_raw.iter().zip(&waveform_raw).map(|(a, &b)| (a-b).powf(2.0)).sum::<f64>()).powf(0.5) < 1e-6);

        println!("Time elapsed in generate_waveform() is: {:?}", duration);
        println!("Time elapsed in the unsafe generate_waveform() is: {:?}", duration_b);

        println!("{}", motif);

        let random_motif = Motif::rand_mot(&sequence, &mut rng);

        println!("Random motif\n{}", random_motif);

        println!("Kernel length {}", background.kernel_ref().len());

        assert!(background.kernel_ref().len() == 121);

        assert!((random_motif.peak_height.abs() >= MIN_HEIGHT) && (random_motif.peak_height.abs() <= MAX_HEIGHT));

        let matrix = motif.pwm();
        let _nbases = matrix.len();

        for base in matrix {
            println!("{:?}", base.show());
        }

        println!("{:?}", motif.best_motif());

        println!("{:?}", sequence.return_bases(0,0,20));
        //println!("{:?}", sequence.unique_kmers(motif.len()));
        println!("{}", Sequence::kmer_to_u64(&motif.best_motif()));
        let matrix = motif.rev_complement();

        for base in &matrix {
            println!("{:?}", base.show());
        }

        //assert!(((motif.pwm_prior()/gamma::ln_gamma(BASE_L as f64))+(motif.len() as f64)).abs() < 1e-6);

        println!("{} {} {} PWM PRIOR",sequence.kmer_in_seq(&motif.best_motif()), motif.pwm_prior(&sequence), (sequence.number_unique_kmers(motif.len()) as f64).ln() - (motif.len() as f64)*BASE_PRIOR_DENS.ln());
        assert!((motif.pwm_prior(&sequence)+(sequence.number_unique_kmers(motif.len()) as f64).ln()
                 -(motif.len() as f64)*BASE_PRIOR_DENS.ln()).abs() < 1e-6);

        let un_mot: Motif = Motif::from_motif(vec![Bp::C;20],&mut rng);//Sequence

        assert!(un_mot.pwm_prior(&sequence) < 0.0 && un_mot.pwm_prior(&sequence).is_infinite());

        let dist: TruncatedLogNormal = TruncatedLogNormal::new(LOG_HEIGHT_MEAN, LOG_HEIGHT_SD, MIN_HEIGHT, MAX_HEIGHT).unwrap();

        let mut base_prior = motif.height_prior();
        if motif.peak_height() > 0.0 {
            base_prior -= PROB_POS_PEAK.ln();
        } else {
            base_prior -= (1.0-PROB_POS_PEAK).ln();
        }

        println!("{} {} pro", base_prior.exp(), dist.pdf(motif.peak_height().abs()));
        assert!((base_prior.exp()-dist.pdf(motif.peak_height().abs())).abs() < 1e-6);


        let best_mot = motif.best_motif();

        let bindy = unsafe{ motif.prop_binding(&best_mot) };

        assert!(((bindy.0-1.0).abs() < 1e-6) && !bindy.1);

        let rev_best = best_mot.iter().rev().map(|a| a.complement()).collect::<Vec<Bp>>();

        let bindy = unsafe {motif.prop_binding(&rev_best) };

        assert!(((bindy.0-1.0).abs() < 1e-6) && bindy.1);

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

                            let defect_2 = defect*pwm[k][m];
                            let mut for_mot_2 = for_mot.clone();
                            for_mot_2[k] = m;
                            let mut rev_mot_2 = rev_mot.clone();
                            rev_mot_2[motif.len()-1-k] = m.complement();

                            //println!("for {k} {m} {:?}, rev {k} {m} {:?}", for_mot_2, rev_mot_2);
                            let bindy = unsafe{ motif.prop_binding(&for_mot_2)};
                            let rbind = unsafe{ motif.prop_binding(&rev_mot_2) };

                            //println!("defect {defect} pos {}, defect_2 {defect_2} bindy {:?} rbind {:?}",pwm[k][m], bindy, rbind);
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

            let forward_defect = altermot.iter().enumerate().map(|(i, &b)| rand_pwm[i][b]).product::<f64>();
            let reverse_defect = altermot.iter().rev().enumerate().map(|(i, &b)| rand_pwm[i][b.complement()]).product::<f64>();

            let is_rev = reverse_defect > forward_defect;

            let proper_defect = if is_rev {reverse_defect} else {forward_defect};

            println!("alter for {forward_defect} rev {reverse_defect} is_rev {is_rev}");

            let bindy = unsafe{ random_motif.prop_binding(&altermot)};

            assert!(((bindy.0-proper_defect).abs() < 1e-6) && (bindy.1 == is_rev));

        }
        
        for i in 0..null_block_lens.len() {
        
            let mut significant_binds: Vec<f64> = Vec::with_capacity(null_block_lens[i]);

            let group = [i];

            let mut binds = motif.return_any_null_binds_in_group(&null_seq,&group);

            for j in 0..(null_block_lens[i]-motif.len()) {

                let mot = null_seq.return_bases(i, j, motif.len());
                let (prop, _) = unsafe{ motif.prop_binding(&mot)};

                if prop > (-motif.peak_height).exp2() { significant_binds.push(prop); }

            }

            binds.sort_by(|b, a| a.partial_cmp(b).unwrap());
            significant_binds.sort_by(|b, a| a.partial_cmp(b).unwrap());

            if significant_binds.len() > CAPACITY_FOR_NULL {
                let _ = significant_binds.drain(CAPACITY_FOR_NULL..).collect::<Vec<_>>();
            }
            assert!(binds.len() == significant_binds.len(), "bind scores giving different lengths in null seq");
            for k in 0..binds.len() {

                assert!((binds[k]-significant_binds[k]).abs() < 1e-6, "not capturing proper binding {} {} {}", k, binds[k], significant_binds[k]);

            }

        }


        let wave_block: Vec<u8> = vec![2,0,0,0, 170, 170, 170, 170, 170, 170, 170, 170,170, 170, 170, 170, 170, 170, 170]; 
        let _wave_inds: Vec<usize> = vec![0, 9]; 
        let _wave_starts: Vec<usize> = vec![0, 36];
        let wave_lens: Vec<usize> = vec![36, 40];
        let wave_start_bases: Vec<usize> = vec![0, 9324];
        let wave_seq: Sequence = Sequence::new_manual(wave_block, wave_lens);
        let wave_wave: Waveform = Waveform::create_zero(&wave_seq,1);
        let wave_background = Background::new(0.25, 2.64, 1.0);


        let null_zeros: Vec<usize> = vec![144, 813];

        //This is in units of number of u8s
        let null_sizes: Vec<usize> = vec![78,17];

        let null_makeup: Vec<Vec<usize>> = null_sizes.iter().map(|&a| (0..(4*a)).map(|_| rng.gen_range(0_usize..4)).collect::<Vec<usize>>()).collect::<Vec<Vec<usize>>>();

        let invented_null: NullSequence =  NullSequence::new(null_makeup);



        let wave_data_seq = unsafe{ AllDataUse::new_unchecked_data(wave_wave,&invented_null, &wave_start_bases, &null_zeros, &wave_background) }; 
        let theory_base = [1.0, 1e-5, 1e-5, 0.2];

        let mat: Vec<Base> = (0..15).map(|_| Base::new(theory_base.clone()).expect("We designed theory_base to be statically valid")).collect::<Vec<_>>();


        println!("DF");
        let little_motif: Motif = Motif::raw_pwm(mat, 10.0); //wave_seq

        print!("{}", little_motif);
        println!("{:?}",little_motif.generate_waveform(&wave_data_seq).raw_wave());

        let small_block: Vec<u8> = vec![44, 24, 148, 240, 84, 64, 200, 80, 68, 92, 196, 144]; 
        let _small_inds: Vec<usize> = vec![0, 6]; 
        let small_lens: Vec<usize> = vec![24, 24];
        let small: Sequence = Sequence::new_manual(small_block, small_lens);
        let _small_wave: Waveform = Waveform::new(vec![0.1, 0.6, 0.9, 0.6, 0.1, -0.2, -0.4, -0.6, -0.6, -0.4], &small, 5);

        let mat: Vec<Base> = (0..15).map(|_| Base::new(theory_base.clone()).expect("We designed theory_base to be statically valid")).collect::<Vec<_>>();
        let wave_motif: Motif = Motif::raw_pwm(mat, 10.0); //small

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

        for i in 0..block_n {
            for j in 0..(bp_per_block-motif.len()) {

                //let test_against = motif.prop_binding(&VecDeque::from(sequence.return_bases(i, j, motif.len())));
                //SAFETY: sequence.return_bases is called with motif.len() as its length
                //        and always called so that we have that many bps left available
                let test_against = unsafe{ motif.prop_binding(&(sequence.return_bases(i, j, motif.len())))};
                assert!((binds.0[i*bp_per_block+j]-test_against.0).abs() < 1e-6);
                assert!(binds.1[i*bp_per_block+j] == test_against.1);

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

        let kernel_check = (background.kernel_ref()*1.0).get_curve().clone();

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
                    binds.0[a] > (-motif.peak_height()).exp2() 
                }).collect::<Vec<_>>();


                if relevant_binds.len() > 0 {

                    let mut score: f64 = 0.0;
                    for k in 0..relevant_binds.len() {
                        let to_add = (motif.peak_height()+binds.0[relevant_binds[k]].log2())*kernel_check[(kernel_mid+(start_bases[i]+space*j))-(relevant_binds[k]+half_len)];
                        //println!("to_add {to_add}");
                        score+=to_add;
                    }

                    scores[start_dats[i]+j] += score;
                    //println!("wave {} score {} ratio {}", wave_gen[start_dats[i]+j], score, wave_gen[start_dats[i]+j]/score); 
                    assert!((wave_gen[start_dats[i]+j]-score).abs() < 1e-6);


                } else {
                    assert!(wave_gen[start_dats[i]+j].abs() < 1e-6);
                }


            }
        }

        //println!("wave_gen-scores {:?}", scores.iter().zip(wave_gen.iter()).map(|(&a, &b)| a-b).collect::<Vec<_>>());


    }








} 
