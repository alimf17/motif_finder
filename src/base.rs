//pub mod bases {
use rand::Rng;
use rand::prelude::IteratorRandom;
use rand::seq::SliceRandom;
use rand::distributions::{Distribution, Uniform, WeightedIndex};
use statrs::distribution::{Continuous, ContinuousCDF, LogNormal, Normal, Dirichlet, Exp};
use statrs::statistics::{Min, Max, Distribution as OtherDistribution};
use statrs::Result as otherResult;
use crate::waveform::{Kernel, Waveform, WaveformDef, Noise, Background, WIDE};
use crate::sequence::{Sequence, BP_PER_U8, U64_BITMASK, BITS_PER_BP};
use crate::modified_t::ContinuousLnCDF;
use crate::{NUM_RJ_STEPS, NUM_BASE_LEAP_STEPS, NUM_HMC_STEPS, MAX_IND_RJ, MAX_IND_LEAP, MAX_IND_HMC, MOMENTUM_DIST, HMC_TRACE_STEPS, HMC_EPSILON};
use crate::data_struct::All_Data;
use statrs::function::gamma::*;
use statrs::{consts, StatsError};
use std::f64;
use std::fmt;
use std::error::Error;
use std::collections::{VecDeque, HashMap};
use std::time::{Duration, Instant};
use once_cell::sync::Lazy;
use assume::assume;
use rayon::prelude::*;

use log::warn;

use regex::Regex;
use std::fs;
use nalgebra::{DMatrix, DVector, dvector};

use serde::{ser::*, Serialize,Serializer, Deserialize};
use serde::de::{
    self, DeserializeSeed, EnumAccess, IntoDeserializer, MapAccess, SeqAccess,
    VariantAccess, Visitor,
};

use serde_json::{Result as JsonResult, Value};
use core::fmt::{Debug, Formatter};

const SQRT_2: f64 = 1.41421356237;
const SQRT_3: f64 = 1.73205080757;


pub const BPS: [char; 4] = ['A', 'C', 'G', 'T'];
pub const BASE_L: usize = BPS.len();

pub const VERTICES: [[f64; BASE_L-1]; (BASE_L)] = [[2.*SQRT_2/3., 0., 1.0/3.],[-SQRT_2/3., SQRT_2*SQRT_3/3., -1.0/3.], [-SQRT_2/3., -SQRT_2*SQRT_3/3., -1.0/3.],[0., 0., 1.0]];

//This MUST be the transpose of VERTICES
pub const SIMPLEX_VERTICES: [[f64; BASE_L]; (BASE_L-1)] = [[2.*SQRT_2/3. , -SQRT_2/3., -SQRT_2/3., 0.0], 
                                                           [0.              , SQRT_2*SQRT_3/3.   , -SQRT_2*SQRT_3/3., 0.0],
                                                           [1.0/3.          , -1.0/3.           , -1.0/3.           , 1.0]];

pub const INVERT_SIMPLEX: [[f64; BASE_L]; BASE_L] = [[ 3.0*SQRT_2/5.0,  0.0            , -0.3, 0.3], 
                                                         [-3.0*SQRT_2/20.,  SQRT_2*SQRT_3/4., -0.3, 0.3],
                                                         [-3.0*SQRT_2/20., -SQRT_2*SQRT_3/4., -0.3, 0.3],
                                                         [-3.0*SQRT_2/10.,  0.0            ,  0.9, 0.1]];


//Obviously, this should always be the transpose of INVERT_SIMPLEX. I'm just not good enough at compile time code to make it happen automatically
pub const COL_PRIMARY_INVERT_SIMPLEX: [[f64; BASE_L]; BASE_L] = [[ 3.0*SQRT_2/5.0,  -3.0*SQRT_2/20., -3.0*SQRT_2/20.  , -3.0*SQRT_2/10.], 
                                                                 [ 0.0           , SQRT_2*SQRT_3/4., -SQRT_2*SQRT_3/4.,   0.0          ],
                                                                 [ -0.3          ,  -0.3           ,  -0.3            ,   0.9          ],
                                                                 [  0.3          ,   0.3           ,   0.3            ,   0.1          ]];

pub const VERTEX_DOT: f64 = -1.0/((BASE_L-1) as f64);

const RT: f64 =  8.31446261815324*298./4184.; //in kcal/mol

const CLOSE: f64 = 1e-5;

pub const MIN_BASE: usize = 8;
pub const MAX_BASE: usize = 20; //For a four base system, the hardware limit here is 32. 
                                //To make sure this runs before the heat death of the universe
                                //while also not taking more memory than every human brain combined,
                                //We store many kmers as u64s. If you have 20 "bases" (amino acids)
                                //You need to force it to be at most 12. 


const MIN_HEIGHT: f64 = 3.;
const MAX_HEIGHT: f64 = 15.;
const LOG_HEIGHT_MEAN: f64 = 2.302585092994046; //This is ~ln(10). Can't use ln in a constant, and this depends on no other variables
const LOG_HEIGHT_SD: f64 = 0.25;

static HEIGHT_DIST: Lazy<TruncatedLogNormal> = Lazy::new(|| TruncatedLogNormal::new(LOG_HEIGHT_MEAN, LOG_HEIGHT_SD, MIN_HEIGHT, MAX_HEIGHT).unwrap() );

const PROB_POS_PEAK: f64 = 0.9;

pub const THRESH: f64 = 1e-3; //SAFETY: This must ALWAYS be strictly greater than 0, or else we violate safety guarentees later.  

const SPREAD_HMC_CONV: f64 = 15.0;

//This is roughly how much an additional motif should improve the ln posterior before it's taken seriously
//The more you increase this, the fewer motifs you will get, on average
const NECESSARY_MOTIF_IMPROVEMENT: f64 = 10.0_f64;

//When converting between gradient compatible and incompatible representations
//We sometimes end up with numerical errors that basically make infinities where there shouldn't be
//CONVERSION_MARGIN protects us during conversion, and REFLECTOR cuts off boundary values in HMC
//before they can get infinite and thus unfixable.  
//These numbers were empirically determined, not theoretically. 
const REFLECTOR: f64 = 40.0;
static PROP_CUTOFF: Lazy<f64> = Lazy::new(|| THRESH); 
static PROP_UPPER_CUTOFF: Lazy<f64> = Lazy::new(|| 1.0-2.0_f64.powf(-9.0));

static PROP_POS_REFLECTOR: Lazy<f64> = Lazy::new(|| Base::hmc_to_prop(REFLECTOR)); 
static PROP_NEG_REFLECTOR: Lazy<f64> = Lazy::new(|| Base::hmc_to_prop(-REFLECTOR)); 

static BASE_DIST: Lazy<Exp> = Lazy::new(|| Exp::new(1.0).unwrap());

pub const RJ_MOVE_NAMES: [&str; 4] = ["New motif", "Delete motif", "Extend motif", "Contract Motif"];

//BEGIN BASE

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Base {
   props: [ f64; BASE_L],
}

impl PartialEq for Base {
    fn eq(&self, other: &Self) -> bool {
        self.dist(Some(other)) < CLOSE    
    }
} 


impl Base {

    //Note: This will break if all the bindings are zeros, or any are negative
    pub fn new(props: [ f64; BASE_L]) -> Base {

        let mut props = props;

        let mut any_neg: bool = false;

        for i in 0..props.len() {
            any_neg |= props[i] < 0.0 ;
        }
        
        if any_neg{
           panic!("All base entries must be positive!"); 
        }

        let max = Self::max(&props);
        
        //We can rely on perfect float equality because max copies its result
        let mut maxes = props.iter().enumerate().filter(|(_,&a)| (a == max)).map(|(b, _)| b).collect::<Vec<usize>>(); 
        let max_ind = maxes[0]; //We should rarely if ever have exact equality. This is just me defending against the alternative
       

        for i in 0..props.len() {
            if i == max_ind {
                props[i] = 1.0;
            } else {
                props[i] = (props[i]/max)*(*PROP_POS_REFLECTOR-*PROP_NEG_REFLECTOR)+*PROP_NEG_REFLECTOR; //The prior is a bit broader than what we actually allow
                                                                                                         //This is for numerical reasons
                                                                                                
            }
        }




        Base { props }
    }


    pub fn rand_new<R: Rng + ?Sized>(rng: &mut R) -> Base {


        let mut att: [f64; BASE_L] = [0.0; BASE_L];

        for i in 0..att.len() {
            att[i] = rng.gen();
        }

        let max_ind = rng.gen_range(0..BASE_L);

        for i in 0..att.len() {
            if i == max_ind {
                att[i] = 1.0;
            } else {
                att[i] =  (att[i])*(*PROP_POS_REFLECTOR-*PROP_NEG_REFLECTOR)+*PROP_NEG_REFLECTOR;;
            }
        }

        Base { props: att}
    }

    pub fn from_bp<R: Rng + ?Sized>(best: usize, rng: &mut R) -> Base {

        Base::rand_new(rng).make_best(best)

    }

    pub fn make_best(&self, best: usize) -> Base {

        let mut base2 = self.props.clone();

        let which_b = Self::argmax(&base2);

        if best != which_b {

            let tmp = base2[which_b];
            base2[which_b] = base2[best];
            base2[best] = tmp;
        }


        Base{ props: base2 }
    }

    pub fn best_base(&self) -> usize {
        Self::argmax(&self.props)
    }

    pub fn dist(&self, base: Option<&Base>) -> f64 {

        let as_simplex: [f64; BASE_L-1] = self.as_simplex();
        match(base) {

            None => as_simplex.iter().map(|a| a.powi(2)).sum::<f64>().sqrt(),
            Some(other) => {
                let other_simplex = other.as_simplex();
                as_simplex.iter().zip(other_simplex).map(|(a, b)| (a-b).powi(2)).sum::<f64>().sqrt()
            }
        }

    }

    pub fn as_probabilities(&self) -> [f64; BASE_L] {
        let magnitude: f64 = self.props.iter().sum();
        self.props.iter().map(|a| a/magnitude).collect::<Vec<f64>>().try_into().expect("I'm never worried about error here because all Base are guarenteed to be length BASE_L")
    }

    pub fn as_simplex(&self) -> [f64; BASE_L-1] {
        let probs = self.as_probabilities();

        let simplex: [f64; BASE_L-1] = SIMPLEX_VERTICES.iter().map(|a| a.iter().zip(probs.iter()).map(|(&s, &b)| s*b).sum::<f64>())
                                                       .collect::<Vec<f64>>().try_into().unwrap();

        simplex

    }

    //This should never be publicly exposed, because it assumes an invariant that simplex_coords
    //will always be inside of the tetrahedron. Failing to uphold this invariant will produce
    //nonsense
    fn simplex_to_base(simplex_coords: &[f64; BASE_L-1]) -> Base {

        let mut mod_simplex = simplex_coords.to_vec();
        mod_simplex.push(1.0);

        let probs: [f64; BASE_L] = INVERT_SIMPLEX.iter().map(|a| a.iter().zip(mod_simplex.iter()).map(|(&b, &c)| b*c).sum::<f64>()).collect::<Vec<_>>().try_into().unwrap();

        let max = probs[Base::argmax(&probs)];

        let b_form: [f64; BASE_L]  = probs.into_iter().map(|a| a/max).collect::<Vec<_>>().try_into().unwrap();

        Base{props : b_form}

    }

    //This result is given in column major order
    fn d_base_d_simplex(&self) -> [[f64; (BASE_L-1)]; (BASE_L-1)] {

        let prob_base = self.as_probabilities();
        let best_bp = self.best_base();
        let mut bp_inds: Vec<usize> = (0..BASE_L).collect();
        bp_inds.retain(|&b| b != best_bp);
        let pmax: f64 = prob_base[best_bp];
        let pmax_sq: f64 = pmax.powi(2);
        let ps: [f64; BASE_L-1] = bp_inds.iter().map(|&a| prob_base[a]).collect::<Vec<_>>().try_into().expect("bp inds is always the same size after its defining ommission.");

        println!("pro {:?} ps {:?}", prob_base, ps);
        let mut jacobian = [[0_f64; (BASE_L-1)]; (BASE_L-1)];
        for k in 0..(BASE_L-1) {
            let mut pstar = [ps[k]; BASE_L-1];
            pstar[k] += pmax;
            
            for ti in 0..(BASE_L-1) {
                let m_inv_vec = (0..BASE_L-1).map(|m| COL_PRIMARY_INVERT_SIMPLEX[ti][bp_inds[m]]);
                println!("{:?} {:?}", pstar, m_inv_vec.clone().collect::<Vec<_>>());
                jacobian[ti][k] = m_inv_vec.zip(pstar).map(|(a, b)| a*b).sum::<f64>()/pmax_sq;
            }
        }

        jacobian

    }

    fn max( arr: &[f64]) -> f64 {
        arr.iter().fold(f64::NAN, |x, y| x.max(*y))
    }

    fn argmax( arr: &[f64] ) -> usize {

        if arr.len() == 1 {
            return 0
        }

        let mut arg = 0;

        for ind in (1..(arr.len())){
            arg = if (arr[ind] > arr[arg]) { ind } else {arg};
        }

        arg

    }

    
    pub fn add_in_hmc(&self, addend: [f64; BASE_L-1]) -> Self {

        /*
        let best = Self::argmax(&self.props);

        let mut new_props = self.props.clone();

        let mut ind: usize = 0;

        for i in 0..self.props.len() {
            if i == best {
                new_props[i] = 1.0;
            } else if addend[ind] != 0.0 {
                let mut dg =  Self::prop_to_hmc(self.props[i]);
                dg += addend[ind];
                dg = reflect(dg);
                dg = Self::hmc_to_prop(dg);
                new_props[i] = dg;
                ind += 1;
            } else {
                ind += 1;
            }

        }

        Base {props: new_props}
        */

       let tetra = self.as_simplex();
       Base::simplex_to_base(&reflect_tetra(tetra, addend))

    }

    fn prop_to_hmc(p: f64) -> f64 {
        
        let r = (p-*PROP_CUTOFF)/(*PROP_UPPER_CUTOFF-*PROP_CUTOFF);
        SPREAD_HMC_CONV*(r.ln()-(-r).ln_1p())

    }

    fn hmc_to_prop(m: f64) -> f64 {
        //This particular order of operations ensures that we map infinities correctly
        *PROP_CUTOFF+((*PROP_UPPER_CUTOFF-*PROP_CUTOFF)/(1.0+(-m/SPREAD_HMC_CONV).exp()))
    }

    fn d_prop_d_hmc(p: f64) -> f64 {

        (p-*PROP_CUTOFF)*(*PROP_UPPER_CUTOFF-p)/(SPREAD_HMC_CONV*(*PROP_UPPER_CUTOFF-*PROP_CUTOFF))
    }


    pub fn rev(&self) -> Base {

        let mut revved = self.props.clone();

        revved.reverse();

        Self::new(revved)


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

    pub fn to_unit_sum(&self) -> [f64; BASE_L] {
        let summed = self.props.iter().sum::<f64>();
        let ret: [f64; BASE_L] = self.props.iter().map(|&a| a/summed).collect::<Vec<_>>().try_into().unwrap();
        ret
    }


}

fn reflect(a: f64) -> f64 {
    
    if a.abs() < REFLECTOR {
        return a;
    } else if a.is_infinite() {
        return a.signum()*REFLECTOR;
    }
    let reflect_cond = ((a.abs()+REFLECTOR)/(2.0*REFLECTOR)).floor();
    let a_sign = (-1.0_f64).powi((reflect_cond as i32) & 0x01_i32);
    -reflect_cond*2.0*REFLECTOR*a_sign*a.signum()+a_sign*a
    
}

//Note that arrays of Copy are themselves Copy
fn reflect_tetra(start: [f64; BASE_L-1], push: [f64; BASE_L-1]) -> [f64; BASE_L-1] {
    let mut end_start = start;
    let mut end_push = Some(push);
    let mut count: usize = 0;
    while let Some(push_vec) = end_push {
        (end_start, end_push) = wall_collide(end_start, push_vec);
        count += 1;
        if count > 10 {println!("cc {} {}", count, push_vec.iter().map(|&a| a.powi(2)).sum::<f64>().sqrt());}
    }
    println!("count: {}", count);
    end_start
}

fn wall_collide(start: [f64; BASE_L-1], push: [f64; BASE_L-1]) -> ([f64; BASE_L-1], Option<[f64; BASE_L-1]>) {
    for i in 0..BASE_L {
        let dot = (VERTEX_DOT-VERTICES[i].iter().zip(start.iter()).map(|(&a, &b)| a*b).sum::<f64>())/(VERTICES[i].iter().zip(push.iter()).map(|(&a, &b)| a*b).sum::<f64>());
        if (dot >= 0.0) && (dot < 1.0) {
            return (start.iter().zip(push.iter()).map(|(&a, &b)| a+dot*b).collect::<Vec<f64>>().try_into().expect("Size is pinned down"), 
                Some(push.iter().map(|&b| 1.0-dot*b).collect::<Vec<f64>>().try_into().expect("Size is pinned down")));
        }
    }
    (start.iter().zip(push.iter()).map(|(&a, &b)| a+b).collect::<Vec<f64>>().try_into().expect("Size is pinned down"), None)
}







//BEGIN MOTIF

#[derive(Serialize, Deserialize, Clone)]
pub struct Motif {

    peak_height: f64,
    kernel: Kernel,
    pwm: Vec<Base>,

}

impl Motif {

    //GENERATORS
    //NOTE: all pwm vectors are reserved with a capacity exactly equal to MAX_BASE. This is because motifs can only change size up to that point.        
    //TODO: make sure that this is used iff there's a guarentee that a motif is allowed
    pub fn raw_pwm(mut pwm: Vec<Base>, peak_height: f64, peak_width: f64, spacer:usize) -> Motif {
        let kernel = Kernel::new(peak_width, spacer, peak_height);

        pwm.reserve_exact(MAX_BASE-pwm.len());
        let mut m = Motif {
            peak_height: peak_height,
            kernel: kernel,
            pwm: pwm,
        };

        m
    }

    pub fn rand_height_pwm<R: Rng + ?Sized>(mut pwm: Vec<Base>, peak_width: f64, spacer: usize, rng: &mut R) -> Motif {

        //let height_dist: TruncatedLogNormal = TruncatedLogNormal::new(LOG_HEIGHT_MEAN, LOG_HEIGHT_SD, MIN_HEIGHT, MAX_HEIGHT).unwrap();


        let sign: f64 = rng.gen();
        let sign: f64 = if sign < PROB_POS_PEAK {1.0} else {-1.0};

        let peak_height: f64 = sign*(*HEIGHT_DIST).sample(rng);

        Self::raw_pwm(pwm, peak_height, peak_width, spacer)
    }


    //TODO: make sure that this is used iff there's a guarentee that a motif is allowed
    pub fn from_motif<R: Rng + ?Sized>(best_bases: Vec<usize>, peak_width: f64, spacer: usize, rng: &mut R) -> Motif {
        
        let mut pwm: Vec<Base> = Vec::with_capacity(MAX_BASE);

        pwm = best_bases.iter().map(|a| Base::from_bp(*a, rng)).collect();

        //let height_dist: TruncatedLogNormal = TruncatedLogNormal::new(LOG_HEIGHT_MEAN, LOG_HEIGHT_SD, MIN_HEIGHT, MAX_HEIGHT).unwrap();


        let sign: f64 = rng.gen();
        let sign: f64 = if sign < PROB_POS_PEAK {1.0} else {-1.0};

        let peak_height: f64 = sign*(*HEIGHT_DIST).sample(rng);

        let kernel = Kernel::new(peak_width, spacer, peak_height);

        Motif {
            peak_height: peak_height,
            kernel: kernel,
            pwm: pwm,
        }


    }


    
    pub fn rand_mot<R: Rng + ?Sized>(peak_width: f64, seq: &Sequence, spacer: usize, rng: &mut R) -> Motif {


        let num_bases = rng.gen_range(MIN_BASE..(MAX_BASE+1));

        let mot = seq.random_valid_motif(num_bases);

        
        Self::from_motif(mot, peak_width, spacer, rng)
        

    }
    pub fn rand_mot_with_height<R: Rng + ?Sized>(peak_height: f64, peak_width: f64, seq: &Sequence, spacer: usize, rng: &mut R) -> Motif {


        let num_bases = rng.gen_range(MIN_BASE..(MAX_BASE+1));

        let mot = seq.random_valid_motif(num_bases);

        let mut pwm: Vec<Base> = Vec::with_capacity(MAX_BASE);

        pwm = mot.iter().map(|a| Base::from_bp(*a, rng)).collect();


        let kernel = Kernel::new(peak_width, spacer, peak_height);

        Motif {
            peak_height: peak_height,
            kernel: kernel,
            pwm: pwm,
        }

        

    }
    
    //Panics: if num_bases < MIN_BASE or num_bases > MAX_BASE
    pub fn rand_mot_with_height_and_motif_len<R: Rng + ?Sized>(peak_height: f64, num_bases: usize, peak_width: f64, seq: &Sequence, spacer: usize, rng: &mut R) -> Motif {

        assert!((num_bases >= MIN_BASE) && (num_bases <= MAX_BASE), 
                "Only motif lengths with a length between {MIN_BASE} and {MAX_BASE}, inclusive, are allowed. You tried to make a {num_bases} Base long motif.");

        let mot = seq.random_valid_motif(num_bases);

        let mut pwm: Vec<Base> = Vec::with_capacity(MAX_BASE);

        pwm = mot.iter().map(|a| Base::from_bp(*a, rng)).collect();


        let kernel = Kernel::new(peak_width,spacer, peak_height);

        Motif {
            peak_height: peak_height,
            kernel: kernel,
            pwm: pwm,
        }

        

    }


    pub fn make_opposite(&self) -> Motif {

        let mut opposite = self.clone();

        opposite.peak_height = -self.peak_height;
        opposite.kernel = &(self.kernel)*(-1.0); 
        opposite

    }

    pub fn scramble_to_close_random_valid<R: Rng + ?Sized>(&mut self, seq: &Sequence, randomizer: &mut Option<&mut R>) -> Option<usize> {

        let best_motif = self.best_motif();
        if seq.kmer_in_seq(&best_motif) {return (None);}

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

            let old_base: usize = ((old_best & (U64_BITMASK << (BITS_PER_BP*i))) >> (BITS_PER_BP*i)) as usize;
            let new_base: usize = ((new_best & (U64_BITMASK << (BITS_PER_BP*i))) >> (BITS_PER_BP*i)) as usize;

            if new_base != old_base {
                new_mot.pwm[i] = new_mot.pwm[i].make_best(new_base);
            }

        }

        new_mot

    }

    //Safety: Momentum MUST have a length equal to precisely 1+(BASE_L-1)*self.len()
    unsafe fn add_momentum(&self, eps: f64, momentum: &[f64]) -> Self {

        let mut new_mot = self.clone();
        
        //if momentum[0] != 0.0 {
            let mut h = -SPREAD_HMC_CONV*((MAX_HEIGHT-MIN_HEIGHT)/(self.peak_height.abs()-MIN_HEIGHT)-1.0).ln();
            h += (eps*momentum[0]);
            h = reflect(h);
            h = MIN_HEIGHT+((MAX_HEIGHT-MIN_HEIGHT)/(1.0+((-h/SPREAD_HMC_CONV).exp())));
            new_mot.peak_height = self.peak_height.signum()*h;

        //}
        new_mot.kernel = &(self.kernel)*(new_mot.peak_height/self.peak_height);

        for i in 0..self.len() {
            
            let slice: [f64; BASE_L-1] = (1..(BASE_L)).map(|a| *(momentum.get_unchecked(i*(BASE_L-1)+a))).collect::<Vec<_>>().try_into().unwrap();
            new_mot.pwm[i] = self.pwm[i].add_in_hmc(slice);
        }

        new_mot

    }

    //HELPERS

    pub fn pwm(&self) -> Vec<Base> {
        self.pwm.clone()
    }

    pub fn pwm_ref(&self) -> &Vec<Base> {
        &self.pwm
    }

    pub fn best_motif(&self) -> Vec<usize> {
        self.pwm.iter().map(|a| a.best_base()).collect()
    }

    pub fn rev_complement(&self) -> Vec<Base> {
        self.pwm.iter().rev().map(|a| a.rev()).collect()
    }

    pub fn peak_height(&self) -> f64 {
        self.peak_height
    }

    pub fn kernel(&self) -> Kernel {
        self.kernel.clone()
    }

    pub fn raw_kern(&self) -> &Vec<f64> {
        self.kernel.get_curve()
    }

    pub fn peak_width(&self) -> f64 {
        self.kernel.get_sd()
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

            prior += (((BASE_L-1)*self.len()) as f64)*(*PROP_UPPER_CUTOFF-*PROP_CUTOFF).ln();
            
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

            prior += (((BASE_L-1)*self.len()) as f64)*(*PROP_POS_REFLECTOR-*PROP_NEG_REFLECTOR).ln();
            
            prior

        } else {-f64::INFINITY}
    }

    pub fn height_prior(&self) -> f64 {

        let mut prior = if self.peak_height > 0.0 { PROB_POS_PEAK.ln() } else { (1.0-PROB_POS_PEAK).ln() };
        prior += (*HEIGHT_DIST).ln_pdf(self.peak_height.abs());
        prior
    }


    pub fn d_height_prior(&self) -> f64 {
        (*HEIGHT_DIST).d_ln_pdf(self.peak_height.abs())*self.peak_height.signum()
    }

    pub fn d_height_prior_d_hmc(&self) -> f64 {
        (*HEIGHT_DIST).d_ln_pdf_d_hmc(self.peak_height.abs())*self.peak_height.signum()
    }


    //BINDING FUNCTIONS

    //SAFETY: If kmer is not the same length as the pwm, or if it doesn't use
    //        all values < BASE_L, this will produce undefined behavior
    unsafe fn prop_binding(&self, kmer: &[usize]) -> (f64, bool) { 
        

        //let kmer: Vec<usize> = kmer_slice.to_vec();

        let mut bind_forward: f64 = 1.0;
        let mut bind_reverse: f64 = 1.0;

        unsafe{
        for i in 0..self.len() {
            bind_forward *= self.pwm[i].rel_bind(*kmer.get_unchecked(i));
            bind_reverse *= self.pwm[i].rel_bind(BASE_L-1-*kmer.get_unchecked(self.len()-1-i));
        }
        }

        let reverse: bool = (bind_reverse > bind_forward);

        let bind: f64 = if reverse {bind_reverse} else {bind_forward};

        return (bind, reverse)

    }

    pub fn return_bind_score(&self, seq: &Sequence) -> (Vec<f64>, Vec<bool>) {

        let coded_sequence = seq.seq_blocks();
        let block_lens = seq.block_lens(); //bp space
        let block_starts = seq.block_u8_starts(); //stored index space


        let mut bind_scores: Vec<f64> = vec![0.0; BP_PER_U8*coded_sequence.len()];
        let mut rev_comp: Vec<bool> = vec![false; BP_PER_U8*coded_sequence.len()];

        let mut uncoded_seq: Vec<usize> = vec![0; seq.max_len()];

        let seq_frame = 1+(self.len()/BP_PER_U8);


        let mut ind = 0;

        let mut store = Sequence::code_to_bases(coded_sequence[0]);


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

                //SAFETY: this is one of places where we rely on Sequence being implemented correctly 
                //        in particular, code_to_bases should never be able to return a value >= 
                //        BASE_L in its implementation.
                (bind_scores[ind], rev_comp[ind]) = unsafe {self.prop_binding(binding_borrow) };
                
            }

        }
        }



        (bind_scores, rev_comp)



    }
   
    //NOTE: this will technically mark a base as present if it's simply close enough to the beginning of the next sequence block
    //      This is technically WRONG, but it's faster and shouldn't have an effect because any positions marked incorrectly
    //      as true will have binding scores of 0
    pub fn base_check(&self, seq: &Sequence, rev_comp: &Vec<bool>, bp: usize, motif_pos: usize) -> Vec<bool> {
            
        let coded_sequence = seq.seq_blocks();

        let rev_pos = self.len()-1-motif_pos;

        let forward_bp = bp as u8;
        let comp_bp = (BASE_L-1-bp) as u8;

        let bp_to_look_for: Vec<u8> = rev_comp.iter().map(|&a| if a {comp_bp} else {forward_bp}).collect();

        let bp_lead: Vec<usize> = rev_comp.iter().enumerate().map(|(i, &r)| if r {(i+rev_pos)} else {(i+motif_pos)}).collect();

        let loc_coded_lead: Vec<usize> = bp_lead.iter().map(|b| b/4).collect();
        //This finds 3-mod_4(pos) with bit operations. Faster than (motif_pos & 3) ^ 3 for larger ints
        //Note that we DO need the inverse in Z4. The u8 bases are coded backwards, where the 4^3 place is the last base
        
        const MASKS: [u8; 4] = [0b00000011, 0b00001100, 0b00110000, 0b11000000];
        const SHIFT_BASE_BY: [u8; 4] = [0, 2, 4, 6];

        loc_coded_lead.iter().zip(bp_lead).zip(bp_to_look_for).map(|((&c, s), b)| 
                                            if (c < coded_sequence.len()) { //Have to check if leading base is over the sequence edge
                                                //This 
                                               ((coded_sequence[c] & MASKS[s-4*c]) >> SHIFT_BASE_BY[s-4*c]) == b
                                            } else { false }).collect::<Vec<bool>>()


    } 



    //NOTE: if DATA does not point to the same sequence that self does, this will break. HARD. 
    pub fn generate_waveform<'a>(&self, DATA: &'a Waveform) -> Waveform<'a> {

        let mut occupancy_trace: Waveform = DATA.derive_zero();

        let mut actual_kernel: Kernel = &self.kernel*1.0;

        let starts = DATA.seq().block_u8_starts();

        let lens = DATA.seq().block_lens();

        let (bind_score_floats, _) = self.return_bind_score(DATA.seq());

        for i in 0..starts.len() { //Iterating over each block
            for j in 0..(lens[i]-self.len()) {
                if bind_score_floats[starts[i]*BP_PER_U8+j] > THRESH {
                    actual_kernel = &self.kernel*(bind_score_floats[starts[i]*BP_PER_U8+j]) ;
                    //println!("{}, {}, {}, {}", i, j, lens[i], actual_kernel.len());
                    unsafe {occupancy_trace.place_peak(&actual_kernel, i, j+(self.len()-1)/2);} 

                    //println!("peak {} {} {}", starts[i]*BP_PER_U8+j, bind_score_floats[starts[i]*BP_PER_U8+j], occupancy_trace.read_wave()[(starts[i]*BP_PER_U8+j)/DATA.spacer()]);
                    //count+=1;
                }
            }
        }

        //println!("num peaks {}", count);

        occupancy_trace

    }

    pub fn no_height_waveform<'a>(&self, DATA: &'a Waveform) -> Waveform<'a> {

        let mut occupancy_trace: Waveform = DATA.derive_zero();

        //let base_kernel = &self.kernel*(1.0/self.peak_height);
        //let mut actual_kernel: Kernel = &base_kernel*1.0;
        let mut actual_kernel: Kernel = &self.kernel*1.0;

        let starts = DATA.seq().block_u8_starts();

        let lens = DATA.seq().block_lens();

        let (bind_score_floats, _) = self.return_bind_score(DATA.seq());

        for i in 0..starts.len() { //Iterating over each block
            for j in 0..(lens[i]-self.len()) {
                if bind_score_floats[starts[i]*BP_PER_U8+j] > THRESH {
                    //actual_kernel = &base_kernel*(bind_score_floats[starts[i]*BP_PER_U8+j]);
                    actual_kernel = &self.kernel*(bind_score_floats[starts[i]*BP_PER_U8+j]/self.peak_height);
                    unsafe {occupancy_trace.place_peak(&actual_kernel, i, j+(self.len()-1)/2);} 
                }
            }
        }

        occupancy_trace

    }
    
    pub fn only_pos_waveform<'a>(&self,bp: usize, motif_pos: usize, DATA: &'a Waveform) -> Waveform<'a> {

        let mut occupancy_trace: Waveform = DATA.derive_zero();

        let mut actual_kernel: Kernel = &self.kernel*1.0;

        let starts = DATA.seq().block_u8_starts();

        let lens = DATA.seq().block_lens();

        let (bind_score_floats, bind_score_revs) = self.return_bind_score(DATA.seq());

        let bind = unsafe{self.pwm()[motif_pos].rel_bind(bp)};

        let checked = self.base_check( DATA.seq(), &bind_score_revs, bp, motif_pos);
        for i in 0..starts.len() { //Iterating over each block
            for j in 0..(lens[i]-self.len()) {
                if checked[starts[i]*BP_PER_U8+j] && (bind_score_floats[starts[i]*BP_PER_U8+j] > THRESH) {
                    //println!("safe binding at {}", starts[i]*BP_PER_U8+j);
                    actual_kernel = &self.kernel*(bind_score_floats[starts[i]*BP_PER_U8+j]/bind);
                    unsafe {occupancy_trace.place_peak(&actual_kernel, i, j+(self.len()-1)/2)}; //Note: this technically means that we round down if the motif length is even
                                                                                         //This looks like we can violate the safety guarentee for place peak, but return_bind_score()
                                                                                         //has zeros where we can be going over the motif length. Because THRESH forbids trying
                                                                                         //to place peaks under a certain strength, this preserves our safety guarantees
                }
            }
        }

        occupancy_trace

    }

    //Safety: You MUST ensure that the binding score and reverse complement is valid for this particular motif, because you can technically use 
    //        ANY binding score here, and this code won't catch it, especially if the dimensions check out. We code this primarily for speed of calculation in the gradient calculation
    unsafe fn generate_waveform_from_binds<'a>(&self, binds: &(Vec<f64>, Vec<bool>), DATA: &'a Waveform) -> Waveform<'a> {

        let mut occupancy_trace: Waveform = DATA.derive_zero();

        let mut actual_kernel: Kernel = &self.kernel*1.0;

        let starts = DATA.seq().block_u8_starts();

        let lens = DATA.seq().block_lens();


        for i in 0..starts.len() { //Iterating over each block
            for j in 0..(lens[i]-self.len()) { //-self.len() is critical for maintaining safety of place_peak. 
                                               //It's why we don't allow sequence blocks unless they're bigger than the max motif size
                if binds.0[starts[i]*BP_PER_U8+j] > THRESH {
                    actual_kernel = &self.kernel*(binds.0[starts[i]*BP_PER_U8+j]) ;
                    //println!("{}, {}, {}, {}", i, j, lens[i], actual_kernel.len());
                    occupancy_trace.place_peak(&actual_kernel, i, j+(self.len()-1)/2);//SAFETY Note: this technically means that we round down if the motif length is even
                                                                                         //This looks like we can violate the safety guarentee for place peak, but return_bind_score()
                                                                                         //has zeros where we can be going over the motif length. Because THRESH forbids trying
                                                                                         //to place peaks under a certain strength, this preserves our safety guarantees

                    //println!("peak {} {} {}", starts[i]*BP_PER_U8+j, bind_score_floats[starts[i]*BP_PER_U8+j], occupancy_trace.read_wave()[(starts[i]*BP_PER_U8+j)/DATA.spacer()]);
                }
            }
        }

        occupancy_trace

    }
    //Safety: You MUST ensure that the binding score and reverse complement is valid for this particular motif, because you can technically use 
    //        ANY binding score here, and this code won't catch it, especially if the dimensions check out. We code this primarily for speed of calculation in the gradient calculation
    unsafe fn no_height_waveform_from_binds<'a>(&self, binds: &(Vec<f64>, Vec<bool>), DATA: &'a Waveform) -> Waveform<'a> {

        let mut occupancy_trace: Waveform = DATA.derive_zero();

        let base_kernel = &self.kernel*(1.0/self.peak_height);
        let mut actual_kernel: Kernel = &base_kernel*1.0;
        //let mut actual_kernel: Kernel = &self.kernel*1.0;

        let starts = DATA.seq().block_u8_starts();

        let lens = DATA.seq().block_lens();


        for i in 0..starts.len() { //Iterating over each block
            for j in 0..(lens[i]-self.len()) { //-self.len() is critical for maintaining safety of place_peak. 
                                               //It's why we don't allow sequence blocks unless they're bigger than the max motif size
                if  (binds.0[starts[i]*BP_PER_U8+j] > THRESH) {
                    //actual_kernel = &self.kernel*(binds.0[starts[i]*BP_PER_U8+j]/self.peak_height);
                    actual_kernel = &base_kernel*(binds.0[starts[i]*BP_PER_U8+j]);
                    occupancy_trace.place_peak(&actual_kernel, i, j+(self.len()-1)/2); 
                }
            }
        }

        occupancy_trace



    }
    //Safety: You MUST ensure that the binding score and reverse complement is valid for this particular motif, because you can technically use 
    //        ANY binding score here, and this code won't catch it, especially if the dimensions check out. We code this primarily for speed of calculation in the gradient calculation
    //        You must also ensure that bp < BASE_L, and that motif_pos < self.len()
    unsafe fn only_pos_waveform_from_binds<'a>(&self, binds: &(Vec<f64>, Vec<bool>), bp: usize, motif_pos: usize, DATA: &'a Waveform) -> Waveform<'a> {

        let mut occupancy_trace: Waveform = DATA.derive_zero();

        let mut actual_kernel: Kernel = &self.kernel*1.0;

        let starts = DATA.seq().block_u8_starts();

        let lens = DATA.seq().block_lens();

        let bind = self.pwm()[motif_pos].rel_bind(bp);    

        let checked = self.base_check( DATA.seq(), &binds.1, bp, motif_pos);
        for i in 0..starts.len() { //Iterating over each block
            for j in 0..(lens[i]-self.len()) { //-self.len() is critical for maintaining safety of place_peak. 
                                               //It's why we don't allow sequence blocks unless they're bigger than the max motif size
                if checked[starts[i]*BP_PER_U8+j] && (binds.0[starts[i]*BP_PER_U8+j] > THRESH) {
                    //println!("unsafe binding at {}", starts[i]*BP_PER_U8+j);
                    actual_kernel = &self.kernel*((binds.0[starts[i]*BP_PER_U8+j])/bind);
                    occupancy_trace.place_peak(&actual_kernel, i, j+(self.len()-1)/2); 
                }
            }
        }

        occupancy_trace



    }

    //TODO: Test the absolute crap out of this, and move the ad_grad and ad_calc calculations OUT of this into the many tf gradient calculation
    //Noise needs to be the noise from the total waveform of the motif set, not just the single motif
    pub fn single_motif_grad(&self,  DATA: &Waveform, noise: &Noise) -> (f64, Vec<f64>) {

        let binds = self.return_bind_score(DATA.seq());


        let d_ad_stat_d_noise: Vec<f64> = noise.ad_grad();

        let d_ad_like_d_ad_stat: f64 = Noise::ad_deriv(noise.ad_calc());
        

        //End preuse generation
        let d_noise_d_h = unsafe { self.no_height_waveform_from_binds(&binds, DATA)
                                        .account_auto(noise.background)};
        let d_ad_like_d_grad_form_h = (d_ad_like_d_ad_stat * ((&d_noise_d_h * &d_ad_stat_d_noise)) )
                                    * (self.peak_height().abs()-MIN_HEIGHT) * (MAX_HEIGHT - self.peak_height().abs())
                                    / (self.peak_height().signum() * SPREAD_HMC_CONV * (MAX_HEIGHT-MIN_HEIGHT)) + self.d_height_prior_d_hmc();
      

        let mut d_ad_like_d_grad_form_binds: Vec<f64> = vec![0.0; self.len()*(BASE_L-1)];


        for index in 0..d_ad_like_d_grad_form_binds.len() {

            let base_id = index/(BASE_L-1); //Remember, this is integer division, which Rust rounds down
            let mut bp = index % (BASE_L-1);
            bp += if bp >= self.pwm[base_id].best_base() {1} else {0}; //If best_base == BASE_L-1, then we access bp = 0, 1, .., BASE_L-2. 
                                                                           //At this point, base_id already goes to the next base, skipping bp = BASE_L-1
                                                                           //This is important, because statically guarentees the safety of using rel_bind

            let prop_bp = unsafe { self.pwm[base_id].rel_bind(bp) } ;

            d_ad_like_d_grad_form_binds[index] = unsafe {
                   (&(self.only_pos_waveform_from_binds(&binds, bp, base_id, DATA)
                           .account_auto(noise.background))
                  * &d_ad_stat_d_noise) * d_ad_like_d_ad_stat * Base::d_prop_d_hmc(prop_bp) 
            }
           
        }

            
        (d_ad_like_d_grad_form_h, d_ad_like_d_grad_form_binds)


    }

    //SAFETY: the d_ad_stat_d_noise must be of the same length as the noise vector we get from DATA.
    pub unsafe fn parallel_single_motif_grad(&self,  DATA: &Waveform, d_ad_stat_d_noise: &Vec<f64>, d_ad_like_d_ad_stat: f64, background: &Background) -> Vec<f64> {

        let binds = self.return_bind_score(DATA.seq());

        let n = 1+self.len()*(BASE_L-1);


        let mut d_ad_like_d_grad_form: Vec<f64> = (0..n).into_par_iter().map(|i| {
            if i == 0 {
                let d_noise_d_h = self.no_height_waveform_from_binds(&binds, DATA)
                                               .account_auto(background);
                (d_ad_like_d_ad_stat * ((&d_noise_d_h * d_ad_stat_d_noise)))
                * (self.peak_height().abs()-MIN_HEIGHT) * (MAX_HEIGHT - self.peak_height().abs())
                / (self.peak_height().signum() * SPREAD_HMC_CONV  * (MAX_HEIGHT-MIN_HEIGHT)) +self.d_height_prior_d_hmc()
            } else {
                let index = i-1;
                let base_id = index/(BASE_L-1); //Remember, this is integer division, which Rust rounds down
                let mut bp = index % (BASE_L-1);
                bp += if bp >= self.pwm[base_id].best_base() {1} else {0}; //If best_base == BASE_L-1, then we access bp = 0, 1, .., BASE_L-2. 
                                                                           //At this point, base_id already goes to the next base, skipping bp = BASE_L-1
                                                                           //This is important, because statically guarentees the safety of using rel_bind

                let prop_bp = self.pwm[base_id].rel_bind(bp) ;

                let result =
                      (&(self.only_pos_waveform_from_binds(&binds, bp, base_id, DATA)
                               .account_auto(background))
                      * d_ad_stat_d_noise) * d_ad_like_d_ad_stat
                      ;

                result
            }
        }).collect();

        //let grad_raw_ptrs = (0..self.len()).map(|i| d_ad_like_d_grad_form.as_mut_ptr().add(i*(BASE_L-1)+1)).collect::<Vec<_>>();

        //(0..self.len()).into_par_iter().map(|i| {
        let _ = d_ad_like_d_grad_form.par_rchunks_exact_mut(BASE_L-1).rev().enumerate().map(|(i, grad_chunk)| {
            //SAFETY: the proper construction of this index construction guarentees the safety of both copies and edits later
            let index_into: Vec<usize> = (0..(BASE_L-1)).collect();//(0..(BASE_L-1)).map(|j| 1+i*(BASE_L-1)+j).collect::<Vec<usize>>();
            
            //let mut_gradient_ptr = d_ad_like_d_grad_form as *mut f64;  //We need to be EXTREMELY careful. The variable index_into is covering our behinds against UB
                                                                       //We technically have multiple mutable pointers pointing at the beginning of our vector, but
                                                                       //it's fine. We only ever read or write to data that this one slice is using.
                                                                       //SAFETY RELIES ON INDEX_INTO BEING IMPLEMENTED CORRECTLY. IF ANY TWO INDEX_INTO's OVERLAP, 
                                                                       //IT IS AN IMMEDIATE DATA RACE AND THEREFORE UNDEFINED BEHAVIOR.
            let best_bp = self.pwm[i].best_base();
            let mut bp_inds: Vec<usize> = (0..BASE_L).collect();
            bp_inds.retain(|&b| b != best_bp);
            
            let prob_base = self.pwm[i].as_probabilities();
            let pmax: f64 = prob_base[best_bp];
            let pmax_sq: f64 = pmax.powi(2);
            let ps: [f64; BASE_L-1] = bp_inds.iter().map(|&a| prob_base[a]).collect::<Vec<_>>().try_into().expect("bp inds is always the same size after its defining ommission.");
 
            let ln_like_grads: [f64; BASE_L-1] = index_into.iter().map(|&k| *grad_chunk.get_unchecked(k)).collect::<Vec<_>>().try_into().unwrap();

            let mut like_grad_times_p_matrix = [0.0_f64; BASE_L-1];

            let simplex_grad = self.pwm[i].d_base_d_simplex();
            for k in 0..(BASE_L-1) {

                *grad_chunk.get_unchecked_mut(index_into[k]) = ln_like_grads.iter().zip(simplex_grad[k].iter()).map(|(&a, &b)| a*b).sum::<f64>();
                println!("grad {:?} lnn {:?}", grad_chunk, ln_like_grads); 
            }
           
        }).collect::<Vec<_>>();



            
        d_ad_like_d_grad_form

    }

/*        peak_height: f64,
    kernel: Kernel,
    pwm: Vec<Base>,
*/

/*
    fn col_unit_sum_pwm(&self) -> Vec<[f64; BASE_L]> {
        self.pwm.iter().map(|a| a.to_unit_sum()).collect::<Vec<_>>()
    }
*/

    pub fn distance_function(&self, other_mot: &Motif) -> (f64, isize, bool) {
        let rev = other_mot.rev_complement();
        let best_forward = ((-((self.len()-1) as isize))..((other_mot.len()-1) as isize)).map(|a| (self.little_distance(&other_mot.pwm, a), a, false))
                                                                   .min_by(|x, y| x.0.partial_cmp(&y.0).expect("No NANs should be possible")).unwrap();
        let best_reverse = ((-((self.len()-1) as isize))..((other_mot.len()-1) as isize)).map(|a| (self.little_distance(&rev, a), a, true))
                                                                   .min_by(|x, y| x.0.partial_cmp(&y.0).expect("No NANs should be possible")).unwrap();

        let best = if best_reverse.0 < best_forward.0 { best_reverse } else {best_forward};

        best

    }

    fn little_distance(&self, other_mot: &Vec<Base>, alignment: isize) -> f64 {

        let (mod_len_a, mod_len_b) = if alignment < 0 {(0_usize, (-alignment) as usize)} else {(alignment as usize, 0_usize)};

        let total_len = (self.len()+mod_len_a).max(other_mot.len()+mod_len_b);

        let (pwm_1, pwm_2) = (&self.pwm, other_mot);

        let mut distance: f64 = 0.0;

        for ind in 0..total_len {

            let b1 = if (ind < mod_len_a || ind >= (pwm_1.len()+mod_len_a)) {None} else {Some(&pwm_1[ind-mod_len_a])};
            let b2 = if (ind < mod_len_b || ind >= (pwm_2.len()+mod_len_b)) {None} else {Some(&pwm_2[ind-mod_len_b])};

            distance += match b1 {
                Some(b) => b.dist(b2),
                None => match b2{
                    Some(bb) => bb.dist(None),
                    None => { warn!("PWM alignment in distance is causing a complete miss!"); 0.0},
                },
            };

        }


        distance

    }

    
}

impl fmt::Display for Motif { 
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const DIGITS: usize = 10;
        
        //Writing the peak height (unitless) and the peak width (as defined by # base pairs from one end to another)
        write!(f, "Peak height: {:.DIGITS$}. Total peak width (bp): {}\n", self.peak_height, self.kernel.len());
       
        //I want people to be free to define their own bases, if they like
        //But they will have to change it in the code for formatting: it's not my problem
        //The two extra spaces on either side let the output align in a pretty way
        for b in BPS { write!(f, "  {:<DIGITS$}  ", b); }
        write!(f, "\n");

        for i in 0..self.pwm.len() {

            let base = self.pwm[i].show();
            let begin: bool = (i == 0);
            let end: bool = (i == self.pwm.len()-1);
            let j = (begin, end);

            match j {
                (true, true) => panic!("Something pathological is going on!"),
                (true, false) => write!(f, "[{base:.DIGITS$?},\n", base = base, DIGITS = DIGITS),
                (false, false) => write!(f, " {base:.DIGITS$?},\n", base = base, DIGITS = DIGITS),
                (false, true) => write!(f, " {base:.DIGITS$?}]\n", base = base, DIGITS = DIGITS),
            };
        }
        Ok(())
    }
}

impl fmt::Debug for Motif { 
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const DIGITS: usize = 10;
        
        //Writing the peak height (unitless) and the peak width (as defined by # base pairs from one end to another)
        write!(f, "Peak height: {:.DIGITS$}. Total peak width (bp): {}\n", self.peak_height, self.kernel.len());
       
        //I want people to be free to define their own bases, if they like
        //But they will have to change it in the code for formatting: it's not my problem
        //The two extra spaces on either side let the output align in a pretty way
        for b in BPS { write!(f, "  {:<DIGITS$}  ", b); }
        write!(f, "\n");

        for i in 0..self.pwm.len() {

            let base = self.pwm[i].show();
            let begin: bool = (i == 0);
            let end: bool = (i == self.pwm.len()-1);
            let j = (begin, end);

            match j {
                (true, true) => panic!("Something pathological is going on!"),
                (true, false) => write!(f, "[{base:.DIGITS$?},\n", base = base, DIGITS = DIGITS),
                (false, false) => write!(f, " {base:.DIGITS$?},\n", base = base, DIGITS = DIGITS),
                (false, true) => write!(f, " {base:.DIGITS$?}]\n", base = base, DIGITS = DIGITS),
            };
        }
        write!(f, "\n");
        Ok(())
    }
}

#[derive(Clone)]
//DEFINITELY CANNOT BE DIRECTLY SERIALIZED
//
//SAFETY: any method which adds a Motif to the set directly, without checking
//        that the kernel length is less than the length of data in all sequence
//        blocks, is IMMEDIATELY unsafe. I do not include such methods here, 
//        but if YOU do, you MUST enforce this invariant.
pub struct MotifSet<'a> {

    set: Vec<Motif>, 
    width: f64, 
    signal: Waveform<'a>,
    ln_post: Option<f64>,
    data: &'a Waveform<'a>, 
    background: &'a Background,
}

impl<'a> MotifSet<'a> {
   
    pub fn rand_with_one<R: Rng + ?Sized>(data: &'a Waveform<'a>, background: &'a Background, fragment_length: usize, rng: &mut R) -> Self {

        let width = (fragment_length as f64)/6.0;

        let set = vec![Motif::rand_mot(width, data.seq(), data.spacer(), rng)];

        let signal = set[0].generate_waveform(data);

        let mut mot_set = MotifSet{ set: set, width: width, signal: signal, ln_post: None, data: data, background: background};

        let _ = mot_set.ln_posterior();

        mot_set
    }

    pub fn rand_with_one_height<R: Rng + ?Sized>(peak_height: f64, data: &'a Waveform<'a>, background: &'a Background, fragment_length: usize, rng: &mut R) -> Self {

        let width = (fragment_length as f64)/6.0;

        let set = vec![Motif::rand_mot_with_height(peak_height, width, data.seq(), data.spacer(), rng)];

        let signal = set[0].generate_waveform(data);

        let mut mot_set = MotifSet{ set: set, width: width, signal: signal, ln_post: None, data: data, background: background};

        let _ = mot_set.ln_posterior();

        mot_set
    }
    
    pub fn rand_with_one_height_and_motif_len<R: Rng + ?Sized>(peak_height: f64, motif_len: usize, data: &'a Waveform<'a>, background: &'a Background, fragment_length: usize, rng: &mut R) -> Self {

        let width = (fragment_length as f64)/6.0;

        let set = vec![Motif::rand_mot_with_height_and_motif_len(peak_height, motif_len, width, data.seq(),data.spacer(), rng)];

        let signal = set[0].generate_waveform(data);

        let mut mot_set = MotifSet{ set: set, width: width, signal: signal, ln_post: None, data: data, background: background};

        let _ = mot_set.ln_posterior();

        mot_set
    }

    fn recalced_signal(&self) -> Waveform {

        let mut signal = self.data.derive_zero();

        for mot in self.set.iter() {
            signal += &(mot.generate_waveform(self.data));
        }

        signal

    }

    fn recalc_signal(&mut self) {
        self.signal = self.data.derive_zero();
        for mot in self.set.iter() {
            let w = mot.generate_waveform(self.data);
            self.signal += &(mot.generate_waveform(self.data));
        }
    }

    fn accept_test<R: Rng + ?Sized>(old: f64, new: f64, rng: &mut R) -> bool {

        let diff_ln_like = new-old;

        //Always accept if new likelihood is better
        if (diff_ln_like > 0.0) {
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
            width: self.width, 
            signal: self.signal.clone(),
            ln_post: None,
            data: self.data, //pointer
            background: self.background, //pointer
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
    //and p = 1-NECESSARY_MOTIF_IMPROVEMENT.exp().
    //NOTE: the ommission of ln(p) term is deliberate. It amounts to a normalization constant
    //for the motif set prior, and would obfuscate the true point of this prior
    pub fn motif_num_prior(&self) -> f64 {
        -((self.set.len()-1) as f64)*NECESSARY_MOTIF_IMPROVEMENT
    }

    pub fn ln_prior(&self) -> f64 {
        self.motif_num_prior() + self.set.iter().map(|a| a.height_prior()+a.pwm_prior(self.data.seq())).sum::<f64>()
    }

    pub fn ln_likelihood(&self) -> f64 {
        Noise::ad_like(((self.signal).produce_noise(self.data, self.background).ad_calc()))
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

    fn add_motif(&mut self, new_mot: Motif) -> f64 {

        self.signal += &new_mot.generate_waveform(self.data) ;
        self.set.push(new_mot);
        self.ln_post = None;
        self.ln_posterior()

    }
    
    fn insert_motif(&mut self, new_mot: Motif, position: usize) -> f64 {

        self.signal += &new_mot.generate_waveform(self.data) ;
        self.set.insert(position, new_mot);
        self.ln_post = None;
        self.ln_posterior()

    }

    fn remove_motif(&mut self, rem_id: usize) -> f64{
        assert!(rem_id < self.set.len());

        let rem_mot = self.set.swap_remove(rem_id);
        self.signal -= &rem_mot.generate_waveform(self.data);
        self.ln_post = None;
        self.ln_posterior()

    }

    fn remove_motif_void(&mut self, rem_id: usize) {

        assert!(rem_id < self.set.len());
        let rem_mot = self.set.swap_remove(rem_id);
        self.signal -= &rem_mot.generate_waveform(self.data);
        self.ln_post = None;

    }

    fn replace_motif(&mut self, new_mot: Motif, rem_id: usize) -> f64 {
        let rem_mot = self.set[rem_id].clone();
        self.signal -= &rem_mot.generate_waveform(self.data);
        self.signal += &new_mot.generate_waveform(self.data) ;
        self.set[rem_id] = new_mot;
        self.ln_post = None;
        self.ln_posterior()
    }

    //This proposes a new motif for the next motif set, but does not do any testing vis a vis whether such a move will be _accepted_
    fn propose_new_motif<R: Rng + ?Sized>(&self, rng: &mut R ) -> Option<(Self, f64)> {
        let mut new_set = self.derive_set();
        let new_mot = Motif::rand_mot(self.width, self.data.seq(), self.data.spacer(), rng); //rand_mot always generates a possible motif
        let ln_gen_prob = new_mot.height_prior()+new_mot.pwm_gen_prob(self.data.seq());
        let ln_post = new_set.add_motif(new_mot);
        Some((new_set, ln_post-ln_gen_prob)) //Birth moves subtract the probability of their generation

    }

    //This proposes removing an old motif for the next motif set
    //It does no testing save for checking if removing a motif would produce a possible set
    fn propose_kill_motif<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<(Self, f64)> {

        if (self.set.len() <= 1) { //We never want to propose a set with no motifs, ever
            None
        } else {
            let mut new_set = self.derive_set();
            let rem_id = rng.gen_range(0..self.set.len());
            let ln_gen_prob = self.set[rem_id].height_prior()+self.set[rem_id].pwm_gen_prob(self.data.seq());
            let ln_post = new_set.remove_motif(rem_id);
            Some((new_set, ln_post+ln_gen_prob)) //Death moves add the probability of the generation of their deleted variable(s)
        }
    }


    //I'm only extending motifs on one end
    //This saves a bit of time from not having to reshuffle motif vectors
    fn propose_extend_motif<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<(Self, f64)> {
        let mut new_set = self.derive_set();
       
        let extend_id = rng.gen_range(0..self.set.len());

        if (self.set[extend_id].len() >= MAX_BASE) { //We want to immediately reject any move which extends a motif beyond maximum length
            None
        } else {

            let mut new_mot = self.set[extend_id].clone();
            new_mot.pwm.push(Base::rand_new(rng));
            //let ln_gen_prob = new_mot.height_prior()+new_mot.pwm_prior(self.data.seq());
            if self.data.seq().kmer_in_seq(&new_mot.best_motif()) { //When we extend a motif, its best base sequence may no longer be in the sequence
                let ln_post = new_set.replace_motif(new_mot, extend_id);
                //There are BASE_L ways that we sample BASE_L-1 times from a U(*PROP_POS_REFLECTOR,*PROP_NEG_REFLECTOR) distribution
                //So, the probability density is (1./(1./(*PROP_POS_REFLECTOR,*PROP_NEG_REFLECTOR))^(BASE_L-1))/BASE_L
                let base_ln_density = ((BASE_L-1) as f64)*(*PROP_POS_REFLECTOR-*PROP_NEG_REFLECTOR).ln()-(BASE_L as f64).ln(); 
                Some((new_set, ln_post-base_ln_density)) //Birth moves subtract the probability of their generation
            } else {
                None
            }
        }
    }

    //I'm only extending motifs on one end
    //This saves a bit of time from not having to reshuffle motif vectors
    fn propose_contract_motif<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<(Self, f64)> {
        let mut new_set = self.derive_set();
       
        let contract_id = rng.gen_range(0..self.set.len());

        if (self.set[contract_id].len() <= MIN_BASE) { //We want to immediately reject any move which contracts a motif below minimum length
            None
        } else {
            let mut new_mot = self.set[contract_id].clone();
            let old_base = new_mot.pwm.pop();
            let ln_post = new_set.replace_motif(new_mot, contract_id);
            let base_ln_density = ((BASE_L-1) as f64)*(*PROP_POS_REFLECTOR-*PROP_NEG_REFLECTOR).ln()-(BASE_L as f64).ln(); 
            Some((new_set, ln_post+base_ln_density)) //Birth moves subtract the probability of their generation
        }
    }



    //MOVE TO CALL
    //For borrow checker reasons, this will only work if the motif calling already has a generated likelihood
    //And I can't have it do it here
    //If you use this on a motif without such a likelihood, it will panic
    pub fn run_rj_move<R: Rng + ?Sized>(&self, rng: &mut R) -> (Self, usize, bool) {

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

            None => (self.clone(), which_rj, false),
            Some((new_mot, modded_ln_like)) => {

                if Self::accept_test(self.ln_post.unwrap(), modded_ln_like, rng) {
                    (new_mot, which_rj, true)
                } else { 
                    (self.clone(), which_rj, false)
                }
            }
        }

    }


    //MOVE TO CALL
    pub fn base_leap<R: Rng + ?Sized>(&self, rng: &mut R) -> Self {


       //We want to shuffle this randomly, in case there is some kind of codependency between particular TFs
       let mut ids: Vec<usize> = (0..self.set.len()).collect();
       ids.shuffle(rng);

       let mut current_set = self.clone();

       for id in ids {

           let current_mot = current_set.set[id].clone();

           let mut base_set = current_set.clone();
           base_set.remove_motif_void(id);
           //This was numerically derived, and not a hard rule. I wanted less than 150 kmers per leap
           let threshold = if current_mot.len() < 12 {2} else { (current_mot.len()+1)/2-4}; 

           let kmer_ids = self.data.seq().all_kmers_within_hamming(&current_mot.best_motif(), threshold);

           let ids_cartesian_bools = kmer_ids.into_iter().flat_map(|k| [(k, true), (k, false)]).collect::<Vec<_>>();

           let likes_and_mots: Vec<(f64, Self)> = ids_cartesian_bools.clone().into_par_iter().map(|a| {
               let mut to_add = base_set.clone();
               let add_mot = current_mot.scramble_by_id_to_valid(a.0, a.1, self.data.seq());
               let lnlike = to_add.insert_motif(add_mot, id);
               (lnlike, to_add)
           }).collect();

           //We want to pick these based on their relative ln posteriors
           //But these are going to be small. We normalize based on the max
           //ln likelihood because it prevents errors from infinities
           
           let mut selection_probs: Vec<f64> = vec![0.0; likes_and_mots.len()];

           let normalize_ln_like: f64 = likes_and_mots.iter().map(|(a, _)| a).fold(-f64::INFINITY, |a, &b| a.max(b)) ;

           let mut sum_probs: f64 = 0.0;

           for i in 0..selection_probs.len() {
               //This subtraction might seem technically unnecessary, but
               //but computers are not infinitely precise. We want to 
               //ensure that we minimize numerical issues

               /*println!("like {} {}", i, likes_and_mots[i].0);
               if likes_and_mots[i].0.is_nan() {
                   for mot in likes_and_mots[i].1.set.iter() {
                       println!("{}", mot);
                   }

                   println!("{:?}", likes_and_mots[i].1);
               }*/
               selection_probs[i] = (likes_and_mots[i].0-normalize_ln_like).exp().abs();
               sum_probs+=selection_probs[i];
           }

           /*if selection_probs.len() == 0 {
               panic!("No states being selected from!");
           }*/

           //println!("{:?}", selection_probs);

           let dist = WeightedIndex::new(&selection_probs).unwrap();
           current_set = likes_and_mots[dist.sample(rng)].1.clone();
       }

       current_set

   }


   fn gradient(&self) -> Vec<f64> {

       let noise = self.signal.produce_noise(self.data, self.background);
       let d_ad_stat_d_noise = noise.ad_grad();
       let d_ad_like_d_ad_stat = Noise::ad_deriv(noise.ad_calc());
       let mut len_grad: usize = self.set.len();

       for i in 0..self.set.len() {
           len_grad += self.set[i].len()*(BASE_L-1);
       }

       let mut gradient = vec![0.0_f64; len_grad];

       let mut finished_compute: usize = 0;

       for i in 0..self.set.len() {
           
           let motif = &self.set[i];
           let compute_to = finished_compute+(motif.len() * (BASE_L-1) +1);
           let motif_grad = &mut gradient[finished_compute..compute_to];
           //SAFETY: we know that we derived our noise from the same data waveform that we used for d_ad_stat_d_noise 
           let grad_vec = unsafe { motif.parallel_single_motif_grad(self.data, &d_ad_stat_d_noise, d_ad_like_d_ad_stat, self.background)};
       
           for i in 0..motif_grad.len() {
               motif_grad[i] = grad_vec[i];
           }
          
           finished_compute = compute_to;
       }

       gradient

   }
    
   //MOVE TO CALL 
   pub fn hmc<R: Rng + ?Sized>(&self, rng: &mut R) ->  (Self, bool, f64) {
       
       
       let total_len = self.set.len() + (0..self.set.len()).map(|i| self.set[i].len()*(BASE_L-1)).sum::<usize>();

       let momentum: Vec<f64> = (0..total_len).map(|_| MOMENTUM_DIST.sample(rng)).collect();

       let mut final_trace: Vec<Self> = Vec::with_capacity(HMC_TRACE_STEPS);

       let mut prior_set = self.clone();
      
       let mut gradient_old = prior_set.gradient();
       let mut momentum_apply = momentum.clone();

       for _ in 0..HMC_TRACE_STEPS {
       
           let mut new_set = self.clone();
           new_set.ln_post = None;
       
       
           for i in 0..momentum_apply.len(){
               //We calculate the gradient of the ln posterior with gradient
               //BUT, our potential energy is -ln_posterior. 
               //Because we subtract the gradient of the potential energy
               //we ADD the gradient of the ln_posterior
               momentum_apply[i] += (HMC_EPSILON*gradient_old[i])/2.0;
           }


           let mut start = 0;
           let mut next_start = 0;
           for k in 0..self.set.len() {
               next_start += self.set[k].len()*(BASE_L-1)+1;
               new_set.set[k] = unsafe{ prior_set.set[k].add_momentum(HMC_EPSILON, &momentum_apply[start..next_start])};
               start = next_start;
           }

           new_set.recalc_signal();

           let gradient_new = new_set.gradient();

           for i in 0..momentum_apply.len() {
               momentum_apply[i] += (HMC_EPSILON*(gradient_new[i])/2.0);
           }

           //We want gradient_old to take ownership of the gradient_new values, and gradient_old's prior values to be released
           //Same with prior_set and new_set
           gradient_old = gradient_new;
           prior_set = new_set;

       }

       let mut delta_kinetic_energy: f64 = 0.0;

       for i in 0..momentum.len() {
           delta_kinetic_energy += (momentum_apply[i].powi(2) - momentum[i].powi(2))/2.0;
       }

       let delta_potential_energy = prior_set.ln_posterior()-self.ln_post.unwrap();

       if Self::accept_test(0.0, delta_kinetic_energy+delta_potential_energy, rng){
           (prior_set, true, delta_potential_energy+delta_kinetic_energy)
       } else {
           (self.clone(), false,delta_potential_energy+delta_kinetic_energy)
       }


       
   }


    #[cfg(test)]
    fn numerical_gradient(&self) -> Vec<f64> {
        let h: f64 = 1e-6;

        let num_motifs = self.set.len();

        let curr_post = self.calc_ln_post();
        let curr_like = self.ln_likelihood();

        let gradient: Vec<f64> = (self.set).iter().enumerate().map(|(k,a)| {
 
            let len_gradient_form = 1+a.len()*(BASE_L-1);

            let motif_grad: Vec<f64> = (0..len_gradient_form).into_par_iter().map(|i| {

                let mut perturb_vec = vec![0.0_f64; len_gradient_form];
               
                perturb_vec[i] = h;
                let mod_mot = unsafe {a.add_momentum(1.0, perturb_vec.as_slice())};

                let mut alter_set = self.clone();
                let new_ln_post = alter_set.replace_motif(mod_mot,k);
                let new_ln_like = new_ln_post-alter_set.ln_prior();
                //println!("mod i: {} \n {:?}", i, alter_set);
                (new_ln_post-curr_post)/h
                //(new_ln_like-curr_like)/h
            }).collect::<Vec<f64>>();

            motif_grad
            
        }).flatten().collect();

        gradient

    }

    #[cfg(test)]
    fn no_par_gradient(&self) -> Vec<f64> {

       let noise = self.signal.produce_noise(self.data, self.background);

       let mut len_grad: usize = self.set.len();

       for i in 0..self.set.len() {
           len_grad += self.set[i].len()*(BASE_L-1);
       }

       let mut gradient = vec![0.0_f64; len_grad];

       let mut finished_compute: usize = 0;

       for motif in &self.set {
      
           let compute_to = finished_compute+(motif.len() * (BASE_L-1) +1);
           //println!("inds a {} {}", finished_compute, compute_to);
           let motif_grad = &mut gradient[finished_compute..compute_to];
           //SAFETY: we know that we derived our noise from the same data waveform that we used for d_ad_stat_d_noise 
           let (g, mut grad_vec) = motif.single_motif_grad(self.data, &noise);
       
           grad_vec.insert(0, g);
           //println!("{} {} lens", motif_grad.len(), grad_vec.len());
           //println!("{:?}", grad_vec);
           for i in 0..motif_grad.len() {
               motif_grad[i] = grad_vec[i];
           }
          
           finished_compute = compute_to;
       }

       gradient

   }





}


impl Debug for MotifSet<'_> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, "width:{}, ln_post: {:?}, background: {:?} \n set: {:#?}", self.width, self.ln_post, self.background, self.set)
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct StrippedMotifSet {
    set: Vec<Motif>,
    width: f64,
    ln_post: f64,
}

impl StrippedMotifSet {

    //Yes, self, not &self. I'm destroying the stripped motif set whenever I make the active motif set
    pub fn reactivate_set<'a>(&self, data: &'a Waveform, background: &'a Background) -> MotifSet<'a> {

        let mut revived = MotifSet {

            set: self.set.clone(),
            width: self.width,
            signal: data.derive_zero(),
            ln_post: None,
            data: data,
            background: background,
        };

        _ = revived.ln_posterior(); //I don't believe these ln posteriors if I'm going to be using them for inference. 
                                    //I only have this struct save a ln posterior because I want to record it for post processing

        revived
    }
    
    pub fn motif_num_prior(&self) -> f64 {
        -((self.set.len()-1) as f64)*NECESSARY_MOTIF_IMPROVEMENT
    }

    pub fn ln_prior(&self, seq: &Sequence) -> f64 {
        self.motif_num_prior() + self.set.iter().map(|a| a.height_prior()+a.pwm_prior(seq)).sum::<f64>()
    }

}

enum AnyMotifSet<'a> {
    Active(MotifSet<'a>),
    Passive(StrippedMotifSet),
}

impl<'a> AnyMotifSet<'a> {
 
    fn give_activated(&self, data: &'a Waveform, background: &'a Background) -> MotifSet<'a> {
        match self {
            AnyMotifSet::Passive(set) => set.reactivate_set(data, background),
            AnyMotifSet::Active(ready) => ready.clone(),
        }
    }

    fn give_stored(&self) -> StrippedMotifSet {

        match self {
            AnyMotifSet::Passive(set) => set.clone(),
            AnyMotifSet::Active(ready) => StrippedMotifSet::from(ready),
        }

    }

    fn activate(&mut self, data: &'a Waveform, background: &'a Background){
        *self = AnyMotifSet::Active(self.give_activated(data, background));                                
    }

    fn store(&mut self) {

        if let AnyMotifSet::Active(set) = self {
            *self = AnyMotifSet::Passive(StrippedMotifSet::from(&*set));
        }
                                         

    }

}

impl<'a> From<&'a MotifSet<'a>> for StrippedMotifSet {
    fn from(other: &'a MotifSet) -> Self {
        Self {
            set: other.set.clone(),
            width: other.width,
            ln_post: other.calc_ln_post(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct MotifSetDef {

    set: Vec<Motif>,
    width: f64,
    signal: WaveformDef,
    ln_post: f64,

}

impl<'a> From<&'a MotifSet<'a>> for MotifSetDef {
    fn from(other: &'a MotifSet) -> Self {

        let signal = WaveformDef::from(&other.signal);

        Self {
            set: other.set.clone(), 
            width: other.width,
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
    pub fn get_motif_set<'a, R: Rng + ?Sized>(self, recalculate: bool, validate_motifs: bool, randomizer: &mut Option<&mut R>, data: &'a Waveform, background: &'a Background ) -> MotifSet<'a> {
        
        let mut set = self.set.clone();

        const WIDTH_TO_SD: f64 = 6.0;

        //We fix up any kernels that don't match the width defined in our MotifSet
        let mut changed_kernels = false;
        let mut scrambled_motifs = false;
        for mot in set.iter_mut() {
            let same_sd = (mot.kernel.get_sd() - self.width).abs() < 1e-6;
            let same_len = (WIDE*mot.kernel.get_sd() - (mot.kernel.len() as f64)).abs() < 1e-6;

            if !(same_sd && same_len) {
                changed_kernels = true;
                mot.kernel = Kernel::new(self.width, data.spacer(), mot.kernel.get_height());
            }

            if validate_motifs && mot.pwm_prior(data.seq()) == f64::NEG_INFINITY {
                scrambled_motifs = true;
                _ = mot.scramble_to_close_random_valid(data.seq(), randomizer);
            }
 
        }

        //We check to see if we need to recalculate the wave signal.
        let signal_will_make_ub =  (self.signal.len() != data.read_wave().len()) || (self.signal.spacer() != data.spacer());
        let redo_signal = recalculate || changed_kernels || scrambled_motifs || signal_will_make_ub;

        let signal = if !redo_signal { //Notice how we carefully checked to make sure that signal follows our safety guarentees
            unsafe {self.signal.get_waveform(data.point_lens(), data.start_dats(), data.seq())}
        } else {
            data.derive_zero() //This isn't our final signal. This just ensures our safety guarentees
        };
        
        let mut pre_set = MotifSet {
            set: set,
            width: self.width,
            signal: signal,
            ln_post: None,
            data: data, 
            background: background,
        };

        if !redo_signal {
            pre_set.ln_post = Some(self.ln_post);
        } else {
            if !recalculate {
                warn!("Had to adjust your motif set and recalculate its occupancy trace because:");
                if changed_kernels { warn!("-Some of your Gaussian kernels are incompatible with your reported experimental fragment width and have been rederived.");}
                if scrambled_motifs { warn!("-Some of your motifs' best binding kmers were unrepresented in your sequence, and you wanted them validated. They've been scrambled to have a best kmer that is compatible with your sequence while remaining as close to the unrepresented kmers as possible.");}
                if signal_will_make_ub { warn!("The occupancy trace you reported for your motif set is incompatible with your ChIP data. We calculated the signal to prevent segfaults.");}
            }

            pre_set.recalc_signal();
            _ = pre_set.ln_posterior();
        }

        pre_set


    }

}






//THIS SHOULD BE (DE)SERIALIZABLE WITH A CUSTOM IMPLEMENTATION
//SINCE ALL MOTIFS IN THE SET AND THE WAVEFORM SHOULD POINT TO seq
//AND THE MOTIF_SET ITSELF SHOULD ALSO POINT TO data AND background
pub struct SetTrace<'a> {
    trace: Vec<AnyMotifSet<'a>>,
    capacity: usize,
    AllDataFile: String,
    data: &'a Waveform<'a>, 
    background: &'a Background,
}


//TODO: 
    //Create an "advance" function which does the appropriate number of HMCs, base leaps, and RJ steps
    //Create an "output and reset" type of function which outputs all motifs but the final one to a file and starts us to just that motif
    //Create an initialize function that reads in the prepared sequence, waveform, and background distribution from some pre initialized source
    //          and allows us to either start from a prior SetTrace file, from a MEME motif file, or from completely anew

impl<'a> SetTrace<'a> {

    //All three of these references should be effectively static. They won't be ACTUALLY, because they're going to depend on user input, but still
    pub fn new_empty(capacity: usize, AllDataFile: String, data: &'a Waveform<'a>, background: &'a Background) -> SetTrace<'a> {


        SetTrace{
            trace: Vec::<AnyMotifSet<'a>>::with_capacity(capacity),
            capacity: capacity, 
            AllDataFile: AllDataFile,
            data: data, 
            background: background,
        }

    }

    //Panics: if self.trace is empty
    //Suggested defaults: 1.0, 1 or 2, 2^(some negative integer power between 5 and 10), 5-20, 80. But fiddle with this for acceptance rates
    //You should aim for your HMC to accept maybe 80-90% of the time. Changing RJ acceptances is hard, but you should hope for like 20%ish
    pub fn advance<R: Rng + ?Sized>(&mut self, rng: &mut R) -> (usize,bool) { //1 for each of the RJ moves, plus one for hmc
        

        let setty = self.current_set();

        let select_move: usize = rng.gen_range(0..=MAX_IND_HMC);

        const range_rj: usize = MAX_IND_RJ+1;
        const range_leap: usize = MAX_IND_LEAP+1;

        let (new_set, retval): (MotifSet, (usize,bool)) = match select_move {

            0..=MAX_IND_RJ => {
                let (set, move_type, accept) = setty.run_rj_move(rng);
                (set, (move_type, accept))
            },
            range_rj..=MAX_IND_LEAP => {
                let set = setty.base_leap(rng);
                (set, (4, true))
            },
            range_leap..=MAX_IND_HMC => {
                let (set, accept, _) = setty.hmc(rng);
                (set, (5, accept))
            },
            _ => unreachable!(),
        };
       
        self.quiet_last_set();

        self.trace.push(AnyMotifSet::Active(new_set));

        retval

    }

    //WARNING: Be aware that this motif set WILL be coerced to follow the seq and data of the motif set
    //         The ln posterior will also be recalculated if the motif set isn't correctly pointed
    //         To ensure the safety of this function, there is a recalculation step that occurs if the 
    //         sequence or data changes. I assure you, you do not want this recalculation to occur:
    //         It's going to be very slow
    pub fn push_set(&mut self, set: MotifSet<'a>) {


        let mut repoint_set = set.clone();

        let mut recalc_ln_post = (!std::ptr::eq((self.data), repoint_set.data)) || 
                                 (!std::ptr::eq((self.background), repoint_set.background)) ;
        if recalc_ln_post {
            repoint_set.data = &self.data;
            repoint_set.background = &self.background;
        }

        if recalc_ln_post {
            repoint_set.signal = self.data.derive_zero();
            for i in 0..repoint_set.set.len() {
                repoint_set.signal += &repoint_set.set[i].generate_waveform(&self.data);
            }
            repoint_set.ln_post = None;
            repoint_set.ln_posterior();
        }

        self.trace.push(AnyMotifSet::Active(repoint_set));

    }


    pub fn push_last_state_from_json<R: Rng + ?Sized>(&mut self, always_recalculate: bool, validate_motif: bool, validate_randomizer: &mut Option<&mut R>,json_file: &str) {

        let json_string: String = fs::read_to_string(json_file).expect("Json file MUST be valid!");

        let prior_state: MotifSetDef = serde_json::from_str(&json_string).expect("Json file MUST be a valid motif set!");

        self.push_set_def(always_recalculate, validate_motif, validate_randomizer, prior_state);

    }

    //Note: if the likelihoods are calculated off of a different sequence/data, this WILL 
    //      just give you a wrong answer that seems to work
    pub fn push_set_def<R: Rng + ?Sized>(&mut self, always_recalculate: bool,validate_motif: bool, validate_randomizer: &mut Option<&mut R>, set: MotifSetDef) {
        self.trace.push(AnyMotifSet::Active(set.get_motif_set(always_recalculate, validate_motif, validate_randomizer,&self.data, &self.background)));
    }

    //SPEED: You REALLY don't want to have to be in a situation where always_recalculate should be set to true, 
    //       or the signals can't be reconciled with the data. It will take FOREVER
    fn push_set_def_many<R: Rng + ?Sized>(&mut self, sets: Vec<AnyMotifSet<'a>>) {
        for set in sets {
            self.trace.push(set);
        }
    }


    pub fn current_set(&self) -> MotifSet<'a> {
        self.trace[self.trace.len()-1].give_activated(self.data, self.background)
    }

    pub fn quiet_last_set(&mut self) {
        
        if let Some(set_mut_ref) = self.trace.last_mut() {
            set_mut_ref.store();
        }

    }

    pub fn save_initial_state(&self, output_dir: &str, run_name: &str) {

        let savestate_file: String = output_dir.to_owned()+"/"+run_name+"_savestate.json";
        
        match &self.trace[0] {
        
            AnyMotifSet::Active(set) => {
                let init_set: MotifSetDef = MotifSetDef::from(set);
                fs::write(savestate_file.as_str(), serde_json::to_string(&init_set).unwrap());
            },
            AnyMotifSet::Passive(set) => {fs::write(savestate_file.as_str(), serde_json::to_string(&set).unwrap());},
        };
    }
       
    pub fn save_and_drop_history(&mut self, output_dir: &str, run_name: &str, zeroth_step: usize) {

        let len_trace = self.trace.len();

        //We want to keep the last element in the SetTrace, so that the markov chain can proceed
        let trace = self.trace.drain(0..(len_trace-1)).map(|a| a.give_stored()).collect();


        let history = SetTraceDef {
            AllDataFile: self.AllDataFile.clone(),
            trace: trace, 
        };

        let trace_file: String = format!("{}/{}_trace_from_step_{:0>7}.json",output_dir,run_name,zeroth_step);

        fs::write(trace_file.as_str(), serde_json::to_string(&history).unwrap());


        self.save_initial_state(output_dir, run_name);
    }


    /*

   pub struct MotifSet<'a> {

    set: Vec<Motif>,
    width: f64,
    signal: Waveform<'a>,
    ln_post: Option<f64>,
    data: &'a Waveform<'a>,
    background: &'a Background,
}

       */

    pub fn trace_from_meme<R: Rng + ?Sized>(&mut self, meme_file_name: &str, seq: &Sequence, e_value_cutoff: f64, fragment_length: usize, rng: &mut R) {

        let width: f64 = (fragment_length as f64)/6.0;

        let meme_file_string = fs::read_to_string(meme_file_name).expect("Invalid FASTA file name!");
        let mut meme_as_vec = meme_file_string.split("\n").collect::<Vec<_>>();

        let re = Regex::new(r"letter-probability matrix: alength= (\d+)  w= (\d+) nsites= \d+ E= (\d+\.\de[-+]\d\d\d)").unwrap();

        let start_matrix_lines: Vec<usize> = meme_as_vec.iter().enumerate().filter(|&(_,a)| re.is_match(a)).map(|(n, _)| n).collect();

        if start_matrix_lines.len() == 0 {
            panic!("Cannot read this file: it is invalid MEME output");
        }

        let mut set: Vec<Motif> = Vec::new();

        for (mot_num, line) in start_matrix_lines.iter().enumerate() {
            let captures = re.captures(meme_as_vec[*line]).unwrap();
            let alphabet_len: usize = captures[1].parse().unwrap();

            assert!(alphabet_len == BASE_L, "MEME alphabet must be the same length as supplied alphabet");

            let motif_len: usize = captures[2].parse().unwrap();
            assert!((motif_len >= MIN_BASE) && (motif_len <= MAX_BASE), "Motif length must be compatible with supplied minimum and maximum");


            let e_value: f64 = captures[3].parse().unwrap();

            if (set.len() > 0) && (e_value > e_value_cutoff) {
                break;
            }
            
            let mut base_vec: Vec<Base> = Vec::with_capacity(MAX_BASE);

            for i in 1..(motif_len+1) {
                let mut props: [ f64; BASE_L] = [0.0; BASE_L];
                let mut line_split = meme_as_vec[*line+i].split_whitespace();
                for j in 0..BASE_L {
                    props[j] = line_split.next().expect(format!("MEME file doesn't deliver on alphabet length on line {}", *line+i+1).as_str())
                               .parse().expect(format!("MEME file doesn't deliver on bases being floats on line {}", *line+i+1).as_str());
                }
            
                base_vec.push(Base::new(props));
            }

            let mut motif = Motif::rand_height_pwm(base_vec, width,self.data.spacer(), rng);
           
            let poss_hamming = motif.scramble_to_close_random_valid(seq, &mut Some(rng));

            match poss_hamming {
                Some(hamming) => warn!("{}", format!("Motif number {} from the MEME file does not exist in the parts of the sequence with peaks! Moving it to a valid motif within a Hamming distance of {}!", mot_num, hamming)),
                None => (),
            };


            set.push(motif);    

        }


        let mut signal = self.data.derive_zero();

        for mot in &set {
            signal += &(mot.generate_waveform(self.data));
        }

        let mut full_set = MotifSet {
            set: set,
            width: width, 
            signal: signal, 
            ln_post: None, 
            data: self.data, 
            background: self.background
        };

        let _ = full_set.ln_posterior();

        self.trace.push(AnyMotifSet::Active(full_set));


    }
    
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SetTraceDef {

    trace: Vec<StrippedMotifSet>,
    AllDataFile: String,

}




impl SetTraceDef {

    //Panics: all MotifSetDefs in the trace must be of the same dimension and spacing as the data.
    //        We also check to make sure that the first set's posterior density is accurate
    pub fn get_set_trace<'a, R: Rng + ?Sized>(self, data: &'a Waveform, background: &'a Background) -> SetTrace<'a> {
    
        //We only validate the ln likelihood for the active motif
        let mut trace = self.trace.into_iter().map(|a| AnyMotifSet::Passive(a)).collect::<Vec<_>>();


        if let Some(last_state) = trace.last_mut() {
            last_state.activate(data, background)
        }

        SetTrace {
            AllDataFile: self.AllDataFile.clone(),
            capacity: trace.len(),
            trace: trace,
            data: data,
            background: background,
        }


    }

    pub fn len(&self) -> usize {
        self.trace.len()
    }

    pub fn ln_posterior_trace(&self) -> Vec<f64> {
        self.trace.iter().map(|a| a.ln_post).collect::<Vec<f64>>()
    }

    pub fn ln_likelihood_trace(&self) -> Result<Vec<f64>, Box<dyn Error>> {
        
        let check_data = fs::read_to_string(self.AllDataFile.as_str())?;

        let full_data: All_Data = serde_json::from_str(check_data.as_str())?;

        Ok(self.trace.iter().map(|a| a.ln_post-a.ln_prior(full_data.seq())).collect::<Vec<f64>>())

    }

    pub fn motif_num_trace(&self) -> Vec<f64> {
        self.trace.iter().map(|a| a.set.len() as f64).collect::<Vec<_>>()
    }


    //PWMs are chosen by making a random choice of SET, and a random choice of ONE motif per set
    pub fn ret_rand_motifs<R: Rng + ?Sized>(&self, num_motifs: usize, rng: &mut R) -> Vec<Motif> {
 
        let set_picks: Vec<&StrippedMotifSet> = self.trace.iter().choose_multiple(rng, num_motifs);

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
 
    pub fn extract_best_motif_per_set(&self, reference_motif: &Motif, tail_start: usize, cutoff: f64) -> Vec<(Motif, (f64, isize, bool))> {

        self.trace[(self.len()-tail_start)..self.len()].iter().map(|mot_set| {

           mot_set.set.iter().map(|mot| (mot.clone(),mot.distance_function(&reference_motif)))
                             .min_by(|a,b| a.1.0.partial_cmp(&b.1.0).expect("No NANs should be present in distances"))
                             .expect("Motif sets all need at least one motif")


       }).filter(|(a, (b, c, d))| *b < cutoff).collect::<Vec<_>>()

    }

    pub fn append(&mut self, mut attachment: SetTraceDef) {
        self.trace.append(&mut attachment.trace);
    }

}




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
   
    fn d_ln_pdf_d_hmc(&self, x: f64) -> f64 {

        if x < self.min || x > self.max {
            unreachable!("The height translation from hmc height seems to be going awry");
        } else {
            self.d_ln_pdf(x)*(x-self.min)*(self.max-x)/(SPREAD_HMC_CONV*(self.max-self.min))
        }

    }
}

impl ::rand::distributions::Distribution<f64> for TruncatedLogNormal {
    
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
       
        let norm: Normal = Normal::new(self.location, self.scale).unwrap();
        let mut sample: f64 = norm.sample(rng).exp();
        let mut invalid: bool = ((sample > self.max) | (sample < self.min));
        while invalid {
            sample = norm.sample(rng).exp();
            invalid = ((sample > self.max) | (sample < self.min));
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

    use std::time::{Duration, Instant};
    /*use crate::base::bases::Base;
    use crate::base::bases::GBase;
    use crate::base::bases::TruncatedLogNormal;
    use crate::base::bases::{Motif, THRESH};*/
    use super::*;
    use crate::sequence::{Sequence, BP_PER_U8};
    use statrs::distribution::{Continuous, ContinuousCDF, LogNormal, Normal};
    use statrs::statistics::{Min, Max};
    use statrs::function::gamma;
    use rand::Rng;
    use std::ptr;
    use std::collections::VecDeque;
    use crate::waveform::*;
    use rand::distributions::{Distribution, Uniform};


    fn produce_bps_and_pos(seq_motif: &Vec<usize>) -> (Vec<usize>, Vec<usize>) {

        let mut bps: Vec<usize> = Vec::with_capacity(seq_motif.len()*(BASE_L-1));
        let mut pos: Vec<usize> = Vec::with_capacity(seq_motif.len()*(BASE_L-1));
        
        for i in 0..seq_motif.len() {
            let mut b = 0;
            for j in 0..(BASE_L-1) {
                if seq_motif[i] == b { b+=1;}
                bps.push(b);
                pos.push(i);
                b+=1;
            }
        }
        (bps, pos)
    }

    #[test]
    fn simplex_test() {
        let mut rng = rand::thread_rng();
        let b = Base::rand_new(&mut rng);

        let simplex = b.as_simplex();
        
        let mod_b = Base::simplex_to_base(&simplex);

        println!("{:?} {:?} {:?} {} {} {}", b, simplex, mod_b, b.dist(None), b.dist(Some(&b)), simplex.iter().map(|&a| a.powi(2)).sum::<f64>().sqrt());

        let jacob = b.d_base_d_simplex();

        let b0 = b.add_in_hmc([1e-6, 0.0, 0.0]);
        let b1 =  b.add_in_hmc([0.0, 1e-6, 0.0]);
        let b2 =  b.add_in_hmc([0.0, 0.0, 1e-6]);

        println!("{:?} {:?} {} {} {}", b.as_simplex(), b0.as_simplex(), b0.as_simplex()[0]-b.as_simplex()[0],  b0.as_simplex()[1]-b.as_simplex()[1],  b0.as_simplex()[2]-b.as_simplex()[2]);
        println!("{:?} {:?} {} {} {}", b.as_simplex(), b1.as_simplex(), b1.as_simplex()[0]-b.as_simplex()[0],  b1.as_simplex()[1]-b.as_simplex()[1],  b1.as_simplex()[2]-b.as_simplex()[2]);
        println!("{:?} {:?} {} {} {}", b.as_simplex(), b2.as_simplex(), b2.as_simplex()[0]-b.as_simplex()[0],  b2.as_simplex()[1]-b.as_simplex()[1],  b2.as_simplex()[2]-b.as_simplex()[2]);

        println!("{:?}", jacob);
        println!("{:?} {:?} {:?}", b0.props.iter().zip(b.props.iter()).map(|(&a, &b)| (a-b)/1e-6).collect::<Vec<_>>(), b1.props.iter().zip(b.props.iter()).map(|(&a, &b)| (a-b)/1e-6).collect::<Vec<_>>(),
        b2.props.iter().zip(b.props.iter()).map(|(&a, &b)| (a-b)/1e-6).collect::<Vec<_>>());

    }

    #[test]
    fn gradient_test() {


        let mut rng = rand::thread_rng(); //fastrand::Rng::new();

        let block_n: usize = 200;
        let u8_per_block: usize = 100;
        let bp_per_block: usize = u8_per_block*4;
        let bp: usize = block_n*bp_per_block;
        let u8_count: usize = u8_per_block*block_n;

        println!("begin grad set gen");
        let start_gen = Instant::now();
        //let blocks: Vec<u8> = (0..u8_count).map(|_| rng.u8(..)).collect();
        //let preblocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.u8(..)).collect();
        let preblocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.gen::<u8>()).collect();
        let blocks: Vec<u8> = preblocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>(); 
        let block_u8_starts: Vec<usize> = (0..block_n).map(|a| a*u8_per_block).collect();
        let block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
        let mut start_bases: Vec<usize> = (0..block_n).map(|a| a*bp_per_block).collect();
        let sequence: Sequence = Sequence::new_manual(blocks, block_lens.clone());
        let wave: Waveform = Waveform::create_zero(&sequence, 5);
        let duration = start_gen.elapsed();
        println!("grad gen {} bp {:?}", bp, duration);

        let corrs: Vec<f64> = vec![0.9, -0.1];
        let background = Background::new(0.25, 2.64, &corrs);
        let mut motif_set = MotifSet::rand_with_one_height(13.2, &wave, &background, 350, &mut rng);

        _ = motif_set.add_motif(Motif::rand_mot_with_height(13.2,motif_set.width, wave.seq(), wave.spacer(), &mut rng));
     

        println!("CV");
        let analytical_grad = motif_set.gradient();
        println!("AG");
        let numerical_grad = motif_set.numerical_gradient();
        println!("NG");

        let b: Base = Base::new([0.1, 0.2, 0.3, 0.4]);

        let b2 = b.add_in_hmc([0.0; 3]);

        println!("2");
        let b3 = b.add_in_hmc([0.0001, 0.0, 0.0]);

        println!("b {:?} b2 {:?} b3 {:?}", b, b2, b3);

        let mot = motif_set.get_nth_motif(0);
        let (bps, pos) = produce_bps_and_pos(&(mot.best_motif()));
        let mot_a = motif_set.get_nth_motif(1);
        let (bps_a, pos_a) = produce_bps_and_pos(&(mot_a.best_motif()));
        let bind1 = mot.return_bind_score(&wave.seq());
        let wave1 = unsafe{mot.generate_waveform_from_binds(&bind1, &wave)};
        let mom = vec![0.0_f64; 1+(mot.len()*(BASE_L-1))];

        let h = 1e-6;

        println!("Mot \n {}", mot);
        println!("Mot_a \n {}", mot_a);
        for (j, (&bp, &p)) in bps.iter().zip(pos.iter()).enumerate() {
        
            let wave_check = unsafe{mot.only_pos_waveform_from_binds( &bind1, bp, p, &wave)};
            
            let n1 = wave_check.produce_noise(&wave, &background);
            let mut add_vec = wave_check.derive_zero().raw_wave();

            add_vec[2] = h;
            let dwave = Waveform::new(add_vec, &sequence, 5);

            let n2 = (&wave_check+&dwave).produce_noise(&wave, &background);

            

            let resid_care: Vec<f64> = n1.resids()[0..3].iter().zip(n2.resids()[0..3].iter()).map(|(&a, &b)| (b-a)/h).collect();
            let prop_resid = dwave.account_auto(&background);

            println!("Resid change: {:?} {:?}", resid_care, prop_resid.resids()[0..3].iter().map(|&a| a/h).collect::<Vec<_>>());
            let mut mommy = mom.clone();
            mommy[j+1] = h;
        
            let mot2 = unsafe{mot.add_momentum(1.0, &mommy)};
       
            let wave2 = mot2.generate_waveform(&wave);

            let prop_bp = unsafe{mot.pwm()[p].rel_bind(bp)};
            let n_diff: Vec<f64> = wave2.produce_noise(&wave, &background).resids().iter()
                              .zip(wave1.produce_noise(&wave, &background).resids().iter())
                              .map(|(&a, &b)| (a-b)/h).collect();
                     
            //let deriv_n = ((&wave_check).account_auto(&background));

            //let ad_deriv = (wave2.produce_noise(&wave, &background).ad_calc()-wave1.produce_noise(&wave, &background).ad_calc())/h;
            //let ana_deriv = (&deriv_n*&(wave1.produce_noise(&wave, &background).ad_grad()))*Base::d_prop_d_hmc(prop_bp);

            //println!("ad {} ana {} diff {}", ad_deriv, ana_deriv, ad_deriv-ana_deriv);
            /*for i in 0..n_diff.len() {
                println!("{} {} {} {}", i, n_diff[i], deriv_n[i], deriv_n[i]/n_diff[i]);
            }*/
            let w1 = wave1.raw_wave();
            let w2 = wave2.raw_wave();
            let wc = wave_check.raw_wave();

            let prop_bp2 = unsafe{mot2.pwm()[p].rel_bind(bp)};
            let normalizer = Base::d_prop_d_hmc(prop_bp); //(*PROP_UPPER_CUTOFF-prop_bp) * (prop_bp-*PROP_CUTOFF)/(*PROP_UPPER_CUTOFF-*PROP_CUTOFF);
            let mut i = 0;
            let mut calc: f64 = (w2[i]-w1[i])/h;
            let mut ana: f64 = wc[i];
            println!("Wave checking {} in pos {}", BPS[bp], p);
            println!("bp0: {}, bp1: {}, dbp: {}, analytical dbp: {}", prop_bp, prop_bp2, (prop_bp2-prop_bp)/h, normalizer);
            let ratio = ((prop_bp2-prop_bp)/h)/normalizer;
            println!("{} {} {} {} diffs", unsafe{mot2.pwm()[p].rel_bind(0)-mot.pwm()[p].rel_bind(0)}, unsafe{mot2.pwm()[p].rel_bind(1)-mot.pwm()[p].rel_bind(1)},
                                          unsafe{mot2.pwm()[p].rel_bind(2)-mot.pwm()[p].rel_bind(2)}, unsafe{mot2.pwm()[p].rel_bind(3)-mot.pwm()[p].rel_bind(3)});       
            while (i < (w2.len()-2)) && ((calc == 0.) || (ana == 0.)) {
                calc = (w2[i]-w1[i])/h;
                ana = wc[i];
                i += 1;
            }
            
            let calc_b = (w2[i+1]-w1[i+1])/h;
            let ana_b = wc[i+1];


            let num_diff = calc/normalizer;
            let num_diff_b = calc_b/normalizer;
            println!("{} {} {} {} {} {} {} {}", i,  num_diff, ana, (num_diff/ana), (num_diff_b/ana_b), normalizer, (calc/ana),((calc/ana)/normalizer));

        }

        


        let mut grad_reses: Vec<Result<(), String>> = Vec::with_capacity(analytical_grad.len());
        println!("{} {} {} {}", analytical_grad.len(), mot.len()*(BASE_L-1)+1, mot_a.len()*(BASE_L-1)+1, mot.len()*(BASE_L-1)+1+mot_a.len()*(BASE_L-1)+1);
        println!("Analytical    Numerical    Difference(abs)    Quotient    Prior par");
        for i in 0..analytical_grad.len() {
            if i == 0 || i == ((BASE_L-1)*mot.len()+1) {
                println!("height!");
                let mot_ref = if (i == 0) { &mot } else {&mot_a};
                println!("{} {} {} {} {}",i, analytical_grad[i], numerical_grad[i], numerical_grad[i]-analytical_grad[i], mot_ref.d_height_prior_d_hmc());
            } else if i < ((BASE_L-1)*mot.len()+1) {
                println!("{} {} {} {} {}",i,  analytical_grad[i], numerical_grad[i], numerical_grad[i]-analytical_grad[i], ((numerical_grad[i]-analytical_grad[i])/numerical_grad[i]));
            } else {
                println!("{} {}", pos_a[i-((BASE_L-1)*mot.len()+2)], bps_a[i-((BASE_L-1)*mot.len()+2)]);
                println!("{} {} {} {} {}",i, analytical_grad[i], numerical_grad[i], numerical_grad[i]-analytical_grad[i], ((numerical_grad[i]-analytical_grad[i])/numerical_grad[i]));
            }
            //let success = if (numerical_grad[i].abs() != 0.) {(((numerical_grad[i]-analytical_grad[i])/numerical_grad[i]).abs() < 1e-2)} else {(numerical_grad[i]-analytical_grad[i]).abs() < 1e-3};
            let success = ((numerical_grad[i]-analytical_grad[i]).abs() < 5e-2);
            if success {
                grad_reses.push(Ok(()));
            } else {
                let zeroth_mot = i <= mot.len()*(BASE_L-1);
                let mot_num = if zeroth_mot {0} else {1};
                let ind = if zeroth_mot {i} else {i - ((mot.len()*(BASE_L-1))+1)};
                
                let loc_message = if ind == 0 { "height".to_owned() } else {
                    let bp = if zeroth_mot {bps[ind-1]} else {bps_a[ind-1]};
                    let po = if zeroth_mot {pos[ind-1]} else {pos_a[ind-1]};
                    format!("binding at position {} of base {}", po, BPS[bp])};

                grad_reses.push(Err(format!("In motif {}, the {} yields a bad gradient. Absolute difference {}, relative difference {}", mot_num, loc_message, numerical_grad[i]-analytical_grad[i], (numerical_grad[i]-analytical_grad[i])/numerical_grad[i])));
            }
        }
 
        let error_messages: Vec<_> = grad_reses.iter().filter(|a| a.is_err()).map(|a| a.clone().err().unwrap()).collect();

        if error_messages.len() > 0 {
            for em in error_messages { println!("{}", em);}
            panic!();
        }

        let mom_dist = Normal::new(0.0, 10.0).unwrap();
        let mut rng = rand::thread_rng();
        let eps = 1e-6;
        let (new_set, acc, dham) = motif_set.hmc(&mut rng);

        let should_wave = new_set.recalced_signal();

        let diff_wave_check = (&should_wave-&new_set.signal).raw_wave().iter().map(|&a| a.powi(2)).sum::<f64>().sqrt();
        println!("diff_wave_check {}", diff_wave_check);
        assert!(diff_wave_check < 1e-6, "diff wave fail");

        println!("I'm not setting a firm unit test here. Instead, the test should be that as epsilon approaches 0, D hamiltonian does as well");
        println!("Epsilon {} D hamiltonian {} acc {} \n old_set: {:?} \n new_set: {:?}", eps, dham, acc, motif_set,new_set);

    }
    #[test]
    fn leap_test() {


        let mut rng = rand::thread_rng(); //fastrand::Rng::new();

        let block_n: usize = 200;
        let u8_per_block: usize = 435;
        let bp_per_block: usize = u8_per_block*4;
        let bp: usize = block_n*bp_per_block;
        let u8_count: usize = u8_per_block*block_n;

        println!("begin grad set gen");
        let start_gen = Instant::now();
        //let blocks: Vec<u8> = (0..u8_count).map(|_| rng.u8(..)).collect();
        //let preblocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.u8(..)).collect();
        let preblocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.gen::<u8>()).collect();
        let blocks: Vec<u8> = preblocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>(); 
        let block_u8_starts: Vec<usize> = (0..block_n).map(|a| a*u8_per_block).collect();
        let block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
        let mut start_bases: Vec<usize> = (0..block_n).map(|a| a*bp_per_block).collect();
        let sequence: Sequence = Sequence::new_manual(blocks, block_lens.clone());
        let wave: Waveform = Waveform::create_zero(&sequence, 5);
        let duration = start_gen.elapsed();
        println!("grad gen {} bp {:?}", bp, duration);

        let corrs: Vec<f64> = vec![0.9, -0.1];
        let background = Background::new(0.25, 2.64, &corrs);
        let mut motif_set = MotifSet::rand_with_one_height(-9.6, &wave, &background, 350, &mut rng);

        _ = motif_set.add_motif(Motif::rand_mot_with_height(13.2,motif_set.width, wave.seq(), wave.spacer(), &mut rng));
     
        let mot = motif_set.get_nth_motif(0);
        let mot1 = motif_set.get_nth_motif(1);
        let mot_kmer = mot.best_motif();
        let mot1_kmer = mot1.best_motif();
        let id: usize = 4;
        let mot_scram = mot.scramble_by_id_to_valid(id, true, &sequence);
        let scram_kmer = unsafe{ Sequence::u64_to_kmer(sequence.idth_unique_kmer(mot.len(),id), mot.len())};
       
        println!("old {} \n new {}", mot, mot_scram);

        let mut all_base_correct = true;
        let mut all_scramble_correct = true;
        for base in 0..mot.len() {
            all_base_correct &= (unsafe{mot_scram.pwm[base].rel_bind(scram_kmer[base])} == 1.0);
            let best_old = mot_kmer[base];
            let best_new = scram_kmer[base];
            for bp in 0..BASE_L {

                if bp == best_old {
                    all_scramble_correct &= (unsafe{mot_scram.pwm[base].rel_bind(bp) == mot.pwm[base].rel_bind(best_new)});
                } else if bp == best_new {
                    all_scramble_correct &= (unsafe{mot_scram.pwm[base].rel_bind(bp) == mot.pwm[base].rel_bind(best_old)});
                } else {
                    all_scramble_correct &= (unsafe{mot_scram.pwm[base].rel_bind(bp) == mot.pwm[base].rel_bind(bp)});
                }
                    
            }

        }

        assert!(mot_scram.peak_height() == -mot.peak_height(), "peak height doesn't flip when it should");
        assert!(all_base_correct, "Not all the new swapped bases are what they should be");
        assert!(all_scramble_correct, "Not all of the swaps preserve values unless they're being swapped and also swap correctly.");

        let mot_scram = mot.scramble_by_id_to_valid(id, false, &sequence);
        assert!(mot_scram.peak_height() == mot.peak_height(), "peak height isn't preserved when it should be");

        let leaped = motif_set.base_leap(&mut rng);

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
            for bp in 0..BASE_L {
                if bp == best_old {
                    all_scramble_correct &= (unsafe{leap.pwm[base].rel_bind(bp) == mot.pwm[base].rel_bind(best_new)});
                } else if bp == best_new {
                    all_scramble_correct &= (unsafe{leap.pwm[base].rel_bind(bp) == mot.pwm[base].rel_bind(best_old)});
                } else {
                    all_scramble_correct &= (unsafe{leap.pwm[base].rel_bind(bp) == mot.pwm[base].rel_bind(bp)});
                }
                    
            }

        }

        assert!(all_scramble_correct, "zeroth motif not correctly leaped");

        let mut all_scramble_correct = true;
        for base in 0..mot1.len() {
            let best_old = mot1_kmer[base];
            let best_new = leap1_kmer[base];
            for bp in 0..BASE_L {
                if bp == best_old {
                    all_scramble_correct &= (unsafe{leap1.pwm[base].rel_bind(bp) == mot1.pwm[base].rel_bind(best_new)});
                } else if bp == best_new {
                    all_scramble_correct &= (unsafe{leap1.pwm[base].rel_bind(bp) == mot1.pwm[base].rel_bind(best_old)});
                } else {
                    all_scramble_correct &= (unsafe{leap1.pwm[base].rel_bind(bp) == mot1.pwm[base].rel_bind(bp)});
                }
                    
            }

        }

        assert!(all_scramble_correct, "first motif not correctly leaped");

        let mut alt_leaped = leaped.clone();
        alt_leaped.ln_post = None;
        let ln_post = alt_leaped.ln_posterior();

        println!("diff ln_post {}", ln_post-leaped.ln_post.unwrap());
      
        assert!((ln_post-leaped.ln_post.unwrap()).abs() <= 64.0*std::f64::EPSILON, "ln posteriors not lining up"); 

        let recalced_signal = leaped.recalced_signal();

        let sig_diff = &leaped.signal-&recalced_signal;

        let any_diffs: Vec<f64> = sig_diff.raw_wave().iter().filter(|a| a.abs() > 64.0*std::f64::EPSILON).map(|&a| a).collect();

        println!("any diff {:?}", any_diffs);
        
        assert!(any_diffs.len() == 0, "waves not lining up"); 
    }

    #[test]
    fn rj_manipulator_tests() {


        let mut rng = rand::thread_rng(); //fastrand::Rng::new();

        let block_n: usize = 200;
        let u8_per_block: usize = 200;
        let bp_per_block: usize = u8_per_block*4;
        let bp: usize = block_n*bp_per_block;
        let u8_count: usize = u8_per_block*block_n;

        println!("begin grad set gen");
        let start_gen = Instant::now();
        //let blocks: Vec<u8> = (0..u8_count).map(|_| rng.u8(..)).collect();
        //let preblocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.u8(..)).collect();
        let preblocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.gen::<u8>()).collect();
        let blocks: Vec<u8> = preblocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>(); 
        let block_u8_starts: Vec<usize> = (0..block_n).map(|a| a*u8_per_block).collect();
        let block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
        let mut start_bases: Vec<usize> = (0..block_n).map(|a| a*bp_per_block).collect();
        let sequence: Sequence = Sequence::new_manual(blocks, block_lens.clone());
        let wave: Waveform = Waveform::create_zero(&sequence, 5);
        let duration = start_gen.elapsed();
        println!("grad gen {} bp {:?}", bp, duration);

        let corrs: Vec<f64> = vec![0.9, -0.1];
        let background = Background::new(0.25, 2.64, &corrs);
        let mut motif_set = MotifSet::rand_with_one_height(-9.6, &wave, &background, 350, &mut rng);

        let check_set = motif_set.clone();

        let mid_mot = (MIN_BASE+MAX_BASE)/2;
        //TESTING THE MANIPULATOR FUNCTIONS: these should simply mutate the motif set to conform and usually output the new ln posterior
        let add_mot = Motif::rand_mot_with_height_and_motif_len(13.2,mid_mot,motif_set.width, wave.seq(), wave.spacer(), &mut rng);
        
        //Testing: fn add_motif(&mut self, new_mot: Motif) -> f64 
        let new_like = motif_set.add_motif(add_mot.clone());
        
        //I'm comfortable enforcing exactly equality because new_like should literally be a copy of ln_post
        assert!(motif_set.ln_post == Some(new_like));

        let wave_diff = &motif_set.signal-&check_set.signal;

        let check_diff = &wave_diff-&add_mot.generate_waveform(motif_set.data);

        let all_good = check_diff.read_wave().iter().map(|&a| a.abs() < 1024.0*std::f64::EPSILON).fold(true, |f, x| f && x);

        assert!(all_good);

        assert!(motif_set.set.len() == 2);

        println!("add motif correct wave and like");

        //Testing fn remove_motif(&mut self, rem_id: usize) -> f64
        let new_like = motif_set.remove_motif(1);
        
        assert!((motif_set.calc_ln_post()- check_set.calc_ln_post()).abs() < 1e-8, "{}", format!("{:?} {:?}", motif_set.ln_post, check_set.ln_post));

        let wave_diff = &motif_set.signal-&check_set.signal;

        //let check_diff = &wave_diff-&add_mot.generate_waveform(motif_set.data);

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

        let check_diff = &wave_diff-&add_mot.generate_waveform(motif_set.data);

        let all_good = check_diff.read_wave().iter().map(|&a| a.abs() < 1024.0*std::f64::EPSILON).fold(true, |f, x| f && x);

        assert!(all_good);

        //I'm comfortable enforcing exactly equality because this should be an exact copy
        assert!(motif_set.set[0].peak_height() == add_mot.peak_height());


        assert!(motif_set.set.len() == 2);

        //Testing fn replace_motif(&mut self, new_mot: Motif, rem_id: usize) -> f64

        let add_mot2 = Motif::rand_mot_with_height(-6.2,motif_set.width, wave.seq(), wave.spacer(), &mut rng);
        
        let new_like = motif_set.replace_motif(add_mot2.clone(), 0);

        assert!(motif_set.ln_post == Some(new_like));

        let bring_wave_back = &motif_set.signal -&add_mot2.generate_waveform(motif_set.data);

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


        //propose_new_motif<R: Rng + ?Sized>(&self, rng: &mut R ) -> Option<(Self, f64)>

        let birthed = motif_set.propose_new_motif(&mut rng);

        assert!(birthed.is_some());//proposing a new motif should NEVER create an impossible motif

        let (birth_mot, ln_prop) = birthed.expect("Birth should never produce an impossible set!");
        assert!(birth_mot.set.len() == (motif_set.set.len()+1));

        let l = birth_mot.set.len()-1;
        for i in 0..motif_set.set.len() {
            assert!(motif_set.set[i].peak_height() == birth_mot.set[i].peak_height());
        }

        let should_prior = ln_prop-(birth_mot.calc_ln_post());

        let actual_prior = birth_mot.set[l].height_prior()+birth_mot.set[l].pwm_gen_prob(&sequence);

        assert!((should_prior+actual_prior).abs() < 1e-6, "{}", format!("{}", should_prior+actual_prior).as_str());
       
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

        let actual_prior = motif_set.set[l].height_prior()+motif_set.set[l].pwm_gen_prob(&sequence);

        assert!((should_prior-actual_prior).abs() < 1e-6, "{}", format!("{}", should_prior-actual_prior).as_str());

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

        let actual_prior = ((BASE_L-1) as f64)*(*PROP_POS_REFLECTOR-*PROP_NEG_REFLECTOR).ln()-(BASE_L as f64).ln();

        assert!((should_prior+actual_prior).abs() < 1e-6, "{}", format!("{}", should_prior+actual_prior).as_str());
 
        let cant_extend_set = MotifSet::rand_with_one_height_and_motif_len(-9.6, MAX_BASE, &wave, &background, 350, &mut rng);

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

        let actual_prior = ((BASE_L-1) as f64)*(*PROP_POS_REFLECTOR-*PROP_NEG_REFLECTOR).ln()-(BASE_L as f64).ln();

        assert!((should_prior-actual_prior).abs() < 1e-6, "{}", format!("{}", should_prior-actual_prior).as_str());

        let cant_contract_set = MotifSet::rand_with_one_height_and_motif_len(-9.6, MIN_BASE, &wave, &background, 350, &mut rng);

        assert!(cant_contract_set.propose_contract_motif(&mut rng).is_none(), "Can contract PWM beyond allowed limits!");



        //Testing full RJ move
        for _ in 0..3{
            let add_mot = Motif::rand_mot_with_height_and_motif_len(13.2,mid_mot,motif_set.width, wave.seq(), wave.spacer(), &mut rng);
            _ = motif_set.add_motif(add_mot);
        }
        for i in 0..100 {
            let (step_set, selected_move, accepted) = motif_set.run_rj_move(&mut rng);
            match accepted {

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
            };
        }
    }

    #[test]
    fn it_works() {
        let mut rng = rand::thread_rng();
        let base = 3;
        let try_base: Base = Base::rand_new(&mut rng);
        let b = try_base.make_best(base);
        assert_eq!(base, b.best_base());

        let bc: Base = b.rev();

        assert_eq!(bc.best_base(), 3-base);

        let tc: Base = try_base.rev();

        assert_eq!(tc.best_base(), 3-try_base.best_base());

        assert!(!(tc == try_base));
        assert!(b == b.clone());

        let b_mag: f64 = b.show().iter().sum();
        let supposed_default_dist = b.show().iter().map(|a| ((a/b_mag)-(1.0/(BASE_L as f64))).powi(2)).sum::<f64>().sqrt();

        assert!(supposed_default_dist == b.dist(None));
      
        //println!("Conversion dists: {:?}, {:?}, {}", b.show(),  b.to_gbase().to_base().show(), b.dist(Some(&b.to_gbase().to_base())));
        //assert!(b == b.to_gbase().to_base());


        //let td: Base = Base::new([0.1, 0.2, 0.4, 0.3]);

        //assert!((td.rel_bind(1)-0.5_f64).abs() < 1e-6);
        //assert!((td.rel_bind(2)-1_f64).abs() < 1e-6);

        //let tg: GBase = GBase::new([0.82094531732, 0.41047265866, 0.17036154577], 2);


        //assert!(tg.to_base() == td);

    }

    #[test]
    fn trun_ln_normal_tests() {

        let dist: TruncatedLogNormal = TruncatedLogNormal::new(LOG_HEIGHT_MEAN, LOG_HEIGHT_SD, MIN_HEIGHT, MAX_HEIGHT).unwrap();

        let free_dist: LogNormal = LogNormal::new(LOG_HEIGHT_MEAN, LOG_HEIGHT_SD).unwrap();

        assert!((dist.pdf(6.0)-dist.ln_pdf(6.0).exp()).abs() < 1e-6);

        let mut rng = rand::thread_rng();

        let mut h: f64 = 0.;
        for i in 0..30000 { h = dist.sample(&mut rng); assert!((h >= dist.min()) && (h <= dist.max()))}

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

        let block_n: usize = 200;
        let u8_per_block: usize = 4375;
        let bp_per_block: usize = u8_per_block*4;
        let bp: usize = block_n*bp_per_block;
        let u8_count: usize = u8_per_block*block_n;

        println!("DF");
        let start_gen = Instant::now();
        //let blocks: Vec<u8> = (0..u8_count).map(|_| rng.u8(..)).collect();
        //let preblocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.u8(..)).collect();
        let preblocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.gen::<u8>()).collect();
        let blocks: Vec<u8> = preblocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>(); 
        let block_u8_starts: Vec<usize> = (0..block_n).map(|a| a*u8_per_block).collect();
        let block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
        let mut start_bases: Vec<usize> = (0..block_n).map(|a| a*bp_per_block).collect();
        let sequence: Sequence = Sequence::new_manual(blocks, block_lens.clone());
        let wave: Waveform = Waveform::create_zero(&sequence, 5);
        let duration = start_gen.elapsed();
        println!("Done gen {} bp {:?}", bp, duration);

        println!("{} gamma", gamma(4.));
        println!("{} gamma", ln_gamma(4.));

        //println!("{:?}", wave.raw_wave());

        let motif: Motif = Motif::from_motif(sequence.return_bases(0,0,20), 20., 5, &mut rng); //sequence

        let motif2: Motif = Motif::from_motif(sequence.return_bases(0,2,20), 20., 5, &mut rng); //sequence

        let start = Instant::now();

        let waveform = motif.generate_waveform(&wave);
        let duration = start.elapsed();
        
        let waveform2 = &waveform + &(motif2.generate_waveform(&wave));

        let corrs: Vec<f64> = vec![0.9, -0.1];
        let background = Background::new(0.25, 2.64, &corrs);
        let noise: Noise = waveform.produce_noise(&waveform2, &background);

        let grad = motif.single_motif_grad(&waveform2, &noise);




        let waveform_raw = waveform.raw_wave();

        let binds = motif.return_bind_score(&sequence);

        let start_b = Instant::now();
        let unsafe_waveform = unsafe{ motif.generate_waveform_from_binds(&binds, &wave) };
        let duration_b = start_b.elapsed();

        let unsafe_raw = unsafe_waveform.raw_wave();

        assert!(unsafe_raw.len() == waveform_raw.len());
        assert!((unsafe_raw.iter().zip(&waveform_raw).map(|(a, &b)| (a-b).powf(2.0)).sum::<f64>()).powf(0.5) < 1e-6);
        
        println!("Time elapsed in generate_waveform() is: {:?}", duration);
        println!("Time elapsed in the unsafe generate_waveform() is: {:?}", duration_b);

        println!("{}", motif);

        let random_motif = Motif::rand_mot(20., &sequence, 5, &mut rng);

        println!("Random motif\n{}", random_motif);

        assert!(random_motif.raw_kern().len() == 25);

        assert!((random_motif.peak_height.abs() >= MIN_HEIGHT) && (random_motif.peak_height.abs() <= MAX_HEIGHT));

        let matrix = motif.pwm();
        let nbases = matrix.len();

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

        println!("{} {} {} PWM PRIOR",sequence.kmer_in_seq(&motif.best_motif()), motif.pwm_prior(&sequence), (sequence.number_unique_kmers(motif.len()) as f64).ln() -(((BASE_L-1)*motif.len()) as f64)*((*PROP_UPPER_CUTOFF-*PROP_CUTOFF).ln()));
        assert!((motif.pwm_prior(&sequence)+(sequence.number_unique_kmers(motif.len()) as f64).ln()
                 -(((BASE_L-1)*motif.len()) as f64)*((*PROP_UPPER_CUTOFF-*PROP_CUTOFF).ln())).abs() < 1e-6);

        let un_mot: Motif = Motif::from_motif(vec![1usize;20], 10., 5, &mut rng);//Sequence

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

        let rev_best = best_mot.iter().rev().map(|a| BASE_L-1-a).collect::<Vec<usize>>();

        let bindy = unsafe {motif.prop_binding(&rev_best) };
        
        assert!(((bindy.0-1.0).abs() < 1e-6) && bindy.1);

        let pwm = motif.pwm();

        for i in 0..motif.len() {
            for j in 0..BASE_L {

                let mut for_mot = best_mot.clone();
                for_mot[i] = j;
                let mut rev_mot = rev_best.clone();
                rev_mot[motif.len()-1-i] = BASE_L-1-j;

                let defect: f64 = unsafe{ pwm[i].rel_bind(j)} ;

                
                let bindy = unsafe{ motif.prop_binding(&for_mot)};
                let rbind = unsafe{ motif.prop_binding(&rev_mot) };

                
                assert!(((bindy.0-defect).abs() < 1e-6) && !bindy.1);
                assert!(((rbind.0-defect).abs() < 1e-6) && rbind.1);

            }
        }

        let wave_block: Vec<u8> = vec![2,0,0,0, 170, 170, 170, 170, 170, 170, 170, 170,170, 170, 170, 170, 170, 170, 170]; 
        let wave_inds: Vec<usize> = vec![0, 9]; 
        let wave_starts: Vec<usize> = vec![0, 36];
        let wave_lens: Vec<usize> = vec![36, 40];
        let wave_seq: Sequence = Sequence::new_manual(wave_block, wave_lens);
        let wave_wave: Waveform = Waveform::create_zero(&wave_seq,1);

        let theory_base = [1.0, 1e-5, 1e-5, 0.2];

        let mat: Vec<Base> = (0..15).map(|_| Base::new(theory_base.clone())).collect::<Vec<_>>();


        println!("DF");
        let little_motif: Motif = Motif::raw_pwm(mat, 10.0, 1.0, 5); //wave_seq

        print!("{}", little_motif);
        println!("{:?}",little_motif.generate_waveform(&wave_wave).raw_wave());

        let small_block: Vec<u8> = vec![44, 24, 148, 240, 84, 64, 200, 80, 68, 92, 196, 144]; 
        let small_inds: Vec<usize> = vec![0, 6]; 
        let small_lens: Vec<usize> = vec![24, 24];
        let small: Sequence = Sequence::new_manual(small_block, small_lens);
        let small_wave: Waveform = Waveform::new(vec![0.1, 0.6, 0.9, 0.6, 0.1, -0.2, -0.4, -0.6, -0.6, -0.4], &small, 5);

        let mat: Vec<Base> = (0..15).map(|_| Base::new(theory_base.clone())).collect::<Vec<_>>();
        let wave_motif: Motif = Motif::raw_pwm(mat, 10.0, 1.0, 5); //small

        let rev_comp: Vec<bool> = (0..48).map(|_| rng.gen::<bool>()).collect();

        let checked = wave_motif.base_check( &small, &rev_comp, 0, 4);

        let forward: Vec<bool> = vec![true, false, false, true, true, false, false, false, true, true, false, false, true, false, false, false, true, true, true, false, true, false, true, false, true, true, false, false, true, false, true, false, true, false, false, false, true, false, true, false, true, true, false, false, false, false, false, false];

        let reverse: Vec<bool> = vec![true, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

        let correct: Vec<bool> = rev_comp.iter().enumerate().map(|(a, &b)| if b {reverse[a]} else {forward[a]}).collect();

        println!("correct: {:?}", correct);
        println!("checked: {:?}", checked);

        println!("small bl: {:?} {:?} {:?} {:?}", small.seq_blocks(), small.block_lens(), small.block_u8_starts(), small.return_bases(0, 0, 24));
        println!("blocks in seq: {:?}", small.seq_blocks());



        //TESTING base_check()
        for i in 0..2 {
            for j in 0..24 {

                let ind = if rev_comp[24*i+j] { j+wave_motif.len()-1-4 } else {j+4}; 
                if ind < 24 {
                    let bp = small.return_bases(i, ind, 1)[0];
                    let bp2 = small.return_bases(i, ind, 1)[0];
                    let matcher = if rev_comp[24*i+j] { bp == 3 } else { bp == 0};
                    println!("start loc: {}, bp: {}, bp2: {}, ind: {}, rev: {}, matcher: {}, checked: {}, correct: {}", 24*i+j, bp, bp2, ind, rev_comp[24*i+j], matcher, checked[24*i+j], correct[24*i+j]);
                    assert!(checked[24*i+j] == matcher);
                }
            }
        }


        let start = Instant::now();

        //        let (bind_score_floats, bind_score_revs) = self.return_bind_score(DATA.seq());

        let binds = motif.return_bind_score(&sequence);

        let duration = start.elapsed();
        println!("Time elapsed in bind_score() is: {:?}", duration);

        let start = Instant::now();
        let checked = motif.base_check(&sequence, &binds.1, 2, 4);
        //let checked = motif.base_check(&binds.1, 2, 4, &sequence);
        let duration = start.elapsed();
        println!("Time elapsed in check is: {:?}", duration);


        //TESTING return_bind_score()
        
        for i in 0..block_n {
            for j in 0..(bp_per_block-motif.len()) {

                //let test_against = motif.prop_binding(&VecDeque::from(sequence.return_bases(i, j, motif.len())));
                let test_against = unsafe{ motif.prop_binding(&(sequence.return_bases(i, j, motif.len())))};
                assert!((binds.0[i*bp_per_block+j]-test_against.0).abs() < 1e-6);
                assert!(binds.1[i*bp_per_block+j] == test_against.1);

            }
        }

        //TESTING generate_waveform() 


        let pwm_bp: usize = 3;
        let pwm_pos: usize = 6;
        let prop_bp = unsafe{motif.pwm[pwm_pos].rel_bind(pwm_bp)};
        let wave_main = motif.generate_waveform(&wave);
        let start0 = Instant::now();
        let wave_noh = motif.no_height_waveform(&wave);
        let duration0 = start.elapsed();
        let wave_gen: Vec<f64> = wave_main.raw_wave();
        let wave_sho: Vec<f64> = wave_noh.raw_wave();

        let checked = motif.base_check(&sequence, &binds.1, 3, 6);

        let start = Instant::now();
        let wave_filter = motif.only_pos_waveform(3, 6, &wave);
        let duration = start.elapsed();

        let raw_filter: Vec<f64> = wave_filter.raw_wave();

        let point_lens = wave_main.point_lens();
        start_bases.push(bp);
        let start_dats = wave_main.start_dats();
        let space: usize = wave_main.spacer();

        let half_len: usize = (motif.len()-1)/2;
        
        let kernel_check = motif.raw_kern();

        let kernel_mid = (kernel_check.len()-1)/2;

        //println!("STARTS: {:?}", sequence.block_u8_starts().iter().map(|a| a*BP_PER_U8).collect::<Vec<_>>());

        let start2 = Instant::now();
        let unsafe_filter = unsafe{motif.only_pos_waveform_from_binds(&binds, 3, 6, &wave)};
        let duration2 = start2.elapsed();

        let start3 = Instant::now();
        let just_divide = unsafe{motif.no_height_waveform_from_binds(&binds, &wave)};
        let duration3 = start3.elapsed();

        let unsafe_sho = just_divide.raw_wave();

        println!("Time in raw filter {:?}. Time in unsafe filter {:?}. Time spent in safe divide: {:?}. Time spent in divide {:?}", duration, duration2, duration0, duration3);
        let unsafe_raw_filter = unsafe_filter.raw_wave();

        assert!(unsafe_raw_filter.len() == raw_filter.len());
        let wave_s = unsafe_raw_filter.iter().zip(&raw_filter).map(|(a, &b)| (a-b).powf(2.0)).collect::<Vec<f64>>();
        println!("Wave_s {:?}", wave_s.iter().enumerate().filter(|(_,&a)| a > 0.).map(|(a, _)| a).collect::<Vec<_>>());

        assert!(wave_s.iter().sum::<f64>().powf(0.5) < 1e-6);

        assert!(unsafe_sho.len() == wave_sho.len());
        assert!((unsafe_sho.iter().zip(&wave_sho).map(|(a, &b)| (a-b).powf(2.0)).sum::<f64>()).powf(0.5) < 1e-6);
        
        println!("unsafe filter same as filter when properly made");
                
        //println!("filts {:?}", checked.iter().enumerate().filter(|(_, &b)| b).map(|(a, _)| a).collect::<Vec<_>>());
        for i in 0..block_n {
            for j in 0..point_lens[i] {

                let cut_low = if space*j >= kernel_mid+half_len {start_bases[i]+space*j-(kernel_mid+half_len)} else {start_bases[i]} ;
                let cut_high = if j*space+kernel_mid <= ((start_bases[i+1]+half_len)-start_bases[i]) {space*j+start_bases[i]+kernel_mid+1-half_len} else {start_bases[i+1]};
                let relevant_binds = (cut_low..cut_high).filter(|&a| binds.0[a] > THRESH).collect::<Vec<_>>();

                let relevant_filt = (cut_low..cut_high).filter(|&a| (binds.0[a] > THRESH) & checked[a]).collect::<Vec<_>>();

                if relevant_binds.len() > 0 {

                    let mut score: f64 = 0.0;
                    let mut filt_score: f64 = 0.0;
                    for k in 0..relevant_binds.len() {
                        score += binds.0[relevant_binds[k]]*kernel_check[(kernel_mid+(start_bases[i]+space*j))-(relevant_binds[k]+half_len)];
                    }
                    for k in 0..relevant_filt.len() {
                        filt_score += binds.0[relevant_filt[k]]*kernel_check[(kernel_mid+(start_bases[i]+space*j))-(relevant_filt[k]+half_len)];
                    }
                    assert!((wave_gen[start_dats[i]+j]-score).abs() < 1e-6);
                    assert!((wave_sho[start_dats[i]+j]-score/motif.peak_height()).abs() < 1e-6);

                    assert!((raw_filter[start_dats[i]+j]-filt_score/prop_bp).abs() < 1e-6);

                } else {
                    assert!(wave_gen[start_dats[i]+j].abs() < 1e-6);
                    assert!((wave_sho[start_dats[i]+j]).abs() < 1e-6);
                    assert!(raw_filter[start_dats[i]+j].abs() < 1e-6);
                }


            }
        }



    }








}
