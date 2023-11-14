//pub mod bases {
use std::{f64, fmt, fs};
use std::error::Error;
use std::ops::{Index, IndexMut};
//use std::time::{Duration, Instant};

use core::fmt::{Debug, Formatter};

use crate::waveform::{Kernel, Waveform, WaveformDef, Noise, Background};
use crate::sequence::{Sequence, BP_PER_U8, U64_BITMASK, BITS_PER_BP};
use crate::modified_t::ContinuousLnCDF;
use crate::{MAX_IND_RJ, MAX_IND_LEAP, MAX_IND_HMC, MOMENTUM_DIST, HMC_TRACE_STEPS, HMC_EPSILON};
use crate::data_struct::AllData;

use rand::Rng;
//use rand::prelude::IteratorRandom;
use rand::seq::SliceRandom;
use rand::distributions::{Distribution, WeightedIndex};

use statrs::{consts, StatsError};
use statrs::distribution::{Continuous, ContinuousCDF, LogNormal, Normal, Dirichlet};
use statrs::statistics::{Min, Max};

use once_cell::sync::Lazy;

use rayon::prelude::*;

use plotters::prelude::*;

use log::warn;

use regex::Regex;

use serde::{Serialize, Deserialize};


pub const SQRT_2: f64 = 1.41421356237;
pub const SQRT_3: f64 = 1.73205080757;


pub const BPS: [char; 4] = ['A', 'C', 'G', 'T'];
pub const BASE_L: usize = BPS.len();

pub const BASE_PRIOR_DENS: f64 = 1./6.; //Set to 1/Gamma(BASE_L)

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


const CLOSE: f64 = 1e-5;

pub const MIN_BASE: usize = 8;
pub const MAX_BASE: usize = 20; //For a four base system, the hardware limit here is 32. 
                                //To make sure this runs before the heat death of the universe
                                //while also not taking more memory than every human brain combined,
                                //We store many kmers as u64s. If you have 20 "bases" (amino acids)
                                //You need to force it to be at most 12. 


pub const MIN_HEIGHT: f64 = 3.;
pub const MAX_HEIGHT: f64 = 15.;
const LOG_HEIGHT_MEAN: f64 = 2.302585092994046; //This is ~ln(10). Can't use ln in a constant, and this depends on no other variables
const LOG_HEIGHT_SD: f64 = 0.25;

static HEIGHT_DIST: Lazy<TruncatedLogNormal> = Lazy::new(|| TruncatedLogNormal::new(LOG_HEIGHT_MEAN, LOG_HEIGHT_SD, MIN_HEIGHT, MAX_HEIGHT).unwrap() );

static PROPOSE_EXTEND: Lazy<Dirichlet> = Lazy::new(|| Dirichlet::new(vec![10.0_f64; BASE_L]).unwrap());

const PROB_POS_PEAK: f64 = 0.9;

pub const THRESH: f64 = 1e-2; //SAFETY: This must ALWAYS be strictly greater than 0, or else we violate safety guarentees later.  

//This is roughly how much an additional motif should improve the ln posterior before it's taken seriously
//The more you increase this, the fewer motifs you will get, on average
const NECESSARY_MOTIF_IMPROVEMENT: f64 = 20.0_f64;

pub const RJ_MOVE_NAMES: [&str; 4] = ["New motif", "Delete motif", "Extend motif", "Contract Motif"];

pub const BP_ARRAY: [Bp; BASE_L] = [Bp::A, Bp::C, Bp::G, Bp::T];

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
        //SAFETY: Bp always corresponds to a valid usize
        unsafe{std::mem::transmute::<Bp, usize>(bp)}
    }
}

impl TryFrom<usize> for Bp {
    type Error = &'static str;
    fn try_from(bp: usize) -> Result<Self, Self::Error> {
        if bp < BASE_L {
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Base {
   props: [ f64; BASE_L],
}

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
        
        for i in 0..props.len() {
                props[i] = props[i]/max;
        }




        Base { props }
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

    pub fn propose_safe_new<R: Rng + ?Sized>(rng: &mut R) -> Base {

        let props: [f64; BASE_L] = (*PROPOSE_EXTEND.sample(rng).data.as_vec()).clone().try_into().expect("We constructed PROPOSE_EXTEND based on BASE_L");
        Base::new(props)

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


    pub fn make_best(&self, best: Bp) -> Base {

        let mut base2 = self.clone();

        let which_b = base2.best_base();

        if best != which_b {

            let tmp = base2[which_b];
            base2[which_b] = base2[best];
            base2[best] = tmp;
        }


        base2
    }

    pub fn best_base(&self) -> Bp {
        //Safety: self.props has a size of BASE_L, so this always produces a valid result
        unsafe{ std::mem::transmute::<usize, Bp>(Self::argmax(&self.props))}
    }

    pub fn dist_sq(&self, base: Option<&Base>) -> f64 {

        let as_simplex: [f64; BASE_L-1] = self.as_simplex();
        match base {

            None => as_simplex.iter().map(|a| a.powi(2)).sum::<f64>(),
            Some(other) => {
                let other_simplex = other.as_simplex();
                as_simplex.iter().zip(other_simplex).map(|(a, b)| (a-b).powi(2)).sum::<f64>()
            }
        }

    }

    pub fn as_probabilities(&self) -> [f64; BASE_L] {
        let magnitude: f64 = self.props.iter().sum();
        self.props.iter().map(|a| a/magnitude).collect::<Vec<f64>>().try_into().expect("I'm never worried about error here because all Base are guarenteed to be length BASE_L")
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

    //This should never be publicly exposed, because it assumes an invariant that simplex_coords
    //will always be inside of the tetrahedron. Failing to uphold this invariant will produce
    //nonsense
    fn simplex_to_base(simplex_coords: &[f64; BASE_L-1]) -> Base {

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

        let b_form: [f64; BASE_L]  = probs.into_iter().map(|a| a/max).collect::<Vec<_>>().try_into().unwrap();

        Base{props : b_form}

    }

    //This result is given in column major order
    fn d_base_d_simplex(&self) -> [[f64; BASE_L-1]; BASE_L-1] {

        let prob_base = self.as_probabilities();
        let best_bp = self.best_base();
        let mut bp_inds: Vec<Bp> = BP_ARRAY.to_vec();
        bp_inds.retain(|&b| b != best_bp);
        let pmax: f64 = prob_base[best_bp as usize];
        let pmax_sq: f64 = pmax.powi(2);
        let ps: [f64; BASE_L-1] = bp_inds.iter().map(|&a| prob_base[a as usize]).collect::<Vec<_>>().try_into().expect("bp inds is always the same size after its defining ommission.");

        let mut jacobian = [[0_f64; (BASE_L-1)]; (BASE_L-1)];
        for k in 0..(BASE_L-1) {
            let mut pstar = [ps[k]; BASE_L-1];
            pstar[k] += pmax;
            
            for ti in 0..(BASE_L-1) {
                let m_inv_vec = (0..BASE_L-1).map(|m| COL_PRIMARY_INVERT_SIMPLEX[ti][bp_inds[m] as usize]);
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
            return 0;
        }

        let mut arg = 0;

        for ind in 1..(arr.len()) {
            arg = if arr[ind] > arr[arg] { ind } else {arg};
        }

        arg

    }

    pub fn add_in_hmc(&self, addend: [f64; BASE_L-1], confine_base: bool) -> Self {

       let tetra = self.as_simplex();
       Base::simplex_to_base(&reflect_tetra(tetra, addend, confine_base))

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
    pub fn safe_bind(&self, bp: usize) -> f64 {
        self.props[bp]
    }

    pub fn to_unit_sum(&self) -> [f64; BASE_L] {
        let summed = self.props.iter().sum::<f64>();
        let ret: [f64; BASE_L] = self.props.iter().map(|&a| a/summed).collect::<Vec<_>>().try_into().unwrap();
        ret
    }


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
fn reflect_tetra(start: [f64; BASE_L-1], push: [f64; BASE_L-1], confine_base: bool) -> [f64; BASE_L-1] {
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

}



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

        let peak_height: f64 = sign*(*HEIGHT_DIST).sample(rng);

        Self::raw_pwm(pwm, peak_height)
    }


    //TODO: make sure that this is used iff there's a guarentee that a motif is allowed
    pub fn from_motif<R: Rng + ?Sized>(best_bases: Vec<Bp>, rng: &mut R) -> Motif {
        
        let mut pwm: Vec<Base> = best_bases.iter().map(|a| Base::from_bp(*a, rng)).collect();
        pwm.reserve_exact(MAX_BASE-best_bases.len());

        //let height_dist: truncatedlognormal = truncatedlognormal::new(log_height_mean, log_height_sd, min_height, max_height).unwrap();


        let sign: f64 = rng.gen();
        let sign: f64 = if sign < PROB_POS_PEAK {1.0} else {-1.0};

        let peak_height: f64 = sign*(*HEIGHT_DIST).sample(rng);


        Motif {
            peak_height: peak_height,
            pwm: pwm,
        }


    }

    pub fn from_motif_uniform<R: Rng + ?Sized>(best_bases: Vec<Bp>, rng: &mut R) -> Motif {
        
        let mut pwm: Vec<Base> = best_bases.iter().map(|a| Base::from_bp_with_uniforms(*a, rng)).collect();
        pwm.reserve_exact(MAX_BASE-best_bases.len());

        //let height_dist: truncatedlognormal = truncatedlognormal::new(log_height_mean, log_height_sd, min_height, max_height).unwrap();


        let sign: f64 = rng.gen();
        let sign: f64 = if sign < PROB_POS_PEAK {1.0} else {-1.0};

        let peak_height: f64 = sign*(*HEIGHT_DIST).sample(rng);


        Motif {
            peak_height: peak_height,
            pwm: pwm,
        }


    }


    
    pub fn rand_mot<R: Rng + ?Sized>(seq: &Sequence, rng: &mut R) -> Motif {


        let num_bases = rng.gen_range(MIN_BASE..(MAX_BASE+1));

        let mot = seq.random_valid_motif(num_bases);

        
        Self::from_motif(mot,  rng)
        

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

        Some((Self::from_motif_uniform(mot, rng), propensities[selection]/sum_prop))
    }

    pub fn rand_mot_with_height<R: Rng + ?Sized>(peak_height: f64, seq: &Sequence, rng: &mut R) -> Motif {


        let num_bases = rng.gen_range(MIN_BASE..(MAX_BASE+1));

        let mot = seq.random_valid_motif(num_bases);

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

        let mot = seq.random_valid_motif(num_bases);

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

            let old_base: u64 = ((old_best & (U64_BITMASK << (BITS_PER_BP*i))) >> (BITS_PER_BP*i));
            let new_base: u64 = ((new_best & (U64_BITMASK << (BITS_PER_BP*i))) >> (BITS_PER_BP*i));

            if new_base != old_base {
                //SAFETY: new_base is always constrained to be less than BASE_L (4)
                new_mot.pwm[i] = new_mot.pwm[i].make_best(unsafe{Bp::usize_to_bp(new_base as usize)});
            }

        }

        new_mot

    }

    //Safety: Momentum MUST have a length equal to precisely 1+(BASE_L-1)*self.len()
    unsafe fn add_momentum(&self, eps: f64, momentum: &[f64], confine_base: bool) -> Self {

        let mut new_mot = self.clone();
        
        //if momentum[0] != 0.0 {
            let mut h = self.peak_height.abs();
            h += eps*momentum[0];
            h = reflect_abs_height(h);
            //h = MIN_HEIGHT+((MAX_HEIGHT-MIN_HEIGHT)/(1.0+((-h/SPREAD_HMC_CONV).exp())));
            new_mot.peak_height = self.peak_height.signum()*h;

        //}

        for i in 0..self.len() {
            
            let slice: [f64; BASE_L-1] = (1..(BASE_L)).map(|a| *(momentum.get_unchecked(i*(BASE_L-1)+a))*eps).collect::<Vec<_>>().try_into().unwrap();
            new_mot.pwm[i] = self.pwm[i].add_in_hmc(slice, confine_base);
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
    unsafe fn prop_binding(&self, kmer: &[Bp]) -> (f64, bool) { 
        

        //let kmer: Vec<usize> = kmer_slice.to_vec();

        let mut bind_forward: f64 = 1.0;
        let mut bind_reverse: f64 = 1.0;

        unsafe{
        for i in 0..self.len() {
            bind_forward *= self.pwm[i][*kmer.get_unchecked(i)];
            //bind_reverse *= self.pwm[i].rel_bind(BASE_L-1-*kmer.get_unchecked(self.len()-1-i));
            bind_reverse *= self.pwm[i][kmer.get_unchecked(self.len()-1-i).complement()];
        }
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
    pub fn base_check(&self, seq: &Sequence, rev_comp: &Vec<bool>, bp: Bp, motif_pos: usize) -> Vec<bool> {
            
        let coded_sequence = seq.seq_blocks();

        let rev_pos = self.len()-1-motif_pos;

        let forward_bp = bp as usize as u8;
        let comp_bp = bp.complement() as usize as u8;

        let bp_to_look_for: Vec<u8> = rev_comp.iter().map(|&a| if a {comp_bp} else {forward_bp}).collect();

        let bp_lead: Vec<usize> = rev_comp.iter().enumerate().map(|(i, &r)| if r { i+rev_pos } else { i+motif_pos }).collect();

        let loc_coded_lead: Vec<usize> = bp_lead.iter().map(|b| b/4).collect();
        //This finds 3-mod_4(pos) with bit operations. Faster than (motif_pos & 3) ^ 3 for larger ints
        //Note that we DO need the inverse in Z4. The u8 bases are coded backwards, where the 4^3 place is the last base
        
        const MASKS: [u8; 4] = [0b00000011, 0b00001100, 0b00110000, 0b11000000];
        const SHIFT_BASE_BY: [u8; 4] = [0, 2, 4, 6];

        loc_coded_lead.iter().zip(bp_lead).zip(bp_to_look_for).map(|((&c, s), b)| 
                                            if c < coded_sequence.len() { //Have to check if leading base is over the sequence edge
                                                //This 
                                               ((coded_sequence[c] & MASKS[s-4*c]) >> SHIFT_BASE_BY[s-4*c]) == b
                                            } else { false }).collect::<Vec<bool>>()


    } 



    //NOTE: if data does not point to the same sequence that self does, this will break. HARD. 
    pub fn generate_waveform<'a>(&self, data: &'a Waveform, unit_kernel: &Kernel) -> Waveform<'a> {

        let mut occupancy_trace: Waveform = data.derive_zero();

        let mut actual_kernel: Kernel;

        let starts = data.seq().block_u8_starts();

        let lens = data.seq().block_lens();

        let (bind_score_floats, _) = self.return_bind_score(data.seq());

        for i in 0..starts.len() { //Iterating over each block
            for j in 0..(lens[i]-self.len()) {
                if bind_score_floats[starts[i]*BP_PER_U8+j] > THRESH {
                    actual_kernel = unit_kernel*(bind_score_floats[starts[i]*BP_PER_U8+j]*self.peak_height) ;
                    //println!("{}, {}, {}, {}", i, j, lens[i], actual_kernel.len());
                    unsafe {occupancy_trace.place_peak(&actual_kernel, i, j+(self.len()-1)/2);} 

                    //println!("peak {} {} {}", starts[i]*BP_PER_U8+j, bind_score_floats[starts[i]*BP_PER_U8+j], occupancy_trace.read_wave()[(starts[i]*BP_PER_U8+j)/data.spacer()]);
                    //count+=1;
                }
            }
        }

        //println!("num peaks {}", count);

        occupancy_trace

    }

    pub fn no_height_waveform<'a>(&self, data: &'a Waveform, unit_kernel: &Kernel) -> Waveform<'a> {

        let mut occupancy_trace: Waveform = data.derive_zero();

        //let base_kernel = &self.kernel*(1.0/self.peak_height);
        //let mut actual_kernel: Kernel = &base_kernel*1.0;
        let mut actual_kernel: Kernel;

        let starts = data.seq().block_u8_starts();

        let lens = data.seq().block_lens();

        let (bind_score_floats, _) = self.return_bind_score(data.seq());

        for i in 0..starts.len() { //Iterating over each block
            for j in 0..(lens[i]-self.len()) {
                if bind_score_floats[starts[i]*BP_PER_U8+j] > THRESH {
                    //actual_kernel = &base_kernel*(bind_score_floats[starts[i]*BP_PER_U8+j]);
                    actual_kernel = unit_kernel*(bind_score_floats[starts[i]*BP_PER_U8+j]);
                    unsafe {occupancy_trace.place_peak(&actual_kernel, i, j+(self.len()-1)/2);} 
                }
            }
        }

        occupancy_trace

    }
    
    pub fn only_pos_waveform<'a>(&self,bp: Bp, motif_pos: usize, data: &'a Waveform, unit_kernel: &Kernel) -> Waveform<'a> {

        let mut occupancy_trace: Waveform = data.derive_zero();

        let mut actual_kernel: Kernel;

        let starts = data.seq().block_u8_starts();

        let lens = data.seq().block_lens();

        let (bind_score_floats, bind_score_revs) = self.return_bind_score(data.seq());

        let bind = self.pwm()[motif_pos][bp];

        let checked = self.base_check( data.seq(), &bind_score_revs, bp, motif_pos);
        for i in 0..starts.len() { //Iterating over each block
            for j in 0..(lens[i]-self.len()) {
                if checked[starts[i]*BP_PER_U8+j] && (bind_score_floats[starts[i]*BP_PER_U8+j] > THRESH) {
                    //println!("safe binding at {}", starts[i]*BP_PER_U8+j);
                    actual_kernel = unit_kernel*(bind_score_floats[starts[i]*BP_PER_U8+j]*self.peak_height/bind);
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
    #[allow(dead_code)]
    unsafe fn generate_waveform_from_binds<'a>(&self, binds: &(Vec<f64>, Vec<bool>), data: &'a Waveform, unit_kernel: &Kernel) -> Waveform<'a> {

        let mut occupancy_trace: Waveform = data.derive_zero();

        let mut actual_kernel: Kernel;

        let starts = data.seq().block_u8_starts();

        let lens = data.seq().block_lens();


        for i in 0..starts.len() { //Iterating over each block
            for j in 0..(lens[i]-self.len()) { //-self.len() is critical for maintaining safety of place_peak. 
                                               //It's why we don't allow sequence blocks unless they're bigger than the max motif size
                if binds.0[starts[i]*BP_PER_U8+j] > THRESH {
                    actual_kernel = unit_kernel*(binds.0[starts[i]*BP_PER_U8+j]*self.peak_height) ;
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
    //Safety: You MUST ensure that the binding score and reverse complement is valid for this particular motif, because you can technically use 
    //        ANY binding score here, and this code won't catch it, especially if the dimensions check out. We code this primarily for speed of calculation in the gradient calculation
    unsafe fn no_height_waveform_from_binds<'a>(&self, binds: &(Vec<f64>, Vec<bool>), data: &'a Waveform, unit_kernel: &Kernel) -> Waveform<'a> {

        let mut occupancy_trace: Waveform = data.derive_zero();

        let mut actual_kernel: Kernel;
        //let mut actual_kernel: Kernel = &self.kernel*1.0;

        let starts = data.seq().block_u8_starts();

        let lens = data.seq().block_lens();


        for i in 0..starts.len() { //Iterating over each block
            for j in 0..(lens[i]-self.len()) { //-self.len() is critical for maintaining safety of place_peak. 
                                               //It's why we don't allow sequence blocks unless they're bigger than the max motif size
                if  binds.0[starts[i]*BP_PER_U8+j] > THRESH {
                    //actual_kernel = &self.kernel*(binds.0[starts[i]*BP_PER_U8+j]/self.peak_height);
                    actual_kernel = unit_kernel*(binds.0[starts[i]*BP_PER_U8+j]);
                    occupancy_trace.place_peak(&actual_kernel, i, j+(self.len()-1)/2); 
                }
            }
        }

        occupancy_trace



    }
    //Safety: You MUST ensure that the binding score and reverse complement is valid for this particular motif, because you can technically use 
    //        ANY binding score here, and this code won't catch it, especially if the dimensions check out. We code this primarily for speed of calculation in the gradient calculation
    //        You must also ensure that bp < BASE_L, and that motif_pos < self.len()
    unsafe fn only_pos_waveform_from_binds<'a>(&self, binds: &(Vec<f64>, Vec<bool>), bp: Bp, motif_pos: usize, data: &'a Waveform, unit_kernel: &Kernel) -> Waveform<'a> {

        let mut occupancy_trace: Waveform = data.derive_zero();

        let mut actual_kernel: Kernel;

        let starts = data.seq().block_u8_starts();

        let lens = data.seq().block_lens();

        let bind = self.pwm()[motif_pos][bp];    

        let checked = self.base_check( data.seq(), &binds.1, bp, motif_pos);
        for i in 0..starts.len() { //Iterating over each block
            for j in 0..(lens[i]-self.len()) { //-self.len() is critical for maintaining safety of place_peak. 
                                               //It's why we don't allow sequence blocks unless they're bigger than the max motif size
                if checked[starts[i]*BP_PER_U8+j] && (binds.0[starts[i]*BP_PER_U8+j] > THRESH) {
                    //println!("unsafe binding at {}", starts[i]*BP_PER_U8+j);
                    actual_kernel = unit_kernel*((binds.0[starts[i]*BP_PER_U8+j]*self.peak_height)/bind);
                    occupancy_trace.place_peak(&actual_kernel, i, j+(self.len()-1)/2); 
                }
            }
        }

        occupancy_trace



    }


    //SAFETY: the d_ad_stat_d_noise must be of the same length as the noise vector we get from data.
    pub unsafe fn parallel_single_motif_grad(&self,  data: &Waveform, d_ad_stat_d_noise: &Vec<f64>, d_ad_like_d_ad_stat: f64, background: &Background) -> Vec<f64> {

        let binds = self.return_bind_score(data.seq());

        let n = 1+self.len()*(BASE_L-1);


        let mut d_ad_like_d_grad_form: Vec<f64> = (0..n).into_par_iter().map(|i| {
            if i == 0 {
                let d_noise_d_h = self.no_height_waveform_from_binds(&binds, data, background.kernel_ref())
                                               .account_auto(background);
                (d_ad_like_d_ad_stat * ((&d_noise_d_h * d_ad_stat_d_noise))) + self.d_height_prior_d_hmc()
            } else {
                let index = i-1;
                let base_id = index/(BASE_L-1); //Remember, this is integer division, which Rust rounds down
                let mut bp_usize = index % (BASE_L-1);
                bp_usize += if bp_usize >= (self.pwm[base_id].best_base() as usize) {1} else {0}; //If best_base == BASE_L-1, then we access bp = 0, 1, .., BASE_L-2. 
                                                                           //At this point, base_id already goes to the next base, skipping bp = BASE_L-1
                                                                           //This is important, because statically guarentees the safety of using rel_bind

                let bp = Bp::usize_to_bp(bp_usize); //SAFETY: constraints on bp_usize ensure that this is always safe
                let result =
                      (&(self.only_pos_waveform_from_binds(&binds, bp, base_id, data, background.kernel_ref())
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
            
            let best_bp = self.pwm[i].best_base();
            let mut bp_inds: Vec<Bp> = BP_ARRAY.to_vec(); 
            bp_inds.retain(|&b| b != best_bp);
            
            let ln_like_grads: [f64; BASE_L-1] = index_into.iter().map(|&k| *grad_chunk.get_unchecked(k)).collect::<Vec<_>>().try_into().unwrap();

            let simplex_grad = self.pwm[i].d_base_d_simplex();
            for k in 0..(BASE_L-1) {

                *grad_chunk.get_unchecked_mut(index_into[k]) = ln_like_grads.iter().zip(simplex_grad[k].iter()).map(|(&a, &b)| a*b).sum::<f64>();
            }
           
        }).collect::<Vec<_>>();



            
        d_ad_like_d_grad_form

    }


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

            let b1 = if (ind < mod_len_a) || (ind >= (pwm_1.len()+mod_len_a)) {None} else {Some(&pwm_1[ind-mod_len_a])};
            let b2 = if (ind < mod_len_b) || (ind >= (pwm_2.len()+mod_len_b)) {None} else {Some(&pwm_2[ind-mod_len_b])};

            distance += match b1 {
                Some(b) => b.dist_sq(b2),
                None => match b2{
                    Some(bb) => bb.dist_sq(None),
                    None => { warn!("PWM alignment in distance is causing a complete miss!"); 0.0},
                },
            };

        }


        distance.sqrt()

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
//SAFETY: any method which adds a Motif to the set directly, without checking
//        that the kernel length is less than the length of data in all sequence
//        blocks, is IMMEDIATELY unsafe. I do not include such methods here, 
//        but if YOU do, you MUST enforce this invariant.
pub struct MotifSet<'a> {

    set: Vec<Motif>, 
    signal: Waveform<'a>,
    ln_post: Option<f64>,
    data: &'a Waveform<'a>, 
    background: &'a Background,
}

impl<'a> MotifSet<'a> {
   
    pub fn rand_with_one<R: Rng + ?Sized>(data: &'a Waveform<'a>, background: &'a Background, rng: &mut R) -> Self {


        let mut valid: bool;

        let mot_set = loop { //We occasionally seem to randomly generate sets with such high occupancy everywhere as to be ridiculous

            let set = vec![Motif::rand_mot(data.seq(), rng)];

            let signal = set[0].generate_waveform(data, background.kernel_ref());

            let mut mot_set_try = MotifSet{ set: set, signal: signal, ln_post: None, data: data, background: background};

            let like = mot_set_try.ln_posterior();

            valid = like.is_finite();

            if valid {
                break mot_set_try;
            }
        };
        
        mot_set
    }

    pub fn rand_with_one_height<R: Rng + ?Sized>(peak_height: f64, data: &'a Waveform<'a>, background: &'a Background, rng: &mut R) -> Self {

        let set = vec![Motif::rand_mot_with_height(peak_height, data.seq(), rng)];

        let signal = set[0].generate_waveform(data, background.kernel_ref());

        let mut mot_set = MotifSet{ set: set,  signal: signal, ln_post: None, data: data, background: background};

        let _ = mot_set.ln_posterior();

        mot_set
    }
    
    pub fn rand_with_one_height_and_motif_len<R: Rng + ?Sized>(peak_height: f64, motif_len: usize, data: &'a Waveform<'a>, background: &'a Background, rng: &mut R) -> Self {


        let set = vec![Motif::rand_mot_with_height_and_motif_len(peak_height, motif_len, data.seq(), rng)];

        let signal = set[0].generate_waveform(data, background.kernel_ref());

        let mut mot_set = MotifSet{ set: set, signal: signal, ln_post: None, data: data, background: background};

        let _ = mot_set.ln_posterior();

        mot_set
    }
    
    #[cfg(test)]
    fn recalced_signal(&self) -> Waveform {
        let mut signal = self.data.derive_zero();
        for mot in self.set.iter() {
            signal += &(mot.generate_waveform(self.data, self.background.kernel_ref()));
        }

        signal
    }

    fn recalc_signal(&mut self) {
        self.signal = self.data.derive_zero();
        for mot in self.set.iter() {
            self.signal += &(mot.generate_waveform(self.data, self.background.kernel_ref()));
        }
    }

    fn accept_test<R: Rng + ?Sized>(old: f64, new: f64, rng: &mut R) -> bool {

        let diff_ln_like = new-old;

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
    //and p = 1-(-NECESSARY_MOTIF_IMPROVEMENT).exp().
    //NOTE: the ommission of ln(p) term is deliberate. It amounts to a normalization constant
    //for the motif set prior, and would obfuscate the true point of this prior
    pub fn motif_num_prior(&self) -> f64 {
        -((self.set.len()-1) as f64)*NECESSARY_MOTIF_IMPROVEMENT
    }

    pub fn ln_prior(&self) -> f64 {
        self.motif_num_prior() + self.set.iter().map(|a| a.height_prior()+a.pwm_prior(self.data.seq())).sum::<f64>()
    }

    pub fn ln_likelihood(&self) -> f64 {
        Noise::ad_like((self.signal).produce_noise(self.data, self.background).ad_calc())
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

        self.signal += &new_mot.generate_waveform(self.data, self.background.kernel_ref()) ;
        self.set.push(new_mot);
        self.ln_post = None;
        self.ln_posterior()

    }
    
    fn insert_motif(&mut self, new_mot: Motif, position: usize) -> f64 {

        self.signal += &new_mot.generate_waveform(self.data, self.background.kernel_ref()) ;
        self.set.insert(position, new_mot);
        self.ln_post = None;
        self.ln_posterior()

    }

    fn remove_motif(&mut self, rem_id: usize) -> f64{
        assert!(rem_id < self.set.len());

        let rem_mot = self.set.swap_remove(rem_id);
        self.signal -= &rem_mot.generate_waveform(self.data, self.background.kernel_ref());
        self.ln_post = None;
        self.ln_posterior()

    }

    fn remove_motif_void(&mut self, rem_id: usize) {

        assert!(rem_id < self.set.len());
        let rem_mot = self.set.swap_remove(rem_id);
        self.signal -= &rem_mot.generate_waveform(self.data, self.background.kernel_ref());
        self.ln_post = None;

    }

    fn replace_motif(&mut self, new_mot: Motif, rem_id: usize) -> f64 {
        let rem_mot = self.set[rem_id].clone();
        self.signal -= &rem_mot.generate_waveform(self.data, self.background.kernel_ref());
        self.signal += &new_mot.generate_waveform(self.data, self.background.kernel_ref()) ;
        self.set[rem_id] = new_mot;
        self.ln_post = None;
        self.ln_posterior()
    }

    //This proposes a new motif for the next motif set, but does not do any testing vis a vis whether such a move will be _accepted_
    fn propose_new_motif<R: Rng + ?Sized>(&self, rng: &mut R ) -> Option<(Self, f64)> {
        let mut new_set = self.derive_set();
        let remaining = self.data-&self.signal;
        let attempt = Motif::rand_propensity_mot(&remaining, rng); //There may not always be a place with decent propensities
        match attempt {
            None => {/*println!("No good propensities");*/ None},
            Some((new_mot, pick_prob)) => {
                let ln_gen_prob = new_mot.height_prior()+pick_prob.ln()-((MAX_BASE+1-MIN_BASE) as f64).ln();
                //let h_prior = new_mot.height_prior();
                let ln_post = new_set.add_motif(new_mot);
                //println!("propose birth: like: {} height: {}, pick_prob: {}, len sel: {}",ln_post, h_prior, pick_prob.ln(), ((MAX_BASE+1-MIN_BASE) as f64).ln());
                Some((new_set, ln_post-ln_gen_prob)) //Birth moves subtract the probability of their generation
            }
        }
    }

    //This proposes removing an old motif for the next motif set
    //It does no testing save for checking if removing a motif would produce a possible set
    fn propose_kill_motif<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<(Self, f64)> {

        if self.set.len() <= 1 { //We never want to propose a set with no motifs, ever
            None
        } else {
            let mut new_set = self.derive_set();
            let rem_id = rng.gen_range(0..self.set.len());
            let mod_signal = &(self.signal)-&(self.set[rem_id].generate_waveform(self.data, self.background.kernel_ref()));
            let remaining = self.data-&mod_signal;
            let propensities = remaining.kmer_propensities(self.set[rem_id].len());
            let sum_propensities = propensities.iter().sum::<f64>();
            if sum_propensities <= 0.0 {
                return None;
            }
            let pick_prob = propensities[self.data.seq().id_of_u64_kmer_or_die(self.set[rem_id].len(),Sequence::kmer_to_u64(&self.set[rem_id].best_motif()))]/sum_propensities;
            let ln_gen_prob = self.set[rem_id].height_prior()+pick_prob.ln()-((MAX_BASE+1-MIN_BASE) as f64).ln();
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

            let mut new_mot = self.set[extend_id].clone();
            let new_base = Base::propose_safe_new(rng);
            let base_ln_density = PROPOSE_EXTEND.ln_pdf(&(new_base.as_probabilities().to_vec().try_into().unwrap()));
            new_mot.pwm.push(new_base);
            //let ln_gen_prob = new_mot.height_prior()+new_mot.pwm_prior(self.data.seq());
            if self.data.seq().kmer_in_seq(&new_mot.best_motif()) { //When we extend a motif, its best base sequence may no longer be in the sequence
                let ln_post = new_set.replace_motif(new_mot, extend_id);
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

        if self.set[contract_id].len() <= MIN_BASE { //We want to immediately reject any move which contracts a motif below minimum length
            None
        } else {
            let mut new_mot = self.set[contract_id].clone();
            let old_base = new_mot.pwm.pop();
            let ln_post = new_set.replace_motif(new_mot, contract_id);
            let base_ln_density = PROPOSE_EXTEND.ln_pdf(&(old_base.expect("We know this is bigger than 0").as_probabilities().to_vec()).try_into().unwrap());
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

            None => {
                //println!("Failed move {}", which_rj);
                (self.clone(), which_rj, false)
            },
            Some((new_mot, modded_ln_like)) => {
                let accepted = Self::accept_test(self.ln_post.unwrap(), modded_ln_like, rng);
                //println!("old ln P {}, modded ln P {}, move {:?}, accepted: {}", self.ln_post.unwrap(), modded_ln_like, RJ_MOVE_NAMES[which_rj], accepted);
                if accepted {
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

           for i in 0..selection_probs.len() {
               //This subtraction might seem technically unnecessary, but
               //but computers are not infinitely precise. We want to 
               //ensure that we minimize numerical issues

               selection_probs[i] = (likes_and_mots[i].0-normalize_ln_like).exp().abs();
           }

           /*if selection_probs.len() == 0 {
               panic!("No states being selected from!");
           }*/

           //println!("base leap sel {:?}", selection_probs);

           //let likes = (0..likes_and_mots.len()).map(|i| likes_and_mots[i].0).collect::<Vec<_>>();
           //println!("likes base leap {:?}", likes);

           let dist = match WeightedIndex::new(&selection_probs) {

               Ok(weights) => weights, 
               Err(_) => {warn!("Issue with base leap weights in this step. Randomly selecting. Discard inference before this point {:?}", selection_probs);
                   WeightedIndex::new(&(vec![1.0; likes_and_mots.len()])).expect("This is statically valid")},
           };

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
               new_set.set[k] = unsafe{ prior_set.set[k].add_momentum(HMC_EPSILON, &momentum_apply[start..next_start], false)};
              
               if !self.data.seq().kmer_in_seq(&new_set.set[k].best_motif()) {
                   new_set.set[k] = unsafe{ prior_set.set[k].add_momentum(HMC_EPSILON, &momentum_apply[start..next_start], true)};
               }
               start = next_start;
           }

           new_set.recalc_signal();

           let gradient_new = new_set.gradient();

           for i in 0..momentum_apply.len() {
               momentum_apply[i] += (HMC_EPSILON*gradient_new[i])/2.0;
           }

           //We want gradient_old to take ownership of the gradient_new values, and gradient_old's prior values to be released
           //Same with prior_set and new_set
           gradient_old = gradient_new;
           prior_set = new_set;

       }

       let mut delta_kinetic_energy: f64 = 0.0;


       //Our M-H cutoff is exp(-(new hamiltonian-old hamiltonian))
       //ie exp(old_hamiltonian-new hamiltonian)
       //hamiltonian = kinetic+potential, and potential = -ln posterior
       //So, the delta_kinetic_energy is (momentum^2-momentum_apply^2)/2
       //And the delta_potential_energy is prior_set ln_post (which is the proposed posterior) -
       //self.ln_post

       for i in 0..momentum.len() {
           delta_kinetic_energy += (momentum[i].powi(2) - momentum_apply[i].powi(2)  )/2.0;
       }

       let delta_potential_energy = prior_set.ln_posterior()-self.ln_post.unwrap();

       if delta_potential_energy == -f64::INFINITY {
            println!("d k {} p {} old_mots {:?} new mots {:?} mot_dists {:?}",delta_kinetic_energy, delta_potential_energy,self.set.iter().map(|a| a.best_motif()).collect::<Vec<_>>(), prior_set.set.iter().map(|a| a.best_motif()).collect::<Vec<_>>(), self.set.iter().zip(prior_set.set.iter()).map(|(a,b)| a.distance_function(b)).collect::<Vec<_>>()); 
       }
       if Self::accept_test(0.0, delta_kinetic_energy+delta_potential_energy, rng){
           (prior_set, true, delta_potential_energy+delta_kinetic_energy)
       } else {
           //println!("rejected. the old mot is {:?}, and the diff is {}", self, delta_potential_energy+delta_kinetic_energy);
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
                let mod_mot = unsafe {a.add_momentum(1.0, perturb_vec.as_slice(), false)};

                let mut alter_set = self.clone();
                let new_ln_post = alter_set.replace_motif(mod_mot,k);
                let new_ln_like = new_ln_post-alter_set.ln_prior();
                (new_ln_post-curr_post)/h
                //(new_ln_like-curr_like)/h
            }).collect::<Vec<f64>>();

            motif_grad
            
        }).flatten().collect();

        gradient

    }






}


impl Debug for MotifSet<'_> {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        write!(f, " ln_post: {:?}, background: {:?} \n set: {:#?}", self.ln_post, self.background, self.set)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StrippedMotifSet {
    set: Vec<Motif>,
    ln_post: f64,
}

impl StrippedMotifSet {

    //Yes, self, not &self. I'm destroying the stripped motif set whenever I make the active motif set
    pub fn reactivate_set<'a>(&self, data: &'a Waveform, background: &'a Background) -> MotifSet<'a> {

        let mut revived = MotifSet {

            set: self.set.clone(),
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
    fn from(other: &'a MotifSet) -> StrippedMotifSet {
        let ln_post: f64 = other.calc_ln_post();
        StrippedMotifSet {
            set: other.set.clone(),
            ln_post: ln_post,
        }
    }
}

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
    pub fn get_motif_set<'a, R: Rng + ?Sized>(self, recalculate: bool, validate_motifs: bool, randomizer: &mut Option<&mut R>, data: &'a Waveform, background: &'a Background ) -> MotifSet<'a> {
        
        let mut set = self.set.clone();

        //We fix up any kernels that don't match the width defined in our MotifSet
        let mut scrambled_motifs = false;
        for mot in set.iter_mut() {

            if validate_motifs && mot.pwm_prior(data.seq()) == f64::NEG_INFINITY {
                scrambled_motifs = true;
                _ = mot.scramble_to_close_random_valid(data.seq(), randomizer);
            }
 
        }

        //We check to see if we need to recalculate the wave signal.
        let signal_will_make_ub =  (self.signal.len() != data.read_wave().len()) || (self.signal.spacer() != data.spacer());
        let redo_signal = recalculate || scrambled_motifs || signal_will_make_ub;

        let signal = if !redo_signal { //Notice how we carefully checked to make sure that signal follows our safety guarentees
            unsafe {self.signal.get_waveform(data.point_lens(), data.start_dats(), data.seq())}
        } else {
            data.derive_zero() //This isn't our final signal. This just ensures our safety guarentees
        };
        
        let mut pre_set = MotifSet {
            set: set,
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
    all_data_file: String,
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
    pub fn new_empty(capacity: usize, all_data_file: String, data: &'a Waveform<'a>, background: &'a Background) -> SetTrace<'a> {


        SetTrace{
            trace: Vec::<AnyMotifSet<'a>>::with_capacity(capacity),
            all_data_file: all_data_file,
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

        const RANGE_RJ: usize = MAX_IND_RJ+1;
        const RANGE_LEAP: usize = MAX_IND_LEAP+1;

        let (new_set, retval): (MotifSet, (usize,bool)) = match select_move {

            0..=MAX_IND_RJ => {
                let (set, move_type, accept) = setty.run_rj_move(rng);
                (set, (move_type, accept))
            },
            RANGE_RJ..=MAX_IND_LEAP => {
                let set = setty.base_leap(rng);
                (set, (4, true))
            },
            RANGE_LEAP..=MAX_IND_HMC => {
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

        let recalc_ln_post = (!std::ptr::eq(self.data, repoint_set.data)) || 
                             (!std::ptr::eq(self.background, repoint_set.background)) ;
        if recalc_ln_post {
            repoint_set.data = &self.data;
            repoint_set.background = &self.background;
        }

        if recalc_ln_post {
            repoint_set.signal = self.data.derive_zero();
            for i in 0..repoint_set.set.len() {
                repoint_set.signal += &repoint_set.set[i].generate_waveform(&self.data, self.background.kernel_ref());
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
    #[allow(dead_code)]
    fn push_set_def_many<R: Rng + ?Sized>(&mut self, sets: Vec<AnyMotifSet<'a>>) {
        self.trace.reserve(sets.len());
        for set in sets {
            self.trace.push(set);
        }
    }


    pub fn current_set(&self) -> MotifSet<'a> {
        self.trace[self.trace.len()-1].give_activated(self.data, self.background)
    }

    pub fn current_set_to_print(&self) -> StrippedMotifSet {
        self.trace[self.trace.len()-1].give_stored()
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
                fs::write(savestate_file.as_str(), serde_json::to_string(&init_set).unwrap()).expect("Need to give a valid file to write to or inference is pointless");
            },
            AnyMotifSet::Passive(set) => {fs::write(savestate_file.as_str(), 
                                                    serde_json::to_string(&set).unwrap()).expect("Need to give a valid file to write to or inference is pointless");},
        };
    }
       
    pub fn save_and_drop_history(&mut self, output_dir: &str, run_name: &str, zeroth_step: usize) {

        let len_trace = self.trace.len();

        //We want to keep the last element in the SetTrace, so that the markov chain can proceed
        let trace: Vec<StrippedMotifSet> = self.trace.drain(0..(len_trace-1)).map(|a| a.give_stored()).collect();


        let history = SetTraceDef {
            all_data_file: self.all_data_file.clone(),
            trace: trace, 
        };

        let trace_file: String = format!("{}/{}_trace_from_step_{:0>7}.json",output_dir,run_name,zeroth_step);

        fs::write(trace_file.as_str(), serde_json::to_string(&history).unwrap()).expect("Need to give a valid file to write to or inference is pointless");

        self.save_initial_state(output_dir, run_name);
    }

    pub fn save_trace(&self, output_dir: &str, run_name: &str, zeroth_step: usize) {

        let current_active = self.current_set();
    
        let locs = self.data.generate_all_locs();

        let signal_file: String = format!("{}/{}_occupancy_signal_{:0>7}.png",output_dir,run_name,zeroth_step);

        let plot = BitMapBackend::new(&signal_file, (3000, 1000)).into_drawing_area();
       
        plot.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&plot)
            .set_label_area_size(LabelAreaPosition::Left, 40)
            .set_label_area_size(LabelAreaPosition::Bottom, 40)
            .caption("Signal Comparison", ("sans-serif", 40))
            .build_cartesian_2d(0_f64..(*locs.last().unwrap() as f64), (-16_f64)..16_f64).unwrap();


        chart.configure_mesh().draw().unwrap();

        chart.draw_series(current_active.data.read_wave().iter().zip(locs.iter()).map(|(&k, &i)| Circle::new((i as f64, k),1_u32, &BLACK))).unwrap().label("Data Wave");
       
        let current_resid = current_active.data-&current_active.signal;
        chart.draw_series(LineSeries::new(current_resid.read_wave().iter().zip(locs.iter()).map(|(&k, &i)| (i as f64, k)), &BLUE)).unwrap().label("Occpuancy Trace");

        let max_draw = current_active.set.len().min(3);

        let colors = [&RED, &GREEN, &CYAN];

        for i in 0..max_draw {

            let occupancy = current_active.set[i].generate_waveform(current_active.data, current_active.background.kernel_ref());
            chart.draw_series(LineSeries::new(occupancy.read_wave().iter().zip(locs.iter()).map(|(&k, &i)| (i as f64, k)), colors[i])).unwrap().label(format!("Motif {}", i).as_str());

        }

        chart.configure_series_labels().border_style(&BLACK).draw().unwrap();

    }


    pub fn trace_from_meme<R: Rng + ?Sized>(&mut self, meme_file_name: &str, seq: &Sequence, e_value_cutoff: f64,  rng: &mut R) {

        let meme_file_string = fs::read_to_string(meme_file_name).expect("Invalid FASTA file name!");
        let meme_as_vec = meme_file_string.split("\n").collect::<Vec<_>>();

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

            let mut motif = Motif::rand_height_pwm(base_vec, rng);
           
            let poss_hamming = motif.scramble_to_close_random_valid(seq, &mut Some(rng));

            match poss_hamming {
                Some(hamming) => warn!("{}", format!("Motif number {} from the MEME file does not exist in the parts of the sequence with peaks! Moving it to a valid motif within a Hamming distance of {}!", mot_num, hamming)),
                None => (),
            };


            set.push(motif);    

        }


        let mut signal = self.data.derive_zero();

        for mot in &set {
            signal += &(mot.generate_waveform(self.data, self.background.kernel_ref()));
        }

        let mut full_set = MotifSet {
            set: set,
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
    all_data_file: String,

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
            all_data_file: self.all_data_file.clone(),
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
        
        let check_data = fs::read_to_string(self.all_data_file.as_str())?;

        let full_data: AllData = serde_json::from_str(check_data.as_str())?;

        Ok(self.trace.iter().map(|a| a.ln_post-a.ln_prior(full_data.seq())).collect::<Vec<f64>>())

    }

    pub fn motif_num_trace(&self) -> Vec<f64> {
        self.trace.iter().map(|a| a.set.len() as f64).collect::<Vec<_>>()
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
 
    pub fn extract_best_motif_per_set(&self, reference_motif: &Motif, tail_start: usize, cutoff: f64) -> Vec<(Motif, (f64, isize, bool))> {

        self.trace[(self.len()-tail_start)..self.len()].iter().map(|mot_set| {

           mot_set.set.iter().map(|mot| (mot.clone(),mot.distance_function(&reference_motif)))
                             .min_by(|a,b| a.1.0.partial_cmp(&b.1.0).expect("No NANs should be present in distances"))
                             .expect("Motif sets all need at least one motif")


       }).filter(|(_, (b, _, _))| *b < cutoff).collect::<Vec<_>>()

    }

    pub fn append(&mut self, mut attachment: SetTraceDef) {
        self.trace.append(&mut attachment.trace);
    }

    pub fn index_into(&self, range: std::ops::Range<usize>) -> &[StrippedMotifSet] {
        &self.trace[range]
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
            self.d_ln_pdf(x)//*(x-self.min)*(self.max-x)/(SPREAD_HMC_CONV*(self.max-self.min))
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

    use std::time::{Duration, Instant};
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
    use rand::distributions::{Distribution, Uniform};

    fn produce_bps_and_pos(seq_motif: &Vec<Bp>) -> (Vec<Bp>, Vec<usize>) {

        let mut bps: Vec<Bp> = Vec::with_capacity(seq_motif.len()*(BASE_L-1));
        let mut pos: Vec<usize> = Vec::with_capacity(seq_motif.len()*(BASE_L-1));
        
        for i in 0..seq_motif.len() {
            let mut b = Bp::A;
            let mut bp_vec = BP_ARRAY.to_vec();
            bp_vec.retain(|&b| b != seq_motif[i]);
            for b in bp_vec {
                bps.push(b);
                pos.push(i);
            }
        }
        (bps, pos)
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
        

        println!("{:?} {:?}", bp_vec, usize_vec);

        let bad_safe_access = Instant::now();
        let accessed_c: Vec<f64> = usize_vec.iter().map(|&b| base.props[b]).collect();
        let bad_safe_time = bad_safe_access.elapsed();
        
        let safe_access = Instant::now();
        let accessed: Vec<f64> = bp_vec.iter().map(|&b| base[b]).collect();
        let safe_time = safe_access.elapsed();
        
        let unsafe_access = Instant::now();
        let accessed_b: Vec<f64> = usize_vec.iter().map(|&b| unsafe{base.rel_bind(b)}).collect();
        let unsafe_time = unsafe_access.elapsed();


        println!("{:?}", accessed);
        println!("{:?}", accessed_b);
        println!("{:?}", accessed_c);
        println!("safe {:?} unsafe {:?} bad_safe: {:?}", safe_time, unsafe_time, bad_safe_time);
    }

    #[test]
    fn simplex_test() {
        let mut rng = rand::thread_rng();
        let b = Base::rand_new(&mut rng);

        let simplex = b.as_simplex();
        
        let mod_b = Base::simplex_to_base(&simplex);

        println!("{:?} {:?} {:?} {} {} {}", b, simplex, mod_b, b.dist_sq(None), b.dist_sq(Some(&b)), simplex.iter().map(|&a| a.powi(2)).sum::<f64>().sqrt());

        let mot = Motif::from_motif(vec![Bp::A,Bp::C, Bp::T, Bp::G, Bp::C,Bp::T, Bp::T, Bp::A, Bp::C] , &mut rng);

        
        let jacob = b.d_base_d_simplex();

        let b0 = b.add_in_hmc([1e-6, 0.0, 0.0], false);
        let b1 =  b.add_in_hmc([0.0, 1e-6, 0.0], false);
        let b2 =  b.add_in_hmc([0.0, 0.0, 1e-6], false);

        println!("{:?} {:?} {} {} {}", b.as_simplex(), b0.as_simplex(), b0.as_simplex()[0]-b.as_simplex()[0],  b0.as_simplex()[1]-b.as_simplex()[1],  b0.as_simplex()[2]-b.as_simplex()[2]);
        println!("{:?} {:?} {} {} {}", b.as_simplex(), b1.as_simplex(), b1.as_simplex()[0]-b.as_simplex()[0],  b1.as_simplex()[1]-b.as_simplex()[1],  b1.as_simplex()[2]-b.as_simplex()[2]);
        println!("{:?} {:?} {} {} {}", b.as_simplex(), b2.as_simplex(), b2.as_simplex()[0]-b.as_simplex()[0],  b2.as_simplex()[1]-b.as_simplex()[1],  b2.as_simplex()[2]-b.as_simplex()[2]);

        println!("{:?}", jacob);
        println!("{:?} {:?} {:?}", b0.props.iter().zip(b.props.iter()).map(|(&a, &b)| (a-b)/1e-6).collect::<Vec<_>>(), b1.props.iter().zip(b.props.iter()).map(|(&a, &b)| (a-b)/1e-6).collect::<Vec<_>>(),
        b2.props.iter().zip(b.props.iter()).map(|(&a, &b)| (a-b)/1e-6).collect::<Vec<_>>());

        let simp = [0.1_f64, -0.1, -0.2];
        let simp_b = Base::simplex_to_base(&simp);

        let b3a = simp_b.add_in_hmc([0.0, 0.0, -1./30.], false);
        let b3 = simp_b.add_in_hmc([0.0, 0.0, -5./30.], false);

        let sim_b3a = b3a.as_simplex();
        let sim_b3 = b3.as_simplex();

        println!("sim {:?} simb {:?} noref {:?} norefb {:?} ref {:?} refb{:?}", simp, simp_b, b3a.as_simplex(), b3a, b3.as_simplex(), b3);

        assert!((simp[0]-sim_b3a[0]).abs() < 1e-6, "0th element changes with no reflection");
        assert!((simp[0]-sim_b3[0] ).abs() < 1e-6, "0th element changes with a reflection");
        
        assert!((simp[1]-sim_b3a[1]).abs() < 1e-6, "1st element changes with no reflection");
        assert!((simp[1]-sim_b3[1] ).abs() < 1e-6, "1st element changes with a reflection");

        assert!((-0.23333333333333333333333333-sim_b3a[2]).abs() < 1e-6, "2nd element incorrect with no reflection");
        assert!((-0.3-sim_b3[2]).abs() < 1e-6, "2nd element incorrect with a reflection");
    
        let b_a = b.add_in_hmc([10.0, -10.0, 1./3.], false);

        println!("b_a {:?}", b_a);
  
        let b_b = b.add_in_hmc([10.0, -10.0, 1./3.], true);

        println!("best o {:?} best nr {:?} best r {:?}",b.best_base(),b_a.best_base(),b_b.best_base());

        println!("b {:?}, nr {:?}, r {:?}", b, b_a, b_b);

        let confine_start = Base::simplex_to_base(&[1_f64/3., 0., 0.]);


        let mom_confine_ref = [0_f64, 0., SQRT_2/3.];

        //let confine_flip = [0_f64, 0., - SQRT_2/3.];
        println!("pre ref");
        let confine_end = confine_start.add_in_hmc(mom_confine_ref, true);

        let confine_end_simp = confine_end.as_simplex();

        //let should_start = confine_end.add_in_hmc(confine_flip, true);
        println!("end simp {:?} prop_end_simp {:?}", confine_end_simp, [5_f64/9., 0., SQRT_2/3.-(2.*SQRT_2)/9.]);
        
        //println!("start {:?} back_to_start {:?}", confine_start, should_start);



    }

    #[test]
    fn gradient_test() {


        let mut rng = rand::thread_rng(); //fastrand::Rng::new();

        let block_n: usize = 3;
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
        let block_u8_starts: Vec<usize> = (0..block_n).map(|a| a*u8_per_block).collect();
        let block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
        let mut start_bases: Vec<usize> = (0..block_n).map(|a| a*bp_per_block).collect();
        let sequence: Sequence = Sequence::new_manual(blocks, block_lens.clone());
        let wave: Waveform = Waveform::create_zero(&sequence, 5);
        let duration = start_gen.elapsed();
        println!("grad gen {} bp {:?}", bp, duration);

        let corrs: Vec<f64> = vec![0.9, -0.1];
        let background = Background::new(0.25, 2.64, 350./6.0, Some(&corrs));
        let mut motif_set = MotifSet::rand_with_one_height(13.2, &wave, &background, &mut rng);

        _ = motif_set.add_motif(Motif::rand_mot_with_height(13.2, wave.seq(), &mut rng));
     

        println!("CV");
        let analytical_grad = motif_set.gradient();
        println!("AG");
        let numerical_grad = motif_set.numerical_gradient();
        println!("NG");

        let b: Base = Base::new([0.1, 0.2, 0.3, 0.4]);

        let b2 = b.add_in_hmc([0.0; 3], false);

        println!("2");
        let b3 = b.add_in_hmc([0.0001, 0.0, 0.0], false);

        println!("b {:?} b2 {:?} b3 {:?}", b, b2, b3);

        let mot = motif_set.get_nth_motif(0);
        let (bps, pos) = produce_bps_and_pos(&(mot.best_motif()));
        let mot_a = motif_set.get_nth_motif(1);
        let (bps_a, pos_a) = produce_bps_and_pos(&(mot_a.best_motif()));
        let bind1 = mot.return_bind_score(&wave.seq());
        let wave1 = unsafe{mot.generate_waveform_from_binds(&bind1, &wave, background.kernel_ref())};
        let mom = vec![0.0_f64; 1+(mot.len()*(BASE_L-1))];

        let h = 1e-6;

        println!("Mot \n {}", mot);
        println!("Mot_a \n {}", mot_a);
        for (j, (&bp, &p)) in bps.iter().zip(pos.iter()).enumerate() {
        
            let wave_check = unsafe{mot.only_pos_waveform_from_binds( &bind1, bp, p, &wave, background.kernel_ref())};
            
            let n1 = wave_check.produce_noise(&wave, &background);
            let mut add_vec = wave_check.derive_zero().raw_wave();

            add_vec[2] = h;
            let dwave = Waveform::new(add_vec, &sequence, 5);

            let n2 = (&wave_check+&dwave).produce_noise(&wave, &background);

            

            let resid_care: Vec<f64> = n1.resids()[0..3].iter().zip(n2.resids()[0..3].iter()).map(|(&a, &b)| (b-a)/h).collect();
            let prop_resid = dwave.account_auto(&background);

            println!("j: {} pos: {} bp: {:?}", j, p, bp);
            println!("Resid change: {:?} {:?}", resid_care, prop_resid.resids()[0..3].iter().map(|&a| a/h).collect::<Vec<_>>());
            let mut mommy = mom.clone();
            mommy[j+1] = h;
        
            let mot2 = unsafe{mot.add_momentum(1.0, &mommy, false)};
       
            let wave2 = mot2.generate_waveform(&wave, background.kernel_ref());

            let prop_bp = unsafe{mot.pwm()[p][bp]};
            let n_diff: Vec<f64> = wave2.produce_noise(&wave, &background).resids().iter()
                              .zip(wave1.produce_noise(&wave, &background).resids().iter())
                              .map(|(&a, &b)| (a-b)/h).collect();
                     
            let w1 = wave1.raw_wave();
            let w2 = wave2.raw_wave();
            let wc = wave_check.raw_wave();

            let prop_bp2 = unsafe{mot2.pwm()[p][bp]};
            let mut i = 0;
            let mut calc: f64 = (w2[i]-w1[i])/h;
            let mut ana: f64 = wc[i];
            println!("Wave checking {:?} in pos {}", bp, p);
            println!("bp0: {}, bp1: {}, dbp: {}, analytical dbp: {}", prop_bp, prop_bp2, (prop_bp2-prop_bp)/h, 1.0);
            let ratio = (prop_bp2-prop_bp)/h;
            println!("{} {} {} {} diffs", unsafe{mot2.pwm()[p][Bp::A]-mot.pwm()[p][Bp::A]}, unsafe{mot2.pwm()[p][Bp::C]-mot.pwm()[p][Bp::C]},
                                          unsafe{mot2.pwm()[p][Bp::G]-mot.pwm()[p][Bp::G]}, unsafe{mot2.pwm()[p][Bp::T]-mot.pwm()[p][Bp::T]});       
            while (i < (w2.len()-2)) && ((calc == 0.) || (ana == 0.)) {
                calc = (w2[i]-w1[i])/h;
                ana = wc[i];
                i += 1;
            }
            
            let calc_b = (w2[i+1]-w1[i+1])/h;
            let ana_b = wc[i+1];


            let num_diff = calc;
            let num_diff_b = calc_b;
            println!("{} {} {} {} {} {} {} {}", i,  num_diff, ana, (num_diff/ana), (num_diff_b/ana_b), 1.0, (calc/ana),((calc/ana)));

        }

        


        let mut grad_reses: Vec<Result<(), String>> = Vec::with_capacity(analytical_grad.len());
        println!("{} {} {} {}", analytical_grad.len(), mot.len()*(BASE_L-1)+1, mot_a.len()*(BASE_L-1)+1, mot.len()*(BASE_L-1)+1+mot_a.len()*(BASE_L-1)+1);
        println!("Analytical    Numerical    Difference(abs)    Quotient    Prior par");
        for i in 0..analytical_grad.len() {
            if i == 0 || i == ((BASE_L-1)*mot.len()+1) {
                println!("height!");
                let mot_ref = if i == 0 { &mot } else {&mot_a};
                println!("{} {} {} {} {}",i, analytical_grad[i], numerical_grad[i], numerical_grad[i]-analytical_grad[i], mot_ref.d_height_prior_d_hmc());
            } else if i < ((BASE_L-1)*mot.len()+1) {
                println!("{} {} {} {} {}",i,  analytical_grad[i], numerical_grad[i], numerical_grad[i]-analytical_grad[i], ((numerical_grad[i]-analytical_grad[i])/numerical_grad[i]));
            } else {
                println!("{} {:?}", pos_a[i-((BASE_L-1)*mot.len()+2)], bps_a[i-((BASE_L-1)*mot.len()+2)]);
                println!("{} {} {} {} {}",i, analytical_grad[i], numerical_grad[i], numerical_grad[i]-analytical_grad[i], ((numerical_grad[i]-analytical_grad[i])/numerical_grad[i]));
            }
            //let success = if (numerical_grad[i].abs() != 0.) {(((numerical_grad[i]-analytical_grad[i])/numerical_grad[i]).abs() < 1e-2)} else {(numerical_grad[i]-analytical_grad[i]).abs() < 1e-3};
            let success = ((numerical_grad[i]-analytical_grad[i]).abs() < 5e-2) ||  ((numerical_grad[i]-analytical_grad[i]).abs()/numerical_grad[i] < 2e-3);
            if success {
                grad_reses.push(Ok(()));
            } else {
                let zeroth_mot = i <= mot.len()*(BASE_L-1);
                let mot_num = if zeroth_mot {0} else {1};
                let ind = if zeroth_mot {i} else {i - ((mot.len()*(BASE_L-1))+1)};
                
                let loc_message = if ind == 0 { "height".to_owned() } else {
                    let bp = if zeroth_mot {bps[ind-1]} else {bps_a[ind-1]};
                    let po = if zeroth_mot {pos[ind-1]} else {pos_a[ind-1]};
                    format!("binding at position {} of base {:?}", po, bp)};

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
        println!("Epsilon {} D hamiltonian {} acc {} \n old_set: {:?} \n new_set: {:?}", HMC_EPSILON, dham, acc, motif_set,new_set);

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
        let block_u8_starts: Vec<usize> = (0..block_n).map(|a| a*u8_per_block).collect();
        let block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
        let mut start_bases: Vec<usize> = (0..block_n).map(|a| a*bp_per_block).collect();
        let sequence: Sequence = Sequence::new_manual(blocks, block_lens.clone());
        let wave: Waveform = Waveform::create_zero(&sequence, 5);
        let duration = start_gen.elapsed();
        println!("grad gen {} bp {:?}", bp, duration);

        let corrs: Vec<f64> = vec![0.9, -0.1];
        let background = Background::new(0.25, 2.64, 350./6.,Some(&corrs));
        let mut motif_set = MotifSet::rand_with_one_height(-9.6, &wave, &background, &mut rng);

        _ = motif_set.add_motif(Motif::rand_mot_with_height(13.2, wave.seq(), &mut rng));
     
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
            all_base_correct &= (mot_scram.pwm[base][scram_kmer[base]] == 1.0);
            let best_old = mot_kmer[base];
            let best_new = scram_kmer[base];
            for bp in BP_ARRAY {

                if bp == best_old {
                    all_scramble_correct &= unsafe{mot_scram.pwm[base][bp] == mot.pwm[base][best_new]};
                } else if bp == best_new {
                    all_scramble_correct &= unsafe{mot_scram.pwm[base][bp] == mot.pwm[base][best_old]};
                } else {
                    all_scramble_correct &= unsafe{mot_scram.pwm[base][bp] == mot.pwm[base][bp]};
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
            for bp in BP_ARRAY {
                if bp == best_old {
                    all_scramble_correct &= unsafe{leap.pwm[base][bp] == mot.pwm[base][best_new]};
                } else if bp == best_new {
                    all_scramble_correct &= unsafe{leap.pwm[base][bp] == mot.pwm[base][best_old]};
                } else {
                    all_scramble_correct &= unsafe{leap.pwm[base][bp] == mot.pwm[base][bp]};
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
                    all_scramble_correct &= unsafe{leap1.pwm[base][bp] == mot1.pwm[base][best_new]};
                } else if bp == best_new {
                    all_scramble_correct &= unsafe{leap1.pwm[base][bp] == mot1.pwm[base][best_old]};
                } else {
                    all_scramble_correct &= unsafe{leap1.pwm[base][bp] == mot1.pwm[base][bp]};
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
        let block_u8_starts: Vec<usize> = (0..block_n).map(|a| a*u8_per_block).collect();
        let block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
        let mut start_bases: Vec<usize> = (0..block_n).map(|a| a*bp_per_block).collect();
        let sequence: Sequence = Sequence::new_manual(blocks, block_lens.clone());
        let wave: Waveform = Waveform::create_zero(&sequence, 5);
        let duration = start_gen.elapsed();
        println!("grad gen {} bp {:?}", bp, duration);

        let corrs: Vec<f64> = vec![0.9, -0.1];
        let background = Background::new(0.25, 2.64, 350./6., Some(&corrs));
        let mut motif_set = MotifSet::rand_with_one_height(-9.6, &wave, &background, &mut rng);

        let check_set = motif_set.clone();

        let mid_mot = (MIN_BASE+MAX_BASE)/2;
        //TESTING THE MANIPULATOR FUNCTIONS: these should simply mutate the motif set to conform and usually output the new ln posterior
        let add_mot = Motif::rand_mot_with_height_and_motif_len(13.2,mid_mot, wave.seq(), &mut rng);
        
        //Testing: fn add_motif(&mut self, new_mot: Motif) -> f64 
        let new_like = motif_set.add_motif(add_mot.clone());
        
        //I'm comfortable enforcing exactly equality because new_like should literally be a copy of ln_post
        assert!(motif_set.ln_post == Some(new_like));

        let wave_diff = &motif_set.signal-&check_set.signal;

        let check_diff = &wave_diff-&add_mot.generate_waveform(motif_set.data, background.kernel_ref());

        let all_good = check_diff.read_wave().iter().map(|&a| a.abs() < 1024.0*std::f64::EPSILON).fold(true, |f, x| f && x);

        assert!(all_good);

        assert!(motif_set.set.len() == 2);

        println!("add motif correct wave and like");

        //Testing fn remove_motif(&mut self, rem_id: usize) -> f64
        let new_like = motif_set.remove_motif(1);
        
        assert!((motif_set.calc_ln_post()- check_set.calc_ln_post()).abs() < 1e-8, "{}", format!("{:?} {:?}", motif_set.ln_post, check_set.ln_post));

        let wave_diff = &motif_set.signal-&check_set.signal;

        //let check_diff = &wave_diff-&add_mot.generate_waveform(motif_set.data, background.kernel_ref());

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

        let check_diff = &wave_diff-&add_mot.generate_waveform(motif_set.data, background.kernel_ref());

        let all_good = check_diff.read_wave().iter().map(|&a| a.abs() < 1024.0*std::f64::EPSILON).fold(true, |f, x| f && x);

        assert!(all_good);

        //I'm comfortable enforcing exactly equality because this should be an exact copy
        assert!(motif_set.set[0].peak_height() == add_mot.peak_height());


        assert!(motif_set.set.len() == 2);

        //Testing fn replace_motif(&mut self, new_mot: Motif, rem_id: usize) -> f64

        let add_mot2 = Motif::rand_mot_with_height(-6.2, wave.seq(), &mut rng);
        
        let new_like = motif_set.replace_motif(add_mot2.clone(), 0);

        assert!(motif_set.ln_post == Some(new_like));

        let bring_wave_back = &motif_set.signal -&add_mot2.generate_waveform(motif_set.data, background.kernel_ref());

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

        let remaining = motif_set.data-&motif_set.signal;
        let propensities = remaining.kmer_propensities(birth_mot.set[l].len());
        let pick_prob = propensities[motif_set.data.seq().id_of_u64_kmer_or_die(birth_mot.set[l].len(),Sequence::kmer_to_u64(&birth_mot.set[l].best_motif()))]/propensities.iter().sum::<f64>();


        let actual_prior = birth_mot.set[l].height_prior()+pick_prob.ln()-((MAX_BASE+1-MIN_BASE) as f64).ln();

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
        
        let remaining = motif_set.data-&death_mot.signal;
        let propensities = remaining.kmer_propensities(motif_set.set[l].len());
        let pick_prob = propensities[motif_set.data.seq().id_of_u64_kmer_or_die(motif_set.set[l].len(),Sequence::kmer_to_u64(&motif_set.set[l].best_motif()))]/propensities.iter().sum::<f64>();



        let actual_prior = motif_set.set[l].height_prior()+pick_prob.ln()-((MAX_BASE+1-MIN_BASE) as f64).ln();

        println!("priors {} {} {} {}", motif_set.set[l].height_prior(), pick_prob.ln(), ((MAX_BASE+1-MIN_BASE) as f64).ln(), propensities[motif_set.data.seq().id_of_u64_kmer_or_die(motif_set.set[l].len(),Sequence::kmer_to_u64(&motif_set.set[l].best_motif()))]);

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

        let actual_prior = PROPOSE_EXTEND.ln_pdf(&(extend_mot.set[l.unwrap()].pwm.last().expect("We know this is bigger than 0").as_probabilities().to_vec()).try_into().unwrap());

        assert!((should_prior+actual_prior).abs() < 1e-6, "{}", format!("{}", should_prior+actual_prior).as_str());
 
        let cant_extend_set = MotifSet::rand_with_one_height_and_motif_len(-9.6, MAX_BASE, &wave, &background, &mut rng);

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

        let actual_prior = PROPOSE_EXTEND.ln_pdf(&(motif_set.set[l.unwrap()].pwm.last().expect("We know this is bigger than 0").as_probabilities().to_vec()).try_into().unwrap());

        assert!((should_prior-actual_prior).abs() < 1e-6, "{}", format!("{}", should_prior-actual_prior).as_str());

        let cant_contract_set = MotifSet::rand_with_one_height_and_motif_len(-9.6, MIN_BASE, &wave, &background, &mut rng);

        assert!(cant_contract_set.propose_contract_motif(&mut rng).is_none(), "Can contract PWM beyond allowed limits!");



        //Testing full RJ move
        for _ in 0..3{
            let add_mot = Motif::rand_mot_with_height_and_motif_len(13.2,mid_mot, wave.seq(), &mut rng);
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

        let b_mag: f64 = b.show().iter().sum();
        let supposed_default_dist = (b.as_simplex()).iter().map(|a| a.powi(2)).sum::<f64>();

        assert!(supposed_default_dist == b.dist_sq(None));
      
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

        let block_n: usize = 5;
        let u8_per_block: usize = 90;
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
        let corrs: Vec<f64> = vec![0.9, -0.1];
        let background = Background::new(0.25, 2.64, 20., Some(&corrs));
        println!("Done gen {} bp {:?}", bp, duration);

        println!("{} gamma", gamma(4.));
        println!("{} gamma", ln_gamma(4.));

        //println!("{:?}", wave.raw_wave());

        let motif: Motif = Motif::from_motif(sequence.return_bases(0,0,20), &mut rng); //sequence

        let motif2: Motif = Motif::from_motif(sequence.return_bases(0,2,20), &mut rng); //sequence

        let start = Instant::now();

        let waveform = motif.generate_waveform(&wave, background.kernel_ref());
        let duration = start.elapsed();
        
        let waveform2 = &waveform + &(motif2.generate_waveform(&wave, background.kernel_ref()));

        let noise: Noise = waveform.produce_noise(&waveform2, &background);

        let d_ad_stat_d_noise = noise.ad_grad();
        let d_ad_like_d_ad_stat = Noise::ad_deriv(noise.ad_calc());


        let grad = unsafe{ motif.parallel_single_motif_grad(&waveform2, &d_ad_stat_d_noise, d_ad_like_d_ad_stat, &background)};

        let waveform_raw = waveform.raw_wave();

        let binds = motif.return_bind_score(&sequence);

        let start_b = Instant::now();
        let unsafe_waveform = unsafe{ motif.generate_waveform_from_binds(&binds, &wave, background.kernel_ref()) };
        let duration_b = start_b.elapsed();

        let unsafe_raw = unsafe_waveform.raw_wave();

        assert!(unsafe_raw.len() == waveform_raw.len());
        assert!((unsafe_raw.iter().zip(&waveform_raw).map(|(a, &b)| (a-b).powf(2.0)).sum::<f64>()).powf(0.5) < 1e-6);
        
        println!("Time elapsed in generate_waveform() is: {:?}", duration);
        println!("Time elapsed in the unsafe generate_waveform() is: {:?}", duration_b);

        println!("{}", motif);

        let random_motif = Motif::rand_mot(&sequence, &mut rng);

        println!("Random motif\n{}", random_motif);

        assert!(background.kernel_ref().len() == 121);

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

                let defect: f64 = unsafe{ pwm[i][j]} ;

                
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
        let wave_background = Background::new(0.25, 2.64, 1.0, Some(&corrs));

        let theory_base = [1.0, 1e-5, 1e-5, 0.2];

        let mat: Vec<Base> = (0..15).map(|_| Base::new(theory_base.clone())).collect::<Vec<_>>();


        println!("DF");
        let little_motif: Motif = Motif::raw_pwm(mat, 10.0); //wave_seq

        print!("{}", little_motif);
        println!("{:?}",little_motif.generate_waveform(&wave_wave, wave_background.kernel_ref()).raw_wave());

        let small_block: Vec<u8> = vec![44, 24, 148, 240, 84, 64, 200, 80, 68, 92, 196, 144]; 
        let small_inds: Vec<usize> = vec![0, 6]; 
        let small_lens: Vec<usize> = vec![24, 24];
        let small: Sequence = Sequence::new_manual(small_block, small_lens);
        let small_wave: Waveform = Waveform::new(vec![0.1, 0.6, 0.9, 0.6, 0.1, -0.2, -0.4, -0.6, -0.6, -0.4], &small, 5);

        let mat: Vec<Base> = (0..15).map(|_| Base::new(theory_base.clone())).collect::<Vec<_>>();
        let wave_motif: Motif = Motif::raw_pwm(mat, 10.0); //small

        let rev_comp: Vec<bool> = (0..48).map(|_| rng.gen::<bool>()).collect();

        let checked = wave_motif.base_check( &small, &rev_comp, Bp::A, 4);

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
                    let matcher = if rev_comp[24*i+j] { bp == Bp::T } else { bp == Bp::A};
                    println!("start loc: {}, bp: {:?}, bp2: {:?}, ind: {}, rev: {}, matcher: {}, checked: {}, correct: {}", 24*i+j, bp, bp2, ind, rev_comp[24*i+j], matcher, checked[24*i+j], correct[24*i+j]);
                    assert!(checked[24*i+j] == matcher);
                }
            }
        }


        let start = Instant::now();

        //        let (bind_score_floats, bind_score_revs) = self.return_bind_score(data.seq());

        let binds = motif.return_bind_score(&sequence);

        let duration = start.elapsed();
        println!("Time elapsed in bind_score() is: {:?}", duration);

        let start = Instant::now();
        let checked = motif.base_check(&sequence, &binds.1, Bp::G, 4);
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


        let pwm_bp: Bp = Bp::T;
        let pwm_pos: usize = 6;
        let prop_bp = motif.pwm[pwm_pos][pwm_bp];
        let wave_main = motif.generate_waveform(&wave, background.kernel_ref());
        let start0 = Instant::now();
        let wave_noh = motif.no_height_waveform(&wave, background.kernel_ref());
        let duration0 = start.elapsed();
        let wave_gen: Vec<f64> = wave_main.raw_wave();
        let wave_sho: Vec<f64> = wave_noh.raw_wave();

        let checked = motif.base_check(&sequence, &binds.1, Bp::T, 6);

        let start = Instant::now();
        let wave_filter = motif.only_pos_waveform(Bp::T, 6, &wave, background.kernel_ref());
        let duration = start.elapsed();

        let raw_filter: Vec<f64> = wave_filter.raw_wave();

        let point_lens = wave_main.point_lens();
        start_bases.push(bp);
        let start_dats = wave_main.start_dats();
        let space: usize = wave_main.spacer();

        let half_len: usize = (motif.len()-1)/2;
        
        let kernel_check = (background.kernel_ref()*motif.peak_height()).get_curve().clone();

        let kernel_mid = (kernel_check.len()-1)/2;

        //println!("STARTS: {:?}", sequence.block_u8_starts().iter().map(|a| a*BP_PER_U8).collect::<Vec<_>>());

        let start2 = Instant::now();
        let unsafe_filter = unsafe{motif.only_pos_waveform_from_binds(&binds, Bp::T, 6, &wave, background.kernel_ref())};
        let duration2 = start2.elapsed();

        let start3 = Instant::now();
        let just_divide = unsafe{motif.no_height_waveform_from_binds(&binds, &wave, background.kernel_ref())};
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
