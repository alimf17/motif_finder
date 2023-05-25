//pub mod bases {
use rand::Rng;
use rand::seq::SliceRandom;
use rand::distributions::{Distribution, Uniform, WeightedIndex};
use statrs::distribution::{Continuous, ContinuousCDF, LogNormal, Normal, Dirichlet, Exp};
use statrs::statistics::{Min, Max, Distribution as OtherDistribution};
use statrs::Result as otherResult;
use crate::waveform::{Kernel, Waveform, Waveform_Def, Noise, Background};
use crate::sequence::{Sequence, BP_PER_U8, U64_BITMASK, BITS_PER_BP};
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
pub const BPS: [&str; 4] = ["A", "C", "G", "T"];
pub const BASE_L: usize = BPS.len();
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

const PROB_POS_PEAK: f64 = 0.9;

pub const THRESH: f64 = 1e-3; 

//This is roughly how much an additional motif should improve the ln posterior before it's taken seriously
//The more you increase this, the fewer motifs you will get, on average
const NECESSARY_MOTIF_IMPROVEMENT: f64 = 10.0_f64;

//When converting between gradient compatible and incompatible representations
//We sometimes end up with numerical errors that basically make infinities where there shouldn't be
//CONVERSION_MARGIN protects us during conversion, and REFLECTOR cuts off bases that could get too weak for proper conversion
//These numbers were empirically determined, not theoretically. 
const REFLECTOR: f64 = 15.0;
static PROP_CUTOFF: Lazy<f64> = Lazy::new(|| THRESH); 
static PROP_UPPER_CUTOFF: Lazy<f64> = Lazy::new(|| 1.0-2.0_f64.powf(-19.0));

static BASE_DIST: Lazy<Exp> = Lazy::new(|| Exp::new(1.0).unwrap());

const RJ_MOVE_NAMES: [&str; 4] = ["New motif", "Delete motif", "Extend motif", "Contract Motif"];

//BEGIN BASE

#[derive(Serialize, Deserialize, Clone)]
pub struct Base {
   props: [ f64; BASE_L],
}

impl PartialEq for Base {
    fn eq(&self, other: &Self) -> bool {
        self.dist(Some(other)) < CLOSE    
    }
} 


impl Base {

    //Note: This will break if all the bindings are zeros
    pub fn new(props: [ f64; 4]) -> Base {

        let mut props = props;

        let mut any_neg: bool = false;

        for i in 0..props.len() {
            any_neg |= props[i] < 0.0 ;
        }

        let max = Self::max(&props);
        
        let mut rng = rand::thread_rng();

        //We can rely on perfect float equality because max copies its result
        let mut maxes = props.iter().enumerate().filter(|(_,&a)| (a == max)).map(|(b, _)| b).collect::<Vec<usize>>(); 
        let max_ind = maxes.choose(&mut rng).unwrap();
       

        for i in 0..props.len() {
            if i == *max_ind {
                props[i] = 1.0;
            } else {
                props[i] = (props[i]/max)*(*PROP_UPPER_CUTOFF-*PROP_CUTOFF)+*PROP_CUTOFF 
            }
        }

        if any_neg{  //|| ((norm - 1.0).abs() > 1e-12) 
           panic!("All base entries must be positive!"); 
        }



        Base { props }
    }




    //Safety: props must be an array with 
    //      1) exactly one element = 1.0
    //      2) all other elements between *PROP_CUTOFF and *PROP_UPPER_CUTOFF
    //Otherwise, this WILL break. HARD. And in ways I can't determine
    pub unsafe fn proper_new(props: [ f64; BASE_L]) -> Base {
        Base { props }
    }





    //TODO: Change this to sample a "dirichlet" distribution, but normalized to the maximum
    //      This is done by sampling exp(1) distributions and normalizing by the maximum
    //      Sampling directly from the Dirichlet implementation in statrs was a pain with our constraints
    pub fn rand_new() -> Base {

        let mut rng = rand::thread_rng();
        //let die = Uniform::from(*PROP_CUTOFF..*PROP_UPPER_CUTOFF);

        let mut att: [f64; BASE_L] = [0.0; BASE_L];

        for i in 0..att.len() {
            att[i] = rng.gen();
        }

        let max_ind = rng.gen_range(0..BASE_L);

        for i in 0..att.len() {
            if i == max_ind {
                att[i] = 1.0;
            } else {
                att[i] =  (att[i])*(*PROP_UPPER_CUTOFF-*PROP_CUTOFF)+*PROP_CUTOFF;
            }
        }

        unsafe {Base::proper_new(att) }
    }

    pub fn from_bp(best: usize) -> Base {

        Base::rand_new().make_best(best)

    }

    pub fn make_best(&self, best: usize) -> Base {

        let mut base2 = self.props.clone();

        let which_b = Self::argmax(&base2);

        if best != which_b {

            let tmp = base2[which_b];
            base2[which_b] = base2[best];
            base2[best] = tmp;
        }


        Base::new(base2)
    }

    pub fn best_base(&self) -> usize {
        Self::argmax(&self.props)
    }

    pub fn dist(&self, base: Option<&Base>) -> f64 {

        let magnitude: f64 = self.props.iter().sum();
        let as_probs: [f64; BASE_L] = self.props.iter().map(|a| a/magnitude).collect::<Vec<f64>>().try_into().unwrap();//I'm never worried about error here because all Base are guarenteed to be length BASE_L
        println!("{:?} b1", as_probs);
        match(base) {

            None => as_probs.iter().map(|a| (a-1.0/(BASE_L as f64)).powi(2)).sum::<f64>().sqrt(),
            Some(other) => {
                let magnitude: f64 = other.show().iter().sum();
                as_probs.iter().zip(other.show().iter()).map(|(a, &b)| (a-(b/magnitude)).powi(2)).sum::<f64>().sqrt()
            }
        }

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

        let best = Self::argmax(&self.props);

        let mut new_props = [0.0_f64 ; BASE_L];

        let mut ind: usize = 0;

        for i in 0..self.props.len() {
            if i == best {
                new_props[i] = 1.0;
            } else {
                let mut dg = -((*PROP_UPPER_CUTOFF-*PROP_CUTOFF)/(self.props[i]-*PROP_CUTOFF)-1.0).ln();
                dg += addend[ind];
                dg = *PROP_CUTOFF+((*PROP_UPPER_CUTOFF-*PROP_CUTOFF)/(1.0+((-dg).exp())));
                new_props[i] = dg;
                ind += 1;
            }

        }

        Base {props: new_props}



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


    //bp MUST be less than the BASE_L and nonnegative, or else this will produce undefined behavior
    pub unsafe fn rel_bind(&self, bp: usize) -> f64 {
        //self.props[bp]/Self::max(&self.props) //This is the correct answer
        *self.props.get_unchecked(bp) //Put bases in proportion binding form
        
    }



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
    pub fn raw_pwm(mut pwm: Vec<Base>, peak_height: f64, peak_width: f64) -> Motif {
        let kernel = Kernel::new(peak_width, peak_height);

        pwm.reserve_exact(MAX_BASE-pwm.len());
        let mut m = Motif {
            peak_height: peak_height,
            kernel: kernel,
            pwm: pwm,
        };

        m
    }


    //TODO: make sure that this is used iff there's a guarentee that a motif is allowed
    pub fn from_motif(best_bases: Vec<usize>, peak_width: f64) -> Motif {
        
        let mut pwm: Vec<Base> = Vec::with_capacity(MAX_BASE);

        pwm = best_bases.iter().map(|a| Base::from_bp(*a)).collect();

        let height_dist: TruncatedLogNormal = TruncatedLogNormal::new(LOG_HEIGHT_MEAN, LOG_HEIGHT_SD, MIN_HEIGHT, MAX_HEIGHT).unwrap();

        let mut rng = rand::thread_rng();

        let sign: f64 = rng.gen();
        let sign: f64 = if sign < PROB_POS_PEAK {1.0} else {-1.0};

        let peak_height: f64 = sign*height_dist.sample(&mut rng);

        let kernel = Kernel::new(peak_width, peak_height);

        Motif {
            peak_height: peak_height,
            kernel: kernel,
            pwm: pwm,
        }


    }


    
    pub fn rand_mot(peak_width: f64, seq: &Sequence) -> Motif {

        let mut rng = rand::thread_rng();

        let num_bases = rng.gen_range(MIN_BASE..(MAX_BASE+1));

        let mot = seq.random_valid_motif(num_bases);

        
        Self::from_motif(mot, peak_width)
        

    }

    pub fn make_opposite(&self) -> Motif {

        let mut opposite = self.clone();

        opposite.peak_height = -self.peak_height;
        opposite.kernel = Kernel::new(self.kernel.get_sd(), -self.peak_height);
        opposite

    }

    pub fn scramble_by_id_to_valid(&self, id: usize, opposite: bool, seq: &Sequence) -> Motif {

        let mut new_mot: Motif;
        if opposite {
            new_mot = self.make_opposite();
        } else {
            new_mot = self.clone();
        }
        let new_best: u64 = seq.idth_unique_kmer(self.len(), id);
        let old_best: u64 = Sequence::kmer_to_u64(&self.best_motif());

        for i in 0..self.len() {

            let old_base: usize = (old_best & (U64_BITMASK << (BITS_PER_BP*i))) as usize;
            let new_base: usize = (new_best & (U64_BITMASK << (BITS_PER_BP*i))) as usize;

            if new_base != old_base {
                new_mot.pwm[i].make_best(new_base);
            }

        }

        new_mot

    }

    //Safety: Momentum MUST have a length equal to precisely 1+(BASE_L-1)*self.len()
    unsafe fn add_momentum(&self, eps: f64, momentum: &[f64]) -> Self {

        let mut new_mot = self.clone();

        let mut h = -((MAX_HEIGHT-MIN_HEIGHT)/(self.peak_height.abs()-MIN_HEIGHT)-1.0).ln();
        h += (eps*momentum[0]);
        h = MIN_HEIGHT+((MAX_HEIGHT-MIN_HEIGHT)/(1.0+((-h).exp())));
        new_mot.peak_height = self.peak_height.signum()*h;

        new_mot.kernel = Kernel::new(self.kernel.get_sd(), new_mot.peak_height);

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

    pub fn height_prior(&self) -> f64 {

        let mut prior = if self.peak_height > 0.0 { PROB_POS_PEAK.ln() } else { (1.0-PROB_POS_PEAK).ln() };
        prior += TruncatedLogNormal::new(LOG_HEIGHT_MEAN, LOG_HEIGHT_SD, MIN_HEIGHT, MAX_HEIGHT).unwrap().ln_pdf(self.peak_height.abs());
        prior
    }


    //BINDING FUNCTIONS

    //If kmer is not the same length as the pwm, this will produce undefined behavior
    pub unsafe fn prop_binding(&self, kmer: &[usize]) -> (f64, bool) { 
        

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


        let mut bind_scores: Vec<f64> = vec![0.0; 4*coded_sequence.len()];
        let mut rev_comp: Vec<bool> = vec![false; 4*coded_sequence.len()];

        let mut uncoded_seq: Vec<usize> = vec![0; seq.max_len()];

        let seq_frame = 1+(self.len()/4);


        let mut ind = 0;

        let mut store = Sequence::code_to_bases(coded_sequence[0]);


        {
        let uncoded_seq = uncoded_seq.as_mut_slice();
        for i in 0..(block_starts.len()) {


            for jd in 0..(block_lens[i]/4) {

                store = Sequence::code_to_bases(coded_sequence[block_starts[i]+jd]);
                for k in 0..4 {
                    uncoded_seq[4*jd+k] = store[k];
                }

            }


            for j in 0..((block_lens[i])-self.len()) {

                ind = 4*block_starts[i]+(j as usize);

                
                let binding_borrow = unsafe { uncoded_seq.get_unchecked(j..(j+self.len())) };

                
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

        let (bind_score_floats, bind_score_revs) = self.return_bind_score(DATA.seq());

        for i in 0..starts.len() { //Iterating over each block
            for j in 0..(lens[i]-(self.len()-1)/2) {
                if bind_score_floats[starts[i]*BP_PER_U8+j] > THRESH {
                    actual_kernel = &self.kernel*(bind_score_floats[starts[i]*BP_PER_U8+j]) ;
                    //println!("{}, {}, {}, {}", i, j, lens[i], actual_kernel.len());
                    unsafe {occupancy_trace.place_peak(&actual_kernel, i, j+(self.len()-1)/2);} //Note: this technically means that we round down if the motif length is even
                                                                                         //We CAN technically violate the safety guarentee for place peak, but return_bind_score()
                                                                                         //has zeros where we can be going over the motif length. With THRESH, this preserves safety

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

        let (bind_score_floats, bind_score_revs) = self.return_bind_score(DATA.seq());

        for i in 0..starts.len() { //Iterating over each block
            for j in 0..lens[i] {
                if bind_score_floats[starts[i]*BP_PER_U8+j] > THRESH {
                    //actual_kernel = &base_kernel*(bind_score_floats[starts[i]*BP_PER_U8+j]);
                    actual_kernel = &self.kernel*(bind_score_floats[starts[i]*BP_PER_U8+j]/self.peak_height);
                    unsafe {occupancy_trace.place_peak(&actual_kernel, i, j+(self.len()-1)/2);} //Note: this technically means that we round down if the motif length is even
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

        let checked = self.base_check( DATA.seq(), &bind_score_revs, bp, motif_pos);
        for i in 0..starts.len() { //Iterating over each block
            for j in 0..lens[i] {
                if checked[starts[i]*BP_PER_U8+j] && (bind_score_floats[starts[i]*BP_PER_U8+j] > THRESH) {
                    actual_kernel = &self.kernel*(bind_score_floats[starts[i]*BP_PER_U8+j]);
                    unsafe {occupancy_trace.place_peak(&actual_kernel, i, j+(self.len()-1)/2)}; //Note: this technically means that we round down if the motif length is even
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
            for j in 0..(lens[i]-(self.len()-1)/2) {
                if binds.0[starts[i]*BP_PER_U8+j] > THRESH {
                    actual_kernel = &self.kernel*(binds.0[starts[i]*BP_PER_U8+j]) ;
                    //println!("{}, {}, {}, {}", i, j, lens[i], actual_kernel.len());
                    occupancy_trace.place_peak(&actual_kernel, i, j+(self.len()-1)/2);//UNSAFE //Note: this technically means that we round down if the motif length is even
                                                                                         //We CAN technically violate the safety guarentee for place peak, but return_bind_score()
                                                                                         //has zeros where we can be going over the motif length. With THRESH, this preserves safety

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
            for j in 0..lens[i] {
                if  (binds.0[starts[i]*BP_PER_U8+j] > THRESH) {
                    //actual_kernel = &self.kernel*(binds.0[starts[i]*BP_PER_U8+j]/self.peak_height);
                    actual_kernel = &base_kernel*(binds.0[starts[i]*BP_PER_U8+j]);
                    unsafe {occupancy_trace.place_peak(&actual_kernel, i, j+(self.len()-1)/2)}; //Note: this technically means that we round down if the motif length is even
                }
            }
        }

        occupancy_trace



    }
    //Safety: You MUST ensure that the binding score and reverse complement is valid for this particular motif, because you can technically use 
    //        ANY binding score here, and this code won't catch it, especially if the dimensions check out. We code this primarily for speed of calculation in the gradient calculation
    unsafe fn only_pos_waveform_from_binds<'a>(&self, binds: &(Vec<f64>, Vec<bool>), bp: usize, motif_pos: usize, DATA: &'a Waveform) -> Waveform<'a> {

        let mut occupancy_trace: Waveform = DATA.derive_zero();

        let mut actual_kernel: Kernel = &self.kernel*1.0;

        let starts = DATA.seq().block_u8_starts();

        let lens = DATA.seq().block_lens();

        let checked = self.base_check( DATA.seq(), &binds.1, bp, motif_pos);

        for i in 0..starts.len() { //Iterating over each block
            for j in 0..lens[i] {
                if checked[starts[i]*BP_PER_U8+j] && (binds.0[starts[i]*BP_PER_U8+j] > THRESH) {
                    actual_kernel = &self.kernel*(binds.0[starts[i]*BP_PER_U8+j]);
                    unsafe {occupancy_trace.place_peak(&actual_kernel, i, j+(self.len()-1)/2)}; //Note: this technically means that we round down if the motif length is even
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

        let d_ad_like_d_ad_stat: f64 = Noise::ad_diff(noise.ad_calc());
        

        //End preuse generation
        let d_noise_d_h = unsafe { self.no_height_waveform_from_binds(&binds, DATA)
                                        .produce_noise(DATA, noise.background)};
        let d_ad_like_d_grad_form_h = d_ad_like_d_ad_stat * (-(&d_noise_d_h * &d_ad_stat_d_noise)) 
                                    * (self.peak_height().abs()-MIN_HEIGHT) * (MAX_HEIGHT - self.peak_height().abs())
                                    / (self.peak_height().signum() * (MAX_HEIGHT-MIN_HEIGHT));
      

        let mut d_ad_like_d_grad_form_binds: Vec<f64> = vec![0.0; self.len()*(BASE_L-1)];


        for index in 0..d_ad_like_d_grad_form_binds.len() {

            let base_id = index/(BASE_L-1); //Remember, this is integer division, which Rust rounds down
            let mut bp = index % (BASE_L-1);
            bp += if bp >= self.pwm[base_id].best_base() {1} else {0}; //If best_base == BASE_L-1, then we just skipped it like we're supposed to

            let prop_bp = unsafe { self.pwm[base_id].rel_bind(bp) } ;

            d_ad_like_d_grad_form_binds[index] = unsafe {
                  - (&(self.only_pos_waveform_from_binds(&binds, base_id, bp, DATA)
                           .produce_noise(DATA, noise.background))
                  * &d_ad_stat_d_noise) * d_ad_like_d_ad_stat
                  * (*PROP_UPPER_CUTOFF-prop_bp) * (prop_bp-*PROP_CUTOFF)
                  / (*PROP_UPPER_CUTOFF-*PROP_CUTOFF) } ;
            
        }

            
        (d_ad_like_d_grad_form_h, d_ad_like_d_grad_form_binds)


    }

    //SAFETY: the d_ad_stat_d_noise must be of the same length as the noise vector we get from DATA.
    pub unsafe fn parallel_single_motif_grad(&self,  DATA: &Waveform, d_ad_stat_d_noise: &Vec<f64>, d_ad_like_d_ad_stat: f64, background: &Background) -> Vec<f64> {

        let binds = self.return_bind_score(DATA.seq());

        let n = self.len()*(BASE_L-1);


        let d_ad_like_d_grad_form: Vec<f64> = (0..n).into_par_iter().map(|i| {
            if i == 0 {
                let d_noise_d_h = unsafe { self.no_height_waveform_from_binds(&binds, DATA)
                                               .produce_noise(DATA, background)};
                d_ad_like_d_ad_stat * (-(&d_noise_d_h * d_ad_stat_d_noise))
                * (self.peak_height().abs()-MIN_HEIGHT) * (MAX_HEIGHT - self.peak_height().abs())
                / (self.peak_height().signum() * (MAX_HEIGHT-MIN_HEIGHT))
            } else {
                let index = i-1;
                let base_id = index/(BASE_L-1); //Remember, this is integer division, which Rust rounds down
                let mut bp = index % (BASE_L-1);
                bp += if bp >= self.pwm[base_id].best_base() {1} else {0}; //If best_base == BASE_L-1, then we just skipped it like we're supposed to

                let prop_bp = unsafe { self.pwm[base_id].rel_bind(bp) } ;

                unsafe {
                      - (&(self.only_pos_waveform_from_binds(&binds, base_id, bp, DATA)
                               .produce_noise(DATA, background))
                      * d_ad_stat_d_noise) * d_ad_like_d_ad_stat
                      * (*PROP_UPPER_CUTOFF-prop_bp) * (prop_bp-*PROP_CUTOFF)
                      / (*PROP_UPPER_CUTOFF-*PROP_CUTOFF) 
                }
            }
        }).collect();

            
        d_ad_like_d_grad_form

    }






    
}

impl fmt::Display for Motif { 
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const DIGITS: usize = 5;
        
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
                (true, false) => write!(f, "[{:.5?},\n", base),
                (false, false) => write!(f, " {:.5?},\n", base),
                (false, true) => write!(f, " {:.5?}]\n", base),
            };
        }
        Ok(())
    }
}


#[derive(Clone)]
//DEFINITELY CANNOT BE DIRECTLY SERIALIZED
pub struct Motif_Set<'a> {

    set: Vec<Motif>, 
    width: f64, 
    signal: Waveform<'a>,
    ln_post: Option<f64>,
    data: &'a Waveform<'a>, 
    background: &'a Background,
}

impl<'a> Motif_Set<'a> {
    

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

        Motif_Set {
            set: self.set.clone(),
            width: self.width, 
            signal: self.signal.clone(),
            ln_post: None,
            data: self.data, //pointer
            background: self.background, //pointer
        }

    }

    //This is our prior on the number of motifs
    //We do not justify this with a maximum entropy prior
    //Instead, we only want to consider an additional motif if 
    //it brings an improvement of at least NECESSARY_MOTIF_IMPROVEMENT to the ln posterior
    //This amounts to a geometric prior with positive integer support 
    //and p = 1-NECESSARY_MOTIF_IMPROVEMENT.exp().
    //NOTE: the ommission of ln(p) term is deliberate. It amounts to a normalization constant
    //for the motif set prior, and would obfuscate the true point of this prior
    fn motif_num_prior(&self) -> f64 {
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

    fn calc_ln_post(&self) -> f64 { //ln_posterior() should be preferred if you can mutate self, since it ensures the calculation isn't redone too much
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

    //This _adds_ a new motif, but does not do any testing vis a vis whether such a move will be _accepted_
    fn propose_new_motif(&self) -> Option<(Self, f64)> {
        let mut new_set = self.derive_set();
        let new_mot = Motif::rand_mot(self.width, self.data.seq()); //rand_mot always generates a possible motif
        let ln_gen_prob = new_mot.height_prior()+new_mot.pwm_prior(self.data.seq());
        let ln_post = new_set.add_motif(new_mot);
        Some((new_set, ln_post-ln_gen_prob)) //Birth moves subtract the probability of their generation

    }

    fn propose_kill_motif(&self) -> Option<(Self, f64)> {

        if (self.set.len() <= 1) { //We never want to propose a set with no motifs, ever
            None
        } else {
            let mut new_set = self.derive_set();
            let mut rng = rand::thread_rng();
            let rem_id = rng.gen_range(0..self.set.len());
            let ln_gen_prob = self.set[rem_id].height_prior()+self.set[rem_id].pwm_prior(self.data.seq());
            let ln_post = new_set.remove_motif(rem_id);
            Some((new_set, ln_post+ln_gen_prob)) //Death moves add the probability of the generation of their deleted variable(s)
        }
    }


    //I'm only extending motifs on one end
    //This saves a bit of time from not having to reshuffle motif vectors
    fn propose_extend_motif(&self) -> Option<(Self, f64)> {
        let mut new_set = self.derive_set();
        let mut rng = rand::thread_rng();
       
        let extend_id = rng.gen_range(0..self.set.len());

        if (self.set[extend_id].len() >= MAX_BASE) { //We want to immediately reject any move which extends a motif beyond maximum length
            None
        } else {

            let mut new_mot = self.set[extend_id].clone();
            new_mot.pwm.push(Base::rand_new());
            let ln_gen_prob = new_mot.height_prior()+new_mot.pwm_prior(self.data.seq());
            if ln_gen_prob > -f64::INFINITY { //When we extend a motif, its best base sequence may no longer be in the sequence
                let ln_post = new_set.replace_motif(new_mot, extend_id);
                let base_ln_density = ((BASE_L-1) as f64)*((*PROP_UPPER_CUTOFF-*PROP_CUTOFF)/(BASE_L as f64)).ln();
                Some((new_set, ln_post-base_ln_density)) //Birth moves subtract the probability of their generation
            } else {
                None
            }
        }
    }

    //I'm only extending motifs on one end
    //This saves a bit of time from not having to reshuffle motif vectors
    fn propose_contract_motif(&self) -> Option<(Self, f64)> {
        let mut new_set = self.derive_set();
        let mut rng = rand::thread_rng();
       
        let contract_id = rng.gen_range(0..self.set.len());

        if (self.set[contract_id].len() <= MIN_BASE) { //We want to immediately reject any move which contracts a motif below minimum length
            None
        } else {

            let mut new_mot = self.set[contract_id].clone();
            let old_base = new_mot.pwm.pop();
            let ln_gen_prob = new_mot.height_prior()+new_mot.pwm_prior(self.data.seq()); //A contracted motif will always exist in the sequence if the longer motif does
            let ln_post = new_set.replace_motif(new_mot, contract_id);
            let base_ln_density = ((BASE_L-1) as f64)*((*PROP_UPPER_CUTOFF-*PROP_CUTOFF)/(BASE_L as f64)).ln();
            Some((new_set, ln_post-base_ln_density)) //Birth moves subtract the probability of their generation
        }
    }



    //MOVE TO CALL
    //For borrow checker reasons, this will only work if the motif calling already has a generated likelihood
    //And I can't have it do it here
    //If you use this on a motif without such a likelihood, it will panic
    pub fn run_rj_move(&self) -> (Self, usize, bool) {

        let mut rng = rand::thread_rng();
        let which_rj = rng.gen_range(0..4);

        let proposal: Option<(Self, f64)> = match which_rj {

            0 => self.propose_new_motif(),
            1 => self.propose_kill_motif(),
            2 => self.propose_extend_motif(),
            3 => self.propose_contract_motif(),
            _ => unreachable!(),

        };

        match proposal {

            None => (self.clone(), which_rj, false),
            Some((new_mot, modded_ln_like)) => {

                if Self::accept_test(self.ln_post.unwrap(), modded_ln_like, &mut rng) {
                    (new_mot, which_rj, true)
                } else { 
                    (self.clone(), which_rj, false)
                }
            }
        }

    }


    //MOVE TO CALL
    pub fn base_leap(&self) -> Self {

       let mut rng = rand::thread_rng();

       //We want to shuffle this randomly, in case there is some kind of codependency between particular TFs
       let mut ids: Vec<usize> = (0..self.set.len()).collect();
       ids.shuffle(&mut rng);

       let mut current_set = self.clone();

       for id in ids {

           let current_mot = current_set.set[id].clone();

           let mut base_set = current_set.clone();
           base_set.remove_motif_void(id);
           //This was numerically derived, and not a hard rule. I wanted less than 50 kmers per leap
           let threshold = if current_mot.len() < 12 {1} else { (current_mot.len())/2-4}; 

           let kmer_ids = self.data.seq().all_kmers_within_hamming(&current_mot.best_motif(), threshold);

           let ids_cartesian_bools = kmer_ids.into_iter().flat_map(|k| [(k, true), (k, false)]).collect::<Vec<_>>();

           let likes_and_mots: Vec<(f64, Self)> = ids_cartesian_bools.clone().into_par_iter().map(|a| {
               let mut to_add = base_set.clone();
               let add_mot = current_mot.scramble_by_id_to_valid(a.0, a.1, self.data.seq());
               let lnlike = to_add.insert_motif(add_mot, id);
               (lnlike, to_add)
           }).collect();

           //We want to pick these based on their relative ln posteriors
           //But these are going to be small. We normalize based on the first
           //ln likelihood because it's convenient
           
           let mut selection_probs: Vec<f64> = vec![0.0; likes_and_mots.len()];

           let normalize_ln_like: f64 = likes_and_mots[0].0;

           let mut sum_probs: f64 = 0.0;

           for i in 0..selection_probs.len() {
               //This subtraction might seem technically unnecessary, but
               //but computers are not infinitely precise. We want to 
               //ensure that we minimize numerical issues
               selection_probs[i] = (likes_and_mots[i].0-normalize_ln_like).exp();
               sum_probs+=selection_probs[i];
           }

           let dist = WeightedIndex::new(&selection_probs).unwrap();
           current_set = likes_and_mots[dist.sample(&mut rng)].1.clone();
       }

       current_set

   }


   fn gradient(&self) -> Vec<f64> {

       let noise = self.signal.produce_noise(self.data, self.background);
       let d_ad_stat_d_noise = noise.ad_grad();
       let d_ad_like_d_ad_stat = Noise::ad_diff(noise.ad_calc());

       let mut len_grad: usize = self.set.len();

       for i in 0..self.set.len() {
           len_grad += self.set[i].len()*(BASE_L-1);
       }

       let mut gradient = vec![0.0_f64; len_grad];

       let mut finished_compute: usize = 0;

       for motif in &self.set {
           
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
   pub fn hmc<R: Rng + ?Sized>(&self, trace_steps: usize, eps: f64, momentum_dist : &Normal, rng: &mut R) ->  (Self, bool) {
       
       let total_len = self.set.len() + (0..self.set.len()).map(|i| self.set[i].len()*(BASE_L-1)).sum::<usize>();

       let momentum: Vec<f64> = (0..total_len).map(|_| momentum_dist.sample(rng)).collect();

       let mut final_trace: Vec<Self> = Vec::with_capacity(trace_steps);

       let mut prior_set = self.clone();
       let mut gradient_old = prior_set.gradient();
       let mut momentum_apply = momentum.clone();


       for _ in 0..trace_steps {
       
           let mut new_set = self.clone();
           new_set.ln_post = None;
       
           let mut modded_momentum = momentum_apply.clone();
       
           for i in 0..modded_momentum.len(){
               //We actually jump to the final state without calculating p_1/2
               //Notice that there's a factor of epsilon missing, here: that's intentional
               //We multiply it in in the add_momentum step
               modded_momentum[i] = modded_momentum[i]-(eps*gradient_old[i])/2.0;
           }

           let mut start = 0;
           let mut next_start = 0;
           for k in 0..self.set.len() {
               next_start += self.set[k].len()*(BASE_L-1)+1;
               new_set.set[k] = unsafe{ prior_set.set[k].add_momentum(eps, &modded_momentum[start..next_start])};
               start = next_start;
           }

           let gradient_new = new_set.gradient();

           for i in 0..momentum_apply.len() {
               momentum_apply[i] -= (eps*(gradient_old[i]+gradient_new[i])/2.0);
           }

           //We want gradient_old to take ownership of the gradient_new values, and gradient_old's prior values to be released
           //Same with prior_set and new_set
           gradient_old = gradient_new;
           prior_set = new_set;

       }

       let mut delta_kinetic_energy: f64 = 0.0;

       for i in 0..momentum.len() {
           delta_kinetic_energy += ( momentum_apply[i].powi(2) - momentum[i].powi(2) )/2.0;
       }

       let delta_potential_energy = prior_set.ln_posterior()-self.ln_post.unwrap();

       if Self::accept_test(0.0, delta_kinetic_energy+delta_potential_energy, rng){
           (prior_set, true)
       } else {
           (self.clone(), false)
       }


       
   }







}


#[derive(Serialize, Deserialize, Clone)]
pub struct Motif_Set_Def {

    set: Vec<Motif>,
    width: f64,
    signal: Waveform_Def,
    ln_post: f64,

}

impl<'a> From<&'a Motif_Set<'a>> for Motif_Set_Def {
    fn from(other: &'a Motif_Set) -> Self {

        let signal = Waveform_Def::from(&other.signal);

        Self {
            set: other.set.clone(), 
            width: other.width,
            signal: signal,
            ln_post: other.calc_ln_post(),
        }
    }
}

impl Motif_Set_Def {

    //SAFETY: data must be the same dimensions as self.signal, both in absolute length of the signal and the
    //        values at each position of the point_lens, start_lens, and spacer
    //        Memory safety may be POSSIBLE with less strict requirements, but it's guarenteed with these
    //        Which also guarentee the correctness of this code. 
    pub unsafe fn get_motif_set<'a>(self, data: &'a Waveform, background: &'a Background) -> Motif_Set<'a> {
        
        let set = self.set.clone();
        let signal = self.signal.get_waveform(data.point_lens(), data.start_dats(), data.seq());

        Motif_Set {
            set: set,
            width: self.width,
            signal: signal,
            ln_post: Some(self.ln_post),
            data: data, 
            background: background,
        }


    }

}






//THIS SHOULD BE (DE)SERIALIZABLE WITH A CUSTOM IMPLEMENTATION
//SINCE ALL MOTIFS IN THE SET AND THE WAVEFORM SHOULD POINT TO seq
//AND THE MOTIF_SET ITSELF SHOULD ALSO POINT TO data AND background
pub struct Set_Trace<'a> {
    trace: Vec<Motif_Set<'a>>,
    capacity: usize,
    data: &'a Waveform<'a>, 
    seq: &'a Sequence,
    background: &'a Background,
}


//TODO: 
    //Create an "advance" function which does the appropriate number of HMCs, base leaps, and RJ steps
    //Create an "output and reset" type of function which outputs all motifs but the final one to a file and starts us to just that motif
    //Create an initialize function that reads in the prepared sequence, waveform, and background distribution from some pre initialized source
    //          and allows us to either start from a prior Set_Trace file, from a MEME motif file, or from completely anew

impl<'a> Set_Trace<'a> {

    //All three of these references should be effectively static. They won't be ACTUALLY, because they're going to depend on user input, but still
    pub fn new_empty(capacity: usize, seq: &'a Sequence, data: &'a Waveform<'a>, background: &'a Background) -> Set_Trace<'a> {

        if !std::ptr::eq(data.seq(), seq) {
            panic!("Your data needs to be associated with your sequence!");
        }

        Set_Trace{
            trace: Vec::<Motif_Set<'a>>::with_capacity(capacity),
            capacity: capacity, 
            data: data, 
            seq: seq, 
            background: background,
        }

    }


    //WARNING: Be aware that this motif set WILL be coerced to follow the seq and data of the motif set
    //         The ln posterior will also be recalculated if the motif set isn't correctly pointed
    //         To ensure the safety of this function, there is a recalculation step that occurs if the 
    //         sequence or data changes. I assure you, you do not want this recalculation to occur:
    //         It's going to be very slow
    pub fn push_set(&'a mut self, set: Motif_Set<'a>) {

        /*        Motif_Set {
            set: set,
            width: self.width,
            signal: signal,
            ln_post: Some(self.ln_post),
            data: data,
            seq: seq,
            background: background,
        }
        */

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

        self.trace.push(repoint_set);

    }

    pub fn push_set_def(&'a mut self, set: Motif_Set_Def) {

        /*        Motif_Set {
            set: set,
            width: self.width,
            signal: signal,
            ln_post: Some(self.ln_post),
            data: data,
            seq: seq,
            background: background,
        }
        */

        let mut repoint_set = unsafe{set.get_motif_set(&self.data, &self.background)};

        repoint_set.signal = self.data.derive_zero();
        for i in 0..repoint_set.set.len() {
            repoint_set.signal += &repoint_set.set[i].generate_waveform(&self.data);
        }
        repoint_set.ln_post = None;
        repoint_set.ln_posterior();

        self.trace.push(repoint_set);

    }

    //Note: if the likelihoods are calculated off of a different sequence/data, this WILL 
    //      just give you a wrong answer that seems to work
    pub fn push_set_def_trust_like(&'a mut self, set: Motif_Set_Def) {
        self.trace.push(unsafe{set.get_motif_set(&self.data, &self.background)});
    }

    pub fn push_set_def_trust_like_many(&'a mut self, sets: Vec<Motif_Set_Def>) {
        for set in sets {
            self.trace.push(unsafe{set.get_motif_set(&self.data, &self.background)});
        }
    }

    //Suggested defaults: 1.0, 1 or 2, 2^(some negative integer power between 5 and 10), 5-20, 80. But fiddle with this for acceptance rates
    //You should aim for your HMC to accept maybe 80-90% of the time. Changing RJ acceptances is hard, but you should hope for like 20%ish
    //Base leaps run once: after the RJ steps but before the HMC steps: this is basically designed so that each set of steps goes from
    //most drastic to least drastic changes
    pub fn advance<R: Rng + ?Sized>(&'a mut self, rng: &mut R, momentum_sd: f64, hmc_trace_steps: usize, hmc_epsilon: f64, num_rj_steps: usize, num_hmc_steps: usize) -> [usize; 5] { //1 for each of the RJ moves, plus one for hmc
        
        let momentum_dist = Normal::new(0.0, momentum_sd).unwrap();

        let mut num_acceptances: [usize; 5] = [0; 5];

        for _ in 0..num_rj_steps {

            let setty = self.current_set();
            let (mut set, move_type, accept) = setty.run_rj_move();
            if accept{
                num_acceptances[move_type] += 1;
            }


            self.trace.push(set);
        }


        let set = self.current_set().base_leap();

        self.trace.push(set);

        for _ in 0..num_hmc_steps {

            let (set, accept) = self.current_set().hmc(hmc_trace_steps, hmc_epsilon, &momentum_dist, rng);
            if accept {
                num_acceptances[4] += 1;
            }
            self.trace.push(set);
        }


        num_acceptances

    }

    fn current_set<'b>(&self) -> Motif_Set<'a> {
        self.trace[self.trace.len()-1].clone()
    }
   /* 
    fn save_and_drop_history(&mut self) -> Set_Trace_Def {

        let len_trace = self.trace.len();

        //We want to keep the last element in the Set_Trace, so that the markov chain can proceed
        let trace = self.trace.drain(0..(len_trace-1)).map(|a| Motif_Set_Def::from(&a)).collect();


        Set_Trace_Def {
            trace: trace, 
            capacity: self.capacity, 
            data: Waveform_Def::from(&self.data),
            seq: self.seq.clone(),
            background: self.background.clone(),
        }
    }
*/
    //SAFETY: the seq MUST be a clone of what data is currently pointing to, and the set trace MUST be empty
    //This is EXTREMELY unsafe otherwise. 
    /*unsafe fn repoint(&mut self) {
        self.data.repoint(&self.seq);
    }*/

}
/*
#[derive(Serialize, Deserialize, Clone)]
pub struct Set_Trace_Def {

    trace: Vec<Motif_Set_Def>,
    capacity: usize,
    data: Waveform_Def,
    seq: Sequence,
    background: Background,


}

impl Set_Trace_Def {
 
    //You may have noticed that this is a self, not a &self
    //That's intentional. I want Rust to straight up drop the Set_Trace_Def that calls this
    //Remember, this is a super fancy Metropolis-Hastings done on GENOME scale data. 
    //We're compute and memory constrained. 
    fn pre_repoint_set_trace(&self) -> Set_Trace {

        let mut trace = Vec::<Motif_Set>::with_capacity(self.capacity);

        //let last_item = self.trace.pop().unwrap();

        let (point_lens, start_dats) = Waveform::make_dimension_arrays(&self.seq, self.data.spacer());

        let data_len = point_lens.last().unwrap()+start_dats.last().unwrap();
        let safe = (self.data.len() == data_len);

        //(last_item.signal.len() == data_len);

        //This guarentees the memory safety of get_waveform() for the data waveform.
        //There's no graceful way to recover from the data signal being incompatible
        //with your sequence, so we burn everything to the ground.
        if !safe { panic!("Sequence structure is incompatible with data structure!!!!")}

        let data = unsafe{ self.data.clone().get_waveform(point_lens, start_dats, &self.seq) };

        //trace.push(unsafe{last_item.get_motif_set(&self.data, &self.background)});

        Set_Trace{

            trace: trace, 
            capacity: self.capacity, 
            data: data,
            seq: self.seq.clone(),
            background: self.background.clone(),

        }
     
    }

    //SAFETY: self and the set trace must have identical sequences and data
    unsafe fn load_last_state<'a>(&'a self, load_into_set: &'a mut Set_Trace<'a>) {

        let seq_ptr = &load_into_set.seq as *const Sequence;
        load_into_set.data.repoint(seq_ptr);
        load_into_set.push_set_def_trust_like((self.trace.last().unwrap().clone()));

    }

    //SAFETY: self and the set trace must have identical sequences and data
    //NOTE: the safety guarantee also guarantees that push_set_def_trust_like is fine
    unsafe fn load_every_state_trusted_likes<'a>(&'a self, load_into_set: &'a mut Set_Trace<'a>) {
        
        let seq_ptr = &load_into_set.seq as *const Sequence;
        load_into_set.data.repoint(seq_ptr);
        load_into_set.push_set_def_trust_like_many((self.trace.clone()));
    
    }

}
*/


// 
/*

impl Serialize for Set_Trace<'_> {

    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut state = serializer.serialize_struct("Set_Trace", 5)?;
        
        let trace: Vec<Motif_Set_Def> = self.trace.iter().map(|a| Motif_Set_Def::from(a)).collect();
        state.serialize_field("trace", &trace)?;

        state.serialize_field("capacity", &self.capacity)?;
        state.serialize_field("seq", &self.seq)?;
        state.serialize_field("background", &self.background)?;

        state.serialize_field("data", &Waveform_Def::from(&self.data))?;



        state.end()
    }


}



impl<'de> Deserialize<'de> for Set_Trace {
    fn deserialize<D>(deserializer: D) -> Result<i32, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_i32(I32Visitor)
    }
}

// */



//BEGIN TRUNCATED LOGNORMAL

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct TruncatedLogNormal {
    location: f64,
    scale: f64,
    min: f64, 
    max: f64,
}

impl TruncatedLogNormal {

    pub fn new(location: f64, scale: f64, min: f64, max: f64) -> Result<TruncatedLogNormal> {
        if location.is_nan() || scale.is_nan() || scale <= 0.0 || (min > max) || (max < 0.0) {
            Err(StatsError::BadParams)
        } else {
            let min = if min >= 0.0 {min} else {0.0} ;
            Ok(TruncatedLogNormal { location: location, scale: scale, min: min, max: max })
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

            let lndist: LogNormal = LogNormal::new(self.location, self.scale).unwrap();
            lndist.cdf(x)/(lndist.cdf(self.max)-lndist.cdf(self.min))
        }
    }

    fn sf(&self, x: f64) -> f64 {
        if x <= self.min {
            1.0
        } else if x >= self.max() {
            0.0
        } else {
            let lndist: LogNormal = LogNormal::new(self.location, self.scale).unwrap();
            lndist.sf(x)/(lndist.cdf(self.max)-lndist.cdf(self.min))
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
            let lndist: LogNormal = LogNormal::new(self.location, self.scale).unwrap();
            let scale_density = lndist.cdf(self.max)-lndist.cdf(self.min);

            usual_density/scale_density

        }
    }

    fn ln_pdf(&self, x: f64) -> f64 {
        if x < self.min || x > self.max {
        f64::NEG_INFINITY
        } else {
            let d = (x.ln() - self.location) / self.scale;
            let usual_density = (-0.5 * d * d) - consts::LN_SQRT_2PI - (x * self.scale).ln();
            let lndist: LogNormal = LogNormal::new(self.location, self.scale).unwrap();
            let scale_density = (lndist.cdf(self.max)-lndist.cdf(self.min)).ln();

            usual_density-scale_density
        }
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
    /*const MIN_HEIGHT: f64 = 3.;
    const MAX_HEIGHT: f64 = 30.;
    const LOG_HEIGHT_MEAN: f64 = 2.30258509299; //This is ~ln(10). Can't use ln in a constant
    const LOG_HEIGHT_SD: f64 = 0.25;

    const BASE_L: usize = 4;
*/
    
    #[test]
    fn it_works() {
        let base = 3;
        let try_base: Base = Base::rand_new();
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

        let mut rng = fastrand::Rng::new();

        let block_n: usize = 200;
        let u8_per_block: usize = 4375;
        let bp_per_block: usize = u8_per_block*4;
        let bp: usize = block_n*bp_per_block;
        let u8_count: usize = u8_per_block*block_n;

        println!("DF");
        let start_gen = Instant::now();
        //let blocks: Vec<u8> = (0..u8_count).map(|_| rng.u8(..)).collect();
        let preblocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.u8(..)).collect();
        let blocks: Vec<u8> = preblocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>(); 
        let block_u8_starts: Vec<usize> = (0..block_n).map(|a| a*u8_per_block).collect();
        let block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
        let mut start_bases: Vec<usize> = (0..block_n).map(|a| a*bp_per_block).collect();
        let sequence: Sequence = Sequence::new_manual(blocks, block_lens.clone());
        let wave: Waveform = Waveform::create_zero(&sequence, 5);
        let duration = start_gen.elapsed();
        println!("Done gen {} bp {:?}", bp, duration);

        println!("{} gamma", gamma::gamma(4.));
        println!("{} gamma", gamma::ln_gamma(4.));

        //println!("{:?}", wave.raw_wave());

        let motif: Motif = Motif::from_motif(sequence.return_bases(0,0,20), 20.); //sequence

        let motif2: Motif = Motif::from_motif(sequence.return_bases(0,2,20), 20.); //sequence

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

        let random_motif = Motif::rand_mot(20., &sequence);

        println!("Random motif\n{}", random_motif);

        assert!(random_motif.raw_kern().len() == 121);

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

        let un_mot: Motif = Motif::from_motif(vec![1usize;20], 10.);//Sequence

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
        let little_motif: Motif = Motif::raw_pwm(mat, 10.0, 1.0); //wave_seq

        print!("{}", little_motif);
        println!("{:?}",little_motif.generate_waveform(&wave_wave).raw_wave());

        let small_block: Vec<u8> = vec![44, 24, 148, 240, 84, 64, 200, 80, 68, 92, 196, 144]; 
        let small_inds: Vec<usize> = vec![0, 6]; 
        let small_lens: Vec<usize> = vec![24, 24];
        let small: Sequence = Sequence::new_manual(small_block, small_lens);
        let small_wave: Waveform = Waveform::new(vec![0.1, 0.6, 0.9, 0.6, 0.1, -0.2, -0.4, -0.6, -0.6, -0.4], &small, 5);

        let mat: Vec<Base> = (0..15).map(|_| Base::new(theory_base.clone())).collect::<Vec<_>>();
        let wave_motif: Motif = Motif::raw_pwm(mat, 10.0, 1.0); //small

        let rev_comp: Vec<bool> = (0..48).map(|_| rng.bool()).collect();

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
        assert!((unsafe_raw_filter.iter().zip(&raw_filter).map(|(a, &b)| (a-b).powf(2.0)).sum::<f64>()).powf(0.5) < 1e-6);

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

                    assert!((raw_filter[start_dats[i]+j]-filt_score).abs() < 1e-6);

                } else {
                    assert!(wave_gen[start_dats[i]+j].abs() < 1e-6);
                    assert!((wave_sho[start_dats[i]+j]).abs() < 1e-6);
                    assert!(raw_filter[start_dats[i]+j].abs() < 1e-6);
                }


            }
        }



    }
}
