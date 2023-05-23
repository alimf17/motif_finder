

use std::ops::{Add, AddAssign};
use std::ops::{Sub, SubAssign};
use std::ops::Mul;
use std::cmp::max;
use std::cmp::min;
use core::f64::consts::PI;

use crate::sequence::Sequence;
use statrs::distribution::StudentsT;
use statrs::distribution::{Continuous, ContinuousCDF};
use statrs::Result as otherResult;


use aberth::aberth;
use num_complex::Complex;
use num_complex::ComplexFloat;
const EPSILON: f64 = 1e-8;
use once_cell::sync::Lazy;
use num_traits::cast;
use num_traits::float::Float;
use num_traits::float::FloatConst;
use num_traits::identities::{One, Zero};
use num_traits::MulAdd;
use core::iter::zip;

use std::thread;
use std::sync::Arc;
use rayon::prelude::*;

use assume::assume;

use std::time::{Duration, Instant};

use serde::{ser, Serialize,Serializer, Deserialize};
use serde::de::{
    self, DeserializeSeed, EnumAccess, IntoDeserializer, MapAccess, SeqAccess,
    VariantAccess, Visitor,
};
use serde_big_array::BigArray;
const WIDE: f64 = 3.0;

//These are based on using floats with a maximum of 8 binary sigfigs
//after the abscissa and exponents ranging from -12 to 3, inclusive
const MIN_DT_FLOAT: f64 = 0.000244140625;
const MAX_DT_FLOAT: f64 = 15.96875;



const EXP_OFFSET: u64 = (1023_u64-12) << 52;


#[derive(Serialize, Deserialize, Clone)]
pub struct Kernel{

    peak_width: f64,
    kernel: Vec<f64>,
}

impl Mul<f64> for &Kernel {

    type Output = Kernel;

    fn mul(self, rhs: f64) -> Kernel {

        Kernel {
            peak_width: self.peak_width,
            kernel: self.kernel.iter().map(|a| a*rhs).collect(),
        }

    }

}

impl Kernel {
    
    pub fn new(peak_width: f64, peak_height: f64) -> Kernel {

        let span = (peak_width*WIDE) as isize;

        let domain: Vec<isize> = (-span..(span+1)).collect();

        let range = domain.iter().map(|a| (-((*a as f64).powf(2.0))/(2.0*peak_width.powf(2.0))).exp()*peak_height).collect();

        Kernel{
            peak_width: peak_width,
            kernel: range,
        }

    }

    pub fn get_sd(&self) -> f64 {
        self.peak_width
    }

    pub fn get_curve(&self) -> &Vec<f64> {

        &self.kernel
    
    }

    pub fn len(&self) -> usize {
        self.kernel.len()
    }
    


}

//CANNOT BE SERIALIZED
#[derive(Clone)]
pub struct Waveform<'a> {
    wave: Vec<f64>,
    spacer: usize,
    point_lens: Vec<usize>,
    start_dats: Vec<usize>,
    seq: &'a Sequence,
}

impl<'a> Waveform<'a> {

    /*
       This function has an additional constraint:

       The start_data must be organized in accordance with spacer and sequence.
       In particular, each block must have 1+floor((block length-1)/spacer) data points
       HOWEVER, this initializer only knows TO panic if the total number of data points
       cannot be reconciled with seq and spacer. If you have one too MANY points in one
       block and one too FEW points in another, this will NOT know to break. 
       */
    pub fn new(start_data: Vec<f64>, seq: &'a Sequence, spacer: usize) -> Waveform<'a> {


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
        let point_lens: Vec<usize> = seq.block_lens().iter().map(|a| 1+((a-1)/spacer)).collect();

        if point_lens.iter().sum::<usize>() != start_data.len() {
            panic!("IMPOSSIBLE DATA FOR THIS SEQUENCE AND SPACER")
        }

        let mut start_dats: Vec<usize> = Vec::new();

        let mut size: usize = 0;

        for i in 0..point_lens.len(){
            start_dats.push(size);
            size += point_lens[i];
        }

        let tot_L: usize = point_lens.iter().sum();

        Waveform {
            wave: start_data,
            spacer: spacer,
            point_lens: point_lens,
            start_dats: start_dats,
            seq: seq,
        }
    }

    pub fn create_zero(seq: &'a Sequence, spacer: usize) -> Waveform<'a> {
       
        let point_lens: Vec<usize> = seq.block_lens().iter().map(|a| 1+((a-1)/spacer)).collect();

        let mut start_dats: Vec<usize> = Vec::new();

        let mut size: usize = 0;

        for i in 0..point_lens.len(){
            start_dats.push(size);
            size += point_lens[i];
        }

        let tot_L: usize = point_lens.iter().sum();

        println!("TOT {}", tot_L);
        Waveform {
            wave: vec![0.0; tot_L],
            spacer: spacer,
            point_lens: point_lens,
            start_dats: start_dats,
            seq: seq,
        }
    }

    pub fn derive_zero(&self) -> Waveform {

        let tot_L: usize = self.point_lens.iter().sum();

        
        Waveform {
            wave: vec![0.0; tot_L],
            spacer: self.spacer,
            point_lens: self.point_lens.clone(),
            start_dats: self.start_dats.clone(),
            seq: self.seq,
        }

    }

    //block must be less than the number of blocks
    //center must be less than the number of bps in the blockth block
    pub unsafe fn place_peak(&mut self, peak: &Kernel, block: usize, center: usize) {



        //Given how we construct kernels, this will never need to be rounded
        let place_bp = (((peak.len()-1)/2) as isize)-(center as isize); //This moves the center of the peak to where it should be, taking the rest of it with it
        let cc = (place_bp).rem_euclid(self.spacer as isize); // This defines the congruence class of the kernel indices that will be necessary for the signal
       
        let zerdat: usize = self.start_dats[block]; //This will ensure the peak is in the correct block

        let min_kern_bp: usize = max(0, place_bp) as usize;
        let nex_kern_bp: usize = min(peak.len() as isize, ((self.spacer*self.point_lens[block]) as isize)+place_bp) as usize; //Technicaly, the end CAN return a negative int. 
                                                                                 //But if it is, panicking is appropriate: 
                                                                                 //center would necessarily be much bigger than the block length

        let kern_values: Vec<f64> = (min_kern_bp..nex_kern_bp).filter(|&bp| ((bp % self.spacer) == (cc as usize))).map(|f| peak.get_curve()[f as usize]).collect();
        
        
        let completion: usize = ((cc-((peak.len() % self.spacer) as isize)).rem_euclid(self.spacer as isize)) as usize; //This tells us how much is necessary to add to the length 
                                                                            //of the kernel to hit the next base in the cc
        
        let min_kern_cc = max(cc, place_bp);
        let nex_kern_cc = min(((self.point_lens[block]*self.spacer) as isize)+place_bp, ((peak.len()+completion) as isize));

        let min_data: usize = ((min_kern_cc-place_bp)/((self.spacer) as isize)) as usize;  //Always nonnegative: if place_bp > 0, min_kern_bp = place_bp
        let nex_data: usize = ((nex_kern_cc-place_bp)/((self.spacer) as isize)) as usize; //Assume nonnegative for the same reasons as nex_kern_bp

        let w = self.wave.len();
        

        let kern_change = self.wave.get_unchecked_mut((min_data+zerdat)..(nex_data+zerdat));

        if kern_values.len() > 0 {
            //println!("{} {} {} {} {} peak",min_data+zerdat, nex_data+zerdat, kern_values.len(), kern_change.len(), w);
            for i in 0..kern_change.len(){
                kern_change[i] += kern_values[i];
            }
        } 
        
       

    }

    pub fn produce_noise<'b>(&self, data: &Waveform, background: &'b Background) -> Noise<'b> {
        


        let residual = data-self;


        let mut end_dats = residual.start_dats()[1..residual.start_dats.len()].to_vec();

        let resid = residual.wave;
        
        end_dats.push(resid.len());

        let mut len_penalties = vec![0usize; end_dats.len()];

        for k in 0..end_dats.len() {
            len_penalties[k] = ((k+1)*background.ar_corrs.len());
        }

        let filt_lens: Vec<usize> = end_dats.iter().zip(len_penalties).map(|(a, b)| a-b).collect();

        let l_c = background.ar_corrs.len();

        let mut fin_noise: Vec<f64> = vec![0.0; filt_lens.iter().sum()];

        for k in 0..end_dats.len(){

            let sind: usize = if k == 0 {0} else {end_dats[k-1]};


            let mut block: Vec<f64> = resid[(sind+l_c)..end_dats[k]].iter().zip(resid[(sind+l_c-1)..(end_dats[k]-1)].iter()).map(|(a,b)| a-background.ar_corrs[0]*b).collect();
            
            if l_c > 1 {
            
                for i in 1..l_c {
                    block = block.iter().zip(resid[(sind+l_c-(i+1))..(end_dats[k]-(i+1))].iter()).map(|(a,b)| a-background.ar_corrs[i]*b).collect();
                }
            }

            let sind: usize = if k == 0 {0} else {filt_lens[k-1]};

            let block_ref = &mut fin_noise[sind..filt_lens[k]];

            for i in 0..block_ref.len(){

                block_ref[i] = block[i];

            }

        }

        Noise::new(fin_noise,background)


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


        //If we have the same sequence pointer and the same spacer, our lengths are always identical
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

        //If we have the same sequence pointer and the same spacer, our lengths are always identical
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

#[derive(Serialize, Clone)]
pub struct WaveformDef {
    wave: Vec<f64>,
    spacer: usize,
    point_lens: Vec<usize>,
    start_dats: Vec<usize>,

}
impl<'a> From<&'a Waveform<'a>> for WaveformDef {
    fn from(other: &'a Waveform) -> Self {
        Self {
            wave: other.raw_wave(),
            spacer: other.spacer(),
            point_lens: other.point_lens(),
            start_dats: other.start_dats(),
        }
    }
}

impl WaveformDef {

    //SAFETY: If the relevant lengths of the waveform do not correspond to the sequence, this will be unsafe. 
    //        By correspond, I mean that every element of point_lens needs to be 1+seq.block_lens/spacer
    //        And start_dats needs to match the cumulative sum of point_lens. 
    pub unsafe fn get_waveform<'a>(&'a self, seq: &'a Sequence) -> Waveform<'a> {

        Waveform {

            wave: self.wave.clone(),
            spacer: self.spacer, 
            point_lens: self.point_lens.clone(),
            start_dats: self.start_dats.clone(),
            seq: seq
        }


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

#[derive(Serialize, Deserialize, Clone)]
pub struct Background {
    #[serde(with = "StudentsTDef")]
    pub dist: StudentsT,
    pub ar_corrs: Vec<f64>,
    #[serde(with = "BigArray")]
    pub cdf_lookup: [f64; 8192],
    #[serde(with = "BigArray")]
    pub f64_lookup: [f64; 8192],
    #[serde(with = "BigArray")]
    pub pdf_lookup: [f64; 8192],
    #[serde(with = "BigArray")]
    pub dpdf_lookup: [f64; 8192],
}

impl Background {

    pub fn new(sigma_background : f64, df : f64, ar_corrs: &Vec<f64>) -> Background {

        let mut poly = ar_corrs.iter().map(|a| -1.0*a).collect::<Vec<f64>>();
        poly.splice(0..0, [1.0]);
        
        let roots = aberth_vec(&poly, EPSILON).unwrap();

        for root in roots {
            println!("root {}. Abs {}.", root, root.abs());
            if root.abs() <= 1.0+EPSILON { //Technically, the +EPSILON means that we might rule out some stationary models
                panic!("AR model is not stationary!")
            }
        }

        let dist = StudentsT::new(0., sigma_background, df).unwrap();


        let mut domain = [0.0_f64; 8192];
        
        for sign in 0_u64..2 {
            for rest in 0_u64..4096 {
                //This gives us 8 abscissa bits and 3 effective sign bits to play with: the 1023-12 indicates that the minimum 
                //exponent we care about is 2^(-12)
                domain[((4096*sign+rest) as usize)] = unsafe{ std::mem::transmute::<u64, f64>((sign << 63) + (1023-12 << 52) + (rest << 44))};
            }
        }


        let cdf_lookup: [f64; 8192] = domain.iter().map(|&a| dist.cdf(a)).collect::<Vec<_>>().try_into().unwrap();
        let pdf_lookup: [f64; 8192] = domain.iter().map(|&a| dist.pdf(a)).collect::<Vec<_>>().try_into().unwrap();

        let dpdf_lookup: [f64; 8192] = domain.iter().map(|&a| (a/sigma_background)*dist.pdf(a)*(df+1.)/(df+(a/sigma_background).powi(2)))
                                             .collect::<Vec<_>>().try_into().unwrap();

        Background{dist: dist, ar_corrs:ar_corrs.clone(), cdf_lookup:cdf_lookup, f64_lookup: domain, pdf_lookup:pdf_lookup, dpdf_lookup:dpdf_lookup}
    }


    fn get_near_u64(calc: f64) -> u64 {

        let mut bit_check: u64 = calc.to_bits();
       
        if calc.abs() < MIN_DT_FLOAT {
            bit_check = (calc.signum()*MIN_DT_FLOAT).to_bits();
        } else if calc.abs() > MAX_DT_FLOAT {
            bit_check = (calc.signum()*MAX_DT_FLOAT).to_bits();
        }

        (bit_check-EXP_OFFSET) & 0xfffff00000000000


    }

    fn get_near_f64(calc: f64) -> f64 {
        unsafe{ std::mem::transmute::<u64, f64>(Self::get_near_u64(calc))}
    }

    //SAFETY: you need to know that you have a valid lookup index
    unsafe fn get_near_f64_from_ind(&self, ind: usize) -> f64 {
        *self.f64_lookup.get_unchecked(ind)
    }

    fn get_lookup_index(calc: f64) -> usize {

        let bitprep = Self::get_near_u64(calc);
        (((bitprep & 0x8_000_000_000_000_000) >> 51) 
      + ((bitprep & 0x7_fff_f00_000_000_000) >> 44)) as usize
        
    }

    pub fn cdf(&self, calc: f64) -> f64{

        let ind = Self::get_lookup_index(calc);
        unsafe{ *self.cdf_lookup.get_unchecked(ind) + *self.pdf_lookup.get_unchecked(ind)*(calc-self.get_near_f64_from_ind(ind))} //linear approx
    }
    pub fn pdf(&self, calc: f64) -> f64{
        let ind = Self::get_lookup_index(calc);
        unsafe{ *self.pdf_lookup.get_unchecked(ind) + *self.dpdf_lookup.get_unchecked(ind)*(calc-self.get_near_f64_from_ind(ind))}
    }

    //SAFETY: you need to know that you have a valid lookup index
    pub unsafe fn cdf_from_ind(&self, calc_ind: usize) -> f64{
        *self.cdf_lookup.get_unchecked(calc_ind)
    }
    pub unsafe fn pdf_from_ind(&self, calc_ind: usize) -> f64{
        *self.pdf_lookup.get_unchecked(calc_ind)
    }



}


//CANNOT BE SERIALIZED
#[derive(Clone)]
pub struct Noise<'a> {
    resids: Vec<f64>,
    pub background: &'a Background,
}

impl<'a> Noise<'a> {


    pub fn new(resids: Vec<f64>, background: &'a Background) -> Noise<'a> {
        Noise{ resids: resids, background: background}

    }

    pub fn resids(&self) -> Vec<f64> {
        self.resids.clone()
    }
    
    pub fn dist(&self) -> StudentsT {
        self.background.dist.clone()
    }

    
    pub fn rank(&self) -> Vec<f64> {

        
        //This generates a vector where each element is tagged with its original position
        let mut rx: Vec<(usize, f64)> = self.resids.clone().iter().enumerate().map(|(a, b)| (a, *b)).collect();

        //This sorts the elements, but recalls their original positions
        rx.par_sort_unstable_by(|(_,a), (_,b)| a.partial_cmp(b).unwrap());
       
    
        
        let mut ranks: Vec<f64> = vec![0.; rx.len()];

        let mut ind: f64 = 0.0;

        //This uses the sort of elements to put rankings in their original positions
        for &(i, _) in &rx {
            ranks[i] = ind;
            ind += 1.0;
        }

        
        /*let mut prerank = rx.iter().enumerate().map(|(a, (b, _))| (a as f64,b)).collect::<Vec<_>>();

        prerank.sort_unstable_by(|(_, a), (_, b)| a.cmp(b));

        let ranks = prerank.iter().map(|(a,_)| *a).collect::<Vec<_>>();
        */
        ranks
        //let inds = (0..self.resids.len()).collect::<Vec<usize>>();
        //self.rankVec(&inds).iter().map(|&a| a as f64).collect::<Vec<f64>>()
        
        
    }

    /*
    pub fn rankVec(&self, vecInds: &Vec<usize>) -> Vec<usize> {
        
        let pivot = vecInds[((vecInds.len()-1)/2)];

        let (preInds0, prepreInds1): (Vec<&usize>, Vec<&usize>) = vecInds.iter().partition(|&a| self.resids[*a] < self.resids[pivot]);

        let (allSet, preInds1): (Vec<&usize>, Vec<&usize>) = prepreInds1.iter().partition(|&a| self.resids[**a] <= self.resids[pivot]);
        let vecInds0: Vec<usize> = preInds0.iter().map(|&a| *a).collect();
        //let vecSet: Vec<usize> = allSet.iter().map(|&a| *a).collect();
        let vecInds1: Vec<usize> = preInds1.iter().map(|&a| *a).collect();


        let mut coll = allSet.iter().map(|&a| *a).collect::<Vec<usize>>();
        if vecInds0.len() != 0 {
            let colInds0 = if vecInds0.len() == 1 { vecInds0 } else { self.rankVec(&vecInds0) };
            coll = colInds0.iter().chain(coll.iter()).map(|&a| a).collect();
        }
        if vecInds1.len() != 0 {
            let colInds1 = if vecInds1.len() == 1 { vecInds1 } else { self.rankVec(&vecInds1) };
            coll = coll.iter().chain(colInds1.iter()).map(|&a| a).collect();
        }
        coll



    }*/

    pub fn ad_calc(&self) -> f64 {

        let time = Instant::now();
        let mut forward: Vec<f64> = self.resids();
        forward.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let n = forward.len();

        let mut Ad = -(n as f64);
        for i in 0..n {

            let cdf = self.background.cdf(forward[i]);
            let rev_sf = 1.0-self.background.cdf(forward[n-1-i]);
            Ad-= (cdf.ln()+rev_sf.ln()) * ((2*i+1) as f64)/(n as f64)

        }

        println!("ad_calc {:?}", time.elapsed());
        Ad
    }

    
    pub fn ad_grad(&self) -> Vec<f64> {

        let start = Instant::now();
        let ranks = self.rank();

        let n = self.resids.len() as f64;
       
        let mut derivative: Vec<f64> = vec![0.0; self.resids.len()];

        for i in 0..self.resids.len() {
            let cdf = unsafe{ self.background.cdf(self.resids[i])};
            derivative[i] = ((unsafe{self.background.pdf(self.resids[i])})/((1.0-cdf)*n))*(2.*(n)-(((2.*ranks[i]+1.))/cdf));
        }


        println!("ad_grad {:?}", start.elapsed());
        derivative

    }

/*
    pub fn ad_grad(&self) -> Vec<f64> {

        let mut preforward: Vec<_> = self.resids.iter().enumerate().collect();

        preforward.sort_unstable_by(|(_,a),(_,b)| a.partial_cmp(b).unwrap());

        let forward = preforward.iter().map(|(_,b)| *b);
        let inds = preforward.iter().map(|(a,_)| *a);

        let n = forward.len() as f64;
        let finds = forward.map(|&f| Background::get_lookup_index(f));
        let pdf = finds.clone().map(|a| unsafe{ self.background.pdf_from_ind(a)});
        let cdf = finds.map(|a| unsafe{ self.background.cdf_from_ind(a)} );



        /*
        let derivative: Vec<f64> = forward.iter().zip(ranks)
                                    .map(|(&a, b)| (self.background.dist.pdf(a)/(self.background.dist.sf(a)*(n as f64)))*(2.*(n as f64)-(((2*b+1) as f64)/self.background.dist.cdf(a))))
                                    .collect();*/ 
       
        let mut calc_derivative: Vec<(usize, f64)> = cdf.zip(pdf).zip(inds).enumerate()
                                    .map(|(r, ((c, p), i))| (i, (p/((1.0-c)*n))*(2.*(n)-(((2.*(r as f64)+1.))/c))))
                                    .collect();

        calc_derivative.sort_unstable_by(|(a,_),(b, _)| a.cmp(b));

        let derivative = calc_derivative.iter().map(|(_,a)| *a).collect::<Vec<_>>();

        derivative

    }
*/
    fn low_val(lA: f64) -> f64 {

        const C: f64 = PI*PI/8.0;
        let cfs = [2.00012,0.247105,-0.0649821, 0.0347962,-0.0116720,0.00168691];

        let expo = (0..6).map(|a| 2.0*(a as f64)-1.0).collect::<Vec<f64>>();

        let p: f64 = cfs.iter().zip(expo)
                        .map(|(&c, e)| c*e*lA.sqrt().powf(e-2.0)/2.0 + C*c*lA.sqrt().powf(e-4.0)).sum();

        -C/lA+p.ln()

    }


    fn high_val(hA: f64) -> f64 {

        (3.0/(hA*PI)).ln()/2.0-hA

    }

    pub fn ad_like(A: f64) -> f64 {

        const A0: f64 = 2.64;
        const k: f64 = 84.44556;

        let lo = Self::low_val(A);
        let hi = Self::high_val(A);

        let w = 1.0/(1.0+(A/A0).powf(k));

        w*lo+(1.0-w)*hi
    }

    pub fn ad_diff(A: f64) -> f64 {

        const h: f64 = 0.00001;
        (Self::ad_like(A+h)-Self::ad_like(A))/h

    }



}

impl Mul<&Vec<f64>> for &Noise<'_> {

    type Output = f64;

    //fn mul(self, rhs: &Noise) -> f64 {
    fn mul(self, rhs: &Vec<f64>) -> f64 {

        //let rhs_r = rhs.resids();

        //if(self.resids.len() != rhs_r.len()){
        if(self.resids.len() != rhs.len()){
            panic!("Residuals aren't the same length?!")
        }

        //self.resids.iter().zip(rhs_r).map(|(a,b)| a*b).sum()
        //self.resids.iter().zip(rhs).map(|(a,b)| a*b).sum()

        let mut sum = 0.0;

        for i in 0..self.resids.len() {
            sum+= (self.resids[i]*rhs[i]);
        }
        sum
    }
}



//I had to take the source code for below from the aberth crate and modify it for vectors


pub fn aberth_vec(polynomial: &Vec<f64>, epsilon: f64) -> Result<Vec<Complex<f64>>, &'static str> {
  let dydx = &vec_derivative(polynomial); //Got up to here 
  let mut zs: Vec<Complex<f64>> = initial_guesses(polynomial);
  let mut new_zs = zs.clone();

  let iter: usize = ((1.0/epsilon) as usize)+1;
  'iteration: for _ in 0..iter {
    let mut converged = true;

    for i in 0..zs.len() {
      let p_of_z = sample_polynomial(polynomial, zs[i]);
      let dydx_of_z = sample_polynomial(dydx, zs[i]);
      let sum = (0..zs.len())
        .filter(|&k| k != i)
        .fold(Complex::new(0.0f64, 0.0), |acc, k| {
          acc + Complex::new(1.0f64, 0.0) / (zs[i] - zs[k])
        });

      let new_z = zs[i] + p_of_z / (p_of_z * sum - dydx_of_z);
      new_zs[i] = new_z;

      if new_z.re.is_nan()
        || new_z.im.is_nan()
        || new_z.re.is_infinite()
        || new_z.im.is_infinite()
      {
        break 'iteration;
      }

      if !new_z.approx_eq(zs[i], epsilon) {
        converged = false;
      }
    }
    println!("");
    if converged {
      return Ok(new_zs);
    }
    core::mem::swap(&mut zs, &mut new_zs);
  }
  Err("Failed to converge.")
}

pub fn vec_derivative(coefficients: &Vec<f64>) -> Vec<f64> {
  debug_assert!(coefficients.len() != 0);
  coefficients
    .iter()
    .enumerate()
    .skip(1)
    .map(|(power, &coefficient)| {
      let p = power as f64 ;
      p * coefficient
    })
    .collect()
}

fn initial_guesses(polynomial: &Vec<f64>)-> Vec<Complex<f64>> {
  // the degree of the polynomial
  let n = polynomial.len() - 1;
  let n_f = n as f64;
  // convert polynomial to monic form
  let mut monic: Vec<f64> = Vec::new();
  for c in polynomial {
    monic.push(*c / polynomial[n]);
  }
  // let a = - c_1 / n
  let a = -monic[n - 1] / n_f;
  // let z = w + a,
  let p_of_w = {
    // we can recycle monic on the fly.
    for coefficient_index in 0..=n {
      let c = monic[coefficient_index];
      monic[coefficient_index] = 0.0f64;
      for ((index, power), pascal) in zip(
        zip(0..=coefficient_index, (0..=coefficient_index).rev()),
        aberth::PascalRowIter::new(coefficient_index as u32),
      ) {
        let pascal: f64 = pascal as f64;
        monic[index] =
          MulAdd::mul_add(c, pascal * a.powi(power as i32), monic[index]);
      }
    }
    monic
  };
  // convert P(w) into S(w)
  let s_of_w = {
    let mut p = p_of_w;
    // skip the last coefficient
    for i in 0..n {
      p[i] = -p[i].abs()
    }
    p
  };
  // find r_0
  let mut int = 1.0f64;
  let r_0 = loop {
    let s_at_r0 = aberth::sample_polynomial(&s_of_w, int.into());
    if s_at_r0.re > 0.0f64 {
      break int;
    }
    int = int + 1.0f64;
  };
  drop(s_of_w);

  {
    let mut guesses: Vec<Complex<f64>> = Vec::new();

    let frac_2pi_n = f64::PI() * 2.0 / n_f;
    let frac_pi_2n = f64::PI() / 2.0 / n_f;

    for k in 0..n {
      let k_f: f64 = k as f64; 
      let theta = MulAdd::mul_add(frac_2pi_n, k_f, frac_pi_2n);

      let real = MulAdd::mul_add(r_0, theta.cos(), a);
      let imaginary = r_0 * theta.sin();

      let val = Complex::new(real, imaginary);
      guesses.push(val) ;
    }

    guesses
  }
}

pub fn sample_polynomial(coefficients: &Vec<f64>, x: Complex<f64>) -> Complex<f64> {
  debug_assert!(coefficients.len() != 0);
  let mut r = Complex::new(0.0f64, 0.0);
  for c in coefficients.iter().rev() {
    r = r.mul_add(x, c.into())
  }
  r
}

trait ComplexExt<F: Float> {
  fn approx_eq(self, w: Self, epsilon: F) -> bool;
}

impl<F: Float> ComplexExt<F> for Complex<F> {
  /// Cheap comparison of complex numbers to within some margin, epsilon.
  #[inline]
  fn approx_eq(self, w: Complex<F>, epsilon: F) -> bool {
    ((self.re - w.re).powi(2)+(self.im - w.im).powi(2)).sqrt() < epsilon
  }
}

trait Distance<F: Float> {
    fn dist(self, b: Self) -> F;
}

impl<F: Float> Distance<F> for Complex<F> {
  /// Cheap comparison of complex numbers to within some margin, epsilon.
  #[inline]
  fn dist(self, b: Complex<F>) -> F {
    ((self.re - b.re).powi(2)+(self.im - b.im).powi(2)).sqrt() 
  }
}


    



#[cfg(test)]
mod tests{
   
    use super::*;
    use crate::sequence::Sequence;

    use statrs::distribution::ContinuousCDF;

    #[test]
    fn wave_check(){

        let sd = 5;
        let height = 2.0;
        let k = Kernel::new(sd as f64, height);

        let kern = k.get_curve();
        let kernb = &k*4.0;


        assert!(kern.len() == 6*sd+1);

        assert!(kern.iter().zip(kernb.get_curve()).map(|(&a,b)| ((b/a)-4.0).abs() < 1e-6).fold(true, |acc, mk| acc && mk));

        assert!((k.get_sd()-(sd as f64)).abs() < 1e-6);
    }

    #[test]
    fn real_wave_check(){
        let k = Kernel::new(5.0, 2.0);
        let seq = Sequence::new_manual(vec![85;56], vec![84, 68, 72]);
        let mut signal = Waveform::create_zero(&seq, 5);

        unsafe{

        signal.place_peak(&k, 2, 20);

        //Waves are in the correct spot
        assert!((signal.raw_wave()[35]-2.0).abs() < 1e-6);

        signal.place_peak(&k, 1, 20);

        //Waves are in the correct spot
        assert!((signal.raw_wave()[21]-2.0).abs() < 1e-6);

        signal.place_peak(&k, 1, 2);

        //Waves are not contagious
        assert!(signal.raw_wave()[0..17].iter().fold(true, |acc, ch| acc && ((ch-0.0).abs() < 1e-6)));

        }

        let base_w = &signal*0.4;


        let ar: Vec<f64> = vec![0.9, -0.1];

        let background: Background = Background::new(0.25, 2.64, &ar);

        let noise: Noise = signal.produce_noise(&base_w, &background);

        let noi: Vec<f64> = noise.resids();


        let raw_resid = &base_w-&signal;

        let w = raw_resid.raw_wave();

        for i in 0..raw_resid.raw_wave().len(){


            let chopped = raw_resid.start_dats().iter().fold(false, |acc, ch| acc || ((i >= *ch) && (i < *ch+ar.len())));


            let block_id = raw_resid.start_dats().iter().enumerate().filter(|(_, &a)| a <= i).max_by_key(|(_, &value)| value).map(|(idx, _)| idx).unwrap();
            //This gives the index of the maximum start value that still doesn't exceed i, identifying its data block.

            let block_loc = i-raw_resid.start_dats()[block_id];

            if !chopped {

                let start_noi = raw_resid.start_dats()[block_id]-block_id*ar.len();



                let piece: Vec<_> = w[(i-ar.len())..i].iter().rev().collect();

                let sst = ar.iter().zip(piece.clone()).map(|(a, r)| a*r).sum::<f64>() as f64;

                assert!(((noi[start_noi+block_loc-ar.len()] as f64) - ((raw_resid.raw_wave()[i] as f64)-(sst) as f64)).abs() < 1e-6);

            }

        }










    }



    #[test]
    fn noise_check(){

        let ar: Vec<f64> = vec![0.9, -0.1];
        
        let background: Background = Background::new(0.25, 2.64, &ar);

        let n1 = Noise::new(vec![0.4, 0.4, 0.3, 0.2, -1.4], &background);
        let n2 = Noise::new(vec![0.4, 0.4, 0.3, -0.2, 1.4], &background);

        assert!(((&n1*&n2.resids())+1.59).abs() < 1e-6);

        println!("{:?}", n1.resids());
        println!("{:?}", n1.rank());
        println!("{}", n1.ad_calc());

        let mut noise_arr = n1.resids();
        noise_arr.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let noise_length = noise_arr.len();
        let mut ad_try = -(noise_length as f64);

        for i in 0..noise_length {

            let ln_cdf = n1.dist().cdf(noise_arr[i]).ln();
            let ln_sf = (1.-n1.dist().cdf(noise_arr[noise_length-1-i])).ln();
            let mult = (2.0*((i+1) as f64)-1.0)/(noise_length as f64);

            ad_try -= (mult*(ln_cdf+ln_sf));
        }

        println!("calc v try {} {}", n1.ad_calc(), ad_try);
        assert!(((n1.ad_calc()-ad_try)/ad_try).abs() < 5e-2);//Remember, I'm only calculating an approximation. It's going to be a bit off


        let n1 = Noise::new(vec![0.4, 0.39, 0.3, 0.2, -1.4], &background);
        
        assert!(n1.rank().iter().zip([4.,3.,2.,1.,0.]).map(|(&a, b)| a == b).fold(true, |r0, r1| r0 && r1));

        println!("{:?}", n1.ad_grad());

        //let h = 0.0000001;
        let noise_ad = n1.ad_calc();
        let mut noise_arr = n1.resids();
        for i in 0..noise_length {
            let mut noisy = noise_arr.clone();
            let h = unsafe{ std::mem::transmute::<u64,f64>(noisy[i].to_bits() & 0xfff0000000000000)*(1./512.)};
            noisy[i] += h;
            println!("{:?} {} post h and h", noisy, h);
            let n1_plus_h = Noise::new(noisy, &background);
            let diff = (n1_plus_h.ad_calc()-noise_ad)/h;
            println!("{} {} ad_grad", n1.ad_grad()[i], diff);
            assert!(((n1.ad_grad()[i]-diff)/diff).abs() < 5e-2);//Remember, I'm only calculating an approximation. It's going to be a bit off
        }


        //Calculated these with the ln of the numerical derivative of the fast implementation
        //of the pAD function in the goftest package in R
        //This uses Marsaglia's implementation, and is only guarenteed up to 8
        let calced_ads: [(f64, f64); 6] = [(1.0, -0.644472305368), 
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
            assert!(((Noise::ad_like(pairs.0)-pairs.1)/pairs.1).abs() < 5e-2); 

        }


    }

    #[test]
    #[should_panic(expected = "Residuals aren't the same length?!")]
    fn panic_noise() {
        let ar: Vec<f64> = vec![0.9, -0.1];
        
        let background: Background = Background::new(0.25, 2.64, &ar);
        let n1 = Noise::new(vec![0.4, 0.4, 0.3, 0.2, -1.4], &background);
        let n2 = Noise::new(vec![0.4, 0.4, 0.3, -0.2], &background);

        let _ = &n1*&n2.resids();
    }


    #[test]
    #[should_panic(expected = "AR model is not stationary!")]
    fn panic_background() {

        let ar: Vec<f64> = vec![1.5, 1.0];
        
        let background: Background = Background::new(0.25, 2.64, &ar);
    }













}


