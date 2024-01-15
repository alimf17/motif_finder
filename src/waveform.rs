use std::ops::{Add, AddAssign, Sub, SubAssign, Mul};
use std::cmp::{max, min};

//use std::time::{Duration, Instant};

use core::f64::consts::PI;
use core::iter::zip;

use crate::base::Bp;
use crate::sequence::{Sequence, BP_PER_U8};
use crate::modified_t::*;
use crate::data_struct::AllDataUse;

use statrs::distribution::{StudentsT, Continuous, ContinuousCDF};


use aberth;
use num_complex::{Complex, ComplexFloat};
const EPSILON: f64 = 1e-8;

use num_traits::float::{Float, FloatConst};
use num_traits::MulAdd;


use rayon::prelude::*;

use assume::assume;


use serde::{Serialize, Deserialize};

pub const WIDE: f64 = 3.0;



#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Kernel{

    peak_height: f64,
    peak_width: f64,
    kernel: Vec<f64>,
}

impl Mul<f64> for &Kernel {

    type Output = Kernel;

    fn mul(self, rhs: f64) -> Kernel {

        
        Kernel {
            peak_height: self.peak_height*rhs,
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
            peak_height: peak_height,
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
   
    pub fn get_height(&self) -> f64 {
        self.peak_height
    }
    pub fn scaled_kernel(&self, scale_by: f64) -> Kernel {
        self*scale_by
    }


}

//CANNOT BE SERIALIZED
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
       In particular, each block must have 1+floor((block length-1)/spacer) data points
       HOWEVER, this initializer only knows TO panic if the total number of data points
       cannot be reconciled with seq and spacer. If you have one too MANY points in one
       block and one too FEW points in another, this will NOT know to break. If you get 
       this wrong, your run will be memory safe, but it will give garbage and incorrect
       answers.
       */
    pub fn new(start_data: Vec<f64>, seq: &'a Sequence, spacer: usize) -> Waveform<'a> {


        let (point_lens, start_dats) = Self::make_dimension_arrays(seq, spacer);
       
        //println!("{} {} pl {:?}\n sd {:?}", start_data.len(), point_lens.len(), point_lens, start_dats);
        if (point_lens.last().unwrap() + start_dats.last().unwrap()) != start_data.len() {
            panic!("IMPOSSIBLE DATA FOR THIS SEQUENCE AND SPACER")
        }



        Waveform {
            wave: start_data,
            spacer: spacer,
            point_lens: point_lens,
            start_dats: start_dats,
            seq: seq,
        }
    }

    pub fn make_dimension_arrays(seq: &'a Sequence, spacer: usize) -> (Vec<usize>, Vec<usize>) {

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

        let mut size: usize = 0;

        let mut start_dats: Vec<usize> = Vec::new();

        for i in 0..point_lens.len(){
            start_dats.push(size);
            size += point_lens[i];
        }

        (point_lens, start_dats)
    }

    pub fn create_zero(seq: &'a Sequence, spacer: usize) -> Waveform<'a> {
       
        let point_lens: Vec<usize> = seq.block_lens().iter().map(|a| 1+((a-1)/spacer)).collect();

        let mut start_dats: Vec<usize> = Vec::new();

        let mut size: usize = 0;

        for i in 0..point_lens.len(){
            start_dats.push(size);
            size += point_lens[i];
        }

        let tot_l: usize = point_lens.iter().sum();

        Waveform {
            wave: vec![0.0; tot_l],
            spacer: spacer,
            point_lens: point_lens,
            start_dats: start_dats,
            seq: seq,
        }
    }

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

    //SAFETY: -block must be less than the number of blocks
    //        -center must be less than the number of bps in the blockth block
    //        -the length of peak MUST be strictly less than the number of base pairs represented in the smallest data block
    pub(crate) unsafe fn place_peak(&mut self, peak: &Kernel, block: usize, center: usize) {



        //Given how we construct kernels, this will never need to be rounded: they always have an odd number of data points
        let place_bp = (((peak.len()-1)/2) as isize)-(center as isize); //This moves the center of the peak to where it should be, taking the rest of it with it
        let cc = (place_bp).rem_euclid(self.spacer as isize); // This defines the congruence class of the kernel indices that will be necessary for the signal
       
        let zerdat: usize = self.start_dats[block]; //This will ensure the peak is in the correct block

        let min_kern_bp: usize = max(0, place_bp) as usize;
        let nex_kern_bp: usize = min(peak.len() as isize, ((self.spacer*self.point_lens[block]) as isize)+place_bp) as usize; //This is always positive if you uphold the center safety invariant 
 
        //let which_bps = (min_kern_bp..nex_kern_bp).filter(|&bp| ((bp % self.spacer) == (cc as usize)));;
        //let kern_values: Vec<f64> = (min_kern_bp..nex_kern_bp).filter(|&bp| ((bp % self.spacer) == (cc as usize))).map(|f| peak.get_curve()[f as usize]).collect();
        
       

        let completion: usize = ((cc-((peak.len() % self.spacer) as isize)).rem_euclid(self.spacer as isize)) as usize; //This tells us how much is necessary to add to the length 
                                                                            //of the kernel to hit the next base in the cc
        
        let min_kern_cc = max(cc, place_bp);
        let nex_kern_cc = min(((self.point_lens[block]*self.spacer) as isize)+place_bp, (peak.len()+completion) as isize);
        let min_data: usize = ((min_kern_cc-place_bp)/((self.spacer) as isize)) as usize;  //Always nonnegative: if place_bp > 0, min_kern_bp = place_bp
        let nex_data: usize = ((nex_kern_cc-place_bp)/((self.spacer) as isize)) as usize; //Assume nonnegative for the same reasons as nex_kern_bp


        let kern_change = self.wave.get_unchecked_mut((min_data+zerdat)..(nex_data+zerdat));
/*
        if kern_values.len() > 0 {
            //println!("{} {} {} {} {} peak",min_data+zerdat, nex_data+zerdat, kern_values.len(), kern_change.len(), w);
            for i in 0..kern_change.len(){
                kern_change[i] += kern_values[i];
            }
        } 
        
  */
        let kern_start = min_kern_cc as usize;
        for i in 0..kern_change.len() {
            kern_change[i] += peak.kernel.get_unchecked(kern_start+i*self.spacer);
        }


    }


    pub fn kmer_propensities(&self, k: usize) -> Vec<f64> {

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



    pub fn produce_noise<'b>(&self, data_ref: &'b AllDataUse) -> Noise<'b> {
        
        let residual = self-data_ref.data();

        residual.produce_resid_noise(data_ref.background_ref())

    }

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
   
    pub fn account_auto<'b>(&self, background: &'b Background) -> Noise<'b> {

        (self*(1.)).produce_resid_noise(background)

    }

    pub fn produce_resid_noise<'b>(&self, background: &'b Background) -> Noise<'b> {
       
        let residual = self;

        if !(background.is_ar()){
            return Noise::new(self.wave.clone(), background);
        }

        let mut end_dats = residual.start_dats()[1..residual.start_dats.len()].to_vec();

        let resid = &residual.wave;
        
        end_dats.push(resid.len());

        let mut len_penalties = vec![0usize; end_dats.len()];

        for k in 0..end_dats.len() {
            len_penalties[k] = (k+1)*background.ar_corrs.as_ref().expect("We short circuited if this was None.").len();
        }

        let filt_lens: Vec<usize> = end_dats.iter().zip(len_penalties).map(|(a, b)| a-b).collect();

        let l_c = background.ar_corrs.as_ref().expect("We short circuited if this was None.").len();

        let mut fin_noise: Vec<f64> = vec![0.0; filt_lens.iter().sum()];

        for k in 0..end_dats.len(){

            let sind: usize = if k == 0 {0} else {end_dats[k-1]};


            let mut block: Vec<f64> = resid[(sind+l_c)..end_dats[k]].to_vec();
            
            for i in 0..l_c {
                for j in 0..block.len() {
                    block[j] -= background.ar_corrs.as_ref().expect("We short circuited if this was None.")[i]*resid[sind+l_c+j-(i+1)];
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

    pub fn generate_all_locs(&self) -> Vec<usize> {

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
    pub(crate) fn get_waveform<'a>(&self, seq: &'a Sequence) -> Waveform<'a> {
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


#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Background {
    pub dist: BackgroundDist,
    pub kernel: Kernel,
    pub ar_corrs: Option<Vec<f64>>,
}

impl Background {

    pub fn new(sigma_background : f64, df : f64, peak_width: f64, poss_ar_corrs: Option<&Vec<f64>>) -> Background {

        let dist = BackgroundDist::new(sigma_background, df);
        
        let kernel = Kernel::new(peak_width, 1.0);

        match poss_ar_corrs {
            
            Some(ar_corrs) => {
                let mut poly = ar_corrs.iter().map(|a| -1.0*a).collect::<Vec<f64>>();
                poly.splice(0..0, [1.0]);

                let roots = aberth_vec(&poly, EPSILON).unwrap();

                for root in roots {
                    if root.abs() <= 1.0+EPSILON { //Technically, the +EPSILON means that we might rule out some stationary models
                        panic!("AR model is not stationary!")
                    }
                }
                
                Background{dist: dist, kernel: kernel, ar_corrs:Some(ar_corrs.clone())}
            }, 
            None => Background{dist: dist, kernel: kernel, ar_corrs: None},
        }
    }

    pub fn is_ar(&self) -> bool {
        self.ar_corrs.is_some()
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
    pub fn kernel_ref(&self) -> &Kernel {
        &self.kernel
    }


}


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
    
    pub fn dist(&self) -> BackgroundDist {
        self.background.dist.clone()
    }

    
    pub fn rank(&self) -> Vec<f64> {

        
        //This generates a vector where each element is tagged with its original position
        let mut rx: Vec<(usize, f64)> = self.resids.clone().iter().enumerate().map(|(a, b)| (a, *b)).collect();

        //This sorts the elements, but recalls their original positions
        rx.par_sort_unstable_by(|(i,a), (j,b)| a.partial_cmp(b).expect(format!("No NaNs allowed! {} {} {} {}", i,a, j, b).as_str()));
       
    
        
        let mut ranks: Vec<f64> = vec![0.; rx.len()];

        let mut ind: f64 = 0.0;

        //This uses the sort of elements to put rankings in their original positions
        for &(i, _) in &rx {
            ranks[i] = ind;
            ind += 1.0;
        }

        
        ranks
        
        
    }


    pub fn ad_calc(&self) -> f64 {

        //let time = Instant::now();
        let mut forward: Vec<f64> = self.resids();
        forward.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let n = forward.len();
        let mut a_d = -(n as f64);
        for i in 0..n {

            let cdf = self.background.ln_cdf(forward[i]);
            let rev_sf = self.background.ln_sf(forward[n-1-i]);
            a_d -= (cdf+rev_sf) * ((2*i+1) as f64)/(n as f64)
        }
        a_d
    }

    
    pub fn ad_grad(&self) -> Vec<f64> {

        //let start = Instant::now();
        let ranks = self.rank();

        let n = self.resids.len() as f64;
       
        let mut derivative: Vec<f64> = vec![0.0; self.resids.len()];

        for i in 0..self.resids.len() {
            let pdf = self.background.pdf(self.resids[i]);
            let (cdf, sf) = self.background.cd_and_sf(self.resids[i]);
            let mut poc = pdf/cdf;
            if !poc.is_finite() {poc = 0.;} //We assume that pdf outpaces cdf to 0 mathematically, but some cdf implementations are lazier sooner
            let mut pos = pdf/sf;
            if !pos.is_finite() {pos = 0.;} //We assume that pdf outpaces sf to 0 mathematically, but some sf implementations are lazier sooner

            //derivative[i] = -((2.*ranks[i]+1.)*pdf)/(n*cdf)+((2.*n-(2.*ranks[i]+1.))*pdf)/(n*sf);
            derivative[i] = -((2.*ranks[i]+1.)*poc)/(n)+((2.*n-(2.*ranks[i]+1.))*pos)/(n);
        }


        //println!("ad_grad {:?}", start.elapsed());
        derivative

    }

    //My low approximation is screwed up. It's derived from the low tail approximation of the
    //marsaglia (2004) paper
    fn low_val(la: f64) -> f64 {

        const C: f64 = PI*PI/8.0;
        -C/la-2.5*la.ln()+Self::polynomial_sqrt_la(la).ln()
    }

    fn deriv_low_val(la: f64) -> f64 {

        const C: f64 = PI*PI/8.0;
        C/(la*la)- 2.5/la + Self::deriv_polynomial_sqrt_la(la)/Self::polynomial_sqrt_la(la)

    }

    fn polynomial_sqrt_la(la: f64) -> f64 {

        const C: f64 = PI*PI/8.0;
        const CFS: [f64; 6] = [2.00012,0.247105,0.0649821, 0.0347962,0.0116720,0.00168691];

        let mut p: f64 = C*CFS[0]+(C*CFS[1]-CFS[0]/2.)*la.powi(1)+4.5*CFS[5]*la.powi(6);

        for i in 2..CFS.len() {
            p += (-1_f64).powi((i+1) as i32)*(C*CFS[i]-((2*i-3) as f64)*CFS[i-1]/2.)*la.powi(i as i32);
        }
        return p

    }

    fn deriv_polynomial_sqrt_la(la:f64) -> f64 {

        const C: f64 = PI*PI/8.0;
        const CFS: [f64; 6] = [2.00012,0.247105,0.0649821, 0.0347962,0.0116720,0.00168691];

        let mut p: f64 = (C*CFS[1]-CFS[0]/2.)+27.*CFS[5]*la.powi(5);

        for i in 2..CFS.len() {
            p+= (-1_f64).powi((i+1) as i32)*(i as f64)*(C*CFS[i]-((2*i-3) as f64)*CFS[i-1]/2.)*la.powi((i-1) as i32);
        }

        return p

    }

    fn high_val(ha: f64) -> f64 {

        (3.0/(ha*PI)).ln()/2.0-ha

    }

    fn deriv_high_val(ha: f64) -> f64 {
        -0.5/ha-1.
    }

    fn weight_fn(a: f64) -> f64 {
        const A0: f64 = 2.0;
        const K: f64 = 16.0;
        1.0/(1.0+(a/A0).powf(K))
    }

    fn deriv_weight_fn(a: f64) -> f64 {
        const A0: f64 = 2.0;
        const K: f64 = 16.0;
        let aa = a/A0;
        -(K/A0)*(aa.powf(K-1.)/(1.+aa.powf(K)).powi(2))
    }

    pub fn ad_like(a: f64) -> f64 {


        //This has to exist because of numerical issues causing NANs otherwise
        //This is basically caused by cdf implementations that return 0.0.
        //My FastT implementation can't, but the statrs normal DOES. 
        if a == f64::INFINITY { return -f64::INFINITY;} 

        let lo = Self::low_val(a);
        let hi = Self::high_val(a);
        let w = Self::weight_fn(a);

        w*lo+(1.0-w)*hi
    }

    pub fn ad_diff(a: f64) -> f64 {

        const H: f64 = 0.0000001;
        (Self::ad_like(a+H)-Self::ad_like(a))/H
    }

    pub fn ad_deriv(a: f64) -> f64 {
        
        if a.is_infinite() {
            return 0.0;
        }

        let w = Self::weight_fn(a);
        let dl = Self::deriv_low_val(a);
        let dh = Self::deriv_high_val(a);
        let dw = Self::deriv_weight_fn(a);
        let lv = Self::low_val(a);
        let hv = Self::high_val(a);


        //println!("a: {}, w: {}, dl: {}, hl: {}, dw: {}, lv: {}, hv: {}", a, w, dl, hl, dw, lv, hv);
        //w*Self::deriv_low_val(a)+(1.-w)*Self::deriv_high_val(a)
          //  +Self::deriv_weight_fn(a)*(Self::low_val(a)-Self::high_val(a))
        w*dl+(1.-w)*dh + dw*(lv-hv)
       
    }


}

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
    use rand::Rng;
    use std::time::Instant;

    fn empirical_noise_grad(n: &Noise) -> Vec<f64>{

        let h = 0.00001;
        let back = n.background;
        let ad = n.ad_calc();
        let mut grad = vec![0_f64; n.resids.len()]; 
        for i in 0..grad.len() {
             let mut n_res = n.resids();
             n_res[i] += h;
             let nnoise = Noise::new(n_res, back);
             grad[i] = (nnoise.ad_calc()-ad)/h;
        }
        grad
    }

    #[test]
    fn wave_check(){

        let sd = 5;
        let height = 2.0;
        let spacer = 5;
        let k = Kernel::new(sd as f64, height);

        let kern = k.get_curve();
        let kernb = &k*4.0;


        println!("kern len {}", (kern.len()));
        assert!(kern.len() == 6*(sd as  usize)+1);

        assert!(kern.iter().zip(kernb.get_curve()).map(|(&a,b)| ((b/a)-4.0).abs() < 1e-6).fold(true, |acc, mk| acc && mk));

        assert!((k.get_sd()-(sd as f64)).abs() < 1e-6);
    }

    #[test]
    fn real_wave_check(){
        let k = Kernel::new(5.0, 2.0);
        //let seq = Sequence::new_manual(vec![85;56], vec![84, 68, 72]);
        let seq = Sequence::new_manual(vec![192, 49, 250, 10, 164, 119, 66, 254, 19, 229, 212, 6, 240, 221, 195, 112, 207, 180, 135, 45, 157, 89, 196, 117, 168, 154, 246, 210, 245, 16, 97, 125, 46, 239, 150, 205, 74, 241, 122, 64, 43, 109, 17, 153, 250, 224, 17, 178, 179, 123, 197, 168, 85, 181, 237, 32], vec![84, 68, 72]);
        let mut signal = Waveform::create_zero(&seq, 5);

        unsafe{

        signal.place_peak(&k, 1, 20);

        let t = Instant::now();
        //kmer_propensities is tested by inspection, not asserts, because coming up with a good assert case was hard and I didn't want to
        //It passed the inspections I gave it, though
        let propens = signal.kmer_propensities(9);
        println!("duration {:?}", t.elapsed());

        //println!("{:?}", signal);
        println!("{:?}", propens);
        //Waves are in the correct spot
        assert!((signal.raw_wave()[21]-2.0).abs() < 1e-6);

        signal.place_peak(&k, 1, 2);

        println!("after second ins {:?}", signal.kmer_propensities(9));
        //Waves are not contagious
        assert!(signal.raw_wave()[0..17].iter().fold(true, |acc, ch| acc && ((ch-0.0).abs() < 1e-6)));

        //point_lens: Vec<usize>,
        //start_dats: Vec<usize>,
        //

        signal.place_peak(&k, 1, 67);

        //Waves are not contagious
        assert!(signal.raw_wave()[(signal.start_dats[2])..(signal.start_dats[2]+signal.point_lens[2])].iter().fold(true, |acc, ch| acc && ((ch-0.0).abs() < 1e-6)));

        signal.place_peak(&k, 2, 20);

        //Waves are in the correct spot
        assert!((signal.raw_wave()[35]-2.0).abs() < 1e-6);

        //This is a check just for miri
        signal.place_peak(&k, 2, 70)
        }

        let base_w = &signal*0.4;


        let ar: Vec<f64> = vec![0.9, -0.1];

        let background: Background = Background::new(0.25, 2.64, 5.0, Some(&ar));

        let data_seq = unsafe{ AllDataUse::new_unchecked_data(base_w, &background)};

        let noise: Noise = signal.produce_noise(&data_seq);

        let noi: Vec<f64> = noise.resids();


        let raw_resid = &signal-data_seq.data();

        let w = raw_resid.raw_wave();

        println!("Noi {:?}", noi);
        for i in 0..raw_resid.raw_wave().len(){


            let chopped = raw_resid.start_dats().iter().fold(false, |acc, ch| acc || ((i >= *ch) && (i < *ch+ar.len())));


            let block_id = raw_resid.start_dats().iter().enumerate().filter(|(_, &a)| a <= i).max_by_key(|(_, &value)| value).map(|(idx, _)| idx).unwrap();
            //This gives the index of the maximum start value that still doesn't exceed i, identifying its data block.

            let block_loc = i-raw_resid.start_dats()[block_id];

            if !chopped {

                let start_noi = raw_resid.start_dats()[block_id]-block_id*ar.len();



                let piece: Vec<_> = w[(i-ar.len())..i].iter().rev().collect();

                let sst = ar.iter().zip(piece.clone()).map(|(a, r)| a*r).sum::<f64>() as f64;

                println!("i {} noi raw res {}",i, ((noi[start_noi+block_loc-ar.len()] as f64) - ((raw_resid.raw_wave()[i] as f64)-(sst) as f64)));
                assert!(((noi[start_noi+block_loc-ar.len()] as f64) - ((raw_resid.raw_wave()[i] as f64)-(sst) as f64)).abs() < 1e-6);

            }

        }










    }



    #[test]
    fn noise_check(){

        let ar: Vec<f64> = vec![0.9, -0.1];
        
        let background: Background = Background::new(0.25, 2.64, 5.0, Some(&ar));

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

            let ln_cdf = n1.dist().ln_cdf(noise_arr[i]);
            let ln_sf = n1.dist().ln_sf(noise_arr[noise_length-1-i]);
            let mult = (2.0*((i+1) as f64)-1.0)/(noise_length as f64);

            
            ad_try -= mult*(ln_cdf+ln_sf);
        }

        println!("calc v try {} {}", n1.ad_calc(), ad_try);
        assert!(((n1.ad_calc()-ad_try)/ad_try).abs() < 1e-4);//Remember, I'm only calculating an approximation. It's going to be a bit off


        let n1 = Noise::new(vec![0.4, 0.39, 0.3, 0.2, -1.4], &background);
        
        assert!(n1.rank().iter().zip([4.,3.,2.,1.,0.]).map(|(&a, b)| a == b).fold(true, |r0, r1| r0 && r1));

        println!("{:?}", n1.ad_grad());

        let mut rng = rand::thread_rng();
        let n1v: Vec<f64> = (0..1000).map(|_| rng.gen::<f64>()*2.0-1.).collect();
        let n1 = Noise::new(n1v, &background);
        //let h = 0.0000001;
        let noise_ad = n1.ad_calc();
        println!("diff {} deriv {} diffmderiv {}", Noise::ad_diff(noise_ad), Noise::ad_deriv(noise_ad), Noise::ad_diff(noise_ad)-Noise::ad_deriv(noise_ad));

        let mut rng = rand::thread_rng();
        let ads: Vec<f64> = (0..1000).map(|_| rng.gen::<f64>()).collect();
        let mut noise_arr = n1.resids();
        let mut noise_diffs: Vec<(f64, f64, f64)> = vec![(0.,0.,0.);noise_arr.len()];
        for i in 0..(noise_arr.len()) {
            let mut noisy = noise_arr.clone();
            let bits: u64 = 0x3e80000000000000;
            //let h = unsafe{ std::mem::transmute::<u64,f64>(noisy[i].to_bits() & 0xfff0000000000000)*std::mem::transmute::<u64,f64>(0x3f10000000000000) };//*(1./2048.)};
            let h = f64::from_bits(bits);//*(1./2048.)};
            noisy[i] += h;
            println!("{} vs {}. {} post h and h", noise_arr[i], noisy[i], h);
            let n1_plus_h = Noise::new(noisy, &background);
            let diff = (n1_plus_h.ad_calc()-noise_ad)/h;
            println!("{} {} {} {} {} {} {} {} ad_grad", noise_ad, n1_plus_h.resids[i], noise_arr[i], n1.ad_grad()[i], diff, (n1.ad_grad()[i]-diff), (n1.ad_grad()[i]-diff)/diff, h);
            noise_diffs[i] = (noise_arr[i],n1.ad_grad()[i]-diff, ((n1.ad_grad()[i]-diff)/diff));//Remember, I'm only calculating an approximation. It's going to be a bit off. And this does get better as I 
                                                                //reduce h, up to a point (whereupon I'm almost certain I'm running into numerical issues)
        }


        let faulty_inds: Vec<(usize, (f64, f64, f64))> = noise_diffs.iter().enumerate().filter(|(_, &(a,b,c))| b.abs() > 1e-5).map(|(a, &b)| (a, b)).collect();

        if faulty_inds.len() > 0 {
            panic!("mismatches: {:?}", faulty_inds);
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
            println!("Noi {} {} {} {}",Noise::ad_like(pairs.0), pairs.1, Noise::ad_like(pairs.0)-pairs.1, (Noise::ad_like(pairs.0)-pairs.1)/pairs.1);
            assert!(((Noise::ad_like(pairs.0)-pairs.1)/pairs.1).abs() < 5e-2); 

        } 

    }

    #[test]
    #[should_panic(expected = "Residuals aren't the same length?!")]
    fn panic_noise() {
        let ar: Vec<f64> = vec![0.9, -0.1];
        
        let background: Background = Background::new(0.25, 2.64, 5.0, Some(&ar));
        let n1 = Noise::new(vec![0.4, 0.4, 0.3, 0.2, -1.4], &background);
        let n2 = Noise::new(vec![0.4, 0.4, 0.3, -0.2], &background);

        let _ = &n1*&n2.resids();
    }


    #[test]
    #[should_panic(expected = "AR model is not stationary!")]
    fn panic_background() {

        let ar: Vec<f64> = vec![1.5, 1.0];
        
        let background: Background = Background::new(0.25, 2.64,5.0, Some(&ar));
    }













}


