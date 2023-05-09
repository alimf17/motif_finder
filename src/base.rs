//pub mod bases {
    use rand::Rng;
    use rand::seq::SliceRandom;
    use rand::distributions::{Distribution, Uniform};
    use statrs::distribution::{Continuous, ContinuousCDF, LogNormal, Normal, Dirichlet, Exp};
    use statrs::statistics::{Min, Max, Distribution as OtherDistribution};
    use crate::waveform::{Kernel, Waveform, Noise};
    use crate::sequence::{Sequence, BP_PER_U8};
    use statrs::function::gamma;
    use statrs::{consts, Result, StatsError};
    use std::f64;
    use std::fmt;
    use std::collections::{VecDeque, HashMap};
    use std::time::{Duration, Instant};
    use once_cell::sync::Lazy;

    use rayon::prelude::*;
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

    pub const THRESH: f64 = 1e-4;

    //When converting between gradient compatible and incompatible representations
    //We sometimes end up with numerical errors that basically make infinities where there shouldn't be
    //CONVERSION_MARGIN protects us during conversion, and REFLECTOR cuts off bases that could get too weak for proper conversion
    //These numbers were empirically determined, not theoretically. 
    const CONVERSION_MARGIN: f64 = 1e-6;
    const REFLECTOR: f64 = 15.0;
    static DGIBBS_CUTOFF: Lazy<f64> = Lazy::new(|| -RT*THRESH.ln());
    static PROP_CUTOFF: Lazy<f64> = Lazy::new(|| THRESH); 
    static PROP_UPPER_CUTOFF: Lazy<f64> = Lazy::new(|| (-CONVERSION_MARGIN/RT).exp());

    static BASE_DIST: Lazy<Exp> = Lazy::new(|| Exp::new(1.0).unwrap());

    //BEGIN BASE

    pub struct Base {
       props: [ f64; BASE_L],
    }

    impl Clone for Base {

        fn clone(&self) -> Self {

            Self::new(self.props)

        }

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

            let eps = 1e-12;
 
            //let norm: f64 = props.iter().sum();

            let mut any_neg: bool = false;

            for i in 0..props.len() {
                any_neg |= props[i] < 0.0 ;
            }

            let max = Self::max(&props);
            
            props = props.iter().map(|a| a/max).collect::<Vec<_>>().try_into().unwrap();

            let mut rng = rand::thread_rng();
            //We can rely on perfect float equality because max copies its result
            let mut maxes = props.iter().enumerate().filter(|(_,&a)| (a == max)).map(|(b, _)| b).collect::<Vec<usize>>(); 
            let mut one_max = (maxes.len() == 1);
           

            //We just want to decrement the maximum slightly
            //Our implementation breaks if there are two best bases
            while !one_max {
                let ind: usize = *maxes.choose(&mut rng).unwrap();
                props[ind] = *PROP_UPPER_CUTOFF;
                maxes = props.iter().enumerate().filter(|(_,&a)| (a == max)).map(|(b, _)| b).collect::<Vec<usize>>();
                one_max = (maxes.len() == 1);
            }

           
            for i in 0..props.len() {
                if props[i] < *PROP_CUTOFF {props[i] = *PROP_CUTOFF }
            }

            if any_neg{ //|| ((norm - 1.0).abs() > 1e-12) {
               panic!("All is positive: {}. Norm is .", !any_neg)
            }

            Base { props }
        }


        //Safety: props must be an array with 
        //      1) exactly one element = 1.0
        //      2) all other elements between *PROP_CUTOFF and *PROP_UPPER_CUTOFF
        //Otherwise, this WILL break. HARD. And in ways I can't determine
        pub unsafe fn proper_new(props: [ f64; 4]) -> Base {
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
                att[i] = (*BASE_DIST).sample(&mut rng);
            }

            let att_sum: f64 = att.iter().sum();

            //The code from here until the unsafe line is us ABSOLUTELY GUARENTEEING that we fulfill the safety requirements of proper_new
            let max: f64 = Self::max(&att);

            let mut maxes = att.iter().enumerate().filter(|(_,&a)| (a == max)).map(|(b, _)| b).collect::<Vec<usize>>();

            let max_ind: usize = *maxes.choose(&mut rng).unwrap();


            
            for i in 0..att.len() {
                if i == max_ind {
                    att[i] = 1.0;
                } else {
                    att[i] =  (att[i]/att_sum)*(*PROP_UPPER_CUTOFF-*PROP_CUTOFF)+*PROP_CUTOFF;
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


        pub fn to_gbase(&self) -> GBase {

            let max = self.props.iter().copied().fold(f64::NAN, f64::max);

            let best = Self::argmax(&self.props);

            let mut dgs = [0.0_f64 ; BASE_L-1];

            let mut ind = 0;

            for i in 0..self.props.len() {
                if i != Self::argmax(&self.props) {
                    //dgs[ind] = -RT*(self.props[i]/max).ln();
                    dgs[ind] = -RT*(self.props[i]).ln(); //Trying for a proportion binding implementation
                    ind += 1;
                }
            }

            let prelim_dg = dgs.iter().map(|&a| (a-CONVERSION_MARGIN)/(*DGIBBS_CUTOFF-CONVERSION_MARGIN)).collect::<Vec<f64>>();

            dgs = prelim_dg.iter().map(|a| (a/(1.0-a)).ln()).collect::<Vec<f64>>().try_into().unwrap();
                                               

            let newdg: GBase = GBase {
                best: best,
                dgs: dgs,
            };

            newdg

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
    
    //BEGIN GBASE

    pub struct GBase {
        best: usize,
        dgs: [ f64; BASE_L-1],

    }

    impl GBase {

        pub fn new(dg: [ f64; BASE_L-1], be: usize) -> GBase{

            GBase{ best: be, dgs: dg }
        }

        pub fn to_base(&self) -> Base {

 

            let mut prop_pre = self.dgs.iter().map(|&a| CONVERSION_MARGIN+((*DGIBBS_CUTOFF-CONVERSION_MARGIN)/(1.0+((-a).exp())))).collect::<Vec<f64>>();

            let mut props = [0.0_f64 ; BASE_L];


            for i in 0..props.len() {

                if i < self.best {
                    props[i] = (-prop_pre[i]/RT).exp();
                } else if i == self.best {
                    props[i] = 1.0_f64;
                } else {
                    props[i] = (-prop_pre[i-1]/RT).exp();
                }
            }

            /* let norm: f64 = props.iter().sum();

            for p in 0..props.len() {
                props[p] /= norm;
            } */ // Proportion binding implementation of base


            let based: Base = Base {
                props: props,
            };

            based

        }

        pub fn deltas(&self) -> [ f64; BASE_L-1] {
            self.dgs.clone()
        }
    }

    //BEGIN MOTIF
    pub struct Motif<'a> {
    
        peak_height: f64,
        kernel: Kernel,
        pwm: Vec<Base>,
        poss: bool,
        seq: &'a Sequence,

    }

    impl<'a> Motif<'a> {

        //GENERATORS
        //TODO: make sure that this is used iff there's a guarentee that a motif is allowed
        pub unsafe fn raw_pwm(pwm: Vec<Base>, peak_height: f64, peak_width: f64, seq: &'a Sequence) -> Motif<'a> {
            let kernel = Kernel::new(peak_width, peak_height);

            Motif {
                peak_height: peak_height,
                kernel: kernel,
                pwm: pwm,
                poss: true,
                seq: seq,
            }
        }

 
        //TODO: make sure that this is used iff there's a guarentee that a motif is allowed
        pub fn from_motif(best_bases: Vec<usize>, peak_width: f64, seq: &'a Sequence) -> Motif<'a> {

            let pwm: Vec<Base> = best_bases.iter().map(|a| Base::from_bp(*a)).collect();

            let height_dist: TruncatedLogNormal = TruncatedLogNormal::new(LOG_HEIGHT_MEAN, LOG_HEIGHT_SD, MIN_HEIGHT, MAX_HEIGHT).unwrap();

            let mut rng = rand::thread_rng();

            let sign: f64 = rng.gen();
            let sign: f64 = if sign < PROB_POS_PEAK {1.0} else {-1.0};

            let peak_height: f64 = sign*height_dist.sample(&mut rng);

            let kernel = Kernel::new(peak_width, peak_height);

            let poss = seq.kmer_in_seq(&best_bases);

            Motif {
                peak_height: peak_height,
                kernel: kernel,
                pwm: pwm,
                poss: poss,
                seq: seq,
            }


        }


        //Safety: best_bases must appear as a subsequence in seq at least once.
        //        WARNING: failing this safety check will NOT make everything crash and burn. 
        //        THIS WILL BREAK SILENTLY IF SAFETY GUARENTEES ARE NOT KEPT
        pub unsafe fn from_clean_motif(best_bases: Vec<usize>, peak_width: f64, seq: &'a Sequence) -> Motif<'a> {

            let pwm: Vec<Base> = best_bases.iter().map(|a| Base::from_bp(*a)).collect();

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
                poss: true,
                seq: seq,
            }


        }
        //TODO: make sure that this takes a sequence reference variable and that we pull our random motif from there
        pub fn rand_mot(peak_width: f64, seq: &'a Sequence) -> Motif<'a> {

            let mut rng = rand::thread_rng();

            let num_bases = rng.gen_range(MIN_BASE..(MAX_BASE+1));

            let mot = seq.random_valid_motif(num_bases);

            unsafe {
                Self::from_clean_motif(mot, peak_width, &seq)
            }

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

        pub fn raw_kern(&self) -> &Vec<f64> {
            self.kernel.get_curve()
        }

        pub fn peak_width(&self) -> f64 {
            self.kernel.get_sd()
        }

        pub fn len(&self) -> usize {
            self.pwm.len()
        }

        pub fn seq(&self) -> &Sequence {
            self.seq
        }

        pub fn poss(&self) -> bool {
            self.poss
        }

        //PRIORS

        pub fn pwm_prior(&self) -> f64 {

            if self.poss {
                //We have to normalize by the probability that our kmer is possible
                //Which ultimately is to divide by the fraction (number unique kmers)/(number possible kmers)
                //number possible kmers = BASE_L^k, but this actually cancels with our integral
                //over the regions of possible bases, leaving only number unique kmers. 
                let mut prior = -((self.seq.number_unique_kmers(self.len()) as f64).ln()); 

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
        /*pub fn prop_binding(&self, kmer: &[usize]) -> f64 { 

            let bind_forward: f64 = self.pwm.iter().zip(kmer).map(|(a, &b)| a.rel_bind(b)).product::<f64>();

            return bind_forward

        }*/

        pub fn return_bind_score(&self) -> (Vec<f64>, Vec<bool>) {

            //let seq = self.seq;
            let coded_sequence = self.seq.seq_blocks();
            let block_lens = self.seq.block_lens(); //bp space
            let block_starts = self.seq.block_u8_starts(); //stored index space


            let mut bind_scores: Vec<f64> = vec![0.0; 4*coded_sequence.len()];
            let mut rev_comp: Vec<bool> = vec![false; 4*coded_sequence.len()];

            let mut uncoded_seq: Vec<usize> = vec![0; self.seq.max_len()];

            //let seq_ptr = uncoded_seq.as_mut_ptr();
            //let bind_ptr = bind_scores.as_mut_ptr();
            //let comp_ptr = rev_comp.as_mut_ptr();

            let seq_frame = 1+(self.len()/4);


            let mut ind = 0;

            let mut store = Sequence::code_to_bases(coded_sequence[0]);

            //let mut bind_forward: f64 = 1.0;
            //let mut bind_reverse: f64 = 1.0;
            
            //let ilen: isize = self.len() as isize;

            {
            let uncoded_seq = uncoded_seq.as_mut_slice();
            for i in 0..(block_starts.len()) {


                for jd in 0..(block_lens[i]/4) {

                    store = Sequence::code_to_bases(coded_sequence[block_starts[i]+jd]);
                    for k in 0..4 {
                        uncoded_seq[4*jd+k] = store[k];
                        //actual_frame[4*jd+k] = store[k];
                    }

                }


                for j in 0..((block_lens[i])-self.len()) {

                    ind = 4*block_starts[i]+(j as usize);
                    //
                    //ind = j+4*(block_starts[i] as isize);
                    //bind_forward = 1.0;
                    //bind_reverse = 1.0;

                    
                    let binding_borrow = unsafe { uncoded_seq.get_unchecked(j..(j+self.len())) };

                    
                    (bind_scores[ind], rev_comp[ind]) = unsafe {self.prop_binding(binding_borrow) };
                    
                    //(bind_scores[ind], rev_comp[ind]) = self.prop_binding(&uncoded_seq[j..(j+self.len())]);
                }

            }
            }



            (bind_scores, rev_comp)



        }
       
        //NOTE: this will technically mark a base as present if it's simply close enough to the beginning of the next sequence block
        //      This is technically WRONG, but it's faster and shouldn't have an effect because any positions marked incorrectly
        //      as true will have binding scores of 0
        pub fn base_check(&self, rev_comp: &Vec<bool>, bp: usize, motif_pos: usize) -> Vec<bool> {
                
            let coded_sequence = self.seq.seq_blocks();

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
        pub fn generate_waveform(&'a self, DATA: &'a Waveform) -> Waveform {

            let mut occupancy_trace: Waveform = DATA.derive_zero();

            let mut actual_kernel: Kernel = &self.kernel*1.0;

            let starts = self.seq.block_u8_starts();

            let lens = self.seq.block_lens();

            let (bind_score_floats, bind_score_revs) = self.return_bind_score();

            //let mut count: usize = 0;

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

        pub fn no_height_waveform(&'a self, DATA: &'a Waveform) -> Waveform {

            let mut occupancy_trace: Waveform = DATA.derive_zero();

            //let base_kernel = &self.kernel*(1.0/self.peak_height);
            //let mut actual_kernel: Kernel = &base_kernel*1.0;
            let mut actual_kernel: Kernel = &self.kernel*1.0;

            let starts = self.seq.block_u8_starts();

            let lens = self.seq.block_lens();

            let (bind_score_floats, bind_score_revs) = self.return_bind_score();

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
        
        pub fn only_pos_waveform(&'a self,bp: usize, motif_pos: usize, DATA: &'a Waveform) -> Waveform {

            let mut occupancy_trace: Waveform = DATA.derive_zero();

            let mut actual_kernel: Kernel = &self.kernel*1.0;

            let starts = self.seq.block_u8_starts();

            let lens = self.seq.block_lens();

            let (bind_score_floats, bind_score_revs) = self.return_bind_score();

            let checked = self.base_check(&bind_score_revs, bp, motif_pos);
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
        unsafe fn generate_waveform_from_binds(&'a self, binds: &(Vec<f64>, Vec<bool>), DATA: &'a Waveform) -> Waveform {

            let mut occupancy_trace: Waveform = DATA.derive_zero();

            let mut actual_kernel: Kernel = &self.kernel*1.0;

            let starts = self.seq.block_u8_starts();

            let lens = self.seq.block_lens();


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
        unsafe fn no_height_waveform_from_binds(&'a self, binds: &(Vec<f64>, Vec<bool>), DATA: &'a Waveform) -> Waveform {

            let mut occupancy_trace: Waveform = DATA.derive_zero();

            let base_kernel = &self.kernel*(1.0/self.peak_height);
            let mut actual_kernel: Kernel = &base_kernel*1.0;
            //let mut actual_kernel: Kernel = &self.kernel*1.0;

            let starts = self.seq.block_u8_starts();

            let lens = self.seq.block_lens();

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
        unsafe fn only_pos_waveform_from_binds(&'a self, binds: &(Vec<f64>, Vec<bool>), bp: usize, motif_pos: usize, DATA: &'a Waveform) -> Waveform {

            let mut occupancy_trace: Waveform = DATA.derive_zero();

            let mut actual_kernel: Kernel = &self.kernel*1.0;

            let starts = self.seq.block_u8_starts();

            let lens = self.seq.block_lens();

            let checked = self.base_check(&binds.1, bp, motif_pos);

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
        pub fn single_motif_grad(&'a self,  DATA: &'a Waveform, noise: &'a Noise) -> (f64, Vec<f64>) {

            let binds = self.return_bind_score();


            let d_ad_stat_d_noise: Vec<f64> = noise.ad_grad();

            let d_ad_like_d_ad_stat: f64 = Noise::ad_diff(noise.ad_calc());
            

            //End preuse generation
            let d_noise_d_h = unsafe { self.no_height_waveform_from_binds(&binds, DATA)
                                            .produce_noise(DATA, noise.background)};
            let d_ad_like_d_grad_form_h = d_ad_like_d_ad_stat * (-(&d_noise_d_h * &d_ad_stat_d_noise)) 
                                        * (self.peak_height().abs()-MIN_HEIGHT).powi(2)
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
                      * prop_bp * (-RT*prop_bp.ln()-CONVERSION_MARGIN).powi(2)
                      / (-RT * (*DGIBBS_CUTOFF-CONVERSION_MARGIN)) } ;
                
            }

                
            (d_ad_like_d_grad_form_h, d_ad_like_d_grad_form_binds)


        }
   
        pub fn parallel_single_motif_grad(&'a self,  DATA: &'a Waveform, noise: &'a Noise) -> Vec<f64> {

            let binds = self.return_bind_score();


            let d_ad_stat_d_noise: Vec<f64> = noise.ad_grad();

            let d_ad_like_d_ad_stat: f64 = Noise::ad_diff(noise.ad_calc());
            

            //End preuse generation
            let d_noise_d_h = unsafe { self.no_height_waveform_from_binds(&binds, DATA)
                                            .produce_noise(DATA, noise.background)};
          

            //let mut d_ad_like_d_grad_form_binds: Vec<f64> = vec![0.0; self.len()*(BASE_L-1)];


            let n = self.len()*(BASE_L-1);


            let d_ad_like_d_grad_form: Vec<f64> = (0..n).into_par_iter().map(|i| {
                if i == 0 {
                    d_ad_like_d_ad_stat * (-(&d_noise_d_h * &d_ad_stat_d_noise))
                    * (self.peak_height().abs()-MIN_HEIGHT).powi(2)
                    / (self.peak_height().signum() * (MAX_HEIGHT-MIN_HEIGHT))
                } else {
                    let index = i-1;
                    let base_id = index/(BASE_L-1); //Remember, this is integer division, which Rust rounds down
                    let mut bp = index % (BASE_L-1);
                    bp += if bp >= self.pwm[base_id].best_base() {1} else {0}; //If best_base == BASE_L-1, then we just skipped it like we're supposed to

                    let prop_bp = unsafe { self.pwm[base_id].rel_bind(bp) } ;

                    unsafe {
                          - (&(self.only_pos_waveform_from_binds(&binds, base_id, bp, DATA)
                                   .produce_noise(DATA, noise.background))
                          * &d_ad_stat_d_noise) * d_ad_like_d_ad_stat
                          * prop_bp * (-RT*prop_bp.ln()-CONVERSION_MARGIN).powi(2)
                          / (-RT * (*DGIBBS_CUTOFF-CONVERSION_MARGIN)) 
                    }
                }
            }).collect();

                
            d_ad_like_d_grad_form

        }
   
        
    }

    impl fmt::Display for Motif<'_> { 
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            const DIGITS: usize = 5;
            
            //Writing the peak height (unitless) and the peak width (as defined by # base pairs from one end to another)
            write!(f, "Peak height: {:.DIGITS$}. Total peak width (bp): {}\n", self.peak_height, self.kernel.len());
           
            //I want people to be free to define their own bases, if they like
            //But they will have to change it in the code: it's not my problem
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


    //We made this private for a reason
    //This should go without saying, but you REALLY don't want to touch this directly.
    struct Hmc_Motif<'a> {
    
        moveable_height: f64,
        moveable_motif: Vec<GBase>,
        positions: Vec<usize>,
        bp_inds: Vec<usize>,
        pos_height: bool,
        best_motif: Vec<usize>,
        kernel: Kernel,
        poss: bool,
        seq: &'a Sequence,

    }

    impl<'a> Hmc_Motif<'a> {

        pub fn new(mot: &'a Motif) -> Hmc_Motif<'a> {

            let best_motif = mot.best_motif();
            let pos_height = (mot.peak_height() > 0.0);
            let kernel = Kernel::new(mot.peak_height(), mot.peak_width());
            let poss = mot.poss();
            let seq = mot.seq();

            let pre_height: f64 = (mot.peak_height().abs()-MIN_HEIGHT)/(MAX_HEIGHT-MIN_HEIGHT);
            let moveable_height: f64 = (pre_height/(1.0-pre_height)).ln();


            let moveable_motif = mot.pwm().iter().map(|a| a.to_gbase()).collect::<Vec<_>>();

            let positions: Vec<usize> = (0..mot.len()).flat_map(|n| std::iter::repeat(n).take(BASE_L-1)).collect();

            let base_vec: Vec<usize> = (0..BASE_L).collect();

            let mut fake_rem: Vec<usize> = (0..(BASE_L-1)).collect();

            let bp_inds: Vec<usize> = best_motif.iter().flat_map(|n| { fake_rem = base_vec.clone();
                                                                     fake_rem.remove(*n);
                                                                     fake_rem.clone()} )
                                                       .collect();

            let bind_score = mot.return_bind_score();

            Hmc_Motif {

                moveable_height: moveable_height,
                moveable_motif: moveable_motif,
                positions: positions,
                bp_inds: bp_inds,
                pos_height: pos_height, 
                best_motif: best_motif,
                kernel: kernel,
                poss: poss,
                seq: seq,
            }
        }

        /*
        pub fn back_to_motif(&self) -> Motif {

            let mut height: f64 = 1.0/(1.0+(-self.moveable_height).exp());
            height = (MAX_HEIGHT-MIN_HEIGHT)*height+MIN_HEIGHT;
            if !self.pos_height() {
                height *= (-1.0);
            }
            

            Motif::raw_pwm(pwm: Vec<Base>, peak_height: f64, peak_width: f64, seq: &'a Sequence)


        }
      */

    }




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
      
        println!("Conversion dists: {:?}, {:?}, {}", b.show(),  b.to_gbase().to_base().show(), b.dist(Some(&b.to_gbase().to_base())));
        assert!(b == b.to_gbase().to_base());


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

        let motif: Motif = unsafe{Motif::from_clean_motif(sequence.return_bases(0,0,20), 20., &sequence)};

        let motif2: Motif = unsafe{Motif::from_clean_motif(sequence.return_bases(0,2,20), 20., &sequence)};

        let start = Instant::now();

        let waveform = motif.generate_waveform(&wave);
        let duration = start.elapsed();
        
        let waveform2 = &waveform + &(motif2.generate_waveform(&wave));

        let corrs: Vec<f64> = vec![0.9, -0.1];
        let background = Background::new(0.25, 2.64, &corrs);
        let noise: Noise = waveform.produce_noise(&waveform2, &background);

        let grad = motif.single_motif_grad(&waveform2, &noise);




        let waveform_raw = waveform.raw_wave();

        let binds = motif.return_bind_score();

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

        assert!(random_motif.poss() && sequence.kmer_in_seq(&random_motif.best_motif()));

        assert!(random_motif.raw_kern().len() == 121);

        assert!((random_motif.peak_height.abs() >= MIN_HEIGHT) && (random_motif.peak_height.abs() <= MAX_HEIGHT));

        assert!(ptr::eq(&sequence, random_motif.seq()));

        let matrix = motif.pwm();
        let nbases = matrix.len();

        for base in matrix {
            println!("{:?}", base.show());
        }

        println!("{:?}", motif.best_motif());

        let matrix = motif.rev_complement();

        for base in &matrix {
            println!("{:?}", base.show());
        }
        
        //assert!(((motif.pwm_prior()/gamma::ln_gamma(BASE_L as f64))+(motif.len() as f64)).abs() < 1e-6);

        assert!((motif.pwm_prior()+(motif.seq().number_unique_kmers(motif.len()) as f64).ln()
                 -(((BASE_L-1)*motif.len()) as f64)*((*PROP_UPPER_CUTOFF-*PROP_CUTOFF).ln())).abs() < 1e-6);

        let un_mot: Motif = Motif::from_motif(vec![1usize;20], 10., &sequence);

        assert!(un_mot.pwm_prior() < 0.0 && un_mot.pwm_prior().is_infinite());

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
        let little_motif: Motif = unsafe{Motif::raw_pwm(mat, 10.0, 1.0, &wave_seq)};

        print!("{}", little_motif);
        println!("{:?}",little_motif.generate_waveform(&wave_wave).raw_wave());

        let small_block: Vec<u8> = vec![44, 24, 148, 240, 84, 64, 200, 80, 68, 92, 196, 144]; 
        let small_inds: Vec<usize> = vec![0, 6]; 
        let small_lens: Vec<usize> = vec![24, 24];
        let small: Sequence = Sequence::new_manual(small_block, small_lens);
        let small_wave: Waveform = Waveform::new(vec![0.1, 0.6, 0.9, 0.6, 0.1, -0.2, -0.4, -0.6, -0.6, -0.4], &small, 5);

        let mat: Vec<Base> = (0..15).map(|_| Base::new(theory_base.clone())).collect::<Vec<_>>();
        let wave_motif: Motif = unsafe{Motif::raw_pwm(mat, 10.0, 1.0, &small)};

        let rev_comp: Vec<bool> = (0..48).map(|_| rng.bool()).collect();

        let checked = wave_motif.base_check(&rev_comp, 0, 4);

        let forward: Vec<bool> = vec![true, false, false, true, true, false, false, false, true, true, false, false, true, false, false, false, true, true, true, false, true, false, true, false, true, true, false, false, true, false, true, false, true, false, false, false, true, false, true, false, true, true, false, false, false, false, false, false];

        let reverse: Vec<bool> = vec![true, false, false, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, true, false, false, false, false, false, true, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false, false];

        let correct: Vec<bool> = rev_comp.iter().enumerate().map(|(a, &b)| if b {reverse[a]} else {forward[a]}).collect();

        println!("correct: {:?}", correct);
        println!("checked: {:?}", checked);

        println!("small bl: {:?} {:?} {:?} {:?}", small.seq_blocks(), small.block_lens(), small.block_u8_starts(), small.return_bases(0, 0, 24));
        println!("blocks in seq: {:?}", wave_motif.seq().seq_blocks());



        //TESTING base_check()
        for i in 0..2 {
            for j in 0..24 {

                let ind = if rev_comp[24*i+j] { j+wave_motif.len()-1-4 } else {j+4}; 
                if ind < 24 {
                    let bp = small.return_bases(i, ind, 1)[0];
                    let bp2 = wave_motif.seq().return_bases(i, ind, 1)[0];
                    let matcher = if rev_comp[24*i+j] { bp == 3 } else { bp == 0};
                    println!("start loc: {}, bp: {}, bp2: {}, ind: {}, rev: {}, matcher: {}, checked: {}, correct: {}", 24*i+j, bp, bp2, ind, rev_comp[24*i+j], matcher, checked[24*i+j], correct[24*i+j]);
                    assert!(checked[24*i+j] == matcher);
                }
            }
        }


        let start = Instant::now();

        let binds = motif.return_bind_score();

        let duration = start.elapsed();
        println!("Time elapsed in bind_score() is: {:?}", duration);

        let start = Instant::now();
        let checked = motif.base_check(&binds.1, 2, 4);
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

        let checked = motif.base_check(&binds.1, 3, 6);

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
