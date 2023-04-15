pub mod bases {
    use rand::Rng;
    use rand::seq::SliceRandom;
    use rand::distributions::{Distribution, Uniform};
    use statrs::distribution::{Continuous, ContinuousCDF, LogNormal, Normal};
    use statrs::statistics::{Min, Max, Distribution as OtherDistribution};
    use crate::waveform::wave::{Kernel, Waveform};
    use crate::sequence::seq::Sequence;
    use statrs::function::gamma;
    use statrs::{consts, Result, StatsError};
    use std::f64;
    use std::fmt;
    use std::collections::VecDeque;

    const BPS: [&str; 4] = ["A", "C", "G", "T"];
    const BASE_L: usize = BPS.len();
    const RT: f64 =  8.31446261815324*298./4184.; //in kcal/(K*mol)

    const CLOSE: f64 = 1e-5;

    const MIN_BASE: usize = 8;
    const MAX_BASE: usize = 20;

    const MIN_HEIGHT: f64 = 3.;
    const MAX_HEIGHT: f64 = 30.;
    const LOG_HEIGHT_MEAN: f64 = 2.30258509299; //This is ~ln(10). Can't use ln in a constant
    const LOG_HEIGHT_SD: f64 = 0.25;

    const PROB_POS_PEAK: f64 = 0.9;
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
            self.dist(other) < CLOSE    
        }
    }


    fn clean_props(vals: [ f64; BASE_L]) -> [f64; BASE_L] {

        let mut props = vals;

        let norm: f64 = props.iter().sum();

        for i in 0..props.len() {
            props[i] /= norm
        }

        props

    }

    impl Base {

        pub fn new(props: [ f64; 4]) -> Base {

            let eps = 1e-12;
 
            //let norm: f64 = props.iter().sum();

            let mut any_neg: bool = false;

            for i in 0..props.len() {
                any_neg |= props[i] < 0.0 ;
            }

            if any_neg{ //|| ((norm - 1.0).abs() > 1e-12) {
               panic!("All is positive: {}. Norm is .", !any_neg)
            }

            Base { props }
        }

        pub fn rand_new() -> Base {

            let mut rng = rand::thread_rng();
            let die = Uniform::from(0.0..1.0);

            let mut samps : [ f64; BASE_L+1] = [0.0, 0.0, 0.0, 0.0,1.0];

            for i in 1..(samps.len()-1) {
                samps[i] = die.sample(&mut rng);
            }

            samps.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let mut res : [f64; BASE_L] = [0.0, 0.0, 0.0, 0.0];

            for i in 0..res.len() {
                res[i] = samps[i+1]-samps[i];
            }


            Base::new(res)
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
                    dgs[ind] = -RT*(self.props[i]/max).ln();
                    ind += 1;
                }
            }

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

        pub fn dist(&self, other: &Base) -> f64 {

            let diffy: Vec<f64> = self.show().iter().zip(other.show()).map(|(&a, b)| a-b as f64).collect();
            diffy.iter().map(|&a| a.abs() as f64).sum::<f64>()

        }

        pub fn rel_bind(&self, bp: usize) -> f64 {
            self.props[bp]/Self::max(&self.props) //This is the correct answer
            //self.props[bp] //This is incorrect but faster
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


            let mut props = [0.0_f64 ; BASE_L];


            for i in 0..props.len() {

                if i < self.best {
                    props[i] = (-self.dgs[i]/RT).exp();
                } else if i == self.best {
                    props[i] = 1.0_f64;
                } else {
                    props[i] = (-self.dgs[i-1]/RT).exp();
                }
            }

            let norm: f64 = props.iter().sum();

            for p in 0..props.len() {
                props[p] /= norm;
            }


            let based: Base = Base {
                props: props,
            };

            based

        }
    }

    //BEGIN MOTIF
    pub struct Motif {
    
        peak_height: f64,
        kernel: Kernel,
        pwm: Vec<Base>,
        poss: bool,

    }

    impl Motif {

        //GENERATORS
        //TODO: make sure that this is used iff there's a guarentee that a motif is allowed
        pub fn new(pwm: Vec<Base>, peak_height: f64, peak_width: f64) -> Motif {
            let kernel = Kernel::new(peak_width, peak_height);

            Motif {
                peak_height: peak_height,
                kernel: kernel,
                pwm: pwm,
                poss: true,
            }
        }

 
        //TODO: make sure that this is used iff there's a guarentee that a motif is allowed
        pub fn from_motif(best_bases: Vec<usize>, peak_width: f64, seq: &Sequence) -> Motif {

            let pwm: Vec<Base> = best_bases.iter().map(|a| Base::from_bp(*a)).collect();

            let height_dist: TruncatedLogNormal = TruncatedLogNormal::new(LOG_HEIGHT_MEAN, LOG_HEIGHT_SD, MIN_HEIGHT, MAX_HEIGHT).unwrap();

            let mut rng = rand::thread_rng();

            let sign: f64 = rng.gen();
            let sign: f64 = if sign < PROB_POS_PEAK {1.0} else {-1.0};

            let peak_height: f64 = sign*height_dist.sample(&mut rng);

            let kernel = Kernel::new(peak_width, peak_height);

            let poss = seq.kmer_in_seq(best_bases);

            Motif {
                peak_height: peak_height,
                kernel: kernel,
                pwm: pwm,
                poss: poss,
            }


        }

        pub fn from_clean_motif(best_bases: Vec<usize>, peak_width: f64, seq: &Sequence) -> Motif {

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
            }


        }
        //TODO: make sure that this takes a sequence reference variable and that we pull our random motif from there
        pub fn rand_mot(peak_width: f64, seq: &Sequence) -> Motif {

            let mut rng = rand::thread_rng();

            let num_bases = rng.gen_range(MIN_BASE..(MAX_BASE+1));

            let mot = (seq.generate_kmers(num_bases).choose(&mut rng)).unwrap().clone();

            Self::from_clean_motif(mot, peak_width, &seq)


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
        pub fn peak_width(&self) -> f64 {
            self.kernel.get_sd()
        }

        pub fn len(&self) -> usize {
            self.pwm.len()
        }

        //PRIORS

        pub fn pwm_prior(&self) -> f64 {

            //TODO: Insert check for possibility of PWM if I can't guarentee it
            //ALSO TODO: Try to guarentee possibility of PWM

            if self.poss {-(self.pwm.len() as f64)*gamma::ln_gamma(BASE_L as f64)} else {-f64::INFINITY}
        }

        pub fn height_prior(&self) -> f64 {

            TruncatedLogNormal::new(LOG_HEIGHT_MEAN, LOG_HEIGHT_SD, MIN_HEIGHT, MAX_HEIGHT).unwrap().ln_pdf(self.peak_height)


        }


        //BINDING FUNCTIONS
        pub fn prop_binding(&self, kmer: &[usize]) -> (f64, bool) { 

            //let bind_forward: f64 = self.pwm.iter().zip(kmer).map(|(a, &b)| a.rel_bind(b).ln()).sum::<f64>().exp();
            //let bind_reverse: f64 = self.rev_complement().iter().zip(kmer).map(|(a, &b)| a.rel_bind(b).ln()).sum::<f64>().exp();
            
            let bind_forward: f64 = self.pwm.iter().zip(kmer.into_iter()).map(|(a, &b)| a.rel_bind(b)).product::<f64>();
            let bind_reverse: f64 = self.rev_complement().iter().zip(kmer.into_iter()).map(|(a, &b)| a.rel_bind(b)).product::<f64>();


            let reverse: bool = (bind_reverse > bind_forward);

            let bind: f64 = if reverse {bind_reverse} else {bind_forward};

            return (bind, reverse)

        }
/*
        pub fn return_bind_scorea(&self, seq: &Sequence) -> (Vec<f64>, Vec<bool>) {

            let coded_sequence = seq.seq_blocks();
            let block_lens = seq.block_lens(); //bp space
            let block_starts = seq.block_inds(); //stored index space


            let mut bind_scores: Vec<f64> = vec![0.0; 4*coded_sequence.len()];
            let mut rev_comp: Vec<bool> = vec![false; 4*coded_sequence.len()];

            let seq_frame = 1+(self.len()/4);

            let mut uncoded_seq: Vec<usize> = vec![0; *(block_lens.iter().max().unwrap())];

            let mut ind = 0;

            let mut store = Sequence::code_to_bases(coded_sequence[0]);

            for i in 0..(block_starts.len()) {

                let actual_frame = &mut uncoded_seq[0..block_lens[i]];

                for jd in 0..(block_lens[i]/4) {

                    store = Sequence::code_to_bases(coded_sequence[block_starts[i]+jd]);
                    for k in 0..4 {
                        actual_frame[4*jd+k] = store[k];
                    }

                }

                for j in 0..((block_lens[i])-self.len()) {

                    ind = 4*block_starts[i]+j; 
                    
                    (bind_scores[ind], rev_comp[ind]) = self.prop_binding(&actual_frame[j..(j+self.len())]);

                }

            }


            (bind_scores, rev_comp)



        }
  */      
        pub fn return_bind_score(&self, seq: &Sequence) -> (Vec<f64>, Vec<bool>) {

            let mut coded_sequence = seq.seq_blocks();
            let block_lens = seq.block_lens(); //bp space
            let block_starts = seq.block_inds(); //stored index space


            let mut bind_scores: Vec<f64> = vec![0.0; 4*coded_sequence.len()];
            let mut rev_comp: Vec<bool> = vec![false; 4*coded_sequence.len()];

            let seq_frame = 1+(self.len()/4);
            let off = self.len() % 4;

            //let mut uncoded_seq: Vec<usize> = vec![0; *(block_lens.iter().max().unwrap())];


            //We set this to 1200 for now
            //Theoretically, if we're doing full IPOD, we may need to bump up to like 16kbp
            //It actually turns out that using this as an array is slightly slower
            //let mut uncoded_seq: [usize; 1200] = [0;1200];

            let mut current_data = &mut coded_sequence[0];

            let mut ind = 0;

            let mut base_push = 0;

            let mut store: [usize; MAX_BASE] = [0; MAX_BASE];

            let continue_from_coded = if (self.len() % 4) == 0 {0} else {1};

            for i in 0..(block_starts.len()) {


                //This does a few things:
                // 1) The store array is initialized up to the base length
                // 2) The base_push tells us how far we're into the next coded u8 
                //      (0 means untouched, 1 means we've extracted 1 base, etc)
                // 3) This mutates our local copy of the coded sequence so we can remember
                //    where we are in extraction and pull the next base
                for j in 0..(1+(self.len()/4)) {
                    current_data = &mut coded_sequence[block_starts[i]+j];
                    while (base_push < 4) && (4*j+base_push < self.len()) {
                        store[4*j+base_push] = (0b00000011 & *current_data) as usize;
                        *current_data = *current_data >> 2;
                        base_push += 1;
                    }
                    base_push %= 4;
                }
                

                ind = 4*block_starts[i];

                for j in (self.len()/4)..((block_lens[i]/4)) {
                    current_data = &mut coded_sequence[block_starts[i]+j];
                    while (base_push < 4) && (4*j+base_push < block_lens[i]) {
                        (bind_scores[ind], rev_comp[ind]) = self.prop_binding(&store[0..self.len()]);
                        store.rotate_left(1);
                        store[self.len()-1] = (0b00000011 & *current_data) as usize;
                        *current_data = *current_data >> 2;
                        ind+=1;
                        //println!("In method {} {} {} {} {} ", 4*j+base_push , block_lens[i]+off-self.len(), ind, base_push, 4*(block_starts[i]+j)+base_push);
                        base_push += 1;
                    }

                    base_push = 0;
                }


            }


            (bind_scores, rev_comp)



        }




    }

    impl fmt::Display for Motif { 
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



}

#[cfg(test)]
mod tester{

    use std::time::{Duration, Instant};
    use crate::base::bases::Base;
    use crate::base::bases::GBase;
    use crate::base::bases::TruncatedLogNormal;
    use crate::base::bases::Motif;
    use crate::sequence::seq::Sequence;
    use statrs::distribution::{Continuous, ContinuousCDF, LogNormal, Normal};
    use statrs::statistics::{Min, Max};
    use statrs::function::gamma;
    use rand::Rng;
    use rand::distributions::{Distribution, Uniform};
    const MIN_HEIGHT: f64 = 3.;
    const MAX_HEIGHT: f64 = 30.;
    const LOG_HEIGHT_MEAN: f64 = 2.30258509299; //This is ~ln(10). Can't use ln in a constant
    const LOG_HEIGHT_SD: f64 = 0.25;

    const BASE_L: usize = 4;

    
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

       assert!(b == b.to_gbase().to_base());


        let td: Base = Base::new([0.1, 0.2, 0.4, 0.3]);

        assert!((td.rel_bind(1)-0.5_f64).abs() < 1e-6);
        assert!((td.rel_bind(2)-1_f64).abs() < 1e-6);

        let tg: GBase = GBase::new([0.82094531732, 0.41047265866, 0.17036154577], 2);

        assert!(tg.to_base() == td);

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


        let mut rng = fastrand::Rng::new();

        let block_n: usize = 20;
        let u8_per_block: usize = 5000;
        let bp_per_block: usize = u8_per_block*4;
        let bp: usize = block_n*bp_per_block;
        let u8_count: usize = u8_per_block*block_n;

        let start_gen = Instant::now();
        //let blocks: Vec<u8> = (0..u8_count).map(|_| rng.u8(..)).collect();
        let preblocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.u8(..)).collect();
        let blocks: Vec<u8> = preblocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>(); 
        let block_inds: Vec<usize> = (0..block_n).map(|a| a*u8_per_block).collect();
        let start_bases: Vec<usize> = (0..block_n).map(|a| a*bp_per_block).collect();
        let block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
        let sequence: Sequence = Sequence::new_manual(blocks, block_inds, block_lens);
        let duration = start_gen.elapsed();
        println!("Done gen {} bp {:?}", bp, duration);


        println!("{} gamma", gamma::gamma(4.));
        println!("{} gamma", gamma::ln_gamma(4.));

        let motif: Motif = Motif::from_clean_motif(sequence.return_bases(0,0,20), 20., &sequence);

        println!("{}", motif);

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

        //let un_mot: Motif = Motif::from_motif(vec![1usize;20], 20., &sequence);

        //assert!(un_mot.pwm_prior() < 0.0 && un_mot.pwm_prior().is_infinite());

        let start = Instant::now();


        let binds = motif.return_bind_score(&sequence);

        let duration = start.elapsed();
        println!("Time elapsed in bind_score() is: {:?}", duration);

        println!("Length bind is: {}", binds.0.len());

        for i in 0..block_n {
            for j in 0..(bp_per_block-motif.len()) {

                let test_against = motif.prop_binding(&sequence.return_bases(i, j, motif.len()));
                assert!((binds.0[i*bp_per_block+j]-test_against.0).abs() < 1e-6);
                assert!(binds.1[i*bp_per_block+j] == test_against.1);


            }
        }

    }
}
