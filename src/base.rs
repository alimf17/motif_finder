pub mod bases {
    use rand::Rng;
    use rand::distributions::{Distribution, Uniform};
    use statrs::distribution::{Continuous, ContinuousCDF, LogNormal, Normal};
    use statrs::statistics::{Min, Max};
    use crate::waveform::wave::{Kernel, Waveform};

    use statrs::{consts, Result, StatsError};
    use std::f64;

    const BASE_L: usize = 4;
    const RT: f64 =  8.31446261815324*298./4184.; //in kcal/(K*mol)

    const CLOSE: f64 = 1e-5;

    const MIN_BASE: usize = 8;
    const MAX_BASE: usize = 20;

    const MIN_HEIGHT: f64 = 3.;
    const MAX_HEIGHT: f64 = 30.;
    const LOG_HEIGHT_MEAN: f64 = 10.;
    const LOG_HEIGHT_SD: f64 = 0.25;

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

            let norm: f64 = props.iter().sum();

            let mut any_neg: bool = false;

            for i in 0..props.len() {
                any_neg |= props[i] < 0.0 ;
            }

            if any_neg || ((norm - 1.0).abs() > 1e-12) {
                panic!("All is positive: {}. Norm is {}.", !any_neg, norm)
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

            return self.props[bp]/self.props[self.best_base()]

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
    /* pub struct Motif {
    
        peak_height: f64,
        kernel: Kernel,
        pwm: Vec<Base>,

    }

    impl Motif {

        pub fn new(pwm: Vec<Base>, peak_height: f64, peak_width: f64) -> Motif {
            let kernel = Kernel::new(peak_width, peak_height);

            Kernel {

                kernel: kernel,
                pwm: pwm,
            }
        }

        pub fn from_motif(best_bases: Vec<f64>, peak_width: f64) -> Motif {



        }

        pub fn rand_mot(peak_width: f64) -> Motif {

            let mut rng = rand::thread_rng();



        }


    } */

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
                Ok(TruncatedLogNormal { location, scale, min, max })
            }
        }
        
    }

    impl ::rand::distributions::Distribution<f64> for TruncatedLogNormal {
        
        fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
           
            let norm: Normal = Normal::new(self.location, self.scale).unwrap();
            let mut sample: f64 = norm.sample(rng).exp();
            let mut valid: bool = ((sample > self.max) | (sample < self.min));
            while !valid {
                sample = norm.sample(rng).exp();
                valid = ((sample > self.max) | (sample < self.min));
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

    use crate::base::bases::Base;
    use crate::base::bases::GBase;
    use crate::base::bases::TruncatedLogNormal;
    use statrs::distribution::{Continuous, ContinuousCDF, LogNormal, Normal};
    use statrs::statistics::{Min, Max};
    const MIN_HEIGHT: f64 = 3.;
    const MAX_HEIGHT: f64 = 30.;
    const LOG_HEIGHT_MEAN: f64 = 10.0;
    const LOG_HEIGHT_SD: f64 = 0.25;

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

        let val1: f64 = 3.0;
        let val2: f64 = 15.0;

        assert!(((free_dist.pdf(val1)/free_dist.pdf(val2))-(dist.pdf(val1)/dist.pdf(val2))).abs() < 1e-6);
        assert!(((free_dist.cdf(val1)/free_dist.cdf(val2))-(dist.cdf(val1)/dist.cdf(val2))).abs() < 1e-6);

        assert!(dist.ln_pdf(MAX_HEIGHT+1.0).is_infinite() && dist.ln_pdf(MAX_HEIGHT+1.0) < 0.0);

        let dist: TruncatedLogNormal = TruncatedLogNormal::new(LOG_HEIGHT_MEAN, LOG_HEIGHT_SD, -1.0, MAX_HEIGHT).unwrap();

        assert!(dist.min().abs() < 1e-6);

    }
}
