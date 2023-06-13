use rand::Rng;
use rand::seq::SliceRandom;
use rand::distributions::{Distribution, Uniform, WeightedIndex};
use statrs::distribution::{Continuous, ContinuousCDF, LogNormal, Normal, Dirichlet, Exp};
use statrs::function::{gamma::*, erf::*, beta::*};
use statrs::statistics::{Min, Max, Distribution as OtherDistribution};
use num_traits::Float;
use log::warn;
use serde::{Serialize, Deserialize};

//I apologize for this struct in advance, in particular, the FastT implementation.
//It is an unholy combination of R's background C implementation of pt, modified 
//and pared down for my use, and the statrs implementation of things like the gamma 
//function. You'll notice that my code is optimized for getting ln densities. This
//is intentional: we can end up with very small densities and precision is more
//critical for our purposes at the ends of our distribution


const LN_GAMMA_HALF: f64 = 0.5723649429247000870717136756;
const LN_GAMMA_3_HALFS: f64 = -0.120782237635245222;
const M_SQRT_PI: f64 = 1.772453850905516027298167483341;
const NUM_BGRAT_TERMS: usize = 30;
const M_LN_2: f64 = 0.693147180559945309417232121458176;

const GIVE_UP_AND_USE_NORMAL: f64 = 20.0;

pub enum BackgroundDist {
    Normal(Normal),
    FastT(FastT),
}

impl BackgroundDist {

    pub fn new(scale: f64, freedom: f64) -> BackgroundDist {

        if scale < 0.0 {
            panic!("Only give me positive scale parameters!");
        }
        if freedom <= 2.0 { //We need at least a variance to be well defined. There's no working with this, otherwise
            panic!("This background distribution is too ill behaved to do inference on it!");
        } else if freedom >= GIVE_UP_AND_USE_NORMAL {
            BackgroundDist::Normal(Normal::new(0.0, scale*(freedom/(freedom-2.)).sqrt()).expect("This shouldn't be possible."))
        } else {
            BackgroundDist::FastT(FastT{ scale: scale, freedom: freedom})
        }
    }
    pub fn ln_cd_and_sf(&self,x: f64) -> (f64, f64) {
        match self {
            BackgroundDist::Normal(norm) => (norm.cdf(x).ln(), norm.sf(x).ln()),
            BackgroundDist::FastT(fast) => fast.ln_cd_and_sf(x)
        }
    }
}

impl ContinuousCDF<f64, f64> for BackgroundDist {

    fn cdf(&self, x: f64) -> f64 {
        match self {
            BackgroundDist::Normal(norm) => norm.cdf(x),
            BackgroundDist::FastT(fast) => fast.cdf(x)
        }
    }

    fn sf(&self, x: f64) -> f64 {
        match self {
            BackgroundDist::Normal(norm) => norm.sf(x),
            BackgroundDist::FastT(fast) => fast.sf(x)
        }
    }
}

impl ContinuousLnCDF<f64, f64> for BackgroundDist {


    fn ln_cdf(&self, x: f64) -> f64{ 
        match self {
            BackgroundDist::Normal(norm) => norm.cdf(x).ln(),
            BackgroundDist::FastT(fast) => fast.ln_cdf(x)
        }
    }

    fn ln_sf(&self, x: f64) -> f64 {
        match self {
            BackgroundDist::Normal(norm) => norm.sf(x).ln(),
            BackgroundDist::FastT(fast) => fast.ln_sf(x)
        }
    }
}

impl Continuous<f64, f64> for BackgroundDist {

    fn pdf(&self, x: f64) -> f64{
        match self {
            BackgroundDist::Normal(norm) => norm.pdf(x),
            BackgroundDist::FastT(fast) => fast.pdf(x)
        }
    }
    fn ln_pdf(&self, x: f64) -> f64{
        match self {
            BackgroundDist::Normal(norm) => norm.ln_pdf(x),
            BackgroundDist::FastT(fast) => fast.ln_pdf(x)
        }
    }
    
}
impl Min<f64> for BackgroundDist {
    fn min(&self) -> f64 {
        return -f64::INFINITY
    }
}
impl Max<f64> for BackgroundDist {
    fn max(&self) -> f64 {
        return f64::INFINITY
    }
}
#[derive(Serialize, Deserialize)]
struct FastT {
    scale: f64,
    freedom: f64,
}

impl ContinuousCDF<f64, f64> for FastT {


    fn cdf(&self, x: f64) -> f64{
        self.ln_cd_and_sf(x).0.exp()
    }

    fn sf(&self, x: f64) -> f64 {
        self.ln_cd_and_sf(x).1.exp()
    }
}

impl ContinuousLnCDF<f64, f64> for FastT {


    fn ln_cdf(&self, x: f64) -> f64{
        self.ln_cd_and_sf(x).0
    }

    fn ln_sf(&self, x: f64) -> f64 {
        self.ln_cd_and_sf(x).1
    }
}
impl Continuous<f64, f64> for FastT {

    fn pdf(&self, x: f64) -> f64{
        if x.is_infinite() {
            0.0
        } else { //We don't implement the code to give up and use a normal distribution because we only access FastT when we know we aren't giving up 
            let d = x / self.scale;
            (ln_gamma((self.freedom + 1.0) / 2.0) - ln_gamma(self.freedom / 2.0))
                .exp()
                * (1.0 + d * d / self.freedom).powf(-0.5 * (self.freedom + 1.0))
                / (self.freedom * std::f64::consts::PI).sqrt()
                / self.scale
        }

    }
    fn ln_pdf(&self, x: f64) -> f64 {
        if x.is_infinite() {
            f64::NEG_INFINITY
        } else {
            let d = x / self.scale;
            ln_gamma((self.freedom + 1.0) / 2.0)
                - 0.5 * ((self.freedom + 1.0) * (1.0 + d * d / self.freedom).ln())
                - ln_gamma(self.freedom / 2.0)
                - 0.5 * (self.freedom * std::f64::consts::PI).ln()
                - self.scale.ln()
        }
    }

}

impl Min<f64> for FastT {
    fn min(&self) -> f64 {
        return -f64::INFINITY
    }
}
impl Max<f64> for FastT {
    fn max(&self) -> f64 {
        return f64::INFINITY
    }
}

trait ContinuousLnCDF<K: Float, T: Float>: Min<K> + Max<K> {

    fn ln_cdf(&self, x: K) -> T;
    fn ln_sf(&self, x: K) -> T;


}

impl FastT {

    pub fn ln_cd_and_sf(&self,x: f64) -> (f64, f64) {

        let q2 = (x/self.scale).powi(2);

        let a = self.freedom/2.;

        let major_branch = q2 >= self.freedom;

        //inp is technically always mathematically the same thing, but we calculate it slightly 
        //differently because theoretically the same and numerically the same aren't the same thing
        let inp = if major_branch {1./(1.+q2/self.freedom)} else { self.freedom/(self.freedom+q2)};
                    //This will always be <= 0.5                    This will always be >= 0.5
                    //  (no swap)                                   (swap)
                    //So in both cases, once incorporating the swap, a = self.freedom/2., b = 0.5
                    //The only difference in the swap is that the latter requires 1.-inp
                    //And because b = 0.5 < 1, we always take the bpser route in the code
                    //Wow, the C code is written arcanely


        let prew = bpser(a, inp, 1e-14);

        println!("x: {}, q2: {}, inp: {}, b: {}, Ix: {}, Ixf: {},prew: {}, prew1: {}", x, q2,inp, a,beta_inc(a, 0.5, inp)/beta(a,0.5), beta_inc(a, 0.5, inp)/beta(a,0.5),prew, prew);
        if (x > 0.0) {
            ((-0.5*(prew.exp())).ln_1p(), prew-M_LN_2)
        } else {
            (prew-M_LN_2, (-0.5*prew.exp()).ln_1p())
        }
       
    }

}


//This is an attenuated version of the function bpser
//from R's C implementation of its statistics functions
//It is attenuated because we need it for very particular
//use cases

//In particular, when we use it, b = 0.5
//Also, we are using the ln_gamma function from statrs instead of 
//the official R implementation: it's accurate up to about 6 bits at
//the very end of our floats and takes ~60ns regardless of input

fn bpser(a: f64, x: f64, eps: f64) -> f64 {



    let mut ans: f64 = a * x.ln()-ln_beta_half_times_a(a);


    /* ----------------------------------------------------------------------- */
    /*		       COMPUTE THE SERIES */
    /* ----------------------------------------------------------------------- */
    let tol: f64 = eps / a;
    let mut n: f64 = 0.;
    let mut sum: f64 = 0_f64;
    let mut c: f64 = 1.;
    let mut w: f64 = f64::INFINITY;
    while (n < 1e7 && w.abs() > tol) { // sum is alternating as long as n < b (<==> 1 - b/n < 0)
        n += 1.;
        c *= (0.5 - (0.5 / n)+0.5 ) * x;
        w = c / (a + n);
        sum += w;
    }     
    
    if (a*sum > -1.) {ans += (a * sum).ln_1p();}
    else {
        ans = -f64::INFINITY;
    }
    return ans;
} /* bpser */


fn ln_beta(a: f64, b:f64) -> f64 {
    ln_gamma(a+1.)+ln_gamma(b+1.)-ln_gamma(a+b+1.)
}

fn ln_beta_half_times_a(a: f64) -> f64 {
    LN_GAMMA_HALF+ln_gamma(a+1.)-ln_gamma(a+0.5)
}

fn grat_r(x: f64, log_r: f64) -> f64{
    let sx = x.sqrt();
    if x < 0.25 {
        (1.0-erf(sx))*(-log_r).exp()
    } else {
        erfc(sx) * M_SQRT_PI * x.exp()/sx
    }

}

fn logspace_add(lnx: f64, lny: f64) -> f64 {
    lnx.max(lny)+(-(lnx-lny).abs()).exp().ln_1p()
}

#[derive(Serialize, Deserialize)]
pub enum BackgroundDistDef {
    Normal(NormalDef),
    FastT(FastT),
}

impl From<BackgroundDistDef> for BackgroundDist {
    fn from(def: BackgroundDistDef) -> BackgroundDist {
        match def {
            BackgroundDistDef::Normal(norm) => BackgroundDist::Normal(Normal::new(norm.mean, norm.std_dev).unwrap()),
            BackgroundDistDef::FastT(fast) => BackgroundDist::FastT(fast),
        }
    }
}


#[derive(Serialize, Deserialize)]
struct NormalDef {
    mean: f64,
    std_dev: f64,
}

impl From<NormalDef> for Normal {
    fn from(def: NormalDef) -> Normal {
        Normal::new(def.mean, def.std_dev).unwrap()
    }
}
/*
trait Getter {
    fn get_mean(&self) -> f64;
    fn get_sd(&self) -> f64;
}

impl Getter for Normal {
    fn get_mean(&self) -> f64 { self.mean().unwrap() }
    fn get_sd(&self) -> f64 { self.std_dev().unwrap() }
}*/



#[cfg(test)]
mod tests{

    use crate::modified_t::BackgroundDist;
    #[test]
    fn attempt() {

        let dis = BackgroundDist::new(0.25, 2.84);

        let data = [2.2291285570e+0,  4.1510439540e-1, -5.8846691870e-1, -1.4300281780e+0, -1.0174699610e+0,  9.6801737090e-1,  6.1547874390e-1,  1.0320017990e+0, 9.8058816140e-1, 5.7552069380e-1];

        let cd = [-0.001841144352630861, -0.1056418605108091, -2.948328285560595, -5.091991745450134, -4.210455061116229, -0.01694448425305765, -0.04893062330685381, -0.01442399294418845, -0.0164061855049808, -0.05643039312469962];

        let sf = [-6.298288400836528, -2.300056544954666, -0.05385159158458796, -0.006164729845405886, -0.01495082250690833, -4.086273189920248, -3.041717390080652, -4.246065609924472, -4.118288727978808, -2.902829897631842];

        for (i, dat) in data.iter().enumerate() {
            let (c, s) = dis.ln_cd_and_sf(*dat);
            println!("cd {}, sf {}, prop_cd {}, prop_sf {}, diff cd {}, diff sf {}",c, s, cd[i], sf[i], c-cd[i], s-sf[i]);
            assert!(((c-cd[i]).abs() < 1e-10) && ((s-sf[i]).abs() < 1e-10));
        }


    }
}
