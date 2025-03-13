use crate::base::{BASE_L, SCORE_THRESH, Base, LN_2};

use rand::Rng;
use statrs::distribution::{Continuous, ContinuousCDF, Normal, Gamma};
use statrs::consts;
use statrs::statistics::{Min, Max, Distribution};
use num_traits::Float;
use serde::{Serialize, Deserialize};

use std::f64;
use std::error::Error;

use once_cell::sync::Lazy;

//use std::time::{Duration, Instant};

//I apologize for this struct in advance, in particular, the FastT implementation.
//It is an unholy combination of R's background C implementation of pt, modified 
//and pared down for my use, and the statrs implementation of things like the gamma 
//function. You'll notice that my code is optimized for getting ln densities. This
//is intentional: we can end up with very small densities and precision is more
//critical for our purposes at the ends of our distribution


const LN_GAMMA_HALF: f64 = 0.5723649429247000870717136756;
const M_LN_2: f64 = 0.693147180559945309417232121458176;

const GAMMA_DK: [f64; 11] = [
    2.48574089138753565546e-5,
    1.05142378581721974210,
    -3.45687097222016235469,
    4.51227709466894823700,
    -2.98285225323576655721,
    1.05639711577126713077,
    -1.95428773191645869583e-1,
    1.70970543404441224307e-2,
    -5.71926117404305781283e-4,
    4.63399473359905636708e-6,
    -2.71994908488607703910e-9,
];

const GAMMA_INDS: [f64; 11] = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.];

const GAMMA_R: f64 = 10.900511;


const GIVE_UP_AND_USE_NORMAL: f64 = 20.0;

const IMPL_CUT: f64 = 1.; //

static CONDITION_BASE: Lazy<f64> = Lazy::new(|| (-1.0+12.0/(3.+SCORE_THRESH.exp2())-12.0/(2.0+2.0*SCORE_THRESH.exp2())+4.0/(1.0+3.0*SCORE_THRESH.exp2())).ln());

/// This is essentially a T distribution with a failure mode to a normal 
/// approximation if it has more than 20 degrees of freedom. However, it
/// will fail if degrees of freedom are less than 2. 
#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum BackgroundDist {

    #[serde(with = "NormalDef")]
    Normal(Normal),
    FastT(FastT),
}

impl BackgroundDist {

    pub fn new(scale: f64, freedom: f64) -> Result<BackgroundDist, Box<dyn Error>> {

        if freedom >= GIVE_UP_AND_USE_NORMAL {
            let ratio = if freedom.is_infinite() {1.0} else { freedom/(freedom-2.) };
            Ok(BackgroundDist::Normal(Normal::new(0.0, scale*ratio).sqrt())?))
        } else {
            Ok(BackgroundDist::FastT(FastT::new(scale,freedom)?))
        }
    }
    pub fn ln_cd_and_sf(&self,x: f64) -> (f64, f64) {
        match self {
            BackgroundDist::Normal(norm) => (norm.cdf(x).ln(), norm.sf(x).ln()),
            BackgroundDist::FastT(fast) => fast.ln_cd_and_sf(x)
        }
    }
    pub fn cd_and_sf(&self,x: f64) -> (f64, f64) {
        match self {
            BackgroundDist::Normal(norm) => (norm.cdf(x), norm.sf(x)),
            BackgroundDist::FastT(fast) => fast.cd_and_sf(x)
        }
    }

    pub fn get_spread_par(&self) -> f64 {
        match self {
            //UGH. I wish statrs let me access the standard deviation directly, rather than requiring me to square root their squared sd
            //SAFETY: the normal distribution must always have a valid f64 as its standard deviation
            BackgroundDist::Normal(norm) => unsafe{ norm.std_dev().unwrap_unchecked()},
            BackgroundDist::FastT(fast) => fast.scale,
        }
    }

    pub fn get_sd_df(&self) -> (f64, f64) {
        match self {
            BackgroundDist::Normal(norm) => (unsafe{ norm.std_dev().unwrap_unchecked()}, f64::INFINITY),
            BackgroundDist::FastT(fast) => (fast.scale, fast.freedom)
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

impl ::rand::distributions::Distribution<f64> for BackgroundDist {
    fn sample<R: ::rand::Rng + ?Sized>(&self, r: &mut R) -> f64 {

        match self {
            BackgroundDist::Normal(norm) => norm.sample(r),
            BackgroundDist::FastT(fast) => fast.sample(r),
        }
    }
}


#[derive(Copy, Clone, PartialEq, Eq, Debug, Hash)]
#[non_exhaustive]
pub enum FastTError{
    ScaleInvalid,
    FreedomInvalid
}

impl std::fmt::Display for FastTError {
    #[cfg_attr(coverage_nightly, coverage(off))]
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            FastTError::ScaleInvalid => write!(f, "Scale is NaN or non-positive"),
            FastTError::FreedomInvalid => {
                write!(f, "Degrees of freedom are NaN or less than 2.0")
            }
        }
    }
}

impl std::error::Error for FastTError {}

/// This is a reimplementation of the T distribution from the backend of R.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct FastT {
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
        self.ln_pdf(x).exp()
    }
    fn ln_pdf(&self, x: f64) -> f64 {
        if x.is_infinite() {
            f64::NEG_INFINITY
        } else {
            let d = x / self.scale;
            
            ln_gamma((self.freedom + 1.0) / 2.0) - 
            (ln_gamma(self.freedom / 2.0) +self.scale.ln()+self.freedom.ln()/2.+std::f64::consts::PI.ln()/2.
             +((self.freedom + 1.0)/2.0) * (d.powi(2)/self.freedom).ln_1p())
            
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

pub trait ContinuousLnCDF<K: Float, T: Float>: Min<K> + Max<K> {

    fn ln_cdf(&self, x: K) -> T;
    fn ln_sf(&self, x: K) -> T;


}


//copied from statrs crate
impl ::rand::distributions::Distribution<f64> for FastT {
    fn sample<R: ::rand::Rng + ?Sized>(&self, r: &mut R) -> f64 {
        // based on method 2, section 5 in chapter 9 of L. Devroye's
        // "Non-Uniform Random Variate Generation"
        let gamma_dist = statrs::distribution::Gamma::new(0.5 * self.freedom, 0.5).unwrap();
        let gamma = gamma_dist.sample(r);

        let norm_dist = statrs::distribution::Normal::new(0.0, self.scale * (self.freedom / gamma).sqrt()).unwrap();
    
        norm_dist.sample(r)
    }
}



impl FastT {

    pub fn new(scale: f64, freedom: f64) -> Result<Self, FastTError> {

        if scale.is_nan() || scale <= 0.0 { return Err(FastTError::ScaleInvalid);}
        if freedom.is_nan() || freedom < 2.0 { return Err(FastTError::FreedomInvalid);}

        Ok(Self{ scale: scale, freedom:freedom})
    }

    pub fn cd_and_sf(&self,x: f64) -> (f64, f64) {

        //let t = Instant::now();
        let q = x/self.scale;
        let q2 = q.powi(2);

        
        if q.abs() >= IMPL_CUT {
        let a = self.freedom/2.;

        let inp = 1./(1.+q2/self.freedom); 

        //let t = Instant::now();
        
        let prew = bpser(a, inp).exp();

        //println!("inp: {} bpser: {:?}", inp, t.elapsed());
        if x > 0.0 {
            (1.-prew/2., prew/2.)
        } else {
            (prew/2., 1.-prew/2.)
        }
        } else {
            let prew = self.maclaurin_cdf(q);
            (prew, 1.-prew)
        }
       
    }

    pub fn ln_cd_and_sf(&self,x: f64) -> (f64, f64) {

        //let t = Instant::now();
        let q = x/self.scale;
        let q2 = q.powi(2);

        
        if q.abs() >= IMPL_CUT {
        let a = self.freedom/2.;

        let inp = 1./(1.+q2/self.freedom); 

        //let t = Instant::now();
        
        let prew = bpser(a, inp);

        //println!("inp: {} bpser: {:?}", inp, t.elapsed());
        if x > 0.0 {
            ((-0.5*(prew.exp())).ln_1p(), prew-M_LN_2)
        } else {
            (prew-M_LN_2, (-0.5*prew.exp()).ln_1p())
        }
        } else {
            let prew = self.maclaurin_cdf(q);
            (prew.ln(), (-prew).ln_1p())
        }
       
    }

    fn maclaurin_cdf(&self, x: f64) -> f64 {
        let g = (-ln_beta_half(self.freedom/2.)).exp()/(self.freedom.sqrt());

        let mut n: f64 = 1.;
        let mut sum: f64 = 0.5+g*x;
        let mut term: f64 = 1.;
        let x2 = x.powi(2);
        while n < 50. { //50 was good enough for our purposes, 
            term *= -x2*(self.freedom-1.+2.*n)/(self.freedom*2.*n);
            sum += g*x*term/(2.*n+1.);
            n += 1.;
        }

        sum
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

fn bpser(a: f64, x: f64) -> f64 {


    //let t = Instant::now();

    let mut ans: f64 = a * x.ln()-ln_beta_half_times_a(a);

    //println!("time ans: {:?}", t.elapsed());

    /* ----------------------------------------------------------------------- */
    /*		       COMPUTE THE SERIES */
    /* ----------------------------------------------------------------------- */
    //let t = Instant::now();
    let mut n: f64 = 0.;
    let mut sum: f64 = 0_f64;
    let mut c: f64 = 1.;
    let mut w: f64;
    //println!("time ass: {:?}", t.elapsed());
    //let t = Instant::now();
    while n < 50. { //This 50 was not theoretically derived, but numerically experimented for. Using a hard number lets the compiler unroll the loop
        n += 1.;
        c *= (0.5 - (0.5 / n)+0.5 ) * x;
        w = c / (a + n);
        sum += w;
    }     
    //println!("fin n {}", n);
    //println!("time ser: {:?}", t.elapsed());
    if a*sum > -1. {ans += (a * sum).ln_1p();}
    else {
        ans = -f64::INFINITY;
    }
    return ans;
} /* bpser */

fn ln_gamma(x: f64) -> f64 {

    let mut s = GAMMA_DK[0];
    if x < 0.5 {
        for i in 1..11 {
            s += GAMMA_DK[i]/(GAMMA_INDS[i]-x);
        }

        consts::LN_PI
            - (f64::consts::PI * x).sin().ln()
            - s.ln()
            - consts::LN_2_SQRT_E_OVER_PI
            - (0.5 - x) * ((0.5 - x + GAMMA_R) / f64::consts::E).ln()
    
    } else {
        for i in 1..11 {
            s+=GAMMA_DK[i]/(x-1.+GAMMA_INDS[i]);
        }
    
         s.ln()
            + consts::LN_2_SQRT_E_OVER_PI
            + (x - 0.5) * ((x - 0.5 + GAMMA_R) / f64::consts::E).ln()
    }

}

#[allow(dead_code)]
fn ln_beta(a: f64, b:f64) -> f64 {
    ln_gamma(a+1.)+ln_gamma(b+1.)-ln_gamma(a+b+1.)
}

fn ln_beta_half_times_a(a: f64) -> f64 {
    LN_GAMMA_HALF+ln_gamma(a+1.)-ln_gamma(a+0.5)
}
fn ln_beta_half(a: f64) -> f64 {
    LN_GAMMA_HALF+ln_gamma(a)-ln_gamma(a+0.5)
}


#[derive(Serialize, Deserialize, Debug)]
#[serde(remote = "Normal")]
struct NormalDef {
    #[serde(getter = "Normal::get_mean")]
    mean: f64,
    #[serde(getter = "Normal::get_sd")]
    std_dev: f64,
}

impl From<NormalDef> for Normal {
    fn from(def: NormalDef) -> Normal {
        Normal::new(def.mean, def.std_dev).unwrap()
    }
}
 
pub(crate) trait Getter {
    fn get_mean(&self) -> f64;
    fn get_sd(&self) -> f64;
}

impl Getter for Normal {
    fn get_mean(&self) -> f64 { self.mean().unwrap() }
    fn get_sd(&self) -> f64 { self.std_dev().unwrap() }
}


#[derive(Debug)]
pub(crate) struct SymmetricBaseDirichlet {
    alpha: f64,
    ln_normalize: f64,
    gamma_sample: Gamma,
}

const MIN_WELL_DEFINED_ALPHA: f64 = unsafe{ std::mem::transmute::<u64, f64>(0x3faa8d46a351a8d4)};

impl SymmetricBaseDirichlet{

    pub(crate) fn new(alpha: f64) -> Result<Self, String> {

        if alpha <= MIN_WELL_DEFINED_ALPHA || alpha.is_infinite() || alpha.is_nan() { return Err(format!("alpha must be well-defined, finite, and at least {MIN_WELL_DEFINED_ALPHA}")); }

        let ln_normalize = ln_gamma((BASE_L as f64)*alpha)-(BASE_L as f64)*ln_gamma(alpha); 

        let gamma_sample = Gamma::new(alpha, 1.0).expect("We already errored out if alpha violated Gamma's invariants");
        Ok(Self { alpha, ln_normalize, gamma_sample })
    }

}

impl Continuous<&Base, f64> for SymmetricBaseDirichlet {

    fn pdf(&self, x: &Base) -> f64{
        self.ln_pdf(x).exp()
    }
    fn ln_pdf(&self, x: &Base) -> f64 {
        self.ln_normalize+((BASE_L-1) as f64)*LN_2.ln()+LN_2*self.alpha*x.scores().iter().sum::<f64>()
            -self.alpha*(BASE_L as f64)*x.scores().iter().map(|&a| a.exp2()).sum::<f64>().ln()-*CONDITION_BASE
    }

}

impl ::rand::distributions::Distribution<Base> for SymmetricBaseDirichlet {

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Base {
        //Note a small optimization: we don't actually divide by the sum of the samples here. 
        //This is because Base's implementation automatically regularizes its inputs for us

        //Avoiding a panic: we already guarenteed that none of the sampled bases will have even a single zero entry
        //        because we defined MIN_WELL_DEFINED_ALPHA as 53/1022. At most, there is a small probability
        //        that this will be slower because we need to delve into subnormal numbers.
        //        How am I able to make this guarentee? The gamma distribution is usually generated with the
        //        GS algorithm from Ahrens and Dieter's 1973 paper "Computer Methods for Sampling from Gamma, Beta,
        //        Poisson and Binomial Distributions", and when alpha is small (the only time we'd have an issue), 
        //        we stay in normal numbers from the gamma sample if 2^(-53), the small uniform random number from (0,1),
        //        taken to the root of alpha, is at least 2^(-1022), the smallest normal number. We DO divide by
        //        gamma(alpha) after, but at 53/1022, that's a little over 18 (less than 2^5), and we have leeway 
        //        with subnormals

        let gammas: [f64;BASE_L] = core::array::from_fn(|_| self.gamma_sample.sample(rng).log2());
        Base::new(gammas)
    }

}



#[cfg(test)]
mod tests{

    use rand::Rng;
    use super::*;
    use std::time::Instant;
    use statrs::function::gamma;
    #[test]
    fn background_attempt() {

        let scale = 0.25;
        let dis = BackgroundDist::new(scale, 2.84).unwrap();

        println!("lge {}", ln_gamma(4.).exp());
        let mut rng = rand::thread_rng();
        let data = [2.2291285570e+0,  4.1510439540e-1, -5.8846691870e-1, -1.4300281780e+0, -1.0174699610e+0,  9.6801737090e-1,  6.1547874390e-1,  1.0320017990e+0, 9.8058816140e-1, 5.7552069380e-1, 10.5, -10.5, 0.125, 0.05, -0.05, 0.24, 0.26, -0.24, -0.26, 0.124, 0.126, -0.124, -0.126];//, 0.005];

        let cd = [-0.001841144352630861, -0.1056418605108091, -2.948328285560595, -5.091991745450134, -4.210455061116229, -0.01694448425305765, -0.04893062330685381, -0.01442399294418845, -0.0164061855049808, -0.05643039312469962,-2.34219612808483e-05, -10.66184817041387, -0.3954384654261913, -0.5576877143913497, -0.8498744215025229, -0.2303377202464982, -0.2098933691304, -1.581167890431584, -1.664267366838764, -0.3972910110417498, -0.3935939313076429, -1.115163703158927, -1.122785969616946];//, -0.6786179549835148];
        let sf = [-6.298288400836528, -2.300056544954666, -0.05385159158458796, -0.006164729845405886, -0.01495082250690833, -4.086273189920248, -3.041717390080652, -4.246065609924472, -4.118288727978808, -2.902829897631842, -10.66184817041387, -2.34219612808483e-05, -1.118972310171169,-0.8498744215025229, -0.5576877143913497, -1.581167890431584, -1.664267366838764, -0.2303377202464982, -0.2098933691304, -1.115163703158927, -1.122785969616946, -0.3972910110417498, -0.3935939313076429];//, -0.7078906207282351];

        for (i, dat) in data.iter().enumerate() {
            let (c, s) = dis.ln_cd_and_sf(*dat);
            println!("dat {}: cd {}, sf {}, prop_cd {}, prop_sf {}, diff cd {}, diff sf {}",dat, c, s, cd[i], sf[i], c-cd[i], s-sf[i]);
            assert!(((c-cd[i]).abs() < 1e-3) && ((s-sf[i]).abs() < 1e-6));
        }

        let f = match &dis {
            BackgroundDist::Normal(_norm) => 0.5,
            BackgroundDist::FastT(fast) => fast.maclaurin_cdf(IMPL_CUT),
        };

        let (cs, sf) = dis.ln_cd_and_sf(IMPL_CUT*scale);
        let (asc, asf) = dis.cd_and_sf(IMPL_CUT*scale);
        println!("cut {} mac {} {} bpser {} {} diffc {}", IMPL_CUT*scale, f.ln(), (-f).ln_1p(), cs, sf, cs-f.ln());

        println!("confirm ln and arith: exp of ln: {}, arith {}", cs.exp(), asc);

        assert!(((cs.exp()-asc).abs() < 1e-6) && ((sf.exp()-asf).abs() < 1e-6), "ln and arith incompatible");

        for _ in 0..50 {
        let a: f64 = rng.gen::<f64>();
        let t = Instant::now();
        let c = gamma::ln_gamma(a);
        


        println!("t {:?}", t.elapsed());
        println!("A {} c {} ", a, c);
        }
        let mut rng = rand::thread_rng();

        let nums: Vec<f64> = (0..1000000).map(|_| rng.gen::<f64>()).collect();
        let num_pt = nums.as_ptr();
        let (mut n, mut o) = (0_f64, 0_f64);
        let t = Instant::now();
        for i in 0_isize..(nums.len() as isize) {
            unsafe {
            let (a,b) = dis.ln_cd_and_sf(*num_pt.offset(i));
            n+=a;
            o+=b;
            }
        }

        let (v, u) = dis.ln_cd_and_sf(nums[0]);
        println!("Time {:?}", t.elapsed());

        println!("{} {} {} {}", n, o, v, u);
        
    }
}
