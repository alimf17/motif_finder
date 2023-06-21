use rand::Rng;
use rand::seq::SliceRandom;
use rand::distributions::{Distribution, Uniform, WeightedIndex};
use statrs::distribution::{Continuous, ContinuousCDF, LogNormal, Normal, Dirichlet, Exp};
use statrs::function::{erf::*, beta::* };
use statrs::consts;
use statrs::statistics::{Min, Max, Distribution as OtherDistribution};
use num_traits::Float;
use log::warn;
use serde::{Serialize, Deserialize};

use std::f64;
    use std::time::{Duration, Instant};
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

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum BackgroundDist {

    #[serde(with = "NormalDef")]
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
    pub fn cd_and_sf(&self,x: f64) -> (f64, f64) {
        match self {
            BackgroundDist::Normal(norm) => (norm.cdf(x), norm.sf(x)),
            BackgroundDist::FastT(fast) => fast.cd_and_sf(x)
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

pub trait ContinuousLnCDF<K: Float, T: Float>: Min<K> + Max<K> {

    fn ln_cdf(&self, x: K) -> T;
    fn ln_sf(&self, x: K) -> T;


}

impl FastT {

    pub fn cd_and_sf(&self,x: f64) -> (f64, f64) {

        //let t = Instant::now();
        let q = (x/self.scale);
        let q2 = q.powi(2);

        
        if q.abs() >= IMPL_CUT {
        let a = self.freedom/2.;

        let inp = 1./(1.+q2/self.freedom); 

        //let t = Instant::now();
        
        let prew = bpser(a, inp, 1e-8).exp();

        //println!("inp: {} bpser: {:?}", inp, t.elapsed());
        if (x > 0.0) {
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
        let q = (x/self.scale);
        let q2 = q.powi(2);

        
        if q.abs() >= IMPL_CUT {
        let a = self.freedom/2.;

        let inp = 1./(1.+q2/self.freedom); 

        //let t = Instant::now();
        
        let prew = bpser(a, inp, 1e-8);

        //println!("inp: {} bpser: {:?}", inp, t.elapsed());
        if (x > 0.0) {
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
        while n < 50. {
            //println!("n {} sum {}", n, sum);
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

fn bpser(a: f64, x: f64, eps: f64) -> f64 {


    //let t = Instant::now();

    let mut ans: f64 = a * x.ln()-ln_beta_half_times_a(a);

    //println!("time ans: {:?}", t.elapsed());

    /* ----------------------------------------------------------------------- */
    /*		       COMPUTE THE SERIES */
    /* ----------------------------------------------------------------------- */
    //let t = Instant::now();
    let tol: f64 = eps / a;
    let mut n: f64 = 0.;
    let mut sum: f64 = 0_f64;
    let mut c: f64 = 1.;
    let mut w: f64 = f64::INFINITY;
    //println!("time ass: {:?}", t.elapsed());
    //let t = Instant::now();
    //while (n < 1e7 && w.abs() > tol) { // sum is alternating as long as n < b (<==> 1 - b/n < 0)
    while n < 50. { //This 40 was not theoretically derived, but numerically experimented for
        n += 1.;
        c *= (0.5 - (0.5 / n)+0.5 ) * x;
        w = c / (a + n);
        sum += w;
    }     
    //println!("fin n {}", n);
    //println!("time ser: {:?}", t.elapsed());
    if (a*sum > -1.) {ans += (a * sum).ln_1p();}
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

fn ln_beta(a: f64, b:f64) -> f64 {
    ln_gamma(a+1.)+ln_gamma(b+1.)-ln_gamma(a+b+1.)
}

fn ln_beta_half_times_a(a: f64) -> f64 {
    LN_GAMMA_HALF+ln_gamma(a+1.)-ln_gamma(a+0.5)
}
fn ln_beta_half(a: f64) -> f64 {
    LN_GAMMA_HALF+ln_gamma(a)-ln_gamma(a+0.5)
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

#[derive(Debug)]
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


#[derive(Serialize, Deserialize, Debug)]
#[serde(remote = "Normal")]
pub struct NormalDef {
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
 
pub trait Getter {
    fn get_mean(&self) -> f64;
    fn get_sd(&self) -> f64;
}

impl Getter for Normal {
    fn get_mean(&self) -> f64 { self.mean().unwrap() }
    fn get_sd(&self) -> f64 { self.std_dev().unwrap() }
}


#[cfg(test)]
mod tests{

    use rand::Rng;
    use super::*;
    use std::time::{Duration, Instant};
    use statrs::function::gamma;
    #[test]
    fn attempt() {

        let scale = 0.25;
        let dis = BackgroundDist::new(scale, 2.84);

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
            BackgroundDist::Normal(norm) => 0.5,
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

        
    }
}
