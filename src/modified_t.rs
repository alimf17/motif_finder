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

        let b = self.freedom/2.;

        let major_branch = q2 >= self.freedom;

        //inp is technically always mathematically the same thing, but we calculate it slightly 
        //differently because theoretically the same and numerically the same aren't the same thing
        let inp = if major_branch {1./(1.+q2/self.freedom)} else { self.freedom/(self.freedom+q2)};

        let (prew, prew1) = if major_branch {
            let w = modified_bpser(0.5, b, inp, 1e-14);
            let w_alt = bpser(0.5, b, inp, 1e-16);
            println!("mb: {}, b: {}", w, w_alt);
            (w, (-w.exp()).ln_1p())
        } else if inp >= 0.29 {
            let w1 = modified_bpser(b, 0.5, 1.0-inp, 1e-14);
            ((-w1.exp()).ln_1p(), w1)
        } else {
            let n = 20;
            let mut w1 = modified_bup(b, inp, 1e-14, n);
            w1 = modified_bgrat(b+(n as f64), inp, w1, 1e-14);
            ((-w1.exp()).ln_1p(), w1)
        };

        println!("x: {}, q2: {}, inp: {}, b: {}, Ix: {}, Ixf: {},prew: {}, prew1: {}", x, q2,inp, b,beta_inc(0.5, b, inp)/beta(0.5, b), beta_inc(b, 0.5, inp)/beta(b,0.5),prew, prew1);
        if (x > 0.0) {
            ((-0.5*(prew.exp())).ln_1p(), prew1-M_LN_2)
        } else {
            (prew1-M_LN_2, (-0.5*prew.exp()).ln_1p())
        }
       
    }

}


//This is an attenuated version of the function bpser
//from R's C implementation of its statistics functions
//It is attenuated because we need it for very particular
//use cases

//In particular, when we use it, one of the values is always 1/2
//Also, we are using the ln_gamma function from statrs instead of 
//the official R implementation: it's accurate up to about 6 bits at
//the very end of our floats and takes ~60ns regardless of input
fn modified_bpser(a: f64, b: f64, x: f64, eps: f64) -> f64 {

    if x == 0. {return -f64::INFINITY;  }
    if( x.is_nan() ) {return f64::NAN; }

    let a0 = a.min(b);
    let b0 = a.max(b);
 
    let ln_dens = x.ln()/2.-ln_beta(a0,b0);

    let mut sum = 0.0;
    let mut c = 0.0;
    let mut n = 0.;
    let mut w = f64::INFINITY;

    let tol = eps/2.;

    while (n < 1e7) && (w.abs() > tol) {
        n+=1.0;
        c *= (0.5 - (b/n) + 0.5) * x;
        w = c / (a + n);
        sum += w;
    }

    ln_dens+(sum/2.).ln_1p()

    //implemented
}

fn bpser(a: f64, b: f64, x: f64, eps: f64) -> f64 {

    let a0 = a.min(b);

    let b0 = a.max(b);

    let mut ans: f64 = a * x.ln()-ln_gamma(a+1.)-ln_gamma(b) + ln_gamma(a+b);

    println!("preser: {}, x: {}, a: {}, b: {}", ans, x, a, b);

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
        c *= (0.5 - (b / n)+0.5 ) * x;
        w = c / (a + n);
        sum += w;
    }     
    if(w.abs() > tol) { // the series did not converge (in time)
                        // warn only when the result seems to matter:
        warn!("bpser ans doesn't seem to have converged");
    }
    
    if (a*sum > -1.) {ans += (a * sum).ln_1p();}
    else {
        ans = -f64::INFINITY;
    }
    return ans;
} /* bpser */

fn modified_bup(a: f64, x:f64, eps: f64, n: usize) -> f64 {

    let rem = a-a.floor();
    let retval = modified_brcmp(rem, x);

    let apb = a+0.5;
    let ap1 = a+1.;

    let mut w = 1.0_f64;
    let mut d = 1.;

    for i in 0..(n-1) {

        let l = i as f64;
        d *= x*(apb+l)/(ap1+l);
        w+=d;
        if (d <= eps * w) {
            break;
        }

    }

    retval+w

        //implemented

}



fn modified_bgrat(a: f64, x: f64, w: f64, eps: f64) -> f64 {

    let nu = (a-1.)/2.+0.5;
    let lnx = x.ln();
    let z = -nu * lnx;

    let b = 0.5_f64;

    let mut c = [0.0_f64; NUM_BGRAT_TERMS];
    let mut d = [0.0_f64; NUM_BGRAT_TERMS];

    let log_r = a.ln() - ln_gamma(a+1.) + a * z.ln() - z;
    let log_u = log_r - (ln_beta_with_3_halfs(a)-LN_GAMMA_3_HALFS + a * nu.ln());
    let u = log_u.exp();

    let mut wp = w;

    let q_r = grat_r(z, log_r);
    let v = 0.25 / (nu * nu);
	let t2 = lnx * 0.25 * lnx;
	let mut j = q_r;
	let mut sum = j;
	let mut t = 1.;
    let mut cn = 1.;
    let mut n2 = 0.;


    for n in 0..NUM_BGRAT_TERMS {

        let np1 = (n+1) as f64;
        let bp2n = b + n2;
	    j = (bp2n * (bp2n + 1.) * j + (z + bp2n + 1.) * t) * v;
	    n2 += 2.;
	    t *= t2;
	    cn /= n2 * (n2 + 1.);
	    c[n] = cn;
        let mut s: f64 = 0.;
        if (n > 0) {
            let mut coef = b - np1;
            for i in 1..=n {
                s += coef * c[i - 1] * d[n - i];
                coef += b;
            }
        }
        d[n] = (b-1.) * cn + s / np1;
        sum += (d[n] * j);


    }

    logspace_add(w, log_u+sum.ln())
}

//Panics: x not an element of (0,1)
fn modified_brcmp(a: f64, x: f64) -> f64 {

    let a0 = 0.5_f64.min(a);
    let b0 = 0.5_f64.max(a);



    let lnx = x.ln();
    let ln1mx = (-x).ln_1p();
 
    
    let ans = a*lnx+0.5*ln1mx-ln_beta_with_3_halfs(a);

    ans+a0.ln()-(a0/b0).ln_1p()

    //implemented

}

fn ln_beta(a: f64, b:f64) -> f64 {
    ln_gamma(a+1.)+ln_gamma(b+1.)-ln_gamma(a+b+1.)
}

fn ln_beta_with_3_halfs(a: f64) -> f64 {
    LN_GAMMA_3_HALFS+ln_gamma(a+1.)-ln_gamma(a+1.5)
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

        let dis = BackgroundDist::new(1.0, 2.84);

        let data = [-0.3, 0.3, 4.5, -2.9];

        for dat in data.iter() {
            let (c, s) = dis.ln_cd_and_sf(*dat);
            println!("cd {}, sf {}",c, s);
        }


    }
}
