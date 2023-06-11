use rand::Rng;
use rand::seq::SliceRandom;
use rand::distributions::{Distribution, Uniform, WeightedIndex};
use statrs::distribution::{Continuous, ContinuousCDF, LogNormal, Normal, Dirichlet, Exp};
use statrs::function::{gamma, erf}
use statrs::statistics::{Min, Max, Distribution as OtherDistribution};
use num_traits::Float;



pub enum BackgroundDist {
    Normal(Normal),
    FastT(FastT),
}

struct FastT {
    scale: f64,
    freedom: f64,
}

const LN_GAMMA_3_HALFS = -0.120782237635245222;

impl ContinuousCDF<f64, f64> for FastT {


    fn cdf(&self, calc: f64) -> f64{
        todo!()
    }

    fn sf(&self, calc: f64) -> f64 {
        todo!()
    }
}

impl ContinuousLnCDF<f64, f64> for FastT {


    fn ln_cdf(&self, calc: f64) -> f64{
        todo!()
    }

    fn ln_sf(&self, calc: f64) -> f64 {
        todo!()
    }
}
impl Continuous<f64, f64> for FastT {

    fn pdf(&self, calc: f64) -> f64{
        todo!()
    }
    fn ln_pdf(&self, calc: f64) -> f64 {
        todo!()
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

    fn ln_cd_and_sf(&self,x: f64) -> (f64, f64) {

        let q2 = (x/self.scale).powi(2);

        let major_branch = q2 >= self.freedom;

        let inp = if major_branch {1./(1.+q2/self.scale)} else { self.scale/(self.scale+q2)};

        if major_branch {
            let w = modified_bpser(self.scale/2, 1/2, inp, 1e-14, true);
            (w, ln_1p(-w.exp()))
        }

        if inp >= 0.29 {
            let w1 = modified_bpser(self.scale/2, 1/2, inp, 1e-14, true);
            (ln_1p(-w1.exp()), w)
        }

        let n = 20;

        let mut w1 = bup(self.scale/2., inp, 1e-15, n, true);



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
fn modified_bpser(b: f64, x: f64, eps: f64, get_ln: bool) -> f64 {

    if (x == 0) { if ln_dens {-f64:INFINITY } else {0.0} }
    if( x.is_nan() ) {f64::NAN}

    let a0 = 0.5_f64.min(b);
    let b0 = 0.5_f64.max(b);
 
    let ln_dens = x.ln()/2.-ln_beta_with_3_halfs(b0)

    let mut sum = 0.0;
    let mut c = 0.0;
    let mut n = 0.;
    let mut w = f64:INFINITY;

    let tol = eps/2.;

    while (n < 1e7) && (w.abs() > tol) {
        n+=1.0;
        c *= (0.5 - b / n + 0.5) * x;
        w = c / (a + n);
        sum += w;
    }

    if get_len {
        ln_dens+(sum/2.).ln_1p()
    } else {
        ln_dens.exp()*((sum/2.+1.))
    }

    //implemented
}


fn modified_bup(a: f64, x:f64, eps: f64, n: usize, get_ln: bool) -> f64 {

    let rem = a-a.floor();
    let retval = modified_brcmp(rem, x);

    let apb = a+0.5;
    let ap1 = a+1.;

    let mut w = 1.0_f64;
    let mut d = 1.;

    for i in 0..(n-1) {

        l = i as f64;
        d *= x*(apb+l)/(ap1+l);
        w+=d;
        if (d <= eps * w) {
            break;
        }

    }

    if get_ln {
        ret_val+w
    } else {
        (ret_val+w).exp()
    }

    //implemented

}

fn modified_bgrat(a: f64, x: f64, w: f64, eps: f64, get_ln: bool) -> f64 {
    let nu = (a-1.)/2.+0.5;
    let z = -nu * x.ln();

    let log_r = a.ln() - ln_gamma(a+1.) + a * z.ln() - z;
    let log_u = log_r - (ln_beta_with_3_halfs(a)-LN_GAMMA_3_HALFS + a * nu.ln());
    let u = log_u.exp();

    let mut wp = w;

    todo!();



}

//Panics: x not an element of (0,1)
fn modified_brcmp(a: f64, x: f64) -> f64 {

    let a0 = 0.5_f64.min(a);
    let b0 = 0.5_f64.max(a);


    let lnx = x.ln();
    let ln1mx = (-x).ln_1p();
 
    
    let ans = a*lnx+b*ln1mx-ln_beta_with_3_halfs(a);

    ans+a0.ln()-(a0/b0).ln_1p();

    //implemented

}

fn ln_beta_with_3_halfs(a: f64) -> f64 {
    LN_GAMMA_3_HALFS+ln_gamma(a+1.)-ln_gamma(a+1.5)
}


