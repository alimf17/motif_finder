use rand::Rng;
use rand::seq::SliceRandom;
use rand::distributions::{Distribution, Uniform, WeightedIndex};
use statrs::distribution::{Continuous, ContinuousCDF, LogNormal, Normal, Dirichlet, Exp};
use statrs::statistics::{Min, Max, Distribution as OtherDistribution};
use num_traits::Float;



pub enum BackgroundDist {
    Normal(Normal),
    FastT(FastT),
}

struct FastT {

    sigma_background: f64,
    df: f64,
}


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
