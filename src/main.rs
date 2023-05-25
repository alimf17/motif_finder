#[allow(unused_parens)]
mod base;
mod sequence;
mod waveform;
mod data_struct;

/*use crate::base::bases::Base;
use crate::base::bases::GBase;
use crate::base::bases::TruncatedLogNormal;
use crate::base::bases::Motif;*/
use crate::base::*;
use crate::sequence::Sequence;
use crate::waveform::*;
use statrs::distribution::{Continuous, ContinuousCDF, LogNormal, Normal};
use statrs::statistics::{Min, Max};
use statrs::function::gamma;
use rand::Rng;
use std::collections::VecDeque;
use rand::distributions::{Distribution, Uniform};
use aberth::aberth;
use num_complex::Complex;
const EPSILON: f64 = 1e-8;
use once_cell::sync::Lazy;
use num_traits::cast;
use num_traits::float::Float;
use num_traits::float::FloatConst;
use num_traits::identities::{One, Zero};
use num_traits::MulAdd;
use core::iter::zip;
use std::time::{Duration, Instant};
use rayon::iter::IntoParallelIterator;
use rayon::iter::ParallelIterator;

use serde_json::value::Serializer as Json_Serializer;

use serde::{ser, Serialize,Serializer, Deserialize};
use serde::de::{
    self, DeserializeSeed, EnumAccess, IntoDeserializer, MapAccess, SeqAccess,
    VariantAccess, Visitor,
};

// 0 = -1 + 2x + 4x^4 + 11x^9
//let polynomial = [-1., 2., 0., 0., 4., 0., 0., 0., 0., 11.];

//let roots = aberth(&polynomial, EPSILON).unwrap();

fn main() {

    let mut rng = fastrand::Rng::new();

    let block_n: usize = 200;
    let u8_per_block: usize = 4375;
    let bp_per_block: usize = u8_per_block*4;
    let bp: usize = block_n*bp_per_block;
    let u8_count: usize = u8_per_block*block_n;

    //let blocks: Vec<u8> = (0..u8_count).map(|_| rng.u8(..)).collect();
    let preblocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.u8(..)).collect();
    let blocks: Vec<u8> = preblocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>();
    let block_inds: Vec<usize> = (0..block_n).map(|a| a*u8_per_block).collect();
    let block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
    let sequence: Sequence = Sequence::new_manual(blocks, block_lens);

    let predata: Waveform = Waveform::create_zero( &sequence, 5);
    let motif: Motif = Motif::from_motif(sequence.return_bases(0,0,20), 20.);

    let motif2 = Motif::rand_mot(20.,  &sequence);

    let corrs: Vec<f64> = vec![0.9, -0.1];

    let background = Background::new(0.25, 2.64, &corrs);
    

    let binds = motif.return_bind_score(&sequence);

    let single_wave = motif.generate_waveform(&predata);
    let data = &single_wave+&motif2.generate_waveform(&predata);

    let noise = (&data-&single_wave).produce_noise(&data, &background);
    let dist = noise.dist();
    let start = Instant::now();
    let grad = motif.single_motif_grad(&data, &noise);
    println!("Grad calc {:?}", start.elapsed());

    let start = Instant::now();
    let ad_grad = noise.ad_grad();
    let d_ad_like = Noise::ad_diff(noise.ad_calc());
    let omittable = Instant::now();
    let grad2 = unsafe { motif.parallel_single_motif_grad(&data, &ad_grad, d_ad_like, &background)};
    println!("Grad calc par {:?}. Without noise calcs: {:?}", start.elapsed(), omittable.elapsed());

    let mut old_grad = grad.1.clone();
    old_grad.insert(0, grad.0);
    println!("Grad differences: {}", old_grad.iter().zip(grad2.iter()).map(|(a, &b)| (a-b).powi(2)).sum::<f64>().sqrt());

    println!("gradient {:?}", grad2);

    let res = noise.resids();
    let approx = Instant::now();
    let approx_pdf = res.clone().into_iter().map(|a| background.cdf(a)).collect::<Vec<_>>();
    let dur_app = approx.elapsed();
    let approx_par = Instant::now();
    let approx_pdf = res.clone().into_par_iter().map(|a| background.cdf(a)).collect::<Vec<_>>();
    let dur_par = approx_par.elapsed();

    let exact = Instant::now();
    let exact_pdf = res.iter().map(|a| dist.cdf(*a)).collect::<Vec<_>>();
    let exact_app = exact.elapsed();

    println!("Approx: {:?}, Par Approx: {:?}, Exact: {:?}", dur_app,dur_par, exact_app);

    //println!("{}", (&sequence).serialize(Json_Serializer).unwrap());
    //println!("{}", (&background).serialize(Json_Serializer).unwrap());
    /*
    //1+2x^2+x^4 = (1+x^2)^2
    //let polynomial: [f64; 5] = [1.0, 0.0,2.0, 0.0, 1.0];

    let polynomial: [f64; 9] = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0];

    let roots = aberth(&polynomial, EPSILON).unwrap();

    for root in roots {

        println!("root {}", root);
    }

    let other_poly: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.0];

    let roots = aberth2(&other_poly, EPSILON).unwrap();

    println!("root b");
    
    for root in roots {

        println!("rootb {}", root);
    }
*/


}
/*
pub fn aberth2(polynomial: &Vec<f64>, epsilon: f64) -> Result<Vec<Complex<f64>>, &'static str> {
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
    println!("");
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


*/
