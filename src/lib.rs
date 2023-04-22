#[allow(unused_parens)]
pub mod base;
pub mod sequence;
pub mod waveform;


use crate::base::bases::Base;
use crate::base::bases::GBase;
use crate::base::bases::TruncatedLogNormal;
use crate::base::bases::Motif;
use crate::sequence::seq::Sequence;
use statrs::distribution::{Continuous, ContinuousCDF, LogNormal, Normal};
use statrs::statistics::{Min, Max};
use statrs::function::gamma;
use rand::Rng;
use std::collections::VecDeque;
use rand::distributions::{Distribution, Uniform};


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

    let motif: Motif = unsafe{Motif::from_clean_motif(sequence.return_bases(0,0,20), 20., &sequence)};

    let binds = motif.return_bind_score(&sequence);

}

