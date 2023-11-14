pub mod sequence;
pub mod waveform;
pub mod modified_t;
pub mod base;
pub mod data_struct;

use once_cell::sync::Lazy;
use statrs::distribution::Normal;

pub const NULL_CHAR: Option<char> = None;

const MOMENTUM_SD: f64 = 0.1;
static MOMENTUM_DIST: Lazy<Normal> = Lazy::new(|| Normal::new(0.0, MOMENTUM_SD).unwrap() );

pub const NUM_CHECKPOINT_FILES: usize = 25;

pub const NUM_RJ_STEPS: usize = 1;
const MAX_IND_RJ: usize = NUM_RJ_STEPS-1;
pub const NUM_BASE_LEAP_STEPS: usize = 1;
const MAX_IND_LEAP: usize = NUM_RJ_STEPS+NUM_BASE_LEAP_STEPS-1;
pub const NUM_HMC_STEPS: usize = 1;
const MAX_IND_HMC: usize = MAX_IND_LEAP+NUM_HMC_STEPS;

const HMC_TRACE_STEPS: usize = 1;
const HMC_EPSILON: f64 = 1./(4096.*16.); //2^(-21)

//This only matters when taking in a meme file
pub const MAX_E_VAL: f64 = 0.01;

