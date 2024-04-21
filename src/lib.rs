pub mod sequence;
pub mod waveform;
pub mod modified_t;
pub mod base;
pub mod data_struct;

use crate::modified_t::SymmetricBaseDirichlet;

use std::sync::RwLock;

use once_cell::sync::{OnceCell};
use statrs::distribution::Normal;

pub const NULL_CHAR: Option<char> = None;

pub static MOMENTUM_SD: RwLock<f64> = RwLock::new(11.0);
//static MOMENTUM_DIST: Lazy<Normal> = Lazy::new(|| Normal::new(0.0, MOMENTUM_SD).unwrap() );
pub static MOMENTUM_DIST: OnceCell<Normal> = OnceCell::new();//Lazy::new(|| Normal::new(0.0, MOMENTUM_SD).unwrap() );

pub const NUM_CHECKPOINT_FILES: usize = 25;

pub const NUM_RJ_STEPS: usize = 1;
const MAX_IND_RJ: usize = NUM_RJ_STEPS-1;
pub const NUM_BASE_LEAP_STEPS: usize = 1;
const MAX_IND_LEAP: usize = NUM_RJ_STEPS+NUM_BASE_LEAP_STEPS-1;


//This only matters when taking in a meme file
pub const MAX_E_VAL: f64 = 0.01;


//These are mutable statics ONLY at the beginning of the programming. They are never to change
//after initialization. In particular, if they change in the middle of calculations, that's a data
//race, IE UB. 

//pub static PROPOSE_EXTEND: RwLock<Lazy<SymmetricBaseDirichlet>> = RwLock::new(Lazy::new(|| SymmetricBaseDirichlet::new(1.0_f64).unwrap()));
//pub static PROPOSE_EXTEND: RwLock<SymmetricBaseDirichlet> = RwLock::new(SymmetricBaseDirichlet::new(1.0_f64).unwrap());


//pub static DIRICHLET_PWM: RwLock<Lazy<SymmetricBaseDirichlet>> = RwLock::new(Lazy::new(|| SymmetricBaseDirichlet::new(1.0_f64).unwrap()));


pub static PROPOSE_EXTEND: OnceCell<SymmetricBaseDirichlet> = OnceCell::new();


//pub static DIRICHLET_PWM: RwLock<Lazy<SymmetricBaseDirichlet>> = RwLock::new(Lazy::new(|| SymmetricBaseDirichlet::new(1.0_f64).unwrap()));

pub static DIRICHLET_PWM: OnceCell<SymmetricBaseDirichlet> = OnceCell::new();

pub static THRESH: RwLock<f64> = RwLock::new(1e-6);

pub static NECESSARY_MOTIF_IMPROVEMENT: RwLock<f64> = RwLock::new(5.0_f64);

