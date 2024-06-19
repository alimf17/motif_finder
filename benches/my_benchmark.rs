
use motif_finder::sequence::{Sequence, NullSequence};
use motif_finder::waveform::*;
use motif_finder::data_struct::*;
use rand::Rng;
use rand::distributions::Distribution;
use statrs::function::gamma::*;

use motif_finder::base::Motif;

fn main() {

    std::env::set_var("RUST_BACKTRACE", "1");

    let mut rng = rand::thread_rng(); //fastrand::Rng::new();
    let spacing_dist = rand::distributions::Uniform::from(500..5000);
    let block_n: usize = 20;
    let u8_per_block: usize = 5000;
    let bp_per_block: usize = u8_per_block*4;
    let _bp: usize = block_n*bp_per_block;
    let u8_count: usize = u8_per_block*block_n;


    let preblocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.gen::<u8>()).collect();
    let blocks: Vec<u8> = preblocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>();
    let _block_u8_starts: Vec<usize> = (0..block_n).map(|a| a*u8_per_block).collect();
    let block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
    let start_bases: Vec<usize> = (0..block_n).map(|a| a*bp_per_block).collect();
    let sequence: Sequence = Sequence::new_manual(blocks, block_lens.clone());


    let _pre_null_blocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.gen::<u8>()).collect();
    let null_blocks: Vec<u8> = preblocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>();
    let _block_u8_null_starts: Vec<usize> = (0..block_n).map(|a| a*u8_per_block).collect();
    let null_block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
    let start_null_bases: Vec<usize> = (0..block_n).map(|a| 2*a*bp_per_block).collect();
    let null_seq: NullSequence = NullSequence::new_manual(null_blocks, null_block_lens.clone());

    let start_bases_copy = (0..block_n).map(|a| (2*a+1)*bp_per_block+10).collect();


    let wave: Waveform = Waveform::create_zero(&sequence, 5);

    let background = Background::new(0.25, 2.64, 20.);


    let data_seq = unsafe{ AllDataUse::new_unchecked_data(wave.clone(),&null_seq, &start_bases_copy,&start_null_bases, &background)};

    println!("{} gamma", gamma(4.));
    println!("{} gamma", ln_gamma(4.));

    //println!("{:?}", wave.raw_wave());

    let motif: Motif = Motif::from_motif(sequence.return_bases(0,0,20), &mut rng); //sequence

    let motif2: Motif = Motif::from_motif(sequence.return_bases(0,2,20), &mut rng); //sequence


    let waveform = motif.generate_waveform(&data_seq);

    let waveform2 = &waveform + &(motif2.generate_waveform(&data_seq));

    let coordinate_bases: Vec<usize> = start_bases.iter().map(|&a| a+spacing_dist.sample(&mut rng)).collect();

    let data_seq_2 = unsafe{ AllDataUse::new_unchecked_data(waveform2, &null_seq, &coordinate_bases,&start_null_bases ,&background) };

    let _noise: Noise = waveform.produce_noise(&data_seq_2);



}
