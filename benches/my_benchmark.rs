
use motif_finder::sequence::seq::Sequence;
use rand::Rng;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput, BenchmarkId};
use motif_finder::base::bases::Motif;

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
    let start_bases: Vec<usize> = (0..block_n).map(|a| a*bp_per_block).collect();
    let block_lens: Vec<usize> = (1..(block_n+1)).map(|_| bp_per_block).collect();
    let sequence: Sequence = Sequence::new_manual(blocks, block_inds, block_lens);

    let motif: Motif = Motif::from_clean_motif(sequence.return_bases(0,0,20), 20., &sequence);

    let binds = motif.return_bind_score(&sequence);


}

