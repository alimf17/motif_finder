
use motif_finder::sequence::seq::Sequence;
use rand::Rng;
use criterion::{black_box, criterion_group, criterion_main, Criterion, Throughput, BenchmarkId};


fn criterion_benchmark(c: &mut Criterion) {

    let mut rng = rand::thread_rng();
    let blocks: Vec<u8> = (0..875000).map(|_| rng.gen::<u8>()).collect();
    let block_inds: Vec<usize> = (0..280).map(|a| a*3125).collect();
    let start_bases: Vec<usize> = (0..280).map(|a| a*12500).collect();
    let block_lens: Vec<usize> = (0..280).map(|_| 3125).collect();
    let sequence: Sequence = Sequence::new_manual(blocks, block_inds, start_bases,block_lens);

    let nbases = 8;
    let block = (0..28).collect::<Vec<_>>(); 
    let inds = (0..(3125-nbases)).collect::<Vec<_>>();
    
    c.bench_function("blockT", |b| b.iter(|| block.iter().zip(inds.iter()).map(|(&l, &i)| sequence.return_bases(l, i, nbases)).collect::<Vec<_>>()));        
    
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
