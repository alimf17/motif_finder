pub mod seq {

    use std::mem::size_of_val;
    use std::collections::HashMap;
    use itertools::{all, Itertools};
    
    //const PLACE_VALS: [u8; 4] = [64, 16, 4, 1];
    const PLACE_VALS: [u8; 4] = [1, 4, 16, 64];

    pub struct Sequence {
        seq_blocks: Vec<u8>,
        block_inds: Vec<usize>,
        block_lens: Vec<usize>,
        max_len: usize,
    }

    impl Sequence {
        
        pub fn new(blocks: &Vec<Vec<usize>>) -> Sequence {

            let mut block_is: Vec<usize> = Vec::new();

            let mut s_bases: Vec<usize> = Vec::new();

            let mut block_ls: Vec<usize> = Vec::new();

            let mut seq_bls: Vec<u8> = Vec::new();

            let mut p_len: usize = 0;

            let mut b_len: usize = 0;

            let mut max_len: usize = 0;

            
            for block in blocks {
                
                s_bases.push(b_len);
                
                block_is.push(p_len);

                if block.len() % 4 != 0 {
                    panic!("All blocks must have a number of base pairs divisible by 4.")
                }

                let num_entries = block.len()/4;
                
                for i in 0..num_entries {
                    let precoded = &block[(4*i)..(4*i+4)].try_into().expect("slice with incorrect length");
                    seq_bls.push(Self::bases_to_code(precoded));
                }

                p_len += block.len()/4;
                b_len += block.len();

                block_ls.push(block.len());

                max_len = if block.len() > max_len {block.len()} else {max_len};

            }

            Sequence{
                seq_blocks: seq_bls,
                block_inds: block_is,
                block_lens: block_ls,
                max_len: max_len,
            }



        }

        pub fn new_manual(seq_blocks: Vec<u8>, block_inds: Vec<usize>, block_lens: Vec<usize>) -> Sequence {


            let max_len: usize = *block_lens.iter().max().unwrap();

            Sequence{
                seq_blocks: seq_blocks,
                block_inds: block_inds,
                block_lens: block_lens,
                max_len: max_len,
            }

        }

        
        pub fn generate_kmers(&self, len: usize) -> Vec<Vec<usize>> {


                
                let mut unel: Vec<Vec<usize>> = Vec::new();

                for i in 0..self.block_lens.len() {
                    
                    if self.block_lens[i] >= len {
                       
                        for j in 0..(self.block_lens[i]-len+1){
                            unel.push(self.return_bases(i, j, len));
                        }

                    }

                }

                unel = unel.into_iter().unique().collect();
                unel
            
          
        }


        pub fn seq_blocks(&self) -> Vec<u8> {
            self.seq_blocks.clone()
        }

        pub fn coded_place(&self, i: usize) -> u8 {
            self.seq_blocks[i]
        }

        pub fn block_lens(&self) -> Vec<usize> {
            self.block_lens.clone()
        }
        
        pub fn max_len(&self) -> usize {
            self.max_len
        }

        pub fn block_inds(&self) -> Vec<usize> {
            self.block_inds.clone()
        }

        pub fn bases_to_code(bases: &[usize ; 4]) -> u8 {

            bases.iter().zip(PLACE_VALS).map(|(&a, b)| a*b as usize).sum::<usize>() as u8
        
        }

        pub fn code_to_bases(coded: u8) -> [usize ; 4] {

            let mut V: [usize; 4] = [0; 4];

            let mut reference: u8 = coded;

            for i in 0..4 { 

                V[i] = (0b00000011 & reference) as usize;
                reference = reference >> 2; 

            }

            V

        }

        pub fn return_bases(&self, block_id: usize, start_id: usize, num_bases: usize) -> Vec<usize> {

            let start_dec = self.block_inds[block_id]+(start_id/4);
            let end_dec = self.block_inds[block_id]+((start_id+num_bases-1)/4)+1;

            let all_coded = &self.seq_blocks[start_dec..end_dec];

            let all_bases: Vec<_> = all_coded.iter().map(|a| Self::code_to_bases(*a)).flatten().collect();

            
            //let mut all_bases = vec![];

            //for a in all_coded {
            //    all_bases.append(&mut Self::code_to_bases(*a));
            //}


            let new_s = start_id % 4;

            let ret: Vec<usize> = all_bases[new_s..(new_s+num_bases)].to_vec();

            ret

        }

        pub fn kmer_in_seq(&self, kmer: Vec<usize>) -> bool {

            /* self.block_lens.iter().map(|b| (0..*b).map(|i| self.return_bases(*b, i, kmer.len()).iter().
                                                      zip(&kmer).map(|(p, q)| p == q).fold(true, |r, s| r && s)).
                                                      fold(false, |d, e| d || e)).fold(false, |d, e| d || e) */

            let kmer_coll = self.generate_kmers(kmer.len());
            println!("size of kmer collection {}", size_of_val(&*kmer_coll));
            kmer_coll.contains(&kmer)



        }

    
    }


}

#[cfg(test)]
mod tests {
    use crate::sequence::seq::Sequence;
    use std::time::{Duration, Instant};
    use rand::Rng;
    use rand::distributions::{Distribution, Uniform};


    #[test]
    fn specific_seq(){

        let blocked = vec![vec![3,0,3,0,3,3,3,0,0,0,0,3], vec![2,2,1,1,1,1,1,1], vec![3,0,3,0,3,0,3,0,3,0,3,0,3,0,3,0,3,3,3,0,0,0,1,2]];

        let mut press = Sequence::new(&blocked);

        let bases = [0,1,2,3];

        let rebases = Sequence::code_to_bases(Sequence::bases_to_code(&bases));

        println!("{:?}, {:?}", bases, rebases);

        let arr = press.return_bases(0, 7, 5);

        let supp = [0,0,0,0,3];

        assert!(arr.iter().zip(supp).map(|(&a, b)| a == b).fold(true, |acc, mk| acc && mk));

        let arr2 = press.return_bases(1, 1, 7);

        let supp2 = [2,1,1,1,1,1,1];



        assert!(arr2.iter().zip(supp2).map(|(&a, b)| a == b).fold(true, |acc, mk| acc && mk));

        let alt_block = vec![vec![2,2,2,2,2,2,2,2,2,2,2,2],vec![2,2,2,2,2,2,2,2]];

        let mut press2 = Sequence::new(&alt_block);

        let atemers = press2.generate_kmers(8);

        assert!(atemers.len() == 1);

        let sup2 = [2,2,2,2,2,2,2,2];

        assert!(atemers[0].iter().zip(sup2).map(|(&a, b)| a == b).fold(true, |acc, mk| acc && mk));


    }

    #[test]
    fn seq_check2(){

        let mut rng = fastrand::Rng::new();

        let block_n: usize = 300;
        let u8_per_block: usize = 250;
        let bp_per_block: usize = u8_per_block*4;
        let bp: usize = block_n*bp_per_block;
        let u8_count: usize = u8_per_block*block_n;

        let start_gen = Instant::now();
        let blocks: Vec<u8> = (0..u8_count).map(|_| rng.u8(..)).collect();
        //let preblocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.u8(..)).collect();
        //let blocks: Vec<u8> = preblocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>();
        let block_inds: Vec<usize> = (0..block_n).map(|a| a*u8_per_block).collect();
        let block_lens: Vec<usize> = (0..block_n).map(|_| u8_per_block).collect();
        let sequence: Sequence = Sequence::new_manual(blocks, block_inds, block_lens);

        let kmer_c = sequence.generate_kmers(20);

        let start = Instant::now();


        let in_it = sequence.kmer_in_seq(vec![1usize;20]);

        let duration = start_gen.elapsed();
        println!("Done search {} bp {:?}", bp, duration);
    }





}

