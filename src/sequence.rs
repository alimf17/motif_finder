pub mod seq {

    use itertools::Itertools;
    use std::collections::HashMap;

    pub struct Sequence {
        seq_blocks: Vec<u8>,
        block_inds: Vec<usize>,
        end_bases: Vec<usize>,
        block_lens: Vec<usize>,
    }

    impl Sequence {
        
        pub fn new(blocks: &Vec<Vec<usize>>) -> Sequence {

            let mut block_is: Vec<usize> = Vec::new();

            let mut e_bases: Vec<usize> = Vec::new();

            let mut block_ls: Vec<usize> = Vec::new();

            let mut seq_bls: Vec<u8> = Vec::new();

            let mut p_len: usize = 0;

            let mut b_len: usize = 0;

            for block in blocks {
                
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

                e_bases.push(b_len);
                block_ls.push(block.len());


            }

            Sequence{
                seq_blocks: seq_bls,
                block_inds: block_is,
                end_bases: e_bases,
                block_lens: block_ls
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



        fn bases_to_code(bases: &[usize ; 4]) -> u8 {

            const PLACE_VALS: [usize; 4] = [64, 16, 4, 1];
            bases.iter().zip(PLACE_VALS).map(|(&a, b)| a*b as usize).sum::<usize>() as u8
        
        }

        fn code_to_bases(coded: u8) -> [usize; 4] {

            const PLACE_VALS: [u8; 4] = [64, 16, 4, 1];

            let mut V: [usize; 4] = (PLACE_VALS.iter().map(|&a| ((coded/a) as usize)).collect::<Vec<usize>>()).try_into().expect("slice with incorrect length");

            for i in 0..3 { //Remember, we don't want to touch the 0th element
                V[3-i] -= V[2-i]*4;
            }

            V

        }

        pub fn return_bases(&self, block_id: usize, start_id: usize, num_bases: usize) -> Vec<usize> {

            let start_dec = self.block_inds[block_id]+(start_id/4);
            let end_dec = self.block_inds[block_id]+((start_id+num_bases-1)/4)+1;

            let all_coded = &self.seq_blocks[start_dec..end_dec];

            let all_bases_iter = all_coded.iter().map(|&a| Self::code_to_bases(a));

            let mut all_bases = vec![];

            for item in all_bases_iter {
                all_bases.extend(item.to_vec());
            }

            let new_s = start_id % 4;

            let ret: Vec<usize> = all_bases[new_s..(new_s+num_bases)].try_into().expect("slice with incorrect length");

            ret

        }
    
    }


}
