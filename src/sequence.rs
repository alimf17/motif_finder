pub mod seq {

    use itertools::{all, Itertools};
    use std::collections::HashMap;

    pub struct Sequence {
        seq_blocks: Vec<u8>,
        block_inds: Vec<usize>,
        start_bases: Vec<usize>,
        block_lens: Vec<usize>,
    }

    impl Sequence {
        
        pub fn new(blocks: &Vec<Vec<usize>>) -> Sequence {

            let mut block_is: Vec<usize> = Vec::new();

            let mut s_bases: Vec<usize> = Vec::new();

            let mut block_ls: Vec<usize> = Vec::new();

            let mut seq_bls: Vec<u8> = Vec::new();

            let mut p_len: usize = 0;

            let mut b_len: usize = 0;

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


            }

            Sequence{
                seq_blocks: seq_bls,
                block_inds: block_is,
                start_bases: s_bases,
                block_lens: block_ls
            }



        }

        pub fn new_manual(seq_blocks: Vec<u8>, block_inds: Vec<usize>,start_bases: Vec<usize>, block_lens: Vec<usize>) -> Sequence {

            Sequence{
                seq_blocks: seq_blocks,
                block_inds: block_inds,
                start_bases: start_bases,
                block_lens: block_lens
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

        fn code_to_bases(coded: u8) -> Vec<usize> {

            const PLACE_VALS: [u8; 4] = [64, 16, 4, 1];

            let mut V: Vec<usize> = (PLACE_VALS.iter().map(|&a| ((coded/a) as usize)).collect::<Vec<usize>>());

            for i in 0..3 { //Remember, we don't want to touch the 0th element
                V[3-i] -= V[2-i]*4;
            }

            V

        }

        pub fn return_bases(&self, block_id: usize, start_id: usize, num_bases: usize) -> Vec<usize> {

            let start_dec = self.block_inds[block_id]+(start_id/4);
            let end_dec = self.block_inds[block_id]+((start_id+num_bases-1)/4)+1;

            let all_coded = &self.seq_blocks[start_dec..end_dec];

            
            let mut all_bases = vec![];

            for a in all_coded {
                all_bases.append(&mut Self::code_to_bases(*a));
            }

            let new_s = start_id % 4;

            let ret: Vec<usize> = all_bases[new_s..(new_s+num_bases)].to_vec();

            ret

        }
    
    }


}

#[cfg(test)]
mod tests {
    use crate::sequence::seq::Sequence;

    #[test]
    fn seq_check(){

        let blocked = vec![vec![3,0,3,0,3,3,3,0,0,0,0,3], vec![2,2,1,1,1,1,1,1], vec![3,0,3,0,3,0,3,0,3,0,3,0,3,0,3,0,3,3,3,0,0,0,1,2]];

        let press = Sequence::new(&blocked);

        let arr = press.return_bases(0, 7, 5);

        let supp = [0,0,0,0,3];

        assert!(arr.iter().zip(supp).map(|(&a, b)| a == b).fold(true, |acc, mk| acc && mk));

        let arr2 = press.return_bases(1, 1, 7);

        let supp2 = [2,1,1,1,1,1,1];



        assert!(arr2.iter().zip(supp2).map(|(&a, b)| a == b).fold(true, |acc, mk| acc && mk));

        let alt_block = vec![vec![2,2,2,2,2,2,2,2,2,2,2,2],vec![2,2,2,2,2,2,2,2]];

        let press2 = Sequence::new(&alt_block);

        let atemers = press2.generate_kmers(8);

        assert!(atemers.len() == 1);

        let sup2 = [2,2,2,2,2,2,2,2];

        assert!(atemers[0].iter().zip(sup2).map(|(&a, b)| a == b).fold(true, |acc, mk| acc && mk));


    }




}

