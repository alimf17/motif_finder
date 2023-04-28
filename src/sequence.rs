
use std::mem::size_of_val;
use std::collections::HashMap;
use itertools::{all, Itertools};
use rand::Rng;

use crate::base::{BASE_L, MIN_BASE, MAX_BASE}; //I don't want to get caught in a loop of use statements

//const PLACE_VALS: [u8; 4] = [64, 16, 4, 1];
pub const BP_PER_U8: usize = 8*2/BASE_L; //Number of bits per u8*2/number of bases to disambuguate
const PLACE_VALS: [u8; BP_PER_U8] = [1, 4, 16, 64]; //NOTICE: this only works when BASE_L == 4. 
                                            //Because a u8 contains 8 bits of information (duh)
                                            //And it requires exactly two bits disabmiguate 4 things. 


//I promise you, you only every want to use the constructors we have provided you
//You do NOT want to try to initialize this manually. 
//If you think you can:
//   1) No, you can't.
//   2) If you're lucky, your code will panic like a kid in a burning building. 
//   3) If you're UNlucky, your code will silently produce results that are wrong. 
//   4) I've developed this thing for 5 years over two different programming languages 
//   5) I still screw it up when I have to do it manually. 
//   6) Save yourself. 
pub struct Sequence {
    seq_blocks: Vec<u8>,
    block_u8_starts: Vec<usize>,
    block_lens: Vec<usize>,
    max_len: usize,
    kmer_dict: HashMap<usize, Vec<usize>>,
    kmer_nums: HashMap<usize, usize>
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

            if block.len() % BP_PER_U8 != 0 {
                panic!("All blocks must have a number of base pairs divisible by BP_PER_U8.");
            }

            if block.len() <= MAX_BASE {
                panic!("All blocks must be longer than your maximum possible motif size!");
            }

            if block.iter().any(|&a| a >= BASE_L) {
                panic!("All sequence bases must map to a valid base!")
            }

            let num_entries = block.len()/BP_PER_U8;
            
            for i in 0..num_entries {
                let precoded = &block[(BP_PER_U8*i)..(BP_PER_U8*i+BP_PER_U8)].try_into().expect("slice with incorrect length");
                seq_bls.push(Self::bases_to_code(precoded));
            }

            p_len += block.len()/BP_PER_U8;
            b_len += block.len();

            block_ls.push(block.len());

            max_len = if block.len() > max_len {block.len()} else {max_len};

        }

        let mut seq = Sequence{
                    seq_blocks: seq_bls,
                    block_u8_starts: block_is,
                    block_lens: block_ls,
                    max_len: max_len,
                    kmer_dict: HashMap<usize, Vec<usize>>::new(),
                    kmer_nums: HashMap<usize, usize>::new(),
                    };

        seq.initialize_kmer_dicts();
        seq

    }

    //Contract: block_lens.iter().sum() == seq_blocks.len(). block_lens[i] >= the max length of your motifs
    //          Failure to uphold this will result in this panicking for your own good.
    pub fn new_manual(seq_blocks: Vec<u8>, block_lens: Vec<usize>) -> Sequence {

        let mut block_u8_starts: Vec<usize> = vec![0];

        for i in 0..(block_lens.len()-1) {
            if block_lens[i] >= MAX_BASE && block_lens[i] % BP_PER_U8 == 0{
                block_u8_starts.push(block_u8_starts[i]+block_lens[i]/BP_PER_U8);
            } else {
                panic!("All blocks must be longer than your maximum possible motif size and divisible by {}!", BP_PER_U8);
            }
        }

        if block_lens.iter().map(|b| b/4).sum::<usize>() != seq_blocks.len() {
            panic!("stated block lengths do not equal the length of your data!");
        }

        let max_len: usize = *block_lens.iter().max().unwrap();

        let mut seq = Sequence{
                    seq_blocks: seq_blocks,
                    block_u8_starts: block_u8_starts,
                    block_lens: block_lens,
                    max_len: max_len,
                    kmer_dict: HashMap<usize, Vec<usize>>::new(),
                    kmer_nums: HashMap<usize, usize>::new(),
        };

        seq.initialize_kmer_dicts();
        seq

    }

    fn initialize_kmer_dicts(&mut self) {

        let mut kmer_dict: HashMap<usize, Vec<usize>> = HashMap::with_capacity(MAX_BASE+1-MIN_BASE);

        let mut kmer_amount: HashMap<usize, usize> = HashMap::with_capacity(MAX_BASE+1-MIN_BASE);

        for k in (MIN_BASE..MAX_BASE+1) as usize {

            let kmer_arr = self.generate_kmers(k);
            kmer_amount.insert(k, kmer_arr.len()/k);
            kmer_dict.insert(k, kmer_arr);
        }

        self.kmer_dict = kmer_dict;
        self.kmer_amount = kmer_amount;

    }
    
    //NOTE: Pretty much any time anybody can access this array without automatically using the 
    //      the correct len should be immediately considered unsafe, especially because it 
    //      it will NOT fail: it will just give the wrong answer SILENTLY.
    fn generate_kmers(&self, len: usize) -> Vec<usize> {

        
        let max_possible_lenmers:usize = (4_usize.pow(len)).min(self.block_lens.iter().sum()-self.block_lens.len()*len);

        let mut unel: Vec<Vec<usize>> = Vec::with_capacity(max_possible_Lmers);

        for i in 0..self.block_lens.len() {
            
            if self.block_lens[i] >= len {
               
                for j in 0..(self.block_lens[i]-len+1){
                    unel.push(self.return_bases(i, j, len));
                }

            }

        }

        unel = unel.into_iter().unique().collect();
        
        let unel_flat: Vec<usize> = unel.iter.flatten.collect();
        unel_flat.shrink_to_fit();
        unel_flat
      
    }



    //Regular reader functions
    pub fn seq_blocks(&self) -> &Vec<u8> {
        &self.seq_blocks
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

    pub fn block_u8_starts(&self) -> Vec<usize> {
        self.block_u8_starts.clone()
    }


    //Sequence k-mer reader functions
    //Note: these are very much designed to minimize how much self.kmer_dict
    //      is directly accessed. 

    //Contract: k is an element of [MIN_BASE, MAX_BASE]
    pub fn number_unique_kmers(&self, k: usize) -> usize {
        self.kmer_nums[k]
    }



    pub fn bases_to_code(bases: &[usize ; BP_PER_U8]) -> u8 {

        bases.iter().zip(PLACE_VALS).map(|(&a, b)| a*b as usize).sum::<usize>() as u8
    
    }

    pub fn code_to_bases(coded: u8) -> [usize ; BP_PER_U8] {

        let mut V: [usize; BP_PER_U8] = [0; BP_PER_U8];

        let mut reference: u8 = coded;

        for i in 0..BP_PER_U8 { 

            V[i] = (0b00000011 & reference) as usize;
            reference = reference >> 2; 

        }

        V

    }

    pub fn return_bases(&self, block_id: usize, start_id: usize, num_bases: usize) -> Vec<usize> {

        let start_dec = self.block_u8_starts[block_id]+(start_id/BP_PER_U8);
        let end_dec = self.block_u8_starts[block_id]+((start_id+num_bases-1)/BP_PER_U8)+1;

        let all_coded = &self.seq_blocks[start_dec..end_dec];

        let all_bases: Vec<_> = all_coded.iter().map(|a| Self::code_to_bases(*a)).flatten().collect();

        
        //let mut all_bases = vec![];

        //for a in all_coded {
        //    all_bases.append(&mut Self::code_to_bases(*a));
        //}


        let new_s = start_id % BP_PER_U8;

        let ret: Vec<usize> = all_bases[new_s..(new_s+num_bases)].to_vec();

        ret

    }

    pub fn kmer_in_seq(&self, kmer: Vec<usize>) -> bool {

        /* self.block_lens.iter().map(|b| (0..*b).map(|i| self.return_bases(*b, i, kmer.len()).iter().
                                                  zip(&kmer).map(|(p, q)| p == q).fold(true, |r, s| r && s)).
                                                  fold(false, |d, e| d || e)).fold(false, |d, e| d || e) */

        //let kmer_coll = self.generate_kmers(kmer.len());
        //println!("size of kmer collection {}", size_of_val(&*kmer_coll));
        //kmer_coll.contains(&kmer)


        let unique_kmer_ptr = self.kmer_dict[kmer.len()].as_ptr()

        for i in 0..kmer.len() {



        }


    }


}



#[cfg(test)]
mod tests {
    use super::*; 
    use std::time::{Duration, Instant};
    use rand::Rng;
    use rand::distributions::{Distribution, Uniform};


    #[test]
    fn specific_seq(){

        let blocked = vec![vec![3,0,3,0,3,3,3,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0], vec![2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], vec![3,0,3,0,3,0,3,0,3,0,3,0,3,0,3,0,3,3,3,0,0,0,1,2]];

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

        let alt_block = vec![vec![2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],vec![2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]];

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
        let bp_per_block: usize = u8_per_block*BP_PER_U8;
        let bp: usize = block_n*bp_per_block;
        let u8_count: usize = u8_per_block*block_n;

        let start_gen = Instant::now();
        let blocks: Vec<u8> = (0..u8_count).map(|_| rng.u8(..)).collect();
        //let preblocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.u8(..)).collect();
        //let blocks: Vec<u8> = preblocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>();
        let block_u8_starts: Vec<usize> = (0..block_n).map(|a| a*u8_per_block).collect();
        let block_lens: Vec<usize> = (0..block_n).map(|_| bp_per_block).collect();
        let sequence: Sequence = Sequence::new_manual(blocks, block_lens);

        let kmer_c = sequence.generate_kmers(20);

        let start = Instant::now();


        let in_it = sequence.kmer_in_seq(vec![1usize;20]);

        let duration = start_gen.elapsed();
        println!("Done search {} bp {:?}", bp, duration);
    }





}

