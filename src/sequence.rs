
use serde::{ser, Serialize,Serializer, Deserialize};
use serde::de::{
    self, DeserializeSeed, EnumAccess, IntoDeserializer, MapAccess, SeqAccess,
    VariantAccess, Visitor,
};

use std::mem::size_of_val;
use std::collections::HashMap;
use itertools::{all, Itertools};
use rand::Rng;

use crate::base::{BASE_L, MIN_BASE, MAX_BASE}; //I don't want to get caught in a loop of use statements

pub const BITS_PER_BP: usize = 2; //ceiling(log2(BASE_L)), but Rust doesn't like logarithms in constants without a rigarmole
                                  //Don't try to get cute and set this > 8.
pub const BP_PER_U8: usize = 8/BITS_PER_BP; 
const PLACE_VALS: [u8; BP_PER_U8] = [1, 4, 16, 64]; //NOTICE: this only works when BASE_L == 4. 
                                            //Because a u8 contains 8 bits of information (duh)
                                            //And it requires exactly two bits disabmiguate 4 things. 
const U8_BITMASK: u8 = (1_u8 << BITS_PER_BP) - 1; // 2.pow(BITS_PER_BP)-1, but Rust doesn't like exponentiation in constants either
pub const U64_BITMASK: u64 = (1_u64  << BITS_PER_BP)-1;

//I promise you, you only every want to use the constructors we have provided you
//You do NOT want to try to initialize this manually. 
//If you think you can:
//   1) No, you can't.
//   2) If you're lucky, your code will panic like a kid in a burning building. 
//   3) If you're UNlucky, your code will silently produce results that are wrong. 
//   4) I've developed this thing for 5 years over two different programming languages 
//   5) I still screw it up when I have to do it manually. 
//   6) Save yourself. 
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Sequence {
    seq_blocks: Vec<u8>,
    block_u8_starts: Vec<usize>,
    block_lens: Vec<usize>,
    max_len: usize,
    kmer_dict: HashMap<usize, Vec<u64>>,
    kmer_nums: HashMap<usize, usize>
}


/*Minor note: if your blocks total more bases than about 2 billion, you
              will likely want to run on a 64 bit system, for hardware limitation
              reasons. If you manage to reach those limitations on a 64 bit system, 
              why on EARTH are you shoving in tens of millions of 
              Paris japonica genome equivalents into this????
*/
impl Sequence {
    
    pub fn new(blocks: Vec<Vec<usize>>) -> Sequence {

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

            //SAFETY: We have unsafe code that relies on these invariants being upheld
            if block.len() % BP_PER_U8 != 0 {
                panic!("All blocks must have a number of base pairs divisible by {}.", BP_PER_U8);
            }

            if block.len() <= MAX_BASE {
                panic!("All blocks must be longer than your maximum possible motif size!");
            }

            //SAFETY: This is quite possibly the most important guarentee this code has of safety. 
            //If you're getting a panic here, don't try to bull around it. The entire 
            //likelihood calculation algorithm relies on this assertion being statically guarenteed in all
            //uses of Sequence
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

        let orig_dict: HashMap<usize, Vec<u64>> = HashMap::new();
        let orig_nums: HashMap<usize, usize> = HashMap::new();

        let mut seq = Sequence{
                    seq_blocks: seq_bls,
                    block_u8_starts: block_is,
                    block_lens: block_ls,
                    max_len: max_len,
                    kmer_dict: orig_dict, 
                    kmer_nums: orig_nums,
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
        let orig_dict: HashMap<usize, Vec<u64>> = HashMap::new();
        let orig_nums: HashMap<usize, usize> = HashMap::new();

        let mut seq = Sequence{
                    seq_blocks: seq_blocks,
                    block_u8_starts: block_u8_starts,
                    block_lens: block_lens,
                    max_len: max_len,
                    kmer_dict: orig_dict, 
                    kmer_nums: orig_nums,
        };

        seq.initialize_kmer_dicts();
        seq

    }

    fn initialize_kmer_dicts(&mut self) {

        let mut kmer_dict: HashMap<usize, Vec<u64>> = HashMap::with_capacity(MAX_BASE+1-MIN_BASE);

        let mut kmer_nums: HashMap<usize, usize> = HashMap::with_capacity(MAX_BASE+1-MIN_BASE);

        for k in MIN_BASE..MAX_BASE+1 {

            let kmer_arr = self.generate_kmers(k);
            kmer_nums.insert(k, kmer_arr.len());
            kmer_dict.insert(k, kmer_arr);
        }

        self.kmer_dict = kmer_dict;
        self.kmer_nums = kmer_nums;

    }
    
    //NOTE: Pretty much any time anybody can access this array without automatically using the 
    //      the correct len should be immediately considered incorrect, especially because it 
    //      it will NOT fail: it will just give the wrong answer SILENTLY.
    pub fn generate_kmers(&self, len: usize) -> Vec<u64> {

        
        let max_possible_lenmers:usize = (4_usize.pow(len as u32)).min(self.block_lens.iter().sum::<usize>()-self.block_lens.len()*len);

        let mut unel: Vec<u64> = Vec::with_capacity(max_possible_lenmers);
        
        for i in 0..self.block_lens.len() {
            
            if self.block_lens[i] >= len {
               
                for j in 0..(self.block_lens[i]-len+1){
                    unel.push(Self::kmer_to_u64(&self.return_bases(i, j, len)));
                }

            }

        }

        unel = unel.into_iter().unique().collect();

        unel.sort_unstable(); // we filtered uniquely, so unstable sorting doesn't matter
        
        unel.shrink_to_fit();

        unel
      
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
        self.kmer_nums[&k]
    }

    pub fn unique_kmers(&self, k: usize) -> Vec<u64> {

        self.kmer_dict[&k].clone()
    }

    pub fn idth_unique_kmer(&self, k: usize, id: usize) -> u64 {
        self.kmer_dict[&k][id]
    }


    pub fn bases_to_code(bases: &[usize ; BP_PER_U8]) -> u8 {

        bases.iter().zip(PLACE_VALS).map(|(&a, b)| a*b as usize).sum::<usize>() as u8
    
    }


    //SAFETY: THIS function is safe. But unsafe code relies on it always producing values < BASE_L
    pub fn code_to_bases(coded: u8) -> [usize ; BP_PER_U8] {

        let mut V: [usize; BP_PER_U8] = [0; BP_PER_U8];

        let mut reference: u8 = coded;

        for i in 0..BP_PER_U8 { 

            V[i] = (U8_BITMASK & reference) as usize;
            reference = reference >> 2; 

        }

        V

    }
    
    pub fn kmer_to_u64(bases: &Vec<usize>) -> u64 {

        /*
        let ex_pval: Vec<u64> = (0..bases.len()).map(|a| (2u64.pow((a*BITS_PER_BP) as u32)) as u64).collect();

        ex_pval.iter().zip(bases).map(|(a, &b)| a*(b as u64)).sum()*/

        let mut mot : u64 = 0;

        for i in 0..bases.len() {
            mot += (bases[i] as u64) << (i*BITS_PER_BP);
        }

        mot
    }


    //Precision: This has no way to know whether you have the right kmer length.
    //        Ensure you match the correct kmer length to this u64 or it will
    //        silently give you an incorrect result: if len is too long, then
    //        it will append your 0th base (A in DNA), which may not exist in
    //        in your sequence. If it's too short, it will truncate the kmer
    pub fn u64_to_kmer(coded_kmer: u64, len: usize) -> Vec<usize> {

        let mut V: Vec<usize> = vec![0; len];

        let mut reference: u64 = coded_kmer;

        for i in 0..len {

            V[i] = (U64_BITMASK & reference) as usize;
            reference = reference >> 2;
        }

        V

    }


    pub fn return_bases(&self, block_id: usize, start_id: usize, num_bases: usize) -> Vec<usize> {

        let start_dec = self.block_u8_starts[block_id]+(start_id/BP_PER_U8);
        let end_dec = self.block_u8_starts[block_id]+((start_id+num_bases-1)/BP_PER_U8)+1;

        let all_coded = &self.seq_blocks[start_dec..end_dec];

        let all_bases: Vec<_> = all_coded.iter().map(|a| Self::code_to_bases(*a)).flatten().collect();

        

        let new_s = start_id % BP_PER_U8;

        let ret: Vec<usize> = all_bases[new_s..(new_s+num_bases)].to_vec();

        ret

    }

    //We exploit the ordering of the u64 versions of kmer to binary search
    pub fn kmer_in_seq(&self, kmer: &Vec<usize>) -> bool {

        let look_for = Self::kmer_to_u64(kmer);

        let unique_kmer_ptr: *const u64 = self.kmer_dict[&kmer.len()].as_ptr();

        let mut found = false;
       
        let mut lowbound: isize = 0;
        let mut upbound: isize = (self.kmer_nums[&(kmer.len())]-1) as isize;
        let mut midcheck: isize = (upbound+lowbound)/2;
        

        //SAFETY: The initial definitions of lowbound, upbound, and midcheck,
        //        combined with how they're revised, ensures that all pointer
        //        dereferences are always in bound of the relevant kmer_dict array
        unsafe {


            found = found || (*unique_kmer_ptr.offset(lowbound) == look_for);
            found = found || (*unique_kmer_ptr.offset(upbound) == look_for);
            
            let mut terminate = found ||  (*unique_kmer_ptr.offset(lowbound) > look_for) || (*unique_kmer_ptr.offset(upbound) < look_for) ; 
            while !terminate {

                found = look_for == *unique_kmer_ptr.offset(midcheck);
                if !found {
                    if *unique_kmer_ptr.offset(midcheck) > look_for {
                        upbound = midcheck-1;
                    } else {
                        lowbound = midcheck+1;
                    }
                    midcheck = (upbound+lowbound)/2;
                    terminate = upbound < lowbound;

                } else {
                    terminate = true;
                }
            }
        
        } 
       
       /* 
        let mut lowbound: usize = 0;
        let mut upbound: usize = self.kmer_nums[&kmer.len()] as usize;
        let mut midcheck: usize = (upbound+lowbound)/2;

        unsafe{
            found = found || (*self.kmer_dict[&kmer.len()].get_unchecked(lowbound) == look_for);
            found = found || (*self.kmer_dict[&kmer.len()].get_unchecked(upbound-1) == look_for);

            let mut terminate = found;
            while !terminate {
                found =  (*self.kmer_dict[&kmer.len()].get_unchecked(midcheck) == look_for);
                if !found {
                    if(*self.kmer_dict[&kmer.len()].get_unchecked(midcheck) > look_for) {
                        upbound = midcheck;
                    } else {
                        lowbound = midcheck;
                    }
                    midcheck = (upbound+lowbound)/2;
                    terminate = (midcheck == lowbound);

                } else {
                    terminate = true;
                }
            }
        }
        */

        found
    }

    fn u64_kmers_within_hamming(kmer_a: u64, kmer_b: u64, threshold: usize) -> bool {

        let mut check: u64 =  kmer_a ^ kmer_b; //^ is the XOR operator in rust

        let mut hamming: usize = 0;
        
        while (check > 0) && (hamming <= threshold) { //This is guaranteed to terminate in k operations or sooner, with my specific kmer impelementation
            hamming += ((check & U64_BITMASK) > 0) as usize;
            check = check >> 2;
        }

        hamming <= threshold
    }

    //This gives the id of the kmers in the HashMap vector that are under the hamming distance threshold
    pub fn all_kmers_within_hamming(&self, kmer: &Vec<usize>, threshold: usize) -> Vec<usize> {

        let u64_kmer: u64 = Self::kmer_to_u64(kmer);

        let len: usize = kmer.len();

        self.kmer_dict[&len].iter().enumerate()
                           .filter(|(_, &b)| Self::u64_kmers_within_hamming(u64_kmer, b, threshold))
                           .map(|(a, _)| a).collect::<Vec<usize>>()                    

    }


    pub fn random_valid_motif(&self, len: usize) -> Vec<usize>{

        let mut rng = rand::thread_rng();

        let mot_id: usize = rng.gen_range(0..(self.kmer_nums[&len]));

        Self::u64_to_kmer(self.kmer_dict[&len][mot_id], len)
    }

}



#[cfg(test)]
mod tests {
    use super::*; 
    use std::time::{Duration, Instant};
    use rand::Rng;
    use rand::distributions::{Distribution, Uniform};

    use crate::base::*;
   


    #[test]
    fn specific_seq(){

        let blocked = vec![vec![3,0,3,0,3,3,3,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0], vec![2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], vec![3,0,3,0,3,0,3,0,3,0,3,0,3,0,3,0,3,3,3,0,0,0,1,2]];

        let mut press = Sequence::new(blocked);

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

        let press2 = Sequence::new(alt_block);

        let atemers = press2.generate_kmers(8);

        assert!(atemers.len() == 1);

        let sup2 = Sequence::kmer_to_u64(&vec![2,2,2,2,2,2,2,2]);

        assert!(atemers[0] == sup2);
        println!("sup2 {:?}", atemers);
        assert!(atemers[0] == 43690u64);

        assert!(atemers[0] == Sequence::kmer_to_u64(&Sequence::u64_to_kmer(atemers[0], 8)));



        println!("{} {} kmer test", press2.kmer_in_seq(&vec![2,2,2,2,2,2,2,2, 2, 2]), press2.kmer_in_seq(&vec![2,1,2,2,2,2,2,2, 2, 2]));

        println!(" {:?}", press2.unique_kmers(10).iter().map(|&a| Sequence::u64_to_kmer(a, 10)).collect::<Vec<_>>() );
        println!(" {:?}", press2.unique_kmers(10));
        assert!(press2.kmer_in_seq(&vec![2,2,2,2,2,2,2,2, 2, 2]));
        assert!(!press2.kmer_in_seq(&vec![2,1,2,2,2,2,2,2, 2, 2]));


    }

    #[test]
    fn seq_check2(){

        let mut rng = fastrand::Rng::new();

        let block_n: usize = 300;
        let u8_per_block: usize = 2500;
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


        let in_it = sequence.kmer_in_seq(&vec![1usize;20]);

        let duration = start.elapsed();
        println!("Done search {} bp {:?}", bp, duration);

        let thresh: usize = 3;

        for b_l in MIN_BASE..=MAX_BASE {

            let mot = sequence.return_bases(0, 0, b_l);
            let mot_u64 = Sequence::kmer_to_u64(&mot);
            let legal_mot_ids = sequence.all_kmers_within_hamming(&mot, thresh);

            let mut all_within_hamming = true;
            let mut all_exist_in_seq = true;
            for &valid in &legal_mot_ids {
                let val_u64 = sequence.idth_unique_kmer(b_l,valid);
                all_within_hamming &= Sequence::u64_kmers_within_hamming(mot_u64, val_u64, thresh);
                all_exist_in_seq &= sequence.kmer_in_seq(&(Sequence::u64_to_kmer(val_u64, b_l)));
            }

            println!("For kmer size {}, all generated results are within hamming distance ({}) and exist in seq ({})", b_l, all_within_hamming, all_exist_in_seq);

            assert!(all_within_hamming && all_exist_in_seq, "Generated kmer results have an invalidity");
       
            let examine = sequence.unique_kmers(b_l);
            let mut count_close = 0;
            for poss in examine {
                if Sequence::u64_kmers_within_hamming(mot_u64, poss, thresh) { count_close += 1; }
            }

            assert!(legal_mot_ids.len() == count_close, "Missing possible mots");

        }

    }





}

