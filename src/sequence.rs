
use serde::{Serialize,Deserialize};

use std::collections::{HashMap, HashSet};
use itertools::{Itertools};
use rand::Rng;
use rand::prelude::SliceRandom;
use rand::distributions::{Distribution, Uniform};
use wyhash2::WyHash;

use crate::base::{BASE_L, MIN_BASE, MAX_BASE, Bp}; //I don't want to get caught in a loop of use statements

// This is the number of bits it takes to represent a Bp
// It should always be set to ceiling(log2(BASE_L))
// Don't try to get cute and set this > 8
pub(crate) const BITS_PER_BP: usize = 2;

/// This is the number of Bp a u8 can encode. 
pub const BP_PER_U8: usize = 8/BITS_PER_BP; 
const PLACE_VALS: [u8; BP_PER_U8] = [1, 4, 16, 64]; //NOTICE: this only works when BASE_L == 4. 
                                            //Because a u8 contains 8 bits of information (duh)
                                            //And it requires exactly two bits disabmiguate 4 things. 
const U8_BITMASK: u8 = 3; // 2.pow(BITS_PER_BP)-1, but Rust doesn't like exponentiation in constants either
pub(crate) const U64_BITMASK: u64 = 3;

//I promise you, you only every want to use the constructors we have provided you
//You do NOT want to try to initialize this manually. 
//If you think you can:
//   0) No, you can't.
//   1) If you're lucky, your code will panic like a kid in a burning building. 
//   2) If you're UNlucky, your code will silently produce results that are wrong. 
//   3) I've developed this thing for 5 years over two different programming languages 
//   4) I still screw it up when I have to do it manually. 
//   5) Save yourself. 
/// This is the struct which holds the sequence blocks with significant amounts of binding
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Sequence {
    seq_blocks: Vec<u8>,
    block_u8_starts: Vec<usize>,
    block_lens: Vec<usize>,
    max_len: usize,
    kmer_dict:  [Vec<u64>; MAX_BASE+1-MIN_BASE],
    kmer_id_dict: Vec<HashMap<u64, usize, WyHash>>,
    kmer_nums: [usize; MAX_BASE+1-MIN_BASE],
}


/*Minor note: if your blocks total more bases than about 2 billion, you
              will likely want to run on a 64 bit system, for hardware limitation
              reasons. If you manage to reach those limitations on a 64 bit system, 
              why on EARTH are you shoving in tens of millions of 
              Paris japonica genome equivalents into this????
*/
impl Sequence {
    

    /// Initializes a new Sequence from a vector of sequence blocks
    /// # Panics
    /// - If any of the vectors in `blocks` have a length  ` <= [MAX_BASE]`
    /// - If any of the vectors in `blocks` have a length not divisible by `[BP_PER_U8]`
    /// - If any element of any of the vectors in `blocks` is `>= BASE_L`
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
                //SAFETY: the check on the block having the right number of BPs guarentees that this is sound
                let precoded = &block[(BP_PER_U8*i)..(BP_PER_U8*i+BP_PER_U8)].iter().map(|&a| unsafe{Bp::usize_to_bp(a)}).collect::<Vec<_>>().try_into().expect("slice with incorrect length");
                seq_bls.push(Self::bases_to_code(precoded));
            }

            p_len += block.len()/BP_PER_U8;
            b_len += block.len();

            block_ls.push(block.len());

            max_len = if block.len() > max_len {block.len()} else {max_len};

        }

        const F: Vec<u64> = Vec::new();

        let orig_dict: [Vec<u64>; MAX_BASE+1-MIN_BASE] = [F; MAX_BASE+1-MIN_BASE];
        let orig_id_dict: Vec<HashMap<u64, usize, WyHash>> = Vec::new();
        let orig_nums = [0_usize; MAX_BASE+1-MIN_BASE];

        let mut seq = Sequence{
                    seq_blocks: seq_bls,
                    block_u8_starts: block_is,
                    block_lens: block_ls,
                    max_len: max_len,
                    kmer_dict: orig_dict, 
                    kmer_id_dict: orig_id_dict,
                    kmer_nums: orig_nums,
                    };

        seq.initialize_kmer_dicts();
        seq

    }

    /// Initializes a new Sequence from a single vector with base pair lengths 
    /// provides by `block_lens` 
    /// # Panics
    /// - If any element of `block_lens` is not divisible by `[BP_PER_U8]` 
    /// - If any element of `block_lens` is not greater than `[MAX_BASE]`
    /// - If the sum of `block_lens` is not equal to `[BP_PER_U8] * seq_blocks.len()`
    pub fn new_manual(seq_blocks: Vec<u8>, block_lens: Vec<usize>) -> Sequence {

        let mut block_u8_starts: Vec<usize> = vec![0];

        for i in 0..(block_lens.len()-1) {
            if block_lens[i] > MAX_BASE && block_lens[i] % BP_PER_U8 == 0{
                block_u8_starts.push(block_u8_starts[i]+block_lens[i]/BP_PER_U8);
            } else {
                panic!("All blocks must be longer than your maximum possible motif size and divisible by {}!", BP_PER_U8);
            }
        }

        if block_lens.iter().map(|b| b/BP_PER_U8).sum::<usize>() != seq_blocks.len() {
            panic!("stated block lengths do not equal the length of your data!");
        }

        const F: Vec<u64> = Vec::new();

        let max_len: usize = *block_lens.iter().max().unwrap();
        let orig_dict: [Vec<u64>; MAX_BASE+1-MIN_BASE] = [F; MAX_BASE+1-MIN_BASE];
        let orig_nums = [0_usize; MAX_BASE+1-MIN_BASE];
        let orig_id_dict: Vec<HashMap<u64, usize, WyHash>> = Vec::new();

        let mut seq = Sequence{
                    seq_blocks: seq_blocks,
                    block_u8_starts: block_u8_starts,
                    block_lens: block_lens,
                    max_len: max_len,
                    kmer_dict: orig_dict, 
                    kmer_id_dict: orig_id_dict,
                    kmer_nums: orig_nums,
        };

        seq.initialize_kmer_dicts();
        seq

    }

    // Makes a copy of `self`, with the blocks in `remove` removed
    // Any element of `remove` which is at least the size of the number of blocks
    // is simply ignored. Returns `None` if remove contains all legal blocks
    pub(crate) fn with_removed_blocks(&self, remove: &[usize]) -> Option<Self> {
        
        let mut new_blocks = self.seq_blocks.clone();
        let mut new_lens = self.block_lens.clone();
        let mut new_starts = self.block_u8_starts.clone();

        let mut remove_descend: Vec<usize> = remove.to_vec();

        // We always want to remove blocks in descending order
        // Otherwise, previously remove blocks screw with the remaining blocks
        remove_descend.sort_unstable();
        remove_descend.reverse();
        remove_descend.dedup();

        remove_descend = remove_descend.into_iter().filter(|a| *a >= self.block_u8_starts.len()).collect();

        //This is only possible if we have all blocks listed in remove_descend,
        //thanks to sorting and dedup().
        if remove_descend.len() == self.block_lens.len() { return None;}

        let mut i = 0;

        //Ok, this is a slightly weird thing we're doing
        //Basically, we're treating the last blocks specially, because they 
        //don't have a boundary to stop at
        //BUT, we have to iteratitively check new_starts because
        //each following block to remove could also end up needing to be treated
        //like an end block, until we know we have at least one block in front of
        //block i. The first guard is a guard against a panic, in case we completely
        //drain remove_descend doing this
        while (i < remove_descend.len()) && (remove_descend[i] == new_starts.len()-1) {

            let ind = remove_descend[i];
            _ = new_blocks.drain(new_starts[ind]..).collect::<Vec<_>>();
            _ = new_starts.pop();
            _ = new_lens.pop();
            i += 1;
        }

        while (i < remove_descend.len()) {
            let ind = remove_descend[i];
            let (start, stop) = (new_starts[ind], new_starts[ind+1]);
            let len_u8 = stop-start;
            _ = new_blocks.drain(start..stop).collect::<Vec<_>>();
            _ = new_starts[(ind+1)..].iter_mut().map(|a| *a -= len_u8).collect::<Vec<_>>();
            _ = new_starts.remove(ind);
            _ = new_lens.remove(ind);
            i += 1;
        }

        //We already know we shouldn't return None, but ? is easy
        let max_len = *new_lens.iter().max()?;

        const F: Vec<u64> = Vec::new();
        let orig_dict: [Vec<u64>; MAX_BASE+1-MIN_BASE] = [F; MAX_BASE+1-MIN_BASE];
        let orig_id_dict: Vec<HashMap<u64, usize, WyHash>> = Vec::new();
        let orig_nums = [0_usize; MAX_BASE+1-MIN_BASE];
        
        let mut seq = Sequence{
            seq_blocks: new_blocks,
            block_u8_starts: new_starts,
            block_lens: new_lens,
            max_len: max_len,
            kmer_dict: orig_dict,
            kmer_id_dict: orig_id_dict,
            kmer_nums: orig_nums,
        };

        seq.initialize_kmer_dicts();
        Some(seq)


    }

    fn initialize_kmer_dicts(&mut self) {


        let mut kmer_dict: [Vec<u64>; MAX_BASE+1-MIN_BASE] = core::array::from_fn(|_a| {
            Vec::new()
        });

        let mut kmer_id_dict: Vec<HashMap<u64, usize, WyHash>> = Vec::with_capacity(MAX_BASE+1-MIN_BASE);

        let mut kmer_nums = [0_usize; MAX_BASE+1-MIN_BASE]; 

        for k in MIN_BASE..MAX_BASE+1 {

            let kmer_arr = self.generate_kmers(k);
            let hasher = WyHash::with_seed(0);
            let mut minimap: HashMap<u64, usize, _> = HashMap::with_capacity_and_hasher(kmer_arr.len(), hasher);
            let _ = kmer_arr.iter().enumerate().map(|(a, &b)| minimap.insert(b, a)).collect::<Vec<_>>();
            kmer_id_dict.push(minimap);
            kmer_nums[k-MIN_BASE] =  kmer_arr.len();
            kmer_dict[k-MIN_BASE] =  kmer_arr;
        }

        self.kmer_dict = kmer_dict;
        self.kmer_id_dict = kmer_id_dict;
        self.kmer_nums = kmer_nums;

    }

    fn generate_kmers(&self, len: usize) -> Vec<u64> {



        let mut unel: Vec<u64> = Vec::new();

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

    /// # Panics
    ///   On debug mode, if there are more base pairs represented than `usize::MAX/BP_PER_U8`
    ///   On release mode, this will silently fail. 
    ///   On a 32 bit platform, this can be a concern if your genome is much
    ///   larger than a human genome, without omissions. On a 64 bit platform, 
    ///   there is no known genome which should cause this to panic: _Paris japonica`
    ///   has a genome size that is 100 million times smaller than usize::MAX
    pub fn number_bp(&self) -> usize {
        self.seq_blocks.len() << 2 //Each u8 contains four bp, so we multiply by 4
    }

    pub fn seq_blocks(&self) -> &Vec<u8> {
        &self.seq_blocks
    }

    pub fn coded_place(&self, i: usize) -> u8 {
        self.seq_blocks[i]
    }

    pub fn block_lens(&self) -> Vec<usize> {
        self.block_lens.clone()
    }

    pub fn ith_block_len(&self, i: usize) -> usize {
        self.block_lens[i]
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

    /// #Panics
    /// If `k` is not an element of `MIN_BASE..=MAX_BASE`
    pub fn number_unique_kmers(&self, k: usize) -> usize {
        self.kmer_nums[k-MIN_BASE]
    }

    /// #Panics
    /// If `k` is not an element of `MIN_BASE..=MAX_BASE`
    pub fn unique_kmers(&self, k: usize) -> Vec<u64> {

        self.kmer_dict[k-MIN_BASE].clone()
    }

    pub(crate) fn idth_unique_kmer(&self, k: usize, id: usize) -> u64 {
        self.kmer_dict[k-MIN_BASE][id]
    }

    pub(crate) fn idth_unique_kmer_vec(&self, k: usize, id: usize) -> Vec<Bp> {
        Self::u64_to_kmer(self.idth_unique_kmer(k, id), k)
    }

    pub(crate) fn id_of_u64_kmer(&self, k: usize, kmer: u64) -> Option<usize> {
        self.kmer_id_dict[k-MIN_BASE].get(&kmer).copied()
    }

    //Panics: if kmer is not a valid u64 version of a k-mer represented in the sequence
    pub(crate) fn id_of_u64_kmer_or_die(&self, k: usize, kmer: u64) -> usize {
        self.kmer_id_dict[k-MIN_BASE][&kmer]
    }



    fn bases_to_code(bases: &[Bp ; BP_PER_U8]) -> u8 {

        bases.iter().zip(PLACE_VALS).map(|(&a, b)| (a as usize)*(b as usize)).sum::<usize>() as u8

    }


    pub(crate) fn code_to_bases(coded: u8) -> [Bp ; BP_PER_U8] {

        let mut v: [Bp; BP_PER_U8] = [Bp::A; BP_PER_U8];

        let mut reference: u8 = coded;

        for i in 0..BP_PER_U8 { 

            //SAFETY: U8_BITMASK always chops the last two bits from the u8, leaving a usize < BASE_L (4)
            // BIG HUGE ENORMOUS SAFETY NOTE WARNING: IF YOU AT ANY POINT CHANGE THE NUMBER OF BPS THAT WE HAVE,
            // THIS IS THE FUNCTION AND THE PLACE WHERE YOU WILL GET UB IF YOU ARE NOT EXTREMELY CAREFUL
            v[i] = unsafe{Bp::usize_to_bp((U8_BITMASK & reference) as usize)};
            reference = reference >> 2; 

        }

        v

    }



    pub(crate) fn kmer_to_u64(bases: &[Bp]) -> u64 {

        let mut mot : u64 = 0;

        for i in 0..bases.len() {
            mot += (bases[i] as usize as u64) << (i*BITS_PER_BP);
        }

        mot
    }


    //Precision: This has no way to know whether you have the right kmer length.
    //        Ensure you match the correct kmer length to this u64 or it will
    //        silently give you an incorrect result: if len is too long, then
    //        it will append your 0th base (A in DNA), which may not exist in
    //        in your sequence. If it's too short, it will truncate the kmer
    pub(crate) fn u64_to_kmer(coded_kmer: u64, len: usize) -> Vec<Bp> {

        let mut v: Vec<Bp> = vec![Bp::A; len];

        let mut reference: u64 = coded_kmer;

        for i in 0..len {

            //SAFETY: U8_BITMASK always chops the last two bits from the u8, leaving a usize < BASE_L (4)
            v[i] = unsafe{Bp::usize_to_bp((U64_BITMASK & reference) as usize)} ;
            reference = reference >> 2;
        }

        v

    }

    // This is a helper function for motif contraction
    // It counts the number of kmers that `self` has that differ only in their last bp from `bases`
    pub(crate) fn number_kmers_neighboring_by_last_bp(&self, bases: &[Bp]) -> f64 {

        let k = bases.len();

        let start_offset = (k-1)*2;

        let add_offset: u64 = 1 << start_offset;

        let mask_offset: u64 = (1 << (start_offset+2)) - 1;

        let mut check_u64 = Self::kmer_to_u64(bases);

        //There is always at leasst one kmer that neighbors `bases` in its last Bp: itself
        let mut num: f64 = 1.0;

        for _ in 0..(BASE_L-1) {

            // This effectively increments the last bp in bases,
            // going from A -> C -> G -> T
            // We only need to do this BASE_L-1 times because doing it 
            // BASE_L times would loop us back to start.
            // mask_offset basically is my way of correcting for potential carrying
            check_u64 = (check_u64+add_offset) & mask_offset;

            if self.id_of_u64_kmer(k, check_u64).is_some() {
                num += 1.0;
            }

        }

        num

    }

    /// This returns the `num_bases` `[Bp]`s in the `block_id`th block, starting
    /// from index `start_id`. 
    /// # Panics
    /// If `block_id` is not less than the number of sequence blocks, 
    /// or `start_id+num_bases` is more than the number of bases in the `block_id`th sequence block 
    pub fn return_bases(&self, block_id: usize, start_id: usize, num_bases: usize) -> Vec<Bp> {

        let start_dec = self.block_u8_starts[block_id]+(start_id/BP_PER_U8);
        let end_dec = self.block_u8_starts[block_id]+((start_id+num_bases-1)/BP_PER_U8)+1;

        assert!(end_dec-start_dec <= self.block_lens[block_id]/BP_PER_U8, "Number of bases requested overflows the block");

        let all_coded = &self.seq_blocks[start_dec..end_dec];

        let all_bases: Vec<_> = all_coded.iter().map(|a| Self::code_to_bases(*a)).flatten().collect();



        let new_s = start_id % BP_PER_U8;

        let ret: Vec<Bp> = all_bases[new_s..(new_s+num_bases)].to_vec();

        ret

    }



    //We exploit the ordering of the u64 versions of kmer to binary search
    pub fn kmer_in_seq(&self, kmer: &Vec<Bp>) -> bool {

        let look_for = Self::kmer_to_u64(kmer);

        self.id_of_u64_kmer(kmer.len(), look_for).is_some()

    }

    fn u64_kmers_have_hamming(kmer_a: u64, kmer_b: u64, distance: usize) -> bool {

        let mut check: u64 =  kmer_a ^ kmer_b;

        let mut hamming: usize = 0;

        while (check > 0) && (hamming <= distance) { //This is guaranteed to terminate in k operations or sooner, with my specific kmer impelementation
            hamming += ((check & U64_BITMASK) > 0) as usize;
            check = check >> 2;
        }

        hamming == distance

    }

    pub(crate) fn all_kmers_start_minmer(&self, minmer: u64, len: usize) -> Vec<usize> {

        const MINMER_MASK: u64 = (1_u64 << ((MIN_BASE * 2) as u64)) - 1;

        self.kmer_dict[len-MIN_BASE].iter().enumerate()
            .filter(|(_, &b)| ((b & MINMER_MASK) ^ minmer) == 0 ).map(|(a, _)| a).collect::<Vec<usize>>()

    }

    //Returns the kmer id in position 0 if possible and the number of possible kmers in position 1
    //NOTE: Yes, it is possible to have a minmer that does not have a valid extension to a kmer: 
    //      Imagine a minmer that only appears once near the end of a sequence block
    pub(crate) fn rand_kmer_start_minmer_and_number<R: Rng + ?Sized>(&self, minmer: u64, len: usize, rng: &mut R) -> (Option<usize>, usize) {
        let samp = self.all_kmers_start_minmer(minmer, len);
        (samp.choose(rng).copied(), samp.len())
    }


    pub(crate) fn kmer_id_minmer_vec(&self, kmer_ids: &[usize], len: usize) -> Vec<u64> {
        const MINMER_MASK: u64 = (1_u64 << ((MIN_BASE * 2) as u64)) - 1;

        kmer_ids.iter().map(|&a| self.kmer_dict[len-MIN_BASE][a] & MINMER_MASK).collect()
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
    pub(crate) fn all_kmers_within_hamming(&self, kmer: &Vec<Bp>, threshold: usize) -> Vec<usize> {

        let u64_kmer: u64 = Self::kmer_to_u64(kmer);

        let len: usize = kmer.len();

        self.kmer_dict[len-MIN_BASE].iter().enumerate()
            .filter(|(_, &b)| Self::u64_kmers_within_hamming(u64_kmer, b, threshold))
            .map(|(a, _)| a).collect::<Vec<usize>>()                    

    }


    pub fn random_valid_motif<R: Rng + ?Sized>(&self, len: usize, rng: &mut R) -> Vec<Bp>{

        let mot_id: usize = rng.gen_range(0..(self.kmer_nums[len-MIN_BASE]));
        Self::u64_to_kmer(self.kmer_dict[len-MIN_BASE][mot_id], len)

    }

    pub fn n_random_valid_motifs<R: Rng + ?Sized>(&self, len: usize, n: usize, rng: &mut R) -> Vec<usize>{

        let sample_range = Uniform::from(0_usize..(self.kmer_nums[len-MIN_BASE]));

        let mut id_set: HashSet<usize> = HashSet::with_capacity(n);

        let mut tries = n;

        while tries > 0 {
            rng.sample_iter(&sample_range).take(tries).for_each(|a| {id_set.insert(a);} );
            tries = n - id_set.len();
        }

        id_set.into_iter().collect()
    }
}
/// This is the struct which holds the sequence blocks with no notable binding
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct NullSequence {
    seq_blocks: Vec<u8>,
    block_u8_starts: Vec<usize>,
    block_lens: Vec<usize>,
    max_len: usize,
    kmer_counts: [HashMap<u64, usize, WyHash>; MAX_BASE+1-MIN_BASE],
    kmer_lists: [Vec<u64>; MAX_BASE+1-MIN_BASE],
}


impl NullSequence {


    /// Initializes a new NullSequence from a vector of sequence blocks
    /// # Panics
    /// - If any of the vectors in `blocks` have a length  ` <= [MAX_BASE]`
    /// - If any of the vectors in `blocks` have a length not divisible by `[BP_PER_U8]`
    /// - If any element of any of the vectors in `blocks` is `>= BASE_L`
    pub fn new(blocks: Vec<Vec<usize>>) -> NullSequence {

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
                //SAFETY: the check on the block having the right number of BPs guarentees that this is sound
                let precoded = &block[(BP_PER_U8*i)..(BP_PER_U8*i+BP_PER_U8)].iter().map(|&a| unsafe{Bp::usize_to_bp(a)}).collect::<Vec<_>>().try_into().expect("slice with incorrect length");
                seq_bls.push(Sequence::bases_to_code(precoded));
            }

            p_len += block.len()/BP_PER_U8;
            b_len += block.len();

            block_ls.push(block.len());

            max_len = if block.len() > max_len {block.len()} else {max_len};

        }


        let mut seq = NullSequence {
            seq_blocks: seq_bls,
            block_u8_starts: block_is,
            block_lens: block_ls,
            max_len: max_len,
            kmer_counts: core::array::from_fn(|a| HashMap::with_hasher(WyHash::with_seed(0))),
            kmer_lists: core::array::from_fn(|a| Vec::new()),

        };

        seq.initialize_kmer_count();

        seq

    }

    /// Initializes a new Sequence from a single vector with base pair lengths 
    /// provides by `block_lens` 
    /// # Panics
    /// - If any element of `block_lens` is not divisible by `[BP_PER_U8]` 
    /// - If any element of `block_lens` is not greater than `[MAX_BASE]`
    /// - If the sum of `block_lens` is not equal to `[BP_PER_U8] * seq_blocks.len()`
    pub fn new_manual(seq_blocks: Vec<u8>, block_lens: Vec<usize>) -> NullSequence {

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


        let mut seq = NullSequence{
            seq_blocks: seq_blocks,
            block_u8_starts: block_u8_starts,
            block_lens: block_lens,
            max_len: max_len,
            kmer_counts: core::array::from_fn(|a| HashMap::with_hasher(WyHash::with_seed(0))),
            kmer_lists: core::array::from_fn(|a| Vec::new()),
        };

        seq.initialize_kmer_count();
        seq

    }

    fn initialize_kmer_count(&mut self) {

        let kmer_counts: [HashMap<u64, usize, WyHash>; MAX_BASE+1-MIN_BASE] = core::array::from_fn(|a| self.generate_kmer_counts(a+MIN_BASE));
        self.kmer_counts = kmer_counts;
        let kmer_lists: [Vec<u64>; MAX_BASE+1-MIN_BASE] = core::array::from_fn(|a| {
            let mut kmers: Vec<u64> = self.kmer_counts[a].keys().map(|&b| b).collect();
            kmers.sort_unstable();
            kmers
        });
        self.kmer_lists = kmer_lists;

    }

    //NOTE: Pretty much any time anybody can access this array without automatically using the 
    //      the correct len should be immediately considered incorrect, especially because it 
    //      it will NOT fail: it will just give the wrong answer SILENTLY.
    fn generate_kmer_counts(&self, len: usize) -> HashMap<u64,usize, WyHash> {

        let mut unel: Vec<u64> = Vec::new();

        let guess_capacity = 2_usize.pow(len as u32).min(self.block_lens.iter().sum::<usize>());

        let hasher = WyHash::with_seed(0);
        let mut lenmer_counts: HashMap<u64,usize, _> = HashMap::with_capacity_and_hasher(guess_capacity, hasher);

        for i in 0..self.block_lens.len() {
            if self.block_lens[i] >= len {
                for j in 0..(self.block_lens[i]-len){
                    let kmer_u64 = Sequence::kmer_to_u64(&self.return_bases(i, j, len));
                    let rev_u64 = reverse_complement_u64_kmer(kmer_u64, len);
                    lenmer_counts.entry(kmer_u64).and_modify(|count| *count += 1).or_insert(1);
                    lenmer_counts.entry(rev_u64).and_modify(|count| *count += 1).or_insert(1);
                }
            }

        }

        lenmer_counts


    }


    //Regular reader functions

    /// # Panics
    ///   On debug mode, if there are more base pairs represented than `usize::MAX/BP_PER_U8`
    ///   On release mode, this will silently fail. 
    ///   On a 32 bit platform, this can be a concern if your genome is much
    ///   larger than a human genome, without omissions. On a 64 bit platform, 
    ///   there is no known genome which should cause this to panic: _Paris japonica`
    ///   has a genome size that is 100 million times smaller than usize::MAX
    pub fn number_bp(&self) -> usize {
        self.seq_blocks.len() << 2 //Each u8 contains four bp, so we multiply by 4
    }

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

    pub fn num_sequence_blocks(&self) -> usize {
        self.block_lens.len()
    }

    pub fn return_bases(&self, block_id: usize, start_id: usize, num_bases: usize) -> Vec<Bp> {

        let start_dec = self.block_u8_starts[block_id]+(start_id/BP_PER_U8);
        let end_dec = self.block_u8_starts[block_id]+((start_id+num_bases-1)/BP_PER_U8)+1;

        let all_coded = &self.seq_blocks[start_dec..end_dec];

        let all_bases: Vec<_> = all_coded.iter().map(|a| Sequence::code_to_bases(*a)).flatten().collect();



        let new_s = start_id % BP_PER_U8;

        let ret: Vec<Bp> = all_bases[new_s..(new_s+num_bases)].to_vec();

        ret

    }

    pub fn kmer_count(&self, kmer: u64, len: usize) -> Option<usize> {
        self.kmer_counts[len-MIN_BASE].get(&kmer).map(|a| *a) 
    }
}


pub(crate) fn reverse_complement_u64_kmer(kmer: u64, len: usize) -> u64 {

    let mut track_kmer = kmer;
    let mut rev_kmer: u64 = 0;
    for i in 0..len {
        rev_kmer += ((track_kmer & 3) ^ 3) << (2*(len-1-i));
        track_kmer = track_kmer >> 2;
    }

    rev_kmer

}

pub(crate) fn plain_hamming_u64_kmer(kmer_a: u64, kmer_b: u64, len: usize) -> usize {
    let mut kmer_check = (kmer_a ^ kmer_b);
    let mut hamming: usize = 0;
    for _ in 0..len {
        if (kmer_check & 3) > 0 { hamming += 1;}
        kmer_check = kmer_check >> 2
    }
    hamming
}

pub(crate) fn true_have_hamming_u64_kmer(kmer_a: u64, kmer_b: u64, len: usize, hamming_thresh: usize) -> bool {
    let rc = reverse_complement_u64_kmer(kmer_a, len);
    let hamming = plain_hamming_u64_kmer(kmer_a, kmer_b, len).min(plain_hamming_u64_kmer(rc, kmer_b, len));
    hamming == hamming_thresh
}
#[cfg(test)]
mod tests {
    use super::*; 
    use std::time::Instant;

    use crate::base::*;


    #[test]
    fn test_u64() {

        let mut rng = rand::thread_rng();

        for _ in 0..1000 {

            let len = rng.gen_range(MIN_BASE..=MAX_BASE);
            let motif: u64 = rng.gen_range(0_u64..4_u64.pow(len as u32));
            let rev_mot = reverse_complement_u64_kmer(motif, len);

            let mut mot_change = motif;

            for i in 0..len {
                let b0 = mot_change & 3;
                mot_change = mot_change >> 2;
                let b1 = (rev_mot & (3 << (2*(len-1-i)))) >> (2*(len-1-i));
                println!("{:#040b} {b0}, {:#040b} {b1} {len}", motif, rev_mot);
                if b0 ^ b1 != 3 {
                    panic!("failed complement!");
                }
            }

        }

    }


    #[test]
    fn specific_seq(){

        let blocked = vec![vec![3,0,3,0,3,3,3,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0], vec![2,2,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1], vec![3,0,3,0,3,0,3,0,3,0,3,0,3,0,3,0,3,3,3,0,0,0,1,2]];

        let press = Sequence::new(blocked);

        let bases = BP_ARRAY;

        let rebases = Sequence::code_to_bases(Sequence::bases_to_code(&bases));

        println!("{:?}, {:?}", bases, rebases);

        let arr = press.return_bases(0, 7, 5);

        let supp = [0,0,0,0,3];

        assert!(arr.iter().zip(supp).map(|(&a, b)| (a as usize) == b).fold(true, |acc, mk| acc && mk));

        let arr2 = press.return_bases(1, 1, 7);

        let supp2 = [2,1,1,1,1,1,1];



        assert!(arr2.iter().zip(supp2).map(|(&a, b)| (a as usize) == b).fold(true, |acc, mk| acc && mk));

        let alt_block = vec![vec![2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2],vec![2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2]];

        let press2 = Sequence::new(alt_block);

        let atemers = press2.generate_kmers(8);

        assert!(atemers.len() == 1);

        let sup2 = Sequence::kmer_to_u64(&vec![Bp::G; 8]);

        assert!(atemers[0] == sup2);
        println!("sup2 {:?}", atemers);
        assert!(atemers[0] == 43690u64);

        assert!(atemers[0] == Sequence::kmer_to_u64(&Sequence::u64_to_kmer(atemers[0], 8)));



        println!("{} {} kmer test", press2.kmer_in_seq(&vec![Bp::G,Bp::G,Bp::G,Bp::G,Bp::G,Bp::G,Bp::G,Bp::G, Bp::G, Bp::G]), press2.kmer_in_seq(&vec![Bp::G,Bp::C,Bp::G,Bp::G,Bp::G,Bp::G,Bp::G,Bp::G, Bp::G, Bp::G]));

        println!(" {:?}", press2.unique_kmers(10).iter().map(|&a| Sequence::u64_to_kmer(a, 10)).collect::<Vec<_>>() );
        println!(" {:?}", press2.unique_kmers(10));
        assert!(press2.kmer_in_seq(&vec![Bp::G,Bp::G,Bp::G,Bp::G,Bp::G,Bp::G,Bp::G,Bp::G, Bp::G, Bp::G]));
        assert!(!press2.kmer_in_seq(&vec![Bp::G,Bp::C,Bp::G,Bp::G,Bp::G,Bp::G,Bp::G,Bp::G, Bp::G, Bp::G]));


    }

    #[test]
    fn seq_check2(){

        let mut rng = fastrand::Rng::new();

        let block_n: usize = 300;
        let u8_per_block: usize = 2500;
        let bp_per_block: usize = u8_per_block*BP_PER_U8;
        let bp: usize = block_n*bp_per_block;
        let u8_count: usize = u8_per_block*block_n;

        let _start_gen = Instant::now();
        let blocks: Vec<u8> = (0..u8_count).map(|_| rng.u8(..)).collect();
        //let preblocks: Vec<u8> = (0..(u8_count/100)).map(|_| rng.u8(..)).collect();
        //let blocks: Vec<u8> = preblocks.iter().cloned().cycle().take(u8_count).collect::<Vec<_>>();
        let _block_u8_starts: Vec<usize> = (0..block_n).map(|a| a*u8_per_block).collect();
        let block_lens: Vec<usize> = (0..block_n).map(|_| bp_per_block).collect();
        let sequence: Sequence = Sequence::new_manual(blocks, block_lens);

        let _kmer_c = sequence.generate_kmers(20);

        let start = Instant::now();


        let in_it = sequence.kmer_in_seq(&vec![Bp::C;20]);

        let duration = start.elapsed();
        println!("Done search {} found {} bp {:?}", bp, in_it, duration);

        let thresh: usize = 3;

        let mut rng = rand::thread_rng();

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

            let neighbors = sequence.number_kmers_neighboring_by_last_bp(&mot);

            let mut neigh: f64 = 0.0;

            for bp in [Bp::A, Bp::C, Bp::G, Bp::T] {
                let mut other_mot = mot.clone();
                other_mot[mot.len()-1] = bp;
                if sequence.kmer_in_seq(&other_mot) { neigh += 1.0; }
            }

            //println!("{neighbors}, {neigh} neighbs");
            assert!(neighbors == neigh);

            for _ in 0..100 {

                let mot2 = sequence.random_valid_motif(b_l, &mut rng);

                let neighbors = sequence.number_kmers_neighboring_by_last_bp(&mot2);

                let mut neigh: f64 = 0.0;

                for bp in [Bp::A, Bp::C, Bp::G, Bp::T] {
                    let mut other_mot = mot2.clone();
                    other_mot[mot2.len()-1] = bp;
                    if sequence.kmer_in_seq(&other_mot) { neigh += 1.0; }
                }

                //println!("{neighbors}, {neigh} neighbs");
                assert!(neighbors == neigh);

            }
            assert!(all_within_hamming && all_exist_in_seq, "Generated kmer results have an invalidity");

            let examine = sequence.unique_kmers(b_l);

            assert!(examine.len() == sequence.number_unique_kmers(b_l));

            let mut count_close = 0;
            for poss in examine {
                if Sequence::u64_kmers_within_hamming(mot_u64, poss, thresh) { count_close += 1; }
            }

            assert!(legal_mot_ids.len() == count_close, "Missing possible mots");

        }


    }





}

