use motif_finder::sequence::Sequence;
use motif_finder::data_struct::{AllData, AllDataUse};
use motif_finder::base::{Base, Motif, MotifSet, SetTrace, InitializeSet};

use std::io::{Read};
use std::{fs};



fn main() {

    //let bincode_file = "/expanse/lustre/scratch/alimf/temp_project/motif_runs/NC_000913.2_GSM639836_TrpR_Trp_ln_ratio_25_data.bin";

    let bincode_file = "/Users/afarhat/Downloads/sequence(1)_GSM639836_TrpR_Trp_ln_ratio_25_data.bin";
    let bincode_string = bincode_file.to_owned();

    println!("db {bincode_file}");
    
    let mut bincode_file_handle = fs::File::open(bincode_file).expect("Binarray file MUST be valid!");

    let mut buffer: Vec<u8> = Vec::new();

    bincode_file_handle.read_to_end(&mut buffer);

    let mut rng = rand::thread_rng();
    
    let full_dat: AllData = bincode::deserialize(&buffer).expect("Binarray file MUST be a valid motif set!");

    let using: AllDataUse = AllDataUse::new(&full_dat).unwrap();
    println!("{} {}", using.size(), using.number_bp());
/*
    let cho_mot: Motif = Motif::raw_pwm(vec![
                                       Base::new([0.62, 21.89, 1.99, 10.81]),
                                       Base::new([10.63, 1.68, 9.72, 1.54]),
                                       Base::new([6.97, 1.72, 22.42, 2.74]),
                                       Base::new([9.63, 6.01, 0.49, 7.11]),
                                       Base::new([18.54, 0.41, 2.98, 11.12]),
                                       Base::new([1.23, 2.40, 8.99, 25.97]),
                                       Base::new([15.93, 0.21, 0.21, 45.59]),
                                       Base::new([3.68, 0.31, 3.12, 78.50]),
                                       Base::new([1.81, 4.67, 3.98, 40.42]),
                                       Base::new([1.43, 0.68, 0.68, 1.08]),
                                       Base::new([2.68, 11.90, 0.27, 30.47]),
                                       Base::new([2.51, 2.60, 13.85, 3.64]),
                                       Base::new([0.20, 70.41, 0.20, 8.61]),
                                       Base::new([0.28, 7.16, 5.15, 8.61]),
                                       Base::new([4.76, 3.40, 33.33, 2.61])], 10.0);

    let mut cho_set: MotifSet = MotifSet::rand_with_one(&using, &mut rng);

    let cho_like = cho_set.replace_motif(cho_mot, 0);

    println!("mot {:?}", cho_set.get_nth_motif(0).best_motif());
    println!(" reg close {:?}", using.data().seq().all_kmers_within_hamming(&cho_set.get_nth_motif(0).best_motif(), 3).iter().map(|&a| Sequence::u64_to_kmer(using.data().seq().idth_unique_kmer(cho_set.get_nth_motif(0).len(), a), cho_set.get_nth_motif(0).len())).collect::<Vec<_>>());

    println!("cho like {}", cho_like);
    let cho_trace = SetTrace::new_trace(1, bincode_string.clone(), InitializeSet::<rand::rngs::ThreadRng>::Set(cho_set), &using, None);

    cho_trace.save_trace("/expanse/lustre/scratch/alimf/temp_project/motif_runs/", "cho_set", 0);
*/
    //This is the actual regulonDB set. Its best motif, [A, G, T, A, A, C, T, T, A, C, A, A],
    //does not appear directly in the sequence
    //So I modified it: there are only two motifs that are within hamming distance 2 of it:
    //[A, G, T, *C*, A, *T*, T, T, A, C, A, A] and [A, G, T, A, A, C, T, T, A, C, *T*, *G*]
    //I chose the latter because the former requires changing around more decisive bases
    /*let reg_mot: Motif = Motif::raw_pwm(vec![
                                       Base::new([1.0, 0.5, 0.125, 0.125]),
                                       Base::new([1./7., 1./7., 1.0, 5./7.]),
                                       Base::new([0.375, 0.25, 0.125, 1.0]),
                                       Base::new([1.0, 1./7., 1./7., 5./7.]),
                                       Base::new([1.0, 0.1, 0.2, 0.1]),
                                       Base::new([1./11., 1.0, 1./11., 1./11.]),
                                       Base::new([0.1, 0.1, 0.2, 1.0]),
                                       Base::new([1./11., 1./11., 1./11., 1.0]),
                                       Base::new([1.0, 1./6., 1./3., 2./3.]),
                                       Base::new([0.25, 1.0, 0.125, 0.375]),
                                       Base::new([1.0, 1./7., 5./7., 1./7.]),
                                       Base::new([1.0, 1.0, 0.75, 0.75])], 10.0);
*/
    
    let reg_mot: Motif = Motif::raw_pwm(vec![
                                         Base::new_with_pseudo([0.700000, 0.300000, 0.000000, 0.000000], 10., 0.1),
                                         Base::new_with_pseudo([0.000000, 0.000000, 0.600000, 0.400000], 10., 0.1),
                                         Base::new_with_pseudo([0.200000, 0.100000, 0.000000, 0.700000], 10., 0.1),
                                         Base::new_with_pseudo([0.000000, 0.000000, 0.700000, 0.300000], 10., 0.1),
                                         Base::new_with_pseudo([0.600000, 0.000000, 0.000000, 0.400000], 10., 0.1),
                                         Base::new_with_pseudo([0.900000, 0.000000, 0.100000, 0.000000], 10., 0.1),
                                         Base::new_with_pseudo([0.000000, 1.000000, 0.000000, 0.000000], 10., 0.1),
                                         Base::new_with_pseudo([0.000000, 0.000000, 0.100000, 0.900000], 10., 0.1),
                                         Base::new_with_pseudo([0.800000, 0.000000, 0.200000, 0.000000], 10., 0.1),
                                         Base::new_with_pseudo([0.000000, 0.000000, 1.000000, 0.000000], 10., 0.1),
                                         Base::new_with_pseudo([0.000000, 0.000000, 0.000000, 1.000000], 10., 0.1),
                                         Base::new_with_pseudo([0.500000, 0.000000, 0.200000, 0.300000], 10., 0.1),
                                         Base::new_with_pseudo([0.100000, 0.700000, 0.000000, 0.200000], 10., 0.1),
                                         Base::new_with_pseudo([0.600000, 0.000000, 0.400000, 0.000000], 10., 0.1),
                                         Base::new_with_pseudo([0.300000, 0.300000, 0.200000, 0.200000], 10., 0.1),
                                     /*  Base::new([1.0, 0.5, 0.125, 0.125]),
                                       Base::new([1./7., 1./7., 1.0, 5./7.]),
                                       Base::new([0.375, 0.25, 0.125, 1.0]),
                                       Base::new([1.0, 1./7., 1./7., 5./7.]),
                                       Base::new([1.0, 0.1, 0.2, 0.1]),
                                       Base::new([1./11., 1.0, 1./11., 1./11.]),
                                       Base::new([0.1, 0.1, 0.2, 1.0]),
                                       Base::new([1./11., 1./11., 1./11., 1.0]),
                                       Base::new([1.0, 1./6., 1./3., 2./3.]),
                                       //Base::new([0.25, 1.0, 0.125, 0.375]),
                                       //Base::new([1./7., 1./7., 5./7., 1.0]),
                                       //Base::new([0.75, 0.99, 1.0, 0.75]) */
    ].into_iter().collect::<Result<Vec<_>, _>>().unwrap(), 10.0);



    let mut reg_set: MotifSet = MotifSet::rand_with_one(&using, &mut rng);

    let like = reg_set.replace_motif(reg_mot, 0);

    println!("mot {:?}", reg_set.get_nth_motif(0).best_motif());
    println!(" reg close {:?}", using.data().seq().all_kmers_within_hamming(&reg_set.get_nth_motif(0).best_motif(), 2).iter().map(|&a| Sequence::u64_to_kmer(using.data().seq().idth_unique_kmer(reg_set.get_nth_motif(0).len(), a), reg_set.get_nth_motif(0).len())).collect::<Vec<_>>());

    println!("reg like {}", like);

//    2818 |     pub fn new_trace<R: Rng + ?Sized>(capacity: usize, initial_condition: Option<MotifSet<'a>>, data_ref: &'a AllDataUse<'a>, mut thermo_beta: f64, sparse: Option<usize>, rng: &mut R) -> SetTrace<'a> {

    let reg_trace = SetTrace::new_trace(1, Some(reg_set), &using, 1.0, None, &mut rng);

//    let reg_trace = SetTrace::new_trace(1, bincode_string.clone(), InitializeSet::<rand::rngs::ThreadRng>::Set(reg_set), &using, None);
   
    //reg_trace.save_initial_state("/expanse/lustre/scratch/alimf/temp_project/motif_runs/", "reg_set_mod");
    //reg_trace.save_trace("/expanse/lustre/scratch/alimf/temp_project/motif_runs/", "reg_set_mod", 0);
    reg_trace.save_initial_state("/Users/afarhat/Downloads/", "reg_set_ln");
    reg_trace.save_trace("/Users/afarhat/Downloads/", "reg_set_ln", 0);
  /*  let swi_mot: Motif = Motif::raw_pwm(vec![
                                       Base::new([0.2, 0.1, 0.1, 1.0]),
                                       Base::new([0.1, 0.2, 0.1, 1.0]),
                                       Base::new([0.2, 0.1, 0.1, 1.0]),
                                       Base::new([1.0, 1./6., 1./6., 1./6.]),
                                       Base::new([1./6., 1.0, 1./6., 1./6.]),
                                       Base::new([1./6., 1./6., 1./6., 1.0]),
                                       Base::new([1.0, 0.75, 0.25, 0.25]),
                                       Base::new([1./6., 1./6., 1.0, 1./6.]),
                                       Base::new([0.5, 0.5, 0.25, 1.0]),
                                       Base::new([0.25, 0.25, 0.75, 1.0]),
                                       Base::new([1.0, 0.25, 0.25, 0.75]),
                                       Base::new([1.0, 0.2, 0.4, 0.2]),
                                       Base::new([0.75, 1.0, 0.25, 0.25]),
                                       Base::new([0.2, 0.2, 0.4, 1.0]),
                                       Base::new([1.0, 0.25, 0.75, 0.25]),
                                       Base::new([1./6., 1./6., 1.0, 1./6.]),
                                       Base::new([1./6., 1./6., 1./6., 1.0]),
                                       Base::new([1.0, 0.2, 0.4, 0.2]),
                                       Base::new([0.2, 1.0, 0.2, 0.4])], 10.0);


    let mut swi_set: MotifSet = MotifSet::rand_with_one(&using, &mut rng);

    let _ = swi_set.replace_motif(swi_mot, 0);


    let swi_trace = SetTrace::new_trace(1, bincode_string.clone(), InitializeSet::<rand::rngs::ThreadRng>::Set(swi_set), &using, None);


    swi_trace.save_trace("/expanse/lustre/scratch/alimf/temp_project/motif_runs/", "swi_set", 0);


*/


}

