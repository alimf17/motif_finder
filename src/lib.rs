#[allow(unused_parens)]
mod base;
mod sequence;
mod waveform;

//pub use crate::bases::Base;


#[cfg(test)]
mod tests {
    use crate::base::bases::Base;
    use crate::base::bases::GBase;
    use crate::sequence::seq::Sequence;
    use log::warn;
    use crate::waveform::wave::Kernel;
    use crate::waveform::wave::Waveform;
    use crate::waveform::wave::Noise;

    #[test]
    fn it_works() {
        let base = 3;
        let try_base: Base = Base::rand_new();
        let b = try_base.make_best(base);
        assert_eq!(base, b.best_base());

        let bc: Base = b.rev();

        assert_eq!(bc.best_base(), 3-base);

        let tc: Base = try_base.rev();

        assert_eq!(tc.best_base(), 3-try_base.best_base());

        assert!(!(tc == try_base));
        assert!(b == b.clone());

        assert!(b == b.to_gbase().to_base());


        let td: Base = Base::new([0.1, 0.2, 0.4, 0.3]);

        assert!((td.rel_bind(1)-0.5_f64).abs() < 1e-6);
        assert!((td.rel_bind(2)-1_f64).abs() < 1e-6);

        let tg: GBase = GBase::new([0.82094531732, 0.41047265866, 0.17036154577], 2);

        assert!(tg.to_base() == td);

    }

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

    #[test]
    fn wave_check(){

        let sd = 5;
        let height = 2.0;
        let k = Kernel::new(sd as f64, height);

        let kern = k.get_curve();
        let kernb = &k*4.0;


        assert!(kern.len() == 6*sd+1);

        assert!(kern.iter().zip(kernb.get_curve()).map(|(&a,b)| ((b/a)-4.0).abs() < 1e-6).fold(true, |acc, mk| acc && mk));

        assert!((k.get_sd()-(sd as f64)).abs() < 1e-6);
    }

    #[test]
    fn real_wave_check(){
        let k = Kernel::new(5.0, 2.0);
        let mut signal = Waveform::new(vec![84, 68, 72], vec![0, 84, 152], 5);

        
        signal.place_peak(&k, 2, 20);






        //Waves are in the correct spot
        assert!((signal.raw_wave()[35]-2.0).abs() < 1e-6);

        signal.place_peak(&k, 1, 20);

        //Waves are in the correct spot
        assert!((signal.raw_wave()[21]-2.0).abs() < 1e-6);

        signal.place_peak(&k, 1, 2);

        //Waves are not contagious
        assert!(signal.raw_wave()[0..17].iter().fold(true, |acc, ch| acc && ((ch-0.0).abs() < 1e-6)));


        let base_w = &signal*0.4;


        let ar: Vec<f64> = vec![0.9, -0.1];

        let noi: Vec<f64> = signal.produce_noise(&base_w, &ar);


        let raw_resid = &signal-&base_w;

        let w = raw_resid.raw_wave();

        for i in 0..raw_resid.raw_wave().len(){


            let chopped = raw_resid.start_dats().iter().fold(false, |acc, ch| acc || ((i >= *ch) && (i < *ch+ar.len())));
 

            let block_id = raw_resid.start_dats().iter().enumerate().filter(|(_, &a)| a <= i).max_by_key(|(_, &value)| value).map(|(idx, _)| idx).unwrap();
            //This gives the index of the maximum start value that still doesn't exceed i, identifying its data block.

            let block_loc = i-raw_resid.start_dats()[block_id];

            if !chopped {

                let start_noi = raw_resid.start_dats()[block_id]-block_id*ar.len();


                
                let piece: Vec<_> = w[(i-ar.len())..i].iter().rev().collect();

                let sst = ar.iter().zip(piece.clone()).map(|(a, r)| a*r).sum::<f64>() as f64;
                
                assert!(((noi[start_noi+block_loc-ar.len()] as f64) - ((raw_resid.raw_wave()[i] as f64)-(sst) as f64)).abs() < 1e-6);

            }

        }










    }



    #[test]
    fn noise_check(){

        let n1 = Noise::new(vec![0.4, 0.4, 0.3, 0.2, -1.4], 0.25, 2.64);
        let n2 = Noise::new(vec![0.4, 0.4, 0.3, -0.2, 1.4], 0.25, 2.64);

        assert!(((&n1*&n2)+1.59).abs() < 1e-6);

        println!("{:?}", n1.resids());
        println!("{:?}", n1.rank());
        println!("{}", n1.ad_calc());

        println!("{:?}", n1.ad_grad());
    }

    #[test]
    #[should_panic(expected = "Residuals aren't the same length?!")]
    fn panic_noise() {
        let n1 = Noise::new(vec![0.4, 0.4, 0.3, 0.2, -1.4], 0.25, 2.64);
        let n2 = Noise::new(vec![0.4, 0.4, 0.3, -0.2], 0.25, 2.64);

        let a = &n1*&n2;
    }



}
