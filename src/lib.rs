
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

        println!("{}", b.dist(&b.to_gbase().to_base()));

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


        println!("DF");
        println!("{:?}", arr2);

        assert!(arr2.iter().zip(supp2).map(|(&a, b)| a == b).fold(true, |acc, mk| acc && mk));
    
        let alt_block = vec![vec![2,2,2,2,2,2,2,2,2,2,2,2],vec![2,2,2,2,2,2,2,2]];

        let press2 = Sequence::new(&alt_block);

        let atemers = press2.generate_kmers(8);

        assert!(atemers.len() == 1);

        let sup2 = [2,2,2,2,2,2,2,2];
 
        assert!(atemers[0].iter().zip(sup2).map(|(&a, b)| a == b).fold(true, |acc, mk| acc && mk));


        println!("{:?}", press.generate_kmers(8));
        println!("{:?}", press2.generate_kmers(8));
    }

    #[test]
    fn wave_check(){

        let sd = 30;
        let k = Kernel::new(sd as f64);
        println!("{:?}", k.get_curve());
        println!("{:?}", k.mul(4.0));

        let kern = k.get_curve();
        let kernb = k.mul(4.0);

        println!("{}", kern.len());

        assert!(kern.len() == 6*sd+1);

        assert!(kern.iter().zip(kernb).map(|(&a,b)| ((b/a)-4.0).abs() < 1e-6).fold(true, |acc, mk| acc && mk));

        assert!((k.get_sd()-(sd as f64)).abs() < 1e-6);

    }

}
