pub mod wave{
 
    use std::ops::Add;
    use std::ops::Sub;
    use std::ops::Mul;
    use std::cmp::max;
    use std::cmp::min;
    use core::f64::consts::PI;

    use statrs::distribution::StudentsT;
    use statrs::distribution::{Continuous, ContinuousCDF};


    const WIDE: f64 = 3.0;
    const THRESH: f64 = 1e-4; //Going off of binding score now, not peak height, so it's lower

    pub struct Kernel{

        peak_width: f64,
        kernel: Vec<f64>,
    }

    impl Mul<f64> for &Kernel {

        type Output = Kernel;

        fn mul(self, rhs: f64) -> Kernel {

            Kernel {
                peak_width: self.peak_width,
                kernel: self.kernel.iter().map(|a| a*rhs).collect(),
            }

        }

    }

    impl Kernel {
        
        pub fn new(peak_width: f64, peak_height: f64) -> Kernel {

            let span = (peak_width*WIDE) as isize;

            let domain: Vec<isize> = (-span..(span+1)).collect();

            let range = domain.iter().map(|a| (-((*a as f64).powf(2.0))/(2.0*peak_width.powf(2.0))).exp()*peak_height).collect();

            Kernel{
                peak_width: peak_width,
                kernel: range,
            }

        }

        pub fn get_sd(&self) -> f64 {
            self.peak_width
        }

        pub fn get_curve(&self) -> Vec<f64> {

            self.kernel.clone()
        
        }

        pub fn len(&self) -> usize {
            self.kernel.len()
        }
        


    }

    pub struct Waveform {
        wave: Vec<f64>,
        spacer: usize,
        point_lens: Vec<usize>,
        start_bases: Vec<usize>,
        start_dats: Vec<usize>,
        block_lens: Vec<usize>,
    }
    
    impl Waveform {

        pub fn new(block_lens: Vec<usize>, start_bases: Vec<usize>, spacer: usize) -> Waveform {


            /*We require that all data begin with the 0th base for mainly this reason
              We are using the usize round down behavior here to denote how many
              data points per block there are. 

              Say spacer = 5, and a block has length 15. We want three data points:
              base 0, base 5, and base 10. If the block has length 16, we want 4 points:
              base 0, base 5, base 10, and base 15. If data wasn't normalized so that 
              the first base in each block was considered base 0, this behavior
              couldn't be captured. 

              The biggest advantage to doing things like this is that bps can be
              converted to index easily by just dividing by spacer. A bp with a 
              non-integer part in its representation is dropped in the waveform.
             */
            let point_lens: Vec<usize> = block_lens.iter().map(|a| 1+((a-1)/spacer)).collect();

            let mut start_dats: Vec<usize> = Vec::new();

            let mut size: usize = 0;

            for i in 0..point_lens.len(){
                start_dats.push(size);
                size += point_lens[i];
            }

            let tot_L: usize = point_lens.iter().sum();

            Waveform {
                wave: vec![0.0; tot_L],
                spacer: spacer,
                point_lens: point_lens,
                block_lens: block_lens,
                start_bases: start_bases,
                start_dats: start_dats
            }
        }

        pub fn zero(&self) -> Waveform {

            let tot_L: usize = self.point_lens.iter().sum();

            
            Waveform {
                wave: vec![0.0; tot_L],
                spacer: self.spacer,
                point_lens: self.point_lens.clone(),
                block_lens: self.block_lens.clone(),
                start_bases: self.start_bases.clone(),
                start_dats: self.start_dats.clone()
            }

        }

        pub fn place_peak(&mut self, peak: &Kernel, block: usize, center: usize) {

            let half_len = (peak.len()-1)/2; //Given how we construct kernels, this will never need to be rounded
            let range = self.block_lens[block];

            let place_bp = (half_len as isize)-(center as isize); //This moves the center of the peak to where it should be, taking the rest of it with it
            let cc = (place_bp) % (self.spacer as isize); // This defines the congruence class of the kernel indices that will be necessary for the signal
           
            let zerdat: usize = self.start_dats[block]; //This will ensure the peak is in the correct block

            let min_kern_bp: usize = max(0, place_bp) as usize;
            let nex_kern_bp: usize = min(peak.len() as isize, ((self.spacer*self.point_lens[block]) as isize)+place_bp) as usize; //Technicaly, the end CAN return a negative int. 
                                                                                     //But if it is, panicking is appropriate: 
                                                                                     //center would necessarily be much bigger than the block length



            let all_kernel_inds: Vec<usize> = (min_kern_bp..nex_kern_bp).collect();

            let need_kernel_inds: Vec<usize> = all_kernel_inds.iter().filter(|&bp| ((bp % self.spacer) == (cc as usize))).map(|f| *f as usize).collect();

            let kern_values = need_kernel_inds.iter().map( |i| peak.get_curve()[*i] ).collect::<Vec<_>>();

            
            
            let completion: usize = ((cc-((peak.len() % self.spacer) as isize)).rem_euclid(self.spacer as isize)) as usize; //This tells us how much is necessary to add to the length 
                                                                                //of the kernel to hit the next base in the cc
            
            let min_kern_cc = max(cc, place_bp);
            let nex_kern_cc = min(((self.point_lens[block]*self.spacer) as isize)+place_bp, ((peak.len()+completion) as isize));

            let min_data: usize = ((min_kern_cc-place_bp)/((self.spacer) as isize)) as usize;  //Always nonnegative: if place_bp > 0, min_kern_bp = place_bp
            let nex_data: usize = ((nex_kern_cc-place_bp)/((self.spacer) as isize)) as usize; //Assume nonnegative for the same reasons as nex_kern_bp

            

            let kern_change = &mut self.wave[(min_data+zerdat)..(nex_data+zerdat)];

            for i in 0..kern_change.len(){
                kern_change[i] += kern_values[i];
            }
            
           

        }

        pub fn produce_noise(&self, data: &Waveform, ar_corrs: &Vec<f64>) -> Vec<f64> {
            let residual = self-data;


            let mut end_dats = residual.start_dats()[1..residual.start_dats.len()].to_vec();

            let resid = residual.wave;
            
            end_dats.push(resid.len());

            let mut len_penalties = Vec::new();

            for k in 0..end_dats.len() {
                len_penalties.push((k+1)*ar_corrs.len());
            }

            let filt_lens: Vec<usize> = end_dats.iter().zip(len_penalties).map(|(a, b)| a-b).collect();

            let c_num = ar_corrs.len();

            let l_c = ar_corrs.len();

            let mut fin_noise: Vec<f64> = vec![0.0; filt_lens.iter().sum()];

            for k in 0..end_dats.len(){

                let sind: usize = if k == 0 {0} else {end_dats[k-1]};

    
                let mut block: Vec<f64> = resid[(sind+l_c)..end_dats[k]].iter().zip(resid[(sind+l_c-1)..(end_dats[k]-1)].iter()).map(|(a,b)| a-ar_corrs[0]*b).collect();
                
                if l_c > 1 {
                
                    for i in 1..l_c {
                        block = block.iter().zip(resid[(sind+l_c-(i+1))..(end_dats[k]-(i+1))].iter()).map(|(a,b)| a-ar_corrs[i]*b).collect();
                    }
                }

                let sind: usize = if k == 0 {0} else {filt_lens[k-1]};

                let block_ref = &mut fin_noise[sind..filt_lens[k]];

                for i in 0..block_ref.len(){

                    block_ref[i] = block[i];

                }

            }

            fin_noise


        }


        pub fn start_bases(&self) -> Vec<usize> {
            self.start_bases.clone()
        }

        pub fn spacer(&self) -> usize {
            self.spacer
        }

        pub fn raw_wave(&self) -> Vec<f64> {
            self.wave.clone()
        }

        pub fn start_dats(&self)  -> Vec<usize> {
             self.start_dats.clone()
        }

        pub fn point_lens(&self)  -> Vec<usize> {
            self.point_lens.clone()
        }
    }
    
    impl Add<&Waveform> for &Waveform {

        type Output = Waveform;

        fn add(self, wave2: &Waveform) -> Waveform {

            if self.spacer != wave2.spacer()  {
                panic!("These signals do not add! Spacers must be equal!")
            }
 
            let other_wave = wave2.raw_wave();
            
            Waveform {
                wave: self.wave.iter().zip(other_wave).map(|(a, b)| a+b).collect(),
                spacer: self.spacer,
                point_lens: self.point_lens.clone(),
                block_lens: self.block_lens.clone(),
                start_bases: self.start_bases.clone(),
                start_dats: self.start_dats.clone(),
            }

        }

    }
    
    impl Sub<&Waveform> for &Waveform {

        type Output = Waveform;

        fn sub(self, wave2: &Waveform) -> Waveform {

            if self.spacer != wave2.spacer()  {
                panic!("These signals do not subtract! Spacers must be equal!")
            }
            let other_wave = wave2.raw_wave();
            
            Waveform {
                wave: self.wave.iter().zip(other_wave).map(|(a, b)| a-b).collect(),
                spacer: self.spacer,
                point_lens: self.point_lens.clone(),
                block_lens: self.block_lens.clone(),
                start_bases: self.start_bases.clone(),
                start_dats: self.start_dats.clone(),
            }

        }

    }

    impl Mul<f64> for &Waveform {

        type Output = Waveform;

        fn mul(self, rhs: f64) -> Waveform {

            Waveform{
                wave: self.wave.iter().map(|a| a*rhs).collect(),
                spacer: self.spacer,
                point_lens: self.point_lens.clone(),
                block_lens: self.block_lens.clone(),
                start_bases: self.start_bases.clone(),
                start_dats: self.start_dats.clone(),
            }
        }


    }


    pub struct Noise {
        resids: Vec<f64>,
        dist: StudentsT,
    }

    impl Noise {


        pub fn new(resids: Vec<f64>, sigma_back : f64, df : f64) -> Noise {

            let dist = StudentsT::new(0., sigma_back, df).unwrap();
            Noise{ resids: resids, dist: dist}

        }

        pub fn resids(&self) -> Vec<f64> {
            self.resids.clone()
        }

        //The ranks need to be 1 indexed for the AD calculation to work
        pub fn rank(&self) -> Vec<usize> {

            let mut rx: Vec<(usize, f64)> = self.resids.clone().iter().enumerate().map(|(a, b)| (a, *b)).collect();

            rx.sort_unstable_by(|(_,a), (_,b)| a.partial_cmp(b).unwrap());

            let mut ranks: Vec<usize> = vec![0; rx.len()];

            let mut ind = 0;

            for &(i, e) in &rx {
                ranks[i] = ind+1;
                ind += 1;
            }
            
            ranks
        }

        pub fn ad_calc(&self) -> f64 {

            let mut forward: Vec<f64> = self.resids();
            forward.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
            let reverse: Vec<f64> = forward.iter().rev().map(|a| *a).collect();

            let n = forward.len();

            let inds: Vec<f64> = (0..n).map(|a| (2.0*(a as f64)+1.0)/(n as f64)).collect();

            //I dedicated an inhuman amount of work trying to directly implement ln CDF here
            //And then ran a simple numerical test and realized that I don't need to bother
            //The statrc crate is numerically stable enough out even to +/- 200

            let Ad = -(n as f64) - forward.iter().zip(reverse).zip(inds)
                                          .map(|((f,r),m)| m*(self.dist.cdf(*f)+self.dist.sf(r))).sum::<f64>();


            Ad
        }

        pub fn ad_grad(&self) -> Vec<f64> {

            let ranks: Vec<f64> = self.rank().iter().map(|a| *a as f64).collect();

            let forward: Vec<f64> = self.resids();

            let n = forward.len();


            let derivative: Vec<f64> = forward.iter().zip(ranks)
                                        .map(|(&a, b)| (self.dist.pdf(a)/(self.dist.sf(a)*(n as f64)))*(2.*(n as f64)-((2.*b+1.)/self.dist.cdf(a))))
                                        .collect();

            derivative

        }

        fn low_val(lA: f64) -> f64 {

            const C: f64 = PI*PI/8.0;
            let cfs = [2.00012,0.247105,-0.0649821, 0.0347962,-0.0116720,0.00168691];

            let expo = (0..6).map(|a| 2.0*(a as f64)-1.0).collect::<Vec<f64>>();
 
            let p: f64 = cfs.iter().zip(expo)
                            .map(|(&c, e)| c*e*lA.sqrt().powf(e-2.0)/2.0 + C*c*lA.sqrt().powf(e-4.0)).sum();

            -C/lA+p.ln()

        }


        fn high_val(hA: f64) -> f64 {

            (3.0/(hA*PI)).ln()/2.0-hA

        }

        pub fn ad_like(A: f64) -> f64 {

            const A0: f64 = 2.64;
            const k: f64 = 84.44556;

            let lo = Self::low_val(A);
            let hi = Self::high_val(A);

            let w = 1.0/(1.0+(A/A0).powf(k));

            w*lo+(1.0-w)*hi
        }

        pub fn ad_diff(A: f64) -> f64 {

            const h: f64 = 0.00001;
            (Self::ad_like(A+h)-Self::ad_like(A))/h

        }



    }

    impl Mul<&Noise> for &Noise {

        type Output = f64;

        fn mul(self, rhs: &Noise) -> f64 {

            let rhs_r = rhs.resids();

            if(self.resids.len() != rhs_r.len()){
                panic!("Residuals aren't the same length?!")
            }

            self.resids.iter().zip(rhs_r).map(|(a,b)| a*b).sum()
        }
    }


    




















}
