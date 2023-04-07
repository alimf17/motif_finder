pub mod wave{
 
    use std::ops::Add;
    use std::ops::Sub;
    use std::ops::Mul;
    use std::cmp::max;
    use std::cmp::min;
    use core::f64::consts::PI;

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
            point_lens = block_lens.iter().map(|a| 1+((a-1)/spacer)).collect(); 

            Waveform {
                wave: vec[0; point_lens[len(point_lens)-1]],
                spacer: spacer,
                point_lens: point_lens,
                block_lens: block_lens,
                start_bases: start_bases,
            }
        }

        pub fn zero(&self) -> Waveform {

            Waveform {
                wave: vec[0; self.point_lens[len(self.point_lens)-1]],
                spacer: self.spacer,
                point_lens: self.point_lens.clone(),
                block_lens: self.block_lens.clone(),
                start_bases: self.start_bases.clone(),
            }

        }

        pub fn place_a_peak(&mut self, peak: &Kernel, block: usize, center: usize): {

            let half_len = (peak.len-1)/2; //Given how we construct kernels, this will never need to be rounded
            let range = self.block_lens[block];

            let place_bp = (half_len as isize)-(center as isize); //This moves the center of the peak to where it should be, taking the rest of it with it
            let cc = (place_bp) % self.spacer; // This defines the congruence class of the kernel indices that will be necessary for the signal
           
            let zerbase: usize = self.start_bases[block]; //This will ensure the peak is in the correct block

            let min_kern_bp: usize = max(0, place_bp);
            let nex_kern_bp: usize = min(peak.len, spacer*self.point_lens[block]+place_bp); //Technicaly, the end CAN return a negative int. 
                                                                                     //But if it is, panicking is appropriate: 
                                                                                     //center would necessarily be much bigger than the block length



            let all_kernel_inds = (min_kern_bp..nex_kern_bp).collect();

            let need_kernel_inds = all_kernel_inds.iter().filter(|&bp| ((bp % self.spacer) == cc)).map(|f| *f as usize).collect();

            let kern_values = indb.iter().map( |i| need_kern_inds[*i] ).collect::<Vec<_>>();

            
            
            let completion: usize = (cc-(peak.lean % self.spacer))%self.spacer; //This tells us how much is necessary to add to the length 
                                                                                //of the kernel to hit the next base in the cc
            
            let min_kern_cc = max(cc, place_bp);
            let max_kern_cc = min(self.point_lens[block]*spacer+place_bp, peak.len+completion);

            let min_data: usize = (min_kern_cc-place_bp)/spacer; //Always nonnegative: if place_bp > 0, min_kern_bp = place_bp
            let nex_data: usize = (nex_kern_cc-place_bp)/spacer; //Assume nonnegative for the same reasons as nex_kern_bp

            

            let kern_change = &mut self.wave[min_data, nex_data];

            for i in 0..kern_change.len(){
                kern_change[i] += kern_values[i];
            }
            
           

        }

        pub fn spacer(&self) -> usize {
            self.spacer
        }

        pub fn raw_wave(&self) -> Vec<f64> {
            self.wave.clone()
        }
    }
    
    impl Add<&Waveform> for &Waveform {

        type Output = Waveform;

        fn add(self, wave2: &Waveform) -> Waveform {

            if self.spacer != wave2.spacer()  {
                panic!("These signals do not add! Spacers must be equal!")
            }

            other_wave = wave2.raw_wave();
            
            Waveform {
                wave: self.wave.iter().zip(other_wave).map(|a, b| a+b).collect(),
                spacer: self.spacer,
                point_lens: self.point_lens.clone(),
                block_lens: self.block_lens.clone(),
            }

        }

    }
    
    impl Sub<&Waveform> for &Waveform {

        type Output = Waveform;

        fn sub(self, rhs: f64) -> waveform {

            if self.spacer != wave2.spacer()  {
                panic!("These signals do not subtract! Spacers must be equal!")
            }
            other_wave = wave2.raw_wave();
            
            Waveform {
                wave: self.wave.iter().zip(other_wave).map(|a, b| a-b).collect(),
                spacer: self.spacer,
                point_lens: self.point_lens.clone(),
                block_lens: self.block_lens.clone(),
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
            }
        }


    }




    



}
