pub mod wave{
 
    use std::ops::Mul;
    use core::f64::consts::PI;

    const WIDE: f64 = 3.0;

    pub struct Kernel{

        peak_width: f64,
        kernel: Vec<f64>,
    }


    impl Kernel {
        
        pub fn new(peak_width: f64) -> Kernel {

            let span = (peak_width*WIDE) as isize;

            let domain: Vec<isize> = (-span..(span+1)).collect();

            let range = domain.iter().map(|a| (-((*a as f64).powf(2.0))/(2.0*peak_width.powf(2.0))).exp()).collect();

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
        
        pub fn mul(&self, rhs: f64) -> Vec<f64> {
            self.kernel.iter().map(|v| v * rhs).collect()
        }


    }

    



}
