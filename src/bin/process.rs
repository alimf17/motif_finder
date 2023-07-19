use log::warn;

pub fn rhat(chain_pars: &Vec<Vec<f64>>, chain_length: usize) -> f64 {

    let max_poss_chain = chain_pars.iter().map(|a| a.len()).min().expect("Empty chain vectors not allowed");
    
    let real_length = if max_poss_chain < chain_length {
        warn!("Your chain length is larger than the minimum chain length of your parameters!\n Defaulting to the minimum chain length of your parameters.");
        max_poss_chain
        } else {chain_length};

    let chain_sums: Vec<f64> = chain_pars.iter().map(|a| tail_slice(a, real_length).iter().sum::<f64>()).collect();

    let chain_means: Vec<f64> = chain_sums.iter().map(|a| (*a)/(real_length as f64)).collect();
  
    let big_mean = chain_means.iter().sum::<f64>()/((chain_pars.len()) as f64);

    let chain_vars: Vec<f64> = chain_pars.iter().zip(chain_means.iter()).map(|(chain, mean)| {

        tail_slice(chain, real_length).iter().map(|val| (val-mean).powi(2)).sum::<f64>()/((real_length-1) as f64)

    }).collect();

    let b: f64 = ((real_length as f64)/((chain_pars.len()-1) as f64))*chain_means.iter().map(|a| (a-big_mean).powi(2)).sum::<f64>();

    let w: f64 = chain_vars.iter().sum::<f64>()/(chain_pars.len() as f64);

    (1.0+((b/w)-1.0)/(real_length as f64)).sqrt()

}

pub fn tail_slice<T>(slicer: &[T], last_n: usize) -> &[T] {
    let first = if last_n > slicer.len() {0} else {slicer.len()-last_n};
    &slicer[first..slicer.len()]
}

#[cfg(test)]
mod tests {

    use super::*;
    #[test]
    fn rhat_test() {
        
        let chain_pars = vec![vec![1.0_f64, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let r_hat = rhat(&chain_pars, 3);
        println!("{:?} {}", tail_slice(&chain_pars[0], 3), r_hat);
        assert!((r_hat-(31.0_f64/6.0).sqrt()).abs() < 1e-6);

        let r2_hat = rhat(&chain_pars, 2);
        assert!((r2_hat-(19.0/2.0_f64).sqrt()).abs() < 1e-6);

    }

}
