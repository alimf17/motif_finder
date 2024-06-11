use std::ops::{Add, AddAssign, Sub, SubAssign, Mul};
use std::cmp::{max, min};

//use std::time::{Duration, Instant};

use core::f64::consts::PI;
use core::iter::zip;

use crate::base::Bp;
use crate::sequence::{Sequence, BP_PER_U8};
use crate::modified_t::*;
use crate::data_struct::AllDataUse;

use log::warn;

use statrs::distribution::{StudentsT, Continuous, ContinuousCDF};


use aberth;
use num_complex::{Complex, ComplexFloat};
const EPSILON: f64 = 1e-8;

use num_traits::float::{Float, FloatConst};
use num_traits::MulAdd;

use plotters::prelude::*;
use plotters::coord::types::RangedSlice;
use plotters::coord::Shift;
use plotters::prelude::full_palette::ORANGE;

use rayon::prelude::*;

use assume::assume;


use serde::{Serialize, Deserialize};

pub const WIDE: f64 = 3.0;



#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Kernel{

    peak_height: f64,
    peak_width: f64,
    kernel: Vec<f64>,
}

impl Mul<f64> for &Kernel {

    type Output = Kernel;

    fn mul(self, rhs: f64) -> Kernel {

        
        Kernel {
            peak_height: self.peak_height*rhs,
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
            peak_height: peak_height,
            peak_width: peak_width,
            kernel: range,
        }

    }

    pub fn get_sd(&self) -> f64 {
        self.peak_width
    }

    pub fn get_curve(&self) -> &Vec<f64> {

        &self.kernel
    
    }

    pub fn len(&self) -> usize {
        self.kernel.len()
    }
   
    pub fn get_height(&self) -> f64 {
        self.peak_height
    }
    pub fn scaled_kernel(&self, scale_by: f64) -> Kernel {
        self*scale_by
    }

}

//CANNOT BE SERIALIZED
#[derive(Clone, Debug)]
pub struct Waveform<'a> {
    wave: Vec<f64>,
    spacer: usize,
    point_lens: Vec<usize>,
    start_dats: Vec<usize>,
    seq: &'a Sequence,
}

impl<'a> Waveform<'a> {

    /*
       This function has an additional constraint for correctness:

       The start_data must be organized in accordance with spacer and sequence.
       In particular, each block must have 1+floor((block length-1)/spacer) data points
       HOWEVER, this initializer only knows TO panic if the total number of data points
       cannot be reconciled with seq and spacer. If you have one too MANY points in one
       block and one too FEW points in another, this will NOT know to break. If you get 
       this wrong, your run will be memory safe, but it will give garbage and incorrect
       answers.
       */
    pub fn new(start_data: Vec<f64>, seq: &'a Sequence, spacer: usize) -> Waveform<'a> {


        if spacer == 0 {panic!("Spacer has to be positive, or else we're going to divide by zero a bunch.");}

        let (point_lens, start_dats) = Self::make_dimension_arrays(seq, spacer);
       
        //println!("{} {} pl {:?}\n sd {:?}", start_data.len(), point_lens.len(), point_lens, start_dats);
        if (point_lens.last().unwrap() + start_dats.last().unwrap()) != start_data.len() {
            panic!("IMPOSSIBLE DATA FOR THIS SEQUENCE AND SPACER")
        }



        Waveform {
            wave: start_data,
            spacer: spacer,
            point_lens: point_lens,
            start_dats: start_dats,
            seq: seq,
        }
    }

    pub fn make_dimension_arrays(seq: &'a Sequence, spacer: usize) -> (Vec<usize>, Vec<usize>) {

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
        let point_lens: Vec<usize> = seq.block_lens().iter().map(|a| 1+((a-1)/spacer)).collect();

        let mut size: usize = 0;

        let mut start_dats: Vec<usize> = Vec::new();

        for i in 0..point_lens.len(){
            start_dats.push(size);
            size += point_lens[i];
        }

        (point_lens, start_dats)
    }

    pub fn create_zero(seq: &'a Sequence, spacer: usize) -> Waveform<'a> {
       
        let point_lens: Vec<usize> = seq.block_lens().iter().map(|a| 1+((a-1)/spacer)).collect();

        let mut start_dats: Vec<usize> = Vec::new();

        let mut size: usize = 0;

        for i in 0..point_lens.len(){
            start_dats.push(size);
            size += point_lens[i];
        }

        let tot_l: usize = point_lens.iter().sum();

        Waveform {
            wave: vec![0.0; tot_l],
            spacer: spacer,
            point_lens: point_lens,
            start_dats: start_dats,
            seq: seq,
        }
    }

    pub fn derive_zero(&self) -> Waveform {

        let tot_l: usize = self.point_lens.iter().sum();

        
        Waveform {
            wave: vec![0.0; tot_l],
            spacer: self.spacer,
            point_lens: self.point_lens.clone(),
            start_dats: self.start_dats.clone(),
            seq: self.seq,
        }

    }

    //SAFETY: -block must be less than the number of blocks
    //        -center must be less than the number of bps in the blockth block
    //        -the length of peak MUST be strictly less than the number of base pairs represented in the smallest data block
    pub(crate) unsafe fn place_peak(&mut self, peak: &Kernel, block: usize, center: usize) {



        //Given how we construct kernels, this will never need to be rounded: they always have an odd number of data points
        let place_bp = (((peak.len()-1)/2) as isize)-(center as isize); //This moves the center of the peak to where it should be, taking the rest of it with it
        let cc = (place_bp).rem_euclid(self.spacer as isize); // This defines the congruence class of the kernel indices that will be necessary for the signal
       
        let zerdat: usize = self.start_dats[block]; //This will ensure the peak is in the correct block

        //let _min_kern_bp: usize = max(0, place_bp) as usize;
        //let _nex_kern_bp: usize = min(peak.len() as isize, ((self.spacer*self.point_lens[block]) as isize)+place_bp) as usize; //This is always positive if you uphold the center safety invariant 
 
        //let which_bps = (min_kern_bp..nex_kern_bp).filter(|&bp| ((bp % self.spacer) == (cc as usize)));;
        //let kern_values: Vec<f64> = (min_kern_bp..nex_kern_bp).filter(|&bp| ((bp % self.spacer) == (cc as usize))).map(|f| peak.get_curve()[f as usize]).collect();
        
       

        let completion: usize = ((cc-((peak.len() % self.spacer) as isize)).rem_euclid(self.spacer as isize)) as usize; //This tells us how much is necessary to add to the length 
                                                                            //of the kernel to hit the next base in the cc
        
        let min_kern_cc = max(cc, place_bp);
        let nex_kern_cc = min(((self.point_lens[block]*self.spacer) as isize)+place_bp, (peak.len()+completion) as isize);
        let min_data: usize = ((min_kern_cc-place_bp)/((self.spacer) as isize)) as usize;  //Always nonnegative: if place_bp > 0, min_kern_bp = place_bp
        let nex_data: usize = ((nex_kern_cc-place_bp)/((self.spacer) as isize)) as usize; //Assume nonnegative for the same reasons as nex_kern_bp


        let kern_change = self.wave.get_unchecked_mut((min_data+zerdat)..(nex_data+zerdat));
/*
        if kern_values.len() > 0 {
            //println!("{} {} {} {} {} peak",min_data+zerdat, nex_data+zerdat, kern_values.len(), kern_change.len(), w);
            for i in 0..kern_change.len(){
                kern_change[i] += kern_values[i];
            }
        } 
        
  */
        let kern_start = min_kern_cc as usize;
        for i in 0..kern_change.len() {
            kern_change[i] += peak.kernel.get_unchecked(kern_start+i*self.spacer);
        }


    }


    pub fn kmer_propensities(&self, k: usize) -> Vec<f64> {

        let coded_sequence = self.seq.seq_blocks();
        let block_lens = self.seq.block_lens(); //bp space
        let block_starts = self.seq.block_u8_starts(); //stored index space

        let mut uncoded_seq: Vec<Bp> = vec![Bp::A; self.seq.max_len()];

        let mut store: [Bp ; BP_PER_U8];

        let mut propensities: Vec<f64> = vec![0.0; self.seq.number_unique_kmers(k)];

        {
            let uncoded_seq = uncoded_seq.as_mut_slice();
            for i in 0..(block_starts.len()) {


                for jd in 0..(block_lens[i]/BP_PER_U8) {

                    store = Sequence::code_to_bases(coded_sequence[block_starts[i]+jd]);
                    for k in 0..BP_PER_U8 {
                        uncoded_seq[BP_PER_U8*jd+k] = store[k];
                    }

                }


                for j in 0..((block_lens[i])-k) {

                    let center = j + ((k-1)/2);

                    let lower_data_ind = center/self.spacer; 

                    //Center cannot possibly beyond the bases because of how I take j: I stop before j+k runs 
                    //into the sequence end, which will always make lower_data_ind hit, at most the largest
                    //location. BUT, I CAN be over any ability to check the next location

                    //SAFETY: notice how we defined j, and how it guarentees that get_unchecked is fine
                    let u64_mot = Sequence::kmer_to_u64(&(unsafe { uncoded_seq.get_unchecked(j..(j+k)) }).to_vec());
                    let between = center % self.spacer;
                    //The second statement in the OR seems bizarre, until you remember that we stop
                    //including DATA before we stop including BASES. This is our check to make sure
                    //that if we're in that intermediate, then we include such overrun bases
                    let to_add = if (between == 0) || (lower_data_ind + 1 >= self.point_lens[i]) {
                        /*#[cfg(test)]{
                            println!("block {} j {} kmer {} amount {}", i, j,u64_mot, self.wave[self.start_dats[i]+lower_data_ind]); 
                        }*/
                        self.wave[self.start_dats[i]+lower_data_ind]
                    } else {
                        let weight = (between as f64)/(self.spacer as f64);
                        /*#[cfg(test)]{
                            println!("block {} j {} kmer {} amount {}", i, j, u64_mot, self.wave[self.start_dats[i]+lower_data_ind]*(1.-weight)+self.wave[self.start_dats[i]+lower_data_ind+1]*weight);
                        }*/
                        self.wave[self.start_dats[i]+lower_data_ind]*(1.-weight)+self.wave[self.start_dats[i]+lower_data_ind+1]*weight
                    };

                    //A motif should never be proposed unless there's a place in the signal that
                    //could reasonably considered motif-ish
                    //Note: this does NOT interfere with detailed balance: motifs are always
                    //proposed with a minimum height and so when death is proposed, it should
                    //always be the case that there's SOMEWHERE with the height minimum
                    if to_add.abs() > (0.0) {
                        propensities[self.seq.id_of_u64_kmer_or_die(k, u64_mot)] += to_add.powi(2);
                    }
                }

            

            }
            //println!("unique u64 {}-mers: {:?}", k, self.seq.unique_kmers(k));

        }


        propensities


    }

    //Panics: if other_wave doesn't point to the same AllDataUse as self
    pub fn rmse_with_wave(&self, other_wave: &Waveform) -> f64 {

        let residual = self-other_wave;

        let length: f64 = residual.wave.len() as f64;

        (residual.wave.into_iter().map(|a| a.powi(2)).sum::<f64>()/length).sqrt()

    }

    pub fn generate_extraneous_binding(data_ref: &AllDataUse, scaled_heights_array: &[f64]) -> Vec<f64> {

        println!("in extra, {}", data_ref.background_ref().noise_spread_par());

        //Is our potential binding strong enough to even attempt to try extraneous binding?
        let caring_threshold = (3.0*data_ref.background_ref().noise_spread_par());

        println!("pase first min");

        let extraneous_bindings: Vec<_> = scaled_heights_array.iter().filter(|&a| *a > caring_threshold).collect();

        //Do we have any extraneous binding that we need to account for?
        if extraneous_bindings.len() == 0 { return Vec::new();}

        println!("past second min");

        let scaled_heights = extraneous_bindings;

        let sd_over_spacer = data_ref.unit_kernel_ref().get_sd()/(data_ref.data().spacer() as f64);

        let ln_caring_threshold = caring_threshold.ln();


        //TODO: if Kernel goes from being exclusively Gaussians to incorporating other shapes, this needs to change to check those shapes
        //Note: this relies on the fact that casting an f64 to a usize uses the floor function
        let generate_size_kernel = |x: f64| (sd_over_spacer*(2.0*(x.ln()-ln_caring_threshold)).sqrt()) as usize;
        let quadriatic_ratio = (-0.5*(sd_over_spacer).powi(-2)).exp();

        let size_hint: usize = generate_size_kernel(*scaled_heights[0]);

        let mut output_vec: Vec<f64> = Vec::with_capacity(size_hint*scaled_heights.len());

        for height in scaled_heights {

            let kernel_size = generate_size_kernel(*height);

            //If the size of your kernel is genuinely overflowing integers after filtering out by spacer
            //You have such massive problems that this should really be the least of your concerns
            //SAFETY: we have a panic set on BackgroundDist in case this conversion could fail. 
            //        We choose to do that because my word, if your log2 data's 
            //        background distribution has an SD of > a MILLION, your experiment is horribly degenerate
            //        It should be completely impossible to unless you're getting
            //        heights more than exp((i32::MAX/sd_over_spacer)^2) times larger than caring_threshold
            //        And it's only possible for that to happen within f64's specs if the background sd
            //        is more than a million. MORE than that if spacer is bigger than 1

            let kernel_size_i32: i32 = unsafe{ kernel_size.try_into().unwrap_unchecked() };
            let mut kernel_bit: Vec<f64> = Vec::with_capacity(2*kernel_size+1);

            kernel_bit.push(*height);

            if kernel_size > 0 {
                for i in 1..=kernel_size_i32 {
                    let ele = height*quadriatic_ratio.powi(i*i);
                    kernel_bit.push(ele);
                    kernel_bit.push(ele);
                }
            }

            output_vec.append(&mut kernel_bit);
        }

        output_vec


    }

    pub fn produce_noise<'b>(&self, data_ref: &'b AllDataUse) -> Noise<'b> {
        
        let residual = self-data_ref.data();

        return Noise::new(residual.wave, Vec::new(), data_ref.background_ref());
    }

    pub fn produce_noise_with_extraneous<'b>(&self, data_ref: &'b AllDataUse, extraneous_bind_array: &[f64]) -> Noise<'b> {

        let mut noise = self.produce_noise(data_ref);

        let null_sequence_binding = Self::generate_extraneous_binding(data_ref, extraneous_bind_array);

        noise.replace_extraneous(null_sequence_binding);

        noise

    }

    pub fn median_distance_between_waves(&self, data: &Self) -> f64 {
        let residual = self-data;
        let mut abs_wave = residual.wave.iter().map(|&a| a.abs()).collect::<Vec<f64>>();
        abs_wave.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        if (abs_wave.len() & 1) == 1 {
            abs_wave[abs_wave.len()/2]
        } else {
            (abs_wave[abs_wave.len()/2]+abs_wave[abs_wave.len()/2 -1])/2.0
        }
    }
   

    pub fn generate_all_locs(&self) -> Vec<usize> {

        let mut length: usize = 0;
        for &a in self.point_lens.iter() { length += a; }

        let mut locations: Vec<usize> = Vec::with_capacity(length);

        //println!("starts {:?}", self.start_dats);
        //println!("lens {:?}", self.point_lens);
        for i in 0..(self.start_dats.len()){

            for j in 0..(self.point_lens[i]){
                locations.push((self.start_dats[i]+j)*self.spacer);
            }
        }

        locations

    }

    pub fn generate_all_indexed_locs_and_data(&self, start_lens: &[usize]) -> Option<Vec<(Vec<usize>, Vec<f64>)>> {

        if self.start_dats.len() != start_lens.len() {
            return None;
        }

        let mut locations_and_data: Vec<(Vec<usize>, Vec<f64>)> = Vec::with_capacity(start_lens.len());

        for i in 0..start_lens.len() {

            let small_loc: Vec<usize> = (0..self.point_lens[i]).map(|j| start_lens[i]+(j*self.spacer)).collect();
            let end_len = if i < (start_lens.len()-1) { self.start_dats[i+1] } else { self.wave.len() };
            let small_dat: Vec<f64> = self.wave[self.start_dats[i]..end_len].to_owned();
            locations_and_data.push((small_loc, small_dat));
        }

        Some(locations_and_data)

    }

    pub fn save_waveform_to_directory(&self, data_ref: &AllDataUse, signal_directory: &str, trace_color: &RGBColor) {

        let current_resid = data_ref.data()-&self;

        let min_signal = self.wave.iter().min_by(|a,b| a.partial_cmp(b).unwrap()).expect("Waves have elements");
        let min_data_o = data_ref.data().wave.iter().min_by(|a,b| a.partial_cmp(b).unwrap()).expect("Waves have elements");
        let min_resids = current_resid.wave.iter().min_by(|a,b| a.partial_cmp(b).unwrap()).expect("Waves have elements");

        let min = min_signal.min(min_data_o.min(*min_resids))-1.0;

        let max_signal = self.wave.iter().max_by(|a,b| a.partial_cmp(b).unwrap()).expect("Waves have elements");
        let max_data_o = data_ref.data().wave.iter().max_by(|a,b| a.partial_cmp(b).unwrap()).expect("Waves have elements");
        let max_resids = current_resid.wave.iter().max_by(|a,b| a.partial_cmp(b).unwrap()).expect("Waves have elements");

        let max = max_signal.max(max_data_o.max(*min_resids))+1.0;
        
        let blocked_locs_and_signal = self.generate_all_indexed_locs_and_data(data_ref.zero_locs()).expect("We designed signal to correspond to data_ref");

        let blocked_locs_and_data = data_ref.data().generate_all_indexed_locs_and_data(data_ref.zero_locs()).expect("Our data BETTER correspond to data_ref");

        let blocked_locs_and_resid = current_resid.generate_all_indexed_locs_and_data(data_ref.zero_locs()).expect("We designed signal to correspond to data_ref");

        if let Err(creation) = std::fs::create_dir_all(signal_directory) {
            warn!("Could not make or find directory \"{}\"! \n{}", signal_directory, creation);
            return;
        };

        for i in 0..blocked_locs_and_signal.len() {

            let loc_block = &blocked_locs_and_signal[i].0;
            let sig_block = &blocked_locs_and_signal[i].1;
            let dat_block = &blocked_locs_and_data[i].1;
            let res_block = &blocked_locs_and_resid[i].1;

            let signal_file = format!("{}/{i}.png", signal_directory);

            let plot = BitMapBackend::new(&signal_file, (3300, 1500)).into_drawing_area();

            let derived_color = DerivedColorMap::new(&[WHITE, ORANGE, RED]);

            plot.fill(&WHITE).unwrap();

            let (left, right) = plot.split_horizontally((95).percent_width());

            let (right_space, _) = right.split_vertically((95).percent_height());

            let mut bar = ChartBuilder::on(&right_space).margin(10).set_label_area_size(LabelAreaPosition::Right, 100).caption("Deviance", ("sans-serif", 50)).build_cartesian_2d(0_f64..1_f64, 0_f64..1_f64).unwrap();

            bar.configure_mesh()
                .y_label_style(("sans-serif", 40))
                .disable_mesh().draw().unwrap();

            let deviances = (0..10000_usize).map(|x| (x as f64)/10000.0).collect::<Vec<_>>();

            bar.draw_series(deviances.windows(2).map(|x| Rectangle::new([( 0.0, x[0]), (1.0, x[1])], derived_color.get_color(x[0]).filled()))).unwrap();

            let (upper, lower) = left.split_vertically((86).percent_height());

            let mut chart = ChartBuilder::on(&upper)
                .set_label_area_size(LabelAreaPosition::Left, 100)
                .set_label_area_size(LabelAreaPosition::Bottom, 100)
                .caption("Signal Comparison", ("Times New Roman", 80))
                .build_cartesian_2d((loc_block[0] as f64)..(*loc_block.last().unwrap() as f64), min..max).unwrap();

            chart.configure_mesh()
                .x_label_style(("sans-serif", 40))
                .y_label_style(("sans-serif", 40))
                .x_label_formatter(&|v| format!("{:.0}", v))
                .x_desc("Genome Location (Bp)")
                .y_desc("Signal Intensity")
                .disable_mesh().draw().unwrap();

            const horiz_offset: i32 = -5;

            chart.draw_series(dat_block.iter().zip(loc_block.iter()).map(|(&k, &i)| Circle::new((i as f64, k),2_u32, Into::<ShapeStyle>::into(&BLACK).filled()))).unwrap().label("True Occupancy Data").legend(|(x,y)| Circle::new((x+2*horiz_offset,y),5_u32, Into::<ShapeStyle>::into(&BLACK).filled()));


            chart.draw_series(LineSeries::new(sig_block.iter().zip(loc_block.iter()).map(|(&k, &i)| (i as f64, k)), trace_color.filled().stroke_width(10))).unwrap().label("Proposed Occupancy Trace").legend(|(x, y)| Rectangle::new([(x+4*horiz_offset, y-4), (x+4*horiz_offset + 20, y+3)], Into::<ShapeStyle>::into(trace_color).filled()));

            chart.configure_series_labels().position(SeriesLabelPosition::LowerRight).margin(40).legend_area_size(10).border_style(&BLACK).label_font(("Calibri", 40)).draw().unwrap();

            let abs_resid: Vec<(f64, f64)> = res_block.iter().map(|&a| {

                let tup = data_ref.background_ref().cd_and_sf(a);
                if tup.0 >= tup.1 { (tup.0-0.5)*2.0 } else {(tup.1-0.5)*2.0} } ).zip(loc_block.iter()).map(|(a, &b)| (a, b as f64)).collect();

            let mut map = ChartBuilder::on(&lower)
                .set_label_area_size(LabelAreaPosition::Left, 100)
                .set_label_area_size(LabelAreaPosition::Bottom, 50)
                .build_cartesian_2d((loc_block[0] as f64)..(*loc_block.last().unwrap() as f64), 0_f64..1_f64).unwrap();

            map.configure_mesh().x_label_style(("sans-serif", 0)).y_label_style(("sans-serif", 0)).x_desc("Deviance").axis_desc_style(("sans-serif", 40)).set_all_tick_mark_size(0_u32).disable_mesh().draw().unwrap();


            map.draw_series(abs_resid.windows(2).map(|x| Rectangle::new([(x[0].1, 0.0), (x[1].1, 1.0)], derived_color.get_color(x[0].0).filled()))).unwrap();

        }
    

    }

    pub fn spacer(&self) -> usize {
        self.spacer
    }

    pub fn read_wave(&self) -> &Vec<f64> {
        &self.wave
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

    pub fn seq(&self) -> &Sequence {
        self.seq
    }

    pub fn number_bp(&self) -> usize {
        self.seq.number_bp()
    }

    pub fn amount_data(&self) -> usize {
        self.wave.len()
    }

}

impl<'a, 'b> Add<&'b Waveform<'b>> for &'a Waveform<'a> {

    type Output = Waveform<'a>;

    fn add(self, wave2: &'b Waveform) -> Waveform<'a> {

        if !std::ptr::eq(self.seq, wave2.seq) || (self.spacer != wave2.spacer()) {
            panic!("These signals do not add! Spacers must be equal and waves must point to the same sequence!");
        }

        let other_wave = wave2.raw_wave();
        
        Waveform {
            wave: self.wave.iter().zip(other_wave).map(|(a, b)| a+b).collect(),
            spacer: self.spacer,
            point_lens: self.point_lens.clone(),
            start_dats: self.start_dats.clone(),
            seq: self.seq,
        }

    }

}

impl<'a, 'b> AddAssign<&'b Waveform<'b>> for Waveform<'a> {

    fn add_assign(&mut self, wave2: &'b Waveform) {

        if !std::ptr::eq(self.seq, wave2.seq) || (self.spacer != wave2.spacer()) {
            panic!("These signals do not add! Spacers must be equal and waves must point to the same sequence!");
        }

        let other_wave = wave2.raw_wave();
        
        let n = self.wave.len();


        //SAFETY: If we have the same sequence pointer and the same spacer, our lengths are provably always identical
        assume!(unsafe: other_wave.len() == n);

        for i in 0..n {
            self.wave[i] += other_wave[i];
        }


    }

}

impl<'a, 'b> Sub<&'b Waveform<'b>> for &'a Waveform<'a> {

    type Output = Waveform<'a>;

    fn sub(self, wave2: &'b Waveform<'b>) -> Waveform<'a> {

    if !std::ptr::eq(self.seq, wave2.seq) || (self.spacer != wave2.spacer()) {
            panic!("These signals do not add! Spacers must be equal and waves must point to the same sequence!");
    }
        let other_wave = wave2.raw_wave();
        
        Waveform {
            wave: self.wave.iter().zip(other_wave).map(|(a, b)| a-b).collect(),
            spacer: self.spacer,
            point_lens: self.point_lens.clone(),
            start_dats: self.start_dats.clone(),
            seq: self.seq,
        }

    }

}

impl<'a, 'b> SubAssign<&'b Waveform<'b>> for Waveform<'a> {

    fn sub_assign(&mut self, wave2: &'b Waveform) {

        if !std::ptr::eq(self.seq, wave2.seq) || (self.spacer != wave2.spacer()) {
            panic!("These signals do not add! Spacers must be equal and waves must point to the same sequence!");
        }

        let other_wave = wave2.raw_wave();
        
        let n = self.wave.len();

        //SAFETY: If we have the same sequence pointer and the same spacer, our lengths are always identical
        assume!(unsafe: other_wave.len() == n);
        
        for i in 0..n {
            self.wave[i] -= other_wave[i];
        }


    }

}

impl<'a> Mul<f64> for &'a Waveform<'a> {

    type Output = Waveform<'a>;

    fn mul(self, rhs: f64) -> Waveform<'a> {
        
        Waveform{
            wave: self.wave.iter().map(|a| a*rhs).collect(),
            spacer: self.spacer,
            point_lens: self.point_lens.clone(),
            start_dats: self.start_dats.clone(),
            seq: self.seq,
        }
    }

}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct WaveformDef {
    wave: Vec<f64>,
    spacer: usize,

}
impl<'a> From<&'a Waveform<'a>> for WaveformDef {
    fn from(other: &'a Waveform) -> Self {
        Self {
            wave: other.raw_wave(),
            spacer: other.spacer(),
        }
    }
}

impl WaveformDef {

    //SAFETY: If the relevant lengths of the waveform do not correspond to the sequence, this will be unsafe. 
    //        By correspond, I mean that every element of point_lens needs to be 1+seq.block_lens/spacer
    //        And start_dats needs to match the cumulative sum of point_lens. self.wave.len() must also equal
    //        point_lens.last().unwrap()+start_dats.last().unwrap(). 
    //
    //ACCURACY:  Make sure that wave is in fact organized into the blocks implied by point_lens and start_dats.
    pub(crate) fn get_waveform<'a>(&self, seq: &'a Sequence) -> Waveform<'a> {
        Waveform::new(self.wave.clone(), seq, self.spacer)
    }

    pub fn len(&self) -> usize {
        self.wave.len()
    }

    pub fn spacer(&self) -> usize {
        self.spacer
    }
}

#[derive(Serialize, Deserialize)]
#[serde(remote = "StudentsT")]
struct StudentsTDef {
    #[serde(getter = "StudentsT::location")]
    location: f64,
    #[serde(getter = "StudentsT::scale")]
    scale: f64,
    #[serde(getter = "StudentsT::freedom")]
    freedom: f64,
}

impl From<StudentsTDef> for StudentsT {

    fn from(def: StudentsTDef) -> StudentsT {
        StudentsT::new(def.location, def.scale, def.freedom).unwrap()
    }

}


#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Background {
    pub dist: BackgroundDist,
    pub kernel: Kernel,
}

impl Background {

    pub fn new(sigma_background : f64, df : f64, peak_width: f64) -> Background {

        let dist = BackgroundDist::new(sigma_background, df);
        
        let kernel = Kernel::new(peak_width, 1.0);

        Background{dist: dist, kernel: kernel}
    }


    //The functions get_near_u64, get_near_f64,  
    //and get_lookup_index are all legacy code from when 
    //I had to implement my distribution with lookup tables
    //and raw f64 bit manipulation to derive array indices.
    //They're mainly still here because they're so terrifying that
    //I love them and I can't bear to part with them.
    #[allow(dead_code)]
    fn get_near_u64(calc: f64) -> u64 {

        //These are based on using floats with a maximum of 8 binary sigfigs
        //after the abscissa and exponents ranging from -12 to 3, inclusive
        const MIN_DT_FLOAT: f64 = 0.000244140625;
        const MAX_DT_FLOAT: f64 = 15.96875;
        const EXP_OFFSET: u64 = (1023_u64-12) << 52;
        let mut bit_check: u64 = calc.to_bits();
       
        if calc.abs() < MIN_DT_FLOAT {
            bit_check = (calc.signum()*MIN_DT_FLOAT).to_bits();
        } else if calc.abs() > MAX_DT_FLOAT {
            bit_check = (calc.signum()*MAX_DT_FLOAT).to_bits();
        }

        (bit_check-EXP_OFFSET) & 0xfffff00000000000


    }
    
    #[allow(dead_code)]
    fn get_near_f64(calc: f64) -> f64 {
        f64::from_bits(Self::get_near_u64(calc))
    }

    #[allow(dead_code)]
    fn get_lookup_index(calc: f64) -> usize {

        let bitprep = Self::get_near_u64(calc);
        (((bitprep & 0x8_000_000_000_000_000) >> 51) 
      + ((bitprep & 0x7_fff_f00_000_000_000) >> 44)) as usize
        
    }

    pub fn ln_cd_and_sf(&self, calc: f64) -> (f64, f64) {
        self.dist.ln_cd_and_sf(calc)
    }
    pub fn cd_and_sf(&self, calc: f64) -> (f64, f64) {
        self.dist.cd_and_sf(calc)
    }
    pub fn cdf(&self, calc: f64) -> f64{
        self.dist.cdf(calc)
    }
    pub fn pdf(&self, calc: f64) -> f64{
        self.dist.pdf(calc)
    }

    pub fn ln_cdf(&self, calc: f64) -> f64{
        self.dist.ln_cdf(calc)
    }
    pub fn ln_sf(&self, calc: f64) -> f64{
        self.dist.ln_sf(calc)
    }
    pub fn ln_pdf(&self, calc: f64) -> f64{
        self.dist.ln_pdf(calc)
    }

    pub fn noise_spread_par(&self) -> f64 {
        self.dist.get_spread_par()
    }

    pub fn kernel_ref(&self) -> &Kernel {
        &self.kernel
    }

    pub fn bp_span(&self) -> usize{
        self.kernel.len()
    }

    pub fn kernel_sd(&self) -> f64 {
        self.kernel.peak_width
    }

    

}


#[derive(Clone)]
pub struct Noise<'a> {
    resids: Vec<f64>,
    extraneous_resids: Vec<f64>,
    pub background: &'a Background,
}

impl<'a> Noise<'a> {


    pub fn new(resids: Vec<f64>, extraneous_resids: Vec<f64>, background: &'a Background) -> Noise<'a> {
        Noise{ resids: resids, extraneous_resids: extraneous_resids, background: background}

    }

    pub fn resids(&self) -> Vec<f64> {
        self.resids.clone()
    }
    pub fn extraneous_resids(&self) -> Vec<f64> {
        self.extraneous_resids.clone()
    }
    
    
    pub fn dist(&self) -> BackgroundDist {
        self.background.dist.clone()
    }

    pub fn replace_extraneous(&mut self, extraneous_resids: Vec<f64>) {
        self.extraneous_resids = extraneous_resids;
    }

    pub fn noise_with_new_extraneous(&self, extraneous_resids: Vec<f64>) -> Noise<'a> {

        let mut new_noise = self.clone();
        new_noise.extraneous_resids = extraneous_resids;
        new_noise

    }

    
    /*pub fn rank(&self) -> Vec<f64> {

        
        //This generates a vector where each element is tagged with its original position
        let mut rx: Vec<(usize, f64)> = self.resids.clone().iter().enumerate().map(|(a, b)| (a, *b)).collect();

        //This sorts the elements, but recalls their original positions
        rx.par_sort_unstable_by(|(i,a), (j,b)| a.partial_cmp(b).expect(format!("No NaNs allowed! {} {} {} {}", i,a, j, b).as_str()));
       
    
        
        let mut ranks: Vec<f64> = vec![0.; rx.len()];

        let mut ind: f64 = 0.0;

        //This uses the sort of elements to put rankings in their original positions
        for &(i, _) in &rx {
            ranks[i] = ind;
            ind += 1.0;
        }

        
        ranks
        
        
    }*/


    pub fn ad_calc(&self) -> f64 {

        //let time = Instant::now();
        let mut forward: Vec<f64> = self.resids();
        let mut extras: Vec<f64> = self.extraneous_resids();
        forward.append(&mut extras);
        drop(extras);
        forward.par_sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let n = forward.len();
        let mut a_d = -(n as f64);
        for i in 0..n {

            let cdf = self.background.ln_cdf(forward[i]);
            let rev_sf = self.background.ln_sf(forward[n-1-i]);
            a_d -= (cdf+rev_sf) * ((2*i+1) as f64)/(n as f64)
        }
        a_d
    }

    
    /*pub fn ad_grad(&self) -> Vec<f64> {

        //let start = Instant::now();
        let ranks = self.rank();

        let n = self.resids.len() as f64;
       
        let mut derivative: Vec<f64> = vec![0.0; self.resids.len()];

        for i in 0..self.resids.len() {
            let pdf = self.background.pdf(self.resids[i]);
            let (cdf, sf) = self.background.cd_and_sf(self.resids[i]);
            let mut poc = pdf/cdf;
            if !poc.is_finite() {poc = 0.;} //We assume that pdf outpaces cdf to 0 mathematically, but some cdf implementations are lazier sooner
            let mut pos = pdf/sf;
            if !pos.is_finite() {pos = 0.;} //We assume that pdf outpaces sf to 0 mathematically, but some sf implementations are lazier sooner

            //derivative[i] = -((2.*ranks[i]+1.)*pdf)/(n*cdf)+((2.*n-(2.*ranks[i]+1.))*pdf)/(n*sf);
            derivative[i] = -((2.*ranks[i]+1.)*poc)/(n)+((2.*n-(2.*ranks[i]+1.))*pos)/(n);
        }


        //println!("ad_grad {:?}", start.elapsed());
        derivative

    }*/

    //My low approximation is screwed up. It's derived from the low tail approximation of the
    //marsaglia (2004) paper
    fn low_val(la: f64) -> f64 {

        const C: f64 = PI*PI/8.0;
        -C/la-2.5*la.ln()+Self::polynomial_sqrt_la(la).ln()
    }

    fn deriv_low_val(la: f64) -> f64 {

        const C: f64 = PI*PI/8.0;
        C/(la*la)- 2.5/la + Self::deriv_polynomial_sqrt_la(la)/Self::polynomial_sqrt_la(la)

    }

    fn polynomial_sqrt_la(la: f64) -> f64 {

        const C: f64 = PI*PI/8.0;
        const CFS: [f64; 6] = [2.00012,0.247105,0.0649821, 0.0347962,0.0116720,0.00168691];

        let mut p: f64 = C*CFS[0]+(C*CFS[1]-CFS[0]/2.)*la.powi(1)+4.5*CFS[5]*la.powi(6);

        for i in 2..CFS.len() {
            p += (-1_f64).powi((i+1) as i32)*(C*CFS[i]-((2*i-3) as f64)*CFS[i-1]/2.)*la.powi(i as i32);
        }
        return p

    }

    fn deriv_polynomial_sqrt_la(la:f64) -> f64 {

        const C: f64 = PI*PI/8.0;
        const CFS: [f64; 6] = [2.00012,0.247105,0.0649821, 0.0347962,0.0116720,0.00168691];

        let mut p: f64 = (C*CFS[1]-CFS[0]/2.)+27.*CFS[5]*la.powi(5);

        for i in 2..CFS.len() {
            p+= (-1_f64).powi((i+1) as i32)*(i as f64)*(C*CFS[i]-((2*i-3) as f64)*CFS[i-1]/2.)*la.powi((i-1) as i32);
        }

        return p

    }

    fn high_val(ha: f64) -> f64 {

        (3.0/(ha*PI)).ln()/2.0-ha

    }

    fn deriv_high_val(ha: f64) -> f64 {
        -0.5/ha-1.
    }

    fn weight_fn(a: f64) -> f64 {
        const A0: f64 = 2.0;
        const K: f64 = 16.0;
        1.0/(1.0+(a/A0).powf(K))
    }

    fn deriv_weight_fn(a: f64) -> f64 {
        const A0: f64 = 2.0;
        const K: f64 = 16.0;
        let aa = a/A0;
        -(K/A0)*(aa.powf(K-1.)/(1.+aa.powf(K)).powi(2))
    }

    pub fn ad_like(a: f64) -> f64 {


        //This has to exist because of numerical issues causing NANs otherwise
        //This is basically caused by cdf implementations that return 0.0.
        //My FastT implementation can't, but the statrs normal DOES. 
        if a == f64::INFINITY { return -f64::INFINITY;} 

        let lo = Self::low_val(a);
        let hi = Self::high_val(a);
        let w = Self::weight_fn(a);

        w*lo+(1.0-w)*hi
    }

    pub fn ad_diff(a: f64) -> f64 {

        const H: f64 = 0.0000001;
        (Self::ad_like(a+H)-Self::ad_like(a))/H
    }

    pub fn ad_deriv(a: f64) -> f64 {
        
        if a.is_infinite() {
            return 0.0;
        }

        let w = Self::weight_fn(a);
        let dl = Self::deriv_low_val(a);
        let dh = Self::deriv_high_val(a);
        let dw = Self::deriv_weight_fn(a);
        let lv = Self::low_val(a);
        let hv = Self::high_val(a);


        //println!("a: {}, w: {}, dl: {}, hl: {}, dw: {}, lv: {}, hv: {}", a, w, dl, hl, dw, lv, hv);
        //w*Self::deriv_low_val(a)+(1.-w)*Self::deriv_high_val(a)
          //  +Self::deriv_weight_fn(a)*(Self::low_val(a)-Self::high_val(a))
        w*dl+(1.-w)*dh + dw*(lv-hv)
       
    }


}

impl Mul<&Vec<f64>> for &Noise<'_> {

    type Output = f64;

    //fn mul(self, rhs: &Noise) -> f64 {
    fn mul(self, rhs: &Vec<f64>) -> f64 {

        //let rhs_r = rhs.resids();

        //if(self.resids.len() != rhs_r.len()){
        if self.resids.len() != rhs.len() {
            panic!("Residuals aren't the same length?!")
        }

        //self.resids.iter().zip(rhs_r).map(|(a,b)| a*b).sum()
        //self.resids.iter().zip(rhs).map(|(a,b)| a*b).sum()

        let mut sum = 0.0;

        for i in 0..self.resids.len() {
            sum += self.resids[i]*rhs[i];
        }

        sum
    }
}



//I had to take the source code for below from the aberth crate and modify it for vectors


pub fn aberth_vec(polynomial: &Vec<f64>, epsilon: f64) -> Result<Vec<Complex<f64>>, &'static str> {
  let dydx = &vec_derivative(polynomial); //Got up to here 
  let mut zs: Vec<Complex<f64>> = initial_guesses(polynomial);
  let mut new_zs = zs.clone();

  let iter: usize = ((1.0/epsilon) as usize)+1;
  'iteration: for _ in 0..iter {
    let mut converged = true;

    for i in 0..zs.len() {
      let p_of_z = sample_polynomial(polynomial, zs[i]);
      let dydx_of_z = sample_polynomial(dydx, zs[i]);
      let sum = (0..zs.len())
        .filter(|&k| k != i)
        .fold(Complex::new(0.0f64, 0.0), |acc, k| {
          acc + Complex::new(1.0f64, 0.0) / (zs[i] - zs[k])
        });

      let new_z = zs[i] + p_of_z / (p_of_z * sum - dydx_of_z);
      new_zs[i] = new_z;

      if new_z.re.is_nan()
        || new_z.im.is_nan()
        || new_z.re.is_infinite()
        || new_z.im.is_infinite()
      {
        break 'iteration;
      }

      if !new_z.approx_eq(zs[i], epsilon) {
        converged = false;
        }
    }
    if converged {
      return Ok(new_zs);
    }
    core::mem::swap(&mut zs, &mut new_zs);
  }
  Err("Failed to converge.")
}

pub fn vec_derivative(coefficients: &Vec<f64>) -> Vec<f64> {
  debug_assert!(coefficients.len() != 0);
  coefficients
    .iter()
    .enumerate()
    .skip(1)
    .map(|(power, &coefficient)| {
      let p = power as f64 ;
      p * coefficient
    })
    .collect()
}

fn initial_guesses(polynomial: &Vec<f64>)-> Vec<Complex<f64>> {
  // the degree of the polynomial
  let n = polynomial.len() - 1;
  let n_f = n as f64;
  // convert polynomial to monic form
  let mut monic: Vec<f64> = Vec::new();
  for c in polynomial {
    monic.push(*c / polynomial[n]);
  }
  // let a = - c_1 / n
  let a = -monic[n - 1] / n_f;
  // let z = w + a,
  let p_of_w = {
    // we can recycle monic on the fly.
    for coefficient_index in 0..=n {
      let c = monic[coefficient_index];
      monic[coefficient_index] = 0.0f64;
      for ((index, power), pascal) in zip(
        zip(0..=coefficient_index, (0..=coefficient_index).rev()),
        aberth::PascalRowIter::new(coefficient_index as u32),
      ) {
        let pascal: f64 = pascal as f64;
        monic[index] =
          MulAdd::mul_add(c, pascal * a.powi(power as i32), monic[index]);
      }
    }
    monic
  };
  // convert P(w) into S(w)
  let s_of_w = {
    let mut p = p_of_w;
    // skip the last coefficient
    for i in 0..n {
      p[i] = -p[i].abs()
    }
    p
  };
  // find r_0
  let mut int = 1.0f64;
  let r_0 = loop {
    let s_at_r0 = aberth::sample_polynomial(&s_of_w, int.into());
    if s_at_r0.re > 0.0f64 {
      break int;
    }
    int = int + 1.0f64;
  };
  drop(s_of_w);

  {
    let mut guesses: Vec<Complex<f64>> = Vec::new();

    let frac_2pi_n = f64::PI() * 2.0 / n_f;
    let frac_pi_2n = f64::PI() / 2.0 / n_f;

    for k in 0..n {
      let k_f: f64 = k as f64; 
      let theta = MulAdd::mul_add(frac_2pi_n, k_f, frac_pi_2n);

      let real = MulAdd::mul_add(r_0, theta.cos(), a);
      let imaginary = r_0 * theta.sin();

      let val = Complex::new(real, imaginary);
      guesses.push(val) ;
    }

    guesses
  }
}

pub fn sample_polynomial(coefficients: &Vec<f64>, x: Complex<f64>) -> Complex<f64> {
  debug_assert!(coefficients.len() != 0);
  let mut r = Complex::new(0.0f64, 0.0);
  for c in coefficients.iter().rev() {
    r = r.mul_add(x, c.into())
  }
  r
}

trait ComplexExt<F: Float> {
  fn approx_eq(self, w: Self, epsilon: F) -> bool;
}

impl<F: Float> ComplexExt<F> for Complex<F> {
  /// Cheap comparison of complex numbers to within some margin, epsilon.
  #[inline]
  fn approx_eq(self, w: Complex<F>, epsilon: F) -> bool {
    ((self.re - w.re).powi(2)+(self.im - w.im).powi(2)).sqrt() < epsilon
  }
}

trait Distance<F: Float> {
    fn dist(self, b: Self) -> F;
}

impl<F: Float> Distance<F> for Complex<F> {
  /// Cheap comparison of complex numbers to within some margin, epsilon.
  #[inline]
  fn dist(self, b: Complex<F>) -> F {
    ((self.re - b.re).powi(2)+(self.im - b.im).powi(2)).sqrt() 
  }
}


    



#[cfg(test)]
mod tests{
   
    use super::*;
    use crate::sequence::{Sequence, NullSequence};
    use rand::Rng;
    use std::time::Instant;

    fn empirical_noise_grad(n: &Noise) -> Vec<f64>{

        let h = 0.00001;
        let back = n.background;
        let ad = n.ad_calc();
        let mut grad = vec![0_f64; n.resids.len()]; 
        for i in 0..grad.len() {
             let mut n_res = n.resids();
             n_res[i] += h;
             let nnoise = Noise::new(n_res, vec![], back);
             grad[i] = (nnoise.ad_calc()-ad)/h;
        }
        grad
    }

    #[test]
    fn wave_check(){

        let sd = 5;
        let height = 2.0;
        let spacer = 5;
        let k = Kernel::new(sd as f64, height);

        let kern = k.get_curve();
        let kernb = &k*4.0;


        println!("kern len {}", (kern.len()));
        assert!(kern.len() == 6*(sd as  usize)+1);

        assert!(kern.iter().zip(kernb.get_curve()).map(|(&a,b)| ((b/a)-4.0).abs() < 1e-6).fold(true, |acc, mk| acc && mk));

        assert!((k.get_sd()-(sd as f64)).abs() < 1e-6);
    }

    #[test]
    fn real_wave_check(){
        let k = Kernel::new(5.0, 2.0);
        //let seq = Sequence::new_manual(vec![85;56], vec![84, 68, 72]);
        let seq = Sequence::new_manual(vec![192, 49, 250, 10, 164, 119, 66, 254, 19, 229, 212, 6, 240, 221, 195, 112, 207, 180, 135, 45, 157, 89, 196, 117, 168, 154, 246, 210, 245, 16, 97, 125, 46, 239, 150, 205, 74, 241, 122, 64, 43, 109, 17, 153, 250, 224, 17, 178, 179, 123, 197, 168, 85, 181, 237, 32], vec![84, 68, 72]);
        let mut signal = Waveform::create_zero(&seq, 5);

        let zeros: Vec<usize> = vec![0, 465, 892]; //Blocks terminate at bases 136, 737, and 1180

        let null_zeros: Vec<usize> = vec![144, 813];

        //This is in units of number of u8s
        let null_sizes: Vec<usize> = vec![78,17];

        unsafe{

        signal.place_peak(&k, 1, 20);

        let t = Instant::now();
        //kmer_propensities is tested by inspection, not asserts, because coming up with a good assert case was hard and I didn't want to
        //It passed the inspections I gave it, though
        let propens = signal.kmer_propensities(9);
        println!("duration {:?}", t.elapsed());

        //println!("{:?}", signal);
        println!("{:?}", propens);
        //Waves are in the correct spot
        assert!((signal.raw_wave()[21]-2.0).abs() < 1e-6);

        signal.place_peak(&k, 1, 2);

        println!("after second ins {:?}", signal.kmer_propensities(9));
        //Waves are not contagious
        assert!(signal.raw_wave()[0..17].iter().fold(true, |acc, ch| acc && ((ch-0.0).abs() < 1e-6)));

        //point_lens: Vec<usize>,
        //start_dats: Vec<usize>,
        //

        signal.place_peak(&k, 1, 67);

        //Waves are not contagious
        assert!(signal.raw_wave()[(signal.start_dats[2])..(signal.start_dats[2]+signal.point_lens[2])].iter().fold(true, |acc, ch| acc && ((ch-0.0).abs() < 1e-6)));

        signal.place_peak(&k, 2, 20);

        //Waves are in the correct spot
        assert!((signal.raw_wave()[35]-2.0).abs() < 1e-6);

        //This is a check just for miri
        signal.place_peak(&k, 2, 70);

        //This is a check to make sure that miri can catch it when I obviously screw up. This test is designed to have UB. Do NOT uncomment it unless you're checking that Miri can catch stuff
        //signal.place_peak(&k, 2, 456);

        }

        let base_w = &signal*0.4;


        let background: Background = Background::new(0.25, 2.64, 25.0);

        let mut rng = rand::thread_rng();

        let null_makeup: Vec<Vec<usize>> = null_sizes.iter().map(|&a| (0..(4*a)).map(|_| rng.gen_range(0_usize..4)).collect::<Vec<usize>>()).collect::<Vec<Vec<usize>>>();

        println!("{:?} null sizes", null_makeup.iter().map(|a| a.len()).collect::<Vec<_>>());

        let invented_null: NullSequence =  NullSequence::new(null_makeup);

        let data_seq = unsafe{ AllDataUse::new_unchecked_data(base_w, &invented_null, &zeros, &null_zeros, &background)};

        let noise: Noise = signal.produce_noise(&data_seq);

        let noi: Vec<f64> = noise.resids();


        let raw_resid = &signal-data_seq.data();

        let w = raw_resid.raw_wave();

        println!("Noi {:?}", noi);



        //This is based on a peak with height 1.5 after accounting for binding and an sd equal to 5*spacer
        let fake_extraneous: Vec<f64> = vec![1.50000000, 1.47029801, 1.47029801, 1.38467452, 1.38467452, 1.25290532, 1.25290532, 1.08922356, 1.08922356, 0.90979599, 0.90979599];

        let generated_extraneous = Waveform::generate_extraneous_binding(&data_seq, &[1.5]);

        println!("theoretical extra {:?}, gen extra {:?}", fake_extraneous, generated_extraneous);

        assert!(fake_extraneous.len() == generated_extraneous.len(), "Not even generating the right length of kernel from binding heights!");

        for i in 0..fake_extraneous.len() {
            assert!((fake_extraneous[i]-generated_extraneous[i]).abs() < 1e-7, "generated noise not generating correctly");
        }


    }



    #[test]
    fn noise_check(){

        
        let background: Background = Background::new(0.25, 2.64, 5.0);

        let n1 = Noise::new(vec![0.4, 0.4, 0.3, 0.2, -1.4],vec![], &background);
        let n2 = Noise::new(vec![0.4, 0.4, 0.3, -0.2, 1.4],vec![], &background);

        assert!(((&n1*&n2.resids())+1.59).abs() < 1e-6);

        println!("{:?}", n1.resids());
        println!("ad_calc {}", n1.ad_calc());

        let mut noise_arr = n1.resids();
        noise_arr.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let noise_length = noise_arr.len();
        let mut ad_try = -(noise_length as f64);

        for i in 0..noise_length {

            let ln_cdf = n1.dist().ln_cdf(noise_arr[i]);
            let ln_sf = n1.dist().ln_sf(noise_arr[noise_length-1-i]);
            let mult = (2.0*((i+1) as f64)-1.0)/(noise_length as f64);

            
            ad_try -= mult*(ln_cdf+ln_sf);
        }

        println!("ad_try {}", ad_try);

        assert!((n1.ad_calc()-ad_try).abs() < 1e-6, "AD calculation not matching theory without extraneous binding");
 
        //This is based on a peak with height 1.5 after accounting for binding and an sd equal to 5*spacer
        let fake_extraneous: Vec<f64> = vec![1.50000000, 1.47029801, 1.47029801, 1.38467452, 1.38467452, 1.25290532, 1.25290532, 1.08922356, 1.08922356, 0.90979599, 0.90979599];


        let n1_with_extraneous = n1.noise_with_new_extraneous(fake_extraneous);

        let mut extraneous_noises = n1_with_extraneous.resids();

        let mut extra_resids = n1_with_extraneous.extraneous_resids();

        extraneous_noises.append(&mut extra_resids);

        extraneous_noises.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let noise_length = extraneous_noises.len();
        let mut ad_try = -(noise_length as f64);
        
        for i in 0..noise_length {
        
            let ln_cdf = n1.dist().ln_cdf(extraneous_noises[i]);
            let ln_sf = n1.dist().ln_sf(extraneous_noises[noise_length-1-i]);
            let mult = (2.0*((i+1) as f64)-1.0)/(noise_length as f64);
        
            
            ad_try -= mult*(ln_cdf+ln_sf);
        }

        println!("extraneous ad {} ad_try {}", n1_with_extraneous.ad_calc(), ad_try);

        assert!((n1_with_extraneous.ad_calc()-ad_try).abs() < 1e-6, "AD calculation not matching theory with extraneous binding");

        //Calculated these with the ln of the numerical derivative of the fast implementation
        //of the pAD function in the goftest package in R
        //This uses Marsaglia's implementation, and is only guarenteed up to 8
        let calced_ads: [(f64, f64); 6] = [(1.0, -0.644472305368), 
                                           (0.46, 0.026743661078),
                                           (0.82, -0.357453548256),
                                           (2.82, -3.221007453503),
                                           (3.84, -4.439627768456),
                                           (4.24, -4.865014182520)];

        for pairs in calced_ads {

            //I'm considering 5% error mission accomplished for these
           //We're fighting to make sure this approximation is roughly
           //compaitble with another approximation with propogated errors
           //This will not be an exact science
            println!("Noi {} {} {} {}",Noise::ad_like(pairs.0), pairs.1, Noise::ad_like(pairs.0)-pairs.1, (Noise::ad_like(pairs.0)-pairs.1)/pairs.1);
            assert!(((Noise::ad_like(pairs.0)-pairs.1)/pairs.1).abs() < 5e-2); 

        } 

    }

    #[test]
    #[should_panic(expected = "Residuals aren't the same length?!")]
    fn panic_noise() {
        
        let background: Background = Background::new(0.25, 2.64, 5.0);
        let n1 = Noise::new(vec![0.4, 0.4, 0.3, 0.2, -1.4], vec![], &background);
        let n2 = Noise::new(vec![0.4, 0.4, 0.3, -0.2], vec![], &background);

        let _ = &n1*&n2.resids();
    }









}


