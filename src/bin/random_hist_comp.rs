use std::fs;
use std::io::Read;


use motif_finder::ECOLI_FREQ;
use motif_finder::base::*;
use motif_finder::waveform::*;




use motif_finder::data_struct::{AllData, AllDataUse};

use statrs::distribution::Continuous;




use std::env::args;


use rand::prelude::*;

use rayon::prelude::*;



use plotters::prelude::*;
use plotters::prelude::full_palette::ORANGE;
use plotters::coord::types::{RangedSlice, RangedCoordf64};
use plotters::coord::Shift;


fn main() {

    let args: Vec<String> = std::env::args().collect();


    let file_out = args[1].clone();

    let mut try_bincode = fs::File::open(file_out).unwrap();

    let mut buffer: Vec<u8> = Vec::new();
    let _ = try_bincode.read_to_end(&mut buffer);//We don't need to handle this specially, because this will create a different warning later
    let pre_data: AllData = bincode::deserialize(&buffer).unwrap();

    let data = AllDataUse::new(&pre_data, 0.0).unwrap();


    let dists: Vec<f64> = (0..10000).into_par_iter().map(|_| {

        let mut rng = rand::thread_rng();

        let mot_0 = Motif::rand_mot(data.data().seq(), data.height_dist(), &mut rng);
        let mot_1 = Motif::rand_mot(data.data().seq(), data.height_dist(), &mut rng);

        let binds_0 = mot_0.return_bind_score(data.data().seq());
        let binds_1 = mot_1.return_bind_score(data.data().seq());
        
        binds_0.into_iter().zip(binds_1.into_iter()).map(|(a,b)| (a.exp2()-b.exp2()).powi(2)).sum::<f64>()
        //binds_0.into_iter().zip(binds_1.into_iter()).map(|(a,b)| (a.exp2()-b.exp2()).powi(2)).sum::<f64>().log2()

    }).collect();

    let plotting = BitMapBackend::new("RandomDist.png", (8000, 4000)).into_drawing_area();


    plotting.fill(&WHITE).expect("This should just work");

    let _ = quick_hist_local(&dists , &plotting, "Random Sample Of Score Distances".to_string(), 100);


}


fn quick_hist_local<'a, 'b, DB: DrawingBackend, N: Copy+Into<f64>>(raw_data: &[N], area: &'a DrawingArea<DB, Shift>, label: String, num_bins: usize) -> ChartBuilder<'a, 'b, DB> {

    let mut hist = ChartBuilder::on(area);

    hist.margin(10).y_label_area_size(400).x_label_area_size(300);

    hist.caption(label, ("Times New Roman", 200));

    let data: Vec<f64> = raw_data.iter().map(|&a| a.into()).collect();

    let (xs, hist_form) = build_hist_bins_local(data, num_bins);

    let range = RangedSlice::from(xs.as_slice());

    let max_prob = hist_form.iter().map(|&x| x.1).fold(0_f64, |x,y| x.max(y));

    println!("max prob {max_prob}");

    let mut hist_context = hist.build_cartesian_2d(range, 0_f64..max_prob).unwrap();

    hist_context.configure_mesh().x_label_style(("serif", 150))
        .y_label_style(("serif", 150))
        .x_label_formatter(&|v| format!("{:.0}", v))
        .axis_desc_style(("serif",150))
        .x_desc("Agreement score of two random valid motifs")
        .y_desc("Probability").disable_x_mesh().disable_y_mesh().x_label_formatter(&|x| format!("{:.04}", *x)).draw().unwrap();

    //hist_context.draw_series(Histogram::vertical(&hist_context).style(CYAN.filled()).data(trial_data.iter().map(|x| (x, inverse_size)))).unwrap();
    hist_context.draw_series(Histogram::vertical(&hist_context).style(BLUE.mix(0.8).filled()).margin(0).data(hist_form.iter().map(|x| (&x.0, x.1)))).unwrap();//.label("Proposed Moves").legend(|(x, y)| Rectangle::new([(x-20, y-10), (x, y+10)], Into::<ShapeStyle>::into(&CYAN).filled()));


    //hist_context.configure_series_labels().position(SeriesLabelPosition::UpperRight).margin(40).legend_area_size(20).border_style(&BLACK).label_font(("Times New Roman", 80)).draw().unwrap();

    hist


}

fn build_hist_bins_local(mut data: Vec<f64>, num_bins: usize) -> (Vec<f64>, Vec<(f64, f64)>) {
    
    let length = data.len() as f64;

    let mut big_data = data.clone();                
    
    data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    
    big_data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    /*let min = big_data[0];
    let max = *big_data.last().unwrap();

    let step = (max-min)/(num_bins as f64);

    let add_into = 1./length;
    
    let xs = (0..num_bins).map(|i| (min+(i as f64)*step)).collect::<Vec<_>>();
*/
    let min = big_data[0];
    let max = *big_data.last().unwrap();

    let step = (max/min).powf(1.0/(num_bins as f64));

    let add_into = 1./length;

    let xs = (0..num_bins).map(|i| (min*(i as f64).powf(step))).collect::<Vec<_>>();


    let mut bins: Vec<(f64, f64)> = xs.iter().clone().map(|&x| (x, 0.0)).collect();

    
    let mut j: usize = 1;
    let mut k: usize = 1;
    for &dat in data.iter() {

        //Because data and bins are sorted, we only need to find the first bin
        //where the data is less than the top end. We use short circuit && to prevent overflow
        while (j < (bins.len()-1)) && (dat >= bins[j+1].0) { j+= 1;}

        bins[j].1 += add_into;

        //bins[j].1 *= mul_into;
    }

    (xs, bins)

}


