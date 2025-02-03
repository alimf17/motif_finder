
use std::fs;
use std::io::Read;


use motif_finder::ECOLI_FREQ;
use motif_finder::base::*;
use motif_finder::waveform::*;




use motif_finder::data_struct::{AllData, AllDataUse};

use statrs::distribution::Continuous;






 
use rand::prelude::*;




use plotters::prelude::*;
use plotters::prelude::full_palette::ORANGE;


fn main() {


    let args: Vec<String> = std::env::args().collect();

    let file_out = args[1].clone();

    let motif_set_meme = args[2].clone();

    let meme_name = motif_set_meme.clone();

    let mut try_bincode = fs::File::open(file_out).unwrap();

    let mut buffer: Vec<u8> = Vec::new();
    let _ = try_bincode.read_to_end(&mut buffer);//We don't need to handle this specially, because this will create a different warning later
    let pre_data: AllData = bincode::deserialize(&buffer).unwrap();

    let data = AllDataUse::new(&pre_data, 0.0).unwrap();

    buffer.clear();

    let mut rng = rand::thread_rng();

    println!("Freq change");
    let mut mot_set_meme: MotifSet = MotifSet::set_from_meme(&motif_set_meme, &data,None, f64::INFINITY, false, &mut rng).unwrap();

    let mut rmse = f64::INFINITY;

    let len = mot_set_meme.len();

    for id in 0..len {
        (mot_set_meme, rmse) = mot_set_meme.best_height_set_from_rmse_noise(id).unwrap();
    }

    println!("Best set RMSE {}\n set {:?} {} {}", rmse,mot_set_meme, mot_set_meme.ln_prior(), mot_set_meme.nth_motif(0).pwm_prior(data.data().seq()));


    let motif_set_bin = args[3].clone();

    let bin_name = motif_set_bin.clone();

    println!("mot set bin {motif_set_bin}");

    let mut try_bincode = fs::File::open(motif_set_bin).unwrap();

    let mut buffer: Vec<u8> = Vec::new();
    let _ = try_bincode.read_to_end(&mut buffer);//We don't need to handle this specially, because this will create a different warning later

    let prep_mot_set_bin: StrippedMotifSet = bincode::deserialize(&buffer).unwrap();

    let mut mot_set_bin = prep_mot_set_bin.reactivate_set(&data);

    mot_set_bin.sort_by_height();

    let wave_file = format!("{}_waves", args[4].clone());



    mot_set_meme.save_set_trace_comparisons(&mot_set_bin, &wave_file, "compare_wave", &args[5], &args[6]);
    
    if args.len() > 7 {

        for i in 7..args.len() {

            let num_motifs: usize = args[i].parse().unwrap();
            let sub_bin = mot_set_bin.only_n_strongest(num_motifs);

            mot_set_meme.save_set_trace_comparisons(&sub_bin, &wave_file, "compare_sub_wave", &args[5], &format!("TARJIM {num_motifs} Motifs"));


        }

    }



}
pub fn mod_save_set(current_active: &MotifSet, data_ref: &AllDataUse, signal_file: &str) {


        let locs = data_ref.data().generate_all_locs();

        let plot = BitMapBackend::new(signal_file, (3300, 1500)).into_drawing_area();
        
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
            .build_cartesian_2d(0_f64..(*locs.last().unwrap() as f64), (-16_f64)..16_f64).unwrap();


        chart.configure_mesh()
            .x_label_style(("sans-serif", 40))
            .y_label_style(("sans-serif", 40))
            .x_label_formatter(&|v| format!("{:.0}", v))
            .x_desc("Genome Location (Bp)")
            .y_desc("Signal Intensity")
            .disable_mesh().draw().unwrap();

        const HORIZ_OFFSET: i32 = -5;

        chart.draw_series(data_ref.data().read_wave().iter().zip(locs.iter()).map(|(&k, &i)| Circle::new((i as f64, k),2_u32, Into::<ShapeStyle>::into(&BLACK).filled()))).unwrap().label("Occupancy Data").legend(|(x,y)| Circle::new((x+2*HORIZ_OFFSET,y),5_u32, Into::<ShapeStyle>::into(&BLACK).filled()));
        
        let signal = current_active.recalced_signal();

        let current_resid = data_ref.data()-&signal;
        
        chart.draw_series(LineSeries::new(signal.read_wave().iter().zip(locs.iter()).map(|(&k, &i)| (i as f64, k)), BLUE.filled())).unwrap().label("Motif Set Occupancy").legend(|(x, y)| Rectangle::new([(x+4*HORIZ_OFFSET, y-4), (x+4*HORIZ_OFFSET + 20, y+3)], Into::<ShapeStyle>::into(&BLUE).filled()));

        const THICKEN: usize = 20;
        for j in 1..=THICKEN{
        chart.draw_series(LineSeries::new(signal.read_wave().iter().zip(locs.iter()).map(|(&k, &i)| (i as f64, k+0.01*2_f64.powf(k*0.15)*(j as f64))), BLUE.filled())).unwrap();

        }
        chart.configure_series_labels().position(SeriesLabelPosition::LowerRight).margin(40).legend_area_size(10).border_style(&BLACK).label_font(("Calibri", 40)).draw().unwrap();

        let _max_abs_resid = current_resid.read_wave().iter().map(|&a| a.abs()).max_by(|x,y| x.partial_cmp(y).unwrap()).unwrap();

        let abs_resid: Vec<(f64, f64)> = current_resid.read_wave().iter().zip(signal.read_wave().iter()).map(|(&a, _)| {

            let tup = data_ref.background_ref().cd_and_sf(a);
            if tup.0 >= tup.1 { (tup.0-0.5)*2.0 } else {(tup.1-0.5)*2.0} } ).zip(locs.iter()).map(|(a, &b)| (a, b as f64)).collect();
        
        let mut map = ChartBuilder::on(&lower)
            .set_label_area_size(LabelAreaPosition::Left, 100)
            .set_label_area_size(LabelAreaPosition::Bottom, 50)
            .build_cartesian_2d(0_f64..(*locs.last().unwrap() as f64), 0_f64..1_f64).unwrap();

        map.configure_mesh().x_label_style(("sans-serif", 0)).y_label_style(("sans-serif", 0)).x_desc("Deviance").axis_desc_style(("sans-serif", 40)).set_all_tick_mark_size(0_u32).disable_mesh().draw().unwrap();


        map.draw_series(abs_resid.windows(2).map(|x| Rectangle::new([(x[0].1, 0.0), (x[1].1, 1.0)], derived_color.get_color(x[0].0).filled()))).unwrap();


}




