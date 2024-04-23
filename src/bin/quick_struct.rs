
use std::fs;
use std::io::Read;

use motif_finder::base::*;
use motif_finder::base::{SQRT_2, SQRT_3};

use motif_finder::{NECESSARY_MOTIF_IMPROVEMENT};

use motif_finder::data_struct::{AllData, AllDataUse};

use log::warn;

use kmedoids;

use ndarray::prelude::*;

use rand::prelude::*;

use regex::Regex;

use poloto;
use plotters::prelude::*;
use plotters::prelude::full_palette::ORANGE;


fn main() {

    let file_out = "/Users/afarhat/Downloads/NC_000913.2_GSM639836_TrpR_Trp_ln_ratio_25_data.bin";
    let trace_file =  "/Users/afarhat/Downloads/removedHMC_20240407_GSM639836_TrpR_Trp_ln_ratio_D_1_trace_from_step_0999999.bin";

    //let file_out = "/Users/afarhat/Downloads/NC_000913.2_GSM639826_ArgR_Arg_ln_ratio_1_25_data.bin";
    //let trace_file =  "/Users/afarhat/Downloads/removedHMC_20240407_GSM639826_ArgR_Arg_ln_ratio_1_D_1_trace_from_step_0533999.bin";

    //let file_out = "/Users/afarhat/Downloads/NC_000913.2_ArgR_Arg_TrpR_Trp_ln_ratio_25_data.bin";
    //let trace_file =  "/Users/afarhat/Downloads/ReportingReplicaExchange_20240421_ArgR_Arg_TrpR_Trp_ln_ratio_D_1_trace_from_step_0000004.bin";

    let mut try_bincode = fs::File::open(file_out).unwrap();
    let mut try_bin_trace = fs::File::open(trace_file).unwrap();

    let mut buffer: Vec<u8> = Vec::new();
    let _ = try_bincode.read_to_end(&mut buffer);//We don't need to handle this specially, because this will create a different warning later
    let pre_data: AllData = bincode::deserialize(&buffer).unwrap();

    let data = AllDataUse::new(&pre_data).unwrap();

    buffer.clear();

    let _ = try_bin_trace.read_to_end(&mut buffer);
    let pre_trace: SetTraceDef = bincode::deserialize(&buffer).unwrap();

    let mut rng = rand::thread_rng();

    let trace = pre_trace.get_set_trace(&data, &mut rng, None);


    mod_save_trace(&trace, &data);

    println!("{:?}", trace.loan_active());

}

pub fn mod_save_trace(trace: &SetTrace, data_ref: &AllDataUse) {

        let current_active = trace.loan_active();

        let locs = data_ref.data().generate_all_locs();

        let signal_file = "/Users/afarhat/Downloads/traceTrial.png";

        let plot = BitMapBackend::new(signal_file, (3000, 1200)).into_drawing_area();

        plot.fill(&WHITE).unwrap();

        let (upper, lower) = plot.split_vertically((86).percent_height());

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

        const horiz_offset: i32 = -5;

        chart.draw_series(data_ref.data().read_wave().iter().zip(locs.iter()).map(|(&k, &i)| Circle::new((i as f64, k),2_u32, Into::<ShapeStyle>::into(&BLACK).filled()))).unwrap().label("Occupancy Data").legend(|(x,y)| Circle::new((x+2*horiz_offset,y),5_u32, Into::<ShapeStyle>::into(&BLACK).filled()));
        
        let signal = current_active.recalced_signal();

        let current_resid = data_ref.data()-&signal;
        
        chart.draw_series(LineSeries::new(signal.read_wave().iter().zip(locs.iter()).map(|(&k, &i)| (i as f64, k)), BLUE.filled())).unwrap().label("Motif Set Occupancy").legend(|(x, y)| Rectangle::new([(x+4*horiz_offset, y-4), (x+4*horiz_offset + 20, y+3)], Into::<ShapeStyle>::into(&BLUE).filled()));

        const THICKEN: usize = 20;
        for j in 1..=THICKEN{
        chart.draw_series(LineSeries::new(signal.read_wave().iter().zip(locs.iter()).map(|(&k, &i)| (i as f64, k+0.01*2_f64.powf(k*0.15)*(j as f64))), BLUE.filled())).unwrap();

        }
        chart.configure_series_labels().position(SeriesLabelPosition::LowerRight).margin(40).legend_area_size(10).border_style(&BLACK).label_font(("Calibri", 40)).draw().unwrap();

        let max_abs_resid = current_resid.read_wave().iter().map(|&a| a.abs()).max_by(|x,y| x.partial_cmp(y).unwrap()).unwrap();

        let abs_resid: Vec<(f64, f64)> = current_resid.read_wave().iter().zip(signal.read_wave().iter()).map(|(a, b)| (a/b).abs()).zip(locs.iter()).map(|(a, &b)| (a, b as f64)).collect();
        
        let mut map = ChartBuilder::on(&lower)
            .set_label_area_size(LabelAreaPosition::Left, 100)
            .set_label_area_size(LabelAreaPosition::Bottom, 50)
            .build_cartesian_2d(0_f64..(*locs.last().unwrap() as f64), 0_f64..1_f64).unwrap();

        map.configure_mesh().x_label_style(("sans-serif", 0)).y_label_style(("sans-serif", 0)).x_desc("Proportion Residual Error").axis_desc_style(("sans-serif", 40)).set_all_tick_mark_size(0_u32).disable_mesh().draw().unwrap();

        let derived_color = DerivedColorMap::new(&[WHITE,YELLOW, RED]);

        map.draw_series(abs_resid.windows(2).map(|x| Rectangle::new([(x[0].1, 0.0), (x[1].1, 1.0)], derived_color.get_color(x[0].0/max_abs_resid).filled()))).unwrap();


}




