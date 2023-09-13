name=$1
day=$(date +"%Y%m%d")
ngsFile=/home/alimf/motif_finder_project/Data/NGS/$3.pair
output=/expanse/lustre/scratch/alimf/temp_project/motif_runs

export RUST_BACKTRACE=1;

cargo run --release --bin test -- ${name}_$3_$2 $output /home/alimf/motif_finder_project/Data/Fasta/MG1655.fasta true $ngsFile 498 25 > ${output}/${name}_$3_$2.out 2> ${output}/${name}_$3_$2.err

