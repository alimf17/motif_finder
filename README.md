# The Algorithm for Reversible Jump Inference of Motifs (TARJIM) 

A tool for inferring motif number and identity from DNA-protein binding data.

## Requirements
 
Technically, none, other than [Cargo]. However, some of the analysis may use [FIMO], and thus the MEME suite. Not having the MEME suite will not cause any inference to fail, but will limit some of the analysis.

In general, the generated files for preprocessing need about 2GB of space. 

## What this is

This is a tool that takes DNA-protein binding data, such as ChIP-chip, ChIP-seq, ATAC-seq, IPOD, etc, along with a corresponding sequence, and outputs a proposed set of sequence motifs that are represented in the data.

## How to use

First, check out this repository and run

```bash
cargo build --release
```

Download or create a data file for your DNA-protein data, along with the FASTA file containing your relevant genome. There are three steps for every run: preprocessing, the run itself, and post processing.

### Preprocessing

```bash 
Usage: preprocess [OPTIONS] --name <NAME> --output-dir <OUTPUT_DIR> --fasta <FASTA> --data <DATA> --length-of-fragment <LENGTH_OF_FRAGMENT> --spacing <SPACING> --min-height <MIN_HEIGHT> --prior <PRIOR>`

Options:
  -n, --name <NAME>
          Sets the name of your output file
  -o, --output-dir <OUTPUT_DIR>
          Sets the directory location of your output file
  -f, --fasta <FASTA>
          Sets the name of the fasta file for your inference
  -d, --data <DATA>
          Sets the name of your data file for your inference. This data file must have the header "loc data", followed by rows which start with an integer, then a space, then a float. For the inference to produce a correct result, the floats should be the log2 ratio of some experimental condition to some control: eg for ChIP-chip log2(IP/mock)
  -c, --circular
          Indicates whether your genome is circular (true) or linear (false)
  -l, --length-of-fragment <LENGTH_OF_FRAGMENT>
          Gives the average length of the DNA fragments in your prep, in base pairs. This should come out to about half the width of your singleton peaks. Note the word "singleton": a "peak" might actually be multiple binding kernels glommed next to each other, so estimate low
  -s, --spacing <SPACING>
          A positive integer corresponding to the spacing between data points, in base pairs. Note that this shoud probably be close to whatever spacing you have for your probes in your data file: we linearly interpolate if points are missing, but we only linearly interpolate. If this is 0 or not less than length_of_fragment, the preprocessing will panic
  -m, --min-height <MIN_HEIGHT>
          The minimum height of your inference. If this is less than 1.0, it's set to 1.0
  -p, --prior <PRIOR>
          The penalty on increasing your number of inferred motifs. It corresponds to -ln(1-p) in a geometric distribution. If this is not positive, the preprocessing will panic
  -h, --height-scale <HEIGHT_SCALE>
          A scaling of the peak height cutoff for determining whether a region is considered peaky or not. More than 1.0 means being stricter about peaks, less than 1.0 means being laxer about calling a region peak-y. If this is not supplied or infinite, we set it to 1.0. If this is negative, we take the absolute value
  -h, --help
          Print help
  -V, --version
          Print version
```
### TARJIM

```bash
Usage: tarjim [OPTIONS] --name <NAME> --input <INPUT> --output <OUTPUT> --advances <ADVANCES> --beta <BETA> --trace-num <TRACE_NUM>`

Options:
  -n, --name <NAME>
          Sets the name of your run. Note that if you want multiple runs to be considered in parallel, use the syntax `<name>_<letter starting from A>` for your names. If you want runs to be considered as sequential continuations, use the syntax `<name with possible letter>_<number starting from 0>`

  -i, --input <INPUT>
          Sets the input file from preprocessing for your run to infer on

  -o, --output <OUTPUT>
          Sets the output directory for your run

  -a, --advances <ADVANCES>
          Number of advances you want the algorithm to run for. Note, this is in units of number of parallel tempering swaps, which require a number of standard Monte Carlo steps beforehand

  -b, --beta <BETA>
          The minimum thermodynamic beta for the chains. The absolute value of this will be taken, and if you give a number > 1.0, the reciprocal will be taken instead. I have personally seen success with beta = 1/64 when I have 126 intermediate traces. If your chain is accepting too many swaps and not having high acceptance in the smallest beta threads, make this closer to 0. If your chain is not swapping and has many acceptances in the smallest beta threads, make this closer to 1

  -t, --trace-num <TRACE_NUM>
          The number of intermediate traces between the beta = 1.0 thread and the minimum beta you supplied. This number + 2 is also the maximum number of parallel threads we can use productively

  -c, --condition-type <CONDITION_TYPE>
          Possible values:
          - meme:    Initial condition is the name of a meme file
          - bincode: Initial condition is the name of a serialized bincode of a StrippedMotifSet
          - json:    Initial condition is the name of a serialized JSON of a StrippedMotifSet

  -f, --file-initial <FILE_INITIAL>
          

  -s, --starting-tf <STARTING_TF>
          This sets an initial guess on the number of transcription factors. It will be ignored if you supply a valid initial condition

  -h, --help
          Print help (see a summary with '-h')

  -V, --version
          Print version
```
### Post processing

```bash
Usage: `process [OPTIONS] --output <OUTPUT> --num-chains <NUM_CHAINS> --max-runs <MAX_RUNS> <BASE_FILE>`

Arguments:
  <BASE_FILE>  This is prefix your runs share. When doing this processing, we assume that all of your outputted bin files are of the form "{output}/{base_file}_{<Letter starting from A and going up>}_ {<0-indexed count of runs>}_trace_from_step_{<7 digit number, with leading zeros>}.bin"

Options:
  -o, --output <OUTPUT>            This is the directory your run output to
  -n, --num-chains <NUM_CHAINS>    This is the number of independent chains run on this inference. If this is larger than 26, we will only take the first 26 chains
  -m, --max-runs <MAX_RUNS>        This is the number of sequential runs per chain. For example, if the last output file of the initial run of chain "A" is "{output}/{base_file}_A_0_trace_from_step_0100000.bin", it would be immediately followed by "{output}/{base_file}_A_1_trace_from_step_0000000.bin"
  -d, --discard-num <DISCARD_NUM>  Number of sequential runs per chain to discard as burn in. If this is not provided, it's set as 0. If this is larger than max_runs, this script will panic
  -f, --fasta-file <FASTA_FILE>    This is an optional argument to pick a genome to run FIMO against If this is not supplied, FIMO will not be run
  -h, --help                       Print help
  -V, --version                    Print version 
```
## Contributing

Please submit PRs for any minor fixes. For any major changes, please open an issue
before submitting a PR.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
