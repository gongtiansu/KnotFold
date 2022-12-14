# KnotFold
## About The Project

The implementation of the paper "Accurate prediction of RNA secondary structure including pseudoknots through solving minimum-cost flow with learned potentials".

## Getting Started
### Prerequisites
Install [PyTorch 1.6+](https://pytorch.org/),
[python
3.7+](https://www.python.org/downloads/)

### Installation

1. Clone the repo
```sh
git clone https://github.com/gongtiansu/KnotFold.git
```

2. Install python packages
```sh
cd KnotFold
pip install -r requirements.txt
```

## Usage
1. KnotFold_nn.py: predicting base pairing probability from RNA sequence (fasta format).
```sh
python KnotFold_nn.py -i <RNA_fasta> -o <output_probability> (-r) (--cuda)
```
2. KnotFold_mincostflow: constructing RNA secondary structure from base pairing probability using the minimum-cost flow algorithm.
```sh
KnotFold_mincostflow <prior_probability> <reference_probability> 
```
3. KnotFold.py: predicting RNA secondary structure (bpseq format) from RNA sequence (fasta format).  
```sh
python KnotFold.py -i <RNA_fasta> -o <output_dictionary> 
```
## Example
```sh
cd example
./run_example.sh
```

## License
Distributed under the MIT License. See `LICENSE` for more information.
