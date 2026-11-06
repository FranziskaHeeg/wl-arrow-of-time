# Temporal Graph Kernels for Classifying Dissemination Processes ###
Source code for our [SIAM International Conference on Data Mining (SDM20)](https://www.siam.org/conferences/cm/conference/sdm20) paper "[Temporal Graph Kernels for Classifying Dissemination Processes](https://arxiv.org/abs/1911.05496)".

## Compile
For compilation you need the Eigen3 library. 
Set the environment variable EIGEN3_INCLUDE_DIR to the Eigen3 folder.
For example:

    export EIGEN3_INCLUDE_DIR=/home/user/Downloads/eigen-3.3.7

Next create a folder "release" in "tgkernel". Change into this folder and run

    cmake -DCMAKE_BUILD_TYPE=Release ..

Finally, run "make" to compile the code and get the executable file "tgkernel".

## Usage

    ./tgkernel <path to dataset> <action> <kmin> <kmax>

## Data Sets
See [Benchmark Data Sets for Graph Kernels](https://graphlearning.io/) for data sets.

The data sets use endings "_ct1" and "_ct2" for classification task 1 and 2.

## Terms and conditions
Please feel free to use our code. We only ask that you cite:

	@inproceedings{Oettershagen+2020,
	    title={Temporal Graph Kernels for Classifying Dissemination Processes},
	    author={Lutz Oettershagen and Nils Kriege and Christopher Morris and Petra Mutzel},
	    booktitle={Proceedings of the 2020 SIAM International Conference on Data Mining},
	    pages={496--504},
	    year={2020}
	}



## Contact Information
If you have any questions, send an email to Lutz Oettershagen (lutz.oettershagen at cs.uni-bonn.de).
