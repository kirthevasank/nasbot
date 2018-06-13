# nasbot

** Still updating. Should be ready in a week! **

A Python implementation of NASBOT (Neural Architecture Search with Bayesian Optimisation
and Optimal Transport).
This repo also implements OTMANN (Optimal Transport Metric for Architectures of Neural
Networks), which is an optimal transport based distance for neural network architectures.
For more details, please see our paper below.

For questions and bug reports please email kandasamy@cs.cmu.edu.

### Installation

* Download the package.
```bash
$ git clone https://github.com/kirthevasank/nasbot.git
```

* Install the following packages packages via pip: cython, POT (Python Optimal Transport),
* pygraphviz. pygraphviz is only needed to visualise the networks and is not necessary to
run nasbot. However, some unit tests may fail.
```bash
$ pip install cython POT pygraphviz
```
  In addition to the above, you will need numpy and scipy which can also be pip installed.

* Now set `HOME_PATH` in the set_up file to the parent directory of nasbot, i.e.
`HOME_PATH=<path/to/parent/directory>/nasbot`. Then source the set up file.
```bash
$ source set_up
```

* Next, you need to build the direct fortran library. For this `cd` into `utils/direct_fortran` and run `bash make_direct.sh`. You will need a fortran compiler such as gnu95. Once this is done, you can run `python simple_direct_test.py` to make sure that it was installed correctly.

* **Testing the Installation**:
To test the installation, run ```bash run_all_tests.sh```. Some of the tests are
probabilistic and could fail at times. If this happens, run the same test several times
and make sure it is not consistently failing. Running all tests will take a while.

### Getting started
We have some demos in the demos directory.


### Some Details
- A neural network is represented as a graph (see paper below) in
[nn/neural_network.py](https://github.com/kirthevasank/nasbot/blob/master/nn/neural_network.py).


### OTMANN
- The OTMANN distance is implemented in
[nn/nn_comparators.py](https://github.com/kirthevasank/nasbot/blob/master/nn/nn_comparators.py)
in the class `OTMANNDistanceComputer`.
- The function `get_default_otmann_distance` will return an object which can be used to
  evaluate the OTMANN distance with default parameters.
- You can obtain a customised distance via the function `get_otmann_distance_from_args`.


### Citation
If you use any part of this code in your work, please cite our
[Arxiv paper](https://arxiv.org/pdf/1802.07191.pdf):

```bibtex
@article{kandasamy2018neural,
  title={Neural Architecture Search with Bayesian Optimisation and Optimal Transport},
  author={Kandasamy, Kirthevasan and Neiswanger, Willie and Schneider, Jeff and Poczos,
Barnabas and Xing, Eric},
  journal={arXiv preprint arXiv:1802.07191},
  year={2018}
}
```


### License
This software is released under the MIT license. For more details, please refer
[LICENSE.txt](https://github.com/kirthevasank/nasbot/blob/master/LICENSE.txt).

"Copyright 2018 Kirthevasan Kandasamy"

- For questions and bug reports please email kandasamy@cs.cmu.edu

