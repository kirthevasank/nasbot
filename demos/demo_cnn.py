"""
  A demo of NASBOT on a CNN (Convolutional Neural Network) architecture search problem
  on the Cifar10 dataset.
  -- kandasamy@cs.cmu.edu
"""

from argparse import Namespace
import numpy as np
# Local
from nn.nn_constraint_checkers import get_nn_domain_from_constraints
from nn.nn_visualise import visualise_nn
from opt import nasbot
from demos.cnn_function_caller import CNNFunctionCaller
from opt.worker_manager import RealWorkerManager
from utils.reporters import get_reporter
import os

# Data:
# We use the Cifar10 dataset which is converted to .tfrecords format for tensorflow.
# You can either download the original dataset from www.cs.toronto.edu/~kriz/cifar.html
# and follow the instructions in
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10_estimator/generate_cifar10_tfrecords.p
# Alternatively, they are available in the required format at
# www.cs.cmu.edu/~kkandasa/research.html as examples.
# Put the xxx.tfrecords in a directory named cifar-10-data in the demos directory to run
# this demo.

# Search space
MAX_NUM_LAYERS = 60 # The maximum number of layers
MIN_NUM_LAYERS = 5 # The minimum number of layers
MAX_MASS = np.inf # Mass is the total amount of computation at all layers
MIN_MASS = 0
MAX_IN_DEGREE = 5 # Maximum in degree of each layer
MAX_OUT_DEGREE = 55 # Maximum out degree of each layer
MAX_NUM_EDGES = 200 # Maximum number of edges in the network
MAX_NUM_UNITS_PER_LAYER = 1024 # Maximum number of computational units ...
MIN_NUM_UNITS_PER_LAYER = 8    # ... (neurons/conv-filters) per layer.

# Which GPU IDs are available
GPU_IDS = [0]
#GPU_IDS = [0, 1]  # For multiple GPUs

# Specify data directory
DATA_DIR = 'demos/cifar-10-data' # contains the cifar-10-data files

# Specify directories that will be created
EXP_DIR = 'ExperimentDirectory/' # will store output files for demo_cnn
TMP_DIR = EXP_DIR + 'tmp/' # will store temporary model checkpoints
LOG_FILE = EXP_DIR + 'log_cnn' # will log output and results

# Specify the budget (in seconds)
BUDGET = 12 * 60 * 60

# Make EXP_DIR and TMP_DIR
os.mkdir(EXP_DIR)
os.mkdir(TMP_DIR)

# Obtain a reporter object
# REPORTER = get_reporter('default') # Writes results to stdout
REPORTER = get_reporter(open(LOG_FILE, 'w')) # Writes to file LOG_FILE

def main():
  """ Main function. """
  # Obtain the search space
  nn_domain = get_nn_domain_from_constraints('cnn', MAX_NUM_LAYERS, MIN_NUM_LAYERS,
                MAX_MASS, MIN_MASS, MAX_IN_DEGREE, MAX_OUT_DEGREE, MAX_NUM_EDGES,
                MAX_NUM_UNITS_PER_LAYER, MIN_NUM_UNITS_PER_LAYER)
  # Obtain a worker manager: A worker manager (defined in opt/worker_manager.py) is used
  # to manage (possibly) multiple workers. For a RealWorkerManager, the budget should be
  # given in wall clock seconds.
  worker_manager = RealWorkerManager(GPU_IDS)
  # Obtain a function caller: A function_caller is used to evaluate a function defined on
  # neural network architectures. We have defined the CNNFunctionCaller in
  # demos/cnn_function_caller.py. The train_params argument can be used to specify
  # additional training parameters such as the learning rate etc.
  train_params = Namespace(data_dir=DATA_DIR)
  func_caller = CNNFunctionCaller('cifar10', nn_domain, train_params,
                                  tmp_dir=TMP_DIR, reporter=REPORTER)

  # Run nasbot
  opt_val, opt_nn, _ = nasbot.nasbot(func_caller, worker_manager, BUDGET,
                                     reporter=REPORTER)

  # Print the optimal value and visualise the best network.
  REPORTER.writeln('\nOptimum value found: %f' % opt_val)
  REPORTER.writeln('Optimal network visualised in ' + EXP_DIR + 'cnn_opt_network.eps.')
  visualise_nn(opt_nn, EXP_DIR + 'cnn_opt_network')

  # N.B: See function nasbot and class NASBOT in opt/nasbot.py to customise additional
  # parameters of the algorithm.


if __name__ == '__main__':
  main()
