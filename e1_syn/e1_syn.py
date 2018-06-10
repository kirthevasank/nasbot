"""
  Experiments on the synthetic functions.
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=relative-import

from argparse import Namespace
from time import time, clock
# Local
from e2_mlp.e2_mlp import get_method_options
from nn.nn_constraint_checkers import CNNConstraintChecker, MLPConstraintChecker
from nn.nn_modifiers import get_nn_modifier_from_args
from nn.nn_comparators import get_transportnndistance_from_args
from opt.domains import NNDomain
from opt.ga_optimiser import ga_opt_args
from opt.nn_opt_experimenter import NNOptExperimenter
# from opt.nn_gp_bandit import all_nn_gp_bandit_args, all_nn_random_bandit_args
from opt.worker_manager import SyntheticWorkerManager
from opt.function_caller import FunctionCaller
from syn_nn_functions import cnn_syn_func1, mlp_syn_func1, syn_func1_common
from utils.option_handler import load_options
from utils.reporters import get_reporter

# Experiment parameters ================================================================

## Debug mode?
IS_DEBUG = False
# IS_DEBUG = True

EXP_NAME = 'cnn'
# EXP_NAME = 'mlp-class'
# EXP_NAME = 'mlp-reg'

# NN Constraints
MAX_NUM_LAYERS = 60
MIN_NUM_LAYERS = 4
MAX_MASS = 1e8
MIN_MASS = 0
MAX_IN_DEGREE = 5
MAX_OUT_DEGREE = 5
MAX_NUM_EDGES = 200
MAX_NUM_UNITS_PER_LAYER = 1024
MIN_NUM_UNITS_PER_LAYER = 8

# Won't be changing these much
# NUM WORKERS = 1
# MAX_CAPITAL = 100
# TIME_DISTRO = 'const'
NUM_WORKERS = 1
MAX_CAPITAL = 20
TIME_DISTRO = 'const'
GA_MUTATION_STEPS_PROBS = [0.5, 0.25, 0.125, 0.075, 0.05]
BO_GA_MUTATION_STEPS_PROBS = GA_MUTATION_STEPS_PROBS

## Number of experiments
NUM_EXPERIMENTS = 20
SAVE_RESULTS_DIR = 'results'

## Methods
# METHODS = ['asyGA', 'asyHEI', 'asyRAND', 'asyTREE']
# METHODS = ['asyUCB', 'asyHUCB', 'asyEI', 'asyHEI']
# METHODS = ['asyGA', 'asyHEI']
METHODS = ['asyGA', 'asyGA']
# METHODS = ['asyHEI']
# METHODS = ['asyRAND']
# METHODS = ['asyTREE']


def get_prob_params():
  """ Returns the problem parameters. """
  prob = Namespace()
  prob.nn_type = EXP_NAME
  if IS_DEBUG:
    prob.num_experiments = 3
    prob.max_capital = 4
    prob.experiment_name = EXP_NAME
    prob.num_workers = 2
  else:
    prob.num_experiments = NUM_EXPERIMENTS
    prob.max_capital = MAX_CAPITAL
    prob.experiment_name = EXP_NAME
    prob.num_workers = NUM_WORKERS
  # Time
  prob.time_distro = TIME_DISTRO
#   prob.time_distro_params = Namespace
  # Create func_caller, nn_domain and constraint checker
  if prob.experiment_name.startswith('cnn'):
    constraint_checker_constructor = CNNConstraintChecker
    func = cnn_syn_func1
  elif prob.experiment_name == 'mlp-reg':
    constraint_checker_constructor = MLPConstraintChecker
    func = mlp_syn_func1
  elif prob.experiment_name == 'mlp-class':
    constraint_checker_constructor = MLPConstraintChecker
    func = syn_func1_common
  prob.save_file_prefix = '%s_syn'%(prob.experiment_name)
  constraint_checker = constraint_checker_constructor(MAX_NUM_LAYERS, MIN_NUM_LAYERS,
                         MAX_MASS, MIN_MASS, MAX_IN_DEGREE, MAX_OUT_DEGREE, MAX_NUM_EDGES,
                         MAX_NUM_UNITS_PER_LAYER, MIN_NUM_UNITS_PER_LAYER)
  nn_domain = NNDomain(prob.experiment_name, constraint_checker)
  prob.func_caller = FunctionCaller(func, nn_domain)
  prob.constraint_checker = constraint_checker
  # Worker manager
  prob.worker_manager = SyntheticWorkerManager(prob.num_workers, TIME_DISTRO)
  # Other
  prob.methods = METHODS
  prob.save_results_dir = SAVE_RESULTS_DIR
  prob.reporter = get_reporter('default')
  # Experiment options
  experiment_options = Namespace(pre_eval_points='generate')
  prob.experiment_options = experiment_options
  return prob


def main():
  """ Main Function. """
  prob = get_prob_params()
  method_options = get_method_options(prob, 'return_value')
  # get_method_options is now in e2_mlp/e2_mlp.py

  experimenter = NNOptExperimenter(experiment_name=prob.experiment_name,
                                   func_caller=prob.func_caller,
                                   worker_manager=prob.worker_manager,
                                   max_capital=prob.max_capital,
                                   methods=prob.methods,
                                   num_experiments=prob.num_experiments,
                                   save_dir=prob.save_results_dir,
                                   experiment_options=prob.experiment_options,
                                   save_file_prefix=prob.save_file_prefix,
                                   method_options=method_options,
                                   reporter=prob.reporter)
  start_realtime = time()  
  start_cputime = clock()
  experimenter.run_experiments()
  end_realtime = time()
  end_cputime = clock()
  reporter.writeln()
  reporter.writeln('realtime taken: %0.6f'%(end_realtime - start_realtime))
  reporter.writeln('cputime taken: %0.6f'%(end_cputime - start_cputime))


if __name__ == '__main__':
  main()

