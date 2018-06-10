"""
  Unit tests for run_tensorflow.py
"""

import tensorflow as tf
# Local imports
import cg.run_tensorflow 
#import run_tensorflow 
from utils.base_test_class import BaseTestClass, execute_tests
from nn.nn_examples import generate_cnn_architectures, generate_mlp_architectures, generate_many_neural_networks

class RunTensorflowTestCase(BaseTestClass):
  """ Contains unit tests for the run_tensorflow.py function. """

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(RunTensorflowTestCase, self).__init__(*args, **kwargs)
    #self.mlps = generate_mlp_architectures('class')
    self.mlps = generate_many_neural_networks('mlp-reg',5)

  ## Iris data for regression.
  #def test_mlp_iris(self):
    #""" Tests defining and training NNs in tensorflow. """
    #print('\n==========================================================')
    #self.report('Testing defining and training mlps in tensorflow on iris data.')
    #training_set = tf.contrib.learn.datasets.load_iris()
    #for idx, mlp in enumerate(self.mlps):
      #print('\n==========================================================')
      #print(mlp.layer_labels)
      #cg.run_tensorflow.compute_validation_accuracy(mlp,training_set)

  # Iris data for classification.
  #def test_mlp_iris(self):
    #""" Tests defining and training NNs in tensorflow. """
    #print('\n==========================================================')
    #self.report('Testing defining and training mlps in tensorflow on iris data.')
    #training_set = tf.contrib.learn.datasets.load_iris()
    #training_set.target[training_set.target>0] = 1 # CHANGE TO CLASSIFICATION LABELS
    #for idx, mlp in enumerate(self.mlps):
      #print('\n==========================================================')
      #print(mlp.layer_labels)
      #cg.run_tensorflow.compute_validation_accuracy(mlp,training_set)

  # YearPredMSD data for regression.
  def test_mlp_yearpredmsd(self):
    """ Tests defining and training NNs in tensorflow. """
    print('\n==========================================================')
    self.report('Testing: define and train mlps in tensorflow on YearPredMSD data.')

    # Load YearPredMSD data
    from cg.test_yearpredmsd import *

    # Specify tf params
    tf_params = {
      'trainBatchSize':20,
      'valiBatchSize':1000,
      'trainNumStepsPerLoop':5,
      'valiNumStepsPerLoop':10,
      'numLoops':4}
    # OTHERS TO SPECIFY:
    # Learning rate
    # reg-loss
    #class-loss (or maybe these lossees are fixed in code)


    mlp = self.mlps[0]
    #for idx, mlp in enumerate(self.mlps):
    print('==========================================================')
    print(mlp.layer_labels)
    cg.run_tensorflow.compute_validation_error(mlp,data_train,data_vali,tf_params)

if __name__ == '__main__':
  execute_tests()
