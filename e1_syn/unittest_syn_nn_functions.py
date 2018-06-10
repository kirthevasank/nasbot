"""
  Unit tests for syn_functions.py
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=import-error
# pylint: disable=relative-import
# plyint: disable=invalid-name

import os
import shutil
# Local imports
import syn_nn_functions
from nn.nn_visualise import visualise_list_of_nns
from nn.nn_examples import generate_many_neural_networks
from utils.base_test_class import BaseTestClass, execute_tests

class SynFunctionsTestCase(BaseTestClass):
  """ Contains unit tests for the synthetic functions in syn_nn_functions.py """

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(SynFunctionsTestCase, self).__init__(*args, **kwargs)
    num_nets = 10
    self.save_dir = '../scratch/unittest_syn_nn_functions'
    self.cnns = generate_many_neural_networks('cnn', num_nets)
    self.reg_mlps = generate_many_neural_networks('mlp-reg', num_nets)
    self.class_mlps = generate_many_neural_networks('mlp-class', num_nets)

  def _visualise_nns_with_func_vals(self, descr, list_of_nns, list_of_vals):
    """ Visualises a list of NNs with the function values. """
    save_dir = os.path.join(self.save_dir, descr)
    if os.path.exists(save_dir):
      shutil.rmtree(save_dir)
    list_of_labels = [str(x) for x in list_of_vals]
    visualise_list_of_nns(list_of_nns, save_dir, list_of_labels)

  def test_syn1(self):
    """ Unit test for syn1 """
    self.report('Testing synthetic function 1.')
    # cnns
    cnn_syn1_vals = [syn_nn_functions.cnn_syn_func1(nn) for nn in self.cnns]
    self._visualise_nns_with_func_vals('cnn', self.cnns, cnn_syn1_vals)
    # regression mlps
    reg_mlp_syn1_vals = [syn_nn_functions.mlp_syn_func1(nn) for nn in self.reg_mlps]
    self._visualise_nns_with_func_vals('reg_mlp', self.reg_mlps,
                                               reg_mlp_syn1_vals)
    # classification mlps
    class_mlp_syn1_vals = [syn_nn_functions.syn_func1_common(nn) for nn in
                           self.class_mlps]
    self._visualise_nns_with_func_vals('class_mlp', self.class_mlps,
                                               class_mlp_syn1_vals)


if __name__ == '__main__':
  execute_tests()

