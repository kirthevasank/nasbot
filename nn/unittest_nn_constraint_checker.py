"""
  Unit tests for functions/classes in nn_constraint_checker.py
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=no-member
# pylint: disable=invalid-name

# Local imports
from nn import nn_constraint_checkers
from unittest_neural_network import generate_cnn_architectures, generate_mlp_architectures
from utils.base_test_class import BaseTestClass, execute_tests


class NNConstraintCheckerTestCase(BaseTestClass):
  """ Contains unit tests for the TransportNNDistanceComputer class. """

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(NNConstraintCheckerTestCase, self).__init__(*args, **kwargs)
    self.nns = generate_cnn_architectures() + generate_mlp_architectures()
    self.cnn_constraint_checker = nn_constraint_checkers.CNNConstraintChecker(
      25, 5, 5000, 0, 5, 2, 15, 512, 16)
    self.mlp_constraint_checker = nn_constraint_checkers.MLPConstraintChecker(
      25, 5, 5000, 900, 5, 2, 15, 30, 8)

  def test_constraint_checker(self):
    """ Tests if the constraints are satisfied for each network. """
    self.report(('Testing constraint checker with max_num_layers=%d, max_mass=%d,' +
      'max_out_deg=%d, max_in_deg=%d, max_num_edges=%d.')%(
      self.cnn_constraint_checker.max_num_layers,
      self.cnn_constraint_checker.max_mass,
      self.cnn_constraint_checker.max_in_degree,
      self.cnn_constraint_checker.max_out_degree,
      self.cnn_constraint_checker.max_num_edges
      ))
    for nn in self.nns:
      if nn.nn_class == 'cnn':
        violation = self.cnn_constraint_checker(nn, True)
        constrain_satisfied = self.cnn_constraint_checker(nn)
      else:
        violation = self.mlp_constraint_checker(nn, True)
        constrain_satisfied = self.mlp_constraint_checker(nn)
      self.report(('%s: #layers: %d, mass: %d, max_out_deg: %d, max_in_deg: %d, ' +
                   '#edges:%d ::: %s, %s')%(
                   nn.nn_class, len(nn.layer_labels), nn.get_total_mass(),
                   nn.get_out_degrees().max(), nn.get_in_degrees().max(),
                   nn.conn_mat.sum(), str(constrain_satisfied), violation), 'test_result')
      assert (constrain_satisfied and violation == '') or \
             (not constrain_satisfied and violation != '')


if __name__ == '__main__':
  execute_tests()

