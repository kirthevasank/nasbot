import cg.run_tensorflow_wrapper
from opt.nn_opt_utils import get_initial_pool
from utils.base_test_class import BaseTestClass, execute_tests

class RunTensorflowTestCase(BaseTestClass):
  """ Contains unit tests for the run_tensorflow.py function. """

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(RunTensorflowTestCase, self).__init__(*args, **kwargs)
    self.mlps = get_initial_pool('mlp-reg')

  def test_run_yearPredMSD(self):
    """ Tests running on YearPredMSD data. """
    print('\n==========================================================')
    self.report('Testing defining and training mlps in tensorflow on YearPredMSD data.')
    for idx, mlp in enumerate(self.mlps):
      print('\n==========================================================')
      print(mlp.layer_labels)
      cg.run_tensorflow_wrapper.run_yearPredMSD(mlp)

if __name__ == '__main__':
  execute_tests()
