"""
  Unit tests for nn_modifiers.py
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=relative-import
# pylint: disable=import-error

from argparse import Namespace
from copy import deepcopy
import numpy as np
import os
from shutil import rmtree
# Local imports
import nn_modifiers
from nn_visualise import visualise_nn
from unittest_neural_network import generate_cnn_architectures, generate_mlp_architectures
from utils.base_test_class import BaseTestClass, execute_tests


def _visualise_modifications(save_dir, img_prefix, old_nn, new_nn):
  """ Visualises the old and modified neural network. """
  before_img = os.path.join(save_dir, '%s_bef'%(img_prefix))
  after_img = os.path.join(save_dir, '%s_aft'%(img_prefix))
  visualise_nn(old_nn, before_img)
  visualise_nn(new_nn, after_img)

def test_if_two_networks_are_equal(net1, net2, false_if_net1_is_net2=True):
  """ Returns true if both net1 and net2 are equal.
      If any part of net1 is copied onto net2, then the output will be false
      if false_if_net1_is_net2 is True (default).
  """
  is_true = True
  for key in net1.__dict__.keys():
    val1 = net1.__dict__[key]
    val2 = net2.__dict__[key]
    is_true = True
    if isinstance(val1, dict):
      if false_if_net1_is_net2:
        is_true = is_true and (val1 is not val2)
      for val_key in val1.keys():
        is_true = is_true and np.all(val1[val_key] == val2[val_key])
    elif hasattr(val1, '__iter__'):
      if false_if_net1_is_net2:
        is_true = is_true and (val1 is not val2)
      is_true = is_true and np.all(val1 == val2)
    else:
      is_true = is_true and val1 == val2
    if not is_true: # break here if necessary
      return is_true
  return is_true

def _test_for_orig_vs_modified(save_dir, test_networks, get_list_of_modifiers,
                               write_result):
  """ Gets modifers from get_list_of_modifiers for each network in test_networks and
      modifies each network in test_networks. """
  if os.path.exists(save_dir):
    rmtree(save_dir) # remove this directory
  for idx, old_nn in enumerate(test_networks):
    # Visualise the old network and create a copy to test if it has not changed.
    visualise_nn(old_nn, os.path.join(save_dir, '%d_orig'%(idx))) # visualise
    old_nn_copy = deepcopy(old_nn)
    # Get the modifiers and modified networks.
    basic_modifiers = get_list_of_modifiers(old_nn)
    new_nns = [modif(old_nn) for modif in basic_modifiers]
    # Go through each new network.
    num_valid_networks_created = 0
    for new_idx, new_nn in enumerate(new_nns):
      if new_nn is not None:
        num_valid_networks_created += 1
        visualise_nn(new_nn, os.path.join(save_dir, '%d_%d'%(idx, new_idx)))
    # Finally test if the networks have not changed.
    assert test_if_two_networks_are_equal(old_nn, old_nn_copy)
    write_result('%02d (%s):: #modifiers: %d, #valid-networks: %d.'%(
      idx, old_nn.nn_class, len(basic_modifiers), num_valid_networks_created),
      'test_result')


class NNModifierAtomsTestCase(BaseTestClass):
  """ Unit tests for the basic modifiers in nn_modifiers. """

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(NNModifierAtomsTestCase, self).__init__(*args, **kwargs)
    self.cnns = generate_cnn_architectures()
    self.mlps = generate_mlp_architectures()
    self.save_dir = '../scratch/unittest_modifiers/'

  def _visualise_modifications(self, img_prefix, old_nn, new_nn):
    """ Visualises the old and modified neural network. """
    _visualise_modifications(self.save_dir, img_prefix, old_nn, new_nn)

  def test_wedge_layer(self):
    """ Tests wedge_layer. """
    self.report('Testing wedge_layer. Check %s for results.'%(self.save_dir))
    test_instances = [(self.cnns[3], (2, 7), 'avg-pool', None, None),
                      (self.mlps[2], (4, 6), 'elu', 32, None),
                     ]
    for idx, inst in enumerate(test_instances):
      img_prefix = 'wedge_%d_%d-%d'%(idx, inst[1][0], inst[1][1])
      new_layer_attributes = Namespace()
      old_nn_copy = deepcopy(inst[0])
      if inst[0].nn_class == 'cnn':
        new_layer_attributes.stride = inst[4]
      new_nn = nn_modifiers.wedge_layer(inst[0], inst[2], inst[3], inst[1][0], inst[1][1],
                                        new_layer_attributes)
      assert test_if_two_networks_are_equal(old_nn_copy, inst[0])
      self._visualise_modifications(img_prefix, inst[0], new_nn)

  def test_remove_layer(self):
    """ Tests deletion of a layer. """
    self.report('Testing remove_layer. Check %s for results.'%(self.save_dir))
    test_instances = [(self.mlps[2], 6, [(4, 7)]),
                      (self.cnns[0], 3, []),
                      (self.cnns[8], 9, [(7, 11), (3, 11)])]
    for idx, inst in enumerate(test_instances):
      img_prefix = 'remove_%d_%d_%s'%(idx, inst[1], str(inst[2]))
      old_nn_copy = deepcopy(inst[0])
      new_nn = nn_modifiers.remove_layer(inst[0], inst[1], inst[2])
      self._visualise_modifications(img_prefix, inst[0], new_nn)
      assert test_if_two_networks_are_equal(old_nn_copy, inst[0])

  def test_duplicate_branch(self):
    """ Tests creation of a duplicate branch. """
    self.report('Testing branching. Check %s for results.'%(self.save_dir))
    test_instances = [(self.cnns[5], list(range(4, 14))),
                      (self.cnns[5], list(range(10, 21))),
                      (self.cnns[9], [1, 2, 6, 12, 13]),
                     ]
    for idx, inst in enumerate(test_instances):
      img_prefix = 'branch_%d'%(idx)
      old_nn_copy = deepcopy(inst[0])
      new_nn = nn_modifiers.create_duplicate_branch(inst[0], inst[1])
      self._visualise_modifications(img_prefix, inst[0], new_nn)
      assert test_if_two_networks_are_equal(old_nn_copy, inst[0])

  def test_change_num_units(self):
    """ Tests for changing the number of units in a layer. """
    self.report('Testing changing number of nodes. Check %s for results.'%(self.save_dir))
    test_instances = [(self.mlps[3], [5, 2, 1], [32, 32, 16]),
                      (self.cnns[5], [4, 8], [512, 61]),
                      (self.cnns[9], [4, 9], [128, 256])
                     ]
    for idx, inst in enumerate(test_instances):
      img_prefix = 'change_%d_%s_%s'%(idx, str(inst[1]), str(inst[2]))
      old_nn_copy = deepcopy(inst[0])
      new_nn = nn_modifiers.change_num_units_in_layers(inst[0], inst[1], inst[2])
      self._visualise_modifications(img_prefix, inst[0], new_nn)
      assert test_if_two_networks_are_equal(old_nn_copy, inst[0])


class NNModifierPrimitivesTestCase(BaseTestClass):
  """ Unit tests for the primitives that are being tested. """

  def __init__(self, *args, **kwargs):
    """ Constructor. """
    super(NNModifierPrimitivesTestCase, self).__init__(*args, **kwargs)
    self.cnns = generate_cnn_architectures()
    self.mlps = generate_mlp_architectures()
    self.save_dir = '../scratch/unittest_basic_modifiers/'

  @classmethod
  def _visualise_modifications(cls, curr_save_dir, img_prefix, old_nn, new_nn):
    """ Visualises the old and modified neural network. """
    _visualise_modifications(curr_save_dir, img_prefix, old_nn, new_nn)

  def test_increase_en_masse(self):
    """ Tests increasing the number of units. """
    self.report('Test increasing en masse. Check %s for images.'%(self.save_dir))
    test_networks = [self.cnns[4], self.cnns[1], self.cnns[0], self.mlps[3]]
    test_primitives = [nn_modifiers.increase_en_masse_8_8,
                       nn_modifiers.increase_en_masse_1_4,
                       nn_modifiers.increase_en_masse_1_2,
                       nn_modifiers.increase_en_masse_2_2,
                      ]
    curr_save_dir = os.path.join(self.save_dir, 'increase_en_masse')
    for idx, old_nn in enumerate(test_networks):
      img_prefix = 'increase_%d'%(idx)
      old_nn_copy = deepcopy(old_nn)
      new_nn = test_primitives[idx](old_nn)
      self._visualise_modifications(curr_save_dir, img_prefix, old_nn, new_nn)
      assert old_nn.get_total_mass() < new_nn.get_total_mass()
      assert test_if_two_networks_are_equal(old_nn_copy, old_nn)

  def test_decrease_en_masse(self):
    """ Tests decreasing the number of units en masse. """
    self.report('Test decreasing en_masse. Check %s for images.'%(self.save_dir))
    test_networks = [self.cnns[8], self.cnns[4], self.cnns[6], self.mlps[2]]
    test_primitives = [nn_modifiers.decrease_en_masse_2_4,
                       nn_modifiers.decrease_en_masse_7_8,
                       nn_modifiers.decrease_en_masse_4_4,
                       nn_modifiers.decrease_en_masse_1_2,
                      ]
    curr_save_dir = os.path.join(self.save_dir, 'decrease_en_masse')
    for idx, old_nn in enumerate(test_networks):
      img_prefix = 'decrease_%d'%(idx)
      old_nn_copy = deepcopy(old_nn)
      new_nn = test_primitives[idx](old_nn)
      self._visualise_modifications(curr_save_dir, img_prefix, old_nn, new_nn)
      assert old_nn.get_total_mass() > new_nn.get_total_mass()
      assert test_if_two_networks_are_equal(old_nn_copy, old_nn)

  def test_single_num_units_changes(self):
    """ Tests basic modifiers to increase and decrease number of units on a single
        layer. """
    change_num_units_save_dir = os.path.join(self.save_dir, 'change_single_num_units/')
    self.report('Test modifiers for changing num units in a layer. Check %s for images.'%(
                change_num_units_save_dir))
    test_networks = self.cnns + self.mlps
    get_list_of_modifiers = lambda arg_nn: (
      nn_modifiers.get_list_of_single_layer_modifiers(arg_nn, 'increase') +
      nn_modifiers.get_list_of_single_layer_modifiers(arg_nn, 'decrease'))
    _test_for_orig_vs_modified(change_num_units_save_dir, test_networks,
                               get_list_of_modifiers, self.report)

  def test_remove_layers(self):
    """ Tests primitives to remove a layer. """
    remove_save_dir = os.path.join(self.save_dir, 'remove/')
    self.report('Test modifiers for removing a layer. Check %s for images.'%(
                remove_save_dir))
    test_networks = self.cnns + self.mlps
    _test_for_orig_vs_modified(remove_save_dir, test_networks,
                               nn_modifiers.get_list_of_remove_layer_modifiers,
                               self.report)

  def test_branching(self):
    """ Tests primitives for branching a NN. """
    branch_save_dir = os.path.join(self.save_dir, 'branch/')
    self.report('Test modifiers for branching an NN path. Check %s for images.'%(
                branch_save_dir))
    test_networks = self.cnns + self.mlps
    _test_for_orig_vs_modified(branch_save_dir, test_networks,
                               nn_modifiers.get_list_of_branching_modifiers,
                               self.report)

  def test_skipping(self):
    """ Tests modifications for skipping in a NN. """
    skip_save_dir = os.path.join(self.save_dir, 'skipping/')
    self.report('Test modifiers for skipping in a NN. Check %s for images.'%(
                skip_save_dir))
    test_networks = self.cnns + self.mlps
    _test_for_orig_vs_modified(skip_save_dir, test_networks,
                               nn_modifiers.get_list_of_skipping_modifiers,
                               self.report)

  def test_wedging(self):
    """ Tests modifications for wedging a layer in between an edge. """
    wedge_save_dir = os.path.join(self.save_dir, 'wedging/')
    self.report('Test modifiers for wedging a layer. Check %s for images.'%(
                wedge_save_dir))
    test_networks = self.cnns + self.mlps
    _test_for_orig_vs_modified(wedge_save_dir, test_networks,
                               nn_modifiers.get_list_of_wedge_layer_modifiers,
                               self.report)

  def test_swapping(self):
    """ Tests swapping a layer_label with another type. """
    swap_save_dir = os.path.join(self.save_dir, 'swapping/')
    self.report('Test modifiers for swapping a layer. Check %s for images.'%(
                swap_save_dir))
    test_networks = self.cnns + self.mlps
    _test_for_orig_vs_modified(swap_save_dir, test_networks,
                               nn_modifiers.get_list_of_swap_layer_modifiers,
                               self.report)


if __name__ == '__main__':
  execute_tests()

