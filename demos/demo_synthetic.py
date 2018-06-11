"""
  A demo of nasbot on a synthetic function.
  -- kandasamy@cs.cmu.edu
"""

import numpy as np
# Local
from nn.nn_constraint_checkers import get_nn_domain_from_constraints

# Search space
MAX_NUM_LAYERS = 50 # The maximum number of layers
MIN_NUM_LAYERS = 5 # The minimum number of layers
MAX_MASS = np.inf # Mass is the total amount of computation at all layers
MIN_MASS = 0
MAX_IN_DEGREE = 5 # Maximum in degree of each layer
MAX_OUT_DEGREE = 55 # Maximum out degree of each layer
MAX_NUM_EDGES = 200 # Maximum number of edges in the network
MAX_NUM_UNITS_PER_LAYER = 1024 # Maximum number of computational units ...
MIN_NUM_UNITS_PER_LAYER = 8    # ... (neurons/conv-filters) per layer.

def main():
  """ Main function. """
  # Obtain the search space
  nn_domain = get_nn_domain_from_constraints('cnn', MAX_NUM_LAYERS, MIN_NUM_LAYERS,
                MAX_MASS, MIN_MASS, MAX_IN_DEGREE, MAX_OUT_DEGREE, MAX_NUM_EDGES,
                MAX_NUM_UNITS_PER_LAYER, MIN_NUM_UNITS_PER_LAYER)


if __name__ == '__main__':
  main()

