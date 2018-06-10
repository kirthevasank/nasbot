"""
  Unit tests for run_tensorflow.py
"""

import cg.run_tensorflow 
import cPickle as pic

def run_yearPredMSD(mlp):
  # Specify tf params
  tf_params = {
    'trainBatchSize':20,
    'valiBatchSize':1000,
    'trainNumStepsPerLoop':20,
    'valiNumStepsPerLoop':5,
    'numLoops':4,
    'learningRate':0.00001,
    }
  
  # Load data
  with open('data/YearPredictionMSD_small.p','rb') as input_file:
    data = pic.load(input_file)
  data_train = data['train']
  data_vali = data['vali']

  # Get validation error
  vali_error = cg.run_tensorflow.compute_validation_error(mlp,data_train,data_vali,tf_params)
