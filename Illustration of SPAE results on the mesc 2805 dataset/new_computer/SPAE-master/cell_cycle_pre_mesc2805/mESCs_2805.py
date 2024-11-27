import os, sys, inspect
import pandas as pd
from scipy.spatial.transform import rotation
from sklearn import preprocessing
import matplotlib.pyplot as plt
import numpy as np

import tensorflow.compat.v1 as tf

tf.disable_eager_execution()
import random

#def set_seed(seed=40):
#    random.seed(seed)                  # Python 
#    np.random.seed(seed)              # NumPy 
#    tf.set_random_seed(seed)        # TensorFlow 
#set_seed(40)

cell_line = "mESC"

#############################

raw_Y = pd.read_table(r'../mesc_2805/mesc_preprocessed.txt', index_col=0, sep='\t')
raw_Y = raw_Y.T

print(raw_Y.shape)

print("Original dimesion %d cells x %d genes." % raw_Y.shape)

cpt = np.loadtxt('../data/mesc_2805/labels.txt')
print(f"G0/G1 {sum(cpt == 1.0)}, S {sum(cpt == 2.0)}, G2/M {sum(cpt == 3.0)}")

Y = preprocessing.scale(raw_Y)
N, D = Y.shape
print('After filtering %d Cells (instances) x %d Genes (features)' % (N, D))

import spae.models

model = spae.models.AutoEncoder(input_width=Y.shape[1],
                                  encoder_width=[30, 20],
                                  # encoder_width=[361, 20],
                                  encoder_depth=2,
                                  # n_circular_unit=2,
                                  n_linear_bypass=3,
                                  dropout_rate=0.1)

model.train(Y,batch_size=32,  epochs=800, verbose=100, rate=2e-4)
pseudotime = model.predict_pseudotime(Y)


time = pd.DataFrame(pseudotime)
time.to_csv(r"E:\Fig4\mesc288_drop_res\SPAE_old_computer\pseudotime_1.csv")






