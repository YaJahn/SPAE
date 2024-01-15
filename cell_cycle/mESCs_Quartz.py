import pandas as pd
from sklearn import preprocessing
import numpy as np
import spae.models.ae

cell_line = "mESC"


raw_Y = pd.read_table('../data/mesc_Quartz/mESC_Quartz_preprocessed.txt', index_col=0, sep='\t')

raw_Y = raw_Y.T
print(raw_Y.shape)
print("raw_Y:", raw_Y)

print("Original dimesion %d cells x %d genes." % raw_Y.shape)
# Original dimesion 361 cells x 253 genes.

cpt = np.loadtxt('../data/mesc_Quartz/label.txt')
print(f"G0/G1 {sum(cpt == 1.0)}, S {sum(cpt == 2.0)}, G2/M {sum(cpt == 3.0)}")
# G0/G1 [85], S [141], G2/M [135]
Y = preprocessing.scale(raw_Y)
N, D = Y.shape
print('After filtering %d Cells (instances) x %d Genes (features)' % (N, D))



model = spae.models.ae.AutoEncoder(input_width=Y.shape[1],
                                   encoder_width=[30, 20],
                                   encoder_depth=2,
                                   n_circular_unit=2,
                                   n_linear_bypass=3,
                                   dropout_rate=0.1)

model.train(Y, batch_size=35, epochs=1000, verbose=100, rate=2e-4)

pseudotime = model.predict_pseudotime(Y)
pseudotime = pd.DataFrame(pseudotime)
# pseudotime.to_csv(r"E:\cell_cycle_data\mesc_Quartz\Quartz_pseudotime.csv",header=None,index=None)




