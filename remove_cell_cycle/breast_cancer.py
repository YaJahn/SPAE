import pandas as pd
from sklearn import preprocessing
import spae.models.ae

#############################
raw_Y = pd.read_csv(r"E:\hymo_recat\CancerSEA/GSE77308_single_cell_all_label.csv", index_col=0)
print(raw_Y.shape)

print("Original dimesion %d cells x %d genes." % raw_Y.shape)

Y = preprocessing.scale(raw_Y)
N, D = Y.shape
print('After filtering %d Cells (instances) x %d Genes (features)' % (N, D))

model = spae.models.ae.AutoEncoder(input_width=Y.shape[1],
                                   encoder_width=[30, 20],
                                   # encoder_width=[361, 20],
                                   encoder_depth=2,
                                   # n_circular_unit=2,
                                   n_linear_bypass=3,
                                   dropout_rate=0.1)

model.train(Y, batch_size=47, epochs=800, verbose=100, rate=1e-4)

pseudotime = model.predict_pseudotime(Y)

sttpm2 = Y - model.get_circular_component(pseudotime)
sttpm2 = pd.DataFrame(sttpm2)
sttpm2.to_csv(r"E:\breast_cancer/breast_cancer_remove_cell_cycle.csv", header=None, index=None)
