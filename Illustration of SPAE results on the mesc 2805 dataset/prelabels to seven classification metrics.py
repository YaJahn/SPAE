from sklearn import metrics
import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score
import pandas as pd

RI = []
ARI = []
NMI = []
Precision = []
Recall = []
Fscore = []
Acu = []
labels_true = np.loadtxt(r'E:/Fig4/labels.txt')

# print(labels_true)
for i in range(1, 11):
    labels_pred = np.loadtxt(r"E:\Fig4\mesc288_drop_res\SPAE_old_computer\my_computer\10\SPAE_predlabel_" + str(i) + ".txt")

    ri = metrics.rand_score(labels_true, labels_pred)
    ari = metrics.adjusted_rand_score(labels_true, labels_pred)
    nmi = metrics.adjusted_mutual_info_score(labels_true, labels_pred)
    precision = metrics.precision_score(labels_true,labels_pred,average="macro")
    recall = recall_score(labels_true,labels_pred,average="macro")
    fscore = metrics.f1_score(labels_true,labels_pred,average="macro")
    accuracy = metrics.accuracy_score(labels_true, labels_pred)

    RI.append(ri)
    ARI.append(ari)
    NMI.append(nmi)
    Recall.append(recall)
    Precision.append(precision)
    Fscore.append(fscore)
    Acu.append(accuracy)
print("RI:", RI)
print("ARI:", ARI)
print("NMI:", NMI)
print("Recall:", Recall)
print("Precision:", Precision)
print("Fscore:", Fscore)
print("Acu:", Acu)
data = {
    "RI": RI,
    "ARI": ARI,
    "NMI": NMI,
    "Recall": Recall,
    "Precision": Precision,
    "Fscore": Fscore,
    "Accuracy": Acu
}

df = pd.DataFrame.from_dict(data, orient="index")
df.columns = [f" {i+1}" for i in range(df.shape[1])]

excel_filename = "results.xlsx"
df.to_excel(excel_filename, index=True)

print("RI:", np.average(RI))
print("ARI:", np.average(ARI))
print("NMI:", np.average(NMI))
print("Recall:", np.average(Recall))
print("Precision:", np.average(Precision))
print("Fscore:", np.average(Fscore))
print("Acu:", np.average(Acu))


