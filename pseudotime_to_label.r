rm(list = ls())
library(mclust)
#SPAE_pseudotime<-read.csv("E:/cell_cycle_remove/416b.csv", header = F)
SPAE_pseudotime<-read.csv("E:/cell_cycle_data/mesc288/mESC288_pseudotime_test.csv", header = F)

SPAE_pseudotime = read.csv("F:/SC_data/high_var_gene/result_/S_no_ribo.csv", header = F)

#SPAE_pseudotime<-read.csv("E:/cell_cycle_data/mesc_Quartz/Quartz_pseudotime_new_2.csv", header = F)
#SPAE_pseudotime<-read.csv("E:/cell_cycle_data/hESCs/random_gene/random_50/hESCs_pseudotime1.csv", header = F)

#H1 = read.csv("E:/cell_cycle_data/hESCs/random_cell/random_10/data_10.csv")
#groundtruth = H1$label
#groundtruth = as.matrix(groundtruth)
# write.table(groundtruth,file="E:/cell_cycle_data/hESCs/groundtruth.txt",row.names = F,col.names = F,quote = F)

#groundtruth<-read.table("E:/cell_cycle_data/hESCs/groundtruth.txt", header = F)
#groundtruth<-read.table("E:/cell_cycle_data/labels.txt", header = F)

# groundtruth<-read.table("E:/scSimulator/labels.txt", header = F)
# groundtruth$V1

#groundtruth<-read.table("E:/cell_cycle_remove/label.txt", header = F)
#groundtruth<-read.table("E:/mESCs288_data/labels.txt", header = F)
groundtruth<-read.table("E:/mesc_Quartz_data/label.txt", header = F)

groundtruth<-read.csv("F:/SC_data/GSE158724/labels/no_ribo_s_labels.csv")
SPAE_mclust_result <- MclustDA(SPAE_pseudotime$V1, groundtruth$Status)

SPAE_mclust_result <- MclustDA(SPAE_pseudotime$V1, groundtruth$V1)

SPAE_pred <- predict(SPAE_mclust_result, newdata = SPAE_mclust_result$data)
SPAE_predict_label<-SPAE_pred$classification
SPAE_predict_label<-as.data.frame(SPAE_predict_label)
write.table(SPAE_predict_label,file="F:/SC_data/high_var_gene/result_/pre_labels/S_no_ribo_prelabel.txt",row.names = F,col.names = F,quote = F)
