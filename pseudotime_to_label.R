rm(list = ls())
library(mclust)

SPAE_pseudotime<-read.csv("E:/cell_cycle_data/mesc288/mESC288_pseudotime.csv", header = F)

groundtruth<-read.table("E:/mesc_Quartz_data/label.txt", header = F)


SPAE_mclust_result <- MclustDA(SPAE_pseudotime$V1, groundtruth$Status)



SPAE_pred <- predict(SPAE_mclust_result, newdata = SPAE_mclust_result$data)
SPAE_predict_label<-SPAE_pred$classification

SPAE_predict_label<-as.data.frame(SPAE_predict_label)

write.table(SPAE_predict_label,file="E:/SPAE_pre_label/mesc_Quartz_prelabel.txt",row.names = F,col.names = F,quote = F)
