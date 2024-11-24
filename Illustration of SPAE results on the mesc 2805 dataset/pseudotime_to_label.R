rm(list = ls())
library(mclust)

file_count <- 10

for (i in 1:file_count) {
    # Build file path
    file_path <- paste0("E:/Fig4/mesc288_drop_res/SPAE_old_computer", "/pseudotime_", i, ".csv")
   
    # Read the CSV file
    pretime <- read.csv(file_path, header = F)
    pretime <- na.omit(pretime)
    rownames(pretime) <- pretime[, 1]
    pretime <- pretime[, -1]
    pretime = as.data.frame(pretime)
    labels <- read.table("E:/Fig4/labels.txt")
    
    
    SPAE_mclust_result <- MclustDA(pretime$pretime, labels$V1)
    SPAE_pred <- predict(SPAE_mclust_result, newdata = SPAE_mclust_result$data)
    SPAE_predict_label<-SPAE_pred$classification
    SPAE_predict_label<-as.data.frame(SPAE_predict_label)
    
    # Save the results
    result_path <- paste0("E:/Fig4/mesc288_drop_res/SPAE_old_computer","/SPAE_predlabel_", i, ".txt")

    write.table(SPAE_predict_label,result_path,row.names = F,col.names = F,quote = F)
    
    # Clean up memory
    rm(pretime, labels, SPAE_mclust_result, SPAE_pred, SPAE_predict_label)
    gc()
  }


