library(pheatmap)
library(ggplot2)
A <- read.csv('F:/my_results/heatmap/mesc_Q.csv', header=TRUE)
B <- A[,1:10]
col_labels = A$x
col_labels_factor <- factor(col_labels, levels = unique(col_labels))
raw_name = row.names(t(B))
p<-pheatmap(t(B), 
         labels_col = col_labels_factor,
         ylab = "left",
         scale = "row", 
         color = colorRampPalette(c("blue4", "white", "red"))(50), 
         #color = colorRampPalette(c("blue", "white", "red"))(50),
         fontsize = 10, 
         cluster_cols = FALSE,
         lty = 0, 
         #main = "fig"
         )
ggsave(filename = "coor_headmap.tiff",
       plot = p,
       width = 15,
       height = 8,
       units = "cm",
       dpi=300) 