library(ggplot2)
data2 = read.csv("E:/R_cell_cycle/result.csv")
p1 = ggplot(data2,aes(pseudotime,Fzr1))+
  geom_point(color = "black", alpha = .3)+
  geom_smooth(method = "loess",se=FALSE,colour="red")+
  theme(plot.background = element_rect(),panel.grid = element_blank())+mytheme
ggsave(filename = "E:/mesc_Quartz_data/gene_express/Fzr1.tiff",
       plot = p1,
       width = 10,
       height = 10,
       units = "cm",
       dpi=600) 