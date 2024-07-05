library(ggplot2)
library(readxl)
data = read_excel("E:/cell_cycle_data/hESCs/different_cell_result/Recall.xlsx")

# legend.position = 'none',
mytheme <-  theme_bw()+theme(panel.grid=element_blank())+theme(
                  axis.text.y = element_text(size = 10,face = "bold"),
                  axis.text.x = element_text(size = 10,face = "bold")
                  )


colnames(data)<-c('x','type','value')
data$type<-factor(data$type,levels = c("ari_ours","cyclum"))
#data$x<-factor(data$x,levels = c("50","100","200","300","400","500","600"))
data$x<-factor(data$x,levels = c("10","30","50","80","100"))

# +scale_fill_manual(values=c('#e57266','#487CB3')
# geom_jitter(position = position_jitter(0.2))
p1 <- ggplot(data,aes(x=x,y=value,color=type))+xlab("Number of cells")+ylab("Recall")+

      scale_color_manual(values=c('#e57266','#487CB3'),name = "Methods")+
      
      geom_boxplot(width=0.6,alpha=0.8,outlier.shape = NA)+geom_point(position = position_jitterdodge(),alpha = 0.5,size=0.8)+
      
      mytheme


ggsave(filename = "Recall.tiff",
       plot = p1,
       width = 8,
       height = 8,
       units = "cm",
       dpi=600)
