rm(list=ls())
library(ggplot2)
size = 15
#my_pseudotime<-read.csv("E:/细胞周期预测模型对比/CCPE-main/CCPE-main/reproduction/pseudotime_analysis/3_output/CCPE_pseudotime.csv", header = F)
#my_pseudotime<-read.csv("E:/cell_cycle_data/mesc_Quartz/mESC_Quartz_pseudotime_0.94.csv", header = F)
my_pseudotime<-read.csv("E:/cell_cycle_data/mesc_Quartz/pre/mESC_Quartz_pseudotime_1.csv", header = F)

my_pseudotime<-read.csv("E:/cell_cycle_data/mesc_Quartz/Quartz_pseudotime_new.csv", header = F)
Cyclops_pretime<-read.csv("F:/my_results/1_mesc_Quartz/Cyclops_mesc_Quartz.csv", header = F)
Cyclum_pretime<-read.csv("F:/my_results/1_mesc_Quartz/Cyclum_mesc_Quartz.csv", header = F)
reCat_pretime<-read.table("F:/my_results/1_mesc_Quartz/reCAT_mesc_Quartz.txt")
#face = "bold"
groundtruth<-read.table("E:/mesc_Quartz_data/label.txt", header = F)
mytheme<- theme(axis.title.x=element_blank(),
                 axis.text.y = element_text(color = "black",size=16),
                 axis.title.y = element_text(color = "black",size=16),
                 axis.text.x = element_text(hjust=1,color = "black",size=16),
                 legend.position = "none",
                 plot.margin=unit(c(2.5, 0, 2.5,0),'cm'),
                 plot.title = element_text(face = "bold",hjust = 0.5))

my_pseudotime[which(groundtruth==1),2]<-'G1'
my_pseudotime[which(groundtruth==2),2]<-'S'
my_pseudotime[which(groundtruth==3),2]<-'G2M'

Cyclops_pretime[which(groundtruth==1),2]<-'G1'
Cyclops_pretime[which(groundtruth==2),2]<-'S'
Cyclops_pretime[which(groundtruth==3),2]<-'G2M'

Cyclum_pretime[which(groundtruth==1),2]<-'G1'
Cyclum_pretime[which(groundtruth==2),2]<-'S'
Cyclum_pretime[which(groundtruth==3),2]<-'G2M'

reCat_pretime[which(groundtruth==1),2]<-'G1'
reCat_pretime[which(groundtruth==2),2]<-'S'
reCat_pretime[which(groundtruth==3),2]<-'G2M'

colnames(my_pseudotime)<-c('pseudo_time','cell_cycle')
colnames(Cyclops_pretime)<-c('pseudo_time','cell_cycle')
colnames(Cyclum_pretime)<-c('pseudo_time','cell_cycle')
colnames(reCat_pretime)<-c('pseudo_time','cell_cycle')

my_pseudotime$cell_cycle<-factor(my_pseudotime$cell_cycle,levels=c('G1','S','G2M'))
Cyclops_pretime$cell_cycle<-factor(Cyclops_pretime$cell_cycle,levels=c('G1','S','G2M'))
Cyclum_pretime$cell_cycle<-factor(Cyclum_pretime$cell_cycle,levels=c('G1','S','G2M'))
reCat_pretime$cell_cycle<-factor(reCat_pretime$cell_cycle,levels=c('G1','S','G2M'))

p1<-ggplot(my_pseudotime,aes(cell_cycle,pseudo_time,fill=cell_cycle))+
  geom_boxplot(outlier.shape = NA)+labs(x="")+labs(y="Pseudotime")+
  mytheme

# p2<-ggplot(Cyclops_pretime,aes(cell_cycle,pseudo_time,fill=cell_cycle))+
    # geom_boxplot(outlier.shape = NA)+labs(x="")+labs(y="")+
  # mytheme

#p3 <-ggplot(Cyclum_pretime,aes(cell_cycle,pseudo_time,fill=cell_cycle))+
 # geom_boxplot(outlier.shape = NA)+labs(x="")+labs(y="")+
  #mytheme

#p4 <-ggplot(reCat_pretime,aes(cell_cycle,pseudo_time,fill=cell_cycle))+
 # geom_boxplot(outlier.shape = NA)+labs(x="")+labs(y="")+
  #mytheme


#p = cowplot::plot_grid(p1,p3,p2,p4,nrow=1,
 #                  
  #                 label_x = 0.5,label_y = 0.4
   #                )
p = cowplot::plot_grid(p1,nrow=1,
                       label_x = 0.5,label_y = 0.4
)
p1
ggsave(filename = "F:/big_paper/pseudotime8.png",
       plot = p,
       width = 30,
       height = 12,
       units = "cm",
       dpi=200)

ggsave(filename = "F:/big_paper/pseudotime2.png",
       plot = p1,
       width = 10,
       height = 12,
       units = "cm",
       dpi=200)


