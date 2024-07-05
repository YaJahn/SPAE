rm(list=ls())
library(ggpubr)
library(ggplot2)
library(Seurat)
library(patchwork)
size = 12
# mytheme <- theme_bw() + theme(
#   axis.text.x = element_text(size = size,color = 'black', angle = 0),
#   axis.text.y = element_text(size = size,color = 'black', angle = 0),
#   #axis.title.x =  element_text(size = size,color = 'black'),
#   #axis.title.y =  element_text(size = size,color = 'black'),
#   axis.title = element_blank(),
#   title =  element_text(size = size,color = 'black'),
#   plot.title = element_text(hjust = 0.5),
#   panel.grid.major = element_blank(),
#   panel.grid.minor = element_blank(),
#   axis.line = element_line(size=0.5, colour = "black"),
#   panel.border = element_rect(colour ="black",size=0.5),
#   legend.position = 'none',
#   plot.margin=unit(c(2.5, 8, 2.5,8),'cm'),
# )
#axis.title.x=element_blank(),
#plot.margin=unit(c(2.5, 0, 2.5,0),'cm')
#axis.title.x = element_text(face = "bold",color = "black"),
#axis.title.y = element_text(face = "bold",color = "black"),
mytheme<-theme_bw() +
  theme(panel.grid=element_blank())+ theme(
                axis.text.y = element_text(face = "bold",color = "black"),
                axis.text.x = element_text(hjust=1,face = "bold",color = "black"),
                legend.position = "none",
                plot.margin=unit(c(2.4, 8, 2.4,8),'cm'),
                plot.title = element_text(face = "bold",hjust = 0.5))

##input: cell X gene
mesc_DEG<-read.table("E:/mesc_Quartz_data/mESC_Quartz_preprocessed.txt",header=T,row.names = 1)
#mesc_DEG<-read.table("E:/mesc_Quartz_data/mesc_Quartz_after.txt",header=T,row.names = 1)
# mesc_DEG<-read.table("E:/mESCs288_data/mesc_preprocessed.txt",header=T,row.names = 1)

# mesc_DEG<-read.table("E:/mESCs288_data/mesc_preprocessed.txt",header=T,row.names = 1)
mesc_DEG<-t(mesc_DEG)
mesc_DEG<-as.data.frame(mesc_DEG)

my_pseudotime<-read.csv("E:/cell_cycle_data/mESC_Quartz/find_Quartz_pseudotime_coor.csv", header = F)
# my_pseudotime<-read.csv("E:/cell_cycle_data/cyclum_pseudotime_Quartz.csv", header = F)
# my_pseudotime<-read.csv("E:/mesc_Quartz_data/my_after_mESC_Quartz_pseudotime_0.8571.csv", header = F)
# my_pseudotime<-read.csv("E:/cell_cycle_data/mESC288_0.90277.csv", header = F)
#my_pseudotime<-read.csv("C:/Users/adminis/Desktop/mesc_Quartz/0.8571428571428571.csv", header = F)
# my_pseudotime<-read.csv("E:/mesc_Quartz_data/CCPE_pseudotime.csv", header = F)
label<-read.table("E:/mesc_Quartz_data/label.txt", header = F)

# label<-read.table("E:/mESCs288_data/labels.txt", header = F)

correlation<-matrix(0,ncol(mesc_DEG),1)
rownames(correlation)<-colnames(mesc_DEG)


for (i in 1:ncol(mesc_DEG)){
  correlation[i,1]=cor(mesc_DEG[,i],my_pseudotime$V1)
}
#write.table(correlation,"E:/my_mESC_Quartz_pseudotim.csv",row.names = T)

#show genes with high correlations
c<-correlation[,1]
c<-sort(c,decreasing = T)


#plot correlation of gene Aurka
plotdata<-matrix(0,nrow(mesc_DEG),3)
plotdata[,1]<-my_pseudotime$V1
plotdata[,2]<-mesc_DEG$Aurka
plotdata[,3]<-label$V1

plotdata_1<-matrix(0,nrow(mesc_DEG),3)
plotdata_1[,1]<-my_pseudotime$V1
plotdata_1[,2]<-mesc_DEG$Cdca2
plotdata_1[,3]<-label$V1

plotdata_2<-matrix(0,nrow(mesc_DEG),3)
plotdata_2[,1]<-my_pseudotime$V1
plotdata_2[,2]<-mesc_DEG$Kpna2
plotdata_2[,3]<-label$V1

plotdata<-as.data.frame(plotdata)
plotdata_1<-as.data.frame(plotdata_1)
plotdata_2<-as.data.frame(plotdata_2)

colnames(plotdata)<-c("x","y","colors")
colnames(plotdata_1)<-c("x","y","colors")
colnames(plotdata_2)<-c("x","y","colors")

#colnames(plotdata)<-c("Pseudotime","Expression","colors")
p1 = ggplot(plotdata, aes(x=x, y=y))+
  geom_point(aes(color = factor(colors)))+
  labs(x="Pseudotime")+labs(y="Expression")+
  stat_smooth(method="lm",se=FALSE,size=1)+
  stat_cor(data=plotdata, method = "pearson")+
  mytheme
ggsave(filename = "Aurka.png",
       plot = p1,
       units = "cm",
       dpi=2000)

p2 = ggplot(plotdata_1, aes(x=x, y=y))+
  geom_point(aes(color = factor(colors)))+
  labs(x="Pseudotime",y="")+
  stat_smooth(method="lm",se=FALSE)+
  stat_cor(data=plotdata_1, method = "pearson")+mytheme

ggsave(filename = "Cdca2.png",
       plot = p2,
       units = "cm",
       dpi=2000)
p3 = ggplot(plotdata_2, aes(x=x, y=y))+
  geom_point(aes(color = factor(colors)))+
  labs(x="Pseudotime",y="")+
  stat_smooth(method="lm",se=FALSE)+
  stat_cor(data=plotdata_2, method = "pearson")+mytheme

ggsave(filename = "Kpna2.png",
       plot = p3,
       units = "cm",
       dpi=2000)

p3 = ggplot(plotdata_2, aes(x=x, y=y))+
  geom_point(aes(color = factor(colors)))+
  labs(x="Pseudotime",y="")+
  stat_smooth(method="lm",se=FALSE)+
  stat_cor(data=plotdata_2, method = "pearson")

pp = (p1+p2+p3)+plot_layout(widths = c(5,1))

pp = (p1+p2)+plot_layout(widths = c(5,1))
pp1 = p2+plot_spacer()+plot_layout(widths = c(5,1))
pp2 = pp1/pp+plot_layout(heights = c(1, 5))

p = cowplot::plot_grid(p1,p2,p3,nrow=1)
p = p1 + p2 + p3
p

#ggplot(plotdata, aes(x=Pseudotime, y=Expression))+geom_point(aes(color = factor(colors)))+stat_smooth(method="lm",se=FALSE)+stat_cor(data=plotdata, method = "pearson")+mytheme
