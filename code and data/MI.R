setwd('E:\\Work\\Jigsaw Academy\\Corporate Trainings\\Bocconi\\Batch 3\\Online Sessions\\DRT\\DRT')
data=read.csv('Aquisition_Risk.csv')
names(data)
data=data[,-c(14,15)]
data=na.omit(data)
y=data$Good_Bad
x=data[-24]
library(entropy)
##Sample code
mi.plugin(table(data$Good_Bad,data$grade))
MI<-function(x,y){
  MI_Store<-1:length(x)
  names(MI_Store)<-names(x)
  for(i in 1:length(x)){
   MI_Store[i]<- mi.plugin(table(y,x[[i]]))
  }
  MI_Store=data.frame(MutualInfo=sort(MI_Store,decreasing = T))
return(MI_Store)  
}
MI(x=x,y=y)

###IV###
library(smbinning)
names(data)
data$Target<-ifelse(data$Good_Bad=='Good',1,0)
smbinning(data,y="Target",x="loan_amnt",p=0.05)$ivtable
ivout<-smbinning(data,y="Target",x="loan_amnt",p=0.05)
smbinning.sql(ivout = ivout)

###L1 penalty###
mnist_x=read.csv('mnist_x.csv',header = T)
mnist_y=read.csv('mnist_y.csv',header=F,col.names = 'Target')

## convert the problem into a binary classification problem: predict if a number will be more than 7 or not (1-is greater than 7, 0 otherwise)

y=ifelse(mnist_y$Target>7,1,0)
x=as.matrix(mnist_x)
library(glmnet)
mod<-cv.glmnet(x,y,family='binomial',alpha=1)
plot(mod)
mod$lambda.min
coef(mod,s=mod$lambda.min)
e=exp(predict.cv.glmnet(mod,newx = x))
p=e/(1+e)
l=ifelse(p>0.5,1,0)
sum(l==y)/length(y)


#####PCA Example#######
data=read.csv('nyt.frame.csv')
colnames(data)[sample(ncol(data),30)]
signif(data[sample(nrow(data),5),sample(ncol(data),10)],3)
nyt.pca = prcomp(data[,-1])
str(nyt.pca)
?prcomp
nyt.latent.sem = nyt.pca$rotation
# What are the components?
# Show the 30 words with the biggest positive loading on PC1
signif(sort(nyt.latent.sem[,1],decreasing=TRUE)[1:30],2)
# biggest negative loading on PC1, the other end of that scale
signif(sort(nyt.latent.sem[,1],decreasing=FALSE)[1:30],2)

# Ditto for PC 2
signif(sort(nyt.latent.sem[,2],decreasing=TRUE)[1:30],2)
signif(sort(nyt.latent.sem[,2],decreasing=FALSE)[1:30],2)


# Plot the projection of the stories on to the first 2 components
# Establish the plot window
plot(nyt.pca$x[,1:2],type="n")
# Arts stories with red As
points(nyt.pca$x[data[,"class.labels"]=="art",1:2],pch="A",col="red")
# Music stories with blue Ms
points(nyt.pca$x[data[,"class.labels"]=="music",1:2],pch="M",col="blue")
# The separation is very good, even with only two components.

###LDA example####
data=read.csv('authorship.csv')
data$target=ifelse(data$Author=='London',0,1)
library(MASS)

### PCA ###
author_pca=prcomp(data[,-c(70,71,72)])
plot(author_pca$x[,1],author_pca$x[,2])
points(author_pca$x[data$target==1,1:2],col='red')
points(author_pca$x[data$target==0,1:2],col='blue')


mod=lda(target~.,data[,-c(70,71)])
summary(mod)
mod$scaling


##Projected data on LD1###
projection=t((t(mod$scaling)%*%t(as.matrix(data[,-c(70,71,72)]))))
plot(projection,col=as.factor(data$target))

