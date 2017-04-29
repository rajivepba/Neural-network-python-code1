library(MASS)
library(rpart)
library(randomForest)
library(caret)
library(pROC)
setwd("E:\\Work\\Jigsaw Academy\\Corporate Trainings\\Bocconi\\Batch 2\\Online Sessions\\RandomForests-ML")
?fgl

#Comparing a tree and a random forest model: AUC
set.seed(100)
mod_tree<-rpart(type~.,method = "class",data = fgl)
pred<-predict(mod_tree,type="vector")

multiclass.roc(pred,as.numeric(fgl$type))


mod_RandomForest<-randomForest(type~.,data=fgl)
pred1<-as.numeric(predict(mod_RandomForest,type="class"))
#p_trial<-predict(mod_RandomForest,type="class")
multiclass.roc(pred1,as.numeric(fgl$type))

#Most of the variables have appeared as the first split
names(fgl)
first_split<-1:mod_RandomForest$ntree
for (i in 1:mod_RandomForest$ntree)
{
first_split[i]<-as.character(getTree(mod_RandomForest,k = i,labelVar = T)[1,3])
}
v=getTree(mod_RandomForest,k = 2,labelVar = T)

names(fgl) %in% first_split

mod_RandomForest$confusion

sum(diag(mod_RandomForest$confusion[,-7]))/sum(mod_RandomForest$confusion[,-7])

#Tuning a Random Forest Model

set.seed(100)
 
train_spec<-trainControl(method = "oob",
                         number=7,
                         classProbs=TRUE,
                         )
mod_tuned<-train(type~.,data=fgl,method="rf",trControl=train_spec,verbose=F)

#Tuning using grid search
set.seed(100)
tunegrid <- expand.grid(mtry=c(1:6))
mod_tuned1<-train(type~.,data=fgl,method="rf",trControl=train_spec,verbose=F,tuneGrid=tunegrid)

# Regression problem

# Data pertaining to 12 different building shapes simulated in Ecotect. The buildings differ with respect to the glazing area, the glazing area distribution, and the orientation, amongst other parameters. We simulate various settings as functions of the afore-mentioned characteristics to obtain 768 building shapes. The dataset comprises 768 samples and 8 features, aiming to predict two real valued responses. 

build<-read.csv("Building.csv")

mod_build_tree<-rpart(Y1~.,data=build[,-10],method='anova')
predictions<-predict(mod_build_tree)
actual<-build$Y1

RMSE<-function(predictions,actual)
{sqrt(mean((predictions-actual)^2))}

RMSE(predictions,actual)

mod_build_forest<-randomForest(Y1~.,data=build[,-10])
prediction<-predict(mod_build_forest)

RMSE(prediction,actual)

#Take a look at variables appearing in first split
first_split<-1:mod_build_forest$ntree
for (i in 1:mod_build_forest$ntree)
{
  first_split[i]<-as.character(getTree(mod_build_forest,k = i,labelVar = T)[1,3])
}

names(build[,-10]) %in% first_split
