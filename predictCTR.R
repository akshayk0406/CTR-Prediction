setwd("/Users/akshaykulkarni/Documents/gp/CTR-Prediction")
library(caret)
library(gmp)

train_file    <- 'SampleTrain.txt'
test_file     <- 'SampleTest.txt'
req_cols      <- c("hour","C1","banner_pos","device_type","device_conn_type","C14","C15","C16","C17","C18","C19","C20","C21")
output_file   <-  'analysis-output.txt'

input         <- read.csv(file=train_file,head=TRUE,sep=",")
inputdata     <- input[,c("click",req_cols)]

inTrain       <- createDataPartition(y=inputdata$click,p=0.7,list=FALSE)
tr            <- inputdata[inTrain,]
cv            <- inputdata[-inTrain,]
tc            <- trainControl(method="cv",number=3)

rfmodel       <- train(tr$click~.,method="rf",data=tr,trControl=tc,preProcess=c("center","scale"))
saveRDS(rfmodel,"rfcvmodel.RDS")
predrfcv      <- predict(rfmodel,cv[,req_cols])
predrfcv      <- replace(predrfcv, predrfcv<0.5 ,as.integer(0))
predrfcv      <- replace(predrfcv, predrfcv>=0.5 ,as.integer(1))
rfCorrect     <- sum(predrfcv == cv$click)

glmmodel      <- train(tr$click~.,method="glm",family=gaussian(),data=tr,trControl=tc,preProcess=c("center","scale"))
saveRDS(glmmodel,"glmcvmodel.RDS")
predglmcv     <- predict(glmmodel,cv[,req_cols])
predglmcv     <- replace(predglmcv, predglmcv<0.5 ,as.integer(0))
predglmcv     <- replace(predglmcv, predglmcv>=0.5 ,as.integer(1))
glmCorrect    <- sum(predglmcv == cv$click)

if(rfCorrect > glmCorrect) best_model  <- rfmodel else best_model  <- glmmodel

testing       <- read.csv(file=test_file,head=TRUE,sep=",")
testingAdId   <- as.bigz(testing[,1])
testData      <- testing[,req_cols]
num_testData  <- dim(testData)[1]

predcv        <- predict(best_model,testData)
predcv        <- replace(predcv, predcv<0.5 ,as.integer(0))
predcv        <- replace(predcv, predcv>=0.5 ,as.integer(1))

result    <- ""
for(i in 1:num_testData)
    result <- paste(result,paste(testingAdId[i],predcv[i],sep=","),sep="\n")

write(result,file=output_file)