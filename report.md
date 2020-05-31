---
title: "Prediction Course Project"
output:
  html_document:
    keep_md: yes
  pdf_document: default
---
D.V. 31/05/20


# Synopsis
In this work we investigate the quality of execution exercises and try to predict it. The problem is to specify the correct execution and to povide automatic and robust detection of execution mistakes.


Six young health participants were asked to perform one set of 10 repetitions of the Unilateral Dumbbell Biceps Curl in five different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E) ([read more](http://web.archive.org/web/20161224072740/http:/groupware.les.inf.puc-rio.br/har)).


So, we used data from accelerometers on the belt, forearm, arm, and dumbbell (about 54 predictors in final). In this project, our goal was to predict the manner in which they did the exercise (how correctly they did it).


This report describes:

- how the model is built
- use of cross validation
- an estimate of model quality


Accuracy of our model is ~ 99,4%



```r
library(caret)
library(parallel)
library(doParallel)
```

## Preparing data

Now, we load train data, and extract only validated columns (remove timestamps, rownames, mostly NA columns etc.)

```r
set.seed(1313)
traindata <- read.csv("pml-training.csv",na.strings=c("NA","","#DIV/0!"))
valid <- read.csv("pml-testing.csv",na.strings=c("NA","","#DIV/0!")) #validation

traindata$classe <- factor(traindata$classe) #make factor

# Taking 60% for the training data and 40% for the test data
inTrain <- createDataPartition(y = traindata$classe, list = F, p=0.6)
train <- traindata[inTrain,]
test <- traindata[-inTrain,]

train <- train[, -c(1:8)] #remove timestamps, rownames, etc.
train <- train[, -nearZeroVar(train)] #remove near zero vars
nacols <- apply(train, 2, function(x) sum(is.na(x))/nrow(train) > 0.8)
train <- train[, !nacols] #remove columns with more than 80% of NA values
```

## Train RandomForest model

We use the Random Forest model because it automatically selects important variables and is robust to correlated covariates & outliers in general. Random Forest does not depends on preprocessing requirements.


For quality estimation we use cross validation technique (by confusionMatrix) based on our test data set.


We fit the model by parallel computations.

```r
#Configure parallel processing
cluster <- makeCluster(detectCores() - 1) # convention to leave 1 core for OS
registerDoParallel(cluster)
fitControl <- trainControl(method = "cv", number = 5, allowParallel = T)
  
#Train Random Forest Model
rf.model <- train(classe ~ ., data = train, method = "rf", trControl = fitControl)
cm<-confusionMatrix(predict(rf.model, newdata = test), test$classe)

cm
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2229   11    0    0    0
##          B    3 1498    5    0    0
##          C    0    8 1361   10    1
##          D    0    1    2 1274    4
##          E    0    0    0    2 1437
## 
## Overall Statistics
##                                          
##                Accuracy : 0.994          
##                  95% CI : (0.992, 0.9956)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9924         
##                                          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9987   0.9868   0.9949   0.9907   0.9965
## Specificity            0.9980   0.9987   0.9971   0.9989   0.9997
## Pos Pred Value         0.9951   0.9947   0.9862   0.9945   0.9986
## Neg Pred Value         0.9995   0.9968   0.9989   0.9982   0.9992
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2841   0.1909   0.1735   0.1624   0.1832
## Detection Prevalence   0.2855   0.1919   0.1759   0.1633   0.1834
## Balanced Accuracy      0.9983   0.9928   0.9960   0.9948   0.9981
```

```r
#De-register parallel processing cluster
stopCluster(cluster)
registerDoSEQ()
```
As we can see, the accuracy of this model is **0.9940097**. Not bad.

## Prediciting exercise activity using the model


```r
rf.pred <- predict(rf.model, newdata = valid)
rf.pred
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```







.
