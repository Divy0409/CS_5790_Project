# Load the necessary libraries
library(caret)
library(nnet)    # For multinomial logistic regression
library(glmnet)   # For elastic net regression if needed
library(dplyr)
library(modeldata)
library(corrplot)
library(e1071)
library(randomForest)
library(rpart)
library(ggplot2)
library(tidyr)
library(gridExtra)
library(ROCR)
library(pROC)
library(earth)
library(klaR)
library(doParallel)

# Load your cleaned dataset
diabetes <- read.csv("diabetes_cleaned.csv")
str(diabetes)

# Split the data into training and testing sets (70-30 split)
set.seed(123)  # For reproducibility
trainIndex <- createDataPartition(diabetes$readmitted, p = 0.7, 
                                  list = FALSE)
train_data <- diabetes[trainIndex, ]
test_data <- diabetes[-trainIndex, ]

# Set up parallel processing with 6 cores
cl <- makeCluster(6)  # Use exactly 6 cores
registerDoParallel(cl)

control <- trainControl(method = "cv", number = 10, sampling = "down", classProbs = TRUE, summaryFunction = defaultSummary,allowParallel = TRUE)

library(kernlab)
library(caret)
sigmaRange <- c(0.01, 0.1, 0.5, 1)
train_x <- train_data[, -ncol(train_data) + 1]
train_x <- lapply(train_x, function(x) {
  if (is.character(x)) {
    factor(x)
  } else {
    x  
  }
})
train_x <- lapply(train_x, function(x) {
  if (is.factor(x)) {
    as.numeric(x)
  } else {
    x  
  }
})
train_x <- as.data.frame(train_x)

options(warn=1)
sigmaRangeReduced <- sigest(as.matrix(train_x))
tuneGrid <- expand.grid(.sigma = sigmaRangeReduced[1],
                               .C = 2^(seq(-4, 6)))

# Train a binomial logistic regression model using caret
svm_model <- train(readmitted ~ ., data = train_data, 
                   method = "svmRadial", tuneGrid = tuneGrid, metric = "Kappa",
                   trControl = control, preProcess=c("center", "scale"), fit=FALSE)
svm_model
plot(svm_model)
# Make predictions on the test set
# Ensure `test_data$readmitted` is a factor with correct levels
test_data$readmitted <- factor(test_data$readmitted, levels = c("NO", "YES"))


# Make predictions and convert to factor
predictions <- factor(predict(svm_model, newdata = test_data), levels = c("NO", "YES"))

# Statistics for test set
postResample(pred = predictions, obs = test_data$readmitted)

# Generate confusion matrix
confusionMatrix(predictions, test_data$readmitted)

