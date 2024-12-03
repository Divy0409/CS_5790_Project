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
library(mda)

# Load your cleaned dataset
diabetes <- read.csv("diabetes_cleaned.csv")

# Convert categorical variables to factors
categorical_vars <- c(
  "race", "gender", "age", "A1Cresult", "metformin", "glipizide", "glyburide",
  "pioglitazone", "rosiglitazone", "insulin", "change", "diabetesMed", "primary_diagnosis"
)

diabetes[categorical_vars] <- lapply(diabetes[categorical_vars], factor)

# Ensure target variable is a factor
diabetes$readmitted <- factor(diabetes$readmitted, levels = c("NO", "YES"))

# Split the data into training and testing sets (70-30 split)
set.seed(123)  # For reproducibility
#trainIndex <- createDataPartition(diabetes$readmitted, p = 0.7, 
                                #  list = FALSE)
#train_data <- diabetes[trainIndex, ]
#test_data <- diabetes[-trainIndex, ]
resp <- as.data.frame(diabetes$readmitted)
pred <- diabetes[, -ncol(diabetes) + 1]

nc_idx <- createDataPartition(diabetes$readmitted, p = 0.7, list = FALSE)
nc_trainX <- pred[nc_idx, ]
nc_testX <- pred[-nc_idx, ]
nc_trainY <- resp[nc_idx, ]
nc_testY <- resp[-nc_idx, ]

# Preprocessing
# preproc <- preProcess(train_data[, -ncol(train_data)], method = c("center", "scale"))
#train_data_preprocessed <- predict(preproc, train_data[, -ncol(train_data)])
#test_data_preprocessed <- predict(preproc, test_data[, -ncol(test_data)])

# Combine preprocessed predictors with the target variable
#train_data <- cbind(train_data_preprocessed, readmitted = train_data$readmitted)
#test_data <- cbind(test_data_preprocessed, readmitted = test_data$readmitted)

#nzv <- nearZeroVar(train_data, saveMetrics = TRUE)
#if (any(nzv$nzv)) {
#  train_data <- train_data[, !nzv$nzv]
#  test_data <- test_data[, !nzv$nzv]
#}

control <- trainControl(method = "cv", number = 10, sampling = "down", summaryFunction = defaultSummary)

tuneGrid <- expand.grid(fL = 0:1, usekernel = TRUE, adjust = seq(1, 2, 0.5))

# Train a binomial logistic regression model using caret
options(warn=0)
nb_model <- train(x = nc_trainX, 
                  y = nc_trainY,
                   method = "nb",
                   preProcess=c("center", "scale"),
                   trControl = control, metric = "Kappa")
nb_model

# Make predictions on the test set
# Ensure `test_data$readmitted` is a factor with correct levels
test_data$readmitted <- factor(test_data$readmitted, levels = c("NO", "YES"))

# Make predictions and convert to factor
predictions <- factor(predict(nb_model, newdata = test_data), levels = c("NO", "YES"))

# Statistics for test set
postResample(pred = predictions, obs = test_data$readmitted)

# Generate confusion matrix
confusionMatrix(predictions, test_data$readmitted)
