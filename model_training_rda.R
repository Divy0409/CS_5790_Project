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
library(mda)
library(klaR)

# Load your cleaned dataset
diabetes <- read.csv("diabetes_cleaned.csv")
str(diabetes)

# Split the data into training and testing sets (70-30 split)
set.seed(123)  # For reproducibility
trainIndex <- createDataPartition(diabetes$readmitted, p = 0.7, 
                                  list = FALSE)
train_data <- diabetes[trainIndex, ]
test_data <- diabetes[-trainIndex, ]

control <- trainControl(method = "cv", number = 10, sampling = "down", summaryFunction = defaultSummary, savePredictions=TRUE)

# Train a rda model using caret
tune_grid <- expand.grid(.lambda = c(0, 0.01, 0.1, 1, 10), 
                         .gamma = seq(0, 2, length.out = 10))  
rda_model <- train(readmitted ~ ., data = train_data, 
                   method = "rda",metric="Kappa",
                   tuneGrid = tune_grid,
                   trControl = control)
rda_model

plot(rda_model, cex = 1, lwd = 2, pch = 16, main = "RDA Tuning Plot")

# Make predictions on the test set
# Ensure `test_data$readmitted` is a factor with correct levels
test_data$readmitted <- factor(test_data$readmitted, levels = c("NO", "YES"))

# Make predictions and convert to factor
predictions <- predict(rda_model, newdata = test_data)

# Statistics for test set
postResample(pred = predictions, obs = test_data$readmitted)

# Generate confusion matrix
confusionMatrix(predictions, test_data$readmitted)