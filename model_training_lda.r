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


# Load your cleaned dataset
diabetes <- read.csv("diabetes_cleaned.csv")
str(diabetes)
table(diabetes$gender)

# Split the data into training and testing sets (70-30 split)
set.seed(123)  # For reproducibility
trainIndex <- createDataPartition(diabetes$readmitted, p = 0.7, 
                                  list = FALSE)
train_data <- diabetes[trainIndex, ]
test_data <- diabetes[-trainIndex, ]

control <- trainControl(method = "cv", number = 10,  sampling = "down", classProbs = TRUE, summaryFunction = defaultSummary)

# Train a binomial logistic regression model using caret
lda_model <- train(readmitted ~ ., data = train_data, 
                   method = "lda",
                   trControl = control,
                   preProcess=c("center", "scale"),
                   metric = "Kappa")
lda_model

# Make predictions on the test set
# Ensure `test_data$readmitted` is a factor with correct levels
test_data$readmitted <- factor(test_data$readmitted, levels = c("NO", "YES"))

# Make predictions and convert to factor
predictions <- factor(predict(lda_model, newdata = test_data), levels = c("NO", "YES"))

# Statistics for test set
postResample(pred = predictions, obs = test_data$readmitted)

# Generate confusion matrix
confusionMatrix(predictions, test_data$readmitted)



