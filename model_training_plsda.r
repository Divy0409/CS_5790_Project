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
diabetes <- subset(diabetes, gender != "Unknown/Invalid")
table(diabetes$readmitted)
cat_vars <- diabetes %>% dplyr::select(where(is.character))  # Select character columns
sapply(cat_vars, unique)  # Check unique values for each

nzv <- nearZeroVar(diabetes, saveMetrics = TRUE)
nzv_indices <- nearZeroVar(diabetes)  # Indices of near-zero variance columns
diabetes <- diabetes[, -nzv_indices]  # Remove these columns




# Split the data into training and testing sets (70-30 split)
set.seed(123)  # For reproducibility
trainIndex <- createDataPartition(diabetes$readmitted, p = 0.7, 
                                  list = FALSE)
train_data <- diabetes[trainIndex, ]
test_data <- diabetes[-trainIndex, ]

control <- trainControl(method = "repeatedcv", number = 5, repeats = 3, sampling = "down", classProbs = TRUE, summaryFunction = defaultSummary)

# Train a binomial logistic regression model using caret
plsda_model <- train(readmitted ~ ., data = train_data, 
                   method = "pls", metric = "Kappa",
                   trControl = control, preProcess=c("center", "scale"))
plsda_model
plot(plsda_model)

# Make predictions on the test set
# Ensure `test_data$readmitted` is a factor with correct levels
test_data$readmitted <- factor(test_data$readmitted, levels = c("NO", "YES"))

# Make predictions and convert to factor
predictions <- factor(predict(plsda_model, newdata = test_data), levels = c("NO", "YES"))

# Generate confusion matrix
confusionMatrix(predictions, test_data$readmitted)

