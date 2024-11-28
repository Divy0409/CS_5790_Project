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


# Load your cleaned dataset
diabetes <- read.csv("diabetes_cleaned.csv")
str(diabetes)
table(diabetes$gender)
# Remove rows where gender is "Unknown/Invalid"
diabetes <- subset(diabetes, gender != "Unknown/Invalid")
table(diabetes$gender)
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

control <- trainControl(method = "repeatedcv", number = 5, repeats = 3, sampling = "down", classProbs = TRUE, summaryFunction = twoClassSummary)

tuneGrid <- expand.grid(k = 1:10)

# Train a binomial logistic regression model using caret
knn_model <- train(readmitted ~ ., data = train_data, 
                   method = "knn", tuneGrid = tuneGrid,
                   trControl = control, preProcess=c("center", "scale"))
knn_model

# Make predictions on the test set
# Ensure `test_data$readmitted` is a factor with correct levels
test_data$readmitted <- factor(test_data$readmitted, levels = c("NO", "YES"))

# Make predictions and convert to factor
predictions <- factor(predict(knn_model, newdata = test_data), levels = c("NO", "YES"))

# Generate confusion matrix
confusionMatrix(predictions, test_data$readmitted)

# Plot ROC curve
roc_curve <- roc(test_data$readmitted, as.numeric(predictions))
plot(roc_curve, col = "blue", lwd = 2, main = "ROC Curve")
lines(x = c(0, 1), y = c(0, 1), col = "red", lty = 2)
auc(roc_curve)

