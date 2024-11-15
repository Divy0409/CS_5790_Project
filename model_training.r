# Load Libraries
library(caret)

# Load Clean Data
data <- read.csv("diabetes_cleaned.csv")

# Split data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(data$readmitted, p = .8, 
                                  list = FALSE, 
                                  times = 1)
data_train <- data[ trainIndex,]
data_test  <- data[-trainIndex,]

# Train a Logistic Regression Model
model <- train(readmitted)