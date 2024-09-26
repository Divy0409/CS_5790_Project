# MA5790 Semester Project
# Author: Hunter Malinowski, Divy Patel

# Install packages
install.packages(c("caret", "corrplot", "e1071", "lattice", "mlbench"))

# Load data
diabetes = read.csv("diabetic_data.csv")
str(diabetes)

# Replace ? with NA
diabetes[diabetes=="?"]<-NA

# Separate predictors
diabetes_pred <- diabetes[, -c(1:2, ncol(diabetes))]

# Convert categorical vars to dummy
library(caret)
# Continuous vars + categorical with only one level
con_cols = c("time_in_hospital", "num_lab_procedures", "num_procedures", 
         "num_medications", "number_outpatient", "number_emergency", 
         "number_inpatient", "number_diagnoses", "examide", "citoglipton")
cat_cols <- setdiff(names(diabetes_pred), con_cols)
diabetes_pred[, cat_cols] <- lapply(diabetes_pred[, cat_cols], as.factor)
formula <- as.formula(paste("~", paste(cat_cols, collapse = "+")))
dummRes <- dummyVars(formula, data=diabetes_pred, fullRank=TRUE)
diabetes_dum <- data.frame(predict(dummRes, newdata=diabetes_pred))
diabetes_result <- cbind(diabetes_pred[con_cols], diabetes_dum)

# Find near-zero variance
near_zero_idx <- nearZeroVar(diabetes_result, saveMetrics = TRUE)
near_zero_vars <- rownames(near_zero_idx[near_zero_idx$nzv, ])
print(near_zero_vars)

# Missing values
image(is.na(diabetes), main="Missing Values", xlab="Observation", ylab="Variable", xaxt="n", yaxt="n", bty="n", col=topo.colors(6))
axis(1,seq(0,1,length.out = nrow(diabetes)), 1:nrow(diabetes), col="white")

install.packages(c("naniar", "tidyverse"))
library(naniar)
print(n=100, miss_case_summary(diabetes))
miss_var_summary(diabetes)
n_miss(diabetes)

# Find high correlations
correlations <- cor(diabetes)
highCorr <- findCorrelation(correlations, cutoff=0.75, names=TRUE, exact=TRUE)
length(highCorr)

