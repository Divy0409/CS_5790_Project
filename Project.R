# MA5790 Semester Project
# Author: Hunter Malinowski, Divy Patel

# Install packages
install.packages(c("caret", "corrplot", "e1071", "lattice", "mlbench"))
library(corrr)
# Load data
diabetes = read.csv("diabetic_data.csv")
str(diabetes)

# Replace ? with NA
diabetes[diabetes=="?"]<-NA

# Separate predictors
diabetes_pred <- diabetes[, -c(1:2, ncol(diabetes))]

# Convert categorical vars to dummy
library(caret)
# Continuous vars + categorical with only one level or too many levels
con_cols = c("time_in_hospital", "num_lab_procedures", "num_procedures", 
         "num_medications", "number_outpatient", "number_emergency", 
         "number_inpatient", "number_diagnoses", "examide", "citoglipton", 
         "diag_1", "diag_2", "diag_3")
cat_cols <- setdiff(names(diabetes_pred), con_cols)
diabetes_pred[, cat_cols] <- lapply(diabetes_pred[, cat_cols], as.factor)
formula <- as.formula(paste("~", paste(cat_cols, collapse = "+")))
dummRes <- dummyVars(formula, data=diabetes_pred, fullRank=TRUE)
diabetes_dum <- data.frame(predict(dummRes, newdata=diabetes_pred))
diabetes_result <- cbind(diabetes_pred[con_cols], diabetes_dum)

# Find near-zero variance
near_zero_idx <- nearZeroVar(diabetes_result)
diabetes_nzv <- diabetes_result[, -near_zero_idx]

# Missing values
image(is.na(diabetes_nzv), main="Missing Values", xlab="Observation", ylab="Variable", xaxt="n", yaxt="n", bty="n", col=topo.colors(6))
axis(1,seq(0,1,length.out = nrow(diabetes_nzv)), 1:nrow(diabetes_nzv), col="white")

install.packages(c("naniar", "tidyverse"))
library(naniar)
print(n=100, miss_case_summary(diabetes_nzv))
miss_var_summary(diabetes_nzv)
n_miss(diabetes_nzv)

threshold <- 0.2
missing_percent <- colSums(is.na(diabetes_nzv)) / nrow(diabetes_nzv)
cols_to_remove <- names(missing_percent[missing_percent > threshold])
print(cols_to_remove)
diabetes_miss <- diabetes_nzv[, !(names(diabetes_nzv) %in% cols_to_remove)]


## need to impute before running this section, won't work w/NA

# Find high correlations
cols = c("time_in_hospital", "num_lab_procedures", "num_procedures", 
             "num_medications", "number_outpatient", "number_emergency", 
             "number_inpatient", "number_diagnoses")
diabetes_con = diabetes_nzv[, !(names(diabetes_nzv) %in% cols)]
head(diabetes_con)
correlations <- cor(diabetes_con)
highCorr <- findCorrelation(correlations, cutoff=0.75, names=TRUE, exact=TRUE)
length(highCorr)
