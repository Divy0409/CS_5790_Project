# Install necessary packages
install.packages(c("caret", "corrplot", "e1071", "lattice", "mlbench", "naniar", "tidyverse"))
library(caret)
library(corrplot)
library(corrr)
library(naniar)
library(tidyverse)

# Load data
diabetes <- read.csv("diabetic_data.csv")
str(diabetes)

# Replace ? with NA
diabetes[diabetes == "?"] <- NA

# Separate predictor variables
diabetes_pred <- diabetes[, -c(1:2, ncol(diabetes))]  # Exclude ID columns and the target column

# Separate continuous variables and categorical variables
con_cols <- c("time_in_hospital", "num_lab_procedures", "num_procedures", 
              "num_medications", "number_outpatient", "number_emergency", 
              "number_inpatient", "number_diagnoses")

cat_cols <- setdiff(names(diabetes_pred), con_cols)

# Visualize missing values
image(is.na(diabetes_pred), main = "Missing Values", xlab = "Observation", 
      ylab = "Variable", xaxt = "n", yaxt = "n", bty = "n", col = topo.colors(6))
axis(1, seq(0, 1, length.out = nrow(diabetes_pred)), 1:nrow(diabetes_pred), col = "white")

# Summarize missing data
miss_var_summary(diabetes_pred)
n_miss(diabetes_pred)

# Remove columns with more than 20% missing values
threshold <- 0.2
missing_percent <- colSums(is.na(diabetes_pred)) / nrow(diabetes_pred)
cols_to_remove <- names(missing_percent[missing_percent > threshold])
print(cols_to_remove)

diabetes_clean <- diabetes_pred[, !(names(diabetes_pred) %in% cols_to_remove)]

# Impute missing data (median for numeric, mode for categorical)
# For numeric columns
diabetes_clean[con_cols] <- lapply(diabetes_clean[con_cols], function(x) {
  ifelse(is.na(x), median(x, na.rm = TRUE), x)
})

# For categorical columns (impute with mode or create a 'missing' category)
# Re-identify categorical columns after removing columns with too many missing values
cat_cols <- setdiff(names(diabetes_clean), con_cols)

# Impute missing values for categorical columns by creating a 'Missing' category
diabetes_clean[cat_cols] <- lapply(diabetes_clean[cat_cols], function(x) {
  x <- as.factor(x)
  # Replace NA with "Missing" category
  ifelse(is.na(x), 'Missing', as.character(x))  
})

# Convert the factor columns back to factor
diabetes_clean[cat_cols] <- lapply(diabetes_clean[cat_cols], as.factor)

# Remove categorical columns with only one unique level
cat_cols <- cat_cols[sapply(diabetes_clean[cat_cols], function(x) nlevels(x) > 1)]

# Now create dummy variables for categorical predictors
formula <- as.formula(paste("~", paste(cat_cols, collapse = "+")))
dummRes <- dummyVars(formula, data = diabetes_clean, fullRank = TRUE)
diabetes_dum <- data.frame(predict(dummRes, newdata = diabetes_clean))

# Combine continuous variables with dummy variables
diabetes_result <- cbind(diabetes_clean[con_cols], diabetes_dum)

# Remove near-zero variance predictors
near_zero_idx <- nearZeroVar(diabetes_result)
diabetes_nzv <- diabetes_result[, -near_zero_idx]

# Correlation Analysis
# Separate the numeric columns for correlation analysis
num_cols <- c("time_in_hospital", "num_lab_procedures", "num_procedures", 
              "num_medications", "number_outpatient", "number_emergency", 
              "number_inpatient", "number_diagnoses")
diabetes_con <- diabetes_nzv[, num_cols]

# Calculate correlations
correlations <- cor(diabetes_con, use = "complete.obs")


# Visualize correlation matrix
corrplot(correlations, method = "circle", type = "upper", tl.cex = 0.8)

# Find highly correlated variables (correlation > 0.75)
highCorr <- findCorrelation(correlations, cutoff = 0.75, names = TRUE)
print(length(highCorr))
print(highCorr)

# Remove highly correlated variables
diabetes_final <- diabetes_nzv[, !(names(diabetes_nzv) %in% highCorr)]
str(diabetes_final)

