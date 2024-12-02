# Install necessary packages
install.packages(c("caret", "corrplot", "e1071", "lattice", "mlbench", "naniar", "tidyverse", "DMwR2"))
library(caret)
library(corrplot)
library(patchwork) # For combining plots
library(naniar)
library(tidyverse)
library(ggplot2)
library(e1071)
library(MASS)
library(dplyr)
library(tidyr)
library(DMwR2)
library(VIM)

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

# Summarize missing data
miss_var_summary(diabetes_pred)
n_miss(diabetes_pred)

# Remove columns with more than 20% missing values
threshold <- 0.2
missing_percent <- colSums(is.na(diabetes_pred)) / nrow(diabetes_pred)
cols_to_remove <- names(missing_percent[missing_percent > threshold])
print(cols_to_remove)

diabetes_clean <- diabetes_pred[, !(names(diabetes_pred) %in% cols_to_remove)]

# Recalculate cat_cols after removing columns with high missing values
cat_cols <- setdiff(names(diabetes_clean), con_cols)

# Ensure categorical columns are factors
diabetes_clean[cat_cols] <- lapply(diabetes_clean[cat_cols], as.factor)

# Perform kNN imputation for the entire dataset
imputed_data <- kNN(diabetes_clean, variable = names(diabetes_clean), k = 5)

# Check the imputed result
head(imputed_data)

# Remove columns ending with "imp"
imp_cols <- grep("imp$", colnames(imputed_data))
diabetes_clean <- imputed_data[, -imp_cols]

head(diabetes_clean)

# Save the cleaned dataset
write.csv(diabetes_clean, "diabetes_cleaned_NHC.csv", row.names = FALSE)

# Remove categorical columns with only one unique level
cat_cols <- cat_cols[sapply(diabetes_clean[cat_cols], function(x) nlevels(x) > 1)]

# Create dummy variables for categorical predictors
formula <- as.formula(paste("~", paste(cat_cols, collapse = "+")))
dummRes <- dummyVars(formula, data = diabetes_clean, fullRank = TRUE)
diabetes_dum <- data.frame(predict(dummRes, newdata = diabetes_clean))

# Combine continuous variables with dummy variables
diabetes_result <- cbind(diabetes_clean, diabetes_dum)

# Remove near-zero variance predictors
near_zero_idx <- nearZeroVar(diabetes_result)
diabetes_nzv <- diabetes_result[, -near_zero_idx]

write.csv(diabetes_nzv, "diabetes_cleaned_NHC.csv", row.names = FALSE)

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

# Skewness Analysis
skewness_values <- sapply(diabetes_con, skewness, na.rm = TRUE)
print("Skewness of continuous variables:")
print(skewness_values)

# Box-Cox Transformation
trans_boxcox <- preProcess(diabetes_con, method = c("center", "scale", "BoxCox"))
diabetes_transformed <- predict(trans_boxcox, diabetes_clean[con_cols])
str(diabetes_transformed)
skewness_values_transformed <- sapply(diabetes_transformed, skewness, na.rm = TRUE)
print("Skewness of transformed continuous variables:")
print(skewness_values_transformed)

# Combine the transformed continuous variables with the categorical variables
diabetes_final_transformed <- cbind(diabetes_transformed, diabetes_clean[cat_cols])

# Save the final transformed dataset
write.csv(diabetes_final_transformed, "diabetes_cleaned_NHC.csv", row.names = FALSE)

# Apply the spatial sign transformation to only continuous variables
transformed_spatial <- spatialSign(diabetes_final_transformed[con_cols])

# Combine the transformed spatial sign variables with the categorical variables and target
diabetes_final_spatial <- cbind(
  transformed_spatial, 
  diabetes_clean[cat_cols], 
  readmitted = diabetes$readmitted
)

# Check the structure of the final transformed dataset with spatial sign
str(diabetes_final_spatial)

# Save the final dataset with spatial sign transformation
write.csv(diabetes_final_spatial, "diabetes_cleaned_NHC.csv", row.names = FALSE)

diabetes <- read.csv("diabetes_cleaned_NHC.csv")
str(diabetes)

# Remove unnecessary columns
diabetes <- diabetes %>% dplyr::select(-c(admission_type_id, discharge_disposition_id, admission_source_id,diag_2,diag_3))
colnames(diabetes)

# Categorizqation 
diabetes$diag_1 <- as.character(diabetes$diag_1)

diabetes <- mutate(diabetes, primary_diagnosis =
                                          ifelse(str_detect(diag_1, "V") | str_detect(diag_1, "E"),"Other", 
                                          # disease codes starting with V or E are in “other” category;
                                          ifelse(str_detect(diag_1, "250"), "Diabetes",
                                          ifelse((as.integer(diag_1) >= 390 & as.integer(diag_1) <= 459) | as.integer(diag_1) == 785, "Circulatory",
                                          ifelse((as.integer(diag_1) >= 460 & as.integer(diag_1) <= 519) | as.integer(diag_1) == 786, "Respiratory", 
                                          ifelse((as.integer(diag_1) >= 520 & as.integer(diag_1) <= 579) | as.integer(diag_1) == 787, "Digestive", 
                                          ifelse((as.integer(diag_1) >= 580 & as.integer(diag_1) <= 629) | as.integer(diag_1) == 788, "Genitourinary",
                                          ifelse((as.integer(diag_1) >= 140 & as.integer(diag_1) <= 239), "Neoplasms",  
                                          ifelse((as.integer(diag_1) >= 710 & as.integer(diag_1) <= 739), "Musculoskeletal",          
                                          ifelse((as.integer(diag_1) >= 800 & as.integer(diag_1) <= 999), "Injury",                    
                                          "Other"))))))))))

# Drop the original diag_1 column
diabetes <- diabetes %>% dplyr::select(-diag_1)

diabetes$primary_diagnosis <- as.factor(diabetes$primary_diagnosis)
table(diabetes$primary_diagnosis)


diabetes$readmitted <- case_when(diabetes$readmitted %in% c(">30", "NO") ~ "NO",
                                diabetes$readmitted %in% c("<30") ~ "YES")
diabetes$readmitted <- as.factor(diabetes$readmitted)
levels(diabetes$readmitted)

table(diabetes$readmitted)
str(diabetes)

# Remove rows where gender is "Unknown/Invalid"
diabetes <- subset(diabetes, gender != "Unknown/Invalid")

table(diabetes$gender)
table(diabetes$readmitted)

cat_vars <- diabetes %>% dplyr::select(where(is.character))  # Select character columns
sapply(cat_vars, unique)  # Check unique values for each

nzv <- nearZeroVar(diabetes, saveMetrics = TRUE)
nzv_indices <- nearZeroVar(diabetes)  # Indices of near-zero variance columns
nzv_indices
diabetes <- diabetes[, -nzv_indices]  # Remove these columns

# Save the cleaned data
write.csv(diabetes, "diabetes_cleaned_NHC.csv", row.names = FALSE)
