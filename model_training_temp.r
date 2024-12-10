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
str(diabetes_pred)

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

# Save the cleaned dataset before NZV removal
write.csv(diabetes_clean, "diabetes_cleaned_impute.csv", row.names = FALSE)

# Remove near-zero variance predictors
nzv_indices <- nearZeroVar(diabetes_clean)
diabetes_nzv <- diabetes_clean[, -nzv_indices]

# Save the dataset after NZV removal
write.csv(diabetes_nzv, "diabetes_nzv_removed.csv", row.names = FALSE)

# Remove unnecessary columns and recategorize
diabetes <- diabetes_nzv %>% dplyr::select(-c(admission_type_id, discharge_disposition_id, admission_source_id, diag_2, diag_3))
colnames(diabetes)

# Categorize diag_1 into primary diagnosis categories
diabetes$diag_1 <- as.character(diabetes$diag_1)
diabetes <- mutate(diabetes, primary_diagnosis =
                     ifelse(str_detect(diag_1, "V") | str_detect(diag_1, "E"), "Other", 
                     ifelse(str_detect(diag_1, "250"), "Diabetes",
                     ifelse((as.integer(diag_1) >= 390 & as.integer(diag_1) <= 459) | as.integer(diag_1) == 785, "Circulatory",
                     ifelse((as.integer(diag_1) >= 460 & as.integer(diag_1) <= 519) | as.integer(diag_1) == 786, "Respiratory", 
                     ifelse((as.integer(diag_1) >= 520 & as.integer(diag_1) <= 579) | as.integer(diag_1) == 787, "Digestive", 
                     ifelse((as.integer(diag_1) >= 580 & as.integer(diag_1) <= 629) | as.integer(diag_1) == 788, "Genitourinary",
                     ifelse((as.integer(diag_1) >= 140 & as.integer(diag_1) <= 239), "Neoplasms",  
                     ifelse((as.integer(diag_1) >= 710 & as.integer(diag_1) <= 739), "Musculoskeletal",          
                     ifelse((as.integer(diag_1) >= 800 & as.integer(diag_1) <= 999), "Injury", "Other"))))))))))
# Drop diag_1 column
diabetes <- diabetes %>% dplyr::select(-diag_1)

# Convert primary_diagnosis to factor
diabetes$primary_diagnosis <- as.factor(diabetes$primary_diagnosis)
table(diabetes$primary_diagnosis)

# Recode readmitted variable
diabetes$readmitted <- case_when(diabetes$readmitted %in% c(">30", "NO") ~ "NO",
                                 diabetes$readmitted %in% c("<30") ~ "YES")
diabetes$readmitted <- as.factor(diabetes$readmitted)
levels(diabetes$readmitted)

# Remove rows where gender is "Unknown/Invalid"
diabetes <- subset(diabetes, gender != "Unknown/Invalid")

# Save intermediate dataset
write.csv(diabetes, "diabetes_categorized.csv", row.names = FALSE)

# Correlation Analysis
num_cols <- con_cols
correlations <- cor(diabetes[num_cols], use = "complete.obs")
corrplot(correlations, method = "circle", type = "upper", tl.cex = 0.8)

# Remove highly correlated variables
highCorr <- findCorrelation(correlations, cutoff = 0.75, names = TRUE)
diabetes <- diabetes[, !(names(diabetes) %in% highCorr)]

# Skewness and Transformations
diabetes_con <- diabetes[, con_cols]
trans_boxcox <- preProcess(diabetes_con, method = c("center", "scale", "BoxCox"))
diabetes_transformed <- predict(trans_boxcox, diabetes_con)

# Spatial sign transformation
spatial_sign_trans <- spatialSign(diabetes_transformed)
diabetes <- cbind(spatial_sign_trans, diabetes %>% dplyr::select(-con_cols))

# Create dummy variables
cat_cols <- names(diabetes %>% dplyr::select(where(is.factor)))
formula <- as.formula(paste("~", paste(cat_cols, collapse = "+")))
dummRes <- dummyVars(formula, data = diabetes, fullRank = TRUE)
diabetes_dum <- data.frame(predict(dummRes, newdata = diabetes))

# Combine dummy variables with continuous variables
diabetes_final <- cbind(diabetes %>% dplyr::select(where(is.numeric)), diabetes_dum)

# Save the final dataset
write.csv(diabetes_final, "diabetes_final_with_dummies.csv", row.names = FALSE)
str(diabetes_final)

