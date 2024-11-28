# Install necessary packages
install.packages(c("caret", "corrplot", "e1071", "lattice", "mlbench", "naniar", "tidyverse","DMwR2"))
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
length(cat_cols)
# name of the categorical columns
cat_cols

# # Create histograms for each continuous predictor variable
# par(mfrow = c(3, 3))  # Arrange the plots in a 3x3 grid

# for (var in con_cols) {
#   hist_data <- hist(diabetes[[var]], 
#                     main = paste("Histogram of", var), 
#                     xlab = var, 
#                     col = "lightblue", 
#                     border = "black", 
#                     probability = TRUE)  # Set probability = TRUE for density line
  
#   mean_val <- mean(diabetes[[var]], na.rm = TRUE)
#   sd_val <- sd(diabetes[[var]], na.rm = TRUE)
  
#   curve(dnorm(x, mean = mean_val, sd = sd_val), 
#         col = "red", 
#         lwd = 2, 
#         add = TRUE)
# }

# # Visualize missing values
# image(is.na(diabetes_pred), main = "Missing Values", xlab = "Observation", 
#       ylab = "Variable", xaxt = "n", yaxt = "n", bty = "n", col = topo.colors(6))
# axis(1, seq(0, 1, length.out = nrow(diabetes_pred)), 1:nrow(diabetes_pred), col = "white")

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
write.csv(diabetes_clean, "diabetes_cleaned_impute.csv", row.names = FALSE)

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
diabetes_result <- cbind(diabetes_clean, diabetes_dum)

# Remove near-zero variance predictors
near_zero_idx <- nearZeroVar(diabetes_result)
diabetes_nzv <- diabetes_result[, -near_zero_idx]

write.csv(diabetes_nzv, "diabetes_cleaned_impute_dummies.csv", row.names = FALSE)

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

# Calculate skewness for each continuous variable
skewness_values <- sapply(diabetes_final, skewness, na.rm = TRUE)

# Display the skewness values
print("Skewness of continuous variables:")
print(skewness_values)

# Define the continuous variables
con_cols <- c("time_in_hospital", "num_lab_procedures", "num_procedures", 
              "num_medications", "number_outpatient", "number_emergency", 
              "number_inpatient", "number_diagnoses")

# Ensure the continuous columns are available in the dataset
diabetes_clean_con <- diabetes_clean[, con_cols]

# Apply Box-Cox transformation, centering, and scaling to the continuous variables
trans_boxcox <- preProcess(diabetes_clean_con, method = c("center", "scale", "BoxCox"))

# Apply the transformations to the continuous variables
diabetes_transformed <- predict(trans_boxcox, diabetes_clean[con_cols])

# View the transformed data
str(diabetes_transformed)
skewness_values_transformed <- sapply(diabetes_transformed, skewness, na.rm = TRUE)
print("Skewness of transformed continuous variables:")
print(skewness_values_transformed)

# Combine the transformed continuous variables with the categorical variables
diabetes_final_transformed <- cbind(diabetes_transformed, diabetes_clean[cat_cols])

# Check the structure of the final transformed dataset
str(diabetes_final_transformed)

# Create histograms for each continous predictor variable
par(mfrow = c(3, 3))  # Arrange the plots in a 3x3 grid

for (var in con_cols) {
  hist_data <- hist(diabetes_final_transformed[[var]], 
                    main = paste("Histogram of", var), 
                    xlab = var, 
                    col = "lightblue", 
                    border = "black", 
                    probability = TRUE)  # Set probability = TRUE for density line
  
  mean_val <- mean(diabetes_final_transformed[[var]], na.rm = TRUE)
  sd_val <- sd(diabetes_final_transformed[[var]], na.rm = TRUE)
  
  curve(dnorm(x, mean = mean_val, sd = sd_val), 
        col = "red", 
        lwd = 2, 
        add = TRUE)
}

# Boxplots to identify outliers
boxplots <- lapply(con_cols, function(var) {
  ggplot(diabetes_final_transformed, aes_string(y = var)) +
    geom_boxplot(fill = 'lightblue', color = 'black') +
    labs(title = paste("Boxplot of", var)) +
    theme_minimal()
})

# Create histograms for each continous predictor variable
par(mfrow = c(3, 3))  # Arrange the plots in a 3x3 grid

for (var in con_cols) {
  hist_data <- hist(diabetes_final_transformed[[var]], 
                    main = paste("Histogram of", var), 
                    xlab = var, 
                    col = "lightblue", 
                    border = "black", 
                    probability = TRUE)  # Set probability = TRUE for density line
  
  mean_val <- mean(diabetes_final_transformed[[var]], na.rm = TRUE)
  sd_val <- sd(diabetes_final_transformed[[var]], na.rm = TRUE)
  
  curve(dnorm(x, mean = mean_val, sd = sd_val), 
        col = "red", 
        lwd = 2, 
        add = TRUE)
}

# Arrange all boxplots in a 3x3 grid using patchwork
boxplot_grid <- wrap_plots(boxplots, nrow = 3, ncol = 3)
print(boxplot_grid)

# Apply the spatial sign transformation to only continuous variables
transformed_spatial <- spatialSign(diabetes_final_transformed[con_cols])

# Combine the transformed spatial sign variables with the categorical variables
diabetes_final_spatial <- cbind(transformed_spatial, diabetes_clean[cat_cols],readmitted = diabetes$readmitted)

# Check the structure of the final transformed dataset with spatial sign
str(diabetes_final_spatial)


# Boxplots to identify outliers
boxplots <- lapply(con_cols, function(var) {
  ggplot(diabetes_final_spatial, aes_string(y = var)) +
    geom_boxplot(fill = 'lightblue', color = 'black') +
    labs(title = paste("Boxplot of", var)) +
    theme_minimal()
})

# Arrange all boxplots in a 3x3 grid using patchwork
boxplot_grid <- wrap_plots(boxplots, nrow = 3, ncol = 3)
print(boxplot_grid)

# Create histograms for each continous predictor variable
par(mfrow = c(3, 3))  # Arrange the plots in a 3x3 grid

for (var in con_cols) {
  hist_data <- hist(diabetes_final_spatial[[var]], 
                    main = paste("Histogram of", var), 
                    xlab = var, 
                    col = "lightblue", 
                    border = "black", 
                    probability = TRUE)  # Set probability = TRUE for density line
  
  mean_val <- mean(diabetes_final_spatial[[var]], na.rm = TRUE)
  sd_val <- sd(diabetes_final_spatial[[var]], na.rm = TRUE)
  
  curve(dnorm(x, mean = mean_val, sd = sd_val), 
        col = "red", 
        lwd = 2, 
        add = TRUE)
}

# For each categorical variable, create a bar plot showing the distribution of categories
# Set up an empty list to store the bar plots
barplots <- list()



# Loop through categorical columns and create bar plots
for (var in cat_cols) {
  p <- ggplot(diabetes_final_spatial, aes_string(x = var)) +
    geom_bar(fill = 'lightblue', color = 'black') +
    labs(title = paste("Barplot of", var)) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  # Add the created plot to the list
  barplots[[var]] <- p
}

# Use patchwork to arrange the bar plots in a grid layout
# Adjust the number of rows and columns as needed
barplot_grid <- wrap_plots(barplots, nrow = 9, ncol = 4)

# Print the grid of barplots
print(barplot_grid)

# PCA
trans_pca <- preProcess(transformed_spatial, method = "pca")
trans_pca
transformed_pca <- predict(trans_pca, transformed_spatial)  
dim(transformed_spatial)
dim(transformed_pca)

# Save the cleaned data 
write.csv(diabetes_final_spatial, "diabetes_cleaned.csv", row.names = FALSE)

diabetes <- read.csv("diabetes_cleaned.csv")
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

# Save the cleaned data
write.csv(diabetes, "diabetes_cleaned.csv", row.names = FALSE)

