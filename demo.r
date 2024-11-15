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

# Create histograms for each continuous predictor variable
par(mfrow = c(3, 3))  # Arrange the plots in a 3x3 grid

for (var in con_cols) {
  hist_data <- hist(diabetes[[var]], 
                    main = paste("Histogram of", var), 
                    xlab = var, 
                    col = "lightblue", 
                    border = "black", 
                    probability = TRUE)  # Set probability = TRUE for density line
  
  mean_val <- mean(diabetes[[var]], na.rm = TRUE)
  sd_val <- sd(diabetes[[var]], na.rm = TRUE)
  
  curve(dnorm(x, mean = mean_val, sd = sd_val), 
        col = "red", 
        lwd = 2, 
        add = TRUE)
}

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
# Perform k-NN imputation on continuous variables
preprocess_knn <- preProcess(diabetes_clean[con_cols], method = "knnImpute")

# Apply the k-NN imputation
diabetes_clean_knn <- predict(preprocess_knn, diabetes_clean[con_cols])

# Check the structure of the imputed dataset
str(diabetes_clean_knn)
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
diabetes_result <- cbind(diabetes_clean_knn, diabetes_dum)

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

# Define the continuous variables
con_cols <- c("time_in_hospital", "num_lab_procedures", "num_procedures", 
              "num_medications", "number_outpatient", "number_emergency", 
              "number_inpatient", "number_diagnoses")

# Calculate skewness for each continuous variable
skewness_values <- sapply(diabetes_clean_knn, skewness, na.rm = TRUE)

# Display the skewness values
print("Skewness of continuous variables:")
print(skewness_values)

# Apply Box-Cox transformation, centering, and scaling to the continuous variables
trans_boxcox <- preProcess(diabetes_clean_knn, method = c("center", "scale", "BoxCox"))

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
transformed_spatial <- spatialSign(diabetes_transformed)

# Combine the transformed spatial sign variables with the categorical variables
diabetes_final_spatial <- cbind(transformed_spatial, diabetes_clean[cat_cols])

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
write.csv(diabetes_final, "diabetes_cleaned.csv", row.names = FALSE)


