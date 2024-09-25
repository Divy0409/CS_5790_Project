# MA5790 Semester Project
# Author: Hunter Malinowski, Divy Patel

# Install packages
install.packages(c("caret", "corrplot", "e1071", "lattice", "mlbench"))

# Load data
diabetes = read.csv("diabetic_data.csv")
str(diabetes)

# Replace ? with NA
diabetes[diabetes=="?"]<-NA

# Convert non-numeric columns

# Missing values
image(is.na(diabetes), main="Missing Values", xlab="Observation", ylab="Variable", xaxt="n", yaxt="n", bty="n", col=topo.colors(6))
axis(1,seq(0,1,length.out = nrow(diabetes)), 1:nrow(diabetes), col="white")

install.packages(c("naniar", "tidyverse"))
library(naniar)
print(n=100, miss_case_summary(diabetes))
miss_var_summary(diabetes)
n_miss(diabetes)

colNames <- colnames(diabetes)
print(colNames)
for(col in colNames) {
  print(col)
  print(var(diabetes[, col]), na.rm=TRUE)
}

# Find high correlations
correlations <- cor(diabetes)
highCorr <- findCorrelation(correlations, cutoff=0.75, names=TRUE, exact=TRUE)
length(highCorr)

