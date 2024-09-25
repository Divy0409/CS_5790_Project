# MA5790 Semester Project
# Author: Hunter Malinowski, Divy Patel

# Install packages
install.packages(c("caret", "corrplot", "e1071", "lattice", "mlbench"))

# Load data
diabetes = read.csv("diabetic_data.csv")
str(diabetes)
