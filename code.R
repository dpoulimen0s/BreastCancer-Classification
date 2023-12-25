# Libraries
library(mlbench)
library(dplyr)
library(ggplot2)
library(leaps)
library(glmnet)
library(bestglm)
library(corrplot)
library(plyr)
library(nclSLR)
library(tidyr)
library(combinat)
library(MASS)

# Load the data
data(BreastCancer)

# Data shape
dim(BreastCancer)

# Data structure
str(BreastCancer)

# Find the rows with missing values
rows_with_missing_values <- which(apply(BreastCancer, 1, function(row) any(is.na(row))))

# Count the number of rows with missing values
num_missing_rows <- length(rows_with_missing_values)

# Display the result
cat("The number of rows with missing values is:", num_missing_rows, "\n")


# Remove rows with missing values (NA)
New_BreastCancer <- na.omit(BreastCancer[, 2:11])

# Check again for rows with missing values (NA)
num_missing_rows_after_omitting <- sum(apply(New_BreastCancer, 1, function(row) any(is.na(row))))


# Display the result after omitting missing values
cat("After omitting missing values, the new number of rows with missing values is:", num_missing_rows_after_omitting, "\n")


# Create a vector with the predictor columns
predictor_columns <- c(1:9)

# Convert the selected columns to numeric
New_BreastCancer[, predictor_columns] = apply(New_BreastCancer[, predictor_columns], 2, function(x) as.numeric(as.character(x)))

# Check the updated data structure
str(New_BreastCancer)


# Convert the response variable to numeric (0 for 'benign', 1 for 'malignant')
y <- as.numeric(New_BreastCancer[, 10]) - 1

# Extract predictor variables
X1_original <- New_BreastCancer[, -10]

# Standardize predictor variables
X1 <- scale(X1_original)

# Combine standardized predictors and response variables in a new data frame
Breast_Cancer_Final <- data.frame(X1, y)


# Save the number of columns and rows for further use
p = ncol(X1)
n = nrow(X1)


# Display the distribution of classes in the response variable
class_distribution <- table(Breast_Cancer_Final$y)
print(class_distribution)


# Calculate the correlation matrix
correlation <- cor(Breast_Cancer_Final[,1:10])

# Create the correlation plot
corrplot(correlation, method = 'circle', type = 'lower', insig = 'blank',
         addCoef.col = 'black',tl.col = "black", number.cex = 0.9, diag = FALSE, col = COL2('RdYlBu'))


# Calculate the mean of columns for the 'benign' class
benign_class_means <- round(apply(Breast_Cancer_Final[Breast_Cancer_Final[, "y"] == 0, ], 2, mean), 3)
print(benign_class_means)


# Calculate the mean of columns for the 'malignant' class
malignant_class_means <- round(apply(Breast_Cancer_Final[Breast_Cancer_Final[, "y"] == 1, ], 2, mean), 3)
print(malignant_class_means)


# Perform best subset selection using BIC for logistic regression
best_subset_BIC_model <- bestglm(Breast_Cancer_Final, family = binomial,
                                 method = "exhaustive", nvmax = p)

# Retrieve the summary of the best subset selection model using BIC
bss_BIC_summary <- best_subset_BIC_model$Subsets 


# # Identify model with the lowest BIC
best_BIC = which.min(bss_BIC_summary$BIC)-1

# Print the index of the model with the lowest BIC
cat("The model with the lowest BIC is at index:", best_BIC, "\n")


# Perform best subset selection using AIC for logistic regression
best_subset_AIC_model = bestglm(Breast_Cancer_Final, family = binomial, 
                                method="exhaustive", nvmax=p, IC = "AIC")


# Retrieve the summary of the best subset selection model using AIC
bss_AIC_summary = best_subset_AIC_model$Subsets


# Identify model with the lowest AIC
best_AIC = which.min(bss_AIC_summary$AIC) - 1

# Print the index of the model with the lowest AIC
cat("The model with the lowest AIC is at index:", best_AIC, "\n")


# Set the seed to make the analysis reproducible
set.seed(1)

# Set the number of folds
nfolds = 10

# Sample fold-assignment index
fold_index = sample(nfolds, n, replace=TRUE)


# K-fold validation function for subset selection, Ridge, and LASSO
reg_cv = function(X1, y, fold_ind) {
  Xy = data.frame(X1, y=y)
  nfolds = max(fold_ind)
  if(!all.equal(sort(unique(fold_ind)), 1:nfolds)) stop("Invalid fold partition.")
  cv_errors = numeric(nfolds)
  for(fold in 1:nfolds) {
    glm_fit = glm(y ~ ., data=Xy[fold_ind!=fold,], family = binomial)
    phat = predict(glm_fit, Xy[fold_ind==fold,], type = "response")
    yhat = ifelse(phat > 0.5, 1, 0) 
    yobs = y[fold_ind == fold]
    cv_errors[fold] = 1 - mean(yobs == yhat)
  }
  fold_sizes = numeric(nfolds)
  for(fold in 1:nfolds){
    fold_sizes[fold] = length(which(fold_ind==fold))
    test_error = weighted.mean(cv_errors, w=fold_sizes)
    return(test_error)
  }
}


# Computes cross-validation test errors for regression models with best subset selection.
reg_bss_cv = function(X1, y, best_models, fold_index) {
  p = ncol(X1)
  test_errors = numeric(p)
  for(i in 1:p) {
    test_errors[i] = reg_cv(X1[,best_models[i,]], y, fold_index)
  }
  return(test_errors)
}

# Applying the best subset selection model to assess MSE via cross-validation
bss_mse <- reg_bss_cv(X1, y, as.matrix(best_subset_AIC_model$Subsets[2:10,2:10]), fold_index)


# Identify the model with the lowest cross-validation error
best_cv_model <- which.min(bss_mse)

cat("The model with the lowest error is at cross-validation index:", best_cv_model, "\n")


## Create a multi-panel plot to visualize the performance metrics with optimal predictor counts:
par(mfrow = c(1, 3))

# Plot 1: BIC
plot(1:9, bss_BIC_summary$BIC[2:10], xlab="Number of predictors", ylab="BIC", type="b")
points(best_BIC, bss_BIC_summary$BIC[best_BIC + 1], col="red", pch=16)

# Plot 2: AIC
plot(1:9, bss_AIC_summary$AIC[2:10], xlab="Number of predictors", ylab="AIC", type="b")
points(best_AIC, bss_AIC_summary$AIC[best_AIC + 1], col="red", pch=16)

# Plot 3: Test error
plot(1:9, bss_mse, xlab="Number of predictors", ylab="Test error", type="b")
points(best_cv_model, bss_mse[best_cv_model], col="red", pch=16)


# Create a logistic regression model with 4 predictor variables
glm_4_predictors <- glm(y ~ Cl.thickness + Cell.shape + Bare.nuclei + Bl.cromatin, data = Breast_Cancer_Final, family = binomial)

# Generate a summary of the model
summary_glm_4_predictors <- summary(glm_4_predictors)

# Print the coefficients of the selected logistic regression model
coefficients_glm_4_predictors <- summary_glm_4_predictors$coefficients
print(coefficients_glm_4_predictors)

# Print the lowest test error
cat("Lowest error value for Subset Selection:", bss_mse[4], "\n")

# Transforming the X1 unseen data to a dataframe for the predict function
X1_dataframe <- as.data.frame(scale(X1_original))

# Use the fitted ridge regression model to predict probabilities
phat_bss <- predict(glm_4_predictors, newdata = X1_dataframe, type = "response")

# Convert predicted probabilities to binary predictions
yhat_bss <- ifelse(phat_bss > 0.5, 1, 0)

# Create a confusion matrix
confusion_matrix_bss <- table(Actual = y, Predicted = yhat_bss)
print(confusion_matrix_bss)


# Choose a grid of values for the tuning parameter
grid <- 10^seq(5, -3, length = 500)

# Fit a ridge regression model for each value of the tuning parameter
ridge_fit <- glmnet(X1, y, alpha = 0, standardize = FALSE, lambda = grid, family = "binomial")


# Choose the appropriate tuning parameter using 10-fold cross-validation error 
# with the same folds as in subset selection
ridge_cv_fit <- cv.glmnet(X1, y, alpha = 0, standardize = FALSE, lambda = grid, nfolds = nfolds, foldid = fold_index,
                          family = "binomial", type.measure = "class")

# Create a 1x2 layout for plots
par(mfrow = c(1, 2))

# Plot ridge path
plot(ridge_fit, xvar = "lambda", col = 1:10, label = TRUE)

# Examine the effect of the tuning parameter on the MSE
plot(ridge_cv_fit)

# Identify the optimal value for the tuning parameter in ridge regression
lambda_ridge_min <- ridge_cv_fit$lambda.min

# Print the optimal lambda value
cat("The optimal value for the tuning parameter (lambda) in Ridge regression is:", lambda_ridge_min, "\n")


# Identify the index of the optimal lambda in the ridge regression model
which_lambda_ridge <- which(ridge_cv_fit$lambda == lambda_ridge_min)

# Print the rounded parameter estimates for the optimal value of the tuning parameter in ridge regression
rounded_ridge_parameter_estimates <- round(coef(ridge_fit, s = lambda_ridge_min), 3)
print(rounded_ridge_parameter_estimates)

# Fit a logistic regression model using all predictor variables
glm_fit <- glm(y ~ ., data = Breast_Cancer_Final, family = binomial)

# Round the coefficients of the logistic regression model
rounded_glm_fit_coefficients <- round(glm_fit$coefficients, 3)
print(rounded_glm_fit_coefficients)

# Obtain the corresponding cross-validation error for the final ridge regression model
ridge_mse <- ridge_cv_fit$cvm[which_lambda_ridge]

# Print the lowest test error
cat("Lowest error value for Ridge Regression:", ridge_mse, "\n")

# Use the fitted ridge regression model to predict probabilities
phat_ridge <- predict(ridge_fit, s = lambda_ridge_min, newx = X1, type = "response")

# Convert predicted probabilities to binary predictions
yhat_ridge <- ifelse(phat_ridge > 0.5, 1, 0)

# Create a confusion matrix
confusion_matrix_ridge <- table(Actual = y, Predicted = yhat_ridge)
print(confusion_matrix_ridge)

# Choose grid of values for the tuning parameter
grid1 = 10^seq(5, -3, length=500)

# Fit a LASSO regression for each value of the tuning parameter 
lasso_fit = glmnet(X1, y, alpha=1, standardize=FALSE, lambda=grid1, family = "binomial")


# Examine the effect of the tuning parameter on the parameter estimates 
par(mfrow=c(1,2))
plot(lasso_fit, xvar="lambda", col=1:10, label=TRUE)

# Compute 10-fold cross-validation error using the same folds as in ss and ridge regression
lasso_cv_fit = cv.glmnet(X1, y, alpha=1, standardize=FALSE, lambda=grid, nfolds=nfolds, foldid=fold_index,
                         family = "binomial", type.measure = "class")
plot(lasso_cv_fit)

# Identify the optimal value for the tuning parameter
lambda_lasso_min = lasso_cv_fit$lambda.min

# Print the optimal lambda value
cat("The optimal value for the tuning parameter (lambda) in LASSO regression is:", lambda_lasso_min, "\n")

# Print the rounded parameter estimates for the optimal value of the tuning parameter in lasso regression
rounded_lasso_parameter_estimates <- round(coef(lasso_fit, s = lambda_lasso_min), 3)
print(rounded_lasso_parameter_estimates)

# Corresponding cross validation error
which_lambda_lasso = which(lasso_cv_fit$lambda == lambda_lasso_min)
lasso_final_mse = lasso_cv_fit$cvm[which_lambda_lasso]

# Print the lowest test error
cat("Lowest error value for LASSO Regression:", lasso_final_mse, "\n")

# Use the fitted lasso regression model to predict probabilities
phat_lasso <- predict(lasso_fit, s = lambda_lasso_min, newx = X1, type = "response")

# Convert predicted probabilities to binary predictions
yhat_lasso <- ifelse(phat_lasso > 0.5, 1, 0)

# Create a confusion matrix
confusion_matrix_lasso <- table(Actual = y, Predicted = yhat_lasso)
print(confusion_matrix_lasso)

# Create a vector that contain the names of all the predictor variables 
colnames_vector <- colnames(Breast_Cancer_Final)[1:9]

# Now compute all the possible subsets of the variables 
subset_combinations = unlist(lapply(1:length(colnames_vector),  combinat::combn,
                                    x = colnames_vector, simplify = FALSE),
                                    recursive = FALSE)

# Cross validation function for LDA 
reg_cv_lda = function(x1, y, fold_ind){
  Xy = data.frame(x1, y=y)
  nfolds = max(fold_ind)
  if(!all.equal(sort(unique(fold_ind)), 1:nfolds)) stop("Invalid fold partition.")
  cv_errors = numeric(nfolds)
  for (fold in 1:nfolds) {
    tmp_fit = lda(y~., data = Xy[fold_ind!=fold,])
    phat = predict(tmp_fit, Xy[fold_ind == fold,])
    yhat = phat$class
    yobs = y[fold_ind==fold]
    cv_errors[fold] = 1 - mean(yobs == yhat)
  }
  fold_sizes = numeric(nfolds)
  for (fold in 1:nfolds) fold_sizes[fold] = length(which(fold_ind==fold))
  test_error_lda = weighted.mean(cv_errors, w=fold_sizes)
}

# Compute cross-validation test errors for LDA models with various predictor subsets
LDA_errors <- unlist(lapply(subset_combinations, function(subset) {
  selected_predictors <- as.data.frame(Breast_Cancer_Final[, unlist(subset)])
  reg_cv_lda(selected_predictors, y, fold_index)
}))

best_lda_subset_index <- which.min(LDA_errors)

# Print the index of the subset with the lowest cross-validation error for LDA
cat("The subset with the lowest cross-validation error for LDA is at index:", best_lda_subset_index, "\n")

# Print the predictor variables included in the best LDA subset
best_lda_subset <- subset_combinations[best_lda_subset_index]
print(best_lda_subset)

# Perform LDA
lda_fit <- lda(y ~ Cl.thickness + Cell.size + Epith.c.size + Bare.nuclei + Bl.cromatin + 
                 Mitoses, data = Breast_Cancer_Final)

# Print the summary of LDA
lda_fit

# Print the cross-validation error for the best LDA subset
best_lda_error <- LDA_errors[best_lda_subset_index]

cat("The cross-validation error for the best LDA subset is:", best_lda_error, "\n")

X1_dataframe <- as.data.frame(scale(X1_original))

# Use the fitted LDA model to predict probabilities
phat_lda <- predict(lda_fit, newdata = X1_dataframe, type="response")

# Extract the predicted class probabilities from the posterior component
phat_values <- phat_lda$posterior[, 2]  

# Convert predicted probabilities to binary predictions
yhat_lda <- ifelse(phat_values > 0.5, 1, 0)

# Create a confusion matrix
confusion_matrix_lda <- table(Actual = y, Predicted = yhat_lda)
print(confusion_matrix_lda)

# Cross validation function for QDA
reg_cv_qda = function(x1, y, fold_ind){
  Xy = data.frame(x1, y=y)
  nfolds = max(fold_ind)
  if(!all.equal(sort(unique(fold_ind)), 1:nfolds)) stop("Invalid fold partition.")
  cv_errors = numeric(nfolds)
  for (fold in 1:nfolds) {
    tmp_fit = qda(y~., data = Xy[fold_ind!=fold,])
    phat = predict(tmp_fit, Xy[fold_ind == fold,])
    yhat = phat$class
    yobs = y[fold_ind==fold]
    cv_errors[fold] = 1 - mean(yobs == yhat)
  }
  fold_sizes = numeric(nfolds)
  for (fold in 1:nfolds) fold_sizes[fold] = length(which(fold_ind==fold))
  test_error_lda = weighted.mean(cv_errors, w=fold_sizes)
}

# Compute cross-validation test errors for QDA models with various predictor subsets
QDA_errors <- unlist(lapply(subset_combinations, function(subset) {
  selected_predictors <- as.data.frame(Breast_Cancer_Final[, unlist(subset)])
  reg_cv_qda(selected_predictors, y, fold_index)
}))

best_qda_subset_index <- which.min(QDA_errors)

# Print the index of the subset with the lowest cross-validation error for LDA
cat("The subset with the lowest cross-validation error for LDA is at index:", best_qda_subset_index, "\n")

# Print the predictor variables included in the best LDA subset
best_qda_subset <- subset_combinations[best_qda_subset_index]
print(best_qda_subset)

# Perform QDA 
qda_fit = qda(y~ Cl.thickness+ Marg.adhesion + Epith.c.size  + Bare.nuclei, 
               data = Breast_Cancer_Final)

# Print the summary of QDA
qda_fit

# Print the cross-validation error for the best QDA subset
best_qda_error <- QDA_errors[best_qda_subset_index]

cat("The cross-validation error for the best QDA subset is:", best_qda_error, "\n")

X1_dataframe <- as.data.frame(scale(X1_original))

# Use the fitted LDA model to predict probabilities
phat_qda <- predict(qda_fit, newdata = X1_dataframe, type="response")

# Extract the predicted class probabilities from the posterior component
phat_values_qda <- phat_qda$posterior[, 2]  

# Convert predicted probabilities to binary predictions
yhat_qda <- ifelse(phat_values_qda > 0.5, 1, 0)

# Create a confusion matrix
confusion_matrix_qda <- table(Actual = y, Predicted = yhat_qda)
print(confusion_matrix_qda)

# Create a matrix with 4 columns and 1 row
error_values = matrix(c(round(bss_mse[4], 4), round(ridge_mse, 4), round(lasso_final_mse, 4), round(best_lda_error, 4), round(best_qda_error, 4)), ncol=5, byrow=TRUE)
 
# Specify the column names and row names of the matrix
colnames(error_values) = c('BSS-4','Ridge','LASSO','LDA', 'QDA')
rownames(error_values) <- c('Test Error')
 
# Convert the matrix to a data frame
final=as.table(error_values)
 
# Print the data frame
print(final, justify = "right")