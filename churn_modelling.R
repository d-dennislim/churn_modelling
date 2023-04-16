# Import
library(dplyr)
library(DataExplorer)
library(caTools)
library(mice)
library(caret)
library(ROCR)

setwd("C:/Users/Jeff/Desktop/AML")

# Import data
df <- read.csv("churn.csv")

# Initial Data Exploration
glimpse(df)

df$avg_frequency_login_days <- as.double(df$avg_frequency_login_days)
# df$joining_date <- as.Date(df$joining_date)
df <- as.data.frame(unclass(df), stringsAsFactors = TRUE)
summary(df)

plot_str(df)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Exploratory Data Analysis
# age
hist(df$age, breaks = 15)

# gender
tbl <- with(df, table(gender, churn_risk_score))
ggplot(as.data.frame(tbl), aes(gender, Freq, fill = churn_risk_score)) +     
  geom_col(position = 'dodge') + ggtitle('Frequency Plot of Gender')

# region
tbl <- with(df, table(region_category, churn_risk_score))
ggplot(as.data.frame(tbl), aes(region_category, Freq, fill = churn_risk_score)) +     
  geom_col(position = 'dodge') + ggtitle('Frequency Plot of Region')

# membership
tbl <- with(df, table(membership_category, churn_risk_score))
lvl <- c('No Membership', ' Basic Membership', 'Silver Membership', 
         'Gold Membership', 'Platinum Membership', 'Premium Membership')

p <- ggplot(df)
p + geom_bar(aes(x = factor(membership_category, level = lvl)))

ggplot(as.data.frame(tbl), 
       aes(membership_category, Freq, fill = churn_risk_score, level = lvl)) +     
  geom_col(position = 'dodge') + ggtitle('Frequency Plot of Membership') + 
  guides(x =  guide_axis(angle = 30)) + labs()

# average transaction value
hist(df$avg_transaction_value, main = 'Histogram of Average Transaction Value')

# joined through referral
tbl <- with(df, table(joined_through_referral, churn_risk_score))
ggplot(as.data.frame(tbl), aes(joined_through_referral, Freq, fill = churn_risk_score)) +     
  geom_col(position = 'dodge') + ggtitle('Frequency Plot of Referred')

# points in wallet
hist(df$points_in_wallet, main = 'Histogram of Customer Points')

# past complaints
tbl <- with(df, table(past_complaint, churn_risk_score))
ggplot(as.data.frame(tbl), aes(past_complaint, Freq, fill = churn_risk_score)) +     
  geom_col(position = 'dodge') + ggtitle('Frequency Plot of Past Complaints')

# complaint status
tbl <- with(df, table(complaint_status, churn_risk_score))
ggplot(as.data.frame(tbl), aes(complaint_status, Freq, fill = churn_risk_score)) +     
  geom_col(position = 'dodge') + ggtitle('Frequency Plot of Complaint Status') +
  guides(x =  guide_axis(angle = 45))

# target
ggplot(df, aes(x = churn_risk_score)) +
  geom_bar()


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Data pre-processing
# Check for duplicate data
duplicated(df) %>% table()

# Replacing error values with NA
df[df == "?" | df == "Error" | df == "xxxxxxxx" | df == 'Unknown'] <- NA 
df <- mutate_all(df, na_if, "") # mutate empty values to NA 

# Replacing negative values with NA
df$avg_frequency_login_days[!is.na(df$avg_frequency_login_days) &
                              df$avg_frequency_login_days < 0] <- NA

df$days_since_last_login[df$days_since_last_login < 0] <- NA

df$avg_time_spent[df$avg_time_spent < 0] <- NA

df$points_in_wallet[df$points_in_wallet < 0] <- NA


# Removing redundant features
df <- select(df, -c(X, security_no, last_visit_time))

# Missing values
plot_missing(df)
colSums(is.na(df))

# Changing data type
df$avg_frequency_login_days <- as.double(df$avg_frequency_login_days)
df$joining_date <- as.Date(df$joining_date)
df <- as.data.frame(unclass(df), stringsAsFactors = TRUE)

glimpse(df)
summary(df)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Feature engineering
df$joindays <- 
  Sys.Date()-df$joining_date
df$joindays <- as.numeric(df$joindays)

# Scaling (min-max scaling)
process <- preProcess(df, method = 'range')
df_norm <- predict(process, df)

# Imputation
# Fill NA value according to availability of referral ID
df_norm[is.na(df$joined_through_referral),] <- 
  df_norm %>% filter(is.na(joined_through_referral)) %>% 
  mutate(joined_through_referral = if_else(!is.na(referral_id), "Yes", "No"))

df_norm <- select(df_norm, -c(referral_id, joining_date))

# Imputation using MICE
impute <- mice(df_norm, method='pmm', seed = 123)
ds <- complete(impute, 2)
sum(is.na(ds))
str(ds)
summary(ds)
# write.csv(ds, 'complete_data.csv')

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ds <- read.csv('complete_data.csv')
ds <- ds[,-1]
ds <- as.data.frame(unclass(ds), stringsAsFactors = TRUE)
str(ds)

# Variable encoding
dmy <- dummyVars("~.", data = ds, fullRank = T)
da <- data.frame(predict(dmy, newdata = ds))

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Class Balance check
prop.table(table(da$churn_risk_score))

# Splitting dataset
set.seed(123)

split = sample.split(da$churn_risk_score, SplitRatio = 0.7)
train_set= subset(da, split == TRUE)
test_set= subset(da, split == FALSE)

prop.table(table(train_set$churn_risk_score))
prop.table(table(test_set$churn_risk_score))

# Factoring target labels
train_set$churn_risk_score <- as.factor(train_set$churn_risk_score)
test_set$churn_risk_score <- as.factor(test_set$churn_risk_score)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Logistic Regression (LR)
set.seed(111)
lr_base = glm(churn_risk_score ~.,
                 train_set,
                 family = binomial)
summary(lr_base)

# Predicting on Train set 
lr_train <- predict(lr_base, type = 'response', train_set[ ,-37])
lr_train = ifelse(lr_train > 0.5, 1, 0)

confusionMatrix(as.factor(lr_train), 
                train_set$churn_risk_score)

# Predicting on Test set 
lr_test <- predict(lr_base, type = 'response', test_set[ ,-37])
lr_test = ifelse(lr_test > 0.5, 1, 0)

confusionMatrix(as.factor(lr_test), 
                test_set$churn_risk_score)

# LR Cross Validation (CV)
set.seed(111)
control <- trainControl(method="cv", number=10, verboseIter = TRUE)
lr_cv <- train(churn_risk_score ~., data=train_set, method="glm", 
                 family = 'binomial', trControl=control)

# LR Grid Search
set.seed(111)
trControl <- trainControl(method = 'cv',
                          number = 10,
                          search = 'grid', verboseIter = TRUE)

lr_grid <- train(x= train_set[,-37] , y= train_set$churn_risk_score, 
                  method = 'glmnet',
                  trControl = trControl,
                  family = 'binomial' )

lr_grid_test <- predict(lr_grid, type = 'prob', test_set[ ,-37])
lr_grid_test = ifelse(lr_grid_test > 0.5, 1, 0)

confusionMatrix(as.factor(lr_grid_test[,2]), 
                test_set$churn_risk_score)

# LR Random Search
set.seed(111)
trControl <- trainControl(method = 'cv',
                          number = 10,
                          search = 'random', verboseIter = TRUE)

lr_random <- train(x= train_set[,-37] , y= train_set$churn_risk_score, 
                 method = 'glmnet',
                 trControl = trControl,
                 family = 'binomial' )

# Receiver Operating Characteristic (ROC) Baseline ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Predict_ROC = predict(lr_base, test_set, type = 'response')

pred = prediction(Predict_ROC, test_set$churn_risk_score)
perf = performance(pred, 'tpr','fpr')
plot(perf, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))
abline(a = 0, b = 1)

# Area Under Curve (AUC)
auc = as.numeric(performance(pred, "auc")@y.values)
auc = round(auc, 3)
auc

# ROC Random ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Predict_ROC = predict(lr_random, test_set, type = 'prob')

pred = prediction(Predict_ROC[,2], test_set$churn_risk_score)
perf = performance(pred, 'tpr','fpr')
plot(perf, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))
abline(a = 0, b = 1)

# AUC Random
auc = as.numeric(performance(pred, "auc")@y.values)
auc = round(auc, 3)
auc


# ROC Grid ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Predict_ROC = predict(lr_grid, test_set, type = 'prob')

pred = prediction(Predict_ROC[,2], test_set$churn_risk_score)
perf = performance(pred, 'tpr','fpr')
plot(perf, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))
abline(a = 0, b = 1)

# AUC Grid
auc = as.numeric(performance(pred, "auc")@y.values)
auc = round(auc, 3)
auc


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# SVM
library(e1071)
set.seed(222)
svm_r <- svm(churn_risk_score ~.,
             data = train_set,
             kernel = 'radial',
             probability = TRUE)

summary(svm_r)

# Train set
svm_train = predict(svm_r, train_set)
confusionMatrix(svm_train, 
                train_set$churn_risk_score)

# Test set
svm_test = predict(svm_r, test_set)
confusionMatrix(svm_test, 
                test_set$churn_risk_score)


# Receiver Operating Characteristic (ROC) ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Predict_ROC = predict(svm_r, test_set, probability = TRUE)
prob <- attr(Predict_ROC, "probabilities")

pred = prediction(prob[,2], test_set$churn_risk_score)
perf = performance(pred, 'tpr','fpr')
plot(perf, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))
abline(a = 0, b = 1)

# Area Under Curve (AUC)
auc = as.numeric(performance(pred, "auc")@y.values)
auc = round(auc, 3)
auc


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Random Forest
library(randomForest)
set.seed(333)
rf_base <- randomForest(churn_risk_score ~.,
                        data = train_set, 
                        do.trace = TRUE)
print(rf)
attributes(rf)
rf$ntree
rf$importance
plot(rf)
varImpPlot(rf)

# Train set
rf_train <- predict(rf_base, train_set)
table(actual=train_set$churn_risk_score, 
      rf_train)
confusionMatrix(rf_train, 
                train_set$churn_risk_score)
# Test set
rf_test <- predict(rf_base, test_set)
table(actual=train_set$churn_risk_score, 
      rf_test)
confusionMatrix(rf_test, 
                test_set$churn_risk_score)

# RF with Cross Validation (CV)
set.seed(333)
control <- trainControl(method="cv", number=10, verboseIter = TRUE)
# mtry <- floor(sqrt(ncol(train_set)))
# tunegrid <- expand.grid(mtry=mtry)
rf_default <- train(churn_risk_score ~., data=train_set, method="rf", 
                    metric='Accuracy', trControl=control)
print(rf_default)

# RF Tuning (tuneRF)
set.seed(333)
bestmtry <- tuneRF(train_set[,-37], train_set$churn_risk_score, 
                   stepFactor=1.5, improve=1e-5, ntree = 500)
print(bestmtry)

# Tuned model
set.seed(333)
rf_tuned <- randomForest(churn_risk_score ~.,
                        data = train_set, mtry = 13, ntree = 500, 
                        do.trace = TRUE)

# Test set
rf_tuned_test <- predict(rf_tuned, test_set)
table(actual=test_set$churn_risk_score, 
      rf_tuned_test)
confusionMatrix(rf_tuned_test, 
                test_set$churn_risk_score)

# Receiver Operating Characteristic (ROC) Baseline ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Predict_ROC = predict(rf_base, test_set, type = 'prob')
Predict_ROC[,2] # prediction probability for positive (1) churn score

pred = prediction(Predict_ROC[,2], test_set$churn_risk_score)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))
abline(a = 0, b = 1)

# Area Under Curve (AUC) Baseline
auc = as.numeric(performance(pred, "auc")@y.values)
auc = round(auc, 3)
auc

# ROC Tuned
Predict_ROC = predict(rf_tuned, test_set, type = 'prob')
Predict_ROC[,2] # prediction probability for positive (1) churn score

pred = prediction(Predict_ROC[,2], test_set$churn_risk_score)
perf = performance(pred, 'tpr', 'fpr')
plot(perf, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))
abline(a = 0, b = 1)

# AUC Tuned
auc = as.numeric(performance(pred, "auc")@y.values)
auc = round(auc, 3)
auc


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# XGBoost

library(xgboost)

# predictors
x_train <- train_set[,-37]
x_test <- test_set[,-37]
glimpse(y_train)
# target
y_train <- train_set[,37]
y_test <- test_set[,37]

set.seed(444)
xgb <- xgboost(data = as.matrix(x_train),
               label = y_train,
               nround = 10)


# Train set
xgb_train <- predict(xgb, as.matrix(x_train))
xgb_train <- ifelse(xgb_train > 0.5, 1, 0)

confusionMatrix(as.factor(xgb_train), as.factor(y_train))

# Test set
xgb_test <- predict(xgb, as.matrix(x_test))
xgb_test <- ifelse(xgb_test > 0.5, 1, 0)

confusionMatrix(as.factor(xgb_test), as.factor(y_test))


# Hyperparameter tuning
set.seed(444)
xgb_tune <- xgboost(data = as.matrix(x_train),
                   label = y_train,
                   nround = 100,
                   max_depth=10)

# Test set
xgb_tune_test <- predict(xgb_tune, as.matrix(x_test))
xgb_tune_test <- ifelse(xgb_tune_test > 0.5, 1, 0)

confusionMatrix(as.factor(xgb_tune_test), as.factor(y_test))


# Receiver Operating Characteristic (ROC) Baseline ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Predict_ROC = predict(xgb, as.matrix(x_test), type = 'prob')

pred = prediction(Predict_ROC, y_test)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))
abline(a = 0, b = 1)

# Area Under Curve (AUC) Baseline
auc = as.numeric(performance(pred, "auc")@y.values)
auc = round(auc, 3)
auc

# ROC Tuned
Predict_ROC = predict(xgb_tune, as.matrix(x_test), type = 'prob')

pred = prediction(Predict_ROC, y_test)
perf = performance(pred, 'tpr', 'fpr')
plot(perf, colorize=T, 
     main = "ROC curve",
     ylab = "Sensitivity",
     xlab = "Specificity",
     print.cutoffs.at=seq(0,1,0.3),
     text.adj= c(-0.2,1.7))
abline(a = 0, b = 1)

# AUC Tuned
auc = as.numeric(performance(pred, "auc")@y.values)
auc = round(auc, 3)
auc

