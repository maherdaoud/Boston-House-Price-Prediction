######## Introduction ##########
# Non-linear model to predict the the housing prices
# the housing prices of a town or a suburb based on the features of the locality provided to us
# Maher Daoud
# To predict movie ratings in the validation set, train a machine learning algorithm using the inputs from one subset.
# Each record in the database describes a Boston suburb or town. 
# The data was drawn from the Boston Standard Metropolitan Statistical Area (SMSA) in 1970

###### Load Required Libraries #######
# Function to check if packages are installed 
isInstalled <- function(mypkg){
  is.element(mypkg, installed.packages()[,1])
}

if(!isInstalled("readr")) install.packages("readr",repos = "http://cran.us.r-project.org")
if(!isInstalled("dplyr")) install.packages("dplyr",repos = "http://cran.us.r-project.org")
if(!isInstalled("tidyr")) install.packages("readr",repos = "http://cran.us.r-project.org")
if(!isInstalled("caret")) install.packages("caret",repos = "http://cran.us.r-project.org")
if(!isInstalled("rsample")) install.packages("rsample",repos = "http://cran.us.r-project.org")
if(!isInstalled("randomForest")) install.packages("randomForest",repos = "http://cran.us.r-project.org")
if(!isInstalled("ranger")) install.packages("ranger",repos = "http://cran.us.r-project.org")
if(!isInstalled("h2o")) install.packages("h2o",repos = "http://cran.us.r-project.org")
library (readr)
library(dplyr)
library(tidyr)
library(rsample)      # data splitting 
library(randomForest) # basic implementation
library(ranger)       # a faster implementation of randomForest
library(caret)        # an aggregator package for performing many machine learning models
library(h2o)          # an extremely fast java-based platform

###### Import the Data #######

# In case you face any problem with downloading the dataset
# please use this link to download the dataset
# https://github.com/maherdaoud/Boston-House-Price-Prediction/blob/main/Boston.csv
# then use the following code to read it
# df <- read.csv("C:\download-path\Boston.csv)

# Import the data from Github
urlfile <-
  "https://raw.githubusercontent.com/maherdaoud/Boston-House-Price-Prediction/main/Boston.csv"
df <- data.frame(read_csv(url(urlfile), show_col_types = F), stringsAsFactors = F)

###### Transformation #######
# Calculate the log of MEDV
df$MEDV_log <- log(df$MEDV)

# Drop MEDV variable
# We will use the log version 
df$MEDV <- NULL

# As we can see from the report the data is very clean and we don't need more transformation
# Let's Implement the random forest and apply some techniques for tuning

###### Modeling #######
# Create training (70%) and test (30%) sets for the AmesHousing::make_ames() data.
# Use set.seed for reproducibility
set.seed(123)
ames_split <- initial_split(df, prop = .7)
ames_train <- training(ames_split)
ames_test  <- testing(ames_split)

# for reproduciblity
set.seed(123)

# default RF model
m1 <- randomForest(
  formula = MEDV_log ~ .,
  data    = ames_train
)
# Show Results of the Model
m1
# Plot the Model
plot(m1)

# The plotted error rate above is based on the OOB sample error
# and can be accessed directly at m1$mse. Thus, we can find which number of trees
#providing the lowest error rate, which is 
# 402 trees providing an average home sales price error of 0.1469607

# number of trees with lowest MSE
which.min(m1$mse)

# RMSE of this optimal random forest
sqrt(m1$mse[which.min(m1$mse)])

# randomForest also allows us to use a validation set to measure predictive accuracy
# if we did not want to use the OOB samples. 
# Here we split our training set further to create a training and validation set.
# We then supply the validation data in the xtest and ytest arguments.

# create training and validation data 
set.seed(123)
valid_split <- initial_split(ames_train, .8)

# training data
ames_train_v2 <- analysis(valid_split)

# validation data
ames_valid <- assessment(valid_split)
x_test <- ames_valid[setdiff(names(ames_valid), "MEDV_log")]
y_test <- ames_valid$MEDV_log

rf_oob_comp <- randomForest(
  formula = MEDV_log ~ .,
  data    = ames_train_v2,
  xtest   = x_test,
  ytest   = y_test
)

# extract OOB & validation errors
oob <- sqrt(rf_oob_comp$mse)
validation <- sqrt(rf_oob_comp$test$mse)

# compare error rates
tibble::tibble(
  `Out of Bag Error` = oob,
  `Test error` = validation,
  ntrees = 1:rf_oob_comp$ntree
) %>%
  gather(Metric, RMSE, -ntrees) %>%
  ggplot(aes(ntrees, RMSE, color = Metric)) +
  geom_line() +
  xlab("Number of trees") + 
  theme_bw()


##### Basic Tuning #####
## Initial tuning with randomForest ##
# If we are interested with just starting out and tuning the mtry parameter 
# we can use randomForest::tuneRF for a quick and easy tuning assessment. 
# tuneRf will start at a value of mtry that you supply and increase by a certain step factor 
# until the OOB error stops improving be a specified amount. 
# For example, the below starts with mtry = 5 and increases by a factor of 1.5 until 
# the OOB error stops improving by 1%.

# names of features
features <- setdiff(names(ames_train), "MEDV_log")

set.seed(123)

m2 <- tuneRF(
  x          = ames_train[features],
  y          = ames_train$MEDV_log,
  ntreeTry   = 500,
  mtryStart  = 5,
  stepFactor = 1.5,
  improve    = 0.01,
  trace      = FALSE      # to not show real-time progress 
)

##### Full grid search with ranger #####
# To perform the grid search, first we want to construct our grid of hyperparameters. 
# We’re going to search across 96 different models with varying mtry, minimum node size, and sample size.
# hyperparameter grid search
hyper_grid <- expand.grid(
  mtry       = seq(2, 13, by = 2),
  node_size  = seq(3, 9, by = 2),
  sampe_size = c(.55, .632, .70, .80),
  OOB_RMSE   = 0
)

# total number of combinations
nrow(hyper_grid)

# We loop through each hyperparameter combination and apply 500 trees since our previous examples
# illustrated that 500 was plenty to achieve a stable error rate. 
# Also note that we set the random number generator seed. 
# This allows us to consistently sample the same observations for each sample size 
# and make it more clear the impact that each change makes

for(i in 1:nrow(hyper_grid)) {
  # train model
  model <- ranger(
    formula         = MEDV_log ~ ., 
    data            = ames_train, 
    num.trees       = 500,
    mtry            = hyper_grid$mtry[i],
    min.node.size   = hyper_grid$node_size[i],
    sample.fraction = hyper_grid$sampe_size[i],
    seed            = 123
  )
  
  # add OOB error to grid
  hyper_grid$OOB_RMSE[i] <- sqrt(model$prediction.error)
}

hyper_grid %>% 
  dplyr::arrange(OOB_RMSE) %>%
  head(10)

# as we can see the best model until now come with mtry = 6, node_size=3, sampe_size=0.8

OOB_RMSE <- vector(mode = "numeric", length = 100)

for(i in seq_along(OOB_RMSE)) {
  
  optimal_ranger <- ranger(
    formula         = MEDV_log ~ ., 
    data            = ames_train, 
    num.trees       = 500,
    mtry            = 6,
    min.node.size   = 3,
    sample.fraction = .8,
    importance      = 'impurity'
  )
  
  OOB_RMSE[i] <- sqrt(optimal_ranger$prediction.error)
}

hist(OOB_RMSE, breaks = 20)

# Furthermore, you may have noticed we set importance = 'impurity' in the above modeling,
# which allows us to assess variable importance.
# Variable importance is measured by recording the decrease in MSE each time a variable is used
# as a node split in a tree

vi <- data.frame(imp = optimal_ranger$variable.importance)
# Transform the data frame and re-order it
vi <- vi %>% 
  tibble::rownames_to_column() %>% 
  dplyr::rename("variable" = rowname) %>% 
  dplyr::arrange(imp) %>%
  dplyr::mutate(variable = forcats::fct_inorder(variable))
# Plot the importance
ggplot2::ggplot(vi) +
  geom_col(aes(x = variable, y = imp, fill=variable),
           show.legend = F) +
  coord_flip() +
  theme_bw()


##### Full grid search with H2O #####
# h2o is a powerful and efficient java-based interface that provides parallel distributed algorithms. 
# Moreover, h2o allows for different optimal search paths in our grid search. 
# This allows us to be more efficient in tuning our models. 
# Here, I demonstrate how to tune a random forest model with h2o. 
# Lets go ahead and start up h2o

# start up h2o 
h2o.no_progress()
h2o.init(max_mem_size = "5g")

# First, we can try a comprehensive (full cartesian) grid search, 
# which means we will examine every combination of hyperparameter settings that we specify in hyper_grid.h2o


# create feature names
y <- "MEDV_log"
x <- setdiff(names(ames_train), y)

# turn training set into h2o object
train.h2o <- as.h2o(ames_train)

# hyperparameter grid
hyper_grid.h2o <- list(
  ntrees      = seq(200, 500, by = 150),
  mtries      = seq(2, 13, by = 2),
  max_depth   = seq(20, 40, by = 5),
  min_rows    = seq(1, 5, by = 2),
  nbins       = seq(10, 30, by = 5),
  sample_rate = c(.55, .632, .75)
)

# random grid search criteria
search_criteria <- list(
  strategy = "RandomDiscrete",
  stopping_metric = "mse",
  stopping_tolerance = 0.005,
  stopping_rounds = 10,
  max_runtime_secs = 10*60
)

# build grid search
# Maximum process time will be less than 10 minutes
# So please be patient :)
random_grid <- h2o.grid(
  algorithm = "randomForest",
  grid_id = "rf_grid2",
  x = x, 
  y = y, 
  training_frame = train.h2o,
  hyper_params = hyper_grid.h2o,
  search_criteria = search_criteria
)

# collect the results and sort by our model performance metric of choice
grid_perf2 <- h2o.getGrid(
  grid_id = "rf_grid2", 
  sort_by = "mse", 
  decreasing = FALSE
)
print(grid_perf2)

# Once we’ve identified the best model we can get that model 
# and apply it to our hold-out test set to compute our final test error.
# We see that we’ve been able to reduce our RMSE to near 0.1378412

# Grab the model_id for the top model, chosen by validation error
best_model_id <- grid_perf2@model_ids[[1]]
best_model <- h2o.getModel(best_model_id)

# Now let’s evaluate the model performance on a test set
ames_test.h2o <- as.h2o(ames_test)
best_model_perf <- h2o.performance(model = best_model, newdata = ames_test.h2o)

# RMSE of best model
h2o.mse(best_model_perf) %>% sqrt()
# Note that the RMSE for testing dataset is very close with the RMSE of the best model
# which means our modeling is performing good and we don't have over-fitting issue

##### Predicting #####
# for predicting a new data sets we will use the model that generated by H2o
# just for testing purpose we will use ames_test.h2o as a new datset
pred_h2o <- predict(best_model, ames_test.h2o)
# Then we need to find the exponential for the predicted value, it will represents the 
#  Median value of owner-occupied homes in 1000 dollars.
pred_h2o <- exp(pred_h2o)
head(pred_h2o)







