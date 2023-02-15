# Set environment
setwd("/home/ubuntu/projects/CovidDepressionAnalysis")

library(tidyverse)
data <- read.csv("output/IP_weighted_sources/w6.csv")
names(data)

library(tlverse)
library(sl3)
# create the task (i.e., use washb_data to predict outcome using covariates)

adjustments <- c("cvd_cases_7d_avg", "cvd_deaths_7d_avg", "Gender", "Age", "Education", "Race_AA", "Race_W", "Political_Views")
#square_terms = ["cvd_cases_7d_avg", "cvd_deaths_7d_avg", "Age", "Education", "Political_Views"]
treatment <- "Mandatory_SAH"
outcome <- "Depression_adj"

task <- make_sl3_Task(
  data = data,
  outcome = outcome,
  covariates = adjustments
)

# Initiate learners:
lrn_glm <- Lrnr_glm$new()
lrn_mean <- Lrnr_mean$new()
# penalized regressions:
lrn_ridge <- Lrnr_glmnet$new(alpha = 0)
lrn_lasso <- Lrnr_glmnet$new(alpha = 1)

# spline regressions:
lrn_polspline <- Lrnr_polspline$new()
lrn_earth <- Lrnr_earth$new()

# fast highly adaptive lasso (HAL) implementation
lrn_hal <- Lrnr_hal9001$new(max_degree = 2, num_knots = c(3,2), nfolds = 5)

# tree-based methods
lrn_ranger <- Lrnr_ranger$new()
lrn_xgb <- Lrnr_xgboost$new()

lrn_gam <- Lrnr_gam$new()
lrn_bayesglm <- Lrnr_bayesglm$new()

stack <- Stack$new(
  lrn_glm, lrn_mean, lrn_ridge, lrn_lasso, lrn_polspline, lrn_earth, lrn_hal, 
  lrn_ranger, lrn_xgb, lrn_gam, lrn_bayesglm
)


# Run learners
sl <- Lrnr_sl$new(learners = stack, metalearner = Lrnr_nnls$new())

start_time <- proc.time() # start time

set.seed(4197)
sl_fit <- sl$train(task = task)

runtime_sl_fit <- proc.time() - start_time # end time - start time = run time
runtime_sl_fit


sl_preds <- sl_fit$predict(task = task)

# library(data.table)
# library(dplyr)
# library(tmle3)
# library(sl3)
# washb_data <- fread(
#   paste0(
#     "https://raw.githubusercontent.com/tlverse/tlverse-data/master/",
#     "wash-benefits/washb_data.csv"
#   ),
#   stringsAsFactors = TRUE
# )


node_list <- list(
  W = adjustments,
  A = treatment,
  Y = outcome
)


ate_spec <- tmle_ATE(
  treatment_level = "1",
  control_level = "0"
)

# choose base learners
lrnr_mean <- make_learner(Lrnr_mean)
lrnr_rf <- make_learner(Lrnr_ranger)

# define metalearners appropriate to data types
ls_metalearner <- make_learner(Lrnr_nnls)
mn_metalearner <- make_learner(Lrnr_nnls)
sl_Y <- Lrnr_sl$new(
  learners = list(lrnr_mean, lrnr_rf),
  metalearner = ls_metalearner
)
sl_A <- Lrnr_sl$new(
  learners = list(lrnr_mean, lrnr_rf),
  metalearner = mn_metalearner
)
learner_list <- list(A = sl_A, Y = sl_Y)

processed <- process_missing(data, node_list)
data <- processed$data
node_list <- processed$node_list

tmle_fit <- tmle3(ate_spec, data, node_list, learner_list)
