# Set environment
setwd("/home/ubuntu/projects/CovidDepressionAnalysis")

library(tidyverse)


library(tlverse)
library(sl3)
library(tmle3)
# create the task (i.e., use washb_data to predict outcome using covariates)

sl3_list_properties()
sl3_list_learners(c("binomial"))

run_tmle <- function(data){
  adjustments <- c("cvd_cases_7d_avg", "cvd_deaths_7d_avg", "Gender", "Age", "Education", "Race_AA", "Race_W", "Political_Views")
  #square_terms = ["cvd_cases_7d_avg", "cvd_deaths_7d_avg", "Age", "Education", "Political_Views"]
  treatment <- "Mandatory_SAH"
  outcome <- "Depression_adj"

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


  ## learners for Y

  stack_Y <- Stack$new(
    lrn_glm, lrn_mean, lrn_ridge, lrn_lasso, lrn_xgb, lrn_gam, lrn_bayesglm
  )

  # define metalearners appropriate to data types
  ls_metalearner <- make_learner(Lrnr_nnls)
  mn_metalearner <- make_learner(Lrnr_nnls)

  sl_Y <- Lrnr_sl$new(learners = stack_Y, metalearner = ls_metalearner)


  ### Learners for A
  stack_A <- Stack$new(
    lrn_glm, lrn_mean, lrn_ridge, lrn_lasso, lrn_xgb, lrn_gam, lrn_bayesglm
  )
  sl_A <- Lrnr_sl$new(learners = stack_A, metalearner = mn_metalearner)


  # sl_Y <- Lrnr_sl$new(
  #   learners = list(lrnr_mean, lrnr_rf),
  #   metalearner = ls_metalearner
  # )

  learner_list <- list(A = sl_A, Y = sl_Y)

  processed <- process_missing(data, node_list)
  data <- processed$data
  node_list <- processed$node_list

  tmle_fit <- tmle3(ate_spec, data, node_list, learner_list)
  print(tmle_fit)
}

print("Wave 6")
data <- read.csv("output/IP_weighted_sources/w6.csv")
run_tmle(data)
print("Wave 8")
data <- read.csv("output/IP_weighted_sources/w8.csv")
run_tmle(data)
print("Wave 10")
data <- read.csv("output/IP_weighted_sources/w10.csv")
run_tmle(data)
print("Wave 12")
data <- read.csv("output/IP_weighted_sources/w12.csv")
run_tmle(data)
print("Wave 14")
data <- read.csv("output/IP_weighted_sources/w14.csv")
run_tmle(data)
print("Wave 16")
data <- read.csv("output/IP_weighted_sources/w16.csv")
run_tmle(data)

