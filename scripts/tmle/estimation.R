# Set environment
setwd("/home/andy/CovidDepressionAnalysis")

library(tidyverse)
names(data)

library(tlverse)
library(sl3)
library(tmle3)
library(Hmisc)
# create the task (i.e., use washb_data to predict outcome using covariates)

train_tmle <- function(data){
  

    adjustments <- c("Race_A", 
                    "Race_AA", 
                    "Race_W", 
                    "Gender",
                    "Political_Views", 
                    "Age", 
                    "Education", 
                    "cases_avg", 
                    "deaths_avg", 
                    "cases_avg_per_100k", 
                    "deaths_avg_per_100k",
                    "slope_new_cases", 
                    "slope_new_deaths", 
                    "lat", 
                    'lng', 
                    "population")
    treatment <- "Mandatory_SAH"
    outcome <- "BDI"

    node_list <- list(
    W = adjustments,
    A = treatment,
    Y = outcome
    )

    ate_spec <- tmle_ATE(
    treatment_level = "True",
    control_level = "False"
    )

    # choose base learners
    lrnr_mean <- make_learner(Lrnr_mean)
    lrnr_rf <- make_learner(Lrnr_ranger)

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

    # define metalearners appropriate to data types
    ls_metalearner <- make_learner(Lrnr_nnls)
    mn_metalearner <- make_learner(Lrnr_nnls)

    sl_Y <- Lrnr_sl$new(
    learners = list(lrnr_mean, lrnr_rf, lrn_ridge, lrn_lasso, lrn_polspline, lrn_earth),
    metalearner = ls_metalearner
    )
    sl_A <- Lrnr_sl$new(
    learners = list(lrnr_mean, lrnr_rf, lrn_xgb, lrn_gam, lrn_bayesglm),
    metalearner = mn_metalearner
    )
    learner_list <- list(A = sl_A, Y = sl_Y)

    processed <- process_missing(data, node_list)
    data <- processed$data
    node_list <- processed$node_list

    tmle_fit <- tmle3(ate_spec, data, node_list, learner_list)
    return(tmle_fit)
}


run_tmle_for_extro_intro <- function(){
    all_est_intro = c()
    all_lower_intro = c()
    all_upper_intro = c()

    all_est_extro = c()
    all_lower_extro = c()
    all_upper_extro = c()

    for (i in 1:16){
        path = paste0('output/unraked_jasp/wave', toString(i), '.csv')
        data = read.csv(path)

        extro = data %>% filter(NEO_Extraversion >= mean(NEO_Extraversion))
        # intro = data %>% filter(NEO_Extraversion >= mean(NEO_Extraversion))

        tmle_fit = train_tmle(extro)
        estimate = tmle_fit$summary$tmle_est
        upper = tmle_fit$summary$upper
        lower = tmle_fit$summary$lower
        all_est_extro = c(all_est_extro, estimate)
        all_lower_extro = c(all_lower_extro, lower)
        all_upper_extro = c(all_upper_extro, upper)


        # tmle_fit = train_tmle(intro)
        # estimate = tmle_fit$summary$tmle_est
        # upper = tmle_fit$summary$upper
        # lower = tmle_fit$summary$lower
        # all_est_intro = c(all_est_intro, estimate)
        # all_lower_intro = c(all_lower_intro, lower)
        # all_upper_intro = c(all_upper_intro, upper)

    }

    errbar(1:16,all_est_extro,all_upper_extro,all_lower_extro,type='b')
    # errbar(1:16,all_est_intro,all_upper_intro,all_lower_intro,type='b')
    
}

run_tmle_for_extro_intro()

