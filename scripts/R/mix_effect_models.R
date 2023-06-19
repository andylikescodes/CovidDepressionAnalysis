install.packages(c(
  "remotes",             # to be able to install from github
  "tidyverse",           # dplyr and ggplot
  "panelr",              # panel data made easier; wbm() function
  "lmtest",              # robust SEs
  "broom.mixed",         # extracting info from mixed models
  "ggeffects",           # calculating effects
  "huxtable",            # making regression tables
  "patchwork",           # organizing multiple plots
  "here",                # for setting sensible file paths in projects
  "skimr",               # for panelr summary statistics
  "plm",                 # for two-way fixed effects
  "clubSandwich",        # additional robust SEs
  "jtools",              # for theme_nice()
  "modelsummary",        # summarizing models (mostly plotting coefficients)
  "car",                 # for linearHypothesis (testing joint significance)
  "optimx",              # additional (g)lmer optimizers
  "dfoptim"              # additional (g)lmer optimizers
))