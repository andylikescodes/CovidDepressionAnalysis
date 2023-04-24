library(tidyverse)
library(lme4)
library(MuMIn)
library(lmerTest)

data <- read.csv('./output/v3_python/no_outlier.csv')

model1 <- lmer(BDI ~ Mandatory_SAH*week + Fear + NEO_Extraversion + NEO_Neuroticism + NEO_Conscientiousness + NEO_Agreeableness + NEO_Openness + Education + Income_v2 + Gender + Political_Views + Age + Race_W + Race_AA + Race_A + deaths_avg_per_100k + cases_avg_per_100k + (1 |ids), data=data)

summary(model1)

r.squaredGLMM(model1)


model2 <- lmer(BDI ~ Mandatory_SAH*week + Fear + NEO_Extraversion + NEO_Neuroticism + NEO_Conscientiousness + NEO_Agreeableness + NEO_Openness + Education + Income_v2 + Gender + Political_Views + Age + Race_W + Race_AA + Race_A + deaths_avg_per_100k + cases_avg_per_100k + (1 + week |ids), data=data)

summary(model2)

r.squaredGLMM(model2)


AIC(model1, model2)