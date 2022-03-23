##############################################################################################
####### Confidence Intervals for Random Forest Regression using {tidymodels} (sort of) #######
## Load packages
library(tidymodels)
library(tidyverse)
library(randomForest)

## load data
df <- mtcars %>%
  select(mpg, disp, hp, wt, qsec, drat, gear, carb)

head(df)

## Split data into cross-validation sets
set.seed(5)
df_cv <- vfold_cv(df, v = 5)
df_cv

## specify the random forest regression model
rf_spec <- rand_forest(mtry = tune(), trees = tune()) %>%
  set_engine("randomForest") %>%
  set_mode("regression")

## build a tuning grid
rf_tune_grid <- grid_regular(
  mtry(range = c(1, 7)),
  trees(range = c(500, 800)),
  levels = 5
)

## Model recipe
rf_rec <- recipe(mpg ~ ., data = df)

## workflow
rf_workflow <- workflow() %>%
  add_recipe(rf_rec) %>%
  add_model(rf_spec)

## set a control function to save the predictions from the model fit to the CV-folds
ctrl <- control_resamples(save_pred = TRUE)

## fit model
rf_tune <- tune_grid(
  rf_workflow,
  resamples = df_cv,
  grid = rf_tune_grid,
  control = ctrl
)

rf_tune

## view model metrics
collect_metrics(rf_tune)

## Which is the best model?
select_best(rf_tune, "rmse")

## Look at that models performance
collect_metrics(rf_tune) %>%
  filter(mtry == 4, trees == 725)

## Extract the best mtry and trees values to optimize rmse
m <- select_best(rf_tune, "rmse") %>% pull(mtry)
t <- select_best(rf_tune, "rmse") %>% pull(trees)

m
t

## Re-fit the model outside of tidymodels with the optimized values
rf_refit <- randomForest(mpg ~ ., data = df, mtry = m, ntree = t)
rf_refit

## New data
set.seed(859)
row_id <- sample(1:nrow(df), size = 5, replace = TRUE)
newdat <- df[row_id, ]
newdat

## Make Predictions
pred.rf <- predict(rf_refit, newdat, predict.all = TRUE)
pred.rf

## Look at all 725 predictions for the first row of the data
head(pred.rf$individual[1, ])
tail(pred.rf$individual[1, ])

# Average prediction -- what the prediction function returns
mean(pred.rf$individual[1, ])

# SD of predictions
sd(pred.rf$individual[1, ])

# get t-critical value for df = 725 - 1
t_crit <- qt(p = 0.975, df = t - 1)

# 95% CI
mean(pred.rf$individual[1, ]) - t_crit * sd(pred.rf$individual[1, ])
mean(pred.rf$individual[1, ]) + t_crit * sd(pred.rf$individual[1, ])

## Now for all of the predictions
newdat$pred_mpg <- predict(rf_refit, newdat)

## add confidence intervals
lower <- rep(NA, nrow(newdat))
upper <- rep(NA, nrow(newdat))

for(i in 1:nrow(newdat)){
  lower[i] <- mean(pred.rf$individual[i, ]) - t_crit * sd(pred.rf$individual[i, ])
  upper[i] <- mean(pred.rf$individual[i, ]) + t_crit * sd(pred.rf$individual[i, ])
}

newdat$lwr <- lower
newdat$upr <- upper

## Look at the new observations, predctions and confidence intervals and plot the data
## new data
newdat

## plot
newdat %>%
  mutate(car_type = rownames(.)) %>%
  ggplot(aes(x = pred_mpg, y = reorder(car_type, pred_mpg))) +
  geom_point(size = 5) +
  geom_errorbar(aes(xmin = lwr, xmax = upr),
                width = 0.1,
                size = 1.3) +
  geom_point(aes(x = mpg),
             size = 5,
             color = "red") +
  theme_minimal() +
  theme(axis.text = element_text(size = 11, face = "bold")) +
  labs(x = "Predicted vs Actual MPG",
       y = NULL,
       title = "Predicted vs Actual (red) MPFG from Random Forest",
       subtitle = "mpg ~ disp + hp + wt + qsec + draft + gear + carb")

