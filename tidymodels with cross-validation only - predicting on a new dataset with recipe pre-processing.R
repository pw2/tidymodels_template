

### load packages
library(tidymodels)
library(tidyverse)


############ Random Forest With Model Tuning Example ############
### get data
df <- mtcars
head(df)
table(df$cyl)

df$cyl <- as.factor(df$cyl)

### cross validation folds
df_cv <- vfold_cv(df, v = 5)
df_cv

### specify random forest model
rf_spec_with_tuning <-rand_forest(mtry = tune(),
                                  trees = tune()) %>%
  set_engine("randomForest", importance = TRUE) %>%
  set_mode("classification")

### build a tuning grid
rf_tune_grid <- grid_regular(
  mtry(range = c(1, 10)),
  trees(range = c(500, 800)),
  levels = 5
)

rf_tune_grid

### recipe -- set up imputation for new data that might have missing values
cyl_rec <- recipe(cyl ~ mpg + disp + wt, data = df) %>%
  step_impute_knn(mpg, disp, wt,
                  neighbors = 3)

cyl_rec

### workflow
cyl_wf <- workflow() %>%
  add_recipe(cyl_rec) %>%
  add_model(rf_spec_with_tuning)

### set a control function to save the predictions from the model fit to the CV-folds
ctrl <- control_resamples(save_pred = TRUE)

### fit model
cyl_tune <- tune_grid(
  cyl_wf,
  resamples = df_cv,
  grid = rf_tune_grid,
  control = ctrl
)

cyl_tune

### view model metrics
collect_metrics(cyl_tune)

### get predictions
pred_group <- cyl_tune %>%
  unnest(cols = .predictions) %>%
  select(.pred_4, .pred_6, .pred_8, .pred_class, cyl) 

pred_group

table('predicted' = pred_group$.pred_class, 'observed' = pred_group$cyl)

## Fit the final model & extract the workflow
# get the optimized numbers for mtry and trees
m <- select_best(cyl_tune, "roc_auc") %>% pull(mtry)
t <- select_best(cyl_tune, "roc_auc") %>% pull(trees)

m
t

# re-specify the model with the optimized values
rf_spec_tuned <-rand_forest(mtry = m,
                            trees = t) %>%
  set_engine("randomForest", importance = TRUE) %>%
  set_mode("classification")

# re-set workflow
cyl_wf_tuned <- workflow() %>%
  add_recipe(cyl_rec) %>%
  add_model(rf_spec_tuned)

# extract the final recipe for pre-processing of new data
cyl_final_rec <- cyl_wf_tuned %>%
  fit(df) %>%
  extract_recipe()

cyl_final_rec

# extract final model
cyl_final_tuned <- cyl_wf_tuned  %>% 
  fit(df) %>%
  extract_fit_parsnip()

cyl_final_tuned

## Save recipe & model to use later
# save(cyl_final_rec, file = "cyl_final_rec.rda")
# save(cyl_final_tuned, file = "cyl_final_tuned.rda")
# load("cyl_final_rec.rda")
# load("cyl_final_tuned.rda")

########## Apply Recipe & Model to new Data ###############
### Create New Data with NAs
set.seed(947)
row_ids <- sample(x = 1:nrow(mtcars), size = 5, replace = TRUE)

df_new <-mtcars[row_ids, ]
df_new[2, 3] <- NA
df_new[c(1,5), 1] <- NA
df_new[3, 6] <- NA

# get the actual cyl values for this new data
truth <- df_new$cyl
truth

# drop the cyl column to pretend like this is new data
df_new$cyl <- NULL
df_new

### Apply the pre-processing recipe to the new data
df_new_pre_processed <- cyl_final_rec %>%
  bake(new_data = df_new)

df_new_pre_processed

### Make a prediction for cyl
pred_cyl <- predict(cyl_final_tuned, new_data = df_new_pre_processed, type = "class")
df_new_pre_processed <- cbind(df_new_pre_processed, pred_cyl)


### get probability of each class
pred_probs <- predict(cyl_final_tuned, new_data = df_new_pre_processed, type = "prob")
df_new_pre_processed <- cbind(df_new_pre_processed, pred_probs)
df_new_pre_processed
