
## KNN tidymodels tutorial
# Data Source: https://www.kaggle.com/datasets/stefanoleone992/fifa-21-complete-player-dataset

library(tidyverse)
library(tidymodels)

theme_set(theme_light())

fifa <- read_csv("players_21.csv")
fifa %>%
  head()

# drop goalies
fifa <- fifa %>%
  filter(!grepl(pattern = "GK", x = player_positions))

## get a smaller subset of the data
dat <- fifa %>%
  select(sofifa_id, short_name, long_name, age, height_cm, 
         weight_kg, club_name, league_name, overall:wage_eur, 
         team_position, pace:physic, attacking_crossing:mentality_composure, 
         defending_standing_tackle, defending_sliding_tackle)

## check for NAs
dat %>%
  summarize(across(.cols = everything(),
                   ~sum(is.na(.x)))) %>%
  pivot_longer(everything()) %>%
  filter(value > 0)

# nothing that will impact our analysis

## There are some players contract value of 9, despite having statistics
dat %>%
  filter(value_eur == 0) %>%
  nrow()

# put these 207 players into their own data set and then remove them from the rest of the data
dat_value0 <- dat %>%
  filter(value_eur == 0)

dat <- dat %>%
  filter(value_eur > 0)

# plot the log of the contract value
hist(log(dat$value_eur), 
     main = 'Log of Contrated Value',
     xlab = 'Contract Value')

# how many positions are there
barplot(table(dat$team_position), 
        horiz = TRUE, 
        las = 2,
        main = 'FIFA Positions')

# does position influence contract value?
dat %>%
  group_by(team_position) %>%
  summarize(avg_value = mean(value_eur),
            .groups = "drop") %>%
  ggplot(aes(x = avg_value, y = reorder(team_position, avg_value))) +
  geom_col(fill = "light grey",
           alpha = 0.8,
           color = "black") +
  scale_x_continuous(labels = scales::dollar_format()) +
  labs(x = "Contract Value",
       y = NULL,
       title = "Average Contract Value by Position")

##### tidymodels set up
# Aim is to use KNN to estimate the log of contract value
# We will use several of the FiFa rating metrics
# We will do this all in {tidymodels} so we can do all the preprocessing in one step

## Create train/test split
### train/test split
set.seed(1739)
d_split <- initial_split(dat)
d_split

train <- training(d_split)
test <- testing(d_split)

train %>% head()
test %>% head()

## Create Cross Validation Sets
set.seed(24)
df_cv <- vfold_cv(train, v = 5)
df_cv

## specify the KNN model
knn_spec <- nearest_neighbor(neighbors = tune())  %>%
  set_engine("kknn") %>%
  set_mode("regression")

## Create a recipe for all the pre-processing needs
# create the model
fifa_rec <- recipe(value_eur ~ .,
                   data = train %>%
                     select(value_eur, team_position, pace:defending_sliding_tackle)) %>%
  step_log(value_eur) %>%
  step_dummy(team_position) %>%
  step_normalize(pace:defending_sliding_tackle)

# Look at the steps that were performed in the recipe
fifa_rec %>%
  prep()

## See how the steps effected the train data
fifa_rec %>%
  prep() %>%
  bake(new_data = NULL)

## Create the model workflow
knn_wf <- workflow() %>%
  add_recipe(fifa_rec) %>%
  add_model(knn_spec)

knn_wf

## tune the model
# build a tuning grid
knn_tune_grid <- grid_regular(
  neighbors(range = c(2, 10))
)

ctrl <- control_resamples(save_pred = TRUE)

knn_tune <- tune_grid(
  knn_wf,
  resamples = df_cv,
  grid = knn_tune_grid,
  control = ctrl
)

knn_tune

### view model metrics
knn_tune %>%
  collect_metrics()

### which is the best model?
knn_tune %>%
  select_best(metric = "rmse")

### save the best neighbors to fit the final model
best_neighbors <- knn_tune %>%
  select_best(metric = "rmse") %>%
  pull(neighbors)

best_neighbors

## get the model metrics for the bets model
knn_tune %>%
  collect_metrics() %>%
  filter(neighbors == best_neighbors)

  ### Version 1 Model Fitting: Fit using the finalize_workflow() function
  knn_final <- knn_wf %>%
    finalize_workflow(
      select_best(x = knn_tune, metric = "rmse")
    )

knn_final

## get the model predictions on the test set
knn_fit <- knn_final %>%
  last_fit(
    d_split
  )

knn_fit

# plot the predictions against the actual values
knn_fit %>%
  unnest(.predictions) %>%
  ggplot(aes(x = .pred, y = value_eur)) +
  geom_jitter(color = "light blue",
              alpha = 0.5) +
  geom_abline(intercept = 0,
              slope = 1,
              color = "red",
              linetype = "dashed",
              linewidth = 1.2) +
  geom_smooth(color = "black") +
  labs(x = "predicted log of contract value",
       y = "log of contract value",
       title = "Test set predictions")


# Calculate test set rmse
knn_fit %>%
  unnest(.predictions) %>%
  summarize(rmse = sqrt(mean((value_eur - .pred)^2)))

### Version 2 Model Fitting - Re-specify the model with the best_neighbors to the train set
# This is so that we can save the model and use it later, downstream
knn_spec_final <- nearest_neighbor(neighbors = best_neighbors)  %>%
  set_engine("kknn") %>%
  set_mode("regression")

# reset the workflow with the tuned model
knn_wf_tuned <- workflow() %>%
  add_recipe(fifa_rec) %>%
  add_model(knn_spec_final)

knn_wf_tuned

# extract the final recipe, fit to the entire data set, for pre-processing of new data
knn_final_rec <- knn_wf_tuned %>%
  fit(dat) %>%
  extract_recipe()

knn_final_rec

# extract final model
knn_final_tuned <- knn_wf_tuned  %>% 
  fit(dat) %>%
  extract_fit_parsnip()

knn_final_tuned

## Save recipe & model to use later
# save(knn_final_rec, file = "knn_final_rec.rda")
# save(knn_final_tuned, file = "knn_final_tuned.rda")
# load("knn_final_rec.rda")
# load("knn_final_tuned.rda")

## Apply the saved workflow and model to the data where players had 0 value to estimate
# their value.

# There are many players in the 0 value data set without a team position
dat_value0 %>% 
  summarize(across(everything(), ~sum(is.na(.x)))) %>% 
  pivot_longer(everything()) %>% 
  filter(value > 0)

# let's drop those players
dat_value0 <- dat_value0 %>%
  filter(!is.na(team_position))

  ## Apply the pre-processing recipe to the new data
  df_new_pre_processed <- knn_final_rec %>%
    bake(new_data = dat_value0)

df_new_pre_processed

## Make a prediction for contract value
pred_value <- predict(knn_final_tuned, new_data = df_new_pre_processed)
pred_value %>%
  head()

## Add the predictions to the data
dat_value0_final <- cbind(dat_value0, pred_value)

# Let's see the predicted value for the 12 players
dat_value0_final %>%
  ggplot(aes(x = exp(.pred), y = reorder(short_name, .pred))) +
  geom_col() +
  scale_x_continuous(labels = scales::dollar_format()) +
  labs(x = "Predicted Contract Value",
       y = NULL,
       title = "Predicted Contract Value")
