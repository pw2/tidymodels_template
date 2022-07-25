

### Packages -------------------------------------------------------

library(tidyverse)
library(tidymodels)

### Data -------------------------------------------------------

dat <- mtcars
dat %>%
  head()

### Modelling -------------------------------------------------------
## Create 10 Cross Validation Folds

set.seed(1)
cv_folds <- vfold_cv(dat, v = 10)
cv_folds

## Specify the linear regression engine
## model specs
lm_spec <- linear_reg() %>%
  set_engine("lm") 


## Model formula
mpg_formula <- mpg ~ cyl + disp + wt + drat

## Set up workflow
lm_wf <- workflow() %>%
  add_formula(mpg_formula) %>%
  add_model(lm_spec) 

## Fit the model to the cross validation folds
lm_fit <- lm_wf %>%
  fit_resamples(
    resamples = cv_folds,
    control = control_resamples(extract = extract_model, save_pred = TRUE)
  )

## Get model coefficients for each fold
# Model coefficients are in a nested list for each fold in column .extracts
# We can see the first three folds as a list

lm_fit$.extracts %>% 
  .[1:3]

# Let's unnest this and get the coefficients out
model_coefs <- lm_fit %>% 
  select(id, .extracts) %>%                    # get the id and .extracts columns
  unnest(cols = .extracts) %>%                 # unnest .extracts, which produces the model in a list
  mutate(coefs = map(.extracts, tidy)) %>%     # use map() to apply the tidy function and get the coefficients in their own column
  unnest(coefs)                                # unnest the coefs column you just made to get the coefficients for each fold

model_coefs

## Plot the model coefficients and 2*SE across all folds
model_coefs %>%
  filter(term != "(Intercept)") %>%
  select(id, term, estimate, std.error) %>%
  group_by(term) %>%
  mutate(avg_estimate = mean(estimate)) %>%
  ggplot(aes(x = id, y = estimate)) +
  geom_hline(aes(yintercept = avg_estimate),
             size = 1.2,
             linetype = "dashed") +
  geom_point(size = 4) +
  geom_errorbar(aes(ymin = estimate - 2*std.error, ymax = estimate + 2*std.error),
                width = 0.1,
                size = 1.2) +
  facet_wrap(~term, scales = "free_y") +
  labs(x = "CV Folds",
       y = "Estimate ± 95% CI",
       title = "Regression Coefficients ± 95% CI for 10-fold CV",
       subtitle = "Dashed Line = Average Coefficient Estimate over 10 CV Folds per Independent Variable") +
  theme_classic() +
  theme(strip.background = element_rect(fill = "black"),
        strip.text = element_text(face = "bold", size = 12, color = "white"),
        axis.title = element_text(size = 14, face = "bold"),
        axis.text.x = element_text(angle = 60, hjust = 1, face = "bold", size = 12),
        axis.text.y = element_text(face = "bold", size = 12),
        plot.title = element_text(size = 18),
        plot.subtitle = element_text(size = 16))

