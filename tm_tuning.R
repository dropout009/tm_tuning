library(tidymodels)
library(tune)

set.seed(42)

df_split = initial_split(diamonds,  p = 0.8)

df_train = training(df_split)
df_test  = testing(df_split)


# 分割

df_cv = vfold_cv(df_train, v = 5)


# 前処理レシピ
rec = recipe(price ~ ., data = df_train) %>% 
  step_log(price) %>% 
  step_ordinalscore(all_nominal())


# モデル
model = rand_forest(mode = "regression",
                    trees = 50, # 速度重視
                    min_n = tune(),
                    mtry = tune()) %>%
  set_engine("ranger", num.threads = parallel::detectCores(), seed = 42)


# ハイパーパラメータ

df_input = rec %>% 
  prep() %>% 
  juice() %>% 
  select(-price)


params = list(min_n(),
              mtry() %>% finalize(rec %>% prep() %>% juice() %>% select(-price))) %>% 
  parameters()


df_grid = params %>% 
  grid_random(size = 10)


# チューニング

df_tuned = tune_grid(object = rec,
                     model = model,
                     resamples = df_cv,
                     grid = df_grid,
                     metrics = metric_set(rmse, mae, rsq),
                     control = control_grid(verbose = T))


df_tuned %>% 
  collect_metrics()


df_tuned %>% 
  show_best(metric = "rmse", n_top = 3, maximize = FALSE)

df_best_param = df_tuned %>% 
  select_best(metric = "rmse", maximize = FALSE)


model_best = update(model, df_best_param)