# Read data with appropriate column names
data <- read.csv("data_multi.csv", header = TRUE)

# Load necessary libraries for data manipulation
library(tidyverse)
library(tidymodels)

# Impute missing values for numeric columns
data <- data %>%
  mutate(across(where(is.numeric), ~ifelse(is.na(.), mean(., na.rm = TRUE), .)))

# Convert 'gender' to numeric
#data$gender <- ifelse(data$gender == "male", 1, 0)

# Assign the modified dataset to the 'Boston' variable
Boston <- data

# Convert specific columns to factor
#for(i in c(2)){
  Boston[[i]] <- as.factor(Boston[[i]])


# Provide an overview of the modified data
skimr::skim(Boston)

# Split the data into training and testing sets
set.seed(4321)
datasplit <- initial_split(Boston, prop = 0.70, strata = label, breaks = 8)
traindata <- training(datasplit)
testdata <- testing(datasplit)

# Define and prepare the recipe for data pre-processing
datarecipe <- recipe(label ~ ., data = traindata) %>%
  step_dummy(all_nominal_predictors()) %>%
  prep()

# Apply the recipe to both training and testing sets
traindata2 <- bake(datarecipe, new_data = traindata)
testdata2 <- bake(datarecipe, new_data = testdata)

# Provide an overview of the pre-processed training and testing data
skimr::skim(traindata2)
skimr::skim(testdata2)

# Define the decision tree model and the workflow
model_dt <- decision_tree(cost_complexity = tune()) %>%
  set_engine("rpart") %>%
  set_mode("regression") %>%
  set_args(tree_depth = 30L, min_n = 2L)

wk_dt <- 
  workflow() %>%
  add_model(model_dt) %>%
  add_formula(label ~ .)

# Define cross-validation settings
set.seed(42)
folds <- vfold_cv(traindata2, v = 5)

# Define hyperparameter search settings
hpset_dt <- parameters(cost_complexity(range=c(-4, -1)))
hpgrid_dt <- grid_regular(hpset_dt, levels = 5)
hpgrid_dt
log10(hpgrid_dt$cost_complexity)
# Perform cross-validated grid search for hyperparameter tuning
set.seed(42)
tune_dt <- wk_dt %>%
  tune_grid(
    resamples = folds,
    grid = hpgrid_dt,
    metrics = metric_set(rmse, rsq, mae),
    control = control_grid(save_pred = TRUE, verbose = TRUE)
  )

 
 #####################################################
 
 # 贝叶斯优化超参数
 set.seed(42)
 tune_dt <- wk_dt %>%
   tune_bayes(
     resamples = folds,
     initial = 5,
     iter = 30,
     metrics = metric_set(yardstick::rmse, 
                          yardstick::rsq, 
                          yardstick::mae),
     control = control_bayes(save_pred = T, 
                             verbose = T, 
                             no_improve = 5)
   )
 
 #####################################################
 
 # 图示交叉验证结果
 autoplot(tune_dt)
 eval_tune_dt <- tune_dt %>%
   collect_metrics()
 eval_tune_dt
 
 # 经过交叉验证得到的最优超参数
 hpbest_dt <- tune_dt %>%
   select_by_one_std_err(metric = "rmse", desc(cost_complexity))
 hpbest_dt
 
 # 采用最优超参数组合训练最终模型
 final_dt <- wk_dt %>%
   finalize_workflow(hpbest_dt) %>%
   fit(traindata2)
 final_dt
 
 # 提取最终的算法模型
 final_dt2 <- final_dt %>%
   extract_fit_engine()
 library(rpart.plot)
 rpart.plot(final_dt2)
 final_dt2$variable.importance
 barplot(final_dt2$variable.importance, las = 2)
 
 #############################################################
 
 # 应用模型-预测训练集
 predtrain_dt <- final_dt %>%
   predict(new_data = traindata2) %>%
   bind_cols(traindata2 %>% select(label)) %>%
   mutate(dataset = "train")
 predtrain_dt
 predtrain_dt %>%
   ggplot(aes(x = label, y = .pred)) +
   geom_point() +
   geom_abline(intercept = 0, slope = 1, color = "red", linewidth = 1.2) +
   labs(x = "实际值", y = "预测值", 
        title = "决策树模型在训练集上的预测效果") +
   theme_bw()
 
 #############################################################
 
 # 应用模型-预测测试集
 predtest_dt <- final_dt %>%
   predict(new_data = testdata2) %>%
   bind_cols(testdata2 %>% select(label)) %>%
   mutate(dataset = "test") %>%
   mutate(model = "dt")
 predtest_dt
 predtest_dt %>%
   ggplot(aes(x = label, y = .pred)) +
   geom_point() +
   geom_abline(intercept = 0, slope = 1, color = "red", linewidth = 1.2) +
   labs(x = "实际值", y = "预测值", 
        title = "决策树模型在测试集上的预测效果") +
   theme_bw()
 
 #############################################################
 
 # 合并结果
 eval_dt <- predtrain_dt %>%
   bind_rows(predtest_dt) %>%
   group_by(dataset) %>%
   metrics(truth = label, estimate = .pred) %>%
   mutate(model = "dt")
 eval_dt
 
 #############################################################
 
 # 最优超参数的交叉验证指标平均结果
 eval_best_cv_dt <- eval_tune_dt %>%
   inner_join(hpbest_dt[, 1])
 eval_best_cv_dt
 
 # 最优超参数的交叉验证指标具体结果
 eval_best_cv5_dt <- tune_dt %>%
   collect_predictions() %>%
   inner_join(hpbest_dt[, 1]) %>%
   group_by(id) %>%
   metrics(truth = label, estimate = .pred) %>%
   mutate(model = "dt") %>%
   inner_join(eval_best_cv_dt[c(2,4,6)])
 eval_best_cv5_dt
 
 # 保存评估结果
 save(final_dt,
      predtest_dt,
      eval_dt,
      eval_best_cv5_dt, 
      file = ".\\model\\evalresult_dt.RData")
 
 # 最优超参数的交叉验证指标图示
 eval_best_cv5_dt %>%
   filter(.metric == "rmse") %>%
   ggplot(aes(x = id, y = .estimate, group = 1)) +
   geom_point() +
   geom_line() +
   scale_y_continuous(limits = c(0, 7)) +
   labs(x = "", y = "rmse") +
   theme_bw()
 
 
 # 最优超参数的交叉验证图示
 tune_dt %>%
   collect_predictions() %>%
   inner_join(hpbest_dt[, 1]) %>%
   ggplot(aes(x = label, y = .pred, color = id)) +
   geom_point(size = 1.5) +
   geom_smooth(method = lm, se = F) +
   geom_abline(intercept = 0, slope = 1, color = "black", linewidth = 1.2) +
   facet_wrap(~id) +
   theme_bw()
 
 #######################################################################
 
 
 # 自变量数据集
 colnames(traindata2)
 traindatax <- traindata2[,-10]
 colnames(traindatax)
 
 # iml包
 library(iml)
 predictor_model <- Predictor$new(
   final_dt, 
   data = traindatax,
   y = traindata2$label
 )
 
 
 # 变量重要性-基于置换
 set.seed(42)
 imp_model <- FeatureImp$new(predictor_model, loss = "rmse")
 # 数值
 imp_model$results
 # 图示
 imp_model$plot() +
   theme_bw()
 
 
 # 变量效应
 pdp_model <- FeatureEffect$new(
   predictor_model, 
   feature = "area_mean",
   method = "pdp"
 )
 # 数值
 pdp_model$results
 # 图示
 pdp_model$plot() +
   theme_bw()
 
 # 所有变量的效应全部输出
 effs_model <- FeatureEffects$new(predictor_model, method = "pdp")
 # 数值
 effs_model$results %>%
   bind_rows() %>%
   ggplot(aes(x = .borders, y = .value)) +
   geom_line() +
   facet_wrap(~.feature, scales = "free") +
   theme_bw()
 # 图示
 effs_model$plot()
 
 # 单样本shap分析
 shap_model <- Shapley$new(
   predictor_model, 
   x.interest = traindatax[2,]
 )
 # 数值
 shap_model$results
 # 图示
 shap_model$plot() +
   theme_bw()
 
 # 基于所有样本的shap分析
 # fastshap包
 library(fastshap)
 shap <- explain(
   final_dt, 
   X = as.data.frame(traindatax),
   nsim = 10,
   adjust = T,
   pred_wrapper = function(model, newdata) {
     predict(model, newdata) %>% pull(1)
   }
 )
 
 # 单样本图示
 force_plot(object = shap[2, ], 
            feature_values = as.data.frame(traindatax)[2, ], 
            baseline = mean(predtrain_dt$.pred),
            display = "viewer") 
 
 # 变量重要性
 autoplot(shap, fill = "skyblue") +
   theme_bw()
 
 data1 <- shap %>%
   as.data.frame() %>%
   dplyr::mutate(id = 1:n()) %>%
   pivot_longer(cols = -(ncol(traindatax)+1), values_to = "shap")
 shapimp <- data1 %>%
   dplyr::group_by(name) %>%
   dplyr::summarise(shap.abs.mean = mean(abs(shap))) %>%
   dplyr::arrange(shap.abs.mean) %>%
   dplyr::mutate(name = forcats::as_factor(name))
 data2 <- traindatax  %>%
   dplyr::mutate(id = 1:n()) %>%
   pivot_longer(cols = -(ncol(traindatax)+1))
 
 # 所有变量shap图示
 library(ggbeeswarm)
 data1 %>%
   left_join(data2) %>%
   dplyr::rename("feature" = "name") %>%
   dplyr::group_by(feature) %>%
   dplyr::mutate(
     value = (value - min(value)) / (max(value) - min(value)),
     feature = factor(feature, levels = levels(shapimp$name))
   ) %>%
   dplyr::arrange(value) %>%
   dplyr::ungroup() %>%
   ggplot(aes(x = shap, y = feature, color = value)) +
   geom_quasirandom(width = 0.2) +
   scale_color_gradient(
     low = "red", 
     high = "blue", 
     breaks = c(0, 1), 
     labels = c(" Low", "High "), 
     guide = guide_colorbar(barwidth = 1, 
                            barheight = 20,
                            ticks = F,
                            title.plabeltion = "right",
                            title.hjust = 0.5)
   ) +
   labs(x = "SHAP value", color = "Feature value") +
   theme_bw() +
   theme(legend.title = element_text(angle = -90))
 
 # 单变量shap图示
 data1 %>%
   left_join(data2) %>%
   dplyr::rename("feature" = "name") %>%
   dplyr::filter(feature == "rm") %>%
   ggplot(aes(x = value, y = shap)) +
   geom_point() +
   geom_smooth(se = F, span = 0.5) +
   labs(x = "rm") +
   theme_bw()
 
 
 
 
 