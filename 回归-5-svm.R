# 模型机器---R语言tidymodels包机器学习分类与回归模型---回归---支持向量机

# https://www.tidymodels.org/find/parsnip/
# https://parsnip.tidymodels.org/reference/svm_linear.html
# https://parsnip.tidymodels.org/reference/details_svm_linear_kernlab.html

# 模型评估指标
# https://cran.r-project.org/web/packages/yardstick/vignettes/metric-types.html

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

#Boston$diameter_max<-as.numeric(Boston$diameter_max)
# 修正变量类型
# 将分类变量转换为factor
#for(i in c(4)){
  Boston[[i]] <- factor(Boston[[i]])
}
# 变量类型修正后数据概况
skimr::skim(Boston)

#############################################################

# 数据拆分
set.seed(4321)
datasplit <- 
  initial_split(Boston, prop = 0.7, strata = label, breaks = 8)
traindata <- training(datasplit)
testdata <- testing(datasplit)

#############################################################

# 数据预处理
# 先对照训练集写配方
# recipes
datarecipe <- recipe(label ~ ., traindata) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_center(all_predictors()) %>%
  step_scale(all_predictors()) %>%
  prep()
datarecipe

# 按方处理训练集和测试集
traindata2 <- bake(datarecipe, new_data = NULL) %>%
  dplyr::select(label, everything())
testdata2 <- bake(datarecipe, new_data = testdata) %>%
  dplyr::select(label, everything())

# 数据预处理后数据概况
skimr::skim(traindata2)
skimr::skim(testdata2)

#############################################################

# 训练模型
# 设定模型

model_lsvm <- svm_linear(
  mode = "regression",
  engine = "kernlab",
  cost = tune(),
  margin = tune()
)
model_lsvm

# workflow
wk_lsvm <- 
  workflow() %>%
  add_model(model_lsvm) %>%
  add_formula(label ~ .)
wk_lsvm

# 重抽样设定-5折交叉验证
set.seed(42)
folds <- vfold_cv(traindata2, v = 5)
folds

# 超参数寻优范围
hpset_lsvm <- parameters(cost(), svm_margin())
hpgrid_lsvm <- grid_regular(hpset_lsvm, levels = 2)
hpgrid_lsvm


# 交叉验证网格搜索过程
tune_lsvm <- wk_lsvm %>%
  tune_grid(resamples = folds,
            grid = hpgrid_lsvm,
            metrics = metric_set(yardstick::rmse, 
                                 yardstick::rsq, 
                                 yardstick::mae),
            control = control_grid(save_pred = T, verbose = T))

# 图示交叉验证结果
autoplot(tune_lsvm)
eval_tune_lsvm <- tune_lsvm %>%
  collect_metrics()
eval_tune_lsvm


# 经过交叉验证得到的最优超参数
hpbest_lsvm <- tune_lsvm %>%
  select_best(metric = "rmse")
hpbest_lsvm

# 采用最优超参数组合训练最终模型
final_lsvm <- wk_lsvm %>%
  finalize_workflow(hpbest_lsvm) %>%
  fit(traindata2)
final_lsvm

# 提取最终的算法模型
final_lsvm %>%
  extract_fit_engine()

#############################################################

# 应用模型-预测训练集
predtrain_lsvm <- final_lsvm %>%
  predict(new_data = traindata2) %>%
  bind_cols(traindata2$label) %>%
  mutate(dataset = "train")

predtrain_lsvm
colnames(predtrain_lsvm)[2]<-"label"


predtrain_lsvm %>%
  ggplot(aes(x = label, y = .pred)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red", linewidth = 1.2) +
  labs(x = "实际值", y = "预测值", 
       title = "SVM模型在训练集上的预测效果") +
  theme_bw()

#############################################################

# 应用模型-预测测试集
predtest_lsvm <- final_lsvm %>%
  predict(new_data = testdata2) %>%
  bind_cols(testdata2$label) %>%
  mutate(dataset = "test") %>%
  mutate(model = "lsvm")
predtest_lsvm
colnames(predtest_lsvm)[2]<-"label"
predtest_lsvm %>%
  ggplot(aes(x = label, y = .pred)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = "red", linewidth = 1.2) +
  labs(x = "实际值", y = "预测值", 
       title = "SVM模型在测试集上的预测效果") +
  theme_bw()

#############################################################

# 合并结果
eval_lsvm <- predtrain_lsvm %>%
  bind_rows(predtest_lsvm) %>%
  group_by(dataset) %>%
  metrics(truth = label, estimate = .pred) %>%
  mutate(model = "lsvm")
eval_lsvm

#############################################################

# 最优超参数的交叉验证指标平均结果
eval_best_cv_lsvm <- eval_tune_lsvm %>%
  inner_join(hpbest_lsvm[, 1:2])
eval_best_cv_lsvm

# 最优超参数的交叉验证指标具体结果
eval_best_cv5_lsvm <- tune_lsvm %>%
  collect_predictions() %>%
  inner_join(hpbest_lsvm[, 1:2]) %>%
  group_by(id) %>%
  metrics(truth = label, estimate = .pred) %>%
  mutate(model = "lsvm") %>%
  inner_join(eval_best_cv_lsvm[c(3,5,7)])
eval_best_cv5_lsvm

# 保存评估结果
save(final_lsvm,
     predtest_lsvm,
     eval_lsvm,
     eval_best_cv5_lsvm, 
     file = ".\\model\\evalresult_lsvm.RData")

# 最优超参数的交叉验证指标图示
eval_best_cv5_lsvm %>%
  filter(.metric == "rmse") %>%
  ggplot(aes(x = id, y = .estimate, group = 1)) +
  geom_point() +
  geom_line() +
  scale_y_continuous(limits = c(0, 7)) +
  labs(x = "", y = "rmse") +
  theme_bw()


# 最优超参数的交叉验证图示
tune_lsvm %>%
  collect_predictions() %>%
  inner_join(hpbest_lsvm[, 1:2]) %>%
  ggplot(aes(x = label, y = .pred, color = id)) +
  geom_point(size = 1.5) +
  geom_smooth(method = lm, se = F) +
  geom_abline(intercept = 0, slope = 1, color = "black", linewidth = 1.2) +
  facet_wrap(~id) +
  theme_bw()

#####################################################################

# 自变量数据集
colnames(traindata2)
traindatax <- traindata2#[,-1]
colnames(traindatax)

# iml包
library(iml)
predictor_model <- Predictor$new(
  final_lsvm, 
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
  feature = "lstat",
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
  x.interest = traindatax[1,]
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
  final_lsvm, 
  X = as.data.frame(traindatax),
  nsim = 10,
  adjust = T,
  pred_wrapper = function(model, newdata) {
    predict(model, newdata) %>% pull(1)
  }
)

# 单样本图示
force_plot(object = shap[1L, ], 
           feature_values = as.data.frame(traindatax)[1L, ], 
           baseline = mean(predtrain_lsvm$.pred), 
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











###########################################################

# 高斯核SVM
# https://www.tidymodels.org/find/parsnip/
# https://parsnip.tidymodels.org/reference/svm_rbf.html
# https://parsnip.tidymodels.org/reference/details_svm_rbf_kernlab.html

model_rsvm <- svm_rbf(
  mode = "regression",
  engine = "kernlab",
  cost = tune(),
  rbf_sigma = tune(),
  margin = tune()
)
model_rsvm
hpset_rsvm <- parameters(cost(), rbf_sigma(), svm_margin())
hpgrid_rsvm <- grid_regular(hpset_rsvm, levels = c(2,3,2))
hpgrid_rsvm


# 多项式核SVM
# https://www.tidymodels.org/find/parsnip/
# https://parsnip.tidymodels.org/reference/svm_poly.html
# https://parsnip.tidymodels.org/reference/details_svm_poly_kernlab.html
model_psvm <- svm_poly(
  mode = "regression",
  engine = "kernlab",
  cost = tune(),
  degree = tune(),
  scale_factor = tune(),
  margin = tune()
)
model_psvm
hpset_psvm <- parameters(cost(), degree(), scale_factor(), svm_margin())
hpgrid_psvm <- grid_regular(hpset_psvm, levels = c(2,3,4,5))
hpgrid_psvm



