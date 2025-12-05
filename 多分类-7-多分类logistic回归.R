# 模型机器---R语言tidymodels包机器学习分类与回归模型---多分类---多分类logistic回归

# https://www.tidymodels.org/find/parsnip/
# https://parsnip.tidymodels.org/reference/multinom_reg.html
# https://parsnip.tidymodels.org/reference/details_multinom_reg_nnet.html

# 模型评估指标
# https://cran.r-project.org/web/packages/yardstick/vignettes/metric-types.html

library(tidymodels)

# 读取数据
winelabel <- readr::read_csv(file.choose()) # tibble
colnames(winelabel)

# 修正变量类型
# 将分类变量转换为factor
#for(i in c(12)){
  winelabel[[i]] <- factor(winelabel[[i]])
}
# 变量类型修正后数据概况
skimr::skim(winelabel)

###############################################################
winelabel <- winelabel %>%
  mutate(across(where(is.numeric), ~ifelse(is.na(.), mean(., na.rm = TRUE), .)))


winelabel$label<-as.factor(winelabel$label)
# 数据拆分
set.seed(4321)
datasplit <- 
  initial_split(winelabel, prop = 0.7, strata = label)
traindata <- training(datasplit)
testdata <- testing(datasplit)

############################################################

# 数据预处理
# 先对照训练集写配方
datarecipe <- recipe(label ~ ., traindata) %>%
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

###############################################################

# 训练模型
# 设定模型
model_multinom <- multinom_reg(
  mode = "classification",
  engine = "nnet",
  penalty = 0
)
model_multinom


# 拟合模型
fit_multinom <- model_multinom %>%
  fit(label ~ ., traindata2)
fit_multinom$fit
summary(fit_multinom$fit)

# 系数输出
fit_multinom %>%
  tidy()

##########################################################

# 应用模型-预测训练集
predtrain_multinom <- fit_multinom %>%
  predict(new_data = traindata2, type = "prob") %>%
  bind_cols(fit_multinom %>%
              predict(new_data = traindata2, type = "class")) %>%
  bind_cols(traindata2 %>% select(label)) %>%
  mutate(dataset = "train")
predtrain_multinom

roc_cols <- paste(".pred_", levels(predtrain_dt$label), sep = "")
# 评估模型ROC曲线-训练集上
roctrain_multinom <- predtrain_multinom %>%
  roc_curve(label, roc_cols) %>%
  mutate(dataset = "train")
roctrain_multinom
autoplot(roctrain_multinom)

# 混淆矩阵
cmtrain_multinom <- predtrain_multinom %>%
  conf_mat(truth = label, estimate = .pred_class)
cmtrain_multinom
autoplot(cmtrain_multinom, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
eval_train_multinom <- cmtrain_multinom %>%
  summary() %>%
  bind_rows(predtrain_multinom %>%
              roc_auc(label, roc_cols)) %>%
  mutate(dataset = "train")
eval_train_multinom

####################################

# 应用模型-预测测试集
predtest_multinom <- fit_multinom %>%
  predict(new_data = testdata2, type = "prob") %>%
  bind_cols(fit_multinom %>%
              predict(new_data = testdata2, type = "class")) %>%
  bind_cols(testdata2 %>% select(label)) %>%
  mutate(dataset = "test") %>%
  mutate(model = "multinom")
predtest_multinom
# 评估模型ROC曲线-测试集上
roctest_multinom <- predtest_multinom %>%
  roc_curve(label, roc_cols) %>%
  mutate(dataset = "test")
autoplot(roctest_multinom)


# 混淆矩阵
cmtest_multinom <- predtest_multinom %>%
  conf_mat(truth = label, estimate = .pred_class)
cmtest_multinom
autoplot(cmtest_multinom, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))
# 合并指标
eval_test_multinom <- cmtest_multinom %>%
  summary() %>%
  bind_rows(predtest_multinom %>%
              roc_auc(label, roc_cols)) %>%
  mutate(dataset = "test")
eval_test_multinom

####################################

# 合并训练集和测试集上ROC曲线
roctrain_multinom %>%
  bind_rows(roctest_multinom) %>%
  mutate(dataset = factor(dataset, levels = c("train", "test"))) %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = dataset)) +
  geom_path(linewidth = 1) +
  facet_wrap(~.level) +
  theme_bw()

# 合并训练集和测试集上性能指标
eval_multinom <- eval_train_multinom %>%
  bind_rows(eval_test_multinom) %>%
  mutate(model = "multinom")
eval_multinom

#################################################################
#################################################################



# 设定5折交叉验证
set.seed(42)
folds <- vfold_cv(traindata2, v = 5)
folds

# workflow
wf_multinom <- 
  workflow() %>%
  add_model(model_multinom) %>%
  add_formula(label ~ .)
wf_multinom

# 交叉验证
set.seed(42)
cv_multinom <- 
  wf_multinom %>%
  fit_resamples(folds,
                metrics = metric_set(yardstick::accuracy, 
                                     yardstick::roc_auc,
                                     yardstick::pr_auc),
                control = control_resamples(save_pred = T))
cv_multinom

# 交叉验证指标平均结果
eval_cv_multinom <- collect_metrics(cv_multinom)
eval_cv_multinom


# 交叉验证指标具体结果
eval_cv5_multinom <- collect_predictions(cv_multinom) %>%
  group_by(id) %>%
  roc_auc(label, roc_cols) %>%
  ungroup() %>%
  mutate(model = "multinom") %>%
  left_join(eval_cv_multinom[c(1, 3, 5)])
eval_cv5_multinom


# 保存评估结果
save(fit_multinom,
     predtest_multinom,
     eval_multinom,
     eval_cv5_multinom, 
     file = ".\\model\\evalresult_multinom.RData")

# 交叉验证指标图示
eval_cv5_multinom %>%
  filter(.metric == "roc_auc") %>%
  ggplot(aes(x = id, y = .estimate, group = 1)) +
  geom_point() +
  geom_line() +
  scale_y_continuous(limits = c(0, 1)) +
  labs(x = "", y = "roc_auc") +
  theme_bw()

# 交叉验证图示
collect_predictions(cv_multinom) %>%
  group_by(id) %>%
  roc_curve(label, roc_cols) %>%
  ungroup() %>%
  ggplot(aes(x = 1-specificity, y = sensitivity, color = id)) +
  geom_path(linewidth = 1) +
  facet_wrap(~.level) +
  theme_bw()


###################################################################

# 自变量数据集
colnames(traindata2)
traindatax <- traindata2[,-1]
colnames(traindatax)

# iml包
library(iml)

# 变量重要性-基于置换
predictor_model <- Predictor$new(
  fit_multinom, 
  data = traindatax,
  y = traindata2$label
)
imp_model <- FeatureImp$new(
  predictor_model, 
  loss = "ce"
)
# 数值
imp_model$results
# 图示
imp_model$plot() +
  theme_bw()


# 变量效应
predictor_model <- Predictor$new(
  fit_multinom, 
  data = traindatax,
  y = traindata2$label,
  predict.function = function(model, newdata){
    predict(model, newdata, type = "prob") %>% pull(1)
  }
)
pdp_model <- FeatureEffect$new(
  predictor_model, 
  feature = "pH",
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
effs_model$results
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
  fit_multinom, 
  X = as.data.frame(traindatax),
  nsim = 10,
  adjust = T,
  pred_wrapper = function(model, newdata) {
    predict(model, newdata, type = "prob") %>% pull(1)
  }
)

# 单样本图示
force_plot(object = shap[1L, ], 
           feature_values = as.data.frame(traindatax)[1L, ], 
           baseline = mean(predtrain_multinom$.pred_H),
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
                           title.position = "right",
                           title.hjust = 0.5)
  ) +
  labs(x = "SHAP value", color = "Feature value") +
  theme_bw() +
  theme(legend.title = element_text(angle = -90))


# 单变量shap图示
data1 %>%
  left_join(data2) %>%
  dplyr::rename("feature" = "name") %>%
  dplyr::filter(feature == "alcohol") %>%
  ggplot(aes(x = value, y = shap)) +
  geom_point() +
  geom_smooth(se = F, span = 0.5) +
  labs(x = "alcohol") +
  theme_bw()







