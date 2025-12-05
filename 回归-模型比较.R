# 模型机器---R语言tidymodels包机器学习分类与回归模型---回归---模型比较

library(tidymodels)

# 加载各个模型的评估结果
evalfiles <- list.files(".\\model\\", full.names = T)
lapply(evalfiles, load, .GlobalEnv)

# 各个模型在测试集上的误差指标
eval <- bind_rows(
  eval_lm, eval_mlp,eval_lightgbm, eval_knn,eval_enet,
  eval_dt, eval_rf, eval_xgboost, eval_lsvm, 
  
) %>%
 # filter(dataset == "train") %>%
  pivot_wider(names_from = .metric, values_from = .estimate)
eval
# 各个模型在测试集上的误差指标图示
library(RColorBrewer)

mypalette <- colorRampPalette(brewer.pal(8,"Set2"))


eval %>%
  ggplot(aes(x = model, y = rsq, fill = model)) +
  scale_fill_manual(values = mypalette(10))+
  geom_col(width = 0.3, show.legend = F) +
  geom_text(aes(label = round(rsq, 2)), 
            nudge_y = 0) +
  theme_bw()


eval %>%
  ggplot(aes(x = model, y = rmse, fill = dataset)) +
  scale_fill_manual(values = mypalette(2)) +  # 使用适当数量的颜色
  geom_col(position = position_dodge(width = 0.8), width = 0.7) +
  geom_text(aes(label = round(rmse, 2)), 
            position = position_dodge(width = 0.8), 
            vjust = -0.5) +
  coord_polar() +
  theme_bw() +
  labs(x = "Model", y = "rmse", fill = "Dataset") +
  theme_minimal() +  # 使用简洁主题
  theme(
    axis.text.y = element_blank(),  # 隐藏 y 轴标签
    axis.ticks.y = element_blank(),  # 隐藏 y 轴刻度
    #panel.grid = element_blank(),  # 隐藏网格
    panel.border = element_blank(),  # 隐藏边界框
    axis.text.x = element_text(angle = 45, hjust = 1)  # 旋转 x 轴标签
  )
# 各个模型在测试集上
predtest <- bind_rows(
  predtest_lm, predtest_mlp,predtest_lightgbm, predtest_knn,predtest_enet,
   predtest_dt,predtest_rf, predtest_xgboost, predtest_lsvm
 
)
predtest

# 散点图
library(ggplot2)
library(dplyr)

# 计算 Spearman 的 R 值和 P 值
spearman_stats <- predtest %>%
  group_by(model) %>%
  summarize(
    spearman_r = cor(label, .pred, method = "spearman"),
    p_value = cor.test(label, .pred, method = "spearman")$p.value
  )
spearman_stats$p_value<-round(spearman_stats$p_value,3)

# 将统计结果与原数据合并
predtest <- predtest %>%
  left_join(spearman_stats, by = "model")

# 绘图
ggplot(predtest, aes(x = label, y = .pred, color = model)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "Actual Value", y = "Predicted Value") +
  facet_wrap(~model) +
  scale_color_manual(values = mypalette(length(unique(predtest$model)))) +
  theme_bw() +
  # 添加注释
  geom_text(data = spearman_stats, aes(x = Inf, y = Inf, label = paste0("Spearman R: ", round(spearman_r, 2), "\nP-value: ", format.pval(p_value))), hjust = 1.1, vjust = 1.1, size = 3, color = "black")



# ---- Bland-Altman Plot ----
# 计算均值和差值
bland_altman <- predtest %>%
  mutate(
    mean = (label + .pred) / 2,
    diff = .pred - label
  )

# 计算每个模型的均值、上下95% limits of agreement
ba_stats <- bland_altman %>%
  group_by(model) %>%
  summarize(
    mean_diff = mean(diff),
    sd_diff = sd(diff),
    loa_upper = mean_diff + 1.96 * sd_diff,
    loa_lower = mean_diff - 1.96 * sd_diff
  )

# 合并统计值
bland_altman <- left_join(bland_altman, ba_stats, by = "model")

# 绘制Bland-Altman图
p2 <- ggplot(bland_altman, aes(x = mean, y = diff, color = model)) +
  geom_point(alpha = 0.6) +
  geom_hline(aes(yintercept = mean_diff), linetype = "dashed") +
  geom_hline(aes(yintercept = loa_upper), linetype = "dotted", color = "red") +
  geom_hline(aes(yintercept = loa_lower), linetype = "dotted", color = "blue") +
  facet_wrap(~model) +
  labs(x = "Mean of Actual and Predicted", y = "Difference (Predicted - Actual)", 
       title = "Bland-Altman Plot") +
  theme_bw() +
  scale_color_manual(values = mypalette(length(unique(predtest$model)))) +
  geom_text(data = ba_stats, 
            aes(x = Inf, y = Inf, 
                label = paste0("Mean diff: ", round(mean_diff,2), 
                               "\nLoA: [", round(loa_lower,2), ", ", round(loa_upper,2), "]")),
            hjust = 1.1, vjust = 1.1, size = 3, color = "black", inherit.aes = FALSE)




###predtrain

predtrain_lm$model<-"lm";predtrain_mlp$model<-"mlp";predtrain_lightgbm$model<-"lightgbm";
predtrain_knn$model<-"knn";predtrain_enet$model<-"enet";predtrain_dt$model<-"dt";
predtrain_rf$model<-"rf";predtrain_xgboost$model<-"xgboost";predtrain_lsvm$model<-"lsvm";
predtrain <- bind_rows(
  predtrain_lm, predtrain_mlp,predtrain_lightgbm, predtrain_knn,predtrain_enet,
  predtrain_dt,predtrain_rf, predtrain_xgboost, predtrain_lsvm
  
)



predtrain


# 计算 Spearman 的 R 值和 P 值
spearman_stats <- predtrain %>%
  group_by(model) %>%
  summarize(
    spearman_r = cor(label, .pred, method = "spearman"),
    p_value = cor.test(label, .pred, method = "spearman")$p.value
  )
spearman_stats$p_value<-round(spearman_stats$p_value,3)

# 将统计结果与原数据合并
predtrain <- predtrain %>%
  left_join(spearman_stats, by = "model")

# 绘图
ggplot(predtrain, aes(x = label, y = .pred, color = model)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  labs(x = "Actual Value", y = "Predicted Value") +
  facet_wrap(~model) +
  scale_color_manual(values = mypalette(length(unique(predtrain$model)))) +
  theme_bw() +
  # 添加注释
  geom_text(data = spearman_stats, aes(x = Inf, y = Inf, label = paste0("Spearman R: ", round(spearman_r, 2), "\nP-value: ", format.pval(p_value))), hjust = 1.1, vjust = 1.1, size = 3, color = "black")







library(RColorBrewer)
library(ggplot2)
library(ggpubr)
mypalette <- colorRampPalette(brewer.pal(8,"Set2"))
# 残差箱线图
predtest %>%
  mutate(error = label - .pred) %>%
  ggplot(aes(x = model, y = error, fill = model)) +
  scale_fill_manual(values = mypalette(10))+
  #geom_violin(alpha = 0.4, position = position_dodge(width = .75), 
  #            size = 0.8, color="black") +
  geom_boxplot(notch = TRUE, outlier.size = -1, 
               color="black", lwd=0.8, alpha = 0.7) +
  geom_point(shape = 21, size=1.5, # 点的性状和大小
             position = position_jitterdodge(), # 让点散开
             color="black", alpha = 1) +
  stat_compare_means(method = "kruskal.test", label.y = -2.5)+
  theme_bw()


# 各个模型交叉验证的各折指标点线图
evalcv <- bind_rows(
  eval_cv5_lm, eval_best_cv5_mlp, eval_best_cv5_lightgbm, eval_best_cv5_knn,eval_best_cv5_enet,
  eval_best_cv5_dt,eval_best_cv5_rf, eval_best_cv5_xgboost, eval_best_cv5_lsvm
)
evalcv

evalcv %>% 
  filter(.metric == "rmse") %>%
  ggplot(aes(x = id, y = .estimate, 
             group = model, color = model)) +
  scale_fill_manual(values = mypalette(8))+
  geom_point() +
  geom_line() +
  scale_y_continuous(limits = c(100, 250)) +
  labs(x = "", y = "rmse") +
  theme_bw()

# 各个模型交叉验证的指标平均值图(带上下限)
evalcv %>%
  filter(.metric == "rmse") %>%
  group_by(model) %>%
  sample_n(size = 1) %>%
  ungroup() %>%
  ggplot(aes(x = model, y = mean,color = model)) +
  geom_point(size = 2) +
  # geom_line(group = 1) +
  geom_errorbar(aes(ymin = mean-std_err, 
                    ymax = mean+std_err,),
                width = 0.1, linewidth = 1) +
  scale_y_continuous(limits = c(100, 350)) +
  labs(y = "cv rmse") +
  theme_bw()


# 各个模型交叉验证的指标平均值图(带上下限)
evalcv %>%
  filter(.metric == "rsq") %>% 
  group_by(model) %>%
  sample_n(size = 1) %>%
  ungroup() %>%
  ggplot(aes(x = model, y = mean,color = model)) +
  geom_point(size = 2) +
  # geom_line(group = 1) +
  geom_errorbar(aes(ymin = mean-std_err, 
                    ymax = mean+std_err,),
                width = 0.1, linewidth = 1) +
  scale_y_continuous(limits = c(0.05, 0.5)) +
  labs(y = "cv rsq") +
  theme_bw()

