machine_selected<-read.csv("machine_selected.csv",header = T)

machine_selected$Size1<-as.factor(machine_selected$Size1)
machine_selected$Size2<-as.factor(machine_selected$Size2)
machine_selected$ideal_size<-as.factor(machine_selected$ideal_size)
machine_selected$ICL_size<-as.factor(machine_selected$ICL_size)


cmtest_machine <- machine_selected %>%
  conf_mat(truth = ideal_size, estimate =ICL_size )

cmtest_machine
autoplot(cmtest_machine, type = "heatmap") +
  scale_fill_gradient(low = "white", high = "skyblue") +
  theme(text = element_text(size = 15))


##metrix


# 安装并加载必要的包
# 安装并加载必要的包
install.packages("dplyr")
install.packages("caret")
install.packages("ggplot2")
install.packages("reshape2")
library(dplyr)
library(caret)
library(ggplot2)
library(reshape2)

# 读入数据
machine_selected <- read.csv("machine_selected.csv")

# 查看数据结构
str(machine_selected)

# 定义计算准确性、敏感性、特异性及精确性的函数(宏平均)
compute_metrics <- function(true_values, predicted_values) {
  # 创建混淆矩阵
  conf_matrix <- confusionMatrix(as.factor(predicted_values), as.factor(true_values))
  
  # 提取指标
  accuracy <- conf_matrix$overall["Accuracy"]
  
  # 获取所有类别的敏感性、特异性和精确性
  sensitivity <- mean(conf_matrix$byClass[,"Sensitivity"], na.rm = TRUE)
  specificity <- mean(conf_matrix$byClass[,"Specificity"], na.rm = TRUE)
  precision <- mean(conf_matrix$byClass[,"Pos Pred Value"], na.rm = TRUE)
  
  return(c(Accuracy = accuracy, Sensitivity = sensitivity, Specificity = specificity, Precision = precision))
}

# 初始化结果数据框
results_df <- data.frame(Predictor = character(), Accuracy = numeric(), Sensitivity = numeric(), Specificity = numeric(), Precision = numeric(), stringsAsFactors = FALSE)

# 预测列名
predict_columns <- c("Size1", "Size2", "ICL_size", ".pred_class")

# 批量计算每个预测列的指标
for (col in predict_columns) {
  metrics <- compute_metrics(machine_selected$ideal_size, machine_selected[[col]])
  results_df <- rbind(results_df, data.frame(Predictor = col, t(metrics)))
}

# 打印结果数据框
print(results_df)

# 将结果数据框转换为长格式
results_melt <- melt(results_df, id.vars = "Predictor")

# 绘制热图
ggplot(results_melt, aes(x = Predictor, y = variable, fill = value)) +
  geom_tile(color = "white") +
  geom_text(aes(label = round(value, 3)), color = "black", size = 3) +
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", midpoint = 0.7, limit = c(0.4, 1), name = "Value") +
  theme_minimal() +
  theme(
   # axis.text.y = element_blank(),  # 隐藏 y 轴标签
    axis.ticks.y = element_blank(),  # 隐藏 y 轴刻度
    #panel.grid = element_blank(),  # 隐藏网格
    panel.border = element_blank(),  # 隐藏边界框
    axis.text.x = element_text(angle = 45, hjust = 1)  # 旋转 x 轴标签
  )+
  labs(title = "Heatmap of Predictive Metrics",
       x = "Predictor",
       y = "Metric")
