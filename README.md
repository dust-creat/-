# 回归模型
本次作业采用随机森林回归模型、XGBoost回归模型、多层感知机神经网络（MLP）
## 数据预处理
### 缺失值的处理
对于特征（自变量）中的缺失值，可以用当前特征的中位数或众数对缺失值进行填补

对于因变量中的缺失值，只能去除这组数据，避免自行添加数据以产生幻觉
### 自变量与因变量的处理
对于上述回归模型，均需要采用对自变量进行归一化处理，如z-score归一化；

random forest和XGBoost模型的因变量可以不进行归一化处理，但是对于MLP模型的因变量，必须采用归一化处理，[0,1]，如采用min-max归一化
## 模型回归处理
对于Random Forest和XGBoost，均采用单目标回归，对于多目标的情况时，采用循环遍历，并使用矩阵记录7种模型的各自参数

对于多层感知机神经网络（MLP），可以采用单目标回归和多目标回归模型，此处的多目标模型即为用所有的特征，同时预测7个目标变量。

# 后续处理中，发现无论是多目标的多层感知机还是单目标的多层感知机，回归效果太差了



## 单目标MLP

```R
# ==========================================
# 7. 模型3：单目标多层感知机神经网络 (MLP)
# ==========================================
library(neuralnet)

cat("开始训练多层感知机(MLP)模型...\n")

mlp_models <- list()
mlp_preds <- matrix(NA, nrow = nrow(clean_test), ncol = 7)
colnames(mlp_preds) <- response_cols

set.seed(42)

for (i in seq_along(response_cols)) {
  target <- response_cols[i]
  cat("正在训练 MLP -", target, "...\n")
  
  # 提取当前目标变量的训练数据
  y_train <- clean_train[[target]]
  
  # 关键步骤：对因变量 Y 进行 Min-Max 归一化 (缩放到 0~1 之间，极大地帮助神经网络收敛)
  min_y <- min(y_train)
  max_y <- max(y_train)
  y_train_scaled <- (y_train - min_y) / (max_y - min_y)
  
  # 合并当前特征和归一化后的标签
  train_nn_data <- cbind(train_x_scaled, Target = y_train_scaled)
  
  # 构建公式
  x_names_str <- paste(names(train_x_scaled), collapse = " + ")
  nn_formula  <- as.formula(paste("Target ~", x_names_str))
  
  # 训练模型 (单目标训练，速度会快很多)
  # 适当降低 stepmax，如果还是不收敛，说明特征可能不足以解释该标签
  mlp_models[[target]] <- neuralnet(
    formula = nn_formula,
    data = train_nn_data,
    hidden = c(8, 4),        # 因为变成单目标了，可以适当减少神经元以防过拟合和加快速度
    linear.output = TRUE,
    stepmax = 1e5            # 降到10万步，防止无限卡死
  )
  
  # 对测试集进行预测
  preds_scaled <- compute(mlp_models[[target]], test_x_scaled)$net.result
  
  # 关键步骤：将预测结果反向转换为原始尺度！
  mlp_preds[, i] <- preds_scaled * (max_y - min_y) + min_y
}

mlp_preds_df <- as.data.frame(mlp_preds)
cat("MLP 模型训练及预测完成！\n")
```

## 多目标MLP

```R
library(neuralnet)

cat("开始训练多目标 MLP 模型...\n")

# 1. 提取原始因变量数据
y_train_raw <- clean_train[, response_cols]

# 2. 关键步骤：对 7 个因变量 (Y) 分别进行 Min-Max 归一化 (0~1区间)
# 记录每种藻类的最大值和最小值，用于后续的反向还原
min_y <- apply(y_train_raw, 2, min)
max_y <- apply(y_train_raw, 2, max)

# 缩放处理
y_train_scaled <- scale(y_train_raw, center = min_y, scale = max_y - min_y)
y_train_scaled <- as.data.frame(y_train_scaled)

# 3. 合并标准化后的特征 (X) 和 归一化后的标签 (Y)
train_multi_nn_data <- cbind(train_x_scaled, y_train_scaled)

# 4. 动态构建多目标公式: 有害藻类1 + ... + 有害藻类7 ~ 特征1 + ...
x_names_str <- paste(names(train_x_scaled), collapse = " + ")
y_names_str <- paste(response_cols, collapse = " + ")
multi_formula_str <- paste(y_names_str, "~", x_names_str)
multi_nn_formula  <- as.formula(multi_formula_str)

set.seed(42)

# 5. 训练多目标神经网络
# 由于 Y 已经归一化，模型寻找多目标最优解的难度大幅降低
multi_mlp_model <- neuralnet(
  formula = multi_nn_formula,
  data = train_multi_nn_data,
  hidden = c(16, 8),      # 保持你的双隐藏层设计
  linear.output = TRUE,
  stepmax = 5e5           # 50万步足够收敛
)

# 6. 对测试集进行预测
multi_mlp_preds_scaled <- compute(multi_mlp_model, test_x_scaled)$net.result

# 7. 关键步骤：将预测出的 0~1 之间的结果，反向转换为原始的藻类含量尺度
multi_mlp_preds <- matrix(NA, nrow = nrow(test_x_scaled), ncol = length(response_cols))
colnames(multi_mlp_preds) <- response_cols

for (i in seq_along(response_cols)) {
  multi_mlp_preds[, i] <- multi_mlp_preds_scaled[, i] * (max_y[i] - min_y[i]) + min_y[i]
}

multi_mlp_preds_df <- as.data.frame(multi_mlp_preds)
cat("修正版多目标 MLP 模型训练及预测完成！\n")
```

## 输出结果分析（剔除其他结果）

```R
开始进行结果验证与对比分析...

========== 各模型对各藻类的详细预测效果 ==========
               Model        Target   RMSE    MAE        R2
15 Single-target MLP 有害藻类1含量 26.989 19.383  -0.73546
16 Single-target MLP 有害藻类2含量 18.193 12.126  -2.09405
17 Single-target MLP 有害藻类3含量 11.836  7.404  -3.57528
18 Single-target MLP 有害藻类4含量  6.020  3.844  -3.73402
19 Single-target MLP 有害藻类5含量 14.318 10.151  -1.24938
20 Single-target MLP 有害藻类6含量 17.430 11.810  -0.70474
21 Single-target MLP 有害藻类7含量  6.538  4.298  -1.02128
22  Multi-target MLP 有害藻类1含量 38.515 23.667  -2.53413
23  Multi-target MLP 有害藻类2含量 25.791 12.382  -5.21770
24  Multi-target MLP 有害藻类3含量 28.219 12.898 -25.00594
25  Multi-target MLP 有害藻类4含量  6.685  3.622  -4.83873
26  Multi-target MLP 有害藻类5含量 27.636 13.686  -7.38039
27  Multi-target MLP 有害藻类6含量 23.271 12.140  -2.03865
28  Multi-target MLP 有害藻类7含量 17.530  5.966 -13.52968

========== 各模型整体平均表现排名 ==========
# A tibble: 6 × 4
  Model             Avg_RMSE Avg_MAE Avg_R2
  <chr>                <dbl>   <dbl>  <dbl>
5 Single-target MLP    14.5     9.86 -1.87 
6 Multi-target MLP     23.9    12.1  -8.65 
```

