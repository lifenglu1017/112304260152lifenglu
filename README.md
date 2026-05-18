# 机器学习实验：基于 Word2Vec 的情感预测

## 1. 学生信息

- **姓名**：李凤禄
- **学号**：112304260152
- **班级**：\[学生班级]

> 注意：姓名和学号必须填写，否则本次实验提交无效。

***

## 2. 实验任务

本实验基于 IMDB 电影评论数据，使用 **Word2Vec 将文本转为向量特征**，再结合 **逻辑回归分类模型** 完成情感预测任务，并将结果提交到 Kaggle 平台进行评分。

本实验重点包括：

- 文本预处理（去HTML标签、小写化、标点处理、去停用词）
- Word2Vec 词向量训练
- 句子向量表示（词向量均值）
- 逻辑回归分类模型训练
- Kaggle 结果提交与分析（评价指标：AUC-ROC）

***

## 3. 比赛与提交信息

- **比赛名称**：Bag of Words Meets Bags of Popcorn
- **比赛链接**：<https://www.kaggle.com/competitions/word2vec-nlp-tutorial>
- **提交日期**：\[提交日期]
- **GitHub 仓库地址**：<https://github.com/pengle-study/112304260149pengle>
- **GitHub README 地址**：<https://github.com/pengle-study/112304260149pengle/blob/main/README.md>

> 注意：GitHub 仓库首页或 README 页面中，必须能看到“姓名 + 学号”，否则无效。

***

## 4. Kaggle 成绩

请填写你最终提交到 Kaggle 的结果：

- **Public Score**：\[Public Score]
- **Private Score**（如有）：\[Private Score]
- **排名**（如能看到可填写）：\[排名]

***

## 5. Kaggle 截图

请在下方插入 Kaggle 提交结果截图，要求能清楚看到分数信息。

![Kaggle截图](./images/kaggle_score.png)

> 建议将截图保存在 `images` 文件夹中。\
> 截图文件名示例：`2023123456_张三_kaggle_score.png`

***

## 6. 实验方法说明

### （1）文本预处理

**我的做法：**

1. **去 HTML 标签**：使用正则表达式 `r'<br />'` 将 HTML 换行标签替换为空格
2. **小写化**：将所有文本转换为小写，使 "Movie" 和 "movie" 视为同一个词
3. **缩写处理**：处理常见英文缩写，如 `can't` → `cannot`，`won't` → `will not`，`n't` →  ` not`（保留否定含义）
4. **标点符号移除**：移除非字母字符，但保留空格
5. **停用词过滤**：移除英文停用词（如 the, a, is 等），但保留否定词（not, no, never, nor 等），因为否定词对情感分析至关重要

***

### （2）Word2Vec 特征表示

**我的做法：**

- **训练方式**：使用标注数据（25,000条）和无标注数据（50,000条）共75,000条评论训练 Word2Vec 模型
- **词向量维度**：300维
- **训练参数**：窗口大小为5，最小词频为5，使用CBOW算法（sg=0），训练10个epoch
- **句子向量生成**：对每条评论中的所有词向量取平均值，得到300维的句子向量表示

***

### （3）分类模型

**我的做法：**

- 使用 **逻辑回归**（Logistic Regression）作为分类模型
- **模型参数**：最大迭代次数2000，正则化参数C=4.0，随机种子42
- **模型训练**：使用训练集的句子向量作为特征，情感标签作为目标变量进行训练
- **预测输出**：使用 `predict_proba()` 方法输出概率值（0到1之间的小数），而非二分类标签

***

## 7. 实验流程

**我的实验流程：**

1. **读取数据**：加载标注训练集（labeledTrainData.tsv）、无标注训练集（unlabeledTrainData.tsv）和测试集（testData.tsv）
2. **文本预处理**：去 HTML 标签、小写化、处理缩写、去除特殊字符、分词、去停用词（保留否定词）
3. **训练 Word2Vec**：使用标注数据和无标注数据共75,000条评论训练词向量模型（300维）
4. **生成句向量**：对每条评论的词向量取平均值，得到300维句子向量
5. **交叉验证**：使用5折交叉验证评估模型性能，计算AUC-ROC分数
6. **训练分类器**：使用逻辑回归模型在训练集上进行训练
7. **预测测试集**：在测试集上预测情感概率（0到1之间）
8. **生成提交文件**：将预测结果保存为 submission.csv，sentiment 列为概率值

***

## 8. 文件说明

**我的项目结构：**

```text
project/
├─ labeledTrainData.tsv/
│  └─ labeledTrainData.tsv    # 标注训练数据（25,000条评论）
├─ unlabeledTrainData.tsv/
│  └─ unlabeledTrainData.tsv  # 无标注训练数据（50,000条评论）
├─ testData.tsv/
│  └─ testData.tsv            # 测试数据（25,000条评论）
├─ word2vec_sentiment.py      # Word2Vec方法主程序脚本
├─ sentiment_analysis.py      # TF-IDF方法备选脚本
├─ submission.csv             # 提交文件（运行脚本后生成）
└─ README.md                  # 实验报告
```

**文件说明：**

- `labeledTrainData.tsv/`：存放标注训练数据，包含 id、sentiment（0/1）、review 字段
- `unlabeledTrainData.tsv/`：存放无标注训练数据，用于增强 Word2Vec 训练
- `testData.tsv/`：存放测试数据，仅包含 id 和 review 字段
- `word2vec_sentiment.py`：完整的 Word2Vec 实验脚本，包含数据读取、预处理、Word2Vec训练、分类和预测
- `sentiment_analysis.py`：TF-IDF方法备选脚本，当gensim不可用时使用
- `submission.csv`：生成的提交文件，包含 id 和 sentiment（概率值）字段

