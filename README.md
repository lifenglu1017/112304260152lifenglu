# Word2Vec 情感分析实验

<br />

## 一、实验信息

| 项目       | 内容            |
| -------- | ------------- |
| **姓名**   | 李凤禄           |
| **学号**   | 112304260152  |
| **班级**   | 数据1231        |
| **课程**   | 机器学习          |
| **实验名称** | Word2Vec 情感分析 |

## 二、实验目标

1. 掌握 Word2Vec 词向量模型的原理和应用
2. 学习文本预处理技术
3. 理解情感分析的基本流程和评估指标
4. 实现达到及格线（AUC ≥ 0.94）的模型

## 三、实验背景

本实验基于 Kaggle 比赛 [Word2Vec NLP Tutorial](https://www.kaggle.com/competitions/word2vec-nlp-tutorial)，数据集包含 50,000 条电影评论。

## 四、实现方案

### 4.1 文本预处理

- HTML标签去除 → 小写化 → 缩写处理 → 标点去除 → 停用词过滤 → 分词

### 4.2 Word2Vec 训练参数

| 参数    | 值    |
| ----- | ---- |
| 词向量维度 | 300  |
| 窗口大小  | 5    |
| 最小词频  | 5    |
| 算法    | CBOW |

### 4.3 逻辑回归分类

输出概率值（0-1之间）用于 Kaggle 提交。

## 五、实验结果

| 指标             | 分数          |
| -------------- | ----------- |
| **Kaggle AUC** | **0.94873** |
| **及格线**        | 0.94        |
| **状态**         | ✅ 已达标       |

### Kaggle 提交记录

| 提交时间    | 文件             | 分数 (AUC) | 状态   |
| ------- | -------------- | -------- | ---- |
| 2024年5月 | submission.csv | 0.94873  | ✅ 成功 |
| 2024年5月 | submission.csv | 0.94113  | ✅ 成功 |

## 六、文件结构

```
机器学习实验2/
├── word2vec_sentiment.py     # 主程序脚本
├── submission.csv            # Kaggle 提交文件
├── .gitignore                # Git 忽略配置
└── README.md                 # 实验报告
```

## 七、运行方式

```bash
# 安装依赖
pip install pandas numpy gensim scikit-learn nltk beautifulsoup4

# 运行脚本
python word2vec_sentiment.py
```

## 八、技术栈

| 工具           | 用途          |
| ------------ | ----------- |
| Python       | 编程语言        |
| gensim       | Word2Vec 模型 |
| scikit-learn | 逻辑回归        |
| pandas       | 数据处理        |
| nltk         | 文本处理        |

## 九、总结

本实验成功实现了基于 Word2Vec 和逻辑回归的情感分析模型，在 Kaggle 比赛中取得了 **0.94873** 的 AUC 分数，超过了 0.94 的及格线。
