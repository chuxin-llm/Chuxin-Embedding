# Chuxin-Embedding LLM Generate Data

<!-- Provide a quick summary of what the model is/does. -->
本文档介绍 Chuxin-Embedding 使用QWEN-72b生成检索数据的方法。以及清洗检索数据的流程，主要分为难负例挖掘、rerank清洗、rule清洗、LLM clean query等。本文档仅记录chuxin-embedding的实验过程，如有任何疑问，欢迎交流探讨。

## Requirements
```python
pip install FlagEmbedding
pip install vllm
pip install langchain
```

## LLM生成数据
### 介绍
由于公开的检索数据有限，特定领域的检索数据缺乏等因素，使用数据增强的手段提升向量模型检索能力是一种常见的方法。一条检索训练数据由（query，pos，neg）组成，pos为若干与之相关的文档，neg为若干与之不相关的文档。已有海量文档，则可以通过LLM生成query以实现数据增强的目的。

### 主要思想
#### 通用corpus
- 针对一篇文档，首先，生成对文档可能有兴趣的角色，角色会在什么场景对文档感兴趣，文档的类别等，思路来源于[1]。
- 再根据生成的角色、场景和类别，生成具有多样性的query。其中，query具有多样性尤其重要，较简单的query 将影响loss较小，则导致模型无法学习复杂的检索能力。prompt中，角色、场景、对pos分类为财务/实体描述/医疗/政策/科学/论文摘要/百科知识/电影等等，均是为了提升query的多样性。

### 脚本
```python
bash run_llm_qwen2_72b_example.sh $1 $2 $3
```



## 清洗检索数据
该部分针对所有检索数据，包括公开的检索数据集、LLM生成的检索数据等的清洗。

### 难负例挖掘
### 主要思想
当一条检索训练数据（query，pos）存在时，neg 的难易程度将影响模型对于检索能力的学习。
使用检索能力较强的合适的向量模型，检索每一个query的前k条相关文档，前k中在一定范围随机选择15条neg，15条pos。其中，若算力有限，pos的挖掘为可选项（经实验验证，大多数情况下，pos的挖掘可增强模型能力）。参考[2]。
### 脚本
```python
sh run_hard_negative.sh $1
```

### Rerank清洗
### 主要思想
针对难负例挖掘的少量的正例和负例，可能存在一些假正例和假负例。
比较了几个reranker 模型，选了第二个： bge-reranker-large / bge-reranker-large-m3 / minicpm-reranker。
选择阈值  pos>1， neg<0。
### 脚本
```python
python preprocess_rerankclean_pairs.py $1
```

### Rules清洗
### 主要思想
此步骤主要针对合成数据，为了保证文档的高质量，将进行一些规则清洗。
- 文档中可能包括一些无意义的字符
- 无意义的结构性文档（目录等）
- query长度大于pos/neg两倍的（通常意义检索是由短句检索长文档）
- query内容和pos/neg内容高度重合的（过于简单，影响数据的复杂性）
以上几种认定为可删除部分。
### 脚本
```python
python preprocess_rulesclean_pairs.py $1
```

### Query清洗
此步骤主要针对合成数据，由于LLM生成query时存在一些不符合常理的，如过分针对文档描述具体细节，一个query存在多个不相关的问题等。
### 脚本
```python
bash run_preprocess_llmclean_query_example.sh $1 $2
```

## 消融实验

**C_MTEB**
| **Exp**|**LLM**|**HardNegative**|**Rerankclean**| **Average** | **CmedqaRetrieval** | **CovidRetrieval** | **DuRetrieval** | **EcomRetrieval**   | **MedicalRetrieval** | **MMarcoRetrieval** | **T2Retrieval** | **VideoRetrieval** |
| :-------------------: | :---------: |:---------: |:---------: |:---------: | :---------: |:------------: | :-----------: | :-----------: | :-------: | :----------: | :-------: | :----------: |
|1(base)||||65.86|36.83|74.23|81.99| 61.17  |  57.91 | 73.57 |79.04    | 62.14 |
|2|√|√|| 65.62 |  35.70   | 74.10 | 81.43 | 60.17 | 57.76 |  71.34 | 80.31 | 64.14|
|3|√|√|√(pos1)| 67.24 |  33.40   | 78.28 | 83.08 | 62.24 | 57.21 |  73.62 | 82.39 | 67.69|
|4|√|√|√| 67.71 |  35.42   | 77.09 | 84.13 | 63.02 | 58.73 |  73.58 | 82.29 | 67.46|

- 数据介绍：生成数据 dureader corpus，共7万条左右
- 实验2：仅难负例挖掘的数据
- 实验3：难负例挖掘+Rerank清洗（pos仅保留一个文档且score最大）
- 实验4：难负例挖掘+Rerank清洗
- 实验结论：Rerank 清洗，且保留多个pos 策略有效（请注意，数据不同，实验略有差异）


**C_MTEB**
| **Exp**|**LLM**|**HardNegative**|**Rerankclean**|**Queryclean**| **Average** | **CmedqaRetrieval** | **CovidRetrieval** | **DuRetrieval** | **EcomRetrieval**   | **MedicalRetrieval** | **MMarcoRetrieval** | **T2Retrieval** | **VideoRetrieval** |
| :-------------------: | :---------: |:---------: |:---------: |:---------: |:---------: | :---------: |:------------: | :-----------: | :-----------: | :-------: | :----------: | :-------: | :----------: |
|1(base)|||||65.86|36.83|74.23|81.99| 61.17  |  57.91 | 73.57 |79.04    | 62.14 |
|2|√|√|√|| 67.92 |  36.85   | 78.51 | 83.04 | 62.52 | 58.09 |  74.07 | 83.35 | 66.97|
|3|√|√|√|√| 68.02 |  36.76  | 78.85 | 83.29 | 62.61 | 58.35 |  74.20 | 83.46 | 66.65|

- 数据介绍：生成数据 wudao corpus，5万条数据
- 实验2：难负例挖掘+rerank清洗
- 实验3: 难负例挖掘+rerank清洗+query清洗
- 实验结论：Query清洗，策略有效 （请注意，数据不同，实验略有差异）

**C_MTEB**
| **Exp**|**HardNegative**|**Rerankclean**|**Rulesclean**| **Average** | **CmedqaRetrieval** | **CovidRetrieval** | **DuRetrieval** | **EcomRetrieval**   | **MedicalRetrieval** | **MMarcoRetrieval** | **T2Retrieval** | **VideoRetrieval** |
| :-------------------: |:---------: |:---------: |:---------: |:---------: | :---------: |:------------: | :-----------: | :-----------: | :-------: | :----------: | :-------: | :----------: |
|1(base)||||65.86|36.83|74.23|81.99| 61.17  |  57.91 | 73.57 |79.04    | 62.14 |
|2|√(old)|√(old)|| 69.42 | 39.68 | 78.32 | 84.65 | 64.61 | 60.09 | 74.87  | 83.51 | 69.62|
|3|√|√|| 69.78 |  38.93  | 79.27 | 85.05 | 64.42 | 60.21 |  76.33 | 84.47 | 69.59|
|4|√|√|√| 69.88 |  38.73  | 79.26 | 85.18 | 64.63 | 60.19 |  76.46 | 84.53 |70.06|

- 数据介绍：百万级别数据
- 实验2：难负例挖掘（常见向量模型+仅neg难负例挖掘+bge-reranker-large）
- 实验3：难负例挖掘（Chuxin-Embedding向量模型+难负例挖掘（pos，neg）+ bge-reranker-large-m3）
- 实验4：难负例挖掘（Chuxin-Embedding向量模型+难负例挖掘（pos，neg）+ bge-reranker-large-m3）+ Rules 清洗
- 实验结论：清洗策略在百万级别数据上有效。

**AIR-BENCH**
| **Exp**|**LLM**|**HardNegative**|**Rerankclean**|**Queryclean**| **Average** | **wiki_zh** | **web_zh** | **news_zh** | **healthcare_zh**   | **finance_zh** |
| :-------------------: | :---------: |:---------: |:---------: |:---------: |:---------: | :---------: |:------------: | :-----------: | :-----------: | :-------: |
|1(base)|||||63.91|75.88|67.71|63| 61.36  |  51.62 |
|2|√|√|√||64.46|75.89|68.01|63.65|62.85|51.91|

- 数据介绍：生成数据，百万级别数据
- 实验2：难负例挖掘+rerank清洗+query清洗
- 实验结论：生成数据策略+清洗策略在模型能力上有提升 （请注意，数据不同，实验略有差异）


### Reference
1. https://arxiv.org/pdf/2401.00368
2. https://github.com/FlagOpen/FlagEmbedding




