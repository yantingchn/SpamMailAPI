# Spam Check API Design

This document outlines the solution design for the Spam Check API.

## Design Approach
The solution design consists of the following components:

1. Data Preprocessing: 
The initial step in any machine learning task is data preprocessing. For text classification, this involves *tokenization* and *vectorization*.
    - tokenization: breaking the text into individual word tokens, including stop words removal. In this step, simple tokenization with **re** and complex tokenization with **NLTK** are implemented. 
    - vectorization: converting the text into numerical vectors that can be leveraged by ML models. *CountVectorizer* and *TfidfVectorizer* are considered in this step.

2. Model Training: After preprocessing the data, it is used to train a ML model. There are many different types of models that could be used for this task, including Naive Bayes, Support Vector Machines, and Neural Networks. For this task, *Naive Bayes* is selected as the base model due to its good performance in classification tasks.

3. Model Evaluation: The dataset is split into a training set and a test set. The model's performance is measured using various metrics, such as accuracy, precision, recall, and F1 score. The **MLflow** management framework is used for this purpose. Notably, as the data distribution is imbalanced, the AUC score and F1 score are weighted to provide a more comprehensive understanding of the model’s performance by considering both false positives and false negatives.

4. Model Deployment: 
The **FastAPI** library is used to create a web application that wraps the model in a REST API. MLflow provides a functional API to save the trained model to a file, which can then be loaded by the web application.

5. Model Updating: With **MLflow**, we can easily register new ML models and compare their performance. By setting the model alias to champion and reloading the API server, the new model can be deployed.


## Models Evaluation
The evaluation process is spilted into 2 parts:
- The selection of Tokenizer and Vectorizer
- The selection of Classification Model

### The Selection of Tokenizer and Vectorizer
Firstly, 4 different model pipelines are used to examine the performance of Tokenizer and Vectorizer.
For simplicity, Naive Bayes Model is selected as base model.
1. Model 1 (`count-navie-bayes-model`): **RE** Tokenizer + CountVectorizer + Naive Bayes Model
2. Model 2 (`nltk-count-navie-bayes-model`): **NLTK** Tokenizer + CountVectorizer + Naive Bayes Model
3. Model 3 (`tfidf-navie-bayes-model`): **RE** Tokenizer + TfidfVectorizer + Naive Bayes Model
4. Model 4 (`tfidf-nltk-navie-bayes-model`): **NLTK** Tokenizer + TfidfVectorizer + Naive Bayes Model

#### Evaluation Metrics
|         | accuracy | roc_auc | precision | f1 score | recall |
|---------|----------|---------|-----------|----------|--------|
| Model 1 | 0.9782   | 0.9705  | 0.9674    | 0.9603   | 0.9533 |
| Model 2 | 0.9742   | 0.9632  | 0.9669    | 0.9526   | 0.9387 |
| Model 3 | 0.9016   | 0.8221  | 1         | 0.7836   | 0.6443 |
| Model 4 | 0.8984   | 0.8163  | 1         | 0.775    | 0.6326 |

Insights from the evaluation metrics:
1. Due to the size of the dataset and its distribution, a simple model like Naive Bayes can achieve a 0.97 AUC score.
2. TfidfVectorizer (model 3 and model 4) might perform worse if the dataset is too small. It relies on the inverse document frequency, which may not be accurately calculated if the dataset is too small.
3. There is no significant difference when applying NLTK tools (lemmatization and stemming).

Based on the evaluation, the combination of **RE** tokenization and CountVectorizer are selected.

### The Selection of Classification Model
In order to streamline the process of model selection, we utilize **PyCaret**. This tool automates the training and validation process, significantly simplifying the task of model selection. As discussed in the `3. Model Evaluation section`, the AUC score is used as the primary metric to identify the best performing model.


## Potential Improvements
1. The dimension of input features (40000+) could be further reduced to enhance the model's stability. Techniques such as advanced word embedding via NLP and Bag-of-words should be considered.
2. More advanced models such as BERT and GPT could provide a better semantic understanding of the context. Using transfer learning and pre-trained models could potentially improve the model's performance.
