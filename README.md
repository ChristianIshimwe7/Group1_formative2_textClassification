# Comparative Analysis of Text Classification with Multiple Embeddings

## Project Description
This repository contains the code and documentation for a group project focused on a comparative analysis of text classification models using multiple word embedding techniques. The project evaluates both traditional machine learning approaches and deep learning sequence models to understand how embedding choices influence classification performance on a multiclass sentiment analysis task.

---

## Project Objectives
- Analyze the impact of different word embedding techniques on text classification performance  
- Compare traditional machine learning models with recurrent neural network architectures  
- Investigate the compatibility between embeddings and model architectures  
- Identify strengths and limitations of each embedding–model combination  

---

## Dataset
The dataset used in this project is a multiclass sentiment analysis dataset obtained from **Hugging Face**.  
Each text sample is labeled as one of the following sentiment classes:

- **0** — Negative  
- **1** — Neutral  
- **2** — Positive  

The dataset consists of short, informal text samples (e.g., social media–style sentences) and is pre-split into training, validation, and test sets. Due to its moderate size, the original split was preserved.
https://huggingface.co/datasets/Sp1786/multiclass-sentiment-analysis-dataset 
---

## Preprocessing
The following preprocessing steps were applied uniformly across all experiments:

1. Conversion of text to lowercase  
2. Removal of punctuation, URLs, and email addresses  
3. Stopword removal using NLTK  
4. Lemmatization using WordNetLemmatizer  
5. Tokenization for embedding-based models  
6. Padding and truncation for sequence models  

---

## Word Embedding Techniques
The project evaluates the following embedding representations:

- **TF-IDF**
- **Word2Vec**
  - Continuous Bag of Words (CBOW)
- **GloVe (pre-trained embeddings)**

---

## Model Architectures

### Traditional Machine Learning Models
- Logistic Regression  

### Deep Learning Models
- Recurrent Neural Network (RNN)  
- Long Short-Term Memory (LSTM)  
- Gated Recurrent Unit (GRU)  

All deep learning models were implemented using TensorFlow/Keras and evaluated using consistent training and validation procedures.

---

## Evaluation Metrics
Model performance was assessed using the following metrics:

- Accuracy  
- Weighted F1-Score  

Confusion matrices were used for qualitative error analysis.

---
## Group Members and Contributions

- Christian Ishimwe — Traditional Machine Learning Models
- Caline Uwingabire — RNN Models
- Jinelle Nformi — LSTM Models
- Yvette Gahamanyi — GRU Models
