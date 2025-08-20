# Fake News Detection with Deep Learning  

## Project Overview  
Misinformation spreads rapidly across social media and news outlets, influencing elections, public health, and financial markets. Detecting fake news is a **critical real-world challenge** because it helps protect democracy, reduce the spread of harmful rumors, and maintain trust in journalism.  

This project develops a **Fake News Detection system** using **Natural Language Processing (NLP)** and **Deep Learning**. By combining pre-trained **GloVe word embeddings** with a **stacked LSTM neural network**, the model achieves **~99.8% accuracy** in classifying news articles as *real* or *fake*.  
- Tackles a **globally recognized problem** (misinformation).  
- Demonstrates mastery of **modern NLP techniques**.  
- Delivers a solution that could be applied in **industry settings** (social media monitoring, journalism, fact-checking tools).  

---

## Key Skills Demonstrated  
This project highlights **high-demand skills** in data science and machine learning:  

1. **Data Cleaning & Preprocessing** – Tokenization, stopword removal, lemmatization, regex-based text normalization.  
2. **Exploratory Data Analysis (EDA)** – Class balance, subject breakdown, and text-length analysis with visualizations.  
3. **Transfer Learning (Word Embeddings)** – Leveraged pre-trained **GloVe Twitter vectors** to capture semantic meaning.  
4. **Deep Learning with LSTMs** – Designed and trained a bi-layer LSTM model for sequence modeling.  
5. **Model Optimization** – Used dropout, learning rate scheduling (ReduceLROnPlateau), and early stopping.  
6. **Evaluation Metrics** – Classification report, F1-score, precision/recall, and confusion matrix.  
7. **Reproducibility** – Clean notebook pipeline, GitHub release for dataset/embeddings, structured project repo.  

---

## Dataset Exploration  

**Class Distribution:**  
- Fake News: **23,478 samples**  
- Real News: **21,211 samples**  
- Dataset is fairly balanced.  

**Subject Breakdown:**  
- Fake news clusters in *general* and *politics_left*.  
- Real news focuses on *politics* and *world*.  

**Text Length:**  
- Real news: Longer, more consistent articles.  
- Fake news: Shorter and less uniform.  

 *Example Visualizations:*  
- Class Distribution Plot  
- Subject Breakdown Plot  
- Text Length Distribution Plot  

---

## Model Architecture  

The model was built using **TensorFlow/Keras**:  

- **Embedding Layer** (initialized with GloVe 100d vectors, non-trainable).  
- **LSTM Layer 1** – 128 units, return sequences, dropout for regularization.  
- **LSTM Layer 2** – 64 units, dropout for generalization.  
- **Dense Layer** – 32 units, ReLU activation.  
- **Output Layer** – 1 unit, Sigmoid activation (binary classification).  

📷 *Training Curves:*  
- Accuracy and Loss progression during training.  
<img width="1630" height="702" alt="download" src="https://github.com/user-attachments/assets/e464feea-d272-4fe9-be69-0e3ab46348f5" />

---

## Results  

**Performance Metrics:**  
- Accuracy: **99.8%**  
- Precision: **0.998**  
- Recall: **0.998**  
- F1-score: **0.998**  

**Confusion Matrix:**  
- Correctly identifies **>99.7%** of both Real and Fake news.  

 *Confusion Matrix Heatmap:*  
<img width="509" height="470" alt="download" src="https://github.com/user-attachments/assets/015a3c3a-5e7f-4644-a353-2ec10456975a" />

---

## Key Insights  
- **Fake news is often shorter and less structured** compared to real news.  
- Subject categories provide **additional predictive signals**.  
- Pre-trained embeddings (GloVe) significantly boosted model performance compared to random initialization.  
- The model generalizes well without overfitting thanks to dropout and adaptive learning rate scheduling.  

---

## Next Steps / Future Work  
- Experiment with **Transformer models** (BERT, RoBERTa) for contextual embeddings.  
- Expand evaluation to **unseen, real-time data** (Twitter feeds, recent news).  
- Package into a **Streamlit or Flask app** for interactive use.  
- Deploy via **Docker + cloud service** (AWS/GCP/Azure).  


