
# 🎯 Sentiment Analysis on Text | Comparative Machine Learning Project

Welcome!  
This project focuses on **sentiment analysis** using English textual data. I applied **three different machine learning algorithms** (KNN, K-Means, Random Forest) to compare their performance in classifying emotions from text.

---

## 📌 Project Overview

This repository contains a machine learning pipeline for performing sentiment classification on over 21,000 English text samples. The core stages of the project are:

- Text preprocessing and cleaning
- Text vectorization using **TF-IDF**
- **Supervised Learning:**  
  - K-Nearest Neighbors (KNN)  
  - Random Forest Classifier
- **Unsupervised Learning:**  
  - K-Means Clustering with PCA Visualization
- Model evaluation using **Accuracy**, **Confusion Matrix**, and **F1-Score**
- Visualization of results with **matplotlib** and **seaborn**

---

## 🧠 Dataset

The dataset used in this project is:
- 📂 **[Emotions in Text Dataset](https://www.kaggle.com/datasets/ishantjuyal/emotions-in-text)**
- 💬 Contains over 21,000 labeled English sentences categorized by emotions such as *happy*, *sad*, *anger*, *fear*, and more.

---

## ⚙️ Technologies & Libraries

- 🐍 Python 3
- 📚 Libraries:
  - `scikit-learn`
  - `pandas`
  - `numpy`
  - `matplotlib`
  - `seaborn`
- 🧠 NLP Tools:
  - `TF-IDF Vectorizer`
- 📝 Development Environment:
  - Jupyter Notebook (.ipynb)

---

## 🚀 Model Performance Comparison

| Algorithm        | Accuracy | Notes                            |
|------------------|----------|----------------------------------|
| 🎯 Random Forest | **88%**  | Best performer overall           |
| 📘 KNN           | 66%      | Moderate, depends on k-value     |
| 🔵 K-Means       | 34%      | Unsupervised baseline            |

Random Forest clearly outperforms the others in emotion classification tasks due to its ensemble learning capabilities.

---

## 📊 Visualization & Results

- **PCA plots** to visualize K-Means clustering
- **Confusion matrices** for KNN and Random Forest
- **F1-scores**, **accuracy**, and **support metrics** across models
- All evaluation metrics are included in the notebook with high-quality plots

---

## 📽️ YouTube Presentation

I also prepared a video presentation explaining the full project with visuals, coding steps, and results.

🔗 **Watch here:** **[Youtube Video](https://www.youtube.com/watch?v=55jTT_mmddo&t=2s)**  
👍 If you find it useful, don’t forget to like, comment, and subscribe!

---

## 💬 Who Is This For?

This project is ideal for:

- 🎓 Students learning Machine Learning & NLP  
- 🧑‍💻 Developers interested in real-world text classification  
- 🔍 Researchers comparing ML algorithms in practice  
- 🚀 Anyone curious about applied data science and emotion detection

---

## 📁 Repository Structure

```
📦 sentiment-analysis-project/
│
├── data/                     # Dataset files (CSV or JSON)
├── notebook/                 # Jupyter notebook (complete pipeline)
├── results/                  # Confusion matrices, PCA plots
├── models/                   # Trained model files (optional)
└── README.md                 # Project documentation
```

---

## 📬 Contact

Created by **İsmail Bayhan Yaltırık**  
🎓 Artificial Intelligence & Machine Learning Engineering  
🏫 Konya Technical University  
📧 My E-Mail: **info@ismailyaltirik.com**

---

## 🏷️ Tags

`#Python` `#MachineLearning` `#SentimentAnalysis` `#RandomForest` `#KNN` `#KMeans` `#TFIDF` `#TextClassification` `#NLP` `#EmotionsInText` `#DataScience`

---

## ⭐ Support

If you like this project, consider giving it a ⭐ on GitHub and sharing it with your peers. Contributions and feedback are always welcome!

---

> “Data is the new oil, and machine learning is the refinery.”  
> — Inspired by modern AI innovators
