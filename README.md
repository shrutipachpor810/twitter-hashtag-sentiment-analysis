# 🔥 Twitter Hashtag Sentiment Analyzer

## 📌 Overview

This project analyzes the sentiment of Twitter hashtags using advanced Natural Language Processing (NLP) techniques. It leverages a transformer-based deep learning model (**Twitter RoBERTa**) to classify sentiment as **Positive, Negative, or Neutral**, and provides deeper insights such as polarity scores, trend analysis, and visualizations.

---

## 🚀 Features

* 🧠 **AI-based Sentiment Analysis** using Twitter RoBERTa
* 📊 **Polarity Score** (-1 to +1) for sentiment intensity
* 📈 **Trend Score** to identify overall sentiment direction
* 📉 **Sentiment Distribution Graphs**
* 🌈 **WordCloud Visualization** of tweet content
* 🔍 **Top Positive & Negative Tweets Extraction**
* ⚡ **Fallback for unseen hashtags (no dataset needed)**
* 🌐 **Streamlit Web App Interface**

---

## 🛠️ Tech Stack

* **Python**
* **Pandas**
* **Matplotlib**
* **Transformers (Hugging Face)**
* **Streamlit**
* **WordCloud**
* **Regex (re)**

---

## 📂 Project Structure

```
├── app.py                # Streamlit web app
├── main.py               # CLI-based analysis
├── full-corpus.csv       # Dataset
├── requirements.txt      # Dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```
git clone https://github.com/shrutipachpor810/twitter-hashtag-analysis.git
cd twitter-hashtag-analysis
```

### 2. Create virtual environment

```
python -m venv env
env\Scripts\activate   # Windows
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Usage

### 🔹 Run CLI version

```
python main.py
```

### 🔹 Run Streamlit Web App

```
streamlit run app.py
```

---

## 📊 Example Output

* Sentiment distribution (Positive / Negative / Neutral)
* Average polarity score
* Trend score
* WordCloud visualization
* Top positive & negative tweets

---

## 🧠 Model Used

* **Twitter RoBERTa** (`cardiffnlp/twitter-roberta-base-sentiment`)
* Trained on millions of tweets for better understanding of:

  * Slang
  * Emojis
  * Hashtags
  * Informal language

---

## 📌 Applications

* Social media monitoring
* Brand sentiment analysis
* Trend detection
* Public opinion mining

---

## 🔮 Future Enhancements

* 🔴 Real-time Twitter API integration
* 🎨 Advanced UI (dark mode, dashboards)
* 🌍 Multi-language sentiment analysis
* 📊 Model comparison (BERT vs RoBERTa vs VADER)

---

## 👩‍💻 Author

**Shruti Pachpor**
GitHub: https://github.com/shrutipachpor810

---

## ⭐ If you like this project

Give it a ⭐ on GitHub!
