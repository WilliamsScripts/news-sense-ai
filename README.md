# 🧠 NewsSense AI — Multi-Class News Categorizer

An end-to-end machine learning system that classifies news articles into multiple categories such as **World, Sports, Business, and Science/Technology**.

## Features

- 🧠 ML model (TF-IDF + Logistic Regression)
- ⚡ FastAPI backend for inference
- 📊 Model evaluation (Precision, Recall, F1-score)

---

## 🚀 Demo

Paste any news article or headline into the mobile app and get instant predictions.

### Example Output

```json
{
  "category": "Technology",
  "confidence": 0.94
}
```

---

## 🏗️ System Architecture

```
   FastAPI Backend
        ↓
 ML Model (TF-IDF + Logistic Regression)
        ↓
   Prediction + Confidence
```

---

## 📂 Project Structure

```
news-categorizer/
├── app/
│   ├── main.py
│
├── training/
│   ├── train.py
│   ├── model.pkl
│   ├── label_map.pkl
│
├── requirements.txt
└── README.md
```

---

## 🧠 Machine Learning Pipeline

**Step 1:** Data Loading  
- AG News dataset ([HuggingFace](https://huggingface.co/datasets/ag_news))

**Step 2:** Preprocessing  
- TF-IDF vectorization  
- Stop word removal  
- Feature selection (max 10,000 features)

**Step 3:** Model Training  
- Logistic Regression classifier  
- Train-test split (80/20)

**Step 4:** Evaluation  
- Precision  
- Recall  
- F1-score

**Step 5:** Serialization  
- Save model using Joblib

---

## 📊 Dataset

We use the AG News Dataset:

**Categories:**
- World
- Sports
- Business
- Science/Technology

📎 Dataset: [https://huggingface.co/datasets/ag_news](https://huggingface.co/datasets/ag_news)

---

## 🏋️ Model Training

Run training script:

```bash
python training/train.py
```

**Output:**
- Classification report
- Saved model (`model.pkl`)
- Saved label mapping (`label_map.pkl`)

---

## ⚙️ Backend Setup (FastAPI)

Install dependencies:

```bash
cd backend
pip install -r requirements.txt
```

Run server:

```bash
python -m uvicorn app.main:app --reload
```

Server runs at:  
http://127.0.0.1:8000

---

## 📡 API Reference

**POST** `/predict`

**Request:**
```json
{
  "text": "Apple unveils new AI chip for MacBooks"
}
```

**Response:**
```json
{
  "category": "Technology",
  "confidence": 0.94
}
```

---

## 🧩 Tech Stack

**Backend**
- FastAPI
- Scikit-learn
- Pandas
- Joblib

**Machine Learning**
- TF-IDF Vectorizer
- Logistic Regression