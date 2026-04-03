# SentimentIQ – NLP-Based Social Media Sentiment Analysis Dashboard

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![HuggingFace](https://img.shields.io/badge/HuggingFace-DistilBERT-yellow?style=flat-square&logo=huggingface)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square&logo=pytorch)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> **Analyze the sentiment of any text, bulk CSV data, or live YouTube comments — powered by a fine-tuned DistilBERT transformer model.**

---

## 🚀 Live Demo
🔗 **[Click here to try SentimentIQ](https://sentimentiq-dashboard-njiprgwlcchwkuemvrqwn4.streamlit.app/)**

---

## 📌 Features

- **📝 Single Text Analyzer** — Type any sentence and get instant sentiment with confidence score
- **📂 Bulk CSV Analysis** — Upload a CSV file and analyze thousands of rows at once
- **📡 Live YouTube Comments** — Paste any YouTube URL and analyze real comments live
- **😊 3-Class Output** — Positive, Neutral, Negative with emoji indicators
- **🚫 Gibberish Detection** — Rejects random/invalid inputs intelligently
- **📊 Interactive Charts** — Pie chart, bar chart, confidence distribution
- **⬇️ Download Results** — Export analyzed data as CSV

---

## 🧠 Model Details

| Property | Details |
|----------|---------|
| Base Model | `distilbert-base-uncased` |
| Task | Sentiment Classification |
| Dataset | Twitter Sentiment 140 |
| Training Samples | 15,000 tweets |
| Validation Accuracy | **82.0%** |
| F1 Score | **0.82** |
| Framework | HuggingFace Transformers |
| Hosted On | HuggingFace Hub → `vaibhav9700/sentimentiq-distilbert` |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.10 | Core language |
| HuggingFace Transformers | DistilBERT fine-tuning & inference |
| PyTorch | Deep learning backend |
| Streamlit | Web dashboard UI |
| Plotly | Interactive charts |
| YouTube Data API v3 | Live comment fetching |
| Pandas | Data processing |
| HuggingFace Hub | Model hosting |

---

## 📁 Project Structure

```
sentimentiq-dashboard/
│
├── app.py                  # Main Streamlit dashboard
├── requirements.txt        # Python dependencies
├── .streamlit/
│   └── secrets.toml        # API keys (not uploaded to GitHub)
└── README.md               # Project documentation

Model hosted on HuggingFace Hub:
vaibhav9700/sentimentiq-distilbert
```

---

## 📊 Screenshots

### 🖥️ Main Dashboard
![Dashboard](https://raw.githubusercontent.com/vaibhavv-labs/sentimentiq-dashboard/main/assets/dashboard.png)

### 📊 Analytics View
![Analytics](https://raw.githubusercontent.com/vaibhavv-labs/sentimentiq-dashboard/main/assets/analytics.png)

### 📡 Live YouTube Data
![Live Data](https://raw.githubusercontent.com/vaibhavv-labs/sentimentiq-dashboard/main/assets/live-data.png)


## ⚙️ Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/vaibhavv-labs/sentimentiq-dashboard.git
cd sentimentiq-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add API Key
Create `.streamlit/secrets.toml`:
```toml
YOUTUBE_API_KEY = "AIzaSyCvuQijtSSyrL7JLyki8Qq475pp7301m1g"
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 🔍 How It Works

1. **Input** → Text is cleaned and validated (gibberish detection included)
2. **Tokenization** → DistilBERT tokenizer converts text to tokens
3. **Inference** → Fine-tuned model predicts sentiment probabilities
4. **Confidence Check** → Below 65% confidence = Neutral, above = Positive / Negative
5. **Visualization** → Results displayed with charts, metrics, and emoji indicators

---

## 📈 Model Training Details

- **Dataset:** Twitter Sentiment 140 (1.6M tweets, sampled 15,000)
- **Preprocessing:** Lowercasing, URL removal, @mention removal, hashtag cleaning
- **Fine-tuning:** 3 epochs, learning rate 2e-5, batch size 16
- **Evaluation:** Accuracy 82.0%, F1 Score 0.82
- **Training Platform:** Google Colab (T4 GPU)
- **Training Time:** ~20 minutes on GPU

---

## 🙋 Author

**Vaibhav Bhoyate**
- GitHub: [@vaibhavv-labs](https://github.com/vaibhavv-labs)
- LinkedIn: [Vaibhav Bhoyate](https://www.linkedin.com/in/vaibhav-bhoyate-6328802a9/)

---

## 📄 License

This project is licensed under the MIT License.
