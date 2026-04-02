# SentimentIQ – NLP-Based Social Media Sentiment Analysis Dashboard

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![HuggingFace](https://img.shields.io/badge/HuggingFace-DistilBERT-yellow?style=flat-square&logo=huggingface)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

> **Analyze the sentiment of any text, bulk CSV data, or live YouTube comments — powered by a fine-tuned DistilBERT transformer model.**

---

## 🚀 Live Demo
🔗 [Click here to try SentimentIQ](YOUR_STREAMLIT_LINK_HERE)

---

## 📌 Features

- **📝 Single Text Analyzer** — Type any sentence and get instant sentiment with confidence score
- **📂 Bulk CSV Analysis** — Upload a CSV file and analyze thousands of rows at once
- **📡 Live YouTube Comments** — Paste any YouTube URL and analyze real comments live
- **😊 3-Class Output** — Positive, Neutral, Negative with emoji indicators
- **🚫 Gibberish Detection** — Rejects random/invalid inputs intelligently
- **📊 Interactive Charts** — Pie chart, bar chart, confidence histogram
- **⬇️ Download Results** — Export analyzed data as CSV

---

## 🧠 Model Details

| Property | Details |
|----------|---------|
| Base Model | `distilbert-base-uncased` |
| Task | Binary Sentiment Classification |
| Dataset | Twitter Sentiment 140 |
| Training Samples | 15,000 tweets |
| Validation Accuracy | **82.0%** |
| Framework | HuggingFace Transformers |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.10 | Core language |
| HuggingFace Transformers | DistilBERT model |
| PyTorch | Deep learning backend |
| Streamlit | Web dashboard UI |
| Plotly | Interactive charts |
| YouTube Data API v3 | Live comment fetching |
| Pandas | Data processing |

---

## 📁 Project Structure
```
sentimentiq-dashboard/
│
├── app.py                  # Main Streamlit dashboard
├── requirements.txt        # Python dependencies
├── final_model/            # Fine-tuned DistilBERT model
│   ├── config.json
│   └── model.safetensors
├── .streamlit/
│   └── secrets.toml        # API keys (not uploaded to GitHub)
└── README.md               # Project documentation
```

---

## ⚙️ Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/sentimentiq-dashboard.git
cd sentimentiq-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add API Key
Create `.streamlit/secrets.toml`:
```toml
YOUTUBE_API_KEY = "your_youtube_api_key_here"
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 📊 Screenshots

> *(Add screenshots of your dashboard here)*

---

## 🔍 How It Works

1. **Text Input** → Cleaned and validated
2. **Tokenization** → DistilBERT tokenizer processes text
3. **Inference** → Fine-tuned model predicts sentiment
4. **Confidence Check** → Below 65% = Neutral, above = Positive/Negative
5. **Visualization** → Results shown with charts and metrics

---

## 📈 Model Training

- Dataset: Twitter Sentiment 140 (1.6M tweets, sampled 15K)
- Preprocessing: Lowercasing, URL removal, mention removal
- Fine-tuning: 3 epochs, learning rate 2e-5, batch size 16
- Evaluation: Accuracy 82%, F1 Score 0.82

---

## 🙋 Author

**Your Name**
- GitHub: [@your_username](https://github.com/your_username)
- LinkedIn: [your_linkedin](https://linkedin.com/in/your_linkedin)

---

## 📄 License

This project is licensed under the MIT License.
