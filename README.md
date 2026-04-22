<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Space+Mono&size=30&pause=1000&color=FFD21E&center=true&vCenter=true&width=700&lines=SentimentIQ+%F0%9F%A7%A0;NLP-Powered+Sentiment+Analysis;Fine-tuned+DistilBERT+%C2%B7+82%25+Accuracy;Analyze+Text+%7C+CSV+%7C+YouTube+Live" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-DistilBERT-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-00FF88?style=for-the-badge)

<br/>

> 🧠 **Analyze the sentiment of any text, bulk CSV data, or live YouTube comments — powered by a fine-tuned DistilBERT transformer model.**

<br/>

### 🔗 [Try SentimentIQ Live →](https://sentimentiq-dashboard-njiprgwlcchwkuemvrqwn4.streamlit.app/)

<br/>

![Header](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6&height=120&section=header&text=SentimentIQ&fontSize=32&fontColor=ffffff&animation=fadeIn&desc=NLP-Based+Social+Media+Sentiment+Analysis&descSize=15&descAlignY=78)

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 📝 **Single Text Analyzer** | Type any sentence and get instant sentiment with confidence score |
| 📂 **Bulk CSV Analysis** | Upload a CSV file and analyze thousands of rows at once |
| 📡 **Live YouTube Comments** | Paste any YouTube URL and analyze real comments live |
| 😊 **3-Class Output** | Positive, Neutral, Negative with emoji indicators |
| 🚫 **Gibberish Detection** | Rejects random/invalid inputs intelligently |
| 📊 **Interactive Charts** | Pie chart, bar chart, confidence distribution |
| ⬇️ **Download Results** | Export analyzed data as CSV |

---

## 🧠 Model Details

| Property | Details |
|---|---|
| Base Model | `distilbert-base-uncased` |
| Task | Sentiment Classification (3-class) |
| Dataset | Twitter Sentiment 140 |
| Training Samples | 15,000 tweets |
| Validation Accuracy | **82.0%** |
| F1 Score | **0.82** |
| Framework | HuggingFace Transformers + PyTorch |
| Hosted On | `vaibhav9700/sentimentiq-distilbert` |

---

## 🔍 How It Works

```
  📥 Input (Text / CSV / YouTube URL)
           │
           ▼
  🧹 Clean & Validate
  (lowercasing, URL removal, gibberish check)
           │
           ▼
  🔤 DistilBERT Tokenizer
  (text → token IDs)
           │
           ▼
  🧠 Fine-tuned DistilBERT Model
  (token IDs → sentiment probabilities)
           │
           ├── Confidence ≥ 65% ──▶ 😊 Positive / 😤 Negative
           │
           └── Confidence < 65% ──▶ 😐 Neutral
                                          │
                                          ▼
                              📊 Plotly Charts + CSV Export
```

---

## 📊 Screenshots

### 🖥️ Main Dashboard
![Dashboard](https://raw.githubusercontent.com/vaibhavv-labs/sentimentiq-dashboard/main/assets/dashboard.png)

### 📊 Analytics View
![Analytics](https://raw.githubusercontent.com/vaibhavv-labs/sentimentiq-dashboard/main/assets/analytics.png)

### 📡 Live YouTube Data
![Live Data](https://raw.githubusercontent.com/vaibhavv-labs/sentimentiq-dashboard/main/assets/live-data.png)

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
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
├── app.py                  # 🖥️  Main Streamlit dashboard
├── requirements.txt        # 📦  Python dependencies
├── .streamlit/
│   └── secrets.toml        # 🔑  API keys (not in GitHub)
└── README.md               # 📄  Project documentation

🤗 Model on HuggingFace Hub:
   vaibhav9700/sentimentiq-distilbert
```

---

## 📈 Model Training Details

- **Dataset:** Twitter Sentiment 140 (1.6M tweets → sampled 15,000)
- **Preprocessing:** Lowercasing, URL removal, @mention removal, hashtag cleaning
- **Fine-tuning:** 3 epochs · LR: 2e-5 · Batch size: 16
- **Evaluation:** Accuracy **82.0%** · F1 Score **0.82**
- **Platform:** Google Colab (T4 GPU) · ~20 minutes training time

---

## 🚀 Local Setup

### 1. Clone the repository
```bash
git clone https://github.com/vaibhavv-labs/sentimentiq-dashboard.git
cd sentimentiq-dashboard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add your YouTube API Key
```toml
# .streamlit/secrets.toml
YOUTUBE_API_KEY = "your_api_key_here"
```

### 4. Run the app
```bash
streamlit run app.py
```

---

## 🗺️ Roadmap

- [ ] 🐦 Twitter/X live feed integration
- [ ] 🌍 Multi-language sentiment support
- [ ] 📅 Sentiment trend over time graphs
- [ ] 🔔 Alert on negative sentiment spike
- [ ] 🧪 A/B comparison between models

---

## 🙋 Author

**Vaibhav Bhoyate**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/vaibhav-bhoyate-6328802a9/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/vaibhavv-labs)
[![Portfolio](https://img.shields.io/badge/Portfolio-00FF88?style=for-the-badge&logo=vercel&logoColor=black)](https://portfolio-vaibhav13.vercel.app/)

---

## 📄 License

This project is licensed under the **MIT License.**

<div align="center">

⭐ **Star this repo if it helped you!** ⭐

![Footer](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6&height=100&section=footer)

</div>
