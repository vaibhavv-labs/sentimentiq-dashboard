<div align="center">

<img src="https://readme-typing-svg.demolab.com?font=Space+Mono&size=30&pause=1000&color=FFD21E&center=true&vCenter=true&width=700&lines=SentimentIQ+%F0%9F%A7%A0;SaaS+Sentiment+Analysis+Platform;RoBERTa+Model+%C2%B7+Multi-Tenant;FastAPI+%C2%B7+Secure+Auth" alt="Typing SVG" />

<br/>

![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=for-the-badge&logo=python&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-RoBERTa-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![SQLite](https://img.shields.io/badge/SQLite-003B57?style=for-the-badge&logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-00FF88?style=for-the-badge)

<br/>

> 🧠 **A fully-featured SaaS platform for analyzing the sentiment of any text. Powered by a RoBERTa transformer model, FastAPI, and JWT authentication.**

<br/>

![Header](https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6&height=120&section=header&text=SentimentIQ&fontSize=32&fontColor=ffffff&animation=fadeIn&desc=NLP-Based+SaaS+Sentiment+Platform&descSize=15&descAlignY=78)

</div>

---

## ✨ Features

| Feature | Description |
|---|---|
| 🏢 **Multi-Tenant Architecture** | Create isolated workspaces for different organizations |
| 🔐 **Authentication** | Secure Login & Register flows with JWT & Bcrypt |
| ⚡ **FastAPI Backend** | High performance REST API with Bearer token auth |
| 📝 **Text Analyzer** | Type any sentence and get instant sentiment with confidence score |
| 😊 **3-Class Output** | Positive, Neutral, Negative with emoji indicators |
| 🎛️ **Confidence Thresholding** | Dynamic fallback to Neutral if model confidence is < 90% |
| 📂 **CSV Aliasing** | Robust CSV import with flexible column aliases |

---

## 🧠 Model Details

| Property | Details |
|---|---|
| Base Model | `cardiffnlp/twitter-roberta-base-sentiment` |
| Task | Sentiment Classification (3-class) |
| Architecture | RoBERTa (Robustly Optimized BERT Pretraining Approach) |
| Framework | HuggingFace Transformers + PyTorch |
| Logic | Returns Neutral if confidence is below 90% |

---

## 🔍 How It Works

```text
  📥 User Input via Dashboard UI
           │
           ▼
  🔐 Auth Middleware (JWT Token Validation)
           │
           ▼
  🔤 RoBERTa Tokenizer
           │
           ▼
  🧠 RoBERTa Model Prediction
           │
           ├── Confidence ≥ 90% ──▶ 😊 Positive / 😤 Negative
           │
           └── Confidence < 90% ──▶ 😐 Neutral
```

---

## 📊 Screenshots (Placeholder)

### 🖥️ Login / Register UI
![Login Register Placeholder](https://via.placeholder.com/800x400?text=Login+/+Register+UI+Screenshot+Here)

### 🏢 SaaS Dashboard
![Dashboard Placeholder](https://via.placeholder.com/800x400?text=SaaS+Dashboard+UI+Screenshot+Here)

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python 3.10 | Core language |
| FastAPI & Uvicorn | High-performance REST API |
| SQLAlchemy & SQLite | Database & ORM |
| JWT & Passlib | Secure authentication |
| Jinja2 | HTML Template rendering |
| HuggingFace Transformers | RoBERTa inference |
| PyTorch | Deep learning backend |

---

## 📁 Project Structure

```text
sentimentiq-dashboard/
│
├── backend.py              # ⚡  Main FastAPI application & logic
├── saas.db                 # 🗄️  SQLite database (Users & Tenants)
├── requirements.txt        # 📦  Python dependencies
├── templates/
│   ├── base.html           # 🏗️  Base HTML layout
│   ├── login.html          # 🔐  Login page
│   ├── register.html       # 📝  Registration page
│   └── dashboard.html      # 📊  SaaS Dashboard page
└── README.md               # 📄  Project documentation
```

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

### 3. Run the FastAPI Server
```bash
python -m uvicorn backend:app --port 8000 --reload
```

### 4. Open the App
Visit [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser to access the Login/Register flow and the Sentiment Analysis Dashboard.

---

## 🤝 Contributors

**Vaibhav Bhoyate** (Creator/Author)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/vaibhav-bhoyate-6328802a9/)
[![GitHub](https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/vaibhavv-labs)

**Special Thanks to:**
- **kriptoburak** for improving the CSV column alias robustness and adding the MIT License.

---

## 📄 License

This project is licensed under the **MIT License.**
