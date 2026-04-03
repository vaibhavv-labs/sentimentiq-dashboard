import streamlit as st
import torch
import torch.nn.functional as F
import pandas as pd
import plotly.graph_objects as go
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datetime import datetime
from googleapiclient.discovery import build
import re
import time

st.set_page_config(
    page_title="SentimentIQ • Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
.footer {
    position: fixed;
    bottom: 20px;
    right: 20px;
    font-size: 13px;
    color: rgba(255,255,255,0.7);
    background: rgba(0,0,0,0.4);
    padding: 6px 10px;
    border-radius: 8px;
    backdrop-filter: blur(5px);
    z-index: 100;
}
</style>
<div class="footer">
👨‍💻 Built by <b>Vaibhav Bhoyate</b> | 🚀 Streamlit
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:     #070b14;
    --bg2:    #0d1526;
    --bg3:    #111e35;
    --border: rgba(255,255,255,0.07);
    --green:  #05f0a0;
    --red:    #ff3f5b;
    --yellow: #ffc542;
    --blue:   #3d9eff;
    --muted:  #4a5e7a;
    --text:   #e8edf5;
}

html, body, .stApp { background: var(--bg) !important; }
.main .block-container { padding: 1.5rem 2rem 3rem; max-width: 1380px; }
* { font-family: 'DM Sans', sans-serif; box-sizing: border-box; }
h1,h2,h3 { font-family: 'Syne', sans-serif; }

#MainMenu, footer { visibility: hidden; }
header { background: transparent !important; }

section[data-testid="stSidebar"] {
    background: #08101f !important;
    border-right: 1px solid rgba(255,255,255,0.07);
}
.stRadio > div { display: flex !important; flex-direction: column !important; }
.stRadio label {
    display: block !important; visibility: visible !important;
    opacity: 1 !important; margin-bottom: 8px; padding: 10px 12px;
    background: #111e35; border-radius: 8px; cursor: pointer;
}
.stRadio label span { color: white !important; font-size: 14px !important; }
[data-testid="stSidebar"] * { color: var(--text) !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--bg3); border-radius: 4px; }

.hero {
    background: linear-gradient(135deg, #0d1526 0%, #0a1220 60%, #0d1e30 100%);
    border: 1px solid var(--border); border-radius: 20px;
    padding: 32px 40px; margin-bottom: 28px;
    position: relative; overflow: hidden;
}
.hero::after {
    content: ''; position: absolute; inset: 0;
    background:
        radial-gradient(ellipse 600px 300px at 110% 50%, rgba(5,240,160,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 400px 400px at -10% 50%, rgba(61,158,255,0.05) 0%, transparent 60%);
    pointer-events: none;
}
.hero-title {
    font-family: 'Syne', sans-serif; font-size: 2.1rem; font-weight: 800;
    color: var(--text); margin: 0 0 6px; letter-spacing: -1px;
}
.hero-title span { color: var(--green); }
.hero-sub { color: var(--muted); font-size: 0.9rem; margin: 0; }
.pill {
    display: inline-flex; align-items: center; gap: 7px;
    background: rgba(5,240,160,0.08); border: 1px solid rgba(5,240,160,0.25);
    color: var(--green); padding: 5px 14px; border-radius: 30px;
    font-size: 0.71rem; font-weight: 600; letter-spacing: 1px;
    font-family: 'DM Mono', monospace;
}
.pulse {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--green); animation: blink 1.4s ease-in-out infinite;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }

.metric-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 24px; }
.mcard {
    background: var(--bg2); border: 1px solid var(--border);
    border-radius: 16px; padding: 22px 22px 18px;
    position: relative; overflow: hidden;
    transition: border-color .25s, transform .25s;
}
.mcard:hover { border-color: rgba(255,255,255,0.14); transform: translateY(-2px); }
.mcard-bar { position: absolute; top:0; left:0; right:0; height:3px; border-radius: 16px 16px 0 0; }
.mcard-icon { font-size: 1.3rem; opacity:.45; position:absolute; top:18px; right:18px; }
.mcard-label { font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px; color: var(--muted); margin-bottom: 10px; }
.mcard-val { font-family: 'DM Mono', monospace; font-size: 2.3rem; font-weight: 500; color: var(--text); line-height: 1; margin-bottom: 6px; }
.mcard-sub { font-size: 0.76rem; color: var(--muted); }

.cbox { background: var(--bg2); border: 1px solid var(--border); border-radius: 16px; padding: 22px 22px 14px; margin-bottom: 16px; }
.cbox-label {
    font-size: 0.68rem; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1.2px; color: var(--muted); margin-bottom: 16px;
    display: flex; align-items: center; gap: 8px;
}
.cbox-label::before { content:''; display:inline-block; width:7px; height:7px; border-radius:2px; background: var(--green); }

.ibox { background: var(--bg2); border: 1px solid var(--border); border-radius: 16px; padding: 26px; margin-bottom: 16px; }
.ibox-title { font-family: 'Syne', sans-serif; font-size: 1.05rem; font-weight: 700; color: var(--text); margin-bottom: 14px; }

.res-wrap { text-align: center; padding: 28px 20px; border-radius: 14px; margin-top: 18px; border: 1px solid var(--border); animation: fadeup .4s ease; }
@keyframes fadeup { from{opacity:0;transform:translateY(10px)} to{opacity:1;transform:translateY(0)} }
.res-emoji { font-size: 4.5rem; line-height: 1; margin-bottom: 10px; }
.res-label { font-family:'Syne',sans-serif; font-size:1.7rem; font-weight:700; margin-bottom:6px; }
.res-conf { font-size:0.82rem; color:var(--muted); font-family:'DM Mono',monospace; margin-bottom:14px; }
.res-bar-bg { height:5px; background:rgba(255,255,255,0.07); border-radius:5px; max-width:220px; margin:0 auto; }
.res-bar { height:5px; border-radius:5px; }

.err-wrap { text-align: center; padding: 28px; background: rgba(255,63,91,0.05); border: 1px solid rgba(255,63,91,0.18); border-radius: 14px; margin-top: 18px; animation: fadeup .4s ease; }
.err-emoji { font-size: 3.5rem; margin-bottom: 10px; }
.err-msg { color: #ff7a8a; font-weight: 600; font-size: 1rem; margin-bottom: 4px; }
.err-hint { color: var(--muted); font-size: 0.82rem; }

.stButton > button {
    background: linear-gradient(135deg, #05f0a0, #00c07a) !important;
    color: #050f1a !important; font-weight: 700 !important;
    border: none !important; border-radius: 10px !important;
    padding: 11px 28px !important; font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important; width: 100% !important;
    transition: opacity .2s, transform .2s !important;
}
.stButton > button:hover { opacity:.88 !important; transform: translateY(-1px) !important; }

.stTextArea textarea, .stTextInput input {
    background: #0a1220 !important; border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important; color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important; font-size: 0.95rem !important;
}
.stTextArea textarea { resize: none !important; }
.stTextArea textarea:focus, .stTextInput input:focus {
    border-color: rgba(5,240,160,0.4) !important;
    box-shadow: 0 0 0 3px rgba(5,240,160,0.06) !important;
}

[data-testid="stFileUploader"] {
    background: var(--bg2) !important;
    border: 1.5px dashed rgba(255,255,255,0.1) !important;
    border-radius: 14px !important; padding: 8px !important;
}
[data-testid="stDownloadButton"] button {
    background: var(--bg3) !important; color: var(--text) !important;
    border: 1px solid var(--border) !important; border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important;
}

.ex-card { background: var(--bg2); border: 1px solid var(--border); border-radius: 16px; padding: 22px; height: 100%; }
.ex-title { font-size: 0.68rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1.2px; color: var(--muted); margin-bottom: 14px; }
.ex-row { display: flex; align-items: flex-start; gap: 10px; padding: 10px 12px; border-radius: 10px; background: rgba(255,255,255,0.03); margin-bottom: 8px; font-size: 0.84rem; color: var(--text); line-height: 1.45; }
.ex-dot { width:8px; height:8px; border-radius:50%; margin-top:4px; flex-shrink:0; }

.info-card { background: #09121f; border: 1px solid var(--border); border-radius: 12px; padding: 16px; margin-bottom: 12px; }
.info-row { margin-bottom: 12px; }
.info-row:last-child { margin-bottom: 0; }
.info-key { font-size: 0.68rem; font-weight:600; text-transform:uppercase; letter-spacing:1px; color:var(--muted); margin-bottom:2px; }
.info-val { font-size: 0.9rem; color: var(--text); font-weight: 500; }
.info-val.green { color: var(--green); font-family:'DM Mono',monospace; font-size:1.1rem; }

.divider { height:1px; background:var(--border); margin: 18px 0; }

.stTabs [data-baseweb="tab-list"] { background: var(--bg3) !important; border-radius: 12px !important; gap: 4px !important; padding: 4px !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; border-radius: 10px !important; color: var(--muted) !important; font-weight: 600 !important; }
.stTabs [aria-selected="true"] { background: var(--bg2) !important; color: var(--text) !important; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 20px !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD MODEL — vaibhav9700 (CORRECT USERNAME)
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    mdl = AutoModelForSequenceClassification.from_pretrained("vaibhav9700/sentimentiq-distilbert")
    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    return mdl, tok

model, tokenizer = load_model()


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
COLOR = {"green": "#05f0a0", "red": "#ff3f5b", "yellow": "#ffc542", "blue": "#3d9eff"}

PBASE = dict(
    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#4a5e7a', family='DM Sans'),
    margin=dict(l=0, r=0, t=10, b=0)
)

def is_valid_text(text):
    t = text.strip()
    if len(t) < 5:
        return False, "Text is too short!", "Write at least a few meaningful words."
    if re.match(r'^[\d\s\W]+$', t):
        return False, "Only numbers or symbols detected!", "Please enter actual words."
    words = re.findall(r'[a-zA-Z]{3,}', t)
    if len(words) < 2:
        return False, "Not enough words to analyze!", "Try writing a proper sentence."
    if re.findall(r'[^aeiouAEIOU\s]{6,}', t):
        return False, "Gibberish detected!", "Please enter a meaningful English sentence."
    return True, "", ""

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    pred  = torch.argmax(probs).item()
    conf  = probs[0][pred].item() * 100
    if conf < 65:
        return "Neutral",  conf, "😐", COLOR["yellow"], "rgba(255,197,66,0.10)", "rgba(255,197,66,0.28)"
    elif pred == 1:
        return "Positive", conf, "😊", COLOR["green"],  "rgba(5,240,160,0.08)",  "rgba(5,240,160,0.28)"
    else:
        return "Negative", conf, "😡", COLOR["red"],    "rgba(255,63,91,0.08)",  "rgba(255,63,91,0.28)"

def sentiment_charts(pos, neu, neg, tot):
    c1, c2 = st.columns([1, 1.4], gap="medium")
    with c1:
        st.markdown("<div class='cbox'><div class='cbox-label'>Sentiment Split</div>", unsafe_allow_html=True)
        fig = go.Figure(go.Pie(
            labels=["Positive","Neutral","Negative"], values=[pos, neu, neg], hole=0.60,
            marker=dict(colors=["#05f0a0","#ffc542","#ff3f5b"], line=dict(color="#070b14", width=3)),
            textinfo="percent", textfont=dict(size=12, color="white"),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>"
        ))
        fig.update_layout(**PBASE, height=240, showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.22,
                        font=dict(color="#e8edf5", size=11), bgcolor="rgba(0,0,0,0)"),
            annotations=[dict(text=f"<b>{tot}</b><br>texts", x=0.5, y=0.5,
                              showarrow=False, font=dict(size=13, color="#e8edf5"))])
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='cbox'><div class='cbox-label'>Count by Sentiment</div>", unsafe_allow_html=True)
        fig2 = go.Figure(go.Bar(
            x=["Positive","Neutral","Negative"], y=[pos, neu, neg],
            marker=dict(color=["#05f0a0","#ffc542","#ff3f5b"], line=dict(color="rgba(0,0,0,0)", width=0)),
            text=[pos, neu, neg], textposition="outside",
            textfont=dict(color="#e8edf5", size=13),
            hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
        ))
        fig2.update_layout(**PBASE, height=240, bargap=0.38,
            xaxis=dict(showgrid=False, color="#4a5e7a", tickfont=dict(size=12)),
            yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#4a5e7a", showline=False))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

def metric_cards_3(pos, neu, neg):
    st.markdown("<div class='metric-grid' style='grid-template-columns:repeat(3,1fr);'>", unsafe_allow_html=True)
    for lbl, val, grad, icon in [
        ("Positive", pos, "#05f0a0,#00a86b", "😊"),
        ("Neutral",  neu, "#ffc542,#e6a800", "😐"),
        ("Negative", neg, "#ff3f5b,#cc0020", "😡"),
    ]:
        st.markdown(f"""
        <div class='mcard'>
            <div class='mcard-bar' style='background:linear-gradient(90deg,{grad});'></div>
            <div class='mcard-icon'>{icon}</div>
            <div class='mcard-label'>{lbl}</div>
            <div class='mcard-val'>{val}</div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:4px 0 18px;'>
        <div style='font-family:Syne,sans-serif;font-size:1.2rem;font-weight:800;color:#e8edf5;letter-spacing:-0.5px;'>🧠 SentimentIQ</div>
        <div style='font-size:0.73rem;color:#4a5e7a;margin-top:2px;'>Powered by DistilBERT</div>
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio("Select Mode", [
        "📝 Single Text Analyzer",
        "📂 Bulk CSV Analysis",
        "📡 Live Data"
    ])

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.68rem;font-weight:600;text-transform:uppercase;letter-spacing:1.2px;color:#4a5e7a;margin-bottom:12px;'>Model Info</div>
    <div class='info-card'>
        <div class='info-row'><div class='info-key'>Architecture</div><div class='info-val'>DistilBERT-base-uncased</div></div>
        <div class='info-row'><div class='info-key'>Dataset</div><div class='info-val'>Twitter Sentiment 140</div></div>
        <div class='info-row'><div class='info-key'>Training Samples</div><div class='info-val'>15,000 tweets</div></div>
        <div class='info-row'><div class='info-key'>Validation Accuracy</div><div class='info-val green'>82.0%</div></div>
        <div class='info-row'><div class='info-key'>Labels</div><div class='info-val'>Positive · Neutral · Negative</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-size:0.71rem;color:#4a5e7a;text-align:center;line-height:1.9;'>
        Fine-tuned DistilBERT · Streamlit UI<br>
        <span style='color:#3d9eff;'>Sentiment Analysis Project v2.0</span>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HERO
# ─────────────────────────────────────────────
st.markdown("""
<div class='hero'>
    <div style='display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:14px;'>
        <div>
            <div class='hero-title'>Social Media <span>Sentiment</span> Dashboard</div>
            <div class='hero-sub'>Real-time sentiment analysis · DistilBERT transformer · Twitter dataset</div>
        </div>
        <div class='pill'><div class='pulse'></div>LIVE INFERENCE</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  MODE A — SINGLE TEXT
# ═══════════════════════════════════════════
if "Single" in mode:
    col_l, col_r = st.columns([1.6, 1], gap="large")
    with col_l:
        st.markdown("<div class='ibox'>", unsafe_allow_html=True)
        st.markdown("<div class='ibox-title'>Analyze a Text</div>", unsafe_allow_html=True)
        user_text = st.text_area("input", label_visibility="collapsed",
            placeholder='"This product completely exceeded my expectations!"', height=120)
        if st.button("⚡  Run Sentiment Analysis"):
            if not user_text.strip():
                st.warning("Please enter some text first.")
            else:
                valid, err_title, err_hint = is_valid_text(user_text)
                if not valid:
                    st.markdown(f"""
                    <div class='err-wrap'>
                        <div class='err-emoji'>🤷</div>
                        <div class='err-msg'>{err_title}</div>
                        <div class='err-hint'>{err_hint}</div>
                    </div>""", unsafe_allow_html=True)
                else:
                    with st.spinner("Analyzing..."):
                        time.sleep(0.25)
                        label, conf, emoji, color, bg, border_c = predict(user_text)
                    st.markdown(f"""
                    <div class='res-wrap' style='background:{bg};border-color:{border_c};'>
                        <div class='res-emoji'>{emoji}</div>
                        <div class='res-label' style='color:{color};'>{label}</div>
                        <div class='res-conf'>Confidence: {conf:.1f}%</div>
                        <div class='res-bar-bg'>
                            <div class='res-bar' style='width:{int(conf)}%;background:{color};'></div>
                        </div>
                    </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with col_r:
        st.markdown("""
        <div class='ex-card'>
            <div class='ex-title'>Try These Examples</div>
            <div class='ex-row'><div class='ex-dot' style='background:#05f0a0;'></div>"This product completely exceeded my expectations!"</div>
            <div class='ex-row'><div class='ex-dot' style='background:#05f0a0;'></div>"Amazing customer service, I highly recommend it!"</div>
            <div class='ex-row'><div class='ex-dot' style='background:#ffc542;'></div>"Just received my order, will update later."</div>
            <div class='ex-row'><div class='ex-dot' style='background:#ffc542;'></div>"I booked two tickets for the event."</div>
            <div class='ex-row'><div class='ex-dot' style='background:#ff3f5b;'></div>"Terrible experience, complete waste of money!"</div>
            <div class='ex-row'><div class='ex-dot' style='background:#ff3f5b;'></div>"Worst customer service I have ever seen."</div>
            <div style='margin-top:16px;padding-top:14px;border-top:1px solid rgba(255,255,255,0.06);'>
                <div class='ex-title' style='margin-bottom:8px;'>Legend</div>
                <div style='font-size:0.81rem;color:#4a5e7a;line-height:2;'>
                    🟢 Green → Positive &nbsp;·&nbsp; 🟡 Yellow → Neutral &nbsp;·&nbsp; 🔴 Red → Negative
                </div>
            </div>
        </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  MODE B — BULK CSV
# ═══════════════════════════════════════════
elif "Bulk" in mode:
    st.markdown("""
    <div class='cbox'>
        <div style='font-family:Syne,sans-serif;font-size:1.05rem;font-weight:700;color:#e8edf5;margin-bottom:10px;'>Upload CSV File</div>
        <div style='font-size:0.82rem;color:#4a5e7a;margin-bottom:14px;'>
            CSV must have a column named <code style='background:#111e35;padding:2px 7px;border-radius:4px;color:#05f0a0;font-family:DM Mono,monospace;'>text</code>
        </div>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        if "text" not in df.columns:
            st.error("❌  Your CSV must have a column named 'text'.")
        else:
            prog = st.progress(0, text="Running analysis…")
            total_rows = len(df)
            results = []
            for i, txt in enumerate(df["text"].astype(str)):
                valid, _, _ = is_valid_text(txt)
                if valid:
                    label, conf, emoji, color, _, _ = predict(txt)
                else:
                    label, conf, emoji = "Invalid", 0.0, "⚠️"
                results.append((label, round(conf, 1), emoji))
                step = max(1, total_rows // 25)
                if i % step == 0:
                    prog.progress((i + 1) / total_rows, text=f"Analyzing {i+1} / {total_rows} rows…")
            prog.empty()

            df["sentiment"]  = [r[0] for r in results]
            df["confidence"] = [r[1] for r in results]
            df["emoji"]      = [r[2] for r in results]

            vdf = df[df["sentiment"] != "Invalid"]
            pos = (vdf["sentiment"] == "Positive").sum()
            neu = (vdf["sentiment"] == "Neutral").sum()
            neg = (vdf["sentiment"] == "Negative").sum()
            tot = len(vdf)

            if tot > 0:
                if pos > neg and pos > neu:   dominant, dom_icon, dom_color = "Positive", "😊", "#05f0a0,#00a86b"
                elif neg > pos and neg > neu: dominant, dom_icon, dom_color = "Negative", "😡", "#ff3f5b,#cc0020"
                elif neu > pos and neu > neg: dominant, dom_icon, dom_color = "Neutral",  "😐", "#ffc542,#e6a800"
                else:                         dominant, dom_icon, dom_color = "Mixed",    "⚖️", "#3d9eff,#0066cc"
            else:
                dominant, dom_icon, dom_color = "N/A", "➖", "#4a5e7a,#2d3748"

            st.markdown("<div class='metric-grid'>", unsafe_allow_html=True)
            for lbl, val, sub, grad, icon in [
                ("Total Records", f"{tot:,}",  f"{len(df)-tot} skipped",                     "#3d9eff,#0066cc", "📝"),
                ("% Positive",    f"{pos/tot*100:.1f}%" if tot else "0%", f"{pos:,} texts",  "#05f0a0,#00a86b", "😊"),
                ("% Negative",    f"{neg/tot*100:.1f}%" if tot else "0%", f"{neg:,} texts",  "#ff3f5b,#cc0020", "😡"),
                ("Dominant",      dominant, "Overall sentiment",                               dom_color, dom_icon),
            ]:
                st.markdown(f"""
                <div class='mcard'>
                    <div class='mcard-bar' style='background:linear-gradient(90deg,{grad});'></div>
                    <div class='mcard-icon'>{icon}</div>
                    <div class='mcard-label'>{lbl}</div>
                    <div class='mcard-val'>{val}</div>
                    <div class='mcard-sub'>{sub}</div>
                </div>""", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            c1, c2 = st.columns([1, 1], gap="medium")
            with c1:
                st.markdown("<div class='cbox'><div class='cbox-label'>Sentiment Split</div>", unsafe_allow_html=True)
                fig = go.Figure(go.Pie(
                    labels=["Positive","Neutral","Negative"], values=[pos, neu, neg], hole=0.62,
                    marker=dict(colors=["#05f0a0","#ffc542","#ff3f5b"], line=dict(color="#070b14", width=3)),
                    textinfo="percent", textfont=dict(size=12, color="white")))
                fig.update_layout(**PBASE, height=240, showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.22,
                                font=dict(color="#e8edf5", size=11), bgcolor="rgba(0,0,0,0)"),
                    annotations=[dict(text=f"<b>{tot}</b><br>texts", x=0.5, y=0.5,
                                      showarrow=False, font=dict(size=13, color="#e8edf5"))])
                st.plotly_chart(fig, use_container_width=True, config={"displayModeBar":False})
                st.markdown("</div>", unsafe_allow_html=True)
            with c2:
                st.markdown("<div class='cbox'><div class='cbox-label'>Count by Sentiment</div>", unsafe_allow_html=True)
                fig2 = go.Figure(go.Bar(
                    x=["Positive","Neutral","Negative"], y=[pos, neu, neg],
                    marker=dict(color=["#05f0a0","#ffc542","#ff3f5b"]),
                    text=[pos, neu, neg], textposition="outside",
                    textfont=dict(color="#e8edf5", size=13)))
                fig2.update_layout(**PBASE, height=240, bargap=0.38,
                    xaxis=dict(showgrid=False, color="#4a5e7a"),
                    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", color="#4a5e7a"))
                st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar":False})
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='cbox'><div class='cbox-label'>Analyzed Data · Top 10 rows</div>", unsafe_allow_html=True)
            disp = df[["text","emoji","sentiment","confidence"]].head(10).copy()
            disp.columns = ["Text","","Sentiment","Confidence (%)"]
            st.dataframe(disp, use_container_width=True, hide_index=True,
                column_config={
                    "Text": st.column_config.TextColumn(width="large"),
                    "": st.column_config.TextColumn(width="small"),
                    "Sentiment": st.column_config.TextColumn(width="small"),
                    "Confidence (%)": st.column_config.NumberColumn(width="small", format="%.1f %%")
                })
            csv_out = df[["text","sentiment","confidence"]].to_csv(index=False).encode("utf-8")
            st.download_button("⬇️  Download Full Results as CSV", data=csv_out,
                file_name=f"sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv")
            st.markdown("</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════
#  MODE C — LIVE DATA (YouTube)
# ═══════════════════════════════════════════
elif "Live" in mode:
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:1.3rem;font-weight:700;color:#e8edf5;margin-bottom:20px;'>
        📡 Live YouTube Comment Analysis
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='cbox'>", unsafe_allow_html=True)
    st.markdown("""
    <div class='ibox-title'>Paste a YouTube Video URL</div>
    <div style='font-size:0.82rem;color:#4a5e7a;margin-bottom:14px;'>
        Fetches real comments and analyzes sentiment instantly
    </div>
    """, unsafe_allow_html=True)

    yt_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=xxxxxxx", label_visibility="collapsed")
    comment_limit = st.slider("Number of comments to fetch", 20, 100, 50)

    if st.button("▶️  Fetch & Analyze Comments"):
        if not yt_url.strip():
            st.warning("Please paste a YouTube video URL.")
        else:
            match = re.search(r"(?:v=|youtu\.be/)([a-zA-Z0-9_-]{11})", yt_url)
            if not match:
                st.error("❌  Invalid YouTube URL.")
            else:
                video_id = match.group(1)
                try:
                    with st.spinner("Fetching comments from YouTube..."):
                        youtube = build("youtube", "v3", developerKey=st.secrets["YOUTUBE_API_KEY"])
                        response = youtube.commentThreads().list(
                            part="snippet", videoId=video_id,
                            maxResults=comment_limit, textFormat="plainText"
                        ).execute()

                    comments = []
                    for item in response.get("items", []):
                        snippet = item["snippet"]["topLevelComment"]["snippet"]
                        comments.append({"comment": snippet["textDisplay"], "likes": snippet["likeCount"]})

                    if not comments:
                        st.warning("No comments found for this video.")
                    else:
                        with st.spinner("Analyzing sentiment..."):
                            yt_df = pd.DataFrame(comments)
                            results = []
                            for c in yt_df["comment"]:
                                valid, _, _ = is_valid_text(str(c))
                                if valid:
                                    label, conf, emoji, color, bg, bc = predict(str(c))
                                else:
                                    label, conf, emoji = "Neutral", 60.0, "😐"
                                results.append((label, round(conf, 1), emoji))

                        yt_df["sentiment"]  = [r[0] for r in results]
                        yt_df["confidence"] = [r[1] for r in results]
                        yt_df["emoji"]      = [r[2] for r in results]

                        pos = (yt_df["sentiment"] == "Positive").sum()
                        neu = (yt_df["sentiment"] == "Neutral").sum()
                        neg = (yt_df["sentiment"] == "Negative").sum()
                        tot = len(yt_df)

                        st.markdown("<br>", unsafe_allow_html=True)
                        metric_cards_3(pos, neu, neg)
                        sentiment_charts(pos, neu, neg, tot)

                        st.markdown("<div class='cbox'><div class='cbox-label'>Comments with Sentiment</div>", unsafe_allow_html=True)
                        disp = yt_df[["emoji","comment","sentiment","confidence","likes"]].copy()
                        disp = disp.sort_values("likes", ascending=False)
                        disp.columns = ["","Comment","Sentiment","Confidence (%)","👍 Likes"]
                        st.dataframe(disp, use_container_width=True, hide_index=True,
                            column_config={
                                "Comment": st.column_config.TextColumn(width="large"),
                                "Confidence (%)": st.column_config.NumberColumn(format="%.1f %%"),
                                "👍 Likes": st.column_config.NumberColumn(width="small")
                            })
                        csv_out = yt_df[["comment","sentiment","confidence","likes"]].to_csv(index=False).encode("utf-8")
                        st.download_button("⬇️  Download Results as CSV", data=csv_out,
                            file_name=f"youtube_sentiment_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv")
                        st.markdown("</div>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"❌  Error: {e}")
                    st.info("Make sure your YOUTUBE_API_KEY is correct in Streamlit Cloud secrets.")

    st.markdown("</div>", unsafe_allow_html=True)
