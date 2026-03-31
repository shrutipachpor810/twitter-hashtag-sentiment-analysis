import os
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import re
from transformers import pipeline
from wordcloud import WordCloud

# ================================
# PAGE CONFIG
# ================================
st.set_page_config(page_title="Hashtag Sentiment Analyzer", layout="centered")

# ================================
# STYLES
# ================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #07080d;
    color: #e8e6e0;
    font-family: 'Syne', sans-serif;
}
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 60% 40% at 80% 10%, rgba(255,75,110,0.07) 0%, transparent 60%),
        radial-gradient(ellipse 50% 35% at 10% 80%, rgba(0,180,150,0.06) 0%, transparent 55%),
        #07080d;
}
[data-testid="stHeader"], [data-testid="stToolbar"] { background: transparent !important; }
.block-container { padding: 2.5rem 2rem 4rem; max-width: 780px; }

.hero { text-align: center; padding: 3rem 0 2.5rem; }
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.22em;
    color: #ff4b6e;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.hero h1 {
    font-size: clamp(2rem, 5vw, 3rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.03em;
    margin: 0 0 0.6rem;
    background: linear-gradient(135deg, #f5f0e8 30%, #ff4b6e 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.hero-sub { font-size: 0.9rem; color: #555; }

[data-testid="stTextInput"] input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 6px !important;
    color: #e8e6e0 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 1rem !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #ff4b6e !important;
    box-shadow: 0 0 0 3px rgba(255,75,110,0.12) !important;
}
[data-testid="stTextInput"] label {
    color: #555 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.12em !important;
}

[data-testid="stButton"] > button {
    background: #ff4b6e !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.88rem !important;
    letter-spacing: 0.08em !important;
    padding: 0.65rem 2rem !important;
    width: 100% !important;
    transition: opacity 0.15s !important;
}
[data-testid="stButton"] > button:hover { opacity: 0.82 !important; }

hr { border-color: rgba(255,255,255,0.07) !important; margin: 1.8rem 0 !important; }

.sec-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.67rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #aaa;
    margin-bottom: 0.9rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(255,255,255,0.12);
}

.stats-row { display: flex; gap: 10px; margin: 1.4rem 0; }
.stat-card {
    flex: 1;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 1rem 0.6rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.stat-card.pos::before { background: #00c896; }
.stat-card.neg::before { background: #ff4b6e; }
.stat-card.neu::before { background: #f0a500; }
.stat-card.pol::before { background: #7aa2f7; }
.stat-card.trn::before { background: #c0a6f7; }
.stat-val { font-size: 1.55rem; font-weight: 800; letter-spacing: -0.03em; line-height: 1; margin-bottom: 0.3rem; }
.stat-val.pos { color: #00c896; }
.stat-val.neg { color: #ff4b6e; }
.stat-val.neu { color: #f0a500; }
.stat-val.pol { color: #7aa2f7; }
.stat-val.trn { color: #c0a6f7; }
.stat-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #888;
}

.tweet-grid { display: flex; flex-direction: column; gap: 8px; }
.tweet-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 8px;
    padding: 0.85rem 1rem;
    font-size: 0.86rem;
    color: #b0ad a6;
    line-height: 1.55;
    border-left: 3px solid transparent;
}
.tweet-card.pos { border-left-color: #00c896; color: #c8c5be; }
.tweet-card.neg { border-left-color: #ff4b6e; color: #c8c5be; }

h2, h3 { color: #e8e6e0 !important; font-family: 'Syne', sans-serif !important; }
[data-testid="stAlert"] { border-radius: 8px !important; font-family: 'DM Mono', monospace !important; font-size: 0.82rem !important; }
</style>
""", unsafe_allow_html=True)


# ================================
# LOAD MODEL  (unchanged)
# ================================
@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="cardiffnlp/twitter-roberta-base-sentiment"
    )

model = load_model()

label_map = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# ================================
# CLEANING  (unchanged)
# ================================
def clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"[^\w\s#]", "", text)
    return text.strip()

# ================================
# SENTIMENT + POLARITY  (unchanged)
# ================================
def analyze_text(text):
    result = model(text[:512])[0]
    label = label_map[result['label']]
    score = result['score']
    if label == "Neutral":
        polarity = 0
    elif label == "Positive":
        polarity = score
    else:
        polarity = -score
    return label, polarity, score

# ================================
# LOAD DATA  (unchanged)
# ================================
@st.cache_data
def load_data():
    df = pd.read_csv("full-corpus.csv")
    text_col = None
    for col in df.columns:
        if 'text' in col.lower() or 'tweet' in col.lower():
            text_col = col
    df = df[[text_col]].dropna()
    df.columns = ['text']
    return df

df_main = load_data()

# ================================
# HERO
# ================================
st.markdown("""
<div class="hero">
    <p class="hero-eyebrow">Twitter · RoBERTa · NLP</p>
    <h1>Hashtag Sentiment<br>Analyzer</h1>
    <p class="hero-sub">Real-time sentiment, polarity &amp; trend analysis using AI</p>
</div>
""", unsafe_allow_html=True)

# ================================
# INPUT  (unchanged logic)
# ================================
hashtag = st.text_input("HASHTAG", placeholder="#AI  ·  #Python  ·  #ChatGPT")
analyze_btn = st.button("ANALYZE →")


# ================================
# CHART HELPERS  (dark-themed wrappers)
# ================================
def dark_bar(labels, values, colors, title=""):
    fig, ax = plt.subplots(figsize=(5.5, 3))
    fig.patch.set_facecolor("#0d0f18")
    ax.set_facecolor("#0d0f18")
    bars = ax.bar(labels, values, color=colors, width=0.45, edgecolor="none")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + max(values, default=1) * 0.03,
                str(v), ha='center', color='#666', fontsize=9, fontfamily='monospace')
    if title:
        ax.set_title(title, color="#555", fontsize=8, pad=8, fontfamily='monospace')
    ax.tick_params(colors="#555", labelsize=9)
    ax.spines[:].set_visible(False)
    ax.yaxis.set_visible(False)
    for lbl in ax.get_xticklabels():
        lbl.set_fontfamily('monospace')
        lbl.set_color('#888')
    plt.tight_layout()
    return fig

def dark_hist(series):
    fig, ax = plt.subplots(figsize=(5.5, 3))
    fig.patch.set_facecolor("#0d0f18")
    ax.set_facecolor("#0d0f18")
    ax.hist(series, bins=20, color="#7aa2f7", edgecolor="none", alpha=0.85)
    ax.set_xlabel("Polarity  (−1 → +1)", color="#555", fontsize=8, fontfamily='monospace')
    ax.set_ylabel("Frequency", color="#555", fontsize=8, fontfamily='monospace')
    ax.tick_params(colors="#555", labelsize=8)
    ax.spines[:].set_visible(False)
    plt.tight_layout()
    return fig

def dark_wordcloud(text_series):
    text = " ".join(text_series)
    if not text.strip():
        return None
    wc = WordCloud(
        width=900, height=360,
        background_color="#0d0f18",
        colormap="RdYlGn",
        max_words=130,
        prefer_horizontal=0.8
    ).generate(text)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    fig.patch.set_facecolor("#0d0f18")
    ax.set_facecolor("#0d0f18")
    ax.imshow(wc)
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


# ================================
# ANALYSIS  (unchanged logic)
# ================================
if analyze_btn:

    if not hashtag:
        st.warning("Please enter a hashtag.")
        st.stop()

    keyword = hashtag.lower().replace("#", "")

    df = df_main[
        df_main['text'].str.lower().str.contains(f"#{keyword}", na=False) |
        df_main['text'].str.lower().str.contains(keyword, na=False)
    ]

    # always run — if nothing in dataset, use keyword itself as input
    if df.empty:
        df = pd.DataFrame({"text": [f"#{keyword} {keyword}"]})

    df = df.copy()
    df['clean'] = df['text'].apply(clean)
    df = df[df['clean'] != ""]

    with st.spinner("Running inference…"):
        df[['Sentiment', 'Polarity', 'Confidence']] = df['clean'].apply(
            lambda x: pd.Series(analyze_text(x))
        )

    counts = df['Sentiment'].value_counts()
    pos = counts.get('Positive', 0)
    neg = counts.get('Negative', 0)
    neu = counts.get('Neutral', 0)
    total = len(df)
    avg_polarity = df['Polarity'].mean()
    trend_score = (pos - neg) / total

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="stats-row">
        <div class="stat-card pos">
            <div class="stat-val pos">{round(pos/total*100,1)}%</div>
            <div class="stat-lbl">Positive</div>
        </div>
        <div class="stat-card neg">
            <div class="stat-val neg">{round(neg/total*100,1)}%</div>
            <div class="stat-lbl">Negative</div>
        </div>
        <div class="stat-card neu">
            <div class="stat-val neu">{round(neu/total*100,1)}%</div>
            <div class="stat-lbl">Neutral</div>
        </div>
        <div class="stat-card pol">
            <div class="stat-val pol">{avg_polarity:+.3f}</div>
            <div class="stat-lbl">Avg Polarity</div>
        </div>
        <div class="stat-card trn">
            <div class="stat-val trn">{trend_score:+.3f}</div>
            <div class="stat-lbl">Trend Score</div>
        </div>
    </div>
    <p style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#666;margin-top:0.2rem;">
        {total} tweets analysed &nbsp;·&nbsp; <span style="color:#ff4b6e">#{keyword}</span>
    </p>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<p class="sec-label">Sentiment Distribution</p>', unsafe_allow_html=True)
        st.pyplot(dark_bar(['Positive','Negative','Neutral'], [pos,neg,neu],
                           ['#00c896','#ff4b6e','#f0a500']))
    with c2:
        st.markdown('<p class="sec-label">Polarity Distribution</p>', unsafe_allow_html=True)
        st.pyplot(dark_hist(df['Polarity']))

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown('<p class="sec-label">WordCloud</p>', unsafe_allow_html=True)
    wc_fig = dark_wordcloud(df['clean'])
    if wc_fig:
        st.pyplot(wc_fig)

    st.markdown("<hr>", unsafe_allow_html=True)

    t1, t2 = st.columns(2)
    with t1:
        st.markdown('<p class="sec-label"> Top Positive</p>', unsafe_allow_html=True)
        top_pos = df.sort_values('Polarity', ascending=False)['text'].head(2).tolist()
        cards = "".join(f'<div class="tweet-card pos">{t}</div>' for t in top_pos)
        st.markdown(f'<div class="tweet-grid">{cards}</div>', unsafe_allow_html=True)
    with t2:
        st.markdown('<p class="sec-label"> Top Negative</p>', unsafe_allow_html=True)
        top_neg = df.sort_values('Polarity')['text'].head(2).tolist()
        cards = "".join(f'<div class="tweet-card neg">{t}</div>' for t in top_neg)
        st.markdown(f'<div class="tweet-grid">{cards}</div>', unsafe_allow_html=True)