import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from tensorflow.keras.preprocessing.sequence import pad_sequences
import nltk
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

# -----------------------------------------------
# PAGE CONFIG — must be first Streamlit call
# -----------------------------------------------
st.set_page_config(
    page_title="MindSense · Sentiment Monitor",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------
# LUXURY THEME — injected CSS
# -----------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --navy:    #0B0F1A;
    --navy-mid:#111827;
    --navy-lt: #1C2333;
    --gold:    #C9A96E;
    --gold-lt: #E2C998;
    --gold-dk: #A0814E;
    --cream:   #F5EFE6;
    --muted:   #8A95A8;
    --success: #4CAF91;
    --warn:    #E8A45A;
    --info:    #6A9FD4;
    --radius:  12px;
}

/* ── Global resets ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--navy) !important;
    color: var(--cream) !important;
}

.stApp {
    background: linear-gradient(160deg, #0B0F1A 0%, #111827 60%, #0D1520 100%) !important;
    min-height: 100vh;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2.5rem 2rem 4rem !important;
    max-width: 780px !important;
    margin: 0 auto;
}

/* ── Typography ── */
h1, h2, h3 { font-family: 'Cormorant Garamond', serif !important; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3.5rem 0 2rem;
    position: relative;
}
.hero::before {
    content: '';
    position: absolute;
    top: 0; left: 50%;
    transform: translateX(-50%);
    width: 1px;
    height: 56px;
    background: linear-gradient(to bottom, transparent, var(--gold));
}
.hero-eyebrow {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.35em;
    text-transform: uppercase;
    color: var(--gold);
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.6rem;
    font-weight: 300;
    line-height: 1.2;
    color: var(--cream);
    margin: 0 0 0.6rem;
}
.hero-title span { color: var(--gold); font-style: italic; }
.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    font-weight: 300;
    color: var(--muted);
    letter-spacing: 0.04em;
}

/* ── Divider ── */
.divider {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin: 2rem 0;
}
.divider::before, .divider::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(to right, transparent, #2A3347, transparent);
}
.divider-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--gold-dk);
}

/* ── Card / panel ── */
.card {
    background: var(--navy-lt);
    border: 1px solid #1E2A3E;
    border-radius: var(--radius);
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
}
.card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold-dk), transparent);
    opacity: 0.6;
}
.card-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.1rem;
    font-weight: 400;
    color: var(--gold-lt);
    margin-bottom: 0.9rem;
    letter-spacing: 0.02em;
}
.card-body {
    font-size: 0.875rem;
    color: var(--muted);
    line-height: 1.75;
}

/* ── About bullets ── */
.about-item {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    margin-bottom: 0.65rem;
    font-size: 0.875rem;
    color: #A8B4C5;
    line-height: 1.65;
}
.about-icon {
    color: var(--gold);
    font-size: 0.75rem;
    margin-top: 0.25rem;
    flex-shrink: 0;
}

/* ── Section labels ── */
.section-label {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.28em;
    text-transform: uppercase;
    color: var(--gold-dk);
    margin-bottom: 0.5rem;
}

/* ── Text area ── */
.stTextArea textarea {
    background: var(--navy-mid) !important;
    border: 1px solid #2A3347 !important;
    border-radius: 8px !important;
    color: var(--cream) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    padding: 1rem !important;
    line-height: 1.7 !important;
    transition: border-color 0.25s !important;
    box-shadow: none !important;
}
.stTextArea textarea:focus {
    border-color: var(--gold-dk) !important;
    box-shadow: 0 0 0 2px rgba(201,169,110,0.08) !important;
}
.stTextArea textarea::placeholder { color: #4A5568 !important; }

/* ── Selectbox ── */
.stSelectbox > div > div {
    background: var(--navy-mid) !important;
    border: 1px solid #2A3347 !important;
    border-radius: 8px !important;
    color: var(--cream) !important;
    font-size: 0.875rem !important;
}
.stSelectbox [data-baseweb="select"] { background: transparent !important; }

/* ── Buttons ── */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
    transition: all 0.25s !important;
    cursor: pointer !important;
}

/* Primary — Analyze */
.stButton:first-of-type > button,
button[kind="primary"] {
    background: linear-gradient(135deg, var(--gold-dk), var(--gold)) !important;
    color: var(--navy) !important;
    border: none !important;
    padding: 0.6rem 2.2rem !important;
    box-shadow: 0 4px 20px rgba(201,169,110,0.25) !important;
}
.stButton:first-of-type > button:hover {
    box-shadow: 0 6px 28px rgba(201,169,110,0.4) !important;
    transform: translateY(-1px) !important;
}

/* Secondary — Use Sample */
.stButton > button[kind="secondary"],
.stButton + .stButton > button {
    background: transparent !important;
    color: var(--gold) !important;
    border: 1px solid var(--gold-dk) !important;
    padding: 0.55rem 1.4rem !important;
}
.stButton + .stButton > button:hover {
    background: rgba(201,169,110,0.06) !important;
}

/* ── Result metric cards ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
}
.metric-card {
    flex: 1;
    background: var(--navy-mid);
    border: 1px solid #1E2A3E;
    border-radius: var(--radius);
    padding: 1.2rem 1.4rem;
    text-align: center;
    position: relative;
    overflow: hidden;
}
.metric-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--gold), transparent);
    opacity: 0.5;
}
.metric-label {
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 0.5rem;
}
.metric-value {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.9rem;
    font-weight: 300;
    color: var(--gold-lt);
}
.metric-sub {
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 0.2rem;
}

/* ── Guidance card colors ── */
.guidance-happy {
    border-color: var(--success) !important;
    background: rgba(76,175,145,0.04) !important;
}
.guidance-sad {
    border-color: var(--warn) !important;
    background: rgba(232,164,90,0.04) !important;
}
.guidance-neutral {
    border-color: var(--info) !important;
    background: rgba(106,159,212,0.04) !important;
}
.guidance-text {
    font-size: 0.9rem;
    line-height: 1.8;
    color: #C8D0DC;
}
.guidance-tip {
    display: flex;
    align-items: flex-start;
    gap: 0.6rem;
    margin-top: 0.6rem;
    font-size: 0.82rem;
    color: var(--muted);
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--gold) !important; }

/* ── Warning / info overrides ── */
.stAlert { border-radius: 8px !important; font-size: 0.875rem !important; }

/* ── Horizontal rule ── */
hr { border-color: #1E2A3E !important; margin: 1.8rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------
# LOAD MODEL + TOKENIZER + ENCODER
# -----------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = tf.keras.models.load_model("sentiment_model_files/rnn_model.h5")
    with open("sentiment_model_files/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("sentiment_model_files/label_encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    return model, tokenizer, encoder

model, tokenizer, encoder = load_artifacts()
MAX_LEN = 100

stop_words = set(stopwords.words("english"))

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def predict_sentiment(text):
    cleaned = clean_text(text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    prob = model.predict(padded, verbose=0)[0]
    class_index = np.argmax(prob)
    label = encoder.inverse_transform([class_index])[0]
    confidence = float(np.max(prob))
    return label, confidence, prob


# -----------------------------------------------
# SECTION 1 — HERO HEADER
# -----------------------------------------------
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Mental Health · AI · NLP</div>
    <h1 class="hero-title">AI-Based <span>Mental Health</span><br>Sentiment Monitoring</h1>
    <p class="hero-subtitle">Emotion Detection using Simple Recurrent Neural Networks</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="divider"><div class="divider-dot"></div></div>', unsafe_allow_html=True)


# -----------------------------------------------
# SECTION 2 — ABOUT THE PROJECT
# -----------------------------------------------
st.markdown("""
<div class="card">
    <div class="card-title">About the Project</div>
    <div class="about-item">
        <span class="about-icon">◆</span>
        <span>Emotional AI enables machines to detect and interpret human feelings from unstructured text, powering more empathetic and responsive digital experiences.</span>
    </div>
    <div class="about-item">
        <span class="about-icon">◆</span>
        <span>Natural Language Processing (NLP) drives applications ranging from mental health chatbots and sentiment dashboards to real-time social media analysis.</span>
    </div>
    <div class="about-item">
        <span class="about-icon">◆</span>
        <span>Recurrent Neural Networks (RNNs) excel at sequence learning — capturing word order and contextual meaning to classify emotional nuance with high accuracy.</span>
    </div>
</div>
""", unsafe_allow_html=True)


# -----------------------------------------------
# SESSION STATE INIT
# -----------------------------------------------
if "textarea_value" not in st.session_state:
    st.session_state.textarea_value = ""
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False


# -----------------------------------------------
# SECTION 3 — TEXT INPUT
# -----------------------------------------------
st.markdown('<div class="section-label">Your thoughts</div>', unsafe_allow_html=True)

user_input = st.text_area(
    label="",
    placeholder="Enter your thoughts or feelings here…",
    height=130,
    label_visibility="collapsed",
    value=st.session_state.textarea_value,
    key="textarea_widget",
)
# Keep session state in sync with manual typing
st.session_state.textarea_value = user_input

st.markdown('<div class="section-label" style="margin-top:1rem;">Or try a sample</div>', unsafe_allow_html=True)

sample = st.selectbox(
    label="",
    options=[
        "I feel very happy and motivated today",
        "I am stressed and anxious about exams",
        "I feel nothing special, just normal",
        "Everything is going wrong in my life",
        "I am grateful for the people around me",
        "I feel completely overwhelmed and exhausted",
    ],
    label_visibility="collapsed",
)

col_btn1, col_btn2, _ = st.columns([1.4, 1.4, 3])
with col_btn1:
    analyze_clicked = st.button("✦  Analyze Emotion", type="primary", use_container_width=True)
with col_btn2:
    use_sample = st.button("Use Sample", use_container_width=True)

# "Use Sample" — inject text into state and rerun so textarea updates
if use_sample:
    st.session_state.textarea_value = sample
    st.session_state.run_analysis = True
    st.rerun()

# Determine the text to analyze
active_text = st.session_state.textarea_value

# Trigger analysis either from button click or after sample injection
should_analyze = analyze_clicked or st.session_state.run_analysis
if st.session_state.run_analysis:
    st.session_state.run_analysis = False  # reset flag after consuming it

st.markdown('<div class="divider"><div class="divider-dot"></div></div>', unsafe_allow_html=True)


# -----------------------------------------------
# SECTIONS 4–7 — PREDICTION + OUTPUT
# -----------------------------------------------
user_input = active_text  # always use the resolved (possibly injected) text
if should_analyze:
    if not user_input.strip():
        st.warning("Please enter or select some text before analyzing.")
    else:
        with st.spinner("Analyzing emotional patterns…"):
            label, confidence, prob = predict_sentiment(user_input)

        # ── SECTION 5 — PREDICTION OUTPUT ──
        st.markdown('<div class="section-label">Prediction result</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="metric-label">Emotion Detected</div>
                <div class="metric-value">{label.title()}</div>
                <div class="metric-sub">Primary classification</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Confidence</div>
                <div class="metric-value">{confidence * 100:.1f}%</div>
                <div class="metric-sub">Model certainty</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Emotional Status</div>
                <div class="metric-value" style="font-size:1.3rem;padding-top:0.3rem;">{label.title()}</div>
                <div class="metric-sub">Current state</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── SECTION 6 — VISUALIZATION ──
        st.markdown('<div class="section-label" style="margin-top:1.6rem;">Emotion probability distribution</div>', unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(8, 3.2))
        fig.patch.set_facecolor('#111827')
        ax.set_facecolor('#111827')

        classes = encoder.classes_
        x = np.arange(len(classes))
        bar_colors = ['#C9A96E' if i == np.argmax(prob) else '#2A3347' for i in range(len(classes))]
        edge_colors = ['#E2C998' if i == np.argmax(prob) else '#3A4A60' for i in range(len(classes))]

        bars = ax.bar(x, prob, color=bar_colors, edgecolor=edge_colors, linewidth=0.8,
                      width=0.55, zorder=3)

        # Annotate bars
        for bar, p in zip(bars, prob):
            if p > 0.02:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012,
                        f'{p*100:.1f}%', ha='center', va='bottom',
                        fontsize=7.5, color='#8A95A8', fontfamily='DejaVu Sans')

        ax.set_xticks(x)
        ax.set_xticklabels([c.title() for c in classes], color='#8A95A8',
                           fontsize=8.5, fontfamily='DejaVu Sans')
        ax.set_ylabel('Probability', color='#4A5568', fontsize=8, labelpad=10)
        ax.tick_params(colors='#4A5568', length=0)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.spines['bottom'].set_color('#1E2A3E')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0%}'))
        ax.set_ylim(0, min(1.0, max(prob) + 0.18))
        ax.grid(axis='y', color='#1E2A3E', linewidth=0.6, zorder=0)
        ax.set_axisbelow(True)

        # Confidence bar (secondary chart below)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        # ── SECTION 7 — EMOTIONAL GUIDANCE ──
        st.markdown('<div class="section-label" style="margin-top:1.6rem;">Emotional guidance</div>', unsafe_allow_html=True)

        lbl_lower = label.lower()

        if any(w in lbl_lower for w in ["happy", "joy", "positive", "excit", "gratit"]):
            card_class = "guidance-happy"
            icon = "🌟"
            headline = "Your energy is radiant — keep nurturing it."
            tips = [
                ("Share it", "Positivity compounds when shared. Reach out to someone who might need a lift today."),
                ("Anchor the moment", "Write down what made today meaningful — it's a resource for harder days."),
                ("Sustain it", "Protect your energy with rest, movement, and the people who energise you."),
            ]
        elif any(w in lbl_lower for w in ["sad", "anxiet", "anxiety", "stress", "fear", "depress", "worry"]):
            card_class = "guidance-sad"
            icon = "🤝"
            headline = "It's okay not to be okay — give yourself grace."
            tips = [
                ("Breathe", "Try box breathing: inhale 4s, hold 4s, exhale 4s, hold 4s. Repeat three times."),
                ("Connect", "Share what you're feeling with someone you trust — a conversation can shift perspective."),
                ("Ground yourself", "Step outside for five minutes. Fresh air and a change of scenery quietly resets the mind."),
            ]
        else:
            card_class = "guidance-neutral"
            icon = "🌿"
            headline = "A calm state is a gift — use it with intention."
            tips = [
                ("Reflect", "Journal your thoughts — even neutral days hold insights worth capturing."),
                ("Move gently", "A short walk, a stretch, or simply sitting outside can refresh a quiet mind."),
                ("Plan with clarity", "Use this balanced state to set one meaningful intention for tomorrow."),
            ]

        tips_html = "".join([
            f'<div class="guidance-tip"><span style="color:var(--gold);font-size:0.7rem;margin-top:0.15rem;">◆</span>'
            f'<span><strong style="color:#C8D0DC;">{t[0]}.</strong> {t[1]}</span></div>'
            for t in tips
        ])

        st.markdown(f"""
        <div class="card {card_class}">
            <div style="font-size:1.5rem;margin-bottom:0.5rem;">{icon}</div>
            <div class="card-title" style="font-size:1.2rem;margin-bottom:0.75rem;">{headline}</div>
            {tips_html}
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------
# FOOTER
# -----------------------------------------------
st.markdown("""
<div class="divider" style="margin-top:3rem;"><div class="divider-dot"></div></div>
<p style="text-align:center;font-size:0.7rem;color:#3A4A60;letter-spacing:0.15em;text-transform:uppercase;margin-top:0.5rem;">
    MindSense · AI Mental Health Monitoring · Powered by RNN
</p>
""", unsafe_allow_html=True)