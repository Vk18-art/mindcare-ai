# ============================================================
#  MindCare AI – Student Stress Monitoring & Support System
#  A beginner-friendly Streamlit app for a student hackathon
# ============================================================

# ── IMPORTS ──────────────────────────────────────────────────
import streamlit as st          # Web UI framework
import pandas as pd             # Data handling (CSV read/write)
import matplotlib.pyplot as plt # Drawing the stress trend graph
import matplotlib.dates as mdates
from textblob import TextBlob   # Sentiment / emotion detection
from datetime import datetime   # Getting today's date & time
import os                       # Checking if the CSV file exists

# ── CONSTANTS ────────────────────────────────────────────────
CSV_FILE = "mood_data.csv"      # Where we store daily mood entries

MOOD_EMOJIS = {
    "Happy 😊":    "Happy",
    "Neutral 😐":  "Neutral",
    "Stressed 😰": "Stressed",
}

# Suggestions shown to the user depending on their mood
SUGGESTIONS = {
    "Happy": {
        "icon": "🌟",
        "title": "Keep shining!",
        "tips": [
            "🎯  Channel this energy into a challenging task.",
            "🤝  Share your positivity — reach out to a classmate.",
            "📓  Journal what made today great so you can revisit it.",
            "🏃  Celebrate with a short walk or your favourite workout.",
        ],
    },
    "Neutral": {
        "icon": "⚖️",
        "title": "Stay balanced!",
        "tips": [
            "📅  Review your to-do list and prioritise the top 3 tasks.",
            "💧  Drink a glass of water and take a 5-minute stretch break.",
            "🎵  Put on a playlist that lifts your mood while you study.",
            "🌿  Step outside for fresh air — even 10 minutes helps.",
        ],
    },
    "Stressed": {
        "icon": "🧘",
        "title": "Let's ease that stress!",
        "tips": [
            "🌬️  Try box breathing: inhale 4 s → hold 4 s → exhale 4 s → hold 4 s. Repeat 4×.",
            "⏸️   Take a proper 15-minute break — close every tab and rest your eyes.",
            "💬  Text or call a friend. Talking it out really helps.",
            "✍️  Write down exactly what is stressing you — externalising it reduces anxiety.",
        ],
    },
}

# ── HELPER FUNCTIONS ─────────────────────────────────────────

def detect_emotion(text: str):
    """
    Use TextBlob to analyse the polarity of the user's text.
    Returns a human-readable label AND the raw polarity score (-1 to +1).
    """
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        label = "Positive 😊"
    elif polarity < -0.1:
        label = "Negative 😟"
    else:
        label = "Neutral 😐"
    return label, round(polarity, 3)


def load_data() -> pd.DataFrame:
    """
    Load existing mood entries from the CSV file.
    If no file exists yet, return an empty DataFrame with the right columns.
    """
    if os.path.exists(CSV_FILE):
        df = pd.read_csv(CSV_FILE, parse_dates=["Date"])
        return df
    cols = ["Date", "Mood", "Emotion", "Polarity", "Note"]
    return pd.DataFrame(columns=cols)


def save_entry(mood: str, emotion: str, polarity: float, note: str) -> None:
    """
    Append a new mood entry to the CSV file.
    """
    df = load_data()
    new_row = {
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Mood":     mood,
        "Emotion":  emotion,
        "Polarity": polarity,
        "Note":     note,
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(CSV_FILE, index=False)


def check_consecutive_stress(df: pd.DataFrame, n: int = 3) -> bool:
    """
    Return True if the last *n* entries all have Mood == 'Stressed'.
    Used to trigger the proactive alert.
    """
    if len(df) < n:
        return False
    return all(df["Mood"].tail(n) == "Stressed")


def draw_trend_chart(df: pd.DataFrame):
    """
    Draw a line + scatter chart of the mood polarity over time.
    Higher polarity = more positive; lower = more negative/stressed.
    """
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], format="mixed", errors="coerce")

    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor("#0f1117")
    ax.set_facecolor("#1a1d27")

    # Colour each point by mood
    colour_map = {"Happy": "#4ade80", "Neutral": "#facc15", "Stressed": "#f87171"}
    colours = df["Mood"].map(colour_map).fillna("#7dd3fc")

    # Line
    ax.plot(df["Date"], df["Polarity"],
            color="#7dd3fc", linewidth=2, alpha=0.8, zorder=2)
    # Dots
    ax.scatter(df["Date"], df["Polarity"],
               c=colours, s=80, zorder=3, edgecolors="white", linewidths=0.5)

    # Horizontal reference line at 0
    ax.axhline(0, color="#475569", linewidth=1, linestyle="--", alpha=0.6)

    # Shaded zones
    ax.axhspan(0.1,  1.0, alpha=0.07, color="#4ade80")   # positive zone
    ax.axhspan(-1.0, -0.1, alpha=0.07, color="#f87171")  # negative zone

    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("Date / Time", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Sentiment Polarity", color="#94a3b8", fontsize=9)
    ax.set_title("📈  Your Mood Trend Over Time", color="#e2e8f0",
                 fontsize=12, pad=12)
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    plt.xticks(rotation=30)
    for spine in ax.spines.values():
        spine.set_edgecolor("#334155")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4ade80", label="Happy"),
        Patch(facecolor="#facc15", label="Neutral"),
        Patch(facecolor="#f87171", label="Stressed"),
    ]
    ax.legend(handles=legend_elements, loc="upper right",
              framealpha=0.2, labelcolor="#e2e8f0", fontsize=8)

    plt.tight_layout()
    return fig


# ── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="MindCare AI",
    page_icon="🧠",
    layout="centered",
)

# ── CUSTOM CSS ───────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}
.header-banner {
    background: linear-gradient(135deg, #1e3a5f 0%, #0f2540 60%, #162032 100%);
    border-radius: 16px;
    padding: 28px 32px 20px;
    margin-bottom: 24px;
    border: 1px solid #2d4a6e;
    text-align: center;
}
.header-banner h1 {
    font-size: 2.4rem;
    font-weight: 800;
    color: #e0f2fe;
    margin: 0 0 6px;
    letter-spacing: -0.5px;
}
.header-banner p {
    color: #7dd3fc;
    font-size: 1rem;
    margin: 0;
}
.section-card {
    background: #1a1d27;
    border-radius: 14px;
    padding: 22px 26px;
    border: 1px solid #2d3748;
    margin-bottom: 20px;
}
.section-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #93c5fd;
    margin-bottom: 14px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
}
.alert-warning {
    background: #450a0a;
    border: 1px solid #b91c1c;
    border-radius: 10px;
    padding: 16px 20px;
    color: #fca5a5;
    font-size: 0.95rem;
}
.badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 0.9rem;
}
.badge-positive  { background:#14532d; color:#86efac; }
.badge-neutral   { background:#713f12; color:#fde68a; }
.badge-negative  { background:#450a0a; color:#fca5a5; }
.tip-item {
    background: #0f172a;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    margin-bottom: 8px;
    color: #cbd5e1;
    font-size: 0.92rem;
}
</style>
""", unsafe_allow_html=True)


# ── HEADER ───────────────────────────────────────────────────
st.markdown("""
<div class="header-banner">
  <h1>🧠 MindCare AI</h1>
  <p>Your personal student stress monitoring & mental-wellness companion</p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SECTION 1 – INPUT
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">✍️  Section 1 · How are you feeling?</div>',
            unsafe_allow_html=True)

user_text = st.text_area(
    "Describe your current feelings in a sentence or two:",
    placeholder="e.g. I have three exams next week and I can't sleep…",
    height=110,
)

mood_label = st.radio(
    "Pick the mood that best fits you right now:",
    list(MOOD_EMOJIS.keys()),
    horizontal=True,
)
selected_mood = MOOD_EMOJIS[mood_label]

submitted = st.button("💾  Save & Analyse", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SECTION 2 – ANALYSIS & SUGGESTIONS
# ══════════════════════════════════════════════════════════════
if submitted:
    if not user_text.strip():
        st.warning("⚠️ Please write something about how you're feeling before saving.")
    else:
        emotion_label, polarity = detect_emotion(user_text)
        save_entry(selected_mood, emotion_label, polarity, user_text.strip())

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown(
            '<div class="section-title">🔍  Section 2 · Analysis & Suggestions</div>',
            unsafe_allow_html=True,
        )

        badge_class = (
            "badge-positive" if "Positive" in emotion_label
            else "badge-negative" if "Negative" in emotion_label
            else "badge-neutral"
        )
        st.markdown(
            f"**Detected Emotion:** "
            f'<span class="badge {badge_class}">{emotion_label}</span>'
            f"&nbsp;&nbsp;*(Polarity score: {polarity})*",
            unsafe_allow_html=True,
        )
        st.write("")

        df_current = load_data()
        if check_consecutive_stress(df_current):
            st.markdown(
                """<div class="alert-warning">
                🚨 <strong>Heads-up!</strong> You've logged <em>Stressed</em> for your
                last 3+ check-ins. That's a pattern worth taking seriously.<br><br>
                Please consider talking to a counsellor, trusted friend, or family
                member. You don't have to handle everything alone. 💙
                </div>""",
                unsafe_allow_html=True,
            )
            st.write("")

        suggestion = SUGGESTIONS[selected_mood]
        st.markdown(f"### {suggestion['icon']}  {suggestion['title']}")
        for tip in suggestion["tips"]:
            st.markdown(f'<div class="tip-item">{tip}</div>', unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════
# SECTION 3 – STRESS TREND
# ══════════════════════════════════════════════════════════════
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown(
    '<div class="section-title">📊  Section 3 · Stress Trend</div>',
    unsafe_allow_html=True,
)

df = load_data()

if df.empty:
    st.info("No data yet. Submit your first check-in above to start tracking! 🌱")
else:
    total    = len(df)
    happy    = (df["Mood"] == "Happy").sum()
    neutral  = (df["Mood"] == "Neutral").sum()
    stressed = (df["Mood"] == "Stressed").sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total entries", total)
    col2.metric("😊 Happy",      happy)
    col3.metric("😐 Neutral",    neutral)
    col4.metric("😰 Stressed",   stressed)

    st.write("")

    fig = draw_trend_chart(df)
    st.pyplot(fig)
    plt.close(fig)

    with st.expander("🗂️  View recent entries (last 10)"):
        display_df = df.tail(10)[["Date", "Mood", "Emotion", "Polarity", "Note"]]
        st.dataframe(display_df[::-1].reset_index(drop=True), use_container_width=True)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️  Download full mood history (CSV)",
        data=csv_bytes,
        file_name="mindcare_mood_history.csv",
        mime="text/csv",
        use_container_width=True,
    )

st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────
st.markdown(
    "<br><center style='color:#475569;font-size:0.8rem;'>"
    "MindCare AI · Built with ❤️ using Python & Streamlit · Student Hackathon Project"
    "</center>",
    unsafe_allow_html=True,
)