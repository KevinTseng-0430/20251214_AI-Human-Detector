import re

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_dataset():
    """Return small, hand-crafted samples for AI-like and human-like writing."""
    ai_like_en = [
        "This article explores the impact of artificial intelligence on healthcare systems. In conclusion, data-driven triage pipelines will continue to mature.",
        "The model achieves strong accuracy according to our evaluation, demonstrating scalability and consistency across benchmarks.",
        "Overall, the findings suggest that energy efficiency will remain a key metric for sustainable infrastructure planning.",
        "In this section we formalize the problem, introduce the methodology, and discuss limitations of our approach.",
        "The results are summarized in Table 2 and indicate robust performance even under constrained memory settings.",
        "We recommend adopting a phased rollout strategy with clear monitoring criteria and periodic audits.",
        "Future work can extend this framework to multilingual corpora without significant architectural changes.",
        "The experiment was replicated three times to reduce variance, and the confidence intervals are reported accordingly.",
        "This overview highlights how modular design promotes maintainability, reproducibility, and traceability of decisions.",
        "In conclusion, these observations underscore the importance of aligning incentives with measurable outcomes.",
        "The baseline uses deterministic heuristics, while the proposed system integrates probabilistic reasoning and calibration.",
        "We adopt a minimal set of hyperparameters to simplify deployment and reduce operational overhead."
    ]

    ai_like_zh = [
        "本研究探討大型語言模型在客服自動化的效能，實驗結果顯示準確率與成本皆有顯著提升。",
        "整體流程採分階段部署並加入即時監控，以確保模型表現穩定且可追蹤。",
        "數據標註經過雙重審核，並以一致性指標確認品質後才進入訓練流程。",
        "我們使用多語平衡語料進行微調，以降低偏差並提升跨領域的泛化能力。",
        "實驗重複三次以降低隨機性，表 2 列出各項指標的信賴區間與變異程度。",
        "結論強調模型需搭配治理機制與稽核流程，才能長期維持合規性與可解釋性。",
    ]

    human_like_en = [
        "I remember walking to class half asleep and spilling coffee everywhere because I forgot to put the lid on.",
        "My grandma tells stories with little details about the smells in her kitchen and the creaky porch swing.",
        "Sometimes I rewrite the same sentence five times before it feels right, and other days I just give up.",
        "The bus was late again, so everyone at the stop started sharing weather complaints like old friends.",
        "I tried three recipes before finally admitting I just wanted noodles with butter and too much pepper.",
        "My friend texted me a meme during the meeting and I had to stare at the ceiling to stop laughing.",
        "There is a crooked tree on my street that always looks like it's waving when the wind picks up.",
        "The concert was so loud my jacket vibrated, but the encore was worth the ringing ears.",
        "Yesterday I biked through the park and nearly collided with a family of ducks crossing the path.",
        "I wrote this paragraph while waiting for laundry to finish, hoping the dryer wouldn't eat a sock.",
        "We argued about which movie to watch longer than the movie itself would have lasted.",
        "The night smelled like rain and sunscreen after a long day at the beach."
    ]

    human_like_zh = [
        "我在便利商店買了咖啡，卻在出門時打翻，整條路都聞得到甜甜的味道。",
        "朋友傳來一張貓咪梗圖，我在捷運上忍笑到眼淚都飆出來，旁邊的人一頭霧水。",
        "週末和爸媽煮火鍋，結果青菜買太多，最後大家硬著頭皮把它們吃完。",
        "昨晚下班騎車回家時突然開始飄雨，路邊的霓虹燈被打濕後看起來模糊又漂亮。",
        "我常常把筆記本寫到一半就停下來，因為貓跳到桌上把墨水弄得亂七八糟。",
        "隔壁的小孩練鋼琴走音，但聽久了反而有點習慣，像是每日的背景音樂。",
    ]

    ai_like = ai_like_en + ai_like_zh
    human_like = human_like_en + human_like_zh
    texts = ai_like + human_like
    labels = [1] * len(ai_like) + [0] * len(human_like)  # 1=AI-like, 0=Human-like
    return texts, np.array(labels)


@st.cache_resource
def load_model():
    texts, labels = build_dataset()
    pipeline = Pipeline(
        [
            # Char n-grams cover中英文混合，避免斷詞差異
            ("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=6000)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    pipeline.fit(texts, labels)
    return pipeline


def get_feature_weights(model, top_k=10):
    """Return the top positive (AI) and negative (human) n-grams."""
    tfidf = model.named_steps["tfidf"]
    clf = model.named_steps["clf"]
    feature_names = tfidf.get_feature_names_out()
    coefs = clf.coef_[0]
    top_ai = sorted(zip(feature_names, coefs), key=lambda x: x[1], reverse=True)[:top_k]
    top_human = sorted(zip(feature_names, coefs), key=lambda x: x[1])[:top_k]
    return top_ai, top_human


def compute_text_stats(text: str):
    tokens = re.findall(r"[A-Za-z\u4e00-\u9fff']+", text)
    sentences = [s for s in re.split(r"[.!?。！？]", text) if s.strip()]
    punctuation = re.findall(r"[.,;:!?，。！？；：]", text)
    word_count = len(tokens)
    unique_ratio = len(set(tokens)) / word_count if word_count else 0
    avg_word_len = np.mean([len(t) for t in tokens]) if tokens else 0
    sentence_len = word_count / len(sentences) if sentences else word_count
    punct_density = len(punctuation) / max(len(text), 1)
    return {
        "words": word_count,
        "sentences": len(sentences),
        "avg_word_len": round(avg_word_len, 2),
        "unique_ratio": round(unique_ratio * 100, 1),
        "words_per_sentence": round(sentence_len, 1),
        "punct_density": round(punct_density * 100, 2),
    }


def prob_chart(human_pct: float, ai_pct: float):
    data = pd.DataFrame(
        {"label": ["Human", "AI"], "probability": [human_pct, ai_pct], "color": ["#34bfa3", "#4c6fff"]}
    )
    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("probability:Q", axis=alt.Axis(format=".0f", title="Probability (%)")),
            y=alt.Y("label:N", sort=None, title=None),
            color=alt.Color("color:N", scale=None),
            tooltip=["label", alt.Tooltip("probability:Q", format=".1f")],
        )
        .properties(height=120)
    )
    st.altair_chart(chart, use_container_width=True)


def feature_chart(features, title):
    df = pd.DataFrame(features, columns=["feature", "weight"])
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("weight:Q", title="weight"),
            y=alt.Y("feature:N", sort="-x", title=None),
            tooltip=["feature", alt.Tooltip("weight:Q", format=".3f")],
            color=alt.value("#ffaa33"),
        )
        .properties(title=title, height=240)
    )
    st.altair_chart(chart, use_container_width=True)


def inject_style():
    st.markdown(
        """
        <style>
            .main {
                background: radial-gradient(circle at 10% 20%, rgba(76,111,255,0.08), transparent 25%),
                            radial-gradient(circle at 80% 0%, rgba(52,191,163,0.12), transparent 20%),
                            linear-gradient(135deg, #0d1117, #111827);
                color: #e8eef6;
            }
            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                max-width: 1000px;
            }
            .glass {
                background: rgba(255, 255, 255, 0.04);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 14px;
                padding: 1.25rem 1.4rem;
                box-shadow: 0 14px 40px rgba(0,0,0,0.25);
            }
            h1, h2, h3, h4 {
                letter-spacing: 0.4px;
            }
            .stTextArea textarea {
                border-radius: 12px;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main():
    st.set_page_config(page_title="AI vs Human Detector", page_icon="🤖", layout="wide")
    inject_style()

    st.title("AI vs Human Detector")
    st.caption("Enter English or Chinese text → instant AI% / Human% with model weights and language stats.")

    hero = st.container()
    with hero:
        st.markdown(
            "<div class='glass'>Lightweight tf-idf + Logistic Regression; demo-trained on a tiny bilingual corpus for quick checks.</div>",
            unsafe_allow_html=True,
        )

    model = load_model()

    default_text = (
        "This draft summarizes the experiment and concludes with recommendations for deployment."
    )
    user_text = st.text_area("Enter text", default_text, height=220)

    if user_text.strip():
        proba = model.predict_proba([user_text.strip()])[0]
        human_pct = float(proba[0] * 100)
        ai_pct = float(proba[1] * 100)

        st.subheader("Detection results")
        col1, col2, col3 = st.columns([1, 1, 1])
        col1.metric("Human%", f"{human_pct:.1f}%", delta=f"{human_pct - ai_pct:+.1f}% vs AI")
        col2.metric("AI%", f"{ai_pct:.1f}%")
        col3.metric("Confidence gap", f"{abs(ai_pct - human_pct):.1f} pts")

        prob_chart(human_pct, ai_pct)

        stats = compute_text_stats(user_text)
        st.subheader("Language stats")
        c1, c2, c3 = st.columns(3)
        c4, c5, c6 = st.columns(3)
        c1.metric("Words", f"{stats['words']}")
        c2.metric("Sentences", f"{stats['sentences']}")
        c3.metric("Avg word len", f"{stats['avg_word_len']}")
        c4.metric("Lexical diversity", f"{stats['unique_ratio']}%")
        c5.metric("Avg sentence len", f"{stats['words_per_sentence']}")
        c6.metric("Punctuation density", f"{stats['punct_density']}%")

        st.caption("Scores are indicative only; please pair with human review and larger datasets.")

    st.divider()
    st.subheader("Model info & keywords")
    texts, labels = build_dataset()
    st.write(f"Training samples: {len(texts)} (AI-like {labels.sum()} / Human-like {(labels == 0).sum()})")
    top_ai, top_human = get_feature_weights(model)

    col_a, col_b = st.columns(2)
    with col_a:
        feature_chart([(f, w) for f, w in top_ai], "AI-leaning n-grams")
    with col_b:
        feature_chart([(f, abs(w)) for f, w in top_human], "Human-leaning n-grams")

    st.caption("To improve accuracy: use larger real datasets, external LM features, or persist a trained model to skip retraining.")


if __name__ == "__main__":
    main()
