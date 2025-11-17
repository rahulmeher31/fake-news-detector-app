import re
import urllib.parse
from typing import Dict, List, Tuple

import streamlit as st

try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# -----------------------------
# Utility: Text processing
# -----------------------------
def clean_text(text: str) -> str:
    return " ".join(text.split()).strip()


SENSATIONAL_WORDS = [
    "shocking", "unbelievable", "you won't believe", "exposed", "secret",
    "miracle", "banned", "hidden truth", "jaw-dropping", "mind-blowing",
    "BREAKING", "urgent", "viral", "must see", "never before seen"
]

EMOTIONAL_WORDS = [
    "outrage", "furious", "disgusting", "heartbreaking", "terrifying",
    "horrifying", "hate", "love", "rage", "panic", "angry", "devastating"
]

SOURCE_CUES = [
    "according to", "reported by", "as stated by", "data from",
    "as per", "researchers at", "study from", "source:", "cited by"
]

FACT_CHECK_CUES = [
    "fact-check", "snopes.com", "politiFact", "factcheck.org"
]

BALANCED_PHRASES = [
    "however", "on the other hand", "while it is true", "at the same time",
    "critics say", "supporters say", "experts say", "officials said"
]

HEDGING_PHRASES = [
    "may", "might", "could", "suggests", "appears to", "possibly"
]


# -----------------------------
# Media Literacy Heuristics
# -----------------------------
def analyze_media_literacy(headline: str, body: str) -> Tuple[float, List[str], List[str]]:
    """
    Returns (heuristic_fake_probability, red_flag_cues, credibility_cues)
    Probability is between 0 and 1.
    """
    text_lower = f"{headline}\n\n{body}".lower()
    full_text = f"{headline} {body}"
    red_flags = []
    credibility_signals = []

    score = 0.0
    max_score = 15.0  # used to normalize into [0, 1]

    # 1. Sensational / clickbait language (red flag)
    sensational_hits = [w for w in SENSATIONAL_WORDS if w.lower() in text_lower]
    if sensational_hits:
        score += 3
        red_flags.append(
            f"Uses sensational / clickbait language ({', '.join(set(sensational_hits))}). "
            "Sensational wording is often used to provoke strong reactions rather than inform."
        )

    # 2. Emotional tone (red flag)
    emotional_hits = [w for w in EMOTIONAL_WORDS if w.lower() in text_lower]
    if emotional_hits:
        score += 2
        red_flags.append(
            f"Emotional or inflammatory words detected ({', '.join(set(emotional_hits))}). "
            "Strong emotional tone can signal persuasive or biased content."
        )

    # 3. Excessive punctuation (red flag)
    exclamations = full_text.count("!")
    if exclamations >= 3:
        score += 2
        red_flags.append(
            f"Multiple exclamation marks ({exclamations}). "
            "Overuse of exclamation marks is common in misleading or clickbait content."
        )

    # 4. ALL CAPS words (red flag)
    caps_words = re.findall(r"\b[A-Z]{4,}\b", full_text)
    if len(caps_words) >= 3:
        score += 2
        red_flags.append(
            "Frequent ALL-CAPS words detected. This style is often used to grab attention "
            "rather than present balanced information."
        )

    # 5. Sources and attribution
    has_source_cue = any(cue in text_lower for cue in SOURCE_CUES)
    if not has_source_cue:
        score += 2
        red_flags.append(
            "No clear source or attribution phrases found (e.g., 'according to', 'reported by'). "
            "Reliable news usually cites where information comes from."
        )
    else:
        score -= 1
        credibility_signals.append(
            "Mentions sources or attribution (e.g., 'according to', 'reported by'). "
            "Clear sourcing is a positive credibility signal, though you should still check if the source is trustworthy."
        )

    # 6. Dates and numbers
    years = re.findall(r"\b(19|20)\d{2}\b", text_lower)  # years like 1999, 2024, etc.
    numbers = re.findall(r"\b\d+(\.\d+)?\b", text_lower)

    if numbers and not years and not has_source_cue:
        score += 1.5
        red_flags.append(
            "Contains specific numbers but no years or sources. "
            "Be cautious when statistics are given without time context or citation."
        )

    if years and has_source_cue:
        score -= 0.5
        credibility_signals.append(
            "Includes years and source-like phrases, which often indicate time-bound, checkable information."
        )

    # 7. Very short / vague article body (red flag)
    body_len = len(body.split())
    if body_len < 50:
        score += 1.5
        red_flags.append(
            "Article body is very short or vague. "
            "Fake or misleading posts often provide minimal detail."
        )
    else:
        credibility_signals.append(
            "Provides more than a few lines of detail. While length alone does not prove truth, "
            "richer context is more typical of legitimate reporting."
        )

    # 8. Fact-check cues (positive)
    has_fact_check = any(fc in text_lower for fc in FACT_CHECK_CUES)
    if has_fact_check:
        score -= 1
        credibility_signals.append(
            "Mentions fact-checking sites or processes. This can be a positive sign, "
            "though you should still verify that the fact-checker is reputable."
        )

    # 9. Balanced or hedged language (positive)
    has_balanced = any(phrase in text_lower for phrase in BALANCED_PHRASES)
    has_hedging = any(phrase in text_lower for phrase in HEDGING_PHRASES)

    if has_balanced:
        score -= 0.7
        credibility_signals.append(
            "Uses balanced language (e.g., 'however', 'on the other hand', 'critics say'). "
            "Balanced reporting that shows multiple perspectives is a positive credibility cue."
        )

    if has_hedging:
        score -= 0.5
        credibility_signals.append(
            "Uses hedging language such as 'may', 'might', or 'could'. "
            "Careful claims and uncertainty markers are common in responsible reporting."
        )

    # Normalize
    fake_prob = min(max(score / max_score, 0.0), 1.0)
    return fake_prob, red_flags, credibility_signals


# -----------------------------
# ML Component (Zero-shot NLI)
# -----------------------------
@st.cache_resource(show_spinner=True)
def load_zeroshot_classifier():
    if not TRANSFORMERS_AVAILABLE:
        return None

    clf = pipeline(
        "zero-shot-classification",
        model="valhalla/distilbart-mnli-12-1"
    )
    return clf


def predict_ml_fake_prob(text: str) -> Tuple[float, Dict[str, float]]:
    """
    Uses zero-shot classification to estimate probability of fake vs real.
    Returns (fake_probability, label_probs_dict).
    """
    classifier = load_zeroshot_classifier()
    if classifier is None:
        return None, {}

    candidate_labels = ["real news", "fake or misleading news"]

    result = classifier(text, candidate_labels=candidate_labels, multi_label=False)
    label_scores = {
        label: score for label, score in zip(result["labels"], result["scores"])
    }

    fake_prob = label_scores.get("fake or misleading news", 0.5)
    real_prob = label_scores.get("real news", 1 - fake_prob)

    return fake_prob, {"fake": fake_prob, "real": real_prob}


# -----------------------------
# Hybrid Prediction
# -----------------------------
def hybrid_prediction(headline: str, body: str) -> Dict:
    text = clean_text(headline + " " + body)

    heuristic_fake_prob, red_flags, credibility_signals = analyze_media_literacy(headline, body)

    ml_fake_prob, ml_probs = predict_ml_fake_prob(text)

    if ml_fake_prob is None:
        combined_fake_prob = heuristic_fake_prob
        source = "Heuristics only (ML model not available)."
    else:
        combined_fake_prob = 0.6 * ml_fake_prob + 0.4 * heuristic_fake_prob
        source = "Hybrid (60% ML model, 40% media literacy heuristics)."

    label = "Likely Fake or Misleading" if combined_fake_prob >= 0.5 else "Likely Real"
    return {
        "label": label,
        "combined_fake_prob": combined_fake_prob,
        "heuristic_fake_prob": heuristic_fake_prob,
        "media_cues_red": red_flags,
        "media_cues_positive": credibility_signals,
        "ml_fake_prob": ml_fake_prob,
        "ml_probs": ml_probs,
        "source": source,
    }


# -----------------------------
# Explanation Generator
# -----------------------------
def explain_decision(result: Dict, headline: str, body: str) -> str:
    label = result["label"]
    fake_prob = result["combined_fake_prob"]
    heuristic_prob = result["heuristic_fake_prob"]
    ml_fake_prob = result["ml_fake_prob"]
    red_flags = result["media_cues_red"]
    positives = result["media_cues_positive"]

    if label == "Likely Fake or Misleading":
        base = (
            "The article leans **fake or misleading** based on both the AI model and media-literacy checks. "
        )
    else:
        base = (
            "The article leans **more likely real** based on both the AI model and media-literacy checks. "
        )

    model_part = ""
    if ml_fake_prob is not None:
        if label == "Likely Fake or Misleading":
            model_part = (
                f"The language model assigned a higher probability to **fake/misleading news** "
                f"(around {ml_fake_prob:.0%} fake vs {1 - ml_fake_prob:.0%} real). "
            )
        else:
            model_part = (
                f"The language model found the article **more consistent with real news** "
                f"(around {1 - ml_fake_prob:.0%} real vs {ml_fake_prob:.0%} fake). "
            )

    if heuristic_prob >= 0.5:
        heur_part = (
            f"Media-literacy heuristics also detected several risk factors, giving a "
            f"fake-score of about {heuristic_prob:.0%}. "
        )
    else:
        heur_part = (
            f"Media-literacy heuristics were more reassuring, with a fake-score of about "
            f"{heuristic_prob:.0%}. "
        )

    red_summary = ""
    if red_flags:
        top_red = red_flags[:2]
        red_summary = "Key red flags include: " + " ".join(top_red) + " "

    pos_summary = ""
    if positives:
        top_pos = positives[:2]
        pos_summary = "Positive credibility signs include: " + " ".join(top_pos) + " "

    responsibility = (
        "This is **not a final verdict**. Use these cues as a guide, and still cross-check the story with "
        "trusted outlets or official sources, especially if it affects your health, money, or voting decisions."
    )

    return base + model_part + heur_part + red_summary + pos_summary + responsibility


# -----------------------------
# Google Search Links Generator
# -----------------------------
def build_verification_links(headline: str, body: str) -> Dict[str, str]:
    """
    Build helpful Google search URLs so users can verify the story themselves.
    """
    query_base = headline.strip() or " ".join(body.split()[:12])
    query_base = query_base.strip()

    if not query_base:
        return {}

    encoded = urllib.parse.quote_plus(query_base)

    # General search
    general_search = f"https://www.google.com/search?q={encoded}"

    # 'headline' fact check
    fact_check_query = urllib.parse.quote_plus(f'"{query_base}" fact check')
    fact_check_search = f"https://www.google.com/search?q={fact_check_query}"

    # Search restricted to known fact-checking sites
    fact_sites_query = urllib.parse.quote_plus(
        f'"{query_base}" site:snopes.com OR site:politifact.com OR site:factcheck.org'
    )
    fact_sites_search = f"https://www.google.com/search?q={fact_sites_query}"

    return {
        "general": general_search,
        "fact_check": fact_check_search,
        "fact_sites": fact_sites_search,
    }


# -----------------------------
# Streamlit UI
# -----------------------------
def main():
    st.set_page_config(
        page_title="Hybrid Fake News & Media Literacy Checker",
        layout="wide"
    )

    st.title("📰 Hybrid Fake News & Media Literacy Checker")
    st.write(
        "Paste a **headline** and **article text** below. "
        "This tool combines an AI model with media literacy cues to help you judge credibility."
    )

    with st.form("news_input_form"):
        headline = st.text_input(
            "Headline",
            placeholder="e.g., NASA Confirms Alien Signal Detected From Moon’s Dark Side"
        )
        body = st.text_area(
            "Article Text",
            height=220,
            placeholder="Paste the article text here..."
        )
        submitted = st.form_submit_button("Analyze Article")

    if submitted:
        if not headline.strip() and not body.strip():
            st.warning("Please enter at least a headline or some article text.")
            return

        with st.spinner("Analyzing article..."):
            result = hybrid_prediction(headline, body)
            explanation_text = explain_decision(result, headline, body)
            verify_links = build_verification_links(headline, body)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.subheader("Overall Verdict")

            fake_prob = result["combined_fake_prob"]
            label = result["label"]

            if label == "Likely Fake or Misleading":
                st.error(f"🚩 {label}")
            else:
                st.success(f"✅ {label}")

            st.write(f"**Combined Fake Probability:** {fake_prob:.2%}")
            st.caption(result["source"])

            st.subheader("Score Breakdown")
            heuristic_prob = result["heuristic_fake_prob"]
            st.write(f"- Heuristic / Media Literacy Fake Score: **{heuristic_prob:.2%}**")
            ml_fake_prob = result["ml_fake_prob"]

            if ml_fake_prob is not None:
                st.write(f"- ML Model Fake Probability: **{ml_fake_prob:.2%}**")
            else:
                st.write("- ML Model: **Not available** (transformers not installed).")

            # New: Verification links
            if verify_links:
                st.markdown("### 🔗 Verify this story yourself")
                st.markdown(
                    f"- 🌐 [Google search for this headline]({verify_links['general']})  \n"
                    f"- ✅ [Search fact-checks for this headline]({verify_links['fact_check']})  \n"
                    f"- 🕵️ [Check on Snopes/PolitiFact/FactCheck.org]({verify_links['fact_sites']})  \n"
                )
                st.caption(
                    "These links open Google with pre-filled queries so you can cross-check the story "
                    "on your own in another tab."
                )

        with col2:
            st.subheader("🧠 Why this result?")
            st.write(explanation_text)

            st.subheader("🚩 Media Literacy Red Flags")
            red_cues = result["media_cues_red"]
            if red_cues:
                for idx, cue in enumerate(red_cues, 1):
                    st.markdown(f"**{idx}.** {cue}")
            else:
                st.info(
                    "No strong red flags were detected. Still, this does **not** guarantee the article is true."
                )

            st.subheader("✅ Credibility Signals")
            pos_cues = result["media_cues_positive"]
            if pos_cues:
                for idx, cue in enumerate(pos_cues, 1):
                    st.markdown(f"**{idx}.** {cue}")
            else:
                st.info(
                    "No clear credibility signals were detected. Be extra careful and cross-check with trusted sources."
                )

            if result["ml_probs"]:
                with st.expander("🔍 Model Confidence Details"):
                    st.write(
                        f"- **Real news probability:** {result['ml_probs']['real']:.2%}\n"
                        f"- **Fake / misleading probability:** {result['ml_probs']['fake']:.2%}\n\n"
                        "The model uses a general-purpose language understanding approach (zero-shot NLI), "
                        "so treat its output as a **signal**, not a final verdict."
                    )

        st.markdown("---")
        st.subheader("How to Use This Tool Responsibly")
        st.markdown(
            """
- Treat this as an **assistant**, not an oracle. Even accurate models can be wrong.
- If the article seems important (health, elections, money, safety), verify it via:
  - Reputable news outlets
  - Official organizations (.gov, .edu, known NGOs)
  - Established fact-checkers (e.g., Snopes, PolitiFact, FactCheck.org)
- Notice **how** the article talks to you:
  - Is it trying to inform you, or *trigger* you (anger, fear, excitement)?
  - Does it clearly state sources, dates, and evidence you can verify?
            """
        )


if __name__ == "__main__":
    main()
