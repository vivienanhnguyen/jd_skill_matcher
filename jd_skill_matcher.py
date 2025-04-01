
import streamlit as st
st.set_page_config(page_title="AI JD Skill Matcher", layout="wide")
st.title("💼 AI Job Description Skill Matcher")

import fitz  # PyMuPDF
import spacy
import pandas as pd

import io
def skills_to_df(skills):
    return pd.DataFrame(skills, columns=["Skill", "Score"])

from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util


# Load your enhanced skills file (make sure it's in your working directory)
skills_df = pd.read_csv("Combined_IT_Skills_with_Categories.csv")

# Create a flat list and a dictionary by category
all_skills = skills_df['Skill'].tolist()

@st.cache_resource
def load_semantic_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_semantic_model()

def smart_match_skills(text, skills_list, threshold=0.3):
    import re
    text_lower = text.lower()
    matched_skills = {}
    skill_lookup = {s.lower().strip(): s for s in skills_list}  # original case map

    # Step 1: Phrase match (exact word match)
    for skill in skills_list:
        skill_pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(skill_pattern, text_lower):
            matched_skills[skill.lower().strip()] = (skill, threshold)

    # Step 2: Semantic fallback
    unmatched_skills = list(set(skill_lookup.keys()) - set(matched_skills.keys()))
    text_chunks = [line.strip() for line in text_lower.split("\n") if line.strip()]
    text_embeddings = model.encode(text_chunks, convert_to_tensor=True)

    for skill_key in unmatched_skills:
        original_skill = skill_lookup.get(skill_key)
        if not original_skill:
            continue

        # Filter short/ambiguous skills
        if len(skill_key.split()) == 1 and len(skill_key) <= 2:
            continue
        if len(skill_key.split()[-1]) == 1:
            continue

        skill_embedding = model.encode([original_skill], convert_to_tensor=True)
        similarities = util.cos_sim(skill_embedding, text_embeddings)[0]

        if similarities.max().item() >= threshold:
            print(f"✅ {original_skill} matched — max score: {similarities.max().item():.2f}")
            matched_skills[skill_key] = (original_skill, similarities.max().item())
        else:
            print(f"❌ {original_skill} too low — max score: {similarities.max().item():.2f}")


    final_skills = list(matched_skills.values())  # List of (skill, score)
    return final_skills


def group_by_category(matched_skills, skills_df):
    matched_clean = [s[0].lower().strip() for s in matched_skills]  # s[0] is the skill name

    df = skills_df.copy()
    df['skill_lower'] = df['Skill'].str.lower().str.strip()
    df = df[df['skill_lower'].isin(matched_clean)]
    return df.groupby('Category')['Skill'].apply(list).to_dict()

def compare_jd_resume(jd_text, resume_text, all_skills, skills_df, threshold=0.3):
    # Step 1: Match skills with scores
    jd_scored = smart_match_skills(jd_text, all_skills, threshold)         # [(skill, score)]
    resume_scored = smart_match_skills(resume_text, all_skills, threshold) # [(skill, score)]

    # Step 2: Create lookup dictionaries
    jd_dict = {s.lower(): (s, score) for s, score in jd_scored}
    resume_dict = {s.lower(): (s, score) for s, score in resume_scored}

    # Step 3: Compute matched and missing skills
    matched_keys = set(jd_dict.keys()) & set(resume_dict.keys())
    missing_keys = set(jd_dict.keys()) - set(resume_dict.keys())
    extra_keys = set(resume_dict.keys()) - set(jd_dict.keys())

    matched = [jd_dict[k] for k in matched_keys]
    missing = [jd_dict[k] for k in missing_keys]
    extra = [resume_dict[k] for k in extra_keys]

    # Step 4: Match score
    score = round(100 * len(matched) / len(jd_dict), 2) if jd_dict else 0

    return {
        "score": score,
        "matched": matched,     # list of (skill, score)
        "missing": missing,     # list of (skill, score)
        "extra": extra,         # list of (skill, score)
        "jd_skills": jd_scored,       # all JD-scored skills
        "resume_skills": resume_scored  # all resume-scored skills
    }

def merge_all_skills(matched, missing, extra):
    def to_df(skills, label):
        return pd.DataFrame(skills, columns=["Skill", "Score"]).assign(Type=label)

    df_matched = to_df(matched, "Matched")
    df_missing = to_df(missing, "Missing")
    df_extra = to_df(extra, "Extra") if extra else pd.DataFrame(columns=["Skill", "Score", "Type"])

    return pd.concat([df_matched, df_missing, df_extra], ignore_index=True)



# Load spaCy model
import spacy
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    from spacy.cli import download
    download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


# --- Sidebar Uploads ---
st.sidebar.header("📂 Upload Files")
uploaded_resume = st.sidebar.file_uploader("Upload Resume", type=["pdf", "txt"])
uploaded_jd = st.sidebar.file_uploader("Upload Job Description", type=["pdf", "txt"])
st.markdown("Compare your **Resume** against a **Job Description** to identify skill matches and gaps.")
threshold = st.sidebar.slider(
    "🔍 Semantic Matching Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="Lower values = more fuzzy matches, higher = more strict matches"
)


# --- Extract Text Function ---
def extract_text(file):
    if file.name.endswith(".pdf"):
        pdf = fitz.open(stream=file.read(), filetype="pdf")
        return "\n".join([page.get_text() for page in pdf])

resume_text = extract_text(uploaded_resume)
jd_text = extract_text(uploaded_jd)

# --- Skill Extraction ---
def extract_skills_from_text(text, skills_list):
    doc = nlp(text.lower())
    tokens = set([token.lemma_ for token in doc if token.is_alpha and not token.is_stop])
    matched = set()

    for skill in skills_list:
        skill_tokens = set(skill.lower().split())
        if skill_tokens <= tokens:
            matched.add(skill)

    return list(matched)


# --- Display Preview ---
with st.expander("📄 Resume Preview"):
    st.text_area("Resume Content", resume_text, height=200)

with st.expander("📄 Job Description Preview"):
    st.text_area("JD Content", jd_text, height=200)

# --- Run Analysis ---
if resume_text and jd_text:
    with st.spinner("🔍 Analyzing and matching skills..."):
        # Match skills using new smart matcher
        result = compare_jd_resume(jd_text, resume_text, all_skills, skills_df, threshold=threshold)

        grouped_matched = group_by_category(result['matched'], skills_df)
        grouped_missing = group_by_category(result['missing'], skills_df)

        st.metric("🎯 Skill Match Score", f"{result['score']:.1f}%")
        with st.expander("ℹ️ What Does the Score Mean?"):st.markdown("""
    The **score** shows how semantically similar a skill from your resume is to the skill required in the job description.  
    It's computed using **cosine similarity** between text embeddings (via a Sentence Transformer).

    | **Score Range** | **Meaning**                     |
    |-----------------|----------------------------------|
    | 🟢 `≥ 0.85`      | Strong match – highly relevant   |
    | 🟡 `0.65–0.84`   | Good match – likely relevant     |
    | 🟠 `0.50–0.64`   | Possible match – review manually |
    | 🔴 `< 0.50`      | Weak match – unlikely match      |

    Adjust the **threshold** in the sidebar to make the match stricter or fuzzier.
    """)



        st.markdown("### ✅ Matched Skills with Scores")
        if result['matched']:
            for skill, score in result['matched']:
                st.success(f"{skill} (score: {score:.2f})")
        else:
            st.info("No skills matched.")

        st.markdown("### 🚨 Missing Skills from Resume")
        if result['missing']:
            for skill, score in result['missing']:
                st.error(f"{skill} (expected score: {score:.2f})")
        else:
            st.success("None 🎉")

        st.markdown("### 🛠️ Extra Skills in Resume (Not Required by JD)")
        if result['extra']:
            grouped_extra = group_by_category(result['extra'], skills_df)
            for cat, skills in grouped_extra.items():
                st.info(f"**{cat}**: {', '.join(skills)}")
        else:
            st.success("None 🎯")


        # Combined export
        all_skills_df = merge_all_skills(
            result["matched"],
            result["missing"],
            result.get("extra", [])
        )
        csv_all = all_skills_df.to_csv(index=False)
        st.download_button("⬇️ Download Full Skill Match Report", data=csv_all, file_name="skill_match_report.csv", mime="text/csv")


else:
    st.warning("Please upload or paste both resume and job description.")

    



