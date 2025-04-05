import os
import openai
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2


def rank_resumes(resume_dir, job_desc_path, top_n=20, st=None):
    openai.api_key = os.getenv("OPENAI_API_KEY")
    review_dir = "resumes_to_review"

    # --- Load job description ---
    with open(job_desc_path, "r", encoding="utf-8") as f:
        job_description = f.read()

    # --- Convert PDFs to text using PyPDF2 ---
    def extract_text_from_pdf(pdf_path):
        text = ""
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
        except Exception as e:
            print(f"❌ Skipping {pdf_path}: {e}")
        return text

    resumes = []
    for root, _, files in os.walk(resume_dir):
        for file in files:
            if file.startswith("._") or "__MACOSX" in root:
                continue
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                text = extract_text_from_pdf(pdf_path)
                resumes.append({"filename": file, "text": text, "path": pdf_path})

    # --- Get embeddings for cosine similarity ---
    def get_embedding(text, model="text-embedding-3-small"):
        text = text.replace("\n", " ")
        return openai.embeddings.create(input=[text], model=model).data[0].embedding

    jd_embedding = get_embedding(job_description)

    for r in tqdm(resumes, desc="Embedding resumes"):
        try:
            r["embedding"] = get_embedding(r["text"])
            r["cosine_similarity"] = cosine_similarity([jd_embedding], [r["embedding"]])[0][0]
        except Exception as e:
            print(f"❌ Embedding failed for {r['filename']}: {e}")
            r["cosine_similarity"] = 0

    # --- GPT-4 scoring ---
    def gpt4_score_resume(resume_text, jd_text):
        prompt_parts = [
            "You are a technical recruiter scoring a candidate for the role below.\n",
            "Please assign a score from 1 to 10 based on how well the candidate meets the job requirements.\n",
            "Also provide a concise explanation (1–2 short sentences only) without starting with 'Explanation:' or repeating the score.\n",
            "Weight the following traits most heavily, in order of importance:\n",
            "1. Expertise in HPC and applying AI technology to CAE workflows\n",
            "2. Software development experience\n",
            "3. Advanced degree in Machine Learning, Computer Science, Aerospace Engineering, or a related field (preferably a PhD)\n",
            "4. Proficiency in key skills: PyTorch, Modulus, PyG, CAE/CFD/FEA, REST API development, CUDA, Kubernetes, and more\n",
            "5. 5-10 years of industry experience (but consider 3–4 years acceptable if the experience is mainly focused on HPC and applying AI to CAE workflows)\n",
            "6. Experience engaging directly with customers to understand their needs and providing tailored solutions\n\n",
            "Job Description:\n",
            jd_text,
            "\n\nResume:\n",
            resume_text,
            "\n\nResponse (score and explanation):"
        ]
        prompt = "".join(prompt_parts)

        response = openai.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=150
        )
        content = response.choices[0].message.content.strip()
        try:
            import re
            match = re.search(r"\b\d+(\.\d+)?\b", content)
            if match:
                score = float(match.group())
                explanation = content[match.end():].strip(" -:")
            else:
                score = None
                explanation = content
        except:
            score = None
            explanation = content
        return score, explanation

    if st:
        progress_bar = st.progress(0, text="Starting resume scoring...")

    for idx, r in enumerate(resumes, start=1):
        status_msg = f"Scoring resume {idx}/{len(resumes)}: {r['filename']}"
        print(status_msg)
        if st:
            progress_bar.progress(idx / len(resumes), text=status_msg)
        score, explanation = gpt4_score_resume(r["text"], job_description)
        r["gpt4_score"] = score
        r["gpt4_explanation"] = explanation

    # --- Final ranking ---
    for r in resumes:
        r["final_score"] = (
            0.7 * r["gpt4_score"] + 0.3 * (r["cosine_similarity"] * 10)
        ) if r["gpt4_score"] is not None else 0

    ranked = sorted(resumes, key=lambda x: x["final_score"], reverse=True)

    # --- Export to CSV ---
    df = pd.DataFrame([{
        "Filename": r["filename"],
        "FinalScore": round(r["final_score"], 2),
        "GPT4Explanation": r["gpt4_explanation"]
    } for r in ranked if "gpt4_explanation" in r])

    df["Rank"] = np.arange(1, len(df) + 1)

    df.to_csv("ranked_resumes.csv", index=False)

    print("✅ Ranked resumes exported to ranked_resumes.csv")

    # --- Copy top resumes to review folder ---
    os.makedirs(review_dir, exist_ok=True)
    for r in ranked[:top_n]:
        dst_path = os.path.join(review_dir, r["filename"])
        if os.path.exists(r["path"]):
            shutil.copy(r["path"], dst_path)

    print(f"✅ Top {top_n} resumes copied to '{review_dir}/'")
    return df, review_dir
