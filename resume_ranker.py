import os
import openai
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2


def rank_resumes(resume_dir, job_desc_path, top_n=20):
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
            if file.endswith(".pdf"):
                pdf_path = os.path.join(root, file)
                text = extract_text_from_pdf(pdf_path)
                resumes.append({"filename": file, "text": text, "path": pdf_path})

    # --- GPT-4 scoring ---
    def gpt4_score_resume(resume_text, jd_text):
        prompt_parts = [
            "You are a technical recruiter scoring a candidate for the role below.\n",
            "Please assign a score from 1 to 10 based on how well the candidate meets the job requirements.\n",
            "Also provide a brief 2-3 sentence explanation for your score.\n",
            "Weight the following traits most heavily, in order of importance:\n",
            "1. Expertise in HPC and applying AI technology to CAE workflows\n",
            "2. Software development experience\n",
            "3. Advanced degree in Machine Learning, Computer Science, Aerospace Engineering, or a related field (preferably a PhD)\n",
            "4. Proficiency in key skills: PyTorch, Modulus, PyG, CAE/CFD/FEA, REST API development, CUDA, Kubernetes, and more\n",
            "5. 5-10 years of industry experience\n",
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
            score = float(content.split()[0])
        except:
            score = None
        return score, content

    for r in tqdm(resumes, desc="GPT-4 scoring"):
        score, explanation = gpt4_score_resume(r["text"], job_description)
        r["gpt4_score"] = score
        r["gpt4_explanation"] = explanation

    # --- Final ranking ---
    for r in resumes:
        r["final_score"] = r["gpt4_score"] if r["gpt4_score"] is not None else 0

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