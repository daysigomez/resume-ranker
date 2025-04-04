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

    # --- Get real embeddings ---
    def get_embedding(text, model="text-embedding-3-small"):
        text = text.replace("\n", " ")
        return openai.embeddings.create(input=[text], model=model).data[0].embedding

    jd_embedding = get_embedding(job_description)

    valid_resumes = []
    for r in tqdm(resumes, desc="Embedding resumes"):
        if r["text"] and len(r["text"].strip()) > 50:
            try:
                r["embedding"] = get_embedding(r["text"])
                r["cosine_similarity"] = cosine_similarity([jd_embedding], [r["embedding"]])[0][0]
                valid_resumes.append(r)
            except Exception as e:
                print(f"❌ Failed to embed '{r['filename']}': {e}")
        else:
            print(f"⚠️ Skipping '{r['filename']}' — empty or too short.")

    # --- Sort and filter top N ---
    sorted_resumes = sorted(valid_resumes, key=lambda x: x["cosine_similarity"], reverse=True)
    top_resumes = sorted_resumes[:top_n]

    # --- GPT-4 scoring ---
    def gpt4_score_resume(resume_text, jd_text):
        prompt_parts = [
            "You are a technical recruiter scoring a candidate for the role below.\n",
            "Please assign a score from 1 to 10 based on how well the candidate meets the job requirements. Weight the following traits most heavily, in order of importance:\n",
            "- PhD or advanced degree in relevant fields\n",
            "- Experience with HPC and CAE workflows\n",
            "- Expertise in PyTorch, Modulus, PyG, CUDA, Kubernetes\n",
            "- Strong software development and API experience\n\n",
            "Job Description:\n",
            jd_text,
            "\n\nResume:\n",
            resume_text,
            "\n\nScore (just the number):"
        ]
        prompt = "".join(prompt_parts)

        response = openai.chat.completions.create(
            model="gpt-4-0125-preview",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=20
        )
        score_str = response.choices[0].message.content.strip()
        try:
            return float(score_str.split()[0])
        except:
            return None

    for r in tqdm(top_resumes, desc="GPT-4 scoring"):
        r["gpt4_score"] = gpt4_score_resume(r["text"], job_description)

    # --- Final ranking ---
    for r in top_resumes:
        r["final_score"] = (0.7 * r["gpt4_score"]) + (0.3 * r["cosine_similarity"] * 10)

    ranked = sorted(top_resumes, key=lambda x: x["final_score"], reverse=True)

    # --- Export to CSV ---
    df = pd.DataFrame([{
        "Filename": r["filename"],
        "CosineSimilarity": round(r["cosine_similarity"], 4),
        "GPT4Score": r["gpt4_score"],
        "FinalScore": round(r["final_score"], 2)
    } for r in ranked])

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