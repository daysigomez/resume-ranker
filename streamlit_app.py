import streamlit as st
import zipfile
import tempfile
import shutil
import os
import pandas as pd

from resume_ranker import rank_resumes  # This is your refactored logic as a function

st.title("Resume Ranker 💼")
st.markdown(
     """
    Please upload a zip file of the resume PDFs you want to review.  
    You will receive a CSV with all of the resumes ranked as well as a zip file of the top 20 resumes.  
    **Note:** This ranker is optimized for the HPC Engineer, AI and Data role.
    """
)

# Store OpenAI API key securely from Streamlit secrets
os.environ["OPENAI_API_KEY"] = st.secrets["openai"]["api_key"]
uploaded_zip = st.file_uploader("Upload a ZIP file of resumes (PDFs)", type="zip")

if uploaded_zip:
    with tempfile.TemporaryDirectory() as tmpdir:
        zip_path = os.path.join(tmpdir, "resumes.zip")
        with open(zip_path, "wb") as f:
            f.write(uploaded_zip.read())

        # Extract PDFs to a temp folder
        resumes_folder = os.path.join(tmpdir, "resumes")
        os.makedirs(resumes_folder, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(resumes_folder)

        # Use the pre-saved job description file
        job_desc_file = os.path.join(os.path.dirname(__file__), "job_description.txt")

        if st.button("Run Resume Ranking"):
            status_placeholder = st.empty()
            progress_placeholder = st.empty()

            progress_bar = st.progress(0, text="Starting app...")

            ranked_df, review_folder = rank_resumes(
                resumes_folder,
                job_desc_file,
                top_n=20,
                st=st,
                progress_bar=progress_bar
            )

            # Show results
            st.success("Done! Top resumes ranked below:")
            st.dataframe(ranked_df)

            # Download CSV
            csv_path = os.path.join(tmpdir, "ranked_resumes.csv")
            ranked_df.to_csv(csv_path, index=False)
            with open(csv_path, "rb") as f:
                st.download_button("Download CSV", f, file_name="ranked_resumes.csv")

            # Zip the top resumes
            review_zip = os.path.join(tmpdir, "resumes_to_review.zip")
            shutil.make_archive(review_zip.replace(".zip", ""), 'zip', review_folder)
            with open(review_zip, "rb") as f:
                st.download_button("Download Top 20 Resumes (ZIP)", f, file_name="resumes_to_review.zip")