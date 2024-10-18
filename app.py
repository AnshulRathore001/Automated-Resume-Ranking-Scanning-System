import streamlit as st 
import PyPDF2 as pdf
import google.generativeai as genai
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from io import BytesIO

# -----------------------------------------------
# Google API configuration
# -----------------------------------------------
GOOGLE_API = 'AIzaSyDBLb8u6aJ-TGwdSVoEqhRwQPhb_6WdzVg'
genai.configure(api_key=GOOGLE_API)

# -----------------------------------------------
# Function to Get Gemini Response
# -----------------------------------------------
def get_gemini_response(input_prompt):
    """Generate feedback using Google's Gemini AI model"""
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(input_prompt)
    return response.text

# -----------------------------------------------
# Function to Extract Text from PDF
# -----------------------------------------------
def extract_pdf_text(uploaded_file):
    """Extract text from a PDF file"""
    reader = pdf.PdfReader(uploaded_file)
    text = ""
    for page in range(len(reader.pages)):
        page = reader.pages[page]
        text += str(page.extract_text())
    return text

# -----------------------------------------------
# Function to Rank Resumes Based on Job Description
# -----------------------------------------------
def rank_resumes(resumes, job_description):
    """Rank resumes based on their similarity to the job description using TF-IDF"""
    # Vectorize resumes and job description
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(resumes + [job_description])
    
    # Calculate cosine similarity
    cosine_sim = cosine_similarity(vectors[-1], vectors[:-1])
    
    # Rank resumes based on similarity scores
    scores = cosine_sim.flatten()
    ranked_resumes = sorted(zip(resumes, scores), key=lambda x: x[1], reverse=True)
    
    return ranked_resumes

# -----------------------------------------------
# Function to Convert Data to CSV Format
# -----------------------------------------------
def convert_to_csv(data):
    """Convert the data to CSV format"""
    csv_data = data.to_csv(index=False)
    return BytesIO(csv_data.encode())

# -----------------------------------------------
# Streamlit Application UI
# -----------------------------------------------
st.title("AI-Powered Resume Ranking System")
st.subheader("Evaluate and improve resumes based on job descriptions using NLP and AI")

# Input: Job Description from PDF or Text Area
st.write("**Upload Job Description (PDF) or Type Below**")
jd_file = st.file_uploader("Upload your Job Description (PDF format)", type='pdf')
jd_text = st.text_area("OR Enter your Job Description")

# Input: Resumes (Multiple PDF Upload)
uploaded_files = st.file_uploader("Upload your Resumes (PDF format)", type='pdf', accept_multiple_files=True)

# Submit Button
if st.button("Submit"):
    # Extract Job Description from uploaded PDF or text area
    if jd_file:
        job_description = extract_pdf_text(jd_file)
    elif jd_text:
        job_description = jd_text
    else:
        st.warning("Please upload a Job Description or enter it manually.")
        job_description = None

    if uploaded_files and job_description:
        resumes_texts = []
        resumes_names = [uploaded_file.name for uploaded_file in uploaded_files]  # Store CV file names
        
        # Extract text from uploaded resumes
        for uploaded_file in uploaded_files:
            text = extract_pdf_text(uploaded_file)
            resumes_texts.append(text)

        # Rank resumes based on job description
        ranked_resumes = rank_resumes(resumes_texts, job_description)

        # Create a DataFrame for the results (CV file names and scores)
        results_df = pd.DataFrame({
            "CV File Name": resumes_names,
            "Score": [score for _, score in ranked_resumes]
        })

        # Store results in session state for use when "Get Feedback" is clicked
        st.session_state['ranked_resumes'] = ranked_resumes
        st.session_state['job_description'] = job_description

        # Display results as a table
        st.write("Ranking Results:")
        st.dataframe(results_df)

        # Add a Download Button for CSV
        csv_data = convert_to_csv(results_df)
        st.download_button(
            label="Download Results as CSV",
            data=csv_data,
            file_name='ranked_resumes.csv',
            mime='text/csv'
        )

# Feedback Button
if st.button("Get Feedback"):
    # Check if ranked resumes and job description exist in session state
    if 'ranked_resumes' in st.session_state and 'job_description' in st.session_state:
        ranked_resumes = st.session_state['ranked_resumes']
        job_description = st.session_state['job_description']

        # Collect resumes texts for feedback generation
        resumes_texts = [resume for resume, score in ranked_resumes]

        # Generate feedback using Gemini AI
        feedback_prompt = f"""
            Provide feedback on the following resumes based on the job description:
            Job Description: {job_description}
            Resumes: {resumes_texts}
        """
        feedback_response = get_gemini_response(feedback_prompt)
        st.write("Feedback from Gemini AI:")
        st.write(feedback_response)
    else:
        st.warning("Please submit resumes and job description first to get feedback.")
