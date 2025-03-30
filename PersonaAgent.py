!pip install pytesseract pillow pdfplumber python-docx selenium beautifulsoup4 openai scikit-learn spacy
!python -m spacy download en_core_web_sm
!apt-get install -y tesseract-ocr
!pip install pytesseract pillow
!apt-get update
!apt-get install -y chromium-chromedriver
!ip install selenium
!apt-get update
!apt-get install -y chromium-browser
!wget https://chromedriver.storage.googleapis.com/114.0.5735.90/chromedriver_linux64.zip
!unzip chromedriver_linux64.zip
!chmod +x chromedriver
!mv chromedriver /usr/bin/chromedriver
!pip install llama-cpp-python chromadb sentence-transformers pymupdf
!pip install huggingface-hub
!huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.1-GGUF mistral-7b-instruct-v0.1.Q4_K_S.gguf --local-dir . --local-dir-use-symlinks False
!pip install spacy[transformers]
!python -m spacy download en_core_web_trf



import pdfplumber
import docx
import pytesseract
from PIL import Image
import spacy
import json
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from bs4 import BeautifulSoup
from sklearn.cluster import KMeans
from llama_cpp import Llama

# Load NLP model
nlp = spacy.load("en_core_web_sm")

# Ensure Tesseract OCR is set up
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# --- Initialize Llama Model ---
def init_model(model_path):
    llm = Llama(
        model_path=model_path,
        n_ctx=32768,  # The max sequence length to use
        n_threads=8,   # Number of CPU threads to use
        n_gpu_layers=35 # Adjust based on GPU capability
    )
    return llm
model_path='/content/mistral-7b-instruct-v0.1.Q4_K_S.gguf'
llm = init_model(model_path)

def get_response(llm, prompt_question):
    response = llm.create_chat_completion(
        messages=[
            {"role": "system", "content": "You are a Virtual Assistant who can take instructions and work based on those instructions."},
            {"role": "user", "content": prompt_question}
        ]
    )
    return response['choices'][0]['message']['content']

# --- Resume Parsing ---
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_doc(doc_path):
    doc = docx.Document(doc_path)
    return "\n".join([para.text for para in doc.paragraphs])

def extract_text_from_image(image_path):
    image = Image.open(image_path)
    text = pytesseract.image_to_string(image)
    return text

def extract_skills(text):
    doc = nlp(text)
    skills = [token.text.lower() for token in doc if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 2]
    return list(set(skills))

# --- Connect linked-In ---
def get_selenium_driver():
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    service = Service("/usr/bin/chromedriver")
    driver = webdriver.Chrome(service=service, options=options)
    return driver

def scrape_linkedin(username, password):
    driver = get_selenium_driver()
    driver.get("https://www.linkedin.com/login")
    time.sleep(2)
    driver.find_element("id", "username").send_keys(username)
    driver.find_element("id", "password").send_keys(password)
    driver.find_element("xpath", "//button[@type='submit']").click()
    time.sleep(5)
    driver.get(f"https://www.linkedin.com/in/{username}/")
    time.sleep(5)
    soup = BeautifulSoup(driver.page_source, "html.parser")
    driver.quit()
    experience = [exp.text.strip() for exp in soup.find_all("span", class_="t-bold")]
    skills = [skill.text.strip() for skill in soup.find_all("span", class_="t-normal")]
    return {"experience": experience, "skills": skills}

# --- MAP to Career Path ---
def map_to_career_paths(user_skills):
    career_paths = {
        "Data Scientist": ["python", "machine learning", "deep learning", "sql", "statistics", "pandas", "numpy"],
        "Software Engineer": ["java", "python", "c++", "git", "algorithms", "data structures"],
        "Web Developer": ["html", "css", "javascript", "react", "node.js", "express", "typescript"],
        "Cybersecurity Analyst": ["networking", "linux", "penetration testing", "encryption", "firewall", "threat analysis"],
        "AI/ML Engineer": ["python", "deep learning", "tensorflow", "pytorch", "nlp", "computer vision"]
    }
    
    ranked_paths = {}
    for path, skills in career_paths.items():
        matched_skills = len(set(user_skills) & set(skills))
        total_skills = len(skills)
        match_percentage = (matched_skills / total_skills) * 100
        if match_percentage > 20:
            ranked_paths[path] = match_percentage
    
    return sorted(ranked_paths.items(), key=lambda x: x[1], reverse=True)


def generate_learning_path(llm, user_skills, career_suggestions, self_entered_goals):
    suggested_career = career_suggestions[0][0] if career_suggestions else "General Tech Role"
    
    prompt = f"""
    The user wants to achieve the goal: {self_entered_goals}. 
    Based on their current skills: {', '.join(user_skills)}, Suggested careers: {', '.join([c[0] for c in career_suggestions])}
    
    Suggest a structured learning path with:
    - Beginner concepts to start with
    - Intermediate skills they should acquire next
    - Advanced topics to master
    - Recommended online courses/resources for learning
    """

    return get_response(llm, prompt)


# --- Main Execution ---
def main(resume_path, linkedin_user, linkedin_pass, self_entered_goals):
    if resume_path.endswith(".pdf"):
        resume_text = extract_text_from_pdf(resume_path)
    elif resume_path.endswith(".docx"):
        resume_text = extract_text_from_doc(resume_path)
    elif resume_path.endswith((".jpg", ".png")):
        resume_text = extract_text_from_image(resume_path)
    else:
        raise ValueError("Unsupported file format")
    
    resume_skills = extract_skills(resume_text)
    linkedin_data = scrape_linkedin(linkedin_user, linkedin_pass) if linkedin_user and linkedin_pass else {"skills": [], "experience": []}
    all_skills = list(set(resume_skills + linkedin_data["skills"]))
    career_suggestions = map_to_career_paths(all_skills)
    ai_prompts = generate_learning_path(llm, all_skills, career_suggestions, self_entered_goals)
    
    print("\n--- User Profile Summary ---")
    print(f"Extracted Skills: {', '.join(all_skills)}")
    print(f"Suggested Career Paths: {', '.join([c[0] for c in career_suggestions])}")
    print("\n--- AI Recommended Next Steps ---")
    print(ai_prompts)

# --- Execution ---
if __name__ == "__main__":
    resume_file = "sample_resume.pdf"
    linkedin_username = "user@example.com"
    linkedin_password = "password123"
    self_entered_goals = "Become a machine learning expert"
    main(resume_file, linkedin_username, linkedin_password, self_entered_goals)
