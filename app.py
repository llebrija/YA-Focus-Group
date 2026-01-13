import streamlit as st
import numpy as np
import base64
import time
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader

# --- CONFIGURATION ---
st.set_page_config(page_title="Ambo Press Focus Group", page_icon="⛪", layout="wide")

# --- THE PERSONAS (Based on your 12 Interviews) ---
PERSONAS = {
    "Anchored Traditionalist (The Alejandra/Felipe Profile)": (
        "I am a young adult who chose the Episcopal Church for 'Inclusive Orthodoxy.' "
        "Like the interviewees Alejandra and Felipe, I love the ritual, the Eucharist, "
        "and the tradition, but I need it to be fully LGBTQ+ inclusive. "
        "I get annoyed when church feels too casual or 'low church.' "
        "I am here for the sacred mystery, not a coffee hour."
    ),
    "Deconstructing Seeker (The 'Safe Space' Profile)": (
        "I am a young adult who feels like a hypocrite in church. "
        "I have some religious trauma and am hyper-vigilant about power dynamics. "
        "I am looking for a 'safe space' where I can be messy and honest. "
        "I hate 'marketing speak' or glossy production. "
        "If you try to sell me something, I will leave. I value vulnerability above all."
    ),
    "Community Pragmatist (The Social Profile)": (
        "I am a young adult who is lonely and treats church as a 'third place.' "
        "My main questions are: 'Is it awkward?' and 'Will I make friends?' "
        "I judge a church by the warmth of the welcome and the quality of the snacks. "
        "I am less interested in high theology and more interested in small groups, "
        "dinners, and belonging to a tribe."
    ),
    "Intellectual Explorer (The Theological Profile)": (
        "I am a young adult who approaches faith through the head. "
        "I am bored by simple answers. I want to wrestle with difficult texts, "
        "social justice issues, and complex ethics. "
        "I respect the Episcopal Church because I don't have to check my brain at the door. "
        "I want a sermon that challenges me intellectually."
    )
}

# --- THE SCALE (Reference Anchors) ---
ANCHORS = [
    "I hate this. It feels manipulative, irrelevant, or offensive.",  # Score 1
    "I dislike this. It doesn't resonate with me.",                  # Score 2
    "I am neutral. It's fine, but I would ignore it.",               # Score 3
    "I like this. It catches my interest.",                          # Score 4
    "I love this. It speaks directly to my soul and needs."          # Score 5
]

# --- HELPER FUNCTIONS ---

def get_embedding(text, client):
    """Safe embedding wrapper with retry logic"""
    try:
        text = text.replace("\n", " ")
        response = client.embeddings.create(input=[text], model="text-embedding-3-small")
        return response.data[0].embedding
    except Exception as e:
        time.sleep(2) # Wait if error
        try:
            response = client.embeddings.create(input=[text], model="text-embedding-3-small")
            return response.data[0].embedding
        except:
            return None # Return None if it fails twice

def extract_text_from_pdf(uploaded_file):
    try:
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    except:
        return "Error reading PDF."

def run_focus_group(api_key, input_data, input_type, guidance=""):
    client = OpenAI(api_key=api_key)
    
    # 1. Embed the Anchors (BATCHED for safety)
    try:
        anchor_response = client.embeddings.create(
            input=ANCHORS, 
            model="text-embedding-3-small"
        )
        anchor_embeddings = [d.embedding for d in anchor_response.data]
    except Exception as e:
        st.error(f"Error connecting to OpenAI: {e}")
        return {} 
    
    results = {}
    
    progress_text = "Interviewing the focus group (Running safely to avoid rate limits)..."
    my_bar = st.progress(0, text=progress_text)
    
    total_personas = len(PERSONAS)
    current_step = 0
    
    # 2. Loop through each Persona
    for name, bio in PERSONAS.items():
        
        scores = []
        texts = [] 
        
        # --- THE SCIENTIFIC LOOP (10 Iterations) ---
        ITERATIONS = 10 
        
        for i in range(ITERATIONS):
            
            # Construct Prompt with User Guidance
            specific_instruction = f"Additional Instruction from Moderator: {guidance}" if guidance else ""
            
            if input_type == "text":
                prompt_content = f"""
                You are this person: "{bio}"
                
                I am showing you a proposal/text from a church.
                {specific_instruction}
                
                THE CONTENT:
                "{input_data}"
                
                TASK:
                Write 2-3 sentences about your honest, gut reaction. 
                Don't be polite. Be real. How does this make you feel?
                """
                messages = [{"role": "user", "content": prompt_content}]
                
            elif input_type == "image":
                base64_image = base64.b64encode(input_data.getvalue()).decode('utf-8')
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"You are this person: {bio}.
