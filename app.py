import streamlit as st
import numpy as np
import base64
import time
from openai import OpenAI
from pypdf import PdfReader

# --- CONFIGURATION ---
st.set_page_config(page_title="Ambo Press Focus Group", page_icon="⛪", layout="wide")

# --- THE PERSONAS (SPECIFIC INTERVIEW PROFILES) ---
PERSONAS = {
    "Anchored Traditionalist (Alejandra/Felipe Profile)": (
        "I am a young adult who chose the Episcopal Church specifically for 'Inclusive Orthodoxy.' "
        "Like the interviewees Alejandra and Felipe, I love the ritual, the Eucharist, "
        "and the tradition, but I need it to be fully LGBTQ+ inclusive. "
        "I get annoyed when church feels too casual or 'low church.' "
        "I am here for the sacred mystery, not a coffee hour. "
        "I want to feel grounded in ancient practice."
    ),
    "Deconstructing Seeker (The 'Hypocrisy' Alert)": (
        "I am a young adult who feels like a hypocrite in church. "
        "I have some religious trauma and am hyper-vigilant about power dynamics. "
        "I am looking for a 'safe space' where I can be messy and honest. "
        "I hate 'marketing speak' or glossy production. "
        "If you try to sell me something, I will leave. I value vulnerability above all. "
        "I am asking: 'Are you safe? Are you real?'"
    ),
    "Community Pragmatist (The 'Third Place' Profile)": (
        "I am a young adult who is lonely and treats church as a 'third place.' "
        "My main questions are: 'Is it awkward?' and 'Will I make friends?' "
        "I judge a church by the warmth of the welcome and the quality of the snacks. "
        "I am less interested in high theology and more interested in small groups, "
        "dinners, and belonging to a tribe. I am overwhelmed and need connection."
    ),
    "Intellectual Explorer (The Theological Profile)": (
        "I am a young adult who approaches faith through the head. "
        "I am bored by simple answers. I want to wrestle with difficult texts, "
        "social justice issues, and complex ethics. "
        "I respect the Episcopal Church because I don't have to check my brain at the door. "
        "I want a sermon that challenges me intellectually. Don't dumb it down."
    )
}

# --- THE SCALE ---
ANCHORS = [
    "I hate this. It feels manipulative, irrelevant, or offensive.",  # Score 1
    "I dislike this. It doesn't resonate with me.",                  # Score 2
    "I am neutral. It's fine, but I would ignore it.",               # Score 3
    "I like this. It catches my interest.",                          # Score 4
    "I love this. It speaks directly to my soul and needs."          # Score 5
]

# --- HELPER FUNCTIONS ---

def calculate_similarity(vec1, vec2):
    """Manual Cosine Similarity (Replaces scikit-learn dependency)"""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

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
            return None 

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
    
    # 1. Embed the Anchors
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
    
    progress_text = "Running deep analysis (30 iterations per persona)... This may take 2-3 minutes."
    my_bar = st.progress(0, text=progress_text)
    
    total_personas = len(PERSONAS)
    current_step = 0
    
    # 2. Loop through each Persona
    for name, bio in PERSONAS.items():
        
        scores = []
        texts = [] 
        
        # --- THE SCIENTIFIC LOOP (30 Iterations) ---
        ITERATIONS = 30 
        
        for i in range(ITERATIONS):
            
            # Construct Prompt
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
                
                img_prompt = (
                    f"You are this person: {bio}. "
                    f"Look at this image. {specific_instruction} "
                    "Write 2-3 sentences about your honest, gut reaction. Don't be polite. Be real."
                )

                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": img_prompt},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ]

            # --- RETRY LOGIC ---
            max_retries = 3
            reaction_text = "Error"
            
            for attempt in range(max_retries):
                try:
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages
                    )
                    reaction_text = completion.choices[0].message.content
                    break 
                except Exception as e:
                    time.sleep(2) 
            
            # Embed the reaction & Calculate Score Manually
            if reaction_text != "Error":
                reaction_embedding = get_embedding(reaction_text, client)
                if reaction_embedding:
                    # MANUAL MATH (No Scikit-Learn required)
                    sims = [calculate_similarity(reaction_embedding, anchor) for anchor in anchor_embeddings]
                    best_match_index = np.argmax(sims)
                    score = best_match_index + 1
                    
                    scores.append(score)
                    texts.append(reaction_text)
            
            # PAUSE
            time.sleep(1.0)
        
        if scores:
            avg_score = sum(scores) / len(scores)
            results[name] = {"text": texts[0], "score": avg_score}
        else:
            results[name] = {"text": "Could not generate response.", "score": 0}
        
        current_step += 1
        my_bar.progress(current_step / total_personas, text=f"Finished interviewing {name}...")
        
    my_bar.empty()
    return results

# --- THE APP INTERFACE ---

st.title("⛪ Ambo Press Focus Group")
st.markdown("Test your book titles, sermon series, and flyers against **4 Synthetic Personas**.")

# 1. API Key Check
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

# 2. Input Method
tab1, tab2 = st.tabs(["📝 Text Input", "Pg Upload File"])

input_payload = None
input_type = "text"

with tab1:
    text_input = st.text_area("Paste text to test:", height=150, placeholder="e.g., Book Title: 'The Holy Grind'")
    if text_input:
        input_payload = text_input
        input_type = "text"

with tab2:
    uploaded_file = st.file_uploader("Upload an Image or PDF", type=['png', 'jpg', 'jpeg', 'pdf'])
    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            st.info("Extracting text from PDF...")
            input_payload = extract_text_from_pdf(uploaded_file)
            input_type = "text" 
            st.success("PDF Text Extracted!")
            with st.expander("View Extracted Text"):
                st.write(input_payload[:500] + "...")
        else:
            input_payload = uploaded_file
            input_type = "image"
            st.image(uploaded_file, caption="Preview", width=300)

# 3. Moderator Guidance
st.divider()
guidance = st.text_input("Moderator Guidance (Optional)", placeholder="e.g., 'Focus on the second paragraph' or 'Ignore the color scheme'")

# 4. Run Button
if st.button("Run Focus Group (Deep Analysis)", type="primary"):
    if not api_key:
        st.error("No API Key found. Please set it in Streamlit Secrets or the sidebar.")
    elif not input_payload:
        st.error("Please enter text or upload a file first.")
    else:
        with st.spinner("Running 30 simulations per persona. This will take a few minutes..."):
            data = run_focus_group(api_key, input_payload, input_type, guidance)
            
            st.divider()
            
            # Display Results
            cols = st.columns(2)
            for i, (name, res) in enumerate(data.items()):
                with cols[i % 2]:
                    with st.container():
                        st.subheader(f"{name}")
                        
                        score = res['score']
                        if score < 2.5:
                            color = "red"
                        elif score < 3.5:
                            color = "orange"
                        else:
                            color = "green"
                            
                        st.markdown(f"### Resonance: :{color}[{score:.1f}/5]")
                        
                        st.info(f"_{res['text']}_")
                        st.divider()
