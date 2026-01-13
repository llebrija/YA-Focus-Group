import streamlit as st
import numpy as np
import base64
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from pypdf import PdfReader

# --- CONFIGURATION ---
st.set_page_config(page_title="TRI Young Adult Focus Group", page_icon="⛪")

# --- THE PERSONAS (Based on 12 Young Adult Interviews) ---
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

def get_embedding(text, client):
    """Get the vector embedding for a text string."""
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

def extract_text_from_pdf(uploaded_file):
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

import time  # <--- Add this import at the very top of your file with the other imports!

# ... (keep your existing imports and configuration) ...

def run_focus_group(api_key, input_data, input_type="text"):
    client = OpenAI(api_key=api_key)
    
    # 1. Embed the Anchors (The Ruler)
    anchor_embeddings = [get_embedding(a, client) for a in ANCHORS]
    
    results = {}
    
    progress_text = "Interviewing the focus group (Running safely to avoid rate limits)..."
    my_bar = st.progress(0, text=progress_text)
    
    total_personas = len(PERSONAS)
    current_step = 0
    
    # 2. Loop through each Persona
    for name, bio in PERSONAS.items():
        
        scores = []
        texts = [] 
        
        # --- THE SCIENTIFIC LOOP (Reduced to 10 for safety, adjust up if you have higher tier) ---
        ITERATIONS = 10  # Reduced from 30 to 10 to prevent crashing Tier 1 accounts
        
        for i in range(ITERATIONS):
            
            # Prepare the Prompt based on Input Type
            if input_type == "text":
                prompt_content = f"""
                You are this person: "{bio}"
                Read this proposal from a church: "{input_data}"
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
                            {"type": "text", "text": f"You are this person: {bio}. Look at this image from a church. Write 2-3 sentences about your honest, gut reaction. Don't be polite. Be real."},
                            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                        ]
                    }
                ]

            # --- SAFETY BLOCK: Retry if Rate Limited ---
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages
                    )
                    reaction_text = completion.choices[0].message.content
                    break # Success! Exit the retry loop
                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(2) # Wait 2 seconds before retrying
                    else:
                        reaction_text = "Error: Could not get response."
            
            # Score the Reaction
            reaction_embedding = get_embedding(reaction_text, client)
            similarities = cosine_similarity([reaction_embedding], anchor_embeddings)[0]
            best_match_index = np.argmax(similarities)
            score = best_match_index + 1
            
            scores.append(score)
            texts.append(reaction_text)
            
            # --- PAUSE ---
            # Sleep for 0.5 seconds between every single request to stay under the speed limit
            time.sleep(0.5)
        
        avg_score = sum(scores) / len(scores)
        results[name] = {"text": texts[0], "score": avg_score}
        
        current_step += 1
        my_bar.progress(current_step / total_personas, text=f"Finished interviewing {name}...")
        
    my_bar.empty()
    return results

# --- THE APP INTERFACE ---
st.title("⛪ TRI YA Focus Group")
st.markdown("Test your book titles, sermon series, and flyers against **4 Synthetic Personas**.")

# 1. API Key Check
if "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]
else:
    api_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")

# 2. Input Method (Tabs)
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
            input_type = "text" # Treat extracted PDF content as text
            st.success("PDF Text Extracted!")
            with st.expander("View Extracted Text"):
                st.write(input_payload[:500] + "...")
        else:
            input_payload = uploaded_file
            input_type = "image"
            st.image(uploaded_file, caption="Preview", width=300)

if st.button("Run Focus Group"):
    if not api_key:
        st.error("No API Key found. Please set it in Streamlit Secrets or the sidebar.")
    elif not input_payload:
        st.error("Please enter text or upload a file.")
    else:
        with st.spinner("The focus group is reviewing your submission..."):
            data = run_focus_group(api_key, input_payload, input_type)
            
            st.divider()
            
            # Display Results
            cols = st.columns(2)
            for i, (name, res) in enumerate(data.items()):
                with cols[i % 2]:
                    st.subheader(f"{name}")
                    
                    score = res['score']
                    color = "red" if score < 3 else "orange" if score == 3 else "green"
                    st.markdown(f"**Resonance Score:** :{color}[{score:.1f}/5]")
                    
                    st.info(f"_{res['text']}_")
                    st.divider()
