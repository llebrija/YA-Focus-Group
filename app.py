import streamlit as st
import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIGURATION (You will enter your API Key in the web interface) ---
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

def run_focus_group(api_key, proposal_text):
    client = OpenAI(api_key=api_key)
    
    # 1. Embed the Anchors (The Ruler)
    anchor_embeddings = [get_embedding(a, client) for a in ANCHORS]
    
    results = {}
    
    # 2. Loop through each Persona
    for name, bio in PERSONAS.items():
        # Step A: Ask the Persona to "Think Out Loud"
        prompt = f"""
        You are this person: "{bio}"
        
        Read this proposal from a church: "{proposal_text}"
        
        Write 2-3 sentences about your honest, gut reaction. 
        Don't be polite. Be real. How does this make you feel?
        """
        
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a focus group participant."},
                      {"role": "user", "content": prompt}]
        )
        reaction_text = completion.choices[0].message.content
        
        # Step B: Score the Reaction (Math)
        reaction_embedding = get_embedding(reaction_text, client)
        
        # Calculate similarity to all 5 anchors
        similarities = cosine_similarity([reaction_embedding], anchor_embeddings)[0]
        
        # The "Score" is a weighted average of the anchor values (1-5) based on similarity
        # This is a simplified version of the paper's math for stability
        best_match_index = np.argmax(similarities)
        score = best_match_index + 1  # Convert 0-4 index to 1-5 score
        
        results[name] = {"text": reaction_text, "score": score}
        
    return results

# --- THE APP INTERFACE ---
st.title("⛪ Ambo Press Focus Group")
st.markdown("Test your book titles, sermon series, and emails against **4 Synthetic Personas**.")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("Enter OpenAI API Key", type="password")
    st.info("You need an OpenAI API Key to run the focus group.")

proposal = st.text_area("What do you want to test?", height=150, placeholder="e.g., Book Title: 'The Holy Grind: Finding God in the 9-to-5'")

if st.button("Run Focus Group"):
    if not api_key:
        st.error("Please enter an API Key in the sidebar.")
    elif not proposal:
        st.error("Please enter some text to test.")
    else:
        with st.spinner("The focus group is reviewing your idea..."):
            data = run_focus_group(api_key, proposal)
            
            st.divider()
            
            # Display Results
            cols = st.columns(2)
            
            for i, (name, res) in enumerate(data.items()):
                with cols[i % 2]:
                    st.subheader(f"{name}")
                    
                    # Color code the score
                    score = res['score']
                    color = "red" if score < 3 else "orange" if score == 3 else "green"
                    st.markdown(f"**Resonance Score:** :{color}[{score}/5]")
                    
                    st.info(f"_{res['text']}_")
                    st.divider()
