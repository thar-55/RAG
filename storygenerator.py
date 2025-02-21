import streamlit as st
from openai import OpenAI
import requests
from PIL import Image
from io import BytesIO
import os
# Initialize OpenAI client with Together.ai base URL
client = OpenAI(
    api_key=os.getenv('TOGETHER_API_KEY'),
    base_url="https://api.together.xyz/v1"
)

def generate_image(prompt: str):
    """Generate an image using FLUX model and return both image and URL"""
    try:
        response = client.images.generate(
            model="black-forest-labs/FLUX.1-schnell-Free",
            prompt=prompt,
        )
        # Get image URL from response
        image_url = response.data[0].url
        
        # Load image and return both image and URL
        response = requests.get(image_url)
        return Image.open(BytesIO(response.content)), image_url
    except Exception as e:
        st.error(f"Failed to generate image: {str(e)}")
        return None, None

def generate_story_outline(topic: str, num_paragraphs: int):
    """Generate a story outline with multiple paragraph prompts"""
    try:
        prompt = f"""Create {num_paragraphs} different scene descriptions for a story about {topic}. 
        Each scene should be unique and flow together to tell a coherent story.
        Format: Return only the numbered list of scene descriptions, one per line."""
        
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        # Split the response into separate scenes
        scenes = response.choices[0].message.content.strip().split('\n')
        return [scene.strip() for scene in scenes if scene.strip()]
    except Exception as e:
        st.error(f"Failed to generate story outline: {str(e)}")
        return None

def generate_paragraph(image_url: str, scene_description: str, paragraph_number: int):
    """Generate a single paragraph using Llama model with the image URL"""
    try:
        prompt = f"""Look at this image: {image_url}. 
        Write a detailed paragraph for part {paragraph_number} of the story based on this scene: {scene_description}.
        Make sure it flows well with the overall narrative."""
        
        response = client.chat.completions.create(
            model="meta-llama/Llama-3.2-90B-Vision-Instruct-Turbo",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Failed to generate paragraph: {str(e)}")
        return None

# Main app
st.title("🎨 Multi-Scene Story Generator")
st.write("Generate a story with multiple scenes and images!")

# Get user input
topic = st.text_input("What's your story about?", placeholder="e.g., A magical adventure in a forest")
num_paragraphs = st.slider("Number of paragraphs", min_value=2, max_value=5, value=3)

# Generate button
if st.button("Generate Story", type="primary"):
    if topic:
        with st.spinner("Creating your story..."):
            # Generate story outline first
            scene_descriptions = generate_story_outline(topic, num_paragraphs)
            
            if scene_descriptions:
                # Create story container
                story_container = st.container()
                
                with story_container:
                    st.write("## Your Story")
                    
                    # Generate each scene with image and paragraph
                    for i, scene in enumerate(scene_descriptions, 1):
                        st.write(f"### Scene {i}")
                        
                        # Generate image for this scene
                        image, image_url = generate_image(f"A scene about {scene}")
                        
                        if image and image_url:
                            # Display image
                            st.image(image, caption=f"Scene {i}")
                            
                            # Generate and display paragraph
                            paragraph = generate_paragraph(image_url, scene, i)
                            if paragraph:
                                st.write(paragraph)
                            
                            # Add spacing between scenes
                            st.write("---")
    else:
        st.warning("Please enter a topic first!")

# Add some helpful information at the bottom
st.sidebar.markdown("""
### How it works
1. Enter your story topic
2. Choose the number of paragraphs
3. Click Generate
4. Each paragraph will have:
   - A unique scene description
   - A generated image
   - A paragraph of story text
""")