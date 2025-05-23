from transformers import Blip2Processor, Blip2ForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import pandas as pd
import torch
import os
import numpy as np
from moviepy.editor import *
from gtts import gTTS
from moviepy.editor import ImageClip
from PIL import Image, ImageDraw, ImageFont
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip, ColorClip
import moviepy.config as mpy_config
mpy_config.change_settings({"IMAGEMAGICK_BINARY": r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"})

os.environ["IMAGEMAGICK_PATH"] = r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"
# --- Load BLIP-2 Model ---
print("🔧 Loading BLIP-2 image captioning model...")
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# --- Load Sentence Transformer ---
print("🔧 Loading Sentence Transformer...")
sbert = SentenceTransformer("all-MiniLM-L6-v2")

# --- Load Facts ---
print("📂 Loading facts from Facts.csv...")
df = pd.read_csv("Facts.csv")
fact_embeddings = sbert.encode(df['fact'].tolist(), convert_to_tensor=True)

# --- Generate Caption from Image ---
def generate_caption(image_path):
    try:
        image = Image.open(image_path).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        generated_ids = model.generate(**inputs, max_new_tokens=100)
        caption = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        print(f"❌ Error processing image {image_path}: {e}")
        return None

def create_video_with_narration(text, output_file="output.mp4"):
    # Generate speech audio
    tts = gTTS(text)
    tts.save("narration.mp3")

    # Create background clip (solid color)
    audio_clip = AudioFileClip("narration.mp3")
    duration = audio_clip.duration

    # Create a blank image with text using PIL
    img = Image.new('RGB', (1280, 720), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("arial.ttf", 40)  # Make sure you have this font or change to a path of .ttf
    draw.multiline_text((50, 50), text, fill="white", font=font)

    # Convert PIL image to ImageClip
    img_clip = ImageClip(np.array(img)).set_duration(duration)

    # Composite video with audio
    video = CompositeVideoClip([img_clip]).set_audio(audio_clip)
    video.write_videofile(output_file, fps=24)

# --- Match to Most Similar Fact ---
def find_most_similar_fact(text):
    input_embedding = sbert.encode(text, convert_to_tensor=True)
    similarities = util.cos_sim(input_embedding, fact_embeddings)[0]
    top_idx = int(similarities.topk(k=1)[1][0])

    matched_fact = df['fact'].iloc[top_idx]
    matched_reasoning = df['reasoning'].iloc[top_idx]
    score = similarities[top_idx].item()

    return {
        "input": text,
        "fact": matched_fact,
        "reasoning": matched_reasoning,
        "score": round(score, 4)
    }

# --- Process Mixed Inputs ---
def process_inputs(inputs):
    results = []
    for item in inputs:
        if os.path.exists(item) and item.lower().endswith((".png", ".jpg", ".jpeg")):
            print(f"🖼️ Processing image: {item}")
            caption = generate_caption(item)
            if caption:
                fact_info = find_most_similar_fact(caption)
                fact_info["interpreted_caption"] = caption
                results.append(fact_info)
        else:
            print(f"📝 Processing text: {item}")
            fact_info = find_most_similar_fact(item)
            results.append(fact_info)

    return results

# --- Print Combined Output ---
def print_combined_output(results):
    print("\n📊 Combined Analysis Result:\n")
    for i, res in enumerate(results, 1):
        print(f"--- Input #{i} ---")
        if "interpreted_caption" in res:
            print(f"🖼️ Image Caption   : {res['interpreted_caption']}")
        print(f"📥 Input           : {res['input']}")
        print(f"🔍 Matched Fact    : {res['fact']}")
        print(f"🧠 Reasoning       : {res['reasoning']}")
        print(f"📈 Similarity Score: {res['score']}\n")

# --- Individual Scenario Output ---
def generate_individual_scenarios(results):
    print("\n📖 Individual Scenarios:\n")
    for i, res in enumerate(results, 1):
        if res["score"] >= 0.5:
            print(f"Scenario {i}: {res['input']} suggests that {res['reasoning'].lower()}.")
        else:
            print(f"Scenario {i}: Observation '{res['input']}' noted, but no strong inference available.")

# --- Create video from narrative text ---
def narrative_to_text(results):
    story_parts = {
        "before": [],
        "during": [],
        "after": [],
        "misc": []
    }

    keyword_mapping = {
        "before": ["stationary", "waiting", "idle", "standing", "sitting", "asleep", "corner", "pillar","leaning", "ambush", "stalking", "approach", "shadow", "concealed", "observed", "positioned","footprints", "lured", "planned", "targeted", "near door", "unaware", "routine", "browsing","monitoring", "distraction", "unauthorized access", "preparation"],
        "during": ["gunshot", "stab", "shooting", "attack", "assault", "blunt force", "hammer", "weapon", "knife","high-velocity", "projectile", "entry wound", "exit wound", "impact", "blow", "strike", "blood spatter", "arterial spray", "struggle", "fight", "scream", "witnessed", "collision","aggression", "explosion", "burn", "slash", "multiple wounds", "gunfire", "throat slit"],        
        "after": ["glass", "escape", "flee", "ran", "broken", "window", "shattered", "bullet hole", "trail"],
    }

    for res in results:
        score = res["score"]
        reasoning = res["reasoning"].lower()
        input_text = res["input"]

        if score < 0.5:
            continue

        categorized = False
        for phase, keywords in keyword_mapping.items():
            if any(keyword in reasoning for keyword in keywords):
                story_parts[phase].append(f"{input_text} suggests that {reasoning}.")
                categorized = True
                break
        if not categorized:
            story_parts["misc"].append(f"{input_text} suggests that {reasoning}.")

    narrative = []

    if story_parts["before"]:
        narrative.append("Before the Incident: " + " ".join(story_parts["before"]))
    if story_parts["during"]:
        narrative.append("During the Incident: " + " ".join(story_parts["during"]))
    if story_parts["after"]:
        narrative.append("After the Incident: " + " ".join(story_parts["after"]))
    if story_parts["misc"]:
        narrative.append("Additional Observations: " + " ".join(story_parts["misc"]))

    return "\n\n".join(narrative) if narrative else "No strong narrative could be formed."

# --- Updated generate_crime_story function ---
def generate_crime_story(results):
    print("\n🕵️ Narrative: How the Crime Likely Happened\n")

    narrative_text = narrative_to_text(results)
    print(narrative_text + "\n")

    # After printing narrative, create video from this text narration
    print("🎬 Generating video narration from the crime story...")
    create_video_with_narration(narrative_text, output_file="crime_story.mp4")
    print("✅ Video saved as 'crime_story.mp4'\n")


# --- Unified Scenario Generation ---
def generate_unified_scenario(results):
    print("\n🧩 Unified Scenario Based on All Inputs:\n")
    scenario_parts = []
    observations = []

    for res in results:
        if res["score"] >= 0.5:
            part = f"{res['input']} suggests: {res['reasoning'].lower()}"
            scenario_parts.append(part)
        else:
            observations.append(f"Observation noted: '{res['input']}'")

    if scenario_parts:
        for part in scenario_parts:
            print("🔹", part)
    if observations:
        print("\n📌 Additional Observations:")
        for obs in observations:
            print("🔸", obs)

# --- Entry Point ---
if __name__ == "__main__":
    print("\n🧠 Multi-modal Interpreter (Text + Image)\n")
    print("Enter multiple inputs (text or image paths) separated by commas:")
    user_input = input("👉 ").strip()

    inputs = [i.strip() for i in user_input.split(",") if i.strip()]
    if not inputs:
        print("❌ No valid input provided.")
    else:
        results = process_inputs(inputs)
        print_combined_output(results)
        generate_individual_scenarios(results)
        generate_unified_scenario(results)
        generate_crime_story(results) 
