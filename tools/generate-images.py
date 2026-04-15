"""
Generate portfolio illustrations using OpenAI's image API.

Usage:
  export OPENAI_API_KEY="sk-..."
  python3 tools/generate-images.py

Saves images to assets/ directory. Edit the `images` dict to add/modify prompts.
"""

import os
import re
import base64

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    with open(os.path.expanduser("~/.bashrc")) as f:
        for line in f:
            m = re.match(r'export OPENAI_API_KEY="(.+)"', line)
            if m:
                api_key = m.group(1)
                break
    if not api_key:
        print("ERROR: Set OPENAI_API_KEY env var or add it to ~/.bashrc")
        exit(1)

os.environ["OPENAI_API_KEY"] = api_key

from openai import OpenAI
client = OpenAI()

STYLE = "Flat editorial illustration, warm color palette (cream, terracotta, amber, soft brown), clean lines, minimal detail, no text, white/cream background, suitable for a professional portfolio website."

images = {
    "a2-flow": f"{STYLE} A person sitting at a laptop having a conversation with an AI assistant shown as a chat interface. Speech bubbles going back and forth. One side shows a job posting document, the other side shows polished interview answers.",
    "a2-input-output": f"{STYLE} Three items flowing left to right: a resume document, a gear/processing symbol in the center, tailored interview response cards. Arrows connecting them. Shows AI transforming raw career info into interview preparation. No people.",
    "a3-network": f"{STYLE} A simple neural network diagram with three layers: input nodes on the left, hidden layer nodes in the middle, output nodes on the right. Connections shown as thin lines. Nodes are warm-colored circles (amber, terracotta, brown).",
    "a3-classification": f"{STYLE} Left side shows a simple photograph of a cat. An arrow points to a simplified neural network in the center. Right side shows a label tag that says Cat with a confidence bar. Demonstrates image classification.",
    "a4-pipeline": f"{STYLE} A horizontal pipeline diagram: raw messy data on the left (scattered shapes), flowing through cleaning stages in the middle (filter, organize), producing clean structured data on the right (neat rows). Arrows connecting each stage. No people.",
    "a4-good-bad-data": f"{STYLE} Split composition. Left half shows messy, incomplete, scattered data points with question marks and gaps. Right half shows clean, organized, complete data points neatly arranged. A dividing line down the middle.",
    "card-1": f"{STYLE} Landscape format. A vintage computer terminal fading into a modern glowing circuit brain. Conveys AI history. Minimal, works as a small card thumbnail.",
    "card-2": f"{STYLE} Landscape format. A laptop with chat bubbles and a briefcase beside it. Conveys AI-powered interview preparation. Minimal.",
    "card-3": f"{STYLE} Landscape format. Three layers of connected circles forming a simple neural network. Warm amber and terracotta nodes. Minimal.",
    "card-4": f"{STYLE} Landscape format. Scattered puzzle pieces on one side flowing into a complete organized grid on the other. Conveys data quality. Minimal.",
}

os.makedirs("assets", exist_ok=True)

for name, prompt in images.items():
    out_path = f"assets/{name}.png"
    if os.path.exists(out_path):
        print(f"Skipping {name} (already exists)")
        continue
    print(f"Generating {name}...")
    try:
        size = "1536x1024" if name.startswith("card-") else "1024x1024"
        resp = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            n=1,
            size=size,
            quality="high",
        )
        img_bytes = base64.b64decode(resp.data[0].b64_json)
        with open(out_path, "wb") as f:
            f.write(img_bytes)
        print(f"  Saved {out_path}")
    except Exception as e:
        print(f"  ERROR on {name}: {e}")

print("\nDone!")
