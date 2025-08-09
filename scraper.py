import requests
from bs4 import BeautifulSoup
import base64
from PIL import Image
from io import BytesIO
import os

# Ensure output folder exists
os.makedirs("images", exist_ok=True)

base_url = "https://api.mistall.com/v3/frame/5113/Cam{}"
headers = {"User-Agent": "Mozilla/5.0"}

for cam_num in range(1, 5):
    url = base_url.format(cam_num)
    response = requests.get(url, headers=headers)
    html = response.text

    soup = BeautifulSoup(html, "html.parser")
    img_tag = soup.find("img")

    if img_tag and 'src' in img_tag.attrs and img_tag['src'].startswith("data:image"):
        # Decode base64 data
        base64_data = img_tag['src'].split(",")[1]
        image_data = base64.b64decode(base64_data)

        # Load image
        img = Image.open(BytesIO(image_data))

        # Flip horizontally for cam1
        if cam_num == 1:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        # Save image (overwrite existing file)
        output_path = f"images/cam{cam_num}_snapshot.jpg"
        img.save(output_path, "JPEG")
        print(f"Saved {output_path}")
    else:
        print(f"No base64 image found for Cam {cam_num}.")
