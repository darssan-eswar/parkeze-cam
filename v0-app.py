import os
import json
import requests
from bs4 import BeautifulSoup
import base64
from PIL import Image, ImageDraw
from io import BytesIO
import streamlit as st
from transformers import pipeline
import time

# ==== SETTINGS ====
BASE_URL = "https://api.mistall.com/v3/frame/5113/Cam{}"
HEADERS = {"User-Agent": "Mozilla/5.0"}
IMAGES_DIR = "images"
LOT_MAP_PATH = "lot_map_cartoon.png"
SPOTS_FILE = "parking_spots.json"

# ==== UTILS ====
def polygon_to_bbox(polygon):
    xs, ys = zip(*polygon)
    return (min(xs), min(ys), max(xs), max(ys))

def load_spots(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def fetch_cam_image(cam_num, flip_horizontal=False):
    url = BASE_URL.format(cam_num)
    response = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(response.text, "html.parser")
    img_tag = soup.find("img")
    if img_tag and 'src' in img_tag.attrs and img_tag['src'].startswith("data:image"):
        base64_data = img_tag['src'].split(",")[1]
        image_data = base64.b64decode(base64_data)
        img = Image.open(BytesIO(image_data))
        if flip_horizontal:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        path = os.path.join(IMAGES_DIR, f"cam{cam_num}_snapshot.jpg")
        img.save(path, "JPEG")
        return path
    return None

def detect_occupancy(image_path, polygons):
    img = Image.open(image_path)
    occupied = []
    for polygon in polygons:
        bbox = polygon_to_bbox(polygon)
        spot_img = img.crop(bbox)
        spot_results = detector(spot_img)
        has_car = any(r["label"] in ["car", "truck", "bus"] and r["score"] > 0.5 for r in spot_results)
        occupied.append(has_car)
    return occupied

def draw_rounded_rect(draw, xy, radius, fill):
    draw.rounded_rectangle(xy, radius=radius, fill=fill)

def draw_lot_map(status, spots_data):
    lot_map = Image.open(LOT_MAP_PATH).convert("RGBA")
    draw = ImageDraw.Draw(lot_map, "RGBA")

    for cam, occupancy_list in status.items():
        map_spots = spots_data.get(cam, {}).get("map_spots", [])
        for idx, occ in enumerate(occupancy_list):
            if idx >= len(map_spots):
                continue
            x, y = map_spots[idx]
            width, height = 12, 28
            radius = 3
            rect = [x - width//2, y - height//2, x + width//2, y + height//2]
            if occ:
                color = (180, 0, 0, 100)  # translucent red
            else:
                color = (0, 180, 0, 100)  # translucent green
            draw_rounded_rect(draw, rect, radius, color)

    return lot_map

# ==== INIT ====
os.makedirs(IMAGES_DIR, exist_ok=True)
detector = pipeline("object-detection", model="hustvl/yolos-tiny")
spots_data = load_spots(SPOTS_FILE)

# ==== STREAMLIT UI ====
st.set_page_config(page_title="Golden Spur Parking", layout="centered")
st.title("Golden Spur Lot - USC Columbia")
st.caption("Green = free, Red = occupied. Updates every 60 seconds.")

placeholder = st.empty()

while True:
    status = {}
    for cam_str, data in spots_data.items():
        cam_num = int(cam_str[-1])  # assumes 'cam1', 'cam2', ...
        polygons = data.get("polygons", [])
        if not polygons:
            status[cam_str] = []
            continue

        cam_img_path = fetch_cam_image(cam_num)
        if cam_img_path:
            status[cam_str] = detect_occupancy(cam_img_path, polygons)
        else:
            status[cam_str] = []

    lot_image = draw_lot_map(status, spots_data)
    with placeholder.container():
        st.image(lot_image, caption="Live Parking Status")

    time.sleep(60)
