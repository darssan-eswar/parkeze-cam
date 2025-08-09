import os
import json
import requests
from bs4 import BeautifulSoup
import base64
from PIL import Image, ImageDraw
from io import BytesIO
import streamlit as st
import time

# ==== SETTINGS & CONFIGURATION ====
BASE_URL = "https://api.mistall.com/v3/frame/5113/Cam{}"
HEADERS = {"User-Agent": "Mozilla/5.0"}
IMAGES_DIR = "images"
LOT_MAP_PATH = "lot_map_cartoon.png"
SPOTS_FILE = "parking_spots.json"

HF_API_URL = "https://api-inference.huggingface.co/models/facebook/detr-resnet-50"

try:
    HF_API_TOKEN = st.secrets["HF_API_TOKEN"]
except FileNotFoundError:
    st.error("Secrets file not found. Please create a .streamlit/secrets.toml file with your HF_API_TOKEN.")
    st.stop()
except KeyError:
    st.error("HF_API_TOKEN not found in secrets.toml. Please add it.")
    st.stop()

def load_spots_config(file_path):
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Failed to load spots config: {e}")
        st.stop()

def fetch_cam_image(cam_num):
    url = BASE_URL.format(cam_num)
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        st.warning(f"Could not fetch camera {cam_num} image. Error: {e}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")
    img_tag = soup.find("img")
    if img_tag and 'src' in img_tag.attrs and img_tag['src'].startswith("data:image"):
        base64_data = img_tag['src'].split(",")[1]
        image_data = base64.b64decode(base64_data)
        img = Image.open(BytesIO(image_data))
        path = os.path.join(IMAGES_DIR, f"cam{cam_num}_snapshot.png")
        img.save(path, "PNG")
        return path
    return None

def hf_detect_objects(image: Image.Image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "image/png",
    }
    try:
        response = requests.post(HF_API_URL, headers=headers, data=img_bytes, timeout=30)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.warning(f"API request failed: {e}")
        try:
            st.warning(f"API Response Details: {response.json()}")
        except:
            pass
        return []

def check_overlap(box, polygon):
    box_center_x = (box['xmin'] + box['xmax']) / 2
    box_center_y = (box['ymin'] + box['ymax']) / 2

    num_vertices = len(polygon)
    is_inside = False
    p1x, p1y = polygon[0]
    for i in range(num_vertices + 1):
        p2x, p2y = polygon[i % num_vertices]
        if box_center_y > min(p1y, p2y) and box_center_y <= max(p1y, p2y) and box_center_x <= max(p1x, p2x):
            if p1y != p2y:
                x_intersection = (box_center_y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
            if p1x == p2x or box_center_x <= x_intersection:
                is_inside = not is_inside
        p1x, p1y = p2x, p2y
    return is_inside

def detect_occupancy(image_path, polygons):
    img = Image.open(image_path)
    all_detected_objects = hf_detect_objects(img)
    vehicle_labels = {"car", "truck", "bus", "motorcycle"}
    detected_vehicles = [
        obj for obj in all_detected_objects 
        if isinstance(obj, dict) and obj.get("label") in vehicle_labels and obj.get("score", 0) > 0.7
    ]
    if not detected_vehicles:
        return [False] * len(polygons)

    occupied_status = []
    for polygon in polygons:
        is_occupied = any(check_overlap(vehicle['box'], polygon) for vehicle in detected_vehicles)
        occupied_status.append(is_occupied)
    return occupied_status

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
            radius = 5
            rect = [x - width//2, y - height//2, x + width//2, y + height//2]
            color = (220, 40, 40, 180) if occ else (40, 220, 40, 180)
            draw.rounded_rectangle(rect, radius=radius, fill=color)

    return lot_map

def run_status_check(spots_data):
    status = {}
    for cam_str, data in spots_data.items():
        cam_num = int(cam_str.replace("cam", ""))
        polygons = data.get("polygons", [])
        if not polygons:
            status[cam_str] = []
            continue
        cam_img_path = fetch_cam_image(cam_num)
        if cam_img_path:
            status[cam_str] = detect_occupancy(cam_img_path, polygons)
        else:
            status[cam_str] = [False] * len(polygons)
    return status

def main():
    st.set_page_config(page_title="Golden Spur Parking", layout="centered")
    st.title("Golden Spur Lot - USC Columbia")
    st.caption("Green = Available, Red = Occupied.")

    os.makedirs(IMAGES_DIR, exist_ok=True)
    spots_data = load_spots_config(SPOTS_FILE)

    # Controls
    auto_update = st.checkbox("Enable automatic updates every 60 seconds")
    check_status = st.button("Check Status Now")

    placeholder = st.empty()

    # Run status check once on button press
    if check_status:
        with st.spinner("Checking parking status..."):
            status = run_status_check(spots_data)
            lot_image = draw_lot_map(status, spots_data)
            placeholder.image(lot_image, caption=f"Status at {time.strftime('%I:%M:%S %p')}")
    
    # Run automatic update if enabled
    if auto_update:
        while True:
            with st.spinner("Automatically updating parking status..."):
                status = run_status_check(spots_data)
                lot_image = draw_lot_map(status, spots_data)
                placeholder.image(lot_image, caption=f"Live status as of {time.strftime('%I:%M:%S %p')}")
            time.sleep(60)
    else:
        st.info("Press 'Check Status Now' to fetch current parking availability.")

if __name__ == "__main__":
    main()
