import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from inference_sdk import InferenceHTTPClient
from ultralytics import YOLO
from collections import defaultdict
import time

# Roboflow API Setup
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="lV3Y7kLo8n2rWathCnhE"
)

@st.cache_resource
def load_yolo_model():
    return YOLO("Model/ppe.pt")

yolo_model = load_yolo_model()

# Unified Class Mapping
CLASS_MAPPING = {
    "Helmet": "Helmet", "Hardhat": "Helmet", "capacete": "Helmet",
    "NO-Hardhat": "No Helmet Detected", "No Helmet": "No Helmet Detected",
    "Goggles": "Goggles", "oculos": "Goggles",
    "High-Visibility Vest": "Safety Vest", "Safety Vest": "Safety Vest",
    "NO-Safety Vest": "No Safety Vest Detected", "No Vest": "No Safety Vest Detected",
    "Mask": "Face Mask", "Face Mask": "Face Mask",
    "NO-Mask": "No Mask Detected", "No Mask": "No Mask Detected",
    "Gloves": "Gloves", "luvas": "Gloves",
    "Harness": "Safety Harness", "Safety Harness": "Safety Harness",
    "Safety Shoes": "Safety Shoes", "Boots": "Safety Shoes", "bota": "Safety Shoes",
    "Ear Protection": "Hearing Protection", "Earmuffs": "Hearing Protection",
    "Earplugs": "Hearing Protection", "hearing_protection": "Hearing Protection",
    "FR clothing": "Fire-Resistant Clothing", "Fire Resistant": "Fire-Resistant Clothing",
    "fire_protection": "Fire-Resistant Clothing", "protective clothing": "Fire-Resistant Clothing",
    "Anti-flame Suit": "Fire-Resistant Clothing",
    "insulated_tools": "Electrical Safety Equipment", "rubber_mat": "Electrical Safety Equipment",
    "electrical_gear": "Electrical Safety Equipment", "electrical_protection": "Electrical Safety Equipment",
    "electrical_safety": "Electrical Safety Equipment",
    "Person": "Person", "pessoa": "Person"
}

COMMON_GEAR_ITEMS = [
    "Helmet", "Goggles", "Safety Vest", "Face Mask", 
    "Safety Harness", "Safety Shoes"
]

# Draw predictions
def draw_predictions(image, predictions, color):
    for pred in predictions:
        x, y, w, h = map(int, [pred["x"], pred["y"], pred["width"], pred["height"]])
        label = pred["class"]
        conf = pred["confidence"]
        top_left = (x - w // 2, y - h // 2)
        bottom_right = (x + w // 2, y + h // 2)
        cv2.rectangle(image, top_left, bottom_right, color, 2)
        cv2.putText(image, f"{label} {conf:.2f}", (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return image

# Streamlit UI
st.title("🛡 Safety Gear Detection (YOLO + Roboflow)")
st.write("Upload an image or video to detect common safety gear.")

file_type = st.radio("Choose input type:", ["Image", "Video"])

# ------------------ IMAGE HANDLING -------------------
if file_type == "Image":
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_path = temp_file.name

        image = cv2.imread(temp_path)
        gear_confidences = defaultdict(float)

        # YOLO
        results = yolo_model(image)[0]
        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = yolo_model.names[class_id]
            mapped_name = CLASS_MAPPING.get(class_name, class_name)
            gear_confidences[mapped_name] = max(gear_confidences[mapped_name], conf)

        combined_img = image.copy()
        rf_models = [("deteccao-de-epi-v3/4", (0, 255, 0)), ("harness-knfmk/3", (0, 0, 255))]
        for model_id, color in rf_models:
            result = CLIENT.infer(temp_path, model_id=model_id)
            combined_img = draw_predictions(combined_img, result["predictions"], color)
            for pred in result["predictions"]:
                raw_class = pred["class"]
                mapped_name = CLASS_MAPPING.get(raw_class, raw_class)
                conf = pred["confidence"]
                gear_confidences[mapped_name] = max(gear_confidences[mapped_name], conf)

        st.subheader("🖼 Combined Detection Output")
        st.image(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB), caption="Combined Output", use_column_width=True)

        st.subheader("✅ Final PPE Checklist")
        threshold = 0.1
        for item in COMMON_GEAR_ITEMS:
            conf = gear_confidences.get(item, 0.0)
            if conf > threshold:
                st.markdown(f"✅ **{item}** — *{conf*100:.1f}%* confidence")
            else:
                st.markdown(f"❌ **{item}** — Not detected")

        os.remove(temp_path)

# ------------------ VIDEO HANDLING -------------------
elif file_type == "Video":
    uploaded_video = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
            tfile.write(uploaded_video.read())
            video_path = tfile.name

        cap = cv2.VideoCapture(video_path)
        gear_confidences = defaultdict(float)
        frame_count = 0
        selected_frame_interval = 15

        stframe = st.empty()
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % selected_frame_interval != 0:
                continue

            resized_frame = cv2.resize(frame, (640, 480))

            # YOLO detection
            results = yolo_model(resized_frame)[0]
            for box in results.boxes:
                class_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = yolo_model.names[class_id]
                mapped_name = CLASS_MAPPING.get(class_name, class_name)
                gear_confidences[mapped_name] = max(gear_confidences[mapped_name], conf)

            result_img = resized_frame.copy()

            # Roboflow detection
            temp_img_path = os.path.join(tempfile.gettempdir(), "frame.jpg")
            cv2.imwrite(temp_img_path, resized_frame)
            time.sleep(0.05)  # ensure file write completes

            for model_id, color in [("deteccao-de-epi-v3/4", (0, 255, 0)), ("harness-knfmk/3", (255, 0, 0))]:
                try:
                    result = CLIENT.infer(temp_img_path, model_id=model_id)
                    result_img = draw_predictions(result_img, result["predictions"], color)
                    for pred in result["predictions"]:
                        raw_class = pred["class"]
                        mapped_name = CLASS_MAPPING.get(raw_class, raw_class)
                        conf = pred["confidence"]
                        gear_confidences[mapped_name] = max(gear_confidences[mapped_name], conf)
                except Exception as e:
                    st.warning(f"⚠️ Roboflow error on frame {frame_count}: {e}")

            stframe.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)

        cap.release()
        os.remove(video_path)

        st.subheader("✅ Final PPE Checklist (from Video)")
        threshold = 0.1
        for item in COMMON_GEAR_ITEMS:
            conf = gear_confidences.get(item, 0.0)
            if conf > threshold:
                st.markdown(f"✅ **{item}** — *{conf*100:.1f}%* confidence")
            else:
                st.markdown(f"❌ **{item}** — Not detected")
