import streamlit as st
import cv2
import tempfile
import os
from PIL import Image
from ultralytics import YOLO

# Safety gear class list
ALL_CLASSES = [
    "Helmet", "Goggles", "Face Mask", "Gloves", "High-Visibility Vest",
    "Safety Harness",
]

# Map your model class names to the 10 standard ones (change as needed)
CLASS_MAPPING = {
    "Hardhat": "Helmet",
    "Safety Vest": "High-Visibility Vest",
    "Mask": "Face Mask",
    "Gloves": "Gloves",
    "Safety Harness": "Safety Harness",
    "Goggles": "Goggles",
}

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("Model/ppe.pt")  # Adjust path if needed

model = load_model()

# UI
st.title("🦺 Safety Gear Detection")
st.write("Upload an image to detect 10 standard safety items.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(uploaded_file.read())
        temp_path = temp_file.name

    # Load image using OpenCV
    image = cv2.imread(temp_path)
    results = model(image)[0]

    detected_items = set()

    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = model.names[class_id]
        mapped_name = CLASS_MAPPING.get(class_name)
        if mapped_name:
            detected_items.add(mapped_name)

    # Annotated image with bounding boxes
    annotated_image = results.plot()

    # Create two columns
    col1, col2 = st.columns([2, 1])  # Adjust ratio as needed

    with col1:
        st.image(annotated_image, caption="Detected Image", use_column_width=True)

    with col2:
        st.subheader("Detection Checklist")
        for item in ALL_CLASSES:
            if item in detected_items:
                st.markdown(f"✅ **{item}**")
            else:
                st.markdown(f"❌ **{item}**")

    # Clean up temp file
    os.remove(temp_path)
