import os
import cv2
import tempfile
import numpy as np
import base64
from collections import defaultdict
from inference_sdk import InferenceHTTPClient
from ultralytics import YOLO

# Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="lV3Y7kLo8n2rWathCnhE"
)

# Class mapping
CLASS_MAPPING = {
    # Helmets
    "Helmet": "Helmet", "Hardhat": "Helmet", "capacete": "Helmet",
    "NO-Hardhat": "No Helmet Detected", "No Helmet": "No Helmet Detected",

    # Goggles
    "Goggles": "Goggles", "oculos": "Goggles",

    # Safety Vest
    "High-Visibility Vest": "Safety Vest", "Safety Vest": "Safety Vest",
    "NO-Safety Vest": "No Safety Vest Detected", "No Vest": "No Safety Vest Detected",

    # Face Mask
    "Mask": "Face Mask", "Face Mask": "Face Mask",
    "NO-Mask": "No Mask Detected", "No Mask": "No Mask Detected",

    # Gloves
    "Gloves": "Gloves", "glove": "Gloves", "hand_protection": "Gloves", "gloves": "Gloves",

    # Harness
    "Harness": "Safety Harness", "Safety Harness": "Safety Harness",

    # Safety Shoes
    "Safety Shoes": "Safety Shoes", "Boots": "Safety Shoes", "bota": "Safety Shoes",

    # Hearing Protection
    "Ear Protection": "Hearing Protection", "Earmuffs": "Hearing Protection",
    "Earplugs": "Hearing Protection", "hearing_protection": "Hearing Protection",

    # Fire-Resistant Clothing
    "FR clothing": "Fire-Resistant Clothing", "Fire Resistant": "Fire-Resistant Clothing",
    "fire_protection": "Fire-Resistant Clothing", "protective clothing": "Fire-Resistant Clothing",
    "Anti-flame Suit": "Fire-Resistant Clothing",

    # Electrical Safety
    "insulated_tools": "Electrical Safety Equipment", "rubber_mat": "Electrical Safety Equipment",
    "electrical_gear": "Electrical Safety Equipment", "electrical_protection": "Electrical Safety Equipment",
    "electrical_safety": "Electrical Safety Equipment",

    # Person
    "Person": "Person", "pessoa": "Person"
}

COMMON_GEAR_ITEMS = [
    "Helmet", "Goggles", "Safety Vest", "Face Mask",
    "Safety Harness", "Safety Shoes", "Gloves"
]

# Load YOLO once
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "ppe.pt")

yolo_model = YOLO(MODEL_PATH)

def detect_safety_gear(image_bytes):
    """
    Main function to run detection:
    - accepts image bytes
    - runs YOLO and Roboflow detections
    - returns detected items and annotated image base64
    """

    # Save bytes to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(image_bytes)
        temp_path = temp_file.name

    try:
        # Read image
        image = cv2.imread(temp_path)
        gear_confidences = defaultdict(float)

        ### YOLO detection
        results = yolo_model(image)[0]
        for box in results.boxes:
            class_id = int(box.cls[0])
            conf = float(box.conf[0])
            class_name = yolo_model.names[class_id]
            mapped_name = CLASS_MAPPING.get(class_name, class_name)
            gear_confidences[mapped_name] = max(gear_confidences[mapped_name], conf)

        # Roboflow detection
        rf_models = [
            ("deteccao-de-epi-v3/4", (0, 255, 0)),       # General PPE
            ("harness-knfmk/3", (0, 0, 255)),            # Harness
            ("ppe-detection-lnxkm/4", (255, 0, 0))       # Gloves
        ]

        combined_img = image.copy()

        for model_id, color in rf_models:
            result = CLIENT.infer(temp_path, model_id=model_id)
            for pred in result["predictions"]:
                raw_class = pred["class"]
                mapped_name = CLASS_MAPPING.get(raw_class, raw_class)
                conf = pred["confidence"]
                gear_confidences[mapped_name] = max(gear_confidences[mapped_name], conf)

        # Final checklist
        threshold = 0.1
        checklist = {
            item: {
                "detected": gear_confidences.get(item, 0.0) > threshold,
                "confidence": gear_confidences.get(item, 0.0)
            }
            for item in COMMON_GEAR_ITEMS
        }

        # Prepare result
        result = {
            "detected_items": [item for item, v in checklist.items() if v["detected"]],
            "detection_checklist": checklist,
        }
        return result

    finally:
        os.remove(temp_path)
