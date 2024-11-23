import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2

# Load your trained YOLO model
model = YOLO('best.pt')

# Class names
class_names = [
    'crazing',
    'inclusion',
    'patches',
    'pitted_surface',
    'rolled-in_scale',
    'scratches'
]

# Streamlit App
st.sidebar.image('images.jpeg')
st.title("Surface Defect Detection in Manufacturing ")
st.write("Upload an image, and the app will detect defects with bounding boxes.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Detecting defects...")

    # Convert the image to a NumPy array and ensure it's RGB
    image_np = np.array(image.convert("RGB"))

    # Run inference with a confidence threshold
    results = model.predict(image_np, conf=0.1)

    # Annotate the image with bounding boxes
    annotated_image = image_np.copy()
    for box in results[0].boxes:
        # Extract bounding box coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
        class_id = int(box.cls[0])  # Class index
        confidence = box.conf[0].item()  # Confidence score

        # Draw the bounding box
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Add class label and confidence
        label = f"{class_names[class_id]} {confidence:.2f}"
        cv2.putText(
            annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2
        )

    # Convert BGR to RGB for Streamlit display
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Display the annotated image
    st.image(annotated_image, caption="Detected Defects", use_column_width=True)
