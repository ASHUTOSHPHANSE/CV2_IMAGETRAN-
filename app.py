import streamlit as st
import cv2
import numpy as np

# Function to apply image transformations
def apply_transformation(image, transformation_type, parameters):
    rows, cols, _ = image.shape
    if transformation_type == 'Scaling':
        fx, fy = parameters
        scaled_image = cv2.resize(image, None, fx=fx, fy=fy)
        return scaled_image
    elif transformation_type == 'Rotation':
        angle = parameters
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        rotated_image = cv2.warpAffine(image, M, (cols, rows))
        return rotated_image
    elif transformation_type == 'Translation':
        tx, ty = parameters
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        translated_image = cv2.warpAffine(image, M, (cols, rows))
        return translated_image
    elif transformation_type == 'Shearing':
        shear = parameters
        shear_matrix = np.float32([[1, shear, 0], [shear, 1, 0]])
        sheared_image = cv2.warpAffine(image, shear_matrix, (cols, rows))
        return sheared_image

# Streamlit app
st.title("Image Transformation App")

uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image_bytes = uploaded_image.read()
    
    # Check if the uploaded image is in PNG format (which may have transparency)
    if uploaded_image.type == "image/png":
        # If it's a PNG, convert it to JPEG to remove transparency
        image_np = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
        _, image_bytes = cv2.imencode('.jpg', image_np)
    
    # Read the image using OpenCV
    image = cv2.imdecode(np.asarray(bytearray(image_bytes), dtype=np.uint8), cv2.IMREAD_COLOR)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Choose transformation type    
    transformation_type = st.selectbox("Select Transformation Type", ['Scaling', 'Rotation', 'Translation', 'Shearing'])

    # Set transformation parameters based on the selected transformation
    if transformation_type == 'Scaling':
        parameters = st.slider("Set Scaling Factor (fx, fy)", 0.1, 2.0, (1.0, 1.0))
    elif transformation_type == 'Rotation':
        parameters = st.slider("Set Rotation Angle (degrees)", -180.0, 180.0, 0.0)
    elif transformation_type == 'Translation':
        parameters = st.slider("Set Translation (tx, ty)", -100, 100, (0, 0))
    elif transformation_type == 'Shearing':
        parameters = st.slider("Set Shearing Parameter", -1.0, 1.0, 0.0)

    if st.button("Apply Transformation"):
        transformed_image = apply_transformation(image.copy(), transformation_type, parameters)
        st.image(transformed_image, caption=f"{transformation_type} Transformed Image", use_column_width=True)
