import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import efficientnet_v2
from huggingface_hub import hf_hub_download
from pathlib import Path

# Set Streamlit page configuration (must be first Streamlit command)
st.set_page_config(
    page_title="DR Classification - EfficientNetV2B3",
    page_icon="🔬",
    layout="centered"
)

REPO_ID = "LilyThao/thao_efficientnetv2b3-dr"
FILENAME_IN_REPO = "best_efficientnetv2b3.keras"
MODEL_PATH = "best_efficientnetv2b3.keras"
IMG_SIZE = (224, 224)

# Define a mapping from numerical labels to human-readable class names
ID2LABEL = {
    0: "0 - No DR",
    1: "1 - Mild",
    2: "2 - Moderate",
    3: "3 - Severe",
    4: "4 - Proliferative DR"
}

# Use Streamlit's cache_resource decorator to load the model only once
@st.cache_resource
def load_model() -> tf.keras.Model:
    """Load the model from local path or download from Hugging Face if not available."""
    try:
        # Check if model exists locally, if not download it
        if not Path(MODEL_PATH).exists():
            st.info(f"Downloading model from Hugging Face repository: {REPO_ID}...")
            hf_hub_download(
                repo_id=REPO_ID,
                filename=FILENAME_IN_REPO,
                local_dir=".",
                local_dir_use_symlinks=False
            )
            st.success("Model downloaded successfully!")
        
        # Load the pre-trained Keras model
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.stop()

def preprocess_image(img: Image.Image) -> np.ndarray:
    """Preprocess the uploaded image for model prediction using EfficientNetV2 preprocessing.
    
    Args:
        img: PIL Image object
        
    Returns:
        Preprocessed image array with shape (1, 224, 224, 3) using EfficientNetV2 preprocessing
    """
    img = img.convert("RGB")  # Convert image to RGB format
    img = img.resize(IMG_SIZE)  # Resize image to the target dimensions
    img_array = np.array(img).astype("float32")  # Convert PIL Image to NumPy array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, 224, 224, 3)
    # Apply EfficientNetV2-specific preprocessing (same as during training)
    img_array = efficientnet_v2.preprocess_input(img_array)
    return img_array

# Load the model
model = load_model()

# Display the main title of the application
st.title("🔬 Diabetic Retinopathy Severity Classification")

# Display the model name being used
st.write("**Model:** EfficientNetV2B3 (`best_efficientnetv2b3`)")

# Provide a brief description of the application
st.markdown(
    "This application classifies the severity of diabetic retinopathy from retinal images using a fine-tuned EfficientNetV2B3 model."
)
st.divider()

# Create a file uploader widget for image upload
uploaded_file = st.file_uploader(
    "Upload a retinal image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG"
)

# Check if a file has been uploaded
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Create a button for prediction
        if st.button("🔍 Run Prediction", type="primary"):
            with st.spinner("Analyzing image..."):
                try:
                    # Preprocess the uploaded image
                    x = preprocess_image(image)

                    # Get predictions from the model
                    preds = model.predict(x, verbose=0)
                    pred_probs = preds[0]

                    # Determine the predicted class and its confidence score
                    pred_class = int(np.argmax(pred_probs))
                    pred_conf = float(np.max(pred_probs))

                    # Display the predicted result
                    st.success("Prediction completed!")
                    st.subheader("📊 Predicted Result")
                    
                    # Highlight prediction with metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Predicted Class", ID2LABEL.get(pred_class, str(pred_class)))
                    with col2:
                        st.metric("Confidence", f"{pred_conf:.2%}")

                    # Display probabilities for all classes in a table
                    st.subheader("📈 Probability for Each Class")
                    
                    # Create DataFrame for better table formatting
                    prob_df = pd.DataFrame([
                        {
                            "Class": i,
                            "Class Name": ID2LABEL.get(i, f"Class {i}"),
                            "Probability": f"{float(pred_probs[i]):.2%}"
                        }
                        for i in range(len(pred_probs))
                    ])
                    
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
                    st.exception(e)
    
    except Exception as e:
        st.error(f"Error loading image: {str(e)}")
        st.exception(e)
