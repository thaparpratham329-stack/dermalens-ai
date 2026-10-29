import streamlit as st
import tensorflow as tf
import numpy as np
import json
from PIL import Image
from pathlib import Path

st.set_page_config(
    page_title="DermaLens AI",
    page_icon="🔬",
    layout="centered"
)

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        BASE_DIR / "dermalens_ai_model.keras"
    )

with open(BASE_DIR / "class_mapping.json", "r") as f:
    CLASS_MAPPING = json.load(f)

st.title("🔬 DermaLens AI")
st.subheader("A clearer view through artificial intelligence.")

st.write(
    "Upload an image to explore the model's acne severity "
    "classification."
)

st.info(
    "Educational research prototype only. "
    "Not a medical diagnosis or treatment recommendation."
)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded image", width=300)

    if st.button("Analyze Image"):
        with st.spinner("Analyzing image..."):
            image_resized = image.resize((224, 224))

            img_array = np.array(
                image_resized,
                dtype=np.float32
            )

            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
                img_array
            )

            img_array = np.expand_dims(img_array, axis=0)

            model = load_model()
            predictions = model.predict(
                img_array,
                verbose=0
            )[0]

            predicted_index = int(np.argmax(predictions))
            predicted_class = CLASS_MAPPING[str(predicted_index)]

            st.success("Analysis completed!")

            st.subheader("Predicted category")
            st.write(f"### {predicted_class}")

            st.subheader("Model output scores")

            for i, score in enumerate(predictions):
                label = CLASS_MAPPING[str(i)]
                st.write(f"{label}: {score * 100:.2f}%")

            st.caption(
                "These percentages are raw model output scores, "
                "not clinically validated confidence estimates. "
                "Results may be incorrect."
            )

st.divider()

st.caption(
    "DermaLens AI | Educational ECE/AI project | "
    "Not intended for clinical use."
)
