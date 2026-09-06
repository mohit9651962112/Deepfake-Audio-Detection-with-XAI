
import streamlit as st
import tensorflow as tf
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import io
import cv2

from lime import lime_image


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Deepfake Audio Detection using XAI",
    page_icon="🎧",
    layout="centered",
    initial_sidebar_state="expanded"
)


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = (
    "/content/drive/MyDrive/Deepfake-Audio/models/"
    "deepfake_mobilenetv2_balanced_v2_best.keras"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ---------- Main background ---------- */

    .stApp {
        background-color: #0e0f13;
    }

    /* ---------- Main content width ---------- */

    .block-container {
        max-width: 850px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
    }

    /* ---------- Sidebar ---------- */

    [data-testid="stSidebar"] {
        background-color: #24252e;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: white;
    }

    /* ---------- Main title ---------- */

    .main-title {
        font-size: 42px;
        font-weight: 800;
        line-height: 1.2;
        color: #f5f5f5;
        margin-bottom: 35px;
    }

    /* ---------- Section title ---------- */

    .section-title {
        font-size: 25px;
        font-weight: 700;
        color: #f1f1f1;
        margin-top: 30px;
        margin-bottom: 15px;
    }

    /* ---------- Result cards ---------- */

    .result-real {
        background: #123d2b;
        border: 1px solid #27ae60;
        border-radius: 8px;
        padding: 18px 22px;
        margin-top: 10px;
        margin-bottom: 20px;
        color: #55e68a;
        font-size: 17px;
        font-weight: 600;
    }

    .result-fake {
        background: #401d22;
        border: 1px solid #e74c3c;
        border-radius: 8px;
        padding: 18px 22px;
        margin-top: 10px;
        margin-bottom: 20px;
        color: #ff6b60;
        font-size: 17px;
        font-weight: 600;
    }

    /* ---------- Confidence ---------- */

    .confidence-label {
        color: #d0d0d0;
        font-size: 15px;
        margin-top: 15px;
    }

    .confidence-value {
        color: #f5f5f5;
        font-size: 34px;
        font-weight: 700;
        margin-bottom: 20px;
    }

    /* ---------- Probability ---------- */

    .prob-title {
        color: #dddddd;
        font-size: 15px;
        margin-bottom: 6px;
    }

    .prob-value {
        color: #ffffff;
        font-size: 20px;
        font-weight: 600;
    }

    /* ---------- Divider ---------- */

    .divider {
        height: 1px;
        background-color: #383a42;
        margin: 25px 0;
    }

    /* ---------- About ---------- */

    .about-box {
        background: #17191f;
        border: 1px solid #30323a;
        border-radius: 10px;
        padding: 25px;
        color: #dddddd;
        line-height: 1.7;
    }

    /* ---------- Footer ---------- */

    .footer {
        text-align: center;
        color: #777a83;
        font-size: 13px;
        margin-top: 45px;
    }

    /* ---------- Buttons ---------- */

    div.stButton > button {
        width: 100%;
        border-radius: 7px;
        min-height: 42px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    try:

        model = tf.keras.models.load_model(
            MODEL_PATH,
            compile=False
        )

        return model

    except Exception as e:

        st.error(
            f"Unable to load model.\n\n"
            f"Path: {MODEL_PATH}\n\n"
            f"Error: {e}"
        )

        return None


# ============================================================
# AUDIO -> SPECTROGRAM
# ============================================================

def create_spectrogram(audio_bytes):

    """
    Returns:

    model_image:
        224x224x3 tensor normalized to [-1, 1]
        This is used by the trained MobileNetV2 model.

    display_image:
        224x224x3 tensor normalized to [0, 1]
        This is only used for displaying the spectrogram.
    """

    try:

        # Load audio exactly like training preprocessing
        y, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=22050,
            mono=True
        )

        if y is None or len(y) == 0:
            raise ValueError("The uploaded audio file is empty.")

        # Mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr
        )

        # Convert to dB
        log_mel = librosa.power_to_db(
            mel,
            ref=np.max
        )

        # ====================================================
        # DISPLAY IMAGE
        # ====================================================

        display_image = log_mel.astype(np.float32)

        min_value = display_image.min()
        max_value = display_image.max()

        display_image = (
            display_image - min_value
        ) / (
            max_value - min_value + 1e-8
        )

        display_tensor = tf.convert_to_tensor(
            display_image,
            dtype=tf.float32
        )

        display_tensor = display_tensor[..., tf.newaxis]

        display_tensor = tf.image.resize(
            display_tensor,
            (224, 224)
        )

        display_tensor = tf.repeat(
            display_tensor,
            repeats=3,
            axis=-1
        )

        # Ensure [0,1]
        display_tensor = tf.clip_by_value(
            display_tensor,
            0.0,
            1.0
        )

        # ====================================================
        # MODEL IMAGE
        # ====================================================

        model_tensor = display_tensor * 2.0 - 1.0

        return model_tensor, display_tensor

    except Exception as e:

        raise RuntimeError(
            f"Could not process audio: {e}"
        )


# ============================================================
# CREATE BEAUTIFUL MEL SPECTROGRAM FOR UI
# ============================================================

def create_display_spectrogram(audio_bytes):

    try:

        y, sr = librosa.load(
            io.BytesIO(audio_bytes),
            sr=22050,
            mono=True
        )

        if y is None or len(y) == 0:
            raise ValueError("Empty audio.")

        mel = librosa.feature.melspectrogram(
            y=y,
            sr=sr
        )

        log_mel = librosa.power_to_db(
            mel,
            ref=np.max
        )

        fig, ax = plt.subplots(
            figsize=(10, 4)
        )

        librosa.display.specshow(
            log_mel,
            sr=sr,
            x_axis="time",
            y_axis="mel",
            ax=ax
        )

        ax.set_title(
            "Mel Spectrogram",
            fontsize=13
        )

        ax.set_xlabel("Time")
        ax.set_ylabel("Frequency")

        fig.tight_layout()

        buffer = io.BytesIO()

        fig.savefig(
            buffer,
            format="png",
            dpi=150,
            bbox_inches="tight"
        )

        plt.close(fig)

        buffer.seek(0)

        return buffer

    except Exception as e:

        raise RuntimeError(
            f"Could not create spectrogram visualization: {e}"
        )


# ============================================================
# PREDICTION
# ============================================================

def predict_audio(image, model):

    """

    Model output:

        sigmoid output = probability of REAL

    Therefore:

        real_probability = prediction
        fake_probability = 1 - prediction

    """

    image_batch = tf.expand_dims(
        image,
        axis=0
    )

    prediction = model.predict(
        image_batch,
        verbose=0
    )[0][0]

    real_probability = float(prediction)

    fake_probability = 1.0 - real_probability

    if real_probability >= 0.5:

        label = "REAL"

    else:

        label = "FAKE"

    confidence = max(
        real_probability,
        fake_probability
    )

    return (
        label,
        confidence,
        real_probability,
        fake_probability
    )


# ============================================================
# LIME
# ============================================================

def lime_predict(images, model):

    """
    LIME expects images in [0,1].

    Detector model expects [-1,1].

    Therefore convert:

        [0,1] -> [-1,1]

    """

    images = np.asarray(
        images,
        dtype=np.float32
    )

    images = np.clip(
        images,
        0.0,
        1.0
    )

    model_images = (
        images * 2.0 - 1.0
    )

    predictions = model.predict(
        model_images,
        verbose=0
    ).reshape(-1)

    fake_probability = 1.0 - predictions
    real_probability = predictions

    return np.column_stack(
        [
            fake_probability,
            real_probability
        ]
    )


def generate_lime(image, model):

    explainer = lime_image.LimeImageExplainer()

    image_np = image.numpy().astype(np.float32)

    image_np = np.clip(
        image_np,
        0.0,
        1.0
    )

    explanation = explainer.explain_instance(
        image_np,
        lambda imgs: lime_predict(
            imgs,
            model
        ),
        top_labels=2,
        hide_color=0,
        num_features=15,
        positive_only=False,
        num_samples=500
    )

    prediction = lime_predict(
        image_np[None, ...],
        model
    )[0]

    predicted_label = int(
        np.argmax(prediction)
    )

    temp, mask = explanation.get_image_and_mask(
        predicted_label,
        positive_only=False,
        num_features=15,
        hide_rest=False
    )

    temp = np.clip(
        temp,
        0.0,
        1.0
    )

    result = (
        temp * 255
    ).astype(np.uint8)

    # Draw explanation boundaries
    boundary_mask = (
        np.abs(mask) > 0
    ).astype(np.uint8) * 255

    contours, _ = cv2.findContours(
        boundary_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    result = cv2.cvtColor(
        result,
        cv2.COLOR_RGB2BGR
    )

    cv2.drawContours(
        result,
        contours,
        -1,
        (255, 255, 0),
        2
    )

    result = cv2.cvtColor(
        result,
        cv2.COLOR_BGR2RGB
    )

    return result

# ============================================================
# GRAD-CAM
# ============================================================

def grad_cam(image, model):

    """
    Grad-CAM for:

        Sequential
            MobileNetV2
            GlobalAveragePooling
            Dense
            Dropout
            Dense
            Dropout
            Dense

    The MobileNetV2 base model is the nested Keras model.
    """

    try:

        image_batch = tf.expand_dims(
            image,
            axis=0
        )

        # ----------------------------------------------------
        # Find nested MobileNetV2
        # ----------------------------------------------------

        base_model = None

        for layer in model.layers:

            if isinstance(
                layer,
                tf.keras.Model
            ):

                base_model = layer
                break

        if base_model is None:

            raise ValueError(
                "Could not find the MobileNetV2 base model."
            )

        # ----------------------------------------------------
        # Find last convolutional layer
        # ----------------------------------------------------

        try:

            last_conv_layer = base_model.get_layer(
                "out_relu"
            )

        except Exception:

            # Fallback: find last 4D layer
            last_conv_layer = None

            for layer in reversed(
                base_model.layers
            ):

                try:

                    output_shape = layer.output.shape

                    if len(output_shape) == 4:

                        last_conv_layer = layer
                        break

                except Exception:

                    continue

            if last_conv_layer is None:

                raise ValueError(
                    "Could not find convolutional layer."
                )

        # ----------------------------------------------------
        # Gradient model
        # ----------------------------------------------------

        grad_model = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[
                last_conv_layer.output,
                base_model.output
            ]
        )

        with tf.GradientTape() as tape:

            conv_outputs, base_output = grad_model(
                image_batch,
                training=False
            )

            x = base_output

            # Pass MobileNet output through classifier
            for layer in model.layers[1:]:

                x = layer(
                    x,
                    training=False
                )

            prediction = x

            real_probability = prediction[:, 0]

        # ----------------------------------------------------
        # Gradients
        # ----------------------------------------------------

        gradients = tape.gradient(
            real_probability,
            conv_outputs
        )

        if gradients is None:

            raise ValueError(
                "Gradients could not be calculated."
            )

        # Global average pooling
        pooled_gradients = tf.reduce_mean(
            gradients,
            axis=(0, 1, 2)
        )

        conv_outputs = conv_outputs[0]

        # Weighted activation map
        heatmap = tf.reduce_sum(
            conv_outputs *
            pooled_gradients,
            axis=-1
        )

        # ReLU
        heatmap = tf.maximum(
            heatmap,
            0
        )

        max_heatmap = tf.reduce_max(
            heatmap
        )

        heatmap = heatmap / (
            max_heatmap + tf.keras.backend.epsilon()
        )

        heatmap = heatmap.numpy()

        # ----------------------------------------------------
        # Convert model image back to display image
        # ----------------------------------------------------

        display_image = (
            image.numpy() + 1.0
        ) / 2.0

        display_image = np.clip(
            display_image,
            0.0,
            1.0
        )

        display_image = (
            display_image * 255
        ).astype(np.uint8)

        # Resize heatmap
        heatmap = cv2.resize(
            heatmap,
            (
                display_image.shape[1],
                display_image.shape[0]
            )
        )

        heatmap = (
            heatmap * 255
        ).astype(np.uint8)

        # Apply color map
        heatmap_color = cv2.applyColorMap(
            heatmap,
            cv2.COLORMAP_JET
        )

        heatmap_color = cv2.cvtColor(
            heatmap_color,
            cv2.COLOR_BGR2RGB
        )

        # Overlay
        superimposed = cv2.addWeighted(
            display_image,
            0.60,
            heatmap_color,
            0.40,
            0
        )

        return (
            display_image,
            heatmap_color,
            superimposed
        )

    except Exception as e:

        raise RuntimeError(
            f"Grad-CAM failed: {e}"
        )


# ============================================================
# HOMEPAGE
# ============================================================

def homepage():

    # ========================================================
    # TITLE
    # ========================================================

    st.markdown(
        """
        <div class="main-title">
            Deepfake Audio Detection<br>
            using XAI
        </div>
        """,
        unsafe_allow_html=True
    )

    # ========================================================
    # UPLOAD
    # ========================================================

    st.markdown(
        '<div class="section-title">Choose a WAV file</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload WAV audio",
        type=["wav"],
        label_visibility="visible"
    )

    if uploaded_file is None:
        return

    # ========================================================
    # FILE DATA
    # ========================================================

    audio_bytes = uploaded_file.getvalue()

    # ========================================================
    # AUDIO PLAYER
    # ========================================================

    st.markdown(
        '<div class="section-title">Play Audio</div>',
        unsafe_allow_html=True
    )

    st.audio(
        audio_bytes,
        format="audio/wav"
    )

    # ========================================================
    # AUDIO INFORMATION
    # ========================================================

    st.markdown(
        '<div class="section-title">📊 Audio Information</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.caption("File Name")
        st.write(uploaded_file.name)

    with col2:
        st.caption("File Type")
        st.write(uploaded_file.type)

    with col3:
        st.caption("File Size")
        st.write(
            f"{len(audio_bytes) / 1024:.2f} KB"
        )

    # ========================================================
    # SPECTROGRAM
    # ========================================================

    try:

        model_image, display_image = create_spectrogram(
            audio_bytes
        )

        spectrogram_image = create_display_spectrogram(
            audio_bytes
        )

    except Exception as e:

        st.error(
            f"Spectrogram generation failed: {e}"
        )

        return

    st.image(
        spectrogram_image,
        caption="Mel Spectrogram",
        use_container_width=True
    )

    # ========================================================
    # MODEL
    # ========================================================

    model = load_model()

    if model is None:
        return

    # ========================================================
    # ANALYZE BUTTON
    # ========================================================

    if "analysis_done" not in st.session_state:
        st.session_state.analysis_done = False

    if "prediction_result" not in st.session_state:
        st.session_state.prediction_result = None

    # Reset analysis when new file is uploaded
    current_file_id = (
        uploaded_file.name,
        len(audio_bytes)
    )

    if (
        "current_file_id"
        not in st.session_state
        or
        st.session_state.current_file_id
        != current_file_id
    ):

        st.session_state.current_file_id = current_file_id
        st.session_state.analysis_done = False
        st.session_state.prediction_result = None

    # ========================================================
    # ANALYZE
    # ========================================================

    st.markdown(
        '<div class="section-title">🔍 Analysis</div>',
        unsafe_allow_html=True
    )

    if st.button(
        "🚀 Analyze Audio",
        use_container_width=True
    ):

        with st.spinner(
            "Analyzing audio..."
        ):

            try:

                result = predict_audio(
                    model_image,
                    model
                )

                st.session_state.prediction_result = result
                st.session_state.analysis_done = True

            except Exception as e:

                st.error(
                    f"Prediction failed: {e}"
                )

                return

    # ========================================================
    # SHOW RESULTS
    # ========================================================

    if st.session_state.analysis_done:

        (
            label,
            confidence,
            real_probability,
            fake_probability
        ) = st.session_state.prediction_result

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">'
            'Classification Results'
            '</div>',
            unsafe_allow_html=True
        )

        if label == "REAL":

            st.markdown(
                """
                <div class="result-real">
                    🎧 &nbsp; The uploaded audio is <b>REAL</b>
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div class="result-fake">
                    ⚠️ &nbsp; The uploaded audio is <b>FAKE</b>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        st.markdown(
            '<div class="confidence-label">'
            'Confidence'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            f"""
            <div class="confidence-value">
                {confidence * 100:.2f}%
            </div>
            """,
            unsafe_allow_html=True
        )

        # ----------------------------------------------------
        # PROBABILITIES
        # ----------------------------------------------------

        col1, col2 = st.columns(2)

        with col1:

            st.markdown(
                '<div class="prob-title">'
                'Real Probability'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="prob-value">
                    {real_probability * 100:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

        with col2:

            st.markdown(
                '<div class="prob-title">'
                'Fake Probability'
                '</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                f"""
                <div class="prob-value">
                    {fake_probability * 100:.2f}%
                </div>
                """,
                unsafe_allow_html=True
            )

        # ----------------------------------------------------
        # PROGRESS
        # ----------------------------------------------------

        st.progress(
            real_probability,
            text=f"Real: {real_probability * 100:.2f}%"
        )

        st.progress(
            fake_probability,
            text=f"Fake: {fake_probability * 100:.2f}%"
        )

        # ====================================================
        # EXPLAINABLE AI
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            'Explainable AI'
            '</div>',
            unsafe_allow_html=True
        )

        tab1, tab2 = st.tabs(
            [
                "🔎 LIME",
                "🔥 Grad-CAM"
            ]
        )

        # ====================================================
        # LIME
        # ====================================================

        with tab1:

            st.write(
                "LIME highlights regions of the "
                "spectrogram that contributed to "
                "the model prediction."
            )

            if st.button(
                "Generate LIME Explanation",
                key="generate_lime"
            ):

                with st.spinner(
                    "Generating LIME explanation... "
                    "This may take a few seconds."
                ):

                    try:

                        lime_result = generate_lime(
                            model_image,
                            model
                        )

                        st.session_state.lime_result = (
                            lime_result
                        )

                    except Exception as e:

                        st.session_state.lime_result = None

                        st.error(
                            f"LIME failed: {e}"
                        )

            # ------------------------------------------------
            # Show saved LIME result
            # ------------------------------------------------

            if (
                "lime_result"
                in st.session_state
                and
                st.session_state.lime_result
                is not None
            ):

                st.image(
                    st.session_state.lime_result,
                    caption="LIME Explanation",
                    use_container_width=True
                )

        # ====================================================
        # GRAD-CAM
        # ====================================================

        with tab2:

            st.write(
                "Grad-CAM shows the regions of the "
                "spectrogram that influenced the "
                "neural network prediction."
            )

            if st.button(
                "Generate Grad-CAM",
                key="generate_gradcam"
            ):

                with st.spinner(
                    "Generating Grad-CAM..."
                ):

                    try:

                        (
                            original,
                            heatmap,
                            superimposed
                        ) = grad_cam(
                            model_image,
                            model
                        )

                        st.session_state.gradcam_result = (
                            original,
                            heatmap,
                            superimposed
                        )

                    except Exception as e:

                        st.session_state.gradcam_result = None

                        st.error(
                            f"Grad-CAM failed: {e}"
                        )

            # ------------------------------------------------
            # Show saved Grad-CAM result
            # ------------------------------------------------

            if (
                "gradcam_result"
                in st.session_state
                and
                st.session_state.gradcam_result
                is not None
            ):

                (
                    original,
                    heatmap,
                    superimposed
                ) = st.session_state.gradcam_result

                col1, col2 = st.columns(2)

                with col1:

                    st.image(
                        original,
                        caption="Original",
                        use_container_width=True
                    )

                with col2:

                    st.image(
                        superimposed,
                        caption="Grad-CAM",
                        use_container_width=True
                    )

        # ====================================================
        # DISCLAIMER
        # ====================================================

        st.warning(
            "Model confidence represents the model's "
            "predicted probability and should not be "
            "treated as absolute proof of authenticity."
        )

    # ========================================================
    # FOOTER
    # ========================================================

    st.markdown(
        """
        <div class="footer">
            Deepfake Audio Detection with Explainable AI
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# ABOUT
# ============================================================

def about():

    st.markdown(
        """
        <div class="main-title">
            About the Project
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="about-box">

        <h3>🎧 Deepfake Audio Detection using XAI</h3>

        <p>
        This application uses a deep learning model to classify
        audio as either <b>real</b> or <b>fake</b>.
        </p>

        <p>
        The uploaded WAV audio is converted into a
        <b>Mel Spectrogram</b>, which is then processed by a
        MobileNetV2-based neural network.
        </p>

        <h4>Model</h4>

        <p>
        MobileNetV2 with custom dense classification layers.
        </p>

        <h4>Explainable AI</h4>

        <p>
        The application provides two XAI techniques:
        </p>

        <ul>
            <li><b>LIME</b> – explains important regions of
            the spectrogram.</li>

            <li><b>Grad-CAM</b> – visualizes areas that strongly
            influenced the neural network prediction.</li>
        </ul>

        <h4>Pipeline</h4>

        <p>
        WAV Audio → Mel Spectrogram → MobileNetV2 →
        Real/Fake Classification → XAI Explanation
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="footer">
            Deepfake Audio Detection with Explainable AI
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MAIN
# ============================================================

def main():

    page = st.sidebar.selectbox(
        "App Selections",
        [
            "Homepage",
            "About"
        ]
    )

    if page == "Homepage":

        homepage()

    elif page == "About":

        about()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
