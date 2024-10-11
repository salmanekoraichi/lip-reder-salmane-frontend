# Import necessary libraries
import streamlit as st
import os
import random
import time
import tensorflow as tf
from utils import load_data, num_to_char
from modelutil import load_model

# Set the page configuration
st.set_page_config(layout='wide', page_title="Salmane's Lip Reader", page_icon="👄")

# Custom CSS for styling (you can customize this as desired)
st.markdown("""
    <style>
    /* General styles */
    body {
        background-image: url('ensa.png');
        background-size: cover;
        background-position: center;
        color: #F4F5F7;
    }
    /* Overlay to darken the background image for better readability */
    body:before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(15, 32, 39, 0.85); /* Adjust the opacity as needed */
        z-index: -1;
    }
    /* Main title */
    .main-title {
        font-family: 'Arial Black', sans-serif;
        color: #19A7CE;
        text-align: center;
        font-size: 3em;
        margin-top: 20px;
    }
    /* Sidebar */
    .sidebar .sidebar-content {
        background-color: #1c1e21;
        padding: 20px;
    }
    .sidebar h1 {
        color: #19A7CE;
        text-align: center;
    }
    /* Footer */
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #1c1e21;
        color: white;
        text-align: center;
        padding: 10px;
    }
    /* Logo in the header */
    .header-logo {
        text-align: center;
        margin-bottom: 20px;
    }
    /* Adjust the logo image */
    .header-logo img {
        width: 150px;
        height: auto;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar content
def sidebar_content():
    st.sidebar.image('cropped-ensa.png', use_column_width=True)
    st.sidebar.title("💡 Salmane's Lip Reader")
    st.sidebar.info(
        "🌐 Developed by **Salmane Koraichi** at ENSA Tangier, inspired by the groundbreaking LipNet model. "
        "Utilizes **3D CNNs**, **Bidirectional GRUs**, and **CTC Loss** for accurate lip reading."
    )
    st.sidebar.markdown("### Project Development:")
    st.sidebar.markdown("""
    - **Model Architecture**: Combines spatiotemporal convolutions and recurrent networks.
    - **Deep Learning Techniques**: Employs **3D-CNNs** and **Bidirectional GRUs**.
    - **Dataset**: Trained on the GRID dataset with fixed sentence structures.
    - **Real-time Predictions**: Processes video frames and outputs predictions in real-time.
    """)
    # Fun fact button
    fun_facts = [
        "🧠 The human brain processes visual information 60,000 times faster than text.",
        "🌌 There are more possible iterations of a game of chess than atoms in the known universe.",
        "🔬 Quantum computers can solve certain problems exponentially faster than classical computers.",
        "🤖 AI can now write music, paint pictures, and even write code!",
        "🚀 Deep Learning models can contain millions of parameters, mimicking the complexity of the human brain."
    ]
    if st.sidebar.button('🤖 Show me a Fun Science Fact!'):
        st.sidebar.write(random.choice(fun_facts))

# Main content
def main_content():
    # Header with logo
    st.markdown("""
        <div class='header-logo'>
            <img src='data:image/png;base64,{}' alt='Logo'>
        </div>
    """.format(get_base64_encoded_image('cropped-ensa.png')), unsafe_allow_html=True)
    
    st.markdown("<div class='main-title'>LipNet Full Stack AI App</div>", unsafe_allow_html=True)
    
    # Tabs for navigation
    tabs = st.tabs(["Home", "Analysis", "Insights", "Model Building Steps"])
    
    # Home tab
    with tabs[0]:
        st.header("Welcome to Salmane's Lip Reader!")
        st.write("Explore the power of AI in lip reading. Select a video to get started.")
        # Video selection
        options = os.listdir(os.path.join('..', 'data', 's1'))
        if options:
            selected_video = st.selectbox('📹 Choose a video for analysis', options)
            if st.button("Analyze Video"):
                analysis(selected_video)
        else:
            st.error("No videos found in the directory.")
    
    # Analysis tab
    with tabs[1]:
        st.header("Analysis")
        # Placeholder for analysis (if needed)
        pass
    
    # Insights tab
    with tabs[2]:
        st.header("🔬 Visual Explanations")
        st.subheader("Saliency Maps")
        st.image("7-Figure2-1.png", caption="Model attention visualization for words 'please' and 'lay'.")
        st.subheader("Model Architecture")
        st.image("sensors-22-00072-g006.png", caption="3D-CNN with dropout applied to the video frames.")
        st.subheader("Phoneme to Viseme Mapping")
        st.image("13-Figure4-1.png", caption="Phoneme-to-viseme mapping highlighting visually similar phonemes.")
        # Additional content
        st.header("🤖 Explore More")
        st.markdown("""
        - **Lip Reading Applications**: Assisting the hearing impaired, silent communication, security.
        - **Tech Innovations**: Integration with AR/VR technologies for immersive experiences.
        - **Future of AI**: Advancements in unsupervised learning and multimodal AI.
        """)
    
    # Model Building Steps tab
    with tabs[3]:
        st.header("🛠️ Model Building Steps")
        st.write("In this section, we provide a step-by-step explanation of how the LipNet model was built, based on your `lipnet.ipynb` notebook.")
    
        # Step 1
        st.markdown("### Step 1: Install and Import Dependencies")
        st.write("We start by installing and importing all necessary dependencies, such as TensorFlow, OpenCV, and other libraries required for data processing and model building.")
        st.code("""
!pip install opencv-python matplotlib imageio gdown tensorflow

import os
import cv2
import tensorflow as tf
import numpy as np
from typing import List
from matplotlib import pyplot as plt
import imageio
""", language='python')
    
        # Step 2
        st.markdown("### Step 2: Build Data Loading Functions")
        st.write("We build functions to load and preprocess the video data and annotations.")
        st.code("""
def load_video(path: str) -> List[float]:
    cap = cv2.VideoCapture(path)
    frames = []
    for _ in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
        ret, frame = cap.read()
        frame = tf.image.rgb_to_grayscale(frame)
        frames.append(frame[190:236, 80:220, :])
    cap.release()
    mean = tf.math.reduce_mean(frames)
    std = tf.math.reduce_std(tf.cast(frames, tf.float32))
    return tf.cast((frames - mean), tf.float32) / std

def load_alignments(path: str) -> List[str]:
    with open(path, 'r') as f:
        lines = f.readlines()
    tokens = []
    for line in lines:
        line = line.split()
        if line[2] != 'sil':
            tokens.extend([' ', line[2]])
    return char_to_num(tf.reshape(tf.strings.unicode_split(tokens, input_encoding='UTF-8'), (-1)))[1:]
""", language='python')
    
        # Step 3
        st.markdown("### Step 3: Create Data Pipeline")
        st.write("We create a data pipeline using TensorFlow's Dataset API to efficiently load and preprocess the data.")
        st.code("""
data = tf.data.Dataset.list_files('./data/s1/*.mpg')
data = data.shuffle(500, reshuffle_each_iteration=False)
data = data.map(mappable_function)
data = data.padded_batch(2, padded_shapes=([75, None, None, None], [40]))
data = data.prefetch(tf.data.AUTOTUNE)

# Split into training and testing datasets
train = data.take(450)
test = data.skip(450)
""", language='python')
    
        # Step 4
        st.markdown("### Step 4: Design the Deep Neural Network")
        st.write("We design the LipNet model architecture using a combination of 3D Convolutional Neural Networks and Bidirectional LSTMs.")
        st.code("""
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPool3D, Activation, TimeDistributed, Flatten

model = Sequential()
model.add(Conv3D(128, 3, input_shape=(75, 46, 140, 1), padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1, 2, 2)))

model.add(Conv3D(256, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1, 2, 2)))

model.add(Conv3D(75, 3, padding='same'))
model.add(Activation('relu'))
model.add(MaxPool3D((1, 2, 2)))

model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Bidirectional(LSTM(128, return_sequences=True)))
model.add(Dropout(0.5))
model.add(Dense(char_to_num.vocabulary_size() + 1, activation='softmax'))
""", language='python')
    
        # Step 5
        st.markdown("### Step 5: Compile and Train the Model")
        st.write("We compile the model with the CTC (Connectionist Temporal Classification) Loss function and start the training process.")
        st.code("""
def CTCLoss(y_true, y_pred):
    batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
    input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
    label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")
    input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
    loss = tf.keras.backend.ctc_batch_cost(y_true, y_pred, input_length, label_length)
    return loss

model.compile(optimizer=Adam(learning_rate=0.0001), loss=CTCLoss)
model.fit(train, validation_data=test, epochs=100, callbacks=[...])
""", language='python')
    
        # Step 6
        st.markdown("### Step 6: Make Predictions")
        st.write("Finally, we load the trained model weights and use it to make predictions on new video data.")
        st.code("""
# Load model weights
model.load_weights('models/checkpoint')

# Make predictions
yhat = model.predict(sample[0])
decoded = tf.keras.backend.ctc_decode(yhat, input_length=[75], greedy=True)[0][0].numpy()
""", language='python')
        st.write("We then decode the predictions to get the final transcribed text.")
    
    # Add an animation or cool feature (e.g., animated header)
    st.markdown("""
        <style>
        @keyframes gradientBackground {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        .animated-header {
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradientBackground 15s ease infinite;
            padding: 20px;
            color: white;
            text-align: center;
            font-size: 2em;
            margin-bottom: 20px;
        }
        </style>
        <div class='animated-header'>Deep Learning in Action 🚀</div>
    """, unsafe_allow_html=True)

# Helper function to encode images
def get_base64_encoded_image(image_path):
    import base64
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')

# Analysis function
def analysis(selected_video):
    with st.spinner('🤖 AI is analyzing the video...'):
        time.sleep(2)
    
    # Split the layout into two columns
    col1, col2 = st.columns(2)
    
    # Video rendering
    with col1:
        st.subheader("🎬 Original Video")
        file_path = os.path.join('..', 'data', 's1', selected_video)
        os.system(f'ffmpeg -i {file_path} -vcodec libx264 test_video.mp4 -y')
        video = open('test_video.mp4', 'rb')
        video_bytes = video.read()
        st.video(video_bytes, format="video/mp4")
    
    # AI Model Processing
    with col2:
        st.subheader("🤖 AI Model Prediction")
        video_data, annotations = load_data(tf.convert_to_tensor(file_path))
        st.image('animation.gif', width=400)
        model = load_model()
        yhat = model.predict(tf.expand_dims(video_data, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.code(decoder, language='python')
        decoded_text = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.success(f"**Decoded Text:** {decoded_text}")
        st.balloons()

# Footer
def footer():
    st.markdown("""
        <div class="footer">
            🤖 Built with ❤️ by Salmane Koraichi 
        </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    sidebar_content()
    main_content()
    footer()
