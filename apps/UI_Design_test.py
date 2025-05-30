import streamlit as st
import cv2
import numpy as np
import time
import mediapipe as mp
import requests
from translator import Gloss2TextTranslator
import os
import tempfile
import video2landmarks
os.environ['OPENAI_API_KEY'] = 'Your API KEY'

if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False
if "cap" not in st.session_state:
    st.session_state.cap = cv2.VideoCapture(0)
for var in ["gloss_history_display", "gloss_history_plain"]:
    if var not in st.session_state:
        st.session_state[var] = []
if "current_translation" not in st.session_state:
    st.session_state.current_translation = ""
if "landmark_buffer" not in st.session_state:
    st.session_state.landmark_buffer = []  # Add the collected each frame landmarks.
if "gpt_translator" not in st.session_state:
    st.session_state.gpt_translator = Gloss2TextTranslator(model_name='gpt-4o-mini')

# Set Streamlit page configuration
st.set_page_config(page_title="ASL Translator", layout="wide")
# Page title
st.title("🤟 Real-time ASL Translator")

mode = st.sidebar.selectbox(
    "🎛️ Select mode",
    ["Real-time Translation", "Upload Video"]
)

def draw_landmarks_on_frame(frame, results):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    return frame


filtered_hand = list(range(21))  # 21 for one hand
filtered_pose = [11, 12, 13, 14, 15, 16]  # 6 pose points
filtered_face = [0, 4, 7, 8, 10, 13, 14, 17, 21, 33, 37, 39, 40, 46, 52, 53, 54, 55, 58,
                 61, 63, 65, 66, 67, 70, 78, 80, 81, 82, 84, 87, 88, 91, 93, 95, 103, 105,
                 107, 109, 127, 132, 133, 136, 144, 145, 146, 148, 149, 150, 152, 153, 154,
                 155, 157, 158, 159, 160, 161, 162, 163, 172, 173, 176, 178, 181, 185, 191,
                 234, 246, 249, 251, 263, 267, 269, 270, 276, 282, 283, 284, 285, 288, 291,
                 293, 295, 296, 297, 300, 308, 310, 311, 312, 314, 317, 318, 321, 323, 324,
                 332, 334, 336, 338, 356, 361, 362, 365, 373, 374, 375, 377, 378, 379, 380,
                 381, 382, 384, 385, 386, 387, 388, 389, 390, 397, 398, 400, 402, 405, 409,
                 415, 454, 466, 468, 473]  # 153 face points


# extract_180_landmarks
def extract_180_landmarks(results):
    all_landmarks = []

    # ---- 1. Pose
    if results.pose_landmarks:
        pose = results.pose_landmarks.landmark
        for idx in filtered_pose:
            if idx < len(pose):
                lm = pose[idx]
                all_landmarks.append([lm.x, lm.y, lm.z])
            else:
                all_landmarks.append([0.0, 0.0, 0.0])
    else:
        all_landmarks.extend([[0.0, 0.0, 0.0]] * len(filtered_pose))

    # ---- 2. Left hand
    if results.left_hand_landmarks:
        hand = results.left_hand_landmarks.landmark
        for idx in filtered_hand:
            if idx < len(hand):
                lm = hand[idx]
                all_landmarks.append([lm.x, lm.y, lm.z])
            else:
                all_landmarks.append([0.0, 0.0, 0.0])
    else:
        all_landmarks.extend([[0.0, 0.0, 0.0]] * len(filtered_hand))

    # ---- 3. Right hand
    if results.right_hand_landmarks:
        hand = results.right_hand_landmarks.landmark
        for idx in filtered_hand:
            if idx < len(hand):
                lm = hand[idx]
                all_landmarks.append([lm.x, lm.y, lm.z])
            else:
                all_landmarks.append([0.0, 0.0, 0.0])
    else:
        all_landmarks.extend([[0.0, 0.0, 0.0]] * len(filtered_hand))

    # ---- 4. Face
    if results.face_landmarks:
        face = results.face_landmarks.landmark
        for idx in filtered_face:
            if idx < len(face):
                lm = face[idx]
                all_landmarks.append([lm.x, lm.y, lm.z])
            else:
                all_landmarks.append([0.0, 0.0, 0.0])
    else:
        all_landmarks.extend([[0.0, 0.0, 0.0]] * len(filtered_face))

    return np.array(all_landmarks)

extractor = video2landmarks.VideoLandmarksExtractor()

if mode == "Upload Video":
    uploaded_file = st.file_uploader("upload your video( .mp4 / .avi)", type=["mp4", "avi"])
    if uploaded_file is not None:
        filename = uploaded_file.name
        ext = os.path.splitext(filename)[-1].lower()

        # ✅ List of supported extensions
        allowed_ext = [".mp4", ".avi"]

        if ext not in allowed_ext:
            st.error(f"❌ Invalid format：{ext}. Only allow .mp4 or .avi")
        else:
            st.success(f"✅ successfully uploaded：{filename}")
        # Save to temporary file
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video_path = tfile.name

        gpt_translator = Gloss2TextTranslator(model_name='gpt-4o-mini')
        # layout
        col1, col2 = st.columns([2, 3])

        with col1:
            st.video(uploaded_file)

        with col2:
            st.markdown("### 🧾 results")

            # Initialise Mediapipe
            mp_holistic = mp.solutions.holistic
            holistic = mp_holistic.Holistic(static_image_mode=False)

            # Read all frames
            cap = cv2.VideoCapture(video_path)
            frame_buffer = []
            landmark_sequences = []
            frame_count = 0

            st.info("⏳ Frames are being extracted and analysed for sign language...")

            # Read frame by frame and extract landmarks
            with st.spinner("extracting landmarks..."):
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(frame_rgb)
                    # landmarks = extractor.extract_frame_landmarks(results)
                    landmarks = extract_180_landmarks(results)
                    frame_buffer.append(landmarks)
                    frame_count += 1

                    # Every 195 frames as a group
                    if len(frame_buffer) == 195:
                        landmark_sequences.append(np.array(frame_buffer))
                        frame_buffer = []
                if 0 < len(frame_buffer) < 195:
                    needed = 195 - len(frame_buffer)
                    zero_frame = np.zeros((180, 3))
                    frame_buffer.extend([zero_frame] * needed)
                    landmark_sequences.append(np.array(frame_buffer))

            cap.release()
            holistic.close()

            st.success(f"✅ successfully extracted {len(landmark_sequences)} sequences")

            # Model prediction gloss
            predicted_glosses = []
            predicted_display = []

            st.info("🧠 predicting...")
            for seq in landmark_sequences:
                landmark = np.array(seq).tolist()
                url = 'http://127.0.0.1:8000/predict'
                payload = {
                    'landmarks': landmark,
                    'top_n': 1,
                }
                response = requests.post(url, json=payload)
                prediction = response.json()
                if (
                        "predictions" in prediction and
                        isinstance(prediction["predictions"], list) and
                        len(prediction["predictions"]) > 0
                ):
                    top_pred = prediction["predictions"][0]
                    gloss = top_pred.get("gloss", "UNKNOWN")
                    score = top_pred.get("score", 0.0)
                    formatted = f"{gloss} ({score:.2f})"
                predicted_display.append(formatted)
                predicted_glosses.append(gloss)

            # display gloss
            gloss_display = " → ".join(predicted_display)
            st.markdown(
                f"""<div style='background:#f0fff0;padding:10px;border-radius:10px;border:2px solid #4CAF50;'>
                <strong>🧾 Gloss:</strong><br><span style='font-size:18px;'>{gloss_display}</span></div>""",
                unsafe_allow_html=True
            )

            # GPT translation
            try:
                translation = gpt_translator.translate(predicted_glosses)
                if translation:
                    current_translation = translation
            except Exception as e:
                current_translation = f"[translate failed!] {e}"

            st.markdown(
                f"""<div style='background:#e3f2fd;padding:10px;border-radius:10px;border:2px solid #2196F3;'>
                <strong>📝 Sentence:</strong><br><span style='font-size:18px;color:#0d47a1;'>{current_translation}</span></div>""",
                unsafe_allow_html=True
            )
    # process
elif mode == "Real-time Translation":
    # initialise Mediapipe
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic
    progress_container = st.empty()
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f4f6f9;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Control whether to run the camera
    col_run1, col_run2 = st.columns([1, 1])
    with col_run1:
        if st.button("▶️ start WebCam"):
            st.session_state.camera_running = True
    with col_run2:
        if st.button("⏹️ stop WebCam"):
            st.session_state.camera_running = False
            if "holistic" in st.session_state:
                st.session_state.holistic.close()
                del st.session_state.holistic
            if "cap" in st.session_state:
                st.session_state.cap.release()
                del st.session_state.cap
    show_landmarks = st.checkbox("🔍 show Landmarks", value=True)

    # UI layout
    col1, col2 = st.columns([1, 1])

    with col1:
        st.header("📷 Webcam Feed")
        # FRAME_WINDOW = st.image([])
        camera_placeholder = col1.empty()

    with col2:
        st.header("🔤 Prediction")

        # Gloss container
        with st.container():
            gloss_placeholder = st.markdown(
                """
                <div style="
                    border: 2px solid #4CAF50;
                    border-radius: 10px;
                    padding: 15px;
                    background-color: #e8f5e9;
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                ">
                    <h5 style="margin-bottom: 10px;">🧾 <span style='font-size: 20px;'>Gloss</span></h5>
                    <div id="gloss-output" style="font-size:18px; font-family: monospace; color: #2e7d32;">Waiting...</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # Sentence container
        with st.container():
            translation_placeholder = st.markdown(
                """
                <div style="
                    border: 2px solid #2196F3;
                    border-radius: 10px;
                    padding: 15px;
                    background-color: #e3f2fd;
                    box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                    margin-top: 20px;
                ">
                    <h5 style="margin-bottom: 10px;">💬 <span style='font-size: 20px;'>Sentence</span></h5>
                    <div id="sentence-output" style="font-size:18px; font-family: sans-serif; color: #0d47a1;">Waiting...</div>
                </div>
                """,
                unsafe_allow_html=True
            )
    # Run main loop (must be under checkbox control)
    if st.session_state.camera_running:
        cap = st.session_state.cap
        cap.set(3, 640)
        cap.set(4, 480)
        if "holistic" not in st.session_state:
            st.session_state.holistic = mp_holistic.Holistic(
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        holistic = st.session_state.holistic
        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                st.warning("can not read cam.")
                break
            # Flip + Colour conversion
            frame = cv2.flip(frame, 1)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Mediapipe process
            results = holistic.process(image)

            # Plot key points
            if show_landmarks:
                annotated = draw_landmarks_on_frame(frame, results)
            else:
                annotated = frame

            # predict gloss
            landmarks = extract_180_landmarks(results)
            if landmarks.shape == (180, 3):
                st.session_state.landmark_buffer.append(landmarks)
                # extract landmarks

                progress_value = min(len(st.session_state.landmark_buffer) / 195, 1.0)
                progress_percent = int(progress_value * 100)

                progress_container.markdown(f"""
                <div style="width: 300px; height: 18px; background-color: #ddd; border-radius: 9px; overflow: hidden; box-shadow: inset 1px 1px 3px rgba(0,0,0,0.2); margin-bottom: 8px;">
                  <div style="width: {progress_percent}%; height: 100%; background-color: #4CAF50;"></div>
                </div>
                <p style="margin:0; font-size: 13px; color: #555;">📸 collected frames：{len(st.session_state.landmark_buffer)} / 195</p>
                """, unsafe_allow_html=True)

            else:
                pass
                # st.write("⚠️ The current frame is missing key points and has been skipped.")

            gloss = "Collecting..."  # 默认显示
            # Predict when it reaches 195 frames.
            if len(st.session_state.landmark_buffer) == 195:
                sequence = np.array(st.session_state.landmark_buffer).tolist()
                url = 'http://127.0.0.1:8000/predict'
                payload = {
                    'landmarks': sequence,
                    'top_n': 1,
                }
                response = requests.post(url, json=payload)
                prediction = response.json()
                if (
                        "predictions" in prediction and
                        isinstance(prediction["predictions"], list) and
                        len(prediction["predictions"]) > 0
                ):
                    top_pred = prediction["predictions"][0]
                    gloss = top_pred.get("gloss", "UNKNOWN")
                    score = top_pred.get("score", 0.0)
                    formatted = f"{gloss} ({score:.2f})"
                    st.session_state.gloss_history_display.append(formatted)
                    st.session_state.gloss_history_plain.append(gloss)
                else:
                    st.session_state.gloss_history_display.append("UNKNOWN")
                    st.session_state.gloss_history_plain.append("UNKNOWN")
                st.session_state.landmark_buffer = []
                try:
                    translations = st.session_state.gpt_translator.translate(st.session_state.gloss_history_plain)
                    # st.write("✅ Translation returned：", translations)
                    if translations:
                        st.session_state.current_translation = translations
                except Exception as e:
                    st.session_state.current_translation = f"[translate failed!] {e}"

            gloss_sequence = " → ".join(st.session_state.gloss_history_display)
            gloss_placeholder.markdown(
                f"""
                        <div style='
                            border: 2px solid #4CAF50;
                            border-radius: 10px;
                            padding: 15px;
                            background-color: #f0fff0;
                            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                            margin-bottom: 10px;
                        '>
                            <h4 style='margin: 0; color: #2e7d32;'>🧾 Predicted Glosses</h4>
                            <p style='font-size: 20px; font-weight: bold; margin: 5px 0;'>{gloss_sequence}</p>
                        </div>
                        """,
                unsafe_allow_html=True
            )
            sentence = st.session_state.get("current_translation", "")
            # Update translation text
            translation_placeholder.markdown(
                f"""
                        <div style="
                            border: 2px solid #2196F3;
                            border-radius: 10px;
                            padding: 15px;
                            background-color: #e3f2fd;
                            box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
                            margin-top: 20px;
                        ">
                            <h5 style="margin-bottom: 10px;">💬 <span style='font-size: 20px;'>Sentence</span></h5>
                            <div style="font-size:18px; font-family: sans-serif; color: #0d47a1;">{sentence}</div>
                        </div>
                        """,
                unsafe_allow_html=True
            )
            if sentence:
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.8
                font_thickness = 2
                text_color = (255, 255, 255)
                bg_color = (0, 0, 0)
                padding = 10

                # Translation Return Calculate text position (bottom centre)
                (text_width, text_height), _ = cv2.getTextSize(sentence, font, font_scale, font_thickness)
                text_x = int((annotated.shape[1] - text_width) / 2)
                text_y = annotated.shape[0] - 30

                # Draw a black background box
                cv2.rectangle(
                    annotated,
                    (text_x - padding, text_y - text_height - padding),
                    (text_x + text_width + padding, text_y + padding),
                    bg_color,
                    thickness=-1
                )

                # drawing text
                cv2.putText(
                    annotated,
                    sentence,
                    (text_x, text_y),
                    font,
                    font_scale,
                    text_color,
                    font_thickness,
                    cv2.LINE_AA
                )
            # display image
            camera_placeholder.image(annotated, channels="BGR", use_container_width=True)
            # Keep the UI updating smoothly (around 20 fps)
            time.sleep(0.05)
        cap.release()

