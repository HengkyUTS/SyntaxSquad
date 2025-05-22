import streamlit as st
import cv2
import numpy as np
import time
import mediapipe as mp
import requests
from translator import Gloss2TextTranslator
import os
os.environ['OPENAI_API_KEY'] = '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'

# 初始化 Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

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
    st.session_state.landmark_buffer = []  # 将收集的 landmark 每帧加入其中
if "gpt_translator" not in st.session_state:
    st.session_state.gpt_translator = Gloss2TextTranslator(model_name='gpt-4o-mini')

# 设置 Streamlit 页面配置
st.set_page_config(page_title="ASL Translator", layout="wide")
# 页面标题
st.title("🤟 Real-time ASL Translator")

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

# 控制是否运行摄像头
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
        """
        if st.session_state.gloss_history_plain:
            try:
                translations = st.session_state.gpt_translator.translate(st.session_state.gloss_history_plain)
                if translations:
                    st.session_state.current_translation = translations[0]
            except Exception as e:
                st.session_state.current_translation = f"[翻译失败] {e}"
        """
show_landmarks = st.checkbox("🔍 show Landmarks", value=True)

# UI区域布局
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📷 Webcam Feed")
    # FRAME_WINDOW = st.image([])
    camera_placeholder = col1.empty()

with col2:
    st.header("🔤 Prediction")

    # Gloss 卡片
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

    # Sentence 卡片
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

# 定义绘图函数
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

    # ---- 1. Pose (6个点)
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

# 运行主循环（必须在 checkbox 控制下）
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
            st.warning("无法读取摄像头。")
            break
        # 翻转 + 转换色彩
        frame = cv2.flip(frame, 1)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mediapipe 处理
        results = holistic.process(image)

        # 绘制关键点
        if show_landmarks:
            annotated = draw_landmarks_on_frame(frame, results)
        else:
            annotated = frame

        # 显示图像
        camera_placeholder.image(annotated, channels="BGR", use_container_width=True)

        # predict gloss
        landmarks = extract_180_landmarks(results)
        if landmarks.shape == (180, 3):
            st.session_state.landmark_buffer.append(landmarks)
        else:
            st.write("⚠️ 当前帧缺失关键点，已跳过")

        gloss = "Collecting..."  # 默认显示
        # 满 195 帧就预测
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
                st.write("✅ 翻译返回：", translations)
                if translations:
                    st.session_state.current_translation = translations
            except Exception as e:
                st.session_state.current_translation = f"[翻译失败] {e}"

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

        # 保持 UI 更新流畅（20fps 左右）
        time.sleep(0.05)
    cap.release()

