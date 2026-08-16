import streamlit as st
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

from src.components.face_detector import FaceDetector
from src.components.preprocessor import FacePreprocessor
from src.components.face_recognizer import FaceRecognizer
from src.components.db_store import DatabaseHandler
from src.components.evaluator import Evaluator

# ---------- PAGE ----------
st.set_page_config(
    page_title = "GreetSense - Smart Face Recogniser",
    page_icon = "📷",
    layout = "wide"
)

st.markdown("# 🏢 GreetSense - Smart Face Recogniser")
st.caption("Walk in. Get recognized. That's it.")
st.divider()

# ---------- INIT ----------
detector = FaceDetector()
preprocessor = FacePreprocessor()
recognizer = FaceRecognizer(models=["Facenet", "ArcFace"])
db = DatabaseHandler()

# ---------- SESSION ----------
if "last_embeddings" not in st.session_state:
    st.session_state.last_embeddings = None

if "name_history" not in st.session_state:
    st.session_state.name_history = []

if "evaluator" not in st.session_state:
    st.session_state.evaluator = Evaluator()

if "last_true_label" not in st.session_state:
    st.session_state.last_true_label = ""

if "show_results" not in st.session_state:
    st.session_state.show_results = False

if "cap" not in st.session_state:
    st.session_state.cap = cv2.VideoCapture(0)

# ---------- LAYOUT ----------
left_col, right_col = st.columns([1, 1])

# ================= LEFT PANEL =================
with left_col:

    with st.container(border=True):
        st.markdown("#### 👤 Register")
        name_input = st.text_input(
            "Name", label_visibility="collapsed", placeholder="Enter your name"
        )

        if st.button("Register Face", use_container_width=True):
            if name_input.strip() == "":
                st.warning("Enter a name first")
            elif st.session_state.last_embeddings is None:
                st.warning("No face detected")
            else:
                for model_name, embedding in st.session_state.last_embeddings.items():
                    if embedding is not None:
                        db.save_user(name_input.strip(), embedding, model_name)
                st.success(f"{name_input} registered successfully!")

    with st.container(border=True):
        st.markdown("#### 📊 Performance Evaluation")
        st.caption("Optional — enter a name to score recognition accuracy live")

        true_label = st.text_input(
            "Registered name", label_visibility="collapsed",
            placeholder="Enter registered name to evaluate"
        ).strip().lower()

        # reset evaluator for new person
        if true_label != "" and true_label != st.session_state.last_true_label:
            st.session_state.evaluator = Evaluator()
            st.session_state.last_true_label = true_label

        if st.button("Show Results", use_container_width=True):
            st.session_state.show_results = True

        if st.session_state.show_results:

            results, avg_times = st.session_state.evaluator.get_full_results()

            if len(results) > 0:

                st.markdown("##### Model Accuracy")
                cols = st.columns(len(results))
                for col, (model, data) in zip(cols, results.items()):
                    with col:
                        st.metric(
                            model,
                            f"{round(data['accuracy'] * 100, 1)}%",
                            help=f"Avg inference time: {round(avg_times.get(model, 0), 4)}s"
                        )

                with st.expander("View confusion matrices"):
                    for model, data in results.items():
                        st.markdown(f"**{model}**")

                        cm = np.array(data["confusion_matrix"])
                        labels = data["labels"]

                        fig, ax = plt.subplots(figsize=(2, 2))
                        ax.imshow(cm, aspect="auto")

                        ax.set_xticks(range(len(labels)))
                        ax.set_yticks(range(len(labels)))
                        ax.set_xticklabels(labels, fontsize=5)
                        ax.set_yticklabels(labels, fontsize=5)

                        for i in range(len(labels)):
                            for j in range(len(labels)):
                                ax.text(j, i, cm[i, j],
                                        ha="center", va="center", fontsize=5)

                        ax.set_xlabel("Predicted", fontsize=5)
                        ax.set_ylabel("Actual", fontsize=5)

                        st.pyplot(fig)
            else:
                st.info("No evaluation data yet — enter a name and let the camera run for a few seconds.")

    if st.button("🛑 Stop Camera", use_container_width=True):
        if st.session_state.cap is not None and st.session_state.cap.isOpened():
            st.session_state.cap.release()
        st.session_state.cap = None
        st.info("Camera released. Refresh the page to restart it.")
# ================= RIGHT PANEL =================
with right_col:

    with st.container(border=True):
        st.markdown("#### 📷 Live Camera")

        status_placeholder = st.empty()
        frame_placeholder = st.empty()
        greeting_placeholder = st.empty()

        cap = st.session_state.cap

        if cap is None or not cap.isOpened():
            status_placeholder.warning("Camera is not active. Refresh the page to start it.")
        else:
            status_placeholder.success("🟢 Camera active")

            # for controlled loop (not infinite)
            for _ in range(1000):

                ret, frame = cap.read()

                if not ret:
                    st.error("Camera not accessible")
                    break

                frame = cv2.resize(frame, (640, 480))
                faces = detector.detect_faces(frame)
                users_db = db.get_all_users()

                final_name = "detecting..."

                for bbox in faces:
                    face = preprocessor.extract_face(frame, bbox)
                    if face is None:
                        continue

                    face = preprocessor.preprocess_face(face)
                    if face is None:
                        continue

                    embeddings, timings = recognizer.get_embeddings(face)

                    for model_name, t in timings.items():
                        st.session_state.evaluator.add_timing(model_name, t)

                    st.session_state.last_embeddings = embeddings

                    results = []
                    model_predictions = {}

                    for model_name, embedding in embeddings.items():
                        if embedding is None:
                            continue

                        name, distance = recognizer.recognize(
                            embedding,
                            users_db,
                            model_name
                        )

                        name = name.lower()
                        model_predictions[model_name] = name

                        # face_recognizer.py already applies the per-model
                        # threshold and returns "unknown" if nothing matches
                        # confidently — no need to re-filter here.
                        if name != "unknown":
                            results.append((name, distance))

                    if len(results) == 0:
                        detected_name = "unknown"
                    else:
                        detected_name = sorted(results, key=lambda x: x[1])[0][0]

                    st.session_state.name_history.append(detected_name)

                    if len(st.session_state.name_history) > 10:
                        st.session_state.name_history.pop(0)

                    final_name = max(set(st.session_state.name_history),
                                     key=st.session_state.name_history.count)

                    x, y, w, h = bbox
                    color = (0, 255, 0) if final_name != "unknown" else (0, 0, 255)

                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, final_name, (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                    # evaluation
                    if true_label != "":
                        st.session_state.evaluator.add_result(true_label, model_predictions)

                frame_placeholder.image(frame, channels="BGR", use_container_width=True)

                # greeting message
                if final_name not in ("unknown", "detecting..."):
                    greeting_placeholder.success(f"✅ Access granted — Welcome, {final_name.capitalize()}!")
                elif final_name == "unknown":
                    greeting_placeholder.warning("⚠️ Face not recognized")
                else:
                    greeting_placeholder.info("🔍 Scanning...")

                time.sleep(0.03)

