import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import numpy as np
import cv2
import os
from io import BytesIO
from moviepy import VideoFileClip
import tempfile

print("----------------------------------------")

if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.uploaded_file = None


# IMAGE PROCESS
def image_data(result):
    boxes = result.boxes
    class_ids = boxes.cls
    confidences = boxes.conf
    xywh = boxes.xywh

    data = []
    for i in range(len(class_ids)):
        class_name = result.names[int(class_ids[i])]
        confidence = confidences[i]
        x_center, y_center, width, height = xywh[i]
        x0, y0 = int(x_center - width / 2), int(y_center - height / 2)
        x1, y1 = int(x_center + width / 2), int(y_center + height / 2)
        confidence = confidence * 100
        confidence = f"{confidence:.2f} %"
        width = f"{width:.2f}"
        height = f"{height:.2f}"
        data.append([i + 1, class_name, confidence, x0, x1, y0, y1, width, height])
    df = pd.DataFrame(
        data,
        columns=[
            "ID",
            "Class",
            "Confidence",
            "x0",
            "x1",
            "y0",
            "y1",
            "Width (cm)",
            "Height (cm)",
        ],
    )
    return df


def image_plot(result, df):
    result_img = np.array(result.orig_img)
    for i, row in df.iterrows():
        x0, y0, x1, y1 = row["x0"], row["y0"], row["x1"], row["y1"]
        id_label = row["ID"]
        label_x = x0 - 20
        label_y = int(y0 + (y1 - y0) / 2)
        result_img = cv2.rectangle(result_img, (x0, y0), (x1, y1), (0, 255, 0), 2)
        result_img = cv2.putText(
            result_img,
            str(id_label),
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return result_img


# VIDEO PROCESS
def video_process(uploaded_file, model):
    video_bytes = uploaded_file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
        temp_file.write(video_bytes)
        temp_file.flush()  # Ensure all data is written
        temp_file_path = temp_file.name
    if not os.path.exists(temp_file_path):
        st.error(f"Error: Temporary file not found at {temp_file_path}")
        return
    try:
        clip = VideoFileClip(temp_file_path)
    except Exception as e:
        st.error(f"Error loading video: {e}")
        return
    frame_placeholder = st.empty()
    for frame in clip.iter_frames(dtype="uint8"):
        # Run YOLO on the frame
        results = model(frame)
        annotated_frame = results[0].plot()

        # Display the frame in Streamlit
        frame_placeholder.image(
            annotated_frame, channels="RGB", use_container_width=True
        )

    # video_bytes = uploaded_file.read()
    # with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_file:
    #     temp_file.write(video_bytes)
    #     temp_file.flush()  # Ensure all data is written
    #     temp_file_path = temp_file.name
    # if not os.path.exists(temp_file_path):
    #     st.error(f"Error: Temporary file not found at {temp_file_path}")
    #     return
    # try:
    #     clip = VideoFileClip(temp_file_path)
    # except Exception as e:
    #     st.error(f"Error loading video: {e}")
    #     return
    # total_frames = int(clip.reader.nframes)
    # total_duration = int(clip.duration)
    # def process_frame(frame):
    #     # Run YOLO on the frame
    #     results = model(frame)
    #     annotated_frame = results[0].plot()  # Add annotations to the frame
    #     return annotated_frame
    # processed_clip = clip.fl_image(process_frame)
    # processed_video_path = "processed_video.mp4"
    # processed_clip.write_videofile(processed_video_path, codec="libx264", audio_codec="aac")
    # with open(processed_video_path, "rb") as video_file:
    #     st.video(video_file.read())
    # os.remove(temp_file_path)
    # os.remove(processed_video_path)
    return None


# MODEL LOAD
model = YOLO("weight.pt")

# SIDEBAR
file_type = st.sidebar.selectbox("Select file type", ["Image", "Video"])
uploaded_file = None

if file_type == "Image":
    uploaded_file = st.sidebar.file_uploader(
        "Upload an image", type=["png", "jpg", "jpeg"]
    )

elif file_type == "Video":
    uploaded_file = st.sidebar.file_uploader(
        "Upload a video", type=["mp4", "mov", "avi"]
    )

# Confidence Level
confidence_level = st.sidebar.slider(
    "Confidence Level", min_value=0, max_value=100, step=10
)

# MAIN PAGE
if uploaded_file is not None:
    if file_type == "Image":
        st.header("Image Detection")
        image = Image.open(uploaded_file)
        result = model(image, conf=confidence_level / 100)
        result = result[0]
        data = image_data(result)
        result_img = image_plot(result, data)
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Upload")
        with col2:
            st.image(result_img, caption="Result")
        st.write(data)

    elif file_type == "Video":
        st.header("Video Detection")
        col1, col2 = st.columns(2)
        with col1:
            st.video(uploaded_file)
        with col2:
            video_process(uploaded_file, model)
