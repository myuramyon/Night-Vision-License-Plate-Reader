import streamlit as st
import cv2
import tempfile
from main import enhance_image
from ultralytics import YOLO
import easyocr

st.title("🚗 Night Vision Vehicle Plate Reader")

uploaded_video = st.file_uploader("Upload a CCTV Video", type=["mp4", "avi"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)

    model = YOLO('yolov8n.pt')
    ocr_reader = easyocr.Reader(['en'])

    stframe = st.empty()
    plate_list = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        boxes = results[0].boxes.xyxy

        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            roi = frame[y1:y2, x1:x2]
            roi = enhance_image(roi)
            text = ocr_reader.readtext(roi, detail=0)
            plate = text[0] if text else "???"
            plate_list.append(plate)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, plate, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        stframe.image(frame, channels="BGR")

    cap.release()
    st.success("Done processing video.")
    st.write("### Detected Plates:")
    st.dataframe({"Plates": plate_list})
