import cv2
import torch
import easyocr
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('yolov8n.pt')
ocr_reader = easyocr.Reader(['en'])

# Load your video
cap = cv2.VideoCapture('your_video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    boxes = results[0].boxes.xyxy

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        roi = frame[y1:y2, x1:x2]

        # Make it brighter if it’s dark
        roi = cv2.convertScaleAbs(roi, alpha=2.0, beta=50)

        # OCR to read number
        text = ocr_reader.readtext(roi, detail=0)
        plate = text[0] if text else "???"

        # Show number on screen
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, plate, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Vehicle Plate Reader", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
