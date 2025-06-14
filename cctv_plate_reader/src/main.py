import cv2
import torch
import easyocr
from ultralytics import YOLO

# Load YOLOv8
model = YOLO('yolov8n.pt')
ocr_reader = easyocr.Reader(['en'])

# Load video
cap = cv2.VideoCapture('your_video.mp4')

# Function to fix dark + blur images
def enhance_image(img):
    # Brighten
    bright = cv2.convertScaleAbs(img, alpha=2.2, beta=40)

    # Sharpen
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    sharp = cv2.filter2D(bright, -1, kernel)

    return sharp

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    boxes = results[0].boxes.xyxy

    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        roi = frame[y1:y2, x1:x2]

        # ENHANCE cropped region
        roi = enhance_image(roi)

        # OCR to read number
        text = ocr_reader.readtext(roi, detail=0)
        plate = text[0] if text else "???"

        # Show results
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, plate, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    cv2.imshow("Night Vision Plate Reader", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
