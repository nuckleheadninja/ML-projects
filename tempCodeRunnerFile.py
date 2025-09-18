import cv2
import pyttsx3
from ultralytics import YOLO

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Track last spoken objects
spoken_objects = set()

# Load YOLOv8 model (medium size for good accuracy)
model = YOLO("yolov8x.pt")
# Use yolov8n.pt for faster performance if needed

# Start webcam
cap = cv2.VideoCapture(0)
cv2.namedWindow("Live Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Live Object Detection", 400, 400)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model.predict(source=frame, imgsz=640, conf=0.3, device='cpu')  # CPU-only

    current_objects = set()

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]
            conf = box.conf[0].item()

            # Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Draw box and label
            label = f"{name} {conf:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

            current_objects.add(name)

    # Speak newly detected objects
    new_objects = current_objects - spoken_objects
    for obj in new_objects:
        engine.say(f"I see a {obj}")
        engine.runAndWait()

    spoken_objects.update(new_objects)

    # Display frame
    cv2.imshow("Live Object Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()