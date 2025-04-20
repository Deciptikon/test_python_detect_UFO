from ultralytics import YOLO
import cv2

# Загрузка модели (скачается автоматически)
# nano-версия (самая лёгкая) "yolov8n.pt"
# лучшие веса обучения "runs/detect/train/weights/best.pt"  
model = YOLO("runs/detect/train/weights/best.pt")  
print(model.names)
# Источник видео (0 - камера, или путь к файлу)
cap = cv2.VideoCapture("test_UFO.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model.predict(frame, classes=[0], conf=0.2)
    
    annotated_frame = results[0].plot()
    cv2.imshow("Drone Detection", annotated_frame)
    
    if cv2.waitKey(1) == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()