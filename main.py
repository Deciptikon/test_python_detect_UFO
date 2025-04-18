import cv2
import numpy as np

def init_background_subtractor(history=500, threshold=16):
    """Инициализация фонового вычитателя"""
    return cv2.createBackgroundSubtractorMOG2(
        history=history,
        varThreshold=threshold,
        detectShadows=False
    )

def process_frame(frame, fgbg):
    """Обработка кадра: вычитание фона + морфология"""
    fgmask = fgbg.apply(frame)
    
    # Убираем шум (морфологические операции)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    
    return fgmask

def draw_contours(frame, fgmask, min_area=5):
    """Отрисовка контуров движущихся объектов"""
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
    return frame

def main(video_path):
    """Основной цикл обработки видео"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Ошибка: видео не загружено!")
        return
    
    fgbg = init_background_subtractor()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        fgmask = process_frame(frame, fgbg)
        frame_with_contours = draw_contours(frame.copy(), fgmask)
        
        # Вывод результата
        #cv2.imshow("Original", frame)
        #cv2.imshow("Foreground Mask", fgmask)
        cv2.imshow("Detected Objects", frame_with_contours)
        
        if cv2.waitKey(30) == 27:  # Выход по ESC
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    video_path = "test_UFO.mp4"
    main(video_path)