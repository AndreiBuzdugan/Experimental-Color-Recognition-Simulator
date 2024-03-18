import cv2
import numpy as np
import pandas as pd

data_for_csv = []

cap = cv2.VideoCapture(0)

color_to_bgr = {
    "Rosu": (0, 0, 255),
    "Galben": (0, 255, 255),
    "Albastru": (255, 0, 0),
    "Verde": (0, 255, 0),
}

while True:
    ret, frame = cap.read()

    if not ret:
        print("Eroare: Nu s-au putut citi cadrele.")
        break

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for color_lower, color_upper, color_name in [
        (np.array([160, 100, 100], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8), "Rosu"),
        (np.array([20, 100, 100], dtype=np.uint8), np.array([40, 255, 255], dtype=np.uint8), "Galben"),
        (np.array([90, 100, 100], dtype=np.uint8), np.array([120, 255, 255], dtype=np.uint8), "Albastru"),
        (np.array([35, 100, 100], dtype=np.uint8), np.array([85, 255, 255], dtype=np.uint8), "Verde"),
    ]:
        mask = cv2.inRange(hsv_image, color_lower, color_upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

                roi_hsv = hsv_image[y:y + h, x:x + w]
                average_color = np.mean(roi_hsv, axis=(0, 1))

                text_color = color_to_bgr.get(color_name, (255, 255, 255))

                cv2.putText(frame, f'Culoare: {color_name}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)

                data_for_csv.append([average_color[0], average_color[1], average_color[2], color_name])

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

df = pd.DataFrame(data_for_csv, columns=['H', 'S', 'V', 'Culoare'])
df.to_csv('date_antrenament.csv', index=False)

cap.release()
cv2.destroyAllWindows()
