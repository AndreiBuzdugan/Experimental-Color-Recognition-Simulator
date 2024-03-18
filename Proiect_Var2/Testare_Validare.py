import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model('culoare_classifier_model')

cap = cv2.VideoCapture(0)

color_mapping = {
    0: "Albastru",
    1: "Galben",
    2: "Rosu",
    3: "Verde",
}

while True:
    ret, frame = cap.read()
    if not ret:
        print("Eroare: Nu s-au putut citi cadrele video.")
        break

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    color_ranges = [
        (np.array([160, 100, 100], dtype=np.uint8), np.array([180, 255, 255], dtype=np.uint8), "Rosu"),
        (np.array([20, 100, 100], dtype=np.uint8), np.array([40, 255, 255], dtype=np.uint8), "Galben"),
        (np.array([90, 100, 100], dtype=np.uint8), np.array([120, 255, 255], dtype=np.uint8), "Albastru"),
        (np.array([35, 100, 100], dtype=np.uint8), np.array([85, 255, 255], dtype=np.uint8), "Verde"),
    ]

    for color_lower, color_upper, color_name in color_ranges:
        mask = cv2.inRange(hsv_image, color_lower, color_upper)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:
                x, y, w, h = cv2.boundingRect(contour)
                roi_hsv = hsv_image[y:y + h, x:x + w]
                average_hsv = np.mean(np.mean(roi_hsv, axis=0), axis=0).reshape(1, -1)

                prediction = model.predict(average_hsv)
                predicted_class = np.argmax(prediction)
                predicted_color = color_mapping.get(predicted_class, "Necunoscut")

                if predicted_color == "Albastru":
                    text_color = (255, 0, 0)
                elif predicted_color == "Galben":
                    text_color = (0, 255, 255)
                elif predicted_color == "Rosu":
                    text_color = (0, 0, 255)
                elif predicted_color == "Verde":
                    text_color = (0, 255, 0)
                else:
                    text_color = (255, 255, 255)

                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                cv2.putText(frame, f'Predicted: {predicted_color}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)


    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
