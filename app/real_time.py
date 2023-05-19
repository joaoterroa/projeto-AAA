import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model
import keras
from class_names import LetterList

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

class_names = LetterList.getList()

input_dim = 100


def preprocess_image(roi):
    # image = cv.cvtColor(my_img, cv.COLOR_BGR2RGB)
    img_resized = cv2.resize(roi, (input_dim, input_dim))
    img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img_normalized = img_gray / 255.0
    return np.expand_dims(img_normalized, axis=0)


def predict(model, roi):
    processed_image = preprocess_image(roi)
    predictions = model.predict(processed_image)[0]
    top_3_indices = np.argsort(predictions)[-3:][::-1]
    top_3_probs = predictions[top_3_indices]
    return [(class_names[i], prob) for i, prob in zip(top_3_indices, top_3_probs)]


def segment_image(model, roi):
    seg_image = cv2.resize(roi, (128, 128))
    seg_image = np.expand_dims(seg_image, axis=0)
    seg_image = model.predict(seg_image)
    return seg_image[0]


def normal_image(model, roi):
    norm_image = cv2.resize(roi, (128, 128))
    norm_image = np.expand_dims(norm_image, axis=0)
    norm_image = model.predict(norm_image)
    return norm_image[0]


def resize_roi(image, size=(100, 100)):
    if image.size == 0:
        return None
    return cv2.resize(image, size)


def top_3_accuracy(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


def capture_frames(cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue
        frame = cv2.flip(frame, 1)
        yield frame


# model = load_model(
#     "sign.h5",
#     custom_objects={"top_3_accuracy": tf.keras.metrics.TopKCategoricalAccuracy(k=3)},
# )

model = load_model("sign_2.h5")
segmentation_model = load_model("segment.h5")

skip_frame = 2
counter = 0


# For webcam input:
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            print("Ignoring empty camera frame.")
            continue

        image = cv2.flip(image, 1)
        image.flags.writeable = False

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        rgb_image = image

        # Draw the hand annotations on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    # mp_drawing_styles.get_default_hand_landmarks_style(),
                    # mp_drawing_styles.get_default_hand_connections_style(),
                )

                hand_points = []
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * image.shape[1]), int(lm.y * image.shape[0])
                    hand_points.append((x, y))

                x_min, y_min = np.min(hand_points, axis=0)
                x_max, y_max = np.max(hand_points, axis=0)

                extra_length = 50

                cv2.rectangle(
                    image,
                    (x_min - extra_length, y_min - extra_length),
                    (x_max + extra_length, y_max + extra_length),
                    (0, 255, 0),
                    2,
                )
                roi = image[
                    y_min - extra_length : y_max + extra_length,
                    x_min - extra_length : x_max + extra_length,
                ]
                if roi.size > 0:
                    segmented_image = segment_image(segmentation_model, roi)

                    # scale the image to 0-255
                    segmented_image = (segmented_image * 255).astype(np.uint8)

                    h, w = segmented_image.shape[0:2]

                    # Define the margin size
                    margin = 50

                    # Place the segmented image in the bottom left corner
                    # with a 50-pixel margin
                    image[-h - margin : -margin, margin : margin + w] = segmented_image
                    predictions = predict(model, segmented_image)

                    for i, (label, prob) in enumerate(predictions):
                        y = 30 + i * 30
                        color = (171, 100, 24) if i == 0 else (62, 138, 43)
                        cv2.putText(
                            image,
                            f"{label}: {prob * 100:.1f}%",
                            (50, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            color,
                            2,
                        )

                    # put the text on top of the roi
                    cv2.putText(
                        image,
                        f"Letter: {predictions[0][0]}",
                        (x_min - extra_length + 5, y_min - extra_length - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (255, 255, 255),
                        3,
                        cv2.LINE_AA,
                    )

        # Flip the image horizontally for a selfie-view display.
        cv2.imshow("ASL Webcam", image)
        if cv2.waitKey(5) & 0xFF == ord("q"):
            break
cap.release()
