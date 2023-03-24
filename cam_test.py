from keras.models import load_model
import keras
import cv2
import numpy as np


# function for top_k_categorical_accuracy
def top_3_accuracy(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)


# load the model
model = load_model("sign.h5", custom_objects={"top_3_accuracy": top_3_accuracy})

# make a label dictionary where the key is the label and the value is the letter
label_dict = {
    0: "A",
    1: "B",
    2: "C",
    3: "D",
    4: "E",
    5: "F",
    6: "G",
    7: "H",
    8: "I",
    9: "J",
    10: "K",
    11: "L",
    12: "M",
    13: "N",
    14: "O",
    15: "P",
    16: "Q",
    17: "R",
    18: "S",
    19: "T",
    20: "U",
    21: "V",
    22: "W",
    23: "X",
    24: "Y",
    25: "Z",
    26: "del",
    27: "nothing",
    28: "space",
}

# capture video from the MacBook webcam
cap = cv2.VideoCapture(0)

while True:
    # read a frame from the video
    ret, frame = cap.read()

    # flip the frame horizontally
    frame = cv2.flip(frame, 1)

    # extract the region of interest
    roi = frame[
        frame.shape[0] // 2 - 250 : frame.shape[0] // 2 + 250,
        frame.shape[1] // 2 - 250 : frame.shape[1] // 2 + 250,
        :,
    ]

    # resize the ROI to match the input size of the model
    img = cv2.resize(roi, (100, 100))

    # reshape the ROI to match the input size of the model
    img = img.reshape(100, 100, 3)

    # normalize the pixel values
    img = img.astype("float32") / 255.0

    # make a prediction using the trained model
    pred = model.predict(np.expand_dims(img, axis=0))[0]
    pred_label = np.argmax(pred)
    pred_label_letter = label_dict[pred_label]
    pred_prob = pred[pred_label]
    top_pred_labels = np.argsort(pred)[::-1][:3]
    top_pred_labels_letter = [label_dict[label] for label in top_pred_labels]
    top_pred_probs = pred[top_pred_labels]

    # display the frame with the ROI and the predicted label and top three predictions
    frame = cv2.rectangle(
        frame,
        (frame.shape[1] // 2 - 250, frame.shape[0] // 2 - 250),
        (frame.shape[1] // 2 + 250, frame.shape[0] // 2 + 250),
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"Prediction: {pred_label_letter} ({pred_prob:.2f})",
        (50, 50),
        cv2.FONT_HERSHEY_DUPLEX,
        1,
        (171, 100, 24),
        2,
        cv2.LINE_AA,
    )
    for i, label in enumerate(top_pred_labels_letter):
        cv2.putText(
            frame,
            f"{i+1}. {label} ({top_pred_probs[i]:.2f})",
            (50, 100 + 50 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (62, 138, 43),
            2,
            cv2.LINE_AA,
        )
    cv2.imshow("frame", frame)

    # exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# release the video capture and close the window
cap.release()
cv2.destroyAllWindows()


# # Initialize the webcam:
# video = cv2.VideoCapture(0)


# # Define a function to preprocess the image:
# def preprocess(img):
#     img = cv2.resize(img, (100, 100))
#     # convert img to rgb
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     # reshape the image to (1, 100, 100, 3)
#     img = img.reshape(100, 100, 3)
#     # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     # img = np.expand_dims(img, axis=-1)
#     img = img.astype("float32") / 255.0
#     return img


# # Use a loop to continuously capture frames from the webcam, preprocess them, and feed them into the model for prediction:

# while True:
#     _, frame = video.read()
#     img = preprocess(frame)
#     pred = model.predict(np.array([img]))
#     letter = chr(ord("a") + np.argmax(pred[0]))
#     cv2.putText(
#         frame,
#         letter,
#         (10, 50),
#         cv2.FONT_HERSHEY_SIMPLEX,
#         1,
#         (0, 255, 0),
#         2,
#         cv2.LINE_AA,
#     )

#     cv2.imshow("frame", frame)
