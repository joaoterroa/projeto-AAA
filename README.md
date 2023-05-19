# **Project AAA 2022/2023**

Recognizing American Sign Language (ASL) Alphabet in Real-Time using Deep Learning

This project aimed to develop a deep vision neural network that accurately recognizes the ASL alphabet in real-time. We have trained a Convolutional Neural Network (CNN) on an image dataset of ASL alphabets and utilized an autoencoder for image segmentation. The trained models are then integrated with a webcam for real-time predictions.

<img src="imgs/char_cam_test.gif" alt="Char Cam Test GIF" width="100%">

## Authors

- João Terroa - fc53117
- David Conceição - fcxxxxx

## Requirements

To run this project, you need:

- Python 3.8
- Anaconda virtual environment
- Necessary dependencies from the environment.yml file

## Setting Up the Project

### Step 1: Create a Virtual Environment

We recommend using Anaconda to create a virtual environment for the project. This can be done by running the following command:

```bash
conda create --name <env> --file environment.yml
```

Replace `<env>` with the name of your environment.

### Step 2: Download the Models

You can download the .h5 files for the trained models from [here](https://drive.google.com/drive/folders/1ZhxHnUisjtoPmuSekLHPObAlvYcrgkEI?usp=share_link). Place the files in the `app` directory.

### Step 3: Running the Application

After setting up the environment, you can run the application using:

```bash
python real_time.py
```

This script starts the webcam and applies the trained models to recognize the ASL alphabet in real-time.

## Code Overview

### real_time.py

This script is the main entry point of the application. It starts the webcam and processes the video frames using the trained models. The hand landmarks are detected using MediaPipe's Hand solution, and a Region of Interest (ROI) is extracted around the hand. The ROI is then passed to the segmentation model to extract the hand from the background. The segmented image is then passed to the prediction model to recognize the ASL letter being signed. The top 3 predictions along with their probabilities are displayed on the video frame.

## Model Information

Two models are used in this project:

- **Sign Recognition Model (`sign_2.h5`)**: A CNN model trained to recognize ASL alphabet signs from images. It is used to predict the sign being made in the segmented hand images.

- **Segmentation Model (`segment.h5`)**: An autoencoder model trained to segment hand from the background in images. It is used to extract the hand from the ROI.

## Troubleshooting

- If you encounter an error related to an unavailable webcam, make sure the correct webcam is set up and is not being used by another application.

- If the sign recognition is not accurate, ensure the hand is well illuminated and not too close or too far from the camera.

## Contribution

This is a university project, and the codebase isn't open for contribution. However, feel free to fork the repository and use the code according to the specified license.

## License

This project is licensed under the MIT License. See the [LICENSE](https://opensource.org/license/mit/) file for more information.
