## MoveNet Pose Estimation

This repository contains code for performing real-time pose estimation using the MoveNet model. The code utilizes TensorFlow Lite for model inference and OpenCV for video capture and rendering.

### Requirements

To run this code, you need the following dependencies:

- TensorFlow (2.x)
- TensorFlow Lite
- NumPy
- OpenCV (cv2)
- matplotlib

### Pose Estimation

The code performs real-time pose estimation using the MoveNet model. The model is loaded from the `./models/lite-model_movenet_singlepose_lightning_3.tflite` file. The input video frames are captured using OpenCV's `VideoCapture` function.

The pose estimation process involves the following steps:

1. Resize the input frame to a fixed size (192x192 pixels) to match the model's input size.
2. Convert the resized frame to a `tf.float32` tensor.
3. Run the model inference using the TensorFlow Lite interpreter.
4. Obtain the predicted keypoints with confidence scores.
5. Classify the pose based on specific angle measurements.
6. Render the pose estimation results on the frame.
7. Display the frame with the pose estimation overlay using OpenCV.

### Customization

The code includes additional functions for pose classification and rendering. These functions are stored in separate files and imported when needed. You can modify these functions or add your own functionality to suit your requirements.

- `calculateangle.py` contains a function to calculate the angle between three landmarks.
- `classifypose.py` includes a function to classify poses based on specific angle measurements.
- `draw.py` provides functions to draw pose connections and keypoints on the frame.
- `joints.py` is a helper function to extract relevant keypoints from the model's output.

### Usage

To run the code, ensure that you have a webcam connected to your system. Then, execute the script. It will capture video frames from the webcam, perform pose estimation, classify the pose, and display the results in real-time.

Press the 'q' key to exit the application.

Feel free to experiment with the code and customize it to your specific needs.

### Acknowledgments

The MoveNet model used in this code is developed by TensorFlow and can be found in the TensorFlow Model Garden repository.

### References

- TensorFlow: https://www.tensorflow.org/
- TensorFlow Lite: https://www.tensorflow.org/lite
- OpenCV: https://opencv.org/
- MoveNet Model: https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md
