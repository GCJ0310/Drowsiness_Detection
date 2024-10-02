# Drowsiness Detection System Using Eye Aspect Ratio (EAR) and MediaPipe

This project implements a real-time drowsiness detection system using Eye Aspect Ratio (EAR) and the MediaPipe library's face mesh solution. It analyzes the eye landmarks from a video feed and calculates the EAR to determine if the user is drowsy.

## Features

- **Real-time Detection:** Detects drowsiness by analyzing the Eye Aspect Ratio (EAR).
- **Face Mesh with MediaPipe:** Utilizes MediaPipe's face mesh to identify key facial landmarks, including eye regions.
- **Customizable Threshold:** Alerts the user when drowsiness is detected based on a configurable EAR threshold and frame count.
- **Visual Representation:** Enlarges and outlines the eye regions in real-time for better visibility.

## Requirements

Make sure you have the following dependencies installed:

- Python 3.10+
- OpenCV
- MediaPipe
- SciPy
- NumPy

You can install the required dependencies using `pip`:

```bash
pip install opencv-python mediapipe scipy numpy
```

## Eye Aspect Ratio (EAR) Calculation

The EAR is calculated by analyzing the vertical and horizontal distances between eye landmarks. If the EAR value drops below a certain threshold (`EYE_AR_THRESH`) for a specified number of consecutive frames, the system issues a drowsiness alert.

### Formula:

```
EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
```

Where `p1` to `p6` represent the eye landmark points obtained from the face mesh.

## How It Works

1. The program captures a video stream (via a webcam) using OpenCV.
2. MediaPipe processes the video stream to detect facial landmarks.
3. The eye landmarks are extracted, and the EAR is calculated for both eyes.
4. If the EAR value falls below a threshold for a set number of frames, a drowsiness alert is displayed on the screen.
5. The system visually enlarges and draws contours around the eyes for better clarity.

## Usage

1. Clone the repository:

```bash
git clone https://github.com/GCJ0310/Drowsiness_Detection.git
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Run the script:

```bash
python facial_landmarks_detection.py
```

The webcam feed will open, and the system will begin detecting drowsiness. A "DROWSINESS ALERT!" message will appear on the screen when drowsiness is detected.

### Customizable Parameters:

- **EYE_AR_THRESH:** The threshold for the Eye Aspect Ratio (EAR) to determine drowsiness. Default is `0.25`.
- **CLOSED_FRAMES:** The number of consecutive frames where EAR is below the threshold to trigger the alert. Default is `20`.
- **Enlargement Scale:** The `scale` value in the `enlarge_eye` function adjusts the size of the eye contour for better visibility.

### Controls:

- Press **'q'** to exit the program.

## Example Output

- The real-time video feed will display the EAR value.
- When drowsiness is detected, a "DROWSINESS ALERT!" message will be displayed in red text.

## Project Structure

```
.
├── facial_landmarks_detection.py  # Main script for drowsiness detection
├── README.md                      # Documentation
└── requirements.txt               # Python dependencies
```

## Contributing

Feel free to submit issues or pull requests if you have suggestions or improvements.

## License

This project is licensed under the MIT License.

---

### Future Enhancements

- Add sound alerts for drowsiness detection.
- Implement head pose detection to complement eye-based drowsiness detection.
- Explore more advanced face detection methods to improve accuracy.

---
