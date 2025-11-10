#!/bin/bash
mkdir -p ~/.mediapipe/modules/pose_landmark/

echo "Downloading MediaPipe pose model..."
curl -L -o ~/.mediapipe/modules/pose_landmark/pose_landmark_lite.tflite \
  https://storage.googleapis.com/mediapipe-assets/pose_landmark_lite.tflite
