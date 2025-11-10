#!/bin/bash
mkdir -p ~/.mediapipe/modules/pose_landmark/
curl -L -o ~/.mediapipe/modules/pose_landmark/pose_landmark_lite.tflite \
  https://storage.googleapis.com/mediapipe-assets/pose_landmark_lite.tflite
