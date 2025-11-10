#!/bin/bash
echo "📥 Downloading MediaPipe model..."
mkdir -p ~/.mediapipe/modules/pose_landmark/
mkdir -p ./models/
mkdir -p /tmp/mediapipe/modules/pose_landmark/

MODEL_URL="https://storage.googleapis.com/mediapipe-assets/pose_landmark_lite.tflite"

# Download once and copy to all fallback locations
curl -L -o ./models/pose_landmark_lite.tflite "$MODEL_URL"

cp ./models/pose_landmark_lite.tflite ~/.mediapipe/modules/pose_landmark/ 2>/dev/null || true
cp ./models/pose_landmark_lite.tflite /tmp/mediapipe/modules/pose_landmark/ 2>/dev/null || true

echo "✅ Model downloaded to ./models and cache paths."

