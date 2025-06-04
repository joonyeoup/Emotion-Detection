import cv2
import numpy as np
import pickle
import mediapipe as mp
import argparse
import time
from collections import deque
import sys
import os

# Import the PoseFeatureExtractor class
from extract import PoseFeatureExtractor

class RealTimeEmotionRecognizer:
    def __init__(self, model_path, sequence_length=30, confidence_threshold=0.5):
        """
        Initialize the real-time emotion recognizer
        
        Args:
            model_path: Path to the trained model pickle file
            sequence_length: Number of frames to collect for feature extraction
            confidence_threshold: Minimum confidence for pose detection
        """
        self.sequence_length = sequence_length
        self.confidence_threshold = confidence_threshold
        
        # Load the trained model, scaler, and label encoder
        self.load_model(model_path)
        
        # Initialize MediaPipe pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Initialize feature extractor
        self.feature_extractor = PoseFeatureExtractor()
        
        # Store pose sequences
        self.pose_sequence = deque(maxlen=sequence_length)
        
        # Prediction smoothing
        self.prediction_history = deque(maxlen=10)
        
    def load_model(self, model_path):
        """
        Load the trained model, scaler, and label encoder
        
        Args:
            model_path: Path to the model file
        """
        try:
            with open(model_path, 'rb') as f:
                self.model, self.scaler, self.label_encoder = pickle.load(f)
            print(f"Model loaded successfully from {model_path}")
            print(f"Emotion classes: {self.label_encoder.classes_}")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    
    def mediapipe_to_openpose_format(self, landmarks):
        """
        Convert MediaPipe pose landmarks to OpenPose format
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            Pose keypoints in OpenPose format (25 keypoints x 3 coordinates)
        """
        # MediaPipe has 33 landmarks, we need to map to OpenPose 25 keypoints
        # OpenPose body keypoint order:
        # 0: Nose, 1: Neck, 2: RShoulder, 3: RElbow, 4: RWrist,
        # 5: LShoulder, 6: LElbow, 7: LWrist, 8: MidHip, 9: RHip,
        # 10: RKnee, 11: RAnkle, 12: LHip, 13: LKnee, 14: LAnkle,
        # 15: REye, 16: LEye, 17: REar, 18: LEar, 19-24: Foot keypoints (we'll set to 0)
        
        if landmarks is None:
            return np.zeros((25, 3))
        
        # MediaPipe landmark indices
        mp_to_op_mapping = {
            0: 0,   # Nose
            1: 15,  # Left eye (inner)
            2: 16,  # Right eye (inner) 
            7: 17,  # Left ear
            8: 18,  # Right ear
            11: 5,  # Left shoulder
            12: 2,  # Right shoulder
            13: 6,  # Left elbow
            14: 3,  # Right elbow
            15: 7,  # Left wrist
            16: 4,  # Right wrist
            23: 12, # Left hip
            24: 9,  # Right hip
            25: 13, # Left knee
            26: 10, # Right knee
            27: 14, # Left ankle
            28: 11, # Right ankle
        }
        
        # Initialize OpenPose keypoints
        openpose_kpts = np.zeros((25, 3))
        
        # Convert landmarks
        for mp_idx, op_idx in mp_to_op_mapping.items():
            if mp_idx < len(landmarks.landmark):
                landmark = landmarks.landmark[mp_idx]
                openpose_kpts[op_idx, 0] = landmark.x * 640  # Assuming 640x480 resolution
                openpose_kpts[op_idx, 1] = landmark.y * 480
                openpose_kpts[op_idx, 2] = landmark.visibility  # Use visibility as confidence
        
        # Calculate neck position (midpoint between shoulders)
        if openpose_kpts[2, 2] > 0 and openpose_kpts[5, 2] > 0:  # Both shoulders detected
            openpose_kpts[1, 0] = (openpose_kpts[2, 0] + openpose_kpts[5, 0]) / 2
            openpose_kpts[1, 1] = (openpose_kpts[2, 1] + openpose_kpts[5, 1]) / 2
            openpose_kpts[1, 2] = min(openpose_kpts[2, 2], openpose_kpts[5, 2])
        
        # Calculate mid hip position
        if openpose_kpts[9, 2] > 0 and openpose_kpts[12, 2] > 0:  # Both hips detected
            openpose_kpts[8, 0] = (openpose_kpts[9, 0] + openpose_kpts[12, 0]) / 2
            openpose_kpts[8, 1] = (openpose_kpts[9, 1] + openpose_kpts[12, 1]) / 2
            openpose_kpts[8, 2] = min(openpose_kpts[9, 2], openpose_kpts[12, 2])
        
        return openpose_kpts
    
    def process_frame(self, frame):
        """
        Process a single frame to extract pose and make predictions
        
        Args:
            frame: Input video frame
            
        Returns:
            Processed frame with pose and emotion overlays
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process pose
        results = self.pose.process(rgb_frame)
        
        # Draw pose landmarks
        if results.pose_landmarks:
            self.mp_draw.draw_landmarks(
                frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            
            # Convert to OpenPose format
            pose_keypoints = self.mediapipe_to_openpose_format(results.pose_landmarks)
            
            # Add to sequence
            self.pose_sequence.append(pose_keypoints)
            
            # Make prediction if we have enough frames
            if len(self.pose_sequence) >= self.sequence_length:
                emotion, confidence = self.predict_emotion()
                self.prediction_history.append((emotion, confidence))
                
                # Get smoothed prediction
                smoothed_emotion, smoothed_confidence = self.get_smoothed_prediction()
                
                # Draw emotion prediction
                self.draw_emotion_overlay(frame, smoothed_emotion, smoothed_confidence)
        
        # Draw status information
        self.draw_status_overlay(frame)
        
        return frame
    
    def predict_emotion(self):
        """
        Predict emotion from current pose sequence
        
        Returns:
            Predicted emotion and confidence
        """
        try:
            # Convert sequence to numpy array
            pose_array = np.array(list(self.pose_sequence))
            
            # Extract features
            features = self.feature_extractor.extract_features_from_pose_sequence(pose_array)
            
            if features is not None:
                # Reshape for single prediction
                features = features.reshape(1, -1)
                
                # Scale features
                features_scaled = self.scaler.transform(features)
                
                # Make prediction
                prediction_proba = self.model.predict_proba(features_scaled)[0]
                predicted_class = np.argmax(prediction_proba)
                confidence = prediction_proba[predicted_class]
                
                # Convert to emotion name
                emotion = self.label_encoder.classes_[predicted_class]
                
                return emotion, confidence
            else:
                return "Unknown", 0.0
        
        except Exception as e:
            print(f"Error in prediction: {e}")
            return "Error", 0.0
    
    def get_smoothed_prediction(self):
        """
        Get smoothed prediction based on recent history
        
        Returns:
            Smoothed emotion and confidence
        """
        if not self.prediction_history:
            return "Unknown", 0.0
        
        # Count emotion occurrences in recent history
        emotion_counts = {}
        confidence_sums = {}
        
        for emotion, confidence in self.prediction_history:
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
                confidence_sums[emotion] = 0
            emotion_counts[emotion] += 1
            confidence_sums[emotion] += confidence
        
        # Find most frequent emotion
        most_frequent_emotion = max(emotion_counts.keys(), key=lambda k: emotion_counts[k])
        avg_confidence = confidence_sums[most_frequent_emotion] / emotion_counts[most_frequent_emotion]
        
        return most_frequent_emotion, avg_confidence
    
    def draw_emotion_overlay(self, frame, emotion, confidence):
        """
        Draw emotion prediction overlay on frame
        
        Args:
            frame: Video frame
            emotion: Predicted emotion
            confidence: Prediction confidence
        """
        # Define colors for different emotions
        emotion_colors = {
            'Happiness': (0, 255, 0),    # Green
            'Neutral': (255, 255, 255),  # White
            'Sadness': (255, 0, 0),      # Blue
            'Anger': (0, 0, 255),        # Red
            'Fear': (128, 0, 128),       # Purple
            'Surprise': (0, 255, 255),   # Yellow
            'Disgust': (0, 128, 255),    # Orange
        }
        
        color = emotion_colors.get(emotion, (255, 255, 255))
        
        # Draw emotion text
        text = f"{emotion}: {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle
        cv2.rectangle(frame, (10, 10), (text_width + 20, text_height + 20), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(frame, text, (15, text_height + 15), font, font_scale, color, thickness)
        
        # Draw confidence bar
        bar_width = 200
        bar_height = 20
        bar_x = 10
        bar_y = text_height + 40
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        
        # Confidence bar
        confidence_width = int(bar_width * confidence)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + confidence_width, bar_y + bar_height), color, -1)
        
        # Bar outline
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
    
    def draw_status_overlay(self, frame):
        """
        Draw status information overlay
        
        Args:
            frame: Video frame
        """
        height, width = frame.shape[:2]
        
        # Status text
        status_text = f"Frames collected: {len(self.pose_sequence)}/{self.sequence_length}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 1
        
        # Get text size
        (text_width, text_height), _ = cv2.getTextSize(status_text, font, font_scale, thickness)
        
        # Draw status at bottom right
        x = width - text_width - 10
        y = height - 20
        
        cv2.putText(frame, status_text, (x, y), font, font_scale, (255, 255, 255), thickness)
    
    def run(self, camera_index=0):
        """
        Run the real-time emotion recognition
        
        Args:
            camera_index: Camera device index
        """
        # Initialize camera
        cap = cv2.VideoCapture(camera_index)
        
        if not cap.isOpened():
            print(f"Error: Could not open camera {camera_index}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Starting real-time emotion recognition...")
        print("Press 'q' to quit, 'r' to reset sequence")
        
        # Performance tracking
        fps_counter = 0
        fps_start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    print("Error: Could not read frame from camera")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Calculate and display FPS
                fps_counter += 1
                current_time = time.time()
                if current_time - fps_start_time >= 1.0:
                    fps = fps_counter / (current_time - fps_start_time)
                    fps_counter = 0
                    fps_start_time = current_time
                    
                    # Draw FPS
                    cv2.putText(processed_frame, f"FPS: {fps:.1f}", (10, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                
                # Display frame
                cv2.imshow('Real-time Emotion Recognition', processed_frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r'):
                    self.pose_sequence.clear()
                    self.prediction_history.clear()
                    print("Sequence reset")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released and windows closed")

def main():
    parser = argparse.ArgumentParser(description='Real-time emotion recognition from camera')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to the trained model pickle file')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera device index (default: 0)')
    parser.add_argument('--sequence_length', type=int, default=30,
                       help='Number of frames to collect for prediction (default: 30)')
    parser.add_argument('--confidence_threshold', type=float, default=0.5,
                       help='Minimum confidence threshold for pose detection (default: 0.5)')
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        return
    
    # Initialize and run the recognizer
    recognizer = RealTimeEmotionRecognizer(
        model_path=args.model_path,
        sequence_length=args.sequence_length,
        confidence_threshold=args.confidence_threshold
    )
    
    recognizer.run(camera_index=args.camera)

if __name__ == "__main__":
    main()
