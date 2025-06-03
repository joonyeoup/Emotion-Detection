import os
import json
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Constants
EMOTIONS = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
NUM_KEYPOINTS = 25  # OpenPose keypoints as described in the paper

class EmotionDetector:
    def __init__(self, model_path=None):
        """
        Initialize the emotion detection system using body pose
        
        Args:
            model_path: Path to a pre-trained model (optional)
        """
        self.pose_estimator = self._initialize_pose_estimator()
        self.model = self._load_model(model_path) if model_path else None
        
    def _initialize_pose_estimator(self):
        """
        Initialize OpenPose for pose estimation
        Returns: OpenPose model
        """
        # Check if OpenPose is installed and available
        try:
            # Use OpenPose Python API if available
            # For this code to work, you need to have OpenPose installed
            # and properly configured in your system
            from openpose import pyopenpose as op
            
            # Configure OpenPose
            params = {
                "model_folder": "models/",  # Path to OpenPose models
                "net_resolution": "656x368",
                "number_people_max": 1,  # We only need one person as in MEED dataset
                "keypoint_scale": 3,  # Output scale
            }
            
            # Starting OpenPose
            opWrapper = op.WrapperPython()
            opWrapper.configure(params)
            opWrapper.start()
            return opWrapper
            
        except ImportError:
            print("OpenPose not found. Will use pre-extracted pose data or another pose estimator.")
            return None
    
    def _load_model(self, model_path):
        """
        Load a pre-trained emotion classification model
        
        Args:
            model_path: Path to the model
            
        Returns:
            Loaded model
        """
        import pickle
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            print(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    
    def extract_features(self, pose_data):
        """
        Extract relevant features from pose data for emotion classification
        Based on kinematic and postural features as described in the paper
        
        Args:
            pose_data: List of pose keypoints across frames
            
        Returns:
            Extracted features
        """
        # We'll extract various features mentioned in the paper
        features = []
        
        # Convert to numpy array for easier processing
        # Shape: [frames, keypoints, 3] (x, y, confidence)
        pose_data = np.array(pose_data)
        
        if len(pose_data) == 0:
            return None
        
        num_frames = pose_data.shape[0]
        
        # 1. Calculate velocity features (changes between frames)
        if num_frames > 1:
            velocities = np.sqrt(np.sum(np.square(
                pose_data[1:, :, :2] - pose_data[:-1, :, :2]
            ), axis=2))
            
            # Average velocity per keypoint
            avg_velocity = np.mean(velocities, axis=0)
            # Max velocity per keypoint
            max_velocity = np.max(velocities, axis=0)
            
            features.extend(avg_velocity)
            features.extend(max_velocity)
            
            # 2. Calculate acceleration features
            if num_frames > 2:
                accelerations = velocities[1:] - velocities[:-1]
                avg_acceleration = np.mean(accelerations, axis=0)
                max_acceleration = np.max(accelerations, axis=0)
                
                features.extend(avg_acceleration)
                features.extend(max_acceleration)
        
        # 3. Postural features for each frame
        # Average posture across all frames
        avg_pose = np.mean(pose_data[:, :, :2], axis=0)
        
        # Extract limb angles
        # For simplicity focusing on main limbs: arms and legs
        for frame_idx in range(num_frames):
            # Get current frame pose
            curr_pose = pose_data[frame_idx, :, :2]
            
            # Extract key points based on BODY_25 model from OpenPose
            # Refer to Figure 1 in the paper for indices
            
            # Calculate angles for arms
            if np.all(curr_pose[[2, 3, 4], :].sum(axis=1) != 0):  # Right arm
                angle_right_arm = self._calculate_angle(
                    curr_pose[2], curr_pose[3], curr_pose[4]
                )
                features.append(angle_right_arm)
            else:
                features.append(0)
                
            if np.all(curr_pose[[5, 6, 7], :].sum(axis=1) != 0):  # Left arm
                angle_left_arm = self._calculate_angle(
                    curr_pose[5], curr_pose[6], curr_pose[7]
                )
                features.append(angle_left_arm)
            else:
                features.append(0)
                
            # Calculate angles for legs
            if np.all(curr_pose[[9, 10, 11], :].sum(axis=1) != 0):  # Right leg
                angle_right_leg = self._calculate_angle(
                    curr_pose[9], curr_pose[10], curr_pose[11]
                )
                features.append(angle_right_leg)
            else:
                features.append(0)
                
            if np.all(curr_pose[[12, 13, 14], :].sum(axis=1) != 0):  # Left leg
                angle_left_leg = self._calculate_angle(
                    curr_pose[12], curr_pose[13], curr_pose[14]
                )
                features.append(angle_left_leg)
            else:
                features.append(0)
        
        # 4. Symmetry features
        # Calculate symmetry between left and right sides
        left_keypoints = [5, 6, 7, 12, 13, 14]  # Left limbs keypoints
        right_keypoints = [2, 3, 4, 9, 10, 11]  # Right limbs keypoints
        
        # Average positions
        left_avg = np.mean(avg_pose[left_keypoints], axis=0)
        right_avg = np.mean(avg_pose[right_keypoints], axis=0)
        
        # Symmetry score: distance between mirrored right side and left side
        # First mirror the right points across the vertical midline
        midline_x = avg_pose[1, 0]  # X coordinate of neck
        mirrored_right_avg = np.copy(right_avg)
        mirrored_right_avg[0] = 2 * midline_x - mirrored_right_avg[0]
        
        symmetry_distance = np.linalg.norm(mirrored_right_avg - left_avg)
        features.append(symmetry_distance)
        
        # 5. Contraction/expansion features
        # Calculate the bounding box of the body
        min_coords = np.min(avg_pose, axis=0)
        max_coords = np.max(avg_pose, axis=0)
        
        # Box dimensions
        width = max_coords[0] - min_coords[0]
        height = max_coords[1] - min_coords[1]
        
        # Surface area
        surface_area = width * height
        features.append(surface_area)
        
        # 6. Vertical movement
        if num_frames > 1:
            # Track vertical movement of the body center
            body_center_y = np.mean(pose_data[:, [0, 1, 8], 1], axis=1)  # Y coords of nose, neck, mid-hip
            vertical_movement = np.std(body_center_y)
            features.append(vertical_movement)
        
        return np.array(features)
    
    def _calculate_angle(self, point1, point2, point3):
        """
        Calculate the angle between three points
        
        Args:
            point1, point2, point3: Three points where point2 is the vertex
            
        Returns:
            Angle in degrees
        """
        vector1 = point1 - point2
        vector2 = point3 - point2
        
        # Normalize vectors
        vector1 = vector1 / np.linalg.norm(vector1)
        vector2 = vector2 / np.linalg.norm(vector2)
        
        # Calculate dot product
        dot_product = np.dot(vector1, vector2)
        
        # Calculate angle in radians and convert to degrees
        angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
        return np.degrees(angle)
    
    def extract_pose_from_video(self, video_path):
        """
        Extract pose data from a video file
        
        Args:
            video_path: Path to the video file
            
        Returns:
            List of pose keypoints across frames
        """
        if self.pose_estimator is None:
            print("Pose estimator not initialized. Cannot extract from video.")
            return None
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            return None
        
        # List to store poses
        poses = []
        
        # Process each frame
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to RGB for OpenPose
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with OpenPose
            datum = op.Datum()
            datum.cvInputData = frame
            self.pose_estimator.emplaceAndPop([datum])
            
            # Check if pose was detected
            if datum.poseKeypoints is not None and len(datum.poseKeypoints) > 0:
                poses.append(datum.poseKeypoints[0])  # Take first person detected
            else:
                # If no pose detected, add zeros
                poses.append(np.zeros((NUM_KEYPOINTS, 3)))
                
        cap.release()
        return poses
    
    def train_model(self, dataset_path, output_model_path=None):
        """
        Train the emotion detection model using the MEED dataset or similar
        
        Args:
            dataset_path: Path to dataset
            output_model_path: Where to save the trained model
            
        Returns:
            Trained model
        """
        # Features and labels lists
        features = []
        labels = []
        
        # Load and process dataset
        print("Loading and processing dataset...")
        
        # Walk through dataset directories
        for emotion in EMOTIONS:
            emotion_dir = os.path.join(dataset_path, emotion)
            if not os.path.exists(emotion_dir):
                print(f"Warning: Path {emotion_dir} not found.")
                continue
                
            for recording in os.listdir(emotion_dir):
                # Assuming each recording is a directory with JSON files
                recording_path = os.path.join(emotion_dir, recording)
                if os.path.isdir(recording_path):
                    # Load pose data from JSON files
                    pose_data = self._load_pose_from_json(recording_path)
                    if pose_data is not None and len(pose_data) > 0:
                        # Extract features
                        extracted_features = self.extract_features(pose_data)
                        if extracted_features is not None:
                            features.append(extracted_features)
                            labels.append(EMOTIONS.index(emotion))
        
        if len(features) == 0:
            print("No valid data found for training.")
            return None
            
        # Convert to numpy arrays
        X = np.array(features)
        y = np.array(labels)
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Split into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        
        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Train a Random Forest classifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate the model
        y_pred = model.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=EMOTIONS))
        
        print("Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        # Save the model if requested
        if output_model_path:
            import pickle
            with open(output_model_path, 'wb') as f:
                pickle.dump((model, scaler), f)
            print(f"Model saved to {output_model_path}")
        
        # Store model and scaler
        self.model = (model, scaler)
        return self.model
    
    def _load_pose_from_json(self, directory_path):
        """
        Load pose data from JSON files in a directory
        
        Args:
            directory_path: Path to directory containing JSON files
            
        Returns:
            List of pose keypoints across frames
        """
        # List to store poses
        poses = []
        
        # Get JSON files sorted by frame number
        json_files = [f for f in os.listdir(directory_path) if f.endswith('.json')]
        json_files.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
        
        for json_file in json_files:
            file_path = os.path.join(directory_path, json_file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # Extract keypoints based on MEED dataset format
                # Format might vary, adapt accordingly
                if 'people' in data and len(data['people']) > 0:
                    keypoints = np.array(data['people'][0]['pose_keypoints_2d'])
                    # Reshape to [NUM_KEYPOINTS, 3] - (x, y, confidence)
                    keypoints = keypoints.reshape(-1, 3)
                    poses.append(keypoints)
            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
        
        return poses
    
    def predict_emotion(self, pose_data):
        """
        Predict emotion from pose data
        
        Args:
            pose_data: List of pose keypoints across frames
            
        Returns:
            Predicted emotion and confidence
        """
        if self.model is None:
            print("Model not loaded. Please train or load a model first.")
            return None, 0
            
        # Extract features
        features = self.extract_features(pose_data)
        if features is None:
            return None, 0
            
        # Scale features
        model, scaler = self.model
        features_scaled = scaler.transform(features.reshape(1, -1))
        
        # Predict
        emotion_idx = model.predict(features_scaled)[0]
        confidence = np.max(model.predict_proba(features_scaled)[0])
        
        return EMOTIONS[emotion_idx], confidence
    
    def process_video(self, video_path, output_path=None):
        """
        Process a video file to detect emotions
        
        Args:
            video_path: Path to the video file
            output_path: Path to save the processed video
            
        Returns:
            Detected emotion, confidence
        """
        # Extract pose from video
        pose_data = self.extract_pose_from_video(video_path)
        if pose_data is None:
            return None, 0
            
        # Predict emotion
        emotion, confidence = self.predict_emotion(pose_data)
        print(f"Detected emotion: {emotion} (confidence: {confidence:.2f})")
        
        # Create visualization video if output path provided
        if output_path and pose_data is not None:
            self._create_visualization(video_path, pose_data, emotion, confidence, output_path)
            
        return emotion, confidence
    
    def _create_visualization(self, video_path, pose_data, emotion, confidence, output_path):
        """
        Create a visualization video with pose and emotion information
        
        Args:
            video_path: Path to the original video
            pose_data: Extracted pose data
            emotion: Detected emotion
            confidence: Prediction confidence
            output_path: Path to save the visualization
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file {video_path}")
            return
            
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Define colors for each emotion
        emotion_colors = {
            'anger': (0, 0, 255),      # Red in BGR
            'disgust': (0, 255, 0),    # Green
            'fear': (255, 0, 255),     # Magenta
            'happiness': (0, 255, 255), # Yellow
            'sadness': (255, 0, 0),    # Blue
            'surprise': (255, 255, 0),  # Cyan
            'neutral': (255, 255, 255)  # White
        }
        
        # Process each frame
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame_idx >= len(pose_data):
                break
                
            # Draw pose keypoints and connections
            self._draw_pose(frame, pose_data[frame_idx], emotion_colors[emotion])
            
            # Add emotion text
            text = f"Emotion: {emotion} ({confidence:.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, emotion_colors[emotion], 2)
            
            # Write frame
            out.write(frame)
            frame_idx += 1
            
        # Release resources
        cap.release()
        out.release()
        print(f"Visualization saved to {output_path}")
    
    def _draw_pose(self, frame, pose, color):
        """
        Draw pose keypoints and connections on a frame
        
        Args:
            frame: Video frame
            pose: Pose keypoints for this frame
            color: Color to use for drawing
        """
        # Define BODY_25 connections
        connections = [
            # Torso
            (1, 0), (1, 8), (8, 9), (8, 12),
            # Right arm
            (1, 2), (2, 3), (3, 4),
            # Left arm
            (1, 5), (5, 6), (6, 7),
            # Right leg
            (9, 10), (10, 11), (11, 22), (22, 23), (11, 24),
            # Left leg
            (12, 13), (13, 14), (14, 19), (19, 20), (14, 21),
            # Face
            (0, 15), (0, 16), (15, 17), (16, 18)
        ]
        
        # Draw keypoints
        for i in range(NUM_KEYPOINTS):
            x, y, conf = pose[i]
            if conf > 0.2:  # Only draw if confidence is high enough
                cv2.circle(frame, (int(x), int(y)), 5, color, -1)
                
        # Draw connections
        for connection in connections:
            idx1, idx2 = connection
            x1, y1, conf1 = pose[idx1]
            x2, y2, conf2 = pose[idx2]
            
            if conf1 > 0.2 and conf2 > 0.2:  # Draw only if both points are reliable
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

# Example usage
if __name__ == "__main__":
    # Initialize emotion detector
    detector = EmotionDetector()
    
    # Train model on MEED dataset
    # Assuming dataset is organized as:
    # - MEED_dataset/
    #   - anger/
    #     - recording1/
    #       - frame_00001.json
    #       - frame_00002.json
    #       - ...
    #     - recording2/
    #       - ...
    #   - disgust/
    #     - ...
    #   - ...
    
    detector.train_model("path/to/MEED_dataset", "emotion_model.pkl")
    
    # Or load a pre-trained model
    # detector = EmotionDetector("emotion_model.pkl")
    
    # Process a video to detect emotion
    emotion, confidence = detector.process_video("path/to/video.mp4", "output_visualization.mp4")
    print(f"Detected emotion: {emotion} with confidence {confidence:.2f}")