import os
import argparse
import numpy as np
import pandas as pd
import pickle
import cv2
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

EMOTIONS = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Neutral', 'Sadness', 'Surprise']

class EmotionDetector:
    def __init__(self, model_path):
        """
        Initialize the emotion detector
        
        Args:
            model_path: Path to the trained model
        """
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.load_model()
    
    def load_model(self):
        """
        Load the trained model
        """
        if not os.path.exists(self.model_path):
            print(f"Error: Model file not found at {self.model_path}")
            return False
        
        try:
            with open(self.model_path, 'rb') as f:
                # Load model, scaler, and label encoder
                loaded_data = pickle.load(f)
                
                if len(loaded_data) == 3:
                    self.model, self.scaler, self.label_encoder = loaded_data
                    print("Loaded model, scaler, and label encoder")
                elif len(loaded_data) == 2:
                    self.model, self.scaler = loaded_data
                    print("Loaded model and scaler")
                else:
                    self.model = loaded_data
                    print("Loaded model only")
                
                return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def detect_emotion(self, movement_value):
        """
        Detect emotion from movement value
        
        Args:
            movement_value: Movement value to classify
            
        Returns:
            Predicted emotion and confidence
        """
        if self.model is None:
            print("Error: Model not loaded")
            return "Unknown", 0.0
        
        try:
            # Scale the input
            if self.scaler is not None:
                movement_scaled = self.scaler.transform([[movement_value]])
            else:
                movement_scaled = [[movement_value]]
            
            # Predict
            prediction = self.model.predict(movement_scaled)[0]
            probabilities = self.model.predict_proba(movement_scaled)[0]
            confidence = max(probabilities)
            
            # Convert numeric prediction to emotion name
            if self.label_encoder is not None:
                emotion = self.label_encoder.inverse_transform([prediction])[0]
            else:
                emotion = EMOTIONS[prediction]
            
            return emotion, confidence
        
        except Exception as e:
            print(f"Error predicting emotion: {e}")
            return "Unknown", 0.0
    
    def process_video(self, video_path, output_path=None):
        """
        Process a video to detect emotion
        
        Args:
            video_path: Path to the video file
            output_path: Path to save the output video
            
        Returns:
            Predicted emotion and confidence
        """
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            return "Unknown", 0.0
        
        try:
            # Open the video
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                return "Unknown", 0.0
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate movement value from video
            movement_value = self._calculate_movement(cap)
            
            # Detect emotion
            emotion, confidence = self.detect_emotion(movement_value)
            print(f"Detected emotion: {emotion} (confidence: {confidence:.2f})")
            
            # Create output video if requested
            if output_path:
                self._create_visualization(video_path, emotion, confidence, output_path)
            
            cap.release()
            return emotion, confidence
        
        except Exception as e:
            print(f"Error processing video: {e}")
            return "Unknown", 0.0
    
    def _calculate_movement(self, cap):
        """
        Calculate movement value from video
        
        Args:
            cap: Video capture object
            
        Returns:
            Calculated movement value
        """
        # Reset to the beginning of the video
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Read the first frame
        ret, prev_frame = cap.read()
        if not ret:
            return 0
        
        # Convert to grayscale
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        movements = []
        
        while True:
            # Read the next frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate absolute difference
            diff = cv2.absdiff(prev_gray, gray)
            
            # Count pixels with significant change
            movement = np.sum(diff > 10)
            movements.append(movement)
            
            # Update previous frame
            prev_gray = gray
        
        # Calculate average movement
        if movements:
            return np.mean(movements)
        else:
            return 0
    
    def _create_visualization(self, video_path, emotion, confidence, output_path):
        """
        Create a visualization of the emotion detection
        
        Args:
            video_path: Path to the input video
            emotion: Detected emotion
            confidence: Confidence of the prediction
            output_path: Path to save the output video
        """
        # Open the video
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Define colors for each emotion
        colors = {
            'Anger': (0, 0, 255),      # Red (BGR)
            'Disgust': (0, 255, 0),    # Green
            'Fear': (255, 0, 255),     # Magenta
            'Happiness': (0, 255, 255), # Yellow
            'Neutral': (255, 255, 255), # White
            'Sadness': (255, 0, 0),    # Blue
            'Surprise': (255, 255, 0)   # Cyan
        }
        
        color = colors.get(emotion, (255, 255, 255))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Add emotion text to the frame
            text = f"Emotion: {emotion} ({confidence:.2f})"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                      1, color, 2)
            
            # Write the frame
            out.write(frame)
        
        # Release resources
        cap.release()
        out.release()
        print(f"Visualization saved to {output_path}")

def extract_data(data_dir, output_dir):
    """
    Extract data from the MEED dataset
    
    Args:
        data_dir: Path to the data directory
        output_dir: Path to the output directory
    """
    print("Extracting data...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if frontMovement.csv exists
    csv_path = os.path.join(data_dir, 'frontMovement.csv')
    if os.path.exists(csv_path):
        print(f"Found frontMovement.csv, copying to output directory...")
        import shutil
        output_csv = os.path.join(output_dir, 'frontMovement.csv')
        shutil.copy2(csv_path, output_csv)
    else:
        print(f"Error: frontMovement.csv not found in {data_dir}")
    
    print("Data extraction complete")

def train_model(data_dir, output_dir):
    """
    Train the emotion detection model
    
    Args:
        data_dir: Path to the data directory
        output_dir: Path to the output directory
    """
    print("Training model...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if frontMovement.csv exists
    csv_path = os.path.join(data_dir, 'frontMovement.csv')
    if not os.path.exists(csv_path):
        print(f"Error: frontMovement.csv not found in {data_dir}")
        return
    
    try:
        # Load the CSV
        df = pd.read_csv(csv_path)
        print(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Find feature columns (numeric columns that aren't labels)
        feature_cols = [col for col in df.columns if df[col].dtype != 'object' and col != 'emotion']
        
        if 'Objective_movement' in df.columns:
            # Use only Objective_movement as feature
            feature_cols = ['Objective_movement']
        
        print(f"Using feature columns: {feature_cols}")
        
        # Extract features
        features = df[feature_cols].values
        
        # Extract labels
        if 'emotion' in df.columns:
            labels = df['emotion'].values
        else:
            # Try to derive emotions from filenames
            filenames = df.iloc[:, 0].values  # Assume first column is filename
            labels = []
            
            for name in filenames:
                if 'A' in str(name):
                    labels.append('Anger')
                elif 'D' in str(name):
                    labels.append('Disgust')
                elif 'F' in str(name):
                    labels.append('Fear')
                elif 'H' in str(name):
                    labels.append('Happiness')
                elif 'SA' in str(name):
                    labels.append('Sadness')
                elif 'SU' in str(name):
                    labels.append('Surprise')
                elif 'N' in str(name):
                    labels.append('Neutral')
                else:
                    labels.append('Unknown')
            
            labels = np.array(labels)
        
        print(f"Unique labels: {np.unique(labels)}")
        
        # Scale features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Train-test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            features_scaled, labels, test_size=0.2, random_state=42
        )
        
        # Train model
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import LabelEncoder
        
        # Encode labels
        label_encoder = LabelEncoder()
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)
        
        # Train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train_encoded)
        
        # Evaluate
        train_acc = model.score(X_train, y_train_encoded)
        test_acc = model.score(X_test, y_test_encoded)
        
        print(f"Training accuracy: {train_acc:.4f}")
        print(f"Testing accuracy: {test_acc:.4f}")
        
        # Save the model
        model_path = os.path.join(output_dir, 'emotion_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump((model, scaler, label_encoder), f)
        
        print(f"Model saved to {model_path}")
        
        # Create evaluation plots
        create_evaluation_plots(model, X_test, y_test_encoded, label_encoder, output_dir)
    
    except Exception as e:
        print(f"Error training model: {e}")
        import traceback
        traceback.print_exc()

def create_evaluation_plots(model, X_test, y_test, label_encoder, output_dir):
    """
    Create evaluation plots
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        label_encoder: Label encoder
        output_dir: Output directory
    """
    try:
        # Create confusion matrix
        from sklearn.metrics import confusion_matrix
        import seaborn as sns
        
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=label_encoder.classes_,
                   yticklabels=label_encoder.classes_)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        
        # Save the plot
        cm_path = os.path.join(output_dir, 'confusion_matrix.png')
        plt.savefig(cm_path)
        plt.close()
        
        print(f"Confusion matrix saved to {cm_path}")
        
        # Create feature importance plot
        importances = model.feature_importances_
        
        plt.figure(figsize=(10, 6))
        plt.bar(range(len(importances)), importances)
        plt.xlabel('Feature Index')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        
        # Save the plot
        fi_path = os.path.join(output_dir, 'feature_importance.png')
        plt.savefig(fi_path)
        plt.close()
        
        print(f"Feature importance plot saved to {fi_path}")
    
    except Exception as e:
        print(f"Error creating evaluation plots: {e}")

def test_model(data_dir, output_dir, model_path=None, video_path=None):
    """
    Test the emotion detection model
    
    Args:
        data_dir: Path to the data directory
        output_dir: Path to the output directory
        model_path: Path to the model
        video_path: Path to a video file to test
    """
    print("Testing model...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine model path if not provided
    if model_path is None:
        model_path = os.path.join(output_dir, 'emotion_model.pkl')
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return
    
    # Initialize the emotion detector
    detector = EmotionDetector(model_path)
    
    if video_path:
        # Test with provided video
        output_video = os.path.join(output_dir, 'output_visualization.mp4')
        emotion, confidence = detector.process_video(video_path, output_video)
        print(f"Video result: {emotion} (confidence: {confidence:.2f})")
    else:
        # Test with sample videos from the dataset
        # Check for sample recordings
        print("Looking for sample videos to test...")
        samples_dir = os.path.join(data_dir, 'sample')
        
        if os.path.exists(samples_dir) and os.path.isdir(samples_dir):
            # Process sample videos
            sample_files = [f for f in os.listdir(samples_dir) if f.endswith('.mp4')]
            
            if sample_files:
                print(f"Found {len(sample_files)} sample videos")
                
                results = []
                for sample in sample_files[:5]:  # Process up to 5 samples
                    sample_path = os.path.join(samples_dir, sample)
                    print(f"Processing: {sample}")
                    
                    # Try to determine true emotion from filename
                    true_emotion = "Unknown"
                    for emo in EMOTIONS:
                        if emo.lower() in sample.lower():
                            true_emotion = emo
                            break
                    
                    # Process the video
                    output_video = os.path.join(output_dir, f"output_{sample}")
                    emotion, confidence = detector.process_video(sample_path, output_video)
                    
                    results.append({
                        'sample': sample,
                        'true_emotion': true_emotion,
                        'predicted': emotion,
                        'confidence': confidence
                    })
                    
                    print(f"  True: {true_emotion}, Predicted: {emotion}, Confidence: {confidence:.2f}")
                
                # Save results to CSV
                if results:
                    results_df = pd.DataFrame(results)
                    results_path = os.path.join(output_dir, 'test_results.csv')
                    results_df.to_csv(results_path, index=False)
                    print(f"Test results saved to {results_path}")
            else:
                print("No sample videos found")
        else:
            print(f"Sample directory not found: {samples_dir}")
            
            # Test with movement values
            print("Testing with synthetic movement values...")
            
            # Create a range of movement values
            movement_values = np.linspace(5000, 25000, 10)
            
            results = []
            for value in movement_values:
                emotion, confidence = detector.detect_emotion(value)
                results.append({
                    'movement': value,
                    'emotion': emotion,
                    'confidence': confidence
                })
                print(f"Movement: {value:.2f}, Emotion: {emotion}, Confidence: {confidence:.2f}")
            
            # Save results to CSV
            results_df = pd.DataFrame(results)
            results_path = os.path.join(output_dir, 'synthetic_test_results.csv')
            results_df.to_csv(results_path, index=False)
            print(f"Synthetic test results saved to {results_path}")
            
            # Create a plot of movement values vs emotions
            plt.figure(figsize=(12, 6))
            
            # Get unique emotions
            unique_emotions = results_df['emotion'].unique()
            
            # Create a color map
            colors = plt.cm.get_cmap('tab10', len(unique_emotions))
            
            # Plot each emotion
            for i, emotion in enumerate(unique_emotions):
                subset = results_df[results_df['emotion'] == emotion]
                plt.scatter(subset['movement'], subset['confidence'], 
                           label=emotion, color=colors(i), s=100)
            
            plt.xlabel('Movement Value')
            plt.ylabel('Confidence')
            plt.title('Emotion Prediction by Movement Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            plot_path = os.path.join(output_dir, 'emotion_by_movement.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Movement plot saved to {plot_path}")

def main():
    parser = argparse.ArgumentParser(description='Emotion Detection Tool')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for results')
    parser.add_argument('--mode', type=str, default='full', choices=['extract', 'train', 'test', 'full'],
                      help='Operation mode')
    parser.add_argument('--video_path', type=str, help='Path to video for testing (optional)')
    parser.add_argument('--model_path', type=str, help='Path to model file (optional)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the requested operation
    if args.mode == 'extract' or args.mode == 'full':
        extract_data(args.data_dir, args.output_dir)
    
    if args.mode == 'train' or args.mode == 'full':
        train_model(args.data_dir, args.output_dir)
    
    if args.mode == 'test' or args.mode == 'full':
        test_model(args.data_dir, args.output_dir, args.model_path, args.video_path)
    
    print("Process completed!")

if __name__ == "__main__":
    main()