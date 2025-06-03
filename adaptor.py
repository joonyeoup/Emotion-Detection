import os
import zipfile
import json
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm

class SimplifiedMEEDAdapter:
    """
    Adapter class to process the MEED dataset from ZIP files
    """
    
    def __init__(self, data_dir, output_dir):
        """
        Initialize the adapter
        
        Args:
            data_dir: Path to directory containing the ZIP files and MAT files
            output_dir: Path to store the processed dataset
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.emotions = ['anger', 'disgust', 'fear', 'happiness', 'sadness', 'surprise', 'neutral']
        
        # Emotion code mapping
        self.emotion_mapping = {
            'A': 'anger',
            'D': 'disgust',
            'F': 'fear',
            'H': 'happiness',
            'N': 'neutral',
            'SA': 'sadness',
            'SU': 'surprise'
        }
    
    def extract_zip_files(self):
        """
        Extract all ZIP files in the data directory
        """
        print("Extracting ZIP files...")
        
        for filename in os.listdir(self.data_dir):
            if filename.endswith('.zip'):
                file_path = os.path.join(self.data_dir, filename)
                print(f"Extracting {filename}...")
                
                # Create a folder with the same name as the zip file (without extension)
                extract_dir = os.path.join(self.data_dir, os.path.splitext(filename)[0])
                os.makedirs(extract_dir, exist_ok=True)
                
                # Extract the contents
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_dir)
                
                print(f"Extracted to {extract_dir}")
    
    def load_coordinate_data(self):
        """
        Load coordinate data from MAT files or CSV files
        """
        print("Loading coordinate data...")
        
        # Check if MATLAB file exists
        coordinate_files = [f for f in os.listdir(self.data_dir) 
                         if f.endswith('_coordinate.mat') or 
                            f.endswith('_coordinate_emotion.mat') or
                            f.endswith('_coordinate_subjects.mat')]
        
        if coordinate_files:
            try:
                import scipy.io as sio
                
                # Load first found MAT file
                mat_file = os.path.join(self.data_dir, coordinate_files[0])
                print(f"Loading data from {mat_file}")
                mat_data = sio.loadmat(mat_file)
                
                # Print available keys
                print(f"Available data keys: {list(mat_data.keys())}")
                
                return mat_data
            except Exception as e:
                print(f"Error loading MAT file: {e}")
        
        # Check if CSV exists
        csv_files = [f for f in os.listdir(self.data_dir) if f.endswith('.csv')]
        if csv_files and 'frontMovement.csv' in csv_files:
            try:
                csv_file = os.path.join(self.data_dir, 'frontMovement.csv')
                print(f"Loading data from {csv_file}")
                csv_data = pd.read_csv(csv_file)
                
                # Print column names
                print(f"CSV columns: {csv_data.columns.tolist()}")
                print(f"First few rows: \n{csv_data.head()}")
                
                return csv_data
            except Exception as e:
                print(f"Error loading CSV file: {e}")
        
        return None
    
    def organize_dataset(self):
        """
        Organize the extracted data into a structure suitable for the emotion detector
        """
        print("Organizing dataset...")
        
        # Create output directories
        for emotion in self.emotions:
            os.makedirs(os.path.join(self.output_dir, emotion), exist_ok=True)
        
        # Load metadata if available
        metadata = self.load_coordinate_data()
        
        # Process extracted directories (front_F01, front_M02, etc.)
        extracted_dirs = [d for d in os.listdir(self.data_dir) 
                       if os.path.isdir(os.path.join(self.data_dir, d)) and 
                          (d.startswith('front_') or d.startswith('left_') or d.startswith('right_'))]
        
        for dir_name in tqdm(extracted_dirs, desc="Processing directories"):
            dir_path = os.path.join(self.data_dir, dir_name)
            
            # Get view and actor ID from directory name
            parts = dir_name.split('_')
            if len(parts) >= 2:
                view = parts[0]  # front, left, or right
                actor_id = parts[1]  # F01, M02, etc.
                
                # Process all subdirectories in this actor's folder
                for recording_name in os.listdir(dir_path):
                    recording_path = os.path.join(dir_path, recording_name)
                    
                    if os.path.isdir(recording_path):
                        # Try to determine emotion from the recording name
                        emotion_code = None
                        for code in self.emotion_mapping.keys():
                            if code in recording_name:
                                emotion_code = code
                                break
                        
                        if emotion_code is None:
                            # If we can't determine from name, check if we have metadata
                            if metadata is not None:
                                # Try to find this recording in metadata
                                # This depends on the specific format of your metadata
                                emotion_code = 'N'  # Default to neutral if unknown
                            else:
                                print(f"Warning: Could not determine emotion for {recording_name}, skipping")
                                continue
                        
                        emotion = self.emotion_mapping[emotion_code]
                        
                        # Create destination directory
                        dest_dir = os.path.join(self.output_dir, emotion, f"{view}_{actor_id}_{recording_name}")
                        os.makedirs(dest_dir, exist_ok=True)
                        
                        # Process all files in this recording
                        for file_name in os.listdir(recording_path):
                            src_file = os.path.join(recording_path, file_name)
                            
                            # If it's a JSON file with keypoints, copy it
                            if file_name.endswith('.json'):
                                dest_file = os.path.join(dest_dir, file_name)
                                shutil.copy2(src_file, dest_file)
                                
                                # Optionally convert JSON format if needed
                                self.convert_json_format(dest_file)
        
        print("Dataset organization complete.")
    
    def convert_json_format(self, json_file):
        """
        Convert JSON format if it doesn't match what's expected by the emotion detector
        
        Args:
            json_file: Path to the JSON file to convert
        """
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check if the format already matches what we expect
            if 'people' in data and isinstance(data['people'], list):
                # Already in correct format
                return
            
            # If it's in a different format, convert it
            if isinstance(data, dict) and all(k.isdigit() for k in data.keys()):
                # Assume format is {"0": {"x": val, "y": val, "confidence": val}, ...}
                new_data = {
                    "version": 1.3,
                    "people": [
                        {
                            "person_id": [-1],
                            "pose_keypoints_2d": []
                        }
                    ]
                }
                
                keypoints = []
                
                for i in range(25):  # 25 keypoints as in the paper
                    if str(i) in data:
                        point = data[str(i)]
                        keypoints.extend([point.get("x", 0), point.get("y", 0), point.get("confidence", 0)])
                    else:
                        keypoints.extend([0, 0, 0])
                
                new_data["people"][0]["pose_keypoints_2d"] = keypoints
                
                # Write converted JSON
                with open(json_file, 'w') as f:
                    json.dump(new_data, f)
                    
        except Exception as e:
            print(f"Error converting file {json_file}: {e}")
    
    def create_validation_split(self, test_ratio=0.2):
        """
        Create train/test split file
        
        Args:
            test_ratio: Ratio of test set
        """
        print("Creating validation split...")
        
        recordings = []
        
        # Collect all recordings
        for emotion in self.emotions:
            emotion_dir = os.path.join(self.output_dir, emotion)
            
            if not os.path.exists(emotion_dir):
                continue
                
            recording_dirs = [d for d in os.listdir(emotion_dir) if os.path.isdir(os.path.join(emotion_dir, d))]
            
            for recording in recording_dirs:
                recordings.append({
                    "path": os.path.join(emotion, recording),
                    "emotion": emotion
                })
        
        # Shuffle recordings
        import random
        random.shuffle(recordings)
        
        # Split into train and test
        split_idx = int(len(recordings) * (1 - test_ratio))
        train_set = recordings[:split_idx]
        test_set = recordings[split_idx:]
        
        # Create DataFrame
        train_df = pd.DataFrame(train_set)
        test_df = pd.DataFrame(test_set)
        
        # Save to CSV
        train_df.to_csv(os.path.join(self.output_dir, "train_split.csv"), index=False)
        test_df.to_csv(os.path.join(self.output_dir, "test_split.csv"), index=False)
        
        print(f"Created validation split: {len(train_set)} training and {len(test_set)} testing recordings")
    
    def extract_sample_videos(self, num_samples=2):
        """
        Extract a few sample recordings for testing
        
        Args:
            num_samples: Number of samples to extract per emotion
        """
        print("Extracting sample recordings...")
        
        samples_dir = os.path.join(self.output_dir, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        
        for emotion in self.emotions:
            emotion_dir = os.path.join(self.output_dir, emotion)
            
            if not os.path.exists(emotion_dir):
                continue
                
            # Get recordings
            recordings = [d for d in os.listdir(emotion_dir) if os.path.isdir(os.path.join(emotion_dir, d))]
            
            if not recordings:
                continue
                
            # Select random samples
            import random
            selected = random.sample(recordings, min(num_samples, len(recordings)))
            
            for recording in selected:
                src_dir = os.path.join(emotion_dir, recording)
                dest_dir = os.path.join(samples_dir, f"{emotion}_{recording}")
                
                # Copy directory
                shutil.copytree(src_dir, dest_dir)
                
        print(f"Extracted {num_samples} sample recordings per emotion to {samples_dir}")
    
    def process(self):
        """
        Run the complete dataset processing pipeline
        """
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Extract ZIP files
        self.extract_zip_files()
        
        # Organize dataset
        self.organize_dataset()
        
        # Create validation split
        self.create_validation_split()
        
        # Extract sample recordings
        self.extract_sample_videos()
        
        print("Dataset processing complete!")


# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Process MEED dataset from ZIP files')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing ZIP files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for processed dataset')
    
    args = parser.parse_args()
    
    adapter = SimplifiedMEEDAdapter(args.data_dir, args.output_dir)
    adapter.process()