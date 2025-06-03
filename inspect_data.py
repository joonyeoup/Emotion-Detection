import os
import sys
import scipy.io as sio
import numpy as np
import json

def inspect_directory(dir_path):
    """
    Inspect a directory and print information about its contents
    """
    print(f"\n=== Inspecting directory: {dir_path} ===")
    
    if not os.path.exists(dir_path):
        print(f"Error: Directory does not exist")
        return
    
    # List files
    files = os.listdir(dir_path)
    print(f"Total files: {len(files)}")
    
    # Group files by extension
    file_types = {}
    for file in files:
        _, ext = os.path.splitext(file)
        if ext not in file_types:
            file_types[ext] = []
        file_types[ext].append(file)
    
    # Print file types summary
    print("\nFile types found:")
    for ext, files_list in file_types.items():
        print(f"  {ext}: {len(files_list)} files")
    
    # Inspect MAT files
    mat_files = file_types.get('.mat', [])
    if mat_files:
        print("\nInspecting MAT files:")
        for mat_file in mat_files[:3]:  # Look at first 3 only
            file_path = os.path.join(dir_path, mat_file)
            try:
                data = sio.loadmat(file_path)
                print(f"  {mat_file} keys: {list(data.keys())}")
                # Print a sample of each array
                for key in data.keys():
                    if key.startswith('__'):  # Skip metadata
                        continue
                    if isinstance(data[key], np.ndarray):
                        print(f"    {key} shape: {data[key].shape}")
                        if data[key].size > 0:
                            print(f"    {key} sample: {data[key].flatten()[:3]}")
            except Exception as e:
                print(f"  Error inspecting {mat_file}: {e}")
    
    # Inspect JSON files
    json_files = file_types.get('.json', [])
    if json_files:
        print("\nInspecting JSON files:")
        for json_file in json_files[:3]:  # Look at first 3 only
            file_path = os.path.join(dir_path, json_file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                print(f"  {json_file} structure:")
                if isinstance(data, dict):
                    print(f"    Top-level keys: {list(data.keys())[:10]}")
                    # If it's a keypoint JSON, try to determine format
                    if '0' in data and isinstance(data['0'], dict):
                        print(f"    Keypoint format example (key '0'): {data['0']}")
                    elif 'people' in data and isinstance(data['people'], list):
                        print(f"    OpenPose format with {len(data['people'])} people")
                        if data['people'] and 'pose_keypoints_2d' in data['people'][0]:
                            keypoints = data['people'][0]['pose_keypoints_2d']
                            print(f"    First few keypoints: {keypoints[:6]}")
                elif isinstance(data, list):
                    print(f"    List with {len(data)} items")
                    if data:
                        print(f"    First item: {data[0]}")
            except Exception as e:
                print(f"  Error inspecting {json_file}: {e}")
    
    # Inspect folders
    folders = [f for f in files if os.path.isdir(os.path.join(dir_path, f))]
    if folders:
        print("\nSubfolders found:")
        for folder in folders:
            folder_path = os.path.join(dir_path, folder)
            subfolder_files = os.listdir(folder_path)
            print(f"  {folder}: {len(subfolder_files)} files")
            # Check first few files in subfolder
            if subfolder_files:
                print(f"    Examples: {subfolder_files[:3]}")
    
    # Examine coordinate MAT files specifically
    coord_mat_files = [f for f in files if 'coordinate' in f and f.endswith('.mat')]
    if coord_mat_files:
        print("\nExamining coordinate MAT files:")
        for mat_file in coord_mat_files:
            file_path = os.path.join(dir_path, mat_file)
            try:
                data = sio.loadmat(file_path)
                print(f"  {mat_file} keys: {list(data.keys())}")
                
                # Find the actual data key (not metadata)
                data_keys = [k for k in data.keys() if not k.startswith('__')]
                
                if data_keys:
                    for key in data_keys:
                        if isinstance(data[key], np.ndarray):
                            print(f"    {key} shape: {data[key].shape}")
                            # Try to determine structure
                            if len(data[key].shape) > 1:
                                print(f"    Dimensions: {data[key].shape}")
                                print(f"    First few elements: {data[key].flatten()[:10]}")
                                
                                # Check if coordinates are stored in a specific format
                                if data[key].shape[1] in [2, 3]:  # Likely (x,y) or (x,y,confidence)
                                    print(f"    Likely coordinate format with {data[key].shape[1]} values per point")
                                elif data[key].shape[1] % 3 == 0:  # Might be flattened keypoints
                                    print(f"    Possibly flattened keypoints ({data[key].shape[1]/3} points with x,y,confidence)")
            except Exception as e:
                print(f"  Error examining {mat_file}: {e}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_data.py <directory_path>")
        return
    
    dir_path = sys.argv[1]
    inspect_directory(dir_path)
    
    # If there are coordinate MAT files, inspect one of them in detail
    coord_path = None
    if os.path.exists(os.path.join(dir_path, 'front_coordinate.mat')):
        coord_path = os.path.join(dir_path, 'front_coordinate.mat')
    elif os.path.exists(os.path.join(dir_path, 'left_coordinate.mat')):
        coord_path = os.path.join(dir_path, 'left_coordinate.mat')
    elif os.path.exists(os.path.join(dir_path, 'right_coordinate.mat')):
        coord_path = os.path.join(dir_path, 'right_coordinate.mat')
    
    if coord_path:
        print(f"\n=== Detailed inspection of coordinate file: {coord_path} ===")
        try:
            data = sio.loadmat(coord_path)
            print(f"Keys: {list(data.keys())}")
            
            # Find actual data key (not metadata)
            data_keys = [k for k in data.keys() if not k.startswith('__')]
            
            if data_keys:
                for key in data_keys:
                    print(f"\nAnalyzing key: {key}")
                    if isinstance(data[key], np.ndarray):
                        print(f"Shape: {data[key].shape}")
                        
                        # Try to determine if it's a multi-dimensional array of coordinates
                        if len(data[key].shape) == 3:
                            print(f"3D array: likely [recordings, keypoints, coordinates]")
                            print(f"This suggests {data[key].shape[0]} recordings, each with {data[key].shape[1]} keypoints")
                            
                            # Print sample from first recording, first few keypoints
                            print(f"First recording, first 3 keypoints:")
                            for i in range(min(3, data[key].shape[1])):
                                print(f"  Keypoint {i}: {data[key][0, i, :]}")
                        
                        elif len(data[key].shape) == 4:
                            print(f"4D array: likely [recordings, frames, keypoints, coordinates]")
                            print(f"This suggests {data[key].shape[0]} recordings, each with {data[key].shape[1]} frames and {data[key].shape[2]} keypoints")
                            
                            # Print sample from first recording, first frame, first few keypoints
                            print(f"First recording, first frame, first 3 keypoints:")
                            for i in range(min(3, data[key].shape[2])):
                                print(f"  Keypoint {i}: {data[key][0, 0, i, :]}")
        except Exception as e:
            print(f"Error analyzing coordinate file: {e}")

if __name__ == "__main__":
    main()