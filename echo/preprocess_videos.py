"""
Preprocess EchoNet-Dynamic videos and save to disk for faster training.
This script reads all videos, preprocesses them, and saves as .pt files.
"""

import numpy as np
import pandas as pd
import torch
import os
from tqdm import tqdm
import warnings
import gc
warnings.filterwarnings('ignore')

# Try to import video reading libraries
try:
    from torchvision.io import read_video
    USE_TORCHVISION = True
except:
    USE_TORCHVISION = False

try:
    import imageio
    USE_IMAGEIO = True
except ImportError:
    USE_IMAGEIO = False
    print("Warning: imageio not installed. Install with: pip install imageio imageio-ffmpeg")

from torchvision.transforms.functional import resize

def preprocess_video(video_path, num_frames=32, target_size=(112, 112)):
    """
    Preprocess video: sample frames, resize, normalize.
    
    Args:
        video_path: Path to video file
        num_frames: Number of frames to sample (default: 32)
        target_size: Target frame size (H, W) (default: (112, 112))
    
    Returns:
        tensor: Shape (C=3, T=32, H=112, W=112)
    """
    try:
        # Try to read video using torchvision first, fallback to imageio
        if USE_TORCHVISION:
            try:
                video, audio, info = read_video(video_path, output_format="TCHW")
                video_np = video.numpy()
            except Exception as e:
                if USE_IMAGEIO:
                    reader = imageio.get_reader(video_path)
                    frames = []
                    try:
                        for frame in reader:
                            frame_chw = np.transpose(frame, (2, 0, 1))
                            frames.append(frame_chw)
                            del frame, frame_chw
                    finally:
                        reader.close()
                    video_np = np.stack(frames, axis=0)
                    del frames
                else:
                    raise e
        elif USE_IMAGEIO:
            reader = imageio.get_reader(video_path)
            frames = []
            try:
                for frame in reader:
                    frame_chw = np.transpose(frame, (2, 0, 1))
                    frames.append(frame_chw)
                    del frame, frame_chw
            finally:
                reader.close()
            video_np = np.stack(frames, axis=0)
            del frames
        else:
            raise ImportError("Neither torchvision nor imageio available")
        
        T, C, H, W = video_np.shape
        
        # Sample exactly num_frames uniformly
        if T >= num_frames:
            indices = np.linspace(0, T - 1, num_frames, dtype=int)
        else:
            indices = list(range(T)) + [T - 1] * (num_frames - T)
            indices = indices[:num_frames]
        
        sampled_frames = video_np[indices]
        
        # Resize frames - process one at a time to save memory
        resized_frames = []
        for frame in sampled_frames:
            frame_tensor = torch.from_numpy(frame).float()
            frame_resized = resize(frame_tensor, target_size, antialias=True)
            resized_frames.append(frame_resized.numpy())
            # Clean up intermediate tensors
            del frame_tensor, frame_resized
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        # Stack frames: (num_frames, C, H, W) -> (C, num_frames, H, W)
        video_tensor = np.stack(resized_frames, axis=1)
        del resized_frames, sampled_frames, video_np  # Free memory
        
        # Normalize to [0, 1]
        video_tensor = video_tensor.astype(np.float32) / 255.0
        
        result = torch.from_numpy(video_tensor)
        del video_tensor  # Free numpy array
        return result
    
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return None

def main():
    BASE_DIR = "/Volumes/Crucial X6/medical_ai_extra/EchoNet-Dynamic"
    VIDEOS_DIR = os.path.join(BASE_DIR, "Videos")
    LABELS_CSV = os.path.join(BASE_DIR, "FileList.csv")
    PREPROCESSED_DIR = os.path.join(BASE_DIR, "PreprocessedVideos")
    
    # Create preprocessed directory
    os.makedirs(PREPROCESSED_DIR, exist_ok=True)
    
    # Load CSV
    print("Loading FileList.csv...")
    df = pd.read_csv(LABELS_CSV)
    print(f"Total videos: {len(df)}")
    
    # Process each video with memory management
    failed = []
    successful = 0
    batch_size = 50  # Process in batches and cleanup memory
    
    for batch_start in range(0, len(df), batch_size):
        batch_end = min(batch_start + batch_size, len(df))
        batch_df = df.iloc[batch_start:batch_end]
        
        for idx, row in batch_df.iterrows():
            filename = row['FileName']
            if not filename.endswith('.avi'):
                filename = filename + '.avi'
            
            video_path = os.path.join(VIDEOS_DIR, filename)
            output_path = os.path.join(PREPROCESSED_DIR, f"{row['FileName']}.pt")
            
            # Skip if already processed
            if os.path.exists(output_path):
                successful += 1
                continue
            
            try:
                # Preprocess video
                video_tensor = preprocess_video(video_path)
                
                if video_tensor is not None:
                    # Save as .pt file
                    torch.save(video_tensor, output_path)
                    successful += 1
                    del video_tensor  # Free memory immediately
                else:
                    failed.append(row['FileName'])
            except Exception as e:
                print(f"\nError processing {row['FileName']}: {e}")
                failed.append(row['FileName'])
        
        # Force garbage collection after each batch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Progress update
        print(f"Processed {batch_end}/{len(df)} videos (Success: {successful}, Failed: {len(failed)})")
    
    print(f"\n✅ Preprocessing complete!")
    print(f"  Successful: {successful}/{len(df)}")
    print(f"  Failed: {len(failed)}")
    print(f"  Preprocessed videos saved to: {PREPROCESSED_DIR}")
    
    if failed:
        print(f"\n⚠️  Failed videos ({len(failed)}): {failed[:10]}...")  # Show first 10

if __name__ == "__main__":
    main()

