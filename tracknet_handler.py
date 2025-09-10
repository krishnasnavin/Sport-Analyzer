# tracknet_handler.py (Final Corrected Version)
import torch
import cv2
import numpy as np
import sys
import os

# Add TrackNet's source code to the Python path
sys.path.append(os.path.abspath('TrackNet'))
from model import BallTrackerNet

# --- Constants ---
TRACKNET_INPUT_WIDTH = 640
TRACKNET_INPUT_HEIGHT = 360

def load_tracknet_model(model_path, device='cpu'):
    """Loads the pre-trained BallTrackerNet model."""
    # --- CHANGE 1: Match the output channels to the pre-trained file (256) ---
    model = BallTrackerNet(out_channels=256)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"BallTrackerNet model not found at {model_path}")

    # Added weights_only=True to address the FutureWarning
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("BallTrackerNet model loaded successfully.")
    return model

def preprocess_tracknet_input(frames):
    """Prepares a sequence of 3 frames for BallTrackerNet."""
    processed_frames = []
    for frame in frames:
        img = cv2.resize(frame, (TRACKNET_INPUT_WIDTH, TRACKNET_INPUT_HEIGHT))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        processed_frames.append(img_tensor)
    
    stacked_tensor = torch.cat(processed_frames, dim=0)
    return stacked_tensor.unsqueeze(0)

def detect_ball(model, frames, device='cpu'):
    """
    Detects the ball in a sequence of 3 frames.
    Returns the (x, y) coordinates of the ball or None.
    """
    if len(frames) != 3:
        return None, None 

    h, w, _ = frames[1].shape 
    
    inp = preprocess_tracknet_input(frames).to(device)
    
    with torch.no_grad():
        out = model(inp)
    
    # --- CHANGE 2: Handle the 256-channel output from the model ---
    # The model outputs a tensor of shape (Batch, 256, H*W).
    # We reshape it back to a spatial format (Batch, 256, H, W).
    # Then, we find the max value across the 256 channels to get a single 2D heatmap.
    heatmap = out.reshape(-1, 256, TRACKNET_INPUT_HEIGHT, TRACKNET_INPUT_WIDTH)
    heatmap = torch.max(heatmap, dim=1)[0] # Get the max activation across channels
    heatmap = heatmap.cpu().squeeze(0).numpy()
    
    y_pred, x_pred = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    
    confidence = np.max(heatmap)
    if confidence < 0.25: # This threshold may need tuning
        return None, None

    ball_x = int(x_pred * (w / TRACKNET_INPUT_WIDTH))
    ball_y = int(y_pred * (h / TRACKNET_INPUT_HEIGHT))

    return ball_x, ball_y