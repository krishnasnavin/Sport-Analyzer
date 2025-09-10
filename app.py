# app.py

# --- Step 1: All Necessary Imports ---
import gradio as gr
import pandas as pd
import numpy as np
import cv2
import time
import os
import torch
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile
from scipy.spatial.distance import euclidean
from scipy.signal import find_peaks

from torchvision.models.detection import retinanet_resnet50_fpn
from torchvision.ops import nms
from deep_sort_realtime.deepsort_tracker import DeepSort

from collections import deque
from tracknet_handler import load_tracknet_model, detect_ball

# --- Step 2: All Helper Functions and Constants ---

# -- Constants --
KEYPOINT_CONF_THRESH = 0.3
MOVENET_INPUT_SIZE = 192
KEYPOINT_EDGES = [
    (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13),
    (13, 15), (12, 14), (14, 16)
]
COURT_BOUNDARY = np.array([[280, 68], [546, 72], [720, 446], [89, 464]], dtype=np.int32)
CONF_THRESH = 0.5
NMS_THRESH = 0.4
DEEPSORT_MAX_AGE = 30
DEEPSORT_N_INIT = 3
FRAME_SKIP = 3

# --- Constants for Perspective Transform ---
SINGLES_COURT_CORNERS_PX = np.float32([
    [280, 68],    # Top-Left
    [546, 72],    # Top-Right
    [720, 446],   # Bottom-Right
    [89, 464]     # Bottom-Left
])

SINGLES_COURT_WIDTH_M = 8.23
SINGLES_COURT_LENGTH_M = 23.77

TOP_DOWN_COURT_CORNERS_M = np.float32([
    [0, 0],
    [SINGLES_COURT_WIDTH_M, 0],
    [SINGLES_COURT_WIDTH_M, SINGLES_COURT_LENGTH_M],
    [0, SINGLES_COURT_LENGTH_M]
])

PERSPECTIVE_MATRIX = cv2.getPerspectiveTransform(SINGLES_COURT_CORNERS_PX, TOP_DOWN_COURT_CORNERS_M)

# -- Model Loading & Video Processing Functions --
def load_retinanet(weights_path, device='cpu'):
    model = retinanet_resnet50_fpn(weights=None, num_classes=2)
    if os.path.exists(weights_path):
        state_dict = torch.load(weights_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
    else:
        raise FileNotFoundError(f"RetinaNet weights not found at {weights_path}")
    model.to(device); model.eval(); return model

def load_movenet_model():
    model_url = "https://tfhub.dev/google/movenet/singlepose/lightning/4"
    model = hub.load(model_url)
    return model.signatures['serving_default']

def preprocess_frame(frame, target_long=960):
    h, w = frame.shape[:2]
    scale = target_long / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(frame, (new_w, new_h)), scale

def postprocess_boxes(boxes, scale):
    return boxes / scale

def run_movenet(model, image):
    image = tf.convert_to_tensor(image, dtype=tf.int32)
    image = tf.expand_dims(image, axis=0)
    input_image = tf.image.resize_with_pad(image, MOVENET_INPUT_SIZE, MOVENET_INPUT_SIZE)
    input_image = tf.cast(input_image, dtype=tf.int32)
    results = model(input_image)
    return results['output_0'].numpy()[0, 0, :, :]

# -- Analysis Functions --
def plot_player_heatmap(df, player_id, background_frame):
    player_df = df[df['track_id'] == player_id].copy()
    background_frame_rgb = cv2.cvtColor(background_frame, cv2.COLOR_BGR2RGB)
    h, w, _ = background_frame_rgb.shape
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(background_frame_rgb, extent=[0, w, h, 0])
    if player_df.empty:
        ax.set_title(f'No data to plot for Player ID: {player_id}')
    else:
        player_df['pos_x'] = (player_df['left_ankle_x'] + player_df['right_ankle_x']) / 2
        player_df['pos_y'] = (player_df['left_ankle_y'] + player_df['right_ankle_y']) / 2
        sns.kdeplot(x=player_df['pos_x'], y=player_df['pos_y'], cmap="rocket_r", fill=True, thresh=0.1, alpha=0.6, ax=ax)
        ax.set_title(f'Heatmap of Court Position for Player ID: {player_id}')
    ax.set_xlim(0, w); ax.set_ylim(h, 0); ax.axis('off')
    return fig


def filter_player_metrics(selection, analysis_results):
    """Filters the player metrics DataFrame based on the user's selection."""
    if not analysis_results or 'metrics' not in analysis_results:
        return pd.DataFrame() # Return empty DataFrame if no results
    
    full_metrics = analysis_results["metrics"]
    if selection == "All Players":
        return full_metrics
    else:
        return full_metrics[full_metrics['Player'] == selection]

# --- NEW: Function to analyze ball metrics ---
def analyze_ball_metrics(df, fps, perspective_matrix):
    """Analyzes ball data to find speed and bounces, with enhanced filtering."""
    if len(df) < 5:
        results = {'Max Speed (kph)': 'N/A', 'Bounces Detected': 'N/A'}
    else:
        df = df.sort_values(by='frame_id').copy()

        # --- Ball Speed Calculation with Outlier Rejection ---
        pixel_coords = np.float32(df[['x', 'y']].values).reshape(-1, 1, 2)
        meter_coords = cv2.perspectiveTransform(pixel_coords, perspective_matrix).reshape(-1, 2)
        distances = np.linalg.norm(np.diff(meter_coords, axis=0), axis=1)
        frame_diffs = np.diff(df['frame_id'].values)
        time_intervals = frame_diffs / fps
        
        valid_intervals = time_intervals > 0
        speeds_mps = distances[valid_intervals] / time_intervals[valid_intervals]
        
        realistic_speeds_kph = (speeds_mps * 3.6)[(speeds_mps * 3.6) < 300]
        max_speed_kph = np.max(realistic_speeds_kph) if len(realistic_speeds_kph) > 0 else 0

        # --- Ball Bounce Detection with Enhanced Robustness ---
        # **FIX**: Increase smoothing and add prominence to avoid false positives
        df['y_smooth'] = df['y'].rolling(window=5, center=True, min_periods=1).mean()
        min_court_y = SINGLES_COURT_CORNERS_PX[:, 1].min()
        # A prominence of 5 means the peak must rise at least 5 pixels above the surrounding data
        peaks, _ = find_peaks(df['y_smooth'], height=min_court_y, distance=int(fps / 2), prominence=5)
        num_bounces = len(peaks)

        results = {
            'Max Speed (kph)': f"{max_speed_kph:.1f}",
            'Bounces Detected': f"{num_bounces}"
        }

    final_df = pd.DataFrame.from_dict(results, orient='index', columns=['Value'])
    final_df = final_df.reset_index()
    final_df.columns = ['Metric', 'Value']
    return final_df
# MODIFIED: Corrected agility and transposed output table
# MODIFIED: Corrected agility and transposed output table
# MODIFIED: Corrected agility and reformatted output table
def analyze_real_world_metrics(df, fps):
    results = {}
    player_ids = sorted(df['track_id'].unique())

    for player_id in player_ids:
        player_df = df[df['track_id'] == player_id].sort_values(by='frame_id').copy()
        if len(player_df) < 10: continue

        # --- Player Height Approximation ---
        vertical_heights = player_df['right_ankle_y'] - player_df['nose_y']
        avg_height_pixels = vertical_heights[vertical_heights > 0].mean()

        # --- Exertion Metrics ---
        pixel_coords = np.float32(player_df[['right_hip_x', 'right_hip_y']].values).reshape(-1, 1, 2)
        meter_coords = cv2.perspectiveTransform(pixel_coords, PERSPECTIVE_MATRIX).reshape(-1, 2)
        distances = np.sqrt(np.sum(np.diff(meter_coords, axis=0)**2, axis=1))
        total_dist_meters = np.sum(distances)
        time_tracked_sec = len(player_df) / fps
        avg_speed_mps = total_dist_meters / time_tracked_sec if time_tracked_sec > 0 else 0

        # --- Stance & New Derived Metrics ---
        stance_pixels = np.sqrt((player_df['left_ankle_x'] - player_df['right_ankle_x'])**2 +
                                (player_df['left_ankle_y'] - player_df['right_ankle_y'])**2)
        
        avg_stance_pixels = stance_pixels.mean()
        stability_score = (avg_stance_pixels / avg_height_pixels) * 100 if avg_height_pixels > 0 else 0

        stance_change = np.diff(stance_pixels, prepend=stance_pixels.iloc[0])
        peaks, _ = find_peaks(stance_change, height=5, distance=int(fps))
        split_step_count = len(peaks)

        # --- Agility (Stance Variability) ---
        left_ankle_coords = np.float32(player_df[['left_ankle_x', 'left_ankle_y']].values).reshape(-1, 1, 2)
        right_ankle_coords = np.float32(player_df[['right_ankle_x', 'right_ankle_y']].values).reshape(-1, 1, 2)
        left_meter = cv2.perspectiveTransform(left_ankle_coords, PERSPECTIVE_MATRIX).reshape(-1, 2)
        right_meter = cv2.perspectiveTransform(right_ankle_coords, PERSPECTIVE_MATRIX).reshape(-1, 2)
        stance_meters = np.linalg.norm(left_meter - right_meter, axis=1)
        stance_variability_m = np.std(stance_meters)

        # --- Serve Height ---
        peak_serve_frame = player_df.loc[player_df['right_wrist_y'].idxmin()]
        serve_height_pixels = peak_serve_frame['right_wrist_y']

        results[f'Player {player_id}'] = {
            'Total Distance (m)': f"{total_dist_meters:.1f}",
            'Average Speed (m/s)': f"{avg_speed_mps:.2f}",
            'Stability Score (%)': f"{stability_score:.1f}",
            'Split Steps Detected': f"{split_step_count}",
            'Agility (Stance Variability m)': f"{stance_variability_m:.2f}",
            'Peak Serve Height (y-pixel)': f"{serve_height_pixels:.0f}"
        }
    
    if not results:
        return pd.DataFrame()

    # Create the transposed DataFrame
    final_df = pd.DataFrame.from_dict(results, orient='index').T
    final_df = final_df.reset_index().rename(columns={'index': 'Metric'})
    
    # **FIX**: Melt the DataFrame to a long format for clear presentation
    melted_df = final_df.melt(id_vars=['Metric'], var_name='Player', value_name='Value')
    return melted_df

def detect_server_receiver(df):
    player_ids = sorted(df['track_id'].unique())
    if len(player_ids) < 2:
        return "Not enough players to determine roles."

    initial_df = df[df['frame_id'] < 150].copy()
    if initial_df.empty:
        return "No player data in the first few seconds."

    p1_df = initial_df[initial_df['track_id'] == player_ids[0]]
    p2_df = initial_df[initial_df['track_id'] == player_ids[1]]

    if p1_df.empty or p2_df.empty:
        return "Could not track both players in the first few seconds."

    p1_avg_y = p1_df['right_ankle_y'].mean()
    p2_avg_y = p2_df['right_ankle_y'].mean()

    p1_wrist_range = p1_df['right_wrist_y'].max() - p1_df['right_wrist_y'].min()
    p2_wrist_range = p2_df['right_wrist_y'].max() - p2_df['right_wrist_y'].min()

    if p1_wrist_range > p2_wrist_range and p1_avg_y > p2_avg_y:
         server_id = player_ids[0]
    elif p2_wrist_range > p1_wrist_range and p2_avg_y > p1_avg_y:
         server_id = player_ids[1]
    else:
        server_id = player_ids[0] if p1_wrist_range > p2_wrist_range else player_ids[1]

    return f"Player {server_id} appears to be the Server."

# --- Step 3: The Main Processing and Analysis Function ---
def process_and_analyze_video(video_path, progress=gr.Progress(track_tqdm=True)):
    # --- Model Loading ---
    pytorch_device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {pytorch_device}")

    retinanet_model = load_retinanet('tennis_player_retinanet.pth', device=pytorch_device)
    movenet_model = load_movenet_model()

    tracknet_model_path = os.path.join('TrackNet', 'weights', 'TrackNet_best_latest123.pth')
    tracknet_model = load_tracknet_model(tracknet_model_path, device=pytorch_device)

    tracker = DeepSort(max_age=DEEPSORT_MAX_AGE, n_init=DEEPSORT_N_INIT)

    # --- Video I/O Setup ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None, None
    ret, background_frame = cap.read()
    if not ret: return None, None
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    temp_video_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    fourcc = cv2.VideoWriter_fourcc(*'avc1'); out = cv2.VideoWriter(temp_video_file.name, fourcc, fps, (w, h))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*'mp4v');
        out = cv2.VideoWriter(temp_video_file.name, fourcc, fps, (w, h))

    # --- Data Storage and Buffers ---
    all_pose_data = []
    last_known_poses = {}

    # NEW: Add a list to store ball data
    all_ball_data = []
    frame_buffer = deque(maxlen=3)
    last_ball_pos = None

    try:
        for frame_idx in progress.tqdm(range(total_frames), desc="Processing Video"):
            ret, frame = cap.read()
            if not ret: break

            frame_buffer.append(frame.copy())
            vis = frame.copy()

            # --- Ball Tracking (TrackNet) ---
            ball_x, ball_y = None, None
            if len(frame_buffer) == 3:
                ball_x, ball_y = detect_ball(tracknet_model, list(frame_buffer), device=pytorch_device)
                if ball_x is not None:
                    last_ball_pos = (ball_x, ball_y)
                    # NEW: Store ball coordinates with frame index
                    all_ball_data.append({'frame_id': frame_idx, 'x': ball_x, 'y': ball_y})

            # --- Player Detection & Pose Estimation (Runs every few frames) ---
            if frame_idx % (FRAME_SKIP + 1) == 0:
                resized, scale = preprocess_frame(frame)
                img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                img_tensor = torch.from_numpy(img_rgb / 255.).permute(2, 0, 1).float().unsqueeze(0).to(pytorch_device)

                with torch.no_grad():
                    outputs = retinanet_model(img_tensor)[0]

                boxes, scores = outputs['boxes'].cpu().numpy(), outputs['scores'].cpu().numpy()
                keep_mask = scores >= CONF_THRESH
                boxes, scores = boxes[keep_mask], scores[keep_mask]

                if len(boxes) > 0:
                    nms_indices = nms(torch.from_numpy(boxes), torch.from_numpy(scores), NMS_THRESH)
                    boxes, scores = boxes[nms_indices.numpy()], scores[nms_indices.numpy()]
                    boxes_orig = postprocess_boxes(boxes.copy(), scale)

                    dets_for_tracker = []
                    for b, s in zip(boxes_orig, scores):
                        x1, y1, x2, y2 = b
                        if cv2.pointPolygonTest(COURT_BOUNDARY, (int((x1 + x2) / 2), int(y2)), False) >= 0:
                            tlwh = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
                            dets_for_tracker.append((tlwh, float(s), 'person'))

                    tracks = tracker.update_tracks(dets_for_tracker, frame=frame)
                    active_tracks = [t for t in tracks if t.is_confirmed()]

                    current_poses = {}
                    for t in active_tracks:
                        tid = int(t.track_id)
                        x1, y1, x2, y2 = map(int, t.to_tlbr())
                        player_roi = frame[y1:y2, x1:x2]

                        if player_roi.size == 0: continue

                        keypoints = run_movenet(movenet_model, player_roi)
                        player_pose = {'frame_id': frame_idx, 'track_id': tid}
                        keypoint_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']

                        for i, name in enumerate(keypoint_names):
                            y_rel, x_rel, conf = keypoints[i]
                            player_pose[f'{name}_x'] = x1 + x_rel * player_roi.shape[1]
                            player_pose[f'{name}_y'] = y1 + y_rel * player_roi.shape[0]
                            player_pose[f'{name}_conf'] = conf

                        all_pose_data.append(player_pose)
                        current_poses[tid] = {'keypoints': keypoints, 'roi_box': (x1, y1, x2, y2), 'roi_dims': player_roi.shape[:2]}

                    last_known_poses = current_poses

            # --- Drawing Annotations ---
            for tid, pose_data in last_known_poses.items():
                x1, y1, x2, y2 = pose_data['roi_box']
                roi_h, roi_w = pose_data['roi_dims']
                keypoints = pose_data['keypoints']

                color = (0, 255, 0) if (tid % 2 == 0) else (0, 180, 255)
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f'ID:{tid}', (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                for p1_idx, p2_idx in KEYPOINT_EDGES:
                    y1_rel, x1_rel, score1 = keypoints[p1_idx]
                    y2_rel, x2_rel, score2 = keypoints[p2_idx]
                    if score1 > KEYPOINT_CONF_THRESH and score2 > KEYPOINT_CONF_THRESH:
                        pt1 = (int(x1 + x1_rel * roi_w), int(y1 + y1_rel * roi_h))
                        pt2 = (int(x1 + x2_rel * roi_w), int(y1 + y2_rel * roi_h))
                        cv2.line(vis, pt1, pt2, color=(0, 255, 0), thickness=2)

            if last_ball_pos:
                cv2.circle(vis, last_ball_pos, 10, (0, 0, 255), -1)
                cv2.circle(vis, last_ball_pos, 12, (255, 255, 255), 2)

            out.write(vis)
    finally:
        cap.release()
        out.release()

    # --- Final Analysis ---
    if not all_pose_data:
        if os.path.exists(temp_video_file.name):
            os.remove(temp_video_file.name)
        return None, None

    df = pd.DataFrame(all_pose_data)
    ball_df = pd.DataFrame(all_ball_data) # Create ball DataFrame

    player_ids = sorted(df['track_id'].unique())

    # Call all analysis functions
    player_metrics = analyze_real_world_metrics(df, fps)
    ball_metrics = analyze_ball_metrics(ball_df, fps, PERSPECTIVE_MATRIX) # New
    serve_receive_status = detect_server_receiver(df)

    if background_frame is None:
        return temp_video_file.name, None

    all_plots = {f'Player {pid}': plot_player_heatmap(df, pid, background_frame) for pid in player_ids}

    # Update the results dictionary
    analysis_results = {
        "plots": all_plots,
        "metrics": player_metrics,
        "ball_metrics": ball_metrics, # Add new ball metrics
        "serve_status": serve_receive_status,
        "player_ids": [f"Player {pid}" for pid in player_ids] if player_ids else ["No Players"]
    }
    return temp_video_file.name, analysis_results

def update_display(selected_player, analysis_results):
    if not analysis_results or not selected_player: return None, None
    player_heatmap = analysis_results["plots"].get(selected_player)
    # The metrics DataFrame is now transposed, so we select by column
    selected_metrics = analysis_results["metrics"][[selected_player]]
    return player_heatmap, selected_metrics

# --- Step 4: Create and Launch the Gradio UI --- (MODIFIED)
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎾 Tennis Pose and Performance Analysis")
    gr.Markdown("Upload a tennis video to automatically detect players, track their poses, and generate biomechanical and tactical analysis.")
    analysis_state = gr.State(value=None)
    
    with gr.Column():
        video_input = gr.Video(label="Upload Tennis Video")
        submit_button = gr.Button("Analyze Video")
        video_output = gr.File(label="Download Processed Video")
        serve_status_output = gr.Textbox(label="Serve/Receive Status", interactive=False)
        
        # Section for Player Metrics with Filter
        gr.Markdown("---")
        with gr.Row():
            metrics_output = gr.DataFrame(label="Player Metrics", wrap=True)
            metrics_player_selector = gr.Radio(label="Filter by Player", interactive=True)
        
        # Section for Ball Metrics (now below Player Metrics)
        gr.Markdown("---")
        ball_metrics_output = gr.DataFrame(label="Ball Tracking Metrics")
        
        # Section for Heatmap
        gr.Markdown("---")
        with gr.Row():
            heatmap_player_selector = gr.Radio(label="Select Player to Display Heatmap", interactive=True)
            heatmap_output = gr.Plot(label="Selected Player's Position Heatmap")

    # Main analysis event chain
    submit_button.click(
        fn=process_and_analyze_video,
        inputs=video_input,
        outputs=[video_output, analysis_state]
    ).then(
        fn=lambda state: (
            state["serve_status"],
            state["metrics"],
            state["ball_metrics"],
            gr.Radio(choices=["All Players"] + state["player_ids"], value="All Players"),
            gr.Radio(choices=state["player_ids"], value=state["player_ids"][0]),
            state["plots"].get(state["player_ids"][0])
        ) if state and state.get("player_ids") else (
            "Analysis failed or no players found.",
            None, None,
            gr.Radio(choices=[], value=None),
            gr.Radio(choices=[], value=None),
            None
        ),
        inputs=analysis_state,
        outputs=[
            serve_status_output,
            metrics_output,
            ball_metrics_output,
            metrics_player_selector,
            heatmap_player_selector,
            heatmap_output
        ]
    )

    # Event handler for the NEW player metrics filter
    metrics_player_selector.change(
        fn=filter_player_metrics,
        inputs=[metrics_player_selector, analysis_state],
        outputs=[metrics_output]
    )

    # Event handler for the heatmap selector
    heatmap_player_selector.change(
        fn=lambda player, state: state["plots"].get(player),
        inputs=[heatmap_player_selector, analysis_state],
        outputs=[heatmap_output]
    )


if __name__ == "__main__":
    demo.launch(debug=True)