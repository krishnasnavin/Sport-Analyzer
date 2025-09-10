import cv2
import numpy as np

# --- Configuration ---
VIDEO_PATH = 'Tennis Test - 2.mp4'  # <-- IMPORTANT: Change this to your video path
DISPLAY_MAX_WIDTH = 1280       # Max width of the display window to fit your screen

# --- Global variables ---
points = []
scale_factor = 1.0  # To handle resizing

window_name = "Point Selector"
point_labels = ["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
selected_frame = None

def get_instruction():
    """Returns the instruction for the next point to click."""
    num_points = len(points)
    if num_points < 4:
        return f"Click to select the {point_labels[num_points]} point"
    else:
        return "All 4 points selected. Press any key."

def mouse_callback(event, x, y, flags, param):
    """Handles mouse clicks for point selection and scales coordinates."""
    global selected_frame
    
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        # Scale the clicked point back to the original frame's resolution
        original_x = int(x / scale_factor)
        original_y = int(y / scale_factor)
        points.append((original_x, original_y))
        
        point_index = len(points)
        
        # Draw feedback on the display frame
        cv2.circle(selected_frame, (x, y), 7, (0, 0, 255), -1)
        cv2.putText(selected_frame, str(point_index), (x + 10, y + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        
        cv2.setWindowTitle(window_name, get_instruction())
        
        # Print the true coordinates from the original frame
        print(f"-> Point {point_index} ({point_labels[point_index-1]}) selected: [{original_x}, {original_y}]")

def main():
    """Main function to select a frame and then select points."""
    global scale_factor, selected_frame

    # =============== Part 1: Frame Selection ===============
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {VIDEO_PATH}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_idx = 0
    
    print("--- Frame Selection ---")
    print("Use 'A' (back) and 'D' (forward) to navigate frames.")
    print("Press ENTER to select the current frame.")
    print("Press 'Q' to quit.")
    print("-----------------------")

    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            print("Reached end of video.")
            frame_idx -= 1
            continue

        h, w = frame.shape[:2]
        if w > DISPLAY_MAX_WIDTH:
            scale_factor = DISPLAY_MAX_WIDTH / w
            display_h = int(h * scale_factor)
            display_frame = cv2.resize(frame, (DISPLAY_MAX_WIDTH, display_h))
        else:
            scale_factor = 1.0
            display_frame = frame.copy()
        
        # Add navigation instructions on the frame
        info_text = f"Frame: {frame_idx + 1}/{total_frames} | A/D to Navigate | ENTER to Select | Q to Quit"
        cv2.putText(display_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(0) & 0xFF

        if key == ord('d'): # 'd' for next frame
            frame_idx = min(frame_idx + 1, total_frames - 1)
        elif key == ord('a'): # 'a' for previous frame
            frame_idx = max(frame_idx - 1, 0)
        elif key == 13: # 13 is the Enter key
            selected_frame = display_frame.copy() # Use the resized frame for display
            print(f"\nFrame {frame_idx + 1} selected.")
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return

    # =============== Part 2: Point Selection ===============
    cv2.setWindowTitle(window_name, get_instruction())
    cv2.setMouseCallback(window_name, mouse_callback)

    print("\n--- Point Selection ---")
    print("Click on the image to select the four corners.")
    print("Press 'Q' to quit.")
    print("-----------------------")

    # --- START OF FIX ---
    while True:
        cv2.imshow(window_name, selected_frame)
        key = cv2.waitKey(1) & 0xFF

        # First, check for the quit command
        if key == ord('q'):
            break

        # Then, check if we are done selecting points
        if len(points) >= 4:
            print("\nAll 4 points selected. Press any key in the window to finish.")
            cv2.waitKey(0) # Wait indefinitely for one final key press
            break
    # --- END OF FIX ---
            
    # =============== Final Output ===============
    if len(points) == 4:
        print("\n" + "="*40)
        print("## Your Polygon Points (ready to copy) ##")
        print("="*40)
        for i, (p, label) in enumerate(zip(points, point_labels)):
            comma = "," if i < len(points) - 1 else ""
            print(f"    [{p[0]}, {p[1]}]{comma}    # {label}")
        print("="*40)
    else:
        print("\nExited before 4 points were selected.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()