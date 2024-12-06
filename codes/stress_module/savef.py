import cv2
import os

def extract_frames(video_path, output_folder, frame_rate=1):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get the total number of frames and the frame rate
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"Total frames: {total_frames}, FPS: {fps}")

    count = 0
    saved_frames = 0

    while True:
        # Read the next frame
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop if no frame is returned
        
        # Save the frame at the specified interval
        if count % (fps // frame_rate) == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_frames:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frames += 1

        count += 1

    # Release the video capture object
    cap.release()
    print(f"Extracted {saved_frames} frames to '{output_folder}'.")

# Example usage
video_path = 'videos/s1.webm'
output_folder = 'path/to/save/frames'
extract_frames(video_path, output_folder, frame_rate=1)  # Extract 1 frame per second
