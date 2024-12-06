import streamlit as st
import zipfile
import os
import shutil
from moviepy.editor import VideoFileClip
from main import *  # Ensure that `analyze_live_video` is properly imported
import random
import string

# Generate a random string of length 6

# Define paths
zip_file_path = 'output.zip'
output_folder = 'output'

# Function to process video and save output
def process_video(input_file_path, output_folder):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load the video file
    video = VideoFileClip(input_file_path)

    # Example processing: Save each frame as an image (customize as needed)
    for i, frame in enumerate(video.iter_frames()):
        frame_path = os.path.join(output_folder, f'frame_{i:04d}.png')
        # Save the frame as an image (you might need to use a library for this, e.g., PIL)
        # Example: imageio.imwrite(frame_path, frame)
    
    # You can add more processing steps here

# Streamlit app
st.title("Interview assistant")

# Upload video file
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_file_path = 'temp_video_file.mp4'
    with open(temp_file_path, 'wb') as f:
        f.write(uploaded_file.read())
    
    # Create a "Predict" button
    if st.button("Predict"):
        # Process the video and save output
        uid = ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        analyze_live_video(temp_file_path, uid, 1, 1, True, print)
        
        # Compress the output folder into a ZIP file
        def compress_folder(folder_path, zip_file_path):
            with zipfile.ZipFile(zip_file_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(folder_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        zipf.write(file_path, os.path.relpath(file_path, folder_path))        
        compress_folder(output_folder, zip_file_path)
        
        # Read the ZIP file to be downloaded
        with open(zip_file_path, 'rb') as file:
            zip_data = file.read()

        # Create a download button for the ZIP file
        st.download_button(
            label="Download Processed output",
            data=zip_data,
            file_name="output.zip",
            mime="application/zip"
        )
        
        # Clean up temporary files
        os.remove(temp_file_path)
        # shutil.rmtree(output_folder)
        os.remove(zip_file_path)