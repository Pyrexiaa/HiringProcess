import cv2

# Open the video file
# video_path = 'path/to/video/file.mp4'
cap = cv2.VideoCapture(0)

# Check if the video file was opened successfully
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Variables for frame extraction
frame_counter = 0
frame_interval = 24  # Select every 10th frame

while True:
    # Read the next frame from the video
    ret, frame = cap.read()

    # Break the loop if no more frames are available
    if not ret:
        break

    # Check if the current frame should be selected
    if frame_counter % frame_interval == 0:
        # Convert the frame to PNG
        output_path = f'frame_{frame_counter}.png'
        cv2.imwrite(output_path, frame)
        print(f"Saved frame {frame_counter} as {output_path}")

    frame_counter += 1

# Release the video file
cap.release()
