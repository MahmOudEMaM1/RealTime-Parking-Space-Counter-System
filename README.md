# 🚗 Real-Time Parking Space Detection with OpenCV

## Objective
This project is a real-time parking space detection system that processes video footage to detect and highlight available parking spots using OpenCV.

## Tools
- **OpenCV**: For video processing and drawing bounding boxes on parking spots.
- **NumPy**: For image manipulation and computing frame differences.
- **Custom Utility Functions**: `get_parking_spots_bboxes` and `empty_or_not` to handle spot classification and detection logic.

## Key Features
- **Real-Time Parking Spot Detection**: Detects empty and occupied parking spots in real-time from video footage.
- **Frame Comparison**: Compares frames to determine the status of parking spots by checking differences between consecutive frames.
- **Bounding Boxes**: Draws bounding boxes around each parking spot to visually indicate whether the spot is available (green) or occupied (red).
- **Dynamic Spot Counting**: Displays the number of available spots in real-time.
- **Scalable**: Can be applied to various parking lots by adjusting the spot bounding boxes using the mask.

## Usage
The project processes video input to analyze parking availability by comparing frames and visualizing results in real-time.

![Capture](https://github.com/user-attachments/assets/4f86414a-c09b-45a6-93a1-2deae37ca07e)
