# DC-AutoVehical-AI
## Overview
This project implements an advanced object detection and decision-making system for autonomous vehicles, developed by Delta Cognition. It focuses on enhancing vehicle performance in rural and unfamiliar environments through three main tasks:

1. Detecting known objects in video footage
2. Detecting novel objects in video (research stage)
3. Making collision-avoidance decisions during navigation

## Features
- Utilizes YOLOv10-Small for efficient, real-time object detection
- Implements specialized crosswalk detection
- Includes traffic light color classification
- Makes real-time collision avoidance decisions based on detected objects

## Requirements
- Python 3.x
- OpenCV
- Ultralytics YOLO
- NumPy
- traffic-light-classifier

## Installation
1. Clone this repository
2. Install required packages:
   ```
   pip install opencv-python ultralytics numpy traffic-light-classifier
   ```
3. Download the required YOLO models:
   - YOLOv10-Small model
   - Crosswalk detection model

## Usage
1. Prepare your input video
2. Run the main script:
   ```
   python main.py
   ```
3. Enter the path to your input video when prompted
4. Specify the output video name

## File Structure
- `main.py`: Main script for video processing and decision making
- `models/`: Directory containing YOLO model weights
- `stop.txt`: List of objects that trigger a stop action
- `slow_down.txt`: List of objects that trigger a slow down action

## Acknowledgements
- YOLOv10 by Ultralytics
- Crosswalk detection model by xN1ckuz
- Traffic light classifier by Shashank Kumbhare

## Future Work
- Implementation of novel object detection using cooperative foundational models and zero-shot detection techniques

## Contact
Vo Thanh Nghia - vothanhnghia270604@gmail.com

Project Link: [[Project link](https://github.com/nghessss/DC-AutoVehical-AI/)]
