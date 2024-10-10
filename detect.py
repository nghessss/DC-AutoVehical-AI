import cv2
from ultralytics import YOLO
import numpy as np
import cv2
def get_traffic_light_color(traffic_light_image):
    """
    Get the color of the traffic light from the image.

    Parameters:
    - traffic_light_image: Image of the traffic light

    Returns:
    - The color of the traffic light (red, yellow, or green)
    """
    # Perform traffic light classification
    traffic_light_results = traffic_light_detector.predict(traffic_light_image, show_probabilities=False)
    traffic_light_color = np.argmax(traffic_light_results)
    if traffic_light_color == 0:
        return 'red light'
    elif traffic_light_color == 1:
        return 'yellow light'
    elif traffic_light_color == 2:
        return 'green light'
def annotate_frame(frame, results, box_color=(0, 255, 0), label_color=(0, 255, 0)):
    """
    Annotate the frame with bounding boxes and labels from the model results.

    Parameters:
    - frame: The original frame to annotate
    - results: The prediction results from a YOLO model
    - box_color: Color of the bounding box (BGR format)
    - label_color: Color of the text label (BGR format)
    """
    # Loop through detections in the results
    for result in results:
        boxes = result.boxes  # Extract bounding boxes
        for box in boxes:
            # Extract box coordinates, confidence score, and class ID
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Box coordinates
            cls = int(box.cls[0].item())  # Class ID
            class_name = result.names[cls]  
            if class_name == 'traffic light':
                # Perform traffic light classification
                print(y1, y2, x1, x2)
                if y2 - y1 > 20 and x2 - x1 > 20:
                    class_name = get_traffic_light_color(frame[y1:y2, x1:x2])
                
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)  # Draw the bounding box
            cv2.putText(frame, f"{class_name}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)  # Label

    return frame  
def process_detections(results, stop_list, slow_down_list):
    """
    Process detections from YOLO model and print information.
    
    Parameters:
    - results: YOLO model results for a frame.
    """
    for result in results:
        boxes = result.boxes  # Bounding boxes
        for box in boxes:
            # Extract bounding box coordinates
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            if class_name in stop_list:
                print(f"Detected {class_name}, Stop!")
                return
            elif class_name in slow_down_list:
                print(f"Detected {class_name}, Slow Down!")
                return
def read_list_from_file(file_path):
    """
    Read a list of items from a text file.

    Parameters:
    - file_path: Path to the text file.

    Returns:
    - A list of items read from the file.
    """
    with open(file_path, 'r') as file:
        items = file.read().splitlines()
    return items
def create_video_writer(video_cap, output_filename):
    # grab the width, height, and fps of the frames in the video stream.
    frame_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(video_cap.get(cv2.CAP_PROP_FPS))
    # initialize the FourCC and a video writer object
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    writer = cv2.VideoWriter(output_filename, fourcc, fps,
                             (frame_width, frame_height))
    return writer

video_path = str(input("Enter the video path: "))
output_name = str(input("Enter the output video name: "))
cap = cv2.VideoCapture(video_path)
stop_list = read_list_from_file('./stop.txt')
slow_down_list = read_list_from_file('./slow_down.txt')
writer = create_video_writer(cap, output_name)

import traffic_light_classifier as tlc
traffic_light_detector = tlc.Model()
traffic_light_detector.compile()
# Load the YOLO models
model = []
model.append(YOLO('./models/Crosswalks_Model.pt'))  # Model to detect crosswalks
model.append(YOLO('./models/yolov10s.pt'))  # Another model to detect different objects


while cap.isOpened():
    success, frame = cap.read()
    if success:
        results = []
        for m in model:
            results.append(m.predict(frame))
       
        annotated_frame = frame.copy()
        for result in results:
            annotated_frame = annotate_frame(annotated_frame, result)
            process_detections(result, stop_list, slow_down_list)
        resized = cv2.resize(annotated_frame, (1280, 720))
        writer.write(annotated_frame)
        cv2.imshow("Image", resized)    
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
writer.release()
cap.release()
cv2.destroyAllWindows()
