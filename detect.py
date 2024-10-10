import cv2
from ultralytics import YOLO

# Load the YOLO models
model1 = YOLO('./models/Crosswalks_Model.pt')  # Model to detect crosswalks
model2 = YOLO('./models/yolov10s.pt')  # Another model to detect different objects

# Define the function to annotate the frame with detections from both models
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
            conf = box.conf[0].item()  # Confidence score
            cls = int(box.cls[0].item())  # Class ID
            label = f'{cls} {conf:.2f}'  # Create label with class and confidence

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)  # Draw the bounding box
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)  # Label

    return frame  # Return the annotated frame

# Define the video path and open the video
video_path = r"./e.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    if success:
        # Perform inference with both models
        results1 = model1.predict(frame)  # Results from the first model (e.g., crosswalk detection)
        results2 = model2.predict(frame)  # Results from the second model (e.g., object detection)

        # Create an annotated frame by copying the original frame for both model results
        annotated_frame = frame.copy()

        # Annotate the frame with detections from model1 (using green for bounding boxes)
        annotated_frame = annotate_frame(annotated_frame, results1, box_color=(0, 255, 0), label_color=(0, 255, 0))

        # Annotate the frame with detections from model2 (using blue for bounding boxes)
        annotated_frame = annotate_frame(annotated_frame, results2, box_color=(255, 0, 0), label_color=(255, 0, 0))

        # Display the annotated frame with both model detections
        cv2.imshow("YOLO Inference - Combined", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close display windows
cap.release()
cv2.destroyAllWindows()
