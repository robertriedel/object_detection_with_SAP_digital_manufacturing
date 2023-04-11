import cv2
import argparse
import supervision as sv
import requests

from collections import Counter
from ultralytics import YOLO
from oauth2_request import get_access_token

PERSON_LABEL = "cell phone"
API_ENDPOINT = ""
# Define the OAuth 2.0 endpoint and credentials
token_endpoint = ""
client_id = ""
client_secret = ""
#Obtain the access token using the client credentials
access_token = get_access_token(token_endpoint, client_id, client_secret)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution", 
        default=[1280, 720], 
        nargs=2, 
        type=int
    )
    args = parser.parse_args()
    return args


def call_API(object_label, number, access_token, confidence):
    print(confidence)

    # Send API POST request if a specified object is detected
    if object_label == PERSON_LABEL:
        data = {
            'InPlant': "--yourplant---",
            'InWorkCenter': "--yourworkcenter--",
            'InArray': [
                {
                    'object_detected': object_label,
                    'number_of_persons': number
                }
            ]
        }
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}'
        }
        response = requests.post(API_ENDPOINT, json=data, headers=headers)
        if response.status_code != 200:
            print(f"Error sending API POST request: {response.text}")
        #time.sleep(10)


def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLO("yolov8m.pt")

    box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
    )

    detected_objects = Counter()

    while True:
        ret, frame = cap.read()
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_yolov8(result)
        
        labels = []
        confidences = {}
        for detection in detections:
            confidence = detection[2]
            class_id = detection[3]
            label = model.names[class_id]
            labels.append(label)
            confidences[label] = confidence
            # Check if the detected object is new and its confidence is above 0.8
            if label not in detected_objects and confidence > 0.8:
                detected_objects[label] = True
                if label == PERSON_LABEL:
                    print(f"Detected a new {label} with confidence {confidence}")
                    call_API(label, 1, access_token, confidence)



        frame = box_annotator.annotate(
            scene=frame, 
            detections=detections,
            labels=labels
        )


        cv2.imshow("yolov8", frame)

        if (cv2.waitKey(30) == 27):
            break



if __name__ == "__main__":
    main()
