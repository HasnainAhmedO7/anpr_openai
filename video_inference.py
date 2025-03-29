import base64
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

from openai_ocr import extract_text

cap = cv2.VideoCapture("anpr-demo-video.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer setup
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("anpr-output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Load YOLO license plate detection model
model = YOLO("anpr-demo-model.pt")

frame_count = 0
frame_skip = 5
padding = 10  # Adjust the padding value as needed

while cap.isOpened():
    success, im0 = cap.read()
    if not success:
        break

    frame_count += 1
    if frame_count % frame_skip != 0:
        continue

    results = model.predict(im0)[0].boxes
    boxes = results.xyxy.cpu()
    clss = results.cls.cpu()

    ann = Annotator(im0, line_width=3)

    for cls, box in zip(clss, boxes):
        height, width, _ = im0.shape  # Get the dimensions of the original image

        # Calculate padded coordinates
        x1 = max(int(box[0]) - padding, 0)
        y1 = max(int(box[1]) - padding, 0)
        x2 = min(int(box[2]) + padding, width)
        y2 = min(int(box[3]) + padding, height)

        # Crop the object with padding and encode it in base64
        base64_im0 = base64.b64encode(cv2.imencode(".jpg", im0[y1:y2, x1:x2])[1]).decode("utf-8")

        response = extract_text(base64_im0).choices[0].message.content
        print(f"Extracted text: {response}")

        ann.box_label(box, label=str(response), color=colors(cls, True))  # Draw the bounding boxes

    # Show the processed frame
    cv2.imshow("Processed Frame", im0)

    # Required to update the OpenCV window
    if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
        break

    video_writer.write(im0)  # Write the processed frame to the output video

# Release resources
cap.release()
video_writer.release()
cv2.destroyAllWindows()  # Close all OpenCV windows
