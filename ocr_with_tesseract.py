import base64
import cv2
import pytesseract
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

cap = cv2.VideoCapture("anpr-demo-video.mp4")
assert cap.isOpened(), "Error reading video file"

# Video writer setup
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("anpr-output.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Load YOLO license plate detection model
model = YOLO("anpr-demo-model.pt")

# Tesseract configuration for license plates
custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

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

        # Crop the license plate region
        plate_img = im0[y1:y2, x1:x2]
        
        # Preprocess the image for better OCR results
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold to get black text on white background
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Use Tesseract to extract text
        plate_text = pytesseract.image_to_string(thresh, config=custom_config).strip()
        
        print(f"Extracted text: {plate_text}")

        ann.box_label(box, label=str(plate_text), color=colors(cls, True))  # Draw the bounding boxes

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