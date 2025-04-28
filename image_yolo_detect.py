import cv2
import os
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("C:/8TH-SEM/code/output_yolodataset1000/detect/train/weights/best.pt")  # Replace with your trained model path

# Specify the folder containing the images
image_folder = "C:/8TH-SEM/code/test_image"  # Replace with your folder path

# Get a list of all image files in the folder
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Check if the folder contains any images
if not image_files:
    print(f"No images found in folder: {image_folder}")
    exit()

# Specify a separate folder to save annotated images
output_folder = "C:/8TH-SEM/result/yolo_prediction"  # Replace with your desired output folder path
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist

# Process each image in the folder
for image_file in image_files:
    # Load the image
    image_path = os.path.join(image_folder, image_file)
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error reading image: {image_file}")
        continue

    # Run YOLO model on the image
    results = model(image)

    # Draw detections on the image
    annotated_image = results[0].plot()  # Get annotated image with detections

    # Save the annotated image into the separate folder
    output_path = os.path.join(output_folder, image_file)
    cv2.imwrite(output_path, annotated_image)

    # Optionally, display the image with detections
    cv2.imshow("YOLOv8 Detection", annotated_image)

    # Wait for a key press (press 'q' to quit early)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

# Close all OpenCV windows
cv2.destroyAllWindows()

print(f"Processing complete. Annotated images saved to: {output_folder}")
