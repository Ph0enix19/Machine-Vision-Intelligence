import cv2
import numpy as np
from ultralytics import YOLO

# 1. Load your locally trained custom segmentation model (.pt file)
model = YOLO(r"C:\Users\onyan\PycharmProjects\PythonProject\Real-Time_Bean_Inspection\exp-2.pt")

# 2. Open the default laptop webcam (0 is typically the built-in camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

print("Webcam active... Press 'q' inside the video window to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # 3. Run inference on the live camera frame
    # verbose=False prevents console spam, keeping the live feed smooth
    results = model.predict(source=frame, conf=0.6, verbose=False)
    result = results[0]

    # 4. Perform Segmentation Analysis
    if result.masks is not None:
        masks_xy = result.masks.xy
        boxes = result.boxes

        for mask, box in zip(masks_xy, boxes):
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])

            # Calculate the physical pixel area enclosed by the AI's mask boundary
            mask_contour = mask.astype(np.int32)
            mask_pixel_area = cv2.contourArea(mask_contour)

            # Example Analysis Output (can be wired directly to a GUI later)
            # print(f"Detected: {class_name} | Conf: {confidence:.2f} | AI Mask Area: {mask_pixel_area:.1f} px")

    # 5. Generate the visual mask overlay frame
    annotated_frame = result.plot()

    # Display the live camera window
    cv2.imshow("YOLOv8 AI Live Segmentation Analysis", annotated_frame)

    # Break loop safely on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Webcam stream closed cleanly.")