import cv2
import numpy as np

VIDEO_PATH = r"C:\Users\Mohmed\Downloads\mvi data\try2.avi"


# Parameterrs
GREEN_BG_LOWER = np.array([35, 45, 35], dtype=np.uint8)
GREEN_BG_UPPER = np.array([85, 255, 255], dtype=np.uint8)

MIN_BEAN_AREA = 4000
MIN_SOLIDITY = 0.78
MIN_EXTENT = 0.38
SHOW_PROCESS_STAGES = True
DISPLAY_SCALE = 0.6


# Colors for drawing contours and labels


CLASS_COLORS = {
    "White Kidney Bean": (255, 255, 255),     # White
    "Dark Kidney Bean": (0, 0, 255),          # Red
    "Speckled Kidney Bean": (0, 255, 255)     # Yellow
}



# CLASSIFICATION

def classify_bean(mean_s, mean_v, mean_gray):

    if mean_v > 120 and mean_s < 70 and mean_gray > 110:
        return "White Kidney Bean"

    if mean_v < 35 or mean_gray < 25:
        return "Dark Kidney Bean"

    return "Speckled Kidney Bean"



# MASK CREATION

def make_bean_mask(frame):

    blur = cv2.GaussianBlur(frame, (5, 5), 0)

    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    
    green_mask = cv2.inRange(hsv, GREEN_BG_LOWER, GREEN_BG_UPPER)

    
    raw_bean_mask = cv2.bitwise_not(green_mask)

    
    kernel_3 = np.ones((3, 3), np.uint8)

    final_mask = cv2.morphologyEx(
        raw_bean_mask,
        cv2.MORPH_OPEN,
        kernel_3,
        iterations=1
    )

    return green_mask, raw_bean_mask, final_mask


# EDGE DETECTION

def make_edges(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(
        blur,
        50,
        150
    )

    return edges




def classify_frame(frame):

    green_mask, raw_mask, final_mask = make_bean_mask(frame)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    contours, _ = cv2.findContours(
        final_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    beans = []

    for contour in contours:

        area = cv2.contourArea(contour)

        if area < MIN_BEAN_AREA:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        solidity = 0.0 if hull_area == 0 else float(area) / float(hull_area)
        extent =  0.0 if w * h == 0 else float(area) / float(w * h)

        if solidity < MIN_SOLIDITY or extent < MIN_EXTENT:
            continue

        # Bean-only mask
        bean_only = np.zeros(gray.shape, dtype=np.uint8)

        cv2.drawContours(
            bean_only,
            [contour],
            -1,
            255,
            -1
        )

        pixels = bean_only == 255

        mean_h, mean_s, mean_v = np.mean(hsv[pixels], axis=0)

        mean_gray = float(np.mean(gray[pixels]))

        bean_class = classify_bean(
            mean_s,
            mean_v,
            mean_gray
        )

        # Center
        moments = cv2.moments(contour)

        if moments["m00"] != 0:
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
        else:
            center_x = x + w // 2
            center_y = y + h // 2

        beans.append({
            "contour": contour,
            "class": bean_class,
            "area": area,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "center_x": center_x,
            "center_y": center_y
        })

    return beans, green_mask, raw_mask, final_mask


def draw_results(frame, beans):

    output = frame.copy()

    counts = {
        "White Kidney Bean": 0,
        "Dark Kidney Bean": 0,
        "Speckled Kidney Bean": 0
    }

    for bean in beans:

        bean_class = bean["class"]

        counts[bean_class] += 1

        color = CLASS_COLORS[bean_class]

        contour = bean["contour"]

        # Draw contour
        cv2.drawContours(
            output,
            [contour],
            -1,
            color,
            3
        )

        # Draw label
        label_x = max(5, bean["center_x"] - 60)
        label_y = max(20, bean["center_y"] - 10)

        cv2.putText(
            output,
            bean_class,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    # Summary bar
    summary = (
        f"Beans={len(beans)}   "
        f"White={counts['White Kidney Bean']}   "
        f"Dark={counts['Dark Kidney Bean']}   "
        f"Speckled={counts['Speckled Kidney Bean']}"
    )

    cv2.rectangle(
        output,
        (10, 10),
        (950, 50),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        output,
        summary,
        (20, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    return output

# DRAW EDGE + CONTOURS

def draw_edges_with_contours(edges, beans):

    edge_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    for bean in beans:

        color = CLASS_COLORS[bean["class"]]

        cv2.drawContours(
            edge_bgr,
            [bean["contour"]],
            -1,
            color,
            2
        )

    return edge_bgr

#Draw the boxes around the beans

def draw_boxes(frame, beans):

    output = frame.copy()

    for bean in beans:

        x = bean["x"]
        y = bean["y"]
        w = bean["w"]
        h = bean["h"]

        bean_class = bean["class"]

        color = CLASS_COLORS[bean_class]

        # Draw box
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            color,
            3
        )

        # Draw label
        cv2.putText(
            output,
            bean_class,
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

    return output

# Resize all the display windows
def resize_display(image, scale=DISPLAY_SCALE):

    return cv2.resize(
        image,
        None,
        fx=scale,
        fy=scale
    )

# MAIN

def main():

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Could not open video:\n{VIDEO_PATH}"
        )

    paused = False

    while True:

        if not paused:

            ok, frame = cap.read()

            if not ok:
                print("Finished video.")
                break

            
            # PROCESS
            

            beans, green_mask, raw_mask, final_mask = classify_frame(frame)

            output = draw_results(frame, beans)

            edges = make_edges(frame)

            edges_with_contours = draw_edges_with_contours(
                edges,
                beans
            )

            boxes_output = draw_boxes(frame, beans)

            
            # SHOW WINDOWS
            
            cv2.imshow(
                "1 - Original",
                resize_display(frame)
            )

            if SHOW_PROCESS_STAGES:

                cv2.imshow(
                    "2 - Green Mask",
                    resize_display(green_mask)
                )

                cv2.imshow(
                    "3 - Raw Bean Mask",
                    resize_display(raw_mask)
                )

                cv2.imshow(
                    "4 - Final Bean Mask",
                    resize_display(final_mask)
                )

                # Edge only
                cv2.imshow(
                    "5 - Edge Detection Only",
                    resize_display(edges)
                )

                # Edge + contours
                cv2.imshow(
                    "6 - Edge Detection + Contours",
                    resize_display(edges_with_contours)
                )

            # Bounding boxes
            cv2.imshow(
                "7 - Bounding Boxes",
                resize_display(boxes_output)
            )

            # Final classification
            cv2.imshow(
                "8 - Final Classification",
                resize_display(output)
            )
        

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q') or key == 27:
            break

        elif key == ord(' '):
            paused = not paused

    cap.release()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()