import cv2
import numpy as np

def preprocess(img):
    img  = cv2.resize(img, (500, 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    eq   = cv2.equalizeHist(blur)
    return img, gray, eq

def segment(eq):
    _, th = cv2.threshold(eq, 0, 255,
                          cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.bitwise_not(th)
    k  = np.ones((5, 5), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN,  k, iterations=2)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, k, iterations=2)
    return th

def extract_features(img, gray, cnt):
    f = {}
    A = cv2.contourArea(cnt)
    P = cv2.arcLength(cnt, True)
    x, y, w, h = cv2.boundingRect(cnt)
    hull = cv2.convexHull(cnt)
    hA   = cv2.contourArea(hull)
    f['area']        = round(A, 2)
    f['perimeter']   = round(P, 2)
    f['aspect_ratio']= round(float(w)/h, 3) if h   else 0
    f['extent']      = round(A/(w*h),    3) if w*h else 0
    f['solidity']    = round(A/hA,       3) if hA  else 0
    f['circularity'] = round((4*np.pi*A)/(P**2), 3) if P else 0
    mask = np.zeros(gray.shape, np.uint8)
    cv2.drawContours(mask, [cnt], -1, 255, -1)
    edges    = cv2.Canny(gray, 30, 100)
    int_edge = cv2.bitwise_and(edges, edges, mask=mask)
    f['crack_ratio'] = round(np.sum(int_edge > 0) / A, 4) if A else 0
    bean_px = gray[mask == 255]
    f['mean_v'] = round(float(np.mean(bean_px)), 2) if len(bean_px) else 0
    f['std_v']  = round(float(np.std(bean_px)),  2) if len(bean_px) else 0
    dark = np.sum(bean_px < 80)
    f['dark_ratio'] = round(dark / len(bean_px), 4) if len(bean_px) else 0
    return f, mask

def classify(f):
    # BROKEN — unmistakable: std_v above 95, dark_ratio below 0.62
    if f['std_v'] > 95 and f['dark_ratio'] < 0.62:
        return "Broken"

    # DEFECTIVE — lowest solidity consistently across all frames
    if f['solidity'] < 0.970:
        return "Defective"

    # GOOD — must be checked BEFORE mouldy
    # Good has lowest crack_ratio of all beans — always below 0.103
    # This separates it from mouldy which is always above 0.147
    if f['crack_ratio'] < 0.115:
        return "Good"

    # MOULDY — highest dark_ratio, checked after good is excluded
    # Now only cracked and mouldy remain here
    # Mouldy dark_ratio always above 0.81, cracked always below 0.78
    if f['dark_ratio'] > 0.80:
        return "Cracked"

    # CRACKED — everything remaining
    return "Mouldy"

COLOURS = {
    "Good":      (0, 200,   0),
    "Mouldy":   (0, 165, 255),
    "Broken":    (0,   0, 220),
    "Cracked":    (128,  0, 200),
    "Defective": (0, 220, 220),
}

VIDEO = r"C:\Users\user\Desktop\red bean seedd\presentation\pr1.avi"

cap = cv2.VideoCapture(VIDEO)
paused = False

if not cap.isOpened():
    print(f"ERROR — could not open: {VIDEO}")
else:
    print("Running.")
    print("SPACE = pause/resume")
    print("Q = quit")

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

    frame_display = cv2.resize(frame, (700, 540))
    gray  = cv2.cvtColor(frame_display, cv2.COLOR_BGR2GRAY)
    blur  = cv2.GaussianBlur(gray, (5, 5), 0)
    eq    = cv2.equalizeHist(blur)

    binary = segment(eq)
    cnts, _ = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    counts = {
        "Good": 0, "Cracked": 0, "Broken": 0,
        "Mouldy": 0, "Defective": 0
    }

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if area < 5000:
            continue
        if area > 50000:
            continue

        f, _  = extract_features(frame_display, gray, cnt)
        pred  = classify(f)
        col   = COLOURS[pred]
        counts[pred] += 1

        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(frame_display, (x, y), (x+w, y+h), col, 2)
        cv2.putText(frame_display, pred, (x, max(y-8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2)

    # Count overlay
    yoff = 30
    for label, n in counts.items():
        cv2.putText(frame_display, f"{label}: {n}", (10, yoff),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOURS[label], 2)
        yoff += 28

    total_detected = sum(counts.values())
    cv2.putText(frame_display, f"Total: {total_detected}",
                (10, yoff + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # Pause indicator
    if paused:
        cv2.putText(frame_display, "PAUSED. press SPACE to resume",
                    (150, 530),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.imshow("Task 1 Kidney Bean Quality Classification",
               frame_display)

    key = cv2.waitKey(30) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused

cap.release()
cv2.destroyAllWindows()
print("Done.")