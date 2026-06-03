"""
Red Kidney Bean Vision Detection System
Task 1 – Vision-Based Seed Inspection (EE046-3-3-MVI)

Calibrated from real pixel data:
  - Ripe/brown bean:   H 0-20,  S 44-105, V 75-210
  - Dark/olive bean:   H 40-90, S 60-170, V 55-180
  - Near-black bean:   H 0-179, S 0-100,  V 5-90
  - Background:        V > 175 AND S < 55  → always excluded
  - Background glare:  H 60-179, S 0-90, V > 155 → excluded
"""

import cv2
import numpy as np
import sys
from pathlib import Path
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MIN_BEAN_AREA    = 4000
MAX_BEAN_AREA    = 180000
MAX_FRAME_FRAC   = 0.22
MIN_SOLIDITY     = 0.50
MIN_CIRCULARITY  = 0.25
MAX_ASPECT_RATIO = 3.5
PROCESS_EVERY    = 1
DISPLAY_WIDTH    = 960


# ─────────────────────────────────────────────────────────────────────────────
#  MASK BUILDER  –  calibrated to real bean pixel values
# ─────────────────────────────────────────────────────────────────────────────
def build_mask(frame):
    blur = cv2.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)
    hsv  = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # ── Bean ranges (from real pixel data) ───────────────────────────────────
    # Reddish-brown beans (H wraps: 0-20 and 160-179)
    m1 = cv2.inRange(hsv, np.array([0,   40,  50]), np.array([20,  255, 220]))
    m2 = cv2.inRange(hsv, np.array([155, 40,  50]), np.array([179, 255, 220]))
    # Olive/dark-green beans (H 25-90)
    m3 = cv2.inRange(hsv, np.array([25,  40,  40]), np.array([90,  255, 200]))
    # Very dark / near-black beans (any hue, low sat+val)
    m4 = cv2.inRange(hsv, np.array([0,   0,   5]), np.array([179, 110,  95]))
    # Dark teal/grey (catches the reflective dark left bean)
    m5 = cv2.inRange(hsv, np.array([85,  8,   8]), np.array([179, 180, 145]))

    mask = m1 | m2 | m3 | m4 | m5

    # ── Background removal ────────────────────────────────────────────────────
    # Pure white/near-white background
    bg_white = cv2.inRange(hsv, np.array([0,   0, 175]), np.array([179, 55, 255]))
    # Bright glare (green/teal/cyan light reflections off background)
    bg_glare = cv2.inRange(hsv, np.array([60,  0, 155]), np.array([179, 85, 255]))

    mask = cv2.bitwise_and(mask, cv2.bitwise_not(bg_white))
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(bg_glare))

    # ── Morphology ────────────────────────────────────────────────────────────
    k8 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))
    k4 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k8, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k4, iterations=2)
    mask = cv2.dilate(mask, k5, iterations=1)

    # Fill holes per blob
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled  = np.zeros_like(mask)
    for c in cnts:
        cv2.drawContours(filled, [c], -1, 255, cv2.FILLED)
    return filled


# ─────────────────────────────────────────────────────────────────────────────
#  WATERSHED  –  splits touching beans using distance transform peaks
# ─────────────────────────────────────────────────────────────────────────────
def watershed_split(frame, mask):
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)

    # Adaptive threshold: 50% of max — works for both round and elongated beans
    _, sure_fg = cv2.threshold(dist, 0.50, 1.0, cv2.THRESH_BINARY)
    sure_fg    = (sure_fg * 255).astype(np.uint8)

    # Remove very small foreground seeds (noise)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    sure_fg = cv2.erode(sure_fg, k3, iterations=1)

    sure_bg = cv2.dilate(mask,
                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                         iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers   += 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(frame.copy(), markers)
    return markers


# ─────────────────────────────────────────────────────────────────────────────
#  SHAPE FILTER
# ─────────────────────────────────────────────────────────────────────────────
def is_valid_bean(contour, frame_area):
    area = cv2.contourArea(contour)
    if not (MIN_BEAN_AREA < area < MAX_BEAN_AREA):
        return False
    if area / frame_area > MAX_FRAME_FRAC:
        return False
    perimeter = cv2.arcLength(contour, True)
    if perimeter == 0:
        return False
    circ = (4 * np.pi * area) / (perimeter ** 2)
    if circ < MIN_CIRCULARITY:
        return False
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    if hull_area == 0:
        return False
    if area / hull_area < MIN_SOLIDITY:
        return False
    x, y, w, h = cv2.boundingRect(contour)
    if max(w, h) / max(min(w, h), 1) > MAX_ASPECT_RATIO:
        return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN SEGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
def find_bean_contours(frame):
    h_f, w_f   = frame.shape[:2]
    frame_area = h_f * w_f

    mask    = build_mask(frame)
    markers = watershed_split(frame, mask)

    contours = []
    for lbl in np.unique(markers):
        if lbl <= 1:
            continue
        lbl_mask = np.zeros((h_f, w_f), np.uint8)
        lbl_mask[markers == lbl] = 255

        # Only process if this label has meaningful pixels
        if lbl_mask.sum() // 255 < MIN_BEAN_AREA:
            continue

        cnts, _ = cv2.findContours(lbl_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        if not is_valid_bean(c, frame_area):
            continue

        eps = 0.010 * cv2.arcLength(c, True)
        contours.append(cv2.approxPolyDP(c, eps, True))

    return contours, mask


# ─────────────────────────────────────────────────────────────────────────────
#  TEMPORAL SMOOTHER
# ─────────────────────────────────────────────────────────────────────────────
class ContourSmoother:
    def __init__(self):
        self.prev = []

    def update(self, contours):
        if not self.prev:
            self.prev = [self._cen(c) for c in contours]
            return contours
        matched  = [False] * len(self.prev)
        out, new_prev = [], []
        for c in contours:
            cx, cy = self._cen(c)
            bi, bd = -1, float('inf')
            for i, (px, py) in enumerate(self.prev):
                if matched[i]: continue
                d = (cx-px)**2 + (cy-py)**2
                if d < bd: bd, bi = d, i
            if bi >= 0 and bd < 160**2:
                matched[bi] = True
            out.append(c)
            new_prev.append((cx, cy))
        self.prev = new_prev
        return out

    @staticmethod
    def _cen(c):
        M = cv2.moments(c)
        if M['m00'] == 0:
            x, y, w, h = cv2.boundingRect(c)
            return x+w//2, y+h//2
        return int(M['m10']/M['m00']), int(M['m01']/M['m00'])

smoother = ContourSmoother()


# ─────────────────────────────────────────────────────────────────────────────
#  MATURITY  (recalibrated)
# ─────────────────────────────────────────────────────────────────────────────
def classify_maturity(h, s, v):
    STAGES = {
        "Unripe":    {"bgr": (40,  200,  40)},
        "Semi-Ripe": {"bgr": (0,   140, 255)},
        "Ripe":      {"bgr": (50,   50, 230)},
        "Overripe":  {"bgr": (120, 120, 180)},
    }
    if v < 70 and s < 130:                         stage = "Overripe"
    elif v < 50:                                    stage = "Overripe"
    elif 25 <= h <= 90 and s > 35:                  stage = "Unripe"
    elif 5  <= h < 25  and s > 55:                  stage = "Semi-Ripe"
    elif (h >= 150 or h <= 20) and s > 40 and v>=75: stage = "Ripe"
    elif s < 45:                                    stage = "Overripe"
    else:                                           stage = "Semi-Ripe"
    return {"stage": stage, **STAGES[stage]}


# ─────────────────────────────────────────────────────────────────────────────
#  QUALITY
# ─────────────────────────────────────────────────────────────────────────────
def classify_quality(mean_rgb, std_rgb, solidity):
    r, g, b  = mean_rgb
    std_mean = sum(std_rgb) / 3
    if solidity < 0.75:              return "Damaged"
    if std_mean > 48 or g > r*0.68:  return "Discolored"
    return "Healthy"


# ─────────────────────────────────────────────────────────────────────────────
#  SHAPE METRICS
# ─────────────────────────────────────────────────────────────────────────────
def compute_shape(contour):
    x, y, w, h  = cv2.boundingRect(contour)
    area        = cv2.contourArea(contour)
    perimeter   = cv2.arcLength(contour, True)
    ar          = round(w/h, 3) if h > 0 else 0
    circularity = round((4*np.pi*area)/perimeter**2, 3) if perimeter > 0 else 0
    hull        = cv2.convexHull(contour)
    hull_area   = cv2.contourArea(hull)
    solidity    = round(area/hull_area, 3) if hull_area > 0 else 0
    return {"bbox":(x,y,w,h),"length_px":w,"width_px":h,"area_px2":int(area),
            "perimeter_px":round(perimeter,1),"aspect_ratio":ar,
            "circularity":circularity,"solidity":solidity}


# ─────────────────────────────────────────────────────────────────────────────
#  ANALYSE BEAN
# ─────────────────────────────────────────────────────────────────────────────
def analyse_bean(frame, contour):
    rgb_img   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    hsv_img   = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    bean_mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.drawContours(bean_mask, [contour], -1, 255, cv2.FILLED)
    mean_rgb = tuple(int(v) for v in cv2.mean(rgb_img, mask=bean_mask)[:3])
    mean_hsv = tuple(int(v) for v in cv2.mean(hsv_img, mask=bean_mask)[:3])
    std_rgb  = tuple(round(rgb_img[:,:,c][bean_mask==255].std(),1) for c in range(3))
    shape    = compute_shape(contour)
    maturity = classify_maturity(*mean_hsv)
    quality  = classify_quality(mean_rgb, std_rgb, shape["solidity"])
    return {"mean_rgb":mean_rgb,"std_rgb":std_rgb,"mean_hsv":mean_hsv,
            "maturity":maturity["stage"],"maturity_bgr":maturity["bgr"],
            "quality":quality,**shape}


# ─────────────────────────────────────────────────────────────────────────────
#  DRAWING
# ─────────────────────────────────────────────────────────────────────────────
MATURITY_BGR = {
    "Unripe":    (40,  200,  40),
    "Semi-Ripe": (0,   140, 255),
    "Ripe":      (50,   50, 230),
    "Overripe":  (120, 120, 180),
}

def draw_results(frame, beans, contours):
    out  = frame.copy()
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    for b, cnt in zip(beans, contours):
        col = MATURITY_BGR.get(b["maturity"], (200,200,200))

        overlay = out.copy()
        cv2.drawContours(overlay, [cnt], -1, col, cv2.FILLED)
        cv2.addWeighted(overlay, 0.18, out, 0.82, 0, out)
        cv2.drawContours(out, [cnt], -1, col, 3)

        x, y, w, h = b["bbox"]
        r, g, bv   = b["mean_rgb"]
        lines = [
            f"{b['maturity']}  {b['quality']}",
            f"R{r} G{g} B{bv}",
        ]
        tw, th = 210, len(lines)*18 + 8
        lx = max(x, 0)
        ly = max(y - th - 6, 2)
        cv2.rectangle(out, (lx,ly), (lx+tw,ly+th), (15,15,15), -1)
        cv2.rectangle(out, (lx,ly), (lx+tw,ly+th), col, 1)
        for j, ln in enumerate(lines):
            cv2.putText(out, ln, (lx+4, ly+15+j*17),
                        FONT, 0.42, (255,255,255), 1, cv2.LINE_AA)

    mat = Counter(b["maturity"] for b in beans)
    cv2.rectangle(out, (0,0), (320,50), (20,20,20), -1)
    cv2.putText(out, f"Beans: {len(beans)}", (8,18),
                FONT, 0.52, (220,220,220), 1, cv2.LINE_AA)
    cv2.putText(out,
                f"Ripe:{mat.get('Ripe',0)}  Semi:{mat.get('Semi-Ripe',0)}"
                f"  Over:{mat.get('Overripe',0)}  Un:{mat.get('Unripe',0)}",
                (8,38), FONT, 0.46, (200,200,200), 1, cv2.LINE_AA)

    lx2, ly2 = 8, out.shape[0]-95
    cv2.rectangle(out,(lx2-4,ly2-20),(lx2+192,ly2+4*20+4),(20,20,20),-1)
    for k,(lbl,col) in enumerate([
        ("Ripe",        MATURITY_BGR["Ripe"]),
        ("Semi-Ripe",   MATURITY_BGR["Semi-Ripe"]),
        ("Unripe",      MATURITY_BGR["Unripe"]),
        ("Overripe",    MATURITY_BGR["Overripe"]),
    ]):
        cy = ly2+k*20
        cv2.rectangle(out,(lx2,cy-8),(lx2+14,cy+6),col,-1)
        cv2.putText(out,lbl,(lx2+20,cy+4),FONT,0.38,(210,210,210),1,cv2.LINE_AA)

    cv2.putText(out,"S=snapshot  R=report  M=mask  Q=quit",
                (out.shape[1]-340,out.shape[0]-8),
                FONT,0.36,(160,160,160),1,cv2.LINE_AA)
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  REPORT
# ─────────────────────────────────────────────────────────────────────────────
def print_report(beans, frame_no=0):
    SEP = "─"*72
    mat = Counter(b["maturity"] for b in beans)
    qua = Counter(b["quality"]  for b in beans)
    n   = max(len(beans),1)
    print(f"\n{'='*72}")
    print(f"  RED KIDNEY BEAN  –  FRAME {frame_no}  |  Beans: {len(beans)}")
    print(f"{'='*72}")
    print("\n  MATURITY")
    for s in ["Ripe","Semi-Ripe","Unripe","Overripe"]:
        c=mat.get(s,0); print(f"  {s:<12} {'█'*c:<20} {c:>3}  ({c/n*100:5.1f}%)")
    print("\n  QUALITY")
    for s in ["Healthy","Discolored","Damaged"]:
        c=qua.get(s,0); print(f"  {s:<14} {'█'*c:<20} {c:>3}  ({c/n*100:5.1f}%)")
    print(f"\n  {SEP}")
    print(f"  {'Maturity':<11}  {'Quality':<12}  "
          f"{'R':>3}{'G':>4}{'B':>4}  {'H':>3}{'S':>4}{'V':>4}  "
          f"{'Area':>7}  {'Perim':>7}  {'Circ':>5}  {'Solid':>5}")
    print(f"  {SEP}")
    for b in beans:
        r,g,bv=b["mean_rgb"]; h,s,v=b["mean_hsv"]
        print(f"  {b['maturity']:<11}  {b['quality']:<12}  "
              f"{r:>3}{g:>4}{bv:>4}  {h:>3}{s:>4}{v:>4}  "
              f"{b['area_px2']:>7}  {b['perimeter_px']:>7}  "
              f"{b['circularity']:>5.3f}  {b['solidity']:>5.3f}")
    print(f"  {SEP}\n")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────
def run(source):
    global smoother
    smoother = ContourSmoother()

    is_image = (isinstance(source,str) and
                Path(source).suffix.lower() in
                {".jpg",".jpeg",".png",".bmp",".tiff",".webp"})

    if is_image:
        frame = cv2.imread(source)
        if frame is None:
            print(f"[ERROR] Cannot read: {source}"); sys.exit(1)
        contours, mask = find_bean_contours(frame)
        beans  = [analyse_bean(frame,c) for c in contours]
        output = draw_results(frame,beans,contours)
        print_report(beans)
        out_path = str(Path(source).stem)+"_inspected.jpg"
        cv2.imwrite(out_path, output)
        print(f"[INFO] Saved → {out_path}")
        cv2.imshow("Red Bean Inspector", output)
        cv2.waitKey(0); cv2.destroyAllWindows()
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {source}"); sys.exit(1)

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w_v   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_v   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] {source}  |  {w_v}x{h_v}  {fps:.1f}fps  "
          f"{total if total>0 else 'live'} frames")
    print("[INFO] S=snapshot  R=report  M=toggle mask  Q=quit")

    show_mask=False; frame_idx=0
    beans=[]; contours=[]; last_output=None; last_mask=None

    while True:
        ret, frame = cap.read()
        if not ret: print("[INFO] End of video."); break
        frame_idx += 1

        if DISPLAY_WIDTH > 0 and frame.shape[1] > DISPLAY_WIDTH:
            scale = DISPLAY_WIDTH / frame.shape[1]
            frame = cv2.resize(frame,(0,0),fx=scale,fy=scale)

        if frame_idx % PROCESS_EVERY == 0:
            raw_cnts, last_mask = find_bean_contours(frame)
            contours    = smoother.update(raw_cnts)
            beans       = [analyse_bean(frame,c) for c in contours]
            last_output = draw_results(frame,beans,contours)

        display = last_output if last_output is not None else frame
        if show_mask and last_mask is not None:
            mc      = cv2.cvtColor(last_mask, cv2.COLOR_GRAY2BGR)
            display = cv2.addWeighted(display, 0.55, mc, 0.45, 0)

        cv2.imshow("Red Bean Inspector", display)
        key = cv2.waitKey(1) & 0xFF
        if   key == ord('q'): break
        elif key == ord('s'):
            fname = f"snapshot_frame{frame_idx}.jpg"
            cv2.imwrite(fname, display)
            print(f"[INFO] Saved → {fname}")
        elif key == ord('r'): print_report(beans, frame_no=frame_idx)
        elif key == ord('m'):
            show_mask = not show_mask
            print(f"[INFO] Mask: {'ON' if show_mask else 'OFF'}")

    cap.release(); cv2.destroyAllWindows()


# ─────────────────────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    SOURCE = r"C:\Users\AbdelRahman\Downloads\MVI Videos\MVI VID 1.avi"    # ← your video filename here

    if len(sys.argv) > 1:
        try:    SOURCE = int(sys.argv[1])
        except: SOURCE = sys.argv[1]

    run(SOURCE)