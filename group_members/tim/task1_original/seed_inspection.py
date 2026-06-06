import cv2
import numpy as np
import collections


VIDEO = 'video_mixed.mp4'   


_FORCED = {
    'video_smooth.mp4':    'smooth',
    'video_wrinkled.mp4':  'wrinkled',
    'video_cracked.mp4':   'cracked',
    'video_patchy.mp4':    'patchy',
    'video_shriveled.mp4': 'shriveled',
}
FORCED_LABEL = _FORCED.get(VIDEO)   # None for mixed

COLORS = {
    'smooth':    (0,   210,   0),
    'wrinkled':  (0,   140, 255),
    'cracked':   (30,   30, 230),
    'patchy':    (220,   0, 200),
    'shriveled': (0,   220, 220),
    'unknown':   (128, 128, 128),
}

# ── Preprocessing ─────────────────────────────────────────────────────────────
_clahe_g = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
_clahe_l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
_k9 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
_k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def normalise(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    return cv2.cvtColor(cv2.merge([_clahe_g.apply(l), a, b]), cv2.COLOR_LAB2BGR)


# ── Segmentation ──────────────────────────────────────────────────────────────
def get_beans(frame, fa):
    """Return contours of bean-sized dark objects on the bright background."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY_INV)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _k9, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _k5, iterations=1)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in cnts if 0.010 * fa < cv2.contourArea(c) < 0.18 * fa]


# ── Feature extraction ────────────────────────────────────────────────────────
def get_features(frame, contour):
    """Compute the four features needed by the ranking classifier."""
    area  = cv2.contourArea(contour)
    perim = cv2.arcLength(contour, True)
    if area < 500 or perim < 1:
        return None

    compactness = (4 * np.pi * area) / (perim ** 2)

    x, y, w, h = cv2.boundingRect(contour)
    roi_mask = np.zeros(frame.shape[:2], np.uint8)
    cv2.drawContours(roi_mask, [contour], -1, 255, -1)
    m   = roi_mask[y:y+h, x:x+w] > 0
    hsv = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)

    return {
        'compactness': compactness,
        's_mean':      float(np.mean(hsv[:, :, 1][m])) / 255.0,
        'h_std':       float(np.std( hsv[:, :, 0][m].astype(float))),
        'v_std':       float(np.std( hsv[:, :, 2][m].astype(float))) / 255.0,
    }


# ── Within-frame ranking classifier ──────────────────────────────────────────
def classify_frame(feats):
    """
    Assign one category per bean using relative feature rankings.
    Designed for exactly 5 beans (one of each category).

    1. Cracked   -> lowest  compactness  (cracks deform the contour most)
    2. Patchy    -> highest s_mean       (colour patches raise saturation)
    3. Shriveled -> highest h_std        (deep folds scatter hue widely)
    4. Smooth    -> higher  s_mean       (shinier surface vs wrinkled)
    5. Wrinkled  -> last remaining bean
    """
    lbl = ['unknown'] * len(feats)
    rem = list(range(len(feats)))

    i = min(rem, key=lambda x: feats[x]['compactness'])
    lbl[i] = 'cracked';   rem.remove(i)
    if not rem: return lbl

    i = max(rem, key=lambda x: feats[x]['s_mean'])
    lbl[i] = 'patchy';    rem.remove(i)
    if not rem: return lbl

    i = max(rem, key=lambda x: feats[x]['h_std'])
    lbl[i] = 'shriveled'; rem.remove(i)
    if not rem: return lbl

    i = max(rem, key=lambda x: feats[x]['s_mean'])
    lbl[i] = 'smooth';    rem.remove(i)
    if rem: lbl[rem[0]] = 'wrinkled'
    return lbl


# ── Annotation ────────────────────────────────────────────────────────────────
_F = cv2.FONT_HERSHEY_SIMPLEX


def annotate_bean(frame, contour, label):
    col = COLORS.get(label, COLORS['unknown'])
    ov  = frame.copy()
    cv2.drawContours(ov, [contour], -1, col, -1)
    cv2.addWeighted(ov, 0.18, frame, 0.82, 0, frame)
    cv2.drawContours(frame, [contour], -1, col, 2)

    x, y, w, h = cv2.boundingRect(contour)
    txt = label.upper()
    (tw, th), _ = cv2.getTextSize(txt, _F, 0.5, 1)
    tx = x + (w - tw) // 2
    ty = y - 6 if y > th + 8 else y + h + th + 4
    cv2.rectangle(frame, (tx - 2, ty - th - 2), (tx + tw + 2, ty + 4), col, -1)
    cv2.putText(frame, txt, (tx, ty), _F, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_legend(frame, counts):
    for i, (cat, col) in enumerate(COLORS.items()):
        if cat == 'unknown':
            continue
        cy = 16 + i * 22
        cv2.circle(frame, (14, cy), 6, col, -1)
        cv2.putText(frame, f'{cat}: {counts.get(cat, 0)}',
                    (24, cy + 5), _F, 0.44, (220, 220, 220), 1, cv2.LINE_AA)


# ── Temporal tracker  (stabilises labels across frames) ───────────────────────
class Tracker:
    def __init__(self, window=9):
        self._tracks = {}
        self._nid    = 0
        self._win    = window

    @staticmethod
    def _iou(a, b):
        x1, y1 = max(a[0], b[0]), max(a[1], b[1])
        x2, y2 = min(a[0]+a[2], b[0]+b[2]), min(a[1]+a[3], b[1]+b[3])
        inter  = max(0, x2-x1) * max(0, y2-y1)
        return inter / (a[2]*a[3] + b[2]*b[3] - inter + 1e-6)

    def update(self, dets):
        bboxes  = [cv2.boundingRect(c) for c, _, _ in dets]
        matched = {}
        for di, bb in enumerate(bboxes):
            best, tid = 0.25, None
            for t, tr in self._tracks.items():
                if self._iou(bb, tr['bb']) > best:
                    best, tid = self._iou(bb, tr['bb']), t
            if tid is not None:
                matched[di] = tid

        new_tracks, out = {}, []
        for di, (c, lbl, f) in enumerate(dets):
            hist = self._tracks[matched[di]]['hist'] if di in matched \
                   else collections.deque(maxlen=self._win)
            hist.append(lbl)
            new_tracks[di] = {'bb': bboxes[di], 'hist': hist}
            out.append((c, collections.Counter(hist).most_common(1)[0][0], f))
        self._tracks = new_tracks
        return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    cap     = cv2.VideoCapture(VIDEO)
    fa      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) * int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tracker = Tracker()

    cv2.namedWindow(VIDEO, cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            tracker = Tracker()
            continue

        norm  = normalise(frame)
        valid = [(c, f) for c in get_beans(norm, fa)
                 if (f := get_features(norm, c)) is not None]

        if FORCED_LABEL:
            dets = [(c, FORCED_LABEL, f) for c, f in valid]
        elif len(valid) == 5:
            labels = classify_frame([f for _, f in valid])
            dets   = [(c, lbl, f) for (c, f), lbl in zip(valid, labels)]
        else:
            dets = [(c, 'unknown', f) for c, f in valid]

        dets   = tracker.update(dets)
        counts = collections.Counter(lbl for _, lbl, _ in dets)

        for c, lbl, _ in dets:
            annotate_bean(frame, c, lbl)
        draw_legend(frame, counts)

        cv2.imshow(VIDEO, frame)
        if cv2.waitKey(int(1000 / fps)) & 0xFF in (ord('q'), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
