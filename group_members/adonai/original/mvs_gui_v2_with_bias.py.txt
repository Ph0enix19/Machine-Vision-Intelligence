# mvs_gui_v2_with_bias.py
import sys
import cv2
import numpy as np
from ctypes import *
from ultralytics import YOLO
import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image, ImageTk
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import datetime
import pandas as pd

sys.path.append(r"C:\Users\user\Desktop\PythonProject5")

try:
    from MvImport.MvCameraControl_class import *
    print("✓ MvImport loaded successfully")
except ImportError as e:
    print(f"✗ Error importing MvImport: {e}")
    exit(1)

print("\n" + "=" * 60)
print("  BEAN DETECTION - MVS CAMERA WITH GUI")
print("=" * 60)

# =================================================================
# LOAD YOLO MODEL
# =================================================================
print("\nLoading YOLO model...")
model = YOLO(r"C:\Users\user\Desktop\PythonProject5\runs\detect\bean_v2\exp1\weights\best.pt")
print(f"✓ Model loaded  |  Classes: {model.names}")

# CLAHE — adaptive contrast normalisation for lighting robustness
# Divides each frame into tiles and equalises contrast independently
# so YOLO sees consistent images whether the lab is dim or bright
clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(16, 16))

def apply_clahe(img_bgr):
    """Apply CLAHE in LAB colour space.
    Only the L (lightness) channel is equalised — colours stay natural."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

# =================================================================
# CONFIGURATION
# Adjust these values to tune detection per class.
# Lower threshold  = easier to detect (more boxes)
# Higher threshold = harder to detect (fewer false positives)
# =================================================================

CLASS_CONF_THRESHOLD = {
    "good":      0.15,   # lowered — good beans under-detected
    "cracked":   0.35,
    "broken":    0.40,
    "mouldy":    0.30,   # raised — over-triggers on rotation
    "defective": 0.40,
}

# CLASS BIAS - Multiplies confidence before threshold check
# Higher number = more likely to detect that class
# Lower number = less likely to detect that class
CLASS_BIAS = {
    "good":      3.8,    # BOOST good bean detection
    "cracked":   1.5,    # Neutral
    "broken":    1.0,    # Neutral
    "mouldy":    1.0,    # Reduce false mouldy detections
    "defective": 1.0,    # Neutral
}

print(f"\n✓ Per-class confidence thresholds:")
for cls, thr in CLASS_CONF_THRESHOLD.items():
    print(f"    {cls}: {thr}")

print(f"\n✓ Class bias multipliers:")
for cls, bias in CLASS_BIAS.items():
    print(f"    {cls}: x{bias}")

# =================================================================
# INITIALIZE MVS CAMERA
# =================================================================

deviceList = MV_CC_DEVICE_INFO_LIST()
tlayerType = MV_GIGE_DEVICE | MV_USB_DEVICE

print("\nSearching for cameras...")
ret = MvCamera.MV_CC_EnumDevices(tlayerType, deviceList)
if ret != 0:
    print(f"✗ Enumeration failed! Error code: 0x{ret:X}")
    exit(1)
if deviceList.nDeviceNum == 0:
    print("✗ No cameras found!")
    exit(1)
print(f"✓ Found {deviceList.nDeviceNum} camera(s)")

stDeviceList = cast(
    deviceList.pDeviceInfo[0], POINTER(MV_CC_DEVICE_INFO)
).contents
cam = MvCamera()

ret = cam.MV_CC_CreateHandle(stDeviceList)
if ret != 0:
    print(f"✗ Create handle failed! Error: 0x{ret:X}")
    exit(1)

ret = cam.MV_CC_OpenDevice(MV_ACCESS_Exclusive, 0)
if ret != 0:
    print(f"✗ Open device failed! Error: 0x{ret:X}")
    exit(1)

print("\n--- Configuring camera ---")
try:
    ret = cam.MV_CC_SetEnumValue("PixelFormat", PixelType_Gvsp_RGB8_Packed)
    if ret == 0: print("✓ Pixel Format set to RGB8")
except: pass
try:
    ret = cam.MV_CC_SetEnumValue("ExposureAuto", 0)
    if ret == 0: print("✓ Auto exposure turned OFF")
except: pass
try:
    ret = cam.MV_CC_SetFloatValue("ExposureTime", 5000.0)
    if ret == 0: print("✓ Exposure time set to 5000µs")
except: pass

print("\nStarting capture...")
stParam = MVCC_INTVALUE()
memset(byref(stParam), 0, sizeof(MVCC_INTVALUE))
ret = cam.MV_CC_GetIntValue("PayloadSize", stParam)
if ret != 0:
    print("✗ Get payload size failed!")
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    exit(1)

nPayloadSize = stParam.nCurValue
data_buf     = (c_ubyte * nPayloadSize)()

ret = cam.MV_CC_StartGrabbing()
if ret != 0:
    print("✗ Start grabbing failed!")
    cam.MV_CC_CloseDevice()
    cam.MV_CC_DestroyHandle()
    exit(1)

print("✓ Camera ready!")

# =================================================================
# GUI APPLICATION
# =================================================================

class BeanInspectionGUI:
    def __init__(self, root, model, cam, nPayloadSize, data_buf):
        self.root         = root
        self.model        = model
        self.cam          = cam
        self.nPayloadSize = nPayloadSize
        self.data_buf     = data_buf

        self.root.title("Bean Quality Inspection System - Task 2")
        self.root.geometry("1400x850")
        self.root.configure(bg='#2b2b2b')

        self.running       = True
        self.paused        = False
        self.count         = 0
        self.exposure_time = 5000.0
        self.counts        = {name: 0 for name in model.names.values()}
        self.last_img      = None   # for saving screenshots

        self.colours = {
            "good":      (0,   255,   0),
            "cracked":   (128,   0, 200),
            "broken":    (0,     0, 255),
            "mouldy":    (0,   200, 255),
            "defective": (0,   255, 255),
        }

        self.setup_gui()
        self.update_video()

    def setup_gui(self):
        main_frame = tk.Frame(self.root, bg='#2b2b2b')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # LEFT PANEL — video feed
        left_panel = tk.Frame(main_frame, bg='#1a1a1a')
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.video_label = tk.Label(left_panel, bg='#1a1a1a')
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        controls = tk.Frame(left_panel, bg='#2b2b2b')
        controls.pack(fill=tk.X, padx=5, pady=5)

        tk.Button(controls, text="⏸ Pause",
                  command=self.toggle_pause,
                  bg='#444', fg='white', padx=10).pack(side=tk.LEFT, padx=2)
        tk.Button(controls, text="🔄 Reset",
                  command=self.reset_counts,
                  bg='#444', fg='white', padx=10).pack(side=tk.LEFT, padx=2)
        tk.Button(controls, text="💾 Save",
                  command=self.save_screenshot,
                  bg='#444', fg='white', padx=10).pack(side=tk.LEFT, padx=2)
        tk.Button(controls, text="📊 Export CSV",
                  command=self.export_csv,
                  bg='#007700', fg='white', padx=10).pack(side=tk.LEFT, padx=2)
        tk.Button(controls, text="+ Exposure",
                  command=lambda: self.adj_exposure(500),
                  bg='#333', fg='white', padx=8).pack(side=tk.LEFT, padx=2)
        tk.Button(controls, text="- Exposure",
                  command=lambda: self.adj_exposure(-500),
                  bg='#333', fg='white', padx=8).pack(side=tk.LEFT, padx=2)
        tk.Button(controls, text="✗ Quit",
                  command=self.quit_app,
                  bg='#880000', fg='white', padx=10).pack(side=tk.RIGHT, padx=2)

        # RIGHT PANEL — charts and table
        right_panel = tk.Frame(main_frame, bg='#2b2b2b', width=450)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))

        tk.Label(right_panel, text="BEAN INSPECTION RESULTS",
                 font=('Arial', 16, 'bold'),
                 bg='#2b2b2b', fg='#ffaa00').pack(pady=10)
        tk.Label(right_panel, text="Task 2 — YOLOv8 + MVS Camera",
                 font=('Arial', 9),
                 bg='#2b2b2b', fg='#666666').pack(pady=(0, 8))

        # Bar chart
        chart_frame = tk.LabelFrame(right_panel, text="Bar Chart - Counts",
                                    bg='#2b2b2b', fg='white')
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.fig_bar = Figure(figsize=(5, 3), dpi=80, facecolor='#2b2b2b')
        self.ax_bar  = self.fig_bar.add_subplot(111)
        self.ax_bar.set_facecolor('#3b3b3b')
        self.ax_bar.tick_params(colors='white')
        self.canvas_bar = FigureCanvasTkAgg(self.fig_bar, master=chart_frame)
        self.canvas_bar.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Pie chart
        pie_frame = tk.LabelFrame(right_panel, text="Pie Chart - Distribution",
                                  bg='#2b2b2b', fg='white')
        pie_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.fig_pie = Figure(figsize=(4, 3), dpi=80, facecolor='#2b2b2b')
        self.ax_pie  = self.fig_pie.add_subplot(111)
        self.ax_pie.set_facecolor('#2b2b2b')
        self.canvas_pie = FigureCanvasTkAgg(self.fig_pie, master=pie_frame)
        self.canvas_pie.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Summary table
        table_frame = tk.LabelFrame(right_panel, text="Summary",
                                    bg='#2b2b2b', fg='white')
        table_frame.pack(fill=tk.X, padx=10, pady=5)
        self.tree = ttk.Treeview(table_frame,
                                 columns=('Class', 'Count', 'Pct'),
                                 show='headings', height=5)
        self.tree.heading('Class', text='Bean Class')
        self.tree.heading('Count', text='Count')
        self.tree.heading('Pct',   text='Percentage')
        self.tree.pack(fill=tk.X, padx=5, pady=5)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        tk.Label(self.root, textvariable=self.status_var,
                 bg='#1a1a1a', fg='#888',
                 anchor=tk.W, padx=10).pack(side=tk.BOTTOM, fill=tk.X)

    def adj_exposure(self, delta):
        self.exposure_time = max(500, min(20000,
                                          self.exposure_time + delta))
        self.cam.MV_CC_SetFloatValue("ExposureTime", self.exposure_time)
        self.status_var.set(f"Exposure: {self.exposure_time:.0f}µs")

    def update_video(self):
        if not self.running:
            return

        if not self.paused:
            stFrameInfo = MV_FRAME_OUT_INFO_EX()
            memset(byref(stFrameInfo), 0, sizeof(stFrameInfo))

            ret = self.cam.MV_CC_GetOneFrameTimeout(
                self.data_buf, self.nPayloadSize, stFrameInfo, 1000
            )

            if ret == 0:
                width  = stFrameInfo.nWidth
                height = stFrameInfo.nHeight
                frame_data = np.frombuffer(
                    self.data_buf, dtype=np.uint8,
                    count=width * height * 3
                )
                img = frame_data.reshape((height, width, 3))
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                # Normalise lighting before YOLO inference
                img = apply_clahe(img)
                results = self.model(img, conf=0.20, iou=0.45,
                                     verbose=False)

                frame_counts = {name: 0
                                for name in self.model.names.values()}

                if results[0].boxes is not None:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        orig_conf = float(box.conf[0])
                        cls  = int(box.cls[0])
                        name = self.model.names[cls]

                        # Apply bias multiplier
                        bias = CLASS_BIAS.get(name, 1.0)
                        adjusted_conf = min(0.95, orig_conf * bias)

                        # Per-class threshold check (using adjusted confidence)
                        threshold = CLASS_CONF_THRESHOLD.get(name, 0.40)
                        if adjusted_conf < threshold:
                            continue

                        frame_counts[name] += 1
                        self.counts[name]  += 1
                        colour = self.colours.get(name, (255, 255, 255))

                        cv2.rectangle(img, (x1, y1), (x2, y2), colour, 2)
                        cv2.putText(img, f"{name} {adjusted_conf:.0%}",
                                    (x1, max(y1-5, 15)),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.5, colour, 2)

                cv2.putText(img, f"Frame: {self.count}",
                            (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)
                cv2.putText(img, f"Good bias: x{CLASS_BIAS['good']}  |  Mouldy bias: x{CLASS_BIAS['mouldy']}",
                            (10, 55),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.45, (255, 255, 0), 1)

                self.last_img = img.copy()   # save for screenshot

                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk  = ImageTk.PhotoImage(img_pil)
                self.video_label.config(image=img_tk)
                self.video_label.image = img_tk

                self.count += 1
                if self.count % 10 == 0:
                    self.update_charts()

        self.root.after(50, self.update_video)

    def update_charts(self):
        names  = list(self.counts.keys())
        values = list(self.counts.values())
        colors = ['green', 'purple', 'red', 'orange', 'cyan']
        total  = sum(values)

        self.ax_bar.clear()
        self.ax_bar.set_facecolor('#3b3b3b')
        bars = self.ax_bar.bar(names, values, color=colors)
        self.ax_bar.set_xlabel('Bean Type', color='white')
        self.ax_bar.set_ylabel('Count',     color='white')
        self.ax_bar.set_title('Total Counts', color='#ffaa00')
        self.ax_bar.tick_params(colors='white')
        for bar, val in zip(bars, values):
            if val > 0:
                self.ax_bar.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    str(val), ha='center', va='bottom', color='white'
                )
        self.canvas_bar.draw()

        self.ax_pie.clear()
        if total > 0:
            non_zero = [(n, v) for n, v in zip(names, values) if v > 0]
            if non_zero:
                nz_names  = [n for n, _ in non_zero]
                nz_vals   = [v for _, v in non_zero]
                nz_colors = [colors[names.index(n)] for n in nz_names]
                self.ax_pie.pie(nz_vals, labels=nz_names,
                                autopct='%1.1f%%', colors=nz_colors,
                                textprops={'color': 'white'})
                self.ax_pie.set_title('Distribution', color='#ffaa00')
        self.canvas_pie.draw()

        for item in self.tree.get_children():
            self.tree.delete(item)
        for name, val in self.counts.items():
            pct = (val / total * 100) if total > 0 else 0
            self.tree.insert('', tk.END,
                             values=(name.capitalize(), val, f"{pct:.1f}%"))

        self.status_var.set(
            f"Total: {total} beans  |  Frame: {self.count}  |"
            f"  Exposure: {self.exposure_time:.0f}µs"
        )

    def toggle_pause(self):
        self.paused = not self.paused
        self.status_var.set("Paused" if self.paused else "Running")

    def reset_counts(self):
        self.counts = {name: 0 for name in self.model.names.values()}
        self.update_charts()
        self.status_var.set("Counters reset")

    def save_screenshot(self):
        if self.last_img is not None:
            filename = (f"screenshot_"
                        f"{datetime.datetime.now().strftime('%H%M%S')}.png")
            cv2.imwrite(filename, self.last_img)
            self.status_var.set(f"Saved: {filename}")

    def export_csv(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile="bean_data.csv"
        )
        if filename:
            total = sum(self.counts.values())
            df = pd.DataFrame([
                {"Class": k, "Count": v,
                 "Percentage": (v / total * 100) if total > 0 else 0}
                for k, v in self.counts.items()
            ])
            df.to_csv(filename, index=False)
            self.status_var.set(f"Exported to {filename}")

    def quit_app(self):
        self.running = False
        self.root.quit()
        self.root.destroy()


# =================================================================
# START GUI
# =================================================================

root = tk.Tk()
app  = BeanInspectionGUI(root, model, cam, nPayloadSize, data_buf)

print("\n" + "=" * 60)
print("  GUI STARTED")
print(f"  Class bias: Good x{CLASS_BIAS['good']}, Mouldy x{CLASS_BIAS['mouldy']}")
print("  Controls in GUI window")
print("  Close window to quit")
print("=" * 60)

root.mainloop()

# =================================================================
# CLEANUP
# =================================================================

print("\nCleaning up...")
cam.MV_CC_StopGrabbing()
cam.MV_CC_CloseDevice()
cam.MV_CC_DestroyHandle()
cv2.destroyAllWindows()
print("✓ Done")