Deployment notes: OpenCV libGL issue

Problem

OpenCV may fail to import with "ImportError: libGL.so.1: cannot open shared object file" when GUI/OpenGL libraries are missing on the host.

Quick fixes

- Current Debian/Ubuntu: sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0t64
- Streamlit Cloud: packages.txt contains libgl1, libglib2.0-0t64, and the X runtime libraries required by OpenCV.
- If only headless OpenCV is required, ensure opencv-python-headless is installed and uninstall opencv-python: pip uninstall -y opencv-python && pip install --no-deps opencv-python-headless

After installing system packages, redeploy the app.

Notes on Debian versions:
- If libgl1-mesa-glx is not available on your host, use 'libgl1' instead. Some distributions (Debian trixie) provide libGL via 'libgl1'.
- Extra X libraries that sometimes help: libsm6, libxrender1, libxext6.
