Deployment notes: OpenCV libGL issue

Problem

OpenCV may fail to import with "ImportError: libGL.so.1: cannot open shared object file" when GUI/OpenGL libraries are missing on the host.

Quick fixes

- Debian/Ubuntu: sudo apt-get update && sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
- Streamlit Cloud: packages.txt added to this repo (contains libgl1-mesa-glx and libglib2.0-0).
- If only headless OpenCV is required, ensure opencv-python-headless is installed and uninstall opencv-python: pip uninstall -y opencv-python && pip install --no-deps opencv-python-headless

After installing system packages, redeploy the app.
