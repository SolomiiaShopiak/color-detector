## Color detector
A python project that detects and tracks areas with red, blue and green colors in real-time.
### Steps for color detection:
- loading the image
- converting it to HSV color space
- defining HSV color ranges
- creating corresponding masks
- applying opening for noise reduction
- finding contours of masks and bounding them with boxes

## Dependencies
```
opencv-python
numpy
```

## Usage
Run in the command line:
```
python3 color_detector.py
```
Python iterpreter:
```python3
import color_detector
color_detector.detect_color()
```
## Examples
<img width="947" alt="1" src="https://github.com/user-attachments/assets/5b3fe63e-92e4-414a-885c-539ecb4a2f8f" />
