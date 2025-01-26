import cv2
import numpy as np

#defining color ranges in hsv color space for masks
red_lower = np.array([136, 87, 111], np.uint8)
red_upper = np.array([180, 255, 255], np.uint8)

green_lower = np.array([25, 52, 72], np.uint8)
green_upper = np.array([102, 255, 255], np.uint8)

blue_lower = np.array([94, 80, 2], np.uint8)
blue_upper = np.array([120, 255, 255], np.uint8)

# defining kernel for noise reduction morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def track_contours(contours, frame, bbox_color: tuple, text: str) -> None:
    '''bound each contour with a box'''
    
    for contour in contours:
        if (cv2.contourArea(contour) > 700):
            x, y, w, h = cv2.boundingRect(contour)
            frame = cv2.rectangle(frame, (x, y), (x + w, y + h), bbox_color, 2)
            cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, bbox_color, 2)
    

def detect_color() -> None:
    webcam = cv2.VideoCapture(0)

    while True:
        _, frame = webcam.read()

        #convert image from BGR to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # defining masks for different colors,
        # then applying opening (erosion followed by dilation) to remove noise
        red_mask = cv2.inRange(hsv_frame, red_lower, red_upper)
        red_mask = cv2.dilate(red_mask, kernel)
        
        green_mask = cv2.inRange(hsv_frame, green_lower, green_upper)
        green_mask = cv2.dilate(green_mask, kernel)
        
        blue_mask = cv2.inRange(hsv_frame, blue_lower, blue_upper)
        blue_mask = cv2.dilate(blue_mask, kernel)

        #retrieving contours of masks and bounding them with track_contours function
        contours, _ = cv2.findContours(red_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        track_contours(contours, frame, (0, 0, 255), "Red")

        contours, _ = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        track_contours(contours, frame, (0, 255, 0), "Green")
        
        contours, _ = cv2.findContours(blue_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        track_contours(contours, frame, (255,0,0), "Blue")
        
        cv2.imshow('Color Detection', frame)
        
        if cv2.waitKey(1)==27:
            break

    webcam.release()
    cv2.destroyAllWindows()
    

if __name__ == '__main__':
    detect_color()
