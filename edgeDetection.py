import cv2
import numpy as np

def find_squares(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edged = cv2.Canny(gray, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Draw squares on the edged image
    for cnt in contours:
        epsilon = 0.02 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # Check if the polygon has 4 vertices, is convex and has reasonable area to be considered a square
        if len(approx) == 4 and cv2.isContourConvex(approx):
            area = cv2.contourArea(approx)
            if area > 1000:  # filter out very small squares
                x, y, w, h = cv2.boundingRect(approx)
                aspect_ratio = max(w, h) / min(w, h)
                if aspect_ratio < 1.2:  # more tolerant aspect ratio
                    cv2.drawContours(edged, [approx], -1, (255), 2)

    # Show edged image with squares highlighted
    cv2.imshow('Squares on Edged Image', edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
find_squares('square.png')