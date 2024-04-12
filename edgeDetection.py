import cv2
import numpy as np

def find_squares(image_path):
    # Load the image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur, useful for removing noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edged = cv2.Canny(gray, 30, 100)

    cv2.imshow('Edged', edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # # Find contours
    # contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # squares = []
    # for cnt in contours:
    #     # Approximate the contour to a polygon
    #     approx = cv2.approxPolyDP(cnt, 0.02 * cv2.arcLength(cnt, True), True)

    #     # Check if the polygon has 4 vertices and is convex
    #     if len(approx) == 4 and cv2.isContourConvex(approx):
    #         # Compute the bounding box of the contour and use it to calculate aspect ratio
    #         x, y, w, h = cv2.boundingRect(approx)
    #         aspect_ratio = float(w) / h
    #         if 0.95 <= aspect_ratio <= 1.05:  # Aspect ratio close to 1
    #             squares.append(approx)

    # # Draw squares on the image
    # cv2.drawContours(image, squares, -1, (0, 255, 0), 3)
    # cv2.imshow('Squares', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

# Example usage
find_squares('image.png')