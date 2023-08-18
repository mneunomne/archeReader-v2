import cv2 
import cv2.aruco as aruco
import numpy as np

def find_intersections(lines):
  intersections = []
  for i in range(len(lines)):
    for j in range(i + 1, len(lines)):
      x1, y1, x2, y2 = lines[i][0]
      x3, y3, x4, y4 = lines[j][0]
        
      det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
      if det != 0:  # Checking if lines are not parallel
        intersection_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
        intersection_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det
        intersections.append((int(intersection_x), int(intersection_y)))
  return intersections

def draw_squares(image, intersections, side_length):
  for intersection in intersections:
    x, y = intersection
    half_side = side_length // 2
    cv2.rectangle(image, (x - half_side, y - half_side), (x + half_side, y + half_side), (0, 255, 0), 2)

# gray image expected
def findHoughPLine(image, threshold1, threshold2, minLineLength, maxLineGap):
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  edges = cv2.Canny(gray, threshold1, threshold2)
  lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=minLineLength, maxLineGap=maxLineGap)
  return lines, edges

def getCroppedImage(image, size):
  height, width, ch = image.shape
  n = int(size / 2)
  cropped_image = image[int(height/2)-n:int(height/2)+n, int(width/2)-n:int(width/2)+n, :]
  return cropped_image
  