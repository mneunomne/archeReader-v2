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
  
# merge lines that are very close to one another
max_dist = 10
def mergeLines(lines, max_dist=10):
    merged_lines = []
    i = 0
    while i < len(lines):
        x1, y1, x2, y2 = lines[i][0]
        j = i + 1
        while j < len(lines):
            x3, y3, x4, y4 = lines[j][0]
            if abs(x1 - x3) < max_dist and abs(y1 - y3) < max_dist:
                # Merge the two lines
                lines[i][0] = [(x1 + x3) / 2, (y1 + y3) / 2, (x2 + x4) / 2, (y2 + y4) / 2]
                # Remove line j
                lines = np.delete(lines, j, axis=0)
                # Decrement j as the array length has decreased by 1
                j -= 1
            j += 1
        merged_lines.append(lines[i])
        i += 1
    return merged_lines



def find_squares(lines, threshold=10):
    squares = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            for k in range(j + 1, len(lines)):
                for l in range(k + 1, len(lines)):
                    # Check for intersections
                    intersection_points = []
                    for p1 in lines[i]:
                        for p2 in lines[j]:
                            intersection = intersection_point(p1, p2)
                            if intersection is not None:
                                intersection_points.append(intersection)
                    for p1 in lines[k]:
                        for p2 in lines[l]:
                            intersection = intersection_point(p1, p2)
                            if intersection is not None:
                                intersection_points.append(intersection)
                    
                    # Check if intersection points form a square
                    if len(intersection_points) == 4:
                        distances = [
                            np.linalg.norm(intersection_points[0] - intersection_points[1]),
                            np.linalg.norm(intersection_points[0] - intersection_points[2]),
                            np.linalg.norm(intersection_points[0] - intersection_points[3])
                        ]
                        if all(abs(d1 - d2) < threshold for d1 in distances for d2 in distances):
                            squares.append(intersection_points)
    
    return squares

def intersection_point(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    det = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if det != 0:
        intersect_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / det
        intersect_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / det
        return np.array([intersect_x, intersect_y])
    return None

def getLongestHorizontalLine(lines):
    longest = 0
    longest_line = None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(x1 - x2) > longest:
            longest = abs(x1 - x2)
            longest_line = line
    return longest_line

def makeGridFromHorizontalLine(line, gap=10, img_shape=(200, 200)):
    x1, y1, x2, y2 = line[0]
    lines = []
    rows = int(img_shape[1] / gap)
    for i in range(-rows, rows):
        _y1 = y1 + i * gap
        _y2 = y2 + i * gap
        lines.append([[x1, _y1, x2, _y2]])
    return lines

def getLongestVerticalLine(lines):
    longest = 0
    longest_line = None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if abs(y1 - y2) > longest:
            longest = abs(y1 - y2)
            longest_line = line
    return longest_line

def makeGridFromVerticalLine(line, gap=10, img_shape=(200, 200)):
    x1, y1, x2, y2 = line[0]
    lines = []
    cols = int(img_shape[0] / gap)
    for i in range(-cols, cols):
        _x1 = x1 + i * gap
        _x2 = x2 + i * gap
        lines.append([[_x1, y1, _x2, y2]])
    return lines


def find_intersection(line1, line2):
    x1, y1, x2, y2 = line1[0]
    x3, y3, x4, y4 = line2[0]
    
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if denominator == 0:
        return None  # Lines are parallel or coincident
    else:
        intersection_x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
        intersection_y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator
        return int(intersection_x), int(intersection_y)

def getRectsFromIntersectionPoints(horizontal_lines, vertical_lines):
    rectangles = []

    for h_line in horizontal_lines:
        for v_line in vertical_lines:
            x = v_line[0]
            y = h_line[1]
            width = v_line[2] - v_line[0]
            height = h_line[3] - h_line[1]

            rect = (x, y, width, height)
            rectangles.append(rect)

    return rectangles

def arrange_points_clockwise(points):
    center = np.mean(points, axis=0)
    return sorted(points, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))

def create_polygons_and_rectangles(intersection_points):
    polygons = []
    rectangles = []

    # Rearrange points in correct order and generate polygons and rectangles
    for i in range(0, len(intersection_points), 4):
        polygon_points = arrange_points_clockwise(intersection_points[i:i+4])
        polygons.append(polygon_points)

    return polygons, rectangles
