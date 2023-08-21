import cv2
from globals import *
from image_processing import *
import numpy as np

import threading

from utils import list_ports

class ArcheReader:
  
  capture = None
  
  # default values for canny edge detection and hough lines
  threshold1 = 41
  threshold2 = 60
  minLineLength = 50
  maxLineGap = 100
  
  def __init__(self, test):
    self.test = test
    # print("gui", gui)
    self.init()
  
  
  def init(self):
    # if test enabled, use static image
    self.run()
  
  def start_cam(self):
    # detect available cameras
    _,working_ports,_ = list_ports()
    print("working_ports", working_ports)
  
  def get_image(self):
    # if test enabled, use static image
    if self.test:
      return "test.jpg"
    # get current frame from webcam
    if self.capture == None:
      self.start_cam()
      self.capture = cv2.VideoCapture(WEBCAM)
    if self.capture.isOpened():
      #do something
      ret, frame = self.capture.read()
      return frame
    else:
      print("Cannot open camera")
    
  def run(self):
    while True:
      # display image
      self.create_gui()
      image = self.get_image()
      image = self.process_image(image)
      cv2.imshow('frame', image)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # When everything done, release the capture
    self.capture.release()
    cv2.destroyAllWindows()
  
  def create_gui(self):
    cv2.namedWindow('controls')
    cv2.createTrackbar('threshold1', 'controls', self.threshold1, 255, self.on_change_threshold1)
    cv2.createTrackbar('threshold2', 'controls', self.threshold2, 255, self.on_change_threshold2)
    cv2.createTrackbar('minLineLength', 'controls', self.minLineLength, 1000, self.on_change_minLineLength)
    cv2.createTrackbar('maxLineGap', 'controls', self.maxLineGap, 1000, self.on_change_maxLineGap)
    cv2.imshow("controls", np.zeros((500,500,3), np.uint8))
  
  def on_change_threshold1(self, value):
    self.threshold1 = value

  def on_change_threshold2(self, value):
    self.threshold2 = value

  def on_change_minLineLength(self, value):
    self.minLineLength = value

  def on_change_maxLineGap(self, value):
    self.maxLineGap = value

  def process_image(self, raw_image):
    # otsu thresholding
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    image = raw_image.copy()
    
    cropped = getCroppedImage(image, 240)
    
    # image out will use the cropped image
    img_out = cropped.copy()

    # gray only from red channel
    gray = cropped[:,:,2]

    img_out = gray.copy()
    # make image colored
    img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)
    
    # Apply Gaussian blur to reduce noise and improve edge detection
    # blurred = cv2.GaussianBlur(cropped, (3, 3), 0)
    
    # decrease noise 
    img = cv2.fastNlMeansDenoising(cropped, None,10,7,21)
    
    # find lines
    lines, edges = findHoughPLine(img, self.threshold1, self.threshold2, self.minLineLength, self.maxLineGap)
    
    horizontal_lines = []
    vertical_lines = []
    
    if lines is not None:
      lines = mergeLines(lines)
      # squares = find_squares(lines)
      # draw_squares(img_out, find_intersections(lines), 50)
      for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_out, (x1, y1), (x2, y2), (0, 255, 0), 1)
      
      longest_horizonta_line = getLongestHorizontalLine(lines)  
      if longest_horizonta_line is not None:
        horizontal_lines = makeGridFromHorizontalLine(longest_horizonta_line, 13, gray.shape)
        if horizontal_lines is not None:
          for line in horizontal_lines:
            print(line)
            x1, y1, x2, y2 = line[0]
            cv2.line(img_out, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
          # draw longest_horizonta_line 
          x1, y1, x2, y2 = longest_horizonta_line[0]
          cv2.line(img_out, (x1, y1), (x2, y2), (0, 0, 255), 2)
      
      logest_vertical_line = getLongestVerticalLine(lines)
      if logest_vertical_line is not None:
        vertical_lines = makeGridFromVerticalLine(logest_vertical_line, 13, gray.shape)
        if vertical_lines is not None:
          for line in vertical_lines:
            print(line)
            x1, y1, x2, y2 = line[0]
            cv2.line(img_out, (x1, y1), (x2, y2), (255, 0, 0), 2)
        
          # draw logest_vertical_line 
          x1, y1, x2, y2 = logest_vertical_line[0]
          cv2.line(img_out, (x1, y1), (x2, y2), (0, 0, 255), 2)
      
      if len(horizontal_lines) > 0 and len(vertical_lines)> 0:
        # Find intersection points
        intersection_points = []
        for h_line in horizontal_lines:
            for v_line in vertical_lines:
                intersection = find_intersection(h_line, v_line)
                if intersection:
                    intersection_points.append(intersection)
        # Draw intersection points
        for point in intersection_points:
          cv2.circle(img_out, point, 3, (0, 0, 255), -1)
        

        
    blur = cv2.medianBlur(gray, 9)
    sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    # Threshold and morph close
    thresh = cv2.threshold(sharpen, 150, 190, cv2.THRESH_BINARY_INV)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    
    # process image
    return img_out