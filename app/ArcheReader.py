import cv2
from globals import *
from image_processing import *
import numpy as np

import threading

from utils import list_ports

class ArcheReader:
  
  capture = None
  
  # default values for canny edge detection and hough lines
  threshold1 = 10
  threshold2 = 19
  minLineLength = 40
  maxLineGap = 75
  set_update= True
  
  crop_size = 200
  
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
    self.create_gui()
    while True:
      # display image
      if self.set_update:
        image = self.get_image()
        image = self.process_image(image)
        cv2.imshow('frame', image)
        self.set_update = False
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
    self.set_update = True
    self.threshold1 = value

  def on_change_threshold2(self, value):
    self.set_update = True
    self.threshold2 = value

  def on_change_minLineLength(self, value):
    self.set_update = True
    self.minLineLength = value

  def on_change_maxLineGap(self, value):
    self.set_update = True
    self.maxLineGap = value

  def process_image(self, raw_image):
    # otsu thresholding
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    image = raw_image.copy()
    
    cropped = getCroppedImage(image, self.crop_size)
    
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
      lines = remove_too_close_lines(lines)
      # squares = find_squares(lines)
      # draw_squares(img_out, find_intersections(lines), 50)
      
      horizontal_lines = filter_close_lines(filter_lines_by_angle(lines, 0, 5), 10)
      vertical_lines = filter_close_lines(filter_lines_by_angle(lines, 90, 5), 10)
      
      if (len(vertical_lines) > 0):
        vertical_lines = fill_vertical_lines(vertical_lines, self.crop_size)
      
      if (len(horizontal_lines) > 0):
        horizontal_lines = fill_horizontal_lines(horizontal_lines, self.crop_size)
      
      # intersections = find_intersections2(horizontal_lines, vertical_lines)
      
      # smallest_distance = getSmallestDistanceBetweenPoints(intersections)
      
      # print("smallest_distance", smallest_distance)
      #for point in intersections:
      #    cv2.circle(img_out, point, 2, (0, 0, 255), -1)
      
      lines = horizontal_lines + vertical_lines
      
      for line in lines:
        x1, y1, x2, y2 = line[0]
        print("line", line)
        cv2.line(img_out, (x1, y1), (x2, y2), (0, 255, 0), 1)
      
      
      if len(horizontal_lines) > 0 and len(vertical_lines)> 0:
        # find smallest distance between horizontal lines
        # Find intersection points
        intersection_points = []
        for h_line in horizontal_lines:
            for v_line in vertical_lines:
                intersection = find_intersection(h_line, v_line)
                if intersection:
                    intersection_points.append(intersection)
        # Draw intersection points
        for point in intersection_points:
          cv2.circle(img_out, point, 2, (0, 0, 255), -1)
        
        # make 4 sided polygon from each 4 neighboring intersection points
        #polygons = create_rectangles(intersection_points)
        # draw polygons
        #for polygon in polygons:
        #  print("polygon", polygon)
        #  cv2.polylines(img_out, [polygon], True, (0, 255, 255), 1)

        

        
    #blur = cv2.medianBlur(gray, 9)
    #sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    #sharpen = cv2.filter2D(blur, -1, sharpen_kernel)

    # Threshold and morph close
    #thresh = cv2.threshold(sharpen, 150, 190, cv2.THRESH_BINARY_INV)[1]
    #kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    #close = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # find squares
    #cnts = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    
    
    # process image
    return img_out