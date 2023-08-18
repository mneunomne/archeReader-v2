import cv2
from globals import *
from utils import find_intersections, draw_squares
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
    # thread = threading.Thread(target=self.run)
    # thread.start()
    self.run()
    # self.gui.run()
  
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
      image = self.get_image()
      image = self.process_image(image)
      cv2.imshow('frame', image)
      self.create_gui()
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # When everything done, release the capture
    self.capture.release()
    cv2.destroyAllWindows()
  
  def create_gui(self):
    cv2.createTrackbar('threshold1', 'frame', self.threshold1, 255, self.on_change_threshold1)
    cv2.createTrackbar('threshold2', 'frame', self.threshold2, 255, self.on_change_threshold2)
    cv2.createTrackbar('minLineLength', 'frame', self.minLineLength, 1000, self.on_change_minLineLength)
    cv2.createTrackbar('maxLineGap', 'frame', self.maxLineGap, 1000, self.on_change_maxLineGap)
  
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
    
    n = int(600 / 2)
    # get center N, N square of image
    height, width, ch = image.shape
    
    img_out = raw_image.copy()
    # CROP IMG 0UT 
    img_out = img_out[int(height/2)-n:int(height/2)+n, int(width/2)-n:int(width/2)+n, :]
        
    # gray only from red channel
    gray = image[:,:,2]
    
     # get center N, N square of image
    height, width, ch = image.shape    
    cropped = gray[int(height/2)-n:int(height/2)+n, int(width/2)-n:int(width/2)+n]
    
    # decrease noise
    
    
    # gray = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise and improve edge detection
    # blurred = cv2.GaussianBlur(cropped, (7, 7), 0)
    
    # decrease noise 
    denoised_frame = cv2.fastNlMeansDenoising(cropped, None,10,7,21)
    
    # denoised_frame = cv2.fastNlMeansDenoising(cropped, None,10,7,21)

    edges = cv2.Canny(denoised_frame, threshold1=self.threshold1, threshold2=self.threshold2)
    
    img_out = edges.copy()
    # convert back to BGR
    img_out = cv2.cvtColor(img_out, cv2.COLOR_GRAY2BGR)
    
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=self.minLineLength, maxLineGap=self.maxLineGap)
    
    if lines is not None:
      # draw_squares(img_out, find_intersections(lines), 50)
      for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(img_out, (x1, y1), (x2, y2), (0, 255, 0), 1)

    # process image
    return img_out