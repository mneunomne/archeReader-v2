import tkinter as tk
from tkinter import Scale, Button

threshold1_default = 41
threshold2_default = 60
minLineLength_default = 50
maxLineGap_default = 100

class GUI:
  def __init__(self):
    self.root = tk.Tk()
    self.root.title("Image Processing GUI")

    self.threshold1 = Scale(self.root, label="Canny Threshold 1", from_=0, to=255, orient="horizontal", command=self.updateValues)
    self.threshold1.set(threshold1_default)
    self.threshold1.pack()

    self.threshold2 = Scale(self.root, label="Canny Threshold 2", from_=0, to=255, orient="horizontal", command=self.updateValues)
    self.threshold2.set(threshold2_default)
    self.threshold2.pack()

    self.minLineLength = Scale(self.root, label="Min Line Length", from_=0, to=500, orient="horizontal", command=self.updateValues)
    self.minLineLength.set(minLineLength_default)
    self.minLineLength.pack()

    self.maxLineGap = Scale(self.root, label="Max Line Gap", from_=0, to=100, orient="horizontal", command=self.updateValues)
    self.maxLineGap.set(maxLineGap_default)
    self.maxLineGap.pack()
  
  def run(self):
    self.root.mainloop()
  
  def updateValues(self, event):
    print("threshold1", self.threshold1.get())