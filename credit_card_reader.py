import cv2
import imutils
import argparse
import numpy as np
from imutils import contours
from matplotlib import pyplot as plt

# Global constants
BLACK_WHITE_THRESHOLD = 10;
MAX_BINARY_THRESH = 255;
DIGIT_HEIGHT = 57;
DIGIT_LENGTH = 88;
CARD_WIDTH = 
FIRST_DIGIT_TO_PROVIDER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
};

class CommandLineParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser();
    
    def add_arguments():
        self.parser.add_argument("-i", "--image", required=True, help="path to image");
        self.add_argument("-r", "--reference", required=True, help="path to reference OCR-A image");  
        
    def parse():
        parsed = vars(self._parser.parse_args());
        return parsed;

class OCRReader:
    def __init(self, img_path):
        self.image = cv2.imread(img_path);
        
    def convert_image_to_grayscale(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY);
    
    def threshold_grayscale(self, image):
        self.image = cv2.threshold(self.image, BLACK_WHITE_THRESHOLD, MAX_BINARY_THRESH, cv2.THRESH_BINARY_INV);
        
class ReferenceDigitsReader(OCRReader):
    def __init__(self, ref_path):
        super(ref_path);
        self.digit_to_region = {};
    
    def process_image(self):
        self.convert_image_to_gray();
        self.threshold_image();    
        
    def process_contours(self):
        self.set_contours();
        self.sort_contours();
    
    def create_bounding_boxes(self):
        for (digit, contour) in enumerate(self.contours):
            region = self.bound_box(contour);
            self.digit_to_region[digit] = region;
            
    def bound_box(self, contour):
        (x, y, w, h) = cv2.boundingRect(countour);
        region = self.image[y: y + h, x: x + w];
        resized_region = cv2.resize(region, (DIGIT_LENGTH, DIGIT_HEIGHT));
        return resized_region;
    
    def set_contours(self):
        # Only store non-redundant contour coordinates
        contours_obj = cv2.findContours(self.image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
        self.contours = parse_contours_obj(contour_obj);
    
    def parse_contours_obj(self, contour_obj):
        return imutil.grab_contours(contour_obj);
    
    def sort_contours(self):
        self.contours = contours.sort_contours(self.contours, method="left-to-right")[0];
        
    def get_digit_to_region(self):
        return self.digit_to_region;
    
class CardReader(OCRReader):
    def __init__(self, card_path):
        super(card_path);
        
    def process_image(self):
        self.image = imutils.resize(self.image, CARD_WIDTH);