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
CARD_WIDTH = 300;
MIN_ASPECT_RATIO = 2.5;
MAX_ASPECT_RATIO = 4.0;
MIN_BLOCK_LENGTH = 40;
MAX_BLOCK_LENGTH = 55;
MIN_BLOCK_HEIGHT = 10;
MAX_BLOCK_HEIGHT = 20;
GROUP_BUFFER = 5;
FIRST_DIGIT_TO_PROVIDER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
};

class CommandLineParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser();
	self.add_arguments();
    
    def add_arguments(self):
        self.parser.add_argument("-i", "--image", required=True, help="enter path to image");
        self.parser.add_argument("-r", "--reference", required=True, help="enter path to reference image");
        
    def parse(self):
        parsed = vars(self.parser.parse_args());
        return parsed;

class OCRReader:
    def __init(self, img_path):
        self.image = cv2.imread(img_path);
        
    def convert_image_to_grayscale(self):
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY);
    
    def threshold_image(self, image):
        self.image = cv2.threshold(self.image, BLACK_WHITE_THRESHOLD, MAX_BINARY_THRESH, cv2.THRESH_BINARY_INV);
    
    def set_contours(self):
        # Only store non-redundant contour coordinates
        contours_obj = cv2.findContours(self.image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
        self.contours = sort_contours(self.parse_contours_obj(contour_obj));
	                              
    def sort_contours(self, contours):
        return contours.sort_contours(contours, method="left-to-right")[0];
	
    def parse_contours_obj(self, contour_obj):
        return imutil.grab_contours(contour_obj);    
    
    def bound_box(self, contour):
        (x, y, w, h) = cv2.boundingRect(countour);
        region = self.image[y: y + h, x: x + w];
        resized_region = cv2.resize(region, (DIGIT_LENGTH, DIGIT_HEIGHT));
        return resized_region;    
        
class ReferenceDigitsReader(OCRReader):
    def __init__(self, ref_path):
        super(ref_path);
        self.digit_to_region = {};
    
    def process_reference_image(self):
        self.convert_image_to_gray();
        self.threshold_image();
	self.set_contours();
    
    def scan_reference_image(self):
        for (digit, contour) in enumerate(self.contours):
            region = self.bound_box(contour);
            self.digit_to_region[digit] = region;
        
    def get_digit_to_region(self):
        return self.digit_to_region;
    
class CardReader(OCRReader):
    def __init__(self, card_path, digits_to_locs):
        super(card_path);
        self.groups = [];
	self.digits_to_locs = digits_to_locs;
    
    def create_rectangle_kernel(self):
        self.rectangle_kernel = cv2.getStructuringELement(cv2.MORPH_RECT, (9, 3));
        self.square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5));
        
    def create_kernels(self):
        # We use kernels to do image convolution operations
        self.create_rectangle_kernel();
        self.create_square_kernel();
    
    def threshold_image(self):
        threshold = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1];      
        self.image = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, self.square_kernel);
        
    def resize_image(self, ref):
        self.image = imutils.resize(self.image, CARD_WIDTH);
        
    def process_card_image(self):
        self.resize_image();
	self.convert_image_to_grayscale();
	self.gray = self.image;
	self.set_gradient();
	self.threshold_image();
	self.set_contours();
	
    def scan_card_image(self):
	self.get_digits();
	return self.get_output();
	
    def get_output(self):
	output = [];
	for (i, (gX, gY, gW, gH)) in enumerate(self.digits_to_locs):
	    group = [];
	    region = self.extract_region(gX, gY, gW, gH);
	    contours_of_region = self.extract_contours(region);
	    scores = [];
	    for contour in contours_in_region:
		digit = self.bound_box(contour);
		scores = self.template_match();
		group.append(str(np.argmax(scores)));
	    output.extend(group);
    
    def template_match(self):
	scores = [];
	for (digit, digit_region) in self.digit_to_locs.items():
	    res = cv2.matchTemplate(box, digit_region, cv2.TM_CCOEFF);
	    (_, score, _, _) = cv2.minMaxLoc(res);
	    scores.append(score);
	    
    def extract_region(self, gX, gY, gW, gH):
	# Add a buffer so that the whole "group" of four is extracted 
	region = self.gray[gY - GROUP_BUFFER: gY + gH + GROUP_BUFFER, gX - GROUP_BUFFER: gX + gW + GROUP_BUFFER];
	return cv2.threshold(region, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1];
    
    def extract_contours(self, region):
	contour_obj = cv2.findContours(region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE);
        contours = self.parse_contours_obj(contour_obj);
	return contours.sort_contours(digitCnts, method="left-to-right")[0];

    def get_digits(self, contour):
	for (index, contour) in enumerate(self.image):
	    (x, y, w, h) = cv2.boundingRect(c)
	    aspect_ratio = w / float(h);     
	    if (aspect_ratio_compatible(aspect_ratio) and height_width_compatible(w,  h)):
		self.groups.append((x, y, w, h));	    
	self.groups = sorted(self.groups, key=lambda x:x[0]);
	
    def height_width_compatible(self, width, height):
	height_okay = height < MAX_BLOCK_HEIGHT and height > MIN_BLOCK_HEIGHT;
	width_okay = width < MAX_BLOCK_WIDTH and width > MAX_BLOCK_WIDTH;
	return height_okay and width_okay;
    
    def aspect_ratio_compatible(self, aspect_ratio):
        return (aspect_ratio > MIN_ASPECT_RATIO and aspect_ratio < MAX_ASPECT_RATIO);
    
    def set_gradient(self):
        white_against_dark = self.tophat_transform();
        raw_x_gradient = np.absolute(
            cv2.Sobel(white_against_dark, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1));
        normalized_gradiant = self.normalize_min_max(x_gradient).astype("uint8");
        self.image = cv2.morphologyEx(normalized, cv2.MORPH_CLOSE, self.rectangle_kernel);
    
    def normalize_min_max(self, x_gradient):
        (minVal, maxVal) = (np.min(x_gradient), np.max(x_gradient))
        x_gradient = (255 * ((x_gradient - minVal) / (maxVal - minVal)));        
        return x_gradient;
        
    def tophat_transform(self):
        return cv2.morphologyEx(self.image, cv2.MORPH_TOPHAT, self.rectangle_kernel);

# Demo

cl_parser = CommandLineParser();
parsed = cl_parser.parse();

ref_digits_reader = ReferenceDigitsReader(parsed["reference"]);
ref_digits_reader.process_reference_image();
digits = ref_digits_reader.get_digit_to_region();

card_reader = CardReader(parsed["image"], digits);
card_reader.process_card_image();
result = card_reader.scan_scard_image();
print (result);

