import cv2
import imutils
import argparse
import numpy as np
from imutils import contours
from matplotlib import pyplot as plt
import cc_2


# Global constants
BLACK_WHITE_THRESHOLD = 10
MAX_BINARY_THRESH = 255
DIGIT_LENGTH = 57
DIGIT_HEIGHT = 88
CARD_WIDTH = 300
MIN_ASPECT_RATIO = 2.5
MAX_ASPECT_RATIO = 4.0
MIN_BLOCK_WIDTH = 40
MAX_BLOCK_WIDTH = 55
MIN_BLOCK_HEIGHT = 10
MAX_BLOCK_HEIGHT = 20
GROUP_BUFFER = 5
FIRST_DIGIT_TO_PROVIDER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}


class CommandLineParser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.add_arguments()

    def add_arguments(self):
        self.parser.add_argument(
            "-i", "--image", required=True, help="enter path to image")
        self.parser.add_argument(
            "-r", "--reference", required=True, help="enter path to reference image")

    def parse(self):
        parsed = vars(self.parser.parse_args())
        return parsed


class OCRReader:
    def __init__(self, img_path):
        self.image = cv2.imread(img_path)

    def convert_image_to_grayscale(self):
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)


    def set_contours(self, sort=0):
        # Only store non-redundant contour coordinates
        contours_obj = cv2.findContours(
            self.threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = self.parse_contours_obj(contours_obj);
        if (sort):
            self.contours = self.sort_contours(self.contours);

    def sort_contours(self, cnts):
        return contours.sort_contours(cnts, method="left-to-right")[0]

    def parse_contours_obj(self, contour_obj):
        return imutils.grab_contours(contour_obj);

    def bound_box(self, contour):
        (x, y, w, h) = cv2.boundingRect(contour)
        region = self.image[y: y + h, x: x + w]
        resized_region = cv2.resize(region, (DIGIT_LENGTH, DIGIT_HEIGHT))
        return resized_region


class ReferenceDigitsReader(OCRReader):
    def __init__(self, ref_path):
        self.digit_to_region = {}
        OCRReader.__init__(self, ref_path)
        
    def threshold_image(self):
        self.threshold = cv2.threshold(
            self.gray, BLACK_WHITE_THRESHOLD, MAX_BINARY_THRESH, cv2.THRESH_BINARY_INV)[1]    

    def process_reference_image(self):
        self.convert_image_to_grayscale()
        self.threshold_image()
        self.set_contours(sort=1)

    def scan_reference_image(self):
        for (digit, contour) in enumerate(self.contours):
            region = self.bound_box(contour)
            self.digit_to_region[digit] = region

    def get_digit_to_region(self):
        return self.digit_to_region


class CardReader(OCRReader):
    def __init__(self, card_path, digit_to_region):
        self.digit_to_region = digit_to_region
        OCRReader.__init__(self, card_path)
        self.groups = []

    def create_kernels(self):
        # We use kernels to do image convolution operations
        self.rectangle_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (9, 3))
        self.square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

    def threshold_image(self):
        threshold = cv2.threshold(
            self.gradient, 0, MAX_BINARY_THRESH, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.threshold = cv2.morphologyEx(
            threshold, cv2.MORPH_CLOSE, self.square_kernel)

    def resize_image(self):
        self.image = imutils.resize(self.image, width=CARD_WIDTH)

    def process_card_image(self):
        self.create_kernels()
        self.resize_image()
        self.convert_image_to_grayscale()
        self.set_gradient()
        self.threshold_image()
        self.set_contours()

    def scan_card_image(self):
        self.get_digits()
        output = self.get_output()
        return output

    def get_output(self):
        output = []
        for (i, (gX, gY, gW, gH)) in enumerate(self.groups):
            group = []
            region = self.extract_region(gX, gY, gW, gH)
            contours_of_region = self.extract_contours(region)
            for contour in contours_of_region:
                (x, y, w, h) = cv2.boundingRect(contour)
                digit = region[y: y + h, x: x + w]
                digit = cv2.resize(digit, (DIGIT_LENGTH, DIGIT_HEIGHT))
                scores = self.template_match(digit)
                group.append(str(np.argmax(scores)))
            cv2.rectangle(self.image, (gX - 5, gY - 5),
                          (gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
            cv2.putText(self.image, "".join(output), (gX, gY - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
            output.extend(group)
        return output

    def template_match(self, box):
        scores = []
        for (digit, digit_region) in self.digit_to_region.items():
            print(box)
            res = cv2.matchTemplate(box, digit_region, cv2.TM_CCOEFF)
            (_, score, _, _) = cv2.minMaxLoc(res);
            scores.append(score);
        return scores

    def extract_region(self, gX, gY, gW, gH):
        # Add a buffer so that the whole "group" of four is extracted
        region = self.gray[gY - GROUP_BUFFER: gY + gH +
                           GROUP_BUFFER, gX - GROUP_BUFFER: gX + gW + GROUP_BUFFER]
        return cv2.threshold(region, 0, MAX_BINARY_THRESH, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    def extract_contours(self, region):
        contour_obj = cv2.findContours(
            region.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = self.parse_contours_obj(contour_obj)
        return contours.sort_contours(cnts, method="left-to-right")[0]

    def get_digits(self):
        for (index, contour) in enumerate(self.contours):
            (x, y, w, h) = cv2.boundingRect(contour)
            aspect_ratio = w / float(h)
            if (self.aspect_ratio_compatible(aspect_ratio) and self.height_width_compatible(w,h)):
                self.groups.append((x, y, w, h))
        self.groups = sorted(self.groups, key=lambda x: x[0])

    def height_width_compatible(self, width, height):
        height_okay = height < MAX_BLOCK_HEIGHT and height > MIN_BLOCK_HEIGHT
        width_okay = width < MAX_BLOCK_WIDTH and width > MIN_BLOCK_WIDTH
        return height_okay and width_okay

    def aspect_ratio_compatible(self, aspect_ratio):
        return (aspect_ratio > MIN_ASPECT_RATIO) and (aspect_ratio < MAX_ASPECT_RATIO)

    def set_gradient(self):
        white_against_dark = self.tophat_transform()
        x_gradient = np.absolute(
            cv2.Sobel(white_against_dark, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1))
        normalized = self.normalize_min_max(x_gradient).astype("uint8")
        self.gradient = cv2.morphologyEx(
            normalized, cv2.MORPH_CLOSE, self.rectangle_kernel)

    def normalize_min_max(self, x_gradient):
        (minVal, maxVal) = (np.min(x_gradient), np.max(x_gradient))
        x_gradient = (255 * ((x_gradient - minVal) / (maxVal - minVal)))
        return x_gradient

    def tophat_transform(self):
        return cv2.morphologyEx(self.gray, cv2.MORPH_TOPHAT, self.rectangle_kernel)

# Demo


#cl_parser = CommandLineParser()
#parsed = cl_parser.parse()

#ref_digits_reader = ReferenceDigitsReader(parsed["reference"])
#ref_digits_reader.process_reference_image()
#ref_digits_reader.scan_reference_image()
#digits = ref_digits_reader.get_digit_to_region()

#card_reader = CardReader(parsed["image"], digits)
#card_reader.process_card_image()
#card_reader.scan_card_image()
#result = card_reader.get_output()
#cv2.imshow("Image", card_reader.image)
#cv2.waitKey(0)
#print("Provider: {}".format(FIRST_DIGIT_TO_PROVIDER[result[0]]))
#print("CC#: {}".format("".join(result)))

