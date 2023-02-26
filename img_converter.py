# Imports
import pytesseract
import cv2
import numpy as np

pytesseract.pytesseract.tesseract_cmd = 'C:/opt/homebrew/Cellar/tesseract/5.3.0/bin/tesseract.exe'  # your path may be different

# TODO: add tolerance to functions

# Find counters and bounding boxes
def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    bounding_boxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, bounding_boxes) = zip(*sorted(zip(cnts, bounding_boxes),
                                         key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return cnts, bounding_boxes


# Checks if two rectangles overlap
def rectangle_overlap(R1, R2):
    return (abs(R1[0] + R1[2] / 2) - abs(R2[0] + R2[2] / 2)) * 2 < (R1[2] + R2[2]) and (abs(R1[1] + R1[3] / 2) - abs(
        R2[1] + R2[3] / 2)) * 2 < (R1[3] + R2[3]) and (abs(R2[0] + R2[2] / 2) - abs(R1[0] + R1[2] / 2)) * 2 < (
                   R1[2] + R2[2]) and (abs(R2[1] + R2[3] / 2) - abs(
        R1[1] + R1[3] / 2)) * 2 < (R1[3] + R2[3])


class NonogramImage:
    def __init__(self, image_path):
        self.image_path = image_path
        # Opening the image & storing it in an image object
        self.image = cv2.imread(self.image_path, 0)
        # Find boxes and stores in list
        self.boxes = self.img_box_finder()
        # Removes duplicate boxes
        self.remove_duplicates()
        # Sets array of box values
        self.box_values = self.convert_imgs_to_text()

    # Return images of boxes
    def img_box_finder(self):
        # Thresholding the image
        (thresh, img_bin) = cv2.threshold(self.image, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        # Invert the image
        img_bin = 255 - img_bin

        # Defining a kernel length
        kernel_length = np.array(self.image).shape[1] // 80

        # A verticle kernel of (1 X kernel_length), which will detect all the verticle lines from the image.
        verticle_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_length))

        # A horizontal kernel of (kernel_length X 1), which will help to detect all the horizontal line from the image.
        hori_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_length, 1))
        # A kernel of (3 X 3) ones.
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        # Morphological operation to detect vertical lines from an image
        img_temp1 = cv2.erode(img_bin, verticle_kernel, iterations=3)
        verticle_lines_img = cv2.dilate(img_temp1, verticle_kernel, iterations=3)

        # Morphological operation to detect horizontal lines from an image
        img_temp2 = cv2.erode(img_bin, hori_kernel, iterations=3)
        horizontal_lines_img = cv2.dilate(img_temp2, hori_kernel, iterations=3)

        # Weighting parameters, this will decide the quantity of an image to be added to make a new image.
        alpha = 0.5
        beta = 1.0 - alpha

        # This function helps to add two image with specific weight parameter to get a summation of two images
        img_final_bin = cv2.addWeighted(verticle_lines_img, alpha, horizontal_lines_img, beta, 0.0)
        img_final_bin = cv2.erode(~img_final_bin, kernel, iterations=2)
        (thresh, img_final_bin) = cv2.threshold(img_final_bin, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Find contours for image, which will detect all the boxes
        contours, hierarchy = cv2.findContours(img_final_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # Sort all the contours by top to bottom.
        (contours, boundingBoxes) = sort_contours(contours, method="top-to-bottom")

        box_imgs = []
        for c in contours:
            # Returns the location and width,height for every contour
            x, y, w, h = cv2.boundingRect(c)
            new_img = [self.image[y:y + h, x:x + w], (x, y, w, h)]
            box_imgs.append(new_img)
        return box_imgs

    # Removes duplicate boxes
    def remove_duplicates(self):
        duplicate_list = []
        for i in range(len(self.boxes)):
            for j in range(i + 1, len(self.boxes)):
                if rectangle_overlap(self.boxes[i][1], self.boxes[j][1]):
                    box_i_area = self.boxes[i][1][2] * self.boxes[i][1][3]
                    box_j_area = self.boxes[j][1][2] * self.boxes[j][1][3]
                    if box_i_area > box_j_area:
                        duplicate_list.append(j)
                    elif box_i_area < box_j_area:
                        duplicate_list.append(i)
        for i in sorted(duplicate_list, reverse=True):
            self.boxes.pop(i)

    # Finds the top clues
    def find_top_clues(self):
        x_val = []
        for box in range(len(self.boxes)):
            x_val.append(self.boxes[box][1][1])
        x_top = min(x_val)
        x_th = 20
        top_clues = []
        while len(top_clues) < 5:
            for box in range(len(self.boxes)):
                if self.boxes[box][1][1] < x_top + x_th:
                    print(self.boxes[box][1][1], self.box_values[box])
                    top_clues.append(self.box_values[box])
            x_th += 2
            if x_th >= 50:
                return False
        return top_clues

    # Finds the side clues
    def find_side_clues(self, box_imgs):
        pass

    # Extracts text from boxes and outputs as a list
    def convert_imgs_to_text(self):
        text_output = []
        for box in self.boxes:
            # This function will extract the text from the image
            num = pytesseract.image_to_string(box[0], config="--psm 6 outputbase digits")

            # Converts text to int
            str_num = num.replace("\n", "")
            if str_num == '':
                text_output.append(0)
            elif int(str_num) <= 9:
                text_output.append(int(str_num))
            else:
                dec_list = []
                for decimal in num.splitlines():
                    if decimal.isnumeric():
                        dec_list.append(int(decimal))
                text_output.append(dec_list)
        for i in reversed(range(len(text_output))):
            if text_output[i] == 0:
                text_output.pop(i)
                self.boxes.pop(i)
        return text_output

    # Converts img to data form
    def convert_img(self):

        print(self.box_values)
        print(len(self.boxes))
        top_clues = self.find_top_clues()
        print("top_clues:", top_clues)


if __name__ == "__main__":
    image_p = r"Img/Screen Shot 2022-01-16 at 5.16.26 AM.png"
    # image_p = r"Img/Challange.png"
    # image_p = r"Img/Screen Shot 2022-01-16 at 10.58.06 PM.png"
    m1 = NonogramImage(image_p)
    m1.convert_img()
