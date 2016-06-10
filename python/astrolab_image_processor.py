import numpy as np
import cv2

class AstrolabImageProcessor:

    def denoise(self, image):
        #image = image[106:318, 106:318]
        image_size = 112
        image_center = image_size / 2
        image = cv2.resize(image,(image_size, image_size), interpolation = cv2.INTER_CUBIC)

        binary = np.empty(image.size)
        finalImage = np.empty(image.size)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        biggerContourIndex = 0
        maxContourSize = 0
        count = 0

        for contour in contours:
            contourSize = cv2.contourArea(contour)
            if contourSize > maxContourSize and cv2.pointPolygonTest(contour, (image_center, image_center), False) == 1:
                maxContourSize = contourSize
                biggerContourIndex = count
            count += 1

        mask = np.zeros(binary.shape, np.uint8)
        mask = cv2.drawContours(mask, contours, biggerContourIndex, (255,255,255), cv2.FILLED)

        maskedImage = cv2.bitwise_and(gray, gray, mask=mask)

        return maskedImage
