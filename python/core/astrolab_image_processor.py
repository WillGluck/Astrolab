import numpy as np
import cv2

class AstrolabImageProcessor:

    def show(self, image):
        resized_image = cv2.resize(image, (512, 512), interpolation = cv2.INTER_CUBIC)
        cv2.imshow('image', resized_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def denoise(self, image, resize_to_size):
        #image = image[106:318, 106:318]
        image_center = resize_to_size / 2
        image = cv2.resize(image,(resize_to_size, resize_to_size), interpolation = cv2.INTER_CUBIC)

        binary = np.empty(image.size)

        finalImage = np.empty(image.size)

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
        self.show(binary)
        _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        index = 0
        for contour in contours:
            cv2.drawContours(image, contours, index, (0,255,0), 0)
            index += 1
        self.show(image)

        biggerContourIndex = -1
        maxContourSize = -1.0
        count = 0

        for contour in contours:
            contourSize = cv2.contourArea(contour)
            if contourSize > maxContourSize and cv2.pointPolygonTest(contour, (image_center, image_center), False) == 1:
                maxContourSize = contourSize
                biggerContourIndex = count
            count += 1

        if biggerContourIndex != -1:
            mask = np.zeros(binary.shape, np.uint8)
            mask = cv2.drawContours(mask, contours, biggerContourIndex, (255,255,255), cv2.FILLED)
            self.show(mask)

            maskedImage = cv2.bitwise_and(image, image, mask=mask)
            self.show(maskedImage)

            return maskedImage
        else:
            return np.array([])
