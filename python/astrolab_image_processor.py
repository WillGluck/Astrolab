
class AstrolabImageProcessor:

    def denoise(image):
        image = Mat(image, Rect(106, 106, 212, 212))

        gray = Mat()
		binary = Mat()
		finalImage = Mat()

		hierarchy = Mat()
		contours = []

		biggerContour = None

		cv2.cvtColor(image, gray, cv2.COLOR_BGR2GRAY)
		cv2.threshold(gray, binary, 20, 255, cv2.THRESH_BINARY)
		cv2.findContours(binary, contours, hierarchy, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

		for contour in contours:
			area = cv2.contourArea(contour)
			if area < pow(10, 2) or pow(10, 2) < area:
				if biggerContour == None or cv2.contourArea(contour) > cv2.contourArea(biggerContour)
					biggerContour = contour

		maskImage = Mat(image.size(), CvType.CV_8U, Scalar(0));
		cv2.drawContours(maskImage, contours, contours.indexOf(biggerContour), Scalar(255), Core.FILLED)

		gray.copyTo(finalImage, maskImage)
		finalImage.convertTo(finalImage, -1, 1.1, 0)

		return finalImage
