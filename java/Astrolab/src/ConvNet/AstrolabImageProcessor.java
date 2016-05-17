package ConvNet;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class AstrolabImageProcessor {

	public Mat denoise(Mat image) {
		
		image = new Mat(image, new Rect(106, 106, 212, 212));
		
		Mat gray = new Mat();
		Mat binary = new Mat();
		Mat finalImage = new Mat();
		
		Mat hierarchy = new Mat();
		List<MatOfPoint> contours = new ArrayList<>();
		
		MatOfPoint biggerContour = null;
		
		Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);		
		Imgproc.threshold(gray, binary, 20, 255, Imgproc.THRESH_BINARY);		
		Imgproc.findContours(binary, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_NONE);
		
		for (MatOfPoint contour: contours) {
			Double area = Imgproc.contourArea(contour);
			if (area < Math.pow(10, 2) || Math.pow(10, 2) < area ) {
				if (biggerContour == null || Imgproc.contourArea(contour) > Imgproc.contourArea(biggerContour)) {
					biggerContour = contour;
				}
			}
		}
		
		Mat maskImage = new Mat(image.size(), CvType.CV_8U, new Scalar(0));
		Imgproc.drawContours(maskImage, contours, contours.indexOf(biggerContour), new Scalar(255), Core.FILLED);
		
		gray.copyTo(finalImage, maskImage);		
		finalImage.convertTo(finalImage, -1, 1.1, 0);
		
		return finalImage;
		
	}
}
