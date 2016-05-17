package ConvNet;

import java.util.ArrayList;
import java.util.List;

import org.opencv.core.Core;
import org.opencv.core.Mat;

public class ConvNet {
	
	static public void readMnist(String path, List<Mat> vector) {
		
	}
	
	static public void readMnistLabel(String path, Mat vector) {
		
	}
	
	static public void readData(List<Mat> xVector, Mat yVector, String xPath, String yPath, int imagesCount) {
		readMnist(xPath, xVector);
		xVector.stream().forEach(x -> x.convertTo(x, 1, 1.0 / 255, 0));
		yVector = Mat.zeros(1, imagesCount, 1);
		readMnistLabel(yPath, yVector);
	}
	
	public static void main(String[] args) {
		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
		new AstrolabPipeline().denoiseImage();
		
//		System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
//		
//		long start, end;
//		
//		List<Mat> trainX = new ArrayList<>();
//		List<Mat> testX = new ArrayList<>();
//		
//		Mat trainY = new Mat();
//		Mat testY = new Mat();
//		
//		readData(trainX, trainY, "C:/dataset/train-images-idx3-ubyte", "C:/dataset/train-labels-idx1-ubyte", 60000);
//		
	}

}
