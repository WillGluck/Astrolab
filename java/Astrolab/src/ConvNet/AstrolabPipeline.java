package ConvNet;

import java.awt.FlowLayout;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;

import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

public class AstrolabPipeline {
	
	public void denoiseImage() {
		
		Mat image = Imgcodecs.imread("C:/images_test_rev1/102050.jpg");
		displayImage(Mat2BufferedImage(new AstrolabImageProcessor().denoise(image)));
		displayImage(Mat2BufferedImage(image));
		
		
	}
	
   static public void displayImage(Image img2) {   
       ImageIcon icon=new ImageIcon(img2);
       JFrame frame=new JFrame();
       frame.setLayout(new FlowLayout());        
       frame.setSize(img2.getWidth(null)+50, img2.getHeight(null)+50);     
       JLabel lbl=new JLabel();
       lbl.setIcon(icon);
       frame.add(lbl);
       frame.setVisible(true);
       frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

   }
	
    static public BufferedImage Mat2BufferedImage(Mat m){

	    int type = BufferedImage.TYPE_BYTE_GRAY;
	    if ( m.channels() > 1 ) {
	        type = BufferedImage.TYPE_3BYTE_BGR;
	    }
	    int bufferSize = m.channels()*m.cols()*m.rows();
	    byte [] b = new byte[bufferSize];
	    m.get(0,0,b); // get all the pixels
	    BufferedImage image = new BufferedImage(m.cols(),m.rows(), type);
	    final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
	    System.arraycopy(b, 0, targetPixels, 0, b.length);  
	    return image;

    }

}
