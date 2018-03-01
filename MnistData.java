import javax.swing.JFrame;
import java.nio.file.*;
import java.io.IOException;
import java.util.Scanner; 
import java.awt.image.*; 
import java.awt.Color;
import java.util.Arrays;
import java.util.ArrayList;  
public class MnistData { 

    int height, width, numImages; 
    ArrayList<CharImgType> images; 

public MnistData(String fileArg) throws IOException {  
    byte[] data,labeldata;
    // a scanner to cause program to wait until you press enter
    Scanner scan = new Scanner(System.in); 
    Path fileLocation = Paths.get(fileArg+"-images-idx3-ubyte");
    data = Files.readAllBytes(fileLocation);
    fileLocation = Paths.get(fileArg+"-labels-idx1-ubyte");
    labeldata = Files.readAllBytes(fileLocation);
    // offset into the labels file, no need to read its headers; one-dimension, same number of examples as the images
    int labelIndex = 8; 
    // process magic number and number of images
    // see documentation on file format of the mnist data (unpickled) 
    numImages = Byte.toUnsignedInt(data[4])*(int)Math.pow(2,24)+
       Byte.toUnsignedInt(data[5])*(int)Math.pow(2,16)+
       Byte.toUnsignedInt(data[6])*(int)Math.pow(2,8)+Byte.toUnsignedInt(data[7]); 
    height = Byte.toUnsignedInt(data[8])*(int)Math.pow(2,24)+
       Byte.toUnsignedInt(data[9])*(int)Math.pow(2,16)+
       Byte.toUnsignedInt(data[10])*(int)Math.pow(2,8)+Byte.toUnsignedInt(data[11]);
    width = Byte.toUnsignedInt(data[12])*(int)Math.pow(2,24)+
       Byte.toUnsignedInt(data[13])*(int)Math.pow(2,16)+
       Byte.toUnsignedInt(data[14])*(int)Math.pow(2,8)+Byte.toUnsignedInt(data[15]);
    System.out.println("Reading " + numImages + ", height=" + height + " ,width=" + width + ", from " + fileArg + "-images-idx3-ubyte");
    int index = 16; // index into the bytes of the input holding the  array of images;
    // images is an ArrayList of CharImgType, each element has a vector with the pixels of
    // an image, and an int with the label
    images  = new ArrayList<CharImgType>(); 
    // labels tells what digit each trainingMatrix image represents 
    for (int n = 0;  n < numImages; n++ ) {
	CharImgType thisImage = new CharImgType(height*width);
        // BufferedImage img = new BufferedImage(height,width,BufferedImage.TYPE_INT_RGB); 
	for (int i = 0; i < 28; i++) { 
	    for (int j=0 ; j < 28; j++) { 
	        // build an image to display
	    
	        // int gray = 255-Byte.toUnsignedInt(data[index+i+j*height]); 
	        // img.setRGB(i,j, new Color(gray,gray,gray).getRGB()); 
	    
	        // build the input to the Neural Network Trainer 
	        thisImage.image[i+j*height] = Byte.toUnsignedInt(data[index+i+j*height]); 
           }
        }
	thisImage.label = Byte.toUnsignedInt(labeldata[labelIndex++]);
	images.add(thisImage);
	// // code to browse the images as they are read in, comment out when it's right
	// CharacterImage l1 = new CharacterImage(img, thisImage.label);
	// l1.setSize(100, 100);
	// l1.setVisible(true);
	// // wait for user to press a key 
	// System.out.println("Press <enter> for next image"); 
    // String s = scan.nextLine(); 
    
	// move along our pointer into the training image data file
        index +=height*width;
    } 
}

} 