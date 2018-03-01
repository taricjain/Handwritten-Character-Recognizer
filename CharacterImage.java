import java.awt.Graphics;
import java.awt.Graphics2D; 
import javax.swing.JComponent; 
import javax.swing.JFrame; 
import java.awt.Container; 
import java.awt.image.BufferedImage; 

public class CharacterImage  extends JFrame  {

    public CharacterImage(BufferedImage argImage,int label) {
	    Container cp = getContentPane(); 
	    MyCanvas t1 = new MyCanvas(argImage,label);
	    cp.add(t1); 
    }
}

class MyCanvas extends JComponent {

	BufferedImage img; 
    int label; 

    public MyCanvas(BufferedImage argImage, int argLabel) { 
	    img = argImage; 
        label  = argLabel; 
    } 
  
  @Override
    public void paintComponent(Graphics g) {
        if(g instanceof Graphics2D) {
	        Graphics2D g2 = (Graphics2D) g;
	        g2.drawImage(img,30,30,null); 
	        g2.drawString(String.format("%d",label),90,10); 
	    }
    }
}