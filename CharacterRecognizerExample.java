import java.util.ArrayList;
import java.io.IOException;

public class CharacterRecognizerExample  {
    public static void main(String[] args) throws IOException {
   	    // Read all the training images
   
        MnistData trainingData = new MnistData("./Data-unpickled/train");
   	
   	    // The number of input nodes is the number of pixels in an image
   	    // There is one hidden layer with 15 nodes
   	    // There are 10 output nodes, one for each digit 0,1,...,9
        int numHidden = 15; // number of hidden nodes
        int numOutput = 10; // number of output nodes
   
   	    //  Allocate the weights and biases and randomly initialize (all done in
   	    // constructor for Network object)
   	    // The ith element of sizes is the number of nodes in layer i of the neural
   	    // network, layer 0 = the inputs
   
        ArrayList<Integer> sizes = new ArrayList<Integer>();
        sizes.add(trainingData.height*trainingData.width);
        sizes.add(numHidden);
        sizes.add(numOutput);
   	
        Network net = new Network(sizes);
   
   	    // Load the test data
        MnistData testData = new MnistData("./Data-unpickled/t10k");
   	
   	    // Use Stochastic Gradient Descent to calculate weights and biases, evalute with
   	    // test data after each epoch
        int epochs = 30; // number of epochs
        int miniBatchSize = 10; // size of mini batches
        Double eta = 3.0; // learning rate
        net.SGD(trainingData, epochs, miniBatchSize, eta, testData);
   }
}
