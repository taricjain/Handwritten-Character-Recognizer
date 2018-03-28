import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

// note use of Integer and Double object types, not int and double

class Network {
	ArrayList<Integer> sizes;  // List whose ith component is how many neurons are in layer i
	// input layer = layer 0
   	int numLayers;             	// number of layers in the network
   	ArrayList<Double[]> biases;  	// list of vectors, ith vector is the biases for layer i
   	ArrayList<Double[][]> weights;  //  list of matrices, ith matrix has weights for layer i

	// constructor, takes a vector of number of nodes in each layer,
	// layer 0 = input layer, layer sizes.size-1 is the output layer
   	public  Network(ArrayList<Integer> argSizes) {
    	Random rng = new Random();  // random number generator
   		// copy in some globals
      	sizes = argSizes;
      	numLayers = sizes.size();
   		// initialize the weights and biases lists
      	biases = new ArrayList<Double[]>();
      	weights = new ArrayList<Double[][]>();
   
   		// generate initial Gaussianly distributed weights and biases for
   		// layers 1...numLayers-1 (no weights or biases for layer 0)
      	for (int i = 1; i < numLayers; i++) {
      		// generate biases for each layer except 0
         	Double[] thisLayersBiases = new Double[sizes.get(i)];
         	for (int j = 0; j < sizes.get(i); j++) {
            	thisLayersBiases[j] = rng.nextGaussian();
         	}  
        	biases.add(thisLayersBiases);
      
      		// generate weights for this layer, matrix based on size of previous layer
         	Double[][] thisLayersWeights = new Double[sizes.get(i)][sizes.get(i-1)];
         	for (int j = 0; j < sizes.get(i); j++) {
          		for(int k = 0; k < sizes.get(i-1); k++) {
           			thisLayersWeights[j][k] = rng.nextGaussian();
            	}
         	}
         	weights.add(thisLayersWeights);
      	}
   	}

	public Double[] feedForward(Double[] input) {
		// list of outputs from each layer, 0 to numLayers
   		// each output is an array of doubles as long as the number of nodes in that layer
      	ArrayList<Double[]> layerOutputs = new ArrayList<Double[]>();
      	if (input.length != sizes.get(0)) {
         	System.out.println("feedForward: wrong input size");
         	System.exit(-1);
      	}
      	layerOutputs.add(input); // layer 0 is the inputs
   	
   		// calculate each layer of outputs
      	for (int n = 1; n < numLayers; n++) {
        	Double[] thisLayerOutput = new Double[sizes.get(n)];
      		
         	for(int i = 1; i < sizes.get(n); i++) {
            	Double dot = 0.0;
         	
         		// multiply weights by inputs from previous layer (this is a dot product)
            	for(int j = 1; j < sizes.get(n-1); j++) {
               		dot += weights.get(n-1)[i][j] * (layerOutputs.get(n-1)[j]);
            	}
         		// compute the sigmoid
            	thisLayerOutput[i] = (1.0) / (1.0 + Math.exp(-(dot + (biases.get(n-1))[i])));
         	}
      		// add the layer to the list of outputs
         	layerOutputs.add(thisLayerOutput);
      	}
   		// the last layer of outputs is the output of the network
      	return(layerOutputs.get(numLayers-1));
   	}

	// Stochastic Gradient Descent method using backpropagation
   	public  void SGD(MnistData trainingData, int epochs, int miniBatchSize, Double eta, MnistData testData) {
		for (int j= 0; j < epochs; j++) {
      		// randomize the trainingData before picking a minibatch
         	Collections.shuffle(trainingData.images);
      
         	ArrayList<CharImgType> thisMiniBatch = new ArrayList<CharImgType>();
         	for (int k  = 0; k < trainingData.images.size(); k += miniBatchSize) {
            	for (int i = 0; i < miniBatchSize; i++) {
              		CharImgType thisCharImg = trainingData.images.get(k+i);
            		thisMiniBatch.add(thisCharImg);
            	}
            	updateMiniBatch(thisMiniBatch, eta);
        	}
        	if (testData != null) {
            	System.out.println("Epoch " + j + ": " + evaluate(testData) + " / " + testData.images.size());
         	} 
         	else {
           		System.out.println("Epoch " + j + " complete.");
         	}
      	}
	}
	
	//Applying backpropagation() to mini batches to update the weights and biases
	@SuppressWarnings("unchecked")   
   	public void updateMiniBatch(ArrayList<CharImgType> miniBatch, Double eta) {
   		//CLONING biases and weights
      	ArrayList<Double[]> nabla_b = new ArrayList<Double[]>(); 
      	ArrayList<Double[][]> nabla_w = new ArrayList<Double[][]>();
      	for (int i = 1; i < numLayers; i++) {
      		// generate biases for each layer except 0
         	Double[] thisLayersBiases = new Double[sizes.get(i)];
         	for (int j = 0; j < sizes.get(i); j++) {
            	thisLayersBiases[j] = (Double) 0.0;
         	}  
         	nabla_b.add(thisLayersBiases);
      	
      		// generate weights for this layer, matrix based on size of previous layer
         	Double[][] thisLayersWeights = new Double[sizes.get(i)][sizes.get(i-1)];
         	for (int j = 0; j < sizes.get(i); j++) {
            	for(int k = 0; k < sizes.get(i-1); k++) {
               		thisLayersWeights[j][k] = (Double) 0.0;
            	}
         	}
         	nabla_w.add(thisLayersWeights);
      	}
   	
   		//UPDATING NABLA VALUES
      	for (int i = 0; i < miniBatch.size(); i++) {
        	ArrayList<Double[]> delta_b = (ArrayList<Double[]>) nabla_b.clone();
         	ArrayList<Double[][]> delta_w = (ArrayList<Double[][]>) nabla_w.clone();
         	backpropagation(miniBatch.get(i), delta_b, delta_w);
      
      		//updating nabla_b
         	for (int j = 0; j < nabla_b.size(); j++) {
            	for (int k = 0; k < nabla_b.get(j).length; k++) {
            	   	nabla_b.get(j)[k] += delta_b.get(j)[k];
            	}
         	}
      
      		//updating nabla_w
			for (int n = 0; n < nabla_w.size(); n++) { 
				for (int j = 0; j < nabla_w.get(n).length; j++) {
					for(int k = 0; k < nabla_w.get(n)[j].length; k++) {
						nabla_w.get(n)[j][k] += delta_w.get(n)[j][k];
					}
	   			}
			}
     	}
   
   		//UPDATING WEIGHTS AND BIASES FOR NN
		//updating biases
		for (int i = 0; i < biases.size(); i++) {
			for (int j = 0; j < biases.get(i).length; j++) {
				biases.get(i)[j] = biases.get(i)[j] - ((eta/miniBatch.size()) * nabla_b.get(i)[j]);
			}
		}
	
		//updating weights
		for (int n = 0; n < weights.size(); n++) { 
			for (int j = 0; j < weights.get(n).length; j++) {
				for(int k = 0; k < weights.get(n)[j].length; k++) {
					weights.get(n)[j][k] = weights.get(n)[j][k] - ((eta/miniBatch.size()) * nabla_w.get(n)[j][k]);
				}
			}
		}
   	}
	
	public void backpropagation(CharImgType data, ArrayList<Double[]> biases, ArrayList<Double[][]> weights) {
		//FEED FORWARD
   		//would contain all the 'z' values for each layer
    	ArrayList<Double> zs = new ArrayList<Double>();
   		//would contain all the activation values for each layer
      	ArrayList<Double[]> activations = new ArrayList<Double[]>();
   		//converting primitive int to wrapper Double
      	Double[] input = new Double[data.image.length];
      	for (int i = 0; i < input.length; i++) {
      		//activation will be used as the inputLayer i.e. output from the first layer
         	input[i] = new Double(data.image[i]);
      	}
      	activations.add(input);
   
      	for (int i = 1; i < numLayers; i++) {
         	Double z = new Double(0.0);
         	Double[] thisLayerActivation = new Double[sizes.get(i)];
         	for (int j = 0; j < sizes.get(i); j++) {
            	Double dot = new Double(0.0);
            	for (int k = 0; k < sizes.get(i - 1); k++) {
               		dot += weights.get(i-1)[j][k] * activations.get(i-1)[k];
            	}
            	z += dot + biases.get(i-1)[j];
            	thisLayerActivation[j] = (1.0)/((1.0) + Math.exp(-(z)));
         	}
         	zs.add(z);
         	activations.add(thisLayerActivation);
      	}
   	
   		//UPDATE: FORWARD PASS DOES NOT WORK PROPERLY.
		
		   //BACKWARD PASS
		// delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
		// nabla_b[-1] = delta
		// nabla_w[-1] = np.dot(delta, activations[-2].transpose())

		// for l in xrange(2, self.num_layers):
		// 	z = zs[-l]
		// 	sp = sigmoid_prime(z)
		// 	delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
		// 	nabla_b[-l] = delta
		// 	nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())  
		
		
   	}

   	public int evaluate(MnistData testData) { 
		return (0); 
	}

	//Loss function L(predicted y, desired y) = - ( ( desired y * log(predicted y) ) + (1 - desired y) * (log(1 - predicted y)) )
	//Cost function J(weights, biases) = - (1/m) * (∑ L(y, ˆy))
}