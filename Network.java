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
	public  void SGD(MnistData trainingData, int epochs, 
		int miniBatchSize, Double eta, MnistData testData) {
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
				System.out.println("Epoch " + j + ": " + 
					evaluate(testData) + " / " + testData.images.size());
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
				biases.get(i)[j] = biases.get(i)[j] - 
					((eta/miniBatch.size()) * nabla_b.get(i)[j]);
			}
		}
	
		//updating weights
		for (int n = 0; n < weights.size(); n++) { 
			for (int j = 0; j < weights.get(n).length; j++) {
				for(int k = 0; k < weights.get(n)[j].length; k++) {
					weights.get(n)[j][k] = weights.get(n)[j][k] - 
						((eta/miniBatch.size()) * nabla_w.get(n)[j][k]);
				}
			}
		}
   	}
	
	public void backpropagation(CharImgType data, ArrayList<Double[]> delta_b,
		ArrayList<Double[][]> delta_w) {
		// FEED FORWARD
		// contains all z values
    	ArrayList<Double> zs = new ArrayList<Double>();

		// contains all activation values
		ArrayList<Double[]> activations = new ArrayList<Double[]>();
		   
		// converting primitive int to wrapper Double
      	Double[] input = new Double[data.image.length];
      	for (int i = 0; i < input.length; i++) {
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
		Double[] lastActivation = activations.get(activations.size() - 1);
		Double sigmoidLastZ = sigmoidPrime(zs.get(zs.size() - 1));

		Double[] desiredOutputLayer = createOutputLayer(data);
		Double[] delta = new Double[desiredOutputLayer.length];
		Double[] cost_derivative = costDerivative(lastActivation, desiredOutputLayer);
		
		for (int i = 0; i < delta.length; i++) {
			delta[i] = cost_derivative[i] * sigmoidLastZ;
			delta_b.get(delta_b.size() - 1)[i] = delta[i];
		}
		
		/** 
		 * instead of actually transforming the second layer of activations
		 * I had each value of delta multiplied with the entire second layer
		 * of activations and then put that back into nabla_w.
		*/
		ArrayList<Double> dots = new ArrayList<Double>();
		for (int i = 0; i < delta.length; i++) {
			Double dot = 0.0;
			for (int j = 0; j < activations.get(activations.size() - 1).length; j++) {
				dot += delta[i] * activations.get(activations.size() - 1)[j];
			}
			dots.add(dot);
		}
		// placing it back into the nabla_w
		for (int i = 0; i < delta_w.size(); i++) {
			for (int j = 0; j < delta_w.get(i).length; j++) {
				for (int k = 0; k < delta_w.get(i)[j].length; k++) {
					delta_w.get(i)[j][k] = dots.get(i);
				}
			}
		}

		Double z;
		Double sp;
		for (int i = numLayers; i >= 2; i--) {
			z = zs.get(i);
			sp = sigmoidPrime(z);
			for (int d = 0; d < delta.length; d++) {
				Double dot = 0.0;
				for (int j = 0; j < weights.get(i).length; j++) {
					for (int k = 0; k < weights.get(i)[j].length; k++) {
						dot += delta[d] * weights.get(i)[j][k];
					}
				}
				dot *= sp;
				delta[d] = dot;
			}
			for (int j = 0; j < delta_b.get(i).length; j++) {
				delta_b.get(i)[j] = delta[j];
			}
			for (int d = 0; d < delta.length; d++) {
				Double dot = 0.0;
				for (int j = 0; j < activations.get(i-1).length; j++) {
					dot += delta[d] * activations.get(i-1)[j];
				}

				for (int j = 0; j < weights.get(i).length; j++) {
					for (int k = 0; k < weights.get(i)[j].length; k++) {
						delta_w.get(i)[j][k] = dot;
					}
				}
			}
		}
   	}

   	public int evaluate(MnistData testData) { 
		return (0); 
	}

	public Double[] costDerivative(Double[] networkOutput, Double[] desiredOutput) {
		Double[] derivative = new Double[networkOutput.length];
		for (int i = 0; i < networkOutput.length; i++) {
			derivative[i] = networkOutput[i] - desiredOutput[i];
		}
		return derivative;
	}
	
	public Double sigmoidPrime(Double z) {
		//calculate the sigmoid
		Double sigmoid = (1.0)/((1.0) + Math.exp(-(z)));
		//return prime
		return (sigmoid * (1 - sigmoid));
	}

	public Double[] createOutputLayer(CharImgType data) {
		// 10 is the number of neurons in the output layer.
		int numOutputs = 10;
		Double[] y = new Double[numOutputs];
		for (int i = 0; i < y.length; i++) {
			y[i] = (i == data.label) ? new Double(1) : new Double(0);
		}
		return y;
	}
}