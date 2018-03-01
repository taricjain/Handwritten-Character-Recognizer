import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Random;
import java.util.function.LongToDoubleFunction;

// note use of Integer and Double object types, not int and double

class Network {
    ArrayList<Integer> sizes;  // List whose ith component is how many neurons are in layer i
	
									// input layer = layer 0
    int numLayers;             		// number of layers in the network
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
		    		dot += weights.get(n)[i][j] * (layerOutputs.get(n-1)[j]);
				}
				
				// compute the sigmoid
				thisLayerOutput[i] = (1.0) / (1.0 + Math.exp(-(dot + (biases.get(n))[i])));
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
				System.out.println("Epoch " + j + ": " + evaluate(testData) + " / " + testData.images.size());
			} 
			else {
				System.out.println("Epoch " + j + " complete.");
			}
		}
    }
	
	//Applying backpropagation() to mini batches to update the weights and biases
	public void updateMiniBatch(ArrayList<CharImgType> miniBatch, Double eta) { }
	
	public ArrayList<Double[]> backpropagation(CharImgType data) {
		Double[] nabla_w = new Double[weights.get(0).length];
		Double[] nabla_b = new Double[biases.get(0).length];
		ArrayList<Double[]> result = new ArrayList<Double[]>();
		
		//FEED FORWARD

		//need a 'z' value for calculating activation for a layer
		//would contain all the 'z' values for each layer
		ArrayList<Double> zs = new ArrayList<Double>();

		//would contain all the activation values for each layer
		ArrayList<Double[]> activations = new ArrayList<Double[]>();

		//converting primitive int to wrapper Double
		// ArrayList<Double[]> inputLayer = new ArrayList<Double[]>();
		
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
					
					dot += weights.get(i)[j][k] * (activations.get(i-1)[k]);
				
				}
				
				z += dot + (biases.get(i))[j];
				thisLayerActivation[i] = (1.0) / ( (1.0) + Math.exp( -(z) ) );
			
			}
			
			zs.add(z);
			activations.add(thisLayerActivation);
		
		}

		//BACKWARD PASS
		
		// //fix the dimensions
		// Double[] delta = new Double[activations.get(numLayers - 1).length];
		
		// Integer y = new Integer(data.label);
		// // delta = cost_derivative(activations.get(numLayers - 1), y) * ( sigmoid_prime(zs.get(numLayers - 1)) );


		// result.add(nabla_b);
		// result.add(nabla_w);
		
		return activations;
	}

	public int evaluate(MnistData testData) { return (0); }

	// public Double[] cost_derivative(Double[] output_activations, Integer desired_output) {
	// 	//fix the dimensions
	// 	//Double[] derivatives = new Double[];
	// 	for (int i = 0; i < output_activations.length; i++) {
	// 		derivatives[i] = output_activations[i] - y;
	// 	}
	// 	return derivatives;
	// }

	public Double sigmoid_prime(Double zs) { return 0.0; }
}
