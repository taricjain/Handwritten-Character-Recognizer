import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;
import java.util.Iterator;

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
				// update weights and biases with this mini batch
            	updateMiniBatch(thisMiniBatch, eta);
                System.out.println("update:" + k);
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
				// filling everything with 0.0 to be filled later in backprop
				thisLayersBiases[j] = (Double) 0.0;
         	}  
         	nabla_b.add(thisLayersBiases);
      	
      		// generate weights for this layer, matrix based on size of previous layer
			Double[][] thisLayersWeights = new Double[sizes.get(i)][sizes.get(i-1)];
         	for (int j = 0; j < sizes.get(i); j++) {
            	for(int k = 0; k < sizes.get(i-1); k++) {
					// filling everything with 0.0 to be filled later in backprop   
					thisLayersWeights[j][k] = (Double) 0.0;
            	}
         	}
         	nabla_w.add(thisLayersWeights);
      	}
   	
   		//UPDATING NABLA VALUES
      	for (int i = 0; i < miniBatch.size(); i++) {
			// cloning nabla_w and nabla_b to be sent in for backprop
			ArrayList<Double[]> delta_b = (ArrayList<Double[]>) nabla_b.clone();
         	ArrayList<Double[][]> delta_w = (ArrayList<Double[][]>) nabla_w.clone();
			/** 
			 * backprop being applied to everyone image in the minibatch
			 * now, send delta_b and delta_w to calculate the derivatives
			 * */  
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
		
		/**
		 * Feed Forward
		*/
		
		// contains all z values
    	ArrayList<Double[]> zs = new ArrayList<Double[]>();

		// contains all activation values
		ArrayList<Double[]> activations = new ArrayList<Double[]>();
		   
		/**
		 * converting primitive int to wrapper Double
		 * because image data was saved as int[] but
		 * computation is done using wrapper Double values.
		*/
      	activations.add(primitiveToWrapperDouble(data.image));

		// loop for layers
		for (int i = 1; i < numLayers; i++) {
         	Double[] z = new Double[sizes.get(i)];
         	Double[] thisLayerActivation = new Double[sizes.get(i)];
			// loop for neurons 
			for (int j = 0; j < sizes.get(i); j++) {
				Double dot = new Double(0.0);
				// loop for calculating dot product.
				for (int k = 0; k < sizes.get(i - 1); k++) {
                    dot += weights.get(i-1)[j][k] * activations.get(i-1)[k];
				}
				// calculating activation using sigmoid function [-1,1]
				z[j] = dot + biases.get(i-1)[j];
				thisLayerActivation[j] = (1.0)/((1.0) + Math.exp(-(z[j])));
			}
			// saving 'z' values and activation values
         	zs.add(z);
        	activations.add(thisLayerActivation);
      	}

		/**
		 * Feed Backward
		*/
		
		/**
		 * basically we will try and calculate the delta in biases and weights
		 * for the last layer and it update the delta_b and delta_w vectors
		 * then do the same for the hidden layer. But we have to use a step
		 * by step process to accomplish this.
		*/
		// last activation layer
		Double[] lastActivation = activations.get(activations.size() - 1);
		// prime of the last z layer
		Double[] lastZPrime = sigmoidPrime(zs.get(zs.size() - 1));

		// creating desired output layer to get derivative
		Double[] desiredOutputLayer = createOutputLayer(data);
		
		// calculate derivative
		Double[] cost_derivative = costDerivative(lastActivation, desiredOutputLayer);
		
		Double[] delta = new Double[desiredOutputLayer.length];
		// calculating delta and save in delta_b
		// working on output layer now
		for (int i = 0; i < delta.length; i++) {
			delta[i] = cost_derivative[i] * lastZPrime[i];
			delta_b.get(delta_b.size() - 1)[i] = delta[i];
		}
		
		/** 
		 * Instead of actually transforming the second layer of activations
		 * to obtain the transpose. I tried having each value of delta 
		 * multiplied with the each value in second layer
		 * of activations and then put that back into nabla_w.
		*/
		Double[] secondLayerActivation = activations.get(activations.size() - 2);
		ArrayList<Double> dots = new ArrayList<Double>();
		for (int i = 0; i < delta.length; i++) {
			Double dot = 0.0;
			for (int j = 0; j < secondLayerActivation.length; j++) {
				dot = delta[i] * secondLayerActivation[j];
				dots.add(dot);
			}
		}

		// create an iterator to go over all the 150 dot products in the dots arraylist
		Iterator<Double> iterate = dots.iterator();
		// iterating over the last delta_weights matrix.
		for (int i = 0; i < delta_w.get(delta_w.size() - 1).length; i++) {
			for (int j = 0; j < delta_w.get(delta_w.size() - 1)[i].length; j++) {
				delta_w.get(delta_w.size() - 1)[i][j] = iterate.next();
			}
		}

		Double[] z;
		Double[] sigPrime;
		// working on hidden layer now.
		for (int n = numLayers - 1; n >= 2; n--) {
			// z for the hidden layer
			z = zs.get(n-2);
			sigPrime = sigmoidPrime(z);
			/**
			 * to achieve the same effect as the transpose of the output layer
			 * of the weights, we will run this nested loop for 15 (outer)
			 * and 10 (inner) times.
			*/
			ArrayList<Double> delta_List = new ArrayList<Double>();
			for (int cols = 0; cols < sizes.get(n-1); cols++) {
				Double dot = 0.0;
				for (int rows = 0; rows < sizes.get(n); rows++) {
					dot += weights.get(n - 1)[rows][cols] * delta[rows] * sigPrime[rows];
				}
				delta_List.add(dot);
			}

			// saving all values in delta_b
			for (int i = 0; i < delta_b.get(n-2).length; i++) {
				delta_b.get(n-2)[i] = delta_List.get(i);
			}
			
			// get the tranpose of activations(0) [784] dotted by the new delta_list [15]
			// and save in weights[0] [15x784]
			ArrayList<Double> delta_w_list = new ArrayList<Double>();
			for (int i = 0; i < delta.length; i++) {
				Double dot = 0.0;
				for (int j = 0; j < activations.get(n - 2).length; j++) {
					dot = delta_List.get(i) * activations.get(n - 2)[j];
					delta_w_list.add(dot);
				}
			}

			for (int i = 0; i < delta_w.get(n - 2).length; i++) {
				for (int j = 0; j < delta_w.get(n - 2)[i].length; j++) {
					delta_w.get(n - 2)[i][j] = delta_w_list.get(j);
				}
			}
		}
   	}

   	public int evaluate(MnistData testData) { 
		int score = 0;
		for (int i = 0; i < testData.numImages; i++) {
			CharImgType myImage = testData.images.get(i);
			Double prediction = new Double(myImage.label);
			
			// get the network output for this image.
			Double[] networkOutput = feedForward
				(primitiveToWrapperDouble(myImage.image));
			
			// figure out which neuron was fired by
			// finding the largest activation value in the array.
			Double max = networkOutput[0];
			int index = 0;
			for (int j = 1; j < networkOutput.length; j++) {
				if (networkOutput[j] > max) {
					index = j;
				}
			}

			// if the prediction is equal to "index" we have a win.
			// if not then, dont add anything to score and keep going.
			score = (prediction == index) ? score + 1 : score;
		}
		return score; 
	}

	public Double[] costDerivative(Double[] networkOutput, Double[] desiredOutput) {
		Double[] derivative = new Double[networkOutput.length];
		for (int i = 0; i < networkOutput.length; i++) {
			derivative[i] = desiredOutput[i] - networkOutput[i];
		}
		return derivative;
	}
	
	public Double[] sigmoidPrime(Double[] z) {
		Double[] prime = new Double[z.length];
		//calculate the sigmoid
		for (int i = 0; i < z.length; i++) {
			Double sigmoid = (1.0)/((1.0) + Math.exp(-(z[i])));
			prime[i] = sigmoid;
		}
		//return prime
		return prime;
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

	public Double[] primitiveToWrapperDouble(int[] image) {
		Double[] input = new Double[image.length];
		for (int i = 0; i < input.length; i++) {
		   input[i] = new Double(image[i]);
		}
		return input;
	}
}