#include "MLP.h"

double activate(const int mode, const double in) {
	switch(mode) {
		case 0: {
			if(in > 0) {
				return in;
			}
			return 0;
		}
		default: {
			cout << "Invalid activation mode!" << endl;
			return NULL;
		}
	}
}

double activateDerivative(const int mode, const double in) {
	switch(mode) {
		case 0: {
			if(in > 0) {
				return 1;
			}
			return 0;
		}
		default: {
			cout << "Invalid activation mode!" << endl;
			return NULL;
		}
	}
}

void MultilayerPerceptron::initialize() {
	layers = (Layer*)malloc(sizeof(Layer)*numLayers);
	for(int i = 0; i < numLayers-1; ++i) {
		layers[i].activation = 0;
		layers[i].weight = mat(heights[i+1],heights[i],fill::randu);
		layers[i].bias = vec(heights[i+1],fill::randu);
	}
}

/*
This function computes the error of layer l-1 in terms of the error of layer l
*/
vec backpropagate(const Layer & l, const vec & weightedInput, const vec & nextError) {
	mat weightTranspose = l.weight.t();
	vec errorBack = weightTranspose * nextError;
	vec activDiff = weightedInput;
	activDiff.transform([l](double x) {
		return activateDerivative(l.activation,x);
	});
	return errorBack % activDiff;
}

/*
This function computes the error of the output layer
*/
vec outputError(const Layer & output, const vec & weightedInput, const vec & expected) {
	vec activationsL = weightedInput;
	cout << "weightedInput: " << endl << weightedInput << endl;
	activationsL.transform([output](double x) {
		return activate(output.activation,x);
	});
	cout << "activationsL: " << endl << activationsL << endl << "expected: " << endl << expected << endl;
	vec gradientCost = activationsL - expected;
	vec activDiff = weightedInput;
	activDiff.transform([output](double x) {
		return activateDerivative(output.activation,x);
	});
	return gradientCost % activDiff; //% is hadamard product
}

void MultilayerPerceptron::train(const vec & in, const vec & expected) {
	vec* weightedInputs = (vec*)malloc(sizeof(vec)*numLayers);
	vec out = forward(in,weightedInputs);
	for(int i = 0; i < numLayers; ++i) {
		cout << "Weighted inputs for layer " << i << ":" << endl;
		cout << weightedInputs[i] << endl;
	}
	cout << "Going to compute deltaL" << endl;
	vec deltaL = outputError(layers[numLayers-1],weightedInputs[numLayers-1],expected);
	cout << "deltaL is computed." << endl;
	vec* delta = (vec*)malloc(sizeof(vec)*numLayers);
	delta[numLayers-1] = deltaL;
	cout << deltaL << endl;
	cout << "Beginning backpropagation." << endl;
	for(int i = numLayers-2; i >= 0; --i) {
		cout << "Computing error for layer " << i << ": " << endl;
		if(i == 0) {
			delta[i] = backpropagate(layers[i],in,delta[i+1]);
		} else {
			delta[i] = backpropagate(layers[i],weightedInputs[i],delta[i+1]);
		}
		cout << delta[i] << endl;
	} //errors computed
	//bias gradient is equal to error
	//now to compute weight gradient
	mat* weightGradient = (mat*)malloc(sizeof(mat)*numLayers);
	for(int l = 0; l < numLayers; ++l) {
		mat m(layers[l].weight.n_rows,layers[l].weight.n_cols);
		cout << "Starting m: " << endl << m << endl;
		int actMode = layers[l].activation;
		vec activation;
		if(l == 0) {
			activation = in;	
		} else {
			activation = weightedInputs[l].transform([actMode](double x) {
				return activate(actMode,x);
			});
		}
		cout << "Current activation: " << endl << activation << endl;
		cout << "Computing weight gradient matrix " << l << endl;
		for(int k = 0; k < layers[l].weight.n_rows; ++k) {
			for(int j = 0; j < layers[l].weight.n_cols; ++j) {
				//compute derivative of weight l kj
				//= activation l-1[k] * delta[j]
				double value = activation[k]*delta[l][j];
				cout << value << " (" << k << "," << j << ") ";
				m(k,j) = value; //segfaults here. bounds error?
			}
			cout << endl;
		}
		cout << "Weight gradient " << l << ": " << endl << m << endl;
		weightGradient[l]=m;
	} //weight gradient computed, now to apply
	cout << "DEBUG" << endl;
	for(int i = 0; i < numLayers; ++i) {
		cout << "Layer " << i << endl << delta[i] << endl;
	}
	for(int i = 0; i < numLayers-1; ++i) {
		cout << "Change in weight for layer " << i << ":" << endl;
		cout << weightGradient[i];
		layers[i].weight -= learningRate*weightGradient[i];
		cout << "Change in bias for layer " << i << ":" << endl;
		cout << delta[i] << endl;
		cout << "Previous bias: " << endl;
		cout << layers[i].bias;
		layers[i].bias -= learningRate*delta[i+1];
	}
}

vec MultilayerPerceptron::forward(const vec & in, vec* wi) {
	Layer* current = layers;
	vec input(in);
	vec* weightedInput = wi;
	do {
		//cout << "Multiplying matrices in forward" << endl;
		input = (current->weight*input) + current->bias;
		if(wi) {
			*(++weightedInput) = input;
		}
		input.for_each([current](double & x) {
			x = activate(current->activation,x);
		});
	} while((++current)->weight.n_rows);
	//input.for_each([current](double & x) {
	//	x = activate(current->activation,x);
	//});
	return input;
}

void MultilayerPerceptron::printInfo() {
	for(int i = 0; i < numLayers; ++i) {
		cout << "Layer " << i+1 << " weight: " << endl;
		cout << layers[i].weight << endl;
		cout << "Bias: " << endl;
		cout << layers[i].bias << endl;			
	}
}
