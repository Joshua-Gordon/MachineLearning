#pragma once
#include<iostream>
#include<armadillo>
using namespace std;
using namespace arma;

struct Layer {
	mat weight;
	vec bias;
	int activation;
};

vec outputError(const Layer & output, const vec & weightedInput, const vec & expected); //TEMP

class MultilayerPerceptron {
private:
	Layer* layers; //The last layer is null
	int numLayers;
	int* heights;
	double learningRate;
public:
	MultilayerPerceptron(int nl, int* h, double lr) : numLayers(nl), heights(h), learningRate(lr){}
	MultilayerPerceptron(int nl, initializer_list<int> h, double lr) : numLayers(nl), learningRate(lr){
		heights = (int*)malloc(sizeof(int)*nl);
		copy(h.begin(),h.end(),heights);
	}
	~MultilayerPerceptron() {
		free(layers);
		free(heights);
	}
	void initialize();		
	void train(const vec & in, const vec & expected);
	vec forward(const vec & in, vec* wi);
	//for debug
	void printInfo();
	void setWeight(int l,const mat & w){
		 layers[l].weight = w; 
	}
	void setBias(int l,const vec & b){ 
		layers[l].bias = b; 
	}	
};
