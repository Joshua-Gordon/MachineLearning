#include "MLP.h"

int main() {
	/*
	cout << "Hello world!" << endl;
	int layerHeights[] = {2,4,4,2};
	MultilayerPerceptron mlp(4,layerHeights);
	mlp.initialize();
	mlp.printInfo();

	vec test({1,2});
	cout << mlp.forward(test,NULL) << endl;
	*/
	MultilayerPerceptron mlp(3,{3,2,1},0.01);
	mlp.initialize(); //ow my bones
	mlp.printInfo();
	/*mat weight1, weight2;
	vec bias1 = {0,0},bias2 = {0};
	weight1 << 1 << 0 << endr << 0 << 1 << endr;
	weight2 << 2 << 0 << endr;// << 0 << 2 << endr;
	mlp.setWeight(0,weight1);
	mlp.setBias(0,bias1);
	mlp.setWeight(1,weight2);
	mlp.setBias(1,bias2);*/
	mlp.printInfo();
	vec in = {1,2,1};
	vec expect = {5};
	vec out = mlp.forward(in,NULL);
	cout << out << endl;
	cout << "Beginning training" << endl;
	for(int i = 0; i < 100; ++i)
		mlp.train(in,expect);
	cout << "Training ended" << endl;
	cout << mlp.forward(in,NULL);
	/*MultilayerPerceptron mlp(4,{2,4,4,2});
	mlp.initialize();
	vec in = {1,2};
	vec exp = {2,4};
	mlp.train(in,exp);*/
	/*vec a = {1,1,1};
	vec b = {2,3,2};
	Layer l;
	l.activation = 0;
	cout << outputError(l,a,b) << endl;*/	
	return 0;
}
