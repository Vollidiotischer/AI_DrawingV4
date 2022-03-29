#include <iostream>

#include <vector>

#include "SimpleAI/SimpleAI.h"

#include "Variables.h"
#include "MainLoop.h"



int main() {

	
	// load training data
	std::vector<SimpleAI::Data_Point> data;
	SimpleAI::Resource_Manager::load_mnist_data("Data/train-images.idx3-ubyte", "Data/train-labels.idx1-ubyte", data);


	std::cout << "Size of Dataset: " << data.size() << std::endl; 

	// train nn
	SimpleAI::AI_Manager manager(1); 
	MainLoop::start_training(manager, data);

	char c; 
	std::cin >> c; 

	// start canvas with best nn
	MainLoop::start_canvas(*manager.best_instance, data);
	
	
	return 0; 
}

/*

program structure: 
	Modes: 
		1. Train
		2. Create Data
		3. Learn 


*/