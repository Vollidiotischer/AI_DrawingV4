#include <iostream>

#include <vector>
#include <thread>

#include "SimpleAI/SimpleAI.h"

#include "Variables.h"
#include "MainLoop.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>


int main() {
	
	Eigen::Matrix<float, 1, 2> vector = Eigen::Matrix<float, 1, 2>::Random(1, 2); 
	Eigen::Matrix<float, 2, 2> matrix = Eigen::Matrix2f::Random(2, 2); 


	std::cout << vector << "\n" << std::endl; 
	std::cout << matrix << "\n" << std::endl;

	Eigen::Matrix<float, 2, 1> vect2 = matrix * vector.transpose(); 

	std::cout << vector << "\n" << std::endl;
	std::cout << matrix << "\n" << std::endl;
	std::cout << vect2 << "\n" << std::endl;

	std::cout << matrix.size() << std::endl; 

	std::cin.get(); 
	

	// load training data
	std::vector<SimpleAI::Data_Point> data;
	SimpleAI::Resource_Manager::load_mnist_data("Data/train-images.idx3-ubyte", "Data/train-labels.idx1-ubyte", data);


	std::cout << "Size of Dataset: " << data.size() << std::endl; 

	// train nn
	SimpleAI::AI_Manager manager(5); 
	
	MainLoop::start_loop(manager, data); 

	// start canvas with best nn
	MainLoop::start_canvas(*manager.best_instance, data);
	
	
	return 0; 
}

/*

todo: 
	implement multithreading

	Bias kleiner machen ~ 0.1

try: 
	try values between 0 and 1
	Erwartungswert = 0 -> Werte explodieren nicht, Gleichen sich aus (+ und -) 


softmax remember: exp leads to non proportional probability distributions


CHANGE ACTIVATION FUNCTION FOR OUTPUT LAYER TO SOFTMAX DERIVATIVE



What to use. Now to the last question, how does one choose which activation and cost functions to use. These advices will work for majority of cases:

	1. If you do classification, use softmax for the last layer's nonlinearity and cross entropy as a cost function.

	2. If you do regression, use sigmoid or tanh for the last layer's nonlinearity and squared error as a cost function.

	3. Use ReLU as a nonlienearity between layers.

	4. Use better optimizers (AdamOptimizer, AdagradOptimizer) instead of GradientDescentOptimizer, or use momentum for faster convergence,
*/