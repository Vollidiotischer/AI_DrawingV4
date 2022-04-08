#pragma once

#include <stdio.h>

#include <array>
#include <vector>
#include <random>
#include <time.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "AI_Variables.h"
#include "AI_Functionality.h"


namespace SimpleAI {

	namespace {
		struct Random_Device {
			std::default_random_engine engine;
			std::normal_distribution<DATA_TYPE> dist;

			Random_Device() : dist(erwartungswert, standardabweichung), engine(time(NULL)) {}

			DATA_TYPE get_random_number() {

				return dist(engine);
			}

		};
	}


	struct AI_Instance {

		static Random_Device rd;

		DATA_TYPE learn_factor = ai_learn_factor; 

		/*

		Weights is a List of num_layers - 1 matrices
		weights[i] = std::vector<std::vector<Weight>>(ai_layout[i + 1], std::vector<Weight>(ai_layout[i], 0));
		*/

		std::array<Eigen::MatrixX<DATA_TYPE>, num_layers - 1> weights_current_weight;
		std::array<Eigen::MatrixX<DATA_TYPE>, num_layers - 1> weights_delta_value;

		std::array<Eigen::VectorX<DATA_TYPE>, num_layers - 1> biases_current_bias;
		std::array<Eigen::VectorX<DATA_TYPE>, num_layers - 1> biases_delta_value;

		std::array<Eigen::VectorX<DATA_TYPE>, num_layers> neurons_a;
		std::array<Eigen::VectorX<DATA_TYPE>, num_layers> neurons_z;
		std::array<Eigen::VectorX<DATA_TYPE>, num_layers> neurons_delta;

		DATA_TYPE error = 0.f; 


		AI_Instance(DATA_TYPE learn_factor) {

			this->learn_factor = learn_factor; 

			// initialize neurons
			for (int i = 0; i < num_layers; i++) {
				neurons_a[i].resize(ai_layout[i]); 
				neurons_z[i].resize(ai_layout[i]); 
				neurons_delta[i].resize(ai_layout[i]); 

			}


			// initialize weights
			init_weights();

			// initialize biases
			init_biases();

		}

		// todo
		void print_instance() {
			/*
			std::cout << "Status of the ai: " << std::endl;
			for (int i = 0; i < neurons.size(); i++) {

				std::cout << "a: [";

				for (int i2 = 0; i2 < neurons[i].size(); i2++) {
					std::cout << neurons[i][i2].a << (i2 < neurons[i].size() - 1 ? ", " : "]");
				}
				std::cout << std::endl;

				std::cout << "z: [";

				for (int i2 = 0; i2 < neurons[i].size(); i2++) {
					std::cout << neurons[i][i2].z << (i2 < neurons[i].size() - 1 ? ", " : "]");
				}

				std::cout << std::endl; 

				std::cout << "d: [";

				for (int i2 = 0; i2 < neurons[i].size(); i2++) {
					std::cout << neurons[i][i2].delta_value << (i2 < neurons[i].size() - 1 ? ", " : "]");
				}
				std::cout << std::endl << std::endl;
			}

			std::cout << "Error of the ai: " << error << std::endl << std::endl;

			std::cout << "Weights of the ai: " << std::endl;

			std::cout << "Weights_0: [" << std::endl;
			for (auto& s1 : weights) {
				std::cout << "\tWeights_1: [" << std::endl;
				for (auto& s2 : s1) {
					SimpleAI::println_vector_weight(s2, 2);
				}
				std::cout << "\t]" << std::endl;
			}
			std::cout << "]" << std::endl << std::endl;

			std::cout << "Biases of the ai: " << std::endl;

			std::cout << "Biases_0: [" << std::endl;

			for (auto& s1 : biases) {
				SimpleAI::println_vector_bias(s1, 1);
			}

			std::cout << "]" << std::endl;
			*/
		}

		void print_error(std::string app = "") {
			std::cout << "Error: " << error * 100.f << "%" << app;
		}

		void init_weights() {
			for (int i = 0; i < weights_current_weight.size(); i++) {
				// number of rows in the matrix (one row for each neuron in the following layer) 
				weights_current_weight[i] = Eigen::MatrixX<DATA_TYPE>(ai_layout[i+1], ai_layout[i]); 
				weights_delta_value[i] = Eigen::MatrixX<DATA_TYPE>(ai_layout[i+1], ai_layout[i]); 
				
				// Initialize weights to random value
				for (int i2 = 0; i2 < weights_current_weight[i].rows(); i2++) {
					for (int i3 = 0; i3 < weights_current_weight[i].cols(); i3++) {
						weights_current_weight[i](i2, i3) = rd.get_random_number() / sqrt((DATA_TYPE)ai_layout[i]);
					
					}
				}

			}
		}

		void init_biases() {
			for (int i = 0; i < biases_current_bias.size(); i++) {
				biases_current_bias[i] = Eigen::VectorX<DATA_TYPE>(ai_layout[i+1]); 
				biases_delta_value[i] = Eigen::VectorX<DATA_TYPE>(ai_layout[i+1]); 

				// Initialize to random Value
				for (int i2 = 0; i2 < biases_current_bias[i].size(); i2++) {
					biases_current_bias[i](i2) = rd.get_random_number(); 
				}
			}
		}

		static void clear_weight_delta_value(std::array<Eigen::MatrixX<DATA_TYPE>, num_layers - 1>& weights_delta_change) {

			for (auto& w1 : weights_delta_change) {
				w1.setZero(); 
			}

		}

		static void clear_biases_delta_value(std::array<Eigen::VectorX<DATA_TYPE>, num_layers - 1>& biases_delta_change) {

			for (auto& b1 : biases_delta_change) {
				b1.setZero();
			}

		}

		static void evaluate_input_list(AI_Instance& ai, std::vector<Data_Point>& data) {

			ai.error = 0.f;

			for (int i = 0; i < data.size(); i++) {
				// Feed data through AI
				feed_forward_step(ai, data[i]);

			}

			// average out the error
			ai.error /= (DATA_TYPE)data.size();

		}

		static void feed_forward_step(AI_Instance& ai, Data_Point& dp) {

			std::array<DATA_TYPE, ai_layout[num_layers - 1]> result;
			evaluate_input(ai, dp.data, result); // static 

			ai.error += cost_function(result, dp.result); // static

		}

		static void evaluate_input(AI_Instance& ai, const std::array<DATA_TYPE, ai_layout[0]>& data /* IN */, std::array<DATA_TYPE, ai_layout[num_layers - 1]>& result /* OUT */) {

			if (data.size() != ai.neurons_a[0].size()) {
				fprintf(stderr, "Data size (%d) does not match Input Layer size (%d) [Line %d in function '%s']", data.size(), ai.neurons_a[0].size(), __LINE__, __func__);
				system("pause");
				exit(1);
			}

			// copy data to neurons (may be optimised out)
			for (int i2 = 0; i2 < ai.neurons_a[0].size(); i2++) {
				ai.neurons_a[0](i2) = data[i2];
			}

			// Feed data to AI
			for (int i = 0; i < num_layers - 1; i++) {

				ai.neurons_a[i].setZero(); 
				ai.neurons_z[i].setZero(); 
				//zero_out(ai.neurons[i + 1]);

				//matrix_vector_multiply(ai.neurons[i], ai.weights[i], ai.neurons[i + 1]);
				ai.neurons_z[i + 1] = ai.weights_current_weight[i] * ai.neurons_a[i]; 

				//vect_vect_add(ai.neurons[i + 1], ai.biases[i]);
				ai.neurons_z[i + 1] = ai.neurons_z[i + 1] + ai.biases_current_bias[i]; 

				if (i == num_layers - 2) { // output layer -> softmax
					apply_softmax_function(ai.neurons_z[i + 1], ai.neurons_a[i +1 ]);
				}
				else { // hidden layer -> ReLU
					apply_activation_function(ai.neurons_z[i + 1], ai.neurons_a[i + 1]);
				}
			}

			// Copy elements from last neuron layer to result vector (may be optimised out)
			for (int i = 0; i < result.size(); i++) {
				result[i] = ai.neurons_a.back()(i); 
			}

		}

		static void backprop(AI_Instance& ai, std::vector<Data_Point>& data_list, int start_index, int end_index) {

			ai.error = 0.f;

			// clear weight delta values
			clear_weight_delta_value(ai.weights_delta_value);
			// clear bias delta values
			clear_biases_delta_value(ai.biases_delta_value);


			for (int i = start_index; i < end_index; i++) {

				feed_forward_step(ai, data_list[i]);

				backprop_step(ai, data_list[i]);

			}

			float data_size = end_index - start_index;

			// apply delta_weight changes
			for (int i = 0; i < ai.weights_current_weight.size(); i++) {
				ai.weights_current_weight[i] = ai.weights_current_weight[i] + ai.weights_delta_value[i];

			}


			// apply delta_bias changes
			for (int i = 0; i < ai.biases_current_bias.size(); i++) {
				ai.biases_current_bias[i] = ai.biases_current_bias[i] + ai.biases_delta_value[i]; 
			}

			// average out the error
			ai.error /= (DATA_TYPE)data_list.size();
		}

		static void backprop_step(AI_Instance& ai, Data_Point& data) {

			/*

			Tasks:
				1. Output Layer: 
					1.1. Calculate delta values for output neurons
					1.2. Caluclate delta Weight for the Weights connected to the output layer
					1.3. Caluclate delta Biases for the Biases connected to the output layer
				2. Hidden Layers: 
					2.1. Caluclate delta Value for hidden neurons
					2.2. Calculate delta Weight for all remaining Weights
					2.3. Caluclate delta Biases for all remaining Biases
			
			
			Delta Weight Formula: delta_weight = (epsilon) * (delta) * (activation of previous layer)
			*/

			// convert Data_Point result array to Eigen Vector
			Eigen::VectorX<DATA_TYPE> result_vect(data.result.size()); 
			for (int i = 0; i < data.result.size(); i++) {
				result_vect(i) = data.result[i]; 
			}

			// only runs through output layer and all hidden layers (not the input layer)
			for (int i = num_layers - 1; i > 0; i--) {
				
				if (i == num_layers - 1) {
					// output layer: (1)

					// 1.1
					ai.neurons_delta[i] = result_vect - ai.neurons_a[i]; // softmax with cross entropy

					// 1.2
					ai.weights_delta_value[i - 1] += ai.neurons_delta[i] * ai.neurons_a[i - 1].transpose();

					// 1.3
					ai.biases_delta_value[i - 1] += ai.neurons_delta[i]; 

				}
				else { // hidden layers: (2)

					for (int i2 = 0; i2 < ai_layout[i]; i2++) {

						DATA_TYPE z_hid = ai.neurons_z[i](i2);

						// gather weight, delta neuron pairs for neuron delta value (2.1) 

						DATA_TYPE sum = 0;
						for (int i3 = 0; i3 < ai.weights_current_weight[i].rows(); i3++) {
							sum += ai.weights_current_weight[i](i3, i2) * ai.neurons_delta[i + 1](i3);

						}

						// 2.1
						ai.neurons_delta[i](i2) = delta_function_hidden_layer(z_hid, sum);

					}
					// weights (2.2)
					ai.weights_delta_value[i - 1] += ai.neurons_delta[i] * ai.neurons_a[i - 1].transpose();

					// biases (2.3) 
					ai.biases_delta_value[i - 1] += ai.neurons_delta[i]; 


				}

			}

		}

private:

		static DATA_TYPE delta_function_output_layer(DATA_TYPE z_hid, DATA_TYPE a_soll, DATA_TYPE a_ist) {

			// CURRENTLY NOT IN USE

			/*
				IN:
					inp: z von dem aktuellen neuron (z von dem output neuron)
					a_soll: a(sollwert)
					a_ist: a(istwert)

				Formula: f'(inp) * (a_soll - a_ist)
			*/

			DATA_TYPE der_of_cost_res = (a_soll - a_ist); 

			return activation_function_derivative(z_hid) * der_of_cost_res;
		}

		static DATA_TYPE delta_function_hidden_layer(DATA_TYPE z_hid, DATA_TYPE sum) {

			/*
				IN:
					z_hid: z vom aktuellen hidden neuron (Beim Bias 1)
					delta_pairs: pair of delta_value and weight
						delta_value: delta_value of neuron which is connected to the current neuron when viewing the weight
						weight: the weight which connects the current neuron to the neuron with the delta value in this pair (Beim Bias den Bias)

				Formula: f'(z_hid) * Sum(delta_value * weight)
			*/

			DATA_TYPE delta = activation_function_derivative(z_hid);

			return delta * sum;
		}

		static DATA_TYPE activation_function(DATA_TYPE x) {

			return std::max(0.01f * x, x); // Leaky ReLU

			//return 1.f / (1.f + exp(-x)); // Sigmoid
		}

		static DATA_TYPE activation_function_derivative(DATA_TYPE x) {
			
			return (x <= 0.f ? 0.01f : 1.f); // Leaky ReLU
			
			//return activation_function(x) * (1.f - activation_function(x)); // Sigmoid
			
		}

		static void zero_out(std::vector<Neuron>& vect1 /* IN / OUT */) {
			for (int i = 0; i < vect1.size(); i++) {
				vect1[i].a = 0.f;
				vect1[i].z = 0.f;
				vect1[i].delta_value = 0.f;
			}
		}

		static void matrix_vector_multiply(const std::vector<Neuron>& vect /* IN */, const std::vector<std::vector<Weight>>& matrix /* IN */, std::vector<Neuron>& result /* OUT */) {

			if (matrix.size() != result.size()) {
				fprintf(stderr, "Matrix size (%d) does not match Result size (%d) [Line %d in function '%s']", matrix.size(), result.size(), __LINE__, __func__);
				system("pause"); 
				exit(1);
			}

			for (int i = 0; i < matrix.size(); i++) {

				if (matrix[i].size() != vect.size()) {
					fprintf(stderr, "Matrix[i] size (%d) does not match Vect size (%d) [Line %d in function '%s']", matrix[i].size(), vect.size(), __LINE__, __func__);
					system("pause");
					exit(1);
				}

				for (int i2 = 0; i2 < matrix[i].size(); i2++) {
					result[i].z += matrix[i][i2].current_weight * vect[i2].a;
				}

			}

		}

		static void vect_vect_add(std::vector<Neuron>& vect1 /* IN / OUT */, const std::vector<Bias>& vect2 /* IN */) {

			if (vect1.size() != vect2.size()) {
				fprintf(stderr, "Vector sizes do not match (%d and %d) [Line %d in function '%s']", vect1.size(), vect2.size(), __LINE__, __func__);
			}

			for (int i = 0; i < vect1.size(); i++) {
				vect1[i].z += vect2[i].current_bias;
			}
		}

		static void apply_softmax_function(Eigen::VectorX<DATA_TYPE> neurons_z, Eigen::VectorX<DATA_TYPE>& neurons_a) {

			DATA_TYPE sum = 0;

			// Find max element in neurons_z
			DATA_TYPE max = neurons_z.maxCoeff();


			for (int i = 0; i < neurons_z.size(); i++) {
				neurons_z(i) = std::exp(neurons_z(i) - max);
				sum += neurons_z(i);
			}

			for (int i = 0; i < neurons_z.size(); i++) {
				neurons_a(i) = neurons_z(i) / sum;
			}


		}

		static void apply_activation_function(const Eigen::VectorX<DATA_TYPE>& neurons_z, Eigen::VectorX<DATA_TYPE>& neurons_a) {

			for (int i = 0; i < neurons_z.size(); i++) {
				neurons_a(i) = activation_function(neurons_z(i));
			}

		}

		// Calculate the cost for a single result of a single data point
		static DATA_TYPE cost_function(std::array<DATA_TYPE, ai_layout[num_layers - 1]>& calculated_result, std::array<DATA_TYPE, ai_layout[num_layers - 1]>& expected_result) {
			// Squared Sum
			/*
			DATA_TYPE squared_sum = 0.f;

			for (int i = 0; i < ai_layout[num_layers - 1]; i++) {
				squared_sum += (calculated_result[i] - expected_result[i]) * (calculated_result[i] - expected_result[i]);
			}

			return squared_sum;
			*/

			// Cross Entropy
			DATA_TYPE sum = 0.f;

			for (int i = 0; i < calculated_result.size(); i++) {
				sum -= expected_result[i] * log(calculated_result[i] + 0.001);
			}

			return sum; 
		}



	};

	// Initialize static Random_Device
	Random_Device AI_Instance::rd;

}
