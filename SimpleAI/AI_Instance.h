#pragma once

#include <stdio.h>

#include <array>
#include <vector>
#include <random>
#include <time.h>

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

		*/
		std::array<std::vector<std::vector<Weight>>, num_layers - 1> weights;
		std::array<std::vector<Bias>, num_layers - 1> biases;
		std::array<std::vector<Neuron>, num_layers> neurons;

		DATA_TYPE error = 0.f; 


		AI_Instance(DATA_TYPE learn_factor) {

			this->learn_factor = learn_factor; 

			// initialize neurons
			for (int i = 0; i < num_layers; i++) {
				neurons[i].resize(ai_layout[i]);
			}


			// initialize weights
			init_weights();

			// initialize biases
			init_biases();
		}


		void print_instance() {
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
		}

		void print_error(std::string app = "") {
			std::cout << "Error: " << error * 100.f << "%" << app;
		}

		void init_weights() {
			for (int i = 0; i < weights.size(); i++) {
				// number of rows in the matrix (one row for each neuron in the following layer) 
				weights[i] = std::vector<std::vector<Weight>>(ai_layout[i + 1], std::vector<Weight>(ai_layout[i], 0));

				// Initialize weights to random value
				for (int i2 = 0; i2 < weights[i].size(); i2++) {
					for (int i3 = 0; i3 < weights[i][i2].size(); i3++) {
						weights[i][i2][i3].current_weight = rd.get_random_number() / sqrt((DATA_TYPE)neurons[i].size());
					}
				}

			}
		}

		void init_biases() {
			for (int i = 0; i < biases.size(); i++) {
				biases[i] = std::vector<Bias>(ai_layout[i + 1], 0);

				// Initialize to random Value
				for (int i2 = 0; i2 < biases[i].size(); i2++) {
					biases[i][i2].current_bias = rd.get_random_number();
				}
			}
		}

		static void clear_weight_delta_value(std::array<std::vector<std::vector<Weight>>, num_layers - 1>& weights) {

			for (auto& w1 : weights) {
				for (auto& w2 : w1) {
					for (auto& w3 : w2) {
						w3.delta_change = 0.f;
					}
				}
			}

		}

		static void clear_biases_delta_value(std::array<std::vector<Bias>, num_layers - 1>& biases) {

			for (auto& b1 : biases) {
				for (auto& b2 : b1) {
					b2.delta_change = 0.f;
				}
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

			if (data.size() != ai.neurons[0].size()) {
				fprintf(stderr, "Data size (%d) does not match Input Layer size (%d) [Line %d in function '%s']", data.size(), ai.neurons[0].size(), __LINE__, __func__);
				system("pause");
				exit(1);
			}

			// copy data to neurons (may be optimised out)
			for (int i2 = 0; i2 < ai.neurons[0].size(); i2++) {
				ai.neurons[0][i2].a = data[i2];
			}

			// Feed data to AI
			for (int i = 0; i < num_layers - 1; i++) {
				zero_out(ai.neurons[i + 1]);
				matrix_vector_multiply(ai.neurons[i], ai.weights[i], ai.neurons[i + 1]);
				vect_vect_add(ai.neurons[i + 1], ai.biases[i]);
				if (i == num_layers - 2) { // output layer -> softmax
					apply_softmax_function(ai.neurons[i + 1]);
				}
				else { // hidden layer -> ReLU
					apply_activation_function(ai.neurons[i + 1]);
				}
			}

			// Copy elements from last neuron layer to result vector (may be optimised out)
			for (int i = 0; i < result.size(); i++) {
				result[i] = ai.neurons.back()[i].a;
			}

		}

		static void backprop(AI_Instance& ai, std::vector<Data_Point>& data_list, int start_index, int end_index) {

			ai.error = 0.f;

			// clear weight delta values
			clear_weight_delta_value(ai.weights);
			// clear bias delta values
			clear_biases_delta_value(ai.biases);

			for (int i = start_index; i < end_index; i++) {
				feed_forward_step(ai, data_list[i]);

				backprop_step(ai, data_list[i]);

			}

			float data_size = end_index - start_index;

			// apply delta_weight changes
			for (auto& w1 : ai.weights) {
				for (auto& w2 : w1) {
					for (auto& w3 : w2) {
						w3.current_weight += ai.learn_factor * (w3.delta_change / data_size);
					}
				}
			}


			// apply delta_bias changes
			for (auto& b1 : ai.biases) {
				for (auto& b2 : b1) {
					b2.current_bias += ai.learn_factor * (b2.delta_change / data_size);
				}
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

			// only runs through output layer and all hidden layers (not the input layer)
			for (int i = ai.neurons.size() - 1; i > 0; i--) {

				// output layer: (1)
				if (i == ai.neurons.size() - 1) {

					for (int i2 = 0; i2 < ai.neurons[i].size(); i2++) {

						// 1.1
						ai.neurons[i][i2].delta_value = data.result[i2] - ai.neurons[i][i2].a; // softmax with cross entropy (softmax has been applied to the final layer, (result - prediction) is the equation for this activation function, cost function combination) 

						// weights (1.2)
						for (int i3 = 0; i3 < ai.neurons[i - 1].size(); i3++) {

							ai.weights[i - 1][i2][i3].delta_change += ai.neurons[i][i2].delta_value * ai.neurons[i - 1][i3].a;

						}

						// biases (1.3) 
						ai.biases[i - 1][i2].delta_change += ai.neurons[i][i2].delta_value;

					}

				}
				else { // hidden layers: (2)

					for (int i2 = 0; i2 < ai.neurons[i].size(); i2++) {
						DATA_TYPE z_hid = ai.neurons[i][i2].z;

						// gather weight, delta neuron pairs for neuron delta value (2.1) 
						std::vector<std::array<DATA_TYPE, 2>> delta_pairs;

						for (int i3 = 0; i3 < ai.weights[i].size(); i3++) {
							std::array<DATA_TYPE, 2> delta_pair = { ai.weights[i][i3][i2].current_weight, ai.neurons[i + 1][i3].delta_value };
							delta_pairs.push_back(delta_pair);

						}
						// 2.1
						ai.neurons[i][i2].delta_value = delta_function_hidden_layer(z_hid, delta_pairs);

						// weights (2.2)
						for (int i3 = 0; i3 < ai.weights[i - 1][i2].size(); i3++) {
							ai.weights[i - 1][i2][i3].delta_change += ai.neurons[i][i2].delta_value * ai.neurons[i - 1][i3].a;
						}

						// biases (2.3) 
						ai.biases[i - 1][i2].delta_change += ai.neurons[i][i2].delta_value;
					}

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

		static DATA_TYPE delta_function_hidden_layer(DATA_TYPE z_hid, std::vector<std::array<DATA_TYPE, 2>>& delta_pairs) {

			/*
				IN:
					z_hid: z vom aktuellen hidden neuron (Beim Bias 1)
					delta_pairs: pair of delta_value and weight
						delta_value: delta_value of neuron which is connected to the current neuron when viewing the weight
						weight: the weight which connects the current neuron to the neuron with the delta value in this pair (Beim Bias den Bias)

				Formula: f'(z_hid) * Sum(delta_value * weight)
			*/

			DATA_TYPE delta = activation_function_derivative(z_hid);

			DATA_TYPE sum = 0;

			for (int i = 0; i < delta_pairs.size(); i++) {
				sum += delta_pairs[i][0] * delta_pairs[i][1];
			}

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

		static void apply_softmax_function(std::vector<Neuron>& inout) {

			DATA_TYPE sum = 0;


			DATA_TYPE max = inout[0].z;

			for (int i = 0; i < inout.size(); i++) {
				if (inout[i].z > max) {
					max = inout[i].z;
				}
			}


			for (int i = 0; i < inout.size(); i++) {
				inout[i].z -= max;
				sum += std::exp(inout[i].z);
			}

			for (int i = 0; i < inout.size(); i++) {
				inout[i].a = (std::exp(inout[i].z) / sum);
			}


		}

		static void apply_activation_function(std::vector<Neuron>& vect /*IN / OUT */) {

			for (int i = 0; i < vect.size(); i++) {
				vect[i].a = activation_function(vect[i].z);
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
