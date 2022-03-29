#pragma once

#include <stdio.h>

#include <array>
#include <vector>
#include <random>
#include <time.h>

#include "AI_Variables.h"


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

		DATA_TYPE learn_factor = 1.f; 

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

		
		void evaluate_input(const std::array<DATA_TYPE, ai_layout[0]>& data /* IN */, std::array<DATA_TYPE, ai_layout[num_layers - 1]>& result /* OUT */) {

			if (data.size() != neurons[0].size()) {
				fprintf(stderr, "Data size (%d) does not match Input Layer size (%d) [Line %d in function '%s']", data.size(), neurons[0].size(), __LINE__, __func__);
				exit(1); 
			}

			// copy data to neurons (may be optimised out)
			for (int i2 = 0; i2 < neurons[0].size(); i2++) {
				neurons[0][i2].a = data[i2];
			}

			// Feed data to AI
			for (int i = 0; i < num_layers - 1; i++) {
				zero_out(neurons[i + 1]);
				matrix_vector_multiply(neurons[i], weights[i], neurons[i + 1]);
				vect_vect_add(neurons[i + 1], biases[i]);
				apply_activation_function(neurons[i + 1]);
			}

			// Copy elements from last neuron layer to result vector (may be optimised out)
			for (int i = 0; i < result.size(); i++) {
				result[i] = neurons.back()[i].a;
			}

		}
 
		void evaluate_input_list(std::vector<Data_Point>& data) {

			this->error = 0.f; 

			for (int i = 0; i < data.size(); i++) {
				// Feed data through AI
				feed_forward_step(data[i]); 

			}

			// average out the error
			this->error /= (DATA_TYPE)data.size();

		}

		void backprop(std::vector<Data_Point>& data_list, std::vector<int>& indices) {

			this->error = 0.f;

			// clear weight delta values
			clear_weight_delta_value(this->weights);
			// clear bias delta values
			clear_biases_delta_value(this->biases);

			for (int i = 0; i < indices.size(); i++) {
				feed_forward_step(data_list[indices[i]]); 

				backprop_step(data_list[indices[i]]);

			}

			// apply delta_weight changes
			for (auto& w1 : weights) {
				for (auto& w2 : w1) {
					for (auto& w3 : w2) {
						w3.current_weight += learn_factor * w3.delta_change;
					}
				}
			}

			// apply delta_bias changes
			for (auto& b1 : biases) {
				for (auto& b2 : b1) {
					b2.current_bias += learn_factor * b2.delta_change;
				}
			}

			// average out the error
			this->error /= (DATA_TYPE)data_list.size();

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

private:

		void feed_forward_step(Data_Point& data) {
			// FEED FORWARD STEP
			// run data through ai
			std::array<DATA_TYPE, ai_layout[num_layers - 1]> result;
			this->evaluate_input(data.data, result);

			// calculate error for this data / this ai
			this->error += cost_function(result, data.result);

		}

		void backprop_step(Data_Point& data) {
			/*
			
			Tasks: 
				1. clear delta weights
				2. Caluclate delta Weight for the Weights connected to the output layer
				3. Calculate delta Weight for all remaining Weights
				TODO: 4. Calculate Delta Biases
				x. Add Delta weight to weights

			Delta Weight Formula: delta_weight = (epsilon) * (delta) * (activation of previous layer)
			*/

			
			// only runs through output layer and all hidden layers (not the input layer)
			for (int i = neurons.size() - 1; i > 0; i--) {

				// output layer: 
				if (i == neurons.size() - 1) {

					for (int i2 = 0; i2 < neurons[i].size(); i2++) {

						DATA_TYPE z_hid = neurons[i][i2].z;

						neurons[i][i2].delta_value = delta_function_output_layer(z_hid, data.result[i2], neurons[i][i2].a);

						// weights
						for (int i3 = 0; i3 < neurons[i-1].size(); i3++) {
							
							weights[i-1][i2][i3].delta_change += neurons[i][i2].delta_value * neurons[i - 1][i3].a;
							
						}

						// biases
						biases[i-1][i2].delta_change += neurons[i][i2].delta_value;

					}

				}
				else { // hidden layers: 

					for (int i2 = 0; i2 < neurons[i].size(); i2++) {
						DATA_TYPE z_hid = neurons[i][i2].z;

						std::vector<std::array<DATA_TYPE, 2>> delta_pairs; 

						for (int i3 = 0; i3 < weights[i].size(); i3++) {
							std::array<DATA_TYPE, 2> delta_pair = { weights[i][i3][i2].current_weight, neurons[i+1][i3].delta_value };
							delta_pairs.push_back(delta_pair);

						}

						neurons[i][i2].delta_value = delta_function_hidden_layer(z_hid, delta_pairs);

						// weights
						for (int i3 = 0; i3 < weights[i - 1][i2].size(); i3++) {
							weights[i - 1][i2][i3].delta_change += neurons[i][i2].delta_value * neurons[i - 1][i3].a;
						}

						// biases
						biases[i - 1][i2].delta_change += neurons[i][i2].delta_value;
					}

				}

			}


		}


		static DATA_TYPE delta_function_output_layer(DATA_TYPE z_hid, DATA_TYPE a_soll, DATA_TYPE a_ist) {

			/*
				IN:
					inp: z von dem aktuellen neuron (z von dem output neuron)
					a_soll: a(sollwert)
					a_ist: a(istwert)

				Formula: f'(inp) * (a_soll - a_ist)
			*/

			return activation_function_derivative(z_hid) * (a_soll - a_ist);
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
			

			/*
			if (x > 0.1) {
				return x; 
			}

			if (x < -0.1) {
				return 0.01 * x; 
			}
			else {
				return 1; 
			}*/
			return 1.f / (1.f + exp(-x)); 
		}

		static DATA_TYPE activation_function_derivative(DATA_TYPE x) {
			return activation_function(x) * (1.f - activation_function(x)); 
			/*
			if (x < -0.1) {
				return 0.01; 
			}
			else{
				return 1; 
			}*/
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
				exit(1); 
			}

			for (int i = 0; i < matrix.size(); i++) {

				if (matrix[i].size() != vect.size()) {
					fprintf(stderr, "Matrix[i] size (%d) does not match Vect size (%d) [Line %d in function '%s']", matrix[i].size(), vect.size(), __LINE__, __func__);
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

		static void apply_activation_function(std::vector<Neuron>& vect /*IN / OUT */) {

			for (int i = 0; i < vect.size(); i++) {
				vect[i].a = activation_function(vect[i].z);
			}

		}

		// Calculate the cost for a single result of a single data point
		static DATA_TYPE cost_function(std::array<DATA_TYPE, ai_layout[num_layers - 1]>& calculated_result, std::array<DATA_TYPE, ai_layout[num_layers - 1]>& expected_result) {

			DATA_TYPE squared_sum = 0.f;

			for (int i = 0; i < ai_layout[num_layers - 1]; i++) {
				squared_sum += (calculated_result[i] - expected_result[i]) * (calculated_result[i] - expected_result[i]);
			}

			return squared_sum;

		}
	};

	// Initialize static Random_Device
	Random_Device AI_Instance::rd;

}
