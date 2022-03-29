#pragma once

#include <vector>
#include <random>


namespace SimpleAI {

	struct AI_Manager {

		int num_instances; 

		std::vector<AI_Instance> ai_list; 

		AI_Instance *best_instance = NULL; 



		AI_Manager(int num_instances) : num_instances(num_instances) {

			srand(time(NULL)); 
			std::cout << "[AI_Manager] Creating instances..." << std::endl; 

			for (int i = 0; i < num_instances; i++) {
				ai_list.push_back(AI_Instance(ai_learn_factor)); 
			}

			calculate_best_instance(); 

			std::cout << "[AI_Manager] Finished creating Instances" << std::endl; 

		}

		void train_all_instances(std::vector<Data_Point>& data, int batch_size) {
			
			std::vector<int> indices(batch_size); 

			get_random_indeces(indices, data.size()); 

			for (int i2 = 0; i2 < ai_list.size(); i2++) {
				ai_list[i2].backprop(data, indices);
			}


		}
		/*
		void train_all_instances(std::vector<Data_Point>& data, DATA_TYPE goal_percent) {

			goal_percent /= 100.f;

			DATA_TYPE minimum = 1.f; 

			while (minimum > goal_percent){
				for (int i2 = 0; i2 < ai_list.size(); i2++) {

					ai_list[i2].backprop(data);

					if (ai_list[i2].error < minimum) {
						minimum = ai_list[i2].error;
						std::cout.precision(4); 
						std::cout << "Error: " << minimum * 100.f << "% / " << goal_percent * 100.f << "%" << std::endl;

					} 
				}

			}

		}
		*/

		void reshuffel_instances() {

			int pos = calculate_best_instance(); 

			for (int i = 0; i < ai_list.size(); i++) {
				if (i != pos) {
					ai_list[i].init_biases(); 
					ai_list[i].init_weights();
				}
			}
		}

		void evaluate_instances(std::vector<Data_Point>& data) {

			for (int i = 0; i < num_instances; i++) {
				std::cout << "Instance: " << i << "/" << num_instances << std::endl;
				// evaluate data for each ai instance, internally set error variable
				ai_list[i].evaluate_input_list(data); 

			}

		}

		int calculate_best_instance() {

			int pos = 0; 
			DATA_TYPE min_score = ai_list[0].error; 

			for (int i = 1; i < ai_list.size(); i++) {
				if (ai_list[i].error < min_score) {
					min_score = ai_list[i].error;

					pos = i; 

				}
			}

			best_instance = &ai_list[pos]; 

			return pos; 
		}

	private:

		void get_random_indeces(std::vector<int>& indices, int max_index){
		
			//std::cout << "Generating " << indices.size() << " Random Numbers..." << std::endl; 

			int i = 0; 

			while (i < indices.size()) {


				int random_number = (rand() / (float)RAND_MAX) * max_index;

				if (std::find(indices.begin(), indices.end(), random_number) == indices.end()) {
					indices[i] = random_number;
					i++; 
				}
			}

			//std::cout << "Finished generating " << indices.size() << " Random Numbers" << std::endl;

		
		}
	};

}