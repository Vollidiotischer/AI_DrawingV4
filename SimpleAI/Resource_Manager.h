#pragma once

#include <stdio.h>

#include <vector>
#include <array>
#include <fstream>
#include <string>

namespace SimpleAI {

	struct Resource_Manager {

		static void save_data(std::string path, std::vector<Data_Point>& data, std::ios_base::fmtflags flags) {

			/*
			Flags: 
				app -> Append to the end of the file
				trunc -> delete content, then write
			*/

			printf("Saving Data...\n"); 

			std::ofstream file(path, flags); 

			if (file.is_open()) {

				// fill document with new data (only difference -> ai layout has to be written to the beginning of the file) 
				if (flags == std::ostream::trunc) {
					// write ai layout to file
					file << ai_layout[0] << ":" << ai_layout[num_layers - 1] << "!\n"; 

				}

				// write data to file (append data) 
				for (auto& p : data) {
					for (int i = 0; i < p.data.size(); i++) {
						file << (int)p.data[i] << (i == p.data.size() - 1 ? '-' : ',');

					}
					for (int i = 0; i < p.result.size(); i++) {
						file << (int)p.result[i] << (i == p.result.size() - 1 ? '!' : ',');

					}
					file << "\n"; 
				}


				file.close(); 

				printf("Data Saved successfully\n");

			}
			else {
				// Error message, no exit
				fprintf(stderr, "Text File '%s' could not be opened [Line %d in function '%s']", path.c_str(), __LINE__, __func__);
				
			}
		}

		static void load_data(std::string path, std::vector<Data_Point>& data) {

			printf("Loading Data...\n"); 

			std::vector<std::string> parsed; 
			read_txt_file(path, parsed); 

			if (parsed.size() == 0) {
				printf("No training Data available in %s\n", path.c_str()); 
				return; 
			}

			std::array<DATA_TYPE, 2> ai_size;
			parse_data<2>(parsed[0], ai_size); 

			if (ai_size[0] != ai_layout[0] || ai_size[1] != ai_layout[num_layers - 1]) {

				fprintf(stderr, "AI size in Text file %s and in source code does not match(%d:%d and %d:%d) [Line %d in function '%s']", path.c_str(), (int)ai_size[0], (int)ai_size[1], ai_layout[0], ai_layout[num_layers - 1], __LINE__, __func__);
				system("pause");
				exit(1);
			}

			for (int i = 1; i < parsed.size(); i++) {
				data.push_back(Data_Point()); 
				parse_data(parsed[i], i, data[i - 1]); 
			}

			printf("Finished loading Data\n"); 
		}

		static void read_txt_file(std::string& path, std::vector<std::string>& result) {

			std::ifstream file; 
			file.open(path); 

			if (file.is_open()) {

				std::string temp_str; 

				while (std::getline(file, temp_str)) {

					if (temp_str.size() == 0) { continue;  }

					result.push_back(temp_str); 
				}

				file.close();
				
				printf("Loaded Data successfully\n"); 

			}
			else {
				fprintf(stderr, "Text File '%s' could not be opened [Line %d in function '%s']", path.c_str(), __LINE__, __func__); 
				system("pause");
				exit(1); 
			}

		}

		template <int arr_size>
		static void parse_data(const std::string& inp /* IN */, std::array<DATA_TYPE, arr_size>& result /* OUT */) {
			
			int last_pos = 0; 

			int data_index = 0; 

			for (int i = 0; i < inp.size(); i++) {
				if (inp[i] == ':') {
					result[0] = stof(inp.substr(0, i)); 
					result[1] = stof(inp.substr(i + 1));
					return; 
				}

				if (inp[i] == ',' || inp[i] == '-' || inp[i] == '!') {
					result[data_index] = stod(inp.substr(last_pos, (i - last_pos)));
					data_index++; 
					last_pos = i + 1; 
				}

			}
		}

		static void parse_data(const std::string& inp /* IN */, int line, SimpleAI::Data_Point& result /* OUT */) {
			
			int dash_pos = inp.find_first_of('-'); 
			int colon_pos = inp.find_first_of(':'); 

			if (dash_pos != std::string::npos && colon_pos == std::string::npos) {
				parse_data<ai_layout[0]>(inp.substr(0, dash_pos + 1), result.data); 
				parse_data<ai_layout[num_layers - 1]>(inp.substr(dash_pos + 1), result.result);
			}
			 
			if (dash_pos == std::string::npos && colon_pos != std::string::npos) {
				fprintf(stderr, "Tried to find Data-Size in wrong data-parse function (txt line %d) [Line %d in function '%s']", line, __LINE__, __func__);
				system("pause");
				exit(1); 
			}

			if (dash_pos == std::string::npos && colon_pos == std::string::npos) {
				fprintf(stderr, "Invalid string input (txt line %d) [Line %d in function '%s']", line, __LINE__, __func__);
				system("pause");
				exit(1); 
			}

		}

		static void load_mnist_data(std::string images_path, std::string labels_path, std::vector<Data_Point>& data) {
			
			std::vector<unsigned char> images_content; 
			std::vector<unsigned char> labels_content; 

			read_mnist_data(images_path, images_content); 
			read_mnist_data(labels_path, labels_content); 

			parse_mnist_data(images_content, labels_content, data);

		}

		static void read_mnist_data(std::string path, std::vector<unsigned char>& data) {

			printf("Loading Data...\n");

			std::ifstream file(path, std::ios::binary);

			if (file.is_open()) {

				data = std::vector<unsigned char>(std::istreambuf_iterator<char>(file), {});

				file.close();
				printf("Loaded Data successfully\n");

			}
			else {
				// error and exit
				fprintf(stderr, "Text File '%s' could not be opened [Line %d in function '%s']", path.c_str(), __LINE__, __func__);
				system("pause");
				exit(1);
			}
		}

		static void parse_mnist_data(std::vector<unsigned char>& image_content, std::vector<unsigned char>& labels_content, std::vector<Data_Point>& data) {
			// first 12 bytes are irrelevant
			/*
			
			images are 28 * 28 pixels large
				1 pixel = 1 byte
				0 (white), 255 (black)
				1 image -> 784 bytes
		
			Important: Transform white/black values to range [0, 1] !!!!
			*/


			int images_offset = 16; 
			int size = (image_content.size() - images_offset) / (28 * 28);

			int labels_offset = 8; 

			for (int i = 0; i < size; i++) {
				Data_Point dp; 
				
				for (int i2 = 0; i2 < (28 * 28); i2++) {

					dp.data[i2] = image_content[images_offset + i * (28 * 28) + i2] / 255.f;

				}

				unsigned char result = labels_content[labels_offset + i]; 

				for (int i2 = 0; i2 < 10; i2++) {
					dp.result[i2] = (i2 == result); 
				}

				data.push_back(dp); 

			}
		}

	};

}