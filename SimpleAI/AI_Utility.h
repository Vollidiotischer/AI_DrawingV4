#pragma once

#include <iostream>

#include <vector>


namespace SimpleAI {

	template<typename T, size_t size>
	static void print_array(std::array<T, size>& arr) {
	
		std::cout << "["; 
		for (int i = 0; i < size; i++) {
			std::cout << arr[i] << (i < size - 1 ? " " : "]");

		}
	
	}

	template<typename T, size_t size>
	static void println_array(std::array<T, size>& arr) {

		print_array<T, size>(arr); 

		std::cout << std::endl;

	}

	template<typename T>
	static void print_vector(std::vector<T>& arr, int num_tabs) {

		for (int i = 0; i < num_tabs; i++) {
			std::cout << "\t"; 
		}

		std::cout << "[";
		for (int i = 0; i < arr.size(); i++) {
			std::cout << arr[i] << (i < arr.size() - 1 ? " " : "]");

		}
		 
	}

	template<typename T>
	static void println_vector(std::vector<T>& arr, int num_tabs) {
		print_vector<T>(arr, num_tabs); 
		std::cout << std::endl; 
	}

	static void print_vector_weight(std::vector<SimpleAI::Weight>& arr, int num_tabs) {

		for (int i = 0; i < num_tabs; i++) {
			std::cout << "\t";
		}

		std::cout << "[";
		for (int i = 0; i < arr.size(); i++) {
			std::cout << arr[i].current_weight << " (" << arr[i].delta_change << ")" << (i < arr.size() - 1 ? ", " : "]");

		}


	}
	
	static void println_vector_weight(std::vector<SimpleAI::Weight>& arr, int num_tabs) {

		print_vector_weight(arr, num_tabs); 

		std::cout << std::endl;

	}

	static void print_vector_bias(std::vector<SimpleAI::Bias>& arr, int num_tabs) {

		for (int i = 0; i < num_tabs; i++) {
			std::cout << "\t";
		}

		std::cout << "[";
		for (int i = 0; i < arr.size(); i++) {
			std::cout << arr[i].current_bias << " (" << arr[i].delta_change << ")" << (i < arr.size() - 1 ? ", " : "]");

		}


	}

	static void println_vector_bias(std::vector<SimpleAI::Bias>& arr, int num_tabs) {

		print_vector_bias(arr, num_tabs);

		std::cout << std::endl;

	}


}