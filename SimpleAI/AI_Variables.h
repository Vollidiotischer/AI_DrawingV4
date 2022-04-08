#pragma once

#include <array>

/*

num_layers: 
	Mindestens 2 (Input - & Output Layer), alle weiteren sind hidden layers

ai_layout: 
	Number of Neurons per Layer

*/ 

#define DATA_TYPE float


namespace SimpleAI {

	constexpr int num_layers = 6;
	constexpr std::array<int, num_layers> ai_layout = { 784, 512, 256, 128, 20, 10 };

	constexpr DATA_TYPE ai_learn_factor = 0.01f;

	constexpr float erwartungswert = 0.f;
	constexpr float standardabweichung = 3.f;


	struct Data_Point {

		std::array<DATA_TYPE, ai_layout[0]> data;
		std::array<DATA_TYPE, ai_layout[num_layers - 1]> result;

	};

	struct Neuron {

		DATA_TYPE z = 0; // ohne aktivierungsfunktion
		DATA_TYPE a = 0; // mit aktivierungsfunktion 

		DATA_TYPE delta_value = 0; // the delta value calculated in backpropagation

	};

	struct Weight {

		DATA_TYPE current_weight = 0; // momentaner wert
		DATA_TYPE delta_change = 0;   // zukünftige delta Veränderung

		Weight(DATA_TYPE w) {
			this->current_weight = w; 
		}

	};


	struct Bias {

		DATA_TYPE current_bias = 0; // momentaner wert
		DATA_TYPE delta_change = 0;   // zukünftige delta Veränderung

		Bias(DATA_TYPE b) {
			this->current_bias = b;
		}

	};

}