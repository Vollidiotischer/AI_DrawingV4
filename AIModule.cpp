#include <iostream>

#include <vector>
#include <array>

#include "SimpleAI/SimpleAI.h"

#include "Variables.h"


namespace AIModule {

	void canvas_to_datapoint(std::vector<std::array<float, canvas_height>>& canvas, SimpleAI::Data_Point& dp, int label) {

		if (canvas.size() * canvas[0].size() != dp.data.size()) {
			std::cout << "canvas size does not match Data_Point size" << std::endl;
			exit(1);
		}

		for (int i = 0; i < canvas.size(); i++) {
			for (int i2 = 0; i2 < canvas[i].size(); i2++) {
				dp.data[i * canvas.size() + i2] = canvas[i2][i];
			}
		}

		for (int i = 0; i < dp.result.size(); i++) {
			dp.result[i] = (i == label);
		}

	}



	void evaluate_canvas(SimpleAI::AI_Instance& ai, std::vector<std::array<float, canvas_height>>& canvas) {

		SimpleAI::Data_Point dp; 

		canvas_to_datapoint(canvas, dp, 0);

		std::array<float, SimpleAI::ai_layout[SimpleAI::num_layers - 1]> result; 

		ai.evaluate_input(dp.data, result);

		SimpleAI::println_array<float, SimpleAI::ai_layout[SimpleAI::num_layers - 1]>(result);

	}

}