#pragma once


namespace AIModule {

	void evaluate_canvas(SimpleAI::AI_Instance&, std::vector<std::array<float, canvas_height>>&);

	void canvas_to_datapoint(std::vector<std::array<float, canvas_height>>&, SimpleAI::Data_Point&, int);

}