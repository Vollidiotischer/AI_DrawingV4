#pragma once


namespace MainLoop {

	void start_canvas(SimpleAI::AI_Instance&, std::vector<SimpleAI::Data_Point>&); 

	void start_training(SimpleAI::AI_Manager&, std::vector<SimpleAI::Data_Point>&, int);

	void start_loop(SimpleAI::AI_Manager&, std::vector<SimpleAI::Data_Point>&);

}