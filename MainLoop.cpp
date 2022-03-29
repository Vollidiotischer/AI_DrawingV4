#include <iostream>

#include <vector>
#include <array>

#include <SFML/Graphics.hpp>

#include "SimpleAI/SimpleAI.h"

#include "Variables.h"
#include "Events.h"
#include "Drawing.h"
#include "AIModule.h"


namespace MainLoop {

	namespace {
		// TEMP FUN
		void data_to_canvas(std::vector<std::array<float, canvas_height>>& canvas, std::vector<SimpleAI::Data_Point>& data) {

			int index = 59999; 

			SimpleAI::println_array<float, SimpleAI::ai_layout[SimpleAI::num_layers - 1]>(data[index].result); 

			for (int i = 0; i < canvas.size(); i++) {
				for (int i2 = 0; i2 < canvas[i].size(); i2++) {
					canvas[i2][i] = data[index].data[i * canvas_width + i2];
				}
			}

		}
	}

	void start_canvas(SimpleAI::AI_Instance& ai, std::vector<SimpleAI::Data_Point>& data) {

		sf::RenderWindow rw(sf::VideoMode(screen_width, screen_height), "Canvas", sf::Style::Close | sf::Style::Titlebar);
		rw.setKeyRepeatEnabled(false); 

		std::vector<std::array<float, canvas_height>> canvas(canvas_width);

		//data_to_canvas(canvas, data); 

		while (rw.isOpen()) {
			
			Events::event_handler(rw, canvas, data); 

			Drawing::draw_screen(rw, canvas); 

			AIModule::evaluate_canvas(ai, canvas); 

		}

	}

	void start_training(SimpleAI::AI_Manager& manager, std::vector<SimpleAI::Data_Point>& data) {

		int iterations = 100; 

		for (int i = 0; i < iterations; i++) {

			manager.train_all_instances(data, 600); 

			std::cout << "Training: " << i << "/" << iterations << std::endl;

			manager.best_instance->print_error("\n"); 
		}

		manager.evaluate_instances(data);

		manager.calculate_best_instance();

		manager.best_instance->print_error(" --- ");
		manager.reshuffel_instances();
		manager.best_instance->print_error("\n");

	}

}