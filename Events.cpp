#include <iostream>

#include <vector>

#include "SFML/Graphics.hpp"

#include "SimpleAI/SimpleAI.h"

#include "Variables.h"
#include "AIModule.h"

namespace Events {

	namespace {

		void evaluate_1_to_9_keys(sf::Event& events, int& label) {

			if (events.key.code >= sf::Keyboard::Num0 && events.key.code <= sf::Keyboard::Num9) {
				label = events.key.code - sf::Keyboard::Num0;
				std::cout << "Switched label to " << label << std::endl; 
			}

		}

		void clear_canvas(std::vector<std::array<float, canvas_height>>& canvas) {
			for (auto& arr : canvas) {
				for (auto& i : arr) {
					i = 0; 
				}
			}
		}
	}

	void event_handler(sf::RenderWindow& rw, std::vector<std::array<float, canvas_height>>& canvas, std::vector<SimpleAI::Data_Point>& data) {

		static int label = 0; 

		sf::Event events; 

		while (rw.pollEvent(events)) {

			if (events.type == sf::Event::Closed) {
				rw.close(); 
			}

			if (events.type == sf::Event::KeyPressed) {



				if (events.key.code == sf::Keyboard::S) {

					SimpleAI::Resource_Manager::save_data(path_name, data, std::ostream::trunc);

				}

				if (events.key.code == sf::Keyboard::C) {

					clear_canvas(canvas); 

				}

				if (events.key.code == sf::Keyboard::Space) {

					SimpleAI::Data_Point dp; 
					AIModule::canvas_to_datapoint(canvas, dp, label); 
					data.push_back(dp);

					// with shift -> overwrite saved data
					if (sf::Keyboard::isKeyPressed(sf::Keyboard::LShift)) {
						SimpleAI::Resource_Manager::save_data(path_name, data, std::ostream::trunc); 
					}
					else { // without shift -> append to current data
						std::vector<SimpleAI::Data_Point> temp_data = { dp };
						SimpleAI::Resource_Manager::save_data(path_name, temp_data, std::ostream::app);

					}

					clear_canvas(canvas); 

				}
				else {
					evaluate_1_to_9_keys(events, label); 
				}
			}

		}

		if (sf::Mouse::isButtonPressed(sf::Mouse::Left) || sf::Mouse::isButtonPressed(sf::Mouse::Right)) {
			int mox = sf::Mouse::getPosition(rw).x / pixel_size;
			int moy = sf::Mouse::getPosition(rw).y / pixel_size;

			if (mox >= 0 && mox < canvas_width && moy >= 0 && moy < canvas_height) {
				canvas[mox][moy] = sf::Mouse::isButtonPressed(sf::Mouse::Left);
			}
		}

	}

}