#include <iostream>

#include <vector>
#include <array>
#include <thread>

#include <SFML/Graphics.hpp>

#include "SimpleAI/SimpleAI.h"

#include "Variables.h"
#include "Events.h"
#include "Drawing.h"
#include "AIModule.h"

#include <Windows.h>

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


		void notify_taskbar() {

			wchar_t name[100] = { 0 };
			GetConsoleTitleW(name, 100);

			HWND yourHwnd;
			yourHwnd = FindWindowW(NULL, name);

			FLASHWINFO fi;
			fi.cbSize = sizeof(FLASHWINFO);
			fi.hwnd = yourHwnd;
			fi.dwFlags = FLASHW_TRAY;
			fi.uCount = 0;
			fi.dwTimeout = 0;
			FlashWindowEx(&fi);

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

	void start_training(SimpleAI::AI_Manager& manager, std::vector<SimpleAI::Data_Point>& data, int iterations) {

		//int iterations = 1; 
		int batch_size = 240; 
		int num_data_splits = data.size() / batch_size; 

		for (int i = 1; i <= iterations; i++) {

			// iterate over the whole data vector once
			for (int i2 = 0; i2 < num_data_splits; i2++) {
				std::vector<std::thread> _threads;

				for (int i3 = 0; i3 < manager.ai_list.size(); i3++) {
					_threads.push_back(std::thread(SimpleAI::AI_Instance::backprop, std::ref(manager.ai_list[i3]), std::ref(data), i2 * batch_size, ((i2 + 1) * batch_size)));
				}

				for (auto& t : _threads) {
					t.join();
				}

				for (int index = 0; index < manager.ai_list.size(); index++) {
					if (std::isnan(manager.ai_list[index].error) || std::isnan(-manager.ai_list[index].error)) {
						manager.ai_list[index] = SimpleAI::AI_Instance(SimpleAI::ai_learn_factor);
					}
				}

				std::cout << "Training: " << i2 + 1 << "/" << num_data_splits << " (" << i << "/" << iterations << ")" << std::endl;


			}

			/*
			// reshuffle 
			if (i % 2 == 0) {
				if (i == 0) {
					continue;
				}
				std::vector<std::thread> threads;

				for (int i = 0; i < manager.ai_list.size(); i++) {
					threads.push_back(std::thread(SimpleAI::AI_Instance::evaluate_input_list_mt, std::ref(manager.ai_list[i]), std::ref(data)));
				}

				for (auto& t : threads) {
					t.join();
				}

				manager.calculate_best_instance();
				manager.best_instance->print_error("\n");
				manager.reshuffel_instances();

				for (int index = 0; index < manager.ai_list.size(); index++) {
					if (std::isnan(manager.ai_list[index].error) || std::isnan(-manager.ai_list[index].error)) {
						manager.ai_list[index] = SimpleAI::AI_Instance(SimpleAI::ai_learn_factor);
					}
				}
			}
			*/
		}

	}

	void start_loop(SimpleAI::AI_Manager& manager, std::vector<SimpleAI::Data_Point>& data){


		while (true) {

			std::cout << "q = finish training\ns = calculate error\nt = train\nChoice: ";
			char c;
			std::cin >> c;

			if (c == 'q') {
				break;
			}
			

			if (c == 's') {
				std::cout << "Evaluating Instances..." << std::endl;

				std::vector<std::thread> threads;

				for (int i = 0; i < manager.ai_list.size(); i++) {
					threads.push_back(std::thread(SimpleAI::AI_Instance::evaluate_input_list, std::ref(manager.ai_list[i]), std::ref(data)));
				}

				for (auto& t : threads) {
					t.join();
				}

				manager.calculate_best_instance();

				manager.best_instance->print_error("\n");
			}
			if (c == 't') {
				std::cout << "Number of Iterations: ";
				std::cin >> c;

				MainLoop::start_training(manager, data, (int)c - (int)'0');

			}

			notify_taskbar();


		}


	}
}