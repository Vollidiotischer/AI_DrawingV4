#include <iostream>

#include <vector>
#include <array>

#include <SFML/Graphics.hpp>

#include "Variables.h"


namespace Drawing {

	void draw_screen(sf::RenderWindow& rw, std::vector<std::array<float, canvas_height>>& canvas) { 

		sf::RectangleShape rect({ pixel_size, pixel_size }); 
		rect.setOutlineThickness(0); 


		rw.clear(sf::Color::White); 

		for (int i = 0; i < canvas.size(); i++) {
			for (int i2 = 0; i2 < canvas[i].size(); i2++) {
				float color = (1 - canvas[i][i2]) * 255; 
				rect.setFillColor(sf::Color(color, color, color));
				rect.setPosition({(float)i * pixel_size, (float)i2 * pixel_size}); 
				rw.draw(rect); 
			}
		}

		rw.display(); 

	}

}