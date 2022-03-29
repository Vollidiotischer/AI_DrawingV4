#pragma once

// have to be equal
constexpr int screen_width = 700; 
constexpr int screen_height = screen_width;

constexpr float pixel_size = 25; 

// have to be equal
constexpr int canvas_width = screen_width / pixel_size; 
constexpr int canvas_height = canvas_width;

static std::string path_name = "training_data.txt"; 