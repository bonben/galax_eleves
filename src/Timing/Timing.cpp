#include "Timing.hpp"
#include <iostream>
#include <iomanip>

Timing::Timing() : duration_vec(10)
{
	t_display = std::chrono::high_resolution_clock::now();
};


void Timing
::sample_before()
{
	t_before = std::chrono::high_resolution_clock::now();

}

void Timing
::sample_after()
{
	t_after = std::chrono::high_resolution_clock::now();
	std::chrono::duration<float, std::micro> duration_us = t_after - t_before;
	duration_vec.push_back(duration_us);
	duration_vec.erase(duration_vec.begin());
	std::chrono::duration<float, std::milli> duration_display = t_after - t_display;
	if(duration_display.count() > 1000)
	{
		float duration_mean = 0.0;
		for (auto i = 0; i < duration_vec.size(); i++)
			duration_mean += duration_vec[i].count();
		duration_mean /= 10.0;
		auto fps = (float)1.0f / (duration_mean) * 1000000.0f;
		std::cout << "FPS: " << std::setw(3) << fps << "\r" << std::flush;
		t_display = t_after;
	}
}
