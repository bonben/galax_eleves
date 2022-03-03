#ifndef TIMING_HPP_
#define TIMING_HPP_

#include <chrono>
#include <vector>
#include <iostream>
class Timing
{
private:
	std::vector<std::chrono::duration<float, std::milli>> duration_vec;
        float current_average_FPS;

	std::chrono::time_point<std::chrono::high_resolution_clock> t_display;
	std::chrono::time_point<std::chrono::high_resolution_clock> t_before;
	std::chrono::time_point<std::chrono::high_resolution_clock> t_after;

public:
	Timing();

	void sample_before();
	void sample_after();
        float get_current_average_FPS() const;
};


#endif // TIMING_HPP_
