#ifndef TIMING_HPP_
#define TIMING_HPP_

#include <chrono>
#include <vector>
#include <iostream>
class Timing
{
private:
	std::chrono::time_point<std::chrono::high_resolution_clock> t_display;
	std::chrono::time_point<std::chrono::high_resolution_clock> t_before;
	std::chrono::time_point<std::chrono::high_resolution_clock> t_after;
	std::vector<std::chrono::duration<float, std::micro>> duration_vec;

public:
	Timing();

	void sample_before();
	void sample_after();
};


#endif // TIMING_HPP_
