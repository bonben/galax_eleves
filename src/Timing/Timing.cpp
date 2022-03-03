#include "Timing.hpp"

Timing::Timing() : duration_vec(10), current_average_FPS(0.0f)
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
	// We store a fixed number of timing samples before udpating the average FPS
	duration_vec.push_back(t_after - t_before);
	if(duration_vec.size() == duration_vec.capacity())
	{
		float duration_mean = 0.0;
		for(const auto& duration: duration_vec)
			duration_mean += duration.count();
		duration_mean /= duration_vec.size();
		current_average_FPS = 1.0f / duration_mean * 1000.0f;
		duration_vec.clear();
	}
}

float Timing
::get_current_average_FPS() const
{
    return current_average_FPS;
}

