#include "Display_NO.hpp"
#include <csignal>
#include <cstdlib>

bool Display_NO::interrupt_received = false;

Display_NO
::Display_NO(Particles& particles)
: Display(particles)
{
#if !defined(_WIN64) && !defined(_WIN32)
	std::signal(SIGUSR1, Display_NO::signal_interrupt_handler);
	std::signal(SIGUSR2, Display_NO::signal_interrupt_handler);
#endif
	std::signal(SIGINT,  Display_NO::signal_interrupt_handler);
	std::signal(SIGTERM, Display_NO::signal_interrupt_handler);
}

Display_NO::~Display_NO()
{
}

void Display_NO
::update(bool& done)
{
    if (Display_NO::interrupt_received)
    {
        done = true;
    }
}

void Display_NO
::signal_interrupt_handler(int signal)
{
#if defined(_WIN64) || defined(_WIN32)
	if (signal == SIGINT || signal == SIGTERM)
#else
	if (signal == SIGUSR1 || signal == SIGUSR2 || signal == SIGTERM || signal == SIGINT)
#endif
	{
        Display_NO::interrupt_received = true;
    }
}
