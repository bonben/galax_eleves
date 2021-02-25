#ifndef DISPLAY_NO_HPP_
#define DISPLAY_NO_HPP_

#include <vector>

#include "../Display.hpp"

class Display_NO : public Display
{
private:
    static bool interrupt_received;
public:
	Display_NO(Particles& particles);
	~Display_NO();

	virtual void update(bool& done);

private:
	static void signal_interrupt_handler(int signal);

};

#endif // DISPLAY_NO_HPP_