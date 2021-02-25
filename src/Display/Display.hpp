#ifndef DISPLAY_HPP_
#define DISPLAY_HPP_

#include <vector>
#include "../Particles.hpp"

class Display
{
protected:
	Particles& particles;

public:
	Display(Particles& particles);
	~Display();

	virtual void update(bool& done) = 0;
};

#endif // DISPLAY_HPP_