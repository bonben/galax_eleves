#ifdef GALAX_DISPLAY_SDL2

#ifndef DISPLAY_SDL2_HPP_
#define DISPLAY_SDL2_HPP_

#include "GL/glew.h"
#include <SDL2/SDL.h>
#include <SDL2/SDL_opengl.h>

#include "../Display.hpp"

class Display_SDL2 : public Display
{
private:

	const int width  = 640;
	const int height = 480;
	const float g_inertia = 0.5f;

	float oldCamPos[3] = {0.0f, 0.0f, -45.0f};
	float oldCamRot[3] = {0.0f, 0.0f,   0.0f};
	float newCamPos[3] = {0.0f, 0.0f, -45.0f};
	float newCamRot[3] = {0.0f, 0.0f,   0.0f};
	float mouseOriginX = 0.0f;
	float mouseOriginY = 0.0f;
	float mouseMoveX   = 0.0f;
	float mouseMoveY   = 0.0f;
	float mouseDeltaX  = 0.0f;
	float mouseDeltaY  = 0.0f;
	bool  g_showGrid   = true;
	bool  g_showAxes   = true;

	SDL_Event     event;
	SDL_GLContext glWindow;
	SDL_Window*   window;

public:
	Display_SDL2(Particles& particles);
	~Display_SDL2();

	virtual void update(bool& done);

private:
	void DrawPoint (float x,  float y,  float z                          ) const;
	void DrawGridXZ(float ox, float oy, float oz, int w, int h, float sz ) const;
	void ShowAxes  (                                                     ) const;
};

#endif // DISPLAY_SDL2_HPP_

#endif // GALAX_DISPLAY_SDL2