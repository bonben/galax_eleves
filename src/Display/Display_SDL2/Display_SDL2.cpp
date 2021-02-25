#ifdef GALAX_DISPLAY_SDL2

#include "Display_SDL2.hpp"

Display_SDL2
::Display_SDL2(Particles& particles)
: Display(particles)
{
	SDL_DisplayMode current;

	if (SDL_Init (SDL_INIT_EVERYTHING) < 0)
	{
		printf("error: unable to init sdl\n");
        // TODO throw exception
        exit(EXIT_FAILURE);
	}

	if (SDL_GetDesktopDisplayMode(0, &current))
	{
		printf("error: unable to get current display mode\n");
        // TODO throw exception
        exit(EXIT_FAILURE);
	}

	window = SDL_CreateWindow("SDL", 	SDL_WINDOWPOS_CENTERED,
										SDL_WINDOWPOS_CENTERED,
										width, height,
										SDL_WINDOW_OPENGL);

	glWindow = SDL_GL_CreateContext(window);

	GLenum status = glewInit();

	if (status != GLEW_OK)
	{
		printf("error: unable to init glew\n");
		// TODO throw exception
		exit(EXIT_FAILURE);
	}

	SDL_GL_SetSwapInterval(1);
}

Display_SDL2::~Display_SDL2()
{
	SDL_GL_DeleteContext(glWindow);
	SDL_DestroyWindow(window);
	SDL_Quit();
}

void Display_SDL2
::update(bool& done)
{
	int i;

	while (SDL_PollEvent(&event))
	{

		unsigned int e = event.type;

		if (e == SDL_MOUSEMOTION)
		{
			mouseMoveX = event.motion.x;
			mouseMoveY = height - event.motion.y - 1;
		}
		else if (e == SDL_KEYDOWN)
		{
			if (event.key.keysym.sym == SDLK_F1)
				g_showGrid = !g_showGrid;
			else if (event.key.keysym.sym == SDLK_F2)
				g_showAxes = !g_showAxes;
			else if (event.key.keysym.sym == SDLK_ESCAPE)
				done = true;
		}

		if (e == SDL_QUIT)
		{
			printf("quit\n");
			done = true;
		}
	}

	mouseDeltaX = mouseMoveX - mouseOriginX;
	mouseDeltaY = mouseMoveY - mouseOriginY;

	if (SDL_GetMouseState(0, 0) & SDL_BUTTON_LMASK)
	{
		oldCamRot[ 0 ] += -mouseDeltaY / 5.0f;
		oldCamRot[ 1 ] += mouseDeltaX / 5.0f;
	}
	else if (SDL_GetMouseState(0, 0) & SDL_BUTTON_RMASK)
	{
		oldCamPos[ 2 ] += (mouseDeltaY / 100.0f) * 0.5 * fabs(oldCamPos[ 2 ]);
		oldCamPos[ 2 ]  = oldCamPos[ 2 ] > -5.0f ? -5.0f : oldCamPos[ 2 ];
	}

	mouseOriginX = mouseMoveX;
	mouseOriginY = mouseMoveY;

	glViewport     (0, 0, width, height);
	glClearColor   (0.2f, 0.2f, 0.2f, 1.0f);
	glClear        (GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable       (GL_BLEND);
	glBlendEquation(GL_FUNC_ADD);
	glBlendFunc    (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glDisable      (GL_TEXTURE_2D);
	glEnable       (GL_DEPTH_TEST);
	glMatrixMode   (GL_PROJECTION);
	glLoadIdentity ();
	gluPerspective (50.0f, (float)width / (float)height, 0.1f, 100000.0f);
	glMatrixMode   (GL_MODELVIEW);
	glLoadIdentity ();

	for (i = 0; i < 3; ++i)
	{
		newCamPos[i] += (oldCamPos[i] - newCamPos[i]) * g_inertia;
		newCamRot[i] += (oldCamRot[i] - newCamRot[i]) * g_inertia;
	}

	glTranslatef(newCamPos[0], newCamPos[1], newCamPos[2]);
	glRotatef   (newCamRot[0], 1.0f, 0.0f, 0.0f);
	glRotatef   (newCamRot[1], 0.0f, 1.0f, 0.0f);

	if (g_showGrid)
		DrawGridXZ(-100.0f, 0.0f, -100.0f, 20, 20, 10.0);

	if (g_showAxes)
		ShowAxes();

	for (int i = 0; i < particles.x.size(); i++)
	{
		glBegin   (GL_POINTS);
		glColor3f (1.0f, 1.0f, 1.0f);
		glVertex3f(particles.x[i], particles.y[i], particles.z[i]);
		glEnd();
	}

	glMatrixMode  (GL_PROJECTION);
	glLoadIdentity();
	gluOrtho2D    (0, width, 0, height);
	glMatrixMode  (GL_MODELVIEW);
	glLoadIdentity();

	SDL_GL_SwapWindow(window);
	SDL_UpdateWindowSurface(window);
}


void Display_SDL2
::DrawPoint(float x, float y, float z) const
{
		glBegin(GL_POINTS);
		glColor3f(1.0f, 1.0f, 1.0f);
		glVertex3f(x, y, z);
		glEnd();
}

void Display_SDL2
::DrawGridXZ(float ox, float oy, float oz, int w, int h, float sz) const
{

	glLineWidth(1.0f);
	glBegin    (GL_LINES);
	glColor3f  (0.48f, 0.48f, 0.48f);

	for (auto i = 0; i <= h; ++i)
	{
		glVertex3f(ox, oy, oz + i * sz);
		glVertex3f(ox + w * sz, oy, oz + i * sz);
	}

	for (auto i = 0; i <= h; ++i)
	{
		glVertex3f(ox + i * sz, oy, oz);
		glVertex3f(ox + i * sz, oy, oz + h * sz);
	}

	glEnd();
}

void Display_SDL2
::ShowAxes() const
{
	glLineWidth(2.0f);
	glBegin(GL_LINES);

	glColor3f (1.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(2.0f, 0.0f, 0.0f);

	glColor3f (0.0f, 1.0f, 0.0f);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 2.0f, 0.0f);

	glColor3f (0.0f, 0.0f, 1.0f);
	glVertex3f(0.0f, 0.0f, 0.0f);
	glVertex3f(0.0f, 0.0f, 2.0f);

	glEnd();
}

#endif
