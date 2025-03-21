#ifdef GALAX_MODEL_GPU

#ifndef MODEL_GPU_HPP_
#define MODEL_GPU_HPP_

#include "../Model.hpp"

#include <cuda_runtime.h>
#include "kernel.cuh"

class Model_GPU : public Model
{
private:

	std::vector<float4> positionsf4    ;
	std::vector<float4> velocitiesf4   ;
	std::vector<float4> accelerationsf4;

	float4* positionsGPU;
	float4* velocitiesGPU;
	float4* accelerationsGPU;
	
public:
	Model_GPU(const Initstate& initstate, Particles& particles);

	virtual ~Model_GPU();

	virtual void step();
};
#endif // MODEL_GPU_HPP_

#endif // GALAX_MODEL_GPU
