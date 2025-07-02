
/*
* Copyright (c) 2018-2023 NVIDIA Corporation
*
* Permission is hereby granted, free of charge, to any person
* obtaining a copy of this software and associated documentation
* files (the "Software"), to deal in the Software without
* restriction, including without limitation the rights to use,
* copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the software, and to permit persons to whom the
* software is furnished to do so, subject to the following
* conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
* OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
* NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
* HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
* WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
* FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
* OTHER DEALINGS IN THE SOFTWARE.
*/

#include <fstream>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <cuda.h>
#include <sstream>
#include <iterator>
#include "NvOFCuda.h"

#ifdef linux 
#define DLL extern "C"
#else 
#ifdef __WINDOWS 
#define DLL extern "C" __declspec(dllexport)
#else 
#define DLL 
#endif
#endif

void l_swap(uint32_t & a , uint32_t &b)
{
    uint32_t t = a ;
    a = b;
    b = t;
}

struct nv_of_simple_context
{
    NvOFObj nvOpticalFlow;
    std::vector<NvOFBufferObj> inputBuffers;
    std::vector<NvOFBufferObj> outputBuffers;
    uint32_t first_buf_id;
    uint32_t second_buf_id;
    CUcontext cuContext;
    CUstream inputStream;
    CUstream outputStream;
} g_nv_of_context;


nv_of_simple_context NvOFSimpleInit(uint32_t width, uint32_t height, 
		NV_OF_BUFFER_FORMAT  ofBufFormatp,
		NV_OF_CUDA_BUFFER_TYPE inputBufferType,
		NV_OF_CUDA_BUFFER_TYPE  outputBufferType,
		NV_OF_PERF_LEVEL perfPreset,
		int gpuId,
		int gridSize
		)
{
        CUDA_DRVAPI_CALL(cuInit(0));
	nv_of_simple_context context = {};
        CUdevice cuDevice = 0;
        CUDA_DRVAPI_CALL(cuDeviceGet(&cuDevice, gpuId));
        char szDeviceName[80];
        CUDA_DRVAPI_CALL(cuDeviceGetName(szDeviceName, sizeof(szDeviceName), cuDevice));
        std::cout << "GPU in use: " << szDeviceName << std::endl;
        CUDA_DRVAPI_CALL(cuCtxCreate(&context.cuContext, 0, cuDevice));

	CUDA_DRVAPI_CALL(cuStreamCreate(&context.inputStream, CU_STREAM_DEFAULT));
	CUDA_DRVAPI_CALL(cuStreamCreate(&context.outputStream, CU_STREAM_DEFAULT));

	context.nvOpticalFlow = NvOFCuda::Create(context.cuContext, width, height,ofBufFormatp , inputBufferType, outputBufferType, NV_OF_MODE_OPTICALFLOW, perfPreset, context.inputStream, context.outputStream);

    uint32_t nScaleFactor = 1;
    uint32_t hwGridSize;
    if (!context.nvOpticalFlow->CheckGridSize(gridSize))
    {
        if (!context.nvOpticalFlow->GetNextMinGridSize(gridSize, hwGridSize))
        {
            throw std::runtime_error("Invalid parameter");
        }
        else
        {
            nScaleFactor = hwGridSize / gridSize;
        }
    }
    else
    {
        hwGridSize = gridSize;
    }
    //TODO(LEV) calc grid size optimal to hw
    //
    int hintGridSize = 256;
    bool bEnableRoi = false;
    bool enableExternalHints = false;
	context.nvOpticalFlow->Init(hwGridSize, hintGridSize, enableExternalHints, bEnableRoi);

	const uint32_t NUM_INPUT_BUFFERS = 2;
	const uint32_t NUM_OUTPUT_BUFFERS = NUM_INPUT_BUFFERS - 1;
	
    context.inputBuffers = context.nvOpticalFlow->CreateBuffers(NV_OF_BUFFER_USAGE_INPUT, NUM_INPUT_BUFFERS);
    context.outputBuffers = context.nvOpticalFlow->CreateBuffers(NV_OF_BUFFER_USAGE_OUTPUT, NUM_OUTPUT_BUFFERS);

	return context;
}

void NvOFSimpleExecute(nv_of_simple_context & nv_context, NvOFBufferObj * hintBuffer, double &executionTime)
{
        NvOFStopWatch nvStopWatch;
        CUDA_DRVAPI_CALL(cuStreamSynchronize(nv_context.inputStream));
        nvStopWatch.Start();
        //nv_context.nvOpticalFlow->Execute(nv_context.inputBuffers[nv_context.first_buf_id].get(), nv_context.inputBuffers[nv_context.second_buf_id].get(), nv_context.outputBuffers[0].get(), hintBuffer.get());
        nv_context.nvOpticalFlow->Execute(nv_context.inputBuffers[nv_context.first_buf_id].get(), nv_context.inputBuffers[nv_context.second_buf_id].get(), nv_context.outputBuffers[0].get(),NULL );
        CUDA_DRVAPI_CALL(cuStreamSynchronize(nv_context.outputStream));
        executionTime = nvStopWatch.Stop();
        l_swap(nv_context.first_buf_id,nv_context.second_buf_id);
}

void NvOFSimpleDeinit(nv_of_simple_context &context)
{
	CUDA_DRVAPI_CALL(cuStreamDestroy(context.outputStream));
	context.outputStream = nullptr;
	CUDA_DRVAPI_CALL(cuStreamDestroy(context.inputStream));
	context.inputStream = nullptr;
    CUDA_DRVAPI_CALL(cuCtxDestroy(context.cuContext));
}

DLL void * nv_opt_flow_get_context(uint32_t w, uint32_t h, NV_OF_BUFFER_FORMAT bf)
{


    /*
      buffer formats

    NV_OF_BUFFER_FORMAT_UNDEFINED,
    NV_OF_BUFFER_FORMAT_GRAYSCALE8,               < Input buffer format with 8 bit planar format 
    NV_OF_BUFFER_FORMAT_NV12,                      Input buffer format with 8 bit planar, UV interleaved 
    NV_OF_BUFFER_FORMAT_ABGR8,                    < Input buffer format with 8 bit packed A8B8G8R8 
    NV_OF_BUFFER_FORMAT_SHORT,                    < Output or hint buffer format for stereo disparity 
    NV_OF_BUFFER_FORMAT_SHORT2,                   < Output or hint buffer format for optical flow vector 
    NV_OF_BUFFER_FORMAT_UINT,                     < Legacy 32-bit Cost buffer format for optical flow vector / stereo disparity. 
                                                       This cost buffer format is not performance efficient and results in additional GPU usage.
                                                       Hence users are strongly recommended to use the 8-bit cost buffer format. 
                                                       Legacy 32-bit cost buffer format is also planned to be deprecated in future. 
    NV_OF_BUFFER_FORMAT_UINT8,                    < 8-bit Cost buffer format for optical flow vector / stereo disparity. 
    NV_OF_BUFFER_FORMAT_MAX
*/

    /* buffer types 
     
         NV_OF_CUDA_BUFFER_TYPE_CUARRAY 
         NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR;
    */
    /* perf levels
    
        NV_OF_PERF_LEVEL_SLOW  //the goal is to find the assignment that yields the maximum cost
        NV_OF_PERF_LEVEL_MEDIUM 
        NV_OF_PERF_LEVEL_FAST ;
    */
    g_nv_of_context = NvOFSimpleInit(w,h,bf,NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR ,NV_OF_CUDA_BUFFER_TYPE_CUDEVICEPTR, NV_OF_PERF_LEVEL_FAST,0,0 );
    return (void*) (&g_nv_of_context);
}

DLL void nv_opt_flow_get_flow_field(void * contextptr, uint8_t * &data, uint8_t * out_data)
{
	nv_of_simple_context * context = (nv_of_simple_context *) contextptr;
	context->inputBuffers[context->first_buf_id]->UploadData(data,NULL,NULL);
	double executionTime = 0;
	NvOFSimpleExecute(*context,NULL,executionTime);
	context->outputBuffers[0]->DownloadData(out_data);
}

