#include<assert.h>
#include<stdint.h>

#include "nvjpeg2k.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "glib.h"

#define ARR_COUNT(arr)(sizeof((arr))/sizeof((arr)[0])) 

struct nu_j2k_dec_context
{
	nvjpeg2kHandle_t nvjpeg2k_handle;
	nvjpeg2kStream_t nvjpeg2k_stream;
	nvjpeg2kDecodeState_t decode_state;
	nvjpeg2kDecodeParams_t decode_params;

	size_t pitch_in_bytes[3]; 
	uint32_t width; 
	uint32_t height; 
	CUdevice device;
	CUcontext context;
	CUstream cstream;

};

enum nu_j2k_supported_format
{
	J2K_I420_PLANAR,
	J2K_INVALID
};

extern "C" nu_j2k_dec_context nu_j2k_init(uint32_t tile_width, uint32_t tile_height,nu_j2k_supported_format in_format,uint32_t deviceId)
{
	assert(J2K_I420_PLANAR == in_format);
	nu_j2k_dec_context context;
    CUresult initState = cuInit (deviceId);
	cuDeviceGet(&context.device,0);
    cuCtxCreate(&context.context,0,context.device);

	nvjpeg2kCreateSimple(&context.nvjpeg2k_handle);
	nvjpeg2kDecodeStateCreate(context.nvjpeg2k_handle,&context.decode_state);
	nvjpeg2kStreamCreate(&context.nvjpeg2k_stream);
	context.pitch_in_bytes[0] = tile_width * 3; 
	context.pitch_in_bytes[1] = tile_width /2; 
	context.pitch_in_bytes[2] = tile_width /2; 
	nvjpeg2kDecodeParamsCreate(&context.decode_params);

	nvjpeg2kStatus_t sformat =  nvjpeg2kDecodeParamsSetOutputFormat(context.decode_params, NVJPEG2K_FORMAT_INTERLEAVED);
	nvjpeg2kStatus_t setrgbo =  nvjpeg2kDecodeParamsSetRGBOutput(context.decode_params, 1);
	context.height = tile_height;
	context.width  = tile_width;

	return context;
}

struct nu_j2k_decoded_tile
{
	void * data;
	uint64_t size;
};

extern "C" nu_j2k_decoded_tile nu_j2k_decode_tile(nu_j2k_dec_context context, const uint8_t * tileptr, uint64_t size)
{
	nu_j2k_decoded_tile rawtile {};
	guint64 readtime = 0;
	size_t pitch = 0;
	nvjpeg2kImage_t output_image = {};
	gsize length = 0;
	// content of bitstream buffer should not be overwritten until the decoding is complete
	nvjpeg2kStatus_t status = nvjpeg2kStreamParse(context.nvjpeg2k_handle, tileptr, size , 0, 0, context.nvjpeg2k_stream);
	g_print("status %d/%d length %lu \n",status ,NVJPEG2K_STATUS_SUCCESS,length);
	// extract image info
	//	nvjpeg2kImageInfo_t image_info;
	//	nvjpeg2kStreamGetImageInfo(nvjpeg2k_stream, &image_info);

	// assuming the decoding of images with 8 bit precision, and 3 components

	//for (int c = 0; c < image_info.num_components; c++)
	//{
	//	nvjpeg2kStreamGetImageComponentInfo(nvjpeg2k_stream, &image_comp_info[c], c);
	//}
	//

	void * decode_output_p =  NULL;
	cudaError_t err = cudaMallocPitch(&decode_output_p,&pitch,context.pitch_in_bytes[0],context.height);
	//g_print("second pitch %lu err %u\n",pitch,err);

	output_image.pixel_data = (void**)&decode_output_p;
	output_image.pixel_type =  NVJPEG2K_UINT8;
	output_image.pitch_in_bytes = context.pitch_in_bytes;
	output_image.num_components = ARR_COUNT(context.pitch_in_bytes);

	status = nvjpeg2kDecodeImage(context.nvjpeg2k_handle, context.decode_state, context.nvjpeg2k_stream,context.decode_params, &output_image, context.cstream); 
	if(status ==  NVJPEG2K_STATUS_SUCCESS)
	{
		cudaMallocHost(&rawtile.data, context.pitch_in_bytes[0] * context.height);
		err = cudaMemcpy2DAsync(rawtile.data, context.pitch_in_bytes[0], decode_output_p, context.pitch_in_bytes[0],context.pitch_in_bytes[0],context.height,cudaMemcpyDeviceToHost,context.cstream);
		cudaFreeAsync(decode_output_p,context.cstream);
		cudaDeviceSynchronize();
	}

	return rawtile;
}

extern "C" void nu_j2k_deinit(nu_j2k_dec_context context)
{
	assert(0);
}
