#include "nvjpeg2k.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "display_gst.h"
 
#include "glib.h"

//create display pipeline

//start:
//prepare batch 
//allocate output location
//decode batch 
//push image into pipeline 
//if final batch exit
//go to start
//exit:
//destroy display pipeline
//
#define ARR_COUNT(arr)(sizeof((arr))/sizeof((arr)[0])) 
#define RUN_COUNT 100 
#define TILE_COUNT 240 
#define FRAME_WIDTH 640
#define FRAME_HEIGHT 480 
#define FRAME_SIZE ((640*480 * 3))
#define FRAME_PITCH (640 * 3)
#define FULL_FRAME_SIZE (FRAME_SIZE * TILE_COUNT)
#define FULL_FRAME_PITCH (640 * 16 * 3)
#define FULL_FRAME_HEIGHT (480 * 15)
#define NUM_COMPONENTS 3

void decode_tiles(gchar * folder, unsigned char ** bitstream_buffer,guint32 count,nvjpeg2kHandle_t nvjpeg2k_handle,nvjpeg2kStream_t nvjpeg2k_stream,nvjpeg2kDecodeState_t decode_state,void * decode_output,CUstream cstream)  // host or pinned memory
{
		guint64 readtime = 0;
		size_t pitch = 0;
		nvjpeg2kImage_t output_image = {};
		for(int32_t i = 0 ; i < count; ++i)
		{
			gchar * path = g_strdup_printf("%s/tile_%d.j2k",folder,i+1) ;
			gsize length = 0;
			GError * error = 0;
			guint64 readstart = g_get_real_time();
			gboolean gotfile = g_file_get_contents(path,(gchar**)&bitstream_buffer[i],&length,&error);
			if(!gotfile || length == 0 || error != 0)
				continue;
			guint64 readend = g_get_real_time();
			readtime += readend - readstart;
			//g_print("readfile_time = %lu\n",readend - readstart);
			// content of bitstream buffer should not be overwritten until the decoding is complete
			nvjpeg2kStatus_t status = nvjpeg2kStreamParse(nvjpeg2k_handle, bitstream_buffer[i], length, 0, 0, nvjpeg2k_stream);
			//g_print("status %d/%d length %lu \n",status ,NVJPEG2K_STATUS_SUCCESS,length);
			// extract image info
			//			nvjpeg2kImageInfo_t image_info;
			//			nvjpeg2kStreamGetImageInfo(nvjpeg2k_stream, &image_info);

			// assuming the decoding of images with 8 bit precision, and 3 components

			//for (int c = 0; c < image_info.num_components; c++)
			//{
			//	nvjpeg2kStreamGetImageComponentInfo(nvjpeg2k_stream, &image_comp_info[c], c);
			//}
			//

			void * decode_output_p =  NULL;
			cudaError_t err = cudaMallocPitch(&decode_output_p,&pitch,FRAME_PITCH,FRAME_HEIGHT);
			//g_print("second pitch %lu err %u\n",pitch,err);

			size_t pitch_in_bytes[] = {FRAME_PITCH,320,320};
			output_image.pixel_data = (void**)&decode_output_p;
			output_image.pixel_type =  NVJPEG2K_UINT8;
			output_image.pitch_in_bytes = pitch_in_bytes;
			output_image.num_components = ARR_COUNT(pitch_in_bytes);
			nvjpeg2kDecodeParams_t decode_params = {};
			nvjpeg2kDecodeParamsCreate(&decode_params);
			nvjpeg2kStatus_t sformat = nvjpeg2kDecodeParamsSetOutputFormat(decode_params, NVJPEG2K_FORMAT_INTERLEAVED);
			//g_print("set output format %d\n",sformat);

			nvjpeg2kStatus_t setrgbo =  nvjpeg2kDecodeParamsSetRGBOutput(decode_params, 1);
			//g_print("set rgb output %d\n",setrgbo);

			status = nvjpeg2kDecodeImage(nvjpeg2k_handle, decode_state, nvjpeg2k_stream,decode_params, &output_image, cstream); 
			//g_print("decode status %d\n",status);

			//cudaDeviceSynchronize();

			int xOffset = (i % 16) * FRAME_PITCH;
			int yOffset = (i / 16) * FRAME_HEIGHT;

			err = cudaMemcpy2DAsync(decode_output + yOffset * FULL_FRAME_PITCH + xOffset, FULL_FRAME_PITCH ,decode_output_p,FRAME_PITCH,FRAME_PITCH,FRAME_HEIGHT,cudaMemcpyDeviceToDevice,cstream);
			cudaFree(decode_output_p);
		}
}

int main(int argc, char** argv )
{
	dg_initialize();

	char * folders[] = {
"./balmas/dumps/FrameNumber_4798_Sensor_2",
"./balmas/dumps/FrameNumber_4799_Sensor_2",
"./balmas/dumps/FrameNumber_4800_Sensor_2",
"./balmas/dumps/FrameNumber_4801_Sensor_2",
"./balmas/dumps/FrameNumber_4802_Sensor_2",
"./balmas/dumps/FrameNumber_4803_Sensor_2",
"./balmas/dumps/FrameNumber_4804_Sensor_2",
"./balmas/dumps/FrameNumber_4805_Sensor_2",
"./balmas/dumps/FrameNumber_4806_Sensor_2",
"./balmas/dumps/FrameNumber_4807_Sensor_2",
"./balmas/dumps/FrameNumber_4808_Sensor_2",
"./balmas/dumps/FrameNumber_4809_Sensor_2",
"./balmas/dumps/FrameNumber_4810_Sensor_2",
"./balmas/dumps/FrameNumber_4811_Sensor_2",
"./balmas/dumps/FrameNumber_4812_Sensor_2",
"./balmas/dumps/FrameNumber_4813_Sensor_2",
"./balmas/dumps/FrameNumber_4814_Sensor_2",
"./balmas/dumps/FrameNumber_4815_Sensor_2",
"./balmas/dumps/FrameNumber_4816_Sensor_2",
"./balmas/dumps/FrameNumber_4817_Sensor_2",
"./balmas/dumps/FrameNumber_4818_Sensor_2",
"./balmas/dumps/FrameNumber_4819_Sensor_2",
"./balmas/dumps/FrameNumber_4820_Sensor_2",
"./balmas/dumps/FrameNumber_4821_Sensor_2",
"./balmas/dumps/FrameNumber_4822_Sensor_2",
"./balmas/dumps/FrameNumber_4823_Sensor_2",
"./balmas/dumps/FrameNumber_4824_Sensor_2",
"./balmas/dumps/FrameNumber_4825_Sensor_2",
"./balmas/dumps/FrameNumber_4826_Sensor_2",
"./balmas/dumps/FrameNumber_4827_Sensor_2",
"./balmas/dumps/FrameNumber_4828_Sensor_2",
"./balmas/dumps/FrameNumber_4829_Sensor_2",
"./balmas/dumps/FrameNumber_4830_Sensor_2",
"./balmas/dumps/FrameNumber_4831_Sensor_2",
"./balmas/dumps/FrameNumber_4832_Sensor_2",
"./balmas/dumps/FrameNumber_4833_Sensor_2",
"./balmas/dumps/FrameNumber_4834_Sensor_2",
"./balmas/dumps/FrameNumber_4835_Sensor_2",
"./balmas/dumps/FrameNumber_4836_Sensor_2",
"./balmas/dumps/FrameNumber_4837_Sensor_2",
"./balmas/dumps/FrameNumber_4838_Sensor_2",
"./balmas/dumps/FrameNumber_4839_Sensor_2",
"./balmas/dumps/FrameNumber_4840_Sensor_2",
"./balmas/dumps/FrameNumber_4841_Sensor_2",
"./balmas/dumps/FrameNumber_4842_Sensor_2",
"./balmas/dumps/FrameNumber_4843_Sensor_2",
"./balmas/dumps/FrameNumber_4844_Sensor_2",
"./balmas/dumps/FrameNumber_4845_Sensor_2",
"./balmas/dumps/FrameNumber_4846_Sensor_2",
"./balmas/dumps/FrameNumber_4847_Sensor_2",
"./balmas/dumps/FrameNumber_4848_Sensor_2",
"./balmas/dumps/FrameNumber_4849_Sensor_2",
"./balmas/dumps/FrameNumber_4850_Sensor_2",
"./balmas/dumps/FrameNumber_4851_Sensor_2",
"./balmas/dumps/FrameNumber_4852_Sensor_2",
"./balmas/dumps/FrameNumber_4853_Sensor_2",
"./balmas/dumps/FrameNumber_4854_Sensor_2",
"./balmas/dumps/FrameNumber_4855_Sensor_2",
"./balmas/dumps/FrameNumber_4856_Sensor_2",
"./balmas/dumps/FrameNumber_4857_Sensor_2",
"./balmas/dumps/FrameNumber_4858_Sensor_2",
"./balmas/dumps/FrameNumber_4859_Sensor_2",
"./balmas/dumps/FrameNumber_4860_Sensor_2",
"./balmas/dumps/FrameNumber_4861_Sensor_2",
"./balmas/dumps/FrameNumber_4862_Sensor_2",
"./balmas/dumps/FrameNumber_4863_Sensor_2",
"./balmas/dumps/FrameNumber_4864_Sensor_2",
"./balmas/dumps/FrameNumber_4865_Sensor_2",
"./balmas/dumps/FrameNumber_4866_Sensor_2",
"./balmas/dumps/FrameNumber_4867_Sensor_2",
"./balmas/dumps/FrameNumber_4868_Sensor_2",
"./balmas/dumps/FrameNumber_4869_Sensor_2",
"./balmas/dumps/FrameNumber_4870_Sensor_2",
"./balmas/dumps/FrameNumber_4871_Sensor_2",
"./balmas/dumps/FrameNumber_4872_Sensor_2",
"./balmas/dumps/FrameNumber_4873_Sensor_2",
"./balmas/dumps/FrameNumber_4874_Sensor_2",
"./balmas/dumps/FrameNumber_4875_Sensor_2",
"./balmas/dumps/FrameNumber_4876_Sensor_2",
"./balmas/dumps/FrameNumber_4877_Sensor_2",
"./balmas/dumps/FrameNumber_4878_Sensor_2",
"./balmas/dumps/FrameNumber_4879_Sensor_2",
"./balmas/dumps/FrameNumber_4880_Sensor_2",
"./balmas/dumps/FrameNumber_4881_Sensor_2",
"./balmas/dumps/FrameNumber_4882_Sensor_2",
"./balmas/dumps/FrameNumber_4883_Sensor_2",
"./balmas/dumps/FrameNumber_4884_Sensor_2",
"./balmas/dumps/FrameNumber_4885_Sensor_2",
"./balmas/dumps/FrameNumber_4886_Sensor_2",
"./balmas/dumps/FrameNumber_4887_Sensor_2",
"./balmas/dumps/FrameNumber_4888_Sensor_2",
"./balmas/dumps/FrameNumber_4889_Sensor_2",
"./balmas/dumps/FrameNumber_4890_Sensor_2",
"./balmas/dumps/FrameNumber_4891_Sensor_2",
"./balmas/dumps/FrameNumber_4892_Sensor_2",
"./balmas/dumps/FrameNumber_4893_Sensor_2",
"./balmas/dumps/FrameNumber_4894_Sensor_2",
"./balmas/dumps/FrameNumber_4895_Sensor_2",
"./balmas/dumps/FrameNumber_4896_Sensor_2",
"./balmas/dumps/FrameNumber_4897_Sensor_2"};

    CUresult initState = cuInit (0);

	nvjpeg2kHandle_t nvjpeg2k_handle;
	nvjpeg2kStream_t nvjpeg2k_stream;
	nvjpeg2kDecodeState_t decode_state;

	nvjpeg2kCreateSimple(&nvjpeg2k_handle);
	nvjpeg2kDecodeStateCreate(nvjpeg2k_handle,&decode_state);
	nvjpeg2kStreamCreate(&nvjpeg2k_stream);

	size_t length;
	unsigned char *bitstream_buffer[TILE_COUNT];  // host or pinned memory
	guint64 starttime = g_get_real_time();
	cudaStream_t cstream = {};
	cudaStreamCreate(&cstream);
	// read the bitstream from and store it in bitstream_buffer;
	nvjpeg2kImageComponentInfo_t image_comp_info[NUM_COMPONENTS] = {{640,480},{320,240},{320,240}};

	nvjpeg2kImage_t output_image = {};
	unsigned char *decode_output = NULL;
	size_t pitch = 0;
	cudaError_t err = cudaMallocPitch((void**)&decode_output,&pitch,FULL_FRAME_PITCH,FULL_FRAME_HEIGHT);
	g_print("pitch full frame %lu err %u ptr %p\n",pitch,err,decode_output);
	//cudaMallocAsync((void**)&decode_output,  FULL_FRAME_SIZE ,cstream);
	guint64 readtime = 0;

	for(int32_t ii = 0 ; ii < ARR_COUNT(folders); ++ii)
	{
		gchar * folder = folders[ii]; 
		guint64 startframetime = g_get_real_time();
		for(int32_t i = 0 ; i < ARR_COUNT(bitstream_buffer) / 2  ; ++i)
		{
			gchar * path = g_strdup_printf("%s/tile_%d.j2k",folder,i+1) ;
			gsize length = 0;
			GError * error = 0;
			guint64 readstart = g_get_real_time();
			gboolean gotfile = g_file_get_contents(path,(gchar**)&bitstream_buffer[i],&length,&error);
			if(!gotfile || length == 0 || error != 0)
				continue;
			guint64 readend = g_get_real_time();
			readtime += readend - readstart;
			//g_print("readfile_time = %lu\n",readend - readstart);
			// content of bitstream buffer should not be overwritten until the decoding is complete
			nvjpeg2kStatus_t status = nvjpeg2kStreamParse(nvjpeg2k_handle, bitstream_buffer[i], length, 0, 0, nvjpeg2k_stream);
			//g_print("status %d/%d length %lu \n",status ,NVJPEG2K_STATUS_SUCCESS,length);
			// extract image info
//			nvjpeg2kImageInfo_t image_info;
//			nvjpeg2kStreamGetImageInfo(nvjpeg2k_stream, &image_info);

			// assuming the decoding of images with 8 bit precision, and 3 components

			//for (int c = 0; c < image_info.num_components; c++)
			//{
			//	nvjpeg2kStreamGetImageComponentInfo(nvjpeg2k_stream, &image_comp_info[c], c);
			//}
			//
			
			void * decode_output_p =  NULL;
			err = cudaMallocPitch(&decode_output_p,&pitch,FRAME_PITCH,FRAME_HEIGHT);
			//g_print("second pitch %lu err %u\n",pitch,err);

			size_t pitch_in_bytes[] = {FRAME_PITCH,320,320};
			output_image.pixel_data = (void**)&decode_output_p;
			output_image.pixel_type =  NVJPEG2K_UINT8;
			output_image.pitch_in_bytes = pitch_in_bytes;
			output_image.num_components = ARR_COUNT(pitch_in_bytes);
			nvjpeg2kDecodeParams_t decode_params = {};
			nvjpeg2kDecodeParamsCreate(&decode_params);
			nvjpeg2kStatus_t sformat = nvjpeg2kDecodeParamsSetOutputFormat(decode_params, NVJPEG2K_FORMAT_INTERLEAVED);
			//g_print("set output format %d\n",sformat);

			nvjpeg2kStatus_t setrgbo =  nvjpeg2kDecodeParamsSetRGBOutput(decode_params, 1);
			//g_print("set rgb output %d\n",setrgbo);

			status = nvjpeg2kDecodeImage(nvjpeg2k_handle, decode_state, nvjpeg2k_stream,decode_params, &output_image, cstream); 
			//g_print("decode status %d\n",status);

			//cudaDeviceSynchronize();

			int xOffset = (i % 16) * FRAME_PITCH;
            int yOffset = (i / 16) * FRAME_HEIGHT;

			cudaError_t err = cudaMemcpy2DAsync(decode_output + yOffset * FULL_FRAME_PITCH + xOffset, FULL_FRAME_PITCH ,decode_output_p,FRAME_PITCH,FRAME_PITCH,FRAME_HEIGHT,cudaMemcpyDeviceToDevice,cstream);
			cudaFree(decode_output_p);
		}

		cudaDeviceSynchronize();

		void * decode_output_copy;
		cudaMallocHost(&decode_output_copy,FULL_FRAME_SIZE);
		cudaMemcpyAsync(decode_output_copy,decode_output,FULL_FRAME_SIZE,cudaMemcpyDeviceToHost,cstream);
		dg_push_frame(decode_output_copy,FULL_FRAME_SIZE);
		cudaFree(decode_output_copy);
		g_print("time per frame %lu\n readtime %lu ", (g_get_real_time() - startframetime)/1000,readtime/1000);
		readtime = 0;
	}
	g_print("time %lu\n", (g_get_real_time() - starttime)/1000);
	//g_usleep(1000*1000*10);

}
