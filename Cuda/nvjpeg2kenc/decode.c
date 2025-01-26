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

#define WIDTH_TILES 16
#define HEIGHT_TILES 15 

#define FULL_FRAME_SIZE (FRAME_SIZE * TILE_COUNT)
#define FULL_FRAME_PITCH (640 * WIDTH_TILES * 3)
#define FULL_FRAME_HEIGHT (480 * HEIGHT_TILES)
#define NUM_COMPONENTS 3

struct decode_tile_params
{
	gchar * folder;
	unsigned char ** bitstream_buffer;
	guint64 * bitstream_length;
	guint32 bitstream_buffer_count;
	guint32 startindex;
	guint32 endindex;
	guint32 startx;
	guint32 endx;
	guint32 starty;
	guint32 endy;
	guint32 count;
	nvjpeg2kHandle_t nvjpeg2k_handle;
	nvjpeg2kStream_t nvjpeg2k_stream;
	nvjpeg2kDecodeState_t decode_state;
	void * decode_output;
	void * decode_output_p;
	CUstream cstream;
};

void fill_tile_bitstream_buffer_index_based (struct decode_tile_params *prms)
{
	guint64 readtime = 0;
	prms->bitstream_buffer_count = 0;
	for(int32_t i = prms->startindex; i < prms->endindex; ++i )
	{
			gchar * path = g_strdup_printf("%s/tile_%d.j2k",prms->folder,i+1) ;
			//g_print("path %s x : %d y : %d\n",path,x,y);
			gsize length = 0;
			GError * error = 0;
			guint64 readstart = g_get_real_time();
			gboolean gotfile = g_file_get_contents(path,(gchar**)&(prms->bitstream_buffer[prms->bitstream_buffer_count]),&length,&error);
			prms->bitstream_length[prms->bitstream_buffer_count] = length;
			if(!gotfile || length == 0 || error != 0)
			{
				g_print("ERROR : %s \n",error->message);
				continue;
			}
			guint64 readend = g_get_real_time();
			readtime += readend - readstart;
			prms->bitstream_buffer_count++;
	}
	//g_print("filled tiles %d\n",prms->bitstream_buffer_count);
}
void fill_tile_bitstream_buffer(struct decode_tile_params *prms)
{
	guint64 readtime = 0;
	prms->bitstream_buffer_count = 0;
	for(int32_t x = prms->startx ; x < prms->endx; ++x)
	{
		for(int32_t y = prms->starty ; y < prms->endy ; ++y)
		{
			cudaDeviceSynchronize();
			int32_t i = y * (WIDTH_TILES) + x;  
			gchar * path = g_strdup_printf("%s/tile_%d.j2k",prms->folder,i+1) ;
			//g_print("path %s x : %d y : %d\n",path,x,y);
			gsize length = 0;
			GError * error = 0;
			guint64 readstart = g_get_real_time();
			gboolean gotfile = g_file_get_contents(path,(gchar**)&(prms->bitstream_buffer[i]),&length,&error);
			prms->bitstream_length[i] = length;
			if(!gotfile || length == 0 || error != 0)
			{
				g_print("ERROR : %s \n",error->message);
				continue;
			}
			guint64 readend = g_get_real_time();
			readtime += readend - readstart;
			prms->bitstream_buffer_count++;
		}
	}
	g_print("filled tiles %d\n",prms->bitstream_buffer_count);
}
void nu_j2k_decode_tiles_index_based(struct decode_tile_params prms)  // host or pinned memory
{
	guint64 readtime = 0;
	size_t pitch = 0;
	nvjpeg2kImage_t output_image = {};
	uint32_t count = 0;
	for(int32_t i = prms.startindex; i < prms.endindex; ++i)
	{
		gsize length = 0;
		//uint64_t t = g_get_real_time();
		nvjpeg2kStatus_t status = nvjpeg2kStreamParse(prms.nvjpeg2k_handle, prms.bitstream_buffer[count], prms.bitstream_length[count] , 0, 0, prms.nvjpeg2k_stream);
		//g_print("status %d/%d length %lu time = %lu\n",status ,NVJPEG2K_STATUS_SUCCESS,length,g_get_real_time()- t);
		// extract image info
		//			nvjpeg2kImageInfo_t image_info;
		//			nvjpeg2kStreamGetImageInfo(nvjpeg2k_stream, &image_info);

		// assuming the decoding of images with 8 bit precision, and 3 components

		//for (int c = 0; c < image_info.num_components; c++)
		//{
		//	nvjpeg2kStreamGetImageComponentInfo(nvjpeg2k_stream, &image_comp_info[c], c);
		//}
		//

		void * decode_output_p =  prms.decode_output_p;
		cudaError_t err;
		//cudaError_t err = cudaMallocPitch(&decode_output_p,&pitch,FRAME_PITCH,FRAME_HEIGHT);
		//g_usleep(1000*1000);
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

		status = nvjpeg2kDecodeImage(prms.nvjpeg2k_handle, prms.decode_state, prms.nvjpeg2k_stream,decode_params, &output_image, prms.cstream); 
		//g_print("decode status %d\n",status);

		int xOffset = (i % 16) * FRAME_PITCH;
		int yOffset = (i / 16) * FRAME_HEIGHT;

		err = cudaMemcpy2DAsync(prms.decode_output + yOffset * FULL_FRAME_PITCH + xOffset, FULL_FRAME_PITCH ,decode_output_p,FRAME_PITCH,FRAME_PITCH,FRAME_HEIGHT,cudaMemcpyDeviceToDevice,prms.cstream);

		//cudaFree(decode_output_p);
		//cudaFreeAsync(decode_output_p,prms.cstream);
		//cudaDeviceSynchronize();
		//g_free(&prms.bitstream_buffer[count]);
		//nvjpeg2kDecodeParamsDestroy(decode_params);
		count++;
	}
	//nvjpeg2kDecodeStateDestroy(prms.decode_state);
	//nvjpeg2kDestroy(prms.nvjpeg2k_handle);
	//cudaStreamSynchronize(prms.cstream);
	//cudaDeviceSynchronize();
}


void nu_j2k_decode_tiles(struct decode_tile_params prms)  // host or pinned memory
{
	guint64 readtime = 0;
	size_t pitch = 0;
	nvjpeg2kImage_t output_image = {};
	for(int32_t x = prms.startx ; x < prms.endx; ++x)
	{
		for(int32_t y = prms.starty ; y < prms.endy ; ++y)
		{
			int32_t i = y * (WIDTH_TILES) + x;  
			cudaDeviceSynchronize();
			gsize length = 0;
			// content of bitstream buffer should not be overwritten until the decoding is complete
			nvjpeg2kStatus_t status = nvjpeg2kStreamParse(prms.nvjpeg2k_handle, prms.bitstream_buffer[i], prms.bitstream_length[i] , 0, 0, prms.nvjpeg2k_stream);
			g_print("status %d/%d length %lu \n",status ,NVJPEG2K_STATUS_SUCCESS,length);
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

			status = nvjpeg2kDecodeImage(prms.nvjpeg2k_handle, prms.decode_state, prms.nvjpeg2k_stream,decode_params, &output_image, prms.cstream); 
			g_print("decode status %d\n",status);

			int xOffset = (i % 16) * FRAME_PITCH;
			int yOffset = (i / 16) * FRAME_HEIGHT;

			err = cudaMemcpy2DAsync(prms.decode_output + yOffset * FULL_FRAME_PITCH + xOffset, FULL_FRAME_PITCH ,decode_output_p,FRAME_PITCH,FRAME_PITCH,FRAME_HEIGHT,cudaMemcpyDeviceToDevice,prms.cstream);

			//cudaDeviceSynchronize();
			//cudaFree(decode_output_p,prms.cstream);
			cudaFree(decode_output_p);
		}
	}
}

int main(int argc, char** argv )
{

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
	CUdevice device = 0;
	CUcontext context = 0;
	cuDeviceGet(&device,0);
    cuCtxCreate(&context,0,device);

	dg_initialize(context,device);

	nvjpeg2kHandle_t nvjpeg2k_handle;
	nvjpeg2kStream_t nvjpeg2k_stream;
	nvjpeg2kDecodeState_t decode_state;

	nvjpeg2kCreateSimple(&nvjpeg2k_handle);
	nvjpeg2kDecodeStateCreate(nvjpeg2k_handle,&decode_state);
	nvjpeg2kStreamCreate(&nvjpeg2k_stream);

	size_t length;
	unsigned char *bitstream_buffer[TILE_COUNT];  // host or pinned memory
	guint64 bitstream_length[TILE_COUNT];  // host or pinned memory
	guint64 starttime = g_get_real_time();
	cudaStream_t cstream;
	cudaStreamCreate(&cstream);

	nvjpeg2kImageComponentInfo_t image_comp_info[NUM_COMPONENTS] = {{640,480},{320,240},{320,240}};

	nvjpeg2kImage_t output_image = {};
	unsigned char *decode_output = NULL;
	size_t pitch = 0;
	cudaError_t err = cudaMallocPitch((void**)&decode_output, &pitch, FULL_FRAME_PITCH, FULL_FRAME_HEIGHT);
	g_print("pitch full frame %lu err %u ptr %p\n",pitch,err,decode_output);
	//cudaMallocAsync((void**)
	//decode_output,  FULL_FRAME_SIZE ,cstream);
	guint64 readtime = 0;
	const uint32_t BATCH_SIZE = 240;
	void ** decode_output_ps = (void**)malloc(BATCH_SIZE* sizeof(void **));  
	nvjpeg2kHandle_t *nvjpeg2k_handles = (nvjpeg2kHandle_t *) malloc(BATCH_SIZE* sizeof(nvjpeg2kHandle_t));
	nvjpeg2kStream_t * nvjpeg2k_streams = (nvjpeg2kStream_t*) malloc(BATCH_SIZE* sizeof(nvjpeg2kStream_t));
	nvjpeg2kDecodeState_t * nvjpeg2k_decode_states = (nvjpeg2kDecodeState_t*) malloc(BATCH_SIZE* sizeof(nvjpeg2kDecodeState_t));
	cudaStream_t cstreams[BATCH_SIZE];

	for(int32_t kk = 0 ; kk < BATCH_SIZE ; kk++)
	{
		size_t pitch = 0;
		cudaError_t err = cudaMallocPitch(&decode_output_ps[kk],&pitch,FRAME_PITCH,FRAME_HEIGHT);

		nvjpeg2kCreateSimple(&nvjpeg2k_handles[kk]);
		nvjpeg2kStreamCreate(&nvjpeg2k_streams[kk]);
		nvjpeg2kDecodeStateCreate(nvjpeg2k_handles[kk],&nvjpeg2k_decode_states[kk]);
		cudaStreamCreate(&cstreams[kk]);
	}

	
	for(int32_t ii = 0 ; ii < ARR_COUNT(folders); ++ii)
	//for(int32_t ii = 0 ; ii < 10; ++ii)
	{

		const uint32_t TILES_PER_BATCH = TILE_COUNT/BATCH_SIZE;
		guint64 startframetime = g_get_real_time(); 
		uint8_t ** bitstream_buffer = (uint8_t **)malloc(sizeof(uint8_t *) * TILES_PER_BATCH);
		uint64_t * bitstream_length = (uint64_t *)malloc(sizeof(uint64_t)* TILES_PER_BATCH);
		uint32_t cstreamcount = 0;
		
		for(int32_t jj = 0 ; jj < BATCH_SIZE ; ++jj)
		{
			struct decode_tile_params dtparams = {}; 
			dtparams.bitstream_length  = bitstream_length;
			dtparams.bitstream_buffer = bitstream_buffer;
			dtparams.count = ARR_COUNT(folders);
			dtparams.nvjpeg2k_handle = nvjpeg2k_handles[jj];
			dtparams.nvjpeg2k_stream = nvjpeg2k_streams[jj];
			dtparams.decode_state = nvjpeg2k_decode_states[jj];
			dtparams.folder = folders[ii]; 
			dtparams.decode_output_p = decode_output_ps[jj];

			//nvjpeg2kCreateSimple(&dtparams.nvjpeg2k_handle);
			//nvjpeg2kStreamCreate(&dtparams.nvjpeg2k_stream);
			//nvjpeg2kDecodeStateCreate(nvjpeg2k_handle,&dtparams.decode_state);
			//cudaStreamCreateWithFlags(&dtparams.cstream,cudaStreamNonBlocking);
			dtparams.cstream = cstreams[jj];
		    //cstreams[cstreamcount++] = dtparams.cstream;

			dtparams.decode_output = decode_output; 
			
			uint32_t starttile = jj * TILES_PER_BATCH;
			uint32_t endtile = (jj+1) * TILES_PER_BATCH;
			dtparams.startindex = starttile;
			dtparams.endindex= endtile;

			fill_tile_bitstream_buffer_index_based(&dtparams);
			nu_j2k_decode_tiles_index_based(dtparams);
			//g_print("time per frame %lu\n", (g_get_real_time() - startframetime)/1000);
		}
		g_free(bitstream_length);
		g_free(bitstream_buffer[0]);
		g_free(bitstream_buffer);
		for(int32_t cs = 0; cs<cstreamcount;cs++)
		{
			cudaStreamSynchronize(cstreams[cs]);
		}
		//cudaDeviceSynchronize();
		//g_usleep(600 * 1000);

		void * decode_output_copy;
		//cudaMalloc(&decode_output_copy,FULL_FRAME_SIZE);
		//cudaMemcpy(decode_output_copy,decode_output,FULL_FRAME_SIZE,cudaMemcpyDeviceToDevice);

		//cudaMallocHost(&decode_output_copy,FULL_FRAME_SIZE);
		//cudaMemcpyAsync(decode_output_copy,decode_output,FULL_FRAME_SIZE,cudaMemcpyDeviceToHost,cstream);

		//dg_push_frame(decode_output,FULL_FRAME_SIZE);
		//cudaFree(decode_output_copy);
		g_print("after sync and copy time per frame %lu\n", (g_get_real_time() - startframetime)/1000);
		readtime = 0;
	}
	/*for(int32_t ii = 0 ; ii < ARR_COUNT(folders); ++ii)
	{
		struct decode_tile_params dtparams = {}; 
		dtparams.bitstream_length  = bitstream_length;
		dtparams.bitstream_buffer = bitstream_buffer;
		dtparams.count = ARR_COUNT(folders);
		dtparams.nvjpeg2k_handle = nvjpeg2k_handle;
		dtparams.nvjpeg2k_stream = nvjpeg2k_stream;
		dtparams.decode_state = decode_state;
		dtparams.decode_output = decode_output; 
		dtparams.folder = folders[ii]; 
		dtparams.startx = 0;
		dtparams.endx = 8;
		dtparams.starty = 0;
		dtparams.endy = 15;
		dtparams.count = 240;

		guint64 startframetime = g_get_real_time(); 
		fill_tile_bitstream_buffer(&dtparams);
		nu_j2k_decode_tiles(dtparams);
		dtparams.startx = 8;
		dtparams.endx = 16; 
		fill_tile_bitstream_buffer(&dtparams);
		nu_j2k_decode_tiles(dtparams);
		g_print("time per frame %lu\n", (g_get_real_time() - startframetime)/1000);
		//cudaDeviceSynchronize();
		//g_usleep(600 * 1000);

		void * decode_output_copy;
		//cudaMalloc(&decode_output_copy,FULL_FRAME_SIZE);
		//cudaMemcpy(decode_output_copy,decode_output,FULL_FRAME_SIZE,cudaMemcpyDeviceToDevice);

		//cudaMallocHost(&decode_output_copy,FULL_FRAME_SIZE);
		//cudaMemcpyAsync(decode_output_copy,decode_output,FULL_FRAME_SIZE,cudaMemcpyDeviceToHost,cstream);

		dg_push_frame(decode_output,FULL_FRAME_SIZE);
		//cudaFree(decode_output_copy);
		g_print("after sync and copy time per frame %lu\n", (g_get_real_time() - startframetime)/1000);
		readtime = 0;
	}
	*/
	uint64_t dt = (g_get_real_time() - starttime)/1000;
	g_print("time %lu\n", dt);
	g_print("time per full frame %lu\n", dt/ARR_COUNT(folders));
	g_usleep(1000 * 1000 * 3);
	dg_deinitialize();
	//g_usleep(1000*1000*10);

}
