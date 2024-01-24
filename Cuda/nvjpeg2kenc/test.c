#include "nvjpeg2k.h"
#include "cuda.h"
#include <stdio.h>
#include <glib.h>
#define NUM_COMPONENTS 3

nvjpeg2kStatus_t SetEncoderConfig(nvjpeg2kEncodeConfig_t * enc_config,nvjpeg2kEncodeParams_t nv_enc_params ,guint32 image_width,guint32 image_height)
{

    nvjpeg2kImageComponentInfo_t image_comp_info[NUM_COMPONENTS];
    for (int c = 0; c < NUM_COMPONENTS; c++)
    {

        image_comp_info[c].component_width  = image_width;
        image_comp_info[c].component_height = image_height;
        image_comp_info[c].precision        = 8;
        image_comp_info[c].sgn              = 0;
        
    }
    memset(enc_config, 0, sizeof(enc_config));
    enc_config->stream_type      =  NVJPEG2K_STREAM_JP2; // the bitstream will be in JP2 container format
    enc_config->color_space      =  NVJPEG2K_COLORSPACE_SRGB; // input image is in RGB format
    enc_config->image_width      =  image_width;
    enc_config->image_height     =  image_height;
    enc_config->num_components   =  NUM_COMPONENTS;
    enc_config->image_comp_info  =  image_comp_info;
    enc_config->code_block_w     =  64;
    enc_config->code_block_h     =  64;
    enc_config->irreversible     =  0;
    enc_config->mct_mode         =  1;
    enc_config->prog_order       =  NVJPEG2K_LRCP;
    enc_config->num_resolutions  =  6;
    nvjpeg2kStatus_t status = nvjpeg2kEncodeParamsSetEncodeConfig(nv_enc_params, enc_config);

    double target_psnr = 50;
    status = nvjpeg2kEncodeParamsSetQuality(nv_enc_params, target_psnr);

    return status;

}

int main(int argc , char ** argv)
{
    //CUresult initState = cuInit (0);

    //printf("cuda state %d %d\n",initState,CUDA_SUCCESS);
    //nvjpeg2kHandle_t libhandle;
    nvjpeg2kEncoder_t nv_handle = NULL;
    nvjpeg2kEncodeState_t nv_enc_state = NULL;
    nvjpeg2kEncodeParams_t nv_enc_params = NULL;
    nvjpeg2kEncodeConfig_t enc_config;
    cudaStream_t stream;
    int quality = 90; 

    //nvjpeg2kStatus_t libstate = nvjpeg2kCreateSimple(&libhandle);
    // initialize nvjpeg structures
    nvjpeg2kStatus_t jpegstate = nvjpeg2kEncoderCreateSimple(&nv_handle);
    nvjpeg2kStatus_t jpegstatecreate = nvjpeg2kEncodeStateCreate(nv_handle,&nv_enc_state);
    nvjpeg2kStatus_t jpegparamscreate = nvjpeg2kEncodeParamsCreate(&nv_enc_params);
    
    guint w = 1920;
    guint h = 1080;
    nvjpeg2kStatus_t jpegsetquality = SetEncoderConfig(&enc_config,nv_enc_params,w,h);
    g_print("state %d %d %d %d \n", jpegstate,jpegstatecreate,jpegparamscreate,jpegsetquality);
    //guint w = 640 ;
    //guint h = 480 ;
#if 1 
    char * inputdata[NUM_COMPONENTS];
    cudaError_t err = cudaMalloc((void**)&inputdata[0],w * h);
    err = cudaMalloc((void**)&inputdata[1],w * h);
    err = cudaMalloc((void**)&inputdata[2],w * h);
    //cudaMemset2D(inputdata, 16920 * 3 ,255,300,108);
#else 
    unsigned char * inputdata = malloc(640 * 480 * 3);
    for(int i = 0 ; i < 640 * 480 * 3;++i)
    {
        if(i % 3 == 0)
            inputdata[i] = 255;
    }
#endif

    
    nvjpeg2kImage_t nv_image = {};
    size_t pitch_in_bytes[3] = {w,w,w};
    nv_image.pixel_data = (void**)inputdata;
    nv_image.pixel_type = NVJPEG2K_UINT8;
    nv_image.pitch_in_bytes = pitch_in_bytes;
    // Fill nv_image with image data, let's say 640x480 image in RGB format

    guint64 iterations = 1 ;

    gint64 starttime = g_get_real_time(); 
    char * outputdata = NULL;  
    g_print(" time %ld\n",starttime);
    for(int i = 0 ; i < iterations; ++i)
    {
        // Compress image
        nvjpeg2kStatus_t jpegencoded = nvjpeg2kEncode(nv_handle, nv_enc_state, nv_enc_params, &nv_image, stream);
        printf("encoded %d \n",jpegencoded);
        size_t length = 0;
        nvjpeg2kEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, stream);

        cudaStreamSynchronize(stream);
        // get stream itself
        outputdata = malloc(length);
        nvjpeg2kEncodeRetrieveBitstream(nv_handle, nv_enc_state, outputdata, &length, stream);
        if(i == iterations - 1)
        {
            g_print("%d length %ld\n",i,length);
            g_file_set_contents("test.jpeg",outputdata,length,0);
        }

        free(outputdata);
    }
    // get compressed stream size
    gint64 endtime= g_get_real_time(); 
    gint64 dtime = endtime-starttime;
    gfloat timeperiter = (gfloat)dtime/iterations;
    guint64 processedbytes = iterations * (w * h * 3);
    g_print("res %uX%u deltatime %fsec  t/iter : %fus image/sec: %f processed MB: %.3f mb/s: %f \n",w,h,(gfloat)dtime/1e6,timeperiter,1e6 /(timeperiter),processedbytes/1e6,processedbytes/(gfloat)dtime);

    //gulong microseconds;
    //gdouble encodingTime = g_timer_elapsed(tmr,&microseconds);

    // write stream to file
 //   printf("encoding time : %lf \n",encodingTime );
    //std::ofstream output_file("test.jpg", std::ios::out | std::ios::binary);
    //output_file.write(jpeg.data(), length);
    //output_file.close();
    return 0;


}
