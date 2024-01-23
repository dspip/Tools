#include "nvjpeg.h"
#include "cuda.h"
#include <stdio.h>
#include <glib.h>

int main(int argc , char ** argv)
{
    CUresult initState = cuInit (0);

    printf("cuda state %d %d\n",initState,CUDA_SUCCESS);

    nvjpegHandle_t nv_handle;
    nvjpegEncoderState_t nv_enc_state;
    nvjpegEncoderParams_t nv_enc_params;
    cudaStream_t stream;
    int quality = 90; 

    // initialize nvjpeg structures
    nvjpegStatus_t jpegstate = nvjpegCreateSimple(&nv_handle);

    nvjpegStatus_t jpegstatecreate = nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, stream);
    nvjpegStatus_t jpegparamscreate = nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, stream);
    nvjpegStatus_t jpegsetquality = nvjpegEncoderParamsSetQuality(nv_enc_params,quality,stream);
    nvjpegStatus_t jpegsetsamplingfactors = nvjpegEncoderParamsSetSamplingFactors(nv_enc_params, NVJPEG_CSS_444, stream);
    
    g_print("state %d %d %d %d %d\n",jpegstate,jpegstatecreate,jpegparamscreate,jpegsetquality,jpegsetsamplingfactors);
    guint w = 1920 ;
    guint h = 1080;
    //guint w = 640 ;
    //guint h = 480 ;
#if 1 
    char * inputdata = NULL;
    cudaError_t err = cudaMalloc((void**)&inputdata,w * h * 3);
    cudaMemset2D(inputdata, 16920 * 3 ,255,300,108);
#else 
    unsigned char * inputdata = malloc(640 * 480 * 3);
    for(int i = 0 ; i < 640 * 480 * 3;++i)
    {
        if(i % 3 == 0)
            inputdata[i] = 255;
    }
#endif

    
    nvjpegImage_t nv_image = {};
    nv_image.channel[0] = inputdata;
    nv_image.pitch[0] = w * 3;
    // Fill nv_image with image data, let's say 640x480 image in RGB format

    guint64 iterations = 1000 ;

    gint64 starttime = g_get_real_time(); 
    char * outputdata = NULL;  
    g_print(" time %ld\n",starttime);
    for(int i = 0 ; i < iterations; ++i)
    {
        // Compress image
        // g
        nvjpegStatus_t jpegencoded = nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params, &nv_image, NVJPEG_INPUT_RGBI, w, h, stream);
        printf("encoded %d \n",jpegencoded);
        size_t length = 0;
        nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, stream);
        g_print("%d length %ld\n",i,length);

        cudaStreamSynchronize(stream);
        // get stream itself
        outputdata = malloc(length);
        nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, outputdata, &length, 0);
        if(i == iterations - 1)
            g_file_set_contents("test.jpeg",outputdata,length,0);

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
