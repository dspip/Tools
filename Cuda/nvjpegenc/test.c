#include "nvjpeg.h"
#include "cuda.h"
#include <stdio.h>
#include <glib.h>
#include <math.h>

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
    guint w = 640 ;
    guint h = 480;
    //guint w = 640 ;
    //guint h = 480 ;
#if 1 
    char * inputdata = NULL;
    cudaError_t err = cudaMalloc((void**)&inputdata,w * h * 3);

    char * inputdata0 = NULL;
    err = cudaMalloc((void**)&inputdata0,w * h * 3);

    char * filecontents = NULL;
    guint64 length = 0;
    gboolean filefound = g_file_get_contents ( "FRAME1", &filecontents, &length, NULL);
    if(filefound && length == w * h * 3)
    {
        cudaMemcpy2D(inputdata ,w *3 ,filecontents, w * 3 ,w * 3 ,h ,cudaMemcpyHostToDevice);
    }
    for (int i = 0 ; i <  640 * 480 * 3 ; i++) 
    {
        float sinI = sin(i);
        filecontents[i] = 255 * sin(i);
        if(sinI > 0.4)
        {
            i+=9;
        }
    }
    if(filefound && length == w * h * 3)
    {
        cudaMemcpy2D(inputdata0 ,w *3 ,filecontents, w * 3 ,w * 3 ,h ,cudaMemcpyHostToDevice);
    }

#else 
    unsigned char * inputdata = malloc(640 * 480 * 3);
    for(int i = 0 ; i < 640 * 480 * 3;++i)
    {
        if(i % 3 == 0)
            inputdata[i] = 255;
    }
#endif

    
    nvjpegImage_t nv_image1 = {};
    nv_image1.channel[0] = inputdata;
    nv_image1.pitch[0] = w * 3;

    nvjpegImage_t nv_image0 = {};

    nv_image0.channel[0] = inputdata0;
    nv_image0.pitch[0] = w * 3;
    // Fill nv_image with image data, let's say 640x480 image in RGB format

    guint64 iterations = 100 ;
    guint64 qualityiter = 100; 


    gint64 starttime = g_get_real_time(); 
    //char * outputdata = malloc(1000 * 1000 * 2); // ~2MB
    //
    char * outputdata =NULL;  
    err = cudaMalloc((void**)&outputdata, 1000 * 1000 * 2); // ~2MB

    g_print(" time %ld\n",starttime);
    GString * timingsdata = g_string_new("qp_fps_size\n");
    for(int q = 1 ; q < iterations; ++q)
    {
        nvjpegStatus_t jpegsetquality = nvjpegEncoderParamsSetQuality(nv_enc_params,q,stream);

        guint64 startencode = g_get_real_time();
        qualityiter = 1000; 
        for (int j = 0; j < qualityiter; j++) 
        {
            nvjpegImage_t * nv_image = (j % 2 == 0) ? &nv_image0 : &nv_image1;

            nvjpegStatus_t jpegencoded = nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params, nv_image, NVJPEG_INPUT_RGBI, w, h, stream);
            cudaStreamSynchronize(stream);
            
            //printf("encoded %d \n",jpegencoded);
            //To CPU
            //nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, stream);
            //To GPU
            nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, stream);
            //g_print("%d length %ld\n",q,length);

            // get stream itself
            nvjpegEncodeRetrieveBitstreamDevice(nv_handle, nv_enc_state, outputdata, &length, 0);
        }

        guint64 endencode = g_get_real_time();
        gdouble dt = (gdouble)(endencode - startencode)/(qualityiter);

        g_string_append_printf(timingsdata,"%d_%.1f_%ld\n",q,dt,length);
        //g_file_set_contents(filename,outputdata,length,0);
    }
    
    g_file_set_contents("timingdata",timingsdata->str,timingsdata->len,0);
    // get compressed stream size
    gint64 endtime= g_get_real_time(); 
    gint64 dtime = endtime-starttime;
    guint64 muliter = iterations*qualityiter;
    gfloat timeperiter = (gfloat)dtime/muliter;
    guint64 processedbytes = muliter * (w * h * 3);
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
