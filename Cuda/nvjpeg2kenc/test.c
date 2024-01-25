#include "nvjpeg2k.h"
#include "cuda.h"
#include <stdio.h>
#include <glib.h>
#include <math.h>
#define NUM_COMPONENTS 3

//#define RANDOMIMAGE

int main(int argc , char ** argv)
{
    CUresult initState = cuInit (0);

    //printf("cuda state %d %d\n",initState,CUDA_SUCCESS);
    nvjpeg2kHandle_t libhandle = NULL;
    nvjpeg2kEncoder_t nv_handle = NULL;
    nvjpeg2kEncodeState_t nv_enc_state = NULL;
    nvjpeg2kEncodeParams_t nv_enc_params = NULL;
    nvjpeg2kEncodeConfig_t enc_config;
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    int quality = 10 ; 

    nvjpeg2kStatus_t libstate = nvjpeg2kCreateSimple(&libhandle);
    // initialize nvjpeg structures
    nvjpeg2kStatus_t jpegstate = nvjpeg2kEncoderCreateSimple(&nv_handle);
    nvjpeg2kStatus_t jpegstatecreate = nvjpeg2kEncodeStateCreate(nv_handle,&nv_enc_state);
    nvjpeg2kStatus_t jpegparamscreate = nvjpeg2kEncodeParamsCreate(&nv_enc_params);

    nvjpeg2kImage_t nv_image = {};
    guint w = 640;
    guint h = 480; 
    //nvjpeg2kStatus_t jpegsetquality = SetEncoderConfig(&enc_config,nv_enc_params,w,h,&nv_image);

    nvjpeg2kImageComponentInfo_t image_comp_info[NUM_COMPONENTS];
    for (int c = 0; c < NUM_COMPONENTS; c++)
    {
        image_comp_info[c].component_width  = w;
        image_comp_info[c].component_height = h;
        image_comp_info[c].precision        = 8;
        image_comp_info[c].sgn              = 0;
    }
    memset(&enc_config, 0, sizeof(enc_config));
    enc_config.stream_type      =  NVJPEG2K_STREAM_JP2; // the bitstream will be in JP2 container format
    //enc_config.color_space      =  NVJPEG2K_COLORSPACE_SRGB; // input image is in RGB format
    //
    if(NUM_COMPONENTS == 1 )
        enc_config.color_space      =  NVJPEG2K_COLORSPACE_SRGB; // input image is in RGB format
    else 
        enc_config.color_space      =  NVJPEG2K_COLORSPACE_SRGB; // input image is in RGB format
    enc_config.image_width      =  w;
    enc_config.image_height     =  h;
    enc_config.num_components   =  NUM_COMPONENTS;
    enc_config.image_comp_info  =  image_comp_info;
    enc_config.code_block_w     =  64;
    enc_config.code_block_h     =  64; 
    enc_config.irreversible     =  0;
    enc_config.mct_mode         =  1;
    enc_config.prog_order       =  NVJPEG2K_LRCP;
    enc_config.num_resolutions  =  1;
    nvjpeg2kStatus_t status = nvjpeg2kEncodeParamsSetEncodeConfig(nv_enc_params, &enc_config);

    g_print("Set encode params %d \n",status); 
    unsigned char *pixel_data[NUM_COMPONENTS];
    size_t pitch_in_bytes[NUM_COMPONENTS];


    gchar * filecontents = NULL; 
    guint64 length = 0; 

    gboolean filefound = g_file_get_contents ( "FRAME1", &filecontents, &length, NULL);
    guint8 * data[3]; 
    if(filefound)
    {
        data[0] = malloc(w*h);
        data[1] = malloc(w*h);
        data[2] = malloc(w*h);
        guint32 j = 0;

        for (guint32 i = 0; i < w*h*3; i+=3) 
        {
            data[0][j] = filecontents[i]; 
            data[1][j] = filecontents[i + 1]; 
            data[2][j] = filecontents[i + 2]; 
            j++;
            
        }
    }

    for (int c = 0; c < NUM_COMPONENTS; c++)
    {
        CUresult res = cudaMalloc((void**)&(pixel_data[c]), w*h);
        pitch_in_bytes[c] = w;

        // cudaMallocPitch is used to let cuda deterimine the pitch. cudaMalloc can be used if required.
        g_print("%ld %d %p\n",pitch_in_bytes[c],res,pixel_data[c]); 
#ifdef RANDOMIMAGE 
        if(c == 0)
        {
            for (int i = 0; i < h ; i += 20)
            {
                cudaMemset2D(pixel_data[c] + i * w ,w/20, 255, 20,20);
            }
        }
        else if(c == 1)
            for (int i = 0; i < h ; i += 50)
            {
                cudaMemset2D(pixel_data[c] + i * w ,w/16, 255 - i /10, 50,50);
            }
        else if(c == 2)
        {
            cudaMemset2D(pixel_data[c] ,w, 100, w,h/2);
            cudaMemset2D(pixel_data[c] + (h/4)* w ,w/4, 255, w/2,h/2);
        }
#else 

        if(filefound && w * h * 3 == length)
        {
            res = cudaMemcpy2D(pixel_data[c] ,w ,data[c] ,w ,w ,h ,cudaMemcpyHostToDevice);
        }

#endif 
    }

#ifdef RANDOMIMAGE 
    for (int i = 0; i < w * h; i+=4) 
    {
        float sinI = (sin(i) + 1 )/2;
        if(sinI < 0.95)
        {
            cudaMemset2D(pixel_data[0] + i ,w, sinI * 255, 1,1);
            cudaMemset2D(pixel_data[1] + i ,w, sinI * 255, 1,1);
            cudaMemset2D(pixel_data[2] + i ,w, sinI * 255, 1,1);
        }
    }
#endif

    double target_psnr = 30; 
    nvjpeg2kStatus_t jpegsetquality = nvjpeg2kEncodeParamsSetQuality(nv_enc_params, target_psnr);

    nv_image.pixel_data = (void**)pixel_data;
    nv_image.pixel_type = NVJPEG2K_UINT8;
    nv_image.num_components = NUM_COMPONENTS;
    nv_image.pitch_in_bytes = pitch_in_bytes;

    g_print("state %d %d %d %d %d \n",libstate , jpegstate,jpegstatecreate,jpegparamscreate,jpegsetquality);
    //guint w = 640 ;
    //guint h = 480 ;
    //
    guint64 iterations = 100;

    guint64 starttime = g_get_real_time(); 
    char * outputdata = NULL;  
    g_print("time %lums %lus %lum %luh %lud %luy\n",starttime, starttime/(1000l * 1000l), starttime/(1000l * 1000l * 60),starttime/(1000l * 1000l * 60 * 60),starttime/(1000l * 1000l * 60 * 60 * 24), starttime/(1000l * 1000l * 60 * 60 * 24 * 365));

    cudaEvent_t startEvent = NULL, stopEvent;
    cudaError_t e = cudaEventCreateWithFlags(&startEvent,cudaEventBlockingSync);
    g_print("error start: %d\n",e);
    e = cudaEventCreateWithFlags(&stopEvent,cudaEventBlockingSync);
    g_print("error stop : %d\n",e);

    e = cudaEventRecord(startEvent, stream);
    g_print("record : %d\n",e);

    guint64 prevtime = g_get_real_time();
    for(int i = 0 ; i < iterations; ++i)
    {
        nvjpeg2kStatus_t jpegsetquality = nvjpeg2kEncodeParamsSetQuality(nv_enc_params, i);
        cudaEvent_t startEvent = NULL, stopEvent;
        // Compress image
        
        
        guint64 startencode = g_get_real_time();
        nvjpeg2kStatus_t jpegencoded = nvjpeg2kEncode(nv_handle, nv_enc_state, nv_enc_params, &nv_image, stream);
        //if(i% 10 == 0 )
        //{
        //    guint64 currenttime = g_get_real_time();
        //    g_print("%d : encoded \%d t %ld \n",i,jpegencoded,(currenttime-prevtime)/1000 );
        //    prevtime = currenttime;
        //}
        nvjpeg2kEncodeRetrieveBitstream(nv_handle, nv_enc_state, NULL, &length, stream);

        if(outputdata == NULL)
            outputdata = malloc(length);

        cudaStreamSynchronize(stream);
        // get stream itself
        nvjpeg2kEncodeRetrieveBitstream(nv_handle, nv_enc_state, outputdata, &length, stream);
        guint64 endencode = g_get_real_time();

        gchar * filename = g_strdup_printf("images/test_%d.j2k",i);
        g_print("%d length %ld\n",i,length);
        g_file_set_contents(filename,outputdata,length,0);

    }
    // get compressed stream size
    gint64 endtime= g_get_real_time(); 
    gint64 dtime = endtime-starttime;
    gfloat timeperiter = (gfloat)dtime/iterations;
    guint64 processedbytes = iterations * (w * h * 3);
    g_print("res %uX%u  | total deltatime %.3fsec | t/iter : %.3fus | image/sec: %.3f |  processed MB: %.3f  | MB/s: %f \n",w,h,(gfloat)dtime/1e6,timeperiter,1e6 /(timeperiter),processedbytes/1e6,processedbytes/(gfloat)dtime);

    free(outputdata);

    //gulong microseconds;
    //gdouble encodingTime = g_timer_elapsed(tmr,&microseconds);

    // write stream to file
    //   printf("encoding time : %lf \n",encodingTime );
    //std::ofstream output_file("test.jpg", std::ios::out | std::ios::binary);
    //output_file.write(jpeg.data(), length);
    //output_file.close();
    return 0;


}
