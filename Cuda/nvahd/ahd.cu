#include <cuda.h>
#include <glib.h>
//#include<cuda_util.h>

//#define SAVEHORZVERT

#define R 0
#define G 1
#define B 2
#define RGB 3

typedef guint8 pixel;
#define P 16
#define pixel4 char4

#define  odd(n) ((n)&1)
#define even(n) (!odd((n)))

#define get_filter_color_grbg(x,y) (even(x) == even(y) ? G : odd(x) && even(y) ? R : B)

//texture<pixel4, 2, cudaReadModeElementType> src_g;
texture<guint8, 2, cudaReadModeElementType> homo_h_tex;
texture<guint8, 2, cudaReadModeElementType> homo_v_tex;

//#define tex_get_comp(tex,x,y,c) tex2D((src),(mirror((x),width))*3+(c),(mirror((y),height)))
//#define tex_get_comp(tex,x,y,c) tex2D((tex),(x)*3+(c),(y))
#define tex_get_color(tex,x,y,c) tex2D((tex),((x)*3)+(c),(y))
#define texR(tex,x,y) (tex2D<pixel4>((tex),(x),(y)).x)
#define texG(tex,x,y) (tex2D<pixel4>((tex),(x),(y)).y)
#define texB(tex,x,y) (tex2D<pixel4>((tex),(x),(y)).z)

#define cR(c4) (c4.x)
#define cG(c4) (c4.y)
#define cB(c4) (c4.z)

#define clampc(a) ((a) < 0) ? 0 : (((a) > 255) ? 255 : (guchar)(a))
#define tex2Du8 tex2D<guint8>

__global__ void ahd_kernel_interp_g(cudaTextureObject_t src, pixel4 * g_horz_res, pixel4 * g_vert_res, int width, int height)
{

  uint x = blockIdx.x*blockDim.x + threadIdx.x;
  uint y = blockIdx.y*blockDim.y + threadIdx.y;

  if (x < 2 || y < 2 || x >= width-2 || y >= height-2)
  {
      return;
  }
  int filter_color = get_filter_color_grbg(x,y);

  char4 h_res, v_res;
  /* Copy existing value to output */
  cR(h_res) = cR(v_res) = (filter_color == R) * tex2Du8(src,float(x),float(y));
  cG(h_res) = cG(v_res) = (filter_color == G) * tex2Du8(src,float(x),float(y));
  cB(h_res) = cB(v_res) = (filter_color == B) * tex2Du8(src,float(x),float(y));

  /* Interpolate Green values first */
  if (filter_color == R || filter_color == B)
  {
    /* Filter color is red or blue Interpolate green channel horizontally */
    /* Use existing green values */
    float sum = (tex2Du8(src,x-1,y) +
                 tex2Du8(src,x+1,y))/2.0f;

    /* And use existing red/blue values and apply filter 'h' */
    sum += (-tex2Du8(src,x-2,y)/4.0f +
             tex2Du8(src,x,  y)/2.0f +
            -tex2Du8(src,x+2,y)/4.0f)/4.0f;

    cG(h_res) = (guchar)clampc(sum);

    /* Interpolate green channel vertically */
    /* Use existing green values */
    sum = (tex2Du8(src,x,y-1) +
           tex2Du8(src,x,y+1))/2.0f;

    /* And use existing red/blue values and apply filter 'h' */
    sum += (-tex2Du8(src,x,y-2)/4.0f +
             tex2Du8(src,x,y  )/2.0f +
            -tex2Du8(src,x,y+2)/4.0f)/4.0f;

    cG(v_res) = (guchar)clampc(sum);
  }
  int res_index = (y*width + x);
  g_horz_res[res_index] = h_res;
  g_vert_res[res_index] = v_res;
}

#define  labXr_32f  0.433953f /* = xyzXr_32f / 0.950456 */
#define  labXg_32f  0.376219f /* = xyzXg_32f / 0.950456 */
#define  labXb_32f  0.189828f /* = xyzXb_32f / 0.950456 */

#define  labYr_32f  0.212671f /* = xyzYr_32f */
#define  labYg_32f  0.715160f /* = xyzYg_32f */
#define  labYb_32f  0.072169f /* = xyzYb_32f */

#define  labZr_32f  0.017758f /* = xyzZr_32f / 1.088754 */
#define  labZg_32f  0.109477f /* = xyzZg_32f / 1.088754 */
#define  labZb_32f  0.872766f /* = xyzZb_32f / 1.088754 */

#define  labRx_32f  3.0799327f  /* = xyzRx_32f * 0.950456 */
#define  labRy_32f  (-1.53715f) /* = xyzRy_32f */
#define  labRz_32f  (-0.542782f)/* = xyzRz_32f * 1.088754 */

#define  labGx_32f  (-0.921235f)/* = xyzGx_32f * 0.950456 */
#define  labGy_32f  1.875991f   /* = xyzGy_32f */
#define  labGz_32f  0.04524426f /* = xyzGz_32f * 1.088754 */

#define  labBx_32f  0.0528909755f /* = xyzBx_32f * 0.950456 */
#define  labBy_32f  (-0.204043f)  /* = xyzBy_32f */
#define  labBz_32f  1.15115158f   /* = xyzBz_32f * 1.088754 */

#define  labT_32f   0.008856f

#define cvCbrt(value) (__powf(value,1.0f/3.0f))

#define labSmallScale_32f  7.787f
#define labSmallShift_32f  0.13793103448275862f  /* 16/116 */
#define labLScale_32f      116.f
#define labLShift_32f      16.f
#define labLScale2_32f     903.3f

__global__  void ahd_kernel_interp_rb(cudaTextureObject_t src_g,float4* g_result, pixel *g_tmp_result, int width, int height) {
  uint x = blockIdx.x*blockDim.x + threadIdx.x;
  uint y = blockIdx.y*blockDim.y + threadIdx.y;

  // Take account of padding in source image
  x += P;
  y += P;

  if (x >= width-P || y >= height-P) {
    return;
  }
  pixel pixR = texR(src_g,x,y);
  pixel pixB = texB(src_g,x,y);

  guchar filter_color = get_filter_color_grbg(x,y);

  if (filter_color == R || filter_color == B) {
    /* Filter color is red or blue, interpolate missing red or blue channel */
    /* This function operates the same for horiz and vert interpolation */

    int dest_color = (filter_color == R) ? B : R;
    /* Get the difference between the Red/Blue and Green
     * channels */
    float sum =   (-texG(src_g,x-1,y-1)) +
                  (-texG(src_g,x-1,y+1)) +
                  (-texG(src_g,x+1,y-1)) +
                  (-texG(src_g,x+1,y+1));
  if (dest_color == R) {
    sum += texR(src_g,x-1,y-1) +
           texR(src_g,x-1,y+1) +
           texR(src_g,x+1,y-1) +
           texR(src_g,x+1,y+1);
  } else {
    sum += texB(src_g,x-1,y-1) +
           texB(src_g,x-1,y+1) +
           texB(src_g,x+1,y-1) +
           texB(src_g,x+1,y+1);
    }
    /* Apply low pass filter to the difference */
    sum /= 4.0;
    /* Use interpolated or interpolated green value */
    sum += texG(src_g,x,y);
    pixel res = clampc(round(sum));
    if (filter_color == R) {
      pixR = texR(src_g,x,y);
      pixB = res;
    } else {
      pixB = texB(src_g,x,y);
      pixR = res;
    }
    //res_pix[dest_color] = clampc(round(sum));
  } else {
    /* Filter color is green */
    /* Interpolate Red and Blue channels */
    /* This function operates the same for horz and vert interpolation */
    float sum = 0;
    /* Interpolate Red */
    if (even(y)){
      /* Red/Green rows */
      /* Use left and right pixels */
      /* Get the difference between the Red and Green
       * channel (use only the sampled Green values) */
      sum = (texR(src_g,x-1,y) - texG(src_g,x-1,y)) +
            (texR(src_g,x+1,y) - texG(src_g,x+1,y));
    } else {
      /* Blue/Green rows */
      /* Use top and bottom values */
      sum = (texR(src_g,x,y-1) - texG(src_g,x,y-1)) +
            (texR(src_g,x,y+1) - texG(src_g,x,y+1));
    }
    /* Apply low pass filter */
    sum /= 2.0;
    sum += texG(src_g,x,y);
    pixR = clampc(round(sum));;
    //Info("%d,%d Red val %f",x,y,sum);

    /* Interpolate Blue */
    if (odd(y)) {
      /* Blue/Green rows */
      /* Use left and right pixels */
      /* Get the difference between the Red and Green
       * channel (use only the sampled Green values) */
      sum = (texB(src_g,x-1,y) - texG(src_g,x-1,y)) +
            (texB(src_g,x+1,y) - texG(src_g,x+1,y));
    } else {
      /* Red/Green rows */
      /* Use top and bottom values */
      sum = (texB(src_g,x,y-1) - texG(src_g,x,y-1)) +
            (texB(src_g,x,y+1) - texG(src_g,x,y+1));
    }
    /* Apply low pass filter */
    sum /= 2.0;
    sum += texG(src_g,x,y);
    pixB = clampc(round(sum));
    //Info("%d,%d pixB : %d , sum %0.2f G:%d",x,y,pixB,sum,texG(src_g,x,y));
  }

  uint dest_width = width - 2*P;
  int dx = x - P;
  int dy = y - P;

#ifndef _TEST
  if (g_tmp_result != NULL) {
    // During testing, skip global memory access
    pixel *res = &g_tmp_result[y * width + x];

    res[R] = pixR;
    res[G] = texG(src_g,x,y);
    res[B] = pixB;
  }
#endif

  //cuCvRGBtoLab(pixR, pixG, pixB, &res_pix->x, &res_pix->y, &res_pix->z);
  // inlining to avoid passing point arguments

  float4 lab;
  float b = pixB/255.0, r = pixR/255.0;
  float g = texG(src_g,x,y)/255.0;
  float x_, y_, z;

  x_ = b*labXb_32f + g*labXg_32f + r*labXr_32f;
  y_ = b*labYb_32f + g*labYg_32f + r*labYr_32f;
  z =  b*labZb_32f + g*labZg_32f + r*labZr_32f;

  if( x_ > labT_32f )
    x_ = cvCbrt(x_);
  else
    x_ = x_*labSmallScale_32f + labSmallShift_32f;

  if( z > labT_32f )
    z = cvCbrt(z);
  else
    z = z*labSmallScale_32f + labSmallShift_32f;

  if( y_ > labT_32f )
  {
    y_ = cvCbrt(y_);
    lab.x = y_*labLScale_32f - labLShift_32f; // L
  }
  else
  {
    lab.x = y_*labLScale2_32f; // L
    y_ = y_*labSmallScale_32f + labSmallShift_32f;
  }

  lab.y = 500.f*(x_ - y_); // a
  lab.z = 200.f*(y_ - z); // b

  g_result[dx + (dy*dest_width)] = lab;
}

#define SORT(a,b) { if ((a)>(b)) SWAPV((a),(b)); }
#define SWAPV(a,b) { int temp=(a);(a)=(b);(b)=temp; }
__device__ int median4(int * p) 
{
    SORT(p[1], p[2]) ; SORT(p[4], p[5]) ; SORT(p[7], p[8]) ;
    SORT(p[0], p[1]) ; SORT(p[3], p[4]) ; SORT(p[6], p[7]) ;
    SORT(p[1], p[2]) ; SORT(p[4], p[5]) ; SORT(p[7], p[8]) ;
    SORT(p[0], p[3]) ; SORT(p[5], p[8]) ; SORT(p[4], p[7]) ;
    SORT(p[3], p[6]) ; SORT(p[1], p[4]) ; SORT(p[2], p[5]) ;
    SORT(p[4], p[7]) ; SORT(p[4], p[2]) ; SORT(p[6], p[4]) ;
    SORT(p[4], p[2]) ; 
    return(p[4]); 
}

__global__ void AddIntsCuda(int * a , int * b)
{
    *a += *b;
}
cudaTextureObject_t SetupTextureAndData(void * d_data,gsize pitchBytes,gsize widthBytes, gsize height,cudaChannelFormatDesc channelDesc)
{

    cudaTextureObject_t src_image = {};
    struct cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = d_data;
    resDesc.res.pitch2D.desc = channelDesc; 
    resDesc.res.pitch2D.width = widthBytes;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.pitchInBytes = pitchBytes;
    cudaTextureDesc tDesc = {};
    tDesc.readMode = cudaReadModeElementType;

    cudaError_t err;
    err = cudaCreateTextureObject(&src_image, &resDesc, &tDesc, NULL);
    g_print("err %d\n",err);
    g_assert(err == cudaSuccess);

    return src_image;

}


int main(int argc , char ** argv)
{
    if(argc != 2)
    {
        g_print("Enter input file for testing\n");
        return -1;
    }

    GError * gerr = NULL;

    gchar * data;
    gsize length;

    gsize width = 10000;
    gsize pitchWidth = 0;
    gsize height = 7096; 

    if(g_file_get_contents(argv[1],&data,&length,&gerr) && width * height == length)
    {
        cudaChannelFormatDesc pixel_channel = cudaCreateChannelDesc<pixel>();
        cudaChannelFormatDesc pixel4_channel = cudaCreateChannelDesc<pixel4>();
        cudaChannelFormatDesc float4_channel = cudaCreateChannelDesc<float4>();
        cudaChannelFormatDesc float_channel = cudaCreateChannelDesc<float>();

        void * src_bayer = NULL;
        cudaMallocPitch(&src_bayer,&pitchWidth,width,height);
        cudaMemcpy2D(src_bayer,pitchWidth,data,width,width,height,cudaMemcpyHostToDevice);

        cudaTextureObject_t src_image = {};

        struct cudaResourceDesc resDesc = {};
        resDesc.resType = cudaResourceTypePitch2D;
        //resDesc.resType = cudaResourceTypeArray;
        resDesc.res.pitch2D.devPtr = src_bayer;
        resDesc.res.pitch2D.desc = pixel_channel; 
        resDesc.res.pitch2D.width = pitchWidth;
        resDesc.res.pitch2D.height = height;
        resDesc.res.pitch2D.pitchInBytes = pitchWidth;
        cudaTextureDesc tDesc = {};
       // tDesc.filterMode = cudaFilterModeLinear;
        tDesc.readMode = cudaReadModeElementType;

        cudaError_t err;
        err = cudaCreateTextureObject(&src_image, &resDesc, &tDesc, NULL);
        g_print("err %d\n",err);
        g_assert(err == cudaSuccess);

        size_t dest_pbuf_size = pitchWidth * height * sizeof(pixel4);
        pixel4 *d_horz_g = NULL,* d_vert_g;
        
        gsize horzPitch = 0;
        gsize vertPitch = 0;
        cudaMallocPitch(&d_horz_g, &horzPitch, pitchWidth*sizeof(pixel4), height);
        cudaMallocPitch(&d_vert_g, &vertPitch, pitchWidth*sizeof(pixel4), height);

        dim3 threadblock(32,8);
        dim3 gridBlock((width  + threadblock.x - 1)/threadblock.x, (height + threadblock.y - 1)/threadblock.y);
        ahd_kernel_interp_g<<< gridBlock, threadblock >>>(src_image,d_horz_g,d_vert_g,width,height);

        cudaTextureObject_t src_horz = SetupTextureAndData(d_horz_g,horzPitch,horzPitch,height,pixel4_channel);

        cudaTextureObject_t src_vert = SetupTextureAndData(d_horz_g,vertPitch,vertPitch,height,pixel4_channel);

        float4 * d_horz_result; 
        float4 * d_vert_result; 

        cudaMalloc(&d_horz_result, horzPitch*height);
        cudaMalloc(&d_vert_result, vertPitch*height);

        ahd_kernel_interp_rb<<<gridBlock,threadblock>>>(src_horz,d_horz_result , NULL , width, height);

        ahd_kernel_interp_rb<<<gridBlock,threadblock>>>(src_vert,d_vert_result , NULL , width, height);

        cudaDeviceSynchronize();


#ifdef SAVEHORZVERT 

        pixel4 * horz = (pixel4*)g_malloc0(width*height*sizeof(pixel4));
        pixel4 * vert = (pixel4*)g_malloc0(width*height*sizeof(pixel4));
        err = cudaMemcpy2D(horz,width * sizeof(pixel4), d_horz_g, width * sizeof(pixel4), width*sizeof(pixel4), height, cudaMemcpyDeviceToHost);
        g_assert(err == cudaSuccess);

        err = cudaMemcpy2D(vert,width * sizeof(pixel4), d_vert_g, width * sizeof(pixel4), width*sizeof(pixel4), height, cudaMemcpyDeviceToHost);
        g_assert(err == cudaSuccess);
        for (int i = 0; i < width*height; i++)
        {
            horz[i].w = 255;
            vert[i].w = 255;
        }
        g_file_set_contents("horz.ppm",(gchar*)horz,width*height*sizeof(pixel4),&gerr);
        g_file_set_contents("vert.ppm",(gchar*)vert,width*height*sizeof(pixel4),&gerr);
        g_free(horz);
        g_free(vert);
        g_print("file length : %ld\n",length);
#endif

        int a= 5;
        int  b = 10;
        int * d_a;
        int * d_b;

        cudaMalloc(&d_a,sizeof(int));
        cudaMalloc(&d_b,sizeof(int));

        cudaMemcpy(d_a,&a,sizeof(int),cudaMemcpyHostToDevice);
        cudaMemcpy(d_b,&b,sizeof(int),cudaMemcpyHostToDevice);

        AddIntsCuda<<<1,1>>>(d_a,d_b);
        cudaMemcpy(&a,d_a,sizeof(int),cudaMemcpyDeviceToHost);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(src_bayer);

        
        g_print("Result is %d\n",a);
    }

}