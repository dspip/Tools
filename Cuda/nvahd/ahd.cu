#include <cuda.h>
#include <glib.h>
//#include<cuda_util.h>

#define R 0
#define G 1
#define B 2
#define RGB 3

typedef guint8 pixel;
#define P 16
#define pixel4 char4

#define  odd(n) ((n)&1)
#define even(n) (!odd((n)))

#define get_filter_color(x,y) (even(x) ? (even(y) ? R : G) : (odd(y) ? B : G))

//texture<pixel4, 2, cudaReadModeElementType> src_g;
texture<float4, 2, cudaReadModeElementType> horz_tex;
texture<float4, 2, cudaReadModeElementType> vert_tex;
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

__global__ void ahd_kernel_interp_g(cudaTextureObject_t src, pixel4* g_horz_res, pixel4* g_vert_res, int width, int height)
{
  uint x = blockIdx.x*blockDim.x + threadIdx.x;
  uint y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x < 2 || y < 2 || x >= width-2 || y >= height-2) {
    return;
  }
  int filter_color = get_filter_color(x,y);

  char4 h_res, v_res;
  /* Copy existing value to output */
  cR(h_res) = cR(v_res) = (filter_color == R) * tex2Du8(src,float(x),float(y));
  cG(h_res) = cG(v_res) = (filter_color == G) * tex2Du8(src,float(x),float(y));
  cB(h_res) = cB(v_res) = (filter_color == B) * tex2Du8(src,float(x),float(y));


  /* Interpolate Green values first */
  if (filter_color == R || filter_color == B) {
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

  guchar filter_color = get_filter_color(x,y);

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
    //pixel *res = g_tmp_result + res_index;
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

__global__ void AddIntsCuda(int * a , int * b)
{
    *a += *b;
}

int main()
{
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

    g_print("Result is %d\n",a);

}
