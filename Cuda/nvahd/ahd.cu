#include <cuda.h>
#include <glib.h>
//#include<cuda_util.h>

#define MEDIAN_FILTER_ITERATIONS 0
#define SAVEHORZVERT
#define SAVERESULT
#define SAVEHOMO
#define SAVE_RB

#define R 0
#define G 1
#define B 2
#define RGB 3

typedef guint8 pixel;
#define P 0  
#define pixel4 char4

#define  odd(n) ((n)&1)
#define even(n) (!odd((n)))

#define get_filter_color_grbg(x,y) (even(x) == even(y) ? G : odd(x) && even(y) ? R : B)

#define get_pix(buffer,x,y,width) ((buffer) + ((y) * (width) + (x))*RGB)

//texture<pixel4, 2, cudaReadModeElementType> src_g;
texture<guint8, 2, cudaReadModeElementType> homo_h_tex;
texture<guint8, 2, cudaReadModeElementType> homo_v_tex;

//#define tex_get_comp(tex,x,y,c) tex2D((src),(mirror((x),width))*3+(c),(mirror((y),height)))
//#define tex_get_comp(tex,x,y,c) tex2D((tex),(x)*3+(c),(y))
#define tex_get_color(tex,x,y,c) tex2D<pixel>((tex),((x)*4)+(c),(y))
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

__global__  void ahd_kernel_interp_rb(cudaTextureObject_t src_g,float4* g_result, pixel *g_tmp_result, int width, int height)
{
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

__device__ int median_diff(cudaTextureObject_t src, int x, int y, int chan1, int chan2) {
  int diffs[9];
  pixel val1,val2;
  /* Manual unroll */
  /* Avoids local memory use */

  val1 = tex_get_color(src,x+-1,y+-1,chan1);
  val2 = tex_get_color(src,x+-1,y+-1,chan2);
  diffs[0] = val1 - val2;

  val1 = tex_get_color(src,x,y-1,chan1);
  val2 = tex_get_color(src,x,y-1,chan2);
  diffs[1] = val1 - val2;

  val1 = tex_get_color(src,x+1,y+-1,chan1);
  val2 = tex_get_color(src,x+1,y+-1,chan2);
  diffs[2] = val1 - val2;

  val1 = tex_get_color(src,x+-1,y,chan1);
  val2 = tex_get_color(src,x+-1,y,chan2);
  diffs[3] = val1 - val2;

  val1 = tex_get_color(src,x,y,chan1);
  val2 = tex_get_color(src,x,y,chan2);
  diffs[4] = val1 - val2;

  val1 = tex_get_color(src,x+1,y,chan1);
  val2 = tex_get_color(src,x+1,y,chan2);
  diffs[5] = val1 - val2;

  val1 = tex_get_color(src,x+-1,y+1,chan1);
  val2 = tex_get_color(src,x+-1,y+1,chan2);
  diffs[6] = val1 - val2;

  val1 = tex_get_color(src,x,y+1,chan1);
  val2 = tex_get_color(src,x,y+1,chan2);
  diffs[7] = val1 - val2;

  val1 = tex_get_color(src,x+1,y+1,chan1);
  val2 = tex_get_color(src,x+1,y+1,chan2);
  diffs[8] = val1 - val2;

  //int m = median3(diffs,i); /* insertion sort */
  int m = median4(diffs); /* network sort */
  return m;
}

#define diff1(a,b) ((a) > (b)) ? ((a)-(b)) : ((b)-(a))
#define diff2(a1, a2, b1, b2) (((a1) - (b1)) * ((a1) - (b1)) + ((a2) - (b2)) * ((a2) - (b2)));
#define texLs(tex,x,y) ((tex2D<pixel4>((tex),(x),(y)).x))
#define texAs(tex,x,y) ((tex2D<pixel4>((tex),(x),(y)).y))
#define texBs(tex,x,y) ((tex2D<pixel4>((tex),(x),(y)).z))

#define get_homo(buffer,x,y,width) ((buffer) + ((y) * (width) + (x)))

#define ROUND(x) ((x)>=0?(long)((x)+0.5):(long)((x)-0.5))

#define inside(x,y,width,height) ((x)>=0 && (y)>=0 && (x)<(width) && (y)<(height))

#define HOMO_INNER_LOOP() if (inside(x+dx, y+dy, width, height)) { \
                                  nL = texLs(horz_tex, x+dx, y+dy ); \
                                  na = texAs(horz_tex, x+dx, y+dy ); \
                                  nb = texBs(horz_tex, x+dx, y+dy ); \
                                  lum_diff = diff1(horz.x, nL); \
                                  chrom_diff = diff2(horz.y,horz.z,na,nb); \
                                  if (lum_diff <= lum_thres && \
                                      chrom_diff <= chrom_thres){ \
                                    h_homo++; \
                                  } \
                                  nL = texLs(vert_tex, x+dx, y+dy ); \
                                  na = texAs(vert_tex, x+dx, y+dy ); \
                                  nb = texBs(vert_tex, x+dx, y+dy ); \
                                  lum_diff = diff1(vert.x, nL); \
                                  chrom_diff = diff2(vert.y,vert.z,na,nb); \
                                  if (lum_diff <= lum_thres && \
                                      chrom_diff <= chrom_thres){ \
                                    v_homo++; \
                                  } \
                           }

__device__ void cuCvLabtoRGB( float L, float a, float b,
    unsigned char *r_, unsigned char *g_, unsigned char *b_) {

  float x, y, z;

  L = (L + labLShift_32f)*(1.f/labLScale_32f);
  x = (L + a*0.002f);
  z = (L - b*0.005f);
  y = L*L*L;
  x = x*x*x;
  z = z*z*z;

  float g, r;
  b = x*labBx_32f + y*labBy_32f + z*labBz_32f;
  *b_ = ROUND(b*255);

  g = x*labGx_32f + y*labGy_32f + z*labGz_32f;
  *g_ = ROUND(g*255);

  r = x*labRx_32f + y*labRy_32f + z*labRz_32f;
  *r_ = ROUND(r*255);
}

__global__ void ahd_kernel_build_homo_map(cudaTextureObject_t horz_tex, cudaTextureObject_t vert_tex ,guchar *g_horz_res, guchar* g_vert_res, uint width, uint height)
{
    volatile int x = blockIdx.x*blockDim.x + threadIdx.x;
    volatile int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
    {
        return;
    }

    volatile float4 horz = tex2D<float4>(horz_tex,x,y);

    /* Homogenity differences have been calculated for horz and vert directions */
    /* Find the adaptive thresholds, the same threshold is used for horz and vert */
    /* Horizontal case, look at left and right values */
    /* Vertical case, look at top, bottom values */

    /* HORZ */
    // horizontal left and right values

    float lumdiff_1 = diff1(horz.x, texLs(horz_tex, x-1, y));
    float lumdiff_2 = diff1(horz.x, texLs(horz_tex, x+1, y));
    float max_h_lumdiff = MAX(lumdiff_1,lumdiff_2);

    float chromdiff_1 = diff2(horz.y, horz.z,
            texAs(horz_tex, x-1, y),
            texBs(horz_tex, x-1, y));

    float chromdiff_2 = diff2(horz.y, horz.z,
            texAs(horz_tex, x+1, y),
            texBs(horz_tex, x+1, y));

    float max_h_chromdiff = MAX(chromdiff_1,chromdiff_2);

    volatile float4 vert = tex2D<float4>(vert_tex,x,y);

    /* VERT */
    // vertical top and bottom values

    lumdiff_1 = diff1(vert.x, texLs(vert_tex, x, y-1));
    lumdiff_2 = diff1(vert.x, texLs(vert_tex, x, y+1));
    float max_v_lumdiff = MAX(lumdiff_1,lumdiff_2);

    chromdiff_1 = diff2(vert.y, vert.z,
            texAs(vert_tex, x, y-1),
            texBs(vert_tex, x, y-1));

    chromdiff_2 = diff2(vert.y, vert.z,
            texAs(vert_tex, x, y+1),
            texBs(vert_tex, x, y+1));

    float max_v_chromdiff = MAX(chromdiff_1,chromdiff_2);

    /* THRESHOLD */
    float lum_thres = MIN(max_h_lumdiff,max_v_lumdiff);
    float chrom_thres = MIN(max_h_chromdiff,max_v_chromdiff);


    /* Get the lum and chrom differences for the pixel in the
     * neighbourhood.
     */
    int h_homo = 0;
    int v_homo = 0;

    //
    //    /* Manual unroll */
    //    for (int dy = -BALL_DIST; dy <= BALL_DIST; dy++){
    //      for (int dx = -BALL_DIST; dx <= BALL_DIST; dx++) {
    //        if (dx == 0 && dy == 0) continue;
    //
    //            volatile float nL,na,nb,lum_diff,chrom_diff;
    //          nL = texLs(horz_tex, x+dx, y+dy);
    //          na = texAs(horz_tex, x+dx, y+dy);
    //          nb = texBs(horz_tex, x+dx, y+dy);
    //
    //          lum_diff = diff1(horz.x, nL);
    //          chrom_diff = diff2(horz.y,horz.z,na,nb);
    //
    //            if (lum_diff <= lum_thres &&
    //                chrom_diff <= chrom_thres){
    //              h_homo++;
    //            }
    //
    //          nL = texLs(vert_tex, x+dx, y+dy);
    //          na = texAs(vert_tex, x+dx, y+dy);
    //          nb = texBs(vert_tex, x+dx, y+dy);
    //
    //          lum_diff = diff1(vert.x, nL);
    //          chrom_diff = diff2(vert.y,vert.z,na,nb);
    //
    //            if (lum_diff <= lum_thres &&
    //                chrom_diff <= chrom_thres){
    //              v_homo++;
    //            }
    //
    //      }
    //    }

    float chrom_diff,lum_diff,na,nb,nL;
    int dx,dy;
    dx = -2; dy = -2; HOMO_INNER_LOOP();
    dx = -1; dy = -2; HOMO_INNER_LOOP();
    dx = 0; dy = -2; HOMO_INNER_LOOP();
    dx = 1; dy = -2; HOMO_INNER_LOOP();
    dx = 2; dy = -2; HOMO_INNER_LOOP();
    dx = -2; dy = -1; HOMO_INNER_LOOP();
    dx = -1; dy = -1; HOMO_INNER_LOOP();
    dx = 0; dy = -1; HOMO_INNER_LOOP();
    dx = 1; dy = -1; HOMO_INNER_LOOP();
    dx = 2; dy = -1; HOMO_INNER_LOOP();
    dx = -2; dy = 0; HOMO_INNER_LOOP();
    dx = -1; dy = 0; HOMO_INNER_LOOP();
    dx = 1; dy = 0; HOMO_INNER_LOOP();
    dx = 2; dy = 0; HOMO_INNER_LOOP();
    dx = -2; dy = 1; HOMO_INNER_LOOP();
    dx = -1; dy = 1; HOMO_INNER_LOOP();
    dx = 0; dy = 1; HOMO_INNER_LOOP();
    dx = 1; dy = 1; HOMO_INNER_LOOP();
    dx = 2; dy = 1; HOMO_INNER_LOOP();
    dx = -2; dy = 2; HOMO_INNER_LOOP();
    dx = -1; dy = 2; HOMO_INNER_LOOP();
    dx = 0; dy = 2; HOMO_INNER_LOOP();
    dx = 1; dy = 2; HOMO_INNER_LOOP();
    dx = 2; dy = 2; HOMO_INNER_LOOP();

    //    char4 result;
    //    result.x = clampc(h_homo);
    //    result.y = clampc(v_homo);
    //    g_homo_res[x+y*width] = result;
    *(get_homo(g_horz_res,x,y,width)) = h_homo;
    *(get_homo(g_vert_res,x,y,width)) = v_homo;
}


__global__ void ahd_kernel_choose_direction(cudaTextureObject_t homo_h_tex,cudaTextureObject_t homo_v_tex,cudaTextureObject_t horz_tex,cudaTextureObject_t vert_tex , pixel *g_result, float *g_direction, uint width, uint height)
{
    volatile int x = blockIdx.x*blockDim.x + threadIdx.x;
    volatile int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }
    int horz_score = 0;
    int vert_score = 0;
    for (int dy = -1; dy <= 1; dy++){
        for (int dx = -1; dx <= 1; dx++) {
            // todo divide the score by the area so that this
            // works properly at the borders
            horz_score += tex2D<pixel>(homo_h_tex,x+dx,y+dy);
            vert_score += tex2D<pixel>(homo_v_tex,x+dx,y+dy);

        }
    }

    if (g_direction) {
        if (vert_score > horz_score) {// ? VERT : HORZ;
            *get_homo(g_direction,x,y,width) = 0;
        } else {
            *get_homo(g_direction,x,y,width) = 255;
        }
    }

    float L,a,b;
    if (vert_score <= horz_score) {
        L = texLs(horz_tex,x,y);
        a = texAs(horz_tex,x,y);
        b = texBs(horz_tex,x,y);
    } else {
        L = texLs(vert_tex,x,y);
        a = texAs(vert_tex,x,y);
        b = texBs(vert_tex,x,y);
    }
    pixel *res = get_pix(g_result,x,y,width);
    cuCvLabtoRGB(L,a,b,res+R,res+G,res+B);
}

__global__ void ahd_kernel_remove_artefacts (cudaTextureObject_t src,pixel *g_result, uint width, uint height){
    volatile int x = blockIdx.x*blockDim.x + threadIdx.x;
    volatile int y = blockIdx.y*blockDim.y + threadIdx.y;
    if (x >= width || y >= height) {
        return;
    }

    //volatile pixel *dest = get_pix(g_result,x,y,width);

    int res = median_diff(src,x,y,R,G) + tex_get_color(src,x,y,G);
    g_result[(x+y*width)*4+R] = clampc(res);

    res = median_diff(src,x,y,B,G) + tex_get_color(src,x,y,G);
    g_result[(x+y*width)*4+B] = clampc(res);

    res = round((median_diff(src,x,y,G,R) +
             median_diff(src,x,y,G,B) +
             tex_get_color(src,x,y,R) +
             tex_get_color(src,x,y,B))/2.0);
    g_result[(x+y*width)*4+G] = clampc(res);
}

__global__ void AddIntsCuda(int * a , int * b)
{
    *a += *b;
}

cudaTextureObject_t SetupTextureAndData(void * d_data,gsize pitchBytes, gsize height,cudaChannelFormatDesc channelDesc)
{
    cudaTextureObject_t src_image = {};
    struct cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = d_data;
    resDesc.res.pitch2D.desc = channelDesc; 
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.pitchInBytes = pitchBytes;
    cudaTextureDesc tDesc = {};
    tDesc.readMode = cudaReadModeElementType;

    cudaError_t err;
    err = cudaCreateTextureObject(&src_image, &resDesc, &tDesc, NULL);
    g_print("err: %d\n",err);
    g_assert(err == cudaSuccess);

    return src_image;
}

void TestTextureObjectCreation()
{
    cudaError_t err;
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
    float4 * d_data; 
    gint width = 10000 * sizeof(float4);
    gint height= 7096;
    gsize pitch;

    err = cudaMallocPitch(&d_data,&pitch,width,height);

    cudaTextureObject_t src_image = {};
    struct cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = d_data;
    resDesc.res.pitch2D.desc = channelDesc; 
    //resDesc.res.pitch2D.width = pitch;
    resDesc.res.pitch2D.height = height;
    resDesc.res.pitch2D.pitchInBytes = pitch;
    cudaTextureDesc tDesc = {};
    tDesc.readMode = cudaReadModeElementType;

    err = cudaCreateTextureObject(&src_image, &resDesc, &tDesc, NULL);
    g_print("err: %d\n",err);
    g_assert(err == cudaSuccess);

}

int main(int argc , char ** argv)
{      
    //TestTextureObjectCreation();
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

        cudaTextureObject_t src_image = SetupTextureAndData(src_bayer,pitchWidth,height,pixel_channel);

        cudaError_t err;

        size_t dest_pbuf_size = pitchWidth * height * sizeof(pixel4);
        pixel4 *d_horz_g = NULL,* d_vert_g;
        
        gsize horzPitch = 0;
        gsize vertPitch = 0;
        cudaMallocPitch(&d_horz_g, &horzPitch, pitchWidth*sizeof(pixel4), height);
        cudaMallocPitch(&d_vert_g, &vertPitch, pitchWidth*sizeof(pixel4), height);

        dim3 threadblock(32,8);
        dim3 gridBlock((width  + threadblock.x - 1)/threadblock.x, (height + threadblock.y - 1)/threadblock.y);
        ahd_kernel_interp_g<<< gridBlock, threadblock >>>(src_image,d_horz_g,d_vert_g,width,height);
        err = cudaDeviceSynchronize();

        cudaTextureObject_t src_horz = SetupTextureAndData(d_horz_g,width*sizeof(pixel4),height,pixel4_channel);
        cudaTextureObject_t src_vert = SetupTextureAndData(d_vert_g,width*sizeof(pixel4),height,pixel4_channel);

        float4 * d_horz_result; 
        float4 * d_vert_result; 

        gsize horz4fPitch;
        gsize vert4fPitch;

        err = cudaMallocPitch(&d_horz_result,&horz4fPitch, width * sizeof(float4),height);
        err = cudaMallocPitch(&d_vert_result,&vert4fPitch, width * sizeof(float4),height);

        ahd_kernel_interp_rb<<<gridBlock,threadblock>>>(src_horz,d_horz_result , NULL , width, height);
        ahd_kernel_interp_rb<<<gridBlock,threadblock>>>(src_vert,d_vert_result , NULL , width, height);

        src_horz = SetupTextureAndData(d_horz_result,width*sizeof(float4),height,float4_channel);
        src_vert = SetupTextureAndData(d_vert_result,width*sizeof(float4),height,float4_channel);

        guchar * d_homo_horz; 
        guchar * d_homo_vert; 

        err = cudaMallocPitch(&d_homo_horz, &horzPitch, width, height);
        err = cudaMallocPitch(&d_homo_vert, &vertPitch, width, height);

        ahd_kernel_build_homo_map<<<gridBlock,threadblock>>>(src_horz,src_vert,d_homo_horz, d_homo_vert, width, height);

        cudaTextureObject_t homo_h_tex = SetupTextureAndData(d_homo_horz,horzPitch,height,pixel_channel);
        cudaTextureObject_t homo_v_tex = SetupTextureAndData(d_homo_vert,vertPitch,height,pixel_channel);
        pixel * d_res;
        float * d_direction; 

        err = cudaMalloc(&d_res, pitchWidth*height * sizeof(pixel4));

        err = cudaMalloc(&d_direction, sizeof(float) * pitchWidth*height);

        ahd_kernel_choose_direction<<<gridBlock,threadblock>>>(homo_h_tex, homo_v_tex, src_horz, src_vert, d_res, d_direction, width,height);

         for (uint i = 0; i < MEDIAN_FILTER_ITERATIONS ; i++)
         {
             g_print("Removing artifacts\n");
     
             gsize pitchArtifacts;
             pixel * d_result;
             err = cudaMallocPitch(&d_result, &pitchArtifacts, width * sizeof(pixel4), height);

             cudaTextureObject_t d_src_tex = SetupTextureAndData(d_res, width * sizeof(pixel4), height, pixel_channel); 
             ahd_kernel_remove_artefacts<<<gridBlock,threadblock>>>(d_src_tex ,d_result, width, height);
             d_res = d_result;
        }
        err = cudaDeviceSynchronize();

#ifdef SAVEHORZVERT 

        pixel4 * horz = (pixel4*)g_malloc0(width*height*sizeof(pixel4));
        pixel4 * vert = (pixel4*)g_malloc0(width*height*sizeof(pixel4));
        err = cudaMemcpy2D(horz,width * sizeof(pixel4), d_horz_g, width * sizeof(pixel4), width*sizeof(pixel4), height, cudaMemcpyDeviceToHost);
        g_assert(err == cudaSuccess);

        err = cudaMemcpy2D(vert,width * sizeof(pixel4), d_vert_g, width * sizeof(pixel4), width*sizeof(pixel4), height, cudaMemcpyDeviceToHost);
        g_assert(err == cudaSuccess);

        g_file_set_contents("horz.ppm",(gchar*)horz,width*height*sizeof(pixel4),&gerr);
        g_file_set_contents("vert.ppm",(gchar*)vert,width*height*sizeof(pixel4),&gerr);
        g_free(horz);
        g_free(vert);
        g_print("file length : %ld\n",length);
#endif

#ifdef SAVE_RB
        float4 * horz_rb = (float4*)g_malloc0(width*height*sizeof(float4));
        float4 * vert_rb = (float4*)g_malloc0(width*height*sizeof(float4));
        err = cudaMemcpy2D(horz_rb,width * sizeof(float4), d_horz_result, width * sizeof(float4), width*sizeof(float4), height, cudaMemcpyDeviceToHost);
        g_assert(err == cudaSuccess);
        err = cudaMemcpy2D(vert_rb,width * sizeof(float4), d_vert_result, width * sizeof(float4), width*sizeof(float4), height, cudaMemcpyDeviceToHost);
        g_assert(err ==  cudaSuccess);

        g_file_set_contents("rb_horz.ppm",(gchar*)horz_rb,width*height*sizeof(float4),&gerr);
        g_file_set_contents("rb_vert.ppm",(gchar*)vert_rb,width*height*sizeof(float4),&gerr);
#endif

#ifdef SAVERESULT
        pixel4 * resu = (pixel4*)g_malloc0(width*height*sizeof(pixel4));
        err = cudaMemcpy2D(resu,width * sizeof(pixel4),d_res, width * sizeof(pixel4), width*sizeof(pixel4), height, cudaMemcpyDeviceToHost);
        g_assert(err == cudaSuccess);

        g_file_set_contents("final_res.ppm",(gchar*)resu,width*height*sizeof(pixel4),&gerr);
#endif

#ifdef SAVEHOMO
        pixel4 * resuhomo = (pixel4*)g_malloc0(width*height*sizeof(pixel));

        err = cudaMemcpy2D(resuhomo,width ,d_homo_vert, width , width, height, cudaMemcpyDeviceToHost);
        g_assert(err == cudaSuccess);
        g_file_set_contents("homo_v_res.ppm",(gchar*)resuhomo,width*height*sizeof(pixel),&gerr);

        err = cudaMemcpy2D(resuhomo,width ,d_homo_horz, width , width, height, cudaMemcpyDeviceToHost);
        g_assert(err == cudaSuccess);
        g_file_set_contents("homo_h_res.ppm",(gchar*)resuhomo,width*height*sizeof(pixel),&gerr);

#endif

        int a = 5;
        int b = 10;
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
