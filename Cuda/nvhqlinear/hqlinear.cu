
#include <cuda.h>
#include <glib.h>
//#include<cuda_util.h>

#define R 0
#define G 1
#define B 2
#define RGB 3
#define RGBA 4

typedef guint8 pixel;
#define P 2 
#define pixel3 char3
#define upixel4 uchar4

#define  odd(n) ((n)&1)
#define even(n) (!odd((n)))

#define get_filter_color_grbg(x,y) (even(x) == even(y) ? G : odd(x) && even(y) ? R : B)

#define get_pix(buffer,x,y,width) ((buffer) + (y) * (width) + (x))

#define tex_get_color(tex,x,y,c) (tex2D<upixel4>((tex),x,y).x * (c==R) + tex2D<pixel4>((tex),x,y).y * (c == G) + tex2D<pixel4>((tex),x,y).z * (c == B))
#define texR(tex,x,y) (tex2D<upixel4>((tex),(x),(y)).x)
#define texG(tex,x,y) (tex2D<upixel4>((tex),(x),(y)).y)
#define texB(tex,x,y) (tex2D<upixel4>((tex),(x),(y)).z)

#define cR(c4) (c4.x)
#define cG(c4) (c4.y)
#define cB(c4) (c4.z)

#define clampc(a) a > 255 ? 255 : (guchar)a; 
#define round(x) ((x)>=0?(long)((x)+0.5):(long)((x)-0.5))

#define tex2Du8 tex2D<guint8>

#define SQ(x) ((x)*(x))

__device__ inline float lerp(float a ,float b,float t)
{
    return (1.0 - t) * a + (t * b);
}

//Real time demosaicing for embedded system 
__device__ inline float rws_get(float s)
{
    //return 0;
    return (s/128) - 1.0f;  
}

#define CudaMallocP(dest,pitch, width,height) g_assert(cudaSuccess == cudaMallocPitch(dest,pitch,width,height))
#define CudaSync()  g_assert(cudaSuccess == cudaDeviceSynchronize())
#define CudaMemcpy2D(dst,dwidth, src, pitch,width, height, direction) g_assert(cudaSuccess == cudaMemcpy2D(dst,dwidth,src,width,width,height,direction))
#define CudaMalloc(dst,size) g_assert(cudaSuccess == cudaMalloc(dst,size));

#define diff1(a,b) ((a) > (b)) ? ((a)-(b)) : ((b)-(a))
#define diff2(a1, a2, b1, b2) (((a1) - (b1)) * ((a1) - (b1)) + ((a2) - (b2)) * ((a2) - (b2)));
#define texLs(tex,x,y) ((tex2D<float4>((tex),(x),(y)).x))
#define texAs(tex,x,y) ((tex2D<float4>((tex),(x),(y)).y))
#define texBs(tex,x,y) ((tex2D<float4>((tex),(x),(y)).z))

#define inside(x,y,width,height) ((x)>=0 && (y)>=0 && (x)<(width) && (y)<(height))

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

    CudaMallocP(&d_data,&pitch,width,height);

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

__global__ void MHCdemosaic(cudaTextureObject_t src, upixel4 *img, int width, int height) {

    // In our sensor, upper-right is R
    const int2 firstRed = {1, 0};

    uint x = blockIdx.x*blockDim.x + threadIdx.x;
    uint y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x < 2 || y < 2 || x >= width-2 || y >= height-2)
    {
        return;
    }

    // Gets information about work-item (which pixel to process)
    // Add offset for first red
    int2 centerRed = {x + firstRed.x, y + firstRed.y};

    int4 xCoord = {x - 2, x - 1, x + 1, x + 2};
    int4 yCoord = {y - 2, y - 1, y + 1, y + 2};

    float C = tex2Du8(src,x,y);

    const float4 kC = {4.0f / 8.0f, 6.0f / 8.0f, 5.0f / 8.0f, 5.0f / 8.0f};

    // Determine which of four types of pixels we are on.
    //Note(lev): not sure about this read about fmodf
    float2 alternate = {fmodf(centerRed.x , 2.0f), fmodf(centerRed.y, 2.0f)};

    float4 Dvec = {tex2Du8(src,xCoord.y, yCoord.y),
                   tex2Du8(src,xCoord.y, yCoord.z),
                   tex2Du8(src,xCoord.z, yCoord.y),
                   tex2Du8(src,xCoord.z, yCoord.z)};

    float4 PATTERN = {kC.x * C, kC.y * C, kC.z * C, kC.w * C};

    float D = Dvec.x + Dvec.y + Dvec.z + Dvec.w;

    float4 value = {tex2Du8(src,x       ,  yCoord.x), 
                    tex2Du8(src,x       ,  yCoord.y),
                    tex2Du8(src,xCoord.x,  y), 
                    tex2Du8(src,xCoord.y,  y)};

    float4 temp = {tex2Du8(src, x,       yCoord.w), 
                   tex2Du8(src, x,       yCoord.z), 
                   tex2Du8(src, xCoord.w, y), 
                   tex2Du8(src, xCoord.z, y)};

    //Note(Lev) coords look ok so far idk

    const float4 kA = {-1.0f / 8.0f, -1.5f / 8.0f, 0.5f / 8.0f, -1.0f / 8.0f};
    const float4 kB = {2.0f / 8.0f, 0.0f, 0.0f, 4.0f / 8.0f};
    const float4 kD = {0.0f, 2.0f / 8.0f, -1.0f / 8.0f, -1.0f / 8.0f};

    const float4 kE = {kA.x, kA.y, kA.w, kA.z}; 
    const float4 kF = {kB.x, kB.y, kB.w, kB.z}; 

    //Note(Lev) ks matching the opencl shader 

    value.x += temp.x;
    value.y += temp.y;
    value.z += temp.z;
    value.w += temp.w;

    float At = value.x;
    float Bt = value.y;
    float Dt = D;
    float Et = value.z;
    float Ft = value.w;

    float4 kDtemp = {kD.y * Dt,kD.z *Dt,0.0,0.0};

    //Note(Lev) kDtemp matching the opencl shader 

    PATTERN.y += kDtemp.x;
    PATTERN.z += kDtemp.y;
    PATTERN.w += kDtemp.y;


    float4 kEtemp = {kE.x * Et,
                     kE.y * Et,
                     kE.w * Et,
                     0.0};

    PATTERN.x += kA.x* At + kEtemp.x; 
    PATTERN.y += kA.y* At + kEtemp.y;
    PATTERN.z += kA.z* At + kEtemp.x;
    PATTERN.w += kA.w* At + kEtemp.z;

    PATTERN.x += kB.x * Bt + kF.x * Ft ;
    PATTERN.z += kF.z * Ft;
    PATTERN.w += kB.w * Bt;

    float4 pixelColor = (alternate.y == 0.0f) ?
        ((alternate.x == 0.0f) ?
         make_float4(C, PATTERN.x, PATTERN.y, 1.0f) :
         make_float4(PATTERN.z, C, PATTERN.w, 1.0f)) :
        ((alternate.x == 0.0f) ?
         make_float4(PATTERN.w, C, PATTERN.z, 1.0f) :
         make_float4(PATTERN.y, PATTERN.x, C, 1.0f));

    // output needed in BGR sequence; ignore alpha
    int ind = x + y * width;
    img[ind].x = (guchar) clampc(pixelColor.x); // R
    img[ind].y = (guchar) clampc(pixelColor.y); // G
    img[ind].z = (guchar) clampc(pixelColor.z); // B
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
    cudaError_t err = {} ;

    gchar * data;
    gsize length;

    gsize width = 10000;
    gsize pitchWidth = 0;
    gsize height = 7096; 

    if(g_file_get_contents(argv[1],&data,&length,&gerr) && (width * height) == length)
    {
        cudaChannelFormatDesc pixel_channel  = cudaCreateChannelDesc<pixel>();
        cudaChannelFormatDesc pixel4_channel = cudaCreateChannelDesc<upixel4>();
        cudaChannelFormatDesc float4_channel = cudaCreateChannelDesc<float4>();
        cudaChannelFormatDesc float_channel  = cudaCreateChannelDesc<float>();

        gsize pWidth  = width * sizeof(pixel);
        gsize p3Width = width * sizeof(pixel3);
        gsize p4Width = width * sizeof(upixel4);
        gsize fWidth  = width * sizeof(float);
        gsize f4Width = width * sizeof(float4);

        dim3 threadblock(32,8);
        dim3 gridBlock((width  + threadblock.x - 1)/threadblock.x, (height + threadblock.y - 1)/threadblock.y);

        void * src_bayer = NULL;
        CudaMallocP(&src_bayer,&pitchWidth,pWidth,height);

        CudaMemcpy2D(src_bayer,pitchWidth,data,width,width,height,cudaMemcpyHostToDevice);

        cudaTextureObject_t src_image = SetupTextureAndData(src_bayer,pitchWidth,height,pixel_channel);

        upixel4 *d_res = NULL;

        CudaMallocP(&d_res,&pitchWidth,p4Width,height);

        for (int i = 0; i < 10; i++)
        {
            MHCdemosaic<<< gridBlock, threadblock >>>(src_image , d_res,width,height);
        }

        CudaSync();
        upixel4 * rescpu = (upixel4*)g_malloc0(p4Width* height);

        CudaMemcpy2D(rescpu,p4Width ,d_res, p4Width , p4Width, height, cudaMemcpyDeviceToHost);
        g_file_set_contents("hqlinear_res.ppm",(gchar*)rescpu,p4Width * height,&gerr);
    }

}
