#include <glib.h>
#include "bayer.h"
#include <math.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "ext/stb_image_write.h"
#define AHD
#define VNG
#define HQLinear

#define CLIP(in, out)\
   in = in < 0 ? 0 : in;\
   in = in > 255 ? 255 : in;\
   out=in;

#define FORC3 for (c=0; c < 3; c++)

#define SQR(x) ((x)*(x))
//#define ABS(x) (((int)(x) ^ ((int)(x) >> 31)) - ((int)(x) >> 31))
#ifndef MIN
  #define MIN(a,b) ((a) < (b) ? (a) : (b))
#endif
#ifndef MAX
  #define MAX(a,b) ((a) > (b) ? (a) : (b))
#endif
#define LIM(x,min,max) MAX(min,MIN(x,max))

#define ULIM(x,y,z) (y) < (z) ? LIM(x,y,z) : LIM(x,z,y)

void
ClearBorders(uint8_t *rgb, int sx, int sy, int w)
{
    int i, j;
    // black edges are added with a width w:
    i = 3 * sx * w - 1;
    j = 3 * sx * sy - 1;
    while (i >= 0) {
        rgb[i--] = 0;
        rgb[j--] = 0;
    }

    int low = sx * (w - 1) * 3 - 1 + w * 3;
    i = low + sx * (sy - w * 2 + 1) * 3;
    while (i > low) {
        j = 6 * w;
        while (j > 0) {
            rgb[i--] = 0;
            j--;
        }
        i -= (sx - 2 * w) * 3;
    }
}

gboolean dc1394_bayer_NearestNeighbor(const guint8 * bayer, guint8 * rgb, int sx, int sy, int tile)
{
    const int bayerStep = sx;
    const int rgbStep = 3 * sx;
    int width = sx;
    int height = sy;
    int blue = tile == DC1394_COLOR_FILTER_BGGR
        || tile == DC1394_COLOR_FILTER_GBRG ? -1 : 1;
    int start_with_green = tile == DC1394_COLOR_FILTER_GBRG
        || tile == DC1394_COLOR_FILTER_GRBG;
    int i, imax, iinc;

    if ((tile>DC1394_COLOR_FILTER_MAX)||(tile<DC1394_COLOR_FILTER_MIN))
      return 0;

    /* add black border */
    imax = sx * sy * 3;
    for (i = sx * (sy - 1) * 3; i < imax; i++) {
        rgb[i] = 0;
    }
    iinc = (sx - 1) * 3;
    for (i = (sx - 1) * 3; i < imax; i += iinc) {
        rgb[i++] = 0;
        rgb[i++] = 0;
        rgb[i++] = 0;
    }

    rgb += 1;
    width -= 1;
    height -= 1;

    for (; height--; bayer += bayerStep, rgb += rgbStep) 
    {
      //int t0, t1;
        const uint8_t *bayerEnd = bayer + width;

        if (start_with_green) {
            rgb[-blue] = bayer[1];
            rgb[0] = bayer[bayerStep + 1];
            rgb[blue] = bayer[bayerStep];
            bayer++;
            rgb += 3;
        }

        if (blue > 0) {
            for (; bayer <= bayerEnd - 2; bayer += 2, rgb += 6) {
                rgb[-1] = bayer[0];
                rgb[0] = bayer[1];
                rgb[1] = bayer[bayerStep + 1];

                rgb[2] = bayer[2];
                rgb[3] = bayer[bayerStep + 2];
                rgb[4] = bayer[bayerStep + 1];
            }
        } else {
            for (; bayer <= bayerEnd - 2; bayer += 2, rgb += 6) {
                rgb[1] = bayer[0];
                rgb[0] = bayer[1];
                rgb[-1] = bayer[bayerStep + 1];

                rgb[4] = bayer[2];
                rgb[3] = bayer[bayerStep + 2];
                rgb[2] = bayer[bayerStep + 1];
            }
        }

        if (bayer < bayerEnd) {
            rgb[-blue] = bayer[0];
            rgb[0] = bayer[1];
            rgb[blue] = bayer[bayerStep + 1];
            bayer++;
            rgb += 3;
        }

        bayer -= width;
        rgb -= width * 3;

        blue = -blue;
        start_with_green = !start_with_green;
    }

    return 1;
}

gboolean dc1394_bayer_Simple(const uint8_t * bayer, uint8_t * rgb, int sx, int sy, int tile)
{
    const int bayerStep = sx;
    const int rgbStep = 3 * sx;
    int width = sx;
    int height = sy;
    int blue = tile == DC1394_COLOR_FILTER_BGGR
        || tile == DC1394_COLOR_FILTER_GBRG ? -1 : 1;
    int start_with_green = tile == DC1394_COLOR_FILTER_GBRG
        || tile == DC1394_COLOR_FILTER_GRBG;
    int i, imax, iinc;

    if ((tile>DC1394_COLOR_FILTER_MAX)||(tile<DC1394_COLOR_FILTER_MIN))
      return 0;

    /* add black border */
    imax = sx * sy * 3;
    for (i = sx * (sy - 1) * 3; i < imax; i++) {
        rgb[i] = 0;
    }
    iinc = (sx - 1) * 3;
    for (i = (sx - 1) * 3; i < imax; i += iinc) {
        rgb[i++] = 0;
        rgb[i++] = 0;
        rgb[i++] = 0;
    }

    rgb += 1;
    width -= 1;
    height -= 1;

    for (; height--; bayer += bayerStep, rgb += rgbStep) {
        const uint8_t *bayerEnd = bayer + width;

        if (start_with_green) {
            rgb[-blue] = bayer[1];
            rgb[0] = (bayer[0] + bayer[bayerStep + 1] + 1) >> 1;
            rgb[blue] = bayer[bayerStep];
            bayer++;
            rgb += 3;
        }

        if (blue > 0) {
            for (; bayer <= bayerEnd - 2; bayer += 2, rgb += 6) {
                rgb[-1] = bayer[0];
                rgb[0] = (bayer[1] + bayer[bayerStep] + 1) >> 1;
                rgb[1] = bayer[bayerStep + 1];

                rgb[2] = bayer[2];
                rgb[3] = (bayer[1] + bayer[bayerStep + 2] + 1) >> 1;
                rgb[4] = bayer[bayerStep + 1];
            }
        } else {
            for (; bayer <= bayerEnd - 2; bayer += 2, rgb += 6) {
                rgb[1] = bayer[0];
                rgb[0] = (bayer[1] + bayer[bayerStep] + 1) >> 1;
                rgb[-1] = bayer[bayerStep + 1];

                rgb[4] = bayer[2];
                rgb[3] = (bayer[1] + bayer[bayerStep + 2] + 1) >> 1;
                rgb[2] = bayer[bayerStep + 1];
            }
        }

        if (bayer < bayerEnd) {
            rgb[-blue] = bayer[0];
            rgb[0] = (bayer[1] + bayer[bayerStep] + 1) >> 1;
            rgb[blue] = bayer[bayerStep + 1];
            bayer++;
            rgb += 3;
        }

        bayer -= width;
        rgb -= width * 3;

        blue = -blue;
        start_with_green = !start_with_green;
    }

    return 1;
}
gboolean dc1394_bayer_EdgeSense(const uint8_t * bayer, uint8_t * rgb, int sx, int sy, int tile)
{
    uint8_t *outR, *outG, *outB;
    int i3, j3, base;
    int i, j;
    int dh, dv;
    int tmp;
	int sx3=sx*3;

    // sx and sy should be even
    switch (tile) {
    case DC1394_COLOR_FILTER_GRBG:
    case DC1394_COLOR_FILTER_BGGR:
        outR = &rgb[0];
        outG = &rgb[1];
        outB = &rgb[2];
        break;
    case DC1394_COLOR_FILTER_GBRG:
    case DC1394_COLOR_FILTER_RGGB:
        outR = &rgb[2];
        outG = &rgb[1];
        outB = &rgb[0];
        break;
    default:
		return 0;
    }

    switch (tile) {
    case DC1394_COLOR_FILTER_GRBG:        //---------------------------------------------------------
    case DC1394_COLOR_FILTER_GBRG:
        // copy original RGB data to output images
		for (i = 0, i3=0; i < sy*sx; i += (sx<<1), i3 += (sx3<<1)) {
			for (j = 0, j3=0; j < sx; j += 2, j3+=6) {
				base=i3+j3;
				outG[base]           = bayer[i + j];
				outG[base + sx3 + 3] = bayer[i + j + sx + 1];
				outR[base + 3]       = bayer[i + j + 1];
				outB[base + sx3]     = bayer[i + j + sx];
			}
		}
		// process GREEN channel
		for (i3= 3*sx3; i3 < (sy - 2)*sx3; i3 += (sx3<<1)) {
			for (j3=6; j3 < sx3 - 9; j3+=6) {
				base=i3+j3;
				dh = abs(((outB[base - 6] +
						   outB[base + 6]) >> 1) -
						   outB[base]);
				dv = abs(((outB[base - (sx3<<1)] +
						   outB[base + (sx3<<1)]) >> 1) -
						   outB[base]);
				tmp = (((outG[base - 3]   + outG[base + 3]) >> 1) * (dh<=dv) +
					   ((outG[base - sx3] + outG[base + sx3]) >> 1) * (dh>dv));
				//tmp = (dh==dv) ? tmp>>1 : tmp;
				CLIP(tmp, outG[base]);
			}
		}
		
		for (i3=2*sx3; i3 < (sy - 3)*sx3; i3 += (sx3<<1)) {
			for (j3=9; j3 < sx3 - 6; j3+=6) {
				base=i3+j3;
				dh = abs(((outR[base - 6] +
						   outR[base + 6]) >>1 ) -
						   outR[base]);
				dv = abs(((outR[base - (sx3<<1)] +
						   outR[base + (sx3<<1)]) >>1 ) -
						   outR[base]);
				tmp = (((outG[base - 3]   + outG[base + 3]) >> 1) * (dh<=dv) +
					   ((outG[base - sx3] + outG[base + sx3]) >> 1) * (dh>dv));
				//tmp = (dh==dv) ? tmp>>1 : tmp;
				CLIP(tmp, outG[base]);
			}
		}
		// process RED channel
		for (i3=0; i3 < (sy - 1)*sx3; i3 += (sx3<<1)) {
			for (j3=6; j3 < sx3 - 3; j3+=6) {
				base=i3+j3;
				tmp = outG[base] +
					((outR[base - 3] -
					  outG[base - 3] +
					  outR[base + 3] -
					  outG[base + 3]) >> 1);
				CLIP(tmp, outR[base]);
			}
		}
		for (i3=sx3; i3 < (sy - 2)*sx3; i3 += (sx3<<1)) {
			for (j3=3; j3 < sx3; j3+=6) {
				base=i3+j3;
				tmp = outG[base] +
					((outR[base - sx3] -
					  outG[base - sx3] +
					  outR[base + sx3] -
					  outG[base + sx3]) >> 1);
				CLIP(tmp, outR[base]);
			}
			for (j3=6; j3 < sx3 - 3; j3+=6) {
				base=i3+j3;
				tmp = outG[base] +
					((outR[base - sx3 - 3] -
					  outG[base - sx3 - 3] +
					  outR[base - sx3 + 3] -
					  outG[base - sx3 + 3] +
					  outR[base + sx3 - 3] -
					  outG[base + sx3 - 3] +
					  outR[base + sx3 + 3] -
					  outG[base + sx3 + 3]) >> 2);
				CLIP(tmp, outR[base]);
			}
		}

		// process BLUE channel
		for (i3=sx3; i3 < sy*sx3; i3 += (sx3<<1)) {
			for (j3=3; j3 < sx3 - 6; j3+=6) {
				base=i3+j3;
				tmp = outG[base] +
					((outB[base - 3] -
					  outG[base - 3] +
					  outB[base + 3] -
					  outG[base + 3]) >> 1);
				CLIP(tmp, outB[base]);
			}
		}
		for (i3=2*sx3; i3 < (sy - 1)*sx3; i3 += (sx3<<1)) {
			for (j3=0; j3 < sx3 - 3; j3+=6) {
				base=i3+j3;
				tmp = outG[base] +
					((outB[base - sx3] -
					  outG[base - sx3] +
					  outB[base + sx3] -
					  outG[base + sx3]) >> 1);
				CLIP(tmp, outB[base]);
			}
			for (j3=3; j3 < sx3 - 6; j3+=6) {
				base=i3+j3;
				tmp = outG[base] +
					((outB[base - sx3 - 3] -
					  outG[base - sx3 - 3] +
					  outB[base - sx3 + 3] -
					  outG[base - sx3 + 3] +
					  outB[base + sx3 - 3] -
					  outG[base + sx3 - 3] +
					  outB[base + sx3 + 3] -
					  outG[base + sx3 + 3]) >> 2);
				CLIP(tmp, outB[base]);
			}
		}
		break;

    case DC1394_COLOR_FILTER_BGGR:        //---------------------------------------------------------
    case DC1394_COLOR_FILTER_RGGB:
        // copy original RGB data to output images
		for (i = 0, i3=0; i < sy*sx; i += (sx<<1), i3 += (sx3<<1)) {
			for (j = 0, j3=0; j < sx; j += 2, j3+=6) {
				base=i3+j3;
				outB[base] = bayer[i + j];
				outR[base + sx3 + 3] = bayer[i + sx + (j + 1)];
				outG[base + 3] = bayer[i + j + 1];
				outG[base + sx3] = bayer[i + sx + j];
			}
		}
		// process GREEN channel
		for (i3=2*sx3; i3 < (sy - 2)*sx3; i3 += (sx3<<1)) {
			for (j3=6; j3 < sx3 - 9; j3+=6) {
				base=i3+j3;
				dh = abs(((outB[base - 6] +
						   outB[base + 6]) >> 1) -
						   outB[base]);
				dv = abs(((outB[base - (sx3<<1)] +
						   outB[base + (sx3<<1)]) >> 1) -
						   outB[base]);
				tmp = (((outG[base - 3]   + outG[base + 3]) >> 1) * (dh<=dv) +
					   ((outG[base - sx3] + outG[base + sx3]) >> 1) * (dh>dv));
				//tmp = (dh==dv) ? tmp>>1 : tmp;
				CLIP(tmp, outG[base]);
			}
		}
		for (i3=3*sx3; i3 < (sy - 3)*sx3; i3 += (sx3<<1)) {
			for (j3=9; j3 < sx3 - 6; j3+=6) {
				base=i3+j3;
				dh = abs(((outR[base - 6] +
						   outR[base + 6]) >> 1) -
						   outR[base]);
				dv = abs(((outR[base - (sx3<<1)] +
						   outR[base + (sx3<<1)]) >> 1) -
						   outR[base]);
				tmp = (((outG[base - 3]   + outG[base + 3]) >> 1) * (dh<=dv) +
					   ((outG[base - sx3] + outG[base + sx3]) >> 1) * (dh>dv));
				//tmp = (dh==dv) ? tmp>>1 : tmp;
				CLIP(tmp, outG[base]);
			}
		}
		// process RED channel
		for (i3=sx3; i3 < (sy - 1)*sx3; i3 += (sx3<<1)) {        // G-points (1/2)
			for (j3=6; j3 < sx3 - 3; j3+=6) {
				base=i3+j3;
				tmp = outG[base] +
					((outR[base - 3] -
					  outG[base - 3] +
					  outR[base + 3] -
					  outG[base + 3]) >>1);
				CLIP(tmp, outR[base]);
			}
		}
		for (i3=2*sx3; i3 < (sy - 2)*sx3; i3 += (sx3<<1)) {
			for (j3=3; j3 < sx3; j3+=6) {        // G-points (2/2)
				base=i3+j3;
				tmp = outG[base] +
					((outR[base - sx3] -
					  outG[base - sx3] +
					  outR[base + sx3] -
					  outG[base + sx3]) >> 1);
				CLIP(tmp, outR[base]);
			}
			for (j3=6; j3 < sx3 - 3; j3+=6) {        // B-points
				base=i3+j3;
				tmp = outG[base] +
					((outR[base - sx3 - 3] -
					  outG[base - sx3 - 3] +
					  outR[base - sx3 + 3] -
					  outG[base - sx3 + 3] +
					  outR[base + sx3 - 3] -
					  outG[base + sx3 - 3] +
					  outR[base + sx3 + 3] -
					  outG[base + sx3 + 3]) >> 2);
				CLIP(tmp, outR[base]);
			}
		}

		// process BLUE channel
		for (i = 0,i3=0; i < sy*sx; i += (sx<<1), i3 += (sx3<<1)) {
			for (j = 1, j3=3; j < sx - 2; j += 2, j3+=6) {
				base=i3+j3;
				tmp = outG[base] +
					((outB[base - 3] -
					  outG[base - 3] +
					  outB[base + 3] -
					  outG[base + 3]) >> 1);
				CLIP(tmp, outB[base]);
			}
		}
		for (i3=sx3; i3 < (sy - 1)*sx3; i3 += (sx3<<1)) {
			for (j3=0; j3 < sx3 - 3; j3+=6) {
				base=i3+j3;
				tmp = outG[base] +
					((outB[base - sx3] -
					  outG[base - sx3] +
					  outB[base + sx3] -
					  outG[base + sx3]) >> 1);
				CLIP(tmp, outB[base]);
			}
			for (j3=3; j3 < sx3 - 6; j3+=6) {
				base=i3+j3;
				tmp = outG[base] +
					((outB[base - sx3 - 3] -
					  outG[base - sx3 - 3] +
					  outB[base - sx3 + 3] -
					  outG[base - sx3 + 3] +
					  outB[base + sx3 - 3] -
					  outG[base + sx3 - 3] +
					  outB[base + sx3 + 3] -
					  outG[base + sx3 + 3]) >> 2);
				CLIP(tmp, outB[base]);
			}
		}
		break;
    }

    ClearBorders(rgb, sx, sy, 3);

    return 1; 
}

#define CLIPOUT(x)        LIM(x,0,255)
#define CLIPOUT16(x,bits) LIM(x,0,((1<<bits)-1))

static const double xyz_rgb[3][3] = {                        /* XYZ from RGB */
  { 0.412453, 0.357580, 0.180423 },
  { 0.212671, 0.715160, 0.072169 },
  { 0.019334, 0.119193, 0.950227 } };
static const float d65_white[3] = { 0.950456f, 1.f, 1.088754f };


static void cam_to_cielab (uint16_t cam[3], float lab[3]) /* [SA] */
{
    int c, i, j;
    float r, xyz[3];
    static float cbrt[0x10000], xyz_cam[3][4];

    if (cam == NULL) {
        for (i=0; i < 0x10000; i++) {
            r = i / 65535.0;
            cbrt[i] = r > 0.008856 ? pow(r,1/3.0) : 7.787*r + 16/116.0;
        }
        for (i=0; i < 3; i++)
            for (j=0; j < 3; j++)                           /* [SA] */
                xyz_cam[i][j] = xyz_rgb[i][j] / d65_white[i]; /* [SA] */
    } else {
        xyz[0] = xyz[1] = xyz[2] = 0.5;
        FORC3 { /* [SA] */
            xyz[0] += xyz_cam[0][c] * cam[c];
            xyz[1] += xyz_cam[1][c] * cam[c];
            xyz[2] += xyz_cam[2][c] * cam[c];
        }
        xyz[0] = cbrt[CLIPOUT16((int) xyz[0],16)];        /* [SA] */
        xyz[1] = cbrt[CLIPOUT16((int) xyz[1],16)];        /* [SA] */
        xyz[2] = cbrt[CLIPOUT16((int) xyz[2],16)];        /* [SA] */
        lab[0] = 116 * xyz[1] - 16;
        lab[1] = 500 * (xyz[0] - xyz[1]);
        lab[2] = 200 * (xyz[1] - xyz[2]);
    }
}
static dc1394bool_t ahd_inited = DC1394_FALSE; /* WARNING: not multi-processor safe */

#define TS 256                /* Tile Size */

#define FC(row,col) \
        (filters >> ((((row) << 1 & 14) + ((col) & 1)) << 1) & 3)


gboolean dc1394_bayer_AHD(const uint8_t *bayer, uint8_t *dst, int sx, int sy, dc1394color_filter_t pattern)
{
    int i, j, top, left, row, col, tr, tc, fc, c, d, val, hm[2];
    /* the following has the same type as the image */
    uint8_t (*pix)[3], (*rix)[3];      /* [SA] */
    uint16_t rix16[3];                 /* [SA] */
    static const int dir[4] = { -1, 1, -TS, TS };
    unsigned ldiff[2][4], abdiff[2][4], leps, abeps;
    float flab[3];                     /* [SA] */
    uint8_t (*rgb)[TS][TS][3];
    short (*lab)[TS][TS][3];
    char (*homo)[TS][TS], *buffer;

    /* start - new code for libdc1394 */
    uint32_t filters;
    const int height = sy, width = sx;
    int x, y;

    if (ahd_inited==DC1394_FALSE) {
        /* WARNING: this might not be multi-processor safe */
        cam_to_cielab (NULL,NULL);
        ahd_inited = DC1394_TRUE;
    }

    switch(pattern) {
    case DC1394_COLOR_FILTER_BGGR:
        filters = 0x16161616;
        break;
    case DC1394_COLOR_FILTER_GRBG:
        filters = 0x61616161;
        break;
    case DC1394_COLOR_FILTER_RGGB:
        filters = 0x94949494;
        break;
    case DC1394_COLOR_FILTER_GBRG:
        filters = 0x49494949;
        break;
    default:
        return 0;
    }

    /* fill-in destination with known exact values */
    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            int channel = FC(y,x);
            dst[(y*width+x)*3 + channel] = bayer[y*width+x];
        }
    }
    /* end - new code for libdc1394 */

    /* start - code from border_interpolate (int border) */
    {
        int border = 3;
        unsigned row, col, y, x, f, c, sum[8];

        for (row=0; row < height; row++)
            for (col=0; col < width; col++) {
                if (col==border && row >= border && row < height-border)
                    col = width-border;
                memset (sum, 0, sizeof sum);
                for (y=row-1; y != row+2; y++)
                    for (x=col-1; x != col+2; x++)
                        if (y < height && x < width) {
                            f = FC(y,x);
                            sum[f] += dst[(y*width+x)*3 + f];           /* [SA] */
                            sum[f+4]++;
                        }
                f = FC(row,col);
                FORC3 if (c != f && sum[c+4])                     /* [SA] */
                    dst[(row*width+col)*3 + c] = sum[c] / sum[c+4]; /* [SA] */
            }
    }
    /* end - code from border_interpolate (int border) */


    buffer = (char *) malloc (26*TS*TS);                /* 1664 kB */
    /* merror (buffer, "ahd_interpolate()"); */
    rgb  = (uint8_t(*)[TS][TS][3]) buffer;                /* [SA] */
    lab  = (short (*)[TS][TS][3])(buffer + 12*TS*TS);
    homo = (char  (*)[TS][TS])   (buffer + 24*TS*TS);

    for (top=0; top < height; top += TS-6)
        for (left=0; left < width; left += TS-6) {
            memset (rgb, 0, 12*TS*TS);

            /*  Interpolate green horizontally and vertically:                */
            for (row = top < 2 ? 2:top; row < top+TS && row < height-2; row++) {
                col = left + (FC(row,left) == 1);
                if (col < 2) col += 2;
                for (fc = FC(row,col); col < left+TS && col < width-2; col+=2) {

                    pix = (uint8_t (*)[3])dst + (row*width+col);          /* [SA] */
                    val = ((pix[-1][1] + pix[0][fc] + pix[1][1]) * 2 - pix[-2][fc] - pix[2][fc]) >> 2;

                    rgb[0][row-top][col-left][1] = ULIM(val,pix[-1][1],pix[1][1]);
                    val = ((pix[-width][1] + pix[0][fc] + pix[width][1]) * 2
                           - pix[-2*width][fc] - pix[2*width][fc]) >> 2;
                    rgb[1][row-top][col-left][1] = ULIM(val,pix[-width][1],pix[width][1]);
                }
            }
            /*  Interpolate red and blue, and convert to CIELab:                */
            for (d=0; d < 2; d++)
                for (row=top+1; row < top+TS-1 && row < height-1; row++)
                    for (col=left+1; col < left+TS-1 && col < width-1; col++) {
                        pix = (uint8_t (*)[3])dst + (row*width+col);        /* [SA] */
                        rix = &rgb[d][row-top][col-left];
                        if ((c = 2 - FC(row,col)) == 1) {
                            c = FC(row+1,col);
                            val = pix[0][1] + (( pix[-1][2-c] + pix[1][2-c]
                                                 - rix[-1][1] - rix[1][1] ) >> 1);
                            rix[0][2-c] = CLIPOUT(val);         /* [SA] */
                            val = pix[0][1] + (( pix[-width][c] + pix[width][c]
                                                 - rix[-TS][1] - rix[TS][1] ) >> 1);
                        } else
                            val = rix[0][1] + (( pix[-width-1][c] + pix[-width+1][c]
                                                 + pix[+width-1][c] + pix[+width+1][c]
                                                 - rix[-TS-1][1] - rix[-TS+1][1]
                                                 - rix[+TS-1][1] - rix[+TS+1][1] + 1) >> 2);
                        rix[0][c] = CLIPOUT(val);             /* [SA] */
                        c = FC(row,col);
                        rix[0][c] = pix[0][c];
                        rix16[0] = rix[0][0];                 /* [SA] */
                        rix16[1] = rix[0][1];                 /* [SA] */
                        rix16[2] = rix[0][2];                 /* [SA] */
                        cam_to_cielab (rix16, flab);          /* [SA] */
                        FORC3 lab[d][row-top][col-left][c] = 64*flab[c];
                    }
            /*  Build homogeneity maps from the CIELab images:                */
            memset (homo, 0, 2*TS*TS);
            for (row=top+2; row < top+TS-2 && row < height; row++) {
                tr = row-top;
                for (col=left+2; col < left+TS-2 && col < width; col++) {
                    tc = col-left;
                    for (d=0; d < 2; d++)
                        for (i=0; i < 4; i++)
                            ldiff[d][i] = ABS(lab[d][tr][tc][0]-lab[d][tr][tc+dir[i]][0]);
                    leps = MIN(MAX(ldiff[0][0],ldiff[0][1]),
                               MAX(ldiff[1][2],ldiff[1][3]));
                    for (d=0; d < 2; d++)
                        for (i=0; i < 4; i++)
                            if (i >> 1 == d || ldiff[d][i] <= leps)
                                abdiff[d][i] = SQR(lab[d][tr][tc][1]-lab[d][tr][tc+dir[i]][1])
                                    + SQR(lab[d][tr][tc][2]-lab[d][tr][tc+dir[i]][2]);
                    abeps = MIN(MAX(abdiff[0][0],abdiff[0][1]),
                                MAX(abdiff[1][2],abdiff[1][3]));
                    for (d=0; d < 2; d++)
                        for (i=0; i < 4; i++)
                            if (ldiff[d][i] <= leps && abdiff[d][i] <= abeps)
                                homo[d][tr][tc]++;
                }
            }
            /*  Combine the most homogenous pixels for the final result:        */
            for (row=top+3; row < top+TS-3 && row < height-3; row++) {
                tr = row-top;
                for (col=left+3; col < left+TS-3 && col < width-3; col++) {
                    tc = col-left;
                    for (d=0; d < 2; d++)
                        for (hm[d]=0, i=tr-1; i <= tr+1; i++)
                            for (j=tc-1; j <= tc+1; j++)
                                hm[d] += homo[d][i][j];
                    if (hm[0] != hm[1])
                        FORC3 dst[(row*width+col)*3 + c] = CLIPOUT(rgb[hm[1] > hm[0]][tr][tc][c]); /* [SA] */
                    else
                        FORC3 dst[(row*width+col)*3 + c] =
                            CLIPOUT((rgb[0][tr][tc][c] + rgb[1][tr][tc][c]) >> 1);      /* [SA] */
                }
            }
        }
    free (buffer);

    return 1;
}
gboolean dc1394_bayer_Bilinear(const uint8_t * bayer, uint8_t * rgb, int sx, int sy, int tile)
{
    const int bayerStep = sx;
    const int rgbStep = 3 * sx;
    int width = sx;
    int height = sy;
    /*
       the two letters  of the OpenCV name are respectively
       the 4th and 3rd letters from the blinky name,
       and we also have to switch R and B (OpenCV is BGR)

       CV_BayerBG2BGR <-> DC1394_COLOR_FILTER_BGGR
       CV_BayerGB2BGR <-> DC1394_COLOR_FILTER_GBRG
       CV_BayerGR2BGR <-> DC1394_COLOR_FILTER_GRBG

       int blue = tile == CV_BayerBG2BGR || tile == CV_BayerGB2BGR ? -1 : 1;
       int start_with_green = tile == CV_BayerGB2BGR || tile == CV_BayerGR2BGR;
     */
    int blue = tile == DC1394_COLOR_FILTER_BGGR
        || tile == DC1394_COLOR_FILTER_GBRG ? -1 : 1;
    int start_with_green = tile == DC1394_COLOR_FILTER_GBRG
        || tile == DC1394_COLOR_FILTER_GRBG;

    if ((tile>DC1394_COLOR_FILTER_MAX)||(tile<DC1394_COLOR_FILTER_MIN))
        return 0;

    ClearBorders(rgb, sx, sy, 1);
    rgb += rgbStep + 3 + 1;
    height -= 2;
    width -= 2;

    for (; height--; bayer += bayerStep, rgb += rgbStep) {
        int t0, t1;
        const uint8_t *bayerEnd = bayer + width;

        if (start_with_green) {
            /* OpenCV has a bug in the next line, which was
               t0 = (bayer[0] + bayer[bayerStep * 2] + 1) >> 1; */
            t0 = (bayer[1] + bayer[bayerStep * 2 + 1] + 1) >> 1;
            t1 = (bayer[bayerStep] + bayer[bayerStep + 2] + 1) >> 1;
            rgb[-blue] = (uint8_t) t0;
            rgb[0] = bayer[bayerStep + 1];
            rgb[blue] = (uint8_t) t1;
            bayer++;
            rgb += 3;
        }

        if (blue > 0) {
            for (; bayer <= bayerEnd - 2; bayer += 2, rgb += 6) {
                t0 = (bayer[0] + bayer[2] + bayer[bayerStep * 2] +
                      bayer[bayerStep * 2 + 2] + 2) >> 2;
                t1 = (bayer[1] + bayer[bayerStep] +
                      bayer[bayerStep + 2] + bayer[bayerStep * 2 + 1] +
                      2) >> 2;
                rgb[-1] = (uint8_t) t0;
                rgb[0] = (uint8_t) t1;
                rgb[1] = bayer[bayerStep + 1];

                t0 = (bayer[2] + bayer[bayerStep * 2 + 2] + 1) >> 1;
                t1 = (bayer[bayerStep + 1] + bayer[bayerStep + 3] +
                      1) >> 1;
                rgb[2] = (uint8_t) t0;
                rgb[3] = bayer[bayerStep + 2];
                rgb[4] = (uint8_t) t1;
            }
        } else {
            for (; bayer <= bayerEnd - 2; bayer += 2, rgb += 6) {
                t0 = (bayer[0] + bayer[2] + bayer[bayerStep * 2] +
                      bayer[bayerStep * 2 + 2] + 2) >> 2;
                t1 = (bayer[1] + bayer[bayerStep] +
                      bayer[bayerStep + 2] + bayer[bayerStep * 2 + 1] +
                      2) >> 2;
                rgb[1] = (uint8_t) t0;
                rgb[0] = (uint8_t) t1;
                rgb[-1] = bayer[bayerStep + 1];

                t0 = (bayer[2] + bayer[bayerStep * 2 + 2] + 1) >> 1;
                t1 = (bayer[bayerStep + 1] + bayer[bayerStep + 3] +
                      1) >> 1;
                rgb[4] = (uint8_t) t0;
                rgb[3] = bayer[bayerStep + 2];
                rgb[2] = (uint8_t) t1;
            }
        }

        if (bayer < bayerEnd) {
            t0 = (bayer[0] + bayer[2] + bayer[bayerStep * 2] +
                  bayer[bayerStep * 2 + 2] + 2) >> 2;
            t1 = (bayer[1] + bayer[bayerStep] +
                  bayer[bayerStep + 2] + bayer[bayerStep * 2 + 1] +
                  2) >> 2;
            rgb[-blue] = (uint8_t) t0;
            rgb[0] = (uint8_t) t1;
            rgb[blue] = bayer[bayerStep + 1];
            bayer++;
            rgb += 3;
        }

        bayer -= width;
        rgb -= width * 3;

        blue = -blue;
        start_with_green = !start_with_green;
    }
    return 1;
}

static const int8_t bayervng_terms[] = {
    -2,-2,+0,-1,0,(int8_t)0x01, -2,-2,+0,+0,1,(int8_t)0x01, -2,-1,-1,+0,0,(int8_t)0x01,
    -2,-1,+0,-1,0,(int8_t)0x02, -2,-1,+0,+0,0,(int8_t)0x03, -2,-1,+0,+1,1,(int8_t)0x01,
    -2,+0,+0,-1,0,(int8_t)0x06, -2,+0,+0,+0,1,(int8_t)0x02, -2,+0,+0,+1,0,(int8_t)0x03,
    -2,+1,-1,+0,0,(int8_t)0x04, -2,+1,+0,-1,1,(int8_t)0x04, -2,+1,+0,+0,0,(int8_t)0x06,
    -2,+1,+0,+1,0,(int8_t)0x02, -2,+2,+0,+0,1,(int8_t)0x04, -2,+2,+0,+1,0,(int8_t)0x04,
    -1,-2,-1,+0,0,(int8_t)0x80, -1,-2,+0,-1,0,(int8_t)0x01, -1,-2,+1,-1,0,(int8_t)0x01,
    -1,-2,+1,+0,1,(int8_t)0x01, -1,-1,-1,+1,0,(int8_t)0x88, -1,-1,+1,-2,0,(int8_t)0x40,
    -1,-1,+1,-1,0,(int8_t)0x22, -1,-1,+1,+0,0,(int8_t)0x33, -1,-1,+1,+1,1,(int8_t)0x11,
    -1,+0,-1,+2,0,(int8_t)0x08, -1,+0,+0,-1,0,(int8_t)0x44, -1,+0,+0,+1,0,(int8_t)0x11,
    -1,+0,+1,-2,1,(int8_t)0x40, -1,+0,+1,-1,0,(int8_t)0x66, -1,+0,+1,+0,1,(int8_t)0x22,
    -1,+0,+1,+1,0,(int8_t)0x33, -1,+0,+1,+2,1,(int8_t)0x10, -1,+1,+1,-1,1,(int8_t)0x44,
    -1,+1,+1,+0,0,(int8_t)0x66, -1,+1,+1,+1,0,(int8_t)0x22, -1,+1,+1,+2,0,(int8_t)0x10,
    -1,+2,+0,+1,0,(int8_t)0x04, -1,+2,+1,+0,1,(int8_t)0x04, -1,+2,+1,+1,0,(int8_t)0x04,
    +0,-2,+0,+0,1,(int8_t)0x80, +0,-1,+0,+1,1,(int8_t)0x88, +0,-1,+1,-2,0,(int8_t)0x40,
    +0,-1,+1,+0,0,(int8_t)0x11, +0,-1,+2,-2,0,(int8_t)0x40, +0,-1,+2,-1,0,(int8_t)0x20,
    +0,-1,+2,+0,0,(int8_t)0x30, +0,-1,+2,+1,1,(int8_t)0x10, +0,+0,+0,+2,1,(int8_t)0x08,
    +0,+0,+2,-2,1,(int8_t)0x40, +0,+0,+2,-1,0,(int8_t)0x60, +0,+0,+2,+0,1,(int8_t)0x20,
    +0,+0,+2,+1,0,(int8_t)0x30, +0,+0,+2,+2,1,(int8_t)0x10, +0,+1,+1,+0,0,(int8_t)0x44,
    +0,+1,+1,+2,0,(int8_t)0x10, +0,+1,+2,-1,1,(int8_t)0x40, +0,+1,+2,+0,0,(int8_t)0x60,
    +0,+1,+2,+1,0,(int8_t)0x20, +0,+1,+2,+2,0,(int8_t)0x10, +1,-2,+1,+0,0,(int8_t)0x80,
    +1,-1,+1,+1,0,(int8_t)0x88, +1,+0,+1,+2,0,(int8_t)0x08, +1,+0,+2,-1,0,(int8_t)0x40,
    +1,+0,+2,+1,0,(int8_t)0x10
}, bayervng_chood[] = { -1,-1, -1,0, -1,+1, 0,+1, +1,+1, +1,0, +1,-1, 0,-1 };

gboolean dc1394_bayer_VNG(const uint8_t * bayer, uint8_t * dst, int sx, int sy, dc1394color_filter_t pattern)
{
    const int height = sy, width = sx;
    static const signed char *cp;
    /* the following has the same type as the image */
    uint8_t (*brow[5])[3], *pix;          /* [FD] */
    int code[8][2][320], *ip, gval[8], gmin, gmax, sum[4];
    int row, col, x, y, x1, x2, y1, y2, t, weight, grads, color, diag;
    int g, diff, thold, num, c;
    uint32_t filters;                     /* [FD] */

    /* first, use bilinear bayer decoding */
    dc1394_bayer_Bilinear(bayer, dst, sx, sy, pattern);

    switch(pattern) {
    case DC1394_COLOR_FILTER_BGGR:
        filters = 0x16161616;
        break;
    case DC1394_COLOR_FILTER_GRBG:
        filters = 0x61616161;
        break;
    case DC1394_COLOR_FILTER_RGGB:
        filters = 0x94949494;
        break;
    case DC1394_COLOR_FILTER_GBRG:
        filters = 0x49494949;
        break;
    default:
        return 0;
    }

    for (row=0; row < 8; row++) {                /* Precalculate for VNG */
        for (col=0; col < 2; col++) {
            ip = code[row][col];
            for (cp=bayervng_terms, t=0; t < 64; t++) {
                y1 = *cp++;  x1 = *cp++;
                y2 = *cp++;  x2 = *cp++;
                weight = *cp++;
                grads = *cp++;
                color = FC(row+y1,col+x1);
                if (FC(row+y2,col+x2) != color) continue;
                diag = (FC(row,col+1) == color && FC(row+1,col) == color) ? 2:1;
                if (abs(y1-y2) == diag && abs(x1-x2) == diag) continue;
                *ip++ = (y1*width + x1)*3 + color; /* [FD] */
                *ip++ = (y2*width + x2)*3 + color; /* [FD] */
                *ip++ = weight;
                for (g=0; g < 8; g++)
                    if (grads & 1<<g) *ip++ = g;
                *ip++ = -1;
            }
            *ip++ = INT_MAX;
            for (cp=bayervng_chood, g=0; g < 8; g++) {
                y = *cp++;  x = *cp++;
                *ip++ = (y*width + x) * 3;      /* [FD] */
                color = FC(row,col);
                if (FC(row+y,col+x) != color && FC(row+y*2,col+x*2) == color)
                    *ip++ = (y*width + x) * 6 + color; /* [FD] */
                else
                    *ip++ = 0;
            }
        }
    }
    brow[4] = (uint8_t (*)[3]) calloc ((size_t)width*3, sizeof **brow);
    //merror (brow[4], "vng_interpolate()");
    for (row=0; row < 3; row++)
        brow[row] = brow[4] + row*width;
    for (row=2; row < height-2; row++) {                /* Do VNG interpolation */
        for (col=2; col < width-2; col++) {
            pix = dst + (row*width+col)*3;        /* [FD] */
            ip = code[row & 7][col & 1];
            memset (gval, 0, sizeof gval);
            while ((g = ip[0]) != INT_MAX) {                /* Calculate gradients */
                diff = ABS(pix[g] - pix[ip[1]]) << ip[2];
                gval[ip[3]] += diff;
                ip += 5;
                if ((g = ip[-1]) == -1) continue;
                gval[g] += diff;
                while ((g = *ip++) != -1)
                    gval[g] += diff;
            }
            ip++;
            gmin = gmax = gval[0];                        /* Choose a threshold */
            for (g=1; g < 8; g++) {
                if (gmin > gval[g]) gmin = gval[g];
                if (gmax < gval[g]) gmax = gval[g];
            }
            if (gmax == 0) {
                memcpy (brow[2][col], pix, 3 * sizeof *dst); /* [FD] */
                continue;
            }
            thold = gmin + (gmax >> 1);
            memset (sum, 0, sizeof sum);
            color = FC(row,col);
            for (num=g=0; g < 8; g++,ip+=2) {                /* Average the neighbors */
                if (gval[g] <= thold) {
                    for (c=0; c < 3; c++)         /* [FD] */
                        if (c == color && ip[1])
                            sum[c] += (pix[c] + pix[ip[1]]) >> 1;
                        else
                            sum[c] += pix[ip[0] + c];
                    num++;
                }
            }
            for (c=0; c < 3; c++) {               /* [FD] Save to buffer */
                t = pix[color];
                if (c != color)
                    t += (sum[c] - sum[color]) / num;
                CLIP(t,brow[2][col][c]);          /* [FD] */
            }
        }
        if (row > 3)                                /* Write buffer to image */
            memcpy (dst + 3*((row-2)*width+2), brow[0]+2, (width-4)*3*sizeof *dst); /* [FD] */
        for (g=0; g < 4; g++)
            brow[(g-1) & 3] = brow[g];
    }
    memcpy (dst + 3*((row-2)*width+2), brow[0]+2, (width-4)*3*sizeof *dst);
    memcpy (dst + 3*((row-1)*width+2), brow[1]+2, (width-4)*3*sizeof *dst);
    free (brow[4]);

    return 1;
}

gboolean dc1394_bayer_HQLinear(const uint8_t * bayer, uint8_t * rgb, int sx, int sy, int tile)
{
    const int bayerStep = sx;
    const int rgbStep = 3 * sx;
    int width = sx;
    int height = sy;
    int blue = tile == DC1394_COLOR_FILTER_BGGR
        || tile == DC1394_COLOR_FILTER_GBRG ? -1 : 1;
    int start_with_green = tile == DC1394_COLOR_FILTER_GBRG
        || tile == DC1394_COLOR_FILTER_GRBG;

    if ((tile>DC1394_COLOR_FILTER_MAX)||(tile<DC1394_COLOR_FILTER_MIN))
      return DC1394_INVALID_COLOR_FILTER;

    ClearBorders(rgb, sx, sy, 2);
    rgb += 2 * rgbStep + 6 + 1;
    height -= 4;
    width -= 4;

    /* We begin with a (+1 line,+1 column) offset with respect to bilinear decoding, so start_with_green is the same, but blue is opposite */
    blue = -blue;

    for (; height--; bayer += bayerStep, rgb += rgbStep) {
        int t0, t1;
        const uint8_t *bayerEnd = bayer + width;
        const int bayerStep2 = bayerStep * 2;
        const int bayerStep3 = bayerStep * 3;
        const int bayerStep4 = bayerStep * 4;

        if (start_with_green) {
            /* at green pixel */
            rgb[0] = bayer[bayerStep2 + 2];
            t0 = rgb[0] * 5
                + ((bayer[bayerStep + 2] + bayer[bayerStep3 + 2]) << 2)
                - bayer[2]
                - bayer[bayerStep + 1]
                - bayer[bayerStep + 3]
                - bayer[bayerStep3 + 1]
                - bayer[bayerStep3 + 3]
                - bayer[bayerStep4 + 2]
                + ((bayer[bayerStep2] + bayer[bayerStep2 + 4] + 1) >> 1);
            t1 = rgb[0] * 5 +
                ((bayer[bayerStep2 + 1] + bayer[bayerStep2 + 3]) << 2)
                - bayer[bayerStep2]
                - bayer[bayerStep + 1]
                - bayer[bayerStep + 3]
                - bayer[bayerStep3 + 1]
                - bayer[bayerStep3 + 3]
                - bayer[bayerStep2 + 4]
                + ((bayer[2] + bayer[bayerStep4 + 2] + 1) >> 1);
            t0 = (t0 + 4) >> 3;
            CLIP(t0, rgb[-blue]);
            t1 = (t1 + 4) >> 3;
            CLIP(t1, rgb[blue]);
            bayer++;
            rgb += 3;
        }

        if (blue > 0) {
            for (; bayer <= bayerEnd - 2; bayer += 2, rgb += 6) {
                /* B at B */
                rgb[1] = bayer[bayerStep2 + 2];
                /* R at B */
                t0 = ((bayer[bayerStep + 1] + bayer[bayerStep + 3] +
                       bayer[bayerStep3 + 1] + bayer[bayerStep3 + 3]) << 1)
                    -
                    (((bayer[2] + bayer[bayerStep2] +
                       bayer[bayerStep2 + 4] + bayer[bayerStep4 +
                                                     2]) * 3 + 1) >> 1)
                    + rgb[1] * 6;
                /* G at B */
                t1 = ((bayer[bayerStep + 2] + bayer[bayerStep2 + 1] +
                       bayer[bayerStep2 + 3] + bayer[bayerStep3 + 2]) << 1)
                    - (bayer[2] + bayer[bayerStep2] +
                       bayer[bayerStep2 + 4] + bayer[bayerStep4 + 2])
                    + (rgb[1] << 2);
                t0 = (t0 + 4) >> 3;
                CLIP(t0, rgb[-1]);
                t1 = (t1 + 4) >> 3;
                CLIP(t1, rgb[0]);
                /* at green pixel */
                rgb[3] = bayer[bayerStep2 + 3];
                t0 = rgb[3] * 5
                    + ((bayer[bayerStep + 3] + bayer[bayerStep3 + 3]) << 2)
                    - bayer[3]
                    - bayer[bayerStep + 2]
                    - bayer[bayerStep + 4]
                    - bayer[bayerStep3 + 2]
                    - bayer[bayerStep3 + 4]
                    - bayer[bayerStep4 + 3]
                    +
                    ((bayer[bayerStep2 + 1] + bayer[bayerStep2 + 5] +
                      1) >> 1);
                t1 = rgb[3] * 5 +
                    ((bayer[bayerStep2 + 2] + bayer[bayerStep2 + 4]) << 2)
                    - bayer[bayerStep2 + 1]
                    - bayer[bayerStep + 2]
                    - bayer[bayerStep + 4]
                    - bayer[bayerStep3 + 2]
                    - bayer[bayerStep3 + 4]
                    - bayer[bayerStep2 + 5]
                    + ((bayer[3] + bayer[bayerStep4 + 3] + 1) >> 1);
                t0 = (t0 + 4) >> 3;
                CLIP(t0, rgb[2]);
                t1 = (t1 + 4) >> 3;
                CLIP(t1, rgb[4]);
            }
        } else {
            for (; bayer <= bayerEnd - 2; bayer += 2, rgb += 6) {
                /* R at R */
                rgb[-1] = bayer[bayerStep2 + 2];
                /* B at R */
                t0 = ((bayer[bayerStep + 1] + bayer[bayerStep + 3] +
                       bayer[bayerStep3 + 1] + bayer[bayerStep3 + 3]) << 1)
                    -
                    (((bayer[2] + bayer[bayerStep2] +
                       bayer[bayerStep2 + 4] + bayer[bayerStep4 +
                                                     2]) * 3 + 1) >> 1)
                    + rgb[-1] * 6;
                /* G at R */
                t1 = ((bayer[bayerStep + 2] + bayer[bayerStep2 + 1] +
                       bayer[bayerStep2 + 3] + bayer[bayerStep * 3 +
                                                     2]) << 1)
                    - (bayer[2] + bayer[bayerStep2] +
                       bayer[bayerStep2 + 4] + bayer[bayerStep4 + 2])
                    + (rgb[-1] << 2);
                t0 = (t0 + 4) >> 3;
                CLIP(t0, rgb[1]);
                t1 = (t1 + 4) >> 3;
                CLIP(t1, rgb[0]);

                /* at green pixel */
                rgb[3] = bayer[bayerStep2 + 3];
                t0 = rgb[3] * 5
                    + ((bayer[bayerStep + 3] + bayer[bayerStep3 + 3]) << 2)
                    - bayer[3]
                    - bayer[bayerStep + 2]
                    - bayer[bayerStep + 4]
                    - bayer[bayerStep3 + 2]
                    - bayer[bayerStep3 + 4]
                    - bayer[bayerStep4 + 3]
                    +
                    ((bayer[bayerStep2 + 1] + bayer[bayerStep2 + 5] +
                      1) >> 1);
                t1 = rgb[3] * 5 +
                    ((bayer[bayerStep2 + 2] + bayer[bayerStep2 + 4]) << 2)
                    - bayer[bayerStep2 + 1]
                    - bayer[bayerStep + 2]
                    - bayer[bayerStep + 4]
                    - bayer[bayerStep3 + 2]
                    - bayer[bayerStep3 + 4]
                    - bayer[bayerStep2 + 5]
                    + ((bayer[3] + bayer[bayerStep4 + 3] + 1) >> 1);
                t0 = (t0 + 4) >> 3;
                CLIP(t0, rgb[4]);
                t1 = (t1 + 4) >> 3;
                CLIP(t1, rgb[2]);
            }
        }

        if (bayer < bayerEnd) {
            /* B at B */
            rgb[blue] = bayer[bayerStep2 + 2];
            /* R at B */
            t0 = ((bayer[bayerStep + 1] + bayer[bayerStep + 3] +
                   bayer[bayerStep3 + 1] + bayer[bayerStep3 + 3]) << 1)
                -
                (((bayer[2] + bayer[bayerStep2] +
                   bayer[bayerStep2 + 4] + bayer[bayerStep4 +
                                                 2]) * 3 + 1) >> 1)
                + rgb[blue] * 6;
            /* G at B */
            t1 = (((bayer[bayerStep + 2] + bayer[bayerStep2 + 1] +
                    bayer[bayerStep2 + 3] + bayer[bayerStep3 + 2])) << 1)
                - (bayer[2] + bayer[bayerStep2] +
                   bayer[bayerStep2 + 4] + bayer[bayerStep4 + 2])
                + (rgb[blue] << 2);
            t0 = (t0 + 4) >> 3;
            CLIP(t0, rgb[-blue]);
            t1 = (t1 + 4) >> 3;
            CLIP(t1, rgb[0]);
            bayer++;
            rgb += 3;
        }

        bayer -= width;
        rgb -= width * 3;

        blue = -blue;
        start_with_green = !start_with_green;
    }
    return 1;
}
dc1394color_filter_t Debayer_Pattern_Get(gchar * pattern)
{
    guint patterns[] = {*(guint*)"grbg",*(guint*)"rggb",*(guint*)"gbrg",*(guint*)"bggr"};
    dc1394color_filter_t retpatterns[] = {DC1394_COLOR_FILTER_GRBG , DC1394_COLOR_FILTER_RGGB, DC1394_COLOR_FILTER_GBRG, DC1394_COLOR_FILTER_BGGR};

    for (int i = 0; i < 4; i++) 
    {
        if(patterns[i] == *(guint*)pattern)
            return retpatterns[i];
    }
    return DC1394_COLOR_FILTER_INVALID;
}

int main(int argc ,char ** argv)
{
    g_print("Debayer Testbed\n");
    g_print("Required Arguments: \n\tinput_raw width height pattern output_directory\n\npattern has to be one of {grbg,rggb,gbrg,bggr}\n\n");

    guint64 st = g_get_real_time();
    gchar * contents;
    guint64 length;
    GError * error = NULL;

    if(argc != 6)
    {
        g_print("\"input_raw width height pattern output_directory \" should be passed as arguments\n");
        return -1;
    }
    gchar * filepath = argv[1];
    gchar * cwidth = argv[2];
    gchar * cheight= argv[3];
    gchar * cpattern = argv[4];
    gchar * outputpath = argv[5];

    if(FALSE == g_file_test(filepath, G_FILE_TEST_EXISTS))
    {
        g_print("input file %s doesn't exists",filepath);
        return -2;
    }
    if(FALSE == g_file_test(outputpath,G_FILE_TEST_IS_DIR))
    {
        g_print("input path %s doesn't exists",outputpath);
        return -3;
    }
    dc1394color_filter_t filter_pattern = Debayer_Pattern_Get(cpattern);
    if(DC1394_COLOR_FILTER_INVALID == filter_pattern)
    {
        g_print("invalid pattern %s used\n",cpattern);
        return -4;

    }
    
    gchar ** path = g_strsplit(filepath,"/",0);
    gchar * last = NULL;
    while(*path)
    {
        last = *path;
        path++;
    }

    guint64 width; 
    guint64 height;

    if(FALSE == g_ascii_string_to_unsigned( cwidth, 10, 2, 20000, &width, &error))
    {
        g_print("width %s is not supported",cwidth);
        return -5;
    }
    if(FALSE == g_ascii_string_to_unsigned( cheight, 10, 2, 20000, &height, &error))
    {
        g_print("height %s is not supported",cheight);
        return -6;
    }


    guint32 size = width*height;
    if(g_file_get_contents(filepath,&contents,&length,&error) && size == length)
    {
        g_print("file loaded\n");
        guint8 * rgb = g_malloc0(length*3);

#if 0
        st = g_get_real_time();
        if(dc1394_bayer_NearestNeighbor(contents,rgb,width,height,DC1394_COLOR_FILTER_GRBG))
        {
            guint64 et = g_get_real_time();
            guint64 exect = (et - st)/1000;
            g_print("execution time %lu \n",exect);
            gchar * filename = g_strdup_printf("raws/%s_%lums_%u_%s.raw","nn",exect,size,qp);
            g_file_set_contents(filename,rgb,width*height*3,&error);
            g_print("file processed %s\n",filename);
        }
        st = g_get_real_time();
        if(dc1394_bayer_Simple(contents,rgb,width,height,DC1394_COLOR_FILTER_GRBG))
        {
            guint64 et = g_get_real_time();
            guint64 exect = (et - st)/1000;
            g_print("execution time %lu \n",exect);
            gchar * filename = g_strdup_printf("raws/%s_%lums_%u_%s.raw","avt",exect,size,qp);
            g_file_set_contents(filename,rgb,width*height*3,&error);
            g_print("file processed %s\n",filename);
        }
        st = g_get_real_time();
        if(dc1394_bayer_EdgeSense(contents,rgb,width,height,DC1394_COLOR_FILTER_GRBG))
        {
            guint64 et = g_get_real_time();
            guint64 exect = (et - st)/1000;
            g_print("execution time %lu \n",exect);
            gchar * filename = g_strdup_printf("raws/%s_%lums_%u_%s.raw","edgesense",exect,size,qp);
            g_file_set_contents(filename,rgb,width*height*3,&error);
            g_print("file processed %s\n",filename);
        }
#endif
#ifdef AHD
        st = g_get_real_time();
        if(dc1394_bayer_AHD(contents,rgb,width,height,filter_pattern))
        {
            guint64 et = g_get_real_time();
            guint64 exect = (et - st)/1000;
            g_print("execution time %lu \n",exect);
            gchar * filename = g_strdup_printf("raws/%s_%lums.png","AHD",exect);
            stbi_write_png(filename,width, height, 3, rgb, 3*width);
            //g_file_set_contents(filename,rgb,width*height*3,&error);
            g_print("file processed %s\n",filename);
        }
#endif
#ifdef VNG 
        st = g_get_real_time();
        if(dc1394_bayer_VNG(contents,rgb,width,height,filter_pattern))
        {
            guint64 et = g_get_real_time();
            guint64 exect = (et - st)/1000;
            g_print("execution time %lu \n",exect);
            gchar * filename = g_strdup_printf("raws/%s_%lums.png","VNG",exect);
            stbi_write_png(filename,width, height, 3, rgb, 3*width);
            //g_file_set_contents(filename,rgb,width*height*3,&error);
            g_print("file processed %s\n",filename);
        }
#endif
#ifdef HQLinear
        st = g_get_real_time();
        if(dc1394_bayer_HQLinear(contents,rgb,width,height,filter_pattern))
        {
            guint64 et = g_get_real_time();
            guint64 exect = (et - st)/1000;
            g_print("execution time %lu \n",exect);
            gchar * filename = g_strdup_printf("raws/%s_%lums.png","HQLinear",exect);
            stbi_write_png(filename,width, height, 3, rgb, 3*width);
            //g_file_set_contents(filename,rgb,width*height*3,&error);
            g_print("file processed %s\n",filename);
        }
        g_free(rgb);
#endif

    }
    return 0;
}
