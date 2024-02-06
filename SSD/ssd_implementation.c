#include "glib.h"

guint32 GetIndexAt(int x, int y, int w)
{
    return y * w + x;
}

char GetValueAt(char * contents , int x, int y, int w)
{
    return contents[y * w + x];
}

void hamilton_adams(gchar * data, gchar * mask,gchar * odata,gchar wantedcolor,guint32 w,guint32 h)
{
    //the green  for grbg 
    int advanceyby = 1;
    if(wantedcolor == 'b' || wantedcolor == 'r')
        advanceyby = 2;
    
    for (int y = 2; y < h-2; y+= advanceyby) 
    {
        int x = 2; // bgbgbg
        if(wantedcolor == 'g')
        {
            if(y % 2 == 0) // grgrgr
                x = 3; 
        }

        for (; x < w-2; x+=2)
        {
            gchar val = GetValueAt(mask,x,y,w); 
            if(val != wantedcolor)
            {
                gint8 r =  GetValueAt(data,x,y,w);
                gint8 r1 =  GetValueAt(data,x,y-2,w);
                gint8 r7 =  GetValueAt(data,x+2,y,w);
                gint8 r3 =  GetValueAt(data,x-2,y,w);
                gint8 r9 =  GetValueAt(data,x,y+2,w);

                gint8 g4 = GetValueAt(data,x-1,y,w); 
                gint8 g6 = GetValueAt(data,x+1,y,w); 
                gint8 g2 = GetValueAt(data,x,y-1,w); 
                gint8 g8 = GetValueAt(data,x,y+1,w); 

                gint16 dh = abs(g6 - g4) + abs(r - r3 + r - r7);
                gint16 dv = abs(g2 - g8) + abs(r - r1 + r - r9);
                guint8 g = 0;
                if(dh > dv)
                {
                    g = (g4 + g8)/2 + (r - r1 + r - r9)/4;
                }
                else if(dh < dv)
                {
                    g = (g4 + g6)/2 + (r - r3 + r - r7)/4;
                }
                else 
                {
                    g = (g4 + g6 + g2 + g8)/4 + (r - r3 + r - r7 + r - r1 + r - r9)/8;
                }
                //g_print("%u %u %u %u\n", dh, dv, g ,r);
                odata[GetIndexAt(x,y,w)] = g;
            }

        }
    }
}

int main(int argc,char ** argv)
{
    GError * err;
    gchar * contents;
    gsize length = 0;
    guint64 stime = g_get_real_time();
    g_file_get_contents("10000x7096.raw",&contents,&length,&err);
    guint32 width = 10000;
    guint32 height = 7096 ;

    gchar * data = g_malloc0( 3 * width * height);
    gchar * green = data;
    gchar * blue  = data + width * height;
    gchar * red   = data + 2 * width * height;
    gchar * mask = g_malloc0(width * height);

    if(length == height * width)
    {
        for (int i = 0; i < length; i++) 
        {
            guint32 row = i / width;
            guint32 col = i % width;
            if(row % 2 == 0)
            {// grgr
                if(col % 2 == 0)
                {
                    mask[i] = 'g';
                    green[i] = contents[i];
                }
                else 
                {
                    mask[i] = 'r';
                    red[i] = contents[i];
                }
            }
            else 
            {
                //bgbg
                if(col % 2 == 0)
                {

                    mask[i] = 'b';
                    blue[i] = contents[i];
                }
                else
                {
                    mask[i] = 'g';
                    green[i] = contents[i];
                }
            }
        }
    }
    
    hamilton_adams(data,mask,green,'g',width,height);
    hamilton_adams(data,mask,red,'r',width,height);
    hamilton_adams(data,mask,blue,'b',width,height);
    guint64 etime = g_get_real_time();
     
    g_print("length: %lu %lums\n",length,(etime - stime)/1000);
}
