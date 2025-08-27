
#include <gst/gst.h>
#include "NvOptFlowLib/nv_opt_flow_lib.h"

struct timing_info
{
    const char * name;
    guint64 starttime;
    guint64 lasttime;
    guint64 lasttimeB;
    guint64 deltatime;
    guint64 lastdeltatime;
    guint64 dtsum;
    guint64 lastdeltatimeB;
    guint64 dtBsum;
    guint64 intercount;
};

static gboolean bus_call (GstBus     *bus, GstMessage *msg, gpointer    data)
{
  GMainLoop *loop = (GMainLoop *) data;

  switch (GST_MESSAGE_TYPE (msg)) {

    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;

    case GST_MESSAGE_ERROR: {
      gchar  *debug = NULL;
      GError *error = NULL;

      gst_message_parse_error (msg, &error, &debug);

      g_print("Error: %s\nDebug %s\n", error->message,debug ? debug : "none");
      g_error_free (error);
      g_free (debug);

      g_main_loop_quit (loop);
      break;
    }
    default:
      break;
  }

  return TRUE;
}

GstMapInfo GetBufferDataRead(GstBuffer * buffer, void** data, gint64 *size)
{
    GstMapInfo map =  GST_MAP_INFO_INIT;
    gst_buffer_map(buffer,&map, GST_MAP_READ);
    *data = map.data;
    *size = map.size;
    return map;
};

static gint64 thresh = 200;
static gint64 startTime;
static void * nv_opt_context;
//int16_t flow_data[1920*1080*1];
int16_t flow_data[1920*1080*2];
static GstElement * viewappsrc = NULL;
static GstCaps * viewcaps = NULL;

GstFlowReturn OnAppSample(GstElement* appsink, gpointer data)
{
    struct timing_info *timeI = (timing_info*)data;
    GstSample * sample = NULL;
    g_signal_emit_by_name (appsink, "pull-sample", &sample);
    if(sample)
    {
        gint64 timeT = g_get_real_time();

        timeI->deltatime = timeT - timeI->lasttime;
        //g_print("%s: rt %ld dt %ld sum %ld\n",name,timeT,dt,dtASum);
        guint8 * data = 0;
        gint64 size = 0;
        GstBuffer * buffer = gst_sample_get_buffer(sample);
        GstMapInfo info = GetBufferDataRead(buffer,(void**)&data,&size);
        uint32_t floww = 1920;
        uint32_t flowh = 1080;
        nv_opt_flow_get_flow_field(nv_opt_context,data,(uint8_t*)flow_data,floww,flowh);
        if(!viewappsrc)
        {
            GError * err = NULL;
            gchar* viewPipeStr = g_strdup_printf("appsrc name=asrc format=time is-live=true ! video/x-raw,format=ARGB,width=%d,height=%d ! videoconvert ! autovideosink sync=false",floww,flowh);
            GstElement * viewpipe = gst_parse_launch(viewPipeStr,&err);

            viewappsrc = gst_bin_get_by_name(GST_BIN(viewpipe),"asrc");
            viewcaps = gst_caps_new_simple("video/x-raw", "format", G_TYPE_STRING, "ARGB", "width", G_TYPE_INT, floww, "height", G_TYPE_INT, flowh, NULL);
            gst_element_set_state(viewpipe,GST_STATE_PLAYING);
            g_free(viewPipeStr);
        }

        if(viewappsrc)
        {
            int64_t dxsum = 0;
            int64_t dysum = 0;
            
            for (size_t i = 0; i < floww * flowh - 1; i+=2)
            {
                int16_t dx = flow_data[i];
                int16_t dy = flow_data[i + 1];
                dxsum += dx;
                dysum += dy;
            }
            uint64_t dxavg = dxsum /(flowh * floww);
            uint64_t dyavg = dysum /(flowh * floww);
            int16_t tx = 50;
            int16_t ty = 50;
            for (size_t y = 0; y < flowh; y++)
            {
                for (size_t x = 0; x < floww; x++)
                {
					int16_t* dx = (int16_t *) &flow_data[(y * floww+ x) * 2];
					int16_t* dy = (int16_t *) &flow_data[(y * floww+ x) * 2 + 1];
                    *dx -= dxavg;
                    *dy -= dyavg;
                    if (abs(*dx) > tx || abs(*dy) > ty)
                    {
                        uint8_t * argb = (uint8_t *) dx;
                        argb[0] = 255;
                        uint8_t * r = &argb[1]; 
                        uint8_t * g = &argb[2];
                        uint8_t * b = &argb[3];

                        float tx =  ( (float)*dx + 32768) / 65536;
                        float ty =  ( (float)*dy + 32768) / 65536;
                        
                        *r = ty * 255;
                        *g = (1.0 - ty) * 255 + (1.0 - tx) * 255; 
                        *b = (1.0 - ty) * 255 + (tx) * 255;
                    }
                    else
                    {
                        *dx = -1;
                        *dy = -1;
                    }
                }
            }
            GstBuffer * flowbuffer = gst_buffer_new_memdup(flow_data,sizeof(flow_data));
            //GstBuffer * flowbuffer = gst_buffer_new_memdup(flow_data,640 * 480 * 3);
            GstFlowReturn ret;
            GstSample * flowsample = gst_sample_new(flowbuffer,viewcaps,NULL,NULL);
            g_signal_emit_by_name(viewappsrc,"push-sample",flowsample,&ret);
            gst_sample_unref(flowsample);
            gst_buffer_unref(flowbuffer);
            //g_print("sample pushed %d\n", ret);

        }
        //g_print("Got buffer: size %lld pts :%lld  dts :%ld\n",size,buffer->pts,buffer->dts);

        timeI->lasttime = timeT;
        timeI->lastdeltatime = timeI->deltatime;
        gst_buffer_unmap(buffer,&info);

        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }

    return GST_FLOW_ERROR;
}


int main(int argc , char ** argv)
{
	gst_init(&argc,&argv);
	GMainLoop * loop = g_main_loop_new (NULL, FALSE);
	
	GError * err = NULL;

	startTime = g_get_real_time();
	const gchar * pipestr = "filesrc location=test.mkv ! decodebin ! videoconvert ! video/x-raw,format=NV12 ! tee name=t ! queue ! appsink emit-signals=true name=asink t. ! queue ! autovideosink sync=false";

	GstElement * pipe = gst_parse_launch(pipestr,&err);
	GstBus * bus = NULL;

	//DLL void nv_opt_flow_get_flow_field(void * contextptr, uint8_t * &data, uint8_t * out_data);
	uint32_t width = 1920;
	uint32_t height= 1080;
    uint32_t gridSize = 1;
	nv_opt_context = nv_opt_flow_get_context(width, height,gridSize ,   2); // 2 means nv12 format
	if(pipe && !err )
	{
	    g_print("parse ok starting\n");
	    GstElement * snk = gst_bin_get_by_name(GST_BIN(pipe),"asink");
	    struct timing_info a = {"one"};
	    g_signal_connect(snk,"new-sample", G_CALLBACK(OnAppSample),&a);
	    bus = gst_element_get_bus(pipe);
	    gst_bus_add_watch(bus,bus_call ,loop);
	    gst_element_set_state (pipe, GST_STATE_PLAYING);

	    g_main_loop_run(loop);
	}

	g_print("test 29 6 25 done\n");
}
