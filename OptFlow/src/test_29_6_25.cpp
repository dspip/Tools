#include <gst/gst.h>

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
        g_print("Got buffer: size %ld pts :%ld  dts :%ld\n",size,buffer->pts,buffer->dts);

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
	const gchar * pipestr = "filesrc location=test.mkv ! decodebin ! videoconvert ! video/x-raw,format=RGB ! appsink emit-signals=true name=asink";

	GstElement * pipe = gst_parse_launch(pipestr,&err);
	GstBus * bus = NULL;
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
