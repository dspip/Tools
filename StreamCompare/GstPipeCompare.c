#include <gst/gst.h>
#include<glib.h>

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
static gboolean
bus_call (GstBus     *bus,
          GstMessage *msg,
          gpointer    data)
{
  GMainLoop *loop = (GMainLoop *) data;

  switch (GST_MESSAGE_TYPE (msg)) {

    case GST_MESSAGE_EOS:
      g_print ("End of stream\n");
      g_main_loop_quit (loop);
      break;

    case GST_MESSAGE_ERROR: {
      gchar  *debug;
      GError *error;

      gst_message_parse_error (msg, &error, &debug);
      g_free (debug);

      g_printerr ("Error: %s\n", error->message);
      g_error_free (error);

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

GstBuffer * twosBuffer = NULL;

gint64 prevTimeA;
gint64 prevTimeB;
gint64 countA;
gint64 countB;
gint64 dtASum = 0;
gint64 dtBSum = 0; 
gint64 prevDtA;
gint64 prevDtB;
gint64 prevTimeThreshA;
gint64 prevTimeThreshB;
gint64 deltaBlinkSumA;
gint64 deltaBlinkSumB;

static gint64 jump = 4 * 704 * 4;
static gint64 thresh = 200;
static gint64 startTime;
static gint64 blinkCount=0;
GstFlowReturn OnAppSample(GstElement* appsink, gpointer data)
{
    struct timing_info *timeI = data;
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
        data += jump;
        //g_print("size %ld\n",size);

        if(timeI->lasttime !=0)
        {
            timeI->dtsum+=timeI->deltatime;
            timeI->intercount++;
            //g_print("dtASum %ld count %ld avg delta A %ld ddab %ld dddab %ld\n",dtASum - dtBSum, countA,dtASum/countA,timeT - prevTimeB,prevDtA - prevDtB);           

            if(data[0] > thresh && data[1]> thresh && data[2] > thresh)
            {
                gint64 t = (g_get_real_time()-startTime )/1000;
                timeI->starttime = timeI->starttime == 0  ?  t : timeI->starttime;
                gint64 deltaBlink = t - timeI->lasttimeB;
                timeI->dtBsum += deltaBlink;
                timeI->lasttimeB= t;

                g_print("%s %d %d %d %d %ld %ld %ld %ld %ld\n",timeI->name,data[0],data[1],data[2],data[3],t,deltaBlink,timeI->dtBsum,++blinkCount,timeI->lasttimeB - timeI->starttime);
            }
        }

        timeI->lasttime = timeT;
        timeI->lastdeltatime = timeI->deltatime;
        gst_buffer_unmap(buffer,&info);

        gst_sample_unref(sample);
        return GST_FLOW_OK;
    }


    return GST_FLOW_ERROR;
}

int main(int argc,char**argv)
{
    startTime = g_get_real_time();
    GstBus * bus;
    guint bus_watch_id;
    GError * error = 0;
    GError * error1 = 0;
    gst_init(&argc,&argv);
    GMainLoop * loop = g_main_loop_new (NULL, FALSE);

    GstElement * pipeline = gst_parse_launch("udpsrc uri=udp://239.3.0.1:6001  ! mpeg4filtertest enable=true ! mpeg4videoparse ! video/mpeg,framerate=100/1 ! queue max-size-buffers=1 !  avdec_mpeg4 max-errors=-1 ! videoconvert ! video/x-raw,width=704,height=480,format=BGRA ! appsink name=snk sync=true emit-signals=true",&error);
    GstElement * pipeline1 = gst_parse_launch("udpsrc uri=udp://239.3.0.1:6001 ! mpeg4filter enable=true ! mpeg4videoparse ! video/mpeg,framerate=100/1 ! queue max-size-buffers=1 ! avdec_mpeg4 max-errors=-1 ! videoconvert ! video/x-raw,width=704,height=480,format=BGRA ! appsink name=snk sync=true emit-signals=true",&error1);

    if(error || pipeline == NULL) 
    {
        g_printerr ("Error: %s\n", error->message);
        return -1;
    }
    if(pipeline1 == NULL || error1)
    {
        g_printerr ("Error: %s\n", error1->message);
        return -1;
    }

    GstElement * snk = gst_bin_get_by_name(GST_BIN(pipeline),"snk");
    GstElement * snk1 = gst_bin_get_by_name(GST_BIN(pipeline1),"snk");

    struct timing_info a = {"one"};
    struct timing_info b = {"two"};
    g_signal_connect(snk,"new-sample", G_CALLBACK(OnAppSample),&a);
    g_signal_connect(snk1,"new-sample", G_CALLBACK(OnAppSample),&b);

    g_print("snk %p , snk1 %p \n",snk,snk1);

    gst_element_set_state (pipeline, GST_STATE_PLAYING);
    gst_element_set_state (pipeline1, GST_STATE_PLAYING);
    /* Out of the main loop, clean up nicely */
    g_print ("Running...\n");
    g_main_loop_run (loop);

    g_print ("Returned, stopping playback\n");
    gst_element_set_state (pipeline, GST_STATE_NULL);

    g_print ("Deleting pipeline\n");
    gst_object_unref (GST_OBJECT (pipeline));
    g_source_remove (bus_watch_id);
    g_main_loop_unref (loop);

    return 0;

}
