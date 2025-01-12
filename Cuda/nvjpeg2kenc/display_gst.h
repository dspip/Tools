
#include <gst/gst.h>
#pragma comment(lib, "libgstreamer-1.0.lib")

struct
{
	GstElement * pipeline;
	GstElement * appsrc;

} g_display_context;

void dg_initialize()
{
	gst_init(0,0);
	GError * error = NULL;
	g_display_context.pipeline = gst_parse_launch("appsrc format=time name=src ! video/x-raw,format=RGB,width=10240,height=7200,framerate=3/1 ! videoconvert ! videocrop right=240 bottom=104  ! autovideosink sync=false",&error);
	if(g_display_context.pipeline && ! error)
	{
		g_display_context.appsrc = gst_bin_get_by_name(GST_BIN(g_display_context.pipeline),"src");
		gst_element_set_state(g_display_context.pipeline,GST_STATE_PLAYING);
	}
}
void dg_push_frame(void* frame,guint64 size)
{
	if(g_display_context.appsrc)
	{
 
		GstBuffer * buffer = gst_buffer_new_wrapped(frame,size);
		GstSample * sample = gst_sample_new(buffer,NULL,NULL,NULL);
		GstFlowReturn ret = GST_FLOW_OK; 
		g_signal_emit_by_name(g_display_context.appsrc,"push-sample",sample,&ret);
	}
}
void dg_deinitialize()
{
	if(g_display_context.pipeline)
	{
		gst_element_set_state(g_display_context.pipeline,GST_STATE_NULL);
		gst_object_unref(g_display_context.pipeline);
	}
}
