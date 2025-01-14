
#include <gst/gst.h>
#include <gst/cuda/gstcuda.h>
#pragma comment(lib, "libgstreamer-1.0.lib")

struct
{
	GstElement * pipeline;
	GstElement * appsrc;
	GstCudaContext * cucontext; 

} g_display_context;

void dg_initialize(CUcontext handler, CUdevice device)
{
	gst_init(0,0);
	gst_cuda_load_library();
	g_display_context.cucontext =  gst_cuda_context_new_wrapped (handler, device);
	g_print("handler %p device %d\n",handler,device);

	//g_display_context.cucontext =  gst_cuda_context_new(0);
	GError * error = NULL;
	g_display_context.pipeline = gst_parse_launch("appsrc format=time name=src ! video/x-raw(memory:CUDAMemory),format=RGB,width=10240,height=7200,framerate=3/1 ! cudadownload ! video/x-raw(memory:GLMemory) ! glimagesink sync=false",&error);
	//g_display_context.pipeline = gst_parse_launch("appsrc format=time name=src ! video/x-raw(memory:CUDAMemory),format=RGB,width=10240,height=7200,framerate=3/1 ! cudadownload ! video/x-raw(memory:GLMemory)  ! videocrop right=5240 ! video/x-raw,format=I420,width=5000,height=7096 ! autovideosink sync=false",&error);
	//g_display_context.pipeline = gst_parse_launch("appsrc format=time name=src ! video/x-raw(memory:CUDAMemory),format=RGB,width=10240,height=7200,framerate=3/1 ! cudadownload ! video/x-raw, format=RGB,width=10240,height=7200,framerate=3/1 ! videoconvert ! autovideosink",&error);
	//g_display_context.pipeline = gst_parse_launch("appsrc format=time name=src ! video/x-raw,format=RGB,width=10240,height=7200,framerate=3/1 ! videoconvert ! videocrop right=240 bottom=104  ! autovideosink sync=false",&error);
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
		GstBuffer * buffer = NULL;
		GstCaps * caps = NULL;
		if(g_display_context.cucontext)
		{
			caps = gst_caps_new_simple ("video/x-raw",
																 "format", G_TYPE_STRING, "RGB",
																 "framerate", GST_TYPE_FRACTION, 3, 1,
																 "width", G_TYPE_INT, 10240,
																 "height", G_TYPE_INT, 7200,
																 NULL); 
			if(caps)
			{
				buffer = gst_buffer_new();
				GstCapsFeatures *features = gst_caps_features_new("memory:CUDAMemory", NULL);
				gst_caps_set_features(caps, 0, features);
				GstVideoInfo * info =  gst_video_info_new_from_caps (caps);
				uint64_t dev_ptr =(uint64_t)(frame);  
				GstMemory * mem = gst_cuda_allocator_alloc_wrapped (NULL, g_display_context.cucontext,NULL,info,dev_ptr,NULL,NULL);
				g_print("memory %p\n",mem);
				gst_buffer_append_memory(buffer,mem);
				//gst_buffer_insert_memory(buffer,0,mem);
			}
		}
		else
		{
			buffer = gst_buffer_new_wrapped(frame,size);
		}
		if(buffer)
		{
			GstFlowReturn ret = GST_FLOW_OK; 
			GstSample * sample = gst_sample_new(buffer,caps,NULL,NULL);
			g_signal_emit_by_name(g_display_context.appsrc,"push-sample",sample,&ret);
			//g_signal_emit_by_name(g_display_context.appsrc,"push-buffer",buffer,&ret);
		}
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
