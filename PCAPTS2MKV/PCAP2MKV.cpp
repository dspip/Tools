
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
    //#include <libavdevice/avdevice.h>
}

#include "avpacket_logging.h" 

#include <assert.h>
#include <stdio.h>
#include <pcap.h>
#include <netinet/ip.h>
#include <net/ethernet.h>
#include <netinet/udp.h>
#include "types.h"
//#include "klvparse/klvparse.h"
#include <fstream>
#include<list>
#include<vector>

#define MAX_PCAP_SIZE 1 << 30
#define DATA_STREAMS_MAX_COUNT 10
#define MAX_SAMPLING_CONTEXTS 100 
#define MAX_OUTPUT_FIILE_NAME_LENGTH 2048 

#define MIN(X,Y) ((X) < (Y) ? (X) : (Y))

static u8 * g_staticMem = 0;

struct ip_port_location
{
    u32 src_ip;
    u32 src_port;
    u32 dst_ip;
    u16 dst_port;
    u64 location;
    u32 size;
};

struct seeker_format_context;
struct data_seeker
{
    u8 * baseptr;
    u8 * data;
    u64 position;
    u64 data_size;
    u64 total_data_size;
    std::list<seeker_format_context> input_streams; 
    u32 packet_count;
};

struct seeker_format_context
{
    data_seeker seeker;
    AVFormatContext *context;
    ip_port_location boundary;
};

void CloseFreeAVFormatCTX(AVFormatContext ** pctx)
{
    if(pctx && *pctx)
    {
        if((*pctx)->pb)
        {
            av_free((*pctx)->pb->buffer);
            avio_context_free(&(**pctx).pb);
        }
        //avformat_close_input(pctx);
        //avformat_free_context(*pctx);
        *pctx = NULL;
    }
    else 
    {
        av_log(0,AV_LOG_ERROR,"LINE: %d %s called with NULL \n",__LINE__,__FUNCTION__);
    }
}

s32 ReadFunc(void* ptr, uint8_t* buf, int buf_size)
{
    data_seeker * pStream = (data_seeker*)ptr;

    if(pStream->data_size <= pStream->position)
        return AVERROR_EOF;  

    u64 bytesRead = MIN(pStream->data_size-pStream->position,(u64)buf_size);
    void * hr = memcpy(buf,pStream->data + pStream->position,bytesRead);
    if(hr == 0)
        return AVERROR_EOF;  

    pStream->position += bytesRead;

    return bytesRead;
}

s64 SeekFunc(void * ptr, s64 offset, s32 whence)
{
    data_seeker * pStream = (data_seeker*)ptr;
    if(whence == SEEK_SET)
    {
        pStream->position = offset;
    }
    else if (whence == SEEK_CUR)
    {
        pStream->position = MIN(pStream->data_size,pStream->position + offset) ;
    }
    else if(whence == SEEK_END)
    {
        pStream->position = pStream->data_size;
    }

    return pStream->position;
}

AVFormatContext * GetCustomPCAPIO(data_seeker* stream)
{   
    // Create internal Buffer for FFmpeg:
    const int iBufSize = 512  * 512;
    u8 * pBuffer = new u8[iBufSize];
    // Allocate the AVIOContext:
    AVIOContext* pIOCtx = avio_alloc_context(pBuffer,iBufSize,  // internal Buffer and its size
            0,               // bWriteable (1=true,0=false) 
            stream,          // user data ; will be passed to our callback functions
            ReadFunc,        // Read callback 
            0,               // Write callback 
            SeekFunc);       // Seek callback 

    // Allocate the AVFormatContext:
    AVFormatContext* pCtx = avformat_alloc_context();

    pCtx->pb = pIOCtx;
    s32 bufsize = MIN(iBufSize,stream->data_size);
    av_log(0,AV_LOG_INFO,"LINE: %d bufsize %d , datasize %ld\n",__LINE__, iBufSize,stream->data_size);
    memcpy(pBuffer,stream->data,bufsize);

    AVProbeData probeData;
    probeData.buf = pBuffer;
    probeData.buf_size = iBufSize;
    probeData.filename = "";

    pCtx->iformat = av_probe_input_format(&probeData, 0);
    pCtx->flags = AVFMT_FLAG_CUSTOM_IO;
    pCtx->probesize = iBufSize;


    if(avformat_open_input(&pCtx, "", 0, 0) != 0)
    {
        CloseFreeAVFormatCTX(&pCtx);
        return 0;
    }

    pCtx->probesize = 1000 * 2000000;
    pCtx->max_analyze_duration = 3000000000;

    if(avformat_find_stream_info(pCtx, NULL) < 0)
    {
        CloseFreeAVFormatCTX(&pCtx);
        av_log(0,AV_LOG_ERROR,"LINE: %d %s couldn't not get stream info fmt_ctx\n",__LINE__,__FUNCTION__);
        return 0;
    }

    return pCtx;
}

void PrintError(int err)
{
    const int SIZE = 128;
    char errbuf[SIZE];
    av_strerror(err, errbuf, SIZE);
    av_log(0,AV_LOG_ERROR,"error %s\n", errbuf);
}

enum StreamContextState 
{
    SC_STARTED,
    SC_READY
};

struct remux_stream_context
{
    AVStream * stream = 0;
    AVFormatContext *fmt_ctx = 0;
    StreamContextState state;  
};

AVFormatContext * GetOutputStream(AVFormatContext * fmt_ctx,const char * outputFileName,s32 &out_videoStream)
{
    AVFormatContext * outputContext = NULL;
    AVStream * outStream = NULL;
    if(fmt_ctx == NULL)
    {
        av_log(0,AV_LOG_ERROR,"LINE: %d %s got invalid fmt_ctx\n",__LINE__,__FUNCTION__);
        return NULL;
    }
    av_dump_format(fmt_ctx, 0, "", 0);
    for (u32 i = 0; i < fmt_ctx->nb_streams; i++)
    {
        if (out_videoStream == -1 && fmt_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO)
        { 
            out_videoStream = i;
            AVStream * instream = fmt_ctx->streams[i];

            avformat_alloc_output_context2(&outputContext, NULL, NULL, outputFileName);
            if(!outputContext)
            {
                av_log(0,AV_LOG_ERROR,"LINE %d %s Could't alloc out context\n",__LINE__,__FUNCTION__);
                out_videoStream = -1;
            }
            outStream = avformat_new_stream(outputContext, NULL);
            s32 ret = avcodec_parameters_copy(outStream->codecpar,instream->codecpar);
            if (ret < 0)
            {
                av_log(0,AV_LOG_ERROR,"LINE %d %s Could't copy stream information\n",__LINE__,__FUNCTION__);
                out_videoStream = -1;
            }
            outStream->codecpar->codec_tag = 0;
            av_dump_format(outputContext, 0, outputFileName , 1);
            if (!(outputContext->oformat->flags & AVFMT_NOFILE))
            {
                ret = avio_open(&(outputContext->pb), outputFileName, AVIO_FLAG_WRITE);
                if (ret < 0)
                {
                    av_log(0,AV_LOG_ERROR, "LINE %d %s Could not open output file '%s'\n", __LINE__,__FUNCTION__ ,outputFileName);
                    out_videoStream = -1;
                    return NULL;
                }
                ret = avformat_write_header(outputContext, NULL);
                if (ret < 0)
                {
                    av_log(0,AV_LOG_ERROR, "LINE %d %s Could not write head for output file '%s'\n", __LINE__,__FUNCTION__, outputFileName);
                    out_videoStream = -1;
                    return NULL;
                }
            }
        }
    }
    return outputContext;
}

bool IsFileExists(const char * file)
{
    std::ifstream f(file);
    return f.good();
}

u8 * AllocData(u64 data_size)
{
    assert(data_size < MAX_PCAP_SIZE);
    //if(g_staticMem == 0)
    //    g_staticMem = new u8[MAX_PCAP_SIZE];  

    return new u8[data_size];
    //return g_staticMem;
}

void FreeData()
{
    delete [] g_staticMem;
    g_staticMem = 0;
}

data_seeker GetSeekContextFromFile(const s8 * path,s8 * pcapFilter = 0,u32 offset=0)
{
    data_seeker seeker = {};
    if(IsFileExists(path) == false)
        return seeker;
    FILE * fptr = fopen(path,"rb");
    if(fptr)
    {
        fseek(fptr,0,SEEK_END);
        seeker.data_size = ftell(fptr);
        seeker.total_data_size = seeker.data_size;
        rewind(fptr);
        u8 * packetData = new u8[seeker.data_size];
        if(!packetData)
        {
            seeker.data_size = 0;
            seeker.data = 0;
        }
        else
        {
            char errorbuf[PCAP_ERRBUF_SIZE] = {};
            pcap_t *fp = pcap_fopen_offline(fptr,errorbuf);
            if(fp == 0)
            {
                av_log(0,AV_LOG_ERROR,"%d: Error : %s \n",__LINE__,errorbuf);
                delete [] packetData;
                seeker.data = NULL;
                seeker.data_size = 0;
                return seeker;
            }

            u64 pcap_offset = sizeof(ip)+ sizeof(ether_header) + sizeof(udphdr);
            u64 accum = 0;
            bool resFilterCompiled = false;
            bpf_program filterProgram ={};
            if(pcapFilter) 
            {
                s32 resCompile = pcap_compile(fp,&filterProgram,pcapFilter,false, PCAP_NETMASK_UNKNOWN);
                if(resCompile == 0)
                {
                    resFilterCompiled = true;
                    pcap_setfilter(fp,&filterProgram);
                }
            }
            std::vector<std::list<ip_port_location>> packetOrdering;
            while(1)
            {
                u8 * pkt_data;
                pcap_pkthdr *pkt_header;
                s32 r = pcap_next_ex(fp, &pkt_header,(const u8**)&pkt_data);
                if(PCAP_ERROR_BREAK == r || r == PCAP_ERROR)
                    break;
                else 
                {  
                    u8 * dataStart = pkt_data + pcap_offset;
                    const u8 mpegTSIndicator = 0x47 ; 

                    if(mpegTSIndicator != *dataStart)
                        continue;


                    u64 csize = pkt_header->len-pcap_offset;
                    memcpy(packetData + accum, dataStart, csize);

                    u8 dst_addr[4] = {};
                    u8 src_addr[4] = {};
                    memcpy(dst_addr,pkt_data+0x1E,sizeof(dst_addr));
                    memcpy(src_addr,pkt_data+0x1A,sizeof(src_addr));
                    u8 dst_port[2] = {*(pkt_data+0x25),*(pkt_data+0x24)};
                    u8 src_port[2] = {*(pkt_data+0x23),*(pkt_data+0x22)};
                    ip_port_location currentPacket ={}; 
                    currentPacket.dst_port = ((u16 *)&dst_port[0])[0];
                    currentPacket.dst_ip= ((u32 *)&dst_addr[0])[0];
                    currentPacket.src_ip = ((u32 *)&src_addr[0])[0];
                    currentPacket.src_port = ((u16 *)&src_port[0])[0];
                    currentPacket.location = accum;
                    currentPacket.size = csize;
                    bool packetAdded = false;
                    for(auto &l : packetOrdering)
                    {
                        if(l.front().src_ip == currentPacket.src_ip && 
                           l.front().src_port == currentPacket.src_port && 
                           l.front().dst_ip == currentPacket.dst_ip &&
                           l.front().dst_port == currentPacket.dst_port)
                        {

                            l.push_back(currentPacket);
                            packetAdded = true;
                            break;
                        }
                    }
                    if(!packetAdded)
                    {
                        std::list<ip_port_location> newList;
                        newList.push_back(currentPacket);
                        packetOrdering.push_back(newList);
                    }
                    accum += csize;
                    seeker.packet_count++;
                }
            }
            if(resFilterCompiled)
            {
                pcap_freecode(&filterProgram); 
            }
            if(accum == 0)
            {
                delete [] packetData;
                seeker.data = NULL;
            }
            else
            {
                seeker.data = AllocData(accum);

                if(seeker.data)
                    seeker.baseptr = seeker.data;
                else 
                {
                    av_log(0,AV_LOG_ERROR,"LINE %d in %s couldn't allocate memory\n",__LINE__,__FUNCTION__);
                    return seeker;
                }
                
                u64 dataLocation = 0;
                for(u32 i = 0 ; i < packetOrdering.size(); i++)
                {   
                    ip_port_location orderedPacket = {}; 
                    u64 firstPacketLocation= dataLocation;
                    while(packetOrdering[i].size() >0)
                    {
                        orderedPacket = packetOrdering[i].front();
                        memcpy(seeker.data + dataLocation, packetData + orderedPacket.location, orderedPacket.size);
                        dataLocation += orderedPacket.size;
                        packetOrdering[i].pop_front();
                    }

                    orderedPacket.location = firstPacketLocation;
                    orderedPacket.size = dataLocation - firstPacketLocation;
                    data_seeker tseeker = seeker;

                    tseeker.data_size = dataLocation - firstPacketLocation;
                    tseeker.data = tseeker.baseptr + firstPacketLocation;


                    seeker_format_context sfc = {};
                    sfc.seeker = tseeker;
                    sfc.boundary = orderedPacket;

                    seeker.input_streams.push_back(sfc);
                    AVFormatContext * inContext = GetCustomPCAPIO(&(seeker.input_streams.back().seeker));

                    av_log(0,AV_LOG_ERROR,"LINE: %d AVFormatContext : %p \n",__LINE__,inContext);
                    if(inContext)
                    {
                        seeker.input_streams.back().context = inContext;
                    }
                    else 
                    {
                        seeker.input_streams.pop_back();
                    }
                }

                delete [] packetData;
                seeker.data_size = accum;
                av_log(0,AV_LOG_INFO,"LINE: %d Data size %ld\n",__LINE__, dataLocation);
                av_log(0,AV_LOG_INFO,"LINE: %d Data size Seeker %ld\n",__LINE__,seeker.data_size);
            }
            pcap_close(fp);
        }
    }

    av_log(0,AV_LOG_INFO,"LINE: %d returning seeker %ld\n",__LINE__,seeker.data_size);
    return seeker;
}

void CreateFileNameFromTemplate(s8 * outputFileName, const s8 * outputFilenameTemplate, s32 currentFileId,ip_port_location boundary)
{
    snprintf(outputFileName,MAX_OUTPUT_FIILE_NAME_LENGTH, outputFilenameTemplate, currentFileId,(boundary.src_ip) & 0xff ,(boundary.src_ip >> 8) & 0xff ,(boundary.src_ip >> 16) & 0xff ,boundary.src_ip >> 24 ,boundary.src_port,(boundary.dst_ip) & 0xff ,(boundary.dst_ip >> 8) & 0xff ,(boundary.dst_ip >> 16) & 0xff ,boundary.dst_ip >> 24 ,boundary.dst_port);
}

void UpdateOutputFileName(s32 &currentFileId, s8 * outputFileName,const s8 *outputFilenameTemplate,ip_port_location &boundary)
{
    currentFileId++;
    CreateFileNameFromTemplate(outputFileName,outputFilenameTemplate,currentFileId,boundary);
}

#ifdef WRITETS
void WriteTS(const char * outputFileTemplate, data_seeker &seekingContext,ip_port_location boundary)
{
    s8 tsFileName[MAX_OUTPUT_FIILE_NAME_LENGTH] = {};
    CreateFileNameFromTemplate(tsFileName,outputFileTemplate,1,boundary);
    s8 * tsNamePointer = tsFileName;
    s8 * lastDotPointer =0 ;
    while(*(tsNamePointer++))
    {
        if(tsNamePointer[0] == '.')
        {
            lastDotPointer = tsNamePointer;
        }
    }
    
    if(lastDotPointer)
    {
        lastDotPointer[1] = 't';
        lastDotPointer[2] = 's';
        lastDotPointer[3] = '\0';
    }

    std::ofstream tsStream(tsFileName);
    const char * data = (const char *)(seekingContext.data); 
    tsStream.write(data,seekingContext.data_size);
}
#else
void WriteTS(const char * outputFileTemplate, data_seeker &seekingContext,ip_port_location boundary){}
#endif

int main(int argc, char *argv[]) 
{ 
    av_log_set_level(AV_LOG_DEBUG);
    av_log(0,AV_LOG_INFO,"Usage: ./PCAP2MKV /path/to/pcapfile file_duration_millis\n");
    s32 args_duration = 2000000;
    if(argc >= 2)
    {
        av_log(0,AV_LOG_INFO,"Path: %s\n",argv[1]);
        if(argc == 3)
            args_duration = atoi(argv[2]);
        data_seeker seekingContext = GetSeekContextFromFile(argv[1]);
        AVPacket packet;
        s32 inStreamCounter = 0;
        av_log(0,AV_LOG_INFO,"streams found %ld\n",seekingContext.input_streams.size());
        while(seekingContext.input_streams.size()>0)
        {
            char nameBuff[2048] = {};
            s32 countS = seekingContext.input_streams.size();
            const char * nametemplate = "storage/video_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d_%d.mkv";
            seeker_format_context sfc = seekingContext.input_streams.front();
            UpdateOutputFileName(inStreamCounter,nameBuff,nametemplate,sfc.boundary);
            AVFormatContext * inContext = sfc.context;
            s32 videoStream = -1;
            AVFormatContext * outContext = GetOutputStream(inContext,nameBuff,videoStream);

            s32 readState = 0;
            s32 counter = 0;
            s32 pts = 0;
            while(readState >= 0)
            { 
                readState = av_read_frame(inContext, &packet);

                av_log(0,AV_LOG_INFO,"LINE: %d counter %d size: %d stream index %d  pts %ld \n",__LINE__, counter,packet.size,packet.stream_index,packet.pts);
                if(packet.stream_index == videoStream)
                {
                    av_packet_rescale_ts(&packet, inContext->streams[videoStream]->time_base, outContext->streams[0]->time_base);
                    packet.pts = pts += packet.duration;
                    packet.dts = packet.pts;
                    av_interleaved_write_frame(outContext,&packet);

                    if(pts >= args_duration)
                        break;
                }
                counter++;
            }
            av_write_trailer(outContext);

            avio_closep(&(outContext->pb));
            avio_context_free(&(outContext->pb));
            avformat_free_context(outContext);

            if(readState < 0)
            {

                WriteTS(nametemplate,sfc.seeker,sfc.boundary);
                inStreamCounter = 0;
                seekingContext.input_streams.pop_front();
            }

        }
        av_log(0,AV_LOG_INFO,"Line %d Completed processing \n",__LINE__);
//        readState = av_read_frame(inContext, packet);
    }
}
