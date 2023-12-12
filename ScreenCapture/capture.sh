gst-launch-1.0 ximagesrc startx=1920 use-damage=0 ! video/x-raw,framerate=120/1 ! videoconvert ! video/x-raw,width=1920,height=1080,framerate=120/1,format=I420 ! queue !  x264enc speed-preset=ultrafast bitrate=2000000 ! video/x-h264 ! matroskamux ! filesink location=capture.mkv async=true sync=false


