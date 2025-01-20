
CUDA_FLAGS=`pkg-config --cflags --libs cuda-12.6 cudart-12.6 glib-2.0`
FLAGS='-L/usr/lib/x86_64-linux-gnu/libnvjpeg2k/11/ -lnvjpeg2k '
GST_FLAGS=`pkg-config --libs --cflags gstreamer-1.0 gstreamer-cuda-1.0`
#FLAGS='-lnvjpeg2k -o nvenc'
echo $FLAGS
echo $CUDA_FLAGS
time gcc -O3 -ggdb test.c $FLAGS $CUDA_FLAGS -o nvenc -lm
time gcc -O3 -ggdb decode.c $FLAGS $CUDA_FLAGS $GST_FLAGS -o nvdec -lm

time g++ -O3 -ggdb --shared nu_j2k_lib.cpp $FLAGS $CUDA_FLAGS $GST_FLAGS -o libnuj2k.so -lm

echo $?

