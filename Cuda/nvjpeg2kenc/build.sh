
CUDA_FLAGS=`pkg-config --cflags --libs cuda-11.4 cudart-11.4 glib-2.0`
#FLAGS='-L/usr/lib/x86_64-linux-gnu/libnvjpeg2k/12/ -lnvjpeg2k -o nvenc'
FLAGS='-lnvjpeg2k -o nvenc'
echo $FLAGS
echo $CUDA_FLAGS
time gcc -Ofast -ggdb test.c $FLAGS $CUDA_FLAGS
echo $?

