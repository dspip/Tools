
CUDA_FLAGS=`pkg-config --cflags --libs cuda-12.6 cudart-12.6 glib-2.0`
FLAGS='-L/usr/lib/x86_64-linux-gnu/libnvjpeg2k/11/ -lnvjpeg2k '
#FLAGS='-lnvjpeg2k -o nvenc'
echo $FLAGS
echo $CUDA_FLAGS
time gcc -Ofast -ggdb test.c $FLAGS $CUDA_FLAGS -o nvenc -lm
time gcc -Ofast -ggdb decode.c $FLAGS $CUDA_FLAGS -o nvdec -lm
echo $?

