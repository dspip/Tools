
CUDA_FLAGS=`pkg-config --cflags --libs cuda-11.4 cudart-11.4 glib-2.0`
FLAGS='-L/usr/local/cuda-11.4/targets/x86_64-linux/lib/ -lnvjpeg -I /usr/local/cuda-11.4/targets/x86_64-linux/include'
echo $FLAGS
echo $CUDA_FLAGS
time gcc -Og -ggdb test.c -o nvenc $FLAGS $CUDA_FLAGS -lm
time gcc -Og -ggdb test_y_only.c -o nvenc_y $FLAGS $CUDA_FLAGS -lm
echo $?

