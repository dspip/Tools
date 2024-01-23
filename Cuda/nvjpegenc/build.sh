
CUDA_FLAGS=`pkg-config --cflags --libs cuda-11.4 cudart-11.4 glib-2.0`
FLAGS='-L/usr/local/cuda-11.4/targets/x86_64-linux/lib/ -lnvjpeg -o nvenc -I /usr/local/cuda-11.4/targets/x86_64-linux/include'
echo $FLAGS
echo $CUDA_FLAGS
time gcc -Og -ggdb test.c $FLAGS $CUDA_FLAGS
echo $?

