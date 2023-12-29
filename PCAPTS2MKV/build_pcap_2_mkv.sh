g++ -O3 -ggdb -o PCAP2MKV PCAP2MKV.cpp `pkg-config --libs libavformat libavcodec libavutil` -lklvparse -lpcap -D WRITETS
