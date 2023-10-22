CC = g++
CFLAGS = -std=c++11
LDFLAGS = -L./opencv
LIBS = -lopencv_core -lopencv_highgui -lopencv_imgcodecs
INCLUDES = -I./opencv/opencv4

all: main

main: main.cpp
	$(CC) $(CFLAGS) -o main main.cpp $(LDFLAGS) $(LIBS) $(INCLUDES)

clean:
	rm -f main
