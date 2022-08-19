INCLUDE_DIRS = -I../../../include -I/usr/local/include/opencv4

CXX = g++
CXXFLAGS += -O2 -Wall -std=c++11 $(INCLUDE_DIRS) `pkg-config --cflags opencv4`
LDFLAGS = `pkg-config --libs opencv4`


all: preprocessing

preprocessing: preprocessing.cc
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@ 

.PHONY: clean

clean:
	rm -f preprocessing
