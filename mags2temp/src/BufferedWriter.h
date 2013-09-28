#ifndef BUFFEREDWRITER_H_
#define BUFFEREDWRITER_H_

#include <stdio.h>

typedef struct {
  float* buffer;
  int capacity;
  int size;
  FILE* stream;
} BufferedFloatWriter;

BufferedFloatWriter* bfw_new(const char* filename);
void bfw_put(BufferedFloatWriter* bfw, float x);
void bfw_close(BufferedFloatWriter* bfw);

#endif  // BUFFEREDWRITER_H_
