#include "BufferedWriter.h"

#include <stdlib.h>

BufferedFloatWriter* bfw_new(const char* filename) {
  BufferedFloatWriter* bfw = (BufferedFloatWriter*)malloc(
      sizeof(BufferedFloatWriter));
  bfw->capacity = 4096;
  bfw->buffer = (float*)malloc(bfw->capacity*sizeof(float));
  bfw->size = 0;
  bfw->stream = fopen(filename, "w");
  return bfw;
}

void bfw_put(BufferedFloatWriter* bfw, float x) {
  bfw->buffer[bfw->size] = x;
  ++bfw->size;
  if (bfw->size == bfw->capacity) {
    fwrite(bfw->buffer, sizeof(float), bfw->size, bfw->stream);
    bfw->size = 0;
  }
}

void bfw_close(BufferedFloatWriter* bfw) {
  fwrite(bfw->buffer, sizeof(float), bfw->size, bfw->stream);
  fclose(bfw->stream);
  free(bfw->buffer);
  free(bfw);
}
