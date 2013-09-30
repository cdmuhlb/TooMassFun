#include "BufferedWriter.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

BufferedFloatWriter* bfw_newFile(const char* dir, const char* file) {
  const int pathLen = strlen(dir) + strlen(file) + 2;
  char* pathBuf = (char*)malloc(pathLen);
  const int ret = snprintf(pathBuf, pathLen, "%s/%s", dir, file);
  assert(ret < pathLen);
  FILE* out = fopen(pathBuf, "w");
  free(pathBuf);
  return bfw_new(out);
}

BufferedFloatWriter* bfw_new(FILE* out) {
  BufferedFloatWriter* bfw = (BufferedFloatWriter*)malloc(
      sizeof(BufferedFloatWriter));
  bfw->capacity = 4096;
  bfw->buffer = (float*)malloc(bfw->capacity*sizeof(float));
  bfw->size = 0;
  bfw->stream = out;
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
