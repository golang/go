// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdint.h>
#include <stdio.h>
#include <dlfcn.h>

int check_int8(void* handle, const char* fname, int8_t want) {
  int8_t (*fn)();
  fn = (int8_t (*)())dlsym(handle, fname);
  if (!fn) {
    fprintf(stderr, "ERROR: missing %s: %s\n", fname, dlerror());
    return 1;
  }
  signed char ret = fn();
  if (ret != want) {
    fprintf(stderr, "ERROR: %s=%d, want %d\n", fname, ret, want);
    return 1;
  }
  return 0;
}

int check_int32(void* handle, const char* fname, int32_t want) {
  int32_t (*fn)();
  fn = (int32_t (*)())dlsym(handle, fname);
  if (!fn) {
    fprintf(stderr, "ERROR: missing %s: %s\n", fname, dlerror());
    return 1;
  }
  int32_t ret = fn();
  if (ret != want) {
    fprintf(stderr, "ERROR: %s=%d, want %d\n", fname, ret, want);
    return 1;
  }
  return 0;
}

// Tests libgo.so to export the following functions.
//   int8_t DidInitRun() // returns true
//   int8_t DidMainRun() // returns true
//   int32_t FromPkg() // returns 1024
int main(int argc, char** argv) {
  void* handle = dlopen(argv[1], RTLD_LAZY | RTLD_GLOBAL);
  if (!handle) {
    fprintf(stderr, "ERROR: failed to open the shared library: %s\n",
		    dlerror());
    return 2;
  }

  int ret = 0;
  ret = check_int8(handle, "DidInitRun", 1);
  if (ret != 0) {
    return ret;
  }

  ret = check_int8(handle, "DidMainRun", 0);
  if (ret != 0) {
    return ret;
  }

  ret = check_int32(handle, "FromPkg", 1024);
  if (ret != 0) {
   return ret;
  }
  // test.bash looks for "PASS" to ensure this program has reached the end. 
  printf("PASS\n");
  return 0;
}
