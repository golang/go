// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdint.h>
#include <stdio.h>
#include <dlfcn.h>

// Tests "main.main" is exported on android/arm,
// which golang.org/x/mobile/app depends on.
int main(int argc, char** argv) {
  void* handle = dlopen(argv[1], RTLD_LAZY | RTLD_GLOBAL);
  if (!handle) {
    fprintf(stderr, "ERROR: failed to open the shared library: %s\n",
            dlerror());
    return 2;
  }

  uintptr_t main_fn = (uintptr_t)dlsym(handle, "main.main");
  if (!main_fn) {
    fprintf(stderr, "ERROR: missing main.main: %s\n", dlerror());
    return 2;
  }

  // TODO(hyangah): check that main.main can run.

  printf("PASS\n");
  return 0;
}
