// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdint.h>
#include <stdio.h>

#include "p.h"
#include "libgo.h"

// Tests libgo.so to export the following functions.
//   int8_t DidInitRun();
//   int8_t DidMainRun();
//   int32_t FromPkg();
//   uint32_t Divu(uint32_t, uint32_t);
int main(void) {
  int8_t ran_init = DidInitRun();
  if (!ran_init) {
    fprintf(stderr, "ERROR: DidInitRun returned unexpected results: %d\n",
            ran_init);
    return 1;
  }
  int8_t ran_main = DidMainRun();
  if (ran_main) {
    fprintf(stderr, "ERROR: DidMainRun returned unexpected results: %d\n",
            ran_main);
    return 1;
  }
  int32_t from_pkg = FromPkg();
  if (from_pkg != 1024) {
    fprintf(stderr, "ERROR: FromPkg=%d, want %d\n", from_pkg, 1024);
    return 1;
  }
  uint32_t divu = Divu(2264, 31);
  if (divu != 73) {
    fprintf(stderr, "ERROR: Divu(2264, 31)=%d, want %d\n", divu, 73);
    return 1;
  }
  // test.bash looks for "PASS" to ensure this program has reached the end. 
  printf("PASS\n");
  return 0;
}
