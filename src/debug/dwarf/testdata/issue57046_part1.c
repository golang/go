// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Part 1 of the sources for issue 57046 test case.

// Build instructions:
//
// clang-16 -O -g -gdwarf-5 -c issue57046_part1.c
// clang-16 -O -g -gdwarf-5 -c issue57046_part2.c
// clang-16 -o issue57046-clang.elf5 issue57046_part1.o issue57046_part2.o

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

extern const char *mom();

int gadgety() {
  const char *ev = getenv("PATH");
  int n = strlen(ev);
  int s1 = (int)ev[0];
  int s2 = (int)ev[1];
  int s3 = (int)ev[2];
  for (int i = 0; i < strlen(ev); i++) {
    if (s1 == 101) {
	int t = s1;
	s1 = s3;
	s3 = t;
    }
    if (ev[i] == 99) {
      printf("%d\n", i);
    }
  }
  s2 *= 2;
  return n + s1 + s2;
}

int main(int argc, char **argv) {
  printf("Hi %s %d\n", mom(), gadgety());
  return 0;
}
