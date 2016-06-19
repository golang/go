// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define fd (100)

// Tests libgo2.so, which does not export any functions.
// Read a string from the file descriptor and print it.
int main(void) {
  int i;
  ssize_t n;
  char buf[20];
  struct timespec ts;

  // The descriptor will be initialized in a thread, so we have to
  // give a chance to get opened.
  for (i = 0; i < 100; i++) {
    n = read(fd, buf, sizeof buf);
    if (n >= 0)
      break;
    if (errno != EBADF && errno != EINVAL) {
      fprintf(stderr, "BUG: read: %s\n", strerror(errno));
      return 2;
    }

    // An EBADF error means that the shared library has not opened the
    // descriptor yet.
    ts.tv_sec = 0;
    ts.tv_nsec = 1000000;
    nanosleep(&ts, NULL);
  }

  if (n < 0) {
    fprintf(stderr, "BUG: failed to read any data from pipe\n");
    return 2;
  }

  if (n == 0) {
    fprintf(stderr, "BUG: unexpected EOF\n");
    return 2;
  }

  if (n == sizeof buf) {
    n--;
  }
  buf[n] = '\0';
  printf("%s\n", buf);
  return 0;
}
