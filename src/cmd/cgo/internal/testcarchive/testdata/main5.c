// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test for verifying that the Go runtime properly forwards
// signals when non-Go signals are raised.

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/select.h>

#include "libgo2.h"

int *nilp;

int main(int argc, char** argv) {
	int verbose;
	int test;

	if (argc < 2) {
		printf("Missing argument\n");
		return 1;
	}

	test = atoi(argv[1]);

	verbose = (argc > 2);

	Noop();

	switch (test) {
		case 1: {
			if (verbose) {
				printf("attempting segfault\n");
			}

			*nilp = 0;
			break;
		}

		case 2: {
			struct timeval tv;

			if (verbose) {
				printf("attempting external signal test\n");
			}

			fprintf(stderr, "OK\n");
			fflush(stderr);

			// The program should be interrupted before
			// this sleep finishes. We use select rather
			// than sleep because in older versions of
			// glibc the sleep function does some signal
			// fiddling to handle SIGCHLD.  If this
			// program is fiddling signals just when the
			// test program sends the signal, the signal
			// may be delivered to a Go thread which will
			// break this test.
			tv.tv_sec = 60;
			tv.tv_usec = 0;
			select(0, NULL, NULL, NULL, &tv);

			break;
		}
		case 3: {
			if (verbose) {
				printf("attempting SIGPIPE\n");
			}

			int fd[2];
			if (pipe(fd) != 0) {
				printf("pipe(2) failed\n");
				return 0;
			}
			// Close the reading end.
			close(fd[0]);
			// Expect that write(2) fails (EPIPE)
			if (write(fd[1], "some data", 9) != -1) {
				printf("write(2) unexpectedly succeeded\n");
				return 0;
			}
			printf("did not receive SIGPIPE\n");
			return 0;
		}
		case 4: {
			fprintf(stderr, "OK\n");
			fflush(stderr);

			if (verbose) {
				printf("calling Block\n");
			}
			Block();
		}
		default:
			printf("Unknown test: %d\n", test);
			return 0;
	}

	printf("FAIL\n");
	return 0;
}
