// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that using the Go profiler in a C program does not crash.

#include <stddef.h>
#include <sys/time.h>

#include "libgo6.h"

int main(int argc, char **argv) {
	struct timeval tvstart, tvnow;
	int diff;

	gettimeofday(&tvstart, NULL);

	go_start_profile();

	// Busy wait so we have something to profile.
	// If we just sleep the profiling signal will never fire.
	while (1) {
		gettimeofday(&tvnow, NULL);
		diff = (tvnow.tv_sec - tvstart.tv_sec) * 1000 * 1000 + (tvnow.tv_usec - tvstart.tv_usec);

		// Profile frequency is 100Hz so we should definitely
		// get a signal in 50 milliseconds.
		if (diff > 50 * 1000)
			break;
	}

	go_stop_profile();
	return 0;
}
