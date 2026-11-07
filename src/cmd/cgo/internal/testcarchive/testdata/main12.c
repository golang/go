// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that asynchronous preemption preserves the registers of the
// preempted goroutine.

#include <stdio.h>

#include "libgo12.h"

// The result of mix(N) when nothing disturbs it. After a change to mix,
// run this program with GODEBUG=asyncpreemptoff=1, which turns off
// asynchronous preemption, and take the value from the error message.
#define N 5000000
#define WANT 0x6f9ef1896a95d3c7ull

int main(void) {
	GoUint64 got = PreemptedMix(N);
	if (got != WANT) {
		fprintf(stderr, "mix(%d) = %#llx, want %#llx\n", N, (unsigned long long)got, WANT);
		return 2;
	}
	return 0;
}
