// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "cgoopt.h"
#include <stdatomic.h>
#include <time.h>

extern int CgoOptGrow(int value);

static int *retained;
static _Atomic int entered;
static _Atomic int released;

uint64_t opt_scalar(uint64_t value) {
	return value + 1;
}

opt_value opt_aggregate(uint64_t value) {
	opt_value result = {value, (double)value + 0.5};
	return result;
}

int opt_errno(int value) {
	if (value < 0) {
		errno = EINVAL;
	}
	return value;
}

int *opt_retain(int *value) {
	retained = value;
	return value;
}

int opt_retained(void) {
	return *retained;
}

void opt_forget(void) {
	retained = 0;
}

int opt_callback(int value) {
	return CgoOptGrow(value) + 1;
}

int opt_wait(void) {
	time_t start = time(0);
	atomic_store_explicit(&entered, 1, memory_order_release);
	while (!atomic_load_explicit(&released, memory_order_acquire)) {
		// Bound a failed scheduler-progress test even if Go cannot run its timer.
		if (time(0) - start >= 10) {
			return -1;
		}
	}
	return 42;
}

int opt_entered(void) {
	return atomic_load_explicit(&entered, memory_order_acquire);
}

void opt_release(void) {
	atomic_store_explicit(&released, 1, memory_order_release);
}
