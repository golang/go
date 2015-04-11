// Copyright 2015 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: see issue #10410
// +build linux
// +build ppc64 ppc64le

#include <stdio.h>
#include <stdlib.h>

void
x_cgo_sys_thread_create(void* (*func)(void*), void* arg) {
	fprintf(stderr, "x_cgo_sys_thread_create not implemented");
	abort();
}

void
_cgo_wait_runtime_init_done() {
	// TODO(spetrovic): implement this method.
}

void
x_cgo_notify_runtime_init_done(void* dummy) {
	// TODO(spetrovic): implement this method.
}