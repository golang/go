// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdio.h>
#include <stdlib.h>
#include "libcgo.h"

// The context function, used when tracing back C calls into Go.
static void (*cgo_context_function)(struct context_arg*);

void
x_cgo_sys_thread_create(void* (*func)(void*), void* arg) {
	fprintf(stderr, "x_cgo_sys_thread_create not implemented");
	abort();
}

uintptr_t
_cgo_wait_runtime_init_done() {
	void (*pfn)(struct context_arg*);

	// TODO(spetrovic): implement this method.

	pfn = _cgo_get_context_function();
	if (pfn != nil) {
		struct context_arg arg;

		arg.Context = 0;
		(*pfn)(&arg);
		return arg.Context;
	}
	return 0;
}

void
x_cgo_notify_runtime_init_done(void* dummy) {
	// TODO(spetrovic): implement this method.
}

// Sets the context function to call to record the traceback context
// when calling a Go function from C code. Called from runtime.SetCgoTraceback.
void x_cgo_set_context_function(void (*context)(struct context_arg*)) {
	// TODO(iant): Needs synchronization.
	cgo_context_function = context;
}

// Gets the context function.
void (*(_cgo_get_context_function(void)))(struct context_arg*) {
	return cgo_context_function;
}
