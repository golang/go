// Copyright 2016 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo
// +build darwin dragonfly freebsd linux netbsd openbsd solaris windows

#include "libcgo.h"

// The context function, used when tracing back C calls into Go.
void (*x_cgo_context_function)(struct context_arg*);

// Sets the context function to call to record the traceback context
// when calling a Go function from C code. Called from runtime.SetCgoTraceback.
void x_cgo_set_context_function(void (*context)(struct context_arg*)) {
	x_cgo_context_function = context;
}

// Releases the cgo traceback context.
void _cgo_release_context(uintptr_t ctxt) {
	if (ctxt != 0 && x_cgo_context_function != nil) {
		struct context_arg arg;

		arg.Context = ctxt;
		(*x_cgo_context_function)(&arg);
	}
}
