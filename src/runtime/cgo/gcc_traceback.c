// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo,darwin cgo,linux

#include <stdint.h>
#include "libcgo.h"

// Call the user's traceback function and then call sigtramp.
// The runtime signal handler will jump to this code.
// We do it this way so that the user's traceback function will be called
// by a C function with proper unwind info.
void
x_cgo_callers(uintptr_t sig, void *info, void *context, void (*cgoTraceback)(struct cgoTracebackArg*), uintptr_t* cgoCallers, void (*sigtramp)(uintptr_t, void*, void*)) {
	struct cgoTracebackArg arg;

	arg.Context = 0;
	arg.SigContext = (uintptr_t)(context);
	arg.Buf = cgoCallers;
	arg.Max = 32; // must match len(runtime.cgoCallers)
	(*cgoTraceback)(&arg);
	sigtramp(sig, info, context);
}
