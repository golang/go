// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build cgo,darwin cgo,linux

#include <stdint.h>
#include "libcgo.h"

#ifndef __has_feature
#define __has_feature(x) 0
#endif

#if __has_feature(memory_sanitizer)
#include <sanitizer/msan_interface.h>
#endif

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

#if __has_feature(memory_sanitizer)
        // This function is called directly from the signal handler.
        // The arguments are passed in registers, so whether msan
        // considers cgoCallers to be initialized depends on whether
        // it considers the appropriate register to be initialized.
        // That can cause false reports in rare cases.
        // Explicitly unpoison the memory to avoid that.
        // See issue #47543 for more details.
        __msan_unpoison(&arg, sizeof arg);
#endif

	(*cgoTraceback)(&arg);
	sigtramp(sig, info, context);
}
