// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix || windows

#include "libcgo.h"

// Releases the cgo traceback context.
void _cgo_release_context(uintptr_t ctxt) {
	void (*pfn)(struct cgoContextArg*);

	pfn = _cgo_get_context_function();
	if (ctxt != 0 && pfn != nil) {
		struct cgoContextArg arg;

		arg.Context = ctxt;
		(*pfn)(&arg);
	}
}
