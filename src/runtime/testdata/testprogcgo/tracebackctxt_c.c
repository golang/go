// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The C definitions for tracebackctxt.go. That file uses //export so
// it can't put function definitions in the "C" import comment.

#include <stdlib.h>
#include <stdint.h>

// Functions exported from Go.
extern void G1(void);
extern void G2(void);
extern void TracebackContextPreemptionGoFunction(int);
extern void TracebackContextProfileGoFunction(void);

void C1() {
	G1();
}

void C2() {
	G2();
}

struct cgoContextArg {
	uintptr_t context;
};

struct cgoTracebackArg {
	uintptr_t  context;
	uintptr_t  sigContext;
	uintptr_t* buf;
	uintptr_t  max;
};

struct cgoSymbolizerArg {
	uintptr_t   pc;
	const char* file;
	uintptr_t   lineno;
	const char* func;
	uintptr_t   entry;
	uintptr_t   more;
	uintptr_t   data;
};

// Uses atomic adds and subtracts to catch the possibility of
// erroneous calls from multiple threads; that should be impossible in
// this test case, but we check just in case.
static int contextCount;

int getContextCount() {
	return __sync_add_and_fetch(&contextCount, 0);
}

void tcContext(void* parg) {
	struct cgoContextArg* arg = (struct cgoContextArg*)(parg);
	if (arg->context == 0) {
		arg->context = __sync_add_and_fetch(&contextCount, 1);
	} else {
		if (arg->context != __sync_add_and_fetch(&contextCount, 0)) {
			abort();
		}
		__sync_sub_and_fetch(&contextCount, 1);
	}
}

void tcContextSimple(void* parg) {
	struct cgoContextArg* arg = (struct cgoContextArg*)(parg);
	if (arg->context == 0) {
		arg->context = 1;
	}
}

void tcTraceback(void* parg) {
	int base, i;
	struct cgoTracebackArg* arg = (struct cgoTracebackArg*)(parg);
	if (arg->context == 0 && arg->sigContext == 0) {
		// This shouldn't happen in this program.
		abort();
	}
	// Return a variable number of PC values.
	base = arg->context << 8;
	for (i = 0; i < arg->context; i++) {
		if (i < arg->max) {
			arg->buf[i] = base + i;
		}
	}
}

void tcSymbolizer(void *parg) {
	struct cgoSymbolizerArg* arg = (struct cgoSymbolizerArg*)(parg);
	if (arg->pc == 0) {
		return;
	}
	// Report two lines per PC returned by traceback, to test more handling.
	arg->more = arg->file == NULL;
	arg->file = "tracebackctxt.go";
	arg->func = "cFunction";
	arg->lineno = arg->pc + (arg->more << 16);
}

void TracebackContextPreemptionCallGo(int i) {
	TracebackContextPreemptionGoFunction(i);
}

void TracebackContextProfileCallGo(void) {
	TracebackContextProfileGoFunction();
}
