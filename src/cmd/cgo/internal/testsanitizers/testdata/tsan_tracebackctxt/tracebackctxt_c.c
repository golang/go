// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The C definitions for tracebackctxt.go. That file uses //export so
// it can't put function definitions in the "C" import comment.

#include <stdint.h>
#include <stdio.h>

// Functions exported from Go.
extern void G1(void);
extern void G2(void);

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

void tcContext(void* parg) {
	struct cgoContextArg* arg = (struct cgoContextArg*)(parg);
	if (arg->context == 0) {
		arg->context = 1;
	}
}

void tcTraceback(void* parg) {
	int base, i;
	struct cgoTracebackArg* arg = (struct cgoTracebackArg*)(parg);
	if (arg->max < 1) {
		return;
	}
	arg->buf[0] = 6; // Chosen by fair dice roll.
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
