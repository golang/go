// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The C definitions for traceback.go. That file uses //export so
// it can't put function definitions in the "C" import comment.

#include <stddef.h>
#include <stdint.h>

#if defined(__APPLE__) || defined(__linux__)
#include <unwind.h>
#endif

char *p;

int crashInGo;
extern void h1(void);

static uintptr_t unwindPC1, unwindPC2, unwindPC3;

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static void tracebackCrash(void) {
#if defined(__APPLE__) || defined(__linux__)
	unwindPC3 = (uintptr_t)__builtin_extract_return_addr(__builtin_return_address(0));
#endif
	*p = 0;
}

int tracebackF3(void) {
#if defined(__APPLE__) || defined(__linux__)
	unwindPC2 = (uintptr_t)__builtin_extract_return_addr(__builtin_return_address(0));
#endif
	if (crashInGo)
		h1();
	else
		tracebackCrash();
	return 0;
}

int tracebackF2(void) {
#if defined(__APPLE__) || defined(__linux__)
	unwindPC1 = (uintptr_t)__builtin_extract_return_addr(__builtin_return_address(0));
#endif
	return tracebackF3();
}

int tracebackF1(void) {
	return tracebackF2();
}

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

void cgoTraceback(void* parg) {
	struct cgoTracebackArg* arg = (struct cgoTracebackArg*)(parg);
	arg->buf[0] = 1;
	arg->buf[1] = 2;
	arg->buf[2] = 3;
	arg->buf[3] = 0;
}

void cgoSymbolizer(void* parg) {
	struct cgoSymbolizerArg* arg = (struct cgoSymbolizerArg*)(parg);
	if (arg->pc != arg->data + 1) {
		arg->file = "unexpected data";
	} else {
		arg->file = "cgo symbolizer";
	}
	arg->lineno = arg->data + 1;
	arg->data++;
}

#if defined(__APPLE__) || defined(__linux__)
struct unwindState {
	uintptr_t* buf;
	uintptr_t  max;
	uintptr_t  n;
};

static _Unwind_Reason_Code unwindCallback(struct _Unwind_Context* ctx, void* arg) {
	struct unwindState* st = (struct unwindState*)(arg);
	uintptr_t pc = _Unwind_GetIP(ctx);
	if (pc != 0 && st->n + 1 < st->max) {
		st->buf[st->n++] = pc;
	}
	return _URC_NO_REASON;
}
#endif

void cgoUnwindTraceback(void* parg) {
	struct cgoTracebackArg* arg = (struct cgoTracebackArg*)(parg);
#if defined(__APPLE__) || defined(__linux__)
	if (arg->sigContext == 0) {
		arg->buf[0] = 0;
		return;
	}
	struct unwindState st = { arg->buf, arg->max, 0 };
	_Unwind_Backtrace(unwindCallback, &st);
	arg->buf[st.n] = 0;
#else
	arg->buf[0] = 0;
#endif
}

static int matchUnwindPC(uintptr_t pc, uintptr_t target) {
	return target != 0 && (pc == target || pc + 1 == target || pc == target + 1);
}

void cgoUnwindSymbolizer(void* parg) {
	struct cgoSymbolizerArg* arg = (struct cgoSymbolizerArg*)(parg);
	arg->file = NULL;
	arg->func = NULL;
	arg->lineno = 0;
	arg->more = 0;
	if (arg->pc == 0) {
		return;
	}
	if (matchUnwindPC(arg->pc, unwindPC3)) {
		arg->func = "tracebackF3";
		arg->file = "cgo unwind";
		arg->lineno = 3;
	} else if (matchUnwindPC(arg->pc, unwindPC2)) {
		arg->func = "tracebackF2";
		arg->file = "cgo unwind";
		arg->lineno = 2;
	} else if (matchUnwindPC(arg->pc, unwindPC1)) {
		arg->func = "tracebackF1";
		arg->file = "cgo unwind";
		arg->lineno = 1;
	}
}
