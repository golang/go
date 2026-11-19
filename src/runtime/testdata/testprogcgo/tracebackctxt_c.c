// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The C definitions for tracebackctxt.go. That file uses //export so
// it can't put function definitions in the "C" import comment.

#include <stdlib.h>
#include <stdint.h>

#if defined(__APPLE__) || defined(__linux__)
#include <unwind.h>
#endif

// Functions exported from Go.
extern void G1(void);
extern void G2(void);
extern void G1Unwind(void);
extern void G2Unwind(void);
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

static uintptr_t ctxUnwindPC1, ctxUnwindPC2;
static volatile int ctxUnwindSink;

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static void C1UnwindInner(void) {
#if defined(__APPLE__) || defined(__linux__)
	ctxUnwindPC1 = (uintptr_t)__builtin_extract_return_addr(__builtin_return_address(0));
#endif
	G1Unwind();
	ctxUnwindSink++;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
void C1Unwind(void) {
	C1UnwindInner();
	ctxUnwindSink++;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
static void C2UnwindInner(void) {
#if defined(__APPLE__) || defined(__linux__)
	ctxUnwindPC2 = (uintptr_t)__builtin_extract_return_addr(__builtin_return_address(0));
#endif
	G2Unwind();
	ctxUnwindSink++;
}

#if defined(__GNUC__) || defined(__clang__)
__attribute__((noinline))
#endif
void C2Unwind(void) {
	C2UnwindInner();
	ctxUnwindSink++;
}

#define MAX_CTX_SLOTS 4
#define MAX_CTX_PCS 32
static uintptr_t ctxPCs[MAX_CTX_SLOTS][MAX_CTX_PCS];

#if defined(__APPLE__) || defined(__linux__)
struct ctxUnwindState {
	uintptr_t* buf;
	uintptr_t  max;
	uintptr_t  n;
};

static _Unwind_Reason_Code ctxUnwindCallback(struct _Unwind_Context* ctx, void* arg) {
	struct ctxUnwindState* st = (struct ctxUnwindState*)(arg);
	uintptr_t pc = _Unwind_GetIP(ctx);
	if (pc != 0 && st->n + 1 < st->max) {
		st->buf[st->n++] = pc;
	}
	return _URC_NO_REASON;
}
#endif

void tcUnwindContext(void* parg) {
	struct cgoContextArg* arg = (struct cgoContextArg*)(parg);
	if (arg->context == 0) {
		int slot = __sync_add_and_fetch(&contextCount, 1);
		if (slot < 1 || slot > MAX_CTX_SLOTS) {
			abort();
		}
		arg->context = slot;
#if defined(__APPLE__) || defined(__linux__)
		struct ctxUnwindState st = { ctxPCs[slot - 1], MAX_CTX_PCS, 0 };
		_Unwind_Backtrace(ctxUnwindCallback, &st);
		ctxPCs[slot - 1][st.n] = 0;
#else
		ctxPCs[slot - 1][0] = 0;
#endif
	} else {
		if (arg->context != __sync_add_and_fetch(&contextCount, 0)) {
			abort();
		}
		__sync_sub_and_fetch(&contextCount, 1);
	}
}

void tcUnwindTraceback(void* parg) {
	struct cgoTracebackArg* arg = (struct cgoTracebackArg*)(parg);
	if (arg->context < 1 || arg->context > MAX_CTX_SLOTS) {
		arg->buf[0] = 0;
		return;
	}
	uintptr_t* src = ctxPCs[arg->context - 1];
	uintptr_t i;
	for (i = 0; i + 1 < arg->max && src[i] != 0; i++) {
		arg->buf[i] = src[i];
	}
	arg->buf[i] = 0;
}

static int matchCtxPC(uintptr_t pc, uintptr_t target) {
	return target != 0 && (pc == target || pc + 1 == target || pc == target + 1);
}

void tcUnwindSymbolizer(void* parg) {
	struct cgoSymbolizerArg* arg = (struct cgoSymbolizerArg*)(parg);
	arg->file = NULL;
	arg->func = NULL;
	arg->lineno = 0;
	arg->more = 0;
	if (arg->pc == 0) {
		return;
	}
	if (matchCtxPC(arg->pc, ctxUnwindPC2)) {
		arg->func = "C2Unwind";
		arg->file = "tracebackctxt_c.c";
		arg->lineno = 2;
	} else if (matchCtxPC(arg->pc, ctxUnwindPC1)) {
		arg->func = "C1Unwind";
		arg->file = "tracebackctxt_c.c";
		arg->lineno = 1;
	}
}
