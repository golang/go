// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifdef __CYGWIN__
#error "don't use the cygwin compiler to build native Windows programs; use MinGW instead"
#endif

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdio.h>
#include <stdlib.h>

#include "libcgo.h"
#include "libcgo_windows.h"

// Ensure there's one symbol marked __declspec(dllexport).
// If there are no exported symbols, the unfortunate behavior of
// the binutils linker is to also strip the relocations table,
// resulting in non-PIE binary. The other option is the
// --export-all-symbols flag, but we don't need to export all symbols
// and this may overflow the export table (#40795).
// See https://sourceware.org/bugzilla/show_bug.cgi?id=19011
__declspec(dllexport) int _cgo_dummy_export;

static volatile LONG runtime_init_once_gate = 0;
static volatile LONG runtime_init_once_done = 0;

static CRITICAL_SECTION runtime_init_cs;

static HANDLE runtime_init_wait;
static int runtime_init_done;

// No pthreads on Windows, these are always zero.
uintptr_t x_cgo_pthread_key_created;
void (*x_crosscall2_ptr)(void (*fn)(void *), void *, int, size_t);

// Pre-initialize the runtime synchronization objects
void
_cgo_preinit_init() {
	 runtime_init_wait = CreateEvent(NULL, TRUE, FALSE, NULL);
	 if (runtime_init_wait == NULL) {
		fprintf(stderr, "runtime: failed to create runtime initialization wait event.\n");
		abort();
	 }

	 InitializeCriticalSection(&runtime_init_cs);
}

// Make sure that the preinit sequence has run.
void
_cgo_maybe_run_preinit() {
	 if (!InterlockedExchangeAdd(&runtime_init_once_done, 0)) {
			if (InterlockedIncrement(&runtime_init_once_gate) == 1) {
				 _cgo_preinit_init();
				 InterlockedIncrement(&runtime_init_once_done);
			} else {
				 // Decrement to avoid overflow.
				 InterlockedDecrement(&runtime_init_once_gate);
				 while(!InterlockedExchangeAdd(&runtime_init_once_done, 0)) {
						Sleep(0);
				 }
			}
	 }
}

void
x_cgo_sys_thread_create(unsigned long (__stdcall *func)(void*), void* arg) {
	_cgo_beginthread(func, arg);
}

int
_cgo_is_runtime_initialized() {
	 int status;

	 EnterCriticalSection(&runtime_init_cs);
	 status = runtime_init_done;
	 LeaveCriticalSection(&runtime_init_cs);
	 return status;
}

uintptr_t
_cgo_wait_runtime_init_done(void) {
	void (*pfn)(struct cgoContextArg*);

	 _cgo_maybe_run_preinit();
	while (!_cgo_is_runtime_initialized()) {
			WaitForSingleObject(runtime_init_wait, INFINITE);
	}
	pfn = _cgo_get_context_function();
	if (pfn != nil) {
		struct cgoContextArg arg;

		arg.Context = 0;
		(*pfn)(&arg);
		return arg.Context;
	}
	return 0;
}

// Should not be used since x_cgo_pthread_key_created will always be zero.
void x_cgo_bindm(void* dummy) {
	fprintf(stderr, "unexpected cgo_bindm on Windows\n");
	abort();
}

void
x_cgo_notify_runtime_init_done(void* dummy) {
	 _cgo_maybe_run_preinit();

	 EnterCriticalSection(&runtime_init_cs);
	runtime_init_done = 1;
	 LeaveCriticalSection(&runtime_init_cs);

	 if (!SetEvent(runtime_init_wait)) {
		fprintf(stderr, "runtime: failed to signal runtime initialization complete.\n");
		abort();
	}
}

// The traceback function, used when tracing C calls.
static void (*cgo_traceback_function)(struct cgoTracebackArg*);

// The context function, used when tracing back C calls into Go.
static void (*cgo_context_function)(struct cgoContextArg*);

// The symbolizer function, used when symbolizing C frames.
static void (*cgo_symbolizer_function)(struct cgoSymbolizerArg*);

// Sets the traceback, context, and symbolizer functions. Called from
// runtime.SetCgoTraceback.
void x_cgo_set_traceback_functions(struct cgoSetTracebackFunctionsArg* arg) {
	EnterCriticalSection(&runtime_init_cs);
	cgo_traceback_function = arg->Traceback;
	cgo_context_function = arg->Context;
	cgo_symbolizer_function = arg->Symbolizer;
	LeaveCriticalSection(&runtime_init_cs);
}

// Gets the traceback function to call to trace C calls.
void (*(_cgo_get_traceback_function(void)))(struct cgoTracebackArg*) {
	void (*ret)(struct cgoTracebackArg*);

	EnterCriticalSection(&runtime_init_cs);
	ret = cgo_traceback_function;
	LeaveCriticalSection(&runtime_init_cs);
	return ret;
}

// Call the traceback function registered with x_cgo_set_traceback_functions.
//
// On other platforms, this coordinates with C/C++ TSAN. On Windows, there is
// no C/C++ TSAN.
void x_cgo_call_traceback_function(struct cgoTracebackArg* arg) {
	void (*pfn)(struct cgoTracebackArg*);

	pfn = _cgo_get_traceback_function();
	if (pfn == nil) {
		return;
	}

	(*pfn)(arg);
}

// Gets the context function to call to record the traceback context
// when calling a Go function from C code.
void (*(_cgo_get_context_function(void)))(struct cgoContextArg*) {
	void (*ret)(struct cgoContextArg*);

	EnterCriticalSection(&runtime_init_cs);
	ret = cgo_context_function;
	LeaveCriticalSection(&runtime_init_cs);
	return ret;
}

// Gets the symbolizer function to call to symbolize C frames.
void (*(_cgo_get_symbolizer_function(void)))(struct cgoSymbolizerArg*) {
	void (*ret)(struct cgoSymbolizerArg*);

	EnterCriticalSection(&runtime_init_cs);
	ret = cgo_symbolizer_function;
	LeaveCriticalSection(&runtime_init_cs);
	return ret;
}

// Call the symbolizer function registered with x_cgo_set_symbolizer_functions.
//
// On other platforms, this coordinates with C/C++ TSAN. On Windows, there is
// no C/C++ TSAN.
void x_cgo_call_symbolizer_function(struct cgoSymbolizerArg* arg) {
	void (*pfn)(struct cgoSymbolizerArg*);

	pfn = _cgo_get_symbolizer_function();
	if (pfn == nil) {
		return;
	}

	(*pfn)(arg);
}

void _cgo_beginthread(unsigned long (__stdcall *func)(void*), void* arg) {
	int tries;
	HANDLE thandle;

	for (tries = 0; tries < 20; tries++) {
		thandle = CreateThread(NULL, 0, func, arg, 0, NULL);
		if (thandle == 0 && GetLastError() == ERROR_ACCESS_DENIED) {
			// "Insufficient resources", try again in a bit.
			//
			// Note that the first Sleep(0) is a yield.
			Sleep(tries); // milliseconds
			continue;
		} else if (thandle == 0) {
			break;
		}
		CloseHandle(thandle);
		return; // Success!
	}

	fprintf(stderr, "runtime: failed to create new OS thread (%lu)\n", GetLastError());
	abort();
}
