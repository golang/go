// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#undef nil
#define nil ((void*)0)
#define nelem(x) (sizeof(x)/sizeof((x)[0]))

typedef uint32_t uint32;
typedef uint64_t uint64;
typedef uintptr_t uintptr;

/*
 * The beginning of the per-goroutine structure,
 * as defined in ../pkg/runtime/runtime.h.
 * Just enough to edit these two fields.
 */
typedef struct G G;
struct G
{
	uintptr stacklo;
	uintptr stackhi;
};

/*
 * Arguments to the _cgo_thread_start call.
 * Also known to ../pkg/runtime/runtime.h.
 */
typedef struct ThreadStart ThreadStart;
struct ThreadStart
{
	G *g;
	uintptr *tls;
	void (*fn)(void);
};

/*
 * Called by 5c/6c/8c world.
 * Makes a local copy of the ThreadStart and
 * calls _cgo_sys_thread_start(ts).
 */
extern void (*_cgo_thread_start)(ThreadStart *ts);

/*
 * Creates a new operating system thread without updating any Go state
 * (OS dependent).
 */
extern void (*_cgo_sys_thread_create)(void* (*func)(void*), void* arg);

/*
 * Indicates whether a dummy pthread per-thread variable is allocated.
 */
extern uintptr_t *_cgo_pthread_key_created;

/*
 * Creates the new operating system thread (OS, arch dependent).
 */
void _cgo_sys_thread_start(ThreadStart *ts);

/*
 * Waits for the Go runtime to be initialized (OS dependent).
 * If runtime.SetCgoTraceback is used to set a context function,
 * calls the context function and returns the context value.
 */
uintptr_t _cgo_wait_runtime_init_done(void);

/*
 * Get the low and high boundaries of the stack.
 */
void x_cgo_getstackbound(uintptr bounds[2]);

/*
 * Prints error then calls abort. For linux and android.
 */
void fatalf(const char* format, ...) __attribute__ ((noreturn));

/*
 * Registers the current mach thread port for EXC_BAD_ACCESS processing.
 */
void darwin_arm_init_thread_exception_port(void);

/*
 * Starts a mach message server processing EXC_BAD_ACCESS.
 */
void darwin_arm_init_mach_exception_handler(void);

/*
 * The cgo traceback callback. See runtime.SetCgoTraceback.
 */
struct cgoTracebackArg {
	uintptr_t  Context;
	uintptr_t  SigContext;
	uintptr_t* Buf;
	uintptr_t  Max;
};
extern void (*(_cgo_get_traceback_function(void)))(struct cgoTracebackArg*);

/*
 * The cgo context callback. See runtime.SetCgoTraceback.
 */
struct cgoContextArg {
	uintptr_t Context;
};
extern void (*(_cgo_get_context_function(void)))(struct cgoContextArg*);

/*
 * The argument for the cgo symbolizer callback. See runtime.SetCgoTraceback.
 */
struct cgoSymbolizerArg {
	uintptr_t   PC;
	const char* File;
	uintptr_t   Lineno;
	const char* Func;
	uintptr_t   Entry;
	uintptr_t   More;
	uintptr_t   Data;
};
extern void (*(_cgo_get_symbolizer_function(void)))(struct cgoSymbolizerArg*);

/*
 * The argument for x_cgo_set_traceback_functions. See runtime.SetCgoTraceback.
 */
struct cgoSetTracebackFunctionsArg {
	void (*Traceback)(struct cgoTracebackArg*);
	void (*Context)(struct cgoContextArg*);
	void (*Symbolizer)(struct cgoSymbolizerArg*);
};

/*
 * TSAN support.  This is only useful when building with
 *   CGO_CFLAGS="-fsanitize=thread" CGO_LDFLAGS="-fsanitize=thread" go install
 */
#undef CGO_TSAN
#if defined(__has_feature)
# if __has_feature(thread_sanitizer)
#  define CGO_TSAN
# endif
#elif defined(__SANITIZE_THREAD__)
# define CGO_TSAN
#endif

#ifdef CGO_TSAN

// _cgo_tsan_acquire tells C/C++ TSAN that we are acquiring a dummy lock. We
// call this when calling from Go to C. This is necessary because TSAN cannot
// see the synchronization in Go. Note that C/C++ code built with TSAN is not
// the same as the Go race detector.
//
// cmd/cgo generates calls to _cgo_tsan_acquire and _cgo_tsan_release. For
// other cgo calls, manual calls are required.
//
// These must match the definitions in yesTsanProlog in cmd/cgo/out.go.
// In general we should call _cgo_tsan_acquire when we enter C code,
// and call _cgo_tsan_release when we return to Go code.
//
// This is only necessary when calling code that might be instrumented
// by TSAN, which mostly means system library calls that TSAN intercepts.
//
// See the comment in cmd/cgo/out.go for more details.

long long _cgo_sync __attribute__ ((common));

extern void __tsan_acquire(void*);
extern void __tsan_release(void*);

__attribute__ ((unused))
static void _cgo_tsan_acquire() {
	__tsan_acquire(&_cgo_sync);
}

__attribute__ ((unused))
static void _cgo_tsan_release() {
	__tsan_release(&_cgo_sync);
}

#else // !defined(CGO_TSAN)

#define _cgo_tsan_acquire()
#define _cgo_tsan_release()

#endif // !defined(CGO_TSAN)
