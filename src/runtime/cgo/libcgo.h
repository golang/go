// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

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
 * Creates the new operating system thread (OS, arch dependent).
 */
void _cgo_sys_thread_start(ThreadStart *ts);

/*
 * Waits for the Go runtime to be initialized (OS dependent).
 */
void _cgo_wait_runtime_init_done();

/*
 * Call fn in the 6c world.
 */
void crosscall_amd64(void (*fn)(void));

/*
 * Call fn in the 8c world.
 */
void crosscall_386(void (*fn)(void));

/*
 * Prints error then calls abort. For linux and android.
 */
void fatalf(const char* format, ...);

/*
 * Registers the current mach thread port for EXC_BAD_ACCESS processing.
 */
void darwin_arm_init_thread_exception_port(void);

/*
 * Starts a mach message server processing EXC_BAD_ACCESS.
 */
void darwin_arm_init_mach_exception_handler(void);
