// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Created by hand, not machine generated.

enum
{
	// These values are referred to in the source code
	// but really don't matter. Even so, use the standard numbers.
	SIGSEGV = 11,
	SIGPROF = 27,
};

typedef struct Siginfo Siginfo;

// native_client/src/trusted/service_runtime/include/machine/_types.h
typedef struct Timespec Timespec;

struct Timespec
{
	int64 tv_sec;
	int32 tv_nsec;
};

// native_client/src/trusted/service_runtime/nacl_exception.h
// native_client/src/include/nacl/nacl_exception.h

typedef struct ExcContext ExcContext;
typedef struct ExcPortable ExcPortable;
typedef struct ExcRegsARM ExcRegsARM;

struct ExcRegsARM
{
	uint32	r0;
	uint32	r1;
	uint32	r2;
	uint32	r3;
	uint32	r4;
	uint32	r5;
	uint32	r6;
	uint32	r7;
	uint32	r8;
	uint32	r9;	// the value reported here is undefined.
	uint32	r10;
	uint32	r11;
	uint32	r12;
	uint32	sp;	/* r13 */
	uint32	lr;	/* r14 */
	uint32	pc;	/* r15 */
	uint32	cpsr;
};

struct ExcContext
{
	uint32	size;
	uint32	portable_context_offset;
	uint32	portable_context_size;
	uint32	arch;
	uint32	regs_size;
	uint32	reserved[11];
	ExcRegsARM	regs;
};

struct ExcPortableContext
{
	uint32	pc;
	uint32	sp;
	uint32	fp;
};
