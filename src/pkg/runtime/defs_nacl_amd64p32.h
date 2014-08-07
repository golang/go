// Copyright 2013 The Go Authors.  All rights reserved.
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
typedef struct ExcRegs386 ExcRegs386;
typedef struct ExcRegsAmd64 ExcRegsAmd64;

struct ExcRegs386
{
	uint32	eax;
	uint32	ecx;
	uint32	edx;
	uint32	ebx;
	uint32	esp;
	uint32	ebp;
	uint32	esi;
	uint32	edi;
	uint32	eip;
	uint32	eflags;
};

struct ExcRegsAmd64
{
	uint64	rax;
	uint64	rcx;
	uint64	rdx;
	uint64	rbx;
	uint64	rsp;
	uint64	rbp;
	uint64	rsi;
	uint64	rdi;
	uint64	r8;
	uint64	r9;
	uint64	r10;
	uint64	r11;
	uint64	r12;
	uint64	r13;
	uint64	r14;
	uint64	r15;
	uint64	rip;
	uint32	rflags;
};

struct ExcContext
{
	uint32	size;
	uint32	portable_context_offset;
	uint32	portable_context_size;
	uint32	arch;
	uint32	regs_size;
	uint32	reserved[11];
	union {
		ExcRegs386	regs;
		ExcRegsAmd64	regs64;
	} regs;
};

struct ExcPortableContext
{
	uint32	pc;
	uint32	sp;
	uint32	fp;
};
