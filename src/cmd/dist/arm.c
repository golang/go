// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "a.h"

#ifndef __ARMEL__
char *
xgetgoarm(void)
{
	return "6";
}
#else
static void useVFPv3(void);
static void useVFPv1(void);

char *
xgetgoarm(void)
{
#if defined(__NetBSD__) || defined(__FreeBSD__)
	// NetBSD has buggy support for VFPv2 (incorrect inexact, 
	// denormial, and NaN handling). When GOARM=6, some of our
	// math tests fails on Raspberry Pi.
	// Thus we return "5" here for safety, the user is free
	// to override.
	// Note: using GOARM=6 with cgo can trigger a kernel assertion
	// failure and crash NetBSD/evbarm kernel.
	// FreeBSD also have broken VFP support, so disable VFP also
	// on FreeBSD.
	return "5";
#endif
	if(xtryexecfunc(useVFPv3))
		return "7";
	else if(xtryexecfunc(useVFPv1))
		return "6";
	return "5";
}

static void
useVFPv3(void)
{
	// try to run VFPv3-only "vmov.f64 d0, #112" instruction
	// we can't use that instruction directly, because we
	// might be compiling with a soft-float only toolchain.
	//
	// some newer toolchains are configured to use thumb
	// by default, so we need to do some mode changing magic
	// here.
	// We can use "bx pc; nop" here, but GNU as(1) insists
	// on warning us
	// "use of r15 in bx in ARM mode is not really useful"
	// so we workaround that by using "bx r0"
	__asm__ __volatile__ ("mov r0, pc");
	__asm__ __volatile__ ("bx r0");
	__asm__ __volatile__ (".word 0xeeb70b00"); // vmov.f64 d0, #112
	__asm__ __volatile__ (".word 0xe12fff1e"); // bx lr
}

static void
useVFPv1(void)
{
	// try to run "vmov.f64 d0, d0" instruction
	// we can't use that instruction directly, because we
	// might be compiling with a soft-float only toolchain
	//
	// some newer toolchains are configured to use thumb
	// by default, so we need to do some mode changing magic
	// here.
	// We can use "bx pc; nop" here, but GNU as(1) insists
	// on warning us
	// "use of r15 in bx in ARM mode is not really useful"
	// so we workaround that by using "bx r0"
	__asm__ __volatile__ ("mov r0, pc");
	__asm__ __volatile__ ("bx r0");
	__asm__ __volatile__ (".word 0xeeb00b40"); // vomv.f64 d0, d0
	__asm__ __volatile__ (".word 0xe12fff1e"); // bx lr
}

#endif
