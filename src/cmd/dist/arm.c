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
	// might be compiling with a soft-float only toolchain
	__asm__ __volatile__ (".word 0xeeb70b00");
}

static void
useVFPv1(void)
{
	// try to run "vmov.f64 d0, d0" instruction
	// we can't use that instruction directly, because we
	// might be compiling with a soft-float only toolchain
	__asm__ __volatile__ (".word 0xeeb00b40");
}

#endif
