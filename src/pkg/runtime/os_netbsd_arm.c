// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"
#include "signal_GOOS_GOARCH.h"
#include "textflag.h"

void
runtime·lwp_mcontext_init(McontextT *mc, void *stack, M *mp, G *gp, void (*fn)(void))
{
	mc->__gregs[REG_R15] = (uint32)runtime·lwp_tramp;
	mc->__gregs[REG_R13] = (uint32)stack;
	mc->__gregs[REG_R0] = (uint32)mp;
	mc->__gregs[REG_R1] = (uint32)gp;
	mc->__gregs[REG_R2] = (uint32)fn;
}

void
runtime·checkgoarm(void)
{
	// TODO(minux)
}

#pragma textflag NOSPLIT
int64
runtime·cputicks() {
	// Currently cputicks() is used in blocking profiler and to seed runtime·fastrand1().
	// runtime·nanotime() is a poor approximation of CPU ticks that is enough for the profiler.
	// TODO: need more entropy to better seed fastrand1.
	return runtime·nanotime();
}
