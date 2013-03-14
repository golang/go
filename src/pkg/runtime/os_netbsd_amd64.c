// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"

void
runtime·lwp_mcontext_init(McontextT *mc, void *stack, M *mp, G *gp, void (*fn)(void))
{
	// Machine dependent mcontext initialisation for LWP.
	mc->__gregs[REG_RIP] = (uint64)runtime·lwp_tramp;
	mc->__gregs[REG_RSP] = (uint64)stack;
	mc->__gregs[REG_R8] = (uint64)mp;
	mc->__gregs[REG_R9] = (uint64)gp;
	mc->__gregs[REG_R12] = (uint64)fn;
}
