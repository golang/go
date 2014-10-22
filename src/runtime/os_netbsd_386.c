// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"

void
runtime·lwp_mcontext_init(McontextT *mc, void *stack, M *mp, G *gp, void (*fn)(void))
{
	mc->__gregs[REG_EIP] = (uint32)runtime·lwp_tramp;
	mc->__gregs[REG_UESP] = (uint32)stack;
	mc->__gregs[REG_EBX] = (uint32)mp;
	mc->__gregs[REG_EDX] = (uint32)gp;
	mc->__gregs[REG_ESI] = (uint32)fn;
}
