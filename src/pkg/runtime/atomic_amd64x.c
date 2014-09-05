// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64 amd64p32

#include "runtime.h"
#include "textflag.h"

#pragma textflag NOSPLIT
uint32
runtime·atomicload(uint32 volatile* addr)
{
	return *addr;
}

#pragma textflag NOSPLIT
uint64
runtime·atomicload64(uint64 volatile* addr)
{
	return *addr;
}

#pragma textflag NOSPLIT
void*
runtime·atomicloadp(void* volatile* addr)
{
	return *addr;
}
