// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "cgocall.h"

// turn on ffi
#pragma dynld initcgo initcgo "libcgo.so"
#pragma dynld cgo cgo "libcgo.so"

// pull in fib from fib.so
#pragma dynld extern_c_fib fib "fib.so"
void (*extern_c_fib)(void*);

void
fib·Fib(int32 n, int32, int32)
{
	cgocall(extern_c_fib, &n);
}
