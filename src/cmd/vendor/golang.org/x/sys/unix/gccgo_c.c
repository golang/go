// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build gccgo

#include <errno.h>
#include <stdint.h>
#include <unistd.h>

#define _STRINGIFY2_(x) #x
#define _STRINGIFY_(x) _STRINGIFY2_(x)
#define GOSYM_PREFIX _STRINGIFY_(__USER_LABEL_PREFIX__)

// Call syscall from C code because the gccgo support for calling from
// Go to C does not support varargs functions.

struct ret {
	uintptr_t r;
	uintptr_t err;
};

struct ret
gccgoRealSyscall(uintptr_t trap, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5, uintptr_t a6, uintptr_t a7, uintptr_t a8, uintptr_t a9)
{
	struct ret r;

	errno = 0;
	r.r = syscall(trap, a1, a2, a3, a4, a5, a6, a7, a8, a9);
	r.err = errno;
	return r;
}

uintptr_t
gccgoRealSyscallNoError(uintptr_t trap, uintptr_t a1, uintptr_t a2, uintptr_t a3, uintptr_t a4, uintptr_t a5, uintptr_t a6, uintptr_t a7, uintptr_t a8, uintptr_t a9)
{
	return syscall(trap, a1, a2, a3, a4, a5, a6, a7, a8, a9);
}
