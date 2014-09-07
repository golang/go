// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "os_GOOS.h"
#include "cgocall.h"
#include "textflag.h"

typedef struct HandleErr HandleErr;
typedef struct SyscallErr SyscallErr;

struct HandleErr {
	uintptr handle;
	uintptr err;
};

struct SyscallErr {
	uintptr r1;
	uintptr r2;
	uintptr err;
};

#pragma textflag NOSPLIT
HandleErr
syscall·loadlibrary(uint16 *filename)
{
	LibCall c;
	HandleErr r;

	c.fn = runtime·LoadLibrary;
	c.n = 1;
	c.args = &filename;
	runtime·cgocall_errno(runtime·asmstdcall, &c);
	r.handle = c.r1;
	if(r.handle == 0)
		r.err = c.err;
	else
		r.err = 0;
	return r;
}

#pragma textflag NOSPLIT
HandleErr
syscall·getprocaddress(uintptr handle, int8 *procname)
{
	LibCall c;
	HandleErr r;

	USED(procname);
	c.fn = runtime·GetProcAddress;
	c.n = 2;
	c.args = &handle;
	runtime·cgocall_errno(runtime·asmstdcall, &c);
	r.handle = c.r1;
	if(r.handle == 0)
		r.err = c.err;
	else
		r.err = 0;
	return r;
}

#pragma textflag NOSPLIT
SyscallErr
syscall·Syscall(uintptr fn, uintptr nargs, uintptr a1, uintptr a2, uintptr a3)
{
	LibCall c;

	USED(a2);
	USED(a3);
	c.fn = (void*)fn;
	c.n = nargs;
	c.args = &a1;
	runtime·cgocall_errno(runtime·asmstdcall, &c);
	return (SyscallErr){c.r1, c.r2, c.err};
}

#pragma textflag NOSPLIT
SyscallErr
syscall·Syscall6(uintptr fn, uintptr nargs, uintptr a1, uintptr a2, uintptr a3, uintptr a4, uintptr a5, uintptr a6)
{
	LibCall c;

	USED(a2);
	USED(a3);
	USED(a4);
	USED(a5);
	USED(a6);
	c.fn = (void*)fn;
	c.n = nargs;
	c.args = &a1;
	runtime·cgocall_errno(runtime·asmstdcall, &c);
	return (SyscallErr){c.r1, c.r2, c.err};
}

#pragma textflag NOSPLIT
SyscallErr
syscall·Syscall9(uintptr fn, uintptr nargs, uintptr a1, uintptr a2, uintptr a3, uintptr a4, uintptr a5, uintptr a6, uintptr a7, uintptr a8, uintptr a9)
{
	LibCall c;

	USED(a2);
	USED(a3);
	USED(a4);
	USED(a5);
	USED(a6);
	USED(a7);
	USED(a8);
	USED(a9);
	c.fn = (void*)fn;
	c.n = nargs;
	c.args = &a1;
	runtime·cgocall_errno(runtime·asmstdcall, &c);
	return (SyscallErr){c.r1, c.r2, c.err};
}

#pragma textflag NOSPLIT
SyscallErr
syscall·Syscall12(uintptr fn, uintptr nargs, uintptr a1, uintptr a2, uintptr a3, uintptr a4, uintptr a5, uintptr a6, uintptr a7, uintptr a8, uintptr a9, uintptr a10, uintptr a11, uintptr a12)
{
	LibCall c;

	USED(a2);
	USED(a3);
	USED(a4);
	USED(a5);
	USED(a6);
	USED(a7);
	USED(a8);
	USED(a9);
	USED(a10);
	USED(a11);
	USED(a12);
	c.fn = (void*)fn;
	c.n = nargs;
	c.args = &a1;
	runtime·cgocall_errno(runtime·asmstdcall, &c);
	return (SyscallErr){c.r1, c.r2, c.err};
}

#pragma textflag NOSPLIT
SyscallErr
syscall·Syscall15(uintptr fn, uintptr nargs, uintptr a1, uintptr a2, uintptr a3, uintptr a4, uintptr a5, uintptr a6, uintptr a7, uintptr a8, uintptr a9, uintptr a10, uintptr a11, uintptr a12, uintptr a13, uintptr a14, uintptr a15)
{
	LibCall c;

	USED(a2);
	USED(a3);
	USED(a4);
	USED(a5);
	USED(a6);
	USED(a7);
	USED(a8);
	USED(a9);
	USED(a10);
	USED(a11);
	USED(a12);
	USED(a13);
	USED(a14);
	USED(a15);
	c.fn = (void*)fn;
	c.n = nargs;
	c.args = &a1;
	runtime·cgocall_errno(runtime·asmstdcall, &c);
	return (SyscallErr){c.r1, c.r2, c.err};
}
