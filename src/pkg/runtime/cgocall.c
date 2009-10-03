// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "cgocall.h"

void *initcgo;	/* filled in by dynamic linker when Cgo is available */
int64 ncgocall;
void sys·entersyscall(void);
void sys·exitsyscall(void);

void
cgocall(void (*fn)(void*), void *arg)
{
	if(initcgo == nil)
		throw("cgocall unavailable");

	ncgocall++;

	/*
	 * Announce we are entering a system call
	 * so that the scheduler knows to create another
	 * M to run goroutines while we are in the
	 * foreign code.
	 */
	sys·entersyscall();
	g->cgofn = fn;
	g->cgoarg = arg;
	g->status = Gcgocall;
	gosched();
	sys·exitsyscall();
	return;
}

void
runtime·Cgocalls(int64 ret)
{
	ret = ncgocall;
	FLUSH(&ret);
}

void (*_cgo_malloc)(void*);
void (*_cgo_free)(void*);

void*
cmalloc(uintptr n)
{
	struct a {
		uint64 n;
		void *ret;
	} a;

	a.n = n;
	a.ret = nil;
	cgocall(_cgo_malloc, &a);
	return a.ret;
}

void
cfree(void *p)
{
	cgocall(_cgo_free, p);
}

