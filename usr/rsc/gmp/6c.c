// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "cgocall.h"

typedef struct Int Int;
struct Int
{
	void *v;
};

// turn on ffi
#pragma dynld initcgo initcgo "libcgo.so"
#pragma dynld cgo cgo "libcgo.so"

// pull in gmp routines, implemented in gcc.c, from gmp.so

#pragma dynld gmp_addInt gmp_addInt "gmp.so"
void (*gmp_addInt)(void*);

#pragma dynld gmp_stringInt gmp_stringInt "gmp.so"
void (*gmp_stringInt)(void*);

#pragma dynld gmp_newInt gmp_newInt "gmp.so"
void (*gmp_newInt)(void*);

#pragma dynld c_free free "gmp.so"
void (*c_free)(void*);
void
gmp·addInt(Int *z, Int *x, Int *y, Int *ret)
{
	cgocall(gmp_addInt, &z);
}

void
gmp·stringInt(Int *z, String ret)
{
	struct {
		Int *z;
		byte *p;
	} a;
	a.z = z;
	a.p = nil;
	cgocall(gmp_stringInt, &a);
	ret = gostring(a.p);
	cgocall(c_free, a.p);
	FLUSH(&ret);
}

void
gmp·NewInt(uint64 x, Int *z)
{
if(sizeof(uintptr) != 8) *(int32*)0 = 0;
	z = mallocgc(sizeof *z);
	FLUSH(&z);
	cgocall(gmp_newInt, &x);
}

