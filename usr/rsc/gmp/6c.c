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

#pragma dynld c_free free "gmp.so"
void (*c_free)(void*);

// pull in gmp routines, implemented in gcc.c, from gmp.so
#pragma dynld gmp_addInt gmp_addInt "gmp.so"
#pragma dynld gmp_stringInt gmp_stringInt "gmp.so"
#pragma dynld gmp_newInt gmp_newInt "gmp.so"
#pragma dynld gmp_subInt gmp_subInt "gmp.so"
#pragma dynld gmp_mulInt gmp_mulInt "gmp.so"
#pragma dynld gmp_divInt gmp_divInt "gmp.so"
#pragma dynld gmp_modInt gmp_modInt "gmp.so"
#pragma dynld gmp_expInt gmp_expInt "gmp.so"
#pragma dynld gmp_gcdInt gmp_gcdInt "gmp.so"
#pragma dynld gmp_negInt gmp_negInt "gmp.so"
#pragma dynld gmp_absInt gmp_absInt "gmp.so"
#pragma dynld gmp_cmpInt gmp_cmpInt "gmp.so"
#pragma dynld gmp_stringInt gmp_stringInt "gmp.so"
#pragma dynld gmp_probablyPrimeInt gmp_probablyPrimeInt "gmp.so"
#pragma dynld gmp_lshInt gmp_lshInt "gmp.so"
#pragma dynld gmp_rshInt gmp_rshInt "gmp.so"
#pragma dynld gmp_lenInt gmp_lenInt "gmp.so"
#pragma dynld gmp_setInt gmp_setInt "gmp.so"
#pragma dynld gmp_setBytesInt gmp_setBytesInt "gmp.so"
#pragma dynld gmp_setStringInt gmp_setStringInt "gmp.so"
#pragma dynld gmp_bytesInt gmp_bytesInt "gmp.so"
#pragma dynld gmp_divModInt gmp_divModInt "gmp.so"
#pragma dynld gmp_setInt64Int gmp_setInt64Int "gmp.so"
#pragma dynld gmp_int64Int gmp_int64Int "gmp.so"

void (*gmp_addInt)(void*);
void (*gmp_stringInt)(void*);
void (*gmp_newInt)(void*);
void (*gmp_subInt)(void*);
void (*gmp_mulInt)(void*);
void (*gmp_divInt)(void*);
void (*gmp_modInt)(void*);
void (*gmp_expInt)(void*);
void (*gmp_gcdInt)(void*);
void (*gmp_negInt)(void*);
void (*gmp_absInt)(void*);
void (*gmp_cmpInt)(void*);
void (*gmp_stringInt)(void*);
void (*gmp_probablyPrimeInt)(void*);
void (*gmp_lshInt)(void*);
void (*gmp_rshInt)(void*);
void (*gmp_lenInt)(void*);
void (*gmp_setInt)(void*);
void (*gmp_setBytesInt)(void*);
void (*gmp_setStringInt)(void*);
void (*gmp_bytesInt)(void*);
void (*gmp_divModInt)(void*);
void (*gmp_setInt64Int)(void*);
void (*gmp_int64Int)(void*);

void gmp·addInt(Int *z, Int *x, Int *y, Int *ret) { cgocall(gmp_addInt, &z); }
void gmp·subInt(Int *z, Int *x, Int *y, Int *ret) { cgocall(gmp_subInt, &z); }
void gmp·mulInt(Int *z, Int *x, Int *y, Int *ret) { cgocall(gmp_mulInt, &z); }
void gmp·divInt(Int *z, Int *x, Int *y, Int *ret) { cgocall(gmp_divInt, &z); }
void gmp·modInt(Int *z, Int *x, Int *y, Int *ret) { cgocall(gmp_modInt, &z); }
void gmp·expInt(Int *z, Int *x, Int *y, Int *m, Int *ret) { cgocall(gmp_expInt, &z); }
void gmp·GcdInt(Int *d, Int *x, Int *y, Int *a, Int *b) { cgocall(gmp_gcdInt, &d); }
void gmp·negInt(Int *z, Int *x, Int *ret) { cgocall(gmp_negInt, &z); }
void gmp·absInt(Int *z, Int *x, Int *ret) { cgocall(gmp_absInt, &z); }
void gmp·CmpInt(Int *x, Int *y, int32 ret) { cgocall(gmp_cmpInt, &x); }
void gmp·probablyPrimeInt(Int *z, int32 nreps, int32 pad, int32 ret) { cgocall(gmp_probablyPrimeInt, &z); }
void gmp·lshInt(Int *z, Int *x, uint32 s, Int *ret) { cgocall(gmp_lshInt, &z); }
void gmp·rshInt(Int *z, Int *x, uint32 s, Int *ret) { cgocall(gmp_rshInt, &z); }
void gmp·lenInt(Int *z, int32 ret) { cgocall(gmp_lenInt, &z); }
void gmp·setInt(Int *z, Int *x, Int *ret) { cgocall(gmp_setInt, &z); }
void gmp·setBytesInt(Int *z, Array b, Int *ret) { cgocall(gmp_setBytesInt, &z); }
void gmp·setStringInt(Int *z, String s, int32 base, int32 ret) { cgocall(gmp_setStringInt, &z); }
void gmp·bytesInt(Int *z, Array ret) { cgocall(gmp_bytesInt, &z); }
void gmp·DivModInt(Int *q, Int *r, Int *x, Int *y) { cgocall(gmp_divModInt, &q); }
void gmp·setInt64Int(Int *z, int64 x, Int *ret) { cgocall(gmp_setInt64Int, &z); }
void gmp·int64Int(Int *z, int64 ret) { cgocall(gmp_int64Int, &z); }

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

