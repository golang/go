// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "../runtime.h"
#include "../cgocall.h"

// These utility functions are available to be called from code
// compiled with gcc via crosscall2.

// The declaration of crosscall2 is:
//   void crosscall2(void (*fn)(void *, int), void *, int);
// 
// We need to export the symbol crosscall2 in order to support
// callbacks from shared libraries. This applies regardless of
// linking mode.
#pragma cgo_export_static crosscall2
#pragma cgo_export_dynamic crosscall2

// Allocate memory.  This allocates the requested number of bytes in
// memory controlled by the Go runtime.  The allocated memory will be
// zeroed.  You are responsible for ensuring that the Go garbage
// collector can see a pointer to the allocated memory for as long as
// it is valid, e.g., by storing a pointer in a local variable in your
// C function, or in memory allocated by the Go runtime.  If the only
// pointers are in a C global variable or in memory allocated via
// malloc, then the Go garbage collector may collect the memory.

// Call like this in code compiled with gcc:
//   struct { size_t len; void *ret; } a;
//   a.len = /* number of bytes to allocate */;
//   crosscall2(_cgo_allocate, &a, sizeof a);
//   /* Here a.ret is a pointer to the allocated memory.  */

static void
_cgo_allocate_internal(uintptr len, byte *ret)
{
	CgoMal *c;

	ret = runtime·mal(len);
	c = runtime·mal(sizeof(*c));
	c->next = m->cgomal;
	c->alloc = ret;
	m->cgomal = c;
	FLUSH(&ret);
}

#pragma cgo_export_static _cgo_allocate
#pragma cgo_export_dynamic _cgo_allocate
void
_cgo_allocate(void *a, int32 n)
{
	runtime·cgocallback((void(*)(void))_cgo_allocate_internal, a, n);
}

// Panic.  The argument is converted into a Go string.

// Call like this in code compiled with gcc:
//   struct { const char *p; } a;
//   a.p = /* string to pass to panic */;
//   crosscall2(_cgo_panic, &a, sizeof a);
//   /* The function call will not return.  */

extern void ·cgoStringToEface(String, Eface*);

static void
_cgo_panic_internal(byte *p)
{
	String s;
	Eface err;

	s = runtime·gostring(p);
	·cgoStringToEface(s, &err);
	runtime·panic(err);
}

#pragma cgo_export_static _cgo_panic
#pragma cgo_export_dynamic _cgo_panic
void
_cgo_panic(void *a, int32 n)
{
	runtime·cgocallback((void(*)(void))_cgo_panic_internal, a, n);
}

#pragma cgo_import_static x_cgo_init
extern void x_cgo_init(G*);
void (*_cgo_init)(G*) = x_cgo_init;

#pragma cgo_import_static x_cgo_malloc
extern void x_cgo_malloc(void*);
void (*_cgo_malloc)(void*) = x_cgo_malloc;

#pragma cgo_import_static x_cgo_free
extern void x_cgo_free(void*);
void (*_cgo_free)(void*) = x_cgo_free;

#pragma cgo_import_static x_cgo_thread_start
extern void x_cgo_thread_start(void*);
void (*_cgo_thread_start)(void*) = x_cgo_thread_start;
