// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "type.h"
#include "typekind.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"

// Will keep all callbacks in a linked list, so they don't get garbage collected.
typedef	struct	Callback	Callback;
struct	Callback {
	Callback*	link;
	void*		gobody;
	byte		asmbody;
};

typedef	struct	Callbacks	Callbacks;
struct	Callbacks {
	Lock;
	Callback*	link;
	int32		n;
};

static	Callbacks	cbs;

// Call back from windows dll into go.
byte *
runtime·compilecallback(Eface fn, bool /*cleanstack*/)
{
	FuncType *ft;
	Type *t;
	int32 argsize, i, n;
	byte *p;
	Callback *c;

	if(fn.type == nil || fn.type->kind != KindFunc)
		runtime·panicstring("compilecallback: not a function");
	ft = (FuncType*)fn.type;
	if(ft->out.len != 1)
		runtime·panicstring("compilecallback: function must have one output parameter");
	if(((Type**)ft->out.array)[0]->size != sizeof(uintptr))
		runtime·panicstring("compilecallback: output parameter size is wrong");
	argsize = 0;
	for(i=0; i<ft->in.len; i++) {
		t = ((Type**)ft->in.array)[i];
		if(t->size > sizeof(uintptr))
			runtime·panicstring("compilecallback: input parameter size is wrong");
		argsize += sizeof(uintptr);
	}

	// compute size of new fn.
	// must match code laid out below.
	n  = 2+8+1; // MOVQ fn, AX           / PUSHQ AX
	n += 2+8+1; // MOVQ argsize, AX      / PUSHQ AX
	n += 2+8;   // MOVQ callbackasm, AX
	n += 2;     // JMP  AX

	runtime·lock(&cbs);
	for(c = cbs.link; c != nil; c = c->link) {
		if(c->gobody == fn.data) {
			runtime·unlock(&cbs);
			return &c->asmbody;
		}
	}
	if(cbs.n >= 2000)
		runtime·throw("too many callback functions");
	c = runtime·mal(sizeof *c + n);
	c->gobody = fn.data;
	c->link = cbs.link;
	cbs.link = c;
	cbs.n++;
	runtime·unlock(&cbs);

	p = &c->asmbody;

	// MOVQ fn, AX
	*p++ = 0x48;
	*p++ = 0xb8;
	*(uint64*)p = (uint64)(fn.data);
	p += 8;
	// PUSH AX
	*p++ = 0x50;

	// MOVQ argsize, AX
	*p++ = 0x48;
	*p++ = 0xb8;
	*(uint64*)p = argsize;
	p += 8;
	// PUSH AX
	*p++ = 0x50;

	// MOVQ callbackasm, AX
	*p++ = 0x48;
	*p++ = 0xb8;
	*(uint64*)p = (uint64)runtime·callbackasm;
	p += 8;

	// JMP AX
	*p++ = 0xFF;
	*p = 0xE0;

	return &c->asmbody;
}
