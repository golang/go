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
runtime·compilecallback(Eface fn, bool cleanstack)
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
	n = 1+4;		// MOVL fn, AX
	n += 1+4;		// MOVL argsize, DX
	n += 1+4;		// MOVL callbackasm, CX
	n += 2;			// CALL CX
	n += 1;			// RET
	if(cleanstack && argsize!=0)
		n += 2;		// ... argsize

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

	// MOVL fn, AX
	*p++ = 0xb8;
	*(uint32*)p = (uint32)(fn.data);
	p += 4;

	// MOVL argsize, DX
	*p++ = 0xba;
	*(uint32*)p = argsize;
	p += 4;

	// MOVL callbackasm, CX
	*p++ = 0xb9;
	*(uint32*)p = (uint32)runtime·callbackasm;
	p += 4;

	// CALL CX
	*p++ = 0xff;
	*p++ = 0xd1;

	// RET argsize?
	if(cleanstack && argsize!=0) {
		*p++ = 0xc2;
		*(uint16*)p = argsize;
	} else
		*p = 0xc3;

	return &c->asmbody;
}
