// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "type.h"
#include "typekind.h"
#include "defs_GOOS_GOARCH.h"
#include "os_GOOS.h"
#include "zasm_GOOS_GOARCH.h"

typedef	struct	Callbacks	Callbacks;
struct	Callbacks {
	Lock;
	WinCallbackContext*	ctxt[cb_max];
	int32			n;
};

static	Callbacks	cbs;

WinCallbackContext** runtime·cbctxts; // to simplify access to cbs.ctxt in sys_windows_*.s

// Call back from windows dll into go.
byte *
runtime·compilecallback(Eface fn, bool cleanstack)
{
	FuncType *ft;
	Type *t;
	int32 argsize, i, n;
	WinCallbackContext *c;

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

	runtime·lock(&cbs);
	if(runtime·cbctxts == nil)
		runtime·cbctxts = &(cbs.ctxt[0]);
	n = cbs.n;
	for(i=0; i<n; i++) {
		if(cbs.ctxt[i]->gobody == fn.data && cbs.ctxt[i]->cleanstack == cleanstack) {
			runtime·unlock(&cbs);
			// runtime·callbackasm is just a series of CALL instructions
			// (each is 5 bytes long), and we want callback to arrive at
			// correspondent call instruction instead of start of
			// runtime·callbackasm.
			return (byte*)runtime·callbackasm + i * 5;
		}
	}
	if(n >= cb_max)
		runtime·throw("too many callback functions");
	c = runtime·mal(sizeof *c);
	c->gobody = fn.data;
	c->argsize = argsize;
	c->cleanstack = cleanstack;
	if(cleanstack && argsize!=0)
		c->restorestack = argsize;
	else
		c->restorestack = 0;
	cbs.ctxt[n] = c;
	cbs.n++;
	runtime·unlock(&cbs);

	// as before
	return (byte*)runtime·callbackasm + n * 5;
}
