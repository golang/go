// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "cgocall.h"

Cgo *cgo;	/* filled in by dynamic linker when Cgo is available */

void
cgocall(void (*fn)(void*), void *arg)
{
	CgoWork w;
	CgoServer *s;

	if(cgo == nil)
		throw("cgocall unavailable");

	noteclear(&w.note);
	w.next = nil;
	w.fn = fn;
	w.arg = arg;
	lock(&cgo->lock);
	if((s = cgo->idle) != nil) {
		cgo->idle = s->next;
		s->work = &w;
		unlock(&cgo->lock);
		notewakeup(&s->note);
	} else {
		if(cgo->whead == nil) {
			cgo->whead = &w;
		} else
			cgo->wtail->next = &w;
		cgo->wtail = &w;
		unlock(&cgo->lock);
	}
	notesleep(&w.note);
}
