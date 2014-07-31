// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"

void
runtime·printslice_m(G *gp)
{
	void *array;
	uintptr len, cap;

	array = g->m->ptrarg[0];
	g->m->ptrarg[0] = nil;
	len = g->m->scalararg[0];
	cap = g->m->scalararg[1];

	runtime·prints("[");
	runtime·printint(len);
	runtime·prints("/");
	runtime·printint(cap);
	runtime·prints("]");
	runtime·printpointer(array);

	runtime·gogo(&gp->sched);
}
