// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

void
runtime·sigenable_m(void)
{
	uint32 s;
	
	s = g->m->scalararg[0];
	g->m->scalararg[0] = 0;
	runtime·sigenable(s);
}

void
runtime·sigdisable_m(void)
{
	uint32 s;
	
	s = g->m->scalararg[0];
	g->m->scalararg[0] = 0;
	runtime·sigdisable(s);
}
