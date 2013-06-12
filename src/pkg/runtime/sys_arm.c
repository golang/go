// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

// adjust Gobuf as if it executed a call to fn with context ctxt
// and then did an immediate Gosave.
void
runtime·gostartcall(Gobuf *gobuf, void (*fn)(void), void *ctxt)
{
	if(gobuf->lr != 0)
		runtime·throw("invalid use of gostartcall");
	gobuf->lr = gobuf->pc;
	gobuf->pc = (uintptr)fn;
	gobuf->ctxt = ctxt;
}
