// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

// Stubs for stack management.
// In a separate file so they can be overridden during testing of gc.

void*
stackalloc(uint32 n)
{
	return mal(n);
}

void
stackfree(void*)
{
}
