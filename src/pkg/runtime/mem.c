// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs.h"

// Stubs for memory management.
// In a separate file so they can be overridden during testing of gc.

enum
{
	NHUNK		= 20<<20,
};

void
runtime·mal(uint32 n, uint8 *ret)
{
	ret = mal(n);
	FLUSH(&ret);
}
