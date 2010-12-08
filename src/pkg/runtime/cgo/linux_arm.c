// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "libcgo.h"

static void
xinitcgo(void)
{
}

void (*initcgo)(void) = xinitcgo;

void
libcgo_sys_thread_start(ThreadStart *ts)
{
	// unimplemented
	*(int*)0 = 0;
}
