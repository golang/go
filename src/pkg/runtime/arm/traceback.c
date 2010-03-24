// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"

// TODO(rsc): Move this into portable code, with calls to a
// machine-dependent isclosure() function.

void
traceback(byte *pc0, byte *sp, G *g)
{
}

// func caller(n int) (pc uintptr, file string, line int, ok bool)
int32
callers(int32 skip, uintptr *pcbuf, int32 m)
{
	return 0;
}
