// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "arch_GOARCH.h"

void
·GogoBytes(int32 x)
{
	x = RuntimeGogoBytes;
	FLUSH(&x);
}
