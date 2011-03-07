// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <sys/types.h>
#include "_cgo_export.h"

void
callback(void *f)
{
	goCallback(f);
}
