// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <assert.h>
#include <stdlib.h>
#include <string.h>
#include "_cgo_export.h"

/* Test calling a Go function with multiple return values.  */

void
callMulti(void)
{
	struct multi_return result = multi();
	assert(strcmp(result.r0, "multi") == 0);
	assert(result.r1 == 0);
	free(result.r0);
}
