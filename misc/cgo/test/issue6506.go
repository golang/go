// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

// Test handling of size_t in the face of incorrect clang debug information.
// golang.org/issue/6506.

/*
#include <stdlib.h>
#include <string.h>
*/
import "C"

func test6506() {
	// nothing to run, just make sure this compiles
	var x C.size_t

	C.calloc(x, x)
	C.malloc(x)
	C.realloc(nil, x)
	C.memcpy(nil, nil, x)
	C.memcmp(nil, nil, x)
	C.memmove(nil, nil, x)
	C.strncpy(nil, nil, x)
	C.strncmp(nil, nil, x)
	C.strncat(nil, nil, x)
	x = C.strxfrm(nil, nil, x)
	C.memchr(nil, 0, x)
	x = C.strcspn(nil, nil)
	x = C.strspn(nil, nil)
	C.memset(nil, 0, x)
	x = C.strlen(nil)
	_ = x
}
