// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

// Test handling of size_t in the face of incorrect clang debug information.
// golang.org/issue/6506.

/*
#include <stdlib.h>
#include <string.h>

// These functions are clang builtins but not standard on other systems.
// Give them prototypes so that this test can be compiled on other systems.
// One of the great things about this bug is that even with these prototypes
// clang still generates the wrong debug information.

void *alloca(size_t);
void bzero(void*, size_t);
int bcmp(const void*, const void*, size_t);
int strncasecmp(const char*, const char*, size_t n);
size_t strlcpy(char*, const char*, size_t);
size_t strlcat(char*, const char*, size_t);
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
	C.alloca(x)
	C.bzero(nil, x)
	C.strncasecmp(nil, nil, x)
	x = C.strlcpy(nil, nil, x)
	x = C.strlcat(nil, nil, x)
	_ = x
}
