// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
#include <stdlib.h>
#include <string.h>

char* Issue6907CopyString(_GoString_ s) {
	size_t n;
	const char *p;
	char *r;

	n = _GoStringLen(s);
	p = _GoStringPtr(s);
	r = malloc(n + 1);
	memmove(r, p, n);
	r[n] = '\0';
	return r;
}
*/
import "C"

import "testing"

func test6907(t *testing.T) {
	want := "yarn"
	if got := C.GoString(C.Issue6907CopyString(want)); got != want {
		t.Errorf("C.GoString(C.Issue6907CopyString(%q)) == %q, want %q", want, got, want)
	}
}
