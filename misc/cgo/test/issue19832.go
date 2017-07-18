// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 19832. Functions taking a pointer typedef were being expanded and triggering a compiler error.

package cgotest

// typedef struct { int i; } *PS;
// void T19832(PS p) {}
import "C"
import "testing"

func test19832(t *testing.T) {
	C.T19832(nil)
}
