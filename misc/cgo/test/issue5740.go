// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

// int test5740a(void), test5740b(void);
import "C"
import "testing"

func test5740(t *testing.T) {
	if v := C.test5740a() + C.test5740b(); v != 5 {
		t.Errorf("expected 5, got %v", v)
	}
}
