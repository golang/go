// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

// int a(void) { return 5; };
// int r(void) { return 3; };
import "C"
import "testing"

func test23356(t *testing.T) {
	if got, want := C.a(), C.int(5); got != want {
		t.Errorf("C.a() == %v, expected %v", got, want)
	}
	if got, want := C.r(), C.int(3); got != want {
		t.Errorf("C.r() == %v, expected %v", got, want)
	}
}
