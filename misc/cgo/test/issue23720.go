// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test that we can pass compatible typedefs.
// No runtime test; just make sure it compiles.

package cgotest

/*
typedef int *issue23720A;

typedef const int *issue23720B;

void issue23720F(issue23720B a) {}
*/
import "C"

func Issue23720F() {
	var x C.issue23720A
	C.issue23720F(x)
}
