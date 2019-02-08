// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Failed to add type conversion for negative constant.
// Issue 28772: Failed to add type conversion for Go constant set to C constant.
// No runtime test; just make sure it compiles.

package cgotest

/*
#include <complex.h>

#define issue28772Constant 1

static void issue28545F(char **p, int n, complex double a) {}
*/
import "C"

const issue28772Constant = C.issue28772Constant

func issue28545G(p **C.char) {
	C.issue28545F(p, -1, (0))
	C.issue28545F(p, 2+3, complex(1, 1))
	C.issue28545F(p, issue28772Constant, issue28772Constant2)
}
