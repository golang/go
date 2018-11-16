// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Failed to add type conversion for negative constant.
// No runtime test; just make sure it compiles.

package cgotest

/*
#include <complex.h>

static void issue28545F(char **p, int n, complex double a) {}
*/
import "C"

func issue28545G(p **C.char) {
	C.issue28545F(p, -1, (0))
	C.issue28545F(p, 2+3, complex(1, 1))
}
