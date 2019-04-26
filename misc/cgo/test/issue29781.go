// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Error with newline inserted into constant expression.
// Compilation test only, nothing to run.

package cgotest

// static void issue29781F(char **p, int n) {}
// #define ISSUE29781C 0
import "C"

func issue29781G() {
	var p *C.char
	C.issue29781F(&p, C.ISSUE29781C+1)
}
