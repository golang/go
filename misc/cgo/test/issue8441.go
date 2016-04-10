// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 8368 and 8441.  Recursive struct definitions didn't work.
// No runtime test; just make sure it compiles.

package cgotest

/*
typedef struct one one;
typedef struct two two;
struct one {
	two *x;
};
struct two {
	one *x;
};
*/
import "C"

func issue8368(one *C.struct_one, two *C.struct_two) {
}

func issue8441(one *C.one, two *C.two) {
	issue8441(two.x, one.x)
}
