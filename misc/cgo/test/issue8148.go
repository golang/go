// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 8148.  A typedef of an unnamed struct didn't work when used
// with an exported Go function.  No runtime test; just make sure it
// compiles.

package cgotest

/*
typedef struct { int i; } T;

int issue8148Callback(T*);

static int get() {
	T t;
	t.i = 42;
	return issue8148Callback(&t);
}
*/
import "C"

//export issue8148Callback
func issue8148Callback(t *C.T) C.int {
	return t.i
}

func Issue8148() int {
	return int(C.get())
}
