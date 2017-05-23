// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
#define HELLO "hello"
#define WORLD "world"
#define HELLO_WORLD HELLO "\000" WORLD

struct foo { char c; };
#define SIZE_OF(x) sizeof(x)
#define SIZE_OF_FOO SIZE_OF(struct foo)
#define VAR1 VAR
#define VAR var
int var = 5;

#define ADDR &var

#define CALL fn()
int fn(void) {
	return ++var;
}
*/
import "C"
import "testing"

func test18720(t *testing.T) {
	if got, want := C.HELLO_WORLD, "hello\000world"; got != want {
		t.Errorf("C.HELLO_WORLD == %q, expected %q", got, want)
	}

	if got, want := C.VAR1, C.int(5); got != want {
		t.Errorf("C.VAR1 == %v, expected %v", got, want)
	}

	if got, want := *C.ADDR, C.int(5); got != want {
		t.Errorf("*C.ADDR == %v, expected %v", got, want)
	}

	if got, want := C.CALL, C.int(6); got != want {
		t.Errorf("C.CALL == %v, expected %v", got, want)
	}

	if got, want := C.CALL, C.int(7); got != want {
		t.Errorf("C.CALL == %v, expected %v", got, want)
	}

	// Issue 20125.
	if got, want := C.SIZE_OF_FOO, 1; got != want {
		t.Errorf("C.SIZE_OF_FOO == %v, expected %v", got, want)
	}
}
