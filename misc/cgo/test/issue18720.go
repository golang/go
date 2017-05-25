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
*/
import "C"
import "testing"

func test18720(t *testing.T) {
	if C.HELLO_WORLD != "hello\000world" {
		t.Fatalf(`expected "hello\000world", but got %q`, C.HELLO_WORLD)
	}

	// Issue 20125.
	if got, want := C.SIZE_OF_FOO, 1; got != want {
		t.Errorf("C.SIZE_OF_FOO == %v, expected %v", got, want)
	}
}
