// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

/*
int issue20129 = 0;
typedef void issue20129Void;
issue20129Void issue20129Foo() {
	issue20129 = 1;
}
typedef issue20129Void issue20129Void2;
issue20129Void2 issue20129Bar() {
	issue20129 = 2;
}
*/
import "C"
import "testing"

func test20129(t *testing.T) {
	if C.issue20129 != 0 {
		t.Fatal("test is broken")
	}
	C.issue20129Foo()
	if C.issue20129 != 1 {
		t.Errorf("got %v but expected %v", C.issue20129, 1)
	}
	C.issue20129Bar()
	if C.issue20129 != 2 {
		t.Errorf("got %v but expected %v", C.issue20129, 2)
	}
}
