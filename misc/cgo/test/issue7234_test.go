// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

import "testing"

// This test actually doesn't have anything to do with cgo.  It is a
// test of http://golang.org/issue/7234, a compiler/linker bug in
// handling string constants when using -linkmode=external.  The test
// is in this directory because we routinely test -linkmode=external
// here.

var v7234 = [...]string{"runtime/cgo"}

func Test7234(t *testing.T) {
	if v7234[0] != "runtime/cgo" {
		t.Errorf("bad string constant %q", v7234[0])
	}
}
