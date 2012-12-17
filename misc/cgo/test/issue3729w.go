// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 3729:	cmd/cgo: access errno from void C function
// void f(void) returns [0]byte, error in Go world.

// +build windows

package cgotest

import "testing"

func test3729(t *testing.T) {
	t.Log("skip errno test on Windows")
}
