// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// closure.go tests closure operations.
package main

import "testing"

//go:noinline
func testCFunc_ssa() int {
	a := 0
	b := func() {
		switch {
		}
		a++
	}
	b()
	b()
	return a
}

func testCFunc(t *testing.T) {
	if want, got := 2, testCFunc_ssa(); got != want {
		t.Errorf("expected %d, got %d", want, got)
	}
}

// TestClosure tests closure related behavior.
func TestClosure(t *testing.T) {
	testCFunc(t)
}
