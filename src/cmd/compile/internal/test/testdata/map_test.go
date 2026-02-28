// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// map.go tests map operations.
package main

import "testing"

//go:noinline
func lenMap_ssa(v map[int]int) int {
	return len(v)
}

func testLenMap(t *testing.T) {

	v := make(map[int]int)
	v[0] = 0
	v[1] = 0
	v[2] = 0

	if want, got := 3, lenMap_ssa(v); got != want {
		t.Errorf("expected len(map) = %d, got %d", want, got)
	}
}

func testLenNilMap(t *testing.T) {

	var v map[int]int
	if want, got := 0, lenMap_ssa(v); got != want {
		t.Errorf("expected len(nil) = %d, got %d", want, got)
	}
}
func TestMap(t *testing.T) {
	testLenMap(t)
	testLenNilMap(t)
}
