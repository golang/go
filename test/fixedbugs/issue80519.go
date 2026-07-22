// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

var aliased = [3][3]int{{9, 9, 9}, {1, 9, 9}, {9, 9, 9}}
var unsafeAliased [3]unsafe.Pointer

type unsafeFieldTarget struct {
	array *[4]uintptr
	pad   [3]uintptr
}

//go:noinline
func clearAliased() {
	keyp := &aliased[1][0]
	for i := range 3 {
		aliased[*keyp][i] = 0
	}
}

//go:noinline
func clearUnsafeAliased() {
	target := (*[3]unsafe.Pointer)(unsafe.Pointer(&unsafeAliased[0]))
	unsafeAliased[0] = unsafe.Pointer(target)
	targetp := (**[3]unsafe.Pointer)(unsafe.Pointer(&unsafeAliased[0]))
	for i := range 3 {
		(**targetp)[i] = nil
	}
}

//go:noinline
func clearUnsafeField(t *unsafeFieldTarget) {
	for i := range t.array {
		t.array[i] = 0
	}
}

//go:noinline
func clearUnsafeSlice() {
	var values []uintptr
	values = unsafe.Slice((*uintptr)(unsafe.Pointer(&values)), 3)
	for i := range values {
		values[i] = 0
	}
}

func main() {
	want := [3][2]int{{0, 2}, {3, 0}, {5, 6}}

	rows := [3][2]int{{1, 2}, {3, 4}, {5, 6}}
	key := 2
	for key = range 2 {
		rows[key][key] = 0
	}
	if rows != want || key != 1 {
		panic("range clear with assigned index variable")
	}

	rows = [3][2]int{{1, 2}, {3, 4}, {5, 6}}
	for key := range 2 {
		rows[key][key] = 0
	}
	if rows != want {
		panic("range clear with declared index variable")
	}

	slices := [3][]int{{1, 2}, {3, 4}, {5, 6}}
	key = 0
	for key = range slices[key] {
		slices[key][key] = 0
	}
	if slices[0][0] != 0 || slices[0][1] != 2 || slices[1][0] != 3 || slices[1][1] != 0 || key != 1 {
		panic("range clear with index-dependent slice")
	}

	rows = [3][2]int{{1, 2}, {3, 4}, {5, 6}}
	key = 0
	keyp := &key
	for key = range 2 {
		rows[*keyp][key] = 0
	}
	if rows != want || key != 1 {
		panic("range clear with indirect index dependency")
	}

	clearAliased()
	if aliased != [3][3]int{{9, 0, 0}, {0, 9, 9}, {9, 9, 9}} {
		panic("range clear target changed through cleared memory")
	}

	panicked := false
	func() {
		defer func() {
			panicked = recover() != nil
		}()
		clearUnsafeAliased()
	}()
	if !panicked {
		panic("range clear target changed through unsafe pointer")
	}

	fieldTarget := unsafeFieldTarget{}
	fieldTarget.array = (*[4]uintptr)(unsafe.Pointer(&fieldTarget))
	panicked = false
	func() {
		defer func() {
			panicked = recover() != nil
		}()
		clearUnsafeField(&fieldTarget)
	}()
	if !panicked {
		panic("range clear target changed through unsafe pointer field")
	}

	panicked = false
	func() {
		defer func() {
			panicked = recover() != nil
		}()
		clearUnsafeSlice()
	}()
	if !panicked {
		panic("range clear target changed through unsafe slice header")
	}
}
