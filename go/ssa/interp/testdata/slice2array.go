// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test for slice to array conversion introduced in go1.20
// See: https://tip.golang.org/ref/spec#Conversions_from_slice_to_array_pointer

package main

func main() {
	s := make([]byte, 3, 4)
	s[0], s[1], s[2] = 2, 3, 5
	a := ([2]byte)(s)
	s[0] = 7

	if a != [2]byte{2, 3} {
		panic("converted from non-nil slice to array")
	}

	{
		var s []int
		a := ([0]int)(s)
		if a != [0]int{} {
			panic("zero len array is not equal")
		}
	}

	if emptyToEmptyDoesNotPanic() {
		panic("no panic expected from emptyToEmptyDoesNotPanic()")
	}
	if !threeToFourDoesPanic() {
		panic("panic expected from threeToFourDoesPanic()")
	}

	if !fourPanicsWhileOneDoesNot[[4]int]() {
		panic("panic expected from fourPanicsWhileOneDoesNot[[4]int]()")
	}
	if fourPanicsWhileOneDoesNot[[1]int]() {
		panic("no panic expected from fourPanicsWhileOneDoesNot[[1]int]()")
	}

	if !fourPanicsWhileZeroDoesNot[[4]int]() {
		panic("panic expected from fourPanicsWhileZeroDoesNot[[4]int]()")
	}
	if fourPanicsWhileZeroDoesNot[[0]int]() {
		panic("no panic expected from fourPanicsWhileZeroDoesNot[[0]int]()")
	}
}

func emptyToEmptyDoesNotPanic() (raised bool) {
	defer func() {
		if e := recover(); e != nil {
			raised = true
		}
	}()
	var s []int
	_ = ([0]int)(s)
	return false
}

func threeToFourDoesPanic() (raised bool) {
	defer func() {
		if e := recover(); e != nil {
			raised = true
		}
	}()
	s := make([]int, 3, 5)
	_ = ([4]int)(s)
	return false
}

func fourPanicsWhileOneDoesNot[T [1]int | [4]int]() (raised bool) {
	defer func() {
		if e := recover(); e != nil {
			raised = true
		}
	}()
	s := make([]int, 3, 5)
	_ = T(s)
	return false
}

func fourPanicsWhileZeroDoesNot[T [0]int | [4]int]() (raised bool) {
	defer func() {
		if e := recover(); e != nil {
			raised = true
		}
	}()
	var s []int
	_ = T(s)
	return false
}
