// compile

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

var i int
var arr []*int
var f func() int

func g() {
	for i, *(arr[f()]) = range []int{} {
	}
}

func h() {
	var x int
	var f func() int
	var arr []int
	var arr2 [][0]rune
	for arr[x], arr2[arr[f()]][x] = range "" {
	}
}
