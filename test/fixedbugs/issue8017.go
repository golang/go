// compile

// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issues 8017 and 8058: walk modifies nodes generated
// by slicelit and causes an internal error afterwards
// when gen_as_init parses it back.

package main

func F() {
	var ch chan int
	select {
	case <-ch:
	case <-make(chan int, len([2][]int{([][]int{})[len(ch)], []int{}})):
	}
}

func G() {
	select {
	case <-([1][]chan int{[]chan int{}})[0][0]:
	default:
	}
}
