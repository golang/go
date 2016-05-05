// run

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"strings"
	"unsafe"
)

type T []int

func main() {
	n := -1
	shouldPanic("len out of range", func() {_ = make(T, n)})
	shouldPanic("cap out of range", func() {_ = make(T, 0, n)})
	var t *byte
	if unsafe.Sizeof(t) == 8 {
		n = 1<<20
		n <<= 20
		shouldPanic("len out of range", func() {_ = make(T, n)})
		shouldPanic("cap out of range", func() {_ = make(T, 0, n)})
		n <<= 20
		shouldPanic("len out of range", func() {_ = make(T, n)})
		shouldPanic("cap out of range", func() {_ = make(T, 0, n)})
	} else {
		n = 1<<31 - 1
		shouldPanic("len out of range", func() {_ = make(T, n)})
		shouldPanic("cap out of range", func() {_ = make(T, 0, n)})
	}
}

func shouldPanic(str string, f func()) {
	defer func() {
		err := recover()
		if err == nil {
			panic("did not panic")
		}
		s := err.(error).Error()
		if !strings.Contains(s, str) {
			panic("got panic " + s + ", want " + str)
		}
	}()
	
	f()
}
