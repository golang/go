// run

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test the cap predeclared function applied to channels.

package main

import (
	"strings"
	"unsafe"
)

type T chan int

const ptrSize = unsafe.Sizeof((*byte)(nil))

func main() {
	c := make(T, 10)
	if len(c) != 0 || cap(c) != 10 {
		println("chan len/cap ", len(c), cap(c), " want 0 10")
		panic("fail")
	}

	for i := 0; i < 3; i++ {
		c <- i
	}
	if len(c) != 3 || cap(c) != 10 {
		println("chan len/cap ", len(c), cap(c), " want 3 10")
		panic("fail")
	}

	c = make(T)
	if len(c) != 0 || cap(c) != 0 {
		println("chan len/cap ", len(c), cap(c), " want 0 0")
		panic("fail")
	}

	n := -1
	shouldPanic("makechan: size out of range", func() { _ = make(T, n) })
	shouldPanic("makechan: size out of range", func() { _ = make(T, int64(n)) })
	if ptrSize == 8 {
		n = 1 << 20
		n <<= 20
		shouldPanic("makechan: size out of range", func() { _ = make(T, n) })
		n <<= 20
		shouldPanic("makechan: size out of range", func() { _ = make(T, n) })
	} else {
		n = 1<<31 - 1
		shouldPanic("makechan: size out of range", func() { _ = make(T, n) })
		shouldPanic("makechan: size out of range", func() { _ = make(T, int64(n)) })
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
