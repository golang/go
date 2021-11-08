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
	shouldPanic("len out of range", func() { _ = make(T, n) })
	shouldPanic("cap out of range", func() { _ = make(T, 0, n) })
	shouldPanic("len out of range", func() { _ = make(T, int64(n)) })
	shouldPanic("cap out of range", func() { _ = make(T, 0, int64(n)) })
	testMakeInAppend(n)

	var t *byte
	if unsafe.Sizeof(t) == 8 {
		// Test mem > maxAlloc
		var n2 int64 = 1 << 59
		shouldPanic("len out of range", func() { _ = make(T, int(n2)) })
		shouldPanic("cap out of range", func() { _ = make(T, 0, int(n2)) })
		testMakeInAppend(int(n2))
		// Test elem.size*cap overflow
		n2 = 1<<63 - 1
		shouldPanic("len out of range", func() { _ = make(T, int(n2)) })
		shouldPanic("cap out of range", func() { _ = make(T, 0, int(n2)) })
		testMakeInAppend(int(n2))
		var x uint64 = 1<<64 - 1
		shouldPanic("len out of range", func() { _ = make([]byte, x) })
		shouldPanic("cap out of range", func() { _ = make(T, 0, x) })
		testMakeInAppend(int(x))
	} else {
		n = 1<<31 - 1
		shouldPanic("len out of range", func() { _ = make(T, n) })
		shouldPanic("cap out of range", func() { _ = make(T, 0, n) })
		shouldPanic("len out of range", func() { _ = make(T, int64(n)) })
		shouldPanic("cap out of range", func() { _ = make(T, 0, int64(n)) })
		testMakeInAppend(n)
		var x uint64 = 1<<32 - 1
		shouldPanic("len out of range", func() { _ = make([]byte, x) })
		shouldPanic("cap out of range", func() { _ = make(T, 0, x) })
		testMakeInAppend(int(x))
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

// Test make in append panics since the gc compiler optimizes makes in appends.
func testMakeInAppend(n int) {
	lengths := []int{0, 1}
	for _, length := range lengths {
		t := make(T, length)
		shouldPanic("len out of range", func() { _ = append(t, make(T, n)...) })
		shouldPanic("cap out of range", func() { _ = append(t, make(T, 0, n)...) })
		shouldPanic("len out of range", func() { _ = append(t, make(T, int64(n))...) })
		shouldPanic("cap out of range", func() { _ = append(t, make(T, 0, int64(n))...) })
		shouldPanic("len out of range", func() { _ = append(t, make(T, uint64(n))...) })
		shouldPanic("cap out of range", func() { _ = append(t, make(T, 0, uint64(n))...) })
		shouldPanic("len out of range", func() { _ = append(t, make(T, int(n))...) })
		shouldPanic("cap out of range", func() { _ = append(t, make(T, 0, int(n))...) })
		shouldPanic("len out of range", func() { _ = append(t, make(T, uint(n))...) })
		shouldPanic("cap out of range", func() { _ = append(t, make(T, 0, uint(n))...) })
	}
}
