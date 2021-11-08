// run

// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"strings"
	"unsafe"
)

func main() {
	n := -1
	testInts(uint64(n))
	testBytes(uint64(n))

	var t *byte
	if unsafe.Sizeof(t) == 8 {
		// Test mem > maxAlloc
		testInts(1 << 59)

		// Test elem.size*cap overflow
		testInts(1<<63 - 1)

		testInts(1<<64 - 1)
		testBytes(1<<64 - 1)
	} else {
		testInts(1<<31 - 1)

		// Test elem.size*cap overflow
		testInts(1<<32 - 1)
		testBytes(1<<32 - 1)
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

func testInts(n uint64) {
	testMakeInts(n)
	testMakeCopyInts(n)
	testMakeInAppendInts(n)
}

func testBytes(n uint64) {
	testMakeBytes(n)
	testMakeCopyBytes(n)
	testMakeInAppendBytes(n)
}

// Test make panics for given length or capacity n.
func testMakeInts(n uint64) {
	type T []int
	shouldPanic("len out of range", func() { _ = make(T, int(n)) })
	shouldPanic("cap out of range", func() { _ = make(T, 0, int(n)) })
	shouldPanic("len out of range", func() { _ = make(T, uint(n)) })
	shouldPanic("cap out of range", func() { _ = make(T, 0, uint(n)) })
	shouldPanic("len out of range", func() { _ = make(T, int64(n)) })
	shouldPanic("cap out of range", func() { _ = make(T, 0, int64(n)) })
	shouldPanic("len out of range", func() { _ = make(T, uint64(n)) })
	shouldPanic("cap out of range", func() { _ = make(T, 0, uint64(n)) })
}

func testMakeBytes(n uint64) {
	type T []byte
	shouldPanic("len out of range", func() { _ = make(T, int(n)) })
	shouldPanic("cap out of range", func() { _ = make(T, 0, int(n)) })
	shouldPanic("len out of range", func() { _ = make(T, uint(n)) })
	shouldPanic("cap out of range", func() { _ = make(T, 0, uint(n)) })
	shouldPanic("len out of range", func() { _ = make(T, int64(n)) })
	shouldPanic("cap out of range", func() { _ = make(T, 0, int64(n)) })
	shouldPanic("len out of range", func() { _ = make(T, uint64(n)) })
	shouldPanic("cap out of range", func() { _ = make(T, 0, uint64(n)) })
}

// Test make+copy panics since the gc compiler optimizes these
// to runtime.makeslicecopy calls.
func testMakeCopyInts(n uint64) {
	type T []int
	var c = make(T, 8)
	shouldPanic("len out of range", func() { x := make(T, int(n)); copy(x, c) })
	shouldPanic("cap out of range", func() { x := make(T, 0, int(n)); copy(x, c) })
	shouldPanic("len out of range", func() { x := make(T, uint(n)); copy(x, c) })
	shouldPanic("cap out of range", func() { x := make(T, 0, uint(n)); copy(x, c) })
	shouldPanic("len out of range", func() { x := make(T, int64(n)); copy(x, c) })
	shouldPanic("cap out of range", func() { x := make(T, 0, int64(n)); copy(x, c) })
	shouldPanic("len out of range", func() { x := make(T, uint64(n)); copy(x, c) })
	shouldPanic("cap out of range", func() { x := make(T, 0, uint64(n)); copy(x, c) })
}

func testMakeCopyBytes(n uint64) {
	type T []byte
	var c = make(T, 8)
	shouldPanic("len out of range", func() { x := make(T, int(n)); copy(x, c) })
	shouldPanic("cap out of range", func() { x := make(T, 0, int(n)); copy(x, c) })
	shouldPanic("len out of range", func() { x := make(T, uint(n)); copy(x, c) })
	shouldPanic("cap out of range", func() { x := make(T, 0, uint(n)); copy(x, c) })
	shouldPanic("len out of range", func() { x := make(T, int64(n)); copy(x, c) })
	shouldPanic("cap out of range", func() { x := make(T, 0, int64(n)); copy(x, c) })
	shouldPanic("len out of range", func() { x := make(T, uint64(n)); copy(x, c) })
	shouldPanic("cap out of range", func() { x := make(T, 0, uint64(n)); copy(x, c) })
}

// Test make in append panics for int slices since the gc compiler optimizes makes in appends.
func testMakeInAppendInts(n uint64) {
	type T []int
	for _, length := range []int{0, 1} {
		t := make(T, length)
		shouldPanic("len out of range", func() { _ = append(t, make(T, int(n))...) })
		shouldPanic("cap out of range", func() { _ = append(t, make(T, 0, int(n))...) })
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

func testMakeInAppendBytes(n uint64) {
	type T []byte
	for _, length := range []int{0, 1} {
		t := make(T, length)
		shouldPanic("len out of range", func() { _ = append(t, make(T, int(n))...) })
		shouldPanic("cap out of range", func() { _ = append(t, make(T, 0, int(n))...) })
		shouldPanic("len out of range", func() { _ = append(t, make(T, uint(n))...) })
		shouldPanic("cap out of range", func() { _ = append(t, make(T, 0, uint(n))...) })
		shouldPanic("len out of range", func() { _ = append(t, make(T, int64(n))...) })
		shouldPanic("cap out of range", func() { _ = append(t, make(T, 0, int64(n))...) })
		shouldPanic("len out of range", func() { _ = append(t, make(T, uint64(n))...) })
		shouldPanic("cap out of range", func() { _ = append(t, make(T, 0, uint64(n))...) })
	}
}
