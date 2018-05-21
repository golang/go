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
	var t *byte
	if unsafe.Sizeof(t) == 8 {
		var n2 int64 = 1 << 50
		shouldPanic("len out of range", func() { _ = make(T, int(n2)) })
		shouldPanic("cap out of range", func() { _ = make(T, 0, int(n2)) })
		n2 = 1<<63 - 1
		shouldPanic("len out of range", func() { _ = make(T, int(n2)) })
		shouldPanic("cap out of range", func() { _ = make(T, 0, int(n2)) })
	} else {
		n = 1<<31 - 1
		shouldPanic("len out of range", func() { _ = make(T, n) })
		shouldPanic("cap out of range", func() { _ = make(T, 0, n) })
		shouldPanic("len out of range", func() { _ = make(T, int64(n)) })
		shouldPanic("cap out of range", func() { _ = make(T, 0, int64(n)) })
	}

	// Test make in append panics since the gc compiler optimizes makes in appends.
	shouldPanic("len out of range", func() { _ = append(T{}, make(T, n)...) })
	shouldPanic("cap out of range", func() { _ = append(T{}, make(T, 0, n)...) })
	shouldPanic("len out of range", func() { _ = append(T{}, make(T, int64(n))...) })
	shouldPanic("cap out of range", func() { _ = append(T{}, make(T, 0, int64(n))...) })
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
