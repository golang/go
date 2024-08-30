// run

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"strings"
	"unsafe"
)

func shift[T any]() int64 {
	return 1 << unsafe.Sizeof(*new(T))
}

func div[T any]() uintptr {
	return 1 / unsafe.Sizeof(*new(T))
}

func add[T any]() int64 {
	return 1<<63 - 1 + int64(unsafe.Sizeof(*new(T)))
}

func main() {
	shift[[62]byte]()
	shift[[63]byte]()
	shift[[64]byte]()
	shift[[100]byte]()
	shift[[1e6]byte]()

	add[[1]byte]()
	shouldPanic("divide by zero", func() { div[[0]byte]() })
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
