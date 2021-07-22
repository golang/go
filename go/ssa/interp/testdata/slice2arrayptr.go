// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test for slice to array pointer conversion introduced in go1.17
// See: https://tip.golang.org/ref/spec#Conversions_from_slice_to_array_pointer

package main

func main() {
	s := make([]byte, 2, 4)
	if s0 := (*[0]byte)(s); s0 == nil {
		panic("converted from non-nil slice result in nil array pointer")
	}
	if s2 := (*[2]byte)(s); &s2[0] != &s[0] {
		panic("the converted array is not slice underlying array")
	}
	wantPanic(
		func() {
			_ = (*[4]byte)(s) // panics: len([4]byte) > len(s)
		},
		"runtime error: array length is greater than slice length",
	)

	var t []string
	if t0 := (*[0]string)(t); t0 != nil {
		panic("nil slice converted to *[0]byte should be nil")
	}
	wantPanic(
		func() {
			_ = (*[1]string)(t) // panics: len([1]string) > len(t)
		},
		"runtime error: array length is greater than slice length",
	)
}

func wantPanic(fn func(), s string) {
	defer func() {
		err := recover()
		if err == nil {
			panic("expected panic")
		}
		if got := err.(error).Error(); got != s {
			panic("expected panic " + s + " got " + got)
		}
	}()
	fn()
}
