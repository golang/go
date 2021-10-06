// run

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test conversion from slice to array pointer.

package main

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

func main() {
	s := make([]byte, 8, 10)
	if p := (*[8]byte)(s); &p[0] != &s[0] {
		panic("*[8]byte conversion failed")
	}
	wantPanic(
		func() {
			_ = (*[9]byte)(s)
		},
		"runtime error: cannot convert slice with length 8 to pointer to array with length 9",
	)

	var n []byte
	if p := (*[0]byte)(n); p != nil {
		panic("nil slice converted to *[0]byte should be nil")
	}

	z := make([]byte, 0)
	if p := (*[0]byte)(z); p == nil {
		panic("empty slice converted to *[0]byte should be non-nil")
	}

	// Test with named types
	type Slice []int
	type Int4 [4]int
	type PInt4 *[4]int
	ii := make(Slice, 4)
	if p := (*Int4)(ii); &p[0] != &ii[0] {
		panic("*Int4 conversion failed")
	}
	if p := PInt4(ii); &p[0] != &ii[0] {
		panic("PInt4 conversion failed")
	}
}

// test static variable conversion

var (
	ss  = make([]string, 10)
	s5  = (*[5]string)(ss)
	s10 = (*[10]string)(ss)

	ns  []string
	ns0 = (*[0]string)(ns)

	zs  = make([]string, 0)
	zs0 = (*[0]string)(zs)
)

func init() {
	if &ss[0] != &s5[0] {
		panic("s5 conversion failed")
	}
	if &ss[0] != &s10[0] {
		panic("s5 conversion failed")
	}
	if ns0 != nil {
		panic("ns0 should be nil")
	}
	if zs0 == nil {
		panic("zs0 should not be nil")
	}
}
