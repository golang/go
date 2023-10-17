// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import (
	"unsafe"
)

func recurse(i int, s []byte) byte {
	s[0] = byte(i)
	if i == 0 {
		return s[i]
	} else {
		var a [1024]byte
		r := recurse(i-1, a[:])
		return r + a[0]
	}
}

//go:uintptrescapes
func F1(a uintptr) {
	var s [16]byte
	recurse(4096, s[:])
	*(*int)(unsafe.Pointer(a)) = 42
}

//go:uintptrescapes
func F2(a ...uintptr) {
	var s [16]byte
	recurse(4096, s[:])
	*(*int)(unsafe.Pointer(a[0])) = 42
}

type t struct{}

func GetT() *t {
	return &t{}
}

//go:uintptrescapes
func (*t) M1(a uintptr) {
	var s [16]byte
	recurse(4096, s[:])
	*(*int)(unsafe.Pointer(a)) = 42
}

//go:uintptrescapes
func (*t) M2(a ...uintptr) {
	var s [16]byte
	recurse(4096, s[:])
	*(*int)(unsafe.Pointer(a[0])) = 42
}
