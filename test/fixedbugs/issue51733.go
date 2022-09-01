// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo && !aix

package main

import (
	"log"
	"runtime/cgo"
	"unsafe"
)

type S struct{ _ cgo.Incomplete }

func main() {
	p := (*S)(unsafe.Pointer(uintptr(0x8000)))
	var v any = p
	p2 := v.(*S)
	if p != p2 {
		log.Fatalf("%p != %p", unsafe.Pointer(p), unsafe.Pointer(p2))
	}
	p2 = typeAssert[*S](v)
	if p != p2 {
		log.Fatalf("%p != %p from typeAssert", unsafe.Pointer(p), unsafe.Pointer(p2))
	}
}

func typeAssert[T any](v any) T {
	return v.(T)
}
