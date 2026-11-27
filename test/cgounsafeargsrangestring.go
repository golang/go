// run -gcflags=-std

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

// cgo_unsafe_args permits a pointer to an argument to access the entire ABI0
// argument and result area. word retains a clear address-taken flag.
// Mutating it must preserve the string snapshot used by the range.
//
//go:cgo_unsafe_args
func rangeStringCgoUnsafeArgs(arg uint64) (word, sum uint64) {
	p := unsafe.Slice((*byte)(unsafe.Add(unsafe.Pointer(&arg), unsafe.Sizeof(arg))), 8)
	for _, r := range string(p) {
		word = 0x6161616161616161
		sum += uint64(r)
	}
	return
}

func main() {
	word, sum := rangeStringCgoUnsafeArgs(0)
	if word != 0x6161616161616161 || sum != 0 {
		println(word, sum)
		panic("cgo result mutation changed string snapshot")
	}
}
