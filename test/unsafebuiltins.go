// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"math"
	"unsafe"
)

const maxUintptr = 1 << (8 * unsafe.Sizeof(uintptr(0)))

func main() {
	var p [10]byte

	// unsafe.Add
	{
		p1 := unsafe.Pointer(&p[1])
		assert(unsafe.Add(p1, 1) == unsafe.Pointer(&p[2]))
		assert(unsafe.Add(p1, -1) == unsafe.Pointer(&p[0]))
	}

	// unsafe.Slice
	{
		s := unsafe.Slice(&p[0], len(p))
		assert(&s[0] == &p[0])
		assert(len(s) == len(p))
		assert(cap(s) == len(p))

		// nil pointer with zero length returns nil
		assert(unsafe.Slice((*int)(nil), 0) == nil)

		// nil pointer with positive length panics
		mustPanic(func() { _ = unsafe.Slice((*int)(nil), 1) })

		// negative length
		var neg int = -1
		mustPanic(func() { _ = unsafe.Slice(new(byte), neg) })

		// length too large
		var tooBig uint64 = math.MaxUint64
		mustPanic(func() { _ = unsafe.Slice(new(byte), tooBig) })

		// size overflows address space
		mustPanic(func() { _ = unsafe.Slice(new(uint64), maxUintptr/8) })
		mustPanic(func() { _ = unsafe.Slice(new(uint64), maxUintptr/8+1) })

		// sliced memory overflows address space
		last := (*byte)(unsafe.Pointer(^uintptr(0)))
		_ = unsafe.Slice(last, 1)
		mustPanic(func() { _ = unsafe.Slice(last, 2) })
	}

	// unsafe.String
	{
		s := unsafe.String(&p[0], len(p))
		assert(s == string(p[:]))
		assert(len(s) == len(p))

		// the empty string
		assert(unsafe.String(nil, 0) == "")

		// nil pointer with positive length panics
		mustPanic(func() { _ = unsafe.String(nil, 1) })

		// negative length
		var neg int = -1
		mustPanic(func() { _ = unsafe.String(new(byte), neg) })

		// length too large
		var tooBig uint64 = math.MaxUint64
		mustPanic(func() { _ = unsafe.String(new(byte), tooBig) })

		// string memory overflows address space
		last := (*byte)(unsafe.Pointer(^uintptr(0)))
		_ = unsafe.String(last, 1)
		mustPanic(func() { _ = unsafe.String(last, 2) })
	}

	// unsafe.StringData
	{
		var s = "string"
		assert(string(unsafe.Slice(unsafe.StringData(s), len(s))) == s)
	}

	//unsafe.SliceData
	{
		var s = []byte("slice")
		assert(unsafe.String(unsafe.SliceData(s), len(s)) == string(s))
	}
}

func assert(ok bool) {
	if !ok {
		panic("FAIL")
	}
}

func mustPanic(f func()) {
	defer func() {
		assert(recover() != nil)
	}()
	f()
}
