// $G $D/$F.go

// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "unsafe"

func main() {
	var x int
	
	a := uint64(uintptr(unsafe.Pointer(&x)))
	b := uint32(uintptr(unsafe.Pointer(&x)))
	c := uint16(uintptr(unsafe.Pointer(&x)))
	d := int64(uintptr(unsafe.Pointer(&x)))
	e := int32(uintptr(unsafe.Pointer(&x)))
	f := int16(uintptr(unsafe.Pointer(&x)))

	_, _, _, _, _, _ = a, b, c, d, e, f
}
