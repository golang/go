// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build 386 || arm || mips || mipsle

package runtime

import "unsafe"

// The number of bits stored in the numeric tag of a taggedPointer
const taggedPointerBits = 32

// The number of bits allowed in a tag.
const tagBits = 32

// On 32-bit systems, taggedPointer has a 32-bit pointer and 32-bit count.

// taggedPointerPack created a taggedPointer from a pointer and a tag.
// Tag bits that don't fit in the result are discarded.
func taggedPointerPack(ptr unsafe.Pointer, tag uintptr) taggedPointer {
	return taggedPointer(uintptr(ptr))<<32 | taggedPointer(tag)
}

// Pointer returns the pointer from a taggedPointer.
func (tp taggedPointer) pointer() unsafe.Pointer {
	return unsafe.Pointer(uintptr(tp >> 32))
}

// Tag returns the tag from a taggedPointer.
func (tp taggedPointer) tag() uintptr {
	return uintptr(tp)
}
