// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "C"

//export ReturnEmpty
func ReturnEmpty() {
	return
}

//export ReturnOnlyUint8
func ReturnOnlyUint8() (uint8, uint8, uint8) {
	return 1, 2, 3
}

//export ReturnOnlyUint16
func ReturnOnlyUint16() (uint16, uint16, uint16) {
	return 1, 2, 3
}

//export ReturnOnlyUint32
func ReturnOnlyUint32() (uint32, uint32, uint32) {
	return 1, 2, 3
}

//export ReturnOnlyUint64
func ReturnOnlyUint64() (uint64, uint64, uint64) {
	return 1, 2, 3
}

//export ReturnOnlyInt
func ReturnOnlyInt() (int, int, int) {
	return 1, 2, 3
}

//export ReturnOnlyPtr
func ReturnOnlyPtr() (*int, *int, *int) {
	a, b, c := 1, 2, 3
	return &a, &b, &c
}

//export ReturnString
func ReturnString() string {
	return "hello"
}

//export ReturnByteSlice
func ReturnByteSlice() []byte {
	return []byte{1, 2, 3}
}

//export InputAndReturnUint8
func InputAndReturnUint8(a, b, c uint8) (uint8, uint8, uint8) {
	return a, b, c
}

//export MixedTypes
func MixedTypes(a uint8, b uint16, c uint32, d uint64, e int, f *int) (uint8, uint16, uint32, uint64, int, *int) {
	return a, b, c, d, e, f
}
