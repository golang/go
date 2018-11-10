// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cgotest

import "C"

//export exportbyte
func exportbyte() byte {
	return 0
}

//export exportbool
func exportbool() bool {
	return false
}

//export exportrune
func exportrune() rune {
	return 0
}

//export exporterror
func exporterror() error {
	return nil
}

//export exportint
func exportint() int {
	return 0
}

//export exportuint
func exportuint() uint {
	return 0
}

//export exportuintptr
func exportuintptr() uintptr {
	return (uintptr)(0)
}

//export exportint8
func exportint8() int8 {
	return 0
}

//export exportuint8
func exportuint8() uint8 {
	return 0
}

//export exportint16
func exportint16() int16 {
	return 0
}

//export exportuint16
func exportuint16() uint16 {
	return 0
}

//export exportint32
func exportint32() int32 {
	return 0
}

//export exportuint32
func exportuint32() uint32 {
	return 0
}

//export exportint64
func exportint64() int64 {
	return 0
}

//export exportuint64
func exportuint64() uint64 {
	return 0
}

//export exportfloat32
func exportfloat32() float32 {
	return 0
}

//export exportfloat64
func exportfloat64() float64 {
	return 0
}

//export exportcomplex64
func exportcomplex64() complex64 {
	return 0
}

//export exportcomplex128
func exportcomplex128() complex128 {
	return 0
}
