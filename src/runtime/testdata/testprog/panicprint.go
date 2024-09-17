// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "sync"

type MyBool bool
type MyComplex128 complex128
type MyComplex64 complex64
type MyFloat32 float32
type MyFloat64 float64
type MyInt int
type MyInt8 int8
type MyInt16 int16
type MyInt32 int32
type MyInt64 int64
type MyString string
type MyUint uint
type MyUint8 uint8
type MyUint16 uint16
type MyUint32 uint32
type MyUint64 uint64
type MyUintptr uintptr

func panicCustomComplex64() {
	panic(MyComplex64(0.11 + 3i))
}

func panicCustomComplex128() {
	panic(MyComplex128(32.1 + 10i))
}

func panicCustomString() {
	panic(MyString("Panic\nline two"))
}

func panicCustomBool() {
	panic(MyBool(true))
}

func panicCustomInt() {
	panic(MyInt(93))
}

func panicCustomInt8() {
	panic(MyInt8(93))
}

func panicCustomInt16() {
	panic(MyInt16(93))
}

func panicCustomInt32() {
	panic(MyInt32(93))
}

func panicCustomInt64() {
	panic(MyInt64(93))
}

func panicCustomUint() {
	panic(MyUint(93))
}

func panicCustomUint8() {
	panic(MyUint8(93))
}

func panicCustomUint16() {
	panic(MyUint16(93))
}

func panicCustomUint32() {
	panic(MyUint32(93))
}

func panicCustomUint64() {
	panic(MyUint64(93))
}

func panicCustomUintptr() {
	panic(MyUintptr(93))
}

func panicCustomFloat64() {
	panic(MyFloat64(-93.70))
}

func panicCustomFloat32() {
	panic(MyFloat32(-93.70))
}

func panicDeferFatal() {
	var mu sync.Mutex
	defer mu.Unlock()
	var i *int
	*i = 0
}

func panicDoublieDeferFatal() {
	var mu sync.Mutex
	defer mu.Unlock()
	defer func() {
		panic(recover())
	}()
	var i *int
	*i = 0
}

func init() {
	register("panicCustomComplex64", panicCustomComplex64)
	register("panicCustomComplex128", panicCustomComplex128)
	register("panicCustomBool", panicCustomBool)
	register("panicCustomFloat32", panicCustomFloat32)
	register("panicCustomFloat64", panicCustomFloat64)
	register("panicCustomInt", panicCustomInt)
	register("panicCustomInt8", panicCustomInt8)
	register("panicCustomInt16", panicCustomInt16)
	register("panicCustomInt32", panicCustomInt32)
	register("panicCustomInt64", panicCustomInt64)
	register("panicCustomString", panicCustomString)
	register("panicCustomUint", panicCustomUint)
	register("panicCustomUint8", panicCustomUint8)
	register("panicCustomUint16", panicCustomUint16)
	register("panicCustomUint32", panicCustomUint32)
	register("panicCustomUint64", panicCustomUint64)
	register("panicCustomUintptr", panicCustomUintptr)
	register("panicDeferFatal", panicDeferFatal)
	register("panicDoublieDeferFatal", panicDoublieDeferFatal)
}
