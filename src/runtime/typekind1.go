// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

const (
	_KindBool = 1 + iota
	_KindInt
	_KindInt8
	_KindInt16
	_KindInt32
	_KindInt64
	_KindUint
	_KindUint8
	_KindUint16
	_KindUint32
	_KindUint64
	_KindUintptr
	_KindFloat32
	_KindFloat64
	_KindComplex64
	_KindComplex128
	_KindArray
	_KindChan
	_KindFunc
	_KindInterface
	_KindMap
	_KindPtr
	_KindSlice
	_KindString
	_KindStruct
	_KindUnsafePointer

	_KindDirectIface = 1 << 5
	_KindGCProg      = 1 << 6 // Type.gc points to GC program
	_KindNoPointers  = 1 << 7
	_KindMask        = (1 << 5) - 1
)
