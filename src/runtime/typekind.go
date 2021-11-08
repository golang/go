// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

const (
	kindBool = 1 + iota
	kindInt
	kindInt8
	kindInt16
	kindInt32
	kindInt64
	kindUint
	kindUint8
	kindUint16
	kindUint32
	kindUint64
	kindUintptr
	kindFloat32
	kindFloat64
	kindComplex64
	kindComplex128
	kindArray
	kindChan
	kindFunc
	kindInterface
	kindMap
	kindPtr
	kindSlice
	kindString
	kindStruct
	kindUnsafePointer

	kindDirectIface = 1 << 5
	kindGCProg      = 1 << 6
	kindMask        = (1 << 5) - 1
)

// isDirectIface reports whether t is stored directly in an interface value.
func isDirectIface(t *_type) bool {
	return t.kind&kindDirectIface != 0
}
