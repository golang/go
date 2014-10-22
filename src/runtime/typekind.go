// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

const (
	kindBool          = _KindBool
	kindInt           = _KindInt
	kindInt8          = _KindInt8
	kindInt16         = _KindInt16
	kindInt32         = _KindInt32
	kindInt64         = _KindInt64
	kindUint          = _KindUint
	kindUint8         = _KindUint8
	kindUint16        = _KindUint16
	kindUint32        = _KindUint32
	kindUint64        = _KindUint64
	kindUintptr       = _KindUintptr
	kindFloat32       = _KindFloat32
	kindFloat64       = _KindFloat64
	kindComplex64     = _KindComplex64
	kindComplex128    = _KindComplex128
	kindArray         = _KindArray
	kindChan          = _KindChan
	kindFunc          = _KindFunc
	kindInterface     = _KindInterface
	kindMap           = _KindMap
	kindPtr           = _KindPtr
	kindSlice         = _KindSlice
	kindString        = _KindString
	kindStruct        = _KindStruct
	kindUnsafePointer = _KindUnsafePointer

	kindDirectIface = _KindDirectIface
	kindGCProg      = _KindGCProg
	kindNoPointers  = _KindNoPointers
	kindMask        = _KindMask
)

// isDirectIface reports whether t is stored directly in an interface value.
func isDirectIface(t *_type) bool {
	return t.kind&kindDirectIface != 0
}
