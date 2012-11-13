// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// PtrSize vs sizeof(void*): This file is also included from src/cmd/ld/...
// which defines PtrSize to be different from sizeof(void*) when crosscompiling.

enum {
	KindBool = 1,
	KindInt,
	KindInt8,
	KindInt16,
	KindInt32,
	KindInt64,
	KindUint,
	KindUint8,
	KindUint16,
	KindUint32,
	KindUint64,
	KindUintptr,
	KindFloat32,
	KindFloat64,
	KindComplex64,
	KindComplex128,
	KindArray,
	KindChan,
	KindFunc,
	KindInterface,
	KindMap,
	KindPtr,
	KindSlice,
	KindString,
	KindStruct,
	KindUnsafePointer,

	KindNoPointers = 1<<7,

	// size of Type structure.
	CommonSize = 6*PtrSize + 8,
};

