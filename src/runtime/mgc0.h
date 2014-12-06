// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Used by cmd/gc.

enum {
	gcBits = 4,
	BitsPerPointer = 2,
	BitsDead = 0,
	BitsScalar = 1,
	BitsPointer = 2,
	BitsMask = 3,
	PointersPerByte = 8/BitsPerPointer,
	insData = 1,
	insArray,
	insArrayEnd,
	insEnd,

	// 64 bytes cover objects of size 1024/512 on 64/32 bits, respectively.
	MaxGCMask	= 65536, // TODO(rsc): change back to 64
};
