// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package archsimd

// Integer limits. These are the same as the constants in package math, which
// archsimd can't import (see go/build's TestDependencies).
const (
	maxInt8   = 1<<7 - 1
	minInt8   = -1 << 7
	maxInt16  = 1<<15 - 1
	minInt16  = -1 << 15
	maxInt32  = 1<<31 - 1
	minInt32  = -1 << 31
	maxUint8  = 1<<8 - 1
	maxUint16 = 1<<16 - 1
	maxUint32 = 1<<32 - 1
)
