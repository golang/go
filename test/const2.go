// errorcheck

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Verify that large integer constant expressions cause overflow.
// Does not compile.

package main

const (
	A int = 1
	B byte;	// ERROR "type without expr|expected .=."
)

const LargeA = 1000000000000000000
const LargeB = LargeA * LargeA * LargeA
const LargeC = LargeB * LargeB * LargeB // GC_ERROR "constant multiplication overflow"

const AlsoLargeA = LargeA << 400 << 400 >> 400 >> 400 // GC_ERROR "constant shift overflow"
