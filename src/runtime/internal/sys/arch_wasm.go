// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sys

const (
	ArchFamily          = WASM
	BigEndian           = false
	CacheLineSize       = 64
	DefaultPhysPageSize = 65536
	PCQuantum           = 1
	Int64Align          = 8
	HugePageSize        = 0
	MinFrameSize        = 0
)

type Uintreg uint64
