// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sys

const (
	ArchFamily    = AMD64
	BigEndian     = 0
	CacheLineSize = 64
	PhysPageSize  = 65536*GoosNacl + 4096*(1-GoosNacl)
	PCQuantum     = 1
	Int64Align    = 8
	HugePageSize  = 1 << 21
	MinFrameSize  = 0
)

type Uintreg uint64
