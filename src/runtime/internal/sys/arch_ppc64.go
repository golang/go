// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sys

const (
	ArchFamily          = PPC64
	BigEndian           = 1
	CacheLineSize       = 128
	DefaultPhysPageSize = 65536
	PCQuantum           = 4
	Int64Align          = 8
	HugePageSize        = 0
	MinFrameSize        = 32
)

type Uintreg uint64
