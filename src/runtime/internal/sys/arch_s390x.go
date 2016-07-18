// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sys

const (
	ArchFamily          = S390X
	BigEndian           = 1
	CacheLineSize       = 256
	DefaultPhysPageSize = 4096
	PCQuantum           = 2
	Int64Align          = 8
	HugePageSize        = 0
	MinFrameSize        = 8
)

type Uintreg uint64
