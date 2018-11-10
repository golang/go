// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sys

const (
	ArchFamily          = I386
	BigEndian           = 0
	CacheLineSize       = 64
	DefaultPhysPageSize = GoosNacl*65536 + (1-GoosNacl)*4096 // 4k normally; 64k on NaCl
	PCQuantum           = 1
	Int64Align          = 4
	HugePageSize        = 1 << 21
	MinFrameSize        = 0
)

type Uintreg uint32
