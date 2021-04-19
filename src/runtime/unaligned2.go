// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build arm mips mipsle mips64 mips64le

package runtime

import "unsafe"

// Note: These routines perform the read with an unspecified endianness.
func readUnaligned32(p unsafe.Pointer) uint32 {
	q := (*[4]byte)(p)
	return uint32(q[0]) + uint32(q[1])<<8 + uint32(q[2])<<16 + uint32(q[3])<<24
}

func readUnaligned64(p unsafe.Pointer) uint64 {
	q := (*[8]byte)(p)
	return uint64(q[0]) + uint64(q[1])<<8 + uint64(q[2])<<16 + uint64(q[3])<<24 + uint64(q[4])<<32 + uint64(q[5])<<40 + uint64(q[6])<<48 + uint64(q[7])<<56
}
