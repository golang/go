// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crc32

// This file contains the code to call the SSE 4.2 version of the Castagnoli
// CRC.

// haveSSE42 is defined in crc_amd64.s and uses CPUID to test for SSE 4.2
// support.
func haveSSE42() bool

// castagnoliSSE42 is defined in crc_amd64.s and uses the SSE4.2 CRC32
// instruction.
func castagnoliSSE42(crc uint32, p []byte) uint32

var sse42 = haveSSE42()

func updateCastagnoli(crc uint32, p []byte) uint32 {
	if sse42 {
		return castagnoliSSE42(crc, p)
	}
	return update(crc, castagnoliTable, p)
}
