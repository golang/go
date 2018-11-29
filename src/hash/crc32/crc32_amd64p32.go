// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package crc32

import "internal/cpu"

// This file contains the code to call the SSE 4.2 version of the Castagnoli
// CRC.

// castagnoliSSE42 is defined in crc32_amd64p32.s and uses the SSE4.2 CRC32
// instruction.
//go:noescape
func castagnoliSSE42(crc uint32, p []byte) uint32

func archAvailableCastagnoli() bool {
	return cpu.X86.HasSSE42
}

func archInitCastagnoli() {
	if !cpu.X86.HasSSE42 {
		panic("not available")
	}
	// No initialization necessary.
}

func archUpdateCastagnoli(crc uint32, p []byte) uint32 {
	if !cpu.X86.HasSSE42 {
		panic("not available")
	}
	return castagnoliSSE42(crc, p)
}

func archAvailableIEEE() bool                    { return false }
func archInitIEEE()                              { panic("not available") }
func archUpdateIEEE(crc uint32, p []byte) uint32 { panic("not available") }
