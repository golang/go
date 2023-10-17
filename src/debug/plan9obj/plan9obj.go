// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * Plan 9 a.out constants and data structures
 */

package plan9obj

// Plan 9 Program header.
type prog struct {
	Magic uint32 /* magic number */
	Text  uint32 /* size of text segment */
	Data  uint32 /* size of initialized data */
	Bss   uint32 /* size of uninitialized data */
	Syms  uint32 /* size of symbol table */
	Entry uint32 /* entry point */
	Spsz  uint32 /* size of pc/sp offset table */
	Pcsz  uint32 /* size of pc/line number table */
}

// Plan 9 symbol table entries.
type sym struct {
	value uint64
	typ   byte
	name  []byte
}

const (
	Magic64 = 0x8000 // 64-bit expanded header

	Magic386   = (4*11+0)*11 + 7
	MagicAMD64 = (4*26+0)*26 + 7 + Magic64
	MagicARM   = (4*20+0)*20 + 7
)
