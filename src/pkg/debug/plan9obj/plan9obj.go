// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * Plan 9 a.out constants and data structures
 */

package plan9obj

import (
	"bytes"
	"encoding/binary"
)

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
	hsize      = 4 * 8
	_HDR_MAGIC = 0x00008000 /* header expansion */
)

func magic(f, b int) string {
	buf := new(bytes.Buffer)
	var i uint32 = uint32((f) | ((((4 * (b)) + 0) * (b)) + 7))
	binary.Write(buf, binary.BigEndian, i)
	return string(buf.Bytes())
}

var (
	_A_MAGIC = magic(0, 8)           /* 68020 (retired) */
	_I_MAGIC = magic(0, 11)          /* intel 386 */
	_J_MAGIC = magic(0, 12)          /* intel 960 (retired) */
	_K_MAGIC = magic(0, 13)          /* sparc */
	_V_MAGIC = magic(0, 16)          /* mips 3000 BE */
	_X_MAGIC = magic(0, 17)          /* att dsp 3210 (retired) */
	_M_MAGIC = magic(0, 18)          /* mips 4000 BE */
	_D_MAGIC = magic(0, 19)          /* amd 29000 (retired) */
	_E_MAGIC = magic(0, 20)          /* arm */
	_Q_MAGIC = magic(0, 21)          /* powerpc */
	_N_MAGIC = magic(0, 22)          /* mips 4000 LE */
	_L_MAGIC = magic(0, 23)          /* dec alpha (retired) */
	_P_MAGIC = magic(0, 24)          /* mips 3000 LE */
	_U_MAGIC = magic(0, 25)          /* sparc64 (retired) */
	_S_MAGIC = magic(_HDR_MAGIC, 26) /* amd64 */
	_T_MAGIC = magic(_HDR_MAGIC, 27) /* powerpc64 */
	_R_MAGIC = magic(_HDR_MAGIC, 28) /* arm64 */
)

type ExecTable struct {
	Magic string
	Ptrsz int
	Hsize uint32
}

var exectab = []ExecTable{
	{_A_MAGIC, 4, hsize},
	{_I_MAGIC, 4, hsize},
	{_J_MAGIC, 4, hsize},
	{_K_MAGIC, 4, hsize},
	{_V_MAGIC, 4, hsize},
	{_X_MAGIC, 4, hsize},
	{_M_MAGIC, 4, hsize},
	{_D_MAGIC, 4, hsize},
	{_E_MAGIC, 4, hsize},
	{_Q_MAGIC, 4, hsize},
	{_N_MAGIC, 4, hsize},
	{_L_MAGIC, 4, hsize},
	{_P_MAGIC, 4, hsize},
	{_U_MAGIC, 4, hsize},
	{_S_MAGIC, 8, hsize + 8},
	{_T_MAGIC, 8, hsize + 8},
	{_R_MAGIC, 8, hsize + 8},
}
