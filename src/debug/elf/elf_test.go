// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package elf

import (
	"fmt"
	"testing"
)

type nameTest struct {
	val any
	str string
}

var nameTests = []nameTest{
	{ELFOSABI_LINUX, "ELFOSABI_LINUX"},
	{ET_EXEC, "ET_EXEC"},
	{EM_860, "EM_860"},
	{SHN_LOPROC, "SHN_LOPROC"},
	{SHT_PROGBITS, "SHT_PROGBITS"},
	{SHF_MERGE + SHF_TLS, "SHF_MERGE+SHF_TLS"},
	{PT_LOAD, "PT_LOAD"},
	{PF_W + PF_R + 0x50, "PF_W+PF_R+0x50"},
	{DT_SYMBOLIC, "DT_SYMBOLIC"},
	{DF_BIND_NOW, "DF_BIND_NOW"},
	{NT_FPREGSET, "NT_FPREGSET"},
	{STB_GLOBAL, "STB_GLOBAL"},
	{STT_COMMON, "STT_COMMON"},
	{STV_HIDDEN, "STV_HIDDEN"},
	{R_X86_64_PC32, "R_X86_64_PC32"},
	{R_ALPHA_OP_PUSH, "R_ALPHA_OP_PUSH"},
	{R_ARM_THM_ABS5, "R_ARM_THM_ABS5"},
	{R_386_GOT32, "R_386_GOT32"},
	{R_PPC_GOT16_HI, "R_PPC_GOT16_HI"},
	{R_SPARC_GOT22, "R_SPARC_GOT22"},
	{ET_LOOS + 5, "ET_LOOS+5"},
	{ProgFlag(0x50), "0x50"},
}

func TestNames(t *testing.T) {
	for i, tt := range nameTests {
		s := fmt.Sprint(tt.val)
		if s != tt.str {
			t.Errorf("#%d: Sprint(%d) = %q, want %q", i, tt.val, s, tt.str)
		}
	}
}

func TestNobitsSection(t *testing.T) {
	const testdata = "testdata/gcc-amd64-linux-exec"
	f, err := Open(testdata)
	if err != nil {
		t.Fatalf("could not read %s: %v", testdata, err)
	}
	defer f.Close()
	bss := f.Section(".bss")
	bssData, err := bss.Data()
	if err != nil {
		t.Fatalf("error reading .bss section: %v", err)
	}
	if g, w := uint64(len(bssData)), bss.Size; g != w {
		t.Errorf(".bss section length mismatch: got %d, want %d", g, w)
	}
	for i := range bssData {
		if bssData[i] != 0 {
			t.Fatalf("unexpected non-zero byte at offset %d: %#x", i, bssData[i])
		}
	}
}
