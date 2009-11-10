// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package elf

import (
	"fmt";
	"testing";
)

type nameTest struct {
	val	interface{};
	str	string;
}

var nameTests = []nameTest{
	nameTest{ELFOSABI_LINUX, "ELFOSABI_LINUX"},
	nameTest{ET_EXEC, "ET_EXEC"},
	nameTest{EM_860, "EM_860"},
	nameTest{SHN_LOPROC, "SHN_LOPROC"},
	nameTest{SHT_PROGBITS, "SHT_PROGBITS"},
	nameTest{SHF_MERGE + SHF_TLS, "SHF_MERGE+SHF_TLS"},
	nameTest{PT_LOAD, "PT_LOAD"},
	nameTest{PF_W + PF_R + 0x50, "PF_W+PF_R+0x50"},
	nameTest{DT_SYMBOLIC, "DT_SYMBOLIC"},
	nameTest{DF_BIND_NOW, "DF_BIND_NOW"},
	nameTest{NT_FPREGSET, "NT_FPREGSET"},
	nameTest{STB_GLOBAL, "STB_GLOBAL"},
	nameTest{STT_COMMON, "STT_COMMON"},
	nameTest{STV_HIDDEN, "STV_HIDDEN"},
	nameTest{R_X86_64_PC32, "R_X86_64_PC32"},
	nameTest{R_ALPHA_OP_PUSH, "R_ALPHA_OP_PUSH"},
	nameTest{R_ARM_THM_ABS5, "R_ARM_THM_ABS5"},
	nameTest{R_386_GOT32, "R_386_GOT32"},
	nameTest{R_PPC_GOT16_HI, "R_PPC_GOT16_HI"},
	nameTest{R_SPARC_GOT22, "R_SPARC_GOT22"},
	nameTest{ET_LOOS + 5, "ET_LOOS+5"},
	nameTest{ProgFlag(0x50), "0x50"},
}

func TestNames(t *testing.T) {
	for i, tt := range nameTests {
		s := fmt.Sprint(tt.val);
		if s != tt.str {
			t.Errorf("#%d: want %q have %q", i, s, tt.str)
		}
	}
}
