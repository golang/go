// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package elf

import (
	"bytes"
	"compress/gzip"
	"debug/dwarf"
	"encoding/binary"
	"io"
	"math/rand"
	"net"
	"os"
	"path"
	"reflect"
	"runtime"
	"testing"
)

type fileTest struct {
	file     string
	hdr      FileHeader
	sections []SectionHeader
	progs    []ProgHeader
	needed   []string
}

var fileTests = []fileTest{
	{
		"testdata/gcc-386-freebsd-exec",
		FileHeader{ELFCLASS32, ELFDATA2LSB, EV_CURRENT, ELFOSABI_FREEBSD, 0, binary.LittleEndian, ET_EXEC, EM_386, 0x80483cc},
		[]SectionHeader{
			{"", SHT_NULL, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
			{".interp", SHT_PROGBITS, SHF_ALLOC, 0x80480d4, 0xd4, 0x15, 0x0, 0x0, 0x1, 0x0, 0x15},
			{".hash", SHT_HASH, SHF_ALLOC, 0x80480ec, 0xec, 0x90, 0x3, 0x0, 0x4, 0x4, 0x90},
			{".dynsym", SHT_DYNSYM, SHF_ALLOC, 0x804817c, 0x17c, 0x110, 0x4, 0x1, 0x4, 0x10, 0x110},
			{".dynstr", SHT_STRTAB, SHF_ALLOC, 0x804828c, 0x28c, 0xbb, 0x0, 0x0, 0x1, 0x0, 0xbb},
			{".rel.plt", SHT_REL, SHF_ALLOC, 0x8048348, 0x348, 0x20, 0x3, 0x7, 0x4, 0x8, 0x20},
			{".init", SHT_PROGBITS, SHF_ALLOC + SHF_EXECINSTR, 0x8048368, 0x368, 0x11, 0x0, 0x0, 0x4, 0x0, 0x11},
			{".plt", SHT_PROGBITS, SHF_ALLOC + SHF_EXECINSTR, 0x804837c, 0x37c, 0x50, 0x0, 0x0, 0x4, 0x4, 0x50},
			{".text", SHT_PROGBITS, SHF_ALLOC + SHF_EXECINSTR, 0x80483cc, 0x3cc, 0x180, 0x0, 0x0, 0x4, 0x0, 0x180},
			{".fini", SHT_PROGBITS, SHF_ALLOC + SHF_EXECINSTR, 0x804854c, 0x54c, 0xc, 0x0, 0x0, 0x4, 0x0, 0xc},
			{".rodata", SHT_PROGBITS, SHF_ALLOC, 0x8048558, 0x558, 0xa3, 0x0, 0x0, 0x1, 0x0, 0xa3},
			{".data", SHT_PROGBITS, SHF_WRITE + SHF_ALLOC, 0x80495fc, 0x5fc, 0xc, 0x0, 0x0, 0x4, 0x0, 0xc},
			{".eh_frame", SHT_PROGBITS, SHF_ALLOC, 0x8049608, 0x608, 0x4, 0x0, 0x0, 0x4, 0x0, 0x4},
			{".dynamic", SHT_DYNAMIC, SHF_WRITE + SHF_ALLOC, 0x804960c, 0x60c, 0x98, 0x4, 0x0, 0x4, 0x8, 0x98},
			{".ctors", SHT_PROGBITS, SHF_WRITE + SHF_ALLOC, 0x80496a4, 0x6a4, 0x8, 0x0, 0x0, 0x4, 0x0, 0x8},
			{".dtors", SHT_PROGBITS, SHF_WRITE + SHF_ALLOC, 0x80496ac, 0x6ac, 0x8, 0x0, 0x0, 0x4, 0x0, 0x8},
			{".jcr", SHT_PROGBITS, SHF_WRITE + SHF_ALLOC, 0x80496b4, 0x6b4, 0x4, 0x0, 0x0, 0x4, 0x0, 0x4},
			{".got", SHT_PROGBITS, SHF_WRITE + SHF_ALLOC, 0x80496b8, 0x6b8, 0x1c, 0x0, 0x0, 0x4, 0x4, 0x1c},
			{".bss", SHT_NOBITS, SHF_WRITE + SHF_ALLOC, 0x80496d4, 0x6d4, 0x20, 0x0, 0x0, 0x4, 0x0, 0x20},
			{".comment", SHT_PROGBITS, 0x0, 0x0, 0x6d4, 0x12d, 0x0, 0x0, 0x1, 0x0, 0x12d},
			{".debug_aranges", SHT_PROGBITS, 0x0, 0x0, 0x801, 0x20, 0x0, 0x0, 0x1, 0x0, 0x20},
			{".debug_pubnames", SHT_PROGBITS, 0x0, 0x0, 0x821, 0x1b, 0x0, 0x0, 0x1, 0x0, 0x1b},
			{".debug_info", SHT_PROGBITS, 0x0, 0x0, 0x83c, 0x11d, 0x0, 0x0, 0x1, 0x0, 0x11d},
			{".debug_abbrev", SHT_PROGBITS, 0x0, 0x0, 0x959, 0x41, 0x0, 0x0, 0x1, 0x0, 0x41},
			{".debug_line", SHT_PROGBITS, 0x0, 0x0, 0x99a, 0x35, 0x0, 0x0, 0x1, 0x0, 0x35},
			{".debug_frame", SHT_PROGBITS, 0x0, 0x0, 0x9d0, 0x30, 0x0, 0x0, 0x4, 0x0, 0x30},
			{".debug_str", SHT_PROGBITS, 0x0, 0x0, 0xa00, 0xd, 0x0, 0x0, 0x1, 0x0, 0xd},
			{".shstrtab", SHT_STRTAB, 0x0, 0x0, 0xa0d, 0xf8, 0x0, 0x0, 0x1, 0x0, 0xf8},
			{".symtab", SHT_SYMTAB, 0x0, 0x0, 0xfb8, 0x4b0, 0x1d, 0x38, 0x4, 0x10, 0x4b0},
			{".strtab", SHT_STRTAB, 0x0, 0x0, 0x1468, 0x206, 0x0, 0x0, 0x1, 0x0, 0x206},
		},
		[]ProgHeader{
			{PT_PHDR, PF_R + PF_X, 0x34, 0x8048034, 0x8048034, 0xa0, 0xa0, 0x4},
			{PT_INTERP, PF_R, 0xd4, 0x80480d4, 0x80480d4, 0x15, 0x15, 0x1},
			{PT_LOAD, PF_R + PF_X, 0x0, 0x8048000, 0x8048000, 0x5fb, 0x5fb, 0x1000},
			{PT_LOAD, PF_R + PF_W, 0x5fc, 0x80495fc, 0x80495fc, 0xd8, 0xf8, 0x1000},
			{PT_DYNAMIC, PF_R + PF_W, 0x60c, 0x804960c, 0x804960c, 0x98, 0x98, 0x4},
		},
		[]string{"libc.so.6"},
	},
	{
		"testdata/gcc-amd64-linux-exec",
		FileHeader{ELFCLASS64, ELFDATA2LSB, EV_CURRENT, ELFOSABI_NONE, 0, binary.LittleEndian, ET_EXEC, EM_X86_64, 0x4003e0},
		[]SectionHeader{
			{"", SHT_NULL, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
			{".interp", SHT_PROGBITS, SHF_ALLOC, 0x400200, 0x200, 0x1c, 0x0, 0x0, 0x1, 0x0, 0x1c},
			{".note.ABI-tag", SHT_NOTE, SHF_ALLOC, 0x40021c, 0x21c, 0x20, 0x0, 0x0, 0x4, 0x0, 0x20},
			{".hash", SHT_HASH, SHF_ALLOC, 0x400240, 0x240, 0x24, 0x5, 0x0, 0x8, 0x4, 0x24},
			{".gnu.hash", SHT_LOOS + 268435446, SHF_ALLOC, 0x400268, 0x268, 0x1c, 0x5, 0x0, 0x8, 0x0, 0x1c},
			{".dynsym", SHT_DYNSYM, SHF_ALLOC, 0x400288, 0x288, 0x60, 0x6, 0x1, 0x8, 0x18, 0x60},
			{".dynstr", SHT_STRTAB, SHF_ALLOC, 0x4002e8, 0x2e8, 0x3d, 0x0, 0x0, 0x1, 0x0, 0x3d},
			{".gnu.version", SHT_HIOS, SHF_ALLOC, 0x400326, 0x326, 0x8, 0x5, 0x0, 0x2, 0x2, 0x8},
			{".gnu.version_r", SHT_LOOS + 268435454, SHF_ALLOC, 0x400330, 0x330, 0x20, 0x6, 0x1, 0x8, 0x0, 0x20},
			{".rela.dyn", SHT_RELA, SHF_ALLOC, 0x400350, 0x350, 0x18, 0x5, 0x0, 0x8, 0x18, 0x18},
			{".rela.plt", SHT_RELA, SHF_ALLOC, 0x400368, 0x368, 0x30, 0x5, 0xc, 0x8, 0x18, 0x30},
			{".init", SHT_PROGBITS, SHF_ALLOC + SHF_EXECINSTR, 0x400398, 0x398, 0x18, 0x0, 0x0, 0x4, 0x0, 0x18},
			{".plt", SHT_PROGBITS, SHF_ALLOC + SHF_EXECINSTR, 0x4003b0, 0x3b0, 0x30, 0x0, 0x0, 0x4, 0x10, 0x30},
			{".text", SHT_PROGBITS, SHF_ALLOC + SHF_EXECINSTR, 0x4003e0, 0x3e0, 0x1b4, 0x0, 0x0, 0x10, 0x0, 0x1b4},
			{".fini", SHT_PROGBITS, SHF_ALLOC + SHF_EXECINSTR, 0x400594, 0x594, 0xe, 0x0, 0x0, 0x4, 0x0, 0xe},
			{".rodata", SHT_PROGBITS, SHF_ALLOC, 0x4005a4, 0x5a4, 0x11, 0x0, 0x0, 0x4, 0x0, 0x11},
			{".eh_frame_hdr", SHT_PROGBITS, SHF_ALLOC, 0x4005b8, 0x5b8, 0x24, 0x0, 0x0, 0x4, 0x0, 0x24},
			{".eh_frame", SHT_PROGBITS, SHF_ALLOC, 0x4005e0, 0x5e0, 0xa4, 0x0, 0x0, 0x8, 0x0, 0xa4},
			{".ctors", SHT_PROGBITS, SHF_WRITE + SHF_ALLOC, 0x600688, 0x688, 0x10, 0x0, 0x0, 0x8, 0x0, 0x10},
			{".dtors", SHT_PROGBITS, SHF_WRITE + SHF_ALLOC, 0x600698, 0x698, 0x10, 0x0, 0x0, 0x8, 0x0, 0x10},
			{".jcr", SHT_PROGBITS, SHF_WRITE + SHF_ALLOC, 0x6006a8, 0x6a8, 0x8, 0x0, 0x0, 0x8, 0x0, 0x8},
			{".dynamic", SHT_DYNAMIC, SHF_WRITE + SHF_ALLOC, 0x6006b0, 0x6b0, 0x1a0, 0x6, 0x0, 0x8, 0x10, 0x1a0},
			{".got", SHT_PROGBITS, SHF_WRITE + SHF_ALLOC, 0x600850, 0x850, 0x8, 0x0, 0x0, 0x8, 0x8, 0x8},
			{".got.plt", SHT_PROGBITS, SHF_WRITE + SHF_ALLOC, 0x600858, 0x858, 0x28, 0x0, 0x0, 0x8, 0x8, 0x28},
			{".data", SHT_PROGBITS, SHF_WRITE + SHF_ALLOC, 0x600880, 0x880, 0x18, 0x0, 0x0, 0x8, 0x0, 0x18},
			{".bss", SHT_NOBITS, SHF_WRITE + SHF_ALLOC, 0x600898, 0x898, 0x8, 0x0, 0x0, 0x4, 0x0, 0x8},
			{".comment", SHT_PROGBITS, 0x0, 0x0, 0x898, 0x126, 0x0, 0x0, 0x1, 0x0, 0x126},
			{".debug_aranges", SHT_PROGBITS, 0x0, 0x0, 0x9c0, 0x90, 0x0, 0x0, 0x10, 0x0, 0x90},
			{".debug_pubnames", SHT_PROGBITS, 0x0, 0x0, 0xa50, 0x25, 0x0, 0x0, 0x1, 0x0, 0x25},
			{".debug_info", SHT_PROGBITS, 0x0, 0x0, 0xa75, 0x1a7, 0x0, 0x0, 0x1, 0x0, 0x1a7},
			{".debug_abbrev", SHT_PROGBITS, 0x0, 0x0, 0xc1c, 0x6f, 0x0, 0x0, 0x1, 0x0, 0x6f},
			{".debug_line", SHT_PROGBITS, 0x0, 0x0, 0xc8b, 0x13f, 0x0, 0x0, 0x1, 0x0, 0x13f},
			{".debug_str", SHT_PROGBITS, SHF_MERGE + SHF_STRINGS, 0x0, 0xdca, 0xb1, 0x0, 0x0, 0x1, 0x1, 0xb1},
			{".debug_ranges", SHT_PROGBITS, 0x0, 0x0, 0xe80, 0x90, 0x0, 0x0, 0x10, 0x0, 0x90},
			{".shstrtab", SHT_STRTAB, 0x0, 0x0, 0xf10, 0x149, 0x0, 0x0, 0x1, 0x0, 0x149},
			{".symtab", SHT_SYMTAB, 0x0, 0x0, 0x19a0, 0x6f0, 0x24, 0x39, 0x8, 0x18, 0x6f0},
			{".strtab", SHT_STRTAB, 0x0, 0x0, 0x2090, 0x1fc, 0x0, 0x0, 0x1, 0x0, 0x1fc},
		},
		[]ProgHeader{
			{PT_PHDR, PF_R + PF_X, 0x40, 0x400040, 0x400040, 0x1c0, 0x1c0, 0x8},
			{PT_INTERP, PF_R, 0x200, 0x400200, 0x400200, 0x1c, 0x1c, 1},
			{PT_LOAD, PF_R + PF_X, 0x0, 0x400000, 0x400000, 0x684, 0x684, 0x200000},
			{PT_LOAD, PF_R + PF_W, 0x688, 0x600688, 0x600688, 0x210, 0x218, 0x200000},
			{PT_DYNAMIC, PF_R + PF_W, 0x6b0, 0x6006b0, 0x6006b0, 0x1a0, 0x1a0, 0x8},
			{PT_NOTE, PF_R, 0x21c, 0x40021c, 0x40021c, 0x20, 0x20, 0x4},
			{PT_LOOS + 0x474E550, PF_R, 0x5b8, 0x4005b8, 0x4005b8, 0x24, 0x24, 0x4},
			{PT_LOOS + 0x474E551, PF_R + PF_W, 0x0, 0x0, 0x0, 0x0, 0x0, 0x8},
		},
		[]string{"libc.so.6"},
	},
	{
		"testdata/hello-world-core.gz",
		FileHeader{ELFCLASS64, ELFDATA2LSB, EV_CURRENT, ELFOSABI_NONE, 0x0, binary.LittleEndian, ET_CORE, EM_X86_64, 0x0},
		[]SectionHeader{},
		[]ProgHeader{
			{Type: PT_NOTE, Flags: 0x0, Off: 0x3f8, Vaddr: 0x0, Paddr: 0x0, Filesz: 0x8ac, Memsz: 0x0, Align: 0x0},
			{Type: PT_LOAD, Flags: PF_X + PF_R, Off: 0x1000, Vaddr: 0x400000, Paddr: 0x0, Filesz: 0x0, Memsz: 0x1000, Align: 0x1000},
			{Type: PT_LOAD, Flags: PF_R, Off: 0x1000, Vaddr: 0x401000, Paddr: 0x0, Filesz: 0x1000, Memsz: 0x1000, Align: 0x1000},
			{Type: PT_LOAD, Flags: PF_W + PF_R, Off: 0x2000, Vaddr: 0x402000, Paddr: 0x0, Filesz: 0x1000, Memsz: 0x1000, Align: 0x1000},
			{Type: PT_LOAD, Flags: PF_X + PF_R, Off: 0x3000, Vaddr: 0x7f54078b8000, Paddr: 0x0, Filesz: 0x0, Memsz: 0x1b5000, Align: 0x1000},
			{Type: PT_LOAD, Flags: 0x0, Off: 0x3000, Vaddr: 0x7f5407a6d000, Paddr: 0x0, Filesz: 0x0, Memsz: 0x1ff000, Align: 0x1000},
			{Type: PT_LOAD, Flags: PF_R, Off: 0x3000, Vaddr: 0x7f5407c6c000, Paddr: 0x0, Filesz: 0x4000, Memsz: 0x4000, Align: 0x1000},
			{Type: PT_LOAD, Flags: PF_W + PF_R, Off: 0x7000, Vaddr: 0x7f5407c70000, Paddr: 0x0, Filesz: 0x2000, Memsz: 0x2000, Align: 0x1000},
			{Type: PT_LOAD, Flags: PF_W + PF_R, Off: 0x9000, Vaddr: 0x7f5407c72000, Paddr: 0x0, Filesz: 0x5000, Memsz: 0x5000, Align: 0x1000},
			{Type: PT_LOAD, Flags: PF_X + PF_R, Off: 0xe000, Vaddr: 0x7f5407c77000, Paddr: 0x0, Filesz: 0x0, Memsz: 0x22000, Align: 0x1000},
			{Type: PT_LOAD, Flags: PF_W + PF_R, Off: 0xe000, Vaddr: 0x7f5407e81000, Paddr: 0x0, Filesz: 0x3000, Memsz: 0x3000, Align: 0x1000},
			{Type: PT_LOAD, Flags: PF_W + PF_R, Off: 0x11000, Vaddr: 0x7f5407e96000, Paddr: 0x0, Filesz: 0x3000, Memsz: 0x3000, Align: 0x1000},
			{Type: PT_LOAD, Flags: PF_R, Off: 0x14000, Vaddr: 0x7f5407e99000, Paddr: 0x0, Filesz: 0x1000, Memsz: 0x1000, Align: 0x1000},
			{Type: PT_LOAD, Flags: PF_W + PF_R, Off: 0x15000, Vaddr: 0x7f5407e9a000, Paddr: 0x0, Filesz: 0x2000, Memsz: 0x2000, Align: 0x1000},
			{Type: PT_LOAD, Flags: PF_W + PF_R, Off: 0x17000, Vaddr: 0x7fff79972000, Paddr: 0x0, Filesz: 0x23000, Memsz: 0x23000, Align: 0x1000},
			{Type: PT_LOAD, Flags: PF_X + PF_R, Off: 0x3a000, Vaddr: 0x7fff799f8000, Paddr: 0x0, Filesz: 0x1000, Memsz: 0x1000, Align: 0x1000},
			{Type: PT_LOAD, Flags: PF_X + PF_R, Off: 0x3b000, Vaddr: 0xffffffffff600000, Paddr: 0x0, Filesz: 0x1000, Memsz: 0x1000, Align: 0x1000},
		},
		nil,
	},
	{
		"testdata/compressed-32.obj",
		FileHeader{ELFCLASS32, ELFDATA2LSB, EV_CURRENT, ELFOSABI_NONE, 0x0, binary.LittleEndian, ET_REL, EM_386, 0x0},
		[]SectionHeader{
			{"", SHT_NULL, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
			{".text", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, 0x0, 0x34, 0x17, 0x0, 0x0, 0x1, 0x0, 0x17},
			{".rel.text", SHT_REL, SHF_INFO_LINK, 0x0, 0x3dc, 0x10, 0x13, 0x1, 0x4, 0x8, 0x10},
			{".data", SHT_PROGBITS, SHF_WRITE | SHF_ALLOC, 0x0, 0x4b, 0x0, 0x0, 0x0, 0x1, 0x0, 0x0},
			{".bss", SHT_NOBITS, SHF_WRITE | SHF_ALLOC, 0x0, 0x4b, 0x0, 0x0, 0x0, 0x1, 0x0, 0x0},
			{".rodata", SHT_PROGBITS, SHF_ALLOC, 0x0, 0x4b, 0xd, 0x0, 0x0, 0x1, 0x0, 0xd},
			{".debug_info", SHT_PROGBITS, SHF_COMPRESSED, 0x0, 0x58, 0xb4, 0x0, 0x0, 0x1, 0x0, 0x84},
			{".rel.debug_info", SHT_REL, SHF_INFO_LINK, 0x0, 0x3ec, 0xa0, 0x13, 0x6, 0x4, 0x8, 0xa0},
			{".debug_abbrev", SHT_PROGBITS, 0x0, 0x0, 0xdc, 0x5a, 0x0, 0x0, 0x1, 0x0, 0x5a},
			{".debug_aranges", SHT_PROGBITS, 0x0, 0x0, 0x136, 0x20, 0x0, 0x0, 0x1, 0x0, 0x20},
			{".rel.debug_aranges", SHT_REL, SHF_INFO_LINK, 0x0, 0x48c, 0x10, 0x13, 0x9, 0x4, 0x8, 0x10},
			{".debug_line", SHT_PROGBITS, 0x0, 0x0, 0x156, 0x5c, 0x0, 0x0, 0x1, 0x0, 0x5c},
			{".rel.debug_line", SHT_REL, SHF_INFO_LINK, 0x0, 0x49c, 0x8, 0x13, 0xb, 0x4, 0x8, 0x8},
			{".debug_str", SHT_PROGBITS, SHF_MERGE | SHF_STRINGS | SHF_COMPRESSED, 0x0, 0x1b2, 0x10f, 0x0, 0x0, 0x1, 0x1, 0xb3},
			{".comment", SHT_PROGBITS, SHF_MERGE | SHF_STRINGS, 0x0, 0x265, 0x2a, 0x0, 0x0, 0x1, 0x1, 0x2a},
			{".note.GNU-stack", SHT_PROGBITS, 0x0, 0x0, 0x28f, 0x0, 0x0, 0x0, 0x1, 0x0, 0x0},
			{".eh_frame", SHT_PROGBITS, SHF_ALLOC, 0x0, 0x290, 0x38, 0x0, 0x0, 0x4, 0x0, 0x38},
			{".rel.eh_frame", SHT_REL, SHF_INFO_LINK, 0x0, 0x4a4, 0x8, 0x13, 0x10, 0x4, 0x8, 0x8},
			{".shstrtab", SHT_STRTAB, 0x0, 0x0, 0x4ac, 0xab, 0x0, 0x0, 0x1, 0x0, 0xab},
			{".symtab", SHT_SYMTAB, 0x0, 0x0, 0x2c8, 0x100, 0x14, 0xe, 0x4, 0x10, 0x100},
			{".strtab", SHT_STRTAB, 0x0, 0x0, 0x3c8, 0x13, 0x0, 0x0, 0x1, 0x0, 0x13},
		},
		[]ProgHeader{},
		nil,
	},
	{
		"testdata/compressed-64.obj",
		FileHeader{ELFCLASS64, ELFDATA2LSB, EV_CURRENT, ELFOSABI_NONE, 0x0, binary.LittleEndian, ET_REL, EM_X86_64, 0x0},
		[]SectionHeader{
			{"", SHT_NULL, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
			{".text", SHT_PROGBITS, SHF_ALLOC | SHF_EXECINSTR, 0x0, 0x40, 0x1b, 0x0, 0x0, 0x1, 0x0, 0x1b},
			{".rela.text", SHT_RELA, SHF_INFO_LINK, 0x0, 0x488, 0x30, 0x13, 0x1, 0x8, 0x18, 0x30},
			{".data", SHT_PROGBITS, SHF_WRITE | SHF_ALLOC, 0x0, 0x5b, 0x0, 0x0, 0x0, 0x1, 0x0, 0x0},
			{".bss", SHT_NOBITS, SHF_WRITE | SHF_ALLOC, 0x0, 0x5b, 0x0, 0x0, 0x0, 0x1, 0x0, 0x0},
			{".rodata", SHT_PROGBITS, SHF_ALLOC, 0x0, 0x5b, 0xd, 0x0, 0x0, 0x1, 0x0, 0xd},
			{".debug_info", SHT_PROGBITS, SHF_COMPRESSED, 0x0, 0x68, 0xba, 0x0, 0x0, 0x1, 0x0, 0x72},
			{".rela.debug_info", SHT_RELA, SHF_INFO_LINK, 0x0, 0x4b8, 0x1c8, 0x13, 0x6, 0x8, 0x18, 0x1c8},
			{".debug_abbrev", SHT_PROGBITS, 0x0, 0x0, 0xda, 0x5c, 0x0, 0x0, 0x1, 0x0, 0x5c},
			{".debug_aranges", SHT_PROGBITS, SHF_COMPRESSED, 0x0, 0x136, 0x30, 0x0, 0x0, 0x1, 0x0, 0x2f},
			{".rela.debug_aranges", SHT_RELA, SHF_INFO_LINK, 0x0, 0x680, 0x30, 0x13, 0x9, 0x8, 0x18, 0x30},
			{".debug_line", SHT_PROGBITS, 0x0, 0x0, 0x165, 0x60, 0x0, 0x0, 0x1, 0x0, 0x60},
			{".rela.debug_line", SHT_RELA, SHF_INFO_LINK, 0x0, 0x6b0, 0x18, 0x13, 0xb, 0x8, 0x18, 0x18},
			{".debug_str", SHT_PROGBITS, SHF_MERGE | SHF_STRINGS | SHF_COMPRESSED, 0x0, 0x1c5, 0x104, 0x0, 0x0, 0x1, 0x1, 0xc3},
			{".comment", SHT_PROGBITS, SHF_MERGE | SHF_STRINGS, 0x0, 0x288, 0x2a, 0x0, 0x0, 0x1, 0x1, 0x2a},
			{".note.GNU-stack", SHT_PROGBITS, 0x0, 0x0, 0x2b2, 0x0, 0x0, 0x0, 0x1, 0x0, 0x0},
			{".eh_frame", SHT_PROGBITS, SHF_ALLOC, 0x0, 0x2b8, 0x38, 0x0, 0x0, 0x8, 0x0, 0x38},
			{".rela.eh_frame", SHT_RELA, SHF_INFO_LINK, 0x0, 0x6c8, 0x18, 0x13, 0x10, 0x8, 0x18, 0x18},
			{".shstrtab", SHT_STRTAB, 0x0, 0x0, 0x6e0, 0xb0, 0x0, 0x0, 0x1, 0x0, 0xb0},
			{".symtab", SHT_SYMTAB, 0x0, 0x0, 0x2f0, 0x180, 0x14, 0xe, 0x8, 0x18, 0x180},
			{".strtab", SHT_STRTAB, 0x0, 0x0, 0x470, 0x13, 0x0, 0x0, 0x1, 0x0, 0x13},
		},
		[]ProgHeader{},
		nil,
	},
}

func TestOpen(t *testing.T) {
	for i := range fileTests {
		tt := &fileTests[i]

		var f *File
		var err error
		if path.Ext(tt.file) == ".gz" {
			var r io.ReaderAt
			if r, err = decompress(tt.file); err == nil {
				f, err = NewFile(r)
			}
		} else {
			f, err = Open(tt.file)
		}
		if err != nil {
			t.Errorf("cannot open file %s: %v", tt.file, err)
			continue
		}
		defer f.Close()
		if !reflect.DeepEqual(f.FileHeader, tt.hdr) {
			t.Errorf("open %s:\n\thave %#v\n\twant %#v\n", tt.file, f.FileHeader, tt.hdr)
			continue
		}
		for i, s := range f.Sections {
			if i >= len(tt.sections) {
				break
			}
			sh := &tt.sections[i]
			if !reflect.DeepEqual(&s.SectionHeader, sh) {
				t.Errorf("open %s, section %d:\n\thave %#v\n\twant %#v\n", tt.file, i, &s.SectionHeader, sh)
			}
		}
		for i, p := range f.Progs {
			if i >= len(tt.progs) {
				break
			}
			ph := &tt.progs[i]
			if !reflect.DeepEqual(&p.ProgHeader, ph) {
				t.Errorf("open %s, program %d:\n\thave %#v\n\twant %#v\n", tt.file, i, &p.ProgHeader, ph)
			}
		}
		tn := len(tt.sections)
		fn := len(f.Sections)
		if tn != fn {
			t.Errorf("open %s: len(Sections) = %d, want %d", tt.file, fn, tn)
		}
		tn = len(tt.progs)
		fn = len(f.Progs)
		if tn != fn {
			t.Errorf("open %s: len(Progs) = %d, want %d", tt.file, fn, tn)
		}
		tl := tt.needed
		fl, err := f.ImportedLibraries()
		if err != nil {
			t.Error(err)
		}
		if !reflect.DeepEqual(tl, fl) {
			t.Errorf("open %s: DT_NEEDED = %v, want %v", tt.file, tl, fl)
		}
	}
}

// elf.NewFile requires io.ReaderAt, which compress/gzip cannot
// provide. Decompress the file to a bytes.Reader.
func decompress(gz string) (io.ReaderAt, error) {
	in, err := os.Open(gz)
	if err != nil {
		return nil, err
	}
	defer in.Close()
	r, err := gzip.NewReader(in)
	if err != nil {
		return nil, err
	}
	var out bytes.Buffer
	_, err = io.Copy(&out, r)
	return bytes.NewReader(out.Bytes()), err
}

type relocationTestEntry struct {
	entryNumber int
	entry       *dwarf.Entry
}

type relocationTest struct {
	file    string
	entries []relocationTestEntry
}

var relocationTests = []relocationTest{
	{
		"testdata/go-relocation-test-gcc441-x86-64.obj",
		[]relocationTestEntry{
			{0, &dwarf.Entry{
				Offset:   0xb,
				Tag:      dwarf.TagCompileUnit,
				Children: true,
				Field: []dwarf.Field{
					{Attr: dwarf.AttrProducer, Val: "GNU C 4.4.1", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLanguage, Val: int64(1), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrName, Val: "go-relocation-test.c", Class: dwarf.ClassString},
					{Attr: dwarf.AttrCompDir, Val: "/tmp", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLowpc, Val: uint64(0x0), Class: dwarf.ClassAddress},
					{Attr: dwarf.AttrHighpc, Val: uint64(0x6), Class: dwarf.ClassAddress},
					{Attr: dwarf.AttrStmtList, Val: int64(0), Class: dwarf.ClassLinePtr},
				},
			}},
		},
	},
	{
		"testdata/go-relocation-test-gcc441-x86.obj",
		[]relocationTestEntry{
			{0, &dwarf.Entry{
				Offset:   0xb,
				Tag:      dwarf.TagCompileUnit,
				Children: true,
				Field: []dwarf.Field{
					{Attr: dwarf.AttrProducer, Val: "GNU C 4.4.1", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLanguage, Val: int64(1), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrName, Val: "t.c", Class: dwarf.ClassString},
					{Attr: dwarf.AttrCompDir, Val: "/tmp", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLowpc, Val: uint64(0x0), Class: dwarf.ClassAddress},
					{Attr: dwarf.AttrHighpc, Val: uint64(0x5), Class: dwarf.ClassAddress},
					{Attr: dwarf.AttrStmtList, Val: int64(0), Class: dwarf.ClassLinePtr},
				},
			}},
		},
	},
	{
		"testdata/go-relocation-test-gcc424-x86-64.obj",
		[]relocationTestEntry{
			{0, &dwarf.Entry{
				Offset:   0xb,
				Tag:      dwarf.TagCompileUnit,
				Children: true,
				Field: []dwarf.Field{
					{Attr: dwarf.AttrProducer, Val: "GNU C 4.2.4 (Ubuntu 4.2.4-1ubuntu4)", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLanguage, Val: int64(1), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrName, Val: "go-relocation-test-gcc424.c", Class: dwarf.ClassString},
					{Attr: dwarf.AttrCompDir, Val: "/tmp", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLowpc, Val: uint64(0x0), Class: dwarf.ClassAddress},
					{Attr: dwarf.AttrHighpc, Val: uint64(0x6), Class: dwarf.ClassAddress},
					{Attr: dwarf.AttrStmtList, Val: int64(0), Class: dwarf.ClassLinePtr},
				},
			}},
		},
	},
	{
		"testdata/go-relocation-test-gcc482-aarch64.obj",
		[]relocationTestEntry{
			{0, &dwarf.Entry{
				Offset:   0xb,
				Tag:      dwarf.TagCompileUnit,
				Children: true,
				Field: []dwarf.Field{
					{Attr: dwarf.AttrProducer, Val: "GNU C 4.8.2 -g -fstack-protector", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLanguage, Val: int64(1), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrName, Val: "go-relocation-test-gcc482.c", Class: dwarf.ClassString},
					{Attr: dwarf.AttrCompDir, Val: "/tmp", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLowpc, Val: uint64(0x0), Class: dwarf.ClassAddress},
					{Attr: dwarf.AttrHighpc, Val: int64(0x24), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrStmtList, Val: int64(0), Class: dwarf.ClassLinePtr},
				},
			}},
		},
	},
	{
		"testdata/go-relocation-test-gcc492-arm.obj",
		[]relocationTestEntry{
			{0, &dwarf.Entry{
				Offset:   0xb,
				Tag:      dwarf.TagCompileUnit,
				Children: true,
				Field: []dwarf.Field{
					{Attr: dwarf.AttrProducer, Val: "GNU C 4.9.2 20141224 (prerelease) -march=armv7-a -mfloat-abi=hard -mfpu=vfpv3-d16 -mtls-dialect=gnu -g", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLanguage, Val: int64(1), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrName, Val: "go-relocation-test-gcc492.c", Class: dwarf.ClassString},
					{Attr: dwarf.AttrCompDir, Val: "/root/go/src/debug/elf/testdata", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLowpc, Val: uint64(0x0), Class: dwarf.ClassAddress},
					{Attr: dwarf.AttrHighpc, Val: int64(0x28), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrStmtList, Val: int64(0), Class: dwarf.ClassLinePtr},
				},
			}},
		},
	},
	{
		"testdata/go-relocation-test-clang-arm.obj",
		[]relocationTestEntry{
			{0, &dwarf.Entry{
				Offset:   0xb,
				Tag:      dwarf.TagCompileUnit,
				Children: true,
				Field: []dwarf.Field{
					{Attr: dwarf.AttrProducer, Val: "Debian clang version 3.5.0-10 (tags/RELEASE_350/final) (based on LLVM 3.5.0)", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLanguage, Val: int64(12), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrName, Val: "hello.c", Class: dwarf.ClassString},
					{Attr: dwarf.AttrStmtList, Val: int64(0x0), Class: dwarf.ClassLinePtr},
					{Attr: dwarf.AttrCompDir, Val: "/tmp", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLowpc, Val: uint64(0x0), Class: dwarf.ClassAddress},
					{Attr: dwarf.AttrHighpc, Val: int64(48), Class: dwarf.ClassConstant},
				},
			}},
		},
	},
	{
		"testdata/go-relocation-test-gcc5-ppc.obj",
		[]relocationTestEntry{
			{0, &dwarf.Entry{
				Offset:   0xb,
				Tag:      dwarf.TagCompileUnit,
				Children: true,
				Field: []dwarf.Field{
					{Attr: dwarf.AttrProducer, Val: "GNU C11 5.0.0 20150116 (experimental) -Asystem=linux -Asystem=unix -Asystem=posix -g", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLanguage, Val: int64(12), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrName, Val: "go-relocation-test-gcc5-ppc.c", Class: dwarf.ClassString},
					{Attr: dwarf.AttrCompDir, Val: "/tmp", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLowpc, Val: uint64(0x0), Class: dwarf.ClassAddress},
					{Attr: dwarf.AttrHighpc, Val: int64(0x44), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrStmtList, Val: int64(0), Class: dwarf.ClassLinePtr},
				},
			}},
		},
	},
	{
		"testdata/go-relocation-test-gcc482-ppc64le.obj",
		[]relocationTestEntry{
			{0, &dwarf.Entry{
				Offset:   0xb,
				Tag:      dwarf.TagCompileUnit,
				Children: true,
				Field: []dwarf.Field{
					{Attr: dwarf.AttrProducer, Val: "GNU C 4.8.2 -Asystem=linux -Asystem=unix -Asystem=posix -msecure-plt -mtune=power8 -mcpu=power7 -gdwarf-2 -fstack-protector", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLanguage, Val: int64(1), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrName, Val: "go-relocation-test-gcc482-ppc64le.c", Class: dwarf.ClassString},
					{Attr: dwarf.AttrCompDir, Val: "/tmp", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLowpc, Val: uint64(0x0), Class: dwarf.ClassAddress},
					{Attr: dwarf.AttrHighpc, Val: uint64(0x24), Class: dwarf.ClassAddress},
					{Attr: dwarf.AttrStmtList, Val: int64(0), Class: dwarf.ClassLinePtr},
				},
			}},
		},
	},
	{
		"testdata/go-relocation-test-gcc492-mips64.obj",
		[]relocationTestEntry{
			{0, &dwarf.Entry{
				Offset:   0xb,
				Tag:      dwarf.TagCompileUnit,
				Children: true,
				Field: []dwarf.Field{
					{Attr: dwarf.AttrProducer, Val: "GNU C 4.9.2 -meb -mabi=64 -march=mips3 -mtune=mips64 -mllsc -mno-shared -g", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLanguage, Val: int64(1), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrName, Val: "hello.c", Class: dwarf.ClassString},
					{Attr: dwarf.AttrCompDir, Val: "/tmp", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLowpc, Val: uint64(0x0), Class: dwarf.ClassAddress},
					{Attr: dwarf.AttrHighpc, Val: int64(100), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrStmtList, Val: int64(0), Class: dwarf.ClassLinePtr},
				},
			}},
		},
	},
	{
		"testdata/go-relocation-test-gcc531-s390x.obj",
		[]relocationTestEntry{
			{0, &dwarf.Entry{
				Offset:   0xb,
				Tag:      dwarf.TagCompileUnit,
				Children: true,
				Field: []dwarf.Field{
					{Attr: dwarf.AttrProducer, Val: "GNU C11 5.3.1 20160316 -march=zEC12 -m64 -mzarch -g -fstack-protector-strong", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLanguage, Val: int64(12), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrName, Val: "hello.c", Class: dwarf.ClassString},
					{Attr: dwarf.AttrCompDir, Val: "/tmp", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLowpc, Val: uint64(0x0), Class: dwarf.ClassAddress},
					{Attr: dwarf.AttrHighpc, Val: int64(58), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrStmtList, Val: int64(0), Class: dwarf.ClassLinePtr},
				},
			}},
		},
	},
	{
		"testdata/go-relocation-test-gcc620-sparc64.obj",
		[]relocationTestEntry{
			{0, &dwarf.Entry{
				Offset:   0xb,
				Tag:      dwarf.TagCompileUnit,
				Children: true,
				Field: []dwarf.Field{
					{Attr: dwarf.AttrProducer, Val: "GNU C11 6.2.0 20160914 -mcpu=v9 -g -fstack-protector-strong", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLanguage, Val: int64(12), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrName, Val: "hello.c", Class: dwarf.ClassString},
					{Attr: dwarf.AttrCompDir, Val: "/tmp", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLowpc, Val: uint64(0x0), Class: dwarf.ClassAddress},
					{Attr: dwarf.AttrHighpc, Val: int64(0x2c), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrStmtList, Val: int64(0), Class: dwarf.ClassLinePtr},
				},
			}},
		},
	},
	{
		"testdata/go-relocation-test-gcc492-mipsle.obj",
		[]relocationTestEntry{
			{0, &dwarf.Entry{
				Offset:   0xb,
				Tag:      dwarf.TagCompileUnit,
				Children: true,
				Field: []dwarf.Field{
					{Attr: dwarf.AttrProducer, Val: "GNU C 4.9.2 -mel -march=mips2 -mtune=mips32 -mllsc -mno-shared -mabi=32 -g", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLanguage, Val: int64(1), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrName, Val: "hello.c", Class: dwarf.ClassString},
					{Attr: dwarf.AttrCompDir, Val: "/tmp", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLowpc, Val: uint64(0x0), Class: dwarf.ClassAddress},
					{Attr: dwarf.AttrHighpc, Val: int64(0x58), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrStmtList, Val: int64(0), Class: dwarf.ClassLinePtr},
				},
			}},
		},
	},
	{
		"testdata/go-relocation-test-gcc540-mips.obj",
		[]relocationTestEntry{
			{0, &dwarf.Entry{
				Offset:   0xb,
				Tag:      dwarf.TagCompileUnit,
				Children: true,
				Field: []dwarf.Field{
					{Attr: dwarf.AttrProducer, Val: "GNU C11 5.4.0 20160609 -meb -mips32 -mtune=mips32r2 -mfpxx -mllsc -mno-shared -mabi=32 -g -gdwarf-2", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLanguage, Val: int64(12), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrName, Val: "hello.c", Class: dwarf.ClassString},
					{Attr: dwarf.AttrCompDir, Val: "/tmp", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLowpc, Val: uint64(0x0), Class: dwarf.ClassAddress},
					{Attr: dwarf.AttrHighpc, Val: uint64(0x5c), Class: dwarf.ClassAddress},
					{Attr: dwarf.AttrStmtList, Val: int64(0), Class: dwarf.ClassLinePtr},
				},
			}},
		},
	},
	{
		"testdata/go-relocation-test-gcc493-mips64le.obj",
		[]relocationTestEntry{
			{0, &dwarf.Entry{
				Offset:   0xb,
				Tag:      dwarf.TagCompileUnit,
				Children: true,
				Field: []dwarf.Field{
					{Attr: dwarf.AttrProducer, Val: "GNU C 4.9.3 -mel -mabi=64 -mllsc -mno-shared -g -fstack-protector-strong", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLanguage, Val: int64(1), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrName, Val: "hello.c", Class: dwarf.ClassString},
					{Attr: dwarf.AttrCompDir, Val: "/tmp", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLowpc, Val: uint64(0x0), Class: dwarf.ClassAddress},
					{Attr: dwarf.AttrHighpc, Val: int64(100), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrStmtList, Val: int64(0), Class: dwarf.ClassLinePtr},
				},
			}},
		},
	},
	{
		"testdata/go-relocation-test-gcc720-riscv64.obj",
		[]relocationTestEntry{
			{0, &dwarf.Entry{
				Offset:   0xb,
				Tag:      dwarf.TagCompileUnit,
				Children: true,
				Field: []dwarf.Field{
					{Attr: dwarf.AttrProducer, Val: "GNU C11 7.2.0 -march=rv64imafdc -mabi=lp64d -g -gdwarf-2", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLanguage, Val: int64(12), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrName, Val: "hello.c", Class: dwarf.ClassString},
					{Attr: dwarf.AttrCompDir, Val: "/tmp", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLowpc, Val: uint64(0x0), Class: dwarf.ClassAddress},
					{Attr: dwarf.AttrHighpc, Val: uint64(0x2c), Class: dwarf.ClassAddress},
					{Attr: dwarf.AttrStmtList, Val: int64(0), Class: dwarf.ClassLinePtr},
				},
			}},
		},
	},
	{
		"testdata/go-relocation-test-clang-x86.obj",
		[]relocationTestEntry{
			{0, &dwarf.Entry{
				Offset:   0xb,
				Tag:      dwarf.TagCompileUnit,
				Children: true,
				Field: []dwarf.Field{
					{Attr: dwarf.AttrProducer, Val: "clang version google3-trunk (trunk r209387)", Class: dwarf.ClassString},
					{Attr: dwarf.AttrLanguage, Val: int64(12), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrName, Val: "go-relocation-test-clang.c", Class: dwarf.ClassString},
					{Attr: dwarf.AttrStmtList, Val: int64(0), Class: dwarf.ClassLinePtr},
					{Attr: dwarf.AttrCompDir, Val: "/tmp", Class: dwarf.ClassString},
				},
			}},
		},
	},
	{
		"testdata/gcc-amd64-openbsd-debug-with-rela.obj",
		[]relocationTestEntry{
			{203, &dwarf.Entry{
				Offset:   0xc62,
				Tag:      dwarf.TagMember,
				Children: false,
				Field: []dwarf.Field{
					{Attr: dwarf.AttrName, Val: "it_interval", Class: dwarf.ClassString},
					{Attr: dwarf.AttrDeclFile, Val: int64(7), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrDeclLine, Val: int64(236), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrType, Val: dwarf.Offset(0xb7f), Class: dwarf.ClassReference},
					{Attr: dwarf.AttrDataMemberLoc, Val: []byte{0x23, 0x0}, Class: dwarf.ClassExprLoc},
				},
			}},
			{204, &dwarf.Entry{
				Offset:   0xc70,
				Tag:      dwarf.TagMember,
				Children: false,
				Field: []dwarf.Field{
					{Attr: dwarf.AttrName, Val: "it_value", Class: dwarf.ClassString},
					{Attr: dwarf.AttrDeclFile, Val: int64(7), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrDeclLine, Val: int64(237), Class: dwarf.ClassConstant},
					{Attr: dwarf.AttrType, Val: dwarf.Offset(0xb7f), Class: dwarf.ClassReference},
					{Attr: dwarf.AttrDataMemberLoc, Val: []byte{0x23, 0x10}, Class: dwarf.ClassExprLoc},
				},
			}},
		},
	},
}

func TestDWARFRelocations(t *testing.T) {
	for i, test := range relocationTests {
		f, err := Open(test.file)
		if err != nil {
			t.Error(err)
			continue
		}
		dwarf, err := f.DWARF()
		if err != nil {
			t.Error(err)
			continue
		}
		for _, testEntry := range test.entries {
			reader := dwarf.Reader()
			for j := 0; j < testEntry.entryNumber; j++ {
				entry, err := reader.Next()
				if entry == nil || err != nil {
					t.Errorf("Failed to skip to entry %d: %v", testEntry.entryNumber, err)
					continue
				}
			}
			entry, err := reader.Next()
			if err != nil {
				t.Error(err)
				continue
			}
			if !reflect.DeepEqual(testEntry.entry, entry) {
				t.Errorf("#%d/%d: mismatch: got:%#v want:%#v", i, testEntry.entryNumber, entry, testEntry.entry)
				continue
			}
		}
	}
}

func TestCompressedDWARF(t *testing.T) {
	// Test file built with GCC 4.8.4 and as 2.24 using:
	// gcc -Wa,--compress-debug-sections -g -c -o zdebug-test-gcc484-x86-64.obj hello.c
	f, err := Open("testdata/zdebug-test-gcc484-x86-64.obj")
	if err != nil {
		t.Fatal(err)
	}
	dwarf, err := f.DWARF()
	if err != nil {
		t.Fatal(err)
	}
	reader := dwarf.Reader()
	n := 0
	for {
		entry, err := reader.Next()
		if err != nil {
			t.Fatal(err)
		}
		if entry == nil {
			break
		}
		n++
	}
	if n != 18 {
		t.Fatalf("want %d DWARF entries, got %d", 18, n)
	}
}

func TestCompressedSection(t *testing.T) {
	// Test files built with gcc -g -S hello.c and assembled with
	// --compress-debug-sections=zlib-gabi.
	f, err := Open("testdata/compressed-64.obj")
	if err != nil {
		t.Fatal(err)
	}
	sec := f.Section(".debug_info")
	wantData := []byte{
		182, 0, 0, 0, 4, 0, 0, 0, 0, 0, 8, 1, 0, 0, 0, 0,
		1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 8, 7,
		0, 0, 0, 0, 2, 1, 8, 0, 0, 0, 0, 2, 2, 7, 0, 0,
		0, 0, 2, 4, 7, 0, 0, 0, 0, 2, 1, 6, 0, 0, 0, 0,
		2, 2, 5, 0, 0, 0, 0, 3, 4, 5, 105, 110, 116, 0, 2, 8,
		5, 0, 0, 0, 0, 2, 8, 7, 0, 0, 0, 0, 4, 8, 114, 0,
		0, 0, 2, 1, 6, 0, 0, 0, 0, 5, 0, 0, 0, 0, 1, 4,
		0, 0, 0, 0, 0, 0, 0, 0, 27, 0, 0, 0, 0, 0, 0, 0,
		1, 156, 179, 0, 0, 0, 6, 0, 0, 0, 0, 1, 4, 87, 0, 0,
		0, 2, 145, 108, 6, 0, 0, 0, 0, 1, 4, 179, 0, 0, 0, 2,
		145, 96, 0, 4, 8, 108, 0, 0, 0, 0,
	}

	// Test Data method.
	b, err := sec.Data()
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(wantData, b) {
		t.Fatalf("want data %x, got %x", wantData, b)
	}

	// Test Open method and seeking.
	buf, have, count := make([]byte, len(b)), make([]bool, len(b)), 0
	sf := sec.Open()
	if got, err := sf.Seek(0, io.SeekEnd); got != int64(len(b)) || err != nil {
		t.Fatalf("want seek end %d, got %d error %v", len(b), got, err)
	}
	if n, err := sf.Read(buf); n != 0 || err != io.EOF {
		t.Fatalf("want EOF with 0 bytes, got %v with %d bytes", err, n)
	}
	pos := int64(len(buf))
	for count < len(buf) {
		// Construct random seek arguments.
		whence := rand.Intn(3)
		target := rand.Int63n(int64(len(buf)))
		var offset int64
		switch whence {
		case io.SeekStart:
			offset = target
		case io.SeekCurrent:
			offset = target - pos
		case io.SeekEnd:
			offset = target - int64(len(buf))
		}
		pos, err = sf.Seek(offset, whence)
		if err != nil {
			t.Fatal(err)
		}
		if pos != target {
			t.Fatalf("want position %d, got %d", target, pos)
		}

		// Read data from the new position.
		end := pos + 16
		if end > int64(len(buf)) {
			end = int64(len(buf))
		}
		n, err := io.ReadFull(sf, buf[pos:end])
		if err != nil {
			t.Fatal(err)
		}
		for i := 0; i < n; i++ {
			if !have[pos] {
				have[pos] = true
				count++
			}
			pos++
		}
	}
	if !bytes.Equal(wantData, buf) {
		t.Fatalf("want data %x, got %x", wantData, buf)
	}
}

func TestNoSectionOverlaps(t *testing.T) {
	// Ensure cmd/link outputs sections without overlaps.
	switch runtime.GOOS {
	case "aix", "android", "darwin", "js", "nacl", "plan9", "windows":
		t.Skipf("cmd/link doesn't produce ELF binaries on %s", runtime.GOOS)
	}
	_ = net.ResolveIPAddr // force dynamic linkage
	f, err := Open(os.Args[0])
	if err != nil {
		t.Error(err)
		return
	}
	for i, si := range f.Sections {
		sih := si.SectionHeader
		if sih.Type == SHT_NOBITS {
			continue
		}
		for j, sj := range f.Sections {
			sjh := sj.SectionHeader
			if i == j || sjh.Type == SHT_NOBITS || sih.Offset == sjh.Offset && sih.Size == 0 {
				continue
			}
			if sih.Offset >= sjh.Offset && sih.Offset < sjh.Offset+sjh.Size {
				t.Errorf("ld produced ELF with section %s within %s: 0x%x <= 0x%x..0x%x < 0x%x",
					sih.Name, sjh.Name, sjh.Offset, sih.Offset, sih.Offset+sih.Size, sjh.Offset+sjh.Size)
			}
		}
	}
}
