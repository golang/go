// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package elf

import (
	"bytes"
	"compress/gzip"
	"compress/zlib"
	"debug/dwarf"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"math/rand"
	"net"
	"os"
	"path"
	"reflect"
	"runtime"
	"slices"
	"strings"
	"testing"
)

type fileTest struct {
	file     string
	hdr      FileHeader
	sections []SectionHeader
	progs    []ProgHeader
	needed   []string
	symbols  []Symbol
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
		[]Symbol{
			{"", 3, 0, false, 0, 1, 134512852, 0, "", ""},
			{"", 3, 0, false, 0, 2, 134512876, 0, "", ""},
			{"", 3, 0, false, 0, 3, 134513020, 0, "", ""},
			{"", 3, 0, false, 0, 4, 134513292, 0, "", ""},
			{"", 3, 0, false, 0, 5, 134513480, 0, "", ""},
			{"", 3, 0, false, 0, 6, 134513512, 0, "", ""},
			{"", 3, 0, false, 0, 7, 134513532, 0, "", ""},
			{"", 3, 0, false, 0, 8, 134513612, 0, "", ""},
			{"", 3, 0, false, 0, 9, 134513996, 0, "", ""},
			{"", 3, 0, false, 0, 10, 134514008, 0, "", ""},
			{"", 3, 0, false, 0, 11, 134518268, 0, "", ""},
			{"", 3, 0, false, 0, 12, 134518280, 0, "", ""},
			{"", 3, 0, false, 0, 13, 134518284, 0, "", ""},
			{"", 3, 0, false, 0, 14, 134518436, 0, "", ""},
			{"", 3, 0, false, 0, 15, 134518444, 0, "", ""},
			{"", 3, 0, false, 0, 16, 134518452, 0, "", ""},
			{"", 3, 0, false, 0, 17, 134518456, 0, "", ""},
			{"", 3, 0, false, 0, 18, 134518484, 0, "", ""},
			{"", 3, 0, false, 0, 19, 0, 0, "", ""},
			{"", 3, 0, false, 0, 20, 0, 0, "", ""},
			{"", 3, 0, false, 0, 21, 0, 0, "", ""},
			{"", 3, 0, false, 0, 22, 0, 0, "", ""},
			{"", 3, 0, false, 0, 23, 0, 0, "", ""},
			{"", 3, 0, false, 0, 24, 0, 0, "", ""},
			{"", 3, 0, false, 0, 25, 0, 0, "", ""},
			{"", 3, 0, false, 0, 26, 0, 0, "", ""},
			{"", 3, 0, false, 0, 27, 0, 0, "", ""},
			{"", 3, 0, false, 0, 28, 0, 0, "", ""},
			{"", 3, 0, false, 0, 29, 0, 0, "", ""},
			{"crt1.c", 4, 0, false, 0, 65521, 0, 0, "", ""},
			{"/usr/src/lib/csu/i386-elf/crti.S", 4, 0, false, 0, 65521, 0, 0, "", ""},
			{"<command line>", 4, 0, false, 0, 65521, 0, 0, "", ""},
			{"<built-in>", 4, 0, false, 0, 65521, 0, 0, "", ""},
			{"/usr/src/lib/csu/i386-elf/crti.S", 4, 0, false, 0, 65521, 0, 0, "", ""},
			{"crtstuff.c", 4, 0, false, 0, 65521, 0, 0, "", ""},
			{"__CTOR_LIST__", 1, 0, false, 0, 14, 134518436, 0, "", ""},
			{"__DTOR_LIST__", 1, 0, false, 0, 15, 134518444, 0, "", ""},
			{"__EH_FRAME_BEGIN__", 1, 0, false, 0, 12, 134518280, 0, "", ""},
			{"__JCR_LIST__", 1, 0, false, 0, 16, 134518452, 0, "", ""},
			{"p.0", 1, 0, false, 0, 11, 134518276, 0, "", ""},
			{"completed.1", 1, 0, false, 0, 18, 134518484, 1, "", ""},
			{"__do_global_dtors_aux", 2, 0, false, 0, 8, 134513760, 0, "", ""},
			{"object.2", 1, 0, false, 0, 18, 134518488, 24, "", ""},
			{"frame_dummy", 2, 0, false, 0, 8, 134513836, 0, "", ""},
			{"crtstuff.c", 4, 0, false, 0, 65521, 0, 0, "", ""},
			{"__CTOR_END__", 1, 0, false, 0, 14, 134518440, 0, "", ""},
			{"__DTOR_END__", 1, 0, false, 0, 15, 134518448, 0, "", ""},
			{"__FRAME_END__", 1, 0, false, 0, 12, 134518280, 0, "", ""},
			{"__JCR_END__", 1, 0, false, 0, 16, 134518452, 0, "", ""},
			{"__do_global_ctors_aux", 2, 0, false, 0, 8, 134513960, 0, "", ""},
			{"/usr/src/lib/csu/i386-elf/crtn.S", 4, 0, false, 0, 65521, 0, 0, "", ""},
			{"<command line>", 4, 0, false, 0, 65521, 0, 0, "", ""},
			{"<built-in>", 4, 0, false, 0, 65521, 0, 0, "", ""},
			{"/usr/src/lib/csu/i386-elf/crtn.S", 4, 0, false, 0, 65521, 0, 0, "", ""},
			{"hello.c", 4, 0, false, 0, 65521, 0, 0, "", ""},
			{"printf", 18, 0, false, 0, 0, 0, 44, "", ""},
			{"_DYNAMIC", 17, 0, false, 0, 65521, 134518284, 0, "", ""},
			{"__dso_handle", 17, 2, false, 0, 11, 134518272, 0, "", ""},
			{"_init", 18, 0, false, 0, 6, 134513512, 0, "", ""},
			{"environ", 17, 0, false, 0, 18, 134518512, 4, "", ""},
			{"__deregister_frame_info", 32, 0, false, 0, 0, 0, 0, "", ""},
			{"__progname", 17, 0, false, 0, 11, 134518268, 4, "", ""},
			{"_start", 18, 0, false, 0, 8, 134513612, 145, "", ""},
			{"__bss_start", 16, 0, false, 0, 65521, 134518484, 0, "", ""},
			{"main", 18, 0, false, 0, 8, 134513912, 46, "", ""},
			{"_init_tls", 18, 0, false, 0, 0, 0, 5, "", ""},
			{"_fini", 18, 0, false, 0, 9, 134513996, 0, "", ""},
			{"atexit", 18, 0, false, 0, 0, 0, 43, "", ""},
			{"_edata", 16, 0, false, 0, 65521, 134518484, 0, "", ""},
			{"_GLOBAL_OFFSET_TABLE_", 17, 0, false, 0, 65521, 134518456, 0, "", ""},
			{"_end", 16, 0, false, 0, 65521, 134518516, 0, "", ""},
			{"exit", 18, 0, false, 0, 0, 0, 68, "", ""},
			{"_Jv_RegisterClasses", 32, 0, false, 0, 0, 0, 0, "", ""},
			{"__register_frame_info", 32, 0, false, 0, 0, 0, 0, "", ""},
		},
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
		[]Symbol{
			{"", 3, 0, false, 0, 1, 4194816, 0, "", ""},
			{"", 3, 0, false, 0, 2, 4194844, 0, "", ""},
			{"", 3, 0, false, 0, 3, 4194880, 0, "", ""},
			{"", 3, 0, false, 0, 4, 4194920, 0, "", ""},
			{"", 3, 0, false, 0, 5, 4194952, 0, "", ""},
			{"", 3, 0, false, 0, 6, 4195048, 0, "", ""},
			{"", 3, 0, false, 0, 7, 4195110, 0, "", ""},
			{"", 3, 0, false, 0, 8, 4195120, 0, "", ""},
			{"", 3, 0, false, 0, 9, 4195152, 0, "", ""},
			{"", 3, 0, false, 0, 10, 4195176, 0, "", ""},
			{"", 3, 0, false, 0, 11, 4195224, 0, "", ""},
			{"", 3, 0, false, 0, 12, 4195248, 0, "", ""},
			{"", 3, 0, false, 0, 13, 4195296, 0, "", ""},
			{"", 3, 0, false, 0, 14, 4195732, 0, "", ""},
			{"", 3, 0, false, 0, 15, 4195748, 0, "", ""},
			{"", 3, 0, false, 0, 16, 4195768, 0, "", ""},
			{"", 3, 0, false, 0, 17, 4195808, 0, "", ""},
			{"", 3, 0, false, 0, 18, 6293128, 0, "", ""},
			{"", 3, 0, false, 0, 19, 6293144, 0, "", ""},
			{"", 3, 0, false, 0, 20, 6293160, 0, "", ""},
			{"", 3, 0, false, 0, 21, 6293168, 0, "", ""},
			{"", 3, 0, false, 0, 22, 6293584, 0, "", ""},
			{"", 3, 0, false, 0, 23, 6293592, 0, "", ""},
			{"", 3, 0, false, 0, 24, 6293632, 0, "", ""},
			{"", 3, 0, false, 0, 25, 6293656, 0, "", ""},
			{"", 3, 0, false, 0, 26, 0, 0, "", ""},
			{"", 3, 0, false, 0, 27, 0, 0, "", ""},
			{"", 3, 0, false, 0, 28, 0, 0, "", ""},
			{"", 3, 0, false, 0, 29, 0, 0, "", ""},
			{"", 3, 0, false, 0, 30, 0, 0, "", ""},
			{"", 3, 0, false, 0, 31, 0, 0, "", ""},
			{"", 3, 0, false, 0, 32, 0, 0, "", ""},
			{"", 3, 0, false, 0, 33, 0, 0, "", ""},
			{"init.c", 4, 0, false, 0, 65521, 0, 0, "", ""},
			{"initfini.c", 4, 0, false, 0, 65521, 0, 0, "", ""},
			{"call_gmon_start", 2, 0, false, 0, 13, 4195340, 0, "", ""},
			{"crtstuff.c", 4, 0, false, 0, 65521, 0, 0, "", ""},
			{"__CTOR_LIST__", 1, 0, false, 0, 18, 6293128, 0, "", ""},
			{"__DTOR_LIST__", 1, 0, false, 0, 19, 6293144, 0, "", ""},
			{"__JCR_LIST__", 1, 0, false, 0, 20, 6293160, 0, "", ""},
			{"__do_global_dtors_aux", 2, 0, false, 0, 13, 4195376, 0, "", ""},
			{"completed.6183", 1, 0, false, 0, 25, 6293656, 1, "", ""},
			{"p.6181", 1, 0, false, 0, 24, 6293648, 0, "", ""},
			{"frame_dummy", 2, 0, false, 0, 13, 4195440, 0, "", ""},
			{"crtstuff.c", 4, 0, false, 0, 65521, 0, 0, "", ""},
			{"__CTOR_END__", 1, 0, false, 0, 18, 6293136, 0, "", ""},
			{"__DTOR_END__", 1, 0, false, 0, 19, 6293152, 0, "", ""},
			{"__FRAME_END__", 1, 0, false, 0, 17, 4195968, 0, "", ""},
			{"__JCR_END__", 1, 0, false, 0, 20, 6293160, 0, "", ""},
			{"__do_global_ctors_aux", 2, 0, false, 0, 13, 4195680, 0, "", ""},
			{"initfini.c", 4, 0, false, 0, 65521, 0, 0, "", ""},
			{"hello.c", 4, 0, false, 0, 65521, 0, 0, "", ""},
			{"_GLOBAL_OFFSET_TABLE_", 1, 2, false, 0, 23, 6293592, 0, "", ""},
			{"__init_array_end", 0, 2, false, 0, 18, 6293124, 0, "", ""},
			{"__init_array_start", 0, 2, false, 0, 18, 6293124, 0, "", ""},
			{"_DYNAMIC", 1, 2, false, 0, 21, 6293168, 0, "", ""},
			{"data_start", 32, 0, false, 0, 24, 6293632, 0, "", ""},
			{"__libc_csu_fini", 18, 0, false, 0, 13, 4195520, 2, "", ""},
			{"_start", 18, 0, false, 0, 13, 4195296, 0, "", ""},
			{"__gmon_start__", 32, 0, false, 0, 0, 0, 0, "", ""},
			{"_Jv_RegisterClasses", 32, 0, false, 0, 0, 0, 0, "", ""},
			{"puts@@GLIBC_2.2.5", 18, 0, false, 0, 0, 0, 396, "", ""},
			{"_fini", 18, 0, false, 0, 14, 4195732, 0, "", ""},
			{"__libc_start_main@@GLIBC_2.2.5", 18, 0, false, 0, 0, 0, 450, "", ""},
			{"_IO_stdin_used", 17, 0, false, 0, 15, 4195748, 4, "", ""},
			{"__data_start", 16, 0, false, 0, 24, 6293632, 0, "", ""},
			{"__dso_handle", 17, 2, false, 0, 24, 6293640, 0, "", ""},
			{"__libc_csu_init", 18, 0, false, 0, 13, 4195536, 137, "", ""},
			{"__bss_start", 16, 0, false, 0, 65521, 6293656, 0, "", ""},
			{"_end", 16, 0, false, 0, 65521, 6293664, 0, "", ""},
			{"_edata", 16, 0, false, 0, 65521, 6293656, 0, "", ""},
			{"main", 18, 0, false, 0, 13, 4195480, 27, "", ""},
			{"_init", 18, 0, false, 0, 11, 4195224, 0, "", ""},
		},
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
		[]Symbol{
			{"hello.c", 4, 0, false, 0, 65521, 0, 0, "", ""},
			{"", 3, 0, false, 0, 1, 0, 0, "", ""},
			{"", 3, 0, false, 0, 3, 0, 0, "", ""},
			{"", 3, 0, false, 0, 4, 0, 0, "", ""},
			{"", 3, 0, false, 0, 5, 0, 0, "", ""},
			{"", 3, 0, false, 0, 6, 0, 0, "", ""},
			{"", 3, 0, false, 0, 8, 0, 0, "", ""},
			{"", 3, 0, false, 0, 9, 0, 0, "", ""},
			{"", 3, 0, false, 0, 11, 0, 0, "", ""},
			{"", 3, 0, false, 0, 13, 0, 0, "", ""},
			{"", 3, 0, false, 0, 15, 0, 0, "", ""},
			{"", 3, 0, false, 0, 16, 0, 0, "", ""},
			{"", 3, 0, false, 0, 14, 0, 0, "", ""},
			{"main", 18, 0, false, 0, 1, 0, 23, "", ""},
			{"puts", 16, 0, false, 0, 0, 0, 0, "", ""},
		},
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
		[]Symbol{
			{"hello.c", 4, 0, false, 0, 65521, 0, 0, "", ""},
			{"", 3, 0, false, 0, 1, 0, 0, "", ""},
			{"", 3, 0, false, 0, 3, 0, 0, "", ""},
			{"", 3, 0, false, 0, 4, 0, 0, "", ""},
			{"", 3, 0, false, 0, 5, 0, 0, "", ""},
			{"", 3, 0, false, 0, 6, 0, 0, "", ""},
			{"", 3, 0, false, 0, 8, 0, 0, "", ""},
			{"", 3, 0, false, 0, 9, 0, 0, "", ""},
			{"", 3, 0, false, 0, 11, 0, 0, "", ""},
			{"", 3, 0, false, 0, 13, 0, 0, "", ""},
			{"", 3, 0, false, 0, 15, 0, 0, "", ""},
			{"", 3, 0, false, 0, 16, 0, 0, "", ""},
			{"", 3, 0, false, 0, 14, 0, 0, "", ""},
			{"main", 18, 0, false, 0, 1, 0, 27, "", ""},
			{"puts", 16, 0, false, 0, 0, 0, 0, "", ""},
		},
	},
	{
		"testdata/go-relocation-test-gcc620-sparc64.obj",
		FileHeader{Class: ELFCLASS64, Data: ELFDATA2MSB, Version: EV_CURRENT, OSABI: ELFOSABI_NONE, ABIVersion: 0x0, ByteOrder: binary.BigEndian, Type: ET_REL, Machine: EM_SPARCV9, Entry: 0x0},
		[]SectionHeader{
			{"", SHT_NULL, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
			{".text", SHT_PROGBITS, SHF_ALLOC + SHF_EXECINSTR, 0x0, 0x40, 0x2c, 0x0, 0x0, 0x4, 0x0, 0x2c},
			{".rela.text", SHT_RELA, SHF_INFO_LINK, 0x0, 0xa58, 0x48, 0x13, 0x1, 0x8, 0x18, 0x48},
			{".data", SHT_PROGBITS, SHF_WRITE + SHF_ALLOC, 0x0, 0x6c, 0x0, 0x0, 0x0, 0x1, 0x0, 0x0},
			{".bss", SHT_NOBITS, SHF_WRITE + SHF_ALLOC, 0x0, 0x6c, 0x0, 0x0, 0x0, 0x1, 0x0, 0x0},
			{".rodata", SHT_PROGBITS, SHF_ALLOC, 0x0, 0x70, 0xd, 0x0, 0x0, 0x8, 0x0, 0xd},
			{".debug_info", SHT_PROGBITS, 0x0, 0x0, 0x7d, 0x346, 0x0, 0x0, 0x1, 0x0, 0x346},
			{".rela.debug_info", SHT_RELA, SHF_INFO_LINK, 0x0, 0xaa0, 0x630, 0x13, 0x6, 0x8, 0x18, 0x630},
			{".debug_abbrev", SHT_PROGBITS, 0x0, 0x0, 0x3c3, 0xf1, 0x0, 0x0, 0x1, 0x0, 0xf1},
			{".debug_aranges", SHT_PROGBITS, 0x0, 0x0, 0x4b4, 0x30, 0x0, 0x0, 0x1, 0x0, 0x30},
			{".rela.debug_aranges", SHT_RELA, SHF_INFO_LINK, 0x0, 0x10d0, 0x30, 0x13, 0x9, 0x8, 0x18, 0x30},
			{".debug_line", SHT_PROGBITS, 0x0, 0x0, 0x4e4, 0xd3, 0x0, 0x0, 0x1, 0x0, 0xd3},
			{".rela.debug_line", SHT_RELA, SHF_INFO_LINK, 0x0, 0x1100, 0x18, 0x13, 0xb, 0x8, 0x18, 0x18},
			{".debug_str", SHT_PROGBITS, SHF_MERGE + SHF_STRINGS, 0x0, 0x5b7, 0x2a3, 0x0, 0x0, 0x1, 0x1, 0x2a3},
			{".comment", SHT_PROGBITS, SHF_MERGE + SHF_STRINGS, 0x0, 0x85a, 0x2e, 0x0, 0x0, 0x1, 0x1, 0x2e},
			{".note.GNU-stack", SHT_PROGBITS, 0x0, 0x0, 0x888, 0x0, 0x0, 0x0, 0x1, 0x0, 0x0},
			{".debug_frame", SHT_PROGBITS, 0x0, 0x0, 0x888, 0x38, 0x0, 0x0, 0x8, 0x0, 0x38},
			{".rela.debug_frame", SHT_RELA, SHF_INFO_LINK, 0x0, 0x1118, 0x30, 0x13, 0x10, 0x8, 0x18, 0x30},
			{".shstrtab", SHT_STRTAB, 0x0, 0x0, 0x1148, 0xb3, 0x0, 0x0, 0x1, 0x0, 0xb3},
			{".symtab", SHT_SYMTAB, 0x0, 0x0, 0x8c0, 0x180, 0x14, 0xe, 0x8, 0x18, 0x180},
			{".strtab", SHT_STRTAB, 0x0, 0x0, 0xa40, 0x13, 0x0, 0x0, 0x1, 0x0, 0x13},
		},
		[]ProgHeader{},
		nil,
		[]Symbol{
			{"hello.c", 4, 0, false, 0, 65521, 0, 0, "", ""},
			{"", 3, 0, false, 0, 1, 0, 0, "", ""},
			{"", 3, 0, false, 0, 3, 0, 0, "", ""},
			{"", 3, 0, false, 0, 4, 0, 0, "", ""},
			{"", 3, 0, false, 0, 5, 0, 0, "", ""},
			{"", 3, 0, false, 0, 6, 0, 0, "", ""},
			{"", 3, 0, false, 0, 8, 0, 0, "", ""},
			{"", 3, 0, false, 0, 9, 0, 0, "", ""},
			{"", 3, 0, false, 0, 11, 0, 0, "", ""},
			{"", 3, 0, false, 0, 13, 0, 0, "", ""},
			{"", 3, 0, false, 0, 15, 0, 0, "", ""},
			{"", 3, 0, false, 0, 16, 0, 0, "", ""},
			{"", 3, 0, false, 0, 14, 0, 0, "", ""},
			{"main", 18, 0, false, 0, 1, 0, 44, "", ""},
			{"puts", 16, 0, false, 0, 0, 0, 0, "", ""},
		},
	},
	{
		"testdata/gcc-riscv64-linux-exec",
		FileHeader{ELFCLASS64, ELFDATA2LSB, EV_CURRENT, ELFOSABI_NONE, 0, binary.LittleEndian, ET_EXEC, EM_RISCV, 0x10460},
		[]SectionHeader{
			{"", SHT_NULL, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0},
			{".interp", SHT_PROGBITS, SHF_ALLOC, 0x10270, 0x270, 0x21, 0x0, 0x0, 0x1, 0x0, 0x21},
			{".note.gnu.build-id", SHT_NOTE, SHF_ALLOC, 0x10294, 0x294, 0x24, 0x0, 0x0, 0x4, 0x0, 0x24},
			{".note.ABI-tag", SHT_NOTE, SHF_ALLOC, 0x102b8, 0x2b8, 0x20, 0x0, 0x0, 0x4, 0x0, 0x20},
			{".gnu.hash", SHT_GNU_HASH, SHF_ALLOC, 0x102d8, 0x2d8, 0x30, 0x5, 0x0, 0x8, 0x0, 0x30},
			{".dynsym", SHT_DYNSYM, SHF_ALLOC, 0x10308, 0x308, 0x60, 0x6, 0x1, 0x8, 0x18, 0x60},
			{".dynstr", SHT_STRTAB, SHF_ALLOC, 0x10368, 0x368, 0x4a, 0x0, 0x0, 0x1, 0x0, 0x4a},
			{".gnu.version", SHT_GNU_VERSYM, SHF_ALLOC, 0x103b2, 0x3b2, 0x8, 0x5, 0x0, 0x2, 0x2, 0x8},
			{".gnu.version_r", SHT_GNU_VERNEED, SHF_ALLOC, 0x103c0, 0x3c0, 0x30, 0x6, 0x1, 0x8, 0x0, 0x30},
			{".rela.plt", SHT_RELA, SHF_ALLOC + SHF_INFO_LINK, 0x103f0, 0x3f0, 0x30, 0x5, 0x14, 0x8, 0x18, 0x30},
			{".plt", SHT_PROGBITS, SHF_ALLOC + SHF_EXECINSTR, 0x10420, 0x420, 0x40, 0x0, 0x0, 0x10, 0x10, 0x40},
			{".text", SHT_PROGBITS, SHF_ALLOC + SHF_EXECINSTR, 0x10460, 0x460, 0xd8, 0x0, 0x0, 0x4, 0x0, 0xd8},
			{".rodata", SHT_PROGBITS, SHF_ALLOC, 0x10538, 0x538, 0x15, 0x0, 0x0, 0x8, 0x0, 0x15},
			{".eh_frame_hdr", SHT_PROGBITS, SHF_ALLOC, 0x10550, 0x550, 0x24, 0x0, 0x0, 0x4, 0x0, 0x24},
			{".eh_frame", SHT_PROGBITS, SHF_ALLOC, 0x10578, 0x578, 0x6c, 0x0, 0x0, 0x8, 0x0, 0x6c},
			{".preinit_array", SHT_PREINIT_ARRAY, SHF_WRITE + SHF_ALLOC, 0x11e00, 0xe00, 0x8, 0x0, 0x0, 0x1, 0x8, 0x8},
			{".init_array", SHT_INIT_ARRAY, SHF_WRITE + SHF_ALLOC, 0x11e08, 0xe08, 0x8, 0x0, 0x0, 0x8, 0x8, 0x8},
			{".fini_array", SHT_FINI_ARRAY, SHF_WRITE + SHF_ALLOC, 0x11e10, 0xe10, 0x8, 0x0, 0x0, 0x8, 0x8, 0x8},
			{".dynamic", SHT_DYNAMIC, SHF_WRITE + SHF_ALLOC, 0x11e18, 0xe18, 0x1d0, 0x6, 0x0, 0x8, 0x10, 0x1d0},
			{".got", SHT_PROGBITS, SHF_WRITE + SHF_ALLOC, 0x11fe8, 0xfe8, 0x8, 0x0, 0x0, 0x8, 0x8, 0x8},
			{".got.plt", SHT_PROGBITS, SHF_WRITE + SHF_ALLOC, 0x11ff0, 0xff0, 0x20, 0x0, 0x0, 0x8, 0x8, 0x20},
			{".sdata", SHT_PROGBITS, SHF_WRITE + SHF_ALLOC, 0x12010, 0x1010, 0x8, 0x0, 0x0, 0x8, 0x0, 0x8},
			{".bss", SHT_NOBITS, SHF_WRITE + SHF_ALLOC, 0x12018, 0x1018, 0x8, 0x0, 0x0, 0x1, 0x0, 0x8},
			{".comment", SHT_PROGBITS, SHF_MERGE + SHF_STRINGS, 0x0, 0x1018, 0x26, 0x0, 0x0, 0x1, 0x1, 0x26},
			{".riscv.attributes", SHT_RISCV_ATTRIBUTES, 0x0, 0x0, 0x103e, 0x66, 0x0, 0x0, 0x1, 0x0, 0x66},
			{".shstrtab", SHT_STRTAB, 0x0, 0x0, 0x10a4, 0xff, 0x0, 0x0, 0x1, 0x0, 0xff},
		},
		[]ProgHeader{
			{PT_PHDR, PF_R, 0x40, 0x10040, 0x10040, 0x230, 0x230, 0x8},
			{PT_INTERP, PF_R, 0x270, 0x10270, 0x10270, 0x21, 0x21, 0x1},
			{PT_RISCV_ATTRIBUTES, PF_R, 0x103e, 0x0, 0x0, 0x66, 0x0, 0x1},
			{PT_LOAD, PF_X + PF_R, 0x0, 0x10000, 0x10000, 0x5e4, 0x5e4, 0x1000},
			{PT_LOAD, PF_W + PF_R, 0xe00, 0x11e00, 0x11e00, 0x218, 0x220, 0x1000},
			{PT_DYNAMIC, PF_W + PF_R, 0xe18, 0x11e18, 0x11e18, 0x1d0, 0x1d0, 0x8},
			{PT_NOTE, PF_R, 0x294, 0x10294, 0x10294, 0x44, 0x44, 0x4},
			{PT_GNU_EH_FRAME, PF_R, 0x550, 0x10550, 0x10550, 0x24, 0x24, 0x4},
			{PT_GNU_STACK, PF_W + PF_R, 0x0, 0x0, 0x0, 0x0, 0x0, 0x10},
			{PT_GNU_RELRO, PF_R, 0xe00, 0x11e00, 0x11e00, 0x200, 0x200, 0x1},
		},
		[]string{"libc.so.6"},
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
		if f.FileHeader != tt.hdr {
			t.Errorf("open %s:\n\thave %#v\n\twant %#v\n", tt.file, f.FileHeader, tt.hdr)
			continue
		}
		for i, s := range f.Sections {
			if i >= len(tt.sections) {
				break
			}
			sh := tt.sections[i]
			if s.SectionHeader != sh {
				t.Errorf("open %s, section %d:\n\thave %#v\n\twant %#v\n", tt.file, i, s.SectionHeader, sh)
			}
		}
		for i, p := range f.Progs {
			if i >= len(tt.progs) {
				break
			}
			ph := tt.progs[i]
			if p.ProgHeader != ph {
				t.Errorf("open %s, program %d:\n\thave %#v\n\twant %#v\n", tt.file, i, p.ProgHeader, ph)
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
		symbols, err := f.Symbols()
		if tt.symbols == nil {
			if !errors.Is(err, ErrNoSymbols) {
				t.Errorf("open %s: Symbols() expected ErrNoSymbols, have nil", tt.file)
			}
			if symbols != nil {
				t.Errorf("open %s: Symbols() expected no symbols, have %v", tt.file, symbols)
			}
		} else {
			if err != nil {
				t.Errorf("open %s: Symbols() unexpected error %v", tt.file, err)
			}
			if !slices.Equal(symbols, tt.symbols) {
				t.Errorf("open %s: Symbols() = %v, want %v", tt.file, symbols, tt.symbols)
			}
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
	pcRanges    [][2]uint64
}

type relocationTest struct {
	file    string
	entries []relocationTestEntry
}

var relocationTests = []relocationTest{
	{
		"testdata/go-relocation-test-gcc441-x86-64.obj",
		[]relocationTestEntry{
			{
				entry: &dwarf.Entry{
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
				},
				pcRanges: [][2]uint64{{0x0, 0x6}},
			},
		},
	},
	{
		"testdata/go-relocation-test-gcc441-x86.obj",
		[]relocationTestEntry{
			{
				entry: &dwarf.Entry{
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
				},
				pcRanges: [][2]uint64{{0x0, 0x5}},
			},
		},
	},
	{
		"testdata/go-relocation-test-gcc424-x86-64.obj",
		[]relocationTestEntry{
			{
				entry: &dwarf.Entry{
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
				},
				pcRanges: [][2]uint64{{0x0, 0x6}},
			},
		},
	},
	{
		"testdata/go-relocation-test-gcc482-aarch64.obj",
		[]relocationTestEntry{
			{
				entry: &dwarf.Entry{
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
				},
				pcRanges: [][2]uint64{{0x0, 0x24}},
			},
		},
	},
	{
		"testdata/go-relocation-test-gcc492-arm.obj",
		[]relocationTestEntry{
			{
				entry: &dwarf.Entry{
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
				},
				pcRanges: [][2]uint64{{0x0, 0x28}},
			},
		},
	},
	{
		"testdata/go-relocation-test-clang-arm.obj",
		[]relocationTestEntry{
			{
				entry: &dwarf.Entry{
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
						{Attr: dwarf.AttrHighpc, Val: int64(0x30), Class: dwarf.ClassConstant},
					},
				},
				pcRanges: [][2]uint64{{0x0, 0x30}},
			},
		},
	},
	{
		"testdata/go-relocation-test-gcc5-ppc.obj",
		[]relocationTestEntry{
			{
				entry: &dwarf.Entry{
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
				},
				pcRanges: [][2]uint64{{0x0, 0x44}},
			},
		},
	},
	{
		"testdata/go-relocation-test-gcc482-ppc64le.obj",
		[]relocationTestEntry{
			{
				entry: &dwarf.Entry{
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
				},
				pcRanges: [][2]uint64{{0x0, 0x24}},
			},
		},
	},
	{
		"testdata/go-relocation-test-gcc492-mips64.obj",
		[]relocationTestEntry{
			{
				entry: &dwarf.Entry{
					Offset:   0xb,
					Tag:      dwarf.TagCompileUnit,
					Children: true,
					Field: []dwarf.Field{
						{Attr: dwarf.AttrProducer, Val: "GNU C 4.9.2 -meb -mabi=64 -march=mips3 -mtune=mips64 -mllsc -mno-shared -g", Class: dwarf.ClassString},
						{Attr: dwarf.AttrLanguage, Val: int64(1), Class: dwarf.ClassConstant},
						{Attr: dwarf.AttrName, Val: "hello.c", Class: dwarf.ClassString},
						{Attr: dwarf.AttrCompDir, Val: "/tmp", Class: dwarf.ClassString},
						{Attr: dwarf.AttrLowpc, Val: uint64(0x0), Class: dwarf.ClassAddress},
						{Attr: dwarf.AttrHighpc, Val: int64(0x64), Class: dwarf.ClassConstant},
						{Attr: dwarf.AttrStmtList, Val: int64(0), Class: dwarf.ClassLinePtr},
					},
				},
				pcRanges: [][2]uint64{{0x0, 0x64}},
			},
		},
	},
	{
		"testdata/go-relocation-test-gcc531-s390x.obj",
		[]relocationTestEntry{
			{
				entry: &dwarf.Entry{
					Offset:   0xb,
					Tag:      dwarf.TagCompileUnit,
					Children: true,
					Field: []dwarf.Field{
						{Attr: dwarf.AttrProducer, Val: "GNU C11 5.3.1 20160316 -march=zEC12 -m64 -mzarch -g -fstack-protector-strong", Class: dwarf.ClassString},
						{Attr: dwarf.AttrLanguage, Val: int64(12), Class: dwarf.ClassConstant},
						{Attr: dwarf.AttrName, Val: "hello.c", Class: dwarf.ClassString},
						{Attr: dwarf.AttrCompDir, Val: "/tmp", Class: dwarf.ClassString},
						{Attr: dwarf.AttrLowpc, Val: uint64(0x0), Class: dwarf.ClassAddress},
						{Attr: dwarf.AttrHighpc, Val: int64(0x3a), Class: dwarf.ClassConstant},
						{Attr: dwarf.AttrStmtList, Val: int64(0), Class: dwarf.ClassLinePtr},
					},
				},
				pcRanges: [][2]uint64{{0x0, 0x3a}},
			},
		},
	},
	{
		"testdata/go-relocation-test-gcc620-sparc64.obj",
		[]relocationTestEntry{
			{
				entry: &dwarf.Entry{
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
				},
				pcRanges: [][2]uint64{{0x0, 0x2c}},
			},
		},
	},
	{
		"testdata/go-relocation-test-gcc492-mipsle.obj",
		[]relocationTestEntry{
			{
				entry: &dwarf.Entry{
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
				},
				pcRanges: [][2]uint64{{0x0, 0x58}},
			},
		},
	},
	{
		"testdata/go-relocation-test-gcc540-mips.obj",
		[]relocationTestEntry{
			{
				entry: &dwarf.Entry{
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
				},
				pcRanges: [][2]uint64{{0x0, 0x5c}},
			},
		},
	},
	{
		"testdata/go-relocation-test-gcc493-mips64le.obj",
		[]relocationTestEntry{
			{
				entry: &dwarf.Entry{
					Offset:   0xb,
					Tag:      dwarf.TagCompileUnit,
					Children: true,
					Field: []dwarf.Field{
						{Attr: dwarf.AttrProducer, Val: "GNU C 4.9.3 -mel -mabi=64 -mllsc -mno-shared -g -fstack-protector-strong", Class: dwarf.ClassString},
						{Attr: dwarf.AttrLanguage, Val: int64(1), Class: dwarf.ClassConstant},
						{Attr: dwarf.AttrName, Val: "hello.c", Class: dwarf.ClassString},
						{Attr: dwarf.AttrCompDir, Val: "/tmp", Class: dwarf.ClassString},
						{Attr: dwarf.AttrLowpc, Val: uint64(0x0), Class: dwarf.ClassAddress},
						{Attr: dwarf.AttrHighpc, Val: int64(0x64), Class: dwarf.ClassConstant},
						{Attr: dwarf.AttrStmtList, Val: int64(0), Class: dwarf.ClassLinePtr},
					},
				},
				pcRanges: [][2]uint64{{0x0, 0x64}},
			},
		},
	},
	{
		"testdata/go-relocation-test-gcc720-riscv64.obj",
		[]relocationTestEntry{
			{
				entry: &dwarf.Entry{
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
				},
				pcRanges: [][2]uint64{{0x0, 0x2c}},
			},
		},
	},
	{
		"testdata/go-relocation-test-clang-x86.obj",
		[]relocationTestEntry{
			{
				entry: &dwarf.Entry{
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
				},
			},
		},
	},
	{
		"testdata/gcc-amd64-openbsd-debug-with-rela.obj",
		[]relocationTestEntry{
			{
				entryNumber: 203,
				entry: &dwarf.Entry{
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
				},
			},
			{
				entryNumber: 204,
				entry: &dwarf.Entry{
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
				},
			},
		},
	},
	{
		"testdata/go-relocation-test-gcc930-ranges-no-rela-x86-64",
		[]relocationTestEntry{
			{
				entry: &dwarf.Entry{
					Offset:   0xb,
					Tag:      dwarf.TagCompileUnit,
					Children: true,
					Field: []dwarf.Field{
						{Attr: dwarf.AttrProducer, Val: "GNU C17 9.3.0 -mtune=generic -march=x86-64 -g -fno-asynchronous-unwind-tables", Class: dwarf.ClassString},
						{Attr: dwarf.AttrLanguage, Val: int64(12), Class: dwarf.ClassConstant},
						{Attr: dwarf.AttrName, Val: "multiple-code-sections.c", Class: dwarf.ClassString},
						{Attr: dwarf.AttrCompDir, Val: "/tmp", Class: dwarf.ClassString},
						{Attr: dwarf.AttrRanges, Val: int64(0), Class: dwarf.ClassRangeListPtr},
						{Attr: dwarf.AttrLowpc, Val: uint64(0), Class: dwarf.ClassAddress},
						{Attr: dwarf.AttrStmtList, Val: int64(0), Class: dwarf.ClassLinePtr},
					},
				},
				pcRanges: [][2]uint64{
					{0x765, 0x777},
					{0x7e1, 0x7ec},
				},
			},
		},
	},
	{
		"testdata/go-relocation-test-gcc930-ranges-with-rela-x86-64",
		[]relocationTestEntry{
			{
				entry: &dwarf.Entry{
					Offset:   0xb,
					Tag:      dwarf.TagCompileUnit,
					Children: true,
					Field: []dwarf.Field{
						{Attr: dwarf.AttrProducer, Val: "GNU C17 9.3.0 -mtune=generic -march=x86-64 -g -fno-asynchronous-unwind-tables", Class: dwarf.ClassString},
						{Attr: dwarf.AttrLanguage, Val: int64(12), Class: dwarf.ClassConstant},
						{Attr: dwarf.AttrName, Val: "multiple-code-sections.c", Class: dwarf.ClassString},
						{Attr: dwarf.AttrCompDir, Val: "/tmp", Class: dwarf.ClassString},
						{Attr: dwarf.AttrRanges, Val: int64(0), Class: dwarf.ClassRangeListPtr},
						{Attr: dwarf.AttrLowpc, Val: uint64(0), Class: dwarf.ClassAddress},
						{Attr: dwarf.AttrStmtList, Val: int64(0), Class: dwarf.ClassLinePtr},
					},
				},
				pcRanges: [][2]uint64{
					{0x765, 0x777},
					{0x7e1, 0x7ec},
				},
			},
		},
	},
}

func TestDWARFRelocations(t *testing.T) {
	for _, test := range relocationTests {
		test := test
		t.Run(test.file, func(t *testing.T) {
			t.Parallel()
			f, err := Open(test.file)
			if err != nil {
				t.Fatal(err)
			}
			dwarf, err := f.DWARF()
			if err != nil {
				t.Fatal(err)
			}
			reader := dwarf.Reader()
			idx := 0
			for _, testEntry := range test.entries {
				if testEntry.entryNumber < idx {
					t.Fatalf("internal test error: %d < %d", testEntry.entryNumber, idx)
				}
				for ; idx < testEntry.entryNumber; idx++ {
					entry, err := reader.Next()
					if entry == nil || err != nil {
						t.Fatalf("Failed to skip to entry %d: %v", testEntry.entryNumber, err)
					}
				}
				entry, err := reader.Next()
				idx++
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(testEntry.entry, entry) {
					t.Errorf("entry %d mismatch: got:%#v want:%#v", testEntry.entryNumber, entry, testEntry.entry)
				}
				pcRanges, err := dwarf.Ranges(entry)
				if err != nil {
					t.Fatal(err)
				}
				if !reflect.DeepEqual(testEntry.pcRanges, pcRanges) {
					t.Errorf("entry %d: PC range mismatch: got:%#v want:%#v", testEntry.entryNumber, pcRanges, testEntry.pcRanges)
				}
			}
		})
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
	case "aix", "android", "darwin", "ios", "js", "plan9", "windows", "wasip1":
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
		// checking for overlap in file
		for j, sj := range f.Sections {
			sjh := sj.SectionHeader
			if i == j || sjh.Type == SHT_NOBITS || sih.Offset == sjh.Offset && sih.FileSize == 0 {
				continue
			}
			if sih.Offset >= sjh.Offset && sih.Offset < sjh.Offset+sjh.FileSize {
				t.Errorf("ld produced ELF with section offset %s within %s: 0x%x <= 0x%x..0x%x < 0x%x",
					sih.Name, sjh.Name, sjh.Offset, sih.Offset, sih.Offset+sih.FileSize, sjh.Offset+sjh.FileSize)
			}
		}

		if sih.Flags&SHF_ALLOC == 0 {
			continue
		}

		// checking for overlap in address space
		for j, sj := range f.Sections {
			sjh := sj.SectionHeader
			if i == j || sjh.Flags&SHF_ALLOC == 0 || sjh.Type == SHT_NOBITS ||
				sih.Addr == sjh.Addr && sih.Size == 0 {
				continue
			}
			if sih.Addr >= sjh.Addr && sih.Addr < sjh.Addr+sjh.Size {
				t.Errorf("ld produced ELF with section address %s within %s: 0x%x <= 0x%x..0x%x < 0x%x",
					sih.Name, sjh.Name, sjh.Addr, sih.Addr, sih.Addr+sih.Size, sjh.Addr+sjh.Size)
			}
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

	wantError := "unexpected read from SHT_NOBITS section"
	bss := f.Section(".bss")

	_, err = bss.Data()
	if err == nil || err.Error() != wantError {
		t.Fatalf("bss.Data() got error %q, want error %q", err, wantError)
	}

	r := bss.Open()
	p := make([]byte, 1)
	_, err = r.Read(p)
	if err == nil || err.Error() != wantError {
		t.Fatalf("r.Read(p) got error %q, want error %q", err, wantError)
	}
}

// TestLargeNumberOfSections tests the case that a file has greater than or
// equal to 65280 (0xff00) sections.
func TestLargeNumberOfSections(t *testing.T) {
	// A file with >= 0xff00 sections is too big, so we will construct it on the
	// fly. The original file "y.o" is generated by these commands:
	// 1. generate "y.c":
	//   for i in `seq 1 65288`; do
	//     printf -v x "%04x" i;
	//     echo "int var_$x __attribute__((section(\"section_$x\"))) = $i;"
	//   done > y.c
	// 2. compile: gcc -c y.c -m32
	//
	// $readelf -h y.o
	// ELF Header:
	//   Magic:   7f 45 4c 46 01 01 01 00 00 00 00 00 00 00 00 00
	//   Class:                             ELF32
	//   Data:                              2's complement, little endian
	//   Version:                           1 (current)
	//   OS/ABI:                            UNIX - System V
	//   ABI Version:                       0
	//   Type:                              REL (Relocatable file)
	//   Machine:                           Intel 80386
	//   Version:                           0x1
	//   Entry point address:               0x0
	//   Start of program headers:          0 (bytes into file)
	//   Start of section headers:          3003468 (bytes into file)
	//   Flags:                             0x0
	//   Size of this header:               52 (bytes)
	//   Size of program headers:           0 (bytes)
	//   Number of program headers:         0
	//   Size of section headers:           40 (bytes)
	//   Number of section headers:         0 (65298)
	//   Section header string table index: 65535 (65297)
	//
	// $readelf -S y.o
	// There are 65298 section headers, starting at offset 0x2dd44c:
	// Section Headers:
	//   [Nr]    Name              Type            Addr     Off    Size   ES Flg Lk Inf Al
	//   [    0]                   NULL            00000000 000000 00ff12 00     65297   0  0
	//   [    1] .text             PROGBITS        00000000 000034 000000 00  AX  0   0  1
	//   [    2] .data             PROGBITS        00000000 000034 000000 00  WA  0   0  1
	//   [    3] .bss              NOBITS          00000000 000034 000000 00  WA  0   0  1
	//   [    4] section_0001      PROGBITS        00000000 000034 000004 00  WA  0   0  4
	//   [    5] section_0002      PROGBITS        00000000 000038 000004 00  WA  0   0  4
	//   [ section_0003 ~ section_ff06 truncated ]
	//   [65290] section_ff07      PROGBITS        00000000 03fc4c 000004 00  WA  0   0  4
	//   [65291] section_ff08      PROGBITS        00000000 03fc50 000004 00  WA  0   0  4
	//   [65292] .comment          PROGBITS        00000000 03fc54 000027 01  MS  0   0  1
	//   [65293] .note.GNU-stack   PROGBITS        00000000 03fc7b 000000 00      0   0  1
	//   [65294] .symtab           SYMTAB          00000000 03fc7c 0ff0a0 10     65296   2  4
	//   [65295] .symtab_shndx     SYMTAB SECTION  00000000 13ed1c 03fc28 04     65294   0  4
	//   [65296] .strtab           STRTAB          00000000 17e944 08f74d 00      0   0  1
	//   [65297] .shstrtab         STRTAB          00000000 20e091 0cf3bb 00      0   0  1

	var buf bytes.Buffer

	{
		buf.Grow(0x55AF1C) // 3003468 + 40 * 65298

		h := Header32{
			Ident:     [16]byte{0x7F, 'E', 'L', 'F', 0x01, 0x01, 0x01},
			Type:      1,
			Machine:   3,
			Version:   1,
			Shoff:     0x2DD44C,
			Ehsize:    0x34,
			Shentsize: 0x28,
			Shnum:     0,
			Shstrndx:  0xFFFF,
		}
		binary.Write(&buf, binary.LittleEndian, h)

		// Zero out sections [1]~[65294].
		buf.Write(bytes.Repeat([]byte{0}, 0x13ED1C-binary.Size(h)))

		// Write section [65295]. Section [65295] are all zeros except for the
		// last 48 bytes.
		buf.Write(bytes.Repeat([]byte{0}, 0x03FC28-12*4))
		for i := 0; i < 12; i++ {
			binary.Write(&buf, binary.LittleEndian, uint32(0xFF00|i))
		}

		// Write section [65296].
		buf.Write([]byte{0})
		buf.Write([]byte("y.c\x00"))
		for i := 1; i <= 65288; i++ {
			// var_0001 ~ var_ff08
			name := fmt.Sprintf("var_%04x", i)
			buf.Write([]byte(name))
			buf.Write([]byte{0})
		}

		// Write section [65297].
		buf.Write([]byte{0})
		buf.Write([]byte(".symtab\x00"))
		buf.Write([]byte(".strtab\x00"))
		buf.Write([]byte(".shstrtab\x00"))
		buf.Write([]byte(".text\x00"))
		buf.Write([]byte(".data\x00"))
		buf.Write([]byte(".bss\x00"))
		for i := 1; i <= 65288; i++ {
			// s_0001 ~ s_ff08
			name := fmt.Sprintf("section_%04x", i)
			buf.Write([]byte(name))
			buf.Write([]byte{0})
		}
		buf.Write([]byte(".comment\x00"))
		buf.Write([]byte(".note.GNU-stack\x00"))
		buf.Write([]byte(".symtab_shndx\x00"))

		// Write section header table.
		// NULL
		binary.Write(&buf, binary.LittleEndian, Section32{Name: 0, Size: 0xFF12, Link: 0xFF11})
		// .text
		binary.Write(&buf, binary.LittleEndian, Section32{
			Name:      0x1B,
			Type:      uint32(SHT_PROGBITS),
			Flags:     uint32(SHF_ALLOC | SHF_EXECINSTR),
			Off:       0x34,
			Addralign: 0x01,
		})
		// .data
		binary.Write(&buf, binary.LittleEndian, Section32{
			Name:      0x21,
			Type:      uint32(SHT_PROGBITS),
			Flags:     uint32(SHF_WRITE | SHF_ALLOC),
			Off:       0x34,
			Addralign: 0x01,
		})
		// .bss
		binary.Write(&buf, binary.LittleEndian, Section32{
			Name:      0x27,
			Type:      uint32(SHT_NOBITS),
			Flags:     uint32(SHF_WRITE | SHF_ALLOC),
			Off:       0x34,
			Addralign: 0x01,
		})
		// s_1 ~ s_65537
		for i := 0; i < 65288; i++ {
			s := Section32{
				Name:      uint32(0x2C + i*13),
				Type:      uint32(SHT_PROGBITS),
				Flags:     uint32(SHF_WRITE | SHF_ALLOC),
				Off:       uint32(0x34 + i*4),
				Size:      0x04,
				Addralign: 0x04,
			}
			binary.Write(&buf, binary.LittleEndian, s)
		}
		// .comment
		binary.Write(&buf, binary.LittleEndian, Section32{
			Name:      0x0CF394,
			Type:      uint32(SHT_PROGBITS),
			Flags:     uint32(SHF_MERGE | SHF_STRINGS),
			Off:       0x03FC54,
			Size:      0x27,
			Addralign: 0x01,
			Entsize:   0x01,
		})
		// .note.GNU-stack
		binary.Write(&buf, binary.LittleEndian, Section32{
			Name:      0x0CF39D,
			Type:      uint32(SHT_PROGBITS),
			Off:       0x03FC7B,
			Addralign: 0x01,
		})
		// .symtab
		binary.Write(&buf, binary.LittleEndian, Section32{
			Name:      0x01,
			Type:      uint32(SHT_SYMTAB),
			Off:       0x03FC7C,
			Size:      0x0FF0A0,
			Link:      0xFF10,
			Info:      0x02,
			Addralign: 0x04,
			Entsize:   0x10,
		})
		// .symtab_shndx
		binary.Write(&buf, binary.LittleEndian, Section32{
			Name:      0x0CF3AD,
			Type:      uint32(SHT_SYMTAB_SHNDX),
			Off:       0x13ED1C,
			Size:      0x03FC28,
			Link:      0xFF0E,
			Addralign: 0x04,
			Entsize:   0x04,
		})
		// .strtab
		binary.Write(&buf, binary.LittleEndian, Section32{
			Name:      0x09,
			Type:      uint32(SHT_STRTAB),
			Off:       0x17E944,
			Size:      0x08F74D,
			Addralign: 0x01,
		})
		// .shstrtab
		binary.Write(&buf, binary.LittleEndian, Section32{
			Name:      0x11,
			Type:      uint32(SHT_STRTAB),
			Off:       0x20E091,
			Size:      0x0CF3BB,
			Addralign: 0x01,
		})
	}

	data := buf.Bytes()

	f, err := NewFile(bytes.NewReader(data))
	if err != nil {
		t.Errorf("cannot create file from data: %v", err)
	}
	defer f.Close()

	wantFileHeader := FileHeader{
		Class:     ELFCLASS32,
		Data:      ELFDATA2LSB,
		Version:   EV_CURRENT,
		OSABI:     ELFOSABI_NONE,
		ByteOrder: binary.LittleEndian,
		Type:      ET_REL,
		Machine:   EM_386,
	}
	if f.FileHeader != wantFileHeader {
		t.Errorf("\nhave %#v\nwant %#v\n", f.FileHeader, wantFileHeader)
	}

	wantSectionNum := 65298
	if len(f.Sections) != wantSectionNum {
		t.Errorf("len(Sections) = %d, want %d", len(f.Sections), wantSectionNum)
	}

	wantSectionHeader := SectionHeader{
		Name:      "section_0007",
		Type:      SHT_PROGBITS,
		Flags:     SHF_WRITE + SHF_ALLOC,
		Offset:    0x4c,
		Size:      0x4,
		Addralign: 0x4,
		FileSize:  0x4,
	}
	if f.Sections[10].SectionHeader != wantSectionHeader {
		t.Errorf("\nhave %#v\nwant %#v\n", f.Sections[10].SectionHeader, wantSectionHeader)
	}
}

func TestIssue10996(t *testing.T) {
	data := []byte("\u007fELF\x02\x01\x010000000000000" +
		"\x010000000000000000000" +
		"\x00\x00\x00\x00\x00\x00\x00\x0000000000\x00\x00\x00\x00" +
		"0000")
	_, err := NewFile(bytes.NewReader(data))
	if err == nil {
		t.Fatalf("opening invalid ELF file unexpectedly succeeded")
	}
}

func TestDynValue(t *testing.T) {
	const testdata = "testdata/gcc-amd64-linux-exec"
	f, err := Open(testdata)
	if err != nil {
		t.Fatalf("could not read %s: %v", testdata, err)
	}
	defer f.Close()

	vals, err := f.DynValue(DT_VERNEEDNUM)
	if err != nil {
		t.Fatalf("DynValue(DT_VERNEEDNUM): got unexpected error %v", err)
	}

	if len(vals) != 1 || vals[0] != 1 {
		t.Errorf("DynValue(DT_VERNEEDNUM): got %v, want [1]", vals)
	}
}

func TestIssue59208(t *testing.T) {
	// corrupted dwarf data should raise invalid dwarf data instead of invalid zlib
	const orig = "testdata/compressed-64.obj"
	f, err := Open(orig)
	if err != nil {
		t.Fatal(err)
	}
	sec := f.Section(".debug_info")

	data, err := os.ReadFile(orig)
	if err != nil {
		t.Fatal(err)
	}

	dn := make([]byte, len(data))
	zoffset := sec.Offset + uint64(sec.compressionOffset)
	copy(dn, data[:zoffset])

	ozd, err := sec.Data()
	if err != nil {
		t.Fatal(err)
	}
	buf := bytes.NewBuffer(nil)
	wr := zlib.NewWriter(buf)
	// corrupt origin data same as COMPRESS_ZLIB
	copy(ozd, []byte{1, 0, 0, 0})
	wr.Write(ozd)
	wr.Close()

	copy(dn[zoffset:], buf.Bytes())
	copy(dn[sec.Offset+sec.FileSize:], data[sec.Offset+sec.FileSize:])

	nf, err := NewFile(bytes.NewReader(dn))
	if err != nil {
		t.Error(err)
	}

	const want = "decoding dwarf section info"
	_, err = nf.DWARF()
	if err == nil || !strings.Contains(err.Error(), want) {
		t.Errorf("DWARF = %v; want %q", err, want)
	}
}

func BenchmarkSymbols64(b *testing.B) {
	const testdata = "testdata/gcc-amd64-linux-exec"
	f, err := Open(testdata)
	if err != nil {
		b.Fatalf("could not read %s: %v", testdata, err)
	}
	defer f.Close()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		symbols, err := f.Symbols()
		if err != nil {
			b.Fatalf("Symbols(): got unexpected error %v", err)
		}
		if len(symbols) != 73 {
			b.Errorf("\nhave %d symbols\nwant %d symbols\n", len(symbols), 73)
		}
	}
}

func BenchmarkSymbols32(b *testing.B) {
	const testdata = "testdata/gcc-386-freebsd-exec"
	f, err := Open(testdata)
	if err != nil {
		b.Fatalf("could not read %s: %v", testdata, err)
	}
	defer f.Close()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		symbols, err := f.Symbols()
		if err != nil {
			b.Fatalf("Symbols(): got unexpected error %v", err)
		}
		if len(symbols) != 74 {
			b.Errorf("\nhave %d symbols\nwant %d symbols\n", len(symbols), 74)
		}
	}
}
