// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"bytes"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"debug/macho"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"
)

type MachoHdr struct {
	cpu    uint32
	subcpu uint32
}

type MachoSect struct {
	name    string
	segname string
	addr    uint64
	size    uint64
	off     uint32
	align   uint32
	reloc   uint32
	nreloc  uint32
	flag    uint32
	res1    uint32
	res2    uint32
}

type MachoSeg struct {
	name       string
	vsize      uint64
	vaddr      uint64
	fileoffset uint64
	filesize   uint64
	prot1      uint32
	prot2      uint32
	nsect      uint32
	msect      uint32
	sect       []MachoSect
	flag       uint32
}

// MachoPlatformLoad represents a LC_VERSION_MIN_* or
// LC_BUILD_VERSION load command.
type MachoPlatformLoad struct {
	platform MachoPlatform // One of PLATFORM_* constants.
	cmd      MachoLoad
}

type MachoLoad struct {
	type_ uint32
	data  []uint32
}

type MachoPlatform int

/*
 * Total amount of space to reserve at the start of the file
 * for Header, PHeaders, and SHeaders.
 * May waste some.
 */
const (
	INITIAL_MACHO_HEADR = 4 * 1024
)

const (
	MACHO_CPU_AMD64               = 1<<24 | 7
	MACHO_CPU_386                 = 7
	MACHO_SUBCPU_X86              = 3
	MACHO_CPU_ARM                 = 12
	MACHO_SUBCPU_ARM              = 0
	MACHO_SUBCPU_ARMV7            = 9
	MACHO_CPU_ARM64               = 1<<24 | 12
	MACHO_SUBCPU_ARM64_ALL        = 0
	MACHO32SYMSIZE                = 12
	MACHO64SYMSIZE                = 16
	MACHO_X86_64_RELOC_UNSIGNED   = 0
	MACHO_X86_64_RELOC_SIGNED     = 1
	MACHO_X86_64_RELOC_BRANCH     = 2
	MACHO_X86_64_RELOC_GOT_LOAD   = 3
	MACHO_X86_64_RELOC_GOT        = 4
	MACHO_X86_64_RELOC_SUBTRACTOR = 5
	MACHO_X86_64_RELOC_SIGNED_1   = 6
	MACHO_X86_64_RELOC_SIGNED_2   = 7
	MACHO_X86_64_RELOC_SIGNED_4   = 8
	MACHO_ARM_RELOC_VANILLA       = 0
	MACHO_ARM_RELOC_PAIR          = 1
	MACHO_ARM_RELOC_SECTDIFF      = 2
	MACHO_ARM_RELOC_BR24          = 5
	MACHO_ARM64_RELOC_UNSIGNED    = 0
	MACHO_ARM64_RELOC_BRANCH26    = 2
	MACHO_ARM64_RELOC_PAGE21      = 3
	MACHO_ARM64_RELOC_PAGEOFF12   = 4
	MACHO_ARM64_RELOC_ADDEND      = 10
	MACHO_GENERIC_RELOC_VANILLA   = 0
	MACHO_FAKE_GOTPCREL           = 100
)

const (
	MH_MAGIC    = 0xfeedface
	MH_MAGIC_64 = 0xfeedfacf

	MH_OBJECT  = 0x1
	MH_EXECUTE = 0x2

	MH_NOUNDEFS = 0x1
)

const (
	LC_SEGMENT                  = 0x1
	LC_SYMTAB                   = 0x2
	LC_SYMSEG                   = 0x3
	LC_THREAD                   = 0x4
	LC_UNIXTHREAD               = 0x5
	LC_LOADFVMLIB               = 0x6
	LC_IDFVMLIB                 = 0x7
	LC_IDENT                    = 0x8
	LC_FVMFILE                  = 0x9
	LC_PREPAGE                  = 0xa
	LC_DYSYMTAB                 = 0xb
	LC_LOAD_DYLIB               = 0xc
	LC_ID_DYLIB                 = 0xd
	LC_LOAD_DYLINKER            = 0xe
	LC_ID_DYLINKER              = 0xf
	LC_PREBOUND_DYLIB           = 0x10
	LC_ROUTINES                 = 0x11
	LC_SUB_FRAMEWORK            = 0x12
	LC_SUB_UMBRELLA             = 0x13
	LC_SUB_CLIENT               = 0x14
	LC_SUB_LIBRARY              = 0x15
	LC_TWOLEVEL_HINTS           = 0x16
	LC_PREBIND_CKSUM            = 0x17
	LC_LOAD_WEAK_DYLIB          = 0x80000018
	LC_SEGMENT_64               = 0x19
	LC_ROUTINES_64              = 0x1a
	LC_UUID                     = 0x1b
	LC_RPATH                    = 0x8000001c
	LC_CODE_SIGNATURE           = 0x1d
	LC_SEGMENT_SPLIT_INFO       = 0x1e
	LC_REEXPORT_DYLIB           = 0x8000001f
	LC_LAZY_LOAD_DYLIB          = 0x20
	LC_ENCRYPTION_INFO          = 0x21
	LC_DYLD_INFO                = 0x22
	LC_DYLD_INFO_ONLY           = 0x80000022
	LC_LOAD_UPWARD_DYLIB        = 0x80000023
	LC_VERSION_MIN_MACOSX       = 0x24
	LC_VERSION_MIN_IPHONEOS     = 0x25
	LC_FUNCTION_STARTS          = 0x26
	LC_DYLD_ENVIRONMENT         = 0x27
	LC_MAIN                     = 0x80000028
	LC_DATA_IN_CODE             = 0x29
	LC_SOURCE_VERSION           = 0x2A
	LC_DYLIB_CODE_SIGN_DRS      = 0x2B
	LC_ENCRYPTION_INFO_64       = 0x2C
	LC_LINKER_OPTION            = 0x2D
	LC_LINKER_OPTIMIZATION_HINT = 0x2E
	LC_VERSION_MIN_TVOS         = 0x2F
	LC_VERSION_MIN_WATCHOS      = 0x30
	LC_VERSION_NOTE             = 0x31
	LC_BUILD_VERSION            = 0x32
)

const (
	S_REGULAR                  = 0x0
	S_ZEROFILL                 = 0x1
	S_NON_LAZY_SYMBOL_POINTERS = 0x6
	S_SYMBOL_STUBS             = 0x8
	S_MOD_INIT_FUNC_POINTERS   = 0x9
	S_ATTR_PURE_INSTRUCTIONS   = 0x80000000
	S_ATTR_DEBUG               = 0x02000000
	S_ATTR_SOME_INSTRUCTIONS   = 0x00000400
)

const (
	PLATFORM_MACOS    MachoPlatform = 1
	PLATFORM_IOS      MachoPlatform = 2
	PLATFORM_TVOS     MachoPlatform = 3
	PLATFORM_WATCHOS  MachoPlatform = 4
	PLATFORM_BRIDGEOS MachoPlatform = 5
)

// Mach-O file writing
// https://developer.apple.com/mac/library/DOCUMENTATION/DeveloperTools/Conceptual/MachORuntime/Reference/reference.html

var machohdr MachoHdr

var load []MachoLoad

var machoPlatform MachoPlatform

var seg [16]MachoSeg

var nseg int

var ndebug int

var nsect int

const (
	SymKindLocal = 0 + iota
	SymKindExtdef
	SymKindUndef
	NumSymKind
)

var nkind [NumSymKind]int

var sortsym []loader.Sym

var nsortsym int

// Amount of space left for adding load commands
// that refer to dynamic libraries. Because these have
// to go in the Mach-O header, we can't just pick a
// "big enough" header size. The initial header is
// one page, the non-dynamic library stuff takes
// up about 1300 bytes; we overestimate that as 2k.
var loadBudget = INITIAL_MACHO_HEADR - 2*1024

func getMachoHdr() *MachoHdr {
	return &machohdr
}

func newMachoLoad(arch *sys.Arch, type_ uint32, ndata uint32) *MachoLoad {
	if arch.PtrSize == 8 && (ndata&1 != 0) {
		ndata++
	}

	load = append(load, MachoLoad{})
	l := &load[len(load)-1]
	l.type_ = type_
	l.data = make([]uint32, ndata)
	return l
}

func newMachoSeg(name string, msect int) *MachoSeg {
	if nseg >= len(seg) {
		Exitf("too many segs")
	}

	s := &seg[nseg]
	nseg++
	s.name = name
	s.msect = uint32(msect)
	s.sect = make([]MachoSect, msect)
	return s
}

func newMachoSect(seg *MachoSeg, name string, segname string) *MachoSect {
	if seg.nsect >= seg.msect {
		Exitf("too many sects in segment %s", seg.name)
	}

	s := &seg.sect[seg.nsect]
	seg.nsect++
	s.name = name
	s.segname = segname
	nsect++
	return s
}

// Generic linking code.

var dylib []string

var linkoff int64

func machowrite(arch *sys.Arch, out *OutBuf, linkmode LinkMode) int {
	o1 := out.Offset()

	loadsize := 4 * 4 * ndebug
	for i := range load {
		loadsize += 4 * (len(load[i].data) + 2)
	}
	if arch.PtrSize == 8 {
		loadsize += 18 * 4 * nseg
		loadsize += 20 * 4 * nsect
	} else {
		loadsize += 14 * 4 * nseg
		loadsize += 17 * 4 * nsect
	}

	if arch.PtrSize == 8 {
		out.Write32(MH_MAGIC_64)
	} else {
		out.Write32(MH_MAGIC)
	}
	out.Write32(machohdr.cpu)
	out.Write32(machohdr.subcpu)
	if linkmode == LinkExternal {
		out.Write32(MH_OBJECT) /* file type - mach object */
	} else {
		out.Write32(MH_EXECUTE) /* file type - mach executable */
	}
	out.Write32(uint32(len(load)) + uint32(nseg) + uint32(ndebug))
	out.Write32(uint32(loadsize))
	if nkind[SymKindUndef] == 0 {
		out.Write32(MH_NOUNDEFS) /* flags - no undefines */
	} else {
		out.Write32(0) /* flags */
	}
	if arch.PtrSize == 8 {
		out.Write32(0) /* reserved */
	}

	for i := 0; i < nseg; i++ {
		s := &seg[i]
		if arch.PtrSize == 8 {
			out.Write32(LC_SEGMENT_64)
			out.Write32(72 + 80*s.nsect)
			out.WriteStringN(s.name, 16)
			out.Write64(s.vaddr)
			out.Write64(s.vsize)
			out.Write64(s.fileoffset)
			out.Write64(s.filesize)
			out.Write32(s.prot1)
			out.Write32(s.prot2)
			out.Write32(s.nsect)
			out.Write32(s.flag)
		} else {
			out.Write32(LC_SEGMENT)
			out.Write32(56 + 68*s.nsect)
			out.WriteStringN(s.name, 16)
			out.Write32(uint32(s.vaddr))
			out.Write32(uint32(s.vsize))
			out.Write32(uint32(s.fileoffset))
			out.Write32(uint32(s.filesize))
			out.Write32(s.prot1)
			out.Write32(s.prot2)
			out.Write32(s.nsect)
			out.Write32(s.flag)
		}

		for j := uint32(0); j < s.nsect; j++ {
			t := &s.sect[j]
			if arch.PtrSize == 8 {
				out.WriteStringN(t.name, 16)
				out.WriteStringN(t.segname, 16)
				out.Write64(t.addr)
				out.Write64(t.size)
				out.Write32(t.off)
				out.Write32(t.align)
				out.Write32(t.reloc)
				out.Write32(t.nreloc)
				out.Write32(t.flag)
				out.Write32(t.res1) /* reserved */
				out.Write32(t.res2) /* reserved */
				out.Write32(0)      /* reserved */
			} else {
				out.WriteStringN(t.name, 16)
				out.WriteStringN(t.segname, 16)
				out.Write32(uint32(t.addr))
				out.Write32(uint32(t.size))
				out.Write32(t.off)
				out.Write32(t.align)
				out.Write32(t.reloc)
				out.Write32(t.nreloc)
				out.Write32(t.flag)
				out.Write32(t.res1) /* reserved */
				out.Write32(t.res2) /* reserved */
			}
		}
	}

	for i := range load {
		l := &load[i]
		out.Write32(l.type_)
		out.Write32(4 * (uint32(len(l.data)) + 2))
		for j := 0; j < len(l.data); j++ {
			out.Write32(l.data[j])
		}
	}

	return int(out.Offset() - o1)
}

func (ctxt *Link) domacho() {
	if *FlagD {
		return
	}

	// Copy platform load command.
	for _, h := range hostobj {
		load, err := hostobjMachoPlatform(&h)
		if err != nil {
			Exitf("%v", err)
		}
		if load != nil {
			machoPlatform = load.platform
			ml := newMachoLoad(ctxt.Arch, load.cmd.type_, uint32(len(load.cmd.data)))
			copy(ml.data, load.cmd.data)
			break
		}
	}
	if machoPlatform == 0 {
		switch ctxt.Arch.Family {
		default:
			machoPlatform = PLATFORM_MACOS
			if ctxt.LinkMode == LinkInternal {
				// For lldb, must say LC_VERSION_MIN_MACOSX or else
				// it won't know that this Mach-O binary is from OS X
				// (could be iOS or WatchOS instead).
				// Go on iOS uses linkmode=external, and linkmode=external
				// adds this itself. So we only need this code for linkmode=internal
				// and we can assume OS X.
				//
				// See golang.org/issues/12941.
				//
				// The version must be at least 10.9; see golang.org/issues/30488.
				ml := newMachoLoad(ctxt.Arch, LC_VERSION_MIN_MACOSX, 2)
				ml.data[0] = 10<<16 | 9<<8 | 0<<0 // OS X version 10.9.0
				ml.data[1] = 10<<16 | 9<<8 | 0<<0 // SDK 10.9.0
			}
		case sys.ARM, sys.ARM64:
			machoPlatform = PLATFORM_IOS
		}
	}

	// empirically, string table must begin with " \x00".
	s := ctxt.loader.LookupOrCreateSym(".machosymstr", 0)
	sb := ctxt.loader.MakeSymbolUpdater(s)

	sb.SetType(sym.SMACHOSYMSTR)
	sb.SetReachable(true)
	sb.AddUint8(' ')
	sb.AddUint8('\x00')

	s = ctxt.loader.LookupOrCreateSym(".machosymtab", 0)
	sb = ctxt.loader.MakeSymbolUpdater(s)
	sb.SetType(sym.SMACHOSYMTAB)
	sb.SetReachable(true)

	if ctxt.IsInternal() {
		s = ctxt.loader.LookupOrCreateSym(".plt", 0) // will be __symbol_stub
		sb = ctxt.loader.MakeSymbolUpdater(s)
		sb.SetType(sym.SMACHOPLT)
		sb.SetReachable(true)

		s = ctxt.loader.LookupOrCreateSym(".got", 0) // will be __nl_symbol_ptr
		sb = ctxt.loader.MakeSymbolUpdater(s)
		sb.SetType(sym.SMACHOGOT)
		sb.SetReachable(true)
		sb.SetAlign(4)

		s = ctxt.loader.LookupOrCreateSym(".linkedit.plt", 0) // indirect table for .plt
		sb = ctxt.loader.MakeSymbolUpdater(s)
		sb.SetType(sym.SMACHOINDIRECTPLT)
		sb.SetReachable(true)

		s = ctxt.loader.LookupOrCreateSym(".linkedit.got", 0) // indirect table for .got
		sb = ctxt.loader.MakeSymbolUpdater(s)
		sb.SetType(sym.SMACHOINDIRECTGOT)
		sb.SetReachable(true)
	}

	// Add a dummy symbol that will become the __asm marker section.
	if ctxt.IsExternal() {
		s = ctxt.loader.LookupOrCreateSym(".llvmasm", 0)
		sb = ctxt.loader.MakeSymbolUpdater(s)
		sb.SetType(sym.SMACHO)
		sb.SetReachable(true)
		sb.AddUint8(0)
	}
}

func machoadddynlib(lib string, linkmode LinkMode) {
	if seenlib[lib] || linkmode == LinkExternal {
		return
	}
	seenlib[lib] = true

	// Will need to store the library name rounded up
	// and 24 bytes of header metadata. If not enough
	// space, grab another page of initial space at the
	// beginning of the output file.
	loadBudget -= (len(lib)+7)/8*8 + 24

	if loadBudget < 0 {
		HEADR += 4096
		*FlagTextAddr += 4096
		loadBudget += 4096
	}

	dylib = append(dylib, lib)
}

func machoshbits(ctxt *Link, mseg *MachoSeg, sect *sym.Section, segname string) {
	buf := "__" + strings.Replace(sect.Name[1:], ".", "_", -1)

	var msect *MachoSect
	if sect.Rwx&1 == 0 && segname != "__DWARF" && (ctxt.Arch.Family == sys.ARM64 ||
		ctxt.Arch.Family == sys.ARM ||
		(ctxt.Arch.Family == sys.AMD64 && ctxt.BuildMode != BuildModeExe)) {
		// Darwin external linker on arm and arm64, and on amd64 in c-shared/c-archive buildmode
		// complains about absolute relocs in __TEXT, so if the section is not
		// executable, put it in __DATA segment.
		msect = newMachoSect(mseg, buf, "__DATA")
	} else {
		msect = newMachoSect(mseg, buf, segname)
	}

	if sect.Rellen > 0 {
		msect.reloc = uint32(sect.Reloff)
		msect.nreloc = uint32(sect.Rellen / 8)
	}

	for 1<<msect.align < sect.Align {
		msect.align++
	}
	msect.addr = sect.Vaddr
	msect.size = sect.Length

	if sect.Vaddr < sect.Seg.Vaddr+sect.Seg.Filelen {
		// data in file
		if sect.Length > sect.Seg.Vaddr+sect.Seg.Filelen-sect.Vaddr {
			Errorf(nil, "macho cannot represent section %s crossing data and bss", sect.Name)
		}
		msect.off = uint32(sect.Seg.Fileoff + sect.Vaddr - sect.Seg.Vaddr)
	} else {
		msect.off = 0
		msect.flag |= S_ZEROFILL
	}

	if sect.Rwx&1 != 0 {
		msect.flag |= S_ATTR_SOME_INSTRUCTIONS
	}

	if sect.Name == ".text" {
		msect.flag |= S_ATTR_PURE_INSTRUCTIONS
	}

	if sect.Name == ".plt" {
		msect.name = "__symbol_stub1"
		msect.flag = S_ATTR_PURE_INSTRUCTIONS | S_ATTR_SOME_INSTRUCTIONS | S_SYMBOL_STUBS
		msect.res1 = 0 //nkind[SymKindLocal];
		msect.res2 = 6
	}

	if sect.Name == ".got" {
		msect.name = "__nl_symbol_ptr"
		msect.flag = S_NON_LAZY_SYMBOL_POINTERS
		msect.res1 = uint32(ctxt.Syms.Lookup(".linkedit.plt", 0).Size / 4) /* offset into indirect symbol table */
	}

	if sect.Name == ".init_array" {
		msect.name = "__mod_init_func"
		msect.flag = S_MOD_INIT_FUNC_POINTERS
	}

	// Some platforms such as watchOS and tvOS require binaries with
	// bitcode enabled. The Go toolchain can't output bitcode, so use
	// a marker section in the __LLVM segment, "__asm", to tell the Apple
	// toolchain that the Go text came from assembler and thus has no
	// bitcode. This is not true, but Kotlin/Native, Rust and Flutter
	// are also using this trick.
	if sect.Name == ".llvmasm" {
		msect.name = "__asm"
		msect.segname = "__LLVM"
	}

	if segname == "__DWARF" {
		msect.flag |= S_ATTR_DEBUG
	}
}

func Asmbmacho(ctxt *Link) {
	/* apple MACH */
	va := *FlagTextAddr - int64(HEADR)

	mh := getMachoHdr()
	switch ctxt.Arch.Family {
	default:
		Exitf("unknown macho architecture: %v", ctxt.Arch.Family)

	case sys.ARM:
		mh.cpu = MACHO_CPU_ARM
		mh.subcpu = MACHO_SUBCPU_ARMV7

	case sys.AMD64:
		mh.cpu = MACHO_CPU_AMD64
		mh.subcpu = MACHO_SUBCPU_X86

	case sys.ARM64:
		mh.cpu = MACHO_CPU_ARM64
		mh.subcpu = MACHO_SUBCPU_ARM64_ALL

	case sys.I386:
		mh.cpu = MACHO_CPU_386
		mh.subcpu = MACHO_SUBCPU_X86
	}

	var ms *MachoSeg
	if ctxt.LinkMode == LinkExternal {
		/* segment for entire file */
		ms = newMachoSeg("", 40)

		ms.fileoffset = Segtext.Fileoff
		ms.filesize = Segdwarf.Fileoff + Segdwarf.Filelen - Segtext.Fileoff
		ms.vsize = Segdwarf.Vaddr + Segdwarf.Length - Segtext.Vaddr
	}

	/* segment for zero page */
	if ctxt.LinkMode != LinkExternal {
		ms = newMachoSeg("__PAGEZERO", 0)
		ms.vsize = uint64(va)
	}

	/* text */
	v := Rnd(int64(uint64(HEADR)+Segtext.Length), int64(*FlagRound))

	if ctxt.LinkMode != LinkExternal {
		ms = newMachoSeg("__TEXT", 20)
		ms.vaddr = uint64(va)
		ms.vsize = uint64(v)
		ms.fileoffset = 0
		ms.filesize = uint64(v)
		ms.prot1 = 7
		ms.prot2 = 5
	}

	for _, sect := range Segtext.Sections {
		machoshbits(ctxt, ms, sect, "__TEXT")
	}

	/* data */
	if ctxt.LinkMode != LinkExternal {
		w := int64(Segdata.Length)
		ms = newMachoSeg("__DATA", 20)
		ms.vaddr = uint64(va) + uint64(v)
		ms.vsize = uint64(w)
		ms.fileoffset = uint64(v)
		ms.filesize = Segdata.Filelen
		ms.prot1 = 3
		ms.prot2 = 3
	}

	for _, sect := range Segdata.Sections {
		machoshbits(ctxt, ms, sect, "__DATA")
	}

	/* dwarf */
	if !*FlagW {
		if ctxt.LinkMode != LinkExternal {
			ms = newMachoSeg("__DWARF", 20)
			ms.vaddr = Segdwarf.Vaddr
			ms.vsize = 0
			ms.fileoffset = Segdwarf.Fileoff
			ms.filesize = Segdwarf.Filelen
		}
		for _, sect := range Segdwarf.Sections {
			machoshbits(ctxt, ms, sect, "__DWARF")
		}
	}

	if ctxt.LinkMode != LinkExternal {
		switch ctxt.Arch.Family {
		default:
			Exitf("unknown macho architecture: %v", ctxt.Arch.Family)

		case sys.ARM:
			ml := newMachoLoad(ctxt.Arch, LC_UNIXTHREAD, 17+2)
			ml.data[0] = 1                           /* thread type */
			ml.data[1] = 17                          /* word count */
			ml.data[2+15] = uint32(Entryvalue(ctxt)) /* start pc */

		case sys.AMD64:
			ml := newMachoLoad(ctxt.Arch, LC_UNIXTHREAD, 42+2)
			ml.data[0] = 4                           /* thread type */
			ml.data[1] = 42                          /* word count */
			ml.data[2+32] = uint32(Entryvalue(ctxt)) /* start pc */
			ml.data[2+32+1] = uint32(Entryvalue(ctxt) >> 32)

		case sys.ARM64:
			ml := newMachoLoad(ctxt.Arch, LC_UNIXTHREAD, 68+2)
			ml.data[0] = 6                           /* thread type */
			ml.data[1] = 68                          /* word count */
			ml.data[2+64] = uint32(Entryvalue(ctxt)) /* start pc */
			ml.data[2+64+1] = uint32(Entryvalue(ctxt) >> 32)

		case sys.I386:
			ml := newMachoLoad(ctxt.Arch, LC_UNIXTHREAD, 16+2)
			ml.data[0] = 1                           /* thread type */
			ml.data[1] = 16                          /* word count */
			ml.data[2+10] = uint32(Entryvalue(ctxt)) /* start pc */
		}
	}

	if !*FlagD {
		// must match domacholink below
		s1 := ctxt.Syms.Lookup(".machosymtab", 0)
		s2 := ctxt.Syms.Lookup(".linkedit.plt", 0)
		s3 := ctxt.Syms.Lookup(".linkedit.got", 0)
		s4 := ctxt.Syms.Lookup(".machosymstr", 0)

		if ctxt.LinkMode != LinkExternal {
			ms := newMachoSeg("__LINKEDIT", 0)
			ms.vaddr = uint64(va) + uint64(v) + uint64(Rnd(int64(Segdata.Length), int64(*FlagRound)))
			ms.vsize = uint64(s1.Size) + uint64(s2.Size) + uint64(s3.Size) + uint64(s4.Size)
			ms.fileoffset = uint64(linkoff)
			ms.filesize = ms.vsize
			ms.prot1 = 7
			ms.prot2 = 3
		}

		ml := newMachoLoad(ctxt.Arch, LC_SYMTAB, 4)
		ml.data[0] = uint32(linkoff)                               /* symoff */
		ml.data[1] = uint32(nsortsym)                              /* nsyms */
		ml.data[2] = uint32(linkoff + s1.Size + s2.Size + s3.Size) /* stroff */
		ml.data[3] = uint32(s4.Size)                               /* strsize */

		machodysymtab(ctxt)

		if ctxt.LinkMode != LinkExternal {
			ml := newMachoLoad(ctxt.Arch, LC_LOAD_DYLINKER, 6)
			ml.data[0] = 12 /* offset to string */
			stringtouint32(ml.data[1:], "/usr/lib/dyld")

			for _, lib := range dylib {
				ml = newMachoLoad(ctxt.Arch, LC_LOAD_DYLIB, 4+(uint32(len(lib))+1+7)/8*2)
				ml.data[0] = 24 /* offset of string from beginning of load */
				ml.data[1] = 0  /* time stamp */
				ml.data[2] = 0  /* version */
				ml.data[3] = 0  /* compatibility version */
				stringtouint32(ml.data[4:], lib)
			}
		}
	}

	a := machowrite(ctxt.Arch, ctxt.Out, ctxt.LinkMode)
	if int32(a) > HEADR {
		Exitf("HEADR too small: %d > %d", a, HEADR)
	}
}

func symkind(ldr *loader.Loader, s loader.Sym) int {
	if ldr.SymType(s) == sym.SDYNIMPORT {
		return SymKindUndef
	}
	if ldr.AttrCgoExport(s) {
		return SymKindExtdef
	}
	return SymKindLocal
}

func collectmachosyms(ctxt *Link) {
	ldr := ctxt.loader

	addsym := func(s loader.Sym) {
		sortsym = append(sortsym, s)
		nkind[symkind(ldr, s)]++
	}

	// Add special runtime.text and runtime.etext symbols.
	// We've already included this symbol in Textp on darwin if ctxt.DynlinkingGo().
	// See data.go:/textaddress
	if !ctxt.DynlinkingGo() {
		s := ldr.Lookup("runtime.text", 0)
		if ldr.SymType(s) == sym.STEXT {
			addsym(s)
		}
		s = ldr.Lookup("runtime.etext", 0)
		if ldr.SymType(s) == sym.STEXT {
			addsym(s)
		}
	}

	// Add text symbols.
	for _, s := range ctxt.Textp2 {
		addsym(s)
	}

	shouldBeInSymbolTable := func(s loader.Sym) bool {
		if ldr.AttrNotInSymbolTable(s) {
			return false
		}
		name := ldr.RawSymName(s) // TODO: try not to read the name
		if name == "" || name[0] == '.' {
			return false
		}
		return true
	}

	// Add data symbols and external references.
	for s := loader.Sym(1); s < loader.Sym(ldr.NSym()); s++ {
		if !ldr.AttrReachable(s) {
			continue
		}
		t := ldr.SymType(s)
		if t >= sym.SELFRXSECT && t < sym.SXREF || t == sym.SCONST { // data sections handled in dodata
			if t == sym.STLSBSS {
				// TLSBSS is not used on darwin. See data.go:allocateDataSections
				continue
			}
			if !shouldBeInSymbolTable(s) {
				continue
			}
			addsym(s)
		}

		switch t {
		case sym.SDYNIMPORT, sym.SHOSTOBJ, sym.SUNDEFEXT:
			addsym(s)
		}

		// Some 64-bit functions have a "$INODE64" or "$INODE64$UNIX2003" suffix.
		if t == sym.SDYNIMPORT && ldr.SymDynimplib(s) == "/usr/lib/libSystem.B.dylib" {
			// But only on macOS.
			if machoPlatform == PLATFORM_MACOS {
				switch n := ldr.SymExtname(s); n {
				case "fdopendir":
					switch objabi.GOARCH {
					case "amd64":
						ldr.SetSymExtname(s, n+"$INODE64")
					case "386":
						ldr.SetSymExtname(s, n+"$INODE64$UNIX2003")
					}
				case "readdir_r", "getfsstat":
					switch objabi.GOARCH {
					case "amd64", "386":
						ldr.SetSymExtname(s, n+"$INODE64")
					}
				}
			}
		}
	}

	nsortsym = len(sortsym)
}

func machosymorder(ctxt *Link) {
	ldr := ctxt.loader

	// On Mac OS X Mountain Lion, we must sort exported symbols
	// So we sort them here and pre-allocate dynid for them
	// See https://golang.org/issue/4029
	for _, s := range ctxt.dynexp2 {
		if !ldr.AttrReachable(s) {
			panic("dynexp symbol is not reachable")
		}
	}
	collectmachosyms(ctxt)
	sort.Slice(sortsym[:nsortsym], func(i, j int) bool {
		s1 := sortsym[i]
		s2 := sortsym[j]
		k1 := symkind(ldr, s1)
		k2 := symkind(ldr, s2)
		if k1 != k2 {
			return k1 < k2
		}
		return ldr.SymExtname(s1) < ldr.SymExtname(s2) // Note: unnamed symbols are not added in collectmachosyms
	})
	for i, s := range sortsym {
		ldr.SetSymDynid(s, int32(i))
	}
}

// machoShouldExport reports whether a symbol needs to be exported.
//
// When dynamically linking, all non-local variables and plugin-exported
// symbols need to be exported.
func machoShouldExport(ctxt *Link, s *sym.Symbol) bool {
	if !ctxt.DynlinkingGo() || s.Attr.Local() {
		return false
	}
	if ctxt.BuildMode == BuildModePlugin && strings.HasPrefix(s.Extname(), objabi.PathToPrefix(*flagPluginPath)) {
		return true
	}
	if strings.HasPrefix(s.Name, "go.itab.") {
		return true
	}
	if strings.HasPrefix(s.Name, "type.") && !strings.HasPrefix(s.Name, "type..") {
		// reduce runtime typemap pressure, but do not
		// export alg functions (type..*), as these
		// appear in pclntable.
		return true
	}
	if strings.HasPrefix(s.Name, "go.link.pkghash") {
		return true
	}
	return s.Type >= sym.SFirstWritable // only writable sections
}

func machosymtab(ctxt *Link) {
	symtab := ctxt.Syms.Lookup(".machosymtab", 0)
	symstr := ctxt.Syms.Lookup(".machosymstr", 0)

	for i := 0; i < nsortsym; i++ {
		s := ctxt.loader.Syms[sortsym[i]]
		symtab.AddUint32(ctxt.Arch, uint32(symstr.Size))

		export := machoShouldExport(ctxt, s)
		isGoSymbol := strings.Contains(s.Extname(), ".")

		// In normal buildmodes, only add _ to C symbols, as
		// Go symbols have dot in the name.
		//
		// Do not export C symbols in plugins, as runtime C
		// symbols like crosscall2 are in pclntab and end up
		// pointing at the host binary, breaking unwinding.
		// See Issue #18190.
		cexport := !isGoSymbol && (ctxt.BuildMode != BuildModePlugin || onlycsymbol(s.Name))
		if cexport || export || isGoSymbol {
			symstr.AddUint8('_')
		}

		// replace "·" as ".", because DTrace cannot handle it.
		Addstring(symstr, strings.Replace(s.Extname(), "·", ".", -1))

		if s.Type == sym.SDYNIMPORT || s.Type == sym.SHOSTOBJ || s.Type == sym.SUNDEFEXT {
			symtab.AddUint8(0x01)                             // type N_EXT, external symbol
			symtab.AddUint8(0)                                // no section
			symtab.AddUint16(ctxt.Arch, 0)                    // desc
			symtab.AddUintXX(ctxt.Arch, 0, ctxt.Arch.PtrSize) // no value
		} else {
			if s.Attr.CgoExport() || export {
				symtab.AddUint8(0x0f)
			} else {
				symtab.AddUint8(0x0e)
			}
			o := s
			for o.Outer != nil {
				o = o.Outer
			}
			if o.Sect == nil {
				Errorf(s, "missing section for symbol")
				symtab.AddUint8(0)
			} else {
				symtab.AddUint8(uint8(o.Sect.Extnum))
			}
			symtab.AddUint16(ctxt.Arch, 0) // desc
			symtab.AddUintXX(ctxt.Arch, uint64(Symaddr(s)), ctxt.Arch.PtrSize)
		}
	}
}

func machodysymtab(ctxt *Link) {
	ml := newMachoLoad(ctxt.Arch, LC_DYSYMTAB, 18)

	n := 0
	ml.data[0] = uint32(n)                   /* ilocalsym */
	ml.data[1] = uint32(nkind[SymKindLocal]) /* nlocalsym */
	n += nkind[SymKindLocal]

	ml.data[2] = uint32(n)                    /* iextdefsym */
	ml.data[3] = uint32(nkind[SymKindExtdef]) /* nextdefsym */
	n += nkind[SymKindExtdef]

	ml.data[4] = uint32(n)                   /* iundefsym */
	ml.data[5] = uint32(nkind[SymKindUndef]) /* nundefsym */

	ml.data[6] = 0  /* tocoffset */
	ml.data[7] = 0  /* ntoc */
	ml.data[8] = 0  /* modtaboff */
	ml.data[9] = 0  /* nmodtab */
	ml.data[10] = 0 /* extrefsymoff */
	ml.data[11] = 0 /* nextrefsyms */

	// must match domacholink below
	s1 := ctxt.Syms.Lookup(".machosymtab", 0)

	s2 := ctxt.Syms.Lookup(".linkedit.plt", 0)
	s3 := ctxt.Syms.Lookup(".linkedit.got", 0)
	ml.data[12] = uint32(linkoff + s1.Size)       /* indirectsymoff */
	ml.data[13] = uint32((s2.Size + s3.Size) / 4) /* nindirectsyms */

	ml.data[14] = 0 /* extreloff */
	ml.data[15] = 0 /* nextrel */
	ml.data[16] = 0 /* locreloff */
	ml.data[17] = 0 /* nlocrel */
}

func Domacholink(ctxt *Link) int64 {
	machosymtab(ctxt)

	// write data that will be linkedit section
	s1 := ctxt.Syms.Lookup(".machosymtab", 0)

	s2 := ctxt.Syms.Lookup(".linkedit.plt", 0)
	s3 := ctxt.Syms.Lookup(".linkedit.got", 0)
	s4 := ctxt.Syms.Lookup(".machosymstr", 0)

	// Force the linkedit section to end on a 16-byte
	// boundary. This allows pure (non-cgo) Go binaries
	// to be code signed correctly.
	//
	// Apple's codesign_allocate (a helper utility for
	// the codesign utility) can do this fine itself if
	// it is run on a dynamic Mach-O binary. However,
	// when it is run on a pure (non-cgo) Go binary, where
	// the linkedit section is mostly empty, it fails to
	// account for the extra padding that it itself adds
	// when adding the LC_CODE_SIGNATURE load command
	// (which must be aligned on a 16-byte boundary).
	//
	// By forcing the linkedit section to end on a 16-byte
	// boundary, codesign_allocate will not need to apply
	// any alignment padding itself, working around the
	// issue.
	for s4.Size%16 != 0 {
		s4.AddUint8(0)
	}

	size := int(s1.Size + s2.Size + s3.Size + s4.Size)

	if size > 0 {
		linkoff = Rnd(int64(uint64(HEADR)+Segtext.Length), int64(*FlagRound)) + Rnd(int64(Segdata.Filelen), int64(*FlagRound)) + Rnd(int64(Segdwarf.Filelen), int64(*FlagRound))
		ctxt.Out.SeekSet(linkoff)

		ctxt.Out.Write(s1.P[:s1.Size])
		ctxt.Out.Write(s2.P[:s2.Size])
		ctxt.Out.Write(s3.P[:s3.Size])
		ctxt.Out.Write(s4.P[:s4.Size])
	}

	return Rnd(int64(size), int64(*FlagRound))
}

func machorelocsect(ctxt *Link, sect *sym.Section, syms []*sym.Symbol) {
	// If main section has no bits, nothing to relocate.
	if sect.Vaddr >= sect.Seg.Vaddr+sect.Seg.Filelen {
		return
	}

	sect.Reloff = uint64(ctxt.Out.Offset())
	for i, s := range syms {
		if !s.Attr.Reachable() {
			continue
		}
		if uint64(s.Value) >= sect.Vaddr {
			syms = syms[i:]
			break
		}
	}

	eaddr := int32(sect.Vaddr + sect.Length)
	for _, s := range syms {
		if !s.Attr.Reachable() {
			continue
		}
		if s.Value >= int64(eaddr) {
			break
		}
		for ri := range s.R {
			r := &s.R[ri]
			if r.Done {
				continue
			}
			if r.Xsym == nil {
				Errorf(s, "missing xsym in relocation")
				continue
			}
			if !r.Xsym.Attr.Reachable() {
				Errorf(s, "unreachable reloc %d (%s) target %v", r.Type, sym.RelocName(ctxt.Arch, r.Type), r.Xsym.Name)
			}
			if !thearch.Machoreloc1(ctxt.Arch, ctxt.Out, s, r, int64(uint64(s.Value+int64(r.Off))-sect.Vaddr)) {
				Errorf(s, "unsupported obj reloc %d (%s)/%d to %s", r.Type, sym.RelocName(ctxt.Arch, r.Type), r.Siz, r.Sym.Name)
			}
		}
	}

	sect.Rellen = uint64(ctxt.Out.Offset()) - sect.Reloff
}

func Machoemitreloc(ctxt *Link) {
	for ctxt.Out.Offset()&7 != 0 {
		ctxt.Out.Write8(0)
	}

	machorelocsect(ctxt, Segtext.Sections[0], ctxt.Textp)
	for _, sect := range Segtext.Sections[1:] {
		machorelocsect(ctxt, sect, ctxt.datap)
	}
	for _, sect := range Segdata.Sections {
		machorelocsect(ctxt, sect, ctxt.datap)
	}
	for i := 0; i < len(Segdwarf.Sections); i++ {
		sect := Segdwarf.Sections[i]
		si := dwarfp[i]
		if si.secSym() != sect.Sym ||
			si.secSym().Sect != sect {
			panic("inconsistency between dwarfp and Segdwarf")
		}
		machorelocsect(ctxt, sect, si.syms)
	}
}

// hostobjMachoPlatform returns the first platform load command found
// in the host object, if any.
func hostobjMachoPlatform(h *Hostobj) (*MachoPlatformLoad, error) {
	f, err := os.Open(h.file)
	if err != nil {
		return nil, fmt.Errorf("%s: failed to open host object: %v\n", h.file, err)
	}
	defer f.Close()
	sr := io.NewSectionReader(f, h.off, h.length)
	m, err := macho.NewFile(sr)
	if err != nil {
		// Not a valid Mach-O file.
		return nil, nil
	}
	return peekMachoPlatform(m)
}

// peekMachoPlatform returns the first LC_VERSION_MIN_* or LC_BUILD_VERSION
// load command found in the Mach-O file, if any.
func peekMachoPlatform(m *macho.File) (*MachoPlatformLoad, error) {
	for _, cmd := range m.Loads {
		raw := cmd.Raw()
		ml := MachoLoad{
			type_: m.ByteOrder.Uint32(raw),
		}
		// Skip the type and command length.
		data := raw[8:]
		var p MachoPlatform
		switch ml.type_ {
		case LC_VERSION_MIN_IPHONEOS:
			p = PLATFORM_IOS
		case LC_VERSION_MIN_MACOSX:
			p = PLATFORM_MACOS
		case LC_VERSION_MIN_WATCHOS:
			p = PLATFORM_WATCHOS
		case LC_VERSION_MIN_TVOS:
			p = PLATFORM_TVOS
		case LC_BUILD_VERSION:
			p = MachoPlatform(m.ByteOrder.Uint32(data))
		default:
			continue
		}
		ml.data = make([]uint32, len(data)/4)
		r := bytes.NewReader(data)
		if err := binary.Read(r, m.ByteOrder, &ml.data); err != nil {
			return nil, err
		}
		return &MachoPlatformLoad{
			platform: p,
			cmd:      ml,
		}, nil
	}
	return nil, nil
}
