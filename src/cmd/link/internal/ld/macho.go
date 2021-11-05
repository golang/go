// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"bytes"
	"cmd/internal/codesign"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"debug/macho"
	"encoding/binary"
	"fmt"
	"internal/buildcfg"
	"io"
	"os"
	"sort"
	"strings"
	"unsafe"
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
	MACHO_CPU_AMD64                      = 1<<24 | 7
	MACHO_CPU_386                        = 7
	MACHO_SUBCPU_X86                     = 3
	MACHO_CPU_ARM                        = 12
	MACHO_SUBCPU_ARM                     = 0
	MACHO_SUBCPU_ARMV7                   = 9
	MACHO_CPU_ARM64                      = 1<<24 | 12
	MACHO_SUBCPU_ARM64_ALL               = 0
	MACHO_SUBCPU_ARM64_V8                = 1
	MACHO_SUBCPU_ARM64E                  = 2
	MACHO32SYMSIZE                       = 12
	MACHO64SYMSIZE                       = 16
	MACHO_X86_64_RELOC_UNSIGNED          = 0
	MACHO_X86_64_RELOC_SIGNED            = 1
	MACHO_X86_64_RELOC_BRANCH            = 2
	MACHO_X86_64_RELOC_GOT_LOAD          = 3
	MACHO_X86_64_RELOC_GOT               = 4
	MACHO_X86_64_RELOC_SUBTRACTOR        = 5
	MACHO_X86_64_RELOC_SIGNED_1          = 6
	MACHO_X86_64_RELOC_SIGNED_2          = 7
	MACHO_X86_64_RELOC_SIGNED_4          = 8
	MACHO_ARM_RELOC_VANILLA              = 0
	MACHO_ARM_RELOC_PAIR                 = 1
	MACHO_ARM_RELOC_SECTDIFF             = 2
	MACHO_ARM_RELOC_BR24                 = 5
	MACHO_ARM64_RELOC_UNSIGNED           = 0
	MACHO_ARM64_RELOC_BRANCH26           = 2
	MACHO_ARM64_RELOC_PAGE21             = 3
	MACHO_ARM64_RELOC_PAGEOFF12          = 4
	MACHO_ARM64_RELOC_GOT_LOAD_PAGE21    = 5
	MACHO_ARM64_RELOC_GOT_LOAD_PAGEOFF12 = 6
	MACHO_ARM64_RELOC_ADDEND             = 10
	MACHO_GENERIC_RELOC_VANILLA          = 0
	MACHO_FAKE_GOTPCREL                  = 100
)

const (
	MH_MAGIC    = 0xfeedface
	MH_MAGIC_64 = 0xfeedfacf

	MH_OBJECT  = 0x1
	MH_EXECUTE = 0x2

	MH_NOUNDEFS = 0x1
	MH_DYLDLINK = 0x4
	MH_PIE      = 0x200000
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
	LC_DYLD_EXPORTS_TRIE        = 0x80000033
	LC_DYLD_CHAINED_FIXUPS      = 0x80000034
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

// rebase table opcode
const (
	REBASE_TYPE_POINTER         = 1
	REBASE_TYPE_TEXT_ABSOLUTE32 = 2
	REBASE_TYPE_TEXT_PCREL32    = 3

	REBASE_OPCODE_MASK                               = 0xF0
	REBASE_IMMEDIATE_MASK                            = 0x0F
	REBASE_OPCODE_DONE                               = 0x00
	REBASE_OPCODE_SET_TYPE_IMM                       = 0x10
	REBASE_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB        = 0x20
	REBASE_OPCODE_ADD_ADDR_ULEB                      = 0x30
	REBASE_OPCODE_ADD_ADDR_IMM_SCALED                = 0x40
	REBASE_OPCODE_DO_REBASE_IMM_TIMES                = 0x50
	REBASE_OPCODE_DO_REBASE_ULEB_TIMES               = 0x60
	REBASE_OPCODE_DO_REBASE_ADD_ADDR_ULEB            = 0x70
	REBASE_OPCODE_DO_REBASE_ULEB_TIMES_SKIPPING_ULEB = 0x80
)

// bind table opcode
const (
	BIND_TYPE_POINTER         = 1
	BIND_TYPE_TEXT_ABSOLUTE32 = 2
	BIND_TYPE_TEXT_PCREL32    = 3

	BIND_SPECIAL_DYLIB_SELF            = 0
	BIND_SPECIAL_DYLIB_MAIN_EXECUTABLE = -1
	BIND_SPECIAL_DYLIB_FLAT_LOOKUP     = -2
	BIND_SPECIAL_DYLIB_WEAK_LOOKUP     = -3

	BIND_OPCODE_MASK                                         = 0xF0
	BIND_IMMEDIATE_MASK                                      = 0x0F
	BIND_OPCODE_DONE                                         = 0x00
	BIND_OPCODE_SET_DYLIB_ORDINAL_IMM                        = 0x10
	BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB                       = 0x20
	BIND_OPCODE_SET_DYLIB_SPECIAL_IMM                        = 0x30
	BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM                = 0x40
	BIND_OPCODE_SET_TYPE_IMM                                 = 0x50
	BIND_OPCODE_SET_ADDEND_SLEB                              = 0x60
	BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB                  = 0x70
	BIND_OPCODE_ADD_ADDR_ULEB                                = 0x80
	BIND_OPCODE_DO_BIND                                      = 0x90
	BIND_OPCODE_DO_BIND_ADD_ADDR_ULEB                        = 0xA0
	BIND_OPCODE_DO_BIND_ADD_ADDR_IMM_SCALED                  = 0xB0
	BIND_OPCODE_DO_BIND_ULEB_TIMES_SKIPPING_ULEB             = 0xC0
	BIND_OPCODE_THREADED                                     = 0xD0
	BIND_SUBOPCODE_THREADED_SET_BIND_ORDINAL_TABLE_SIZE_ULEB = 0x00
	BIND_SUBOPCODE_THREADED_APPLY                            = 0x01
)

const machoHeaderSize64 = 8 * 4 // size of 64-bit Mach-O header

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

func machowrite(ctxt *Link, arch *sys.Arch, out *OutBuf, linkmode LinkMode) int {
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
	flags := uint32(0)
	if nkind[SymKindUndef] == 0 {
		flags |= MH_NOUNDEFS
	}
	if ctxt.IsPIE() && linkmode == LinkInternal {
		flags |= MH_PIE | MH_DYLDLINK
	}
	out.Write32(flags) /* flags */
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
		machoPlatform = PLATFORM_MACOS
		if buildcfg.GOOS == "ios" {
			machoPlatform = PLATFORM_IOS
		}
		if ctxt.LinkMode == LinkInternal && machoPlatform == PLATFORM_MACOS {
			var version uint32
			switch ctxt.Arch.Family {
			case sys.AMD64:
				// The version must be at least 10.9; see golang.org/issues/30488.
				version = 10<<16 | 9<<8 | 0<<0 // 10.9.0
			case sys.ARM64:
				version = 11<<16 | 0<<8 | 0<<0 // 11.0.0
			}
			ml := newMachoLoad(ctxt.Arch, LC_BUILD_VERSION, 4)
			ml.data[0] = uint32(machoPlatform)
			ml.data[1] = version // OS version
			ml.data[2] = version // SDK version
			ml.data[3] = 0       // ntools
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

	// Un-export runtime symbols from plugins. Since the runtime
	// is included in both the main binary and each plugin, these
	// symbols appear in both images. If we leave them exported in
	// the plugin, then the dynamic linker will resolve
	// relocations to these functions in the plugin's functab to
	// point to the main image, causing the runtime to think the
	// plugin's functab is corrupted. By unexporting them, these
	// become static references, which are resolved to the
	// plugin's text.
	//
	// It would be better to omit the runtime from plugins. (Using
	// relative PCs in the functab instead of relocations would
	// also address this.)
	//
	// See issue #18190.
	if ctxt.BuildMode == BuildModePlugin {
		for _, name := range []string{"_cgo_topofstack", "__cgo_topofstack", "_cgo_panic", "crosscall2"} {
			// Most of these are data symbols or C
			// symbols, so they have symbol version 0.
			ver := 0
			// _cgo_panic is a Go function, so it uses ABIInternal.
			if name == "_cgo_panic" {
				ver = abiInternalVer
			}
			s := ctxt.loader.Lookup(name, ver)
			if s != 0 {
				ctxt.loader.SetAttrCgoExportDynamic(s, false)
			}
		}
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

	msect := newMachoSect(mseg, buf, segname)

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
		msect.res1 = uint32(ctxt.loader.SymSize(ctxt.ArchSyms.LinkEditPLT) / 4) /* offset into indirect symbol table */
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

func asmbMacho(ctxt *Link) {
	machlink := doMachoLink(ctxt)
	if !*FlagS && ctxt.IsExternal() {
		symo := int64(Segdwarf.Fileoff + uint64(Rnd(int64(Segdwarf.Filelen), int64(*FlagRound))) + uint64(machlink))
		ctxt.Out.SeekSet(symo)
		machoEmitReloc(ctxt)
	}
	ctxt.Out.SeekSet(0)

	ldr := ctxt.loader

	/* apple MACH */
	va := *FlagTextAddr - int64(HEADR)

	mh := getMachoHdr()
	switch ctxt.Arch.Family {
	default:
		Exitf("unknown macho architecture: %v", ctxt.Arch.Family)

	case sys.AMD64:
		mh.cpu = MACHO_CPU_AMD64
		mh.subcpu = MACHO_SUBCPU_X86

	case sys.ARM64:
		mh.cpu = MACHO_CPU_ARM64
		mh.subcpu = MACHO_SUBCPU_ARM64_ALL
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

	/* rodata */
	if ctxt.LinkMode != LinkExternal && Segrelrodata.Length > 0 {
		ms = newMachoSeg("__DATA_CONST", 20)
		ms.vaddr = Segrelrodata.Vaddr
		ms.vsize = Segrelrodata.Length
		ms.fileoffset = Segrelrodata.Fileoff
		ms.filesize = Segrelrodata.Filelen
		ms.prot1 = 3
		ms.prot2 = 3
		ms.flag = 0x10 // SG_READ_ONLY
	}

	for _, sect := range Segrelrodata.Sections {
		machoshbits(ctxt, ms, sect, "__DATA_CONST")
	}

	/* data */
	if ctxt.LinkMode != LinkExternal {
		ms = newMachoSeg("__DATA", 20)
		ms.vaddr = Segdata.Vaddr
		ms.vsize = Segdata.Length
		ms.fileoffset = Segdata.Fileoff
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

		case sys.AMD64:
			ml := newMachoLoad(ctxt.Arch, LC_UNIXTHREAD, 42+2)
			ml.data[0] = 4                           /* thread type */
			ml.data[1] = 42                          /* word count */
			ml.data[2+32] = uint32(Entryvalue(ctxt)) /* start pc */
			ml.data[2+32+1] = uint32(Entryvalue(ctxt) >> 32)

		case sys.ARM64:
			ml := newMachoLoad(ctxt.Arch, LC_MAIN, 4)
			ml.data[0] = uint32(uint64(Entryvalue(ctxt)) - (Segtext.Vaddr - uint64(HEADR)))
			ml.data[1] = uint32((uint64(Entryvalue(ctxt)) - (Segtext.Vaddr - uint64(HEADR))) >> 32)
		}
	}

	var codesigOff int64
	if !*FlagD {
		// must match doMachoLink below
		s1 := ldr.SymSize(ldr.Lookup(".machorebase", 0))
		s2 := ldr.SymSize(ldr.Lookup(".machobind", 0))
		s3 := ldr.SymSize(ldr.Lookup(".machosymtab", 0))
		s4 := ldr.SymSize(ctxt.ArchSyms.LinkEditPLT)
		s5 := ldr.SymSize(ctxt.ArchSyms.LinkEditGOT)
		s6 := ldr.SymSize(ldr.Lookup(".machosymstr", 0))
		s7 := ldr.SymSize(ldr.Lookup(".machocodesig", 0))

		if ctxt.LinkMode != LinkExternal {
			ms := newMachoSeg("__LINKEDIT", 0)
			ms.vaddr = uint64(Rnd(int64(Segdata.Vaddr+Segdata.Length), int64(*FlagRound)))
			ms.vsize = uint64(s1 + s2 + s3 + s4 + s5 + s6 + s7)
			ms.fileoffset = uint64(linkoff)
			ms.filesize = ms.vsize
			ms.prot1 = 1
			ms.prot2 = 1

			codesigOff = linkoff + s1 + s2 + s3 + s4 + s5 + s6
		}

		if ctxt.LinkMode != LinkExternal && ctxt.IsPIE() {
			ml := newMachoLoad(ctxt.Arch, LC_DYLD_INFO_ONLY, 10)
			ml.data[0] = uint32(linkoff)      // rebase off
			ml.data[1] = uint32(s1)           // rebase size
			ml.data[2] = uint32(linkoff + s1) // bind off
			ml.data[3] = uint32(s2)           // bind size
			ml.data[4] = 0                    // weak bind off
			ml.data[5] = 0                    // weak bind size
			ml.data[6] = 0                    // lazy bind off
			ml.data[7] = 0                    // lazy bind size
			ml.data[8] = 0                    // export
			ml.data[9] = 0                    // export size
		}

		ml := newMachoLoad(ctxt.Arch, LC_SYMTAB, 4)
		ml.data[0] = uint32(linkoff + s1 + s2)                /* symoff */
		ml.data[1] = uint32(nsortsym)                         /* nsyms */
		ml.data[2] = uint32(linkoff + s1 + s2 + s3 + s4 + s5) /* stroff */
		ml.data[3] = uint32(s6)                               /* strsize */

		machodysymtab(ctxt, linkoff+s1+s2)

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

		if ctxt.IsInternal() && ctxt.NeedCodeSign() {
			ml := newMachoLoad(ctxt.Arch, LC_CODE_SIGNATURE, 2)
			ml.data[0] = uint32(codesigOff)
			ml.data[1] = uint32(s7)
		}
	}

	a := machowrite(ctxt, ctxt.Arch, ctxt.Out, ctxt.LinkMode)
	if int32(a) > HEADR {
		Exitf("HEADR too small: %d > %d", a, HEADR)
	}

	// Now we have written everything. Compute the code signature (which
	// is a hash of the file content, so it must be done at last.)
	if ctxt.IsInternal() && ctxt.NeedCodeSign() {
		cs := ldr.Lookup(".machocodesig", 0)
		data := ctxt.Out.Data()
		if int64(len(data)) != codesigOff {
			panic("wrong size")
		}
		codesign.Sign(ldr.Data(cs), bytes.NewReader(data), "a.out", codesigOff, int64(Segtext.Fileoff), int64(Segtext.Filelen), ctxt.IsExe() || ctxt.IsPIE())
		ctxt.Out.SeekSet(codesigOff)
		ctxt.Out.Write(ldr.Data(cs))
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
		for n := range Segtext.Sections[1:] {
			s := ldr.Lookup(fmt.Sprintf("runtime.text.%d", n+1), 0)
			if s != 0 {
				addsym(s)
			} else {
				break
			}
		}
		s = ldr.Lookup("runtime.etext", 0)
		if ldr.SymType(s) == sym.STEXT {
			addsym(s)
		}
	}

	// Add text symbols.
	for _, s := range ctxt.Textp {
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
		if t >= sym.SELFRXSECT && t < sym.SXREF { // data sections handled in dodata
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
					switch buildcfg.GOARCH {
					case "amd64":
						ldr.SetSymExtname(s, n+"$INODE64")
					}
				case "readdir_r", "getfsstat":
					switch buildcfg.GOARCH {
					case "amd64":
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
	for _, s := range ctxt.dynexp {
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

// AddMachoSym adds s to Mach-O symbol table, used in GenSymLate.
// Currently only used on ARM64 when external linking.
func AddMachoSym(ldr *loader.Loader, s loader.Sym) {
	ldr.SetSymDynid(s, int32(nsortsym))
	sortsym = append(sortsym, s)
	nsortsym++
	nkind[symkind(ldr, s)]++
}

// machoShouldExport reports whether a symbol needs to be exported.
//
// When dynamically linking, all non-local variables and plugin-exported
// symbols need to be exported.
func machoShouldExport(ctxt *Link, ldr *loader.Loader, s loader.Sym) bool {
	if !ctxt.DynlinkingGo() || ldr.AttrLocal(s) {
		return false
	}
	if ctxt.BuildMode == BuildModePlugin && strings.HasPrefix(ldr.SymExtname(s), objabi.PathToPrefix(*flagPluginPath)) {
		return true
	}
	name := ldr.RawSymName(s)
	if strings.HasPrefix(name, "go.itab.") {
		return true
	}
	if strings.HasPrefix(name, "type.") && !strings.HasPrefix(name, "type..") {
		// reduce runtime typemap pressure, but do not
		// export alg functions (type..*), as these
		// appear in pclntable.
		return true
	}
	if strings.HasPrefix(name, "go.link.pkghash") {
		return true
	}
	return ldr.SymType(s) >= sym.SFirstWritable // only writable sections
}

func machosymtab(ctxt *Link) {
	ldr := ctxt.loader
	symtab := ldr.CreateSymForUpdate(".machosymtab", 0)
	symstr := ldr.CreateSymForUpdate(".machosymstr", 0)

	for _, s := range sortsym[:nsortsym] {
		symtab.AddUint32(ctxt.Arch, uint32(symstr.Size()))

		export := machoShouldExport(ctxt, ldr, s)

		// Prefix symbol names with "_" to match the system toolchain.
		// (We used to only prefix C symbols, which is all required for the build.
		// But some tools don't recognize Go symbols as symbols, so we prefix them
		// as well.)
		symstr.AddUint8('_')

		// replace "·" as ".", because DTrace cannot handle it.
		name := strings.Replace(ldr.SymExtname(s), "·", ".", -1)

		name = mangleABIName(ctxt, ldr, s, name)
		symstr.Addstring(name)

		if t := ldr.SymType(s); t == sym.SDYNIMPORT || t == sym.SHOSTOBJ || t == sym.SUNDEFEXT {
			symtab.AddUint8(0x01)                             // type N_EXT, external symbol
			symtab.AddUint8(0)                                // no section
			symtab.AddUint16(ctxt.Arch, 0)                    // desc
			symtab.AddUintXX(ctxt.Arch, 0, ctxt.Arch.PtrSize) // no value
		} else {
			if export || ldr.AttrCgoExportDynamic(s) {
				symtab.AddUint8(0x0f) // N_SECT | N_EXT
			} else if ldr.AttrCgoExportStatic(s) {
				// Only export statically, not dynamically. (N_PEXT is like hidden visibility)
				symtab.AddUint8(0x1f) // N_SECT | N_EXT | N_PEXT
			} else {
				symtab.AddUint8(0x0e) // N_SECT
			}
			o := s
			if outer := ldr.OuterSym(o); outer != 0 {
				o = outer
			}
			if ldr.SymSect(o) == nil {
				ldr.Errorf(s, "missing section for symbol")
				symtab.AddUint8(0)
			} else {
				symtab.AddUint8(uint8(ldr.SymSect(o).Extnum))
			}
			symtab.AddUint16(ctxt.Arch, 0) // desc
			symtab.AddUintXX(ctxt.Arch, uint64(ldr.SymAddr(s)), ctxt.Arch.PtrSize)
		}
	}
}

func machodysymtab(ctxt *Link, base int64) {
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

	ldr := ctxt.loader

	// must match domacholink below
	s1 := ldr.SymSize(ldr.Lookup(".machosymtab", 0))
	s2 := ldr.SymSize(ctxt.ArchSyms.LinkEditPLT)
	s3 := ldr.SymSize(ctxt.ArchSyms.LinkEditGOT)
	ml.data[12] = uint32(base + s1)     /* indirectsymoff */
	ml.data[13] = uint32((s2 + s3) / 4) /* nindirectsyms */

	ml.data[14] = 0 /* extreloff */
	ml.data[15] = 0 /* nextrel */
	ml.data[16] = 0 /* locreloff */
	ml.data[17] = 0 /* nlocrel */
}

func doMachoLink(ctxt *Link) int64 {
	machosymtab(ctxt)
	machoDyldInfo(ctxt)

	ldr := ctxt.loader

	// write data that will be linkedit section
	s1 := ldr.Lookup(".machorebase", 0)
	s2 := ldr.Lookup(".machobind", 0)
	s3 := ldr.Lookup(".machosymtab", 0)
	s4 := ctxt.ArchSyms.LinkEditPLT
	s5 := ctxt.ArchSyms.LinkEditGOT
	s6 := ldr.Lookup(".machosymstr", 0)

	size := ldr.SymSize(s1) + ldr.SymSize(s2) + ldr.SymSize(s3) + ldr.SymSize(s4) + ldr.SymSize(s5) + ldr.SymSize(s6)

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
	if size%16 != 0 {
		n := 16 - size%16
		s6b := ldr.MakeSymbolUpdater(s6)
		s6b.Grow(s6b.Size() + n)
		s6b.SetSize(s6b.Size() + n)
		size += n
	}

	if size > 0 {
		linkoff = Rnd(int64(uint64(HEADR)+Segtext.Length), int64(*FlagRound)) + Rnd(int64(Segrelrodata.Filelen), int64(*FlagRound)) + Rnd(int64(Segdata.Filelen), int64(*FlagRound)) + Rnd(int64(Segdwarf.Filelen), int64(*FlagRound))
		ctxt.Out.SeekSet(linkoff)

		ctxt.Out.Write(ldr.Data(s1))
		ctxt.Out.Write(ldr.Data(s2))
		ctxt.Out.Write(ldr.Data(s3))
		ctxt.Out.Write(ldr.Data(s4))
		ctxt.Out.Write(ldr.Data(s5))
		ctxt.Out.Write(ldr.Data(s6))

		// Add code signature if necessary. This must be the last.
		s7 := machoCodeSigSym(ctxt, linkoff+size)
		size += ldr.SymSize(s7)
	}

	return Rnd(size, int64(*FlagRound))
}

func machorelocsect(ctxt *Link, out *OutBuf, sect *sym.Section, syms []loader.Sym) {
	// If main section has no bits, nothing to relocate.
	if sect.Vaddr >= sect.Seg.Vaddr+sect.Seg.Filelen {
		return
	}
	ldr := ctxt.loader

	for i, s := range syms {
		if !ldr.AttrReachable(s) {
			continue
		}
		if uint64(ldr.SymValue(s)) >= sect.Vaddr {
			syms = syms[i:]
			break
		}
	}

	eaddr := sect.Vaddr + sect.Length
	for _, s := range syms {
		if !ldr.AttrReachable(s) {
			continue
		}
		if ldr.SymValue(s) >= int64(eaddr) {
			break
		}

		// Compute external relocations on the go, and pass to Machoreloc1
		// to stream out.
		relocs := ldr.Relocs(s)
		for ri := 0; ri < relocs.Count(); ri++ {
			r := relocs.At(ri)
			rr, ok := extreloc(ctxt, ldr, s, r)
			if !ok {
				continue
			}
			if rr.Xsym == 0 {
				ldr.Errorf(s, "missing xsym in relocation")
				continue
			}
			if !ldr.AttrReachable(rr.Xsym) {
				ldr.Errorf(s, "unreachable reloc %d (%s) target %v", r.Type(), sym.RelocName(ctxt.Arch, r.Type()), ldr.SymName(rr.Xsym))
			}
			if !thearch.Machoreloc1(ctxt.Arch, out, ldr, s, rr, int64(uint64(ldr.SymValue(s)+int64(r.Off()))-sect.Vaddr)) {
				ldr.Errorf(s, "unsupported obj reloc %d (%s)/%d to %s", r.Type(), sym.RelocName(ctxt.Arch, r.Type()), r.Siz(), ldr.SymName(r.Sym()))
			}
		}
	}

	// sanity check
	if uint64(out.Offset()) != sect.Reloff+sect.Rellen {
		panic("machorelocsect: size mismatch")
	}
}

func machoEmitReloc(ctxt *Link) {
	for ctxt.Out.Offset()&7 != 0 {
		ctxt.Out.Write8(0)
	}

	sizeExtRelocs(ctxt, thearch.MachorelocSize)
	relocSect, wg := relocSectFn(ctxt, machorelocsect)

	relocSect(ctxt, Segtext.Sections[0], ctxt.Textp)
	for _, sect := range Segtext.Sections[1:] {
		if sect.Name == ".text" {
			relocSect(ctxt, sect, ctxt.Textp)
		} else {
			relocSect(ctxt, sect, ctxt.datap)
		}
	}
	for _, sect := range Segrelrodata.Sections {
		relocSect(ctxt, sect, ctxt.datap)
	}
	for _, sect := range Segdata.Sections {
		relocSect(ctxt, sect, ctxt.datap)
	}
	for i := 0; i < len(Segdwarf.Sections); i++ {
		sect := Segdwarf.Sections[i]
		si := dwarfp[i]
		if si.secSym() != loader.Sym(sect.Sym) ||
			ctxt.loader.SymSect(si.secSym()) != sect {
			panic("inconsistency between dwarfp and Segdwarf")
		}
		relocSect(ctxt, sect, si.syms)
	}
	wg.Wait()
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

// A rebase entry tells the dynamic linker the data at sym+off needs to be
// relocated when the in-memory image moves. (This is somewhat like, say,
// ELF R_X86_64_RELATIVE).
// For now, the only kind of entry we support is that the data is an absolute
// address. That seems all we need.
// In the binary it uses a compact stateful bytecode encoding. So we record
// entries as we go and build the table at the end.
type machoRebaseRecord struct {
	sym loader.Sym
	off int64
}

var machorebase []machoRebaseRecord

func MachoAddRebase(s loader.Sym, off int64) {
	machorebase = append(machorebase, machoRebaseRecord{s, off})
}

// A bind entry tells the dynamic linker the data at GOT+off should be bound
// to the address of the target symbol, which is a dynamic import.
// For now, the only kind of entry we support is that the data is an absolute
// address, and the source symbol is always the GOT. That seems all we need.
// In the binary it uses a compact stateful bytecode encoding. So we record
// entries as we go and build the table at the end.
type machoBindRecord struct {
	off  int64
	targ loader.Sym
}

var machobind []machoBindRecord

func MachoAddBind(off int64, targ loader.Sym) {
	machobind = append(machobind, machoBindRecord{off, targ})
}

// Generate data for the dynamic linker, used in LC_DYLD_INFO_ONLY load command.
// See mach-o/loader.h, struct dyld_info_command, for the encoding.
// e.g. https://opensource.apple.com/source/xnu/xnu-6153.81.5/EXTERNAL_HEADERS/mach-o/loader.h
func machoDyldInfo(ctxt *Link) {
	ldr := ctxt.loader
	rebase := ldr.CreateSymForUpdate(".machorebase", 0)
	bind := ldr.CreateSymForUpdate(".machobind", 0)

	if !(ctxt.IsPIE() && ctxt.IsInternal()) {
		return
	}

	segId := func(seg *sym.Segment) uint8 {
		switch seg {
		case &Segtext:
			return 1
		case &Segrelrodata:
			return 2
		case &Segdata:
			if Segrelrodata.Length > 0 {
				return 3
			}
			return 2
		}
		panic("unknown segment")
	}

	dylibId := func(s loader.Sym) int {
		slib := ldr.SymDynimplib(s)
		for i, lib := range dylib {
			if lib == slib {
				return i + 1
			}
		}
		return BIND_SPECIAL_DYLIB_FLAT_LOOKUP // don't know where it is from
	}

	// Rebase table.
	// TODO: use more compact encoding. The encoding is stateful, and
	// we can use delta encoding.
	rebase.AddUint8(REBASE_OPCODE_SET_TYPE_IMM | REBASE_TYPE_POINTER)
	for _, r := range machorebase {
		seg := ldr.SymSect(r.sym).Seg
		off := uint64(ldr.SymValue(r.sym)+r.off) - seg.Vaddr
		rebase.AddUint8(REBASE_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB | segId(seg))
		rebase.AddUleb(off)

		rebase.AddUint8(REBASE_OPCODE_DO_REBASE_IMM_TIMES | 1)
	}
	rebase.AddUint8(REBASE_OPCODE_DONE)
	sz := Rnd(rebase.Size(), 8)
	rebase.Grow(sz)
	rebase.SetSize(sz)

	// Bind table.
	// TODO: compact encoding, as above.
	// TODO: lazy binding?
	got := ctxt.GOT
	seg := ldr.SymSect(got).Seg
	gotAddr := ldr.SymValue(got)
	bind.AddUint8(BIND_OPCODE_SET_TYPE_IMM | BIND_TYPE_POINTER)
	for _, r := range machobind {
		off := uint64(gotAddr+r.off) - seg.Vaddr
		bind.AddUint8(BIND_OPCODE_SET_SEGMENT_AND_OFFSET_ULEB | segId(seg))
		bind.AddUleb(off)

		d := dylibId(r.targ)
		if d > 0 && d < 128 {
			bind.AddUint8(BIND_OPCODE_SET_DYLIB_ORDINAL_IMM | uint8(d)&0xf)
		} else if d >= 128 {
			bind.AddUint8(BIND_OPCODE_SET_DYLIB_ORDINAL_ULEB)
			bind.AddUleb(uint64(d))
		} else { // d <= 0
			bind.AddUint8(BIND_OPCODE_SET_DYLIB_SPECIAL_IMM | uint8(d)&0xf)
		}

		bind.AddUint8(BIND_OPCODE_SET_SYMBOL_TRAILING_FLAGS_IMM)
		// target symbol name as a C string, with _ prefix
		bind.AddUint8('_')
		bind.Addstring(ldr.SymExtname(r.targ))

		bind.AddUint8(BIND_OPCODE_DO_BIND)
	}
	bind.AddUint8(BIND_OPCODE_DONE)
	sz = Rnd(bind.Size(), 16) // make it 16-byte aligned, see the comment in doMachoLink
	bind.Grow(sz)
	bind.SetSize(sz)

	// TODO: export table.
	// The symbols names are encoded as a trie. I'm really too lazy to do that
	// for now.
	// Without it, the symbols are not dynamically exported, so they cannot be
	// e.g. dlsym'd. But internal linking is not the default in that case, so
	// it is fine.
}

// machoCodeSigSym creates and returns a symbol for code signature.
// The symbol context is left as zeros, which will be generated at the end
// (as it depends on the rest of the file).
func machoCodeSigSym(ctxt *Link, codeSize int64) loader.Sym {
	ldr := ctxt.loader
	cs := ldr.CreateSymForUpdate(".machocodesig", 0)
	if !ctxt.NeedCodeSign() || ctxt.IsExternal() {
		return cs.Sym()
	}
	sz := codesign.Size(codeSize, "a.out")
	cs.Grow(sz)
	cs.SetSize(sz)
	return cs.Sym()
}

// machoCodeSign code-signs Mach-O file fname with an ad-hoc signature.
// This is used for updating an external linker generated binary.
func machoCodeSign(ctxt *Link, fname string) error {
	f, err := os.OpenFile(fname, os.O_RDWR, 0)
	if err != nil {
		return err
	}
	defer f.Close()

	mf, err := macho.NewFile(f)
	if err != nil {
		return err
	}
	if mf.Magic != macho.Magic64 {
		Exitf("not 64-bit Mach-O file: %s", fname)
	}

	// Find existing LC_CODE_SIGNATURE and __LINKEDIT segment
	var sigOff, sigSz, csCmdOff, linkeditOff int64
	var linkeditSeg, textSeg *macho.Segment
	loadOff := int64(machoHeaderSize64)
	get32 := mf.ByteOrder.Uint32
	for _, l := range mf.Loads {
		data := l.Raw()
		cmd, sz := get32(data), get32(data[4:])
		if cmd == LC_CODE_SIGNATURE {
			sigOff = int64(get32(data[8:]))
			sigSz = int64(get32(data[12:]))
			csCmdOff = loadOff
		}
		if seg, ok := l.(*macho.Segment); ok {
			switch seg.Name {
			case "__LINKEDIT":
				linkeditSeg = seg
				linkeditOff = loadOff
			case "__TEXT":
				textSeg = seg
			}
		}
		loadOff += int64(sz)
	}

	if sigOff == 0 {
		// The C linker doesn't generate a signed binary, for some reason.
		// Skip.
		return nil
	}

	fi, err := f.Stat()
	if err != nil {
		return err
	}
	if sigOff+sigSz != fi.Size() {
		// We don't expect anything after the signature (this will invalidate
		// the signature anyway.)
		return fmt.Errorf("unexpected content after code signature")
	}

	sz := codesign.Size(sigOff, "a.out")
	if sz != sigSz {
		// Update the load command,
		var tmp [8]byte
		mf.ByteOrder.PutUint32(tmp[:4], uint32(sz))
		_, err = f.WriteAt(tmp[:4], csCmdOff+12)
		if err != nil {
			return err
		}

		// Uodate the __LINKEDIT segment.
		segSz := sigOff + sz - int64(linkeditSeg.Offset)
		mf.ByteOrder.PutUint64(tmp[:8], uint64(segSz))
		_, err = f.WriteAt(tmp[:8], int64(linkeditOff)+int64(unsafe.Offsetof(macho.Segment64{}.Memsz)))
		if err != nil {
			return err
		}
		_, err = f.WriteAt(tmp[:8], int64(linkeditOff)+int64(unsafe.Offsetof(macho.Segment64{}.Filesz)))
		if err != nil {
			return err
		}
	}

	cs := make([]byte, sz)
	codesign.Sign(cs, f, "a.out", sigOff, int64(textSeg.Offset), int64(textSeg.Filesz), ctxt.IsExe() || ctxt.IsPIE())
	_, err = f.WriteAt(cs, sigOff)
	if err != nil {
		return err
	}
	err = f.Truncate(sigOff + sz)
	return err
}
