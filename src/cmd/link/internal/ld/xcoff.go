// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math/bits"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"cmd/internal/objabi"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
)

// This file handles all algorithms related to XCOFF files generation.
// Most of them are adaptations of the ones in  cmd/link/internal/pe.go
// as PE and XCOFF are based on COFF files.
// XCOFF files generated are 64 bits.

const (
	// Total amount of space to reserve at the start of the file
	// for File Header, Auxiliary Header, and Section Headers.
	// May waste some.
	XCOFFHDRRESERVE = FILHSZ_64 + AOUTHSZ_EXEC64 + SCNHSZ_64*23

	// base on dump -o, then rounded from 32B to 64B to
	// match worst case elf text section alignment on ppc64.
	XCOFFSECTALIGN int64 = 64

	// XCOFF binaries should normally have all its sections position-independent.
	// However, this is not yet possible for .text because of some R_ADDR relocations
	// inside RODATA symbols.
	// .data and .bss are position-independent so their address start inside an unreachable
	// segment during execution to force segfault if something is wrong.
	XCOFFTEXTBASE = 0x100000000 // Start of text address
	XCOFFDATABASE = 0x200000000 // Start of data address
)

// File Header
type XcoffFileHdr64 struct {
	Fmagic   uint16 // Target machine
	Fnscns   uint16 // Number of sections
	Ftimedat int32  // Time and date of file creation
	Fsymptr  uint64 // Byte offset to symbol table start
	Fopthdr  uint16 // Number of bytes in optional header
	Fflags   uint16 // Flags
	Fnsyms   int32  // Number of entries in symbol table
}

const (
	U64_TOCMAGIC = 0767 // AIX 64-bit XCOFF
)

// Flags that describe the type of the object file.
const (
	F_RELFLG    = 0x0001
	F_EXEC      = 0x0002
	F_LNNO      = 0x0004
	F_FDPR_PROF = 0x0010
	F_FDPR_OPTI = 0x0020
	F_DSA       = 0x0040
	F_VARPG     = 0x0100
	F_DYNLOAD   = 0x1000
	F_SHROBJ    = 0x2000
	F_LOADONLY  = 0x4000
)

// Auxiliary Header
type XcoffAoutHdr64 struct {
	Omagic      int16    // Flags - Ignored If Vstamp Is 1
	Ovstamp     int16    // Version
	Odebugger   uint32   // Reserved For Debugger
	Otextstart  uint64   // Virtual Address Of Text
	Odatastart  uint64   // Virtual Address Of Data
	Otoc        uint64   // Toc Address
	Osnentry    int16    // Section Number For Entry Point
	Osntext     int16    // Section Number For Text
	Osndata     int16    // Section Number For Data
	Osntoc      int16    // Section Number For Toc
	Osnloader   int16    // Section Number For Loader
	Osnbss      int16    // Section Number For Bss
	Oalgntext   int16    // Max Text Alignment
	Oalgndata   int16    // Max Data Alignment
	Omodtype    [2]byte  // Module Type Field
	Ocpuflag    uint8    // Bit Flags - Cputypes Of Objects
	Ocputype    uint8    // Reserved for CPU type
	Otextpsize  uint8    // Requested text page size
	Odatapsize  uint8    // Requested data page size
	Ostackpsize uint8    // Requested stack page size
	Oflags      uint8    // Flags And TLS Alignment
	Otsize      uint64   // Text Size In Bytes
	Odsize      uint64   // Data Size In Bytes
	Obsize      uint64   // Bss Size In Bytes
	Oentry      uint64   // Entry Point Address
	Omaxstack   uint64   // Max Stack Size Allowed
	Omaxdata    uint64   // Max Data Size Allowed
	Osntdata    int16    // Section Number For Tdata Section
	Osntbss     int16    // Section Number For Tbss Section
	Ox64flags   uint16   // Additional Flags For 64-Bit Objects
	Oresv3a     int16    // Reserved
	Oresv3      [2]int32 // Reserved
}

// Section Header
type XcoffScnHdr64 struct {
	Sname    [8]byte // Section Name
	Spaddr   uint64  // Physical Address
	Svaddr   uint64  // Virtual Address
	Ssize    uint64  // Section Size
	Sscnptr  uint64  // File Offset To Raw Data
	Srelptr  uint64  // File Offset To Relocation
	Slnnoptr uint64  // File Offset To Line Numbers
	Snreloc  uint32  // Number Of Relocation Entries
	Snlnno   uint32  // Number Of Line Number Entries
	Sflags   uint32  // flags
}

// Flags defining the section type.
const (
	STYP_DWARF  = 0x0010
	STYP_TEXT   = 0x0020
	STYP_DATA   = 0x0040
	STYP_BSS    = 0x0080
	STYP_EXCEPT = 0x0100
	STYP_INFO   = 0x0200
	STYP_TDATA  = 0x0400
	STYP_TBSS   = 0x0800
	STYP_LOADER = 0x1000
	STYP_DEBUG  = 0x2000
	STYP_TYPCHK = 0x4000
	STYP_OVRFLO = 0x8000
)
const (
	SSUBTYP_DWINFO  = 0x10000 // DWARF info section
	SSUBTYP_DWLINE  = 0x20000 // DWARF line-number section
	SSUBTYP_DWPBNMS = 0x30000 // DWARF public names section
	SSUBTYP_DWPBTYP = 0x40000 // DWARF public types section
	SSUBTYP_DWARNGE = 0x50000 // DWARF aranges section
	SSUBTYP_DWABREV = 0x60000 // DWARF abbreviation section
	SSUBTYP_DWSTR   = 0x70000 // DWARF strings section
	SSUBTYP_DWRNGES = 0x80000 // DWARF ranges section
	SSUBTYP_DWLOC   = 0x90000 // DWARF location lists section
	SSUBTYP_DWFRAME = 0xA0000 // DWARF frames section
	SSUBTYP_DWMAC   = 0xB0000 // DWARF macros section
)

// Headers size
const (
	FILHSZ_32      = 20
	FILHSZ_64      = 24
	AOUTHSZ_EXEC32 = 72
	AOUTHSZ_EXEC64 = 120
	SCNHSZ_32      = 40
	SCNHSZ_64      = 72
	LDHDRSZ_32     = 32
	LDHDRSZ_64     = 56
	LDSYMSZ_64     = 24
	RELSZ_64       = 14
)

// Type representing all XCOFF symbols.
type xcoffSym interface {
}

// Symbol Table Entry
type XcoffSymEnt64 struct {
	Nvalue  uint64 // Symbol value
	Noffset uint32 // Offset of the name in string table or .debug section
	Nscnum  int16  // Section number of symbol
	Ntype   uint16 // Basic and derived type specification
	Nsclass uint8  // Storage class of symbol
	Nnumaux int8   // Number of auxiliary entries
}

const SYMESZ = 18

const (
	// Nscnum
	N_DEBUG = -2
	N_ABS   = -1
	N_UNDEF = 0

	//Ntype
	SYM_V_INTERNAL  = 0x1000
	SYM_V_HIDDEN    = 0x2000
	SYM_V_PROTECTED = 0x3000
	SYM_V_EXPORTED  = 0x4000
	SYM_TYPE_FUNC   = 0x0020 // is function
)

// Storage Class.
const (
	C_NULL    = 0   // Symbol table entry marked for deletion
	C_EXT     = 2   // External symbol
	C_STAT    = 3   // Static symbol
	C_BLOCK   = 100 // Beginning or end of inner block
	C_FCN     = 101 // Beginning or end of function
	C_FILE    = 103 // Source file name and compiler information
	C_HIDEXT  = 107 // Unnamed external symbol
	C_BINCL   = 108 // Beginning of include file
	C_EINCL   = 109 // End of include file
	C_WEAKEXT = 111 // Weak external symbol
	C_DWARF   = 112 // DWARF symbol
	C_GSYM    = 128 // Global variable
	C_LSYM    = 129 // Automatic variable allocated on stack
	C_PSYM    = 130 // Argument to subroutine allocated on stack
	C_RSYM    = 131 // Register variable
	C_RPSYM   = 132 // Argument to function or procedure stored in register
	C_STSYM   = 133 // Statically allocated symbol
	C_BCOMM   = 135 // Beginning of common block
	C_ECOML   = 136 // Local member of common block
	C_ECOMM   = 137 // End of common block
	C_DECL    = 140 // Declaration of object
	C_ENTRY   = 141 // Alternate entry
	C_FUN     = 142 // Function or procedure
	C_BSTAT   = 143 // Beginning of static block
	C_ESTAT   = 144 // End of static block
	C_GTLS    = 145 // Global thread-local variable
	C_STTLS   = 146 // Static thread-local variable
)

// File Auxiliary Entry
type XcoffAuxFile64 struct {
	Xzeroes  uint32 // The name is always in the string table
	Xoffset  uint32 // Offset in the string table
	X_pad1   [6]byte
	Xftype   uint8 // Source file string type
	X_pad2   [2]byte
	Xauxtype uint8 // Type of auxiliary entry
}

// Function Auxiliary Entry
type XcoffAuxFcn64 struct {
	Xlnnoptr uint64 // File pointer to line number
	Xfsize   uint32 // Size of function in bytes
	Xendndx  uint32 // Symbol table index of next entry
	Xpad     uint8  // Unused
	Xauxtype uint8  // Type of auxiliary entry
}

// csect Auxiliary Entry.
type XcoffAuxCSect64 struct {
	Xscnlenlo uint32 // Lower 4 bytes of length or symbol table index
	Xparmhash uint32 // Offset of parameter type-check string
	Xsnhash   uint16 // .typchk section number
	Xsmtyp    uint8  // Symbol alignment and type
	Xsmclas   uint8  // Storage-mapping class
	Xscnlenhi uint32 // Upper 4 bytes of length or symbol table index
	Xpad      uint8  // Unused
	Xauxtype  uint8  // Type of auxiliary entry
}

// DWARF Auxiliary Entry
type XcoffAuxDWARF64 struct {
	Xscnlen  uint64 // Length of this symbol section
	X_pad    [9]byte
	Xauxtype uint8 // Type of auxiliary entry
}

// Auxiliary type
const (
	_AUX_EXCEPT = 255
	_AUX_FCN    = 254
	_AUX_SYM    = 253
	_AUX_FILE   = 252
	_AUX_CSECT  = 251
	_AUX_SECT   = 250
)

// Xftype field
const (
	XFT_FN = 0   // Source File Name
	XFT_CT = 1   // Compile Time Stamp
	XFT_CV = 2   // Compiler Version Number
	XFT_CD = 128 // Compiler Defined Information/

)

// Symbol type field.
const (
	XTY_ER  = 0    // External reference
	XTY_SD  = 1    // Section definition
	XTY_LD  = 2    // Label definition
	XTY_CM  = 3    // Common csect definition
	XTY_WK  = 0x8  // Weak symbol
	XTY_EXP = 0x10 // Exported symbol
	XTY_ENT = 0x20 // Entry point symbol
	XTY_IMP = 0x40 // Imported symbol
)

// Storage-mapping class.
const (
	XMC_PR     = 0  // Program code
	XMC_RO     = 1  // Read-only constant
	XMC_DB     = 2  // Debug dictionary table
	XMC_TC     = 3  // TOC entry
	XMC_UA     = 4  // Unclassified
	XMC_RW     = 5  // Read/Write data
	XMC_GL     = 6  // Global linkage
	XMC_XO     = 7  // Extended operation
	XMC_SV     = 8  // 32-bit supervisor call descriptor
	XMC_BS     = 9  // BSS class
	XMC_DS     = 10 // Function descriptor
	XMC_UC     = 11 // Unnamed FORTRAN common
	XMC_TC0    = 15 // TOC anchor
	XMC_TD     = 16 // Scalar data entry in the TOC
	XMC_SV64   = 17 // 64-bit supervisor call descriptor
	XMC_SV3264 = 18 // Supervisor call descriptor for both 32-bit and 64-bit
	XMC_TL     = 20 // Read/Write thread-local data
	XMC_UL     = 21 // Read/Write thread-local data (.tbss)
	XMC_TE     = 22 // TOC entry
)

// Loader Header
type XcoffLdHdr64 struct {
	Lversion int32  // Loader section version number
	Lnsyms   int32  // Number of symbol table entries
	Lnreloc  int32  // Number of relocation table entries
	Listlen  uint32 // Length of import file ID string table
	Lnimpid  int32  // Number of import file IDs
	Lstlen   uint32 // Length of string table
	Limpoff  uint64 // Offset to start of import file IDs
	Lstoff   uint64 // Offset to start of string table
	Lsymoff  uint64 // Offset to start of symbol table
	Lrldoff  uint64 // Offset to start of relocation entries
}

// Loader Symbol
type XcoffLdSym64 struct {
	Lvalue  uint64 // Address field
	Loffset uint32 // Byte offset into string table of symbol name
	Lscnum  int16  // Section number containing symbol
	Lsmtype int8   // Symbol type, export, import flags
	Lsmclas int8   // Symbol storage class
	Lifile  int32  // Import file ID; ordinal of import file IDs
	Lparm   uint32 // Parameter type-check field
}

type xcoffLoaderSymbol struct {
	sym    loader.Sym
	smtype int8
	smclas int8
}

type XcoffLdImportFile64 struct {
	Limpidpath string
	Limpidbase string
	Limpidmem  string
}

type XcoffLdRel64 struct {
	Lvaddr  uint64 // Address Field
	Lrtype  uint16 // Relocation Size and Type
	Lrsecnm int16  // Section Number being relocated
	Lsymndx int32  // Loader-Section symbol table index
}

// xcoffLoaderReloc holds information about a relocation made by the loader.
type xcoffLoaderReloc struct {
	sym    loader.Sym
	roff   int32
	rtype  uint16
	symndx int32
}

const (
	XCOFF_R_POS = 0x00 // A(sym) Positive Relocation
	XCOFF_R_NEG = 0x01 // -A(sym) Negative Relocation
	XCOFF_R_REL = 0x02 // A(sym-*) Relative to self
	XCOFF_R_TOC = 0x03 // A(sym-TOC) Relative to TOC
	XCOFF_R_TRL = 0x12 // A(sym-TOC) TOC Relative indirect load.

	XCOFF_R_TRLA = 0x13 // A(sym-TOC) TOC Rel load address. modifiable inst
	XCOFF_R_GL   = 0x05 // A(external TOC of sym) Global Linkage
	XCOFF_R_TCL  = 0x06 // A(local TOC of sym) Local object TOC address
	XCOFF_R_RL   = 0x0C // A(sym) Pos indirect load. modifiable instruction
	XCOFF_R_RLA  = 0x0D // A(sym) Pos Load Address. modifiable instruction
	XCOFF_R_REF  = 0x0F // AL0(sym) Non relocating ref. No garbage collect
	XCOFF_R_BA   = 0x08 // A(sym) Branch absolute. Cannot modify instruction
	XCOFF_R_RBA  = 0x18 // A(sym) Branch absolute. modifiable instruction
	XCOFF_R_BR   = 0x0A // A(sym-*) Branch rel to self. non modifiable
	XCOFF_R_RBR  = 0x1A // A(sym-*) Branch rel to self. modifiable instr

	XCOFF_R_TLS    = 0x20 // General-dynamic reference to TLS symbol
	XCOFF_R_TLS_IE = 0x21 // Initial-exec reference to TLS symbol
	XCOFF_R_TLS_LD = 0x22 // Local-dynamic reference to TLS symbol
	XCOFF_R_TLS_LE = 0x23 // Local-exec reference to TLS symbol
	XCOFF_R_TLSM   = 0x24 // Module reference to TLS symbol
	XCOFF_R_TLSML  = 0x25 // Module reference to local (own) module

	XCOFF_R_TOCU = 0x30 // Relative to TOC - high order bits
	XCOFF_R_TOCL = 0x31 // Relative to TOC - low order bits
)

type XcoffLdStr64 struct {
	size uint16
	name string
}

// xcoffFile is used to build XCOFF file.
type xcoffFile struct {
	xfhdr           XcoffFileHdr64
	xahdr           XcoffAoutHdr64
	sections        []*XcoffScnHdr64
	sectText        *XcoffScnHdr64
	sectData        *XcoffScnHdr64
	sectBss         *XcoffScnHdr64
	stringTable     xcoffStringTable
	sectNameToScnum map[string]int16
	loaderSize      uint64
	symtabOffset    int64                // offset to the start of symbol table
	symbolCount     uint32               // number of symbol table records written
	symtabSym       []xcoffSym           // XCOFF symbols for the symbol table
	dynLibraries    map[string]int       // Dynamic libraries in .loader section. The integer represents its import file number (- 1)
	loaderSymbols   []*xcoffLoaderSymbol // symbols inside .loader symbol table
	loaderReloc     []*xcoffLoaderReloc  // Reloc that must be made inside loader
	sync.Mutex                           // currently protect loaderReloc
}

// Var used by XCOFF Generation algorithms
var (
	xfile xcoffFile
)

// xcoffStringTable is a XCOFF string table.
type xcoffStringTable struct {
	strings    []string
	stringsLen int
}

// size returns size of string table t.
func (t *xcoffStringTable) size() int {
	// string table starts with 4-byte length at the beginning
	return t.stringsLen + 4
}

// add adds string str to string table t.
func (t *xcoffStringTable) add(str string) int {
	off := t.size()
	t.strings = append(t.strings, str)
	t.stringsLen += len(str) + 1 // each string will have 0 appended to it
	return off
}

// write writes string table t into the output file.
func (t *xcoffStringTable) write(out *OutBuf) {
	out.Write32(uint32(t.size()))
	for _, s := range t.strings {
		out.WriteString(s)
		out.Write8(0)
	}
}

// write writes XCOFF section sect into the output file.
func (sect *XcoffScnHdr64) write(ctxt *Link) {
	binary.Write(ctxt.Out, binary.BigEndian, sect)
	ctxt.Out.Write32(0) // Add 4 empty bytes at the end to match alignment
}

// addSection adds section to the XCOFF file f.
func (f *xcoffFile) addSection(name string, addr uint64, size uint64, fileoff uint64, flags uint32) *XcoffScnHdr64 {
	sect := &XcoffScnHdr64{
		Spaddr:  addr,
		Svaddr:  addr,
		Ssize:   size,
		Sscnptr: fileoff,
		Sflags:  flags,
	}
	copy(sect.Sname[:], name) // copy string to [8]byte
	f.sections = append(f.sections, sect)
	f.sectNameToScnum[name] = int16(len(f.sections))
	return sect
}

// addDwarfSection adds a dwarf section to the XCOFF file f.
// This function is similar to addSection, but Dwarf section names
// must be modified to conventional names and they are various subtypes.
func (f *xcoffFile) addDwarfSection(s *sym.Section) *XcoffScnHdr64 {
	newName, subtype := xcoffGetDwarfSubtype(s.Name)
	return f.addSection(newName, 0, s.Length, s.Seg.Fileoff+s.Vaddr-s.Seg.Vaddr, STYP_DWARF|subtype)
}

// xcoffGetDwarfSubtype returns the XCOFF name of the DWARF section str
// and its subtype constant.
func xcoffGetDwarfSubtype(str string) (string, uint32) {
	switch str {
	default:
		Exitf("unknown DWARF section name for XCOFF: %s", str)
	case ".debug_abbrev":
		return ".dwabrev", SSUBTYP_DWABREV
	case ".debug_info":
		return ".dwinfo", SSUBTYP_DWINFO
	case ".debug_frame":
		return ".dwframe", SSUBTYP_DWFRAME
	case ".debug_line":
		return ".dwline", SSUBTYP_DWLINE
	case ".debug_loc":
		return ".dwloc", SSUBTYP_DWLOC
	case ".debug_pubnames":
		return ".dwpbnms", SSUBTYP_DWPBNMS
	case ".debug_pubtypes":
		return ".dwpbtyp", SSUBTYP_DWPBTYP
	case ".debug_ranges":
		return ".dwrnges", SSUBTYP_DWRNGES
	}
	// never used
	return "", 0
}

// getXCOFFscnum returns the XCOFF section number of a Go section.
func (f *xcoffFile) getXCOFFscnum(sect *sym.Section) int16 {
	switch sect.Seg {
	case &Segtext:
		return f.sectNameToScnum[".text"]
	case &Segdata:
		if sect.Name == ".noptrbss" || sect.Name == ".bss" {
			return f.sectNameToScnum[".bss"]
		}
		if sect.Name == ".tbss" {
			return f.sectNameToScnum[".tbss"]
		}
		return f.sectNameToScnum[".data"]
	case &Segdwarf:
		name, _ := xcoffGetDwarfSubtype(sect.Name)
		return f.sectNameToScnum[name]
	case &Segrelrodata:
		return f.sectNameToScnum[".data"]
	}
	Errorf("getXCOFFscnum not implemented for section %s", sect.Name)
	return -1
}

// Xcoffinit initialised some internal value and setups
// already known header information.
func Xcoffinit(ctxt *Link) {
	xfile.dynLibraries = make(map[string]int)

	HEADR = int32(Rnd(XCOFFHDRRESERVE, XCOFFSECTALIGN))
	if *FlagRound != -1 {
		Errorf("-R not available on AIX")
	}
	*FlagRound = XCOFFSECTALIGN
	if *FlagTextAddr != -1 {
		Errorf("-T not available on AIX")
	}
	*FlagTextAddr = Rnd(XCOFFTEXTBASE, *FlagRound) + int64(HEADR)
}

// SYMBOL TABLE

// type records C_FILE information needed for genasmsym in XCOFF.
type xcoffSymSrcFile struct {
	name         string
	file         *XcoffSymEnt64   // Symbol of this C_FILE
	csectAux     *XcoffAuxCSect64 // Symbol for the current .csect
	csectSymNb   uint64           // Symbol number for the current .csect
	csectVAStart int64
	csectVAEnd   int64
}

var (
	currDwscnoff   = make(map[string]uint64) // Needed to create C_DWARF symbols
	currSymSrcFile xcoffSymSrcFile
	outerSymSize   = make(map[string]int64)
)

// xcoffUpdateOuterSize stores the size of outer symbols in order to have it
// in the symbol table.
func xcoffUpdateOuterSize(ctxt *Link, size int64, stype sym.SymKind) {
	if size == 0 {
		return
	}
	// TODO: use CarrierSymByType

	ldr := ctxt.loader
	switch stype {
	default:
		Errorf("unknown XCOFF outer symbol for type %s", stype.String())
	case sym.SRODATA, sym.SRODATARELRO, sym.SFUNCTAB, sym.SSTRING:
		// Nothing to do
	case sym.STYPERELRO:
		if ctxt.UseRelro() && (ctxt.BuildMode == BuildModeCArchive || ctxt.BuildMode == BuildModeCShared || ctxt.BuildMode == BuildModePIE) {
			// runtime.types size must be removed, as it's a real symbol.
			tsize := ldr.SymSize(ldr.Lookup("runtime.types", 0))
			outerSymSize["typerel.*"] = size - tsize
			return
		}
		fallthrough
	case sym.STYPE:
		if !ctxt.DynlinkingGo() {
			// runtime.types size must be removed, as it's a real symbol.
			tsize := ldr.SymSize(ldr.Lookup("runtime.types", 0))
			outerSymSize["type:*"] = size - tsize
		}
	case sym.SGOSTRING:
		outerSymSize["go:string.*"] = size
	case sym.SGOFUNC:
		if !ctxt.DynlinkingGo() {
			outerSymSize["go:func.*"] = size
		}
	case sym.SGOFUNCRELRO:
		outerSymSize["go:funcrel.*"] = size
	case sym.SGCBITS:
		outerSymSize["runtime.gcbits.*"] = size
	case sym.SPCLNTAB:
		outerSymSize["runtime.pclntab"] = size
	}
}

// addSymbol writes a symbol or an auxiliary symbol entry on ctxt.out.
func (f *xcoffFile) addSymbol(sym xcoffSym) {
	f.symtabSym = append(f.symtabSym, sym)
	f.symbolCount++
}

// xcoffAlign returns the log base 2 of the symbol's alignment.
func xcoffAlign(ldr *loader.Loader, x loader.Sym, t SymbolType) uint8 {
	align := ldr.SymAlign(x)
	if align == 0 {
		if t == TextSym {
			align = int32(Funcalign)
		} else {
			align = symalign(ldr, x)
		}
	}
	return logBase2(int(align))
}

// logBase2 returns the log in base 2 of a.
func logBase2(a int) uint8 {
	return uint8(bits.Len(uint(a)) - 1)
}

// Write symbols needed when a new file appeared:
// - a C_FILE with one auxiliary entry for its name
// - C_DWARF symbols to provide debug information
// - a C_HIDEXT which will be a csect containing all of its functions
// It needs several parameters to create .csect symbols such as its entry point and its section number.
//
// Currently, a new file is in fact a new package. It seems to be OK, but it might change
// in the future.
func (f *xcoffFile) writeSymbolNewFile(ctxt *Link, name string, firstEntry uint64, extnum int16) {
	ldr := ctxt.loader
	/* C_FILE */
	s := &XcoffSymEnt64{
		Noffset: uint32(f.stringTable.add(".file")),
		Nsclass: C_FILE,
		Nscnum:  N_DEBUG,
		Ntype:   0, // Go isn't inside predefined language.
		Nnumaux: 1,
	}
	f.addSymbol(s)
	currSymSrcFile.file = s

	// Auxiliary entry for file name.
	auxf := &XcoffAuxFile64{
		Xoffset:  uint32(f.stringTable.add(name)),
		Xftype:   XFT_FN,
		Xauxtype: _AUX_FILE,
	}
	f.addSymbol(auxf)

	/* Dwarf */
	for _, sect := range Segdwarf.Sections {
		var dwsize uint64
		if ctxt.LinkMode == LinkInternal {
			// Find the size of this corresponding package DWARF compilation unit.
			// This size is set during DWARF generation (see dwarf.go).
			dwsize = getDwsectCUSize(sect.Name, name)
			// .debug_abbrev is common to all packages and not found with the previous function
			if sect.Name == ".debug_abbrev" {
				dwsize = uint64(ldr.SymSize(loader.Sym(sect.Sym)))

			}
		} else {
			// There is only one .FILE with external linking.
			dwsize = sect.Length
		}

		// get XCOFF name
		name, _ := xcoffGetDwarfSubtype(sect.Name)
		s := &XcoffSymEnt64{
			Nvalue:  currDwscnoff[sect.Name],
			Noffset: uint32(f.stringTable.add(name)),
			Nsclass: C_DWARF,
			Nscnum:  f.getXCOFFscnum(sect),
			Nnumaux: 1,
		}

		if currSymSrcFile.csectAux == nil {
			// Dwarf relocations need the symbol number of .dw* symbols.
			// It doesn't need to know it for each package, one is enough.
			// currSymSrcFile.csectAux == nil means first package.
			ldr.SetSymDynid(loader.Sym(sect.Sym), int32(f.symbolCount))

			if sect.Name == ".debug_frame" && ctxt.LinkMode != LinkExternal {
				// CIE size must be added to the first package.
				dwsize += 48
			}
		}

		f.addSymbol(s)

		// update the DWARF section offset in this file
		if sect.Name != ".debug_abbrev" {
			currDwscnoff[sect.Name] += dwsize
		}

		// Auxiliary dwarf section
		auxd := &XcoffAuxDWARF64{
			Xscnlen:  dwsize,
			Xauxtype: _AUX_SECT,
		}

		f.addSymbol(auxd)
	}

	/* .csect */
	// Check if extnum is in text.
	// This is temporary and only here to check if this algorithm is correct.
	if extnum != 1 {
		Exitf("XCOFF symtab: A new file was detected with its first symbol not in .text")
	}

	currSymSrcFile.csectSymNb = uint64(f.symbolCount)

	// No offset because no name
	s = &XcoffSymEnt64{
		Nvalue:  firstEntry,
		Nscnum:  extnum,
		Nsclass: C_HIDEXT,
		Ntype:   0, // check visibility ?
		Nnumaux: 1,
	}
	f.addSymbol(s)

	aux := &XcoffAuxCSect64{
		Xsmclas:  XMC_PR,
		Xsmtyp:   XTY_SD | logBase2(Funcalign)<<3,
		Xauxtype: _AUX_CSECT,
	}
	f.addSymbol(aux)

	currSymSrcFile.csectAux = aux
	currSymSrcFile.csectVAStart = int64(firstEntry)
	currSymSrcFile.csectVAEnd = int64(firstEntry)
}

// Update values for the previous package.
//   - Svalue of the C_FILE symbol: if it is the last one, this Svalue must be -1
//   - Xsclen of the csect symbol.
func (f *xcoffFile) updatePreviousFile(ctxt *Link, last bool) {
	// first file
	if currSymSrcFile.file == nil {
		return
	}

	// Update C_FILE
	cfile := currSymSrcFile.file
	if last {
		cfile.Nvalue = 0xFFFFFFFFFFFFFFFF
	} else {
		cfile.Nvalue = uint64(f.symbolCount)
	}

	// update csect scnlen in this auxiliary entry
	aux := currSymSrcFile.csectAux
	csectSize := currSymSrcFile.csectVAEnd - currSymSrcFile.csectVAStart
	aux.Xscnlenlo = uint32(csectSize & 0xFFFFFFFF)
	aux.Xscnlenhi = uint32(csectSize >> 32)
}

// Write symbol representing a .text function.
// The symbol table is split with C_FILE corresponding to each package
// and not to each source file as it should be.
func (f *xcoffFile) writeSymbolFunc(ctxt *Link, x loader.Sym) []xcoffSym {
	// New XCOFF symbols which will be written.
	syms := []xcoffSym{}

	// Check if a new file is detected.
	ldr := ctxt.loader
	name := ldr.SymName(x)
	if strings.Contains(name, "-tramp") || strings.HasPrefix(name, "runtime.text.") {
		// Trampoline don't have a FILE so there are considered
		// in the current file.
		// Same goes for runtime.text.X symbols.
	} else if ldr.SymPkg(x) == "" { // Undefined global symbol
		// If this happens, the algorithm must be redone.
		if currSymSrcFile.name != "" {
			Exitf("undefined global symbol found inside another file")
		}
	} else {
		// Current file has changed. New C_FILE, C_DWARF, etc must be generated.
		if currSymSrcFile.name != ldr.SymPkg(x) {
			if ctxt.LinkMode == LinkInternal {
				// update previous file values
				xfile.updatePreviousFile(ctxt, false)
				currSymSrcFile.name = ldr.SymPkg(x)
				f.writeSymbolNewFile(ctxt, ldr.SymPkg(x), uint64(ldr.SymValue(x)), xfile.getXCOFFscnum(ldr.SymSect(x)))
			} else {
				// With external linking, ld will crash if there is several
				// .FILE and DWARF debugging enable, somewhere during
				// the relocation phase.
				// Therefore, all packages are merged under a fake .FILE
				// "go_functions".
				// TODO(aix); remove once ld has been fixed or the triggering
				// relocation has been found and fixed.
				if currSymSrcFile.name == "" {
					currSymSrcFile.name = ldr.SymPkg(x)
					f.writeSymbolNewFile(ctxt, "go_functions", uint64(ldr.SymValue(x)), xfile.getXCOFFscnum(ldr.SymSect(x)))
				}
			}

		}
	}

	name = ldr.SymExtname(x)
	name = mangleABIName(ctxt, ldr, x, name)

	s := &XcoffSymEnt64{
		Nsclass: C_EXT,
		Noffset: uint32(xfile.stringTable.add(name)),
		Nvalue:  uint64(ldr.SymValue(x)),
		Nscnum:  f.getXCOFFscnum(ldr.SymSect(x)),
		Ntype:   SYM_TYPE_FUNC,
		Nnumaux: 2,
	}

	if ldr.IsFileLocal(x) || ldr.AttrVisibilityHidden(x) || ldr.AttrLocal(x) {
		s.Nsclass = C_HIDEXT
	}

	ldr.SetSymDynid(x, int32(xfile.symbolCount))
	syms = append(syms, s)

	// Keep track of the section size by tracking the VA range. Individual
	// alignment differences may introduce a few extra bytes of padding
	// which are not fully accounted for by ldr.SymSize(x).
	sv := ldr.SymValue(x) + ldr.SymSize(x)
	if currSymSrcFile.csectVAEnd < sv {
		currSymSrcFile.csectVAEnd = sv
	}

	// create auxiliary entries
	a2 := &XcoffAuxFcn64{
		Xfsize:   uint32(ldr.SymSize(x)),
		Xlnnoptr: 0,                     // TODO
		Xendndx:  xfile.symbolCount + 3, // this symbol + 2 aux entries
		Xauxtype: _AUX_FCN,
	}
	syms = append(syms, a2)

	a4 := &XcoffAuxCSect64{
		Xscnlenlo: uint32(currSymSrcFile.csectSymNb & 0xFFFFFFFF),
		Xscnlenhi: uint32(currSymSrcFile.csectSymNb >> 32),
		Xsmclas:   XMC_PR, // Program Code
		Xsmtyp:    XTY_LD, // label definition (based on C)
		Xauxtype:  _AUX_CSECT,
	}
	a4.Xsmtyp |= uint8(xcoffAlign(ldr, x, TextSym) << 3)

	syms = append(syms, a4)
	return syms
}

// put function used by genasmsym to write symbol table.
func putaixsym(ctxt *Link, x loader.Sym, t SymbolType) {
	// All XCOFF symbols generated by this GO symbols
	// Can be a symbol entry or an auxiliary entry
	syms := []xcoffSym{}

	ldr := ctxt.loader
	name := ldr.SymName(x)
	if t == UndefinedSym {
		name = ldr.SymExtname(x)
	}

	switch t {
	default:
		return

	case TextSym:
		if ldr.SymPkg(x) != "" || strings.Contains(name, "-tramp") || strings.HasPrefix(name, "runtime.text.") {
			// Function within a file
			syms = xfile.writeSymbolFunc(ctxt, x)
		} else {
			// Only runtime.text and runtime.etext come through this way
			if name != "runtime.text" && name != "runtime.etext" && name != "go:buildid" {
				Exitf("putaixsym: unknown text symbol %s", name)
			}
			s := &XcoffSymEnt64{
				Nsclass: C_HIDEXT,
				Noffset: uint32(xfile.stringTable.add(name)),
				Nvalue:  uint64(ldr.SymValue(x)),
				Nscnum:  xfile.getXCOFFscnum(ldr.SymSect(x)),
				Ntype:   SYM_TYPE_FUNC,
				Nnumaux: 1,
			}
			ldr.SetSymDynid(x, int32(xfile.symbolCount))
			syms = append(syms, s)

			size := uint64(ldr.SymSize(x))
			a4 := &XcoffAuxCSect64{
				Xauxtype:  _AUX_CSECT,
				Xscnlenlo: uint32(size & 0xFFFFFFFF),
				Xscnlenhi: uint32(size >> 32),
				Xsmclas:   XMC_PR,
				Xsmtyp:    XTY_SD,
			}
			a4.Xsmtyp |= uint8(xcoffAlign(ldr, x, TextSym) << 3)
			syms = append(syms, a4)
		}

	case DataSym, BSSSym:
		s := &XcoffSymEnt64{
			Nsclass: C_EXT,
			Noffset: uint32(xfile.stringTable.add(name)),
			Nvalue:  uint64(ldr.SymValue(x)),
			Nscnum:  xfile.getXCOFFscnum(ldr.SymSect(x)),
			Nnumaux: 1,
		}

		if ldr.IsFileLocal(x) || ldr.AttrVisibilityHidden(x) || ldr.AttrLocal(x) {
			// There is more symbols in the case of a global data
			// which are related to the assembly generated
			// to access such symbols.
			// But as Golang as its own way to check if a symbol is
			// global or local (the capital letter), we don't need to
			// implement them yet.
			s.Nsclass = C_HIDEXT
		}

		ldr.SetSymDynid(x, int32(xfile.symbolCount))
		syms = append(syms, s)

		// Create auxiliary entry

		// Normally, size should be the size of csect containing all
		// the data and bss symbols of one file/package.
		// However, it's easier to just have a csect for each symbol.
		// It might change
		size := uint64(ldr.SymSize(x))
		a4 := &XcoffAuxCSect64{
			Xauxtype:  _AUX_CSECT,
			Xscnlenlo: uint32(size & 0xFFFFFFFF),
			Xscnlenhi: uint32(size >> 32),
		}

		if ty := ldr.SymType(x); ty >= sym.STYPE && ty <= sym.SPCLNTAB {
			if ctxt.IsExternal() && strings.HasPrefix(ldr.SymSect(x).Name, ".data.rel.ro") {
				// During external linking, read-only datas with relocation
				// must be in .data.
				a4.Xsmclas = XMC_RW
			} else {
				// Read only data
				a4.Xsmclas = XMC_RO
			}
		} else if /*ty == sym.SDATA &&*/ strings.HasPrefix(ldr.SymName(x), "TOC.") && ctxt.IsExternal() {
			a4.Xsmclas = XMC_TC
		} else if ldr.SymName(x) == "TOC" {
			a4.Xsmclas = XMC_TC0
		} else {
			a4.Xsmclas = XMC_RW
		}
		if t == DataSym {
			a4.Xsmtyp |= XTY_SD
		} else {
			a4.Xsmtyp |= XTY_CM
		}

		a4.Xsmtyp |= uint8(xcoffAlign(ldr, x, t) << 3)

		syms = append(syms, a4)

	case UndefinedSym:
		if ty := ldr.SymType(x); ty != sym.SDYNIMPORT && ty != sym.SHOSTOBJ && ty != sym.SUNDEFEXT {
			return
		}
		s := &XcoffSymEnt64{
			Nsclass: C_EXT,
			Noffset: uint32(xfile.stringTable.add(name)),
			Nnumaux: 1,
		}
		ldr.SetSymDynid(x, int32(xfile.symbolCount))
		syms = append(syms, s)

		a4 := &XcoffAuxCSect64{
			Xauxtype: _AUX_CSECT,
			Xsmclas:  XMC_DS,
			Xsmtyp:   XTY_ER | XTY_IMP,
		}

		if ldr.SymName(x) == "__n_pthreads" {
			// Currently, all imported symbols made by cgo_import_dynamic are
			// syscall functions, except __n_pthreads which is a variable.
			// TODO(aix): Find a way to detect variables imported by cgo.
			a4.Xsmclas = XMC_RW
		}

		syms = append(syms, a4)

	case TLSSym:
		s := &XcoffSymEnt64{
			Nsclass: C_EXT,
			Noffset: uint32(xfile.stringTable.add(name)),
			Nscnum:  xfile.getXCOFFscnum(ldr.SymSect(x)),
			Nvalue:  uint64(ldr.SymValue(x)),
			Nnumaux: 1,
		}

		ldr.SetSymDynid(x, int32(xfile.symbolCount))
		syms = append(syms, s)

		size := uint64(ldr.SymSize(x))
		a4 := &XcoffAuxCSect64{
			Xauxtype:  _AUX_CSECT,
			Xsmclas:   XMC_UL,
			Xsmtyp:    XTY_CM,
			Xscnlenlo: uint32(size & 0xFFFFFFFF),
			Xscnlenhi: uint32(size >> 32),
		}

		syms = append(syms, a4)
	}

	for _, s := range syms {
		xfile.addSymbol(s)
	}
}

// Generate XCOFF Symbol table.
// It will be written in out file in Asmbxcoff, because it must be
// at the very end, especially after relocation sections which needs symbols' index.
func (f *xcoffFile) asmaixsym(ctxt *Link) {
	ldr := ctxt.loader
	// Get correct size for symbols wrapping others symbols like go.string.*
	// sym.Size can be used directly as the symbols have already been written.
	for name, size := range outerSymSize {
		sym := ldr.Lookup(name, 0)
		if sym == 0 {
			Errorf("unknown outer symbol with name %s", name)
		} else {
			s := ldr.MakeSymbolUpdater(sym)
			s.SetSize(size)
		}
	}

	// These symbols won't show up in the first loop below because we
	// skip sym.STEXT symbols. Normal sym.STEXT symbols are emitted by walking textp.
	s := ldr.Lookup("runtime.text", 0)
	if ldr.SymType(s).IsText() {
		// We've already included this symbol in ctxt.Textp on AIX with external linker.
		// See data.go:/textaddress
		if !ctxt.IsExternal() {
			putaixsym(ctxt, s, TextSym)
		}
	}

	n := 1
	// Generate base addresses for all text sections if there are multiple
	for _, sect := range Segtext.Sections[1:] {
		if sect.Name != ".text" || ctxt.IsExternal() {
			// On AIX, runtime.text.X are symbols already in the symtab.
			break
		}
		s = ldr.Lookup(fmt.Sprintf("runtime.text.%d", n), 0)
		if s == 0 {
			break
		}
		if ldr.SymType(s).IsText() {
			putaixsym(ctxt, s, TextSym)
		}
		n++
	}

	s = ldr.Lookup("runtime.etext", 0)
	if ldr.SymType(s).IsText() {
		// We've already included this symbol in ctxt.Textp
		// on AIX with external linker.
		// See data.go:/textaddress
		if !ctxt.IsExternal() {
			putaixsym(ctxt, s, TextSym)
		}
	}

	shouldBeInSymbolTable := func(s loader.Sym, name string) bool {
		if ldr.AttrNotInSymbolTable(s) {
			return false
		}
		if (name == "" || name[0] == '.') && !ldr.IsFileLocal(s) && name != ".TOC." {
			return false
		}
		return true
	}

	for s, nsym := loader.Sym(1), loader.Sym(ldr.NSym()); s < nsym; s++ {
		if !shouldBeInSymbolTable(s, ldr.SymName(s)) {
			continue
		}
		st := ldr.SymType(s)
		switch {
		case st == sym.STLSBSS:
			if ctxt.IsExternal() {
				putaixsym(ctxt, s, TLSSym)
			}

		case st == sym.SBSS, st == sym.SNOPTRBSS, st == sym.SLIBFUZZER_8BIT_COUNTER, st == sym.SCOVERAGE_COUNTER:
			if ldr.AttrReachable(s) {
				data := ldr.Data(s)
				if len(data) > 0 {
					ldr.Errorf(s, "should not be bss (size=%d type=%v special=%v)", len(data), ldr.SymType(s), ldr.AttrSpecial(s))
				}
				putaixsym(ctxt, s, BSSSym)
			}

		case st >= sym.SELFRXSECT && st < sym.SXREF: // data sections handled in dodata
			if ldr.AttrReachable(s) {
				putaixsym(ctxt, s, DataSym)
			}

		case st == sym.SUNDEFEXT:
			putaixsym(ctxt, s, UndefinedSym)

		case st == sym.SDYNIMPORT:
			if ldr.AttrReachable(s) {
				putaixsym(ctxt, s, UndefinedSym)
			}
		}
	}

	for _, s := range ctxt.Textp {
		putaixsym(ctxt, s, TextSym)
	}

	if ctxt.Debugvlog != 0 {
		ctxt.Logf("symsize = %d\n", uint32(symSize))
	}
	xfile.updatePreviousFile(ctxt, true)
}

func (f *xcoffFile) genDynSym(ctxt *Link) {
	ldr := ctxt.loader
	var dynsyms []loader.Sym
	for s := loader.Sym(1); s < loader.Sym(ldr.NSym()); s++ {
		if !ldr.AttrReachable(s) {
			continue
		}
		if t := ldr.SymType(s); t != sym.SHOSTOBJ && t != sym.SDYNIMPORT {
			continue
		}
		dynsyms = append(dynsyms, s)
	}

	for _, s := range dynsyms {
		f.adddynimpsym(ctxt, s)

		if _, ok := f.dynLibraries[ldr.SymDynimplib(s)]; !ok {
			f.dynLibraries[ldr.SymDynimplib(s)] = len(f.dynLibraries)
		}
	}
}

// (*xcoffFile)adddynimpsym adds the dynamic symbol "s" to a XCOFF file.
// A new symbol named s.Extname() is created to be the actual dynamic symbol
// in the .loader section and in the symbol table as an External Reference.
// The symbol "s" is transformed to SXCOFFTOC to end up in .data section.
// However, there is no writing protection on those symbols and
// it might need to be added.
// TODO(aix): Handles dynamic symbols without library.
func (f *xcoffFile) adddynimpsym(ctxt *Link, s loader.Sym) {
	// Check that library name is given.
	// Pattern is already checked when compiling.
	ldr := ctxt.loader
	if ctxt.IsInternal() && ldr.SymDynimplib(s) == "" {
		ctxt.Errorf(s, "imported symbol must have a given library")
	}

	sb := ldr.MakeSymbolUpdater(s)
	sb.SetReachable(true)
	sb.SetType(sym.SXCOFFTOC)

	// Create new dynamic symbol
	extsym := ldr.CreateSymForUpdate(ldr.SymExtname(s), 0)
	extsym.SetType(sym.SDYNIMPORT)
	extsym.SetDynimplib(ldr.SymDynimplib(s))
	extsym.SetExtname(ldr.SymExtname(s))
	extsym.SetDynimpvers(ldr.SymDynimpvers(s))

	// Add loader symbol
	lds := &xcoffLoaderSymbol{
		sym:    extsym.Sym(),
		smtype: XTY_IMP,
		smclas: XMC_DS,
	}
	if ldr.SymName(s) == "__n_pthreads" {
		// Currently, all imported symbols made by cgo_import_dynamic are
		// syscall functions, except __n_pthreads which is a variable.
		// TODO(aix): Find a way to detect variables imported by cgo.
		lds.smclas = XMC_RW
	}
	f.loaderSymbols = append(f.loaderSymbols, lds)

	// Relocation to retrieve the external address
	sb.AddBytes(make([]byte, 8))
	r, _ := sb.AddRel(objabi.R_ADDR)
	r.SetSym(extsym.Sym())
	r.SetSiz(uint8(ctxt.Arch.PtrSize))
	// TODO: maybe this could be
	// sb.SetSize(0)
	// sb.SetData(nil)
	// sb.AddAddr(ctxt.Arch, extsym.Sym())
	// If the size is not 0 to begin with, I don't think the added 8 bytes
	// of zeros are necessary.
}

// Xcoffadddynrel adds a dynamic relocation in a XCOFF file.
// This relocation will be made by the loader.
func Xcoffadddynrel(target *Target, ldr *loader.Loader, syms *ArchSyms, s loader.Sym, r loader.Reloc, rIdx int) bool {
	if target.IsExternal() {
		return true
	}
	if ldr.SymType(s) <= sym.SPCLNTAB {
		ldr.Errorf(s, "cannot have a relocation to %s in a text section symbol", ldr.SymName(r.Sym()))
		return false
	}

	xldr := &xcoffLoaderReloc{
		sym:  s,
		roff: r.Off(),
	}
	targ := r.Sym()
	var targType sym.SymKind
	if targ != 0 {
		targType = ldr.SymType(targ)
	}

	switch r.Type() {
	default:
		ldr.Errorf(s, "unexpected .loader relocation to symbol: %s (type: %s)", ldr.SymName(targ), r.Type().String())
		return false
	case objabi.R_ADDR:
		if ldr.SymType(s) == sym.SXCOFFTOC && targType == sym.SDYNIMPORT {
			// Imported symbol relocation
			for i, dynsym := range xfile.loaderSymbols {
				if ldr.SymName(dynsym.sym) == ldr.SymName(targ) {
					xldr.symndx = int32(i + 3) // +3 because of 3 section symbols
					break
				}
			}
		} else if t := ldr.SymType(s); t.IsDATA() || t.IsNOPTRDATA() || t == sym.SBUILDINFO || t == sym.SXCOFFTOC {
			switch ldr.SymSect(targ).Seg {
			default:
				ldr.Errorf(s, "unknown segment for .loader relocation with symbol %s", ldr.SymName(targ))
			case &Segtext:
			case &Segrodata:
				xldr.symndx = 0 // .text
			case &Segdata:
				if targType == sym.SBSS || targType == sym.SNOPTRBSS {
					xldr.symndx = 2 // .bss
				} else {
					xldr.symndx = 1 // .data
				}
			}

		} else {
			ldr.Errorf(s, "unexpected type for .loader relocation R_ADDR for symbol %s: %s to %s", ldr.SymName(targ), ldr.SymType(s), ldr.SymType(targ))
			return false
		}

		xldr.rtype = 0x3F<<8 + XCOFF_R_POS
	}

	xfile.Lock()
	xfile.loaderReloc = append(xfile.loaderReloc, xldr)
	xfile.Unlock()
	return true
}

func (ctxt *Link) doxcoff() {
	ldr := ctxt.loader

	// TOC
	toc := ldr.CreateSymForUpdate("TOC", 0)
	toc.SetType(sym.SXCOFFTOC)
	toc.SetVisibilityHidden(true)

	// Add entry point to .loader symbols.
	ep := ldr.Lookup(*flagEntrySymbol, 0)
	if ep == 0 || !ldr.AttrReachable(ep) {
		Exitf("wrong entry point")
	}

	xfile.loaderSymbols = append(xfile.loaderSymbols, &xcoffLoaderSymbol{
		sym:    ep,
		smtype: XTY_ENT | XTY_SD,
		smclas: XMC_DS,
	})

	xfile.genDynSym(ctxt)

	for s := loader.Sym(1); s < loader.Sym(ldr.NSym()); s++ {
		if strings.HasPrefix(ldr.SymName(s), "TOC.") {
			sb := ldr.MakeSymbolUpdater(s)
			sb.SetType(sym.SXCOFFTOC)
		}
	}

	if ctxt.IsExternal() {
		// Change rt0_go name to match name in runtime/cgo:main().
		rt0 := ldr.Lookup("runtime.rt0_go", 0)
		ldr.SetSymExtname(rt0, "runtime_rt0_go")

		nsym := loader.Sym(ldr.NSym())
		for s := loader.Sym(1); s < nsym; s++ {
			if !ldr.AttrCgoExport(s) {
				continue
			}
			if ldr.IsFileLocal(s) {
				panic("cgo_export on static symbol")
			}

			if ldr.SymType(s).IsText() {
				// On AIX, an exported function must have two symbols:
				// - a .text symbol which must start with a ".".
				// - a .data symbol which is a function descriptor.
				name := ldr.SymExtname(s)
				ldr.SetSymExtname(s, "."+name)

				desc := ldr.MakeSymbolUpdater(ldr.CreateExtSym(name, 0))
				desc.SetReachable(true)
				desc.SetType(sym.SNOPTRDATA)
				desc.AddAddr(ctxt.Arch, s)
				desc.AddAddr(ctxt.Arch, toc.Sym())
				desc.AddUint64(ctxt.Arch, 0)
			}
		}
	}
}

// Loader section
// Currently, this section is created from scratch when assembling the XCOFF file
// according to information retrieved in xfile object.

// Create loader section and returns its size.
func Loaderblk(ctxt *Link, off uint64) {
	xfile.writeLdrScn(ctxt, off)
}

func (f *xcoffFile) writeLdrScn(ctxt *Link, globalOff uint64) {
	var symtab []*XcoffLdSym64
	var strtab []*XcoffLdStr64
	var importtab []*XcoffLdImportFile64
	var reloctab []*XcoffLdRel64
	var dynimpreloc []*XcoffLdRel64

	// As the string table is updated in any loader subsection,
	//  its length must be computed at the same time.
	stlen := uint32(0)

	// Loader Header
	hdr := &XcoffLdHdr64{
		Lversion: 2,
		Lsymoff:  LDHDRSZ_64,
	}

	ldr := ctxt.loader
	/* Symbol table */
	for _, s := range f.loaderSymbols {
		lds := &XcoffLdSym64{
			Loffset: uint32(stlen + 2),
			Lsmtype: s.smtype,
			Lsmclas: s.smclas,
		}
		sym := s.sym
		switch s.smtype {
		default:
			ldr.Errorf(sym, "unexpected loader symbol type: 0x%x", s.smtype)
		case XTY_ENT | XTY_SD:
			lds.Lvalue = uint64(ldr.SymValue(sym))
			lds.Lscnum = f.getXCOFFscnum(ldr.SymSect(sym))
		case XTY_IMP:
			lds.Lifile = int32(f.dynLibraries[ldr.SymDynimplib(sym)] + 1)
		}
		ldstr := &XcoffLdStr64{
			size: uint16(len(ldr.SymName(sym)) + 1), // + null terminator
			name: ldr.SymName(sym),
		}
		stlen += uint32(2 + ldstr.size) // 2 = sizeof ldstr.size
		symtab = append(symtab, lds)
		strtab = append(strtab, ldstr)

	}

	hdr.Lnsyms = int32(len(symtab))
	hdr.Lrldoff = hdr.Lsymoff + uint64(24*hdr.Lnsyms) // 24 = sizeof one symbol
	off := hdr.Lrldoff                                // current offset is the same of reloc offset

	/* Reloc */
	// Ensure deterministic order
	sort.Slice(f.loaderReloc, func(i, j int) bool {
		r1, r2 := f.loaderReloc[i], f.loaderReloc[j]
		if r1.sym != r2.sym {
			return r1.sym < r2.sym
		}
		if r1.roff != r2.roff {
			return r1.roff < r2.roff
		}
		if r1.rtype != r2.rtype {
			return r1.rtype < r2.rtype
		}
		return r1.symndx < r2.symndx
	})

	ep := ldr.Lookup(*flagEntrySymbol, 0)
	xldr := &XcoffLdRel64{
		Lvaddr:  uint64(ldr.SymValue(ep)),
		Lrtype:  0x3F00,
		Lrsecnm: f.getXCOFFscnum(ldr.SymSect(ep)),
		Lsymndx: 0,
	}
	off += 16
	reloctab = append(reloctab, xldr)

	off += uint64(16 * len(f.loaderReloc))
	for _, r := range f.loaderReloc {
		symp := r.sym
		if symp == 0 {
			panic("unexpected 0 sym value")
		}
		xldr = &XcoffLdRel64{
			Lvaddr:  uint64(ldr.SymValue(symp) + int64(r.roff)),
			Lrtype:  r.rtype,
			Lsymndx: r.symndx,
		}

		if ldr.SymSect(symp) != nil {
			xldr.Lrsecnm = f.getXCOFFscnum(ldr.SymSect(symp))
		}

		reloctab = append(reloctab, xldr)
	}

	off += uint64(16 * len(dynimpreloc))
	reloctab = append(reloctab, dynimpreloc...)

	hdr.Lnreloc = int32(len(reloctab))
	hdr.Limpoff = off

	/* Import */
	// Default import: /usr/lib:/lib
	ldimpf := &XcoffLdImportFile64{
		Limpidpath: "/usr/lib:/lib",
	}
	off += uint64(len(ldimpf.Limpidpath) + len(ldimpf.Limpidbase) + len(ldimpf.Limpidmem) + 3) // + null delimiter
	importtab = append(importtab, ldimpf)

	// The map created by adddynimpsym associates the name to a number
	// This number represents the librairie index (- 1) in this import files section
	// Therefore, they must be sorted before being put inside the section
	libsOrdered := make([]string, len(f.dynLibraries))
	for key, val := range f.dynLibraries {
		if libsOrdered[val] != "" {
			continue
		}
		libsOrdered[val] = key
	}

	for _, lib := range libsOrdered {
		// lib string is defined as base.a/mem.o or path/base.a/mem.o
		n := strings.Split(lib, "/")
		path := ""
		base := n[len(n)-2]
		mem := n[len(n)-1]
		if len(n) > 2 {
			path = lib[:len(lib)-len(base)-len(mem)-2]

		}
		ldimpf = &XcoffLdImportFile64{
			Limpidpath: path,
			Limpidbase: base,
			Limpidmem:  mem,
		}
		off += uint64(len(ldimpf.Limpidpath) + len(ldimpf.Limpidbase) + len(ldimpf.Limpidmem) + 3) // + null delimiter
		importtab = append(importtab, ldimpf)
	}

	hdr.Lnimpid = int32(len(importtab))
	hdr.Listlen = uint32(off - hdr.Limpoff)
	hdr.Lstoff = off
	hdr.Lstlen = stlen

	/* Writing */
	ctxt.Out.SeekSet(int64(globalOff))
	binary.Write(ctxt.Out, ctxt.Arch.ByteOrder, hdr)

	for _, s := range symtab {
		binary.Write(ctxt.Out, ctxt.Arch.ByteOrder, s)

	}
	for _, r := range reloctab {
		binary.Write(ctxt.Out, ctxt.Arch.ByteOrder, r)
	}
	for _, f := range importtab {
		ctxt.Out.WriteString(f.Limpidpath)
		ctxt.Out.Write8(0)
		ctxt.Out.WriteString(f.Limpidbase)
		ctxt.Out.Write8(0)
		ctxt.Out.WriteString(f.Limpidmem)
		ctxt.Out.Write8(0)
	}
	for _, s := range strtab {
		ctxt.Out.Write16(s.size)
		ctxt.Out.WriteString(s.name)
		ctxt.Out.Write8(0) // null terminator
	}

	f.loaderSize = off + uint64(stlen)
}

// XCOFF assembling and writing file

func (f *xcoffFile) writeFileHeader(ctxt *Link) {
	// File header
	f.xfhdr.Fmagic = U64_TOCMAGIC
	f.xfhdr.Fnscns = uint16(len(f.sections))
	f.xfhdr.Ftimedat = 0

	if !*FlagS {
		f.xfhdr.Fsymptr = uint64(f.symtabOffset)
		f.xfhdr.Fnsyms = int32(f.symbolCount)
	}

	if ctxt.BuildMode == BuildModeExe && ctxt.LinkMode == LinkInternal {
		ldr := ctxt.loader
		f.xfhdr.Fopthdr = AOUTHSZ_EXEC64
		f.xfhdr.Fflags = F_EXEC

		// auxiliary header
		f.xahdr.Ovstamp = 1 // based on dump -o
		f.xahdr.Omagic = 0x10b
		copy(f.xahdr.Omodtype[:], "1L")
		entry := ldr.Lookup(*flagEntrySymbol, 0)
		f.xahdr.Oentry = uint64(ldr.SymValue(entry))
		f.xahdr.Osnentry = f.getXCOFFscnum(ldr.SymSect(entry))
		toc := ldr.Lookup("TOC", 0)
		f.xahdr.Otoc = uint64(ldr.SymValue(toc))
		f.xahdr.Osntoc = f.getXCOFFscnum(ldr.SymSect(toc))

		f.xahdr.Oalgntext = int16(logBase2(int(XCOFFSECTALIGN)))
		f.xahdr.Oalgndata = 0x5

		binary.Write(ctxt.Out, binary.BigEndian, &f.xfhdr)
		binary.Write(ctxt.Out, binary.BigEndian, &f.xahdr)
	} else {
		f.xfhdr.Fopthdr = 0
		binary.Write(ctxt.Out, binary.BigEndian, &f.xfhdr)
	}

}

func xcoffwrite(ctxt *Link) {
	ctxt.Out.SeekSet(0)

	xfile.writeFileHeader(ctxt)

	for _, sect := range xfile.sections {
		sect.write(ctxt)
	}
}

// Generate XCOFF assembly file.
func asmbXcoff(ctxt *Link) {
	ctxt.Out.SeekSet(0)
	fileoff := int64(Segdwarf.Fileoff + Segdwarf.Filelen)
	fileoff = int64(Rnd(int64(fileoff), *FlagRound))

	xfile.sectNameToScnum = make(map[string]int16)

	// Add sections
	s := xfile.addSection(".text", Segtext.Vaddr, Segtext.Length, Segtext.Fileoff, STYP_TEXT)
	xfile.xahdr.Otextstart = s.Svaddr
	xfile.xahdr.Osntext = xfile.sectNameToScnum[".text"]
	xfile.xahdr.Otsize = s.Ssize
	xfile.sectText = s

	segdataVaddr := Segdata.Vaddr
	segdataFilelen := Segdata.Filelen
	segdataFileoff := Segdata.Fileoff
	segbssFilelen := Segdata.Length - Segdata.Filelen
	if len(Segrelrodata.Sections) > 0 {
		// Merge relro segment to data segment as
		// relro data are inside data segment on AIX.
		segdataVaddr = Segrelrodata.Vaddr
		segdataFileoff = Segrelrodata.Fileoff
		segdataFilelen = Segdata.Vaddr + Segdata.Filelen - Segrelrodata.Vaddr
	}

	s = xfile.addSection(".data", segdataVaddr, segdataFilelen, segdataFileoff, STYP_DATA)
	xfile.xahdr.Odatastart = s.Svaddr
	xfile.xahdr.Osndata = xfile.sectNameToScnum[".data"]
	xfile.xahdr.Odsize = s.Ssize
	xfile.sectData = s

	s = xfile.addSection(".bss", segdataVaddr+segdataFilelen, segbssFilelen, 0, STYP_BSS)
	xfile.xahdr.Osnbss = xfile.sectNameToScnum[".bss"]
	xfile.xahdr.Obsize = s.Ssize
	xfile.sectBss = s

	if ctxt.LinkMode == LinkExternal {
		var tbss *sym.Section
		for _, s := range Segdata.Sections {
			if s.Name == ".tbss" {
				tbss = s
				break
			}
		}
		s = xfile.addSection(".tbss", tbss.Vaddr, tbss.Length, 0, STYP_TBSS)
	}

	// add dwarf sections
	for _, sect := range Segdwarf.Sections {
		xfile.addDwarfSection(sect)
	}

	// add and write remaining sections
	if ctxt.LinkMode == LinkInternal {
		// Loader section
		if ctxt.BuildMode == BuildModeExe {
			Loaderblk(ctxt, uint64(fileoff))
			s = xfile.addSection(".loader", 0, xfile.loaderSize, uint64(fileoff), STYP_LOADER)
			xfile.xahdr.Osnloader = xfile.sectNameToScnum[".loader"]

			// Update fileoff for symbol table
			fileoff += int64(xfile.loaderSize)
		}
	}

	// Create Symbol table
	xfile.asmaixsym(ctxt)

	if ctxt.LinkMode == LinkExternal {
		xfile.emitRelocations(ctxt, fileoff)
	}

	// Write Symbol table
	xfile.symtabOffset = ctxt.Out.Offset()
	for _, s := range xfile.symtabSym {
		binary.Write(ctxt.Out, ctxt.Arch.ByteOrder, s)
	}
	// write string table
	xfile.stringTable.write(ctxt.Out)

	// write headers
	xcoffwrite(ctxt)
}

// emitRelocations emits relocation entries for go.o in external linking.
func (f *xcoffFile) emitRelocations(ctxt *Link, fileoff int64) {
	ctxt.Out.SeekSet(fileoff)
	for ctxt.Out.Offset()&7 != 0 {
		ctxt.Out.Write8(0)
	}

	ldr := ctxt.loader
	// relocsect relocates symbols from first in section sect, and returns
	// the total number of relocations emitted.
	relocsect := func(sect *sym.Section, syms []loader.Sym, base uint64) uint32 {
		// ctxt.Logf("%s 0x%x\n", sect.Name, sect.Vaddr)
		// If main section has no bits, nothing to relocate.
		if sect.Vaddr >= sect.Seg.Vaddr+sect.Seg.Filelen {
			return 0
		}
		sect.Reloff = uint64(ctxt.Out.Offset())
		for i, s := range syms {
			if !ldr.AttrReachable(s) {
				continue
			}
			if uint64(ldr.SymValue(s)) >= sect.Vaddr {
				syms = syms[i:]
				break
			}
		}
		eaddr := int64(sect.Vaddr + sect.Length)
		for _, s := range syms {
			if !ldr.AttrReachable(s) {
				continue
			}
			if ldr.SymValue(s) >= int64(eaddr) {
				break
			}

			// Compute external relocations on the go, and pass to Xcoffreloc1 to stream out.
			// Relocation must be ordered by address, so create a list of sorted indices.
			relocs := ldr.Relocs(s)
			sorted := make([]int, relocs.Count())
			for i := 0; i < relocs.Count(); i++ {
				sorted[i] = i
			}
			sort.Slice(sorted, func(i, j int) bool {
				return relocs.At(sorted[i]).Off() < relocs.At(sorted[j]).Off()
			})

			for _, ri := range sorted {
				r := relocs.At(ri)
				rr, ok := extreloc(ctxt, ldr, s, r)
				if !ok {
					continue
				}
				if rr.Xsym == 0 {
					ldr.Errorf(s, "missing xsym in relocation")
					continue
				}
				if ldr.SymDynid(rr.Xsym) < 0 {
					ldr.Errorf(s, "reloc %s to non-coff symbol %s (outer=%s) %d %d", r.Type(), ldr.SymName(r.Sym()), ldr.SymName(rr.Xsym), ldr.SymType(r.Sym()), ldr.SymDynid(rr.Xsym))
				}
				if !thearch.Xcoffreloc1(ctxt.Arch, ctxt.Out, ldr, s, rr, int64(uint64(ldr.SymValue(s)+int64(r.Off()))-base)) {
					ldr.Errorf(s, "unsupported obj reloc %d(%s)/%d to %s", r.Type(), r.Type(), r.Siz(), ldr.SymName(r.Sym()))
				}
			}
		}
		sect.Rellen = uint64(ctxt.Out.Offset()) - sect.Reloff
		return uint32(sect.Rellen) / RELSZ_64
	}
	sects := []struct {
		xcoffSect *XcoffScnHdr64
		segs      []*sym.Segment
	}{
		{f.sectText, []*sym.Segment{&Segtext}},
		{f.sectData, []*sym.Segment{&Segrelrodata, &Segdata}},
	}
	for _, s := range sects {
		s.xcoffSect.Srelptr = uint64(ctxt.Out.Offset())
		n := uint32(0)
		for _, seg := range s.segs {
			for _, sect := range seg.Sections {
				if sect.Name == ".text" {
					n += relocsect(sect, ctxt.Textp, 0)
				} else {
					n += relocsect(sect, ctxt.datap, 0)
				}
			}
		}
		s.xcoffSect.Snreloc += n
	}

dwarfLoop:
	for i := 0; i < len(Segdwarf.Sections); i++ {
		sect := Segdwarf.Sections[i]
		si := dwarfp[i]
		if si.secSym() != loader.Sym(sect.Sym) ||
			ldr.SymSect(si.secSym()) != sect {
			panic("inconsistency between dwarfp and Segdwarf")
		}
		for _, xcoffSect := range f.sections {
			_, subtyp := xcoffGetDwarfSubtype(sect.Name)
			if xcoffSect.Sflags&0xF0000 == subtyp {
				xcoffSect.Srelptr = uint64(ctxt.Out.Offset())
				xcoffSect.Snreloc = relocsect(sect, si.syms, sect.Vaddr)
				continue dwarfLoop
			}
		}
		Errorf("emitRelocations: could not find %q section", sect.Name)
	}
}

// xcoffCreateExportFile creates a file with exported symbols for
// -Wl,-bE option.
// ld won't export symbols unless they are listed in an export file.
func xcoffCreateExportFile(ctxt *Link) (fname string) {
	fname = filepath.Join(*flagTmpdir, "export_file.exp")
	var buf bytes.Buffer

	ldr := ctxt.loader
	for s := range ldr.ForAllCgoExportStatic() {
		extname := ldr.SymExtname(s)
		if !strings.HasPrefix(extname, "._cgoexp_") {
			continue
		}
		if ldr.IsFileLocal(s) {
			continue // Only export non-static symbols
		}

		// Retrieve the name of the initial symbol
		// exported by cgo.
		// The corresponding Go symbol is:
		// _cgoexp_hashcode_symname.
		name := strings.SplitN(extname, "_", 4)[3]

		buf.Write([]byte(name + "\n"))
	}

	err := os.WriteFile(fname, buf.Bytes(), 0666)
	if err != nil {
		Errorf("WriteFile %s failed: %v", fname, err)
	}

	return fname
}
