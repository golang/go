// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/obj"
	"cmd/internal/sys"
	"encoding/binary"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
)

type IMAGE_FILE_HEADER struct {
	Machine              uint16
	NumberOfSections     uint16
	TimeDateStamp        uint32
	PointerToSymbolTable uint32
	NumberOfSymbols      uint32
	SizeOfOptionalHeader uint16
	Characteristics      uint16
}

type IMAGE_DATA_DIRECTORY struct {
	VirtualAddress uint32
	Size           uint32
}

type IMAGE_OPTIONAL_HEADER struct {
	Magic                       uint16
	MajorLinkerVersion          uint8
	MinorLinkerVersion          uint8
	SizeOfCode                  uint32
	SizeOfInitializedData       uint32
	SizeOfUninitializedData     uint32
	AddressOfEntryPoint         uint32
	BaseOfCode                  uint32
	BaseOfData                  uint32
	ImageBase                   uint32
	SectionAlignment            uint32
	FileAlignment               uint32
	MajorOperatingSystemVersion uint16
	MinorOperatingSystemVersion uint16
	MajorImageVersion           uint16
	MinorImageVersion           uint16
	MajorSubsystemVersion       uint16
	MinorSubsystemVersion       uint16
	Win32VersionValue           uint32
	SizeOfImage                 uint32
	SizeOfHeaders               uint32
	CheckSum                    uint32
	Subsystem                   uint16
	DllCharacteristics          uint16
	SizeOfStackReserve          uint32
	SizeOfStackCommit           uint32
	SizeOfHeapReserve           uint32
	SizeOfHeapCommit            uint32
	LoaderFlags                 uint32
	NumberOfRvaAndSizes         uint32
	DataDirectory               [16]IMAGE_DATA_DIRECTORY
}

type IMAGE_SECTION_HEADER struct {
	Name                 [8]uint8
	VirtualSize          uint32
	VirtualAddress       uint32
	SizeOfRawData        uint32
	PointerToRawData     uint32
	PointerToRelocations uint32
	PointerToLineNumbers uint32
	NumberOfRelocations  uint16
	NumberOfLineNumbers  uint16
	Characteristics      uint32
}

type IMAGE_IMPORT_DESCRIPTOR struct {
	OriginalFirstThunk uint32
	TimeDateStamp      uint32
	ForwarderChain     uint32
	Name               uint32
	FirstThunk         uint32
}

type IMAGE_EXPORT_DIRECTORY struct {
	Characteristics       uint32
	TimeDateStamp         uint32
	MajorVersion          uint16
	MinorVersion          uint16
	Name                  uint32
	Base                  uint32
	NumberOfFunctions     uint32
	NumberOfNames         uint32
	AddressOfFunctions    uint32
	AddressOfNames        uint32
	AddressOfNameOrdinals uint32
}

const (
	PEBASE = 0x00400000

	// SectionAlignment must be greater than or equal to FileAlignment.
	// The default is the page size for the architecture.
	PESECTALIGN = 0x1000

	// FileAlignment should be a power of 2 between 512 and 64 K, inclusive.
	// The default is 512. If the SectionAlignment is less than
	// the architecture's page size, then FileAlignment must match SectionAlignment.
	PEFILEALIGN = 2 << 8
)

const (
	IMAGE_FILE_MACHINE_I386              = 0x14c
	IMAGE_FILE_MACHINE_AMD64             = 0x8664
	IMAGE_FILE_RELOCS_STRIPPED           = 0x0001
	IMAGE_FILE_EXECUTABLE_IMAGE          = 0x0002
	IMAGE_FILE_LINE_NUMS_STRIPPED        = 0x0004
	IMAGE_FILE_LARGE_ADDRESS_AWARE       = 0x0020
	IMAGE_FILE_32BIT_MACHINE             = 0x0100
	IMAGE_FILE_DEBUG_STRIPPED            = 0x0200
	IMAGE_SCN_CNT_CODE                   = 0x00000020
	IMAGE_SCN_CNT_INITIALIZED_DATA       = 0x00000040
	IMAGE_SCN_CNT_UNINITIALIZED_DATA     = 0x00000080
	IMAGE_SCN_MEM_EXECUTE                = 0x20000000
	IMAGE_SCN_MEM_READ                   = 0x40000000
	IMAGE_SCN_MEM_WRITE                  = 0x80000000
	IMAGE_SCN_MEM_DISCARDABLE            = 0x2000000
	IMAGE_SCN_LNK_NRELOC_OVFL            = 0x1000000
	IMAGE_SCN_ALIGN_32BYTES              = 0x600000
	IMAGE_DIRECTORY_ENTRY_EXPORT         = 0
	IMAGE_DIRECTORY_ENTRY_IMPORT         = 1
	IMAGE_DIRECTORY_ENTRY_RESOURCE       = 2
	IMAGE_DIRECTORY_ENTRY_EXCEPTION      = 3
	IMAGE_DIRECTORY_ENTRY_SECURITY       = 4
	IMAGE_DIRECTORY_ENTRY_BASERELOC      = 5
	IMAGE_DIRECTORY_ENTRY_DEBUG          = 6
	IMAGE_DIRECTORY_ENTRY_COPYRIGHT      = 7
	IMAGE_DIRECTORY_ENTRY_ARCHITECTURE   = 7
	IMAGE_DIRECTORY_ENTRY_GLOBALPTR      = 8
	IMAGE_DIRECTORY_ENTRY_TLS            = 9
	IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG    = 10
	IMAGE_DIRECTORY_ENTRY_BOUND_IMPORT   = 11
	IMAGE_DIRECTORY_ENTRY_IAT            = 12
	IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT   = 13
	IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR = 14
	IMAGE_SUBSYSTEM_WINDOWS_GUI          = 2
	IMAGE_SUBSYSTEM_WINDOWS_CUI          = 3
)

// X64
type PE64_IMAGE_OPTIONAL_HEADER struct {
	Magic                       uint16
	MajorLinkerVersion          uint8
	MinorLinkerVersion          uint8
	SizeOfCode                  uint32
	SizeOfInitializedData       uint32
	SizeOfUninitializedData     uint32
	AddressOfEntryPoint         uint32
	BaseOfCode                  uint32
	ImageBase                   uint64
	SectionAlignment            uint32
	FileAlignment               uint32
	MajorOperatingSystemVersion uint16
	MinorOperatingSystemVersion uint16
	MajorImageVersion           uint16
	MinorImageVersion           uint16
	MajorSubsystemVersion       uint16
	MinorSubsystemVersion       uint16
	Win32VersionValue           uint32
	SizeOfImage                 uint32
	SizeOfHeaders               uint32
	CheckSum                    uint32
	Subsystem                   uint16
	DllCharacteristics          uint16
	SizeOfStackReserve          uint64
	SizeOfStackCommit           uint64
	SizeOfHeapReserve           uint64
	SizeOfHeapCommit            uint64
	LoaderFlags                 uint32
	NumberOfRvaAndSizes         uint32
	DataDirectory               [16]IMAGE_DATA_DIRECTORY
}

// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// PE (Portable Executable) file writing
// http://www.microsoft.com/whdc/system/platform/firmware/PECOFF.mspx

// DOS stub that prints out
// "This program cannot be run in DOS mode."
var dosstub = []uint8{
	0x4d,
	0x5a,
	0x90,
	0x00,
	0x03,
	0x00,
	0x04,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0xff,
	0xff,
	0x00,
	0x00,
	0x8b,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x40,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x80,
	0x00,
	0x00,
	0x00,
	0x0e,
	0x1f,
	0xba,
	0x0e,
	0x00,
	0xb4,
	0x09,
	0xcd,
	0x21,
	0xb8,
	0x01,
	0x4c,
	0xcd,
	0x21,
	0x54,
	0x68,
	0x69,
	0x73,
	0x20,
	0x70,
	0x72,
	0x6f,
	0x67,
	0x72,
	0x61,
	0x6d,
	0x20,
	0x63,
	0x61,
	0x6e,
	0x6e,
	0x6f,
	0x74,
	0x20,
	0x62,
	0x65,
	0x20,
	0x72,
	0x75,
	0x6e,
	0x20,
	0x69,
	0x6e,
	0x20,
	0x44,
	0x4f,
	0x53,
	0x20,
	0x6d,
	0x6f,
	0x64,
	0x65,
	0x2e,
	0x0d,
	0x0d,
	0x0a,
	0x24,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
	0x00,
}

var rsrcsym *LSym

var strtbl []byte

var PESECTHEADR int32

var PEFILEHEADR int32

var pe64 int

var pensect int

var nextsectoff int

var nextfileoff int

var textsect int

var datasect int

var bsssect int

var fh IMAGE_FILE_HEADER

var oh IMAGE_OPTIONAL_HEADER

var oh64 PE64_IMAGE_OPTIONAL_HEADER

var sh [16]IMAGE_SECTION_HEADER

var dd []IMAGE_DATA_DIRECTORY

type Imp struct {
	s       *LSym
	off     uint64
	next    *Imp
	argsize int
}

type Dll struct {
	name     string
	nameoff  uint64
	thunkoff uint64
	ms       *Imp
	next     *Dll
}

var dr *Dll

var dexport [1024]*LSym

var nexport int

func addpesection(name string, sectsize int, filesize int) *IMAGE_SECTION_HEADER {
	if pensect == 16 {
		Diag("too many sections")
		errorexit()
	}

	h := &sh[pensect]
	pensect++
	copy(h.Name[:], name)
	h.VirtualSize = uint32(sectsize)
	h.VirtualAddress = uint32(nextsectoff)
	nextsectoff = int(Rnd(int64(nextsectoff)+int64(sectsize), PESECTALIGN))
	h.PointerToRawData = uint32(nextfileoff)
	if filesize > 0 {
		h.SizeOfRawData = uint32(Rnd(int64(filesize), PEFILEALIGN))
		nextfileoff += int(h.SizeOfRawData)
	}

	return h
}

func chksectoff(h *IMAGE_SECTION_HEADER, off int64) {
	if off != int64(h.PointerToRawData) {
		Diag("%s.PointerToRawData = %#x, want %#x", cstring(h.Name[:]), uint64(int64(h.PointerToRawData)), uint64(off))
		errorexit()
	}
}

func chksectseg(h *IMAGE_SECTION_HEADER, s *Segment) {
	if s.Vaddr-PEBASE != uint64(h.VirtualAddress) {
		Diag("%s.VirtualAddress = %#x, want %#x", cstring(h.Name[:]), uint64(int64(h.VirtualAddress)), uint64(int64(s.Vaddr-PEBASE)))
		errorexit()
	}

	if s.Fileoff != uint64(h.PointerToRawData) {
		Diag("%s.PointerToRawData = %#x, want %#x", cstring(h.Name[:]), uint64(int64(h.PointerToRawData)), uint64(int64(s.Fileoff)))
		errorexit()
	}
}

func Peinit() {
	var l int

	switch SysArch.Family {
	// 64-bit architectures
	case sys.AMD64:
		pe64 = 1

		l = binary.Size(&oh64)
		dd = oh64.DataDirectory[:]

	// 32-bit architectures
	default:
		l = binary.Size(&oh)

		dd = oh.DataDirectory[:]
	}

	PEFILEHEADR = int32(Rnd(int64(len(dosstub)+binary.Size(&fh)+l+binary.Size(&sh)), PEFILEALIGN))
	PESECTHEADR = int32(Rnd(int64(PEFILEHEADR), PESECTALIGN))
	nextsectoff = int(PESECTHEADR)
	nextfileoff = int(PEFILEHEADR)

	// some mingw libs depend on this symbol, for example, FindPESectionByName
	xdefine("__image_base__", obj.SDATA, PEBASE)

	xdefine("_image_base__", obj.SDATA, PEBASE)
}

func pewrite() {
	Cseek(0)
	if Linkmode != LinkExternal {
		Cwrite(dosstub)
		strnput("PE", 4)
	}

	binary.Write(&coutbuf, binary.LittleEndian, &fh)

	if pe64 != 0 {
		binary.Write(&coutbuf, binary.LittleEndian, &oh64)
	} else {
		binary.Write(&coutbuf, binary.LittleEndian, &oh)
	}
	binary.Write(&coutbuf, binary.LittleEndian, sh[:pensect])
}

func strput(s string) {
	coutbuf.WriteString(s)
	Cput(0)
	// string must be padded to even size
	if (len(s)+1)%2 != 0 {
		Cput(0)
	}
}

func initdynimport() *Dll {
	var d *Dll

	dr = nil
	var m *Imp
	for _, s := range Ctxt.Allsym {
		if !s.Attr.Reachable() || s.Type != obj.SDYNIMPORT {
			continue
		}
		for d = dr; d != nil; d = d.next {
			if d.name == s.Dynimplib {
				m = new(Imp)
				break
			}
		}

		if d == nil {
			d = new(Dll)
			d.name = s.Dynimplib
			d.next = dr
			dr = d
			m = new(Imp)
		}

		// Because external link requires properly stdcall decorated name,
		// all external symbols in runtime use %n to denote that the number
		// of uinptrs this function consumes. Store the argsize and discard
		// the %n suffix if any.
		m.argsize = -1
		if i := strings.IndexByte(s.Extname, '%'); i >= 0 {
			var err error
			m.argsize, err = strconv.Atoi(s.Extname[i+1:])
			if err != nil {
				Diag("failed to parse stdcall decoration: %v", err)
			}
			m.argsize *= SysArch.PtrSize
			s.Extname = s.Extname[:i]
		}

		m.s = s
		m.next = d.ms
		d.ms = m
	}

	if Linkmode == LinkExternal {
		// Add real symbol name
		for d := dr; d != nil; d = d.next {
			for m = d.ms; m != nil; m = m.next {
				m.s.Type = obj.SDATA
				Symgrow(Ctxt, m.s, int64(SysArch.PtrSize))
				dynName := m.s.Extname
				// only windows/386 requires stdcall decoration
				if SysArch.Family == sys.I386 && m.argsize >= 0 {
					dynName += fmt.Sprintf("@%d", m.argsize)
				}
				dynSym := Linklookup(Ctxt, dynName, 0)
				dynSym.Attr |= AttrReachable
				dynSym.Type = obj.SHOSTOBJ
				r := Addrel(m.s)
				r.Sym = dynSym
				r.Off = 0
				r.Siz = uint8(SysArch.PtrSize)
				r.Type = obj.R_ADDR
			}
		}
	} else {
		dynamic := Linklookup(Ctxt, ".windynamic", 0)
		dynamic.Attr |= AttrReachable
		dynamic.Type = obj.SWINDOWS
		for d := dr; d != nil; d = d.next {
			for m = d.ms; m != nil; m = m.next {
				m.s.Type = obj.SWINDOWS | obj.SSUB
				m.s.Sub = dynamic.Sub
				dynamic.Sub = m.s
				m.s.Value = dynamic.Size
				dynamic.Size += int64(SysArch.PtrSize)
			}

			dynamic.Size += int64(SysArch.PtrSize)
		}
	}

	return dr
}

// peimporteddlls returns the gcc command line argument to link all imported
// DLLs.
func peimporteddlls() []string {
	var dlls []string

	for d := dr; d != nil; d = d.next {
		dlls = append(dlls, "-l"+strings.TrimSuffix(d.name, ".dll"))
	}

	return dlls
}

func addimports(datsect *IMAGE_SECTION_HEADER) {
	startoff := Cpos()
	dynamic := Linklookup(Ctxt, ".windynamic", 0)

	// skip import descriptor table (will write it later)
	n := uint64(0)

	for d := dr; d != nil; d = d.next {
		n++
	}
	Cseek(startoff + int64(binary.Size(&IMAGE_IMPORT_DESCRIPTOR{}))*int64(n+1))

	// write dll names
	for d := dr; d != nil; d = d.next {
		d.nameoff = uint64(Cpos()) - uint64(startoff)
		strput(d.name)
	}

	// write function names
	var m *Imp
	for d := dr; d != nil; d = d.next {
		for m = d.ms; m != nil; m = m.next {
			m.off = uint64(nextsectoff) + uint64(Cpos()) - uint64(startoff)
			Wputl(0) // hint
			strput(m.s.Extname)
		}
	}

	// write OriginalFirstThunks
	oftbase := uint64(Cpos()) - uint64(startoff)

	n = uint64(Cpos())
	for d := dr; d != nil; d = d.next {
		d.thunkoff = uint64(Cpos()) - n
		for m = d.ms; m != nil; m = m.next {
			if pe64 != 0 {
				Vputl(m.off)
			} else {
				Lputl(uint32(m.off))
			}
		}

		if pe64 != 0 {
			Vputl(0)
		} else {
			Lputl(0)
		}
	}

	// add pe section and pad it at the end
	n = uint64(Cpos()) - uint64(startoff)

	isect := addpesection(".idata", int(n), int(n))
	isect.Characteristics = IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE
	chksectoff(isect, startoff)
	strnput("", int(uint64(isect.SizeOfRawData)-n))
	endoff := Cpos()

	// write FirstThunks (allocated in .data section)
	ftbase := uint64(dynamic.Value) - uint64(datsect.VirtualAddress) - PEBASE

	Cseek(int64(uint64(datsect.PointerToRawData) + ftbase))
	for d := dr; d != nil; d = d.next {
		for m = d.ms; m != nil; m = m.next {
			if pe64 != 0 {
				Vputl(m.off)
			} else {
				Lputl(uint32(m.off))
			}
		}

		if pe64 != 0 {
			Vputl(0)
		} else {
			Lputl(0)
		}
	}

	// finally write import descriptor table
	Cseek(startoff)

	for d := dr; d != nil; d = d.next {
		Lputl(uint32(uint64(isect.VirtualAddress) + oftbase + d.thunkoff))
		Lputl(0)
		Lputl(0)
		Lputl(uint32(uint64(isect.VirtualAddress) + d.nameoff))
		Lputl(uint32(uint64(datsect.VirtualAddress) + ftbase + d.thunkoff))
	}

	Lputl(0) //end
	Lputl(0)
	Lputl(0)
	Lputl(0)
	Lputl(0)

	// update data directory
	dd[IMAGE_DIRECTORY_ENTRY_IMPORT].VirtualAddress = isect.VirtualAddress

	dd[IMAGE_DIRECTORY_ENTRY_IMPORT].Size = isect.VirtualSize
	dd[IMAGE_DIRECTORY_ENTRY_IAT].VirtualAddress = uint32(dynamic.Value - PEBASE)
	dd[IMAGE_DIRECTORY_ENTRY_IAT].Size = uint32(dynamic.Size)

	Cseek(endoff)
}

type byExtname []*LSym

func (s byExtname) Len() int           { return len(s) }
func (s byExtname) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s byExtname) Less(i, j int) bool { return s[i].Extname < s[j].Extname }

func initdynexport() {
	nexport = 0
	for _, s := range Ctxt.Allsym {
		if !s.Attr.Reachable() || !s.Attr.CgoExportDynamic() {
			continue
		}
		if nexport+1 > len(dexport) {
			Diag("pe dynexport table is full")
			errorexit()
		}

		dexport[nexport] = s
		nexport++
	}

	sort.Sort(byExtname(dexport[:nexport]))
}

func addexports() {
	var e IMAGE_EXPORT_DIRECTORY

	size := binary.Size(&e) + 10*nexport + len(outfile) + 1
	for i := 0; i < nexport; i++ {
		size += len(dexport[i].Extname) + 1
	}

	if nexport == 0 {
		return
	}

	sect := addpesection(".edata", size, size)
	sect.Characteristics = IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ
	chksectoff(sect, Cpos())
	va := int(sect.VirtualAddress)
	dd[IMAGE_DIRECTORY_ENTRY_EXPORT].VirtualAddress = uint32(va)
	dd[IMAGE_DIRECTORY_ENTRY_EXPORT].Size = sect.VirtualSize

	va_name := va + binary.Size(&e) + nexport*4
	va_addr := va + binary.Size(&e)
	va_na := va + binary.Size(&e) + nexport*8

	e.Characteristics = 0
	e.MajorVersion = 0
	e.MinorVersion = 0
	e.NumberOfFunctions = uint32(nexport)
	e.NumberOfNames = uint32(nexport)
	e.Name = uint32(va+binary.Size(&e)) + uint32(nexport)*10 // Program names.
	e.Base = 1
	e.AddressOfFunctions = uint32(va_addr)
	e.AddressOfNames = uint32(va_name)
	e.AddressOfNameOrdinals = uint32(va_na)

	// put IMAGE_EXPORT_DIRECTORY
	binary.Write(&coutbuf, binary.LittleEndian, &e)

	// put EXPORT Address Table
	for i := 0; i < nexport; i++ {
		Lputl(uint32(dexport[i].Value - PEBASE))
	}

	// put EXPORT Name Pointer Table
	v := int(e.Name + uint32(len(outfile)) + 1)

	for i := 0; i < nexport; i++ {
		Lputl(uint32(v))
		v += len(dexport[i].Extname) + 1
	}

	// put EXPORT Ordinal Table
	for i := 0; i < nexport; i++ {
		Wputl(uint16(i))
	}

	// put Names
	strnput(outfile, len(outfile)+1)

	for i := 0; i < nexport; i++ {
		strnput(dexport[i].Extname, len(dexport[i].Extname)+1)
	}
	strnput("", int(sect.SizeOfRawData-uint32(size)))
}

// perelocsect relocates symbols from first in section sect, and returns
// the total number of relocations emitted.
func perelocsect(sect *Section, syms []*LSym) int {
	// If main section has no bits, nothing to relocate.
	if sect.Vaddr >= sect.Seg.Vaddr+sect.Seg.Filelen {
		return 0
	}

	relocs := 0

	sect.Reloff = uint64(Cpos())
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
	for _, sym := range syms {
		if !sym.Attr.Reachable() {
			continue
		}
		if sym.Value >= int64(eaddr) {
			break
		}
		Ctxt.Cursym = sym

		for ri := 0; ri < len(sym.R); ri++ {
			r := &sym.R[ri]
			if r.Done != 0 {
				continue
			}
			if r.Xsym == nil {
				Diag("missing xsym in relocation")
				continue
			}

			if r.Xsym.Dynid < 0 {
				Diag("reloc %d to non-coff symbol %s (outer=%s) %d", r.Type, r.Sym.Name, r.Xsym.Name, r.Sym.Type)
			}
			if !Thearch.PEreloc1(r, int64(uint64(sym.Value+int64(r.Off))-PEBASE)) {
				Diag("unsupported obj reloc %d/%d to %s", r.Type, r.Siz, r.Sym.Name)
			}

			relocs++
		}
	}

	sect.Rellen = uint64(Cpos()) - sect.Reloff

	return relocs
}

// peemitreloc emits relocation entries for go.o in external linking.
func peemitreloc(text, data, ctors *IMAGE_SECTION_HEADER) {
	for Cpos()&7 != 0 {
		Cput(0)
	}

	text.PointerToRelocations = uint32(Cpos())
	// first entry: extended relocs
	Lputl(0) // placeholder for number of relocation + 1
	Lputl(0)
	Wputl(0)

	n := perelocsect(Segtext.Sect, Ctxt.Textp) + 1
	for sect := Segtext.Sect.Next; sect != nil; sect = sect.Next {
		n += perelocsect(sect, datap)
	}

	cpos := Cpos()
	Cseek(int64(text.PointerToRelocations))
	Lputl(uint32(n))
	Cseek(cpos)
	if n > 0x10000 {
		n = 0x10000
		text.Characteristics |= IMAGE_SCN_LNK_NRELOC_OVFL
	} else {
		text.PointerToRelocations += 10 // skip the extend reloc entry
	}
	text.NumberOfRelocations = uint16(n - 1)

	data.PointerToRelocations = uint32(cpos)
	// first entry: extended relocs
	Lputl(0) // placeholder for number of relocation + 1
	Lputl(0)
	Wputl(0)

	n = 1
	for sect := Segdata.Sect; sect != nil; sect = sect.Next {
		n += perelocsect(sect, datap)
	}

	cpos = Cpos()
	Cseek(int64(data.PointerToRelocations))
	Lputl(uint32(n))
	Cseek(cpos)
	if n > 0x10000 {
		n = 0x10000
		data.Characteristics |= IMAGE_SCN_LNK_NRELOC_OVFL
	} else {
		data.PointerToRelocations += 10 // skip the extend reloc entry
	}
	data.NumberOfRelocations = uint16(n - 1)

	dottext := Linklookup(Ctxt, ".text", 0)
	ctors.NumberOfRelocations = 1
	ctors.PointerToRelocations = uint32(Cpos())
	sectoff := ctors.VirtualAddress
	Lputl(sectoff)
	Lputl(uint32(dottext.Dynid))
	switch obj.Getgoarch() {
	default:
		fmt.Fprintf(os.Stderr, "link: unknown architecture for PE: %q\n", obj.Getgoarch())
		os.Exit(2)
	case "386":
		Wputl(IMAGE_REL_I386_DIR32)
	case "amd64":
		Wputl(IMAGE_REL_AMD64_ADDR64)
	}
}

func dope() {
	/* relocation table */
	rel := Linklookup(Ctxt, ".rel", 0)

	rel.Attr |= AttrReachable
	rel.Type = obj.SELFROSECT

	initdynimport()
	initdynexport()
}

func strtbladd(name string) int {
	off := len(strtbl) + 4 // offset includes 4-byte length at beginning of table
	strtbl = append(strtbl, name...)
	strtbl = append(strtbl, 0)
	return off
}

/*
 * For more than 8 characters section names, name contains a slash (/) that is
 * followed by an ASCII representation of a decimal number that is an offset into
 * the string table.
 * reference: pecoff_v8.docx Page 24.
 * <http://www.microsoft.com/whdc/system/platform/firmware/PECOFFdwn.mspx>
 */
func newPEDWARFSection(name string, size int64) *IMAGE_SECTION_HEADER {
	if size == 0 {
		return nil
	}

	off := strtbladd(name)
	s := fmt.Sprintf("/%d", off)
	h := addpesection(s, int(size), int(size))
	h.Characteristics = IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_DISCARDABLE

	return h
}

// writePESymTableRecords writes all COFF symbol table records.
// It returns number of records written.
func writePESymTableRecords() int {
	var symcnt int

	put := func(s *LSym, name string, type_ int, addr int64, size int64, ver int, gotype *LSym) {
		if s == nil {
			return
		}
		if s.Sect == nil && type_ != 'U' {
			return
		}
		switch type_ {
		default:
			return
		case 'D', 'B', 'T', 'U':
		}

		// only windows/386 requires underscore prefix on external symbols
		if SysArch.Family == sys.I386 &&
			Linkmode == LinkExternal &&
			(s.Type != obj.SDYNIMPORT || s.Attr.CgoExport()) &&
			s.Name == s.Extname &&
			s.Name != "_main" {
			s.Name = "_" + s.Name
		}

		var typ uint16
		var sect int
		var value int64
		// Note: although address of runtime.edata (type SDATA) is at the start of .bss section
		// it still belongs to the .data section, not the .bss section.
		if uint64(s.Value) >= Segdata.Vaddr+Segdata.Filelen && s.Type != obj.SDATA && Linkmode == LinkExternal {
			value = int64(uint64(s.Value) - Segdata.Vaddr - Segdata.Filelen)
			sect = bsssect
		} else if uint64(s.Value) >= Segdata.Vaddr {
			value = int64(uint64(s.Value) - Segdata.Vaddr)
			sect = datasect
		} else if uint64(s.Value) >= Segtext.Vaddr {
			value = int64(uint64(s.Value) - Segtext.Vaddr)
			sect = textsect
		} else if type_ == 'U' {
			typ = IMAGE_SYM_DTYPE_FUNCTION
		} else {
			Diag("addpesym %#x", addr)
		}

		// write COFF symbol table record
		if len(s.Name) > 8 {
			Lputl(0)
			Lputl(uint32(strtbladd(s.Name)))
		} else {
			strnput(s.Name, 8)
		}
		Lputl(uint32(value))
		Wputl(uint16(sect))
		if typ != 0 {
			Wputl(typ)
		} else if Linkmode == LinkExternal {
			Wputl(0)
		} else {
			Wputl(0x0308) // "array of structs"
		}
		Cput(2) // storage class: external
		Cput(0) // no aux entries

		s.Dynid = int32(symcnt)

		symcnt++
	}

	if Linkmode == LinkExternal {
		for d := dr; d != nil; d = d.next {
			for m := d.ms; m != nil; m = m.next {
				s := m.s.R[0].Xsym
				put(s, s.Name, 'U', 0, int64(SysArch.PtrSize), 0, nil)
			}
		}

		s := Linklookup(Ctxt, ".text", 0)
		if s.Type == obj.STEXT {
			put(s, s.Name, 'T', s.Value, s.Size, int(s.Version), nil)
		}
	}

	genasmsym(put)

	return symcnt
}

func addpesymtable() {
	symtabStartPos := Cpos()

	// write COFF symbol table
	var symcnt int
	if Debug['s'] == 0 || Linkmode == LinkExternal {
		symcnt = writePESymTableRecords()
	}

	// update COFF file header and section table
	size := len(strtbl) + 4 + 18*symcnt
	var h *IMAGE_SECTION_HEADER
	if Linkmode != LinkExternal {
		// We do not really need .symtab for go.o, and if we have one, ld
		// will also include it in the exe, and that will confuse windows.
		h = addpesection(".symtab", size, size)
		h.Characteristics = IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_DISCARDABLE
		chksectoff(h, symtabStartPos)
	}
	fh.PointerToSymbolTable = uint32(symtabStartPos)
	fh.NumberOfSymbols = uint32(symcnt)

	// write COFF string table
	Lputl(uint32(len(strtbl)) + 4)
	for i := 0; i < len(strtbl); i++ {
		Cput(strtbl[i])
	}
	if Linkmode != LinkExternal {
		strnput("", int(h.SizeOfRawData-uint32(size)))
	}
}

func setpersrc(sym *LSym) {
	if rsrcsym != nil {
		Diag("too many .rsrc sections")
	}

	rsrcsym = sym
}

func addpersrc() {
	if rsrcsym == nil {
		return
	}

	h := addpesection(".rsrc", int(rsrcsym.Size), int(rsrcsym.Size))
	h.Characteristics = IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE | IMAGE_SCN_CNT_INITIALIZED_DATA
	chksectoff(h, Cpos())

	// relocation
	var p []byte
	var r *Reloc
	var val uint32
	for ri := 0; ri < len(rsrcsym.R); ri++ {
		r = &rsrcsym.R[ri]
		p = rsrcsym.P[r.Off:]
		val = uint32(int64(h.VirtualAddress) + r.Add)

		// 32-bit little-endian
		p[0] = byte(val)

		p[1] = byte(val >> 8)
		p[2] = byte(val >> 16)
		p[3] = byte(val >> 24)
	}

	Cwrite(rsrcsym.P)
	strnput("", int(int64(h.SizeOfRawData)-rsrcsym.Size))

	// update data directory
	dd[IMAGE_DIRECTORY_ENTRY_RESOURCE].VirtualAddress = h.VirtualAddress

	dd[IMAGE_DIRECTORY_ENTRY_RESOURCE].Size = h.VirtualSize
}

func addinitarray() (c *IMAGE_SECTION_HEADER) {
	// The size below was determined by the specification for array relocations,
	// and by observing what GCC writes here. If the initarray section grows to
	// contain more than one constructor entry, the size will need to be 8 * constructor_count.
	// However, the entire Go runtime is initialized from just one function, so it is unlikely
	// that this will need to grow in the future.
	var size int
	switch obj.Getgoarch() {
	default:
		fmt.Fprintf(os.Stderr, "link: unknown architecture for PE: %q\n", obj.Getgoarch())
		os.Exit(2)
	case "386":
		size = 4
	case "amd64":
		size = 8
	}

	c = addpesection(".ctors", size, size)
	c.Characteristics = IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ
	c.SizeOfRawData = uint32(size)

	Cseek(int64(c.PointerToRawData))
	chksectoff(c, Cpos())
	init_entry := Linklookup(Ctxt, INITENTRY, 0)
	addr := uint64(init_entry.Value) - init_entry.Sect.Vaddr

	switch obj.Getgoarch() {
	case "386":
		Lputl(uint32(addr))
	case "amd64":
		Vputl(addr)
	}

	return c
}

func Asmbpe() {
	switch SysArch.Family {
	default:
		Exitf("unknown PE architecture: %v", SysArch.Family)
	case sys.AMD64:
		fh.Machine = IMAGE_FILE_MACHINE_AMD64
	case sys.I386:
		fh.Machine = IMAGE_FILE_MACHINE_I386
	}

	t := addpesection(".text", int(Segtext.Length), int(Segtext.Length))
	t.Characteristics = IMAGE_SCN_CNT_CODE | IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ
	if Linkmode == LinkExternal {
		// some data symbols (e.g. masks) end up in the .text section, and they normally
		// expect larger alignment requirement than the default text section alignment.
		t.Characteristics |= IMAGE_SCN_ALIGN_32BYTES
	}
	chksectseg(t, &Segtext)
	textsect = pensect

	var d *IMAGE_SECTION_HEADER
	var c *IMAGE_SECTION_HEADER
	if Linkmode != LinkExternal {
		d = addpesection(".data", int(Segdata.Length), int(Segdata.Filelen))
		d.Characteristics = IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE
		chksectseg(d, &Segdata)
		datasect = pensect
	} else {
		d = addpesection(".data", int(Segdata.Filelen), int(Segdata.Filelen))
		d.Characteristics = IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE | IMAGE_SCN_ALIGN_32BYTES
		chksectseg(d, &Segdata)
		datasect = pensect

		b := addpesection(".bss", int(Segdata.Length-Segdata.Filelen), 0)
		b.Characteristics = IMAGE_SCN_CNT_UNINITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE | IMAGE_SCN_ALIGN_32BYTES
		b.PointerToRawData = 0
		bsssect = pensect

		c = addinitarray()
	}

	if Debug['s'] == 0 {
		dwarfaddpeheaders()
	}

	Cseek(int64(nextfileoff))
	if Linkmode != LinkExternal {
		addimports(d)
		addexports()
	}
	addpesymtable()
	addpersrc()
	if Linkmode == LinkExternal {
		peemitreloc(t, d, c)
	}

	fh.NumberOfSections = uint16(pensect)

	// Being able to produce identical output for identical input is
	// much more beneficial than having build timestamp in the header.
	fh.TimeDateStamp = 0

	if Linkmode == LinkExternal {
		fh.Characteristics = IMAGE_FILE_LINE_NUMS_STRIPPED
	} else {
		fh.Characteristics = IMAGE_FILE_RELOCS_STRIPPED | IMAGE_FILE_EXECUTABLE_IMAGE | IMAGE_FILE_DEBUG_STRIPPED
	}
	if pe64 != 0 {
		fh.SizeOfOptionalHeader = uint16(binary.Size(&oh64))
		fh.Characteristics |= IMAGE_FILE_LARGE_ADDRESS_AWARE
		oh64.Magic = 0x20b // PE32+
	} else {
		fh.SizeOfOptionalHeader = uint16(binary.Size(&oh))
		fh.Characteristics |= IMAGE_FILE_32BIT_MACHINE
		oh.Magic = 0x10b // PE32
		oh.BaseOfData = d.VirtualAddress
	}

	// Fill out both oh64 and oh. We only use one. Oh well.
	oh64.MajorLinkerVersion = 3

	oh.MajorLinkerVersion = 3
	oh64.MinorLinkerVersion = 0
	oh.MinorLinkerVersion = 0
	oh64.SizeOfCode = t.SizeOfRawData
	oh.SizeOfCode = t.SizeOfRawData
	oh64.SizeOfInitializedData = d.SizeOfRawData
	oh.SizeOfInitializedData = d.SizeOfRawData
	oh64.SizeOfUninitializedData = 0
	oh.SizeOfUninitializedData = 0
	if Linkmode != LinkExternal {
		oh64.AddressOfEntryPoint = uint32(Entryvalue() - PEBASE)
		oh.AddressOfEntryPoint = uint32(Entryvalue() - PEBASE)
	}
	oh64.BaseOfCode = t.VirtualAddress
	oh.BaseOfCode = t.VirtualAddress
	oh64.ImageBase = PEBASE
	oh.ImageBase = PEBASE
	oh64.SectionAlignment = PESECTALIGN
	oh.SectionAlignment = PESECTALIGN
	oh64.FileAlignment = PEFILEALIGN
	oh.FileAlignment = PEFILEALIGN
	oh64.MajorOperatingSystemVersion = 4
	oh.MajorOperatingSystemVersion = 4
	oh64.MinorOperatingSystemVersion = 0
	oh.MinorOperatingSystemVersion = 0
	oh64.MajorImageVersion = 1
	oh.MajorImageVersion = 1
	oh64.MinorImageVersion = 0
	oh.MinorImageVersion = 0
	oh64.MajorSubsystemVersion = 4
	oh.MajorSubsystemVersion = 4
	oh64.MinorSubsystemVersion = 0
	oh.MinorSubsystemVersion = 0
	oh64.SizeOfImage = uint32(nextsectoff)
	oh.SizeOfImage = uint32(nextsectoff)
	oh64.SizeOfHeaders = uint32(PEFILEHEADR)
	oh.SizeOfHeaders = uint32(PEFILEHEADR)
	if headstring == "windowsgui" {
		oh64.Subsystem = IMAGE_SUBSYSTEM_WINDOWS_GUI
		oh.Subsystem = IMAGE_SUBSYSTEM_WINDOWS_GUI
	} else {
		oh64.Subsystem = IMAGE_SUBSYSTEM_WINDOWS_CUI
		oh.Subsystem = IMAGE_SUBSYSTEM_WINDOWS_CUI
	}

	// Disable stack growth as we don't want Windows to
	// fiddle with the thread stack limits, which we set
	// ourselves to circumvent the stack checks in the
	// Windows exception dispatcher.
	// Commit size must be strictly less than reserve
	// size otherwise reserve will be rounded up to a
	// larger size, as verified with VMMap.

	// Go code would be OK with 64k stacks, but we need larger stacks for cgo.
	//
	// The default stack reserve size affects only the main
	// thread, ctrlhandler thread, and profileloop thread. For
	// these, it must be greater than the stack size assumed by
	// externalthreadhandler.
	//
	// For other threads we specify stack size in runtime explicitly
	// (runtime knows whether cgo is enabled or not).
	// For these, the reserve must match STACKSIZE in
	// runtime/cgo/gcc_windows_{386,amd64}.c and the correspondent
	// CreateThread parameter in runtime.newosproc.
	if !iscgo {
		oh64.SizeOfStackReserve = 0x00020000
		oh.SizeOfStackReserve = 0x00020000
		oh64.SizeOfStackCommit = 0x00001000
		oh.SizeOfStackCommit = 0x00001000
	} else {
		oh64.SizeOfStackReserve = 0x00200000
		oh.SizeOfStackReserve = 0x00100000

		// account for 2 guard pages
		oh64.SizeOfStackCommit = 0x00200000 - 0x2000

		oh.SizeOfStackCommit = 0x00100000 - 0x2000
	}

	oh64.SizeOfHeapReserve = 0x00100000
	oh.SizeOfHeapReserve = 0x00100000
	oh64.SizeOfHeapCommit = 0x00001000
	oh.SizeOfHeapCommit = 0x00001000
	oh64.NumberOfRvaAndSizes = 16
	oh.NumberOfRvaAndSizes = 16

	pewrite()
}
