// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"encoding/binary"
	"fmt"
	"sort"
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

var fh IMAGE_FILE_HEADER

var oh IMAGE_OPTIONAL_HEADER

var oh64 PE64_IMAGE_OPTIONAL_HEADER

var sh [16]IMAGE_SECTION_HEADER

var dd []IMAGE_DATA_DIRECTORY

type Imp struct {
	s    *LSym
	off  uint64
	next *Imp
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

type COFFSym struct {
	sym       *LSym
	strtbloff int
	sect      int
	value     int64
}

var coffsym []COFFSym

var ncoffsym int

func addpesection(name string, sectsize int, filesize int) *IMAGE_SECTION_HEADER {
	if pensect == 16 {
		Diag("too many sections")
		Errorexit()
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
		Errorexit()
	}
}

func chksectseg(h *IMAGE_SECTION_HEADER, s *Segment) {
	if s.Vaddr-PEBASE != uint64(h.VirtualAddress) {
		Diag("%s.VirtualAddress = %#x, want %#x", cstring(h.Name[:]), uint64(int64(h.VirtualAddress)), uint64(int64(s.Vaddr-PEBASE)))
		Errorexit()
	}

	if s.Fileoff != uint64(h.PointerToRawData) {
		Diag("%s.PointerToRawData = %#x, want %#x", cstring(h.Name[:]), uint64(int64(h.PointerToRawData)), uint64(int64(s.Fileoff)))
		Errorexit()
	}
}

func Peinit() {
	var l int

	switch Thearch.Thechar {
	// 64-bit architectures
	case '6':
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
	xdefine("__image_base__", SDATA, PEBASE)

	xdefine("_image_base__", SDATA, PEBASE)
}

func pewrite() {
	Cseek(0)
	Cwrite(dosstub)
	strnput("PE", 4)

	binary.Write(&coutbuf, binary.LittleEndian, &fh)

	if pe64 != 0 {
		binary.Write(&coutbuf, binary.LittleEndian, &oh64)
	} else {
		binary.Write(&coutbuf, binary.LittleEndian, &oh)
	}
	binary.Write(&coutbuf, binary.LittleEndian, sh[:pensect])
}

func strput(s string) {
	coutbuf.w.WriteString(s)
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
	for s := Ctxt.Allsym; s != nil; s = s.Allsym {
		if !s.Reachable || s.Type != SDYNIMPORT {
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

		m.s = s
		m.next = d.ms
		d.ms = m
	}

	dynamic := Linklookup(Ctxt, ".windynamic", 0)
	dynamic.Reachable = true
	dynamic.Type = SWINDOWS
	for d := dr; d != nil; d = d.next {
		for m = d.ms; m != nil; m = m.next {
			m.s.Type = SWINDOWS | SSUB
			m.s.Sub = dynamic.Sub
			dynamic.Sub = m.s
			m.s.Value = dynamic.Size
			dynamic.Size += int64(Thearch.Ptrsize)
		}

		dynamic.Size += int64(Thearch.Ptrsize)
	}

	return dr
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

type pescmp []*LSym

func (x pescmp) Len() int {
	return len(x)
}

func (x pescmp) Swap(i, j int) {
	x[i], x[j] = x[j], x[i]
}

func (x pescmp) Less(i, j int) bool {
	s1 := x[i]
	s2 := x[j]
	return stringsCompare(s1.Extname, s2.Extname) < 0
}

func initdynexport() {
	nexport = 0
	for s := Ctxt.Allsym; s != nil; s = s.Allsym {
		if !s.Reachable || s.Cgoexport&CgoExportDynamic == 0 {
			continue
		}
		if nexport+1 > len(dexport) {
			Diag("pe dynexport table is full")
			Errorexit()
		}

		dexport[nexport] = s
		nexport++
	}

	sort.Sort(pescmp(dexport[:nexport]))
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

func dope() {
	/* relocation table */
	rel := Linklookup(Ctxt, ".rel", 0)

	rel.Reachable = true
	rel.Type = SELFROSECT

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

func addpesym(s *LSym, name string, type_ int, addr int64, size int64, ver int, gotype *LSym) {
	if s == nil {
		return
	}

	if s.Sect == nil {
		return
	}

	switch type_ {
	default:
		return

	case 'D',
		'B',
		'T':
		break
	}

	if coffsym != nil {
		cs := &coffsym[ncoffsym]
		cs.sym = s
		if len(s.Name) > 8 {
			cs.strtbloff = strtbladd(s.Name)
		}
		if uint64(s.Value) >= Segdata.Vaddr {
			cs.value = int64(uint64(s.Value) - Segdata.Vaddr)
			cs.sect = datasect
		} else if uint64(s.Value) >= Segtext.Vaddr {
			cs.value = int64(uint64(s.Value) - Segtext.Vaddr)
			cs.sect = textsect
		} else {
			cs.value = 0
			cs.sect = 0
			Diag("addpesym %#x", addr)
		}
	}

	ncoffsym++
}

func addpesymtable() {
	if Debug['s'] == 0 {
		genasmsym(addpesym)
		coffsym = make([]COFFSym, ncoffsym)
		ncoffsym = 0
		genasmsym(addpesym)
	}

	size := len(strtbl) + 4 + 18*ncoffsym
	h := addpesection(".symtab", size, size)
	h.Characteristics = IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_DISCARDABLE
	chksectoff(h, Cpos())
	fh.PointerToSymbolTable = uint32(Cpos())
	fh.NumberOfSymbols = uint32(ncoffsym)

	// put COFF symbol table
	var s *COFFSym
	for i := 0; i < ncoffsym; i++ {
		s = &coffsym[i]
		if s.strtbloff == 0 {
			strnput(s.sym.Name, 8)
		} else {
			Lputl(0)
			Lputl(uint32(s.strtbloff))
		}

		Lputl(uint32(s.value))
		Wputl(uint16(s.sect))
		Wputl(0x0308) // "array of structs"
		Cput(2)       // storage class: external
		Cput(0)       // no aux entries
	}

	// put COFF string table
	Lputl(uint32(len(strtbl)) + 4)

	for i := 0; i < len(strtbl); i++ {
		Cput(uint8(strtbl[i]))
	}
	strnput("", int(h.SizeOfRawData-uint32(size)))
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

func Asmbpe() {
	switch Thearch.Thechar {
	default:
		Diag("unknown PE architecture")
		Errorexit()
		fallthrough

	case '6':
		fh.Machine = IMAGE_FILE_MACHINE_AMD64

	case '8':
		fh.Machine = IMAGE_FILE_MACHINE_I386
	}

	t := addpesection(".text", int(Segtext.Length), int(Segtext.Length))
	t.Characteristics = IMAGE_SCN_CNT_CODE | IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_EXECUTE | IMAGE_SCN_MEM_READ
	chksectseg(t, &Segtext)
	textsect = pensect

	d := addpesection(".data", int(Segdata.Length), int(Segdata.Filelen))
	d.Characteristics = IMAGE_SCN_CNT_INITIALIZED_DATA | IMAGE_SCN_MEM_READ | IMAGE_SCN_MEM_WRITE
	chksectseg(d, &Segdata)
	datasect = pensect

	if Debug['s'] == 0 {
		dwarfaddpeheaders()
	}

	Cseek(int64(nextfileoff))
	addimports(d)
	addexports()
	addpesymtable()
	addpersrc()

	fh.NumberOfSections = uint16(pensect)

	// Being able to produce identical output for identical input is
	// much more beneficial than having build timestamp in the header.
	fh.TimeDateStamp = 0

	fh.Characteristics = IMAGE_FILE_RELOCS_STRIPPED | IMAGE_FILE_EXECUTABLE_IMAGE | IMAGE_FILE_DEBUG_STRIPPED
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
	oh64.AddressOfEntryPoint = uint32(Entryvalue() - PEBASE)
	oh.AddressOfEntryPoint = uint32(Entryvalue() - PEBASE)
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
	// That default stack reserve size affects only the main thread,
	// for other threads we specify stack size in runtime explicitly
	// (runtime knows whether cgo is enabled or not).
	// If you change stack reserve sizes here,
	// change STACKSIZE in runtime/cgo/gcc_windows_{386,amd64}.c as well.
	if !iscgo {
		oh64.SizeOfStackReserve = 0x00010000
		oh.SizeOfStackReserve = 0x00010000
		oh64.SizeOfStackCommit = 0x0000ffff
		oh.SizeOfStackCommit = 0x0000ffff
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
