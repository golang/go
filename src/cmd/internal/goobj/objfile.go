// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This package defines the Go object file format, and provide "low-level" functions
// for reading and writing object files.

// The object file is understood by the compiler, assembler, linker, and tools. They
// have "high level" code that operates on object files, handling application-specific
// logics, and use this package for the actual reading and writing. Specifically, the
// code below:
//
// - cmd/internal/obj/objfile.go (used by cmd/asm and cmd/compile)
// - cmd/internal/objfile/goobj.go (used cmd/nm, cmd/objdump)
// - cmd/link/internal/loader package (used by cmd/link)
//
// If the object file format changes, they may (or may not) need to change.

package goobj

import (
	"cmd/internal/bio"
	"errors"
	"fmt"
	"internal/binary"
	"unsafe"
)

// New object file format.
//
//    Header struct {
//       Magic       [...]byte   // "\x00go120ld"
//       Fingerprint [8]byte
//       Flags       uint32
//       Offsets     [...]uint32 // byte offset of each block below
//    }
//
//    Strings [...]struct {
//       Data [...]byte
//    }
//
//    Autolib  [...]struct { // imported packages (for file loading)
//       Pkg         string
//       Fingerprint [8]byte
//    }
//
//    PkgIndex [...]string // referenced packages by index
//
//    Files [...]string
//
//    SymbolDefs [...]struct {
//       Name  string
//       ABI   uint16
//       Type  uint8
//       Flag  uint8
//       Flag2 uint8
//       Size  uint32
//    }
//    Hashed64Defs [...]struct { // short hashed (content-addressable) symbol definitions
//       ... // same as SymbolDefs
//    }
//    HashedDefs [...]struct { // hashed (content-addressable) symbol definitions
//       ... // same as SymbolDefs
//    }
//    NonPkgDefs [...]struct { // non-pkg symbol definitions
//       ... // same as SymbolDefs
//    }
//    NonPkgRefs [...]struct { // non-pkg symbol references
//       ... // same as SymbolDefs
//    }
//
//    RefFlags [...]struct { // referenced symbol flags
//       Sym   symRef
//       Flag  uint8
//       Flag2 uint8
//    }
//
//    Hash64 [...][8]byte
//    Hash   [...][N]byte
//
//    RelocIndex [...]uint32 // index to Relocs
//    AuxIndex   [...]uint32 // index to Aux
//    DataIndex  [...]uint32 // offset to Data
//
//    Relocs [...]struct {
//       Off  int32
//       Size uint8
//       Type uint16
//       Add  int64
//       Sym  symRef
//    }
//
//    Aux [...]struct {
//       Type uint8
//       Sym  symRef
//    }
//
//    Data   [...]byte
//
//    // blocks only used by tools (objdump, nm)
//
//    RefNames [...]struct { // referenced symbol names
//       Sym  symRef
//       Name string
//       // TODO: include ABI version as well?
//    }
//
// string is encoded as is a uint32 length followed by a uint32 offset
// that points to the corresponding string bytes.
//
// symRef is struct { PkgIdx, SymIdx uint32 }.
//
// Slice type (e.g. []symRef) is encoded as a length prefix (uint32)
// followed by that number of elements.
//
// The types below correspond to the encoded data structure in the
// object file.

// Symbol indexing.
//
// Each symbol is referenced with a pair of indices, { PkgIdx, SymIdx },
// as the symRef struct above.
//
// PkgIdx is either a predeclared index (see PkgIdxNone below) or
// an index of an imported package. For the latter case, PkgIdx is the
// index of the package in the PkgIndex array. 0 is an invalid index.
//
// SymIdx is the index of the symbol in the given package.
// - If PkgIdx is PkgIdxSelf, SymIdx is the index of the symbol in the
//   SymbolDefs array.
// - If PkgIdx is PkgIdxHashed64, SymIdx is the index of the symbol in the
//   Hashed64Defs array.
// - If PkgIdx is PkgIdxHashed, SymIdx is the index of the symbol in the
//   HashedDefs array.
// - If PkgIdx is PkgIdxNone, SymIdx is the index of the symbol in the
//   NonPkgDefs array (could naturally overflow to NonPkgRefs array).
// - Otherwise, SymIdx is the index of the symbol in some other package's
//   SymbolDefs array.
//
// {0, 0} represents a nil symbol. Otherwise PkgIdx should not be 0.
//
// Hash contains the content hashes of content-addressable symbols, of
// which PkgIdx is PkgIdxHashed, in the same order of HashedDefs array.
// Hash64 is similar, for PkgIdxHashed64 symbols.
//
// RelocIndex, AuxIndex, and DataIndex contains indices/offsets to
// Relocs/Aux/Data blocks, one element per symbol, first for all the
// defined symbols, then all the defined hashed and non-package symbols,
// in the same order of SymbolDefs/Hashed64Defs/HashedDefs/NonPkgDefs
// arrays. For N total defined symbols, the array is of length N+1. The
// last element is the total number of relocations (aux symbols, data
// blocks, etc.).
//
// They can be accessed by index. For the i-th symbol, its relocations
// are the RelocIndex[i]-th (inclusive) to RelocIndex[i+1]-th (exclusive)
// elements in the Relocs array. Aux/Data are likewise. (The index is
// 0-based.)

// Auxiliary symbols.
//
// Each symbol may (or may not) be associated with a number of auxiliary
// symbols. They are described in the Aux block. See Aux struct below.
// Currently a symbol's Gotype, FuncInfo, and associated DWARF symbols
// are auxiliary symbols.

const stringRefSize = 8 // two uint32s

type FingerprintType [8]byte

func (fp FingerprintType) IsZero() bool { return fp == FingerprintType{} }

// Package Index.
const (
	PkgIdxNone     = (1<<31 - 1) - iota // Non-package symbols
	PkgIdxHashed64                      // Short hashed (content-addressable) symbols
	PkgIdxHashed                        // Hashed (content-addressable) symbols
	PkgIdxBuiltin                       // Predefined runtime symbols (ex: runtime.newobject)
	PkgIdxSelf                          // Symbols defined in the current package
	PkgIdxSpecial  = PkgIdxSelf         // Indices above it has special meanings
	PkgIdxInvalid  = 0
	// The index of other referenced packages starts from 1.
)

// Blocks
const (
	BlkAutolib = iota
	BlkPkgIdx
	BlkFile
	BlkSymdef
	BlkHashed64def
	BlkHasheddef
	BlkNonpkgdef
	BlkNonpkgref
	BlkRefFlags
	BlkHash64
	BlkHash
	BlkRelocIdx
	BlkAuxIdx
	BlkDataIdx
	BlkReloc
	BlkAux
	BlkData
	BlkRefName
	BlkEnd
	NBlk
)

// File header.
// TODO: probably no need to export this.
type Header struct {
	Magic       string
	Fingerprint FingerprintType
	Flags       uint32
	Offsets     [NBlk]uint32
}

const Magic = "\x00go120ld"

func (h *Header) Write(w *Writer) {
	w.RawString(h.Magic)
	w.Bytes(h.Fingerprint[:])
	w.Uint32(h.Flags)
	for _, x := range h.Offsets {
		w.Uint32(x)
	}
}

func (h *Header) Read(r *Reader) error {
	b := r.BytesAt(0, len(Magic))
	h.Magic = string(b)
	if h.Magic != Magic {
		return errors.New("wrong magic, not a Go object file")
	}
	off := uint32(len(h.Magic))
	copy(h.Fingerprint[:], r.BytesAt(off, len(h.Fingerprint)))
	off += 8
	h.Flags = r.uint32At(off)
	off += 4
	for i := range h.Offsets {
		h.Offsets[i] = r.uint32At(off)
		off += 4
	}
	return nil
}

func (h *Header) Size() int {
	return len(h.Magic) + len(h.Fingerprint) + 4 + 4*len(h.Offsets)
}

// Autolib
type ImportedPkg struct {
	Pkg         string
	Fingerprint FingerprintType
}

const importedPkgSize = stringRefSize + 8

func (p *ImportedPkg) Write(w *Writer) {
	w.StringRef(p.Pkg)
	w.Bytes(p.Fingerprint[:])
}

// Symbol definition.
//
// Serialized format:
//
//	Sym struct {
//	   Name  string
//	   ABI   uint16
//	   Type  uint8
//	   Flag  uint8
//	   Flag2 uint8
//	   Siz   uint32
//	   Align uint32
//	}
type Sym [SymSize]byte

const SymSize = stringRefSize + 2 + 1 + 1 + 1 + 4 + 4

const SymABIstatic = ^uint16(0)

const (
	ObjFlagShared       = 1 << iota // this object is built with -shared
	_                               // was ObjFlagNeedNameExpansion
	ObjFlagFromAssembly             // object is from asm src, not go
	ObjFlagUnlinkable               // unlinkable package (linker will emit an error)
)

// Sym.Flag
const (
	SymFlagDupok = 1 << iota
	SymFlagLocal
	SymFlagTypelink
	SymFlagLeaf
	SymFlagNoSplit
	SymFlagReflectMethod
	SymFlagGoType
)

// Sym.Flag2
const (
	SymFlagUsedInIface = 1 << iota
	SymFlagItab
	SymFlagDict
	SymFlagPkgInit
)

// Returns the length of the name of the symbol.
func (s *Sym) NameLen(r *Reader) int {
	return int(binary.LittleEndian.Uint32(s[:]))
}

func (s *Sym) Name(r *Reader) string {
	len := binary.LittleEndian.Uint32(s[:])
	off := binary.LittleEndian.Uint32(s[4:])
	return r.StringAt(off, len)
}

func (s *Sym) ABI() uint16   { return binary.LittleEndian.Uint16(s[8:]) }
func (s *Sym) Type() uint8   { return s[10] }
func (s *Sym) Flag() uint8   { return s[11] }
func (s *Sym) Flag2() uint8  { return s[12] }
func (s *Sym) Siz() uint32   { return binary.LittleEndian.Uint32(s[13:]) }
func (s *Sym) Align() uint32 { return binary.LittleEndian.Uint32(s[17:]) }

func (s *Sym) Dupok() bool         { return s.Flag()&SymFlagDupok != 0 }
func (s *Sym) Local() bool         { return s.Flag()&SymFlagLocal != 0 }
func (s *Sym) Typelink() bool      { return s.Flag()&SymFlagTypelink != 0 }
func (s *Sym) Leaf() bool          { return s.Flag()&SymFlagLeaf != 0 }
func (s *Sym) NoSplit() bool       { return s.Flag()&SymFlagNoSplit != 0 }
func (s *Sym) ReflectMethod() bool { return s.Flag()&SymFlagReflectMethod != 0 }
func (s *Sym) IsGoType() bool      { return s.Flag()&SymFlagGoType != 0 }
func (s *Sym) UsedInIface() bool   { return s.Flag2()&SymFlagUsedInIface != 0 }
func (s *Sym) IsItab() bool        { return s.Flag2()&SymFlagItab != 0 }
func (s *Sym) IsDict() bool        { return s.Flag2()&SymFlagDict != 0 }
func (s *Sym) IsPkgInit() bool     { return s.Flag2()&SymFlagPkgInit != 0 }

func (s *Sym) SetName(x string, w *Writer) {
	binary.LittleEndian.PutUint32(s[:], uint32(len(x)))
	binary.LittleEndian.PutUint32(s[4:], w.stringOff(x))
}

func (s *Sym) SetABI(x uint16)   { binary.LittleEndian.PutUint16(s[8:], x) }
func (s *Sym) SetType(x uint8)   { s[10] = x }
func (s *Sym) SetFlag(x uint8)   { s[11] = x }
func (s *Sym) SetFlag2(x uint8)  { s[12] = x }
func (s *Sym) SetSiz(x uint32)   { binary.LittleEndian.PutUint32(s[13:], x) }
func (s *Sym) SetAlign(x uint32) { binary.LittleEndian.PutUint32(s[17:], x) }

func (s *Sym) Write(w *Writer) { w.Bytes(s[:]) }

// for testing
func (s *Sym) fromBytes(b []byte) { copy(s[:], b) }

// Symbol reference.
type SymRef struct {
	PkgIdx uint32
	SymIdx uint32
}

func (s SymRef) IsZero() bool { return s == SymRef{} }

// Hash64
type Hash64Type [Hash64Size]byte

const Hash64Size = 8

// Hash
type HashType [HashSize]byte

const HashSize = 16 // truncated SHA256

// Relocation.
//
// Serialized format:
//
//	Reloc struct {
//	   Off  int32
//	   Siz  uint8
//	   Type uint16
//	   Add  int64
//	   Sym  SymRef
//	}
type Reloc [RelocSize]byte

const RelocSize = 4 + 1 + 2 + 8 + 8

func (r *Reloc) Off() int32   { return int32(binary.LittleEndian.Uint32(r[:])) }
func (r *Reloc) Siz() uint8   { return r[4] }
func (r *Reloc) Type() uint16 { return binary.LittleEndian.Uint16(r[5:]) }
func (r *Reloc) Add() int64   { return int64(binary.LittleEndian.Uint64(r[7:])) }
func (r *Reloc) Sym() SymRef {
	return SymRef{binary.LittleEndian.Uint32(r[15:]), binary.LittleEndian.Uint32(r[19:])}
}

func (r *Reloc) SetOff(x int32)   { binary.LittleEndian.PutUint32(r[:], uint32(x)) }
func (r *Reloc) SetSiz(x uint8)   { r[4] = x }
func (r *Reloc) SetType(x uint16) { binary.LittleEndian.PutUint16(r[5:], x) }
func (r *Reloc) SetAdd(x int64)   { binary.LittleEndian.PutUint64(r[7:], uint64(x)) }
func (r *Reloc) SetSym(x SymRef) {
	binary.LittleEndian.PutUint32(r[15:], x.PkgIdx)
	binary.LittleEndian.PutUint32(r[19:], x.SymIdx)
}

func (r *Reloc) Set(off int32, size uint8, typ uint16, add int64, sym SymRef) {
	r.SetOff(off)
	r.SetSiz(size)
	r.SetType(typ)
	r.SetAdd(add)
	r.SetSym(sym)
}

func (r *Reloc) Write(w *Writer) { w.Bytes(r[:]) }

// for testing
func (r *Reloc) fromBytes(b []byte) { copy(r[:], b) }

// Aux symbol info.
//
// Serialized format:
//
//	Aux struct {
//	   Type uint8
//	   Sym  SymRef
//	}
type Aux [AuxSize]byte

const AuxSize = 1 + 8

// Aux Type
const (
	AuxGotype = iota
	AuxFuncInfo
	AuxFuncdata
	AuxDwarfInfo
	AuxDwarfLoc
	AuxDwarfRanges
	AuxDwarfLines
	AuxPcsp
	AuxPcfile
	AuxPcline
	AuxPcinline
	AuxPcdata
	AuxWasmImport
	AuxSehUnwindInfo
)

func (a *Aux) Type() uint8 { return a[0] }
func (a *Aux) Sym() SymRef {
	return SymRef{binary.LittleEndian.Uint32(a[1:]), binary.LittleEndian.Uint32(a[5:])}
}

func (a *Aux) SetType(x uint8) { a[0] = x }
func (a *Aux) SetSym(x SymRef) {
	binary.LittleEndian.PutUint32(a[1:], x.PkgIdx)
	binary.LittleEndian.PutUint32(a[5:], x.SymIdx)
}

func (a *Aux) Write(w *Writer) { w.Bytes(a[:]) }

// for testing
func (a *Aux) fromBytes(b []byte) { copy(a[:], b) }

// Referenced symbol flags.
//
// Serialized format:
//
//	RefFlags struct {
//	   Sym   symRef
//	   Flag  uint8
//	   Flag2 uint8
//	}
type RefFlags [RefFlagsSize]byte

const RefFlagsSize = 8 + 1 + 1

func (r *RefFlags) Sym() SymRef {
	return SymRef{binary.LittleEndian.Uint32(r[:]), binary.LittleEndian.Uint32(r[4:])}
}
func (r *RefFlags) Flag() uint8  { return r[8] }
func (r *RefFlags) Flag2() uint8 { return r[9] }

func (r *RefFlags) SetSym(x SymRef) {
	binary.LittleEndian.PutUint32(r[:], x.PkgIdx)
	binary.LittleEndian.PutUint32(r[4:], x.SymIdx)
}
func (r *RefFlags) SetFlag(x uint8)  { r[8] = x }
func (r *RefFlags) SetFlag2(x uint8) { r[9] = x }

func (r *RefFlags) Write(w *Writer) { w.Bytes(r[:]) }

// Used to construct an artificially large array type when reading an
// item from the object file relocs section or aux sym section (needs
// to work on 32-bit as well as 64-bit). See issue 41621.
const huge = (1<<31 - 1) / RelocSize

// Referenced symbol name.
//
// Serialized format:
//
//	RefName struct {
//	   Sym  symRef
//	   Name string
//	}
type RefName [RefNameSize]byte

const RefNameSize = 8 + stringRefSize

func (n *RefName) Sym() SymRef {
	return SymRef{binary.LittleEndian.Uint32(n[:]), binary.LittleEndian.Uint32(n[4:])}
}
func (n *RefName) Name(r *Reader) string {
	len := binary.LittleEndian.Uint32(n[8:])
	off := binary.LittleEndian.Uint32(n[12:])
	return r.StringAt(off, len)
}

func (n *RefName) SetSym(x SymRef) {
	binary.LittleEndian.PutUint32(n[:], x.PkgIdx)
	binary.LittleEndian.PutUint32(n[4:], x.SymIdx)
}
func (n *RefName) SetName(x string, w *Writer) {
	binary.LittleEndian.PutUint32(n[8:], uint32(len(x)))
	binary.LittleEndian.PutUint32(n[12:], w.stringOff(x))
}

func (n *RefName) Write(w *Writer) { w.Bytes(n[:]) }

type Writer struct {
	wr        *bio.Writer
	stringMap map[string]uint32
	off       uint32 // running offset

	b [8]byte // scratch space for writing bytes
}

func NewWriter(wr *bio.Writer) *Writer {
	return &Writer{wr: wr, stringMap: make(map[string]uint32)}
}

func (w *Writer) AddString(s string) {
	if _, ok := w.stringMap[s]; ok {
		return
	}
	w.stringMap[s] = w.off
	w.RawString(s)
}

func (w *Writer) stringOff(s string) uint32 {
	off, ok := w.stringMap[s]
	if !ok {
		panic(fmt.Sprintf("writeStringRef: string not added: %q", s))
	}
	return off
}

func (w *Writer) StringRef(s string) {
	w.Uint32(uint32(len(s)))
	w.Uint32(w.stringOff(s))
}

func (w *Writer) RawString(s string) {
	w.wr.WriteString(s)
	w.off += uint32(len(s))
}

func (w *Writer) Bytes(s []byte) {
	w.wr.Write(s)
	w.off += uint32(len(s))
}

func (w *Writer) Uint64(x uint64) {
	binary.LittleEndian.PutUint64(w.b[:], x)
	w.wr.Write(w.b[:])
	w.off += 8
}

func (w *Writer) Uint32(x uint32) {
	binary.LittleEndian.PutUint32(w.b[:4], x)
	w.wr.Write(w.b[:4])
	w.off += 4
}

func (w *Writer) Uint16(x uint16) {
	binary.LittleEndian.PutUint16(w.b[:2], x)
	w.wr.Write(w.b[:2])
	w.off += 2
}

func (w *Writer) Uint8(x uint8) {
	w.wr.WriteByte(x)
	w.off++
}

func (w *Writer) Offset() uint32 {
	return w.off
}

type Reader struct {
	b        []byte // mmapped bytes, if not nil
	readonly bool   // whether b is backed with read-only memory

	start uint32
	h     Header // keep block offsets
}

func NewReaderFromBytes(b []byte, readonly bool) *Reader {
	r := &Reader{b: b, readonly: readonly, start: 0}
	err := r.h.Read(r)
	if err != nil {
		return nil
	}
	return r
}

func (r *Reader) BytesAt(off uint32, len int) []byte {
	if len == 0 {
		return nil
	}
	end := int(off) + len
	return r.b[int(off):end:end]
}

func (r *Reader) uint64At(off uint32) uint64 {
	b := r.BytesAt(off, 8)
	return binary.LittleEndian.Uint64(b)
}

func (r *Reader) int64At(off uint32) int64 {
	return int64(r.uint64At(off))
}

func (r *Reader) uint32At(off uint32) uint32 {
	b := r.BytesAt(off, 4)
	return binary.LittleEndian.Uint32(b)
}

func (r *Reader) int32At(off uint32) int32 {
	return int32(r.uint32At(off))
}

func (r *Reader) uint16At(off uint32) uint16 {
	b := r.BytesAt(off, 2)
	return binary.LittleEndian.Uint16(b)
}

func (r *Reader) uint8At(off uint32) uint8 {
	b := r.BytesAt(off, 1)
	return b[0]
}

func (r *Reader) StringAt(off uint32, len uint32) string {
	b := r.b[off : off+len]
	if r.readonly {
		return toString(b) // backed by RO memory, ok to make unsafe string
	}
	return string(b)
}

func toString(b []byte) string {
	if len(b) == 0 {
		return ""
	}
	return unsafe.String(&b[0], len(b))
}

func (r *Reader) StringRef(off uint32) string {
	l := r.uint32At(off)
	return r.StringAt(r.uint32At(off+4), l)
}

func (r *Reader) Fingerprint() FingerprintType {
	return r.h.Fingerprint
}

func (r *Reader) Autolib() []ImportedPkg {
	n := (r.h.Offsets[BlkAutolib+1] - r.h.Offsets[BlkAutolib]) / importedPkgSize
	s := make([]ImportedPkg, n)
	off := r.h.Offsets[BlkAutolib]
	for i := range s {
		s[i].Pkg = r.StringRef(off)
		copy(s[i].Fingerprint[:], r.BytesAt(off+stringRefSize, len(s[i].Fingerprint)))
		off += importedPkgSize
	}
	return s
}

func (r *Reader) Pkglist() []string {
	n := (r.h.Offsets[BlkPkgIdx+1] - r.h.Offsets[BlkPkgIdx]) / stringRefSize
	s := make([]string, n)
	off := r.h.Offsets[BlkPkgIdx]
	for i := range s {
		s[i] = r.StringRef(off)
		off += stringRefSize
	}
	return s
}

func (r *Reader) NPkg() int {
	return int(r.h.Offsets[BlkPkgIdx+1]-r.h.Offsets[BlkPkgIdx]) / stringRefSize
}

func (r *Reader) Pkg(i int) string {
	off := r.h.Offsets[BlkPkgIdx] + uint32(i)*stringRefSize
	return r.StringRef(off)
}

func (r *Reader) NFile() int {
	return int(r.h.Offsets[BlkFile+1]-r.h.Offsets[BlkFile]) / stringRefSize
}

func (r *Reader) File(i int) string {
	off := r.h.Offsets[BlkFile] + uint32(i)*stringRefSize
	return r.StringRef(off)
}

func (r *Reader) NSym() int {
	return int(r.h.Offsets[BlkSymdef+1]-r.h.Offsets[BlkSymdef]) / SymSize
}

func (r *Reader) NHashed64def() int {
	return int(r.h.Offsets[BlkHashed64def+1]-r.h.Offsets[BlkHashed64def]) / SymSize
}

func (r *Reader) NHasheddef() int {
	return int(r.h.Offsets[BlkHasheddef+1]-r.h.Offsets[BlkHasheddef]) / SymSize
}

func (r *Reader) NNonpkgdef() int {
	return int(r.h.Offsets[BlkNonpkgdef+1]-r.h.Offsets[BlkNonpkgdef]) / SymSize
}

func (r *Reader) NNonpkgref() int {
	return int(r.h.Offsets[BlkNonpkgref+1]-r.h.Offsets[BlkNonpkgref]) / SymSize
}

// SymOff returns the offset of the i-th symbol.
func (r *Reader) SymOff(i uint32) uint32 {
	return r.h.Offsets[BlkSymdef] + uint32(i*SymSize)
}

// Sym returns a pointer to the i-th symbol.
func (r *Reader) Sym(i uint32) *Sym {
	off := r.SymOff(i)
	return (*Sym)(unsafe.Pointer(&r.b[off]))
}

// NRefFlags returns the number of referenced symbol flags.
func (r *Reader) NRefFlags() int {
	return int(r.h.Offsets[BlkRefFlags+1]-r.h.Offsets[BlkRefFlags]) / RefFlagsSize
}

// RefFlags returns a pointer to the i-th referenced symbol flags.
// Note: here i is not a local symbol index, just a counter.
func (r *Reader) RefFlags(i int) *RefFlags {
	off := r.h.Offsets[BlkRefFlags] + uint32(i*RefFlagsSize)
	return (*RefFlags)(unsafe.Pointer(&r.b[off]))
}

// Hash64 returns the i-th short hashed symbol's hash.
// Note: here i is the index of short hashed symbols, not all symbols
// (unlike other accessors).
func (r *Reader) Hash64(i uint32) uint64 {
	off := r.h.Offsets[BlkHash64] + uint32(i*Hash64Size)
	return r.uint64At(off)
}

// Hash returns a pointer to the i-th hashed symbol's hash.
// Note: here i is the index of hashed symbols, not all symbols
// (unlike other accessors).
func (r *Reader) Hash(i uint32) *HashType {
	off := r.h.Offsets[BlkHash] + uint32(i*HashSize)
	return (*HashType)(unsafe.Pointer(&r.b[off]))
}

// NReloc returns the number of relocations of the i-th symbol.
func (r *Reader) NReloc(i uint32) int {
	relocIdxOff := r.h.Offsets[BlkRelocIdx] + uint32(i*4)
	return int(r.uint32At(relocIdxOff+4) - r.uint32At(relocIdxOff))
}

// RelocOff returns the offset of the j-th relocation of the i-th symbol.
func (r *Reader) RelocOff(i uint32, j int) uint32 {
	relocIdxOff := r.h.Offsets[BlkRelocIdx] + uint32(i*4)
	relocIdx := r.uint32At(relocIdxOff)
	return r.h.Offsets[BlkReloc] + (relocIdx+uint32(j))*uint32(RelocSize)
}

// Reloc returns a pointer to the j-th relocation of the i-th symbol.
func (r *Reader) Reloc(i uint32, j int) *Reloc {
	off := r.RelocOff(i, j)
	return (*Reloc)(unsafe.Pointer(&r.b[off]))
}

// Relocs returns a pointer to the relocations of the i-th symbol.
func (r *Reader) Relocs(i uint32) []Reloc {
	off := r.RelocOff(i, 0)
	n := r.NReloc(i)
	return (*[huge]Reloc)(unsafe.Pointer(&r.b[off]))[:n:n]
}

// NAux returns the number of aux symbols of the i-th symbol.
func (r *Reader) NAux(i uint32) int {
	auxIdxOff := r.h.Offsets[BlkAuxIdx] + i*4
	return int(r.uint32At(auxIdxOff+4) - r.uint32At(auxIdxOff))
}

// AuxOff returns the offset of the j-th aux symbol of the i-th symbol.
func (r *Reader) AuxOff(i uint32, j int) uint32 {
	auxIdxOff := r.h.Offsets[BlkAuxIdx] + i*4
	auxIdx := r.uint32At(auxIdxOff)
	return r.h.Offsets[BlkAux] + (auxIdx+uint32(j))*uint32(AuxSize)
}

// Aux returns a pointer to the j-th aux symbol of the i-th symbol.
func (r *Reader) Aux(i uint32, j int) *Aux {
	off := r.AuxOff(i, j)
	return (*Aux)(unsafe.Pointer(&r.b[off]))
}

// Auxs returns the aux symbols of the i-th symbol.
func (r *Reader) Auxs(i uint32) []Aux {
	off := r.AuxOff(i, 0)
	n := r.NAux(i)
	return (*[huge]Aux)(unsafe.Pointer(&r.b[off]))[:n:n]
}

// DataOff returns the offset of the i-th symbol's data.
func (r *Reader) DataOff(i uint32) uint32 {
	dataIdxOff := r.h.Offsets[BlkDataIdx] + i*4
	return r.h.Offsets[BlkData] + r.uint32At(dataIdxOff)
}

// DataSize returns the size of the i-th symbol's data.
func (r *Reader) DataSize(i uint32) int {
	dataIdxOff := r.h.Offsets[BlkDataIdx] + i*4
	return int(r.uint32At(dataIdxOff+4) - r.uint32At(dataIdxOff))
}

// Data returns the i-th symbol's data.
func (r *Reader) Data(i uint32) []byte {
	dataIdxOff := r.h.Offsets[BlkDataIdx] + i*4
	base := r.h.Offsets[BlkData]
	off := r.uint32At(dataIdxOff)
	end := r.uint32At(dataIdxOff + 4)
	return r.BytesAt(base+off, int(end-off))
}

// DataString returns the i-th symbol's data as a string.
func (r *Reader) DataString(i uint32) string {
	dataIdxOff := r.h.Offsets[BlkDataIdx] + i*4
	base := r.h.Offsets[BlkData]
	off := r.uint32At(dataIdxOff)
	end := r.uint32At(dataIdxOff + 4)
	return r.StringAt(base+off, end-off)
}

// NRefName returns the number of referenced symbol names.
func (r *Reader) NRefName() int {
	return int(r.h.Offsets[BlkRefName+1]-r.h.Offsets[BlkRefName]) / RefNameSize
}

// RefName returns a pointer to the i-th referenced symbol name.
// Note: here i is not a local symbol index, just a counter.
func (r *Reader) RefName(i int) *RefName {
	off := r.h.Offsets[BlkRefName] + uint32(i*RefNameSize)
	return (*RefName)(unsafe.Pointer(&r.b[off]))
}

// ReadOnly returns whether r.BytesAt returns read-only bytes.
func (r *Reader) ReadOnly() bool {
	return r.readonly
}

// Flags returns the flag bits read from the object file header.
func (r *Reader) Flags() uint32 {
	return r.h.Flags
}

func (r *Reader) Shared() bool       { return r.Flags()&ObjFlagShared != 0 }
func (r *Reader) FromAssembly() bool { return r.Flags()&ObjFlagFromAssembly != 0 }
func (r *Reader) Unlinkable() bool   { return r.Flags()&ObjFlagUnlinkable != 0 }
