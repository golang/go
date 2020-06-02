// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Go new object file format, reading and writing.

package goobj2 // TODO: replace the goobj package?

import (
	"bytes"
	"cmd/internal/bio"
	"encoding/binary"
	"errors"
	"fmt"
	"internal/unsafeheader"
	"io"
	"unsafe"
)

// New object file format.
//
//    Header struct {
//       Magic       [...]byte   // "\x00go115ld"
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
//    DwarfFiles [...]string
//
//    SymbolDefs [...]struct {
//       Name string
//       ABI  uint16
//       Type uint8
//       Flag uint8
//       Size uint32
//    }
//    NonPkgDefs [...]struct { // non-pkg symbol definitions
//       ... // same as SymbolDefs
//    }
//    NonPkgRefs [...]struct { // non-pkg symbol references
//       ... // same as SymbolDefs
//    }
//
//    RelocIndex [...]uint32 // index to Relocs
//    AuxIndex   [...]uint32 // index to Aux
//    DataIndex  [...]uint32 // offset to Data
//
//    Relocs [...]struct {
//       Off  int32
//       Size uint8
//       Type uint8
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
//    Pcdata [...]byte
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
// - If PkgIdx is PkgIdxNone, SymIdx is the index of the symbol in the
//   NonPkgDefs array (could natually overflow to NonPkgRefs array).
// - Otherwise, SymIdx is the index of the symbol in some other package's
//   SymbolDefs array.
//
// {0, 0} represents a nil symbol. Otherwise PkgIdx should not be 0.
//
// RelocIndex, AuxIndex, and DataIndex contains indices/offsets to
// Relocs/Aux/Data blocks, one element per symbol, first for all the
// defined symbols, then all the defined non-package symbols, in the
// same order of SymbolDefs/NonPkgDefs arrays. For N total defined
// symbols, the array is of length N+1. The last element is the total
// number of relocations (aux symbols, data blocks, etc.).
//
// They can be accessed by index. For the i-th symbol, its relocations
// are the RelocIndex[i]-th (inclusive) to RelocIndex[i+1]-th (exclusive)
// elements in the Relocs array. Aux/Data are likewise. (The index is
// 0-based.)

// Auxiliary symbols.
//
// Each symbol may (or may not) be associated with a number of auxiliary
// symbols. They are described in the Aux block. See Aux struct below.
// Currently a symbol's Gotype and FuncInfo are auxiliary symbols. We
// may make use of aux symbols in more cases, e.g. DWARF symbols.

const stringRefSize = 8 // two uint32s

type FingerprintType [8]byte

func (fp FingerprintType) IsZero() bool { return fp == FingerprintType{} }

// Package Index.
const (
	PkgIdxNone    = (1<<31 - 1) - iota // Non-package symbols
	PkgIdxBuiltin                      // Predefined symbols // TODO: not used for now, we could use it for compiler-generated symbols like runtime.newobject
	PkgIdxSelf                         // Symbols defined in the current package
	PkgIdxInvalid = 0
	// The index of other referenced packages starts from 1.
)

// Blocks
const (
	BlkAutolib = iota
	BlkPkgIdx
	BlkDwarfFile
	BlkSymdef
	BlkNonpkgdef
	BlkNonpkgref
	BlkRelocIdx
	BlkAuxIdx
	BlkDataIdx
	BlkReloc
	BlkAux
	BlkData
	BlkPcdata
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

const Magic = "\x00go115ld"

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
	return len(h.Magic) + 4 + 4*len(h.Offsets)
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
// Sym struct {
//    Name  string
//    ABI   uint16
//    Type  uint8
//    Flag  uint8
//    Siz   uint32
//    Align uint32
// }
type Sym [SymSize]byte

const SymSize = stringRefSize + 2 + 1 + 1 + 4 + 4

const SymABIstatic = ^uint16(0)

const (
	ObjFlagShared = 1 << iota
)

const (
	SymFlagDupok = 1 << iota
	SymFlagLocal
	SymFlagTypelink
	SymFlagLeaf
	SymFlagNoSplit
	SymFlagReflectMethod
	SymFlagGoType
	SymFlagTopFrame
)

func (s *Sym) Name(r *Reader) string {
	len := binary.LittleEndian.Uint32(s[:])
	off := binary.LittleEndian.Uint32(s[4:])
	return r.StringAt(off, len)
}

func (s *Sym) ABI() uint16   { return binary.LittleEndian.Uint16(s[8:]) }
func (s *Sym) Type() uint8   { return s[10] }
func (s *Sym) Flag() uint8   { return s[11] }
func (s *Sym) Siz() uint32   { return binary.LittleEndian.Uint32(s[12:]) }
func (s *Sym) Align() uint32 { return binary.LittleEndian.Uint32(s[16:]) }

func (s *Sym) Dupok() bool         { return s.Flag()&SymFlagDupok != 0 }
func (s *Sym) Local() bool         { return s.Flag()&SymFlagLocal != 0 }
func (s *Sym) Typelink() bool      { return s.Flag()&SymFlagTypelink != 0 }
func (s *Sym) Leaf() bool          { return s.Flag()&SymFlagLeaf != 0 }
func (s *Sym) NoSplit() bool       { return s.Flag()&SymFlagNoSplit != 0 }
func (s *Sym) ReflectMethod() bool { return s.Flag()&SymFlagReflectMethod != 0 }
func (s *Sym) IsGoType() bool      { return s.Flag()&SymFlagGoType != 0 }
func (s *Sym) TopFrame() bool      { return s.Flag()&SymFlagTopFrame != 0 }

func (s *Sym) SetName(x string, w *Writer) {
	binary.LittleEndian.PutUint32(s[:], uint32(len(x)))
	binary.LittleEndian.PutUint32(s[4:], w.stringOff(x))
}

func (s *Sym) SetABI(x uint16)   { binary.LittleEndian.PutUint16(s[8:], x) }
func (s *Sym) SetType(x uint8)   { s[10] = x }
func (s *Sym) SetFlag(x uint8)   { s[11] = x }
func (s *Sym) SetSiz(x uint32)   { binary.LittleEndian.PutUint32(s[12:], x) }
func (s *Sym) SetAlign(x uint32) { binary.LittleEndian.PutUint32(s[16:], x) }

func (s *Sym) Write(w *Writer) { w.Bytes(s[:]) }

// for testing
func (s *Sym) fromBytes(b []byte) { copy(s[:], b) }

// Symbol reference.
type SymRef struct {
	PkgIdx uint32
	SymIdx uint32
}

// Relocation.
//
// Serialized format:
// Reloc struct {
//    Off  int32
//    Siz  uint8
//    Type uint8
//    Add  int64
//    Sym  SymRef
// }
type Reloc [RelocSize]byte

const RelocSize = 4 + 1 + 1 + 8 + 8

func (r *Reloc) Off() int32  { return int32(binary.LittleEndian.Uint32(r[:])) }
func (r *Reloc) Siz() uint8  { return r[4] }
func (r *Reloc) Type() uint8 { return r[5] }
func (r *Reloc) Add() int64  { return int64(binary.LittleEndian.Uint64(r[6:])) }
func (r *Reloc) Sym() SymRef {
	return SymRef{binary.LittleEndian.Uint32(r[14:]), binary.LittleEndian.Uint32(r[18:])}
}

func (r *Reloc) SetOff(x int32)  { binary.LittleEndian.PutUint32(r[:], uint32(x)) }
func (r *Reloc) SetSiz(x uint8)  { r[4] = x }
func (r *Reloc) SetType(x uint8) { r[5] = x }
func (r *Reloc) SetAdd(x int64)  { binary.LittleEndian.PutUint64(r[6:], uint64(x)) }
func (r *Reloc) SetSym(x SymRef) {
	binary.LittleEndian.PutUint32(r[14:], x.PkgIdx)
	binary.LittleEndian.PutUint32(r[18:], x.SymIdx)
}

func (r *Reloc) Set(off int32, size uint8, typ uint8, add int64, sym SymRef) {
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
// Aux struct {
//    Type uint8
//    Sym  SymRef
// }
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

	// TODO: more. Pcdata?
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

// Referenced symbol name.
//
// Serialized format:
// RefName struct {
//    Sym  symRef
//    Name string
// }
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
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], x)
	w.wr.Write(b[:])
	w.off += 8
}

func (w *Writer) Uint32(x uint32) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], x)
	w.wr.Write(b[:])
	w.off += 4
}

func (w *Writer) Uint16(x uint16) {
	var b [2]byte
	binary.LittleEndian.PutUint16(b[:], x)
	w.wr.Write(b[:])
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

	rd    io.ReaderAt
	start uint32
	h     Header // keep block offsets
}

func NewReaderFromBytes(b []byte, readonly bool) *Reader {
	r := &Reader{b: b, readonly: readonly, rd: bytes.NewReader(b), start: 0}
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

	var s string
	hdr := (*unsafeheader.String)(unsafe.Pointer(&s))
	hdr.Data = unsafe.Pointer(&b[0])
	hdr.Len = len(b)

	return s
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

func (r *Reader) NDwarfFile() int {
	return int(r.h.Offsets[BlkDwarfFile+1]-r.h.Offsets[BlkDwarfFile]) / stringRefSize
}

func (r *Reader) DwarfFile(i int) string {
	off := r.h.Offsets[BlkDwarfFile] + uint32(i)*stringRefSize
	return r.StringRef(off)
}

func (r *Reader) NSym() int {
	return int(r.h.Offsets[BlkSymdef+1]-r.h.Offsets[BlkSymdef]) / SymSize
}

func (r *Reader) NNonpkgdef() int {
	return int(r.h.Offsets[BlkNonpkgdef+1]-r.h.Offsets[BlkNonpkgdef]) / SymSize
}

func (r *Reader) NNonpkgref() int {
	return int(r.h.Offsets[BlkNonpkgref+1]-r.h.Offsets[BlkNonpkgref]) / SymSize
}

// SymOff returns the offset of the i-th symbol.
func (r *Reader) SymOff(i int) uint32 {
	return r.h.Offsets[BlkSymdef] + uint32(i*SymSize)
}

// Sym returns a pointer to the i-th symbol.
func (r *Reader) Sym(i int) *Sym {
	off := r.SymOff(i)
	return (*Sym)(unsafe.Pointer(&r.b[off]))
}

// NReloc returns the number of relocations of the i-th symbol.
func (r *Reader) NReloc(i int) int {
	relocIdxOff := r.h.Offsets[BlkRelocIdx] + uint32(i*4)
	return int(r.uint32At(relocIdxOff+4) - r.uint32At(relocIdxOff))
}

// RelocOff returns the offset of the j-th relocation of the i-th symbol.
func (r *Reader) RelocOff(i int, j int) uint32 {
	relocIdxOff := r.h.Offsets[BlkRelocIdx] + uint32(i*4)
	relocIdx := r.uint32At(relocIdxOff)
	return r.h.Offsets[BlkReloc] + (relocIdx+uint32(j))*uint32(RelocSize)
}

// Reloc returns a pointer to the j-th relocation of the i-th symbol.
func (r *Reader) Reloc(i int, j int) *Reloc {
	off := r.RelocOff(i, j)
	return (*Reloc)(unsafe.Pointer(&r.b[off]))
}

// Relocs returns a pointer to the relocations of the i-th symbol.
func (r *Reader) Relocs(i int) []Reloc {
	off := r.RelocOff(i, 0)
	n := r.NReloc(i)
	return (*[1 << 20]Reloc)(unsafe.Pointer(&r.b[off]))[:n:n]
}

// NAux returns the number of aux symbols of the i-th symbol.
func (r *Reader) NAux(i int) int {
	auxIdxOff := r.h.Offsets[BlkAuxIdx] + uint32(i*4)
	return int(r.uint32At(auxIdxOff+4) - r.uint32At(auxIdxOff))
}

// AuxOff returns the offset of the j-th aux symbol of the i-th symbol.
func (r *Reader) AuxOff(i int, j int) uint32 {
	auxIdxOff := r.h.Offsets[BlkAuxIdx] + uint32(i*4)
	auxIdx := r.uint32At(auxIdxOff)
	return r.h.Offsets[BlkAux] + (auxIdx+uint32(j))*uint32(AuxSize)
}

// Aux returns a pointer to the j-th aux symbol of the i-th symbol.
func (r *Reader) Aux(i int, j int) *Aux {
	off := r.AuxOff(i, j)
	return (*Aux)(unsafe.Pointer(&r.b[off]))
}

// Auxs returns the aux symbols of the i-th symbol.
func (r *Reader) Auxs(i int) []Aux {
	off := r.AuxOff(i, 0)
	n := r.NAux(i)
	return (*[1 << 20]Aux)(unsafe.Pointer(&r.b[off]))[:n:n]
}

// DataOff returns the offset of the i-th symbol's data.
func (r *Reader) DataOff(i int) uint32 {
	dataIdxOff := r.h.Offsets[BlkDataIdx] + uint32(i*4)
	return r.h.Offsets[BlkData] + r.uint32At(dataIdxOff)
}

// DataSize returns the size of the i-th symbol's data.
func (r *Reader) DataSize(i int) int {
	dataIdxOff := r.h.Offsets[BlkDataIdx] + uint32(i*4)
	return int(r.uint32At(dataIdxOff+4) - r.uint32At(dataIdxOff))
}

// Data returns the i-th symbol's data.
func (r *Reader) Data(i int) []byte {
	dataIdxOff := r.h.Offsets[BlkDataIdx] + uint32(i*4)
	base := r.h.Offsets[BlkData]
	off := r.uint32At(dataIdxOff)
	end := r.uint32At(dataIdxOff + 4)
	return r.BytesAt(base+off, int(end-off))
}

// AuxDataBase returns the base offset of the aux data block.
func (r *Reader) PcdataBase() uint32 {
	return r.h.Offsets[BlkPcdata]
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
