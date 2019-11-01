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
	"io"
	"unsafe"
)

// New object file format.
//
//    Header struct {
//       Magic   [...]byte   // "\x00go114LD"
//       Flags   uint32
//       // TODO: Fingerprint
//       Offsets [...]uint32 // byte offset of each block below
//    }
//
//    Strings [...]struct {
//       Len  uint32
//       Data [...]byte
//    }
//
//    Autolib  [...]stringOff // imported packages (for file loading) // TODO: add fingerprints
//    PkgIndex [...]stringOff // referenced packages by index
//
//    DwarfFiles [...]stringOff
//
//    SymbolDefs [...]struct {
//       Name stringOff
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
// stringOff is a uint32 (?) offset that points to the corresponding
// string, which is a uint32 length followed by that number of bytes.
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
	NBlk
)

// File header.
// TODO: probably no need to export this.
type Header struct {
	Magic   string
	Flags   uint32
	Offsets [NBlk]uint32
}

const Magic = "\x00go114LD"

func (h *Header) Write(w *Writer) {
	w.RawString(h.Magic)
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

// Symbol definition.
type Sym struct {
	Name string
	ABI  uint16
	Type uint8
	Flag uint8
	Siz  uint32
}

const SymABIstatic = ^uint16(0)

const (
	ObjFlagShared = 1 << iota
)

const (
	SymFlagDupok = 1 << iota
	SymFlagLocal
	SymFlagTypelink
	SymFlagLeaf
	SymFlagCFunc
	SymFlagReflectMethod
	SymFlagGoType
	SymFlagTopFrame
)

func (s *Sym) Write(w *Writer) {
	w.StringRef(s.Name)
	w.Uint16(s.ABI)
	w.Uint8(s.Type)
	w.Uint8(s.Flag)
	w.Uint32(s.Siz)
}

func (s *Sym) Read(r *Reader, off uint32) {
	s.Name = r.StringRef(off)
	s.ABI = r.uint16At(off + 4)
	s.Type = r.uint8At(off + 6)
	s.Flag = r.uint8At(off + 7)
	s.Siz = r.uint32At(off + 8)
}

func (s *Sym) Size() int {
	return 4 + 2 + 1 + 1 + 4
}

func (s *Sym) Dupok() bool         { return s.Flag&SymFlagDupok != 0 }
func (s *Sym) Local() bool         { return s.Flag&SymFlagLocal != 0 }
func (s *Sym) Typelink() bool      { return s.Flag&SymFlagTypelink != 0 }
func (s *Sym) Leaf() bool          { return s.Flag&SymFlagLeaf != 0 }
func (s *Sym) CFunc() bool         { return s.Flag&SymFlagCFunc != 0 }
func (s *Sym) ReflectMethod() bool { return s.Flag&SymFlagReflectMethod != 0 }
func (s *Sym) IsGoType() bool      { return s.Flag&SymFlagGoType != 0 }
func (s *Sym) TopFrame() bool      { return s.Flag&SymFlagTopFrame != 0 }

// Symbol reference.
type SymRef struct {
	PkgIdx uint32
	SymIdx uint32
}

func (s *SymRef) Write(w *Writer) {
	w.Uint32(s.PkgIdx)
	w.Uint32(s.SymIdx)
}

func (s *SymRef) Read(r *Reader, off uint32) {
	s.PkgIdx = r.uint32At(off)
	s.SymIdx = r.uint32At(off + 4)
}

func (s *SymRef) Size() int {
	return 4 + 4
}

// Relocation.
type Reloc struct {
	Off  int32
	Siz  uint8
	Type uint8
	Add  int64
	Sym  SymRef
}

func (r *Reloc) Write(w *Writer) {
	w.Uint32(uint32(r.Off))
	w.Uint8(r.Siz)
	w.Uint8(r.Type)
	w.Uint64(uint64(r.Add))
	r.Sym.Write(w)
}

func (o *Reloc) Read(r *Reader, off uint32) {
	o.Off = r.int32At(off)
	o.Siz = r.uint8At(off + 4)
	o.Type = r.uint8At(off + 5)
	o.Add = r.int64At(off + 6)
	o.Sym.Read(r, off+14)
}

func (r *Reloc) Size() int {
	return 4 + 1 + 1 + 8 + r.Sym.Size()
}

// Aux symbol info.
type Aux struct {
	Type uint8
	Sym  SymRef
}

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

func (a *Aux) Write(w *Writer) {
	w.Uint8(a.Type)
	a.Sym.Write(w)
}

func (a *Aux) Read(r *Reader, off uint32) {
	a.Type = r.uint8At(off)
	a.Sym.Read(r, off+1)
}

func (a *Aux) Size() int {
	return 1 + a.Sym.Size()
}

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
	w.Uint32(uint32(len(s)))
	w.RawString(s)
}

func (w *Writer) StringRef(s string) {
	off, ok := w.stringMap[s]
	if !ok {
		panic(fmt.Sprintf("writeStringRef: string not added: %q", s))
	}
	w.Uint32(off)
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

func (r *Reader) StringAt(off uint32) string {
	l := r.uint32At(off)
	b := r.b[off+4 : off+4+l]
	if r.readonly {
		return toString(b) // backed by RO memory, ok to make unsafe string
	}
	return string(b)
}

func toString(b []byte) string {
	type stringHeader struct {
		str unsafe.Pointer
		len int
	}

	if len(b) == 0 {
		return ""
	}
	ss := stringHeader{str: unsafe.Pointer(&b[0]), len: len(b)}
	s := *(*string)(unsafe.Pointer(&ss))
	return s
}

func (r *Reader) StringRef(off uint32) string {
	return r.StringAt(r.uint32At(off))
}

func (r *Reader) Autolib() []string {
	n := (r.h.Offsets[BlkAutolib+1] - r.h.Offsets[BlkAutolib]) / 4
	s := make([]string, n)
	for i := range s {
		off := r.h.Offsets[BlkAutolib] + uint32(i)*4
		s[i] = r.StringRef(off)
	}
	return s
}

func (r *Reader) Pkglist() []string {
	n := (r.h.Offsets[BlkPkgIdx+1] - r.h.Offsets[BlkPkgIdx]) / 4
	s := make([]string, n)
	for i := range s {
		off := r.h.Offsets[BlkPkgIdx] + uint32(i)*4
		s[i] = r.StringRef(off)
	}
	return s
}

func (r *Reader) NPkg() int {
	return int(r.h.Offsets[BlkPkgIdx+1]-r.h.Offsets[BlkPkgIdx]) / 4
}

func (r *Reader) Pkg(i int) string {
	off := r.h.Offsets[BlkPkgIdx] + uint32(i)*4
	return r.StringRef(off)
}

func (r *Reader) NDwarfFile() int {
	return int(r.h.Offsets[BlkDwarfFile+1]-r.h.Offsets[BlkDwarfFile]) / 4
}

func (r *Reader) DwarfFile(i int) string {
	off := r.h.Offsets[BlkDwarfFile] + uint32(i)*4
	return r.StringRef(off)
}

func (r *Reader) NSym() int {
	symsiz := (&Sym{}).Size()
	return int(r.h.Offsets[BlkSymdef+1]-r.h.Offsets[BlkSymdef]) / symsiz
}

func (r *Reader) NNonpkgdef() int {
	symsiz := (&Sym{}).Size()
	return int(r.h.Offsets[BlkNonpkgdef+1]-r.h.Offsets[BlkNonpkgdef]) / symsiz
}

func (r *Reader) NNonpkgref() int {
	symsiz := (&Sym{}).Size()
	return int(r.h.Offsets[BlkNonpkgref+1]-r.h.Offsets[BlkNonpkgref]) / symsiz
}

// SymOff returns the offset of the i-th symbol.
func (r *Reader) SymOff(i int) uint32 {
	symsiz := (&Sym{}).Size()
	return r.h.Offsets[BlkSymdef] + uint32(i*symsiz)
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
	relocsiz := (&Reloc{}).Size()
	return r.h.Offsets[BlkReloc] + (relocIdx+uint32(j))*uint32(relocsiz)
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
	auxsiz := (&Aux{}).Size()
	return r.h.Offsets[BlkAux] + (auxIdx+uint32(j))*uint32(auxsiz)
}

// DataOff returns the offset of the i-th symbol's data.
func (r *Reader) DataOff(i int) uint32 {
	dataIdxOff := r.h.Offsets[BlkDataIdx] + uint32(i*4)
	return r.h.Offsets[BlkData] + r.uint32At(dataIdxOff)
}

// DataSize returns the size of the i-th symbol's data.
func (r *Reader) DataSize(i int) int {
	return int(r.DataOff(i+1) - r.DataOff(i))
}

// Data returns the i-th symbol's data.
func (r *Reader) Data(i int) []byte {
	return r.BytesAt(r.DataOff(i), r.DataSize(i))
}

// AuxDataBase returns the base offset of the aux data block.
func (r *Reader) PcdataBase() uint32 {
	return r.h.Offsets[BlkPcdata]
}

// ReadOnly returns whether r.BytesAt returns read-only bytes.
func (r *Reader) ReadOnly() bool {
	return r.readonly
}

// Flags returns the flag bits read from the object file header.
func (r *Reader) Flags() uint32 {
	return r.h.Flags
}
