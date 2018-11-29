// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package objfile reads Go object files for the Go linker, cmd/link.
//
// This package is similar to cmd/internal/objfile which also reads
// Go object files.
package objfile

import (
	"bufio"
	"bytes"
	"cmd/internal/bio"
	"cmd/internal/dwarf"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/sym"
	"io"
	"log"
	"strconv"
	"strings"
)

const (
	startmagic = "\x00go112ld"
	endmagic   = "\xffgo112ld"
)

var emptyPkg = []byte(`"".`)

// objReader reads Go object files.
type objReader struct {
	rd              *bufio.Reader
	arch            *sys.Arch
	syms            *sym.Symbols
	lib             *sym.Library
	pn              string
	dupSym          *sym.Symbol
	localSymVersion int

	// rdBuf is used by readString and readSymName as scratch for reading strings.
	rdBuf []byte

	// List of symbol references for the file being read.
	refs        []*sym.Symbol
	data        []byte
	reloc       []sym.Reloc
	pcdata      []sym.Pcdata
	autom       []sym.Auto
	funcdata    []*sym.Symbol
	funcdataoff []int64
	file        []*sym.Symbol
}

// Load loads an object file f into library lib.
// The symbols loaded are added to syms.
func Load(arch *sys.Arch, syms *sym.Symbols, f *bio.Reader, lib *sym.Library, length int64, pn string) {
	start := f.Offset()
	r := &objReader{
		rd:              f.Reader,
		lib:             lib,
		arch:            arch,
		syms:            syms,
		pn:              pn,
		dupSym:          &sym.Symbol{Name: ".dup"},
		localSymVersion: syms.IncVersion(),
	}
	r.loadObjFile()
	if f.Offset() != start+length {
		log.Fatalf("%s: unexpected end at %d, want %d", pn, f.Offset(), start+length)
	}
}

func (r *objReader) loadObjFile() {
	// Magic header
	var buf [8]uint8
	r.readFull(buf[:])
	if string(buf[:]) != startmagic {
		log.Fatalf("%s: invalid file start %x %x %x %x %x %x %x %x", r.pn, buf[0], buf[1], buf[2], buf[3], buf[4], buf[5], buf[6], buf[7])
	}

	// Version
	c, err := r.rd.ReadByte()
	if err != nil || c != 1 {
		log.Fatalf("%s: invalid file version number %d", r.pn, c)
	}

	// Autolib
	for {
		lib := r.readString()
		if lib == "" {
			break
		}
		r.lib.ImportStrings = append(r.lib.ImportStrings, lib)
	}

	// Symbol references
	r.refs = []*sym.Symbol{nil} // zeroth ref is nil
	for {
		c, err := r.rd.Peek(1)
		if err != nil {
			log.Fatalf("%s: peeking: %v", r.pn, err)
		}
		if c[0] == 0xff {
			r.rd.ReadByte()
			break
		}
		r.readRef()
	}

	// Lengths
	r.readSlices()

	// Data section
	r.readFull(r.data)

	// Defined symbols
	for {
		c, err := r.rd.Peek(1)
		if err != nil {
			log.Fatalf("%s: peeking: %v", r.pn, err)
		}
		if c[0] == 0xff {
			break
		}
		r.readSym()
	}

	// Magic footer
	buf = [8]uint8{}
	r.readFull(buf[:])
	if string(buf[:]) != endmagic {
		log.Fatalf("%s: invalid file end", r.pn)
	}
}

func (r *objReader) readSlices() {
	n := r.readInt()
	r.data = make([]byte, n)
	n = r.readInt()
	r.reloc = make([]sym.Reloc, n)
	n = r.readInt()
	r.pcdata = make([]sym.Pcdata, n)
	n = r.readInt()
	r.autom = make([]sym.Auto, n)
	n = r.readInt()
	r.funcdata = make([]*sym.Symbol, n)
	r.funcdataoff = make([]int64, n)
	n = r.readInt()
	r.file = make([]*sym.Symbol, n)
}

// Symbols are prefixed so their content doesn't get confused with the magic footer.
const symPrefix = 0xfe

func (r *objReader) readSym() {
	var c byte
	var err error
	if c, err = r.rd.ReadByte(); c != symPrefix || err != nil {
		log.Fatalln("readSym out of sync")
	}
	if c, err = r.rd.ReadByte(); err != nil {
		log.Fatalln("error reading input: ", err)
	}
	t := sym.AbiSymKindToSymKind[c]
	s := r.readSymIndex()
	flags := r.readInt()
	dupok := flags&1 != 0
	local := flags&2 != 0
	makeTypelink := flags&4 != 0
	size := r.readInt()
	typ := r.readSymIndex()
	data := r.readData()
	nreloc := r.readInt()
	pkg := objabi.PathToPrefix(r.lib.Pkg)
	isdup := false

	var dup *sym.Symbol
	if s.Type != 0 && s.Type != sym.SXREF {
		if (t == sym.SDATA || t == sym.SBSS || t == sym.SNOPTRBSS) && len(data) == 0 && nreloc == 0 {
			if s.Size < int64(size) {
				s.Size = int64(size)
			}
			if typ != nil && s.Gotype == nil {
				s.Gotype = typ
			}
			return
		}

		if (s.Type == sym.SDATA || s.Type == sym.SBSS || s.Type == sym.SNOPTRBSS) && len(s.P) == 0 && len(s.R) == 0 {
			goto overwrite
		}
		if s.Type != sym.SBSS && s.Type != sym.SNOPTRBSS && !dupok && !s.Attr.DuplicateOK() {
			log.Fatalf("duplicate symbol %s (types %d and %d) in %s and %s", s.Name, s.Type, t, s.File, r.pn)
		}
		if len(s.P) > 0 {
			dup = s
			s = r.dupSym
			isdup = true
		}
	}

overwrite:
	s.File = pkg
	s.Lib = r.lib
	if dupok {
		s.Attr |= sym.AttrDuplicateOK
	}
	if t == sym.SXREF {
		log.Fatalf("bad sxref")
	}
	if t == 0 {
		log.Fatalf("missing type for %s in %s", s.Name, r.pn)
	}
	if t == sym.SBSS && (s.Type == sym.SRODATA || s.Type == sym.SNOPTRBSS) {
		t = s.Type
	}
	s.Type = t
	if s.Size < int64(size) {
		s.Size = int64(size)
	}
	s.Attr.Set(sym.AttrLocal, local)
	s.Attr.Set(sym.AttrMakeTypelink, makeTypelink)
	if typ != nil {
		s.Gotype = typ
	}
	if isdup && typ != nil { // if bss sym defined multiple times, take type from any one def
		dup.Gotype = typ
	}
	s.P = data
	if nreloc > 0 {
		s.R = r.reloc[:nreloc:nreloc]
		if !isdup {
			r.reloc = r.reloc[nreloc:]
		}

		for i := 0; i < nreloc; i++ {
			s.R[i] = sym.Reloc{
				Off:  r.readInt32(),
				Siz:  r.readUint8(),
				Type: objabi.RelocType(r.readInt32()),
				Add:  r.readInt64(),
				Sym:  r.readSymIndex(),
			}
		}
	}

	if s.Type == sym.STEXT {
		s.FuncInfo = new(sym.FuncInfo)
		pc := s.FuncInfo

		pc.Args = r.readInt32()
		pc.Locals = r.readInt32()
		if r.readUint8() != 0 {
			s.Attr |= sym.AttrNoSplit
		}
		flags := r.readInt()
		if flags&(1<<2) != 0 {
			s.Attr |= sym.AttrReflectMethod
		}
		if flags&(1<<3) != 0 {
			s.Attr |= sym.AttrShared
		}
		n := r.readInt()
		pc.Autom = r.autom[:n:n]
		if !isdup {
			r.autom = r.autom[n:]
		}

		for i := 0; i < n; i++ {
			pc.Autom[i] = sym.Auto{
				Asym:    r.readSymIndex(),
				Aoffset: r.readInt32(),
				Name:    r.readInt16(),
				Gotype:  r.readSymIndex(),
			}
		}

		pc.Pcsp.P = r.readData()
		pc.Pcfile.P = r.readData()
		pc.Pcline.P = r.readData()
		pc.Pcinline.P = r.readData()
		n = r.readInt()
		pc.Pcdata = r.pcdata[:n:n]
		if !isdup {
			r.pcdata = r.pcdata[n:]
		}
		for i := 0; i < n; i++ {
			pc.Pcdata[i].P = r.readData()
		}
		n = r.readInt()
		pc.Funcdata = r.funcdata[:n:n]
		pc.Funcdataoff = r.funcdataoff[:n:n]
		if !isdup {
			r.funcdata = r.funcdata[n:]
			r.funcdataoff = r.funcdataoff[n:]
		}
		for i := 0; i < n; i++ {
			pc.Funcdata[i] = r.readSymIndex()
		}
		for i := 0; i < n; i++ {
			pc.Funcdataoff[i] = r.readInt64()
		}
		n = r.readInt()
		pc.File = r.file[:n:n]
		if !isdup {
			r.file = r.file[n:]
		}
		for i := 0; i < n; i++ {
			pc.File[i] = r.readSymIndex()
		}
		n = r.readInt()
		pc.InlTree = make([]sym.InlinedCall, n)
		for i := 0; i < n; i++ {
			pc.InlTree[i].Parent = r.readInt32()
			pc.InlTree[i].File = r.readSymIndex()
			pc.InlTree[i].Line = r.readInt32()
			pc.InlTree[i].Func = r.readSymIndex()
		}

		if !dupok {
			if s.Attr.OnList() {
				log.Fatalf("symbol %s listed multiple times", s.Name)
			}
			s.Attr |= sym.AttrOnList
			r.lib.Textp = append(r.lib.Textp, s)
		} else {
			// there may ba a dup in another package
			// put into a temp list and add to text later
			if !isdup {
				r.lib.DupTextSyms = append(r.lib.DupTextSyms, s)
			} else {
				r.lib.DupTextSyms = append(r.lib.DupTextSyms, dup)
			}
		}
	}
	if s.Type == sym.SDWARFINFO {
		r.patchDWARFName(s)
	}
}

func (r *objReader) patchDWARFName(s *sym.Symbol) {
	// This is kind of ugly. Really the package name should not
	// even be included here.
	if s.Size < 1 || s.P[0] != dwarf.DW_ABRV_FUNCTION {
		return
	}
	e := bytes.IndexByte(s.P, 0)
	if e == -1 {
		return
	}
	p := bytes.Index(s.P[:e], emptyPkg)
	if p == -1 {
		return
	}
	pkgprefix := []byte(objabi.PathToPrefix(r.lib.Pkg) + ".")
	patched := bytes.Replace(s.P[:e], emptyPkg, pkgprefix, -1)

	s.P = append(patched, s.P[e:]...)
	delta := int64(len(s.P)) - s.Size
	s.Size = int64(len(s.P))
	for i := range s.R {
		r := &s.R[i]
		if r.Off > int32(e) {
			r.Off += int32(delta)
		}
	}
}

func (r *objReader) readFull(b []byte) {
	_, err := io.ReadFull(r.rd, b)
	if err != nil {
		log.Fatalf("%s: error reading %s", r.pn, err)
	}
}

func (r *objReader) readRef() {
	if c, err := r.rd.ReadByte(); c != symPrefix || err != nil {
		log.Fatalf("readSym out of sync")
	}
	name := r.readSymName()
	var v int
	if abi := r.readInt(); abi == -1 {
		// Static
		v = r.localSymVersion
	} else if abiver := sym.ABIToVersion(obj.ABI(abi)); abiver != -1 {
		// Note that data symbols are "ABI0", which maps to version 0.
		v = abiver
	} else {
		log.Fatalf("invalid symbol ABI for %q: %d", name, abi)
	}
	s := r.syms.Lookup(name, v)
	r.refs = append(r.refs, s)

	if s == nil || v == r.localSymVersion {
		return
	}
	if s.Name[0] == '$' && len(s.Name) > 5 && s.Type == 0 && len(s.P) == 0 {
		x, err := strconv.ParseUint(s.Name[5:], 16, 64)
		if err != nil {
			log.Panicf("failed to parse $-symbol %s: %v", s.Name, err)
		}
		s.Type = sym.SRODATA
		s.Attr |= sym.AttrLocal
		switch s.Name[:5] {
		case "$f32.":
			if uint64(uint32(x)) != x {
				log.Panicf("$-symbol %s too large: %d", s.Name, x)
			}
			s.AddUint32(r.arch, uint32(x))
		case "$f64.", "$i64.":
			s.AddUint64(r.arch, x)
		default:
			log.Panicf("unrecognized $-symbol: %s", s.Name)
		}
		s.Attr.Set(sym.AttrReachable, false)
	}
	if strings.HasPrefix(s.Name, "runtime.gcbits.") {
		s.Attr |= sym.AttrLocal
	}
}

func (r *objReader) readInt64() int64 {
	uv := uint64(0)
	for shift := uint(0); ; shift += 7 {
		if shift >= 64 {
			log.Fatalf("corrupt input")
		}
		c, err := r.rd.ReadByte()
		if err != nil {
			log.Fatalln("error reading input: ", err)
		}
		uv |= uint64(c&0x7F) << shift
		if c&0x80 == 0 {
			break
		}
	}

	return int64(uv>>1) ^ (int64(uv<<63) >> 63)
}

func (r *objReader) readInt() int {
	n := r.readInt64()
	if int64(int(n)) != n {
		log.Panicf("%v out of range for int", n)
	}
	return int(n)
}

func (r *objReader) readInt32() int32 {
	n := r.readInt64()
	if int64(int32(n)) != n {
		log.Panicf("%v out of range for int32", n)
	}
	return int32(n)
}

func (r *objReader) readInt16() int16 {
	n := r.readInt64()
	if int64(int16(n)) != n {
		log.Panicf("%v out of range for int16", n)
	}
	return int16(n)
}

func (r *objReader) readUint8() uint8 {
	n := r.readInt64()
	if int64(uint8(n)) != n {
		log.Panicf("%v out of range for uint8", n)
	}
	return uint8(n)
}

func (r *objReader) readString() string {
	n := r.readInt()
	if cap(r.rdBuf) < n {
		r.rdBuf = make([]byte, 2*n)
	}
	r.readFull(r.rdBuf[:n])
	return string(r.rdBuf[:n])
}

func (r *objReader) readData() []byte {
	n := r.readInt()
	p := r.data[:n:n]
	r.data = r.data[n:]
	return p
}

// readSymName reads a symbol name, replacing all "". with pkg.
func (r *objReader) readSymName() string {
	pkg := objabi.PathToPrefix(r.lib.Pkg)
	n := r.readInt()
	if n == 0 {
		r.readInt64()
		return ""
	}
	if cap(r.rdBuf) < n {
		r.rdBuf = make([]byte, 2*n)
	}
	origName, err := r.rd.Peek(n)
	if err == bufio.ErrBufferFull {
		// Long symbol names are rare but exist. One source is type
		// symbols for types with long string forms. See #15104.
		origName = make([]byte, n)
		r.readFull(origName)
	} else if err != nil {
		log.Fatalf("%s: error reading symbol: %v", r.pn, err)
	}
	adjName := r.rdBuf[:0]
	for {
		i := bytes.Index(origName, emptyPkg)
		if i == -1 {
			s := string(append(adjName, origName...))
			// Read past the peeked origName, now that we're done with it,
			// using the rfBuf (also no longer used) as the scratch space.
			// TODO: use bufio.Reader.Discard if available instead?
			if err == nil {
				r.readFull(r.rdBuf[:n])
			}
			r.rdBuf = adjName[:0] // in case 2*n wasn't enough
			return s
		}
		adjName = append(adjName, origName[:i]...)
		adjName = append(adjName, pkg...)
		adjName = append(adjName, '.')
		origName = origName[i+len(emptyPkg):]
	}
}

// Reads the index of a symbol reference and resolves it to a symbol
func (r *objReader) readSymIndex() *sym.Symbol {
	i := r.readInt()
	return r.refs[i]
}
