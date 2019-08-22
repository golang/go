// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Writing Go object files.

package obj

import (
	"bytes"
	"cmd/internal/bio"
	"cmd/internal/goobj2"
	"cmd/internal/objabi"
	"fmt"
	"strings"
)

// Entry point of writing new object file.
func WriteObjFile2(ctxt *Link, b *bio.Writer, pkgpath string) {
	genFuncInfoSyms(ctxt)

	w := writer{
		Writer:  goobj2.NewWriter(b),
		ctxt:    ctxt,
		pkgpath: objabi.PathToPrefix(pkgpath),
	}

	start := b.Offset()
	w.init()

	// Header
	// We just reserve the space. We'll fill in the offsets later.
	h := goobj2.Header{Magic: goobj2.Magic}
	h.Write(w.Writer)

	// String table
	w.StringTable()

	// Package references
	h.Offsets[goobj2.BlkPkgIdx] = w.Offset()
	for _, pkg := range w.pkglist {
		w.StringRef(pkg)
	}

	// Symbol definitions
	h.Offsets[goobj2.BlkSymdef] = w.Offset()
	for _, s := range ctxt.defs {
		w.Sym(s)
	}

	// Non-pkg symbol definitions
	h.Offsets[goobj2.BlkNonpkgdef] = w.Offset()
	for _, s := range ctxt.nonpkgdefs {
		w.Sym(s)
	}

	// Non-pkg symbol references
	h.Offsets[goobj2.BlkNonpkgref] = w.Offset()
	for _, s := range ctxt.nonpkgrefs {
		w.Sym(s)
	}

	// Reloc indexes
	h.Offsets[goobj2.BlkRelocIdx] = w.Offset()
	nreloc := uint32(0)
	lists := [][]*LSym{ctxt.defs, ctxt.nonpkgdefs}
	for _, list := range lists {
		for _, s := range list {
			w.Uint32(nreloc)
			nreloc += uint32(len(s.R))
		}
	}
	w.Uint32(nreloc)

	// Symbol Info indexes
	h.Offsets[goobj2.BlkAuxIdx] = w.Offset()
	naux := uint32(0)
	for _, list := range lists {
		for _, s := range list {
			w.Uint32(naux)
			if s.Gotype != nil {
				naux++
			}
			if s.Func != nil {
				// FuncInfo is an aux symbol, each Funcdata is an aux symbol
				naux += 1 + uint32(len(s.Func.Pcln.Funcdata))
			}
		}
	}
	w.Uint32(naux)

	// Data indexes
	h.Offsets[goobj2.BlkDataIdx] = w.Offset()
	dataOff := uint32(0)
	for _, list := range lists {
		for _, s := range list {
			w.Uint32(dataOff)
			dataOff += uint32(len(s.P))
		}
	}
	w.Uint32(dataOff)

	// Relocs
	h.Offsets[goobj2.BlkReloc] = w.Offset()
	for _, list := range lists {
		for _, s := range list {
			for i := range s.R {
				w.Reloc(&s.R[i])
			}
		}
	}

	// Aux symbol info
	h.Offsets[goobj2.BlkAux] = w.Offset()
	for _, list := range lists {
		for _, s := range list {
			w.Aux(s)
		}
	}

	// Data
	h.Offsets[goobj2.BlkData] = w.Offset()
	for _, list := range lists {
		for _, s := range list {
			w.Bytes(s.P)
		}
	}

	// Pcdata
	h.Offsets[goobj2.BlkPcdata] = w.Offset()
	for _, s := range ctxt.Text { // iteration order must match genFuncInfoSyms
		if s.Func != nil {
			pc := &s.Func.Pcln
			w.Bytes(pc.Pcsp.P)
			w.Bytes(pc.Pcfile.P)
			w.Bytes(pc.Pcline.P)
			w.Bytes(pc.Pcinline.P)
			for i := range pc.Pcdata {
				w.Bytes(pc.Pcdata[i].P)
			}
		}
	}

	// Fix up block offsets in the header
	end := start + int64(w.Offset())
	b.MustSeek(start, 0)
	h.Write(w.Writer)
	b.MustSeek(end, 0)
}

type writer struct {
	*goobj2.Writer
	ctxt    *Link
	pkgpath string   // the package import path (escaped), "" if unknown
	pkglist []string // list of packages referenced, indexed by ctxt.pkgIdx
}

// prepare package index list
func (w *writer) init() {
	w.pkglist = make([]string, len(w.ctxt.pkgIdx)+1)
	w.pkglist[0] = "" // dummy invalid package for index 0
	for pkg, i := range w.ctxt.pkgIdx {
		w.pkglist[i] = pkg
	}

	// Also make sure imported packages appear in the list (even if no symbol is referenced).
	for _, pkg := range w.ctxt.Imports {
		if _, ok := w.ctxt.pkgIdx[pkg]; !ok {
			w.pkglist = append(w.pkglist, pkg)
		}
	}
}

func (w *writer) StringTable() {
	w.AddString("")
	for _, pkg := range w.ctxt.Imports {
		w.AddString(pkg)
	}
	for _, pkg := range w.pkglist {
		w.AddString(pkg)
	}
	w.ctxt.traverseSyms(traverseAll, func(s *LSym) {
		if w.pkgpath != "" {
			s.Name = strings.Replace(s.Name, "\"\".", w.pkgpath+".", -1)
		}
		w.AddString(s.Name)
	})
	w.ctxt.traverseSyms(traverseDefs, func(s *LSym) {
		if s.Type != objabi.STEXT {
			return
		}
		pc := &s.Func.Pcln
		for _, f := range pc.File {
			w.AddString(f)
		}
		for _, call := range pc.InlTree.nodes {
			f, _ := linkgetlineFromPos(w.ctxt, call.Pos)
			w.AddString(f)
		}
	})
}

func (w *writer) Sym(s *LSym) {
	abi := uint16(s.ABI())
	if s.Static() {
		abi = goobj2.SymABIstatic
	}
	flag := uint8(0)
	if s.DuplicateOK() {
		flag |= goobj2.SymFlagDupok
	}
	if s.Local() {
		flag |= goobj2.SymFlagLocal
	}
	if s.MakeTypelink() {
		flag |= goobj2.SymFlagTypelink
	}
	o := goobj2.Sym{
		Name: s.Name,
		ABI:  abi,
		Type: uint8(s.Type),
		Flag: flag,
		Siz:  uint32(s.Size),
	}
	o.Write(w.Writer)
}

func makeSymRef(s *LSym) goobj2.SymRef {
	if s == nil {
		return goobj2.SymRef{}
	}
	if s.PkgIdx == 0 || !s.Indexed() {
		fmt.Printf("unindexed symbol reference: %v\n", s)
		panic("unindexed symbol reference")
	}
	return goobj2.SymRef{PkgIdx: uint32(s.PkgIdx), SymIdx: uint32(s.SymIdx)}
}

func (w *writer) Reloc(r *Reloc) {
	o := goobj2.Reloc{
		Off:  r.Off,
		Siz:  r.Siz,
		Type: uint8(r.Type),
		Add:  r.Add,
		Sym:  makeSymRef(r.Sym),
	}
	o.Write(w.Writer)
}

func (w *writer) Aux(s *LSym) {
	if s.Gotype != nil {
		o := goobj2.Aux{
			Type: goobj2.AuxGotype,
			Sym:  makeSymRef(s.Gotype),
		}
		o.Write(w.Writer)
	}
	if s.Func != nil {
		o := goobj2.Aux{
			Type: goobj2.AuxFuncInfo,
			Sym:  makeSymRef(s.Func.FuncInfoSym),
		}
		o.Write(w.Writer)

		for _, d := range s.Func.Pcln.Funcdata {
			o := goobj2.Aux{
				Type: goobj2.AuxFuncdata,
				Sym:  makeSymRef(d),
			}
			o.Write(w.Writer)
		}
	}
}

// generate symbols for FuncInfo.
func genFuncInfoSyms(ctxt *Link) {
	infosyms := make([]*LSym, 0, len(ctxt.Text))
	var pcdataoff uint32
	var b bytes.Buffer
	symidx := int32(len(ctxt.defs))
	for _, s := range ctxt.Text {
		if s.Func == nil {
			continue
		}
		nosplit := uint8(0)
		if s.NoSplit() {
			nosplit = 1
		}
		flags := uint8(0)
		if s.Leaf() {
			flags |= goobj2.FuncFlagLeaf
		}
		if s.CFunc() {
			flags |= goobj2.FuncFlagCFunc
		}
		if s.ReflectMethod() {
			flags |= goobj2.FuncFlagReflectMethod
		}
		if ctxt.Flag_shared { // This is really silly
			flags |= goobj2.FuncFlagShared
		}
		if s.TopFrame() {
			flags |= goobj2.FuncFlagTopFrame
		}
		o := goobj2.FuncInfo{
			NoSplit: nosplit,
			Flags:   flags,
			Args:    uint32(s.Func.Args),
			Locals:  uint32(s.Func.Locals),
		}
		pc := &s.Func.Pcln
		o.Pcsp = pcdataoff
		pcdataoff += uint32(len(pc.Pcsp.P))
		o.Pcfile = pcdataoff
		pcdataoff += uint32(len(pc.Pcfile.P))
		o.Pcline = pcdataoff
		pcdataoff += uint32(len(pc.Pcline.P))
		o.Pcinline = pcdataoff
		pcdataoff += uint32(len(pc.Pcinline.P))
		o.Pcdata = make([]uint32, len(pc.Pcdata))
		for i, pcd := range pc.Pcdata {
			o.Pcdata[i] = pcdataoff
			pcdataoff += uint32(len(pcd.P))
		}
		o.PcdataEnd = pcdataoff
		o.Funcdataoff = make([]uint32, len(pc.Funcdataoff))
		for i, x := range pc.Funcdataoff {
			o.Funcdataoff[i] = uint32(x)
		}
		o.File = make([]goobj2.SymRef, len(pc.File))
		for i, f := range pc.File {
			fsym := ctxt.Lookup(f)
			o.File[i] = makeSymRef(fsym)
		}

		o.Write(&b)
		isym := &LSym{
			Type:   objabi.SDATA, // for now, I don't think it matters
			PkgIdx: goobj2.PkgIdxSelf,
			SymIdx: symidx,
			P:      append([]byte(nil), b.Bytes()...),
		}
		isym.Set(AttrIndexed, true)
		symidx++
		infosyms = append(infosyms, isym)
		s.Func.FuncInfoSym = isym
		b.Reset()
	}
	ctxt.defs = append(ctxt.defs, infosyms...)
}
