// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Writing Go object files.

package obj

import (
	"bytes"
	"cmd/internal/bio"
	"cmd/internal/goobj"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"crypto/sha1"
	"encoding/binary"
	"fmt"
	"io"
	"path/filepath"
	"sort"
	"strings"
)

// Entry point of writing new object file.
func WriteObjFile(ctxt *Link, b *bio.Writer) {

	debugAsmEmit(ctxt)

	genFuncInfoSyms(ctxt)

	w := writer{
		Writer:  goobj.NewWriter(b),
		ctxt:    ctxt,
		pkgpath: objabi.PathToPrefix(ctxt.Pkgpath),
	}

	start := b.Offset()
	w.init()

	// Header
	// We just reserve the space. We'll fill in the offsets later.
	flags := uint32(0)
	if ctxt.Flag_shared {
		flags |= goobj.ObjFlagShared
	}
	if w.pkgpath == "" {
		flags |= goobj.ObjFlagNeedNameExpansion
	}
	if ctxt.IsAsm {
		flags |= goobj.ObjFlagFromAssembly
	}
	h := goobj.Header{
		Magic:       goobj.Magic,
		Fingerprint: ctxt.Fingerprint,
		Flags:       flags,
	}
	h.Write(w.Writer)

	// String table
	w.StringTable()

	// Autolib
	h.Offsets[goobj.BlkAutolib] = w.Offset()
	for i := range ctxt.Imports {
		ctxt.Imports[i].Write(w.Writer)
	}

	// Package references
	h.Offsets[goobj.BlkPkgIdx] = w.Offset()
	for _, pkg := range w.pkglist {
		w.StringRef(pkg)
	}

	// File table (for DWARF and pcln generation).
	h.Offsets[goobj.BlkFile] = w.Offset()
	for _, f := range ctxt.PosTable.FileTable() {
		w.StringRef(filepath.ToSlash(f))
	}

	// Symbol definitions
	h.Offsets[goobj.BlkSymdef] = w.Offset()
	for _, s := range ctxt.defs {
		w.Sym(s)
	}

	// Short hashed symbol definitions
	h.Offsets[goobj.BlkHashed64def] = w.Offset()
	for _, s := range ctxt.hashed64defs {
		w.Sym(s)
	}

	// Hashed symbol definitions
	h.Offsets[goobj.BlkHasheddef] = w.Offset()
	for _, s := range ctxt.hasheddefs {
		w.Sym(s)
	}

	// Non-pkg symbol definitions
	h.Offsets[goobj.BlkNonpkgdef] = w.Offset()
	for _, s := range ctxt.nonpkgdefs {
		w.Sym(s)
	}

	// Non-pkg symbol references
	h.Offsets[goobj.BlkNonpkgref] = w.Offset()
	for _, s := range ctxt.nonpkgrefs {
		w.Sym(s)
	}

	// Referenced package symbol flags
	h.Offsets[goobj.BlkRefFlags] = w.Offset()
	w.refFlags()

	// Hashes
	h.Offsets[goobj.BlkHash64] = w.Offset()
	for _, s := range ctxt.hashed64defs {
		w.Hash64(s)
	}
	h.Offsets[goobj.BlkHash] = w.Offset()
	for _, s := range ctxt.hasheddefs {
		w.Hash(s)
	}
	// TODO: hashedrefs unused/unsupported for now

	// Reloc indexes
	h.Offsets[goobj.BlkRelocIdx] = w.Offset()
	nreloc := uint32(0)
	lists := [][]*LSym{ctxt.defs, ctxt.hashed64defs, ctxt.hasheddefs, ctxt.nonpkgdefs}
	for _, list := range lists {
		for _, s := range list {
			w.Uint32(nreloc)
			nreloc += uint32(len(s.R))
		}
	}
	w.Uint32(nreloc)

	// Symbol Info indexes
	h.Offsets[goobj.BlkAuxIdx] = w.Offset()
	naux := uint32(0)
	for _, list := range lists {
		for _, s := range list {
			w.Uint32(naux)
			naux += uint32(nAuxSym(s))
		}
	}
	w.Uint32(naux)

	// Data indexes
	h.Offsets[goobj.BlkDataIdx] = w.Offset()
	dataOff := uint32(0)
	for _, list := range lists {
		for _, s := range list {
			w.Uint32(dataOff)
			dataOff += uint32(len(s.P))
		}
	}
	w.Uint32(dataOff)

	// Relocs
	h.Offsets[goobj.BlkReloc] = w.Offset()
	for _, list := range lists {
		for _, s := range list {
			for i := range s.R {
				w.Reloc(&s.R[i])
			}
		}
	}

	// Aux symbol info
	h.Offsets[goobj.BlkAux] = w.Offset()
	for _, list := range lists {
		for _, s := range list {
			w.Aux(s)
		}
	}

	// Data
	h.Offsets[goobj.BlkData] = w.Offset()
	for _, list := range lists {
		for _, s := range list {
			w.Bytes(s.P)
		}
	}

	// Pcdata
	h.Offsets[goobj.BlkPcdata] = w.Offset()
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

	// Blocks used only by tools (objdump, nm).

	// Referenced symbol names from other packages
	h.Offsets[goobj.BlkRefName] = w.Offset()
	w.refNames()

	h.Offsets[goobj.BlkEnd] = w.Offset()

	// Fix up block offsets in the header
	end := start + int64(w.Offset())
	b.MustSeek(start, 0)
	h.Write(w.Writer)
	b.MustSeek(end, 0)
}

type writer struct {
	*goobj.Writer
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
}

func (w *writer) StringTable() {
	w.AddString("")
	for _, p := range w.ctxt.Imports {
		w.AddString(p.Pkg)
	}
	for _, pkg := range w.pkglist {
		w.AddString(pkg)
	}
	w.ctxt.traverseSyms(traverseAll, func(s *LSym) {
		// TODO: this includes references of indexed symbols from other packages,
		// for which the linker doesn't need the name. Consider moving them to
		// a separate block (for tools only).
		if w.pkgpath != "" {
			s.Name = strings.Replace(s.Name, "\"\".", w.pkgpath+".", -1)
		}
		// Don't put names of builtins into the string table (to save
		// space).
		if s.PkgIdx == goobj.PkgIdxBuiltin {
			return
		}
		w.AddString(s.Name)
	})

	// All filenames are in the postable.
	for _, f := range w.ctxt.PosTable.FileTable() {
		w.AddString(filepath.ToSlash(f))
	}
}

func (w *writer) Sym(s *LSym) {
	abi := uint16(s.ABI())
	if s.Static() {
		abi = goobj.SymABIstatic
	}
	flag := uint8(0)
	if s.DuplicateOK() {
		flag |= goobj.SymFlagDupok
	}
	if s.Local() {
		flag |= goobj.SymFlagLocal
	}
	if s.MakeTypelink() {
		flag |= goobj.SymFlagTypelink
	}
	if s.Leaf() {
		flag |= goobj.SymFlagLeaf
	}
	if s.NoSplit() {
		flag |= goobj.SymFlagNoSplit
	}
	if s.ReflectMethod() {
		flag |= goobj.SymFlagReflectMethod
	}
	if s.TopFrame() {
		flag |= goobj.SymFlagTopFrame
	}
	if strings.HasPrefix(s.Name, "type.") && s.Name[5] != '.' && s.Type == objabi.SRODATA {
		flag |= goobj.SymFlagGoType
	}
	flag2 := uint8(0)
	if s.UsedInIface() {
		flag2 |= goobj.SymFlagUsedInIface
	}
	if strings.HasPrefix(s.Name, "go.itab.") && s.Type == objabi.SRODATA {
		flag2 |= goobj.SymFlagItab
	}
	name := s.Name
	if strings.HasPrefix(name, "gofile..") {
		name = filepath.ToSlash(name)
	}
	var align uint32
	if s.Func != nil {
		align = uint32(s.Func.Align)
	}
	if s.ContentAddressable() {
		// We generally assume data symbols are natually aligned,
		// except for strings. If we dedup a string symbol and a
		// non-string symbol with the same content, we should keep
		// the largest alignment.
		// TODO: maybe the compiler could set the alignment for all
		// data symbols more carefully.
		if s.Size != 0 && !strings.HasPrefix(s.Name, "go.string.") {
			switch {
			case w.ctxt.Arch.PtrSize == 8 && s.Size%8 == 0:
				align = 8
			case s.Size%4 == 0:
				align = 4
			case s.Size%2 == 0:
				align = 2
			}
			// don't bother setting align to 1.
		}
	}
	var o goobj.Sym
	o.SetName(name, w.Writer)
	o.SetABI(abi)
	o.SetType(uint8(s.Type))
	o.SetFlag(flag)
	o.SetFlag2(flag2)
	o.SetSiz(uint32(s.Size))
	o.SetAlign(align)
	o.Write(w.Writer)
}

func (w *writer) Hash64(s *LSym) {
	if !s.ContentAddressable() || len(s.R) != 0 {
		panic("Hash of non-content-addresable symbol")
	}
	b := contentHash64(s)
	w.Bytes(b[:])
}

func (w *writer) Hash(s *LSym) {
	if !s.ContentAddressable() {
		panic("Hash of non-content-addresable symbol")
	}
	b := w.contentHash(s)
	w.Bytes(b[:])
}

func contentHash64(s *LSym) goobj.Hash64Type {
	var b goobj.Hash64Type
	copy(b[:], s.P)
	return b
}

// Compute the content hash for a content-addressable symbol.
// We build a content hash based on its content and relocations.
// Depending on the category of the referenced symbol, we choose
// different hash algorithms such that the hash is globally
// consistent.
// - For referenced content-addressable symbol, its content hash
//   is globally consistent.
// - For package symbol and builtin symbol, its local index is
//   globally consistent.
// - For non-package symbol, its fully-expanded name is globally
//   consistent. For now, we require we know the current package
//   path so we can always expand symbol names. (Otherwise,
//   symbols with relocations are not considered hashable.)
//
// For now, we assume there is no circular dependencies among
// hashed symbols.
func (w *writer) contentHash(s *LSym) goobj.HashType {
	h := sha1.New()
	// The compiler trims trailing zeros _sometimes_. We just do
	// it always.
	h.Write(bytes.TrimRight(s.P, "\x00"))
	var tmp [14]byte
	for i := range s.R {
		r := &s.R[i]
		binary.LittleEndian.PutUint32(tmp[:4], uint32(r.Off))
		tmp[4] = r.Siz
		tmp[5] = uint8(r.Type)
		binary.LittleEndian.PutUint64(tmp[6:14], uint64(r.Add))
		h.Write(tmp[:])
		rs := r.Sym
		switch rs.PkgIdx {
		case goobj.PkgIdxHashed64:
			h.Write([]byte{0})
			t := contentHash64(rs)
			h.Write(t[:])
		case goobj.PkgIdxHashed:
			h.Write([]byte{1})
			t := w.contentHash(rs)
			h.Write(t[:])
		case goobj.PkgIdxNone:
			h.Write([]byte{2})
			io.WriteString(h, rs.Name) // name is already expanded at this point
		case goobj.PkgIdxBuiltin:
			h.Write([]byte{3})
			binary.LittleEndian.PutUint32(tmp[:4], uint32(rs.SymIdx))
			h.Write(tmp[:4])
		case goobj.PkgIdxSelf:
			io.WriteString(h, w.pkgpath)
			binary.LittleEndian.PutUint32(tmp[:4], uint32(rs.SymIdx))
			h.Write(tmp[:4])
		default:
			io.WriteString(h, rs.Pkg)
			binary.LittleEndian.PutUint32(tmp[:4], uint32(rs.SymIdx))
			h.Write(tmp[:4])
		}
	}
	var b goobj.HashType
	copy(b[:], h.Sum(nil))
	return b
}

func makeSymRef(s *LSym) goobj.SymRef {
	if s == nil {
		return goobj.SymRef{}
	}
	if s.PkgIdx == 0 || !s.Indexed() {
		fmt.Printf("unindexed symbol reference: %v\n", s)
		panic("unindexed symbol reference")
	}
	return goobj.SymRef{PkgIdx: uint32(s.PkgIdx), SymIdx: uint32(s.SymIdx)}
}

func (w *writer) Reloc(r *Reloc) {
	var o goobj.Reloc
	o.SetOff(r.Off)
	o.SetSiz(r.Siz)
	o.SetType(uint8(r.Type))
	o.SetAdd(r.Add)
	o.SetSym(makeSymRef(r.Sym))
	o.Write(w.Writer)
}

func (w *writer) aux1(typ uint8, rs *LSym) {
	var o goobj.Aux
	o.SetType(typ)
	o.SetSym(makeSymRef(rs))
	o.Write(w.Writer)
}

func (w *writer) Aux(s *LSym) {
	if s.Gotype != nil {
		w.aux1(goobj.AuxGotype, s.Gotype)
	}
	if s.Func != nil {
		w.aux1(goobj.AuxFuncInfo, s.Func.FuncInfoSym)

		for _, d := range s.Func.Pcln.Funcdata {
			w.aux1(goobj.AuxFuncdata, d)
		}

		if s.Func.dwarfInfoSym != nil && s.Func.dwarfInfoSym.Size != 0 {
			w.aux1(goobj.AuxDwarfInfo, s.Func.dwarfInfoSym)
		}
		if s.Func.dwarfLocSym != nil && s.Func.dwarfLocSym.Size != 0 {
			w.aux1(goobj.AuxDwarfLoc, s.Func.dwarfLocSym)
		}
		if s.Func.dwarfRangesSym != nil && s.Func.dwarfRangesSym.Size != 0 {
			w.aux1(goobj.AuxDwarfRanges, s.Func.dwarfRangesSym)
		}
		if s.Func.dwarfDebugLinesSym != nil && s.Func.dwarfDebugLinesSym.Size != 0 {
			w.aux1(goobj.AuxDwarfLines, s.Func.dwarfDebugLinesSym)
		}
	}
}

// Emits flags of referenced indexed symbols.
func (w *writer) refFlags() {
	seen := make(map[*LSym]bool)
	w.ctxt.traverseSyms(traverseRefs, func(rs *LSym) { // only traverse refs, not auxs, as tools don't need auxs
		switch rs.PkgIdx {
		case goobj.PkgIdxNone, goobj.PkgIdxHashed64, goobj.PkgIdxHashed, goobj.PkgIdxBuiltin, goobj.PkgIdxSelf: // not an external indexed reference
			return
		case goobj.PkgIdxInvalid:
			panic("unindexed symbol reference")
		}
		if seen[rs] {
			return
		}
		seen[rs] = true
		symref := makeSymRef(rs)
		flag2 := uint8(0)
		if rs.UsedInIface() {
			flag2 |= goobj.SymFlagUsedInIface
		}
		if flag2 == 0 {
			return // no need to write zero flags
		}
		var o goobj.RefFlags
		o.SetSym(symref)
		o.SetFlag2(flag2)
		o.Write(w.Writer)
	})
}

// Emits names of referenced indexed symbols, used by tools (objdump, nm)
// only.
func (w *writer) refNames() {
	seen := make(map[*LSym]bool)
	w.ctxt.traverseSyms(traverseRefs, func(rs *LSym) { // only traverse refs, not auxs, as tools don't need auxs
		switch rs.PkgIdx {
		case goobj.PkgIdxNone, goobj.PkgIdxHashed64, goobj.PkgIdxHashed, goobj.PkgIdxBuiltin, goobj.PkgIdxSelf: // not an external indexed reference
			return
		case goobj.PkgIdxInvalid:
			panic("unindexed symbol reference")
		}
		if seen[rs] {
			return
		}
		seen[rs] = true
		symref := makeSymRef(rs)
		var o goobj.RefName
		o.SetSym(symref)
		o.SetName(rs.Name, w.Writer)
		o.Write(w.Writer)
	})
	// TODO: output in sorted order?
	// Currently tools (cmd/internal/goobj package) doesn't use mmap,
	// and it just read it into a map in memory upfront. If it uses
	// mmap, if the output is sorted, it probably could avoid reading
	// into memory and just do lookups in the mmap'd object file.
}

// return the number of aux symbols s have.
func nAuxSym(s *LSym) int {
	n := 0
	if s.Gotype != nil {
		n++
	}
	if s.Func != nil {
		// FuncInfo is an aux symbol, each Funcdata is an aux symbol
		n += 1 + len(s.Func.Pcln.Funcdata)
		if s.Func.dwarfInfoSym != nil && s.Func.dwarfInfoSym.Size != 0 {
			n++
		}
		if s.Func.dwarfLocSym != nil && s.Func.dwarfLocSym.Size != 0 {
			n++
		}
		if s.Func.dwarfRangesSym != nil && s.Func.dwarfRangesSym.Size != 0 {
			n++
		}
		if s.Func.dwarfDebugLinesSym != nil && s.Func.dwarfDebugLinesSym.Size != 0 {
			n++
		}
	}
	return n
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
		o := goobj.FuncInfo{
			Args:   uint32(s.Func.Args),
			Locals: uint32(s.Func.Locals),
			FuncID: objabi.FuncID(s.Func.FuncID),
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
		i := 0
		o.File = make([]goobj.CUFileIndex, len(pc.UsedFiles))
		for f := range pc.UsedFiles {
			o.File[i] = f
			i++
		}
		sort.Slice(o.File, func(i, j int) bool { return o.File[i] < o.File[j] })
		o.InlTree = make([]goobj.InlTreeNode, len(pc.InlTree.nodes))
		for i, inl := range pc.InlTree.nodes {
			f, l := getFileIndexAndLine(ctxt, inl.Pos)
			o.InlTree[i] = goobj.InlTreeNode{
				Parent:   int32(inl.Parent),
				File:     goobj.CUFileIndex(f),
				Line:     l,
				Func:     makeSymRef(inl.Func),
				ParentPC: inl.ParentPC,
			}
		}

		o.Write(&b)
		isym := &LSym{
			Type:   objabi.SDATA, // for now, I don't think it matters
			PkgIdx: goobj.PkgIdxSelf,
			SymIdx: symidx,
			P:      append([]byte(nil), b.Bytes()...),
		}
		isym.Set(AttrIndexed, true)
		symidx++
		infosyms = append(infosyms, isym)
		s.Func.FuncInfoSym = isym
		b.Reset()

		dwsyms := []*LSym{s.Func.dwarfRangesSym, s.Func.dwarfLocSym, s.Func.dwarfDebugLinesSym, s.Func.dwarfInfoSym}
		for _, s := range dwsyms {
			if s == nil || s.Size == 0 {
				continue
			}
			s.PkgIdx = goobj.PkgIdxSelf
			s.SymIdx = symidx
			s.Set(AttrIndexed, true)
			symidx++
			infosyms = append(infosyms, s)
		}
	}
	ctxt.defs = append(ctxt.defs, infosyms...)
}

// debugDumpAux is a dumper for selected aux symbols.
func writeAuxSymDebug(ctxt *Link, par *LSym, aux *LSym) {
	// Most aux symbols (ex: funcdata) are not interesting--
	// pick out just the DWARF ones for now.
	if aux.Type != objabi.SDWARFLOC &&
		aux.Type != objabi.SDWARFFCN &&
		aux.Type != objabi.SDWARFABSFCN &&
		aux.Type != objabi.SDWARFLINES &&
		aux.Type != objabi.SDWARFRANGE {
		return
	}
	ctxt.writeSymDebugNamed(aux, "aux for "+par.Name)
}

func debugAsmEmit(ctxt *Link) {
	if ctxt.Debugasm > 0 {
		ctxt.traverseSyms(traverseDefs, ctxt.writeSymDebug)
		if ctxt.Debugasm > 1 {
			fn := func(par *LSym, aux *LSym) {
				writeAuxSymDebug(ctxt, par, aux)
			}
			ctxt.traverseAuxSyms(traverseAux, fn)
		}
	}
}

func (ctxt *Link) writeSymDebug(s *LSym) {
	ctxt.writeSymDebugNamed(s, s.Name)
}

func (ctxt *Link) writeSymDebugNamed(s *LSym, name string) {
	ver := ""
	if ctxt.Debugasm > 1 {
		ver = fmt.Sprintf("<%d>", s.ABI())
	}
	fmt.Fprintf(ctxt.Bso, "%s%s ", name, ver)
	if s.Type != 0 {
		fmt.Fprintf(ctxt.Bso, "%v ", s.Type)
	}
	if s.Static() {
		fmt.Fprint(ctxt.Bso, "static ")
	}
	if s.DuplicateOK() {
		fmt.Fprintf(ctxt.Bso, "dupok ")
	}
	if s.CFunc() {
		fmt.Fprintf(ctxt.Bso, "cfunc ")
	}
	if s.NoSplit() {
		fmt.Fprintf(ctxt.Bso, "nosplit ")
	}
	if s.TopFrame() {
		fmt.Fprintf(ctxt.Bso, "topframe ")
	}
	fmt.Fprintf(ctxt.Bso, "size=%d", s.Size)
	if s.Type == objabi.STEXT {
		fmt.Fprintf(ctxt.Bso, " args=%#x locals=%#x funcid=%#x", uint64(s.Func.Args), uint64(s.Func.Locals), uint64(s.Func.FuncID))
		if s.Leaf() {
			fmt.Fprintf(ctxt.Bso, " leaf")
		}
	}
	fmt.Fprintf(ctxt.Bso, "\n")
	if s.Type == objabi.STEXT {
		for p := s.Func.Text; p != nil; p = p.Link {
			fmt.Fprintf(ctxt.Bso, "\t%#04x ", uint(int(p.Pc)))
			if ctxt.Debugasm > 1 {
				io.WriteString(ctxt.Bso, p.String())
			} else {
				p.InnermostString(ctxt.Bso)
			}
			fmt.Fprintln(ctxt.Bso)
		}
	}
	for i := 0; i < len(s.P); i += 16 {
		fmt.Fprintf(ctxt.Bso, "\t%#04x", uint(i))
		j := i
		for ; j < i+16 && j < len(s.P); j++ {
			fmt.Fprintf(ctxt.Bso, " %02x", s.P[j])
		}
		for ; j < i+16; j++ {
			fmt.Fprintf(ctxt.Bso, "   ")
		}
		fmt.Fprintf(ctxt.Bso, "  ")
		for j = i; j < i+16 && j < len(s.P); j++ {
			c := int(s.P[j])
			b := byte('.')
			if ' ' <= c && c <= 0x7e {
				b = byte(c)
			}
			ctxt.Bso.WriteByte(b)
		}

		fmt.Fprintf(ctxt.Bso, "\n")
	}

	sort.Sort(relocByOff(s.R)) // generate stable output
	for _, r := range s.R {
		name := ""
		ver := ""
		if r.Sym != nil {
			name = r.Sym.Name
			if ctxt.Debugasm > 1 {
				ver = fmt.Sprintf("<%d>", s.ABI())
			}
		} else if r.Type == objabi.R_TLS_LE {
			name = "TLS"
		}
		if ctxt.Arch.InFamily(sys.ARM, sys.PPC64) {
			fmt.Fprintf(ctxt.Bso, "\trel %d+%d t=%d %s%s+%x\n", int(r.Off), r.Siz, r.Type, name, ver, uint64(r.Add))
		} else {
			fmt.Fprintf(ctxt.Bso, "\trel %d+%d t=%d %s%s+%d\n", int(r.Off), r.Siz, r.Type, name, ver, r.Add)
		}
	}
}

// relocByOff sorts relocations by their offsets.
type relocByOff []Reloc

func (x relocByOff) Len() int           { return len(x) }
func (x relocByOff) Less(i, j int) bool { return x[i].Off < x[j].Off }
func (x relocByOff) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
