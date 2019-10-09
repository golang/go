// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package objfile

import (
	"bytes"
	"cmd/internal/bio"
	"cmd/internal/dwarf"
	"cmd/internal/goobj2"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/sym"
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
)

var _ = fmt.Print

// Sym encapsulates a global symbol index, used to identify a specific
// Go symbol. The 0-valued Sym is corresponds to an invalid symbol.
type Sym int

// Relocs encapsulates the set of relocations on a given symbol; an
// instance of this type is returned by the Loader Relocs() method.
type Relocs struct {
	Count int // number of relocs

	li int      // local index of symbol whose relocs we're examining
	r  *oReader // object reader for containing package
	l  *Loader  // loader
}

// Reloc contains the payload for a specific relocation.
// TODO: replace this with sym.Reloc, once we change the
// relocation target from "*sym.Symbol" to "loader.Sym" in sym.Reloc.
type Reloc struct {
	Off  int32            // offset to rewrite
	Size uint8            // number of bytes to rewrite: 0, 1, 2, or 4
	Type objabi.RelocType // the relocation type
	Add  int64            // addend
	Sym  Sym              // global index of symbol the reloc addresses
}

// oReader is a wrapper type of obj.Reader, along with some
// extra information.
// TODO: rename to objReader once the old one is gone?
type oReader struct {
	*goobj2.Reader
	unit      *sym.CompilationUnit
	version   int // version of static symbol
	pkgprefix string
}

type objIdx struct {
	r *oReader
	i Sym // start index
}

type nameVer struct {
	name string
	v    int
}

type bitmap []uint32

// set the i-th bit.
func (bm bitmap) Set(i Sym) {
	n, r := uint(i)/32, uint(i)%32
	bm[n] |= 1 << r
}

// whether the i-th bit is set.
func (bm bitmap) Has(i Sym) bool {
	n, r := uint(i)/32, uint(i)%32
	return bm[n]&(1<<r) != 0
}

func makeBitmap(n int) bitmap {
	return make(bitmap, (n+31)/32)
}

// A Loader loads new object files and resolves indexed symbol references.
//
// TODO: describe local-global index mapping.
type Loader struct {
	start    map[*oReader]Sym // map from object file to its start index
	objs     []objIdx         // sorted by start index (i.e. objIdx.i)
	max      Sym              // current max index
	extStart Sym              // from this index on, the symbols are externally defined

	symsByName map[nameVer]Sym // map symbol name to index

	objByPkg map[string]*oReader // map package path to its Go object reader

	Syms []*sym.Symbol // indexed symbols. XXX we still make sym.Symbol for now.

	Reachable bitmap // bitmap of reachable symbols, indexed by global index
}

func NewLoader() *Loader {
	return &Loader{
		start:      make(map[*oReader]Sym),
		objs:       []objIdx{{nil, 0}},
		symsByName: make(map[nameVer]Sym),
		objByPkg:   make(map[string]*oReader),
		Syms:       []*sym.Symbol{nil},
	}
}

// Return the start index in the global index space for a given object file.
func (l *Loader) StartIndex(r *oReader) Sym {
	return l.start[r]
}

// Add object file r, return the start index.
func (l *Loader) AddObj(pkg string, r *oReader) Sym {
	if _, ok := l.start[r]; ok {
		panic("already added")
	}
	pkg = objabi.PathToPrefix(pkg) // the object file contains escaped package path
	if _, ok := l.objByPkg[pkg]; !ok {
		l.objByPkg[pkg] = r
	}
	n := r.NSym() + r.NNonpkgdef()
	i := l.max + 1
	l.start[r] = i
	l.objs = append(l.objs, objIdx{r, i})
	l.max += Sym(n)
	return i
}

// Add a symbol with a given index, return if it is added.
func (l *Loader) AddSym(name string, ver int, i Sym, dupok bool) bool {
	if l.extStart != 0 {
		panic("AddSym called after AddExtSym is called")
	}
	nv := nameVer{name, ver}
	if _, ok := l.symsByName[nv]; ok {
		if dupok || true { // TODO: "true" isn't quite right. need to implement "overwrite" logic.
			return false
		}
		panic("duplicated definition of symbol " + name)
	}
	l.symsByName[nv] = i
	return true
}

// Add an external symbol (without index). Return the index of newly added
// symbol, or 0 if not added.
func (l *Loader) AddExtSym(name string, ver int) Sym {
	nv := nameVer{name, ver}
	if _, ok := l.symsByName[nv]; ok {
		return 0
	}
	i := l.max + 1
	l.symsByName[nv] = i
	l.max++
	if l.extStart == 0 {
		l.extStart = i
	}
	return i
}

// Convert a local index to a global index.
func (l *Loader) ToGlobal(r *oReader, i int) Sym {
	return l.StartIndex(r) + Sym(i)
}

// Convert a global index to a local index.
func (l *Loader) ToLocal(i Sym) (*oReader, int) {
	if l.extStart != 0 && i >= l.extStart {
		return nil, int(i - l.extStart)
	}
	// Search for the local object holding index i.
	// Below k is the first one that has its start index > i,
	// so k-1 is the one we want.
	k := sort.Search(len(l.objs), func(k int) bool {
		return l.objs[k].i > i
	})
	return l.objs[k-1].r, int(i - l.objs[k-1].i)
}

// Resolve a local symbol reference. Return global index.
func (l *Loader) Resolve(r *oReader, s goobj2.SymRef) Sym {
	var rr *oReader
	switch p := s.PkgIdx; p {
	case goobj2.PkgIdxInvalid:
		if s.SymIdx != 0 {
			panic("bad sym ref")
		}
		return 0
	case goobj2.PkgIdxNone:
		// Resolve by name
		i := int(s.SymIdx) + r.NSym()
		osym := goobj2.Sym{}
		osym.Read(r.Reader, r.SymOff(i))
		name := strings.Replace(osym.Name, "\"\".", r.pkgprefix, -1)
		v := abiToVer(osym.ABI, r.version)
		nv := nameVer{name, v}
		return l.symsByName[nv]
	case goobj2.PkgIdxBuiltin:
		panic("PkgIdxBuiltin not used")
	case goobj2.PkgIdxSelf:
		rr = r
	default:
		pkg := r.Pkg(int(p))
		var ok bool
		rr, ok = l.objByPkg[pkg]
		if !ok {
			log.Fatalf("reference of nonexisted package %s, from %v", pkg, r.unit.Lib)
		}
	}
	return l.ToGlobal(rr, int(s.SymIdx))
}

// Look up a symbol by name, return global index, or 0 if not found.
// This is more like Syms.ROLookup than Lookup -- it doesn't create
// new symbol.
func (l *Loader) Lookup(name string, ver int) Sym {
	nv := nameVer{name, ver}
	return l.symsByName[nv]
}

// Number of total symbols.
func (l *Loader) NSym() int {
	return int(l.max + 1)
}

// Returns the raw (unpatched) name of the i-th symbol.
func (l *Loader) RawSymName(i Sym) string {
	if l.extStart != 0 && i >= l.extStart {
		return ""
	}
	r, li := l.ToLocal(i)
	osym := goobj2.Sym{}
	osym.Read(r.Reader, r.SymOff(li))
	return osym.Name
}

// Returns the (patched) name of the i-th symbol.
func (l *Loader) SymName(i Sym) string {
	if l.extStart != 0 && i >= l.extStart {
		return ""
	}
	r, li := l.ToLocal(i)
	osym := goobj2.Sym{}
	osym.Read(r.Reader, r.SymOff(li))
	return strings.Replace(osym.Name, "\"\".", r.pkgprefix, -1)
}

// Returns the type of the i-th symbol.
func (l *Loader) SymType(i Sym) sym.SymKind {
	if l.extStart != 0 && i >= l.extStart {
		return 0
	}
	r, li := l.ToLocal(i)
	osym := goobj2.Sym{}
	osym.Read(r.Reader, r.SymOff(li))
	return sym.AbiSymKindToSymKind[objabi.SymKind(osym.Type)]
}

// Returns the attributes of the i-th symbol.
func (l *Loader) SymAttr(i Sym) uint8 {
	if l.extStart != 0 && i >= l.extStart {
		return 0
	}
	r, li := l.ToLocal(i)
	osym := goobj2.Sym{}
	osym.Read(r.Reader, r.SymOff(li))
	return osym.Flag
}

// Returns whether the i-th symbol has ReflectMethod attribute set.
func (l *Loader) IsReflectMethod(i Sym) bool {
	return l.SymAttr(i)&goobj2.SymFlagReflectMethod != 0
}

// Returns the symbol content of the i-th symbol. i is global index.
func (l *Loader) Data(i Sym) []byte {
	if l.extStart != 0 && i >= l.extStart {
		return nil
	}
	r, li := l.ToLocal(i)
	return r.Data(li)
}

// Returns the number of aux symbols given a global index.
func (l *Loader) NAux(i Sym) int {
	if l.extStart != 0 && i >= l.extStart {
		return 0
	}
	r, li := l.ToLocal(i)
	return r.NAux(li)
}

// Returns the referred symbol of the j-th aux symbol of the i-th
// symbol.
func (l *Loader) AuxSym(i Sym, j int) Sym {
	if l.extStart != 0 && i >= l.extStart {
		return 0
	}
	r, li := l.ToLocal(i)
	a := goobj2.Aux{}
	a.Read(r.Reader, r.AuxOff(li, j))
	return l.Resolve(r, a.Sym)
}

// Initialize Reachable bitmap for running deadcode pass.
func (l *Loader) InitReachable() {
	l.Reachable = makeBitmap(l.NSym())
}

// At method returns the j-th reloc for a global symbol.
func (relocs *Relocs) At(j int) Reloc {
	rel := goobj2.Reloc{}
	rel.Read(relocs.r.Reader, relocs.r.RelocOff(relocs.li, j))
	target := relocs.l.Resolve(relocs.r, rel.Sym)
	return Reloc{
		Off:  rel.Off,
		Size: rel.Siz,
		Type: objabi.RelocType(rel.Type),
		Add:  rel.Add,
		Sym:  target,
	}
}

// Relocs returns a Relocs object for the given global sym.
func (l *Loader) Relocs(i Sym) Relocs {
	if l.extStart != 0 && i >= l.extStart {
		return Relocs{}
	}
	r, li := l.ToLocal(i)
	return l.relocs(r, li)
}

// Relocs returns a Relocs object given a local sym index and reader.
func (l *Loader) relocs(r *oReader, li int) Relocs {
	return Relocs{
		Count: r.NReloc(li),
		li:    li,
		r:     r,
		l:     l,
	}
}

// Preload a package: add autolibs, add symbols to the symbol table.
// Does not read symbol data yet.
func LoadNew(l *Loader, arch *sys.Arch, syms *sym.Symbols, f *bio.Reader, lib *sym.Library, unit *sym.CompilationUnit, length int64, pn string, flags int) {
	roObject, readonly, err := f.Slice(uint64(length))
	if err != nil {
		log.Fatal("cannot read object file:", err)
	}
	r := goobj2.NewReaderFromBytes(roObject, readonly)
	if r == nil {
		panic("cannot read object file")
	}
	localSymVersion := syms.IncVersion()
	pkgprefix := objabi.PathToPrefix(lib.Pkg) + "."
	or := &oReader{r, unit, localSymVersion, pkgprefix}

	// Autolib
	lib.ImportStrings = append(lib.ImportStrings, r.Autolib()...)

	// DWARF file table
	nfile := r.NDwarfFile()
	unit.DWARFFileTable = make([]string, nfile)
	for i := range unit.DWARFFileTable {
		unit.DWARFFileTable[i] = r.DwarfFile(i)
	}

	istart := l.AddObj(lib.Pkg, or)

	ndef := r.NSym()
	nnonpkgdef := r.NNonpkgdef()

	// XXX add all symbols for now
	l.Syms = append(l.Syms, make([]*sym.Symbol, ndef+nnonpkgdef)...)
	for i, n := 0, ndef+nnonpkgdef; i < n; i++ {
		osym := goobj2.Sym{}
		osym.Read(r, r.SymOff(i))
		name := strings.Replace(osym.Name, "\"\".", pkgprefix, -1)
		if name == "" {
			continue // don't add unnamed aux symbol
		}
		v := abiToVer(osym.ABI, localSymVersion)
		dupok := osym.Flag&goobj2.SymFlagDupok != 0
		if l.AddSym(name, v, istart+Sym(i), dupok) {
			s := syms.Newsym(name, v)
			preprocess(arch, s) // TODO: put this at a better place
			l.Syms[istart+Sym(i)] = s
		}
	}

	// The caller expects us consuming all the data
	f.MustSeek(length, os.SEEK_CUR)
}

// Make sure referenced symbols are added. Most of them should already be added.
// This should only be needed for referenced external symbols.
func LoadRefs(l *Loader, arch *sys.Arch, syms *sym.Symbols) {
	for _, o := range l.objs[1:] {
		loadObjRefs(l, o.r, arch, syms)
	}
}

func loadObjRefs(l *Loader, r *oReader, arch *sys.Arch, syms *sym.Symbols) {
	lib := r.unit.Lib
	pkgprefix := objabi.PathToPrefix(lib.Pkg) + "."
	ndef := r.NSym() + r.NNonpkgdef()
	for i, n := 0, r.NNonpkgref(); i < n; i++ {
		osym := goobj2.Sym{}
		osym.Read(r.Reader, r.SymOff(ndef+i))
		name := strings.Replace(osym.Name, "\"\".", pkgprefix, -1)
		v := abiToVer(osym.ABI, r.version)
		if ii := l.AddExtSym(name, v); ii != 0 {
			s := syms.Newsym(name, v)
			preprocess(arch, s) // TODO: put this at a better place
			if ii != Sym(len(l.Syms)) {
				panic("AddExtSym returned bad index")
			}
			l.Syms = append(l.Syms, s)
		}
	}
}

func abiToVer(abi uint16, localSymVersion int) int {
	var v int
	if abi == goobj2.SymABIstatic {
		// Static
		v = localSymVersion
	} else if abiver := sym.ABIToVersion(obj.ABI(abi)); abiver != -1 {
		// Note that data symbols are "ABI0", which maps to version 0.
		v = abiver
	} else {
		log.Fatalf("invalid symbol ABI: %d", abi)
	}
	return v
}

func preprocess(arch *sys.Arch, s *sym.Symbol) {
	if s.Name != "" && s.Name[0] == '$' && len(s.Name) > 5 && s.Type == 0 && len(s.P) == 0 {
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
			s.AddUint32(arch, uint32(x))
		case "$f64.", "$i64.":
			s.AddUint64(arch, x)
		default:
			log.Panicf("unrecognized $-symbol: %s", s.Name)
		}
		s.Attr.Set(sym.AttrReachable, false)
	}
}

// Load relocations for building the dependency graph in deadcode pass.
// For now, we load symbol types, relocations, gotype, and the contents
// of type symbols, which are needed in deadcode.
func LoadReloc(l *Loader) {
	for _, o := range l.objs[1:] {
		loadObjReloc(l, o.r)
	}
}

func loadObjReloc(l *Loader, r *oReader) {
	lib := r.unit.Lib
	pkgprefix := objabi.PathToPrefix(lib.Pkg) + "."
	istart := l.StartIndex(r)

	resolveSymRef := func(s goobj2.SymRef) *sym.Symbol {
		i := l.Resolve(r, s)
		return l.Syms[i]
	}

	for i, n := 0, r.NSym()+r.NNonpkgdef(); i < n; i++ {
		s := l.Syms[istart+Sym(i)]
		if s == nil || s.Name == "" {
			continue
		}

		osym := goobj2.Sym{}
		osym.Read(r.Reader, r.SymOff(i))
		name := strings.Replace(osym.Name, "\"\".", pkgprefix, -1)
		if s.Name != name { // Sanity check. We can remove it in the final version.
			fmt.Println("name mismatch:", lib, i, s.Name, name)
			panic("name mismatch")
		}

		if s.Type != 0 && s.Type != sym.SXREF {
			// We've already seen this symbol, it likely came from a host object.
			continue
		}

		t := sym.AbiSymKindToSymKind[objabi.SymKind(osym.Type)]
		if t == sym.SXREF {
			log.Fatalf("bad sxref")
		}
		if t == 0 {
			log.Fatalf("missing type for %s in %s", s.Name, lib)
		}
		if !s.Attr.Reachable() && (t < sym.SDWARFSECT || t > sym.SDWARFLINES) && !(t == sym.SRODATA && strings.HasPrefix(name, "type.")) {
			// No need to load unreachable symbols.
			// XXX DWARF symbols may be used but are not marked reachable.
			// XXX type symbol's content may be needed in DWARF code, but they are not marked.
			continue
		}
		if t == sym.SBSS && (s.Type == sym.SRODATA || s.Type == sym.SNOPTRBSS) {
			t = s.Type
		}
		s.Type = t
		s.Unit = r.unit

		// Relocs
		relocs := l.relocs(r, i)
		s.R = make([]sym.Reloc, relocs.Count)
		for j := range s.R {
			r := relocs.At(j)
			rs := r.Sym
			sz := r.Size
			rt := r.Type
			if rt == objabi.R_METHODOFF {
				if l.Reachable.Has(rs) {
					rt = objabi.R_ADDROFF
				} else {
					sz = 0
					rs = 0
				}
			}
			if rt == objabi.R_WEAKADDROFF && !l.Reachable.Has(rs) {
				rs = 0
				sz = 0
			}
			if rs != 0 && l.SymType(rs) == sym.SABIALIAS {
				rsrelocs := l.Relocs(rs)
				rs = rsrelocs.At(0).Sym
			}
			s.R[j] = sym.Reloc{
				Off:  r.Off,
				Siz:  sz,
				Type: rt,
				Add:  r.Add,
				Sym:  l.Syms[rs],
			}
		}

		// Aux symbol
		naux := r.NAux(i)
		for j := 0; j < naux; j++ {
			a := goobj2.Aux{}
			a.Read(r.Reader, r.AuxOff(i, j))
			switch a.Type {
			case goobj2.AuxGotype:
				typ := resolveSymRef(a.Sym)
				if typ != nil {
					s.Gotype = typ
				}
			case goobj2.AuxFuncdata:
				pc := s.FuncInfo
				if pc == nil {
					pc = &sym.FuncInfo{Funcdata: make([]*sym.Symbol, 0, 4)}
					s.FuncInfo = pc
				}
				pc.Funcdata = append(pc.Funcdata, resolveSymRef(a.Sym))
			}
		}

		if s.Type == sym.STEXT {
			dupok := osym.Flag&goobj2.SymFlagDupok != 0
			if !dupok {
				if s.Attr.OnList() {
					log.Fatalf("symbol %s listed multiple times", s.Name)
				}
				s.Attr |= sym.AttrOnList
				lib.Textp = append(lib.Textp, s)
			} else {
				// there may ba a dup in another package
				// put into a temp list and add to text later
				lib.DupTextSyms = append(lib.DupTextSyms, s)
			}
		}
	}
}

// Load full contents.
// TODO: For now, some contents are already load in LoadReloc. Maybe
// we should combine LoadReloc back into this, once we rewrite deadcode
// pass to use index directly.
func LoadFull(l *Loader) {
	for _, o := range l.objs[1:] {
		loadObjFull(l, o.r)
	}
}

func loadObjFull(l *Loader, r *oReader) {
	lib := r.unit.Lib
	pkgprefix := objabi.PathToPrefix(lib.Pkg) + "."
	istart := l.StartIndex(r)

	resolveSymRef := func(s goobj2.SymRef) *sym.Symbol {
		i := l.Resolve(r, s)
		return l.Syms[i]
	}

	pcdataBase := r.PcdataBase()
	for i, n := 0, r.NSym()+r.NNonpkgdef(); i < n; i++ {
		s := l.Syms[istart+Sym(i)]
		if s == nil || s.Name == "" {
			continue
		}
		if !s.Attr.Reachable() && (s.Type < sym.SDWARFSECT || s.Type > sym.SDWARFLINES) && !(s.Type == sym.SRODATA && strings.HasPrefix(s.Name, "type.")) {
			// No need to load unreachable symbols.
			// XXX DWARF symbols may be used but are not marked reachable.
			// XXX type symbol's content may be needed in DWARF code, but they are not marked.
			continue
		}

		osym := goobj2.Sym{}
		osym.Read(r.Reader, r.SymOff(i))
		name := strings.Replace(osym.Name, "\"\".", pkgprefix, -1)
		if s.Name != name { // Sanity check. We can remove it in the final version.
			fmt.Println("name mismatch:", lib, i, s.Name, name)
			panic("name mismatch")
		}

		dupok := osym.Flag&goobj2.SymFlagDupok != 0
		local := osym.Flag&goobj2.SymFlagLocal != 0
		makeTypelink := osym.Flag&goobj2.SymFlagTypelink != 0
		size := osym.Siz

		// Symbol data
		s.P = r.Data(i)
		s.Attr.Set(sym.AttrReadOnly, r.ReadOnly())

		// Aux symbol info
		isym := -1
		naux := r.NAux(i)
		for j := 0; j < naux; j++ {
			a := goobj2.Aux{}
			a.Read(r.Reader, r.AuxOff(i, j))
			switch a.Type {
			case goobj2.AuxGotype, goobj2.AuxFuncdata:
				// already loaded
			case goobj2.AuxFuncInfo:
				if a.Sym.PkgIdx != goobj2.PkgIdxSelf {
					panic("funcinfo symbol not defined in current package")
				}
				isym = int(a.Sym.SymIdx)
			default:
				panic("unknown aux type")
			}
		}

		s.File = pkgprefix[:len(pkgprefix)-1]
		if dupok {
			s.Attr |= sym.AttrDuplicateOK
		}
		if s.Size < int64(size) {
			s.Size = int64(size)
		}
		s.Attr.Set(sym.AttrLocal, local)
		s.Attr.Set(sym.AttrMakeTypelink, makeTypelink)

		if s.Type == sym.SDWARFINFO {
			// For DWARF symbols, replace `"".` to actual package prefix
			// in the symbol content.
			// TODO: maybe we should do this in the compiler and get rid
			// of this.
			patchDWARFName(s, r)
		}

		if s.Type != sym.STEXT {
			continue
		}

		// FuncInfo
		if isym == -1 {
			continue
		}
		b := r.Data(isym)
		info := goobj2.FuncInfo{}
		info.Read(b)

		if info.NoSplit != 0 {
			s.Attr |= sym.AttrNoSplit
		}
		if osym.Flag&goobj2.SymFlagReflectMethod != 0 {
			s.Attr |= sym.AttrReflectMethod
		}
		if osym.Flag&goobj2.SymFlagShared != 0 {
			s.Attr |= sym.AttrShared
		}
		if osym.Flag&goobj2.SymFlagTopFrame != 0 {
			s.Attr |= sym.AttrTopFrame
		}

		info.Pcdata = append(info.Pcdata, info.PcdataEnd) // for the ease of knowing where it ends
		pc := s.FuncInfo
		if pc == nil {
			pc = &sym.FuncInfo{}
			s.FuncInfo = pc
		}
		pc.Args = int32(info.Args)
		pc.Locals = int32(info.Locals)
		pc.Pcdata = make([]sym.Pcdata, len(info.Pcdata)-1) // -1 as we appended one above
		pc.Funcdataoff = make([]int64, len(info.Funcdataoff))
		pc.File = make([]*sym.Symbol, len(info.File))
		pc.Pcsp.P = r.BytesAt(pcdataBase+info.Pcsp, int(info.Pcfile-info.Pcsp))
		pc.Pcfile.P = r.BytesAt(pcdataBase+info.Pcfile, int(info.Pcline-info.Pcfile))
		pc.Pcline.P = r.BytesAt(pcdataBase+info.Pcline, int(info.Pcinline-info.Pcline))
		pc.Pcinline.P = r.BytesAt(pcdataBase+info.Pcinline, int(info.Pcdata[0]-info.Pcinline))
		for k := range pc.Pcdata {
			pc.Pcdata[k].P = r.BytesAt(pcdataBase+info.Pcdata[k], int(info.Pcdata[k+1]-info.Pcdata[k]))
		}
		for k := range pc.Funcdataoff {
			pc.Funcdataoff[k] = int64(info.Funcdataoff[k])
		}
		for k := range pc.File {
			pc.File[k] = resolveSymRef(info.File[k])
		}
	}
}

func patchDWARFName(s *sym.Symbol, r *oReader) {
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
	pkgprefix := []byte(r.pkgprefix)
	patched := bytes.Replace(s.P[:e], emptyPkg, pkgprefix, -1)

	s.P = append(patched, s.P[e:]...)
	s.Attr.Set(sym.AttrReadOnly, false)
	delta := int64(len(s.P)) - s.Size
	s.Size = int64(len(s.P))
	for i := range s.R {
		r := &s.R[i]
		if r.Off > int32(e) {
			r.Off += int32(delta)
		}
	}
}
