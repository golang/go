// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loader

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

	ext *sym.Symbol // external symbol if not nil
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
	version   int    // version of static symbol
	flags     uint32 // read from object file
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
type Loader struct {
	start    map[*oReader]Sym // map from object file to its start index
	objs     []objIdx         // sorted by start index (i.e. objIdx.i)
	max      Sym              // current max index
	extStart Sym              // from this index on, the symbols are externally defined
	extSyms  []nameVer        // externally defined symbols

	symsByName map[nameVer]Sym // map symbol name to index
	overwrite  map[Sym]Sym     // overwrite[i]=j if symbol j overwrites symbol i

	itablink map[Sym]struct{} // itablink[j] defined if j is go.itablink.*

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
		overwrite:  make(map[Sym]Sym),
		itablink:   make(map[Sym]struct{}),
	}
}

// Return the start index in the global index space for a given object file.
func (l *Loader) startIndex(r *oReader) Sym {
	return l.start[r]
}

// Add object file r, return the start index.
func (l *Loader) addObj(pkg string, r *oReader) Sym {
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
func (l *Loader) AddSym(name string, ver int, i Sym, r *oReader, dupok bool, typ sym.SymKind) bool {
	if l.extStart != 0 {
		panic("AddSym called after AddExtSym is called")
	}
	nv := nameVer{name, ver}
	if oldi, ok := l.symsByName[nv]; ok {
		if dupok {
			return false
		}
		overwrite := r.DataSize(int(i-l.startIndex(r))) != 0
		if overwrite {
			// new symbol overwrites old symbol.
			oldr, li := l.toLocal(oldi)
			oldsym := goobj2.Sym{}
			oldsym.Read(oldr.Reader, oldr.SymOff(li))
			oldtyp := sym.AbiSymKindToSymKind[objabi.SymKind(oldsym.Type)]
			if !oldsym.Dupok() && !((oldtyp == sym.SDATA || oldtyp == sym.SNOPTRDATA || oldtyp == sym.SBSS || oldtyp == sym.SNOPTRBSS) && oldr.DataSize(li) == 0) { // only allow overwriting 0-sized data symbol
				log.Fatalf("duplicated definition of symbol " + name)
			}
			l.overwrite[oldi] = i
		} else {
			// old symbol overwrites new symbol.
			if typ != sym.SDATA && typ != sym.SNOPTRDATA && typ != sym.SBSS && typ != sym.SNOPTRBSS { // only allow overwriting data symbol
				log.Fatalf("duplicated definition of symbol " + name)
			}
			l.overwrite[i] = oldi
			return false
		}
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
	l.extSyms = append(l.extSyms, nv)
	l.growSyms(int(i))
	return i
}

// Returns whether i is an external symbol.
func (l *Loader) isExternal(i Sym) bool {
	return l.extStart != 0 && i >= l.extStart
}

// Ensure Syms slice als enough space.
func (l *Loader) growSyms(i int) {
	n := len(l.Syms)
	if n > i {
		return
	}
	l.Syms = append(l.Syms, make([]*sym.Symbol, i+1-n)...)
}

// Convert a local index to a global index.
func (l *Loader) toGlobal(r *oReader, i int) Sym {
	g := l.startIndex(r) + Sym(i)
	if ov, ok := l.overwrite[g]; ok {
		return ov
	}
	return g
}

// Convert a global index to a local index.
func (l *Loader) toLocal(i Sym) (*oReader, int) {
	if ov, ok := l.overwrite[i]; ok {
		i = ov
	}
	if l.isExternal(i) {
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
func (l *Loader) resolve(r *oReader, s goobj2.SymRef) Sym {
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
	return l.toGlobal(rr, int(s.SymIdx))
}

// Look up a symbol by name, return global index, or 0 if not found.
// This is more like Syms.ROLookup than Lookup -- it doesn't create
// new symbol.
func (l *Loader) Lookup(name string, ver int) Sym {
	nv := nameVer{name, ver}
	return l.symsByName[nv]
}

// Returns whether i is a dup of another symbol, and i is not
// "primary", i.e. Lookup i by name will not return i.
func (l *Loader) IsDup(i Sym) bool {
	if _, ok := l.overwrite[i]; ok {
		return true
	}
	if l.isExternal(i) {
		return false
	}
	r, li := l.toLocal(i)
	osym := goobj2.Sym{}
	osym.Read(r.Reader, r.SymOff(li))
	if !osym.Dupok() {
		return false
	}
	if osym.Name == "" {
		return false
	}
	name := strings.Replace(osym.Name, "\"\".", r.pkgprefix, -1)
	ver := abiToVer(osym.ABI, r.version)
	return l.symsByName[nameVer{name, ver}] != i
}

// Number of total symbols.
func (l *Loader) NSym() int {
	return int(l.max + 1)
}

// Number of defined Go symbols.
func (l *Loader) NDef() int {
	return int(l.extStart)
}

// Returns the raw (unpatched) name of the i-th symbol.
func (l *Loader) RawSymName(i Sym) string {
	if l.isExternal(i) {
		if s := l.Syms[i]; s != nil {
			return s.Name
		}
		return ""
	}
	r, li := l.toLocal(i)
	osym := goobj2.Sym{}
	osym.Read(r.Reader, r.SymOff(li))
	return osym.Name
}

// Returns the (patched) name of the i-th symbol.
func (l *Loader) SymName(i Sym) string {
	if l.isExternal(i) {
		if s := l.Syms[i]; s != nil {
			return s.Name // external name should already be patched?
		}
		return ""
	}
	r, li := l.toLocal(i)
	osym := goobj2.Sym{}
	osym.Read(r.Reader, r.SymOff(li))
	return strings.Replace(osym.Name, "\"\".", r.pkgprefix, -1)
}

// Returns the type of the i-th symbol.
func (l *Loader) SymType(i Sym) sym.SymKind {
	if l.isExternal(i) {
		if s := l.Syms[i]; s != nil {
			return s.Type
		}
		return 0
	}
	r, li := l.toLocal(i)
	osym := goobj2.Sym{}
	osym.Read(r.Reader, r.SymOff(li))
	return sym.AbiSymKindToSymKind[objabi.SymKind(osym.Type)]
}

// Returns the attributes of the i-th symbol.
func (l *Loader) SymAttr(i Sym) uint8 {
	if l.isExternal(i) {
		// TODO: do something? External symbols have different representation of attributes. For now, ReflectMethod is the only thing matters and it cannot be set by external symbol.
		return 0
	}
	r, li := l.toLocal(i)
	osym := goobj2.Sym{}
	osym.Read(r.Reader, r.SymOff(li))
	return osym.Flag
}

// Returns whether the i-th symbol has ReflectMethod attribute set.
func (l *Loader) IsReflectMethod(i Sym) bool {
	return l.SymAttr(i)&goobj2.SymFlagReflectMethod != 0
}

// Returns whether this is a Go type symbol.
func (l *Loader) IsGoType(i Sym) bool {
	return l.SymAttr(i)&goobj2.SymFlagGoType != 0
}

// Returns whether this is a "go.itablink.*" symbol.
func (l *Loader) IsItabLink(i Sym) bool {
	if _, ok := l.itablink[i]; ok {
		return true
	}
	return false
}

// Returns the symbol content of the i-th symbol. i is global index.
func (l *Loader) Data(i Sym) []byte {
	if l.isExternal(i) {
		if s := l.Syms[i]; s != nil {
			return s.P
		}
		return nil
	}
	r, li := l.toLocal(i)
	return r.Data(li)
}

// Returns the number of aux symbols given a global index.
func (l *Loader) NAux(i Sym) int {
	if l.isExternal(i) {
		return 0
	}
	r, li := l.toLocal(i)
	return r.NAux(li)
}

// Returns the referred symbol of the j-th aux symbol of the i-th
// symbol.
func (l *Loader) AuxSym(i Sym, j int) Sym {
	if l.isExternal(i) {
		return 0
	}
	r, li := l.toLocal(i)
	a := goobj2.Aux{}
	a.Read(r.Reader, r.AuxOff(li, j))
	return l.resolve(r, a.Sym)
}

// Initialize Reachable bitmap for running deadcode pass.
func (l *Loader) InitReachable() {
	l.Reachable = makeBitmap(l.NSym())
}

// At method returns the j-th reloc for a global symbol.
func (relocs *Relocs) At(j int) Reloc {
	if relocs.ext != nil {
		rel := &relocs.ext.R[j]
		return Reloc{
			Off:  rel.Off,
			Size: rel.Siz,
			Type: rel.Type,
			Add:  rel.Add,
			Sym:  relocs.l.Lookup(rel.Sym.Name, int(rel.Sym.Version)),
		}
	}
	rel := goobj2.Reloc{}
	rel.Read(relocs.r.Reader, relocs.r.RelocOff(relocs.li, j))
	target := relocs.l.resolve(relocs.r, rel.Sym)
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
	if l.isExternal(i) {
		if s := l.Syms[i]; s != nil {
			return Relocs{Count: len(s.R), l: l, ext: s}
		}
		return Relocs{}
	}
	r, li := l.toLocal(i)
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
func (l *Loader) Preload(arch *sys.Arch, syms *sym.Symbols, f *bio.Reader, lib *sym.Library, unit *sym.CompilationUnit, length int64, pn string, flags int) {
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
	or := &oReader{r, unit, localSymVersion, r.Flags(), pkgprefix}

	// Autolib
	lib.ImportStrings = append(lib.ImportStrings, r.Autolib()...)

	// DWARF file table
	nfile := r.NDwarfFile()
	unit.DWARFFileTable = make([]string, nfile)
	for i := range unit.DWARFFileTable {
		unit.DWARFFileTable[i] = r.DwarfFile(i)
	}

	istart := l.addObj(lib.Pkg, or)

	ndef := r.NSym()
	nnonpkgdef := r.NNonpkgdef()
	for i, n := 0, ndef+nnonpkgdef; i < n; i++ {
		osym := goobj2.Sym{}
		osym.Read(r, r.SymOff(i))
		name := strings.Replace(osym.Name, "\"\".", pkgprefix, -1)
		if name == "" {
			continue // don't add unnamed aux symbol
		}
		v := abiToVer(osym.ABI, localSymVersion)
		dupok := osym.Dupok()
		added := l.AddSym(name, v, istart+Sym(i), or, dupok, sym.AbiSymKindToSymKind[objabi.SymKind(osym.Type)])
		if added && strings.HasPrefix(name, "go.itablink.") {
			l.itablink[istart+Sym(i)] = struct{}{}
		}
	}

	// The caller expects us consuming all the data
	f.MustSeek(length, os.SEEK_CUR)
}

// Make sure referenced symbols are added. Most of them should already be added.
// This should only be needed for referenced external symbols.
func (l *Loader) LoadRefs(arch *sys.Arch, syms *sym.Symbols) {
	for _, o := range l.objs[1:] {
		loadObjRefs(l, o.r, arch, syms)
	}
}

func loadObjRefs(l *Loader, r *oReader, arch *sys.Arch, syms *sym.Symbols) {
	ndef := r.NSym() + r.NNonpkgdef()
	for i, n := 0, r.NNonpkgref(); i < n; i++ {
		osym := goobj2.Sym{}
		osym.Read(r.Reader, r.SymOff(ndef+i))
		name := strings.Replace(osym.Name, "\"\".", r.pkgprefix, -1)
		v := abiToVer(osym.ABI, r.version)
		l.AddExtSym(name, v)
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
	}
}

// Load full contents.
func (l *Loader) LoadFull(arch *sys.Arch, syms *sym.Symbols) {
	// create all Symbols first.
	l.growSyms(l.NSym())
	for _, o := range l.objs[1:] {
		loadObjSyms(l, syms, o.r)
	}

	// external symbols
	for i := l.extStart; i <= l.max; i++ {
		if s := l.Syms[i]; s != nil {
			s.Attr.Set(sym.AttrReachable, l.Reachable.Has(i))
			continue // already loaded from external object
		}
		nv := l.extSyms[i-l.extStart]
		if l.Reachable.Has(i) || strings.HasPrefix(nv.name, "gofile..") { // XXX file symbols are used but not marked
			s := syms.Newsym(nv.name, nv.v)
			preprocess(arch, s)
			s.Attr.Set(sym.AttrReachable, l.Reachable.Has(i))
			l.Syms[i] = s
		}
	}

	// load contents of defined symbols
	for _, o := range l.objs[1:] {
		loadObjFull(l, o.r)
	}
}

func loadObjSyms(l *Loader, syms *sym.Symbols, r *oReader) {
	lib := r.unit.Lib
	istart := l.startIndex(r)

	for i, n := 0, r.NSym()+r.NNonpkgdef(); i < n; i++ {
		osym := goobj2.Sym{}
		osym.Read(r.Reader, r.SymOff(i))
		name := strings.Replace(osym.Name, "\"\".", r.pkgprefix, -1)
		if name == "" {
			continue
		}
		ver := abiToVer(osym.ABI, r.version)
		if l.symsByName[nameVer{name, ver}] != istart+Sym(i) {
			continue
		}

		t := sym.AbiSymKindToSymKind[objabi.SymKind(osym.Type)]
		if t == sym.SXREF {
			log.Fatalf("bad sxref")
		}
		if t == 0 {
			log.Fatalf("missing type for %s in %s", name, lib)
		}
		if !l.Reachable.Has(istart+Sym(i)) && !(t == sym.SRODATA && strings.HasPrefix(name, "type.")) && name != "runtime.addmoduledata" && name != "runtime.lastmoduledatap" {
			// No need to load unreachable symbols.
			// XXX some type symbol's content may be needed in DWARF code, but they are not marked.
			// XXX reference to runtime.addmoduledata may be generated later by the linker in plugin mode.
			continue
		}

		s := syms.Newsym(name, ver)
		if s.Type != 0 && s.Type != sym.SXREF {
			fmt.Println("symbol already processed:", lib, i, s)
			panic("symbol already processed")
		}
		if t == sym.SBSS && (s.Type == sym.SRODATA || s.Type == sym.SNOPTRBSS) {
			t = s.Type
		}
		s.Type = t
		s.Unit = r.unit
		s.Attr.Set(sym.AttrReachable, l.Reachable.Has(istart+Sym(i)))
		l.Syms[istart+Sym(i)] = s
	}
}

func loadObjFull(l *Loader, r *oReader) {
	lib := r.unit.Lib
	istart := l.startIndex(r)

	resolveSymRef := func(s goobj2.SymRef) *sym.Symbol {
		i := l.resolve(r, s)
		return l.Syms[i]
	}

	pcdataBase := r.PcdataBase()
	for i, n := 0, r.NSym()+r.NNonpkgdef(); i < n; i++ {
		osym := goobj2.Sym{}
		osym.Read(r.Reader, r.SymOff(i))
		name := strings.Replace(osym.Name, "\"\".", r.pkgprefix, -1)
		if name == "" {
			continue
		}
		ver := abiToVer(osym.ABI, r.version)
		dupok := osym.Dupok()
		if dupsym := l.symsByName[nameVer{name, ver}]; dupsym != istart+Sym(i) {
			if dupok && l.Reachable.Has(dupsym) {
				// A dupok symbol is resolved to another package. We still need
				// to record its presence in the current package, as the trampoline
				// pass expects packages are laid out in dependency order.
				s := l.Syms[dupsym]
				if s.Type == sym.STEXT {
					lib.DupTextSyms = append(lib.DupTextSyms, s)
				}
			}
			continue
		}

		s := l.Syms[istart+Sym(i)]
		if s == nil {
			continue
		}
		if s.Name != name { // Sanity check. We can remove it in the final version.
			fmt.Println("name mismatch:", lib, i, s.Name, name)
			panic("name mismatch")
		}

		local := osym.Local()
		makeTypelink := osym.Typelink()
		size := osym.Siz

		// Symbol data
		s.P = r.Data(i)
		s.Attr.Set(sym.AttrReadOnly, r.ReadOnly())

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

		// Aux symbol info
		isym := -1
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
			case goobj2.AuxFuncInfo:
				if a.Sym.PkgIdx != goobj2.PkgIdxSelf {
					panic("funcinfo symbol not defined in current package")
				}
				isym = int(a.Sym.SymIdx)
			case goobj2.AuxDwarfInfo, goobj2.AuxDwarfLoc, goobj2.AuxDwarfRanges, goobj2.AuxDwarfLines:
				// ignored for now
			default:
				panic("unknown aux type")
			}
		}

		s.File = r.pkgprefix[:len(r.pkgprefix)-1]
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
		if osym.ReflectMethod() {
			s.Attr |= sym.AttrReflectMethod
		}
		if r.Flags()&goobj2.ObjFlagShared != 0 {
			s.Attr |= sym.AttrShared
		}
		if osym.TopFrame() {
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
		pc.InlTree = make([]sym.InlinedCall, len(info.InlTree))
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
		for k := range pc.InlTree {
			inl := &info.InlTree[k]
			pc.InlTree[k] = sym.InlinedCall{
				Parent:   inl.Parent,
				File:     resolveSymRef(inl.File),
				Line:     inl.Line,
				Func:     l.SymName(l.resolve(r, inl.Func)),
				ParentPC: inl.ParentPC,
			}
		}

		if !dupok {
			if s.Attr.OnList() {
				log.Fatalf("symbol %s listed multiple times", s.Name)
			}
			s.Attr.Set(sym.AttrOnList, true)
			lib.Textp = append(lib.Textp, s)
		} else {
			// there may ba a dup in another package
			// put into a temp list and add to text later
			lib.DupTextSyms = append(lib.DupTextSyms, s)
		}
	}
}

var emptyPkg = []byte(`"".`)

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

// For debugging.
func (l *Loader) Dump() {
	fmt.Println("objs")
	for _, obj := range l.objs {
		if obj.r != nil {
			fmt.Println(obj.i, obj.r.unit.Lib)
		}
	}
	fmt.Println("syms")
	for i, s := range l.Syms {
		if i == 0 {
			continue
		}
		if s != nil {
			fmt.Println(i, s, s.Type)
		} else {
			fmt.Println(i, l.SymName(Sym(i)), "<not loaded>")
		}
	}
	fmt.Println("overwrite:", l.overwrite)
	fmt.Println("symsByName")
	for nv, i := range l.symsByName {
		fmt.Println(i, nv.name, nv.v)
	}
}
