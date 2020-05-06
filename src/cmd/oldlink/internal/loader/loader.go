// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loader

import (
	"bytes"
	"cmd/internal/bio"
	"cmd/internal/dwarf"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/oldlink/internal/sym"
	"fmt"
	"log"
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
	//*goobj2.Reader
	unit      *sym.CompilationUnit
	version   int    // version of static symbol
	flags     uint32 // read from object file
	pkgprefix string
	rcache    []Sym // cache mapping local PkgNone symbol to resolved Sym
}

type objIdx struct {
	r *oReader
	i Sym // start index
	e Sym // end index
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
	start       map[*oReader]Sym // map from object file to its start index
	objs        []objIdx         // sorted by start index (i.e. objIdx.i)
	max         Sym              // current max index
	extStart    Sym              // from this index on, the symbols are externally defined
	extSyms     []nameVer        // externally defined symbols
	builtinSyms []Sym            // global index of builtin symbols
	ocache      int              // index (into 'objs') of most recent lookup

	symsByName    [2]map[string]Sym // map symbol name to index, two maps are for ABI0 and ABIInternal
	extStaticSyms map[nameVer]Sym   // externally defined static symbols, keyed by name
	overwrite     map[Sym]Sym       // overwrite[i]=j if symbol j overwrites symbol i

	itablink map[Sym]struct{} // itablink[j] defined if j is go.itablink.*

	objByPkg map[string]*oReader // map package path to its Go object reader

	Syms []*sym.Symbol // indexed symbols. XXX we still make sym.Symbol for now.

	anonVersion int // most recently assigned ext static sym pseudo-version

	Reachable bitmap // bitmap of reachable symbols, indexed by global index

	// Used to implement field tracking; created during deadcode if
	// field tracking is enabled. Reachparent[K] contains the index of
	// the symbol that triggered the marking of symbol K as live.
	Reachparent []Sym

	relocBatch []sym.Reloc // for bulk allocation of relocations

	flags uint32

	strictDupMsgs int // number of strict-dup warning/errors, when FlagStrictDups is enabled
}

const (
	// Loader.flags
	FlagStrictDups = 1 << iota
)

func NewLoader(flags uint32) *Loader {
	log.Fatal("-newobj in oldlink should not be used")
	panic("unreachable")
}

// Return the start index in the global index space for a given object file.
func (l *Loader) startIndex(r *oReader) Sym {
	return l.start[r]
}

// Add a symbol with a given index, return if it is added.
func (l *Loader) AddSym(name string, ver int, i Sym, r *oReader, dupok bool, typ sym.SymKind) bool {
	panic("unreachable")
}

// Add an external symbol (without index). Return the index of newly added
// symbol, or 0 if not added.
func (l *Loader) AddExtSym(name string, ver int) Sym {
	static := ver >= sym.SymVerStatic
	if static {
		if _, ok := l.extStaticSyms[nameVer{name, ver}]; ok {
			return 0
		}
	} else {
		if _, ok := l.symsByName[ver][name]; ok {
			return 0
		}
	}
	i := l.max + 1
	if static {
		l.extStaticSyms[nameVer{name, ver}] = i
	} else {
		l.symsByName[ver][name] = i
	}
	l.max++
	if l.extStart == 0 {
		l.extStart = i
	}
	l.extSyms = append(l.extSyms, nameVer{name, ver})
	l.growSyms(int(i))
	return i
}

func (l *Loader) IsExternal(i Sym) bool {
	return l.extStart != 0 && i >= l.extStart
}

// Ensure Syms slice has enough space.
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
	if l.IsExternal(i) {
		return nil, int(i - l.extStart)
	}
	oc := l.ocache
	if oc != 0 && i >= l.objs[oc].i && i <= l.objs[oc].e {
		return l.objs[oc].r, int(i - l.objs[oc].i)
	}
	// Search for the local object holding index i.
	// Below k is the first one that has its start index > i,
	// so k-1 is the one we want.
	k := sort.Search(len(l.objs), func(k int) bool {
		return l.objs[k].i > i
	})
	l.ocache = k - 1
	return l.objs[k-1].r, int(i - l.objs[k-1].i)
}

// Look up a symbol by name, return global index, or 0 if not found.
// This is more like Syms.ROLookup than Lookup -- it doesn't create
// new symbol.
func (l *Loader) Lookup(name string, ver int) Sym {
	if ver >= sym.SymVerStatic || ver < 0 {
		return l.extStaticSyms[nameVer{name, ver}]
	}
	return l.symsByName[ver][name]
}

// Returns whether i is a dup of another symbol, and i is not
// "primary", i.e. Lookup i by name will not return i.
func (l *Loader) IsDup(i Sym) bool {
	panic("unreachable")
}

// Check that duplicate symbols have same contents.
func (l *Loader) checkdup(name string, i Sym, r *oReader, dup Sym) {
	panic("unreachable")
}

func (l *Loader) NStrictDupMsgs() int { return l.strictDupMsgs }

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
	panic("unreachable")
}

// Returns the (patched) name of the i-th symbol.
func (l *Loader) SymName(i Sym) string {
	panic("unreachable")
}

// Returns the type of the i-th symbol.
func (l *Loader) SymType(i Sym) sym.SymKind {
	panic("unreachable")
}

// Returns the attributes of the i-th symbol.
func (l *Loader) SymAttr(i Sym) uint8 {
	panic("unreachable")
}

// Returns whether the i-th symbol has ReflectMethod attribute set.
func (l *Loader) IsReflectMethod(i Sym) bool {
	panic("unreachable")
}

// Returns whether this is a Go type symbol.
func (l *Loader) IsGoType(i Sym) bool {
	panic("unreachable")
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
	panic("unreachable")
}

// Returns the number of aux symbols given a global index.
func (l *Loader) NAux(i Sym) int {
	panic("unreachable")
}

// Returns the referred symbol of the j-th aux symbol of the i-th
// symbol.
func (l *Loader) AuxSym(i Sym, j int) Sym {
	panic("unreachable")
}

// ReadAuxSyms reads the aux symbol ids for the specified symbol into the
// slice passed as a parameter. If the slice capacity is not large enough, a new
// larger slice will be allocated. Final slice is returned.
func (l *Loader) ReadAuxSyms(symIdx Sym, dst []Sym) []Sym {
	panic("unreachable")
}

// OuterSym gets the outer symbol for host object loaded symbols.
func (l *Loader) OuterSym(i Sym) Sym {
	sym := l.Syms[i]
	if sym != nil && sym.Outer != nil {
		outer := sym.Outer
		return l.Lookup(outer.Name, int(outer.Version))
	}
	return 0
}

// SubSym gets the subsymbol for host object loaded symbols.
func (l *Loader) SubSym(i Sym) Sym {
	sym := l.Syms[i]
	if sym != nil && sym.Sub != nil {
		sub := sym.Sub
		return l.Lookup(sub.Name, int(sub.Version))
	}
	return 0
}

// Initialize Reachable bitmap for running deadcode pass.
func (l *Loader) InitReachable() {
	l.Reachable = makeBitmap(l.NSym())
}

// At method returns the j-th reloc for a global symbol.
func (relocs *Relocs) At(j int) Reloc {
	panic("unreachable")
}

// ReadAll method reads all relocations for a symbol into the
// specified slice. If the slice capacity is not large enough, a new
// larger slice will be allocated. Final slice is returned.
func (relocs *Relocs) ReadAll(dst []Reloc) []Reloc {
	panic("unreachable")
}

// Relocs returns a Relocs object for the given global sym.
func (l *Loader) Relocs(i Sym) Relocs {
	panic("unreachable")
}

// Preload a package: add autolibs, add symbols to the symbol table.
// Does not read symbol data yet.
func (l *Loader) Preload(arch *sys.Arch, syms *sym.Symbols, f *bio.Reader, lib *sym.Library, unit *sym.CompilationUnit, length int64, pn string, flags int) {
	panic("unreachable")
}

// Make sure referenced symbols are added. Most of them should already be added.
// This should only be needed for referenced external symbols.
func (l *Loader) LoadRefs(arch *sys.Arch, syms *sym.Symbols) {
	for _, o := range l.objs[1:] {
		loadObjRefs(l, o.r, arch, syms)
	}
}

func loadObjRefs(l *Loader, r *oReader, arch *sys.Arch, syms *sym.Symbols) {
	panic("unreachable")
}

func abiToVer(abi uint16, localSymVersion int) int {
	panic("unreachable")
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

	nr := 0 // total number of sym.Reloc's we'll need
	for _, o := range l.objs[1:] {
		nr += loadObjSyms(l, syms, o.r)
	}

	// allocate a single large slab of relocations for all live symbols
	l.relocBatch = make([]sym.Reloc, nr)

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

	// Resolve ABI aliases for external symbols. This is only
	// needed for internal cgo linking.
	// (The old code does this in deadcode, but deadcode2 doesn't
	// do this.)
	for i := l.extStart; i <= l.max; i++ {
		if s := l.Syms[i]; s != nil && s.Attr.Reachable() {
			for ri := range s.R {
				r := &s.R[ri]
				if r.Sym != nil && r.Sym.Type == sym.SABIALIAS {
					r.Sym = r.Sym.R[0].Sym
				}
			}
		}
	}
}

// ExtractSymbols grabs the symbols out of the loader for work that hasn't been
// ported to the new symbol type.
func (l *Loader) ExtractSymbols(syms *sym.Symbols) {
	// Nil out overwritten symbols.
	// Overwritten Go symbols aren't a problem (as they're lazy loaded), but
	// symbols loaded from host object loaders are fully loaded, and we might
	// have multiple symbols with the same name. This loop nils them out.
	for oldI := range l.overwrite {
		l.Syms[oldI] = nil
	}

	// Add symbols to the ctxt.Syms lookup table. This explicitly
	// skips things created via loader.Create (marked with versions
	// less than zero), since if we tried to add these we'd wind up
	// with collisions. Along the way, update the version from the
	// negative anon version to something larger than sym.SymVerStatic
	// (needed so that sym.symbol.IsFileLocal() works properly).
	anonVerReplacement := syms.IncVersion()
	for _, s := range l.Syms {
		if s == nil {
			continue
		}
		if s.Name != "" && s.Version >= 0 {
			syms.Add(s)
		}
		if s.Version < 0 {
			s.Version = int16(anonVerReplacement)
		}
	}
}

// addNewSym adds a new sym.Symbol to the i-th index in the list of symbols.
func (l *Loader) addNewSym(i Sym, syms *sym.Symbols, name string, ver int, unit *sym.CompilationUnit, t sym.SymKind) *sym.Symbol {
	s := syms.Newsym(name, ver)
	if s.Type != 0 && s.Type != sym.SXREF {
		fmt.Println("symbol already processed:", unit.Lib, i, s)
		panic("symbol already processed")
	}
	if t == sym.SBSS && (s.Type == sym.SRODATA || s.Type == sym.SNOPTRBSS) {
		t = s.Type
	}
	s.Type = t
	s.Unit = unit
	l.growSyms(int(i))
	l.Syms[i] = s
	return s
}

// loadObjSyms creates sym.Symbol objects for the live Syms in the
// object corresponding to object reader "r". Return value is the
// number of sym.Reloc entries required for all the new symbols.
func loadObjSyms(l *Loader, syms *sym.Symbols, r *oReader) int {
	panic("unreachable")
}

// LoadSymbol loads a single symbol by name.
// This function should only be used by the host object loaders.
// NB: This function does NOT set the symbol as reachable.
func (l *Loader) LoadSymbol(name string, version int, syms *sym.Symbols) *sym.Symbol {
	panic("unreachable")
}

// LookupOrCreate looks up a symbol by name, and creates one if not found.
// Either way, it will also create a sym.Symbol for it, if not already.
// This should only be called when interacting with parts of the linker
// that still works on sym.Symbols (i.e. internal cgo linking, for now).
func (l *Loader) LookupOrCreate(name string, version int, syms *sym.Symbols) *sym.Symbol {
	i := l.Lookup(name, version)
	if i != 0 {
		// symbol exists
		if int(i) < len(l.Syms) && l.Syms[i] != nil {
			return l.Syms[i] // already loaded
		}
		if l.IsExternal(i) {
			panic("Can't load an external symbol.")
		}
		return l.LoadSymbol(name, version, syms)
	}
	i = l.AddExtSym(name, version)
	s := syms.Newsym(name, version)
	l.Syms[i] = s
	return s
}

// Create creates a symbol with the specified name, returning a
// sym.Symbol object for it. This method is intended for static/hidden
// symbols discovered while loading host objects. We can see more than
// one instance of a given static symbol with the same name/version,
// so we can't add them to the lookup tables "as is". Instead assign
// them fictitious (unique) versions, starting at -1 and decreasing by
// one for each newly created symbol, and record them in the
// extStaticSyms hash.
func (l *Loader) Create(name string, syms *sym.Symbols) *sym.Symbol {
	i := l.max + 1
	l.max++
	if l.extStart == 0 {
		l.extStart = i
	}

	// Assign a new unique negative version -- this is to mark the
	// symbol so that it can be skipped when ExtractSymbols is adding
	// ext syms to the sym.Symbols hash.
	l.anonVersion--
	ver := l.anonVersion
	l.extSyms = append(l.extSyms, nameVer{name, ver})
	l.growSyms(int(i))
	s := syms.Newsym(name, ver)
	l.Syms[i] = s
	l.extStaticSyms[nameVer{name, ver}] = i

	return s
}

func loadObjFull(l *Loader, r *oReader) {
	panic("unreachable")
}

var emptyPkg = []byte(`"".`)

func patchDWARFName1(p []byte, r *oReader) ([]byte, int) {
	// This is kind of ugly. Really the package name should not
	// even be included here.
	if len(p) < 1 || p[0] != dwarf.DW_ABRV_FUNCTION {
		return p, -1
	}
	e := bytes.IndexByte(p, 0)
	if e == -1 {
		return p, -1
	}
	if !bytes.Contains(p[:e], emptyPkg) {
		return p, -1
	}
	pkgprefix := []byte(r.pkgprefix)
	patched := bytes.Replace(p[:e], emptyPkg, pkgprefix, -1)
	return append(patched, p[e:]...), e
}

func patchDWARFName(s *sym.Symbol, r *oReader) {
	patched, e := patchDWARFName1(s.P, r)
	if e == -1 {
		return
	}
	s.P = patched
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
	for name, i := range l.symsByName[0] {
		fmt.Println(i, name, 0)
	}
	for name, i := range l.symsByName[1] {
		fmt.Println(i, name, 1)
	}
}
