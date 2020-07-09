// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package loader

import (
	"bytes"
	"cmd/internal/bio"
	"cmd/internal/goobj2"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/sym"
	"debug/elf"
	"fmt"
	"log"
	"math/bits"
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
	rs []goobj2.Reloc

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

// ExtReloc contains the payload for an external relocation.
type ExtReloc struct {
	Idx  int // index of the original relocation
	Xsym Sym
	Xadd int64
}

// ExtRelocView is a view of an external relocation.
// It is intended to be constructed on the fly, such as ExtRelocs.At.
// It is not the data structure used to store the payload internally.
type ExtRelocView struct {
	Reloc2
	*ExtReloc
}

// Reloc2 holds a "handle" to access a relocation record from an
// object file.
type Reloc2 struct {
	*goobj2.Reloc
	r *oReader
	l *Loader

	// External reloc types may not fit into a uint8 which the Go object file uses.
	// Store it here, instead of in the byte of goobj2.Reloc2.
	// For Go symbols this will always be zero.
	// goobj2.Reloc2.Type() + typ is always the right type, for both Go and external
	// symbols.
	typ objabi.RelocType
}

func (rel Reloc2) Type() objabi.RelocType { return objabi.RelocType(rel.Reloc.Type()) + rel.typ }
func (rel Reloc2) Sym() Sym               { return rel.l.resolve(rel.r, rel.Reloc.Sym()) }
func (rel Reloc2) SetSym(s Sym)           { rel.Reloc.SetSym(goobj2.SymRef{PkgIdx: 0, SymIdx: uint32(s)}) }

func (rel Reloc2) SetType(t objabi.RelocType) {
	if t != objabi.RelocType(uint8(t)) {
		panic("SetType: type doesn't fit into Reloc2")
	}
	rel.Reloc.SetType(uint8(t))
	if rel.typ != 0 {
		// should use SymbolBuilder.SetRelocType
		panic("wrong method to set reloc type")
	}
}

// Aux2 holds a "handle" to access an aux symbol record from an
// object file.
type Aux2 struct {
	*goobj2.Aux
	r *oReader
	l *Loader
}

func (a Aux2) Sym() Sym { return a.l.resolve(a.r, a.Aux.Sym()) }

// oReader is a wrapper type of obj.Reader, along with some
// extra information.
// TODO: rename to objReader once the old one is gone?
type oReader struct {
	*goobj2.Reader
	unit      *sym.CompilationUnit
	version   int    // version of static symbol
	flags     uint32 // read from object file
	pkgprefix string
	syms      []Sym  // Sym's global index, indexed by local index
	ndef      int    // cache goobj2.Reader.NSym()
	objidx    uint32 // index of this reader in the objs slice
}

type objIdx struct {
	r *oReader
	i Sym // start index
}

// objSym represents a symbol in an object file. It is a tuple of
// the object and the symbol's local index.
// For external symbols, r is l.extReader, s is its index into the
// payload array.
// {nil, 0} represents the nil symbol.
type objSym struct {
	r *oReader
	s int // local index
}

type nameVer struct {
	name string
	v    int
}

type Bitmap []uint32

// set the i-th bit.
func (bm Bitmap) Set(i Sym) {
	n, r := uint(i)/32, uint(i)%32
	bm[n] |= 1 << r
}

// unset the i-th bit.
func (bm Bitmap) Unset(i Sym) {
	n, r := uint(i)/32, uint(i)%32
	bm[n] &^= (1 << r)
}

// whether the i-th bit is set.
func (bm Bitmap) Has(i Sym) bool {
	n, r := uint(i)/32, uint(i)%32
	return bm[n]&(1<<r) != 0
}

// return current length of bitmap in bits.
func (bm Bitmap) Len() int {
	return len(bm) * 32
}

// return the number of bits set.
func (bm Bitmap) Count() int {
	s := 0
	for _, x := range bm {
		s += bits.OnesCount32(x)
	}
	return s
}

func MakeBitmap(n int) Bitmap {
	return make(Bitmap, (n+31)/32)
}

// growBitmap insures that the specified bitmap has enough capacity,
// reallocating (doubling the size) if needed.
func growBitmap(reqLen int, b Bitmap) Bitmap {
	curLen := b.Len()
	if reqLen > curLen {
		b = append(b, MakeBitmap(reqLen+1-curLen)...)
	}
	return b
}

// A Loader loads new object files and resolves indexed symbol references.
//
// Notes on the layout of global symbol index space:
//
// - Go object files are read before host object files; each Go object
//   read adds its defined package symbols to the global index space.
//   Nonpackage symbols are not yet added.
//
// - In loader.LoadNonpkgSyms, add non-package defined symbols and
//   references in all object files to the global index space.
//
// - Host object file loading happens; the host object loader does a
//   name/version lookup for each symbol it finds; this can wind up
//   extending the external symbol index space range. The host object
//   loader stores symbol payloads in loader.payloads using SymbolBuilder.
//
// - For now, in loader.LoadFull we convert all symbols (Go + external)
//   to sym.Symbols.
//
// - At some point (when the wayfront is pushed through all of the
//   linker), all external symbols will be payload-based, and we can
//   get rid of the loader.Syms array.
//
// - Each symbol gets a unique global index. For duplicated and
//   overwriting/overwritten symbols, the second (or later) appearance
//   of the symbol gets the same global index as the first appearance.
type Loader struct {
	start       map[*oReader]Sym // map from object file to its start index
	objs        []objIdx         // sorted by start index (i.e. objIdx.i)
	extStart    Sym              // from this index on, the symbols are externally defined
	builtinSyms []Sym            // global index of builtin symbols

	objSyms []objSym // global index mapping to local index

	symsByName    [2]map[string]Sym // map symbol name to index, two maps are for ABI0 and ABIInternal
	extStaticSyms map[nameVer]Sym   // externally defined static symbols, keyed by name

	extReader    *oReader // a dummy oReader, for external symbols
	payloadBatch []extSymPayload
	payloads     []*extSymPayload // contents of linker-materialized external syms
	values       []int64          // symbol values, indexed by global sym index

	sects    []*sym.Section // sections
	symSects []uint16       // symbol's section, index to sects array

	align []uint8 // symbol 2^N alignment, indexed by global index

	outdata   [][]byte     // symbol's data in the output buffer
	extRelocs [][]ExtReloc // symbol's external relocations

	itablink         map[Sym]struct{} // itablink[j] defined if j is go.itablink.*
	deferReturnTramp map[Sym]bool     // whether the symbol is a trampoline of a deferreturn call

	objByPkg map[string]*oReader // map package path to its Go object reader

	Syms     []*sym.Symbol // indexed symbols. XXX we still make sym.Symbol for now.
	symBatch []sym.Symbol  // batch of symbols.

	anonVersion int // most recently assigned ext static sym pseudo-version

	// Bitmaps and other side structures used to store data used to store
	// symbol flags/attributes; these are to be accessed via the
	// corresponding loader "AttrXXX" and "SetAttrXXX" methods. Please
	// visit the comments on these methods for more details on the
	// semantics / interpretation of the specific flags or attribute.
	attrReachable        Bitmap // reachable symbols, indexed by global index
	attrOnList           Bitmap // "on list" symbols, indexed by global index
	attrLocal            Bitmap // "local" symbols, indexed by global index
	attrNotInSymbolTable Bitmap // "not in symtab" symbols, indexed by glob idx
	attrVisibilityHidden Bitmap // hidden symbols, indexed by ext sym index
	attrDuplicateOK      Bitmap // dupOK symbols, indexed by ext sym index
	attrShared           Bitmap // shared symbols, indexed by ext sym index
	attrExternal         Bitmap // external symbols, indexed by ext sym index

	attrReadOnly         map[Sym]bool     // readonly data for this sym
	attrTopFrame         map[Sym]struct{} // top frame symbols
	attrSpecial          map[Sym]struct{} // "special" frame symbols
	attrCgoExportDynamic map[Sym]struct{} // "cgo_export_dynamic" symbols
	attrCgoExportStatic  map[Sym]struct{} // "cgo_export_static" symbols

	// Outer and Sub relations for symbols.
	// TODO: figure out whether it's more efficient to just have these
	// as fields on extSymPayload (note that this won't be a viable
	// strategy if somewhere in the linker we set sub/outer for a
	// non-external sym).
	outer map[Sym]Sym
	sub   map[Sym]Sym

	dynimplib   map[Sym]string      // stores Dynimplib symbol attribute
	dynimpvers  map[Sym]string      // stores Dynimpvers symbol attribute
	localentry  map[Sym]uint8       // stores Localentry symbol attribute
	extname     map[Sym]string      // stores Extname symbol attribute
	elfType     map[Sym]elf.SymType // stores elf type symbol property
	elfSym      map[Sym]int32       // stores elf sym symbol property
	localElfSym map[Sym]int32       // stores "local" elf sym symbol property
	symPkg      map[Sym]string      // stores package for symbol, or library for shlib-derived syms
	plt         map[Sym]int32       // stores dynimport for pe objects
	got         map[Sym]int32       // stores got for pe objects
	dynid       map[Sym]int32       // stores Dynid for symbol

	relocVariant map[relocId]sym.RelocVariant // stores variant relocs

	// Used to implement field tracking; created during deadcode if
	// field tracking is enabled. Reachparent[K] contains the index of
	// the symbol that triggered the marking of symbol K as live.
	Reachparent []Sym

	relocBatch    []sym.Reloc    // for bulk allocation of relocations
	relocExtBatch []sym.RelocExt // for bulk allocation of relocations

	flags uint32

	strictDupMsgs int // number of strict-dup warning/errors, when FlagStrictDups is enabled

	elfsetstring elfsetstringFunc

	errorReporter *ErrorReporter

	SymLookup func(name string, ver int) *sym.Symbol
}

const (
	pkgDef = iota
	nonPkgDef
	nonPkgRef
)

type elfsetstringFunc func(s *sym.Symbol, str string, off int)

// extSymPayload holds the payload (data + relocations) for linker-synthesized
// external symbols (note that symbol value is stored in a separate slice).
type extSymPayload struct {
	name     string // TODO: would this be better as offset into str table?
	size     int64
	ver      int
	kind     sym.SymKind
	objidx   uint32 // index of original object if sym made by cloneToExternal
	gotype   Sym    // Gotype (0 if not present)
	relocs   []goobj2.Reloc
	reltypes []objabi.RelocType // relocation types
	data     []byte
	auxs     []goobj2.Aux
}

const (
	// Loader.flags
	FlagStrictDups = 1 << iota
)

func NewLoader(flags uint32, elfsetstring elfsetstringFunc, reporter *ErrorReporter) *Loader {
	nbuiltin := goobj2.NBuiltin()
	ldr := &Loader{
		start:                make(map[*oReader]Sym),
		objs:                 []objIdx{{}}, // reserve index 0 for nil symbol
		objSyms:              []objSym{{}}, // reserve index 0 for nil symbol
		extReader:            &oReader{},
		symsByName:           [2]map[string]Sym{make(map[string]Sym, 100000), make(map[string]Sym, 50000)}, // preallocate ~2MB for ABI0 and ~1MB for ABI1 symbols
		objByPkg:             make(map[string]*oReader),
		outer:                make(map[Sym]Sym),
		sub:                  make(map[Sym]Sym),
		dynimplib:            make(map[Sym]string),
		dynimpvers:           make(map[Sym]string),
		localentry:           make(map[Sym]uint8),
		extname:              make(map[Sym]string),
		attrReadOnly:         make(map[Sym]bool),
		elfType:              make(map[Sym]elf.SymType),
		elfSym:               make(map[Sym]int32),
		localElfSym:          make(map[Sym]int32),
		symPkg:               make(map[Sym]string),
		plt:                  make(map[Sym]int32),
		got:                  make(map[Sym]int32),
		dynid:                make(map[Sym]int32),
		attrTopFrame:         make(map[Sym]struct{}),
		attrSpecial:          make(map[Sym]struct{}),
		attrCgoExportDynamic: make(map[Sym]struct{}),
		attrCgoExportStatic:  make(map[Sym]struct{}),
		itablink:             make(map[Sym]struct{}),
		deferReturnTramp:     make(map[Sym]bool),
		extStaticSyms:        make(map[nameVer]Sym),
		builtinSyms:          make([]Sym, nbuiltin),
		flags:                flags,
		elfsetstring:         elfsetstring,
		errorReporter:        reporter,
		sects:                []*sym.Section{nil}, // reserve index 0 for nil section
	}
	reporter.ldr = ldr
	return ldr
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
	i := Sym(len(l.objSyms))
	l.start[r] = i
	l.objs = append(l.objs, objIdx{r, i})
	return i
}

// Add a symbol from an object file, return the global index and whether it is added.
// If the symbol already exist, it returns the index of that symbol.
func (l *Loader) AddSym(name string, ver int, r *oReader, li int, kind int, dupok bool, typ sym.SymKind) (Sym, bool) {
	if l.extStart != 0 {
		panic("AddSym called after external symbol is created")
	}
	i := Sym(len(l.objSyms))
	addToGlobal := func() {
		l.objSyms = append(l.objSyms, objSym{r, li})
	}
	if name == "" {
		addToGlobal()
		return i, true // unnamed aux symbol
	}
	if ver == r.version {
		// Static symbol. Add its global index but don't
		// add to name lookup table, as it cannot be
		// referenced by name.
		addToGlobal()
		return i, true
	}
	if kind == pkgDef {
		// Defined package symbols cannot be dup to each other.
		// We load all the package symbols first, so we don't need
		// to check dup here.
		// We still add it to the lookup table, as it may still be
		// referenced by name (e.g. through linkname).
		l.symsByName[ver][name] = i
		addToGlobal()
		return i, true
	}

	// Non-package (named) symbol. Check if it already exists.
	oldi, existed := l.symsByName[ver][name]
	if !existed {
		l.symsByName[ver][name] = i
		addToGlobal()
		return i, true
	}
	// symbol already exists
	if dupok {
		if l.flags&FlagStrictDups != 0 {
			l.checkdup(name, r, li, oldi)
		}
		return oldi, false
	}
	oldr, oldli := l.toLocal(oldi)
	oldsym := oldr.Sym(oldli)
	if oldsym.Dupok() {
		return oldi, false
	}
	overwrite := r.DataSize(li) != 0
	if overwrite {
		// new symbol overwrites old symbol.
		oldtyp := sym.AbiSymKindToSymKind[objabi.SymKind(oldsym.Type())]
		if !(oldtyp.IsData() && oldr.DataSize(oldli) == 0) {
			log.Fatalf("duplicated definition of symbol " + name)
		}
		l.objSyms[oldi] = objSym{r, li}
	} else {
		// old symbol overwrites new symbol.
		if !typ.IsData() { // only allow overwriting data symbol
			log.Fatalf("duplicated definition of symbol " + name)
		}
	}
	return oldi, true
}

// newExtSym creates a new external sym with the specified
// name/version.
func (l *Loader) newExtSym(name string, ver int) Sym {
	i := Sym(len(l.objSyms))
	if l.extStart == 0 {
		l.extStart = i
	}
	l.growSyms(int(i))
	pi := l.newPayload(name, ver)
	l.objSyms = append(l.objSyms, objSym{l.extReader, int(pi)})
	l.extReader.syms = append(l.extReader.syms, i)
	return i
}

// LookupOrCreateSym looks up the symbol with the specified name/version,
// returning its Sym index if found. If the lookup fails, a new external
// Sym will be created, entered into the lookup tables, and returned.
func (l *Loader) LookupOrCreateSym(name string, ver int) Sym {
	i := l.Lookup(name, ver)
	if i != 0 {
		return i
	}
	i = l.newExtSym(name, ver)
	static := ver >= sym.SymVerStatic || ver < 0
	if static {
		l.extStaticSyms[nameVer{name, ver}] = i
	} else {
		l.symsByName[ver][name] = i
	}
	return i
}

func (l *Loader) IsExternal(i Sym) bool {
	r, _ := l.toLocal(i)
	return l.isExtReader(r)
}

func (l *Loader) isExtReader(r *oReader) bool {
	return r == l.extReader
}

// For external symbol, return its index in the payloads array.
// XXX result is actually not a global index. We (ab)use the Sym type
// so we don't need conversion for accessing bitmaps.
func (l *Loader) extIndex(i Sym) Sym {
	_, li := l.toLocal(i)
	return Sym(li)
}

// Get a new payload for external symbol, return its index in
// the payloads array.
func (l *Loader) newPayload(name string, ver int) int {
	pi := len(l.payloads)
	pp := l.allocPayload()
	pp.name = name
	pp.ver = ver
	l.payloads = append(l.payloads, pp)
	l.growExtAttrBitmaps()
	return pi
}

// getPayload returns a pointer to the extSymPayload struct for an
// external symbol if the symbol has a payload. Will panic if the
// symbol in question is bogus (zero or not an external sym).
func (l *Loader) getPayload(i Sym) *extSymPayload {
	if !l.IsExternal(i) {
		panic(fmt.Sprintf("bogus symbol index %d in getPayload", i))
	}
	pi := l.extIndex(i)
	return l.payloads[pi]
}

// allocPayload allocates a new payload.
func (l *Loader) allocPayload() *extSymPayload {
	batch := l.payloadBatch
	if len(batch) == 0 {
		batch = make([]extSymPayload, 1000)
	}
	p := &batch[0]
	l.payloadBatch = batch[1:]
	return p
}

func (ms *extSymPayload) Grow(siz int64) {
	if int64(int(siz)) != siz {
		log.Fatalf("symgrow size %d too long", siz)
	}
	if int64(len(ms.data)) >= siz {
		return
	}
	if cap(ms.data) < int(siz) {
		cl := len(ms.data)
		ms.data = append(ms.data, make([]byte, int(siz)+1-cl)...)
		ms.data = ms.data[0:cl]
	}
	ms.data = ms.data[:siz]
}

// Ensure Syms slice has enough space.
func (l *Loader) growSyms(i int) {
	n := len(l.Syms)
	if n > i {
		return
	}
	l.Syms = append(l.Syms, make([]*sym.Symbol, i+1-n)...)
	l.growValues(int(i) + 1)
	l.growAttrBitmaps(int(i) + 1)
}

// Convert a local index to a global index.
func (l *Loader) toGlobal(r *oReader, i int) Sym {
	return r.syms[i]
}

// Convert a global index to a local index.
func (l *Loader) toLocal(i Sym) (*oReader, int) {
	return l.objSyms[i].r, int(l.objSyms[i].s)
}

// Resolve a local symbol reference. Return global index.
func (l *Loader) resolve(r *oReader, s goobj2.SymRef) Sym {
	var rr *oReader
	switch p := s.PkgIdx; p {
	case goobj2.PkgIdxInvalid:
		// {0, X} with non-zero X is never a valid sym reference from a Go object.
		// We steal this space for symbol references from external objects.
		// In this case, X is just the global index.
		if l.isExtReader(r) {
			return Sym(s.SymIdx)
		}
		if s.SymIdx != 0 {
			panic("bad sym ref")
		}
		return 0
	case goobj2.PkgIdxNone:
		i := int(s.SymIdx) + r.ndef
		return r.syms[i]
	case goobj2.PkgIdxBuiltin:
		return l.builtinSyms[s.SymIdx]
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
	if ver >= sym.SymVerStatic || ver < 0 {
		return l.extStaticSyms[nameVer{name, ver}]
	}
	return l.symsByName[ver][name]
}

// Check that duplicate symbols have same contents.
func (l *Loader) checkdup(name string, r *oReader, li int, dup Sym) {
	p := r.Data(li)
	rdup, ldup := l.toLocal(dup)
	pdup := rdup.Data(ldup)
	if bytes.Equal(p, pdup) {
		return
	}
	reason := "same length but different contents"
	if len(p) != len(pdup) {
		reason = fmt.Sprintf("new length %d != old length %d", len(p), len(pdup))
	}
	fmt.Fprintf(os.Stderr, "cmd/link: while reading object for '%v': duplicate symbol '%s', previous def at '%v', with mismatched payload: %s\n", r.unit.Lib, name, rdup.unit.Lib, reason)

	// For the moment, allow DWARF subprogram DIEs for
	// auto-generated wrapper functions. What seems to happen
	// here is that we get different line numbers on formal
	// params; I am guessing that the pos is being inherited
	// from the spot where the wrapper is needed.
	allowed := strings.HasPrefix(name, "go.info.go.interface") ||
		strings.HasPrefix(name, "go.info.go.builtin") ||
		strings.HasPrefix(name, "go.debuglines")
	if !allowed {
		l.strictDupMsgs++
	}
}

func (l *Loader) NStrictDupMsgs() int { return l.strictDupMsgs }

// Number of total symbols.
func (l *Loader) NSym() int {
	return len(l.objSyms)
}

// Number of defined Go symbols.
func (l *Loader) NDef() int {
	return int(l.extStart)
}

// Number of reachable symbols.
func (l *Loader) NReachableSym() int {
	return l.attrReachable.Count()
}

// Returns the raw (unpatched) name of the i-th symbol.
func (l *Loader) RawSymName(i Sym) string {
	if l.IsExternal(i) {
		pp := l.getPayload(i)
		return pp.name
	}
	r, li := l.toLocal(i)
	return r.Sym(li).Name(r.Reader)
}

// Returns the (patched) name of the i-th symbol.
func (l *Loader) SymName(i Sym) string {
	if l.IsExternal(i) {
		pp := l.getPayload(i)
		return pp.name
	}
	r, li := l.toLocal(i)
	return strings.Replace(r.Sym(li).Name(r.Reader), "\"\".", r.pkgprefix, -1)
}

// Returns the version of the i-th symbol.
func (l *Loader) SymVersion(i Sym) int {
	if l.IsExternal(i) {
		pp := l.getPayload(i)
		return pp.ver
	}
	r, li := l.toLocal(i)
	return int(abiToVer(r.Sym(li).ABI(), r.version))
}

// Returns the type of the i-th symbol.
func (l *Loader) SymType(i Sym) sym.SymKind {
	if l.IsExternal(i) {
		pp := l.getPayload(i)
		if pp != nil {
			return pp.kind
		}
		return 0
	}
	r, li := l.toLocal(i)
	return sym.AbiSymKindToSymKind[objabi.SymKind(r.Sym(li).Type())]
}

// Returns the attributes of the i-th symbol.
func (l *Loader) SymAttr(i Sym) uint8 {
	if l.IsExternal(i) {
		// TODO: do something? External symbols have different representation of attributes.
		// For now, ReflectMethod, NoSplit, GoType, and Typelink are used and they cannot be
		// set by external symbol.
		return 0
	}
	r, li := l.toLocal(i)
	return r.Sym(li).Flag()
}

// Returns the size of the i-th symbol.
func (l *Loader) SymSize(i Sym) int64 {
	if l.IsExternal(i) {
		pp := l.getPayload(i)
		return pp.size
	}
	r, li := l.toLocal(i)
	return int64(r.Sym(li).Siz())
}

// AttrReachable returns true for symbols that are transitively
// referenced from the entry points. Unreachable symbols are not
// written to the output.
func (l *Loader) AttrReachable(i Sym) bool {
	return l.attrReachable.Has(i)
}

// SetAttrReachable sets the reachability property for a symbol (see
// AttrReachable).
func (l *Loader) SetAttrReachable(i Sym, v bool) {
	if v {
		l.attrReachable.Set(i)
	} else {
		l.attrReachable.Unset(i)
	}
}

// AttrOnList returns true for symbols that are on some list (such as
// the list of all text symbols, or one of the lists of data symbols)
// and is consulted to avoid bugs where a symbol is put on a list
// twice.
func (l *Loader) AttrOnList(i Sym) bool {
	return l.attrOnList.Has(i)
}

// SetAttrOnList sets the "on list" property for a symbol (see
// AttrOnList).
func (l *Loader) SetAttrOnList(i Sym, v bool) {
	if v {
		l.attrOnList.Set(i)
	} else {
		l.attrOnList.Unset(i)
	}
}

// AttrLocal returns true for symbols that are only visible within the
// module (executable or shared library) being linked. This attribute
// is applied to thunks and certain other linker-generated symbols.
func (l *Loader) AttrLocal(i Sym) bool {
	return l.attrLocal.Has(i)
}

// SetAttrLocal the "local" property for a symbol (see AttrLocal above).
func (l *Loader) SetAttrLocal(i Sym, v bool) {
	if v {
		l.attrLocal.Set(i)
	} else {
		l.attrLocal.Unset(i)
	}
}

// SymAddr checks that a symbol is reachable, and returns its value.
func (l *Loader) SymAddr(i Sym) int64 {
	if !l.AttrReachable(i) {
		panic("unreachable symbol in symaddr")
	}
	return l.values[i]
}

// AttrNotInSymbolTable returns true for symbols that should not be
// added to the symbol table of the final generated load module.
func (l *Loader) AttrNotInSymbolTable(i Sym) bool {
	return l.attrNotInSymbolTable.Has(i)
}

// SetAttrNotInSymbolTable the "not in symtab" property for a symbol
// (see AttrNotInSymbolTable above).
func (l *Loader) SetAttrNotInSymbolTable(i Sym, v bool) {
	if v {
		l.attrNotInSymbolTable.Set(i)
	} else {
		l.attrNotInSymbolTable.Unset(i)
	}
}

// AttrVisibilityHidden symbols returns true for ELF symbols with
// visibility set to STV_HIDDEN. They become local symbols in
// the final executable. Only relevant when internally linking
// on an ELF platform.
func (l *Loader) AttrVisibilityHidden(i Sym) bool {
	if !l.IsExternal(i) {
		return false
	}
	return l.attrVisibilityHidden.Has(l.extIndex(i))
}

// SetAttrVisibilityHidden sets the "hidden visibility" property for a
// symbol (see AttrVisibilityHidden).
func (l *Loader) SetAttrVisibilityHidden(i Sym, v bool) {
	if !l.IsExternal(i) {
		panic("tried to set visibility attr on non-external symbol")
	}
	if v {
		l.attrVisibilityHidden.Set(l.extIndex(i))
	} else {
		l.attrVisibilityHidden.Unset(l.extIndex(i))
	}
}

// AttrDuplicateOK returns true for a symbol that can be present in
// multiple object files.
func (l *Loader) AttrDuplicateOK(i Sym) bool {
	if !l.IsExternal(i) {
		// TODO: if this path winds up being taken frequently, it
		// might make more sense to copy the flag value out of the object
		// into a larger bitmap during preload.
		r, li := l.toLocal(i)
		return r.Sym(li).Dupok()
	}
	return l.attrDuplicateOK.Has(l.extIndex(i))
}

// SetAttrDuplicateOK sets the "duplicate OK" property for an external
// symbol (see AttrDuplicateOK).
func (l *Loader) SetAttrDuplicateOK(i Sym, v bool) {
	if !l.IsExternal(i) {
		panic("tried to set dupok attr on non-external symbol")
	}
	if v {
		l.attrDuplicateOK.Set(l.extIndex(i))
	} else {
		l.attrDuplicateOK.Unset(l.extIndex(i))
	}
}

// AttrShared returns true for symbols compiled with the -shared option.
func (l *Loader) AttrShared(i Sym) bool {
	if !l.IsExternal(i) {
		// TODO: if this path winds up being taken frequently, it
		// might make more sense to copy the flag value out of the
		// object into a larger bitmap during preload.
		r, _ := l.toLocal(i)
		return (r.Flags() & goobj2.ObjFlagShared) != 0
	}
	return l.attrShared.Has(l.extIndex(i))
}

// SetAttrShared sets the "shared" property for an external
// symbol (see AttrShared).
func (l *Loader) SetAttrShared(i Sym, v bool) {
	if !l.IsExternal(i) {
		panic(fmt.Sprintf("tried to set shared attr on non-external symbol %d %s", i, l.SymName(i)))
	}
	if v {
		l.attrShared.Set(l.extIndex(i))
	} else {
		l.attrShared.Unset(l.extIndex(i))
	}
}

// AttrExternal returns true for function symbols loaded from host
// object files.
func (l *Loader) AttrExternal(i Sym) bool {
	if !l.IsExternal(i) {
		return false
	}
	return l.attrExternal.Has(l.extIndex(i))
}

// SetAttrExternal sets the "external" property for an host object
// symbol (see AttrExternal).
func (l *Loader) SetAttrExternal(i Sym, v bool) {
	if !l.IsExternal(i) {
		panic(fmt.Sprintf("tried to set external attr on non-external symbol %q", l.RawSymName(i)))
	}
	if v {
		l.attrExternal.Set(l.extIndex(i))
	} else {
		l.attrExternal.Unset(l.extIndex(i))
	}
}

// AttrTopFrame returns true for a function symbol that is an entry
// point, meaning that unwinders should stop when they hit this
// function.
func (l *Loader) AttrTopFrame(i Sym) bool {
	_, ok := l.attrTopFrame[i]
	return ok
}

// SetAttrTopFrame sets the "top frame" property for a symbol (see
// AttrTopFrame).
func (l *Loader) SetAttrTopFrame(i Sym, v bool) {
	if v {
		l.attrTopFrame[i] = struct{}{}
	} else {
		delete(l.attrTopFrame, i)
	}
}

// AttrSpecial returns true for a symbols that do not have their
// address (i.e. Value) computed by the usual mechanism of
// data.go:dodata() & data.go:address().
func (l *Loader) AttrSpecial(i Sym) bool {
	_, ok := l.attrSpecial[i]
	return ok
}

// SetAttrSpecial sets the "special" property for a symbol (see
// AttrSpecial).
func (l *Loader) SetAttrSpecial(i Sym, v bool) {
	if v {
		l.attrSpecial[i] = struct{}{}
	} else {
		delete(l.attrSpecial, i)
	}
}

// AttrCgoExportDynamic returns true for a symbol that has been
// specially marked via the "cgo_export_dynamic" compiler directive
// written by cgo (in response to //export directives in the source).
func (l *Loader) AttrCgoExportDynamic(i Sym) bool {
	_, ok := l.attrCgoExportDynamic[i]
	return ok
}

// SetAttrCgoExportDynamic sets the "cgo_export_dynamic" for a symbol
// (see AttrCgoExportDynamic).
func (l *Loader) SetAttrCgoExportDynamic(i Sym, v bool) {
	if v {
		l.attrCgoExportDynamic[i] = struct{}{}
	} else {
		delete(l.attrCgoExportDynamic, i)
	}
}

// AttrCgoExportStatic returns true for a symbol that has been
// specially marked via the "cgo_export_static" directive
// written by cgo.
func (l *Loader) AttrCgoExportStatic(i Sym) bool {
	_, ok := l.attrCgoExportStatic[i]
	return ok
}

// SetAttrCgoExportStatic sets the "cgo_export_static" for a symbol
// (see AttrCgoExportStatic).
func (l *Loader) SetAttrCgoExportStatic(i Sym, v bool) {
	if v {
		l.attrCgoExportStatic[i] = struct{}{}
	} else {
		delete(l.attrCgoExportStatic, i)
	}
}

func (l *Loader) AttrCgoExport(i Sym) bool {
	return l.AttrCgoExportDynamic(i) || l.AttrCgoExportStatic(i)
}

// AttrReadOnly returns true for a symbol whose underlying data
// is stored via a read-only mmap.
func (l *Loader) AttrReadOnly(i Sym) bool {
	if v, ok := l.attrReadOnly[i]; ok {
		return v
	}
	if l.IsExternal(i) {
		pp := l.getPayload(i)
		if pp.objidx != 0 {
			return l.objs[pp.objidx].r.ReadOnly()
		}
		return false
	}
	r, _ := l.toLocal(i)
	return r.ReadOnly()
}

// SetAttrReadOnly sets the "data is read only" property for a symbol
// (see AttrReadOnly).
func (l *Loader) SetAttrReadOnly(i Sym, v bool) {
	l.attrReadOnly[i] = v
}

// AttrSubSymbol returns true for symbols that are listed as a
// sub-symbol of some other outer symbol. The sub/outer mechanism is
// used when loading host objects (sections from the host object
// become regular linker symbols and symbols go on the Sub list of
// their section) and for constructing the global offset table when
// internally linking a dynamic executable.
//
// Note that in later stages of the linker, we set Outer(S) to some
// container symbol C, but don't set Sub(C). Thus we have two
// distinct scenarios:
//
// - Outer symbol covers the address ranges of its sub-symbols.
//   Outer.Sub is set in this case.
// - Outer symbol doesn't conver the address ranges. It is zero-sized
//   and doesn't have sub-symbols. In the case, the inner symbol is
//   not actually a "SubSymbol". (Tricky!)
//
// This method returns TRUE only for sub-symbols in the first scenario.
//
// FIXME: would be better to do away with this and have a better way
// to represent container symbols.

func (l *Loader) AttrSubSymbol(i Sym) bool {
	// we don't explicitly store this attribute any more -- return
	// a value based on the sub-symbol setting.
	o := l.OuterSym(i)
	if o == 0 {
		return false
	}
	return l.SubSym(o) != 0
}

// Note that we don't have a 'SetAttrSubSymbol' method in the loader;
// clients should instead use the PrependSub method to establish
// outer/sub relationships for host object symbols.

// Returns whether the i-th symbol has ReflectMethod attribute set.
func (l *Loader) IsReflectMethod(i Sym) bool {
	return l.SymAttr(i)&goobj2.SymFlagReflectMethod != 0
}

// Returns whether the i-th symbol is nosplit.
func (l *Loader) IsNoSplit(i Sym) bool {
	return l.SymAttr(i)&goobj2.SymFlagNoSplit != 0
}

// Returns whether this is a Go type symbol.
func (l *Loader) IsGoType(i Sym) bool {
	return l.SymAttr(i)&goobj2.SymFlagGoType != 0
}

// Returns whether this symbol should be included in typelink.
func (l *Loader) IsTypelink(i Sym) bool {
	return l.SymAttr(i)&goobj2.SymFlagTypelink != 0
}

// Returns whether this is a "go.itablink.*" symbol.
func (l *Loader) IsItabLink(i Sym) bool {
	if _, ok := l.itablink[i]; ok {
		return true
	}
	return false
}

// Return whether this is a trampoline of a deferreturn call.
func (l *Loader) IsDeferReturnTramp(i Sym) bool {
	return l.deferReturnTramp[i]
}

// Set that i is a trampoline of a deferreturn call.
func (l *Loader) SetIsDeferReturnTramp(i Sym, v bool) {
	l.deferReturnTramp[i] = v
}

// growValues grows the slice used to store symbol values.
func (l *Loader) growValues(reqLen int) {
	curLen := len(l.values)
	if reqLen > curLen {
		l.values = append(l.values, make([]int64, reqLen+1-curLen)...)
	}
}

// SymValue returns the value of the i-th symbol. i is global index.
func (l *Loader) SymValue(i Sym) int64 {
	return l.values[i]
}

// SetSymValue sets the value of the i-th symbol. i is global index.
func (l *Loader) SetSymValue(i Sym, val int64) {
	l.values[i] = val
}

// AddToSymValue adds to the value of the i-th symbol. i is the global index.
func (l *Loader) AddToSymValue(i Sym, val int64) {
	l.values[i] += val
}

// Returns the symbol content of the i-th symbol. i is global index.
func (l *Loader) Data(i Sym) []byte {
	if l.IsExternal(i) {
		pp := l.getPayload(i)
		if pp != nil {
			return pp.data
		}
		return nil
	}
	r, li := l.toLocal(i)
	return r.Data(li)
}

// Returns the data of the i-th symbol in the output buffer.
func (l *Loader) OutData(i Sym) []byte {
	if int(i) < len(l.outdata) && l.outdata[i] != nil {
		return l.outdata[i]
	}
	return l.Data(i)
}

// SetOutData sets the position of the data of the i-th symbol in the output buffer.
// i is global index.
func (l *Loader) SetOutData(i Sym, data []byte) {
	if l.IsExternal(i) {
		pp := l.getPayload(i)
		if pp != nil {
			pp.data = data
			return
		}
	}
	l.outdata[i] = data
}

// InitOutData initializes the slice used to store symbol output data.
func (l *Loader) InitOutData() {
	l.outdata = make([][]byte, l.extStart)
}

// SetExtRelocs sets the external relocations of the i-th symbol. i is global index.
func (l *Loader) SetExtRelocs(i Sym, relocs []ExtReloc) {
	l.extRelocs[i] = relocs
}

// InitExtRelocs initialize the slice used to store external relocations.
func (l *Loader) InitExtRelocs() {
	l.extRelocs = make([][]ExtReloc, l.NSym())
}

// SymAlign returns the alignment for a symbol.
func (l *Loader) SymAlign(i Sym) int32 {
	if int(i) >= len(l.align) {
		// align is extended lazily -- it the sym in question is
		// outside the range of the existing slice, then we assume its
		// alignment has not yet been set.
		return 0
	}
	// TODO: would it make sense to return an arch-specific
	// alignment depending on section type? E.g. STEXT => 32,
	// SDATA => 1, etc?
	abits := l.align[i]
	if abits == 0 {
		return 0
	}
	return int32(1 << (abits - 1))
}

// SetSymAlign sets the alignment for a symbol.
func (l *Loader) SetSymAlign(i Sym, align int32) {
	// Reject nonsense alignments.
	if align < 0 || align&(align-1) != 0 {
		panic("bad alignment value")
	}
	if int(i) >= len(l.align) {
		l.align = append(l.align, make([]uint8, l.NSym()-len(l.align))...)
	}
	if align == 0 {
		l.align[i] = 0
	}
	l.align[i] = uint8(bits.Len32(uint32(align)))
}

// SymValue returns the section of the i-th symbol. i is global index.
func (l *Loader) SymSect(i Sym) *sym.Section {
	if int(i) >= len(l.symSects) {
		// symSects is extended lazily -- it the sym in question is
		// outside the range of the existing slice, then we assume its
		// section has not yet been set.
		return nil
	}
	return l.sects[l.symSects[i]]
}

// SetSymValue sets the section of the i-th symbol. i is global index.
func (l *Loader) SetSymSect(i Sym, sect *sym.Section) {
	if int(i) >= len(l.symSects) {
		l.symSects = append(l.symSects, make([]uint16, l.NSym()-len(l.symSects))...)
	}
	l.symSects[i] = sect.Index
}

// growSects grows the slice used to store symbol sections.
func (l *Loader) growSects(reqLen int) {
	curLen := len(l.symSects)
	if reqLen > curLen {
		l.symSects = append(l.symSects, make([]uint16, reqLen+1-curLen)...)
	}
}

// NewSection creates a new (output) section.
func (l *Loader) NewSection() *sym.Section {
	sect := new(sym.Section)
	idx := len(l.sects)
	if idx != int(uint16(idx)) {
		panic("too many sections created")
	}
	sect.Index = uint16(idx)
	l.sects = append(l.sects, sect)
	return sect
}

// SymDynImplib returns the "dynimplib" attribute for the specified
// symbol, making up a portion of the info for a symbol specified
// on a "cgo_import_dynamic" compiler directive.
func (l *Loader) SymDynimplib(i Sym) string {
	return l.dynimplib[i]
}

// SetSymDynimplib sets the "dynimplib" attribute for a symbol.
func (l *Loader) SetSymDynimplib(i Sym, value string) {
	// reject bad symbols
	if i >= Sym(len(l.objSyms)) || i == 0 {
		panic("bad symbol index in SetDynimplib")
	}
	if value == "" {
		delete(l.dynimplib, i)
	} else {
		l.dynimplib[i] = value
	}
}

// SymDynimpvers returns the "dynimpvers" attribute for the specified
// symbol, making up a portion of the info for a symbol specified
// on a "cgo_import_dynamic" compiler directive.
func (l *Loader) SymDynimpvers(i Sym) string {
	return l.dynimpvers[i]
}

// SetSymDynimpvers sets the "dynimpvers" attribute for a symbol.
func (l *Loader) SetSymDynimpvers(i Sym, value string) {
	// reject bad symbols
	if i >= Sym(len(l.objSyms)) || i == 0 {
		panic("bad symbol index in SetDynimpvers")
	}
	if value == "" {
		delete(l.dynimpvers, i)
	} else {
		l.dynimpvers[i] = value
	}
}

// SymExtname returns the "extname" value for the specified
// symbol.
func (l *Loader) SymExtname(i Sym) string {
	if s, ok := l.extname[i]; ok {
		return s
	}
	return l.SymName(i)
}

// SetSymExtname sets the  "extname" attribute for a symbol.
func (l *Loader) SetSymExtname(i Sym, value string) {
	// reject bad symbols
	if i >= Sym(len(l.objSyms)) || i == 0 {
		panic("bad symbol index in SetExtname")
	}
	if value == "" {
		delete(l.extname, i)
	} else {
		l.extname[i] = value
	}
}

// SymElfType returns the previously recorded ELF type for a symbol
// (used only for symbols read from shared libraries by ldshlibsyms).
// It is not set for symbols defined by the packages being linked or
// by symbols read by ldelf (and so is left as elf.STT_NOTYPE).
func (l *Loader) SymElfType(i Sym) elf.SymType {
	if et, ok := l.elfType[i]; ok {
		return et
	}
	return elf.STT_NOTYPE
}

// SetSymElfType sets the elf type attribute for a symbol.
func (l *Loader) SetSymElfType(i Sym, et elf.SymType) {
	// reject bad symbols
	if i >= Sym(len(l.objSyms)) || i == 0 {
		panic("bad symbol index in SetSymElfType")
	}
	if et == elf.STT_NOTYPE {
		delete(l.elfType, i)
	} else {
		l.elfType[i] = et
	}
}

// SymElfSym returns the ELF symbol index for a given loader
// symbol, assigned during ELF symtab generation.
func (l *Loader) SymElfSym(i Sym) int32 {
	return l.elfSym[i]
}

// SetSymElfSym sets the elf symbol index for a symbol.
func (l *Loader) SetSymElfSym(i Sym, es int32) {
	if i == 0 {
		panic("bad sym index")
	}
	if es == 0 {
		delete(l.elfSym, i)
	} else {
		l.elfSym[i] = es
	}
}

// SymLocalElfSym returns the "local" ELF symbol index for a given loader
// symbol, assigned during ELF symtab generation.
func (l *Loader) SymLocalElfSym(i Sym) int32 {
	return l.localElfSym[i]
}

// SetSymLocalElfSym sets the "local" elf symbol index for a symbol.
func (l *Loader) SetSymLocalElfSym(i Sym, es int32) {
	if i == 0 {
		panic("bad sym index")
	}
	if es == 0 {
		delete(l.localElfSym, i)
	} else {
		l.localElfSym[i] = es
	}
}

// SymPlt returns the plt value for pe symbols.
func (l *Loader) SymPlt(s Sym) int32 {
	if v, ok := l.plt[s]; ok {
		return v
	}
	return -1
}

// SetPlt sets the plt value for pe symbols.
func (l *Loader) SetPlt(i Sym, v int32) {
	if i >= Sym(len(l.objSyms)) || i == 0 {
		panic("bad symbol for SetPlt")
	}
	if v == -1 {
		delete(l.plt, i)
	} else {
		l.plt[i] = v
	}
}

// SymGot returns the got value for pe symbols.
func (l *Loader) SymGot(s Sym) int32 {
	if v, ok := l.got[s]; ok {
		return v
	}
	return -1
}

// SetGot sets the got value for pe symbols.
func (l *Loader) SetGot(i Sym, v int32) {
	if i >= Sym(len(l.objSyms)) || i == 0 {
		panic("bad symbol for SetGot")
	}
	if v == -1 {
		delete(l.got, i)
	} else {
		l.got[i] = v
	}
}

// SymDynid returns the "dynid" property for the specified symbol.
func (l *Loader) SymDynid(i Sym) int32 {
	if s, ok := l.dynid[i]; ok {
		return s
	}
	return -1
}

// SetSymDynid sets the "dynid" property for a symbol.
func (l *Loader) SetSymDynid(i Sym, val int32) {
	// reject bad symbols
	if i >= Sym(len(l.objSyms)) || i == 0 {
		panic("bad symbol index in SetSymDynid")
	}
	if val == -1 {
		delete(l.dynid, i)
	} else {
		l.dynid[i] = val
	}
}

// DynIdSyms returns the set of symbols for which dynID is set to an
// interesting (non-default) value. This is expected to be a fairly
// small set.
func (l *Loader) DynidSyms() []Sym {
	sl := make([]Sym, 0, len(l.dynid))
	for s := range l.dynid {
		sl = append(sl, s)
	}
	sort.Slice(sl, func(i, j int) bool { return sl[i] < sl[j] })
	return sl
}

// SymGoType returns the 'Gotype' property for a given symbol (set by
// the Go compiler for variable symbols). This version relies on
// reading aux symbols for the target sym -- it could be that a faster
// approach would be to check for gotype during preload and copy the
// results in to a map (might want to try this at some point and see
// if it helps speed things up).
func (l *Loader) SymGoType(i Sym) Sym {
	if l.IsExternal(i) {
		pp := l.getPayload(i)
		return pp.gotype
	}
	r, li := l.toLocal(i)
	auxs := r.Auxs(li)
	for j := range auxs {
		a := &auxs[j]
		switch a.Type() {
		case goobj2.AuxGotype:
			return l.resolve(r, a.Sym())
		}
	}
	return 0
}

// SymUnit returns the compilation unit for a given symbol (which will
// typically be nil for external or linker-manufactured symbols).
func (l *Loader) SymUnit(i Sym) *sym.CompilationUnit {
	if l.IsExternal(i) {
		pp := l.getPayload(i)
		if pp.objidx != 0 {
			r := l.objs[pp.objidx].r
			return r.unit
		}
		return nil
	}
	r, _ := l.toLocal(i)
	return r.unit
}

// SymPkg returns the package where the symbol came from (for
// regular compiler-generated Go symbols), but in the case of
// building with "-linkshared" (when a symbol is read from a
// shared library), will hold the library name.
// NOTE: this correspondes to sym.Symbol.File field.
func (l *Loader) SymPkg(i Sym) string {
	if f, ok := l.symPkg[i]; ok {
		return f
	}
	if l.IsExternal(i) {
		pp := l.getPayload(i)
		if pp.objidx != 0 {
			r := l.objs[pp.objidx].r
			return r.unit.Lib.Pkg
		}
		return ""
	}
	r, _ := l.toLocal(i)
	return r.unit.Lib.Pkg
}

// SetSymPkg sets the package/library for a symbol. This is
// needed mainly for external symbols, specifically those imported
// from shared libraries.
func (l *Loader) SetSymPkg(i Sym, pkg string) {
	// reject bad symbols
	if i >= Sym(len(l.objSyms)) || i == 0 {
		panic("bad symbol index in SetSymPkg")
	}
	l.symPkg[i] = pkg
}

// SymLocalentry returns the "local entry" value for the specified
// symbol.
func (l *Loader) SymLocalentry(i Sym) uint8 {
	return l.localentry[i]
}

// SetSymLocalentry sets the "local entry" attribute for a symbol.
func (l *Loader) SetSymLocalentry(i Sym, value uint8) {
	// reject bad symbols
	if i >= Sym(len(l.objSyms)) || i == 0 {
		panic("bad symbol index in SetSymLocalentry")
	}
	if value == 0 {
		delete(l.localentry, i)
	} else {
		l.localentry[i] = value
	}
}

// Returns the number of aux symbols given a global index.
func (l *Loader) NAux(i Sym) int {
	if l.IsExternal(i) {
		return 0
	}
	r, li := l.toLocal(i)
	return r.NAux(li)
}

// Returns the "handle" to the j-th aux symbol of the i-th symbol.
func (l *Loader) Aux2(i Sym, j int) Aux2 {
	if l.IsExternal(i) {
		return Aux2{}
	}
	r, li := l.toLocal(i)
	if j >= r.NAux(li) {
		return Aux2{}
	}
	return Aux2{r.Aux(li, j), r, l}
}

// GetFuncDwarfAuxSyms collects and returns the auxiliary DWARF
// symbols associated with a given function symbol.  Prior to the
// introduction of the loader, this was done purely using name
// lookups, e.f. for function with name XYZ we would then look up
// go.info.XYZ, etc.
// FIXME: once all of dwarfgen is converted over to the loader,
// it would save some space to make these aux symbols nameless.
func (l *Loader) GetFuncDwarfAuxSyms(fnSymIdx Sym) (auxDwarfInfo, auxDwarfLoc, auxDwarfRanges, auxDwarfLines Sym) {
	if l.SymType(fnSymIdx) != sym.STEXT {
		log.Fatalf("error: non-function sym %d/%s t=%s passed to GetFuncDwarfAuxSyms", fnSymIdx, l.SymName(fnSymIdx), l.SymType(fnSymIdx).String())
	}
	if l.IsExternal(fnSymIdx) {
		// Current expectation is that any external function will
		// not have auxsyms.
		return
	}
	r, li := l.toLocal(fnSymIdx)
	auxs := r.Auxs(li)
	for i := range auxs {
		a := &auxs[i]
		switch a.Type() {
		case goobj2.AuxDwarfInfo:
			auxDwarfInfo = l.resolve(r, a.Sym())
			if l.SymType(auxDwarfInfo) != sym.SDWARFINFO {
				panic("aux dwarf info sym with wrong type")
			}
		case goobj2.AuxDwarfLoc:
			auxDwarfLoc = l.resolve(r, a.Sym())
			if l.SymType(auxDwarfLoc) != sym.SDWARFLOC {
				panic("aux dwarf loc sym with wrong type")
			}
		case goobj2.AuxDwarfRanges:
			auxDwarfRanges = l.resolve(r, a.Sym())
			if l.SymType(auxDwarfRanges) != sym.SDWARFRANGE {
				panic("aux dwarf ranges sym with wrong type")
			}
		case goobj2.AuxDwarfLines:
			auxDwarfLines = l.resolve(r, a.Sym())
			if l.SymType(auxDwarfLines) != sym.SDWARFLINES {
				panic("aux dwarf lines sym with wrong type")
			}
		}
	}
	return
}

// PrependSub prepends 'sub' onto the sub list for outer symbol 'outer'.
// Will panic if 'sub' already has an outer sym or sub sym.
// FIXME: should this be instead a method on SymbolBuilder?
func (l *Loader) PrependSub(outer Sym, sub Sym) {
	// NB: this presupposes that an outer sym can't be a sub symbol of
	// some other outer-outer sym (I'm assuming this is true, but I
	// haven't tested exhaustively).
	if l.OuterSym(outer) != 0 {
		panic("outer has outer itself")
	}
	if l.SubSym(sub) != 0 {
		panic("sub set for subsym")
	}
	if l.OuterSym(sub) != 0 {
		panic("outer already set for subsym")
	}
	l.sub[sub] = l.sub[outer]
	l.sub[outer] = sub
	l.outer[sub] = outer
}

// OuterSym gets the outer symbol for host object loaded symbols.
func (l *Loader) OuterSym(i Sym) Sym {
	// FIXME: add check for isExternal?
	return l.outer[i]
}

// SubSym gets the subsymbol for host object loaded symbols.
func (l *Loader) SubSym(i Sym) Sym {
	// NB: note -- no check for l.isExternal(), since I am pretty sure
	// that later phases in the linker set subsym for "type." syms
	return l.sub[i]
}

// SetOuterSym sets the outer symbol of i to o (without setting
// sub symbols).
func (l *Loader) SetOuterSym(i Sym, o Sym) {
	if o != 0 {
		l.outer[i] = o
	} else {
		delete(l.outer, i)
	}
}

// Initialize Reachable bitmap and its siblings for running deadcode pass.
func (l *Loader) InitReachable() {
	l.growAttrBitmaps(l.NSym() + 1)
}

type symWithVal struct {
	s Sym
	v int64
}
type bySymValue []symWithVal

func (s bySymValue) Len() int           { return len(s) }
func (s bySymValue) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
func (s bySymValue) Less(i, j int) bool { return s[i].v < s[j].v }

// SortSub walks through the sub-symbols for 's' and sorts them
// in place by increasing value. Return value is the new
// sub symbol for the specified outer symbol.
func (l *Loader) SortSub(s Sym) Sym {

	if s == 0 || l.sub[s] == 0 {
		return s
	}

	// Sort symbols using a slice first. Use a stable sort on the off
	// chance that there's more than once symbol with the same value,
	// so as to preserve reproducible builds.
	sl := []symWithVal{}
	for ss := l.sub[s]; ss != 0; ss = l.sub[ss] {
		sl = append(sl, symWithVal{s: ss, v: l.SymValue(ss)})
	}
	sort.Stable(bySymValue(sl))

	// Then apply any changes needed to the sub map.
	ns := Sym(0)
	for i := len(sl) - 1; i >= 0; i-- {
		s := sl[i].s
		l.sub[s] = ns
		ns = s
	}

	// Update sub for outer symbol, then return
	l.sub[s] = sl[0].s
	return sl[0].s
}

// Insure that reachable bitmap and its siblings have enough size.
func (l *Loader) growAttrBitmaps(reqLen int) {
	if reqLen > l.attrReachable.Len() {
		// These are indexed by global symbol
		l.attrReachable = growBitmap(reqLen, l.attrReachable)
		l.attrOnList = growBitmap(reqLen, l.attrOnList)
		l.attrLocal = growBitmap(reqLen, l.attrLocal)
		l.attrNotInSymbolTable = growBitmap(reqLen, l.attrNotInSymbolTable)
	}
	l.growExtAttrBitmaps()
}

func (l *Loader) growExtAttrBitmaps() {
	// These are indexed by external symbol index (e.g. l.extIndex(i))
	extReqLen := len(l.payloads)
	if extReqLen > l.attrVisibilityHidden.Len() {
		l.attrVisibilityHidden = growBitmap(extReqLen, l.attrVisibilityHidden)
		l.attrDuplicateOK = growBitmap(extReqLen, l.attrDuplicateOK)
		l.attrShared = growBitmap(extReqLen, l.attrShared)
		l.attrExternal = growBitmap(extReqLen, l.attrExternal)
	}
}

func (relocs *Relocs) Count() int { return len(relocs.rs) }

// At2 returns the j-th reloc for a global symbol.
func (relocs *Relocs) At2(j int) Reloc2 {
	if relocs.l.isExtReader(relocs.r) {
		pp := relocs.l.payloads[relocs.li]
		return Reloc2{&relocs.rs[j], relocs.r, relocs.l, pp.reltypes[j]}
	}
	return Reloc2{&relocs.rs[j], relocs.r, relocs.l, 0}
}

// Relocs returns a Relocs object for the given global sym.
func (l *Loader) Relocs(i Sym) Relocs {
	r, li := l.toLocal(i)
	if r == nil {
		panic(fmt.Sprintf("trying to get oreader for invalid sym %d\n\n", i))
	}
	return l.relocs(r, li)
}

// Relocs returns a Relocs object given a local sym index and reader.
func (l *Loader) relocs(r *oReader, li int) Relocs {
	var rs []goobj2.Reloc
	if l.isExtReader(r) {
		pp := l.payloads[li]
		rs = pp.relocs
	} else {
		rs = r.Relocs(li)
	}
	return Relocs{
		rs: rs,
		li: li,
		r:  r,
		l:  l,
	}
}

// ExtRelocs returns the external relocations of the i-th symbol.
func (l *Loader) ExtRelocs(i Sym) ExtRelocs {
	return ExtRelocs{l.Relocs(i), l.extRelocs[i]}
}

// ExtRelocs represents the set of external relocations of a symbol.
type ExtRelocs struct {
	rs Relocs
	es []ExtReloc
}

func (ers ExtRelocs) Count() int { return len(ers.es) }

func (ers ExtRelocs) At(j int) ExtRelocView {
	i := ers.es[j].Idx
	return ExtRelocView{ers.rs.At2(i), &ers.es[j]}
}

// RelocByOff implements sort.Interface for sorting relocations by offset.

type RelocByOff []Reloc

func (x RelocByOff) Len() int           { return len(x) }
func (x RelocByOff) Swap(i, j int)      { x[i], x[j] = x[j], x[i] }
func (x RelocByOff) Less(i, j int) bool { return x[i].Off < x[j].Off }

// FuncInfo provides hooks to access goobj2.FuncInfo in the objects.
type FuncInfo struct {
	l       *Loader
	r       *oReader
	data    []byte
	auxs    []goobj2.Aux
	lengths goobj2.FuncInfoLengths
}

func (fi *FuncInfo) Valid() bool { return fi.r != nil }

func (fi *FuncInfo) Args() int {
	return int((*goobj2.FuncInfo)(nil).ReadArgs(fi.data))
}

func (fi *FuncInfo) Locals() int {
	return int((*goobj2.FuncInfo)(nil).ReadLocals(fi.data))
}

func (fi *FuncInfo) Pcsp() []byte {
	pcsp, end := (*goobj2.FuncInfo)(nil).ReadPcsp(fi.data)
	return fi.r.BytesAt(fi.r.PcdataBase()+pcsp, int(end-pcsp))
}

func (fi *FuncInfo) Pcfile() []byte {
	pcf, end := (*goobj2.FuncInfo)(nil).ReadPcfile(fi.data)
	return fi.r.BytesAt(fi.r.PcdataBase()+pcf, int(end-pcf))
}

func (fi *FuncInfo) Pcline() []byte {
	pcln, end := (*goobj2.FuncInfo)(nil).ReadPcline(fi.data)
	return fi.r.BytesAt(fi.r.PcdataBase()+pcln, int(end-pcln))
}

// Preload has to be called prior to invoking the various methods
// below related to pcdata, funcdataoff, files, and inltree nodes.
func (fi *FuncInfo) Preload() {
	fi.lengths = (*goobj2.FuncInfo)(nil).ReadFuncInfoLengths(fi.data)
}

func (fi *FuncInfo) Pcinline() []byte {
	if !fi.lengths.Initialized {
		panic("need to call Preload first")
	}
	pcinl, end := (*goobj2.FuncInfo)(nil).ReadPcinline(fi.data, fi.lengths.PcdataOff)
	return fi.r.BytesAt(fi.r.PcdataBase()+pcinl, int(end-pcinl))
}

func (fi *FuncInfo) NumPcdata() uint32 {
	if !fi.lengths.Initialized {
		panic("need to call Preload first")
	}
	return fi.lengths.NumPcdata
}

func (fi *FuncInfo) Pcdata(k int) []byte {
	if !fi.lengths.Initialized {
		panic("need to call Preload first")
	}
	pcdat, end := (*goobj2.FuncInfo)(nil).ReadPcdata(fi.data, fi.lengths.PcdataOff, uint32(k))
	return fi.r.BytesAt(fi.r.PcdataBase()+pcdat, int(end-pcdat))
}

func (fi *FuncInfo) NumFuncdataoff() uint32 {
	if !fi.lengths.Initialized {
		panic("need to call Preload first")
	}
	return fi.lengths.NumFuncdataoff
}

func (fi *FuncInfo) Funcdataoff(k int) int64 {
	if !fi.lengths.Initialized {
		panic("need to call Preload first")
	}
	return (*goobj2.FuncInfo)(nil).ReadFuncdataoff(fi.data, fi.lengths.FuncdataoffOff, uint32(k))
}

func (fi *FuncInfo) Funcdata(syms []Sym) []Sym {
	if !fi.lengths.Initialized {
		panic("need to call Preload first")
	}
	if int(fi.lengths.NumFuncdataoff) > cap(syms) {
		syms = make([]Sym, 0, fi.lengths.NumFuncdataoff)
	} else {
		syms = syms[:0]
	}
	for j := range fi.auxs {
		a := &fi.auxs[j]
		if a.Type() == goobj2.AuxFuncdata {
			syms = append(syms, fi.l.resolve(fi.r, a.Sym()))
		}
	}
	return syms
}

func (fi *FuncInfo) NumFile() uint32 {
	if !fi.lengths.Initialized {
		panic("need to call Preload first")
	}
	return fi.lengths.NumFile
}

func (fi *FuncInfo) File(k int) Sym {
	if !fi.lengths.Initialized {
		panic("need to call Preload first")
	}
	sr := (*goobj2.FuncInfo)(nil).ReadFile(fi.data, fi.lengths.FileOff, uint32(k))
	return fi.l.resolve(fi.r, sr)
}

type InlTreeNode struct {
	Parent   int32
	File     Sym
	Line     int32
	Func     Sym
	ParentPC int32
}

func (fi *FuncInfo) NumInlTree() uint32 {
	if !fi.lengths.Initialized {
		panic("need to call Preload first")
	}
	return fi.lengths.NumInlTree
}

func (fi *FuncInfo) InlTree(k int) InlTreeNode {
	if !fi.lengths.Initialized {
		panic("need to call Preload first")
	}
	node := (*goobj2.FuncInfo)(nil).ReadInlTree(fi.data, fi.lengths.InlTreeOff, uint32(k))
	return InlTreeNode{
		Parent:   node.Parent,
		File:     fi.l.resolve(fi.r, node.File),
		Line:     node.Line,
		Func:     fi.l.resolve(fi.r, node.Func),
		ParentPC: node.ParentPC,
	}
}

func (l *Loader) FuncInfo(i Sym) FuncInfo {
	var r *oReader
	var auxs []goobj2.Aux
	if l.IsExternal(i) {
		pp := l.getPayload(i)
		if pp.objidx == 0 {
			return FuncInfo{}
		}
		r = l.objs[pp.objidx].r
		auxs = pp.auxs
	} else {
		var li int
		r, li = l.toLocal(i)
		auxs = r.Auxs(li)
	}
	for j := range auxs {
		a := &auxs[j]
		if a.Type() == goobj2.AuxFuncInfo {
			b := r.Data(int(a.Sym().SymIdx))
			return FuncInfo{l, r, b, auxs, goobj2.FuncInfoLengths{}}
		}
	}
	return FuncInfo{}
}

// Preload a package: add autolibs, add defined package symbols to the symbol table.
// Does not add non-package symbols yet, which will be done in LoadNonpkgSyms.
// Does not read symbol data.
// Returns the fingerprint of the object.
func (l *Loader) Preload(syms *sym.Symbols, f *bio.Reader, lib *sym.Library, unit *sym.CompilationUnit, length int64) goobj2.FingerprintType {
	roObject, readonly, err := f.Slice(uint64(length)) // TODO: no need to map blocks that are for tools only (e.g. RefName)
	if err != nil {
		log.Fatal("cannot read object file:", err)
	}
	r := goobj2.NewReaderFromBytes(roObject, readonly)
	if r == nil {
		if len(roObject) >= 8 && bytes.Equal(roObject[:8], []byte("\x00go114ld")) {
			log.Fatalf("found object file %s in old format, but -go115newobj is true\nset -go115newobj consistently in all -gcflags, -asmflags, and -ldflags", f.File().Name())
		}
		panic("cannot read object file")
	}
	localSymVersion := syms.IncVersion()
	pkgprefix := objabi.PathToPrefix(lib.Pkg) + "."
	ndef := r.NSym()
	nnonpkgdef := r.NNonpkgdef()
	or := &oReader{r, unit, localSymVersion, r.Flags(), pkgprefix, make([]Sym, ndef+nnonpkgdef+r.NNonpkgref()), ndef, uint32(len(l.objs))}

	// Autolib
	lib.Autolib = append(lib.Autolib, r.Autolib()...)

	// DWARF file table
	nfile := r.NDwarfFile()
	unit.DWARFFileTable = make([]string, nfile)
	for i := range unit.DWARFFileTable {
		unit.DWARFFileTable[i] = r.DwarfFile(i)
	}

	l.addObj(lib.Pkg, or)
	l.preloadSyms(or, pkgDef)

	// The caller expects us consuming all the data
	f.MustSeek(length, os.SEEK_CUR)

	return r.Fingerprint()
}

// Preload symbols of given kind from an object.
func (l *Loader) preloadSyms(r *oReader, kind int) {
	ndef := r.NSym()
	nnonpkgdef := r.NNonpkgdef()
	var start, end int
	switch kind {
	case pkgDef:
		start = 0
		end = ndef
	case nonPkgDef:
		start = ndef
		end = ndef + nnonpkgdef
	default:
		panic("preloadSyms: bad kind")
	}
	l.growSyms(len(l.objSyms) + end - start)
	l.growAttrBitmaps(len(l.objSyms) + end - start)
	for i := start; i < end; i++ {
		osym := r.Sym(i)
		name := strings.Replace(osym.Name(r.Reader), "\"\".", r.pkgprefix, -1)
		v := abiToVer(osym.ABI(), r.version)
		dupok := osym.Dupok()
		gi, added := l.AddSym(name, v, r, i, kind, dupok, sym.AbiSymKindToSymKind[objabi.SymKind(osym.Type())])
		r.syms[i] = gi
		if !added {
			continue
		}
		if osym.TopFrame() {
			l.SetAttrTopFrame(gi, true)
		}
		if osym.Local() {
			l.SetAttrLocal(gi, true)
		}
		if strings.HasPrefix(name, "go.itablink.") {
			l.itablink[gi] = struct{}{}
		}
		if strings.HasPrefix(name, "runtime.") {
			if bi := goobj2.BuiltinIdx(name, v); bi != -1 {
				// This is a definition of a builtin symbol. Record where it is.
				l.builtinSyms[bi] = gi
			}
		}
		if strings.HasPrefix(name, "go.string.") ||
			strings.HasPrefix(name, "gclocals·") ||
			strings.HasPrefix(name, "runtime.gcbits.") {
			l.SetAttrNotInSymbolTable(gi, true)
		}
		if a := osym.Align(); a != 0 {
			l.SetSymAlign(gi, int32(a))
		}
	}
}

// Add non-package symbols and references to external symbols (which are always
// named).
func (l *Loader) LoadNonpkgSyms(arch *sys.Arch) {
	for _, o := range l.objs[1:] {
		l.preloadSyms(o.r, nonPkgDef)
	}
	for _, o := range l.objs[1:] {
		loadObjRefs(l, o.r, arch)
	}
}

func loadObjRefs(l *Loader, r *oReader, arch *sys.Arch) {
	ndef := r.NSym() + r.NNonpkgdef()
	for i, n := 0, r.NNonpkgref(); i < n; i++ {
		osym := r.Sym(ndef + i)
		name := strings.Replace(osym.Name(r.Reader), "\"\".", r.pkgprefix, -1)
		v := abiToVer(osym.ABI(), r.version)
		r.syms[ndef+i] = l.LookupOrCreateSym(name, v)
		gi := r.syms[ndef+i]
		if osym.Local() {
			l.SetAttrLocal(gi, true)
		}
		l.preprocess(arch, gi, name)
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

// preprocess looks for integer/floating point constant symbols whose
// content is encoded into the symbol name, and promotes them into
// real symbols with RODATA type and a payload that matches the
// encoded content.
func (l *Loader) preprocess(arch *sys.Arch, s Sym, name string) {
	if name != "" && name[0] == '$' && len(name) > 5 && l.SymType(s) == 0 && len(l.Data(s)) == 0 {
		x, err := strconv.ParseUint(name[5:], 16, 64)
		if err != nil {
			log.Panicf("failed to parse $-symbol %s: %v", name, err)
		}
		su := l.MakeSymbolUpdater(s)
		su.SetType(sym.SRODATA)
		su.SetLocal(true)
		switch name[:5] {
		case "$f32.":
			if uint64(uint32(x)) != x {
				log.Panicf("$-symbol %s too large: %d", name, x)
			}
			su.AddUint32(arch, uint32(x))
		case "$f64.", "$i64.":
			su.AddUint64(arch, x)
		default:
			log.Panicf("unrecognized $-symbol: %s", name)
		}
	}
}

// Load full contents.
func (l *Loader) LoadFull(arch *sys.Arch, syms *sym.Symbols, needReloc, needExtReloc bool) {
	// create all Symbols first.
	l.growSyms(l.NSym())
	l.growSects(l.NSym())

	if needReloc && len(l.extRelocs) != 0 {
		// If needReloc is true, we are going to convert the loader's
		// "internal" relocations to sym.Relocs. In this case, external
		// relocations shouldn't be used.
		panic("phase error")
	}

	nr := 0 // total number of sym.Reloc's we'll need
	for _, o := range l.objs[1:] {
		nr += loadObjSyms(l, syms, o.r, needReloc, needExtReloc)
	}

	// Make a first pass through the external symbols, making
	// sure that each external symbol has a non-nil entry in
	// l.Syms (note that relocations and symbol content will
	// be copied in a later loop).
	toConvert := make([]Sym, 0, len(l.payloads))
	for _, i := range l.extReader.syms {
		if !l.attrReachable.Has(i) {
			continue
		}
		pp := l.getPayload(i)
		if needReloc {
			nr += len(pp.relocs)
		}
		if needExtReloc && int(i) < len(l.extRelocs) {
			nr += len(l.extRelocs[i])
		}
		// create and install the sym.Symbol here so that l.Syms will
		// be fully populated when we do relocation processing and
		// outer/sub processing below. Note that once we do this,
		// we'll need to get at the payload for a symbol with direct
		// reference to l.payloads[] as opposed to calling l.getPayload().
		s := l.allocSym(pp.name, 0)
		l.installSym(i, s)
		toConvert = append(toConvert, i)
	}

	// allocate a single large slab of relocations for all live symbols
	if nr != 0 {
		l.relocBatch = make([]sym.Reloc, nr)
		if needExtReloc {
			l.relocExtBatch = make([]sym.RelocExt, nr)
		}
	}

	// convert payload-based external symbols into sym.Symbol-based
	for _, i := range toConvert {

		// Copy kind/size/value etc.
		pp := l.payloads[l.extIndex(i)]
		s := l.Syms[i]
		s.Version = int16(pp.ver)
		s.Type = pp.kind
		s.Size = pp.size

		// Copy relocations
		if needReloc {
			batch := l.relocBatch
			s.R = batch[:len(pp.relocs):len(pp.relocs)]
			l.relocBatch = batch[len(pp.relocs):]
			relocs := l.Relocs(i)
			l.convertRelocations(i, &relocs, s, false)
		}
		if needExtReloc {
			l.convertExtRelocs(s, i)
		}

		// Copy data
		s.P = pp.data

		// Transfer over attributes.
		l.migrateAttributes(i, s)
	}

	// load contents of defined symbols
	for _, o := range l.objs[1:] {
		loadObjFull(l, o.r, needReloc, needExtReloc)
	}

	// Sanity check: we should have consumed all batched allocations.
	if len(l.relocBatch) != 0 || len(l.relocExtBatch) != 0 {
		panic("batch allocation mismatch")
	}

	// Note: resolution of ABI aliases is now also handled in
	// loader.convertRelocations, so once the host object loaders move
	// completely to loader.Sym, we can remove the code below.

	// Resolve ABI aliases for external symbols. This is only
	// needed for internal cgo linking.
	if needReloc {
		for _, i := range l.extReader.syms {
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

	// Free some memory.
	// At this point we still need basic index mapping, and some fields of
	// external symbol payloads, but not much else.
	l.values = nil
	l.symSects = nil
	l.outdata = nil
	l.itablink = nil
	l.attrOnList = nil
	l.attrLocal = nil
	l.attrNotInSymbolTable = nil
	l.attrVisibilityHidden = nil
	l.attrDuplicateOK = nil
	l.attrShared = nil
	l.attrExternal = nil
	l.attrReadOnly = nil
	l.attrTopFrame = nil
	l.attrSpecial = nil
	l.attrCgoExportDynamic = nil
	l.attrCgoExportStatic = nil
	l.outer = nil
	l.align = nil
	l.dynimplib = nil
	l.dynimpvers = nil
	l.localentry = nil
	l.extname = nil
	l.elfType = nil
	l.plt = nil
	l.got = nil
	l.dynid = nil
	if needExtReloc { // converted to sym.Relocs, drop loader references
		l.relocVariant = nil
		l.extRelocs = nil
	}

	// Drop fields that are no longer needed.
	for _, i := range l.extReader.syms {
		pp := l.getPayload(i)
		pp.name = ""
		pp.auxs = nil
		pp.data = nil
		if needExtReloc {
			pp.relocs = nil
			pp.reltypes = nil
		}
	}
}

// ResolveABIAlias given a symbol returns the ABI alias target of that
// symbol. If the sym in question is not an alias, the sym itself is
// returned.
func (l *Loader) ResolveABIAlias(s Sym) Sym {
	if s == 0 {
		return 0
	}
	if l.SymType(s) != sym.SABIALIAS {
		return s
	}
	relocs := l.Relocs(s)
	target := relocs.At2(0).Sym()
	if l.SymType(target) == sym.SABIALIAS {
		panic(fmt.Sprintf("ABI alias %s references another ABI alias %s", l.SymName(s), l.SymName(target)))
	}
	return target
}

// PropagateSymbolChangesBackToLoader is a temporary shim function
// that copies over a given sym.Symbol into the equivalent representation
// in the loader world. The intent is to enable converting a given
// linker phase/pass from dealing with sym.Symbol's to a modernized
// pass that works with loader.Sym, in cases where the "loader.Sym
// wavefront" has not yet reached the pass in question. For such work
// the recipe is to first call PropagateSymbolChangesBackToLoader(),
// then exexute the pass working with the loader, then call
// PropagateLoaderChangesToSymbols to copy the changes made by the
// pass back to the sym.Symbol world.
func (l *Loader) PropagateSymbolChangesBackToLoader() {

	// For the moment we only copy symbol values, and we don't touch
	// any new sym.Symbols created since loadlibfull() was run. This
	// seems to be what's needed for DWARF gen.
	for i := Sym(1); i < Sym(len(l.objSyms)); i++ {
		s := l.Syms[i]
		if s != nil {
			if s.Value != l.SymValue(i) {
				l.SetSymValue(i, s.Value)
			}
		}
	}
}

// PropagateLoaderChangesToSymbols is a temporary shim function that
// takes a list of loader.Sym symbols and works to copy their contents
// and attributes over to a corresponding sym.Symbol. The parameter
// anonVerReplacement specifies a version number for any new anonymous
// symbols encountered on the list, when creating sym.Symbols for them
// (or zero if we don't expect to encounter any new anon symbols). See
// the PropagateSymbolChangesBackToLoader header comment for more
// info.
//
// WARNING: this function is brittle and depends heavily on loader
// implementation. A key problem with doing this is that as things
// stand at the moment, some sym.Symbol contents/attributes are
// populated only when converting from loader.Sym to sym.Symbol in
// loadlibfull, meaning we may wipe out some information when copying
// back.

func (l *Loader) PropagateLoaderChangesToSymbols(toconvert []Sym, anonVerReplacement int) []*sym.Symbol {

	result := []*sym.Symbol{}
	relocfixup := []Sym{}

	// Note: this loop needs to allow for the possibility that we may
	// see "new" symbols on the 'toconvert' list that come from object
	// files (for example, DWARF location lists), as opposed to just
	// newly manufactured symbols (ex: DWARF section symbols such as
	// ".debug_info").  This means that we have to be careful not to
	// stomp on sym.Symbol attributes/content that was set up in
	// in loadlibfull().

	// Also note that in order for the relocation fixup to work, we
	// have to do this in two passes -- one pass to create the symbols,
	// and then a second fix up the relocations once all necessary
	// sym.Symbols are created.

	// First pass, symbol creation and symbol data fixup.
	for _, cand := range toconvert {

		sn := l.SymName(cand)
		sv := l.SymVersion(cand)
		st := l.SymType(cand)
		if sv < 0 {
			if anonVerReplacement == 0 {
				panic("expected valid anon version replacement")
			}
			sv = anonVerReplacement
		}

		s := l.Syms[cand]

		isnew := false
		if sn == "" {
			// Don't install anonymous symbols in the lookup tab.
			if s == nil {
				s = l.allocSym(sn, sv)
				l.installSym(cand, s)
			}
			isnew = true
		} else {
			if s != nil {
				// Already have a symbol for this -- it must be
				// something that was previously processed by
				// loadObjFull. Note that the symbol in question may
				// or may not be in the name lookup map.
			} else {
				isnew = true
				s = l.SymLookup(sn, sv)
			}
		}
		result = append(result, s)

		// Always copy these from new to old.
		s.Value = l.SymValue(cand)
		s.Type = st

		// If the data for a symbol has increased in size, make sure
		// we bring the new content across.
		relfix := isnew
		if isnew || len(l.Data(cand)) > len(s.P) {
			s.P = l.Data(cand)
			s.Size = int64(len(s.P))
			relfix = true
		}

		// For 'new' symbols, copy other content.
		if relfix {
			relocfixup = append(relocfixup, cand)
		}

		// If new symbol, call a helper to migrate attributes.
		// Otherwise touch only not-in-symbol-table, since there are
		// some attrs that are only set up at the point where we
		// convert loader.Sym to sym.Symbol.
		if isnew {
			l.migrateAttributes(cand, s)
		} else {
			if l.AttrNotInSymbolTable(cand) {
				s.Attr.Set(sym.AttrNotInSymbolTable, true)
			}
		}
	}

	// Second pass to fix up relocations.
	for _, cand := range relocfixup {
		s := l.Syms[cand]
		relocs := l.Relocs(cand)
		if len(s.R) != relocs.Count() {
			s.R = make([]sym.Reloc, relocs.Count())
		}
		l.convertRelocations(cand, &relocs, s, true)
	}

	return result
}

// ExtractSymbols grabs the symbols out of the loader for work that hasn't been
// ported to the new symbol type.
func (l *Loader) ExtractSymbols(syms *sym.Symbols) {
	// Add symbols to the ctxt.Syms lookup table. This explicitly skips things
	// created via loader.Create (marked with versions less than zero), since
	// if we tried to add these we'd wind up with collisions. We do, however,
	// add these symbols to the list of global symbols so that other future
	// steps (like pclntab generation) can find these symbols if neceassary.
	// Along the way, update the version from the negative anon version to
	// something larger than sym.SymVerStatic (needed so that
	// sym.symbol.IsFileLocal() works properly).
	anonVerReplacement := syms.IncVersion()
	for _, s := range l.Syms {
		if s == nil {
			continue
		}
		if s.Version < 0 {
			s.Version = int16(anonVerReplacement)
		}
	}

	// Provide lookup functions for sym.Symbols.
	l.SymLookup = func(name string, ver int) *sym.Symbol {
		i := l.LookupOrCreateSym(name, ver)
		if s := l.Syms[i]; s != nil {
			return s
		}
		s := l.allocSym(name, ver)
		l.installSym(i, s)
		return s
	}
	syms.Lookup = l.SymLookup
	syms.ROLookup = func(name string, ver int) *sym.Symbol {
		i := l.Lookup(name, ver)
		return l.Syms[i]
	}
}

// allocSym allocates a new symbol backing.
func (l *Loader) allocSym(name string, version int) *sym.Symbol {
	batch := l.symBatch
	if len(batch) == 0 {
		batch = make([]sym.Symbol, 1000)
	}
	s := &batch[0]
	l.symBatch = batch[1:]

	s.Dynid = -1
	s.Name = name
	s.Version = int16(version)

	return s
}

// installSym sets the underlying sym.Symbol for the specified sym index.
func (l *Loader) installSym(i Sym, s *sym.Symbol) {
	if s == nil {
		panic("installSym nil symbol")
	}
	if l.Syms[i] != nil {
		panic("sym already present in installSym")
	}
	l.Syms[i] = s
	s.SymIdx = sym.LoaderSym(i)
}

// addNewSym adds a new sym.Symbol to the i-th index in the list of symbols.
func (l *Loader) addNewSym(i Sym, name string, ver int, unit *sym.CompilationUnit, t sym.SymKind) *sym.Symbol {
	s := l.allocSym(name, ver)
	if s.Type != 0 && s.Type != sym.SXREF {
		fmt.Println("symbol already processed:", unit.Lib, i, s)
		panic("symbol already processed")
	}
	if t == sym.SBSS && (s.Type == sym.SRODATA || s.Type == sym.SNOPTRBSS) {
		t = s.Type
	}
	s.Type = t
	l.growSyms(int(i))
	l.installSym(i, s)
	return s
}

// TopLevelSym tests a symbol (by name and kind) to determine whether
// the symbol first class sym (participating in the link) or is an
// anonymous aux or sub-symbol containing some sub-part or payload of
// another symbol.
func (l *Loader) TopLevelSym(s Sym) bool {
	return topLevelSym(l.RawSymName(s), l.SymType(s))
}

// topLevelSym tests a symbol name and kind to determine whether
// the symbol first class sym (participating in the link) or is an
// anonymous aux or sub-symbol containing some sub-part or payload of
// another symbol.
func topLevelSym(sname string, skind sym.SymKind) bool {
	if sname != "" {
		return true
	}
	switch skind {
	case sym.SDWARFINFO, sym.SDWARFRANGE, sym.SDWARFLOC, sym.SDWARFLINES, sym.SGOFUNC:
		return true
	default:
		return false
	}
}

// loadObjSyms creates sym.Symbol objects for the live Syms in the
// object corresponding to object reader "r". Return value is the
// number of sym.Reloc entries required for all the new symbols.
func loadObjSyms(l *Loader, syms *sym.Symbols, r *oReader, needReloc, needExtReloc bool) int {
	nr := 0
	for i, n := 0, r.NSym()+r.NNonpkgdef(); i < n; i++ {
		gi := r.syms[i]
		if r2, i2 := l.toLocal(gi); r2 != r || i2 != i {
			continue // come from a different object
		}
		osym := r.Sym(i)
		name := strings.Replace(osym.Name(r.Reader), "\"\".", r.pkgprefix, -1)
		t := sym.AbiSymKindToSymKind[objabi.SymKind(osym.Type())]

		// Skip non-dwarf anonymous symbols (e.g. funcdata),
		// since they will never be turned into sym.Symbols.
		if !topLevelSym(name, t) {
			continue
		}
		ver := abiToVer(osym.ABI(), r.version)
		if t == sym.SXREF {
			log.Fatalf("bad sxref")
		}
		if t == 0 {
			log.Fatalf("missing type for %s in %s", name, r.unit.Lib)
		}
		if !l.attrReachable.Has(gi) && name != "runtime.addmoduledata" && name != "runtime.lastmoduledatap" {
			// No need to load unreachable symbols.
			// XXX reference to runtime.addmoduledata may be generated later by the linker in plugin mode.
			continue
		}

		l.addNewSym(gi, name, ver, r.unit, t)
		if needReloc {
			nr += r.NReloc(i)
		}
		if needExtReloc && int(gi) < len(l.extRelocs) {
			nr += len(l.extRelocs[gi])
		}
	}
	return nr
}

// cloneToExternal takes the existing object file symbol (symIdx)
// and creates a new external symbol payload that is a clone with
// respect to name, version, type, relocations, etc. The idea here
// is that if the linker decides it wants to update the contents of
// a symbol originally discovered as part of an object file, it's
// easier to do this if we make the updates to an external symbol
// payload.
// XXX maybe rename? makeExtPayload?
func (l *Loader) cloneToExternal(symIdx Sym) {
	if l.IsExternal(symIdx) {
		panic("sym is already external, no need for clone")
	}
	l.growSyms(int(symIdx))

	// Read the particulars from object.
	r, li := l.toLocal(symIdx)
	osym := r.Sym(li)
	sname := strings.Replace(osym.Name(r.Reader), "\"\".", r.pkgprefix, -1)
	sver := abiToVer(osym.ABI(), r.version)
	skind := sym.AbiSymKindToSymKind[objabi.SymKind(osym.Type())]

	// Create new symbol, update version and kind.
	pi := l.newPayload(sname, sver)
	pp := l.payloads[pi]
	pp.kind = skind
	pp.ver = sver
	pp.size = int64(osym.Siz())
	pp.objidx = r.objidx

	// If this is a def, then copy the guts. We expect this case
	// to be very rare (one case it may come up is with -X).
	if li < (r.NSym() + r.NNonpkgdef()) {

		// Copy relocations
		relocs := l.Relocs(symIdx)
		pp.relocs = make([]goobj2.Reloc, relocs.Count())
		pp.reltypes = make([]objabi.RelocType, relocs.Count())
		for i := range pp.relocs {
			// Copy the relocs slice.
			// Convert local reference to global reference.
			rel := relocs.At2(i)
			pp.relocs[i].Set(rel.Off(), rel.Siz(), 0, rel.Add(), goobj2.SymRef{PkgIdx: 0, SymIdx: uint32(rel.Sym())})
			pp.reltypes[i] = rel.Type()
		}

		// Copy data
		pp.data = r.Data(li)
	}

	// If we're overriding a data symbol, collect the associated
	// Gotype, so as to propagate it to the new symbol.
	auxs := r.Auxs(li)
	pp.auxs = auxs
loop:
	for j := range auxs {
		a := &auxs[j]
		switch a.Type() {
		case goobj2.AuxGotype:
			pp.gotype = l.resolve(r, a.Sym())
			break loop
		default:
			// nothing to do
		}
	}

	// Install new payload to global index space.
	// (This needs to happen at the end, as the accessors above
	// need to access the old symbol content.)
	l.objSyms[symIdx] = objSym{l.extReader, pi}
	l.extReader.syms = append(l.extReader.syms, symIdx)
}

// Copy the payload of symbol src to dst. Both src and dst must be external
// symbols.
// The intended use case is that when building/linking against a shared library,
// where we do symbol name mangling, the Go object file may have reference to
// the original symbol name whereas the shared library provides a symbol with
// the mangled name. When we do mangling, we copy payload of mangled to original.
func (l *Loader) CopySym(src, dst Sym) {
	if !l.IsExternal(dst) {
		panic("dst is not external") //l.newExtSym(l.SymName(dst), l.SymVersion(dst))
	}
	if !l.IsExternal(src) {
		panic("src is not external") //l.cloneToExternal(src)
	}
	l.payloads[l.extIndex(dst)] = l.payloads[l.extIndex(src)]
	l.SetSymPkg(dst, l.SymPkg(src))
	// TODO: other attributes?
}

// CopyAttributes copies over all of the attributes of symbol 'src' to
// symbol 'dst'.
func (l *Loader) CopyAttributes(src Sym, dst Sym) {
	l.SetAttrReachable(dst, l.AttrReachable(src))
	l.SetAttrOnList(dst, l.AttrOnList(src))
	l.SetAttrLocal(dst, l.AttrLocal(src))
	l.SetAttrNotInSymbolTable(dst, l.AttrNotInSymbolTable(src))
	if l.IsExternal(dst) {
		l.SetAttrVisibilityHidden(dst, l.AttrVisibilityHidden(src))
		l.SetAttrDuplicateOK(dst, l.AttrDuplicateOK(src))
		l.SetAttrShared(dst, l.AttrShared(src))
		l.SetAttrExternal(dst, l.AttrExternal(src))
	} else {
		// Some attributes are modifiable only for external symbols.
		// In such cases, don't try to transfer over the attribute
		// from the source even if there is a clash. This comes up
		// when copying attributes from a dupOK ABI wrapper symbol to
		// the real target symbol (which may not be marked dupOK).
	}
	l.SetAttrTopFrame(dst, l.AttrTopFrame(src))
	l.SetAttrSpecial(dst, l.AttrSpecial(src))
	l.SetAttrCgoExportDynamic(dst, l.AttrCgoExportDynamic(src))
	l.SetAttrCgoExportStatic(dst, l.AttrCgoExportStatic(src))
	l.SetAttrReadOnly(dst, l.AttrReadOnly(src))
}

// migrateAttributes copies over all of the attributes of symbol 'src' to
// sym.Symbol 'dst'.
func (l *Loader) migrateAttributes(src Sym, dst *sym.Symbol) {
	dst.Value = l.SymValue(src)
	dst.Align = l.SymAlign(src)
	dst.Sect = l.SymSect(src)

	dst.Attr.Set(sym.AttrReachable, l.AttrReachable(src))
	dst.Attr.Set(sym.AttrOnList, l.AttrOnList(src))
	dst.Attr.Set(sym.AttrLocal, l.AttrLocal(src))
	dst.Attr.Set(sym.AttrNotInSymbolTable, l.AttrNotInSymbolTable(src))
	dst.Attr.Set(sym.AttrNoSplit, l.IsNoSplit(src))
	dst.Attr.Set(sym.AttrVisibilityHidden, l.AttrVisibilityHidden(src))
	dst.Attr.Set(sym.AttrDuplicateOK, l.AttrDuplicateOK(src))
	dst.Attr.Set(sym.AttrShared, l.AttrShared(src))
	dst.Attr.Set(sym.AttrExternal, l.AttrExternal(src))
	dst.Attr.Set(sym.AttrTopFrame, l.AttrTopFrame(src))
	dst.Attr.Set(sym.AttrSpecial, l.AttrSpecial(src))
	dst.Attr.Set(sym.AttrCgoExportDynamic, l.AttrCgoExportDynamic(src))
	dst.Attr.Set(sym.AttrCgoExportStatic, l.AttrCgoExportStatic(src))
	dst.Attr.Set(sym.AttrReadOnly, l.AttrReadOnly(src))

	// Convert outer relationship
	if outer, ok := l.outer[src]; ok {
		dst.Outer = l.Syms[outer]
	}

	// Set sub-symbol attribute. See the comment on the AttrSubSymbol
	// method for more on this, there is some tricky stuff here.
	dst.Attr.Set(sym.AttrSubSymbol, l.outer[src] != 0 && l.sub[l.outer[src]] != 0)

	// Copy over dynimplib, dynimpvers, extname.
	if name, ok := l.extname[src]; ok {
		dst.SetExtname(name)
	}
	if l.SymDynimplib(src) != "" {
		dst.SetDynimplib(l.SymDynimplib(src))
	}
	if l.SymDynimpvers(src) != "" {
		dst.SetDynimpvers(l.SymDynimpvers(src))
	}

	// Copy ELF type if set.
	if et, ok := l.elfType[src]; ok {
		dst.SetElfType(et)
	}

	// Copy pe objects values if set.
	if plt, ok := l.plt[src]; ok {
		dst.SetPlt(plt)
	}
	if got, ok := l.got[src]; ok {
		dst.SetGot(got)
	}

	// Copy dynid
	if dynid, ok := l.dynid[src]; ok {
		dst.Dynid = dynid
	}
}

// CreateExtSym creates a new external symbol with the specified name
// without adding it to any lookup tables, returning a Sym index for it.
func (l *Loader) CreateExtSym(name string, ver int) Sym {
	return l.newExtSym(name, ver)
}

// CreateStaticSym creates a new static symbol with the specified name
// without adding it to any lookup tables, returning a Sym index for it.
func (l *Loader) CreateStaticSym(name string) Sym {
	// Assign a new unique negative version -- this is to mark the
	// symbol so that it can be skipped when ExtractSymbols is adding
	// ext syms to the sym.Symbols hash.
	l.anonVersion--
	return l.newExtSym(name, l.anonVersion)
}

func (l *Loader) FreeSym(i Sym) {
	if l.IsExternal(i) {
		pp := l.getPayload(i)
		*pp = extSymPayload{}
	}
}

func loadObjFull(l *Loader, r *oReader, needReloc, needExtReloc bool) {
	for i, n := 0, r.NSym()+r.NNonpkgdef(); i < n; i++ {
		// A symbol may be a dup or overwritten. In this case, its
		// content will actually be provided by a different object
		// (to which its global index points). Skip those symbols.
		gi := l.toGlobal(r, i)
		if r2, i2 := l.toLocal(gi); r2 != r || i2 != i {
			continue
		}
		s := l.Syms[gi]
		if s == nil {
			continue
		}

		l.migrateAttributes(gi, s)
		// Be careful not to overwrite attributes set by the linker.
		// Don't use the attributes from the object file.

		osym := r.Sym(i)
		size := osym.Siz()

		// Symbol data
		s.P = l.OutData(gi)

		// Relocs
		if needReloc {
			relocs := l.relocs(r, i)
			batch := l.relocBatch
			s.R = batch[:relocs.Count():relocs.Count()]
			l.relocBatch = batch[relocs.Count():]
			l.convertRelocations(gi, &relocs, s, false)
		}
		if needExtReloc {
			l.convertExtRelocs(s, gi)
		}

		// Aux symbol info
		auxs := r.Auxs(i)
		for j := range auxs {
			a := &auxs[j]
			switch a.Type() {
			case goobj2.AuxFuncInfo, goobj2.AuxFuncdata, goobj2.AuxGotype:
				// already handled
			case goobj2.AuxDwarfInfo, goobj2.AuxDwarfLoc, goobj2.AuxDwarfRanges, goobj2.AuxDwarfLines:
				// ignored for now
			default:
				panic("unknown aux type")
			}
		}

		if s.Size < int64(size) {
			s.Size = int64(size)
		}
	}
}

// convertRelocations takes a vector of loader.Reloc relocations and
// translates them into an equivalent set of sym.Reloc relocations on
// the symbol "dst", performing fixups along the way for ABI aliases,
// etc. It is assumed that the caller has pre-allocated the dst symbol
// relocations slice. If 'strict' is set, then this method will
// panic if it finds a relocation targeting a nil symbol.
func (l *Loader) convertRelocations(symIdx Sym, src *Relocs, dst *sym.Symbol, strict bool) {
	for j := range dst.R {
		r := src.At2(j)
		rs := r.Sym()
		sz := r.Siz()
		rt := r.Type()
		if rt == objabi.R_METHODOFF {
			if l.attrReachable.Has(rs) {
				rt = objabi.R_ADDROFF
			} else {
				sz = 0
				rs = 0
			}
		}
		if rt == objabi.R_WEAKADDROFF && !l.attrReachable.Has(rs) {
			rs = 0
			sz = 0
		}
		if rs != 0 && l.Syms[rs] != nil && l.Syms[rs].Type == sym.SABIALIAS {
			rsrelocs := l.Relocs(rs)
			rs = rsrelocs.At2(0).Sym()
		}
		if strict && rs != 0 && l.Syms[rs] == nil && rt != objabi.R_USETYPE {
			panic("nil reloc target in convertRelocations")
		}
		dst.R[j] = sym.Reloc{
			Off:  r.Off(),
			Siz:  sz,
			Type: rt,
			Add:  r.Add(),
			Sym:  l.Syms[rs],
		}
		if rv := l.RelocVariant(symIdx, j); rv != 0 {
			dst.R[j].InitExt()
			dst.R[j].Variant = rv
		}
	}
}

// Convert external relocations to sym.Relocs on symbol dst.
func (l *Loader) convertExtRelocs(dst *sym.Symbol, src Sym) {
	if int(src) >= len(l.extRelocs) {
		return
	}
	extRelocs := l.extRelocs[src]
	if len(extRelocs) == 0 {
		return
	}
	if len(dst.R) != 0 {
		panic("bad")
	}

	n := len(extRelocs)
	batch := l.relocBatch
	dst.R = batch[:n:n]
	l.relocBatch = batch[n:]
	relocs := l.Relocs(src)
	for i := range dst.R {
		er := &extRelocs[i]
		sr := relocs.At2(er.Idx)
		r := &dst.R[i]
		r.RelocExt = &l.relocExtBatch[0]
		l.relocExtBatch = l.relocExtBatch[1:]
		r.Off = sr.Off()
		r.Siz = sr.Siz()
		r.Type = sr.Type()
		r.Sym = l.Syms[l.ResolveABIAlias(sr.Sym())]
		r.Add = sr.Add()
		r.Xsym = l.Syms[er.Xsym]
		r.Xadd = er.Xadd
		if rv := l.RelocVariant(src, er.Idx); rv != 0 {
			r.Variant = rv
		}
	}
}

// relocId is essentially a <S,R> tuple identifying the Rth
// relocation of symbol S.
type relocId struct {
	sym  Sym
	ridx int
}

// SetRelocVariant sets the 'variant' property of a relocation on
// some specific symbol.
func (l *Loader) SetRelocVariant(s Sym, ri int, v sym.RelocVariant) {
	// sanity check
	if relocs := l.Relocs(s); ri >= relocs.Count() {
		panic("invalid relocation ID")
	}
	if l.relocVariant == nil {
		l.relocVariant = make(map[relocId]sym.RelocVariant)
	}
	if v != 0 {
		l.relocVariant[relocId{s, ri}] = v
	} else {
		delete(l.relocVariant, relocId{s, ri})
	}
}

// RelocVariant returns the 'variant' property of a relocation on
// some specific symbol.
func (l *Loader) RelocVariant(s Sym, ri int) sym.RelocVariant {
	return l.relocVariant[relocId{s, ri}]
}

// UndefinedRelocTargets iterates through the global symbol index
// space, looking for symbols with relocations targeting undefined
// references. The linker's loadlib method uses this to determine if
// there are unresolved references to functions in system libraries
// (for example, libgcc.a), presumably due to CGO code. Return
// value is a list of loader.Sym's corresponding to the undefined
// cross-refs. The "limit" param controls the maximum number of
// results returned; if "limit" is -1, then all undefs are returned.
func (l *Loader) UndefinedRelocTargets(limit int) []Sym {
	result := []Sym{}
	for si := Sym(1); si < Sym(len(l.objSyms)); si++ {
		relocs := l.Relocs(si)
		for ri := 0; ri < relocs.Count(); ri++ {
			r := relocs.At2(ri)
			rs := r.Sym()
			if rs != 0 && l.SymType(rs) == sym.SXREF && l.RawSymName(rs) != ".got" {
				result = append(result, rs)
				if limit != -1 && len(result) >= limit {
					break
				}
			}
		}
	}
	return result
}

// AssignTextSymbolOrder populates the Textp2 slices within each
// library and compilation unit, insuring that packages are laid down
// in dependency order (internal first, then everything else). Return value
// is a slice of all text syms.
func (l *Loader) AssignTextSymbolOrder(libs []*sym.Library, intlibs []bool, extsyms []Sym) []Sym {

	// Library Textp2 lists should be empty at this point.
	for _, lib := range libs {
		if len(lib.Textp2) != 0 {
			panic("expected empty Textp2 slice for library")
		}
		if len(lib.DupTextSyms2) != 0 {
			panic("expected empty DupTextSyms2 slice for library")
		}
	}

	// Used to record which dupok symbol we've assigned to a unit.
	// Can't use the onlist attribute here because it will need to
	// clear for the later assignment of the sym.Symbol to a unit.
	// NB: we can convert to using onList once we no longer have to
	// call the regular addToTextp.
	assignedToUnit := MakeBitmap(l.NSym() + 1)

	// Start off textp2 with reachable external syms.
	textp2 := []Sym{}
	for _, sym := range extsyms {
		if !l.attrReachable.Has(sym) {
			continue
		}
		textp2 = append(textp2, sym)
	}

	// Walk through all text symbols from Go object files and append
	// them to their corresponding library's textp2 list.
	for _, o := range l.objs[1:] {
		r := o.r
		lib := r.unit.Lib
		for i, n := 0, r.NSym()+r.NNonpkgdef(); i < n; i++ {
			gi := l.toGlobal(r, i)
			if !l.attrReachable.Has(gi) {
				continue
			}
			osym := r.Sym(i)
			st := sym.AbiSymKindToSymKind[objabi.SymKind(osym.Type())]
			if st != sym.STEXT {
				continue
			}
			dupok := osym.Dupok()
			if r2, i2 := l.toLocal(gi); r2 != r || i2 != i {
				// A dupok text symbol is resolved to another package.
				// We still need to record its presence in the current
				// package, as the trampoline pass expects packages
				// are laid out in dependency order.
				lib.DupTextSyms2 = append(lib.DupTextSyms2, sym.LoaderSym(gi))
				continue // symbol in different object
			}
			if dupok {
				lib.DupTextSyms2 = append(lib.DupTextSyms2, sym.LoaderSym(gi))
				continue
			}

			lib.Textp2 = append(lib.Textp2, sym.LoaderSym(gi))
		}
	}

	// Now assemble global textp, and assign text symbols to units.
	for _, doInternal := range [2]bool{true, false} {
		for idx, lib := range libs {
			if intlibs[idx] != doInternal {
				continue
			}
			lists := [2][]sym.LoaderSym{lib.Textp2, lib.DupTextSyms2}
			for i, list := range lists {
				for _, s := range list {
					sym := Sym(s)
					if l.attrReachable.Has(sym) && !assignedToUnit.Has(sym) {
						textp2 = append(textp2, sym)
						unit := l.SymUnit(sym)
						if unit != nil {
							unit.Textp2 = append(unit.Textp2, s)
							assignedToUnit.Set(sym)
						}
						// Dupok symbols may be defined in multiple packages; the
						// associated package for a dupok sym is chosen sort of
						// arbitrarily (the first containing package that the linker
						// loads). Canonicalizes its Pkg to the package with which
						// it will be laid down in text.
						if i == 1 /* DupTextSyms2 */ && l.SymPkg(sym) != lib.Pkg {
							l.SetSymPkg(sym, lib.Pkg)
						}
					}
				}
			}
			lib.Textp2 = nil
			lib.DupTextSyms2 = nil
		}
	}

	return textp2
}

// ErrorReporter is a helper class for reporting errors.
type ErrorReporter struct {
	ldr              *Loader
	AfterErrorAction func()
}

// Errorf method logs an error message.
//
// After each error, the error actions function will be invoked; this
// will either terminate the link immediately (if -h option given)
// or it will keep a count and exit if more than 20 errors have been printed.
//
// Logging an error means that on exit cmd/link will delete any
// output file and return a non-zero error code.
//
func (reporter *ErrorReporter) Errorf(s Sym, format string, args ...interface{}) {
	if s != 0 && reporter.ldr.SymName(s) != "" {
		format = reporter.ldr.SymName(s) + ": " + format
	} else {
		format = fmt.Sprintf("sym %d: %s", s, format)
	}
	format += "\n"
	fmt.Fprintf(os.Stderr, format, args...)
	reporter.AfterErrorAction()
}

// GetErrorReporter returns the loader's associated error reporter.
func (l *Loader) GetErrorReporter() *ErrorReporter {
	return l.errorReporter
}

// Errorf method logs an error message. See ErrorReporter.Errorf for details.
func (l *Loader) Errorf(s Sym, format string, args ...interface{}) {
	l.errorReporter.Errorf(s, format, args...)
}

// For debugging.
func (l *Loader) Dump() {
	fmt.Println("objs")
	for _, obj := range l.objs {
		if obj.r != nil {
			fmt.Println(obj.i, obj.r.unit.Lib)
		}
	}
	fmt.Println("extStart:", l.extStart)
	fmt.Println("Nsyms:", len(l.objSyms))
	fmt.Println("syms")
	for i := Sym(1); i < Sym(len(l.objSyms)); i++ {
		pi := interface{}("")
		if l.IsExternal(i) {
			pi = fmt.Sprintf("<ext %d>", l.extIndex(i))
		}
		var s *sym.Symbol
		if int(i) < len(l.Syms) {
			s = l.Syms[i]
		}
		if s != nil {
			fmt.Println(i, s, s.Type, pi)
		} else {
			fmt.Println(i, l.SymName(i), "<not loaded>", pi)
		}
	}
	fmt.Println("symsByName")
	for name, i := range l.symsByName[0] {
		fmt.Println(i, name, 0)
	}
	for name, i := range l.symsByName[1] {
		fmt.Println(i, name, 1)
	}
	fmt.Println("payloads:")
	for i := range l.payloads {
		pp := l.payloads[i]
		fmt.Println(i, pp.name, pp.ver, pp.kind)
	}
}
