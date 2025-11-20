// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO/NICETOHAVE:
//   - eliminate DW_CLS_ if not used
//   - package info in compilation units
//   - assign types to their packages
//   - gdb uses c syntax, meaning clumsy quoting is needed for go identifiers. eg
//     ptype struct '[]uint8' and qualifiers need to be quoted away
//   - file:line info for variables
//   - make strings a typedef so prettyprinters can see the underlying string type

package ld

import (
	"cmd/internal/dwarf"
	"cmd/internal/obj"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"cmd/internal/sys"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"cmp"
	"fmt"
	"internal/abi"
	"internal/buildcfg"
	"log"
	"path"
	"runtime"
	"slices"
	"strings"
	"sync"
)

// dwctxt is a wrapper intended to satisfy the method set of
// dwarf.Context, so that functions like dwarf.PutAttrs will work with
// DIEs that use loader.Sym as opposed to *sym.Symbol. It is also
// being used as a place to store tables/maps that are useful as part
// of type conversion (this is just a convenience; it would be easy to
// split these things out into another type if need be).
type dwctxt struct {
	linkctxt *Link
	ldr      *loader.Loader
	arch     *sys.Arch

	// This maps type name string (e.g. "uintptr") to loader symbol for
	// the DWARF DIE for that type (e.g. "go:info.type.uintptr")
	tmap map[string]loader.Sym

	// This maps loader symbol for the DWARF DIE symbol generated for
	// a type (e.g. "go:info.uintptr") to the type symbol itself
	// ("type:uintptr").
	// FIXME: try converting this map (and the next one) to a single
	// array indexed by loader.Sym -- this may perform better.
	rtmap map[loader.Sym]loader.Sym

	// This maps Go type symbol (e.g. "type:XXX") to loader symbol for
	// the typedef DIE for that type (e.g. "go:info.XXX..def")
	tdmap map[loader.Sym]loader.Sym

	// Cache these type symbols, so as to avoid repeatedly looking them up
	typeRuntimeEface loader.Sym
	typeRuntimeIface loader.Sym
	uintptrInfoSym   loader.Sym

	// Used at various points in that parallel portion of DWARF gen to
	// protect against conflicting updates to globals (such as "gdbscript")
	dwmu *sync.Mutex
}

// dwSym wraps a loader.Sym; this type is meant to obey the interface
// rules for dwarf.Sym from the cmd/internal/dwarf package. DwDie and
// DwAttr objects contain references to symbols via this type.
type dwSym loader.Sym

func (c dwctxt) PtrSize() int {
	return c.arch.PtrSize
}

func (c dwctxt) Size(s dwarf.Sym) int64 {
	return int64(len(c.ldr.Data(loader.Sym(s.(dwSym)))))
}

func (c dwctxt) AddInt(s dwarf.Sym, size int, i int64) {
	ds := loader.Sym(s.(dwSym))
	dsu := c.ldr.MakeSymbolUpdater(ds)
	dsu.AddUintXX(c.arch, uint64(i), size)
}

func (c dwctxt) AddBytes(s dwarf.Sym, b []byte) {
	ds := loader.Sym(s.(dwSym))
	dsu := c.ldr.MakeSymbolUpdater(ds)
	dsu.AddBytes(b)
}

func (c dwctxt) AddString(s dwarf.Sym, v string) {
	ds := loader.Sym(s.(dwSym))
	dsu := c.ldr.MakeSymbolUpdater(ds)
	dsu.Addstring(v)
}

func (c dwctxt) AddAddress(s dwarf.Sym, data any, value int64) {
	ds := loader.Sym(s.(dwSym))
	dsu := c.ldr.MakeSymbolUpdater(ds)
	if value != 0 {
		value -= dsu.Value()
	}
	tgtds := loader.Sym(data.(dwSym))
	dsu.AddAddrPlus(c.arch, tgtds, value)
}

func (c dwctxt) AddCURelativeAddress(s dwarf.Sym, data any, value int64) {
	ds := loader.Sym(s.(dwSym))
	dsu := c.ldr.MakeSymbolUpdater(ds)
	if value != 0 {
		value -= dsu.Value()
	}
	tgtds := loader.Sym(data.(dwSym))
	dsu.AddCURelativeAddrPlus(c.arch, tgtds, value)
}

func (c dwctxt) AddSectionOffset(s dwarf.Sym, size int, t any, ofs int64) {
	ds := loader.Sym(s.(dwSym))
	dsu := c.ldr.MakeSymbolUpdater(ds)
	tds := loader.Sym(t.(dwSym))
	switch size {
	default:
		c.linkctxt.Errorf(ds, "invalid size %d in adddwarfref\n", size)
	case c.arch.PtrSize, 4:
	}
	dsu.AddSymRef(c.arch, tds, ofs, objabi.R_ADDROFF, size)
}

func (c dwctxt) AddDWARFAddrSectionOffset(s dwarf.Sym, t any, ofs int64) {
	size := 4
	if isDwarf64(c.linkctxt) {
		size = 8
	}
	ds := loader.Sym(s.(dwSym))
	dsu := c.ldr.MakeSymbolUpdater(ds)
	tds := loader.Sym(t.(dwSym))
	switch size {
	default:
		c.linkctxt.Errorf(ds, "invalid size %d in adddwarfref\n", size)
	case c.arch.PtrSize, 4:
	}
	dsu.AddSymRef(c.arch, tds, ofs, objabi.R_DWARFSECREF, size)
}

func (c dwctxt) AddIndirectTextRef(s dwarf.Sym, t any) {
	ds := loader.Sym(s.(dwSym))
	dsu := c.ldr.MakeSymbolUpdater(ds)
	tds := loader.Sym(t.(dwSym))
	dsu.AddSymRef(c.arch, tds, 0, objabi.R_DWTXTADDR_U4, 4)
}

func (c dwctxt) Logf(format string, args ...any) {
	c.linkctxt.Logf(format, args...)
}

// At the moment these interfaces are only used in the compiler.

func (c dwctxt) CurrentOffset(s dwarf.Sym) int64 {
	panic("should be used only in the compiler")
}

func (c dwctxt) RecordDclReference(s dwarf.Sym, t dwarf.Sym, dclIdx int, inlIndex int) {
	panic("should be used only in the compiler")
}

func (c dwctxt) RecordChildDieOffsets(s dwarf.Sym, vars []*dwarf.Var, offsets []int32) {
	panic("should be used only in the compiler")
}

func isDwarf64(ctxt *Link) bool {
	return ctxt.HeadType == objabi.Haix
}

// https://sourceware.org/gdb/onlinedocs/gdb/dotdebug_005fgdb_005fscripts-section.html
// Each entry inside .debug_gdb_scripts section begins with a non-null prefix
// byte that specifies the kind of entry. The following entries are supported:
const (
	GdbScriptPythonFileId = 1
	GdbScriptSchemeFileId = 3
	GdbScriptPythonTextId = 4
	GdbScriptSchemeTextId = 6
)

var gdbscript string

// dwarfSecInfo holds information about a DWARF output section,
// specifically a section symbol and a list of symbols contained in
// that section. On the syms list, the first symbol will always be the
// section symbol, then any remaining symbols (if any) will be
// sub-symbols in that section. Note that for some sections (eg:
// .debug_abbrev), the section symbol is all there is (all content is
// contained in it). For other sections (eg: .debug_info), the section
// symbol is empty and all the content is in the sub-symbols. Finally
// there are some sections (eg: .debug_ranges) where it is a mix (both
// the section symbol and the sub-symbols have content)
type dwarfSecInfo struct {
	syms []loader.Sym
}

// secSym returns the section symbol for the section.
func (dsi *dwarfSecInfo) secSym() loader.Sym {
	if len(dsi.syms) == 0 {
		return 0
	}
	return dsi.syms[0]
}

// subSyms returns a list of sub-symbols for the section.
func (dsi *dwarfSecInfo) subSyms() []loader.Sym {
	if len(dsi.syms) == 0 {
		return []loader.Sym{}
	}
	return dsi.syms[1:]
}

// dwarfp stores the collected DWARF symbols created during
// dwarf generation.
var dwarfp []dwarfSecInfo

func (d *dwctxt) writeabbrev() dwarfSecInfo {
	abrvs := d.ldr.CreateSymForUpdate(".debug_abbrev", 0)
	abrvs.SetType(sym.SDWARFSECT)
	abrvs.AddBytes(dwarf.GetAbbrev())
	return dwarfSecInfo{syms: []loader.Sym{abrvs.Sym()}}
}

var dwtypes dwarf.DWDie

// newattr attaches a new attribute to the specified DIE.
//
// FIXME: at the moment attributes are stored in a linked list in a
// fairly space-inefficient way -- it might be better to instead look
// up all attrs in a single large table, then store indices into the
// table in the DIE. This would allow us to common up storage for
// attributes that are shared by many DIEs (ex: byte size of N).
func newattr(die *dwarf.DWDie, attr uint16, cls int, value int64, data any) {
	a := new(dwarf.DWAttr)
	a.Link = die.Attr
	die.Attr = a
	a.Atr = attr
	a.Cls = uint8(cls)
	a.Value = value
	a.Data = data
}

// Each DIE (except the root ones) has at least 1 attribute: its
// name. getattr moves the desired one to the front so
// frequently searched ones are found faster.
func getattr(die *dwarf.DWDie, attr uint16) *dwarf.DWAttr {
	if die.Attr.Atr == attr {
		return die.Attr
	}

	a := die.Attr
	b := a.Link
	for b != nil {
		if b.Atr == attr {
			a.Link = b.Link
			b.Link = die.Attr
			die.Attr = b
			return b
		}

		a = b
		b = b.Link
	}

	return nil
}

// Every DIE manufactured by the linker has at least an AT_name
// attribute (but it will only be written out if it is listed in the abbrev).
// The compiler does create nameless DWARF DIEs (ex: concrete subprogram
// instance).
// FIXME: it would be more efficient to bulk-allocate DIEs.
func (d *dwctxt) newdie(parent *dwarf.DWDie, abbrev int, name string) *dwarf.DWDie {
	die := new(dwarf.DWDie)
	die.Abbrev = abbrev
	die.Link = parent.Child
	parent.Child = die

	newattr(die, dwarf.DW_AT_name, dwarf.DW_CLS_STRING, int64(len(name)), name)

	// Sanity check: all DIEs created in the linker should be named.
	if name == "" {
		panic("nameless DWARF DIE")
	}

	var st sym.SymKind
	switch abbrev {
	case dwarf.DW_ABRV_FUNCTYPEPARAM, dwarf.DW_ABRV_FUNCTYPEOUTPARAM, dwarf.DW_ABRV_DOTDOTDOT, dwarf.DW_ABRV_STRUCTFIELD, dwarf.DW_ABRV_ARRAYRANGE:
		// There are no relocations against these dies, and their names
		// are not unique, so don't create a symbol.
		return die
	case dwarf.DW_ABRV_COMPUNIT, dwarf.DW_ABRV_COMPUNIT_TEXTLESS:
		// Avoid collisions with "real" symbol names.
		name = fmt.Sprintf(".pkg.%s.%d", name, len(d.linkctxt.compUnits))
		st = sym.SDWARFCUINFO
	case dwarf.DW_ABRV_VARIABLE:
		st = sym.SDWARFVAR
	default:
		// Everything else is assigned a type of SDWARFTYPE. that
		// this also includes loose ends such as STRUCT_FIELD.
		st = sym.SDWARFTYPE
	}
	ds := d.ldr.LookupOrCreateSym(dwarf.InfoPrefix+name, 0)
	dsu := d.ldr.MakeSymbolUpdater(ds)
	dsu.SetType(st)
	d.ldr.SetAttrNotInSymbolTable(ds, true)
	d.ldr.SetAttrReachable(ds, true)
	die.Sym = dwSym(ds)
	if abbrev >= dwarf.DW_ABRV_NULLTYPE && abbrev <= dwarf.DW_ABRV_TYPEDECL {
		d.tmap[name] = ds
	}

	return die
}

func walktypedef(die *dwarf.DWDie) *dwarf.DWDie {
	if die == nil {
		return nil
	}
	// Resolve typedef if present.
	if die.Abbrev == dwarf.DW_ABRV_TYPEDECL {
		for attr := die.Attr; attr != nil; attr = attr.Link {
			if attr.Atr == dwarf.DW_AT_type && attr.Cls == dwarf.DW_CLS_REFERENCE && attr.Data != nil {
				return attr.Data.(*dwarf.DWDie)
			}
		}
	}

	return die
}

func (d *dwctxt) walksymtypedef(symIdx loader.Sym) loader.Sym {

	// We're being given the loader symbol for the type DIE, e.g.
	// "go:info.type.uintptr". Map that first to the type symbol (e.g.
	// "type:uintptr") and then to the typedef DIE for the type.
	// FIXME: this seems clunky, maybe there is a better way to do this.

	if ts, ok := d.rtmap[symIdx]; ok {
		if def, ok := d.tdmap[ts]; ok {
			return def
		}
		d.linkctxt.Errorf(ts, "internal error: no entry for sym %d in tdmap\n", ts)
		return 0
	}
	d.linkctxt.Errorf(symIdx, "internal error: no entry for sym %d in rtmap\n", symIdx)
	return 0
}

// Find child by AT_name using hashtable if available or linear scan
// if not.
func findchild(die *dwarf.DWDie, name string) *dwarf.DWDie {
	var prev *dwarf.DWDie
	for ; die != prev; prev, die = die, walktypedef(die) {
		for a := die.Child; a != nil; a = a.Link {
			if name == getattr(a, dwarf.DW_AT_name).Data {
				return a
			}
		}
		continue
	}
	return nil
}

// find looks up the loader symbol for the DWARF DIE generated for the
// type with the specified name.
func (d *dwctxt) find(name string) loader.Sym {
	return d.tmap[name]
}

func (d *dwctxt) mustFind(name string) loader.Sym {
	r := d.find(name)
	if r == 0 {
		Exitf("dwarf find: cannot find %s", name)
	}
	return r
}

func (d *dwctxt) adddwarfref(sb *loader.SymbolBuilder, t loader.Sym, size int) {
	switch size {
	default:
		d.linkctxt.Errorf(sb.Sym(), "invalid size %d in adddwarfref\n", size)
	case d.arch.PtrSize, 4:
	}
	sb.AddSymRef(d.arch, t, 0, objabi.R_DWARFSECREF, size)
}

func (d *dwctxt) newrefattr(die *dwarf.DWDie, attr uint16, ref loader.Sym) {
	if ref == 0 {
		return
	}
	newattr(die, attr, dwarf.DW_CLS_REFERENCE, 0, dwSym(ref))
}

func (d *dwctxt) dtolsym(s dwarf.Sym) loader.Sym {
	if s == nil {
		return 0
	}
	dws := loader.Sym(s.(dwSym))
	return dws
}

func (d *dwctxt) putdie(syms []loader.Sym, die *dwarf.DWDie) []loader.Sym {
	s := d.dtolsym(die.Sym)
	if s == 0 {
		s = syms[len(syms)-1]
	} else {
		syms = append(syms, s)
	}
	sDwsym := dwSym(s)
	dwarf.Uleb128put(d, sDwsym, int64(die.Abbrev))
	dwarf.PutAttrs(d, sDwsym, die.Abbrev, die.Attr)
	if dwarf.HasChildren(die) {
		for die := die.Child; die != nil; die = die.Link {
			syms = d.putdie(syms, die)
		}
		dsu := d.ldr.MakeSymbolUpdater(syms[len(syms)-1])
		dsu.AddUint8(0)
	}
	return syms
}

func reverselist(list **dwarf.DWDie) {
	curr := *list
	var prev *dwarf.DWDie
	for curr != nil {
		next := curr.Link
		curr.Link = prev
		prev = curr
		curr = next
	}

	*list = prev
}

func reversetree(list **dwarf.DWDie) {
	reverselist(list)
	for die := *list; die != nil; die = die.Link {
		if dwarf.HasChildren(die) {
			reversetree(&die.Child)
		}
	}
}

func newmemberoffsetattr(die *dwarf.DWDie, offs int32) {
	newattr(die, dwarf.DW_AT_data_member_location, dwarf.DW_CLS_CONSTANT, int64(offs), nil)
}

func (d *dwctxt) lookupOrDiag(n string) loader.Sym {
	symIdx := d.ldr.Lookup(n, 0)
	if symIdx == 0 {
		Exitf("dwarf: missing type: %s", n)
	}
	if len(d.ldr.Data(symIdx)) == 0 {
		Exitf("dwarf: missing type (no data): %s", n)
	}

	return symIdx
}

func (d *dwctxt) dotypedef(parent *dwarf.DWDie, name string, def *dwarf.DWDie) *dwarf.DWDie {
	// Only emit typedefs for real names.
	if strings.HasPrefix(name, "map[") {
		return nil
	}
	if strings.HasPrefix(name, "struct {") {
		return nil
	}
	// cmd/compile uses "noalg.struct {...}" as type name when hash and eq algorithm generation of
	// this struct type is suppressed.
	if strings.HasPrefix(name, "noalg.struct {") {
		return nil
	}
	if strings.HasPrefix(name, "chan ") {
		return nil
	}
	if name[0] == '[' || name[0] == '*' {
		return nil
	}
	if def == nil {
		Errorf("dwarf: bad def in dotypedef")
	}

	// Create a new loader symbol for the typedef. We no longer
	// do lookups of typedef symbols by name, so this is going
	// to be an anonymous symbol (we want this for perf reasons).
	tds := d.ldr.CreateExtSym("", 0)
	tdsu := d.ldr.MakeSymbolUpdater(tds)
	tdsu.SetType(sym.SDWARFTYPE)
	def.Sym = dwSym(tds)
	d.ldr.SetAttrNotInSymbolTable(tds, true)
	d.ldr.SetAttrReachable(tds, true)

	// The typedef entry must be created after the def,
	// so that future lookups will find the typedef instead
	// of the real definition. This hooks the typedef into any
	// circular definition loops, so that gdb can understand them.
	die := d.newdie(parent, dwarf.DW_ABRV_TYPEDECL, name)

	d.newrefattr(die, dwarf.DW_AT_type, tds)

	return die
}

// Define gotype, for composite ones recurse into constituents.
func (d *dwctxt) defgotype(gotype loader.Sym) loader.Sym {
	if gotype == 0 {
		return d.mustFind("<unspecified>")
	}

	// If we already have a tdmap entry for the gotype, return it.
	if ds, ok := d.tdmap[gotype]; ok {
		return ds
	}

	sn := d.ldr.SymName(gotype)
	if !strings.HasPrefix(sn, "type:") {
		d.linkctxt.Errorf(gotype, "dwarf: type name doesn't start with \"type:\"")
		return d.mustFind("<unspecified>")
	}
	name := sn[len("type:"):] // could also decode from Type.string

	sdie := d.find(name)
	if sdie != 0 {
		return sdie
	}

	gtdwSym := d.newtype(gotype)
	d.tdmap[gotype] = loader.Sym(gtdwSym.Sym.(dwSym))
	return loader.Sym(gtdwSym.Sym.(dwSym))
}

func (d *dwctxt) newtype(gotype loader.Sym) *dwarf.DWDie {
	sn := d.ldr.SymName(gotype)
	name := sn[len("type:"):] // could also decode from Type.string
	tdata := d.ldr.Data(gotype)
	if len(tdata) == 0 {
		d.linkctxt.Errorf(gotype, "missing type")
	}
	kind := decodetypeKind(d.arch, tdata)
	bytesize := decodetypeSize(d.arch, tdata)

	var die, typedefdie *dwarf.DWDie
	switch kind {
	case abi.Bool:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_BASETYPE, name)
		newattr(die, dwarf.DW_AT_encoding, dwarf.DW_CLS_CONSTANT, dwarf.DW_ATE_boolean, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case abi.Int,
		abi.Int8,
		abi.Int16,
		abi.Int32,
		abi.Int64:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_BASETYPE, name)
		newattr(die, dwarf.DW_AT_encoding, dwarf.DW_CLS_CONSTANT, dwarf.DW_ATE_signed, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case abi.Uint,
		abi.Uint8,
		abi.Uint16,
		abi.Uint32,
		abi.Uint64,
		abi.Uintptr:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_BASETYPE, name)
		newattr(die, dwarf.DW_AT_encoding, dwarf.DW_CLS_CONSTANT, dwarf.DW_ATE_unsigned, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case abi.Float32,
		abi.Float64:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_BASETYPE, name)
		newattr(die, dwarf.DW_AT_encoding, dwarf.DW_CLS_CONSTANT, dwarf.DW_ATE_float, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case abi.Complex64,
		abi.Complex128:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_BASETYPE, name)
		newattr(die, dwarf.DW_AT_encoding, dwarf.DW_CLS_CONSTANT, dwarf.DW_ATE_complex_float, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case abi.Array:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_ARRAYTYPE, name)
		typedefdie = d.dotypedef(&dwtypes, name, die)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		s := decodetypeArrayElem(d.linkctxt, d.arch, gotype)
		d.newrefattr(die, dwarf.DW_AT_type, d.defgotype(s))
		fld := d.newdie(die, dwarf.DW_ABRV_ARRAYRANGE, "range")

		// use actual length not upper bound; correct for 0-length arrays.
		newattr(fld, dwarf.DW_AT_count, dwarf.DW_CLS_CONSTANT, decodetypeArrayLen(d.ldr, d.arch, gotype), 0)

		d.newrefattr(fld, dwarf.DW_AT_type, d.uintptrInfoSym)

	case abi.Chan:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_CHANTYPE, name)
		s := decodetypeChanElem(d.ldr, d.arch, gotype)
		d.newrefattr(die, dwarf.DW_AT_go_elem, d.defgotype(s))
		// Save elem type for synthesizechantypes. We could synthesize here
		// but that would change the order of DIEs we output.
		d.newrefattr(die, dwarf.DW_AT_type, s)

	case abi.Func:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_FUNCTYPE, name)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		typedefdie = d.dotypedef(&dwtypes, name, die)
		data := d.ldr.Data(gotype)
		// FIXME: add caching or reuse reloc slice.
		relocs := d.ldr.Relocs(gotype)
		nfields := decodetypeFuncInCount(d.arch, data)
		for i := 0; i < nfields; i++ {
			s := decodetypeFuncInType(d.ldr, d.arch, gotype, &relocs, i)
			sn := d.ldr.SymName(s)
			fld := d.newdie(die, dwarf.DW_ABRV_FUNCTYPEPARAM, sn[5:])
			d.newrefattr(fld, dwarf.DW_AT_type, d.defgotype(s))
		}

		if decodetypeFuncDotdotdot(d.arch, data) {
			d.newdie(die, dwarf.DW_ABRV_DOTDOTDOT, "...")
		}
		nfields = decodetypeFuncOutCount(d.arch, data)
		for i := 0; i < nfields; i++ {
			s := decodetypeFuncOutType(d.ldr, d.arch, gotype, &relocs, i)
			sn := d.ldr.SymName(s)
			fld := d.newdie(die, dwarf.DW_ABRV_FUNCTYPEOUTPARAM, sn[5:])
			newattr(fld, dwarf.DW_AT_variable_parameter, dwarf.DW_CLS_FLAG, 1, 0)
			d.newrefattr(fld, dwarf.DW_AT_type, d.defgotype(s))
		}

	case abi.Interface:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_IFACETYPE, name)
		typedefdie = d.dotypedef(&dwtypes, name, die)
		data := d.ldr.Data(gotype)
		nfields := int(decodetypeIfaceMethodCount(d.arch, data))
		var s loader.Sym
		if nfields == 0 {
			s = d.typeRuntimeEface
		} else {
			s = d.typeRuntimeIface
		}
		d.newrefattr(die, dwarf.DW_AT_type, d.defgotype(s))

	case abi.Map:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_MAPTYPE, name)
		s := decodetypeMapKey(d.ldr, d.arch, gotype)
		d.newrefattr(die, dwarf.DW_AT_go_key, d.defgotype(s))
		s = decodetypeMapValue(d.ldr, d.arch, gotype)
		d.newrefattr(die, dwarf.DW_AT_go_elem, d.defgotype(s))
		// Save gotype for use in synthesizemaptypes. We could synthesize here,
		// but that would change the order of the DIEs.
		d.newrefattr(die, dwarf.DW_AT_type, gotype)

	case abi.Pointer:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_PTRTYPE, name)
		typedefdie = d.dotypedef(&dwtypes, name, die)
		s := decodetypePtrElem(d.ldr, d.arch, gotype)
		d.newrefattr(die, dwarf.DW_AT_type, d.defgotype(s))

	case abi.Slice:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_SLICETYPE, name)
		typedefdie = d.dotypedef(&dwtypes, name, die)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		s := decodetypeArrayElem(d.linkctxt, d.arch, gotype)
		elem := d.defgotype(s)
		d.newrefattr(die, dwarf.DW_AT_go_elem, elem)

	case abi.String:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_STRINGTYPE, name)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case abi.Struct:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_STRUCTTYPE, name)
		typedefdie = d.dotypedef(&dwtypes, name, die)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		nfields := decodetypeStructFieldCount(d.ldr, d.arch, gotype)
		for i := 0; i < nfields; i++ {
			f := decodetypeStructFieldName(d.ldr, d.arch, gotype, i)
			s := decodetypeStructFieldType(d.linkctxt, d.arch, gotype, i)
			if f == "" {
				sn := d.ldr.SymName(s)
				f = sn[5:] // skip "type:"
			}
			fld := d.newdie(die, dwarf.DW_ABRV_STRUCTFIELD, f)
			d.newrefattr(fld, dwarf.DW_AT_type, d.defgotype(s))
			offset := decodetypeStructFieldOffset(d.ldr, d.arch, gotype, i)
			newmemberoffsetattr(fld, int32(offset))
			if decodetypeStructFieldEmbedded(d.ldr, d.arch, gotype, i) {
				newattr(fld, dwarf.DW_AT_go_embedded_field, dwarf.DW_CLS_FLAG, 1, 0)
			}
		}

	case abi.UnsafePointer:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_BARE_PTRTYPE, name)

	default:
		d.linkctxt.Errorf(gotype, "dwarf: definition of unknown kind %d", kind)
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_TYPEDECL, name)
		d.newrefattr(die, dwarf.DW_AT_type, d.mustFind("<unspecified>"))
	}

	newattr(die, dwarf.DW_AT_go_kind, dwarf.DW_CLS_CONSTANT, int64(kind), 0)

	if d.ldr.AttrReachable(gotype) {
		newattr(die, dwarf.DW_AT_go_runtime_type, dwarf.DW_CLS_GO_TYPEREF, 0, dwSym(gotype))
	}

	// Sanity check.
	if _, ok := d.rtmap[gotype]; ok {
		log.Fatalf("internal error: rtmap entry already installed\n")
	}

	ds := loader.Sym(die.Sym.(dwSym))
	if typedefdie != nil {
		ds = loader.Sym(typedefdie.Sym.(dwSym))
	}
	d.rtmap[ds] = gotype

	if _, ok := prototypedies[sn]; ok {
		prototypedies[sn] = die
	}

	if typedefdie != nil {
		return typedefdie
	}
	return die
}

func (d *dwctxt) nameFromDIESym(dwtypeDIESym loader.Sym) string {
	sn := d.ldr.SymName(dwtypeDIESym)
	return sn[len(dwarf.InfoPrefix):]
}

func (d *dwctxt) defptrto(dwtype loader.Sym) loader.Sym {

	// FIXME: it would be nice if the compiler attached an aux symbol
	// ref from the element type to the pointer type -- it would be
	// more efficient to do it this way as opposed to via name lookups.

	ptrname := "*" + d.nameFromDIESym(dwtype)
	if die := d.find(ptrname); die != 0 {
		return die
	}

	pdie := d.newdie(&dwtypes, dwarf.DW_ABRV_PTRTYPE, ptrname)
	d.newrefattr(pdie, dwarf.DW_AT_type, dwtype)

	// The DWARF info synthesizes pointer types that don't exist at the
	// language level, like *hash<...> and *bucket<...>, and the data
	// pointers of slices. Link to the ones we can find.
	gts := d.ldr.Lookup("type:"+ptrname, 0)
	if gts != 0 && d.ldr.AttrReachable(gts) {
		newattr(pdie, dwarf.DW_AT_go_kind, dwarf.DW_CLS_CONSTANT, int64(abi.Pointer), 0)
		newattr(pdie, dwarf.DW_AT_go_runtime_type, dwarf.DW_CLS_GO_TYPEREF, 0, dwSym(gts))
	}

	if gts != 0 {
		ds := loader.Sym(pdie.Sym.(dwSym))
		d.rtmap[ds] = gts
		d.tdmap[gts] = ds
	}

	return d.dtolsym(pdie.Sym)
}

// Copies src's children into dst. Copies attributes by value.
// DWAttr.data is copied as pointer only. If except is one of
// the top-level children, it will not be copied.
func (d *dwctxt) copychildrenexcept(ctxt *Link, dst *dwarf.DWDie, src *dwarf.DWDie, except *dwarf.DWDie) {
	for src = src.Child; src != nil; src = src.Link {
		if src == except {
			continue
		}
		c := d.newdie(dst, src.Abbrev, getattr(src, dwarf.DW_AT_name).Data.(string))
		for a := src.Attr; a != nil; a = a.Link {
			newattr(c, a.Atr, int(a.Cls), a.Value, a.Data)
		}
		d.copychildrenexcept(ctxt, c, src, nil)
	}

	reverselist(&dst.Child)
}

func (d *dwctxt) copychildren(ctxt *Link, dst *dwarf.DWDie, src *dwarf.DWDie) {
	d.copychildrenexcept(ctxt, dst, src, nil)
}

// Search children (assumed to have TAG_member) for the one named
// field and set its AT_type to dwtype
func (d *dwctxt) substitutetype(structdie *dwarf.DWDie, field string, dwtype loader.Sym) {
	child := findchild(structdie, field)
	if child == nil {
		Exitf("dwarf substitutetype: %s does not have member %s",
			getattr(structdie, dwarf.DW_AT_name).Data, field)
		return
	}

	a := getattr(child, dwarf.DW_AT_type)
	if a != nil {
		a.Data = dwSym(dwtype)
	} else {
		d.newrefattr(child, dwarf.DW_AT_type, dwtype)
	}
}

func (d *dwctxt) findprotodie(ctxt *Link, name string) *dwarf.DWDie {
	die, ok := prototypedies[name]
	if ok && die == nil {
		d.defgotype(d.lookupOrDiag(name))
		die = prototypedies[name]
	}
	if die == nil {
		log.Fatalf("internal error: DIE generation failed for %s\n", name)
	}
	return die
}

func (d *dwctxt) synthesizestringtypes(ctxt *Link, die *dwarf.DWDie) {
	prototype := walktypedef(d.findprotodie(ctxt, "type:runtime.stringStructDWARF"))
	if prototype == nil {
		return
	}

	for ; die != nil; die = die.Link {
		if die.Abbrev != dwarf.DW_ABRV_STRINGTYPE {
			continue
		}
		d.copychildren(ctxt, die, prototype)
	}
}

func (d *dwctxt) synthesizeslicetypes(ctxt *Link, die *dwarf.DWDie) {
	prototype := walktypedef(d.findprotodie(ctxt, "type:runtime.slice"))
	if prototype == nil {
		return
	}

	for ; die != nil; die = die.Link {
		if die.Abbrev != dwarf.DW_ABRV_SLICETYPE {
			continue
		}
		d.copychildren(ctxt, die, prototype)
		elem := loader.Sym(getattr(die, dwarf.DW_AT_go_elem).Data.(dwSym))
		d.substitutetype(die, "array", d.defptrto(elem))
	}
}

func mkinternaltypename(base string, arg1 string, arg2 string) string {
	if arg2 == "" {
		return fmt.Sprintf("%s<%s>", base, arg1)
	}
	return fmt.Sprintf("%s<%s,%s>", base, arg1, arg2)
}

func (d *dwctxt) mkinternaltype(ctxt *Link, abbrev int, typename, keyname, valname string, f func(*dwarf.DWDie)) loader.Sym {
	name := mkinternaltypename(typename, keyname, valname)
	symname := dwarf.InfoPrefix + name
	s := d.ldr.Lookup(symname, 0)
	if s != 0 && d.ldr.SymType(s) == sym.SDWARFTYPE {
		return s
	}
	die := d.newdie(&dwtypes, abbrev, name)
	f(die)
	return d.dtolsym(die.Sym)
}

func (d *dwctxt) synthesizemaptypes(ctxt *Link, die *dwarf.DWDie) {
	mapType := walktypedef(d.findprotodie(ctxt, "type:internal/runtime/maps.Map"))
	tableType := walktypedef(d.findprotodie(ctxt, "type:internal/runtime/maps.table"))
	groupsReferenceType := walktypedef(d.findprotodie(ctxt, "type:internal/runtime/maps.groupsReference"))

	for ; die != nil; die = die.Link {
		if die.Abbrev != dwarf.DW_ABRV_MAPTYPE {
			continue
		}
		gotype := loader.Sym(getattr(die, dwarf.DW_AT_type).Data.(dwSym))

		keyType := decodetypeMapKey(d.ldr, d.arch, gotype)
		valType := decodetypeMapValue(d.ldr, d.arch, gotype)
		groupType := decodetypeMapGroup(d.ldr, d.arch, gotype)

		keyType = d.walksymtypedef(d.defgotype(keyType))
		valType = d.walksymtypedef(d.defgotype(valType))
		groupType = d.walksymtypedef(d.defgotype(groupType))

		keyName := d.nameFromDIESym(keyType)
		valName := d.nameFromDIESym(valType)

		// Construct groupsReference[K,V]
		dwGroupsReference := d.mkinternaltype(ctxt, dwarf.DW_ABRV_STRUCTTYPE, "groupReference", keyName, valName, func(dwh *dwarf.DWDie) {
			d.copychildren(ctxt, dwh, groupsReferenceType)
			// data *group[K,V]
			//
			// This is actually a pointer to an array
			// *[lengthMask+1]group[K,V], but the length is
			// variable, so we can't statically record the length.
			d.substitutetype(dwh, "data", d.defptrto(groupType))
			newattr(dwh, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, getattr(groupsReferenceType, dwarf.DW_AT_byte_size).Value, nil)
			newattr(dwh, dwarf.DW_AT_go_kind, dwarf.DW_CLS_CONSTANT, int64(abi.Struct), 0)
		})

		// Construct table[K,V]
		dwTable := d.mkinternaltype(ctxt, dwarf.DW_ABRV_STRUCTTYPE, "table", keyName, valName, func(dwh *dwarf.DWDie) {
			d.copychildren(ctxt, dwh, tableType)
			d.substitutetype(dwh, "groups", dwGroupsReference)
			newattr(dwh, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, getattr(tableType, dwarf.DW_AT_byte_size).Value, nil)
			newattr(dwh, dwarf.DW_AT_go_kind, dwarf.DW_CLS_CONSTANT, int64(abi.Struct), 0)
		})

		// Construct map[K,V]
		dwMap := d.mkinternaltype(ctxt, dwarf.DW_ABRV_STRUCTTYPE, "map", keyName, valName, func(dwh *dwarf.DWDie) {
			d.copychildren(ctxt, dwh, mapType)
			// dirPtr is a pointer to a variable-length array of
			// *table[K,V], of length dirLen.
			//
			// Since we can't directly define a variable-length
			// array, store this as **table[K,V]. i.e., pointer to
			// the first entry in the array.
			d.substitutetype(dwh, "dirPtr", d.defptrto(d.defptrto(dwTable)))
			newattr(dwh, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, getattr(mapType, dwarf.DW_AT_byte_size).Value, nil)
			newattr(dwh, dwarf.DW_AT_go_kind, dwarf.DW_CLS_CONSTANT, int64(abi.Struct), 0)
		})

		// make map type a pointer to map[K,V]
		d.newrefattr(die, dwarf.DW_AT_type, d.defptrto(dwMap))
	}
}

func (d *dwctxt) synthesizechantypes(ctxt *Link, die *dwarf.DWDie) {
	sudog := walktypedef(d.findprotodie(ctxt, "type:runtime.sudog"))
	waitq := walktypedef(d.findprotodie(ctxt, "type:runtime.waitq"))
	hchan := walktypedef(d.findprotodie(ctxt, "type:runtime.hchan"))
	if sudog == nil || waitq == nil || hchan == nil {
		return
	}

	sudogsize := int(getattr(sudog, dwarf.DW_AT_byte_size).Value)

	for ; die != nil; die = die.Link {
		if die.Abbrev != dwarf.DW_ABRV_CHANTYPE {
			continue
		}
		elemgotype := loader.Sym(getattr(die, dwarf.DW_AT_type).Data.(dwSym))
		tname := d.ldr.SymName(elemgotype)
		elemname := tname[5:]
		elemtype := d.walksymtypedef(d.defgotype(d.lookupOrDiag(tname)))

		// sudog<T>
		dwss := d.mkinternaltype(ctxt, dwarf.DW_ABRV_STRUCTTYPE, "sudog", elemname, "", func(dws *dwarf.DWDie) {
			d.copychildren(ctxt, dws, sudog)
			d.substitutetype(dws, "elem", d.defptrto(elemtype))
			newattr(dws, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, int64(sudogsize), nil)
		})

		// waitq<T>
		dwws := d.mkinternaltype(ctxt, dwarf.DW_ABRV_STRUCTTYPE, "waitq", elemname, "", func(dww *dwarf.DWDie) {

			d.copychildren(ctxt, dww, waitq)
			d.substitutetype(dww, "first", d.defptrto(dwss))
			d.substitutetype(dww, "last", d.defptrto(dwss))
			newattr(dww, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, getattr(waitq, dwarf.DW_AT_byte_size).Value, nil)
		})

		// hchan<T>
		dwhs := d.mkinternaltype(ctxt, dwarf.DW_ABRV_STRUCTTYPE, "hchan", elemname, "", func(dwh *dwarf.DWDie) {
			d.copychildren(ctxt, dwh, hchan)
			d.substitutetype(dwh, "recvq", dwws)
			d.substitutetype(dwh, "sendq", dwws)
			newattr(dwh, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, getattr(hchan, dwarf.DW_AT_byte_size).Value, nil)
		})

		d.newrefattr(die, dwarf.DW_AT_type, d.defptrto(dwhs))
	}
}

// createUnitLength creates the initial length field with value v and update
// offset of unit_length if needed.
func (d *dwctxt) createUnitLength(su *loader.SymbolBuilder, v uint64) {
	if isDwarf64(d.linkctxt) {
		su.AddUint32(d.arch, 0xFFFFFFFF)
	}
	d.addDwarfAddrField(su, v)
}

// addDwarfAddrField adds a DWARF field in DWARF 64bits or 32bits.
func (d *dwctxt) addDwarfAddrField(sb *loader.SymbolBuilder, v uint64) {
	if isDwarf64(d.linkctxt) {
		sb.AddUint(d.arch, v)
	} else {
		sb.AddUint32(d.arch, uint32(v))
	}
}

// addDwarfAddrRef adds a DWARF pointer in DWARF 64bits or 32bits.
func (d *dwctxt) addDwarfAddrRef(sb *loader.SymbolBuilder, t loader.Sym) {
	if isDwarf64(d.linkctxt) {
		d.adddwarfref(sb, t, 8)
	} else {
		d.adddwarfref(sb, t, 4)
	}
}

// calcCompUnitRanges calculates the PC ranges of the compilation units.
func (d *dwctxt) calcCompUnitRanges() {
	var prevUnit *sym.CompilationUnit
	for _, s := range d.linkctxt.Textp {
		sym := s

		fi := d.ldr.FuncInfo(sym)
		if !fi.Valid() {
			continue
		}

		// Skip linker-created functions (ex: runtime.addmoduledata), since they
		// don't have DWARF to begin with.
		unit := d.ldr.SymUnit(sym)
		if unit == nil {
			continue
		}

		// Update PC ranges.
		//
		// We don't simply compare the end of the previous
		// symbol with the start of the next because there's
		// often a little padding between them. Instead, we
		// only create boundaries between symbols from
		// different units.
		sval := d.ldr.SymValue(sym)
		u0val := d.ldr.SymValue(unit.Textp[0])
		if prevUnit != unit {
			unit.PCs = append(unit.PCs, dwarf.Range{Start: sval - u0val})
			prevUnit = unit
		}
		unit.PCs[len(unit.PCs)-1].End = sval - u0val + int64(len(d.ldr.Data(sym)))
	}
}

func movetomodule(ctxt *Link, parent *dwarf.DWDie) {
	die := ctxt.runtimeCU.DWInfo.Child
	if die == nil {
		ctxt.runtimeCU.DWInfo.Child = parent.Child
		return
	}
	for die.Link != nil {
		die = die.Link
	}
	die.Link = parent.Child
}

/*
 * Generate a sequence of opcodes that is as short as possible.
 * See section 6.2.5
 */
const (
	LINE_BASE   = -4
	LINE_RANGE  = 10
	PC_RANGE    = (255 - OPCODE_BASE) / LINE_RANGE
	OPCODE_BASE = 11
)

/*
 * Walk prog table, emit line program and build DIE tree.
 */

func getCompilationDir() string {
	// OSX requires this be set to something, but it's not easy to choose
	// a value. Linking takes place in a temporary directory, so there's
	// no point including it here. Paths in the file table are usually
	// absolute, in which case debuggers will ignore this value. -trimpath
	// produces relative paths, but we don't know where they start, so
	// all we can do here is try not to make things worse.
	return "."
}

func (d *dwctxt) importInfoSymbol(dsym loader.Sym) {
	d.ldr.SetAttrReachable(dsym, true)
	d.ldr.SetAttrNotInSymbolTable(dsym, true)
	dst := d.ldr.SymType(dsym)
	if dst != sym.SDWARFCONST && dst != sym.SDWARFABSFCN {
		log.Fatalf("error: DWARF info sym %d/%s with incorrect type %s", dsym, d.ldr.SymName(dsym), d.ldr.SymType(dsym).String())
	}
	relocs := d.ldr.Relocs(dsym)
	for i := 0; i < relocs.Count(); i++ {
		r := relocs.At(i)
		if r.Type() != objabi.R_DWARFSECREF {
			continue
		}
		rsym := r.Sym()
		// If there is an entry for the symbol in our rtmap, then it
		// means we've processed the type already, and can skip this one.
		if _, ok := d.rtmap[rsym]; ok {
			// type already generated
			continue
		}
		// FIXME: is there a way we could avoid materializing the
		// symbol name here?
		sn := d.ldr.SymName(rsym)
		tn := sn[len(dwarf.InfoPrefix):]
		ts := d.ldr.Lookup("type:"+tn, 0)
		d.defgotype(ts)
	}
}

func expandFile(fname string) string {
	fname = strings.TrimPrefix(fname, src.FileSymPrefix)
	return expandGoroot(fname)
}

// writeDirFileTables emits the portion of the DWARF line table
// prologue containing the include directories and file names,
// described in section 6.2.4 of the DWARF standard. It walks the
// filepaths for the unit to discover any common directories, which
// are emitted to the directory table first, then the file table is
// emitted after that.
//
// Note that there are some differences betwen DWARF versions 4 and 5
// regarding how files and directories are handled; the chief item is
// that in version 4 we have an implicit directory index 0 that holds
// the compilation dir, and an implicit file index 0 that holds the
// "primary" source file being compiled (with the assumption that
// we'll get these two pieces of info from attributes on the
// compilation unit DIE). DWARF version 5 does away with these
// implicit entries; if you want to refer to a file or dir, you have
// to mention it explicitly.
//
// So, let's say we have a Go package with import path
// "github.com/fruit/bowl" with two source files, "apple.go" and
// "orange.go". The directory and file table in version 4 might look
// like:
//
//	Dir Table:
//	1		github.com/fruit/bowl
//
//	File Table:
//	Entry	Dir	Time	Size	Name
//	1		0	0		0		<autogenerated>
//	2		1	0		0		apple.go
//	3		1	0		0		orange.go
//
// With DWARF version 5, the file and directory tables might look like
//
//	Dir Table:
//	0		.
//	1		github.com/fruit/bowl
//
//	File Table:
//	Entry	Dir	  Name
//	0		0	  ?                /* see remark below about this entry */
//	1		0	  <autogenerated>
//	2		1	  apple.go
//	3		1	  orange.go
//
// Also worth noting that the line table prolog allows you to control
// what items or fields appear in file and die entries (as opposed
// to the fixed dir/time/size/name format for files in DWARF 4).
func (d *dwctxt) writeDirFileTables(unit *sym.CompilationUnit, lsu *loader.SymbolBuilder) {
	type fileDir struct {
		base string
		dir  int
	}
	dirNums := make(map[string]int)
	dirs := []string{"."}
	dirNums["."] = 0
	files := []fileDir{}
	if buildcfg.Experiment.Dwarf5 {
		// Add a dummy zero entry (not used for anything) so as to
		// ensure that the first useful file index is 1, since that's
		// the default value of the line table file register).  Note
		// the "?"; this was originally an empty string, but doing
		// this triggers crashes in some versions of GDB, see
		// https://sourceware.org/bugzilla/show_bug.cgi?id=30357 for
		// the details.
		files = append(files, fileDir{base: "?", dir: 0})
	}

	// Preprocess files to collect directories. This assumes that the
	// file table is already de-duped.
	for i, name := range unit.FileTable {
		name := expandFile(name)
		if len(name) == 0 {
			// Can't have empty filenames, and having a unique
			// filename is quite useful for debugging.
			name = fmt.Sprintf("<missing>_%d", i)
		}
		// Note the use of "path" here and not "filepath". The compiler
		// hard-codes to use "/" in DWARF paths (even for Windows), so we
		// want to maintain that here.
		file := path.Base(name)
		dir := path.Dir(name)
		dirIdx, ok := dirNums[dir]
		if !ok && dir != "." {
			dirIdx = len(dirNums)
			dirNums[dir] = dirIdx
			dirs = append(dirs, dir)
		}
		files = append(files, fileDir{base: file, dir: dirIdx})

		// We can't use something that may be dead-code
		// eliminated from a binary here. proc.go contains
		// main and the scheduler, so it's not going anywhere.
		if i := strings.Index(name, "runtime/proc.go"); i >= 0 && unit.Lib.Pkg == "runtime" {
			d.dwmu.Lock()
			if gdbscript == "" {
				k := strings.Index(name, "runtime/proc.go")
				gdbscript = name[:k] + "runtime/runtime-gdb.py"
			}
			d.dwmu.Unlock()
		}
	}

	lsDwsym := dwSym(lsu.Sym())
	if buildcfg.Experiment.Dwarf5 {
		// Emit DWARF5 directory and file sections. The key added
		// element here for version 5 is that we emit a small prolog
		// describing the format of each file/dir entry, then a count
		// of the entries, then the entries themselves.

		// Dir table first...
		lsu.AddUint8(1) // directory_entry_format_count
		// each directory entry is just a single path string.
		dwarf.Uleb128put(d, lsDwsym, dwarf.DW_LNCT_path)
		dwarf.Uleb128put(d, lsDwsym, dwarf.DW_FORM_string)
		// emit count, then dirs
		dwarf.Uleb128put(d, lsDwsym, int64(len(dirs)))
		for k := 0; k < len(dirs); k++ {
			d.AddString(lsDwsym, dirs[k])
		}

		// ... now file table.  With DWARF version 5 we put out a
		// prolog that describes the format of each file entry; our
		// chosen format is a pair <F,D> where F is the file and D is
		// the dir index (zero-based).
		lsu.AddUint8(2) // file_entry_format_count
		// each file entry is just a file path followed by dir index.
		dwarf.Uleb128put(d, lsDwsym, dwarf.DW_LNCT_path)
		dwarf.Uleb128put(d, lsDwsym, dwarf.DW_FORM_string)
		dwarf.Uleb128put(d, lsDwsym, dwarf.DW_LNCT_directory_index)
		dwarf.Uleb128put(d, lsDwsym, dwarf.DW_FORM_udata)
		// emit count, then files
		dwarf.Uleb128put(d, lsDwsym, int64(len(files)))
		for k := 0; k < len(files); k++ {
			d.AddString(lsDwsym, files[k].base)
			dwarf.Uleb128put(d, lsDwsym, int64(files[k].dir))
		}
	} else {
		// Emit DWARF4 directory and file sections.

		// Dir table first. This just a series of nul terminated
		// strings, followed by a single zero byte.
		for k := 1; k < len(dirs); k++ {
			d.AddString(lsDwsym, dirs[k])
		}
		lsu.AddUint8(0) // terminator needed in V4 case.

		// File table next. This is a series of
		// file/dirindex/mtime/length tuples (where dirindex is
		// 1-based) followed by a single zero byte.
		for k := 0; k < len(files); k++ {
			d.AddString(lsDwsym, files[k].base)
			dwarf.Uleb128put(d, lsDwsym, int64(files[k].dir))
			lsu.AddUint8(0) // mtime
			lsu.AddUint8(0) // length
		}
		lsu.AddUint8(0) // terminator
	}
}

// writelines collects up and chains together the symbols needed to
// form the DWARF line table for the specified compilation unit,
// returning a list of symbols. The returned list will include an
// initial symbol containing the line table header and prologue (with
// file table), then a series of compiler-emitted line table symbols
// (one per live function), and finally an epilog symbol containing an
// end-of-sequence operator. The prologue and epilog symbols are passed
// in (having been created earlier); here we add content to them.
func (d *dwctxt) writelines(unit *sym.CompilationUnit, lineProlog loader.Sym) []loader.Sym {
	is_stmt := uint8(1) // initially = recommended default_is_stmt = 1, tracks is_stmt toggles.

	unitstart := int64(-1)
	headerstart := int64(-1)
	headerend := int64(-1)

	syms := make([]loader.Sym, 0, len(unit.Textp)+2)
	syms = append(syms, lineProlog)
	lsu := d.ldr.MakeSymbolUpdater(lineProlog)
	lsDwsym := dwSym(lineProlog)
	newattr(unit.DWInfo, dwarf.DW_AT_stmt_list, dwarf.DW_CLS_PTR, 0, lsDwsym)

	// Write .debug_line Line Number Program Header (sec 6.2.4)
	// Fields marked with (*) must be changed for 64-bit dwarf
	unitLengthOffset := lsu.Size()
	d.createUnitLength(lsu, 0) // unit_length (*), filled in at end
	unitstart = lsu.Size()
	if buildcfg.Experiment.Dwarf5 {
		lsu.AddUint16(d.arch, 5)
		// DWARF5 requires address_size and segment selector sizes
		// here, both ubyte format.
		lsu.AddUint8(uint8(d.arch.PtrSize))
		lsu.AddUint8(0)
	} else {
		lsu.AddUint16(d.arch, 2) // dwarf version (appendix F) -- version 3 is incompatible w/ XCode 9.0's dsymutil, latest supported on OSX 10.12 as of 2018-05
	}
	headerLengthOffset := lsu.Size()
	d.addDwarfAddrField(lsu, 0) // header_length (*), filled in at end
	headerstart = lsu.Size()

	lsu.AddUint8(1) // minimum_instruction_length
	if buildcfg.Experiment.Dwarf5 {
		lsu.AddUint8(1) // maximum_operations_per_instruction
	}
	lsu.AddUint8(is_stmt)          // default_is_stmt
	lsu.AddUint8(LINE_BASE & 0xFF) // line_base
	lsu.AddUint8(LINE_RANGE)       // line_range
	lsu.AddUint8(OPCODE_BASE)      // opcode_base
	lsu.AddUint8(0)                // standard_opcode_lengths[1]
	lsu.AddUint8(1)                // standard_opcode_lengths[2]
	lsu.AddUint8(1)                // standard_opcode_lengths[3]
	lsu.AddUint8(1)                // standard_opcode_lengths[4]
	lsu.AddUint8(1)                // standard_opcode_lengths[5]
	lsu.AddUint8(0)                // standard_opcode_lengths[6]
	lsu.AddUint8(0)                // standard_opcode_lengths[7]
	lsu.AddUint8(0)                // standard_opcode_lengths[8]
	lsu.AddUint8(1)                // standard_opcode_lengths[9]
	lsu.AddUint8(0)                // standard_opcode_lengths[10]

	// Call helper to emit dir and file sections.
	d.writeDirFileTables(unit, lsu)

	// capture length at end of file names.
	headerend = lsu.Size()
	unitlen := lsu.Size() - unitstart

	// Output the state machine for each function remaining.
	for _, s := range unit.Textp {
		fnSym := s
		_, _, _, lines := d.ldr.GetFuncDwarfAuxSyms(fnSym)

		// Chain the line symbol onto the list.
		if lines != 0 {
			syms = append(syms, lines)
			unitlen += int64(len(d.ldr.Data(lines)))
		}
	}

	if d.linkctxt.HeadType == objabi.Haix {
		addDwsectCUSize(".debug_line", unit.Lib.Pkg, uint64(unitlen))
	}

	if isDwarf64(d.linkctxt) {
		lsu.SetUint(d.arch, unitLengthOffset+4, uint64(unitlen)) // +4 because of 0xFFFFFFFF
		lsu.SetUint(d.arch, headerLengthOffset, uint64(headerend-headerstart))
	} else {
		lsu.SetUint32(d.arch, unitLengthOffset, uint32(unitlen))
		lsu.SetUint32(d.arch, headerLengthOffset, uint32(headerend-headerstart))
	}

	return syms
}

// writepcranges generates the DW_AT_ranges table for compilation unit
// "unit", and returns a collection of ranges symbols (one for the
// compilation unit DIE itself and the remainder from functions in the unit).
func (d *dwctxt) writepcranges(unit *sym.CompilationUnit, base loader.Sym, pcs []dwarf.Range, rangeProlog loader.Sym, debugaddrsym loader.Sym) []loader.Sym {

	syms := make([]loader.Sym, 0, len(unit.RangeSyms)+1)
	syms = append(syms, rangeProlog)
	rsu := d.ldr.MakeSymbolUpdater(rangeProlog)
	rDwSym := dwSym(rangeProlog)

	// Create PC ranges for the compilation unit DIE.
	newattr(unit.DWInfo, dwarf.DW_AT_ranges, dwarf.DW_CLS_PTR, rsu.Size(), rDwSym)
	newattr(unit.DWInfo, dwarf.DW_AT_low_pc, dwarf.DW_CLS_ADDRESS, 0, dwSym(base))
	if buildcfg.Experiment.Dwarf5 && debugaddrsym != 0 {
		debugaddr := d.ldr.MakeSymbolUpdater(debugaddrsym)
		// Write DWARF5-based ranges for the unit. This will introduce
		// a series of new R_DWTXTADDR_* relocs, which we'll process
		// in the loop immediately following this call.
		dwarf.PutRngListRanges(d, rDwSym, dwSym(base), pcs)
		drelocs := d.ldr.Relocs(rangeProlog)
		for ri := 0; ri < drelocs.Count(); ri++ {
			r := drelocs.At(ri)
			if !r.Type().IsDwTxtAddr() {
				continue
			}
			cusym := d.dtolsym(unit.DWInfo.Sym)
			d.assignDebugAddrSlot(unit, cusym, r, debugaddr)
		}
	} else {
		dwarf.PutBasedRanges(d, rDwSym, pcs)
	}

	// Collect up the ranges for functions in the unit.
	rsize := uint64(rsu.Size())
	for _, ls := range unit.RangeSyms {
		s := ls
		syms = append(syms, s)
		rsize += uint64(d.ldr.SymSize(s))
	}

	if d.linkctxt.HeadType == objabi.Haix {
		addDwsectCUSize(".debug_ranges", unit.Lib.Pkg, rsize)
	}

	return syms
}

/*
 *  Emit .debug_frame
 */
const (
	dataAlignmentFactor = -4
)

// appendPCDeltaCFA appends per-PC CFA deltas to b and returns the final slice.
func appendPCDeltaCFA(arch *sys.Arch, b []byte, deltapc, cfa int64) []byte {
	b = append(b, dwarf.DW_CFA_def_cfa_offset_sf)
	b = dwarf.AppendSleb128(b, cfa/dataAlignmentFactor)

	switch {
	case deltapc < 0x40:
		b = append(b, uint8(dwarf.DW_CFA_advance_loc+deltapc))
	case deltapc < 0x100:
		b = append(b, dwarf.DW_CFA_advance_loc1)
		b = append(b, uint8(deltapc))
	case deltapc < 0x10000:
		b = append(b, dwarf.DW_CFA_advance_loc2, 0, 0)
		arch.ByteOrder.PutUint16(b[len(b)-2:], uint16(deltapc))
	default:
		b = append(b, dwarf.DW_CFA_advance_loc4, 0, 0, 0, 0)
		arch.ByteOrder.PutUint32(b[len(b)-4:], uint32(deltapc))
	}
	return b
}

func (d *dwctxt) writeframes(fs loader.Sym) dwarfSecInfo {
	fsd := dwSym(fs)
	fsu := d.ldr.MakeSymbolUpdater(fs)
	fsu.SetType(sym.SDWARFSECT)
	isdw64 := isDwarf64(d.linkctxt)
	haslr := d.linkctxt.Arch.HasLR

	// Length field is 4 bytes on Dwarf32 and 12 bytes on Dwarf64
	lengthFieldSize := int64(4)
	if isdw64 {
		lengthFieldSize += 8
	}

	// Emit the CIE, Section 6.4.1
	cieReserve := uint32(16)
	if haslr {
		cieReserve = 32
	}
	if isdw64 {
		cieReserve += 4 // 4 bytes added for cid
	}
	d.createUnitLength(fsu, uint64(cieReserve))         // initial length, must be multiple of thearch.ptrsize
	d.addDwarfAddrField(fsu, ^uint64(0))                // cid
	fsu.AddUint8(3)                                     // dwarf version (appendix F)
	fsu.AddUint8(0)                                     // augmentation ""
	dwarf.Uleb128put(d, fsd, 1)                         // code_alignment_factor
	dwarf.Sleb128put(d, fsd, dataAlignmentFactor)       // all CFI offset calculations include multiplication with this factor
	dwarf.Uleb128put(d, fsd, int64(thearch.Dwarfreglr)) // return_address_register

	fsu.AddUint8(dwarf.DW_CFA_def_cfa)                  // Set the current frame address..
	dwarf.Uleb128put(d, fsd, int64(thearch.Dwarfregsp)) // ...to use the value in the platform's SP register (defined in l.go)...
	if haslr {
		dwarf.Uleb128put(d, fsd, int64(0)) // ...plus a 0 offset.

		fsu.AddUint8(dwarf.DW_CFA_same_value) // The platform's link register is unchanged during the prologue.
		dwarf.Uleb128put(d, fsd, int64(thearch.Dwarfreglr))

		fsu.AddUint8(dwarf.DW_CFA_val_offset)               // The previous value...
		dwarf.Uleb128put(d, fsd, int64(thearch.Dwarfregsp)) // ...of the platform's SP register...
		dwarf.Uleb128put(d, fsd, int64(0))                  // ...is CFA+0.
	} else {
		dwarf.Uleb128put(d, fsd, int64(d.arch.PtrSize)) // ...plus the word size (because the call instruction implicitly adds one word to the frame).

		fsu.AddUint8(dwarf.DW_CFA_offset_extended)                           // The previous value...
		dwarf.Uleb128put(d, fsd, int64(thearch.Dwarfreglr))                  // ...of the return address...
		dwarf.Uleb128put(d, fsd, int64(-d.arch.PtrSize)/dataAlignmentFactor) // ...is saved at [CFA - (PtrSize/4)].
	}

	pad := int64(cieReserve) + lengthFieldSize - int64(len(d.ldr.Data(fs)))

	if pad < 0 {
		Exitf("dwarf: cieReserve too small by %d bytes.", -pad)
	}

	internalExec := d.linkctxt.BuildMode == BuildModeExe && d.linkctxt.IsInternal()
	addAddrPlus := loader.GenAddAddrPlusFunc(internalExec)

	fsu.AddBytes(zeros[:pad])

	var deltaBuf []byte
	pcsp := obj.NewPCIter(uint32(d.arch.MinLC))
	for _, s := range d.linkctxt.Textp {
		fn := s
		fi := d.ldr.FuncInfo(fn)
		if !fi.Valid() {
			continue
		}
		fpcsp := d.ldr.Pcsp(s)

		// Emit a FDE, Section 6.4.1.
		// First build the section contents into a byte buffer.
		deltaBuf = deltaBuf[:0]
		if haslr && fi.TopFrame() {
			// Mark the link register as having an undefined value.
			// This stops call stack unwinders progressing any further.
			// TODO: similar mark on non-LR architectures.
			deltaBuf = append(deltaBuf, dwarf.DW_CFA_undefined)
			deltaBuf = dwarf.AppendUleb128(deltaBuf, uint64(thearch.Dwarfreglr))
		}

		for pcsp.Init(d.linkctxt.loader.Data(fpcsp)); !pcsp.Done; pcsp.Next() {
			nextpc := pcsp.NextPC

			// pciterinit goes up to the end of the function,
			// but DWARF expects us to stop just before the end.
			if int64(nextpc) == int64(len(d.ldr.Data(fn))) {
				nextpc--
				if nextpc < pcsp.PC {
					continue
				}
			}

			spdelta := int64(pcsp.Value)
			if !haslr {
				// Return address has been pushed onto stack.
				spdelta += int64(d.arch.PtrSize)
			}

			if haslr && !fi.TopFrame() {
				// TODO(bryanpkc): This is imprecise. In general, the instruction
				// that stores the return address to the stack frame is not the
				// same one that allocates the frame.
				if pcsp.Value > 0 {
					// The return address is preserved at (CFA-frame_size)
					// after a stack frame has been allocated.
					deltaBuf = append(deltaBuf, dwarf.DW_CFA_offset_extended_sf)
					deltaBuf = dwarf.AppendUleb128(deltaBuf, uint64(thearch.Dwarfreglr))
					deltaBuf = dwarf.AppendSleb128(deltaBuf, -spdelta/dataAlignmentFactor)
				} else {
					// The return address is restored into the link register
					// when a stack frame has been de-allocated.
					deltaBuf = append(deltaBuf, dwarf.DW_CFA_same_value)
					deltaBuf = dwarf.AppendUleb128(deltaBuf, uint64(thearch.Dwarfreglr))
				}
			}

			deltaBuf = appendPCDeltaCFA(d.arch, deltaBuf, int64(nextpc)-int64(pcsp.PC), spdelta)
		}
		pad := int(Rnd(int64(len(deltaBuf)), int64(d.arch.PtrSize))) - len(deltaBuf)
		deltaBuf = append(deltaBuf, zeros[:pad]...)

		// Emit the FDE header, Section 6.4.1.
		//	4 bytes: length, must be multiple of thearch.ptrsize
		//	4/8 bytes: Pointer to the CIE above, at offset 0
		//	ptrsize: initial location
		//	ptrsize: address range

		fdeLength := uint64(4 + 2*d.arch.PtrSize + len(deltaBuf))
		if isdw64 {
			fdeLength += 4 // 4 bytes added for CIE pointer
		}
		d.createUnitLength(fsu, fdeLength)

		if d.linkctxt.LinkMode == LinkExternal {
			d.addDwarfAddrRef(fsu, fs)
		} else {
			d.addDwarfAddrField(fsu, 0) // CIE offset
		}
		addAddrPlus(fsu, d.arch, s, 0)
		fsu.AddUintXX(d.arch, uint64(len(d.ldr.Data(fn))), d.arch.PtrSize) // address range
		fsu.AddBytes(deltaBuf)

		if d.linkctxt.HeadType == objabi.Haix {
			addDwsectCUSize(".debug_frame", d.ldr.SymPkg(fn), fdeLength+uint64(lengthFieldSize))
		}
	}

	return dwarfSecInfo{syms: []loader.Sym{fs}}
}

/*
 *  Walk DWarfDebugInfoEntries, and emit .debug_info
 */

func (d *dwctxt) writeUnitInfo(u *sym.CompilationUnit, abbrevsym loader.Sym, addrsym loader.Sym, infoEpilog loader.Sym) []loader.Sym {
	syms := []loader.Sym{}
	if len(u.Textp) == 0 && u.DWInfo.Child == nil && len(u.VarDIEs) == 0 {
		return syms
	}

	compunit := u.DWInfo
	s := d.dtolsym(compunit.Sym)
	su := d.ldr.MakeSymbolUpdater(s)

	// Write .debug_info Compilation Unit Header (sec 7.5.1)
	// Fields marked with (*) must be changed for 64-bit dwarf
	d.createUnitLength(su, 0) // unit_length (*), will be filled in later.
	dwver := 4
	if buildcfg.Experiment.Dwarf5 {
		dwver = 5
	}
	su.AddUint16(d.arch, uint16(dwver)) // dwarf version (appendix F)

	if buildcfg.Experiment.Dwarf5 {
		// DWARF5 at this point requires
		// 1. unit type
		// 2. address size
		// 3. abbrev offset
		su.AddUint8(uint8(dwarf.DW_UT_compile))
		su.AddUint8(uint8(d.arch.PtrSize))
		d.addDwarfAddrRef(su, abbrevsym)
	} else {
		// DWARF4 requires
		// 1. abbrev offset
		// 2. address size
		d.addDwarfAddrRef(su, abbrevsym)
		su.AddUint8(uint8(d.arch.PtrSize))
	}

	ds := dwSym(s)
	dwarf.Uleb128put(d, ds, int64(compunit.Abbrev))
	if buildcfg.Experiment.Dwarf5 {
		// If this CU has functions, update the DW_AT_addr_base
		// attribute to point to the correct section symbol.
		abattr := getattr(compunit, dwarf.DW_AT_addr_base)
		if abattr != nil {
			abattr.Data = dwSym(addrsym)
		}
	}
	dwarf.PutAttrs(d, ds, compunit.Abbrev, compunit.Attr)

	// This is an under-estimate; more will be needed for type DIEs.
	cu := make([]loader.Sym, 0, len(u.AbsFnDIEs)+len(u.FuncDIEs))
	cu = append(cu, s)
	cu = append(cu, u.AbsFnDIEs...)
	cu = append(cu, u.FuncDIEs...)
	if u.Consts != 0 {
		cu = append(cu, u.Consts)
	}
	cu = append(cu, u.VarDIEs...)
	var cusize int64
	for _, child := range cu {
		cusize += int64(len(d.ldr.Data(child)))
	}

	for die := compunit.Child; die != nil; die = die.Link {
		l := len(cu)
		lastSymSz := int64(len(d.ldr.Data(cu[l-1])))
		cu = d.putdie(cu, die)
		if lastSymSz != int64(len(d.ldr.Data(cu[l-1]))) {
			// putdie will sometimes append directly to the last symbol of the list
			cusize = cusize - lastSymSz + int64(len(d.ldr.Data(cu[l-1])))
		}
		for _, child := range cu[l:] {
			cusize += int64(len(d.ldr.Data(child)))
		}
	}

	culu := d.ldr.MakeSymbolUpdater(infoEpilog)
	culu.AddUint8(0) // closes compilation unit DIE
	cu = append(cu, infoEpilog)
	cusize++

	// Save size for AIX symbol table.
	if d.linkctxt.HeadType == objabi.Haix {
		addDwsectCUSize(".debug_info", d.getPkgFromCUSym(s), uint64(cusize))
	}
	if isDwarf64(d.linkctxt) {
		cusize -= 12                          // exclude the length field.
		su.SetUint(d.arch, 4, uint64(cusize)) // 4 because of 0XFFFFFFFF
	} else {
		cusize -= 4 // exclude the length field.
		su.SetUint32(d.arch, 0, uint32(cusize))
	}
	return append(syms, cu...)
}

func (d *dwctxt) writegdbscript() dwarfSecInfo {
	// TODO (aix): make it available
	if d.linkctxt.HeadType == objabi.Haix {
		return dwarfSecInfo{}
	}
	if d.linkctxt.LinkMode == LinkExternal && d.linkctxt.HeadType == objabi.Hwindows && d.linkctxt.BuildMode == BuildModeCArchive {
		// gcc on Windows places .debug_gdb_scripts in the wrong location, which
		// causes the program not to run. See https://golang.org/issue/20183
		// Non c-archives can avoid this issue via a linker script
		// (see fix near writeGDBLinkerScript).
		// c-archive users would need to specify the linker script manually.
		// For UX it's better not to deal with this.
		return dwarfSecInfo{}
	}
	if gdbscript == "" {
		return dwarfSecInfo{}
	}

	gs := d.ldr.CreateSymForUpdate(".debug_gdb_scripts", 0)
	gs.SetType(sym.SDWARFSECT)

	gs.AddUint8(GdbScriptPythonFileId)
	gs.Addstring(gdbscript)
	return dwarfSecInfo{syms: []loader.Sym{gs.Sym()}}
}

// FIXME: might be worth looking replacing this map with a function
// that switches based on symbol instead.

var prototypedies map[string]*dwarf.DWDie

func dwarfEnabled(ctxt *Link) bool {
	if *FlagW { // disable dwarf
		return false
	}
	if ctxt.HeadType == objabi.Hplan9 || ctxt.HeadType == objabi.Hjs || ctxt.HeadType == objabi.Hwasip1 {
		return false
	}

	if ctxt.LinkMode == LinkExternal {
		switch {
		case ctxt.IsELF:
		case ctxt.HeadType == objabi.Hdarwin:
		case ctxt.HeadType == objabi.Hwindows:
		case ctxt.HeadType == objabi.Haix:
			res, err := dwarf.IsDWARFEnabledOnAIXLd(ctxt.extld())
			if err != nil {
				Exitf("%v", err)
			}
			return res
		default:
			return false
		}
	}

	return true
}

// mkBuiltinType populates the dwctxt2 sym lookup maps for the
// newly created builtin type DIE 'typeDie'.
func (d *dwctxt) mkBuiltinType(ctxt *Link, abrv int, tname string) *dwarf.DWDie {
	// create type DIE
	die := d.newdie(&dwtypes, abrv, tname)

	// Look up type symbol.
	gotype := d.lookupOrDiag("type:" + tname)

	// Map from die sym to type sym
	ds := loader.Sym(die.Sym.(dwSym))
	d.rtmap[ds] = gotype

	// Map from type to def sym
	d.tdmap[gotype] = ds

	return die
}

// assignDebugAddrSlot assigns a slot (if needed) in the .debug_addr
// section fragment of the specified unit for the function pointed to
// by R_DWTXTADDR_* relocation r.  The slot index selected will be
// filled in when the relocation is actually applied/resolved.
func (d *dwctxt) assignDebugAddrSlot(unit *sym.CompilationUnit, fnsym loader.Sym, r loader.Reloc, sb *loader.SymbolBuilder) {
	rsym := r.Sym()
	if unit.Addrs == nil {
		unit.Addrs = make(map[sym.LoaderSym]uint32)
	}
	if _, ok := unit.Addrs[rsym]; ok {
		// already present, no work needed
	} else {
		sl := len(unit.Addrs)
		rt := r.Type()
		lim, _ := rt.DwTxtAddrRelocParams()
		if sl > lim {
			log.Fatalf("internal error: %s relocation overflow on infosym for %s", rt.String(), d.ldr.SymName(fnsym))
		}
		unit.Addrs[rsym] = uint32(sl)
		sb.AddAddrPlus(d.arch, rsym, 0)
		data := sb.Data()
		if d.arch.PtrSize == 4 {
			d.arch.ByteOrder.PutUint32(data[len(data)-4:], uint32(d.ldr.SymValue(rsym)))
		} else {
			d.arch.ByteOrder.PutUint64(data[len(data)-8:], uint64(d.ldr.SymValue(rsym)))
		}
	}
}

// dwarfVisitFunction takes a function (text) symbol and processes the
// subprogram DIE for the function and picks up any other DIEs
// (absfns, types) that it references.
func (d *dwctxt) dwarfVisitFunction(fnSym loader.Sym, unit *sym.CompilationUnit) {
	// The DWARF subprogram DIE symbol is listed as an aux sym
	// of the text (fcn) symbol, so ask the loader to retrieve it,
	// as well as the associated range symbol.
	infosym, _, rangesym, _ := d.ldr.GetFuncDwarfAuxSyms(fnSym)
	if infosym == 0 {
		return
	}
	d.ldr.SetAttrNotInSymbolTable(infosym, true)
	d.ldr.SetAttrReachable(infosym, true)
	unit.FuncDIEs = append(unit.FuncDIEs, infosym)
	if rangesym != 0 {
		d.ldr.SetAttrNotInSymbolTable(rangesym, true)
		d.ldr.SetAttrReachable(rangesym, true)
		unit.RangeSyms = append(unit.RangeSyms, rangesym)
	}

	// Walk the relocations of the subprogram DIE symbol to discover
	// references to abstract function DIEs, Go type DIES, and
	// (via R_USETYPE relocs) types that were originally assigned to
	// locals/params but were optimized away.
	drelocs := d.ldr.Relocs(infosym)
	for ri := 0; ri < drelocs.Count(); ri++ {
		r := drelocs.At(ri)
		// Look for "use type" relocs.
		if r.Type() == objabi.R_USETYPE {
			d.defgotype(r.Sym())
			continue
		}
		if r.Type() != objabi.R_DWARFSECREF {
			continue
		}

		rsym := r.Sym()
		rst := d.ldr.SymType(rsym)

		// Look for abstract function references.
		if rst == sym.SDWARFABSFCN {
			if !d.ldr.AttrOnList(rsym) {
				// abstract function
				d.ldr.SetAttrOnList(rsym, true)
				unit.AbsFnDIEs = append(unit.AbsFnDIEs, rsym)
				d.importInfoSymbol(rsym)
			}
			continue
		}

		// Look for type references.
		if rst != sym.SDWARFTYPE && rst != sym.Sxxx {
			continue
		}
		if _, ok := d.rtmap[rsym]; ok {
			// type already generated
			continue
		}

		rsn := d.ldr.SymName(rsym)
		tn := rsn[len(dwarf.InfoPrefix):]
		ts := d.ldr.Lookup("type:"+tn, 0)
		d.defgotype(ts)
	}
}

// dwarfGenerateDebugInfo generated debug info entries for all types,
// variables and functions in the program.
// Along with dwarfGenerateDebugSyms they are the two main entry points into
// dwarf generation: dwarfGenerateDebugInfo does all the work that should be
// done before symbol names are mangled while dwarfGenerateDebugSyms does
// all the work that can only be done after addresses have been assigned to
// text symbols.
func dwarfGenerateDebugInfo(ctxt *Link) {
	if !dwarfEnabled(ctxt) {
		return
	}

	d := &dwctxt{
		linkctxt: ctxt,
		ldr:      ctxt.loader,
		arch:     ctxt.Arch,
		tmap:     make(map[string]loader.Sym),
		tdmap:    make(map[loader.Sym]loader.Sym),
		rtmap:    make(map[loader.Sym]loader.Sym),
	}
	d.typeRuntimeEface = d.lookupOrDiag("type:runtime.eface")
	d.typeRuntimeIface = d.lookupOrDiag("type:runtime.iface")

	if ctxt.HeadType == objabi.Haix {
		// Initial map used to store package size for each DWARF section.
		dwsectCUSize = make(map[string]uint64)
	}

	// For ctxt.Diagnostic messages.
	newattr(&dwtypes, dwarf.DW_AT_name, dwarf.DW_CLS_STRING, int64(len("dwtypes")), "dwtypes")

	// Unspecified type. There are no references to this in the symbol table.
	d.newdie(&dwtypes, dwarf.DW_ABRV_NULLTYPE, "<unspecified>")

	// Some types that must exist to define other ones (uintptr in particular
	// is needed for array size)
	unsafeptrDie := d.mkBuiltinType(ctxt, dwarf.DW_ABRV_BARE_PTRTYPE, "unsafe.Pointer")
	newattr(unsafeptrDie, dwarf.DW_AT_go_runtime_type, dwarf.DW_CLS_GO_TYPEREF, 0, dwSym(d.lookupOrDiag("type:unsafe.Pointer")))
	uintptrDie := d.mkBuiltinType(ctxt, dwarf.DW_ABRV_BASETYPE, "uintptr")
	newattr(uintptrDie, dwarf.DW_AT_encoding, dwarf.DW_CLS_CONSTANT, dwarf.DW_ATE_unsigned, 0)
	newattr(uintptrDie, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, int64(d.arch.PtrSize), 0)
	newattr(uintptrDie, dwarf.DW_AT_go_kind, dwarf.DW_CLS_CONSTANT, int64(abi.Uintptr), 0)
	newattr(uintptrDie, dwarf.DW_AT_go_runtime_type, dwarf.DW_CLS_GO_TYPEREF, 0, dwSym(d.lookupOrDiag("type:uintptr")))

	d.uintptrInfoSym = d.mustFind("uintptr")

	// Prototypes needed for type synthesis.
	prototypedies = map[string]*dwarf.DWDie{
		"type:runtime.stringStructDWARF":             nil,
		"type:runtime.slice":                         nil,
		"type:runtime.sudog":                         nil,
		"type:runtime.waitq":                         nil,
		"type:runtime.hchan":                         nil,
		"type:internal/runtime/maps.Map":             nil,
		"type:internal/runtime/maps.table":           nil,
		"type:internal/runtime/maps.groupsReference": nil,
	}

	// Needed by the prettyprinter code for interface inspection.
	for _, typ := range []string{
		"type:internal/abi.Type",
		"type:internal/abi.ArrayType",
		"type:internal/abi.ChanType",
		"type:internal/abi.FuncType",
		"type:internal/abi.MapType",
		"type:internal/abi.PtrType",
		"type:internal/abi.SliceType",
		"type:internal/abi.StructType",
		"type:internal/abi.InterfaceType",
		"type:internal/abi.ITab",
		"type:internal/abi.Imethod"} {
		d.defgotype(d.lookupOrDiag(typ))
	}

	// fake root DIE for compile unit DIEs
	var dwroot dwarf.DWDie
	flagVariants := make(map[string]bool)

	for _, lib := range ctxt.Library {

		consts := d.ldr.Lookup(dwarf.ConstInfoPrefix+lib.Pkg, 0)
		for _, unit := range lib.Units {
			// We drop the constants into the first CU.
			if consts != 0 {
				unit.Consts = consts
				d.importInfoSymbol(consts)
				consts = 0
			}
			ctxt.compUnits = append(ctxt.compUnits, unit)

			// We need at least one runtime unit.
			if unit.Lib.Pkg == "runtime" {
				ctxt.runtimeCU = unit
			}

			cuabrv := dwarf.DW_ABRV_COMPUNIT
			if len(unit.Textp) == 0 {
				cuabrv = dwarf.DW_ABRV_COMPUNIT_TEXTLESS
			}
			unit.DWInfo = d.newdie(&dwroot, cuabrv, unit.Lib.Pkg)
			newattr(unit.DWInfo, dwarf.DW_AT_language, dwarf.DW_CLS_CONSTANT, int64(dwarf.DW_LANG_Go), 0)
			// OS X linker requires compilation dir or absolute path in comp unit name to output debug info.
			compDir := getCompilationDir()
			// TODO: Make this be the actual compilation directory, not
			// the linker directory. If we move CU construction into the
			// compiler, this should happen naturally.
			newattr(unit.DWInfo, dwarf.DW_AT_comp_dir, dwarf.DW_CLS_STRING, int64(len(compDir)), compDir)

			var peData []byte
			if producerExtra := d.ldr.Lookup(dwarf.CUInfoPrefix+"producer."+unit.Lib.Pkg, 0); producerExtra != 0 {
				peData = d.ldr.Data(producerExtra)
			}
			producer := "Go cmd/compile " + buildcfg.Version
			if len(peData) > 0 {
				// We put a semicolon before the flags to clearly
				// separate them from the version, which can be long
				// and have lots of weird things in it in development
				// versions. We promise not to put a semicolon in the
				// version, so it should be safe for readers to scan
				// forward to the semicolon.
				producer += "; " + string(peData)
				flagVariants[string(peData)] = true
			} else {
				flagVariants[""] = true
			}

			newattr(unit.DWInfo, dwarf.DW_AT_producer, dwarf.DW_CLS_STRING, int64(len(producer)), producer)

			var pkgname string
			if pnSymIdx := d.ldr.Lookup(dwarf.CUInfoPrefix+"packagename."+unit.Lib.Pkg, 0); pnSymIdx != 0 {
				pnsData := d.ldr.Data(pnSymIdx)
				pkgname = string(pnsData)
			}
			newattr(unit.DWInfo, dwarf.DW_AT_go_package_name, dwarf.DW_CLS_STRING, int64(len(pkgname)), pkgname)

			if buildcfg.Experiment.Dwarf5 && cuabrv == dwarf.DW_ABRV_COMPUNIT {
				// For DWARF5, the CU die will have an attribute that
				// points to the offset into the .debug_addr section
				// that contains the compilation unit's
				// contributions. Add this attribute now. Note that
				// we'll later on update the Data field in this attr
				// once we know the .debug_addr sym for the unit.
				newattr(unit.DWInfo, dwarf.DW_AT_addr_base, dwarf.DW_CLS_REFERENCE, 0, 0)
			}

			// Scan all functions in this compilation unit, create
			// DIEs for all referenced types, find all referenced
			// abstract functions, visit range symbols. Note that
			// Textp has been dead-code-eliminated already.
			for _, s := range unit.Textp {
				d.dwarfVisitFunction(s, unit)
			}
		}
	}

	// Fix for 31034: if the objects feeding into this link were compiled
	// with different sets of flags, then don't issue an error if
	// the -strictdups checks fail.
	if checkStrictDups > 1 && len(flagVariants) > 1 {
		checkStrictDups = 1
	}

	// Make a pass through all data symbols, looking for those
	// corresponding to reachable, Go-generated, user-visible
	// global variables. For each global of this sort, locate
	// the corresponding compiler-generated DIE symbol and tack
	// it onto the list associated with the unit.
	// Also looks for dictionary symbols and generates DIE symbols for each
	// type they reference.
	for idx := loader.Sym(1); idx < loader.Sym(d.ldr.NDef()); idx++ {
		if !d.ldr.AttrReachable(idx) ||
			d.ldr.AttrNotInSymbolTable(idx) ||
			d.ldr.SymVersion(idx) >= sym.SymVerStatic {
			continue
		}
		t := d.ldr.SymType(idx)
		switch {
		case t.IsRODATA(), t.IsDATA(), t.IsNOPTRDATA(),
			t == sym.STYPE, t == sym.SBSS, t == sym.SNOPTRBSS, t == sym.STLSBSS:
			// ok
		default:
			continue
		}
		// Skip things with no type, unless it's a dictionary
		gt := d.ldr.SymGoType(idx)
		if gt == 0 {
			if t == sym.SRODATA {
				if d.ldr.IsDict(idx) {
					// This is a dictionary, make sure that all types referenced by this dictionary are reachable
					relocs := d.ldr.Relocs(idx)
					for i := 0; i < relocs.Count(); i++ {
						reloc := relocs.At(i)
						if reloc.Type() == objabi.R_USEIFACE {
							d.defgotype(reloc.Sym())
						}
					}
				}
			}
			continue
		}
		// Skip file local symbols (this includes static tmps, stack
		// object symbols, and local symbols in assembler src files).
		if d.ldr.IsFileLocal(idx) {
			continue
		}

		// Find compiler-generated DWARF info sym for global in question,
		// and tack it onto the appropriate unit.  Note that there are
		// circumstances under which we can't find the compiler-generated
		// symbol-- this typically happens as a result of compiler options
		// (e.g. compile package X with "-dwarf=0").
		varDIE := d.ldr.GetVarDwarfAuxSym(idx)
		if varDIE != 0 {
			unit := d.ldr.SymUnit(idx)
			d.defgotype(gt)
			unit.VarDIEs = append(unit.VarDIEs, varDIE)
		}
	}

	d.synthesizestringtypes(ctxt, dwtypes.Child)
	d.synthesizeslicetypes(ctxt, dwtypes.Child)
	d.synthesizemaptypes(ctxt, dwtypes.Child)
	d.synthesizechantypes(ctxt, dwtypes.Child)
}

// dwarfGenerateDebugSyms constructs debug_line, debug_frame, and
// debug_loc. It also writes out the debug_info section using symbols
// generated in dwarfGenerateDebugInfo.
func dwarfGenerateDebugSyms(ctxt *Link) {
	if !dwarfEnabled(ctxt) {
		return
	}
	d := &dwctxt{
		linkctxt: ctxt,
		ldr:      ctxt.loader,
		arch:     ctxt.Arch,
		dwmu:     new(sync.Mutex),
	}
	d.dwarfGenerateDebugSyms()
}

// dwUnitSyms stores input and output symbols for DWARF generation
// for a given compilation unit.
type dwUnitSyms struct {
	// Inputs for a given unit.
	lineProlog  loader.Sym
	rangeProlog loader.Sym
	infoEpilog  loader.Sym

	// Outputs for a given unit.
	linesyms   []loader.Sym
	infosyms   []loader.Sym
	locsyms    []loader.Sym
	rangessyms []loader.Sym
	addrsym    loader.Sym
}

// dwUnitPortion assembles the DWARF content for a given comp unit:
// debug_info, debug_lines, debug_ranges(V4) or debug_rnglists (V5),
// debug_loc (V4) or debug_loclists (V5) and debug_addr (V5);
// debug_frame is handled elsewhere. Order is important; the calls to
// writelines and writepcranges below make updates to the compilation
// unit DIE, hence they have to happen before the call to
// writeUnitInfo.
func (d *dwctxt) dwUnitPortion(u *sym.CompilationUnit, abbrevsym loader.Sym, us *dwUnitSyms) {
	if u.DWInfo.Abbrev != dwarf.DW_ABRV_COMPUNIT_TEXTLESS {
		us.linesyms = d.writelines(u, us.lineProlog)
		base := u.Textp[0]
		if buildcfg.Experiment.Dwarf5 {
			d.writedebugaddr(u, us.addrsym)
		}
		us.rangessyms = d.writepcranges(u, base, u.PCs, us.rangeProlog, us.addrsym)
		us.locsyms = d.collectUnitLocs(u)
	}
	us.infosyms = d.writeUnitInfo(u, abbrevsym, us.addrsym, us.infoEpilog)
}

// writedebugaddr scans the symbols of interest in unit for
// R_DWTXTADDR_R* relocations and converts these into the material
// we'll need to generate the .debug_addr section for the unit. This
// will create a map within the unit (as a side effect), mapping func
// symbol to debug_addr slot.
func (d *dwctxt) writedebugaddr(unit *sym.CompilationUnit, debugaddr loader.Sym) {
	dasu := d.ldr.MakeSymbolUpdater(debugaddr)

	var dsyms []loader.Sym
	for _, s := range unit.Textp {
		fnSym := s
		// NB: this looks at SDWARFFCN; it will need to also look
		// at range and loc when they get there.
		infosym, locsym, rangessym, _ := d.ldr.GetFuncDwarfAuxSyms(fnSym)

		// Walk the relocations of the various DWARF symbols to
		// collect relocations corresponding to indirect function
		// references via .debug_addr.
		dsyms = dsyms[:0]
		dsyms = append(dsyms, infosym)
		if rangessym != 0 {
			dsyms = append(dsyms, rangessym)
		}
		if locsym != 0 {
			dsyms = append(dsyms, locsym)
		}
		for _, dsym := range dsyms {
			drelocs := d.ldr.Relocs(dsym)
			for ri := 0; ri < drelocs.Count(); ri++ {
				r := drelocs.At(ri)
				if !r.Type().IsDwTxtAddr() {
					continue
				}
				rsym := r.Sym()
				rst := d.ldr.SymType(rsym)
				// Do some consistency checks.
				if !rst.IsText() {
					// R_DWTXTADDR_* relocation should only refer to
					// text symbols, so something apparently went
					// wrong here.
					log.Fatalf("internal error: R_DWTXTADDR_* relocation on dwinfosym for %s against non-function %s type:%s", d.ldr.SymName(fnSym), d.ldr.SymName(rsym), rst.String())
				}
				if runit := d.ldr.SymUnit(rsym); runit != unit {
					log.Fatalf("internal error: R_DWTXTADDR_* relocation target text sym unit mismatch (want %q got %q)", unit.Lib.Pkg, runit.Lib.Pkg)
				}
				d.assignDebugAddrSlot(unit, fnSym, r, dasu)
			}
		}
	}
}

func (d *dwctxt) dwarfGenerateDebugSyms() {
	abbrevSec := d.writeabbrev()
	dwarfp = append(dwarfp, abbrevSec)
	d.calcCompUnitRanges()
	slices.SortFunc(d.linkctxt.compUnits, compilationUnitByStartPCCmp)

	// newdie adds DIEs to the *beginning* of the parent's DIE list.
	// Now that we're done creating DIEs, reverse the trees so DIEs
	// appear in the order they were created.
	for _, u := range d.linkctxt.compUnits {
		reversetree(&u.DWInfo.Child)
	}
	reversetree(&dwtypes.Child)
	movetomodule(d.linkctxt, &dwtypes)

	mkSecSym := func(name string) loader.Sym {
		s := d.ldr.CreateSymForUpdate(name, 0)
		s.SetType(sym.SDWARFSECT)
		s.SetReachable(true)
		return s.Sym()
	}

	mkAnonSym := func(kind sym.SymKind) loader.Sym {
		s := d.ldr.MakeSymbolUpdater(d.ldr.CreateExtSym("", 0))
		s.SetType(kind)
		s.SetReachable(true)
		return s.Sym()
	}

	// Create the section symbols.
	frameSym := mkSecSym(".debug_frame")
	lineSym := mkSecSym(".debug_line")
	var rangesSym, locSym loader.Sym
	if buildcfg.Experiment.Dwarf5 {
		rangesSym = mkSecSym(".debug_rnglists")
		locSym = mkSecSym(".debug_loclists")
	} else {
		rangesSym = mkSecSym(".debug_ranges")
		locSym = mkSecSym(".debug_loc")
	}
	infoSym := mkSecSym(".debug_info")
	var addrSym loader.Sym
	if buildcfg.Experiment.Dwarf5 {
		addrSym = mkSecSym(".debug_addr")
	}

	// Create the section objects
	lineSec := dwarfSecInfo{syms: []loader.Sym{lineSym}}
	frameSec := dwarfSecInfo{syms: []loader.Sym{frameSym}}
	infoSec := dwarfSecInfo{syms: []loader.Sym{infoSym}}
	var addrSec, rangesSec, locSec dwarfSecInfo
	if buildcfg.Experiment.Dwarf5 {
		addrHdr := d.writeDebugAddrHdr()
		addrSec.syms = []loader.Sym{addrSym, addrHdr}
		rnglistsHdr := d.writeDebugRngListsHdr()
		rangesSec.syms = []loader.Sym{rangesSym, rnglistsHdr}
		loclistsHdr := d.writeDebugLocListsHdr()
		locSec.syms = []loader.Sym{locSym, loclistsHdr}
	} else {
		rangesSec = dwarfSecInfo{syms: []loader.Sym{rangesSym}}
		locSec = dwarfSecInfo{syms: []loader.Sym{locSym}}
	}

	// Create any new symbols that will be needed during the
	// parallel portion below.
	ncu := len(d.linkctxt.compUnits)
	unitSyms := make([]dwUnitSyms, ncu)
	for i := 0; i < ncu; i++ {
		us := &unitSyms[i]
		us.lineProlog = mkAnonSym(sym.SDWARFLINES)
		us.rangeProlog = mkAnonSym(sym.SDWARFRANGE)
		us.infoEpilog = mkAnonSym(sym.SDWARFFCN)
		us.addrsym = mkAnonSym(sym.SDWARFADDR)
	}

	var wg sync.WaitGroup
	sema := make(chan struct{}, runtime.GOMAXPROCS(0))

	// Kick off generation of .debug_frame, since it doesn't have
	// any entanglements and can be started right away.
	wg.Add(1)
	go func() {
		sema <- struct{}{}
		defer func() {
			<-sema
			wg.Done()
		}()
		frameSec = d.writeframes(frameSym)
	}()

	// Create a goroutine per comp unit to handle the generation that
	// unit's portion of .debug_line, .debug_loc, .debug_ranges, and
	// .debug_info.
	wg.Add(len(d.linkctxt.compUnits))
	for i := 0; i < ncu; i++ {
		go func(u *sym.CompilationUnit, us *dwUnitSyms) {
			sema <- struct{}{}
			defer func() {
				<-sema
				wg.Done()
			}()
			d.dwUnitPortion(u, abbrevSec.secSym(), us)
		}(d.linkctxt.compUnits[i], &unitSyms[i])
	}
	wg.Wait()

	markReachable := func(syms []loader.Sym) []loader.Sym {
		for _, s := range syms {
			d.ldr.SetAttrNotInSymbolTable(s, true)
			d.ldr.SetAttrReachable(s, true)
		}
		return syms
	}

	patchHdr := func(sec *dwarfSecInfo, len uint64) {
		hdrsym := sec.syms[1]
		len += uint64(d.ldr.SymSize(hdrsym))
		su := d.ldr.MakeSymbolUpdater(hdrsym)
		if isDwarf64(d.linkctxt) {
			len -= 12                  // sub size of length field
			su.SetUint(d.arch, 4, len) // 4 because of 0XFFFFFFFF
		} else {
			len -= 4 // subtract size of length field
			su.SetUint32(d.arch, 0, uint32(len))
		}
	}

	if buildcfg.Experiment.Dwarf5 {
		// Compute total size of the DWARF5-specific .debug_* syms in
		// each compilation unit.
		var rltot, addrtot, loctot uint64
		for i := 0; i < ncu; i++ {
			addrtot += uint64(d.ldr.SymSize(unitSyms[i].addrsym))
			rs := unitSyms[i].rangessyms
			for _, s := range rs {
				rltot += uint64(d.ldr.SymSize(s))
			}
			loc := unitSyms[i].locsyms
			for _, s := range loc {
				loctot += uint64(d.ldr.SymSize(s))
			}
		}
		// Call a helper to patch the length field in the headers.
		patchHdr(&addrSec, addrtot)
		patchHdr(&rangesSec, rltot)
		patchHdr(&locSec, loctot)
	}

	// Stitch together the results.
	for i := 0; i < ncu; i++ {
		r := &unitSyms[i]
		lineSec.syms = append(lineSec.syms, markReachable(r.linesyms)...)
		infoSec.syms = append(infoSec.syms, markReachable(r.infosyms)...)
		locSec.syms = append(locSec.syms, markReachable(r.locsyms)...)
		rangesSec.syms = append(rangesSec.syms, markReachable(r.rangessyms)...)
		if buildcfg.Experiment.Dwarf5 && r.addrsym != 0 {
			addrSec.syms = append(addrSec.syms, r.addrsym)
		}
	}
	dwarfp = append(dwarfp, lineSec)
	dwarfp = append(dwarfp, frameSec)
	gdbScriptSec := d.writegdbscript()
	if gdbScriptSec.secSym() != 0 {
		dwarfp = append(dwarfp, gdbScriptSec)
	}
	dwarfp = append(dwarfp, infoSec)
	if len(locSec.syms) > 1 {
		dwarfp = append(dwarfp, locSec)
	}
	dwarfp = append(dwarfp, rangesSec)
	if buildcfg.Experiment.Dwarf5 {
		dwarfp = append(dwarfp, addrSec)
	}

	// Check to make sure we haven't listed any symbols more than once
	// in the info section. This used to be done by setting and
	// checking the OnList attribute in "putdie", but that strategy
	// was not friendly for concurrency.
	seen := loader.MakeBitmap(d.ldr.NSym())
	for _, s := range infoSec.syms {
		if seen.Has(s) {
			log.Fatalf("dwarf symbol %s listed multiple times",
				d.ldr.SymName(s))
		}
		seen.Set(s)
	}
}

func (d *dwctxt) collectUnitLocs(u *sym.CompilationUnit) []loader.Sym {
	syms := []loader.Sym{}
	for _, fn := range u.FuncDIEs {
		relocs := d.ldr.Relocs(fn)
		for i := 0; i < relocs.Count(); i++ {
			reloc := relocs.At(i)
			if reloc.Type() != objabi.R_DWARFSECREF {
				continue
			}
			rsym := reloc.Sym()
			if d.ldr.SymType(rsym) == sym.SDWARFLOC {
				syms = append(syms, rsym)
				// One location list entry per function, but many relocations to it. Don't duplicate.
				break
			}
		}
	}
	return syms
}

// Add DWARF section names to the section header string table, by calling add
// on each name. ELF only.
func dwarfaddshstrings(ctxt *Link, add func(string)) {
	if *FlagW { // disable dwarf
		return
	}

	secs := []string{"abbrev", "frame", "info", "loc", "line", "gdb_scripts"}
	if buildcfg.Experiment.Dwarf5 {
		secs = append(secs, "addr", "rnglists", "loclists")
	} else {
		secs = append(secs, "ranges", "loc")
	}

	for _, sec := range secs {
		add(".debug_" + sec)
		if ctxt.IsExternal() {
			add(elfRelType + ".debug_" + sec)
		}
	}
}

func dwarfaddelfsectionsyms(ctxt *Link) {
	if *FlagW { // disable dwarf
		return
	}
	if ctxt.LinkMode != LinkExternal {
		return
	}

	ldr := ctxt.loader
	for _, si := range dwarfp {
		s := si.secSym()
		sect := ldr.SymSect(si.secSym())
		putelfsectionsym(ctxt, ctxt.Out, s, sect.Elfsect.(*ElfShdr).shnum)
	}
}

// dwarfcompress compresses the DWARF sections. Relocations are applied
// on the fly. After this, dwarfp will contain a different (new) set of
// symbols, and sections may have been replaced.
func dwarfcompress(ctxt *Link) {
	// compressedSect is a helper type for parallelizing compression.
	type compressedSect struct {
		index      int
		compressed []byte
		syms       []loader.Sym
	}

	supported := ctxt.IsELF || ctxt.IsWindows() || ctxt.IsDarwin()
	if !ctxt.compressDWARF || !supported || ctxt.IsExternal() {
		return
	}

	var compressedCount int
	resChannel := make(chan compressedSect)
	for i := range dwarfp {
		go func(resIndex int, syms []loader.Sym) {
			resChannel <- compressedSect{resIndex, compressSyms(ctxt, syms), syms}
		}(compressedCount, dwarfp[i].syms)
		compressedCount++
	}
	res := make([]compressedSect, compressedCount)
	for ; compressedCount > 0; compressedCount-- {
		r := <-resChannel
		res[r.index] = r
	}

	ldr := ctxt.loader
	var newDwarfp []dwarfSecInfo
	Segdwarf.Sections = Segdwarf.Sections[:0]
	for _, z := range res {
		s := z.syms[0]
		if z.compressed == nil {
			// Compression didn't help.
			ds := dwarfSecInfo{syms: z.syms}
			newDwarfp = append(newDwarfp, ds)
			Segdwarf.Sections = append(Segdwarf.Sections, ldr.SymSect(s))
		} else {
			var compressedSegName string
			if ctxt.IsELF {
				compressedSegName = ldr.SymSect(s).Name
			} else {
				compressedSegName = ".zdebug_" + ldr.SymSect(s).Name[len(".debug_"):]
			}
			sect := addsection(ctxt.loader, ctxt.Arch, &Segdwarf, compressedSegName, 04)
			sect.Align = int32(ctxt.Arch.Alignment)
			sect.Length = uint64(len(z.compressed))
			sect.Compressed = true
			newSym := ldr.MakeSymbolBuilder(compressedSegName)
			ldr.SetAttrReachable(s, true)
			newSym.SetData(z.compressed)
			newSym.SetSize(int64(len(z.compressed)))
			ldr.SetSymSect(newSym.Sym(), sect)
			ds := dwarfSecInfo{syms: []loader.Sym{newSym.Sym()}}
			newDwarfp = append(newDwarfp, ds)

			// compressed symbols are no longer needed.
			for _, s := range z.syms {
				ldr.SetAttrReachable(s, false)
				ldr.FreeSym(s)
			}
		}
	}
	dwarfp = newDwarfp

	// Re-compute the locations of the compressed DWARF symbols
	// and sections, since the layout of these within the file is
	// based on Section.Vaddr and Symbol.Value.
	pos := Segdwarf.Vaddr
	var prevSect *sym.Section
	for _, si := range dwarfp {
		for _, s := range si.syms {
			sect := ldr.SymSect(s)
			if sect != prevSect {
				if ctxt.IsWindows() {
					pos = uint64(Rnd(int64(pos), PEFILEALIGN))
				}
				sect.Vaddr = pos
				prevSect = sect
			}
			ldr.SetSymValue(s, int64(pos))
			if ldr.SubSym(s) != 0 {
				log.Fatalf("%s: unexpected sub-symbols", ldr.SymName(s))
			}
			pos += uint64(ldr.SymSize(s))
		}
	}
	Segdwarf.Length = pos - Segdwarf.Vaddr
}

func compilationUnitByStartPCCmp(a, b *sym.CompilationUnit) int {
	switch {
	case len(a.Textp) == 0 && len(b.Textp) == 0:
		return strings.Compare(a.Lib.Pkg, b.Lib.Pkg)
	case len(a.Textp) != 0 && len(b.Textp) == 0:
		return -1
	case len(a.Textp) == 0 && len(b.Textp) != 0:
		return +1
	default:
		return cmp.Compare(a.PCs[0].Start, b.PCs[0].Start)
	}
}

// getPkgFromCUSym returns the package name for the compilation unit
// represented by s.
// The prefix dwarf.InfoPrefix+".pkg." needs to be removed in order to get
// the package name.
func (d *dwctxt) getPkgFromCUSym(s loader.Sym) string {
	return strings.TrimPrefix(d.ldr.SymName(s), dwarf.InfoPrefix+".pkg.")
}

// On AIX, the symbol table needs to know where are the compilation units parts
// for a specific package in each .dw section.
// dwsectCUSize map will save the size of a compilation unit for
// the corresponding .dw section.
// This size can later be retrieved with the index "sectionName.pkgName".
var dwsectCUSizeMu sync.Mutex
var dwsectCUSize map[string]uint64

// getDwsectCUSize retrieves the corresponding package size inside the current section.
func getDwsectCUSize(sname string, pkgname string) uint64 {
	return dwsectCUSize[sname+"."+pkgname]
}

func addDwsectCUSize(sname string, pkgname string, size uint64) {
	dwsectCUSizeMu.Lock()
	defer dwsectCUSizeMu.Unlock()
	dwsectCUSize[sname+"."+pkgname] += size
}

// writeDebugMiscSecHdr writes a header section for the new new DWARF5
// sections ".debug_addr", ".debug_loclists", and ".debug_rnglists".
// A description of the format/layout of these headers can be found in
// the DWARF5 spec in sections 7.27 (.debug_addr), 7.28
// (.debug_rnglists) and 7.29 (.debug_loclists).
func (d *dwctxt) writeDebugMiscSecHdr(st sym.SymKind, addOffsetEntryCount bool) loader.Sym {
	su := d.ldr.MakeSymbolUpdater(d.ldr.CreateExtSym("", 0))
	su.SetType(st)
	su.SetReachable(true)
	d.createUnitLength(su, 0)          // will be filled in later.
	su.AddUint16(d.arch, 5)            // dwarf version (appendix F)
	su.AddUint8(uint8(d.arch.PtrSize)) // address_size
	su.AddUint8(0)                     // segment selector
	if addOffsetEntryCount {
		su.AddUint32(d.arch, 0) // offset entry count (required but unused)
	}
	return su.Sym()
}

func (d *dwctxt) writeDebugRngListsHdr() loader.Sym {
	return d.writeDebugMiscSecHdr(sym.SDWARFRANGE, true)
}

func (d *dwctxt) writeDebugLocListsHdr() loader.Sym {
	return d.writeDebugMiscSecHdr(sym.SDWARFLOC, true)
}

func (d *dwctxt) writeDebugAddrHdr() loader.Sym {
	return d.writeDebugMiscSecHdr(sym.SDWARFADDR, false)
}
