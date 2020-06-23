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
	"fmt"
	"log"
	"sort"
	"strings"
)

// dwctxt2 is a wrapper intended to satisfy the method set of
// dwarf.Context, so that functions like dwarf.PutAttrs will work with
// DIEs that use loader.Sym as opposed to *sym.Symbol. It is also
// being used as a place to store tables/maps that are useful as part
// of type conversion (this is just a convenience; it would be easy to
// split these things out into another type if need be).
type dwctxt2 struct {
	linkctxt *Link
	ldr      *loader.Loader
	arch     *sys.Arch

	// This maps type name string (e.g. "uintptr") to loader symbol for
	// the DWARF DIE for that type (e.g. "go.info.type.uintptr")
	tmap map[string]loader.Sym

	// This maps loader symbol for the DWARF DIE symbol generated for
	// a type (e.g. "go.info.uintptr") to the type symbol itself
	// ("type.uintptr").
	// FIXME: try converting this map (and the next one) to a single
	// array indexed by loader.Sym -- this may perform better.
	rtmap map[loader.Sym]loader.Sym

	// This maps Go type symbol (e.g. "type.XXX") to loader symbol for
	// the typedef DIE for that type (e.g. "go.info.XXX..def")
	tdmap map[loader.Sym]loader.Sym

	// Cache these type symbols, so as to avoid repeatedly looking them up
	typeRuntimeEface loader.Sym
	typeRuntimeIface loader.Sym
	uintptrInfoSym   loader.Sym
}

func newdwctxt2(linkctxt *Link, forTypeGen bool) dwctxt2 {
	d := dwctxt2{
		linkctxt: linkctxt,
		ldr:      linkctxt.loader,
		arch:     linkctxt.Arch,
		tmap:     make(map[string]loader.Sym),
		tdmap:    make(map[loader.Sym]loader.Sym),
		rtmap:    make(map[loader.Sym]loader.Sym),
	}
	d.typeRuntimeEface = d.lookupOrDiag("type.runtime.eface")
	d.typeRuntimeIface = d.lookupOrDiag("type.runtime.iface")
	return d
}

// dwSym wraps a loader.Sym; this type is meant to obey the interface
// rules for dwarf.Sym from the cmd/internal/dwarf package. DwDie and
// DwAttr objects contain references to symbols via this type.
type dwSym loader.Sym

func (s dwSym) Length(dwarfContext interface{}) int64 {
	l := dwarfContext.(dwctxt2).ldr
	return int64(len(l.Data(loader.Sym(s))))
}

func (c dwctxt2) PtrSize() int {
	return c.arch.PtrSize
}

func (c dwctxt2) AddInt(s dwarf.Sym, size int, i int64) {
	ds := loader.Sym(s.(dwSym))
	dsu := c.ldr.MakeSymbolUpdater(ds)
	dsu.AddUintXX(c.arch, uint64(i), size)
}

func (c dwctxt2) AddBytes(s dwarf.Sym, b []byte) {
	ds := loader.Sym(s.(dwSym))
	dsu := c.ldr.MakeSymbolUpdater(ds)
	dsu.AddBytes(b)
}

func (c dwctxt2) AddString(s dwarf.Sym, v string) {
	ds := loader.Sym(s.(dwSym))
	dsu := c.ldr.MakeSymbolUpdater(ds)
	dsu.Addstring(v)
}

func (c dwctxt2) AddAddress(s dwarf.Sym, data interface{}, value int64) {
	ds := loader.Sym(s.(dwSym))
	dsu := c.ldr.MakeSymbolUpdater(ds)
	if value != 0 {
		value -= dsu.Value()
	}
	tgtds := loader.Sym(data.(dwSym))
	dsu.AddAddrPlus(c.arch, tgtds, value)
}

func (c dwctxt2) AddCURelativeAddress(s dwarf.Sym, data interface{}, value int64) {
	ds := loader.Sym(s.(dwSym))
	dsu := c.ldr.MakeSymbolUpdater(ds)
	if value != 0 {
		value -= dsu.Value()
	}
	tgtds := loader.Sym(data.(dwSym))
	dsu.AddCURelativeAddrPlus(c.arch, tgtds, value)
}

func (c dwctxt2) AddSectionOffset(s dwarf.Sym, size int, t interface{}, ofs int64) {
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

func (c dwctxt2) AddDWARFAddrSectionOffset(s dwarf.Sym, t interface{}, ofs int64) {
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

func (c dwctxt2) Logf(format string, args ...interface{}) {
	c.linkctxt.Logf(format, args...)
}

// At the moment these interfaces are only used in the compiler.

func (c dwctxt2) AddFileRef(s dwarf.Sym, f interface{}) {
	panic("should be used only in the compiler")
}

func (c dwctxt2) CurrentOffset(s dwarf.Sym) int64 {
	panic("should be used only in the compiler")
}

func (c dwctxt2) RecordDclReference(s dwarf.Sym, t dwarf.Sym, dclIdx int, inlIndex int) {
	panic("should be used only in the compiler")
}

func (c dwctxt2) RecordChildDieOffsets(s dwarf.Sym, vars []*dwarf.Var, offsets []int32) {
	panic("should be used only in the compiler")
}

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

// dwarfp2 stores the collected DWARF symbols created during
// dwarf generation.
var dwarfp2 []dwarfSecInfo

func (d *dwctxt2) writeabbrev() dwarfSecInfo {
	abrvs := d.ldr.LookupOrCreateSym(".debug_abbrev", 0)
	u := d.ldr.MakeSymbolUpdater(abrvs)
	u.SetType(sym.SDWARFSECT)
	u.AddBytes(dwarf.GetAbbrev())
	return dwarfSecInfo{syms: []loader.Sym{abrvs}}
}

var dwtypes dwarf.DWDie

// newattr attaches a new attribute to the specified DIE.
//
// FIXME: at the moment attributes are stored in a linked list in a
// fairly space-inefficient way -- it might be better to instead look
// up all attrs in a single large table, then store indices into the
// table in the DIE. This would allow us to common up storage for
// attributes that are shared by many DIEs (ex: byte size of N).
func newattr(die *dwarf.DWDie, attr uint16, cls int, value int64, data interface{}) *dwarf.DWAttr {
	a := new(dwarf.DWAttr)
	a.Link = die.Attr
	die.Attr = a
	a.Atr = attr
	a.Cls = uint8(cls)
	a.Value = value
	a.Data = data
	return a
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
func (d *dwctxt2) newdie(parent *dwarf.DWDie, abbrev int, name string, version int) *dwarf.DWDie {
	die := new(dwarf.DWDie)
	die.Abbrev = abbrev
	die.Link = parent.Child
	parent.Child = die

	newattr(die, dwarf.DW_AT_name, dwarf.DW_CLS_STRING, int64(len(name)), name)

	if name != "" && (abbrev <= dwarf.DW_ABRV_VARIABLE || abbrev >= dwarf.DW_ABRV_NULLTYPE) {
		// Q: do we need version here? My understanding is that all these
		// symbols should be version 0.
		if abbrev != dwarf.DW_ABRV_VARIABLE || version == 0 {
			if abbrev == dwarf.DW_ABRV_COMPUNIT {
				// Avoid collisions with "real" symbol names.
				name = fmt.Sprintf(".pkg.%s.%d", name, len(d.linkctxt.compUnits))
			}
			ds := d.ldr.LookupOrCreateSym(dwarf.InfoPrefix+name, version)
			dsu := d.ldr.MakeSymbolUpdater(ds)
			dsu.SetType(sym.SDWARFINFO)
			d.ldr.SetAttrNotInSymbolTable(ds, true)
			d.ldr.SetAttrReachable(ds, true)
			die.Sym = dwSym(ds)
			if abbrev >= dwarf.DW_ABRV_NULLTYPE && abbrev <= dwarf.DW_ABRV_TYPEDECL {
				d.tmap[name] = ds
			}
		}
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

func (d *dwctxt2) walksymtypedef(symIdx loader.Sym) loader.Sym {

	// We're being given the loader symbol for the type DIE, e.g.
	// "go.info.type.uintptr". Map that first to the type symbol (e.g.
	// "type.uintptr") and then to the typedef DIE for the type.
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

// Used to avoid string allocation when looking up dwarf symbols
var prefixBuf = []byte(dwarf.InfoPrefix)

// find looks up the loader symbol for the DWARF DIE generated for the
// type with the specified name.
func (d *dwctxt2) find(name string) loader.Sym {
	return d.tmap[name]
}

func (d *dwctxt2) mustFind(name string) loader.Sym {
	r := d.find(name)
	if r == 0 {
		Exitf("dwarf find: cannot find %s", name)
	}
	return r
}

func (d *dwctxt2) adddwarfref(sb *loader.SymbolBuilder, t loader.Sym, size int) int64 {
	var result int64
	switch size {
	default:
		d.linkctxt.Errorf(sb.Sym(), "invalid size %d in adddwarfref\n", size)
	case d.arch.PtrSize, 4:
	}
	result = sb.AddSymRef(d.arch, t, 0, objabi.R_DWARFSECREF, size)
	return result
}

func (d *dwctxt2) newrefattr(die *dwarf.DWDie, attr uint16, ref loader.Sym) *dwarf.DWAttr {
	if ref == 0 {
		return nil
	}
	return newattr(die, attr, dwarf.DW_CLS_REFERENCE, 0, dwSym(ref))
}

func (d *dwctxt2) dtolsym(s dwarf.Sym) loader.Sym {
	if s == nil {
		return 0
	}
	dws := loader.Sym(s.(dwSym))
	return dws
}

func (d *dwctxt2) putdie(syms []loader.Sym, die *dwarf.DWDie) []loader.Sym {
	s := d.dtolsym(die.Sym)
	if s == 0 {
		s = syms[len(syms)-1]
	} else {
		if d.ldr.AttrOnList(s) {
			log.Fatalf("symbol %s listed multiple times", d.ldr.SymName(s))
		}
		d.ldr.SetAttrOnList(s, true)
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

// GDB doesn't like FORM_addr for AT_location, so emit a
// location expression that evals to a const.
func (d *dwctxt2) newabslocexprattr(die *dwarf.DWDie, addr int64, symIdx loader.Sym) {
	newattr(die, dwarf.DW_AT_location, dwarf.DW_CLS_ADDRESS, addr, dwSym(symIdx))
}

func (d *dwctxt2) lookupOrDiag(n string) loader.Sym {
	symIdx := d.ldr.Lookup(n, 0)
	if symIdx == 0 {
		Exitf("dwarf: missing type: %s", n)
	}
	if len(d.ldr.Data(symIdx)) == 0 {
		Exitf("dwarf: missing type (no data): %s", n)
	}

	return symIdx
}

func (d *dwctxt2) dotypedef(parent *dwarf.DWDie, gotype loader.Sym, name string, def *dwarf.DWDie) *dwarf.DWDie {
	// Only emit typedefs for real names.
	if strings.HasPrefix(name, "map[") {
		return nil
	}
	if strings.HasPrefix(name, "struct {") {
		return nil
	}
	if strings.HasPrefix(name, "chan ") {
		return nil
	}
	if name[0] == '[' || name[0] == '*' {
		return nil
	}
	if def == nil {
		Errorf(nil, "dwarf: bad def in dotypedef")
	}

	// Create a new loader symbol for the typedef. We no longer
	// do lookups of typedef symbols by name, so this is going
	// to be an anonymous symbol (we want this for perf reasons).
	tds := d.ldr.CreateExtSym("", 0)
	tdsu := d.ldr.MakeSymbolUpdater(tds)
	tdsu.SetType(sym.SDWARFINFO)
	def.Sym = dwSym(tds)
	d.ldr.SetAttrNotInSymbolTable(tds, true)
	d.ldr.SetAttrReachable(tds, true)

	// The typedef entry must be created after the def,
	// so that future lookups will find the typedef instead
	// of the real definition. This hooks the typedef into any
	// circular definition loops, so that gdb can understand them.
	die := d.newdie(parent, dwarf.DW_ABRV_TYPEDECL, name, 0)

	d.newrefattr(die, dwarf.DW_AT_type, tds)

	return die
}

// Define gotype, for composite ones recurse into constituents.
func (d *dwctxt2) defgotype(gotype loader.Sym) loader.Sym {
	if gotype == 0 {
		return d.mustFind("<unspecified>")
	}

	// If we already have a tdmap entry for the gotype, return it.
	if ds, ok := d.tdmap[gotype]; ok {
		return ds
	}

	sn := d.ldr.SymName(gotype)
	if !strings.HasPrefix(sn, "type.") {
		d.linkctxt.Errorf(gotype, "dwarf: type name doesn't start with \"type.\"")
		return d.mustFind("<unspecified>")
	}
	name := sn[5:] // could also decode from Type.string

	sdie := d.find(name)
	if sdie != 0 {
		return sdie
	}

	gtdwSym := d.newtype(gotype)
	d.tdmap[gotype] = loader.Sym(gtdwSym.Sym.(dwSym))
	return loader.Sym(gtdwSym.Sym.(dwSym))
}

func (d *dwctxt2) newtype(gotype loader.Sym) *dwarf.DWDie {
	sn := d.ldr.SymName(gotype)
	name := sn[5:] // could also decode from Type.string
	tdata := d.ldr.Data(gotype)
	kind := decodetypeKind(d.arch, tdata)
	bytesize := decodetypeSize(d.arch, tdata)

	var die, typedefdie *dwarf.DWDie
	switch kind {
	case objabi.KindBool:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_BASETYPE, name, 0)
		newattr(die, dwarf.DW_AT_encoding, dwarf.DW_CLS_CONSTANT, dwarf.DW_ATE_boolean, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case objabi.KindInt,
		objabi.KindInt8,
		objabi.KindInt16,
		objabi.KindInt32,
		objabi.KindInt64:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_BASETYPE, name, 0)
		newattr(die, dwarf.DW_AT_encoding, dwarf.DW_CLS_CONSTANT, dwarf.DW_ATE_signed, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case objabi.KindUint,
		objabi.KindUint8,
		objabi.KindUint16,
		objabi.KindUint32,
		objabi.KindUint64,
		objabi.KindUintptr:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_BASETYPE, name, 0)
		newattr(die, dwarf.DW_AT_encoding, dwarf.DW_CLS_CONSTANT, dwarf.DW_ATE_unsigned, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case objabi.KindFloat32,
		objabi.KindFloat64:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_BASETYPE, name, 0)
		newattr(die, dwarf.DW_AT_encoding, dwarf.DW_CLS_CONSTANT, dwarf.DW_ATE_float, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case objabi.KindComplex64,
		objabi.KindComplex128:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_BASETYPE, name, 0)
		newattr(die, dwarf.DW_AT_encoding, dwarf.DW_CLS_CONSTANT, dwarf.DW_ATE_complex_float, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case objabi.KindArray:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_ARRAYTYPE, name, 0)
		typedefdie = d.dotypedef(&dwtypes, gotype, name, die)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		s := decodetypeArrayElem(d.ldr, d.arch, gotype)
		d.newrefattr(die, dwarf.DW_AT_type, d.defgotype(s))
		fld := d.newdie(die, dwarf.DW_ABRV_ARRAYRANGE, "range", 0)

		// use actual length not upper bound; correct for 0-length arrays.
		newattr(fld, dwarf.DW_AT_count, dwarf.DW_CLS_CONSTANT, decodetypeArrayLen(d.ldr, d.arch, gotype), 0)

		d.newrefattr(fld, dwarf.DW_AT_type, d.uintptrInfoSym)

	case objabi.KindChan:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_CHANTYPE, name, 0)
		s := decodetypeChanElem(d.ldr, d.arch, gotype)
		d.newrefattr(die, dwarf.DW_AT_go_elem, d.defgotype(s))
		// Save elem type for synthesizechantypes. We could synthesize here
		// but that would change the order of DIEs we output.
		d.newrefattr(die, dwarf.DW_AT_type, s)

	case objabi.KindFunc:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_FUNCTYPE, name, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		typedefdie = d.dotypedef(&dwtypes, gotype, name, die)
		data := d.ldr.Data(gotype)
		// FIXME: add caching or reuse reloc slice.
		relocs := d.ldr.Relocs(gotype)
		nfields := decodetypeFuncInCount(d.arch, data)
		for i := 0; i < nfields; i++ {
			s := decodetypeFuncInType(d.ldr, d.arch, gotype, &relocs, i)
			sn := d.ldr.SymName(s)
			fld := d.newdie(die, dwarf.DW_ABRV_FUNCTYPEPARAM, sn[5:], 0)
			d.newrefattr(fld, dwarf.DW_AT_type, d.defgotype(s))
		}

		if decodetypeFuncDotdotdot(d.arch, data) {
			d.newdie(die, dwarf.DW_ABRV_DOTDOTDOT, "...", 0)
		}
		nfields = decodetypeFuncOutCount(d.arch, data)
		for i := 0; i < nfields; i++ {
			s := decodetypeFuncOutType(d.ldr, d.arch, gotype, &relocs, i)
			sn := d.ldr.SymName(s)
			fld := d.newdie(die, dwarf.DW_ABRV_FUNCTYPEPARAM, sn[5:], 0)
			d.newrefattr(fld, dwarf.DW_AT_type, d.defptrto(d.defgotype(s)))
		}

	case objabi.KindInterface:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_IFACETYPE, name, 0)
		typedefdie = d.dotypedef(&dwtypes, gotype, name, die)
		data := d.ldr.Data(gotype)
		nfields := int(decodetypeIfaceMethodCount(d.arch, data))
		var s loader.Sym
		if nfields == 0 {
			s = d.typeRuntimeEface
		} else {
			s = d.typeRuntimeIface
		}
		d.newrefattr(die, dwarf.DW_AT_type, d.defgotype(s))

	case objabi.KindMap:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_MAPTYPE, name, 0)
		s := decodetypeMapKey(d.ldr, d.arch, gotype)
		d.newrefattr(die, dwarf.DW_AT_go_key, d.defgotype(s))
		s = decodetypeMapValue(d.ldr, d.arch, gotype)
		d.newrefattr(die, dwarf.DW_AT_go_elem, d.defgotype(s))
		// Save gotype for use in synthesizemaptypes. We could synthesize here,
		// but that would change the order of the DIEs.
		d.newrefattr(die, dwarf.DW_AT_type, gotype)

	case objabi.KindPtr:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_PTRTYPE, name, 0)
		typedefdie = d.dotypedef(&dwtypes, gotype, name, die)
		s := decodetypePtrElem(d.ldr, d.arch, gotype)
		d.newrefattr(die, dwarf.DW_AT_type, d.defgotype(s))

	case objabi.KindSlice:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_SLICETYPE, name, 0)
		typedefdie = d.dotypedef(&dwtypes, gotype, name, die)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		s := decodetypeArrayElem(d.ldr, d.arch, gotype)
		elem := d.defgotype(s)
		d.newrefattr(die, dwarf.DW_AT_go_elem, elem)

	case objabi.KindString:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_STRINGTYPE, name, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case objabi.KindStruct:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_STRUCTTYPE, name, 0)
		typedefdie = d.dotypedef(&dwtypes, gotype, name, die)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		nfields := decodetypeStructFieldCount(d.ldr, d.arch, gotype)
		for i := 0; i < nfields; i++ {
			f := decodetypeStructFieldName(d.ldr, d.arch, gotype, i)
			s := decodetypeStructFieldType(d.ldr, d.arch, gotype, i)
			if f == "" {
				sn := d.ldr.SymName(s)
				f = sn[5:] // skip "type."
			}
			fld := d.newdie(die, dwarf.DW_ABRV_STRUCTFIELD, f, 0)
			d.newrefattr(fld, dwarf.DW_AT_type, d.defgotype(s))
			offsetAnon := decodetypeStructFieldOffsAnon(d.ldr, d.arch, gotype, i)
			newmemberoffsetattr(fld, int32(offsetAnon>>1))
			if offsetAnon&1 != 0 { // is embedded field
				newattr(fld, dwarf.DW_AT_go_embedded_field, dwarf.DW_CLS_FLAG, 1, 0)
			}
		}

	case objabi.KindUnsafePointer:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_BARE_PTRTYPE, name, 0)

	default:
		d.linkctxt.Errorf(gotype, "dwarf: definition of unknown kind %d", kind)
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_TYPEDECL, name, 0)
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

func (d *dwctxt2) nameFromDIESym(dwtypeDIESym loader.Sym) string {
	sn := d.ldr.SymName(dwtypeDIESym)
	return sn[len(dwarf.InfoPrefix):]
}

func (d *dwctxt2) defptrto(dwtype loader.Sym) loader.Sym {

	// FIXME: it would be nice if the compiler attached an aux symbol
	// ref from the element type to the pointer type -- it would be
	// more efficient to do it this way as opposed to via name lookups.

	ptrname := "*" + d.nameFromDIESym(dwtype)
	if die := d.find(ptrname); die != 0 {
		return die
	}

	pdie := d.newdie(&dwtypes, dwarf.DW_ABRV_PTRTYPE, ptrname, 0)
	d.newrefattr(pdie, dwarf.DW_AT_type, dwtype)

	// The DWARF info synthesizes pointer types that don't exist at the
	// language level, like *hash<...> and *bucket<...>, and the data
	// pointers of slices. Link to the ones we can find.
	gts := d.ldr.Lookup("type."+ptrname, 0)
	if gts != 0 && d.ldr.AttrReachable(gts) {
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
func (d *dwctxt2) copychildrenexcept(ctxt *Link, dst *dwarf.DWDie, src *dwarf.DWDie, except *dwarf.DWDie) {
	for src = src.Child; src != nil; src = src.Link {
		if src == except {
			continue
		}
		c := d.newdie(dst, src.Abbrev, getattr(src, dwarf.DW_AT_name).Data.(string), 0)
		for a := src.Attr; a != nil; a = a.Link {
			newattr(c, a.Atr, int(a.Cls), a.Value, a.Data)
		}
		d.copychildrenexcept(ctxt, c, src, nil)
	}

	reverselist(&dst.Child)
}

func (d *dwctxt2) copychildren(ctxt *Link, dst *dwarf.DWDie, src *dwarf.DWDie) {
	d.copychildrenexcept(ctxt, dst, src, nil)
}

// Search children (assumed to have TAG_member) for the one named
// field and set its AT_type to dwtype
func (d *dwctxt2) substitutetype(structdie *dwarf.DWDie, field string, dwtype loader.Sym) {
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

func (d *dwctxt2) findprotodie(ctxt *Link, name string) *dwarf.DWDie {
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

func (d *dwctxt2) synthesizestringtypes(ctxt *Link, die *dwarf.DWDie) {
	prototype := walktypedef(d.findprotodie(ctxt, "type.runtime.stringStructDWARF"))
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

func (d *dwctxt2) synthesizeslicetypes(ctxt *Link, die *dwarf.DWDie) {
	prototype := walktypedef(d.findprotodie(ctxt, "type.runtime.slice"))
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

// synthesizemaptypes is way too closely married to runtime/hashmap.c
const (
	MaxKeySize = 128
	MaxValSize = 128
	BucketSize = 8
)

func (d *dwctxt2) mkinternaltype(ctxt *Link, abbrev int, typename, keyname, valname string, f func(*dwarf.DWDie)) loader.Sym {
	name := mkinternaltypename(typename, keyname, valname)
	symname := dwarf.InfoPrefix + name
	s := d.ldr.Lookup(symname, 0)
	if s != 0 && d.ldr.SymType(s) == sym.SDWARFINFO {
		return s
	}
	die := d.newdie(&dwtypes, abbrev, name, 0)
	f(die)
	return d.dtolsym(die.Sym)
}

func (d *dwctxt2) synthesizemaptypes(ctxt *Link, die *dwarf.DWDie) {
	hash := walktypedef(d.findprotodie(ctxt, "type.runtime.hmap"))
	bucket := walktypedef(d.findprotodie(ctxt, "type.runtime.bmap"))

	if hash == nil {
		return
	}

	for ; die != nil; die = die.Link {
		if die.Abbrev != dwarf.DW_ABRV_MAPTYPE {
			continue
		}
		gotype := loader.Sym(getattr(die, dwarf.DW_AT_type).Data.(dwSym))
		keytype := decodetypeMapKey(d.ldr, d.arch, gotype)
		valtype := decodetypeMapValue(d.ldr, d.arch, gotype)
		keydata := d.ldr.Data(keytype)
		valdata := d.ldr.Data(valtype)
		keysize, valsize := decodetypeSize(d.arch, keydata), decodetypeSize(d.arch, valdata)
		keytype, valtype = d.walksymtypedef(d.defgotype(keytype)), d.walksymtypedef(d.defgotype(valtype))

		// compute size info like hashmap.c does.
		indirectKey, indirectVal := false, false
		if keysize > MaxKeySize {
			keysize = int64(d.arch.PtrSize)
			indirectKey = true
		}
		if valsize > MaxValSize {
			valsize = int64(d.arch.PtrSize)
			indirectVal = true
		}

		// Construct type to represent an array of BucketSize keys
		keyname := d.nameFromDIESym(keytype)
		dwhks := d.mkinternaltype(ctxt, dwarf.DW_ABRV_ARRAYTYPE, "[]key", keyname, "", func(dwhk *dwarf.DWDie) {
			newattr(dwhk, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, BucketSize*keysize, 0)
			t := keytype
			if indirectKey {
				t = d.defptrto(keytype)
			}
			d.newrefattr(dwhk, dwarf.DW_AT_type, t)
			fld := d.newdie(dwhk, dwarf.DW_ABRV_ARRAYRANGE, "size", 0)
			newattr(fld, dwarf.DW_AT_count, dwarf.DW_CLS_CONSTANT, BucketSize, 0)
			d.newrefattr(fld, dwarf.DW_AT_type, d.uintptrInfoSym)
		})

		// Construct type to represent an array of BucketSize values
		valname := d.nameFromDIESym(valtype)
		dwhvs := d.mkinternaltype(ctxt, dwarf.DW_ABRV_ARRAYTYPE, "[]val", valname, "", func(dwhv *dwarf.DWDie) {
			newattr(dwhv, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, BucketSize*valsize, 0)
			t := valtype
			if indirectVal {
				t = d.defptrto(valtype)
			}
			d.newrefattr(dwhv, dwarf.DW_AT_type, t)
			fld := d.newdie(dwhv, dwarf.DW_ABRV_ARRAYRANGE, "size", 0)
			newattr(fld, dwarf.DW_AT_count, dwarf.DW_CLS_CONSTANT, BucketSize, 0)
			d.newrefattr(fld, dwarf.DW_AT_type, d.uintptrInfoSym)
		})

		// Construct bucket<K,V>
		dwhbs := d.mkinternaltype(ctxt, dwarf.DW_ABRV_STRUCTTYPE, "bucket", keyname, valname, func(dwhb *dwarf.DWDie) {
			// Copy over all fields except the field "data" from the generic
			// bucket. "data" will be replaced with keys/values below.
			d.copychildrenexcept(ctxt, dwhb, bucket, findchild(bucket, "data"))

			fld := d.newdie(dwhb, dwarf.DW_ABRV_STRUCTFIELD, "keys", 0)
			d.newrefattr(fld, dwarf.DW_AT_type, dwhks)
			newmemberoffsetattr(fld, BucketSize)
			fld = d.newdie(dwhb, dwarf.DW_ABRV_STRUCTFIELD, "values", 0)
			d.newrefattr(fld, dwarf.DW_AT_type, dwhvs)
			newmemberoffsetattr(fld, BucketSize+BucketSize*int32(keysize))
			fld = d.newdie(dwhb, dwarf.DW_ABRV_STRUCTFIELD, "overflow", 0)
			d.newrefattr(fld, dwarf.DW_AT_type, d.defptrto(d.dtolsym(dwhb.Sym)))
			newmemberoffsetattr(fld, BucketSize+BucketSize*(int32(keysize)+int32(valsize)))
			if d.arch.RegSize > d.arch.PtrSize {
				fld = d.newdie(dwhb, dwarf.DW_ABRV_STRUCTFIELD, "pad", 0)
				d.newrefattr(fld, dwarf.DW_AT_type, d.uintptrInfoSym)
				newmemberoffsetattr(fld, BucketSize+BucketSize*(int32(keysize)+int32(valsize))+int32(d.arch.PtrSize))
			}

			newattr(dwhb, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, BucketSize+BucketSize*keysize+BucketSize*valsize+int64(d.arch.RegSize), 0)
		})

		// Construct hash<K,V>
		dwhs := d.mkinternaltype(ctxt, dwarf.DW_ABRV_STRUCTTYPE, "hash", keyname, valname, func(dwh *dwarf.DWDie) {
			d.copychildren(ctxt, dwh, hash)
			d.substitutetype(dwh, "buckets", d.defptrto(dwhbs))
			d.substitutetype(dwh, "oldbuckets", d.defptrto(dwhbs))
			newattr(dwh, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, getattr(hash, dwarf.DW_AT_byte_size).Value, nil)
		})

		// make map type a pointer to hash<K,V>
		d.newrefattr(die, dwarf.DW_AT_type, d.defptrto(dwhs))
	}
}

func (d *dwctxt2) synthesizechantypes(ctxt *Link, die *dwarf.DWDie) {
	sudog := walktypedef(d.findprotodie(ctxt, "type.runtime.sudog"))
	waitq := walktypedef(d.findprotodie(ctxt, "type.runtime.waitq"))
	hchan := walktypedef(d.findprotodie(ctxt, "type.runtime.hchan"))
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

func (d *dwctxt2) dwarfDefineGlobal(ctxt *Link, symIdx loader.Sym, str string, v int64, gotype loader.Sym) {
	// Find a suitable CU DIE to include the global.
	// One would think it's as simple as just looking at the unit, but that might
	// not have any reachable code. So, we go to the runtime's CU if our unit
	// isn't otherwise reachable.
	unit := d.ldr.SymUnit(symIdx)
	if unit == nil {
		unit = ctxt.runtimeCU
	}
	ver := d.ldr.SymVersion(symIdx)
	dv := d.newdie(unit.DWInfo, dwarf.DW_ABRV_VARIABLE, str, int(ver))
	d.newabslocexprattr(dv, v, symIdx)
	if d.ldr.SymVersion(symIdx) < sym.SymVerStatic {
		newattr(dv, dwarf.DW_AT_external, dwarf.DW_CLS_FLAG, 1, 0)
	}
	dt := d.defgotype(gotype)
	d.newrefattr(dv, dwarf.DW_AT_type, dt)
}

// createUnitLength creates the initial length field with value v and update
// offset of unit_length if needed.
func (d *dwctxt2) createUnitLength(su *loader.SymbolBuilder, v uint64) {
	if isDwarf64(d.linkctxt) {
		su.AddUint32(d.arch, 0xFFFFFFFF)
	}
	d.addDwarfAddrField(su, v)
}

// addDwarfAddrField adds a DWARF field in DWARF 64bits or 32bits.
func (d *dwctxt2) addDwarfAddrField(sb *loader.SymbolBuilder, v uint64) {
	if isDwarf64(d.linkctxt) {
		sb.AddUint(d.arch, v)
	} else {
		sb.AddUint32(d.arch, uint32(v))
	}
}

// addDwarfAddrRef adds a DWARF pointer in DWARF 64bits or 32bits.
func (d *dwctxt2) addDwarfAddrRef(sb *loader.SymbolBuilder, t loader.Sym) {
	if isDwarf64(d.linkctxt) {
		d.adddwarfref(sb, t, 8)
	} else {
		d.adddwarfref(sb, t, 4)
	}
}

// calcCompUnitRanges calculates the PC ranges of the compilation units.
func (d *dwctxt2) calcCompUnitRanges() {
	var prevUnit *sym.CompilationUnit
	for _, s := range d.linkctxt.Textp2 {
		sym := loader.Sym(s)

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
		u0val := d.ldr.SymValue(loader.Sym(unit.Textp2[0]))
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

func (d *dwctxt2) importInfoSymbol(ctxt *Link, dsym loader.Sym) {
	d.ldr.SetAttrReachable(dsym, true)
	d.ldr.SetAttrNotInSymbolTable(dsym, true)
	if d.ldr.SymType(dsym) != sym.SDWARFINFO {
		log.Fatalf("error: DWARF info sym %d/%s with incorrect type %s", dsym, d.ldr.SymName(dsym), d.ldr.SymType(dsym).String())
	}
	relocs := d.ldr.Relocs(dsym)
	for i := 0; i < relocs.Count(); i++ {
		r := relocs.At2(i)
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
		ts := d.ldr.Lookup("type."+tn, 0)
		d.defgotype(ts)
	}
}

func expandFile(fname string) string {
	if strings.HasPrefix(fname, src.FileSymPrefix) {
		fname = fname[len(src.FileSymPrefix):]
	}
	return expandGoroot(fname)
}

func expandFileSym(l *loader.Loader, fsym loader.Sym) string {
	return expandFile(l.SymName(fsym))
}

func (d *dwctxt2) writelines(unit *sym.CompilationUnit, ls loader.Sym) {

	is_stmt := uint8(1) // initially = recommended default_is_stmt = 1, tracks is_stmt toggles.

	unitstart := int64(-1)
	headerstart := int64(-1)
	headerend := int64(-1)

	lsu := d.ldr.MakeSymbolUpdater(ls)
	newattr(unit.DWInfo, dwarf.DW_AT_stmt_list, dwarf.DW_CLS_PTR, lsu.Size(), dwSym(ls))

	internalExec := d.linkctxt.BuildMode == BuildModeExe && d.linkctxt.IsInternal()
	addAddrPlus := loader.GenAddAddrPlusFunc(internalExec)

	// Write .debug_line Line Number Program Header (sec 6.2.4)
	// Fields marked with (*) must be changed for 64-bit dwarf
	unitLengthOffset := lsu.Size()
	d.createUnitLength(lsu, 0) // unit_length (*), filled in at end

	unitstart = lsu.Size()
	lsu.AddUint16(d.arch, 2) // dwarf version (appendix F) -- version 3 is incompatible w/ XCode 9.0's dsymutil, latest supported on OSX 10.12 as of 2018-05
	headerLengthOffset := lsu.Size()
	d.addDwarfAddrField(lsu, 0) // header_length (*), filled in at end
	headerstart = lsu.Size()

	// cpos == unitstart + 4 + 2 + 4
	lsu.AddUint8(1)                // minimum_instruction_length
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
	lsu.AddUint8(0)                // include_directories  (empty)

	// Copy over the file table.
	fileNums := make(map[string]int)
	lsDwsym := dwSym(ls)
	for i, name := range unit.DWARFFileTable {
		name := expandFile(name)
		if len(name) == 0 {
			// Can't have empty filenames, and having a unique
			// filename is quite useful for debugging.
			name = fmt.Sprintf("<missing>_%d", i)
		}
		fileNums[name] = i + 1
		d.AddString(lsDwsym, name)
		lsu.AddUint8(0)
		lsu.AddUint8(0)
		lsu.AddUint8(0)
		if gdbscript == "" {
			// We can't use something that may be dead-code
			// eliminated from a binary here. proc.go contains
			// main and the scheduler, so it's not going anywhere.
			if i := strings.Index(name, "runtime/proc.go"); i >= 0 {
				k := strings.Index(name, "runtime/proc.go")
				gdbscript = name[:k] + "runtime/runtime-gdb.py"
			}
		}
	}

	// 4 zeros: the string termination + 3 fields.
	lsu.AddUint8(0)
	// terminate file_names.
	headerend = lsu.Size()

	// Output the state machine for each function remaining.
	var lastAddr int64
	for _, s := range unit.Textp2 {
		fnSym := loader.Sym(s)

		// Set the PC.
		lsu.AddUint8(0)
		dwarf.Uleb128put(d, lsDwsym, 1+int64(d.arch.PtrSize))
		lsu.AddUint8(dwarf.DW_LNE_set_address)
		addr := addAddrPlus(lsu, d.arch, fnSym, 0)
		// Make sure the units are sorted.
		if addr < lastAddr {
			d.linkctxt.Errorf(fnSym, "address wasn't increasing %x < %x",
				addr, lastAddr)
		}
		lastAddr = addr

		// Output the line table.
		// TODO: Now that we have all the debug information in separate
		// symbols, it would make sense to use a rope, and concatenate them all
		// together rather then the append() below. This would allow us to have
		// the compiler emit the DW_LNE_set_address and a rope data structure
		// to concat them all together in the output.
		_, _, _, lines := d.ldr.GetFuncDwarfAuxSyms(fnSym)
		if lines != 0 {
			lsu.AddBytes(d.ldr.Data(lines))
		}
	}

	// Issue 38192: the DWARF standard specifies that when you issue
	// an end-sequence op, the PC value should be one past the last
	// text address in the translation unit, so apply a delta to the
	// text address before the end sequence op. If this isn't done,
	// GDB will assign a line number of zero the last row in the line
	// table, which we don't want. The 1 + ptrsize amount is somewhat
	// arbitrary, this is chosen to be consistent with the way LLVM
	// emits its end sequence ops.
	lsu.AddUint8(dwarf.DW_LNS_advance_pc)
	dwarf.Uleb128put(d, lsDwsym, int64(1+d.arch.PtrSize))

	// Emit an end-sequence at the end of the unit.
	lsu.AddUint8(0) // start extended opcode
	dwarf.Uleb128put(d, lsDwsym, 1)
	lsu.AddUint8(dwarf.DW_LNE_end_sequence)

	if d.linkctxt.HeadType == objabi.Haix {
		saveDwsectCUSize(".debug_line", unit.Lib.Pkg, uint64(lsu.Size()-unitLengthOffset))
	}
	if isDwarf64(d.linkctxt) {
		lsu.SetUint(d.arch, unitLengthOffset+4, uint64(lsu.Size()-unitstart)) // +4 because of 0xFFFFFFFF
		lsu.SetUint(d.arch, headerLengthOffset, uint64(headerend-headerstart))
	} else {
		lsu.SetUint32(d.arch, unitLengthOffset, uint32(lsu.Size()-unitstart))
		lsu.SetUint32(d.arch, headerLengthOffset, uint32(headerend-headerstart))
	}
}

// writepcranges generates the DW_AT_ranges table for compilation unit cu.
func (d *dwctxt2) writepcranges(unit *sym.CompilationUnit, base loader.Sym, pcs []dwarf.Range, ranges loader.Sym) {

	rsu := d.ldr.MakeSymbolUpdater(ranges)
	rDwSym := dwSym(ranges)

	unitLengthOffset := rsu.Size()

	// Create PC ranges for this CU.
	newattr(unit.DWInfo, dwarf.DW_AT_ranges, dwarf.DW_CLS_PTR, rsu.Size(), rDwSym)
	newattr(unit.DWInfo, dwarf.DW_AT_low_pc, dwarf.DW_CLS_ADDRESS, 0, dwSym(base))
	dwarf.PutBasedRanges(d, rDwSym, pcs)

	if d.linkctxt.HeadType == objabi.Haix {
		addDwsectCUSize(".debug_ranges", unit.Lib.Pkg, uint64(rsu.Size()-unitLengthOffset))
	}
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

func (d *dwctxt2) writeframes() dwarfSecInfo {
	fs := d.ldr.LookupOrCreateSym(".debug_frame", 0)
	fsd := dwSym(fs)
	fsu := d.ldr.MakeSymbolUpdater(fs)
	fsu.SetType(sym.SDWARFSECT)
	isdw64 := isDwarf64(d.linkctxt)
	haslr := haslinkregister(d.linkctxt)

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
	for _, s := range d.linkctxt.Textp2 {
		fn := loader.Sym(s)
		fi := d.ldr.FuncInfo(fn)
		if !fi.Valid() {
			continue
		}
		fpcsp := fi.Pcsp()

		// Emit a FDE, Section 6.4.1.
		// First build the section contents into a byte buffer.
		deltaBuf = deltaBuf[:0]
		if haslr && d.ldr.AttrTopFrame(fn) {
			// Mark the link register as having an undefined value.
			// This stops call stack unwinders progressing any further.
			// TODO: similar mark on non-LR architectures.
			deltaBuf = append(deltaBuf, dwarf.DW_CFA_undefined)
			deltaBuf = dwarf.AppendUleb128(deltaBuf, uint64(thearch.Dwarfreglr))
		}

		for pcsp.Init(fpcsp); !pcsp.Done; pcsp.Next() {
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

			if haslr && !d.ldr.AttrTopFrame(fn) {
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

const (
	COMPUNITHEADERSIZE = 4 + 2 + 4 + 1
)

// appendSyms appends the syms from 'src' into 'syms' and returns the
// result. This can go away once we do away with sym.LoaderSym
// entirely.
func appendSyms(syms []loader.Sym, src []sym.LoaderSym) []loader.Sym {
	for _, s := range src {
		syms = append(syms, loader.Sym(s))
	}
	return syms
}

func (d *dwctxt2) writeinfo(units []*sym.CompilationUnit, abbrevsym loader.Sym, pubNames, pubTypes *pubWriter2) dwarfSecInfo {

	infosec := d.ldr.LookupOrCreateSym(".debug_info", 0)
	disu := d.ldr.MakeSymbolUpdater(infosec)
	disu.SetType(sym.SDWARFINFO)
	d.ldr.SetAttrReachable(infosec, true)
	syms := []loader.Sym{infosec}

	for _, u := range units {
		compunit := u.DWInfo
		s := d.dtolsym(compunit.Sym)
		su := d.ldr.MakeSymbolUpdater(s)

		if len(u.Textp2) == 0 && u.DWInfo.Child == nil {
			continue
		}

		pubNames.beginCompUnit(compunit)
		pubTypes.beginCompUnit(compunit)

		// Write .debug_info Compilation Unit Header (sec 7.5.1)
		// Fields marked with (*) must be changed for 64-bit dwarf
		// This must match COMPUNITHEADERSIZE above.
		d.createUnitLength(su, 0) // unit_length (*), will be filled in later.
		su.AddUint16(d.arch, 4)   // dwarf version (appendix F)

		// debug_abbrev_offset (*)
		d.addDwarfAddrRef(su, abbrevsym)

		su.AddUint8(uint8(d.arch.PtrSize)) // address_size

		ds := dwSym(s)
		dwarf.Uleb128put(d, ds, int64(compunit.Abbrev))
		dwarf.PutAttrs(d, ds, compunit.Abbrev, compunit.Attr)

		cu := []loader.Sym{s}
		cu = appendSyms(cu, u.AbsFnDIEs2)
		cu = appendSyms(cu, u.FuncDIEs2)
		if u.Consts2 != 0 {
			cu = append(cu, loader.Sym(u.Consts2))
		}
		var cusize int64
		for _, child := range cu {
			cusize += int64(len(d.ldr.Data(child)))
		}

		for die := compunit.Child; die != nil; die = die.Link {
			l := len(cu)
			lastSymSz := int64(len(d.ldr.Data(cu[l-1])))
			cu = d.putdie(cu, die)
			if ispubname(die) {
				pubNames.add(die, cusize)
			}
			if ispubtype(die) {
				pubTypes.add(die, cusize)
			}
			if lastSymSz != int64(len(d.ldr.Data(cu[l-1]))) {
				// putdie will sometimes append directly to the last symbol of the list
				cusize = cusize - lastSymSz + int64(len(d.ldr.Data(cu[l-1])))
			}
			for _, child := range cu[l:] {
				cusize += int64(len(d.ldr.Data(child)))
			}
		}

		culu := d.ldr.MakeSymbolUpdater(cu[len(cu)-1])
		culu.AddUint8(0) // closes compilation unit DIE
		cusize++

		// Save size for AIX symbol table.
		if d.linkctxt.HeadType == objabi.Haix {
			saveDwsectCUSize(".debug_info", d.getPkgFromCUSym(s), uint64(cusize))
		}
		if isDwarf64(d.linkctxt) {
			cusize -= 12                          // exclude the length field.
			su.SetUint(d.arch, 4, uint64(cusize)) // 4 because of 0XFFFFFFFF
		} else {
			cusize -= 4 // exclude the length field.
			su.SetUint32(d.arch, 0, uint32(cusize))
		}
		pubNames.endCompUnit(compunit, uint32(cusize)+4)
		pubTypes.endCompUnit(compunit, uint32(cusize)+4)
		syms = append(syms, cu...)
	}

	return dwarfSecInfo{syms: syms}
}

/*
 *  Emit .debug_pubnames/_types.  _info must have been written before,
 *  because we need die->offs and infoo/infosize;
 */

type pubWriter2 struct {
	d     *dwctxt2
	s     loader.Sym
	su    *loader.SymbolBuilder
	sname string

	sectionstart int64
	culengthOff  int64
}

func newPubWriter2(d *dwctxt2, sname string) *pubWriter2 {
	s := d.ldr.LookupOrCreateSym(sname, 0)
	u := d.ldr.MakeSymbolUpdater(s)
	u.SetType(sym.SDWARFSECT)
	return &pubWriter2{d: d, s: s, su: u, sname: sname}
}

func (pw *pubWriter2) beginCompUnit(compunit *dwarf.DWDie) {
	pw.sectionstart = pw.su.Size()

	// Write .debug_pubnames/types	Header (sec 6.1.1)
	pw.d.createUnitLength(pw.su, 0)                         // unit_length (*), will be filled in later.
	pw.su.AddUint16(pw.d.arch, 2)                           // dwarf version (appendix F)
	pw.d.addDwarfAddrRef(pw.su, pw.d.dtolsym(compunit.Sym)) // debug_info_offset (of the Comp unit Header)
	pw.culengthOff = pw.su.Size()
	pw.d.addDwarfAddrField(pw.su, uint64(0)) // debug_info_length, will be filled in later.
}

func (pw *pubWriter2) add(die *dwarf.DWDie, offset int64) {
	dwa := getattr(die, dwarf.DW_AT_name)
	name := dwa.Data.(string)
	if pw.d.dtolsym(die.Sym) == 0 {
		fmt.Println("Missing sym for ", name)
	}
	pw.d.addDwarfAddrField(pw.su, uint64(offset))
	pw.su.Addstring(name)
}

func (pw *pubWriter2) endCompUnit(compunit *dwarf.DWDie, culength uint32) {
	pw.d.addDwarfAddrField(pw.su, 0) // Null offset

	// On AIX, save the current size of this compilation unit.
	if pw.d.linkctxt.HeadType == objabi.Haix {
		saveDwsectCUSize(pw.sname, pw.d.getPkgFromCUSym(pw.d.dtolsym(compunit.Sym)), uint64(pw.su.Size()-pw.sectionstart))
	}
	if isDwarf64(pw.d.linkctxt) {
		pw.su.SetUint(pw.d.arch, pw.sectionstart+4, uint64(pw.su.Size()-pw.sectionstart)-12) // exclude the length field.
		pw.su.SetUint(pw.d.arch, pw.culengthOff, uint64(culength))
	} else {
		pw.su.SetUint32(pw.d.arch, pw.sectionstart, uint32(pw.su.Size()-pw.sectionstart)-4) // exclude the length field.
		pw.su.SetUint32(pw.d.arch, pw.culengthOff, culength)
	}
}

func ispubname(die *dwarf.DWDie) bool {
	switch die.Abbrev {
	case dwarf.DW_ABRV_FUNCTION, dwarf.DW_ABRV_VARIABLE:
		a := getattr(die, dwarf.DW_AT_external)
		return a != nil && a.Value != 0
	}

	return false
}

func ispubtype(die *dwarf.DWDie) bool {
	return die.Abbrev >= dwarf.DW_ABRV_NULLTYPE
}

func (d *dwctxt2) writegdbscript() dwarfSecInfo {
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

	gs := d.ldr.LookupOrCreateSym(".debug_gdb_scripts", 0)
	u := d.ldr.MakeSymbolUpdater(gs)
	u.SetType(sym.SDWARFSECT)

	u.AddUint8(1) // magic 1 byte?
	u.Addstring(gdbscript)
	return dwarfSecInfo{syms: []loader.Sym{gs}}
}

// FIXME: might be worth looking replacing this map with a function
// that switches based on symbol instead.

var prototypedies map[string]*dwarf.DWDie

func dwarfEnabled(ctxt *Link) bool {
	if *FlagW { // disable dwarf
		return false
	}
	if *FlagS && ctxt.HeadType != objabi.Hdarwin {
		return false
	}
	if ctxt.HeadType == objabi.Hplan9 || ctxt.HeadType == objabi.Hjs {
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
func (d *dwctxt2) mkBuiltinType(ctxt *Link, abrv int, tname string) *dwarf.DWDie {
	// create type DIE
	die := d.newdie(&dwtypes, abrv, tname, 0)

	// Look up type symbol.
	gotype := d.lookupOrDiag("type." + tname)

	// Map from die sym to type sym
	ds := loader.Sym(die.Sym.(dwSym))
	d.rtmap[ds] = gotype

	// Map from type to def sym
	d.tdmap[gotype] = ds

	return die
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

	d := newdwctxt2(ctxt, true)

	if ctxt.HeadType == objabi.Haix {
		// Initial map used to store package size for each DWARF section.
		dwsectCUSize = make(map[string]uint64)
	}

	// For ctxt.Diagnostic messages.
	newattr(&dwtypes, dwarf.DW_AT_name, dwarf.DW_CLS_STRING, int64(len("dwtypes")), "dwtypes")

	// Unspecified type. There are no references to this in the symbol table.
	d.newdie(&dwtypes, dwarf.DW_ABRV_NULLTYPE, "<unspecified>", 0)

	// Some types that must exist to define other ones (uintptr in particular
	// is needed for array size)
	d.mkBuiltinType(ctxt, dwarf.DW_ABRV_BARE_PTRTYPE, "unsafe.Pointer")
	die := d.mkBuiltinType(ctxt, dwarf.DW_ABRV_BASETYPE, "uintptr")
	newattr(die, dwarf.DW_AT_encoding, dwarf.DW_CLS_CONSTANT, dwarf.DW_ATE_unsigned, 0)
	newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, int64(d.arch.PtrSize), 0)
	newattr(die, dwarf.DW_AT_go_kind, dwarf.DW_CLS_CONSTANT, objabi.KindUintptr, 0)
	newattr(die, dwarf.DW_AT_go_runtime_type, dwarf.DW_CLS_ADDRESS, 0, dwSym(d.lookupOrDiag("type.uintptr")))

	d.uintptrInfoSym = d.mustFind("uintptr")

	// Prototypes needed for type synthesis.
	prototypedies = map[string]*dwarf.DWDie{
		"type.runtime.stringStructDWARF": nil,
		"type.runtime.slice":             nil,
		"type.runtime.hmap":              nil,
		"type.runtime.bmap":              nil,
		"type.runtime.sudog":             nil,
		"type.runtime.waitq":             nil,
		"type.runtime.hchan":             nil,
	}

	// Needed by the prettyprinter code for interface inspection.
	for _, typ := range []string{
		"type.runtime._type",
		"type.runtime.arraytype",
		"type.runtime.chantype",
		"type.runtime.functype",
		"type.runtime.maptype",
		"type.runtime.ptrtype",
		"type.runtime.slicetype",
		"type.runtime.structtype",
		"type.runtime.interfacetype",
		"type.runtime.itab",
		"type.runtime.imethod"} {
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
				unit.Consts2 = sym.LoaderSym(consts)
				d.importInfoSymbol(ctxt, consts)
				consts = 0
			}
			ctxt.compUnits = append(ctxt.compUnits, unit)

			// We need at least one runtime unit.
			if unit.Lib.Pkg == "runtime" {
				ctxt.runtimeCU = unit
			}

			unit.DWInfo = d.newdie(&dwroot, dwarf.DW_ABRV_COMPUNIT, unit.Lib.Pkg, 0)
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
			producer := "Go cmd/compile " + objabi.Version
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

			if len(unit.Textp2) == 0 {
				unit.DWInfo.Abbrev = dwarf.DW_ABRV_COMPUNIT_TEXTLESS
			}

			// Scan all functions in this compilation unit, create DIEs for all
			// referenced types, create the file table for debug_line, find all
			// referenced abstract functions.
			// Collect all debug_range symbols in unit.rangeSyms
			for _, s := range unit.Textp2 { // textp2 has been dead-code-eliminated already.
				fnSym := loader.Sym(s)
				infosym, _, rangesym, _ := d.ldr.GetFuncDwarfAuxSyms(fnSym)
				if infosym == 0 {
					continue
				}
				d.ldr.SetAttrNotInSymbolTable(infosym, true)
				d.ldr.SetAttrReachable(infosym, true)

				unit.FuncDIEs2 = append(unit.FuncDIEs2, sym.LoaderSym(infosym))
				if rangesym != 0 {
					rs := len(d.ldr.Data(rangesym))
					d.ldr.SetAttrNotInSymbolTable(rangesym, true)
					d.ldr.SetAttrReachable(rangesym, true)
					if ctxt.HeadType == objabi.Haix {
						addDwsectCUSize(".debug_ranges", unit.Lib.Pkg, uint64(rs))
					}
					unit.RangeSyms2 = append(unit.RangeSyms2, sym.LoaderSym(rangesym))
				}

				drelocs := d.ldr.Relocs(infosym)
				for ri := 0; ri < drelocs.Count(); ri++ {
					r := drelocs.At2(ri)
					if r.Type() == objabi.R_DWARFSECREF {
						rsym := r.Sym()
						rsn := d.ldr.SymName(rsym)
						if len(rsn) == 0 {
							continue
						}
						// NB: there should be a better way to do this that doesn't involve materializing the symbol name and doing string prefix+suffix checks.
						if strings.HasPrefix(rsn, dwarf.InfoPrefix) && strings.HasSuffix(rsn, dwarf.AbstractFuncSuffix) && !d.ldr.AttrOnList(rsym) {
							// abstract function
							d.ldr.SetAttrOnList(rsym, true)
							unit.AbsFnDIEs2 = append(unit.AbsFnDIEs2, sym.LoaderSym(rsym))
							d.importInfoSymbol(ctxt, rsym)
							continue
						}
						if _, ok := d.rtmap[rsym]; ok {
							// type already generated
							continue
						}
						tn := rsn[len(dwarf.InfoPrefix):]
						ts := d.ldr.Lookup("type."+tn, 0)
						d.defgotype(ts)
					}
				}
			}
		}
	}

	// Fix for 31034: if the objects feeding into this link were compiled
	// with different sets of flags, then don't issue an error if
	// the -strictdups checks fail.
	if checkStrictDups > 1 && len(flagVariants) > 1 {
		checkStrictDups = 1
	}

	// Create DIEs for global variables and the types they use.
	// FIXME: ideally this should be done in the compiler, since
	// for globals there isn't any abiguity about which package
	// a global belongs to.
	for idx := loader.Sym(1); idx < loader.Sym(d.ldr.NDef()); idx++ {
		if !d.ldr.AttrReachable(idx) ||
			d.ldr.AttrNotInSymbolTable(idx) ||
			d.ldr.SymVersion(idx) >= sym.SymVerStatic {
			continue
		}
		t := d.ldr.SymType(idx)
		switch t {
		case sym.SRODATA, sym.SDATA, sym.SNOPTRDATA, sym.STYPE, sym.SBSS, sym.SNOPTRBSS, sym.STLSBSS:
			// ok
		default:
			continue
		}
		// Skip things with no type
		if d.ldr.SymGoType(idx) == 0 {
			continue
		}

		sn := d.ldr.SymName(idx)
		if ctxt.LinkMode != LinkExternal && isStaticTemp(sn) {
			continue
		}
		if sn == "" {
			// skip aux symbols
			continue
		}

		// Create DIE for global.
		sv := d.ldr.SymValue(idx)
		gt := d.ldr.SymGoType(idx)
		d.dwarfDefineGlobal(ctxt, idx, sn, sv, gt)
	}

	// Create DIEs for variable types indirectly referenced by function
	// autos (which may not appear directly as param/var DIEs).
	for _, lib := range ctxt.Library {
		for _, unit := range lib.Units {
			lists := [][]sym.LoaderSym{unit.AbsFnDIEs2, unit.FuncDIEs2}
			for _, list := range lists {
				for _, s := range list {
					symIdx := loader.Sym(s)
					relocs := d.ldr.Relocs(symIdx)
					for i := 0; i < relocs.Count(); i++ {
						r := relocs.At2(i)
						if r.Type() == objabi.R_USETYPE {
							d.defgotype(r.Sym())
						}
					}
				}
			}
		}
	}

	d.synthesizestringtypes(ctxt, dwtypes.Child)
	d.synthesizeslicetypes(ctxt, dwtypes.Child)
	d.synthesizemaptypes(ctxt, dwtypes.Child)
	d.synthesizechantypes(ctxt, dwtypes.Child)

	// NB: at this stage we have all the DIE objects constructed, but
	// they have loader.Sym attributes and not sym.Symbol attributes.
	// At the point when loadlibfull runs we will need to visit
	// every DIE constructed and convert the symbols.
}

// dwarfGenerateDebugSyms constructs debug_line, debug_frame, debug_loc,
// debug_pubnames and debug_pubtypes. It also writes out the debug_info
// section using symbols generated in dwarfGenerateDebugInfo2.
func dwarfGenerateDebugSyms(ctxt *Link) {
	if !dwarfEnabled(ctxt) {
		return
	}
	d := &dwctxt2{
		linkctxt: ctxt,
		ldr:      ctxt.loader,
		arch:     ctxt.Arch,
	}
	d.dwarfGenerateDebugSyms()
}

func (d *dwctxt2) dwarfGenerateDebugSyms() {
	abbrevSec := d.writeabbrev()
	dwarfp2 = append(dwarfp2, abbrevSec)

	d.calcCompUnitRanges()
	sort.Sort(compilationUnitByStartPC(d.linkctxt.compUnits))

	// Create .debug_line and .debug_ranges section symbols
	debugLine := d.ldr.LookupOrCreateSym(".debug_line", 0)
	dlu := d.ldr.MakeSymbolUpdater(debugLine)
	dlu.SetType(sym.SDWARFSECT)
	d.ldr.SetAttrReachable(debugLine, true)
	dwarfp2 = append(dwarfp2, dwarfSecInfo{syms: []loader.Sym{debugLine}})

	debugRanges := d.ldr.LookupOrCreateSym(".debug_ranges", 0)
	dru := d.ldr.MakeSymbolUpdater(debugRanges)
	dru.SetType(sym.SDWARFRANGE)
	d.ldr.SetAttrReachable(debugRanges, true)

	// Write per-package line and range tables and start their CU DIEs.
	for _, u := range d.linkctxt.compUnits {
		reversetree(&u.DWInfo.Child)
		if u.DWInfo.Abbrev == dwarf.DW_ABRV_COMPUNIT_TEXTLESS {
			continue
		}
		d.writelines(u, debugLine)
		base := loader.Sym(u.Textp2[0])
		d.writepcranges(u, base, u.PCs, debugRanges)
	}

	// newdie adds DIEs to the *beginning* of the parent's DIE list.
	// Now that we're done creating DIEs, reverse the trees so DIEs
	// appear in the order they were created.
	reversetree(&dwtypes.Child)
	movetomodule(d.linkctxt, &dwtypes)

	pubNames := newPubWriter2(d, ".debug_pubnames")
	pubTypes := newPubWriter2(d, ".debug_pubtypes")

	infoSec := d.writeinfo(d.linkctxt.compUnits, abbrevSec.secSym(), pubNames, pubTypes)

	framesSec := d.writeframes()
	dwarfp2 = append(dwarfp2, framesSec)
	dwarfp2 = append(dwarfp2, dwarfSecInfo{syms: []loader.Sym{pubNames.s}})
	dwarfp2 = append(dwarfp2, dwarfSecInfo{syms: []loader.Sym{pubTypes.s}})
	gdbScriptSec := d.writegdbscript()
	if gdbScriptSec.secSym() != 0 {
		dwarfp2 = append(dwarfp2, gdbScriptSec)
	}
	dwarfp2 = append(dwarfp2, infoSec)
	locSec := d.collectlocs(d.linkctxt.compUnits)
	if locSec.secSym() != 0 {
		dwarfp2 = append(dwarfp2, locSec)
	}

	rsyms := []loader.Sym{debugRanges}
	for _, unit := range d.linkctxt.compUnits {
		for _, s := range unit.RangeSyms2 {
			rsyms = append(rsyms, loader.Sym(s))
		}
	}
	dwarfp2 = append(dwarfp2, dwarfSecInfo{syms: rsyms})
}

func (d *dwctxt2) collectlocs(units []*sym.CompilationUnit) dwarfSecInfo {
	empty := true
	syms := []loader.Sym{}
	for _, u := range units {
		for _, fn := range u.FuncDIEs2 {
			relocs := d.ldr.Relocs(loader.Sym(fn))
			for i := 0; i < relocs.Count(); i++ {
				reloc := relocs.At2(i)
				if reloc.Type() != objabi.R_DWARFSECREF {
					continue
				}
				rsym := reloc.Sym()
				if d.ldr.SymType(rsym) == sym.SDWARFLOC {
					d.ldr.SetAttrReachable(rsym, true)
					d.ldr.SetAttrNotInSymbolTable(rsym, true)
					syms = append(syms, rsym)
					empty = false
					// One location list entry per function, but many relocations to it. Don't duplicate.
					break
				}
			}
		}
	}

	// Don't emit .debug_loc if it's empty -- it makes the ARM linker mad.
	if empty {
		return dwarfSecInfo{}
	}

	locsym := d.ldr.LookupOrCreateSym(".debug_loc", 0)
	u := d.ldr.MakeSymbolUpdater(locsym)
	u.SetType(sym.SDWARFLOC)
	d.ldr.SetAttrReachable(locsym, true)
	return dwarfSecInfo{syms: append([]loader.Sym{locsym}, syms...)}
}

/*
 *  Elf.
 */
func (d *dwctxt2) dwarfaddshstrings(ctxt *Link, shstrtab loader.Sym) {
	panic("not yet implemented")
}

// Add section symbols for DWARF debug info.  This is called before
// dwarfaddelfheaders.
func (d *dwctxt2) dwarfaddelfsectionsyms(ctxt *Link) {
	panic("not yet implemented")
}

// dwarfcompress compresses the DWARF sections. Relocations are applied
// on the fly. After this, dwarfp will contain a different (new) set of
// symbols, and sections may have been replaced.
func (d *dwctxt2) dwarfcompress(ctxt *Link) {
	panic("not yet implemented")
}

// getPkgFromCUSym returns the package name for the compilation unit
// represented by s.
// The prefix dwarf.InfoPrefix+".pkg." needs to be removed in order to get
// the package name.
func (d *dwctxt2) getPkgFromCUSym(s loader.Sym) string {
	return strings.TrimPrefix(d.ldr.SymName(s), dwarf.InfoPrefix+".pkg.")
}

// On AIX, the symbol table needs to know where are the compilation units parts
// for a specific package in each .dw section.
// dwsectCUSize map will save the size of a compilation unit for
// the corresponding .dw section.
// This size can later be retrieved with the index "sectionName.pkgName".
var dwsectCUSize map[string]uint64

// getDwsectCUSize retrieves the corresponding package size inside the current section.
func getDwsectCUSize(sname string, pkgname string) uint64 {
	return dwsectCUSize[sname+"."+pkgname]
}

func saveDwsectCUSize(sname string, pkgname string, size uint64) {
	dwsectCUSize[sname+"."+pkgname] = size
}

func addDwsectCUSize(sname string, pkgname string, size uint64) {
	dwsectCUSize[sname+"."+pkgname] += size
}
