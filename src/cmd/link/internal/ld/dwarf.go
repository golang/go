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
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/loader"
	"cmd/link/internal/sym"
	"fmt"
	"log"
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

func (c dwctxt2) PtrSize() int {
	return c.arch.PtrSize
}

func (c dwctxt2) AddInt(s dwarf.Sym, size int, i int64) {
	ds := s.(dwSym)
	dsu := ds.l.MakeSymbolUpdater(ds.s)
	dsu.AddUintXX(c.arch, uint64(i), size)
}

func (c dwctxt2) AddBytes(s dwarf.Sym, b []byte) {
	ds := s.(dwSym)
	dsu := ds.l.MakeSymbolUpdater(ds.s)
	dsu.AddBytes(b)
}

func (c dwctxt2) AddString(s dwarf.Sym, v string) {
	ds := s.(dwSym)
	dsu := ds.l.MakeSymbolUpdater(ds.s)
	dsu.Addstring(v)
}

func (c dwctxt2) AddAddress(s dwarf.Sym, data interface{}, value int64) {
	ds := s.(dwSym)
	dsu := ds.l.MakeSymbolUpdater(ds.s)
	if value != 0 {
		value -= dsu.Value()
	}
	tgtds := data.(dwSym)
	dsu.AddAddrPlus(c.arch, tgtds.s, value)
}

func (c dwctxt2) AddCURelativeAddress(s dwarf.Sym, data interface{}, value int64) {
	ds := s.(dwSym)
	dsu := ds.l.MakeSymbolUpdater(ds.s)
	if value != 0 {
		value -= dsu.Value()
	}
	tgtds := data.(dwSym)
	dsu.AddCURelativeAddrPlus(c.arch, tgtds.s, value)
}

func (c dwctxt2) AddSectionOffset(s dwarf.Sym, size int, t interface{}, ofs int64) {
	ds := s.(dwSym)
	dsu := ds.l.MakeSymbolUpdater(ds.s)
	tds := t.(dwSym)
	switch size {
	default:
		c.linkctxt.Errorf(ds.s, "invalid size %d in adddwarfref\n", size)
		fallthrough
	case c.arch.PtrSize:
		dsu.AddAddrPlus(c.arch, tds.s, 0)
	case 4:
		dsu.AddAddrPlus4(c.arch, tds.s, 0)
	}
	rsl := dsu.Relocs()
	r := &rsl[len(rsl)-1]
	r.Type = objabi.R_ADDROFF
	r.Add = ofs
}

func (c dwctxt2) AddDWARFAddrSectionOffset(s dwarf.Sym, t interface{}, ofs int64) {
	size := 4
	if isDwarf64(c.linkctxt) {
		size = 8
	}

	c.AddSectionOffset(s, size, t, ofs)

	ds := s.(dwSym)
	dsu := ds.l.MakeSymbolUpdater(ds.s)
	rsl := dsu.Relocs()
	r := &rsl[len(rsl)-1]
	r.Type = objabi.R_DWARFSECREF
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

// dwSym wraps a loader.Sym; objects of this type are stored
// in the 'Sym' field of dwarf.DIE objects.
//
// FIXME: the main reason we need the loader.Loader pointer field is
// that the dwarf.Sym interface has a Len() method with no parameters.
// If we changed this method to accept a dwxtxt (from which we could
// access the loader) then we could get rid of this field and/or avoid
// using a struct.
type dwSym struct {
	s loader.Sym
	l *loader.Loader
}

func (s dwSym) Len() int64 {
	return int64(len(s.l.Data(s.s)))
}

var gdbscript string

var dwarfp2 []loader.Sym

func (d *dwctxt2) writeabbrev(ctxt *Link) loader.Sym {
	panic("not yet implemented")
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
			die.Sym = dwSym{s: ds, l: d.ldr}
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
		fallthrough
	case d.arch.PtrSize:
		result = sb.AddAddrPlus(d.arch, t, 0)
	case 4:
		result = sb.AddAddrPlus4(d.arch, t, 0)
	}
	rsl := sb.Relocs()
	r := &rsl[len(rsl)-1]
	r.Type = objabi.R_DWARFSECREF
	return result
}

func (d *dwctxt2) newrefattr(die *dwarf.DWDie, attr uint16, ref loader.Sym) *dwarf.DWAttr {
	if ref == 0 {
		return nil
	}
	return newattr(die, attr, dwarf.DW_CLS_REFERENCE, 0, dwSym{s: ref, l: d.ldr})
}

func (d *dwctxt2) dtolsym(s dwarf.Sym) loader.Sym {
	if s == nil {
		return 0
	}
	dws := s.(dwSym)
	return dws.s
}

func (d *dwctxt2) putdie(syms []loader.Sym, die *dwarf.DWDie) []loader.Sym {
	panic("not yet implemented")
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
	newattr(die, dwarf.DW_AT_location, dwarf.DW_CLS_ADDRESS, addr, dwSym{s: symIdx, l: d.ldr})
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
	tds := d.ldr.CreateExtSym("")
	tdsu := d.ldr.MakeSymbolUpdater(tds)
	tdsu.SetType(sym.SDWARFINFO)
	def.Sym = dwSym{s: tds, l: d.ldr}
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
	d.tdmap[gotype] = gtdwSym.Sym.(dwSym).s
	return gtdwSym.Sym.(dwSym).s
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
		s := decodetypeArrayElem2(d.ldr, d.arch, gotype)
		d.newrefattr(die, dwarf.DW_AT_type, d.defgotype(s))
		fld := d.newdie(die, dwarf.DW_ABRV_ARRAYRANGE, "range", 0)

		// use actual length not upper bound; correct for 0-length arrays.
		newattr(fld, dwarf.DW_AT_count, dwarf.DW_CLS_CONSTANT, decodetypeArrayLen2(d.ldr, d.arch, gotype), 0)

		d.newrefattr(fld, dwarf.DW_AT_type, d.uintptrInfoSym)

	case objabi.KindChan:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_CHANTYPE, name, 0)
		s := decodetypeChanElem2(d.ldr, d.arch, gotype)
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
		rslice := relocs.ReadAll(nil)
		nfields := decodetypeFuncInCount(d.arch, data)
		for i := 0; i < nfields; i++ {
			s := decodetypeFuncInType2(d.ldr, d.arch, gotype, rslice, i)
			sn := d.ldr.SymName(s)
			fld := d.newdie(die, dwarf.DW_ABRV_FUNCTYPEPARAM, sn[5:], 0)
			d.newrefattr(fld, dwarf.DW_AT_type, d.defgotype(s))
		}

		if decodetypeFuncDotdotdot(d.arch, data) {
			d.newdie(die, dwarf.DW_ABRV_DOTDOTDOT, "...", 0)
		}
		nfields = decodetypeFuncOutCount(d.arch, data)
		for i := 0; i < nfields; i++ {
			s := decodetypeFuncOutType2(d.ldr, d.arch, gotype, rslice, i)
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
		s := decodetypeMapKey2(d.ldr, d.arch, gotype)
		d.newrefattr(die, dwarf.DW_AT_go_key, d.defgotype(s))
		s = decodetypeMapValue2(d.ldr, d.arch, gotype)
		d.newrefattr(die, dwarf.DW_AT_go_elem, d.defgotype(s))
		// Save gotype for use in synthesizemaptypes. We could synthesize here,
		// but that would change the order of the DIEs.
		d.newrefattr(die, dwarf.DW_AT_type, gotype)

	case objabi.KindPtr:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_PTRTYPE, name, 0)
		typedefdie = d.dotypedef(&dwtypes, gotype, name, die)
		s := decodetypePtrElem2(d.ldr, d.arch, gotype)
		d.newrefattr(die, dwarf.DW_AT_type, d.defgotype(s))

	case objabi.KindSlice:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_SLICETYPE, name, 0)
		typedefdie = d.dotypedef(&dwtypes, gotype, name, die)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		s := decodetypeArrayElem2(d.ldr, d.arch, gotype)
		elem := d.defgotype(s)
		d.newrefattr(die, dwarf.DW_AT_go_elem, elem)

	case objabi.KindString:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_STRINGTYPE, name, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case objabi.KindStruct:
		die = d.newdie(&dwtypes, dwarf.DW_ABRV_STRUCTTYPE, name, 0)
		typedefdie = d.dotypedef(&dwtypes, gotype, name, die)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		nfields := decodetypeStructFieldCount2(d.ldr, d.arch, gotype)
		for i := 0; i < nfields; i++ {
			f := decodetypeStructFieldName2(d.ldr, d.arch, gotype, i)
			s := decodetypeStructFieldType2(d.ldr, d.arch, gotype, i)
			if f == "" {
				sn := d.ldr.SymName(s)
				f = sn[5:] // skip "type."
			}
			fld := d.newdie(die, dwarf.DW_ABRV_STRUCTFIELD, f, 0)
			d.newrefattr(fld, dwarf.DW_AT_type, d.defgotype(s))
			offsetAnon := decodetypeStructFieldOffsAnon2(d.ldr, d.arch, gotype, i)
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
		newattr(die, dwarf.DW_AT_go_runtime_type, dwarf.DW_CLS_GO_TYPEREF, 0, dwSym{s: gotype, l: d.ldr})
	}

	// Sanity check.
	if _, ok := d.rtmap[gotype]; ok {
		log.Fatalf("internal error: rtmap entry already installed\n")
	}

	ds := die.Sym.(dwSym)
	if typedefdie != nil {
		ds = typedefdie.Sym.(dwSym)
	}
	d.rtmap[ds.s] = gotype

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
		newattr(pdie, dwarf.DW_AT_go_runtime_type, dwarf.DW_CLS_GO_TYPEREF, 0, dwSym{s: gts, l: d.ldr})
	}

	if gts != 0 {
		ds := pdie.Sym.(dwSym)
		d.rtmap[ds.s] = gts
		d.tdmap[gts] = ds.s
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
		a.Data = dwSym{s: dwtype, l: d.ldr}
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
		elem := getattr(die, dwarf.DW_AT_go_elem).Data.(dwSym).s
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
		gotype := getattr(die, dwarf.DW_AT_type).Data.(dwSym).s
		keytype := decodetypeMapKey2(d.ldr, d.arch, gotype)
		valtype := decodetypeMapValue2(d.ldr, d.arch, gotype)
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
			fld := newdie(ctxt, dwhk, dwarf.DW_ABRV_ARRAYRANGE, "size", 0)
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
			fld := newdie(ctxt, dwhv, dwarf.DW_ABRV_ARRAYRANGE, "size", 0)
			newattr(fld, dwarf.DW_AT_count, dwarf.DW_CLS_CONSTANT, BucketSize, 0)
			d.newrefattr(fld, dwarf.DW_AT_type, d.uintptrInfoSym)
		})

		// Construct bucket<K,V>
		dwhbs := d.mkinternaltype(ctxt, dwarf.DW_ABRV_STRUCTTYPE, "bucket", keyname, valname, func(dwhb *dwarf.DWDie) {
			// Copy over all fields except the field "data" from the generic
			// bucket. "data" will be replaced with keys/values below.
			d.copychildrenexcept(ctxt, dwhb, bucket, findchild(bucket, "data"))

			fld := newdie(ctxt, dwhb, dwarf.DW_ABRV_STRUCTFIELD, "keys", 0)
			d.newrefattr(fld, dwarf.DW_AT_type, dwhks)
			newmemberoffsetattr(fld, BucketSize)
			fld = newdie(ctxt, dwhb, dwarf.DW_ABRV_STRUCTFIELD, "values", 0)
			d.newrefattr(fld, dwarf.DW_AT_type, dwhvs)
			newmemberoffsetattr(fld, BucketSize+BucketSize*int32(keysize))
			fld = newdie(ctxt, dwhb, dwarf.DW_ABRV_STRUCTFIELD, "overflow", 0)
			d.newrefattr(fld, dwarf.DW_AT_type, d.defptrto(d.dtolsym(dwhb.Sym)))
			newmemberoffsetattr(fld, BucketSize+BucketSize*(int32(keysize)+int32(valsize)))
			if d.arch.RegSize > d.arch.PtrSize {
				fld = newdie(ctxt, dwhb, dwarf.DW_ABRV_STRUCTFIELD, "pad", 0)
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
		elemgotype := getattr(die, dwarf.DW_AT_type).Data.(dwSym).s
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
func (d *dwctxt2) createUnitLength(ctxt *Link, symIdx loader.Sym, v uint64) {
	panic("not yet implemented")
}

// addDwarfAddrField adds a DWARF field in DWARF 64bits or 32bits.
func (d *dwctxt2) addDwarfAddrField(ctxt *Link, symIdx loader.Sym, v uint64) {
	panic("not yet implemented")
}

// addDwarfAddrRef adds a DWARF pointer in DWARF 64bits or 32bits.
func (d *dwctxt2) addDwarfAddrRef(ctxt *Link, symIdx loader.Sym, t *sym.Symbol) {
	panic("not yet implemented")
}

// calcCompUnitRanges calculates the PC ranges of the compilation units.
func (d *dwctxt2) calcCompUnitRanges(ctxt *Link) {
	panic("not yet implemented")
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

// If the pcln table contains runtime/proc.go, use that to set gdbscript path.
func (d *dwctxt2) finddebugruntimepath(symIdx loader.Sym) {
	panic("not yet implemented")
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
	drelocs := d.ldr.Relocs(dsym)
	rslice := drelocs.ReadSyms(nil)
	for i := 0; i < len(rslice); i++ {
		r := &rslice[i]
		if r.Type != objabi.R_DWARFSECREF {
			continue
		}
		// If there is an entry for the symbol in our rtmap, then it
		// means we've processed the type already, and can skip this one.
		if _, ok := d.rtmap[r.Sym]; ok {
			// type already generated
			continue
		}
		// FIXME: is there a way we could avoid materializing the
		// symbol name here?
		sn := d.ldr.SymName(r.Sym)
		tn := sn[len(dwarf.InfoPrefix):]
		ts := d.ldr.Lookup("type."+tn, 0)
		d.defgotype(ts)
	}
}

func (d *dwctxt2) writelines(ctxt *Link, unit *sym.CompilationUnit, ls loader.Sym) {
	panic("not yet implemented")
}

// writepcranges generates the DW_AT_ranges table for compilation unit cu.
func (d *dwctxt2) writepcranges(ctxt *Link, unit *sym.CompilationUnit, base loader.Sym, pcs []dwarf.Range, ranges loader.Sym) {
	panic("not yet implemented")
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

func (d *dwctxt2) writeframes([]loader.Sym) []loader.Sym {
	panic("not yet implemented")
}

/*
 *  Walk DWarfDebugInfoEntries, and emit .debug_info
 */

const (
	COMPUNITHEADERSIZE = 4 + 2 + 4 + 1
)

func (d *dwctxt2) writeinfo(ctxt *Link, syms []loader.Sym, units []*sym.CompilationUnit, abbrevsym loader.Sym, pubNames, pubTypes *pubWriter) []loader.Sym {
	panic("not yet implemented")
}

/*
 *  Emit .debug_pubnames/_types.  _info must have been written before,
 *  because we need die->offs and infoo/infosize;
 */

/*
 *  Emit .debug_pubnames/_types.  _info must have been written before,
 *  because we need die->offs and infoo/infosize;
 */
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

func (d *dwctxt2) writegdbscript(syms []loader.Sym) []loader.Sym {
	panic("not yet implemented")
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
	ds := die.Sym.(dwSym)
	d.rtmap[ds.s] = gotype

	// Map from type to def sym
	d.tdmap[gotype] = ds.s

	return die
}

// dwarfGenerateDebugInfo generated debug info entries for all types,
// variables and functions in the program.
// Along with dwarfGenerateDebugSyms they are the two main entry points into
// dwarf generation: dwarfGenerateDebugInfo does all the work that should be
// done before symbol names are mangled while dwarfGenerateDebugSyms does
// all the work that can only be done after addresses have been assigned to
// text symbols.
func dwarfGenerateDebugInfo2(ctxt *Link) {
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
	newattr(die, dwarf.DW_AT_go_runtime_type, dwarf.DW_CLS_ADDRESS, 0, dwSym{s: d.lookupOrDiag("type.uintptr"), l: d.ldr})

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
	var relocs []loader.Reloc

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
				relocs = drelocs.ReadSyms(relocs)
				for ri := 0; ri < drelocs.Count; ri++ {
					r := &relocs[ri]
					if r.Type == objabi.R_DWARFSECREF {
						rsym := r.Sym
						// NB: there should be a better way to do this that doesn't involve materializing the symbol name and doing string prefix+suffix checks.
						rsn := d.ldr.SymName(rsym)
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
					srelocs := d.ldr.Relocs(symIdx)
					relocs = srelocs.ReadSyms(relocs)
					for i := 0; i < len(relocs); i++ {
						r := &relocs[i]
						if r.Type == objabi.R_USETYPE {
							d.defgotype(r.Sym)
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

// dwarfConvertSymbols is invoked around the time that loader.LoadFull
// runs (converting all loader.Sym's into sym.Symbols); it walks
// through dwarf DIE objects and rewrites loader.Sym refs to
// sym.Symbol there as well. This is obviously a temporary function.
func dwarfConvertSymbols(ctxt *Link) {
	convdies := make(map[*dwarf.DWDie]bool)
	for _, lib := range ctxt.Library {
		for _, unit := range lib.Units {
			convertSymbolsInDIE(ctxt, unit.DWInfo, convdies)
		}
	}
	convertSymbolsInDIE(ctxt, &dwtypes, convdies)

	// Convert over the unit function DIE and abstract function DIE lists.
	for _, lib := range ctxt.Library {
		for _, unit := range lib.Units {
			for _, fd := range unit.FuncDIEs2 {
				ds := ctxt.loader.Syms[fd]
				if ds == nil {
					panic("bad")
				}
				unit.FuncDIEs = append(unit.FuncDIEs, ds)
			}
			for _, fd := range unit.RangeSyms2 {
				ds := ctxt.loader.Syms[fd]
				if ds == nil {
					panic("bad")
				}
				unit.RangeSyms = append(unit.RangeSyms, ds)
			}
			for _, fd := range unit.AbsFnDIEs2 {
				ds := ctxt.loader.Syms[fd]
				if ds == nil {
					panic("bad")
				}
				unit.AbsFnDIEs = append(unit.AbsFnDIEs, ds)
			}
			if unit.Consts2 != 0 {
				ds := ctxt.loader.Syms[unit.Consts2]
				if ds == nil {
					panic("bad")
				}
				unit.Consts = ds
			}
		}
	}
}

func convertSymbolsInDIE(ctxt *Link, die *dwarf.DWDie, convdies map[*dwarf.DWDie]bool) {
	if die == nil {
		return
	}
	if convdies[die] {
		return
	}
	convdies[die] = true
	if die.Sym != nil {
		symIdx, ok := die.Sym.(dwSym)
		if !ok {
			panic("bad die sym field")
		}
		ls := symIdx.s
		if ls == 0 {
			panic("zero loader sym for die")
		}
		die.Sym = ctxt.loader.Syms[symIdx.s]
	}
	for a := die.Attr; a != nil; a = a.Link {
		if attrSym, ok := a.Data.(dwSym); ok {
			a.Data = ctxt.loader.Syms[attrSym.s]
		}
	}
	convertSymbolsInDIE(ctxt, die.Child, convdies)
	convertSymbolsInDIE(ctxt, die.Link, convdies)
}

// dwarfGenerateDebugSyms constructs debug_line, debug_frame, debug_loc,
// debug_pubnames and debug_pubtypes. It also writes out the debug_info
// section using symbols generated in dwarfGenerateDebugInfo2.
func (d *dwctxt2) dwarfGenerateDebugSyms(ctxt *Link) {
	if !dwarfEnabled(ctxt) {
		return
	}
	panic("not yet implemented")
}

func (d *dwctxt2) collectlocs(ctxt *Link, syms []loader.Sym, units []*sym.CompilationUnit) []loader.Sym {
	panic("not yet implemented")
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
