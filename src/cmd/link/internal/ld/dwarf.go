// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO/NICETOHAVE:
//   - eliminate DW_CLS_ if not used
//   - package info in compilation units
//   - assign global variables and types to their packages
//   - gdb uses c syntax, meaning clumsy quoting is needed for go identifiers. eg
//     ptype struct '[]uint8' and qualifiers need to be quoted away
//   - file:line info for variables
//   - make strings a typedef so prettyprinters can see the underlying string type

package ld

import (
	"cmd/internal/dwarf"
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"cmd/link/internal/sym"
	"fmt"
	"log"
	"strings"
)

type dwctxt struct {
	linkctxt *Link
}

func (c dwctxt) PtrSize() int {
	return c.linkctxt.Arch.PtrSize
}
func (c dwctxt) AddInt(s dwarf.Sym, size int, i int64) {
	ls := s.(*sym.Symbol)
	ls.AddUintXX(c.linkctxt.Arch, uint64(i), size)
}
func (c dwctxt) AddBytes(s dwarf.Sym, b []byte) {
	ls := s.(*sym.Symbol)
	ls.AddBytes(b)
}
func (c dwctxt) AddString(s dwarf.Sym, v string) {
	Addstring(s.(*sym.Symbol), v)
}

func (c dwctxt) AddAddress(s dwarf.Sym, data interface{}, value int64) {
	if value != 0 {
		value -= (data.(*sym.Symbol)).Value
	}
	s.(*sym.Symbol).AddAddrPlus(c.linkctxt.Arch, data.(*sym.Symbol), value)
}

func (c dwctxt) AddCURelativeAddress(s dwarf.Sym, data interface{}, value int64) {
	if value != 0 {
		value -= (data.(*sym.Symbol)).Value
	}
	s.(*sym.Symbol).AddCURelativeAddrPlus(c.linkctxt.Arch, data.(*sym.Symbol), value)
}

func (c dwctxt) AddSectionOffset(s dwarf.Sym, size int, t interface{}, ofs int64) {
	ls := s.(*sym.Symbol)
	switch size {
	default:
		Errorf(ls, "invalid size %d in adddwarfref\n", size)
		fallthrough
	case c.linkctxt.Arch.PtrSize:
		ls.AddAddr(c.linkctxt.Arch, t.(*sym.Symbol))
	case 4:
		ls.AddAddrPlus4(t.(*sym.Symbol), 0)
	}
	r := &ls.R[len(ls.R)-1]
	r.Type = objabi.R_DWARFSECREF
	r.Add = ofs
}

func (c dwctxt) Logf(format string, args ...interface{}) {
	c.linkctxt.Logf(format, args...)
}

// At the moment these interfaces are only used in the compiler.

func (c dwctxt) AddFileRef(s dwarf.Sym, f interface{}) {
	panic("should be used only in the compiler")
}

func (c dwctxt) CurrentOffset(s dwarf.Sym) int64 {
	panic("should be used only in the compiler")
}

func (c dwctxt) RecordDclReference(s dwarf.Sym, t dwarf.Sym, dclIdx int, inlIndex int) {
	panic("should be used only in the compiler")
}

func (c dwctxt) RecordChildDieOffsets(s dwarf.Sym, vars []*dwarf.Var, offsets []int32) {
	panic("should be used only in the compiler")
}

var gdbscript string

var dwarfp []*sym.Symbol

func writeabbrev(ctxt *Link) *sym.Symbol {
	s := ctxt.Syms.Lookup(".debug_abbrev", 0)
	s.Type = sym.SDWARFSECT
	s.AddBytes(dwarf.GetAbbrev())
	return s
}

/*
 * Root DIEs for compilation units, types and global variables.
 */
var dwroot dwarf.DWDie

var dwtypes dwarf.DWDie

var dwglobals dwarf.DWDie

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
func newdie(ctxt *Link, parent *dwarf.DWDie, abbrev int, name string, version int) *dwarf.DWDie {
	die := new(dwarf.DWDie)
	die.Abbrev = abbrev
	die.Link = parent.Child
	parent.Child = die

	newattr(die, dwarf.DW_AT_name, dwarf.DW_CLS_STRING, int64(len(name)), name)

	if name != "" && (abbrev <= dwarf.DW_ABRV_VARIABLE || abbrev >= dwarf.DW_ABRV_NULLTYPE) {
		if abbrev != dwarf.DW_ABRV_VARIABLE || version == 0 {
			if abbrev == dwarf.DW_ABRV_COMPUNIT {
				// Avoid collisions with "real" symbol names.
				name = ".pkg." + name
			}
			s := ctxt.Syms.Lookup(dwarf.InfoPrefix+name, version)
			s.Attr |= sym.AttrNotInSymbolTable
			s.Type = sym.SDWARFINFO
			die.Sym = s
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

func walksymtypedef(ctxt *Link, s *sym.Symbol) *sym.Symbol {
	if t := ctxt.Syms.ROLookup(s.Name+"..def", int(s.Version)); t != nil {
		return t
	}
	return s
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

func find(ctxt *Link, name string) *sym.Symbol {
	n := append(prefixBuf, name...)
	// The string allocation below is optimized away because it is only used in a map lookup.
	s := ctxt.Syms.ROLookup(string(n), 0)
	prefixBuf = n[:len(dwarf.InfoPrefix)]
	if s != nil && s.Type == sym.SDWARFINFO {
		return s
	}
	return nil
}

func mustFind(ctxt *Link, name string) *sym.Symbol {
	r := find(ctxt, name)
	if r == nil {
		Exitf("dwarf find: cannot find %s", name)
	}
	return r
}

func adddwarfref(ctxt *Link, s *sym.Symbol, t *sym.Symbol, size int) int64 {
	var result int64
	switch size {
	default:
		Errorf(s, "invalid size %d in adddwarfref\n", size)
		fallthrough
	case ctxt.Arch.PtrSize:
		result = s.AddAddr(ctxt.Arch, t)
	case 4:
		result = s.AddAddrPlus4(t, 0)
	}
	r := &s.R[len(s.R)-1]
	r.Type = objabi.R_DWARFSECREF
	return result
}

func newrefattr(die *dwarf.DWDie, attr uint16, ref *sym.Symbol) *dwarf.DWAttr {
	if ref == nil {
		return nil
	}
	return newattr(die, attr, dwarf.DW_CLS_REFERENCE, 0, ref)
}

func putdies(linkctxt *Link, ctxt dwarf.Context, syms []*sym.Symbol, die *dwarf.DWDie) []*sym.Symbol {
	for ; die != nil; die = die.Link {
		syms = putdie(linkctxt, ctxt, syms, die)
	}
	syms[len(syms)-1].AddUint8(0)

	return syms
}

func dtolsym(s dwarf.Sym) *sym.Symbol {
	if s == nil {
		return nil
	}
	return s.(*sym.Symbol)
}

func putdie(linkctxt *Link, ctxt dwarf.Context, syms []*sym.Symbol, die *dwarf.DWDie) []*sym.Symbol {
	s := dtolsym(die.Sym)
	if s == nil {
		s = syms[len(syms)-1]
	} else {
		if s.Attr.OnList() {
			log.Fatalf("symbol %s listed multiple times", s.Name)
		}
		s.Attr |= sym.AttrOnList
		syms = append(syms, s)
	}
	dwarf.Uleb128put(ctxt, s, int64(die.Abbrev))
	dwarf.PutAttrs(ctxt, s, die.Abbrev, die.Attr)
	if dwarf.HasChildren(die) {
		return putdies(linkctxt, ctxt, syms, die.Child)
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
func newabslocexprattr(die *dwarf.DWDie, addr int64, sym *sym.Symbol) {
	newattr(die, dwarf.DW_AT_location, dwarf.DW_CLS_ADDRESS, addr, sym)
	// below
}

// Lookup predefined types
func lookupOrDiag(ctxt *Link, n string) *sym.Symbol {
	s := ctxt.Syms.ROLookup(n, 0)
	if s == nil || s.Size == 0 {
		Exitf("dwarf: missing type: %s", n)
	}

	return s
}

func dotypedef(ctxt *Link, parent *dwarf.DWDie, name string, def *dwarf.DWDie) {
	// Only emit typedefs for real names.
	if strings.HasPrefix(name, "map[") {
		return
	}
	if strings.HasPrefix(name, "struct {") {
		return
	}
	if strings.HasPrefix(name, "chan ") {
		return
	}
	if name[0] == '[' || name[0] == '*' {
		return
	}
	if def == nil {
		Errorf(nil, "dwarf: bad def in dotypedef")
	}

	s := ctxt.Syms.Lookup(dtolsym(def.Sym).Name+"..def", 0)
	s.Attr |= sym.AttrNotInSymbolTable
	s.Type = sym.SDWARFINFO
	def.Sym = s

	// The typedef entry must be created after the def,
	// so that future lookups will find the typedef instead
	// of the real definition. This hooks the typedef into any
	// circular definition loops, so that gdb can understand them.
	die := newdie(ctxt, parent, dwarf.DW_ABRV_TYPEDECL, name, 0)

	newrefattr(die, dwarf.DW_AT_type, s)
}

// Define gotype, for composite ones recurse into constituents.
func defgotype(ctxt *Link, gotype *sym.Symbol) *sym.Symbol {
	if gotype == nil {
		return mustFind(ctxt, "<unspecified>")
	}

	if !strings.HasPrefix(gotype.Name, "type.") {
		Errorf(gotype, "dwarf: type name doesn't start with \"type.\"")
		return mustFind(ctxt, "<unspecified>")
	}

	name := gotype.Name[5:] // could also decode from Type.string

	sdie := find(ctxt, name)

	if sdie != nil {
		return sdie
	}

	return newtype(ctxt, gotype).Sym.(*sym.Symbol)
}

func newtype(ctxt *Link, gotype *sym.Symbol) *dwarf.DWDie {
	name := gotype.Name[5:] // could also decode from Type.string
	kind := decodetypeKind(ctxt.Arch, gotype)
	bytesize := decodetypeSize(ctxt.Arch, gotype)

	var die *dwarf.DWDie
	switch kind {
	case objabi.KindBool:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_BASETYPE, name, 0)
		newattr(die, dwarf.DW_AT_encoding, dwarf.DW_CLS_CONSTANT, dwarf.DW_ATE_boolean, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case objabi.KindInt,
		objabi.KindInt8,
		objabi.KindInt16,
		objabi.KindInt32,
		objabi.KindInt64:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_BASETYPE, name, 0)
		newattr(die, dwarf.DW_AT_encoding, dwarf.DW_CLS_CONSTANT, dwarf.DW_ATE_signed, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case objabi.KindUint,
		objabi.KindUint8,
		objabi.KindUint16,
		objabi.KindUint32,
		objabi.KindUint64,
		objabi.KindUintptr:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_BASETYPE, name, 0)
		newattr(die, dwarf.DW_AT_encoding, dwarf.DW_CLS_CONSTANT, dwarf.DW_ATE_unsigned, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case objabi.KindFloat32,
		objabi.KindFloat64:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_BASETYPE, name, 0)
		newattr(die, dwarf.DW_AT_encoding, dwarf.DW_CLS_CONSTANT, dwarf.DW_ATE_float, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case objabi.KindComplex64,
		objabi.KindComplex128:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_BASETYPE, name, 0)
		newattr(die, dwarf.DW_AT_encoding, dwarf.DW_CLS_CONSTANT, dwarf.DW_ATE_complex_float, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case objabi.KindArray:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_ARRAYTYPE, name, 0)
		dotypedef(ctxt, &dwtypes, name, die)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		s := decodetypeArrayElem(ctxt.Arch, gotype)
		newrefattr(die, dwarf.DW_AT_type, defgotype(ctxt, s))
		fld := newdie(ctxt, die, dwarf.DW_ABRV_ARRAYRANGE, "range", 0)

		// use actual length not upper bound; correct for 0-length arrays.
		newattr(fld, dwarf.DW_AT_count, dwarf.DW_CLS_CONSTANT, decodetypeArrayLen(ctxt.Arch, gotype), 0)

		newrefattr(fld, dwarf.DW_AT_type, mustFind(ctxt, "uintptr"))

	case objabi.KindChan:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_CHANTYPE, name, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		s := decodetypeChanElem(ctxt.Arch, gotype)
		newrefattr(die, dwarf.DW_AT_go_elem, defgotype(ctxt, s))
		// Save elem type for synthesizechantypes. We could synthesize here
		// but that would change the order of DIEs we output.
		newrefattr(die, dwarf.DW_AT_type, s)

	case objabi.KindFunc:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_FUNCTYPE, name, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		dotypedef(ctxt, &dwtypes, name, die)
		newrefattr(die, dwarf.DW_AT_type, mustFind(ctxt, "void"))
		nfields := decodetypeFuncInCount(ctxt.Arch, gotype)
		var fld *dwarf.DWDie
		var s *sym.Symbol
		for i := 0; i < nfields; i++ {
			s = decodetypeFuncInType(ctxt.Arch, gotype, i)
			fld = newdie(ctxt, die, dwarf.DW_ABRV_FUNCTYPEPARAM, s.Name[5:], 0)
			newrefattr(fld, dwarf.DW_AT_type, defgotype(ctxt, s))
		}

		if decodetypeFuncDotdotdot(ctxt.Arch, gotype) {
			newdie(ctxt, die, dwarf.DW_ABRV_DOTDOTDOT, "...", 0)
		}
		nfields = decodetypeFuncOutCount(ctxt.Arch, gotype)
		for i := 0; i < nfields; i++ {
			s = decodetypeFuncOutType(ctxt.Arch, gotype, i)
			fld = newdie(ctxt, die, dwarf.DW_ABRV_FUNCTYPEPARAM, s.Name[5:], 0)
			newrefattr(fld, dwarf.DW_AT_type, defptrto(ctxt, defgotype(ctxt, s)))
		}

	case objabi.KindInterface:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_IFACETYPE, name, 0)
		dotypedef(ctxt, &dwtypes, name, die)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		nfields := int(decodetypeIfaceMethodCount(ctxt.Arch, gotype))
		var s *sym.Symbol
		if nfields == 0 {
			s = lookupOrDiag(ctxt, "type.runtime.eface")
		} else {
			s = lookupOrDiag(ctxt, "type.runtime.iface")
		}
		newrefattr(die, dwarf.DW_AT_type, defgotype(ctxt, s))

	case objabi.KindMap:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_MAPTYPE, name, 0)
		s := decodetypeMapKey(ctxt.Arch, gotype)
		newrefattr(die, dwarf.DW_AT_go_key, defgotype(ctxt, s))
		s = decodetypeMapValue(ctxt.Arch, gotype)
		newrefattr(die, dwarf.DW_AT_go_elem, defgotype(ctxt, s))
		// Save gotype for use in synthesizemaptypes. We could synthesize here,
		// but that would change the order of the DIEs.
		newrefattr(die, dwarf.DW_AT_type, gotype)

	case objabi.KindPtr:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_PTRTYPE, name, 0)
		dotypedef(ctxt, &dwtypes, name, die)
		s := decodetypePtrElem(ctxt.Arch, gotype)
		newrefattr(die, dwarf.DW_AT_type, defgotype(ctxt, s))

	case objabi.KindSlice:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_SLICETYPE, name, 0)
		dotypedef(ctxt, &dwtypes, name, die)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		s := decodetypeArrayElem(ctxt.Arch, gotype)
		elem := defgotype(ctxt, s)
		newrefattr(die, dwarf.DW_AT_go_elem, elem)

	case objabi.KindString:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_STRINGTYPE, name, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case objabi.KindStruct:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_STRUCTTYPE, name, 0)
		dotypedef(ctxt, &dwtypes, name, die)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		nfields := decodetypeStructFieldCount(ctxt.Arch, gotype)
		for i := 0; i < nfields; i++ {
			f := decodetypeStructFieldName(ctxt.Arch, gotype, i)
			s := decodetypeStructFieldType(ctxt.Arch, gotype, i)
			if f == "" {
				f = s.Name[5:] // skip "type."
			}
			fld := newdie(ctxt, die, dwarf.DW_ABRV_STRUCTFIELD, f, 0)
			newrefattr(fld, dwarf.DW_AT_type, defgotype(ctxt, s))
			offsetAnon := decodetypeStructFieldOffsAnon(ctxt.Arch, gotype, i)
			newmemberoffsetattr(fld, int32(offsetAnon>>1))
			if offsetAnon&1 != 0 { // is embedded field
				newattr(fld, dwarf.DW_AT_go_embedded_field, dwarf.DW_CLS_FLAG, 1, 0)
			}
		}

	case objabi.KindUnsafePointer:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_BARE_PTRTYPE, name, 0)

	default:
		Errorf(gotype, "dwarf: definition of unknown kind %d", kind)
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_TYPEDECL, name, 0)
		newrefattr(die, dwarf.DW_AT_type, mustFind(ctxt, "<unspecified>"))
	}

	newattr(die, dwarf.DW_AT_go_kind, dwarf.DW_CLS_CONSTANT, int64(kind), 0)

	if _, ok := prototypedies[gotype.Name]; ok {
		prototypedies[gotype.Name] = die
	}

	return die
}

func nameFromDIESym(dwtype *sym.Symbol) string {
	return strings.TrimSuffix(dwtype.Name[len(dwarf.InfoPrefix):], "..def")
}

// Find or construct *T given T.
func defptrto(ctxt *Link, dwtype *sym.Symbol) *sym.Symbol {
	ptrname := "*" + nameFromDIESym(dwtype)
	die := find(ctxt, ptrname)
	if die == nil {
		pdie := newdie(ctxt, &dwtypes, dwarf.DW_ABRV_PTRTYPE, ptrname, 0)
		newrefattr(pdie, dwarf.DW_AT_type, dwtype)
		return dtolsym(pdie.Sym)
	}

	return die
}

// Copies src's children into dst. Copies attributes by value.
// DWAttr.data is copied as pointer only. If except is one of
// the top-level children, it will not be copied.
func copychildrenexcept(ctxt *Link, dst *dwarf.DWDie, src *dwarf.DWDie, except *dwarf.DWDie) {
	for src = src.Child; src != nil; src = src.Link {
		if src == except {
			continue
		}
		c := newdie(ctxt, dst, src.Abbrev, getattr(src, dwarf.DW_AT_name).Data.(string), 0)
		for a := src.Attr; a != nil; a = a.Link {
			newattr(c, a.Atr, int(a.Cls), a.Value, a.Data)
		}
		copychildrenexcept(ctxt, c, src, nil)
	}

	reverselist(&dst.Child)
}

func copychildren(ctxt *Link, dst *dwarf.DWDie, src *dwarf.DWDie) {
	copychildrenexcept(ctxt, dst, src, nil)
}

// Search children (assumed to have TAG_member) for the one named
// field and set its AT_type to dwtype
func substitutetype(structdie *dwarf.DWDie, field string, dwtype *sym.Symbol) {
	child := findchild(structdie, field)
	if child == nil {
		Exitf("dwarf substitutetype: %s does not have member %s",
			getattr(structdie, dwarf.DW_AT_name).Data, field)
		return
	}

	a := getattr(child, dwarf.DW_AT_type)
	if a != nil {
		a.Data = dwtype
	} else {
		newrefattr(child, dwarf.DW_AT_type, dwtype)
	}
}

func findprotodie(ctxt *Link, name string) *dwarf.DWDie {
	die, ok := prototypedies[name]
	if ok && die == nil {
		defgotype(ctxt, lookupOrDiag(ctxt, name))
		die = prototypedies[name]
	}
	return die
}

func synthesizestringtypes(ctxt *Link, die *dwarf.DWDie) {
	prototype := walktypedef(findprotodie(ctxt, "type.runtime.stringStructDWARF"))
	if prototype == nil {
		return
	}

	for ; die != nil; die = die.Link {
		if die.Abbrev != dwarf.DW_ABRV_STRINGTYPE {
			continue
		}
		copychildren(ctxt, die, prototype)
	}
}

func synthesizeslicetypes(ctxt *Link, die *dwarf.DWDie) {
	prototype := walktypedef(findprotodie(ctxt, "type.runtime.slice"))
	if prototype == nil {
		return
	}

	for ; die != nil; die = die.Link {
		if die.Abbrev != dwarf.DW_ABRV_SLICETYPE {
			continue
		}
		copychildren(ctxt, die, prototype)
		elem := getattr(die, dwarf.DW_AT_go_elem).Data.(*sym.Symbol)
		substitutetype(die, "array", defptrto(ctxt, elem))
	}
}

func mkinternaltypename(base string, arg1 string, arg2 string) string {
	var buf string

	if arg2 == "" {
		buf = fmt.Sprintf("%s<%s>", base, arg1)
	} else {
		buf = fmt.Sprintf("%s<%s,%s>", base, arg1, arg2)
	}
	n := buf
	return n
}

// synthesizemaptypes is way too closely married to runtime/hashmap.c
const (
	MaxKeySize = 128
	MaxValSize = 128
	BucketSize = 8
)

func mkinternaltype(ctxt *Link, abbrev int, typename, keyname, valname string, f func(*dwarf.DWDie)) *sym.Symbol {
	name := mkinternaltypename(typename, keyname, valname)
	symname := dwarf.InfoPrefix + name
	s := ctxt.Syms.ROLookup(symname, 0)
	if s != nil && s.Type == sym.SDWARFINFO {
		return s
	}
	die := newdie(ctxt, &dwtypes, abbrev, name, 0)
	f(die)
	return dtolsym(die.Sym)
}

func synthesizemaptypes(ctxt *Link, die *dwarf.DWDie) {
	hash := walktypedef(findprotodie(ctxt, "type.runtime.hmap"))
	bucket := walktypedef(findprotodie(ctxt, "type.runtime.bmap"))

	if hash == nil {
		return
	}

	for ; die != nil; die = die.Link {
		if die.Abbrev != dwarf.DW_ABRV_MAPTYPE {
			continue
		}
		gotype := getattr(die, dwarf.DW_AT_type).Data.(*sym.Symbol)
		keytype := decodetypeMapKey(ctxt.Arch, gotype)
		valtype := decodetypeMapValue(ctxt.Arch, gotype)
		keysize, valsize := decodetypeSize(ctxt.Arch, keytype), decodetypeSize(ctxt.Arch, valtype)
		keytype, valtype = walksymtypedef(ctxt, defgotype(ctxt, keytype)), walksymtypedef(ctxt, defgotype(ctxt, valtype))

		// compute size info like hashmap.c does.
		indirectKey, indirectVal := false, false
		if keysize > MaxKeySize {
			keysize = int64(ctxt.Arch.PtrSize)
			indirectKey = true
		}
		if valsize > MaxValSize {
			valsize = int64(ctxt.Arch.PtrSize)
			indirectVal = true
		}

		// Construct type to represent an array of BucketSize keys
		keyname := nameFromDIESym(keytype)
		dwhks := mkinternaltype(ctxt, dwarf.DW_ABRV_ARRAYTYPE, "[]key", keyname, "", func(dwhk *dwarf.DWDie) {
			newattr(dwhk, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, BucketSize*keysize, 0)
			t := keytype
			if indirectKey {
				t = defptrto(ctxt, keytype)
			}
			newrefattr(dwhk, dwarf.DW_AT_type, t)
			fld := newdie(ctxt, dwhk, dwarf.DW_ABRV_ARRAYRANGE, "size", 0)
			newattr(fld, dwarf.DW_AT_count, dwarf.DW_CLS_CONSTANT, BucketSize, 0)
			newrefattr(fld, dwarf.DW_AT_type, mustFind(ctxt, "uintptr"))
		})

		// Construct type to represent an array of BucketSize values
		valname := nameFromDIESym(valtype)
		dwhvs := mkinternaltype(ctxt, dwarf.DW_ABRV_ARRAYTYPE, "[]val", valname, "", func(dwhv *dwarf.DWDie) {
			newattr(dwhv, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, BucketSize*valsize, 0)
			t := valtype
			if indirectVal {
				t = defptrto(ctxt, valtype)
			}
			newrefattr(dwhv, dwarf.DW_AT_type, t)
			fld := newdie(ctxt, dwhv, dwarf.DW_ABRV_ARRAYRANGE, "size", 0)
			newattr(fld, dwarf.DW_AT_count, dwarf.DW_CLS_CONSTANT, BucketSize, 0)
			newrefattr(fld, dwarf.DW_AT_type, mustFind(ctxt, "uintptr"))
		})

		// Construct bucket<K,V>
		dwhbs := mkinternaltype(ctxt, dwarf.DW_ABRV_STRUCTTYPE, "bucket", keyname, valname, func(dwhb *dwarf.DWDie) {
			// Copy over all fields except the field "data" from the generic
			// bucket. "data" will be replaced with keys/values below.
			copychildrenexcept(ctxt, dwhb, bucket, findchild(bucket, "data"))

			fld := newdie(ctxt, dwhb, dwarf.DW_ABRV_STRUCTFIELD, "keys", 0)
			newrefattr(fld, dwarf.DW_AT_type, dwhks)
			newmemberoffsetattr(fld, BucketSize)
			fld = newdie(ctxt, dwhb, dwarf.DW_ABRV_STRUCTFIELD, "values", 0)
			newrefattr(fld, dwarf.DW_AT_type, dwhvs)
			newmemberoffsetattr(fld, BucketSize+BucketSize*int32(keysize))
			fld = newdie(ctxt, dwhb, dwarf.DW_ABRV_STRUCTFIELD, "overflow", 0)
			newrefattr(fld, dwarf.DW_AT_type, defptrto(ctxt, dtolsym(dwhb.Sym)))
			newmemberoffsetattr(fld, BucketSize+BucketSize*(int32(keysize)+int32(valsize)))
			if ctxt.Arch.RegSize > ctxt.Arch.PtrSize {
				fld = newdie(ctxt, dwhb, dwarf.DW_ABRV_STRUCTFIELD, "pad", 0)
				newrefattr(fld, dwarf.DW_AT_type, mustFind(ctxt, "uintptr"))
				newmemberoffsetattr(fld, BucketSize+BucketSize*(int32(keysize)+int32(valsize))+int32(ctxt.Arch.PtrSize))
			}

			newattr(dwhb, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, BucketSize+BucketSize*keysize+BucketSize*valsize+int64(ctxt.Arch.RegSize), 0)
		})

		// Construct hash<K,V>
		dwhs := mkinternaltype(ctxt, dwarf.DW_ABRV_STRUCTTYPE, "hash", keyname, valname, func(dwh *dwarf.DWDie) {
			copychildren(ctxt, dwh, hash)
			substitutetype(dwh, "buckets", defptrto(ctxt, dwhbs))
			substitutetype(dwh, "oldbuckets", defptrto(ctxt, dwhbs))
			newattr(dwh, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, getattr(hash, dwarf.DW_AT_byte_size).Value, nil)
		})

		// make map type a pointer to hash<K,V>
		newrefattr(die, dwarf.DW_AT_type, defptrto(ctxt, dwhs))
	}
}

func synthesizechantypes(ctxt *Link, die *dwarf.DWDie) {
	sudog := walktypedef(findprotodie(ctxt, "type.runtime.sudog"))
	waitq := walktypedef(findprotodie(ctxt, "type.runtime.waitq"))
	hchan := walktypedef(findprotodie(ctxt, "type.runtime.hchan"))
	if sudog == nil || waitq == nil || hchan == nil {
		return
	}

	sudogsize := int(getattr(sudog, dwarf.DW_AT_byte_size).Value)

	for ; die != nil; die = die.Link {
		if die.Abbrev != dwarf.DW_ABRV_CHANTYPE {
			continue
		}
		elemgotype := getattr(die, dwarf.DW_AT_type).Data.(*sym.Symbol)
		elemname := elemgotype.Name[5:]
		elemtype := walksymtypedef(ctxt, defgotype(ctxt, elemgotype))

		// sudog<T>
		dwss := mkinternaltype(ctxt, dwarf.DW_ABRV_STRUCTTYPE, "sudog", elemname, "", func(dws *dwarf.DWDie) {
			copychildren(ctxt, dws, sudog)
			substitutetype(dws, "elem", defptrto(ctxt, elemtype))
			newattr(dws, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, int64(sudogsize), nil)
		})

		// waitq<T>
		dwws := mkinternaltype(ctxt, dwarf.DW_ABRV_STRUCTTYPE, "waitq", elemname, "", func(dww *dwarf.DWDie) {

			copychildren(ctxt, dww, waitq)
			substitutetype(dww, "first", defptrto(ctxt, dwss))
			substitutetype(dww, "last", defptrto(ctxt, dwss))
			newattr(dww, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, getattr(waitq, dwarf.DW_AT_byte_size).Value, nil)
		})

		// hchan<T>
		dwhs := mkinternaltype(ctxt, dwarf.DW_ABRV_STRUCTTYPE, "hchan", elemname, "", func(dwh *dwarf.DWDie) {
			copychildren(ctxt, dwh, hchan)
			substitutetype(dwh, "recvq", dwws)
			substitutetype(dwh, "sendq", dwws)
			newattr(dwh, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, getattr(hchan, dwarf.DW_AT_byte_size).Value, nil)
		})

		newrefattr(die, dwarf.DW_AT_type, defptrto(ctxt, dwhs))
	}
}

// For use with pass.c::genasmsym
func defdwsymb(ctxt *Link, s *sym.Symbol, str string, t SymbolType, v int64, gotype *sym.Symbol) {
	if strings.HasPrefix(str, "go.string.") {
		return
	}
	if strings.HasPrefix(str, "runtime.gcbits.") {
		return
	}

	if strings.HasPrefix(str, "type.") && str != "type.*" && !strings.HasPrefix(str, "type..") {
		defgotype(ctxt, s)
		return
	}

	var dv *dwarf.DWDie

	var dt *sym.Symbol
	switch t {
	default:
		return

	case DataSym, BSSSym:
		dv = newdie(ctxt, &dwglobals, dwarf.DW_ABRV_VARIABLE, str, int(s.Version))
		newabslocexprattr(dv, v, s)
		if s.Version == 0 {
			newattr(dv, dwarf.DW_AT_external, dwarf.DW_CLS_FLAG, 1, 0)
		}
		fallthrough

	case AutoSym, ParamSym, DeletedAutoSym:
		dt = defgotype(ctxt, gotype)
	}

	if dv != nil {
		newrefattr(dv, dwarf.DW_AT_type, dt)
	}
}

// compilationUnit is per-compilation unit (equivalently, per-package)
// debug-related data.
type compilationUnit struct {
	lib       *sym.Library
	consts    *sym.Symbol   // Package constants DIEs
	pcs       []dwarf.Range // PC ranges, relative to textp[0]
	dwinfo    *dwarf.DWDie  // CU root DIE
	funcDIEs  []*sym.Symbol // Function DIE subtrees
	absFnDIEs []*sym.Symbol // Abstract function DIE subtrees
}

// getCompilationUnits divides the symbols in ctxt.Textp by package.
func getCompilationUnits(ctxt *Link) []*compilationUnit {
	units := []*compilationUnit{}
	index := make(map[*sym.Library]*compilationUnit)
	var prevUnit *compilationUnit
	for _, s := range ctxt.Textp {
		if s.FuncInfo == nil {
			continue
		}
		unit := index[s.Lib]
		if unit == nil {
			unit = &compilationUnit{lib: s.Lib}
			if s := ctxt.Syms.ROLookup(dwarf.ConstInfoPrefix+s.Lib.Pkg, 0); s != nil {
				importInfoSymbol(ctxt, s)
				unit.consts = s
			}
			units = append(units, unit)
			index[s.Lib] = unit
		}

		// Update PC ranges.
		//
		// We don't simply compare the end of the previous
		// symbol with the start of the next because there's
		// often a little padding between them. Instead, we
		// only create boundaries between symbols from
		// different units.
		if prevUnit != unit {
			unit.pcs = append(unit.pcs, dwarf.Range{Start: s.Value - unit.lib.Textp[0].Value})
			prevUnit = unit
		}
		unit.pcs[len(unit.pcs)-1].End = s.Value - unit.lib.Textp[0].Value + s.Size
	}
	return units
}

func movetomodule(parent *dwarf.DWDie) {
	die := dwroot.Child.Child
	if die == nil {
		dwroot.Child.Child = parent.Child
		return
	}
	for die.Link != nil {
		die = die.Link
	}
	die.Link = parent.Child
}

// If the pcln table contains runtime/proc.go, use that to set gdbscript path.
func finddebugruntimepath(s *sym.Symbol) {
	if gdbscript != "" {
		return
	}

	for i := range s.FuncInfo.File {
		f := s.FuncInfo.File[i]
		// We can't use something that may be dead-code
		// eliminated from a binary here. proc.go contains
		// main and the scheduler, so it's not going anywhere.
		if i := strings.Index(f.Name, "runtime/proc.go"); i >= 0 {
			gdbscript = f.Name[:i] + "runtime/runtime-gdb.py"
			break
		}
	}
}

/*
 * Generate a sequence of opcodes that is as short as possible.
 * See section 6.2.5
 */
const (
	LINE_BASE   = -4
	LINE_RANGE  = 10
	PC_RANGE    = (255 - OPCODE_BASE) / LINE_RANGE
	OPCODE_BASE = 10
)

func putpclcdelta(linkctxt *Link, ctxt dwarf.Context, s *sym.Symbol, deltaPC uint64, deltaLC int64) {
	// Choose a special opcode that minimizes the number of bytes needed to
	// encode the remaining PC delta and LC delta.
	var opcode int64
	if deltaLC < LINE_BASE {
		if deltaPC >= PC_RANGE {
			opcode = OPCODE_BASE + (LINE_RANGE * PC_RANGE)
		} else {
			opcode = OPCODE_BASE + (LINE_RANGE * int64(deltaPC))
		}
	} else if deltaLC < LINE_BASE+LINE_RANGE {
		if deltaPC >= PC_RANGE {
			opcode = OPCODE_BASE + (deltaLC - LINE_BASE) + (LINE_RANGE * PC_RANGE)
			if opcode > 255 {
				opcode -= LINE_RANGE
			}
		} else {
			opcode = OPCODE_BASE + (deltaLC - LINE_BASE) + (LINE_RANGE * int64(deltaPC))
		}
	} else {
		if deltaPC <= PC_RANGE {
			opcode = OPCODE_BASE + (LINE_RANGE - 1) + (LINE_RANGE * int64(deltaPC))
			if opcode > 255 {
				opcode = 255
			}
		} else {
			// Use opcode 249 (pc+=23, lc+=5) or 255 (pc+=24, lc+=1).
			//
			// Let x=deltaPC-PC_RANGE.  If we use opcode 255, x will be the remaining
			// deltaPC that we need to encode separately before emitting 255.  If we
			// use opcode 249, we will need to encode x+1.  If x+1 takes one more
			// byte to encode than x, then we use opcode 255.
			//
			// In all other cases x and x+1 take the same number of bytes to encode,
			// so we use opcode 249, which may save us a byte in encoding deltaLC,
			// for similar reasons.
			switch deltaPC - PC_RANGE {
			// PC_RANGE is the largest deltaPC we can encode in one byte, using
			// DW_LNS_const_add_pc.
			//
			// (1<<16)-1 is the largest deltaPC we can encode in three bytes, using
			// DW_LNS_fixed_advance_pc.
			//
			// (1<<(7n))-1 is the largest deltaPC we can encode in n+1 bytes for
			// n=1,3,4,5,..., using DW_LNS_advance_pc.
			case PC_RANGE, (1 << 7) - 1, (1 << 16) - 1, (1 << 21) - 1, (1 << 28) - 1,
				(1 << 35) - 1, (1 << 42) - 1, (1 << 49) - 1, (1 << 56) - 1, (1 << 63) - 1:
				opcode = 255
			default:
				opcode = OPCODE_BASE + LINE_RANGE*PC_RANGE - 1 // 249
			}
		}
	}
	if opcode < OPCODE_BASE || opcode > 255 {
		panic(fmt.Sprintf("produced invalid special opcode %d", opcode))
	}

	// Subtract from deltaPC and deltaLC the amounts that the opcode will add.
	deltaPC -= uint64((opcode - OPCODE_BASE) / LINE_RANGE)
	deltaLC -= int64((opcode-OPCODE_BASE)%LINE_RANGE + LINE_BASE)

	// Encode deltaPC.
	if deltaPC != 0 {
		if deltaPC <= PC_RANGE {
			// Adjust the opcode so that we can use the 1-byte DW_LNS_const_add_pc
			// instruction.
			opcode -= LINE_RANGE * int64(PC_RANGE-deltaPC)
			if opcode < OPCODE_BASE {
				panic(fmt.Sprintf("produced invalid special opcode %d", opcode))
			}
			s.AddUint8(dwarf.DW_LNS_const_add_pc)
		} else if (1<<14) <= deltaPC && deltaPC < (1<<16) {
			s.AddUint8(dwarf.DW_LNS_fixed_advance_pc)
			s.AddUint16(linkctxt.Arch, uint16(deltaPC))
		} else {
			s.AddUint8(dwarf.DW_LNS_advance_pc)
			dwarf.Uleb128put(ctxt, s, int64(deltaPC))
		}
	}

	// Encode deltaLC.
	if deltaLC != 0 {
		s.AddUint8(dwarf.DW_LNS_advance_line)
		dwarf.Sleb128put(ctxt, s, deltaLC)
	}

	// Output the special opcode.
	s.AddUint8(uint8(opcode))
}

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

func importInfoSymbol(ctxt *Link, dsym *sym.Symbol) {
	dsym.Attr |= sym.AttrNotInSymbolTable | sym.AttrReachable
	dsym.Type = sym.SDWARFINFO
	for _, r := range dsym.R {
		if r.Type == objabi.R_DWARFSECREF && r.Sym.Size == 0 {
			if ctxt.BuildMode == BuildModeShared {
				// These type symbols may not be present in BuildModeShared. Skip.
				continue
			}
			n := nameFromDIESym(r.Sym)
			defgotype(ctxt, ctxt.Syms.Lookup("type."+n, 0))
		}
	}
}

// For the specified function, collect symbols corresponding to any
// "abstract" subprogram DIEs referenced. The first case of interest
// is a concrete subprogram DIE, which will refer to its corresponding
// abstract subprogram DIE, and then there can be references from a
// non-abstract subprogram DIE to the abstract subprogram DIEs for any
// functions inlined into this one.
//
// A given abstract subprogram DIE can be referenced in numerous
// places (even within the same DIE), so it is important to make sure
// it gets imported and added to the absfuncs lists only once.

func collectAbstractFunctions(ctxt *Link, fn *sym.Symbol, dsym *sym.Symbol, absfuncs []*sym.Symbol) []*sym.Symbol {

	var newabsfns []*sym.Symbol

	// Walk the relocations on the primary subprogram DIE and look for
	// references to abstract funcs.
	for _, reloc := range dsym.R {
		candsym := reloc.Sym
		if reloc.Type != objabi.R_DWARFSECREF {
			continue
		}
		if !strings.HasPrefix(candsym.Name, dwarf.InfoPrefix) {
			continue
		}
		if !strings.HasSuffix(candsym.Name, dwarf.AbstractFuncSuffix) {
			continue
		}
		if candsym.Attr.OnList() {
			continue
		}
		candsym.Attr |= sym.AttrOnList
		newabsfns = append(newabsfns, candsym)
	}

	// Import any new symbols that have turned up.
	for _, absdsym := range newabsfns {
		importInfoSymbol(ctxt, absdsym)
		absfuncs = append(absfuncs, absdsym)
	}

	return absfuncs
}

func writelines(ctxt *Link, lib *sym.Library, textp []*sym.Symbol, ls *sym.Symbol) (dwinfo *dwarf.DWDie, funcs []*sym.Symbol, absfuncs []*sym.Symbol) {

	var dwarfctxt dwarf.Context = dwctxt{ctxt}

	unitstart := int64(-1)
	headerstart := int64(-1)
	headerend := int64(-1)

	lang := dwarf.DW_LANG_Go

	dwinfo = newdie(ctxt, &dwroot, dwarf.DW_ABRV_COMPUNIT, lib.Pkg, 0)
	newattr(dwinfo, dwarf.DW_AT_language, dwarf.DW_CLS_CONSTANT, int64(lang), 0)
	newattr(dwinfo, dwarf.DW_AT_stmt_list, dwarf.DW_CLS_PTR, ls.Size, ls)
	// OS X linker requires compilation dir or absolute path in comp unit name to output debug info.
	compDir := getCompilationDir()
	// TODO: Make this be the actual compilation directory, not
	// the linker directory. If we move CU construction into the
	// compiler, this should happen naturally.
	newattr(dwinfo, dwarf.DW_AT_comp_dir, dwarf.DW_CLS_STRING, int64(len(compDir)), compDir)
	producerExtra := ctxt.Syms.Lookup(dwarf.CUInfoPrefix+"producer."+lib.Pkg, 0)
	producer := "Go cmd/compile " + objabi.Version
	if len(producerExtra.P) > 0 {
		// We put a semicolon before the flags to clearly
		// separate them from the version, which can be long
		// and have lots of weird things in it in development
		// versions. We promise not to put a semicolon in the
		// version, so it should be safe for readers to scan
		// forward to the semicolon.
		producer += "; " + string(producerExtra.P)
	}
	newattr(dwinfo, dwarf.DW_AT_producer, dwarf.DW_CLS_STRING, int64(len(producer)), producer)

	// Write .debug_line Line Number Program Header (sec 6.2.4)
	// Fields marked with (*) must be changed for 64-bit dwarf
	unitLengthOffset := ls.Size
	ls.AddUint32(ctxt.Arch, 0) // unit_length (*), filled in at end.
	unitstart = ls.Size
	ls.AddUint16(ctxt.Arch, 2) // dwarf version (appendix F)
	headerLengthOffset := ls.Size
	ls.AddUint32(ctxt.Arch, 0) // header_length (*), filled in at end.
	headerstart = ls.Size

	// cpos == unitstart + 4 + 2 + 4
	ls.AddUint8(1)                // minimum_instruction_length
	ls.AddUint8(1)                // default_is_stmt
	ls.AddUint8(LINE_BASE & 0xFF) // line_base
	ls.AddUint8(LINE_RANGE)       // line_range
	ls.AddUint8(OPCODE_BASE)      // opcode_base
	ls.AddUint8(0)                // standard_opcode_lengths[1]
	ls.AddUint8(1)                // standard_opcode_lengths[2]
	ls.AddUint8(1)                // standard_opcode_lengths[3]
	ls.AddUint8(1)                // standard_opcode_lengths[4]
	ls.AddUint8(1)                // standard_opcode_lengths[5]
	ls.AddUint8(0)                // standard_opcode_lengths[6]
	ls.AddUint8(0)                // standard_opcode_lengths[7]
	ls.AddUint8(0)                // standard_opcode_lengths[8]
	ls.AddUint8(1)                // standard_opcode_lengths[9]
	ls.AddUint8(0)                // include_directories  (empty)

	// Create the file table. fileNums maps from global file
	// indexes (created by numberfile) to CU-local indexes.
	fileNums := make(map[int]int)
	for _, s := range textp {
		for _, f := range s.FuncInfo.File {
			if _, ok := fileNums[int(f.Value)]; ok {
				continue
			}
			// File indexes are 1-based.
			fileNums[int(f.Value)] = len(fileNums) + 1
			Addstring(ls, f.Name)
			ls.AddUint8(0)
			ls.AddUint8(0)
			ls.AddUint8(0)
		}

		// Look up the .debug_info sym for the function. We do this
		// now so that we can walk the sym's relocations to discover
		// files that aren't mentioned in S.FuncInfo.File (for
		// example, files mentioned only in an inlined subroutine).
		dsym := ctxt.Syms.Lookup(dwarf.InfoPrefix+s.Name, int(s.Version))
		importInfoSymbol(ctxt, dsym)
		for ri := 0; ri < len(dsym.R); ri++ {
			r := &dsym.R[ri]
			if r.Type != objabi.R_DWARFFILEREF {
				continue
			}
			_, ok := fileNums[int(r.Sym.Value)]
			if !ok {
				fileNums[int(r.Sym.Value)] = len(fileNums) + 1
				Addstring(ls, r.Sym.Name)
				ls.AddUint8(0)
				ls.AddUint8(0)
				ls.AddUint8(0)
			}
		}
	}

	// 4 zeros: the string termination + 3 fields.
	ls.AddUint8(0)
	// terminate file_names.
	headerend = ls.Size

	ls.AddUint8(0) // start extended opcode
	dwarf.Uleb128put(dwarfctxt, ls, 1+int64(ctxt.Arch.PtrSize))
	ls.AddUint8(dwarf.DW_LNE_set_address)

	s := textp[0]
	pc := s.Value
	line := 1
	file := 1
	ls.AddAddr(ctxt.Arch, s)

	var pcfile Pciter
	var pcline Pciter
	for _, s := range textp {
		dsym := ctxt.Syms.Lookup(dwarf.InfoPrefix+s.Name, int(s.Version))
		funcs = append(funcs, dsym)
		absfuncs = collectAbstractFunctions(ctxt, s, dsym, absfuncs)

		finddebugruntimepath(s)

		pciterinit(ctxt, &pcfile, &s.FuncInfo.Pcfile)
		pciterinit(ctxt, &pcline, &s.FuncInfo.Pcline)
		epc := pc
		for pcfile.done == 0 && pcline.done == 0 {
			if epc-s.Value >= int64(pcfile.nextpc) {
				pciternext(&pcfile)
				continue
			}

			if epc-s.Value >= int64(pcline.nextpc) {
				pciternext(&pcline)
				continue
			}

			if int32(file) != pcfile.value {
				ls.AddUint8(dwarf.DW_LNS_set_file)
				idx, ok := fileNums[int(pcfile.value)]
				if !ok {
					Exitf("pcln table file missing from DWARF line table")
				}
				dwarf.Uleb128put(dwarfctxt, ls, int64(idx))
				file = int(pcfile.value)
			}

			putpclcdelta(ctxt, dwarfctxt, ls, uint64(s.Value+int64(pcline.pc)-pc), int64(pcline.value)-int64(line))

			pc = s.Value + int64(pcline.pc)
			line = int(pcline.value)
			if pcfile.nextpc < pcline.nextpc {
				epc = int64(pcfile.nextpc)
			} else {
				epc = int64(pcline.nextpc)
			}
			epc += s.Value
		}
	}

	ls.AddUint8(0) // start extended opcode
	dwarf.Uleb128put(dwarfctxt, ls, 1)
	ls.AddUint8(dwarf.DW_LNE_end_sequence)

	ls.SetUint32(ctxt.Arch, unitLengthOffset, uint32(ls.Size-unitstart))
	ls.SetUint32(ctxt.Arch, headerLengthOffset, uint32(headerend-headerstart))

	// Apply any R_DWARFFILEREF relocations, since we now know the
	// line table file indices for this compilation unit. Note that
	// this loop visits only subprogram DIEs: if the compiler is
	// changed to generate DW_AT_decl_file attributes for other
	// DIE flavors (ex: variables) then those DIEs would need to
	// be included below.
	missing := make(map[int]interface{})
	for fidx := 0; fidx < len(funcs); fidx++ {
		f := funcs[fidx]
		for ri := 0; ri < len(f.R); ri++ {
			r := &f.R[ri]
			if r.Type != objabi.R_DWARFFILEREF {
				continue
			}
			// Mark relocation as applied (signal to relocsym)
			r.Done = true
			idx, ok := fileNums[int(r.Sym.Value)]
			if ok {
				if int(int32(idx)) != idx {
					Errorf(f, "bad R_DWARFFILEREF relocation: file index overflow")
				}
				if r.Siz != 4 {
					Errorf(f, "bad R_DWARFFILEREF relocation: has size %d, expected 4", r.Siz)
				}
				if r.Off < 0 || r.Off+4 > int32(len(f.P)) {
					Errorf(f, "bad R_DWARFFILEREF relocation offset %d + 4 would write past length %d", r.Off, len(s.P))
					continue
				}
				ctxt.Arch.ByteOrder.PutUint32(f.P[r.Off:r.Off+4], uint32(idx))
			} else {
				_, found := missing[int(r.Sym.Value)]
				if !found {
					Errorf(f, "R_DWARFFILEREF relocation file missing: %v idx %d", r.Sym, r.Sym.Value)
					missing[int(r.Sym.Value)] = nil
				}
			}
		}
	}

	return dwinfo, funcs, absfuncs
}

// writepcranges generates the DW_AT_ranges table for compilation unit cu.
func writepcranges(ctxt *Link, cu *dwarf.DWDie, base *sym.Symbol, pcs []dwarf.Range, ranges *sym.Symbol) {
	var dwarfctxt dwarf.Context = dwctxt{ctxt}

	// Create PC ranges for this CU.
	newattr(cu, dwarf.DW_AT_ranges, dwarf.DW_CLS_PTR, ranges.Size, ranges)
	newattr(cu, dwarf.DW_AT_low_pc, dwarf.DW_CLS_ADDRESS, base.Value, base)
	dwarf.PutRanges(dwarfctxt, ranges, nil, pcs)
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

func writeframes(ctxt *Link, syms []*sym.Symbol) []*sym.Symbol {
	var dwarfctxt dwarf.Context = dwctxt{ctxt}
	fs := ctxt.Syms.Lookup(".debug_frame", 0)
	fs.Type = sym.SDWARFSECT
	syms = append(syms, fs)

	// Emit the CIE, Section 6.4.1
	cieReserve := uint32(16)
	if haslinkregister(ctxt) {
		cieReserve = 32
	}
	fs.AddUint32(ctxt.Arch, cieReserve)                        // initial length, must be multiple of thearch.ptrsize
	fs.AddUint32(ctxt.Arch, 0xffffffff)                        // cid.
	fs.AddUint8(3)                                             // dwarf version (appendix F)
	fs.AddUint8(0)                                             // augmentation ""
	dwarf.Uleb128put(dwarfctxt, fs, 1)                         // code_alignment_factor
	dwarf.Sleb128put(dwarfctxt, fs, dataAlignmentFactor)       // all CFI offset calculations include multiplication with this factor
	dwarf.Uleb128put(dwarfctxt, fs, int64(Thearch.Dwarfreglr)) // return_address_register

	fs.AddUint8(dwarf.DW_CFA_def_cfa)                          // Set the current frame address..
	dwarf.Uleb128put(dwarfctxt, fs, int64(Thearch.Dwarfregsp)) // ...to use the value in the platform's SP register (defined in l.go)...
	if haslinkregister(ctxt) {
		dwarf.Uleb128put(dwarfctxt, fs, int64(0)) // ...plus a 0 offset.

		fs.AddUint8(dwarf.DW_CFA_same_value) // The platform's link register is unchanged during the prologue.
		dwarf.Uleb128put(dwarfctxt, fs, int64(Thearch.Dwarfreglr))

		fs.AddUint8(dwarf.DW_CFA_val_offset)                       // The previous value...
		dwarf.Uleb128put(dwarfctxt, fs, int64(Thearch.Dwarfregsp)) // ...of the platform's SP register...
		dwarf.Uleb128put(dwarfctxt, fs, int64(0))                  // ...is CFA+0.
	} else {
		dwarf.Uleb128put(dwarfctxt, fs, int64(ctxt.Arch.PtrSize)) // ...plus the word size (because the call instruction implicitly adds one word to the frame).

		fs.AddUint8(dwarf.DW_CFA_offset_extended)                                      // The previous value...
		dwarf.Uleb128put(dwarfctxt, fs, int64(Thearch.Dwarfreglr))                     // ...of the return address...
		dwarf.Uleb128put(dwarfctxt, fs, int64(-ctxt.Arch.PtrSize)/dataAlignmentFactor) // ...is saved at [CFA - (PtrSize/4)].
	}

	// 4 is to exclude the length field.
	pad := int64(cieReserve) + 4 - fs.Size

	if pad < 0 {
		Exitf("dwarf: cieReserve too small by %d bytes.", -pad)
	}

	fs.AddBytes(zeros[:pad])

	var deltaBuf []byte
	var pcsp Pciter
	for _, s := range ctxt.Textp {
		if s.FuncInfo == nil {
			continue
		}

		// Emit a FDE, Section 6.4.1.
		// First build the section contents into a byte buffer.
		deltaBuf = deltaBuf[:0]
		for pciterinit(ctxt, &pcsp, &s.FuncInfo.Pcsp); pcsp.done == 0; pciternext(&pcsp) {
			nextpc := pcsp.nextpc

			// pciterinit goes up to the end of the function,
			// but DWARF expects us to stop just before the end.
			if int64(nextpc) == s.Size {
				nextpc--
				if nextpc < pcsp.pc {
					continue
				}
			}

			if haslinkregister(ctxt) {
				// TODO(bryanpkc): This is imprecise. In general, the instruction
				// that stores the return address to the stack frame is not the
				// same one that allocates the frame.
				if pcsp.value > 0 {
					// The return address is preserved at (CFA-frame_size)
					// after a stack frame has been allocated.
					deltaBuf = append(deltaBuf, dwarf.DW_CFA_offset_extended_sf)
					deltaBuf = dwarf.AppendUleb128(deltaBuf, uint64(Thearch.Dwarfreglr))
					deltaBuf = dwarf.AppendSleb128(deltaBuf, -int64(pcsp.value)/dataAlignmentFactor)
				} else {
					// The return address is restored into the link register
					// when a stack frame has been de-allocated.
					deltaBuf = append(deltaBuf, dwarf.DW_CFA_same_value)
					deltaBuf = dwarf.AppendUleb128(deltaBuf, uint64(Thearch.Dwarfreglr))
				}
				deltaBuf = appendPCDeltaCFA(ctxt.Arch, deltaBuf, int64(nextpc)-int64(pcsp.pc), int64(pcsp.value))
			} else {
				deltaBuf = appendPCDeltaCFA(ctxt.Arch, deltaBuf, int64(nextpc)-int64(pcsp.pc), int64(ctxt.Arch.PtrSize)+int64(pcsp.value))
			}
		}
		pad := int(Rnd(int64(len(deltaBuf)), int64(ctxt.Arch.PtrSize))) - len(deltaBuf)
		deltaBuf = append(deltaBuf, zeros[:pad]...)

		// Emit the FDE header, Section 6.4.1.
		//	4 bytes: length, must be multiple of thearch.ptrsize
		//	4 bytes: Pointer to the CIE above, at offset 0
		//	ptrsize: initial location
		//	ptrsize: address range
		fs.AddUint32(ctxt.Arch, uint32(4+2*ctxt.Arch.PtrSize+len(deltaBuf))) // length (excludes itself)
		if ctxt.LinkMode == LinkExternal {
			adddwarfref(ctxt, fs, fs, 4)
		} else {
			fs.AddUint32(ctxt.Arch, 0) // CIE offset
		}
		fs.AddAddr(ctxt.Arch, s)
		fs.AddUintXX(ctxt.Arch, uint64(s.Size), ctxt.Arch.PtrSize) // address range
		fs.AddBytes(deltaBuf)
	}
	return syms
}

func writeranges(ctxt *Link, syms []*sym.Symbol) []*sym.Symbol {
	for _, s := range ctxt.Textp {
		rangeSym := ctxt.Syms.ROLookup(dwarf.RangePrefix+s.Name, int(s.Version))
		if rangeSym == nil || rangeSym.Size == 0 {
			continue
		}
		rangeSym.Attr |= sym.AttrReachable | sym.AttrNotInSymbolTable
		rangeSym.Type = sym.SDWARFRANGE
		syms = append(syms, rangeSym)
	}
	return syms
}

/*
 *  Walk DWarfDebugInfoEntries, and emit .debug_info
 */
const (
	COMPUNITHEADERSIZE = 4 + 2 + 4 + 1
)

func writeinfo(ctxt *Link, syms []*sym.Symbol, units []*compilationUnit, abbrevsym *sym.Symbol) []*sym.Symbol {
	infosec := ctxt.Syms.Lookup(".debug_info", 0)
	infosec.Type = sym.SDWARFINFO
	infosec.Attr |= sym.AttrReachable
	syms = append(syms, infosec)

	var dwarfctxt dwarf.Context = dwctxt{ctxt}

	// Re-index per-package information by its CU die.
	unitByDIE := make(map[*dwarf.DWDie]*compilationUnit)
	for _, u := range units {
		unitByDIE[u.dwinfo] = u
	}

	for compunit := dwroot.Child; compunit != nil; compunit = compunit.Link {
		s := dtolsym(compunit.Sym)
		u := unitByDIE[compunit]

		// Write .debug_info Compilation Unit Header (sec 7.5.1)
		// Fields marked with (*) must be changed for 64-bit dwarf
		// This must match COMPUNITHEADERSIZE above.
		s.AddUint32(ctxt.Arch, 0) // unit_length (*), will be filled in later.
		s.AddUint16(ctxt.Arch, 4) // dwarf version (appendix F)

		// debug_abbrev_offset (*)
		adddwarfref(ctxt, s, abbrevsym, 4)

		s.AddUint8(uint8(ctxt.Arch.PtrSize)) // address_size

		dwarf.Uleb128put(dwarfctxt, s, int64(compunit.Abbrev))
		dwarf.PutAttrs(dwarfctxt, s, compunit.Abbrev, compunit.Attr)

		cu := []*sym.Symbol{s}
		cu = append(cu, u.absFnDIEs...)
		cu = append(cu, u.funcDIEs...)
		if u.consts != nil {
			cu = append(cu, u.consts)
		}
		cu = putdies(ctxt, dwarfctxt, cu, compunit.Child)
		var cusize int64
		for _, child := range cu {
			cusize += child.Size
		}
		cusize -= 4 // exclude the length field.
		s.SetUint32(ctxt.Arch, 0, uint32(cusize))
		// Leave a breadcrumb for writepub. This does not
		// appear in the DWARF output.
		newattr(compunit, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, cusize, 0)
		syms = append(syms, cu...)
	}
	return syms
}

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

func writepub(ctxt *Link, sname string, ispub func(*dwarf.DWDie) bool, syms []*sym.Symbol) []*sym.Symbol {
	s := ctxt.Syms.Lookup(sname, 0)
	s.Type = sym.SDWARFSECT
	syms = append(syms, s)

	for compunit := dwroot.Child; compunit != nil; compunit = compunit.Link {
		sectionstart := s.Size
		culength := uint32(getattr(compunit, dwarf.DW_AT_byte_size).Value) + 4

		// Write .debug_pubnames/types	Header (sec 6.1.1)
		s.AddUint32(ctxt.Arch, 0)                      // unit_length (*), will be filled in later.
		s.AddUint16(ctxt.Arch, 2)                      // dwarf version (appendix F)
		adddwarfref(ctxt, s, dtolsym(compunit.Sym), 4) // debug_info_offset (of the Comp unit Header)
		s.AddUint32(ctxt.Arch, culength)               // debug_info_length

		for die := compunit.Child; die != nil; die = die.Link {
			if !ispub(die) {
				continue
			}
			dwa := getattr(die, dwarf.DW_AT_name)
			name := dwa.Data.(string)
			if die.Sym == nil {
				fmt.Println("Missing sym for ", name)
			}
			adddwarfref(ctxt, s, dtolsym(die.Sym), 4)
			Addstring(s, name)
		}

		s.AddUint32(ctxt.Arch, 0)

		s.SetUint32(ctxt.Arch, sectionstart, uint32(s.Size-sectionstart)-4) // exclude the length field.
	}

	return syms
}

func writegdbscript(ctxt *Link, syms []*sym.Symbol) []*sym.Symbol {
	if ctxt.LinkMode == LinkExternal && ctxt.HeadType == objabi.Hwindows && ctxt.BuildMode == BuildModeCArchive {
		// gcc on Windows places .debug_gdb_scripts in the wrong location, which
		// causes the program not to run. See https://golang.org/issue/20183
		// Non c-archives can avoid this issue via a linker script
		// (see fix near writeGDBLinkerScript).
		// c-archive users would need to specify the linker script manually.
		// For UX it's better not to deal with this.
		return syms
	}

	if gdbscript != "" {
		s := ctxt.Syms.Lookup(".debug_gdb_scripts", 0)
		s.Type = sym.SDWARFSECT
		syms = append(syms, s)
		s.AddUint8(1) // magic 1 byte?
		Addstring(s, gdbscript)
	}

	return syms
}

var prototypedies map[string]*dwarf.DWDie

/*
 * This is the main entry point for generating dwarf.  After emitting
 * the mandatory debug_abbrev section, it calls writelines() to set up
 * the per-compilation unit part of the DIE tree, while simultaneously
 * emitting the debug_line section.  When the final tree contains
 * forward references, it will write the debug_info section in 2
 * passes.
 *
 */
func dwarfgeneratedebugsyms(ctxt *Link) {
	if *FlagW { // disable dwarf
		return
	}
	if *FlagS && ctxt.HeadType != objabi.Hdarwin {
		return
	}
	if ctxt.HeadType == objabi.Hplan9 {
		return
	}

	if ctxt.LinkMode == LinkExternal {
		switch {
		case ctxt.IsELF:
		case ctxt.HeadType == objabi.Hdarwin:
		case ctxt.HeadType == objabi.Hwindows:
		default:
			return
		}
	}

	if ctxt.Debugvlog != 0 {
		ctxt.Logf("%5.2f dwarf\n", Cputime())
	}

	// Forctxt.Diagnostic messages.
	newattr(&dwtypes, dwarf.DW_AT_name, dwarf.DW_CLS_STRING, int64(len("dwtypes")), "dwtypes")

	// Some types that must exist to define other ones.
	newdie(ctxt, &dwtypes, dwarf.DW_ABRV_NULLTYPE, "<unspecified>", 0)

	newdie(ctxt, &dwtypes, dwarf.DW_ABRV_NULLTYPE, "void", 0)
	newdie(ctxt, &dwtypes, dwarf.DW_ABRV_BARE_PTRTYPE, "unsafe.Pointer", 0)

	die := newdie(ctxt, &dwtypes, dwarf.DW_ABRV_BASETYPE, "uintptr", 0) // needed for array size
	newattr(die, dwarf.DW_AT_encoding, dwarf.DW_CLS_CONSTANT, dwarf.DW_ATE_unsigned, 0)
	newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, int64(ctxt.Arch.PtrSize), 0)
	newattr(die, dwarf.DW_AT_go_kind, dwarf.DW_CLS_CONSTANT, objabi.KindUintptr, 0)

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
		defgotype(ctxt, lookupOrDiag(ctxt, typ))
	}

	genasmsym(ctxt, defdwsymb)

	abbrev := writeabbrev(ctxt)
	syms := []*sym.Symbol{abbrev}

	units := getCompilationUnits(ctxt)

	// Write per-package line and range tables and start their CU DIEs.
	debugLine := ctxt.Syms.Lookup(".debug_line", 0)
	debugLine.Type = sym.SDWARFSECT
	debugRanges := ctxt.Syms.Lookup(".debug_ranges", 0)
	debugRanges.Type = sym.SDWARFRANGE
	debugRanges.Attr |= sym.AttrReachable
	syms = append(syms, debugLine)
	for _, u := range units {
		u.dwinfo, u.funcDIEs, u.absFnDIEs = writelines(ctxt, u.lib, u.lib.Textp, debugLine)
		writepcranges(ctxt, u.dwinfo, u.lib.Textp[0], u.pcs, debugRanges)
	}

	synthesizestringtypes(ctxt, dwtypes.Child)
	synthesizeslicetypes(ctxt, dwtypes.Child)
	synthesizemaptypes(ctxt, dwtypes.Child)
	synthesizechantypes(ctxt, dwtypes.Child)

	// newdie adds DIEs to the *beginning* of the parent's DIE list.
	// Now that we're done creating DIEs, reverse the trees so DIEs
	// appear in the order they were created.
	reversetree(&dwroot.Child)
	reversetree(&dwtypes.Child)
	reversetree(&dwglobals.Child)

	movetomodule(&dwtypes)
	movetomodule(&dwglobals)

	// Need to reorder symbols so sym.SDWARFINFO is after all sym.SDWARFSECT
	// (but we need to generate dies before writepub)
	infosyms := writeinfo(ctxt, nil, units, abbrev)

	syms = writeframes(ctxt, syms)
	syms = writepub(ctxt, ".debug_pubnames", ispubname, syms)
	syms = writepub(ctxt, ".debug_pubtypes", ispubtype, syms)
	syms = writegdbscript(ctxt, syms)
	// Now we're done writing SDWARFSECT symbols, so we can write
	// other SDWARF* symbols.
	syms = append(syms, infosyms...)
	syms = collectlocs(ctxt, syms, units)
	syms = append(syms, debugRanges)
	syms = writeranges(ctxt, syms)
	dwarfp = syms
}

func collectlocs(ctxt *Link, syms []*sym.Symbol, units []*compilationUnit) []*sym.Symbol {
	empty := true
	for _, u := range units {
		for _, fn := range u.funcDIEs {
			for _, reloc := range fn.R {
				if reloc.Type == objabi.R_DWARFSECREF && strings.HasPrefix(reloc.Sym.Name, dwarf.LocPrefix) {
					reloc.Sym.Attr |= sym.AttrReachable | sym.AttrNotInSymbolTable
					syms = append(syms, reloc.Sym)
					empty = false
					// One location list entry per function, but many relocations to it. Don't duplicate.
					break
				}
			}
		}
	}
	// Don't emit .debug_loc if it's empty -- it makes the ARM linker mad.
	if !empty {
		locsym := ctxt.Syms.Lookup(".debug_loc", 0)
		locsym.Type = sym.SDWARFLOC
		locsym.Attr |= sym.AttrReachable
		syms = append(syms, locsym)
	}
	return syms
}

/*
 *  Elf.
 */
func dwarfaddshstrings(ctxt *Link, shstrtab *sym.Symbol) {
	if *FlagW { // disable dwarf
		return
	}

	Addstring(shstrtab, ".debug_abbrev")
	Addstring(shstrtab, ".debug_frame")
	Addstring(shstrtab, ".debug_info")
	Addstring(shstrtab, ".debug_loc")
	Addstring(shstrtab, ".debug_line")
	Addstring(shstrtab, ".debug_pubnames")
	Addstring(shstrtab, ".debug_pubtypes")
	Addstring(shstrtab, ".debug_gdb_scripts")
	Addstring(shstrtab, ".debug_ranges")
	if ctxt.LinkMode == LinkExternal {
		Addstring(shstrtab, elfRelType+".debug_info")
		Addstring(shstrtab, elfRelType+".debug_loc")
		Addstring(shstrtab, elfRelType+".debug_line")
		Addstring(shstrtab, elfRelType+".debug_frame")
		Addstring(shstrtab, elfRelType+".debug_pubnames")
		Addstring(shstrtab, elfRelType+".debug_pubtypes")
		Addstring(shstrtab, elfRelType+".debug_ranges")
	}
}

// Add section symbols for DWARF debug info.  This is called before
// dwarfaddelfheaders.
func dwarfaddelfsectionsyms(ctxt *Link) {
	if *FlagW { // disable dwarf
		return
	}
	if ctxt.LinkMode != LinkExternal {
		return
	}
	s := ctxt.Syms.Lookup(".debug_info", 0)
	putelfsectionsym(ctxt.Out, s, s.Sect.Elfsect.(*ElfShdr).shnum)
	s = ctxt.Syms.Lookup(".debug_abbrev", 0)
	putelfsectionsym(ctxt.Out, s, s.Sect.Elfsect.(*ElfShdr).shnum)
	s = ctxt.Syms.Lookup(".debug_line", 0)
	putelfsectionsym(ctxt.Out, s, s.Sect.Elfsect.(*ElfShdr).shnum)
	s = ctxt.Syms.Lookup(".debug_frame", 0)
	putelfsectionsym(ctxt.Out, s, s.Sect.Elfsect.(*ElfShdr).shnum)
	s = ctxt.Syms.Lookup(".debug_loc", 0)
	if s.Sect != nil {
		putelfsectionsym(ctxt.Out, s, s.Sect.Elfsect.(*ElfShdr).shnum)
	}
	s = ctxt.Syms.Lookup(".debug_ranges", 0)
	if s.Sect != nil {
		putelfsectionsym(ctxt.Out, s, s.Sect.Elfsect.(*ElfShdr).shnum)
	}
}
