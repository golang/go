// Copyright 2010 The Go Authors. All rights reserved.
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
	"cmd/internal/sys"
	"cmd/link/internal/sym"
	"fmt"
	"log"
	"sort"
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
	r.Type = objabi.R_ADDROFF
	r.Add = ofs
}

func (c dwctxt) AddDWARFAddrSectionOffset(s dwarf.Sym, t interface{}, ofs int64) {
	size := 4
	if isDwarf64(c.linkctxt) {
		size = 8
	}

	c.AddSectionOffset(s, size, t, ofs)
	ls := s.(*sym.Symbol)
	ls.R[len(ls.R)-1].Type = objabi.R_DWARFSECREF
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

func isDwarf64(ctxt *Link) bool {
	return ctxt.HeadType == objabi.Haix
}

var gdbscript string

var dwarfp []*sym.Symbol

func writeabbrev(ctxt *Link) *sym.Symbol {
	s := ctxt.Syms.Lookup(".debug_abbrev", 0)
	s.Type = sym.SDWARFSECT
	s.AddBytes(dwarf.GetAbbrev())
	return s
}

var dwtypes dwarf.DWDie

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

// dwarfFuncSym looks up a DWARF metadata symbol for function symbol s.
// If the symbol does not exist, it creates it if create is true,
// or returns nil otherwise.
func dwarfFuncSym(ctxt *Link, s *sym.Symbol, meta string, create bool) *sym.Symbol {
	// All function ABIs use symbol version 0 for the DWARF data.
	//
	// TODO(austin): It may be useful to have DWARF info for ABI
	// wrappers, in which case we may want these versions to
	// align. Better yet, replace these name lookups with a
	// general way to attach metadata to a symbol.
	ver := 0
	if s.IsFileLocal() {
		ver = int(s.Version)
	}
	if create {
		return ctxt.Syms.Lookup(meta+s.Name, ver)
	}
	return ctxt.Syms.ROLookup(meta+s.Name, ver)
}

func dotypedef(ctxt *Link, parent *dwarf.DWDie, name string, def *dwarf.DWDie) *dwarf.DWDie {
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

	return die
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

	var die, typedefdie *dwarf.DWDie
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
		typedefdie = dotypedef(ctxt, &dwtypes, name, die)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		s := decodetypeArrayElem(ctxt.Arch, gotype)
		newrefattr(die, dwarf.DW_AT_type, defgotype(ctxt, s))
		fld := newdie(ctxt, die, dwarf.DW_ABRV_ARRAYRANGE, "range", 0)

		// use actual length not upper bound; correct for 0-length arrays.
		newattr(fld, dwarf.DW_AT_count, dwarf.DW_CLS_CONSTANT, decodetypeArrayLen(ctxt.Arch, gotype), 0)

		newrefattr(fld, dwarf.DW_AT_type, mustFind(ctxt, "uintptr"))

	case objabi.KindChan:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_CHANTYPE, name, 0)
		s := decodetypeChanElem(ctxt.Arch, gotype)
		newrefattr(die, dwarf.DW_AT_go_elem, defgotype(ctxt, s))
		// Save elem type for synthesizechantypes. We could synthesize here
		// but that would change the order of DIEs we output.
		newrefattr(die, dwarf.DW_AT_type, s)

	case objabi.KindFunc:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_FUNCTYPE, name, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		typedefdie = dotypedef(ctxt, &dwtypes, name, die)
		nfields := decodetypeFuncInCount(ctxt.Arch, gotype)
		for i := 0; i < nfields; i++ {
			s := decodetypeFuncInType(ctxt.Arch, gotype, i)
			fld := newdie(ctxt, die, dwarf.DW_ABRV_FUNCTYPEPARAM, s.Name[5:], 0)
			newrefattr(fld, dwarf.DW_AT_type, defgotype(ctxt, s))
		}

		if decodetypeFuncDotdotdot(ctxt.Arch, gotype) {
			newdie(ctxt, die, dwarf.DW_ABRV_DOTDOTDOT, "...", 0)
		}
		nfields = decodetypeFuncOutCount(ctxt.Arch, gotype)
		for i := 0; i < nfields; i++ {
			s := decodetypeFuncOutType(ctxt.Arch, gotype, i)
			fld := newdie(ctxt, die, dwarf.DW_ABRV_FUNCTYPEPARAM, s.Name[5:], 0)
			newrefattr(fld, dwarf.DW_AT_type, defptrto(ctxt, defgotype(ctxt, s)))
		}

	case objabi.KindInterface:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_IFACETYPE, name, 0)
		typedefdie = dotypedef(ctxt, &dwtypes, name, die)
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
		typedefdie = dotypedef(ctxt, &dwtypes, name, die)
		s := decodetypePtrElem(ctxt.Arch, gotype)
		newrefattr(die, dwarf.DW_AT_type, defgotype(ctxt, s))

	case objabi.KindSlice:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_SLICETYPE, name, 0)
		typedefdie = dotypedef(ctxt, &dwtypes, name, die)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		s := decodetypeArrayElem(ctxt.Arch, gotype)
		elem := defgotype(ctxt, s)
		newrefattr(die, dwarf.DW_AT_go_elem, elem)

	case objabi.KindString:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_STRINGTYPE, name, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case objabi.KindStruct:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_STRUCTTYPE, name, 0)
		typedefdie = dotypedef(ctxt, &dwtypes, name, die)
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
	if gotype.Attr.Reachable() {
		newattr(die, dwarf.DW_AT_go_runtime_type, dwarf.DW_CLS_GO_TYPEREF, 0, gotype)
	}

	if _, ok := prototypedies[gotype.Name]; ok {
		prototypedies[gotype.Name] = die
	}

	if typedefdie != nil {
		return typedefdie
	}
	return die
}

func nameFromDIESym(dwtype *sym.Symbol) string {
	return strings.TrimSuffix(dwtype.Name[len(dwarf.InfoPrefix):], "..def")
}

// Find or construct *T given T.
func defptrto(ctxt *Link, dwtype *sym.Symbol) *sym.Symbol {
	ptrname := "*" + nameFromDIESym(dwtype)
	if die := find(ctxt, ptrname); die != nil {
		return die
	}

	pdie := newdie(ctxt, &dwtypes, dwarf.DW_ABRV_PTRTYPE, ptrname, 0)
	newrefattr(pdie, dwarf.DW_AT_type, dwtype)

	// The DWARF info synthesizes pointer types that don't exist at the
	// language level, like *hash<...> and *bucket<...>, and the data
	// pointers of slices. Link to the ones we can find.
	gotype := ctxt.Syms.ROLookup("type."+ptrname, 0)
	if gotype != nil && gotype.Attr.Reachable() {
		newattr(pdie, dwarf.DW_AT_go_runtime_type, dwarf.DW_CLS_GO_TYPEREF, 0, gotype)
	}
	return dtolsym(pdie.Sym)
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

func dwarfDefineGlobal(ctxt *Link, s *sym.Symbol, str string, v int64, gotype *sym.Symbol) {
	lib := s.Lib
	if lib == nil {
		lib = ctxt.LibraryByPkg["runtime"]
	}
	dv := newdie(ctxt, ctxt.compUnitByPackage[lib].dwinfo, dwarf.DW_ABRV_VARIABLE, str, int(s.Version))
	newabslocexprattr(dv, v, s)
	if !s.IsFileLocal() {
		newattr(dv, dwarf.DW_AT_external, dwarf.DW_CLS_FLAG, 1, 0)
	}
	dt := defgotype(ctxt, gotype)
	newrefattr(dv, dwarf.DW_AT_type, dt)
}

// For use with pass.c::genasmsym
func defdwsymb(ctxt *Link, s *sym.Symbol, str string, t SymbolType, v int64, gotype *sym.Symbol) {
	if strings.HasPrefix(str, "go.string.") {
		return
	}
	if strings.HasPrefix(str, "runtime.gcbits.") {
		return
	}

	switch t {
	case DataSym, BSSSym:
		switch s.Type {
		case sym.SDATA, sym.SNOPTRDATA, sym.STYPE, sym.SBSS, sym.SNOPTRBSS, sym.STLSBSS:
			// ok
		case sym.SRODATA:
			if gotype != nil {
				defgotype(ctxt, gotype)
			}
			return
		default:
			return
		}
		if ctxt.LinkMode != LinkExternal && isStaticTemp(s.Name) {
			return
		}
		dwarfDefineGlobal(ctxt, s, str, v, gotype)

	case AutoSym, ParamSym, DeletedAutoSym:
		defgotype(ctxt, gotype)
	}
}

// createUnitLength creates the initial length field with value v and update
// offset of unit_length if needed.
func createUnitLength(ctxt *Link, s *sym.Symbol, v uint64) {
	if isDwarf64(ctxt) {
		s.AddUint32(ctxt.Arch, 0xFFFFFFFF)
	}
	addDwarfAddrField(ctxt, s, v)
}

// addDwarfAddrField adds a DWARF field in DWARF 64bits or 32bits.
func addDwarfAddrField(ctxt *Link, s *sym.Symbol, v uint64) {
	if isDwarf64(ctxt) {
		s.AddUint(ctxt.Arch, v)
	} else {
		s.AddUint32(ctxt.Arch, uint32(v))
	}
}

// addDwarfAddrRef adds a DWARF pointer in DWARF 64bits or 32bits.
func addDwarfAddrRef(ctxt *Link, s *sym.Symbol, t *sym.Symbol) {
	if isDwarf64(ctxt) {
		adddwarfref(ctxt, s, t, 8)
	} else {
		adddwarfref(ctxt, s, t, 4)
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
	rangeSyms []*sym.Symbol // symbols for debug_range
}

// calcCompUnitRanges calculates the PC ranges of the compilation units.
func calcCompUnitRanges(ctxt *Link) {
	var prevUnit *compilationUnit
	for _, s := range ctxt.Textp {
		if s.FuncInfo == nil {
			continue
		}
		unit := ctxt.compUnitByPackage[s.Lib]

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
}

func movetomodule(ctxt *Link, parent *dwarf.DWDie) {
	runtimelib := ctxt.LibraryByPkg["runtime"]
	die := ctxt.compUnitByPackage[runtimelib].dwinfo.Child
	if die == nil {
		ctxt.compUnitByPackage[runtimelib].dwinfo.Child = parent.Child
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
	OPCODE_BASE = 11
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
	deltaLC -= (opcode-OPCODE_BASE)%LINE_RANGE + LINE_BASE

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
	for i := range dsym.R {
		r := &dsym.R[i] // Copying sym.Reloc has measurable impact on performance
		if r.Type == objabi.R_DWARFSECREF && r.Sym.Size == 0 {
			n := nameFromDIESym(r.Sym)
			defgotype(ctxt, ctxt.Syms.Lookup("type."+n, 0))
		}
	}
}

func writelines(ctxt *Link, unit *compilationUnit, ls *sym.Symbol) {

	var dwarfctxt dwarf.Context = dwctxt{ctxt}
	is_stmt := uint8(1) // initially = recommended default_is_stmt = 1, tracks is_stmt toggles.

	unitstart := int64(-1)
	headerstart := int64(-1)
	headerend := int64(-1)

	newattr(unit.dwinfo, dwarf.DW_AT_stmt_list, dwarf.DW_CLS_PTR, ls.Size, ls)

	// Write .debug_line Line Number Program Header (sec 6.2.4)
	// Fields marked with (*) must be changed for 64-bit dwarf
	unitLengthOffset := ls.Size
	createUnitLength(ctxt, ls, 0) // unit_length (*), filled in at end
	unitstart = ls.Size
	ls.AddUint16(ctxt.Arch, 2) // dwarf version (appendix F) -- version 3 is incompatible w/ XCode 9.0's dsymutil, latest supported on OSX 10.12 as of 2018-05
	headerLengthOffset := ls.Size
	addDwarfAddrField(ctxt, ls, 0) // header_length (*), filled in at end
	headerstart = ls.Size

	// cpos == unitstart + 4 + 2 + 4
	ls.AddUint8(1)                // minimum_instruction_length
	ls.AddUint8(is_stmt)          // default_is_stmt
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
	ls.AddUint8(0)                // standard_opcode_lengths[10]
	ls.AddUint8(0)                // include_directories  (empty)

	// Create the file table. fileNums maps from global file
	// indexes (created by numberfile) to CU-local indexes.
	fileNums := make(map[int]int)
	for _, s := range unit.lib.Textp { // textp has been dead-code-eliminated already.
		dsym := dwarfFuncSym(ctxt, s, dwarf.InfoPrefix, true)
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
		for ri := 0; ri < len(dsym.R); ri++ {
			r := &dsym.R[ri]
			if r.Type != objabi.R_DWARFFILEREF {
				continue
			}
			// A file that is only mentioned in an inlined subroutine will appear
			// as a R_DWARFFILEREF but not in s.FuncInfo.File
			if _, ok := fileNums[int(r.Sym.Value)]; ok {
				continue
			}
			fileNums[int(r.Sym.Value)] = len(fileNums) + 1
			Addstring(ls, r.Sym.Name)
			ls.AddUint8(0)
			ls.AddUint8(0)
			ls.AddUint8(0)
		}
	}

	// 4 zeros: the string termination + 3 fields.
	ls.AddUint8(0)
	// terminate file_names.
	headerend = ls.Size

	ls.AddUint8(0) // start extended opcode
	dwarf.Uleb128put(dwarfctxt, ls, 1+int64(ctxt.Arch.PtrSize))
	ls.AddUint8(dwarf.DW_LNE_set_address)

	s := unit.lib.Textp[0]
	pc := s.Value
	line := 1
	file := 1
	ls.AddAddr(ctxt.Arch, s)

	var pcfile Pciter
	var pcline Pciter
	var pcstmt Pciter
	for i, s := range unit.lib.Textp {
		finddebugruntimepath(s)

		pciterinit(ctxt, &pcfile, &s.FuncInfo.Pcfile)
		pciterinit(ctxt, &pcline, &s.FuncInfo.Pcline)

		isStmtSym := dwarfFuncSym(ctxt, s, dwarf.IsStmtPrefix, false)
		if isStmtSym != nil && len(isStmtSym.P) > 0 {
			pciterinit(ctxt, &pcstmt, &sym.Pcdata{P: isStmtSym.P})
		} else {
			// Assembly files lack a pcstmt section, we assume that every instruction
			// is a valid statement.
			pcstmt.done = 1
			pcstmt.value = 1
		}

		var thispc uint32
		// TODO this loop looks like it could exit with work remaining.
		for pcfile.done == 0 && pcline.done == 0 {
			// Only changed if it advanced
			if int32(file) != pcfile.value {
				ls.AddUint8(dwarf.DW_LNS_set_file)
				idx, ok := fileNums[int(pcfile.value)]
				if !ok {
					Exitf("pcln table file missing from DWARF line table")
				}
				dwarf.Uleb128put(dwarfctxt, ls, int64(idx))
				file = int(pcfile.value)
			}

			// Only changed if it advanced
			if is_stmt != uint8(pcstmt.value) {
				new_stmt := uint8(pcstmt.value)
				switch new_stmt &^ 1 {
				case obj.PrologueEnd:
					ls.AddUint8(uint8(dwarf.DW_LNS_set_prologue_end))
				case obj.EpilogueBegin:
					// TODO if there is a use for this, add it.
					// Don't forget to increase OPCODE_BASE by 1 and add entry for standard_opcode_lengths[11]
				}
				new_stmt &= 1
				if is_stmt != new_stmt {
					is_stmt = new_stmt
					ls.AddUint8(uint8(dwarf.DW_LNS_negate_stmt))
				}
			}

			// putpcldelta makes a row in the DWARF matrix, always, even if line is unchanged.
			putpclcdelta(ctxt, dwarfctxt, ls, uint64(s.Value+int64(thispc)-pc), int64(pcline.value)-int64(line))

			pc = s.Value + int64(thispc)
			line = int(pcline.value)

			// Take the minimum step forward for the three iterators
			thispc = pcfile.nextpc
			if pcline.nextpc < thispc {
				thispc = pcline.nextpc
			}
			if pcstmt.done == 0 && pcstmt.nextpc < thispc {
				thispc = pcstmt.nextpc
			}

			if pcfile.nextpc == thispc {
				pciternext(&pcfile)
			}
			if pcstmt.done == 0 && pcstmt.nextpc == thispc {
				pciternext(&pcstmt)
			}
			if pcline.nextpc == thispc {
				pciternext(&pcline)
			}
		}
		if is_stmt == 0 && i < len(unit.lib.Textp)-1 {
			// If there is more than one function, ensure default value is established.
			is_stmt = 1
			ls.AddUint8(uint8(dwarf.DW_LNS_negate_stmt))
		}
	}

	ls.AddUint8(0) // start extended opcode
	dwarf.Uleb128put(dwarfctxt, ls, 1)
	ls.AddUint8(dwarf.DW_LNE_end_sequence)

	if ctxt.HeadType == objabi.Haix {
		saveDwsectCUSize(".debug_line", unit.lib.String(), uint64(ls.Size-unitLengthOffset))
	}
	if isDwarf64(ctxt) {
		ls.SetUint(ctxt.Arch, unitLengthOffset+4, uint64(ls.Size-unitstart)) // +4 because of 0xFFFFFFFF
		ls.SetUint(ctxt.Arch, headerLengthOffset, uint64(headerend-headerstart))
	} else {
		ls.SetUint32(ctxt.Arch, unitLengthOffset, uint32(ls.Size-unitstart))
		ls.SetUint32(ctxt.Arch, headerLengthOffset, uint32(headerend-headerstart))
	}

	// Apply any R_DWARFFILEREF relocations, since we now know the
	// line table file indices for this compilation unit. Note that
	// this loop visits only subprogram DIEs: if the compiler is
	// changed to generate DW_AT_decl_file attributes for other
	// DIE flavors (ex: variables) then those DIEs would need to
	// be included below.
	missing := make(map[int]interface{})
	for _, f := range unit.funcDIEs {
		for ri := range f.R {
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
	createUnitLength(ctxt, fs, uint64(cieReserve))             // initial length, must be multiple of thearch.ptrsize
	addDwarfAddrField(ctxt, fs, 0xffffffff)                    // cid.
	fs.AddUint8(3)                                             // dwarf version (appendix F)
	fs.AddUint8(0)                                             // augmentation ""
	dwarf.Uleb128put(dwarfctxt, fs, 1)                         // code_alignment_factor
	dwarf.Sleb128put(dwarfctxt, fs, dataAlignmentFactor)       // all CFI offset calculations include multiplication with this factor
	dwarf.Uleb128put(dwarfctxt, fs, int64(thearch.Dwarfreglr)) // return_address_register

	fs.AddUint8(dwarf.DW_CFA_def_cfa)                          // Set the current frame address..
	dwarf.Uleb128put(dwarfctxt, fs, int64(thearch.Dwarfregsp)) // ...to use the value in the platform's SP register (defined in l.go)...
	if haslinkregister(ctxt) {
		dwarf.Uleb128put(dwarfctxt, fs, int64(0)) // ...plus a 0 offset.

		fs.AddUint8(dwarf.DW_CFA_same_value) // The platform's link register is unchanged during the prologue.
		dwarf.Uleb128put(dwarfctxt, fs, int64(thearch.Dwarfreglr))

		fs.AddUint8(dwarf.DW_CFA_val_offset)                       // The previous value...
		dwarf.Uleb128put(dwarfctxt, fs, int64(thearch.Dwarfregsp)) // ...of the platform's SP register...
		dwarf.Uleb128put(dwarfctxt, fs, int64(0))                  // ...is CFA+0.
	} else {
		dwarf.Uleb128put(dwarfctxt, fs, int64(ctxt.Arch.PtrSize)) // ...plus the word size (because the call instruction implicitly adds one word to the frame).

		fs.AddUint8(dwarf.DW_CFA_offset_extended)                                      // The previous value...
		dwarf.Uleb128put(dwarfctxt, fs, int64(thearch.Dwarfreglr))                     // ...of the return address...
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
					deltaBuf = dwarf.AppendUleb128(deltaBuf, uint64(thearch.Dwarfreglr))
					deltaBuf = dwarf.AppendSleb128(deltaBuf, -int64(pcsp.value)/dataAlignmentFactor)
				} else {
					// The return address is restored into the link register
					// when a stack frame has been de-allocated.
					deltaBuf = append(deltaBuf, dwarf.DW_CFA_same_value)
					deltaBuf = dwarf.AppendUleb128(deltaBuf, uint64(thearch.Dwarfreglr))
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
			addDwarfAddrRef(ctxt, fs, fs)
		} else {
			addDwarfAddrField(ctxt, fs, 0) // CIE offset
		}
		fs.AddAddr(ctxt.Arch, s)
		fs.AddUintXX(ctxt.Arch, uint64(s.Size), ctxt.Arch.PtrSize) // address range
		fs.AddBytes(deltaBuf)
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

	for _, u := range units {
		compunit := u.dwinfo
		s := dtolsym(compunit.Sym)

		if len(u.lib.Textp) == 0 && u.dwinfo.Child == nil {
			continue
		}

		// Write .debug_info Compilation Unit Header (sec 7.5.1)
		// Fields marked with (*) must be changed for 64-bit dwarf
		// This must match COMPUNITHEADERSIZE above.
		createUnitLength(ctxt, s, 0) // unit_length (*), will be filled in later.
		s.AddUint16(ctxt.Arch, 4)    // dwarf version (appendix F)

		// debug_abbrev_offset (*)
		addDwarfAddrRef(ctxt, s, abbrevsym)

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
		// Save size for AIX symbol table.
		if ctxt.HeadType == objabi.Haix {
			saveDwsectCUSize(".debug_info", getPkgFromCUSym(s), uint64(cusize))
		}
		if isDwarf64(ctxt) {
			cusize -= 12                            // exclude the length field.
			s.SetUint(ctxt.Arch, 4, uint64(cusize)) // 4 because of 0XFFFFFFFF
		} else {
			cusize -= 4 // exclude the length field.
			s.SetUint32(ctxt.Arch, 0, uint32(cusize))
		}
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

	for _, u := range ctxt.compUnits {
		if len(u.lib.Textp) == 0 && u.dwinfo.Child == nil {
			continue
		}
		compunit := u.dwinfo
		sectionstart := s.Size
		culength := uint32(getattr(compunit, dwarf.DW_AT_byte_size).Value) + 4

		// Write .debug_pubnames/types	Header (sec 6.1.1)
		createUnitLength(ctxt, s, 0)                    // unit_length (*), will be filled in later.
		s.AddUint16(ctxt.Arch, 2)                       // dwarf version (appendix F)
		addDwarfAddrRef(ctxt, s, dtolsym(compunit.Sym)) // debug_info_offset (of the Comp unit Header)
		addDwarfAddrField(ctxt, s, uint64(culength))    // debug_info_length

		for die := compunit.Child; die != nil; die = die.Link {
			if !ispub(die) {
				continue
			}
			dwa := getattr(die, dwarf.DW_AT_name)
			name := dwa.Data.(string)
			if die.Sym == nil {
				fmt.Println("Missing sym for ", name)
			}
			addDwarfAddrRef(ctxt, s, dtolsym(die.Sym))
			Addstring(s, name)
		}

		addDwarfAddrField(ctxt, s, 0) // Null offset

		// On AIX, save the current size of this compilation unit.
		if ctxt.HeadType == objabi.Haix {
			saveDwsectCUSize(sname, getPkgFromCUSym(dtolsym(compunit.Sym)), uint64(s.Size-sectionstart))
		}
		if isDwarf64(ctxt) {
			s.SetUint(ctxt.Arch, sectionstart+4, uint64(s.Size-sectionstart)-12) // exclude the length field.
		} else {
			s.SetUint32(ctxt.Arch, sectionstart, uint32(s.Size-sectionstart)-4) // exclude the length field.
		}
	}

	return syms
}

func writegdbscript(ctxt *Link, syms []*sym.Symbol) []*sym.Symbol {
	// TODO (aix): make it available
	if ctxt.HeadType == objabi.Haix {
		return syms
	}
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
		default:
			return false
		}
	}

	return true
}

// dwarfGenerateDebugInfo generated debug info entries for all types,
// variables and functions in the program.
// Along with dwarfGenerateDebugSyms they are the two main entry points into
// dwarf generation: dwarfGenerateDebugInfo does all the work that should be
// done before symbol names are mangled while dwarfgeneratedebugsyms does
// all the work that can only be done after addresses have been assigned to
// text symbols.
func dwarfGenerateDebugInfo(ctxt *Link) {
	if !dwarfEnabled(ctxt) {
		return
	}

	ctxt.compUnitByPackage = make(map[*sym.Library]*compilationUnit)

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
	newattr(die, dwarf.DW_AT_go_runtime_type, dwarf.DW_CLS_ADDRESS, 0, lookupOrDiag(ctxt, "type.uintptr"))

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

	// fake root DIE for compile unit DIEs
	var dwroot dwarf.DWDie

	for _, lib := range ctxt.Library {
		unit := &compilationUnit{lib: lib}
		if s := ctxt.Syms.ROLookup(dwarf.ConstInfoPrefix+lib.Pkg, 0); s != nil {
			importInfoSymbol(ctxt, s)
			unit.consts = s
		}
		ctxt.compUnits = append(ctxt.compUnits, unit)
		ctxt.compUnitByPackage[lib] = unit

		unit.dwinfo = newdie(ctxt, &dwroot, dwarf.DW_ABRV_COMPUNIT, unit.lib.Pkg, 0)
		newattr(unit.dwinfo, dwarf.DW_AT_language, dwarf.DW_CLS_CONSTANT, int64(dwarf.DW_LANG_Go), 0)
		// OS X linker requires compilation dir or absolute path in comp unit name to output debug info.
		compDir := getCompilationDir()
		// TODO: Make this be the actual compilation directory, not
		// the linker directory. If we move CU construction into the
		// compiler, this should happen naturally.
		newattr(unit.dwinfo, dwarf.DW_AT_comp_dir, dwarf.DW_CLS_STRING, int64(len(compDir)), compDir)
		producerExtra := ctxt.Syms.Lookup(dwarf.CUInfoPrefix+"producer."+unit.lib.Pkg, 0)
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
		newattr(unit.dwinfo, dwarf.DW_AT_producer, dwarf.DW_CLS_STRING, int64(len(producer)), producer)

		if len(lib.Textp) == 0 {
			unit.dwinfo.Abbrev = dwarf.DW_ABRV_COMPUNIT_TEXTLESS
		}

		// Scan all functions in this compilation unit, create DIEs for all
		// referenced types, create the file table for debug_line, find all
		// referenced abstract functions.
		// Collect all debug_range symbols in unit.rangeSyms
		for _, s := range lib.Textp { // textp has been dead-code-eliminated already.
			dsym := dwarfFuncSym(ctxt, s, dwarf.InfoPrefix, false)
			dsym.Attr |= sym.AttrNotInSymbolTable | sym.AttrReachable
			dsym.Type = sym.SDWARFINFO
			unit.funcDIEs = append(unit.funcDIEs, dsym)

			rangeSym := dwarfFuncSym(ctxt, s, dwarf.RangePrefix, false)
			if rangeSym != nil && rangeSym.Size > 0 {
				rangeSym.Attr |= sym.AttrReachable | sym.AttrNotInSymbolTable
				rangeSym.Type = sym.SDWARFRANGE
				// LLVM doesn't support base address entries. Strip them out so LLDB and dsymutil don't get confused.
				if ctxt.HeadType == objabi.Hdarwin {
					removeDwarfAddrListBaseAddress(ctxt, dsym, rangeSym, false)
				}
				unit.rangeSyms = append(unit.rangeSyms, rangeSym)
			}

			for ri := 0; ri < len(dsym.R); ri++ {
				r := &dsym.R[ri]
				if r.Type == objabi.R_DWARFSECREF {
					rsym := r.Sym
					if strings.HasPrefix(rsym.Name, dwarf.InfoPrefix) && strings.HasSuffix(rsym.Name, dwarf.AbstractFuncSuffix) && !rsym.Attr.OnList() {
						// abstract function
						rsym.Attr |= sym.AttrOnList
						unit.absFnDIEs = append(unit.absFnDIEs, rsym)
						importInfoSymbol(ctxt, rsym)
					} else if rsym.Size == 0 {
						// a type we do not have a DIE for
						n := nameFromDIESym(rsym)
						defgotype(ctxt, ctxt.Syms.Lookup("type."+n, 0))
					}
				}
			}
		}
	}

	// Create DIEs for global variables and the types they use.
	genasmsym(ctxt, defdwsymb)

	synthesizestringtypes(ctxt, dwtypes.Child)
	synthesizeslicetypes(ctxt, dwtypes.Child)
	synthesizemaptypes(ctxt, dwtypes.Child)
	synthesizechantypes(ctxt, dwtypes.Child)
}

// dwarfGenerateDebugSyms constructs debug_line, debug_frame, debug_loc,
// debug_pubnames and debug_pubtypes. It also writes out the debug_info
// section using symbols generated in dwarfGenerateDebugInfo.
func dwarfGenerateDebugSyms(ctxt *Link) {
	if !dwarfEnabled(ctxt) {
		return
	}

	if ctxt.Debugvlog != 0 {
		ctxt.Logf("%5.2f dwarf\n", Cputime())
	}

	abbrev := writeabbrev(ctxt)
	syms := []*sym.Symbol{abbrev}

	calcCompUnitRanges(ctxt)
	sort.Sort(compilationUnitByStartPC(ctxt.compUnits))

	// Write per-package line and range tables and start their CU DIEs.
	debugLine := ctxt.Syms.Lookup(".debug_line", 0)
	debugLine.Type = sym.SDWARFSECT
	debugRanges := ctxt.Syms.Lookup(".debug_ranges", 0)
	debugRanges.Type = sym.SDWARFRANGE
	debugRanges.Attr |= sym.AttrReachable
	syms = append(syms, debugLine)
	for _, u := range ctxt.compUnits {
		reversetree(&u.dwinfo.Child)
		if u.dwinfo.Abbrev == dwarf.DW_ABRV_COMPUNIT_TEXTLESS {
			continue
		}
		writelines(ctxt, u, debugLine)
		writepcranges(ctxt, u.dwinfo, u.lib.Textp[0], u.pcs, debugRanges)
	}

	// newdie adds DIEs to the *beginning* of the parent's DIE list.
	// Now that we're done creating DIEs, reverse the trees so DIEs
	// appear in the order they were created.
	reversetree(&dwtypes.Child)
	movetomodule(ctxt, &dwtypes)

	// Need to reorder symbols so sym.SDWARFINFO is after all sym.SDWARFSECT
	// (but we need to generate dies before writepub)
	infosyms := writeinfo(ctxt, nil, ctxt.compUnits, abbrev)

	syms = writeframes(ctxt, syms)
	syms = writepub(ctxt, ".debug_pubnames", ispubname, syms)
	syms = writepub(ctxt, ".debug_pubtypes", ispubtype, syms)
	syms = writegdbscript(ctxt, syms)
	// Now we're done writing SDWARFSECT symbols, so we can write
	// other SDWARF* symbols.
	syms = append(syms, infosyms...)
	syms = collectlocs(ctxt, syms, ctxt.compUnits)
	syms = append(syms, debugRanges)
	for _, unit := range ctxt.compUnits {
		syms = append(syms, unit.rangeSyms...)
	}
	dwarfp = syms
}

func collectlocs(ctxt *Link, syms []*sym.Symbol, units []*compilationUnit) []*sym.Symbol {
	empty := true
	for _, u := range units {
		for _, fn := range u.funcDIEs {
			for i := range fn.R {
				reloc := &fn.R[i] // Copying sym.Reloc has measurable impact on performance
				if reloc.Type == objabi.R_DWARFSECREF && strings.HasPrefix(reloc.Sym.Name, dwarf.LocPrefix) {
					reloc.Sym.Attr |= sym.AttrReachable | sym.AttrNotInSymbolTable
					syms = append(syms, reloc.Sym)
					empty = false
					// LLVM doesn't support base address entries. Strip them out so LLDB and dsymutil don't get confused.
					if ctxt.HeadType == objabi.Hdarwin {
						removeDwarfAddrListBaseAddress(ctxt, fn, reloc.Sym, true)
					}
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

// removeDwarfAddrListBaseAddress removes base address selector entries from
// DWARF location lists and range lists.
func removeDwarfAddrListBaseAddress(ctxt *Link, info, list *sym.Symbol, isloclist bool) {
	// The list symbol contains multiple lists, but they're all for the
	// same function, and it's not empty.
	fn := list.R[0].Sym

	// Discard the relocations for the base address entries.
	list.R = list.R[:0]

	// Add relocations for each location entry's start and end addresses,
	// so that the base address entries aren't necessary.
	// We could remove them entirely, but that's more work for a relatively
	// small size win. If dsymutil runs it'll throw them away anyway.

	// relocate adds a CU-relative relocation to fn+addr at offset.
	relocate := func(addr uint64, offset int) {
		list.R = append(list.R, sym.Reloc{
			Off:  int32(offset),
			Siz:  uint8(ctxt.Arch.PtrSize),
			Type: objabi.R_ADDRCUOFF,
			Add:  int64(addr),
			Sym:  fn,
		})
	}

	for i := 0; i < len(list.P); {
		first := readPtr(ctxt, list.P[i:])
		second := readPtr(ctxt, list.P[i+ctxt.Arch.PtrSize:])

		if (first == 0 && second == 0) ||
			first == ^uint64(0) ||
			(ctxt.Arch.PtrSize == 4 && first == uint64(^uint32(0))) {
			// Base address selection entry or end of list. Ignore.
			i += ctxt.Arch.PtrSize * 2
			continue
		}

		relocate(first, i)
		relocate(second, i+ctxt.Arch.PtrSize)

		// Skip past the actual location.
		i += ctxt.Arch.PtrSize * 2
		if isloclist {
			i += 2 + int(ctxt.Arch.ByteOrder.Uint16(list.P[i:]))
		}
	}

	// Rewrite the DIE's relocations to point to the first location entry,
	// not the now-useless base address selection entry.
	for i := range info.R {
		r := &info.R[i]
		if r.Sym != list {
			continue
		}
		r.Add += int64(2 * ctxt.Arch.PtrSize)
	}
}

// Read a pointer-sized uint from the beginning of buf.
func readPtr(ctxt *Link, buf []byte) uint64 {
	switch ctxt.Arch.PtrSize {
	case 4:
		return uint64(ctxt.Arch.ByteOrder.Uint32(buf))
	case 8:
		return ctxt.Arch.ByteOrder.Uint64(buf)
	default:
		panic("unexpected pointer size")
	}
}

/*
 *  Elf.
 */
func dwarfaddshstrings(ctxt *Link, shstrtab *sym.Symbol) {
	if *FlagW { // disable dwarf
		return
	}

	secs := []string{"abbrev", "frame", "info", "loc", "line", "pubnames", "pubtypes", "gdb_scripts", "ranges"}
	for _, sec := range secs {
		Addstring(shstrtab, ".debug_"+sec)
		if ctxt.LinkMode == LinkExternal {
			Addstring(shstrtab, elfRelType+".debug_"+sec)
		} else {
			Addstring(shstrtab, ".zdebug_"+sec)
		}
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

// dwarfcompress compresses the DWARF sections. This must happen after
// relocations are applied. After this, dwarfp will contain a
// different (new) set of symbols, and sections may have been replaced.
func dwarfcompress(ctxt *Link) {
	supported := ctxt.IsELF || ctxt.HeadType == objabi.Hwindows || ctxt.HeadType == objabi.Hdarwin
	if !ctxt.compressDWARF || !supported || ctxt.LinkMode != LinkInternal {
		return
	}

	var start int
	var newDwarfp []*sym.Symbol
	Segdwarf.Sections = Segdwarf.Sections[:0]
	for i, s := range dwarfp {
		// Find the boundaries between sections and compress
		// the whole section once we've found the last of its
		// symbols.
		if i+1 >= len(dwarfp) || s.Sect != dwarfp[i+1].Sect {
			s1 := compressSyms(ctxt, dwarfp[start:i+1])
			if s1 == nil {
				// Compression didn't help.
				newDwarfp = append(newDwarfp, dwarfp[start:i+1]...)
				Segdwarf.Sections = append(Segdwarf.Sections, s.Sect)
			} else {
				compressedSegName := ".zdebug_" + s.Sect.Name[len(".debug_"):]
				sect := addsection(ctxt.Arch, &Segdwarf, compressedSegName, 04)
				sect.Length = uint64(len(s1))
				newSym := ctxt.Syms.Lookup(compressedSegName, 0)
				newSym.P = s1
				newSym.Size = int64(len(s1))
				newSym.Sect = sect
				newDwarfp = append(newDwarfp, newSym)
			}
			start = i + 1
		}
	}
	dwarfp = newDwarfp

	// Re-compute the locations of the compressed DWARF symbols
	// and sections, since the layout of these within the file is
	// based on Section.Vaddr and Symbol.Value.
	pos := Segdwarf.Vaddr
	var prevSect *sym.Section
	for _, s := range dwarfp {
		s.Value = int64(pos)
		if s.Sect != prevSect {
			s.Sect.Vaddr = uint64(s.Value)
			prevSect = s.Sect
		}
		if s.Sub != nil {
			log.Fatalf("%s: unexpected sub-symbols", s)
		}
		pos += uint64(s.Size)
		if ctxt.HeadType == objabi.Hwindows {
			pos = uint64(Rnd(int64(pos), PEFILEALIGN))
		}

	}
	Segdwarf.Length = pos - Segdwarf.Vaddr
}

type compilationUnitByStartPC []*compilationUnit

func (v compilationUnitByStartPC) Len() int      { return len(v) }
func (v compilationUnitByStartPC) Swap(i, j int) { v[i], v[j] = v[j], v[i] }

func (v compilationUnitByStartPC) Less(i, j int) bool {
	switch {
	case len(v[i].lib.Textp) == 0 && len(v[j].lib.Textp) == 0:
		return v[i].lib.Pkg < v[j].lib.Pkg
	case len(v[i].lib.Textp) != 0 && len(v[j].lib.Textp) == 0:
		return true
	case len(v[i].lib.Textp) == 0 && len(v[j].lib.Textp) != 0:
		return false
	default:
		return v[i].lib.Textp[0].Value < v[j].lib.Textp[0].Value
	}
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

// getPkgFromCUSym returns the package name for the compilation unit
// represented by s.
// The prefix dwarf.InfoPrefix+".pkg." needs to be removed in order to get
// the package name.
func getPkgFromCUSym(s *sym.Symbol) string {
	return strings.TrimPrefix(s.Name, dwarf.InfoPrefix+".pkg.")
}
