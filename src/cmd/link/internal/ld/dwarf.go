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
	"fmt"
	"log"
	"os"
	"strings"
)

type dwctxt struct {
	linkctxt *Link
}

func (c dwctxt) PtrSize() int {
	return SysArch.PtrSize
}
func (c dwctxt) AddInt(s dwarf.Sym, size int, i int64) {
	ls := s.(*Symbol)
	adduintxx(c.linkctxt, ls, uint64(i), size)
}
func (c dwctxt) AddBytes(s dwarf.Sym, b []byte) {
	ls := s.(*Symbol)
	Addbytes(ls, b)
}
func (c dwctxt) AddString(s dwarf.Sym, v string) {
	Addstring(s.(*Symbol), v)
}
func (c dwctxt) SymValue(s dwarf.Sym) int64 {
	return s.(*Symbol).Value
}

func (c dwctxt) AddAddress(s dwarf.Sym, data interface{}, value int64) {
	if value != 0 {
		value -= (data.(*Symbol)).Value
	}
	Addaddrplus(c.linkctxt, s.(*Symbol), data.(*Symbol), value)
}

func (c dwctxt) AddSectionOffset(s dwarf.Sym, size int, t interface{}, ofs int64) {
	ls := s.(*Symbol)
	switch size {
	default:
		Errorf(ls, "invalid size %d in adddwarfref\n", size)
		fallthrough
	case SysArch.PtrSize:
		Addaddr(c.linkctxt, ls, t.(*Symbol))
	case 4:
		addaddrplus4(c.linkctxt, ls, t.(*Symbol), 0)
	}
	r := &ls.R[len(ls.R)-1]
	r.Type = objabi.R_DWARFREF
	r.Add = ofs
}

/*
 * Offsets and sizes of the debug_* sections in the cout file.
 */
var abbrevsym *Symbol
var arangessec *Symbol
var framesec *Symbol
var infosec *Symbol
var linesec *Symbol
var rangesec *Symbol

var gdbscript string

var dwarfp []*Symbol

func writeabbrev(ctxt *Link, syms []*Symbol) []*Symbol {
	s := ctxt.Syms.Lookup(".debug_abbrev", 0)
	s.Type = SDWARFSECT
	abbrevsym = s
	Addbytes(s, dwarf.GetAbbrev())
	return append(syms, s)
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

// Every DIE has at least a AT_name attribute (but it will only be
// written out if it is listed in the abbrev).
func newdie(ctxt *Link, parent *dwarf.DWDie, abbrev int, name string, version int) *dwarf.DWDie {
	die := new(dwarf.DWDie)
	die.Abbrev = abbrev
	die.Link = parent.Child
	parent.Child = die

	newattr(die, dwarf.DW_AT_name, dwarf.DW_CLS_STRING, int64(len(name)), name)

	if name != "" && (abbrev <= dwarf.DW_ABRV_VARIABLE || abbrev >= dwarf.DW_ABRV_NULLTYPE) {
		if abbrev != dwarf.DW_ABRV_VARIABLE || version == 0 {
			sym := ctxt.Syms.Lookup(dwarf.InfoPrefix+name, version)
			sym.Attr |= AttrNotInSymbolTable
			sym.Type = SDWARFINFO
			die.Sym = sym
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

func walksymtypedef(ctxt *Link, s *Symbol) *Symbol {
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

func find(ctxt *Link, name string) *Symbol {
	n := append(prefixBuf, name...)
	// The string allocation below is optimized away because it is only used in a map lookup.
	s := ctxt.Syms.ROLookup(string(n), 0)
	prefixBuf = n[:len(dwarf.InfoPrefix)]
	if s != nil && s.Type == SDWARFINFO {
		return s
	}
	return nil
}

func mustFind(ctxt *Link, name string) *Symbol {
	r := find(ctxt, name)
	if r == nil {
		Exitf("dwarf find: cannot find %s", name)
	}
	return r
}

func adddwarfref(ctxt *Link, s *Symbol, t *Symbol, size int) int64 {
	var result int64
	switch size {
	default:
		Errorf(s, "invalid size %d in adddwarfref\n", size)
		fallthrough
	case SysArch.PtrSize:
		result = Addaddr(ctxt, s, t)
	case 4:
		result = addaddrplus4(ctxt, s, t, 0)
	}
	r := &s.R[len(s.R)-1]
	r.Type = objabi.R_DWARFREF
	return result
}

func newrefattr(die *dwarf.DWDie, attr uint16, ref *Symbol) *dwarf.DWAttr {
	if ref == nil {
		return nil
	}
	return newattr(die, attr, dwarf.DW_CLS_REFERENCE, 0, ref)
}

func putdies(linkctxt *Link, ctxt dwarf.Context, syms []*Symbol, die *dwarf.DWDie) []*Symbol {
	for ; die != nil; die = die.Link {
		syms = putdie(linkctxt, ctxt, syms, die)
	}
	Adduint8(linkctxt, syms[len(syms)-1], 0)

	return syms
}

func dtolsym(s dwarf.Sym) *Symbol {
	if s == nil {
		return nil
	}
	return s.(*Symbol)
}

func putdie(linkctxt *Link, ctxt dwarf.Context, syms []*Symbol, die *dwarf.DWDie) []*Symbol {
	s := dtolsym(die.Sym)
	if s == nil {
		s = syms[len(syms)-1]
	} else {
		if s.Attr.OnList() {
			log.Fatalf("symbol %s listed multiple times", s.Name)
		}
		s.Attr |= AttrOnList
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
		var next *dwarf.DWDie = curr.Link
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
	var block [20]byte
	b := append(block[:0], dwarf.DW_OP_plus_uconst)
	b = dwarf.AppendUleb128(b, uint64(offs))
	newattr(die, dwarf.DW_AT_data_member_location, dwarf.DW_CLS_BLOCK, int64(len(b)), b)
}

// GDB doesn't like FORM_addr for AT_location, so emit a
// location expression that evals to a const.
func newabslocexprattr(die *dwarf.DWDie, addr int64, sym *Symbol) {
	newattr(die, dwarf.DW_AT_location, dwarf.DW_CLS_ADDRESS, addr, sym)
	// below
}

// Lookup predefined types
func lookupOrDiag(ctxt *Link, n string) *Symbol {
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

	sym := ctxt.Syms.Lookup(dtolsym(def.Sym).Name+"..def", 0)
	sym.Attr |= AttrNotInSymbolTable
	sym.Type = SDWARFINFO
	def.Sym = sym

	// The typedef entry must be created after the def,
	// so that future lookups will find the typedef instead
	// of the real definition. This hooks the typedef into any
	// circular definition loops, so that gdb can understand them.
	die := newdie(ctxt, parent, dwarf.DW_ABRV_TYPEDECL, name, 0)

	newrefattr(die, dwarf.DW_AT_type, sym)
}

// Define gotype, for composite ones recurse into constituents.
func defgotype(ctxt *Link, gotype *Symbol) *Symbol {
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

	return newtype(ctxt, gotype).Sym.(*Symbol)
}

func newtype(ctxt *Link, gotype *Symbol) *dwarf.DWDie {
	name := gotype.Name[5:] // could also decode from Type.string
	kind := decodetypeKind(gotype)
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
		s := decodetypeArrayElem(gotype)
		newrefattr(die, dwarf.DW_AT_type, defgotype(ctxt, s))
		fld := newdie(ctxt, die, dwarf.DW_ABRV_ARRAYRANGE, "range", 0)

		// use actual length not upper bound; correct for 0-length arrays.
		newattr(fld, dwarf.DW_AT_count, dwarf.DW_CLS_CONSTANT, decodetypeArrayLen(ctxt.Arch, gotype), 0)

		newrefattr(fld, dwarf.DW_AT_type, mustFind(ctxt, "uintptr"))

	case objabi.KindChan:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_CHANTYPE, name, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		s := decodetypeChanElem(gotype)
		newrefattr(die, dwarf.DW_AT_go_elem, defgotype(ctxt, s))
		// Save elem type for synthesizechantypes. We could synthesize here
		// but that would change the order of DIEs we output.
		newrefattr(die, dwarf.DW_AT_type, s)

	case objabi.KindFunc:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_FUNCTYPE, name, 0)
		dotypedef(ctxt, &dwtypes, name, die)
		newrefattr(die, dwarf.DW_AT_type, mustFind(ctxt, "void"))
		nfields := decodetypeFuncInCount(ctxt.Arch, gotype)
		var fld *dwarf.DWDie
		var s *Symbol
		for i := 0; i < nfields; i++ {
			s = decodetypeFuncInType(gotype, i)
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
		var s *Symbol
		if nfields == 0 {
			s = lookupOrDiag(ctxt, "type.runtime.eface")
		} else {
			s = lookupOrDiag(ctxt, "type.runtime.iface")
		}
		newrefattr(die, dwarf.DW_AT_type, defgotype(ctxt, s))

	case objabi.KindMap:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_MAPTYPE, name, 0)
		s := decodetypeMapKey(gotype)
		newrefattr(die, dwarf.DW_AT_go_key, defgotype(ctxt, s))
		s = decodetypeMapValue(gotype)
		newrefattr(die, dwarf.DW_AT_go_elem, defgotype(ctxt, s))
		// Save gotype for use in synthesizemaptypes. We could synthesize here,
		// but that would change the order of the DIEs.
		newrefattr(die, dwarf.DW_AT_type, gotype)

	case objabi.KindPtr:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_PTRTYPE, name, 0)
		dotypedef(ctxt, &dwtypes, name, die)
		s := decodetypePtrElem(gotype)
		newrefattr(die, dwarf.DW_AT_type, defgotype(ctxt, s))

	case objabi.KindSlice:
		die = newdie(ctxt, &dwtypes, dwarf.DW_ABRV_SLICETYPE, name, 0)
		dotypedef(ctxt, &dwtypes, name, die)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		s := decodetypeArrayElem(gotype)
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
			f := decodetypeStructFieldName(gotype, i)
			s := decodetypeStructFieldType(gotype, i)
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

func nameFromDIESym(dwtype *Symbol) string {
	return strings.TrimSuffix(dwtype.Name[len(dwarf.InfoPrefix):], "..def")
}

// Find or construct *T given T.
func defptrto(ctxt *Link, dwtype *Symbol) *Symbol {
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
func substitutetype(structdie *dwarf.DWDie, field string, dwtype *Symbol) {
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
		elem := getattr(die, dwarf.DW_AT_go_elem).Data.(*Symbol)
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

func mkinternaltype(ctxt *Link, abbrev int, typename, keyname, valname string, f func(*dwarf.DWDie)) *Symbol {
	name := mkinternaltypename(typename, keyname, valname)
	symname := dwarf.InfoPrefix + name
	s := ctxt.Syms.ROLookup(symname, 0)
	if s != nil && s.Type == SDWARFINFO {
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
		gotype := getattr(die, dwarf.DW_AT_type).Data.(*Symbol)
		keytype := decodetypeMapKey(gotype)
		valtype := decodetypeMapValue(gotype)
		keysize, valsize := decodetypeSize(ctxt.Arch, keytype), decodetypeSize(ctxt.Arch, valtype)
		keytype, valtype = walksymtypedef(ctxt, defgotype(ctxt, keytype)), walksymtypedef(ctxt, defgotype(ctxt, valtype))

		// compute size info like hashmap.c does.
		indirectKey, indirectVal := false, false
		if keysize > MaxKeySize {
			keysize = int64(SysArch.PtrSize)
			indirectKey = true
		}
		if valsize > MaxValSize {
			valsize = int64(SysArch.PtrSize)
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
			if SysArch.RegSize > SysArch.PtrSize {
				fld = newdie(ctxt, dwhb, dwarf.DW_ABRV_STRUCTFIELD, "pad", 0)
				newrefattr(fld, dwarf.DW_AT_type, mustFind(ctxt, "uintptr"))
				newmemberoffsetattr(fld, BucketSize+BucketSize*(int32(keysize)+int32(valsize))+int32(SysArch.PtrSize))
			}

			newattr(dwhb, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, BucketSize+BucketSize*keysize+BucketSize*valsize+int64(SysArch.RegSize), 0)
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
		elemgotype := getattr(die, dwarf.DW_AT_type).Data.(*Symbol)
		elemsize := decodetypeSize(ctxt.Arch, elemgotype)
		elemname := elemgotype.Name[5:]
		elemtype := walksymtypedef(ctxt, defgotype(ctxt, elemgotype))

		// sudog<T>
		dwss := mkinternaltype(ctxt, dwarf.DW_ABRV_STRUCTTYPE, "sudog", elemname, "", func(dws *dwarf.DWDie) {
			copychildren(ctxt, dws, sudog)
			substitutetype(dws, "elem", elemtype)
			if elemsize > 8 {
				elemsize -= 8
			} else {
				elemsize = 0
			}
			newattr(dws, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, int64(sudogsize)+elemsize, nil)
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
func defdwsymb(ctxt *Link, sym *Symbol, s string, t SymbolType, v int64, gotype *Symbol) {
	if strings.HasPrefix(s, "go.string.") {
		return
	}
	if strings.HasPrefix(s, "runtime.gcbits.") {
		return
	}

	if strings.HasPrefix(s, "type.") && s != "type.*" && !strings.HasPrefix(s, "type..") {
		defgotype(ctxt, sym)
		return
	}

	var dv *dwarf.DWDie

	var dt *Symbol
	switch t {
	default:
		return

	case DataSym, BSSSym:
		dv = newdie(ctxt, &dwglobals, dwarf.DW_ABRV_VARIABLE, s, int(sym.Version))
		newabslocexprattr(dv, v, sym)
		if sym.Version == 0 {
			newattr(dv, dwarf.DW_AT_external, dwarf.DW_CLS_FLAG, 1, 0)
		}
		fallthrough

	case AutoSym, ParamSym:
		dt = defgotype(ctxt, gotype)
	}

	if dv != nil {
		newrefattr(dv, dwarf.DW_AT_type, dt)
	}
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

// If the pcln table contains runtime/runtime.go, use that to set gdbscript path.
func finddebugruntimepath(s *Symbol) {
	if gdbscript != "" {
		return
	}

	for i := range s.FuncInfo.File {
		f := s.FuncInfo.File[i]
		if i := strings.Index(f.Name, "runtime/debug.go"); i >= 0 {
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

func putpclcdelta(linkctxt *Link, ctxt dwarf.Context, s *Symbol, deltaPC uint64, deltaLC int64) {
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
			Adduint8(linkctxt, s, dwarf.DW_LNS_const_add_pc)
		} else if (1<<14) <= deltaPC && deltaPC < (1<<16) {
			Adduint8(linkctxt, s, dwarf.DW_LNS_fixed_advance_pc)
			Adduint16(linkctxt, s, uint16(deltaPC))
		} else {
			Adduint8(linkctxt, s, dwarf.DW_LNS_advance_pc)
			dwarf.Uleb128put(ctxt, s, int64(deltaPC))
		}
	}

	// Encode deltaLC.
	if deltaLC != 0 {
		Adduint8(linkctxt, s, dwarf.DW_LNS_advance_line)
		dwarf.Sleb128put(ctxt, s, deltaLC)
	}

	// Output the special opcode.
	Adduint8(linkctxt, s, uint8(opcode))
}

/*
 * Walk prog table, emit line program and build DIE tree.
 */

func getCompilationDir() string {
	if dir, err := os.Getwd(); err == nil {
		return dir
	}
	return "/"
}

func writelines(ctxt *Link, syms []*Symbol) ([]*Symbol, []*Symbol) {
	var dwarfctxt dwarf.Context = dwctxt{ctxt}
	if linesec == nil {
		linesec = ctxt.Syms.Lookup(".debug_line", 0)
	}
	linesec.Type = SDWARFSECT
	linesec.R = linesec.R[:0]

	ls := linesec
	syms = append(syms, ls)
	var funcs []*Symbol

	unitstart := int64(-1)
	headerstart := int64(-1)
	headerend := int64(-1)
	epc := int64(0)
	var epcs *Symbol
	var dwinfo *dwarf.DWDie

	lang := dwarf.DW_LANG_Go

	s := ctxt.Textp[0]
	if ctxt.DynlinkingGo() && Headtype == objabi.Hdarwin {
		s = ctxt.Textp[1] // skip runtime.text
	}

	dwinfo = newdie(ctxt, &dwroot, dwarf.DW_ABRV_COMPUNIT, "go", 0)
	newattr(dwinfo, dwarf.DW_AT_language, dwarf.DW_CLS_CONSTANT, int64(lang), 0)
	newattr(dwinfo, dwarf.DW_AT_stmt_list, dwarf.DW_CLS_PTR, 0, linesec)
	newattr(dwinfo, dwarf.DW_AT_low_pc, dwarf.DW_CLS_ADDRESS, s.Value, s)
	// OS X linker requires compilation dir or absolute path in comp unit name to output debug info.
	compDir := getCompilationDir()
	newattr(dwinfo, dwarf.DW_AT_comp_dir, dwarf.DW_CLS_STRING, int64(len(compDir)), compDir)
	producer := "Go cmd/compile " + objabi.Version
	newattr(dwinfo, dwarf.DW_AT_producer, dwarf.DW_CLS_STRING, int64(len(producer)), producer)

	// Write .debug_line Line Number Program Header (sec 6.2.4)
	// Fields marked with (*) must be changed for 64-bit dwarf
	unitLengthOffset := ls.Size
	Adduint32(ctxt, ls, 0) // unit_length (*), filled in at end.
	unitstart = ls.Size
	Adduint16(ctxt, ls, 2) // dwarf version (appendix F)
	headerLengthOffset := ls.Size
	Adduint32(ctxt, ls, 0) // header_length (*), filled in at end.
	headerstart = ls.Size

	// cpos == unitstart + 4 + 2 + 4
	Adduint8(ctxt, ls, 1)              // minimum_instruction_length
	Adduint8(ctxt, ls, 1)              // default_is_stmt
	Adduint8(ctxt, ls, LINE_BASE&0xFF) // line_base
	Adduint8(ctxt, ls, LINE_RANGE)     // line_range
	Adduint8(ctxt, ls, OPCODE_BASE)    // opcode_base
	Adduint8(ctxt, ls, 0)              // standard_opcode_lengths[1]
	Adduint8(ctxt, ls, 1)              // standard_opcode_lengths[2]
	Adduint8(ctxt, ls, 1)              // standard_opcode_lengths[3]
	Adduint8(ctxt, ls, 1)              // standard_opcode_lengths[4]
	Adduint8(ctxt, ls, 1)              // standard_opcode_lengths[5]
	Adduint8(ctxt, ls, 0)              // standard_opcode_lengths[6]
	Adduint8(ctxt, ls, 0)              // standard_opcode_lengths[7]
	Adduint8(ctxt, ls, 0)              // standard_opcode_lengths[8]
	Adduint8(ctxt, ls, 1)              // standard_opcode_lengths[9]
	Adduint8(ctxt, ls, 0)              // include_directories  (empty)

	for _, f := range ctxt.Filesyms {
		Addstring(ls, f.Name)
		Adduint8(ctxt, ls, 0)
		Adduint8(ctxt, ls, 0)
		Adduint8(ctxt, ls, 0)
	}

	// 4 zeros: the string termination + 3 fields.
	Adduint8(ctxt, ls, 0)
	// terminate file_names.
	headerend = ls.Size

	Adduint8(ctxt, ls, 0) // start extended opcode
	dwarf.Uleb128put(dwarfctxt, ls, 1+int64(SysArch.PtrSize))
	Adduint8(ctxt, ls, dwarf.DW_LNE_set_address)

	pc := s.Value
	line := 1
	file := 1
	Addaddr(ctxt, ls, s)

	var pcfile Pciter
	var pcline Pciter
	for _, s := range ctxt.Textp {

		epc = s.Value + s.Size
		epcs = s

		dsym := ctxt.Syms.Lookup(dwarf.InfoPrefix+s.Name, int(s.Version))
		dsym.Attr |= AttrNotInSymbolTable | AttrReachable
		dsym.Type = SDWARFINFO
		for _, r := range dsym.R {
			if r.Type == objabi.R_DWARFREF && r.Sym.Size == 0 {
				if Buildmode == BuildmodeShared {
					// These type symbols may not be present in BuildmodeShared. Skip.
					continue
				}
				n := nameFromDIESym(r.Sym)
				defgotype(ctxt, ctxt.Syms.Lookup("type."+n, 0))
			}
		}
		funcs = append(funcs, dsym)

		if s.FuncInfo == nil {
			continue
		}

		finddebugruntimepath(s)

		pciterinit(ctxt, &pcfile, &s.FuncInfo.Pcfile)
		pciterinit(ctxt, &pcline, &s.FuncInfo.Pcline)
		epc = pc
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
				Adduint8(ctxt, ls, dwarf.DW_LNS_set_file)
				dwarf.Uleb128put(dwarfctxt, ls, int64(pcfile.value))
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

	Adduint8(ctxt, ls, 0) // start extended opcode
	dwarf.Uleb128put(dwarfctxt, ls, 1)
	Adduint8(ctxt, ls, dwarf.DW_LNE_end_sequence)

	newattr(dwinfo, dwarf.DW_AT_high_pc, dwarf.DW_CLS_ADDRESS, epc+1, epcs)

	setuint32(ctxt, ls, unitLengthOffset, uint32(ls.Size-unitstart))
	setuint32(ctxt, ls, headerLengthOffset, uint32(headerend-headerstart))

	return syms, funcs
}

/*
 *  Emit .debug_frame
 */
const (
	dataAlignmentFactor = -4
)

// appendPCDeltaCFA appends per-PC CFA deltas to b and returns the final slice.
func appendPCDeltaCFA(b []byte, deltapc, cfa int64) []byte {
	b = append(b, dwarf.DW_CFA_def_cfa_offset_sf)
	b = dwarf.AppendSleb128(b, cfa/dataAlignmentFactor)

	switch {
	case deltapc < 0x40:
		b = append(b, uint8(dwarf.DW_CFA_advance_loc+deltapc))
	case deltapc < 0x100:
		b = append(b, dwarf.DW_CFA_advance_loc1)
		b = append(b, uint8(deltapc))
	case deltapc < 0x10000:
		b = append(b, dwarf.DW_CFA_advance_loc2)
		b = Thearch.Append16(b, uint16(deltapc))
	default:
		b = append(b, dwarf.DW_CFA_advance_loc4)
		b = Thearch.Append32(b, uint32(deltapc))
	}
	return b
}

func writeframes(ctxt *Link, syms []*Symbol) []*Symbol {
	var dwarfctxt dwarf.Context = dwctxt{ctxt}
	if framesec == nil {
		framesec = ctxt.Syms.Lookup(".debug_frame", 0)
	}
	framesec.Type = SDWARFSECT
	framesec.R = framesec.R[:0]
	fs := framesec
	syms = append(syms, fs)

	// Emit the CIE, Section 6.4.1
	cieReserve := uint32(16)
	if haslinkregister(ctxt) {
		cieReserve = 32
	}
	Adduint32(ctxt, fs, cieReserve)                            // initial length, must be multiple of thearch.ptrsize
	Adduint32(ctxt, fs, 0xffffffff)                            // cid.
	Adduint8(ctxt, fs, 3)                                      // dwarf version (appendix F)
	Adduint8(ctxt, fs, 0)                                      // augmentation ""
	dwarf.Uleb128put(dwarfctxt, fs, 1)                         // code_alignment_factor
	dwarf.Sleb128put(dwarfctxt, fs, dataAlignmentFactor)       // all CFI offset calculations include multiplication with this factor
	dwarf.Uleb128put(dwarfctxt, fs, int64(Thearch.Dwarfreglr)) // return_address_register

	Adduint8(ctxt, fs, dwarf.DW_CFA_def_cfa)                   // Set the current frame address..
	dwarf.Uleb128put(dwarfctxt, fs, int64(Thearch.Dwarfregsp)) // ...to use the value in the platform's SP register (defined in l.go)...
	if haslinkregister(ctxt) {
		dwarf.Uleb128put(dwarfctxt, fs, int64(0)) // ...plus a 0 offset.

		Adduint8(ctxt, fs, dwarf.DW_CFA_same_value) // The platform's link register is unchanged during the prologue.
		dwarf.Uleb128put(dwarfctxt, fs, int64(Thearch.Dwarfreglr))

		Adduint8(ctxt, fs, dwarf.DW_CFA_val_offset)                // The previous value...
		dwarf.Uleb128put(dwarfctxt, fs, int64(Thearch.Dwarfregsp)) // ...of the platform's SP register...
		dwarf.Uleb128put(dwarfctxt, fs, int64(0))                  // ...is CFA+0.
	} else {
		dwarf.Uleb128put(dwarfctxt, fs, int64(SysArch.PtrSize)) // ...plus the word size (because the call instruction implicitly adds one word to the frame).

		Adduint8(ctxt, fs, dwarf.DW_CFA_offset_extended)                             // The previous value...
		dwarf.Uleb128put(dwarfctxt, fs, int64(Thearch.Dwarfreglr))                   // ...of the return address...
		dwarf.Uleb128put(dwarfctxt, fs, int64(-SysArch.PtrSize)/dataAlignmentFactor) // ...is saved at [CFA - (PtrSize/4)].
	}

	// 4 is to exclude the length field.
	pad := int64(cieReserve) + 4 - fs.Size

	if pad < 0 {
		Exitf("dwarf: cieReserve too small by %d bytes.", -pad)
	}

	Addbytes(fs, zeros[:pad])

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
				deltaBuf = appendPCDeltaCFA(deltaBuf, int64(nextpc)-int64(pcsp.pc), int64(pcsp.value))
			} else {
				deltaBuf = appendPCDeltaCFA(deltaBuf, int64(nextpc)-int64(pcsp.pc), int64(SysArch.PtrSize)+int64(pcsp.value))
			}
		}
		pad := int(Rnd(int64(len(deltaBuf)), int64(SysArch.PtrSize))) - len(deltaBuf)
		deltaBuf = append(deltaBuf, zeros[:pad]...)

		// Emit the FDE header, Section 6.4.1.
		//	4 bytes: length, must be multiple of thearch.ptrsize
		//	4 bytes: Pointer to the CIE above, at offset 0
		//	ptrsize: initial location
		//	ptrsize: address range
		Adduint32(ctxt, fs, uint32(4+2*SysArch.PtrSize+len(deltaBuf))) // length (excludes itself)
		if Linkmode == LinkExternal {
			adddwarfref(ctxt, fs, framesec, 4)
		} else {
			Adduint32(ctxt, fs, 0) // CIE offset
		}
		Addaddr(ctxt, fs, s)
		adduintxx(ctxt, fs, uint64(s.Size), SysArch.PtrSize) // address range
		Addbytes(fs, deltaBuf)
	}
	return syms
}

func writeranges(ctxt *Link, syms []*Symbol) []*Symbol {
	if rangesec == nil {
		rangesec = ctxt.Syms.Lookup(".debug_ranges", 0)
	}
	rangesec.Type = SDWARFSECT
	rangesec.Attr |= AttrReachable
	rangesec.R = rangesec.R[:0]

	for _, s := range ctxt.Textp {
		rangeSym := ctxt.Syms.Lookup(dwarf.RangePrefix+s.Name, int(s.Version))
		rangeSym.Attr |= AttrReachable
		rangeSym.Type = SDWARFRANGE
		rangeSym.Value = rangesec.Size
		rangesec.P = append(rangesec.P, rangeSym.P...)
		for _, r := range rangeSym.R {
			r.Off += int32(rangesec.Size)
			rangesec.R = append(rangesec.R, r)
		}
		rangesec.Size += rangeSym.Size
	}
	if rangesec.Size > 0 {
		// PE does not like empty sections
		syms = append(syms, rangesec)
	}
	return syms
}

/*
 *  Walk DWarfDebugInfoEntries, and emit .debug_info
 */
const (
	COMPUNITHEADERSIZE = 4 + 2 + 4 + 1
)

func writeinfo(ctxt *Link, syms []*Symbol, funcs []*Symbol) []*Symbol {
	if infosec == nil {
		infosec = ctxt.Syms.Lookup(".debug_info", 0)
	}
	infosec.R = infosec.R[:0]
	infosec.Type = SDWARFINFO
	infosec.Attr |= AttrReachable
	syms = append(syms, infosec)

	if arangessec == nil {
		arangessec = ctxt.Syms.Lookup(".dwarfaranges", 0)
	}
	arangessec.R = arangessec.R[:0]

	var dwarfctxt dwarf.Context = dwctxt{ctxt}

	for compunit := dwroot.Child; compunit != nil; compunit = compunit.Link {
		s := dtolsym(compunit.Sym)

		// Write .debug_info Compilation Unit Header (sec 7.5.1)
		// Fields marked with (*) must be changed for 64-bit dwarf
		// This must match COMPUNITHEADERSIZE above.
		Adduint32(ctxt, s, 0) // unit_length (*), will be filled in later.
		Adduint16(ctxt, s, 4) // dwarf version (appendix F)

		// debug_abbrev_offset (*)
		adddwarfref(ctxt, s, abbrevsym, 4)

		Adduint8(ctxt, s, uint8(SysArch.PtrSize)) // address_size

		dwarf.Uleb128put(dwarfctxt, s, int64(compunit.Abbrev))
		dwarf.PutAttrs(dwarfctxt, s, compunit.Abbrev, compunit.Attr)

		cu := []*Symbol{s}
		if funcs != nil {
			cu = append(cu, funcs...)
			funcs = nil
		}
		cu = putdies(ctxt, dwarfctxt, cu, compunit.Child)
		var cusize int64
		for _, child := range cu {
			cusize += child.Size
		}
		cusize -= 4 // exclude the length field.
		setuint32(ctxt, s, 0, uint32(cusize))
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

func writepub(ctxt *Link, sname string, ispub func(*dwarf.DWDie) bool, syms []*Symbol) []*Symbol {
	s := ctxt.Syms.Lookup(sname, 0)
	s.Type = SDWARFSECT
	syms = append(syms, s)

	for compunit := dwroot.Child; compunit != nil; compunit = compunit.Link {
		sectionstart := s.Size
		culength := uint32(getattr(compunit, dwarf.DW_AT_byte_size).Value) + 4

		// Write .debug_pubnames/types	Header (sec 6.1.1)
		Adduint32(ctxt, s, 0)                          // unit_length (*), will be filled in later.
		Adduint16(ctxt, s, 2)                          // dwarf version (appendix F)
		adddwarfref(ctxt, s, dtolsym(compunit.Sym), 4) // debug_info_offset (of the Comp unit Header)
		Adduint32(ctxt, s, culength)                   // debug_info_length

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

		Adduint32(ctxt, s, 0)

		setuint32(ctxt, s, sectionstart, uint32(s.Size-sectionstart)-4) // exclude the length field.
	}

	return syms
}

/*
 *  emit .debug_aranges.  _info must have been written before,
 *  because we need die->offs of dwarf.DW_globals.
 */
func writearanges(ctxt *Link, syms []*Symbol) []*Symbol {
	s := ctxt.Syms.Lookup(".debug_aranges", 0)
	s.Type = SDWARFSECT
	// The first tuple is aligned to a multiple of the size of a single tuple
	// (twice the size of an address)
	headersize := int(Rnd(4+2+4+1+1, int64(SysArch.PtrSize*2))) // don't count unit_length field itself

	for compunit := dwroot.Child; compunit != nil; compunit = compunit.Link {
		b := getattr(compunit, dwarf.DW_AT_low_pc)
		if b == nil {
			continue
		}
		e := getattr(compunit, dwarf.DW_AT_high_pc)
		if e == nil {
			continue
		}

		// Write .debug_aranges	 Header + entry	 (sec 6.1.2)
		unitlength := uint32(headersize) + 4*uint32(SysArch.PtrSize) - 4
		Adduint32(ctxt, s, unitlength) // unit_length (*)
		Adduint16(ctxt, s, 2)          // dwarf version (appendix F)

		adddwarfref(ctxt, s, dtolsym(compunit.Sym), 4)

		Adduint8(ctxt, s, uint8(SysArch.PtrSize)) // address_size
		Adduint8(ctxt, s, 0)                      // segment_size
		padding := headersize - (4 + 2 + 4 + 1 + 1)
		for i := 0; i < padding; i++ {
			Adduint8(ctxt, s, 0)
		}

		Addaddrplus(ctxt, s, b.Data.(*Symbol), b.Value-(b.Data.(*Symbol)).Value)
		adduintxx(ctxt, s, uint64(e.Value-b.Value), SysArch.PtrSize)
		adduintxx(ctxt, s, 0, SysArch.PtrSize)
		adduintxx(ctxt, s, 0, SysArch.PtrSize)
	}
	if s.Size > 0 {
		syms = append(syms, s)
	}
	return syms
}

func writegdbscript(ctxt *Link, syms []*Symbol) []*Symbol {
	if Linkmode == LinkExternal && Headtype == objabi.Hwindows && Buildmode == BuildmodeCArchive {
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
		s.Type = SDWARFSECT
		syms = append(syms, s)
		Adduint8(ctxt, s, 1) // magic 1 byte?
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
	if *FlagS && Headtype != objabi.Hdarwin {
		return
	}
	if Headtype == objabi.Hplan9 {
		return
	}

	if Linkmode == LinkExternal {
		switch {
		case Iself:
		case Headtype == objabi.Hdarwin:
		case Headtype == objabi.Hwindows:
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
	newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, int64(SysArch.PtrSize), 0)
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

	syms := writeabbrev(ctxt, nil)
	syms, funcs := writelines(ctxt, syms)
	syms = writeframes(ctxt, syms)
	syms = writeranges(ctxt, syms)

	synthesizestringtypes(ctxt, dwtypes.Child)
	synthesizeslicetypes(ctxt, dwtypes.Child)
	synthesizemaptypes(ctxt, dwtypes.Child)
	synthesizechantypes(ctxt, dwtypes.Child)

	reversetree(&dwroot.Child)
	reversetree(&dwtypes.Child)
	reversetree(&dwglobals.Child)

	movetomodule(&dwtypes)
	movetomodule(&dwglobals)

	// Need to reorder symbols so SDWARFINFO is after all SDWARFSECT
	// (but we need to generate dies before writepub)
	infosyms := writeinfo(ctxt, nil, funcs)

	syms = writepub(ctxt, ".debug_pubnames", ispubname, syms)
	syms = writepub(ctxt, ".debug_pubtypes", ispubtype, syms)
	syms = writearanges(ctxt, syms)
	syms = writegdbscript(ctxt, syms)
	syms = append(syms, infosyms...)
	dwarfp = syms
}

/*
 *  Elf.
 */
func dwarfaddshstrings(ctxt *Link, shstrtab *Symbol) {
	if *FlagW { // disable dwarf
		return
	}

	Addstring(shstrtab, ".debug_abbrev")
	Addstring(shstrtab, ".debug_aranges")
	Addstring(shstrtab, ".debug_frame")
	Addstring(shstrtab, ".debug_info")
	Addstring(shstrtab, ".debug_line")
	Addstring(shstrtab, ".debug_pubnames")
	Addstring(shstrtab, ".debug_pubtypes")
	Addstring(shstrtab, ".debug_gdb_scripts")
	Addstring(shstrtab, ".debug_ranges")
	if Linkmode == LinkExternal {
		Addstring(shstrtab, elfRelType+".debug_info")
		Addstring(shstrtab, elfRelType+".debug_aranges")
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
	if Linkmode != LinkExternal {
		return
	}
	sym := ctxt.Syms.Lookup(".debug_info", 0)
	putelfsectionsym(sym, sym.Sect.Elfsect.shnum)
	sym = ctxt.Syms.Lookup(".debug_abbrev", 0)
	putelfsectionsym(sym, sym.Sect.Elfsect.shnum)
	sym = ctxt.Syms.Lookup(".debug_line", 0)
	putelfsectionsym(sym, sym.Sect.Elfsect.shnum)
	sym = ctxt.Syms.Lookup(".debug_frame", 0)
	putelfsectionsym(sym, sym.Sect.Elfsect.shnum)
	sym = ctxt.Syms.Lookup(".debug_ranges", 0)
	if sym.Sect != nil {
		putelfsectionsym(sym, sym.Sect.Elfsect.shnum)
	}
}

/*
 * Windows PE
 */
func dwarfaddpeheaders(ctxt *Link) {
	if *FlagW { // disable dwarf
		return
	}
	for _, sect := range Segdwarf.Sections {
		h := newPEDWARFSection(ctxt, sect.Name, int64(sect.Length))
		fileoff := sect.Vaddr - Segdwarf.Vaddr + Segdwarf.Fileoff
		if uint64(h.PointerToRawData) != fileoff {
			Exitf("%s.PointerToRawData = %#x, want %#x", sect.Name, h.PointerToRawData, fileoff)
		}
	}
}
