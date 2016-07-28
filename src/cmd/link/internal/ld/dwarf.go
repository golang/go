// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO/NICETOHAVE:
//   - eliminate DW_CLS_ if not used
//   - package info in compilation units
//   - assign global variables and types to their packages
//   - gdb uses c syntax, meaning clumsy quoting is needed for go identifiers. eg
//     ptype struct '[]uint8' and qualifiers need to be quoted away
//   - lexical scoping is lost, so gdb gets confused as to which 'main.i' you mean.
//   - file:line info for variables
//   - make strings a typedef so prettyprinters can see the underlying string type

package ld

import (
	"cmd/internal/dwarf"
	"cmd/internal/obj"
	"fmt"
	"log"
	"os"
	"strings"
)

type dwCtxt struct{}

func (c dwCtxt) PtrSize() int {
	return SysArch.PtrSize
}
func (c dwCtxt) AddInt(s dwarf.Sym, size int, i int64) {
	ls := s.(*LSym)
	adduintxx(Ctxt, ls, uint64(i), size)
}
func (c dwCtxt) AddBytes(s dwarf.Sym, b []byte) {
	ls := s.(*LSym)
	Addbytes(Ctxt, ls, b)
}
func (c dwCtxt) AddString(s dwarf.Sym, v string) {
	Addstring(s.(*LSym), v)
}
func (c dwCtxt) SymValue(s dwarf.Sym) int64 {
	return s.(*LSym).Value
}

func (c dwCtxt) AddAddress(s dwarf.Sym, data interface{}, value int64) {
	if value != 0 {
		value -= (data.(*LSym)).Value
	}
	Addaddrplus(Ctxt, s.(*LSym), data.(*LSym), value)
}

func (c dwCtxt) AddSectionOffset(s dwarf.Sym, size int, t interface{}, ofs int64) {
	ls := s.(*LSym)
	switch size {
	default:
		Diag("invalid size %d in adddwarfref\n", size)
		fallthrough
	case SysArch.PtrSize:
		Addaddr(Ctxt, ls, t.(*LSym))
	case 4:
		addaddrplus4(Ctxt, ls, t.(*LSym), 0)
	}
	r := &ls.R[len(ls.R)-1]
	r.Type = obj.R_DWARFREF
	r.Add = ofs
}

/*
 * Offsets and sizes of the debug_* sections in the cout file.
 */
var abbrevsym *LSym
var arangessec *LSym
var framesec *LSym
var infosec *LSym
var linesec *LSym

var gdbscript string

var dwarfp *LSym

func writeabbrev(syms []*LSym) []*LSym {
	s := Linklookup(Ctxt, ".debug_abbrev", 0)
	s.Type = obj.SDWARFSECT
	abbrevsym = s
	Addbytes(Ctxt, s, dwarf.GetAbbrev())
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
func newdie(parent *dwarf.DWDie, abbrev int, name string, version int) *dwarf.DWDie {
	die := new(dwarf.DWDie)
	die.Abbrev = abbrev
	die.Link = parent.Child
	parent.Child = die

	newattr(die, dwarf.DW_AT_name, dwarf.DW_CLS_STRING, int64(len(name)), name)

	if name != "" && (abbrev <= dwarf.DW_ABRV_VARIABLE || abbrev >= dwarf.DW_ABRV_NULLTYPE) {
		if abbrev != dwarf.DW_ABRV_VARIABLE || version == 0 {
			sym := Linklookup(Ctxt, dwarf.InfoPrefix+name, version)
			sym.Attr |= AttrHidden
			sym.Type = obj.SDWARFINFO
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

func walksymtypedef(s *LSym) *LSym {
	if t := Linkrlookup(Ctxt, s.Name+"..def", int(s.Version)); t != nil {
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

func find(name string) *LSym {
	n := append(prefixBuf, name...)
	// The string allocation below is optimized away because it is only used in a map lookup.
	s := Linkrlookup(Ctxt, string(n), 0)
	prefixBuf = n[:len(dwarf.InfoPrefix)]
	if s != nil && s.Type == obj.SDWARFINFO {
		return s
	}
	return nil
}

func mustFind(name string) *LSym {
	r := find(name)
	if r == nil {
		Exitf("dwarf find: cannot find %s", name)
	}
	return r
}

func adddwarfref(ctxt *Link, s *LSym, t *LSym, size int) int64 {
	var result int64
	switch size {
	default:
		Diag("invalid size %d in adddwarfref\n", size)
		fallthrough
	case SysArch.PtrSize:
		result = Addaddr(ctxt, s, t)
	case 4:
		result = addaddrplus4(ctxt, s, t, 0)
	}
	r := &s.R[len(s.R)-1]
	r.Type = obj.R_DWARFREF
	return result
}

func newrefattr(die *dwarf.DWDie, attr uint16, ref *LSym) *dwarf.DWAttr {
	if ref == nil {
		return nil
	}
	return newattr(die, attr, dwarf.DW_CLS_REFERENCE, 0, ref)
}

func putdies(ctxt dwarf.Context, syms []*LSym, die *dwarf.DWDie) []*LSym {
	for ; die != nil; die = die.Link {
		syms = putdie(ctxt, syms, die)
	}
	Adduint8(Ctxt, syms[len(syms)-1], 0)

	return syms
}

func dtolsym(s dwarf.Sym) *LSym {
	if s == nil {
		return nil
	}
	return s.(*LSym)
}

func putdie(ctxt dwarf.Context, syms []*LSym, die *dwarf.DWDie) []*LSym {
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
		return putdies(ctxt, syms, die.Child)
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
func newabslocexprattr(die *dwarf.DWDie, addr int64, sym *LSym) {
	newattr(die, dwarf.DW_AT_location, dwarf.DW_CLS_ADDRESS, addr, sym)
	// below
}

// Lookup predefined types
func lookup_or_diag(n string) *LSym {
	s := Linkrlookup(Ctxt, n, 0)
	if s == nil || s.Size == 0 {
		Exitf("dwarf: missing type: %s", n)
	}

	return s
}

func dotypedef(parent *dwarf.DWDie, name string, def *dwarf.DWDie) {
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
		Diag("dwarf: bad def in dotypedef")
	}

	sym := Linklookup(Ctxt, dtolsym(def.Sym).Name+"..def", 0)
	sym.Attr |= AttrHidden
	sym.Type = obj.SDWARFINFO
	def.Sym = sym

	// The typedef entry must be created after the def,
	// so that future lookups will find the typedef instead
	// of the real definition. This hooks the typedef into any
	// circular definition loops, so that gdb can understand them.
	die := newdie(parent, dwarf.DW_ABRV_TYPEDECL, name, 0)

	newrefattr(die, dwarf.DW_AT_type, sym)
}

// Define gotype, for composite ones recurse into constituents.
func defgotype(gotype *LSym) *LSym {
	if gotype == nil {
		return mustFind("<unspecified>")
	}

	if !strings.HasPrefix(gotype.Name, "type.") {
		Diag("dwarf: type name doesn't start with \"type.\": %s", gotype.Name)
		return mustFind("<unspecified>")
	}

	name := gotype.Name[5:] // could also decode from Type.string

	sdie := find(name)

	if sdie != nil {
		return sdie
	}

	return newtype(gotype).Sym.(*LSym)
}

func newtype(gotype *LSym) *dwarf.DWDie {
	name := gotype.Name[5:] // could also decode from Type.string
	kind := decodetype_kind(gotype)
	bytesize := decodetype_size(gotype)

	var die *dwarf.DWDie
	switch kind {
	case obj.KindBool:
		die = newdie(&dwtypes, dwarf.DW_ABRV_BASETYPE, name, 0)
		newattr(die, dwarf.DW_AT_encoding, dwarf.DW_CLS_CONSTANT, dwarf.DW_ATE_boolean, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case obj.KindInt,
		obj.KindInt8,
		obj.KindInt16,
		obj.KindInt32,
		obj.KindInt64:
		die = newdie(&dwtypes, dwarf.DW_ABRV_BASETYPE, name, 0)
		newattr(die, dwarf.DW_AT_encoding, dwarf.DW_CLS_CONSTANT, dwarf.DW_ATE_signed, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case obj.KindUint,
		obj.KindUint8,
		obj.KindUint16,
		obj.KindUint32,
		obj.KindUint64,
		obj.KindUintptr:
		die = newdie(&dwtypes, dwarf.DW_ABRV_BASETYPE, name, 0)
		newattr(die, dwarf.DW_AT_encoding, dwarf.DW_CLS_CONSTANT, dwarf.DW_ATE_unsigned, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case obj.KindFloat32,
		obj.KindFloat64:
		die = newdie(&dwtypes, dwarf.DW_ABRV_BASETYPE, name, 0)
		newattr(die, dwarf.DW_AT_encoding, dwarf.DW_CLS_CONSTANT, dwarf.DW_ATE_float, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case obj.KindComplex64,
		obj.KindComplex128:
		die = newdie(&dwtypes, dwarf.DW_ABRV_BASETYPE, name, 0)
		newattr(die, dwarf.DW_AT_encoding, dwarf.DW_CLS_CONSTANT, dwarf.DW_ATE_complex_float, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case obj.KindArray:
		die = newdie(&dwtypes, dwarf.DW_ABRV_ARRAYTYPE, name, 0)
		dotypedef(&dwtypes, name, die)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		s := decodetype_arrayelem(gotype)
		newrefattr(die, dwarf.DW_AT_type, defgotype(s))
		fld := newdie(die, dwarf.DW_ABRV_ARRAYRANGE, "range", 0)

		// use actual length not upper bound; correct for 0-length arrays.
		newattr(fld, dwarf.DW_AT_count, dwarf.DW_CLS_CONSTANT, decodetype_arraylen(gotype), 0)

		newrefattr(fld, dwarf.DW_AT_type, mustFind("uintptr"))

	case obj.KindChan:
		die = newdie(&dwtypes, dwarf.DW_ABRV_CHANTYPE, name, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		s := decodetype_chanelem(gotype)
		newrefattr(die, dwarf.DW_AT_go_elem, defgotype(s))
		// Save elem type for synthesizechantypes. We could synthesize here
		// but that would change the order of DIEs we output.
		newrefattr(die, dwarf.DW_AT_type, s)

	case obj.KindFunc:
		die = newdie(&dwtypes, dwarf.DW_ABRV_FUNCTYPE, name, 0)
		dotypedef(&dwtypes, name, die)
		newrefattr(die, dwarf.DW_AT_type, mustFind("void"))
		nfields := decodetype_funcincount(gotype)
		var fld *dwarf.DWDie
		var s *LSym
		for i := 0; i < nfields; i++ {
			s = decodetype_funcintype(gotype, i)
			fld = newdie(die, dwarf.DW_ABRV_FUNCTYPEPARAM, s.Name[5:], 0)
			newrefattr(fld, dwarf.DW_AT_type, defgotype(s))
		}

		if decodetype_funcdotdotdot(gotype) {
			newdie(die, dwarf.DW_ABRV_DOTDOTDOT, "...", 0)
		}
		nfields = decodetype_funcoutcount(gotype)
		for i := 0; i < nfields; i++ {
			s = decodetype_funcouttype(gotype, i)
			fld = newdie(die, dwarf.DW_ABRV_FUNCTYPEPARAM, s.Name[5:], 0)
			newrefattr(fld, dwarf.DW_AT_type, defptrto(defgotype(s)))
		}

	case obj.KindInterface:
		die = newdie(&dwtypes, dwarf.DW_ABRV_IFACETYPE, name, 0)
		dotypedef(&dwtypes, name, die)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		nfields := int(decodetype_ifacemethodcount(gotype))
		var s *LSym
		if nfields == 0 {
			s = lookup_or_diag("type.runtime.eface")
		} else {
			s = lookup_or_diag("type.runtime.iface")
		}
		newrefattr(die, dwarf.DW_AT_type, defgotype(s))

	case obj.KindMap:
		die = newdie(&dwtypes, dwarf.DW_ABRV_MAPTYPE, name, 0)
		s := decodetype_mapkey(gotype)
		newrefattr(die, dwarf.DW_AT_go_key, defgotype(s))
		s = decodetype_mapvalue(gotype)
		newrefattr(die, dwarf.DW_AT_go_elem, defgotype(s))
		// Save gotype for use in synthesizemaptypes. We could synthesize here,
		// but that would change the order of the DIEs.
		newrefattr(die, dwarf.DW_AT_type, gotype)

	case obj.KindPtr:
		die = newdie(&dwtypes, dwarf.DW_ABRV_PTRTYPE, name, 0)
		dotypedef(&dwtypes, name, die)
		s := decodetype_ptrelem(gotype)
		newrefattr(die, dwarf.DW_AT_type, defgotype(s))

	case obj.KindSlice:
		die = newdie(&dwtypes, dwarf.DW_ABRV_SLICETYPE, name, 0)
		dotypedef(&dwtypes, name, die)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		s := decodetype_arrayelem(gotype)
		elem := defgotype(s)
		newrefattr(die, dwarf.DW_AT_go_elem, elem)

	case obj.KindString:
		die = newdie(&dwtypes, dwarf.DW_ABRV_STRINGTYPE, name, 0)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)

	case obj.KindStruct:
		die = newdie(&dwtypes, dwarf.DW_ABRV_STRUCTTYPE, name, 0)
		dotypedef(&dwtypes, name, die)
		newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, bytesize, 0)
		nfields := decodetype_structfieldcount(gotype)
		var f string
		var fld *dwarf.DWDie
		var s *LSym
		for i := 0; i < nfields; i++ {
			f = decodetype_structfieldname(gotype, i)
			s = decodetype_structfieldtype(gotype, i)
			if f == "" {
				f = s.Name[5:] // skip "type."
			}
			fld = newdie(die, dwarf.DW_ABRV_STRUCTFIELD, f, 0)
			newrefattr(fld, dwarf.DW_AT_type, defgotype(s))
			newmemberoffsetattr(fld, int32(decodetype_structfieldoffs(gotype, i)))
		}

	case obj.KindUnsafePointer:
		die = newdie(&dwtypes, dwarf.DW_ABRV_BARE_PTRTYPE, name, 0)

	default:
		Diag("dwarf: definition of unknown kind %d: %s", kind, gotype.Name)
		die = newdie(&dwtypes, dwarf.DW_ABRV_TYPEDECL, name, 0)
		newrefattr(die, dwarf.DW_AT_type, mustFind("<unspecified>"))
	}

	newattr(die, dwarf.DW_AT_go_kind, dwarf.DW_CLS_CONSTANT, int64(kind), 0)

	if _, ok := prototypedies[gotype.Name]; ok {
		prototypedies[gotype.Name] = die
	}

	return die
}

func nameFromDIESym(dwtype *LSym) string {
	return strings.TrimSuffix(dwtype.Name[len(dwarf.InfoPrefix):], "..def")
}

// Find or construct *T given T.
func defptrto(dwtype *LSym) *LSym {
	ptrname := "*" + nameFromDIESym(dwtype)
	die := find(ptrname)
	if die == nil {
		pdie := newdie(&dwtypes, dwarf.DW_ABRV_PTRTYPE, ptrname, 0)
		newrefattr(pdie, dwarf.DW_AT_type, dwtype)
		return dtolsym(pdie.Sym)
	}

	return die
}

// Copies src's children into dst. Copies attributes by value.
// DWAttr.data is copied as pointer only. If except is one of
// the top-level children, it will not be copied.
func copychildrenexcept(dst *dwarf.DWDie, src *dwarf.DWDie, except *dwarf.DWDie) {
	for src = src.Child; src != nil; src = src.Link {
		if src == except {
			continue
		}
		c := newdie(dst, src.Abbrev, getattr(src, dwarf.DW_AT_name).Data.(string), 0)
		for a := src.Attr; a != nil; a = a.Link {
			newattr(c, a.Atr, int(a.Cls), a.Value, a.Data)
		}
		copychildrenexcept(c, src, nil)
	}

	reverselist(&dst.Child)
}

func copychildren(dst *dwarf.DWDie, src *dwarf.DWDie) {
	copychildrenexcept(dst, src, nil)
}

// Search children (assumed to have TAG_member) for the one named
// field and set its AT_type to dwtype
func substitutetype(structdie *dwarf.DWDie, field string, dwtype *LSym) {
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

func findprotodie(name string) *dwarf.DWDie {
	die, ok := prototypedies[name]
	if ok && die == nil {
		defgotype(lookup_or_diag(name))
		die = prototypedies[name]
	}
	return die
}

func synthesizestringtypes(die *dwarf.DWDie) {
	prototype := walktypedef(findprotodie("type.runtime.stringStructDWARF"))
	if prototype == nil {
		return
	}

	for ; die != nil; die = die.Link {
		if die.Abbrev != dwarf.DW_ABRV_STRINGTYPE {
			continue
		}
		copychildren(die, prototype)
	}
}

func synthesizeslicetypes(die *dwarf.DWDie) {
	prototype := walktypedef(findprotodie("type.runtime.slice"))
	if prototype == nil {
		return
	}

	for ; die != nil; die = die.Link {
		if die.Abbrev != dwarf.DW_ABRV_SLICETYPE {
			continue
		}
		copychildren(die, prototype)
		elem := getattr(die, dwarf.DW_AT_go_elem).Data.(*LSym)
		substitutetype(die, "array", defptrto(elem))
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

func mkinternaltype(abbrev int, typename, keyname, valname string, f func(*dwarf.DWDie)) *LSym {
	name := mkinternaltypename(typename, keyname, valname)
	symname := dwarf.InfoPrefix + name
	s := Linkrlookup(Ctxt, symname, 0)
	if s != nil && s.Type == obj.SDWARFINFO {
		return s
	}
	die := newdie(&dwtypes, abbrev, name, 0)
	f(die)
	return dtolsym(die.Sym)
}

func synthesizemaptypes(die *dwarf.DWDie) {
	hash := walktypedef(findprotodie("type.runtime.hmap"))
	bucket := walktypedef(findprotodie("type.runtime.bmap"))

	if hash == nil {
		return
	}

	for ; die != nil; die = die.Link {
		if die.Abbrev != dwarf.DW_ABRV_MAPTYPE {
			continue
		}
		gotype := getattr(die, dwarf.DW_AT_type).Data.(*LSym)
		keytype := decodetype_mapkey(gotype)
		valtype := decodetype_mapvalue(gotype)
		keysize, valsize := decodetype_size(keytype), decodetype_size(valtype)
		keytype, valtype = walksymtypedef(defgotype(keytype)), walksymtypedef(defgotype(valtype))

		// compute size info like hashmap.c does.
		indirect_key, indirect_val := false, false
		if keysize > MaxKeySize {
			keysize = int64(SysArch.PtrSize)
			indirect_key = true
		}
		if valsize > MaxValSize {
			valsize = int64(SysArch.PtrSize)
			indirect_val = true
		}

		// Construct type to represent an array of BucketSize keys
		keyname := nameFromDIESym(keytype)
		dwhks := mkinternaltype(dwarf.DW_ABRV_ARRAYTYPE, "[]key", keyname, "", func(dwhk *dwarf.DWDie) {
			newattr(dwhk, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, BucketSize*keysize, 0)
			t := keytype
			if indirect_key {
				t = defptrto(keytype)
			}
			newrefattr(dwhk, dwarf.DW_AT_type, t)
			fld := newdie(dwhk, dwarf.DW_ABRV_ARRAYRANGE, "size", 0)
			newattr(fld, dwarf.DW_AT_count, dwarf.DW_CLS_CONSTANT, BucketSize, 0)
			newrefattr(fld, dwarf.DW_AT_type, mustFind("uintptr"))
		})

		// Construct type to represent an array of BucketSize values
		valname := nameFromDIESym(valtype)
		dwhvs := mkinternaltype(dwarf.DW_ABRV_ARRAYTYPE, "[]val", valname, "", func(dwhv *dwarf.DWDie) {
			newattr(dwhv, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, BucketSize*valsize, 0)
			t := valtype
			if indirect_val {
				t = defptrto(valtype)
			}
			newrefattr(dwhv, dwarf.DW_AT_type, t)
			fld := newdie(dwhv, dwarf.DW_ABRV_ARRAYRANGE, "size", 0)
			newattr(fld, dwarf.DW_AT_count, dwarf.DW_CLS_CONSTANT, BucketSize, 0)
			newrefattr(fld, dwarf.DW_AT_type, mustFind("uintptr"))
		})

		// Construct bucket<K,V>
		dwhbs := mkinternaltype(dwarf.DW_ABRV_STRUCTTYPE, "bucket", keyname, valname, func(dwhb *dwarf.DWDie) {
			// Copy over all fields except the field "data" from the generic
			// bucket. "data" will be replaced with keys/values below.
			copychildrenexcept(dwhb, bucket, findchild(bucket, "data"))

			fld := newdie(dwhb, dwarf.DW_ABRV_STRUCTFIELD, "keys", 0)
			newrefattr(fld, dwarf.DW_AT_type, dwhks)
			newmemberoffsetattr(fld, BucketSize)
			fld = newdie(dwhb, dwarf.DW_ABRV_STRUCTFIELD, "values", 0)
			newrefattr(fld, dwarf.DW_AT_type, dwhvs)
			newmemberoffsetattr(fld, BucketSize+BucketSize*int32(keysize))
			fld = newdie(dwhb, dwarf.DW_ABRV_STRUCTFIELD, "overflow", 0)
			newrefattr(fld, dwarf.DW_AT_type, defptrto(dtolsym(dwhb.Sym)))
			newmemberoffsetattr(fld, BucketSize+BucketSize*(int32(keysize)+int32(valsize)))
			if SysArch.RegSize > SysArch.PtrSize {
				fld = newdie(dwhb, dwarf.DW_ABRV_STRUCTFIELD, "pad", 0)
				newrefattr(fld, dwarf.DW_AT_type, mustFind("uintptr"))
				newmemberoffsetattr(fld, BucketSize+BucketSize*(int32(keysize)+int32(valsize))+int32(SysArch.PtrSize))
			}

			newattr(dwhb, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, BucketSize+BucketSize*keysize+BucketSize*valsize+int64(SysArch.RegSize), 0)
		})

		// Construct hash<K,V>
		dwhs := mkinternaltype(dwarf.DW_ABRV_STRUCTTYPE, "hash", keyname, valname, func(dwh *dwarf.DWDie) {
			copychildren(dwh, hash)
			substitutetype(dwh, "buckets", defptrto(dwhbs))
			substitutetype(dwh, "oldbuckets", defptrto(dwhbs))
			newattr(dwh, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, getattr(hash, dwarf.DW_AT_byte_size).Value, nil)
		})

		// make map type a pointer to hash<K,V>
		newrefattr(die, dwarf.DW_AT_type, defptrto(dwhs))
	}
}

func synthesizechantypes(die *dwarf.DWDie) {
	sudog := walktypedef(findprotodie("type.runtime.sudog"))
	waitq := walktypedef(findprotodie("type.runtime.waitq"))
	hchan := walktypedef(findprotodie("type.runtime.hchan"))
	if sudog == nil || waitq == nil || hchan == nil {
		return
	}

	sudogsize := int(getattr(sudog, dwarf.DW_AT_byte_size).Value)

	for ; die != nil; die = die.Link {
		if die.Abbrev != dwarf.DW_ABRV_CHANTYPE {
			continue
		}
		elemgotype := getattr(die, dwarf.DW_AT_type).Data.(*LSym)
		elemsize := decodetype_size(elemgotype)
		elemname := elemgotype.Name[5:]
		elemtype := walksymtypedef(defgotype(elemgotype))

		// sudog<T>
		dwss := mkinternaltype(dwarf.DW_ABRV_STRUCTTYPE, "sudog", elemname, "", func(dws *dwarf.DWDie) {
			copychildren(dws, sudog)
			substitutetype(dws, "elem", elemtype)
			if elemsize > 8 {
				elemsize -= 8
			} else {
				elemsize = 0
			}
			newattr(dws, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, int64(sudogsize)+elemsize, nil)
		})

		// waitq<T>
		dwws := mkinternaltype(dwarf.DW_ABRV_STRUCTTYPE, "waitq", elemname, "", func(dww *dwarf.DWDie) {

			copychildren(dww, waitq)
			substitutetype(dww, "first", defptrto(dwss))
			substitutetype(dww, "last", defptrto(dwss))
			newattr(dww, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, getattr(waitq, dwarf.DW_AT_byte_size).Value, nil)
		})

		// hchan<T>
		dwhs := mkinternaltype(dwarf.DW_ABRV_STRUCTTYPE, "hchan", elemname, "", func(dwh *dwarf.DWDie) {
			copychildren(dwh, hchan)
			substitutetype(dwh, "recvq", dwws)
			substitutetype(dwh, "sendq", dwws)
			newattr(dwh, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, getattr(hchan, dwarf.DW_AT_byte_size).Value, nil)
		})

		newrefattr(die, dwarf.DW_AT_type, defptrto(dwhs))
	}
}

// For use with pass.c::genasmsym
func defdwsymb(sym *LSym, s string, t int, v int64, size int64, ver int, gotype *LSym) {
	if strings.HasPrefix(s, "go.string.") {
		return
	}
	if strings.HasPrefix(s, "runtime.gcbits.") {
		return
	}

	if strings.HasPrefix(s, "type.") && s != "type.*" && !strings.HasPrefix(s, "type..") {
		defgotype(sym)
		return
	}

	var dv *dwarf.DWDie

	var dt *LSym
	switch t {
	default:
		return

	case 'd', 'b', 'D', 'B':
		dv = newdie(&dwglobals, dwarf.DW_ABRV_VARIABLE, s, ver)
		newabslocexprattr(dv, v, sym)
		if ver == 0 {
			newattr(dv, dwarf.DW_AT_external, dwarf.DW_CLS_FLAG, 1, 0)
		}
		fallthrough

	case 'a', 'p':
		dt = defgotype(gotype)
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
func finddebugruntimepath(s *LSym) {
	if gdbscript != "" {
		return
	}

	for i := range s.FuncInfo.File {
		f := s.FuncInfo.File[i]
		if i := strings.Index(f.Name, "runtime/runtime.go"); i >= 0 {
			gdbscript = f.Name[:i] + "runtime/runtime-gdb.py"
			break
		}
	}
}

/*
 * Generate short opcodes when possible, long ones when necessary.
 * See section 6.2.5
 */
const (
	LINE_BASE   = -1
	LINE_RANGE  = 4
	OPCODE_BASE = 10
)

func putpclcdelta(ctxt dwarf.Context, s *LSym, delta_pc int64, delta_lc int64) {
	if LINE_BASE <= delta_lc && delta_lc < LINE_BASE+LINE_RANGE {
		var opcode int64 = OPCODE_BASE + (delta_lc - LINE_BASE) + (LINE_RANGE * delta_pc)
		if OPCODE_BASE <= opcode && opcode < 256 {
			Adduint8(Ctxt, s, uint8(opcode))
			return
		}
	}

	if delta_pc != 0 {
		Adduint8(Ctxt, s, dwarf.DW_LNS_advance_pc)
		dwarf.Sleb128put(ctxt, s, delta_pc)
	}

	Adduint8(Ctxt, s, dwarf.DW_LNS_advance_line)
	dwarf.Sleb128put(ctxt, s, delta_lc)
	Adduint8(Ctxt, s, dwarf.DW_LNS_copy)
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

func writelines(syms []*LSym) ([]*LSym, []*LSym) {
	var dwarfctxt dwarf.Context = dwCtxt{}
	if linesec == nil {
		linesec = Linklookup(Ctxt, ".debug_line", 0)
	}
	linesec.Type = obj.SDWARFSECT
	linesec.R = linesec.R[:0]

	ls := linesec
	syms = append(syms, ls)
	var funcs []*LSym

	unitstart := int64(-1)
	headerstart := int64(-1)
	headerend := int64(-1)
	epc := int64(0)
	var epcs *LSym
	var dwinfo *dwarf.DWDie

	lang := dwarf.DW_LANG_Go

	s := Ctxt.Textp[0]

	dwinfo = newdie(&dwroot, dwarf.DW_ABRV_COMPUNIT, "go", 0)
	newattr(dwinfo, dwarf.DW_AT_language, dwarf.DW_CLS_CONSTANT, int64(lang), 0)
	newattr(dwinfo, dwarf.DW_AT_stmt_list, dwarf.DW_CLS_PTR, 0, linesec)
	newattr(dwinfo, dwarf.DW_AT_low_pc, dwarf.DW_CLS_ADDRESS, s.Value, s)
	// OS X linker requires compilation dir or absolute path in comp unit name to output debug info.
	compDir := getCompilationDir()
	newattr(dwinfo, dwarf.DW_AT_comp_dir, dwarf.DW_CLS_STRING, int64(len(compDir)), compDir)

	// Write .debug_line Line Number Program Header (sec 6.2.4)
	// Fields marked with (*) must be changed for 64-bit dwarf
	unit_length_offset := ls.Size
	Adduint32(Ctxt, ls, 0) // unit_length (*), filled in at end.
	unitstart = ls.Size
	Adduint16(Ctxt, ls, 2) // dwarf version (appendix F)
	header_length_offset := ls.Size
	Adduint32(Ctxt, ls, 0) // header_length (*), filled in at end.
	headerstart = ls.Size

	// cpos == unitstart + 4 + 2 + 4
	Adduint8(Ctxt, ls, 1)              // minimum_instruction_length
	Adduint8(Ctxt, ls, 1)              // default_is_stmt
	Adduint8(Ctxt, ls, LINE_BASE&0xFF) // line_base
	Adduint8(Ctxt, ls, LINE_RANGE)     // line_range
	Adduint8(Ctxt, ls, OPCODE_BASE)    // opcode_base
	Adduint8(Ctxt, ls, 0)              // standard_opcode_lengths[1]
	Adduint8(Ctxt, ls, 1)              // standard_opcode_lengths[2]
	Adduint8(Ctxt, ls, 1)              // standard_opcode_lengths[3]
	Adduint8(Ctxt, ls, 1)              // standard_opcode_lengths[4]
	Adduint8(Ctxt, ls, 1)              // standard_opcode_lengths[5]
	Adduint8(Ctxt, ls, 0)              // standard_opcode_lengths[6]
	Adduint8(Ctxt, ls, 0)              // standard_opcode_lengths[7]
	Adduint8(Ctxt, ls, 0)              // standard_opcode_lengths[8]
	Adduint8(Ctxt, ls, 1)              // standard_opcode_lengths[9]
	Adduint8(Ctxt, ls, 0)              // include_directories  (empty)

	for _, f := range Ctxt.Filesyms {
		Addstring(ls, f.Name)
		Adduint8(Ctxt, ls, 0)
		Adduint8(Ctxt, ls, 0)
		Adduint8(Ctxt, ls, 0)
	}

	// 4 zeros: the string termination + 3 fields.
	Adduint8(Ctxt, ls, 0)
	// terminate file_names.
	headerend = ls.Size

	Adduint8(Ctxt, ls, 0) // start extended opcode
	dwarf.Uleb128put(dwarfctxt, ls, 1+int64(SysArch.PtrSize))
	Adduint8(Ctxt, ls, dwarf.DW_LNE_set_address)

	pc := s.Value
	line := 1
	file := 1
	Addaddr(Ctxt, ls, s)

	var pcfile Pciter
	var pcline Pciter
	for _, Ctxt.Cursym = range Ctxt.Textp {
		s := Ctxt.Cursym

		epc = s.Value + s.Size
		epcs = s

		dsym := Linklookup(Ctxt, dwarf.InfoPrefix+s.Name, int(s.Version))
		dsym.Attr |= AttrHidden
		dsym.Type = obj.SDWARFINFO
		for _, r := range dsym.R {
			if r.Type == obj.R_DWARFREF && r.Sym.Size == 0 {
				if Buildmode == BuildmodeShared {
					// These type symbols may not be present in BuildmodeShared. Skip.
					continue
				}
				n := nameFromDIESym(r.Sym)
				defgotype(Linklookup(Ctxt, "type."+n, 0))
			}
		}
		funcs = append(funcs, dsym)

		if s.FuncInfo == nil {
			continue
		}

		finddebugruntimepath(s)

		pciterinit(Ctxt, &pcfile, &s.FuncInfo.Pcfile)
		pciterinit(Ctxt, &pcline, &s.FuncInfo.Pcline)
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
				Adduint8(Ctxt, ls, dwarf.DW_LNS_set_file)
				dwarf.Uleb128put(dwarfctxt, ls, int64(pcfile.value))
				file = int(pcfile.value)
			}

			putpclcdelta(dwarfctxt, ls, s.Value+int64(pcline.pc)-pc, int64(pcline.value)-int64(line))

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

	Adduint8(Ctxt, ls, 0) // start extended opcode
	dwarf.Uleb128put(dwarfctxt, ls, 1)
	Adduint8(Ctxt, ls, dwarf.DW_LNE_end_sequence)

	newattr(dwinfo, dwarf.DW_AT_high_pc, dwarf.DW_CLS_ADDRESS, epc+1, epcs)

	setuint32(Ctxt, ls, unit_length_offset, uint32(ls.Size-unitstart))
	setuint32(Ctxt, ls, header_length_offset, uint32(headerend-headerstart))

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

func writeframes(syms []*LSym) []*LSym {
	var dwarfctxt dwarf.Context = dwCtxt{}
	if framesec == nil {
		framesec = Linklookup(Ctxt, ".debug_frame", 0)
	}
	framesec.Type = obj.SDWARFSECT
	framesec.R = framesec.R[:0]
	fs := framesec
	syms = append(syms, fs)

	// Emit the CIE, Section 6.4.1
	cieReserve := uint32(16)
	if haslinkregister() {
		cieReserve = 32
	}
	Adduint32(Ctxt, fs, cieReserve)                            // initial length, must be multiple of thearch.ptrsize
	Adduint32(Ctxt, fs, 0xffffffff)                            // cid.
	Adduint8(Ctxt, fs, 3)                                      // dwarf version (appendix F)
	Adduint8(Ctxt, fs, 0)                                      // augmentation ""
	dwarf.Uleb128put(dwarfctxt, fs, 1)                         // code_alignment_factor
	dwarf.Sleb128put(dwarfctxt, fs, dataAlignmentFactor)       // all CFI offset calculations include multiplication with this factor
	dwarf.Uleb128put(dwarfctxt, fs, int64(Thearch.Dwarfreglr)) // return_address_register

	Adduint8(Ctxt, fs, dwarf.DW_CFA_def_cfa)                   // Set the current frame address..
	dwarf.Uleb128put(dwarfctxt, fs, int64(Thearch.Dwarfregsp)) // ...to use the value in the platform's SP register (defined in l.go)...
	if haslinkregister() {
		dwarf.Uleb128put(dwarfctxt, fs, int64(0)) // ...plus a 0 offset.

		Adduint8(Ctxt, fs, dwarf.DW_CFA_same_value) // The platform's link register is unchanged during the prologue.
		dwarf.Uleb128put(dwarfctxt, fs, int64(Thearch.Dwarfreglr))

		Adduint8(Ctxt, fs, dwarf.DW_CFA_val_offset)                // The previous value...
		dwarf.Uleb128put(dwarfctxt, fs, int64(Thearch.Dwarfregsp)) // ...of the platform's SP register...
		dwarf.Uleb128put(dwarfctxt, fs, int64(0))                  // ...is CFA+0.
	} else {
		dwarf.Uleb128put(dwarfctxt, fs, int64(SysArch.PtrSize)) // ...plus the word size (because the call instruction implicitly adds one word to the frame).

		Adduint8(Ctxt, fs, dwarf.DW_CFA_offset_extended)                             // The previous value...
		dwarf.Uleb128put(dwarfctxt, fs, int64(Thearch.Dwarfreglr))                   // ...of the return address...
		dwarf.Uleb128put(dwarfctxt, fs, int64(-SysArch.PtrSize)/dataAlignmentFactor) // ...is saved at [CFA - (PtrSize/4)].
	}

	// 4 is to exclude the length field.
	pad := int64(cieReserve) + 4 - fs.Size

	if pad < 0 {
		Exitf("dwarf: cieReserve too small by %d bytes.", -pad)
	}

	Addbytes(Ctxt, fs, zeros[:pad])

	var deltaBuf []byte
	var pcsp Pciter
	for _, Ctxt.Cursym = range Ctxt.Textp {
		s := Ctxt.Cursym
		if s.FuncInfo == nil {
			continue
		}

		// Emit a FDE, Section 6.4.1.
		// First build the section contents into a byte buffer.
		deltaBuf = deltaBuf[:0]
		for pciterinit(Ctxt, &pcsp, &s.FuncInfo.Pcsp); pcsp.done == 0; pciternext(&pcsp) {
			nextpc := pcsp.nextpc

			// pciterinit goes up to the end of the function,
			// but DWARF expects us to stop just before the end.
			if int64(nextpc) == s.Size {
				nextpc--
				if nextpc < pcsp.pc {
					continue
				}
			}

			if haslinkregister() {
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
		Adduint32(Ctxt, fs, uint32(4+2*SysArch.PtrSize+len(deltaBuf))) // length (excludes itself)
		if Linkmode == LinkExternal {
			adddwarfref(Ctxt, fs, framesec, 4)
		} else {
			Adduint32(Ctxt, fs, 0) // CIE offset
		}
		Addaddr(Ctxt, fs, s)
		adduintxx(Ctxt, fs, uint64(s.Size), SysArch.PtrSize) // address range
		Addbytes(Ctxt, fs, deltaBuf)
	}
	return syms
}

/*
 *  Walk DWarfDebugInfoEntries, and emit .debug_info
 */
const (
	COMPUNITHEADERSIZE = 4 + 2 + 4 + 1
)

func writeinfo(syms []*LSym, funcs []*LSym) []*LSym {
	if infosec == nil {
		infosec = Linklookup(Ctxt, ".debug_info", 0)
	}
	infosec.R = infosec.R[:0]
	infosec.Type = obj.SDWARFINFO
	infosec.Attr |= AttrReachable
	syms = append(syms, infosec)

	if arangessec == nil {
		arangessec = Linklookup(Ctxt, ".dwarfaranges", 0)
	}
	arangessec.R = arangessec.R[:0]

	var dwarfctxt dwarf.Context = dwCtxt{}

	for compunit := dwroot.Child; compunit != nil; compunit = compunit.Link {
		s := dtolsym(compunit.Sym)

		// Write .debug_info Compilation Unit Header (sec 7.5.1)
		// Fields marked with (*) must be changed for 64-bit dwarf
		// This must match COMPUNITHEADERSIZE above.
		Adduint32(Ctxt, s, 0) // unit_length (*), will be filled in later.
		Adduint16(Ctxt, s, 2) // dwarf version (appendix F)

		// debug_abbrev_offset (*)
		adddwarfref(Ctxt, s, abbrevsym, 4)

		Adduint8(Ctxt, s, uint8(SysArch.PtrSize)) // address_size

		dwarf.Uleb128put(dwarfctxt, s, int64(compunit.Abbrev))
		dwarf.PutAttrs(dwarfctxt, s, compunit.Abbrev, compunit.Attr)

		cu := []*LSym{s}
		if funcs != nil {
			cu = append(cu, funcs...)
			funcs = nil
		}
		cu = putdies(dwarfctxt, cu, compunit.Child)
		var cusize int64
		for _, child := range cu {
			cusize += child.Size
		}
		cusize -= 4 // exclude the length field.
		setuint32(Ctxt, s, 0, uint32(cusize))
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

func writepub(sname string, ispub func(*dwarf.DWDie) bool, syms []*LSym) []*LSym {
	s := Linklookup(Ctxt, sname, 0)
	s.Type = obj.SDWARFSECT
	syms = append(syms, s)

	for compunit := dwroot.Child; compunit != nil; compunit = compunit.Link {
		sectionstart := s.Size
		culength := uint32(getattr(compunit, dwarf.DW_AT_byte_size).Value) + 4

		// Write .debug_pubnames/types	Header (sec 6.1.1)
		Adduint32(Ctxt, s, 0)                          // unit_length (*), will be filled in later.
		Adduint16(Ctxt, s, 2)                          // dwarf version (appendix F)
		adddwarfref(Ctxt, s, dtolsym(compunit.Sym), 4) // debug_info_offset (of the Comp unit Header)
		Adduint32(Ctxt, s, culength)                   // debug_info_length

		for die := compunit.Child; die != nil; die = die.Link {
			if !ispub(die) {
				continue
			}
			dwa := getattr(die, dwarf.DW_AT_name)
			name := dwa.Data.(string)
			if die.Sym == nil {
				fmt.Println("Missing sym for ", name)
			}
			adddwarfref(Ctxt, s, dtolsym(die.Sym), 4)
			Addstring(s, name)
		}

		Adduint32(Ctxt, s, 0)

		setuint32(Ctxt, s, sectionstart, uint32(s.Size-sectionstart)-4) // exclude the length field.
	}

	return syms
}

/*
 *  emit .debug_aranges.  _info must have been written before,
 *  because we need die->offs of dwarf.DW_globals.
 */
func writearanges(syms []*LSym) []*LSym {
	s := Linklookup(Ctxt, ".debug_aranges", 0)
	s.Type = obj.SDWARFSECT
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
		Adduint32(Ctxt, s, unitlength) // unit_length (*)
		Adduint16(Ctxt, s, 2)          // dwarf version (appendix F)

		adddwarfref(Ctxt, s, dtolsym(compunit.Sym), 4)

		Adduint8(Ctxt, s, uint8(SysArch.PtrSize)) // address_size
		Adduint8(Ctxt, s, 0)                      // segment_size
		padding := headersize - (4 + 2 + 4 + 1 + 1)
		for i := 0; i < padding; i++ {
			Adduint8(Ctxt, s, 0)
		}

		Addaddrplus(Ctxt, s, b.Data.(*LSym), b.Value-(b.Data.(*LSym)).Value)
		adduintxx(Ctxt, s, uint64(e.Value-b.Value), SysArch.PtrSize)
		adduintxx(Ctxt, s, 0, SysArch.PtrSize)
		adduintxx(Ctxt, s, 0, SysArch.PtrSize)
	}
	if s.Size > 0 {
		syms = append(syms, s)
	}
	return syms
}

func writegdbscript(syms []*LSym) []*LSym {

	if gdbscript != "" {
		s := Linklookup(Ctxt, ".debug_gdb_scripts", 0)
		s.Type = obj.SDWARFSECT
		syms = append(syms, s)
		Adduint8(Ctxt, s, 1) // magic 1 byte?
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
func dwarfgeneratedebugsyms() {
	if Debug['w'] != 0 { // disable dwarf
		return
	}
	if Debug['s'] != 0 && HEADTYPE != obj.Hdarwin {
		return
	}
	if HEADTYPE == obj.Hplan9 {
		return
	}

	if Linkmode == LinkExternal {
		if !Iself && HEADTYPE != obj.Hdarwin {
			return
		}
	}

	if Debug['v'] != 0 {
		fmt.Fprintf(Bso, "%5.2f dwarf\n", obj.Cputime())
	}

	// For diagnostic messages.
	newattr(&dwtypes, dwarf.DW_AT_name, dwarf.DW_CLS_STRING, int64(len("dwtypes")), "dwtypes")

	// Some types that must exist to define other ones.
	newdie(&dwtypes, dwarf.DW_ABRV_NULLTYPE, "<unspecified>", 0)

	newdie(&dwtypes, dwarf.DW_ABRV_NULLTYPE, "void", 0)
	newdie(&dwtypes, dwarf.DW_ABRV_BARE_PTRTYPE, "unsafe.Pointer", 0)

	die := newdie(&dwtypes, dwarf.DW_ABRV_BASETYPE, "uintptr", 0) // needed for array size
	newattr(die, dwarf.DW_AT_encoding, dwarf.DW_CLS_CONSTANT, dwarf.DW_ATE_unsigned, 0)
	newattr(die, dwarf.DW_AT_byte_size, dwarf.DW_CLS_CONSTANT, int64(SysArch.PtrSize), 0)
	newattr(die, dwarf.DW_AT_go_kind, dwarf.DW_CLS_CONSTANT, obj.KindUintptr, 0)

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
	defgotype(lookup_or_diag("type.runtime._type"))

	defgotype(lookup_or_diag("type.runtime.interfacetype"))
	defgotype(lookup_or_diag("type.runtime.itab"))

	genasmsym(defdwsymb)

	syms := writeabbrev(nil)
	syms, funcs := writelines(syms)
	syms = writeframes(syms)

	synthesizestringtypes(dwtypes.Child)
	synthesizeslicetypes(dwtypes.Child)
	synthesizemaptypes(dwtypes.Child)
	synthesizechantypes(dwtypes.Child)

	reversetree(&dwroot.Child)
	reversetree(&dwtypes.Child)
	reversetree(&dwglobals.Child)

	movetomodule(&dwtypes)
	movetomodule(&dwglobals)

	// Need to reorder symbols so SDWARFINFO is after all SDWARFSECT
	// (but we need to generate dies before writepub)
	infosyms := writeinfo(nil, funcs)

	syms = writepub(".debug_pubnames", ispubname, syms)
	syms = writepub(".debug_pubtypes", ispubtype, syms)
	syms = writearanges(syms)
	syms = writegdbscript(syms)
	syms = append(syms, infosyms...)
	dwarfp = syms[0]
	for i := 1; i < len(syms); i++ {
		syms[i-1].Next = syms[i]
	}
	syms[len(syms)-1].Next = nil
}

/*
 *  Elf.
 */
func dwarfaddshstrings(shstrtab *LSym) {
	if Debug['w'] != 0 { // disable dwarf
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
	if Linkmode == LinkExternal {
		Addstring(shstrtab, elfRelType+".debug_info")
		Addstring(shstrtab, elfRelType+".debug_aranges")
		Addstring(shstrtab, elfRelType+".debug_line")
		Addstring(shstrtab, elfRelType+".debug_frame")
		Addstring(shstrtab, elfRelType+".debug_pubnames")
		Addstring(shstrtab, elfRelType+".debug_pubtypes")
	}
}

// Add section symbols for DWARF debug info.  This is called before
// dwarfaddelfheaders.
func dwarfaddelfsectionsyms() {
	if Debug['w'] != 0 { // disable dwarf
		return
	}
	if Linkmode != LinkExternal {
		return
	}
	sym := Linklookup(Ctxt, ".debug_info", 0)
	putelfsectionsym(sym, sym.Sect.Elfsect.shnum)
	sym = Linklookup(Ctxt, ".debug_abbrev", 0)
	putelfsectionsym(sym, sym.Sect.Elfsect.shnum)
	sym = Linklookup(Ctxt, ".debug_line", 0)
	putelfsectionsym(sym, sym.Sect.Elfsect.shnum)
	sym = Linklookup(Ctxt, ".debug_frame", 0)
	putelfsectionsym(sym, sym.Sect.Elfsect.shnum)
}

/*
 * Windows PE
 */
func dwarfaddpeheaders() {
	if Debug['w'] != 0 { // disable dwarf
		return
	}
	for sect := Segdwarf.Sect; sect != nil; sect = sect.Next {
		h := newPEDWARFSection(sect.Name, int64(sect.Length))
		fileoff := sect.Vaddr - Segdwarf.Vaddr + Segdwarf.Fileoff
		if uint64(h.PointerToRawData) != fileoff {
			Diag("%s.PointerToRawData = %#x, want %#x", sect.Name, h.PointerToRawData, fileoff)
			errorexit()
		}
	}
}
