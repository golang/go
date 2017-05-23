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
	"cmd/internal/obj"
	"fmt"
	"log"
	"os"
	"strings"
)

const infoprefix = "go.dwarf.info."

/*
 * Offsets and sizes of the debug_* sections in the cout file.
 */
var abbrevsym *LSym
var arangessec *LSym
var framesec *LSym
var infosec *LSym
var linesec *LSym

var gdbscript string

/*
 *  Basic I/O
 */
func addrput(s *LSym, addr int64) {
	switch SysArch.PtrSize {
	case 4:
		Adduint32(Ctxt, s, uint32(addr))

	case 8:
		Adduint64(Ctxt, s, uint64(addr))
	}
}

func appendUleb128(b []byte, v uint64) []byte {
	for {
		c := uint8(v & 0x7f)
		v >>= 7
		if v != 0 {
			c |= 0x80
		}
		b = append(b, c)
		if c&0x80 == 0 {
			break
		}
	}
	return b
}

func appendSleb128(b []byte, v int64) []byte {
	for {
		c := uint8(v & 0x7f)
		s := uint8(v & 0x40)
		v >>= 7
		if (v != -1 || s == 0) && (v != 0 || s != 0) {
			c |= 0x80
		}
		b = append(b, c)
		if c&0x80 == 0 {
			break
		}
	}
	return b
}

var encbuf [10]byte

func uleb128put(s *LSym, v int64) {
	b := appendUleb128(encbuf[:0], uint64(v))
	Addbytes(Ctxt, s, b)
}

func sleb128put(s *LSym, v int64) {
	b := appendSleb128(encbuf[:0], v)
	Addbytes(Ctxt, s, b)
}

/*
 * Defining Abbrevs.  This is hardcoded, and there will be
 * only a handful of them.  The DWARF spec places no restriction on
 * the ordering of attributes in the Abbrevs and DIEs, and we will
 * always write them out in the order of declaration in the abbrev.
 */
type DWAttrForm struct {
	attr uint16
	form uint8
}

// Go-specific type attributes.
const (
	DW_AT_go_kind = 0x2900
	DW_AT_go_key  = 0x2901
	DW_AT_go_elem = 0x2902

	DW_AT_internal_location = 253 // params and locals; not emitted
)

// Index into the abbrevs table below.
// Keep in sync with ispubname() and ispubtype() below.
// ispubtype considers >= NULLTYPE public
const (
	DW_ABRV_NULL = iota
	DW_ABRV_COMPUNIT
	DW_ABRV_FUNCTION
	DW_ABRV_VARIABLE
	DW_ABRV_AUTO
	DW_ABRV_PARAM
	DW_ABRV_STRUCTFIELD
	DW_ABRV_FUNCTYPEPARAM
	DW_ABRV_DOTDOTDOT
	DW_ABRV_ARRAYRANGE
	DW_ABRV_NULLTYPE
	DW_ABRV_BASETYPE
	DW_ABRV_ARRAYTYPE
	DW_ABRV_CHANTYPE
	DW_ABRV_FUNCTYPE
	DW_ABRV_IFACETYPE
	DW_ABRV_MAPTYPE
	DW_ABRV_PTRTYPE
	DW_ABRV_BARE_PTRTYPE // only for void*, no DW_AT_type attr to please gdb 6.
	DW_ABRV_SLICETYPE
	DW_ABRV_STRINGTYPE
	DW_ABRV_STRUCTTYPE
	DW_ABRV_TYPEDECL
	DW_NABRV
)

type DWAbbrev struct {
	tag      uint8
	children uint8
	attr     []DWAttrForm
}

var abbrevs = [DW_NABRV]DWAbbrev{
	/* The mandatory DW_ABRV_NULL entry. */
	{0, 0, []DWAttrForm{}},

	/* COMPUNIT */
	{
		DW_TAG_compile_unit,
		DW_CHILDREN_yes,
		[]DWAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_language, DW_FORM_data1},
			{DW_AT_low_pc, DW_FORM_addr},
			{DW_AT_high_pc, DW_FORM_addr},
			{DW_AT_stmt_list, DW_FORM_data4},
			{DW_AT_comp_dir, DW_FORM_string},
		},
	},

	/* FUNCTION */
	{
		DW_TAG_subprogram,
		DW_CHILDREN_yes,
		[]DWAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_low_pc, DW_FORM_addr},
			{DW_AT_high_pc, DW_FORM_addr},
			{DW_AT_external, DW_FORM_flag},
		},
	},

	/* VARIABLE */
	{
		DW_TAG_variable,
		DW_CHILDREN_no,
		[]DWAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_location, DW_FORM_block1},
			{DW_AT_type, DW_FORM_ref_addr},
			{DW_AT_external, DW_FORM_flag},
		},
	},

	/* AUTO */
	{
		DW_TAG_variable,
		DW_CHILDREN_no,
		[]DWAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_location, DW_FORM_block1},
			{DW_AT_type, DW_FORM_ref_addr},
		},
	},

	/* PARAM */
	{
		DW_TAG_formal_parameter,
		DW_CHILDREN_no,
		[]DWAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_location, DW_FORM_block1},
			{DW_AT_type, DW_FORM_ref_addr},
		},
	},

	/* STRUCTFIELD */
	{
		DW_TAG_member,
		DW_CHILDREN_no,
		[]DWAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_data_member_location, DW_FORM_block1},
			{DW_AT_type, DW_FORM_ref_addr},
		},
	},

	/* FUNCTYPEPARAM */
	{
		DW_TAG_formal_parameter,
		DW_CHILDREN_no,

		// No name!
		[]DWAttrForm{
			{DW_AT_type, DW_FORM_ref_addr},
		},
	},

	/* DOTDOTDOT */
	{
		DW_TAG_unspecified_parameters,
		DW_CHILDREN_no,
		[]DWAttrForm{},
	},

	/* ARRAYRANGE */
	{
		DW_TAG_subrange_type,
		DW_CHILDREN_no,

		// No name!
		[]DWAttrForm{
			{DW_AT_type, DW_FORM_ref_addr},
			{DW_AT_count, DW_FORM_udata},
		},
	},

	// Below here are the types considered public by ispubtype
	/* NULLTYPE */
	{
		DW_TAG_unspecified_type,
		DW_CHILDREN_no,
		[]DWAttrForm{
			{DW_AT_name, DW_FORM_string},
		},
	},

	/* BASETYPE */
	{
		DW_TAG_base_type,
		DW_CHILDREN_no,
		[]DWAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_encoding, DW_FORM_data1},
			{DW_AT_byte_size, DW_FORM_data1},
			{DW_AT_go_kind, DW_FORM_data1},
		},
	},

	/* ARRAYTYPE */
	// child is subrange with upper bound
	{
		DW_TAG_array_type,
		DW_CHILDREN_yes,
		[]DWAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_type, DW_FORM_ref_addr},
			{DW_AT_byte_size, DW_FORM_udata},
			{DW_AT_go_kind, DW_FORM_data1},
		},
	},

	/* CHANTYPE */
	{
		DW_TAG_typedef,
		DW_CHILDREN_no,
		[]DWAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_type, DW_FORM_ref_addr},
			{DW_AT_go_kind, DW_FORM_data1},
			{DW_AT_go_elem, DW_FORM_ref_addr},
		},
	},

	/* FUNCTYPE */
	{
		DW_TAG_subroutine_type,
		DW_CHILDREN_yes,
		[]DWAttrForm{
			{DW_AT_name, DW_FORM_string},
			// {DW_AT_type,	DW_FORM_ref_addr},
			{DW_AT_go_kind, DW_FORM_data1},
		},
	},

	/* IFACETYPE */
	{
		DW_TAG_typedef,
		DW_CHILDREN_yes,
		[]DWAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_type, DW_FORM_ref_addr},
			{DW_AT_go_kind, DW_FORM_data1},
		},
	},

	/* MAPTYPE */
	{
		DW_TAG_typedef,
		DW_CHILDREN_no,
		[]DWAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_type, DW_FORM_ref_addr},
			{DW_AT_go_kind, DW_FORM_data1},
			{DW_AT_go_key, DW_FORM_ref_addr},
			{DW_AT_go_elem, DW_FORM_ref_addr},
		},
	},

	/* PTRTYPE */
	{
		DW_TAG_pointer_type,
		DW_CHILDREN_no,
		[]DWAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_type, DW_FORM_ref_addr},
			{DW_AT_go_kind, DW_FORM_data1},
		},
	},

	/* BARE_PTRTYPE */
	{
		DW_TAG_pointer_type,
		DW_CHILDREN_no,
		[]DWAttrForm{
			{DW_AT_name, DW_FORM_string},
		},
	},

	/* SLICETYPE */
	{
		DW_TAG_structure_type,
		DW_CHILDREN_yes,
		[]DWAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_byte_size, DW_FORM_udata},
			{DW_AT_go_kind, DW_FORM_data1},
			{DW_AT_go_elem, DW_FORM_ref_addr},
		},
	},

	/* STRINGTYPE */
	{
		DW_TAG_structure_type,
		DW_CHILDREN_yes,
		[]DWAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_byte_size, DW_FORM_udata},
			{DW_AT_go_kind, DW_FORM_data1},
		},
	},

	/* STRUCTTYPE */
	{
		DW_TAG_structure_type,
		DW_CHILDREN_yes,
		[]DWAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_byte_size, DW_FORM_udata},
			{DW_AT_go_kind, DW_FORM_data1},
		},
	},

	/* TYPEDECL */
	{
		DW_TAG_typedef,
		DW_CHILDREN_no,
		[]DWAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_type, DW_FORM_ref_addr},
		},
	},
}

var dwarfp *LSym

func writeabbrev() *LSym {
	s := Linklookup(Ctxt, ".debug_abbrev", 0)
	s.Type = obj.SDWARFSECT
	abbrevsym = s

	for i := 1; i < DW_NABRV; i++ {
		// See section 7.5.3
		uleb128put(s, int64(i))

		uleb128put(s, int64(abbrevs[i].tag))
		Adduint8(Ctxt, s, abbrevs[i].children)
		for _, f := range abbrevs[i].attr {
			uleb128put(s, int64(f.attr))
			uleb128put(s, int64(f.form))
		}
		uleb128put(s, 0)
		uleb128put(s, 0)
	}

	Adduint8(Ctxt, s, 0)
	return s
}

/*
 * Debugging Information Entries and their attributes.
 */

// For DW_CLS_string and _block, value should contain the length, and
// data the data, for _reference, value is 0 and data is a DWDie* to
// the referenced instance, for all others, value is the whole thing
// and data is null.

type DWAttr struct {
	link  *DWAttr
	atr   uint16 // DW_AT_
	cls   uint8  // DW_CLS_
	value int64
	data  interface{}
}

type DWDie struct {
	abbrev int
	link   *DWDie
	child  *DWDie
	attr   *DWAttr
	sym    *LSym
}

/*
 * Root DIEs for compilation units, types and global variables.
 */
var dwroot DWDie

var dwtypes DWDie

var dwglobals DWDie

func newattr(die *DWDie, attr uint16, cls int, value int64, data interface{}) *DWAttr {
	a := new(DWAttr)
	a.link = die.attr
	die.attr = a
	a.atr = attr
	a.cls = uint8(cls)
	a.value = value
	a.data = data
	return a
}

// Each DIE (except the root ones) has at least 1 attribute: its
// name. getattr moves the desired one to the front so
// frequently searched ones are found faster.
func getattr(die *DWDie, attr uint16) *DWAttr {
	if die.attr.atr == attr {
		return die.attr
	}

	a := die.attr
	b := a.link
	for b != nil {
		if b.atr == attr {
			a.link = b.link
			b.link = die.attr
			die.attr = b
			return b
		}

		a = b
		b = b.link
	}

	return nil
}

// Every DIE has at least a DW_AT_name attribute (but it will only be
// written out if it is listed in the abbrev).
func newdie(parent *DWDie, abbrev int, name string, version int) *DWDie {
	die := new(DWDie)
	die.abbrev = abbrev
	die.link = parent.child
	parent.child = die

	newattr(die, DW_AT_name, DW_CLS_STRING, int64(len(name)), name)

	if name != "" && (abbrev <= DW_ABRV_VARIABLE || abbrev >= DW_ABRV_NULLTYPE) {
		if abbrev != DW_ABRV_VARIABLE || version == 0 {
			die.sym = Linklookup(Ctxt, infoprefix+name, version)
			die.sym.Attr |= AttrHidden
			die.sym.Type = obj.SDWARFINFO
		}
	}

	return die
}

func walktypedef(die *DWDie) *DWDie {
	// Resolve typedef if present.
	if die.abbrev == DW_ABRV_TYPEDECL {
		for attr := die.attr; attr != nil; attr = attr.link {
			if attr.atr == DW_AT_type && attr.cls == DW_CLS_REFERENCE && attr.data != nil {
				return attr.data.(*DWDie)
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
func findchild(die *DWDie, name string) *DWDie {
	var prev *DWDie
	for ; die != prev; prev, die = die, walktypedef(die) {
		for a := die.child; a != nil; a = a.link {
			if name == getattr(a, DW_AT_name).data {
				return a
			}
		}
		continue
	}
	return nil
}

// Used to avoid string allocation when looking up dwarf symbols
var prefixBuf = []byte(infoprefix)

func find(name string) *LSym {
	n := append(prefixBuf, name...)
	// The string allocation below is optimized away because it is only used in a map lookup.
	s := Linkrlookup(Ctxt, string(n), 0)
	prefixBuf = n[:len(infoprefix)]
	return s
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

func newrefattr(die *DWDie, attr uint16, ref *LSym) *DWAttr {
	if ref == nil {
		return nil
	}
	return newattr(die, attr, DW_CLS_REFERENCE, 0, ref)
}

func putattr(s *LSym, abbrev int, form int, cls int, value int64, data interface{}) {
	switch form {
	case DW_FORM_addr: // address
		if Linkmode == LinkExternal {
			value -= (data.(*LSym)).Value
			Addaddrplus(Ctxt, s, data.(*LSym), value)
			break
		}

		addrput(s, value)

	case DW_FORM_block1: // block
		if cls == DW_CLS_ADDRESS {
			Adduint8(Ctxt, s, uint8(1+SysArch.PtrSize))
			Adduint8(Ctxt, s, DW_OP_addr)
			Addaddr(Ctxt, s, data.(*LSym))
			break
		}

		value &= 0xff
		Adduint8(Ctxt, s, uint8(value))
		p := data.([]byte)
		for i := 0; int64(i) < value; i++ {
			Adduint8(Ctxt, s, p[i])
		}

	case DW_FORM_block2: // block
		value &= 0xffff

		Adduint16(Ctxt, s, uint16(value))
		p := data.([]byte)
		for i := 0; int64(i) < value; i++ {
			Adduint8(Ctxt, s, p[i])
		}

	case DW_FORM_block4: // block
		value &= 0xffffffff

		Adduint32(Ctxt, s, uint32(value))
		p := data.([]byte)
		for i := 0; int64(i) < value; i++ {
			Adduint8(Ctxt, s, p[i])
		}

	case DW_FORM_block: // block
		uleb128put(s, value)

		p := data.([]byte)
		for i := 0; int64(i) < value; i++ {
			Adduint8(Ctxt, s, p[i])
		}

	case DW_FORM_data1: // constant
		Adduint8(Ctxt, s, uint8(value))

	case DW_FORM_data2: // constant
		Adduint16(Ctxt, s, uint16(value))

	case DW_FORM_data4: // constant, {line,loclist,mac,rangelist}ptr
		if Linkmode == LinkExternal && cls == DW_CLS_PTR {
			adddwarfref(Ctxt, s, linesec, 4)
			break
		}

		Adduint32(Ctxt, s, uint32(value))

	case DW_FORM_data8: // constant, {line,loclist,mac,rangelist}ptr
		Adduint64(Ctxt, s, uint64(value))

	case DW_FORM_sdata: // constant
		sleb128put(s, value)

	case DW_FORM_udata: // constant
		uleb128put(s, value)

	case DW_FORM_string: // string
		str := data.(string)
		Addstring(s, str)
		for i := int64(len(str)); i < value; i++ {
			Adduint8(Ctxt, s, 0)
		}

	case DW_FORM_flag: // flag
		if value != 0 {
			Adduint8(Ctxt, s, 1)
		} else {
			Adduint8(Ctxt, s, 0)
		}

		// In DWARF 2 (which is what we claim to generate),
	// the ref_addr is the same size as a normal address.
	// In DWARF 3 it is always 32 bits, unless emitting a large
	// (> 4 GB of debug info aka "64-bit") unit, which we don't implement.
	case DW_FORM_ref_addr: // reference to a DIE in the .info section
		if data == nil {
			Diag("dwarf: null reference in %d", abbrev)
			if SysArch.PtrSize == 8 {
				Adduint64(Ctxt, s, 0) // invalid dwarf, gdb will complain.
			} else {
				Adduint32(Ctxt, s, 0) // invalid dwarf, gdb will complain.
			}
		} else {
			dsym := data.(*LSym)
			adddwarfref(Ctxt, s, dsym, SysArch.PtrSize)
		}

	case DW_FORM_ref1, // reference within the compilation unit
		DW_FORM_ref2,      // reference
		DW_FORM_ref4,      // reference
		DW_FORM_ref8,      // reference
		DW_FORM_ref_udata, // reference

		DW_FORM_strp,     // string
		DW_FORM_indirect: // (see Section 7.5.3)
		fallthrough
	default:
		Exitf("dwarf: unsupported attribute form %d / class %d", form, cls)
	}
}

// Note that we can (and do) add arbitrary attributes to a DIE, but
// only the ones actually listed in the Abbrev will be written out.
func putattrs(s *LSym, abbrev int, attr *DWAttr) {
Outer:
	for _, f := range abbrevs[abbrev].attr {
		for ap := attr; ap != nil; ap = ap.link {
			if ap.atr == f.attr {
				putattr(s, abbrev, int(f.form), int(ap.cls), ap.value, ap.data)
				continue Outer
			}
		}

		putattr(s, abbrev, int(f.form), 0, 0, nil)
	}
}

func putdies(prev *LSym, die *DWDie) *LSym {
	for ; die != nil; die = die.link {
		prev = putdie(prev, die)
	}
	Adduint8(Ctxt, prev, 0)
	return prev
}

func putdie(prev *LSym, die *DWDie) *LSym {
	s := die.sym
	if s == nil {
		s = prev
	} else {
		if s.Attr.OnList() {
			log.Fatalf("symbol %s listed multiple times", s.Name)
		}
		s.Attr |= AttrOnList
		prev.Next = s
	}
	uleb128put(s, int64(die.abbrev))
	putattrs(s, die.abbrev, die.attr)
	if abbrevs[die.abbrev].children != 0 {
		return putdies(s, die.child)
	}
	return s
}

func reverselist(list **DWDie) {
	curr := *list
	var prev *DWDie
	for curr != nil {
		var next *DWDie = curr.link
		curr.link = prev
		prev = curr
		curr = next
	}

	*list = prev
}

func reversetree(list **DWDie) {
	reverselist(list)
	for die := *list; die != nil; die = die.link {
		if abbrevs[die.abbrev].children != 0 {
			reversetree(&die.child)
		}
	}
}

func newmemberoffsetattr(die *DWDie, offs int32) {
	var block [20]byte
	b := append(block[:0], DW_OP_plus_uconst)
	b = appendUleb128(b, uint64(offs))
	newattr(die, DW_AT_data_member_location, DW_CLS_BLOCK, int64(len(b)), b)
}

// GDB doesn't like DW_FORM_addr for DW_AT_location, so emit a
// location expression that evals to a const.
func newabslocexprattr(die *DWDie, addr int64, sym *LSym) {
	newattr(die, DW_AT_location, DW_CLS_ADDRESS, addr, sym)
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

func dotypedef(parent *DWDie, name string, def *DWDie) {
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

	def.sym = Linklookup(Ctxt, def.sym.Name+"..def", 0)
	def.sym.Attr |= AttrHidden
	def.sym.Type = obj.SDWARFINFO

	// The typedef entry must be created after the def,
	// so that future lookups will find the typedef instead
	// of the real definition. This hooks the typedef into any
	// circular definition loops, so that gdb can understand them.
	die := newdie(parent, DW_ABRV_TYPEDECL, name, 0)

	newrefattr(die, DW_AT_type, def.sym)
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

	return newtype(gotype).sym
}

func newtype(gotype *LSym) *DWDie {
	name := gotype.Name[5:] // could also decode from Type.string
	kind := decodetype_kind(gotype)
	bytesize := decodetype_size(gotype)

	var die *DWDie
	switch kind {
	case obj.KindBool:
		die = newdie(&dwtypes, DW_ABRV_BASETYPE, name, 0)
		newattr(die, DW_AT_encoding, DW_CLS_CONSTANT, DW_ATE_boolean, 0)
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0)

	case obj.KindInt,
		obj.KindInt8,
		obj.KindInt16,
		obj.KindInt32,
		obj.KindInt64:
		die = newdie(&dwtypes, DW_ABRV_BASETYPE, name, 0)
		newattr(die, DW_AT_encoding, DW_CLS_CONSTANT, DW_ATE_signed, 0)
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0)

	case obj.KindUint,
		obj.KindUint8,
		obj.KindUint16,
		obj.KindUint32,
		obj.KindUint64,
		obj.KindUintptr:
		die = newdie(&dwtypes, DW_ABRV_BASETYPE, name, 0)
		newattr(die, DW_AT_encoding, DW_CLS_CONSTANT, DW_ATE_unsigned, 0)
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0)

	case obj.KindFloat32,
		obj.KindFloat64:
		die = newdie(&dwtypes, DW_ABRV_BASETYPE, name, 0)
		newattr(die, DW_AT_encoding, DW_CLS_CONSTANT, DW_ATE_float, 0)
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0)

	case obj.KindComplex64,
		obj.KindComplex128:
		die = newdie(&dwtypes, DW_ABRV_BASETYPE, name, 0)
		newattr(die, DW_AT_encoding, DW_CLS_CONSTANT, DW_ATE_complex_float, 0)
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0)

	case obj.KindArray:
		die = newdie(&dwtypes, DW_ABRV_ARRAYTYPE, name, 0)
		dotypedef(&dwtypes, name, die)
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0)
		s := decodetype_arrayelem(gotype)
		newrefattr(die, DW_AT_type, defgotype(s))
		fld := newdie(die, DW_ABRV_ARRAYRANGE, "range", 0)

		// use actual length not upper bound; correct for 0-length arrays.
		newattr(fld, DW_AT_count, DW_CLS_CONSTANT, decodetype_arraylen(gotype), 0)

		newrefattr(fld, DW_AT_type, mustFind("uintptr"))

	case obj.KindChan:
		die = newdie(&dwtypes, DW_ABRV_CHANTYPE, name, 0)
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0)
		s := decodetype_chanelem(gotype)
		newrefattr(die, DW_AT_go_elem, defgotype(s))
		// Save elem type for synthesizechantypes. We could synthesize here
		// but that would change the order of DIEs we output.
		newrefattr(die, DW_AT_type, s)

	case obj.KindFunc:
		die = newdie(&dwtypes, DW_ABRV_FUNCTYPE, name, 0)
		dotypedef(&dwtypes, name, die)
		newrefattr(die, DW_AT_type, mustFind("void"))
		nfields := decodetype_funcincount(gotype)
		var fld *DWDie
		var s *LSym
		for i := 0; i < nfields; i++ {
			s = decodetype_funcintype(gotype, i)
			fld = newdie(die, DW_ABRV_FUNCTYPEPARAM, s.Name[5:], 0)
			newrefattr(fld, DW_AT_type, defgotype(s))
		}

		if decodetype_funcdotdotdot(gotype) {
			newdie(die, DW_ABRV_DOTDOTDOT, "...", 0)
		}
		nfields = decodetype_funcoutcount(gotype)
		for i := 0; i < nfields; i++ {
			s = decodetype_funcouttype(gotype, i)
			fld = newdie(die, DW_ABRV_FUNCTYPEPARAM, s.Name[5:], 0)
			newrefattr(fld, DW_AT_type, defptrto(defgotype(s)))
		}

	case obj.KindInterface:
		die = newdie(&dwtypes, DW_ABRV_IFACETYPE, name, 0)
		dotypedef(&dwtypes, name, die)
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0)
		nfields := int(decodetype_ifacemethodcount(gotype))
		var s *LSym
		if nfields == 0 {
			s = lookup_or_diag("type.runtime.eface")
		} else {
			s = lookup_or_diag("type.runtime.iface")
		}
		newrefattr(die, DW_AT_type, defgotype(s))

	case obj.KindMap:
		die = newdie(&dwtypes, DW_ABRV_MAPTYPE, name, 0)
		s := decodetype_mapkey(gotype)
		newrefattr(die, DW_AT_go_key, defgotype(s))
		s = decodetype_mapvalue(gotype)
		newrefattr(die, DW_AT_go_elem, defgotype(s))
		// Save gotype for use in synthesizemaptypes. We could synthesize here,
		// but that would change the order of the DIEs.
		newrefattr(die, DW_AT_type, gotype)

	case obj.KindPtr:
		die = newdie(&dwtypes, DW_ABRV_PTRTYPE, name, 0)
		dotypedef(&dwtypes, name, die)
		s := decodetype_ptrelem(gotype)
		newrefattr(die, DW_AT_type, defgotype(s))

	case obj.KindSlice:
		die = newdie(&dwtypes, DW_ABRV_SLICETYPE, name, 0)
		dotypedef(&dwtypes, name, die)
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0)
		s := decodetype_arrayelem(gotype)
		elem := defgotype(s)
		newrefattr(die, DW_AT_go_elem, elem)

	case obj.KindString:
		die = newdie(&dwtypes, DW_ABRV_STRINGTYPE, name, 0)
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0)

	case obj.KindStruct:
		die = newdie(&dwtypes, DW_ABRV_STRUCTTYPE, name, 0)
		dotypedef(&dwtypes, name, die)
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0)
		nfields := decodetype_structfieldcount(gotype)
		var f string
		var fld *DWDie
		var s *LSym
		for i := 0; i < nfields; i++ {
			f = decodetype_structfieldname(gotype, i)
			s = decodetype_structfieldtype(gotype, i)
			if f == "" {
				f = s.Name[5:] // skip "type."
			}
			fld = newdie(die, DW_ABRV_STRUCTFIELD, f, 0)
			newrefattr(fld, DW_AT_type, defgotype(s))
			newmemberoffsetattr(fld, int32(decodetype_structfieldoffs(gotype, i)))
		}

	case obj.KindUnsafePointer:
		die = newdie(&dwtypes, DW_ABRV_BARE_PTRTYPE, name, 0)

	default:
		Diag("dwarf: definition of unknown kind %d: %s", kind, gotype.Name)
		die = newdie(&dwtypes, DW_ABRV_TYPEDECL, name, 0)
		newrefattr(die, DW_AT_type, mustFind("<unspecified>"))
	}

	newattr(die, DW_AT_go_kind, DW_CLS_CONSTANT, int64(kind), 0)

	if _, ok := prototypedies[gotype.Name]; ok {
		prototypedies[gotype.Name] = die
	}

	return die
}

func nameFromDIESym(dwtype *LSym) string {
	return strings.TrimSuffix(dwtype.Name[len(infoprefix):], "..def")
}

// Find or construct *T given T.
func defptrto(dwtype *LSym) *LSym {
	ptrname := "*" + nameFromDIESym(dwtype)
	die := find(ptrname)
	if die == nil {
		pdie := newdie(&dwtypes, DW_ABRV_PTRTYPE, ptrname, 0)
		newrefattr(pdie, DW_AT_type, dwtype)
		return pdie.sym
	}

	return die
}

// Copies src's children into dst. Copies attributes by value.
// DWAttr.data is copied as pointer only. If except is one of
// the top-level children, it will not be copied.
func copychildrenexcept(dst *DWDie, src *DWDie, except *DWDie) {
	for src = src.child; src != nil; src = src.link {
		if src == except {
			continue
		}
		c := newdie(dst, src.abbrev, getattr(src, DW_AT_name).data.(string), 0)
		for a := src.attr; a != nil; a = a.link {
			newattr(c, a.atr, int(a.cls), a.value, a.data)
		}
		copychildrenexcept(c, src, nil)
	}

	reverselist(&dst.child)
}

func copychildren(dst *DWDie, src *DWDie) {
	copychildrenexcept(dst, src, nil)
}

// Search children (assumed to have DW_TAG_member) for the one named
// field and set its DW_AT_type to dwtype
func substitutetype(structdie *DWDie, field string, dwtype *LSym) {
	child := findchild(structdie, field)
	if child == nil {
		Exitf("dwarf substitutetype: %s does not have member %s",
			getattr(structdie, DW_AT_name).data, field)
		return
	}

	a := getattr(child, DW_AT_type)
	if a != nil {
		a.data = dwtype
	} else {
		newrefattr(child, DW_AT_type, dwtype)
	}
}

func findprotodie(name string) *DWDie {
	die, ok := prototypedies[name]
	if ok && die == nil {
		defgotype(lookup_or_diag(name))
		die = prototypedies[name]
	}
	return die
}

func synthesizestringtypes(die *DWDie) {
	prototype := walktypedef(findprotodie("type.runtime.stringStructDWARF"))
	if prototype == nil {
		return
	}

	for ; die != nil; die = die.link {
		if die.abbrev != DW_ABRV_STRINGTYPE {
			continue
		}
		copychildren(die, prototype)
	}
}

func synthesizeslicetypes(die *DWDie) {
	prototype := walktypedef(findprotodie("type.runtime.slice"))
	if prototype == nil {
		return
	}

	for ; die != nil; die = die.link {
		if die.abbrev != DW_ABRV_SLICETYPE {
			continue
		}
		copychildren(die, prototype)
		elem := getattr(die, DW_AT_go_elem).data.(*LSym)
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

func mkinternaltype(abbrev int, typename, keyname, valname string, f func(*DWDie)) *LSym {
	name := mkinternaltypename(typename, keyname, valname)
	symname := infoprefix + name
	s := Linkrlookup(Ctxt, symname, 0)
	if s != nil {
		return s
	}
	die := newdie(&dwtypes, abbrev, name, 0)
	f(die)
	return die.sym
}

func synthesizemaptypes(die *DWDie) {
	hash := walktypedef(findprotodie("type.runtime.hmap"))
	bucket := walktypedef(findprotodie("type.runtime.bmap"))

	if hash == nil {
		return
	}

	for ; die != nil; die = die.link {
		if die.abbrev != DW_ABRV_MAPTYPE {
			continue
		}
		gotype := getattr(die, DW_AT_type).data.(*LSym)
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
		dwhks := mkinternaltype(DW_ABRV_ARRAYTYPE, "[]key", keyname, "", func(dwhk *DWDie) {
			newattr(dwhk, DW_AT_byte_size, DW_CLS_CONSTANT, BucketSize*keysize, 0)
			t := keytype
			if indirect_key {
				t = defptrto(keytype)
			}
			newrefattr(dwhk, DW_AT_type, t)
			fld := newdie(dwhk, DW_ABRV_ARRAYRANGE, "size", 0)
			newattr(fld, DW_AT_count, DW_CLS_CONSTANT, BucketSize, 0)
			newrefattr(fld, DW_AT_type, mustFind("uintptr"))
		})

		// Construct type to represent an array of BucketSize values
		valname := nameFromDIESym(valtype)
		dwhvs := mkinternaltype(DW_ABRV_ARRAYTYPE, "[]val", valname, "", func(dwhv *DWDie) {
			newattr(dwhv, DW_AT_byte_size, DW_CLS_CONSTANT, BucketSize*valsize, 0)
			t := valtype
			if indirect_val {
				t = defptrto(valtype)
			}
			newrefattr(dwhv, DW_AT_type, t)
			fld := newdie(dwhv, DW_ABRV_ARRAYRANGE, "size", 0)
			newattr(fld, DW_AT_count, DW_CLS_CONSTANT, BucketSize, 0)
			newrefattr(fld, DW_AT_type, mustFind("uintptr"))
		})

		// Construct bucket<K,V>
		dwhbs := mkinternaltype(DW_ABRV_STRUCTTYPE, "bucket", keyname, valname, func(dwhb *DWDie) {
			// Copy over all fields except the field "data" from the generic
			// bucket. "data" will be replaced with keys/values below.
			copychildrenexcept(dwhb, bucket, findchild(bucket, "data"))

			fld := newdie(dwhb, DW_ABRV_STRUCTFIELD, "keys", 0)
			newrefattr(fld, DW_AT_type, dwhks)
			newmemberoffsetattr(fld, BucketSize)
			fld = newdie(dwhb, DW_ABRV_STRUCTFIELD, "values", 0)
			newrefattr(fld, DW_AT_type, dwhvs)
			newmemberoffsetattr(fld, BucketSize+BucketSize*int32(keysize))
			fld = newdie(dwhb, DW_ABRV_STRUCTFIELD, "overflow", 0)
			newrefattr(fld, DW_AT_type, defptrto(dwhb.sym))
			newmemberoffsetattr(fld, BucketSize+BucketSize*(int32(keysize)+int32(valsize)))
			if SysArch.RegSize > SysArch.PtrSize {
				fld = newdie(dwhb, DW_ABRV_STRUCTFIELD, "pad", 0)
				newrefattr(fld, DW_AT_type, mustFind("uintptr"))
				newmemberoffsetattr(fld, BucketSize+BucketSize*(int32(keysize)+int32(valsize))+int32(SysArch.PtrSize))
			}

			newattr(dwhb, DW_AT_byte_size, DW_CLS_CONSTANT, BucketSize+BucketSize*keysize+BucketSize*valsize+int64(SysArch.RegSize), 0)
		})

		// Construct hash<K,V>
		dwhs := mkinternaltype(DW_ABRV_STRUCTTYPE, "hash", keyname, valname, func(dwh *DWDie) {
			copychildren(dwh, hash)
			substitutetype(dwh, "buckets", defptrto(dwhbs))
			substitutetype(dwh, "oldbuckets", defptrto(dwhbs))
			newattr(dwh, DW_AT_byte_size, DW_CLS_CONSTANT, getattr(hash, DW_AT_byte_size).value, nil)
		})

		// make map type a pointer to hash<K,V>
		newrefattr(die, DW_AT_type, defptrto(dwhs))
	}
}

func synthesizechantypes(die *DWDie) {
	sudog := walktypedef(findprotodie("type.runtime.sudog"))
	waitq := walktypedef(findprotodie("type.runtime.waitq"))
	hchan := walktypedef(findprotodie("type.runtime.hchan"))
	if sudog == nil || waitq == nil || hchan == nil {
		return
	}

	sudogsize := int(getattr(sudog, DW_AT_byte_size).value)

	for ; die != nil; die = die.link {
		if die.abbrev != DW_ABRV_CHANTYPE {
			continue
		}
		elemgotype := getattr(die, DW_AT_type).data.(*LSym)
		elemsize := decodetype_size(elemgotype)
		elemname := elemgotype.Name[5:]
		elemtype := walksymtypedef(defgotype(elemgotype))

		// sudog<T>
		dwss := mkinternaltype(DW_ABRV_STRUCTTYPE, "sudog", elemname, "", func(dws *DWDie) {
			copychildren(dws, sudog)
			substitutetype(dws, "elem", elemtype)
			if elemsize > 8 {
				elemsize -= 8
			} else {
				elemsize = 0
			}
			newattr(dws, DW_AT_byte_size, DW_CLS_CONSTANT, int64(sudogsize)+elemsize, nil)
		})

		// waitq<T>
		dwws := mkinternaltype(DW_ABRV_STRUCTTYPE, "waitq", elemname, "", func(dww *DWDie) {

			copychildren(dww, waitq)
			substitutetype(dww, "first", defptrto(dwss))
			substitutetype(dww, "last", defptrto(dwss))
			newattr(dww, DW_AT_byte_size, DW_CLS_CONSTANT, getattr(waitq, DW_AT_byte_size).value, nil)
		})

		// hchan<T>
		dwhs := mkinternaltype(DW_ABRV_STRUCTTYPE, "hchan", elemname, "", func(dwh *DWDie) {
			copychildren(dwh, hchan)
			substitutetype(dwh, "recvq", dwws)
			substitutetype(dwh, "sendq", dwws)
			newattr(dwh, DW_AT_byte_size, DW_CLS_CONSTANT, getattr(hchan, DW_AT_byte_size).value, nil)
		})

		newrefattr(die, DW_AT_type, defptrto(dwhs))
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

	var dv *DWDie

	var dt *LSym
	switch t {
	default:
		return

	case 'd', 'b', 'D', 'B':
		dv = newdie(&dwglobals, DW_ABRV_VARIABLE, s, ver)
		newabslocexprattr(dv, v, sym)
		if ver == 0 {
			newattr(dv, DW_AT_external, DW_CLS_FLAG, 1, 0)
		}
		fallthrough

	case 'a', 'p':
		dt = defgotype(gotype)
	}

	if dv != nil {
		newrefattr(dv, DW_AT_type, dt)
	}
}

func movetomodule(parent *DWDie) {
	die := dwroot.child.child
	for die.link != nil {
		die = die.link
	}
	die.link = parent.child
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

func putpclcdelta(s *LSym, delta_pc int64, delta_lc int64) {
	if LINE_BASE <= delta_lc && delta_lc < LINE_BASE+LINE_RANGE {
		var opcode int64 = OPCODE_BASE + (delta_lc - LINE_BASE) + (LINE_RANGE * delta_pc)
		if OPCODE_BASE <= opcode && opcode < 256 {
			Adduint8(Ctxt, s, uint8(opcode))
			return
		}
	}

	if delta_pc != 0 {
		Adduint8(Ctxt, s, DW_LNS_advance_pc)
		sleb128put(s, delta_pc)
	}

	Adduint8(Ctxt, s, DW_LNS_advance_line)
	sleb128put(s, delta_lc)
	Adduint8(Ctxt, s, DW_LNS_copy)
}

func newcfaoffsetattr(die *DWDie, offs int32) {
	var block [20]byte
	b := append(block[:0], DW_OP_call_frame_cfa)

	if offs != 0 {
		b = append(b, DW_OP_consts)
		b = appendSleb128(b, int64(offs))
		b = append(b, DW_OP_plus)
	}

	newattr(die, DW_AT_location, DW_CLS_BLOCK, int64(len(b)), b)
}

func mkvarname(name string, da int) string {
	buf := fmt.Sprintf("%s#%d", name, da)
	n := buf
	return n
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

func writelines(prev *LSym) *LSym {
	if linesec == nil {
		linesec = Linklookup(Ctxt, ".debug_line", 0)
	}
	linesec.Type = obj.SDWARFSECT
	linesec.R = linesec.R[:0]

	ls := linesec
	prev.Next = ls

	unitstart := int64(-1)
	headerstart := int64(-1)
	headerend := int64(-1)
	epc := int64(0)
	var epcs *LSym
	var dwinfo *DWDie

	lang := DW_LANG_Go

	s := Ctxt.Textp[0]

	dwinfo = newdie(&dwroot, DW_ABRV_COMPUNIT, "go", 0)
	newattr(dwinfo, DW_AT_language, DW_CLS_CONSTANT, int64(lang), 0)
	newattr(dwinfo, DW_AT_stmt_list, DW_CLS_PTR, 0, 0)
	newattr(dwinfo, DW_AT_low_pc, DW_CLS_ADDRESS, s.Value, s)
	// OS X linker requires compilation dir or absolute path in comp unit name to output debug info.
	compDir := getCompilationDir()
	newattr(dwinfo, DW_AT_comp_dir, DW_CLS_STRING, int64(len(compDir)), compDir)

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
	uleb128put(ls, 1+int64(SysArch.PtrSize))
	Adduint8(Ctxt, ls, DW_LNE_set_address)

	pc := s.Value
	line := 1
	file := 1
	if Linkmode == LinkExternal {
		Addaddr(Ctxt, ls, s)
	} else {
		addrput(ls, pc)
	}

	var pcfile Pciter
	var pcline Pciter
	for _, Ctxt.Cursym = range Ctxt.Textp {
		s := Ctxt.Cursym

		dwfunc := newdie(dwinfo, DW_ABRV_FUNCTION, s.Name, int(s.Version))
		newattr(dwfunc, DW_AT_low_pc, DW_CLS_ADDRESS, s.Value, s)
		epc = s.Value + s.Size
		epcs = s
		newattr(dwfunc, DW_AT_high_pc, DW_CLS_ADDRESS, epc, s)
		if s.Version == 0 {
			newattr(dwfunc, DW_AT_external, DW_CLS_FLAG, 1, 0)
		}

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
				Adduint8(Ctxt, ls, DW_LNS_set_file)
				uleb128put(ls, int64(pcfile.value))
				file = int(pcfile.value)
			}

			putpclcdelta(ls, s.Value+int64(pcline.pc)-pc, int64(pcline.value)-int64(line))

			pc = s.Value + int64(pcline.pc)
			line = int(pcline.value)
			if pcfile.nextpc < pcline.nextpc {
				epc = int64(pcfile.nextpc)
			} else {
				epc = int64(pcline.nextpc)
			}
			epc += s.Value
		}

		var (
			dt, da int
			offs   int64
		)
		for _, a := range s.FuncInfo.Autom {
			switch a.Name {
			case obj.A_AUTO:
				dt = DW_ABRV_AUTO
				offs = int64(a.Aoffset)
				if !haslinkregister() {
					offs -= int64(SysArch.PtrSize)
				}
				if obj.Framepointer_enabled(obj.Getgoos(), obj.Getgoarch()) {
					// The frame pointer is saved
					// between the CFA and the
					// autos.
					offs -= int64(SysArch.PtrSize)
				}

			case obj.A_PARAM:
				dt = DW_ABRV_PARAM
				offs = int64(a.Aoffset) + Ctxt.FixedFrameSize()

			default:
				continue
			}

			if strings.Contains(a.Asym.Name, ".autotmp_") {
				continue
			}
			var n string
			if findchild(dwfunc, a.Asym.Name) != nil {
				n = mkvarname(a.Asym.Name, da)
			} else {
				n = a.Asym.Name
			}

			// Drop the package prefix from locals and arguments.
			if i := strings.LastIndex(n, "."); i >= 0 {
				n = n[i+1:]
			}

			dwvar := newdie(dwfunc, dt, n, 0)
			newcfaoffsetattr(dwvar, int32(offs))
			newrefattr(dwvar, DW_AT_type, defgotype(a.Gotype))

			// push dwvar down dwfunc->child to preserve order
			newattr(dwvar, DW_AT_internal_location, DW_CLS_CONSTANT, offs, nil)

			dwfunc.child = dwvar.link // take dwvar out from the top of the list
			dws := &dwfunc.child
			for ; *dws != nil; dws = &(*dws).link {
				if offs > getattr(*dws, DW_AT_internal_location).value {
					break
				}
			}
			dwvar.link = *dws
			*dws = dwvar

			da++
		}
	}

	Adduint8(Ctxt, ls, 0) // start extended opcode
	uleb128put(ls, 1)
	Adduint8(Ctxt, ls, DW_LNE_end_sequence)

	newattr(dwinfo, DW_AT_high_pc, DW_CLS_ADDRESS, epc+1, epcs)

	setuint32(Ctxt, ls, unit_length_offset, uint32(ls.Size-unitstart))
	setuint32(Ctxt, ls, header_length_offset, uint32(headerend-headerstart))

	return ls
}

/*
 *  Emit .debug_frame
 */
const (
	dataAlignmentFactor = -4
)

// appendPCDeltaCFA appends per-PC CFA deltas to b and returns the final slice.
func appendPCDeltaCFA(b []byte, deltapc, cfa int64) []byte {
	b = append(b, DW_CFA_def_cfa_offset_sf)
	b = appendSleb128(b, cfa/dataAlignmentFactor)

	switch {
	case deltapc < 0x40:
		b = append(b, uint8(DW_CFA_advance_loc+deltapc))
	case deltapc < 0x100:
		b = append(b, DW_CFA_advance_loc1)
		b = append(b, uint8(deltapc))
	case deltapc < 0x10000:
		b = append(b, DW_CFA_advance_loc2)
		b = Thearch.Append16(b, uint16(deltapc))
	default:
		b = append(b, DW_CFA_advance_loc4)
		b = Thearch.Append32(b, uint32(deltapc))
	}
	return b
}

func writeframes(prev *LSym) *LSym {
	if framesec == nil {
		framesec = Linklookup(Ctxt, ".debug_frame", 0)
	}
	framesec.Type = obj.SDWARFSECT
	framesec.R = framesec.R[:0]
	fs := framesec
	prev.Next = fs

	// Emit the CIE, Section 6.4.1
	cieReserve := uint32(16)
	if haslinkregister() {
		cieReserve = 32
	}
	Adduint32(Ctxt, fs, cieReserve)           // initial length, must be multiple of pointer size
	Adduint32(Ctxt, fs, 0xffffffff)           // cid.
	Adduint8(Ctxt, fs, 3)                     // dwarf version (appendix F)
	Adduint8(Ctxt, fs, 0)                     // augmentation ""
	uleb128put(fs, 1)                         // code_alignment_factor
	sleb128put(fs, dataAlignmentFactor)       // all CFI offset calculations include multiplication with this factor
	uleb128put(fs, int64(Thearch.Dwarfreglr)) // return_address_register

	Adduint8(Ctxt, fs, DW_CFA_def_cfa)        // Set the current frame address..
	uleb128put(fs, int64(Thearch.Dwarfregsp)) // ...to use the value in the platform's SP register (defined in l.go)...
	if haslinkregister() {
		uleb128put(fs, int64(0)) // ...plus a 0 offset.

		Adduint8(Ctxt, fs, DW_CFA_same_value) // The platform's link register is unchanged during the prologue.
		uleb128put(fs, int64(Thearch.Dwarfreglr))

		Adduint8(Ctxt, fs, DW_CFA_val_offset)     // The previous value...
		uleb128put(fs, int64(Thearch.Dwarfregsp)) // ...of the platform's SP register...
		uleb128put(fs, int64(0))                  // ...is CFA+0.
	} else {
		uleb128put(fs, int64(SysArch.PtrSize)) // ...plus the word size (because the call instruction implicitly adds one word to the frame).

		Adduint8(Ctxt, fs, DW_CFA_offset_extended)                  // The previous value...
		uleb128put(fs, int64(Thearch.Dwarfreglr))                   // ...of the return address...
		uleb128put(fs, int64(-SysArch.PtrSize)/dataAlignmentFactor) // ...is saved at [CFA - (PtrSize/4)].
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
					deltaBuf = append(deltaBuf, DW_CFA_offset_extended_sf)
					deltaBuf = appendUleb128(deltaBuf, uint64(Thearch.Dwarfreglr))
					deltaBuf = appendSleb128(deltaBuf, -int64(pcsp.value)/dataAlignmentFactor)
				} else {
					// The return address is restored into the link register
					// when a stack frame has been de-allocated.
					deltaBuf = append(deltaBuf, DW_CFA_same_value)
					deltaBuf = appendUleb128(deltaBuf, uint64(Thearch.Dwarfreglr))
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
		addrput(fs, s.Size) // address range
		Addbytes(Ctxt, fs, deltaBuf)
	}
	return fs
}

/*
 *  Walk DWarfDebugInfoEntries, and emit .debug_info
 */
const (
	COMPUNITHEADERSIZE = 4 + 2 + 4 + 1
)

func writeinfo(prev *LSym) *LSym {
	if infosec == nil {
		infosec = Linklookup(Ctxt, ".debug_info", 0)
	}
	infosec.R = infosec.R[:0]
	infosec.Type = obj.SDWARFINFO
	infosec.Attr |= AttrReachable
	prev.Next, prev = infosec, infosec

	if arangessec == nil {
		arangessec = Linklookup(Ctxt, ".dwarfaranges", 0)
	}
	arangessec.R = arangessec.R[:0]

	for compunit := dwroot.child; compunit != nil; compunit = compunit.link {
		s := compunit.sym
		prev.Next, prev = s, s

		// Write .debug_info Compilation Unit Header (sec 7.5.1)
		// Fields marked with (*) must be changed for 64-bit dwarf
		// This must match COMPUNITHEADERSIZE above.
		Adduint32(Ctxt, s, 0) // unit_length (*), will be filled in later.
		Adduint16(Ctxt, s, 2) // dwarf version (appendix F)

		// debug_abbrev_offset (*)
		adddwarfref(Ctxt, s, abbrevsym, 4)

		Adduint8(Ctxt, s, uint8(SysArch.PtrSize)) // address_size

		prev = putdie(prev, compunit)
		cusize := s.Size - 4 // exclude the length field.
		for child := s.Next; child != nil; child = child.Next {
			cusize += child.Size
		}

		setuint32(Ctxt, s, 0, uint32(cusize))
		newattr(compunit, DW_AT_byte_size, DW_CLS_CONSTANT, cusize, 0)
	}
	return prev
}

/*
 *  Emit .debug_pubnames/_types.  _info must have been written before,
 *  because we need die->offs and infoo/infosize;
 */
func ispubname(die *DWDie) bool {
	switch die.abbrev {
	case DW_ABRV_FUNCTION, DW_ABRV_VARIABLE:
		a := getattr(die, DW_AT_external)
		return a != nil && a.value != 0
	}

	return false
}

func ispubtype(die *DWDie) bool {
	return die.abbrev >= DW_ABRV_NULLTYPE
}

func writepub(sname string, ispub func(*DWDie) bool, prev *LSym) *LSym {
	s := Linklookup(Ctxt, sname, 0)
	s.Type = obj.SDWARFSECT
	prev.Next = s

	for compunit := dwroot.child; compunit != nil; compunit = compunit.link {
		sectionstart := s.Size
		culength := uint32(getattr(compunit, DW_AT_byte_size).value) + 4

		// Write .debug_pubnames/types	Header (sec 6.1.1)
		Adduint32(Ctxt, s, 0)                 // unit_length (*), will be filled in later.
		Adduint16(Ctxt, s, 2)                 // dwarf version (appendix F)
		adddwarfref(Ctxt, s, compunit.sym, 4) // debug_info_offset (of the Comp unit Header)
		Adduint32(Ctxt, s, culength)          // debug_info_length

		for die := compunit.child; die != nil; die = die.link {
			if !ispub(die) {
				continue
			}
			dwa := getattr(die, DW_AT_name)
			name := dwa.data.(string)
			if die.sym == nil {
				fmt.Println("Missing sym for ", name)
			}
			adddwarfref(Ctxt, s, die.sym, 4)
			Addstring(s, name)
		}

		Adduint32(Ctxt, s, 0)

		setuint32(Ctxt, s, sectionstart, uint32(s.Size-sectionstart)-4) // exclude the length field.
	}

	return s
}

/*
 *  emit .debug_aranges.  _info must have been written before,
 *  because we need die->offs of dw_globals.
 */
func writearanges(prev *LSym) *LSym {
	s := Linklookup(Ctxt, ".debug_aranges", 0)
	s.Type = obj.SDWARFSECT
	// The first tuple is aligned to a multiple of the size of a single tuple
	// (twice the size of an address)
	headersize := int(Rnd(4+2+4+1+1, int64(SysArch.PtrSize*2))) // don't count unit_length field itself

	for compunit := dwroot.child; compunit != nil; compunit = compunit.link {
		b := getattr(compunit, DW_AT_low_pc)
		if b == nil {
			continue
		}
		e := getattr(compunit, DW_AT_high_pc)
		if e == nil {
			continue
		}

		// Write .debug_aranges	 Header + entry	 (sec 6.1.2)
		unitlength := uint32(headersize) + 4*uint32(SysArch.PtrSize) - 4
		Adduint32(Ctxt, s, unitlength) // unit_length (*)
		Adduint16(Ctxt, s, 2)          // dwarf version (appendix F)

		adddwarfref(Ctxt, s, compunit.sym, 4)

		Adduint8(Ctxt, s, uint8(SysArch.PtrSize)) // address_size
		Adduint8(Ctxt, s, 0)                      // segment_size
		padding := headersize - (4 + 2 + 4 + 1 + 1)
		for i := 0; i < padding; i++ {
			Adduint8(Ctxt, s, 0)
		}

		Addaddrplus(Ctxt, s, b.data.(*LSym), b.value-(b.data.(*LSym)).Value)
		addrput(s, e.value-b.value)
		addrput(s, 0)
		addrput(s, 0)
	}
	if s.Size > 0 {
		prev.Next = s
		prev = s
	}
	return prev
}

func writegdbscript(prev *LSym) *LSym {

	if gdbscript != "" {
		s := Linklookup(Ctxt, ".debug_gdb_scripts", 0)
		s.Type = obj.SDWARFSECT
		prev.Next = s
		prev = s
		Adduint8(Ctxt, s, 1) // magic 1 byte?
		Addstring(s, gdbscript)
	}

	return prev
}

var prototypedies map[string]*DWDie

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
	newattr(&dwtypes, DW_AT_name, DW_CLS_STRING, int64(len("dwtypes")), "dwtypes")

	// Some types that must exist to define other ones.
	newdie(&dwtypes, DW_ABRV_NULLTYPE, "<unspecified>", 0)

	newdie(&dwtypes, DW_ABRV_NULLTYPE, "void", 0)
	newdie(&dwtypes, DW_ABRV_BARE_PTRTYPE, "unsafe.Pointer", 0)

	die := newdie(&dwtypes, DW_ABRV_BASETYPE, "uintptr", 0) // needed for array size
	newattr(die, DW_AT_encoding, DW_CLS_CONSTANT, DW_ATE_unsigned, 0)
	newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, int64(SysArch.PtrSize), 0)
	newattr(die, DW_AT_go_kind, DW_CLS_CONSTANT, obj.KindUintptr, 0)

	// Prototypes needed for type synthesis.
	prototypedies = map[string]*DWDie{
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

	dwarfp = writeabbrev()
	last := dwarfp
	last = writelines(last)
	last = writeframes(last)

	synthesizestringtypes(dwtypes.child)
	synthesizeslicetypes(dwtypes.child)
	synthesizemaptypes(dwtypes.child)
	synthesizechantypes(dwtypes.child)

	reversetree(&dwroot.child)
	reversetree(&dwtypes.child)
	reversetree(&dwglobals.child)

	movetomodule(&dwtypes)
	movetomodule(&dwglobals)

	// Need to reorder symbols so SDWARFINFO is after all SDWARFSECT
	// (but we need to generate dies before writepub)
	writeinfo(last)
	infosyms := last.Next

	last = writepub(".debug_pubnames", ispubname, last)
	last = writepub(".debug_pubtypes", ispubtype, last)
	last = writearanges(last)
	last = writegdbscript(last)
	last.Next = infosyms
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
