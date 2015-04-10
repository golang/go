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
	"strings"
)

/*
 * Offsets and sizes of the debug_* sections in the cout file.
 */
var abbrevo int64

var abbrevsize int64

var abbrevsym *LSym

var abbrevsympos int64

var lineo int64

var linesize int64

var linesym *LSym

var linesympos int64

var infoo int64 // also the base for DWDie->offs and reference attributes.

var infosize int64

var infosym *LSym

var infosympos int64

var frameo int64

var framesize int64

var framesym *LSym

var framesympos int64

var pubnameso int64

var pubnamessize int64

var pubtypeso int64

var pubtypessize int64

var arangeso int64

var arangessize int64

var gdbscripto int64

var gdbscriptsize int64

var infosec *LSym

var inforeloco int64

var inforelocsize int64

var arangessec *LSym

var arangesreloco int64

var arangesrelocsize int64

var linesec *LSym

var linereloco int64

var linerelocsize int64

var framesec *LSym

var framereloco int64

var framerelocsize int64

var gdbscript string

/*
 *  Basic I/O
 */
func addrput(addr int64) {
	switch Thearch.Ptrsize {
	case 4:
		Thearch.Lput(uint32(addr))

	case 8:
		Thearch.Vput(uint64(addr))
	}
}

func uleb128enc(v uint64, dst []byte) int {
	var c uint8

	length := uint8(0)
	for {
		c = uint8(v & 0x7f)
		v >>= 7
		if v != 0 {
			c |= 0x80
		}
		if dst != nil {
			dst[0] = byte(c)
			dst = dst[1:]
		}
		length++
		if c&0x80 == 0 {
			break
		}
	}

	return int(length)
}

func sleb128enc(v int64, dst []byte) int {
	var c uint8
	var s uint8

	length := uint8(0)
	for {
		c = uint8(v & 0x7f)
		s = uint8(v & 0x40)
		v >>= 7
		if (v != -1 || s == 0) && (v != 0 || s != 0) {
			c |= 0x80
		}
		if dst != nil {
			dst[0] = byte(c)
			dst = dst[1:]
		}
		length++
		if c&0x80 == 0 {
			break
		}
	}

	return int(length)
}

func uleb128put(v int64) {
	var buf [10]byte
	n := uleb128enc(uint64(v), buf[:])
	Cwrite(buf[:n])
}

func sleb128put(v int64) {
	var buf [10]byte
	n := sleb128enc(v, buf[:])
	Cwrite(buf[:n])
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

func writeabbrev() {
	abbrevo = Cpos()
	for i := 1; i < DW_NABRV; i++ {
		// See section 7.5.3
		uleb128put(int64(i))

		uleb128put(int64(abbrevs[i].tag))
		Cput(abbrevs[i].children)
		for _, f := range abbrevs[i].attr {
			uleb128put(int64(f.attr))
			uleb128put(int64(f.form))
		}
		uleb128put(0)
		uleb128put(0)
	}

	Cput(0)
	abbrevsize = Cpos() - abbrevo
}

/*
 * Debugging Information Entries and their attributes.
 */
const (
	HASHSIZE = 107
)

func dwarfhashstr(s string) uint32 {
	h := uint32(0)
	for s != "" {
		h = h + h + h + uint32(s[0])
		s = s[1:]
	}
	return h % HASHSIZE
}

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
	// offset into .debug_info section, i.e relative to
	// infoo. only valid after call to putdie()
	offs  int64
	hash  []*DWDie // optional index of children by name, enabled by mkindex()
	hlink *DWDie   // bucket chain in parent's index
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
// written out if it is listed in the abbrev).	If its parent is
// keeping an index, the new DIE will be inserted there.
func newdie(parent *DWDie, abbrev int, name string) *DWDie {
	die := new(DWDie)
	die.abbrev = abbrev
	die.link = parent.child
	parent.child = die

	newattr(die, DW_AT_name, DW_CLS_STRING, int64(len(name)), name)

	if parent.hash != nil {
		h := int(dwarfhashstr(name))
		die.hlink = parent.hash[h]
		parent.hash[h] = die
	}

	return die
}

func mkindex(die *DWDie) {
	die.hash = make([]*DWDie, HASHSIZE)
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

// Find child by AT_name using hashtable if available or linear scan
// if not.
func find(die *DWDie, name string) *DWDie {
	var a *DWDie
	var b *DWDie
	var die2 *DWDie
	var h int

top:
	if die.hash == nil {
		for a = die.child; a != nil; a = a.link {
			if name == getattr(a, DW_AT_name).data {
				return a
			}
		}
		goto notfound
	}

	h = int(dwarfhashstr(name))
	a = die.hash[h]

	if a == nil {
		goto notfound
	}

	if name == getattr(a, DW_AT_name).data {
		return a
	}

	// Move found ones to head of the list.
	b = a.hlink

	for b != nil {
		if name == getattr(b, DW_AT_name).data {
			a.hlink = b.hlink
			b.hlink = die.hash[h]
			die.hash[h] = b
			return b
		}

		a = b
		b = b.hlink
	}

notfound:
	die2 = walktypedef(die)
	if die2 != die {
		die = die2
		goto top
	}

	return nil
}

func find_or_diag(die *DWDie, name string) *DWDie {
	r := find(die, name)
	if r == nil {
		Diag("dwarf find: %s %p has no %s", getattr(die, DW_AT_name).data, die, name)
		Errorexit()
	}

	return r
}

func adddwarfrel(sec *LSym, sym *LSym, offsetbase int64, siz int, addend int64) {
	r := Addrel(sec)
	r.Sym = sym
	r.Xsym = sym
	r.Off = int32(Cpos() - offsetbase)
	r.Siz = uint8(siz)
	r.Type = R_ADDR
	r.Add = addend
	r.Xadd = addend
	if Iself && Thearch.Thechar == '6' {
		addend = 0
	}
	switch siz {
	case 4:
		Thearch.Lput(uint32(addend))

	case 8:
		Thearch.Vput(uint64(addend))

	default:
		Diag("bad size in adddwarfrel")
	}
}

func newrefattr(die *DWDie, attr uint16, ref *DWDie) *DWAttr {
	if ref == nil {
		return nil
	}
	return newattr(die, attr, DW_CLS_REFERENCE, 0, ref)
}

var fwdcount int

func putattr(abbrev int, form int, cls int, value int64, data interface{}) {
	switch form {
	case DW_FORM_addr: // address
		if Linkmode == LinkExternal {
			value -= (data.(*LSym)).Value
			adddwarfrel(infosec, data.(*LSym), infoo, Thearch.Ptrsize, value)
			break
		}

		addrput(value)

	case DW_FORM_block1: // block
		if cls == DW_CLS_ADDRESS {
			Cput(uint8(1 + Thearch.Ptrsize))
			Cput(DW_OP_addr)
			if Linkmode == LinkExternal {
				value -= (data.(*LSym)).Value
				adddwarfrel(infosec, data.(*LSym), infoo, Thearch.Ptrsize, value)
				break
			}

			addrput(value)
			break
		}

		value &= 0xff
		Cput(uint8(value))
		p := data.([]byte)
		for i := 0; int64(i) < value; i++ {
			Cput(uint8(p[i]))
		}

	case DW_FORM_block2: // block
		value &= 0xffff

		Thearch.Wput(uint16(value))
		p := data.([]byte)
		for i := 0; int64(i) < value; i++ {
			Cput(uint8(p[i]))
		}

	case DW_FORM_block4: // block
		value &= 0xffffffff

		Thearch.Lput(uint32(value))
		p := data.([]byte)
		for i := 0; int64(i) < value; i++ {
			Cput(uint8(p[i]))
		}

	case DW_FORM_block: // block
		uleb128put(value)

		p := data.([]byte)
		for i := 0; int64(i) < value; i++ {
			Cput(uint8(p[i]))
		}

	case DW_FORM_data1: // constant
		Cput(uint8(value))

	case DW_FORM_data2: // constant
		Thearch.Wput(uint16(value))

	case DW_FORM_data4: // constant, {line,loclist,mac,rangelist}ptr
		if Linkmode == LinkExternal && cls == DW_CLS_PTR {
			adddwarfrel(infosec, linesym, infoo, 4, value)
			break
		}

		Thearch.Lput(uint32(value))

	case DW_FORM_data8: // constant, {line,loclist,mac,rangelist}ptr
		Thearch.Vput(uint64(value))

	case DW_FORM_sdata: // constant
		sleb128put(value)

	case DW_FORM_udata: // constant
		uleb128put(value)

	case DW_FORM_string: // string
		strnput(data.(string), int(value+1))

	case DW_FORM_flag: // flag
		if value != 0 {
			Cput(1)
		} else {
			Cput(0)
		}

		// In DWARF 2 (which is what we claim to generate),
	// the ref_addr is the same size as a normal address.
	// In DWARF 3 it is always 32 bits, unless emitting a large
	// (> 4 GB of debug info aka "64-bit") unit, which we don't implement.
	case DW_FORM_ref_addr: // reference to a DIE in the .info section
		if data == nil {
			Diag("dwarf: null reference in %d", abbrev)
			if Thearch.Ptrsize == 8 {
				Thearch.Vput(0) // invalid dwarf, gdb will complain.
			} else {
				Thearch.Lput(0) // invalid dwarf, gdb will complain.
			}
		} else {
			off := (data.(*DWDie)).offs
			if off == 0 {
				fwdcount++
			}
			if Linkmode == LinkExternal {
				adddwarfrel(infosec, infosym, infoo, Thearch.Ptrsize, off)
				break
			}

			addrput(off)
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
		Diag("dwarf: unsupported attribute form %d / class %d", form, cls)

		Errorexit()
	}
}

// Note that we can (and do) add arbitrary attributes to a DIE, but
// only the ones actually listed in the Abbrev will be written out.
func putattrs(abbrev int, attr *DWAttr) {
Outer:
	for _, f := range abbrevs[abbrev].attr {
		for ap := attr; ap != nil; ap = ap.link {
			if ap.atr == f.attr {
				putattr(abbrev, int(f.form), int(ap.cls), ap.value, ap.data)
				continue Outer
			}
		}

		putattr(abbrev, int(f.form), 0, 0, nil)
	}
}

func putdies(die *DWDie) {
	for ; die != nil; die = die.link {
		putdie(die)
	}
}

func putdie(die *DWDie) {
	die.offs = Cpos() - infoo
	uleb128put(int64(die.abbrev))
	putattrs(die.abbrev, die.attr)
	if abbrevs[die.abbrev].children != 0 {
		putdies(die.child)
		Cput(0)
	}
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

	i := 0
	block[i] = DW_OP_plus_uconst
	i++
	i += uleb128enc(uint64(offs), block[i:])
	newattr(die, DW_AT_data_member_location, DW_CLS_BLOCK, int64(i), block[:i])
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
		Diag("dwarf: missing type: %s", n)
		Errorexit()
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

	// The typedef entry must be created after the def,
	// so that future lookups will find the typedef instead
	// of the real definition. This hooks the typedef into any
	// circular definition loops, so that gdb can understand them.
	die := newdie(parent, DW_ABRV_TYPEDECL, name)

	newrefattr(die, DW_AT_type, def)
}

// Define gotype, for composite ones recurse into constituents.
func defgotype(gotype *LSym) *DWDie {
	if gotype == nil {
		return find_or_diag(&dwtypes, "<unspecified>")
	}

	if !strings.HasPrefix(gotype.Name, "type.") {
		Diag("dwarf: type name doesn't start with \".type\": %s", gotype.Name)
		return find_or_diag(&dwtypes, "<unspecified>")
	}

	name := gotype.Name[5:] // could also decode from Type.string

	die := find(&dwtypes, name)

	if die != nil {
		return die
	}

	if false && Debug['v'] > 2 {
		fmt.Printf("new type: %%Y\n", gotype)
	}

	kind := decodetype_kind(gotype)
	bytesize := decodetype_size(gotype)

	switch kind {
	case obj.KindBool:
		die = newdie(&dwtypes, DW_ABRV_BASETYPE, name)
		newattr(die, DW_AT_encoding, DW_CLS_CONSTANT, DW_ATE_boolean, 0)
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0)

	case obj.KindInt,
		obj.KindInt8,
		obj.KindInt16,
		obj.KindInt32,
		obj.KindInt64:
		die = newdie(&dwtypes, DW_ABRV_BASETYPE, name)
		newattr(die, DW_AT_encoding, DW_CLS_CONSTANT, DW_ATE_signed, 0)
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0)

	case obj.KindUint,
		obj.KindUint8,
		obj.KindUint16,
		obj.KindUint32,
		obj.KindUint64,
		obj.KindUintptr:
		die = newdie(&dwtypes, DW_ABRV_BASETYPE, name)
		newattr(die, DW_AT_encoding, DW_CLS_CONSTANT, DW_ATE_unsigned, 0)
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0)

	case obj.KindFloat32,
		obj.KindFloat64:
		die = newdie(&dwtypes, DW_ABRV_BASETYPE, name)
		newattr(die, DW_AT_encoding, DW_CLS_CONSTANT, DW_ATE_float, 0)
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0)

	case obj.KindComplex64,
		obj.KindComplex128:
		die = newdie(&dwtypes, DW_ABRV_BASETYPE, name)
		newattr(die, DW_AT_encoding, DW_CLS_CONSTANT, DW_ATE_complex_float, 0)
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0)

	case obj.KindArray:
		die = newdie(&dwtypes, DW_ABRV_ARRAYTYPE, name)
		dotypedef(&dwtypes, name, die)
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0)
		s := decodetype_arrayelem(gotype)
		newrefattr(die, DW_AT_type, defgotype(s))
		fld := newdie(die, DW_ABRV_ARRAYRANGE, "range")

		// use actual length not upper bound; correct for 0-length arrays.
		newattr(fld, DW_AT_count, DW_CLS_CONSTANT, decodetype_arraylen(gotype), 0)

		newrefattr(fld, DW_AT_type, find_or_diag(&dwtypes, "uintptr"))

	case obj.KindChan:
		die = newdie(&dwtypes, DW_ABRV_CHANTYPE, name)
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0)
		s := decodetype_chanelem(gotype)
		newrefattr(die, DW_AT_go_elem, defgotype(s))

	case obj.KindFunc:
		die = newdie(&dwtypes, DW_ABRV_FUNCTYPE, name)
		dotypedef(&dwtypes, name, die)
		newrefattr(die, DW_AT_type, find_or_diag(&dwtypes, "void"))
		nfields := decodetype_funcincount(gotype)
		var fld *DWDie
		var s *LSym
		for i := 0; i < nfields; i++ {
			s = decodetype_funcintype(gotype, i)
			fld = newdie(die, DW_ABRV_FUNCTYPEPARAM, s.Name[5:])
			newrefattr(fld, DW_AT_type, defgotype(s))
		}

		if decodetype_funcdotdotdot(gotype) != 0 {
			newdie(die, DW_ABRV_DOTDOTDOT, "...")
		}
		nfields = decodetype_funcoutcount(gotype)
		for i := 0; i < nfields; i++ {
			s = decodetype_funcouttype(gotype, i)
			fld = newdie(die, DW_ABRV_FUNCTYPEPARAM, s.Name[5:])
			newrefattr(fld, DW_AT_type, defptrto(defgotype(s)))
		}

	case obj.KindInterface:
		die = newdie(&dwtypes, DW_ABRV_IFACETYPE, name)
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
		die = newdie(&dwtypes, DW_ABRV_MAPTYPE, name)
		s := decodetype_mapkey(gotype)
		newrefattr(die, DW_AT_go_key, defgotype(s))
		s = decodetype_mapvalue(gotype)
		newrefattr(die, DW_AT_go_elem, defgotype(s))

	case obj.KindPtr:
		die = newdie(&dwtypes, DW_ABRV_PTRTYPE, name)
		dotypedef(&dwtypes, name, die)
		s := decodetype_ptrelem(gotype)
		newrefattr(die, DW_AT_type, defgotype(s))

	case obj.KindSlice:
		die = newdie(&dwtypes, DW_ABRV_SLICETYPE, name)
		dotypedef(&dwtypes, name, die)
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0)
		s := decodetype_arrayelem(gotype)
		newrefattr(die, DW_AT_go_elem, defgotype(s))

	case obj.KindString:
		die = newdie(&dwtypes, DW_ABRV_STRINGTYPE, name)
		newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, bytesize, 0)

	case obj.KindStruct:
		die = newdie(&dwtypes, DW_ABRV_STRUCTTYPE, name)
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
			fld = newdie(die, DW_ABRV_STRUCTFIELD, f)
			newrefattr(fld, DW_AT_type, defgotype(s))
			newmemberoffsetattr(fld, int32(decodetype_structfieldoffs(gotype, i)))
		}

	case obj.KindUnsafePointer:
		die = newdie(&dwtypes, DW_ABRV_BARE_PTRTYPE, name)

	default:
		Diag("dwarf: definition of unknown kind %d: %s", kind, gotype.Name)
		die = newdie(&dwtypes, DW_ABRV_TYPEDECL, name)
		newrefattr(die, DW_AT_type, find_or_diag(&dwtypes, "<unspecified>"))
	}

	newattr(die, DW_AT_go_kind, DW_CLS_CONSTANT, int64(kind), 0)

	return die
}

// Find or construct *T given T.
func defptrto(dwtype *DWDie) *DWDie {
	ptrname := fmt.Sprintf("*%s", getattr(dwtype, DW_AT_name).data)
	die := find(&dwtypes, ptrname)
	if die == nil {
		die = newdie(&dwtypes, DW_ABRV_PTRTYPE, ptrname)
		newrefattr(die, DW_AT_type, dwtype)
	}

	return die
}

// Copies src's children into dst. Copies attributes by value.
// DWAttr.data is copied as pointer only.  If except is one of
// the top-level children, it will not be copied.
func copychildrenexcept(dst *DWDie, src *DWDie, except *DWDie) {
	var c *DWDie
	var a *DWAttr

	for src = src.child; src != nil; src = src.link {
		if src == except {
			continue
		}
		c = newdie(dst, src.abbrev, getattr(src, DW_AT_name).data.(string))
		for a = src.attr; a != nil; a = a.link {
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
func substitutetype(structdie *DWDie, field string, dwtype *DWDie) {
	child := find_or_diag(structdie, field)
	if child == nil {
		return
	}

	a := getattr(child, DW_AT_type)
	if a != nil {
		a.data = dwtype
	} else {
		newrefattr(child, DW_AT_type, dwtype)
	}
}

func synthesizestringtypes(die *DWDie) {
	prototype := walktypedef(defgotype(lookup_or_diag("type.runtime._string")))
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
	prototype := walktypedef(defgotype(lookup_or_diag("type.runtime.slice")))
	if prototype == nil {
		return
	}

	var elem *DWDie
	for ; die != nil; die = die.link {
		if die.abbrev != DW_ABRV_SLICETYPE {
			continue
		}
		copychildren(die, prototype)
		elem = getattr(die, DW_AT_go_elem).data.(*DWDie)
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

func synthesizemaptypes(die *DWDie) {
	hash := walktypedef(defgotype(lookup_or_diag("type.runtime.hmap")))
	bucket := walktypedef(defgotype(lookup_or_diag("type.runtime.bmap")))

	if hash == nil {
		return
	}

	var a *DWAttr
	var dwh *DWDie
	var dwhb *DWDie
	var dwhk *DWDie
	var dwhv *DWDie
	var fld *DWDie
	var indirect_key int
	var indirect_val int
	var keysize int
	var keytype *DWDie
	var t *DWDie
	var valsize int
	var valtype *DWDie
	for ; die != nil; die = die.link {
		if die.abbrev != DW_ABRV_MAPTYPE {
			continue
		}

		keytype = walktypedef(getattr(die, DW_AT_go_key).data.(*DWDie))
		valtype = walktypedef(getattr(die, DW_AT_go_elem).data.(*DWDie))

		// compute size info like hashmap.c does.
		a = getattr(keytype, DW_AT_byte_size)

		if a != nil {
			keysize = int(a.value)
		} else {
			keysize = Thearch.Ptrsize
		}
		a = getattr(valtype, DW_AT_byte_size)
		if a != nil {
			valsize = int(a.value)
		} else {
			valsize = Thearch.Ptrsize
		}
		indirect_key = 0
		indirect_val = 0
		if keysize > MaxKeySize {
			keysize = Thearch.Ptrsize
			indirect_key = 1
		}

		if valsize > MaxValSize {
			valsize = Thearch.Ptrsize
			indirect_val = 1
		}

		// Construct type to represent an array of BucketSize keys
		dwhk = newdie(&dwtypes, DW_ABRV_ARRAYTYPE, mkinternaltypename("[]key", getattr(keytype, DW_AT_name).data.(string), ""))

		newattr(dwhk, DW_AT_byte_size, DW_CLS_CONSTANT, BucketSize*int64(keysize), 0)
		t = keytype
		if indirect_key != 0 {
			t = defptrto(keytype)
		}
		newrefattr(dwhk, DW_AT_type, t)
		fld = newdie(dwhk, DW_ABRV_ARRAYRANGE, "size")
		newattr(fld, DW_AT_count, DW_CLS_CONSTANT, BucketSize, 0)
		newrefattr(fld, DW_AT_type, find_or_diag(&dwtypes, "uintptr"))

		// Construct type to represent an array of BucketSize values
		dwhv = newdie(&dwtypes, DW_ABRV_ARRAYTYPE, mkinternaltypename("[]val", getattr(valtype, DW_AT_name).data.(string), ""))

		newattr(dwhv, DW_AT_byte_size, DW_CLS_CONSTANT, BucketSize*int64(valsize), 0)
		t = valtype
		if indirect_val != 0 {
			t = defptrto(valtype)
		}
		newrefattr(dwhv, DW_AT_type, t)
		fld = newdie(dwhv, DW_ABRV_ARRAYRANGE, "size")
		newattr(fld, DW_AT_count, DW_CLS_CONSTANT, BucketSize, 0)
		newrefattr(fld, DW_AT_type, find_or_diag(&dwtypes, "uintptr"))

		// Construct bucket<K,V>
		dwhb = newdie(&dwtypes, DW_ABRV_STRUCTTYPE, mkinternaltypename("bucket", getattr(keytype, DW_AT_name).data.(string), getattr(valtype, DW_AT_name).data.(string)))

		// Copy over all fields except the field "data" from the generic bucket.
		// "data" will be replaced with keys/values below.
		copychildrenexcept(dwhb, bucket, find(bucket, "data"))

		fld = newdie(dwhb, DW_ABRV_STRUCTFIELD, "keys")
		newrefattr(fld, DW_AT_type, dwhk)
		newmemberoffsetattr(fld, BucketSize)
		fld = newdie(dwhb, DW_ABRV_STRUCTFIELD, "values")
		newrefattr(fld, DW_AT_type, dwhv)
		newmemberoffsetattr(fld, BucketSize+BucketSize*int32(keysize))
		fld = newdie(dwhb, DW_ABRV_STRUCTFIELD, "overflow")
		newrefattr(fld, DW_AT_type, defptrto(dwhb))
		newmemberoffsetattr(fld, BucketSize+BucketSize*(int32(keysize)+int32(valsize)))
		if Thearch.Regsize > Thearch.Ptrsize {
			fld = newdie(dwhb, DW_ABRV_STRUCTFIELD, "pad")
			newrefattr(fld, DW_AT_type, find_or_diag(&dwtypes, "uintptr"))
			newmemberoffsetattr(fld, BucketSize+BucketSize*(int32(keysize)+int32(valsize))+int32(Thearch.Ptrsize))
		}

		newattr(dwhb, DW_AT_byte_size, DW_CLS_CONSTANT, BucketSize+BucketSize*int64(keysize)+BucketSize*int64(valsize)+int64(Thearch.Regsize), 0)

		// Construct hash<K,V>
		dwh = newdie(&dwtypes, DW_ABRV_STRUCTTYPE, mkinternaltypename("hash", getattr(keytype, DW_AT_name).data.(string), getattr(valtype, DW_AT_name).data.(string)))

		copychildren(dwh, hash)
		substitutetype(dwh, "buckets", defptrto(dwhb))
		substitutetype(dwh, "oldbuckets", defptrto(dwhb))
		newattr(dwh, DW_AT_byte_size, DW_CLS_CONSTANT, getattr(hash, DW_AT_byte_size).value, nil)

		// make map type a pointer to hash<K,V>
		newrefattr(die, DW_AT_type, defptrto(dwh))
	}
}

func synthesizechantypes(die *DWDie) {
	sudog := walktypedef(defgotype(lookup_or_diag("type.runtime.sudog")))
	waitq := walktypedef(defgotype(lookup_or_diag("type.runtime.waitq")))
	hchan := walktypedef(defgotype(lookup_or_diag("type.runtime.hchan")))
	if sudog == nil || waitq == nil || hchan == nil {
		return
	}

	sudogsize := int(getattr(sudog, DW_AT_byte_size).value)

	var a *DWAttr
	var dwh *DWDie
	var dws *DWDie
	var dww *DWDie
	var elemsize int
	var elemtype *DWDie
	for ; die != nil; die = die.link {
		if die.abbrev != DW_ABRV_CHANTYPE {
			continue
		}
		elemtype = getattr(die, DW_AT_go_elem).data.(*DWDie)
		a = getattr(elemtype, DW_AT_byte_size)
		if a != nil {
			elemsize = int(a.value)
		} else {
			elemsize = Thearch.Ptrsize
		}

		// sudog<T>
		dws = newdie(&dwtypes, DW_ABRV_STRUCTTYPE, mkinternaltypename("sudog", getattr(elemtype, DW_AT_name).data.(string), ""))

		copychildren(dws, sudog)
		substitutetype(dws, "elem", elemtype)
		if elemsize > 8 {
			elemsize -= 8
		} else {
			elemsize = 0
		}
		newattr(dws, DW_AT_byte_size, DW_CLS_CONSTANT, int64(sudogsize)+int64(elemsize), nil)

		// waitq<T>
		dww = newdie(&dwtypes, DW_ABRV_STRUCTTYPE, mkinternaltypename("waitq", getattr(elemtype, DW_AT_name).data.(string), ""))

		copychildren(dww, waitq)
		substitutetype(dww, "first", defptrto(dws))
		substitutetype(dww, "last", defptrto(dws))
		newattr(dww, DW_AT_byte_size, DW_CLS_CONSTANT, getattr(waitq, DW_AT_byte_size).value, nil)

		// hchan<T>
		dwh = newdie(&dwtypes, DW_ABRV_STRUCTTYPE, mkinternaltypename("hchan", getattr(elemtype, DW_AT_name).data.(string), ""))

		copychildren(dwh, hchan)
		substitutetype(dwh, "recvq", dww)
		substitutetype(dwh, "sendq", dww)
		newattr(dwh, DW_AT_byte_size, DW_CLS_CONSTANT, getattr(hchan, DW_AT_byte_size).value, nil)

		newrefattr(die, DW_AT_type, defptrto(dwh))
	}
}

// For use with pass.c::genasmsym
func defdwsymb(sym *LSym, s string, t int, v int64, size int64, ver int, gotype *LSym) {
	if strings.HasPrefix(s, "go.string.") {
		return
	}

	if strings.HasPrefix(s, "type.") && s != "type.*" && !strings.HasPrefix(s, "type..") {
		defgotype(sym)
		return
	}

	var dv *DWDie

	var dt *DWDie
	switch t {
	default:
		return

	case 'd', 'b', 'D', 'B':
		dv = newdie(&dwglobals, DW_ABRV_VARIABLE, s)
		newabslocexprattr(dv, v, sym)
		if ver == 0 {
			newattr(dv, DW_AT_external, DW_CLS_FLAG, 1, 0)
		}
		fallthrough

		// fallthrough
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

	var f *LSym
	var p string
	for i := 0; i < s.Pcln.Nfile; i++ {
		f = s.Pcln.File[i]
		_ = p
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

func putpclcdelta(delta_pc int64, delta_lc int64) {
	if LINE_BASE <= delta_lc && delta_lc < LINE_BASE+LINE_RANGE {
		var opcode int64 = OPCODE_BASE + (delta_lc - LINE_BASE) + (LINE_RANGE * delta_pc)
		if OPCODE_BASE <= opcode && opcode < 256 {
			Cput(uint8(opcode))
			return
		}
	}

	if delta_pc != 0 {
		Cput(DW_LNS_advance_pc)
		sleb128put(delta_pc)
	}

	Cput(DW_LNS_advance_line)
	sleb128put(delta_lc)
	Cput(DW_LNS_copy)
}

func newcfaoffsetattr(die *DWDie, offs int32) {
	var block [20]byte

	i := 0

	block[i] = DW_OP_call_frame_cfa
	i++
	if offs != 0 {
		block[i] = DW_OP_consts
		i++
		i += sleb128enc(int64(offs), block[i:])
		block[i] = DW_OP_plus
		i++
	}

	newattr(die, DW_AT_location, DW_CLS_BLOCK, int64(i), block[:i])
}

func mkvarname(name string, da int) string {
	buf := fmt.Sprintf("%s#%d", name, da)
	n := buf
	return n
}

/*
 * Walk prog table, emit line program and build DIE tree.
 */

// flush previous compilation unit.
func flushunit(dwinfo *DWDie, pc int64, pcsym *LSym, unitstart int64, header_length int32) {
	if dwinfo != nil && pc != 0 {
		newattr(dwinfo, DW_AT_high_pc, DW_CLS_ADDRESS, pc+1, pcsym)
	}

	if unitstart >= 0 {
		Cput(0) // start extended opcode
		uleb128put(1)
		Cput(DW_LNE_end_sequence)

		here := Cpos()
		Cseek(unitstart)
		Thearch.Lput(uint32(here - unitstart - 4)) // unit_length
		Thearch.Wput(2)                            // dwarf version
		Thearch.Lput(uint32(header_length))        // header length starting here
		Cseek(here)
	}
}

func writelines() {
	if linesec == nil {
		linesec = Linklookup(Ctxt, ".dwarfline", 0)
	}
	linesec.R = linesec.R[:0]

	unitstart := int64(-1)
	headerend := int64(-1)
	epc := int64(0)
	var epcs *LSym
	lineo = Cpos()
	var dwinfo *DWDie

	flushunit(dwinfo, epc, epcs, unitstart, int32(headerend-unitstart-10))
	unitstart = Cpos()

	lang := DW_LANG_Go

	s := Ctxt.Textp

	dwinfo = newdie(&dwroot, DW_ABRV_COMPUNIT, "go")
	newattr(dwinfo, DW_AT_language, DW_CLS_CONSTANT, int64(lang), 0)
	newattr(dwinfo, DW_AT_stmt_list, DW_CLS_PTR, unitstart-lineo, 0)
	newattr(dwinfo, DW_AT_low_pc, DW_CLS_ADDRESS, s.Value, s)

	// Write .debug_line Line Number Program Header (sec 6.2.4)
	// Fields marked with (*) must be changed for 64-bit dwarf
	Thearch.Lput(0) // unit_length (*), will be filled in by flushunit.
	Thearch.Wput(2) // dwarf version (appendix F)
	Thearch.Lput(0) // header_length (*), filled in by flushunit.

	// cpos == unitstart + 4 + 2 + 4
	Cput(1)                // minimum_instruction_length
	Cput(1)                // default_is_stmt
	Cput(LINE_BASE & 0xFF) // line_base
	Cput(LINE_RANGE)       // line_range
	Cput(OPCODE_BASE)      // opcode_base
	Cput(0)                // standard_opcode_lengths[1]
	Cput(1)                // standard_opcode_lengths[2]
	Cput(1)                // standard_opcode_lengths[3]
	Cput(1)                // standard_opcode_lengths[4]
	Cput(1)                // standard_opcode_lengths[5]
	Cput(0)                // standard_opcode_lengths[6]
	Cput(0)                // standard_opcode_lengths[7]
	Cput(0)                // standard_opcode_lengths[8]
	Cput(1)                // standard_opcode_lengths[9]
	Cput(0)                // include_directories  (empty)

	files := make([]*LSym, Ctxt.Nhistfile)

	for f := Ctxt.Filesyms; f != nil; f = f.Next {
		files[f.Value-1] = f
	}

	for i := 0; int32(i) < Ctxt.Nhistfile; i++ {
		strnput(files[i].Name, len(files[i].Name)+4)
	}

	// 4 zeros: the string termination + 3 fields.
	Cput(0)
	// terminate file_names.
	headerend = Cpos()

	Cput(0) // start extended opcode
	uleb128put(1 + int64(Thearch.Ptrsize))
	Cput(DW_LNE_set_address)

	pc := s.Value
	line := 1
	file := 1
	if Linkmode == LinkExternal {
		adddwarfrel(linesec, s, lineo, Thearch.Ptrsize, 0)
	} else {
		addrput(pc)
	}

	var a *Auto
	var da int
	var dt int
	var dwfunc *DWDie
	var dws **DWDie
	var dwvar *DWDie
	var n string
	var nn string
	var offs int64
	var pcfile Pciter
	var pcline Pciter
	var varhash [HASHSIZE]*DWDie
	for Ctxt.Cursym = Ctxt.Textp; Ctxt.Cursym != nil; Ctxt.Cursym = Ctxt.Cursym.Next {
		s = Ctxt.Cursym

		dwfunc = newdie(dwinfo, DW_ABRV_FUNCTION, s.Name)
		newattr(dwfunc, DW_AT_low_pc, DW_CLS_ADDRESS, s.Value, s)
		epc = s.Value + s.Size
		epcs = s
		newattr(dwfunc, DW_AT_high_pc, DW_CLS_ADDRESS, epc, s)
		if s.Version == 0 {
			newattr(dwfunc, DW_AT_external, DW_CLS_FLAG, 1, 0)
		}

		if s.Pcln == nil {
			continue
		}

		finddebugruntimepath(s)

		pciterinit(Ctxt, &pcfile, &s.Pcln.Pcfile)
		pciterinit(Ctxt, &pcline, &s.Pcln.Pcline)
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
				Cput(DW_LNS_set_file)
				uleb128put(int64(pcfile.value))
				file = int(pcfile.value)
			}

			putpclcdelta(s.Value+int64(pcline.pc)-pc, int64(pcline.value)-int64(line))

			pc = s.Value + int64(pcline.pc)
			line = int(pcline.value)
			if pcfile.nextpc < pcline.nextpc {
				epc = int64(pcfile.nextpc)
			} else {
				epc = int64(pcline.nextpc)
			}
			epc += s.Value
		}

		da = 0
		dwfunc.hash = varhash[:] // enable indexing of children by name
		varhash = [HASHSIZE]*DWDie{}
		for a = s.Autom; a != nil; a = a.Link {
			switch a.Name {
			case A_AUTO:
				dt = DW_ABRV_AUTO
				offs = int64(a.Aoffset) - int64(Thearch.Ptrsize)

			case A_PARAM:
				dt = DW_ABRV_PARAM
				offs = int64(a.Aoffset)

			default:
				continue
			}

			if strings.Contains(a.Asym.Name, ".autotmp_") {
				continue
			}
			if find(dwfunc, a.Asym.Name) != nil {
				n = mkvarname(a.Asym.Name, da)
			} else {
				n = a.Asym.Name
			}

			// Drop the package prefix from locals and arguments.
			_ = nn
			if i := strings.LastIndex(n, "."); i >= 0 {
				n = n[i+1:]
			}

			dwvar = newdie(dwfunc, dt, n)
			newcfaoffsetattr(dwvar, int32(offs))
			newrefattr(dwvar, DW_AT_type, defgotype(a.Gotype))

			// push dwvar down dwfunc->child to preserve order
			newattr(dwvar, DW_AT_internal_location, DW_CLS_CONSTANT, offs, nil)

			dwfunc.child = dwvar.link // take dwvar out from the top of the list
			for dws = &dwfunc.child; *dws != nil; dws = &(*dws).link {
				if offs > getattr(*dws, DW_AT_internal_location).value {
					break
				}
			}
			dwvar.link = *dws
			*dws = dwvar

			da++
		}

		dwfunc.hash = nil
	}

	flushunit(dwinfo, epc, epcs, unitstart, int32(headerend-unitstart-10))
	linesize = Cpos() - lineo
}

/*
 *  Emit .debug_frame
 */
const (
	CIERESERVE          = 16
	DATAALIGNMENTFACTOR = -4
	FAKERETURNCOLUMN    = 16 // TODO gdb6 doesn't like > 15?
)

func putpccfadelta(deltapc int64, cfa int64) {
	Cput(DW_CFA_def_cfa_offset_sf)
	sleb128put(cfa / DATAALIGNMENTFACTOR)

	if deltapc < 0x40 {
		Cput(uint8(DW_CFA_advance_loc + deltapc))
	} else if deltapc < 0x100 {
		Cput(DW_CFA_advance_loc1)
		Cput(uint8(deltapc))
	} else if deltapc < 0x10000 {
		Cput(DW_CFA_advance_loc2)
		Thearch.Wput(uint16(deltapc))
	} else {
		Cput(DW_CFA_advance_loc4)
		Thearch.Lput(uint32(deltapc))
	}
}

func writeframes() {
	if framesec == nil {
		framesec = Linklookup(Ctxt, ".dwarfframe", 0)
	}
	framesec.R = framesec.R[:0]
	frameo = Cpos()

	// Emit the CIE, Section 6.4.1
	Thearch.Lput(CIERESERVE)        // initial length, must be multiple of thearch.ptrsize
	Thearch.Lput(0xffffffff)        // cid.
	Cput(3)                         // dwarf version (appendix F)
	Cput(0)                         // augmentation ""
	uleb128put(1)                   // code_alignment_factor
	sleb128put(DATAALIGNMENTFACTOR) // guess
	uleb128put(FAKERETURNCOLUMN)    // return_address_register

	Cput(DW_CFA_def_cfa)

	uleb128put(int64(Thearch.Dwarfregsp)) // register SP (**ABI-dependent, defined in l.h)
	uleb128put(int64(Thearch.Ptrsize))    // offset

	Cput(DW_CFA_offset + FAKERETURNCOLUMN)                    // return address
	uleb128put(int64(-Thearch.Ptrsize) / DATAALIGNMENTFACTOR) // at cfa - x*4

	// 4 is to exclude the length field.
	pad := CIERESERVE + frameo + 4 - Cpos()

	if pad < 0 {
		Diag("dwarf: CIERESERVE too small by %d bytes.", -pad)
		Errorexit()
	}

	strnput("", int(pad))

	var fdeo int64
	var fdesize int64
	var nextpc uint32
	var pcsp Pciter
	var s *LSym
	for Ctxt.Cursym = Ctxt.Textp; Ctxt.Cursym != nil; Ctxt.Cursym = Ctxt.Cursym.Next {
		s = Ctxt.Cursym
		if s.Pcln == nil {
			continue
		}

		fdeo = Cpos()

		// Emit a FDE, Section 6.4.1, starting wit a placeholder.
		Thearch.Lput(0) // length, must be multiple of thearch.ptrsize
		Thearch.Lput(0) // Pointer to the CIE above, at offset 0
		addrput(0)      // initial location
		addrput(0)      // address range

		for pciterinit(Ctxt, &pcsp, &s.Pcln.Pcsp); pcsp.done == 0; pciternext(&pcsp) {
			nextpc = pcsp.nextpc

			// pciterinit goes up to the end of the function,
			// but DWARF expects us to stop just before the end.
			if int64(nextpc) == s.Size {
				nextpc--
				if nextpc < pcsp.pc {
					continue
				}
			}

			putpccfadelta(int64(nextpc)-int64(pcsp.pc), int64(Thearch.Ptrsize)+int64(pcsp.value))
		}

		fdesize = Cpos() - fdeo - 4 // exclude the length field.
		pad = Rnd(fdesize, int64(Thearch.Ptrsize)) - fdesize
		strnput("", int(pad))
		fdesize += pad

		// Emit the FDE header for real, Section 6.4.1.
		Cseek(fdeo)

		Thearch.Lput(uint32(fdesize))
		if Linkmode == LinkExternal {
			adddwarfrel(framesec, framesym, frameo, 4, 0)
			adddwarfrel(framesec, s, frameo, Thearch.Ptrsize, 0)
		} else {
			Thearch.Lput(0)
			addrput(s.Value)
		}

		addrput(s.Size)
		Cseek(fdeo + 4 + fdesize)
	}

	Cflush()
	framesize = Cpos() - frameo
}

/*
 *  Walk DWarfDebugInfoEntries, and emit .debug_info
 */
const (
	COMPUNITHEADERSIZE = 4 + 2 + 4 + 1
)

func writeinfo() {
	fwdcount = 0
	if infosec == nil {
		infosec = Linklookup(Ctxt, ".dwarfinfo", 0)
	}
	infosec.R = infosec.R[:0]

	if arangessec == nil {
		arangessec = Linklookup(Ctxt, ".dwarfaranges", 0)
	}
	arangessec.R = arangessec.R[:0]

	var here int64
	var unitstart int64
	for compunit := dwroot.child; compunit != nil; compunit = compunit.link {
		unitstart = Cpos()

		// Write .debug_info Compilation Unit Header (sec 7.5.1)
		// Fields marked with (*) must be changed for 64-bit dwarf
		// This must match COMPUNITHEADERSIZE above.
		Thearch.Lput(0) // unit_length (*), will be filled in later.
		Thearch.Wput(2) // dwarf version (appendix F)

		// debug_abbrev_offset (*)
		if Linkmode == LinkExternal {
			adddwarfrel(infosec, abbrevsym, infoo, 4, 0)
		} else {
			Thearch.Lput(0)
		}

		Cput(uint8(Thearch.Ptrsize)) // address_size

		putdie(compunit)

		here = Cpos()
		Cseek(unitstart)
		Thearch.Lput(uint32(here - unitstart - 4)) // exclude the length field.
		Cseek(here)
	}

	Cflush()
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

func writepub(ispub func(*DWDie) bool) int64 {
	var die *DWDie
	var dwa *DWAttr
	var unitstart int64
	var unitend int64
	var here int64

	sectionstart := Cpos()

	for compunit := dwroot.child; compunit != nil; compunit = compunit.link {
		unitstart = compunit.offs - COMPUNITHEADERSIZE
		if compunit.link != nil {
			unitend = compunit.link.offs - COMPUNITHEADERSIZE
		} else {
			unitend = infoo + infosize
		}

		// Write .debug_pubnames/types	Header (sec 6.1.1)
		Thearch.Lput(0)                           // unit_length (*), will be filled in later.
		Thearch.Wput(2)                           // dwarf version (appendix F)
		Thearch.Lput(uint32(unitstart))           // debug_info_offset (of the Comp unit Header)
		Thearch.Lput(uint32(unitend - unitstart)) // debug_info_length

		for die = compunit.child; die != nil; die = die.link {
			if !ispub(die) {
				continue
			}
			Thearch.Lput(uint32(die.offs - unitstart))
			dwa = getattr(die, DW_AT_name)
			strnput(dwa.data.(string), int(dwa.value+1))
		}

		Thearch.Lput(0)

		here = Cpos()
		Cseek(sectionstart)
		Thearch.Lput(uint32(here - sectionstart - 4)) // exclude the length field.
		Cseek(here)
	}

	return sectionstart
}

/*
 *  emit .debug_aranges.  _info must have been written before,
 *  because we need die->offs of dw_globals.
 */
func writearanges() int64 {
	var b *DWAttr
	var e *DWAttr
	var value int64

	sectionstart := Cpos()
	headersize := int(Rnd(4+2+4+1+1, int64(Thearch.Ptrsize))) // don't count unit_length field itself

	for compunit := dwroot.child; compunit != nil; compunit = compunit.link {
		b = getattr(compunit, DW_AT_low_pc)
		if b == nil {
			continue
		}
		e = getattr(compunit, DW_AT_high_pc)
		if e == nil {
			continue
		}

		// Write .debug_aranges	 Header + entry	 (sec 6.1.2)
		Thearch.Lput(uint32(headersize) + 4*uint32(Thearch.Ptrsize) - 4) // unit_length (*)
		Thearch.Wput(2)                                                  // dwarf version (appendix F)

		value = compunit.offs - COMPUNITHEADERSIZE // debug_info_offset
		if Linkmode == LinkExternal {
			adddwarfrel(arangessec, infosym, sectionstart, 4, value)
		} else {
			Thearch.Lput(uint32(value))
		}

		Cput(uint8(Thearch.Ptrsize))        // address_size
		Cput(0)                             // segment_size
		strnput("", headersize-(4+2+4+1+1)) // align to thearch.ptrsize

		if Linkmode == LinkExternal {
			adddwarfrel(arangessec, b.data.(*LSym), sectionstart, Thearch.Ptrsize, b.value-(b.data.(*LSym)).Value)
		} else {
			addrput(b.value)
		}

		addrput(e.value - b.value)
		addrput(0)
		addrput(0)
	}

	Cflush()
	return sectionstart
}

func writegdbscript() int64 {
	sectionstart := Cpos()

	if gdbscript != "" {
		Cput(1) // magic 1 byte?
		strnput(gdbscript, len(gdbscript)+1)
		Cflush()
	}

	return sectionstart
}

func align(size int64) {
	if HEADTYPE == Hwindows { // Only Windows PE need section align.
		strnput("", int(Rnd(size, PEFILEALIGN)-size))
	}
}

func writedwarfreloc(s *LSym) int64 {
	var i int
	var r *Reloc

	start := Cpos()
	for ri := 0; ri < len(s.R); ri++ {
		r = &s.R[ri]
		if Iself {
			i = Thearch.Elfreloc1(r, int64(r.Off))
		} else if HEADTYPE == Hdarwin {
			i = Thearch.Machoreloc1(r, int64(r.Off))
		} else {
			i = -1
		}
		if i < 0 {
			Diag("unsupported obj reloc %d/%d to %s", r.Type, r.Siz, r.Sym.Name)
		}
	}

	return start
}

/*
 * This is the main entry point for generating dwarf.  After emitting
 * the mandatory debug_abbrev section, it calls writelines() to set up
 * the per-compilation unit part of the DIE tree, while simultaneously
 * emitting the debug_line section.  When the final tree contains
 * forward references, it will write the debug_info section in 2
 * passes.
 *
 */
func Dwarfemitdebugsections() {
	if Debug['w'] != 0 { // disable dwarf
		return
	}

	if Linkmode == LinkExternal && !Iself {
		return
	}

	// For diagnostic messages.
	newattr(&dwtypes, DW_AT_name, DW_CLS_STRING, int64(len("dwtypes")), "dwtypes")

	mkindex(&dwroot)
	mkindex(&dwtypes)
	mkindex(&dwglobals)

	// Some types that must exist to define other ones.
	newdie(&dwtypes, DW_ABRV_NULLTYPE, "<unspecified>")

	newdie(&dwtypes, DW_ABRV_NULLTYPE, "void")
	newdie(&dwtypes, DW_ABRV_BARE_PTRTYPE, "unsafe.Pointer")

	die := newdie(&dwtypes, DW_ABRV_BASETYPE, "uintptr") // needed for array size
	newattr(die, DW_AT_encoding, DW_CLS_CONSTANT, DW_ATE_unsigned, 0)
	newattr(die, DW_AT_byte_size, DW_CLS_CONSTANT, int64(Thearch.Ptrsize), 0)
	newattr(die, DW_AT_go_kind, DW_CLS_CONSTANT, obj.KindUintptr, 0)

	// Needed by the prettyprinter code for interface inspection.
	defgotype(lookup_or_diag("type.runtime._type"))

	defgotype(lookup_or_diag("type.runtime.interfacetype"))
	defgotype(lookup_or_diag("type.runtime.itab"))

	genasmsym(defdwsymb)

	writeabbrev()
	align(abbrevsize)
	writelines()
	align(linesize)
	writeframes()
	align(framesize)

	synthesizestringtypes(dwtypes.child)
	synthesizeslicetypes(dwtypes.child)
	synthesizemaptypes(dwtypes.child)
	synthesizechantypes(dwtypes.child)

	reversetree(&dwroot.child)
	reversetree(&dwtypes.child)
	reversetree(&dwglobals.child)

	movetomodule(&dwtypes)
	movetomodule(&dwglobals)

	infoo = Cpos()
	writeinfo()
	infoe := Cpos()
	pubnameso = infoe
	pubtypeso = infoe
	arangeso = infoe
	gdbscripto = infoe

	if fwdcount > 0 {
		if Debug['v'] != 0 {
			fmt.Fprintf(&Bso, "%5.2f dwarf pass 2.\n", obj.Cputime())
		}
		Cseek(infoo)
		writeinfo()
		if fwdcount > 0 {
			Diag("dwarf: unresolved references after first dwarf info pass")
			Errorexit()
		}

		if infoe != Cpos() {
			Diag("dwarf: inconsistent second dwarf info pass")
			Errorexit()
		}
	}

	infosize = infoe - infoo
	align(infosize)

	pubnameso = writepub(ispubname)
	pubnamessize = Cpos() - pubnameso
	align(pubnamessize)

	pubtypeso = writepub(ispubtype)
	pubtypessize = Cpos() - pubtypeso
	align(pubtypessize)

	arangeso = writearanges()
	arangessize = Cpos() - arangeso
	align(arangessize)

	gdbscripto = writegdbscript()
	gdbscriptsize = Cpos() - gdbscripto
	align(gdbscriptsize)

	for Cpos()&7 != 0 {
		Cput(0)
	}
	inforeloco = writedwarfreloc(infosec)
	inforelocsize = Cpos() - inforeloco
	align(inforelocsize)

	arangesreloco = writedwarfreloc(arangessec)
	arangesrelocsize = Cpos() - arangesreloco
	align(arangesrelocsize)

	linereloco = writedwarfreloc(linesec)
	linerelocsize = Cpos() - linereloco
	align(linerelocsize)

	framereloco = writedwarfreloc(framesec)
	framerelocsize = Cpos() - framereloco
	align(framerelocsize)
}

/*
 *  Elf.
 */
const (
	ElfStrDebugAbbrev = iota
	ElfStrDebugAranges
	ElfStrDebugFrame
	ElfStrDebugInfo
	ElfStrDebugLine
	ElfStrDebugLoc
	ElfStrDebugMacinfo
	ElfStrDebugPubNames
	ElfStrDebugPubTypes
	ElfStrDebugRanges
	ElfStrDebugStr
	ElfStrGDBScripts
	ElfStrRelDebugInfo
	ElfStrRelDebugAranges
	ElfStrRelDebugLine
	ElfStrRelDebugFrame
	NElfStrDbg
)

var elfstrdbg [NElfStrDbg]int64

func dwarfaddshstrings(shstrtab *LSym) {
	if Debug['w'] != 0 { // disable dwarf
		return
	}

	elfstrdbg[ElfStrDebugAbbrev] = Addstring(shstrtab, ".debug_abbrev")
	elfstrdbg[ElfStrDebugAranges] = Addstring(shstrtab, ".debug_aranges")
	elfstrdbg[ElfStrDebugFrame] = Addstring(shstrtab, ".debug_frame")
	elfstrdbg[ElfStrDebugInfo] = Addstring(shstrtab, ".debug_info")
	elfstrdbg[ElfStrDebugLine] = Addstring(shstrtab, ".debug_line")
	elfstrdbg[ElfStrDebugLoc] = Addstring(shstrtab, ".debug_loc")
	elfstrdbg[ElfStrDebugMacinfo] = Addstring(shstrtab, ".debug_macinfo")
	elfstrdbg[ElfStrDebugPubNames] = Addstring(shstrtab, ".debug_pubnames")
	elfstrdbg[ElfStrDebugPubTypes] = Addstring(shstrtab, ".debug_pubtypes")
	elfstrdbg[ElfStrDebugRanges] = Addstring(shstrtab, ".debug_ranges")
	elfstrdbg[ElfStrDebugStr] = Addstring(shstrtab, ".debug_str")
	elfstrdbg[ElfStrGDBScripts] = Addstring(shstrtab, ".debug_gdb_scripts")
	if Linkmode == LinkExternal {
		switch Thearch.Thechar {
		case '6', '7', '9':
			elfstrdbg[ElfStrRelDebugInfo] = Addstring(shstrtab, ".rela.debug_info")
			elfstrdbg[ElfStrRelDebugAranges] = Addstring(shstrtab, ".rela.debug_aranges")
			elfstrdbg[ElfStrRelDebugLine] = Addstring(shstrtab, ".rela.debug_line")
			elfstrdbg[ElfStrRelDebugFrame] = Addstring(shstrtab, ".rela.debug_frame")
		default:
			elfstrdbg[ElfStrRelDebugInfo] = Addstring(shstrtab, ".rel.debug_info")
			elfstrdbg[ElfStrRelDebugAranges] = Addstring(shstrtab, ".rel.debug_aranges")
			elfstrdbg[ElfStrRelDebugLine] = Addstring(shstrtab, ".rel.debug_line")
			elfstrdbg[ElfStrRelDebugFrame] = Addstring(shstrtab, ".rel.debug_frame")
		}

		infosym = Linklookup(Ctxt, ".debug_info", 0)
		infosym.Hide = 1

		abbrevsym = Linklookup(Ctxt, ".debug_abbrev", 0)
		abbrevsym.Hide = 1

		linesym = Linklookup(Ctxt, ".debug_line", 0)
		linesym.Hide = 1

		framesym = Linklookup(Ctxt, ".debug_frame", 0)
		framesym.Hide = 1
	}
}

// Add section symbols for DWARF debug info.  This is called before
// dwarfaddelfheaders.
func dwarfaddelfsectionsyms() {
	if infosym != nil {
		infosympos = Cpos()
		putelfsectionsym(infosym, 0)
	}

	if abbrevsym != nil {
		abbrevsympos = Cpos()
		putelfsectionsym(abbrevsym, 0)
	}

	if linesym != nil {
		linesympos = Cpos()
		putelfsectionsym(linesym, 0)
	}

	if framesym != nil {
		framesympos = Cpos()
		putelfsectionsym(framesym, 0)
	}
}

func dwarfaddelfrelocheader(elfstr int, shdata *ElfShdr, off int64, size int64) {
	sh := newElfShdr(elfstrdbg[elfstr])
	switch Thearch.Thechar {
	case '6', '7', '9':
		sh.type_ = SHT_RELA
	default:
		sh.type_ = SHT_REL
	}

	sh.entsize = uint64(Thearch.Ptrsize) * 2
	if sh.type_ == SHT_RELA {
		sh.entsize += uint64(Thearch.Ptrsize)
	}
	sh.link = uint32(elfshname(".symtab").shnum)
	sh.info = uint32(shdata.shnum)
	sh.off = uint64(off)
	sh.size = uint64(size)
	sh.addralign = uint64(Thearch.Ptrsize)
}

func dwarfaddelfheaders() {
	if Debug['w'] != 0 { // disable dwarf
		return
	}

	sh := newElfShdr(elfstrdbg[ElfStrDebugAbbrev])
	sh.type_ = SHT_PROGBITS
	sh.off = uint64(abbrevo)
	sh.size = uint64(abbrevsize)
	sh.addralign = 1
	if abbrevsympos > 0 {
		putelfsymshndx(abbrevsympos, sh.shnum)
	}

	sh = newElfShdr(elfstrdbg[ElfStrDebugLine])
	sh.type_ = SHT_PROGBITS
	sh.off = uint64(lineo)
	sh.size = uint64(linesize)
	sh.addralign = 1
	if linesympos > 0 {
		putelfsymshndx(linesympos, sh.shnum)
	}
	shline := sh

	sh = newElfShdr(elfstrdbg[ElfStrDebugFrame])
	sh.type_ = SHT_PROGBITS
	sh.off = uint64(frameo)
	sh.size = uint64(framesize)
	sh.addralign = 1
	if framesympos > 0 {
		putelfsymshndx(framesympos, sh.shnum)
	}
	shframe := sh

	sh = newElfShdr(elfstrdbg[ElfStrDebugInfo])
	sh.type_ = SHT_PROGBITS
	sh.off = uint64(infoo)
	sh.size = uint64(infosize)
	sh.addralign = 1
	if infosympos > 0 {
		putelfsymshndx(infosympos, sh.shnum)
	}
	shinfo := sh

	if pubnamessize > 0 {
		sh := newElfShdr(elfstrdbg[ElfStrDebugPubNames])
		sh.type_ = SHT_PROGBITS
		sh.off = uint64(pubnameso)
		sh.size = uint64(pubnamessize)
		sh.addralign = 1
	}

	if pubtypessize > 0 {
		sh := newElfShdr(elfstrdbg[ElfStrDebugPubTypes])
		sh.type_ = SHT_PROGBITS
		sh.off = uint64(pubtypeso)
		sh.size = uint64(pubtypessize)
		sh.addralign = 1
	}

	var sharanges *ElfShdr
	if arangessize != 0 {
		sh := newElfShdr(elfstrdbg[ElfStrDebugAranges])
		sh.type_ = SHT_PROGBITS
		sh.off = uint64(arangeso)
		sh.size = uint64(arangessize)
		sh.addralign = 1
		sharanges = sh
	}

	if gdbscriptsize != 0 {
		sh := newElfShdr(elfstrdbg[ElfStrGDBScripts])
		sh.type_ = SHT_PROGBITS
		sh.off = uint64(gdbscripto)
		sh.size = uint64(gdbscriptsize)
		sh.addralign = 1
	}

	if inforelocsize != 0 {
		dwarfaddelfrelocheader(ElfStrRelDebugInfo, shinfo, inforeloco, inforelocsize)
	}

	if arangesrelocsize != 0 {
		dwarfaddelfrelocheader(ElfStrRelDebugAranges, sharanges, arangesreloco, arangesrelocsize)
	}

	if linerelocsize != 0 {
		dwarfaddelfrelocheader(ElfStrRelDebugLine, shline, linereloco, linerelocsize)
	}

	if framerelocsize != 0 {
		dwarfaddelfrelocheader(ElfStrRelDebugFrame, shframe, framereloco, framerelocsize)
	}
}

/*
 * Macho
 */
func dwarfaddmachoheaders() {
	if Debug['w'] != 0 { // disable dwarf
		return
	}

	// Zero vsize segments won't be loaded in memory, even so they
	// have to be page aligned in the file.
	fakestart := abbrevo &^ 0xfff

	nsect := 4
	if pubnamessize > 0 {
		nsect++
	}
	if pubtypessize > 0 {
		nsect++
	}
	if arangessize > 0 {
		nsect++
	}
	if gdbscriptsize > 0 {
		nsect++
	}

	ms := newMachoSeg("__DWARF", nsect)
	ms.fileoffset = uint64(fakestart)
	ms.filesize = uint64(abbrevo) - uint64(fakestart)
	ms.vaddr = ms.fileoffset + Segdata.Vaddr - Segdata.Fileoff

	msect := newMachoSect(ms, "__debug_abbrev", "__DWARF")
	msect.off = uint32(abbrevo)
	msect.size = uint64(abbrevsize)
	msect.addr = uint64(msect.off) + Segdata.Vaddr - Segdata.Fileoff
	ms.filesize += msect.size

	msect = newMachoSect(ms, "__debug_line", "__DWARF")
	msect.off = uint32(lineo)
	msect.size = uint64(linesize)
	msect.addr = uint64(msect.off) + Segdata.Vaddr - Segdata.Fileoff
	ms.filesize += msect.size

	msect = newMachoSect(ms, "__debug_frame", "__DWARF")
	msect.off = uint32(frameo)
	msect.size = uint64(framesize)
	msect.addr = uint64(msect.off) + Segdata.Vaddr - Segdata.Fileoff
	ms.filesize += msect.size

	msect = newMachoSect(ms, "__debug_info", "__DWARF")
	msect.off = uint32(infoo)
	msect.size = uint64(infosize)
	msect.addr = uint64(msect.off) + Segdata.Vaddr - Segdata.Fileoff
	ms.filesize += msect.size

	if pubnamessize > 0 {
		msect := newMachoSect(ms, "__debug_pubnames", "__DWARF")
		msect.off = uint32(pubnameso)
		msect.size = uint64(pubnamessize)
		msect.addr = uint64(msect.off) + Segdata.Vaddr - Segdata.Fileoff
		ms.filesize += msect.size
	}

	if pubtypessize > 0 {
		msect := newMachoSect(ms, "__debug_pubtypes", "__DWARF")
		msect.off = uint32(pubtypeso)
		msect.size = uint64(pubtypessize)
		msect.addr = uint64(msect.off) + Segdata.Vaddr - Segdata.Fileoff
		ms.filesize += msect.size
	}

	if arangessize > 0 {
		msect := newMachoSect(ms, "__debug_aranges", "__DWARF")
		msect.off = uint32(arangeso)
		msect.size = uint64(arangessize)
		msect.addr = uint64(msect.off) + Segdata.Vaddr - Segdata.Fileoff
		ms.filesize += msect.size
	}

	// TODO(lvd) fix gdb/python to load MachO (16 char section name limit)
	if gdbscriptsize > 0 {
		msect := newMachoSect(ms, "__debug_gdb_scripts", "__DWARF")
		msect.off = uint32(gdbscripto)
		msect.size = uint64(gdbscriptsize)
		msect.addr = uint64(msect.off) + Segdata.Vaddr - Segdata.Fileoff
		ms.filesize += msect.size
	}
}

/*
 * Windows PE
 */
func dwarfaddpeheaders() {
	if Debug['w'] != 0 { // disable dwarf
		return
	}

	newPEDWARFSection(".debug_abbrev", abbrevsize)
	newPEDWARFSection(".debug_line", linesize)
	newPEDWARFSection(".debug_frame", framesize)
	newPEDWARFSection(".debug_info", infosize)
	newPEDWARFSection(".debug_pubnames", pubnamessize)
	newPEDWARFSection(".debug_pubtypes", pubtypessize)
	newPEDWARFSection(".debug_aranges", arangessize)
	newPEDWARFSection(".debug_gdb_scripts", gdbscriptsize)
}
