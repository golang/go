// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package dwarf generates DWARF debugging information.
// DWARF generation is split between the compiler and the linker,
// this package contains the shared code.
package dwarf

import (
	"fmt"
)

// InfoPrefix is the prefix for all the symbols containing DWARF info entries.
const InfoPrefix = "go.info."

// Sym represents a symbol.
type Sym interface {
}

// A Var represents a local variable or a function parameter.
type Var struct {
	Name   string
	Abbrev int // Either DW_ABRV_AUTO or DW_ABRV_PARAM
	Offset int32
	Type   Sym
	Link   *Var
}

// A Context specifies how to add data to a Sym.
type Context interface {
	PtrSize() int
	AddInt(s Sym, size int, i int64)
	AddBytes(s Sym, b []byte)
	AddAddress(s Sym, t interface{}, ofs int64)
	AddSectionOffset(s Sym, size int, t interface{}, ofs int64)
	AddString(s Sym, v string)
	SymValue(s Sym) int64
}

// AppendUleb128 appends v to b using DWARF's unsigned LEB128 encoding.
func AppendUleb128(b []byte, v uint64) []byte {
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

// AppendSleb128 appends v to b using DWARF's signed LEB128 encoding.
func AppendSleb128(b []byte, v int64) []byte {
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

var encbuf [20]byte

// AppendUleb128 appends v to s using DWARF's unsigned LEB128 encoding.
func Uleb128put(ctxt Context, s Sym, v int64) {
	b := AppendUleb128(encbuf[:0], uint64(v))
	ctxt.AddBytes(s, b)
}

// AppendUleb128 appends v to s using DWARF's signed LEB128 encoding.
func Sleb128put(ctxt Context, s Sym, v int64) {
	b := AppendSleb128(encbuf[:0], v)
	ctxt.AddBytes(s, b)
}

/*
 * Defining Abbrevs.  This is hardcoded, and there will be
 * only a handful of them.  The DWARF spec places no restriction on
 * the ordering of attributes in the Abbrevs and DIEs, and we will
 * always write them out in the order of declaration in the abbrev.
 */
type dwAttrForm struct {
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

type dwAbbrev struct {
	tag      uint8
	children uint8
	attr     []dwAttrForm
}

var abbrevs = [DW_NABRV]dwAbbrev{
	/* The mandatory DW_ABRV_NULL entry. */
	{0, 0, []dwAttrForm{}},

	/* COMPUNIT */
	{
		DW_TAG_compile_unit,
		DW_CHILDREN_yes,
		[]dwAttrForm{
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
		[]dwAttrForm{
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
		[]dwAttrForm{
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
		[]dwAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_location, DW_FORM_block1},
			{DW_AT_type, DW_FORM_ref_addr},
		},
	},

	/* PARAM */
	{
		DW_TAG_formal_parameter,
		DW_CHILDREN_no,
		[]dwAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_location, DW_FORM_block1},
			{DW_AT_type, DW_FORM_ref_addr},
		},
	},

	/* STRUCTFIELD */
	{
		DW_TAG_member,
		DW_CHILDREN_no,
		[]dwAttrForm{
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
		[]dwAttrForm{
			{DW_AT_type, DW_FORM_ref_addr},
		},
	},

	/* DOTDOTDOT */
	{
		DW_TAG_unspecified_parameters,
		DW_CHILDREN_no,
		[]dwAttrForm{},
	},

	/* ARRAYRANGE */
	{
		DW_TAG_subrange_type,
		DW_CHILDREN_no,

		// No name!
		[]dwAttrForm{
			{DW_AT_type, DW_FORM_ref_addr},
			{DW_AT_count, DW_FORM_udata},
		},
	},

	// Below here are the types considered public by ispubtype
	/* NULLTYPE */
	{
		DW_TAG_unspecified_type,
		DW_CHILDREN_no,
		[]dwAttrForm{
			{DW_AT_name, DW_FORM_string},
		},
	},

	/* BASETYPE */
	{
		DW_TAG_base_type,
		DW_CHILDREN_no,
		[]dwAttrForm{
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
		[]dwAttrForm{
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
		[]dwAttrForm{
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
		[]dwAttrForm{
			{DW_AT_name, DW_FORM_string},
			// {DW_AT_type,	DW_FORM_ref_addr},
			{DW_AT_go_kind, DW_FORM_data1},
		},
	},

	/* IFACETYPE */
	{
		DW_TAG_typedef,
		DW_CHILDREN_yes,
		[]dwAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_type, DW_FORM_ref_addr},
			{DW_AT_go_kind, DW_FORM_data1},
		},
	},

	/* MAPTYPE */
	{
		DW_TAG_typedef,
		DW_CHILDREN_no,
		[]dwAttrForm{
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
		[]dwAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_type, DW_FORM_ref_addr},
			{DW_AT_go_kind, DW_FORM_data1},
		},
	},

	/* BARE_PTRTYPE */
	{
		DW_TAG_pointer_type,
		DW_CHILDREN_no,
		[]dwAttrForm{
			{DW_AT_name, DW_FORM_string},
		},
	},

	/* SLICETYPE */
	{
		DW_TAG_structure_type,
		DW_CHILDREN_yes,
		[]dwAttrForm{
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
		[]dwAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_byte_size, DW_FORM_udata},
			{DW_AT_go_kind, DW_FORM_data1},
		},
	},

	/* STRUCTTYPE */
	{
		DW_TAG_structure_type,
		DW_CHILDREN_yes,
		[]dwAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_byte_size, DW_FORM_udata},
			{DW_AT_go_kind, DW_FORM_data1},
		},
	},

	/* TYPEDECL */
	{
		DW_TAG_typedef,
		DW_CHILDREN_no,
		[]dwAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_type, DW_FORM_ref_addr},
		},
	},
}

// GetAbbrev returns the contents of the .debug_abbrev section.
func GetAbbrev() []byte {
	var buf []byte
	for i := 1; i < DW_NABRV; i++ {
		// See section 7.5.3
		buf = AppendUleb128(buf, uint64(i))

		buf = AppendUleb128(buf, uint64(abbrevs[i].tag))
		buf = append(buf, byte(abbrevs[i].children))
		for _, f := range abbrevs[i].attr {
			buf = AppendUleb128(buf, uint64(f.attr))
			buf = AppendUleb128(buf, uint64(f.form))
		}
		buf = append(buf, 0, 0)
	}
	return append(buf, 0)
}

/*
 * Debugging Information Entries and their attributes.
 */

// DWAttr represents an attribute of a DWDie.
//
// For DW_CLS_string and _block, value should contain the length, and
// data the data, for _reference, value is 0 and data is a DWDie* to
// the referenced instance, for all others, value is the whole thing
// and data is null.
type DWAttr struct {
	Link  *DWAttr
	Atr   uint16 // DW_AT_
	Cls   uint8  // DW_CLS_
	Value int64
	Data  interface{}
}

// DWDie represents a DWARF debug info entry.
type DWDie struct {
	Abbrev int
	Link   *DWDie
	Child  *DWDie
	Attr   *DWAttr
	Sym    Sym
}

func putattr(ctxt Context, s Sym, abbrev int, form int, cls int, value int64, data interface{}) error {
	switch form {
	case DW_FORM_addr: // address
		ctxt.AddAddress(s, data, value)

	case DW_FORM_block1: // block
		if cls == DW_CLS_ADDRESS {
			ctxt.AddInt(s, 1, int64(1+ctxt.PtrSize()))
			ctxt.AddInt(s, 1, DW_OP_addr)
			ctxt.AddAddress(s, data, 0)
			break
		}

		value &= 0xff
		ctxt.AddInt(s, 1, value)
		p := data.([]byte)[:value]
		ctxt.AddBytes(s, p)

	case DW_FORM_block2: // block
		value &= 0xffff

		ctxt.AddInt(s, 2, value)
		p := data.([]byte)[:value]
		ctxt.AddBytes(s, p)

	case DW_FORM_block4: // block
		value &= 0xffffffff

		ctxt.AddInt(s, 4, value)
		p := data.([]byte)[:value]
		ctxt.AddBytes(s, p)

	case DW_FORM_block: // block
		Uleb128put(ctxt, s, value)

		p := data.([]byte)[:value]
		ctxt.AddBytes(s, p)

	case DW_FORM_data1: // constant
		ctxt.AddInt(s, 1, value)

	case DW_FORM_data2: // constant
		ctxt.AddInt(s, 2, value)

	case DW_FORM_data4: // constant, {line,loclist,mac,rangelist}ptr
		if cls == DW_CLS_PTR { // DW_AT_stmt_list
			ctxt.AddSectionOffset(s, 4, data, 0)
			break
		}
		ctxt.AddInt(s, 4, value)

	case DW_FORM_data8: // constant, {line,loclist,mac,rangelist}ptr
		ctxt.AddInt(s, 8, value)

	case DW_FORM_sdata: // constant
		Sleb128put(ctxt, s, value)

	case DW_FORM_udata: // constant
		Uleb128put(ctxt, s, value)

	case DW_FORM_string: // string
		str := data.(string)
		ctxt.AddString(s, str)
		// TODO(ribrdb): verify padded strings are never used and remove this
		for i := int64(len(str)); i < value; i++ {
			ctxt.AddInt(s, 1, 0)
		}

	case DW_FORM_flag: // flag
		if value != 0 {
			ctxt.AddInt(s, 1, 1)
		} else {
			ctxt.AddInt(s, 1, 0)
		}

	// In DWARF 2 (which is what we claim to generate),
	// the ref_addr is the same size as a normal address.
	// In DWARF 3 it is always 32 bits, unless emitting a large
	// (> 4 GB of debug info aka "64-bit") unit, which we don't implement.
	case DW_FORM_ref_addr: // reference to a DIE in the .info section
		if data == nil {
			return fmt.Errorf("dwarf: null reference in %d", abbrev)
		} else {
			ctxt.AddSectionOffset(s, ctxt.PtrSize(), data, 0)
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
		return fmt.Errorf("dwarf: unsupported attribute form %d / class %d", form, cls)
	}
	return nil
}

// PutAttrs writes the attributes for a DIE to symbol 's'.
//
// Note that we can (and do) add arbitrary attributes to a DIE, but
// only the ones actually listed in the Abbrev will be written out.
func PutAttrs(ctxt Context, s Sym, abbrev int, attr *DWAttr) {
Outer:
	for _, f := range abbrevs[abbrev].attr {
		for ap := attr; ap != nil; ap = ap.Link {
			if ap.Atr == f.attr {
				putattr(ctxt, s, abbrev, int(f.form), int(ap.Cls), ap.Value, ap.Data)
				continue Outer
			}
		}

		putattr(ctxt, s, abbrev, int(f.form), 0, 0, nil)
	}
}

// HasChildren returns true if 'die' uses an abbrev that supports children.
func HasChildren(die *DWDie) bool {
	return abbrevs[die.Abbrev].children != 0
}

// PutFunc writes a DIE for a function to s.
// It also writes child DIEs for each variable in vars.
func PutFunc(ctxt Context, s Sym, name string, external bool, startPC Sym, size int64, vars *Var) {
	Uleb128put(ctxt, s, DW_ABRV_FUNCTION)
	putattr(ctxt, s, DW_ABRV_FUNCTION, DW_FORM_string, DW_CLS_STRING, int64(len(name)), name)
	putattr(ctxt, s, DW_ABRV_FUNCTION, DW_FORM_addr, DW_CLS_ADDRESS, 0, startPC)
	putattr(ctxt, s, DW_ABRV_FUNCTION, DW_FORM_addr, DW_CLS_ADDRESS, size+ctxt.SymValue(startPC), startPC)
	var ev int64
	if external {
		ev = 1
	}
	putattr(ctxt, s, DW_ABRV_FUNCTION, DW_FORM_flag, DW_CLS_FLAG, ev, 0)
	names := make(map[string]bool)
	for v := vars; v != nil; v = v.Link {
		var n string
		if names[v.Name] {
			n = fmt.Sprintf("%s#%d", v.Name, len(names))
		} else {
			n = v.Name
		}
		names[n] = true

		Uleb128put(ctxt, s, int64(v.Abbrev))
		putattr(ctxt, s, v.Abbrev, DW_FORM_string, DW_CLS_STRING, int64(len(n)), n)
		loc := append(encbuf[:0], DW_OP_call_frame_cfa)
		if v.Offset != 0 {
			loc = append(loc, DW_OP_consts)
			loc = AppendSleb128(loc, int64(v.Offset))
			loc = append(loc, DW_OP_plus)
		}
		putattr(ctxt, s, v.Abbrev, DW_FORM_block1, DW_CLS_BLOCK, int64(len(loc)), loc)
		putattr(ctxt, s, v.Abbrev, DW_FORM_ref_addr, DW_CLS_REFERENCE, 0, v.Type)

	}
	Uleb128put(ctxt, s, 0)
}
