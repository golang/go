// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package dwarf generates DWARF debugging information.
// DWARF generation is split between the compiler and the linker,
// this package contains the shared code.
package dwarf

import (
	"errors"
	"fmt"
)

// InfoPrefix is the prefix for all the symbols containing DWARF info entries.
const InfoPrefix = "go.info."

// RangePrefix is the prefix for all the symbols containing DWARF range lists.
const RangePrefix = "go.range."

// Sym represents a symbol.
type Sym interface {
	Len() int64
}

// A Var represents a local variable or a function parameter.
type Var struct {
	Name   string
	Abbrev int // Either DW_ABRV_AUTO or DW_ABRV_PARAM
	Offset int32
	Scope  int32
	Type   Sym
}

// A Scope represents a lexical scope. All variables declared within a
// scope will only be visible to instructions covered by the scope.
// Lexical scopes are contiguous in source files but can end up being
// compiled to discontiguous blocks of instructions in the executable.
// The Ranges field lists all the blocks of instructions that belong
// in this scope.
type Scope struct {
	Parent int32
	Ranges []Range
	Vars   []*Var
}

// A Range represents a half-open interval [Start, End).
type Range struct {
	Start, End int64
}

// UnifyRanges merges the list of ranges of c into the list of ranges of s
func (s *Scope) UnifyRanges(c *Scope) {
	out := make([]Range, 0, len(s.Ranges)+len(c.Ranges))

	i, j := 0, 0
	for {
		var cur Range
		if i < len(s.Ranges) && j < len(c.Ranges) {
			if s.Ranges[i].Start < c.Ranges[j].Start {
				cur = s.Ranges[i]
				i++
			} else {
				cur = c.Ranges[j]
				j++
			}
		} else if i < len(s.Ranges) {
			cur = s.Ranges[i]
			i++
		} else if j < len(c.Ranges) {
			cur = c.Ranges[j]
			j++
		} else {
			break
		}

		if n := len(out); n > 0 && cur.Start <= out[n-1].End {
			out[n-1].End = cur.End
		} else {
			out = append(out, cur)
		}
	}

	s.Ranges = out
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

// sevenbits contains all unsigned seven bit numbers, indexed by their value.
var sevenbits = [...]byte{
	0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f,
	0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f,
	0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2a, 0x2b, 0x2c, 0x2d, 0x2e, 0x2f,
	0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39, 0x3a, 0x3b, 0x3c, 0x3d, 0x3e, 0x3f,
	0x40, 0x41, 0x42, 0x43, 0x44, 0x45, 0x46, 0x47, 0x48, 0x49, 0x4a, 0x4b, 0x4c, 0x4d, 0x4e, 0x4f,
	0x50, 0x51, 0x52, 0x53, 0x54, 0x55, 0x56, 0x57, 0x58, 0x59, 0x5a, 0x5b, 0x5c, 0x5d, 0x5e, 0x5f,
	0x60, 0x61, 0x62, 0x63, 0x64, 0x65, 0x66, 0x67, 0x68, 0x69, 0x6a, 0x6b, 0x6c, 0x6d, 0x6e, 0x6f,
	0x70, 0x71, 0x72, 0x73, 0x74, 0x75, 0x76, 0x77, 0x78, 0x79, 0x7a, 0x7b, 0x7c, 0x7d, 0x7e, 0x7f,
}

// sevenBitU returns the unsigned LEB128 encoding of v if v is seven bits and nil otherwise.
// The contents of the returned slice must not be modified.
func sevenBitU(v int64) []byte {
	if uint64(v) < uint64(len(sevenbits)) {
		return sevenbits[v : v+1]
	}
	return nil
}

// sevenBitS returns the signed LEB128 encoding of v if v is seven bits and nil otherwise.
// The contents of the returned slice must not be modified.
func sevenBitS(v int64) []byte {
	if uint64(v) <= 63 {
		return sevenbits[v : v+1]
	}
	if uint64(-v) <= 64 {
		return sevenbits[128+v : 128+v+1]
	}
	return nil
}

// Uleb128put appends v to s using DWARF's unsigned LEB128 encoding.
func Uleb128put(ctxt Context, s Sym, v int64) {
	b := sevenBitU(v)
	if b == nil {
		var encbuf [20]byte
		b = AppendUleb128(encbuf[:0], uint64(v))
	}
	ctxt.AddBytes(s, b)
}

// Sleb128put appends v to s using DWARF's signed LEB128 encoding.
func Sleb128put(ctxt Context, s Sym, v int64) {
	b := sevenBitS(v)
	if b == nil {
		var encbuf [20]byte
		b = AppendSleb128(encbuf[:0], v)
	}
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
	// Attribute for DW_TAG_member of a struct type.
	// Nonzero value indicates the struct field is an embedded field.
	DW_AT_go_embedded_field = 0x2903

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
	DW_ABRV_LEXICAL_BLOCK_RANGES
	DW_ABRV_LEXICAL_BLOCK_SIMPLE
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
			{DW_AT_stmt_list, DW_FORM_sec_offset},
			{DW_AT_comp_dir, DW_FORM_string},
			{DW_AT_producer, DW_FORM_string},
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
			{DW_AT_frame_base, DW_FORM_block1},
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
	/* LEXICAL_BLOCK_RANGES */
	{
		DW_TAG_lexical_block,
		DW_CHILDREN_yes,
		[]dwAttrForm{
			{DW_AT_ranges, DW_FORM_sec_offset},
		},
	},

	/* LEXICAL_BLOCK_SIMPLE */
	{
		DW_TAG_lexical_block,
		DW_CHILDREN_yes,
		[]dwAttrForm{
			{DW_AT_low_pc, DW_FORM_addr},
			{DW_AT_high_pc, DW_FORM_addr},
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
			{DW_AT_go_embedded_field, DW_FORM_flag},
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
		if cls == DW_CLS_PTR { // DW_AT_stmt_list and DW_AT_ranges
			ctxt.AddSectionOffset(s, 4, data, value)
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

	// As of DWARF 3 the ref_addr is always 32 bits, unless emitting a large
	// (> 4 GB of debug info aka "64-bit") unit, which we don't implement.
	case DW_FORM_ref_addr: // reference to a DIE in the .info section
		fallthrough
	case DW_FORM_sec_offset: // offset into a DWARF section other than .info
		if data == nil {
			return fmt.Errorf("dwarf: null reference in %d", abbrev)
		}
		ctxt.AddSectionOffset(s, 4, data, value)

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
func PutFunc(ctxt Context, s, ranges Sym, name string, external bool, startPC Sym, size int64, scopes []Scope) error {
	Uleb128put(ctxt, s, DW_ABRV_FUNCTION)
	putattr(ctxt, s, DW_ABRV_FUNCTION, DW_FORM_string, DW_CLS_STRING, int64(len(name)), name)
	putattr(ctxt, s, DW_ABRV_FUNCTION, DW_FORM_addr, DW_CLS_ADDRESS, 0, startPC)
	putattr(ctxt, s, DW_ABRV_FUNCTION, DW_FORM_addr, DW_CLS_ADDRESS, size, startPC)
	putattr(ctxt, s, DW_ABRV_FUNCTION, DW_FORM_block1, DW_CLS_BLOCK, 1, []byte{DW_OP_call_frame_cfa})
	var ev int64
	if external {
		ev = 1
	}
	putattr(ctxt, s, DW_ABRV_FUNCTION, DW_FORM_flag, DW_CLS_FLAG, ev, 0)
	if len(scopes) > 0 {
		var encbuf [20]byte
		if putscope(ctxt, s, ranges, startPC, 0, scopes, encbuf[:0]) < int32(len(scopes)) {
			return errors.New("multiple toplevel scopes")
		}
	}

	Uleb128put(ctxt, s, 0)
	return nil
}

func putscope(ctxt Context, s, ranges Sym, startPC Sym, curscope int32, scopes []Scope, encbuf []byte) int32 {
	for _, v := range scopes[curscope].Vars {
		putvar(ctxt, s, v, encbuf)
	}
	this := curscope
	curscope++
	for curscope < int32(len(scopes)) {
		scope := scopes[curscope]
		if scope.Parent != this {
			return curscope
		}

		if len(scope.Ranges) == 1 {
			Uleb128put(ctxt, s, DW_ABRV_LEXICAL_BLOCK_SIMPLE)
			putattr(ctxt, s, DW_ABRV_LEXICAL_BLOCK_SIMPLE, DW_FORM_addr, DW_CLS_ADDRESS, scope.Ranges[0].Start, startPC)
			putattr(ctxt, s, DW_ABRV_LEXICAL_BLOCK_SIMPLE, DW_FORM_addr, DW_CLS_ADDRESS, scope.Ranges[0].End, startPC)
		} else {
			Uleb128put(ctxt, s, DW_ABRV_LEXICAL_BLOCK_RANGES)
			putattr(ctxt, s, DW_ABRV_LEXICAL_BLOCK_RANGES, DW_FORM_sec_offset, DW_CLS_PTR, ranges.Len(), ranges)

			ctxt.AddAddress(ranges, nil, -1)
			ctxt.AddAddress(ranges, startPC, 0)
			for _, r := range scope.Ranges {
				ctxt.AddAddress(ranges, nil, r.Start)
				ctxt.AddAddress(ranges, nil, r.End)
			}
			ctxt.AddAddress(ranges, nil, 0)
			ctxt.AddAddress(ranges, nil, 0)
		}

		curscope = putscope(ctxt, s, ranges, startPC, curscope, scopes, encbuf)

		Uleb128put(ctxt, s, 0)
	}
	return curscope
}

func putvar(ctxt Context, s Sym, v *Var, encbuf []byte) {
	n := v.Name

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

// VarsByOffset attaches the methods of sort.Interface to []*Var,
// sorting in increasing Offset.
type VarsByOffset []*Var

func (s VarsByOffset) Len() int           { return len(s) }
func (s VarsByOffset) Less(i, j int) bool { return s[i].Offset < s[j].Offset }
func (s VarsByOffset) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }
