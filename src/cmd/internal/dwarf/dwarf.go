// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package dwarf generates DWARF debugging information.
// DWARF generation is split between the compiler and the linker,
// this package contains the shared code.
package dwarf

import (
	"bytes"
	"cmd/internal/objabi"
	"errors"
	"fmt"
	exec "internal/execabs"
	"sort"
	"strconv"
	"strings"
)

// InfoPrefix is the prefix for all the symbols containing DWARF info entries.
const InfoPrefix = "go.info."

// ConstInfoPrefix is the prefix for all symbols containing DWARF info
// entries that contain constants.
const ConstInfoPrefix = "go.constinfo."

// CUInfoPrefix is the prefix for symbols containing information to
// populate the DWARF compilation unit info entries.
const CUInfoPrefix = "go.cuinfo."

// Used to form the symbol name assigned to the DWARF 'abstract subprogram"
// info entry for a function
const AbstractFuncSuffix = "$abstract"

// Controls logging/debugging for selected aspects of DWARF subprogram
// generation (functions, scopes).
var logDwarf bool

// Sym represents a symbol.
type Sym interface {
	Length(dwarfContext interface{}) int64
}

// A Var represents a local variable or a function parameter.
type Var struct {
	Name          string
	Abbrev        int // Either DW_ABRV_AUTO[_LOCLIST] or DW_ABRV_PARAM[_LOCLIST]
	IsReturnValue bool
	IsInlFormal   bool
	StackOffset   int32
	// This package can't use the ssa package, so it can't mention ssa.FuncDebug,
	// so indirect through a closure.
	PutLocationList func(listSym, startPC Sym)
	Scope           int32
	Type            Sym
	DeclFile        string
	DeclLine        uint
	DeclCol         uint
	InlIndex        int32 // subtract 1 to form real index into InlTree
	ChildIndex      int32 // child DIE index in abstract function
	IsInAbstract    bool  // variable exists in abstract function
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

// This container is used by the PutFunc* variants below when
// creating the DWARF subprogram DIE(s) for a function.
type FnState struct {
	Name          string
	Importpath    string
	Info          Sym
	Filesym       Sym
	Loc           Sym
	Ranges        Sym
	Absfn         Sym
	StartPC       Sym
	Size          int64
	External      bool
	Scopes        []Scope
	InlCalls      InlCalls
	UseBASEntries bool
}

func EnableLogging(doit bool) {
	logDwarf = doit
}

// MergeRanges creates a new range list by merging the ranges from
// its two arguments, then returns the new list.
func MergeRanges(in1, in2 []Range) []Range {
	out := make([]Range, 0, len(in1)+len(in2))
	i, j := 0, 0
	for {
		var cur Range
		if i < len(in2) && j < len(in1) {
			if in2[i].Start < in1[j].Start {
				cur = in2[i]
				i++
			} else {
				cur = in1[j]
				j++
			}
		} else if i < len(in2) {
			cur = in2[i]
			i++
		} else if j < len(in1) {
			cur = in1[j]
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

	return out
}

// UnifyRanges merges the ranges from 'c' into the list of ranges for 's'.
func (s *Scope) UnifyRanges(c *Scope) {
	s.Ranges = MergeRanges(s.Ranges, c.Ranges)
}

// AppendRange adds r to s, if r is non-empty.
// If possible, it extends the last Range in s.Ranges; if not, it creates a new one.
func (s *Scope) AppendRange(r Range) {
	if r.End <= r.Start {
		return
	}
	i := len(s.Ranges)
	if i > 0 && s.Ranges[i-1].End == r.Start {
		s.Ranges[i-1].End = r.End
		return
	}
	s.Ranges = append(s.Ranges, r)
}

type InlCalls struct {
	Calls []InlCall
}

type InlCall struct {
	// index into ctx.InlTree describing the call inlined here
	InlIndex int

	// Symbol of file containing inlined call site (really *obj.LSym).
	CallFile Sym

	// Line number of inlined call site.
	CallLine uint32

	// Dwarf abstract subroutine symbol (really *obj.LSym).
	AbsFunSym Sym

	// Indices of child inlines within Calls array above.
	Children []int

	// entries in this list are PAUTO's created by the inliner to
	// capture the promoted formals and locals of the inlined callee.
	InlVars []*Var

	// PC ranges for this inlined call.
	Ranges []Range

	// Root call (not a child of some other call).
	Root bool
}

// A Context specifies how to add data to a Sym.
type Context interface {
	PtrSize() int
	AddInt(s Sym, size int, i int64)
	AddBytes(s Sym, b []byte)
	AddAddress(s Sym, t interface{}, ofs int64)
	AddCURelativeAddress(s Sym, t interface{}, ofs int64)
	AddSectionOffset(s Sym, size int, t interface{}, ofs int64)
	AddDWARFAddrSectionOffset(s Sym, t interface{}, ofs int64)
	CurrentOffset(s Sym) int64
	RecordDclReference(from Sym, to Sym, dclIdx int, inlIndex int)
	RecordChildDieOffsets(s Sym, vars []*Var, offsets []int32)
	AddString(s Sym, v string)
	AddFileRef(s Sym, f interface{})
	Logf(format string, args ...interface{})
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
 * Defining Abbrevs. This is hardcoded on a per-platform basis (that is,
 * each platform will see a fixed abbrev table for all objects); the number
 * of abbrev entries is fairly small (compared to C++ objects).  The DWARF
 * spec places no restriction on the ordering of attributes in the
 * Abbrevs and DIEs, and we will always write them out in the order
 * of declaration in the abbrev.
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
	DW_AT_go_runtime_type   = 0x2904

	DW_AT_go_package_name = 0x2905 // Attribute for DW_TAG_compile_unit

	DW_AT_internal_location = 253 // params and locals; not emitted
)

// Index into the abbrevs table below.
// Keep in sync with ispubname() and ispubtype() in ld/dwarf.go.
// ispubtype considers >= NULLTYPE public
const (
	DW_ABRV_NULL = iota
	DW_ABRV_COMPUNIT
	DW_ABRV_COMPUNIT_TEXTLESS
	DW_ABRV_FUNCTION
	DW_ABRV_FUNCTION_ABSTRACT
	DW_ABRV_FUNCTION_CONCRETE
	DW_ABRV_INLINED_SUBROUTINE
	DW_ABRV_INLINED_SUBROUTINE_RANGES
	DW_ABRV_VARIABLE
	DW_ABRV_INT_CONSTANT
	DW_ABRV_AUTO
	DW_ABRV_AUTO_LOCLIST
	DW_ABRV_AUTO_ABSTRACT
	DW_ABRV_AUTO_CONCRETE
	DW_ABRV_AUTO_CONCRETE_LOCLIST
	DW_ABRV_PARAM
	DW_ABRV_PARAM_LOCLIST
	DW_ABRV_PARAM_ABSTRACT
	DW_ABRV_PARAM_CONCRETE
	DW_ABRV_PARAM_CONCRETE_LOCLIST
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

var abbrevsFinalized bool

// expandPseudoForm takes an input DW_FORM_xxx value and translates it
// into a platform-appropriate concrete form. Existing concrete/real
// DW_FORM values are left untouched. For the moment the only
// pseudo-form is DW_FORM_udata_pseudo, which gets expanded to
// DW_FORM_data4 on Darwin and DW_FORM_udata everywhere else. See
// issue #31459 for more context.
func expandPseudoForm(form uint8) uint8 {
	// Is this a pseudo-form?
	if form != DW_FORM_udata_pseudo {
		return form
	}
	expandedForm := DW_FORM_udata
	if objabi.GOOS == "darwin" || objabi.GOOS == "ios" {
		expandedForm = DW_FORM_data4
	}
	return uint8(expandedForm)
}

// Abbrevs() returns the finalized abbrev array for the platform,
// expanding any DW_FORM pseudo-ops to real values.
func Abbrevs() []dwAbbrev {
	if abbrevsFinalized {
		return abbrevs[:]
	}
	for i := 1; i < DW_NABRV; i++ {
		for j := 0; j < len(abbrevs[i].attr); j++ {
			abbrevs[i].attr[j].form = expandPseudoForm(abbrevs[i].attr[j].form)
		}
	}
	abbrevsFinalized = true
	return abbrevs[:]
}

// abbrevs is a raw table of abbrev entries; it needs to be post-processed
// by the Abbrevs() function above prior to being consumed, to expand
// the 'pseudo-form' entries below to real DWARF form values.

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
			{DW_AT_stmt_list, DW_FORM_sec_offset},
			{DW_AT_low_pc, DW_FORM_addr},
			{DW_AT_ranges, DW_FORM_sec_offset},
			{DW_AT_comp_dir, DW_FORM_string},
			{DW_AT_producer, DW_FORM_string},
			{DW_AT_go_package_name, DW_FORM_string},
		},
	},

	/* COMPUNIT_TEXTLESS */
	{
		DW_TAG_compile_unit,
		DW_CHILDREN_yes,
		[]dwAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_language, DW_FORM_data1},
			{DW_AT_comp_dir, DW_FORM_string},
			{DW_AT_producer, DW_FORM_string},
			{DW_AT_go_package_name, DW_FORM_string},
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
			{DW_AT_decl_file, DW_FORM_data4},
			{DW_AT_external, DW_FORM_flag},
		},
	},

	/* FUNCTION_ABSTRACT */
	{
		DW_TAG_subprogram,
		DW_CHILDREN_yes,
		[]dwAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_inline, DW_FORM_data1},
			{DW_AT_external, DW_FORM_flag},
		},
	},

	/* FUNCTION_CONCRETE */
	{
		DW_TAG_subprogram,
		DW_CHILDREN_yes,
		[]dwAttrForm{
			{DW_AT_abstract_origin, DW_FORM_ref_addr},
			{DW_AT_low_pc, DW_FORM_addr},
			{DW_AT_high_pc, DW_FORM_addr},
			{DW_AT_frame_base, DW_FORM_block1},
		},
	},

	/* INLINED_SUBROUTINE */
	{
		DW_TAG_inlined_subroutine,
		DW_CHILDREN_yes,
		[]dwAttrForm{
			{DW_AT_abstract_origin, DW_FORM_ref_addr},
			{DW_AT_low_pc, DW_FORM_addr},
			{DW_AT_high_pc, DW_FORM_addr},
			{DW_AT_call_file, DW_FORM_data4},
			{DW_AT_call_line, DW_FORM_udata_pseudo}, // pseudo-form
		},
	},

	/* INLINED_SUBROUTINE_RANGES */
	{
		DW_TAG_inlined_subroutine,
		DW_CHILDREN_yes,
		[]dwAttrForm{
			{DW_AT_abstract_origin, DW_FORM_ref_addr},
			{DW_AT_ranges, DW_FORM_sec_offset},
			{DW_AT_call_file, DW_FORM_data4},
			{DW_AT_call_line, DW_FORM_udata_pseudo}, // pseudo-form
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

	/* INT CONSTANT */
	{
		DW_TAG_constant,
		DW_CHILDREN_no,
		[]dwAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_type, DW_FORM_ref_addr},
			{DW_AT_const_value, DW_FORM_sdata},
		},
	},

	/* AUTO */
	{
		DW_TAG_variable,
		DW_CHILDREN_no,
		[]dwAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_decl_line, DW_FORM_udata},
			{DW_AT_type, DW_FORM_ref_addr},
			{DW_AT_location, DW_FORM_block1},
		},
	},

	/* AUTO_LOCLIST */
	{
		DW_TAG_variable,
		DW_CHILDREN_no,
		[]dwAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_decl_line, DW_FORM_udata},
			{DW_AT_type, DW_FORM_ref_addr},
			{DW_AT_location, DW_FORM_sec_offset},
		},
	},

	/* AUTO_ABSTRACT */
	{
		DW_TAG_variable,
		DW_CHILDREN_no,
		[]dwAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_decl_line, DW_FORM_udata},
			{DW_AT_type, DW_FORM_ref_addr},
		},
	},

	/* AUTO_CONCRETE */
	{
		DW_TAG_variable,
		DW_CHILDREN_no,
		[]dwAttrForm{
			{DW_AT_abstract_origin, DW_FORM_ref_addr},
			{DW_AT_location, DW_FORM_block1},
		},
	},

	/* AUTO_CONCRETE_LOCLIST */
	{
		DW_TAG_variable,
		DW_CHILDREN_no,
		[]dwAttrForm{
			{DW_AT_abstract_origin, DW_FORM_ref_addr},
			{DW_AT_location, DW_FORM_sec_offset},
		},
	},

	/* PARAM */
	{
		DW_TAG_formal_parameter,
		DW_CHILDREN_no,
		[]dwAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_variable_parameter, DW_FORM_flag},
			{DW_AT_decl_line, DW_FORM_udata},
			{DW_AT_type, DW_FORM_ref_addr},
			{DW_AT_location, DW_FORM_block1},
		},
	},

	/* PARAM_LOCLIST */
	{
		DW_TAG_formal_parameter,
		DW_CHILDREN_no,
		[]dwAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_variable_parameter, DW_FORM_flag},
			{DW_AT_decl_line, DW_FORM_udata},
			{DW_AT_type, DW_FORM_ref_addr},
			{DW_AT_location, DW_FORM_sec_offset},
		},
	},

	/* PARAM_ABSTRACT */
	{
		DW_TAG_formal_parameter,
		DW_CHILDREN_no,
		[]dwAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_variable_parameter, DW_FORM_flag},
			{DW_AT_type, DW_FORM_ref_addr},
		},
	},

	/* PARAM_CONCRETE */
	{
		DW_TAG_formal_parameter,
		DW_CHILDREN_no,
		[]dwAttrForm{
			{DW_AT_abstract_origin, DW_FORM_ref_addr},
			{DW_AT_location, DW_FORM_block1},
		},
	},

	/* PARAM_CONCRETE_LOCLIST */
	{
		DW_TAG_formal_parameter,
		DW_CHILDREN_no,
		[]dwAttrForm{
			{DW_AT_abstract_origin, DW_FORM_ref_addr},
			{DW_AT_location, DW_FORM_sec_offset},
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
			{DW_AT_data_member_location, DW_FORM_udata},
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
			{DW_AT_go_runtime_type, DW_FORM_addr},
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
			{DW_AT_go_runtime_type, DW_FORM_addr},
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
			{DW_AT_go_runtime_type, DW_FORM_addr},
			{DW_AT_go_elem, DW_FORM_ref_addr},
		},
	},

	/* FUNCTYPE */
	{
		DW_TAG_subroutine_type,
		DW_CHILDREN_yes,
		[]dwAttrForm{
			{DW_AT_name, DW_FORM_string},
			{DW_AT_byte_size, DW_FORM_udata},
			{DW_AT_go_kind, DW_FORM_data1},
			{DW_AT_go_runtime_type, DW_FORM_addr},
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
			{DW_AT_go_runtime_type, DW_FORM_addr},
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
			{DW_AT_go_runtime_type, DW_FORM_addr},
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
			{DW_AT_go_runtime_type, DW_FORM_addr},
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
			{DW_AT_go_runtime_type, DW_FORM_addr},
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
			{DW_AT_go_runtime_type, DW_FORM_addr},
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
			{DW_AT_go_runtime_type, DW_FORM_addr},
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
	abbrevs := Abbrevs()
	var buf []byte
	for i := 1; i < DW_NABRV; i++ {
		// See section 7.5.3
		buf = AppendUleb128(buf, uint64(i))
		buf = AppendUleb128(buf, uint64(abbrevs[i].tag))
		buf = append(buf, abbrevs[i].children)
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
		// Allow nil addresses for DW_AT_go_runtime_type.
		if data == nil && value == 0 {
			ctxt.AddInt(s, ctxt.PtrSize(), 0)
			break
		}
		if cls == DW_CLS_GO_TYPEREF {
			ctxt.AddSectionOffset(s, ctxt.PtrSize(), data, value)
			break
		}
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
			ctxt.AddDWARFAddrSectionOffset(s, data, value)
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
		ctxt.AddDWARFAddrSectionOffset(s, data, value)

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
	abbrevs := Abbrevs()
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

// HasChildren reports whether 'die' uses an abbrev that supports children.
func HasChildren(die *DWDie) bool {
	abbrevs := Abbrevs()
	return abbrevs[die.Abbrev].children != 0
}

// PutIntConst writes a DIE for an integer constant
func PutIntConst(ctxt Context, info, typ Sym, name string, val int64) {
	Uleb128put(ctxt, info, DW_ABRV_INT_CONSTANT)
	putattr(ctxt, info, DW_ABRV_INT_CONSTANT, DW_FORM_string, DW_CLS_STRING, int64(len(name)), name)
	putattr(ctxt, info, DW_ABRV_INT_CONSTANT, DW_FORM_ref_addr, DW_CLS_REFERENCE, 0, typ)
	putattr(ctxt, info, DW_ABRV_INT_CONSTANT, DW_FORM_sdata, DW_CLS_CONSTANT, val, nil)
}

// PutBasedRanges writes a range table to sym. All addresses in ranges are
// relative to some base address, which must be arranged by the caller
// (e.g., with a DW_AT_low_pc attribute, or in a BASE-prefixed range).
func PutBasedRanges(ctxt Context, sym Sym, ranges []Range) {
	ps := ctxt.PtrSize()
	// Write ranges.
	for _, r := range ranges {
		ctxt.AddInt(sym, ps, r.Start)
		ctxt.AddInt(sym, ps, r.End)
	}
	// Write trailer.
	ctxt.AddInt(sym, ps, 0)
	ctxt.AddInt(sym, ps, 0)
}

// PutRanges writes a range table to s.Ranges.
// All addresses in ranges are relative to s.base.
func (s *FnState) PutRanges(ctxt Context, ranges []Range) {
	ps := ctxt.PtrSize()
	sym, base := s.Ranges, s.StartPC

	if s.UseBASEntries {
		// Using a Base Address Selection Entry reduces the number of relocations, but
		// this is not done on macOS because it is not supported by dsymutil/dwarfdump/lldb
		ctxt.AddInt(sym, ps, -1)
		ctxt.AddAddress(sym, base, 0)
		PutBasedRanges(ctxt, sym, ranges)
		return
	}

	// Write ranges full of relocations
	for _, r := range ranges {
		ctxt.AddCURelativeAddress(sym, base, r.Start)
		ctxt.AddCURelativeAddress(sym, base, r.End)
	}
	// Write trailer.
	ctxt.AddInt(sym, ps, 0)
	ctxt.AddInt(sym, ps, 0)
}

// Return TRUE if the inlined call in the specified slot is empty,
// meaning it has a zero-length range (no instructions), and all
// of its children are empty.
func isEmptyInlinedCall(slot int, calls *InlCalls) bool {
	ic := &calls.Calls[slot]
	if ic.InlIndex == -2 {
		return true
	}
	live := false
	for _, k := range ic.Children {
		if !isEmptyInlinedCall(k, calls) {
			live = true
		}
	}
	if len(ic.Ranges) > 0 {
		live = true
	}
	if !live {
		ic.InlIndex = -2
	}
	return !live
}

// Slot -1:    return top-level inlines
// Slot >= 0:  return children of that slot
func inlChildren(slot int, calls *InlCalls) []int {
	var kids []int
	if slot != -1 {
		for _, k := range calls.Calls[slot].Children {
			if !isEmptyInlinedCall(k, calls) {
				kids = append(kids, k)
			}
		}
	} else {
		for k := 0; k < len(calls.Calls); k += 1 {
			if calls.Calls[k].Root && !isEmptyInlinedCall(k, calls) {
				kids = append(kids, k)
			}
		}
	}
	return kids
}

func inlinedVarTable(inlcalls *InlCalls) map[*Var]bool {
	vars := make(map[*Var]bool)
	for _, ic := range inlcalls.Calls {
		for _, v := range ic.InlVars {
			vars[v] = true
		}
	}
	return vars
}

// The s.Scopes slice contains variables were originally part of the
// function being emitted, as well as variables that were imported
// from various callee functions during the inlining process. This
// function prunes out any variables from the latter category (since
// they will be emitted as part of DWARF inlined_subroutine DIEs) and
// then generates scopes for vars in the former category.
func putPrunedScopes(ctxt Context, s *FnState, fnabbrev int) error {
	if len(s.Scopes) == 0 {
		return nil
	}
	scopes := make([]Scope, len(s.Scopes), len(s.Scopes))
	pvars := inlinedVarTable(&s.InlCalls)
	for k, s := range s.Scopes {
		var pruned Scope = Scope{Parent: s.Parent, Ranges: s.Ranges}
		for i := 0; i < len(s.Vars); i++ {
			_, found := pvars[s.Vars[i]]
			if !found {
				pruned.Vars = append(pruned.Vars, s.Vars[i])
			}
		}
		sort.Sort(byChildIndex(pruned.Vars))
		scopes[k] = pruned
	}
	var encbuf [20]byte
	if putscope(ctxt, s, scopes, 0, fnabbrev, encbuf[:0]) < int32(len(scopes)) {
		return errors.New("multiple toplevel scopes")
	}
	return nil
}

// Emit DWARF attributes and child DIEs for an 'abstract' subprogram.
// The abstract subprogram DIE for a function contains its
// location-independent attributes (name, type, etc). Other instances
// of the function (any inlined copy of it, or the single out-of-line
// 'concrete' instance) will contain a pointer back to this abstract
// DIE (as a space-saving measure, so that name/type etc doesn't have
// to be repeated for each inlined copy).
func PutAbstractFunc(ctxt Context, s *FnState) error {

	if logDwarf {
		ctxt.Logf("PutAbstractFunc(%v)\n", s.Absfn)
	}

	abbrev := DW_ABRV_FUNCTION_ABSTRACT
	Uleb128put(ctxt, s.Absfn, int64(abbrev))

	fullname := s.Name
	if strings.HasPrefix(s.Name, "\"\".") {
		// Generate a fully qualified name for the function in the
		// abstract case. This is so as to avoid the need for the
		// linker to process the DIE with patchDWARFName(); we can't
		// allow the name attribute of an abstract subprogram DIE to
		// be rewritten, since it would change the offsets of the
		// child DIEs (which we're relying on in order for abstract
		// origin references to work).
		fullname = objabi.PathToPrefix(s.Importpath) + "." + s.Name[3:]
	}
	putattr(ctxt, s.Absfn, abbrev, DW_FORM_string, DW_CLS_STRING, int64(len(fullname)), fullname)

	// DW_AT_inlined value
	putattr(ctxt, s.Absfn, abbrev, DW_FORM_data1, DW_CLS_CONSTANT, int64(DW_INL_inlined), nil)

	var ev int64
	if s.External {
		ev = 1
	}
	putattr(ctxt, s.Absfn, abbrev, DW_FORM_flag, DW_CLS_FLAG, ev, 0)

	// Child variables (may be empty)
	var flattened []*Var

	// This slice will hold the offset in bytes for each child var DIE
	// with respect to the start of the parent subprogram DIE.
	var offsets []int32

	// Scopes/vars
	if len(s.Scopes) > 0 {
		// For abstract subprogram DIEs we want to flatten out scope info:
		// lexical scope DIEs contain range and/or hi/lo PC attributes,
		// which we explicitly don't want for the abstract subprogram DIE.
		pvars := inlinedVarTable(&s.InlCalls)
		for _, scope := range s.Scopes {
			for i := 0; i < len(scope.Vars); i++ {
				_, found := pvars[scope.Vars[i]]
				if found || !scope.Vars[i].IsInAbstract {
					continue
				}
				flattened = append(flattened, scope.Vars[i])
			}
		}
		if len(flattened) > 0 {
			sort.Sort(byChildIndex(flattened))

			if logDwarf {
				ctxt.Logf("putAbstractScope(%v): vars:", s.Info)
				for i, v := range flattened {
					ctxt.Logf(" %d:%s", i, v.Name)
				}
				ctxt.Logf("\n")
			}

			// This slice will hold the offset in bytes for each child
			// variable DIE with respect to the start of the parent
			// subprogram DIE.
			for _, v := range flattened {
				offsets = append(offsets, int32(ctxt.CurrentOffset(s.Absfn)))
				putAbstractVar(ctxt, s.Absfn, v)
			}
		}
	}
	ctxt.RecordChildDieOffsets(s.Absfn, flattened, offsets)

	Uleb128put(ctxt, s.Absfn, 0)
	return nil
}

// Emit DWARF attributes and child DIEs for an inlined subroutine. The
// first attribute of an inlined subroutine DIE is a reference back to
// its corresponding 'abstract' DIE (containing location-independent
// attributes such as name, type, etc). Inlined subroutine DIEs can
// have other inlined subroutine DIEs as children.
func PutInlinedFunc(ctxt Context, s *FnState, callersym Sym, callIdx int) error {
	ic := s.InlCalls.Calls[callIdx]
	callee := ic.AbsFunSym

	abbrev := DW_ABRV_INLINED_SUBROUTINE_RANGES
	if len(ic.Ranges) == 1 {
		abbrev = DW_ABRV_INLINED_SUBROUTINE
	}
	Uleb128put(ctxt, s.Info, int64(abbrev))

	if logDwarf {
		ctxt.Logf("PutInlinedFunc(caller=%v,callee=%v,abbrev=%d)\n", callersym, callee, abbrev)
	}

	// Abstract origin.
	putattr(ctxt, s.Info, abbrev, DW_FORM_ref_addr, DW_CLS_REFERENCE, 0, callee)

	if abbrev == DW_ABRV_INLINED_SUBROUTINE_RANGES {
		putattr(ctxt, s.Info, abbrev, DW_FORM_sec_offset, DW_CLS_PTR, s.Ranges.Length(ctxt), s.Ranges)
		s.PutRanges(ctxt, ic.Ranges)
	} else {
		st := ic.Ranges[0].Start
		en := ic.Ranges[0].End
		putattr(ctxt, s.Info, abbrev, DW_FORM_addr, DW_CLS_ADDRESS, st, s.StartPC)
		putattr(ctxt, s.Info, abbrev, DW_FORM_addr, DW_CLS_ADDRESS, en, s.StartPC)
	}

	// Emit call file, line attrs.
	ctxt.AddFileRef(s.Info, ic.CallFile)
	form := int(expandPseudoForm(DW_FORM_udata_pseudo))
	putattr(ctxt, s.Info, abbrev, form, DW_CLS_CONSTANT, int64(ic.CallLine), nil)

	// Variables associated with this inlined routine instance.
	vars := ic.InlVars
	sort.Sort(byChildIndex(vars))
	inlIndex := ic.InlIndex
	var encbuf [20]byte
	for _, v := range vars {
		if !v.IsInAbstract {
			continue
		}
		putvar(ctxt, s, v, callee, abbrev, inlIndex, encbuf[:0])
	}

	// Children of this inline.
	for _, sib := range inlChildren(callIdx, &s.InlCalls) {
		absfn := s.InlCalls.Calls[sib].AbsFunSym
		err := PutInlinedFunc(ctxt, s, absfn, sib)
		if err != nil {
			return err
		}
	}

	Uleb128put(ctxt, s.Info, 0)
	return nil
}

// Emit DWARF attributes and child DIEs for a 'concrete' subprogram,
// meaning the out-of-line copy of a function that was inlined at some
// point during the compilation of its containing package. The first
// attribute for a concrete DIE is a reference to the 'abstract' DIE
// for the function (which holds location-independent attributes such
// as name, type), then the remainder of the attributes are specific
// to this instance (location, frame base, etc).
func PutConcreteFunc(ctxt Context, s *FnState) error {
	if logDwarf {
		ctxt.Logf("PutConcreteFunc(%v)\n", s.Info)
	}
	abbrev := DW_ABRV_FUNCTION_CONCRETE
	Uleb128put(ctxt, s.Info, int64(abbrev))

	// Abstract origin.
	putattr(ctxt, s.Info, abbrev, DW_FORM_ref_addr, DW_CLS_REFERENCE, 0, s.Absfn)

	// Start/end PC.
	putattr(ctxt, s.Info, abbrev, DW_FORM_addr, DW_CLS_ADDRESS, 0, s.StartPC)
	putattr(ctxt, s.Info, abbrev, DW_FORM_addr, DW_CLS_ADDRESS, s.Size, s.StartPC)

	// cfa / frame base
	putattr(ctxt, s.Info, abbrev, DW_FORM_block1, DW_CLS_BLOCK, 1, []byte{DW_OP_call_frame_cfa})

	// Scopes
	if err := putPrunedScopes(ctxt, s, abbrev); err != nil {
		return err
	}

	// Inlined subroutines.
	for _, sib := range inlChildren(-1, &s.InlCalls) {
		absfn := s.InlCalls.Calls[sib].AbsFunSym
		err := PutInlinedFunc(ctxt, s, absfn, sib)
		if err != nil {
			return err
		}
	}

	Uleb128put(ctxt, s.Info, 0)
	return nil
}

// Emit DWARF attributes and child DIEs for a subprogram. Here
// 'default' implies that the function in question was not inlined
// when its containing package was compiled (hence there is no need to
// emit an abstract version for it to use as a base for inlined
// routine records).
func PutDefaultFunc(ctxt Context, s *FnState) error {
	if logDwarf {
		ctxt.Logf("PutDefaultFunc(%v)\n", s.Info)
	}
	abbrev := DW_ABRV_FUNCTION
	Uleb128put(ctxt, s.Info, int64(abbrev))

	// Expand '"".' to import path.
	name := s.Name
	if s.Importpath != "" {
		name = strings.Replace(name, "\"\".", objabi.PathToPrefix(s.Importpath)+".", -1)
	}

	putattr(ctxt, s.Info, DW_ABRV_FUNCTION, DW_FORM_string, DW_CLS_STRING, int64(len(name)), name)
	putattr(ctxt, s.Info, abbrev, DW_FORM_addr, DW_CLS_ADDRESS, 0, s.StartPC)
	putattr(ctxt, s.Info, abbrev, DW_FORM_addr, DW_CLS_ADDRESS, s.Size, s.StartPC)
	putattr(ctxt, s.Info, abbrev, DW_FORM_block1, DW_CLS_BLOCK, 1, []byte{DW_OP_call_frame_cfa})
	ctxt.AddFileRef(s.Info, s.Filesym)

	var ev int64
	if s.External {
		ev = 1
	}
	putattr(ctxt, s.Info, abbrev, DW_FORM_flag, DW_CLS_FLAG, ev, 0)

	// Scopes
	if err := putPrunedScopes(ctxt, s, abbrev); err != nil {
		return err
	}

	// Inlined subroutines.
	for _, sib := range inlChildren(-1, &s.InlCalls) {
		absfn := s.InlCalls.Calls[sib].AbsFunSym
		err := PutInlinedFunc(ctxt, s, absfn, sib)
		if err != nil {
			return err
		}
	}

	Uleb128put(ctxt, s.Info, 0)
	return nil
}

func putscope(ctxt Context, s *FnState, scopes []Scope, curscope int32, fnabbrev int, encbuf []byte) int32 {

	if logDwarf {
		ctxt.Logf("putscope(%v,%d): vars:", s.Info, curscope)
		for i, v := range scopes[curscope].Vars {
			ctxt.Logf(" %d:%d:%s", i, v.ChildIndex, v.Name)
		}
		ctxt.Logf("\n")
	}

	for _, v := range scopes[curscope].Vars {
		putvar(ctxt, s, v, s.Absfn, fnabbrev, -1, encbuf)
	}
	this := curscope
	curscope++
	for curscope < int32(len(scopes)) {
		scope := scopes[curscope]
		if scope.Parent != this {
			return curscope
		}

		if len(scopes[curscope].Vars) == 0 {
			curscope = putscope(ctxt, s, scopes, curscope, fnabbrev, encbuf)
			continue
		}

		if len(scope.Ranges) == 1 {
			Uleb128put(ctxt, s.Info, DW_ABRV_LEXICAL_BLOCK_SIMPLE)
			putattr(ctxt, s.Info, DW_ABRV_LEXICAL_BLOCK_SIMPLE, DW_FORM_addr, DW_CLS_ADDRESS, scope.Ranges[0].Start, s.StartPC)
			putattr(ctxt, s.Info, DW_ABRV_LEXICAL_BLOCK_SIMPLE, DW_FORM_addr, DW_CLS_ADDRESS, scope.Ranges[0].End, s.StartPC)
		} else {
			Uleb128put(ctxt, s.Info, DW_ABRV_LEXICAL_BLOCK_RANGES)
			putattr(ctxt, s.Info, DW_ABRV_LEXICAL_BLOCK_RANGES, DW_FORM_sec_offset, DW_CLS_PTR, s.Ranges.Length(ctxt), s.Ranges)

			s.PutRanges(ctxt, scope.Ranges)
		}

		curscope = putscope(ctxt, s, scopes, curscope, fnabbrev, encbuf)

		Uleb128put(ctxt, s.Info, 0)
	}
	return curscope
}

// Given a default var abbrev code, select corresponding concrete code.
func concreteVarAbbrev(varAbbrev int) int {
	switch varAbbrev {
	case DW_ABRV_AUTO:
		return DW_ABRV_AUTO_CONCRETE
	case DW_ABRV_PARAM:
		return DW_ABRV_PARAM_CONCRETE
	case DW_ABRV_AUTO_LOCLIST:
		return DW_ABRV_AUTO_CONCRETE_LOCLIST
	case DW_ABRV_PARAM_LOCLIST:
		return DW_ABRV_PARAM_CONCRETE_LOCLIST
	default:
		panic("should never happen")
	}
}

// Pick the correct abbrev code for variable or parameter DIE.
func determineVarAbbrev(v *Var, fnabbrev int) (int, bool, bool) {
	abbrev := v.Abbrev

	// If the variable was entirely optimized out, don't emit a location list;
	// convert to an inline abbreviation and emit an empty location.
	missing := false
	switch {
	case abbrev == DW_ABRV_AUTO_LOCLIST && v.PutLocationList == nil:
		missing = true
		abbrev = DW_ABRV_AUTO
	case abbrev == DW_ABRV_PARAM_LOCLIST && v.PutLocationList == nil:
		missing = true
		abbrev = DW_ABRV_PARAM
	}

	// Determine whether to use a concrete variable or regular variable DIE.
	concrete := true
	switch fnabbrev {
	case DW_ABRV_FUNCTION:
		concrete = false
		break
	case DW_ABRV_FUNCTION_CONCRETE:
		// If we're emitting a concrete subprogram DIE and the variable
		// in question is not part of the corresponding abstract function DIE,
		// then use the default (non-concrete) abbrev for this param.
		if !v.IsInAbstract {
			concrete = false
		}
	case DW_ABRV_INLINED_SUBROUTINE, DW_ABRV_INLINED_SUBROUTINE_RANGES:
	default:
		panic("should never happen")
	}

	// Select proper abbrev based on concrete/non-concrete
	if concrete {
		abbrev = concreteVarAbbrev(abbrev)
	}

	return abbrev, missing, concrete
}

func abbrevUsesLoclist(abbrev int) bool {
	switch abbrev {
	case DW_ABRV_AUTO_LOCLIST, DW_ABRV_AUTO_CONCRETE_LOCLIST,
		DW_ABRV_PARAM_LOCLIST, DW_ABRV_PARAM_CONCRETE_LOCLIST:
		return true
	default:
		return false
	}
}

// Emit DWARF attributes for a variable belonging to an 'abstract' subprogram.
func putAbstractVar(ctxt Context, info Sym, v *Var) {
	// Remap abbrev
	abbrev := v.Abbrev
	switch abbrev {
	case DW_ABRV_AUTO, DW_ABRV_AUTO_LOCLIST:
		abbrev = DW_ABRV_AUTO_ABSTRACT
	case DW_ABRV_PARAM, DW_ABRV_PARAM_LOCLIST:
		abbrev = DW_ABRV_PARAM_ABSTRACT
	}

	Uleb128put(ctxt, info, int64(abbrev))
	putattr(ctxt, info, abbrev, DW_FORM_string, DW_CLS_STRING, int64(len(v.Name)), v.Name)

	// Isreturn attribute if this is a param
	if abbrev == DW_ABRV_PARAM_ABSTRACT {
		var isReturn int64
		if v.IsReturnValue {
			isReturn = 1
		}
		putattr(ctxt, info, abbrev, DW_FORM_flag, DW_CLS_FLAG, isReturn, nil)
	}

	// Line
	if abbrev != DW_ABRV_PARAM_ABSTRACT {
		// See issue 23374 for more on why decl line is skipped for abs params.
		putattr(ctxt, info, abbrev, DW_FORM_udata, DW_CLS_CONSTANT, int64(v.DeclLine), nil)
	}

	// Type
	putattr(ctxt, info, abbrev, DW_FORM_ref_addr, DW_CLS_REFERENCE, 0, v.Type)

	// Var has no children => no terminator
}

func putvar(ctxt Context, s *FnState, v *Var, absfn Sym, fnabbrev, inlIndex int, encbuf []byte) {
	// Remap abbrev according to parent DIE abbrev
	abbrev, missing, concrete := determineVarAbbrev(v, fnabbrev)

	Uleb128put(ctxt, s.Info, int64(abbrev))

	// Abstract origin for concrete / inlined case
	if concrete {
		// Here we are making a reference to a child DIE of an abstract
		// function subprogram DIE. The child DIE has no LSym, so instead
		// after the call to 'putattr' below we make a call to register
		// the child DIE reference.
		putattr(ctxt, s.Info, abbrev, DW_FORM_ref_addr, DW_CLS_REFERENCE, 0, absfn)
		ctxt.RecordDclReference(s.Info, absfn, int(v.ChildIndex), inlIndex)
	} else {
		// Var name, line for abstract and default cases
		n := v.Name
		putattr(ctxt, s.Info, abbrev, DW_FORM_string, DW_CLS_STRING, int64(len(n)), n)
		if abbrev == DW_ABRV_PARAM || abbrev == DW_ABRV_PARAM_LOCLIST || abbrev == DW_ABRV_PARAM_ABSTRACT {
			var isReturn int64
			if v.IsReturnValue {
				isReturn = 1
			}
			putattr(ctxt, s.Info, abbrev, DW_FORM_flag, DW_CLS_FLAG, isReturn, nil)
		}
		putattr(ctxt, s.Info, abbrev, DW_FORM_udata, DW_CLS_CONSTANT, int64(v.DeclLine), nil)
		putattr(ctxt, s.Info, abbrev, DW_FORM_ref_addr, DW_CLS_REFERENCE, 0, v.Type)
	}

	if abbrevUsesLoclist(abbrev) {
		putattr(ctxt, s.Info, abbrev, DW_FORM_sec_offset, DW_CLS_PTR, s.Loc.Length(ctxt), s.Loc)
		v.PutLocationList(s.Loc, s.StartPC)
	} else {
		loc := encbuf[:0]
		switch {
		case missing:
			break // no location
		case v.StackOffset == 0:
			loc = append(loc, DW_OP_call_frame_cfa)
		default:
			loc = append(loc, DW_OP_fbreg)
			loc = AppendSleb128(loc, int64(v.StackOffset))
		}
		putattr(ctxt, s.Info, abbrev, DW_FORM_block1, DW_CLS_BLOCK, int64(len(loc)), loc)
	}

	// Var has no children => no terminator
}

// VarsByOffset attaches the methods of sort.Interface to []*Var,
// sorting in increasing StackOffset.
type VarsByOffset []*Var

func (s VarsByOffset) Len() int           { return len(s) }
func (s VarsByOffset) Less(i, j int) bool { return s[i].StackOffset < s[j].StackOffset }
func (s VarsByOffset) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// byChildIndex implements sort.Interface for []*dwarf.Var by child index.
type byChildIndex []*Var

func (s byChildIndex) Len() int           { return len(s) }
func (s byChildIndex) Less(i, j int) bool { return s[i].ChildIndex < s[j].ChildIndex }
func (s byChildIndex) Swap(i, j int)      { s[i], s[j] = s[j], s[i] }

// IsDWARFEnabledOnAIX returns true if DWARF is possible on the
// current extld.
// AIX ld doesn't support DWARF with -bnoobjreorder with version
// prior to 7.2.2.
func IsDWARFEnabledOnAIXLd(extld string) (bool, error) {
	out, err := exec.Command(extld, "-Wl,-V").CombinedOutput()
	if err != nil {
		// The normal output should display ld version and
		// then fails because ".main" is not defined:
		// ld: 0711-317 ERROR: Undefined symbol: .main
		if !bytes.Contains(out, []byte("0711-317")) {
			return false, fmt.Errorf("%s -Wl,-V failed: %v\n%s", extld, err, out)
		}
	}
	// gcc -Wl,-V output should be:
	//   /usr/bin/ld: LD X.X.X(date)
	//   ...
	out = bytes.TrimPrefix(out, []byte("/usr/bin/ld: LD "))
	vers := string(bytes.Split(out, []byte("("))[0])
	subvers := strings.Split(vers, ".")
	if len(subvers) != 3 {
		return false, fmt.Errorf("cannot parse %s -Wl,-V (%s): %v\n", extld, out, err)
	}
	if v, err := strconv.Atoi(subvers[0]); err != nil || v < 7 {
		return false, nil
	} else if v > 7 {
		return true, nil
	}
	if v, err := strconv.Atoi(subvers[1]); err != nil || v < 2 {
		return false, nil
	} else if v > 2 {
		return true, nil
	}
	if v, err := strconv.Atoi(subvers[2]); err != nil || v < 2 {
		return false, nil
	}
	return true, nil
}
