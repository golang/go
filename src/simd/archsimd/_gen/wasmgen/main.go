// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"bytes"
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"text/template"
	"unicode"

	"simd/archsimd/_gen/sgutil"
)

var (
	gorootsrc = flag.String("gorootsrc", "../../../../../src", "root of destination directory, normally GOROOT/src")

	genTypesFile      = flag.String("types", "GOROOTSRC/simd/archsimd/types_wasm.go", "output file for simd types (e.g. types_wasm.go)")
	genOpsFile        = flag.String("ops", "GOROOTSRC/simd/archsimd/ops_wasm.go", "output file for simd ops (e.g. ops_wasm.go)")
	genSSAOpsFile     = flag.String("ssaops", "GOROOTSRC/cmd/compile/internal/ssa/_gen/simdWasmops.go", "output file for ssa ops (e.g. simdWasmops.go)")
	genGenOpsFile     = flag.String("genops", "GOROOTSRC/cmd/compile/internal/ssa/_gen/simdgenericOps.go", "output file for generic ssa ops (e.g. simdgenericOps.go)")
	genSSARulesFile   = flag.String("ssarules", "GOROOTSRC/cmd/compile/internal/ssa/_gen/simdWasm.rules", "output file for ssa rules (e.g. simdWasm.rules)")
	genWasmSSAFile    = flag.String("wasmssa", "GOROOTSRC/cmd/compile/internal/wasm/simdssa.go", "output file for wasm ssa (e.g. simdssa.go)")
	genIntrinsicsFile = flag.String("intrinsics", "GOROOTSRC/cmd/compile/internal/ssagen/simdWasmintrinsics.go", "output file for intrinsics (e.g. simdWasmintrinsics.go)")

	list = flag.Bool("list", false, "list all the opcodes")
)

type simdType struct {
	Name       string // e.g. "Int8x16"
	Elem       string // e.g. "int8"
	Count      int    // e.g. 2, 4, 8, 16
	ElemSize   int    // e.g. 8, 16, 32, 64
	Unsigned   bool
	Float      bool
	Methods    map[string]*wasmOp
	IntShaped  *simdType // refers to the same-shape signed integer type
	UintShaped *simdType // refers to the same-shape unsigned integer type
}

func (t *simdType) ElemBits() int {
	return t.ElemSize
}

func (t *simdType) HalfCount() int {
	return t.Count / 2
}

func (t *simdType) TwiceCount() int {
	return t.Count * 2
}

func (t *simdType) Name_() string {
	return t.Name
}

func (t *simdType) Article() string {
	// uint => "you-int" => "a you-int"
	if t.Elem[0] == 'i' {
		return "an"
	}
	return "a"
}

func (a *simdType) Compare(b *simdType) int {
	if a.Name == b.Name {
		return 0
	}
	if d := a.ElemSize - b.ElemSize; d != 0 {
		return d
	}
	if d := a.Count - b.Count; d != 0 {
		// never happens for WASM
		return d
	}

	ao := strings.Index("iIuUfFmM", a.Name[:1])
	bo := strings.Index("iIuUfFmM", b.Name[:1])
	if ao == -1 || bo == -1 {
		panic(fmt.Errorf("a.Elem=%s, b.Elem=%s, unexpected first characters (should be in \"iIuUfFmM\")", a.Elem, b.Elem))
	}
	return ao - bo
}

// WasmUName returns the Capitalized wasm type name e.g. I64x2
func (s *simdType) WasmUName() string {
	T := s.Name[:1]
	if T == "U" {
		T = "I"
	}
	return fmt.Sprintf("%s%dx%d", T, s.ElemSize, s.Count)
}

// WasmUName returns the uncapitalized wasm type name e.g. i64x2
func (s *simdType) WasmLName() string {
	T := s.Elem[:1]
	if T == "u" {
		T = "i"
	}
	return fmt.Sprintf("%s%dx%d", T, s.ElemSize, s.Count)
}

func CapitalizeFirst(s string) string {
	return strings.ToUpper(s[:1]) + s[1:]
}

func (s *simdType) MaskFor() *simdType {
	return maskFor[s]
}

// WidenElements doubles the width of the elements, and changes the type
func (s *simdType) WidenElements(newStem string) string {
	return fmt.Sprintf("%s%dx%d", newStem, s.ElemSize*2, s.Count/2)
}

// ShrinkElements halves the width of the elements, and changes their type
func (s *simdType) ShrinkElements(newStem string) string {
	return fmt.Sprintf("%s%dx%d", newStem, s.ElemSize/2, s.Count*2)
}

func (s *simdType) IntFor() *simdType {
	if s.IntShaped == nil {
		return s
	}
	return s.IntShaped
}

func (s *simdType) UintFor() *simdType {
	if s.UintShaped == nil {
		return s
	}
	return s.UintShaped
}

func (s *simdType) String() string {
	return s.Name
}

func (s *simdType) IsMask() bool {
	return s.Name[0] == 'M'
}

func (u *simdType) setUintShaped(s *simdType) *simdType {
	s.UintShaped = u
	return u
}

const pkg = "simd/archsimd"

var (
	vi8  = &simdType{"Int8x16", "int8", 16, 8, false, false, make(map[string]*wasmOp), nil, nil}
	vi16 = &simdType{"Int16x8", "int16", 8, 16, false, false, make(map[string]*wasmOp), nil, nil}
	vi32 = &simdType{"Int32x4", "int32", 4, 32, false, false, make(map[string]*wasmOp), nil, nil}
	vi64 = &simdType{"Int64x2", "int64", 2, 64, false, false, make(map[string]*wasmOp), nil, nil}

	vu8  = (&simdType{"Uint8x16", "uint8", 16, 8, true, false, make(map[string]*wasmOp), vi8, nil}).setUintShaped(vi8) // For sign-ignoring operations (add, sub), use these types
	vu16 = (&simdType{"Uint16x8", "uint16", 8, 16, true, false, make(map[string]*wasmOp), vi16, nil}).setUintShaped(vi16)
	vu32 = (&simdType{"Uint32x4", "uint32", 4, 32, true, false, make(map[string]*wasmOp), vi32, nil}).setUintShaped(vi32)
	vu64 = (&simdType{"Uint64x2", "uint64", 2, 64, true, false, make(map[string]*wasmOp), vi64, nil}).setUintShaped(vi64)

	vf32 = &simdType{"Float32x4", "float32", 4, 32, false, true, make(map[string]*wasmOp), vi32, vu32}
	vf64 = &simdType{"Float64x2", "float64", 2, 64, false, true, make(map[string]*wasmOp), vi64, vu64}

	vm8  = &simdType{"Mask8x16", "int8", 16, 8, false, false, make(map[string]*wasmOp), vi8, vu8} // for non-bitwise operations (eq, ne), use these types
	vm16 = &simdType{"Mask16x8", "int16", 8, 16, false, false, make(map[string]*wasmOp), vi16, vu16}
	vm32 = &simdType{"Mask32x4", "int32", 4, 32, false, false, make(map[string]*wasmOp), vi32, vu32}
	vm64 = &simdType{"Mask64x2", "int64", 2, 64, false, false, make(map[string]*wasmOp), vi64, vu64}
)

var maskFor map[*simdType]*simdType = map[*simdType]*simdType{
	vm8:  vm8,
	vi8:  vm8,
	vu8:  vm8,
	vm16: vm16,
	vi16: vm16,
	vu16: vm16,
	vm32: vm32,
	vi32: vm32,
	vu32: vm32,
	vf32: vm32,
	vm64: vm64,
	vi64: vm64,
	vu64: vm64,
	vf64: vm64,
}

type OpFlags uint16

const (
	IsConst = OpFlags(1) << iota
	IsLoad
	IsStore
	IsShift
	IsSplat
	IsBitwise
	IsRelation
	IsTest
	IsExtract // Returns element type
	IsCommutative
	IsConversion  // "Wasm type" is result type, "Go method type" is input time
	NameHasFormat //
	NonSigned     // Same bitwise operation whether signed or unsigned.  E.g. 2's complement addition.
	EmulatedRule  // Emulation occurs at rule expansion.
)

func (o OpFlags) OneString() string {
	switch o {
	case IsConst:
		return "IsConst"
	case IsLoad:
		return "IsLoad"
	case IsStore:
		return "IsStore"
	case IsShift:
		return "IsShift"
	case IsSplat:
		return "IsSplat"
	case IsBitwise:
		return "IsBitwise"
	case IsRelation:
		return "IsRelation"
	case IsTest:
		return "IsTest"
	case IsExtract:
		return "IsExtract"
	case IsCommutative:
		return "IsCommutative"
	case IsConversion:
		return "IsConversion"
	case NameHasFormat:
		return "NameHasFormat"
	case NonSigned:
		return "NonSigned"
	case EmulatedRule:
		return "RmulatedRule"
	}
	return fmt.Sprintf("0x%x", o)
}

var allFlags = []OpFlags{
	IsConst,
	IsLoad,
	IsStore,
	IsShift,
	IsSplat,
	IsBitwise,
	IsRelation,
	IsTest,
	IsExtract,
	IsCommutative,
	IsConversion,
	NameHasFormat,
	NonSigned,
	EmulatedRule,
}

func (o OpFlags) String() string {
	sep := ""
	var ret strings.Builder

	for _, x := range allFlags {
		if x&o != 0 {
			ret.WriteString(sep + x.OneString())
			sep = "+"
		}
	}
	return ret.String()
}

// wasmOp represents a WebAssembly SIMD instruction.
type wasmOp struct {
	t          *simdType // the receiver and default arg type
	op         string    // the basic Op type, e.g. "load", "add"
	argCount   int       // Number of arguments (inputs)
	argType    string    // (Binary) arg type (e.g., "v128", "i32", "void") -- defaults to t
	resultType string    // Result type (e.g., "v128", "i32", "void") -- defaults to t
	opFlags    OpFlags
	doc        string

	immRange uint8  // Max immediate value; for lane-oriented operations. 0 => no immediate.
	immName  string // The parameter name for an immediate operations
	arg1Name string // The 1st (non-immediate) arg name. Arg1Name() defaults to "y"
	arg2Name string // The 2nd (non-immediate) arg name. Arg2Name() defaults to "z"
}

func (o *wasmOp) String() string {
	return fmt.Sprintf(
		"t=%s, op=%s, arity=%d, Method=%s, ArgType=%s, ResultType=%s, GenOp=%s, SsaWasmOp=%s, AsmOp=%s, WasmInstruction=%s, ImmRange=%d, ImmName=%s, Flags=%s",
		o.Type().Name, o.Op(), o.ArgCount(), o.Method(), o.ArgType(), o.ResultType(), o.SsaGenOp(),
		o.SsaWasmOp(), o.AsmOp(), o.WasmInstruction(), o.ImmRange(), o.ImmName(), o.OpFlags())
}

func compareWasmOps(a, b *wasmOp) int {
	am, bm := a.NUMethod(), b.NUMethod()
	if cmp := strings.Compare(am, bm); cmp != 0 {
		return cmp
	}
	if cmp := a.t.Compare(b.t); cmp != 0 {
		return cmp
	}
	return strings.Compare(a.op, b.op)
}

func (o *wasmOp) Type() *simdType {
	return o.t
}

func (o *wasmOp) ArgCount() int {
	return o.argCount
}

func (o *wasmOp) OpFlags() OpFlags {
	return o.opFlags
}

func (o *wasmOp) Flag(f OpFlags) bool {
	return o.opFlags&f != 0
}

func (o *wasmOp) ImmRange() uint8 {
	return o.immRange
}

func (o *wasmOp) ImmName() string {
	if o.immName == "" {
		return "_"
	}
	return o.immName
}

func snakeToCamel(s string) string {
	capnext := true
	var result strings.Builder
	for _, c := range s {
		if c == '_' {
			capnext = true
			continue
		}
		if '0' <= c && c <= '9' {
			capnext = true
		} else {
			if capnext {
				c = unicode.ToUpper(c)
				capnext = false
			}
		}
		result.WriteString(string(c))
	}
	return result.String()
}

// Op returns the snakeToCamel version of the WASM operation,
// e.g. return_call_indirect
func (o *wasmOp) Op() string {
	return snakeToCamel(o.op)
}

func (o *wasmOp) T() *simdType {
	return o.t
}

// Sub, AddSaturated
// this appears in the Go API declaration files and in the intrinsic registration files.
func (o *wasmOp) Method() string {
	goname := gonames[o.op]
	if o.Flag(NameHasFormat) {
		// This is an extend method of the form Extend{Lo,Hi}%dTo%s
		count := o.T().Count / 2
		x := strings.Index(o.ResultType(), "x")
		typ := o.ResultType()[:x]
		return fmt.Sprintf(goname, count, typ)
	}
	if len(goname) >= 1 {
		if len(goname) == 1 {
			// it's something like "-" or "?" or "!"
			return ""
		}
		return goname
	}
	return snakeToCamel(o.op)
}

func (o *wasmOp) NUMethod() string {
	s := o.Method()
	if s != "" && s[0] == '_' {
		s = s[1:]
	}
	return s
}

func (o *wasmOp) SsaResultType() string {
	if o.Flag(IsTest) {
		return "Bool"
	}
	return "Vec128" // TODO this is not always right
}

func (o *wasmOp) RegInfo() string {
	if o.argType == "" {
		if o.resultType == "" {
			switch o.argCount {
			case 1:
				return "v11"
			case 2:
				return "v21"
			case 3:
				return "v31"
			}
		} else if o.Flag(IsConversion) {
			if o.argCount == 1 {
				return "v11"
			} else if o.argCount == 2 {
				// widening multiplies
				return "v21"
			}
		} else {
			if o.argCount == 1 {
				// extract lane
				if o.resultType[0] == 'i' || o.resultType[0] == 'u' {
					return "v11gp"
				}
				if o.resultType == "float32" {
					return "v11fp32"
				}
				if o.resultType == "float64" {
					return "v11fp64"
				}
			}
		}
	} else if o.argCount == 2 {
		// replace lane
		if o.argType[0] == 'i' || o.argType[0] == 'u' {
			return "v1gpv"
		} else if o.argType == "float32" {
			return "v1fp32v"
		} else if o.argType == "float64" {
			return "v1fp64v"
		}
	} else if o.argCount == 3 {
		// bitSelect
		return "v31"
	} else if o.Flag(IsSplat) {
		if o.argType[0] == 'i' {
			return "gpv"
		} else if o.argType == "float32" {
			return "fp32v"
		} else {
			return "fp64v"
		}
	}

	panic("RegInfo not implemented for " + o.String())
}

func (o *wasmOp) DefinesGeneric() bool {
	return o.Method() != "" && !o.T().IsMask() && !o.Flag(NonSigned)
}

// SubInt8x16
func (o *wasmOp) SsaGenOp() string {
	m := o.Method()
	if m == "" {
		return ""
	}
	// if o.Flag(IsBitwise) {
	// 	return m + "V128"
	// }
	if m[0] == '_' {
		// strip leading underscore from name, for generics.
		m = m[1:]
	}
	t := o.T()
	if t.IsMask() || o.Flag(NonSigned) {
		t = t.IntShaped
	}
	// Rotate instructions on amd 64 are single op + immediate.
	if strings.HasPrefix(o.op, "RotateAll") {
		m += "Var"
	}
	r := m + t.Name
	return r
}

// conversionCorrectedOp separates the s/u suffix from the operation
// for example input "foo_s" -> returns "foo", "_s", "S"
func (o *wasmOp) conversionCorrectedOp() (op, lowerSuffix, upperSuffix string) {
	op = o.op
	if strings.HasSuffix(op, "_s") {
		op = op[:len(op)-2]
		upperSuffix = "S"
		lowerSuffix = "_s"
	} else if strings.HasSuffix(op, "_u") {
		op = op[:len(op)-2]
		upperSuffix = "U"
		lowerSuffix = "_u"
	}
	return
}

// SsaWasmOp returns the name of the WASM-specific SSA Op
// examples: I8x16Sub, I64x2ExtendLowI32x4U
func (o *wasmOp) SsaWasmOp() string {
	// Wasm puts the main input type first
	// Except conversions start with their result type
	oop := o.Op()
	if oop == "BitSelect" {
		// Good names for methods versus consistent names for ASM, resolved here.
		// Snake_To_Camel => Bitselect but BitSelect is better and Go programmers are
		// more important than the naming choices of (virtual) hardware designers.
		oop = "Bitselect"
	} else if oop == "Shuffle" {
		oop += "16"
	}

	if o.Flag(IsBitwise) {
		return "V128" + oop
	}
	if o.Flag(IsConversion) {
		// I64x2ExtendLowI32x4U
		op, _, suffix := o.conversionCorrectedOp()
		op = snakeToCamel(op)
		return CapitalizeFirst(wasmResultType(o.ResultType())) + op + o.T().WasmUName() + suffix
	}
	r := o.T().WasmUName() + oop
	return r
}

func wasmResultType(s string) string {
	ti := strings.Index(s, "t") // Uint, Int, Float
	if s[ti-1] == 'n' {         // Uint, Int
		s = "i" + s[ti+1:]
	} else {
		s = "f" + s[ti+1:]
	}
	return s
}

// WasmInstruction returns the from-the-Wasm-spec assembler instruction
// examples: i8x16.sub, i64x2.extend_low.i32x4_u
func (o *wasmOp) WasmInstruction() string {
	oop := o.op
	if o.Flag(IsBitwise) {
		return "v128" + "." + oop
	}
	if o.Flag(IsConversion) {
		// i64x2.extend_low
		op, suffix, _ := o.conversionCorrectedOp()
		return wasmResultType(o.ResultType()) + "." + op + "_" + o.T().WasmLName() + suffix
	}
	r := o.T().WasmLName() + "." + oop
	return r
}

func (o *wasmOp) RcvrType() string {
	return o.T().Name
}

func (o *wasmOp) Arg2Name() string {
	if n := o.arg2Name; n != "" {
		return n
	}
	return "z"
}

func (o *wasmOp) Arg1Name() string {
	if n := o.arg1Name; n != "" {
		return n
	}
	return "y"
}

// v128
func (o *wasmOp) ResultType() string {
	if o.Flag(IsRelation) {
		return o.T().MaskFor().Name
	}
	if o.resultType == "" {
		return o.T().Name
	}
	return o.resultType
}

func (o *wasmOp) ArgType() string {
	if o.argType == "" {
		return o.T().Name
	}
	return o.argType
}

// I8x16Sub -- no need to prefix the "A", that is handled by the SSA rules generator
func (o *wasmOp) AsmOp() string {
	r := o.SsaWasmOp()
	return r
}

var (
	// Sets of types assocated with various (groups of) wasm instructions

	allTypes = []*simdType{vi8, vi16, vi32, vi64, vu8, vu16, vu32, vu64, vf32, vf64}
	signed   = []*simdType{vi8, vi16, vi32, vi64}
	unsigned = []*simdType{vu8, vu16, vu32, vu64}
	ints     = []*simdType{vi8, vi16, vi32, vi64, vu8, vu16, vu32, vu64}
	floats   = []*simdType{vf32, vf64}
	sle16    = []*simdType{vi8, vi16}                          // signed LEQ 16 bits
	ule16    = []*simdType{vu8, vu16}                          // unsigned LEQ 16 bits
	ige16    = []*simdType{vi16, vi32, vi64, vu16, vu32, vu64} // signed GEQ 16 bits
	nge32    = []*simdType{vi32, vi64, vu32, vu64, vf32, vf64} // numbers GEQ 32 bits
	sle32    = []*simdType{vi8, vi16, vi32}                    // signed LEQ 32 bits
	ule32    = []*simdType{vu8, vu16, vu32}                    // unsigned LEQ 32 bits
	ieq32    = []*simdType{vi32, vu32}                         // integer EQ 32 bits (convert_lo)
	masks    = []*simdType{vm8, vm16, vm32, vm64}

	// All these names are transcribed roughly directly from the WASM 3 list of vector instructions on page 22
	// prefix_suffix translates to receivertype_category
	//
	// receiver type = i(int), s(signed), u(unsigned), f(float), n(number); w/ possible size restriction suffix.
	//
	// 1, 2, 3 = unary, binary, ternary; suffix in number indicates arg type(s)
	// t = test (unary w/ signed scalar result);
	// r = relation (binary w/ signed vector result);
	// c = convert (unary w/ type change)
	// Suffix "q" means "have not figured out what this operation does yet"

	iv_1 = []string{"not"}                        // vector, unary -- but not float
	iv_2 = []string{"and", "andnot", "or", "xor"} // vector, binary -- but not float
	v_3  = []string{"bitSelect"}                  // vector, ternary

	v_t = []string{"any_true"} // vector, test (scalar result)

	s_1  = []string{"abs", "neg"}                                              // integer, unary
	f_1  = []string{"abs", "neg", "sqrt", "ceil", "floor", "trunc", "nearest"} // float, unary
	i8_1 = []string{"popcnt"}                                                  // int8, unary

	i_2     = []string{"add", "sub"}                       // integer, binary
	sle16_2 = []string{"add_sat_s", "sub_sat_s"}           // signed 8, 16, binary
	ule16_2 = []string{"add_sat_u", "sub_sat_u", "avgr_u"} // unsigned 8, 16, binary
	ige16_2 = []string{"mul"}                              // integer 16, 32, 64, binary
	s16_2   = []string{"q15mulr_sat_s", "relaxed_q15mulr_s"}
	sle32_2 = []string{"min_s", "max_s"}
	ule32_2 = []string{"min_u", "max_u"}

	f_2 = []string{"add", "sub", "mul", "div", "min", "max", "pmin", "pmax", "relaxed_min", "relaxed_max"}

	i_3 = []string{"relaxed_laneselect"}

	f_3 = []string{"relaxed_madd", "relaxed_nmadd"}

	i_t = []string{"all_true"} // shape, test (scalar result)

	// WASM SIMD relations return vectors of integer 0/1-valued lanes.  To use these as bitwise masks,
	// they need to be negated.

	i_r     = []string{"eq", "ne"}
	s_r     = []string{"lt_s", "gt_s", "le_s", "ge_s"}
	ule32_r = []string{"lt_u", "gt_u", "le_u", "ge_u"}

	f_r = []string{"eq", "ne", "lt", "gt", "le", "ge"}

	// TODO see if semantics can be derived from spec.
	i8_swiz = []string{"swizzle"} //, "relaxed_swizzle"}
	i8_shuf = []string{"shuffle"}

	// shift ops; integer arg, vector result
	u_s = []string{"shl", "shr_u"}
	s_s = []string{"shl", "shr_s"}

	sle16_q1 = []string{"extadd_pairwise_s"}
	ule16_q1 = []string{"extadd_pairwise_u"}

	s_q2   = []string{"extmul_low_s", "extmul_high_s"}
	u_q2   = []string{"extmul_low_u", "extmul_high_u"}
	s16_q2 = []string{"dot_s"}
	s8_q2  = []string{"relaxed_dot_s"}

	s8_q3 = []string{"relaxed_dot_add_s"}

	// extend_{u,s} widen integers x2; convert_halfs widen integers to floats x2
	// naming, to match amd64, is
	// ExtendLo<N>To<Type> where result type is <Type>x<N>
	extend_u      = []string{"extend_low_u", "extend_high_u"}
	extend_s      = []string{"extend_low_s", "extend_high_s"}
	convert_low_u = []string{"convert_low_u"}
	convert_low_s = []string{"convert_low_s"}

	// In same size, 32 bit integer to float.
	convert_s = []string{"convert_s"}
	convert_u = []string{"convert_u"}

	f_c   = []string{"trunc_sat_s", "trunc_sat_u"} // f32 -> s/u 32 (2 choices)
	f32_c = []string{"promote_low"}                // produces f64 vector
	f64_c = []string{"demote_zero"}                // produces f32 vector

	// these have an immediate operand, and return an element_type operand
	nge32_x = []string{"extract_lane"}
	sle16_x = []string{"extract_lane_s"}
	ule16_x = []string{"extract_lane_u"}

	// has an immediate operand, and an element type operand
	n_x = []string{"replace_lane"}

	rotates = []string{"RotateAllLeft", "RotateAllRight"}

	// TODO, once the semantics are understood.
	// N1xM1.narrow_N2xM2_{S,U}
	// N1xM1 is the result type.  There are two N2xM2 inputs, where N1 = N2/2 and M1 = M2*2
	// Each N2-sized input element is narrow, saturating signed or unsigned in the process.

	// Some operations are renamed, and some are not directly present as an intrinsic
	// or generic SIMD operation.  A name beginning "_" means the intrinsic begins with
	// that underscore and there must be an emulation written in terms of that intrinsic.
	gonames = map[string]string{
		// not a direct intrinsic
		"any_true": "-", // not IsZero()
		// rename
		"add_sat_s":    "AddSaturated",
		"add_sat_u":    "AddSaturated",
		"sub_sat_s":    "SubSaturated",
		"sub_sat_u":    "SubSaturated",
		"andnot":       "AndNot",
		"nearest":      "Round",
		"popcnt":       "OnesCount",
		"avgr_u":       "Average",
		"eq":           "Equal",
		"ne":           "NotEqual",
		"le":           "LessEqual",
		"ge":           "GreaterEqual",
		"lt":           "Less",
		"gt":           "Greater",
		"le_s":         "LessEqual",
		"ge_s":         "GreaterEqual",
		"lt_s":         "Less",
		"gt_s":         "Greater",
		"le_u":         "LessEqual",
		"ge_u":         "GreaterEqual",
		"lt_u":         "Less",
		"gt_u":         "Greater",
		"relaxed_madd": "-",
		"shl":          "ShiftAllLeft",

		"extract_lane":   "GetElem",
		"extract_lane_s": "GetElem",
		"extract_lane_u": "GetElem",

		"replace_lane": "SetElem",

		"bitselect": "BitSelect",

		// remove sign/unsigned
		"min_s": "Min",
		"max_s": "Max",
		"min_u": "Min",
		"max_u": "Max",
		"shr_u": "ShiftAllRight",
		"shr_s": "ShiftAllRight",

		// need to figure out what these are.
		"q15mulr_sat_s":     "?",
		"relaxed_q15mulr_s": "?",
		// not sure we need these
		"pmin":        "-",
		"pmax":        "-",
		"relaxed_min": "-",
		"relaxed_max": "-",
		// rename
		// available via optimization.
		"relaxed_nmadd": "-",
		// not sure of the exact name
		"all_true": "?",
		// need to verify semantics
		"extadd_pairwise_s": "?",
		"extadd_pairwise_u": "?",
		"extmul_low_s":      "MulWidenLo",
		"extmul_low_u":      "MulWidenLo",
		"extmul_high_s":     "MulWidenHi",
		"extmul_high_u":     "MulWidenHi",
		"dot_s":             "?",
		"relaxed_dot_s":     "?",

		// widen integer types
		"extend_low_s":  "ExtendLo%dTo%s",
		"extend_high_s": "ExtendHi%dTo%s",
		"extend_low_u":  "ExtendLo%dTo%s",
		"extend_high_u": "ExtendHi%dTo%s",

		// [u]int32x4 to float32x4
		"convert_s": "ConvertToFloat32",
		"convert_u": "ConvertToFloat32",

		// [u]int32x4 to float32x4
		"trunc_sat_s": "ConvertToInt32",
		"trunc_sat_u": "ConvertToUint32",

		// [u]int32x4 to float64x2
		"convert_low_s": "ConvertLo2ToFloat64",
		"convert_low_u": "ConvertLo2ToFloat64",

		"swizzle": "LookupOrZero",
		"splat":   "Broadcast",
	}
)

func initWasmOps() {
	// Local flag-setting functions handle various special cases.
	isBitwise := func(s string, _ *simdType) OpFlags {
		return IsBitwise
	}
	isTest := func(s string, _ *simdType) OpFlags {
		if s == "any_true" {
			return IsBitwise | IsTest
		}
		return IsTest
	}
	binBitwise := func(s string, t *simdType) OpFlags {
		if s == "andnot" {
			return IsBitwise
		}
		return IsBitwise | IsCommutative
	}
	unShape := func(s string, t *simdType) OpFlags {
		return 0
	}
	binShape := func(s string, t *simdType) OpFlags {
		if !t.Float && t.IntShaped != nil {
			if s == "add" {
				return IsCommutative | NonSigned
			}
			if s == "sub" {
				return NonSigned
			}
		}
		if strings.HasPrefix(s, "sub") || s == "div" {
			return 0
		}
		return IsCommutative
	}
	isMask := func(s string, t *simdType) OpFlags {
		flags := IsRelation
		if s == "eq" || s == "ne" {
			flags |= NonSigned
		} else {
			flags |= IsBitwise
		}
		if s == "ne" || s == "eq" || s == "and" || s == "or" || s == "xor" {
			flags |= IsCommutative
		}
		return flags
	}

	isRelation := func(s string, t *simdType) OpFlags {
		flags := IsRelation

		if s == "ne" || s == "eq" || s == "and" || s == "or" || s == "xor" {
			flags |= IsCommutative
		}
		return flags
	}

	bitSelect := func(op *wasmOp) {
		op.opFlags = IsBitwise
		op.arg2Name = "cond"
	}

	addWasmOps(ints, iv_1, 1, isBitwise)  // because reasons, not floats
	addWasmOps(ints, iv_2, 2, binBitwise) // because reasons, not floats
	addWasmOpsDetail(ints, v_3, 3, bitSelect)
	addWasmOps(allTypes, v_t, 1, isTest)
	addWasmOps(signed, s_1, 1, unShape)
	addWasmOps(floats, f_1, 1, unShape)
	addWasmOps([]*simdType{vi8}, i8_1, 1, nil)

	addWasmOps(ints, i_2, 2, binShape)
	addWasmOps(sle16, sle16_2, 2, binShape)
	addWasmOps(ule16, ule16_2, 2, binShape)
	addWasmOps(ige16, ige16_2, 2, binShape)

	addWasmOps([]*simdType{vi16}, s16_2, 2, nil) // afaik not commutative

	addWasmOps(sle32, sle32_2, 2, binShape)
	addWasmOps(ule32, ule32_2, 2, binShape)

	addWasmOps(floats, f_2, 2, binShape)

	// addWasmOps(floats, f_3, 3, nil) // relaxed_madd does not work
	// addWasmOps(ints, i_3, 3, nil)
	addWasmOps(ints, i_t, 1, isTest)

	addWasmOps(ints, i_r, 2, isRelation)
	addWasmOps(signed, s_r, 2, isRelation)
	addWasmOps(ule32, ule32_r, 2, isRelation)

	addWasmOps(floats, f_r, 2, isRelation)

	// Shuffle is a mess, it takes a 8x16 vector in and SIXTEEN immediates specifying the indices.
	// addWasmOps([]*simdType{vi8}, i8_shuf, 1, nil)
	addWasmOpsDetail([]*simdType{vi8}, i8_swiz, 2, func(op *wasmOp) { op.arg1Name = "i" })

	// Masks have some operations.
	addWasmOps(masks, iv_2, 2, isMask)

	extractImmediate := func(op *wasmOp) {
		op.resultType = op.t.Elem
		op.immRange = uint8(op.t.Count)
		op.immName = "index"
	}

	addWasmOpsDetail(nge32, nge32_x, 1, extractImmediate)
	addWasmOpsDetail(sle16, sle16_x, 1, extractImmediate)
	addWasmOpsDetail(ule16, ule16_x, 1, extractImmediate)

	replaceImmediate := func(op *wasmOp) {
		op.argType = op.t.Elem
		op.immRange = uint8(op.t.Count)
		op.immName = "index"
	}
	addWasmOpsDetail(allTypes, n_x, 2, replaceImmediate)

	shift := func(op *wasmOp) {
		op.argType = "uint64"
		op.opFlags = IsShift
	}
	addWasmOpsDetail(signed, s_s, 2, shift)
	addWasmOpsDetail(unsigned, u_s, 2, shift)

	splat := func(op *wasmOp) {
		op.opFlags = IsSplat
		op.argType = op.t.Elem
		op.resultType = op.t.Name
		if op.argType[0] == 'u' {
			op.opFlags |= NonSigned
		}
	}
	addWasmOpsDetail(allTypes, []string{"splat"}, 1, splat)

	// To match the extend patterns for amd64, signed extends to signed, unsigned extends to unsigned
	extendHalf := func(op *wasmOp) {
		t := op.t
		op.opFlags = IsConversion | NameHasFormat
		// result type is twice the width, signedness from the op, half the count
		stem := "Int"
		if op.op[len(op.op)-1] == 'u' {
			stem = "Uint"
		}
		op.resultType = t.WidenElements(stem)
	}
	mulHalf := func(op *wasmOp) {
		t := op.t
		op.opFlags = IsConversion | IsCommutative // this is ALSO a conversion, with same naming conventions.
		// result type is twice the width, signedness from the op, half the count
		stem := "Int"
		if op.op[len(op.op)-1] == 'u' {
			stem = "Uint"
		}
		op.resultType = t.WidenElements(stem)
	}
	convertHalf := func(op *wasmOp) {
		// amd64 has these instructions but we use the vector-register-widening ones,
		// thus the generics are not defined by amd64.
		op.opFlags = IsConversion
		op.resultType = "Float64x2"
	}
	convert := func(op *wasmOp) {
		op.opFlags = IsConversion
		op.resultType = "Float32x4"
	}
	truncSat := func(op *wasmOp) {
		op.opFlags = IsConversion
		if op.op == "trunc_sat_s" {
			op.resultType = "Int32x4"
		} else {
			op.resultType = "Uint32x4"
		}
	}

	rotate := func(op *wasmOp) {
		op.opFlags = EmulatedRule | IsShift
		op.argType = "uint64"
		op.arg1Name = "shift"
	}

	addWasmOpsDetail(ule32, extend_u, 1, extendHalf)
	addWasmOpsDetail(sle32, extend_s, 1, extendHalf)
	addWasmOpsDetail([]*simdType{vi32}, convert_low_s, 1, convertHalf)
	addWasmOpsDetail([]*simdType{vu32}, convert_low_u, 1, convertHalf)
	addWasmOpsDetail([]*simdType{vi32}, convert_s, 1, convert)
	addWasmOpsDetail([]*simdType{vu32}, convert_u, 1, convert)

	addWasmOpsDetail([]*simdType{vf32}, f_c, 1, truncSat)

	addWasmOpsDetail(sle32, s_q2, 2, mulHalf)
	addWasmOpsDetail(ule32, u_q2, 2, mulHalf)

	addWasmOpsDetail(ints, rotates, 2, rotate)

	slices.SortFunc(wasmOps, compareWasmOps)

	for i := 1; i < len(wasmOps); i++ {
		c := compareWasmOps(wasmOps[i-1], wasmOps[i])
		if c >= 0 {
			d := compareWasmOps(wasmOps[i-1], wasmOps[i])
			fmt.Printf("Two wasm ops compared out of order, c=%d, \n%v\n%v\n", d, wasmOps[i-1], wasmOps[i])
		}
	}
}

var wasmOps = []*wasmOp{}

// Given a slice of simd types and a slice of operations with the specified argCount,
// add the resulting wasm operations to wasmOps.  after is a function that applies
// operation-specific customization to the generated wasmOp.
func addWasmOpsDetail(types []*simdType, ops []string, argCount int, after func(op *wasmOp)) {
	for _, t := range types {
		for _, o := range ops {
			op := &wasmOp{t: t, op: o, argCount: argCount}
			after(op)
			if t.Methods[o] != nil {
				panic("Double addition of method " + o + " for " + t.Name)
			}
			t.Methods[o] = op
			wasmOps = append(wasmOps, op)
		}
	}

}

// Given a slice of simd types and a slice of operatinos with the specified argCount,
// add the resulting wasm operations to wasmOps.  flags is an optional function that
// adjusts the flags of the WasmOp.
func addWasmOps(types []*simdType, ops []string, argCount int, flags func(op string, ty *simdType) OpFlags) {
	addWasmOpsDetail(types, ops, argCount, func(op *wasmOp) {
		if flags == nil {
			return
		}
		op.opFlags = flags(op.op, op.T())
	})
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}

func mustVal[T any](v T, err error) T {
	if err != nil {
		panic(err)
	}
	return v
}

func main() {

	flag.Parse()

	initWasmOps()

	if *list {
		// Helper to list opcodes for a.out.go
		for i := range wasmOps {
			fmt.Println(wasmOps[i])
		}
		return
	}

	*genTypesFile = mustVal(filepath.Abs(strings.ReplaceAll(*genTypesFile, "GOROOTSRC", *gorootsrc)))
	*genOpsFile = mustVal(filepath.Abs(strings.ReplaceAll(*genOpsFile, "GOROOTSRC", *gorootsrc)))
	*genSSAOpsFile = mustVal(filepath.Abs(strings.ReplaceAll(*genSSAOpsFile, "GOROOTSRC", *gorootsrc)))
	*genGenOpsFile = mustVal(filepath.Abs(strings.ReplaceAll(*genGenOpsFile, "GOROOTSRC", *gorootsrc)))
	*genSSARulesFile = mustVal(filepath.Abs(strings.ReplaceAll(*genSSARulesFile, "GOROOTSRC", *gorootsrc)))
	*genWasmSSAFile = mustVal(filepath.Abs(strings.ReplaceAll(*genWasmSSAFile, "GOROOTSRC", *gorootsrc)))
	*genIntrinsicsFile = mustVal(filepath.Abs(strings.ReplaceAll(*genIntrinsicsFile, "GOROOTSRC", *gorootsrc)))

	log.Println("types file =", *genTypesFile)
	log.Println("ops file =", *genOpsFile)
	log.Println("ssa ops file =", *genSSAOpsFile)
	log.Println("ssa generic ops file =", *genGenOpsFile)
	log.Println("ssa rules file =", *genSSARulesFile)
	log.Println("ssa wasm ops file =", *genWasmSSAFile)
	log.Println("intrinsics file =", *genIntrinsicsFile)

	log.Println("Generating WASM SIMD files...")

	sgutil.FormatWriteAndClose(genTypes(), *genTypesFile)
	sgutil.FormatWriteAndClose(genOps(), *genOpsFile)
	sgutil.FormatWriteAndClose(genSSAOps(), *genSSAOpsFile)
	sgutil.FormatWriteAndClose(genWasmSSA(), *genWasmSSAFile)
	sgutil.FormatWriteAndClose(genIntrinsics(), *genIntrinsicsFile)
	sgutil.FormatWriteAndClose(genGenerics(), *genGenOpsFile)

	genSSARules()

}

func templateOf(name, text string) *template.Template {
	return template.Must(template.New(name).Parse(text))
}

var loadDecl = templateOf("load from array", `
// Load{{.Name}}Array loads {{.Article}} {{.Name}} from a [{{.Count}}]{{.Elem}}.
//
//go:noescape
func Load{{.Name}}Array(y *[{{.Count}}]{{.Elem}}) {{.Name}}

// Load{{.Name}} loads {{.Article}} {{.Name}} from a slice of at least {{.Count}} {{.Elem}}s.
func Load{{.Name}}(s []{{.Elem}}) {{.Name}} {
	return Load{{.Name}}Array((*[{{.Count}}]{{.Elem}})(s))
}
`)

var storeDecl = templateOf("store to array", `
// StoreArray stores {{.Article}} {{.Name}} to a [{{.Count}}]{{.Elem}}.
//
//go:noescape
func (x {{.Name}}) StoreArray(y *[{{.Count}}]{{.Elem}})

// Store stores x into a slice of at least {{.Count}} {{.Elem}}s.
func (x {{.Name}}) Store(s []{{.Elem}}) {
	x.StoreArray((*[{{.Count}}]{{.Elem}})(s))
}
`)

var splatDecl = templateOf("broadcast an element", `
// Broadcast{{.Name}} broadcasts {{.Article}} {{.Elem}} to all elements of {{.Article}} {{.Name}} vector.
func Broadcast{{.Name}}(x {{.Elem}}) {{.Name}}
`)

var typeDecl = templateOf("type decl", `
// {{.Name}} is a 128-bit SIMD vector of {{.Count}} {{.Elem}}s.
type {{.Name}} struct {
	{{.Elem}}x{{.Count}} v128
	vals      [{{.Count}}]{{.Elem}}
}
`)

var maskTypeDecl = templateOf("type decl", `
// {{.Name}} is a 128-bit SIMD mask of {{.Count}} {{.Elem}}s.
type {{.Name}} struct {
	{{.Elem}}x{{.Count}} v128
	vals      [{{.Count}}]{{.Elem}}
}
`)

var lenDecl = templateOf("len decl", `
// Len returns the number of elements in {{.Article}} {{.Name}}.
func (x {{.Name}}) Len() int { return {{.Count}} }
`)

func genTypes() (f *bytes.Buffer) {
	f = new(bytes.Buffer)
	fmt.Fprintln(f, "// Code generated by 'wasmgen'; DO NOT EDIT.")
	fmt.Fprintln(f)
	fmt.Fprintln(f, "//go:build goexperiment.simd && wasm")
	fmt.Fprintln(f)
	fmt.Fprintln(f, "package archsimd")
	fmt.Fprintln(f)
	fmt.Fprintln(f, "// v128 is a tag type that tells the compiler that this is really 128-bit SIMD")
	fmt.Fprintln(f, "type v128 struct {")
	fmt.Fprintln(f, "\t_128 [0]func() // uncomparable")
	fmt.Fprintln(f, "}")

	for _, t := range allTypes {
		typeDecl.Execute(f, t)
		lenDecl.Execute(f, t)
		loadDecl.Execute(f, t)
		storeDecl.Execute(f, t)
		splatDecl.Execute(f, t)
	}
	for _, t := range masks {
		maskTypeDecl.Execute(f, t)
	}
	return
}

var docForOp map[string]string = map[string]string{
	"Add":                 " returns the result of adding x and y, elementwise.",
	"Sub":                 " returns the result of subtracting y from x, elementwise.",
	"Mul":                 " returns the result of multiplying x and y, elementwise.",
	"Div":                 " returns the result of dividing x by y, elementwise.",
	"Neg":                 " returns the elementwise negation of x.",
	"Abs":                 " returns the elementwise absolute value of x.",
	"Sqrt":                " returns the elementwise square root of x.",
	"Not":                 " returns the bitwise NOT of x.",
	"And":                 " returns the bitwise AND of x and y.",
	"Or":                  " returns the bitwise OR of x and y.",
	"Xor":                 " returns the bitwise XOR of x and y.",
	"AndNot":              " returns the bitwise AND NOT of x and y (x & ^y).",
	"Min":                 " returns the elementwise minimum of x and y.",
	"Max":                 " returns the elementwise maximum of x and y.",
	"Round":               " returns the elementwise nearest integer, rounding ties to even.",
	"OnesCount":           " returns the elementwise population count (number of bits set).",
	"Average":             " returns the elementwise average of unsigned integers in x and y.",
	"Equal":               " returns true if x equals y, elementwise.",
	"NotEqual":            " returns true if x does not equal y, elementwise.",
	"Less":                " returns true if x is less than y, elementwise.",
	"Greater":             " returns true if x is greater than y, elementwise.",
	"LessEqual":           " returns true if x is less than or equal to y, elementwise.",
	"GreaterEqual":        " returns true if x is greater than or equal to y, elementwise.",
	"MulAdd":              " returns the elementwise multiply-add of x, y, and z.",
	"ShiftAllLeft":        " returns the elementwise left shift of x by y bits.",
	"ShiftAllRight":       " returns the elementwise right shift of x by y bits.",
	"Ceil":                " returns the elementwise ceiling of x.",
	"Floor":               " returns the elementwise floor of x.",
	"Trunc":               " returns the elementwise truncation of x.",
	"BitSelect":           " returns the bitwise selection if mask[i] then x[i] else y[i]",
	"GetElem":             " gets the lane value at the given index.",
	"SetElem":             " sets the lane at the given index to y.",
	"TruncSatS":           " returns the elementwise saturating signed conversion to integer.",
	"TruncSatU":           " returns the elementwise saturating unsigned conversion to integer.",
	"PromoteLow":          " promotes the lower half elements of x to double width values.",
	"DemoteZero":          " demotes elements of x to half width values in lower elements of result, with zeroes in the upper elements",
	"ConvertToFloat32":    " converts elements of x to Float32x4.",
	"ConvertToInt32":      " converts elements of x to Int32x4.",
	"ConvertToUint32":     " converts elements of x to Uint32x4.",
	"ConvertLo2ToFloat64": " converts the first two elements of x to Float64x2.",
	"MulWidenHi": ` returns the doubled-width product of respective elements of the upper halves of x and y.
//
//	Result[i] = x[i+{{.Type.HalfCount}}] * y[i+{{.Type.HalfCount}}], for 0 <= i < {{.Type.HalfCount}} == |x|/2.`,
	"MulWidenLo": ` returns the doubled-width product of respective elements of the lower halves of x and y.
//
//	Result[i] = x[i] * y[i], for 0 <= i < {{.Type.HalfCount}} == |x|/2.`,
	// Size-specific extend operations
	"ExtendLo8ToInt16":   " extends the lower 8 elements of x to 16-bit integers.",
	"ExtendLo16ToInt32":  " extends the lower 4 elements of x to 32-bit integers.",
	"ExtendLo32ToInt64":  " extends the lower 2 elements of x to 64-bit integers.",
	"ExtendHi8ToInt16":   " extends the higher 8 elements of x to 16-bit integers.",
	"ExtendHi16ToInt32":  " extends the higher 4 elements of x to 32-bit integers.",
	"ExtendHi32ToInt64":  " extends the higher 2 elements of x to 64-bit integers.",
	"ExtendLo8ToUint16":  " extends the lower 8 elements of x to 16-bit unsigned integers.",
	"ExtendLo16ToUint32": " extends the lower 4 elements of x to 32-bit unsigned integers.",
	"ExtendLo32ToUint64": " extends the lower 2 elements of x to 64-bit unsigned integers.",
	"ExtendHi8ToUint16":  " extends the higher 8 elements of x to 16-bit unsigned integers.",
	"ExtendHi16ToUint32": " extends the higher 4 elements of x to 32-bit unsigned integers.",
	"ExtendHi32ToUint64": " extends the higher 2 elements of x to 64-bit unsigned integers.",
	"AddSaturated":       " returns the result of adding x and y, saturating instead of overflowing, elementwise.",
	"SubSaturated":       " returns the result of subtracting x and y, saturating instead of overflowing, elementwise.",
	"Shuffle":            " returns the elements of y concatenated with z that are selected by elements of x",
	"LookupOrZero": ` returns the elements of x as indexed by the elements of i. If an index is out of range, its result is 0.
//
//	if 0 <= indices[i] && indices[i] < len(table) {
//	    result[i] = table[indices[i]]
//	} else {
//	    result[i] = 0
//	}`,
	"RelaxedSwizzle":    "",
	"RelaxedLaneselect": "",
}

func (w *wasmOp) DocRest() string {
	m := w.Method()
	d := docForOp[m]
	if d == "" {
		return ""
	}

	var buf bytes.Buffer
	if e := templateOf(m, d).Execute(&buf, w); e != nil {
		panic(e)
	}
	return buf.String()
}

var unOp = templateOf("unaryOp", `
	// {{.Method}}{{.DocRest}}
	//
	// Asm: {{.AsmOp}}
	func (x {{.RcvrType}}) {{.Method}}() {{.ResultType}}
`)

var binOp = templateOf("binaryOp", `
	// {{.Method}}{{.DocRest}}
	//
	// Asm: {{.AsmOp}}
	func (x {{.RcvrType}}) {{.Method}}({{.Arg1Name}} {{.ArgType}}) {{.ResultType}}
`)

var ternOp = templateOf("ternaryOp", `
	// {{.Method}}{{.DocRest}}
	//
	// Asm: {{.AsmOp}}
	func (x {{.RcvrType}}) {{.Method}}(y {{.RcvrType}}, {{.Arg2Name}} {{.ArgType}}) {{.ResultType}}
`)

var unOpImm = templateOf("unaryOpImm", `
	// {{.Method}}{{.DocRest}}
	//
	// Asm: {{.AsmOp}}
	func (x {{.RcvrType}}) {{.Method}}({{.ImmName}} uint8) {{.ResultType}}
`)

var binOpImm = templateOf("binaryOpImm", `
	// {{.Method}}{{.DocRest}}
	//
	// Asm: {{.AsmOp}}
	func (x {{.RcvrType}}) {{.Method}}({{.ImmName}} uint8, y {{.ArgType}}) {{.ResultType}}
`)

var toMask = templateOf("toMask", `
	// ToMask translates {{.Article}} {{.From.Name}} vector to a {{.To.Name}} mask vector
	// zero becomes false, not-zero becomes true
	func (x {{.From.Name}}) ToMask() {{.To.Name}}
`)

var fromMask = templateOf("fromMask", `
	// To{{.To.Name}} translates a {{.From.Name}} mask vector to {{.Article}} {{.To.Name}} int vector
	// false becomes zero, true becomes -1
	func (x {{.From.Name}}) To{{.To.Name}}() {{.To.Name}}
`)

var maskMergeUnsigned = templateOf("maskMerge",
	`// Masked returns x but with elements zeroed where mask is false.
func (x {{.Name}}) Masked(mask Mask{{.ElemSize}}x{{.Count}}) {{.Name}} {
	im := mask.ToInt{{.ElemSize}}x{{.Count}}().ToBits()
	return im.And(x)
}

// IfElse returns x but with elements set to y where mask is false.
func (x {{.Name}}) IfElse(mask Mask{{.ElemSize}}x{{.Count}}, y {{.Name}}) {{.Name}} {
	im := mask.ToInt{{.ElemSize}}x{{.Count}}().ToBits()
	return x.BitSelect(y, im)
}
`)

var maskMergeFloat = templateOf("maskMerge",
	`// Masked returns x but with elements zeroed where mask is false.
func (x {{.Name}}) Masked(mask Mask{{.ElemSize}}x{{.Count}}) {{.Name}} {
	im := mask.ToInt{{.ElemSize}}x{{.Count}}().ToBits()
	return im.And(x.ToBits()).BitsToFloat{{.ElemSize}}()
}

// IfElse returns x but with elements set to y where mask is false.
func (x {{.Name}}) IfElse(mask Mask{{.ElemSize}}x{{.Count}}, y {{.Name}}) {{.Name}} {
	im := mask.ToInt{{.ElemSize}}x{{.Count}}().ToBits()
	ix := x.ToBits()
	iy := y.ToBits()
	return ix.BitSelect(iy, im).BitsToFloat{{.ElemSize}}()
}
`)

var maskMergeInt = templateOf("maskMergeInt",
	`// Masked returns x but with elements zeroed where mask is false.
func (x {{.Name}}) Masked(mask Mask{{.ElemSize}}x{{.Count}}) {{.Name}} {
	im := mask.ToInt{{.ElemSize}}x{{.Count}}()
	return im.And(x)
}

// IfElse returns x but with elements set to y where mask is false.
func (x {{.Name}}) IfElse(mask Mask{{.ElemSize}}x{{.Count}}, y {{.Name}}) {{.Name}} {
	im := mask.ToInt{{.ElemSize}}x{{.Count}}()
	return x.BitSelect(y, im)
}
`)

var toString = templateOf("toString",
	`// String returns a string representation of SIMD vector x.
func (x {{.Name}}) String() string {
	var s [{{.Count}}]{{.Elem}}
	x.StoreArray(&s)
	return sliceToString(s[:])
}
`)

var maskToString = templateOf("maskToString",
	`// String returns a string representation of SIMD mask x.
func (x Mask{{.ElemSize}}x{{.Count}}) String() string {
	var s [{{.Count}}]{{.Elem}}
	x.ToInt{{.ElemSize}}x{{.Count}}().Neg().StoreArray(&s)
	return sliceToString(s[:])
}
`)

type asConversion struct {
	From, To *simdType
	Article  string
}

func forAllAsConversions(f func(from, to *simdType)) {
	for _, from := range allTypes {
		for _, to := range allTypes {
			if from == to {
				continue
			}
			f(from, to)
		}
	}
}

func fromUnsignedToFloats(f func(from, to *simdType)) {
	for _, to := range floats {
		from := to.UintFor()
		f(from, to)
	}
}

func fromUnsignedToInts(f func(from, to *simdType)) {
	for _, to := range signed {
		from := to.UintFor()
		f(from, to)
	}
}

func forAllReshape(f func(from, to *simdType)) {
	for _, from := range unsigned {
		for _, to := range unsigned {
			if from == to {
				continue
			}
			f(from, to)
		}
	}
}

func genOps() (f *bytes.Buffer) {
	f = new(bytes.Buffer)
	fmt.Fprintln(f, "// Code generated by 'wasmgen'; DO NOT EDIT.")
	fmt.Fprintln(f)
	fmt.Fprintln(f, "//go:build goexperiment.simd && wasm")
	fmt.Fprintln(f)
	fmt.Fprintln(f, "package archsimd")
	fmt.Fprintln(f)

	// Basic operations
	for _, op := range wasmOps {
		if op.OpFlags()&(IsLoad|IsStore|IsSplat) != 0 {
			continue // Handled elsewhere or skipped for methods
		}

		if op.Method() == "" {
			continue
		}

		if op.ImmRange() > 0 {
			switch op.ArgCount() {
			case 1:
				unOpImm.Execute(f, op)
			case 2:
				binOpImm.Execute(f, op)
			default:
				panic(fmt.Errorf("Unexpected arg count %d for %v", op.ArgCount(), op))
			}

		} else {
			switch op.ArgCount() {
			case 1:
				unOp.Execute(f, op)
			case 2:
				binOp.Execute(f, op)
			case 3:
				ternOp.Execute(f, op)
			default:
				panic(fmt.Errorf("Unexpected arg count %d for %v", op.ArgCount(), op))
			}
		}
	}

	for _, t := range signed {
		// Conversions to/from mask types
		// func (x Int8x16) ToMask() Mask8x16
		// func (x Mask8x16) ToInt8x16() Int8x16
		toMask.Execute(f, &asConversion{t, t.MaskFor(), t.Article()})
		fromMask.Execute(f, &asConversion{t.MaskFor(), t, t.Article()})
	}

	// Mask and Merge ops
	for _, t := range allTypes {
		if t.Name[0] == 'I' {
			maskMergeInt.Execute(f, t)
		} else if t.Name[0] == 'F' {
			maskMergeFloat.Execute(f, t)
		} else {
			maskMergeUnsigned.Execute(f, t)
		}
	}

	// String
	for _, t := range allTypes {
		toString.Execute(f, t)
	}

	// Mask to String
	for _, t := range signed {
		maskToString.Execute(f, t)
	}

	fromUnsignedToFloats(func(from, to *simdType) {
		sgutil.ToFloatsDcl.Execute(f, sgutil.Conversion(from, to))
		sgutil.ToBitsDcl.Execute(f, sgutil.Conversion(to, from))
	})

	fromUnsignedToInts(func(from, to *simdType) {
		sgutil.ToIntsDcl.Execute(f, sgutil.Conversion(from, to))
		sgutil.ToBitsDcl.Execute(f, sgutil.Conversion(to, from))
	})

	forAllReshape(func(from, to *simdType) {
		sgutil.ReshapeDcl.Execute(f, sgutil.Conversion(from, to))
	})
	return
}

// genSSARules generates the definitions for WASM-specific SIMD SSA operations.
// The expected target directory is cmd/compile/internal/ssa/_gen
func genSSAOps() (f *bytes.Buffer) {
	f = new(bytes.Buffer)
	fmt.Fprintln(f, "// Code generated by 'wasmgen'; DO NOT EDIT.")
	fmt.Fprintln(f)
	fmt.Fprintln(f, "package main")
	fmt.Fprintln(f)

	fmt.Fprintln(f)
	fmt.Fprintln(f, "func simdWasmOps(vload, vstore, v11, v21, v31, v11gp, v11fp32, v11fp64, v1gpv, v1fp32v, v1fp64v, gpv, fp32v, fp64v regInfo) []opData {")
	fmt.Fprintln(f, "\treturn []opData{")

	done := make(map[string]string)

	for _, op := range wasmOps {
		if op.Flag(NonSigned | EmulatedRule) {
			// There's only one (signed input) version of the op, or no op at all.
			continue
		}
		var toPrint string
		ssaWasmOp := op.SsaWasmOp()
		// TODO also incorporate immediate operands.
		if op.ImmRange() > 0 {
			// These are currently not commutative
			toPrint = fmt.Sprintf("\t\t{name: \"%s\", argLength: %d, reg: %s, asm: \"%s\", aux: \"UInt8\", typ: \"%s\"},\n", ssaWasmOp, op.ArgCount(), op.RegInfo(), op.AsmOp(), op.SsaResultType())
		} else if op.Flag(IsCommutative) {
			toPrint = fmt.Sprintf("\t\t{name: \"%s\", argLength: %d, reg: %s, asm: \"%s\", commutative: true, typ: \"%s\"},\n", ssaWasmOp, op.ArgCount(), op.RegInfo(), op.AsmOp(), op.SsaResultType())
		} else {
			toPrint = fmt.Sprintf("\t\t{name: \"%s\", argLength: %d, reg: %s, asm: \"%s\", typ: \"%s\"},\n", ssaWasmOp, op.ArgCount(), op.RegInfo(), op.AsmOp(), op.SsaResultType())
		}
		if old := done[ssaWasmOp]; old != "" {
			if old != toPrint {
				panic(fmt.Errorf("Second definition of SSA WASM Op %s differed: \nold: %s\nnew: %s\n", ssaWasmOp, old, toPrint))
			}
			continue
		}
		done[ssaWasmOp] = toPrint
		fmt.Fprint(f, toPrint)

	}

	fmt.Fprintln(f, "\t}")
	fmt.Fprintln(f, "}")
	return
}

// genSSARules generates the rules that convert SSA generic (SIMD) operations
// into WASM-specific SIMD SSA operations.
// The expected target directory is cmd/compile/internal/ssa/_gen
func genSSARules() {
	f, err := os.Create(*genSSARulesFile)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	fmt.Fprintln(f, "// Code generated by 'wasmgen'; DO NOT EDIT.")
	fmt.Fprintln(f)

	for _, op := range wasmOps {
		// (GoOp x y) => (WasmOp x y)
		if op.Flag(NonSigned) {
			// skip these generics
			continue
		}
		g := op.SsaGenOp()
		if g == "" {
			continue
		}
		if op.T().IsMask() {
			continue // mask ops use Int generics
		}
		switch op.ArgCount() {
		case 1, 2, 3:
			wasmOp := op.SsaWasmOp()
			if op.Flag(IsShift) {
				t := op.T()
				elemSize := t.ElemSize
				if op.Op()[0] == 'S' { // shifts, not rotates
					fmt.Fprintf(f, "(%s x d:(Const64 [c])) && uint64(c) < %d => (%s x (I64Const [c]))\n", g, elemSize, wasmOp)
					fmt.Fprintf(f, "(%s x d:(I64Const [c])) && uint64(c) < %d => (%s x d)\n", g, elemSize, wasmOp)
					if op.op != "shr_s" { // shift right signed
						fmt.Fprintf(f, "// TODO need to do 'shiftIsBounded' for WASM SIMD Shifts\n")
						fmt.Fprintf(f, "(%s x y) => (SelectV (%s x y) (%s x x) (I64LtU y (I64Const [%d])))\n", g, wasmOp, t.Methods["xor"].SsaWasmOp(), elemSize)
					} else {
						// Signed, smear the sign bit
						fmt.Fprintf(f, "// TODO need to do 'shiftIsBounded' for WASM SIMD Shifts\n")
						fmt.Fprintf(f, "(%s x y) => (SelectV (%s x y) (%s x (I64Const [%d])) (I64LtU y (I64Const [%d])))\n", g, wasmOp, wasmOp, elemSize-1, elemSize)
					}
				} else { // rotates are okay with the implicit WASM modulus
					shl := t.UintFor().Methods["shl"].SsaWasmOp()
					shr := t.UintFor().Methods["shr_u"].SsaWasmOp()
					or := t.UintFor().Methods["or"].SsaWasmOp()
					if strings.Contains(op.Op(), "Left") {
						fmt.Fprintf(f, "(%s x y) => (%s (%s x y) (%s x (I64Sub (I64Const [%d]) y)))\n", g, or, shl, shr, elemSize)
					} else {
						fmt.Fprintf(f, "(%s x y) => (%s (%s x y) (%s x (I64Sub (I64Const [%d]) y)))\n", g, or, shr, shl, elemSize)
					}
				}
				continue
			}
			if op.Flag(NonSigned) {
				// hop over to the non-signed version of the type and use that method.
				wasmOp = op.T().IntShaped.Methods[op.op].SsaWasmOp()
			}
			fmt.Fprintf(f, "(%s ...) => (%s ...)\n", g, wasmOp)
			continue
		default:
			panic("Haven't figured out SSA rule for " + op.String())
		}

	}
}

// genWasmSSA creates the file/function that converts SSA nodes for WASM SIMD operations
// into the appropriate assembly language.
// The expected target directory is cmd/compile/internal/wasm
func genWasmSSA() (f *bytes.Buffer) {
	f = new(bytes.Buffer)

	fmt.Fprintln(f, "// Code generated by 'wasmgen'; DO NOT EDIT.")
	fmt.Fprintln(f)
	fmt.Fprintln(f, "package wasm")
	fmt.Fprintln(f)
	fmt.Fprintln(f, "import (")
	fmt.Fprintln(f, "\t\"cmd/compile/internal/ssa\"")
	fmt.Fprintln(f, "\t\"cmd/compile/internal/ssagen\"")
	fmt.Fprintln(f, "\t\"cmd/internal/obj\"")
	fmt.Fprintln(f, "\t\"cmd/internal/obj/wasm\"")
	fmt.Fprintln(f, ")")
	fmt.Fprintln(f)
	fmt.Fprintln(f, "func ssaGenSIMDValue(s *ssagen.State, v *ssa.Value, extend bool) bool {")
	fmt.Fprintln(f, "\tswitch v.Op {")

	const (
		NONE = iota
		IMM1_64
		IMM1_U
		IMM1_S

		IMM2_32
		IMM2_64

		// splat operations
		I32V
		I64V
		F32V
		F64V

		V
		VV
		V32
		VVV
		OP
		DONE
	)

	type classifiedOP struct {
		op    *wasmOp
		class int
	}
	var ssagenWasmOps []classifiedOP

	// Classify mimics the decision tree for how to convert
	// SSA HW-specific op into ASM, returning small integers
	// that can be sorted so that like operations are all
	// grouped together.
	classify := func(op *wasmOp) int {
		if op.ImmRange() > 0 {
			switch op.ArgCount() {
			case 1:
				if op.T().ElemSize < 64 {
					switch op.T().Elem[0] {
					case 'u':
						return IMM1_U
					case 'i':
						return IMM1_S
					}
				}
				return IMM1_64
			case 2:
				if op.T().ElemSize < 64 && !op.T().Float { // empirically, float32 is 64 bits wide.
					return IMM2_32
				}
				return IMM2_64
			}
		} else {
			switch op.ArgCount() {
			case 1:
				if op.Flag(IsSplat) {
					switch op.ArgType() {
					case "int8", "int16", "int32":
						return I32V
					case "int64":
						return I64V
					case "float32":
						return F32V
					case "float64":
						return F64V
					default:
						panic(fmt.Errorf("op %s has unexpected splat arg type", op.String()))
					}
				}
				return V
			case 2:
				if c := op.ArgType()[0]; c == 'i' || c == 'u' {
					return V32
				} else {
					return VV
				}
			case 3:
				return VVV
			default:
				return OP
			}
		}
		panic(fmt.Errorf("op %s has class NONE", op.String()))
	}

	done := make(map[string]bool)

	for _, op := range wasmOps {
		if op.Flag(NonSigned | EmulatedRule) {
			// no hardware-specific op to generate asm from
			continue
		}
		if done[op.SsaWasmOp()] {
			continue
		}
		done[op.SsaWasmOp()] = true
		ssagenWasmOps = append(ssagenWasmOps, classifiedOP{op, classify(op)})
	}

	slices.SortFunc(ssagenWasmOps, func(a, b classifiedOP) int {
		if c := a.class - b.class; c != 0 {
			return c
		}
		return compareWasmOps(a.op, b.op)
	})

	ssagenWasmOps = append(ssagenWasmOps, classifiedOP{nil, DONE})

	lastClass := NONE
	lastCR := 0
	for i, op := range ssagenWasmOps {
		if op.class != lastClass {
			lastCR = i
			// Print the appropriate action for ssagen
			switch lastClass {
			case NONE:
			case IMM1_64:
				fmt.Fprintf(f,
					`		getValue128(s, v.Args[0])
		p := s.Prog(v.Op.Asm())
		p.To = obj.Addr{Type: obj.TYPE_CONST, Offset: v.AuxInt}
`)
			case IMM1_U:
				fmt.Fprintf(f,
					`		getValue128(s, v.Args[0])
		p := s.Prog(v.Op.Asm())
		p.To = obj.Addr{Type: obj.TYPE_CONST, Offset: v.AuxInt}
`)
				fmt.Fprintf(f, "\tif extend {\n\t\ts.Prog(wasm.AI64ExtendI32U)\n\t}\n")

			case IMM1_S:
				fmt.Fprintf(f,
					`		getValue128(s, v.Args[0])
		p := s.Prog(v.Op.Asm())
		p.To = obj.Addr{Type: obj.TYPE_CONST, Offset: v.AuxInt}
`)
				fmt.Fprintf(f, "\tif extend {\n\t\ts.Prog(wasm.AI64ExtendI32S)\n\t}\n")

			case IMM2_32:
				fmt.Fprintf(f,
					`		getValue128(s, v.Args[0])
		getValue32(s, v.Args[1])
		p := s.Prog(v.Op.Asm())
		p.To = obj.Addr{Type: obj.TYPE_CONST, Offset: v.AuxInt}
`)
			case IMM2_64:
				fmt.Fprintf(f,
					`		getValue128(s, v.Args[0])
		getValue64(s, v.Args[1])
		p := s.Prog(v.Op.Asm())
		p.To = obj.Addr{Type: obj.TYPE_CONST, Offset: v.AuxInt}
`)

			case I32V:
				fmt.Fprintf(f,
					`		getValue32(s, v.Args[0])
		s.Prog(v.Op.Asm())
`)
			case I64V:
				fmt.Fprintf(f,
					`		getValue64(s, v.Args[0])
		s.Prog(v.Op.Asm())
`)
			case F32V:
				fmt.Fprintf(f,
					`		getValueFxx(s, v.Args[0])
		s.Prog(v.Op.Asm())
`)
			case F64V:
				fmt.Fprintf(f,
					`		getValueFxx(s, v.Args[0])
		s.Prog(v.Op.Asm())
`)

			case V:
				fmt.Fprintf(f,
					`		getValue128(s, v.Args[0])
		s.Prog(v.Op.Asm())
`)
			case VV:
				fmt.Fprintf(f,
					`		getValue128(s, v.Args[0])
		getValue128(s, v.Args[1])
		s.Prog(v.Op.Asm())
`)
			case V32:
				// shifts, 32-bit operand
				fmt.Fprintf(f,
					`		getValue128(s, v.Args[0])
		getValue32(s, v.Args[1])
		s.Prog(v.Op.Asm())
`)
			case VVV:
				fmt.Fprintf(f,
					`		getValue128(s, v.Args[0])
		getValue128(s, v.Args[1])
		getValue128(s, v.Args[2])
		s.Prog(v.Op.Asm())
`)
			case OP:
				fmt.Fprintf(f,
					`		s.Prog(v.Op.Asm())
`)
			}
			if op.class == DONE {
				fmt.Fprintln(f, `
	default:
		return false
	}
	return true
}`)
				return
			}
			// Otherwise begin the next case
			fmt.Fprint(f, "case ")
			lastCR = i - 1
			lastClass = op.class
		}

		sep := ","
		if op.class != ssagenWasmOps[i+1].class {
			sep = ":"
		}
		fmt.Fprintf(f, "ssa.OpWasm%s%s", op.op.SsaWasmOp(), sep)
		if i >= lastCR+3 {
			fmt.Fprintln(f)
			lastCR = i
		}
	}
	return f
}

// genGenerics creates SSA generic ops that are implied by WASM SIMD instructions
// that were not previously implied by AMD64 SIMD instructions.
// The expected target directory is cmd/compile/internal/ssa/_gen
func genGenerics() *bytes.Buffer {
	var newOps []sgutil.GenericOpsData

	for _, op := range wasmOps {
		if op.SsaGenOp() == "" {
			continue
		}
		if !op.DefinesGeneric() {
			continue
		}
		newOp := sgutil.GenericOpsData{
			OpName:  op.SsaGenOp(),
			OpInLen: op.ArgCount(),
			Comm:    op.Flag(IsCommutative),
			HasAux:  op.ImmRange() > 0,
		}
		newOps = append(newOps, newOp)
	}

	buf := sgutil.MergeSIMDGenericOps(newOps, *genGenOpsFile, "wasm")

	return buf
}

// genIntrinsics creates the function that registers all of the WASM SIMD intrinsics.
// The expected target directory is cmd/compile/internal/ssagen
func genIntrinsics() (f *bytes.Buffer) {
	f = new(bytes.Buffer)

	fmt.Fprint(f, `// Code generated by 'wasmgen'; DO NOT EDIT.

package ssagen

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/ssa"
	"cmd/compile/internal/types"
	"cmd/internal/sys"
)

func initWasmSIMD() {
	makeSimdOp1 := func(op ssa.Op) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue1(op, types.TypeVec128, args[0])
		}
	}
	makeSimdOp2 := func(op ssa.Op) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue2(op, types.TypeVec128, args[0], args[1])
		}
	}
	makeSimdOp3 := func(op ssa.Op) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue3(op, types.TypeVec128, args[0], args[1], args[2])
		}
	}

	// "As" is a type pun, just return the bits
	makeAsOp := func() func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return args[0]
		}
	}

	// converting to a mask is an not-equals comparison with zero, zero obtained by x XOR x.
	makeToMask := func(op, xor ssa.Op) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			return s.newValue2(op, types.TypeVec128, args[0], s.newValue2(xor, n.Type(), args[0], args[0]))
		}
	}

	makeSimdOp1Imm8 := func(op ssa.Op, immLimit uint64) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			t := n.Type()
			if args[1].Op == ssa.OpConst8 && uint64(args[1].AuxInt) < immLimit {
				return s.newValue1I(op, t, args[1].AuxInt, args[0])
			}
			return immJumpTableN(s, args[1], n, immLimit, func(sNew *state, idx int) {
				// Encode as int8 due to requirement of AuxInt, check its comment for details.
				s.vars[n] = sNew.newValue1I(op, t, int64(int8(idx)), args[0])
			})
		}
	}

	makeSimdOp2Imm8 := func(op ssa.Op, immLimit uint64) func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
		return func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value {
			t := types.TypeVec128
			if args[1].Op == ssa.OpConst8 && uint64(args[1].AuxInt) < immLimit {
				return s.newValue2I(op, t, args[1].AuxInt, args[0], args[2])
			}
			return immJumpTableN(s, args[1], n, immLimit, func(sNew *state, idx int) {
				// Encode as int8 due to requirement of AuxInt, check its comment for details.
				s.vars[n] = sNew.newValue2I(op, t, int64(int8(idx)), args[0], args[2])
			})
		}
	}

	addWasmSIMD := func(pkg, fn string, builder func(s *state, n *ir.CallExpr, args []*ssa.Value) *ssa.Value) {
		intrinsics.add(sys.ArchWasm, pkg, fn, builder)
	}

`)

	for _, op := range wasmOps {

		typeName := op.RcvrType()
		funcName := op.Method()

		// We want <Type>.<Method>
		// The key in intrinsics map is pkg, name.
		// For methods, name is "Type.Method".

		fullname := typeName + "." + funcName

		g := op.SsaGenOp()
		if g == "" || op.Flag(IsSplat) {
			continue
		}
		genOp := "ssa.Op" + g

		if op.ImmRange() > 0 {
			switch op.ArgCount() {
			case 1:
				fmt.Fprintf(f, "\taddWasmSIMD(\"%s\", \"%s\", makeSimdOp1Imm8(%s, %d))\n", pkg, fullname, genOp, op.ImmRange())
			case 2:
				fmt.Fprintf(f, "\taddWasmSIMD(\"%s\", \"%s\", makeSimdOp2Imm8(%s, %d))\n", pkg, fullname, genOp, op.ImmRange())
			default:
				panic("unexpected wasm simd intrinsic " + op.String())
			}
		} else {
			switch op.ArgCount() {
			case 1:
				fmt.Fprintf(f, "\taddWasmSIMD(\"%s\", \"%s\", makeSimdOp1(%s))\n", pkg, fullname, genOp)
			case 2:
				fmt.Fprintf(f, "\taddWasmSIMD(\"%s\", \"%s\", makeSimdOp2(%s))\n", pkg, fullname, genOp)
			case 3:
				fmt.Fprintf(f, "\taddWasmSIMD(\"%s\", \"%s\",makeSimdOp3(%s))\n", pkg, fullname, genOp)
			default:
				panic("unexpected wasm simd intrinsic " + op.String())
			}
		}

	}

	for _, t := range signed {
		// Conversions to/from mask types
		// func (x Int8x16) ToMask() Mask8x16 -> x.Ne(x xor x)
		// func (x Mask8x16) ToInt8x16() Int8x16 -> AsInt8x16 (just a pun, masks are negative)
		fmt.Fprintf(f, "\taddWasmSIMD(\"%s\", \"%s\", makeToMask(%s, %s))\n", pkg, t.Name+".ToMask", "ssa.Op"+t.Methods["ne"].SsaGenOp(), "ssa.Op"+t.Methods["xor"].SsaGenOp())
		fmt.Fprintf(f, "\taddWasmSIMD(\"%s\", \"%s\", makeAsOp())\n", pkg, t.MaskFor().Name+".To"+t.Name)
	}

	// load and store intrinsics
	for _, t := range allTypes {
		u := t
		if t.Unsigned {
			u = t.IntShaped
		}
		fmt.Fprintf(f, "\taddWasmSIMD(\"%s\", \"%s\", simdLoad())\n", pkg, "Load"+t.Name+"Array")
		fmt.Fprintf(f, "\taddWasmSIMD(\"%s\", \"%s\", simdBroadcast(ssa.OpBroadcast%s))\n", pkg, "Broadcast"+t.Name, u.Name)
		fmt.Fprintf(f, "\taddWasmSIMD(\"%s\", \"%s\", simdStore())\n", pkg, t.Name+".StoreArray")
	}

	// As conversions
	forAllAsConversions(func(from, to *simdType) {
		fullname := from.Name + ".As" + to.Name
		fmt.Fprintf(f, "\taddWasmSIMD(\"%s\", \"%s\",makeAsOp())\n", pkg, fullname)
	})

	var WasmTypeDotMethodIntrinsic = templateOf("wasm bit pun intrinsic", `addWasmSIMD("`+pkg+`", "{{.TypeDotMethod}}", makeAsOp())
	`)

	// Factored bitwise reinterpretation methods
	// these produces a much smaller API
	fromUnsignedToFloats(func(from, to *simdType) {
		sgutil.Conversion(from, to).ExecuteIntrinsicTemplateOfTypeDotMethod(f, WasmTypeDotMethodIntrinsic)
	})

	fromUnsignedToInts(func(from, to *simdType) {
		sgutil.Conversion(from, to).ExecuteIntrinsicTemplateOfTypeDotMethod(f, WasmTypeDotMethodIntrinsic)
	})

	forAllReshape(func(from, to *simdType) {
		sgutil.Conversion(from, to).ExecuteIntrinsicTemplateOfTypeDotMethod(f, WasmTypeDotMethodIntrinsic)
	})

	fmt.Fprintln(f, "}")
	return
}
