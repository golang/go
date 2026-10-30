// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"go/ast"
	"go/parser"
	"go/token"
	"slices"
	"strconv"
)

// BindingType describes a C or Go type declared in a //cgo:binding file
// for C ABI classification and Go wrapper generation.
type BindingType struct {
	Name  string // Mangled C type name (e.g. "_Ctype_foo") or Go helper type name
	Def   string // Underlying Go type expression (e.g. "int32", "struct { a float32 }"); empty if opaque
	Size  int64  // Size in bytes on the target architecture
	Align int64  // Alignment in bytes on the target architecture
}

// BindingFunc describes a bound C function declared with //cgo:binding.
type BindingFunc struct {
	Name   string   // Internal cgo function identifier (used in _Cfunc_<Name>, _cgo_arg_<Name>, and _cgo_trampoline_<Name>)
	Params []string // Declared parameter types (Go type expressions, e.g. "int32", "*byte", "_Ctype_S")
	Ret    string   // Return type expression, or "" for void
}

// ParamLoc describes where a function parameter is passed in the C ABI.
type ParamLoc struct {
	GoOffset int64    // Byte offset in Go _cgo_arg_<Name> struct
	Size     int64    // Parameter size in bytes
	IsFloat  bool     // Floating-point parameter
	Regs     []string // Register(s) if passed in register(s)
	StackOff int64    // Byte offset in C stack frame when passed on the stack (len(Regs) == 0)
}

// RetLoc describes how a function return value is returned in the C ABI.
type RetLoc struct {
	GoOffset int64    // Byte offset in Go _cgo_arg_<Name> struct
	Size     int64    // Return value size in bytes
	IsFloat  bool     // Floating-point return
	Indirect bool     // True if caller passes a hidden sret return buffer pointer
	Regs     []string // Register(s) holding the return value upon C return (e.g. AX, X0)
}

// FuncLayout describes the complete Go argument struct and C ABI layout for a call.
type FuncLayout struct {
	Params        []ParamLoc
	Ret           RetLoc
	StackSize     int64 // C stack frame size in bytes (aligned to ABIConfig.StackAlign)
	FloatRegCount int   // Number of floating-point argument registers allocated (for SysV AMD64 AX)
}

// abiState tracks register and stack allocation cursors while computing a FuncLayout.
type abiState struct {
	intIdx   int
	floatIdx int
	stackOff int64
}

// leafField represents a single scalar leaf within a flattened composite type.
type leafField struct {
	offset  int64
	size    int64
	align   int64
	isFloat bool
}

// typeInfo holds the byte size, alignment, ABI classification, and flattened
// scalar leaves of a type.
type typeInfo struct {
	size      int64
	align     int64
	isFloat   bool
	signed    bool
	composite bool
	fields    []leafField
	fieldsOK  bool
}

// ABIConfig defines the C calling convention rules
// for a target OS/architecture. OS, PtrSize, CLongSize, StackAlign, and AssignParam must always be set.
type ABIConfig struct {
	Arch       string
	OS         string
	PtrSize    int64
	CLongSize  int64 // Byte size of C long and unsigned long
	StackAlign int64

	IntRegs      []string
	FloatRegs    []string
	IntRetRegs   []string
	FloatRetRegs []string

	MaxRegStructSize int64
	IndirectRetReg   string // Register holding hidden sret buffer pointer
	IndirectRetInArg bool   // Hidden sret pointer consumes IntRegs[0]

	// AssignParam allocates a single parameter in registers or on the C stack.
	AssignParam func(cfg *ABIConfig, fn BindingFunc, idx int, info typeInfo, st *abiState) ParamLoc

	// Optional platform-specific ABI hooks. Returning handled=false falls back
	// to defaultAssignRet.
	AssignRet func(cfg *ABIConfig, fn BindingFunc, info typeInfo, st *abiState) (RetLoc, bool)
}

// alignUp rounds off up to the next multiple of power-of-two align.
func alignUp(off, align int64) int64 {
	if align <= 1 {
		return off
	}
	return (off + align - 1) &^ (align - 1)
}

// isHostLayoutExpr reports whether expr is the zero-sized structs.HostLayout
// marker embedded in Go struct definitions for C types.
func isHostLayoutExpr(expr ast.Expr) bool {
	sel, ok := expr.(*ast.SelectorExpr)
	if !ok {
		return false
	}
	id, ok := sel.X.(*ast.Ident)
	return ok && id.Name == "structs" && sel.Sel.Name == "HostLayout"
}

// ptrTypeInfo returns the typeInfo for a pointer-sized scalar on cfg.
func (cfg *ABIConfig) ptrTypeInfo() typeInfo {
	return typeInfo{
		size:     cfg.PtrSize,
		align:    cfg.PtrSize,
		fields:   []leafField{{size: cfg.PtrSize, align: cfg.PtrSize}},
		fieldsOK: true,
	}
}

// inspectType summarizes the ABI layout properties (size, alignment, float,
// signedness, composite status, and flattened scalar leaves) of Go/cgo type
// expression t under cfg.
func (cfg *ABIConfig) inspectType(t string, types map[string]BindingType) typeInfo {
	if cfg.OS == "" {
		panic("ABIConfig.OS must be set")
	}
	expr, err := parser.ParseExpr(t)
	if err != nil {
		return typeInfo{size: cfg.PtrSize, align: cfg.PtrSize}
	}
	return cfg.inspectExpr(expr, types, nil)
}

// inspectExpr walks a parsed Go type AST expression to compute its byte size,
// alignment, aggregate classification, and flattened scalar leaf fields.
func (cfg *ABIConfig) inspectExpr(expr ast.Expr, types map[string]BindingType, seen map[string]bool) typeInfo {
	ptrSize := cfg.PtrSize
	if isHostLayoutExpr(expr) {
		return typeInfo{align: 1, fieldsOK: true}
	}
	switch e := expr.(type) {
	case *ast.ParenExpr:
		return cfg.inspectExpr(e.X, types, seen)
	case *ast.StarExpr:
		return cfg.ptrTypeInfo()
	case *ast.ArrayType:
		if e.Len == nil {
			// A nil Len represents a Go slice ([]T), whose runtime layout is a
			// 3-word slice header (Data, Len, Cap).
			return typeInfo{size: ptrSize * 3, align: ptrSize}
		}
		// Binding parsing evaluates constant array lengths (such as named constants)
		// to integer literals before calling inspectType.
		lit, ok := e.Len.(*ast.BasicLit)
		if !ok || lit.Kind != token.INT {
			panic("non-integer-literal array length in ABI type")
		}
		n, err := strconv.ParseInt(lit.Value, 0, 64)
		if err != nil || n < 0 {
			panic("invalid array length in ABI type: " + lit.Value)
		}
		ed := cfg.inspectExpr(e.Elt, types, seen)
		if n == 0 {
			return typeInfo{align: ed.align, composite: true, fieldsOK: true}
		}
		if !ed.fieldsOK {
			return typeInfo{size: ed.size * n, align: ed.align, composite: true}
		}
		fields := make([]leafField, 0, int64(len(ed.fields))*n)
		for i := range n {
			delta := i * ed.size
			for _, f := range ed.fields {
				f.offset += delta
				fields = append(fields, f)
			}
		}
		return typeInfo{
			size:      ed.size * n,
			align:     ed.align,
			composite: true,
			fields:    fields,
			fieldsOK:  true,
		}
	case *ast.StructType:
		var off int64
		maxAlign := int64(1)
		var fields []leafField
		fieldsOK := true
		for _, fld := range e.Fields.List {
			if isHostLayoutExpr(fld.Type) {
				continue
			}
			fd := cfg.inspectExpr(fld.Type, types, seen)
			maxAlign = max(maxAlign, fd.align)
			if !fd.fieldsOK {
				fieldsOK = false
				fields = nil
			}
			for range max(1, len(fld.Names)) {
				off = alignUp(off, fd.align)
				if fieldsOK {
					for _, f := range fd.fields {
						f.offset += off
						fields = append(fields, f)
					}
				}
				off += fd.size
			}
		}
		return typeInfo{
			size:      alignUp(off, maxAlign),
			align:     maxAlign,
			composite: true,
			fields:    fields,
			fieldsOK:  fieldsOK,
		}
	case *ast.SelectorExpr:
		if e.Sel.Name == "Incomplete" {
			return typeInfo{align: 1, fieldsOK: true}
		}
		if id, ok := e.X.(*ast.Ident); ok && id.Name == "C" {
			return cfg.inspectTypeName("_Ctype_"+e.Sel.Name, types, seen)
		}
		return cfg.ptrTypeInfo()
	case *ast.Ident:
		return cfg.inspectTypeName(e.Name, types, seen)
	}
	return typeInfo{size: ptrSize, align: ptrSize}
}

// scalarSpec describes the size and classification of a built-in Go or C scalar type.
type scalarSpec struct {
	size    int64 // Fixed byte size, or 0 when ptr or cLong is set
	ptr     bool  // Size equals ABIConfig.PtrSize
	cLong   bool  // Size equals ABIConfig.CLongSize
	signed  bool
	isFloat bool
}

var scalarTypes = map[string]scalarSpec{
	"uintptr":          {ptr: true},
	"uint":             {ptr: true},
	"_Ctype_size_t":    {ptr: true},
	"int":              {ptr: true, signed: true},
	"_Ctype_ssize_t":   {ptr: true, signed: true},
	"_Ctype_ptrdiff_t": {ptr: true, signed: true},
	"bool":             {size: 1},
	"uint8":            {size: 1},
	"byte":             {size: 1},
	"_Ctype_uchar":     {size: 1},
	"_Ctype__Bool":     {size: 1},
	"int8":             {size: 1, signed: true},
	"_Ctype_char":      {size: 1, signed: true},
	"_Ctype_schar":     {size: 1, signed: true},
	"uint16":           {size: 2},
	"_Ctype_ushort":    {size: 2},
	"int16":            {size: 2, signed: true},
	"_Ctype_short":     {size: 2, signed: true},
	"uint32":           {size: 4},
	"_Ctype_uint":      {size: 4},
	"int32":            {size: 4, signed: true},
	"rune":             {size: 4, signed: true},
	"_Ctype_int":       {size: 4, signed: true},
	"_Ctype_ulong":     {cLong: true},
	"_Ctype_long":      {cLong: true, signed: true},
	"uint64":           {size: 8},
	"_Ctype_ulonglong": {size: 8},
	"int64":            {size: 8, signed: true},
	"_Ctype_longlong":  {size: 8, signed: true},
	"float32":          {size: 4, isFloat: true},
	"_Ctype_float":     {size: 4, isFloat: true},
	"float64":          {size: 8, isFloat: true},
	"_Ctype_double":    {size: 8, isFloat: true},
}

// inspectTypeName resolves a Go or mangled C type identifier into its ABI size,
// alignment, scalar/composite classification, and flattened leaves while
// rejecting recursive type cycles.
func (cfg *ABIConfig) inspectTypeName(name string, types map[string]BindingType, seen map[string]bool) typeInfo {
	if s, ok := scalarTypes[name]; ok {
		sz := s.size
		switch {
		case s.ptr:
			sz = cfg.PtrSize
		case s.cLong:
			if cfg.CLongSize <= 0 {
				panic("ABIConfig.CLongSize must be set")
			}
			sz = cfg.CLongSize
		}
		d := typeInfo{size: sz, align: sz, signed: s.signed, isFloat: s.isFloat}
		if sz > 0 {
			d.fields = []leafField{{size: d.size, align: d.align, isFloat: d.isFloat}}
			d.fieldsOK = true
		}
		return d
	}

	bt, ok := types[name]
	if !ok {
		return cfg.ptrTypeInfo()
	}
	if bt.Def == "" {
		return typeInfo{size: bt.Size, align: bt.Align, composite: true}
	}
	if seen[name] {
		return typeInfo{size: bt.Size, align: bt.Align}
	}
	expr, err := parser.ParseExpr(bt.Def)
	if err != nil {
		return typeInfo{size: bt.Size, align: bt.Align}
	}
	if seen == nil {
		seen = make(map[string]bool)
	}
	seen[name] = true
	d := cfg.inspectExpr(expr, types, seen)
	delete(seen, name)
	if d.composite && (bt.Size > 0 || bt.Align > 0) {
		d.size, d.align = bt.Size, bt.Align
	}
	return d
}

// layout computes the Go argument struct layout and C calling convention
// register and stack assignments for fn.
func (cfg *ABIConfig) layout(fn BindingFunc, types map[string]BindingType) FuncLayout {
	if cfg.OS == "" {
		panic("ABIConfig.OS must be set")
	}
	var st abiState
	ret, retAlign := cfg.assignRet(fn, &st, types)
	params, goOff := cfg.assignParams(fn, &st, types)
	if fn.Ret != "" {
		ret.GoOffset = alignUp(goOff, retAlign)
	}

	return FuncLayout{
		Params:        params,
		Ret:           ret,
		StackSize:     alignUp(st.stackOff, cfg.StackAlign),
		FloatRegCount: st.floatIdx,
	}
}

// assignRet determines how fn's return value is returned in the C ABI and
// returns its alignment in the Go argument struct, invoking cfg.AssignRet
// first and falling back to defaultAssignRet.
func (cfg *ABIConfig) assignRet(fn BindingFunc, st *abiState, types map[string]BindingType) (RetLoc, int64) {
	if fn.Ret == "" {
		return RetLoc{}, 1
	}
	info := cfg.inspectType(fn.Ret, types)
	if cfg.AssignRet != nil {
		if loc, handled := cfg.AssignRet(cfg, fn, info, st); handled {
			return loc, info.align
		}
	}
	return cfg.defaultAssignRet(info, st), info.align
}

// defaultAssignRet implements standard return-value placement when
// cfg.AssignRet is nil (for ABIs without special composite-return rules) or
// returns handled=false after handling ABI-specific composite cases (such as
// scalar returns on amd64, or scalar and non-HFA struct returns on arm64).
// Composites exceeding MaxRegStructSize use a hidden sret buffer pointer
// (consuming IntRegs[0] when IndirectRetInArg is set), floats use
// FloatRetRegs[0], and integers or small composites pack into IntRetRegs.
func (cfg *ABIConfig) defaultAssignRet(info typeInfo, st *abiState) RetLoc {
	ptrSize := cfg.PtrSize
	if !info.isFloat && info.size > cfg.MaxRegStructSize && (info.composite || info.size > ptrSize*int64(len(cfg.IntRetRegs))) {
		if cfg.IndirectRetInArg {
			st.intIdx = 1
		}
		return RetLoc{Size: info.size, Indirect: true}
	}
	if info.isFloat {
		return RetLoc{Size: info.size, Regs: []string{cfg.FloatRetRegs[0]}, IsFloat: true}
	}
	if info.size > 0 {
		numRegs := (info.size + ptrSize - 1) / ptrSize
		return RetLoc{Size: info.size, Regs: slices.Clone(cfg.IntRetRegs[:numRegs])}
	}
	return RetLoc{}
}

// assignParams allocates C ABI registers or stack offsets and Go argument
// struct offsets for each parameter of fn in left-to-right order, returning
// the parameter locations and the Go struct byte offset after the last parameter.
func (cfg *ABIConfig) assignParams(fn BindingFunc, st *abiState, types map[string]BindingType) ([]ParamLoc, int64) {
	if cfg.AssignParam == nil {
		panic("ABIConfig.AssignParam must be set")
	}
	var goOff int64
	paramTypes := fn.Params
	params := make([]ParamLoc, len(paramTypes))
	for i, pt := range paramTypes {
		info := cfg.inspectType(pt, types)
		loc := cfg.AssignParam(cfg, fn, i, info, st)
		goOff = alignUp(goOff, info.align)
		loc.GoOffset = goOff
		goOff += loc.Size
		params[i] = loc
	}
	return params, goOff
}

// assignFloatParam allocates a floating-point scalar parameter to the next
// available register in cfg.FloatRegs, or to an aligned C stack slot.
func (cfg *ABIConfig) assignFloatParam(info typeInfo, st *abiState) ParamLoc {
	if st.floatIdx < len(cfg.FloatRegs) {
		reg := cfg.FloatRegs[st.floatIdx]
		st.floatIdx++
		return ParamLoc{Size: info.size, IsFloat: true, Regs: []string{reg}}
	}
	return cfg.assignStackParam(info.size, true, st)
}

// assignIntParam allocates an integer scalar or small composite parameter
// across consecutive registers in cfg.IntRegs, or to an aligned C stack slot.
func (cfg *ABIConfig) assignIntParam(info typeInfo, st *abiState) ParamLoc {
	if info.size == 0 {
		return ParamLoc{}
	}
	numRegs := int((info.size + cfg.PtrSize - 1) / cfg.PtrSize)
	if st.intIdx+numRegs <= len(cfg.IntRegs) {
		regs := slices.Clone(cfg.IntRegs[st.intIdx : st.intIdx+numRegs])
		st.intIdx += numRegs
		return ParamLoc{Size: info.size, Regs: regs}
	}
	return cfg.assignStackParam(info.size, false, st)
}

// assignStackParam allocates size bytes on the C argument stack aligned to PtrSize.
func (cfg *ABIConfig) assignStackParam(size int64, isFloat bool, st *abiState) ParamLoc {
	off := st.stackOff
	st.stackOff += alignUp(size, cfg.PtrSize)
	return ParamLoc{Size: size, IsFloat: isFloat, StackOff: off}
}

// System V AMD64 C calling convention (used on Linux, macOS/iOS, Android,
// Solaris/Illumos, and the BSDs).
//
// Specification: "System V Application Binary Interface: AMD64 Architecture
// Processor Supplement", Section 3.2 ("Function Calling Sequence"),
// https://gitlab.com/x86-psABIs/x86-64-ABI.
//
// Summary:
//   - Integer and pointer arguments use DI, SI, DX, CX, R8, R9; floating-point
//     arguments use X0..X7 (Section 3.2.3, Figure 3.4).
//   - Aggregates up to 16 bytes with naturally aligned fields are partitioned
//     into one or two 8-byte "eightbytes", each classified as SSE (all
//     float/double leaves) or INTEGER (any integer/pointer leaf); all
//     eightbytes of an aggregate must fit in remaining registers or the entire
//     aggregate spills to an 8-byte-aligned stack slot without consuming
//     registers (Section 3.2.3, "Aggregate and Union Arguments").
//   - Aggregates larger than 16 bytes or containing unaligned fields have class
//     MEMORY: passed on the stack, and returned via a hidden sret buffer
//     pointer in DI (consuming the first integer argument register) echoed in AX.
//   - Scalar and register-classified aggregate results use AX, DX and/or X0, X1
//     (Section 3.2.3, "Returning of Values").
//   - The stack frame is 16-byte aligned before CALL, and AL holds an upper
//     bound (0..8) on the number of XMM registers used in variadic calls
//     (Sections 3.2.2 and 3.5.7).

// sysVEightbyte describes the register class and transfer width of one 8-byte
// slice of a System V AMD64 aggregate (SysV AMD64 psABI Section 3.2.3).
type sysVEightbyte struct {
	isFloat bool
	size    int64
}

// classifySysVStruct classifies a composite type of up to 16 bytes into one or
// two System V AMD64 eightbytes (SSE vs INTEGER) per SysV AMD64 psABI
// Section 3.2.3 ("Parameter Passing"). It returns ok=false if the aggregate
// exceeds 16 bytes or contains unaligned fields, requiring MEMORY placement.
func classifySysVStruct(info typeInfo) ([]sysVEightbyte, bool) {
	size := info.size
	if size <= 0 || size > 16 {
		return nil, false
	}
	var hasFloat, hasInt [2]bool
	var maxEnd [2]int64
	if info.fieldsOK {
		for _, f := range info.fields {
			if f.size <= 0 {
				continue
			}
			eb := f.offset / 8
			if (f.align > 1 && f.offset%f.align != 0) || eb >= 2 || (f.offset+f.size-1)/8 != eb {
				return nil, false
			}
			maxEnd[eb] = max(maxEnd[eb], f.offset%8+f.size)
			if f.isFloat {
				hasFloat[eb] = true
				continue
			}
			hasInt[eb] = true
		}
	}

	ebCount := (size + 7) / 8
	out := make([]sysVEightbyte, ebCount)
	for k := range ebCount {
		if hasFloat[k] && !hasInt[k] {
			sz := int64(8)
			if maxEnd[k] <= 4 {
				sz = 4
			}
			out[k] = sysVEightbyte{isFloat: true, size: sz}
			continue
		}
		out[k] = sysVEightbyte{size: min(8, size-k*8)}
	}
	return out, true
}

// assignSysVRegs allocates integer and SSE registers for ebs from intRegs and
// floatRegs starting at *intIdx and *floatIdx, advancing the cursors only when
// all eightbytes fit in registers (SysV AMD64 psABI Section 3.2.3).
func assignSysVRegs(ebs []sysVEightbyte, intRegs, floatRegs []string, intIdx, floatIdx *int) (regs []string, allFloat, ok bool) {
	var needInt, needFloat int
	for _, eb := range ebs {
		if eb.isFloat {
			needFloat++
			continue
		}
		needInt++
	}
	if *intIdx+needInt > len(intRegs) || *floatIdx+needFloat > len(floatRegs) {
		return nil, false, false
	}
	regs = make([]string, len(ebs))
	allFloat = true
	for k, eb := range ebs {
		if eb.isFloat {
			regs[k] = floatRegs[*floatIdx]
			*floatIdx++
			continue
		}
		regs[k] = intRegs[*intIdx]
		*intIdx++
		allFloat = false
	}
	return regs, allFloat, true
}

// amd64AssignRet assigns System V AMD64 composite return values to AX/DX and
// X0/X1 according to their eightbyte classes, or to a hidden sret pointer in DI
// (SysV AMD64 psABI Section 3.2.3, "Returning of Values").
func amd64AssignRet(cfg *ABIConfig, _ BindingFunc, info typeInfo, st *abiState) (RetLoc, bool) {
	if !info.composite {
		return RetLoc{}, false
	}
	if ebs, ok := classifySysVStruct(info); ok {
		var retInt, retFloat int
		if regs, allFloat, fit := assignSysVRegs(ebs, cfg.IntRetRegs, cfg.FloatRetRegs, &retInt, &retFloat); fit {
			return RetLoc{Size: info.size, Regs: regs, IsFloat: allFloat}, true
		}
	}
	st.intIdx = 1
	return RetLoc{Size: info.size, Indirect: true}, true
}

// amd64AssignParam assigns System V AMD64 parameters per SysV AMD64 psABI
// Section 3.2.3, placing aggregates all-or-nothing in registers when enough
// integer and SSE registers remain and spilling to 8-byte-aligned stack slots
// otherwise.
func amd64AssignParam(cfg *ABIConfig, _ BindingFunc, _ int, info typeInfo, st *abiState) ParamLoc {
	if info.composite {
		if ebs, ok := classifySysVStruct(info); ok {
			if regs, allFloat, fit := assignSysVRegs(ebs, cfg.IntRegs, cfg.FloatRegs, &st.intIdx, &st.floatIdx); fit {
				return ParamLoc{Size: info.size, Regs: regs, IsFloat: allFloat}
			}
		}
		return cfg.assignStackParam(info.size, false, st)
	}
	if info.isFloat {
		return cfg.assignFloatParam(info, st)
	}
	return cfg.assignIntParam(info, st)
}

var amd64ABI = &ABIConfig{
	Arch:             "amd64",
	PtrSize:          8,
	CLongSize:        8,
	StackAlign:       16,
	IntRegs:          []string{"DI", "SI", "DX", "CX", "R8", "R9"},
	FloatRegs:        []string{"X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7"},
	IntRetRegs:       []string{"AX", "DX"},
	FloatRetRegs:     []string{"X0", "X1"},
	MaxRegStructSize: 16,
	IndirectRetReg:   "DI",
	IndirectRetInArg: true,
	AssignParam:      amd64AssignParam,
	AssignRet:        amd64AssignRet,
}

// getABI returns the ABIConfig for the given targetOS/targetArch.
func getABI(targetOS, targetArch string) *ABIConfig {
	var cfg ABIConfig
	switch targetArch {
	case "amd64":
		cfg = *amd64ABI
	default:
		panic("unsupported arch: " + targetArch)
	}
	cfg.OS = targetOS
	return &cfg
}
