// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"io"
	"slices"
	"strconv"
)

// BindingType describes a C or Go type used by cgo binding declarations
// for C ABI classification and Go wrapper generation.
type BindingType struct {
	Name  string // Mangled C type name (e.g. "_Ctype_foo") or Go helper type name
	Def   string // Underlying Go type expression (e.g. "int32", "struct { a float32 }"); empty if opaque
	Size  int64  // Size in bytes on the target architecture
	Align int64  // Alignment in bytes on the target architecture
}

// BindingFunc describes a bound C function declared with //cgo:func.
type BindingFunc struct {
	// Identifier and symbol names:
	Name   string // Internal cgo function identifier (used in _Cfunc_<Name>, _cgo_arg_<Name>, and _cgo_trampoline_<Name>)
	GoName string // Go function named by the //cgo:func pragma (used by binding.go)
	CName  string // Target C symbol name
	BindC  bool   // The pragma wrote the C name as C.<CName>, so Go code may also call the function as C.<CName> (used by binding.go)

	// Signature (Go type expressions, e.g. "int32", "*byte", "_Ctype_S"):
	Params []string // Declared parameter types
	Ret    string   // Return type expression, or "" for void

	// Go wrapper and import metadata (used by binding.go):
	Library string // Shared library or DLL name for dynamic import directives; empty for libc and static
	LibC    bool   // The symbol comes from the platform C runtime (libc)
	Static  bool   // The symbol is resolved at link time from a .syso or .a (static)
}

// ParamLoc describes where a function parameter is passed in the C ABI.
type ParamLoc struct {
	GoOffset  int64    // Byte offset in Go _cgo_arg_<Name> struct
	Size      int64    // Parameter size in bytes
	IsFloat   bool     // Floating-point parameter
	Regs      []string // Register(s) if passed in register(s)
	StackOff  int64    // Byte offset in C stack frame when passed on the stack (len(Regs) == 0)
	PassByPtr bool     // True if passed by hidden reference (pointer in Regs[0] or on stack)
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

// hasStackArgs reports whether any argument is passed on the C stack.
func (lay FuncLayout) hasStackArgs() bool {
	for _, p := range lay.Params {
		if p.Size > 0 && len(p.Regs) == 0 {
			return true
		}
	}
	return false
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

// regSlot describes a single register piece of a parameter or return value.
type regSlot struct {
	reg     string
	offset  int64 // Byte offset relative to ParamLoc.GoOffset or RetLoc.GoOffset
	size    int64 // Byte size of this piece (1..8)
	isFloat bool
	signed  bool
}

// asmOps defines the assembly mnemonics and format templates for
// an architecture's load, store, arithmetic, and stack-pointer operations.
// All fields must be populated.
type asmOps struct {
	mov    string // Pointer-sized integer load/store/reg-move (e.g. "MOVQ", "MOVD")
	loadB  string // Signed 1-byte load
	loadBU string // Unsigned 1-byte load
	loadH  string // Signed 2-byte load
	loadHU string // Unsigned 2-byte load
	loadW  string // Signed 4-byte load
	loadWU string // Unsigned 4-byte load
	storeB string // 1-byte store
	storeH string // 2-byte store
	storeW string // 4-byte store
	fmovS  string // 32-bit float load/store
	fmovD  string // 64-bit float load/store
	shl    string // Format for in-place left shift; args: (bits, reg)
	shr    string // Format for in-place logical right shift; args: (bits, reg)
	or     string // Format for bitwise OR dst |= src; args: (src, dst)
	addr   string // Format for effective address reg = base+off; args: (base, off, reg)
	subSP    string // Format for decrementing SP; args: (size, spReg)
	addSP    string // Format for incrementing SP; args: (size, spReg)
	useAdjSP bool   // True if SP adjustments must use ADJSP so obj tracks pcsp without FuncFlagSPWrite
}

// ABIConfig defines the C calling convention rules and assembly primitives
// for a target OS/architecture. OS, PtrSize, CLongSize, StackAlign, and AssignParam must always be set.
type ABIConfig struct {
	Arch        string
	OS          string
	PtrSize     int64
	CLongSize   int64 // Byte size of C long and unsigned long
	StackAlign  int64
	ShadowSpace int64 // Bytes reserved at the bottom of the C call frame by the ABI (e.g. 32-byte home space on Windows x64)

	IntRegs      []string
	FloatRegs    []string
	IntRetRegs   []string
	FloatRetRegs []string

	MaxRegStructSize int64
	IndirectRetReg   string // Register holding hidden sret buffer pointer
	IndirectRetInArg bool   // Hidden sret pointer consumes IntRegs[0]
	PackStackArgs    bool   // Pack narrow scalar and HFA stack args at natural alignment (Apple ARM64)

	// AssignParam allocates a single parameter in registers or on the C stack.
	AssignParam func(cfg *ABIConfig, fn BindingFunc, idx int, info typeInfo, st *abiState) ParamLoc

	// AssignRet determines how the return value is returned in the C ABI.
	// ABIs without special return rules use defaultAssignRet.
	AssignRet func(cfg *ABIConfig, fn BindingFunc, info typeInfo, st *abiState) RetLoc

	// Optional platform-specific ABI hook. Returning handled=false falls back
	// to default GPR chunking.
	CompositeRegSlots func(cfg *ABIConfig, info typeInfo, isFloat bool, regs []string) ([]regSlot, bool)

	// Trampoline assembly generation parameters:
	SPReg          string // Hardware stack pointer register in asm (e.g. "SP", "RSP", "R1", "X2")
	CalleeSavedReg string // C callee-saved register holding the Go _cgo_arg struct pointer across C calls
	TmpReg         string // Primary caller-saved scratch register for stack/memory copies and address math
	TmpReg2        string // Secondary caller-saved scratch register for multi-chunk loads and copies
	LRReserve      int64  // Bytes reserved at 0(SP) for saved LR when !hasStackArgs and ShadowSpace == 0

	asm                asmOps
	TrampolinePrologue func(w io.Writer, lay FuncLayout) (epilogue func(w io.Writer))
	PreCall            func(cfg *ABIConfig, w io.Writer, fn BindingFunc, lay FuncLayout)
	CallSymbol         func(w io.Writer, sym string)
}

// isFloatReg reports whether reg is an architecture floating-point register.
func (cfg *ABIConfig) isFloatReg(reg string) bool {
	return slices.Contains(cfg.FloatRegs, reg) || slices.Contains(cfg.FloatRetRegs, reg)
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
	st := abiState{stackOff: cfg.ShadowSpace}
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

// assignRet determines how fn's return value is returned in the C ABI using
// cfg.AssignRet and returns its alignment in the Go argument struct.
func (cfg *ABIConfig) assignRet(fn BindingFunc, st *abiState, types map[string]BindingType) (RetLoc, int64) {
	if cfg.AssignRet == nil {
		panic("ABIConfig.AssignRet must be set")
	}
	if fn.Ret == "" {
		return RetLoc{}, 1
	}
	info := cfg.inspectType(fn.Ret, types)
	return cfg.AssignRet(cfg, fn, info, st), info.align
}

// defaultAssignRet implements standard return-value placement. It is used
// directly as AssignRet by ABIs without special composite-return rules, and
// called by ABI-specific AssignRet functions for the cases they do not handle
// themselves (such as scalar returns on amd64, or scalar and non-HFA struct
// returns on arm64).
// Composites exceeding MaxRegStructSize use a hidden sret buffer pointer
// (consuming IntRegs[0] when IndirectRetInArg is set), floats use
// FloatRetRegs[0], and integers or small composites pack into IntRetRegs.
func defaultAssignRet(cfg *ABIConfig, _ BindingFunc, info typeInfo, st *abiState) RetLoc {
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
	if cfg.PackStackArgs {
		return cfg.assignPackedStackParam(info.size, info.align, true, st)
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
	// Apple ARM64 packs narrow scalar stack arguments at natural alignment,
	// while non-HFA composites <= 16 bytes still occupy PtrSize-aligned slots.
	if cfg.PackStackArgs && !info.composite {
		return cfg.assignPackedStackParam(info.size, info.align, false, st)
	}
	return cfg.assignStackParam(info.size, false, st)
}

// assignPtrParam passes a composite of size bytes by hidden pointer in the next
// available register in cfg.IntRegs, or in a pointer-sized C stack slot.
func (cfg *ABIConfig) assignPtrParam(size int64, st *abiState) ParamLoc {
	if st.intIdx < len(cfg.IntRegs) {
		reg := cfg.IntRegs[st.intIdx]
		st.intIdx++
		return ParamLoc{Size: size, PassByPtr: true, Regs: []string{reg}}
	}
	off := alignUp(st.stackOff, cfg.PtrSize)
	st.stackOff = off + cfg.PtrSize
	return ParamLoc{Size: size, PassByPtr: true, StackOff: off}
}

// assignPackedStackParam allocates size bytes on the C argument stack at
// natural alignment without rounding the slot up to PtrSize (Apple ARM64).
func (cfg *ABIConfig) assignPackedStackParam(size, align int64, isFloat bool, st *abiState) ParamLoc {
	off := alignUp(st.stackOff, max(align, 1))
	st.stackOff = off + size
	return ParamLoc{Size: size, IsFloat: isFloat, StackOff: off}
}

// assignStackParam allocates size bytes on the C argument stack aligned to PtrSize.
func (cfg *ABIConfig) assignStackParam(size int64, isFloat bool, st *abiState) ParamLoc {
	off := alignUp(st.stackOff, cfg.PtrSize)
	st.stackOff = off + alignUp(size, cfg.PtrSize)
	return ParamLoc{Size: size, IsFloat: isFloat, StackOff: off}
}

// decomposePow2 splits a byte size (1..8) into descending power-of-two chunks
// (8, 4, 2, 1) so odd-sized struct pieces can be loaded or stored without
// reading or writing past the bounds of the Go argument struct.
func decomposePow2(size int64) []int64 {
	var chunks []int64
	for rem := size; rem > 0; {
		switch {
		case rem >= 8:
			chunks = append(chunks, 8)
			rem -= 8
		case rem >= 4:
			chunks = append(chunks, 4)
			rem -= 4
		case rem >= 2:
			chunks = append(chunks, 2)
			rem -= 2
		default:
			chunks = append(chunks, 1)
			rem -= 1
		}
	}
	return chunks
}

// maxPow2Chunk returns the largest power-of-two byte width (1, 2, 4, or 8)
// that fits within both rem and ptrSize.
func maxPow2Chunk(rem, ptrSize int64) int64 {
	if ptrSize >= 8 && rem >= 8 {
		return 8
	}
	if rem >= 4 {
		return 4
	}
	if rem >= 2 {
		return 2
	}
	return 1
}

// emitScalarLoadInt emits a single power-of-two (1, 2, 4, or 8 byte) integer
// load from off(base) into reg, sign- or zero-extending narrow loads.
func (cfg *ABIConfig) emitScalarLoadInt(w io.Writer, base string, off, size int64, signed bool, reg string) {
	a := &cfg.asm
	var op string
	switch {
	case size == 1 && signed:
		op = a.loadB
	case size == 1:
		op = a.loadBU
	case size == 2 && signed:
		op = a.loadH
	case size == 2:
		op = a.loadHU
	case size == 4 && signed:
		op = a.loadW
	case size == 4:
		op = a.loadWU
	default:
		op = a.mov
	}
	fmt.Fprintf(w, "\t%s\t%d(%s), %s\n", op, off, base, reg)
}

// emitScalarStoreInt emits a single power-of-two (1, 2, 4, or 8 byte) integer
// store from reg into off(base).
func (cfg *ABIConfig) emitScalarStoreInt(w io.Writer, reg, base string, off, size int64) {
	a := &cfg.asm
	var op string
	switch size {
	case 1:
		op = a.storeB
	case 2:
		op = a.storeH
	case 4:
		op = a.storeW
	default:
		op = a.mov
	}
	fmt.Fprintf(w, "\t%s\t%s, %d(%s)\n", op, reg, off, base)
}

// emitShiftLeft shifts reg left in place by bits.
func (cfg *ABIConfig) emitShiftLeft(w io.Writer, bits int64, reg string) {
	fmt.Fprintf(w, "\t"+cfg.asm.shl+"\n", bits, reg)
}

// emitShiftRight logically shifts reg right in place by bits.
func (cfg *ABIConfig) emitShiftRight(w io.Writer, bits int64, reg string) {
	fmt.Fprintf(w, "\t"+cfg.asm.shr+"\n", bits, reg)
}

// emitOrReg bitwise-ORs src into dst.
func (cfg *ABIConfig) emitOrReg(w io.Writer, src, dst string) {
	fmt.Fprintf(w, "\t"+cfg.asm.or+"\n", src, dst)
}

// emitLoadInt loads an integer or struct register chunk of 1..8 bytes from
// off(base) into reg, decomposing non-power-of-two sizes into exact sub-loads
// combined with shifts and ORs.
func (cfg *ABIConfig) emitLoadInt(w io.Writer, base string, off, size int64, signed bool, reg string) {
	switch size {
	case 1, 2, 4, 8:
		cfg.emitScalarLoadInt(w, base, off, size, signed, reg)
		return
	}
	chunks := decomposePow2(size)
	scratch := cfg.TmpReg2
	if reg == scratch || base == scratch {
		scratch = cfg.TmpReg
	}
	cfg.emitScalarLoadInt(w, base, off, chunks[0], false, reg)
	curOff := chunks[0]
	for _, ck := range chunks[1:] {
		cfg.emitScalarLoadInt(w, base, off+curOff, ck, false, scratch)
		cfg.emitShiftLeft(w, curOff*8, scratch)
		cfg.emitOrReg(w, scratch, reg)
		curOff += ck
	}
}

// emitStoreInt stores an integer or struct register chunk of 1..8 bytes from
// reg into off(base), decomposing non-power-of-two sizes into exact sub-stores.
func (cfg *ABIConfig) emitStoreInt(w io.Writer, reg, base string, off, size int64) {
	switch size {
	case 1, 2, 4, 8:
		cfg.emitScalarStoreInt(w, reg, base, off, size)
		return
	}
	chunks := decomposePow2(size)
	curOff := int64(0)
	for i, ck := range chunks {
		cfg.emitScalarStoreInt(w, reg, base, off+curOff, ck)
		curOff += ck
		if i+1 < len(chunks) {
			cfg.emitShiftRight(w, ck*8, reg)
		}
	}
}

// emitLoadFloat loads a 4-byte or 8-byte floating-point value from off(base)
// into floating-point register reg.
func (cfg *ABIConfig) emitLoadFloat(w io.Writer, base string, off, size int64, reg string) {
	op := cfg.asm.fmovD
	if size == 4 {
		op = cfg.asm.fmovS
	}
	fmt.Fprintf(w, "\t%s\t%d(%s), %s\n", op, off, base, reg)
}

// emitStoreFloat stores a 4-byte or 8-byte floating-point value from
// floating-point register reg into off(base).
func (cfg *ABIConfig) emitStoreFloat(w io.Writer, reg, base string, off, size int64) {
	op := cfg.asm.fmovD
	if size == 4 {
		op = cfg.asm.fmovS
	}
	fmt.Fprintf(w, "\t%s\t%s, %d(%s)\n", op, reg, off, base)
}

// emitAddr computes the effective address base+off into reg.
func (cfg *ABIConfig) emitAddr(w io.Writer, base string, off int64, reg string) {
	fmt.Fprintf(w, "\t"+cfg.asm.addr+"\n", base, off, reg)
}

// emitMovReg copies pointer-sized register src into dst.
func (cfg *ABIConfig) emitMovReg(w io.Writer, src, dst string) {
	fmt.Fprintf(w, "\t%s\t%s, %s\n", cfg.asm.mov, src, dst)
}

// emitSubSP decrements the architecture stack pointer by size bytes.
func (cfg *ABIConfig) emitSubSP(w io.Writer, size int64) {
	if cfg.asm.useAdjSP {
		fmt.Fprintf(w, "\tADJSP\t$%d\n", size)
		return
	}
	fmt.Fprintf(w, "\t"+cfg.asm.subSP+"\n", size, cfg.SPReg)
}

// emitAddSP increments the architecture stack pointer by size bytes.
func (cfg *ABIConfig) emitAddSP(w io.Writer, size int64) {
	if cfg.asm.useAdjSP {
		fmt.Fprintf(w, "\tADJSP\t$-%d\n", size)
		return
	}
	fmt.Fprintf(w, "\t"+cfg.asm.addSP+"\n", size, cfg.SPReg)
}

// emitCallSymbol emits a call to external or runtime symbol sym.
func (cfg *ABIConfig) emitCallSymbol(w io.Writer, sym string) {
	if cfg.CallSymbol != nil {
		cfg.CallSymbol(w, sym)
		return
	}
	fmt.Fprintf(w, "\tCALL\t%s(SB)\n", sym)
}

// emitCopyMem copies size bytes from srcOff(srcBase) to dstOff(dstBase) using
// TmpReg-sized power-of-two loads and stores.
func (cfg *ABIConfig) emitCopyMem(w io.Writer, srcBase string, srcOff int64, dstBase string, dstOff, size int64, tmpReg string) {
	ptrSize := cfg.PtrSize
	for off := int64(0); off < size; {
		chunk := maxPow2Chunk(size-off, ptrSize)
		cfg.emitLoadInt(w, srcBase, srcOff+off, chunk, false, tmpReg)
		cfg.emitStoreInt(w, tmpReg, dstBase, dstOff+off, chunk)
		off += chunk
	}
}

// paramRegSlots decomposes register-passed parameter p into per-register
// load/store descriptors.
func (cfg *ABIConfig) paramRegSlots(fn BindingFunc, idx int, p ParamLoc, types map[string]BindingType) []regSlot {
	if len(p.Regs) == 0 {
		return nil
	}
	if p.PassByPtr {
		return []regSlot{{reg: p.Regs[0], offset: 0, size: cfg.PtrSize, isFloat: false}}
	}
	var pt string
	if paramTypes := fn.Params; idx < len(paramTypes) {
		pt = paramTypes[idx]
	}
	return cfg.valueRegSlots(pt, p.Size, p.IsFloat, p.Regs, types)
}

// retRegSlots decomposes register-returned result ret into per-register
// load/store descriptors.
func (cfg *ABIConfig) retRegSlots(fn BindingFunc, ret RetLoc, types map[string]BindingType) []regSlot {
	if len(ret.Regs) == 0 || ret.Indirect {
		return nil
	}
	return cfg.valueRegSlots(fn.Ret, ret.Size, ret.IsFloat, ret.Regs, types)
}

// valueRegSlots maps a value of typeName (with total size and assigned regs)
// onto individual register pieces, delegating composite register layouts to
// cfg.CompositeRegSlots when present and otherwise slicing into PtrSize chunks.
func (cfg *ABIConfig) valueRegSlots(typeName string, size int64, isFloat bool, regs []string, types map[string]BindingType) []regSlot {
	ptrSize := cfg.PtrSize
	info := cfg.inspectType(typeName, types)
	if info.composite && cfg.CompositeRegSlots != nil {
		if slots, ok := cfg.CompositeRegSlots(cfg, info, isFloat, regs); ok {
			return slots
		}
	}
	slots := make([]regSlot, len(regs))
	for k, r := range regs {
		fl := cfg.isFloatReg(r)
		chunkSz := min(ptrSize, size-int64(k)*ptrSize)
		if len(regs) == 1 {
			chunkSz = size
		}
		slots[k] = regSlot{
			reg:     r,
			offset:  int64(k) * ptrSize,
			size:    chunkSz,
			isFloat: fl,
			signed:  !info.composite && !fl && info.signed,
		}
	}
	return slots
}

// paramTransfer describes a single register or stack move needed to marshal a
// parameter from the Go argument struct into its C ABI location.
type paramTransfer struct {
	offset    int64
	reg       string
	stackOff  int64
	storeSize int64
	load      func(w io.Writer, base string, off int64, dstReg string)
}

// paramTransfers decomposes parameter p into register or C-stack transfers.
func (cfg *ABIConfig) paramTransfers(fn BindingFunc, idx int, p ParamLoc, types map[string]BindingType) []paramTransfer {
	ptrSize := cfg.PtrSize
	if p.PassByPtr {
		if len(p.Regs) > 0 {
			return []paramTransfer{{reg: p.Regs[0], load: cfg.emitAddr}}
		}
		return []paramTransfer{{stackOff: p.StackOff, storeSize: ptrSize, load: cfg.emitAddr}}
	}
	loadInt := func(size int64, signed bool) func(io.Writer, string, int64, string) {
		return func(w io.Writer, base string, off int64, dstReg string) {
			cfg.emitLoadInt(w, base, off, size, signed, dstReg)
		}
	}
	loadFloat := func(size int64) func(io.Writer, string, int64, string) {
		return func(w io.Writer, base string, off int64, dstReg string) {
			cfg.emitLoadFloat(w, base, off, size, dstReg)
		}
	}

	if slots := cfg.paramRegSlots(fn, idx, p, types); len(slots) > 0 {
		ops := make([]paramTransfer, len(slots))
		for k, slot := range slots {
			ld := loadInt(slot.size, slot.signed)
			if slot.isFloat {
				ld = loadFloat(slot.size)
			}
			ops[k] = paramTransfer{offset: slot.offset, reg: slot.reg, load: ld}
		}
		return ops
	}

	var pt string
	if paramTypes := fn.Params; idx < len(paramTypes) {
		pt = paramTypes[idx]
	}
	info := cfg.inspectType(pt, types)
	if !info.composite && (p.Size == 1 || p.Size == 2 || p.Size == 4 || p.Size == 8) && p.Size <= ptrSize {
		storeSize := max(ptrSize, p.Size)
		if cfg.PackStackArgs {
			storeSize = p.Size
		}
		return []paramTransfer{{
			stackOff:  p.StackOff,
			storeSize: storeSize,
			load:      loadInt(p.Size, info.signed),
		}}
	}

	var ops []paramTransfer
	for off := int64(0); off < p.Size; {
		chunk := maxPow2Chunk(p.Size-off, ptrSize)
		ops = append(ops, paramTransfer{
			offset:    off,
			stackOff:  p.StackOff + off,
			storeSize: chunk,
			load:      loadInt(chunk, false),
		})
		off += chunk
	}
	return ops
}

// emitStoreSlot stores a register return piece into off(base).
func (cfg *ABIConfig) emitStoreSlot(w io.Writer, slot regSlot, base string, off int64) {
	if slot.isFloat {
		cfg.emitStoreFloat(w, slot.reg, base, off+slot.offset, slot.size)
		return
	}
	cfg.emitStoreInt(w, slot.reg, base, off+slot.offset, slot.size)
}

// writeTrampoline emits the Go-to-C assembly trampoline for fn.
// Go callers pack parameters into a _cgo_arg_<Name> struct and pass its address
// to runtime.cgocall, which enters a syscall, switches to the OS-allocated g0
// system stack via asmcgocall, and invokes this trampoline using the platform C
// ABI for void fn(void *arg) (passing the struct pointer in IntRegs[0], or
// 0(SP) on 386). The trampoline unpacks arguments into C ABI registers and
// stack slots, calls the target C function, and writes any result back to the
// _cgo_arg_<Name> struct.
func (cfg *ABIConfig) writeTrampoline(w io.Writer, fn BindingFunc, types map[string]BindingType) {
	lay := cfg.layout(fn, types)

	var epilogue func(io.Writer)
	if cfg.TrampolinePrologue != nil {
		fmt.Fprintf(w, "TEXT _cgo%s_trampoline_%s(SB), NOSPLIT|NOFRAME, $0-0\n", cPrefix, fn.Name)
		epilogue = cfg.TrampolinePrologue(w, lay)
	} else {
		fmt.Fprintf(w, "TEXT _cgo%s_trampoline_%s(SB), NOSPLIT, $0-0\n", cPrefix, fn.Name)
		// Hold the _cgo_arg struct pointer in CalleeSavedReg across the C call:
		// the C callee preserves it per the C ABI, and our caller (runtime.asmcgocall)
		// does not keep a live value in CalleeSavedReg, so we need not save/restore it.
		cfg.emitMovReg(w, cfg.IntRegs[0], cfg.CalleeSavedReg)

		if lay.StackSize > 0 {
			cfg.emitSubSP(w, lay.StackSize)
		}
	}

	base := cfg.CalleeSavedReg

	if lay.Ret.Indirect {
		cfg.emitAddr(w, base, lay.Ret.GoOffset, cfg.IndirectRetReg)
	}

	for i, p := range lay.Params {
		for _, op := range cfg.paramTransfers(fn, i, p, types) {
			if op.reg != "" {
				op.load(w, base, p.GoOffset+op.offset, op.reg)
			} else {
				op.load(w, base, p.GoOffset+op.offset, cfg.TmpReg)
				cfg.emitStoreInt(w, cfg.TmpReg, cfg.SPReg, op.stackOff, op.storeSize)
			}
		}
	}

	if cfg.PreCall != nil {
		cfg.PreCall(cfg, w, fn, lay)
	}

	if fn.CName == "" {
		panic("BindingFunc.CName must be set")
	}
	if fn.Static {
		// A statically linked symbol is defined in the final binary, so
		// call it directly rather than through a dynamic import slot.
		fmt.Fprintf(w, "\tCALL\t%s(SB)\n", fn.CName)
	} else {
		cfg.emitCallSymbol(w, fn.CName)
	}

	if fn.Ret != "" && lay.Ret.Size > 0 && !lay.Ret.Indirect {
		for _, slot := range cfg.retRegSlots(fn, lay.Ret, types) {
			cfg.emitStoreSlot(w, slot, base, lay.Ret.GoOffset)
		}
	}

	if cfg.TrampolinePrologue == nil && lay.StackSize > 0 {
		cfg.emitAddSP(w, lay.StackSize)
	}
	if epilogue != nil {
		epilogue(w)
	}

	fmt.Fprintf(w, "\tRET\n\n")
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
func amd64AssignRet(cfg *ABIConfig, fn BindingFunc, info typeInfo, st *abiState) RetLoc {
	if !info.composite {
		return defaultAssignRet(cfg, fn, info, st)
	}
	if ebs, ok := classifySysVStruct(info); ok {
		var retInt, retFloat int
		if regs, allFloat, fit := assignSysVRegs(ebs, cfg.IntRetRegs, cfg.FloatRetRegs, &retInt, &retFloat); fit {
			return RetLoc{Size: info.size, Regs: regs, IsFloat: allFloat}
		}
	}
	st.intIdx = 1
	return RetLoc{Size: info.size, Indirect: true}
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

// amd64CompositeRegSlots maps each eightbyte of a register-passed or
// register-returned SysV AMD64 struct to its integer or SSE register piece.
func amd64CompositeRegSlots(_ *ABIConfig, info typeInfo, _ bool, regs []string) ([]regSlot, bool) {
	ebs, ok := classifySysVStruct(info)
	if !ok || len(ebs) != len(regs) {
		return nil, false
	}
	slots := make([]regSlot, len(regs))
	for k, r := range regs {
		slots[k] = regSlot{reg: r, offset: int64(k) * 8, size: ebs[k].size, isFloat: ebs[k].isFloat}
	}
	return slots, true
}

var amd64ABI = &ABIConfig{
	Arch:              "amd64",
	PtrSize:           8,
	CLongSize:         8,
	StackAlign:        16,
	IntRegs:           []string{"DI", "SI", "DX", "CX", "R8", "R9"},
	FloatRegs:         []string{"X0", "X1", "X2", "X3", "X4", "X5", "X6", "X7"},
	IntRetRegs:        []string{"AX", "DX"},
	FloatRetRegs:      []string{"X0", "X1"},
	MaxRegStructSize:  16,
	IndirectRetReg:    "DI",
	IndirectRetInArg:  true,
	AssignParam:       amd64AssignParam,
	AssignRet:         amd64AssignRet,
	CompositeRegSlots: amd64CompositeRegSlots,
	SPReg:             "SP",
	CalleeSavedReg:    "R12",
	TmpReg:            "R11",
	TmpReg2:           "R10",
	asm: asmOps{
		mov:   "MOVQ",
		loadB: "MOVBLSX", loadBU: "MOVBLZX", loadH: "MOVWLSX", loadHU: "MOVWLZX", loadW: "MOVL", loadWU: "MOVL",
		storeB: "MOVB", storeH: "MOVW", storeW: "MOVL",
		fmovS: "MOVSS", fmovD: "MOVSD",
		shl:   "SHLQ\t$%[1]d, %[2]s",
		shr:   "SHRQ\t$%[1]d, %[2]s",
		or:    "ORQ\t%[1]s, %[2]s",
		addr:     "LEAQ\t%[2]d(%[1]s), %[3]s",
		subSP:    "SUBQ\t$%[1]d, %[2]s",
		addSP:    "ADDQ\t$%[1]d, %[2]s",
		useAdjSP: true,
	},
	// System V AMD64 psABI Section 3.5.7 requires AL to hold an upper bound
	// (0..8) on the number of vector registers used to pass arguments to a
	// variadic (or unprototyped) C function. Always set AX—clearing it when
	// FloatRegCount == 0—since a bound C function (such as open, fcntl, or
	// ioctl) may be variadic in C.
	PreCall: func(cfg *ABIConfig, w io.Writer, _ BindingFunc, lay FuncLayout) {
		if lay.FloatRegCount == 0 {
			fmt.Fprintf(w, "\tXORL\tAX, AX\n")
		} else {
			fmt.Fprintf(w, "\tMOVL\t$%d, AX\n", min(lay.FloatRegCount, len(cfg.FloatRegs)))
		}
	},
}

// Windows x64 (Microsoft x64) C calling convention.
//
// Specification: "x64 calling convention", Microsoft Learn,
// https://learn.microsoft.com/en-us/cpp/build/x64-calling-convention.
//
// Summary:
//   - The first four argument positions use shared positional slots: CX/X0,
//     DX/X1, R8/X2, and R9/X3; later arguments pass in 8-byte stack slots above
//     a mandatory 32-byte caller-allocated shadow ("home") space
//     ("Parameter passing").
//   - Aggregates of exactly 1, 2, 4, or 8 bytes pass and return by value in a
//     single GPR (even if all fields are floating-point); all other aggregate
//     sizes pass by caller-allocated pointer and return via a hidden sret
//     pointer in CX that consumes positional slot 0 ("Return values").
//   - Scalar integer/pointer results return in AX; floating-point results
//     return in X0.
//   - For variadic calls, floating-point arguments in slots 0..3 must be
//     duplicated in both the XMM register and the corresponding GPR
//     ("Varargs").

// isWin64RegStructSize reports whether a composite byte size is passed and
// returned directly in a GPR under the Windows x64 (and x86) C ABI.
func isWin64RegStructSize(size int64) bool {
	return size == 1 || size == 2 || size == 4 || size == 8
}

// win64AssignRet assigns Windows x64 composite returns of size 1, 2, 4, or 8
// bytes to AX, and all other composite sizes to a hidden sret pointer in CX
// consuming positional slot 0 ("Return values" in the x64 calling convention).
func win64AssignRet(cfg *ABIConfig, fn BindingFunc, info typeInfo, st *abiState) RetLoc {
	if !info.composite {
		return defaultAssignRet(cfg, fn, info, st)
	}
	if isWin64RegStructSize(info.size) {
		return RetLoc{Size: info.size, Regs: []string{cfg.IntRetRegs[0]}}
	}
	st.intIdx = 1
	st.floatIdx = 1
	return RetLoc{Size: info.size, Indirect: true}
}

// win64AssignParam implements Windows x64 positional argument allocation
// across the 4 shared register slots (CX/X0, DX/X1, R8/X2, R9/X3), passing
// non-power-of-two or >8-byte structs by hidden pointer ("Parameter passing"
// in the x64 calling convention).
func win64AssignParam(cfg *ABIConfig, _ BindingFunc, _ int, info typeInfo, st *abiState) ParamLoc {
	slot := max(st.intIdx, st.floatIdx)
	loc := ParamLoc{Size: info.size, IsFloat: info.isFloat}
	if info.composite && !isWin64RegStructSize(info.size) {
		loc.PassByPtr = true
	}
	if info.isFloat {
		if slot < len(cfg.FloatRegs) {
			loc.Regs = []string{cfg.FloatRegs[slot]}
			slot++
		} else {
			loc.StackOff = st.stackOff
			st.stackOff += 8
		}
	} else if slot < len(cfg.IntRegs) {
		loc.Regs = []string{cfg.IntRegs[slot]}
		slot++
	} else {
		loc.StackOff = st.stackOff
		st.stackOff += 8
	}
	st.intIdx, st.floatIdx = slot, slot
	return loc
}

// newWin64ABI derives the Windows x64 ABIConfig from amd64ABI, configuring
// the 32-byte home space, positional register sets, and PE IAT indirect calls.
func newWin64ABI() *ABIConfig {
	cfg := *amd64ABI
	cfg.OS = "windows"
	cfg.CLongSize = 4
	cfg.ShadowSpace = 32
	cfg.IntRegs = []string{"CX", "DX", "R8", "R9"}
	cfg.FloatRegs = []string{"X0", "X1", "X2", "X3"}
	cfg.IntRetRegs = []string{"AX"}
	cfg.FloatRetRegs = []string{"X0"}
	cfg.MaxRegStructSize = 8
	cfg.IndirectRetReg = "CX"
	cfg.PreCall = nil
	cfg.AssignParam = win64AssignParam
	cfg.AssignRet = win64AssignRet
	cfg.CompositeRegSlots = nil
	// On PE, a //go:cgo_import_dynamic symbol is not the function itself
	// but a pointer-sized slot holding its address: an IAT entry when
	// linking internally, or a data word relocated to the host symbol when
	// linking externally (see initdynimport in cmd/link/internal/ld/pe.go).
	// A direct CALL sym(SB) would jump into that data, and the linker does
	// not generate stubs for Go calls to dynamic imports on PE, so load the
	// function address from the slot and call through a register, as the
	// runtime does for DLL calls and as C compilers do for dllimport calls.
	cfg.CallSymbol = func(w io.Writer, sym string) {
		fmt.Fprintf(w, "\tMOVQ\t%s(SB), R11\n", sym)
		fmt.Fprintf(w, "\tCALL\tR11\n")
	}
	return &cfg
}

var win64ABI = newWin64ABI()

// ARM64 (AArch64) C calling convention (AAPCS64 and Darwin/Apple ARM64).
//
// Specifications:
//   - "Procedure Call Standard for the Arm 64-bit Architecture (AAPCS64)",
//     Section 6 ("The Base Procedure Call Standard"),
//     https://github.com/ARM-software/abi-aa/blob/main/aapcs64/aapcs64.rst
//   - Apple ARM64 ABI differences: "Writing ARM64 code for Apple platforms",
//     https://developer.apple.com/documentation/xcode/writing-arm64-code-for-apple-platforms
//
// Summary:
//   - Integer and pointer arguments use R0..R7; floating-point scalar arguments
//     use F0..F7 (AAPCS64 Section 6.8.2).
//   - Homogeneous Floating-Point Aggregates (HFAs) of 1 to 4 float32 or float64
//     elements with no padding pass in consecutive FP registers F0..F7 (or
//     spill to the stack while exhausting remaining FP registers) and return in
//     F0..F3 (Sections 5.9.5.1 and 6.8.2 Stage C.1–C.5).
//   - Non-HFA composites up to 16 bytes pass in consecutive GPRs R0..R7 (without
//     backfilling GPRs once a multi-register argument spills, Stage C.14) and
//     return in R0..R1.
//   - Non-HFA composites larger than 16 bytes pass by caller-allocated copy via
//     a pointer in a GPR or stack slot (Stage B.4), and return via a dedicated
//     indirect result address in R8 (XR) without consuming R0 (Section 6.9).
//   - The stack pointer (RSP) is 16-byte aligned at call boundaries.
//   - On Apple platforms (darwin/arm64, ios/arm64), non-variadic narrow scalar
//     and float32 HFA stack arguments pack at their natural alignment with
//     exact-width stores, whereas non-HFA composites <= 16 bytes and hidden
//     PassByPtr pointers remain 8-byte aligned.

// classifyARM64HFA reports whether composite type info is a Homogeneous
// Floating-Point Aggregate (1 to 4 identically sized float32 or float64
// members with no padding) per AAPCS64 Section 5.9.5.1, returning its element
// size and member count.
func classifyARM64HFA(info typeInfo) (elemSize int64, count int, isHFA bool) {
	if info.size <= 0 || info.size > 32 || !info.fieldsOK || len(info.fields) < 1 || len(info.fields) > 4 {
		return 0, 0, false
	}
	elemSize = info.fields[0].size
	if !info.fields[0].isFloat || (elemSize != 4 && elemSize != 8) || info.size != int64(len(info.fields))*elemSize {
		return 0, 0, false
	}
	for i, f := range info.fields {
		if !f.isFloat || f.size != elemSize || f.offset != int64(i)*elemSize {
			return 0, 0, false
		}
	}
	return elemSize, len(info.fields), true
}

// arm64AssignRet returns Homogeneous Floating-Point Aggregates in F0..F3 and
// falls back to defaultAssignRet (R0..R1 or indirect sret in R8) otherwise
// (AAPCS64 Section 6.9).
func arm64AssignRet(cfg *ABIConfig, fn BindingFunc, info typeInfo, st *abiState) RetLoc {
	if info.composite {
		if _, count, ok := classifyARM64HFA(info); ok {
			return RetLoc{Size: info.size, Regs: slices.Clone(cfg.FloatRetRegs[:count]), IsFloat: true}
		}
	}
	return defaultAssignRet(cfg, fn, info, st)
}

// arm64AssignParam allocates AAPCS64 parameters per Section 6.8.2: HFAs in
// F0..F7, floats in F0..F7, >16-byte composites by pointer, and integers or
// <=16-byte non-HFA composites in R0..R7 without GPR backfilling.
func arm64AssignParam(cfg *ABIConfig, _ BindingFunc, _ int, info typeInfo, st *abiState) ParamLoc {
	if info.composite {
		if _, count, ok := classifyARM64HFA(info); ok {
			if st.floatIdx+count <= len(cfg.FloatRegs) {
				regs := slices.Clone(cfg.FloatRegs[st.floatIdx : st.floatIdx+count])
				st.floatIdx += count
				return ParamLoc{Size: info.size, IsFloat: true, Regs: regs}
			}
			st.floatIdx = len(cfg.FloatRegs)
			if cfg.PackStackArgs {
				return cfg.assignPackedStackParam(info.size, info.align, true, st)
			}
			return cfg.assignStackParam(info.size, true, st)
		}
		if info.size > cfg.MaxRegStructSize {
			return cfg.assignPtrParam(info.size, st)
		}
	}
	if info.isFloat {
		return cfg.assignFloatParam(info, st)
	}
	loc := cfg.assignIntParam(info, st)
	if info.size > 0 && len(loc.Regs) == 0 {
		st.intIdx = len(cfg.IntRegs)
	}
	return loc
}

// arm64CompositeRegSlots maps each member of a register-passed or
// register-returned ARM64 HFA to its consecutive floating-point register.
func arm64CompositeRegSlots(_ *ABIConfig, info typeInfo, isFloat bool, regs []string) ([]regSlot, bool) {
	if !isFloat {
		return nil, false
	}
	elemSz := info.size / int64(len(regs))
	slots := make([]regSlot, len(regs))
	for k, r := range regs {
		slots[k] = regSlot{reg: r, offset: int64(k) * elemSz, size: elemSz, isFloat: true}
	}
	return slots, true
}

// emitARM64TrampolinePrologue emits a NOFRAME arm64 Go-to-C trampoline prologue
// and returns its matching epilogue. Maintaining a single frame with an AAPCS64
// (R29, R30) frame record at the top (so CFA = R29+16 = caller SP) avoids
// placing R29 below RSP where C callees clobber it, and keeps 0(RSP) = saved R30
// via atomic STP.W/LDP.P whenever no C stack arguments occupy 0(RSP).
func emitARM64TrampolinePrologue(w io.Writer, lay FuncLayout) func(io.Writer) {
	if !lay.hasStackArgs() {
		total := max(int64(16), lay.StackSize) + 16
		fpOff := total - 16
		if total <= 504 {
			fmt.Fprintf(w, "\tSTP.W\t(R30, R19), -%d(RSP)\n", total)
			fmt.Fprintf(w, "\tSTP\t(R29, R30), %d(RSP)\n", fpOff)
		} else {
			fmt.Fprintf(w, "\tSUB\t$%d, RSP\n", total)
			fmt.Fprintf(w, "\tMOVD\tR30, 0(RSP)\n")
			fmt.Fprintf(w, "\tMOVD\tR19, 8(RSP)\n")
			fmt.Fprintf(w, "\tMOVD\tR29, %d(RSP)\n", fpOff)
			fmt.Fprintf(w, "\tMOVD\tR30, %d(RSP)\n", fpOff+8)
		}
		fmt.Fprintf(w, "\tADD\t$%d, RSP, R29\n", fpOff)
		fmt.Fprintf(w, "\tMOVD\tR0, R19\n")
		return func(w io.Writer) {
			fmt.Fprintf(w, "\tMOVD\t%d(RSP), R29\n", fpOff)
			if total <= 504 {
				fmt.Fprintf(w, "\tLDP.P\t%d(RSP), (R30, R19)\n", total)
			} else {
				fmt.Fprintf(w, "\tMOVD\t8(RSP), R19\n")
				fmt.Fprintf(w, "\tMOVD\t0(RSP), R30\n")
				fmt.Fprintf(w, "\tADD\t$%d, RSP\n", total)
			}
		}
	}
	total := lay.StackSize + 32
	fpOff := total - 16
	r19Off := total - 32
	fmt.Fprintf(w, "\tSUB\t$%d, RSP\n", total)
	if fpOff <= 504 {
		fmt.Fprintf(w, "\tSTP\t(R29, R30), %d(RSP)\n", fpOff)
	} else {
		fmt.Fprintf(w, "\tMOVD\tR29, %d(RSP)\n", fpOff)
		fmt.Fprintf(w, "\tMOVD\tR30, %d(RSP)\n", fpOff+8)
	}
	fmt.Fprintf(w, "\tADD\t$%d, RSP, R29\n", fpOff)
	fmt.Fprintf(w, "\tMOVD\tR19, %d(RSP)\n", r19Off)
	fmt.Fprintf(w, "\tMOVD\tR0, R19\n")
	return func(w io.Writer) {
		fmt.Fprintf(w, "\tMOVD\t%d(RSP), R19\n", r19Off)
		if fpOff <= 504 {
			fmt.Fprintf(w, "\tLDP\t%d(RSP), (R29, R30)\n", fpOff)
		} else {
			fmt.Fprintf(w, "\tMOVD\t%d(RSP), R29\n", fpOff)
			fmt.Fprintf(w, "\tMOVD\t%d(RSP), R30\n", fpOff+8)
		}
		fmt.Fprintf(w, "\tADD\t$%d, RSP\n", total)
	}
}

var arm64ABI = &ABIConfig{
	Arch:              "arm64",
	PtrSize:           8,
	CLongSize:         8,
	StackAlign:        16,
	IntRegs:           []string{"R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7"},
	FloatRegs:         []string{"F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7"},
	IntRetRegs:        []string{"R0", "R1"},
	FloatRetRegs:      []string{"F0", "F1", "F2", "F3"},
	MaxRegStructSize:  16,
	IndirectRetReg:    "R8",
	AssignParam:       arm64AssignParam,
	AssignRet:         arm64AssignRet,
	CompositeRegSlots: arm64CompositeRegSlots,
	SPReg:             "RSP",
	CalleeSavedReg:    "R19",
	TmpReg:            "R12",
	TmpReg2:           "R13",
	LRReserve:         16,
	asm: asmOps{
		mov:   "MOVD",
		loadB: "MOVB", loadBU: "MOVBU", loadH: "MOVH", loadHU: "MOVHU", loadW: "MOVW", loadWU: "MOVWU",
		storeB: "MOVB", storeH: "MOVH", storeW: "MOVW",
		fmovS: "FMOVS", fmovD: "FMOVD",
		shl:   "LSL\t$%[1]d, %[2]s, %[2]s",
		shr:   "LSR\t$%[1]d, %[2]s, %[2]s",
		or:    "ORR\t%[1]s, %[2]s, %[2]s",
		addr:  "ADD\t$%[2]d, %[1]s, %[3]s",
		subSP: "SUB\t$%[1]d, %[2]s",
		addSP: "ADD\t$%[1]d, %[2]s",
	},
	TrampolinePrologue: emitARM64TrampolinePrologue,
}

// getABI returns the ABIConfig for the given targetOS/targetArch.
func getABI(targetOS, targetArch string) *ABIConfig {
	var cfg ABIConfig
	switch targetArch {
	case "amd64":
		if targetOS == "windows" {
			return win64ABI
		}
		cfg = *amd64ABI
	case "arm64":
		cfg = *arm64ABI
		if targetOS == "darwin" || targetOS == "ios" {
			cfg.PackStackArgs = true
		}
	default:
		panic("unsupported arch: " + targetArch)
	}
	cfg.OS = targetOS
	return &cfg
}
