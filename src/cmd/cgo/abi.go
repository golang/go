// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"cmp"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"internal/buildcfg"
	"io"
	"os"
	"slices"
	"strconv"
	"strings"
)

// BindingType describes a C or Go type used by cgo binding declarations
// for C ABI classification and Go wrapper generation.
type BindingType struct {
	Name        string // Mangled C type name (e.g. "_Ctype_foo") or Go helper type name
	Def         string // Underlying Go type expression (e.g. "int32", "struct { a float32 }"); empty if opaque
	Size        int64  // Size in bytes on the target architecture
	Align       int64  // Alignment in bytes on the target architecture
	HasPointers bool   // Whether the type contains pointers (for GC keepalives and callback return zeroing)
}

// BindingFunc describes a C function bound with //cgo:func, either a
// function linked with the program or a method of a //cgo:dynload type that
// calls a function in a library loaded at run time, or an exported Go
// callback (//export).
type BindingFunc struct {
	// Identifier and symbol names:
	Name   string // Internal cgo function identifier (used in _Cfunc_<Name>, _cgo_arg_<Name>, and _cgo_trampoline_<Name>)
	GoName string // Go function named by the //cgo:func pragma (used by binding.go)
	CName  string // Target C symbol name
	BindC  bool   // The pragma wrote the C name as C.<CName>, so Go code may also call the function as C.<CName> (used by binding.go)

	// Signature (Go type expressions, e.g. "int32", "*byte", "_Ctype_S"):
	Params []string // Declared parameter types
	Ret    string   // Return type expression, or "" for void

	// Calling convention modifiers (used by ABI layout and trampoline generation):
	Indirect    bool // Call target address is passed at offset 0 of the Go arg struct (//cgo:dynload methods)
	MayCallback bool // C call may re-enter Go; spills pointer-passed args/returns to g0 stack and tracks _cgo_topofstack

	// Go wrapper and import metadata (used by binding.go):
	Library string // Shared library or DLL name for dynamic import directives; empty for libc and static
	LibC    bool   // The symbol comes from the platform C runtime (libc)
	Static  bool   // The symbol is resolved at link time from a .syso or .a (static)

	// Methods of //cgo:dynload types (used by binding.go):
	RecvType string // The //cgo:dynload type, for a //cgo:func LibT.GoName pragma
	SymIndex int    // Index of the symbol in RecvType's per-instance symbol table
}

// ParamLoc describes where a function parameter is passed in the C ABI.
type ParamLoc struct {
	GoOffset   int64    // Byte offset in Go _cgo_arg_<Name> struct
	Size       int64    // Parameter size in bytes
	IsFloat    bool     // Floating-point parameter
	Regs       []string // Register(s) if passed in register(s) (or leading register half of a split param)
	StackOff   int64    // Byte offset in C stack frame when passed on the stack (or trailing stack half when StackBytes > 0)
	StackBytes int64    // Trailing byte count placed on the stack when a composite splits across Regs and stack
	PassByPtr  bool     // True if passed by hidden reference (pointer in Regs[0] or on stack)
	G0Off      int64    // Scratch offset on g0 stack when MayCallback && PassByPtr
}

// RetLoc describes how a function return value is returned in the C ABI.
type RetLoc struct {
	GoOffset        int64    // Byte offset in Go _cgo_arg_<Name> struct
	Size            int64    // Return value size in bytes
	IsFloat         bool     // Floating-point return
	Indirect        bool     // True if caller passes a hidden sret return buffer pointer
	IndirectOnStack bool     // True if hidden sret pointer is passed at 0(SP) (386 ABI)
	Regs            []string // Register(s) holding the return value upon C return (e.g. AX, X0)
	G0Off           int64    // Scratch offset on g0 stack when MayCallback
}

// FuncLayout describes the complete Go argument struct and C ABI layout for a call.
type FuncLayout struct {
	Params        []ParamLoc
	Ret           RetLoc
	StackSize     int64 // C stack frame size in bytes (aligned to ABIConfig.StackAlign)
	FloatRegCount int   // Number of floating-point argument registers allocated (for SysV AMD64 AX)
	StkTopOff     int64 // Saved _cgo_topofstack slot on g0 stack when MayCallback
}

// hasStackArgs reports whether any argument is passed on the C stack.
func (lay FuncLayout) hasStackArgs() bool {
	if lay.Ret.IndirectOnStack {
		return true
	}
	for _, p := range lay.Params {
		if (p.Size > 0 && len(p.Regs) == 0) || p.StackBytes > 0 {
			return true
		}
	}
	return false
}

// abiState tracks register and stack allocation cursors while computing a FuncLayout.
type abiState struct {
	intIdx       int
	floatIdx     int
	vfpMask      uint16 // bitmask of allocated S0..S15 VFP lanes (32-bit ARM only)
	stackOff     int64
	fprsDisabled bool // disables FPR arg allocation once a non-float/sret arg or F14 is used (32-bit MIPS O32 only)
}

// leafField represents a single scalar leaf within a flattened composite type.
type leafField struct {
	offset  int64
	size    int64
	align   int64
	isFloat bool
	signed  bool
	direct  bool // Scalar member of the enclosing struct itself, not flattened from an array, nested struct, or complex type
}

// typeInfo holds the byte size, alignment, ABI classification, and flattened
// scalar leaves of a type.
type typeInfo struct {
	size              int64
	align             int64
	isFloat           bool
	signed            bool
	composite         bool
	isComplex         bool // complex64 or complex128 (C _Complex float or _Complex double)
	isStruct          bool // Go struct type (C struct), as opposed to an array, complex, or scalar type
	directFields      int  // Number of members of a struct type, not counting structs.HostLayout
	singleFloatStruct bool // Struct whose only member is a float32, float64, or such a struct
	hasPointers       bool
	fields            []leafField
	fieldsOK          bool
}

// regSlot describes a single register piece of a parameter or return value.
type regSlot struct {
	reg         string
	offset      int64 // Byte offset relative to ParamLoc.GoOffset or RetLoc.GoOffset
	size        int64 // Byte size of this piece (1..8)
	align       int64 // Guaranteed alignment of the backing field (0 means PtrSize)
	isFloat     bool
	signed      bool
	leftJustify bool // Big-endian composite piece < PtrSize is left-justified in the upper bytes of reg
}

// asmOps defines the assembly mnemonics for an architecture's load, store,
// shift, arithmetic, and stack-pointer operations.
type asmOps struct {
	mov        string // Pointer-sized integer load/store/reg-move (e.g. "MOVQ", "MOVD")
	loadB      string // Signed 1-byte load
	loadBU     string // Unsigned 1-byte load
	loadH      string // Signed 2-byte load
	loadHU     string // Unsigned 2-byte load
	loadW      string // Signed 4-byte load
	loadWU     string // Unsigned 4-byte load
	storeB     string // 1-byte store
	storeH     string // 2-byte store
	storeW     string // 4-byte store
	fmovS      string // 32-bit float load/store
	fmovD      string // 64-bit float load/store
	shl        string // Left shift mnemonic (e.g. "SHLQ", "LSL", "SLL")
	shr        string // Logical right shift mnemonic (e.g. "SHRQ", "LSR", "SRL")
	or         string // Bitwise OR mnemonic (e.g. "ORQ", "ORR", "OR")
	add        string // Add mnemonic (e.g. "ADDQ", "ADD", "ADDV", "ADDU")
	sub        string // Subtract mnemonic (e.g. "SUBQ", "SUB", "SUBV", "SUBU")
	lea        string // Effective-address mnemonic on x86 ("LEAQ", "LEAL"), or "" to use add
	twoOpALU   bool   // True if or/add/sub take 2 operands (src, dst) rather than 3 (src, dst, dst)
	twoOpShift bool   // True if shl/shr take 2 operands ($bits, reg) rather than 3 ($bits, reg, reg)
	useAdjSP   bool   // True if SP adjustments must use ADJSP so obj tracks pcsp without FuncFlagSPWrite
	negSubSP   bool   // True if decrementing SP uses add with $-size rather than sub with $size
	alignDisp4 bool   // True if 8-byte loads/stores and signed 4-byte loads require 4-byte-aligned displacements (PPC64 DS-form)
}

// ABIConfig defines the C calling convention rules and assembly primitives
// for a target OS/architecture. OS, PtrSize, CLongSize, StackAlign, and AssignParam must always be set.
type ABIConfig struct {
	Arch             string
	OS               string
	PtrSize          int64
	CLongSize        int64 // Byte size of C long and unsigned long
	StackAlign       int64
	ShadowSpace      int64 // Bytes reserved at the bottom of the C call frame by the ABI (e.g. 32-byte home space on Windows x64)
	MinParamSaveArea int64 // Minimum parameter save area reserved above ShadowSpace before g0 scratch (64B on PPC64/AIX)

	IntRegs      []string
	FloatRegs    []string
	IntRetRegs   []string
	FloatRetRegs []string

	MaxRegStructSize   int64
	IndirectRetReg     string // Register holding hidden sret buffer pointer
	IndirectRetInArg   bool   // Hidden sret pointer consumes IntRegs[0]
	IndirectRetOnStack bool   // Hidden sret pointer is passed at 0(SP) (386)
	SysVI386SretPop    bool   // Callee pops 4B sret pointer on return (SysV 386)
	PackStackArgs      bool   // Pack narrow scalar and HFA stack args at natural alignment (Apple ARM64)

	// AssignParam allocates a single parameter in registers or on the C stack.
	AssignParam func(cfg *ABIConfig, fn BindingFunc, idx int, info typeInfo, st *abiState) ParamLoc

	// AssignRet determines how the return value is returned in the C ABI.
	// ABIs without special return rules use defaultAssignRet.
	AssignRet func(cfg *ABIConfig, fn BindingFunc, info typeInfo, st *abiState) RetLoc

	// Optional platform-specific ABI hook. Returning handled=false falls back
	// to default GPR chunking.
	CompositeRegSlots func(cfg *ABIConfig, info typeInfo, isFloat bool, regs []string) ([]regSlot, bool)

	// Trampoline and callback assembly generation parameters:
	SPReg           string // Hardware stack pointer register in asm (e.g. "SP", "RSP", "R1", "X2")
	CalleeSavedReg  string // C callee-saved register holding the Go _cgo_arg struct pointer across C calls
	TmpReg          string // Primary caller-saved scratch register for stack/memory copies and address math
	TmpReg2         string // Secondary caller-saved scratch register for multi-chunk loads and copies
	IndCallReg      string // Register for indirect C calls when different from TmpReg (e.g. "R25" on MIPS)
	LinkReg         string // Link register saved by default RISC trampoline/callback prologue/epilogue (e.g. "X1", "R31")
	AsmTmpReg       string // Callee-saved assembler REGTMP saved in RISC callback prologue (e.g. "R23", "R30")
	LRReserve       int64  // Bytes reserved at 0(SP) for saved LR when !hasStackArgs and ShadowSpace == 0
	ShadowUsesSlot0 bool   // True if C callees may overwrite 0(SP) inside ShadowSpace (MIPS O32)
	ZeroVal         string // Zero operand or register for zeroing pointer-containing callback return slots, or "" to zero TmpReg
	LocalFrameOff   int64  // Byte offset from SPReg to the local Go arg struct in writeCallbackTrampoline
	IndirectSymCall bool   // External C symbols are PE IAT pointers called indirectly (Windows)
	HasFuncDesc     bool   // C and trampoline symbols use 24-byte (PC, TOC, env) XCOFF descriptors (AIX)

	BigEndian                    bool // Multi-byte integer chunks in registers and stack slots use big-endian byte order
	LeftJustifyBEComposite       bool // Big-endian composite register chunks < PtrSize are left-justified (MIPS, AIX)
	LeftJustifyBETrailing        bool // Big-endian trailing register chunks of >PtrSize composites are left-justified (ELFv2 BE)
	RightJustifyBECompositeStack bool // Big-endian sub-word composites on the stack are right-justified (s390x, ELFv2 BE)
	LeftJustifyBEFloatStack      bool // Big-endian float32 stack arguments are left-justified in their slot (AIX, MIPS64)
	ComplexFloatSplitStack       bool // The float32 halves of a complex64 stack argument occupy separate PtrSize slots (PPC64)
	SignExtUint32                bool // 32-bit unsigned integers are sign-extended to 64 bits in registers and stack slots (MIPS64, RISC-V, LoongArch)
	StrictAlign                  bool // Architecture traps on unaligned multi-byte loads/stores (MIPS, ARM)

	asm                asmOps
	IsFloatReg         func(reg string) bool
	LoadFloat          func(w io.Writer, base string, off, size int64, reg string) bool
	StoreFloat         func(w io.Writer, reg, base string, off, size int64) bool
	TrampolinePrologue func(w io.Writer, lay FuncLayout) (epilogue func(w io.Writer))
	Prologue           func(w io.Writer) (epilogue func(w io.Writer))
	PreCall            func(cfg *ABIConfig, w io.Writer, fn BindingFunc, lay FuncLayout)
	CallIndirect       func(w io.Writer, base string)
	CallSymbol         func(w io.Writer, sym string, static bool)
	StackArgAddr       func(stackOff, callerSPOff int64) (base string, off int64)
	CallbackPrologue   func(w io.Writer, frameSize int64) (callerSPOff int64)
	CallbackEpilogue   func(w io.Writer, frameSize int64, sretPop bool)
	CallCrosscall2     func(w io.Writer, fnName string, frameSize, ctxtOff int64)
}

// isFloatReg reports whether reg is an architecture floating-point register.
func (cfg *ABIConfig) isFloatReg(reg string) bool {
	if cfg.IsFloatReg != nil && cfg.IsFloatReg(reg) {
		return true
	}
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
		fields:   []leafField{{size: cfg.PtrSize, align: cfg.PtrSize, direct: true}},
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
//
// Go strings, slices, maps, channels, funcs, interfaces, and instantiated
// generic types have no C ABI layout. cTypeError rejects them before any
// layout is computed, so inspectExpr panics if it sees one.
func (cfg *ABIConfig) inspectExpr(expr ast.Expr, types map[string]BindingType, seen map[string]bool) typeInfo {
	if isHostLayoutExpr(expr) {
		return typeInfo{align: 1, fieldsOK: true}
	}
	switch e := expr.(type) {
	case *ast.ParenExpr:
		return cfg.inspectExpr(e.X, types, seen)
	case *ast.StarExpr:
		d := cfg.ptrTypeInfo()
		d.hasPointers = true
		return d
	case *ast.ArrayType:
		if e.Len == nil {
			panic("Go slice type in ABI type")
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
			return typeInfo{size: ed.size * n, align: ed.align, composite: true, hasPointers: ed.hasPointers}
		}
		fields := make([]leafField, 0, int64(len(ed.fields))*n)
		for i := range n {
			delta := i * ed.size
			for _, f := range ed.fields {
				f.offset += delta
				f.direct = false
				fields = append(fields, f)
			}
		}
		return typeInfo{
			size:        ed.size * n,
			align:       ed.align,
			composite:   true,
			hasPointers: ed.hasPointers,
			fields:      fields,
			fieldsOK:    true,
		}
	case *ast.StructType:
		var off int64
		maxAlign := int64(1)
		var fields []leafField
		fieldsOK := true
		hasPointers := false
		nFields := 0
		singleFloat := false
		for _, fld := range e.Fields.List {
			if isHostLayoutExpr(fld.Type) {
				continue
			}
			fd := cfg.inspectExpr(fld.Type, types, seen)
			maxAlign = max(maxAlign, fd.align)
			hasPointers = hasPointers || fd.hasPointers
			if !fd.fieldsOK {
				fieldsOK = false
				fields = nil
			}
			n := max(1, len(fld.Names))
			nFields += n
			singleFloat = nFields == 1 && (fd.isFloat || fd.singleFloatStruct)
			for range n {
				off = alignUp(off, fd.align)
				if fieldsOK {
					for _, f := range fd.fields {
						f.offset += off
						// Leaves of a nested struct are not direct
						// members of this struct.
						f.direct = f.direct && !fd.isStruct
						fields = append(fields, f)
					}
				}
				off += fd.size
			}
		}
		return typeInfo{
			size:              alignUp(off, maxAlign),
			align:             maxAlign,
			composite:         true,
			isStruct:          true,
			directFields:      nFields,
			singleFloatStruct: singleFloat,
			hasPointers:       hasPointers,
			fields:            fields,
			fieldsOK:          fieldsOK,
		}
	case *ast.SelectorExpr:
		if id, ok := e.X.(*ast.Ident); ok {
			if (id.Name == "structs" && e.Sel.Name == "HostLayout") ||
				((id.Name == "_cgopackage" || id.Name == "cgo") && e.Sel.Name == "Incomplete") {
				return typeInfo{align: 1, fieldsOK: true}
			}
			if id.Name == "C" {
				return cfg.inspectTypeName("_Ctype_"+e.Sel.Name, types, seen)
			}
		}
		d := cfg.ptrTypeInfo()
		d.hasPointers = true
		return d
	case *ast.Ident:
		return cfg.inspectTypeName(e.Name, types, seen)
	}
	panic(fmt.Sprintf("unsupported type expression %T in ABI type", expr))
}

// scalarSpec describes the size and classification of a built-in Go or C scalar type.
type scalarSpec struct {
	size    int64 // Fixed byte size, or 0 when ptr or cLong is set
	ptr     bool  // Size equals ABIConfig.PtrSize
	cLong   bool  // Size equals ABIConfig.CLongSize
	cChar   bool  // Signedness depends on target C ABI (char)
	signed  bool
	isFloat bool
}

func targetCCharIsUnsigned(os, arch string) bool {
	if os == "darwin" || os == "ios" || os == "windows" {
		return false
	}
	switch arch {
	case "arm", "arm64", "ppc64", "ppc64le", "riscv64", "s390x":
		return true
	}
	return false
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
	"_Ctype_char":      {size: 1, cChar: true},
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
		signed := s.signed
		if s.cChar {
			signed = !targetCCharIsUnsigned(cfg.OS, cfg.Arch)
		}
		align := sz
		if cfg.Arch == "386" && cfg.OS != "windows" {
			align = min(sz, 4)
		}
		d := typeInfo{size: sz, align: align, signed: signed, isFloat: s.isFloat}
		if sz > 0 {
			d.fields = []leafField{{size: d.size, align: d.align, isFloat: d.isFloat, signed: d.signed, direct: true}}
			d.fieldsOK = true
		}
		return d
	}

	switch name {
	case "string", "any", "error":
		panic("Go type " + name + " in ABI type")
	case "complex64":
		return typeInfo{
			size:      8,
			align:     4,
			composite: true,
			isComplex: true,
			fields: []leafField{
				{offset: 0, size: 4, align: 4, isFloat: true},
				{offset: 4, size: 4, align: 4, isFloat: true},
			},
			fieldsOK: true,
		}
	case "complex128":
		al := cfg.inspectTypeName("float64", nil, nil).align
		return typeInfo{
			size:      16,
			align:     al,
			composite: true,
			isComplex: true,
			fields: []leafField{
				{offset: 0, size: 8, align: al, isFloat: true},
				{offset: 8, size: 8, align: al, isFloat: true},
			},
			fieldsOK: true,
		}
	}

	bt, ok := types[name]
	if !ok {
		return cfg.ptrTypeInfo()
	}
	if bt.Def == "" {
		return typeInfo{size: bt.Size, align: bt.Align, composite: true, hasPointers: bt.HasPointers}
	}
	if seen[name] {
		return typeInfo{size: bt.Size, align: bt.Align, hasPointers: bt.HasPointers}
	}
	expr, err := parser.ParseExpr(bt.Def)
	if err != nil {
		return typeInfo{size: bt.Size, align: bt.Align, hasPointers: bt.HasPointers}
	}
	if seen == nil {
		seen = make(map[string]bool)
	}
	seen[name] = true
	d := cfg.inspectExpr(expr, types, seen)
	delete(seen, name)
	d.hasPointers = d.hasPointers || bt.HasPointers
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
		ret.GoOffset = alignUp(goOff, min(retAlign, cfg.PtrSize))
	}
	if cfg.MinParamSaveArea > 0 {
		saveArea := max(cfg.MinParamSaveArea, int64(st.intIdx)*cfg.PtrSize)
		st.stackOff = max(st.stackOff, cfg.ShadowSpace+saveArea)
	}
	stkTopOff := cfg.assignG0Scratch(fn, params, &ret, &st)

	return FuncLayout{
		Params:        params,
		Ret:           ret,
		StackSize:     alignUp(st.stackOff, cfg.StackAlign),
		FloatRegCount: st.floatIdx,
		StkTopOff:     stkTopOff,
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
	if cfg.IndirectRetOnStack && info.composite && info.size > 0 {
		st.stackOff += 4
		return RetLoc{Size: info.size, Indirect: true, IndirectOnStack: true}
	}
	if info.size > cfg.MaxRegStructSize && (!info.isFloat || len(cfg.FloatRetRegs) == 0) && (info.composite || info.size > ptrSize*int64(len(cfg.IntRetRegs))) {
		if cfg.IndirectRetInArg {
			st.intIdx = 1
			st.fprsDisabled = true
		}
		return RetLoc{Size: info.size, Indirect: true}
	}
	if info.isFloat && len(cfg.FloatRetRegs) > 0 {
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
	if fn.Indirect {
		goOff = cfg.PtrSize
	}
	paramTypes := fn.Params
	params := make([]ParamLoc, len(paramTypes))
	for i, pt := range paramTypes {
		info := cfg.inspectType(pt, types)
		loc := cfg.AssignParam(cfg, fn, i, info, st)
		goOff = alignUp(goOff, min(info.align, cfg.PtrSize))
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

// assignG0Scratch reserves scratch space on the g0 stack frame when MayCallback
// is true so pointer-passed arguments, indirect return buffers, and the saved
// _cgo_topofstack pointer survive Go stack growth during C-to-Go callbacks.
func (cfg *ABIConfig) assignG0Scratch(fn BindingFunc, params []ParamLoc, ret *RetLoc, st *abiState) int64 {
	if !fn.MayCallback {
		return 0
	}
	needsScratch := fn.Ret != "" && ret.Size > 0
	for i := range params {
		if params[i].PassByPtr {
			needsScratch = true
			break
		}
	}
	if !needsScratch {
		return 0
	}
	if cfg.LRReserve > 0 && st.stackOff == 0 {
		// Leave 0..LRReserve-1(SP) free for the trampoline's saved LR slot
		// when no C stack arguments or ABI shadow space occupy 0(SP).
		st.stackOff = cfg.LRReserve
	}
	ptrSize := cfg.PtrSize
	for i := range params {
		if params[i].PassByPtr {
			st.stackOff = alignUp(st.stackOff, ptrSize)
			params[i].G0Off = st.stackOff
			st.stackOff += alignUp(params[i].Size, ptrSize)
		}
	}
	var stkTopOff int64
	if fn.Ret != "" && ret.Size > 0 {
		st.stackOff = alignUp(st.stackOff, ptrSize)
		ret.G0Off = st.stackOff
		st.stackOff += alignUp(ret.Size, ptrSize)
		stkTopOff = st.stackOff
		st.stackOff += ptrSize
	}
	return stkTopOff
}

// decomposePow2Aligned splits a byte size (1..8) into descending power-of-two
// chunks bounded by maxWidth (1, 2, 4, or 8).
func decomposePow2Aligned(size, maxWidth int64) []int64 {
	var chunks []int64
	for rem := size; rem > 0; {
		ck := maxPow2Chunk(rem, maxWidth)
		chunks = append(chunks, ck)
		rem -= ck
	}
	return chunks
}

// maxPow2Chunk returns the largest power-of-two byte width (1, 2, 4, or 8)
// that fits within both rem and maxWidth.
func maxPow2Chunk(rem, maxWidth int64) int64 {
	if maxWidth >= 8 && rem >= 8 {
		return 8
	}
	if maxWidth >= 4 && rem >= 4 {
		return 4
	}
	if maxWidth >= 2 && rem >= 2 {
		return 2
	}
	return 1
}

// effAlign returns the guaranteed power-of-two alignment of base+off given the
// backing field's alignment (0 means ptrSize).
func effAlign(off, align, ptrSize int64) int64 {
	if align <= 0 || align > ptrSize {
		align = ptrSize
	}
	if off != 0 {
		offAlign := off & -off
		if offAlign < align {
			align = offAlign
		}
	}
	if align < 1 {
		return 1
	}
	return align
}

func (cfg *ABIConfig) emitAlignDisp4(w io.Writer, reg, base string, off int64) (string, int64) {
	scratch := cfg.TmpReg
	if reg == scratch || base == scratch {
		scratch = cfg.TmpReg2
	}
	cfg.emitAddr(w, base, off, scratch)
	return scratch, 0
}

// emitScalarLoadInt emits a single power-of-two (1, 2, 4, or 8 byte) integer
// load from off(base) into reg, sign- or zero-extending narrow loads.
func (cfg *ABIConfig) emitScalarLoadInt(w io.Writer, base string, off, size int64, signed bool, reg string) {
	if cfg.asm.alignDisp4 && (size >= 8 || (size == 4 && signed)) && off%4 != 0 {
		base, off = cfg.emitAlignDisp4(w, reg, base, off)
	}
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
	if cfg.asm.alignDisp4 && size >= 8 && off%4 != 0 {
		base, off = cfg.emitAlignDisp4(w, reg, base, off)
	}
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

func (cfg *ABIConfig) emitShiftOp(w io.Writer, op string, bits int64, reg string) {
	if cfg.asm.twoOpShift {
		fmt.Fprintf(w, "\t%s\t$%d, %s\n", op, bits, reg)
	} else {
		fmt.Fprintf(w, "\t%s\t$%d, %s, %s\n", op, bits, reg, reg)
	}
}

func (cfg *ABIConfig) emitALUOp(w io.Writer, op, src, dst string) {
	if cfg.asm.twoOpALU {
		fmt.Fprintf(w, "\t%s\t%s, %s\n", op, src, dst)
	} else {
		fmt.Fprintf(w, "\t%s\t%s, %s, %s\n", op, src, dst, dst)
	}
}

// emitShiftLeft shifts reg left in place by bits.
func (cfg *ABIConfig) emitShiftLeft(w io.Writer, bits int64, reg string) {
	cfg.emitShiftOp(w, cfg.asm.shl, bits, reg)
}

// emitShiftRight logically shifts reg right in place by bits.
func (cfg *ABIConfig) emitShiftRight(w io.Writer, bits int64, reg string) {
	cfg.emitShiftOp(w, cfg.asm.shr, bits, reg)
}

// emitOrReg bitwise-ORs src into dst.
func (cfg *ABIConfig) emitOrReg(w io.Writer, src, dst string) {
	cfg.emitALUOp(w, cfg.asm.or, src, dst)
}

// emitLoadInt loads an integer or struct register chunk of 1..8 bytes from
// off(base) into reg, decomposing non-power-of-two sizes into exact sub-loads
// combined with shifts and ORs.
func (cfg *ABIConfig) emitLoadInt(w io.Writer, base string, off, size int64, signed bool, reg string) {
	cfg.emitLoadIntAlign(w, base, off, size, cfg.PtrSize, signed, reg)
}

func (cfg *ABIConfig) emitLoadIntAlign(w io.Writer, base string, off, size, align int64, signed bool, reg string) {
	maxWidth := cfg.PtrSize
	if cfg.StrictAlign {
		maxWidth = effAlign(off, align, cfg.PtrSize)
	}
	if (size == 1 || size == 2 || size == 4 || size == 8) && size <= maxWidth {
		cfg.emitScalarLoadInt(w, base, off, size, signed, reg)
		return
	}
	chunks := decomposePow2Aligned(size, maxWidth)
	scratch := cfg.TmpReg2
	if reg == scratch || base == scratch {
		scratch = cfg.TmpReg
	}
	cfg.emitScalarLoadInt(w, base, off, chunks[0], false, reg)
	curOff := chunks[0]
	for _, ck := range chunks[1:] {
		if !cfg.BigEndian {
			cfg.emitScalarLoadInt(w, base, off+curOff, ck, false, scratch)
			cfg.emitShiftLeft(w, curOff*8, scratch)
		} else {
			cfg.emitShiftLeft(w, ck*8, reg)
			cfg.emitScalarLoadInt(w, base, off+curOff, ck, false, scratch)
		}
		cfg.emitOrReg(w, scratch, reg)
		curOff += ck
	}
}

// emitStoreInt stores an integer or struct register chunk of 1..8 bytes from
// reg into off(base), decomposing non-power-of-two sizes into exact sub-stores.
func (cfg *ABIConfig) emitStoreInt(w io.Writer, reg, base string, off, size int64) {
	cfg.emitStoreIntAlign(w, reg, base, off, size, cfg.PtrSize)
}

func (cfg *ABIConfig) emitStoreIntAlign(w io.Writer, reg, base string, off, size, align int64) {
	maxWidth := cfg.PtrSize
	if cfg.StrictAlign {
		maxWidth = effAlign(off, align, cfg.PtrSize)
	}
	if (size == 1 || size == 2 || size == 4 || size == 8) && size <= maxWidth {
		cfg.emitScalarStoreInt(w, reg, base, off, size)
		return
	}
	chunks := decomposePow2Aligned(size, maxWidth)
	if !cfg.BigEndian {
		curOff := int64(0)
		for i, ck := range chunks {
			cfg.emitScalarStoreInt(w, reg, base, off+curOff, ck)
			curOff += ck
			if i+1 < len(chunks) {
				cfg.emitShiftRight(w, ck*8, reg)
			}
		}
	} else {
		curOff := size
		for i := len(chunks) - 1; i >= 0; i-- {
			ck := chunks[i]
			curOff -= ck
			cfg.emitScalarStoreInt(w, reg, base, off+curOff, ck)
			if i > 0 {
				cfg.emitShiftRight(w, ck*8, reg)
			}
		}
	}
}

// emitLoadFloat loads a 4-byte or 8-byte floating-point value from off(base)
// into floating-point register reg.
func (cfg *ABIConfig) emitLoadFloat(w io.Writer, base string, off, size int64, reg string) {
	if cfg.LoadFloat != nil && cfg.LoadFloat(w, base, off, size, reg) {
		return
	}
	op := cfg.asm.fmovD
	if size == 4 {
		op = cfg.asm.fmovS
	}
	fmt.Fprintf(w, "\t%s\t%d(%s), %s\n", op, off, base, reg)
}

// emitStoreFloat stores a 4-byte or 8-byte floating-point value from
// floating-point register reg into off(base).
func (cfg *ABIConfig) emitStoreFloat(w io.Writer, reg, base string, off, size int64) {
	if cfg.StoreFloat != nil && cfg.StoreFloat(w, reg, base, off, size) {
		return
	}
	op := cfg.asm.fmovD
	if size == 4 {
		op = cfg.asm.fmovS
	}
	fmt.Fprintf(w, "\t%s\t%s, %d(%s)\n", op, reg, off, base)
}

// emitAddr computes the effective address base+off into reg.
func (cfg *ABIConfig) emitAddr(w io.Writer, base string, off int64, reg string) {
	if cfg.asm.lea != "" {
		fmt.Fprintf(w, "\t%s\t%d(%s), %s\n", cfg.asm.lea, off, base, reg)
		return
	}
	fmt.Fprintf(w, "\t%s\t$%d, %s, %s\n", cfg.asm.add, off, base, reg)
}

// emitMovReg copies pointer-sized register src into dst.
func (cfg *ABIConfig) emitMovReg(w io.Writer, src, dst string) {
	fmt.Fprintf(w, "\t%s\t%s, %s\n", cfg.asm.mov, src, dst)
}

// emitSubReg subtracts register src from dst (dst -= src).
func (cfg *ABIConfig) emitSubReg(w io.Writer, src, dst string) {
	cfg.emitALUOp(w, cfg.asm.sub, src, dst)
}

// emitAddReg adds register src into dst (dst += src).
func (cfg *ABIConfig) emitAddReg(w io.Writer, src, dst string) {
	cfg.emitALUOp(w, cfg.asm.add, src, dst)
}

// emitPPC64AddSP adjusts R1 by delta bytes using signed 16-bit ADD steps so
// cmd/internal/obj/ppc64 never rewrites large offsets to ADDIS (which drops
// Spadj tracking, triggers auto-SPWRITE, and can overflow pctospadj) and never
// uses REGTMP (R31) or REGZERO (R0).
func emitPPC64AddSP(w io.Writer, delta int64) {
	if delta < 0 {
		for rem := -delta; rem > 0; {
			step := min(rem, 32768)
			fmt.Fprintf(w, "\tADD\t$-%d, R1\n", step)
			rem -= step
		}
		return
	}
	for rem := delta; rem > 0; {
		step := min(rem, 32752)
		fmt.Fprintf(w, "\tADD\t$%d, R1\n", step)
		rem -= step
	}
}

// emitPPC64AllocBackchain decrements R1 by size bytes and stores the caller's
// R1 at 0(R1) as the PowerPC64 ELFv2 / AIX XCOFF stack back-chain pointer,
// using a single MOVDU when size fits in a signed 16-bit displacement.
// Otherwise it uses tmpReg to hold the caller's R1.
func emitPPC64AllocBackchain(w io.Writer, tmpReg string, size int64) {
	if size <= 32760 {
		fmt.Fprintf(w, "\tMOVDU\tR1, -%d(R1)\n", size)
	} else {
		fmt.Fprintf(w, "\tMOVD\tR1, %s\n", tmpReg)
		emitPPC64AddSP(w, -size)
		fmt.Fprintf(w, "\tMOVD\t%s, 0(R1)\n", tmpReg)
	}
}

// emitSubSP decrements the architecture stack pointer by size bytes.
func (cfg *ABIConfig) emitSubSP(w io.Writer, size int64) {
	if cfg.asm.useAdjSP {
		fmt.Fprintf(w, "\tADJSP\t$%d\n", size)
		return
	}
	if cfg.asm.negSubSP {
		fmt.Fprintf(w, "\t%s\t$-%d, %s\n", cfg.asm.add, size, cfg.SPReg)
		return
	}
	fmt.Fprintf(w, "\t%s\t$%d, %s\n", cfg.asm.sub, size, cfg.SPReg)
}

// emitAddSP increments the architecture stack pointer by size bytes.
func (cfg *ABIConfig) emitAddSP(w io.Writer, size int64) {
	if cfg.asm.useAdjSP {
		fmt.Fprintf(w, "\tADJSP\t$-%d\n", size)
		return
	}
	fmt.Fprintf(w, "\t%s\t$%d, %s\n", cfg.asm.add, size, cfg.SPReg)
}

// emitLoadStackArg loads a C-caller stack argument at stackOff into reg
// during a C-to-Go callback trampoline.
func (cfg *ABIConfig) emitLoadStackArg(w io.Writer, stackOff, callerSPOff, size int64, reg string) {
	base, off := cfg.SPReg, callerSPOff+stackOff
	if cfg.StackArgAddr != nil {
		base, off = cfg.StackArgAddr(stackOff, callerSPOff)
	}
	cfg.emitLoadInt(w, base, off, size, false, reg)
}

// emitCallReg emits an indirect branch-and-link through register reg.
func (cfg *ABIConfig) emitCallReg(w io.Writer, reg string) {
	if cfg.asm.lea != "" {
		fmt.Fprintf(w, "\tCALL\t%s\n", reg)
	} else if cfg.Arch == "riscv64" {
		fmt.Fprintf(w, "\tJALR\tX1, (%s)\n", reg)
	} else {
		fmt.Fprintf(w, "\tCALL\t(%s)\n", reg)
	}
}

// emitCallIndirect calls the C function pointer stored at 0(base).
func (cfg *ABIConfig) emitCallIndirect(w io.Writer, base string) {
	if cfg.CallIndirect != nil {
		cfg.CallIndirect(w, base)
		return
	}
	reg := cmp.Or(cfg.IndCallReg, cfg.TmpReg)
	cfg.emitLoadInt(w, base, 0, cfg.PtrSize, false, reg)
	cfg.emitCallReg(w, reg)
}

// emitCallRuntime emits a direct call to an internal Go runtime helper symbol.
func (cfg *ABIConfig) emitCallRuntime(w io.Writer, sym string) {
	if cfg.HasFuncDesc && sym == "_cgo_topofstack" {
		sym = "__cgo_topofstack"
	}
	fmt.Fprintf(w, "\tCALL\t%s(SB)\n", sym)
	if (cfg.Arch == "ppc64" || cfg.Arch == "ppc64le") && sym != "_cgo_topofstack" && sym != "__cgo_topofstack" {
		// _cgo_wait_runtime_init_done, crosscall2, and _cgo_release_context
		// restore host CR via R0 in UNSTACK_AND_RESTORE_GO_TO_HOST_ABI, leaving
		// R0 non-zero. Re-zero R0 before any subsequent large-constant load or
		// large-offset addition that assembles to ORI/ORIS with R0.
		fmt.Fprintf(w, "\tXOR\tR0, R0\n")
	}
}

// emitCallSymbol emits a call to external C symbol sym.
func (cfg *ABIConfig) emitCallSymbol(w io.Writer, sym string, static bool) {
	if cfg.CallSymbol != nil {
		cfg.CallSymbol(w, sym, static)
		return
	}
	if cfg.IndirectSymCall && !static {
		// On PE, a //go:cgo_import_dynamic symbol is a pointer-sized slot
		// holding the function's address (an IAT entry for internal linking,
		// or a relocated data word for external linking; see initdynimport
		// in cmd/link/internal/ld/pe.go). Load the address and call indirectly.
		reg := cmp.Or(cfg.IndCallReg, cfg.TmpReg)
		fmt.Fprintf(w, "\t%s\t%s(SB), %s\n", cfg.asm.mov, sym, reg)
		cfg.emitCallReg(w, reg)
		return
	}
	fmt.Fprintf(w, "\tCALL\t%s(SB)\n", sym)
}

// emitCopyMem copies size bytes from srcOff(srcBase) to dstOff(dstBase) using
// alignment-safe power-of-two loads and stores through tmpReg.
func (cfg *ABIConfig) emitCopyMem(w io.Writer, srcBase string, srcOff int64, dstBase string, dstOff, size, align int64, tmpReg string) {
	ptrSize := cfg.PtrSize
	for off := int64(0); off < size; {
		maxWidth := ptrSize
		if cfg.StrictAlign {
			srcA := effAlign(srcOff+off, align, ptrSize)
			dstA := effAlign(dstOff+off, align, ptrSize)
			maxWidth = min(srcA, dstA)
		}
		chunk := maxPow2Chunk(size-off, maxWidth)
		cfg.emitScalarLoadInt(w, srcBase, srcOff+off, chunk, false, tmpReg)
		cfg.emitScalarStoreInt(w, tmpReg, dstBase, dstOff+off, chunk)
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
	regSize := p.Size - p.StackBytes
	return cfg.valueRegSlots(pt, regSize, p.IsFloat, p.Regs, types)
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
	goAlign := min(info.align, ptrSize)
	if goAlign < 1 {
		goAlign = 1
	}
	slots := make([]regSlot, len(regs))
	for k, r := range regs {
		fl := cfg.isFloatReg(r)
		chunkSz := min(ptrSize, size-int64(k)*ptrSize)
		if len(regs) == 1 {
			chunkSz = size
		}
		leftJustify := cfg.BigEndian && info.composite && !fl && chunkSz < ptrSize &&
			(cfg.LeftJustifyBEComposite || (cfg.LeftJustifyBETrailing && info.size > ptrSize))
		slots[k] = regSlot{
			reg:         r,
			offset:      int64(k) * ptrSize,
			size:        chunkSz,
			align:       goAlign,
			isFloat:     fl,
			signed:      !info.composite && !fl && (info.signed || (cfg.SignExtUint32 && chunkSz == 4)),
			leftJustify: leftJustify,
		}
	}
	return slots
}

func (cfg *ABIConfig) goTypeAlign(typeName string, types map[string]BindingType) int64 {
	if typeName == "" {
		return 1
	}
	a := min(cfg.inspectType(typeName, types).align, cfg.PtrSize)
	if a < 1 {
		return 1
	}
	return a
}

func (cfg *ABIConfig) paramTypeInfo(fn BindingFunc, idx int, types map[string]BindingType) typeInfo {
	var pt string
	if paramTypes := fn.Params; idx < len(paramTypes) {
		pt = paramTypes[idx]
	}
	return cfg.inspectType(pt, types)
}

// emitLoadSlot loads a single register piece from off+slot.offset(base) into slot.reg.
func (cfg *ABIConfig) emitLoadSlot(w io.Writer, base string, off int64, slot regSlot) {
	if slot.isFloat {
		cfg.emitLoadFloat(w, base, off+slot.offset, slot.size, slot.reg)
		return
	}
	cfg.emitLoadIntAlign(w, base, off+slot.offset, slot.size, slot.align, slot.signed, slot.reg)
	if slot.leftJustify {
		cfg.emitShiftLeft(w, (cfg.PtrSize-slot.size)*8, slot.reg)
	}
}

// emitStoreSlot stores a register piece into off+slot.offset(base).
func (cfg *ABIConfig) emitStoreSlot(w io.Writer, slot regSlot, base string, off int64) {
	if slot.isFloat {
		cfg.emitStoreFloat(w, slot.reg, base, off+slot.offset, slot.size)
		return
	}
	if slot.leftJustify {
		cfg.emitShiftRight(w, (cfg.PtrSize-slot.size)*8, slot.reg)
	}
	cfg.emitStoreIntAlign(w, slot.reg, base, off+slot.offset, slot.size, slot.align)
}

// complexFloatStackPad returns the byte offset of each float32 half of a
// complex64 stack argument within its PtrSize slot when
// ComplexFloatSplitStack is set.
func (cfg *ABIConfig) complexFloatStackPad() int64 {
	if cfg.BigEndian && !cfg.LeftJustifyBEFloatStack {
		return cfg.PtrSize - 4
	}
	return 0
}

// emitRISCTrampolinePrologue emits a NOFRAME Go-to-C trampoline prologue for
// link-register architectures and returns its matching epilogue. When 0(SP) is
// not occupied by C stack arguments or C ABI shadow space, it stores LinkReg at
// 0(SP) using Go's three-instruction sequence (store at -total(SP), decrement
// SP, re-store at 0(SP)) so Go traceback and .debug_frame unwinding see saved
// LR at 0(SP) with no window.
func (cfg *ABIConfig) emitRISCTrampolinePrologue(w io.Writer, lay FuncLayout) func(io.Writer) {
	ptrSize := cfg.PtrSize
	lrAtZero := !cfg.ShadowUsesSlot0 && (cfg.ShadowSpace > 0 || !lay.hasStackArgs())
	if lrAtZero {
		total := max(cfg.LRReserve, lay.StackSize)
		cfg.emitStoreInt(w, cfg.LinkReg, cfg.SPReg, -total, ptrSize)
		cfg.emitSubSP(w, total)
		cfg.emitStoreInt(w, cfg.LinkReg, cfg.SPReg, 0, ptrSize)
		cfg.emitMovReg(w, cfg.IntRegs[0], cfg.CalleeSavedReg)
		return func(w io.Writer) {
			cfg.emitLoadInt(w, cfg.SPReg, 0, ptrSize, false, cfg.LinkReg)
			cfg.emitAddSP(w, total)
		}
	}
	total := lay.StackSize + cfg.StackAlign
	lrOff := total - ptrSize
	cfg.emitSubSP(w, total)
	cfg.emitStoreInt(w, cfg.LinkReg, cfg.SPReg, lrOff, ptrSize)
	cfg.emitMovReg(w, cfg.IntRegs[0], cfg.CalleeSavedReg)
	return func(w io.Writer) {
		cfg.emitLoadInt(w, cfg.SPReg, lrOff, ptrSize, false, cfg.LinkReg)
		cfg.emitAddSP(w, total)
	}
}

// emitRISCCallbackPrologue saves AsmTmpReg (if any) and LinkReg on the stack
// and allocates LocalFrameOff+frameSize bytes for writeCallbackTrampoline.
func (cfg *ABIConfig) emitRISCCallbackPrologue(w io.Writer, frameSize int64) int64 {
	ptrSize := cfg.PtrSize
	inner := cfg.LocalFrameOff + frameSize
	if cfg.AsmTmpReg == "" || inner <= 2032 {
		if cfg.ShadowSpace == 0 {
			cfg.emitStoreInt(w, cfg.LinkReg, cfg.SPReg, -inner, ptrSize)
		}
		cfg.emitSubSP(w, inner)
		cfg.emitStoreInt(w, cfg.LinkReg, cfg.SPReg, cfg.ShadowSpace, ptrSize)
		if cfg.AsmTmpReg != "" {
			cfg.emitStoreInt(w, cfg.AsmTmpReg, cfg.SPReg, cfg.ShadowSpace+ptrSize, ptrSize)
		}
		return inner
	}
	outer := cfg.StackAlign
	cfg.emitSubSP(w, outer)
	cfg.emitStoreInt(w, cfg.AsmTmpReg, cfg.SPReg, 0, ptrSize)
	if cfg.ShadowSpace == 0 {
		cfg.emitStoreInt(w, cfg.LinkReg, cfg.SPReg, -inner, ptrSize)
	}
	cfg.emitSubSP(w, inner)
	cfg.emitStoreInt(w, cfg.LinkReg, cfg.SPReg, cfg.ShadowSpace, ptrSize)
	return outer + inner
}

// emitRISCCallbackEpilogue restores LinkReg, AsmTmpReg, and SPReg and returns.
func (cfg *ABIConfig) emitRISCCallbackEpilogue(w io.Writer, frameSize int64) {
	ptrSize := cfg.PtrSize
	inner := cfg.LocalFrameOff + frameSize
	if cfg.AsmTmpReg == "" || inner <= 2032 {
		if cfg.AsmTmpReg != "" {
			cfg.emitLoadInt(w, cfg.SPReg, cfg.ShadowSpace+ptrSize, ptrSize, false, cfg.AsmTmpReg)
		}
		cfg.emitLoadInt(w, cfg.SPReg, cfg.ShadowSpace, ptrSize, false, cfg.LinkReg)
		cfg.emitAddSP(w, inner)
		fmt.Fprintf(w, "\tRET\n")
		return
	}
	cfg.emitLoadInt(w, cfg.SPReg, cfg.ShadowSpace, ptrSize, false, cfg.LinkReg)
	cfg.emitAddSP(w, inner)
	cfg.emitLoadInt(w, cfg.SPReg, 0, ptrSize, false, cfg.AsmTmpReg)
	cfg.emitAddSP(w, cfg.StackAlign)
	fmt.Fprintf(w, "\tRET\n")
}

// emitTextHeader emits the TEXT directive for sym (wrapped in an XCOFF
// function descriptor when cfg.HasFuncDesc is set).
func (cfg *ABIConfig) emitTextHeader(w io.Writer, sym, flags string) {
	if cfg.HasFuncDesc {
		textSym := sym + "_text<>"
		fmt.Fprintf(w, "DATA %s+0(SB)/8, $%s(SB)\n", sym, textSym)
		fmt.Fprintf(w, "DATA %s+8(SB)/8, $TOC(SB)\n", sym)
		fmt.Fprintf(w, "DATA %s+16(SB)/8, $0\n", sym)
		fmt.Fprintf(w, "GLOBL %s(SB), NOPTR, $24\n", sym)
		fmt.Fprintf(w, "TEXT %s(SB), %s, $0-0\n", textSym, flags)
		return
	}
	fmt.Fprintf(w, "TEXT %s(SB), %s, $0-0\n", sym, flags)
}

// emitCallCrosscall2 emits the shared register-ABI callback sequence:
// _cgo_wait_runtime_init_done -> crosscall2(fn, &frame, frameSize, ctxt) -> _cgo_release_context.
func (cfg *ABIConfig) emitCallCrosscall2(w io.Writer, fnName string, frameSize, ctxtOff, fnDescOff int64) {
	if cfg.CallCrosscall2 != nil {
		cfg.CallCrosscall2(w, fnName, frameSize, ctxtOff)
		return
	}
	ptrSize := cfg.PtrSize
	arg0, arg1, arg2, arg3 := cfg.IntRegs[0], cfg.IntRegs[1], cfg.IntRegs[2], cfg.IntRegs[3]
	ret0 := cfg.IntRetRegs[0]
	cfg.emitCallRuntime(w, "_cgo_wait_runtime_init_done")
	cfg.emitStoreInt(w, ret0, cfg.SPReg, ctxtOff, ptrSize)
	cfg.emitMovReg(w, ret0, arg3)
	if cfg.Arch == "arm" {
		// Use MOVW $sym + double load so -dynlink GOT rewrites load directly into
		// arg0 without clobbering C callee-saved R9.
		fmt.Fprintf(w, "\tMOVW\t$·_cgo_fn_%s(SB), %s\n", fnName, arg0)
		fmt.Fprintf(w, "\tMOVW\t0(%s), %s\n", arg0, arg0)
		fmt.Fprintf(w, "\tMOVW\t0(%s), %s\n", arg0, arg0)
	} else if cfg.Arch == "amd64" {
		// Use LEAQ + double load so -dynlink GOT rewrites do not clobber C callee-saved R15.
		fmt.Fprintf(w, "\tLEAQ\t·_cgo_fn_%s(SB), %s\n", fnName, arg0)
		fmt.Fprintf(w, "\tMOVQ\t0(%s), %s\n", arg0, arg0)
		fmt.Fprintf(w, "\tMOVQ\t0(%s), %s\n", arg0, arg0)
	} else if cfg.HasFuncDesc {
		fmt.Fprintf(w, "\tMOVD\t·_cgo_fn_%s(SB), %s\n", fnName, arg0)
		fmt.Fprintf(w, "\tMOVD\t0(%s), %s\n", arg0, arg0)
		cfg.emitStoreInt(w, arg0, cfg.SPReg, fnDescOff, 8)
		cfg.emitStoreInt(w, "R2", cfg.SPReg, fnDescOff+8, 8)
		cfg.emitAddr(w, cfg.SPReg, fnDescOff, arg0)
	} else {
		fmt.Fprintf(w, "\t%s\t·_cgo_fn_%s(SB), %s\n", cfg.asm.mov, fnName, arg0)
		cfg.emitLoadInt(w, arg0, 0, ptrSize, false, arg0)
	}
	cfg.emitAddr(w, cfg.SPReg, cfg.LocalFrameOff, arg1)
	fmt.Fprintf(w, "\t%s\t$%d, %s\n", cfg.asm.mov, frameSize, arg2)
	cfg.emitCallRuntime(w, "crosscall2")
	cfg.emitLoadInt(w, cfg.SPReg, ctxtOff, ptrSize, false, arg0)
	cfg.emitCallRuntime(w, "_cgo_release_context")
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
	ptrSize := cfg.PtrSize
	lay := cfg.layout(fn, types)

	trampSym := fmt.Sprintf("_cgo%s_trampoline_%s", cPrefix, fn.Name)
	var epilogue func(io.Writer)
	if cfg.TrampolinePrologue != nil {
		cfg.emitTextHeader(w, trampSym, "NOSPLIT|NOFRAME")
		epilogue = cfg.TrampolinePrologue(w, lay)
	} else if cfg.LinkReg != "" {
		cfg.emitTextHeader(w, trampSym, "NOSPLIT|NOFRAME")
		epilogue = cfg.emitRISCTrampolinePrologue(w, lay)
	} else {
		cfg.emitTextHeader(w, trampSym, "NOSPLIT")
		// Hold the _cgo_arg struct pointer in CalleeSavedReg across the C call:
		// the C callee preserves it per the C ABI, and our caller (runtime.asmcgocall)
		// does not keep a live value in CalleeSavedReg, so we need not save/restore it.
		if cfg.Prologue != nil {
			epilogue = cfg.Prologue(w)
		} else {
			cfg.emitMovReg(w, cfg.IntRegs[0], cfg.CalleeSavedReg)
		}

		if lay.StackSize > 0 {
			cfg.emitSubSP(w, lay.StackSize)
		}
	}

	base := cfg.CalleeSavedReg

	if fn.MayCallback {
		if fn.Ret != "" && lay.Ret.Size > 0 {
			cfg.emitCallRuntime(w, "_cgo_topofstack")
			cfg.emitStoreInt(w, cfg.IntRetRegs[0], cfg.SPReg, lay.StkTopOff, ptrSize)
		}
		for i, p := range lay.Params {
			if p.PassByPtr {
				pAlign := min(cfg.paramTypeInfo(fn, i, types).align, ptrSize)
				cfg.emitCopyMem(w, base, p.GoOffset, cfg.SPReg, p.G0Off, p.Size, pAlign, cfg.TmpReg)
			}
		}
	}

	if lay.Ret.Indirect {
		retBase, retOff := base, lay.Ret.GoOffset
		if fn.MayCallback {
			retBase, retOff = cfg.SPReg, lay.Ret.G0Off
		}
		if lay.Ret.IndirectOnStack {
			cfg.emitAddr(w, retBase, retOff, cfg.TmpReg)
			cfg.emitStoreInt(w, cfg.TmpReg, cfg.SPReg, 0, ptrSize)
		} else {
			cfg.emitAddr(w, retBase, retOff, cfg.IndirectRetReg)
		}
	}

	for i, p := range lay.Params {
		srcBase, srcOff := base, p.GoOffset
		if fn.MayCallback && p.PassByPtr {
			srcBase, srcOff = cfg.SPReg, p.G0Off
		}
		if p.PassByPtr {
			if len(p.Regs) > 0 {
				cfg.emitAddr(w, srcBase, srcOff, p.Regs[0])
			} else {
				cfg.emitAddr(w, srcBase, srcOff, cfg.TmpReg)
				cfg.emitStoreInt(w, cfg.TmpReg, cfg.SPReg, p.StackOff, ptrSize)
			}
			continue
		}
		info := cfg.paramTypeInfo(fn, i, types)
		pAlign := min(info.align, ptrSize)
		if slots := cfg.paramRegSlots(fn, i, p, types); len(slots) > 0 {
			for _, slot := range slots {
				cfg.emitLoadSlot(w, srcBase, srcOff, slot)
			}
			if p.StackBytes > 0 {
				regBytes := p.Size - p.StackBytes
				cfg.emitCopyMem(w, srcBase, srcOff+regBytes, cfg.SPReg, p.StackOff, p.StackBytes, pAlign, cfg.TmpReg)
			}
			continue
		}
		if cfg.ComplexFloatSplitStack && info.isComplex && p.Size == 8 {
			// The float32 halves of a complex64 occupy separate stack slots.
			pad := cfg.complexFloatStackPad()
			cfg.emitCopyMem(w, srcBase, srcOff, cfg.SPReg, p.StackOff+pad, 4, pAlign, cfg.TmpReg)
			cfg.emitCopyMem(w, srcBase, srcOff+4, cfg.SPReg, p.StackOff+ptrSize+pad, 4, pAlign, cfg.TmpReg)
			continue
		}
		if (!info.composite || cfg.RightJustifyBECompositeStack) && (p.Size == 1 || p.Size == 2 || p.Size == 4 || p.Size == 8) && p.Size <= ptrSize {
			signed := !info.composite && !info.isFloat && (info.signed || (cfg.SignExtUint32 && p.Size == 4))
			cfg.emitLoadIntAlign(w, srcBase, srcOff, p.Size, pAlign, signed, cfg.TmpReg)
			storeSize := ptrSize
			if cfg.PackStackArgs || (cfg.LeftJustifyBEFloatStack && info.isFloat) {
				storeSize = p.Size
			}
			cfg.emitStoreInt(w, cfg.TmpReg, cfg.SPReg, p.StackOff, storeSize)
			continue
		}
		dstStackOff := p.StackOff
		if cfg.BigEndian && cfg.RightJustifyBECompositeStack && info.composite && p.Size < ptrSize {
			dstStackOff += ptrSize - p.Size
		}
		cfg.emitCopyMem(w, srcBase, srcOff, cfg.SPReg, dstStackOff, p.Size, pAlign, cfg.TmpReg)
	}

	if cfg.PreCall != nil {
		cfg.PreCall(cfg, w, fn, lay)
	}

	switch {
	case fn.Indirect:
		cfg.emitCallIndirect(w, base)
	case fn.CName == "":
		panic("BindingFunc.CName must be set")
	default:
		cfg.emitCallSymbol(w, fn.CName, fn.Static)
	}
	if lay.Ret.IndirectOnStack && cfg.SysVI386SretPop {
		// The C callee returned with RET $4, popping the 4-byte sret slot
		// without obj6's deltasp tracking it. Emit SUBL rather than ADJSP so
		// SP is restored without unbalancing deltasp.
		fmt.Fprintf(w, "\tSUBL\t$4, SP\n")
	}

	if fn.Ret != "" && lay.Ret.Size > 0 {
		retDstBase, retDstOff := base, lay.Ret.GoOffset
		if fn.MayCallback {
			retDstBase, retDstOff = cfg.SPReg, lay.Ret.G0Off
		}
		if !lay.Ret.Indirect {
			for _, slot := range cfg.retRegSlots(fn, lay.Ret, types) {
				cfg.emitStoreSlot(w, slot, retDstBase, retDstOff)
			}
		}
		if fn.MayCallback {
			cfg.emitCallRuntime(w, "_cgo_topofstack")
			cfg.emitLoadInt(w, cfg.SPReg, lay.StkTopOff, ptrSize, false, cfg.TmpReg)
			cfg.emitSubReg(w, cfg.TmpReg, cfg.IntRetRegs[0])
			cfg.emitAddReg(w, cfg.IntRetRegs[0], base)
			cfg.emitCopyMem(w, cfg.SPReg, lay.Ret.G0Off, base, lay.Ret.GoOffset, lay.Ret.Size, cfg.goTypeAlign(fn.Ret, types), cfg.TmpReg)
		}
	}

	if cfg.TrampolinePrologue == nil && cfg.LinkReg == "" && lay.StackSize > 0 {
		cfg.emitAddSP(w, lay.StackSize)
	}
	if epilogue != nil {
		epilogue(w)
	}
	fmt.Fprintf(w, "\tRET\n\n")
}

// writeCallbackTrampoline emits the C-callable export trampoline for an
// //export Go function. It spills C ABI register and stack arguments into a
// packed Go argument struct on the C stack, invokes Go via crosscall2, and
// loads the return value into C ABI return registers or the sret buffer.
func (cfg *ABIConfig) writeCallbackTrampoline(w io.Writer, fn BindingFunc, types map[string]BindingType) {
	lay := cfg.layout(fn, types)
	ptrSize := cfg.PtrSize

	structSize := ptrSize
	if fn.Ret != "" {
		structSize = lay.Ret.GoOffset + lay.Ret.Size
	} else if len(lay.Params) > 0 {
		last := lay.Params[len(lay.Params)-1]
		structSize = last.GoOffset + last.Size
	}
	retBufOff := int64(-1)
	if lay.Ret.Indirect {
		retBufOff = alignUp(structSize, ptrSize)
		structSize = retBufOff + ptrSize
	}
	ctxtOff := alignUp(structSize, ptrSize)
	structSize = ctxtOff + ptrSize
	fnDescOff := int64(-1)
	if cfg.HasFuncDesc {
		fnDescOff = alignUp(structSize, ptrSize)
		structSize = fnDescOff + 16
	}
	frameSize := max(16, alignUp(structSize, 16))

	var callerSPOff int64
	switch {
	case cfg.CallbackPrologue != nil:
		cfg.emitTextHeader(w, fn.Name, "NOSPLIT|NOFRAME")
		callerSPOff = cfg.CallbackPrologue(w, frameSize)
	case cfg.LinkReg != "":
		fmt.Fprintf(w, "TEXT %s(SB), NOSPLIT|NOFRAME, $0-0\n", fn.Name)
		callerSPOff = cfg.emitRISCCallbackPrologue(w, frameSize)
	default:
		totalFrame := cfg.LocalFrameOff + frameSize
		fmt.Fprintf(w, "TEXT %s(SB), NOSPLIT, $%d-0\n", fn.Name, totalFrame)
		callerSPOff = totalFrame
	}

	baseOff := cfg.LocalFrameOff

	if lay.Ret.Indirect {
		if lay.Ret.IndirectOnStack {
			cfg.emitLoadStackArg(w, 0, callerSPOff, ptrSize, cfg.TmpReg)
			cfg.emitStoreInt(w, cfg.TmpReg, cfg.SPReg, baseOff+retBufOff, ptrSize)
		} else {
			cfg.emitStoreInt(w, cfg.IndirectRetReg, cfg.SPReg, baseOff+retBufOff, ptrSize)
		}
	}

	for i, p := range lay.Params {
		info := cfg.paramTypeInfo(fn, i, types)
		pAlign := min(info.align, ptrSize)
		slots := cfg.paramRegSlots(fn, i, p, types)

		if p.PassByPtr {
			ptrReg := cfg.TmpReg
			copyReg := cfg.TmpReg2
			if len(slots) > 0 {
				ptrReg = slots[0].reg
				copyReg = cfg.TmpReg
			} else {
				cfg.emitLoadStackArg(w, p.StackOff, callerSPOff, ptrSize, ptrReg)
			}
			cfg.emitCopyMem(w, ptrReg, 0, cfg.SPReg, baseOff+p.GoOffset, p.Size, pAlign, copyReg)
		} else if len(slots) > 0 {
			for _, slot := range slots {
				cfg.emitStoreSlot(w, slot, cfg.SPReg, baseOff+p.GoOffset)
			}
			if p.StackBytes > 0 {
				regBytes := p.Size - p.StackBytes
				srcBase, srcOff := cfg.SPReg, callerSPOff+p.StackOff
				if cfg.StackArgAddr != nil {
					srcBase, srcOff = cfg.StackArgAddr(p.StackOff, callerSPOff)
				}
				cfg.emitCopyMem(w, srcBase, srcOff, cfg.SPReg, baseOff+p.GoOffset+regBytes, p.StackBytes, pAlign, cfg.TmpReg)
			}
		} else if cfg.ComplexFloatSplitStack && info.isComplex && p.Size == 8 {
			// The float32 halves of a complex64 occupy separate stack slots.
			pad := cfg.complexFloatStackPad()
			for k := range int64(2) {
				srcBase, srcOff := cfg.SPReg, callerSPOff+p.StackOff+k*ptrSize+pad
				if cfg.StackArgAddr != nil {
					srcBase, srcOff = cfg.StackArgAddr(p.StackOff+k*ptrSize+pad, callerSPOff)
				}
				cfg.emitCopyMem(w, srcBase, srcOff, cfg.SPReg, baseOff+p.GoOffset+k*4, 4, pAlign, cfg.TmpReg)
			}
		} else {
			srcStackOff := p.StackOff
			if cfg.BigEndian && p.Size < ptrSize && (!info.composite || cfg.RightJustifyBECompositeStack) &&
				!(cfg.LeftJustifyBEFloatStack && info.isFloat) {
				srcStackOff += ptrSize - p.Size
			}
			srcBase, srcOff := cfg.SPReg, callerSPOff+srcStackOff
			if cfg.StackArgAddr != nil {
				srcBase, srcOff = cfg.StackArgAddr(srcStackOff, callerSPOff)
			}
			cfg.emitCopyMem(w, srcBase, srcOff, cfg.SPReg, baseOff+p.GoOffset, p.Size, pAlign, cfg.TmpReg)
		}
	}

	// Zero pointer-containing return slots on the C stack so Go GC write
	// barriers in _cgoexp_* never observe uninitialized stack bytes.
	if fn.Ret != "" && lay.Ret.Size > 0 && cfg.inspectType(fn.Ret, types).hasPointers {
		zeroSrc := cfg.ZeroVal
		if zeroSrc == "" {
			fmt.Fprintf(w, "\t%s\t$0, %s\n", cfg.asm.mov, cfg.TmpReg)
			zeroSrc = cfg.TmpReg
		}
		for off := int64(0); off < lay.Ret.Size; {
			sz := maxPow2Chunk(lay.Ret.Size-off, ptrSize)
			cfg.emitStoreInt(w, zeroSrc, cfg.SPReg, baseOff+lay.Ret.GoOffset+off, sz)
			off += sz
		}
	}

	cfg.emitCallCrosscall2(w, fn.Name, frameSize, baseOff+ctxtOff, baseOff+fnDescOff)

	if fn.Ret != "" {
		if lay.Ret.Indirect {
			bufReg := cfg.TmpReg
			copyReg := cfg.TmpReg2
			if (cfg.IndirectRetInArg || lay.Ret.IndirectOnStack) && len(cfg.IntRetRegs) > 0 {
				bufReg = cfg.IntRetRegs[0]
				copyReg = cfg.TmpReg
			}
			cfg.emitLoadInt(w, cfg.SPReg, baseOff+retBufOff, ptrSize, false, bufReg)
			cfg.emitCopyMem(w, cfg.SPReg, baseOff+lay.Ret.GoOffset, bufReg, 0, lay.Ret.Size, cfg.goTypeAlign(fn.Ret, types), copyReg)
		} else {
			for _, slot := range cfg.retRegSlots(fn, lay.Ret, types) {
				cfg.emitLoadSlot(w, cfg.SPReg, baseOff+lay.Ret.GoOffset, slot)
			}
		}
	}

	switch {
	case cfg.CallbackEpilogue != nil:
		cfg.CallbackEpilogue(w, frameSize, lay.Ret.IndirectOnStack && cfg.SysVI386SretPop)
		fmt.Fprintf(w, "\n")
	case cfg.LinkReg != "":
		cfg.emitRISCCallbackEpilogue(w, frameSize)
		fmt.Fprintf(w, "\n")
	default:
		fmt.Fprintf(w, "\tRET\n\n")
	}
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
	ZeroVal:           "$0",
	asm: asmOps{
		mov:   "MOVQ",
		loadB: "MOVBLSX", loadBU: "MOVBLZX", loadH: "MOVWLSX", loadHU: "MOVWLZX", loadW: "MOVL", loadWU: "MOVL",
		storeB: "MOVB", storeH: "MOVW", storeW: "MOVL",
		fmovS: "MOVSS", fmovD: "MOVSD",
		shl: "SHLQ", shr: "SHRQ", or: "ORQ", add: "ADDQ", sub: "SUBQ", lea: "LEAQ",
		twoOpALU: true, twoOpShift: true, useAdjSP: true,
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
	StackArgAddr: func(stackOff, _ int64) (string, int64) {
		return "BP", 16 + stackOff
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
	if info.size == 0 {
		return ParamLoc{}
	}
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
	cfg.LocalFrameOff = 32
	cfg.IndirectSymCall = true
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

// classifyHFA reports whether composite type info is a Homogeneous
// Floating-Point Aggregate (1 to 4 identically sized float32 or float64
// members with no padding) per AAPCS32/AAPCS64, returning its element
// size and member count.
func classifyHFA(info typeInfo) (elemSize int64, count int, isHFA bool) {
	return classifyHFAMax(info, 4)
}

// classifyHFAMax reports whether composite type info is an HFA of 1 to maxCount
// identically sized float32 or float64 members with no padding.
func classifyHFAMax(info typeInfo, maxCount int) (elemSize int64, count int, isHFA bool) {
	if info.size <= 0 || !info.fieldsOK || len(info.fields) < 1 || len(info.fields) > maxCount {
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
		if _, count, ok := classifyHFA(info); ok {
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
		if _, count, ok := classifyHFA(info); ok {
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

// hfaCompositeRegSlots maps each member of a register-passed or
// register-returned HFA to its consecutive floating-point register.
func hfaCompositeRegSlots(_ *ABIConfig, info typeInfo, isFloat bool, regs []string) ([]regSlot, bool) {
	if !isFloat {
		return nil, false
	}
	elemSz := info.size / int64(len(regs))
	if info.fieldsOK && len(info.fields) > 0 {
		elemSz = info.fields[0].size
	}
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
	CompositeRegSlots: hfaCompositeRegSlots,
	SPReg:             "RSP",
	CalleeSavedReg:    "R19",
	TmpReg:            "R12",
	TmpReg2:           "R13",
	LRReserve:         16,
	ZeroVal:           "ZR",
	LocalFrameOff:     16,
	asm: asmOps{
		mov:   "MOVD",
		loadB: "MOVB", loadBU: "MOVBU", loadH: "MOVH", loadHU: "MOVHU", loadW: "MOVW", loadWU: "MOVWU",
		storeB: "MOVB", storeH: "MOVH", storeW: "MOVW",
		fmovS: "FMOVS", fmovD: "FMOVD",
		shl: "LSL", shr: "LSR", or: "ORR", add: "ADD", sub: "SUB",
	},
	TrampolinePrologue: emitARM64TrampolinePrologue,
	CallbackPrologue: func(w io.Writer, frameSize int64) int64 {
		total, small := arm64CallbackFrame(frameSize)
		fpOff := total - 16
		if small {
			fmt.Fprintf(w, "\tSTP.W\t(R30, R27), -%d(RSP)\n", total)
			fmt.Fprintf(w, "\tSTP\t(R29, R30), %d(RSP)\n", fpOff)
		} else {
			// total may not be encodable as an ADD/SUB immediate, in which
			// case the assembler materializes it in REGTMP (R27), which is
			// callee-saved in the C ABI. Save the frame record and R27 with
			// small pre-indexed stores before the large SP adjustment.
			fmt.Fprintf(w, "\tSTP.W\t(R29, R30), -16(RSP)\n")
			fmt.Fprintf(w, "\tMOVD.W\tR27, -16(RSP)\n")
			fmt.Fprintf(w, "\tSUB\t$%d, RSP\n", total-32)
			fmt.Fprintf(w, "\tMOVD\tR30, 0(RSP)\n")
		}
		fmt.Fprintf(w, "\tADD\t$%d, RSP, R29\n", fpOff)
		return total
	},
	CallbackEpilogue: func(w io.Writer, frameSize int64, _ bool) {
		total, small := arm64CallbackFrame(frameSize)
		if small {
			fmt.Fprintf(w, "\tMOVD\t%d(RSP), R29\n", total-16)
			fmt.Fprintf(w, "\tLDP.P\t%d(RSP), (R30, R27)\n", total)
		} else {
			// Undo the large SP adjustment (possibly via REGTMP) before
			// restoring R27 and the frame record.
			fmt.Fprintf(w, "\tADD\t$%d, RSP\n", total-32)
			fmt.Fprintf(w, "\tMOVD.P\t16(RSP), R27\n")
			fmt.Fprintf(w, "\tLDP.P\t16(RSP), (R29, R30)\n")
		}
		fmt.Fprintf(w, "\tRET\n")
	},
}

// arm64CallbackFrame returns the total C stack frame size of an arm64
// callback trampoline whose local Go argument area is frameSize bytes, and
// whether it uses the compact frame layout:
//
//	0(RSP)          saved LR (Go frame convention)
//	8(RSP)          saved R27 (compact layout only)
//	16(RSP)         local Go argument area (frameSize bytes)
//	total-32(RSP)   saved R27 (large layout only)
//	total-16(RSP)   frame record (R29, R30)
//
// The compact layout is used when total fits the signed, scaled 7-bit
// immediate of STP.W/LDP.P.
func arm64CallbackFrame(frameSize int64) (total int64, small bool) {
	total = 32 + frameSize
	if total <= 504 {
		return total, true
	}
	return 48 + frameSize, false
}

// 32-bit x86 (i386) C calling convention:
//   - SysV i386 psABI Section 2.2: https://gitlab.com/x86-psABIs/i386-ABI
//   - Windows x86 (__cdecl): https://learn.microsoft.com/en-us/cpp/cpp/argument-passing-and-naming-conventions

func win32AssignRet(cfg *ABIConfig, fn BindingFunc, info typeInfo, st *abiState) RetLoc {
	if info.composite && isWin64RegStructSize(info.size) {
		if info.size <= 4 {
			return RetLoc{Size: info.size, Regs: []string{"AX"}}
		}
		return RetLoc{Size: info.size, Regs: []string{"AX", "DX"}}
	}
	return defaultAssignRet(cfg, fn, info, st)
}

// sysv386AssignRet implements return values on Linux and NetBSD, whose C
// compilers return every composite through a hidden pointer that the callee
// pops, except _Complex float (complex64), which is not an aggregate in the
// SysV i386 psABI and is returned in AX:DX like a 64-bit integer.
// _Complex double (complex128) is returned through a hidden pointer.
func sysv386AssignRet(cfg *ABIConfig, fn BindingFunc, info typeInfo, st *abiState) RetLoc {
	if info.isComplex && info.size == 8 {
		return RetLoc{Size: info.size, Regs: []string{"AX", "DX"}}
	}
	return defaultAssignRet(cfg, fn, info, st)
}

// bsd386AssignRet implements struct return on FreeBSD and OpenBSD, whose C
// compilers default to -freg-struct-return on 386: as on Windows, composites
// of 1, 2, 4, or 8 bytes are returned in AX or AX:DX, except that a composite
// whose only element is a float or double is returned in F0. Other composites
// are returned through a hidden pointer, which the callee pops.
func bsd386AssignRet(cfg *ABIConfig, fn BindingFunc, info typeInfo, st *abiState) RetLoc {
	if info.composite && isWin64RegStructSize(info.size) && info.fieldsOK &&
		len(info.fields) == 1 && info.fields[0].isFloat && info.fields[0].size == info.size {
		return RetLoc{Size: info.size, Regs: []string{"F0"}, IsFloat: true}
	}
	return win32AssignRet(cfg, fn, info, st)
}

func i386AssignParam(cfg *ABIConfig, _ BindingFunc, _ int, info typeInfo, st *abiState) ParamLoc {
	if info.size == 0 {
		return ParamLoc{IsFloat: info.isFloat}
	}
	return cfg.assignStackParam(info.size, info.isFloat, st)
}

var i386ABI = &ABIConfig{
	Arch:               "386",
	PtrSize:            4,
	CLongSize:          4,
	StackAlign:         16,
	IndirectRetOnStack: true,
	SysVI386SretPop:    true,
	IntRetRegs:         []string{"AX", "DX"},
	FloatRetRegs:       []string{"F0"},
	AssignParam:        i386AssignParam,
	AssignRet:          sysv386AssignRet,
	SPReg:              "SP",
	CalleeSavedReg:     "BX",
	TmpReg:             "CX",
	TmpReg2:            "DX",
	ZeroVal:            "$0",
	asm: asmOps{
		mov:   "MOVL",
		loadB: "MOVBLSX", loadBU: "MOVBLZX", loadH: "MOVWLSX", loadHU: "MOVWLZX", loadW: "MOVL", loadWU: "MOVL",
		storeB: "MOVB", storeH: "MOVW", storeW: "MOVL",
		shl: "SHLL", shr: "SHRL", or: "ORL", add: "ADDL", sub: "SUBL", lea: "LEAL",
		twoOpALU: true, twoOpShift: true, useAdjSP: true,
	},
	StackArgAddr: func(stackOff, _ int64) (string, int64) {
		return "BP", 8 + stackOff
	},
	LoadFloat: func(w io.Writer, base string, off, size int64, reg string) bool {
		if reg != "F0" {
			return false
		}
		if size == 4 {
			fmt.Fprintf(w, "\tFMOVF\t%d(%s), F0\n", off, base)
		} else {
			fmt.Fprintf(w, "\tFMOVD\t%d(%s), F0\n", off, base)
		}
		return true
	},
	StoreFloat: func(w io.Writer, reg, base string, off, size int64) bool {
		if reg != "F0" {
			return false
		}
		if size == 4 {
			fmt.Fprintf(w, "\tFMOVFP\tF0, %d(%s)\n", off, base)
		} else {
			fmt.Fprintf(w, "\tFMOVDP\tF0, %d(%s)\n", off, base)
		}
		return true
	},
	Prologue: func(w io.Writer) func(w io.Writer) {
		fmt.Fprintf(w, "\tPUSHL\tBP\n")
		fmt.Fprintf(w, "\tMOVL\tSP, BP\n")
		fmt.Fprintf(w, "\tADJSP\t$8\n")
		fmt.Fprintf(w, "\tMOVL\tBX, -4(BP)\n")
		fmt.Fprintf(w, "\tMOVL\t8(BP), BX\n")
		return func(w io.Writer) {
			fmt.Fprintf(w, "\tMOVL\t-4(BP), BX\n")
			fmt.Fprintf(w, "\tADJSP\t$-8\n")
			fmt.Fprintf(w, "\tPOPL\tBP\n")
		}
	},
	CallbackPrologue: func(w io.Writer, frameSize int64) int64 {
		fmt.Fprintf(w, "\tPUSHL\tBP\n")
		fmt.Fprintf(w, "\tMOVL\tSP, BP\n")
		fmt.Fprintf(w, "\tADJSP\t$%d\n", frameSize+8)
		// Save C callee-saved BX in the 8-byte alignment padding: under -dynlink,
		// obj6 prepends LEAL _GLOBAL_OFFSET_TABLE_, BX before non-local CALLs.
		fmt.Fprintf(w, "\tMOVL\tBX, -4(BP)\n")
		return 0
	},
	CallbackEpilogue: func(w io.Writer, frameSize int64, sretPop bool) {
		fmt.Fprintf(w, "\tMOVL\t-4(BP), BX\n")
		fmt.Fprintf(w, "\tADJSP\t$-%d\n", frameSize+8)
		fmt.Fprintf(w, "\tPOPL\tBP\n")
		if sretPop {
			fmt.Fprintf(w, "\tBYTE\t$0xc2\n\tWORD\t$4\n")
		} else {
			fmt.Fprintf(w, "\tRET\n")
		}
	},
	CallCrosscall2: func(w io.Writer, fnName string, frameSize, ctxtOff int64) {
		fmt.Fprintf(w, "\tCALL\t_cgo_wait_runtime_init_done(SB)\n")
		fmt.Fprintf(w, "\tMOVL\tAX, %d(SP)\n", ctxtOff)
		fmt.Fprintf(w, "\tADJSP\t$16\n")
		fmt.Fprintf(w, "\tMOVL\tAX, 12(SP)\n")
		fmt.Fprintf(w, "\tMOVL\t·_cgo_fn_%s(SB), AX\n", fnName)
		fmt.Fprintf(w, "\tMOVL\t0(AX), AX\n")
		fmt.Fprintf(w, "\tMOVL\tAX, 0(SP)\n")
		fmt.Fprintf(w, "\tLEAL\t16(SP), AX\n")
		fmt.Fprintf(w, "\tMOVL\tAX, 4(SP)\n")
		fmt.Fprintf(w, "\tMOVL\t$%d, 8(SP)\n", frameSize)
		fmt.Fprintf(w, "\tCALL\tcrosscall2(SB)\n")
		fmt.Fprintf(w, "\tMOVL\t%d(SP), AX\n", 16+ctxtOff)
		fmt.Fprintf(w, "\tMOVL\tAX, 0(SP)\n")
		fmt.Fprintf(w, "\tCALL\t_cgo_release_context(SB)\n")
		fmt.Fprintf(w, "\tADJSP\t$-16\n")
	},
}

// 32-bit ARM C calling convention:
//   - AAPCS32 Sections 6 & 7.1 (VFP): https://github.com/ARM-software/abi-aa/blob/main/aapcs32/aapcs32.rst
//   - Android armeabi-v7a (soft-float): https://developer.android.com/ndk/guides/abis
//
// android/arm and openbsd/arm always use the base (soft-float) procedure
// call standard, which passes floating-point values in core registers.
// Other targets use the VFP variant unless GOARM selects soft float, since
// code built for a CPU without VFP cannot use VFP registers.

// goarmSoftFloat reports whether GOARM selects soft float (GOARM=5, or a
// ",softfloat" suffix). It reads the environment at each call, rather than
// using buildcfg.GOARM computed at startup, so tests can change GOARM.
func goarmSoftFloat() bool {
	v := os.Getenv("GOARM")
	if v == "" {
		return buildcfg.GOARM.SoftFloat
	}
	if strings.HasSuffix(v, ",softfloat") {
		return true
	}
	if strings.HasSuffix(v, ",hardfloat") {
		return false
	}
	return v == "5"
}

func arm32AssignRet(cfg *ABIConfig, fn BindingFunc, info typeInfo, st *abiState) RetLoc {
	if info.composite {
		if elemSz, count, ok := classifyHFA(info); ok {
			if elemSz == 8 {
				return RetLoc{Size: info.size, Regs: slices.Clone(cfg.FloatRetRegs[:count]), IsFloat: true}
			}
			regs := make([]string, count)
			for k := range count {
				regs[k] = fmt.Sprintf("S%d", k)
			}
			return RetLoc{Size: info.size, Regs: regs, IsFloat: true}
		}
	}
	if info.isFloat && info.size == 4 {
		return RetLoc{Size: 4, Regs: []string{"S0"}, IsFloat: true}
	}
	return defaultAssignRet(cfg, fn, info, st)
}

func arm32AssignParam(cfg *ABIConfig, _ BindingFunc, _ int, info typeInfo, st *abiState) ParamLoc {
	if info.size == 0 {
		return ParamLoc{}
	}
	if len(cfg.FloatRegs) > 0 {
		elemSz, count, isVFP := info.size, 1, info.isFloat && (info.size == 4 || info.size == 8)
		if !isVFP && info.composite {
			elemSz, count, isVFP = classifyHFA(info)
		}
		if isVFP {
			step := int(elemSz / 4)
			lanes := count * step
			mask := (uint16(1) << lanes) - 1
			for s := 0; s <= 16-lanes; s += step {
				if st.vfpMask&(mask<<s) == 0 {
					st.vfpMask |= mask << s
					st.floatIdx += count
					regs := make([]string, count)
					for k := range count {
						if elemSz == 4 {
							regs[k] = fmt.Sprintf("S%d", s+k)
						} else {
							regs[k] = fmt.Sprintf("F%d", s/2+k)
						}
					}
					return ParamLoc{Size: info.size, IsFloat: true, Regs: regs}
				}
			}
			st.vfpMask = 0xffff
			if elemSz == 8 {
				st.stackOff = alignUp(st.stackOff, 8)
			}
			return cfg.assignStackParam(info.size, true, st)
		}
	}

	need8 := (!info.composite && info.size == 8) || info.align == 8
	if need8 && st.intIdx < len(cfg.IntRegs) && st.intIdx%2 != 0 {
		st.intIdx++
	}
	numRegs := int((info.size + cfg.PtrSize - 1) / cfg.PtrSize)
	if st.intIdx < len(cfg.IntRegs) && st.intIdx+numRegs > len(cfg.IntRegs) && info.composite && st.stackOff == 0 {
		// AAPCS32 Section 6.1.1 Rule C.5: split a composite across the remaining
		// core registers R[intIdx]..R3 and the stack starting at 0(SP), but only
		// if no argument has been placed on the stack yet (NSAA == SP). Otherwise
		// Rules C.6-C.8 below place the composite wholly on the stack.
		regs := slices.Clone(cfg.IntRegs[st.intIdx:])
		regBytes := int64(len(regs)) * cfg.PtrSize
		stackBytes := info.size - regBytes
		st.intIdx = len(cfg.IntRegs)
		off := st.stackOff
		st.stackOff += alignUp(stackBytes, cfg.PtrSize)
		return ParamLoc{Size: info.size, Regs: regs, StackOff: off, StackBytes: stackBytes}
	}
	if st.intIdx+numRegs > len(cfg.IntRegs) {
		st.intIdx = len(cfg.IntRegs)
		if need8 {
			st.stackOff = alignUp(st.stackOff, 8)
		}
		return cfg.assignStackParam(info.size, false, st)
	}
	return cfg.assignIntParam(info, st)
}

// emitARM32TrampolinePrologue emits a NOFRAME 32-bit arm Go-to-C trampoline
// prologue and returns its matching epilogue, keeping R13 8-byte aligned and
// atomically storing R14 at 0(R13) via MOVW.W/MOVW.P whenever no C stack
// arguments occupy 0(R13).
func emitARM32TrampolinePrologue(w io.Writer, lay FuncLayout) func(io.Writer) {
	if !lay.hasStackArgs() {
		total := max(int64(8), lay.StackSize)
		if total <= 4095 {
			fmt.Fprintf(w, "\tMOVW.W\tR14, -%d(R13)\n", total)
		} else {
			fmt.Fprintf(w, "\tSUB\t$%d, R13\n", total)
			fmt.Fprintf(w, "\tMOVW\tR14, 0(R13)\n")
		}
		fmt.Fprintf(w, "\tMOVW\tR0, R4\n")
		return func(w io.Writer) {
			if total <= 4095 {
				fmt.Fprintf(w, "\tMOVW.P\t%d(R13), R14\n", total)
			} else {
				fmt.Fprintf(w, "\tMOVW\t0(R13), R14\n")
				fmt.Fprintf(w, "\tADD\t$%d, R13\n", total)
			}
		}
	}
	// AAPCS32 places 5th+ arguments at 0(R13), so save R14 above them.
	total := lay.StackSize + 8
	lrOff := total - 4
	fmt.Fprintf(w, "\tSUB\t$%d, R13\n", total)
	fmt.Fprintf(w, "\tMOVW\tR14, %d(R13)\n", lrOff)
	fmt.Fprintf(w, "\tMOVW\tR0, R4\n")
	return func(w io.Writer) {
		fmt.Fprintf(w, "\tMOVW\t%d(R13), R14\n", lrOff)
		fmt.Fprintf(w, "\tADD\t$%d, R13\n", total)
	}
}

var armABI = &ABIConfig{
	Arch:              "arm",
	PtrSize:           4,
	CLongSize:         4,
	StackAlign:        8,
	IntRegs:           []string{"R0", "R1", "R2", "R3"},
	FloatRegs:         []string{"F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7"},
	IntRetRegs:        []string{"R0", "R1"},
	FloatRetRegs:      []string{"F0", "F1", "F2", "F3"},
	MaxRegStructSize:  4,
	IndirectRetReg:    "R0",
	IndirectRetInArg:  true,
	AssignParam:       arm32AssignParam,
	AssignRet:         arm32AssignRet,
	CompositeRegSlots: hfaCompositeRegSlots,
	SPReg:             "R13",
	CalleeSavedReg:    "R4",
	TmpReg:            "R12",
	TmpReg2:           "R8",
	LRReserve:         8,
	LocalFrameOff:     8,
	StrictAlign:       true,
	asm: asmOps{
		mov:   "MOVW",
		loadB: "MOVB", loadBU: "MOVBU", loadH: "MOVH", loadHU: "MOVHU", loadW: "MOVW", loadWU: "MOVW",
		storeB: "MOVB", storeH: "MOVH", storeW: "MOVW",
		fmovS: "MOVF", fmovD: "MOVD",
		shl: "SLL", shr: "SRL", or: "ORR", add: "ADD", sub: "SUB",
		twoOpShift: true,
	},
	TrampolinePrologue: emitARM32TrampolinePrologue,
	CallbackPrologue: func(w io.Writer, frameSize int64) int64 {
		// Save callee-saved R8 (used as TmpReg2) alongside R11 (REGTMP) and R14 (LR).
		total := 16 + frameSize
		if total <= 4095 {
			fmt.Fprintf(w, "\tMOVW.W\tR14, -%d(R13)\n", total)
			fmt.Fprintf(w, "\tMOVW\tR8, 4(R13)\n")
			fmt.Fprintf(w, "\tMOVW\tR11, %d(R13)\n", 8+frameSize)
			return total
		}
		fmt.Fprintf(w, "\tSUB\t$8, R13\n")
		fmt.Fprintf(w, "\tMOVW\tR11, 0(R13)\n")
		fmt.Fprintf(w, "\tMOVW\tR8, 4(R13)\n")
		fmt.Fprintf(w, "\tSUB\t$%d, R13\n", 8+frameSize)
		fmt.Fprintf(w, "\tMOVW\tR14, 0(R13)\n")
		return total
	},
	CallbackEpilogue: func(w io.Writer, frameSize int64, _ bool) {
		total := 16 + frameSize
		if total <= 4095 {
			fmt.Fprintf(w, "\tMOVW\t%d(R13), R11\n", 8+frameSize)
			fmt.Fprintf(w, "\tMOVW\t4(R13), R8\n")
			fmt.Fprintf(w, "\tMOVW.P\t%d(R13), R14\n", total)
			fmt.Fprintf(w, "\tRET\n")
			return
		}
		fmt.Fprintf(w, "\tMOVW\t0(R13), R14\n")
		fmt.Fprintf(w, "\tADD\t$%d, R13\n", 8+frameSize)
		fmt.Fprintf(w, "\tMOVW\t4(R13), R8\n")
		fmt.Fprintf(w, "\tMOVW\t0(R13), R11\n")
		fmt.Fprintf(w, "\tADD\t$8, R13\n")
		fmt.Fprintf(w, "\tRET\n")
	},
	IsFloatReg: func(reg string) bool {
		return len(reg) >= 2 && reg[0] == 'S'
	},
	LoadFloat: func(w io.Writer, base string, off, _ int64, reg string) bool {
		if len(reg) < 2 || reg[0] != 'S' {
			return false
		}
		sIdx, _ := strconv.Atoi(reg[1:])
		if sIdx%2 == 0 {
			fmt.Fprintf(w, "\tMOVF\t%d(%s), F%d\n", off, base, sIdx/2)
		} else {
			// VMOV s(2n+1), R12 (transfer R12 to odd VFP single register).
			fmt.Fprintf(w, "\tMOVW\t%d(%s), R12\n", off, base)
			fmt.Fprintf(w, "\tWORD\t$0x%08x\n", 0xee00ca90|(uint32(sIdx/2)<<16))
		}
		return true
	},
	StoreFloat: func(w io.Writer, reg, base string, off, _ int64) bool {
		if len(reg) < 2 || reg[0] != 'S' {
			return false
		}
		sIdx, _ := strconv.Atoi(reg[1:])
		if sIdx%2 == 0 {
			fmt.Fprintf(w, "\tMOVF\tF%d, %d(%s)\n", sIdx/2, off, base)
		} else {
			// VMOV R12, s(2n+1) (transfer odd VFP single register to R12).
			fmt.Fprintf(w, "\tWORD\t$0x%08x\n", 0xee10ca90|(uint32(sIdx/2)<<16))
			fmt.Fprintf(w, "\tMOVW\tR12, %d(%s)\n", off, base)
		}
		return true
	},
}

// 32-bit MIPS O32 C calling convention:
// "System V ABI: MIPS RISC Processor Supplement", 3rd Edition, Chapter 3.

func mips32AssignParam(cfg *ABIConfig, _ BindingFunc, _ int, info typeInfo, st *abiState) ParamLoc {
	if info.size == 0 {
		return ParamLoc{}
	}
	nWords := int((info.size + 3) / 4)
	need8 := (!info.composite && info.size == 8) || info.align == 8
	if need8 && st.intIdx < 4 && st.intIdx%2 != 0 {
		st.intIdx++
	}
	loc := ParamLoc{Size: info.size, IsFloat: info.isFloat}
	// MIPS O32 only passes the first two floating-point arguments in F12/F14,
	// and only while no preceding non-float (or hidden sret) argument has been
	// seen. Once F14 is assigned or any non-float argument is encountered, all
	// subsequent arguments (including floats) are passed in GPRs or on the stack.
	if len(cfg.FloatRegs) > 0 && !st.fprsDisabled && info.isFloat && st.intIdx < 4 {
		switch {
		case info.size == 4 && st.intIdx == 0:
			loc.Regs = []string{"F12"}
			st.intIdx = 1
			st.floatIdx++
		case info.size == 4 && st.intIdx == 1:
			loc.Regs = []string{"F14"}
			st.intIdx = 2
			st.floatIdx++
			st.fprsDisabled = true
		case info.size == 4 && st.intIdx == 2:
			loc.Regs = []string{"F14"}
			st.intIdx = 3
			st.floatIdx++
			st.fprsDisabled = true
		case info.size == 8 && st.intIdx == 0:
			loc.Regs = []string{"F12"}
			st.intIdx = 2
			st.floatIdx++
		case info.size == 8 && st.intIdx == 2:
			loc.Regs = []string{"F14"}
			st.intIdx = 4
			st.floatIdx++
			st.fprsDisabled = true
		default:
			st.fprsDisabled = true
		}
	}
	if len(loc.Regs) == 0 {
		st.fprsDisabled = true
		loc.IsFloat = false
		if st.intIdx+nWords <= len(cfg.IntRegs) {
			loc.Regs = slices.Clone(cfg.IntRegs[st.intIdx : st.intIdx+nWords])
			st.intIdx += nWords
		} else if st.intIdx < len(cfg.IntRegs) && info.composite {
			// MIPS O32: aggregates starting in R4..R7 split across remaining GPRs and 16(R29)+.
			loc.Regs = slices.Clone(cfg.IntRegs[st.intIdx:])
			regBytes := int64(len(loc.Regs)) * 4
			stackBytes := info.size - regBytes
			st.intIdx = len(cfg.IntRegs)
			loc.StackOff = st.stackOff
			loc.StackBytes = stackBytes
			st.stackOff += alignUp(stackBytes, 4)
		} else {
			st.intIdx = len(cfg.IntRegs)
			if need8 {
				st.stackOff = alignUp(st.stackOff, 8)
			}
			loc.StackOff = st.stackOff
			st.stackOff += int64(nWords) * 4
		}
	}
	return loc
}

// mips32AssignRet returns _Complex float and _Complex double values in F0
// and F2 (one per half) with hard float, and in R2, R3 (complex64) or
// R2..R5 (complex128) with soft float. O32 returns every other composite,
// including structs of two floats or doubles, through a hidden pointer.
func mips32AssignRet(cfg *ABIConfig, fn BindingFunc, info typeInfo, st *abiState) RetLoc {
	if info.isComplex {
		if len(cfg.FloatRetRegs) >= 2 {
			return RetLoc{Size: info.size, IsFloat: true, Regs: slices.Clone(cfg.FloatRetRegs[:2])}
		}
		if info.size == 8 {
			return RetLoc{Size: info.size, Regs: []string{"R2", "R3"}}
		}
		return RetLoc{Size: info.size, Regs: []string{"R2", "R3", "R4", "R5"}}
	}
	return defaultAssignRet(cfg, fn, info, st)
}

func newMIPS32ABI(arch string, bigEndian bool) *ABIConfig {
	return &ABIConfig{
		Arch:                   arch,
		PtrSize:                4,
		CLongSize:              4,
		StackAlign:             8,
		ShadowSpace:            16,
		IntRegs:                []string{"R4", "R5", "R6", "R7"},
		FloatRegs:              []string{"F12", "F14"},
		IntRetRegs:             []string{"R2", "R3"},
		FloatRetRegs:           []string{"F0", "F2"},
		MaxRegStructSize:       0,
		IndirectRetReg:         "R4",
		IndirectRetInArg:       true,
		AssignParam:            mips32AssignParam,
		AssignRet:              mips32AssignRet,
		CompositeRegSlots:      hfaCompositeRegSlots,
		SPReg:                  "R29",
		CalleeSavedReg:         "R16",
		TmpReg:                 "R8",
		TmpReg2:                "R9",
		IndCallReg:             "R25",
		LinkReg:                "R31",
		AsmTmpReg:              "R23",
		ShadowUsesSlot0:        true,
		ZeroVal:                "R0",
		LocalFrameOff:          24,
		BigEndian:              bigEndian,
		LeftJustifyBEComposite: bigEndian,
		StrictAlign:            true,
		asm: asmOps{
			mov:   "MOVW",
			loadB: "MOVB", loadBU: "MOVBU", loadH: "MOVH", loadHU: "MOVHU", loadW: "MOVW", loadWU: "MOVW",
			storeB: "MOVB", storeH: "MOVH", storeW: "MOVW",
			fmovS: "MOVF", fmovD: "MOVD",
			shl: "SLL", shr: "SRL", or: "OR", add: "ADDU", sub: "SUBU",
			negSubSP: true,
		},
		CallSymbol: func(w io.Writer, sym string, _ bool) {
			// Load $t9 (R25) with target address per MIPS PIC (-mabicalls) ABI.
			fmt.Fprintf(w, "\tMOVW\t$%s(SB), R25\n", sym)
			fmt.Fprintf(w, "\tCALL\t(R25)\n")
		},
	}
}

var mipsABI = newMIPS32ABI("mips", true)
var mipsleABI = newMIPS32ABI("mipsle", false)

// 64-bit PowerPC C calling conventions:
//   - Linux ELFv2: https://openpowerfoundation.org/specifications/64bitelfabi/
//   - AIX 64-bit XCOFF: https://www.ibm.com/docs/en/aix/7.3?topic=overview-subroutine-linkage-convention

// ppc64AssignRet returns _Complex float and _Complex double values in F1 and
// F2 on both ELFv2 and AIX, and ELFv2 homogeneous floating-point aggregates
// of up to 8 members in F1..F8. AIX returns every other composite through a
// hidden pointer.
func ppc64AssignRet(cfg *ABIConfig, fn BindingFunc, info typeInfo, st *abiState) RetLoc {
	if info.isComplex {
		return RetLoc{Size: info.size, Regs: slices.Clone(cfg.FloatRetRegs[:2]), IsFloat: true}
	}
	if !cfg.HasFuncDesc && info.composite {
		if _, count, ok := classifyHFAMax(info, 8); ok {
			return RetLoc{Size: info.size, Regs: slices.Clone(cfg.FloatRetRegs[:count]), IsFloat: true}
		}
	}
	return defaultAssignRet(cfg, fn, info, st)
}

func ppc64AssignParam(cfg *ABIConfig, _ BindingFunc, _ int, info typeInfo, st *abiState) ParamLoc {
	if info.size == 0 {
		return ParamLoc{}
	}
	nSlots := (info.size + 7) / 8
	slot := st.intIdx

	loc := ParamLoc{Size: info.size, IsFloat: info.isFloat}
	if info.isComplex {
		// Both ELFv2 and AIX pass _Complex float and _Complex double as two
		// separate float or double arguments: each part takes the next FPR
		// and its own doubleword of the parameter save area.
		elemSz := info.size / 2
		st.intIdx = slot + 2
		loc.IsFloat = true
		switch {
		case st.floatIdx+2 <= len(cfg.FloatRegs):
			loc.Regs = slices.Clone(cfg.FloatRegs[st.floatIdx : st.floatIdx+2])
			st.floatIdx += 2
		case st.floatIdx < len(cfg.FloatRegs):
			// The real part takes F13 and the imaginary part the second
			// doubleword, where a float is right-justified on ELFv2 BE.
			loc.Regs = []string{cfg.FloatRegs[st.floatIdx]}
			st.floatIdx = len(cfg.FloatRegs)
			loc.StackOff = cfg.ShadowSpace + int64(slot+1)*8
			if elemSz == 4 {
				loc.StackOff += cfg.complexFloatStackPad()
			}
			loc.StackBytes = elemSz
			st.stackOff = max(st.stackOff, cfg.ShadowSpace+int64(slot+2)*8)
		default:
			// ComplexFloatSplitStack places the parts of a complex64 in
			// separate doublewords.
			loc.StackOff = cfg.ShadowSpace + int64(slot)*8
			st.stackOff = max(st.stackOff, loc.StackOff+16)
		}
		return loc
	}
	if !cfg.HasFuncDesc && info.composite {
		if elemSz, count, ok := classifyHFAMax(info, 8); ok {
			st.intIdx = slot + int(nSlots)
			loc.IsFloat = true
			if st.floatIdx+count <= len(cfg.FloatRegs) {
				loc.Regs = slices.Clone(cfg.FloatRegs[st.floatIdx : st.floatIdx+count])
				st.floatIdx += count
				return loc
			}
			if st.floatIdx < len(cfg.FloatRegs) {
				avail := len(cfg.FloatRegs) - st.floatIdx
				loc.Regs = slices.Clone(cfg.FloatRegs[st.floatIdx:])
				st.floatIdx = len(cfg.FloatRegs)
				loc.StackOff = cfg.ShadowSpace + int64(slot)*8 + int64(avail)*elemSz
				loc.StackBytes = int64(count-avail) * elemSz
				st.stackOff = max(st.stackOff, cfg.ShadowSpace+int64(slot+int(nSlots))*8)
				return loc
			}
			loc.StackOff = cfg.ShadowSpace + int64(slot)*8
			st.stackOff = max(st.stackOff, loc.StackOff+nSlots*8)
			return loc
		}
	}
	if !cfg.HasFuncDesc && info.align >= 16 && slot%2 != 0 {
		// ELFv2 starts an aggregate with 16-byte alignment (other than
		// a homogeneous floating-point aggregate) at an even doubleword of
		// the parameter save area, skipping an odd GPR. AIX does not.
		slot++
	}
	st.intIdx = slot + int(nSlots)
	switch {
	case info.isFloat && st.floatIdx < len(cfg.FloatRegs):
		loc.Regs = []string{cfg.FloatRegs[st.floatIdx]}
		st.floatIdx++
	case !info.isFloat && slot+int(nSlots) <= len(cfg.IntRegs):
		loc.Regs = slices.Clone(cfg.IntRegs[slot : slot+int(nSlots)])
	case !info.isFloat && slot < len(cfg.IntRegs) && info.composite:
		// Split aggregate straddling R3..R10 and the parameter save area.
		loc.Regs = slices.Clone(cfg.IntRegs[slot:])
		regBytes := int64(len(loc.Regs)) * 8
		loc.StackOff = cfg.ShadowSpace + int64(len(cfg.IntRegs))*8
		loc.StackBytes = info.size - regBytes
		st.stackOff = max(st.stackOff, cfg.ShadowSpace+int64(slot+int(nSlots))*8)
	default:
		loc.StackOff = cfg.ShadowSpace + int64(slot)*8
		st.stackOff = max(st.stackOff, loc.StackOff+nSlots*8)
	}
	return loc
}

func newPPC64ABI(arch string, bigEndian bool, shadowSpace int64, isAIX bool) *ABIConfig {
	emitAIXDescCall := func(w io.Writer, descSrc string) {
		fmt.Fprintf(w, "\tMOVD\tR2, 40(R1)\n")
		fmt.Fprintf(w, "\tMOVD\t%s, R12\n", descSrc)
		fmt.Fprintf(w, "\tMOVD\t0(R12), R0\n")
		fmt.Fprintf(w, "\tMOVD\t8(R12), R2\n")
		fmt.Fprintf(w, "\tMOVD\tR0, CTR\n")
		fmt.Fprintf(w, "\tBL\t(CTR)\n")
		fmt.Fprintf(w, "\tMOVD\t40(R1), R2\n")
		fmt.Fprintf(w, "\tXOR\tR0, R0\n")
	}
	maxRegStructSize := int64(16)
	floatRetRegs := []string{"F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8"}
	if isAIX {
		// AIX returns every composite except _Complex float and
		// _Complex double through a hidden pointer.
		maxRegStructSize = 0
		floatRetRegs = []string{"F1", "F2"}
	}
	localFrameOff := shadowSpace + 64
	return &ABIConfig{
		Arch:                   arch,
		PtrSize:                8,
		CLongSize:              8,
		StackAlign:             16,
		ShadowSpace:            shadowSpace,
		MinParamSaveArea:       64,
		IntRegs:                []string{"R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"},
		FloatRegs:              []string{"F1", "F2", "F3", "F4", "F5", "F6", "F7", "F8", "F9", "F10", "F11", "F12", "F13"},
		IntRetRegs:             []string{"R3", "R4"},
		FloatRetRegs:           floatRetRegs,
		MaxRegStructSize:       maxRegStructSize,
		IndirectRetReg:         "R3",
		IndirectRetInArg:       true,
		AssignParam:            ppc64AssignParam,
		AssignRet:              ppc64AssignRet,
		CompositeRegSlots:      hfaCompositeRegSlots,
		SPReg:                  "R1",
		CalleeSavedReg:         "R14",
		TmpReg:                 "R11",
		TmpReg2:                "R12",
		ZeroVal:                "R0",
		LocalFrameOff:          localFrameOff,
		BigEndian:              bigEndian,
		LeftJustifyBEComposite: isAIX,
		LeftJustifyBETrailing:  !isAIX && bigEndian,
		// ELFv2 BE right-justifies sub-doubleword composites and floats
		// in their stack doubleword; AIX left-justifies both.
		RightJustifyBECompositeStack: !isAIX && bigEndian,
		LeftJustifyBEFloatStack:      isAIX,
		ComplexFloatSplitStack:       true,
		HasFuncDesc:                  isAIX,
		asm: asmOps{
			mov:   "MOVD",
			loadB: "MOVB", loadBU: "MOVBZ", loadH: "MOVH", loadHU: "MOVHZ", loadW: "MOVW", loadWU: "MOVWZ",
			storeB: "MOVB", storeH: "MOVH", storeW: "MOVW",
			fmovS: "FMOVS", fmovD: "FMOVD",
			shl: "SLD", shr: "SRD", or: "OR", add: "ADD", sub: "SUB",
			alignDisp4: true,
		},
		TrampolinePrologue: func(w io.Writer, lay FuncLayout) func(io.Writer) {
			total := lay.StackSize
			fmt.Fprintf(w, "\tMOVD\tLR, R0\n")
			fmt.Fprintf(w, "\tMOVD\tR0, 16(R1)\n")
			emitPPC64AllocBackchain(w, "R11", total)
			fmt.Fprintf(w, "\tXOR\tR0, R0\n")
			if !isAIX {
				fmt.Fprintf(w, "\tMOVD\tR2, 24(R1)\n")
			}
			fmt.Fprintf(w, "\tMOVD\tR3, R14\n")
			return func(w io.Writer) {
				if !isAIX {
					fmt.Fprintf(w, "\tMOVD\t24(R1), R2\n")
				}
				emitPPC64AddSP(w, total)
				fmt.Fprintf(w, "\tMOVD\t16(R1), R0\n")
				fmt.Fprintf(w, "\tMOVD\tR0, LR\n")
				fmt.Fprintf(w, "\tXOR\tR0, R0\n")
			}
		},
		CallIndirect: func(w io.Writer, base string) {
			if isAIX {
				emitAIXDescCall(w, fmt.Sprintf("0(%s)", base))
				return
			}
			fmt.Fprintf(w, "\tMOVD\tR2, 24(R1)\n")
			fmt.Fprintf(w, "\tMOVD\t0(%s), R12\n", base)
			fmt.Fprintf(w, "\tMOVD\tR12, CTR\n")
			fmt.Fprintf(w, "\tBL\t(CTR)\n")
			fmt.Fprintf(w, "\tMOVD\t24(R1), R2\n")
			fmt.Fprintf(w, "\tXOR\tR0, R0\n")
		},
		CallSymbol: func(w io.Writer, sym string, static bool) {
			if isAIX {
				if static {
					emitAIXDescCall(w, fmt.Sprintf("$%s(SB)", sym))
				} else {
					emitAIXDescCall(w, fmt.Sprintf("__cgo_imp_%s(SB)", sym))
				}
				return
			}
			fmt.Fprintf(w, "\tBL\t%s(SB)\n", sym)
			fmt.Fprintf(w, "\tWORD\t$0x60000000\n")
			fmt.Fprintf(w, "\tXOR\tR0, R0\n")
		},
		CallbackPrologue: func(w io.Writer, frameSize int64) int64 {
			inner := localFrameOff + frameSize
			total := inner + 16
			fmt.Fprintf(w, "\tMOVD\tLR, R12\n")
			fmt.Fprintf(w, "\tMOVD\tR12, 16(R1)\n")
			if total <= 32760 {
				emitPPC64AllocBackchain(w, "R12", total)
				fmt.Fprintf(w, "\tMOVD\tR31, %d(R1)\n", inner)
				fmt.Fprintf(w, "\tMOVD\tR2, %d(R1)\n", inner+8)
				fmt.Fprintf(w, "\tXOR\tR0, R0\n")
				return total
			}
			// For total > 32760, large store offsets clobber REGTMP (R31)
			// before it is saved (and on GOPPC64 < 10, MOVD R31, largeOff(R1)
			// self-clobbers R31). Allocate an outer 16-byte frame to save R31
			// and R2 at small displacements before allocating the inner frame.
			fmt.Fprintf(w, "\tMOVD\tR1, R12\n")
			fmt.Fprintf(w, "\tADD\t$-16, R1\n")
			fmt.Fprintf(w, "\tMOVD\tR31, 0(R1)\n")
			fmt.Fprintf(w, "\tMOVD\tR2, 8(R1)\n")
			fmt.Fprintf(w, "\tXOR\tR0, R0\n")
			emitPPC64AddSP(w, -inner)
			fmt.Fprintf(w, "\tMOVD\tR12, 0(R1)\n")
			return total
		},
		CallbackEpilogue: func(w io.Writer, frameSize int64, _ bool) {
			inner := localFrameOff + frameSize
			total := inner + 16
			if total <= 32760 {
				fmt.Fprintf(w, "\tMOVD\t%d(R1), R2\n", inner+8)
				fmt.Fprintf(w, "\tMOVD\t%d(R1), R31\n", inner)
				fmt.Fprintf(w, "\tADD\t$%d, R1\n", total)
				fmt.Fprintf(w, "\tMOVD\t16(R1), R12\n")
				fmt.Fprintf(w, "\tMOVD\tR12, LR\n")
				fmt.Fprintf(w, "\tRET\n")
				return
			}
			// Re-zero R0 and undo the inner SP adjustment before restoring
			// R2, R31, and the outer 16-byte frame.
			fmt.Fprintf(w, "\tXOR\tR0, R0\n")
			emitPPC64AddSP(w, inner)
			fmt.Fprintf(w, "\tMOVD\t8(R1), R2\n")
			fmt.Fprintf(w, "\tMOVD\t0(R1), R31\n")
			fmt.Fprintf(w, "\tADD\t$16, R1\n")
			fmt.Fprintf(w, "\tMOVD\t16(R1), R12\n")
			fmt.Fprintf(w, "\tMOVD\tR12, LR\n")
			fmt.Fprintf(w, "\tRET\n")
		},
	}
}

var ppc64leABI = newPPC64ABI("ppc64le", false, 32, false)
var ppc64ABI = newPPC64ABI("ppc64", true, 32, false)
var aixPPC64ABI = newPPC64ABI("ppc64", true, 48, true)

// IBM Z (s390x) 64-bit ELF C calling convention:
// "ELF ABI s390x Supplement", Section 1.2: https://github.com/IBM/s390x-abi
//
// Summary:
//   - Integer and pointer arguments use R2..R6; float and double arguments
//     use F0, F2, F4, F6. Later arguments go in 8-byte stack slots above the
//     160-byte register save area, right-justified.
//   - A struct of 1, 2, 4, or 8 bytes is passed in a GPR, except that a
//     struct whose only member is a float or double (or such a struct) is
//     passed in an FPR. Arrays (including a one-member array of float), and
//     unions are not such structs.
//   - Every other composite, including _Complex float and _Complex double,
//     is passed by reference to a caller-made copy.
//   - Every composite is returned through a hidden pointer in R2.

// s390xRegStructSize reports whether a struct or union of size bytes is
// passed by value in a register under the s390x ELF ABI.
func s390xRegStructSize(size int64) bool {
	return size == 1 || size == 2 || size == 4 || size == 8
}

func s390xAssignParam(cfg *ABIConfig, _ BindingFunc, _ int, info typeInfo, st *abiState) ParamLoc {
	if info.size == 0 {
		return ParamLoc{}
	}
	if info.isFloat {
		return cfg.assignFloatParam(info, st)
	}
	if info.composite {
		if info.isComplex || !s390xRegStructSize(info.size) {
			return cfg.assignPtrParam(info.size, st)
		}
		if info.singleFloatStruct && (info.size == 4 || info.size == 8) {
			return cfg.assignFloatParam(info, st)
		}
	}
	return cfg.assignIntParam(info, st)
}

var s390xABI = &ABIConfig{
	Arch:                         "s390x",
	PtrSize:                      8,
	CLongSize:                    8,
	StackAlign:                   8,
	ShadowSpace:                  160,
	IntRegs:                      []string{"R2", "R3", "R4", "R5", "R6"},
	FloatRegs:                    []string{"F0", "F2", "F4", "F6"},
	IntRetRegs:                   []string{"R2"},
	FloatRetRegs:                 []string{"F0"},
	MaxRegStructSize:             0,
	IndirectRetReg:               "R2",
	IndirectRetInArg:             true,
	AssignParam:                  s390xAssignParam,
	AssignRet:                    defaultAssignRet,
	SPReg:                        "R15",
	CalleeSavedReg:               "R8",
	TmpReg:                       "R1",
	TmpReg2:                      "R9",
	LinkReg:                      "R14",
	ZeroVal:                      "$0",
	LocalFrameOff:                160,
	BigEndian:                    true,
	RightJustifyBECompositeStack: true,
	asm: asmOps{
		mov:   "MOVD",
		loadB: "MOVB", loadBU: "MOVBZ", loadH: "MOVH", loadHU: "MOVHZ", loadW: "MOVW", loadWU: "MOVWZ",
		storeB: "MOVB", storeH: "MOVH", storeW: "MOVW",
		fmovS: "FMOVS", fmovD: "FMOVD",
		shl: "SLD", shr: "SRD", or: "OR", add: "ADD", sub: "SUB",
		twoOpALU: true,
		negSubSP: true,
	},
	CallbackPrologue: func(w io.Writer, frameSize int64) int64 {
		total := 160 + frameSize
		fmt.Fprintf(w, "\tSTMG\tR6, R14, 48(R15)\n")
		fmt.Fprintf(w, "\tMOVD\tR14, -%d(R15)\n", total)
		fmt.Fprintf(w, "\tADD\t$-%d, R15\n", total)
		fmt.Fprintf(w, "\tMOVD\tR14, 0(R15)\n")
		return total
	},
	CallbackEpilogue: func(w io.Writer, frameSize int64, _ bool) {
		total := 160 + frameSize
		fmt.Fprintf(w, "\tMOVD\t0(R15), R14\n")
		fmt.Fprintf(w, "\tADD\t$%d, R15\n", total)
		fmt.Fprintf(w, "\tLMG\t48(R15), R6, R14\n")
		fmt.Fprintf(w, "\tRET\n")
	},
}

// RISC-V RV64GC and LoongArch64 LP64D C calling conventions:
//   - RISC-V: https://github.com/riscv-non-isa/riscv-elf-psabi-doc/blob/master/riscv-cc.adoc
//   - LoongArch64: https://loongson.github.io/LoongArch-Documentation/LoongArch-ELF-ABI-EN.html

// classifyLP64DStruct classifies a <=16-byte composite per the LP64D hardware
// floating-point calling convention (1-2 FP fields, or 1 FP + 1 integer field).
//
// Only structs (including structs with array members) and complex numbers
// are eligible; unions, represented as arrays, are not.
func classifyLP64DStruct(info typeInfo) (fields []leafField, nFP, nInt int, ok bool) {
	if (!info.isStruct && !info.isComplex) || !info.fieldsOK || info.size <= 0 || info.size > 16 {
		return nil, 0, 0, false
	}
	if len(info.fields) < 1 || len(info.fields) > 2 {
		return nil, 0, 0, false
	}
	for _, f := range info.fields {
		if f.isFloat {
			nFP++
		} else {
			nInt++
		}
	}
	if nFP == 0 {
		return nil, 0, 0, false
	}
	return info.fields, nFP, nInt, true
}

func lp64dAssignRet(cfg *ABIConfig, fn BindingFunc, info typeInfo, st *abiState) RetLoc {
	if fields, _, _, ok := classifyLP64DStruct(info); ok {
		regs := make([]string, len(fields))
		fpIdx, intIdx := 0, 0
		for k, f := range fields {
			if f.isFloat {
				regs[k] = cfg.FloatRetRegs[fpIdx]
				fpIdx++
			} else {
				regs[k] = cfg.IntRetRegs[intIdx]
				intIdx++
			}
		}
		return RetLoc{Size: info.size, IsFloat: true, Regs: regs}
	}
	return defaultAssignRet(cfg, fn, info, st)
}

func lp64dCompositeRegSlots(cfg *ABIConfig, info typeInfo, isFloat bool, regs []string) ([]regSlot, bool) {
	if !isFloat || !info.fieldsOK || len(info.fields) != len(regs) {
		return nil, false
	}
	slots := make([]regSlot, len(regs))
	for k, f := range info.fields {
		slots[k] = regSlot{
			reg:     regs[k],
			offset:  f.offset,
			size:    f.size,
			align:   f.align,
			isFloat: f.isFloat,
			signed:  !f.isFloat && (f.signed || (cfg.SignExtUint32 && f.size == 4)),
		}
	}
	return slots, true
}

func lp64dAssignParam(cfg *ABIConfig, _ BindingFunc, _ int, info typeInfo, st *abiState) ParamLoc {
	if info.size == 0 {
		return ParamLoc{}
	}
	if info.isFloat {
		if st.floatIdx < len(cfg.FloatRegs) {
			return cfg.assignFloatParam(info, st)
		}
		if st.intIdx < len(cfg.IntRegs) {
			reg := cfg.IntRegs[st.intIdx]
			st.intIdx++
			return ParamLoc{Size: info.size, Regs: []string{reg}}
		}
		return cfg.assignStackParam(info.size, true, st)
	}
	if info.composite {
		if fields, nFP, nInt, ok := classifyLP64DStruct(info); ok &&
			st.floatIdx+nFP <= len(cfg.FloatRegs) && st.intIdx+nInt <= len(cfg.IntRegs) {
			regs := make([]string, len(fields))
			for k, f := range fields {
				if f.isFloat {
					regs[k] = cfg.FloatRegs[st.floatIdx]
					st.floatIdx++
				} else {
					regs[k] = cfg.IntRegs[st.intIdx]
					st.intIdx++
				}
			}
			return ParamLoc{Size: info.size, IsFloat: true, Regs: regs}
		}
		if info.size > cfg.MaxRegStructSize {
			return cfg.assignPtrParam(info.size, st)
		}
	}
	// Unlike variadic arguments, named arguments with 2*XLEN alignment do
	// not skip to an even register.
	numRegs := int((info.size + cfg.PtrSize - 1) / cfg.PtrSize)
	if st.intIdx+numRegs <= len(cfg.IntRegs) {
		regs := slices.Clone(cfg.IntRegs[st.intIdx : st.intIdx+numRegs])
		st.intIdx += numRegs
		return ParamLoc{Size: info.size, Regs: regs}
	}
	if st.intIdx < len(cfg.IntRegs) && info.composite {
		// Split 8B-aligned 9..16B aggregate straddling the last GPR (a7) and stack.
		regs := slices.Clone(cfg.IntRegs[st.intIdx:])
		regBytes := int64(len(regs)) * cfg.PtrSize
		stackBytes := info.size - regBytes
		st.intIdx = len(cfg.IntRegs)
		off := st.stackOff
		st.stackOff += alignUp(stackBytes, cfg.PtrSize)
		return ParamLoc{Size: info.size, Regs: regs, StackOff: off, StackBytes: stackBytes}
	}
	st.intIdx = len(cfg.IntRegs)
	if info.align >= 16 {
		st.stackOff = alignUp(st.stackOff, 16)
	}
	return cfg.assignStackParam(info.size, false, st)
}

var riscv64ABI = &ABIConfig{
	Arch:              "riscv64",
	PtrSize:           8,
	CLongSize:         8,
	StackAlign:        16,
	IntRegs:           []string{"X10", "X11", "X12", "X13", "X14", "X15", "X16", "X17"},
	FloatRegs:         []string{"F10", "F11", "F12", "F13", "F14", "F15", "F16", "F17"},
	IntRetRegs:        []string{"X10", "X11"},
	FloatRetRegs:      []string{"F10", "F11"},
	MaxRegStructSize:  16,
	IndirectRetReg:    "X10",
	IndirectRetInArg:  true,
	AssignParam:       lp64dAssignParam,
	AssignRet:         lp64dAssignRet,
	CompositeRegSlots: lp64dCompositeRegSlots,
	SPReg:             "X2",
	CalleeSavedReg:    "X9",
	TmpReg:            "X5",
	TmpReg2:           "X6",
	IndCallReg:        "X6", // Use T1 (X6) instead of T0 (X5) so JALR X1, (reg) does not trigger a RAS pop+push.
	LinkReg:           "X1",
	LRReserve:         16,
	ZeroVal:           "X0",
	LocalFrameOff:     16,
	SignExtUint32:     true,
	asm: asmOps{
		mov:   "MOV",
		loadB: "MOVB", loadBU: "MOVBU", loadH: "MOVH", loadHU: "MOVHU", loadW: "MOVW", loadWU: "MOVWU",
		storeB: "MOVB", storeH: "MOVH", storeW: "MOVW",
		fmovS: "MOVF", fmovD: "MOVD",
		shl: "SLL", shr: "SRL", or: "OR", add: "ADD", sub: "SUB",
		negSubSP: true,
	},
}

var loong64ABI = &ABIConfig{
	Arch:              "loong64",
	PtrSize:           8,
	CLongSize:         8,
	StackAlign:        16,
	IntRegs:           []string{"R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"},
	FloatRegs:         []string{"F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7"},
	IntRetRegs:        []string{"R4", "R5"},
	FloatRetRegs:      []string{"F0", "F1"},
	MaxRegStructSize:  16,
	IndirectRetReg:    "R4",
	IndirectRetInArg:  true,
	AssignParam:       lp64dAssignParam,
	AssignRet:         lp64dAssignRet,
	CompositeRegSlots: lp64dCompositeRegSlots,
	SPReg:             "R3",
	CalleeSavedReg:    "R23",
	TmpReg:            "R12",
	TmpReg2:           "R13",
	LinkReg:           "R1",
	AsmTmpReg:         "R30",
	LRReserve:         16,
	ZeroVal:           "R0",
	LocalFrameOff:     16,
	SignExtUint32:     true,
	asm: asmOps{
		mov:   "MOVV",
		loadB: "MOVB", loadBU: "MOVBU", loadH: "MOVH", loadHU: "MOVHU", loadW: "MOVW", loadWU: "MOVWU",
		storeB: "MOVB", storeH: "MOVH", storeW: "MOVW",
		fmovS: "MOVF", fmovD: "MOVD",
		shl: "SLLV", shr: "SRLV", or: "OR", add: "ADDV", sub: "SUBV",
		negSubSP: true,
	},
}

// 64-bit MIPS N64 C calling convention:
// "MIPSpro N32 ABI Handbook", Chapter 2 (SGI Document 007-2816-004).

// classifyMIPS64FPRet reports whether a composite is returned in F0 and F2
// under N64 with hard float: _Complex float and _Complex double, and structs
// whose only members are one or two floats or doubles. Arrays, nested
// structs, and unions are returned in GPRs, even if they contain only
// floating-point values.
func classifyMIPS64FPRet(info typeInfo) ([]leafField, bool) {
	if !info.composite || !info.fieldsOK || info.size <= 0 || info.size > 16 || len(info.fields) < 1 || len(info.fields) > 2 {
		return nil, false
	}
	if info.isComplex {
		return info.fields, len(info.fields) == 2
	}
	if !info.isStruct || len(info.fields) != info.directFields || info.fields[0].offset != 0 {
		return nil, false
	}
	for _, f := range info.fields {
		if !f.direct || !f.isFloat {
			return nil, false
		}
	}
	return info.fields, true
}

// mips64StructSlotIsDouble reports whether the 8-byte slot at offset off of
// a struct argument holds a double member of the struct itself, which N64
// passes in an FPR. Doubles in arrays, nested structs, or unions are passed
// in GPRs.
func mips64StructSlotIsDouble(info typeInfo, off int64) bool {
	if !info.isStruct || !info.fieldsOK {
		return false
	}
	for _, f := range info.fields {
		if f.direct && f.offset == off && f.isFloat && f.size == 8 {
			return true
		}
	}
	return false
}

func mips64AssignRet(cfg *ABIConfig, fn BindingFunc, info typeInfo, st *abiState) RetLoc {
	if fields, ok := classifyMIPS64FPRet(info); ok {
		if len(cfg.FloatRetRegs) >= len(fields) {
			return RetLoc{Size: info.size, IsFloat: true, Regs: slices.Clone(cfg.FloatRetRegs[:len(fields)])}
		}
		if info.isComplex {
			// With soft float, each part of a complex number is
			// returned in its own GPR.
			return RetLoc{Size: info.size, IsFloat: true, Regs: []string{"R2", "R3"}}
		}
	}
	return defaultAssignRet(cfg, fn, info, st)
}

// mips64CompositeRegSlots maps each member of a composite returned in F0 and
// F2 (or R2 and R3 with soft float), or of a complex argument passed in two
// consecutive FPRs (or GPRs with soft float), to its register.
func mips64CompositeRegSlots(cfg *ABIConfig, info typeInfo, isFloat bool, regs []string) ([]regSlot, bool) {
	if !isFloat || len(regs) == 0 {
		return nil, false
	}
	fields, ok := classifyMIPS64FPRet(info)
	if !ok || len(fields) != len(regs) {
		return nil, false
	}
	slots := make([]regSlot, len(regs))
	for k, f := range fields {
		fl := cfg.isFloatReg(regs[k])
		slots[k] = regSlot{reg: regs[k], offset: f.offset, size: f.size, align: f.align, isFloat: fl, signed: !fl && cfg.SignExtUint32 && f.size == 4}
	}
	return slots, true
}

func mips64AssignParam(cfg *ABIConfig, _ BindingFunc, _ int, info typeInfo, st *abiState) ParamLoc {
	if info.size == 0 {
		return ParamLoc{}
	}
	slot := st.intIdx
	if info.isComplex && slot+2 <= len(cfg.IntRegs) {
		// N64 passes the parts of _Complex float and _Complex double
		// like two float or double arguments, in two consecutive FPRs (or
		// GPRs with soft float). In the last argument slot, a complex
		// number is passed like a struct instead, in R11 and the stack.
		st.intIdx = slot + 2
		if len(cfg.FloatRegs) > 0 {
			st.floatIdx += 2
			return ParamLoc{Size: info.size, IsFloat: true, Regs: []string{cfg.FloatRegs[slot], cfg.FloatRegs[slot+1]}}
		}
		return ParamLoc{Size: info.size, IsFloat: true, Regs: []string{cfg.IntRegs[slot], cfg.IntRegs[slot+1]}}
	}
	if info.isFloat {
		st.intIdx = slot + 1
		if slot < len(cfg.IntRegs) {
			if len(cfg.FloatRegs) > 0 {
				st.floatIdx++
				return ParamLoc{Size: info.size, IsFloat: true, Regs: []string{cfg.FloatRegs[slot]}}
			}
			return ParamLoc{Size: info.size, Regs: []string{cfg.IntRegs[slot]}}
		}
		return cfg.assignStackParam(info.size, true, st)
	}
	if info.align >= 16 && slot < len(cfg.IntRegs) && slot%2 != 0 {
		slot++
	}
	nSlots := (info.size + 7) / 8
	if slot < len(cfg.IntRegs) {
		nReg := min(int(nSlots), len(cfg.IntRegs)-slot)
		regs := make([]string, nReg)
		for k := range nReg {
			if len(cfg.FloatRegs) > 0 && mips64StructSlotIsDouble(info, int64(k)*8) {
				regs[k] = cfg.FloatRegs[slot+k]
			} else {
				regs[k] = cfg.IntRegs[slot+k]
			}
		}
		if slot+int(nSlots) <= len(cfg.IntRegs) {
			st.intIdx = slot + int(nSlots)
			return ParamLoc{Size: info.size, Regs: regs}
		}
		// Split aggregate straddling R[slot]..R11 / F[12+slot]..F19 and stack.
		regBytes := int64(nReg) * 8
		stackBytes := info.size - regBytes
		st.intIdx = len(cfg.IntRegs)
		off := st.stackOff
		st.stackOff += alignUp(stackBytes, 8)
		return ParamLoc{Size: info.size, Regs: regs, StackOff: off, StackBytes: stackBytes}
	}
	st.intIdx = len(cfg.IntRegs)
	if info.align >= 16 {
		st.stackOff = alignUp(st.stackOff, 16)
	}
	return cfg.assignStackParam(info.size, false, st)
}

func newMIPS64ABI(arch string, bigEndian bool) *ABIConfig {
	return &ABIConfig{
		Arch:                   arch,
		PtrSize:                8,
		CLongSize:              8,
		StackAlign:             16,
		IntRegs:                []string{"R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"},
		FloatRegs:              []string{"F12", "F13", "F14", "F15", "F16", "F17", "F18", "F19"},
		IntRetRegs:             []string{"R2", "R3"},
		FloatRetRegs:           []string{"F0", "F2"},
		MaxRegStructSize:       16,
		IndirectRetReg:         "R4",
		IndirectRetInArg:       true,
		AssignParam:            mips64AssignParam,
		AssignRet:              mips64AssignRet,
		CompositeRegSlots:      mips64CompositeRegSlots,
		SPReg:                  "R29",
		CalleeSavedReg:         "R16",
		TmpReg:                 "R12",
		TmpReg2:                "R13",
		IndCallReg:             "R25",
		LinkReg:                "R31",
		LRReserve:              16,
		ZeroVal:                "R0",
		LocalFrameOff:          16,
		BigEndian:              bigEndian,
		LeftJustifyBEComposite: bigEndian,
		// N64 left-justifies a float in its stack slot on big-endian
		// targets, and sign-extends uint32 values to 64 bits.
		LeftJustifyBEFloatStack: bigEndian,
		SignExtUint32:           true,
		StrictAlign:             true,
		asm: asmOps{
			mov:   "MOVV",
			loadB: "MOVB", loadBU: "MOVBU", loadH: "MOVH", loadHU: "MOVHU", loadW: "MOVW", loadWU: "MOVWU",
			storeB: "MOVB", storeH: "MOVH", storeW: "MOVW",
			fmovS: "MOVF", fmovD: "MOVD",
			shl: "SLLV", shr: "SRLV", or: "OR", add: "ADDV", sub: "SUBV",
			negSubSP: true,
		},
		CallSymbol: func(w io.Writer, sym string, _ bool) {
			// Load $t9 (R25) with target address per MIPS64 PIC (-mabicalls) ABI.
			fmt.Fprintf(w, "\tMOVV\t$%s(SB), R25\n", sym)
			fmt.Fprintf(w, "\tCALL\t(R25)\n")
		},
		CallbackPrologue: func(w io.Writer, frameSize int64) int64 {
			// Save C callee-saved R23 (REGTMP) and RSB (R28/$gp), then initialize
			// RSB = PC & 0xffffffff00000000 for Go (SB) addressing.
			total := 32 + frameSize
			if total <= 32760 {
				fmt.Fprintf(w, "\tMOVV\tR31, -%d(R29)\n", total)
				fmt.Fprintf(w, "\tADDV\t$-%d, R29\n", total)
				fmt.Fprintf(w, "\tMOVV\tR31, 0(R29)\n")
				fmt.Fprintf(w, "\tMOVV\tRSB, 8(R29)\n")
				fmt.Fprintf(w, "\tMOVV\tR23, %d(R29)\n", 16+frameSize)
			} else {
				fmt.Fprintf(w, "\tADDV\t$-16, R29\n")
				fmt.Fprintf(w, "\tMOVV\tR23, 0(R29)\n")
				fmt.Fprintf(w, "\tMOVV\tRSB, 8(R29)\n")
				fmt.Fprintf(w, "\tMOVV\tR31, -%d(R29)\n", 16+frameSize)
				fmt.Fprintf(w, "\tADDV\t$-%d, R29\n", 16+frameSize)
				fmt.Fprintf(w, "\tMOVV\tR31, 0(R29)\n")
			}
			fmt.Fprintf(w, "\tBGEZAL\tR0, 1(PC)\n")
			fmt.Fprintf(w, "\tSRLV\t$32, R31, RSB\n")
			fmt.Fprintf(w, "\tSLLV\t$32, RSB\n")
			return total
		},
		CallbackEpilogue: func(w io.Writer, frameSize int64, _ bool) {
			total := 32 + frameSize
			if total <= 32760 {
				fmt.Fprintf(w, "\tMOVV\t%d(R29), R23\n", 16+frameSize)
				fmt.Fprintf(w, "\tMOVV\t8(R29), RSB\n")
				fmt.Fprintf(w, "\tMOVV\t0(R29), R31\n")
				fmt.Fprintf(w, "\tADDV\t$%d, R29\n", total)
				fmt.Fprintf(w, "\tRET\n")
				return
			}
			fmt.Fprintf(w, "\tMOVV\t0(R29), R31\n")
			fmt.Fprintf(w, "\tADDV\t$%d, R29\n", 16+frameSize)
			fmt.Fprintf(w, "\tMOVV\t8(R29), RSB\n")
			fmt.Fprintf(w, "\tMOVV\t0(R29), R23\n")
			fmt.Fprintf(w, "\tADDV\t$16, R29\n")
			fmt.Fprintf(w, "\tRET\n")
		},
	}
}

var mips64ABI = newMIPS64ABI("mips64", true)
var mips64leABI = newMIPS64ABI("mips64le", false)

// getABI returns the ABIConfig for the given targetOS/targetArch.
func getABI(targetOS, targetArch string) *ABIConfig {
	var cfg ABIConfig
	switch targetArch {
	case "amd64":
		if targetOS == "windows" {
			cfg = *win64ABI
		} else {
			cfg = *amd64ABI
		}
	case "arm64":
		cfg = *arm64ABI
		switch targetOS {
		case "darwin", "ios":
			cfg.PackStackArgs = true
		case "windows":
			cfg.CLongSize = 4
			cfg.IndirectSymCall = true
		}
	case "386":
		cfg = *i386ABI
		switch targetOS {
		case "windows":
			cfg.SysVI386SretPop = false
			cfg.AssignRet = win32AssignRet
			cfg.IndirectSymCall = true
		case "freebsd", "openbsd":
			cfg.AssignRet = bsd386AssignRet
		}
	case "arm":
		cfg = *armABI
		if targetOS == "android" || targetOS == "openbsd" || goarmSoftFloat() {
			cfg.IsFloatReg = nil
			cfg.LoadFloat = nil
			cfg.StoreFloat = nil
			cfg.FloatRegs = nil
			cfg.FloatRetRegs = nil
			cfg.AssignRet = defaultAssignRet
			cfg.CompositeRegSlots = nil
		}
	case "riscv64":
		cfg = *riscv64ABI
	case "loong64":
		cfg = *loong64ABI
	case "ppc64":
		if targetOS == "aix" {
			cfg = *aixPPC64ABI
		} else {
			cfg = *ppc64ABI
		}
	case "ppc64le":
		cfg = *ppc64leABI
	case "s390x":
		cfg = *s390xABI
	case "mips64", "mips64le":
		if targetArch == "mips64" {
			cfg = *mips64ABI
		} else {
			cfg = *mips64leABI
		}
		if gomips64 := os.Getenv("GOMIPS64"); gomips64 == "softfloat" || (gomips64 == "" && buildcfg.GOMIPS64 == "softfloat") {
			cfg.FloatRegs = nil
			cfg.FloatRetRegs = nil
		}
	case "mips", "mipsle":
		if targetArch == "mips" {
			cfg = *mipsABI
		} else {
			cfg = *mipsleABI
		}
		if gomips := os.Getenv("GOMIPS"); gomips == "softfloat" || (gomips == "" && buildcfg.GOMIPS == "softfloat") {
			cfg.FloatRegs = nil
			cfg.FloatRetRegs = nil
		}
	default:
		panic("unsupported arch: " + targetArch)
	}
	cfg.OS = targetOS
	return &cfg
}
