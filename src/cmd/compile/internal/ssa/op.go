// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/abi"
	"cmd/compile/internal/base"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"fmt"
	rtabi "internal/abi"
	"strings"
)

// An Op encodes the specific operation that a Value performs.
// Opcodes' semantics can be modified by the type and aux fields of the Value.
// For instance, OpAdd can be 32 or 64 bit, signed or unsigned, float or complex, depending on Value.Type.
// Semantics of each op are described in the opcode files in _gen/*Ops.go.
// There is one file for generic (architecture-independent) ops and one file
// for each architecture.
type Op int32

type OpInfo struct {
	Name              string
	Reg               RegInfo
	AuxType           AuxType_
	ArgLen            int32 // the number of arguments, -1 if variable length
	asm               obj.As
	Generic           bool      // this is a generic (arch-independent) opcode
	Rematerializeable bool      // this op is rematerializeable
	Commutative       bool      // this operation is commutative (e.g. addition)
	ResultInArg0      bool      // (first, if a tuple) output of v and v.Args[0] must be allocated to the same register
	ResultNotInArgs   bool      // outputs must not be allocated to the same registers as inputs
	ClobberFlags      bool      // this op clobbers flags register
	NeedIntTemp       bool      // need a temporary free integer register
	Call              bool      // is a function call
	tailCall          bool      // is a tail call
	NilCheck          bool      // this op is a nil check on arg0
	FaultOnNilArg0    bool      // this op will fault if arg0 is nil (and aux encodes a small offset)
	FaultOnNilArg1    bool      // this op will fault if arg1 is nil (and aux encodes a small offset)
	usesScratch       bool      // this op requires scratch memory space
	HasSideEffects    bool      // for "reasons", not to be eliminated.  E.g., atomic store, #19182.
	ZeroWidth         bool      // op never translates into any machine code. example: copy, which may sometimes translate to machine code, is not zero-width.
	unsafePoint       bool      // this op is an unsafe point, i.e. not safe for async preemption
	FixedReg          bool      // this op will be assigned a fixed register
	EarlyOk           bool      // executing this op in an earlier block is ok
	AddrSinkArg0      bool      // the address in arg0 does not propagate to the result
	AddrSinkArg1      bool      // the address in arg1 does not propagate to the result
	symEffect         SymEffect // effect this op has on symbol in aux
	scale             uint8     // amd64/386 indexed load scale
	ZeroUpperBits     uint8     // the op writes a 64-bit GPR whose upper N bits are always zero (0, 32, 48 or 56); for a tuple op, this holds for every integer result
}

type InputInfo struct {
	Idx  int     // index in Args array
	Regs RegMask // allowed input registers
}

type OutputInfo struct {
	Idx  int     // index in output tuple
	Regs RegMask // allowed output registers
}

type RegInfo struct {
	// Inputs encodes the register restrictions for an instruction's Inputs.
	// Each entry specifies an allowed register set for a particular input.
	// They are listed in the order in which regalloc should pick a register
	// from the register set (most constrained first).
	// Inputs which do not need registers are not listed.
	Inputs []InputInfo
	// Clobbers encodes the set of registers that are overwritten by
	// the instruction (other than the output registers).
	Clobbers RegMask
	// Instruction clobbers the register containing input 0.
	ClobbersArg0 bool
	// Instruction clobbers the register containing input 1.
	ClobbersArg1 bool
	// Outputs is the same as inputs, but for the Outputs of the instruction.
	Outputs []OutputInfo
}

func (r *RegInfo) String() string {
	s := ""
	s += "INS:\n"
	for _, i := range r.Inputs {
		mask := fmt.Sprintf("%64b", i.Regs)
		mask = strings.ReplaceAll(mask, "0", ".")
		s += fmt.Sprintf("%2d |%s|\n", i.Idx, mask)
	}
	s += "OUTS:\n"
	for _, i := range r.Outputs {
		mask := fmt.Sprintf("%64b", i.Regs)
		mask = strings.ReplaceAll(mask, "0", ".")
		s += fmt.Sprintf("%2d |%s|\n", i.Idx, mask)
	}
	s += "CLOBBERS:\n"
	mask := fmt.Sprintf("%64b", r.Clobbers)
	mask = strings.ReplaceAll(mask, "0", ".")
	s += fmt.Sprintf("   |%s|\n", mask)
	return s
}

type AuxType_ int8

type AuxNameOffset struct {
	Name   *ir.Name
	Offset int64
}

func (a *AuxNameOffset) CanBeAnSSAAux() {}
func (a *AuxNameOffset) String() string {
	return fmt.Sprintf("%s+%d", a.Name.Sym().Name, a.Offset)
}

func (a *AuxNameOffset) FrameOffset() int64 {
	return a.Name.FrameOffset() + a.Offset
}

type AuxCall struct {
	Fn       *obj.LSym
	RegCache *RegInfo // regInfo for this call
	AbiInfo  *abi.ABIParamResultInfo
}

// Reg returns the regInfo for a given call, combining the derived in/out register masks
// with the machine-specific register information in the input i.  (The machine-specific
// regInfo is much handier at the call site than it is when the AuxCall is being constructed,
// therefore do this lazily).
//
// TODO: there is a Clever Hack that allows pre-generation of a small-ish number of the slices
// of inputInfo and outputInfo used here, provided that we are willing to reorder the inputs
// and outputs from calls, so that all integer registers come first, then all floating registers.
// At this point (active development of register ABI) that is very premature,
// but if this turns out to be a cost, we could do it.
func (a *AuxCall) Reg(i *RegInfo, c *Config) *RegInfo {
	if !a.RegCache.Clobbers.Empty() {
		// Already updated
		return a.RegCache
	}
	if a.AbiInfo.InRegistersUsed()+a.AbiInfo.OutRegistersUsed() == 0 {
		// Shortcut for zero case, also handles old ABI.
		a.RegCache = i
		return a.RegCache
	}

	k := len(i.Inputs)
	for _, p := range a.AbiInfo.InParams() {
		for _, r := range p.Registers {
			m := ArchRegForAbiReg(r, c)
			a.RegCache.Inputs = append(a.RegCache.Inputs, InputInfo{Idx: k, Regs: RegMaskAt(Register(m))})
			k++
		}
	}
	a.RegCache.Inputs = append(a.RegCache.Inputs, i.Inputs...) // These are less constrained, thus should come last
	k = len(i.Outputs)
	for _, p := range a.AbiInfo.OutParams() {
		for _, r := range p.Registers {
			m := ArchRegForAbiReg(r, c)
			a.RegCache.Outputs = append(a.RegCache.Outputs, OutputInfo{Idx: k, Regs: RegMaskAt(Register(m))})
			k++
		}
	}
	a.RegCache.Outputs = append(a.RegCache.Outputs, i.Outputs...)
	a.RegCache.Clobbers = i.Clobbers
	return a.RegCache
}
func (a *AuxCall) ABI() *abi.ABIConfig {
	return a.AbiInfo.Config()
}
func (a *AuxCall) ABIInfo() *abi.ABIParamResultInfo {
	return a.AbiInfo
}
func (a *AuxCall) ResultReg(c *Config) *RegInfo {
	if a.AbiInfo.OutRegistersUsed() == 0 {
		return a.RegCache
	}
	if len(a.RegCache.Inputs) > 0 {
		return a.RegCache
	}
	k := 0
	for _, p := range a.AbiInfo.OutParams() {
		for _, r := range p.Registers {
			m := ArchRegForAbiReg(r, c)
			a.RegCache.Inputs = append(a.RegCache.Inputs, InputInfo{Idx: k, Regs: RegMaskAt(Register(m))})
			k++
		}
	}
	return a.RegCache
}

// For ABI register index r, returns the (dense) register number used in
// SSA backend.
func ArchRegForAbiReg(r abi.RegIndex, c *Config) uint8 {
	var m int8
	if int(r) < len(c.IntParamRegs) {
		m = c.IntParamRegs[r]
	} else {
		m = c.FloatParamRegs[int(r)-len(c.IntParamRegs)]
	}
	return uint8(m)
}

// For ABI register index r, returns the register number used in the obj
// package (assembler).
func ObjRegForAbiReg(r abi.RegIndex, c *Config) int16 {
	m := ArchRegForAbiReg(r, c)
	return c.Registers[m].ObjNum
}

// ArgWidth returns the amount of stack needed for all the inputs
// and outputs of a function or method, including ABI-defined parameter
// slots and ABI-defined spill slots for register-resident parameters.
//
// The name is taken from the types package's ArgWidth(<function type>),
// which predated changes to the ABI; this version handles those changes.
func (a *AuxCall) ArgWidth() int64 {
	return a.AbiInfo.ArgWidth()
}

// ParamAssignmentForResult returns the ABI Parameter assignment for result which (indexed 0, 1, etc).
func (a *AuxCall) ParamAssignmentForResult(which int64) *abi.ABIParamAssignment {
	return a.AbiInfo.OutParam(int(which))
}

// OffsetOfResult returns the SP offset of result which (indexed 0, 1, etc).
func (a *AuxCall) OffsetOfResult(which int64) int64 {
	n := int64(a.AbiInfo.OutParam(int(which)).Offset())
	return n
}

// OffsetOfArg returns the SP offset of argument which (indexed 0, 1, etc).
// If the call is to a method, the receiver is the first argument (i.e., index 0)
func (a *AuxCall) OffsetOfArg(which int64) int64 {
	n := int64(a.AbiInfo.InParam(int(which)).Offset())
	return n
}

// RegsOfResult returns the register(s) used for result which (indexed 0, 1, etc).
func (a *AuxCall) RegsOfResult(which int64) []abi.RegIndex {
	return a.AbiInfo.OutParam(int(which)).Registers
}

// RegsOfArg returns the register(s) used for argument which (indexed 0, 1, etc).
// If the call is to a method, the receiver is the first argument (i.e., index 0)
func (a *AuxCall) RegsOfArg(which int64) []abi.RegIndex {
	return a.AbiInfo.InParam(int(which)).Registers
}

// NameOfResult returns the ir.Name of result which (indexed 0, 1, etc).
func (a *AuxCall) NameOfResult(which int64) *ir.Name {
	return a.AbiInfo.OutParam(int(which)).Name
}

// TypeOfResult returns the type of result which (indexed 0, 1, etc).
func (a *AuxCall) TypeOfResult(which int64) *types.Type {
	return a.AbiInfo.OutParam(int(which)).Type
}

// TypeOfArg returns the type of argument which (indexed 0, 1, etc).
// If the call is to a method, the receiver is the first argument (i.e., index 0)
func (a *AuxCall) TypeOfArg(which int64) *types.Type {
	return a.AbiInfo.InParam(int(which)).Type
}

// SizeOfResult returns the size of result which (indexed 0, 1, etc).
func (a *AuxCall) SizeOfResult(which int64) int64 {
	return a.TypeOfResult(which).Size()
}

// SizeOfArg returns the size of argument which (indexed 0, 1, etc).
// If the call is to a method, the receiver is the first argument (i.e., index 0)
func (a *AuxCall) SizeOfArg(which int64) int64 {
	return a.TypeOfArg(which).Size()
}

// NResults returns the number of results.
func (a *AuxCall) NResults() int64 {
	return int64(len(a.AbiInfo.OutParams()))
}

// LateExpansionResultType returns the result type (including trailing mem)
// for a call that will be expanded later in the SSA phase.
func (a *AuxCall) LateExpansionResultType() *types.Type {
	var tys []*types.Type
	for i := int64(0); i < a.NResults(); i++ {
		tys = append(tys, a.TypeOfResult(i))
	}
	tys = append(tys, types.TypeMem)
	return types.NewResults(tys)
}

// NArgs returns the number of arguments (including receiver, if there is one).
func (a *AuxCall) NArgs() int64 {
	return int64(len(a.AbiInfo.InParams()))
}

// String returns "AuxCall{<fn>}"
func (a *AuxCall) String() string {
	var fn string
	if a.Fn == nil {
		fn = "AuxCall{nil" // could be interface/closure etc.
	} else {
		fn = fmt.Sprintf("AuxCall{%v", a.Fn)
	}
	// TODO how much of the ABI should be printed?

	return fn + "}"
}

// StaticAuxCall returns an AuxCall for a static call.
func StaticAuxCall(sym *obj.LSym, paramResultInfo *abi.ABIParamResultInfo) *AuxCall {
	if paramResultInfo == nil {
		panic(fmt.Errorf("Nil paramResultInfo, sym=%v", sym))
	}
	var reg *RegInfo
	if paramResultInfo.InRegistersUsed()+paramResultInfo.OutRegistersUsed() > 0 {
		reg = &RegInfo{}
	}
	return &AuxCall{Fn: sym, AbiInfo: paramResultInfo, RegCache: reg}
}

// InterfaceAuxCall returns an AuxCall for an interface call.
func InterfaceAuxCall(paramResultInfo *abi.ABIParamResultInfo) *AuxCall {
	var reg *RegInfo
	if paramResultInfo.InRegistersUsed()+paramResultInfo.OutRegistersUsed() > 0 {
		reg = &RegInfo{}
	}
	return &AuxCall{Fn: nil, AbiInfo: paramResultInfo, RegCache: reg}
}

// ClosureAuxCall returns an AuxCall for a closure call.
func ClosureAuxCall(paramResultInfo *abi.ABIParamResultInfo) *AuxCall {
	var reg *RegInfo
	if paramResultInfo.InRegistersUsed()+paramResultInfo.OutRegistersUsed() > 0 {
		reg = &RegInfo{}
	}
	return &AuxCall{Fn: nil, AbiInfo: paramResultInfo, RegCache: reg}
}

func (*AuxCall) CanBeAnSSAAux() {}

// OwnAuxCall returns a function's own AuxCall.
func OwnAuxCall(fn *obj.LSym, paramResultInfo *abi.ABIParamResultInfo) *AuxCall {
	// TODO if this remains identical to ClosureAuxCall above after new ABI is done, should deduplicate.
	var reg *RegInfo
	if paramResultInfo.InRegistersUsed()+paramResultInfo.OutRegistersUsed() > 0 {
		reg = &RegInfo{}
	}
	return &AuxCall{Fn: fn, AbiInfo: paramResultInfo, RegCache: reg}
}

const (
	AuxTypeNone           AuxType_ = iota
	AuxTypeBool                    // auxInt is 0/1 for false/true
	AuxTypeInt8                    // auxInt is an 8-bit integer
	AuxTypeInt16                   // auxInt is a 16-bit integer
	AuxTypeInt32                   // auxInt is a 32-bit integer
	AuxTypeInt64                   // auxInt is a 64-bit integer
	AuxTypeInt128                  // auxInt represents a 128-bit integer.  Always 0.
	AuxTypeUInt8                   // auxInt is an 8-bit unsigned integer
	AuxTypeFloat32                 // auxInt is a float32 (encoded with math.Float64bits)
	AuxTypeFloat64                 // auxInt is a float64 (encoded with math.Float64bits)
	AuxTypeFlagConstant            // auxInt is a flagConstant
	AuxTypeCCop                    // auxInt is a ssa.Op that represents a flags-to-bool conversion (e.g. LessThan)
	AuxTypeNameOffsetInt8          // aux is a &struct{Name ir.Name, Offset int64}; auxInt is index in parameter registers array
	AuxTypeString                  // aux is a string
	AuxTypeSym                     // aux is a symbol (a *ir.Name for locals, an *obj.LSym for globals, or nil for none)
	AuxTypeSymOff                  // aux is a symbol, auxInt is an offset
	AuxTypeSymValAndOff            // aux is a symbol, auxInt is a ValAndOff
	AuxTypeTyp                     // aux is a type
	AuxTypeTypSize                 // aux is a type, auxInt is a size, must have Aux.(Type).Size() == AuxInt
	AuxTypeCall                    // aux is a *ssa.AuxCall
	AuxTypeCallOff                 // aux is a *ssa.AuxCall, AuxInt is int64 param (in+out) size

	AuxTypePanicBoundsC  // constant for a bounds failure
	AuxTypePanicBoundsCC // two constants for a bounds failure

	// architecture specific aux types
	AuxTypeARM64BitField          // aux is an arm64 bitfield lsb and width packed into auxInt
	AuxTypeARM64ConditionalParams // aux is a structure, which contains condition, NZCV flags and constant with indicator of using it
	AuxTypeS390XRotateParams      // aux is a s390x rotate parameters object encoding start bit, end bit and rotate amount
	AuxTypeS390XCCMask            // aux is a s390x 4-bit condition code mask
	AuxTypeS390XCCMaskInt8        // aux is a s390x 4-bit condition code mask, auxInt is an int8 immediate
	AuxTypeS390XCCMaskUint8       // aux is a s390x 4-bit condition code mask, auxInt is a uint8 immediate
)

// A SymEffect describes the effect that an SSA Value has on the variable
// identified by the symbol in its Aux field.
type SymEffect int8

const (
	SymRead SymEffect = 1 << iota
	SymWrite
	SymAddr

	SymRdWr = SymRead | SymWrite

	SymNone SymEffect = 0
)

// A Sym represents a symbolic offset from a base register.
// Currently a Sym can be one of 3 things:
//   - a *ir.Name, for an offset from SP (the stack pointer)
//   - a *obj.LSym, for an offset from SB (the global pointer)
//   - nil, for no offset
type Sym interface {
	Aux
	CanBeAnSSASym()
}

// A ValAndOff is used by the several opcodes. It holds
// both a value and a pointer offset.
// A ValAndOff is intended to be encoded into an AuxInt field.
// The zero ValAndOff encodes a value of 0 and an offset of 0.
// The high 32 bits hold a value.
// The low 32 bits hold a pointer offset.
type ValAndOff int64

func (x ValAndOff) Val() int32   { return int32(int64(x) >> 32) }
func (x ValAndOff) Val64() int64 { return int64(x) >> 32 }
func (x ValAndOff) Val16() int16 { return int16(int64(x) >> 32) }
func (x ValAndOff) Val8() int8   { return int8(int64(x) >> 32) }

func (x ValAndOff) Off64() int64 { return int64(int32(x)) }
func (x ValAndOff) Off() int32   { return int32(x) }

func (x ValAndOff) String() string {
	return fmt.Sprintf("val=%d,off=%d", x.Val(), x.Off())
}

func MakeValAndOff(val, off int32) ValAndOff {
	return ValAndOff(int64(val)<<32 + int64(uint32(off)))
}

func (x ValAndOff) CanAdd32(off int32) bool {
	newoff := x.Off64() + int64(off)
	return newoff == int64(int32(newoff))
}
func (x ValAndOff) CanAdd64(off int64) bool {
	newoff := x.Off64() + off
	return newoff == int64(int32(newoff))
}

func (x ValAndOff) AddOffset32(off int32) ValAndOff {
	if !x.CanAdd32(off) {
		panic("invalid ValAndOff.addOffset32")
	}
	return MakeValAndOff(x.Val(), x.Off()+off)
}
func (x ValAndOff) AddOffset64(off int64) ValAndOff {
	if !x.CanAdd64(off) {
		panic("invalid ValAndOff.addOffset64")
	}
	return MakeValAndOff(x.Val(), x.Off()+int32(off))
}

// int128 is a type that stores a 128-bit constant.
// The only allowed constant right now is 0, so we can cheat quite a bit.
type int128 int64

type BoundsKind uint8

const (
	BoundsIndex       BoundsKind = iota // indexing operation, 0 <= idx < len failed
	BoundsIndexU                        // ... with unsigned idx
	BoundsSliceAlen                     // 2-arg slicing operation, 0 <= high <= len failed
	BoundsSliceAlenU                    // ... with unsigned high
	BoundsSliceAcap                     // 2-arg slicing operation, 0 <= high <= cap failed
	BoundsSliceAcapU                    // ... with unsigned high
	BoundsSliceB                        // 2-arg slicing operation, 0 <= low <= high failed
	BoundsSliceBU                       // ... with unsigned low
	BoundsSlice3Alen                    // 3-arg slicing operation, 0 <= max <= len failed
	BoundsSlice3AlenU                   // ... with unsigned max
	BoundsSlice3Acap                    // 3-arg slicing operation, 0 <= max <= cap failed
	BoundsSlice3AcapU                   // ... with unsigned max
	BoundsSlice3B                       // 3-arg slicing operation, 0 <= high <= max failed
	BoundsSlice3BU                      // ... with unsigned high
	BoundsSlice3C                       // 3-arg slicing operation, 0 <= low <= high failed
	BoundsSlice3CU                      // ... with unsigned low
	BoundsConvert                       // conversion to array pointer failed
	BoundsKindCount
)

// Returns the bounds error code needed by the runtime, and
// whether the x field is signed.
func (b BoundsKind) Code() (rtabi.BoundsErrorCode, bool) {
	switch b {
	case BoundsIndex:
		return rtabi.BoundsIndex, true
	case BoundsIndexU:
		return rtabi.BoundsIndex, false
	case BoundsSliceAlen:
		return rtabi.BoundsSliceAlen, true
	case BoundsSliceAlenU:
		return rtabi.BoundsSliceAlen, false
	case BoundsSliceAcap:
		return rtabi.BoundsSliceAcap, true
	case BoundsSliceAcapU:
		return rtabi.BoundsSliceAcap, false
	case BoundsSliceB:
		return rtabi.BoundsSliceB, true
	case BoundsSliceBU:
		return rtabi.BoundsSliceB, false
	case BoundsSlice3Alen:
		return rtabi.BoundsSlice3Alen, true
	case BoundsSlice3AlenU:
		return rtabi.BoundsSlice3Alen, false
	case BoundsSlice3Acap:
		return rtabi.BoundsSlice3Acap, true
	case BoundsSlice3AcapU:
		return rtabi.BoundsSlice3Acap, false
	case BoundsSlice3B:
		return rtabi.BoundsSlice3B, true
	case BoundsSlice3BU:
		return rtabi.BoundsSlice3B, false
	case BoundsSlice3C:
		return rtabi.BoundsSlice3C, true
	case BoundsSlice3CU:
		return rtabi.BoundsSlice3C, false
	case BoundsConvert:
		return rtabi.BoundsConvert, false
	default:
		base.Fatalf("bad bounds kind %d", b)
		return 0, false
	}
}

// Arm64BitField is the GO type of ARM64BitField auxInt.
// if x is an ARM64BitField, then width=x&0xff, lsb=(x>>8)&0xff, and
// width+lsb<64 for 64-bit variant, width+lsb<32 for 32-bit variant.
// the meaning of width and lsb are instruction-dependent.
type Arm64BitField int16

// Arm64ConditionalParams is the GO type of ARM64ConditionalParams auxInt.
type Arm64ConditionalParams struct {
	Cond     Op    // Condition code to evaluate
	NzcvVal  uint8 // Fallback NZCV flags value when condition is false
	ConstVal uint8 // Immediate value for constant comparisons
	Ind      bool  // Constant comparison indicator
}

func (op Op) IsLoweredGetClosurePtr() bool {
	switch op {
	case OpAMD64LoweredGetClosurePtr, OpPPC64LoweredGetClosurePtr, OpARMLoweredGetClosurePtr, OpARM64LoweredGetClosurePtr,
		Op386LoweredGetClosurePtr, OpMIPS64LoweredGetClosurePtr, OpLOONG64LoweredGetClosurePtr, OpS390XLoweredGetClosurePtr, OpMIPSLoweredGetClosurePtr,
		OpRISCV64LoweredGetClosurePtr, OpWasmLoweredGetClosurePtr:
		return true
	}
	return false
}
