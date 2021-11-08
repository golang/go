// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/abi"
	"cmd/compile/internal/ir"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"fmt"
	"strings"
)

// An Op encodes the specific operation that a Value performs.
// Opcodes' semantics can be modified by the type and aux fields of the Value.
// For instance, OpAdd can be 32 or 64 bit, signed or unsigned, float or complex, depending on Value.Type.
// Semantics of each op are described in the opcode files in gen/*Ops.go.
// There is one file for generic (architecture-independent) ops and one file
// for each architecture.
type Op int32

type opInfo struct {
	name              string
	reg               regInfo
	auxType           auxType
	argLen            int32 // the number of arguments, -1 if variable length
	asm               obj.As
	generic           bool      // this is a generic (arch-independent) opcode
	rematerializeable bool      // this op is rematerializeable
	commutative       bool      // this operation is commutative (e.g. addition)
	resultInArg0      bool      // (first, if a tuple) output of v and v.Args[0] must be allocated to the same register
	resultNotInArgs   bool      // outputs must not be allocated to the same registers as inputs
	clobberFlags      bool      // this op clobbers flags register
	call              bool      // is a function call
	tailCall          bool      // is a tail call
	nilCheck          bool      // this op is a nil check on arg0
	faultOnNilArg0    bool      // this op will fault if arg0 is nil (and aux encodes a small offset)
	faultOnNilArg1    bool      // this op will fault if arg1 is nil (and aux encodes a small offset)
	usesScratch       bool      // this op requires scratch memory space
	hasSideEffects    bool      // for "reasons", not to be eliminated.  E.g., atomic store, #19182.
	zeroWidth         bool      // op never translates into any machine code. example: copy, which may sometimes translate to machine code, is not zero-width.
	unsafePoint       bool      // this op is an unsafe point, i.e. not safe for async preemption
	symEffect         SymEffect // effect this op has on symbol in aux
	scale             uint8     // amd64/386 indexed load scale
}

type inputInfo struct {
	idx  int     // index in Args array
	regs regMask // allowed input registers
}

type outputInfo struct {
	idx  int     // index in output tuple
	regs regMask // allowed output registers
}

type regInfo struct {
	// inputs encodes the register restrictions for an instruction's inputs.
	// Each entry specifies an allowed register set for a particular input.
	// They are listed in the order in which regalloc should pick a register
	// from the register set (most constrained first).
	// Inputs which do not need registers are not listed.
	inputs []inputInfo
	// clobbers encodes the set of registers that are overwritten by
	// the instruction (other than the output registers).
	clobbers regMask
	// outputs is the same as inputs, but for the outputs of the instruction.
	outputs []outputInfo
}

func (r *regInfo) String() string {
	s := ""
	s += "INS:\n"
	for _, i := range r.inputs {
		mask := fmt.Sprintf("%64b", i.regs)
		mask = strings.Replace(mask, "0", ".", -1)
		s += fmt.Sprintf("%2d |%s|\n", i.idx, mask)
	}
	s += "OUTS:\n"
	for _, i := range r.outputs {
		mask := fmt.Sprintf("%64b", i.regs)
		mask = strings.Replace(mask, "0", ".", -1)
		s += fmt.Sprintf("%2d |%s|\n", i.idx, mask)
	}
	s += "CLOBBERS:\n"
	mask := fmt.Sprintf("%64b", r.clobbers)
	mask = strings.Replace(mask, "0", ".", -1)
	s += fmt.Sprintf("   |%s|\n", mask)
	return s
}

type auxType int8

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
	Fn      *obj.LSym
	reg     *regInfo // regInfo for this call
	abiInfo *abi.ABIParamResultInfo
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
func (a *AuxCall) Reg(i *regInfo, c *Config) *regInfo {
	if a.reg.clobbers != 0 {
		// Already updated
		return a.reg
	}
	if a.abiInfo.InRegistersUsed()+a.abiInfo.OutRegistersUsed() == 0 {
		// Shortcut for zero case, also handles old ABI.
		a.reg = i
		return a.reg
	}

	k := len(i.inputs)
	for _, p := range a.abiInfo.InParams() {
		for _, r := range p.Registers {
			m := archRegForAbiReg(r, c)
			a.reg.inputs = append(a.reg.inputs, inputInfo{idx: k, regs: (1 << m)})
			k++
		}
	}
	a.reg.inputs = append(a.reg.inputs, i.inputs...) // These are less constrained, thus should come last
	k = len(i.outputs)
	for _, p := range a.abiInfo.OutParams() {
		for _, r := range p.Registers {
			m := archRegForAbiReg(r, c)
			a.reg.outputs = append(a.reg.outputs, outputInfo{idx: k, regs: (1 << m)})
			k++
		}
	}
	a.reg.outputs = append(a.reg.outputs, i.outputs...)
	a.reg.clobbers = i.clobbers
	return a.reg
}
func (a *AuxCall) ABI() *abi.ABIConfig {
	return a.abiInfo.Config()
}
func (a *AuxCall) ABIInfo() *abi.ABIParamResultInfo {
	return a.abiInfo
}
func (a *AuxCall) ResultReg(c *Config) *regInfo {
	if a.abiInfo.OutRegistersUsed() == 0 {
		return a.reg
	}
	if len(a.reg.inputs) > 0 {
		return a.reg
	}
	k := 0
	for _, p := range a.abiInfo.OutParams() {
		for _, r := range p.Registers {
			m := archRegForAbiReg(r, c)
			a.reg.inputs = append(a.reg.inputs, inputInfo{idx: k, regs: (1 << m)})
			k++
		}
	}
	return a.reg
}

// For ABI register index r, returns the (dense) register number used in
// SSA backend.
func archRegForAbiReg(r abi.RegIndex, c *Config) uint8 {
	var m int8
	if int(r) < len(c.intParamRegs) {
		m = c.intParamRegs[r]
	} else {
		m = c.floatParamRegs[int(r)-len(c.intParamRegs)]
	}
	return uint8(m)
}

// For ABI register index r, returns the register number used in the obj
// package (assembler).
func ObjRegForAbiReg(r abi.RegIndex, c *Config) int16 {
	m := archRegForAbiReg(r, c)
	return c.registers[m].objNum
}

// ArgWidth returns the amount of stack needed for all the inputs
// and outputs of a function or method, including ABI-defined parameter
// slots and ABI-defined spill slots for register-resident parameters.
//
// The name is taken from the types package's ArgWidth(<function type>),
// which predated changes to the ABI; this version handles those changes.
func (a *AuxCall) ArgWidth() int64 {
	return a.abiInfo.ArgWidth()
}

// ParamAssignmentForResult returns the ABI Parameter assignment for result which (indexed 0, 1, etc).
func (a *AuxCall) ParamAssignmentForResult(which int64) *abi.ABIParamAssignment {
	return a.abiInfo.OutParam(int(which))
}

// OffsetOfResult returns the SP offset of result which (indexed 0, 1, etc).
func (a *AuxCall) OffsetOfResult(which int64) int64 {
	n := int64(a.abiInfo.OutParam(int(which)).Offset())
	return n
}

// OffsetOfArg returns the SP offset of argument which (indexed 0, 1, etc).
// If the call is to a method, the receiver is the first argument (i.e., index 0)
func (a *AuxCall) OffsetOfArg(which int64) int64 {
	n := int64(a.abiInfo.InParam(int(which)).Offset())
	return n
}

// RegsOfResult returns the register(s) used for result which (indexed 0, 1, etc).
func (a *AuxCall) RegsOfResult(which int64) []abi.RegIndex {
	return a.abiInfo.OutParam(int(which)).Registers
}

// RegsOfArg returns the register(s) used for argument which (indexed 0, 1, etc).
// If the call is to a method, the receiver is the first argument (i.e., index 0)
func (a *AuxCall) RegsOfArg(which int64) []abi.RegIndex {
	return a.abiInfo.InParam(int(which)).Registers
}

// NameOfResult returns the type of result which (indexed 0, 1, etc).
func (a *AuxCall) NameOfResult(which int64) *ir.Name {
	name := a.abiInfo.OutParam(int(which)).Name
	if name == nil {
		return nil
	}
	return name.(*ir.Name)
}

// TypeOfResult returns the type of result which (indexed 0, 1, etc).
func (a *AuxCall) TypeOfResult(which int64) *types.Type {
	return a.abiInfo.OutParam(int(which)).Type
}

// TypeOfArg returns the type of argument which (indexed 0, 1, etc).
// If the call is to a method, the receiver is the first argument (i.e., index 0)
func (a *AuxCall) TypeOfArg(which int64) *types.Type {
	return a.abiInfo.InParam(int(which)).Type
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

// NResults returns the number of results
func (a *AuxCall) NResults() int64 {
	return int64(len(a.abiInfo.OutParams()))
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
	return int64(len(a.abiInfo.InParams()))
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
	var reg *regInfo
	if paramResultInfo.InRegistersUsed()+paramResultInfo.OutRegistersUsed() > 0 {
		reg = &regInfo{}
	}
	return &AuxCall{Fn: sym, abiInfo: paramResultInfo, reg: reg}
}

// InterfaceAuxCall returns an AuxCall for an interface call.
func InterfaceAuxCall(paramResultInfo *abi.ABIParamResultInfo) *AuxCall {
	var reg *regInfo
	if paramResultInfo.InRegistersUsed()+paramResultInfo.OutRegistersUsed() > 0 {
		reg = &regInfo{}
	}
	return &AuxCall{Fn: nil, abiInfo: paramResultInfo, reg: reg}
}

// ClosureAuxCall returns an AuxCall for a closure call.
func ClosureAuxCall(paramResultInfo *abi.ABIParamResultInfo) *AuxCall {
	var reg *regInfo
	if paramResultInfo.InRegistersUsed()+paramResultInfo.OutRegistersUsed() > 0 {
		reg = &regInfo{}
	}
	return &AuxCall{Fn: nil, abiInfo: paramResultInfo, reg: reg}
}

func (*AuxCall) CanBeAnSSAAux() {}

// OwnAuxCall returns a function's own AuxCall
func OwnAuxCall(fn *obj.LSym, paramResultInfo *abi.ABIParamResultInfo) *AuxCall {
	// TODO if this remains identical to ClosureAuxCall above after new ABI is done, should deduplicate.
	var reg *regInfo
	if paramResultInfo.InRegistersUsed()+paramResultInfo.OutRegistersUsed() > 0 {
		reg = &regInfo{}
	}
	return &AuxCall{Fn: fn, abiInfo: paramResultInfo, reg: reg}
}

const (
	auxNone           auxType = iota
	auxBool                   // auxInt is 0/1 for false/true
	auxInt8                   // auxInt is an 8-bit integer
	auxInt16                  // auxInt is a 16-bit integer
	auxInt32                  // auxInt is a 32-bit integer
	auxInt64                  // auxInt is a 64-bit integer
	auxInt128                 // auxInt represents a 128-bit integer.  Always 0.
	auxUInt8                  // auxInt is an 8-bit unsigned integer
	auxFloat32                // auxInt is a float32 (encoded with math.Float64bits)
	auxFloat64                // auxInt is a float64 (encoded with math.Float64bits)
	auxFlagConstant           // auxInt is a flagConstant
	auxNameOffsetInt8         // aux is a &struct{Name ir.Name, Offset int64}; auxInt is index in parameter registers array
	auxString                 // aux is a string
	auxSym                    // aux is a symbol (a *gc.Node for locals, an *obj.LSym for globals, or nil for none)
	auxSymOff                 // aux is a symbol, auxInt is an offset
	auxSymValAndOff           // aux is a symbol, auxInt is a ValAndOff
	auxTyp                    // aux is a type
	auxTypSize                // aux is a type, auxInt is a size, must have Aux.(Type).Size() == AuxInt
	auxCCop                   // aux is a ssa.Op that represents a flags-to-bool conversion (e.g. LessThan)
	auxCall                   // aux is a *ssa.AuxCall
	auxCallOff                // aux is a *ssa.AuxCall, AuxInt is int64 param (in+out) size

	// architecture specific aux types
	auxARM64BitField     // aux is an arm64 bitfield lsb and width packed into auxInt
	auxS390XRotateParams // aux is a s390x rotate parameters object encoding start bit, end bit and rotate amount
	auxS390XCCMask       // aux is a s390x 4-bit condition code mask
	auxS390XCCMaskInt8   // aux is a s390x 4-bit condition code mask, auxInt is a int8 immediate
	auxS390XCCMaskUint8  // aux is a s390x 4-bit condition code mask, auxInt is a uint8 immediate
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
//  - a *gc.Node, for an offset from SP (the stack pointer)
//  - a *obj.LSym, for an offset from SB (the global pointer)
//  - nil, for no offset
type Sym interface {
	CanBeAnSSASym()
	CanBeAnSSAAux()
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

// validVal reports whether the value can be used
// as an argument to makeValAndOff.
func validVal(val int64) bool {
	return val == int64(int32(val))
}

func makeValAndOff(val, off int32) ValAndOff {
	return ValAndOff(int64(val)<<32 + int64(uint32(off)))
}

func (x ValAndOff) canAdd32(off int32) bool {
	newoff := x.Off64() + int64(off)
	return newoff == int64(int32(newoff))
}
func (x ValAndOff) canAdd64(off int64) bool {
	newoff := x.Off64() + off
	return newoff == int64(int32(newoff))
}

func (x ValAndOff) addOffset32(off int32) ValAndOff {
	if !x.canAdd32(off) {
		panic("invalid ValAndOff.addOffset32")
	}
	return makeValAndOff(x.Val(), x.Off()+off)
}
func (x ValAndOff) addOffset64(off int64) ValAndOff {
	if !x.canAdd64(off) {
		panic("invalid ValAndOff.addOffset64")
	}
	return makeValAndOff(x.Val(), x.Off()+int32(off))
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

// boundsAPI determines which register arguments a bounds check call should use. For an [a:b:c] slice, we do:
//   CMPQ c, cap
//   JA   fail1
//   CMPQ b, c
//   JA   fail2
//   CMPQ a, b
//   JA   fail3
//
// fail1: CALL panicSlice3Acap (c, cap)
// fail2: CALL panicSlice3B (b, c)
// fail3: CALL panicSlice3C (a, b)
//
// When we register allocate that code, we want the same register to be used for
// the first arg of panicSlice3Acap and the second arg to panicSlice3B. That way,
// initializing that register once will satisfy both calls.
// That desire ends up dividing the set of bounds check calls into 3 sets. This function
// determines which set to use for a given panic call.
// The first arg for set 0 should be the second arg for set 1.
// The first arg for set 1 should be the second arg for set 2.
func boundsABI(b int64) int {
	switch BoundsKind(b) {
	case BoundsSlice3Alen,
		BoundsSlice3AlenU,
		BoundsSlice3Acap,
		BoundsSlice3AcapU,
		BoundsConvert:
		return 0
	case BoundsSliceAlen,
		BoundsSliceAlenU,
		BoundsSliceAcap,
		BoundsSliceAcapU,
		BoundsSlice3B,
		BoundsSlice3BU:
		return 1
	case BoundsIndex,
		BoundsIndexU,
		BoundsSliceB,
		BoundsSliceBU,
		BoundsSlice3C,
		BoundsSlice3CU:
		return 2
	default:
		panic("bad BoundsKind")
	}
}

// arm64BitFileld is the GO type of ARM64BitField auxInt.
// if x is an ARM64BitField, then width=x&0xff, lsb=(x>>8)&0xff, and
// width+lsb<64 for 64-bit variant, width+lsb<32 for 32-bit variant.
// the meaning of width and lsb are instruction-dependent.
type arm64BitField int16
