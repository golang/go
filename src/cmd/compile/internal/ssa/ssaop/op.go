// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssaop

import (
	"fmt"
	"strings"

	"cmd/internal/obj"
)

type AuxType int8

const (
	AuxTypeNone           AuxType = iota
	AuxTypeBool                   // auxInt is 0/1 for false/true
	AuxTypeInt8                   // auxInt is an 8-bit integer
	AuxTypeInt16                  // auxInt is a 16-bit integer
	AuxTypeInt32                  // auxInt is a 32-bit integer
	AuxTypeInt64                  // auxInt is a 64-bit integer
	AuxTypeInt128                 // auxInt represents a 128-bit integer.  Always 0.
	AuxTypeUInt8                  // auxInt is an 8-bit unsigned integer
	AuxTypeFloat32                // auxInt is a float32 (encoded with math.Float64bits)
	AuxTypeFloat64                // auxInt is a float64 (encoded with math.Float64bits)
	AuxTypeFlagConstant           // auxInt is a flagConstant
	AuxTypeCCop                   // auxInt is a ssa.Op that represents a flags-to-bool conversion (e.g. LessThan)
	AuxTypeNameOffsetInt8         // aux is a &struct{Name ir.Name, Offset int64}; auxInt is index in parameter registers array
	AuxTypeString                 // aux is a string
	AuxTypeSym                    // aux is a symbol (a *ir.Name for locals, an *obj.LSym for globals, or nil for none)
	AuxTypeSymOff                 // aux is a symbol, auxInt is an offset
	AuxTypeSymValAndOff           // aux is a symbol, auxInt is a ValAndOff
	AuxTypeTyp                    // aux is a type
	AuxTypeTypSize                // aux is a type, auxInt is a size, must have Aux.(Type).Size() == AuxInt
	AuxTypeCall                   // aux is a *ssa.AuxCall
	AuxTypeCallOff                // aux is a *ssa.AuxCall, AuxInt is int64 param (in+out) size

	AuxTypePanicBoundsC  // constant for a bounds failure
	AuxTypePanicBoundsCC // two constants for a bounds failure

	// architecture specific aux types
	AuxTypeARM64BitField          // aux is an arm64 bitfield lsb and width packed into auxInt
	AuxTypeARM64ConditionalParams // aux is a structure, which contains condition, NZCV flags and constant with indicator of using it
	AuxTypeS390XRotateParams      // aux is a s390x rotate parameters object encoding start bit, end bit and rotate amount
	AuxTypeS390XCCMask            // aux is a s390x 4-bit condition code mask
	AuxTypeS390XCCMaskInt8        // aux is a s390x 4-bit condition code mask, auxInt is an int8 immediate
	AuxTypeS390XCCMaskUint8       // aux is a s390x 4-bit condition code mask, auxInt is a uint8 immediate
	AuxTypeSizeAndAlign           // auxInt is an int64 size, aux is an int64 alignment
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
	AuxType           AuxType
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

type InputInfo struct {
	Idx  int     // index in Args array
	Regs RegMask // allowed input registers
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

func (op Op) IsLoweredGetClosurePtr() bool {
	switch op {
	case OpAMD64LoweredGetClosurePtr, OpPPC64LoweredGetClosurePtr, OpARMLoweredGetClosurePtr, OpARM64LoweredGetClosurePtr,
		Op386LoweredGetClosurePtr, OpMIPS64LoweredGetClosurePtr, OpLOONG64LoweredGetClosurePtr, OpS390XLoweredGetClosurePtr, OpMIPSLoweredGetClosurePtr,
		OpRISCV64LoweredGetClosurePtr, OpWasmLoweredGetClosurePtr:
		return true
	}
	return false
}
