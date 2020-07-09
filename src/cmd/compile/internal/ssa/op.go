// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/internal/obj"
	"fmt"
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

type auxType int8

const (
	auxNone         auxType = iota
	auxBool                 // auxInt is 0/1 for false/true
	auxInt8                 // auxInt is an 8-bit integer
	auxInt16                // auxInt is a 16-bit integer
	auxInt32                // auxInt is a 32-bit integer
	auxInt64                // auxInt is a 64-bit integer
	auxInt128               // auxInt represents a 128-bit integer.  Always 0.
	auxFloat32              // auxInt is a float32 (encoded with math.Float64bits)
	auxFloat64              // auxInt is a float64 (encoded with math.Float64bits)
	auxFlagConstant         // auxInt is a flagConstant
	auxString               // aux is a string
	auxSym                  // aux is a symbol (a *gc.Node for locals, an *obj.LSym for globals, or nil for none)
	auxSymOff               // aux is a symbol, auxInt is an offset
	auxSymValAndOff         // aux is a symbol, auxInt is a ValAndOff
	auxTyp                  // aux is a type
	auxTypSize              // aux is a type, auxInt is a size, must have Aux.(Type).Size() == AuxInt
	auxCCop                 // aux is a ssa.Op that represents a flags-to-bool conversion (e.g. LessThan)

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
	String() string
	CanBeAnSSASym()
}

// A ValAndOff is used by the several opcodes. It holds
// both a value and a pointer offset.
// A ValAndOff is intended to be encoded into an AuxInt field.
// The zero ValAndOff encodes a value of 0 and an offset of 0.
// The high 32 bits hold a value.
// The low 32 bits hold a pointer offset.
type ValAndOff int64

func (x ValAndOff) Val() int64   { return int64(x) >> 32 }
func (x ValAndOff) Val32() int32 { return int32(int64(x) >> 32) }
func (x ValAndOff) Val16() int16 { return int16(int64(x) >> 32) }
func (x ValAndOff) Val8() int8   { return int8(int64(x) >> 32) }

func (x ValAndOff) Off() int64   { return int64(int32(x)) }
func (x ValAndOff) Off32() int32 { return int32(x) }

func (x ValAndOff) Int64() int64 {
	return int64(x)
}
func (x ValAndOff) String() string {
	return fmt.Sprintf("val=%d,off=%d", x.Val(), x.Off())
}

// validVal reports whether the value can be used
// as an argument to makeValAndOff.
func validVal(val int64) bool {
	return val == int64(int32(val))
}

// validOff reports whether the offset can be used
// as an argument to makeValAndOff.
func validOff(off int64) bool {
	return off == int64(int32(off))
}

// validValAndOff reports whether we can fit the value and offset into
// a ValAndOff value.
func validValAndOff(val, off int64) bool {
	if !validVal(val) {
		return false
	}
	if !validOff(off) {
		return false
	}
	return true
}

// makeValAndOff encodes a ValAndOff into an int64 suitable for storing in an AuxInt field.
func makeValAndOff(val, off int64) int64 {
	if !validValAndOff(val, off) {
		panic("invalid makeValAndOff")
	}
	return ValAndOff(val<<32 + int64(uint32(off))).Int64()
}
func makeValAndOff32(val, off int32) ValAndOff {
	return ValAndOff(int64(val)<<32 + int64(uint32(off)))
}

func (x ValAndOff) canAdd(off int64) bool {
	newoff := x.Off() + off
	return newoff == int64(int32(newoff))
}

func (x ValAndOff) canAdd32(off int32) bool {
	newoff := x.Off() + int64(off)
	return newoff == int64(int32(newoff))
}

func (x ValAndOff) add(off int64) int64 {
	if !x.canAdd(off) {
		panic("invalid ValAndOff.add")
	}
	return makeValAndOff(x.Val(), x.Off()+off)
}

func (x ValAndOff) addOffset32(off int32) ValAndOff {
	if !x.canAdd32(off) {
		panic("invalid ValAndOff.add")
	}
	return ValAndOff(makeValAndOff(x.Val(), x.Off()+int64(off)))
}

func (x ValAndOff) addOffset64(off int64) ValAndOff {
	if !x.canAdd(off) {
		panic("invalid ValAndOff.add")
	}
	return ValAndOff(makeValAndOff(x.Val(), x.Off()+off))
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
		BoundsSlice3AcapU:
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
