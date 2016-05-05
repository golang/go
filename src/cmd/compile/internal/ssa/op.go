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
	argLen            int32 // the number of arugments, -1 if variable length
	asm               obj.As
	generic           bool // this is a generic (arch-independent) opcode
	rematerializeable bool // this op is rematerializeable
	commutative       bool // this operation is commutative (e.g. addition)
	resultInArg0      bool // v and v.Args[0] must be allocated to the same register
}

type inputInfo struct {
	idx  int     // index in Args array
	regs regMask // allowed input registers
}

type regInfo struct {
	inputs   []inputInfo // ordered in register allocation order
	clobbers regMask
	outputs  []regMask // NOTE: values can only have 1 output for now.
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
	auxString               // aux is a string
	auxSym                  // aux is a symbol
	auxSymOff               // aux is a symbol, auxInt is an offset
	auxSymValAndOff         // aux is a symbol, auxInt is a ValAndOff

	auxSymInt32 // aux is a symbol, auxInt is a 32-bit integer
)

// A ValAndOff is used by the several opcodes. It holds
// both a value and a pointer offset.
// A ValAndOff is intended to be encoded into an AuxInt field.
// The zero ValAndOff encodes a value of 0 and an offset of 0.
// The high 32 bits hold a value.
// The low 32 bits hold a pointer offset.
type ValAndOff int64

func (x ValAndOff) Val() int64 {
	return int64(x) >> 32
}
func (x ValAndOff) Off() int64 {
	return int64(int32(x))
}
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

func (x ValAndOff) canAdd(off int64) bool {
	newoff := x.Off() + off
	return newoff == int64(int32(newoff))
}

func (x ValAndOff) add(off int64) int64 {
	if !x.canAdd(off) {
		panic("invalid ValAndOff.add")
	}
	return makeValAndOff(x.Val(), x.Off()+off)
}
