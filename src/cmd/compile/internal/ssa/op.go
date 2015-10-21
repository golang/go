// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// An Op encodes the specific operation that a Value performs.
// Opcodes' semantics can be modified by the type and aux fields of the Value.
// For instance, OpAdd can be 32 or 64 bit, signed or unsigned, float or complex, depending on Value.Type.
// Semantics of each op are described in the opcode files in gen/*Ops.go.
// There is one file for generic (architecture-independent) ops and one file
// for each architecture.
type Op int32

type opInfo struct {
	name    string
	asm     int
	reg     regInfo
	generic bool // this is a generic (arch-independent) opcode
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

// A StoreConst is used by the MOVXstoreconst opcodes.  It holds
// both the value to store and an offset from the store pointer.
// A StoreConst is intended to be encoded into an AuxInt field.
// The zero StoreConst encodes a value of 0 and an offset of 0.
// The high 32 bits hold a value to be stored.
// The low 32 bits hold a pointer offset.
type StoreConst int64

func (sc StoreConst) Val() int64 {
	return int64(sc) >> 32
}
func (sc StoreConst) Off() int64 {
	return int64(int32(sc))
}
func (sc StoreConst) Int64() int64 {
	return int64(sc)
}

// validStoreConstOff reports whether the offset can be used
// as an argument to makeStoreConst.
func validStoreConstOff(off int64) bool {
	return off == int64(int32(off))
}

// validStoreConst reports whether we can fit the value and offset into
// a StoreConst value.
func validStoreConst(val, off int64) bool {
	if val != int64(int32(val)) {
		return false
	}
	if !validStoreConstOff(off) {
		return false
	}
	return true
}

// encode encodes a StoreConst into an int64 suitable for storing in an AuxInt field.
func makeStoreConst(val, off int64) int64 {
	if !validStoreConst(val, off) {
		panic("invalid makeStoreConst")
	}
	return StoreConst(val<<32 + int64(uint32(off))).Int64()
}

func (sc StoreConst) canAdd(off int64) bool {
	newoff := sc.Off() + off
	return newoff == int64(int32(newoff))
}
func (sc StoreConst) add(off int64) int64 {
	if !sc.canAdd(off) {
		panic("invalid StoreConst.add")
	}
	return makeStoreConst(sc.Val(), sc.Off()+off)
}
