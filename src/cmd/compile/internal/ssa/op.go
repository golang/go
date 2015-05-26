// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"log"
)

// An Op encodes the specific operation that a Value performs.
// Opcodes' semantics can be modified by the type and aux fields of the Value.
// For instance, OpAdd can be 32 or 64 bit, signed or unsigned, float or complex, depending on Value.Type.
// Semantics of each op are described below.
//
// Ops come in two flavors, architecture-independent and architecture-dependent.
// Architecture-independent opcodes appear in this file.
// Architecture-dependent opcodes appear in op{arch}.go files.
type Op int32

// Opcode ranges, a generic one and one for each architecture.
const (
	opInvalid     Op = 0
	opGenericBase    = 1 + 1000*iota
	opAMD64Base
	op386Base

	opMax // sentinel
)

// Generic opcodes
const (
	opGenericStart Op = opGenericBase + iota

	// 2-input arithmetic
	OpAdd // arg0 + arg1
	OpSub // arg0 - arg1
	OpMul // arg0 * arg1
	OpLsh // arg0 << arg1
	OpRsh // arg0 >> arg1 (signed/unsigned depending on signedness of type)

	// 2-input comparisons
	OpLess // arg0 < arg1

	// constants.  Constant values are stored in the aux field.
	// booleans have a bool aux field, strings have a string aux
	// field, and so on.  All integer types store their value
	// in the aux field as an int64 (including int, uint64, etc.).
	// We could store int8 as an int8, but that won't work for int,
	// as it may be different widths on the host and target.
	OpConst

	OpArg    // address of a function parameter/result.  Memory input is an arg called ".mem".  aux is a string (TODO: make it something other than a string?)
	OpGlobal // the address of a global variable aux.(*gc.Sym)
	OpFunc   // entry address of a function
	OpFP     // frame pointer
	OpSP     // stack pointer

	OpCopy // output = arg0
	OpMove // arg0=destptr, arg1=srcptr, arg2=mem, aux.(int64)=size.  Returns memory.
	OpPhi  // select an argument based on which predecessor block we came from

	OpSliceMake // arg0=ptr, arg1=len, arg2=cap
	OpSlicePtr  // ptr(arg0)
	OpSliceLen  // len(arg0)
	OpSliceCap  // cap(arg0)

	OpStringMake // arg0=ptr, arg1=len
	OpStringPtr  // ptr(arg0)
	OpStringLen  // len(arg0)

	OpLoad       // Load from arg0.  arg1=memory
	OpStore      // Store arg1 to arg0.  arg2=memory.  Returns memory.
	OpArrayIndex // arg0=array, arg1=index.  Returns a[i]
	OpPtrIndex   // arg0=ptr, arg1=index. Computes ptr+sizeof(*v.type)*index, where index is extended to ptrwidth type
	OpIsNonNil   // arg0 != nil
	OpIsInBounds // 0 <= arg0 < arg1

	// function calls.  Arguments to the call have already been written to the stack.
	// Return values appear on the stack.  The method receiver, if any, is treated
	// as a phantom first argument.
	OpCall       // arg0=code pointer, arg1=context ptr, arg2=memory.  Returns memory.
	OpStaticCall // call function aux.(*gc.Sym), arg0=memory.  Returns memory.

	OpConvert // convert arg0 to another type
	OpConvNop // interpret arg0 as another type

	OpOffPtr // arg0 + aux.(int64) (arg0 and result are pointers)

	// spill&restore ops for the register allocator.  These are
	// semantically identical to OpCopy; they do not take/return
	// stores like regular memory ops do.  We can get away without memory
	// args because we know there is no aliasing of spill slots on the stack.
	OpStoreReg8
	OpLoadReg8

	// used during ssa construction.  Like OpCopy, but the arg has not been specified yet.
	OpFwdRef

	OpGenericEnd
)

// GlobalOffset represents a fixed offset within a global variable
type GlobalOffset struct {
	Global interface{} // holds a *gc.Sym
	Offset int64
}

// offset adds x to the location specified by g and returns it.
func (g GlobalOffset) offset(x int64) GlobalOffset {
	y := g.Offset
	z := x + y
	if x^y >= 0 && x^z < 0 {
		log.Panicf("offset overflow %d %d\n", x, y)
	}
	return GlobalOffset{g.Global, z}
}

func (g GlobalOffset) String() string {
	return fmt.Sprintf("%v+%d", g.Global, g.Offset)
}

//go:generate stringer -type=Op

type opInfo struct {
	flags int32

	// returns a reg constraint for the instruction. [0] gives a reg constraint
	// for each input, [1] gives a reg constraint for each output. (Values have
	// exactly one output for now)
	reg [2][]regMask
}

const (
	// possible properties of opcodes
	OpFlagCommutative int32 = 1 << iota
)

// Opcodes that represent the input Go program
var genericTable = map[Op]opInfo{
	// the unknown op is used only during building and should not appear in a
	// fully formed ssa representation.

	OpAdd:  {flags: OpFlagCommutative},
	OpSub:  {},
	OpMul:  {flags: OpFlagCommutative},
	OpLess: {},

	OpConst:  {}, // aux matches the type (e.g. bool, int64 float64)
	OpArg:    {}, // aux is the name of the input variable.  Currently only ".mem" is used
	OpGlobal: {}, // address of a global variable
	OpFunc:   {},
	OpCopy:   {},
	OpPhi:    {},

	OpConvNop: {}, // aux is the type to convert to

	/*
		// build and take apart slices
		{name: "slicemake"}, // (ptr,len,cap) -> slice
		{name: "sliceptr"},  // pointer part of slice
		{name: "slicelen"},  // length part of slice
		{name: "slicecap"},  // capacity part of slice

		// build and take apart strings
		{name: "stringmake"}, // (ptr,len) -> string
		{name: "stringptr"},  // pointer part of string
		{name: "stringlen"},  // length part of string

		// operations on arrays/slices/strings
		{name: "slice"},     // (s, i, j) -> s[i:j]
		{name: "index"},     // (mem, ptr, idx) -> val
		{name: "indexaddr"}, // (ptr, idx) -> ptr

		// loads & stores
		{name: "load"},  // (mem, check, ptr) -> val
		{name: "store"}, // (mem, check, ptr, val) -> mem

		// checks
		{name: "checknil"},   // (mem, ptr) -> check
		{name: "checkbound"}, // (mem, idx, len) -> check

		// functions
		{name: "call"},

		// builtins
		{name: "len"},
		{name: "convert"},

		// tuples
		{name: "tuple"},         // build a tuple out of its arguments
		{name: "extract"},       // aux is an int64.  Extract that index out of a tuple
		{name: "extractsuffix"}, // aux is an int64.  Slice a tuple with [aux:]

	*/
}

// table of opcodes, indexed by opcode ID
var opcodeTable [opMax]opInfo

func init() {
	for op, info := range genericTable {
		opcodeTable[op] = info
	}
}
