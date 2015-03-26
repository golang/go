// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// An Op encodes the specific operation that a Value performs.
// Opcodes' semantics can be modified by the type and aux fields of the Value.
// For instance, OpAdd can be 32 or 64 bit, signed or unsigned, float or complex, depending on Value.Type.
// Semantics of each op are described below.
// Ops come in two flavors, architecture-independent and architecture-dependent.
type Op int32

// All the opcodes
const (
	OpUnknown Op = iota

	// machine-independent opcodes

	OpNop   // should never be used, appears only briefly during construction,  Has type Void.
	OpThunk // used during ssa construction.  Like OpCopy, but the arg has not been specified yet.

	// 2-input arithmetic
	OpAdd
	OpSub
	OpMul

	// 2-input comparisons
	OpLess

	// constants
	OpConst

	OpArg    // address of a function parameter/result.  Memory input is an arg called ".mem".
	OpGlobal // address of a global variable
	OpFunc   // entry address of a function
	OpCopy   // output = input
	OpPhi    // select an input based on which predecessor we came from

	OpSliceMake // args are ptr/len/cap
	OpSlicePtr
	OpSliceLen
	OpSliceCap

	OpStringMake // args are ptr/len
	OpStringPtr
	OpStringLen

	OpSlice
	OpIndex
	OpIndexAddr

	OpLoad  // args are ptr, memory
	OpStore // args are ptr, value, memory, returns memory

	OpCheckNil   // arg[0] != nil
	OpCheckBound // 0 <= arg[0] < arg[1]

	// function calls.  Arguments to the call have already been written to the stack.
	// Return values appear on the stack.  The method receiver, if any, is treated
	// as a phantom first argument.
	// TODO: closure pointer must be in a register.
	OpCall       // args are function ptr, memory
	OpStaticCall // aux is function, arg is memory

	OpConvert
	OpConvNop

	// These ops return a pointer to a location on the stack.  Aux contains an int64
	// indicating an offset from the base pointer.
	OpFPAddr // offset from FP (+ == args from caller, - == locals)
	OpSPAddr // offset from SP

	// load/store from constant offsets from SP/FP
	// The distinction between FP/SP needs to be maintained until after
	// register allocation because we don't know the size of the frame yet.
	OpLoadFP
	OpLoadSP
	OpStoreFP
	OpStoreSP

	// spill&restore ops for the register allocator.  These are
	// semantically identical to OpCopy; they do not take/return
	// stores like regular memory ops do.  We can get away without memory
	// args because we know there is no aliasing of spill slots on the stack.
	OpStoreReg8
	OpLoadReg8

	// machine-dependent opcodes go here

	// amd64
	OpADDQ
	OpSUBQ
	OpADDCQ // 1 input arg.  output = input + aux.(int64)
	OpSUBCQ // 1 input arg.  output = input - aux.(int64)
	OpNEGQ
	OpCMPQ
	OpCMPCQ // 1 input arg.  Compares input with aux.(int64)
	OpADDL
	OpSETL // generate bool = "flags encode less than"
	OpSETGE

	// InvertFlags reverses direction of flags register interpretation:
	// (InvertFlags (OpCMPQ a b)) == (OpCMPQ b a)
	// This is a pseudo-op which can't appear in assembly output.
	OpInvertFlags

	OpLEAQ  // x+y
	OpLEAQ2 // x+2*y
	OpLEAQ4 // x+4*y
	OpLEAQ8 // x+8*y

	// load/store 8-byte integer register from stack slot.
	OpLoadFP8
	OpLoadSP8
	OpStoreFP8
	OpStoreSP8

	OpMax // sentinel
)

//go:generate stringer -type=Op

type OpInfo struct {
	flags int32

	// assembly template
	// %In: location of input n
	// %On: location of output n
	// %A: print aux with fmt.Print
	asm string

	// returns a reg constraint for the instruction. [0] gives a reg constraint
	// for each input, [1] gives a reg constraint for each output. (Values have
	// exactly one output for now)
	reg [2][]regMask
}

type regMask uint64

var regs386 = [...]string{
	"AX",
	"BX",
	"CX",
	"DX",
	"SI",
	"DI",
	"SP",
	"BP",
	"X0",

	// pseudo registers
	"FLAGS",
	"OVERWRITE0", // the same register as the first input
}

// TODO: match up these with regs386 above
var gp regMask = 0xff
var cx regMask = 0x4
var flags regMask = 1 << 9
var overwrite0 regMask = 1 << 10

const (
	// possible properties of opcodes
	OpFlagCommutative int32 = 1 << iota

	// architecture constants
	Arch386
	ArchAmd64
	ArchArm
)

// general purpose registers, 2 input, 1 output
var gp21 = [2][]regMask{{gp, gp}, {gp}}
var gp21_overwrite = [2][]regMask{{gp, gp}, {overwrite0}}

// general purpose registers, 1 input, 1 output
var gp11 = [2][]regMask{{gp}, {gp}}
var gp11_overwrite = [2][]regMask{{gp}, {overwrite0}}

// shift operations
var shift = [2][]regMask{{gp, cx}, {overwrite0}}

var gp2_flags = [2][]regMask{{gp, gp}, {flags}}
var gp1_flags = [2][]regMask{{gp}, {flags}}
var gpload = [2][]regMask{{gp, 0}, {gp}}
var gpstore = [2][]regMask{{gp, gp, 0}, {0}}

// Opcodes that represent the input Go program
var genericTable = [...]OpInfo{
	// the unknown op is used only during building and should not appear in a
	// fully formed ssa representation.

	OpAdd:  {flags: OpFlagCommutative},
	OpSub:  {},
	OpMul:  {flags: OpFlagCommutative},
	OpLess: {},

	OpConst:  {}, // aux matches the type (e.g. bool, int64 float64)
	OpArg:    {}, // aux is the name of the input variable  TODO:?
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

// Opcodes that appear in an output amd64 program
var amd64Table = [...]OpInfo{
	OpADDQ:  {flags: OpFlagCommutative, asm: "ADDQ\t%I0,%I1,%O0", reg: gp21}, // TODO: overwrite
	OpADDCQ: {asm: "ADDQ\t$%A,%I0,%O0", reg: gp11_overwrite},                 // aux = int64 constant to add
	OpSUBQ:  {asm: "SUBQ\t%I0,%I1,%O0", reg: gp21},
	OpSUBCQ: {asm: "SUBQ\t$%A,%I0,%O0", reg: gp11_overwrite},

	OpCMPQ:  {asm: "CMPQ\t%I0,%I1", reg: gp2_flags}, // compute arg[0]-arg[1] and produce flags
	OpCMPCQ: {asm: "CMPQ\t$%A,%I0", reg: gp1_flags},

	OpLEAQ:  {flags: OpFlagCommutative, asm: "LEAQ\t%A(%I0)(%I1*1),%O0", reg: gp21}, // aux = int64 constant to add
	OpLEAQ2: {asm: "LEAQ\t%A(%I0)(%I1*2),%O0"},
	OpLEAQ4: {asm: "LEAQ\t%A(%I0)(%I1*4),%O0"},
	OpLEAQ8: {asm: "LEAQ\t%A(%I0)(%I1*8),%O0"},

	//OpLoad8:  {asm: "MOVQ\t%A(%I0),%O0", reg: gpload},
	//OpStore8: {asm: "MOVQ\t%I1,%A(%I0)", reg: gpstore},

	OpStaticCall: {asm: "CALL\t%A(SB)"},

	OpCopy: {asm: "MOVQ\t%I0,%O0", reg: gp11},

	// convert from flags back to boolean
	OpSETL: {},

	// ops for load/store to stack
	OpLoadFP8:  {asm: "MOVQ\t%A(FP),%O0"},
	OpLoadSP8:  {asm: "MOVQ\t%A(SP),%O0"},
	OpStoreFP8: {asm: "MOVQ\t%I0,%A(FP)"},
	OpStoreSP8: {asm: "MOVQ\t%I0,%A(SP)"},

	// ops for spilling of registers
	// unlike regular loads & stores, these take no memory argument.
	// They are just like OpCopy but we use them during register allocation.
	// TODO: different widths, float
	OpLoadReg8:  {asm: "MOVQ\t%I0,%O0", reg: gp11},
	OpStoreReg8: {asm: "MOVQ\t%I0,%O0", reg: gp11},
}

// A Table is a list of opcodes with a common set of flags.
type Table struct {
	t     []OpInfo
	flags int32
}

var tables = []Table{
	{genericTable[:], 0},
	{amd64Table[:], ArchAmd64}, // TODO: pick this dynamically
}

// table of opcodes, indexed by opcode ID
var opcodeTable [OpMax]OpInfo

// map from opcode names to opcode IDs
var nameToOp map[string]Op

func init() {
	// build full opcode table
	// Note that the arch-specific table overwrites the generic table
	for _, t := range tables {
		for op, entry := range t.t {
			entry.flags |= t.flags
			opcodeTable[op] = entry
		}
	}
	// build name to opcode mapping
	nameToOp = make(map[string]Op)
	for op := range opcodeTable {
		nameToOp[Op(op).String()] = Op(op)
	}
}
