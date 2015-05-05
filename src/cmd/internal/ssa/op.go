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

	OpNop    // should never be used, appears only briefly during construction,  Has type Void.
	OpFwdRef // used during ssa construction.  Like OpCopy, but the arg has not been specified yet.

	// 2-input arithmetic
	OpAdd
	OpSub
	OpMul

	// 2-input comparisons
	OpLess

	// constants.  Constant values are stored in the aux field.
	// booleans have a bool aux field, strings have a string aux
	// field, and so on.  All integer types store their value
	// in the aux field as an int64 (including int, uint64, etc.).
	// We could store int8 as an int8, but that won't work for int,
	// as it may be different widths on the host and target.
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

	OpSliceIndex
	OpSliceIndexAddr

	OpLoad  // args are ptr, memory.  Loads from ptr+aux.(int64)
	OpStore // args are ptr, value, memory, returns memory.  Stores to ptr+aux.(int64)

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
	OpMULQ
	OpMULCQ // output = input * aux.(int64)
	OpSHLQ  // output = input0 << input1
	OpSHLCQ // output = input << aux.(int64)
	OpNEGQ
	OpCMPQ
	OpCMPCQ // 1 input arg.  Compares input with aux.(int64)
	OpADDL
	OpTESTQ // compute flags of arg[0] & arg[1]
	OpSETEQ
	OpSETNE

	// generate boolean based on the flags setting
	OpSETL  // less than
	OpSETGE // >=
	OpSETB  // "below" = unsigned less than

	// InvertFlags reverses direction of flags register interpretation:
	// (InvertFlags (OpCMPQ a b)) == (OpCMPQ b a)
	// This is a pseudo-op which can't appear in assembly output.
	OpInvertFlags

	OpLEAQ  // x+y
	OpLEAQ2 // x+2*y
	OpLEAQ4 // x+4*y
	OpLEAQ8 // x+8*y

	OpMOVQload   // (ptr, mem): loads from ptr+aux.(int64)
	OpMOVQstore  // (ptr, val, mem): stores val to ptr+aux.(int64), returns mem
	OpMOVQload8  // (ptr,idx,mem): loads from ptr+idx*8+aux.(int64)
	OpMOVQstore8 // (ptr,idx,val,mem): stores to ptr+idx*8+aux.(int64), returns mem

	// load/store 8-byte integer register from stack slot.
	OpMOVQloadFP
	OpMOVQloadSP
	OpMOVQstoreFP
	OpMOVQstoreSP

	// materialize a constant into a register
	OpMOVQconst

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
	"CX",
	"DX",
	"BX",
	"SP",
	"BP",
	"SI",
	"DI",

	// pseudo registers
	"FLAGS",
	"OVERWRITE0", // the same register as the first input
}

// TODO: match up these with regs386 above
var gp regMask = 0xef
var cx regMask = 0x2
var flags regMask = 1 << 8
var overwrite0 regMask = 1 << 9

const (
	// possible properties of opcodes
	OpFlagCommutative int32 = 1 << iota

	// architecture constants
	Arch386
	ArchAMD64
	ArchARM
)

// general purpose registers, 2 input, 1 output
var gp21 = [2][]regMask{{gp, gp}, {gp}}
var gp21_overwrite = [2][]regMask{{gp, gp}, {gp}}

// general purpose registers, 1 input, 1 output
var gp11 = [2][]regMask{{gp}, {gp}}
var gp11_overwrite = [2][]regMask{{gp}, {gp}}

// general purpose registers, 0 input, 1 output
var gp01 = [2][]regMask{{}, {gp}}

// shift operations
var shift = [2][]regMask{{gp, cx}, {gp}}

var gp2_flags = [2][]regMask{{gp, gp}, {flags}}
var gp1_flags = [2][]regMask{{gp}, {flags}}
var gpload = [2][]regMask{{gp, 0}, {gp}}
var gploadX = [2][]regMask{{gp, gp, 0}, {gp}} // indexed loads
var gpstore = [2][]regMask{{gp, gp, 0}, {0}}
var gpstoreX = [2][]regMask{{gp, gp, gp, 0}, {0}} // indexed stores

var gpload_stack = [2][]regMask{{0}, {gp}}
var gpstore_stack = [2][]regMask{{gp, 0}, {0}}

// Opcodes that represent the input Go program
var genericTable = [...]OpInfo{
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

// Opcodes that appear in an output amd64 program
var amd64Table = [...]OpInfo{
	OpADDQ:  {flags: OpFlagCommutative, asm: "ADDQ\t%I0,%I1,%O0", reg: gp21}, // TODO: overwrite
	OpADDCQ: {asm: "ADDQ\t$%A,%I0,%O0", reg: gp11_overwrite},                 // aux = int64 constant to add
	OpSUBQ:  {asm: "SUBQ\t%I0,%I1,%O0", reg: gp21},
	OpSUBCQ: {asm: "SUBQ\t$%A,%I0,%O0", reg: gp11_overwrite},
	OpMULQ:  {asm: "MULQ\t%I0,%I1,%O0", reg: gp21},
	OpMULCQ: {asm: "MULQ\t$%A,%I0,%O0", reg: gp11_overwrite},
	OpSHLQ:  {asm: "SHLQ\t%I0,%I1,%O0", reg: gp21},
	OpSHLCQ: {asm: "SHLQ\t$%A,%I0,%O0", reg: gp11_overwrite},

	OpCMPQ:  {asm: "CMPQ\t%I0,%I1", reg: gp2_flags}, // compute arg[0]-arg[1] and produce flags
	OpCMPCQ: {asm: "CMPQ\t$%A,%I0", reg: gp1_flags},
	OpTESTQ: {asm: "TESTQ\t%I0,%I1", reg: gp2_flags},

	OpLEAQ:  {flags: OpFlagCommutative, asm: "LEAQ\t%A(%I0)(%I1*1),%O0", reg: gp21}, // aux = int64 constant to add
	OpLEAQ2: {asm: "LEAQ\t%A(%I0)(%I1*2),%O0"},
	OpLEAQ4: {asm: "LEAQ\t%A(%I0)(%I1*4),%O0"},
	OpLEAQ8: {asm: "LEAQ\t%A(%I0)(%I1*8),%O0"},

	// loads and stores
	OpMOVQload:   {asm: "MOVQ\t%A(%I0),%O0", reg: gpload},
	OpMOVQstore:  {asm: "MOVQ\t%I1,%A(%I0)", reg: gpstore},
	OpMOVQload8:  {asm: "MOVQ\t%A(%I0)(%I1*8),%O0", reg: gploadX},
	OpMOVQstore8: {asm: "MOVQ\t%I2,%A(%I0)(%I1*8)", reg: gpstoreX},

	OpMOVQconst: {asm: "MOVQ\t$%A,%O0", reg: gp01},

	OpStaticCall: {asm: "CALL\t%A(SB)"},

	OpCopy: {asm: "MOVQ\t%I0,%O0", reg: gp11},

	// convert from flags back to boolean
	OpSETL: {},

	// ops for load/store to stack
	OpMOVQloadFP:  {asm: "MOVQ\t%A(FP),%O0", reg: gpload_stack},  // mem -> value
	OpMOVQloadSP:  {asm: "MOVQ\t%A(SP),%O0", reg: gpload_stack},  // mem -> value
	OpMOVQstoreFP: {asm: "MOVQ\t%I0,%A(FP)", reg: gpstore_stack}, // mem, value -> mem
	OpMOVQstoreSP: {asm: "MOVQ\t%I0,%A(SP)", reg: gpstore_stack}, // mem, value -> mem

	// ops for spilling of registers
	// unlike regular loads & stores, these take no memory argument.
	// They are just like OpCopy but we use them during register allocation.
	// TODO: different widths, float
	OpLoadReg8:  {asm: "MOVQ\t%I0,%O0"},
	OpStoreReg8: {asm: "MOVQ\t%I0,%O0"},
}

// A Table is a list of opcodes with a common set of flags.
type Table struct {
	t     []OpInfo
	flags int32
}

var tables = []Table{
	{genericTable[:], 0},
	{amd64Table[:], ArchAMD64}, // TODO: pick this dynamically
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
