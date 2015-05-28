// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// amd64-specific opcodes

const (
	blockAMD64Start BlockKind = blockAMD64Base + iota

	BlockEQ
	BlockNE
	BlockLT
	BlockLE
	BlockGT
	BlockGE
	BlockULT
	BlockULE
	BlockUGT
	BlockUGE
)

const (
	opAMD64start Op = opAMD64Base + iota

	// Suffixes encode the bit width of various instructions.
	// Q = 64 bit, L = 32 bit, W = 16 bit, B = 8 bit

	// arithmetic
	OpADDQ      // arg0 + arg1
	OpADDQconst // arg + aux.(int64)
	OpSUBQ      // arg0 - arg1
	OpSUBQconst // arg - aux.(int64)
	OpMULQ      // arg0 * arg1
	OpMULQconst // arg * aux.(int64)
	OpSHLQ      // arg0 << arg1
	OpSHLQconst // arg << aux.(int64)
	OpNEGQ      // -arg
	OpADDL      // arg0 + arg1

	// Flags value generation.
	// We pretend the flags type is an opaque thing that comparisons generate
	// and from which we can extract boolean conditions like <, ==, etc.
	OpCMPQ      // arg0 compare to arg1
	OpCMPQconst // arg0 compare to aux.(int64)
	OpTESTQ     // (arg0 & arg1) compare to 0
	OpTESTB     // (arg0 & arg1) compare to 0

	// These opcodes extract a particular boolean condition from a flags value.
	OpSETEQ // extract == condition from arg0
	OpSETNE // extract != condition from arg0
	OpSETL  // extract signed < condition from arg0
	OpSETG  // extract signed > condition from arg0
	OpSETGE // extract signed >= condition from arg0
	OpSETB  // extract unsigned < condition from arg0

	// InvertFlags reverses the direction of a flags type interpretation:
	// (InvertFlags (CMPQ a b)) == (CMPQ b a)
	// So if we want (SETL (CMPQ a b)) but we can't do that because a is a constant,
	// then we do (SETL (InvertFlags (CMPQ b a))) instead.
	// Rewrites will convert this to (SETG (CMPQ b a)).
	// InvertFlags is a pseudo-op which can't appear in assembly output.
	OpInvertFlags // reverse direction of arg0

	OpLEAQ       // arg0 + arg1 + aux.(int64)
	OpLEAQ2      // arg0 + 2*arg1 + aux.(int64)
	OpLEAQ4      // arg0 + 4*arg1 + aux.(int64)
	OpLEAQ8      // arg0 + 8*arg1 + aux.(int64)
	OpLEAQglobal // no args.  address of aux.(GlobalOffset)

	// Load/store from general address
	OpMOVBload // Load from arg0+aux.(int64).  arg1=memory
	OpMOVBQZXload
	OpMOVBQSXload
	OpMOVQload
	OpMOVQstore     // Store arg1 to arg0+aux.(int64).  arg2=memory, returns memory.
	OpMOVQloadidx8  // Load from arg0+arg1*8+aux.(int64).  arg2=memory
	OpMOVQstoreidx8 // Store arg2 to arg0+arg1*8+aux.(int64).  arg3=memory, returns memory.

	// Load/store from global.  Same as the above loads, but arg0 is missing and aux is a GlobalOffset instead of an int64.
	OpMOVQloadglobal  // arg0 = memory
	OpMOVQstoreglobal // store arg0.  arg1=memory, returns memory.

	// materialize a constant into a register
	OpMOVQconst // (takes no arguments)

	// move memory
	OpREPMOVSB // arg0=destptr, arg1=srcptr, arg2=len, arg3=mem
)

type regMask uint64

var regsAMD64 = [...]string{
	"AX",
	"CX",
	"DX",
	"BX",
	"SP",
	"BP",
	"SI",
	"DI",
	"R8",
	"R9",
	"R10",
	"R11",
	"R12",
	"R13",
	"R14",
	"R15",

	// pseudo registers
	"FP",
	"FLAGS",
	"OVERWRITE0", // the same register as the first input
}

var gp regMask = 0x1ffff   // all integer registers including SP&FP
var gpout regMask = 0xffef // integer registers not including SP&FP
var cx regMask = 1 << 1
var si regMask = 1 << 6
var di regMask = 1 << 7
var flags regMask = 1 << 17

var (
	// gp = general purpose (integer) registers
	gp21      = [2][]regMask{{gp, gp}, {gpout}} // 2 input, 1 output
	gp11      = [2][]regMask{{gp}, {gpout}}     // 1 input, 1 output
	gp01      = [2][]regMask{{}, {gpout}}       // 0 input, 1 output
	shift     = [2][]regMask{{gp, cx}, {gpout}} // shift operations
	gp2_flags = [2][]regMask{{gp, gp}, {flags}} // generate flags from 2 gp regs
	gp1_flags = [2][]regMask{{gp}, {flags}}     // generate flags from 1 gp reg

	gpload     = [2][]regMask{{gp, 0}, {gpout}}
	gploadidx  = [2][]regMask{{gp, gp, 0}, {gpout}}
	gpstore    = [2][]regMask{{gp, gp, 0}, {0}}
	gpstoreidx = [2][]regMask{{gp, gp, gp, 0}, {0}}

	gpload_stack  = [2][]regMask{{0}, {gpout}}
	gpstore_stack = [2][]regMask{{gp, 0}, {0}}
)

// Opcodes that appear in an output amd64 program
var amd64Table = map[Op]opInfo{
	OpADDQ:      {flags: OpFlagCommutative, reg: gp21}, // TODO: overwrite
	OpADDQconst: {reg: gp11},                           // aux = int64 constant to add
	OpSUBQ:      {reg: gp21},
	OpSUBQconst: {reg: gp11},
	OpMULQ:      {reg: gp21},
	OpMULQconst: {reg: gp11},
	OpSHLQ:      {reg: gp21},
	OpSHLQconst: {reg: gp11},

	OpCMPQ:      {reg: gp2_flags}, // compute arg[0]-arg[1] and produce flags
	OpCMPQconst: {reg: gp1_flags},
	OpTESTQ:     {reg: gp2_flags},
	OpTESTB:     {reg: gp2_flags},

	OpLEAQ:       {flags: OpFlagCommutative, reg: gp21}, // aux = int64 constant to add
	OpLEAQ2:      {},
	OpLEAQ4:      {},
	OpLEAQ8:      {},
	OpLEAQglobal: {reg: gp01},

	// loads and stores
	OpMOVBload:      {reg: gpload},
	OpMOVQload:      {reg: gpload},
	OpMOVQstore:     {reg: gpstore},
	OpMOVQloadidx8:  {reg: gploadidx},
	OpMOVQstoreidx8: {reg: gpstoreidx},

	OpMOVQconst: {reg: gp01},

	OpStaticCall: {},

	OpCopy:    {reg: gp11}, // TODO: make arch-specific
	OpConvNop: {reg: gp11}, // TODO: make arch-specific.  Or get rid of this altogether.

	// convert from flags back to boolean
	OpSETL: {},

	// ops for spilling of registers
	// unlike regular loads & stores, these take no memory argument.
	// They are just like OpCopy but we use them during register allocation.
	// TODO: different widths, float
	OpLoadReg8:  {},
	OpStoreReg8: {},

	OpREPMOVSB: {reg: [2][]regMask{{di, si, cx, 0}, {0}}}, // TODO: record that si/di/cx are clobbered
}

func init() {
	for op, info := range amd64Table {
		opcodeTable[op] = info
	}
}
