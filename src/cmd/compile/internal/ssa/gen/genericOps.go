// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var genericOps = []opData{
	// 2-input arithmetic
	// Types must be consistent with Go typing.  Add, for example, must take two values
	// of the same type and produces that same type.
	{name: "Add8"}, // arg0 + arg1
	{name: "Add16"},
	{name: "Add32"},
	{name: "Add64"},
	{name: "Add8U"},
	{name: "Add16U"},
	{name: "Add32U"},
	{name: "Add64U"},
	{name: "AddPtr"},
	// TODO: Add32F, Add64F, Add64C, Add128C

	{name: "Sub8"}, // arg0 - arg1
	{name: "Sub16"},
	{name: "Sub32"},
	{name: "Sub64"},
	{name: "Sub8U"},
	{name: "Sub16U"},
	{name: "Sub32U"},
	{name: "Sub64U"},
	// TODO: Sub32F, Sub64F, Sub64C, Sub128C

	{name: "Mul8"}, // arg0 * arg1
	{name: "Mul16"},
	{name: "Mul32"},
	{name: "Mul64"},
	{name: "Mul8U"},
	{name: "Mul16U"},
	{name: "Mul32U"},
	{name: "Mul64U"},
	{name: "MulPtr"}, // MulPtr is used for address calculations

	{name: "Lsh8"}, // arg0 << arg1
	{name: "Lsh16"},
	{name: "Lsh32"},
	{name: "Lsh64"},

	{name: "Rsh8"}, // arg0 >> arg1
	{name: "Rsh8U"},
	{name: "Rsh16"},
	{name: "Rsh16U"},
	{name: "Rsh32"},
	{name: "Rsh32U"},
	{name: "Rsh64"},
	{name: "Rsh64U"},

	// 2-input comparisons
	{name: "Eq8"}, // arg0 == arg1
	{name: "Eq16"},
	{name: "Eq32"},
	{name: "Eq64"},

	{name: "Neq8"}, // arg0 != arg1
	{name: "Neq16"},
	{name: "Neq32"},
	{name: "Neq64"},

	{name: "Less8"}, // arg0 < arg1
	{name: "Less8U"},
	{name: "Less16"},
	{name: "Less16U"},
	{name: "Less32"},
	{name: "Less32U"},
	{name: "Less64"},
	{name: "Less64U"},

	{name: "Leq8"}, // arg0 <= arg1
	{name: "Leq8U"},
	{name: "Leq16"},
	{name: "Leq16U"},
	{name: "Leq32"},
	{name: "Leq32U"},
	{name: "Leq64"},
	{name: "Leq64U"},

	{name: "Greater8"}, // arg0 > arg1
	{name: "Greater8U"},
	{name: "Greater16"},
	{name: "Greater16U"},
	{name: "Greater32"},
	{name: "Greater32U"},
	{name: "Greater64"},
	{name: "Greater64U"},

	{name: "Geq8"}, // arg0 <= arg1
	{name: "Geq8U"},
	{name: "Geq16"},
	{name: "Geq16U"},
	{name: "Geq32"},
	{name: "Geq32U"},
	{name: "Geq64"},
	{name: "Geq64U"},

	// 1-input ops
	{name: "Not"}, // !arg0

	{name: "Neg8"}, // - arg0
	{name: "Neg16"},
	{name: "Neg32"},
	{name: "Neg64"},
	{name: "Neg8U"},
	{name: "Neg16U"},
	{name: "Neg32U"},
	{name: "Neg64U"},

	// Data movement
	{name: "Phi"},  // select an argument based on which predecessor block we came from
	{name: "Copy"}, // output = arg0

	// constants.  Constant values are stored in the aux field.
	// booleans have a bool aux field, strings have a string aux
	// field, and so on.  All integer types store their value
	// in the aux field as an int64 (including int, uint64, etc.).
	// We could store int8 as an int8, but that won't work for int,
	// as it may be different widths on the host and target.
	{name: "Const"},

	// Constant-like things
	{name: "Arg"}, // memory input to the function.

	// The address of a variable.  arg0 is the base pointer (SB or SP, depending
	// on whether it is a global or stack variable).  The Aux field identifies the
	// variable.  It will be either an *ExternSymbol (with arg0=SB), *ArgSymbol (arg0=SP),
	// or *AutoSymbol (arg0=SP).
	{name: "Addr"}, // Address of a variable.  Arg0=SP or SB.  Aux identifies the variable.

	{name: "SP"},   // stack pointer
	{name: "SB"},   // static base pointer (a.k.a. globals pointer)
	{name: "Func"}, // entry address of a function

	// Memory operations
	{name: "Load"},  // Load from arg0.  arg1=memory
	{name: "Store"}, // Store arg1 to arg0.  arg2=memory.  Returns memory.
	{name: "Move"},  // arg0=destptr, arg1=srcptr, arg2=mem, auxint=size.  Returns memory.
	{name: "Zero"},  // arg0=destptr, arg1=mem, auxint=size. Returns memory.

	// Function calls.  Arguments to the call have already been written to the stack.
	// Return values appear on the stack.  The method receiver, if any, is treated
	// as a phantom first argument.
	{name: "ClosureCall"}, // arg0=code pointer, arg1=context ptr, arg2=memory.  Returns memory.
	{name: "StaticCall"},  // call function aux.(*gc.Sym), arg0=memory.  Returns memory.

	// Conversions
	{name: "Convert"}, // convert arg0 to another type
	{name: "ConvNop"}, // interpret arg0 as another type

	// Automatically inserted safety checks
	{name: "IsNonNil"},   // arg0 != nil
	{name: "IsInBounds"}, // 0 <= arg0 < arg1

	// Indexing operations
	{name: "ArrayIndex"},   // arg0=array, arg1=index.  Returns a[i]
	{name: "PtrIndex"},     // arg0=ptr, arg1=index. Computes ptr+sizeof(*v.type)*index, where index is extended to ptrwidth type
	{name: "OffPtr"},       // arg0 + auxint (arg0 and result are pointers)
	{name: "StructSelect"}, // arg0=struct, auxint=field offset.  Returns field at that offset (size=size of result type)

	// Slices
	{name: "SliceMake"}, // arg0=ptr, arg1=len, arg2=cap
	{name: "SlicePtr"},  // ptr(arg0)
	{name: "SliceLen"},  // len(arg0)
	{name: "SliceCap"},  // cap(arg0)

	// Strings
	{name: "StringMake"}, // arg0=ptr, arg1=len
	{name: "StringPtr"},  // ptr(arg0)
	{name: "StringLen"},  // len(arg0)

	// Spill&restore ops for the register allocator.  These are
	// semantically identical to OpCopy; they do not take/return
	// stores like regular memory ops do.  We can get away without memory
	// args because we know there is no aliasing of spill slots on the stack.
	{name: "StoreReg"},
	{name: "LoadReg"},

	// Used during ssa construction.  Like Copy, but the arg has not been specified yet.
	{name: "FwdRef"},
}

//     kind           control    successors
//   ------------------------------------------
//     Exit        return mem                []
//    Plain               nil            [next]
//       If   a boolean Value      [then, else]
//     Call               mem  [nopanic, panic]  (control opcode should be OpCall or OpStaticCall)

var genericBlocks = []blockData{
	{name: "Exit"},  // no successors.  There should only be 1 of these.
	{name: "Dead"},  // no successors; determined to be dead but not yet removed
	{name: "Plain"}, // a single successor
	{name: "If"},    // 2 successors, if control goto Succs[0] else goto Succs[1]
	{name: "Call"},  // 2 successors, normal return and panic
	// TODO(khr): BlockPanic for the built-in panic call, has 1 edge to the exit block
}

func init() {
	archs = append(archs, arch{"generic", genericOps, genericBlocks, nil})
}
