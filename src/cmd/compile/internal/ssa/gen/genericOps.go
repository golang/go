// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

var genericOps = []opData{
	// 2-input arithmetic
	// Types must be consistent with Go typing.  Add, for example, must take two values
	// of the same type and produces that same type.
	{name: "Add"}, // arg0 + arg1
	{name: "Sub"}, // arg0 - arg1
	{name: "Mul"}, // arg0 * arg1
	{name: "Lsh"}, // arg0 << arg1
	{name: "Rsh"}, // arg0 >> arg1 (signed/unsigned depending on signedness of type)

	// 2-input comparisons
	{name: "Eq"},      // arg0 == arg1
	{name: "Neq"},     // arg0 != arg1
	{name: "Less"},    // arg0 < arg1
	{name: "Leq"},     // arg0 <= arg1
	{name: "Greater"}, // arg0 > arg1
	{name: "Geq"},     // arg0 <= arg1

	// 1-input ops
	{name: "Not"}, // !arg0

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

	// Safety checks
	{name: "IsNonNil"},   // arg0 != nil
	{name: "IsInBounds"}, // 0 <= arg0 < arg1

	// Indexing operations
	{name: "ArrayIndex"}, // arg0=array, arg1=index.  Returns a[i]
	{name: "PtrIndex"},   // arg0=ptr, arg1=index. Computes ptr+sizeof(*v.type)*index, where index is extended to ptrwidth type
	{name: "OffPtr"},     // arg0 + auxint (arg0 and result are pointers)

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
	// TODO: remove these, make them arch-specific ops stored
	// in the fields of Config instead.
	{name: "StoreReg8"},
	{name: "LoadReg8"},

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
