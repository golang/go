// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

// Generic opcodes typically specify a width. The inputs and outputs
// of that op are the given number of bits wide. There is no notion of
// "sign", so Add32 can be used both for signed and unsigned 32-bit
// addition.

// Signed/unsigned is explicit with the extension ops
// (SignExt*/ZeroExt*) and implicit as the arg to some opcodes
// (e.g. the second argument to shifts is unsigned). If not mentioned,
// all args take signed inputs, or don't care whether their inputs
// are signed or unsigned.

// Unused portions of AuxInt are filled by sign-extending the used portion.
// Users of AuxInt which interpret AuxInt as unsigned (e.g. shifts) must be careful.
var genericOps = []opData{
	// 2-input arithmetic
	// Types must be consistent with Go typing. Add, for example, must take two values
	// of the same type and produces that same type.
	{name: "Add8", argLength: 2, commutative: true}, // arg0 + arg1
	{name: "Add16", argLength: 2, commutative: true},
	{name: "Add32", argLength: 2, commutative: true},
	{name: "Add64", argLength: 2, commutative: true},
	{name: "AddPtr", argLength: 2}, // For address calculations.  arg0 is a pointer and arg1 is an int.
	{name: "Add32F", argLength: 2, commutative: true},
	{name: "Add64F", argLength: 2, commutative: true},

	{name: "Sub8", argLength: 2}, // arg0 - arg1
	{name: "Sub16", argLength: 2},
	{name: "Sub32", argLength: 2},
	{name: "Sub64", argLength: 2},
	{name: "SubPtr", argLength: 2},
	{name: "Sub32F", argLength: 2},
	{name: "Sub64F", argLength: 2},

	{name: "Mul8", argLength: 2, commutative: true}, // arg0 * arg1
	{name: "Mul16", argLength: 2, commutative: true},
	{name: "Mul32", argLength: 2, commutative: true},
	{name: "Mul64", argLength: 2, commutative: true},
	{name: "Mul32F", argLength: 2, commutative: true},
	{name: "Mul64F", argLength: 2, commutative: true},

	{name: "Div32F", argLength: 2}, // arg0 / arg1
	{name: "Div64F", argLength: 2},

	{name: "Hmul32", argLength: 2, commutative: true},
	{name: "Hmul32u", argLength: 2, commutative: true},
	{name: "Hmul64", argLength: 2, commutative: true},
	{name: "Hmul64u", argLength: 2, commutative: true},

	{name: "Mul32uhilo", argLength: 2, typ: "(UInt32,UInt32)", commutative: true}, // arg0 * arg1, returns (hi, lo)
	{name: "Mul64uhilo", argLength: 2, typ: "(UInt64,UInt64)", commutative: true}, // arg0 * arg1, returns (hi, lo)

	// Weird special instructions for use in the strength reduction of divides.
	// These ops compute unsigned (arg0 + arg1) / 2, correct to all
	// 32/64 bits, even when the intermediate result of the add has 33/65 bits.
	// These ops can assume arg0 >= arg1.
	// Note: these ops aren't commutative!
	{name: "Avg32u", argLength: 2, typ: "UInt32"}, // 32-bit platforms only
	{name: "Avg64u", argLength: 2, typ: "UInt64"}, // 64-bit platforms only

	{name: "Div8", argLength: 2},  // arg0 / arg1, signed
	{name: "Div8u", argLength: 2}, // arg0 / arg1, unsigned
	{name: "Div16", argLength: 2},
	{name: "Div16u", argLength: 2},
	{name: "Div32", argLength: 2},
	{name: "Div32u", argLength: 2},
	{name: "Div64", argLength: 2},
	{name: "Div64u", argLength: 2},
	{name: "Div128u", argLength: 3}, // arg0:arg1 / arg2 (128-bit divided by 64-bit), returns (q, r)

	{name: "Mod8", argLength: 2},  // arg0 % arg1, signed
	{name: "Mod8u", argLength: 2}, // arg0 % arg1, unsigned
	{name: "Mod16", argLength: 2},
	{name: "Mod16u", argLength: 2},
	{name: "Mod32", argLength: 2},
	{name: "Mod32u", argLength: 2},
	{name: "Mod64", argLength: 2},
	{name: "Mod64u", argLength: 2},

	{name: "And8", argLength: 2, commutative: true}, // arg0 & arg1
	{name: "And16", argLength: 2, commutative: true},
	{name: "And32", argLength: 2, commutative: true},
	{name: "And64", argLength: 2, commutative: true},

	{name: "Or8", argLength: 2, commutative: true}, // arg0 | arg1
	{name: "Or16", argLength: 2, commutative: true},
	{name: "Or32", argLength: 2, commutative: true},
	{name: "Or64", argLength: 2, commutative: true},

	{name: "Xor8", argLength: 2, commutative: true}, // arg0 ^ arg1
	{name: "Xor16", argLength: 2, commutative: true},
	{name: "Xor32", argLength: 2, commutative: true},
	{name: "Xor64", argLength: 2, commutative: true},

	// For shifts, AxB means the shifted value has A bits and the shift amount has B bits.
	// Shift amounts are considered unsigned.
	{name: "Lsh8x8", argLength: 2}, // arg0 << arg1
	{name: "Lsh8x16", argLength: 2},
	{name: "Lsh8x32", argLength: 2},
	{name: "Lsh8x64", argLength: 2},
	{name: "Lsh16x8", argLength: 2},
	{name: "Lsh16x16", argLength: 2},
	{name: "Lsh16x32", argLength: 2},
	{name: "Lsh16x64", argLength: 2},
	{name: "Lsh32x8", argLength: 2},
	{name: "Lsh32x16", argLength: 2},
	{name: "Lsh32x32", argLength: 2},
	{name: "Lsh32x64", argLength: 2},
	{name: "Lsh64x8", argLength: 2},
	{name: "Lsh64x16", argLength: 2},
	{name: "Lsh64x32", argLength: 2},
	{name: "Lsh64x64", argLength: 2},

	{name: "Rsh8x8", argLength: 2}, // arg0 >> arg1, signed
	{name: "Rsh8x16", argLength: 2},
	{name: "Rsh8x32", argLength: 2},
	{name: "Rsh8x64", argLength: 2},
	{name: "Rsh16x8", argLength: 2},
	{name: "Rsh16x16", argLength: 2},
	{name: "Rsh16x32", argLength: 2},
	{name: "Rsh16x64", argLength: 2},
	{name: "Rsh32x8", argLength: 2},
	{name: "Rsh32x16", argLength: 2},
	{name: "Rsh32x32", argLength: 2},
	{name: "Rsh32x64", argLength: 2},
	{name: "Rsh64x8", argLength: 2},
	{name: "Rsh64x16", argLength: 2},
	{name: "Rsh64x32", argLength: 2},
	{name: "Rsh64x64", argLength: 2},

	{name: "Rsh8Ux8", argLength: 2}, // arg0 >> arg1, unsigned
	{name: "Rsh8Ux16", argLength: 2},
	{name: "Rsh8Ux32", argLength: 2},
	{name: "Rsh8Ux64", argLength: 2},
	{name: "Rsh16Ux8", argLength: 2},
	{name: "Rsh16Ux16", argLength: 2},
	{name: "Rsh16Ux32", argLength: 2},
	{name: "Rsh16Ux64", argLength: 2},
	{name: "Rsh32Ux8", argLength: 2},
	{name: "Rsh32Ux16", argLength: 2},
	{name: "Rsh32Ux32", argLength: 2},
	{name: "Rsh32Ux64", argLength: 2},
	{name: "Rsh64Ux8", argLength: 2},
	{name: "Rsh64Ux16", argLength: 2},
	{name: "Rsh64Ux32", argLength: 2},
	{name: "Rsh64Ux64", argLength: 2},

	// 2-input comparisons
	{name: "Eq8", argLength: 2, commutative: true, typ: "Bool"}, // arg0 == arg1
	{name: "Eq16", argLength: 2, commutative: true, typ: "Bool"},
	{name: "Eq32", argLength: 2, commutative: true, typ: "Bool"},
	{name: "Eq64", argLength: 2, commutative: true, typ: "Bool"},
	{name: "EqPtr", argLength: 2, commutative: true, typ: "Bool"},
	{name: "EqInter", argLength: 2, typ: "Bool"}, // arg0 or arg1 is nil; other cases handled by frontend
	{name: "EqSlice", argLength: 2, typ: "Bool"}, // arg0 or arg1 is nil; other cases handled by frontend
	{name: "Eq32F", argLength: 2, commutative: true, typ: "Bool"},
	{name: "Eq64F", argLength: 2, commutative: true, typ: "Bool"},

	{name: "Neq8", argLength: 2, commutative: true, typ: "Bool"}, // arg0 != arg1
	{name: "Neq16", argLength: 2, commutative: true, typ: "Bool"},
	{name: "Neq32", argLength: 2, commutative: true, typ: "Bool"},
	{name: "Neq64", argLength: 2, commutative: true, typ: "Bool"},
	{name: "NeqPtr", argLength: 2, commutative: true, typ: "Bool"},
	{name: "NeqInter", argLength: 2, typ: "Bool"}, // arg0 or arg1 is nil; other cases handled by frontend
	{name: "NeqSlice", argLength: 2, typ: "Bool"}, // arg0 or arg1 is nil; other cases handled by frontend
	{name: "Neq32F", argLength: 2, commutative: true, typ: "Bool"},
	{name: "Neq64F", argLength: 2, commutative: true, typ: "Bool"},

	{name: "Less8", argLength: 2, typ: "Bool"},  // arg0 < arg1, signed
	{name: "Less8U", argLength: 2, typ: "Bool"}, // arg0 < arg1, unsigned
	{name: "Less16", argLength: 2, typ: "Bool"},
	{name: "Less16U", argLength: 2, typ: "Bool"},
	{name: "Less32", argLength: 2, typ: "Bool"},
	{name: "Less32U", argLength: 2, typ: "Bool"},
	{name: "Less64", argLength: 2, typ: "Bool"},
	{name: "Less64U", argLength: 2, typ: "Bool"},
	{name: "Less32F", argLength: 2, typ: "Bool"},
	{name: "Less64F", argLength: 2, typ: "Bool"},

	{name: "Leq8", argLength: 2, typ: "Bool"},  // arg0 <= arg1, signed
	{name: "Leq8U", argLength: 2, typ: "Bool"}, // arg0 <= arg1, unsigned
	{name: "Leq16", argLength: 2, typ: "Bool"},
	{name: "Leq16U", argLength: 2, typ: "Bool"},
	{name: "Leq32", argLength: 2, typ: "Bool"},
	{name: "Leq32U", argLength: 2, typ: "Bool"},
	{name: "Leq64", argLength: 2, typ: "Bool"},
	{name: "Leq64U", argLength: 2, typ: "Bool"},
	{name: "Leq32F", argLength: 2, typ: "Bool"},
	{name: "Leq64F", argLength: 2, typ: "Bool"},

	{name: "Greater8", argLength: 2, typ: "Bool"},  // arg0 > arg1, signed
	{name: "Greater8U", argLength: 2, typ: "Bool"}, // arg0 > arg1, unsigned
	{name: "Greater16", argLength: 2, typ: "Bool"},
	{name: "Greater16U", argLength: 2, typ: "Bool"},
	{name: "Greater32", argLength: 2, typ: "Bool"},
	{name: "Greater32U", argLength: 2, typ: "Bool"},
	{name: "Greater64", argLength: 2, typ: "Bool"},
	{name: "Greater64U", argLength: 2, typ: "Bool"},
	{name: "Greater32F", argLength: 2, typ: "Bool"},
	{name: "Greater64F", argLength: 2, typ: "Bool"},

	{name: "Geq8", argLength: 2, typ: "Bool"},  // arg0 <= arg1, signed
	{name: "Geq8U", argLength: 2, typ: "Bool"}, // arg0 <= arg1, unsigned
	{name: "Geq16", argLength: 2, typ: "Bool"},
	{name: "Geq16U", argLength: 2, typ: "Bool"},
	{name: "Geq32", argLength: 2, typ: "Bool"},
	{name: "Geq32U", argLength: 2, typ: "Bool"},
	{name: "Geq64", argLength: 2, typ: "Bool"},
	{name: "Geq64U", argLength: 2, typ: "Bool"},
	{name: "Geq32F", argLength: 2, typ: "Bool"},
	{name: "Geq64F", argLength: 2, typ: "Bool"},

	// boolean ops
	{name: "AndB", argLength: 2, commutative: true, typ: "Bool"}, // arg0 && arg1 (not shortcircuited)
	{name: "OrB", argLength: 2, commutative: true, typ: "Bool"},  // arg0 || arg1 (not shortcircuited)
	{name: "EqB", argLength: 2, commutative: true, typ: "Bool"},  // arg0 == arg1
	{name: "NeqB", argLength: 2, commutative: true, typ: "Bool"}, // arg0 != arg1
	{name: "Not", argLength: 1, typ: "Bool"},                     // !arg0, boolean

	// 1-input ops
	{name: "Neg8", argLength: 1}, // -arg0
	{name: "Neg16", argLength: 1},
	{name: "Neg32", argLength: 1},
	{name: "Neg64", argLength: 1},
	{name: "Neg32F", argLength: 1},
	{name: "Neg64F", argLength: 1},

	{name: "Com8", argLength: 1}, // ^arg0
	{name: "Com16", argLength: 1},
	{name: "Com32", argLength: 1},
	{name: "Com64", argLength: 1},

	{name: "Ctz32", argLength: 1},    // Count trailing (low order) zeroes (returns 0-32)
	{name: "Ctz64", argLength: 1},    // Count trailing zeroes (returns 0-64)
	{name: "BitLen32", argLength: 1}, // Number of bits in arg[0] (returns 0-32)
	{name: "BitLen64", argLength: 1}, // Number of bits in arg[0] (returns 0-64)

	{name: "Bswap32", argLength: 1}, // Swap bytes
	{name: "Bswap64", argLength: 1}, // Swap bytes

	{name: "BitRev8", argLength: 1},  // Reverse the bits in arg[0]
	{name: "BitRev16", argLength: 1}, // Reverse the bits in arg[0]
	{name: "BitRev32", argLength: 1}, // Reverse the bits in arg[0]
	{name: "BitRev64", argLength: 1}, // Reverse the bits in arg[0]

	{name: "PopCount8", argLength: 1},  // Count bits in arg[0]
	{name: "PopCount16", argLength: 1}, // Count bits in arg[0]
	{name: "PopCount32", argLength: 1}, // Count bits in arg[0]
	{name: "PopCount64", argLength: 1}, // Count bits in arg[0]

	{name: "Sqrt", argLength: 1}, // sqrt(arg0), float64 only

	// Data movement, max argument length for Phi is indefinite so just pick
	// a really large number
	{name: "Phi", argLength: -1}, // select an argument based on which predecessor block we came from
	{name: "Copy", argLength: 1}, // output = arg0
	// Convert converts between pointers and integers.
	// We have a special op for this so as to not confuse GC
	// (particularly stack maps).  It takes a memory arg so it
	// gets correctly ordered with respect to GC safepoints.
	// arg0=ptr/int arg1=mem, output=int/ptr
	{name: "Convert", argLength: 2},

	// constants. Constant values are stored in the aux or
	// auxint fields.
	{name: "ConstBool", aux: "Bool"},     // auxint is 0 for false and 1 for true
	{name: "ConstString", aux: "String"}, // value is aux.(string)
	{name: "ConstNil", typ: "BytePtr"},   // nil pointer
	{name: "Const8", aux: "Int8"},        // auxint is sign-extended 8 bits
	{name: "Const16", aux: "Int16"},      // auxint is sign-extended 16 bits
	{name: "Const32", aux: "Int32"},      // auxint is sign-extended 32 bits
	// Note: ConstX are sign-extended even when the type of the value is unsigned.
	// For instance, uint8(0xaa) is stored as auxint=0xffffffffffffffaa.
	{name: "Const64", aux: "Int64"},    // value is auxint
	{name: "Const32F", aux: "Float32"}, // value is math.Float64frombits(uint64(auxint)) and is exactly prepresentable as float 32
	{name: "Const64F", aux: "Float64"}, // value is math.Float64frombits(uint64(auxint))
	{name: "ConstInterface"},           // nil interface
	{name: "ConstSlice"},               // nil slice

	// Constant-like things
	{name: "InitMem"},                               // memory input to the function.
	{name: "Arg", aux: "SymOff", symEffect: "None"}, // argument to the function.  aux=GCNode of arg, off = offset in that arg.

	// The address of a variable.  arg0 is the base pointer (SB or SP, depending
	// on whether it is a global or stack variable).  The Aux field identifies the
	// variable. It will be either an *ExternSymbol (with arg0=SB), *ArgSymbol (arg0=SP),
	// or *AutoSymbol (arg0=SP).
	{name: "Addr", argLength: 1, aux: "Sym", symEffect: "Addr"}, // Address of a variable.  Arg0=SP or SB.  Aux identifies the variable.

	{name: "SP"},                 // stack pointer
	{name: "SB", typ: "Uintptr"}, // static base pointer (a.k.a. globals pointer)
	{name: "Invalid"},            // unused value

	// Memory operations
	{name: "Load", argLength: 2},                             // Load from arg0.  arg1=memory
	{name: "Store", argLength: 3, typ: "Mem", aux: "Typ"},    // Store arg1 to arg0.  arg2=memory, aux=type.  Returns memory.
	{name: "Move", argLength: 3, typ: "Mem", aux: "TypSize"}, // arg0=destptr, arg1=srcptr, arg2=mem, auxint=size, aux=type.  Returns memory.
	{name: "Zero", argLength: 2, typ: "Mem", aux: "TypSize"}, // arg0=destptr, arg1=mem, auxint=size, aux=type. Returns memory.

	// Memory operations with write barriers.
	// Expand to runtime calls. Write barrier will be removed if write on stack.
	{name: "StoreWB", argLength: 3, typ: "Mem", aux: "Typ"},    // Store arg1 to arg0. arg2=memory, aux=type.  Returns memory.
	{name: "MoveWB", argLength: 3, typ: "Mem", aux: "TypSize"}, // arg0=destptr, arg1=srcptr, arg2=mem, auxint=size, aux=type.  Returns memory.
	{name: "ZeroWB", argLength: 2, typ: "Mem", aux: "TypSize"}, // arg0=destptr, arg1=mem, auxint=size, aux=type. Returns memory.

	// Function calls. Arguments to the call have already been written to the stack.
	// Return values appear on the stack. The method receiver, if any, is treated
	// as a phantom first argument.
	{name: "ClosureCall", argLength: 3, aux: "Int64", call: true},                    // arg0=code pointer, arg1=context ptr, arg2=memory.  auxint=arg size.  Returns memory.
	{name: "StaticCall", argLength: 1, aux: "SymOff", call: true, symEffect: "None"}, // call function aux.(*obj.LSym), arg0=memory.  auxint=arg size.  Returns memory.
	{name: "InterCall", argLength: 2, aux: "Int64", call: true},                      // interface call.  arg0=code pointer, arg1=memory, auxint=arg size.  Returns memory.

	// Conversions: signed extensions, zero (unsigned) extensions, truncations
	{name: "SignExt8to16", argLength: 1, typ: "Int16"},
	{name: "SignExt8to32", argLength: 1, typ: "Int32"},
	{name: "SignExt8to64", argLength: 1, typ: "Int64"},
	{name: "SignExt16to32", argLength: 1, typ: "Int32"},
	{name: "SignExt16to64", argLength: 1, typ: "Int64"},
	{name: "SignExt32to64", argLength: 1, typ: "Int64"},
	{name: "ZeroExt8to16", argLength: 1, typ: "UInt16"},
	{name: "ZeroExt8to32", argLength: 1, typ: "UInt32"},
	{name: "ZeroExt8to64", argLength: 1, typ: "UInt64"},
	{name: "ZeroExt16to32", argLength: 1, typ: "UInt32"},
	{name: "ZeroExt16to64", argLength: 1, typ: "UInt64"},
	{name: "ZeroExt32to64", argLength: 1, typ: "UInt64"},
	{name: "Trunc16to8", argLength: 1},
	{name: "Trunc32to8", argLength: 1},
	{name: "Trunc32to16", argLength: 1},
	{name: "Trunc64to8", argLength: 1},
	{name: "Trunc64to16", argLength: 1},
	{name: "Trunc64to32", argLength: 1},

	{name: "Cvt32to32F", argLength: 1},
	{name: "Cvt32to64F", argLength: 1},
	{name: "Cvt64to32F", argLength: 1},
	{name: "Cvt64to64F", argLength: 1},
	{name: "Cvt32Fto32", argLength: 1},
	{name: "Cvt32Fto64", argLength: 1},
	{name: "Cvt64Fto32", argLength: 1},
	{name: "Cvt64Fto64", argLength: 1},
	{name: "Cvt32Fto64F", argLength: 1},
	{name: "Cvt64Fto32F", argLength: 1},

	// Force rounding to precision of type.
	{name: "Round32F", argLength: 1},
	{name: "Round64F", argLength: 1},

	// Automatically inserted safety checks
	{name: "IsNonNil", argLength: 1, typ: "Bool"},        // arg0 != nil
	{name: "IsInBounds", argLength: 2, typ: "Bool"},      // 0 <= arg0 < arg1. arg1 is guaranteed >= 0.
	{name: "IsSliceInBounds", argLength: 2, typ: "Bool"}, // 0 <= arg0 <= arg1. arg1 is guaranteed >= 0.
	{name: "NilCheck", argLength: 2, typ: "Void"},        // arg0=ptr, arg1=mem. Panics if arg0 is nil. Returns void.

	// Pseudo-ops
	{name: "GetG", argLength: 1}, // runtime.getg() (read g pointer). arg0=mem
	{name: "GetClosurePtr"},      // get closure pointer from dedicated register

	// Indexing operations
	{name: "PtrIndex", argLength: 2},             // arg0=ptr, arg1=index. Computes ptr+sizeof(*v.type)*index, where index is extended to ptrwidth type
	{name: "OffPtr", argLength: 1, aux: "Int64"}, // arg0 + auxint (arg0 and result are pointers)

	// Slices
	{name: "SliceMake", argLength: 3},                // arg0=ptr, arg1=len, arg2=cap
	{name: "SlicePtr", argLength: 1, typ: "BytePtr"}, // ptr(arg0)
	{name: "SliceLen", argLength: 1},                 // len(arg0)
	{name: "SliceCap", argLength: 1},                 // cap(arg0)

	// Complex (part/whole)
	{name: "ComplexMake", argLength: 2}, // arg0=real, arg1=imag
	{name: "ComplexReal", argLength: 1}, // real(arg0)
	{name: "ComplexImag", argLength: 1}, // imag(arg0)

	// Strings
	{name: "StringMake", argLength: 2},                // arg0=ptr, arg1=len
	{name: "StringPtr", argLength: 1, typ: "BytePtr"}, // ptr(arg0)
	{name: "StringLen", argLength: 1, typ: "Int"},     // len(arg0)

	// Interfaces
	{name: "IMake", argLength: 2},                // arg0=itab, arg1=data
	{name: "ITab", argLength: 1, typ: "BytePtr"}, // arg0=interface, returns itable field
	{name: "IData", argLength: 1},                // arg0=interface, returns data field

	// Structs
	{name: "StructMake0"},                              // Returns struct with 0 fields.
	{name: "StructMake1", argLength: 1},                // arg0=field0.  Returns struct.
	{name: "StructMake2", argLength: 2},                // arg0,arg1=field0,field1.  Returns struct.
	{name: "StructMake3", argLength: 3},                // arg0..2=field0..2.  Returns struct.
	{name: "StructMake4", argLength: 4},                // arg0..3=field0..3.  Returns struct.
	{name: "StructSelect", argLength: 1, aux: "Int64"}, // arg0=struct, auxint=field index.  Returns the auxint'th field.

	// Arrays
	{name: "ArrayMake0"},                              // Returns array with 0 elements
	{name: "ArrayMake1", argLength: 1},                // Returns array with 1 element
	{name: "ArraySelect", argLength: 1, aux: "Int64"}, // arg0=array, auxint=index. Returns a[i].

	// Spill&restore ops for the register allocator. These are
	// semantically identical to OpCopy; they do not take/return
	// stores like regular memory ops do. We can get away without memory
	// args because we know there is no aliasing of spill slots on the stack.
	{name: "StoreReg", argLength: 1},
	{name: "LoadReg", argLength: 1},

	// Used during ssa construction. Like Copy, but the arg has not been specified yet.
	{name: "FwdRef", aux: "Sym", symEffect: "None"},

	// Unknown value. Used for Values whose values don't matter because they are dead code.
	{name: "Unknown"},

	{name: "VarDef", argLength: 1, aux: "Sym", typ: "Mem", symEffect: "None"}, // aux is a *gc.Node of a variable that is about to be initialized.  arg0=mem, returns mem
	{name: "VarKill", argLength: 1, aux: "Sym", symEffect: "None"},            // aux is a *gc.Node of a variable that is known to be dead.  arg0=mem, returns mem
	{name: "VarLive", argLength: 1, aux: "Sym", symEffect: "None"},            // aux is a *gc.Node of a variable that must be kept live.  arg0=mem, returns mem
	{name: "KeepAlive", argLength: 2, typ: "Mem"},                             // arg[0] is a value that must be kept alive until this mark.  arg[1]=mem, returns mem

	// Ops for breaking 64-bit operations on 32-bit architectures
	{name: "Int64Make", argLength: 2, typ: "UInt64"}, // arg0=hi, arg1=lo
	{name: "Int64Hi", argLength: 1, typ: "UInt32"},   // high 32-bit of arg0
	{name: "Int64Lo", argLength: 1, typ: "UInt32"},   // low 32-bit of arg0

	{name: "Add32carry", argLength: 2, commutative: true, typ: "(UInt32,Flags)"}, // arg0 + arg1, returns (value, carry)
	{name: "Add32withcarry", argLength: 3, commutative: true},                    // arg0 + arg1 + arg2, arg2=carry (0 or 1)

	{name: "Sub32carry", argLength: 2, typ: "(UInt32,Flags)"}, // arg0 - arg1, returns (value, carry)
	{name: "Sub32withcarry", argLength: 3},                    // arg0 - arg1 - arg2, arg2=carry (0 or 1)

	{name: "Signmask", argLength: 1, typ: "Int32"},  // 0 if arg0 >= 0, -1 if arg0 < 0
	{name: "Zeromask", argLength: 1, typ: "UInt32"}, // 0 if arg0 == 0, 0xffffffff if arg0 != 0
	{name: "Slicemask", argLength: 1},               // 0 if arg0 == 0, -1 if arg0 > 0, undef if arg0<0. Type is native int size.

	{name: "Cvt32Uto32F", argLength: 1}, // uint32 -> float32, only used on 32-bit arch
	{name: "Cvt32Uto64F", argLength: 1}, // uint32 -> float64, only used on 32-bit arch
	{name: "Cvt32Fto32U", argLength: 1}, // float32 -> uint32, only used on 32-bit arch
	{name: "Cvt64Fto32U", argLength: 1}, // float64 -> uint32, only used on 32-bit arch
	{name: "Cvt64Uto32F", argLength: 1}, // uint64 -> float32, only used on archs that has the instruction
	{name: "Cvt64Uto64F", argLength: 1}, // uint64 -> float64, only used on archs that has the instruction
	{name: "Cvt32Fto64U", argLength: 1}, // float32 -> uint64, only used on archs that has the instruction
	{name: "Cvt64Fto64U", argLength: 1}, // float64 -> uint64, only used on archs that has the instruction

	// pseudo-ops for breaking Tuple
	{name: "Select0", argLength: 1}, // the first component of a tuple
	{name: "Select1", argLength: 1}, // the second component of a tuple

	// Atomic operations used for semantically inlining runtime/internal/atomic.
	// Atomic loads return a new memory so that the loads are properly ordered
	// with respect to other loads and stores.
	// TODO: use for sync/atomic at some point.
	{name: "AtomicLoad32", argLength: 2, typ: "(UInt32,Mem)"},                               // Load from arg0.  arg1=memory.  Returns loaded value and new memory.
	{name: "AtomicLoad64", argLength: 2, typ: "(UInt64,Mem)"},                               // Load from arg0.  arg1=memory.  Returns loaded value and new memory.
	{name: "AtomicLoadPtr", argLength: 2, typ: "(BytePtr,Mem)"},                             // Load from arg0.  arg1=memory.  Returns loaded value and new memory.
	{name: "AtomicStore32", argLength: 3, typ: "Mem", hasSideEffects: true},                 // Store arg1 to *arg0.  arg2=memory.  Returns memory.
	{name: "AtomicStore64", argLength: 3, typ: "Mem", hasSideEffects: true},                 // Store arg1 to *arg0.  arg2=memory.  Returns memory.
	{name: "AtomicStorePtrNoWB", argLength: 3, typ: "Mem", hasSideEffects: true},            // Store arg1 to *arg0.  arg2=memory.  Returns memory.
	{name: "AtomicExchange32", argLength: 3, typ: "(UInt32,Mem)", hasSideEffects: true},     // Store arg1 to *arg0.  arg2=memory.  Returns old contents of *arg0 and new memory.
	{name: "AtomicExchange64", argLength: 3, typ: "(UInt64,Mem)", hasSideEffects: true},     // Store arg1 to *arg0.  arg2=memory.  Returns old contents of *arg0 and new memory.
	{name: "AtomicAdd32", argLength: 3, typ: "(UInt32,Mem)", hasSideEffects: true},          // Do *arg0 += arg1.  arg2=memory.  Returns sum and new memory.
	{name: "AtomicAdd64", argLength: 3, typ: "(UInt64,Mem)", hasSideEffects: true},          // Do *arg0 += arg1.  arg2=memory.  Returns sum and new memory.
	{name: "AtomicCompareAndSwap32", argLength: 4, typ: "(Bool,Mem)", hasSideEffects: true}, // if *arg0==arg1, then set *arg0=arg2.  Returns true iff store happens and new memory.
	{name: "AtomicCompareAndSwap64", argLength: 4, typ: "(Bool,Mem)", hasSideEffects: true}, // if *arg0==arg1, then set *arg0=arg2.  Returns true iff store happens and new memory.
	{name: "AtomicAnd8", argLength: 3, typ: "Mem", hasSideEffects: true},                    // *arg0 &= arg1.  arg2=memory.  Returns memory.
	{name: "AtomicOr8", argLength: 3, typ: "Mem", hasSideEffects: true},                     // *arg0 |= arg1.  arg2=memory.  Returns memory.

	// Clobber experiment op
	{name: "Clobber", argLength: 0, typ: "Void", aux: "SymOff", symEffect: "None"}, // write an invalid pointer value to the given pointer slot of a stack variable
}

//     kind           control    successors       implicit exit
//   ----------------------------------------------------------
//     Exit        return mem                []             yes
//      Ret        return mem                []             yes
//   RetJmp        return mem                []             yes
//    Plain               nil            [next]
//       If   a boolean Value      [then, else]
//     Call               mem            [next]             yes  (control opcode should be OpCall or OpStaticCall)
//    Check              void            [next]             yes  (control opcode should be Op{Lowered}NilCheck)
//    First               nil    [always,never]

var genericBlocks = []blockData{
	{name: "Plain"},  // a single successor
	{name: "If"},     // 2 successors, if control goto Succs[0] else goto Succs[1]
	{name: "Defer"},  // 2 successors, Succs[0]=defer queued, Succs[1]=defer recovered. control is call op (of memory type)
	{name: "Ret"},    // no successors, control value is memory result
	{name: "RetJmp"}, // no successors, jumps to b.Aux.(*gc.Sym)
	{name: "Exit"},   // no successors, control value generates a panic

	// transient block state used for dead code removal
	{name: "First"}, // 2 successors, always takes the first one (second is dead)
}

func init() {
	archs = append(archs, arch{
		name:    "generic",
		ops:     genericOps,
		blocks:  genericBlocks,
		generic: true,
	})
}
