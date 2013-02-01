// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Garbage collector (GC)

// GC instruction opcodes.
//
// The opcode of an instruction is followed by zero or more
// arguments to the instruction.
//
// Meaning of arguments:
//   off      Offset (in bytes) from the start of the current object
//   objgc    Pointer to GC info of an object
//   objgcrel Offset to GC info of an object
//   len      Length of an array
//   elemsize Size (in bytes) of an element
//   size     Size (in bytes)
enum {
	GC_END,         // End of object, loop or subroutine. Args: none
	GC_PTR,         // A typed pointer. Args: (off, objgc)
	GC_APTR,        // Pointer to an arbitrary object. Args: (off)
	GC_ARRAY_START, // Start an array with a fixed length. Args: (off, len, elemsize)
	GC_ARRAY_NEXT,  // The next element of an array. Args: none
	GC_CALL,        // Call a subroutine. Args: (off, objgcrel)
	GC_MAP_PTR,     // Go map. Args: (off, MapType*)
	GC_STRING,      // Go string. Args: (off)
	GC_EFACE,       // interface{}. Args: (off)
	GC_IFACE,       // interface{...}. Args: (off)
	GC_SLICE,       // Go slice. Args: (off, objgc)
	GC_REGION,      // A region/part of the current object. Args: (off, size, objgc)

	GC_NUM_INSTR,   // Number of instruction opcodes
};

enum {
	// Size of GC's fixed stack.
	//
	// The current GC implementation permits:
	//  - at most 1 stack allocation because of GC_CALL
	//  - at most GC_STACK_CAPACITY allocations because of GC_ARRAY_START
	GC_STACK_CAPACITY = 8,	
};
