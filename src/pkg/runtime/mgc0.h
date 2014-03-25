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
//
// NOTE: There is a copy of these in ../reflect/type.go.
// They must be kept in sync.
enum {
	GC_END,         // End of object, loop or subroutine. Args: none
	GC_PTR,         // A typed pointer. Args: (off, objgc)
	GC_APTR,        // Pointer to an arbitrary object. Args: (off)
	GC_ARRAY_START, // Start an array with a fixed length. Args: (off, len, elemsize)
	GC_ARRAY_NEXT,  // The next element of an array. Args: none
	GC_CALL,        // Call a subroutine. Args: (off, objgcrel)
	GC_CHAN_PTR,    // Go channel. Args: (off, ChanType*)
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

enum {
	ScanStackByFrames = 1,
	IgnorePreciseGC = 0,

	// Four bits per word (see #defines below).
	wordsPerBitmapWord = sizeof(void*)*8/4,
	bitShift = sizeof(void*)*8/4,
};

// Bits in per-word bitmap.
// #defines because enum might not be able to hold the values.
//
// Each word in the bitmap describes wordsPerBitmapWord words
// of heap memory.  There are 4 bitmap bits dedicated to each heap word,
// so on a 64-bit system there is one bitmap word per 16 heap words.
// The bits in the word are packed together by type first, then by
// heap location, so each 64-bit bitmap word consists of, from top to bottom,
// the 16 bitMarked bits for the corresponding heap words,
// then the 16 bitScan/bitBlockBoundary bits, then the 16 bitAllocated bits.
// This layout makes it easier to iterate over the bits of a given type.
//
// The bitmap starts at mheap.arena_start and extends *backward* from
// there.  On a 64-bit system the off'th word in the arena is tracked by
// the off/16+1'th word before mheap.arena_start.  (On a 32-bit system,
// the only difference is that the divisor is 8.)
//
// To pull out the bits corresponding to a given pointer p, we use:
//
//	off = p - (uintptr*)mheap.arena_start;  // word offset
//	b = (uintptr*)mheap.arena_start - off/wordsPerBitmapWord - 1;
//	shift = off % wordsPerBitmapWord
//	bits = *b >> shift;
//	/* then test bits & bitAllocated, bits & bitMarked, etc. */
//
#define bitAllocated		((uintptr)1<<(bitShift*0))	/* block start; eligible for garbage collection */
#define bitScan			((uintptr)1<<(bitShift*1))	/* when bitAllocated is set */
#define bitMarked		((uintptr)1<<(bitShift*2))	/* when bitAllocated is set */
#define bitBlockBoundary	((uintptr)1<<(bitShift*1))	/* when bitAllocated is NOT set - mark for FlagNoGC objects */

#define bitMask (bitAllocated | bitScan | bitMarked)
