// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Garbage collector (GC)

enum {
	// Four bits per word (see #defines below).
	gcBits = 4,
	wordsPerBitmapByte = 8/gcBits,

	// GC type info programs.
	// The programs allow to store type info required for GC in a compact form.
	// Most importantly arrays take O(1) space instead of O(n).
	// The program grammar is:
	//
	// Program = {Block} "insEnd"
	// Block = Data | Array
	// Data = "insData" DataSize DataBlock
	// DataSize = int // size of the DataBlock in bit pairs, 1 byte
	// DataBlock = binary // dense GC mask (2 bits per word) of size ]DataSize/4[ bytes
	// Array = "insArray" ArrayLen Block "insArrayEnd"
	// ArrayLen = int // length of the array, 8 bytes (4 bytes for 32-bit arch)
	//
	// Each instruction (insData, insArray, etc) is 1 byte.
	// For example, for type struct { x []byte; y [20]struct{ z int; w *byte }; }
	// the program looks as:
	//
	// insData 3 (BitsMultiWord BitsSlice BitsScalar)
	//	insArray 20 insData 2 (BitsScalar BitsPointer) insArrayEnd insEnd
	//
	// Total size of the program is 17 bytes (13 bytes on 32-bits).
	// The corresponding GC mask would take 43 bytes (it would be repeated
	// because the type has odd number of words).
	insData = 1,
	insArray,
	insArrayEnd,
	insEnd,

	// Pointer map
	BitsPerPointer	= 2,
	BitsMask	= (1<<BitsPerPointer)-1,
	PointersPerByte	= 8/BitsPerPointer,

	// If you change these, also change scanblock.
	// scanblock does "if(bits == BitsScalar || bits == BitsDead)" as "if(bits <= BitsScalar)".
	BitsDead	= 0,
	BitsScalar	= 1,
	BitsPointer	= 2,
	BitsMultiWord	= 3,
	// BitsMultiWord will be set for the first word of a multi-word item.
	// When it is set, one of the following will be set for the second word.
	// NOT USED ANYMORE: BitsString	= 0,
	// NOT USED ANYMORE: BitsSlice	= 1,
	BitsIface	= 2,
	BitsEface	= 3,

	// 64 bytes cover objects of size 1024/512 on 64/32 bits, respectively.
	MaxGCMask	= 64,
};

// Bits in per-word bitmap.
// #defines because we shift the values beyond 32 bits.
//
// Each word in the bitmap describes wordsPerBitmapWord words
// of heap memory.  There are 4 bitmap bits dedicated to each heap word,
// so on a 64-bit system there is one bitmap word per 16 heap words.
//
// The bitmap starts at mheap.arena_start and extends *backward* from
// there.  On a 64-bit system the off'th word in the arena is tracked by
// the off/16+1'th word before mheap.arena_start.  (On a 32-bit system,
// the only difference is that the divisor is 8.)
enum {
	bitBoundary = 1, // boundary of an object
	bitMarked = 2, // marked object
	bitMask = bitBoundary | bitMarked,
	bitPtrMask = BitsMask<<2,
};
