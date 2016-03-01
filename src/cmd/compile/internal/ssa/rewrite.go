// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"fmt"
	"math"
)

func applyRewrite(f *Func, rb func(*Block) bool, rv func(*Value, *Config) bool) {
	// repeat rewrites until we find no more rewrites
	var curb *Block
	var curv *Value
	defer func() {
		if curb != nil {
			curb.Fatalf("panic during rewrite of block %s\n", curb.LongString())
		}
		if curv != nil {
			curv.Fatalf("panic during rewrite of value %s\n", curv.LongString())
			// TODO(khr): print source location also
		}
	}()
	config := f.Config
	for {
		change := false
		for _, b := range f.Blocks {
			if b.Kind == BlockDead {
				continue
			}
			if b.Control != nil && b.Control.Op == OpCopy {
				for b.Control.Op == OpCopy {
					b.Control = b.Control.Args[0]
				}
			}
			curb = b
			if rb(b) {
				change = true
			}
			curb = nil
			for _, v := range b.Values {
				copyelimValue(v)
				change = phielimValue(v) || change

				// apply rewrite function
				curv = v
				if rv(v, config) {
					change = true
				}
				curv = nil
			}
		}
		if !change {
			return
		}
	}
}

// Common functions called from rewriting rules

func is64BitFloat(t Type) bool {
	return t.Size() == 8 && t.IsFloat()
}

func is32BitFloat(t Type) bool {
	return t.Size() == 4 && t.IsFloat()
}

func is64BitInt(t Type) bool {
	return t.Size() == 8 && t.IsInteger()
}

func is32BitInt(t Type) bool {
	return t.Size() == 4 && t.IsInteger()
}

func is16BitInt(t Type) bool {
	return t.Size() == 2 && t.IsInteger()
}

func is8BitInt(t Type) bool {
	return t.Size() == 1 && t.IsInteger()
}

func isPtr(t Type) bool {
	return t.IsPtr()
}

func isSigned(t Type) bool {
	return t.IsSigned()
}

func typeSize(t Type) int64 {
	return t.Size()
}

// addOff adds two int64 offsets. Fails if wraparound happens.
func addOff(x, y int64) int64 {
	z := x + y
	// x and y have same sign and z has a different sign => overflow
	if x^y >= 0 && x^z < 0 {
		panic(fmt.Sprintf("offset overflow %d %d", x, y))
	}
	return z
}

// mergeSym merges two symbolic offsets.  There is no real merging of
// offsets, we just pick the non-nil one.
func mergeSym(x, y interface{}) interface{} {
	if x == nil {
		return y
	}
	if y == nil {
		return x
	}
	panic(fmt.Sprintf("mergeSym with two non-nil syms %s %s", x, y))
	return nil
}
func canMergeSym(x, y interface{}) bool {
	return x == nil || y == nil
}

func inBounds8(idx, len int64) bool       { return int8(idx) >= 0 && int8(idx) < int8(len) }
func inBounds16(idx, len int64) bool      { return int16(idx) >= 0 && int16(idx) < int16(len) }
func inBounds32(idx, len int64) bool      { return int32(idx) >= 0 && int32(idx) < int32(len) }
func inBounds64(idx, len int64) bool      { return idx >= 0 && idx < len }
func sliceInBounds32(idx, len int64) bool { return int32(idx) >= 0 && int32(idx) <= int32(len) }
func sliceInBounds64(idx, len int64) bool { return idx >= 0 && idx <= len }

// nlz returns the number of leading zeros.
func nlz(x int64) int64 {
	// log2(0) == 1, so nlz(0) == 64
	return 63 - log2(x)
}

// ntz returns the number of trailing zeros.
func ntz(x int64) int64 {
	return 64 - nlz(^x&(x-1))
}

// nlo returns the number of leading ones.
func nlo(x int64) int64 {
	return nlz(^x)
}

// nto returns the number of trailing ones.
func nto(x int64) int64 {
	return ntz(^x)
}

// log2 returns logarithm in base of uint64(n), with log2(0) = -1.
func log2(n int64) (l int64) {
	l = -1
	x := uint64(n)
	for ; x >= 0x8000; x >>= 16 {
		l += 16
	}
	if x >= 0x80 {
		x >>= 8
		l += 8
	}
	if x >= 0x8 {
		x >>= 4
		l += 4
	}
	if x >= 0x2 {
		x >>= 2
		l += 2
	}
	if x >= 0x1 {
		l++
	}
	return
}

// isPowerOfTwo reports whether n is a power of 2.
func isPowerOfTwo(n int64) bool {
	return n > 0 && n&(n-1) == 0
}

// is32Bit reports whether n can be represented as a signed 32 bit integer.
func is32Bit(n int64) bool {
	return n == int64(int32(n))
}

// b2i translates a boolean value to 0 or 1 for assigning to auxInt.
func b2i(b bool) int64 {
	if b {
		return 1
	}
	return 0
}

// f2i is used in the rules for storing a float in AuxInt.
func f2i(f float64) int64 {
	return int64(math.Float64bits(f))
}

// uaddOvf returns true if unsigned a+b would overflow.
func uaddOvf(a, b int64) bool {
	return uint64(a)+uint64(b) < uint64(a)
}

// isSamePtr reports whether p1 and p2 point to the same address.
func isSamePtr(p1, p2 *Value) bool {
	if p1 == p2 {
		return true
	}
	// Aux isn't used  in OffPtr, and AuxInt isn't currently used in
	// Addr, but this still works as the values will be null/0
	return (p1.Op == OpOffPtr || p1.Op == OpAddr) && p1.Op == p2.Op &&
		p1.Aux == p2.Aux && p1.AuxInt == p2.AuxInt &&
		p1.Args[0] == p2.Args[0]
}

// DUFFZERO consists of repeated blocks of 4 MOVUPSs + ADD,
// See runtime/mkduff.go.
const (
	dzBlocks    = 16 // number of MOV/ADD blocks
	dzBlockLen  = 4  // number of clears per block
	dzBlockSize = 19 // size of instructions in a single block
	dzMovSize   = 4  // size of single MOV instruction w/ offset
	dzAddSize   = 4  // size of single ADD instruction
	dzClearStep = 16 // number of bytes cleared by each MOV instruction

	dzTailLen  = 4 // number of final STOSQ instructions
	dzTailSize = 2 // size of single STOSQ instruction

	dzClearLen = dzClearStep * dzBlockLen // bytes cleared by one block
	dzSize     = dzBlocks * dzBlockSize
)

func duffStart(size int64) int64 {
	x, _ := duff(size)
	return x
}
func duffAdj(size int64) int64 {
	_, x := duff(size)
	return x
}

// duff returns the offset (from duffzero, in bytes) and pointer adjust (in bytes)
// required to use the duffzero mechanism for a block of the given size.
func duff(size int64) (int64, int64) {
	if size < 32 || size > 1024 || size%dzClearStep != 0 {
		panic("bad duffzero size")
	}
	// TODO: arch-dependent
	steps := size / dzClearStep
	blocks := steps / dzBlockLen
	steps %= dzBlockLen
	off := dzBlockSize * (dzBlocks - blocks)
	var adj int64
	if steps != 0 {
		off -= dzAddSize
		off -= dzMovSize * steps
		adj -= dzClearStep * (dzBlockLen - steps)
	}
	return off, adj
}
