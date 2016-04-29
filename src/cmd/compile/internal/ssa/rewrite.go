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
			if b.Control != nil && b.Control.Op == OpCopy {
				for b.Control.Op == OpCopy {
					b.SetControl(b.Control.Args[0])
				}
			}
			curb = b
			if rb(b) {
				change = true
			}
			curb = nil
			for _, v := range b.Values {
				change = phielimValue(v) || change

				// Eliminate copy inputs.
				// If any copy input becomes unused, mark it
				// as invalid and discard its argument. Repeat
				// recursively on the discarded argument.
				// This phase helps remove phantom "dead copy" uses
				// of a value so that a x.Uses==1 rule condition
				// fires reliably.
				for i, a := range v.Args {
					if a.Op != OpCopy {
						continue
					}
					v.SetArg(i, copySource(a))
					change = true
					for a.Uses == 0 {
						b := a.Args[0]
						a.reset(OpInvalid)
						a = b
					}
				}

				// apply rewrite function
				curv = v
				if rv(v, config) {
					change = true
				}
				curv = nil
			}
		}
		if !change {
			break
		}
	}
	// remove clobbered values
	for _, b := range f.Blocks {
		j := 0
		for i, v := range b.Values {
			if v.Op == OpInvalid {
				f.freeValue(v)
				continue
			}
			if i != j {
				b.Values[j] = v
			}
			j++
		}
		if j != len(b.Values) {
			tail := b.Values[j:]
			for j := range tail {
				tail[j] = nil
			}
			b.Values = b.Values[:j]
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
	return t.IsPtrShaped()
}

func isSigned(t Type) bool {
	return t.IsSigned()
}

func typeSize(t Type) int64 {
	return t.Size()
}

// mergeSym merges two symbolic offsets. There is no real merging of
// offsets, we just pick the non-nil one.
func mergeSym(x, y interface{}) interface{} {
	if x == nil {
		return y
	}
	if y == nil {
		return x
	}
	panic(fmt.Sprintf("mergeSym with two non-nil syms %s %s", x, y))
}
func canMergeSym(x, y interface{}) bool {
	return x == nil || y == nil
}

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

// i2f is used in rules for converting from an AuxInt to a float.
func i2f(i int64) float64 {
	return math.Float64frombits(uint64(i))
}

// i2f32 is used in rules for converting from an AuxInt to a float32.
func i2f32(i int64) float32 {
	return float32(math.Float64frombits(uint64(i)))
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
	if p1.Op != p2.Op {
		return false
	}
	switch p1.Op {
	case OpOffPtr:
		return p1.AuxInt == p2.AuxInt && isSamePtr(p1.Args[0], p2.Args[0])
	case OpAddr:
		// OpAddr's 0th arg is either OpSP or OpSB, which means that it is uniquely identified by its Op.
		// Checking for value equality only works after [z]cse has run.
		return p1.Aux == p2.Aux && p1.Args[0].Op == p2.Args[0].Op
	case OpAddPtr:
		return p1.Args[1] == p2.Args[1] && isSamePtr(p1.Args[0], p2.Args[0])
	}
	return false
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

// mergePoint finds a block among a's blocks which dominates b and is itself
// dominated by all of a's blocks. Returns nil if it can't find one.
// Might return nil even if one does exist.
func mergePoint(b *Block, a ...*Value) *Block {
	// Walk backward from b looking for one of the a's blocks.

	// Max distance
	d := 100

	for d > 0 {
		for _, x := range a {
			if b == x.Block {
				goto found
			}
		}
		if len(b.Preds) > 1 {
			// Don't know which way to go back. Abort.
			return nil
		}
		b = b.Preds[0]
		d--
	}
	return nil // too far away
found:
	// At this point, r is the first value in a that we find by walking backwards.
	// if we return anything, r will be it.
	r := b

	// Keep going, counting the other a's that we find. They must all dominate r.
	na := 0
	for d > 0 {
		for _, x := range a {
			if b == x.Block {
				na++
			}
		}
		if na == len(a) {
			// Found all of a in a backwards walk. We can return r.
			return r
		}
		if len(b.Preds) > 1 {
			return nil
		}
		b = b.Preds[0]
		d--

	}
	return nil // too far away
}

// clobber invalidates v.  Returns true.
// clobber is used by rewrite rules to:
//   A) make sure v is really dead and never used again.
//   B) decrement use counts of v's args.
func clobber(v *Value) bool {
	v.reset(OpInvalid)
	// Note: leave v.Block intact.  The Block field is used after clobber.
	return true
}
