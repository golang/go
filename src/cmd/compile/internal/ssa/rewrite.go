// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import (
	"cmd/compile/internal/base"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/reflectdata"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/s390x"
	"cmd/internal/objabi"
	"cmd/internal/src"
	"encoding/binary"
	"fmt"
	"internal/buildcfg"
	"io"
	"math"
	"math/bits"
	"os"
	"path/filepath"
	"strings"
)

type deadValueChoice bool

const (
	leaveDeadValues  deadValueChoice = false
	removeDeadValues                 = true
)

// deadcode indicates whether rewrite should try to remove any values that become dead.
func applyRewrite(f *Func, rb blockRewriter, rv valueRewriter, deadcode deadValueChoice) {
	// repeat rewrites until we find no more rewrites
	pendingLines := f.cachedLineStarts // Holds statement boundaries that need to be moved to a new value/block
	pendingLines.clear()
	debug := f.pass.debug
	if debug > 1 {
		fmt.Printf("%s: rewriting for %s\n", f.pass.name, f.Name)
	}
	// if the number of rewrite iterations reaches itersLimit we will
	// at that point turn on cycle detection. Instead of a fixed limit,
	// size the limit according to func size to allow for cases such
	// as the one in issue #66773.
	itersLimit := f.NumBlocks()
	if itersLimit < 20 {
		itersLimit = 20
	}
	var iters int
	var states map[string]bool
	for {
		change := false
		deadChange := false
		for _, b := range f.Blocks {
			var b0 *Block
			if debug > 1 {
				b0 = new(Block)
				*b0 = *b
				b0.Succs = append([]Edge{}, b.Succs...) // make a new copy, not aliasing
			}
			for i, c := range b.ControlValues() {
				for c.Op == OpCopy {
					c = c.Args[0]
					b.ReplaceControl(i, c)
				}
			}
			if rb(b) {
				change = true
				if debug > 1 {
					fmt.Printf("rewriting %s  ->  %s\n", b0.LongString(), b.LongString())
				}
			}
			for j, v := range b.Values {
				var v0 *Value
				if debug > 1 {
					v0 = new(Value)
					*v0 = *v
					v0.Args = append([]*Value{}, v.Args...) // make a new copy, not aliasing
				}
				if v.Uses == 0 && v.removeable() {
					if v.Op != OpInvalid && deadcode == removeDeadValues {
						// Reset any values that are now unused, so that we decrement
						// the use count of all of its arguments.
						// Not quite a deadcode pass, because it does not handle cycles.
						// But it should help Uses==1 rules to fire.
						v.reset(OpInvalid)
						deadChange = true
					}
					// No point rewriting values which aren't used.
					continue
				}

				vchange := phielimValue(v)
				if vchange && debug > 1 {
					fmt.Printf("rewriting %s  ->  %s\n", v0.LongString(), v.LongString())
				}

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
					aa := copySource(a)
					v.SetArg(i, aa)
					// If a, a copy, has a line boundary indicator, attempt to find a new value
					// to hold it.  The first candidate is the value that will replace a (aa),
					// if it shares the same block and line and is eligible.
					// The second option is v, which has a as an input.  Because aa is earlier in
					// the data flow, it is the better choice.
					if a.Pos.IsStmt() == src.PosIsStmt {
						if aa.Block == a.Block && aa.Pos.Line() == a.Pos.Line() && aa.Pos.IsStmt() != src.PosNotStmt {
							aa.Pos = aa.Pos.WithIsStmt()
						} else if v.Block == a.Block && v.Pos.Line() == a.Pos.Line() && v.Pos.IsStmt() != src.PosNotStmt {
							v.Pos = v.Pos.WithIsStmt()
						} else {
							// Record the lost line and look for a new home after all rewrites are complete.
							// TODO: it's possible (in FOR loops, in particular) for statement boundaries for the same
							// line to appear in more than one block, but only one block is stored, so if both end
							// up here, then one will be lost.
							pendingLines.set(a.Pos, int32(a.Block.ID))
						}
						a.Pos = a.Pos.WithNotStmt()
					}
					vchange = true
					for a.Uses == 0 {
						b := a.Args[0]
						a.reset(OpInvalid)
						a = b
					}
				}
				if vchange && debug > 1 {
					fmt.Printf("rewriting %s  ->  %s\n", v0.LongString(), v.LongString())
				}

				// apply rewrite function
				if rv(v) {
					vchange = true
					// If value changed to a poor choice for a statement boundary, move the boundary
					if v.Pos.IsStmt() == src.PosIsStmt {
						if k := nextGoodStatementIndex(v, j, b); k != j {
							v.Pos = v.Pos.WithNotStmt()
							b.Values[k].Pos = b.Values[k].Pos.WithIsStmt()
						}
					}
				}

				change = change || vchange
				if vchange && debug > 1 {
					fmt.Printf("rewriting %s  ->  %s\n", v0.LongString(), v.LongString())
				}
			}
		}
		if !change && !deadChange {
			break
		}
		iters++
		if (iters > itersLimit || debug >= 2) && change {
			// We've done a suspiciously large number of rewrites (or we're in debug mode).
			// As of Sep 2021, 90% of rewrites complete in 4 iterations or fewer
			// and the maximum value encountered during make.bash is 12.
			// Start checking for cycles. (This is too expensive to do routinely.)
			// Note: we avoid this path for deadChange-only iterations, to fix #51639.
			if states == nil {
				states = make(map[string]bool)
			}
			h := f.rewriteHash()
			if _, ok := states[h]; ok {
				// We've found a cycle.
				// To diagnose it, set debug to 2 and start again,
				// so that we'll print all rules applied until we complete another cycle.
				// If debug is already >= 2, we've already done that, so it's time to crash.
				if debug < 2 {
					debug = 2
					states = make(map[string]bool)
				} else {
					f.Fatalf("rewrite cycle detected")
				}
			}
			states[h] = true
		}
	}
	// remove clobbered values
	for _, b := range f.Blocks {
		j := 0
		for i, v := range b.Values {
			vl := v.Pos
			if v.Op == OpInvalid {
				if v.Pos.IsStmt() == src.PosIsStmt {
					pendingLines.set(vl, int32(b.ID))
				}
				f.freeValue(v)
				continue
			}
			if v.Pos.IsStmt() != src.PosNotStmt && !notStmtBoundary(v.Op) && pendingLines.get(vl) == int32(b.ID) {
				pendingLines.remove(vl)
				v.Pos = v.Pos.WithIsStmt()
			}
			if i != j {
				b.Values[j] = v
			}
			j++
		}
		if pendingLines.get(b.Pos) == int32(b.ID) {
			b.Pos = b.Pos.WithIsStmt()
			pendingLines.remove(b.Pos)
		}
		b.truncateValues(j)
	}
}

// Common functions called from rewriting rules

func is64BitFloat(t *types.Type) bool {
	return t.Size() == 8 && t.IsFloat()
}

func is32BitFloat(t *types.Type) bool {
	return t.Size() == 4 && t.IsFloat()
}

func is64BitInt(t *types.Type) bool {
	return t.Size() == 8 && t.IsInteger()
}

func is32BitInt(t *types.Type) bool {
	return t.Size() == 4 && t.IsInteger()
}

func is16BitInt(t *types.Type) bool {
	return t.Size() == 2 && t.IsInteger()
}

func is8BitInt(t *types.Type) bool {
	return t.Size() == 1 && t.IsInteger()
}

func isPtr(t *types.Type) bool {
	return t.IsPtrShaped()
}

// mergeSym merges two symbolic offsets. There is no real merging of
// offsets, we just pick the non-nil one.
func mergeSym(x, y Sym) Sym {
	if x == nil {
		return y
	}
	if y == nil {
		return x
	}
	panic(fmt.Sprintf("mergeSym with two non-nil syms %v %v", x, y))
}

func canMergeSym(x, y Sym) bool {
	return x == nil || y == nil
}

// canMergeLoadClobber reports whether the load can be merged into target without
// invalidating the schedule.
// It also checks that the other non-load argument x is something we
// are ok with clobbering.
func canMergeLoadClobber(target, load, x *Value) bool {
	// The register containing x is going to get clobbered.
	// Don't merge if we still need the value of x.
	// We don't have liveness information here, but we can
	// approximate x dying with:
	//  1) target is x's only use.
	//  2) target is not in a deeper loop than x.
	if x.Uses != 1 {
		return false
	}
	loopnest := x.Block.Func.loopnest()
	loopnest.calculateDepths()
	if loopnest.depth(target.Block.ID) > loopnest.depth(x.Block.ID) {
		return false
	}
	return canMergeLoad(target, load)
}

// canMergeLoad reports whether the load can be merged into target without
// invalidating the schedule.
func canMergeLoad(target, load *Value) bool {
	if target.Block.ID != load.Block.ID {
		// If the load is in a different block do not merge it.
		return false
	}

	// We can't merge the load into the target if the load
	// has more than one use.
	if load.Uses != 1 {
		return false
	}

	mem := load.MemoryArg()

	// We need the load's memory arg to still be alive at target. That
	// can't be the case if one of target's args depends on a memory
	// state that is a successor of load's memory arg.
	//
	// For example, it would be invalid to merge load into target in
	// the following situation because newmem has killed oldmem
	// before target is reached:
	//     load = read ... oldmem
	//   newmem = write ... oldmem
	//     arg0 = read ... newmem
	//   target = add arg0 load
	//
	// If the argument comes from a different block then we can exclude
	// it immediately because it must dominate load (which is in the
	// same block as target).
	var args []*Value
	for _, a := range target.Args {
		if a != load && a.Block.ID == target.Block.ID {
			args = append(args, a)
		}
	}

	// memPreds contains memory states known to be predecessors of load's
	// memory state. It is lazily initialized.
	var memPreds map[*Value]bool
	for i := 0; len(args) > 0; i++ {
		const limit = 100
		if i >= limit {
			// Give up if we have done a lot of iterations.
			return false
		}
		v := args[len(args)-1]
		args = args[:len(args)-1]
		if target.Block.ID != v.Block.ID {
			// Since target and load are in the same block
			// we can stop searching when we leave the block.
			continue
		}
		if v.Op == OpPhi {
			// A Phi implies we have reached the top of the block.
			// The memory phi, if it exists, is always
			// the first logical store in the block.
			continue
		}
		if v.Type.IsTuple() && v.Type.FieldType(1).IsMemory() {
			// We could handle this situation however it is likely
			// to be very rare.
			return false
		}
		if v.Op.SymEffect()&SymAddr != 0 {
			// This case prevents an operation that calculates the
			// address of a local variable from being forced to schedule
			// before its corresponding VarDef.
			// See issue 28445.
			//   v1 = LOAD ...
			//   v2 = VARDEF
			//   v3 = LEAQ
			//   v4 = CMPQ v1 v3
			// We don't want to combine the CMPQ with the load, because
			// that would force the CMPQ to schedule before the VARDEF, which
			// in turn requires the LEAQ to schedule before the VARDEF.
			return false
		}
		if v.Type.IsMemory() {
			if memPreds == nil {
				// Initialise a map containing memory states
				// known to be predecessors of load's memory
				// state.
				memPreds = make(map[*Value]bool)
				m := mem
				const limit = 50
				for i := 0; i < limit; i++ {
					if m.Op == OpPhi {
						// The memory phi, if it exists, is always
						// the first logical store in the block.
						break
					}
					if m.Block.ID != target.Block.ID {
						break
					}
					if !m.Type.IsMemory() {
						break
					}
					memPreds[m] = true
					if len(m.Args) == 0 {
						break
					}
					m = m.MemoryArg()
				}
			}

			// We can merge if v is a predecessor of mem.
			//
			// For example, we can merge load into target in the
			// following scenario:
			//      x = read ... v
			//    mem = write ... v
			//   load = read ... mem
			// target = add x load
			if memPreds[v] {
				continue
			}
			return false
		}
		if len(v.Args) > 0 && v.Args[len(v.Args)-1] == mem {
			// If v takes mem as an input then we know mem
			// is valid at this point.
			continue
		}
		for _, a := range v.Args {
			if target.Block.ID == a.Block.ID {
				args = append(args, a)
			}
		}
	}

	return true
}

// isSameCall reports whether sym is the same as the given named symbol.
func isSameCall(sym interface{}, name string) bool {
	fn := sym.(*AuxCall).Fn
	return fn != nil && fn.String() == name
}

// canLoadUnaligned reports if the architecture supports unaligned load operations.
func canLoadUnaligned(c *Config) bool {
	return c.ctxt.Arch.Alignment == 1
}

// nlzX returns the number of leading zeros.
func nlz64(x int64) int { return bits.LeadingZeros64(uint64(x)) }
func nlz32(x int32) int { return bits.LeadingZeros32(uint32(x)) }
func nlz16(x int16) int { return bits.LeadingZeros16(uint16(x)) }
func nlz8(x int8) int   { return bits.LeadingZeros8(uint8(x)) }

// ntzX returns the number of trailing zeros.
func ntz64(x int64) int { return bits.TrailingZeros64(uint64(x)) }
func ntz32(x int32) int { return bits.TrailingZeros32(uint32(x)) }
func ntz16(x int16) int { return bits.TrailingZeros16(uint16(x)) }
func ntz8(x int8) int   { return bits.TrailingZeros8(uint8(x)) }

func oneBit(x int64) bool   { return x&(x-1) == 0 && x != 0 }
func oneBit8(x int8) bool   { return x&(x-1) == 0 && x != 0 }
func oneBit16(x int16) bool { return x&(x-1) == 0 && x != 0 }
func oneBit32(x int32) bool { return x&(x-1) == 0 && x != 0 }
func oneBit64(x int64) bool { return x&(x-1) == 0 && x != 0 }

// nto returns the number of trailing ones.
func nto(x int64) int64 {
	return int64(ntz64(^x))
}

// logX returns logarithm of n base 2.
// n must be a positive power of 2 (isPowerOfTwoX returns true).
func log8(n int8) int64 {
	return int64(bits.Len8(uint8(n))) - 1
}
func log16(n int16) int64 {
	return int64(bits.Len16(uint16(n))) - 1
}
func log32(n int32) int64 {
	return int64(bits.Len32(uint32(n))) - 1
}
func log64(n int64) int64 {
	return int64(bits.Len64(uint64(n))) - 1
}

// log2uint32 returns logarithm in base 2 of uint32(n), with log2(0) = -1.
// Rounds down.
func log2uint32(n int64) int64 {
	return int64(bits.Len32(uint32(n))) - 1
}

// isPowerOfTwoX functions report whether n is a power of 2.
func isPowerOfTwo8(n int8) bool {
	return n > 0 && n&(n-1) == 0
}
func isPowerOfTwo16(n int16) bool {
	return n > 0 && n&(n-1) == 0
}
func isPowerOfTwo32(n int32) bool {
	return n > 0 && n&(n-1) == 0
}
func isPowerOfTwo64(n int64) bool {
	return n > 0 && n&(n-1) == 0
}

// isUint64PowerOfTwo reports whether uint64(n) is a power of 2.
func isUint64PowerOfTwo(in int64) bool {
	n := uint64(in)
	return n > 0 && n&(n-1) == 0
}

// isUint32PowerOfTwo reports whether uint32(n) is a power of 2.
func isUint32PowerOfTwo(in int64) bool {
	n := uint64(uint32(in))
	return n > 0 && n&(n-1) == 0
}

// is32Bit reports whether n can be represented as a signed 32 bit integer.
func is32Bit(n int64) bool {
	return n == int64(int32(n))
}

// is16Bit reports whether n can be represented as a signed 16 bit integer.
func is16Bit(n int64) bool {
	return n == int64(int16(n))
}

// is8Bit reports whether n can be represented as a signed 8 bit integer.
func is8Bit(n int64) bool {
	return n == int64(int8(n))
}

// isU8Bit reports whether n can be represented as an unsigned 8 bit integer.
func isU8Bit(n int64) bool {
	return n == int64(uint8(n))
}

// isU12Bit reports whether n can be represented as an unsigned 12 bit integer.
func isU12Bit(n int64) bool {
	return 0 <= n && n < (1<<12)
}

// isU16Bit reports whether n can be represented as an unsigned 16 bit integer.
func isU16Bit(n int64) bool {
	return n == int64(uint16(n))
}

// isU32Bit reports whether n can be represented as an unsigned 32 bit integer.
func isU32Bit(n int64) bool {
	return n == int64(uint32(n))
}

// is20Bit reports whether n can be represented as a signed 20 bit integer.
func is20Bit(n int64) bool {
	return -(1<<19) <= n && n < (1<<19)
}

// b2i translates a boolean value to 0 or 1 for assigning to auxInt.
func b2i(b bool) int64 {
	if b {
		return 1
	}
	return 0
}

// b2i32 translates a boolean value to 0 or 1.
func b2i32(b bool) int32 {
	if b {
		return 1
	}
	return 0
}

// shiftIsBounded reports whether (left/right) shift Value v is known to be bounded.
// A shift is bounded if it is shifting by less than the width of the shifted value.
func shiftIsBounded(v *Value) bool {
	return v.AuxInt != 0
}

// canonLessThan returns whether x is "ordered" less than y, for purposes of normalizing
// generated code as much as possible.
func canonLessThan(x, y *Value) bool {
	if x.Op != y.Op {
		return x.Op < y.Op
	}
	if !x.Pos.SameFileAndLine(y.Pos) {
		return x.Pos.Before(y.Pos)
	}
	return x.ID < y.ID
}

// truncate64Fto32F converts a float64 value to a float32 preserving the bit pattern
// of the mantissa. It will panic if the truncation results in lost information.
func truncate64Fto32F(f float64) float32 {
	if !isExactFloat32(f) {
		panic("truncate64Fto32F: truncation is not exact")
	}
	if !math.IsNaN(f) {
		return float32(f)
	}
	// NaN bit patterns aren't necessarily preserved across conversion
	// instructions so we need to do the conversion manually.
	b := math.Float64bits(f)
	m := b & ((1 << 52) - 1) // mantissa (a.k.a. significand)
	//          | sign                  | exponent   | mantissa       |
	r := uint32(((b >> 32) & (1 << 31)) | 0x7f800000 | (m >> (52 - 23)))
	return math.Float32frombits(r)
}

// extend32Fto64F converts a float32 value to a float64 value preserving the bit
// pattern of the mantissa.
func extend32Fto64F(f float32) float64 {
	if !math.IsNaN(float64(f)) {
		return float64(f)
	}
	// NaN bit patterns aren't necessarily preserved across conversion
	// instructions so we need to do the conversion manually.
	b := uint64(math.Float32bits(f))
	//   | sign                  | exponent      | mantissa                    |
	r := ((b << 32) & (1 << 63)) | (0x7ff << 52) | ((b & 0x7fffff) << (52 - 23))
	return math.Float64frombits(r)
}

// DivisionNeedsFixUp reports whether the division needs fix-up code.
func DivisionNeedsFixUp(v *Value) bool {
	return v.AuxInt == 0
}

// auxFrom64F encodes a float64 value so it can be stored in an AuxInt.
func auxFrom64F(f float64) int64 {
	if f != f {
		panic("can't encode a NaN in AuxInt field")
	}
	return int64(math.Float64bits(f))
}

// auxFrom32F encodes a float32 value so it can be stored in an AuxInt.
func auxFrom32F(f float32) int64 {
	if f != f {
		panic("can't encode a NaN in AuxInt field")
	}
	return int64(math.Float64bits(extend32Fto64F(f)))
}

// auxTo32F decodes a float32 from the AuxInt value provided.
func auxTo32F(i int64) float32 {
	return truncate64Fto32F(math.Float64frombits(uint64(i)))
}

// auxTo64F decodes a float64 from the AuxInt value provided.
func auxTo64F(i int64) float64 {
	return math.Float64frombits(uint64(i))
}

func auxIntToBool(i int64) bool {
	if i == 0 {
		return false
	}
	return true
}
func auxIntToInt8(i int64) int8 {
	return int8(i)
}
func auxIntToInt16(i int64) int16 {
	return int16(i)
}
func auxIntToInt32(i int64) int32 {
	return int32(i)
}
func auxIntToInt64(i int64) int64 {
	return i
}
func auxIntToUint8(i int64) uint8 {
	return uint8(i)
}
func auxIntToFloat32(i int64) float32 {
	return float32(math.Float64frombits(uint64(i)))
}
func auxIntToFloat64(i int64) float64 {
	return math.Float64frombits(uint64(i))
}
func auxIntToValAndOff(i int64) ValAndOff {
	return ValAndOff(i)
}
func auxIntToArm64BitField(i int64) arm64BitField {
	return arm64BitField(i)
}
func auxIntToInt128(x int64) int128 {
	if x != 0 {
		panic("nonzero int128 not allowed")
	}
	return 0
}
func auxIntToFlagConstant(x int64) flagConstant {
	return flagConstant(x)
}

func auxIntToOp(cc int64) Op {
	return Op(cc)
}

func boolToAuxInt(b bool) int64 {
	if b {
		return 1
	}
	return 0
}
func int8ToAuxInt(i int8) int64 {
	return int64(i)
}
func int16ToAuxInt(i int16) int64 {
	return int64(i)
}
func int32ToAuxInt(i int32) int64 {
	return int64(i)
}
func int64ToAuxInt(i int64) int64 {
	return int64(i)
}
func uint8ToAuxInt(i uint8) int64 {
	return int64(int8(i))
}
func float32ToAuxInt(f float32) int64 {
	return int64(math.Float64bits(float64(f)))
}
func float64ToAuxInt(f float64) int64 {
	return int64(math.Float64bits(f))
}
func valAndOffToAuxInt(v ValAndOff) int64 {
	return int64(v)
}
func arm64BitFieldToAuxInt(v arm64BitField) int64 {
	return int64(v)
}
func int128ToAuxInt(x int128) int64 {
	if x != 0 {
		panic("nonzero int128 not allowed")
	}
	return 0
}
func flagConstantToAuxInt(x flagConstant) int64 {
	return int64(x)
}

func opToAuxInt(o Op) int64 {
	return int64(o)
}

// Aux is an interface to hold miscellaneous data in Blocks and Values.
type Aux interface {
	CanBeAnSSAAux()
}

// for now only used to mark moves that need to avoid clobbering flags
type auxMark bool

func (auxMark) CanBeAnSSAAux() {}

var AuxMark auxMark

// stringAux wraps string values for use in Aux.
type stringAux string

func (stringAux) CanBeAnSSAAux() {}

func auxToString(i Aux) string {
	return string(i.(stringAux))
}
func auxToSym(i Aux) Sym {
	// TODO: kind of a hack - allows nil interface through
	s, _ := i.(Sym)
	return s
}
func auxToType(i Aux) *types.Type {
	return i.(*types.Type)
}
func auxToCall(i Aux) *AuxCall {
	return i.(*AuxCall)
}
func auxToS390xCCMask(i Aux) s390x.CCMask {
	return i.(s390x.CCMask)
}
func auxToS390xRotateParams(i Aux) s390x.RotateParams {
	return i.(s390x.RotateParams)
}

func StringToAux(s string) Aux {
	return stringAux(s)
}
func symToAux(s Sym) Aux {
	return s
}
func callToAux(s *AuxCall) Aux {
	return s
}
func typeToAux(t *types.Type) Aux {
	return t
}
func s390xCCMaskToAux(c s390x.CCMask) Aux {
	return c
}
func s390xRotateParamsToAux(r s390x.RotateParams) Aux {
	return r
}

// uaddOvf reports whether unsigned a+b would overflow.
func uaddOvf(a, b int64) bool {
	return uint64(a)+uint64(b) < uint64(a)
}

// loadLSymOffset simulates reading a word at an offset into a
// read-only symbol's runtime memory. If it would read a pointer to
// another symbol, that symbol is returned. Otherwise, it returns nil.
func loadLSymOffset(lsym *obj.LSym, offset int64) *obj.LSym {
	if lsym.Type != objabi.SRODATA {
		return nil
	}

	for _, r := range lsym.R {
		if int64(r.Off) == offset && r.Type&^objabi.R_WEAK == objabi.R_ADDR && r.Add == 0 {
			return r.Sym
		}
	}

	return nil
}

func devirtLECall(v *Value, sym *obj.LSym) *Value {
	v.Op = OpStaticLECall
	auxcall := v.Aux.(*AuxCall)
	auxcall.Fn = sym
	// Remove first arg
	v.Args[0].Uses--
	copy(v.Args[0:], v.Args[1:])
	v.Args[len(v.Args)-1] = nil // aid GC
	v.Args = v.Args[:len(v.Args)-1]
	if f := v.Block.Func; f.pass.debug > 0 {
		f.Warnl(v.Pos, "de-virtualizing call")
	}
	return v
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
	case OpAddr, OpLocalAddr:
		return p1.Aux == p2.Aux
	case OpAddPtr:
		return p1.Args[1] == p2.Args[1] && isSamePtr(p1.Args[0], p2.Args[0])
	}
	return false
}

func isStackPtr(v *Value) bool {
	for v.Op == OpOffPtr || v.Op == OpAddPtr {
		v = v.Args[0]
	}
	return v.Op == OpSP || v.Op == OpLocalAddr
}

// disjoint reports whether the memory region specified by [p1:p1+n1)
// does not overlap with [p2:p2+n2).
// A return value of false does not imply the regions overlap.
func disjoint(p1 *Value, n1 int64, p2 *Value, n2 int64) bool {
	if n1 == 0 || n2 == 0 {
		return true
	}
	if p1 == p2 {
		return false
	}
	baseAndOffset := func(ptr *Value) (base *Value, offset int64) {
		base, offset = ptr, 0
		for base.Op == OpOffPtr {
			offset += base.AuxInt
			base = base.Args[0]
		}
		if opcodeTable[base.Op].nilCheck {
			base = base.Args[0]
		}
		return base, offset
	}
	p1, off1 := baseAndOffset(p1)
	p2, off2 := baseAndOffset(p2)
	if isSamePtr(p1, p2) {
		return !overlap(off1, n1, off2, n2)
	}
	// p1 and p2 are not the same, so if they are both OpAddrs then
	// they point to different variables.
	// If one pointer is on the stack and the other is an argument
	// then they can't overlap.
	switch p1.Op {
	case OpAddr, OpLocalAddr:
		if p2.Op == OpAddr || p2.Op == OpLocalAddr || p2.Op == OpSP {
			return true
		}
		return (p2.Op == OpArg || p2.Op == OpArgIntReg) && p1.Args[0].Op == OpSP
	case OpArg, OpArgIntReg:
		if p2.Op == OpSP || p2.Op == OpLocalAddr {
			return true
		}
	case OpSP:
		return p2.Op == OpAddr || p2.Op == OpLocalAddr || p2.Op == OpArg || p2.Op == OpArgIntReg || p2.Op == OpSP
	}
	return false
}

// moveSize returns the number of bytes an aligned MOV instruction moves.
func moveSize(align int64, c *Config) int64 {
	switch {
	case align%8 == 0 && c.PtrSize == 8:
		return 8
	case align%4 == 0:
		return 4
	case align%2 == 0:
		return 2
	}
	return 1
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
		b = b.Preds[0].b
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
		b = b.Preds[0].b
		d--

	}
	return nil // too far away
}

// clobber invalidates values. Returns true.
// clobber is used by rewrite rules to:
//
//	A) make sure the values are really dead and never used again.
//	B) decrement use counts of the values' args.
func clobber(vv ...*Value) bool {
	for _, v := range vv {
		v.reset(OpInvalid)
		// Note: leave v.Block intact.  The Block field is used after clobber.
	}
	return true
}

// clobberIfDead resets v when use count is 1. Returns true.
// clobberIfDead is used by rewrite rules to decrement
// use counts of v's args when v is dead and never used.
func clobberIfDead(v *Value) bool {
	if v.Uses == 1 {
		v.reset(OpInvalid)
	}
	// Note: leave v.Block intact.  The Block field is used after clobberIfDead.
	return true
}

// noteRule is an easy way to track if a rule is matched when writing
// new ones.  Make the rule of interest also conditional on
//
//	noteRule("note to self: rule of interest matched")
//
// and that message will print when the rule matches.
func noteRule(s string) bool {
	fmt.Println(s)
	return true
}

// countRule increments Func.ruleMatches[key].
// If Func.ruleMatches is non-nil at the end
// of compilation, it will be printed to stdout.
// This is intended to make it easier to find which functions
// which contain lots of rules matches when developing new rules.
func countRule(v *Value, key string) bool {
	f := v.Block.Func
	if f.ruleMatches == nil {
		f.ruleMatches = make(map[string]int)
	}
	f.ruleMatches[key]++
	return true
}

// warnRule generates compiler debug output with string s when
// v is not in autogenerated code, cond is true and the rule has fired.
func warnRule(cond bool, v *Value, s string) bool {
	if pos := v.Pos; pos.Line() > 1 && cond {
		v.Block.Func.Warnl(pos, s)
	}
	return true
}

// for a pseudo-op like (LessThan x), extract x.
func flagArg(v *Value) *Value {
	if len(v.Args) != 1 || !v.Args[0].Type.IsFlags() {
		return nil
	}
	return v.Args[0]
}

// arm64Negate finds the complement to an ARM64 condition code,
// for example !Equal -> NotEqual or !LessThan -> GreaterEqual
//
// For floating point, it's more subtle because NaN is unordered. We do
// !LessThanF -> NotLessThanF, the latter takes care of NaNs.
func arm64Negate(op Op) Op {
	switch op {
	case OpARM64LessThan:
		return OpARM64GreaterEqual
	case OpARM64LessThanU:
		return OpARM64GreaterEqualU
	case OpARM64GreaterThan:
		return OpARM64LessEqual
	case OpARM64GreaterThanU:
		return OpARM64LessEqualU
	case OpARM64LessEqual:
		return OpARM64GreaterThan
	case OpARM64LessEqualU:
		return OpARM64GreaterThanU
	case OpARM64GreaterEqual:
		return OpARM64LessThan
	case OpARM64GreaterEqualU:
		return OpARM64LessThanU
	case OpARM64Equal:
		return OpARM64NotEqual
	case OpARM64NotEqual:
		return OpARM64Equal
	case OpARM64LessThanF:
		return OpARM64NotLessThanF
	case OpARM64NotLessThanF:
		return OpARM64LessThanF
	case OpARM64LessEqualF:
		return OpARM64NotLessEqualF
	case OpARM64NotLessEqualF:
		return OpARM64LessEqualF
	case OpARM64GreaterThanF:
		return OpARM64NotGreaterThanF
	case OpARM64NotGreaterThanF:
		return OpARM64GreaterThanF
	case OpARM64GreaterEqualF:
		return OpARM64NotGreaterEqualF
	case OpARM64NotGreaterEqualF:
		return OpARM64GreaterEqualF
	default:
		panic("unreachable")
	}
}

// arm64Invert evaluates (InvertFlags op), which
// is the same as altering the condition codes such
// that the same result would be produced if the arguments
// to the flag-generating instruction were reversed, e.g.
// (InvertFlags (CMP x y)) -> (CMP y x)
func arm64Invert(op Op) Op {
	switch op {
	case OpARM64LessThan:
		return OpARM64GreaterThan
	case OpARM64LessThanU:
		return OpARM64GreaterThanU
	case OpARM64GreaterThan:
		return OpARM64LessThan
	case OpARM64GreaterThanU:
		return OpARM64LessThanU
	case OpARM64LessEqual:
		return OpARM64GreaterEqual
	case OpARM64LessEqualU:
		return OpARM64GreaterEqualU
	case OpARM64GreaterEqual:
		return OpARM64LessEqual
	case OpARM64GreaterEqualU:
		return OpARM64LessEqualU
	case OpARM64Equal, OpARM64NotEqual:
		return op
	case OpARM64LessThanF:
		return OpARM64GreaterThanF
	case OpARM64GreaterThanF:
		return OpARM64LessThanF
	case OpARM64LessEqualF:
		return OpARM64GreaterEqualF
	case OpARM64GreaterEqualF:
		return OpARM64LessEqualF
	case OpARM64NotLessThanF:
		return OpARM64NotGreaterThanF
	case OpARM64NotGreaterThanF:
		return OpARM64NotLessThanF
	case OpARM64NotLessEqualF:
		return OpARM64NotGreaterEqualF
	case OpARM64NotGreaterEqualF:
		return OpARM64NotLessEqualF
	default:
		panic("unreachable")
	}
}

// evaluate an ARM64 op against a flags value
// that is potentially constant; return 1 for true,
// -1 for false, and 0 for not constant.
func ccARM64Eval(op Op, flags *Value) int {
	fop := flags.Op
	if fop == OpARM64InvertFlags {
		return -ccARM64Eval(op, flags.Args[0])
	}
	if fop != OpARM64FlagConstant {
		return 0
	}
	fc := flagConstant(flags.AuxInt)
	b2i := func(b bool) int {
		if b {
			return 1
		}
		return -1
	}
	switch op {
	case OpARM64Equal:
		return b2i(fc.eq())
	case OpARM64NotEqual:
		return b2i(fc.ne())
	case OpARM64LessThan:
		return b2i(fc.lt())
	case OpARM64LessThanU:
		return b2i(fc.ult())
	case OpARM64GreaterThan:
		return b2i(fc.gt())
	case OpARM64GreaterThanU:
		return b2i(fc.ugt())
	case OpARM64LessEqual:
		return b2i(fc.le())
	case OpARM64LessEqualU:
		return b2i(fc.ule())
	case OpARM64GreaterEqual:
		return b2i(fc.ge())
	case OpARM64GreaterEqualU:
		return b2i(fc.uge())
	}
	return 0
}

// logRule logs the use of the rule s. This will only be enabled if
// rewrite rules were generated with the -log option, see _gen/rulegen.go.
func logRule(s string) {
	if ruleFile == nil {
		// Open a log file to write log to. We open in append
		// mode because all.bash runs the compiler lots of times,
		// and we want the concatenation of all of those logs.
		// This means, of course, that users need to rm the old log
		// to get fresh data.
		// TODO: all.bash runs compilers in parallel. Need to synchronize logging somehow?
		w, err := os.OpenFile(filepath.Join(os.Getenv("GOROOT"), "src", "rulelog"),
			os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0666)
		if err != nil {
			panic(err)
		}
		ruleFile = w
	}
	_, err := fmt.Fprintln(ruleFile, s)
	if err != nil {
		panic(err)
	}
}

var ruleFile io.Writer

func min(x, y int64) int64 {
	if x < y {
		return x
	}
	return y
}
func max(x, y int64) int64 {
	if x > y {
		return x
	}
	return y
}

func isConstZero(v *Value) bool {
	switch v.Op {
	case OpConstNil:
		return true
	case OpConst64, OpConst32, OpConst16, OpConst8, OpConstBool, OpConst32F, OpConst64F:
		return v.AuxInt == 0
	}
	return false
}

// reciprocalExact64 reports whether 1/c is exactly representable.
func reciprocalExact64(c float64) bool {
	b := math.Float64bits(c)
	man := b & (1<<52 - 1)
	if man != 0 {
		return false // not a power of 2, denormal, or NaN
	}
	exp := b >> 52 & (1<<11 - 1)
	// exponent bias is 0x3ff.  So taking the reciprocal of a number
	// changes the exponent to 0x7fe-exp.
	switch exp {
	case 0:
		return false // ±0
	case 0x7ff:
		return false // ±inf
	case 0x7fe:
		return false // exponent is not representable
	default:
		return true
	}
}

// reciprocalExact32 reports whether 1/c is exactly representable.
func reciprocalExact32(c float32) bool {
	b := math.Float32bits(c)
	man := b & (1<<23 - 1)
	if man != 0 {
		return false // not a power of 2, denormal, or NaN
	}
	exp := b >> 23 & (1<<8 - 1)
	// exponent bias is 0x7f.  So taking the reciprocal of a number
	// changes the exponent to 0xfe-exp.
	switch exp {
	case 0:
		return false // ±0
	case 0xff:
		return false // ±inf
	case 0xfe:
		return false // exponent is not representable
	default:
		return true
	}
}

// check if an immediate can be directly encoded into an ARM's instruction.
func isARMImmRot(v uint32) bool {
	for i := 0; i < 16; i++ {
		if v&^0xff == 0 {
			return true
		}
		v = v<<2 | v>>30
	}

	return false
}

// overlap reports whether the ranges given by the given offset and
// size pairs overlap.
func overlap(offset1, size1, offset2, size2 int64) bool {
	if offset1 >= offset2 && offset2+size2 > offset1 {
		return true
	}
	if offset2 >= offset1 && offset1+size1 > offset2 {
		return true
	}
	return false
}

func areAdjacentOffsets(off1, off2, size int64) bool {
	return off1+size == off2 || off1 == off2+size
}

// check if value zeroes out upper 32-bit of 64-bit register.
// depth limits recursion depth. In AMD64.rules 3 is used as limit,
// because it catches same amount of cases as 4.
func zeroUpper32Bits(x *Value, depth int) bool {
	switch x.Op {
	case OpAMD64MOVLconst, OpAMD64MOVLload, OpAMD64MOVLQZX, OpAMD64MOVLloadidx1,
		OpAMD64MOVWload, OpAMD64MOVWloadidx1, OpAMD64MOVBload, OpAMD64MOVBloadidx1,
		OpAMD64MOVLloadidx4, OpAMD64ADDLload, OpAMD64SUBLload, OpAMD64ANDLload,
		OpAMD64ORLload, OpAMD64XORLload, OpAMD64CVTTSD2SL,
		OpAMD64ADDL, OpAMD64ADDLconst, OpAMD64SUBL, OpAMD64SUBLconst,
		OpAMD64ANDL, OpAMD64ANDLconst, OpAMD64ORL, OpAMD64ORLconst,
		OpAMD64XORL, OpAMD64XORLconst, OpAMD64NEGL, OpAMD64NOTL,
		OpAMD64SHRL, OpAMD64SHRLconst, OpAMD64SARL, OpAMD64SARLconst,
		OpAMD64SHLL, OpAMD64SHLLconst:
		return true
	case OpARM64REV16W, OpARM64REVW, OpARM64RBITW, OpARM64CLZW, OpARM64EXTRWconst,
		OpARM64MULW, OpARM64MNEGW, OpARM64UDIVW, OpARM64DIVW, OpARM64UMODW,
		OpARM64MADDW, OpARM64MSUBW, OpARM64RORW, OpARM64RORWconst:
		return true
	case OpArg: // note: but not ArgIntReg
		// amd64 always loads args from the stack unsigned.
		// most other architectures load them sign/zero extended based on the type.
		return x.Type.Size() == 4 && (x.Type.IsUnsigned() || x.Block.Func.Config.arch == "amd64")
	case OpPhi, OpSelect0, OpSelect1:
		// Phis can use each-other as an arguments, instead of tracking visited values,
		// just limit recursion depth.
		if depth <= 0 {
			return false
		}
		for i := range x.Args {
			if !zeroUpper32Bits(x.Args[i], depth-1) {
				return false
			}
		}
		return true

	}
	return false
}

// zeroUpper48Bits is similar to zeroUpper32Bits, but for upper 48 bits.
func zeroUpper48Bits(x *Value, depth int) bool {
	switch x.Op {
	case OpAMD64MOVWQZX, OpAMD64MOVWload, OpAMD64MOVWloadidx1, OpAMD64MOVWloadidx2:
		return true
	case OpArg: // note: but not ArgIntReg
		return x.Type.Size() == 2 && (x.Type.IsUnsigned() || x.Block.Func.Config.arch == "amd64")
	case OpPhi, OpSelect0, OpSelect1:
		// Phis can use each-other as an arguments, instead of tracking visited values,
		// just limit recursion depth.
		if depth <= 0 {
			return false
		}
		for i := range x.Args {
			if !zeroUpper48Bits(x.Args[i], depth-1) {
				return false
			}
		}
		return true

	}
	return false
}

// zeroUpper56Bits is similar to zeroUpper32Bits, but for upper 56 bits.
func zeroUpper56Bits(x *Value, depth int) bool {
	switch x.Op {
	case OpAMD64MOVBQZX, OpAMD64MOVBload, OpAMD64MOVBloadidx1:
		return true
	case OpArg: // note: but not ArgIntReg
		return x.Type.Size() == 1 && (x.Type.IsUnsigned() || x.Block.Func.Config.arch == "amd64")
	case OpPhi, OpSelect0, OpSelect1:
		// Phis can use each-other as an arguments, instead of tracking visited values,
		// just limit recursion depth.
		if depth <= 0 {
			return false
		}
		for i := range x.Args {
			if !zeroUpper56Bits(x.Args[i], depth-1) {
				return false
			}
		}
		return true

	}
	return false
}

func isInlinableMemclr(c *Config, sz int64) bool {
	if sz < 0 {
		return false
	}
	// TODO: expand this check to allow other architectures
	// see CL 454255 and issue 56997
	switch c.arch {
	case "amd64", "arm64":
		return true
	case "ppc64le", "ppc64":
		return sz < 512
	}
	return false
}

// isInlinableMemmove reports whether the given arch performs a Move of the given size
// faster than memmove. It will only return true if replacing the memmove with a Move is
// safe, either because Move will do all of its loads before any of its stores, or
// because the arguments are known to be disjoint.
// This is used as a check for replacing memmove with Move ops.
func isInlinableMemmove(dst, src *Value, sz int64, c *Config) bool {
	// It is always safe to convert memmove into Move when its arguments are disjoint.
	// Move ops may or may not be faster for large sizes depending on how the platform
	// lowers them, so we only perform this optimization on platforms that we know to
	// have fast Move ops.
	switch c.arch {
	case "amd64":
		return sz <= 16 || (sz < 1024 && disjoint(dst, sz, src, sz))
	case "386", "arm64":
		return sz <= 8
	case "s390x", "ppc64", "ppc64le":
		return sz <= 8 || disjoint(dst, sz, src, sz)
	case "arm", "loong64", "mips", "mips64", "mipsle", "mips64le":
		return sz <= 4
	}
	return false
}
func IsInlinableMemmove(dst, src *Value, sz int64, c *Config) bool {
	return isInlinableMemmove(dst, src, sz, c)
}

// logLargeCopy logs the occurrence of a large copy.
// The best place to do this is in the rewrite rules where the size of the move is easy to find.
// "Large" is arbitrarily chosen to be 128 bytes; this may change.
func logLargeCopy(v *Value, s int64) bool {
	if s < 128 {
		return true
	}
	if logopt.Enabled() {
		logopt.LogOpt(v.Pos, "copy", "lower", v.Block.Func.Name, fmt.Sprintf("%d bytes", s))
	}
	return true
}
func LogLargeCopy(funcName string, pos src.XPos, s int64) {
	if s < 128 {
		return
	}
	if logopt.Enabled() {
		logopt.LogOpt(pos, "copy", "lower", funcName, fmt.Sprintf("%d bytes", s))
	}
}

// hasSmallRotate reports whether the architecture has rotate instructions
// for sizes < 32-bit.  This is used to decide whether to promote some rotations.
func hasSmallRotate(c *Config) bool {
	switch c.arch {
	case "amd64", "386":
		return true
	default:
		return false
	}
}

func supportsPPC64PCRel() bool {
	// PCRel is currently supported for >= power10, linux only
	// Internal and external linking supports this on ppc64le; internal linking on ppc64.
	return buildcfg.GOPPC64 >= 10 && buildcfg.GOOS == "linux"
}

func newPPC64ShiftAuxInt(sh, mb, me, sz int64) int32 {
	if sh < 0 || sh >= sz {
		panic("PPC64 shift arg sh out of range")
	}
	if mb < 0 || mb >= sz {
		panic("PPC64 shift arg mb out of range")
	}
	if me < 0 || me >= sz {
		panic("PPC64 shift arg me out of range")
	}
	return int32(sh<<16 | mb<<8 | me)
}

func GetPPC64Shiftsh(auxint int64) int64 {
	return int64(int8(auxint >> 16))
}

func GetPPC64Shiftmb(auxint int64) int64 {
	return int64(int8(auxint >> 8))
}

func GetPPC64Shiftme(auxint int64) int64 {
	return int64(int8(auxint))
}

// Test if this value can encoded as a mask for a rlwinm like
// operation.  Masks can also extend from the msb and wrap to
// the lsb too.  That is, the valid masks are 32 bit strings
// of the form: 0..01..10..0 or 1..10..01..1 or 1...1
func isPPC64WordRotateMask(v64 int64) bool {
	// Isolate rightmost 1 (if none 0) and add.
	v := uint32(v64)
	vp := (v & -v) + v
	// Likewise, for the wrapping case.
	vn := ^v
	vpn := (vn & -vn) + vn
	return (v&vp == 0 || vn&vpn == 0) && v != 0
}

// Compress mask and shift into single value of the form
// me | mb<<8 | rotate<<16 | nbits<<24 where me and mb can
// be used to regenerate the input mask.
func encodePPC64RotateMask(rotate, mask, nbits int64) int64 {
	var mb, me, mbn, men int

	// Determine boundaries and then decode them
	if mask == 0 || ^mask == 0 || rotate >= nbits {
		panic(fmt.Sprintf("invalid PPC64 rotate mask: %x %d %d", uint64(mask), rotate, nbits))
	} else if nbits == 32 {
		mb = bits.LeadingZeros32(uint32(mask))
		me = 32 - bits.TrailingZeros32(uint32(mask))
		mbn = bits.LeadingZeros32(^uint32(mask))
		men = 32 - bits.TrailingZeros32(^uint32(mask))
	} else {
		mb = bits.LeadingZeros64(uint64(mask))
		me = 64 - bits.TrailingZeros64(uint64(mask))
		mbn = bits.LeadingZeros64(^uint64(mask))
		men = 64 - bits.TrailingZeros64(^uint64(mask))
	}
	// Check for a wrapping mask (e.g bits at 0 and 63)
	if mb == 0 && me == int(nbits) {
		// swap the inverted values
		mb, me = men, mbn
	}

	return int64(me) | int64(mb<<8) | int64(rotate<<16) | int64(nbits<<24)
}

// Merge (RLDICL [encoded] (SRDconst [s] x)) into (RLDICL [new_encoded] x)
// SRDconst on PPC64 is an extended mnemonic of RLDICL. If the input to an
// RLDICL is an SRDconst, and the RLDICL does not rotate its value, the two
// operations can be combined. This functions assumes the two opcodes can
// be merged, and returns an encoded rotate+mask value of the combined RLDICL.
func mergePPC64RLDICLandSRDconst(encoded, s int64) int64 {
	mb := s
	r := 64 - s
	// A larger mb is a smaller mask.
	if (encoded>>8)&0xFF < mb {
		encoded = (encoded &^ 0xFF00) | mb<<8
	}
	// The rotate is expected to be 0.
	if (encoded & 0xFF0000) != 0 {
		panic("non-zero rotate")
	}
	return encoded | r<<16
}

// DecodePPC64RotateMask is the inverse operation of encodePPC64RotateMask.  The values returned as
// mb and me satisfy the POWER ISA definition of MASK(x,y) where MASK(mb,me) = mask.
func DecodePPC64RotateMask(sauxint int64) (rotate, mb, me int64, mask uint64) {
	auxint := uint64(sauxint)
	rotate = int64((auxint >> 16) & 0xFF)
	mb = int64((auxint >> 8) & 0xFF)
	me = int64((auxint >> 0) & 0xFF)
	nbits := int64((auxint >> 24) & 0xFF)
	mask = ((1 << uint(nbits-mb)) - 1) ^ ((1 << uint(nbits-me)) - 1)
	if mb > me {
		mask = ^mask
	}
	if nbits == 32 {
		mask = uint64(uint32(mask))
	}

	// Fixup ME to match ISA definition.  The second argument to MASK(..,me)
	// is inclusive.
	me = (me - 1) & (nbits - 1)
	return
}

// This verifies that the mask is a set of
// consecutive bits including the least
// significant bit.
func isPPC64ValidShiftMask(v int64) bool {
	if (v != 0) && ((v+1)&v) == 0 {
		return true
	}
	return false
}

func getPPC64ShiftMaskLength(v int64) int64 {
	return int64(bits.Len64(uint64(v)))
}

// Decompose a shift right into an equivalent rotate/mask,
// and return mask & m.
func mergePPC64RShiftMask(m, s, nbits int64) int64 {
	smask := uint64((1<<uint(nbits))-1) >> uint(s)
	return m & int64(smask)
}

// Combine (ANDconst [m] (SRWconst [s])) into (RLWINM [y]) or return 0
func mergePPC64AndSrwi(m, s int64) int64 {
	mask := mergePPC64RShiftMask(m, s, 32)
	if !isPPC64WordRotateMask(mask) {
		return 0
	}
	return encodePPC64RotateMask((32-s)&31, mask, 32)
}

// Test if a word shift right feeding into a CLRLSLDI can be merged into RLWINM.
// Return the encoded RLWINM constant, or 0 if they cannot be merged.
func mergePPC64ClrlsldiSrw(sld, srw int64) int64 {
	mask_1 := uint64(0xFFFFFFFF >> uint(srw))
	// for CLRLSLDI, it's more convenient to think of it as a mask left bits then rotate left.
	mask_2 := uint64(0xFFFFFFFFFFFFFFFF) >> uint(GetPPC64Shiftmb(int64(sld)))

	// Rewrite mask to apply after the final left shift.
	mask_3 := (mask_1 & mask_2) << uint(GetPPC64Shiftsh(sld))

	r_1 := 32 - srw
	r_2 := GetPPC64Shiftsh(sld)
	r_3 := (r_1 + r_2) & 31 // This can wrap.

	if uint64(uint32(mask_3)) != mask_3 || mask_3 == 0 {
		return 0
	}
	return encodePPC64RotateMask(int64(r_3), int64(mask_3), 32)
}

// Test if a doubleword shift right feeding into a CLRLSLDI can be merged into RLWINM.
// Return the encoded RLWINM constant, or 0 if they cannot be merged.
func mergePPC64ClrlsldiSrd(sld, srd int64) int64 {
	mask_1 := uint64(0xFFFFFFFFFFFFFFFF) >> uint(srd)
	// for CLRLSLDI, it's more convenient to think of it as a mask left bits then rotate left.
	mask_2 := uint64(0xFFFFFFFFFFFFFFFF) >> uint(GetPPC64Shiftmb(int64(sld)))

	// Rewrite mask to apply after the final left shift.
	mask_3 := (mask_1 & mask_2) << uint(GetPPC64Shiftsh(sld))

	r_1 := 64 - srd
	r_2 := GetPPC64Shiftsh(sld)
	r_3 := (r_1 + r_2) & 63 // This can wrap.

	if uint64(uint32(mask_3)) != mask_3 || mask_3 == 0 {
		return 0
	}
	// This combine only works when selecting and shifting the lower 32 bits.
	v1 := bits.RotateLeft64(0xFFFFFFFF00000000, int(r_3))
	if v1&mask_3 != 0 {
		return 0
	}
	return encodePPC64RotateMask(int64(r_3-32), int64(mask_3), 32)
}

// Test if a RLWINM feeding into a CLRLSLDI can be merged into RLWINM.  Return
// the encoded RLWINM constant, or 0 if they cannot be merged.
func mergePPC64ClrlsldiRlwinm(sld int32, rlw int64) int64 {
	r_1, _, _, mask_1 := DecodePPC64RotateMask(rlw)
	// for CLRLSLDI, it's more convenient to think of it as a mask left bits then rotate left.
	mask_2 := uint64(0xFFFFFFFFFFFFFFFF) >> uint(GetPPC64Shiftmb(int64(sld)))

	// combine the masks, and adjust for the final left shift.
	mask_3 := (mask_1 & mask_2) << uint(GetPPC64Shiftsh(int64(sld)))
	r_2 := GetPPC64Shiftsh(int64(sld))
	r_3 := (r_1 + r_2) & 31 // This can wrap.

	// Verify the result is still a valid bitmask of <= 32 bits.
	if !isPPC64WordRotateMask(int64(mask_3)) || uint64(uint32(mask_3)) != mask_3 {
		return 0
	}
	return encodePPC64RotateMask(r_3, int64(mask_3), 32)
}

// Compute the encoded RLWINM constant from combining (SLDconst [sld] (SRWconst [srw] x)),
// or return 0 if they cannot be combined.
func mergePPC64SldiSrw(sld, srw int64) int64 {
	if sld > srw || srw >= 32 {
		return 0
	}
	mask_r := uint32(0xFFFFFFFF) >> uint(srw)
	mask_l := uint32(0xFFFFFFFF) >> uint(sld)
	mask := (mask_r & mask_l) << uint(sld)
	return encodePPC64RotateMask((32-srw+sld)&31, int64(mask), 32)
}

// Convert a PPC64 opcode from the Op to OpCC form. This converts (op x y)
// to (Select0 (opCC x y)) without having to explicitly fixup every user
// of op.
//
// E.g consider the case:
// a = (ADD x y)
// b = (CMPconst [0] a)
// c = (OR a z)
//
// A rule like (CMPconst [0] (ADD x y)) => (CMPconst [0] (Select0 (ADDCC x y)))
// would produce:
// a  = (ADD x y)
// a' = (ADDCC x y)
// a” = (Select0 a')
// b  = (CMPconst [0] a”)
// c  = (OR a z)
//
// which makes it impossible to rewrite the second user. Instead the result
// of this conversion is:
// a' = (ADDCC x y)
// a  = (Select0 a')
// b  = (CMPconst [0] a)
// c  = (OR a z)
//
// Which makes it trivial to rewrite b using a lowering rule.
func convertPPC64OpToOpCC(op *Value) *Value {
	ccOpMap := map[Op]Op{
		OpPPC64ADD:      OpPPC64ADDCC,
		OpPPC64ADDconst: OpPPC64ADDCCconst,
		OpPPC64AND:      OpPPC64ANDCC,
		OpPPC64ANDN:     OpPPC64ANDNCC,
		OpPPC64CNTLZD:   OpPPC64CNTLZDCC,
		OpPPC64OR:       OpPPC64ORCC,
		OpPPC64SUB:      OpPPC64SUBCC,
		OpPPC64NEG:      OpPPC64NEGCC,
		OpPPC64NOR:      OpPPC64NORCC,
		OpPPC64XOR:      OpPPC64XORCC,
	}
	b := op.Block
	opCC := b.NewValue0I(op.Pos, ccOpMap[op.Op], types.NewTuple(op.Type, types.TypeFlags), op.AuxInt)
	opCC.AddArgs(op.Args...)
	op.reset(OpSelect0)
	op.AddArgs(opCC)
	return op
}

// Convenience function to rotate a 32 bit constant value by another constant.
func rotateLeft32(v, rotate int64) int64 {
	return int64(bits.RotateLeft32(uint32(v), int(rotate)))
}

func rotateRight64(v, rotate int64) int64 {
	return int64(bits.RotateLeft64(uint64(v), int(-rotate)))
}

// encodes the lsb and width for arm(64) bitfield ops into the expected auxInt format.
func armBFAuxInt(lsb, width int64) arm64BitField {
	if lsb < 0 || lsb > 63 {
		panic("ARM(64) bit field lsb constant out of range")
	}
	if width < 1 || lsb+width > 64 {
		panic("ARM(64) bit field width constant out of range")
	}
	return arm64BitField(width | lsb<<8)
}

// returns the lsb part of the auxInt field of arm64 bitfield ops.
func (bfc arm64BitField) getARM64BFlsb() int64 {
	return int64(uint64(bfc) >> 8)
}

// returns the width part of the auxInt field of arm64 bitfield ops.
func (bfc arm64BitField) getARM64BFwidth() int64 {
	return int64(bfc) & 0xff
}

// checks if mask >> rshift applied at lsb is a valid arm64 bitfield op mask.
func isARM64BFMask(lsb, mask, rshift int64) bool {
	shiftedMask := int64(uint64(mask) >> uint64(rshift))
	return shiftedMask != 0 && isPowerOfTwo64(shiftedMask+1) && nto(shiftedMask)+lsb < 64
}

// returns the bitfield width of mask >> rshift for arm64 bitfield ops.
func arm64BFWidth(mask, rshift int64) int64 {
	shiftedMask := int64(uint64(mask) >> uint64(rshift))
	if shiftedMask == 0 {
		panic("ARM64 BF mask is zero")
	}
	return nto(shiftedMask)
}

// sizeof returns the size of t in bytes.
// It will panic if t is not a *types.Type.
func sizeof(t interface{}) int64 {
	return t.(*types.Type).Size()
}

// registerizable reports whether t is a primitive type that fits in
// a register. It assumes float64 values will always fit into registers
// even if that isn't strictly true.
func registerizable(b *Block, typ *types.Type) bool {
	if typ.IsPtrShaped() || typ.IsFloat() || typ.IsBoolean() {
		return true
	}
	if typ.IsInteger() {
		return typ.Size() <= b.Func.Config.RegSize
	}
	return false
}

// needRaceCleanup reports whether this call to racefuncenter/exit isn't needed.
func needRaceCleanup(sym *AuxCall, v *Value) bool {
	f := v.Block.Func
	if !f.Config.Race {
		return false
	}
	if !isSameCall(sym, "runtime.racefuncenter") && !isSameCall(sym, "runtime.racefuncexit") {
		return false
	}
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			switch v.Op {
			case OpStaticCall, OpStaticLECall:
				// Check for racefuncenter will encounter racefuncexit and vice versa.
				// Allow calls to panic*
				s := v.Aux.(*AuxCall).Fn.String()
				switch s {
				case "runtime.racefuncenter", "runtime.racefuncexit",
					"runtime.panicdivide", "runtime.panicwrap",
					"runtime.panicshift":
					continue
				}
				// If we encountered any call, we need to keep racefunc*,
				// for accurate stacktraces.
				return false
			case OpPanicBounds, OpPanicExtend:
				// Note: these are panic generators that are ok (like the static calls above).
			case OpClosureCall, OpInterCall, OpClosureLECall, OpInterLECall:
				// We must keep the race functions if there are any other call types.
				return false
			}
		}
	}
	if isSameCall(sym, "runtime.racefuncenter") {
		// TODO REGISTER ABI this needs to be cleaned up.
		// If we're removing racefuncenter, remove its argument as well.
		if v.Args[0].Op != OpStore {
			if v.Op == OpStaticLECall {
				// there is no store, yet.
				return true
			}
			return false
		}
		mem := v.Args[0].Args[2]
		v.Args[0].reset(OpCopy)
		v.Args[0].AddArg(mem)
	}
	return true
}

// symIsRO reports whether sym is a read-only global.
func symIsRO(sym interface{}) bool {
	lsym := sym.(*obj.LSym)
	return lsym.Type == objabi.SRODATA && len(lsym.R) == 0
}

// symIsROZero reports whether sym is a read-only global whose data contains all zeros.
func symIsROZero(sym Sym) bool {
	lsym := sym.(*obj.LSym)
	if lsym.Type != objabi.SRODATA || len(lsym.R) != 0 {
		return false
	}
	for _, b := range lsym.P {
		if b != 0 {
			return false
		}
	}
	return true
}

// isFixed32 returns true if the int32 at offset off in symbol sym
// is known and constant.
func isFixed32(c *Config, sym Sym, off int64) bool {
	return isFixed(c, sym, off, 4)
}

// isFixed returns true if the range [off,off+size] of the symbol sym
// is known and constant.
func isFixed(c *Config, sym Sym, off, size int64) bool {
	lsym := sym.(*obj.LSym)
	if lsym.Extra == nil {
		return false
	}
	if _, ok := (*lsym.Extra).(*obj.TypeInfo); ok {
		if off == 2*c.PtrSize && size == 4 {
			return true // type hash field
		}
	}
	return false
}
func fixed32(c *Config, sym Sym, off int64) int32 {
	lsym := sym.(*obj.LSym)
	if ti, ok := (*lsym.Extra).(*obj.TypeInfo); ok {
		if off == 2*c.PtrSize {
			return int32(types.TypeHash(ti.Type.(*types.Type)))
		}
	}
	base.Fatalf("fixed32 data not known for %s:%d", sym, off)
	return 0
}

// isFixedSym returns true if the contents of sym at the given offset
// is known and is the constant address of another symbol.
func isFixedSym(sym Sym, off int64) bool {
	lsym := sym.(*obj.LSym)
	switch {
	case lsym.Type == objabi.SRODATA:
		// itabs, dictionaries
	default:
		return false
	}
	for _, r := range lsym.R {
		if (r.Type == objabi.R_ADDR || r.Type == objabi.R_WEAKADDR) && int64(r.Off) == off && r.Add == 0 {
			return true
		}
	}
	return false
}
func fixedSym(f *Func, sym Sym, off int64) Sym {
	lsym := sym.(*obj.LSym)
	for _, r := range lsym.R {
		if (r.Type == objabi.R_ADDR || r.Type == objabi.R_WEAKADDR) && int64(r.Off) == off {
			if strings.HasPrefix(r.Sym.Name, "type:") {
				// In case we're loading a type out of a dictionary, we need to record
				// that the containing function might put that type in an interface.
				// That information is currently recorded in relocations in the dictionary,
				// but if we perform this load at compile time then the dictionary
				// might be dead.
				reflectdata.MarkTypeSymUsedInInterface(r.Sym, f.fe.Func().Linksym())
			} else if strings.HasPrefix(r.Sym.Name, "go:itab") {
				// Same, but if we're using an itab we need to record that the
				// itab._type might be put in an interface.
				reflectdata.MarkTypeSymUsedInInterface(r.Sym, f.fe.Func().Linksym())
			}
			return r.Sym
		}
	}
	base.Fatalf("fixedSym data not known for %s:%d", sym, off)
	return nil
}

// read8 reads one byte from the read-only global sym at offset off.
func read8(sym interface{}, off int64) uint8 {
	lsym := sym.(*obj.LSym)
	if off >= int64(len(lsym.P)) || off < 0 {
		// Invalid index into the global sym.
		// This can happen in dead code, so we don't want to panic.
		// Just return any value, it will eventually get ignored.
		// See issue 29215.
		return 0
	}
	return lsym.P[off]
}

// read16 reads two bytes from the read-only global sym at offset off.
func read16(sym interface{}, off int64, byteorder binary.ByteOrder) uint16 {
	lsym := sym.(*obj.LSym)
	// lsym.P is written lazily.
	// Bytes requested after the end of lsym.P are 0.
	var src []byte
	if 0 <= off && off < int64(len(lsym.P)) {
		src = lsym.P[off:]
	}
	buf := make([]byte, 2)
	copy(buf, src)
	return byteorder.Uint16(buf)
}

// read32 reads four bytes from the read-only global sym at offset off.
func read32(sym interface{}, off int64, byteorder binary.ByteOrder) uint32 {
	lsym := sym.(*obj.LSym)
	var src []byte
	if 0 <= off && off < int64(len(lsym.P)) {
		src = lsym.P[off:]
	}
	buf := make([]byte, 4)
	copy(buf, src)
	return byteorder.Uint32(buf)
}

// read64 reads eight bytes from the read-only global sym at offset off.
func read64(sym interface{}, off int64, byteorder binary.ByteOrder) uint64 {
	lsym := sym.(*obj.LSym)
	var src []byte
	if 0 <= off && off < int64(len(lsym.P)) {
		src = lsym.P[off:]
	}
	buf := make([]byte, 8)
	copy(buf, src)
	return byteorder.Uint64(buf)
}

// sequentialAddresses reports true if it can prove that x + n == y
func sequentialAddresses(x, y *Value, n int64) bool {
	if x == y && n == 0 {
		return true
	}
	if x.Op == Op386ADDL && y.Op == Op386LEAL1 && y.AuxInt == n && y.Aux == nil &&
		(x.Args[0] == y.Args[0] && x.Args[1] == y.Args[1] ||
			x.Args[0] == y.Args[1] && x.Args[1] == y.Args[0]) {
		return true
	}
	if x.Op == Op386LEAL1 && y.Op == Op386LEAL1 && y.AuxInt == x.AuxInt+n && x.Aux == y.Aux &&
		(x.Args[0] == y.Args[0] && x.Args[1] == y.Args[1] ||
			x.Args[0] == y.Args[1] && x.Args[1] == y.Args[0]) {
		return true
	}
	if x.Op == OpAMD64ADDQ && y.Op == OpAMD64LEAQ1 && y.AuxInt == n && y.Aux == nil &&
		(x.Args[0] == y.Args[0] && x.Args[1] == y.Args[1] ||
			x.Args[0] == y.Args[1] && x.Args[1] == y.Args[0]) {
		return true
	}
	if x.Op == OpAMD64LEAQ1 && y.Op == OpAMD64LEAQ1 && y.AuxInt == x.AuxInt+n && x.Aux == y.Aux &&
		(x.Args[0] == y.Args[0] && x.Args[1] == y.Args[1] ||
			x.Args[0] == y.Args[1] && x.Args[1] == y.Args[0]) {
		return true
	}
	return false
}

// flagConstant represents the result of a compile-time comparison.
// The sense of these flags does not necessarily represent the hardware's notion
// of a flags register - these are just a compile-time construct.
// We happen to match the semantics to those of arm/arm64.
// Note that these semantics differ from x86: the carry flag has the opposite
// sense on a subtraction!
//
//	On amd64, C=1 represents a borrow, e.g. SBB on amd64 does x - y - C.
//	On arm64, C=0 represents a borrow, e.g. SBC on arm64 does x - y - ^C.
//	 (because it does x + ^y + C).
//
// See https://en.wikipedia.org/wiki/Carry_flag#Vs._borrow_flag
type flagConstant uint8

// N reports whether the result of an operation is negative (high bit set).
func (fc flagConstant) N() bool {
	return fc&1 != 0
}

// Z reports whether the result of an operation is 0.
func (fc flagConstant) Z() bool {
	return fc&2 != 0
}

// C reports whether an unsigned add overflowed (carry), or an
// unsigned subtract did not underflow (borrow).
func (fc flagConstant) C() bool {
	return fc&4 != 0
}

// V reports whether a signed operation overflowed or underflowed.
func (fc flagConstant) V() bool {
	return fc&8 != 0
}

func (fc flagConstant) eq() bool {
	return fc.Z()
}
func (fc flagConstant) ne() bool {
	return !fc.Z()
}
func (fc flagConstant) lt() bool {
	return fc.N() != fc.V()
}
func (fc flagConstant) le() bool {
	return fc.Z() || fc.lt()
}
func (fc flagConstant) gt() bool {
	return !fc.Z() && fc.ge()
}
func (fc flagConstant) ge() bool {
	return fc.N() == fc.V()
}
func (fc flagConstant) ult() bool {
	return !fc.C()
}
func (fc flagConstant) ule() bool {
	return fc.Z() || fc.ult()
}
func (fc flagConstant) ugt() bool {
	return !fc.Z() && fc.uge()
}
func (fc flagConstant) uge() bool {
	return fc.C()
}

func (fc flagConstant) ltNoov() bool {
	return fc.lt() && !fc.V()
}
func (fc flagConstant) leNoov() bool {
	return fc.le() && !fc.V()
}
func (fc flagConstant) gtNoov() bool {
	return fc.gt() && !fc.V()
}
func (fc flagConstant) geNoov() bool {
	return fc.ge() && !fc.V()
}

func (fc flagConstant) String() string {
	return fmt.Sprintf("N=%v,Z=%v,C=%v,V=%v", fc.N(), fc.Z(), fc.C(), fc.V())
}

type flagConstantBuilder struct {
	N bool
	Z bool
	C bool
	V bool
}

func (fcs flagConstantBuilder) encode() flagConstant {
	var fc flagConstant
	if fcs.N {
		fc |= 1
	}
	if fcs.Z {
		fc |= 2
	}
	if fcs.C {
		fc |= 4
	}
	if fcs.V {
		fc |= 8
	}
	return fc
}

// Note: addFlags(x,y) != subFlags(x,-y) in some situations:
//  - the results of the C flag are different
//  - the results of the V flag when y==minint are different

// addFlags64 returns the flags that would be set from computing x+y.
func addFlags64(x, y int64) flagConstant {
	var fcb flagConstantBuilder
	fcb.Z = x+y == 0
	fcb.N = x+y < 0
	fcb.C = uint64(x+y) < uint64(x)
	fcb.V = x >= 0 && y >= 0 && x+y < 0 || x < 0 && y < 0 && x+y >= 0
	return fcb.encode()
}

// subFlags64 returns the flags that would be set from computing x-y.
func subFlags64(x, y int64) flagConstant {
	var fcb flagConstantBuilder
	fcb.Z = x-y == 0
	fcb.N = x-y < 0
	fcb.C = uint64(y) <= uint64(x) // This code follows the arm carry flag model.
	fcb.V = x >= 0 && y < 0 && x-y < 0 || x < 0 && y >= 0 && x-y >= 0
	return fcb.encode()
}

// addFlags32 returns the flags that would be set from computing x+y.
func addFlags32(x, y int32) flagConstant {
	var fcb flagConstantBuilder
	fcb.Z = x+y == 0
	fcb.N = x+y < 0
	fcb.C = uint32(x+y) < uint32(x)
	fcb.V = x >= 0 && y >= 0 && x+y < 0 || x < 0 && y < 0 && x+y >= 0
	return fcb.encode()
}

// subFlags32 returns the flags that would be set from computing x-y.
func subFlags32(x, y int32) flagConstant {
	var fcb flagConstantBuilder
	fcb.Z = x-y == 0
	fcb.N = x-y < 0
	fcb.C = uint32(y) <= uint32(x) // This code follows the arm carry flag model.
	fcb.V = x >= 0 && y < 0 && x-y < 0 || x < 0 && y >= 0 && x-y >= 0
	return fcb.encode()
}

// logicFlags64 returns flags set to the sign/zeroness of x.
// C and V are set to false.
func logicFlags64(x int64) flagConstant {
	var fcb flagConstantBuilder
	fcb.Z = x == 0
	fcb.N = x < 0
	return fcb.encode()
}

// logicFlags32 returns flags set to the sign/zeroness of x.
// C and V are set to false.
func logicFlags32(x int32) flagConstant {
	var fcb flagConstantBuilder
	fcb.Z = x == 0
	fcb.N = x < 0
	return fcb.encode()
}

func makeJumpTableSym(b *Block) *obj.LSym {
	s := base.Ctxt.Lookup(fmt.Sprintf("%s.jump%d", b.Func.fe.Func().LSym.Name, b.ID))
	// The jump table symbol is accessed only from the function symbol.
	s.Set(obj.AttrStatic, true)
	return s
}

// canRotate reports whether the architecture supports
// rotates of integer registers with the given number of bits.
func canRotate(c *Config, bits int64) bool {
	if bits > c.PtrSize*8 {
		// Don't rewrite to rotates bigger than the machine word.
		return false
	}
	switch c.arch {
	case "386", "amd64", "arm64", "riscv64":
		return true
	case "arm", "s390x", "ppc64", "ppc64le", "wasm", "loong64":
		return bits >= 32
	default:
		return false
	}
}

// isARM64bitcon reports whether a constant can be encoded into a logical instruction.
func isARM64bitcon(x uint64) bool {
	if x == 1<<64-1 || x == 0 {
		return false
	}
	// determine the period and sign-extend a unit to 64 bits
	switch {
	case x != x>>32|x<<32:
		// period is 64
		// nothing to do
	case x != x>>16|x<<48:
		// period is 32
		x = uint64(int64(int32(x)))
	case x != x>>8|x<<56:
		// period is 16
		x = uint64(int64(int16(x)))
	case x != x>>4|x<<60:
		// period is 8
		x = uint64(int64(int8(x)))
	default:
		// period is 4 or 2, always true
		// 0001, 0010, 0100, 1000 -- 0001 rotate
		// 0011, 0110, 1100, 1001 -- 0011 rotate
		// 0111, 1011, 1101, 1110 -- 0111 rotate
		// 0101, 1010             -- 01   rotate, repeat
		return true
	}
	return sequenceOfOnes(x) || sequenceOfOnes(^x)
}

// sequenceOfOnes tests whether a constant is a sequence of ones in binary, with leading and trailing zeros.
func sequenceOfOnes(x uint64) bool {
	y := x & -x // lowest set bit of x. x is good iff x+y is a power of 2
	y += x
	return (y-1)&y == 0
}

// isARM64addcon reports whether x can be encoded as the immediate value in an ADD or SUB instruction.
func isARM64addcon(v int64) bool {
	/* uimm12 or uimm24? */
	if v < 0 {
		return false
	}
	if (v & 0xFFF) == 0 {
		v >>= 12
	}
	return v <= 0xFFF
}

// setPos sets the position of v to pos, then returns true.
// Useful for setting the result of a rewrite's position to
// something other than the default.
func setPos(v *Value, pos src.XPos) bool {
	v.Pos = pos
	return true
}
