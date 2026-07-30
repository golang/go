// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"encoding/binary"
	"fmt"
	"internal/buildcfg"
	"io"
	"math"
	"math/bits"
	"os"
	"path/filepath"

	"cmd/compile/internal/base"
	"cmd/compile/internal/logopt"
	"cmd/compile/internal/ssa/ssacore"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/compile/internal/types"
	"cmd/internal/obj"
	"cmd/internal/obj/s390x"
	"cmd/internal/objabi"
	"cmd/internal/src"
)

type DeadValueChoice bool

const (
	LeaveDeadValues  DeadValueChoice = false
	RemoveDeadValues                 = true

	RepZeroThreshold = 1408 // size beyond which we use REP STOS for zeroing
	RepMoveThreshold = 1408 // size beyond which we use REP MOVS for copying
)

// deadcode indicates whether rewrite should try to remove any values that become dead.
func applyRewrite(f *ssacore.Func, rb ssacore.BlockRewriter, rv ssacore.ValueRewriter, deadcode DeadValueChoice) {
	// repeat rewrites until we find no more rewrites
	pendingLines := f.CachedLineStarts // Holds statement boundaries that need to be moved to a new value/block
	pendingLines.Clear()
	debug := f.Pass.Debug
	if debug > 1 {
		fmt.Printf("%s: rewriting for %s\n", f.Pass.Name, f.Name)
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
		if debug > 1 {
			fmt.Printf("%s: iter %d\n", f.Pass.Name, iters)
		}
		change := false
		deadChange := false
		for _, b := range f.Blocks {
			var b0 *ssacore.Block
			if debug > 1 {
				fmt.Printf("%s: start block\n", f.Pass.Name)
				b0 = new(ssacore.Block)
				*b0 = *b
				b0.Succs = append([]ssacore.Edge{}, b.Succs...) // make a new copy, not aliasing
			}
			for i, c := range b.ControlValues() {
				for c.Op == ssaop.OpCopy {
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
				if debug > 1 {
					fmt.Printf("%s: consider %v\n", f.Pass.Name, v.LongString())
				}
				var v0 *ssacore.Value
				if debug > 1 {
					v0 = new(ssacore.Value)
					*v0 = *v
					v0.Args = append([]*ssacore.Value{}, v.Args...) // make a new copy, not aliasing
				}
				if v.Uses == 0 && v.Removeable() {
					if v.Op != ssaop.OpInvalid && deadcode == RemoveDeadValues {
						// Reset any values that are now unused, so that we decrement
						// the use count of all of its arguments.
						// Not quite a deadcode pass, because it does not handle cycles.
						// But it should help Uses==1 rules to fire.
						v.Reset(ssaop.OpInvalid)
						deadChange = true
					}
					// No point rewriting values which aren't used.
					continue
				}

				vchange := ssacore.PhiElimValue(v)
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
					if a.Op != ssaop.OpCopy {
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
							pendingLines.Set(a.Pos, int32(a.Block.ID))
						}
						a.Pos = a.Pos.WithNotStmt()
					}
					vchange = true
					for a.Uses == 0 {
						b := a.Args[0]
						a.Reset(ssaop.OpInvalid)
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
			h := f.RewriteHash()
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
			if v.Op == ssaop.OpInvalid {
				if v.Pos.IsStmt() == src.PosIsStmt {
					pendingLines.Set(vl, int32(b.ID))
				}
				f.FreeValue(v)
				continue
			}
			if v.Pos.IsStmt() != src.PosNotStmt && !ssacore.NotStmtBoundary(v.Op) {
				if pl, ok := pendingLines.Get(vl); ok && pl == int32(b.ID) {
					pendingLines.Remove(vl)
					v.Pos = v.Pos.WithIsStmt()
				}
			}
			if i != j {
				b.Values[j] = v
			}
			j++
		}
		if pl, ok := pendingLines.Get(b.Pos); ok && pl == int32(b.ID) {
			b.Pos = b.Pos.WithIsStmt()
			pendingLines.Remove(b.Pos)
		}
		b.TruncateValues(j)
	}
}

// Common functions called from rewriting rules

func Is64BitFloat(t *types.Type) bool {
	return t.Size() == 8 && t.IsFloat()
}

func Is32BitFloat(t *types.Type) bool {
	return t.Size() == 4 && t.IsFloat()
}

func Is64BitInt(t *types.Type) bool {
	return t.Size() == 8 && t.IsInteger()
}

func Is32BitInt(t *types.Type) bool {
	return t.Size() == 4 && t.IsInteger()
}

func Is16BitInt(t *types.Type) bool {
	return t.Size() == 2 && t.IsInteger()
}

func Is8BitInt(t *types.Type) bool {
	return t.Size() == 1 && t.IsInteger()
}

func IsPtr(t *types.Type) bool {
	return t.IsPtrShaped()
}

// MergeSym merges two symbolic offsets. There is no real merging of
// offsets, we just pick the non-nil one.
func MergeSym(x, y ssacore.Sym) ssacore.Sym {
	if x == nil {
		return y
	}
	if y == nil {
		return x
	}
	panic(fmt.Sprintf("mergeSym with two non-nil syms %v %v", x, y))
}

func CanMergeSym(x, y ssacore.Sym) bool {
	return x == nil || y == nil
}

// CanMergeLoadClobber reports whether the load can be merged into target without
// invalidating the schedule.
// It also checks that the other non-load argument x is something we
// are ok with clobbering.
func CanMergeLoadClobber(target, load, x *ssacore.Value) bool {
	// The register containing x is going to get clobbered.
	// Don't merge if we still need the value of x.
	// We don't have liveness information here, but we can
	// approximate x dying with:
	//  1) target is x's only use.
	//  2) target is not in a deeper loop than x.
	switch {
	case x.Uses == 2 && x.Op == ssaop.OpPhi && len(x.Args) == 2 && (x.Args[0] == target || x.Args[1] == target) && target.Uses == 1:
		// This is a simple detector to determine that x is probably
		// not live after target. (It does not need to be perfect,
		// regalloc will issue a reg-reg move to save it if we are wrong.)
		// We have:
		//   x = Phi(?, target)
		//   target = Op(load, x)
		// Because target has only one use as a Phi argument, we can schedule it
		// very late. Hopefully, later than the other use of x. (The other use died
		// between x and target, or exists on another branch entirely).
	case x.Uses > 1:
		return false
	}
	loopnest := x.Block.Func.Loopnest()
	if loopnest.Depth(target.Block.ID) > loopnest.Depth(x.Block.ID) {
		return false
	}
	return CanMergeLoad(target, load)
}

// CanMergeLoad reports whether the load can be merged into target without
// invalidating the schedule.
func CanMergeLoad(target, load *ssacore.Value) bool {
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
	var args []*ssacore.Value
	for _, a := range target.Args {
		if a != load && a.Block.ID == target.Block.ID {
			args = append(args, a)
		}
	}

	f := target.Block.Func
	visited := f.NewSparseSet(f.NumValues())
	defer f.RetSparseSet(visited)

	// memPreds contains memory states known to be predecessors of load's
	// memory state. It is lazily initialized.
	var memPreds map[*ssacore.Value]bool
	for len(args) > 0 {
		const limit = 2048 // enough to comfortably cover unrolled crypto blocks
		if visited.Size() >= limit {
			// Give up if we have visited a lot of values.
			return false
		}
		v := args[len(args)-1]
		args = args[:len(args)-1]
		if visited.Contains(v.ID) {
			continue
		}
		visited.Add(v.ID)
		if target.Block.ID != v.Block.ID {
			// Since target and load are in the same block
			// we can stop searching when we leave the block.
			continue
		}
		if v.Op == ssaop.OpPhi {
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
		if v.Op.SymEffect()&ssaop.SymAddr != 0 {
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
				memPreds = make(map[*ssacore.Value]bool)
				m := mem
				const limit = 50
				for i := 0; i < limit; i++ {
					if m.Op == ssaop.OpPhi {
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

// IsSameCall reports whether aux is the same as the given named symbol.
func IsSameCall(aux ssacore.Aux, name string) bool {
	fn := aux.(*ssacore.AuxCall).Fn
	return fn != nil && fn.String() == name
}

// ntzX returns the number of trailing zeros.
func Ntz64(x int64) int { return bits.TrailingZeros64(uint64(x)) }

// OneBit reports whether x contains exactly one set bit.
func OneBit[T int8 | int16 | int32 | int64](x T) bool {
	return x&(x-1) == 0 && x != 0
}

func Log16(n int16) int64 { return Log16u(uint16(n)) }
func Log32(n int32) int64 { return Log32u(uint32(n)) }
func Log64(n int64) int64 { return Log64u(uint64(n)) }

// logXu returns the logarithm of n base 2.
// n must be a power of 2 (isPowerOfTwo returns true)
func Log8u(n uint8) int64   { return int64(bits.Len8(n)) - 1 }
func Log16u(n uint16) int64 { return int64(bits.Len16(n)) - 1 }
func Log32u(n uint32) int64 { return int64(bits.Len32(n)) - 1 }
func Log64u(n uint64) int64 { return int64(bits.Len64(n)) - 1 }

// isPowerOfTwoX functions report whether n is a power of 2.
func IsPowerOfTwo[T int8 | int16 | int32 | int64 | uint8 | uint16 | uint32 | uint64](n T) bool {
	return n > 0 && n&(n-1) == 0
}

// Is32Bit reports whether n can be represented as a signed 32 bit integer.
func Is32Bit(n int64) bool {
	return n == int64(int32(n))
}

// Is16Bit reports whether n can be represented as a signed 16 bit integer.
func Is16Bit(n int64) bool {
	return n == int64(int16(n))
}

// Is12Bit reports whether n can be represented as a signed 12 bit integer.
func Is12Bit(n int64) bool {
	return -(1<<11) <= n && n < (1<<11)
}

// IsU32Bit reports whether n can be represented as an unsigned 32 bit integer.
func IsU32Bit(n int64) bool {
	return n == int64(uint32(n))
}

// Is20Bit reports whether n can be represented as a signed 20 bit integer.
func Is20Bit(n int64) bool {
	return -(1<<19) <= n && n < (1<<19)
}

// B2i translates a boolean value to 0 or 1 for assigning to auxInt.
func B2i(b bool) int64 {
	if b {
		return 1
	}
	return 0
}

// B2i32 translates a boolean value to 0 or 1.
func B2i32(b bool) int32 {
	if b {
		return 1
	}
	return 0
}

func CanMulStrengthReduce(config *ssacore.Config, x int64) bool {
	_, ok := config.MulRecipes[x]
	return ok
}
func CanMulStrengthReduce32(config *ssacore.Config, x int32) bool {
	_, ok := config.MulRecipes[int64(x)]
	return ok
}

// MulStrengthReduce returns v*x evaluated at the location
// (block and source position) of m.
// canMulStrengthReduce must have returned true.
func MulStrengthReduce(m *ssacore.Value, v *ssacore.Value, x int64) *ssacore.Value {
	return v.Block.Func.Config.MulRecipes[x].Build(m, v)
}

// MulStrengthReduce32 returns v*x evaluated at the location
// (block and source position) of m.
// canMulStrengthReduce32 must have returned true.
// The upper 32 bits of m might be set to junk.
func MulStrengthReduce32(m *ssacore.Value, v *ssacore.Value, x int32) *ssacore.Value {
	return v.Block.Func.Config.MulRecipes[int64(x)].Build(m, v)
}

// ShiftIsBounded reports whether (left/right) shift Value v is known to be bounded.
// A shift is bounded if it is shifting by less than the width of the shifted value.
func ShiftIsBounded(v *ssacore.Value) bool {
	return v.AuxInt != 0
}

// CanonLessThan returns whether x is "ordered" less than y, for purposes of normalizing
// generated code as much as possible.
func CanonLessThan(x, y *ssacore.Value) bool {
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

// auxTo32F decodes a float32 from the AuxInt value provided.
func auxTo32F(i int64) float32 {
	return truncate64Fto32F(math.Float64frombits(uint64(i)))
}

func AuxIntToBool(i int64) bool {
	if i == 0 {
		return false
	}
	return true
}
func AuxIntToInt8(i int64) int8 {
	return int8(i)
}
func AuxIntToInt16(i int64) int16 {
	return int16(i)
}
func AuxIntToInt32(i int64) int32 {
	return int32(i)
}
func AuxIntToInt64(i int64) int64 {
	return i
}
func AuxIntToUint8(i int64) uint8 {
	return uint8(i)
}
func AuxIntToUint64(i int64) uint64 {
	return uint64(i)
}
func AuxIntToFloat32(i int64) float32 {
	return float32(math.Float64frombits(uint64(i)))
}
func AuxIntToFloat64(i int64) float64 {
	return math.Float64frombits(uint64(i))
}
func AuxIntToValAndOff(i int64) ssacore.ValAndOff {
	return ssacore.ValAndOff(i)
}
func AuxIntToArm64BitField(i int64) ssacore.Arm64BitField {
	return ssacore.Arm64BitField(i)
}
func AuxIntToFlagConstant(x int64) ssacore.FlagConstant {
	return ssacore.FlagConstant(x)
}

func AuxIntToOp(cc int64) ssaop.Op {
	return ssaop.Op(cc)
}

func Int8ToAuxInt(i int8) int64 {
	return int64(i)
}
func Int16ToAuxInt(i int16) int64 {
	return int64(i)
}
func Int32ToAuxInt(i int32) int64 {
	return int64(i)
}
func Int64ToAuxInt(i int64) int64 {
	return i
}
func Uint8ToAuxInt(i uint8) int64 {
	return int64(int8(i))
}
func Uint64ToAuxInt(i uint64) int64 {
	return int64(i)
}
func Float32ToAuxInt(f float32) int64 {
	return int64(math.Float64bits(float64(f)))
}
func Float64ToAuxInt(f float64) int64 {
	return int64(math.Float64bits(f))
}
func ValAndOffToAuxInt(v ssacore.ValAndOff) int64 {
	return int64(v)
}
func Arm64BitFieldToAuxInt(v ssacore.Arm64BitField) int64 {
	return int64(v)
}
func Arm64ConditionalParamsToAuxInt(v ssacore.Arm64ConditionalParams) int64 {
	if v.Cond&^0xffff != 0 {
		panic("condition value exceeds 16 bits")
	}

	var i int64
	if v.Ind {
		i = 1 << 25
	}
	i |= int64(v.ConstVal) << 20
	i |= int64(v.NzcvVal) << 16
	i |= int64(v.Cond)
	return i
}

func FlagConstantToAuxInt(x ssacore.FlagConstant) int64 {
	return int64(x)
}

func OpToAuxInt(o ssaop.Op) int64 {
	return int64(o)
}

func AuxToString(i ssacore.Aux) string {
	return string(i.(ssacore.StringAux))
}
func AuxToSym(i ssacore.Aux) ssacore.Sym {
	// TODO: kind of a hack - allows nil interface through
	s, _ := i.(ssacore.Sym)
	return s
}
func AuxToType(i ssacore.Aux) *types.Type {
	return i.(*types.Type)
}
func AuxToCall(i ssacore.Aux) *ssacore.AuxCall {
	return i.(*ssacore.AuxCall)
}
func AuxToS390xCCMask(i ssacore.Aux) s390x.CCMask {
	return i.(s390x.CCMask)
}
func AuxToS390xRotateParams(i ssacore.Aux) s390x.RotateParams {
	return i.(s390x.RotateParams)
}

func SymToAux(s ssacore.Sym) ssacore.Aux {
	return s
}
func CallToAux(s *ssacore.AuxCall) ssacore.Aux {
	return s
}
func TypeToAux(t *types.Type) ssacore.Aux {
	return t
}
func S390xCCMaskToAux(c s390x.CCMask) ssacore.Aux {
	return c
}
func S390xRotateParamsToAux(r s390x.RotateParams) ssacore.Aux {
	return r
}

// MoveSize returns the number of bytes an aligned MOV instruction moves.
func MoveSize(align int64, c *ssacore.Config) int64 {
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
func mergePoint(b *ssacore.Block, a ...*ssacore.Value) *ssacore.Block {
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
		b = b.Preds[0].B
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
		b = b.Preds[0].B
		d--

	}
	return nil // too far away
}

// Clobber invalidates values. Returns true.
// Clobber is used by rewrite rules to:
//
//	A) make sure the values are really dead and never used again.
//	B) decrement use counts of the values' args.
func Clobber(vv ...*ssacore.Value) bool {
	for _, v := range vv {
		v.Reset(ssaop.OpInvalid)
		// Note: leave v.Block intact.  The Block field is used after clobber.
	}
	return true
}

// ClobberIfDead resets v when use count is 1. Returns true.
// ClobberIfDead is used by rewrite rules to decrement
// use counts of v's args when v is dead and never used.
func ClobberIfDead(v *ssacore.Value) bool {
	if v.Uses == 1 {
		v.Reset(ssaop.OpInvalid)
	}
	// Note: leave v.Block intact.  The Block field is used after clobberIfDead.
	return true
}

// NoteRule is an easy way to track if a rule is matched when writing
// new ones.  Make the rule of interest also conditional on
//
//	NoteRule("note to self: rule of interest matched")
//
// and that message will print when the rule matches.
func NoteRule(s string) bool {
	fmt.Println(s)
	return true
}

// CountRule increments Func.ruleMatches[key].
// If Func.ruleMatches is non-nil at the end
// of compilation, it will be printed to stdout.
// This is intended to make it easier to find which functions
// which contain lots of rules matches when developing new rules.
func CountRule(v *ssacore.Value, key string) bool {
	f := v.Block.Func
	if f.RuleMatches == nil {
		f.RuleMatches = make(map[string]int)
	}
	f.RuleMatches[key]++
	return true
}

// for a pseudo-op like (LessThan x), extract x.
func FlagArg(v *ssacore.Value) *ssacore.Value {
	if len(v.Args) != 1 || !v.Args[0].Type.IsFlags() {
		return nil
	}
	return v.Args[0]
}

// LogRule logs the use of the rule s. This will only be enabled if
// rewrite rules were generated with the -log option, see _gen/rulegen.go.
func LogRule(s string) {
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
	// Ignore errors in case of multiple processes fighting over the file.
	fmt.Fprintln(ruleFile, s)
}

var ruleFile io.Writer

// LogLargeCopyValue logs the occurrence of a large copy.
// The best place to do this is in the rewrite rules where the size of the move is easy to find.
// "Large" is arbitrarily chosen to be 128 bytes; this may change.
func LogLargeCopyValue(v *ssacore.Value, s int64) bool {
	if s < 128 {
		return true
	}
	if logopt.Enabled() {
		logopt.LogOpt(v.Pos, "copy", "lower", v.Block.Func.Name, fmt.Sprintf("%d bytes", s))
	}
	return true
}
func SupportsPPC64PCRel() bool {
	// PCRel is currently supported for >= power10, linux only
	// Internal and external linking supports this on ppc64le; internal linking on ppc64.
	return buildcfg.GOPPC64 >= 10 && buildcfg.GOOS == "linux"
}

func NewPPC64ShiftAuxInt(sh, mb, me, sz int64) int32 {
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

// Test if this value can encoded as a mask for a rlwinm like
// operation.  Masks can also extend from the msb and wrap to
// the lsb too.  That is, the valid masks are 32 bit strings
// of the form: 0..01..10..0 or 1..10..01..1 or 1...1
//
// Note: This ignores the upper 32 bits of the input. When a
// zero extended result is desired (e.g a 64 bit result), the
// user must verify the upper 32 bits are 0 and the mask is
// contiguous (that is, non-wrapping).
func IsPPC64WordRotateMask(v64 int64) bool {
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
func EncodePPC64RotateMask(rotate, mask, nbits int64) int64 {
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

	return int64(me) | int64(mb<<8) | rotate<<16 | nbits<<24
}

// This verifies that the mask is a set of
// consecutive bits including the least
// significant bit.
func IsPPC64ValidShiftMask(v int64) bool {
	if (v != 0) && ((v+1)&v) == 0 {
		return true
	}
	return false
}

// Decompose a shift right into an equivalent rotate/mask,
// and return mask & m.
func MergePPC64RShiftMask(m, s, nbits int64) int64 {
	smask := uint64((1<<uint(nbits))-1) >> uint(s)
	return m & int64(smask)
}

// Combine (ANDconst [m] (SRWconst [s])) into (RLWINM [y]) or return 0
func MergePPC64AndSrwi(m, s int64) int64 {
	mask := MergePPC64RShiftMask(m, s, 32)
	if !IsPPC64WordRotateMask(mask) {
		return 0
	}
	return EncodePPC64RotateMask((32-s)&31, mask, 32)
}

// Test if a word shift right feeding into a CLRLSLDI can be merged into RLWINM.
// Return the encoded RLWINM constant, or 0 if they cannot be merged.
func MergePPC64ClrlsldiSrw(sld, srw int64) int64 {
	mask_1 := uint64(0xFFFFFFFF >> uint(srw))
	// for CLRLSLDI, it's more convenient to think of it as a mask left bits then rotate left.
	mask_2 := uint64(0xFFFFFFFFFFFFFFFF) >> uint(ssacore.GetPPC64Shiftmb(sld))

	// Rewrite mask to apply after the final left shift.
	mask_3 := (mask_1 & mask_2) << uint(ssacore.GetPPC64Shiftsh(sld))

	r_1 := 32 - srw
	r_2 := ssacore.GetPPC64Shiftsh(sld)
	r_3 := (r_1 + r_2) & 31 // This can wrap.

	if uint64(uint32(mask_3)) != mask_3 || mask_3 == 0 {
		return 0
	}
	return EncodePPC64RotateMask(r_3, int64(mask_3), 32)
}

// Test if a RLWINM feeding into a CLRLSLDI can be merged into RLWINM.  Return
// the encoded RLWINM constant, or 0 if they cannot be merged.
func MergePPC64ClrlsldiRlwinm(sld int32, rlw int64) int64 {
	r_1, _, _, mask_1 := ssacore.DecodePPC64RotateMask(rlw)
	// for CLRLSLDI, it's more convenient to think of it as a mask left bits then rotate left.
	mask_2 := uint64(0xFFFFFFFFFFFFFFFF) >> uint(ssacore.GetPPC64Shiftmb(int64(sld)))

	// combine the masks, and adjust for the final left shift.
	mask_3 := (mask_1 & mask_2) << uint(ssacore.GetPPC64Shiftsh(int64(sld)))
	r_2 := ssacore.GetPPC64Shiftsh(int64(sld))
	r_3 := (r_1 + r_2) & 31 // This can wrap.

	// Verify the result is still a valid bitmask of <= 32 bits.
	if !IsPPC64WordRotateMask(int64(mask_3)) || uint64(uint32(mask_3)) != mask_3 {
		return 0
	}
	return EncodePPC64RotateMask(r_3, int64(mask_3), 32)
}

// Compute the encoded RLWINM constant from combining (SLDconst [sld] (SRWconst [srw] x)),
// or return 0 if they cannot be combined.
func MergePPC64SldiSrw(sld, srw int64) int64 {
	if sld > srw || srw >= 32 {
		return 0
	}
	mask_r := uint32(0xFFFFFFFF) >> uint(srw)
	mask_l := uint32(0xFFFFFFFF) >> uint(sld)
	mask := (mask_r & mask_l) << uint(sld)
	return EncodePPC64RotateMask((32-srw+sld)&31, int64(mask), 32)
}

// encodes the lsb and width for arm(64) bitfield ops into the expected auxInt format.
func ArmBFAuxInt(lsb, width int64) ssacore.Arm64BitField {
	if lsb < 0 || lsb > 63 {
		panic("ARM(64) bit field lsb constant out of range")
	}
	if width < 1 || lsb+width > 64 {
		panic("ARM(64) bit field width constant out of range")
	}
	return ssacore.Arm64BitField(width | lsb<<8)
}

// encodes condition code and NZCV flags into result.
func arm64ConditionalParamsAuxInt(cond ssaop.Op, nzcv uint8) ssacore.Arm64ConditionalParams {
	if cond < ssaop.OpARM64Equal || cond > ssaop.OpARM64GreaterEqualU {
		panic("Wrong conditional operation")
	}
	if nzcv&0x0f != nzcv {
		panic("Wrong value of NZCV flag")
	}
	return ssacore.Arm64ConditionalParams{Cond: cond, NzcvVal: nzcv, ConstVal: 0, Ind: false}
}

// encodes condition code, NZCV flags and constant value into auxint.
func arm64ConditionalParamsAuxIntWithValue(cond ssaop.Op, nzcv uint8, value uint8) ssacore.Arm64ConditionalParams {
	if value&0x1f != value {
		panic("Wrong value of constant")
	}
	params := arm64ConditionalParamsAuxInt(cond, nzcv)
	params.ConstVal = value
	params.Ind = true
	return params
}

// SymIsRO reports whether sym is a read-only global.
func SymIsRO(sym ssacore.Sym) bool {
	lsym := sym.(*obj.LSym)
	return lsym.Type == objabi.SRODATA && len(lsym.R) == 0
}

// Read8 reads one byte from the read-only global sym at offset off.
func Read8(sym ssacore.Sym, off int64) uint8 {
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

// Read16 reads two bytes from the read-only global sym at offset off.
func Read16(sym ssacore.Sym, off int64, byteorder binary.ByteOrder) uint16 {
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

// Read32 reads four bytes from the read-only global sym at offset off.
func Read32(sym ssacore.Sym, off int64, byteorder binary.ByteOrder) uint32 {
	lsym := sym.(*obj.LSym)
	var src []byte
	if 0 <= off && off < int64(len(lsym.P)) {
		src = lsym.P[off:]
	}
	buf := make([]byte, 4)
	copy(buf, src)
	return byteorder.Uint32(buf)
}

// Read64 reads eight bytes from the read-only global sym at offset off.
func Read64(sym ssacore.Sym, off int64, byteorder binary.ByteOrder) uint64 {
	lsym := sym.(*obj.LSym)
	var src []byte
	if 0 <= off && off < int64(len(lsym.P)) {
		src = lsym.P[off:]
	}
	buf := make([]byte, 8)
	copy(buf, src)
	return byteorder.Uint64(buf)
}

type FlagConstantBuilder struct {
	N bool
	Z bool
	C bool
	V bool
}

func (fcs FlagConstantBuilder) Encode() ssacore.FlagConstant {
	var fc ssacore.FlagConstant
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

// AddFlags64 returns the flags that would be set from computing x+y.
func AddFlags64(x, y int64) ssacore.FlagConstant {
	var fcb FlagConstantBuilder
	fcb.Z = x+y == 0
	fcb.N = x+y < 0
	fcb.C = uint64(x+y) < uint64(x)
	fcb.V = x >= 0 && y >= 0 && x+y < 0 || x < 0 && y < 0 && x+y >= 0
	return fcb.Encode()
}

// SubFlags64 returns the flags that would be set from computing x-y.
func SubFlags64(x, y int64) ssacore.FlagConstant {
	var fcb FlagConstantBuilder
	fcb.Z = x-y == 0
	fcb.N = x-y < 0
	fcb.C = uint64(y) <= uint64(x) // This code follows the arm carry flag model.
	fcb.V = x >= 0 && y < 0 && x-y < 0 || x < 0 && y >= 0 && x-y >= 0
	return fcb.Encode()
}

// AddFlags32 returns the flags that would be set from computing x+y.
func AddFlags32(x, y int32) ssacore.FlagConstant {
	var fcb FlagConstantBuilder
	fcb.Z = x+y == 0
	fcb.N = x+y < 0
	fcb.C = uint32(x+y) < uint32(x)
	fcb.V = x >= 0 && y >= 0 && x+y < 0 || x < 0 && y < 0 && x+y >= 0
	return fcb.Encode()
}

// SubFlags32 returns the flags that would be set from computing x-y.
func SubFlags32(x, y int32) ssacore.FlagConstant {
	var fcb FlagConstantBuilder
	fcb.Z = x-y == 0
	fcb.N = x-y < 0
	fcb.C = uint32(y) <= uint32(x) // This code follows the arm carry flag model.
	fcb.V = x >= 0 && y < 0 && x-y < 0 || x < 0 && y >= 0 && x-y >= 0
	return fcb.Encode()
}

// LogicFlags64 returns flags set to the sign/zeroness of x.
// C and V are set to false.
func LogicFlags64(x int64) ssacore.FlagConstant {
	var fcb FlagConstantBuilder
	fcb.Z = x == 0
	fcb.N = x < 0
	return fcb.Encode()
}

// LogicFlags32 returns flags set to the sign/zeroness of x.
// C and V are set to false.
func LogicFlags32(x int32) ssacore.FlagConstant {
	var fcb FlagConstantBuilder
	fcb.Z = x == 0
	fcb.N = x < 0
	return fcb.Encode()
}

func MakeJumpTableSym(b *ssacore.Block) *obj.LSym {
	s := base.Ctxt.Lookup(fmt.Sprintf("%s.jump%d", b.Func.Fe.Func().LSym.Name, b.ID))
	// The jump table symbol is accessed only from the function symbol.
	s.Set(obj.AttrStatic, true)
	return s
}

// SetPos sets the position of v to pos, then returns true.
// Useful for setting the result of a rewrite's position to
// something other than the default.
func SetPos(v *ssacore.Value, pos src.XPos) bool {
	v.Pos = pos
	return true
}

func RewriteStructStore(v *ssacore.Value) *ssacore.Value {
	b := v.Block
	dst := v.Args[0]
	x := v.Args[1]
	if x.Op != ssaop.OpStructMake {
		base.Fatalf("invalid struct store: %v", x)
	}
	mem := v.Args[2]

	t := x.Type
	for i, arg := range x.Args {
		ft := t.FieldType(i)

		addr := b.NewValue1I(v.Pos, ssaop.OpOffPtr, ft.PtrTo(), t.FieldOff(i), dst)
		mem = b.NewValue3A(v.Pos, ssaop.OpStore, types.TypeMem, TypeToAux(ft), addr, arg, mem)
	}

	return mem
}

func AuxToPanicBoundsC(i ssacore.Aux) ssacore.PanicBoundsC {
	return i.(ssacore.PanicBoundsC)
}
func AuxToPanicBoundsCC(i ssacore.Aux) ssacore.PanicBoundsCC {
	return i.(ssacore.PanicBoundsCC)
}
func PanicBoundsCToAux(p ssacore.PanicBoundsC) ssacore.Aux {
	return p
}
func PanicBoundsCCToAux(p ssacore.PanicBoundsCC) ssacore.Aux {
	return p
}

// When v is (IMake typ (StructMake ...)), convert to
// (IMake typ arg) where arg is the pointer-y argument to
// the StructMake (there must be exactly one).
func ImakeOfStructMake(v *ssacore.Value) *ssacore.Value {
	var arg *ssacore.Value
	for _, a := range v.Args[1].Args {
		if a.Type.Size() > 0 {
			arg = a
			break
		}
	}
	return v.Block.NewValue2(v.Pos, ssaop.OpIMake, v.Type, v.Args[0], arg)
}

func ModularMultiplicativeInverse(x uint64) (y uint64) {
	if x%2 != 1 {
		panic("even numbers in a power-of-two modulus do not have a multiplicative inverse")
	}
	// we start with 3 bits of precision because each odd number is its own multiplicative inverse mod 8
	y = x // 3 bits

	// now use the Newton-Raphson method to double the number of correct bits in each iteration.
	y *= 2 - x*y // 6 bits
	y *= 2 - x*y // 12 bits
	y *= 2 - x*y // 24 bits
	y *= 2 - x*y // 48 bits
	y *= 2 - x*y // 96 bits; good enough
	return
}
