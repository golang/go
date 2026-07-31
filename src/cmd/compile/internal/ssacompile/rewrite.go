// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"fmt"
	"math"

	"cmd/compile/internal/ssa"
	"cmd/compile/internal/ssa/ssaop"
	"cmd/internal/src"
)

// deadcode indicates whether rewrite should try to remove any values that become dead.
func applyRewrite(f *ssa.Func, rb ssa.BlockRewriter, rv ssa.ValueRewriter, deadcode ssa.DeadValueChoice) {
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
			var b0 *ssa.Block
			if debug > 1 {
				fmt.Printf("%s: start block\n", f.Pass.Name)
				b0 = new(ssa.Block)
				*b0 = *b
				b0.Succs = append([]ssa.Edge{}, b.Succs...) // make a new copy, not aliasing
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
				var v0 *ssa.Value
				if debug > 1 {
					v0 = new(ssa.Value)
					*v0 = *v
					v0.Args = append([]*ssa.Value{}, v.Args...) // make a new copy, not aliasing
				}
				if v.Uses == 0 && v.Removeable() {
					if v.Op != ssaop.OpInvalid && deadcode == ssa.RemoveDeadValues {
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

				vchange := ssa.PhiElimValue(v)
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
			if v.Pos.IsStmt() != src.PosNotStmt && !ssa.NotStmtBoundary(v.Op) {
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

// mergePoint finds a block among a's blocks which dominates b and is itself
// dominated by all of a's blocks. Returns nil if it can't find one.
// Might return nil even if one does exist.
func mergePoint(b *ssa.Block, a ...*ssa.Value) *ssa.Block {
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

// encodes condition code and NZCV flags into result.
func arm64ConditionalParamsAuxInt(cond ssaop.Op, nzcv uint8) ssa.Arm64ConditionalParams {
	if cond < ssaop.OpARM64Equal || cond > ssaop.OpARM64GreaterEqualU {
		panic("Wrong conditional operation")
	}
	if nzcv&0x0f != nzcv {
		panic("Wrong value of NZCV flag")
	}
	return ssa.Arm64ConditionalParams{Cond: cond, NzcvVal: nzcv, ConstVal: 0, Ind: false}
}

// encodes condition code, NZCV flags and constant value into auxint.
func arm64ConditionalParamsAuxIntWithValue(cond ssaop.Op, nzcv uint8, value uint8) ssa.Arm64ConditionalParams {
	if value&0x1f != value {
		panic("Wrong value of constant")
	}
	params := arm64ConditionalParamsAuxInt(cond, nzcv)
	params.ConstVal = value
	params.Ind = true
	return params
}
