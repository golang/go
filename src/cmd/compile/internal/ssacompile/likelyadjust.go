// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssacompile

import (
	"cmd/compile/internal/ssa/block"
	"cmd/compile/internal/ssa/ssacore"
	"cmd/compile/internal/ssa/ssaop"
)

const (
	blDEFAULT = 0
	blMin     = blDEFAULT
	blCALL    = 1
	blRET     = 2
	blEXIT    = 3
)

var bllikelies = [4]string{"default", "call", "ret", "exit"}

func describePredictionAgrees(b *ssacore.Block, prediction ssacore.BranchPrediction) string {
	s := ""
	if prediction == b.Likely {
		s = " (agrees with previous)"
	} else if b.Likely != ssacore.BranchUnknown {
		s = " (disagrees with previous, ignored)"
	}
	return s
}

func describeBranchPrediction(f *ssacore.Func, b *ssacore.Block, likely, not int8, prediction ssacore.BranchPrediction) {
	f.Warnl(b.Pos, "Branch prediction rule %s < %s%s",
		bllikelies[likely-blMin], bllikelies[not-blMin], describePredictionAgrees(b, prediction))
}

func likelyadjust(f *ssacore.Func) {
	// The values assigned to certain and local only matter
	// in their rank order.  0 is default, more positive
	// is less likely. It's possible to assign a negative
	// unlikeliness (though not currently the case).
	certain := f.Cache.AllocInt8Slice(f.NumBlocks()) // In the long run, all outcomes are at least this bad. Mainly for Exit
	defer f.Cache.FreeInt8Slice(certain)
	local := f.Cache.AllocInt8Slice(f.NumBlocks()) // for our immediate predecessors.
	defer f.Cache.FreeInt8Slice(local)

	po := f.Postorder()
	nest := f.Loopnest()
	b2l := nest.B2L

	for _, b := range po {
		switch b.Kind {
		case block.BlockExit:
			// Very unlikely.
			local[b.ID] = blEXIT
			certain[b.ID] = blEXIT

			// Ret, it depends.
		case block.BlockRet, block.BlockRetJmp:
			local[b.ID] = blRET
			certain[b.ID] = blRET

			// Calls. TODO not all calls are equal, names give useful clues.
			// Any name-based heuristics are only relative to other calls,
			// and less influential than inferences from loop structure.
		case block.BlockDefer:
			local[b.ID] = blCALL
			certain[b.ID] = max(blCALL, certain[b.Succs[0].B.ID])

		default:
			if len(b.Succs) == 1 {
				certain[b.ID] = certain[b.Succs[0].B.ID]
			} else if len(b.Succs) == 2 {
				// If successor is an unvisited backedge, it's in loop and we don't care.
				// Its default unlikely is also zero which is consistent with favoring loop edges.
				// Notice that this can act like a "reset" on unlikeliness at loops; the
				// default "everything returns" unlikeliness is erased by min with the
				// backedge likeliness; however a loop with calls on every path will be
				// tagged with call cost. Net effect is that loop entry is favored.
				b0 := b.Succs[0].B.ID
				b1 := b.Succs[1].B.ID
				certain[b.ID] = min(certain[b0], certain[b1])

				l := b2l[b.ID]
				l0 := b2l[b0]
				l1 := b2l[b1]

				prediction := b.Likely
				// Weak loop heuristic -- both source and at least one dest are in loops,
				// and there is a difference in the destinations.
				// TODO what is best arrangement for nested loops?
				if l != nil && l0 != l1 {
					noprediction := false
					switch {
					// prefer not to exit loops
					case l1 == nil:
						prediction = ssacore.BranchLikely
					case l0 == nil:
						prediction = ssacore.BranchUnlikely

						// prefer to stay in loop, not exit to outer.
					case l == l0:
						prediction = ssacore.BranchLikely
					case l == l1:
						prediction = ssacore.BranchUnlikely
					default:
						noprediction = true
					}
					if f.Pass.Debug > 0 && !noprediction {
						f.Warnl(b.Pos, "Branch prediction rule stay in loop%s",
							describePredictionAgrees(b, prediction))
					}

				} else {
					// Lacking loop structure, fall back on heuristics.
					if certain[b1] > certain[b0] {
						prediction = ssacore.BranchLikely
						if f.Pass.Debug > 0 {
							describeBranchPrediction(f, b, certain[b0], certain[b1], prediction)
						}
					} else if certain[b0] > certain[b1] {
						prediction = ssacore.BranchUnlikely
						if f.Pass.Debug > 0 {
							describeBranchPrediction(f, b, certain[b1], certain[b0], prediction)
						}
					} else if local[b1] > local[b0] {
						prediction = ssacore.BranchLikely
						if f.Pass.Debug > 0 {
							describeBranchPrediction(f, b, local[b0], local[b1], prediction)
						}
					} else if local[b0] > local[b1] {
						prediction = ssacore.BranchUnlikely
						if f.Pass.Debug > 0 {
							describeBranchPrediction(f, b, local[b1], local[b0], prediction)
						}
					}
				}
				if b.Likely != prediction {
					if b.Likely == ssacore.BranchUnknown {
						b.Likely = prediction
					}
				}
			}
			// Look for calls in the block.  If there is one, make this block unlikely.
			for _, v := range b.Values {
				if ssaop.OpcodeTable[v.Op].Call {
					local[b.ID] = blCALL
					certain[b.ID] = max(blCALL, certain[b.Succs[0].B.ID])
					break
				}
			}
		}
		if f.Pass.Debug > 2 {
			f.Warnl(b.Pos, "BP: Block %s, local=%s, certain=%s", b, bllikelies[local[b.ID]-blMin], bllikelies[certain[b.ID]-blMin])
		}

	}
}
