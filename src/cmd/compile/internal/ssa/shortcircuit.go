// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// Shortcircuit finds situations where branch directions
// are always correlated and rewrites the CFG to take
// advantage of that fact.
// This optimization is useful for compiling && and || expressions.
func shortcircuit(f *Func) {
	// Step 1: Replace a phi arg with a constant if that arg
	// is the control value of a preceding If block.
	// b1:
	//    If a goto b2 else b3
	// b2: <- b1 ...
	//    x = phi(a, ...)
	//
	// We can replace the "a" in the phi with the constant true.
	var ct, cf *Value
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op != OpPhi {
				continue
			}
			if !v.Type.IsBoolean() {
				continue
			}
			for i, a := range v.Args {
				e := b.Preds[i]
				p := e.b
				if p.Kind != BlockIf {
					continue
				}
				if p.Controls[0] != a {
					continue
				}
				if e.i == 0 {
					if ct == nil {
						ct = f.ConstBool(f.Config.Types.Bool, true)
					}
					v.SetArg(i, ct)
				} else {
					if cf == nil {
						cf = f.ConstBool(f.Config.Types.Bool, false)
					}
					v.SetArg(i, cf)
				}
			}
		}
	}

	// Step 2: Redirect control flow around known branches.
	// p:
	//   ... goto b ...
	// b: <- p ...
	//   v = phi(true, ...)
	//   if v goto t else u
	// We can redirect p to go directly to t instead of b.
	// (If v is not live after b).
	for changed := true; changed; {
		changed = false
		for i := len(f.Blocks) - 1; i >= 0; i-- {
			b := f.Blocks[i]
			if fuseBlockPlain(b) {
				changed = true
				continue
			}
			changed = shortcircuitBlock(b) || changed
		}
		if changed {
			f.invalidateCFG()
		}
	}
}

// shortcircuitBlock checks for a CFG of the form
//
//   p   other pred(s)
//    \ /
//     b
//    / \
//   s   other succ
//
// in which b is an If block containing a single phi value with a single use,
// which has a ConstBool arg.
// The only use of the phi value must be the control value of b.
// p is the predecessor determined by the argument slot in which the ConstBool is found.
//
// It rewrites this into
//
//   p   other pred(s)
//   |  /
//   | b
//   |/ \
//   s   other succ
//
// and removes the appropriate phi arg(s).
func shortcircuitBlock(b *Block) bool {
	if b.Kind != BlockIf {
		return false
	}
	// Look for control values of the form Copy(Not(Copy(Phi(const, ...)))).
	// Those must be the only values in the b, and they each must be used only by b.
	// Track the negations so that we can swap successors as needed later.
	ctl := b.Controls[0]
	nval := 1 // the control value
	var swap int64
	for ctl.Uses == 1 && ctl.Block == b && (ctl.Op == OpCopy || ctl.Op == OpNot) {
		if ctl.Op == OpNot {
			swap = 1 ^ swap
		}
		ctl = ctl.Args[0]
		nval++ // wrapper around control value
	}
	if len(b.Values) != nval || ctl.Op != OpPhi || ctl.Block != b || ctl.Uses != 1 {
		return false
	}

	// Locate index of first const phi arg.
	cidx := -1
	for i, a := range ctl.Args {
		if a.Op == OpConstBool {
			cidx = i
			break
		}
	}
	if cidx == -1 {
		return false
	}

	// p is the predecessor corresponding to cidx.
	pe := b.Preds[cidx]
	p := pe.b
	pi := pe.i

	// t is the "taken" branch: the successor we always go to when coming in from p.
	ti := 1 ^ ctl.Args[cidx].AuxInt ^ swap
	te := b.Succs[ti]
	t := te.b
	if p == b || t == b {
		// This is an infinite loop; we can't remove it. See issue 33903.
		return false
	}

	// We're committed. Update CFG and Phis.

	// Remove b's incoming edge from p.
	b.removePred(cidx)
	n := len(b.Preds)
	ctl.Args[cidx].Uses--
	ctl.Args[cidx] = ctl.Args[n]
	ctl.Args[n] = nil
	ctl.Args = ctl.Args[:n]

	// Redirect p's outgoing edge to t.
	p.Succs[pi] = Edge{t, len(t.Preds)}

	// Fix up t to have one more predecessor.
	t.Preds = append(t.Preds, Edge{p, pi})
	for _, v := range t.Values {
		if v.Op != OpPhi {
			continue
		}
		v.AddArg(v.Args[te.i])
	}

	if len(b.Preds) == 0 {
		// Block is now dead.
		b.Kind = BlockInvalid
	}

	phielimValue(ctl)
	return true
}
