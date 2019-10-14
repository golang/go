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
	v := b.Controls[0]
	nval := 1 // the control value
	swap := false
	for v.Uses == 1 && v.Block == b && (v.Op == OpCopy || v.Op == OpNot) {
		if v.Op == OpNot {
			swap = !swap
		}
		v = v.Args[0]
		nval++ // wrapper around control value
	}
	if len(b.Values) != nval || v.Op != OpPhi || v.Block != b || v.Uses != 1 {
		return false
	}

	// Check for const phi args.
	var changed bool
	for i := 0; i < len(v.Args); i++ {
		a := v.Args[i]
		if a.Op != OpConstBool {
			continue
		}
		// The predecessor we come in from.
		e1 := b.Preds[i]
		p := e1.b
		pi := e1.i

		// The successor we always go to when coming in
		// from that predecessor.
		si := 1 - a.AuxInt
		if swap {
			si = 1 - si
		}
		e2 := b.Succs[si]
		t := e2.b
		if p == b || t == b {
			// This is an infinite loop; we can't remove it. See issue 33903.
			continue
		}
		ti := e2.i

		// Update CFG and Phis.
		changed = true

		// Remove b's incoming edge from p.
		b.removePred(i)
		n := len(b.Preds)
		v.Args[i].Uses--
		v.Args[i] = v.Args[n]
		v.Args[n] = nil
		v.Args = v.Args[:n]

		// Redirect p's outgoing edge to t.
		p.Succs[pi] = Edge{t, len(t.Preds)}

		// Fix up t to have one more predecessor.
		t.Preds = append(t.Preds, Edge{p, pi})
		for _, w := range t.Values {
			if w.Op != OpPhi {
				continue
			}
			w.AddArg(w.Args[ti])
		}
		i--
	}

	if !changed {
		return false
	}

	if len(b.Preds) == 0 {
		// Block is now dead.
		b.Kind = BlockInvalid
		return true
	}

	phielimValue(v)
	return true
}
