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
				if p.Control != a {
					continue
				}
				if e.i == 0 {
					if ct == nil {
						ct = f.ConstBool(f.Entry.Pos, f.Config.Types.Bool, true)
					}
					v.SetArg(i, ct)
				} else {
					if cf == nil {
						cf = f.ConstBool(f.Entry.Pos, f.Config.Types.Bool, false)
					}
					v.SetArg(i, cf)
				}
			}
		}
	}

	// Step 2: Compute which values are live across blocks.
	live := make([]bool, f.NumValues())
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			for _, a := range v.Args {
				if a.Block != v.Block {
					live[a.ID] = true
				}
			}
		}
		if b.Control != nil && b.Control.Block != b {
			live[b.Control.ID] = true
		}
	}

	// Step 3: Redirect control flow around known branches.
	// p:
	//   ... goto b ...
	// b: <- p ...
	//   v = phi(true, ...)
	//   if v goto t else u
	// We can redirect p to go directly to t instead of b.
	// (If v is not live after b).
	for _, b := range f.Blocks {
		if b.Kind != BlockIf {
			continue
		}
		if len(b.Values) != 1 {
			continue
		}
		v := b.Values[0]
		if v.Op != OpPhi {
			continue
		}
		if b.Control != v {
			continue
		}
		if live[v.ID] {
			continue
		}
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
			e2 := b.Succs[1-a.AuxInt]
			t := e2.b
			ti := e2.i

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

			if len(b.Preds) == 1 {
				v.Op = OpCopy
				// No longer a phi, stop optimizing here.
				break
			}
			i--
		}
	}
}
