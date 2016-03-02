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
	ct := f.ConstBool(f.Entry.Line, f.Config.fe.TypeBool(), true)
	cf := f.ConstBool(f.Entry.Line, f.Config.fe.TypeBool(), false)
	for _, b := range f.Blocks {
		for _, v := range b.Values {
			if v.Op != OpPhi {
				continue
			}
			if !v.Type.IsBoolean() {
				continue
			}
			for i, a := range v.Args {
				p := b.Preds[i]
				if p.Kind != BlockIf {
					continue
				}
				if p.Control != a {
					continue
				}
				if p.Succs[0] == b {
					v.Args[i] = ct
				} else {
					v.Args[i] = cf
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
			p := b.Preds[i]
			// The successor we always go to when coming in
			// from that predecessor.
			t := b.Succs[1-a.AuxInt]

			// Change the edge p->b to p->t.
			for j, x := range p.Succs {
				if x == b {
					p.Succs[j] = t
					break
				}
			}

			// Fix up t to have one more predecessor.
			j := predIdx(t, b)
			t.Preds = append(t.Preds, p)
			for _, w := range t.Values {
				if w.Op != OpPhi {
					continue
				}
				w.Args = append(w.Args, w.Args[j])
			}

			// Fix up b to have one less predecessor.
			n := len(b.Preds) - 1
			b.Preds[i] = b.Preds[n]
			b.Preds[n] = nil
			b.Preds = b.Preds[:n]
			v.Args[i] = v.Args[n]
			v.Args[n] = nil
			v.Args = v.Args[:n]
			if n == 1 {
				v.Op = OpCopy
				// No longer a phi, stop optimizing here.
				break
			}
			i--
		}
	}
}

// predIdx returns the index where p appears in the predecessor list of b.
// p must be in the predecessor list of b.
func predIdx(b, p *Block) int {
	for i, x := range b.Preds {
		if x == p {
			return i
		}
	}
	panic("predecessor not found")
}
