// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

// combine copyelim and phielim into a single pass.
// copyelim removes all uses of OpCopy values from f.
// A subsequent deadcode pass is needed to actually remove the copies.
func copyelim(f *Func) {
	phielim(f)

	// loop of copyelimValue(v) process has been done in phielim() pass.
	// Update block control values.
	for _, b := range f.Blocks {
		for i, v := range b.ControlValues() {
			if v.Op == OpCopy {
				b.ReplaceControl(i, v.Args[0])
			}
		}
	}

	// Update named values.
	for _, name := range f.Names {
		values := f.NamedValues[*name]
		for i, v := range values {
			if v.Op == OpCopy {
				values[i] = v.Args[0]
			}
		}
	}
}

// copySource returns the (non-copy) op which is the
// ultimate source of v.  v must be a copy op.
func copySource(v *Value) *Value {
	w := v.Args[0]

	// This loop is just:
	// for w.Op == OpCopy {
	//     w = w.Args[0]
	// }
	// but we take some extra care to make sure we
	// don't get stuck in an infinite loop.
	// Infinite copy loops may happen in unreachable code.
	// (TODO: or can they? Needs a test.)
	slow := w
	var advance bool
	for w.Op == OpCopy {
		w = w.Args[0]
		if w == slow {
			w.reset(OpUnknown)
			break
		}
		if advance {
			slow = slow.Args[0]
		}
		advance = !advance
	}

	// The answer is w.  Update all the copies we saw
	// to point directly to w.  Doing this update makes
	// sure that we don't end up doing O(n^2) work
	// for a chain of n copies.
	for v != w {
		x := v.Args[0]
		v.SetArg(0, w)
		v = x
	}
	return w
}

// copyelimValue ensures that no args of v are copies.
func copyelimValue(v *Value) {
	for i, a := range v.Args {
		if a.Op == OpCopy {
			v.SetArg(i, copySource(a))
		}
	}
}

// phielim eliminates redundant phi values from f.
// A phi is redundant if its arguments are all equal. For
// purposes of counting, ignore the phi itself. Both of
// these phis are redundant:
//
//	v = phi(x,x,x)
//	v = phi(x,v,x,v)
//
// We repeat this process to also catch situations like:
//
//	v = phi(x, phi(x, x), phi(x, v))
//
// TODO: Can we also simplify cases like:
//
//	v = phi(v, w, x)
//	w = phi(v, w, x)
//
// and would that be useful?
func phielim(f *Func) {
	for {
		change := false
		for _, b := range f.Blocks {
			for _, v := range b.Values {
				// This is an early place in SSA where all values are examined.
				// Rewrite all 0-sized Go values to remove accessors, dereferences, loads, etc.
				if t := v.Type; (t.IsStruct() || t.IsArray()) && t.Size() == 0 {
					if t.IsStruct() {
						v.reset(OpStructMake0)
					} else {
						v.reset(OpArrayMake0)
					}
				}
				// Modify all values so no arg (including args
				// of OpCopy) is a copy.
				copyelimValue(v)
				change = phielimValue(v) || change
			}
		}
		if !change {
			break
		}
	}
}

// phielimValue tries to convert the phi v to a copy.
func phielimValue(v *Value) bool {
	if v.Op != OpPhi {
		return false
	}

	// If there are two distinct args of v which
	// are not v itself, then the phi must remain.
	// Otherwise, we can replace it with a copy.
	var w *Value
	for _, x := range v.Args {
		if x == v {
			continue
		}
		if x == w {
			continue
		}
		if w != nil {
			return false
		}
		w = x
	}

	if w == nil {
		// v references only itself. It must be in
		// a dead code loop. Don't bother modifying it.
		return false
	}
	v.Op = OpCopy
	v.SetArgs1(w)
	f := v.Block.Func
	if f.pass.debug > 0 {
		f.Warnl(v.Pos, "eliminated phi")
	}
	return true
}
