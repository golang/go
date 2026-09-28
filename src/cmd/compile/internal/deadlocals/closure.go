// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package deadlocals

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/typecheck"
)

// removeUnusedClosureDicts removes dictionary captures that are no longer used
// after inlining and dead local elimination. It must run before escape analysis,
// which decides how variables are captured and whether the closure escapes.
// Waiting until after inlining also preserves the unified IR reader's assumption
// that a generic closure's final capture is its dictionary.
func removeUnusedClosureDicts(fns []*ir.Func) {
	done := make(map[*ir.Func]bool)
	var prune func(*ir.Func)
	prune = func(fn *ir.Func) {
		// Inlined-away closures may already have an LSym solely for inline
		// metadata. Keep their IsClosure classification to avoid enqueueing
		// them again as ordinary functions.
		if fn.OClosure == nil || fn.LSym != nil || len(fn.ClosureVars) == 0 {
			return
		}
		last := len(fn.ClosureVars) - 1
		dict := fn.ClosureVars[last]
		if !dict.IsClosureVar() || dict.Sym().Name != typecheck.LocalDictName {
			return
		}
		if done[fn] {
			return
		}
		done[fn] = true

		// Derived-type variable metadata can need the dictionary for DWARF even
		// when the executable IR does not load any dictionary entries.
		for _, n := range fn.Dcl {
			if n.DictIndex != 0 || n.Canonical().DictIndex != 0 {
				return
			}
		}
		if fn.Inl != nil {
			for _, n := range fn.Inl.Dcl {
				if n.DictIndex != 0 || n.Canonical().DictIndex != 0 {
					return
				}
			}
		}
		for _, n := range fn.ClosureVars {
			if n.DictIndex != 0 || n.Canonical().DictIndex != 0 {
				return
			}
		}

		// Compare canonical names within this function only. A sibling closure
		// that uses the same dictionary must not keep this capture alive.
		dict = dict.Canonical()
		used := false
		var visit func(ir.Node) bool
		visit = func(n ir.Node) bool {
			if n == nil {
				return false
			}
			if name, ok := n.(*ir.Name); ok && name.Canonical() == dict {
				used = true
			}
			if clo, ok := n.(*ir.ClosureExpr); ok && n.Op() == ir.OCLOSURE {
				prune(clo.Func)
				// A retained capture is a use in the enclosing function even if the
				// enclosing function itself never indexes its dictionary.
				for _, cv := range clo.Func.ClosureVars {
					if cv.Outer.Canonical() == dict {
						used = true
					}
				}
			}
			// TypeWord, RType, and similar hidden fields contain real runtime uses.
			ir.DoChildrenWithHidden(n, visit)
			return false
		}
		for _, n := range fn.Body {
			visit(n)
		}
		if !used {
			fn.ClosureVars = fn.ClosureVars[:last]
		}
	}
	for _, fn := range fns {
		prune(fn)
	}
}
