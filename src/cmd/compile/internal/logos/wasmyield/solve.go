// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wasmyield

import "cmd/compile/internal/base"

// solve propagates yielding backwards over the call graph. A function
// yields when it has a yield source of its own or reaches a function that
// yields; the least fixpoint leaves pure recursion pure. A callee that is
// not recorded, or that is duplicate-OK, yields, unless an imported
// package has a fact that says otherwise (calleeYield).
func solve(all []*unit) {
	callers := make(map[*unit][]*unit)
	var work []*unit
	for _, u := range all {
		u.yield = u.hard
		for _, c := range u.calls {
			if base.LogosWasmAssumed(c.Name) {
				continue
			}
			t := byKey[keyOf(c)]
			if t == nil {
				u.yield = u.yield || calleeYield(c)
				continue
			}
			if t.sym.DuplicateOK() {
				u.yield = true
				continue
			}
			callers[t] = append(callers[t], u)
		}
		if u.yield {
			work = append(work, u)
		}
	}
	for len(work) > 0 {
		v := work[len(work)-1]
		work = work[:len(work)-1]
		for _, c := range callers[v] {
			if !c.yield {
				c.yield = true
				work = append(work, c)
			}
		}
	}
}
