// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
)

// LoGo/s M6-S2 type rules for `&expr` over non-addressable operands
// (SPEC T-104.5). Upstream files call in through hook lines only:
// expr.go (the `&` case of unary) and api.go (the LogosAddrFresh info
// field). Every rule below applies only to a package of .lgo files
// (Config.Logos), so plain Go keeps its types exactly as upstream.
//
// An `&` over a composite literal or an addressable operand stays on the
// upstream path (`&T{}` and `&x` keep their form and meaning). Any other
// valid non-nil operand types as a pointer to a fresh variable holding
// the operand value; lowering (logos/addr.go) rewrites exactly the
// recorded operations to `new(expr)`, so each evaluation allocates.

// logosAddrFresh type-checks `&x` for a non-addressable operand x in
// LogosMode. The result is a value of pointer type: untyped operands
// take their default type, mirroring `new(expr)`. It records the
// operation in Info.LogosAddrFresh for lowering and reports true when
// it handled the operation; otherwise it reports false and the caller
// runs the upstream path.
func (check *Checker) logosAddrFresh(x *operand, e *syntax.Operation) bool {
	if !check.conf.Logos {
		return false
	}
	if _, ok := syntax.Unparen(e.X).(*syntax.CompositeLit); ok {
		return false
	}
	if !x.isValid() || x.isNil() || x.mode() == variable {
		return false
	}
	x.mode_ = value
	x.typ_ = &Pointer{base: Default(x.typ())}
	if check.Info != nil {
		if check.Info.LogosAddrFresh == nil {
			check.Info.LogosAddrFresh = make(map[syntax.Expr]bool)
		}
		check.Info.LogosAddrFresh[e] = true
	}
	return true
}
