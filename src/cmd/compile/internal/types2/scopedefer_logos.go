// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2

import (
	"cmd/compile/internal/syntax"
	. "internal/types/errors"
)

// LoGo/s M6-S10 type rules for `scope defer` (T-104.7). stmt.go calls in
// through one hook line; the lowering lives in logos/scopedefer.go.
//
// A scope defer cleanup follows `defer` call rules: the suspendedCall
// check types the callee and arguments once, so every use is marked and
// genuine call errors surface at the statement.
func (check *Checker) logosScopeDefer(s *syntax.ScopeDeferStmt) {
	if !check.conf.Logos {
		check.error(s, InvalidSyntaxTree, "scope defer requires logos mode")
		return
	}
	if call, ok := syntax.Unparen(s.Call).(*syntax.CallExpr); ok {
		check.suspendedCall("scope defer", call)
		return
	}
	check.use(s.Call)
}
